# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test runner wrapper for pandas compatibility tests.

Wraps execution of run-pandas-tests.sh within a specific worktree with
GPU isolation via CUDA_VISIBLE_DEVICES. Supports baseline and verification
run modes, flakiness reruns, JSON report-log parsing, and configurable timeouts.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OOMError(RuntimeError):
    """Raised when a test process dies from an out-of-memory condition."""

    def __init__(
        self, message: str = "GPU OOM detected", *, retry_hint: str = ""
    ):
        super().__init__(message)
        self.retry_hint: str = (
            retry_hint or "Reduce parallelism or restart the worker."
        )


class TestRunnerError(RuntimeError):
    """Generic test runner failure."""


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class TestOutcome(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    XFAILED = "xfailed"
    XPASSED = "xpassed"
    ERRORED = "errored"


@dataclass(slots=True)
class TestResult:
    """Result of a single test node execution."""

    node_id: str
    outcome: TestOutcome
    duration: float = 0.0
    longrepr: str = ""
    stdout: str = ""


@dataclass(slots=True)
class SuiteResult:
    """Aggregated result from running a suite of tests."""

    passed: int = 0
    failed: int = 0
    skipped: int = 0
    xfailed: int = 0
    xpassed: int = 0
    errored: int = 0
    total: int = 0
    duration_seconds: float = 0.0
    results: list[TestResult] = field(default_factory=list)
    returncode: int = 0

    @property
    def success(self) -> bool:
        """Suite passes when no failures and no unexpected passes."""
        return self.failed == 0 and self.xpassed == 0 and self.errored == 0


# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

_OOM_PATTERNS = (
    "CUDA out of memory",
    "cudaErrorMemoryAllocation",
    "OutOfMemoryError",
    "CUDA error: out of memory",
    "RuntimeError: CUDA error: the launch timed out",
)

RunMode = Literal["baseline", "verify"]

_DEFAULT_SINGLE_TIMEOUT_MINUTES = 30
_DEFAULT_SUITE_TIMEOUT_MINUTES = 120


def _detect_oom(output: str) -> bool:
    """Check subprocess output for OOM indicators."""
    return any(pattern in output for pattern in _OOM_PATTERNS)


def _build_env(
    gpu_id: int, *, fail_on_fallback: bool = False
) -> dict[str, str]:
    """Build subprocess environment with GPU isolation."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if fail_on_fallback:
        env["CUDF_PANDAS_FAIL_ON_FALLBACK"] = "1"
    # Ensure pytest does not prompt for user input
    env["CI"] = "1"
    env["PANDAS_CI"] = "1"
    return env


def _pyproject_path(worktree_path: Path) -> Path:
    """Return pyproject.toml path inside the pandas-testing directory."""
    return worktree_path / "pandas-testing" / "pandas-tests" / "pyproject.toml"


def _set_xfail_strict(pyproject: Path, strict: bool) -> str:
    """Toggle xfail_strict in pyproject.toml, returning original content."""
    original = pyproject.read_text(encoding="utf-8")
    old_val = "xfail_strict = true" if not strict else "xfail_strict = false"
    new_val = "xfail_strict = true" if strict else "xfail_strict = false"
    modified = original.replace(old_val, new_val)
    # Handle case where the value is already what we want
    if modified == original and new_val not in original:
        # Force-set it
        modified = (
            original.replace("xfail_strict = false", new_val)
            if strict
            else original.replace("xfail_strict = true", new_val)
        )
    pyproject.write_text(modified, encoding="utf-8")
    return original


def _restore_pyproject(pyproject: Path, original_content: str) -> None:
    """Restore pyproject.toml to its original content."""
    pyproject.write_text(original_content, encoding="utf-8")


def _parse_report_log(report_path: Path) -> list[TestResult]:
    """Parse pytest --report-log JSON (one JSON object per line).

    Each line is an independent JSON object. We only care about lines that
    represent test reports (have 'nodeid' and 'outcome' keys).
    """
    results: list[TestResult] = []
    if not report_path.exists():
        return results

    with report_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Only process test call results
            if "nodeid" not in entry or "outcome" not in entry:
                continue

            when = entry.get("when", "")
            outcome_str = entry["outcome"]

            # Determine effective outcome
            if when != "call":
                if outcome_str == "failed":
                    outcome = TestOutcome.ERRORED
                elif outcome_str == "skipped":
                    longrepr = entry.get("longrepr", "")
                    if isinstance(longrepr, list):
                        longrepr = " ".join(str(x) for x in longrepr)
                    elif longrepr is None:
                        longrepr = ""
                    if "Skipped: XPASSes with cudf.pandas enabled." in str(
                        longrepr
                    ):
                        outcome = TestOutcome.XPASSED
                    else:
                        continue
                else:
                    continue
            else:
                if entry.get("wasxfail", False) and outcome_str == "passed":
                    outcome = TestOutcome.XPASSED
                elif entry.get("wasxfail", False) and outcome_str == "failed":
                    outcome = TestOutcome.XFAILED
                elif outcome_str == "passed":
                    outcome = TestOutcome.PASSED
                elif outcome_str == "failed":
                    outcome = TestOutcome.FAILED
                elif outcome_str == "skipped":
                    outcome = TestOutcome.SKIPPED
                else:
                    outcome = TestOutcome.ERRORED

            longrepr_raw = entry.get("longrepr", "")
            if isinstance(longrepr_raw, list):
                longrepr_text = "\n".join(str(x) for x in longrepr_raw)
            elif longrepr_raw is None:
                longrepr_text = ""
            else:
                longrepr_text = str(longrepr_raw)

            results.append(
                TestResult(
                    node_id=entry["nodeid"],
                    outcome=outcome,
                    duration=entry.get("duration", 0.0),
                    longrepr=longrepr_text,
                    stdout=entry.get("sections", [{}])[0]
                    if entry.get("sections")
                    else "",
                )
            )

    return results


def _aggregate_results(
    results: list[TestResult], returncode: int, duration: float
) -> SuiteResult:
    """Aggregate individual test results into a SuiteResult."""
    suite = SuiteResult(
        results=results,
        returncode=returncode,
        duration_seconds=duration,
        total=len(results),
    )
    for r in results:
        match r.outcome:
            case TestOutcome.PASSED:
                suite.passed += 1
            case TestOutcome.FAILED:
                suite.failed += 1
            case TestOutcome.SKIPPED:
                suite.skipped += 1
            case TestOutcome.XFAILED:
                suite.xfailed += 1
            case TestOutcome.XPASSED:
                suite.xpassed += 1
            case TestOutcome.ERRORED:
                suite.errored += 1
    return suite


def _run_subprocess(
    cmd: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with timeout and OOM detection."""
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            # Use process group so we can kill child tree on timeout
            preexec_fn=os.setsid,
        )
    except subprocess.TimeoutExpired as exc:
        # Kill the entire process group
        try:
            os.killpg(
                os.getpgid(exc.args[0] if isinstance(exc.args[0], int) else 0),
                signal.SIGKILL,
            )
        except (ProcessLookupError, OSError):
            pass
        raise TestRunnerError(
            f"Test execution timed out after {timeout_seconds}s"
        ) from exc

    combined_output = (result.stdout or "") + (result.stderr or "")
    if _detect_oom(combined_output):
        raise OOMError(
            f"OOM detected during test execution (returncode={result.returncode})",
            retry_hint="Reduce --numprocesses or use a GPU with more memory.",
        )

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_test(
    worktree_path: str | Path,
    node_id: str,
    gpu_id: int,
    *,
    run_mode: RunMode = "verify",
    fail_on_fallback: bool | None = None,
    timeout_minutes: int = _DEFAULT_SINGLE_TIMEOUT_MINUTES,
    flakiness_reruns: int = 0,
) -> TestResult:
    """Run a single test node within a worktree with GPU isolation.

    Parameters
    ----------
    worktree_path : str | Path
        Path to the git worktree root.
    node_id : str
        Pytest node ID (e.g., "tests/groupby/test_groupby.py::test_func[param]").
    gpu_id : int
        GPU index for CUDA_VISIBLE_DEVICES.
    run_mode : RunMode
        "verify" sets xfail_strict=true and fail_on_fallback=True by default.
        "baseline" is permissive.
    fail_on_fallback : bool | None
        Override fail-on-fallback. If None, derived from run_mode.
    timeout_minutes : int
        Timeout in minutes for the test execution.
    flakiness_reruns : int
        If > 0, run the test N times; all must pass.

    Returns
    -------
    TestResult
        Result of the test execution.

    Raises
    ------
    OOMError
        If OOM is detected in subprocess output.
    TestRunnerError
        On timeout or other execution failures.
    """
    worktree_path = Path(worktree_path)
    is_verify = run_mode == "verify"

    # Derive fail_on_fallback from mode if not explicitly set
    if fail_on_fallback is None:
        fail_on_fallback = is_verify

    env = _build_env(gpu_id, fail_on_fallback=fail_on_fallback)
    timeout_s = timeout_minutes * 60

    # Set xfail_strict for verify mode
    pyproject = _pyproject_path(worktree_path)
    original_content: str | None = None
    if is_verify and pyproject.exists():
        original_content = _set_xfail_strict(pyproject, strict=True)

    try:
        total_runs = max(1, flakiness_reruns) if flakiness_reruns > 0 else 1

        last_result: TestResult | None = None
        for run_idx in range(total_runs):
            with tempfile.NamedTemporaryFile(
                suffix=".json",
                prefix="report_",
                dir=str(worktree_path),
                delete=False,
            ) as tmp:
                report_path = Path(tmp.name)

            try:
                test_script = (
                    worktree_path
                    / "python"
                    / "cudf"
                    / "cudf"
                    / "pandas"
                    / "scripts"
                    / "run-pandas-tests.sh"
                )
                cmd = [
                    "bash",
                    str(test_script),
                    f"--report-log={report_path.name}",
                    node_id,
                    "-xvs",
                ]

                proc = _run_subprocess(
                    cmd, env=env, cwd=worktree_path, timeout_seconds=timeout_s
                )

                # Parse results from report log
                # The script cds into pandas-testing, then pandas-tests, then
                # moves *.json back to pandas-testing. So report could be in
                # pandas-testing/<name>.json
                actual_report = (
                    worktree_path / "pandas-testing" / report_path.name
                )
                if not actual_report.exists():
                    actual_report = report_path

                results = _parse_report_log(actual_report)

                # Clean up temp report
                for p in (report_path, actual_report):
                    if p.exists():
                        p.unlink()

                # Find the specific test result
                matching = [r for r in results if node_id in r.node_id]
                if matching:
                    last_result = matching[0]
                else:
                    # Infer from return code
                    outcome = (
                        TestOutcome.PASSED
                        if proc.returncode == 0
                        else TestOutcome.FAILED
                    )
                    output = ((proc.stdout or "") + (proc.stderr or ""))[
                        -4000:
                    ]
                    last_result = TestResult(
                        node_id=node_id,
                        outcome=outcome,
                        longrepr=output,
                    )

                # For flakiness reruns, all must pass. Preserve the underlying
                # pytest details so callers can diagnose real verification
                # failures instead of seeing only the synthetic rerun marker.
                if (
                    flakiness_reruns > 0
                    and last_result.outcome != TestOutcome.PASSED
                ):
                    details = (
                        last_result.longrepr or last_result.stdout or ""
                    ).strip()
                    if not details and proc.stderr:
                        details = proc.stderr.strip()
                    if not details and proc.stdout:
                        details = proc.stdout.strip()
                    message = (
                        f"Flakiness rerun {run_idx + 1}/{total_runs} failed."
                    )
                    if details:
                        message += "\n" + details[-4000:]
                    last_result = TestResult(
                        node_id=node_id,
                        outcome=TestOutcome.FAILED,
                        duration=last_result.duration,
                        longrepr=message,
                        stdout=last_result.stdout,
                    )
                    return last_result

            except Exception:
                # Clean up report on error
                if report_path.exists():
                    report_path.unlink()
                raise

        assert last_result is not None
        return last_result

    finally:
        # Always restore xfail_strict
        if original_content is not None and pyproject.exists():
            _restore_pyproject(pyproject, original_content)


def run_test_family(
    worktree_path: str | Path,
    base_test_name: str,
    gpu_id: int,
    *,
    run_mode: RunMode = "verify",
    fail_on_fallback: bool | None = None,
    timeout_minutes: int = _DEFAULT_SINGLE_TIMEOUT_MINUTES,
    parallelism: int = 4,
) -> list[TestResult]:
    """Run all parametrizations of a base test name.

    Parameters
    ----------
    worktree_path : str | Path
        Path to the git worktree root.
    base_test_name : str
        Base test path without parametrization
        (e.g., "tests/groupby/test_groupby.py::test_func").
    gpu_id : int
        GPU index for CUDA_VISIBLE_DEVICES.
    run_mode : RunMode
        "verify" or "baseline".
    fail_on_fallback : bool | None
        Override fail-on-fallback setting.
    timeout_minutes : int
        Timeout per test family run.
    parallelism : int
        Number of parallel pytest workers (-n).

    Returns
    -------
    list[TestResult]
        Results for each parametrization found.

    Raises
    ------
    OOMError
        If OOM is detected.
    TestRunnerError
        On timeout or other failures.
    """
    worktree_path = Path(worktree_path)
    is_verify = run_mode == "verify"

    if fail_on_fallback is None:
        fail_on_fallback = is_verify

    env = _build_env(gpu_id, fail_on_fallback=fail_on_fallback)
    timeout_s = timeout_minutes * 60

    pyproject = _pyproject_path(worktree_path)
    original_content: str | None = None
    if is_verify and pyproject.exists():
        original_content = _set_xfail_strict(pyproject, strict=True)

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".json",
            prefix="family_report_",
            dir=str(worktree_path),
            delete=False,
        ) as tmp:
            report_path = Path(tmp.name)

        try:
            # Use -k filter to match all parametrizations of base test
            # The base_test_name is a pytest node path like tests/x.py::TestClass::test_name
            test_script = (
                worktree_path
                / "python"
                / "cudf"
                / "cudf"
                / "pandas"
                / "scripts"
                / "run-pandas-tests.sh"
            )

            cmd = [
                "bash",
                str(test_script),
                f"--report-log={report_path.name}",
                "--tb=line",
                "-q",
            ]
            if parallelism > 1:
                cmd.extend(["-n", str(parallelism)])

            # Pass the base test path (file::class::function without params)
            cmd.append(base_test_name)

            proc = _run_subprocess(
                cmd, env=env, cwd=worktree_path, timeout_seconds=timeout_s
            )

            # Report location
            actual_report = worktree_path / "pandas-testing" / report_path.name
            if not actual_report.exists():
                actual_report = report_path

            results = _parse_report_log(actual_report)

            # Clean up
            for p in (report_path, actual_report):
                if p.exists():
                    p.unlink()

            return (
                results
                if results
                else [
                    TestResult(
                        node_id=base_test_name,
                        outcome=TestOutcome.PASSED
                        if proc.returncode == 0
                        else TestOutcome.FAILED,
                        longrepr=proc.stdout[-2000:] if proc.stdout else "",
                    )
                ]
            )
        except Exception:
            if report_path.exists():
                report_path.unlink()
            raise

    finally:
        if original_content is not None and pyproject.exists():
            _restore_pyproject(pyproject, original_content)


def run_full_suite(
    worktree_path: str | Path,
    gpu_id: int,
    *,
    parallelism: int = 16,
    run_mode: RunMode = "baseline",
    fail_on_fallback: bool | None = None,
    timeout_minutes: int = _DEFAULT_SUITE_TIMEOUT_MINUTES,
    extra_pytest_args: list[str] | None = None,
) -> SuiteResult:
    """Run the full pandas test suite within a worktree.

    Parameters
    ----------
    worktree_path : str | Path
        Path to the git worktree root.
    gpu_id : int
        GPU index for CUDA_VISIBLE_DEVICES.
    parallelism : int
        Number of parallel pytest workers.
    run_mode : RunMode
        "verify" or "baseline".
    fail_on_fallback : bool | None
        Override fail-on-fallback setting.
    timeout_minutes : int
        Timeout in minutes for the full suite.
    extra_pytest_args : list[str] | None
        Additional pytest arguments.

    Returns
    -------
    SuiteResult
        Aggregated test results.

    Raises
    ------
    OOMError
        If OOM is detected.
    TestRunnerError
        On timeout.
    """
    worktree_path = Path(worktree_path)
    is_verify = run_mode == "verify"

    if fail_on_fallback is None:
        fail_on_fallback = is_verify

    env = _build_env(gpu_id, fail_on_fallback=fail_on_fallback)
    timeout_s = timeout_minutes * 60

    pyproject = _pyproject_path(worktree_path)
    original_content: str | None = None
    if is_verify and pyproject.exists():
        original_content = _set_xfail_strict(pyproject, strict=True)

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".json",
            prefix="suite_report_",
            dir=str(worktree_path),
            delete=False,
        ) as tmp:
            report_path = Path(tmp.name)

        try:
            import time

            start_time = time.monotonic()

            test_script = (
                worktree_path
                / "python"
                / "cudf"
                / "cudf"
                / "pandas"
                / "scripts"
                / "run-pandas-tests.sh"
            )

            cmd = [
                "bash",
                str(test_script),
                f"--report-log={report_path.name}",
                "--tb=line",
                "-q",
                f"--numprocesses={parallelism}",
                "--dist=worksteal",
                "--max-worker-restart=3",
            ]

            if extra_pytest_args:
                cmd.extend(extra_pytest_args)

            proc = _run_subprocess(
                cmd, env=env, cwd=worktree_path, timeout_seconds=timeout_s
            )

            elapsed = time.monotonic() - start_time

            # Report location
            actual_report = worktree_path / "pandas-testing" / report_path.name
            if not actual_report.exists():
                actual_report = report_path

            results = _parse_report_log(actual_report)

            # Clean up
            for p in (report_path, actual_report):
                if p.exists():
                    p.unlink()

            return _aggregate_results(results, proc.returncode, elapsed)

        except Exception:
            if report_path.exists():
                report_path.unlink()
            raise

    finally:
        if original_content is not None and pyproject.exists():
            _restore_pyproject(pyproject, original_content)
