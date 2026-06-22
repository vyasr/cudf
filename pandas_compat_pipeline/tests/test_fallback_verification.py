# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for FixerAgent._verify_fix() — documents Bug 2.

Bug 2: when fail_on_fallback=True, conftest.py's import of pytz via
import_optional_dependency triggers a cudf.pandas fallback and crashes
BEFORE the target test runs.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path[:0] = [
    _path
    for _path in (
        str(Path(__file__).resolve().parents[1]),
        str(Path(__file__).resolve().parents[1].parent),
    )
    if _path not in sys.path
]

from pandas_compat_pipeline.src.agents.fixer import FixerAgent  # noqa: E402
from pandas_compat_pipeline.src.agents.llm_client import (  # noqa: E402
    LLMClient,
)
from pandas_compat_pipeline.src.utils.models import TestGroup  # noqa: E402
from pandas_compat_pipeline.src.utils.test_runner import (  # noqa: E402
    TestOutcome,
    TestResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent() -> FixerAgent:
    """Create a FixerAgent with a mocked LLM client."""
    client = MagicMock(spec=LLMClient)
    agent = FixerAgent(llm_client=client)
    return agent


def _make_test_group(
    node_id: str = "tests/io/test_sql.py::test_read_sql[param1]",
) -> TestGroup:
    return TestGroup(
        base_name="test_read_sql",
        file_path="tests/io/test_sql.py",
        class_name=None,
        node_ids=[node_id],
    )


def _passed_result(
    node_id: str = "tests/io/test_sql.py::test_read_sql[param1]",
) -> TestResult:
    return TestResult(
        node_id=node_id,
        outcome=TestOutcome.PASSED,
        duration=0.1,
        longrepr="",
        stdout="",
    )


def _is_conftest_fallback_failure(result: TestResult) -> bool:
    """Detect if a TestResult failure was caused by conftest pytz fallback.

    This helper documents what the fix SHOULD check: the longrepr contains
    evidence that conftest.py triggered a NotImplementedFallbackError via
    import_optional_dependency before the actual test body ran.
    """
    if result.outcome not in (TestOutcome.FAILED, TestOutcome.ERRORED):
        return False
    longrepr = result.longrepr
    return (
        "conftest.py" in longrepr
        and "import_optional_dependency" in longrepr
        and "NotImplementedFallbackError" in longrepr
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_verify_fix_runs_standard_mode() -> None:
    """_verify_fix runs both standard (fail_on_fallback=False) and strict
    (fail_on_fallback=True) modes when _diag_baseline_passed is False.
    """
    agent = _make_agent()
    agent._diag_baseline_passed = False

    agent._run_test_async = AsyncMock(return_value=_passed_result())

    group = _make_test_group()
    results = asyncio.run(
        agent._verify_fix(worktree_path="/tmp/wt", test_group=group, gpu_id=0)
    )

    # Collect all calls and their fail_on_fallback values
    calls = agent._run_test_async.call_args_list
    fof_values = [
        call.kwargs.get(
            "fail_on_fallback", call.args[4] if len(call.args) > 4 else None
        )
        for call in calls
    ]

    # Both modes should be exercised
    assert False in fof_values, (
        "Expected at least one call with fail_on_fallback=False"
    )
    assert True in fof_values, (
        "Expected at least one call with fail_on_fallback=True"
    )
    assert len(results) == 2


def test_verify_fix_skips_fallback_when_baseline_passed() -> None:
    """When _diag_baseline_passed is True (stale-xfail), _verify_fix only runs
    standard mode and skips the fail_on_fallback=True verification.
    """
    agent = _make_agent()
    agent._diag_baseline_passed = True

    agent._run_test_async = AsyncMock(return_value=_passed_result())

    group = _make_test_group()
    results = asyncio.run(
        agent._verify_fix(worktree_path="/tmp/wt", test_group=group, gpu_id=0)
    )

    calls = agent._run_test_async.call_args_list
    fof_values = [
        call.kwargs.get(
            "fail_on_fallback", call.args[4] if len(call.args) > 4 else None
        )
        for call in calls
    ]

    # Only standard mode should run
    assert False in fof_values, "Expected call with fail_on_fallback=False"
    assert True not in fof_values, (
        "Should NOT call with fail_on_fallback=True for stale-xfail"
    )
    assert len(results) == 1


def test_conftest_fallback_pattern_in_longrepr() -> None:
    """Documents Bug 2: conftest.py pytz import triggers NotImplementedFallbackError.

    When fail_on_fallback=True is active, conftest.py:95 imports pytz via
    import_optional_dependency('pytz', errors='ignore') which triggers
    NotImplementedFallbackError BEFORE any target test runs. This test
    documents the longrepr pattern and the helper that should detect it.
    """
    # Simulate the actual error longrepr observed in the wild
    longrepr = (
        "conftest.py:95: in <module>\n"
        "    import_optional_dependency('pytz', errors='ignore')\n"
        "E   cudf.pandas.fast_slow_proxy.NotImplementedFallbackError: pytz"
    )

    result = TestResult(
        node_id="tests/io/test_sql.py::test_read_sql[param1]",
        outcome=TestOutcome.FAILED,
        duration=0.0,
        longrepr=longrepr,
        stdout="",
    )

    # Verify the pattern matches what we expect
    assert "conftest.py" in result.longrepr
    assert "import_optional_dependency" in result.longrepr
    assert "NotImplementedFallbackError" in result.longrepr

    # The helper correctly identifies this as a conftest fallback crash
    assert _is_conftest_fallback_failure(result) is True

    # A PASSED result should NOT be flagged
    passed = _passed_result()
    assert _is_conftest_fallback_failure(passed) is False


def test_verify_fix_filters_conftest_fallback_result() -> None:
    """_verify_fix filters out conftest fallback failures from results.

    When fail_on_fallback=True triggers a conftest import error (not a real
    target test failure), the result should be discarded so it doesn't cause
    the fix to be rejected.
    """
    agent = _make_agent()
    agent._diag_baseline_passed = False

    # Standard mode: passes fine.
    # Fallback mode: conftest crashes before test runs.
    conftest_crash_longrepr = (
        "conftest.py:95: in <module>\n"
        "    import_optional_dependency('pytz', errors='ignore')\n"
        "E   cudf.pandas.fast_slow_proxy.NotImplementedFallbackError: pytz"
    )
    conftest_fail = TestResult(
        node_id="tests/io/test_sql.py::test_read_sql[param1]",
        outcome=TestOutcome.FAILED,
        duration=0.0,
        longrepr=conftest_crash_longrepr,
        stdout="",
    )

    agent._run_test_async = AsyncMock(
        side_effect=[_passed_result(), conftest_fail]
    )

    group = _make_test_group()
    results = asyncio.run(
        agent._verify_fix(worktree_path="/tmp/wt", test_group=group, gpu_id=0)
    )

    # After fix: only the standard result should be in results.
    # The conftest fallback failure should be filtered out.
    assert len(results) == 1
    assert results[0].outcome == TestOutcome.PASSED


def test_verify_fix_returns_all_results() -> None:
    """_verify_fix returns results from both standard and fallback modes."""
    agent = _make_agent()
    agent._diag_baseline_passed = False

    agent._run_test_async = AsyncMock(return_value=_passed_result())

    group = _make_test_group()
    results = asyncio.run(
        agent._verify_fix(worktree_path="/tmp/wt", test_group=group, gpu_id=0)
    )

    # With 1 node_id and _diag_baseline_passed=False, we get 2 results:
    # one from standard mode, one from fallback mode
    assert len(results) == 2
    assert all(r.outcome == TestOutcome.PASSED for r in results)
