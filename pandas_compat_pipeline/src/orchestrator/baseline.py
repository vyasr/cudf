# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Baseline runner node for the pandas compatibility pipeline.

Runs the full pandas test suite before any fixes begin, establishing a
baseline snapshot of test results. Detects stale xfail entries (tests that
now pass or are no longer collected) and new failures (tests that fail but
are not in the xfail list).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import PipelineConfig, load_config
from ..utils.test_runner import (
    OOMError,
    SuiteResult,
    TestOutcome,
    run_full_suite,
)
from ..utils.xfail_parser import parse_xfail_list

logger = logging.getLogger(__name__)

_BASELINE_RESULTS_FILENAME = "baseline_results.json"
_PIPELINE_DIR = Path(__file__).resolve().parents[3] / "pandas_compat_pipeline"


@dataclass(slots=True)
class BaselineResult:
    """Aggregated baseline test suite results."""

    passed: int = 0
    failed: int = 0
    xfailed: int = 0
    xpassed: int = 0
    errors: int = 0
    unexpected_failures: list[str] = field(default_factory=list)
    unexpected_passes: list[str] = field(default_factory=list)
    stale_entries: list[str] = field(default_factory=list)
    new_failures: list[str] = field(default_factory=list)
    total_collected: int = 0


def _collect_known_xfail_node_ids(plugin_path: str | None = None) -> set[str]:
    """Get the set of all node IDs listed in NODEIDS_THAT_FAIL."""
    groups = parse_xfail_list(plugin_path)
    all_ids: set[str] = set()
    for group in groups:
        all_ids.update(group.node_ids)
    return all_ids


def _build_baseline_result(
    suite: SuiteResult, known_xfails: set[str]
) -> BaselineResult:
    """Analyze suite results against the known xfail list."""
    collected_node_ids: set[str] = set()

    unexpected_failures: list[str] = []
    unexpected_passes: list[str] = []

    for result in suite.results:
        collected_node_ids.add(result.node_id)

        if result.outcome == TestOutcome.FAILED:
            if result.node_id not in known_xfails:
                unexpected_failures.append(result.node_id)
        elif result.outcome == TestOutcome.XPASSED:
            unexpected_passes.append(result.node_id)

    # Stale entries: xfail node IDs that either xpassed (now passing) or
    # were never collected (test renamed/removed/no longer parametrized)
    stale_entries: list[str] = []
    for node_id in sorted(known_xfails):
        if node_id in unexpected_passes:
            stale_entries.append(node_id)
        elif node_id not in collected_node_ids:
            stale_entries.append(node_id)

    # New failures: tests that FAILED but are NOT in the xfail list
    new_failures = sorted(unexpected_failures)

    return BaselineResult(
        passed=suite.passed,
        failed=suite.failed,
        xfailed=suite.xfailed,
        xpassed=suite.xpassed,
        errors=suite.errored,
        unexpected_failures=unexpected_failures,
        unexpected_passes=sorted(unexpected_passes),
        stale_entries=stale_entries,
        new_failures=new_failures,
        total_collected=suite.total,
    )


def _save_results(result: BaselineResult, output_dir: Path) -> None:
    """Persist baseline results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / _BASELINE_RESULTS_FILENAME
    _ = output_path.write_text(
        json.dumps(asdict(result), indent=2, default=str), encoding="utf-8"
    )
    logger.info("Baseline results saved to %s", output_path)


async def run_baseline(
    worktree_path: str | Path,
    gpu_id: int,
    scope: str | None = None,
    config: PipelineConfig | None = None,
) -> BaselineResult:
    """Run the baseline test suite and analyze results.

    Parameters
    ----------
    worktree_path : str | Path
        Path to the git worktree root containing the cudf repository.
    gpu_id : int
        GPU index for CUDA_VISIBLE_DEVICES.
    scope : str | None
        Optional subset scope (e.g. "tests/apply/") to run only matching tests.
    config : PipelineConfig | None
        Pipeline configuration. Loaded from default if None.

    Returns
    -------
    BaselineResult
        Aggregated and analyzed baseline results.

    Raises
    ------
    OOMError
        If OOM is detected; caller should retry with reduced parallelism.
    """
    if config is None:
        config = load_config()

    worktree_path = Path(worktree_path)
    parallelism = config.pytest_parallelism

    extra_args: list[str] | None = None
    if scope:
        extra_args = [scope]

    logger.info(
        "Starting baseline run: worktree=%s, gpu=%d, parallelism=%d, scope=%s",
        worktree_path,
        gpu_id,
        parallelism,
        scope or "full",
    )

    try:
        suite = run_full_suite(
            worktree_path,
            gpu_id,
            parallelism=parallelism,
            run_mode="baseline",
            timeout_minutes=config.integration_timeout_minutes,
            extra_pytest_args=extra_args,
        )
    except OOMError:
        logger.warning(
            "OOM detected, retrying with reduced parallelism (%d)",
            config.pytest_parallelism_fallback,
        )
        suite = run_full_suite(
            worktree_path,
            gpu_id,
            parallelism=config.pytest_parallelism_fallback,
            run_mode="baseline",
            timeout_minutes=config.integration_timeout_minutes,
            extra_pytest_args=extra_args,
        )

    plugin_path = str(worktree_path / config.plugin_path)
    known_xfails = _collect_known_xfail_node_ids(plugin_path)

    result = _build_baseline_result(suite, known_xfails)

    logger.info(
        "Baseline complete: total=%d, passed=%d, failed=%d, xfailed=%d, "
        + "xpassed=%d, errors=%d, new_failures=%d, stale_entries=%d",
        result.total_collected,
        result.passed,
        result.failed,
        result.xfailed,
        result.xpassed,
        result.errors,
        len(result.new_failures),
        len(result.stale_entries),
    )

    _save_results(result, _PIPELINE_DIR)

    return result
