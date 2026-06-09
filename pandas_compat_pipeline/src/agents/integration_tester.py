# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# pyright: reportExplicitAny=false, reportAny=false
import asyncio
import inspect
import logging
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..config import PipelineConfig, load_config
from ..utils.branch_manager import MergeResult, merge_fix_branches
from ..utils.test_runner import (
    OOMError,
    SuiteResult,
    TestOutcome,
    run_full_suite,
)

LOGGER = logging.getLogger(__name__)
BRANCH_PREFIX = "pandas-compat/"
DEFAULT_INTEGRATION_GPU = 7
DEFAULT_INTEGRATION_PARALLELISM = 16
DEFAULT_INTEGRATION_PARALLELISM_FALLBACK = 12
REGRESSION_OUTCOMES = {TestOutcome.FAILED, TestOutcome.ERRORED}


@dataclass(slots=True)
class IntegrationResult:
    passed: bool
    regressions: list[str]
    batch_branches: list[str]
    integration_branch: str
    batch_number: int


@dataclass(slots=True)
class _BaselineSets:
    known_xfails: set[str] = field(default_factory=set)
    currently_passing: set[str] = field(default_factory=set)


class IntegrationTesterAgent:
    """Merge completed fix branches and run a full-suite regression check."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        regression_debugger: Any | None = None,
    ) -> None:
        self.config: PipelineConfig = config or load_config()
        self.integration_trigger_every_n: int = (
            self.config.integration_trigger_every_n
        )
        self.integration_gpu: int = (
            self.config.integration_gpu
            if self.config.integration_gpu is not None
            else DEFAULT_INTEGRATION_GPU
        )
        self.integration_verified: set[str] = set()
        self._regression_debugger: Any | None = regression_debugger

    def should_trigger(
        self,
        fixes_since_last_integration: int,
        integration_queue: list[str] | None = None,
    ) -> bool:
        """Return True once the integration queue reaches the configured cadence."""

        if fixes_since_last_integration < self.integration_trigger_every_n:
            return False
        return integration_queue is None or len(integration_queue) > 0

    def select_batch(self, integration_queue: list[str]) -> list[str]:
        """Select the next integration batch from queued completed fix branches."""

        return list(integration_queue[: self.integration_trigger_every_n])

    async def test(
        self,
        batch_branches: list[str],
        worktree_path: str,
        gpu_id: int,
        batch_number: int,
        baseline_result: dict[str, Any],
    ) -> IntegrationResult:
        """Create an integration branch, merge fixes, and detect regressions."""

        batch = list(batch_branches)
        integration_branch = f"{BRANCH_PREFIX}integration-{batch_number}"
        if not batch:
            return IntegrationResult(
                passed=True,
                regressions=[],
                batch_branches=[],
                integration_branch=integration_branch,
                batch_number=batch_number,
            )

        gpu_to_use = gpu_id if gpu_id >= 0 else self.integration_gpu

        try:
            await asyncio.to_thread(
                self._create_integration_branch,
                worktree_path,
                integration_branch,
            )
            merge_result = await asyncio.to_thread(
                merge_fix_branches, batch, integration_branch, worktree_path
            )
            if not merge_result.success:
                regressions = self._merge_conflict_regressions(merge_result)
                await self._notify_regression_debugger(
                    regressions=regressions,
                    batch_branches=batch,
                    integration_branch=integration_branch,
                    worktree_path=worktree_path,
                )
                return IntegrationResult(
                    passed=False,
                    regressions=regressions,
                    batch_branches=batch,
                    integration_branch=integration_branch,
                    batch_number=batch_number,
                )

            suite_result = await self._run_full_suite_with_oom_fallback(
                worktree_path, gpu_to_use
            )
            regressions = self._detect_regressions(
                suite_result, baseline_result
            )

            if regressions:
                await self._notify_regression_debugger(
                    regressions=regressions,
                    batch_branches=batch,
                    integration_branch=integration_branch,
                    worktree_path=worktree_path,
                )
            else:
                self.integration_verified.update(batch)

            return IntegrationResult(
                passed=not regressions,
                regressions=regressions,
                batch_branches=batch,
                integration_branch=integration_branch,
                batch_number=batch_number,
            )
        except (
            Exception
        ) as exc:  # pragma: no cover - integration failure payload path
            LOGGER.exception("Integration test batch %s failed", batch_number)
            regressions = [f"integration_error:{exc}"]
            await self._notify_regression_debugger(
                regressions=regressions,
                batch_branches=batch,
                integration_branch=integration_branch,
                worktree_path=worktree_path,
            )
            return IntegrationResult(
                passed=False,
                regressions=regressions,
                batch_branches=batch,
                integration_branch=integration_branch,
                batch_number=batch_number,
            )

    def _create_integration_branch(
        self, worktree_path: str, integration_branch: str
    ) -> None:
        _ = _run_git(["checkout", "-B", integration_branch], worktree_path)

    async def _run_full_suite_with_oom_fallback(
        self, worktree_path: str, gpu_id: int
    ) -> SuiteResult:
        try:
            return await asyncio.to_thread(
                run_full_suite,
                worktree_path,
                gpu_id,
                parallelism=self.config.pytest_parallelism
                or DEFAULT_INTEGRATION_PARALLELISM,
                run_mode="verify",
                timeout_minutes=self.config.integration_timeout_minutes,
            )
        except OOMError:
            LOGGER.warning(
                "Full suite OOM at -n%s; retrying at -n%s",
                self.config.pytest_parallelism,
                self.config.pytest_parallelism_fallback,
            )
            return await asyncio.to_thread(
                run_full_suite,
                worktree_path,
                gpu_id,
                parallelism=self.config.pytest_parallelism_fallback
                or DEFAULT_INTEGRATION_PARALLELISM_FALLBACK,
                run_mode="verify",
                timeout_minutes=self.config.integration_timeout_minutes,
            )

    def _detect_regressions(
        self, suite_result: SuiteResult, baseline_result: dict[str, Any]
    ) -> list[str]:
        baseline = self._baseline_sets(baseline_result)
        regressions: list[str] = []

        for node_id in self._failing_node_ids(suite_result):
            previously_passing = node_id in baseline.currently_passing
            not_originally_xfailed = node_id not in baseline.known_xfails
            if previously_passing or not_originally_xfailed:
                regressions.append(node_id)

        return sorted(set(regressions))

    def _baseline_sets(self, baseline_result: dict[str, Any]) -> _BaselineSets:
        return _BaselineSets(
            known_xfails=self._extract_known_xfails(baseline_result),
            currently_passing=self._extract_currently_passing(baseline_result),
        )

    def _extract_known_xfails(
        self, baseline_result: dict[str, Any]
    ) -> set[str]:
        candidates = (
            "known_xfails",
            "known_xfail_list",
            "xfail_node_ids",
            "xfails",
            "original_xfail_list",
            "original_xfails",
            "nodeids_that_fail",
            "NODEIDS_THAT_FAIL",
        )
        for key in candidates:
            if key in baseline_result:
                return _node_ids_from_value(baseline_result[key])
        return set()

    def _extract_currently_passing(
        self, baseline_result: dict[str, Any]
    ) -> set[str]:
        explicit = _first_present_set(
            baseline_result,
            (
                "currently_passing",
                "passing_tests",
                "passed_tests",
                "passed_node_ids",
            ),
        )
        if explicit:
            return explicit
        return _nodes_with_outcomes(
            baseline_result.get("results", []),
            {"passed", "xpassed"},
        )

    def _failing_node_ids(self, suite_result: SuiteResult) -> list[str]:
        return [
            result.node_id
            for result in suite_result.results
            if result.outcome in REGRESSION_OUTCOMES
        ]

    def _merge_conflict_regressions(
        self, merge_result: MergeResult
    ) -> list[str]:
        return [
            f"merge_conflict:{branch}" for branch in merge_result.conflicts
        ]

    async def _notify_regression_debugger(
        self,
        *,
        regressions: list[str],
        batch_branches: list[str],
        integration_branch: str,
        worktree_path: str,
    ) -> None:
        if not regressions or self._regression_debugger is None:
            return

        debugger = self._regression_debugger
        if hasattr(debugger, "debug"):
            result = debugger.debug(
                regressions=regressions,
                batch_branches=batch_branches,
                integration_branch=integration_branch,
                worktree_path=worktree_path,
            )
        elif callable(debugger):
            result = debugger(
                regressions=regressions,
                batch_branches=batch_branches,
                integration_branch=integration_branch,
                worktree_path=worktree_path,
            )
        else:
            LOGGER.warning("Regression debugger is not callable: %r", debugger)
            return

        if inspect.isawaitable(result):
            await result


def _run_git(
    args: list[str], worktree_path: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )


def _first_present_set(
    baseline_result: dict[str, Any], keys: tuple[str, ...]
) -> set[str]:
    for key in keys:
        if key in baseline_result:
            values = _node_ids_from_value(baseline_result[key])
            if values:
                return values
    return set()


def _node_ids_from_value(value: object) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, Mapping):
        string_keys = {key for key in value if isinstance(key, str)}
        if string_keys:
            return string_keys
        return _node_ids_from_iterable(value.values())
    if isinstance(value, list | tuple | set | frozenset):
        return _node_ids_from_iterable(value)
    return set()


def _node_ids_from_iterable(values: Iterable[object]) -> set[str]:
    node_ids: set[str] = set()
    for item in values:
        if isinstance(item, str):
            node_ids.add(item)
        elif isinstance(item, Mapping):
            node_id = _mapping_node_id(item)
            if isinstance(node_id, str):
                node_ids.add(node_id)
    return node_ids


def _mapping_node_id(item: Mapping[object, object]) -> object:
    node_id = item.get("node_id")
    return node_id if node_id is not None else item.get("nodeid")


def _nodes_with_outcomes(results: object, outcomes: set[str]) -> set[str]:
    if not isinstance(results, list | tuple):
        return set()

    node_ids: set[str] = set()
    for item in results:
        if not isinstance(item, Mapping):
            continue
        outcome = item.get("outcome")
        node_id = _mapping_node_id(item)
        if isinstance(outcome, TestOutcome):
            outcome_value = outcome.value
        else:
            outcome_value = str(outcome).lower()
        if isinstance(node_id, str) and outcome_value in outcomes:
            node_ids.add(node_id)
    return node_ids
