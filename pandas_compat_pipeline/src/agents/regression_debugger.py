# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# pyright: reportExplicitAny=false, reportAny=false
import asyncio
import itertools
import logging
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..config import PipelineConfig, load_config
from ..utils.branch_manager import (
    BRANCH_PREFIX,
    revert_branch_from_integration,
)
from ..utils.test_runner import TestOutcome, run_test

LOGGER = logging.getLogger(__name__)
DEFAULT_DEBUG_GPU = 0


@dataclass(slots=True)
class RegressionResult:
    culprit_branch: str | None
    affected_tests: list[str]
    requeue_tasks: list[dict[str, str]]


@dataclass(slots=True)
class _Diagnosis:
    culprit_branch: str | None
    active_branches: list[str]
    affected_tests: list[str]


class RegressionDebuggerAgent:
    """Diagnose integration regressions by selectively reverting fix branches."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config: PipelineConfig = config or load_config()
        self.gpu_id: int = (
            self.config.integration_gpu
            if self.config.integration_gpu is not None
            else DEFAULT_DEBUG_GPU
        )

    async def debug(
        self,
        regressions: list[str],
        batch_branches: list[str],
        integration_branch: str,
        worktree_path: str,
    ) -> RegressionResult:
        """Find the fix branch or branch interaction responsible for regressions."""

        affected_tests = _dedupe(regressions)
        branches = _dedupe(batch_branches)
        if not affected_tests or not branches:
            return RegressionResult(
                culprit_branch=None,
                affected_tests=affected_tests,
                requeue_tasks=[],
            )

        try:
            await self._reset_integration(integration_branch, worktree_path)
            if len(branches) == 1:
                diagnosis = await self._diagnose_small_set(
                    branches,
                    affected_tests,
                    branches,
                    integration_branch,
                    worktree_path,
                )
            else:
                diagnosis = await self._bisect(
                    branches, affected_tests, integration_branch, worktree_path
                )
        except Exception:
            LOGGER.exception(
                "Regression debugging failed for %s", affected_tests
            )
            diagnosis = _Diagnosis(
                culprit_branch=None,
                active_branches=[],
                affected_tests=affected_tests,
            )
        finally:
            await self._reset_integration(integration_branch, worktree_path)

        return RegressionResult(
            culprit_branch=diagnosis.culprit_branch,
            affected_tests=diagnosis.affected_tests,
            requeue_tasks=self._build_requeue_tasks(
                diagnosis.culprit_branch, diagnosis.affected_tests
            ),
        )

    async def _bisect(
        self,
        branches: list[str],
        regressions: list[str],
        integration_branch: str,
        worktree_path: str,
    ) -> _Diagnosis:
        all_batch_branches = list(branches)
        candidates = list(branches)
        while len(candidates) > 3:
            midpoint = len(candidates) // 2
            left = candidates[:midpoint]
            right = candidates[midpoint:]

            right_failures = await self._failing_tests_with_active_branches(
                active_branches=right,
                all_branches=all_batch_branches,
                regressions=regressions,
                integration_branch=integration_branch,
                worktree_path=worktree_path,
            )
            if right_failures:
                candidates = right
                regressions = right_failures
                continue

            left_failures = await self._failing_tests_with_active_branches(
                active_branches=left,
                all_branches=all_batch_branches,
                regressions=regressions,
                integration_branch=integration_branch,
                worktree_path=worktree_path,
            )
            if left_failures:
                candidates = left
                regressions = left_failures
                continue

            interaction = await self._find_cross_half_interaction(
                left,
                right,
                all_batch_branches,
                regressions,
                integration_branch,
                worktree_path,
            )
            if interaction.culprit_branch is not None:
                return interaction

            return _Diagnosis(
                culprit_branch=None,
                active_branches=[],
                affected_tests=regressions,
            )

        return await self._diagnose_small_set(
            candidates,
            regressions,
            all_batch_branches,
            integration_branch,
            worktree_path,
        )

    async def _diagnose_small_set(
        self,
        candidates: list[str],
        regressions: list[str],
        all_branches: list[str],
        integration_branch: str,
        worktree_path: str,
    ) -> _Diagnosis:
        for branch in candidates:
            failures = await self._failing_tests_with_active_branches(
                active_branches=[branch],
                all_branches=all_branches,
                regressions=regressions,
                integration_branch=integration_branch,
                worktree_path=worktree_path,
            )
            if failures:
                return _Diagnosis(
                    culprit_branch=branch,
                    active_branches=[branch],
                    affected_tests=failures,
                )

        if len(candidates) > 1:
            for pair in itertools.combinations(candidates, 2):
                active_pair = list(pair)
                failures = await self._failing_tests_with_active_branches(
                    active_branches=active_pair,
                    all_branches=all_branches,
                    regressions=regressions,
                    integration_branch=integration_branch,
                    worktree_path=worktree_path,
                )
                if failures:
                    return _Diagnosis(
                        culprit_branch=_interaction_label(active_pair),
                        active_branches=active_pair,
                        affected_tests=failures,
                    )

        failures = await self._failing_tests_with_active_branches(
            active_branches=candidates,
            all_branches=all_branches,
            regressions=regressions,
            integration_branch=integration_branch,
            worktree_path=worktree_path,
        )
        if failures:
            culprit = (
                candidates[0]
                if len(candidates) == 1
                else _interaction_label(candidates)
            )
            return _Diagnosis(
                culprit_branch=culprit,
                active_branches=candidates,
                affected_tests=failures,
            )

        return _Diagnosis(
            culprit_branch=None,
            active_branches=[],
            affected_tests=regressions,
        )

    async def _find_cross_half_interaction(
        self,
        left: list[str],
        right: list[str],
        all_branches: list[str],
        regressions: list[str],
        integration_branch: str,
        worktree_path: str,
    ) -> _Diagnosis:
        for first in left:
            for second in right:
                active_pair = [first, second]
                failures = await self._failing_tests_with_active_branches(
                    active_branches=active_pair,
                    all_branches=all_branches,
                    regressions=regressions,
                    integration_branch=integration_branch,
                    worktree_path=worktree_path,
                )
                if failures:
                    return _Diagnosis(
                        culprit_branch=_interaction_label(active_pair),
                        active_branches=active_pair,
                        affected_tests=failures,
                    )
        return _Diagnosis(
            culprit_branch=None, active_branches=[], affected_tests=[]
        )

    async def _failing_tests_with_active_branches(
        self,
        *,
        active_branches: Sequence[str],
        all_branches: Sequence[str],
        regressions: Sequence[str],
        integration_branch: str,
        worktree_path: str,
    ) -> list[str]:
        active_set = set(active_branches)
        branches_to_revert = [
            branch for branch in all_branches if branch not in active_set
        ]
        await self._apply_revert_subset(
            branches_to_revert, integration_branch, worktree_path
        )
        try:
            return await self._run_regression_tests(regressions, worktree_path)
        finally:
            await self._reset_integration(integration_branch, worktree_path)

    async def _apply_revert_subset(
        self,
        branches: Iterable[str],
        integration_branch: str,
        worktree_path: str,
    ) -> None:
        await self._reset_integration(integration_branch, worktree_path)
        for branch in branches:
            reverted = await asyncio.to_thread(
                revert_branch_from_integration,
                branch,
                integration_branch,
                worktree_path,
            )
            if not reverted:
                await self._reset_integration(
                    integration_branch, worktree_path
                )
                raise RuntimeError(
                    f"Could not revert {branch} from {integration_branch}"
                )

    async def _run_regression_tests(
        self, regressions: Sequence[str], worktree_path: str
    ) -> list[str]:
        failures: list[str] = []
        for node_id in regressions:
            result = await asyncio.to_thread(
                run_test,
                worktree_path,
                node_id,
                self.gpu_id,
                run_mode="verify",
                timeout_minutes=self.config.single_test_timeout_minutes,
                flakiness_reruns=self.config.flakiness_reruns,
            )
            if result.outcome in {
                TestOutcome.FAILED,
                TestOutcome.ERRORED,
                TestOutcome.XPASSED,
            }:
                failures.append(node_id)
        return failures

    async def _reset_integration(
        self, integration_branch: str, worktree_path: str
    ) -> None:
        _ = await asyncio.to_thread(
            _run_git, ["checkout", integration_branch], worktree_path
        )
        _ = await asyncio.to_thread(
            _run_git, ["reset", "--hard", integration_branch], worktree_path
        )

    def _build_requeue_tasks(
        self, culprit_branch: str | None, affected_tests: Sequence[str]
    ) -> list[dict[str, str]]:
        if culprit_branch is None:
            return []
        original_test = _original_test_from_branch(culprit_branch)
        return [
            {
                "test_group": original_test,
                "priority": "high",
                "context": f"regression against {failing_test}",
            }
            for failing_test in affected_tests
        ]


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


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _interaction_label(branches: Sequence[str]) -> str:
    return "interaction: " + " + ".join(branches)


def _original_test_from_branch(culprit_branch: str) -> str:
    if culprit_branch.startswith("interaction: "):
        return culprit_branch
    if culprit_branch.startswith(BRANCH_PREFIX):
        return culprit_branch.removeprefix(BRANCH_PREFIX)
    return culprit_branch
