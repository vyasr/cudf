# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from pandas_compat_pipeline.src.agents.llm_client import LLMClient  # noqa: E402
from pandas_compat_pipeline.src.utils.models import TestGroup  # noqa: E402
from pandas_compat_pipeline.src.utils.test_runner import (  # noqa: E402
    TestOutcome,
    TestResult,
)


def _make_agent() -> FixerAgent:
    """Create a FixerAgent with a mocked LLM client."""
    client = MagicMock(spec=LLMClient)
    agent = FixerAgent(llm_client=client)
    return agent


def _make_test_group() -> TestGroup:
    return TestGroup(
        base_name="test_something",
        file_path="tests/test_something.py",
        class_name=None,
        node_ids=["tests/test_something.py::test_something[param1]"],
    )


def test_stale_xfail_bypass() -> None:
    """Stale-xfail group (_diag_baseline_passed=True): skip fail_on_fallback pass.

    When the strict pass succeeds, verify that _run_test_async is only called
    with fail_on_fallback=False (the strict pass), not fail_on_fallback=True.
    This avoids conftest import errors blocking stale-xfail fixes.
    """
    agent = _make_agent()
    agent._diag_baseline_passed = True

    test_group = _make_test_group()

    # Strict pass returns PASSED
    strict_result = TestResult(
        node_id=test_group.node_ids[0],
        outcome=TestOutcome.PASSED,
        duration=1.0,
    )

    agent._run_test_async = AsyncMock(return_value=strict_result)

    results = asyncio.run(
        agent._verify_fix("/tmp/worktree", test_group, gpu_id=0)
    )

    # Only strict pass should have been run (1 call per node_id)
    assert agent._run_test_async.call_count == 1
    call_kwargs = agent._run_test_async.call_args_list[0]
    assert (
        call_kwargs.kwargs.get("fail_on_fallback") is False
        or (call_kwargs.args[4] if len(call_kwargs.args) > 4 else None)
        is False
    )
    # Results contain only the strict pass result
    assert len(results) == 1
    assert results[0].outcome == TestOutcome.PASSED


def test_non_stale_xfail_no_bypass() -> None:
    """Non-stale-xfail group: both passes run, fallback failure → results contain failures.

    When _diag_baseline_passed is False, both verification passes (strict and
    fallback) are executed. A failure in the fallback pass is included in results.
    """
    agent = _make_agent()
    agent._diag_baseline_passed = False

    test_group = _make_test_group()

    strict_result = TestResult(
        node_id=test_group.node_ids[0],
        outcome=TestOutcome.PASSED,
        duration=1.0,
    )
    fallback_result = TestResult(
        node_id=test_group.node_ids[0],
        outcome=TestOutcome.ERRORED,
        duration=0.5,
        longrepr="ImportError while loading conftest",
    )

    # First call (strict) returns PASSED, second call (fallback) returns ERRORED
    agent._run_test_async = AsyncMock(
        side_effect=[strict_result, fallback_result]
    )

    results = asyncio.run(
        agent._verify_fix("/tmp/worktree", test_group, gpu_id=0)
    )

    # Both passes should have run (2 calls: 1 strict + 1 fallback)
    assert agent._run_test_async.call_count == 2
    # Results include both outcomes
    assert len(results) == 2
    assert results[0].outcome == TestOutcome.PASSED
    assert results[1].outcome == TestOutcome.ERRORED

    # Verify the calls used correct fail_on_fallback values
    first_call = agent._run_test_async.call_args_list[0]
    second_call = agent._run_test_async.call_args_list[1]
    # positional: worktree_path, node_id, gpu_id; keyword: run_mode, fail_on_fallback
    assert first_call.kwargs["fail_on_fallback"] is False
    assert second_call.kwargs["fail_on_fallback"] is True
