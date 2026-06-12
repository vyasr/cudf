# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

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

import pytest  # noqa: E402


def _make_agent() -> FixerAgent:
    """Create a FixerAgent with a mocked LLM client."""
    client = MagicMock(spec=LLMClient)
    agent = FixerAgent(llm_client=client)
    agent._diag_cycles = []
    return agent


def _make_test_group() -> TestGroup:
    """Create a minimal TestGroup for commit tests."""
    return TestGroup(
        base_name="tests/test_foo.py::test_bar",
        file_path="tests/test_foo.py",
        class_name=None,
        node_ids=["tests/test_foo.py::test_bar[param1]"],
    )


def _success_result(cmd: str = "") -> dict:
    return {"success": True, "stdout": "", "stderr": "", "returncode": 0, "timed_out": False, "cmd": cmd}


def _failure_result(cmd: str = "", stderr: str = "hook failed") -> dict:
    return {"success": False, "stdout": "", "stderr": stderr, "returncode": 1, "timed_out": False, "cmd": cmd}


def test_precommit_ruff_called_before_git_add() -> None:
    """pre-commit run ruff-format is called BEFORE git add in the call order."""
    agent = _make_agent()
    agent._validate_or_raise = MagicMock()

    call_log: list[str] = []

    def mock_run_command(cmd: str, cwd: str | None = None, timeout: int = 300) -> dict:
        call_log.append(cmd)
        return _success_result(cmd)

    with patch("pandas_compat_pipeline.src.agents.fixer.run_command", side_effect=mock_run_command):
        asyncio.run(
            agent._commit_success(
                worktree_path="/tmp/wt",
                branch_name="fix-branch",
                modified_files=["src/foo.py"],
                test_group=_make_test_group(),
            )
        )

    ruff_indices = [i for i, c in enumerate(call_log) if "pre-commit run ruff-format" in c]
    add_indices = [i for i, c in enumerate(call_log) if c.startswith("git add")]
    assert ruff_indices, "pre-commit ruff-format was never called"
    assert add_indices, "git add was never called"
    assert ruff_indices[0] < add_indices[0], "ruff-format must be called before git add"


def test_ruff_failure_does_not_abort() -> None:
    """When ruff-format returns non-zero (reformatted), commit still proceeds."""
    agent = _make_agent()
    agent._validate_or_raise = MagicMock()

    def mock_run_command(cmd: str, cwd: str | None = None, timeout: int = 300) -> dict:
        if "pre-commit run ruff-format" in cmd:
            return _failure_result(cmd, stderr="reformatted src/foo.py")
        return _success_result(cmd)

    with patch("pandas_compat_pipeline.src.agents.fixer.run_command", side_effect=mock_run_command):
        # Should NOT raise
        asyncio.run(
            agent._commit_success(
                worktree_path="/tmp/wt",
                branch_name="fix-branch",
                modified_files=["src/foo.py"],
                test_group=_make_test_group(),
            )
        )


def test_noverify_fallback_on_commit_failure() -> None:
    """When first git commit fails, retry with --no-verify."""
    agent = _make_agent()
    agent._validate_or_raise = MagicMock()

    call_log: list[str] = []

    def mock_run_command(cmd: str, cwd: str | None = None, timeout: int = 300) -> dict:
        call_log.append(cmd)
        if cmd.startswith("git commit") and "--no-verify" not in cmd:
            return _failure_result(cmd, stderr="mypy timeout")
        return _success_result(cmd)

    with patch("pandas_compat_pipeline.src.agents.fixer.run_command", side_effect=mock_run_command):
        asyncio.run(
            agent._commit_success(
                worktree_path="/tmp/wt",
                branch_name="fix-branch",
                modified_files=["src/foo.py"],
                test_group=_make_test_group(),
            )
        )

    noverify_calls = [c for c in call_log if "--no-verify" in c]
    assert noverify_calls, "Expected --no-verify fallback call"


def test_both_commits_fail_raises() -> None:
    """When both commit attempts fail, RuntimeError is raised."""
    agent = _make_agent()
    agent._validate_or_raise = MagicMock()

    def mock_run_command(cmd: str, cwd: str | None = None, timeout: int = 300) -> dict:
        if cmd.startswith("git commit"):
            return _failure_result(cmd, stderr="fatal error")
        return _success_result(cmd)

    with patch("pandas_compat_pipeline.src.agents.fixer.run_command", side_effect=mock_run_command):
        with pytest.raises(RuntimeError, match="git commit failed on"):
            asyncio.run(
                agent._commit_success(
                    worktree_path="/tmp/wt",
                    branch_name="fix-branch",
                    modified_files=["src/foo.py"],
                    test_group=_make_test_group(),
                )
            )


def test_first_commit_success_no_fallback() -> None:
    """When first git commit succeeds, --no-verify is NOT called."""
    agent = _make_agent()
    agent._validate_or_raise = MagicMock()

    call_log: list[str] = []

    def mock_run_command(cmd: str, cwd: str | None = None, timeout: int = 300) -> dict:
        call_log.append(cmd)
        return _success_result(cmd)

    with patch("pandas_compat_pipeline.src.agents.fixer.run_command", side_effect=mock_run_command):
        asyncio.run(
            agent._commit_success(
                worktree_path="/tmp/wt",
                branch_name="fix-branch",
                modified_files=["src/foo.py"],
                test_group=_make_test_group(),
            )
        )

    noverify_calls = [c for c in call_log if "--no-verify" in c]
    assert not noverify_calls, "Should NOT call --no-verify when first commit succeeds"
