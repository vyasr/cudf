# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
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

from pandas_compat_pipeline.src.agents.fixer import (  # noqa: E402
    FixerAgent,
    HumanReviewRequired,
)
from pandas_compat_pipeline.src.agents.llm_client import (  # noqa: E402
    LLMClient,
)


def _make_agent() -> FixerAgent:
    """Create a FixerAgent with a mocked LLM client."""
    client = MagicMock(spec=LLMClient)
    agent = FixerAgent(llm_client=client)
    agent._diag_cycles = []
    return agent


def test_reprompt_on_prose_then_tool_calls() -> None:
    """LLM returns prose on first call, valid tool_calls on second → tool execution occurs."""
    agent = _make_agent()
    agent.config.max_fix_attempts = 1

    # First call: prose (no valid JSON tool calls)
    # Second call: valid tool call JSON
    valid_tool_call = json.dumps(
        [
            {
                "function": "read_file",
                "arguments": {"path": "test.py"},
            }
        ]
    )

    agent.llm_client.call_fixer = AsyncMock(
        side_effect=["I think you should look at...", valid_tool_call]
    )
    agent._execute_tool_call = AsyncMock(
        return_value={"wrote_patch": True, "output": "ok"}
    )

    messages: list[dict[str, str]] = [
        {"role": "user", "content": "Fix the test"}
    ]

    # With max_fix_attempts=1, after successful tool execution (attempts=1),
    # the loop exits with HumanReviewRequired since 1 < 1 is false.
    try:
        asyncio.run(
            agent._tool_loop(
                messages=messages,
                worktree_path="/tmp/wt",
                modified_files=[],
            )
        )
        raised = False
    except HumanReviewRequired:
        raised = True

    assert raised
    # Tool execution should have occurred (second LLM call returned valid tool calls)
    assert agent._execute_tool_call.called
    # call_fixer called exactly 2 times: prose (reprompted) + valid tool calls
    assert agent.llm_client.call_fixer.call_count == 2
    # The reprompt message should have been appended to messages
    assert any(
        "ERROR: Your response must be a JSON array" in m.get("content", "")
        for m in messages
        if m.get("role") == "user"
    )
    # Diagnostic should have captured reprompted=True for the prose cycle
    prose_diags = [
        d for d in agent._diag_cycles if d.get("reprompted") is True
    ]
    assert len(prose_diags) == 1


def test_reprompt_budget_enforced() -> None:
    """LLM returns prose 3+ times → _tool_loop returns after 2 reprompts (not infinite loop)."""
    agent = _make_agent()

    # Always return prose (never valid tool calls)
    agent.llm_client.call_fixer = AsyncMock(
        return_value="Let me explain what happened..."
    )
    agent._execute_tool_call = AsyncMock(
        return_value={"wrote_patch": False, "output": ""}
    )

    messages: list[dict[str, str]] = [
        {"role": "user", "content": "Fix the test"}
    ]

    result = asyncio.run(
        agent._tool_loop(
            messages=messages,
            worktree_path="/tmp/wt",
            modified_files=[],
        )
    )

    # Should return (not raise, not infinite loop)
    assert "diagnosis" in result
    # attempts should be 0 (no real tool-call iteration ever executed)
    assert result["attempts"] == 0
    # LLM was called exactly 3 times: initial + 2 reprompts
    assert agent.llm_client.call_fixer.call_count == 3
    # _execute_tool_call should never have been called
    assert not agent._execute_tool_call.called
    # Diagnostics should show reprompted=True for first 2, reprompted=False for last
    assert len(agent._diag_cycles) == 3
    assert agent._diag_cycles[0]["reprompted"] is True
    assert agent._diag_cycles[1]["reprompted"] is True
    assert agent._diag_cycles[2]["reprompted"] is False
