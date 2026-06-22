# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests documenting the multi-turn tool-calling protocol in FixerAgent._tool_loop().

These tests exercise the conversation format between the agent and the LLM,
including how tool calls are parsed, executed, and how results are appended
to the message history. They also document Bug 1's fixed behavior: assistant
messages store tool calls in the proper OpenAI tool_calls field instead of raw
JSON content.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
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


def _tool_call_json(name: str, **arguments) -> str:
    """Build a JSON string representing a single tool call list."""
    return json.dumps(
        [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ]
    )


# ---------------------------------------------------------------------------
# Test 1: _parse_tool_calls correctly parses structured response
# ---------------------------------------------------------------------------


def test_tool_calls_parsed_from_structured_response() -> None:
    """_parse_tool_calls parses a valid JSON tool-call array."""
    agent = _make_agent()

    response = json.dumps(
        [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "test.py"}',
                },
            }
        ]
    )

    tool_calls = agent._parse_tool_calls(response)

    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "read_file"


# ---------------------------------------------------------------------------
# Test 2: Multi-turn tool loop executes tools across iterations
# ---------------------------------------------------------------------------


def test_multi_turn_tool_loop_executes_tools() -> None:
    """_tool_loop executes tool calls across multiple iterations until max_fix_attempts."""
    agent = _make_agent()
    agent.config.max_fix_attempts = 2

    # First call: read_file tool call; Second call: write_file tool call
    agent.llm_client.call_fixer = AsyncMock(
        side_effect=[
            _tool_call_json("read_file", path="test.py"),
            _tool_call_json("write_file", path="test.py", content="fixed"),
        ]
    )
    agent._execute_tool_call = AsyncMock(
        side_effect=[
            {"wrote_patch": False, "output": "file contents"},
            {"wrote_patch": True, "output": "written"},
        ]
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Fix the test"}
    ]

    # After 2 iterations (attempts == max_fix_attempts), HumanReviewRequired is raised
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
    assert agent._execute_tool_call.call_count == 2


# ---------------------------------------------------------------------------
# Test 3: Prose response triggers reprompt message
# ---------------------------------------------------------------------------


def test_prose_response_triggers_reprompt() -> None:
    """When LLM returns prose, a reprompt error message is appended before retrying."""
    agent = _make_agent()
    agent.config.max_fix_attempts = 1

    # First call: prose; Second call: valid tool call
    agent.llm_client.call_fixer = AsyncMock(
        side_effect=[
            "Let me explain what's happening here...",
            _tool_call_json("read_file", path="test.py"),
        ]
    )
    agent._execute_tool_call = AsyncMock(
        return_value={"wrote_patch": True, "output": "ok"}
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Fix the test"}
    ]

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
    # Reprompt message should be in the messages
    reprompt_msgs = [
        m
        for m in messages
        if m.get("role") == "user"
        and "ERROR: Your response must be a JSON array" in m.get("content", "")
    ]
    assert len(reprompt_msgs) >= 1


# ---------------------------------------------------------------------------
# Test 4: Three prose responses returns diagnosis (budget enforced)
# ---------------------------------------------------------------------------


def test_three_prose_responses_returns_diagnosis() -> None:
    """After 3 consecutive prose responses (initial + 2 reprompts), _tool_loop returns a diagnosis.

    This documents the reprompt budget: the agent gives up after 2 reprompts
    and returns the last prose response as a diagnosis dict.
    """
    agent = _make_agent()

    agent.llm_client.call_fixer = AsyncMock(
        return_value="I cannot produce tool calls."
    )
    agent._execute_tool_call = AsyncMock(
        return_value={"wrote_patch": False, "output": ""}
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Fix the test"}
    ]

    result = asyncio.run(
        agent._tool_loop(
            messages=messages,
            worktree_path="/tmp/wt",
            modified_files=[],
        )
    )

    assert "diagnosis" in result
    assert agent.llm_client.call_fixer.call_count == 3
    assert not agent._execute_tool_call.called


# ---------------------------------------------------------------------------
# Test 5: Tool results format in messages
# ---------------------------------------------------------------------------


def test_tool_results_format_in_messages() -> None:
    r"""After tool execution, results are appended as a user message with JSON content.

    The format is:
      {"role": "user", "content": "Tool results:\n" + json.dumps(results)}
    """
    agent = _make_agent()
    agent.config.max_fix_attempts = 1

    agent.llm_client.call_fixer = AsyncMock(
        return_value=_tool_call_json("read_file", path="test.py")
    )
    agent._execute_tool_call = AsyncMock(
        return_value={"wrote_patch": False, "output": "file contents here"}
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Fix the test"}
    ]

    try:
        asyncio.run(
            agent._tool_loop(
                messages=messages,
                worktree_path="/tmp/wt",
                modified_files=[],
            )
        )
    except HumanReviewRequired:
        pass

    # Find user message with "Tool results:\n"
    tool_result_msgs = [
        m
        for m in messages
        if m.get("role") == "user"
        and m.get("content", "").startswith("Tool results:\n")
    ]
    assert len(tool_result_msgs) >= 1

    # Parse the JSON content after the prefix
    content = tool_result_msgs[0]["content"]
    json_str = content[len("Tool results:\n") :]
    parsed = json.loads(json_str)

    assert isinstance(parsed, list)
    assert parsed[0]["output"] == "file contents here"


# ---------------------------------------------------------------------------
# Test 6: Assistant message contains structured tool_calls (Bug 1 fixed)
# ---------------------------------------------------------------------------


def test_assistant_message_contains_response_string() -> None:
    """FIXED BEHAVIOR: tool_calls field used instead of raw JSON content.

    Proper OpenAI protocol uses a tool_calls field on the assistant message
    rather than stuffing the raw JSON string into content.
    """
    agent = _make_agent()
    agent.config.max_fix_attempts = 1

    tool_call_str = _tool_call_json("read_file", path="test.py")

    agent.llm_client.call_fixer = AsyncMock(return_value=tool_call_str)
    agent._execute_tool_call = AsyncMock(
        return_value={"wrote_patch": False, "output": "contents"}
    )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Fix the test"}
    ]

    try:
        asyncio.run(
            agent._tool_loop(
                messages=messages,
                worktree_path="/tmp/wt",
                modified_files=[],
            )
        )
    except HumanReviewRequired:
        pass

    # Find assistant messages
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    assert len(assistant_msgs) >= 1
    assert assistant_msgs[0]["role"] == "assistant"
    assert "tool_calls" in assistant_msgs[0]
    assert (
        assistant_msgs[0]["tool_calls"][0]["function"]["name"] == "read_file"
    )

    # The raw JSON string must not be stored in content when tool_calls exist.
    assert assistant_msgs[0].get("content") is None
