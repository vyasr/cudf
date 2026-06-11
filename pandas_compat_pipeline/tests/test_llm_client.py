# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path[:0] = [
    _path
    for _path in (
        str(Path(__file__).resolve().parents[1]),
        str(Path(__file__).resolve().parents[1].parent),
    )
    if _path not in sys.path
]

LLMClient = importlib.import_module(
    "pandas_compat_pipeline.src.agents.llm_client"
).LLMClient


def _mock_response(
    *, content: str | None, tool_calls: list[SimpleNamespace] | None
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls)
            )
        ]
    )


def test_tool_calls_prioritized_over_content() -> None:
    tool_call = SimpleNamespace(
        model_dump=lambda: {
            "id": "call_1",
            "type": "function",
            "function": {"name": "apply_patch", "arguments": "{}"},
        }
    )
    client = LLMClient(api_key="test-key")

    with patch(
        "pandas_compat_pipeline.src.agents.llm_client.litellm.acompletion",
        new=AsyncMock(
            return_value=_mock_response(
                content="thinking text",
                tool_calls=[tool_call],
            )
        ),
    ):
        result = asyncio.run(
            client._call_with_retry(
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
            )
        )

    assert json.loads(result) == [tool_call.model_dump()]


def test_content_returned_when_no_tool_calls() -> None:
    client = LLMClient(api_key="test-key")

    with patch(
        "pandas_compat_pipeline.src.agents.llm_client.litellm.acompletion",
        new=AsyncMock(
            return_value=_mock_response(
                content="plain response",
                tool_calls=None,
            )
        ),
    ):
        result = asyncio.run(
            client._call_with_retry(
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
            )
        )

    assert result == "plain response"
