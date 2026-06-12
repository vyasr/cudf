# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for OpenAI/LiteLLM tool definition format."""

from pandas_compat_pipeline.src.agents.tools import get_tools


def test_get_tools_has_function_type_wrapper() -> None:
    tools = get_tools()

    assert all(entry["type"] == "function" for entry in tools)


def test_get_tools_function_names() -> None:
    tools = get_tools()

    names = {entry["function"]["name"] for entry in tools}

    assert names == {
        "read_file",
        "write_file",
        "run_command",
        "search_code",
    }


def test_get_tools_returns_four_tools() -> None:
    assert len(get_tools()) == 4


def test_get_tools_function_keys() -> None:
    tools = get_tools()

    for entry in tools:
        assert "name" in entry["function"]
        assert "description" in entry["function"]
        assert "parameters" in entry["function"]
