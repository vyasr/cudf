# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangGraph state schema for the pandas compatibility fix pipeline."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def _dict_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Reducer that merges two dicts (right overwrites left on conflict)."""
    return {**left, **right}


def _overwrite(_left: Any, right: Any) -> Any:
    """Reducer that always takes the newest value."""
    return right


class PipelineState(TypedDict):
    """TypedDict state schema for the LangGraph pandas compatibility fix pipeline.

    Uses Annotated types with reducers for LangGraph's state management.
    All fields are JSON-serializable for subprocess-based workers.
    State does NOT contain the full xfail list — only test names by reference.
    """

    pending_tests: Annotated[list[str], operator.add]
    """Test group base names in priority order (highest-weight first)."""

    in_progress: Annotated[dict[str, Any], _dict_merge]
    """Currently assigned tests. Keys are test group names, values are assignment info."""

    completed: Annotated[list[Any], operator.add]
    """Successfully fixed tests with metadata (group name, branch, timestamp)."""

    failed: Annotated[list[Any], operator.add]
    """Tests that exhausted all fix attempts."""

    flagged_for_human: Annotated[list[Any], operator.add]
    """Tests needing C++ changes or otherwise unresolvable by automation."""

    integration_queue: Annotated[list[str], operator.add]
    """Branches pending integration test."""

    integration_results: Annotated[list[Any], operator.add]
    """Integration test pass/fail history."""

    baseline_results: Annotated[Any | None, _overwrite]
    """Baseline run result (initial test suite state)."""

    fixes_since_last_integration: Annotated[int, _overwrite]
    """Counter of fixes since last integration test run."""

    total_fixes: Annotated[int, _overwrite]
    """Total number of successful fixes across the pipeline run."""

    worker_status: Annotated[dict[int, Any], _dict_merge]
    """Per-GPU worker state. Keys are GPU indices, values are status dicts."""
