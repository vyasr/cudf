# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import AsyncMock, patch

_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PIPELINE_ROOT.parent
for _path in (str(_PIPELINE_ROOT), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from pandas_compat_pipeline.src.orchestrator import graph  # noqa: E402
from pandas_compat_pipeline.src.orchestrator.state import PipelineState  # noqa: E402
from pandas_compat_pipeline.src.utils.models import TestGroup  # noqa: E402

_META_KEY = "__graph_meta__"


def _make_state(
    test_group: TestGroup,
    baseline_results: Mapping[str, object] | None = None,
) -> PipelineState:
    """Build a minimal PipelineState with a single task assignment."""
    task_name = test_group.base_name
    return {
        "in_progress": {
            task_name: {
                "worker_id": 0,
                "gpu_id": 0,
                "worktree_path": "/tmp/worktree",
                "test_group": {
                    "base_name": test_group.base_name,
                    "file_path": test_group.file_path,
                    "class_name": test_group.class_name,
                    "parametrizations": test_group.parametrizations,
                    "weight": test_group.weight,
                    "reasons": test_group.reasons,
                    "node_ids": test_group.node_ids,
                },
                "status": "assigned",
                "attempts": 0,
            },
            _META_KEY: {"current_task": task_name},
        },
        "completed": [],
        "failed": [],
        "flagged_for_human": [],
        "worker_status": {},
        "pending_tests": [],
        "integration_queue": [],
        "integration_results": [],
        "baseline_results": baseline_results,
        "fixes_since_last_integration": 0,
        "total_fixes": 0,
    }


@patch(
    "pandas_compat_pipeline.src.orchestrator.graph.load_config",
)
@patch(
    "pandas_compat_pipeline.src.orchestrator.graph.Dispatcher",
)
@patch(
    "pandas_compat_pipeline.src.orchestrator.graph.FixerAgent",
)
def test_collection_failure_pre_filtered(
    mock_fixer_cls, mock_dispatcher_cls, mock_load_config
) -> None:
    """Group with collection failure reason is pre-filtered without calling FixerAgent."""
    mock_dispatcher_cls.return_value.fail_task = AsyncMock()

    group = TestGroup(
        base_name="tests/test_foo.py::test_bar",
        file_path="tests/test_foo.py",
        class_name=None,
        node_ids=["tests/test_foo.py::test_bar[param1]"],
        reasons=["not found"],
    )
    state = _make_state(group)

    result = graph.fix(state)

    # Should NOT call FixerAgent.fix()
    mock_fixer_cls.assert_not_called()

    # Should emit to failed with pre_filtered reason
    assert result["failed"] == [
        {
            "test_name": group.base_name,
            "reason": "pre_filtered: collection_failure",
        }
    ]
    assert result["in_progress"][group.base_name]["status"] == "failed"


@patch(
    "pandas_compat_pipeline.src.orchestrator.graph.load_config",
)
@patch(
    "pandas_compat_pipeline.src.orchestrator.graph.Dispatcher",
)
@patch(
    "pandas_compat_pipeline.src.orchestrator.graph.FixerAgent",
)
def test_deselected_pre_filtered(
    mock_fixer_cls, mock_dispatcher_cls, mock_load_config
) -> None:
    """Group with deselected reason is pre-filtered without calling FixerAgent."""
    mock_dispatcher_cls.return_value.fail_task = AsyncMock()

    group = TestGroup(
        base_name="tests/test_baz.py::test_qux",
        file_path="tests/test_baz.py",
        class_name=None,
        node_ids=["tests/test_baz.py::test_qux[x]"],
        reasons=["collected 1 item / 1 deselected"],
    )
    state = _make_state(group)

    result = graph.fix(state)

    # Should NOT call FixerAgent.fix()
    mock_fixer_cls.assert_not_called()

    # Should emit to failed with pre_filtered deselected reason
    assert result["failed"] == [
        {"test_name": group.base_name, "reason": "pre_filtered: deselected"}
    ]
    assert result["in_progress"][group.base_name]["status"] == "failed"


@patch(
    "pandas_compat_pipeline.src.orchestrator.graph.load_config",
)
@patch(
    "pandas_compat_pipeline.src.orchestrator.graph.Dispatcher",
)
@patch(
    "pandas_compat_pipeline.src.orchestrator.graph.FixerAgent",
)
def test_normal_group_not_pre_filtered(
    mock_fixer_cls, mock_dispatcher_cls, mock_load_config
) -> None:
    """Normal failure group proceeds to FixerAgent.fix() without pre-filtering."""
    mock_config = mock_load_config.return_value
    mock_config.max_fix_attempts = 3

    mock_fix_result = AsyncMock()
    mock_fix_result.status = "success"
    mock_fixer_cls.return_value.fix = AsyncMock(return_value=mock_fix_result)

    group = TestGroup(
        base_name="tests/test_normal.py::test_works",
        file_path="tests/test_normal.py",
        class_name=None,
        node_ids=["tests/test_normal.py::test_works[a]"],
        reasons=["TypeError: unsupported operand"],
    )
    state = _make_state(group)

    result = graph.fix(state)

    # Should call FixerAgent.fix()
    mock_fixer_cls.return_value.fix.assert_called_once()

    # Should NOT appear in failed
    assert "failed" not in result or result.get("failed") is None
    assert result["in_progress"][group.base_name]["status"] == "success"


@patch(
    "pandas_compat_pipeline.src.orchestrator.graph.load_config",
)
@patch(
    "pandas_compat_pipeline.src.orchestrator.graph.Dispatcher",
)
@patch(
    "pandas_compat_pipeline.src.orchestrator.graph.FixerAgent",
)
def test_stale_baseline_entries_pre_filtered(
    mock_fixer_cls, mock_dispatcher_cls, mock_load_config
) -> None:
    """Stale baseline entries alone do not pre-filter a group."""
    mock_config = mock_load_config.return_value
    mock_config.max_fix_attempts = 3

    mock_fix_result = AsyncMock()
    mock_fix_result.status = "success"
    mock_fixer_cls.return_value.fix = AsyncMock(return_value=mock_fix_result)

    group = TestGroup(
        base_name="tests/test_old.py::test_removed",
        file_path="tests/test_old.py",
        class_name=None,
        node_ids=[
            "tests/test_old.py::test_removed[a]",
            "tests/test_old.py::test_removed[b]",
        ],
        reasons=["some unrelated reason"],
    )
    baseline = {
        "stale_entries": [
            "tests/test_old.py::test_removed[a]",
            "tests/test_old.py::test_removed[b]",
            "tests/test_other.py::test_x",
        ],
    }
    state = _make_state(group, baseline_results=baseline)

    result = graph.fix(state)

    # Should still proceed to FixerAgent.fix()
    mock_fixer_cls.return_value.fix.assert_called_once()
    assert "failed" not in result or result.get("failed") is None
    assert result["in_progress"][group.base_name]["status"] == "success"
