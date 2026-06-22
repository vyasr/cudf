# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Bug 3: _pre_filter_reason misses marker-deselected tests.

The pre-filter currently only checks test_group.reasons text for patterns
like "deselected" or "0 selected". But tests that are deselected by marker
expression (e.g., -m "not slow and not single_cpu") simply don't appear in
xfail output — they have no "deselected" reason string. The pre-filter
should catch them but currently does not.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pandas_compat_pipeline.src.orchestrator.state import (
        PipelineState,
    )

_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PIPELINE_ROOT.parent
for _path in (str(_PIPELINE_ROOT), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from pandas_compat_pipeline.src.orchestrator import graph  # noqa: E402
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


def test_single_cpu_module_test_is_filtered() -> None:
    """FIXED: tests/io/test_sql.py is module-level single_cpu and is now correctly
    filtered by _pre_filter_reason via _MARKER_DESELECTED_MODULES.

    The runner invokes pytest with -m "not slow and not single_cpu and not db
    and not network", so any test in tests/io/test_sql.py will be deselected.
    """
    group = TestGroup(
        base_name="tests/io/test_sql.py::test_execute_sql[sqlite_engine_iris]",
        file_path="tests/io/test_sql.py",
        class_name=None,
        node_ids=[
            "tests/io/test_sql.py::test_execute_sql[sqlite_engine_iris]"
        ],
        reasons=["AssertionError: assert 1 == 2"],
    )
    state = _make_state(group)

    result = graph._pre_filter_reason(group, state)

    assert result == "pre_filtered: deselected"


def test_slow_test_is_filtered() -> None:
    """FIXED: test_range_difference is @pytest.mark.slow and is now correctly
    filtered by _pre_filter_reason via _MARKER_DESELECTED_TESTS.

    The runner uses -m "not slow and not single_cpu ...", so individually
    @pytest.mark.slow tests will be deselected.
    """
    group = TestGroup(
        base_name="tests/indexes/ranges/test_setops.py::test_range_difference",
        file_path="tests/indexes/ranges/test_setops.py",
        class_name=None,
        node_ids=[
            "tests/indexes/ranges/test_setops.py::test_range_difference"
        ],
        reasons=["TimeoutError"],
    )
    state = _make_state(group)

    result = graph._pre_filter_reason(group, state)

    assert result == "pre_filtered: deselected"


def test_valid_test_not_filtered() -> None:
    """Normal failure group that is NOT marker-deselected should pass through
    the pre-filter without being filtered out.
    """
    group = TestGroup(
        base_name="tests/frame/test_constructors.py::test_constructor_from_dict",
        file_path="tests/frame/test_constructors.py",
        class_name=None,
        node_ids=[
            "tests/frame/test_constructors.py::test_constructor_from_dict"
        ],
        reasons=["AssertionError"],
    )
    state = _make_state(group)

    result = graph._pre_filter_reason(group, state)

    # Normal test should NOT be filtered
    assert result is None


def test_deselected_reason_text_still_filtered() -> None:
    """When the reason text explicitly contains deselection language,
    the existing pre-filter correctly catches it. This tests preserves
    the existing behavior for reason-text-based filtering.
    """
    group = TestGroup(
        base_name="tests/test_baz.py::test_qux",
        file_path="tests/test_baz.py",
        class_name=None,
        node_ids=["tests/test_baz.py::test_qux[x]"],
        reasons=["collected 1 item / 1 deselected"],
    )
    state = _make_state(group)

    result = graph._pre_filter_reason(group, state)

    # Existing behavior: explicit deselection text is caught
    assert result == "pre_filtered: deselected"


def test_after_fix_single_cpu_module_is_filtered() -> None:
    """After Bug 3 fix: confirmed _pre_filter_reason returns deselected
    for tests in single_cpu-marked modules.
    """
    group = TestGroup(
        base_name="tests/io/test_sql.py::test_read_sql_delegate",
        file_path="tests/io/test_sql.py",
        class_name=None,
        node_ids=["tests/io/test_sql.py::test_read_sql_delegate"],
        reasons=["AssertionError: DataFrame columns mismatch"],
    )
    state = _make_state(group)

    result = graph._pre_filter_reason(group, state)

    assert result == "pre_filtered: deselected"
