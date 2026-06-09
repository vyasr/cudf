# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from langgraph.types import Send

_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PIPELINE_ROOT.parent
for _path in (str(_PIPELINE_ROOT), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from pandas_compat_pipeline.src.orchestrator import graph
from pandas_compat_pipeline.src.orchestrator.state import PipelineState
from pandas_compat_pipeline.src.utils.patch_validator import validate_patch

_META_KEY = "__graph_meta__"


def test_rejects_out_of_scope_docs_path() -> None:
    result = validate_patch(["docs/source/foo.md"])

    assert not result.approved
    assert "Out-of-scope" in result.reason


def test_rejects_dependency_added_to_existing_pyproject_block() -> None:
    diff = """@@ -10,7 +10,8 @@
 [project]
 dependencies = [
     "numpy>=1.23",
+    "newdep>=1.0",
 ]
 """

    result = validate_patch(["python/cudf/pyproject.toml"], diff_content=diff)

    assert not result.approved
    assert "dependencies" in result.reason


def test_allows_cudf_source_and_xfail_removal_scope() -> None:
    result = validate_patch(
        [
            "python/cudf/cudf/core/series.py",
            "python/cudf/cudf/pandas/scripts/pandas-testing-plugin.py",
        ]
    )

    assert result.approved
    assert result.classification == "both"


def test_route_after_dispatch_returns_parallel_send_objects() -> None:
    state: PipelineState = {
        "in_progress": {
            "task-a": {"status": "assigned", "worker_id": 0},
            "task-b": {"status": "assigned", "worker_id": 1},
            _META_KEY: {
                "current_task": "task-a",
                "current_tasks": ["task-a", "task-b"],
            },
        },
        "completed": [],
        "failed": [],
        "flagged_for_human": [],
        "worker_status": {},
        "pending_tests": [],
        "integration_queue": [],
        "integration_results": [],
        "baseline_results": None,
        "fixes_since_last_integration": 0,
        "total_fixes": 0,
    }

    sends = graph.route_after_dispatch(state)

    assert isinstance(sends, list)
    assert len(sends) == 2
    assert all(isinstance(item, Send) for item in sends)
    assert [item.node for item in sends] == ["fix", "fix"]
    assert [
        item.arg["in_progress"][_META_KEY]["current_task"] for item in sends
    ] == [
        "task-a",
        "task-b",
    ]


def test_setup_worktrees_bootstrap_is_static_and_shell_valid() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "setup-worktrees.sh"
    )
    text = script.read_text(encoding="utf-8")

    assert "ensure_pandas_testing" in text
    assert "run-pandas-tests.sh" in text
    assert "pandas-testing/pandas-tests/tests" in text
    _ = subprocess.run(["bash", "-n", str(script)], check=True)
