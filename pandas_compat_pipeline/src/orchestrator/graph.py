# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangGraph graph assembly for the pandas compatibility fix pipeline."""

from __future__ import annotations

import asyncio
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from ..agents.fixer import FixerAgent
from ..agents.integration_tester import IntegrationTesterAgent
from ..agents.regression_debugger import RegressionDebuggerAgent
from ..agents.reviewer import ReviewerAgent
from ..config import PipelineConfig, load_config
from ..orchestrator.baseline import run_baseline
from ..orchestrator.dispatcher import Dispatcher
from ..orchestrator.state import PipelineState
from ..utils.models import TestGroup
from ..utils.test_runner import TestOutcome, run_test
from ..utils.xfail_parser import parse_xfail_list

_META_KEY = "__graph_meta__"


def baseline(state: PipelineState) -> dict[str, Any]:
    """Run baseline test suite to establish initial state."""
    if state.get("baseline_results") is not None:
        return {"baseline_results": state["baseline_results"]}

    config = load_config()
    worker_id, worker = _first_worker(state, config)
    worktree_path = _worktree_path(worker, config, worker_id)
    gpu_id = _gpu_id(worker_id, worker)

    result = _run_async(run_baseline(worktree_path, gpu_id, config=config))
    for stale_entry in result.stale_entries:
        print(f"stale xfail entry: {stale_entry}")
    return {"baseline_results": _to_state_value(result)}


def dispatch(state: PipelineState) -> dict[str, Any]:
    """Assign pending test groups to available workers."""
    config = load_config()
    meta = _meta(state)
    if _pilot_limit_reached(state, config):
        return {"in_progress": {_META_KEY: {**meta, "pipeline_done": True}}}

    retry_task = meta.get("retry_task")
    if isinstance(retry_task, str) and isinstance(
        state.get("in_progress", {}).get(retry_task), dict
    ):
        assignment = dict(state["in_progress"][retry_task])
        assignment["status"] = "assigned"
        return {
            "in_progress": {
                retry_task: assignment,
                _META_KEY: {
                    **meta,
                    "current_task": retry_task,
                    "retry_task": None,
                },
            }
        }

    queue = _queue_from_state(state, config)
    active_names = _active_task_names(state)
    completed_names = _completed_task_names(state)
    failed_names = _terminal_task_names(state.get("failed", []))
    flagged_names = _terminal_task_names(state.get("flagged_for_human", []))

    assignments: dict[str, Any] = {}
    assigned_names: list[str] = []
    remaining: list[str] = []
    if not queue:
        done = not active_names
        return {
            "in_progress": {
                _META_KEY: {**meta, "queue": remaining, "pipeline_done": done}
            }
        }

    groups_by_name = _groups_by_name(state, config)
    dispatcher = Dispatcher(
        [groups_by_name[name] for name in queue if name in groups_by_name]
    )

    terminal_names = completed_names | failed_names | flagged_names
    excluded_names = active_names | terminal_names
    terminal_count = (
        len(completed_names) + len(failed_names) + len(flagged_names)
    )
    capacity = max(
        0, config.pilot_max_groups - terminal_count - len(active_names)
    )
    available_workers = _available_workers(state, config)[:capacity]

    for worker_id, worker in available_workers:
        while True:
            candidate = _run_async(dispatcher.get_next_task(worker_id))
            if candidate is None:
                break
            if candidate.base_name in excluded_names:
                continue
            assignment = _assignment_for_task(
                worker_id, worker, config, candidate
            )
            assignments[candidate.base_name] = assignment
            assigned_names.append(candidate.base_name)
            excluded_names.add(candidate.base_name)
            break

    stats_pending = dispatcher.get_pending_names()
    remaining = [
        name
        for name in stats_pending
        if name not in terminal_names | set(assigned_names)
    ]

    if not assignments:
        done = not active_names
        return {
            "in_progress": {
                _META_KEY: {**meta, "queue": remaining, "pipeline_done": done}
            }
        }

    return {
        "in_progress": {
            **assignments,
            _META_KEY: {
                **meta,
                "queue": remaining,
                "current_task": assigned_names[0],
                "current_tasks": assigned_names,
                "pipeline_done": False,
            },
        }
    }


def fix(state: PipelineState) -> dict[str, Any]:
    """Invoke fixer agent to generate a patch for the current test group."""
    config = load_config()
    task_name, assignment = _current_assignment(state)
    test_group = _test_group(assignment)

    # Pre-filter: route unfixable groups directly to failed without LLM call
    pre_filter_reason = _pre_filter_reason(test_group, state)
    if pre_filter_reason is not None:
        _run_async(
            Dispatcher([]).fail_task(
                int(assignment.get("worker_id", 0)),
                task_name,
                pre_filter_reason,
            )
        )
        return {
            "in_progress": {task_name: {**assignment, "status": "failed"}},
            "failed": [{"test_name": task_name, "reason": pre_filter_reason}],
        }

    attempts = int(assignment.get("attempts", 0)) + 1
    result = _run_async(
        FixerAgent(config=config).fix(
            test_group,
            str(assignment["worktree_path"]),
            int(assignment["gpu_id"]),
        )
    )
    result_payload = _to_state_value(result)
    updated = {
        **assignment,
        "attempts": attempts,
        "fix_result": result_payload,
        "status": result.status,
    }
    meta = _meta(state)
    if result.status == "failed" and attempts < config.max_fix_attempts:
        meta = {**meta, "retry_task": task_name}
    return {"in_progress": {task_name: updated, _META_KEY: meta}}


def review(state: PipelineState) -> dict[str, Any]:
    """Invoke reviewer agent to evaluate the proposed fix."""
    task_name, assignment = _current_assignment(state)
    fix_result = cast(dict[str, Any], assignment.get("fix_result", {}))
    worktree_path = str(assignment["worktree_path"])
    diff = _git_diff_for_review(worktree_path)
    result = _run_async(
        ReviewerAgent().review(
            diff=diff,
            test_node_id=_primary_node_id(assignment),
            diagnosis=str(fix_result.get("diagnosis", "")),
            worktree_path=worktree_path,
        )
    )
    return {
        "in_progress": {
            task_name: {
                **assignment,
                "review_result": _to_state_value(result),
                "status": "reviewed",
            }
        }
    }


def verify(state: PipelineState) -> dict[str, Any]:
    """Run the fixed test to verify correctness and fallback behavior."""
    config = load_config()
    task_name, assignment = _current_assignment(state)
    worktree_path = str(assignment["worktree_path"])
    gpu_id = int(assignment["gpu_id"])
    test_group = _test_group(assignment)

    strict_results = [
        _to_state_value(
            run_test(
                worktree_path,
                node_id,
                gpu_id,
                run_mode="verify",
                fail_on_fallback=False,
                timeout_minutes=config.single_test_timeout_minutes,
                flakiness_reruns=config.flakiness_reruns,
            )
        )
        for node_id in test_group.node_ids
    ]
    fallback_results = [
        _to_state_value(
            run_test(
                worktree_path,
                node_id,
                gpu_id,
                run_mode="verify",
                fail_on_fallback=True,
                timeout_minutes=config.single_test_timeout_minutes,
                flakiness_reruns=config.flakiness_reruns,
            )
        )
        for node_id in test_group.node_ids
    ]
    sanity = _run_import_sanity(worktree_path, gpu_id)
    verification = {
        "strict_passed": _all_passed(strict_results),
        "no_fallback_passed": _all_passed(fallback_results),
        "sanity_passed": sanity["passed"],
        "strict_results": strict_results,
        "fallback_results": fallback_results,
        "sanity": sanity,
    }
    verification["passed"] = bool(
        verification["strict_passed"]
        and verification["no_fallback_passed"]
        and verification["sanity_passed"]
    )
    return {
        "in_progress": {
            task_name: {
                **assignment,
                "verification": verification,
                "status": "verified",
            }
        }
    }


def commit(state: PipelineState) -> dict[str, Any]:
    """Commit the verified fix and record completion metadata."""
    task_name, assignment = _current_assignment(state)
    worktree_path = str(assignment["worktree_path"])
    message = f"fix(pandas-compat): {task_name}"
    commit_result = _git_commit(worktree_path, message)
    payload = {
        "test_name": task_name,
        "branch": assignment.get("fix_result", {}).get("branch_name"),
        "worker_id": assignment.get("worker_id"),
        "commit": commit_result,
    }
    _run_async(
        Dispatcher([]).complete_task(
            int(assignment.get("worker_id", 0)), task_name, payload
        )
    )
    return {
        "in_progress": {task_name: {**assignment, "status": "completed"}},
        "completed": [payload],
        "fixes_since_last_integration": int(
            state.get("fixes_since_last_integration", 0)
        )
        + 1,
        "total_fixes": int(state.get("total_fixes", 0)) + 1,
    }


def check_integration_trigger(state: PipelineState) -> dict[str, Any]:
    """Decide whether to run an integration test based on fix count."""
    config = load_config()
    count = int(state.get("fixes_since_last_integration", 0))
    meta = _meta(state)
    if count < config.integration_trigger_every_n:
        return {"in_progress": {_META_KEY: {**meta, "run_integration": False}}}
    branches = _recent_completed_branches(
        state, config.integration_trigger_every_n
    )
    return {
        "integration_queue": branches,
        "fixes_since_last_integration": 0,
        "in_progress": {
            _META_KEY: {**meta, "run_integration": bool(branches)}
        },
    }


def integration_test(state: PipelineState) -> dict[str, Any]:
    """Run full integration test suite on merged branches."""
    config = load_config()
    meta = _meta(state)
    batch = _integration_batch(state, config)
    worker_id, worker = _first_worker(state, config)
    worktree_path = _worktree_path(worker, config, worker_id)
    gpu_id = (
        config.integration_gpu
        if config.integration_gpu is not None
        else _gpu_id(worker_id, worker)
    )
    result = _run_async(
        IntegrationTesterAgent(config=config).test(
            batch,
            worktree_path,
            int(gpu_id),
            len(state.get("integration_results", [])) + 1,
            _baseline_dict(state),
        )
    )
    payload = _to_state_value(result)
    return {
        "integration_results": [payload],
        "in_progress": {
            _META_KEY: {**meta, "last_integration_result": payload}
        },
    }


def regression_debug(state: PipelineState) -> dict[str, Any]:
    """Diagnose regressions found during integration testing."""
    config = load_config()
    result = _latest_integration_result(state)
    worker_id, worker = _first_worker(state, config)
    worktree_path = _worktree_path(worker, config)
    debug_result = _run_async(
        RegressionDebuggerAgent(config=config).debug(
            regressions=list(result.get("regressions", [])),
            batch_branches=list(result.get("batch_branches", [])),
            integration_branch=str(result.get("integration_branch", "")),
            worktree_path=worktree_path,
        )
    )
    payload = _to_state_value(debug_result)
    queue_additions: list[str] = []
    for item in payload.get("requeue_tasks", []):
        test_name = item.get("test_group")
        group = _groups_by_name(state, config).get(test_name)
        if group is not None:
            _run_async(
                Dispatcher([]).requeue_task(
                    group, item.get("priority", "high")
                )
            )
            queue_additions.append(group.base_name)
    meta = _meta(state)
    queue = list(meta.get("queue", []))
    return {
        "in_progress": {
            _META_KEY: {
                **meta,
                "queue": queue_additions + queue,
                "regression_result": payload,
            }
        }
    }


def fail_node(state: PipelineState) -> dict[str, Any]:
    """Mark a test group as failed (exhausted attempts)."""
    task_name, assignment = _current_assignment(state)
    reason = str(
        assignment.get("fix_result", {}).get(
            "rejection_reason", "max attempts exhausted"
        )
    )
    _run_async(
        Dispatcher([]).fail_task(
            int(assignment.get("worker_id", 0)), task_name, reason
        )
    )
    return {
        "in_progress": {task_name: {**assignment, "status": "failed"}},
        "failed": [{"test_name": task_name, "reason": reason}],
    }


def flag_node(state: PipelineState) -> dict[str, Any]:
    """Flag a test group for human review (needs C++ or unresolvable)."""
    task_name, assignment = _current_assignment(state)
    reason = str(
        assignment.get("fix_result", {}).get(
            "rejection_reason", "human review required"
        )
    )
    _run_async(Dispatcher([]).requeue_flagged(task_name, reason))
    return {
        "in_progress": {
            task_name: {**assignment, "status": "flagged_for_human"}
        },
        "flagged_for_human": [{"test_name": task_name, "reason": reason}],
    }


def route_after_dispatch(state: PipelineState) -> list[Send] | Literal["end"]:
    """Fan out assigned workers with LangGraph Send, or stop when no work remains."""
    meta = _meta(state)
    if meta.get("pipeline_done"):
        return "end"
    assignments = _task_assignments(state)
    task_names = [str(name) for name in meta.get("current_tasks", [])]
    if not task_names and isinstance(meta.get("current_task"), str):
        task_names = [str(meta["current_task"])]

    sends = [
        Send("fix", _state_for_task(state, task_name, assignments[task_name]))
        for task_name in task_names
        if task_name in assignments
        and str(assignments[task_name].get("status", "")).lower() == "assigned"
    ]
    return sends if sends else "end"


def route_after_fix(
    state: PipelineState,
) -> Literal["review", "dispatch", "fail", "flag"]:
    """Route after fix attempt: review if success, retry/dispatch, fail, or flag."""
    _, assignment = _current_assignment(state)
    fix_result = cast(dict[str, Any], assignment.get("fix_result", {}))
    status = fix_result.get("status")
    if status == "success":
        return "review"
    if status == "flagged_for_human":
        return "flag"
    if int(assignment.get("attempts", 0)) < load_config().max_fix_attempts:
        return "dispatch"
    return "fail"


def route_after_review(
    state: PipelineState,
) -> Literal["verify", "fix", "flag"]:
    """Route after review: verify if approved, fix if rejected, flag if needs human."""
    _, assignment = _current_assignment(state)
    verdict = cast(dict[str, Any], assignment.get("review_result", {})).get(
        "verdict"
    )
    if verdict == "approved":
        return "verify"
    if verdict == "rejected":
        return "fix"
    return "flag"


def route_after_verify(state: PipelineState) -> Literal["commit", "fix"]:
    """Route after verification: commit if pass, fix again if fail."""
    _, assignment = _current_assignment(state)
    verification = cast(dict[str, Any], assignment.get("verification", {}))
    return "commit" if verification.get("passed") is True else "fix"


def route_after_integration(
    state: PipelineState,
) -> Literal["regression_debug", "dispatch"]:
    """Route after integration test: debug regressions or continue dispatching."""
    result = _latest_integration_result(state)
    return "regression_debug" if result.get("regressions") else "dispatch"


def route_after_check_trigger(
    state: PipelineState,
) -> Literal["integration_test", "dispatch"]:
    """Route after checking trigger: run integration or continue."""
    return (
        "integration_test"
        if _meta(state).get("run_integration")
        else "dispatch"
    )


def build_graph(checkpointer: Any | None = None) -> Any:
    """Build and compile the LangGraph state graph.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. AsyncPostgresSaver).
                      If None, graph runs without persistence.

    Returns:
        Compiled LangGraph graph ready for invocation.
    """
    builder = StateGraph(PipelineState)

    builder.add_node("baseline", baseline)
    builder.add_node("dispatch", dispatch)
    builder.add_node("fix", fix)
    builder.add_node("review", review)
    builder.add_node("verify", verify)
    builder.add_node("commit", commit)
    builder.add_node("check_integration_trigger", check_integration_trigger)
    builder.add_node("integration_test", integration_test)
    builder.add_node("regression_debug", regression_debug)
    builder.add_node("fail", fail_node)
    builder.add_node("flag", flag_node)

    builder.set_entry_point("baseline")

    builder.add_edge("baseline", "dispatch")

    builder.add_conditional_edges(
        "dispatch",
        route_after_dispatch,
        {"end": END},
    )

    builder.add_conditional_edges(
        "fix",
        route_after_fix,
        {
            "review": "review",
            "dispatch": "dispatch",
            "fail": "fail",
            "flag": "flag",
        },
    )

    builder.add_conditional_edges(
        "review",
        route_after_review,
        {"verify": "verify", "fix": "fix", "flag": "flag"},
    )

    builder.add_conditional_edges(
        "verify",
        route_after_verify,
        {"commit": "commit", "fix": "fix"},
    )

    builder.add_edge("commit", "check_integration_trigger")

    builder.add_conditional_edges(
        "check_integration_trigger",
        route_after_check_trigger,
        {"integration_test": "integration_test", "dispatch": "dispatch"},
    )

    builder.add_conditional_edges(
        "integration_test",
        route_after_integration,
        {"regression_debug": "regression_debug", "dispatch": "dispatch"},
    )

    builder.add_edge("regression_debug", "dispatch")

    builder.add_edge("fail", "dispatch")
    builder.add_edge("flag", "dispatch")

    return builder.compile(checkpointer=checkpointer)


def _run_async(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, awaitable).result()


def _repo_root() -> str:
    return str(Path(__file__).resolve().parents[3])


_COLLECTION_FAILURE_PATTERNS = (
    "not found",
    "no match in any of",
    "found no collectors",
    "collected 0 items",
    "ERROR: not found",
    "no tests ran",
)

_DESELECTED_PATTERNS = (
    "deselected / 0 selected",
    "1 deselected",
    "0 selected",
    "collected 1 item / 1 deselected",
)


def _pre_filter_reason(
    test_group: TestGroup, state: PipelineState
) -> str | None:
    """Return a pre-filter reason if the group is unfixable, else None."""
    # Check baseline stale entries: if ALL node_ids are stale, it's a collection failure
    baseline = state.get("baseline_results")
    if isinstance(baseline, dict):
        stale = set(baseline.get("stale_entries", []))
        if stale and all(nid in stale for nid in test_group.node_ids):
            return "pre_filtered: collection_failure"

    # Check reasons from the xfail plugin for collection/deselection patterns
    reasons_text = " ".join(test_group.reasons).lower()
    for pattern in _COLLECTION_FAILURE_PATTERNS:
        if pattern.lower() in reasons_text:
            return "pre_filtered: collection_failure"
    for pattern in _DESELECTED_PATTERNS:
        if pattern.lower() in reasons_text:
            return "pre_filtered: deselected"

    return None


def _meta(state: PipelineState) -> dict[str, Any]:
    raw = state.get("in_progress", {}).get(_META_KEY, {})
    return dict(raw) if isinstance(raw, dict) else {}


def _first_worker(
    state: PipelineState, config: PipelineConfig
) -> tuple[int, dict[str, Any]]:
    statuses = state.get("worker_status", {})
    if statuses:
        key = sorted(statuses, key=lambda value: int(value))[0]
        worker = statuses[key]
        return int(key), dict(worker) if isinstance(worker, dict) else {}
    return 0, {
        "gpu_id": 0,
        "worktree_path": str(
            Path(config.worktree_base_path).expanduser() / "worker-0"
        ),
        "state": "idle",
    }


def _first_available_worker(
    state: PipelineState, config: PipelineConfig
) -> tuple[int, dict[str, Any]]:
    statuses = state.get("worker_status", {})
    if not statuses:
        return _first_worker(state, config)
    busy = {
        int(item.get("worker_id", -1))
        for item in _task_assignments(state).values()
    }
    for key in sorted(statuses, key=lambda value: int(value)):
        worker = statuses[key]
        worker_dict = dict(worker) if isinstance(worker, dict) else {}
        status = str(
            worker_dict.get("state", worker_dict.get("status", "idle"))
        ).lower()
        if int(key) not in busy and status in {
            "idle",
            "ready",
            "available",
            "stopped",
        }:
            return int(key), worker_dict
    return _first_worker(state, config)


def _available_workers(
    state: PipelineState, config: PipelineConfig
) -> list[tuple[int, dict[str, Any]]]:
    statuses = state.get("worker_status", {})
    if not statuses:
        return [_first_worker(state, config)]

    busy = {
        int(item.get("worker_id", -1))
        for item in _task_assignments(state).values()
    }
    workers: list[tuple[int, dict[str, Any]]] = []
    for key in sorted(statuses, key=lambda value: int(value)):
        worker = statuses[key]
        worker_dict = dict(worker) if isinstance(worker, dict) else {}
        status = str(
            worker_dict.get("state", worker_dict.get("status", "idle"))
        ).lower()
        worker_id = int(key)
        if worker_id not in busy and status in {
            "idle",
            "ready",
            "available",
            "stopped",
        }:
            workers.append((worker_id, worker_dict))
    return workers or [_first_worker(state, config)]


def _assignment_for_task(
    worker_id: int,
    worker: dict[str, Any],
    config: PipelineConfig,
    test_group: TestGroup,
) -> dict[str, Any]:
    return {
        "worker_id": worker_id,
        "gpu_id": _gpu_id(worker_id, worker),
        "worktree_path": _worktree_path(worker, config, worker_id),
        "test_group": asdict(test_group),
        "status": "assigned",
        "attempts": 0,
    }


def _state_for_task(
    state: PipelineState,
    task_name: str,
    assignment: dict[str, Any],
) -> dict[str, Any]:
    meta = _meta(state)
    return {
        **state,
        "in_progress": {
            task_name: assignment,
            _META_KEY: {
                **meta,
                "current_task": task_name,
                "current_tasks": [task_name],
            },
        },
    }


def _worktree_path(
    worker: dict[str, Any], config: PipelineConfig, worker_id: int = 0
) -> str:
    path = (
        worker.get("worktree_path")
        or worker.get("worktree")
        or worker.get("path")
    )
    if path:
        return str(Path(str(path)).expanduser())
    return str(
        Path(config.worktree_base_path).expanduser() / f"worker-{worker_id}"
    )


def _gpu_id(worker_id: int, worker: dict[str, Any]) -> int:
    return int(worker.get("gpu_id", worker.get("gpu", worker_id)))


def _queue_from_state(
    state: PipelineState, config: PipelineConfig
) -> list[str]:
    meta_queue = _meta(state).get("queue")
    if isinstance(meta_queue, list):
        return [str(item) for item in meta_queue]
    if "pending_tests" in state:
        pending = state.get("pending_tests", [])
        if pending:
            return list(dict.fromkeys(str(item) for item in pending))[
                : config.pilot_max_groups
            ]
    return [
        group.base_name
        for group in _xfail_node_groups(state, config)[
            : config.pilot_max_groups
        ]
    ]


def _groups_by_name(
    state: PipelineState, config: PipelineConfig
) -> dict[str, TestGroup]:
    return {
        group.base_name: group for group in _xfail_node_groups(state, config)
    }


def _xfail_node_groups(
    state: PipelineState, config: PipelineConfig
) -> list[TestGroup]:
    worker_id, worker = _first_worker(state, config)
    plugin_path = (
        Path(_worktree_path(worker, config, worker_id)) / config.plugin_path
    )
    if plugin_path.exists():
        return parse_xfail_list(plugin_path)
    return parse_xfail_list()


def _task_assignments(state: PipelineState) -> dict[str, dict[str, Any]]:
    assignments: dict[str, dict[str, Any]] = {}
    for key, value in state.get("in_progress", {}).items():
        if key == _META_KEY or not isinstance(value, dict):
            continue
        assignments[str(key)] = dict(value)
    return assignments


def _active_task_names(state: PipelineState) -> set[str]:
    terminal = {"completed", "failed", "flagged_for_human"}
    return {
        name
        for name, assignment in _task_assignments(state).items()
        if str(assignment.get("status", "")).lower() not in terminal
    }


def _completed_task_names(state: PipelineState) -> set[str]:
    return _terminal_task_names(state.get("completed", [])) | {
        name
        for name, assignment in _task_assignments(state).items()
        if assignment.get("status") == "completed"
    }


def _terminal_task_names(items: list[Any]) -> set[str]:
    names: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            name = item.get("test_name") or item.get("name")
            if isinstance(name, str):
                names.add(name)
        elif isinstance(item, str):
            names.add(item)
    return names


def _current_assignment(state: PipelineState) -> tuple[str, dict[str, Any]]:
    meta = _meta(state)
    current = meta.get("current_task")
    assignments = _task_assignments(state)
    if isinstance(current, str) and current in assignments:
        return current, assignments[current]
    if assignments:
        return next(iter(assignments.items()))
    raise RuntimeError("No current task assignment in graph state")


def _test_group(assignment: dict[str, Any]) -> TestGroup:
    raw = assignment.get("test_group")
    if isinstance(raw, TestGroup):
        return raw
    if isinstance(raw, dict):
        return TestGroup(**raw)
    raise RuntimeError(
        "Current assignment does not include a test_group payload"
    )


def _primary_node_id(assignment: dict[str, Any]) -> str:
    group = _test_group(assignment)
    return group.node_ids[0] if group.node_ids else group.base_name


def _to_state_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            key: _to_state_value(item) for key, item in asdict(value).items()
        }
    if isinstance(value, dict):
        return {str(key): _to_state_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_state_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _git_diff_for_review(worktree_path: str) -> str:
    diff = subprocess.run(
        ["git", "diff", "--no-ext-diff", "HEAD"],
        cwd=worktree_path,
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    if diff.strip():
        return diff
    return subprocess.run(
        ["git", "show", "--format=", "--no-ext-diff", "HEAD"],
        cwd=worktree_path,
        check=False,
        capture_output=True,
        text=True,
    ).stdout


def _all_passed(results: list[dict[str, Any]]) -> bool:
    return bool(results) and all(
        result.get("outcome") == TestOutcome.PASSED.value for result in results
    )


def _run_import_sanity(worktree_path: str, gpu_id: int) -> dict[str, Any]:
    env = {**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    proc = subprocess.run(
        ["python", "-c", "import cudf; import pandas as pd; print('ok')"],
        cwd=worktree_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return {
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-2000:],
    }


def _git_commit(worktree_path: str, message: str) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=worktree_path,
        check=False,
        capture_output=True,
        text=True,
    )
    if not status.stdout.strip():
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=worktree_path,
            check=False,
            capture_output=True,
            text=True,
        )
        return {
            "committed": False,
            "returncode": 0,
            "stdout": head.stdout.strip(),
            "stderr": "",
        }
    command = f"git add -A && git commit -m {message!r}"
    proc = subprocess.run(
        ["bash", "-lc", command],
        cwd=worktree_path,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "committed": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-2000:],
    }


def _recent_completed_branches(state: PipelineState, limit: int) -> list[str]:
    branches: list[str] = []
    for item in reversed(state.get("completed", [])):
        if isinstance(item, dict) and isinstance(item.get("branch"), str):
            branches.append(item["branch"])
        if len(branches) >= limit:
            break
    return list(reversed(branches))


def _integration_batch(
    state: PipelineState, config: PipelineConfig
) -> list[str]:
    queue = list(state.get("integration_queue", []))
    if not queue:
        queue = _recent_completed_branches(
            state, config.integration_trigger_every_n
        )
    return IntegrationTesterAgent(config=config).select_batch(queue)


def _baseline_dict(state: PipelineState) -> dict[str, Any]:
    value = state.get("baseline_results")
    return dict(value) if isinstance(value, dict) else {}


def _latest_integration_result(state: PipelineState) -> dict[str, Any]:
    meta_result = _meta(state).get("last_integration_result")
    if isinstance(meta_result, dict):
        return meta_result
    results = state.get("integration_results", [])
    if results and isinstance(results[-1], dict):
        return dict(results[-1])
    return {}


def _pilot_limit_reached(state: PipelineState, config: PipelineConfig) -> bool:
    return (
        int(state.get("total_fixes", 0))
        + len(state.get("failed", []))
        + len(state.get("flagged_for_human", []))
        >= config.pilot_max_groups
    )
