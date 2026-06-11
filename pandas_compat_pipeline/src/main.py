# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402, T201

"""Pipeline entry point and CLI for the autonomous pandas test fix pipeline.

Usage:
    python -m src.main --pilot                    # Full pilot run (default 200 groups)
    python -m src.main --pilot --max-groups=50    # Pilot with custom group count
    python -m src.main --baseline-only            # Only run baseline, produce results
    python -m src.main --dry-run                  # Validate prerequisites, do not run
    python -m src.main --status                   # Print current checkpoint state
    python -m src.main --resume                   # Resume from last checkpoint
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any

# Ensure both the pipeline package root and cudf repo root are on sys.path so
# `python src/main.py` works from pandas_compat_pipeline/ while absolute
# imports such as `pandas_compat_pipeline.src...` also resolve.
_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PIPELINE_ROOT.parent
for _path in (str(_PIPELINE_ROOT), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from pandas_compat_pipeline.src.config import PipelineConfig, load_config
from pandas_compat_pipeline.src.orchestrator.graph import build_graph
from pandas_compat_pipeline.src.orchestrator.persistence import (
    get_checkpointer,
)
from pandas_compat_pipeline.src.orchestrator.worker_manager import (
    WorkerManager,
)

logger = logging.getLogger(__name__)

_PIPELINE_DIR = _REPO_ROOT / "pandas_compat_pipeline"
_STATE_FILE = _PIPELINE_DIR / "pipeline_state.json"
_FLAGGED_FILE = _PIPELINE_DIR / "flagged_for_human.json"
_REPORT_FILE = _PIPELINE_DIR / "pilot_report.json"
_BASELINE_FILE = _PIPELINE_DIR / "baseline_results.json"

_shutdown_requested = False
_worker_manager: WorkerManager | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous pandas test fix pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--pilot",
        action="store_true",
        help="Run the full fix pipeline",
    )
    mode.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run baseline test suite only, produce baseline_results.json",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all prerequisites without running any fixes",
    )
    mode.add_argument(
        "--status",
        action="store_true",
        help="Print current pipeline state from checkpoint",
    )
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Resume pipeline from the last checkpoint",
    )

    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Maximum number of test groups to attempt (default: from config)",
    )

    return parser.parse_args()


def _load_pipeline_state() -> dict[str, Any]:
    """Load persisted pipeline state (thread_id, etc.)."""
    if _STATE_FILE.exists():
        return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_pipeline_state(state: dict[str, Any]) -> None:
    """Persist pipeline state (thread_id, etc.)."""
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(
        json.dumps(state, indent=2, default=str), encoding="utf-8"
    )


def _generate_thread_id() -> str:
    """Generate a new thread ID for LangGraph."""
    return f"pilot-{int(time.time())}"


def _setup_logging(config: PipelineConfig) -> None:
    """Configure logging based on config."""
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _check_conda_env() -> tuple[bool, str]:
    """Check if conda env is accessible and cudf can be imported."""
    import subprocess

    result = subprocess.run(
        ["python", "-c", "import cudf; print('ok')"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode == 0:
        return True, "conda env OK (cudf importable)"
    return False, f"cudf import failed: {result.stderr.strip()[:200]}"


def _check_worktrees(config: PipelineConfig) -> tuple[bool, str]:
    """Check that at least worker-0 worktree exists with .gpu marker."""
    base = Path(config.worktree_base_path).expanduser()
    worker0 = base / "worker-0"
    gpu_marker = worker0 / ".gpu"
    if not worker0.exists():
        return False, f"worktree dir not found: {worker0}"
    if not gpu_marker.exists():
        return False, f"GPU marker not found: {gpu_marker}"
    return True, f"worktree OK: {worker0}"


def _check_postgres(config: PipelineConfig) -> tuple[bool, str]:
    """Check TCP connectivity to PostgreSQL."""
    url = config.postgres_url
    # Parse host:port from URL like postgresql://user:pass@host:port/db
    try:
        # Simple parse: after @ and before /
        after_at = url.split("@")[1] if "@" in url else url.split("//")[1]
        host_port = after_at.split("/")[0]
        if ":" in host_port:
            host, port_str = host_port.rsplit(":", 1)
            port = int(port_str)
        else:
            host = host_port
            port = 5432
    except (IndexError, ValueError):
        return False, f"Cannot parse postgres URL: {url}"

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((host, port))
        sock.close()
        return True, f"PostgreSQL reachable at {host}:{port}"
    except (OSError, socket.timeout) as exc:
        return False, f"PostgreSQL not reachable at {host}:{port}: {exc}"


def _check_pandas_testing(config: PipelineConfig) -> tuple[bool, str]:
    """Check that pandas-testing test directory exists in worker-0."""
    base = Path(config.worktree_base_path).expanduser()
    pandas_tests_dir = base / "worker-0" / "pandas-testing" / "pandas-tests"
    if pandas_tests_dir.exists():
        return True, f"pandas-testing OK: {pandas_tests_dir}"
    return False, f"pandas-testing dir not found: {pandas_tests_dir}"


def _run_prerequisites_check(config: PipelineConfig) -> bool:
    """Run all prerequisite checks. Returns True if all pass."""
    checks = [
        ("Conda env (cudf)", _check_conda_env),
        ("Worktrees", lambda: _check_worktrees(config)),
        ("PostgreSQL", lambda: _check_postgres(config)),
        ("pandas-testing", lambda: _check_pandas_testing(config)),
    ]

    all_passed = True
    for name, check_fn in checks:
        try:
            passed, msg = check_fn()
        except Exception as exc:
            passed, msg = False, f"Exception: {exc}"
        status = "✓" if passed else "✗"
        print(f"  {status} {name}: {msg}")
        if not passed:
            all_passed = False

    return all_passed


def _write_flagged_file(flagged: list[Any]) -> None:
    """Write flagged_for_human.json."""
    _FLAGGED_FILE.parent.mkdir(parents=True, exist_ok=True)
    _FLAGGED_FILE.write_text(
        json.dumps(flagged, indent=2, default=str), encoding="utf-8"
    )


def _write_report(state: dict[str, Any]) -> None:
    """Write pilot_report.json with summary."""
    completed = state.get("completed", [])
    failed = state.get("failed", [])
    flagged = state.get("flagged_for_human", [])

    report = {
        "summary": {
            "total_fixes": len(completed),
            "total_failed": len(failed),
            "total_flagged": len(flagged),
            "total_attempted": len(completed) + len(failed) + len(flagged),
        },
        "completed": completed,
        "failed": failed,
        "flagged_for_human": flagged,
    }
    _REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_FILE.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )


def _print_summary(state: dict[str, Any]) -> None:
    """Print a final summary of the pipeline run."""
    completed = state.get("completed", [])
    failed = state.get("failed", [])
    flagged = state.get("flagged_for_human", [])

    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Fixed:   {len(completed)}")
    print(f"  Failed:  {len(failed)}")
    print(f"  Flagged: {len(flagged)}")
    print(f"  Total:   {len(completed) + len(failed) + len(flagged)}")
    print("=" * 60)


def _signal_handler(signum: int, frame: Any) -> None:
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    logger.warning("Received %s, initiating graceful shutdown...", sig_name)
    _shutdown_requested = True


async def _run_status(config: PipelineConfig) -> int:
    """Print current checkpoint state and exit."""
    pipeline_state = _load_pipeline_state()
    thread_id = pipeline_state.get("thread_id")

    if not thread_id:
        print("No pipeline state found (no previous run detected).")
        return 0

    print(f"Thread ID: {thread_id}")

    try:
        checkpointer = await get_checkpointer()
        graph = build_graph(checkpointer=checkpointer)
        graph_config = {"configurable": {"thread_id": thread_id}}
        state = await graph.aget_state(graph_config)

        if state and state.values:
            values = state.values
            completed = values.get("completed", [])
            failed = values.get("failed", [])
            flagged = values.get("flagged_for_human", [])
            pending = values.get("pending_tests", [])

            print(f"  Completed: {len(completed)}")
            print(f"  Failed:    {len(failed)}")
            print(f"  Flagged:   {len(flagged)}")
            print(f"  Pending:   {len(pending)}")
            print(f"  Total fixes: {values.get('total_fixes', 0)}")
        else:
            print("  No checkpoint data found for this thread.")
    except Exception as exc:
        print(f"  Could not read checkpoint: {exc}")
        print("  (Is PostgreSQL running?)")

    return 0


async def _run_baseline_only(config: PipelineConfig) -> int:
    """Run baseline only, save results, exit."""
    from pandas_compat_pipeline.src.orchestrator.baseline import run_baseline

    print("Running baseline test suite...")
    worktree_base = Path(config.worktree_base_path).expanduser()
    worktree_path = worktree_base / "worker-0"

    result = await run_baseline(worktree_path, gpu_id=0, config=config)

    print("\nBaseline complete:")
    print(f"  Total collected: {result.total_collected}")
    print(f"  Passed:          {result.passed}")
    print(f"  Failed:          {result.failed}")
    print(f"  XFailed:         {result.xfailed}")
    print(f"  XPassed:         {result.xpassed}")
    print(f"  Errors:          {result.errors}")
    print(f"  Stale entries:   {len(result.stale_entries)}")
    print(f"  New failures:    {len(result.new_failures)}")
    print(f"\nResults saved to: {_BASELINE_FILE}")
    return 0


async def _run_pilot(
    config: PipelineConfig, max_groups: int | None, resume: bool
) -> int:
    """Run the full pilot pipeline."""
    global _worker_manager, _shutdown_requested

    # Determine thread ID
    pipeline_state = _load_pipeline_state()
    if resume:
        thread_id = pipeline_state.get("thread_id")
        if not thread_id:
            print("ERROR: --resume requested but no previous thread_id found.")
            return 1
        print(f"Resuming pipeline with thread_id: {thread_id}")
    else:
        thread_id = _generate_thread_id()
        print(f"Starting new pipeline run with thread_id: {thread_id}")

    # Override max groups if specified
    if max_groups is not None:
        os.environ["PIPELINE_PILOT_MAX_GROUPS"] = str(max_groups)
        config = load_config()  # Reload to pick up override

    # Save thread_id
    pipeline_state["thread_id"] = thread_id
    pipeline_state["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _save_pipeline_state(pipeline_state)

    # Connect to PostgreSQL and build graph
    print("Connecting to PostgreSQL...")
    try:
        checkpointer = await get_checkpointer()
    except Exception as exc:
        print(f"ERROR: Cannot connect to PostgreSQL: {exc}")
        return 1

    graph = build_graph(checkpointer=checkpointer)
    graph_config = {"configurable": {"thread_id": thread_id}}

    # Start worker manager
    print(f"Starting {config.gpus} workers...")
    _worker_manager = WorkerManager(
        worktree_base=config.worktree_base_path,
        pytest_parallelism=config.pytest_parallelism,
        integration_gpu=config.integration_gpu
        if config.integration_gpu is not None
        else config.gpus - 1,
    )
    _worker_manager.start_workers(num_workers=config.gpus)

    # Initial state for new runs
    initial_state: dict[str, Any] | None = None
    if not resume:
        # Build priority-ordered pending_tests: stale xfails first
        import json as _json

        _baseline_path = _PIPELINE_DIR / "baseline_results.json"
        _pending_ordered: list[str] = []
        try:
            from pandas_compat_pipeline.src.utils.xfail_parser import (
                parse_xfail_list,
            )

            _all_groups = parse_xfail_list()
            _all_names = [g.base_name for g in _all_groups]
            if _baseline_path.exists():
                _baseline = _json.loads(_baseline_path.read_text())
                _stale = set(_baseline.get("stale_entries", []))
                _fully_stale = [
                    g.base_name
                    for g in _all_groups
                    if _stale and all(nid in _stale for nid in g.node_ids)
                ]
                _not_stale = [
                    n for n in _all_names if n not in set(_fully_stale)
                ]
                _pending_ordered = _fully_stale + _not_stale
            else:
                _pending_ordered = _all_names
        except Exception as _e:
            logger.warning("Could not build priority pending_tests: %s", _e)
            _pending_ordered = []

        initial_state = {
            "pending_tests": _pending_ordered,
            "in_progress": {},
            "completed": [],
            "failed": [],
            "flagged_for_human": [],
            "integration_queue": [],
            "integration_results": [],
            "baseline_results": _json.loads(_baseline_path.read_text())
            if _baseline_path.exists()
            else None,
            "fixes_since_last_integration": 0,
            "total_fixes": 0,
            "worker_status": _worker_manager.get_worker_status(),
        }

    # Run the graph
    print("Starting pipeline execution...")
    final_state: dict[str, Any] = {}
    try:
        if initial_state is not None:
            async for event in graph.astream(
                initial_state, config=graph_config
            ):
                if _shutdown_requested:
                    break
                # Update final state from streamed events
                for key, value in event.items():
                    if key != "__end__":
                        final_state.update(
                            value if isinstance(value, dict) else {key: value}
                        )
        else:
            # Resume: pass None to continue from checkpoint
            async for event in graph.astream(None, config=graph_config):
                if _shutdown_requested:
                    break
                for key, value in event.items():
                    if key != "__end__":
                        final_state.update(
                            value if isinstance(value, dict) else {key: value}
                        )

    except Exception as exc:
        logger.error("Pipeline error: %s", exc, exc_info=True)
        print(f"\nPipeline error: {exc}")
    finally:
        # Graceful shutdown
        print("\nStopping workers...")
        if _worker_manager is not None:
            _worker_manager.stop_workers()
            _worker_manager = None

    # Try to get final state from checkpoint
    try:
        checkpoint_state = await graph.aget_state(graph_config)
        if checkpoint_state and checkpoint_state.values:
            final_state = dict(checkpoint_state.values)
    except Exception:
        pass

    # Write output files
    flagged = final_state.get("flagged_for_human", [])
    _write_flagged_file(flagged)
    _write_report(final_state)
    _print_summary(final_state)

    # Update pipeline state
    pipeline_state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    pipeline_state["total_fixes"] = final_state.get("total_fixes", 0)
    _save_pipeline_state(pipeline_state)

    print("\nOutput files:")
    print(f"  Report:  {_REPORT_FILE}")
    print(f"  Flagged: {_FLAGGED_FILE}")

    return 0


async def _async_main() -> int:
    """Async entry point."""
    args = _parse_args()
    config = load_config()
    _setup_logging(config)

    # Install signal handlers
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if args.dry_run:
        print("Dry-run: checking prerequisites...\n")
        all_ok = _run_prerequisites_check(config)
        if all_ok:
            print("\nAll prerequisites satisfied. Ready to run.")
            return 0
        else:
            print("\nSome prerequisites failed. Fix them before running.")
            return 1

    if args.status:
        return await _run_status(config)

    if args.baseline_only:
        # Check prerequisites first
        print("Checking prerequisites...\n")
        if not _run_prerequisites_check(config):
            print("\nPrerequisites check failed. Exiting.")
            return 1
        print()
        return await _run_baseline_only(config)

    if args.pilot or args.resume:
        # Check prerequisites first
        print("Checking prerequisites...\n")
        if not _run_prerequisites_check(config):
            print("\nPrerequisites check failed. Exiting.")
            return 1
        print()
        return await _run_pilot(config, args.max_groups, resume=args.resume)

    # Should not reach here due to mutually exclusive group
    return 1


def main() -> int:
    """Synchronous entry point."""
    return asyncio.run(_async_main())


if __name__ == "__main__":
    sys.exit(main())
