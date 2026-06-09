# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker subprocess manager for the pandas compatibility fix pipeline.

Manages N worker subprocesses (one per GPU), each operating in its own
git worktree with strict GPU isolation via CUDA_VISIBLE_DEVICES.

Workers are true OS processes (multiprocessing) — not asyncio coroutines —
to ensure real GPU isolation and fault containment.
"""

from __future__ import annotations

import logging
import multiprocessing
import multiprocessing.synchronize
import os
import signal
import time
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import Process, Queue
from typing import Any

logger = logging.getLogger(__name__)


class WorkerState(str, Enum):
    """Lifecycle states for a worker subprocess."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    CRASHED = "crashed"
    STOPPED = "stopped"


@dataclass
class WorkerInfo:
    """Metadata and runtime state for a single worker subprocess."""

    worker_id: int
    gpu_id: int
    worktree_path: str
    process: Process | None = None
    state: WorkerState = WorkerState.STOPPED
    consecutive_failures: int = 0
    max_consecutive_failures: int = 3
    pytest_parallelism: int = 16
    oom_count: int = 0


@dataclass
class WorkerResult:
    """Result reported by a worker subprocess back to the orchestrator."""

    worker_id: int
    test_name: str
    success: bool
    error: str | None = None
    is_oom: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


# Sentinel value to signal worker shutdown
_SHUTDOWN_SENTINEL = "__SHUTDOWN__"
_PAUSE_SENTINEL = "__PAUSE__"
_RESUME_SENTINEL = "__RESUME__"


def _worker_loop(
    worker_id: int,
    gpu_id: int,
    worktree_path: str,
    task_queue: "Queue[Any]",
    result_queue: "Queue[WorkerResult]",
    pause_event: "multiprocessing.synchronize.Event",
    pytest_parallelism: int,
) -> None:
    """Main loop for a worker subprocess.

    Runs in a separate process with CUDA_VISIBLE_DEVICES set to the
    assigned GPU. Waits for tasks from task_queue, executes fix cycles,
    and reports results to result_queue.

    Parameters
    ----------
    worker_id : int
        Unique identifier for this worker.
    gpu_id : int
        GPU index assigned to this worker.
    worktree_path : str
        Path to the git worktree for this worker.
    task_queue : Queue
        Queue from which this worker receives task assignments.
    result_queue : Queue
        Queue to which this worker reports results.
    pause_event : multiprocessing.Event
        Event that is cleared when the worker should pause.
        When set, the worker is allowed to accept new tasks.
    pytest_parallelism : int
        Number of parallel pytest workers for this subprocess.
    """
    # Set GPU isolation
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["WORKER_ID"] = str(worker_id)
    os.environ["WORKTREE_PATH"] = worktree_path
    os.environ["PYTEST_PARALLELISM"] = str(pytest_parallelism)

    logger.info(
        "Worker %d started: GPU=%d, worktree=%s, parallelism=%d",
        worker_id,
        gpu_id,
        worktree_path,
        pytest_parallelism,
    )

    while True:
        # Check if paused — wait until resumed
        if not pause_event.is_set():
            logger.info(
                "Worker %d paused, waiting for resume signal", worker_id
            )
            pause_event.wait()
            logger.info("Worker %d resumed", worker_id)

        # Wait for next task (blocking with timeout for pause checks)
        try:
            task = task_queue.get(timeout=1.0)
        except Exception:
            # Timeout or empty queue — loop back to check pause/shutdown
            continue

        # Handle control messages
        if task == _SHUTDOWN_SENTINEL:
            logger.info("Worker %d received shutdown signal", worker_id)
            break
        if task == _PAUSE_SENTINEL:
            pause_event.clear()
            continue
        if task == _RESUME_SENTINEL:
            pause_event.set()
            continue

        # Execute the fix cycle for this task
        test_name = task.get("test_name", "unknown")
        logger.info("Worker %d processing task: %s", worker_id, test_name)

        try:
            # Import and run the fix cycle. The actual agent APIs are async,
            # so we run them in an event loop within this subprocess.
            import asyncio as _asyncio

            from pandas_compat_pipeline.src.agents.fixer import (
                FixerAgent,  # pyright: ignore[reportMissingImports]
            )
            from pandas_compat_pipeline.src.config import (
                load_config,  # pyright: ignore[reportMissingImports]
            )
            from pandas_compat_pipeline.src.utils.models import (
                TestGroup,  # type: ignore[import-not-found]
            )

            config = load_config()
            # Reconstruct TestGroup from task dict
            test_group = (
                TestGroup(**task["test_group"])
                if "test_group" in task
                else None
            )

            fixer = FixerAgent(config=config)
            if test_group is not None:
                fix_result = _asyncio.run(
                    fixer.fix(test_group, worktree_path, gpu_id)
                )
                success = fix_result.status == "success"
                result = WorkerResult(
                    worker_id=worker_id,
                    test_name=test_name,
                    success=success,
                    error=fix_result.rejection_reason if not success else None,
                    metadata={
                        "branch": fix_result.branch_name,
                        "modified_files": fix_result.modified_files,
                        "attempts": fix_result.attempts,
                    },
                )
            else:
                result = WorkerResult(
                    worker_id=worker_id,
                    test_name=test_name,
                    success=False,
                    error="No test_group in task payload",
                )

        except MemoryError:
            logger.error("Worker %d OOM on task: %s", worker_id, test_name)
            result = WorkerResult(
                worker_id=worker_id,
                test_name=test_name,
                success=False,
                error="Out of memory",
                is_oom=True,
            )
        except Exception as exc:
            logger.exception(
                "Worker %d error on task: %s", worker_id, test_name
            )
            result = WorkerResult(
                worker_id=worker_id,
                test_name=test_name,
                success=False,
                error=str(exc),
            )

        result_queue.put(result)

    logger.info("Worker %d exiting", worker_id)


class WorkerManager:
    """Manages worker subprocesses with GPU assignment and health monitoring.

    Each worker is a separate Python process with strict GPU isolation
    (CUDA_VISIBLE_DEVICES), operating in its own git worktree. The manager
    handles lifecycle, health monitoring, OOM recovery, and graceful shutdown.

    Parameters
    ----------
    worktree_base : str
        Base path for git worktrees (e.g., ~/local/worktrees/pandas-fix).
    pytest_parallelism : int
        Default number of parallel pytest workers per subprocess.
    integration_gpu : int
        GPU index reserved for integration testing (default: 7).
    """

    def __init__(
        self,
        worktree_base: str = "~/local/worktrees/pandas-fix",
        pytest_parallelism: int = 16,
        integration_gpu: int = 7,
    ) -> None:
        self._worktree_base = os.path.expanduser(worktree_base)
        self._default_pytest_parallelism = pytest_parallelism
        self._integration_gpu = integration_gpu
        self._workers: dict[int, WorkerInfo] = {}
        self._task_queues: dict[int, "Queue[Any]"] = {}
        self._result_queue: "Queue[WorkerResult]" = Queue()
        self._pause_events: dict[int, multiprocessing.synchronize.Event] = {}
        self._shutdown_requested = False

    def start_workers(self, num_workers: int = 8) -> None:
        """Start worker subprocesses, one per GPU.

        Parameters
        ----------
        num_workers : int
            Number of workers to start (default: 8, one per GPU 0-7).
        """
        if self._workers:
            logger.warning(
                "Workers already running. Call stop_workers() first."
            )
            return

        logger.info("Starting %d worker subprocesses", num_workers)

        for worker_id in range(num_workers):
            self._start_single_worker(worker_id)

        logger.info("All %d workers started successfully", num_workers)

    def _start_single_worker(self, worker_id: int) -> None:
        """Start or restart a single worker subprocess.

        Parameters
        ----------
        worker_id : int
            The worker/GPU index to start.
        """
        worktree_path = os.path.join(
            self._worktree_base, f"worker-{worker_id}"
        )

        # Get existing info or create new
        existing = self._workers.get(worker_id)
        pytest_parallelism = (
            existing.pytest_parallelism
            if existing
            else self._default_pytest_parallelism
        )

        # Create IPC channels
        task_queue: "Queue[Any]" = Queue()
        pause_event = multiprocessing.Event()
        pause_event.set()  # Start in running (not paused) state

        process = Process(
            target=_worker_loop,
            args=(
                worker_id,
                worker_id,  # gpu_id == worker_id
                worktree_path,
                task_queue,
                self._result_queue,
                pause_event,
                pytest_parallelism,
            ),
            daemon=True,
            name=f"worker-{worker_id}-gpu{worker_id}",
        )
        process.start()

        worker_info = WorkerInfo(
            worker_id=worker_id,
            gpu_id=worker_id,
            worktree_path=worktree_path,
            process=process,
            state=WorkerState.IDLE,
            pytest_parallelism=pytest_parallelism,
            consecutive_failures=(
                existing.consecutive_failures if existing else 0
            ),
            oom_count=existing.oom_count if existing else 0,
        )

        self._workers[worker_id] = worker_info
        self._task_queues[worker_id] = task_queue
        self._pause_events[worker_id] = pause_event

        logger.info(
            "Worker %d started (pid=%d, gpu=%d, worktree=%s)",
            worker_id,
            process.pid,
            worker_id,
            worktree_path,
        )

    def stop_workers(self) -> None:
        """Gracefully stop all worker subprocesses.

        Sends SIGTERM first, waits up to 30 seconds, then SIGKILL stragglers.
        """
        if not self._workers:
            return

        self._shutdown_requested = True
        logger.info("Stopping all workers (graceful shutdown)")

        # Send shutdown sentinel to all workers
        for worker_id, task_queue in self._task_queues.items():
            try:
                task_queue.put(_SHUTDOWN_SENTINEL)
            except Exception:
                pass

        # Wait up to 30 seconds for graceful exit
        deadline = time.time() + 30.0
        alive_workers = [
            w
            for w in self._workers.values()
            if w.process is not None and w.process.is_alive()
        ]

        while alive_workers and time.time() < deadline:
            time.sleep(0.5)
            alive_workers = [
                w
                for w in self._workers.values()
                if w.process is not None and w.process.is_alive()
            ]

        # SIGKILL any stragglers
        for worker_info in alive_workers:
            if (
                worker_info.process is not None
                and worker_info.process.is_alive()
            ):
                pid = worker_info.process.pid
                logger.warning(
                    "Worker %d did not exit gracefully, sending SIGKILL (pid=%s)",
                    worker_info.worker_id,
                    pid,
                )
                if pid is not None:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

        # Join all processes
        for worker_info in self._workers.values():
            if worker_info.process is not None:
                worker_info.process.join(timeout=5.0)
            worker_info.state = WorkerState.STOPPED
            worker_info.process = None

        self._workers.clear()
        self._task_queues.clear()
        self._pause_events.clear()
        self._shutdown_requested = False

        logger.info("All workers stopped")

    def pause_worker(self, worker_id: int) -> None:
        """Pause a specific worker (it finishes current task, then waits).

        Used for GPU 7 to yield to integration testing.

        Parameters
        ----------
        worker_id : int
            The worker to pause.
        """
        if worker_id not in self._workers:
            logger.warning("Cannot pause worker %d: not found", worker_id)
            return

        pause_event = self._pause_events.get(worker_id)
        if pause_event is not None:
            pause_event.clear()
            self._workers[worker_id].state = WorkerState.PAUSED
            logger.info(
                "Worker %d paused (will finish current task first)", worker_id
            )

    def resume_worker(self, worker_id: int) -> None:
        """Resume a paused worker.

        Parameters
        ----------
        worker_id : int
            The worker to resume.
        """
        if worker_id not in self._workers:
            logger.warning("Cannot resume worker %d: not found", worker_id)
            return

        pause_event = self._pause_events.get(worker_id)
        if pause_event is not None:
            pause_event.set()
            self._workers[worker_id].state = WorkerState.IDLE
            logger.info("Worker %d resumed", worker_id)

    def pause_integration_gpu(self) -> None:
        """Pause the integration GPU worker (GPU 7 by default).

        Call this when integration testing is about to start.
        The worker will finish its current task and then pause.
        """
        self.pause_worker(self._integration_gpu)

    def resume_integration_gpu(self) -> None:
        """Resume the integration GPU worker after integration testing completes."""
        self.resume_worker(self._integration_gpu)

    def assign_task(self, worker_id: int, task: dict[str, Any]) -> bool:
        """Assign a task to a specific worker.

        Parameters
        ----------
        worker_id : int
            Target worker.
        task : dict
            Task payload containing at minimum 'test_name'.

        Returns
        -------
        bool
            True if the task was successfully queued, False otherwise.
        """
        if worker_id not in self._task_queues:
            logger.error(
                "Cannot assign task to worker %d: not running", worker_id
            )
            return False

        worker = self._workers[worker_id]
        if worker.state == WorkerState.PAUSED:
            logger.warning("Worker %d is paused, task not assigned", worker_id)
            return False

        self._task_queues[worker_id].put(task)
        worker.state = WorkerState.RUNNING
        return True

    def collect_results(self, timeout: float = 0.1) -> list[WorkerResult]:
        """Collect all available results from workers.

        Parameters
        ----------
        timeout : float
            How long to wait for the first result (seconds).

        Returns
        -------
        list[WorkerResult]
            All results available in the result queue.
        """
        results: list[WorkerResult] = []
        try:
            # Get first result with timeout
            result = self._result_queue.get(timeout=timeout)
            results.append(result)
            # Drain any additional results that are immediately available
            while not self._result_queue.empty():
                try:
                    result = self._result_queue.get_nowait()
                    results.append(result)
                except Exception:
                    break
        except Exception:
            pass

        # Process results for health monitoring
        for result in results:
            self._handle_result(result)

        return results

    def _handle_result(self, result: WorkerResult) -> None:
        """Process a worker result for health monitoring and OOM handling.

        Parameters
        ----------
        result : WorkerResult
            The result from a worker subprocess.
        """
        worker = self._workers.get(result.worker_id)
        if worker is None:
            return

        if result.success:
            worker.consecutive_failures = 0
            worker.state = WorkerState.IDLE
        else:
            worker.consecutive_failures += 1

            if result.is_oom:
                worker.oom_count += 1
                # Reduce parallelism for subsequent runs
                new_parallelism = max(1, worker.pytest_parallelism // 2)
                logger.warning(
                    "Worker %d OOM (count=%d). Reducing parallelism: %d -> %d",
                    result.worker_id,
                    worker.oom_count,
                    worker.pytest_parallelism,
                    new_parallelism,
                )
                worker.pytest_parallelism = new_parallelism

            worker.state = WorkerState.IDLE

    def check_health(self) -> dict[int, WorkerState]:
        """Check health of all workers, restart crashed ones.

        Detects crashed worker processes and restarts them (up to
        max_consecutive_failures times).

        Returns
        -------
        dict[int, WorkerState]
            Current state of each worker.
        """
        states: dict[int, WorkerState] = {}

        for worker_id, worker_info in list(self._workers.items()):
            if worker_info.process is None:
                states[worker_id] = WorkerState.STOPPED
                continue

            if not worker_info.process.is_alive():
                exit_code = worker_info.process.exitcode
                if exit_code != 0 and not self._shutdown_requested:
                    logger.error(
                        "Worker %d crashed (exit_code=%s, consecutive_failures=%d)",
                        worker_id,
                        exit_code,
                        worker_info.consecutive_failures,
                    )
                    worker_info.state = WorkerState.CRASHED
                    worker_info.consecutive_failures += 1

                    if (
                        worker_info.consecutive_failures
                        < worker_info.max_consecutive_failures
                    ):
                        logger.info(
                            "Restarting worker %d (attempt %d/%d)",
                            worker_id,
                            worker_info.consecutive_failures,
                            worker_info.max_consecutive_failures,
                        )
                        self._start_single_worker(worker_id)
                    else:
                        logger.error(
                            "Worker %d exceeded max consecutive failures (%d). "
                            "Not restarting.",
                            worker_id,
                            worker_info.max_consecutive_failures,
                        )

            states[worker_id] = self._workers[worker_id].state

        return states

    def get_worker_status(self) -> dict[int, dict[str, Any]]:
        """Get serializable status of all workers (for graph state).

        Returns
        -------
        dict[int, dict]
            Status dict suitable for PipelineState.worker_status.
        """
        status: dict[int, dict[str, Any]] = {}
        for worker_id, worker_info in self._workers.items():
            status[worker_id] = {
                "state": worker_info.state.value,
                "gpu_id": worker_info.gpu_id,
                "worktree": worker_info.worktree_path,
                "consecutive_failures": worker_info.consecutive_failures,
                "oom_count": worker_info.oom_count,
                "pytest_parallelism": worker_info.pytest_parallelism,
                "pid": (
                    worker_info.process.pid
                    if worker_info.process is not None
                    else None
                ),
            }
        return status

    @property
    def active_worker_count(self) -> int:
        """Number of workers currently alive."""
        return sum(
            1
            for w in self._workers.values()
            if w.process is not None and w.process.is_alive()
        )

    @property
    def integration_gpu(self) -> int:
        """The GPU index reserved for integration testing."""
        return self._integration_gpu
