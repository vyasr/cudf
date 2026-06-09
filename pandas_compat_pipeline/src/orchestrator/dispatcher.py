# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic task queue and dispatcher for the pandas compatibility fix pipeline.

The Dispatcher manages test assignment across parallel workers within a single
orchestrator process. It uses asyncio.Lock to guarantee atomic state transitions
— no duplicate assignments are possible even under concurrent access.

State is in-memory only. LangGraph graph state is the persistence layer; the
Dispatcher is a helper used within a single orchestrator process.

Note: Tests from NODEIDS_TO_SKIP or NODEIDS_PATHS_TO_SKIP are never present
in the queue because parse_xfail_list() only parses NODEIDS_THAT_FAIL.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import asdict
from typing import Any

from pandas_compat_pipeline.src.utils.models import TestGroup
from pandas_compat_pipeline.src.utils.xfail_parser import parse_xfail_list


class Dispatcher:
    """Atomic task queue managing test assignment across parallel workers.

    Parameters
    ----------
    test_groups : list[TestGroup]
        Ordered list of test groups to process (typically from parse_xfail_list).
    """

    def __init__(self, test_groups: list[TestGroup]) -> None:
        self._lock = asyncio.Lock()
        self._pending: deque[TestGroup] = deque(test_groups)
        self._in_progress: dict[int, TestGroup] = {}
        self._completed: list[dict[str, Any]] = []
        self._failed: list[dict[str, Any]] = []
        self._flagged_for_human: list[dict[str, Any]] = []

    @classmethod
    def from_xfail_list(cls, plugin_path: str | None = None) -> Dispatcher:
        """Create a Dispatcher by parsing the xfail list from the plugin source.

        Parameters
        ----------
        plugin_path : str | None
            Path to the pandas testing plugin. If None, uses the default path.

        Returns
        -------
        Dispatcher
            A new Dispatcher populated with all failing test groups.
        """
        test_groups = parse_xfail_list(plugin_path)
        return cls(test_groups)

    async def get_next_task(self, worker_id: int) -> TestGroup | None:
        """Atomically assign the next pending test group to a worker.

        Pops the first item from the pending queue and records it as
        in-progress for the given worker. Returns None if the queue is empty.

        Parameters
        ----------
        worker_id : int
            The worker (GPU index) requesting a task.

        Returns
        -------
        TestGroup | None
            The assigned test group, or None if no tasks remain.
        """
        async with self._lock:
            if not self._pending:
                return None
            task = self._pending.popleft()
            self._in_progress[worker_id] = task
            return task

    async def complete_task(
        self, worker_id: int, test_name: str, result: dict[str, Any]
    ) -> None:
        """Mark a task as successfully completed.

        Removes the task from in-progress and records it as completed.

        Parameters
        ----------
        worker_id : int
            The worker that completed the task.
        test_name : str
            The base name of the test group that was completed.
        result : dict
            Metadata about the fix (branch, commit, etc.).
        """
        async with self._lock:
            self._in_progress.pop(worker_id, None)
            self._completed.append({"test_name": test_name, "result": result})

    async def fail_task(
        self, worker_id: int, test_name: str, reason: str
    ) -> None:
        """Mark a task as failed (exhausted all fix attempts).

        Removes the task from in-progress and records it as failed.

        Parameters
        ----------
        worker_id : int
            The worker that failed the task.
        test_name : str
            The base name of the test group that failed.
        reason : str
            Description of why the task failed.
        """
        async with self._lock:
            self._in_progress.pop(worker_id, None)
            self._failed.append({"test_name": test_name, "reason": reason})

    async def requeue_task(
        self, test_group: TestGroup, priority: str = "high"
    ) -> None:
        """Re-add a test group to the pending queue.

        Parameters
        ----------
        test_group : TestGroup
            The test group to requeue.
        priority : str
            If 'high', inserts at the front of the queue. If 'normal',
            appends to the back.
        """
        async with self._lock:
            if priority == "high":
                self._pending.appendleft(test_group)
            else:
                self._pending.append(test_group)

    async def requeue_flagged(self, test_name: str, reason: str) -> None:
        """Move a test from in-progress to flagged-for-human review.

        This is for tests that require C++ changes or are otherwise
        unresolvable by automation.

        Parameters
        ----------
        test_name : str
            The base name of the test group being flagged.
        reason : str
            Description of why human intervention is needed.
        """
        async with self._lock:
            # Find and remove from in_progress by matching test_name
            worker_id_to_remove: int | None = None
            for wid, group in self._in_progress.items():
                if group.base_name == test_name:
                    worker_id_to_remove = wid
                    break
            if worker_id_to_remove is not None:
                self._in_progress.pop(worker_id_to_remove)
            self._flagged_for_human.append(
                {"test_name": test_name, "reason": reason}
            )

    def get_stats(self) -> dict[str, int]:
        """Return counts of tests in each state.

        Returns
        -------
        dict
            Counts for pending, in_progress, completed, failed, and
            flagged_for_human.
        """
        return {
            "pending": len(self._pending),
            "in_progress": len(self._in_progress),
            "completed": len(self._completed),
            "failed": len(self._failed),
            "flagged_for_human": len(self._flagged_for_human),
        }

    def get_pending_names(self) -> list[str]:
        """Return base names of all pending test groups (for serialization)."""
        return [g.base_name for g in self._pending]

    def get_in_progress_info(self) -> dict[int, dict[str, Any]]:
        """Return serializable info about in-progress assignments."""
        return {wid: asdict(group) for wid, group in self._in_progress.items()}


if __name__ == "__main__":
    import asyncio as _asyncio

    async def _self_test() -> None:
        """Self-contained test verifying atomicity and state transitions."""
        # Create 10 mock TestGroup objects
        mock_groups = [
            TestGroup(
                base_name=f"tests/test_file.py::TestClass::test_{i}",
                file_path="tests/test_file.py",
                class_name="TestClass",
                parametrizations=[f"param_{j}" for j in range(i + 1)],
                weight=10 - i,
                reasons=["FIXME"],
                node_ids=[
                    f"tests/test_file.py::TestClass::test_{i}[param_{j}]"
                    for j in range(i + 1)
                ],
            )
            for i in range(10)
        ]

        dispatcher = Dispatcher(mock_groups)

        # Verify initial stats
        stats = dispatcher.get_stats()
        assert stats["pending"] == 10, (
            f"Expected 10 pending, got {stats['pending']}"
        )
        assert stats["in_progress"] == 0
        assert stats["completed"] == 0
        assert stats["failed"] == 0
        assert stats["flagged_for_human"] == 0

        # Concurrently assign 2 tasks — must get different tests
        task0, task1 = await _asyncio.gather(
            dispatcher.get_next_task(0),
            dispatcher.get_next_task(1),
        )

        assert task0 is not None, "Worker 0 should get a task"
        assert task1 is not None, "Worker 1 should get a task"
        assert task0.base_name != task1.base_name, (
            f"Workers got same task: {task0.base_name}"
        )

        stats = dispatcher.get_stats()
        assert stats["pending"] == 8
        assert stats["in_progress"] == 2

        # Complete one task
        await dispatcher.complete_task(
            0, task0.base_name, {"branch": "fix/test_0"}
        )
        stats = dispatcher.get_stats()
        assert stats["completed"] == 1
        assert stats["in_progress"] == 1

        # Fail one task
        await dispatcher.fail_task(
            1, task1.base_name, "Max attempts exhausted"
        )
        stats = dispatcher.get_stats()
        assert stats["failed"] == 1
        assert stats["in_progress"] == 0

        # Requeue with high priority — should be next
        await dispatcher.requeue_task(task1, priority="high")
        stats = dispatcher.get_stats()
        assert stats["pending"] == 9  # 8 original + 1 requeued

        next_task = await dispatcher.get_next_task(2)
        assert next_task is not None
        assert next_task.base_name == task1.base_name, (
            "High-priority requeue should be at front of queue"
        )

        # Test requeue_flagged
        await dispatcher.requeue_flagged(
            next_task.base_name, "Needs C++ changes"
        )
        stats = dispatcher.get_stats()
        assert stats["flagged_for_human"] == 1
        assert stats["in_progress"] == 0

        # Drain remaining tasks — verify no duplicates
        assigned_names: set[str] = set()
        for worker_id in range(8):
            t = await dispatcher.get_next_task(worker_id)
            if t is not None:
                assigned_names.add(t.base_name)

        # Should have no duplicates (set size == number assigned)
        assert len(assigned_names) == 8, (
            f"Expected 8 unique, got {len(assigned_names)}"
        )

        # Queue should now be empty
        empty_task = await dispatcher.get_next_task(0)
        assert empty_task is None, "Queue should be empty"

        print("ALL ASSERTIONS PASSED")

    _asyncio.run(_self_test())
