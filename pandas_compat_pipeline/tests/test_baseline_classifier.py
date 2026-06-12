# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path[:0] = [
    _path
    for _path in (
        str(Path(__file__).resolve().parents[1]),
        str(Path(__file__).resolve().parents[1].parent),
    )
    if _path not in sys.path
]

from pandas_compat_pipeline.src.agents.fixer import FixerAgent  # noqa: E402
from pandas_compat_pipeline.src.agents.llm_client import (  # noqa: E402
    LLMClient,
)
from pandas_compat_pipeline.src.utils.models import TestGroup  # noqa: E402
from pandas_compat_pipeline.src.utils.test_runner import (  # noqa: E402
    TestOutcome,
    TestResult,
)


def _make_agent() -> FixerAgent:
    """Create a FixerAgent with a mocked LLM client."""
    client = MagicMock(spec=LLMClient)
    agent = FixerAgent(llm_client=client)
    return agent


def _make_test_group(file_path: str = "tests/io/test_sql.py") -> TestGroup:
    return TestGroup(
        base_name="test_read_sql",
        file_path=file_path,
        class_name=None,
        node_ids=["tests/io/test_sql.py::test_read_sql[param1]"],
    )


def test_deselected_single_cpu_marker() -> None:
    """Test with '1 deselected in 1.03s' → 'deselected'."""
    agent = _make_agent()
    results = [
        TestResult(
            node_id="x",
            outcome=TestOutcome.FAILED,
            longrepr="1 deselected in 1.03s",
            stdout="",
            duration=0.0,
        )
    ]
    classification = agent._classify_baseline_results(
        results, "/some/worktree", _make_test_group()
    )
    assert classification == "deselected"


def test_deselected_zero_selected() -> None:
    """Test with 'deselected / 0 selected' → 'deselected'."""
    agent = _make_agent()
    results = [
        TestResult(
            node_id="x",
            outcome=TestOutcome.FAILED,
            longrepr="collected 1 item / 1 deselected / 0 selected",
            stdout="",
            duration=0.0,
        )
    ]
    classification = agent._classify_baseline_results(
        results, "/some/worktree", _make_test_group()
    )
    assert classification == "deselected"


def test_deselected_multi_results_all_match() -> None:
    """3 results all with '1 deselected' → 'deselected'."""
    agent = _make_agent()
    results = [
        TestResult(
            node_id=f"x{i}",
            outcome=TestOutcome.FAILED,
            longrepr="1 deselected in 0.5s",
            stdout="",
            duration=0.0,
        )
        for i in range(3)
    ]
    classification = agent._classify_baseline_results(
        results, "/some/worktree", _make_test_group()
    )
    assert classification == "deselected"


def test_missing_not_found() -> None:
    """Test with 'not found' collection error → 'missing'."""
    agent = _make_agent()
    results = [
        TestResult(
            node_id="y",
            outcome=TestOutcome.FAILED,
            longrepr=(
                "collected 0 items\n"
                "ERROR: not found: y (no match in any of [<Class Foo>])"
            ),
            stdout="",
            duration=0.0,
        )
    ]
    classification = agent._classify_baseline_results(
        results, "/some/worktree", _make_test_group()
    )
    assert classification == "missing"


def test_missing_no_match_in_any_of() -> None:
    """Test with 'no match in any of' → 'missing'."""
    agent = _make_agent()
    results = [
        TestResult(
            node_id="y",
            outcome=TestOutcome.FAILED,
            longrepr="no match in any of [<Module test_sql.py>]",
            stdout="",
            duration=0.0,
        )
    ]
    classification = agent._classify_baseline_results(
        results, "/some/worktree", _make_test_group()
    )
    assert classification == "missing"


def test_skipped_found_no_collectors() -> None:
    """Test with 'found no collectors' but NOT 'not found' → 'deselected'."""
    agent = _make_agent()
    results = [
        TestResult(
            node_id="z",
            outcome=TestOutcome.FAILED,
            longrepr="found no collectors for test_odswriter.py::test_cell_value_type",
            stdout="",
            duration=0.0,
        )
    ]
    classification = agent._classify_baseline_results(
        results, "/some/worktree", _make_test_group()
    )
    assert classification == "deselected"


def test_genuine_failure_not_classified() -> None:
    """Genuine test failure → None (not classified)."""
    agent = _make_agent()
    results = [
        TestResult(
            node_id="z",
            outcome=TestOutcome.FAILED,
            longrepr="AssertionError: assert 1 == 2",
            stdout="",
            duration=0.0,
        )
    ]
    classification = agent._classify_baseline_results(
        results, "/some/worktree", _make_test_group()
    )
    assert classification is None


def test_mixed_results_deselected_and_real_failure() -> None:
    """One '1 deselected', one 'AssertionError' → None (not all match)."""
    agent = _make_agent()
    results = [
        TestResult(
            node_id="a",
            outcome=TestOutcome.FAILED,
            longrepr="1 deselected in 1.03s",
            stdout="",
            duration=0.0,
        ),
        TestResult(
            node_id="b",
            outcome=TestOutcome.FAILED,
            longrepr="AssertionError: assert 1 == 2",
            stdout="",
            duration=0.0,
        ),
    ]
    classification = agent._classify_baseline_results(
        results, "/some/worktree", _make_test_group()
    )
    assert classification is None


def test_empty_results() -> None:
    """Empty results list → None."""
    agent = _make_agent()
    classification = agent._classify_baseline_results(
        [], "/some/worktree", _make_test_group()
    )
    assert classification is None


def test_ambiguous_collected_zero_file_exists(tmp_path: Path) -> None:
    """'collected 0 items' with test file existing → 'deselected'."""
    agent = _make_agent()
    # Create the expected file path in the temp worktree
    test_file = (
        tmp_path
        / "pandas-testing"
        / "pandas-tests"
        / "tests"
        / "io"
        / "test_sql.py"
    )
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("# test file")

    results = [
        TestResult(
            node_id="x",
            outcome=TestOutcome.FAILED,
            longrepr="collected 0 items",
            stdout="",
            duration=0.0,
        )
    ]
    classification = agent._classify_baseline_results(
        results, str(tmp_path), _make_test_group()
    )
    assert classification == "deselected"


def test_ambiguous_collected_zero_file_missing(tmp_path: Path) -> None:
    """'collected 0 items' with test file NOT existing → 'missing'."""
    agent = _make_agent()
    # Do NOT create the file — tmp_path is empty
    results = [
        TestResult(
            node_id="x",
            outcome=TestOutcome.FAILED,
            longrepr="collected 0 items",
            stdout="",
            duration=0.0,
        )
    ]
    classification = agent._classify_baseline_results(
        results, str(tmp_path), _make_test_group()
    )
    assert classification == "missing"


def test_missing_no_tests_ran(tmp_path: Path) -> None:
    """'no tests ran' with test file NOT existing → 'missing'."""
    agent = _make_agent()
    # Do NOT create the file — tmp_path is empty
    results = [
        TestResult(
            node_id="x",
            outcome=TestOutcome.FAILED,
            longrepr="no tests ran",
            stdout="",
            duration=0.0,
        )
    ]
    classification = agent._classify_baseline_results(
        results, str(tmp_path), _make_test_group()
    )
    assert classification == "missing"


def test_mixed_deselected_and_missing_not_classified() -> None:
    """Mixed group: one deselected, one missing → should return None (not classify)."""
    agent = _make_agent()
    results = [
        TestResult(
            node_id="test_a",
            outcome=TestOutcome.FAILED,
            longrepr="1 deselected in 0.5s",
            stdout="",
            duration=0.0,
        ),
        TestResult(
            node_id="test_b",
            outcome=TestOutcome.FAILED,
            longrepr="ERROR: not found: test_b (no match in any of [<Module foo>])",
            stdout="",
            duration=0.0,
        ),
    ]
    classification = agent._classify_baseline_results(
        results, "/tmp/fake", _make_test_group()
    )
    assert classification is None
