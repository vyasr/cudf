# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for upstream Polars fallback report aggregation and presentation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_reporter() -> ModuleType:
    return _load_module(
        Path(__file__).parents[2]
        / "python/cudf_polars/cudf_polars/testing/fallback_report.py"
    )


def _report(
    nodeid: str, when: str, outcome: str, fallback: str = "false"
) -> pytest.TestReport:
    return pytest.TestReport(
        nodeid=nodeid,
        location=("test.py", 1, nodeid),
        keywords={},
        outcome=outcome,
        longrepr=None,
        when=when,
        user_properties=[("cudf_polars_fallback", fallback)],
    )


def test_build_report_classifies_fallback_per_engine(tmp_path: Path) -> None:
    analyzer = _load_module(
        Path(__file__).parents[1] / "analyze_cudf_polars_fallbacks.py"
    )
    in_memory = tmp_path / "in-memory.json"
    spmd = tmp_path / "spmd.json"
    in_memory.write_text(
        json.dumps(
            {
                "exitstatus": 1,
                "collected": 2,
                "tests": [
                    {
                        "nodeid": "test_a",
                        "outcome": "passed",
                        "fallback": "true",
                    },
                    {
                        "nodeid": "test_b",
                        "outcome": "failed",
                        "fallback": "false",
                    },
                ],
            }
        )
    )
    spmd.write_text(
        json.dumps(
            {
                "exitstatus": 2,
                "collected": 5,
                "tests": [
                    {
                        "nodeid": "test_a",
                        "outcome": "skipped",
                        "fallback": "false",
                    },
                    {
                        "nodeid": "test_b",
                        "outcome": "error",
                        "fallback": "unknown",
                    },
                ],
            }
        )
    )

    report = analyzer.build_report({"in-memory": in_memory, "spmd": spmd})

    assert report["summary"] == {
        "in-memory": {
            "exitstatus": 1,
            "collected": 2,
            "total": 2,
            "fallback": 1,
            "no_fallback_observed": 1,
            "unknown": 0,
            "outcomes": {"failed": 1, "passed": 1},
        },
        "spmd": {
            "exitstatus": 2,
            "collected": 5,
            "total": 2,
            "fallback": 0,
            "no_fallback_observed": 1,
            "unknown": 1,
            "outcomes": {"error": 1, "skipped": 1},
        },
    }
    assert report["tests"][0] == {
        "nodeid": "test_a",
        "engine": "in-memory",
        "outcome": "passed",
        "fallback": "true",
    }


def test_report_merges_call_failure_and_teardown_fallback(
    tmp_path: Path,
) -> None:
    reporter = _load_reporter().FallbackReport(tmp_path / "report.json")
    for report in (
        _report("test_a", "setup", "passed"),
        _report("test_a", "call", "failed"),
        _report("test_a", "teardown", "failed", "true"),
    ):
        reporter.pytest_runtest_logreport(report)

    assert reporter.tests == {
        "test_a": {"nodeid": "test_a", "outcome": "error", "fallback": "true"},
    }


def test_report_preserves_skips_xfails_xpasses_and_incomplete_items(
    tmp_path: Path,
) -> None:
    reporter = _load_reporter().FallbackReport(tmp_path / "report.json")
    skipped = _report("test_skip", "setup", "skipped")
    xfailed = _report("test_xfail", "call", "skipped", "true")
    xpassed = _report("test_xpass", "call", "passed")
    setattr(xfailed, "wasxfail", "unsupported")
    setattr(xpassed, "wasxfail", "now supported")
    incomplete = _report("test_incomplete", "setup", "passed")
    incomplete.user_properties = []
    for report in (skipped, xfailed, xpassed, incomplete):
        reporter.pytest_runtest_logreport(report)

    assert {
        node: test["outcome"] for node, test in reporter.tests.items()
    } == {
        "test_skip": "skipped",
        "test_xfail": "xfailed",
        "test_xpass": "xpassed",
        "test_incomplete": "incomplete",
    }
    assert reporter.tests["test_incomplete"]["fallback"] == "unknown"
    assert reporter.tests["test_xfail"]["fallback"] == "true"
