# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the upstream Polars fallback diagnostic report."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def _load_analyzer() -> ModuleType:
    path = Path(__file__).parents[1] / "analyze_cudf_polars_fallbacks.py"
    spec = importlib.util.spec_from_file_location("fallback_analysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_junit(path: Path, testcases: str) -> None:
    path.write_text(f"<testsuite>{testcases}</testsuite>")


def test_build_report_classifies_fallback_per_engine(tmp_path: Path) -> None:
    analyzer = _load_analyzer()
    in_memory = tmp_path / "in-memory.xml"
    spmd = tmp_path / "spmd.xml"
    _write_junit(
        in_memory,
        """
        <testcase><properties>
          <property name="cudf_polars_nodeid" value="tests/unit/test_a.py::test_a" />
          <property name="cudf_polars_fallback" value="true" />
        </properties></testcase>
        <testcase><properties>
          <property name="cudf_polars_nodeid" value="tests/unit/test_b.py::test_b" />
          <property name="cudf_polars_fallback" value="false" />
        </properties><failure /></testcase>
        """,
    )
    _write_junit(
        spmd,
        """
        <testcase><properties>
          <property name="cudf_polars_nodeid" value="tests/unit/test_a.py::test_a" />
          <property name="cudf_polars_fallback" value="false" />
        </properties></testcase>
        <testcase name="missing-telemetry" />
        """,
    )

    report = analyzer.build_report({"in-memory": in_memory, "spmd": spmd})

    assert report["summary"] == {
        "in-memory": {
            "total": 2,
            "fallback": 1,
            "no_fallback_observed": 1,
            "unknown": 0,
            "outcomes": {"failed": 1, "passed": 1},
        },
        "spmd": {
            "total": 2,
            "fallback": 0,
            "no_fallback_observed": 1,
            "unknown": 1,
            "outcomes": {"passed": 2},
        },
    }
    assert report["tests"] == [
        {
            "nodeid": "tests/unit/test_a.py::test_a",
            "engine": "in-memory",
            "outcome": "passed",
            "fallback": "true",
        },
        {
            "nodeid": "tests/unit/test_b.py::test_b",
            "engine": "in-memory",
            "outcome": "failed",
            "fallback": "false",
        },
        {
            "nodeid": "missing-telemetry",
            "engine": "spmd",
            "outcome": "passed",
            "fallback": "unknown",
        },
        {
            "nodeid": "tests/unit/test_a.py::test_a",
            "engine": "spmd",
            "outcome": "passed",
            "fallback": "false",
        },
    ]


def test_report_merges_call_failure_and_teardown_error(tmp_path: Path) -> None:
    analyzer = _load_analyzer()
    path = tmp_path / "phases.xml"
    _write_junit(
        path,
        """
        <testcase><properties>
          <property name="cudf_polars_nodeid" value="tests/unit/test_a.py::test_a" />
          <property name="cudf_polars_fallback" value="false" />
        </properties><failure /></testcase>
        <testcase><properties>
          <property name="cudf_polars_nodeid" value="tests/unit/test_a.py::test_a" />
          <property name="cudf_polars_fallback" value="true" />
        </properties><error /></testcase>
        """,
    )

    report = analyzer.build_report({"in-memory": path})

    assert report["summary"]["in-memory"] == {
        "total": 1,
        "fallback": 1,
        "no_fallback_observed": 0,
        "unknown": 0,
        "outcomes": {"error": 1},
    }
    assert report["tests"][0]["outcome"] == "error"
    assert report["tests"][0]["fallback"] == "true"


def test_report_preserves_skips_xfails_and_missing_telemetry(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    path = tmp_path / "skips.xml"
    _write_junit(
        path,
        """
        <testcase name="test_skip"><properties>
          <property name="cudf_polars_nodeid" value="tests/unit/test_a.py::test_skip" />
          <property name="cudf_polars_fallback" value="false" />
        </properties><skipped type="pytest.skip" /></testcase>
        <testcase name="test_xfail"><properties>
          <property name="cudf_polars_nodeid" value="tests/unit/test_a.py::test_xfail" />
          <property name="cudf_polars_fallback" value="true" />
        </properties><skipped type="pytest.xfail" /></testcase>
        <testcase classname="tests.unit.test_a" name="test_missing"><error /></testcase>
        """,
    )

    report = analyzer.build_report({"spmd": path})

    assert report["summary"]["spmd"] == {
        "total": 3,
        "fallback": 1,
        "no_fallback_observed": 1,
        "unknown": 1,
        "outcomes": {"error": 1, "skipped": 1, "xfailed": 1},
    }
    missing = next(
        test for test in report["tests"] if test["fallback"] == "unknown"
    )
    assert missing["nodeid"] == "tests.unit.test_a::test_missing"
