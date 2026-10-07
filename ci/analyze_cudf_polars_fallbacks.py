#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize JSON fallback diagnostics from upstream Polars pytest runs."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _report_argument(value: str) -> tuple[str, Path]:
    """Parse an ENGINE=REPORT_JSON command-line argument."""
    engine, separator, path = value.partition("=")
    if not separator or not engine or not path:
        raise argparse.ArgumentTypeError(
            "reports must have the form ENGINE=REPORT_JSON"
        )
    return engine, Path(path)


def build_report(reports: dict[str, Path]) -> dict[str, Any]:
    """Build a compact per-engine summary and a per-node diagnostic table."""
    tests = []
    summaries = {}
    for engine, path in reports.items():
        run = json.loads(path.read_text())
        engine_tests = [
            {**test, "engine": engine}
            for test in sorted(run["tests"], key=lambda test: test["nodeid"])
        ]
        tests.extend(engine_tests)
        summaries[engine] = {
            "exitstatus": run["exitstatus"],
            "collected": run["collected"],
            "total": len(engine_tests),
            "fallback": sum(
                test["fallback"] == "true" for test in engine_tests
            ),
            "no_fallback_observed": sum(
                test["fallback"] == "false" for test in engine_tests
            ),
            "unknown": sum(
                test["fallback"] == "unknown" for test in engine_tests
            ),
            "outcomes": dict(
                sorted(
                    Counter(test["outcome"] for test in engine_tests).items()
                )
            ),
        }
    return {"summary": summaries, "tests": tests}


def write_html(report: dict[str, Any], output: Path) -> None:
    """Write a readable version of the JSON diagnostic data."""
    summaries = "".join(
        "<li>"
        f"{html.escape(engine)}: {summary['fallback']} fallback; "
        f"{summary['no_fallback_observed']} no fallback observed; "
        f"{summary['unknown']} missing telemetry; "
        f"{summary['total']} reported of {summary['collected']} collected; "
        f"exit status {summary['exitstatus']}.</li>"
        for engine, summary in report["summary"].items()
    )
    rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(test['nodeid'])}</code></td>"
        f"<td>{html.escape(test['engine'])}</td>"
        f"<td>{html.escape(test['outcome'])}</td>"
        f"<td>{html.escape(test['fallback'])}</td>"
        "</tr>"
        for test in report["tests"]
    )
    output.write_text(
        "<!doctype html><meta charset=utf-8>"
        "<title>cudf-polars fallback diagnostics</title>"
        "<h1>cudf-polars fallback diagnostics</h1>"
        "<p><code>false</code> means no CPU fallback was observed; it does not "
        "by itself prove that a test invoked LazyFrame.collect.</p>"
        "<p><code>unknown</code> means fallback telemetry is missing. Verify "
        "that the installed injection plugin exports telemetry.</p>"
        f"<ul>{summaries}</ul>"
        "<table><thead><tr><th>Upstream node</th><th>Engine</th>"
        "<th>Outcome</th><th>CPU fallback observed</th>"
        "</tr></thead><tbody>"
        f"{rows}</tbody></table>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        type=_report_argument,
        metavar="ENGINE=REPORT_JSON",
        required=True,
        help="JSON diagnostics from one injected-engine run",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    reports = dict(args.report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(reports)
    (args.output_dir / "fallback-diagnostics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    write_html(report, args.output_dir / "index.html")


if __name__ == "__main__":
    main()
