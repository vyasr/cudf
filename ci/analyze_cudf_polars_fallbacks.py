#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize JSON fallback diagnostics from upstream Polars CI runs."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def build_report(paths: list[Path]) -> dict[str, Any]:
    """Combine available shard/configuration reports without losing run identity."""
    tests = []
    runs = []
    unreadable = []
    for path in paths:
        try:
            run = json.loads(path.read_text())
            engine = run["engine"]
            if engine == "spmd":
                engine += f"-{run['blocksize']}"
            run["exitstatus"]
            run["collected"]
            run_tests = run.pop("tests")
            # Equal node IDs in different matrix jobs are separate executions,
            # not duplicates: fallback can depend on the tested configuration.
            run_tests = [
                {
                    "nodeid": test["nodeid"],
                    "outcome": test["outcome"],
                    "fallback": test["fallback"],
                    "engine": engine,
                    "run": path.name,
                }
                for test in run_tests
            ]
            tests.extend(run_tests)
            runs.append({**run, "engine": engine, "run": path.name})
        except (OSError, ValueError, KeyError, TypeError) as error:
            unreadable.append({"run": path.name, "error": str(error)})

    grouped = defaultdict(list)
    for test in tests:
        grouped[test["engine"]].append(test)
    summaries = {}
    for engine in sorted({run["engine"] for run in runs}):
        engine_runs = [run for run in runs if run["engine"] == engine]
        engine_tests = grouped[engine]
        summaries[engine] = {
            "runs": len(engine_runs),
            "nonzero_exitstatus": sum(
                run["exitstatus"] != 0 for run in engine_runs
            ),
            "collected": sum(run["collected"] for run in engine_runs),
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
    return {
        "summary": summaries,
        "runs": runs,
        "unreadable_reports": unreadable,
        "tests": sorted(
            tests,
            key=lambda test: (test["engine"], test["nodeid"], test["run"]),
        ),
    }


def summary_markdown(report: dict[str, Any]) -> str:
    """Describe observed executions, not an assumed complete baseline."""
    lines = [
        "## Upstream Polars CPU fallback diagnostics",
        "",
        "Counts include executions across the available shards and matrix configurations. "
        "Missing jobs are not represented; failed or partial runs are not a complete baseline.",
        "",
        "No fallback observed does not prove GPU execution: eager-only tests also report false.",
        "",
        "| Engine | Reports | CPU fallback | No fallback observed | Missing telemetry | Reported / collected | Runs with nonzero exit status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for engine, summary in report["summary"].items():
        lines.append(
            f"| {engine} | {summary['runs']} | {summary['fallback']} | "
            f"{summary['no_fallback_observed']} | {summary['unknown']} | "
            f"{summary['total']} / {summary['collected']} | {summary['nonzero_exitstatus']} |"
        )
    if not report["runs"]:
        lines.extend(
            ["", "No readable diagnostics were uploaded by the upstream jobs."]
        )
    if report["unreadable_reports"]:
        lines.extend(
            [
                "",
                f"Unreadable reports: {len(report['unreadable_reports'])}; see the JSON artifact.",
            ]
        )
    return "\n".join(lines) + "\n"


def write_html(report: dict[str, Any], output: Path) -> None:
    """Include run identity so repeated nodes across configurations stay traceable."""
    rows = "".join(
        "<tr>"
        + "".join(
            f"<td><code>{html.escape(test[key])}</code></td>"
            for key in ("nodeid", "engine", "run", "outcome", "fallback")
        )
        + "</tr>"
        for test in report["tests"]
    )
    output.write_text(
        "<!doctype html><meta charset=utf-8><title>cudf-polars fallback diagnostics</title>"
        f"<pre>{html.escape(summary_markdown(report))}</pre>"
        "<table><thead><tr><th>Upstream node</th><th>Engine</th><th>Run</th>"
        "<th>Outcome</th><th>CPU fallback observed</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-file", type=Path)
    args = parser.parse_args()
    report = build_report(sorted(args.reports_dir.glob("*.json")))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fallback-diagnostics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    write_html(report, args.output_dir / "index.html")
    if args.summary_file:
        with args.summary_file.open("a") as summary:
            summary.write(summary_markdown(report))


if __name__ == "__main__":
    main()
