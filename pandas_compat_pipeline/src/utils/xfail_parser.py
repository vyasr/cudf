# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for parsing cudf pandas xfail node lists."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from .models import TestGroup

DEFAULT_PLUGIN_PATH = Path(
    "python/cudf/cudf/pandas/scripts/pandas-testing-plugin.py"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_plugin_path(plugin_path: str | Path | None) -> Path:
    if plugin_path is None:
        return _repo_root() / DEFAULT_PLUGIN_PATH

    resolved = Path(plugin_path)
    if resolved.is_absolute():
        return resolved
    return _repo_root() / resolved


def _extract_fail_dict(source: str) -> dict[str, str]:
    match = re.search(
        r"NODEIDS_THAT_FAIL\s*=\s*(\{.*?\n\})", source, re.DOTALL
    )
    if match is None:
        raise ValueError("Could not locate NODEIDS_THAT_FAIL in plugin source")
    parsed = cast(object, ast.literal_eval(match.group(1)))
    if not isinstance(parsed, dict):
        raise ValueError("NODEIDS_THAT_FAIL must evaluate to a dictionary")
    parsed_mapping = cast(Mapping[object, object], parsed)
    return {str(key): str(value) for key, value in parsed_mapping.items()}


@dataclass(slots=True)
class _GroupedFailure:
    file_path: str
    class_name: str | None
    parametrizations: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)


def _split_parametrization(test_name: str) -> tuple[str, str | None]:
    bracket_depth = 0
    for index, char in enumerate(test_name):
        if char == "[":
            if bracket_depth == 0:
                return test_name[:index], test_name[index + 1 : -1]
            bracket_depth += 1
        elif char == "]":
            if bracket_depth > 0:
                bracket_depth -= 1
    return test_name, None


def get_base_test_name(node_id: str) -> str:
    """Return the node id without parametrization suffixes."""

    parts = node_id.split("::")
    test_name = parts[-1]
    base_test_name, _ = _split_parametrization(test_name)
    prefix = parts[:-1]
    return "::".join([*prefix, base_test_name])


def parse_xfail_list(plugin_path: str | Path | None = None) -> list[TestGroup]:
    """Parse NODEIDS_THAT_FAIL from the pandas testing plugin."""

    plugin_source = _resolve_plugin_path(plugin_path).read_text(
        encoding="utf-8"
    )
    fail_map = _extract_fail_dict(plugin_source)

    grouped: dict[str, _GroupedFailure] = {}

    for node_id, reason in fail_map.items():
        parts = node_id.split("::")
        file_path = parts[0]
        class_name = parts[1] if len(parts) == 3 else None
        test_name = parts[-1]
        base_test_name, parametrization = _split_parametrization(test_name)
        base_name = "::".join(
            [file_path, class_name, base_test_name]
            if class_name
            else [file_path, base_test_name]
        )

        group = grouped.setdefault(
            base_name, _GroupedFailure(file_path, class_name)
        )
        group.node_ids.append(node_id)
        group.reasons.append(reason)
        if parametrization is not None:
            group.parametrizations.append(parametrization)

    results = [
        TestGroup(
            base_name=base_name,
            file_path=data.file_path,
            class_name=data.class_name,
            parametrizations=sorted(data.parametrizations),
            weight=len(data.node_ids),
            reasons=sorted(set(data.reasons)),
            node_ids=sorted(data.node_ids),
        )
        for base_name, data in grouped.items()
    ]
    return sorted(results, key=lambda group: (-group.weight, group.base_name))


def parse_xfail_node_list(
    plugin_path: str | Path | None = None,
) -> list[TestGroup]:
    """Parse NODEIDS_THAT_FAIL as one concrete node per TestGroup."""

    results: list[TestGroup] = []
    for group in parse_xfail_list(plugin_path):
        reason = group.reasons[0] if group.reasons else ""
        for node_id in group.node_ids:
            parts = node_id.split("::")
            _, parametrization = _split_parametrization(parts[-1])
            results.append(
                TestGroup(
                    base_name=node_id,
                    file_path=group.file_path,
                    class_name=group.class_name,
                    parametrizations=[parametrization]
                    if parametrization is not None
                    else [],
                    weight=1,
                    reasons=[reason],
                    node_ids=[node_id],
                )
            )
    return results


def validate_against_collection(
    test_groups: list[TestGroup], pandas_test_dir: str | Path
) -> tuple[list[TestGroup], list[TestGroup]]:
    """Return groups whose backing test files exist and those that appear stale."""

    test_root = Path(pandas_test_dir)
    if not test_root.is_absolute():
        test_root = _repo_root() / test_root

    valid: list[TestGroup] = []
    stale: list[TestGroup] = []
    for group in test_groups:
        target = test_root / group.file_path
        if target.exists():
            valid.append(group)
        else:
            stale.append(group)
    return valid, stale
