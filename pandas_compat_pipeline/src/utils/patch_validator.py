# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from dataclasses import dataclass

PANDAS_TESTING_PLUGIN_PATH = (
    "python/cudf/cudf/pandas/scripts/pandas-testing-plugin.py"
)
ALLOWED_PREFIXES = ("python/cudf/cudf/",)
FORBIDDEN_PREFIXES = (
    "cpp/",
    "python/pylibcudf/",
    "python/libcudf/",
    "pandas-testing/pandas-tests/",
)
FORBIDDEN_SUFFIXES = (".pyx", ".cu", ".cuh")
KEY_ENTRY_PATTERN = re.compile(r'^\+\s+"tests/.*":\s+".*"')
PACKAGE_ENTRY_PATTERN = re.compile(r'^\s*["\']([^"\']+)["\']\s*,?\s*$')
DEPENDENCY_BLOCK_PATTERN = re.compile(r"^dependencies\s*=\s*\[")
OPTIONAL_DEPENDENCY_BLOCK_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+\s*=\s*\[")
SECTION_PATTERN = re.compile(r"^\[([^\]]+)]")


@dataclass(slots=True)
class ValidationResult:
    approved: bool
    reason: str
    classification: str
    modified_files: list[str]
    violations: list[str]


def classify_change(modified_files: list[str]) -> str:
    normalized_files = [
        path.strip() for path in modified_files if path.strip()
    ]
    has_plugin = PANDAS_TESTING_PLUGIN_PATH in normalized_files
    has_source = any(
        path.startswith("python/cudf/cudf/")
        and path != PANDAS_TESTING_PLUGIN_PATH
        for path in normalized_files
    )

    if has_plugin and has_source:
        return "both"
    if has_plugin and len(normalized_files) == 1:
        return "xfail_removal"
    if has_source and not has_plugin:
        return "source_fix"
    return "unknown"


def validate_patch(
    modified_files: list[str], diff_content: str = ""
) -> ValidationResult:
    normalized_files = [
        path.strip() for path in modified_files if path.strip()
    ]
    violations: list[str] = []

    for path in normalized_files:
        _add_path_violation(path, violations)

    violations.extend(_validate_diff_content(diff_content, normalized_files))

    approved = not violations
    classification = classify_change(normalized_files)
    reason = (
        "Patch approved" if approved else "; ".join(dict.fromkeys(violations))
    )

    return ValidationResult(
        approved=approved,
        reason=reason,
        classification=classification,
        modified_files=normalized_files,
        violations=list(dict.fromkeys(violations)),
    )


def _add_path_violation(path: str, violations: list[str]) -> None:
    for prefix in FORBIDDEN_PREFIXES:
        if path.startswith(prefix):
            if prefix == "cpp/":
                violations.append(f"C++ files not permitted: {path}")
            elif prefix == "python/pylibcudf/":
                violations.append(f"pylibcudf files not permitted: {path}")
            elif prefix == "python/libcudf/":
                violations.append(
                    f"libcudf Python bindings not permitted: {path}"
                )
            else:
                violations.append(
                    f"Vendored pandas tests not permitted: {path}"
                )
            return

    if path.endswith(FORBIDDEN_SUFFIXES):
        violations.append(f"Compiled extension files not permitted: {path}")
        return

    if path.endswith("CMakeLists.txt"):
        violations.append(f"CMake files not permitted: {path}")
        return

    if not _is_allowed_path(path):
        violations.append(f"Out-of-scope file not permitted: {path}")


def _is_allowed_path(path: str) -> bool:
    if path == PANDAS_TESTING_PLUGIN_PATH:
        return True
    return any(path.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def _validate_diff_content(
    diff_content: str, modified_files: list[str]
) -> list[str]:
    if not diff_content:
        return []

    violations: list[str] = []
    if _has_skip_dict_addition(diff_content, "NODEIDS_TO_SKIP"):
        violations.append("Adding entries to NODEIDS_TO_SKIP is not permitted")

    if _has_skip_dict_addition(diff_content, "NODEIDS_PATHS_TO_SKIP"):
        violations.append(
            "Adding entries to NODEIDS_PATHS_TO_SKIP is not permitted"
        )

    if _has_pyproject_dependency_addition(diff_content, modified_files):
        violations.append(
            "Adding pyproject.toml dependencies is not permitted"
        )

    return violations


def _has_skip_dict_addition(diff_content: str, dict_name: str) -> bool:
    lines = diff_content.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("+") or line.startswith("+++"):
            continue
        if dict_name in line:
            return True
        if not KEY_ENTRY_PATTERN.match(line):
            continue
        start = max(0, index - 20)
        end = min(len(lines), index + 21)
        context = "\n".join(lines[start:end])
        if dict_name in context:
            return True
    return False


def _has_pyproject_dependency_addition(
    diff_content: str, modified_files: list[str]
) -> bool:
    if "pyproject.toml" not in diff_content and not any(
        path.endswith("pyproject.toml") for path in modified_files
    ):
        return False

    in_dependencies_block = False
    in_optional_dependencies_block = False
    current_section = ""

    for raw_line in diff_content.splitlines():
        if raw_line.startswith(("+++", "---", "@@")):
            continue
        if not raw_line.startswith(("+", "-", " ")):
            continue

        marker = raw_line[0]
        content = raw_line[1:].strip()

        section_match = SECTION_PATTERN.match(content)
        if section_match:
            current_section = section_match.group(1)
            in_dependencies_block = False
            in_optional_dependencies_block = False
            continue

        if current_section == "project" and DEPENDENCY_BLOCK_PATTERN.match(
            content
        ):
            in_dependencies_block = True
            if marker == "+" and PACKAGE_ENTRY_PATTERN.search(content):
                return True
            continue

        if in_dependencies_block:
            if marker == "+" and PACKAGE_ENTRY_PATTERN.match(content):
                return True
            if "]" in content:
                in_dependencies_block = False

        if (
            current_section == "project.optional-dependencies"
            and OPTIONAL_DEPENDENCY_BLOCK_PATTERN.match(content)
        ):
            in_optional_dependencies_block = True
            if marker == "+" and PACKAGE_ENTRY_PATTERN.search(content):
                return True
            continue

        if (
            in_optional_dependencies_block
            and marker == "+"
            and PACKAGE_ENTRY_PATTERN.match(content)
        ):
            return True
        if in_optional_dependencies_block and "]" in content:
            in_optional_dependencies_block = False

    return False
