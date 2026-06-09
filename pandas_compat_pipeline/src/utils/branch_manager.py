# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

LOGGER = logging.getLogger(__name__)
PLUGIN_RELATIVE_PATH = Path(
    "python/cudf/cudf/pandas/scripts/pandas-testing-plugin.py"
)
BRANCH_PREFIX = "pandas-compat/"
MAX_BRANCH_NAME_LENGTH = 200


@dataclass(slots=True)
class MergeResult:
    success: bool
    merged: list[str]
    conflicts: list[str]
    strategy_used: str


def sanitize_branch_name(node_id_or_name: str) -> str:
    """Convert a pandas node id or test name into a git-safe branch name."""

    sanitized = node_id_or_name.replace("::", "/")
    sanitized = sanitized.replace("[", "_").replace("]", "")
    sanitized = sanitized.replace(" ", "-")
    sanitized = re.sub(r"[^A-Za-z0-9/._-]", "", sanitized)
    sanitized = sanitized.strip("/.-_")
    branch_name = (
        f"{BRANCH_PREFIX}{sanitized}"
        if sanitized
        else BRANCH_PREFIX.rstrip("/")
    )
    return branch_name[:MAX_BRANCH_NAME_LENGTH]


def create_fix_branch(test_base_name: str, worktree_path: str) -> str:
    """Create a fix branch from the current HEAD without switching to it."""

    branch_name = sanitize_branch_name(test_base_name)
    try:
        _ = _run_git(["branch", branch_name], worktree_path)
    except subprocess.CalledProcessError:
        LOGGER.exception("Failed to create fix branch %s", branch_name)
        raise
    return branch_name


def create_integration_branch(
    batch_number: int, base_branch: str = "HEAD"
) -> str:
    """Create an integration branch from the provided base ref."""

    branch_name = f"{BRANCH_PREFIX}integration-{batch_number}"
    try:
        _ = _run_git(["branch", branch_name, base_branch], ".")
    except subprocess.CalledProcessError:
        LOGGER.exception("Failed to create integration branch %s", branch_name)
        raise
    return branch_name


def merge_fix_branches(
    branches: list[str], target_branch: str, worktree_path: str
) -> MergeResult:
    """Merge fix branches into a target branch, resolving known plugin conflicts."""

    merged: list[str] = []
    conflicts: list[str] = []
    strategy_used = "fast-forward"
    original_branch = _current_branch(worktree_path)

    try:
        _ = _run_git(["checkout", target_branch], worktree_path)
        for branch in branches:
            try:
                _ = _run_git(
                    ["merge", "--no-ff", "--no-commit", branch], worktree_path
                )
                _ = _run_git(
                    [
                        "commit",
                        "-m",
                        f"Merge branch '{branch}' into {target_branch}",
                    ],
                    worktree_path,
                )
                merged.append(branch)
                strategy_used = "git-merge"
            except subprocess.CalledProcessError:
                unmerged_files = _get_unmerged_files(worktree_path)
                if unmerged_files == [str(PLUGIN_RELATIVE_PATH)]:
                    _resolve_plugin_conflict(worktree_path)
                    _ = _run_git(
                        [
                            "commit",
                            "-m",
                            f"Merge branch '{branch}' into {target_branch}",
                        ],
                        worktree_path,
                    )
                    merged.append(branch)
                    strategy_used = "plugin-union"
                    continue

                conflicts.append(branch)
                strategy_used = "conflict"
                LOGGER.exception(
                    "Failed to merge branch %s into %s", branch, target_branch
                )
                _abort_merge(worktree_path)

        return MergeResult(
            success=not conflicts,
            merged=merged,
            conflicts=conflicts,
            strategy_used=strategy_used,
        )
    finally:
        if original_branch:
            try:
                _ = _run_git(["checkout", original_branch], worktree_path)
            except subprocess.CalledProcessError:
                LOGGER.exception(
                    "Failed to restore original branch %s", original_branch
                )


def revert_branch_from_integration(
    fix_branch: str, integration_branch: str, worktree_path: str
) -> bool:
    """Revert all commits from a fix branch on an integration branch."""

    original_branch = _current_branch(worktree_path)
    try:
        _ = _run_git(["checkout", integration_branch], worktree_path)
        for commit_hash in reversed(
            get_branch_commits(fix_branch, worktree_path)
        ):
            _ = _run_git(["revert", "--no-commit", commit_hash], worktree_path)
        return True
    except subprocess.CalledProcessError:
        LOGGER.exception(
            "Failed to revert branch %s from integration branch %s",
            fix_branch,
            integration_branch,
        )
        return False
    finally:
        if original_branch:
            try:
                _ = _run_git(["checkout", original_branch], worktree_path)
            except subprocess.CalledProcessError:
                LOGGER.exception(
                    "Failed to restore original branch %s", original_branch
                )


def get_branch_commits(branch_name: str, worktree_path: str) -> list[str]:
    """Return commit hashes unique to the branch relative to HEAD."""

    try:
        completed = _run_git(
            ["rev-list", "HEAD.." + branch_name], worktree_path
        )
    except subprocess.CalledProcessError:
        LOGGER.exception("Failed to list commits for branch %s", branch_name)
        return []
    return [line for line in completed.stdout.splitlines() if line]


def _run_git(
    args: list[str], worktree_path: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )


def _current_branch(worktree_path: str) -> str:
    try:
        completed = _run_git(["branch", "--show-current"], worktree_path)
    except subprocess.CalledProcessError:
        LOGGER.exception("Failed to determine current branch")
        return ""
    return completed.stdout.strip()


def _get_unmerged_files(worktree_path: str) -> list[str]:
    completed = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in completed.stdout.splitlines() if line]


def _abort_merge(worktree_path: str) -> None:
    try:
        _ = _run_git(["merge", "--abort"], worktree_path)
    except subprocess.CalledProcessError:
        LOGGER.exception("Failed to abort merge")


def _resolve_plugin_conflict(worktree_path: str) -> None:
    base_source = _show_stage_file("1", worktree_path)
    ours_source = _show_stage_file("2", worktree_path)
    theirs_source = _show_stage_file("3", worktree_path)

    merged_source = _merge_plugin_source(
        base_source, ours_source, theirs_source
    )
    plugin_path = Path(worktree_path) / PLUGIN_RELATIVE_PATH
    _ = plugin_path.write_text(merged_source, encoding="utf-8")
    _ = _run_git(["add", str(PLUGIN_RELATIVE_PATH)], worktree_path)


def _show_stage_file(stage: str, worktree_path: str) -> str:
    completed = _run_git(
        ["show", f":{stage}:{PLUGIN_RELATIVE_PATH.as_posix()}"], worktree_path
    )
    return completed.stdout


def _merge_plugin_source(
    base_source: str, ours_source: str, theirs_source: str
) -> str:
    base_dict = _extract_fail_dict(base_source)
    ours_dict = _extract_fail_dict(ours_source)
    theirs_dict = _extract_fail_dict(theirs_source)

    merged_dict = {
        key: base_dict[key]
        for key in sorted(base_dict)
        if key in ours_dict and key in theirs_dict
    }

    merged_literal = _format_fail_dict(merged_dict)
    return re.sub(
        r"NODEIDS_THAT_FAIL\s*=\s*\{.*?\n\}",
        f"NODEIDS_THAT_FAIL = {merged_literal}",
        base_source,
        count=1,
        flags=re.DOTALL,
    )


def _extract_fail_dict(source: str) -> dict[str, str]:
    match = re.search(
        r"NODEIDS_THAT_FAIL\s*=\s*(\{.*?\n\})", source, re.DOTALL
    )
    if match is None:
        raise ValueError("Could not locate NODEIDS_THAT_FAIL in plugin source")
    namespace: dict[str, object] = {}
    exec(match.group(0), {"__builtins__": {}}, namespace)
    return cast(dict[str, str], namespace["NODEIDS_THAT_FAIL"])


def _format_fail_dict(entries: dict[str, str]) -> str:
    lines = ["{"]
    for key, value in sorted(entries.items()):
        lines.append(f"    {key!r}: {value!r},")
    lines.append("}")
    return "\n".join(lines)
