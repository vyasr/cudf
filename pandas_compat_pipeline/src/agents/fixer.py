# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# pyright: reportExplicitAny=false, reportAny=false, reportUnknownVariableType=false, reportUnannotatedClassAttribute=false
import asyncio
import json
import logging
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..config import PipelineConfig, load_config
from ..utils.branch_manager import sanitize_branch_name
from ..utils.models import TestGroup
from ..utils.patch_validator import validate_patch
from ..utils.test_runner import TestOutcome, TestResult, run_test
from .llm_client import LLMClient
from .tools import get_tools, read_file, run_command, search_code, write_file

LOGGER = logging.getLogger(__name__)

PLUGIN_PATH = "python/cudf/cudf/pandas/scripts/pandas-testing-plugin.py"
SKILL_PATH = ".agents/skills/debug-cudf-pandas/SKILL.md"
MAX_TOOL_CONTENT_CHARS = 8000
MAX_INVESTIGATION_CYCLES_WITHOUT_PATCH = 6
PROHIBITED_RESPONSE_MARKERS = (
    "intentional behavioral divergence",
    "by design",
)
HUMAN_REVIEW_BLOCKER_MARKERS = (
    "pytest_load_initial_conftests",
    "cudf/pandas/_wrappers/",
    "cudf\\pandas\\_wrappers\\",
    "cudf/pandas/__init__.py",
    "mv: cannot stat './*.json'",
)
HUMAN_REVIEW_EXCEPTION_MARKERS = (
    "AttributeError:",
    "ImportError:",
    "ModuleNotFoundError:",
)
MUTATING_COMMAND_MARKERS = (
    " >",
    ">",
    "tee ",
    "sed -i",
    "perl -pi",
    "cat <<",
    "rm ",
    "mv ",
    "cp ",
)


@dataclass(slots=True)
class FixResult:
    status: Literal["success", "failed", "flagged_for_human"]
    branch_name: str
    modified_files: list[str] = field(default_factory=list)
    test_results: list[dict[str, Any]] = field(default_factory=list)
    diagnosis: str = ""
    attempts: int = 0
    rejection_reason: str = ""


class HumanReviewRequired(RuntimeError):
    """Raised when the fixer reaches a stop condition."""


class FixerAgent:
    """LLM-powered cudf.pandas pandas-test fixer."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.config: PipelineConfig = config or load_config()
        self.llm_client: LLMClient = llm_client or LLMClient(
            base_url=self.config.inference_hub_url,
            max_concurrent=self.config.max_concurrent_llm_calls,
        )

        # Diagnostic state (initialized per fix() call)
        self._diag_cycles: list[dict[str, Any]] = []
        self._diag_verify: list[dict[str, Any]] = []
        self._diag_start_time: float = 0.0
        self._diag_xfail_removed: bool = False
        self._diag_baseline_passed: bool = False
        self._diag_test_group: TestGroup | None = None

    async def fix(
        self, test_group: TestGroup, worktree_path: str, gpu_id: int
    ) -> FixResult:
        branch_name = sanitize_branch_name(test_group.base_name)
        attempts = 0
        modified_files: list[str] = []
        test_results: list[dict[str, Any]] = []
        diagnosis = ""

        # Diagnostic accumulators
        self._diag_cycles = []
        self._diag_verify = []
        self._diag_start_time = time.time()
        self._diag_xfail_removed = False
        self._diag_baseline_passed = False
        self._diag_test_group = test_group

        result: FixResult | None = None
        try:
            await self._checkout_fix_branch(worktree_path, branch_name)

            removed = await self._remove_xfail_entries(
                worktree_path, test_group.node_ids
            )
            if removed:
                modified_files.append(PLUGIN_PATH)
                self._diag_xfail_removed = True

            baseline_results = await self._run_group_tests(
                worktree_path,
                test_group,
                gpu_id,
                run_mode="baseline",
                fail_on_fallback=False,
            )
            test_results.extend(
                self._result_to_dict(result_item)
                for result_item in baseline_results
            )

            blocker_reason = self._human_review_blocker_reason(
                baseline_results, phase="baseline"
            )
            if blocker_reason:
                await self._restore_removed_xfails(
                    worktree_path, modified_files
                )
                result = FixResult(
                    status="flagged_for_human",
                    branch_name=branch_name,
                    modified_files=modified_files,
                    test_results=test_results,
                    diagnosis=diagnosis,
                    attempts=attempts,
                    rejection_reason=blocker_reason,
                )
                return result

            if self._all_passed(baseline_results):
                self._diag_baseline_passed = True
                diagnosis = "Removed stale xfail entry; target tests now pass."
            else:
                messages = await self._initial_messages(
                    test_group=test_group,
                    worktree_path=worktree_path,
                    branch_name=branch_name,
                    baseline_results=baseline_results,
                    removed_xfail=removed,
                )
                loop_result = await self._tool_loop(
                    messages=messages,
                    worktree_path=worktree_path,
                    modified_files=modified_files,
                )
                attempts = loop_result["attempts"]
                diagnosis = loop_result["diagnosis"]

            current_modified = await self._git_modified_files(worktree_path)
            for path in current_modified:
                if path not in modified_files:
                    self._validate_or_raise(path)
                    modified_files.append(path)

            verification_results = await self._verify_fix(
                worktree_path, test_group, gpu_id
            )
            test_results.extend(
                self._result_to_dict(result_item)
                for result_item in verification_results
            )

            # Capture verification diagnostics
            try:
                for vr in verification_results:
                    self._diag_verify.append(
                        {
                            "node_id": vr.node_id,
                            "outcome": vr.outcome.value
                            if hasattr(vr.outcome, "value")
                            else str(vr.outcome),
                            "longrepr": vr.longrepr or "",
                            "stdout": vr.stdout or "",
                            "duration": vr.duration,
                            "run_mode": "verification",
                        }
                    )
            except Exception:
                pass

            if not self._all_passed(verification_results):
                reason = (
                    "Verification failed; no commit created. "
                    + self._summarize_failures(verification_results)
                )
                blocker_reason = self._human_review_blocker_reason(
                    verification_results, phase="verification", prefix=reason
                )
                if blocker_reason:
                    await self._restore_removed_xfails(
                        worktree_path, modified_files
                    )
                    result = FixResult(
                        status="flagged_for_human",
                        branch_name=branch_name,
                        modified_files=modified_files,
                        test_results=test_results,
                        diagnosis=diagnosis,
                        attempts=attempts,
                        rejection_reason=blocker_reason,
                    )
                    return result
                result = FixResult(
                    status="failed",
                    branch_name=branch_name,
                    modified_files=modified_files,
                    test_results=test_results,
                    diagnosis=diagnosis,
                    attempts=attempts,
                    rejection_reason=reason,
                )
                return result

            await self._commit_success(
                worktree_path, branch_name, modified_files, test_group
            )
            result = FixResult(
                status="success",
                branch_name=branch_name,
                modified_files=modified_files,
                test_results=test_results,
                diagnosis=diagnosis,
                attempts=attempts,
                rejection_reason="",
            )
            return result
        except HumanReviewRequired as exc:
            result = FixResult(
                status="flagged_for_human",
                branch_name=branch_name,
                modified_files=modified_files,
                test_results=test_results,
                diagnosis=diagnosis,
                attempts=attempts,
                rejection_reason=str(exc),
            )
            return result
        except Exception as exc:  # pragma: no cover - failure payload path
            LOGGER.exception("Fixer failed for %s", test_group.base_name)
            result = FixResult(
                status="failed",
                branch_name=branch_name,
                modified_files=modified_files,
                test_results=test_results,
                diagnosis=diagnosis,
                attempts=attempts,
                rejection_reason=str(exc),
            )
            return result
        finally:
            self._write_diagnostic(test_group, result, worktree_path)

    async def _initial_messages(
        self,
        *,
        test_group: TestGroup,
        worktree_path: str,
        branch_name: str,
        baseline_results: list[TestResult],
        removed_xfail: bool,
    ) -> list[dict[str, str]]:
        skill_text = await self._load_skill_text(worktree_path)
        test_source = await self._read_test_source(
            worktree_path, test_group.file_path
        )
        return [
            {"role": "system", "content": self._system_prompt(skill_text)},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "Fix this cudf.pandas pandas compatibility failure by following the embedded debug-cudf-pandas process.",
                        "branch_name": branch_name,
                        "worktree_path": worktree_path,
                        "test_group": {
                            "base_name": test_group.base_name,
                            "file_path": test_group.file_path,
                            "class_name": test_group.class_name,
                            "parametrizations": test_group.parametrizations,
                            "node_ids": test_group.node_ids,
                            "reasons": test_group.reasons,
                        },
                        "xfail_entry_removed": removed_xfail,
                        "baseline_results": [
                            self._result_to_dict(result)
                            for result in baseline_results
                        ],
                        "test_source_tool_result": {
                            "tool": "read_file",
                            "path": self._test_source_path(
                                test_group.file_path
                            ),
                            "content": test_source,
                        },
                        "instructions": [
                            "Diagnose root cause before writing patches.",
                            "Use search_code/read_file/run_command as needed for diagnosis.",
                            "Use write_file for every file modification; do not write via shell commands.",
                            "Return final text with diagnosis after patching, or explain why no safe patch is possible.",
                        ],
                    },
                    indent=2,
                ),
            },
        ]

    def _system_prompt(self, skill_text: str) -> str:
        return f"""
You are the Fixer Agent for the cudf.pandas pandas compatibility pipeline.
You are powered by Claude 4.6 Opus through LLMClient.call_fixer().

You MUST follow the six-step debug-cudf-pandas process:
1. Remove the xfail entry from pandas-testing-plugin.py when present.
2. Reproduce the failure using the provided run_test baseline results.
3. Read and understand the pandas test source; the initial user message includes a read_file tool result for it.
4. Diagnose root cause with reasoning and tools. Always check direct cudf behavior before proxy fixes.
5. Implement the fix using write_file only. Every write is patch-validator gated by the host.
6. Verification is run by the host after your final response; do not declare success without a concrete patch or stale-xfail diagnosis.

Hard prohibitions from the skill:
- No esoteric or test-specific special cases.
- No pandas CPU fallback as the fix.
- No private pandas API imports or calls, including pandas._libs, pandas.core, pandas.compat, and underscored pandas modules.
- No pyarrow.compute as an execution backend.
- No returning pandas objects from cudf public APIs.
- No diverging from pandas behavior to pass a test.
- No modifications to vendored pandas tests under pandas-testing/pandas-tests/.
- No C++, CUDA, Cython, pylibcudf, libcudf bindings, CMake, .pyx, .cu, or .cuh changes.
- Do not add mode.pandas_compatible guards.
- Do not add entries to skip dictionaries or dependencies.

Tool rules:
- Use write_file for any file write; never write files through run_command.
- Do not ask for the full pandas-testing-plugin.py xfail list.
- Keep changes minimal and general to the API contract.
- If the only possible fix violates a prohibition, stop and say it must be flagged for human review.

The complete debug-cudf-pandas skill content is embedded below. Follow it exactly.

<debug-cudf-pandas-skill>
{skill_text}
</debug-cudf-pandas-skill>
""".strip()

    async def _tool_loop(
        self,
        *,
        messages: list[dict[str, str]],
        worktree_path: str,
        modified_files: list[str],
    ) -> dict[str, Any]:
        attempts = 0
        reprompt_count = 0
        cycles_without_patch = 0
        diagnosis = ""

        while attempts < self.config.max_fix_attempts:
            # Diagnostic: snapshot messages before LLM call
            try:
                diag_messages_before = list(messages)
            except Exception:
                diag_messages_before = []

            response = await self.llm_client.call_fixer(
                messages, tools=get_tools()
            )
            if self._mentions_stop_condition(response):
                raise HumanReviewRequired(
                    "LLM reported intentional behavioral divergence or by-design behavior."
                )

            tool_calls = self._parse_tool_calls(response)
            if not tool_calls:
                diagnosis = response

                # Diagnostic: capture cycle with no tool calls
                try:
                    self._diag_cycles.append(
                        {
                            "cycle": attempts,
                            "messages_before": diag_messages_before,
                            "raw_llm_response": response,
                            "tool_calls_parsed": [],
                            "tool_results_full": [],
                            "wrote_patch": False,
                            "cycles_without_patch": cycles_without_patch,
                            "reprompted": reprompt_count < 2,
                        }
                    )
                except Exception:
                    pass

                if reprompt_count < 2:
                    messages.append({"role": "assistant", "content": response})
                    messages.append(
                        {
                            "role": "user",
                            "content": "ERROR: Your response must be a JSON array of tool calls. Use read_file, search_code, run_command, or write_file. Do not respond with prose.",
                        }
                    )
                    reprompt_count += 1
                    continue

                if (
                    cycles_without_patch
                    >= MAX_INVESTIGATION_CYCLES_WITHOUT_PATCH
                ):
                    raise HumanReviewRequired(
                        "3+ investigation cycles completed without producing a patch."
                    )
                return {"attempts": attempts, "diagnosis": diagnosis}

            attempts += 1
            reprompt_count = 0
            wrote_patch = False
            tool_results: list[dict[str, Any]] = []
            cycle_tool_results_full: list[dict[str, Any]] = []
            for tool_call in tool_calls:
                result = await self._execute_tool_call(
                    tool_call, worktree_path, modified_files
                )
                wrote_patch = wrote_patch or result.get("wrote_patch", False)
                tool_results.append(result)
                # Diagnostic: capture full tool call and result
                try:
                    cycle_tool_results_full.append(
                        {
                            "tool_call": tool_call,
                            "result": result,
                        }
                    )
                except Exception:
                    pass

            if wrote_patch:
                cycles_without_patch = 0
            else:
                cycles_without_patch += 1
                if (
                    cycles_without_patch
                    >= MAX_INVESTIGATION_CYCLES_WITHOUT_PATCH
                ):
                    raise HumanReviewRequired(
                        "3+ investigation cycles completed without producing a patch."
                    )

            # Diagnostic: capture full cycle data
            try:
                self._diag_cycles.append(
                    {
                        "cycle": attempts,
                        "messages_before": diag_messages_before,
                        "raw_llm_response": response,
                        "tool_calls_parsed": tool_calls,
                        "tool_results_full": cycle_tool_results_full,
                        "wrote_patch": wrote_patch,
                        "cycles_without_patch": cycles_without_patch,
                    }
                )
            except Exception:
                pass

            messages.append({"role": "assistant", "content": response})
            messages.append(
                {
                    "role": "user",
                    "content": "Tool results:\n"
                    + json.dumps(tool_results, indent=2, default=str),
                }
            )

        raise HumanReviewRequired(
            f"Exceeded max fixer attempts ({self.config.max_fix_attempts})."
        )

    async def _execute_tool_call(
        self,
        tool_call: dict[str, Any],
        worktree_path: str,
        modified_files: list[str],
    ) -> dict[str, Any]:
        call_id = str(tool_call.get("id", ""))
        function = tool_call.get("function", {})
        name = function.get("name")
        arguments = self._parse_arguments(function.get("arguments", {}))

        try:
            if name == "read_file":
                rel_path = self._normalize_relative_path(
                    arguments["path"], worktree_path
                )
                content = await asyncio.to_thread(
                    read_file, str(Path(worktree_path) / rel_path)
                )
                return {
                    "tool_call_id": call_id,
                    "tool": name,
                    "path": rel_path,
                    "content": self._truncate(content),
                }

            if name == "search_code":
                rel_path = self._normalize_relative_path(
                    arguments.get("path", "."), worktree_path
                )
                output = await asyncio.to_thread(
                    search_code,
                    str(arguments["pattern"]),
                    str(Path(worktree_path) / rel_path),
                )
                return {
                    "tool_call_id": call_id,
                    "tool": name,
                    "path": rel_path,
                    "output": self._truncate(output),
                }

            if name == "run_command":
                cmd = str(arguments["cmd"])
                if self._command_looks_mutating(cmd):
                    return {
                        "tool_call_id": call_id,
                        "tool": name,
                        "error": "Mutating shell commands are blocked. Use write_file for file writes.",
                    }
                cwd = arguments.get("cwd")
                command_cwd = (
                    str(
                        Path(worktree_path)
                        / self._normalize_relative_path(cwd, worktree_path)
                    )
                    if cwd
                    else worktree_path
                )
                result = await asyncio.to_thread(
                    run_command,
                    cmd,
                    command_cwd,
                    int(arguments.get("timeout", 300)),
                )
                result["stdout"] = self._truncate(
                    str(result.get("stdout", ""))
                )
                result["stderr"] = self._truncate(
                    str(result.get("stderr", ""))
                )
                return {
                    "tool_call_id": call_id,
                    "tool": name,
                    "result": result,
                }

            if name == "write_file":
                rel_path = self._normalize_relative_path(
                    arguments["path"], worktree_path
                )
                self._validate_or_raise(rel_path)
                abs_path = Path(worktree_path) / rel_path
                result = await asyncio.to_thread(
                    write_file, str(abs_path), str(arguments["content"])
                )
                self._validate_or_raise(rel_path)
                if not result.get("success"):
                    raise RuntimeError(
                        result.get("error", "write_file failed")
                    )
                if rel_path not in modified_files:
                    modified_files.append(rel_path)
                return {
                    "tool_call_id": call_id,
                    "tool": name,
                    "path": rel_path,
                    "result": result,
                    "wrote_patch": True,
                }

            return {
                "tool_call_id": call_id,
                "tool": name,
                "error": f"Unknown tool: {name}",
            }
        except HumanReviewRequired:
            raise
        except Exception as exc:
            return {
                "tool_call_id": call_id,
                "tool": name,
                "error": str(exc),
            }

    async def _remove_xfail_entries(
        self, worktree_path: str, node_ids: list[str]
    ) -> bool:
        if not node_ids:
            return False
        self._validate_or_raise(PLUGIN_PATH)
        plugin = Path(worktree_path) / PLUGIN_PATH

        def remove_entries() -> bool:
            source = plugin.read_text(encoding="utf-8")
            lines = source.splitlines(keepends=True)
            filtered: list[str] = []
            removed = False
            for line in lines:
                stripped = line.lstrip()
                if any(
                    stripped.startswith(f"{node_id!r}:")
                    or stripped.startswith(f"{json.dumps(node_id)}:")
                    for node_id in node_ids
                ):
                    removed = True
                    continue
                filtered.append(line)
            if not removed:
                return False
            _ = plugin.write_text("".join(filtered), encoding="utf-8")
            return True

        removed = await asyncio.to_thread(remove_entries)
        if removed:
            self._validate_or_raise(PLUGIN_PATH)
            parse_cmd = "python -c " + shlex.quote(
                "exec(open('python/cudf/cudf/pandas/scripts/pandas-testing-plugin.py').read())"
            )
            parse_result = await asyncio.to_thread(
                run_command, parse_cmd, worktree_path, 60
            )
            if not parse_result.get("success"):
                raise RuntimeError(
                    "pandas-testing-plugin.py failed to parse after xfail removal: "
                    + str(
                        parse_result.get("stderr")
                        or parse_result.get("stdout")
                    )
                )
        return removed

    async def _verify_fix(
        self, worktree_path: str, test_group: TestGroup, gpu_id: int
    ) -> list[TestResult]:
        results: list[TestResult] = []
        node_ids = self._node_ids(test_group)

        for node_id in node_ids:
            results.append(
                await self._run_test_async(
                    worktree_path,
                    node_id,
                    gpu_id,
                    run_mode="verify",
                    fail_on_fallback=False,
                )
            )
        for node_id in node_ids:
            results.append(
                await self._run_test_async(
                    worktree_path,
                    node_id,
                    gpu_id,
                    run_mode="verify",
                    fail_on_fallback=True,
                )
            )

        return results

    async def _restore_removed_xfails(
        self, worktree_path: str, modified_files: list[str]
    ) -> None:
        if PLUGIN_PATH not in modified_files:
            return
        result = await asyncio.to_thread(
            run_command,
            "GIT_MASTER=1 git restore -- " + shlex.quote(PLUGIN_PATH),
            worktree_path,
            120,
        )
        if result.get("success"):
            modified_files.remove(PLUGIN_PATH)
            return
        LOGGER.warning(
            "Failed to restore %s after non-successful fix attempt: %s",
            PLUGIN_PATH,
            result.get("stderr")
            or result.get("stdout")
            or result.get("error"),
        )

    async def _run_group_tests(
        self,
        worktree_path: str,
        test_group: TestGroup,
        gpu_id: int,
        *,
        run_mode: Literal["baseline", "verify"],
        fail_on_fallback: bool | None,
    ) -> list[TestResult]:
        return [
            await self._run_test_async(
                worktree_path,
                node_id,
                gpu_id,
                run_mode=run_mode,
                fail_on_fallback=fail_on_fallback,
            )
            for node_id in self._node_ids(test_group)
        ]

    async def _run_test_async(
        self,
        worktree_path: str,
        node_id: str,
        gpu_id: int,
        *,
        run_mode: Literal["baseline", "verify"],
        fail_on_fallback: bool | None,
    ) -> TestResult:
        return await asyncio.to_thread(
            run_test,
            worktree_path,
            node_id,
            gpu_id,
            run_mode=run_mode,
            fail_on_fallback=fail_on_fallback,
            timeout_minutes=self.config.single_test_timeout_minutes,
            flakiness_reruns=(
                self.config.flakiness_reruns if run_mode == "verify" else 0
            ),
        )

    async def _checkout_fix_branch(
        self, worktree_path: str, branch_name: str
    ) -> None:
        command = "git checkout -B " + shlex.quote(branch_name)
        result = await asyncio.to_thread(
            run_command, command, worktree_path, 120
        )
        if not result.get("success"):
            raise RuntimeError(
                "Failed to create/switch fix branch: "
                + str(
                    result.get("stderr")
                    or result.get("stdout")
                    or result.get("error")
                )
            )

    async def _commit_success(
        self,
        worktree_path: str,
        branch_name: str,
        modified_files: list[str],
        test_group: TestGroup,
    ) -> None:
        files_to_commit = [
            path for path in dict.fromkeys(modified_files) if path
        ]
        if not files_to_commit:
            raise RuntimeError(
                "No modified files to commit after successful verification."
            )
        for path in files_to_commit:
            self._validate_or_raise(path)
        add_cmd = "git add " + " ".join(
            shlex.quote(path) for path in files_to_commit
        )
        add_result = await asyncio.to_thread(
            run_command, add_cmd, worktree_path, 120
        )
        if not add_result.get("success"):
            raise RuntimeError(
                "git add failed: " + str(add_result.get("stderr"))
            )
        message = (
            "fix(cudf.pandas): "
            + test_group.base_name.split("::")[-1].split("[")[0]
            + "\n\nFixes pandas compatibility test group:\n- "
            + "\n- ".join(self._node_ids(test_group))
        )
        commit_cmd = "git commit -m " + shlex.quote(message)
        commit_result = await asyncio.to_thread(
            run_command, commit_cmd, worktree_path, 120
        )
        if not commit_result.get("success"):
            raise RuntimeError(
                f"git commit failed on {branch_name}: "
                + str(
                    commit_result.get("stderr") or commit_result.get("stdout")
                )
            )

    async def _git_modified_files(self, worktree_path: str) -> list[str]:
        result = await asyncio.to_thread(
            run_command, "git diff --name-only", worktree_path, 60
        )
        if not result.get("success"):
            return []
        return [
            line.strip()
            for line in str(result.get("stdout", "")).splitlines()
            if line.strip()
        ]

    async def _load_skill_text(self, worktree_path: str) -> str:
        path = Path(worktree_path) / SKILL_PATH
        return await asyncio.to_thread(path.read_text, encoding="utf-8")

    async def _read_test_source(
        self, worktree_path: str, file_path: str
    ) -> str:
        rel_path = self._test_source_path(file_path)
        content = await asyncio.to_thread(
            read_file, str(Path(worktree_path) / rel_path)
        )
        return self._truncate(content)

    def _test_source_path(self, file_path: str) -> str:
        clean = file_path.removeprefix("pandas-testing/pandas-tests/")
        return str(Path("pandas-testing") / "pandas-tests" / clean)

    def _node_ids(self, test_group: TestGroup) -> list[str]:
        return test_group.node_ids or [test_group.base_name]

    def _module_dir(self, file_path: str) -> str:
        clean = file_path.removeprefix("pandas-testing/pandas-tests/")
        return str(Path(clean).parent) + "/"

    def _summarize_failures(self, results: list[TestResult]) -> str:
        failing = [
            result
            for result in results
            if result.outcome != TestOutcome.PASSED
        ]
        if not failing:
            return "No failing test result details were captured."
        details: list[str] = []
        for result in failing[:3]:
            excerpt = (
                (result.longrepr or result.stdout or "")
                .strip()
                .replace("\n", " ")
            )
            if len(excerpt) > 500:
                excerpt = excerpt[:250] + " ... " + excerpt[-250:]
            details.append(
                f"{result.node_id}: {result.outcome.value}; {excerpt}"
            )
        if len(failing) > len(details):
            details.append(
                f"{len(failing) - len(details)} additional failing result(s) omitted."
            )
        return " ".join(details)

    def _human_review_blocker_reason(
        self,
        results: list[TestResult],
        *,
        phase: str,
        prefix: str = "",
    ) -> str:
        text = "\n".join(
            (result.longrepr or "") + "\n" + str(result.stdout or "")
            for result in results
            if result.outcome != TestOutcome.PASSED
        )
        if not text:
            return ""
        has_global_import_marker = any(
            marker in text for marker in HUMAN_REVIEW_BLOCKER_MARKERS
        )
        has_import_exception = any(
            marker in text for marker in HUMAN_REVIEW_EXCEPTION_MARKERS
        )
        if not (has_global_import_marker and has_import_exception):
            return ""

        summary = prefix or self._summarize_failures(results)
        tail = text.strip()[-1000:].replace("\n", " ")
        return (
            f"Human review required: global cudf.pandas import/environment blocker "
            f"during {phase}; pytest did not reach target-test validation. "
            f"{summary} Traceback tail: {tail}"
        )

    def _validate_or_raise(self, path: str) -> None:
        result = validate_patch([path])
        if result.approved:
            return
        reason = result.reason
        if any(
            marker in reason
            for marker in (
                "C++ files not permitted",
                "pylibcudf files not permitted",
                "Compiled extension files not permitted",
                "CMake files not permitted",
            )
        ):
            raise HumanReviewRequired(reason)
        raise ValueError(reason)

    def _normalize_relative_path(self, path: str, worktree_path: str) -> str:
        raw = Path(path).expanduser()
        worktree = Path(worktree_path).resolve()
        if raw.is_absolute():
            resolved = raw.resolve()
            try:
                return resolved.relative_to(worktree).as_posix()
            except ValueError as exc:
                raise ValueError(f"Path is outside worktree: {path}") from exc
        return raw.as_posix().lstrip("./") or "."

    def _parse_tool_calls(self, response: str) -> list[dict[str, Any]]:
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        return [
            item
            for item in parsed
            if isinstance(item, dict) and "function" in item
        ]

    def _parse_arguments(self, raw_arguments: Any) -> dict[str, Any]:
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if isinstance(raw_arguments, str):
            return json.loads(raw_arguments or "{}")
        return {}

    def _mentions_stop_condition(self, text: str) -> bool:
        lower_text = text.lower()
        return any(
            marker in lower_text for marker in PROHIBITED_RESPONSE_MARKERS
        )

    def _command_looks_mutating(self, command: str) -> bool:
        stripped = command.strip()
        if stripped.startswith(("git commit", "git add", "git checkout")):
            return True
        return any(marker in stripped for marker in MUTATING_COMMAND_MARKERS)

    def _truncate(self, content: str) -> str:
        if len(content) <= MAX_TOOL_CONTENT_CHARS:
            return content
        return (
            content[:4000]
            + "\n\n... <truncated middle> ...\n\n"
            + content[-4000:]
        )

    def _all_passed(self, results: list[TestResult]) -> bool:
        return bool(results) and all(
            result.outcome == TestOutcome.PASSED for result in results
        )

    def _result_to_dict(self, result: TestResult) -> dict[str, Any]:
        return {
            "node_id": result.node_id,
            "outcome": result.outcome.value,
            "duration": result.duration,
            "longrepr": self._truncate(result.longrepr),
            "stdout": self._truncate(str(result.stdout)),
        }

    def _write_diagnostic(
        self,
        test_group: TestGroup | None,
        fix_result: FixResult | None,
        worktree_path: str,
    ) -> None:
        try:
            # Output directory: {repo_root}/pandas_compat_pipeline/diagnostic_logs/
            repo_root = Path(__file__).resolve().parents[3]
            out_dir = repo_root / "pandas_compat_pipeline" / "diagnostic_logs"
            out_dir.mkdir(parents=True, exist_ok=True)

            safe_name = (
                sanitize_branch_name(test_group.base_name).replace("/", "_")
                if test_group
                else "unknown"
            )
            out_path = out_dir / f"diag_{safe_name}.json"
            tmp_path = out_dir / f"diag_{safe_name}.json.tmp"

            payload = {
                "test_name": test_group.base_name if test_group else "unknown",
                "test_group": {
                    "base_name": test_group.base_name,
                    "file_path": test_group.file_path,
                    "node_ids": test_group.node_ids,
                    "reasons": test_group.reasons,
                }
                if test_group
                else {},
                "xfail_removed": self._diag_xfail_removed,
                "baseline_all_passed": self._diag_baseline_passed,
                "final_status": fix_result.status
                if fix_result
                else "exception",
                "final_rejection_reason": fix_result.rejection_reason
                if fix_result
                else "unknown",
                "diagnosis": fix_result.diagnosis if fix_result else "",
                "modified_files": fix_result.modified_files
                if fix_result
                else [],
                "attempts_detail": self._diag_cycles,
                "verification_results": self._diag_verify,
                "elapsed_seconds": time.time() - self._diag_start_time,
            }

            tmp_path.write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
            tmp_path.rename(out_path)
        except Exception:
            pass  # NEVER let diagnostic failures affect the pipeline


__all__ = ["FixResult", "FixerAgent"]
