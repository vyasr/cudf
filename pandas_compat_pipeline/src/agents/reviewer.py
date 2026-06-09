# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM-powered reviewer for cudf.pandas compatibility fixes."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal, cast

from ..utils.patch_validator import validate_patch
from .llm_client import LLMClient

logger = logging.getLogger(__name__)

ReviewVerdict = Literal["approved", "rejected", "needs_human"]
VALID_VERDICTS: set[str] = {"approved", "rejected", "needs_human"}


@dataclass(slots=True)
class ReviewResult:
    verdict: ReviewVerdict
    feedback: str
    concerns: list[str]


class ReviewerAgent:
    """Review proposed cudf.pandas fixes before they are committed.

    The reviewer is intentionally read-only: callers provide the diff, test node
    id, diagnosis, and worktree path; this class only validates paths/content and
    asks the reviewer LLM for a structured judgment.
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client: LLMClient | None = llm_client

    async def review(
        self,
        diff: str,
        test_node_id: str,
        diagnosis: str,
        worktree_path: str,
    ) -> ReviewResult:
        """Review a fix diff and return a structured verdict.

        Args:
            diff: Full git diff for the proposed fix.
            test_node_id: pandas pytest node id being fixed.
            diagnosis: Fixer agent's diagnosis/root-cause explanation.
            worktree_path: Worktree containing the fix; included for context only.

        Returns:
            ReviewResult with verdict, feedback, and concerns.
        """
        modified_files = _extract_modified_files(diff)
        if not modified_files:
            return ReviewResult(
                verdict="needs_human",
                feedback=(
                    "No modified files could be extracted from the diff; "
                    "human review is needed to determine what changed."
                ),
                concerns=["Unable to validate an empty or non-standard diff."],
            )

        validation = validate_patch(modified_files, diff_content=diff)
        if not validation.approved:
            return ReviewResult(
                verdict="needs_human",
                feedback=(
                    "Programmatic patch validation rejected this fix before LLM "
                    f"review: {validation.reason}. Human attention is required "
                    "because policy violations cannot be auto-approved."
                ),
                concerns=validation.violations or [validation.reason],
            )

        messages = _build_review_messages(
            diff=diff,
            test_node_id=test_node_id,
            diagnosis=diagnosis,
            worktree_path=worktree_path,
            modified_files=modified_files,
            validation_classification=validation.classification,
        )

        try:
            response = await self._client.call_reviewer(messages)
        except Exception as exc:  # pragma: no cover - exercised by integration
            logger.exception("Reviewer LLM call failed")
            return ReviewResult(
                verdict="needs_human",
                feedback=(
                    "Reviewer LLM call failed, so the fix cannot be safely "
                    f"auto-reviewed: {exc}. Human attention is required."
                ),
                concerns=[f"LLM reviewer error: {exc}"],
            )

        return _parse_review_response(response)

    @property
    def _client(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client


def _extract_modified_files(diff: str) -> list[str]:
    """Extract destination paths from git diff ``+++ b/...`` lines."""
    modified_files: list[str] = []
    seen: set[str] = set()

    for line in diff.splitlines():
        if not line.startswith("+++ b/"):
            continue
        path = line.removeprefix("+++ b/").strip()
        if not path or path == "/dev/null" or path in seen:
            continue
        seen.add(path)
        modified_files.append(path)

    return modified_files


def _build_review_messages(
    *,
    diff: str,
    test_node_id: str,
    diagnosis: str,
    worktree_path: str,
    modified_files: list[str],
    validation_classification: str,
) -> list[dict[str, str]]:
    system_prompt = """You are the read-only code reviewer for cudf.pandas compatibility fixes.

Review the proposed fix for correctness, maintainability, and policy compliance. You must reject fixes with ANY prohibited pattern:
- Test-specific special cases, including pattern-matching on a specific pytest node id, test name, parameter value, dtype/shape combination, or fixture value instead of fixing the general API contract.
- CPU fallback as a fix strategy, including removing CUDA execution paths, intentionally raising to force pandas execution, or returning pandas objects from cudf APIs.
- Private pandas API imports or calls, including `from pandas._libs import ...`, `from pandas.core import ...`, `pandas.compat`, or any underscored pandas module.
- `pyarrow.compute` as an execution backend for cudf behavior.
- Modifications to files under `pandas-testing/pandas-tests/`.
- Additions to `NODEIDS_TO_SKIP` or `NODEIDS_PATHS_TO_SKIP`; only removals from `NODEIDS_THAT_FAIL` are acceptable.
- Broad refactoring or unrelated cleanup while fixing; the fix must be minimal and targeted.
- New Python dependencies introduced in pyproject, requirements, setup metadata, or imports that require a new dependency.
- C++/CUDA/CMake/pylibcudf/libcudf binding changes for this pandas-compat pipeline.

Verdict rules:
- `approved`: the diff is minimal, targeted, policy-compliant, and plausibly fixes the diagnosed root cause.
- `rejected`: the fix is unsafe/wrong or violates a prohibited pattern. Feedback must give specific, actionable guidance for the fixer's next attempt.
- `needs_human`: use only when the diff cannot be confidently reviewed automatically, the evidence is ambiguous, or manual judgment is required. Feedback must explain exactly why human attention is needed.

Output strict JSON only, with no markdown fences or prose outside JSON:
{"verdict":"approved|rejected|needs_human","feedback":"...","concerns":["..."]}
"""

    user_prompt = f"""Review this cudf.pandas compatibility fix.

Worktree path (context only; do not access it):
{worktree_path}

Test node id being fixed:
{test_node_id}

Fixer diagnosis:
{diagnosis}

Programmatic patch validation: approved
Validation classification: {validation_classification}
Modified files:
{json.dumps(modified_files, indent=2)}

Full git diff:
```diff
{diff}
```

Return only JSON with fields `verdict`, `feedback`, and `concerns`.
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _parse_review_response(response: str) -> ReviewResult:
    payload = _loads_review_json(response)
    verdict = _coerce_verdict(
        payload.get("verdict") if payload else None, response
    )
    feedback = _coerce_feedback(
        payload.get("feedback") if payload else None, verdict
    )
    concerns = _coerce_concerns(payload.get("concerns") if payload else None)

    if not payload:
        concerns.append("Reviewer response was not valid JSON.")
        if verdict == "approved":
            verdict = "needs_human"
            feedback = (
                "Reviewer response could not be parsed as valid JSON; human "
                "attention is required before approving this fix."
            )

    return ReviewResult(verdict=verdict, feedback=feedback, concerns=concerns)


def _loads_review_json(response: str) -> dict[str, object]:
    text = _strip_markdown_fences(response).strip()
    try:
        data: object = json.loads(text)  # pyright: ignore[reportAny]
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))  # pyright: ignore[reportAny]
        except json.JSONDecodeError:
            return {}

    return cast(dict[str, object], data) if isinstance(data, dict) else {}


def _strip_markdown_fences(response: str) -> str:
    text = response.strip()
    fence = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE
    )
    return fence.group(1) if fence else text


def _coerce_verdict(raw_verdict: object, response: str) -> ReviewVerdict:
    if isinstance(raw_verdict, str):
        verdict = raw_verdict.strip().lower()
        if verdict == "approved":
            return "approved"
        if verdict == "rejected":
            return "rejected"
        if verdict == "needs_human":
            return "needs_human"

    match = re.search(
        r"\b(approved|rejected|needs_human)\b", response, flags=re.IGNORECASE
    )
    if match:
        verdict = match.group(1).lower()
        if verdict == "approved":
            return "approved"
        if verdict == "rejected":
            return "rejected"
        return "needs_human"

    return "needs_human"


def _coerce_feedback(raw_feedback: object, verdict: ReviewVerdict) -> str:
    if isinstance(raw_feedback, str) and raw_feedback.strip():
        return raw_feedback.strip()

    if verdict == "approved":
        return "Approved by reviewer."
    if verdict == "rejected":
        return (
            "Rejected, but the reviewer did not provide actionable guidance; "
            "retry with a minimal, policy-compliant fix that addresses the "
            "general API contract rather than the specific test case."
        )
    return (
        "Reviewer did not provide enough structured feedback to make an "
        "automatic decision; human attention is required."
    )


def _coerce_concerns(raw_concerns: object) -> list[str]:
    if isinstance(raw_concerns, list):
        concerns = [
            str(item).strip()
            for item in cast(list[object], raw_concerns)
            if str(item).strip()
        ]
        return concerns
    if isinstance(raw_concerns, str) and raw_concerns.strip():
        return [raw_concerns.strip()]
    return []
