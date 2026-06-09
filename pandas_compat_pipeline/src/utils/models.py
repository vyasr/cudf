# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data models for pandas xfail grouping."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TestVariant:
    """Single failing test node variant."""

    node_id: str
    parametrization: str | None
    reason: str


@dataclass(slots=True)
class TestGroup:
    """Grouped failing test variants sharing the same base test."""

    base_name: str
    file_path: str
    class_name: str | None
    parametrizations: list[str] = field(default_factory=list)
    weight: int = 0
    reasons: list[str] = field(default_factory=list)
    node_ids: list[str] = field(default_factory=list)
