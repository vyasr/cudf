# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pydantic import ValidationInfo

try:
    from pydantic import (
        BaseModel,
        ConfigDict,
        field_validator,
        model_validator,
    )
except (
    ModuleNotFoundError
) as exc:  # pragma: no cover - dependency enforced by runtime env
    raise ModuleNotFoundError(
        "pydantic is required to use pandas_compat_pipeline.src.config"
    ) from exc


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "pandas_compat_pipeline/config/default.yaml"
ENV_PREFIX = "PIPELINE_"


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gpus: int = 8
    integration_gpu: int | None = None

    max_fix_attempts: int = 5
    flakiness_reruns: int = 3

    integration_trigger_every_n: int = 10
    pytest_parallelism: int = 16
    pytest_parallelism_fallback: int = 12

    branch_prefix: str = "pandas-compat/"

    pilot_max_groups: int = 200

    fixer_model: str = "openai/aws/anthropic/bedrock-claude-opus-4-6"
    reviewer_model: str = "openai/openai/openai/gpt-5.5"
    inference_hub_url: str = "https://inference-api.nvidia.com/v1"

    fixer_timeout_minutes: int = 30
    reviewer_timeout_minutes: int = 10
    integration_timeout_minutes: int = 120
    single_test_timeout_minutes: int = 30

    postgres_url: str = "postgresql://cudf:cudf@localhost:5432/langgraph"

    worktree_base_path: str = "~/local/worktrees/pandas-fix"
    cuda_env_file: str = "conda/environments/all_cuda-129_arch-x86_64.yaml"
    plugin_path: str = (
        "python/cudf/cudf/pandas/scripts/pandas-testing-plugin.py"
    )
    test_runner_script: str = (
        "python/cudf/cudf/pandas/scripts/run-pandas-tests.sh"
    )

    max_concurrent_llm_calls: int = 4

    log_level: str = "INFO"
    evidence_dir: str = ".sisyphus/evidence"

    @field_validator("gpus", "max_fix_attempts", "integration_trigger_every_n")
    @classmethod
    def validate_positive(cls, value: int, info: ValidationInfo) -> int:
        if value <= 0:
            raise ValueError(f"{info.field_name} must be > 0")
        return value

    @model_validator(mode="after")
    def set_integration_gpu(self) -> "PipelineConfig":
        if self.integration_gpu is None:
            self.integration_gpu = self.gpus - 1
        if self.integration_gpu >= self.gpus:
            raise ValueError("integration_gpu must be < gpus")
        return self


def _coerce_env_value(raw_value: str, field_name: str) -> int | str:
    annotation = PipelineConfig.model_fields[field_name].annotation
    if annotation is int:
        return int(raw_value)
    return raw_value


def _env_overrides() -> dict[str, int | str]:
    overrides: dict[str, int | str] = {}
    for field_name in PipelineConfig.model_fields:
        env_name = f"{ENV_PREFIX}{field_name.upper()}"
        raw_value = os.getenv(env_name)
        if raw_value is not None:
            overrides[field_name] = _coerce_env_value(raw_value, field_name)
    return overrides


def load_config(path: str | os.PathLike[str] | None = None) -> PipelineConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as file_obj:
        loaded = yaml.safe_load(file_obj) or {}

    if not isinstance(loaded, dict):
        raise ValueError(
            f"Config file must contain a YAML mapping: {config_path}"
        )

    merged_config: dict[str, Any] = {**loaded, **_env_overrides()}
    return PipelineConfig.model_validate(merged_config)
