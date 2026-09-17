#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

source ./ci/test_python_common.sh test_python_other

rapids-logger "Check GPU usage"
nvidia-smi
rapids-print-env

PROFILE_ARGS=()
PROFILE_MONITOR_PID=""
if [[ "${CI_PROFILE:-false}" == "true" ]]; then
  PROFILE_ARGS=(
    "--ci-profile-json=${RAPIDS_TESTS_DIR}/cudf-polars-profile-${RAPIDS_CUDA_VERSION}.json"
    --ci-profile-top-n=100
  )
  profile_gpu_metrics="${RAPIDS_TESTS_DIR}/cudf-polars-gpu-metrics-${RAPIDS_CUDA_VERSION}.csv"
  (
    echo "timestamp,gpu,utilization.gpu [%],utilization.memory [%],memory.used [MiB],clocks.sm [MHz],clocks.mem [MHz],power.draw [W],temperature.gpu"
    while true; do
      nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu \
        --format=csv,noheader,nounits || true
      sleep 5
    done
  ) > "${profile_gpu_metrics}" &
  PROFILE_MONITOR_PID=$!
  trap 'kill "${PROFILE_MONITOR_PID}" 2>/dev/null || true' EXIT
  rapids-logger "CI profiling enabled: ${profile_gpu_metrics}"
fi

rapids-logger "pytest cudf-polars"
# Fail fast (-x) rather than trying to continue because failed tests pollute the state.
./ci/run_cudf_polars_pytests.sh \
  -x \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cudf-polars.xml" \
  --numprocesses=4 \
  --dist=worksteal \
  --cov-config=./pyproject.toml \
  --cov=cudf_polars \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cudf-polars-coverage.xml" \
  --cov-report=term \
  --engine-pool-timings \
  "${PROFILE_ARGS[@]}" \
  --durations=50 --durations-min=1
