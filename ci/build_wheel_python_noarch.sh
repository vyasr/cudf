#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

build_wheel() {
  local package_key=$1
  local build_script=$2

  unset RAPIDS_PACKAGE_NAME
  export RAPIDS_WHEEL_BLD_OUTPUT_DIR="${PWD}/wheel-output/${package_key}"
  mkdir -p "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

  # shellcheck disable=SC1090
  source "${build_script}"

  {
    echo "${package_key}_artifact_name=${RAPIDS_PACKAGE_NAME}"
    echo "${package_key}_output_dir=${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
  } >> "${GITHUB_OUTPUT}"
}

build_wheel dask_cudf ci/build_wheel_dask_cudf.sh
build_wheel cudf_polars ci/build_wheel_cudf_polars.sh
