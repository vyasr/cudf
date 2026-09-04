#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

set_wheel_output_dir() {
  local package_key=$1

  export RAPIDS_WHEEL_BLD_OUTPUT_DIR="${PWD}/wheel-output/${package_key}"
  mkdir -p "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
}

record_wheel_artifact() {
  local package_key=$1
  local package_name=$2
  {
    echo "${package_key}_artifact_name=${package_name}"
    echo "${package_key}_output_dir=${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
  } >> "${GITHUB_OUTPUT}"
}

build_noarch_wheel() {
  local package_key=$1
  local package_name=$2
  local package_dir=$3

  set_wheel_output_dir "${package_key}"
  ./ci/build_wheel.sh "${package_name}" "${package_dir}"
  cp "${package_dir}"/dist/* "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}/"
  ./ci/validate_wheel.sh "${package_dir}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
  record_wheel_artifact \
    "${package_key}" \
    "$(rapids-artifact-name wheel_python "${package_name}" cudf --pure --arch any --cuda "${RAPIDS_CUDA_VERSION}")"
}

build_noarch_wheel dask_cudf dask-cudf python/dask_cudf
build_noarch_wheel cudf_polars cudf-polars python/cudf_polars
