#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# shellcheck source=ci/build_wheel_stage_common.sh
source ./ci/build_wheel_stage_common.sh

build_noarch_wheel() {
  local package_key=$1
  local package_name=$2
  local package_dir=$3
  local max_wheel_size=$4

  set_wheel_output_dir "${package_key}"
  ./ci/build_wheel.sh "${package_name}" "${package_dir}"
  cp "${package_dir}"/dist/* "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}/"
  ./ci/validate_wheel.sh "${package_dir}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" "${max_wheel_size}"
  record_wheel_artifact \
    "${package_key}" \
    "$(rapids-artifact-name wheel_python "${package_name}" cudf --pure --arch any --cuda "${RAPIDS_CUDA_VERSION}")"
}

build_noarch_wheel dask_cudf dask-cudf python/dask_cudf 10M
build_noarch_wheel cudf_polars cudf-polars python/cudf_polars 10M
