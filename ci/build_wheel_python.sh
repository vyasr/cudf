#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

BASE_PIP_CONSTRAINT="$(mktemp)"
cp "${PIP_CONSTRAINT}" "${BASE_PIP_CONSTRAINT}"
trap 'rm -f "${BASE_PIP_CONSTRAINT}"' EXIT

build_wheel() {
  local package_key=$1
  local build_script=$2

  unset RAPIDS_PACKAGE_NAME
  cp "${BASE_PIP_CONSTRAINT}" "${PIP_CONSTRAINT}"
  export RAPIDS_WHEEL_BLD_OUTPUT_DIR="${PWD}/wheel-output/${package_key}"
  mkdir -p "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

  # shellcheck disable=SC1090
  source "${build_script}"

  {
    echo "${package_key}_artifact_name=${RAPIDS_PACKAGE_NAME}"
    echo "${package_key}_output_dir=${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
  } >> "${GITHUB_OUTPUT}"
}

build_wheel pylibcudf ci/build_wheel_pylibcudf.sh
export RAPIDS_PYLIBCUDF_WHEELHOUSE="${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

build_wheel cudf ci/build_wheel_cudf.sh
build_wheel cudf_streaming ci/build_wheel_cudf_streaming.sh
