#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

BASE_PIP_CONSTRAINT="$(mktemp)"
cp "${PIP_CONSTRAINT}" "${BASE_PIP_CONSTRAINT}"
trap 'rm -f "${BASE_PIP_CONSTRAINT}"' EXIT

set_wheel_output_dir() {
  local package_key=$1

  cp "${BASE_PIP_CONSTRAINT}" "${PIP_CONSTRAINT}"
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

add_wheel_constraint() {
  local package_name=$1
  local wheelhouse=$2
  local wheel_pattern=$3

  echo "${package_name}-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${wheelhouse}"/${wheel_pattern})" >> "${PIP_CONSTRAINT}"
}

set_stable_abi() {
  RAPIDS_PY_API="cp${RAPIDS_PY_VERSION//./}"
  export RAPIDS_PY_API
}

check_cython_performance_hints() {
  local package_name=$1
  local build_log=$2

  rapids-logger "Checking for Cython performance warnings"
  if grep -Fq "performance hint:" "${build_log}"; then
    echo "Cython performance hints found in ${package_name} build:"
    grep -F "performance hint:" "${build_log}"
    exit 1
  fi
}

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# pylibcudf
set_wheel_output_dir pylibcudf
LIBCUDF_WHEELHOUSE="$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcudf cudf --cuda "${RAPIDS_CUDA_VERSION}")")"
add_wheel_constraint libcudf "${LIBCUDF_WHEELHOUSE}" 'libcudf_*.whl'
set_stable_abi
./ci/build_wheel.sh pylibcudf python/pylibcudf --stable 2>&1 | tee pylibcudf-wheel-build-output.log
check_cython_performance_hints pylibcudf pylibcudf-wheel-build-output.log

python -m auditwheel repair \
  --exclude libcudf.so \
  --exclude libnvcomp.so.* \
  --exclude libkvikio.so \
  --exclude librapids_logger.so \
  --exclude librmm.so \
  -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
  python/pylibcudf/dist/*

./ci/validate_wheel.sh python/pylibcudf "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
record_wheel_artifact pylibcudf "$(rapids-artifact-name wheel_python pylibcudf cudf --stable --cuda "${RAPIDS_CUDA_VERSION}")"

# cudf
PYLIBCUDF_WHEELHOUSE="${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
set_wheel_output_dir cudf
add_wheel_constraint libcudf "${LIBCUDF_WHEELHOUSE}" 'libcudf_*.whl'
add_wheel_constraint pylibcudf "${PYLIBCUDF_WHEELHOUSE}" 'pylibcudf_*.whl'
set_stable_abi
./ci/build_wheel.sh cudf python/cudf --stable

python -m auditwheel repair \
  --exclude libcudf.so \
  --exclude libnvcomp.so.* \
  --exclude libkvikio.so \
  --exclude librapids_logger.so \
  --exclude librmm.so \
  -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
  python/cudf/dist/*

./ci/validate_wheel.sh python/cudf "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
record_wheel_artifact cudf "$(rapids-artifact-name wheel_python cudf cudf --stable --cuda "${RAPIDS_CUDA_VERSION}")"

# cudf-streaming
set_wheel_output_dir cudf_streaming
LIBCUDF_STREAMING_WHEELHOUSE="${RAPIDS_LIBCUDF_STREAMING_WHEELHOUSE:-$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcudf-streaming cudf --cuda "${RAPIDS_CUDA_VERSION}")")}"
add_wheel_constraint libcudf-streaming "${LIBCUDF_STREAMING_WHEELHOUSE}" 'libcudf_streaming_*.whl'
add_wheel_constraint libcudf "${LIBCUDF_WHEELHOUSE}" 'libcudf_*.whl'
add_wheel_constraint pylibcudf "${PYLIBCUDF_WHEELHOUSE}" 'pylibcudf_*.whl'

set_stable_abi
./ci/build_wheel.sh cudf-streaming python/cudf_streaming --stable 2>&1 | tee cudf-streaming-wheel-build-output.log
check_cython_performance_hints cudf-streaming cudf-streaming-wheel-build-output.log

python -m auditwheel repair \
  --exclude libcudf.so \
  --exclude libcudf_streaming.so \
  --exclude librapidsmpf.so \
  --exclude librapids_logger.so \
  --exclude librmm.so \
  --exclude libucxx.so \
  --exclude libucp.so.0 \
  -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
  python/cudf_streaming/dist/*

./ci/validate_wheel.sh python/cudf_streaming "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
record_wheel_artifact \
  cudf_streaming \
  "$(rapids-artifact-name wheel_python cudf-streaming cudf --stable --cuda "${RAPIDS_CUDA_VERSION}")"
