#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

install_build_requirements() {
  local package_name=$1

  rapids-logger "Generating build requirements for '${package_name}'"
  rapids-dependency-file-generator \
    --output requirements \
    --file-key "py_build_${package_name}" \
    --file-key "py_rapids_build_${package_name}" \
    --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};cuda_suffixed=true;use_cuda_wheels=true" \
  | tee /tmp/requirements-build.txt

  rapids-logger "Installing build requirements for '${package_name}'"
  rapids-pip-retry install \
    -v \
    --prefer-binary \
    -r /tmp/requirements-build.txt
}

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

# libcudf
set_wheel_output_dir libcudf
install_build_requirements libcudf
export SKBUILD_CMAKE_ARGS="-DUSE_NVCOMP_RUNTIME_WHEEL=ON"
./ci/build_wheel.sh libcudf python/libcudf

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
python -m auditwheel repair \
  --exclude libkvikio.so \
  --exclude libnvcomp.so.5 \
  --exclude librapids_logger.so \
  --exclude librmm.so \
  --exclude "libnvrtc.so.${RAPIDS_CUDA_MAJOR}" \
  --exclude "libnvJitLink.so.${RAPIDS_CUDA_MAJOR}" \
  -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
  python/libcudf/dist/*

WHEEL_EXPORT_DIR="$(mktemp -d)"
unzip -d "${WHEEL_EXPORT_DIR}" "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"/*
LIBCUDF_LIBRARY="$(find "${WHEEL_EXPORT_DIR}" -type f -name libcudf.so)"
./ci/check_symbols.sh "${LIBCUDF_LIBRARY}"

./ci/validate_wheel.sh python/libcudf "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
record_wheel_artifact libcudf "$(rapids-artifact-name wheel_cpp libcudf cudf --cuda "${RAPIDS_CUDA_VERSION}")"

# libcudf-streaming. Its distinct scikit-build project consumes the freshly built
# libcudf wheel from this stage rather than downloading it as a CI artifact.
LIBCUDF_WHEELHOUSE="${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
set_wheel_output_dir libcudf_streaming

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
echo "libcudf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBCUDF_WHEELHOUSE}"/libcudf_*.whl)" >> "${PIP_CONSTRAINT}"

install_build_requirements libcudf_streaming
unset SKBUILD_CMAKE_ARGS
./ci/build_wheel.sh libcudf_streaming python/libcudf_streaming

python -m auditwheel repair \
  --exclude libcudf.so \
  --exclude librapidsmpf.so \
  --exclude librapids_logger.so \
  --exclude librmm.so \
  --exclude libucxx.so \
  --exclude libucp.so.0 \
  -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
  python/libcudf_streaming/dist/*

./ci/validate_wheel.sh python/libcudf_streaming "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
record_wheel_artifact \
  libcudf_streaming \
  "$(rapids-artifact-name wheel_cpp libcudf-streaming cudf --cuda "${RAPIDS_CUDA_VERSION}")"
