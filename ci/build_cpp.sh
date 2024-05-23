#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-generate-version > ./VERSION

rapids-logger "Begin cpp build"

# With boa installed conda build forward to boa
RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) rapids-conda-retry mambabuild \
    conda/recipes/libcudf

rapids-upload-conda-to-s3 cpp
