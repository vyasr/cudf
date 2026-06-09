#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_NAME="${PIPELINE_CONDA_ENV:-${CONDA_DEFAULT_ENV:-all_cuda-129_arch-x86_64}}"
PIPELINE_BUILD_ROOT="${PIPELINE_BUILD_ROOT:-/tmp/opencode/cudf-task20-build}"

if [[ -t 1 ]]; then
  GREEN='\033[0;32m'
  BLUE='\033[0;34m'
  RED='\033[0;31m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  GREEN=''
  BLUE=''
  RED=''
  BOLD=''
  NC=''
fi

step() {
  printf "%b\n" "${BLUE}➡${NC} ${BOLD}$1${NC}"
}

success() {
  printf "%b\n" "${GREEN}✅ $1${NC}"
}

fail() {
  printf "%b\n" "${RED}❌ $1${NC}" >&2
  exit 1
}

step "Verifying conda environment ${ENV_NAME} exists"
conda env list | grep -E "^[[:space:]]*${ENV_NAME}[[:space:]]" >/dev/null 2>&1 || fail "Conda environment ${ENV_NAME} does not exist. Run setup-conda-env.sh first."

step "Building libcudf, pylibcudf, and cudf from ${REPO_ROOT}"
mkdir -p "${PIPELINE_BUILD_ROOT}"
conda run -n "${ENV_NAME}" bash -lc 'cd "'"${REPO_ROOT}"'" && LIB_BUILD_DIR="'"${PIPELINE_BUILD_ROOT}"'/libcudf" ./build.sh libcudf pylibcudf cudf --pydevelop'
success "Base build complete"
