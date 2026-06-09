#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_NAME="${PIPELINE_CONDA_ENV:-${CONDA_DEFAULT_ENV:-all_cuda-129_arch-x86_64}}"
CONDA_ENV_FILE="${REPO_ROOT}/conda/environments/all_cuda-129_arch-x86_64.yaml"
PIPELINE_BUILD_ROOT="${PIPELINE_BUILD_ROOT:-/tmp/opencode/cudf-task20-build}"
PIP_PACKAGES=(
  langgraph
  langgraph-checkpoint-postgres
  langchain-core
  langchain-anthropic
  langchain-openai
  litellm
  asyncpg
  'psycopg[binary]'
  'pydantic>=2.0'
  pyyaml
  aiofiles
)

if [[ -t 1 ]]; then
  GREEN='\033[0;32m'
  BLUE='\033[0;34m'
  YELLOW='\033[1;33m'
  RED='\033[0;31m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  GREEN=''
  BLUE=''
  YELLOW=''
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

warn() {
  printf "%b\n" "${YELLOW}⚠️  $1${NC}"
}

fail() {
  printf "%b\n" "${RED}❌ $1${NC}" >&2
  exit 1
}

[[ -f "${CONDA_ENV_FILE}" ]] || fail "Conda environment file not found: ${CONDA_ENV_FILE}"

step "Checking for existing conda environment: ${ENV_NAME}"
if conda env list | grep -E "^[[:space:]]*${ENV_NAME}[[:space:]]" >/dev/null 2>&1; then
  warn "Environment ${ENV_NAME} already exists; skipping creation"
else
  step "Creating conda environment ${ENV_NAME} from ${CONDA_ENV_FILE}"
  conda env create -n "${ENV_NAME}" -f "${CONDA_ENV_FILE}"
  success "Created conda environment ${ENV_NAME}"
fi

step "Building libcudf, pylibcudf, and cudf in ${ENV_NAME}"
mkdir -p "${PIPELINE_BUILD_ROOT}"
conda run -n "${ENV_NAME}" bash -c "cd ${REPO_ROOT} && LIB_BUILD_DIR='${PIPELINE_BUILD_ROOT}/libcudf' ./build.sh libcudf pylibcudf cudf --pydevelop"
success "libcudf, pylibcudf, and cudf built"

step "Installing pipeline Python packages into ${ENV_NAME}"
conda run -n "${ENV_NAME}" pip install "${PIP_PACKAGES[@]}"
success "Installed pipeline Python packages"

step "Validating imports inside ${ENV_NAME}"
conda run -n "${ENV_NAME}" python -c "import cudf; import pylibcudf; import langgraph; import litellm; print('✓ All imports OK')"
success "Environment validation passed"

success "Conda environment setup complete"
