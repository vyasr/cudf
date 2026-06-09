#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_NAME="${PIPELINE_CONDA_ENV:-${CONDA_DEFAULT_ENV:-all_cuda-129_arch-x86_64}}"
WORKTREE_BASE="${WORKTREE_BASE:-/raid/vyasr/local/worktrees/pandas-fix}"
NUM_WORKERS=8

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --num-workers=*)
      NUM_WORKERS="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--num-workers N]" >&2
      exit 1
      ;;
  esac
done

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

ensure_pandas_testing() {
  local worker_dir="$1"
  local worker_label="$2"
  local pandas_tests_dir="${worker_dir}/pandas-testing/pandas-tests"
  local bootstrap_script="${worker_dir}/python/cudf/cudf/pandas/scripts/run-pandas-tests.sh"

  if [[ -d "${pandas_tests_dir}/tests" && -f "${pandas_tests_dir}/pyproject.toml" ]]; then
    warn "pandas-testing for ${worker_label} already exists; skipping bootstrap"
    return 0
  fi

  if [[ ! -x "${bootstrap_script}" ]]; then
    fail "Cannot bootstrap pandas-testing for ${worker_label}; missing ${bootstrap_script}"
  fi

  step "Bootstrapping pandas-testing for ${worker_label}"
  warn "This uses the worker-local run-pandas-tests.sh harness and may clone pandas on first run"
  set +e
  (
    cd "${worker_dir}"
    "${bootstrap_script}" --collect-only --maxfail=1 tests/series/test_api.py
  )
  local bootstrap_status=$?
  set -e

  if [[ -d "${pandas_tests_dir}/tests" && -f "${pandas_tests_dir}/pyproject.toml" ]]; then
    if [[ ${bootstrap_status} -ne 0 ]]; then
      warn "pandas-testing bootstrap command exited ${bootstrap_status}, but required test tree exists"
    fi
    success "pandas-testing ready for ${worker_label}"
    return 0
  fi

  fail "pandas-testing bootstrap failed for ${worker_label}; expected ${pandas_tests_dir}/tests"
}

step "Provisioning ${NUM_WORKERS} worktrees at ${WORKTREE_BASE}"

# Ensure base directory exists
mkdir -p "${WORKTREE_BASE}"

# Get existing worktree paths for idempotency check
EXISTING_WORKTREES="$(GIT_MASTER=1 git -C "${REPO_ROOT}" worktree list --porcelain | grep '^worktree ' | awk '{print $2}')"

for i in $(seq 0 $((NUM_WORKERS - 1))); do
  WORKER_DIR="${WORKTREE_BASE}/worker-${i}"

  # --- Create worktree ---
  step "Setting up worker-${i} at ${WORKER_DIR}"

  if echo "${EXISTING_WORKTREES}" | grep -qF "${WORKER_DIR}"; then
    warn "Worktree worker-${i} already exists; skipping creation"
  else
    GIT_MASTER=1 git -C "${REPO_ROOT}" worktree add "${WORKER_DIR}" HEAD
    success "Created worktree worker-${i}"
  fi

  # --- Create venv ---
  if [[ -d "${WORKER_DIR}/.venv" ]]; then
    warn "Venv for worker-${i} already exists; skipping creation"
  else
    step "Creating venv for worker-${i}"
    conda run -n "${ENV_NAME}" python -m venv --system-site-packages "${WORKER_DIR}/.venv"
    success "Created venv for worker-${i}"
  fi

  # --- Editable install of cudf ---
  step "Installing cudf (editable) into worker-${i} venv"
  (
    source "${WORKER_DIR}/.venv/bin/activate"
    pip install --no-build-isolation --no-deps -e "${REPO_ROOT}/python/cudf/"
  )
  success "Installed cudf into worker-${i}"

  # --- Worker-local pandas test infrastructure ---
  ensure_pandas_testing "${WORKER_DIR}" "worker-${i}"

  # --- GPU assignment ---
  echo "${i}" > "${WORKER_DIR}/.gpu"

done

# --- Validation ---
step "Validating worktrees"
for i in $(seq 0 $((NUM_WORKERS - 1))); do
  WORKER_DIR="${WORKTREE_BASE}/worker-${i}"
  (
    source "${WORKER_DIR}/.venv/bin/activate"
    python -c "import cudf; print('worker-${i} GPU=${i} cudf=' + cudf.__version__)"
  )
  [[ -d "${WORKER_DIR}/pandas-testing/pandas-tests/tests" ]] || fail "worker-${i} missing pandas-testing tests"
  [[ -f "${WORKER_DIR}/pandas-testing/pandas-tests/pyproject.toml" ]] || fail "worker-${i} missing pandas-testing pyproject.toml"
done

success "All ${NUM_WORKERS} worktrees provisioned and validated"
