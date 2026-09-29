#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

: "${CUDF_IWYU_LOG_FILE:?CUDF_IWYU_LOG_FILE must be set}"
: "${CUDF_IWYU_REAL_EXE:?CUDF_IWYU_REAL_EXE must be set}"

iwyu_args=("$@")

log_invocation() {
  local event="$1"
  local status="${2:-}"

  {
    printf '%s event=%s pid=%s ppid=%s status=%s command=' "$(date -u +%FT%TZ)" "${event}" "$$" "${PPID}" "${status}"
    printf '%q ' "${CUDF_IWYU_REAL_EXE}" "${iwyu_args[@]}"
    printf '\n'
  } >> "${CUDF_IWYU_LOG_FILE}"
}

log_invocation start
"${CUDF_IWYU_REAL_EXE}" "$@"
status=$?
log_invocation end "${status}"
exit "${status}"
