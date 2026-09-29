#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail

: "${CUDF_IWYU_LOG_FILE:?CUDF_IWYU_LOG_FILE must be set}"
: "${CUDF_IWYU_REAL_EXE:?CUDF_IWYU_REAL_EXE must be set}"
: "${CUDF_IWYU_DIAGNOSTICS_DIR:?CUDF_IWYU_DIAGNOSTICS_DIR must be set}"

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
time_log="${CUDF_IWYU_DIAGNOSTICS_DIR}/iwyu.${BASHPID}.time"
time_exe="$(dirname "${CUDF_IWYU_REAL_EXE}")/time"
if [[ ! -x "${time_exe}" ]]; then
  printf 'GNU time is unavailable at %s\n' "${time_exe}" >&2
  exit 1
fi
trace_iwyu=0
for arg in "${iwyu_args[@]}"; do
  if [[ "${arg}" =~ ${CUDF_IWYU_STRACE_SOURCE_REGEX:-^$} ]]; then
    trace_iwyu=1
    break
  fi
done

if (( trace_iwyu )); then
  trace_prefix="${CUDF_IWYU_DIAGNOSTICS_DIR}/iwyu.${BASHPID}.strace"
  "${time_exe}" -v -o "${time_log}" strace -ff -ttt -T -s 256 -o "${trace_prefix}" \
    -e trace=%file,%process,%network "${CUDF_IWYU_REAL_EXE}" "$@"
else
  "${time_exe}" -v -o "${time_log}" "${CUDF_IWYU_REAL_EXE}" "$@"
fi
status=$?
log_invocation end "${status}"
exit "${status}"
