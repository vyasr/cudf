#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

# Produce a stable, machine-readable size report for a linked libcudf shared library.
#
# `cuobjdump` is intentionally mandatory: host ELF size alone cannot identify repeated CUDA
# kernels. The SASS byte counts are instruction-address spans (16 bytes per SASS instruction),
# grouped by GPU architecture and mangled kernel name. A repeated (architecture, kernel) pair is
# a candidate for duplicate device code from multiple translation units.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/libcudf.so" >&2
  exit 2
fi

libcudf=$1
if [[ ! -f ${libcudf} ]]; then
  echo "libcudf does not exist: ${libcudf}" >&2
  exit 2
fi

for tool in readelf nm cuobjdump; do
  if ! command -v "${tool}" >/dev/null; then
    echo "Missing required CUDA binary-inspection tool '${tool}'. Install a CUDA toolkit that provides cuobjdump." >&2
    exit 127
  fi
done

echo "# library=${libcudf}"
echo "# elf_bytes=$(stat --format=%s "${libcudf}")"
echo "# nv_fatbin_bytes=$(readelf --section-headers --wide "${libcudf}" | awk '$2 == ".nv_fatbin" { print "0x" $6 }' | xargs -r printf '%d')"
echo "# Host symbols with repeated names (not CUDA-device deduplication):"
nm --defined-only --size-sort --demangle "${libcudf}" |
  awk '$2 ~ /^[TtWw]$/ { count[$4]++; bytes[$4] += strtonum("0x" $1) }
       END { for (name in count) if (count[name] > 1) print bytes[name] "\t" count[name] "\t" name }' |
  sort --numeric-sort --reverse | head -n 50 || true

echo "# Device SASS kernel instances: architecture, estimated_bytes, occurrences, kernel"
cuobjdump --dump-sass "${libcudf}" |
  awk '
    function emit() {
      if (function_name != "" && max_offset >= 0) {
        print architecture "\t" (max_offset + 16) "\t" function_name
      }
    }
    /^code for / {
      emit(); function_name = ""; max_offset = -1; architecture = $3; next
    }
    /^[[:space:]]*Function[[:space:]]*:/ {
      emit(); function_name = $0; sub(/^[^:]*:[[:space:]]*/, "", function_name); max_offset = -1; next
    }
    /\/\*[0-9A-Fa-f]+\*\// && function_name != "" {
      line = $0; sub(/^.*\/\*/, "", line); sub(/\*\/.*$/, "", line)
      offset = strtonum("0x" line); if (offset > max_offset) max_offset = offset
    }
    END { emit() }
  ' |
  sort | awk -F '\t' '
    { key = $1 SUBSEP $3; bytes[key] += $2; occurrences[key]++ }
    END {
      for (key in occurrences) {
        split(key, fields, SUBSEP)
        print fields[1] "\t" bytes[key] "\t" occurrences[key] "\t" fields[2]
      }
    }
  ' | sort --key=2,2nr --key=3,3nr
