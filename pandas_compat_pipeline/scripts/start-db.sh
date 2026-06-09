#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
MAX_WAIT_SECONDS=60
START_TIME=$(date +%s)

docker compose -f "${COMPOSE_FILE}" up -d

while true; do
  HEALTH_STATUS=$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}unavailable{{end}}' cudf-pandas-fix-postgres 2>/dev/null || true)

  if [[ "${HEALTH_STATUS}" == "healthy" ]]; then
    printf 'PostgreSQL is healthy and ready.\n'
    exit 0
  fi

  CURRENT_TIME=$(date +%s)
  ELAPSED=$((CURRENT_TIME - START_TIME))
  if (( ELAPSED >= MAX_WAIT_SECONDS )); then
    printf 'PostgreSQL failed to become healthy within %s seconds.\n' "${MAX_WAIT_SECONDS}" >&2
    exit 1
  fi

  sleep 2
done
