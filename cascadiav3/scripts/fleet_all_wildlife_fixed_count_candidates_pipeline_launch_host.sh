#!/usr/bin/env bash
set -euo pipefail

PIPELINE_TAG="${PIPELINE_TAG:?set PIPELINE_TAG}"
SHARD_HOST="${SHARD_HOST:?set SHARD_HOST}"

case "$PIPELINE_TAG:$SHARD_HOST" in
  *[!A-Za-z0-9._:-]*)
    echo "PIPELINE_TAG and SHARD_HOST must be safe identifiers" >&2
    exit 64
    ;;
esac

ROOT="${HOME}/cascadia"
LOG_DIR="${ROOT}/cascadiav3/logs"
PIPELINE="${ROOT}/cascadiav3/scripts/fleet_all_wildlife_fixed_count_candidates_pipeline.sh"
LOG="${LOG_DIR}/all_wildlife_fixed_pipeline_${PIPELINE_TAG}_${SHARD_HOST}.log"
PID_FILE="${LOG_DIR}/all_wildlife_fixed_pipeline_${PIPELINE_TAG}_${SHARD_HOST}.pid"

cd "$ROOT"
mkdir -p "$LOG_DIR"
test -x "$PIPELINE"
rm -f "$PID_FILE"

/usr/bin/nohup /bin/bash "$PIPELINE" > "$LOG" 2>&1 < /dev/null &
nohup_pid=$!
printf '%s\n' "$nohup_pid" > "$PID_FILE"
printf '%s\n' "$nohup_pid"
