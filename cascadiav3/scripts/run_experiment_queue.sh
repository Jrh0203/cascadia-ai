#!/usr/bin/env bash
set -u

# Lightweight experiment launcher.
#
# Queue entries may run sequentially (default) or concurrently
# (QUEUE_PARALLEL=1). There are no source pins, receipts, done markers,
# HOLD files, heartbeats, singleton locks, or scientific-job limits.

ROOT="${ROOT:-/home/john0/cascadia}"
PYTHON="${PYTHON:-python3}"
LOG_DIR="${LOG_DIR:-cascadiav3/logs}"
QUEUE_PARALLEL="${QUEUE_PARALLEL:-0}"
QUEUE_FILE="${1:?usage: bash run_experiment_queue.sh <queue-file>}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PYTHONPATH:-cascadiav3/src}"

cd "$ROOT" || exit 1
mkdir -p "$LOG_DIR"

STAGES="$("$PYTHON" -m cascadiav3.experiment_queue "$QUEUE_FILE" --root .)" || exit 1

run_stage() {
  local stage_name="$1"
  local stage_script="$2"
  local stage_env="$3"
  local stage_log="$LOG_DIR/queue_${stage_name}.log"
  echo "[experiment-queue] $(date '+%F %T') $stage_name starting"
  eval "env $stage_env bash \"\$stage_script\"" >"$stage_log" 2>&1 &
  local stage_pid=$!
  LAST_STAGE_PID="$stage_pid"
  echo "$stage_pid" > "$LOG_DIR/queue_${stage_name}.pid"
  STAGE_PIDS+=("$stage_pid")
  STAGE_NAMES+=("$stage_name")
}

STAGE_PIDS=()
STAGE_NAMES=()
LAST_STAGE_PID=""
failures=0

while IFS=$'\t' read -r stage_name stage_script stage_env; do
  [ -n "$stage_name" ] || continue
  run_stage "$stage_name" "$stage_script" "$stage_env"
  if [ "$QUEUE_PARALLEL" != "1" ]; then
    if wait "$LAST_STAGE_PID"; then
      echo "[experiment-queue] $stage_name complete"
    else
      echo "[experiment-queue] $stage_name failed; see $LOG_DIR/queue_${stage_name}.log"
      failures=$((failures + 1))
    fi
    STAGE_PIDS=()
    STAGE_NAMES=()
  fi
done <<< "$STAGES"

if [ "$QUEUE_PARALLEL" = "1" ]; then
  for index in "${!STAGE_PIDS[@]}"; do
    if wait "${STAGE_PIDS[$index]}"; then
      echo "[experiment-queue] ${STAGE_NAMES[$index]} complete"
    else
      echo "[experiment-queue] ${STAGE_NAMES[$index]} failed; see $LOG_DIR/queue_${STAGE_NAMES[$index]}.log"
      failures=$((failures + 1))
    fi
  done
fi

echo "[experiment-queue] finished with $failures failed stage(s)"
[ "$failures" -eq 0 ]
