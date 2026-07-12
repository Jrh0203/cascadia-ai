#!/usr/bin/env bash
set -u
# Deliberately no `set -e`: a failing stage must not kill the queue (its
# failure is logged loudly and the queue continues). Every critical setup
# step therefore checks its own result via preflight().

# Experiment-queue runner: executes the stages of a JSONL queue file
# sequentially on john0 so the GPU never idles between preregistered
# experiments (AGENTS.md saturation rule) while preserving "one scientific
# job at a time". Each queue line is one stage:
#
#   {"name": "<stage>", "script": "cascadiav3/scripts/run_x.sh",
#    "env": {"KEY": "value", ...}}
#
# Usage (see cascadiav3/queues/README.md for the detached launch pattern):
#   SOURCE_REVISION=<rev> bash cascadiav3/scripts/run_experiment_queue.sh \
#     cascadiav3/queues/<queue>.jsonl
#
# Contract:
#   - Fail closed: the entire queue file is validated (JSONL shape, stage
#     names, env-key hygiene, script existence) via the unit-tested
#     cascadiav3.experiment_queue module BEFORE any stage runs.
#   - Pause: touch cascadiav3/logs/HOLD_experiment_queue to hold the queue
#     before its next stage (heartbeats continue while holding); remove it
#     to resume. A stage that is already running is not interrupted — use
#     the stage's own HOLD_<name> file for intra-stage gates.
#   - Idempotent resume: a stage is skipped while its done marker
#     cascadiav3/logs/queue_done_<name> exists; delete the marker to rerun.
#   - Stage failure is loud but non-fatal: it is heartbeat-logged with the
#     exit code and the queue continues to the next stage. The runner exits
#     1 at the end if any stage failed (summary table always prints).
#   - Stage stdout/stderr (build output included — never silenced) append to
#     cascadiav3/logs/queue_<name>.log; the stage pid is written next to it
#     in cascadiav3/logs/queue_<name>.pid.
#   - SOURCE_REVISION is pinned by the runner for every stage; a queue entry
#     cannot override it (reserved key, rejected at validation).
#
# WAITER_POLL_SECONDS overrides the heartbeat/poll interval (default 60).

ROOT="${ROOT:-/home/john0/cascadia}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION to the deployed revision}"
PYTHON="${PYTHON:-python3}"
LOG_DIR="${LOG_DIR:-cascadiav3/logs}"
DEPLOYED_REVISION_FILE="${DEPLOYED_REVISION_FILE:-$LOG_DIR/exact_k1_deployed_revision.txt}"
QUEUE_FILE="${1:?usage: SOURCE_REVISION=<rev> bash run_experiment_queue.sh <queue-file>}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="cascadiav3/src"

preflight() {
  local label="$1"
  shift
  if ! "$@"; then
    echo "[experiment-queue] preflight failed: $label" >&2
    exit 1
  fi
}

preflight "cannot cd to ROOT: $ROOT" cd "$ROOT"
preflight "waiter library missing: cascadiav3/scripts/lib_waiter.sh" \
  test -f cascadiav3/scripts/lib_waiter.sh
WAITER_LOG_DIR="$LOG_DIR"
# The explicit "sourced" argument keeps lib_waiter's selftest branch off
# regardless of this script's own positional parameters.
# shellcheck disable=SC1091
source cascadiav3/scripts/lib_waiter.sh sourced

mkdir -p "$LOG_DIR"
preflight "queue file missing or empty: $QUEUE_FILE" test -s "$QUEUE_FILE"
preflight "python interpreter not found: $PYTHON" command -v "$PYTHON" >/dev/null 2>&1
preflight "cascadiav3.experiment_queue not importable (PYTHONPATH=$PYTHONPATH)" \
  "$PYTHON" -c "import cascadiav3.experiment_queue"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [ "$(git rev-parse HEAD)" != "$SOURCE_REVISION" ]; then
    echo "[experiment-queue] preflight failed: SOURCE_REVISION does not match HEAD" >&2
    exit 1
  fi
elif [ ! -s "$DEPLOYED_REVISION_FILE" ] \
  || [ "$(tr -d '[:space:]' < "$DEPLOYED_REVISION_FILE")" != "$SOURCE_REVISION" ]; then
  echo "[experiment-queue] preflight failed: source snapshot lacks the deployed revision marker" >&2
  exit 1
fi

# One scientific job at a time: refuse to start while another queue runner
# is live. A stale pid file (dead pid) is normal and does not block.
RUNNER_PID_FILE="$LOG_DIR/queue_runner.pid"
if [ -s "$RUNNER_PID_FILE" ] && kill -0 "$(cat "$RUNNER_PID_FILE")" 2>/dev/null; then
  echo "[experiment-queue] preflight failed: another queue runner is live" \
    "(pid $(cat "$RUNNER_PID_FILE"), $RUNNER_PID_FILE)" >&2
  exit 1
fi
echo "$$" > "$RUNNER_PID_FILE"

# Fail-closed validation and shell hand-off: the heredoc only wires the
# unit-tested cascadiav3.experiment_queue module to the shell. Any error
# (printed to stderr) rejects the whole queue before any stage runs.
STAGES="$("$PYTHON" - "$QUEUE_FILE" <<'PY'
import sys

from cascadiav3.experiment_queue import missing_scripts, parse_queue, shell_stage_line

try:
    stages = parse_queue(sys.argv[1])
except ValueError as exc:
    raise SystemExit(f"queue validation failed: {exc}")
missing = missing_scripts(stages)
if missing:
    raise SystemExit("queue references missing scripts: " + ", ".join(missing))
for stage in stages:
    print(shell_stage_line(stage))
PY
)" || {
  echo "[experiment-queue] preflight failed: queue file invalid: $QUEUE_FILE (see error above)" >&2
  exit 1
}

STAGE_COUNT="$(printf '%s\n' "$STAGES" | grep -c .)"
waiter_heartbeat experiment_queue \
  "queue validated: $STAGE_COUNT stages from $QUEUE_FILE (source=$SOURCE_REVISION)"

run_stage_script() {
  # $1 = env prefix produced by cascadiav3.experiment_queue.shell_env — keys
  #      regex-checked, values single-quoted, control characters rejected —
  #      which is what makes this eval safe. $2 = stage script path.
  # SOURCE_REVISION is appended after the queue env so it always wins.
  eval "env $1 SOURCE_REVISION=\"\$SOURCE_REVISION\" bash \"\$2\""
}

total=0
completed=0
failed=0
skipped=0
NAMES=()
RESULTS=()

while IFS=$'\t' read -r stage_name stage_script stage_env; do
  [ -n "$stage_name" ] || continue
  total=$((total + 1))
  waiter_gate experiment_queue
  done_marker="$LOG_DIR/queue_done_${stage_name}"
  stage_log="$LOG_DIR/queue_${stage_name}.log"
  stage_pid_file="$LOG_DIR/queue_${stage_name}.pid"
  if [ -e "$done_marker" ]; then
    waiter_heartbeat experiment_queue "$stage_name SKIPPED (done marker: $done_marker)"
    NAMES+=("$stage_name")
    RESULTS+=(SKIPPED)
    skipped=$((skipped + 1))
    continue
  fi
  waiter_heartbeat experiment_queue "$stage_name STARTING (script=$stage_script log=$stage_log)"
  printf '\n[experiment-queue] %s stage %s starting (source=%s script=%s)\n' \
    "$(date '+%F %T')" "$stage_name" "$SOURCE_REVISION" "$stage_script" >> "$stage_log"
  run_stage_script "$stage_env" "$stage_script" >> "$stage_log" 2>&1 &
  stage_pid=$!
  echo "$stage_pid" > "$stage_pid_file"
  while kill -0 "$stage_pid" 2>/dev/null; do
    waiter_heartbeat experiment_queue "$stage_name running (pid $stage_pid, log $stage_log)"
    sleep "$WAITER_POLL_SECONDS"
  done
  wait "$stage_pid"
  stage_status=$?
  if [ "$stage_status" -eq 0 ]; then
    printf '%s %s\n' "$(date '+%F %T')" "$SOURCE_REVISION" > "$done_marker"
    waiter_heartbeat experiment_queue "$stage_name COMPLETE"
    NAMES+=("$stage_name")
    RESULTS+=(COMPLETE)
    completed=$((completed + 1))
  else
    waiter_heartbeat experiment_queue "$stage_name FAILED exit=$stage_status (see $stage_log)"
    NAMES+=("$stage_name")
    RESULTS+=(FAILED)
    failed=$((failed + 1))
  fi
done <<< "$STAGES"

echo
echo "[experiment-queue] $(date '+%F %T') queue summary: $QUEUE_FILE"
printf '  %-40s %s\n' "STAGE" "STATUS"
i=0
while [ "$i" -lt "${#NAMES[@]}" ]; do
  printf '  %-40s %s\n' "${NAMES[$i]}" "${RESULTS[$i]}"
  i=$((i + 1))
done
waiter_heartbeat experiment_queue \
  "queue finished: $completed complete, $failed failed, $skipped skipped (of $total)"

if [ "$failed" -gt 0 ]; then
  exit 1
fi
exit 0
