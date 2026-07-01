#!/bin/zsh

set -euo pipefail

readonly REPOSITORY="/Users/johnherrick/cascadia"
readonly CAMPAIGN_ROOT="/Users/john2/cascadia-bench/r2-map-v1"
readonly CODEX_BIN="/Users/johnherrick/.local/bin/codex"
readonly SESSION_ID="${R2_MAP_CODEX_SESSION_ID:-019ed7e5-29b8-7d32-a9e4-ab466f9d75e6}"
readonly MAX_TURNS="${R2_MAP_HEADLESS_MAX_TURNS:-200}"
readonly START_DELAY_SECONDS="${R2_MAP_HEADLESS_START_DELAY_SECONDS:-20}"
readonly PROMPT_PATH="${REPOSITORY}/tools/r2_map_headless_resume_prompt.txt"
readonly CONTINUATION_PATH="${REPOSITORY}/tools/r2_map_headless_continuation_prompt.txt"
readonly TERMINAL_RELATIVE="control/headless-terminal.json"
readonly STOP_RELATIVE="control/headless-STOP"
readonly LOCK_NAME="headless-runner"
readonly LOCK_OWNER="john1-headless-$$"
readonly LOCK_LEASE_SECONDS=3600
readonly SESSION_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
readonly REMOTE_TOOL="${REPOSITORY}/tools/r2_map_remote_storage.py"
readonly TURN_RUNNER="${REPOSITORY}/tools/r2_map_headless_turn.py"
typeset -ar REMOTE_CLI=(
  /usr/bin/env
  PYTHONDONTWRITEBYTECODE=1
  PYTHONPATH="${REPOSITORY}/python"
  "${REPOSITORY}/.venv/bin/python"
  "${REMOTE_TOOL}"
)

umask 077
typeset lock_token=""
integer remote_lock_held=0
integer event_sequence=0

remote_put_stream() {
  trap - EXIT INT TERM HUP
  local relative="$1"
  local maximum_bytes="$2"
  "${REMOTE_CLI[@]}" put-stream \
    --relative "${relative}" \
    --max-bytes "${maximum_bytes}" \
    --expected-current absent >/dev/null
}

remote_read_optional() {
  trap - EXIT INT TERM HUP
  "${REMOTE_CLI[@]}" fetch \
    --relative "$1" \
    --window-bytes 65536 2>/dev/null
}

remote_event() {
  local label="$1"
  local message="$2"
  (( event_sequence += 1 ))
  print -r -- "${message}" | remote_put_stream \
    "logs/headless/${SESSION_RUN_ID}/event-$(printf '%04d' "${event_sequence}")-${label}.log" \
    65536
}

extract_token() {
  trap - EXIT INT TERM HUP
  PYTHONDONTWRITEBYTECODE=1 "${REPOSITORY}/.venv/bin/python" -c \
    'import json,sys; value=json.load(sys.stdin); token=value.get("token"); assert isinstance(token,str) and len(token)==64; print(token)'
}

extract_turn_exit() {
  trap - EXIT INT TERM HUP
  PYTHONDONTWRITEBYTECODE=1 "${REPOSITORY}/.venv/bin/python" -c '
import json
import sys

value = json.load(sys.stdin)
code = value.get("codex_exit_code")
assert value.get("sinks_verified") is True
assert isinstance(code, int) and not isinstance(code, bool) and 0 <= code <= 255
sinks = value.get("sinks")
assert isinstance(sinks, list) and len(sinks) == 2
assert len({sink.get("relative") for sink in sinks if isinstance(sink, dict)}) == 2
print(code)
'
}

bound_diagnostic() {
  trap - EXIT INT TERM HUP
  PYTHONDONTWRITEBYTECODE=1 "${REPOSITORY}/.venv/bin/python" -c '
import sys

payload = sys.stdin.buffer.read(4097)
sys.stdout.buffer.write(payload[:4096])
if len(payload) > 4096:
    sys.stdout.buffer.write(b"\n[diagnostic truncated]")
'
}

acquire_remote_lock() {
  local receipt
  receipt="$("${REMOTE_CLI[@]}" lock acquire \
    --name "${LOCK_NAME}" \
    --owner "${LOCK_OWNER}" \
    --lease-epoch "acquire-${SESSION_RUN_ID}" \
    --lease-seconds "${LOCK_LEASE_SECONDS}")"
  lock_token="$(print -r -- "${receipt}" | extract_token)"
  remote_lock_held=1
}

renew_remote_lock() {
  local epoch="$1"
  "${REMOTE_CLI[@]}" lock renew \
    --name "${LOCK_NAME}" \
    --owner "${LOCK_OWNER}" \
    --token "${lock_token}" \
    --lease-epoch "${epoch}" \
    --lease-seconds "${LOCK_LEASE_SECONDS}" >/dev/null
}

lock_heartbeat() {
  trap - EXIT INT TERM HUP
  integer renewal_epoch=0
  while sleep 300; do
    (( renewal_epoch += 1 ))
    renew_remote_lock "renew-${SESSION_RUN_ID}-${renewal_epoch}" || return 74
  done
}

cleanup() {
  if (( remote_lock_held == 1 )); then
    "${REMOTE_CLI[@]}" lock release \
      --name "${LOCK_NAME}" \
      --owner "${LOCK_OWNER}" \
      --token "${lock_token}" \
      --lease-epoch "release-${SESSION_RUN_ID}" \
      --lease-seconds "${LOCK_LEASE_SECONDS}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

if [[ ! -x "${CODEX_BIN}" ]]; then
  print -u2 "Codex CLI is not executable: ${CODEX_BIN}"
  exit 69
fi
if [[ ! -d "${REPOSITORY}/.git" ]]; then
  print -u2 "Repository is absent: ${REPOSITORY}"
  exit 72
fi
if [[ ! -f "${PROMPT_PATH}" || ! -f "${CONTINUATION_PATH}" ]]; then
  print -u2 "Headless prompt files are absent"
  exit 66
fi
if [[ ! -f "${REMOTE_TOOL}" ]]; then
  print -u2 "R2-MAP remote-storage operator is absent: ${REMOTE_TOOL}"
  exit 66
fi
if [[ ! -f "${TURN_RUNNER}" ]]; then
  print -u2 "R2-MAP headless turn runner is absent: ${TURN_RUNNER}"
  exit 66
fi

"${REMOTE_CLI[@]}" install-worker >/dev/null
"${REMOTE_CLI[@]}" preflight >/dev/null
acquire_remote_lock
print -r -- "R2-MAP headless session ${SESSION_RUN_ID}; all artifacts stream to ${CAMPAIGN_ROOT}"
print -r -- \
  "{\"campaign_root\":\"${CAMPAIGN_ROOT}\",\"runner_host\":\"john1\",\"pid\":$$,\"session_run_id\":\"${SESSION_RUN_ID}\",\"started_utc\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" \
  | remote_put_stream "control/headless/owners/${SESSION_RUN_ID}.json" 65536

sleep "${START_DELAY_SECONDS}"

integer turn=1
integer consecutive_failures=0
typeset run_id events_relative stderr_relative prompt_path turn_result diagnostic exit_code runner_status value heartbeat_pid
integer heartbeat_failed=0
while (( turn <= MAX_TURNS )); do
  renew_remote_lock
  if value="$(remote_read_optional "${STOP_RELATIVE}")"; then
    remote_event "stop" "stop requested before turn ${turn}"
    exit 0
  fi
  if value="$(remote_read_optional "${TERMINAL_RELATIVE}")" && \
    print -r -- "${value}" | /usr/bin/grep -Eq \
      '"status"[[:space:]]*:[[:space:]]*"(complete|authorization-blocked)"'; then
    remote_event "terminal" "terminal sentinel observed before turn ${turn}"
    exit 0
  fi

  run_id="$(printf '%04d' "${turn}")"
  events_relative="logs/headless/${SESSION_RUN_ID}/turn-${run_id}.jsonl"
  stderr_relative="logs/headless/${SESSION_RUN_ID}/turn-${run_id}.stderr.log"
  if (( turn == 1 )); then
    prompt_path="${PROMPT_PATH}"
  else
    prompt_path="${CONTINUATION_PATH}"
  fi

  remote_event "turn-${run_id}-start" \
    "starting turn ${turn} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  lock_heartbeat &
  heartbeat_pid=$!
  set +e
  turn_result="$(
    PYTHONDONTWRITEBYTECODE=1 "${REPOSITORY}/.venv/bin/python" "${TURN_RUNNER}" \
      --codex-bin "${CODEX_BIN}" \
      --remote-python "${REPOSITORY}/.venv/bin/python" \
      --remote-tool "${REMOTE_TOOL}" \
      --python-root "${REPOSITORY}/python" \
      --repository "${REPOSITORY}" \
      --session-id "${SESSION_ID}" \
      --prompt-path "${prompt_path}" \
      --events-relative "${events_relative}" \
      --stderr-relative "${stderr_relative}" \
      --events-max-bytes 268435456 \
      --stderr-max-bytes 67108864 \
      --heartbeat-pid "${heartbeat_pid}" \
      2>&1
  )"
  runner_status=$?
  set -e

  heartbeat_failed=0
  if kill -0 "${heartbeat_pid}" 2>/dev/null; then
    kill "${heartbeat_pid}" 2>/dev/null || true
    wait "${heartbeat_pid}" 2>/dev/null || true
  else
    wait "${heartbeat_pid}" 2>/dev/null || heartbeat_failed=1
  fi

  if (( heartbeat_failed != 0 )); then
    remote_event "turn-${run_id}-lock-failure" \
      "remote lock heartbeat failed during turn ${turn}"
    print -u2 "Remote lock heartbeat failed during turn ${turn}"
    exit 74
  fi

  if (( runner_status != 0 )); then
    diagnostic="$(print -r -- "${turn_result}" | bound_diagnostic)"
    remote_event "turn-${run_id}-stream-failure" \
      "verified John2 artifact stream failed during turn ${turn} status=${runner_status}
${diagnostic}"
    print -u2 "A verified John2 artifact stream failed during turn ${turn}"
    exit "${runner_status}"
  fi
  if ! exit_code="$(print -r -- "${turn_result}" | extract_turn_exit)"; then
    remote_event "turn-${run_id}-receipt-failure" \
      "headless turn runner returned malformed sink evidence during turn ${turn}"
    print -u2 "Headless turn runner returned malformed sink evidence"
    exit 74
  fi

  renew_remote_lock
  remote_event "turn-${run_id}-finish" \
    "finished turn ${turn} exit=${exit_code} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if (( exit_code == 0 )); then
    consecutive_failures=0
  else
    (( consecutive_failures += 1 ))
    if (( consecutive_failures >= 3 )); then
      remote_event "failure" "three consecutive Codex failures; operator action required"
      exit "${exit_code}"
    fi
    sleep $(( 10 * consecutive_failures ))
  fi
  (( turn += 1 ))
done

remote_event "turn-limit" "maximum headless turns exhausted without a terminal sentinel"
exit 75
