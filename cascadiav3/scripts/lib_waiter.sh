#!/usr/bin/env bash
# Shared helpers for queued waiter/chain scripts (AGENTS.md operational rules).
#
# Contract:
#   - Waiter scripts and their pinned inputs live under cascadiav3/logs/,
#     never /tmp, so they are reconstructible after a reboot.
#   - Every stage boundary checks a HOLD file: touch
#     cascadiav3/logs/HOLD_<name> to pause the waiter before its next stage,
#     remove it to resume. Pausing a waiter that is already mid-stage requires
#     kill -STOP <pid> / kill -CONT <pid> (user permission required).
#   - Waiters emit timestamped heartbeats so status checks can verify
#     liveness by log freshness, not just pid existence.
#
# Usage:
#   source cascadiav3/scripts/lib_waiter.sh
#   waiter_wait_for_pids "<name>" <pid>...   # block until all pids exit,
#                                            # then block while HOLD exists
#   waiter_gate "<name>"                     # block while HOLD exists
#   waiter_heartbeat "<name>" "<message>"    # one timestamped log line
#
# WAITER_POLL_SECONDS overrides the poll interval (default 60; tests use a
# sub-second value).

WAITER_LOG_DIR="${WAITER_LOG_DIR:-cascadiav3/logs}"
WAITER_POLL_SECONDS="${WAITER_POLL_SECONDS:-60}"

waiter_heartbeat() {
  local name="$1"
  shift
  printf '[%s] %s heartbeat %s\n' "$(date '+%F %T')" "$name" "$*"
}

waiter_hold_path() {
  printf '%s/HOLD_%s' "$WAITER_LOG_DIR" "$1"
}

waiter_gate() {
  local name="$1"
  local hold
  hold="$(waiter_hold_path "$name")"
  while [ -e "$hold" ]; do
    waiter_heartbeat "$name" "paused by $hold"
    sleep "$WAITER_POLL_SECONDS"
  done
}

waiter_wait_for_pids() {
  local name="$1"
  shift
  while :; do
    local alive=0 pid
    for pid in "$@"; do
      if kill -0 "$pid" 2>/dev/null; then
        alive=1
      fi
    done
    if [ "$alive" -eq 0 ]; then
      break
    fi
    waiter_heartbeat "$name" "waiting on pids: $*"
    sleep "$WAITER_POLL_SECONDS"
  done
  waiter_gate "$name"
}

# Self-test: bash cascadiav3/scripts/lib_waiter.sh selftest
if [ "${1:-}" = "selftest" ]; then
  set -euo pipefail
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  WAITER_LOG_DIR="$tmp"
  WAITER_POLL_SECONDS=0.1

  # 1. waiter_gate blocks while HOLD exists and resumes after removal.
  hold="$(waiter_hold_path selftest)"
  touch "$hold"
  (sleep 0.5 && rm -f "$hold") &
  start=$(date +%s)
  waiter_gate selftest > /dev/null
  [ -e "$hold" ] && { echo "FAIL: gate returned while HOLD present"; exit 1; }

  # 2. waiter_wait_for_pids returns after the watched pid exits.
  sleep 0.4 &
  watched=$!
  waiter_wait_for_pids selftest "$watched" > /dev/null
  kill -0 "$watched" 2>/dev/null && { echo "FAIL: returned with pid alive"; exit 1; }

  # 3. Heartbeats are timestamped.
  line="$(waiter_heartbeat selftest probe)"
  case "$line" in
    \[*\]\ selftest\ heartbeat\ probe) ;;
    *) echo "FAIL: heartbeat format: $line"; exit 1 ;;
  esac
  echo "lib_waiter selftest: OK ($(( $(date +%s) - start ))s)"
fi
