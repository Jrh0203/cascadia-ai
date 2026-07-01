#!/bin/zsh
set -euo pipefail

readonly ROOT="${CASCADIA_CLUSTER_ROOT:?CASCADIA_CLUSTER_ROOT is required}"
readonly PID_FILE="$ROOT/state/bacalhau-supervisor.pid"
if [[ -s "$PID_FILE" ]]; then
  fallback_pid="$(<"$PID_FILE")"
  if [[ "$fallback_pid" =~ '^[0-9]+$' ]] && kill -0 "$fallback_pid" 2>/dev/null; then
    kill "$fallback_pid" 2>/dev/null || true
    sleep 2
  fi
  rm -f "$PID_FILE"
fi
exec "$ROOT/bin/run-node.zsh"
