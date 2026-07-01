#!/bin/zsh
set -u
umask 077

readonly ROOT="${CASCADIA_CLUSTER_ROOT:?CASCADIA_CLUSTER_ROOT is required}"
readonly PID_FILE="$ROOT/state/bacalhau-supervisor.pid"
print -r -- $$ >| "$PID_FILE"
trap 'rm -f "$PID_FILE"; exit 0' TERM INT EXIT

delay=2
while true; do
  "$ROOT/bin/run-node.zsh"
  status=$?
  print -u2 "Bacalhau exited with status $status; restart in ${delay}s"
  sleep "$delay"
  (( delay = delay < 30 ? delay * 2 : 30 ))
done
