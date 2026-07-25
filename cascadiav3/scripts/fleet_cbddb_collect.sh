#!/usr/bin/env bash
set -euo pipefail

# Report or collect whatever CBDDB shards currently exist. Missing hosts do not
# block useful results, and files are not checked against a ledger or checksum.

CYCLE_TAG="${1:?usage: fleet_cbddb_collect.sh <cycle_tag> [status|collect]}"
ACTION="${2:-status}"
HOSTS="${HOSTS:-john1 john2 john3 john4}"
JOHN0="${JOHN0:-}"
JOHN0_ROOT="${JOHN0_ROOT:-/home/john0/cascadia}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGING="$ROOT/cascadiav3/fleet/staging_${CYCLE_TAG}"
mkdir -p "$STAGING"
read -r -a host_list <<< "$HOSTS"

collected=()
for host in "${host_list[@]}"; do
  tag="cbddb_${CYCLE_TAG}_shard_${host}"
  remote="~/cascadia/cascadiav3/fixtures"
  if ssh "$host" "test -s $remote/${tag}_manifest.json"; then
    echo "[$host] ready"
    if [ "$ACTION" = collect ]; then
      for suffix in tensor.npz manifest.json decisions.jsonl; do
        if ssh "$host" "test -s $remote/${tag}_${suffix}"; then
          rsync -a "$host:$remote/${tag}_${suffix}" "$STAGING/"
        fi
      done
      collected+=("$host")
    fi
  else
    echo "[$host] no output yet"
  fi
done

if [ "$ACTION" = collect ]; then
  echo "[fleet:$CYCLE_TAG] collected ${#collected[@]} shard(s) into $STAGING"
  if [ -n "$JOHN0" ] && [ "${#collected[@]}" -gt 0 ]; then
    rsync -a "$STAGING/" "$JOHN0:$JOHN0_ROOT/cascadiav3/fixtures/"
    echo "[fleet:$CYCLE_TAG] copied available shards to $JOHN0"
  fi
fi
