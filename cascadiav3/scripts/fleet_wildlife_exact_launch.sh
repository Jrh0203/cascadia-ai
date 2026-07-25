#!/usr/bin/env bash
set -euo pipefail

# Deploy and launch independent exact-catalog shards across the Mac fleet.
# Reruns are allowed. The current files are copied directly; no git, source
# revision, hash, collision, idleness, receipt, or single-use-tag checks.

RULESET="${RULESET:?set RULESET (aaaaa or cbddb)}"
FLEET_TAG="${FLEET_TAG:?set FLEET_TAG}"
CANDIDATES="${CANDIDATES:?set CANDIDATES}"
COUNTS_FILE="${COUNTS_FILE:?set COUNTS_FILE}"
IMPORT_CATALOG="${IMPORT_CATALOG:-${IMPORT_LEDGER:-}}"
HOSTS="${HOSTS:-john1 john2 john3 john4}"
REMOTE_ROOT="${REMOTE_ROOT:-~/cascadia}"
WILDLIFE_VENV="${WILDLIFE_VENV:-wildlife-venv-py312}"
LOCAL_WILDLIFE_VENV="${LOCAL_WILDLIFE_VENV:-.venv}"
JOBS="${JOBS:-2}"
SOLVER_WORKERS="${SOLVER_WORKERS:-4}"
RELAXATION_TIME_LIMIT="${RELAXATION_TIME_LIMIT:-60}"
CONNECTED_TIME_LIMIT="${CONNECTED_TIME_LIMIT:-120}"
BASE_SEED="${BASE_SEED:-1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

test -s "$CANDIDATES"
test -s "$COUNTS_FILE"
read -r -a host_list <<< "$HOSTS"
shard_count="${#host_list[@]}"
[ "$shard_count" -gt 0 ]

is_local_host() {
  case "$1" in
    john1|localhost|127.0.0.1|"$(hostname -s)") return 0 ;;
    *) return 1 ;;
  esac
}

for index in "${!host_list[@]}"; do
  host="${host_list[$index]}"
  echo "[fleet:$FLEET_TAG] deploying shard $index/$shard_count to $host"
  if is_local_host "$host"; then
    input_dir="$ROOT/cascadiav3/fleet_inputs/$FLEET_TAG"
    mkdir -p "$input_dir" "$ROOT/cascadiav3/fleet_outputs/$FLEET_TAG" "$ROOT/cascadiav3/logs"
    cp "$CANDIDATES" "$input_dir/candidates.json"
    cp "$COUNTS_FILE" "$input_dir/counts.json"
    if [ -n "$IMPORT_CATALOG" ] && [ -s "$IMPORT_CATALOG" ]; then
      cp "$IMPORT_CATALOG" "$input_dir/import_catalog.json"
    else
      rm -f "$input_dir/import_catalog.json"
    fi
    log="$ROOT/cascadiav3/logs/wildlife_${FLEET_TAG}_shard_${host}.log"
    pid_file="$ROOT/cascadiav3/logs/wildlife_${FLEET_TAG}_shard_${host}.pid"
    nohup env \
      ROOT="$ROOT" RULESET="$RULESET" FLEET_TAG="$FLEET_TAG" SHARD_HOST="$host" \
      SHARD_INDEX="$index" SHARD_COUNT="$shard_count" JOBS="$JOBS" \
      SOLVER_WORKERS="$SOLVER_WORKERS" RELAXATION_TIME_LIMIT="$RELAXATION_TIME_LIMIT" \
      CONNECTED_TIME_LIMIT="$CONNECTED_TIME_LIMIT" BASE_SEED="$BASE_SEED" \
      WILDLIFE_VENV="$LOCAL_WILDLIFE_VENV" \
      bash "$ROOT/cascadiav3/scripts/fleet_wildlife_exact_worker.sh" \
      >"$log" 2>&1 < /dev/null &
    echo $! > "$pid_file"
  else
    ssh "$host" "mkdir -p $REMOTE_ROOT/cascadiav3/fleet_inputs/$FLEET_TAG \
      $REMOTE_ROOT/cascadiav3/fleet_outputs/$FLEET_TAG $REMOTE_ROOT/cascadiav3/logs"
    rsync -a "$CANDIDATES" "$host:$REMOTE_ROOT/cascadiav3/fleet_inputs/$FLEET_TAG/candidates.json"
    rsync -a "$COUNTS_FILE" "$host:$REMOTE_ROOT/cascadiav3/fleet_inputs/$FLEET_TAG/counts.json"
    if [ -n "$IMPORT_CATALOG" ] && [ -s "$IMPORT_CATALOG" ]; then
      rsync -a "$IMPORT_CATALOG" "$host:$REMOTE_ROOT/cascadiav3/fleet_inputs/$FLEET_TAG/import_catalog.json"
    else
      ssh "$host" "rm -f $REMOTE_ROOT/cascadiav3/fleet_inputs/$FLEET_TAG/import_catalog.json"
    fi
    rsync -a "$ROOT/tools/" "$host:$REMOTE_ROOT/tools/"
    rsync -a "$ROOT/cascadiav3/scripts/fleet_wildlife_exact_worker.sh" \
      "$host:$REMOTE_ROOT/cascadiav3/scripts/fleet_wildlife_exact_worker.sh"
    ssh "$host" "cd $REMOTE_ROOT && nohup env \
      RULESET='$RULESET' FLEET_TAG='$FLEET_TAG' SHARD_HOST='$host' \
      SHARD_INDEX='$index' SHARD_COUNT='$shard_count' JOBS='$JOBS' \
      SOLVER_WORKERS='$SOLVER_WORKERS' RELAXATION_TIME_LIMIT='$RELAXATION_TIME_LIMIT' \
      CONNECTED_TIME_LIMIT='$CONNECTED_TIME_LIMIT' BASE_SEED='$BASE_SEED' \
      WILDLIFE_VENV='$WILDLIFE_VENV' \
      bash cascadiav3/scripts/fleet_wildlife_exact_worker.sh \
      > cascadiav3/logs/wildlife_${FLEET_TAG}_shard_${host}.log 2>&1 < /dev/null & \
      echo \$! > cascadiav3/logs/wildlife_${FLEET_TAG}_shard_${host}.pid"
  fi
done

echo "[fleet:$FLEET_TAG] launched $shard_count shards"
