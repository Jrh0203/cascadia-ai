#!/usr/bin/env bash
set -euo pipefail

# Launch CBDDB tensor generation across any requested hosts. The current
# checkpoint and worker are copied directly. Reruns and overlapping jobs are
# allowed; no seed ledger, source pin, receipt, or host-role policy is imposed.

CYCLE_TAG="${CYCLE_TAG:?set CYCLE_TAG}"
FIRST_SEED="${FIRST_SEED:?set FIRST_SEED}"
SEEDS_PER_HOST="${SEEDS_PER_HOST:?set SEEDS_PER_HOST}"
INCUMBENT_DIR="${INCUMBENT_DIR:?set INCUMBENT_DIR (local checkpoint dir)}"
HOSTS="${HOSTS:-john1 john2 john3 john4}"
GEN_N_SIMULATIONS="${GEN_N_SIMULATIONS:-128}"
GEN_DETERMINIZATIONS="${GEN_DETERMINIZATIONS:-2}"
SESSIONS="${SESSIONS:-6}"
RAYON_THREADS="${RAYON_THREADS:-8}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

test -s "$INCUMBENT_DIR/best_locked_val.manifest.json"
CKPT_NAME="$(basename "$INCUMBENT_DIR")"
REMOTE_CKPT="cascadiav3/checkpoints/${CKPT_NAME}"
REMOTE_MANIFEST="${REMOTE_CKPT}/best_locked_val.manifest.json"

read -r -a host_list <<< "$HOSTS"
for index in "${!host_list[@]}"; do
  host="${host_list[$index]}"
  first=$((FIRST_SEED + index * SEEDS_PER_HOST))
  echo "[fleet:$CYCLE_TAG] $host seeds=${first}x${SEEDS_PER_HOST}"
  rsync -a "$INCUMBENT_DIR/" "$host:~/cascadia/$REMOTE_CKPT/"
  rsync -a "$ROOT/cascadiav3/scripts/fleet_cbddb_gen.sh" \
    "$host:~/cascadia/cascadiav3/scripts/fleet_cbddb_gen.sh"
  ssh "$host" "cd ~/cascadia && nohup env \
    CYCLE_TAG='$CYCLE_TAG' SHARD_HOST='$host' FIRST_SEED='$first' \
    SEED_COUNT='$SEEDS_PER_HOST' INCUMBENT='$REMOTE_MANIFEST' \
    GEN_N_SIMULATIONS='$GEN_N_SIMULATIONS' \
    GEN_DETERMINIZATIONS='$GEN_DETERMINIZATIONS' \
    SESSIONS='$SESSIONS' RAYON_THREADS='$RAYON_THREADS' \
    bash cascadiav3/scripts/fleet_cbddb_gen.sh \
    > cascadiav3/logs/cbddb_${CYCLE_TAG}_shard_${host}.log 2>&1 < /dev/null &"
done

echo "[fleet:$CYCLE_TAG] launched ${#host_list[@]} shard(s)"
