#!/usr/bin/env bash
set -euo pipefail

# CBDDB fleet launcher — runs on the Mac. Allocates contiguous seed
# ranges across the mini fleet, rsyncs the incumbent checkpoint to each
# host, launches fleet_cbddb_gen.sh detached (pid file per host), and
# writes a fleet ledger json that fleet_cbddb_collect.sh consumes.
#
# Prereqs per host (see EXPERIMENT_LOG 2026-07-22): current source
# rsynced to ~/cascadia, exporter built (cargo build --release), venv
# with torch+MPS. john1 note: doubles as web-UI host — include it only
# when the UI is not in use (INFRASTRUCTURE.md §Fleet operations).
#
# Required env:
#   CYCLE_TAG        e.g. fs_c3
#   FIRST_SEED       start of the fleet's overall seed block (fresh,
#                    never used — audit EXPERIMENT_LOG seed ledger first)
#   SEEDS_PER_HOST   contiguous seeds allocated to each host in order
#   INCUMBENT_DIR    LOCAL path to the checkpoint dir (manifest+weights)
#   SOURCE_REVISION  git revision deployed to the hosts
# Optional env:
#   HOSTS            default "john2 john3 john4"
#   GEN_N_SIMULATIONS / GEN_DETERMINIZATIONS / SESSIONS / RAYON_THREADS
#                    passed through to fleet_cbddb_gen.sh

CYCLE_TAG="${CYCLE_TAG:?set CYCLE_TAG}"
FIRST_SEED="${FIRST_SEED:?set FIRST_SEED}"
SEEDS_PER_HOST="${SEEDS_PER_HOST:?set SEEDS_PER_HOST}"
INCUMBENT_DIR="${INCUMBENT_DIR:?set INCUMBENT_DIR (local checkpoint dir)}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION}"
HOSTS="${HOSTS:-john2 john3 john4}"
GEN_N_SIMULATIONS="${GEN_N_SIMULATIONS:-128}"
GEN_DETERMINIZATIONS="${GEN_DETERMINIZATIONS:-2}"
SESSIONS="${SESSIONS:-6}"
RAYON_THREADS="${RAYON_THREADS:-8}"

test -s "$INCUMBENT_DIR/best_locked_val.manifest.json"
CKPT_NAME="$(basename "$INCUMBENT_DIR")"
REMOTE_CKPT="cascadiav3/checkpoints/${CKPT_NAME}"
REMOTE_MANIFEST="${REMOTE_CKPT}/best_locked_val.manifest.json"

FLEET_DIR="$(cd "$(dirname "$0")/.." && pwd)/fleet"
mkdir -p "$FLEET_DIR"
LEDGER="$FLEET_DIR/cbddb_${CYCLE_TAG}_fleet.json"
if [ -s "$LEDGER" ]; then
  echo "ledger $LEDGER already exists — refusing to double-launch" >&2
  exit 1
fi

i=0
entries=""
for h in $HOSTS; do
  first=$((FIRST_SEED + i * SEEDS_PER_HOST))
  echo "[fleet] $h: seeds ${first}x${SEEDS_PER_HOST} — syncing incumbent + script"
  rsync -a "$INCUMBENT_DIR/" "$h:~/cascadia/$REMOTE_CKPT/"
  rsync -a "$(dirname "$0")/fleet_cbddb_gen.sh" "$h:~/cascadia/cascadiav3/scripts/"
  ssh "$h" "cd ~/cascadia && \
    CYCLE_TAG='$CYCLE_TAG' SHARD_HOST='$h' FIRST_SEED='$first' \
    SEED_COUNT='$SEEDS_PER_HOST' INCUMBENT='$REMOTE_MANIFEST' \
    SOURCE_REVISION='$SOURCE_REVISION' \
    GEN_N_SIMULATIONS='$GEN_N_SIMULATIONS' GEN_DETERMINIZATIONS='$GEN_DETERMINIZATIONS' \
    SESSIONS='$SESSIONS' RAYON_THREADS='$RAYON_THREADS' \
    nohup bash cascadiav3/scripts/fleet_cbddb_gen.sh \
      > cascadiav3/logs/cbddb_${CYCLE_TAG}_shard_${h}.log 2>&1 < /dev/null & \
    echo \$! > cascadiav3/logs/cbddb_${CYCLE_TAG}_shard_${h}.pid; \
    cat cascadiav3/logs/cbddb_${CYCLE_TAG}_shard_${h}.pid"
  entries="$entries{\"host\":\"$h\",\"first_seed\":$first,\"seed_count\":$SEEDS_PER_HOST},"
  i=$((i + 1))
done

cat > "$LEDGER" <<EOF
{
  "cycle_tag": "$CYCLE_TAG",
  "source_revision": "$SOURCE_REVISION",
  "incumbent": "$REMOTE_MANIFEST",
  "gen": {"n_simulations": $GEN_N_SIMULATIONS, "determinizations": $GEN_DETERMINIZATIONS},
  "shards": [${entries%,}]
}
EOF
echo "[fleet] launched $(echo "$HOSTS" | wc -w | tr -d ' ') hosts; ledger: $LEDGER"
echo "[fleet] total seeds: first=$FIRST_SEED count=$((i * SEEDS_PER_HOST)) — record in EXPERIMENT_LOG seed ledger"
