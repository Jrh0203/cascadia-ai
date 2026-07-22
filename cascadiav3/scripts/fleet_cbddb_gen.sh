#!/usr/bin/env bash
set -euo pipefail

# CBDDB fleet shard generator — runs ON a Mac mini (john1-4), native
# MPS torch bridge. Parameterized successor of the hand-edited
# fleet5_gen.sh pattern (docs/v3/INFRASTRUCTURE.md §Fleet operations).
# Launched detached by fleet_cbddb_launch.sh; do not run on john0.
#
# Required env:
#   CYCLE_TAG    e.g. fs_c3 (artifact namespace cbddb_<tag>_shard_<host>_*)
#   SHARD_HOST   this host's name (john1..john4), used in artifact names
#   FIRST_SEED / SEED_COUNT   this shard's contiguous seed range —
#                allocated by the launcher, recorded in the fleet ledger
#   INCUMBENT    manifest path relative to ~/cascadia (checkpoint dir
#                must have been rsynced here by the launcher)
#   SOURCE_REVISION  git revision of the deployed tree
# Optional env: GEN_N_SIMULATIONS (128), GEN_DETERMINIZATIONS (2),
#   SESSIONS (6), RAYON_THREADS (8), DEVICE (mps), PLIES (80).

ROOT="${ROOT:-$HOME/cascadia}"
CYCLE_TAG="${CYCLE_TAG:?set CYCLE_TAG}"
SHARD_HOST="${SHARD_HOST:?set SHARD_HOST}"
FIRST_SEED="${FIRST_SEED:?set FIRST_SEED}"
SEED_COUNT="${SEED_COUNT:?set SEED_COUNT}"
INCUMBENT="${INCUMBENT:?set INCUMBENT manifest path}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION}"
GEN_N_SIMULATIONS="${GEN_N_SIMULATIONS:-128}"
GEN_DETERMINIZATIONS="${GEN_DETERMINIZATIONS:-2}"
SESSIONS="${SESSIONS:-6}"
RAYON_THREADS="${RAYON_THREADS:-8}"
DEVICE="${DEVICE:-mps}"
PLIES="${PLIES:-80}"

cd "$ROOT"
BINARY=cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter
PYBIN="$ROOT/venv/bin/python3"
[ -x "$PYBIN" ] || PYBIN="$ROOT/.venv/bin/python3"
[ -x "$PYBIN" ] || { echo "no venv python on $SHARD_HOST" >&2; exit 1; }
test -x "$BINARY"
test -s "$INCUMBENT"
grep -q 'rules_2026_07_19' cascadiav3/real-root-exporter/src/main.rs

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="cascadiav3/src"
mkdir -p cascadiav3/fixtures cascadiav3/logs

TAG="cbddb_${CYCLE_TAG}_shard_${SHARD_HOST}"
hb(){ echo "[$(date "+%F %T")] [fleet-$CYCLE_TAG-$SHARD_HOST] $*"; }

hb "GEN starting seeds=${FIRST_SEED}x${SEED_COUNT} n${GEN_N_SIMULATIONS}/d${GEN_DETERMINIZATIONS} device=$DEVICE rev=$SOURCE_REVISION"
"$BINARY" \
  --gumbel-selfplay-tensor-corpus \
  --scoring-cards cbddb \
  --model-service "$PYBIN -m cascadiav3.torch_inference_bridge --manifest $INCUMBENT --device $DEVICE" \
  --model-manifest "$INCUMBENT" \
  --model-timeout-ms 300000 \
  --gumbel-n-simulations "$GEN_N_SIMULATIONS" --gumbel-top-m 16 --gumbel-depth-rounds 1 \
  --gumbel-determinizations "$GEN_DETERMINIZATIONS" --gumbel-market-decision-samples 8 \
  --gumbel-exact-endgame-turns 0 --gumbel-blend-weight 0.5 --k-interior 16 \
  --source-revision "$SOURCE_REVISION" \
  --first-seed "$FIRST_SEED" --seed-count "$SEED_COUNT" --plies-per-seed "$PLIES" \
  --max-actions 8 --rollouts-per-action 1 --rollout-top-k 4 \
  --tensor-compression stored \
  --rayon-threads "$RAYON_THREADS" --model-sessions "$SESSIONS" --shared-model-session \
  --decisions-out "cascadiav3/fixtures/${TAG}_decisions.jsonl" \
  --out "cascadiav3/fixtures/${TAG}_tensor.npz" \
  --manifest "cascadiav3/fixtures/${TAG}_manifest.json"
hb "GEN DONE"
