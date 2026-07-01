#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-john0}"
SSH_PORT="${SSH_PORT:-2222}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/john0/cascadia}"
REMOTE_VENV="${REMOTE_VENV:-/home/john0/venvs/torch}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RSYNC_SSH="ssh -p ${SSH_PORT}"

TRAIN_OUT="${TRAIN_OUT:-cascadiav3/fixtures/crt_token_merit_train.jsonl}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-cascadiav3/fixtures/crt_token_merit_train_manifest.json}"
VAL_OUT="${VAL_OUT:-cascadiav3/fixtures/crt_token_merit_val.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-cascadiav3/fixtures/crt_token_merit_val_manifest.json}"

TRAIN_FIRST_SEED="${TRAIN_FIRST_SEED:-2026070000}"
TRAIN_SEED_COUNT="${TRAIN_SEED_COUNT:-100}"
VAL_FIRST_SEED="${VAL_FIRST_SEED:-2026079000}"
VAL_SEED_COUNT="${VAL_SEED_COUNT:-25}"
PLIES_PER_SEED="${PLIES_PER_SEED:-4}"
MAX_ACTIONS="${MAX_ACTIONS:-16}"
ROLLOUTS_PER_ACTION="${ROLLOUTS_PER_ACTION:-1}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-1}"
REGENERATE_ROOTS="${REGENERATE_ROOTS:-0}"

STEPS="${STEPS:-1600}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-0.0005}"
HIDDEN_DIM="${HIDDEN_DIM:-160}"
LAYERS="${LAYERS:-3}"
HEADS="${HEADS:-5}"
MLP_DIM="${MLP_DIM:-320}"
EXPERIMENT_ID="${EXPERIMENT_ID:-crt-relation-bias-query-merit-v1}"
TRAIN_MODULE="${TRAIN_MODULE:-cascadiav3.torch_relation_bias_merit}"
LOSS_MODE="${LOSS_MODE:-standard}"
Q_LOSS_WEIGHT="${Q_LOSS_WEIGHT:-0.25}"
POLICY_LOSS_WEIGHT="${POLICY_LOSS_WEIGHT:-0.5}"
BEST_MARGIN_LOSS_WEIGHT="${BEST_MARGIN_LOSS_WEIGHT:-1.0}"
RETENTION_LOSS_WEIGHT="${RETENTION_LOSS_WEIGHT:-1.0}"
RETENTION_K="${RETENTION_K:-16}"
PAIRWISE_MARGIN="${PAIRWISE_MARGIN:-0.25}"
POLICY_TEMPERATURE="${POLICY_TEMPERATURE:-0.5}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

REPORT="${REPORT:-cascadiav3/reports/crt_relation_bias_pilot.json}"
CHECKPOINT="${CHECKPOINT:-cascadiav3/checkpoints/crt_relation_bias_pilot.pt}"

cd "$LOCAL_ROOT"

ssh -p "$SSH_PORT" "$REMOTE" "mkdir -p '$REMOTE_ROOT/crates'"

rsync -az -e "$RSYNC_SSH" Cargo.toml Cargo.lock "$REMOTE:$REMOTE_ROOT/"
rsync -az -e "$RSYNC_SSH" --delete \
  --exclude 'target/' \
  crates/cascadia-game/ "$REMOTE:$REMOTE_ROOT/crates/cascadia-game/"
rsync -az -e "$RSYNC_SSH" --delete \
  --exclude 'target/' \
  crates/cascadia-sim/ "$REMOTE:$REMOTE_ROOT/crates/cascadia-sim/"

rsync -az -e "$RSYNC_SSH" --delete \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'target/' \
  cascadiav3/ "$REMOTE:$REMOTE_ROOT/cascadiav3/"

ssh -p "$SSH_PORT" "$REMOTE" "set -euo pipefail
cd '$REMOTE_ROOT'
. ~/.cargo/env 2>/dev/null || true
export BLAKE3_NO_ASM=1
if [ -x /home/john0/.local/bin/zig-cc ]; then
  export CC=/home/john0/.local/bin/zig-cc
  export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/home/john0/.local/bin/zig-cc
fi
cargo test --manifest-path cascadiav3/real-root-exporter/Cargo.toml
if [ '$REGENERATE_ROOTS' = '1' ] || [ ! -s '$TRAIN_OUT' ] || [ ! -s '$VAL_OUT' ]; then
  FIRST_SEED='$TRAIN_FIRST_SEED' \
  SEED_COUNT='$TRAIN_SEED_COUNT' \
  PLIES_PER_SEED='$PLIES_PER_SEED' \
  MAX_ACTIONS='$MAX_ACTIONS' \
  ROLLOUTS_PER_ACTION='$ROLLOUTS_PER_ACTION' \
  ROLLOUT_TOP_K='$ROLLOUT_TOP_K' \
  OUT='$TRAIN_OUT' \
  MANIFEST='$TRAIN_MANIFEST' \
  ./cascadiav3/scripts/generate_real_roots.sh
  FIRST_SEED='$VAL_FIRST_SEED' \
  SEED_COUNT='$VAL_SEED_COUNT' \
  PLIES_PER_SEED='$PLIES_PER_SEED' \
  MAX_ACTIONS='$MAX_ACTIONS' \
  ROLLOUTS_PER_ACTION='$ROLLOUTS_PER_ACTION' \
  ROLLOUT_TOP_K='$ROLLOUT_TOP_K' \
  OUT='$VAL_OUT' \
  MANIFEST='$VAL_MANIFEST' \
  ./cascadiav3/scripts/generate_real_roots.sh
fi
. '$REMOTE_VENV/bin/activate'
export LD_LIBRARY_PATH=/usr/lib/wsl/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
/usr/lib/wsl/lib/nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,temperature.gpu,power.draw,power.limit --format=csv
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m unittest discover -s cascadiav3/tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m '$TRAIN_MODULE' \
  --train '$TRAIN_OUT' \
  --val '$VAL_OUT' \
  --steps '$STEPS' \
  --batch-size '$BATCH_SIZE' \
  --lr '$LR' \
  --hidden-dim '$HIDDEN_DIM' \
  --layers '$LAYERS' \
  --heads '$HEADS' \
  --mlp-dim '$MLP_DIM' \
  --loss-mode '$LOSS_MODE' \
  --q-loss-weight '$Q_LOSS_WEIGHT' \
  --policy-loss-weight '$POLICY_LOSS_WEIGHT' \
  --best-margin-loss-weight '$BEST_MARGIN_LOSS_WEIGHT' \
  --retention-loss-weight '$RETENTION_LOSS_WEIGHT' \
  --retention-k '$RETENTION_K' \
  --pairwise-margin '$PAIRWISE_MARGIN' \
  --policy-temperature '$POLICY_TEMPERATURE' \
  --experiment-id '$EXPERIMENT_ID' \
  --out '$REPORT' \
  --checkpoint '$CHECKPOINT' \
  $EXTRA_TRAIN_ARGS
/usr/lib/wsl/lib/nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,power.draw --format=csv
"

rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/$TRAIN_OUT" cascadiav3/fixtures/
rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/$TRAIN_MANIFEST" cascadiav3/fixtures/
rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/$VAL_OUT" cascadiav3/fixtures/
rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/$VAL_MANIFEST" cascadiav3/fixtures/
rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/cascadiav3/real-root-exporter/Cargo.lock" cascadiav3/real-root-exporter/Cargo.lock
rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/cascadiav3/reports/" cascadiav3/reports/
rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/cascadiav3/checkpoints/" cascadiav3/checkpoints/
