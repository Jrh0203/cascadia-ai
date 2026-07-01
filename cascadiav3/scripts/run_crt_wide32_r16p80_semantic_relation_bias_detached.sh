#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
REMOTE="${REMOTE:-john0}"
SSH_PORT="${SSH_PORT:-2222}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/john0/cascadia}"
REMOTE_VENV="${REMOTE_VENV:-/home/john0/venvs/torch}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RSYNC_SSH="ssh -p ${SSH_PORT}"

TRAIN_OUT="${TRAIN_OUT:-cascadiav3/fixtures/crt_wide32_r16p80_semantic_train.jsonl}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-cascadiav3/fixtures/crt_wide32_r16p80_semantic_train_manifest.json}"
VAL_OUT="${VAL_OUT:-cascadiav3/fixtures/crt_wide32_r16p80_semantic_val.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-cascadiav3/fixtures/crt_wide32_r16p80_semantic_val_manifest.json}"
TRAIN_FIRST_SEED="${TRAIN_FIRST_SEED:-2026130000}"
TRAIN_SEED_COUNT="${TRAIN_SEED_COUNT:-30}"
VAL_FIRST_SEED="${VAL_FIRST_SEED:-2026139000}"
VAL_SEED_COUNT="${VAL_SEED_COUNT:-8}"
PLIES_PER_SEED="${PLIES_PER_SEED:-80}"
MAX_ACTIONS="${MAX_ACTIONS:-32}"
ROLLOUTS_PER_ACTION="${ROLLOUTS_PER_ACTION:-16}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-4}"
REGENERATE_ROOTS="${REGENERATE_ROOTS:-1}"

STEPS="${STEPS:-7600}"
BATCH_SIZE="${BATCH_SIZE:-12}"
LR="${LR:-0.00032}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
LAYERS="${LAYERS:-4}"
HEADS="${HEADS:-8}"
MLP_DIM="${MLP_DIM:-512}"
LOSS_MODE="${LOSS_MODE:-standard}"
Q_LOSS_WEIGHT="${Q_LOSS_WEIGHT:-0.25}"
POLICY_LOSS_WEIGHT="${POLICY_LOSS_WEIGHT:-0.5}"
BEST_MARGIN_LOSS_WEIGHT="${BEST_MARGIN_LOSS_WEIGHT:-1.0}"
RETENTION_LOSS_WEIGHT="${RETENTION_LOSS_WEIGHT:-1.0}"
RETENTION_K="${RETENTION_K:-16}"
PAIRWISE_MARGIN="${PAIRWISE_MARGIN:-0.25}"
POLICY_TEMPERATURE="${POLICY_TEMPERATURE:-0.5}"
EXPERIMENT_ID="${EXPERIMENT_ID:-crt-wide32-r16p80-semantic-relation-bias-v1}"
PREFILTER_EXPERIMENT_ID="${PREFILTER_EXPERIMENT_ID:-crt-wide32-r16p80-semantic-prefilter-eval-v1}"
REPORT="${REPORT:-cascadiav3/reports/crt_wide32_r16p80_semantic_relation_bias_pilot.json}"
CHECKPOINT="${CHECKPOINT:-cascadiav3/checkpoints/crt_wide32_r16p80_semantic_relation_bias_pilot.pt}"
PREFILTER_REPORT="${PREFILTER_REPORT:-cascadiav3/reports/crt_wide32_r16p80_semantic_prefilter_eval.json}"
PER_ROOT_OUT="${PER_ROOT_OUT:-cascadiav3/reports/crt_wide32_r16p80_semantic_prefilter_eval_roots.jsonl}"
REMOTE_LOG_DIR="$REMOTE_ROOT/cascadiav3/logs"
REMOTE_JOB="$REMOTE_LOG_DIR/r16p80_semantic_relation_bias_job.sh"
REMOTE_LOG="$REMOTE_LOG_DIR/r16p80_semantic_relation_bias_job.log"
REMOTE_PID="$REMOTE_LOG_DIR/r16p80_semantic_relation_bias_job.pid"

sync_sources() {
  cd "$LOCAL_ROOT"
  ssh -p "$SSH_PORT" "$REMOTE" "mkdir -p '$REMOTE_ROOT/crates' '$REMOTE_LOG_DIR'"
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
    --exclude 'logs/' \
    --exclude 'target/' \
    cascadiav3/ "$REMOTE:$REMOTE_ROOT/cascadiav3/"
  ssh -p "$SSH_PORT" "$REMOTE" "mkdir -p '$REMOTE_LOG_DIR'"
}

write_remote_job() {
  ssh -p "$SSH_PORT" "$REMOTE" "mkdir -p '$REMOTE_LOG_DIR'"
  ssh -p "$SSH_PORT" "$REMOTE" "cat > '$REMOTE_JOB'" <<REMOTE_JOB
#!/usr/bin/env bash
set -euo pipefail
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
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m cascadiav3.torch_semantic_relation_bias_merit \
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
  --checkpoint '$CHECKPOINT'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m cascadiav3.torch_prefilter_eval \
  --val '$VAL_OUT' \
  --checkpoint '$CHECKPOINT' \
  --batch-size '$BATCH_SIZE' \
  --k-values '4,8,16,24,32' \
  --min-recall '0.75' \
  --max-oracle-regret '0.25' \
  --experiment-id '$PREFILTER_EXPERIMENT_ID' \
  --out '$PREFILTER_REPORT' \
  --per-root-out '$PER_ROOT_OUT'
/usr/lib/wsl/lib/nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,power.draw --format=csv
REMOTE_JOB
  ssh -p "$SSH_PORT" "$REMOTE" "chmod 700 '$REMOTE_JOB'"
}

launch_job() {
  sync_sources
  write_remote_job
  ssh -p "$SSH_PORT" "$REMOTE" "set -euo pipefail
if [ -s '$REMOTE_PID' ] && kill -0 \"\$(cat '$REMOTE_PID')\" 2>/dev/null; then
  echo \"already running pid \$(cat '$REMOTE_PID')\"
  echo '$REMOTE_LOG'
  exit 0
fi
nohup '$REMOTE_JOB' > '$REMOTE_LOG' 2>&1 < /dev/null &
echo \$! > '$REMOTE_PID'
echo \"launched pid \$(cat '$REMOTE_PID')\"
echo '$REMOTE_LOG'"
}

status_job() {
  ssh -p "$SSH_PORT" "$REMOTE" "set -euo pipefail
if [ -s '$REMOTE_PID' ] && kill -0 \"\$(cat '$REMOTE_PID')\" 2>/dev/null; then
  echo \"running pid \$(cat '$REMOTE_PID')\"
else
  echo 'not running'
fi
ls -lh '$TRAIN_OUT' '$VAL_OUT' '$REPORT' '$PREFILTER_REPORT' '$CHECKPOINT' 2>/dev/null || true
tail -n 80 '$REMOTE_LOG' 2>/dev/null || true"
}

fetch_artifacts() {
  cd "$LOCAL_ROOT"
  rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/$TRAIN_OUT" cascadiav3/fixtures/
  rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/$TRAIN_MANIFEST" cascadiav3/fixtures/
  rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/$VAL_OUT" cascadiav3/fixtures/
  rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/$VAL_MANIFEST" cascadiav3/fixtures/
  rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/cascadiav3/reports/" cascadiav3/reports/
  rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/cascadiav3/checkpoints/" cascadiav3/checkpoints/
  rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/cascadiav3/logs/" cascadiav3/logs/
}

case "$ACTION" in
  launch)
    launch_job
    ;;
  status)
    status_job
    ;;
  fetch)
    fetch_artifacts
    ;;
  *)
    echo "usage: $0 {launch|status|fetch}" >&2
    exit 2
    ;;
esac
