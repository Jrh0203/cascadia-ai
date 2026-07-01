#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
REMOTE="${REMOTE:-john0}"
SSH_PORT="${SSH_PORT:-2222}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/john0/cascadia}"
REMOTE_VENV="${REMOTE_VENV:-/home/john0/venvs/torch}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RSYNC_SSH="ssh -p ${SSH_PORT}"

TRAIN_OUT="${TRAIN_OUT:-cascadiav3/fixtures/crt_wide32_r16p80x2_semantic_train.jsonl}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-cascadiav3/fixtures/crt_wide32_r16p80x2_semantic_train_manifest.json}"
VAL_OUT="${VAL_OUT:-cascadiav3/fixtures/crt_wide32_r16p80x2_semantic_val.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-cascadiav3/fixtures/crt_wide32_r16p80x2_semantic_val_manifest.json}"
TRAIN_FIRST_SEED="${TRAIN_FIRST_SEED:-2026140000}"
TRAIN_SEED_COUNT="${TRAIN_SEED_COUNT:-60}"
VAL_FIRST_SEED="${VAL_FIRST_SEED:-2026149000}"
VAL_SEED_COUNT="${VAL_SEED_COUNT:-16}"
PLIES_PER_SEED="${PLIES_PER_SEED:-80}"
MAX_ACTIONS="${MAX_ACTIONS:-32}"
ROLLOUTS_PER_ACTION="${ROLLOUTS_PER_ACTION:-16}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-4}"
REGENERATE_ROOTS="${REGENERATE_ROOTS:-0}"

STEPS="${STEPS:-12000}"
BATCH_SIZE="${BATCH_SIZE:-12}"
LR="${LR:-0.00030}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
LAYERS="${LAYERS:-4}"
HEADS="${HEADS:-8}"
MLP_DIM="${MLP_DIM:-512}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-0.25}"
SEEDS="${SEEDS:-20260640,20260641,20260642}"
LOSS_MODE="${LOSS_MODE:-topk-retention}"
Q_LOSS_WEIGHT="${Q_LOSS_WEIGHT:-0.15}"
POLICY_LOSS_WEIGHT="${POLICY_LOSS_WEIGHT:-0.25}"
BEST_MARGIN_LOSS_WEIGHT="${BEST_MARGIN_LOSS_WEIGHT:-1.0}"
RETENTION_LOSS_WEIGHT="${RETENTION_LOSS_WEIGHT:-1.50}"
RETENTION_K="${RETENTION_K:-16}"
PAIRWISE_MARGIN="${PAIRWISE_MARGIN:-0.15}"
POLICY_TEMPERATURE="${POLICY_TEMPERATURE:-0.75}"

EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-crt-wide32-r16p80x2-semantic-residual-attention}"
REPORT_PREFIX="${REPORT_PREFIX:-cascadiav3/reports/crt_wide32_r16p80x2_semantic_residual_attention}"
CHECKPOINT_PREFIX="${CHECKPOINT_PREFIX:-cascadiav3/checkpoints/crt_wide32_r16p80x2_semantic_residual_attention}"
PREFILTER_PREFIX="${PREFILTER_PREFIX:-cascadiav3/reports/crt_wide32_r16p80x2_semantic_residual_attention}"
ENSEMBLE_REPORT="${ENSEMBLE_REPORT:-cascadiav3/reports/crt_wide32_r16p80x2_residual_seed_ensemble_3x_eval.json}"
ENSEMBLE_PER_ROOT_OUT="${ENSEMBLE_PER_ROOT_OUT:-cascadiav3/reports/crt_wide32_r16p80x2_residual_seed_ensemble_3x_eval_roots.jsonl}"
ENSEMBLE_SUMMARY_OUT="${ENSEMBLE_SUMMARY_OUT:-cascadiav3/reports/crt_wide32_r16p80x2_residual_seed_ensemble_3x_eval_summary.md}"
SWEEP_SUMMARY_OUT="${SWEEP_SUMMARY_OUT:-cascadiav3/reports/crt_wide32_r16p80x2_semantic_residual_attention_seed_sweep_summary.md}"
REMOTE_LOG_DIR="$REMOTE_ROOT/cascadiav3/logs"
REMOTE_JOB="$REMOTE_LOG_DIR/r16p80x2_semantic_residual_attention_sweep_job.sh"
REMOTE_LOG="$REMOTE_LOG_DIR/r16p80x2_semantic_residual_attention_sweep_job.log"
REMOTE_PID="$REMOTE_LOG_DIR/r16p80x2_semantic_residual_attention_sweep_job.pid"

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

echo "[sweep] started \$(date -Is)"
echo "[sweep] train=$TRAIN_OUT val=$VAL_OUT seeds=$SEEDS"
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

inputs=""
for seed in \$(echo '$SEEDS' | tr ',' ' '); do
  report="${REPORT_PREFIX}_seed\${seed}_pilot.json"
  checkpoint="${CHECKPOINT_PREFIX}_seed\${seed}_pilot.pt"
  prefilter_report="${PREFILTER_PREFIX}_seed\${seed}_prefilter_eval.json"
  per_root="${PREFILTER_PREFIX}_seed\${seed}_prefilter_eval_roots.jsonl"
  echo "[sweep] training residual-attention seed \${seed} -> \${report}"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m cascadiav3.torch_semantic_residual_attention_merit \
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
    --residual-scale '$RESIDUAL_SCALE' \
    --seed "\${seed}" \
    --experiment-id '${EXPERIMENT_PREFIX}-seed'"\${seed}"'-v1' \
    --out "\${report}" \
    --checkpoint "\${checkpoint}"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m cascadiav3.torch_prefilter_eval \
    --val '$VAL_OUT' \
    --checkpoint "\${checkpoint}" \
    --batch-size '$BATCH_SIZE' \
    --k-values '4,8,16,24,32' \
    --min-recall '0.75' \
    --max-oracle-regret '0.25' \
    --experiment-id '${EXPERIMENT_PREFIX}-seed'"\${seed}"'-prefilter-v1' \
    --out "\${prefilter_report}" \
    --per-root-out "\${per_root}"
  if [ -z "\$inputs" ]; then
    inputs="\${per_root}"
  else
    inputs="\${inputs},\${per_root}"
  fi
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m cascadiav3.torch_prefilter_seed_ensemble_eval \
  --inputs "\$inputs" \
  --k-values '4,8,16,24,32' \
  --min-recall '0.75' \
  --max-oracle-regret '0.25' \
  --experiment-id '${EXPERIMENT_PREFIX}-seed-ensemble-3x-v1' \
  --out '$ENSEMBLE_REPORT' \
  --per-root-out '$ENSEMBLE_PER_ROOT_OUT' \
  --summary-out '$ENSEMBLE_SUMMARY_OUT'

SWEEP_SEEDS='$SEEDS' \
REPORT_PREFIX='$REPORT_PREFIX' \
ENSEMBLE_REPORT='$ENSEMBLE_REPORT' \
SUMMARY_OUT='$SWEEP_SUMMARY_OUT' \
TRAIN_OUT='$TRAIN_OUT' \
VAL_OUT='$VAL_OUT' \
TRAIN_SEED_COUNT='$TRAIN_SEED_COUNT' \
VAL_SEED_COUNT='$VAL_SEED_COUNT' \
PLIES_PER_SEED='$PLIES_PER_SEED' \
ROLLOUTS_PER_ACTION='$ROLLOUTS_PER_ACTION' \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python - <<'PY'
import json
import os
from pathlib import Path

seeds = [seed.strip() for seed in os.environ["SWEEP_SEEDS"].split(",") if seed.strip()]
report_prefix = os.environ["REPORT_PREFIX"]
ensemble = json.loads(Path(os.environ["ENSEMBLE_REPORT"]).read_text(encoding="utf-8"))
lines = [
    "# CRT Wide-32 R16p80x2 Semantic Residual-Attention Seed Sweep",
    "",
    "Status: completed.",
    "",
    "## Data",
    "",
    f"- Train: {os.environ['TRAIN_OUT']}",
    f"- Validation: {os.environ['VAL_OUT']}",
    f"- Train seed count: {os.environ['TRAIN_SEED_COUNT']}",
    f"- Validation seed count: {os.environ['VAL_SEED_COUNT']}",
    f"- Plies per seed: {os.environ['PLIES_PER_SEED']}",
    f"- Rollout samples/action: {os.environ['ROLLOUTS_PER_ACTION']}",
    "",
    "## Results",
    "",
    "| Artifact | K=16 recall | K=16 oracle regret | K=24 recall | K=24 oracle regret | Mean regret |",
    "|---|---:|---:|---:|---:|---:|",
]
for seed in seeds:
    report = json.loads(Path(f"{report_prefix}_seed{seed}_pilot.json").read_text(encoding="utf-8"))
    metrics = report["models"]["residual_attention_transformer"]["metrics"]
    k16 = metrics["prefilter"]["16"]
    k24 = metrics["prefilter"]["24"]
    lines.append(
        f"| seed {seed} | {k16['recall']:.4f} | {k16['mean_oracle_regret']:.4f} | "
        f"{k24['recall']:.4f} | {k24['mean_oracle_regret']:.4f} | {metrics['mean_regret']:.4f} |"
    )
ensemble_metrics = ensemble["metrics"]
ensemble_k16 = ensemble_metrics["prefilter"]["16"]
ensemble_k24 = ensemble_metrics["prefilter"]["24"]
decision = ensemble["serving_decision"]
lines.append(
    f"| fixed 3-seed ensemble | {ensemble_k16['recall']:.4f} | {ensemble_k16['mean_oracle_regret']:.4f} | "
    f"{ensemble_k24['recall']:.4f} | {ensemble_k24['mean_oracle_regret']:.4f} | {ensemble_metrics['mean_regret']:.4f} |"
)
lines.extend(
    [
        "",
        "## Decision",
        "",
        f"- Passes serving gate: {decision['passes']}",
        f"- Recommended K: {decision['recommended_k']}",
        f"- K=16 recall: {ensemble_k16['recall']:.4f}",
        f"- K=16 oracle regret: {ensemble_k16['mean_oracle_regret']:.4f}",
        "",
        "This is still dry-run sampled-teacher evidence, not gameplay strength.",
    ]
)
summary_path = Path(os.environ["SUMMARY_OUT"])
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

/usr/lib/wsl/lib/nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,power.draw --format=csv
echo "[sweep] completed \$(date -Is)"
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
nohup setsid '$REMOTE_JOB' > '$REMOTE_LOG' 2>&1 < /dev/null &
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
ps -eo pid=,ppid=,args= | awk '
  /cascadiav3-real-root-exporter/ && /crt_wide32_r16p80x2/ && !/awk/ {print \"matching exporter pid \" \$1 \" ppid \" \$2}
  /torch_semantic_residual_attention_merit/ && /r16p80x2/ && !/awk/ {print \"matching trainer pid \" \$1 \" ppid \" \$2}
  /torch_prefilter/ && /r16p80x2/ && !/awk/ {print \"matching evaluator pid \" \$1 \" ppid \" \$2}
'
ls -lh '$TRAIN_OUT' '$VAL_OUT' '$ENSEMBLE_REPORT' '$ENSEMBLE_SUMMARY_OUT' '$SWEEP_SUMMARY_OUT' 2>/dev/null || true
tail -n 100 '$REMOTE_LOG' 2>/dev/null || true"
}

stop_job() {
  ssh -p "$SSH_PORT" "$REMOTE" "set -euo pipefail
if [ -s '$REMOTE_PID' ]; then
  pid=\"\$(cat '$REMOTE_PID')\"
  if kill -0 \"\$pid\" 2>/dev/null; then
    kill -TERM -\"\$pid\" 2>/dev/null || kill -TERM \"\$pid\" 2>/dev/null || true
    sleep 2
    kill -KILL -\"\$pid\" 2>/dev/null || kill -KILL \"\$pid\" 2>/dev/null || true
  fi
fi
pids=\"\$(ps -eo pid=,args= | awk '
  /cascadiav3-real-root-exporter/ && /crt_wide32_r16p80x2/ && !/awk/ {print \$1}
  /torch_semantic_residual_attention_merit/ && /r16p80x2/ && !/awk/ {print \$1}
  /torch_prefilter/ && /r16p80x2/ && !/awk/ {print \$1}
' | sort -u | tr '\n' ' ')\"
if [ -n \"\$pids\" ]; then
  kill -TERM \$pids 2>/dev/null || true
  sleep 2
  kill -KILL \$pids 2>/dev/null || true
fi
rm -f '$REMOTE_PID'
echo stopped"
}

fetch_artifacts() {
  cd "$LOCAL_ROOT"
  mkdir -p cascadiav3/fixtures cascadiav3/reports cascadiav3/checkpoints cascadiav3/logs
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
  stop)
    stop_job
    ;;
  fetch)
    fetch_artifacts
    ;;
  *)
    echo "usage: $0 {launch|status|stop|fetch}" >&2
    exit 2
    ;;
esac
