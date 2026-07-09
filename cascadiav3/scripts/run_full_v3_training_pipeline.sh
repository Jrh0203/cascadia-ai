#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
REMOTE="${REMOTE:-john0}"
SSH_PORT="${SSH_PORT:-2222}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/john0/cascadia}"
REMOTE_VENV="${REMOTE_VENV:-/home/john0/venvs/torch}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_REVISION="${SOURCE_REVISION:-$(git -C "$LOCAL_ROOT" rev-parse HEAD)}"
RSYNC_SSH="ssh -p ${SSH_PORT}"

JOB_SLUG="${JOB_SLUG:-full_v3_training_pipeline}"
RAYON_THREADS="${RAYON_THREADS:-32}"

# This is the largest safe first stage for packed expert tensor shards. JSONL is
# intentionally retained only for tiny reconstruction and boundary audit gates.
PROFILE="${PROFILE:-phase0_bootstrap_tensor}"
TRAIN_FIRST_SEED="${TRAIN_FIRST_SEED:-2026400000}"
TRAIN_SEED_COUNT="${TRAIN_SEED_COUNT:-125}"
VAL_FIRST_SEED="${VAL_FIRST_SEED:-2026500000}"
VAL_SEED_COUNT="${VAL_SEED_COUNT:-25}"
PLIES_PER_SEED="${PLIES_PER_SEED:-80}"
MAX_ACTIONS="${MAX_ACTIONS:-32}"
ROLLOUTS_PER_ACTION="${ROLLOUTS_PER_ACTION:-1}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-4}"
FILTER_TOP_K="${FILTER_TOP_K:-256}"
FILTER_MODE="${FILTER_MODE:-top-q-with-selected}"
FILTER_GREEDY_PREFIX_K="${FILTER_GREEDY_PREFIX_K:-}"
EXPERT_TENSOR_MODE="${EXPERT_TENSOR_MODE:-expert}"
REGENERATE_ROOTS="${REGENERATE_ROOTS:-0}"
MODEL_SERVICE="${MODEL_SERVICE:-}"
MODEL_MANIFEST="${MODEL_MANIFEST:-}"
MODEL_TIMEOUT_MS="${MODEL_TIMEOUT_MS:-10000}"
ALLOW_MODEL_FALLBACK="${ALLOW_MODEL_FALLBACK:-0}"
GUMBEL_N_SIMULATIONS="${GUMBEL_N_SIMULATIONS:-64}"
GUMBEL_TOP_M="${GUMBEL_TOP_M:-16}"
GUMBEL_DEPTH_ROUNDS="${GUMBEL_DEPTH_ROUNDS:-1}"
GUMBEL_DETERMINIZATIONS="${GUMBEL_DETERMINIZATIONS:-4}"
GUMBEL_MARKET_DECISION_SAMPLES="${GUMBEL_MARKET_DECISION_SAMPLES:-8}"
GUMBEL_EXACT_ENDGAME_TURNS="${GUMBEL_EXACT_ENDGAME_TURNS:-0}"
GUMBEL_BLEND_WEIGHT="${GUMBEL_BLEND_WEIGHT:-0.5}"
GUMBEL_K_INTERIOR="${GUMBEL_K_INTERIOR:-16}"
GUMBEL_MAX_ROOT_ACTIONS="${GUMBEL_MAX_ROOT_ACTIONS:-}"
MODEL_SESSIONS="${MODEL_SESSIONS:-}"
SHARED_MODEL_SESSION="${SHARED_MODEL_SESSION:-0}"

MODEL_SIZE="${MODEL_SIZE:-S}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
WARMUP_FRACTION="${WARMUP_FRACTION:-0.02}"
VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-8}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-250}"
MIN_SELECTION_GREEDY_TOP1="${MIN_SELECTION_GREEDY_TOP1:-0}"
EARLY_STOP_SELECTION_GUARD_FAILURES="${EARLY_STOP_SELECTION_GUARD_FAILURES:-0}"
EARLY_STOP_AFTER_STEP="${EARLY_STOP_AFTER_STEP:-0}"
SWA_FRACTION="${SWA_FRACTION:-0.20}"
SEED="${SEED:-20260630}"
OBJECTIVE="${OBJECTIVE:-expert}"
MAX_EXAMPLE_PASSES="${MAX_EXAMPLE_PASSES:-0}"
SELECTION_METRIC="${SELECTION_METRIC:-locked_val_total}"
SELECTION_MODE="${SELECTION_MODE:-min}"
INIT_MANIFEST="${INIT_MANIFEST:-}"
# "none" = explicit from-scratch init even when a wrapper defaults
# INIT_MANIFEST to the incumbent (S weights cannot warm-start M).
if [ "$INIT_MANIFEST" = "none" ]; then INIT_MANIFEST=""; fi
# Extra trainer flags appended verbatim (perf knobs: --data-workers 4
# --tf32 --fused-optimizer --cgab-fused --grad-checkpoint on ...).
TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS:-}"
RESUME_MANIFEST="${RESUME_MANIFEST:-}"
TRAIN_SOURCE_WEIGHTS="${TRAIN_SOURCE_WEIGHTS:-}"
EXTRA_TRAIN_TAIL_TENSORS="${EXTRA_TRAIN_TAIL_TENSORS:-}"

BINARY="cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter"
REMOTE_LOG_DIR="$REMOTE_ROOT/cascadiav3/logs"
REMOTE_JOB="$REMOTE_LOG_DIR/${JOB_SLUG}_job.sh"
REMOTE_LOG="$REMOTE_LOG_DIR/${JOB_SLUG}_job.log"
REMOTE_PID="$REMOTE_LOG_DIR/${JOB_SLUG}_job.pid"

TRAIN_ROOTS="${TRAIN_ROOTS:-cascadiav3/fixtures/full_v3_${PROFILE}_train_roots.jsonl}"
TRAIN_TENSOR="${TRAIN_TENSOR:-cascadiav3/fixtures/full_v3_${PROFILE}_train_tensor.npz}"
TRAIN_FILTERED_TENSOR="${TRAIN_FILTERED_TENSOR:-cascadiav3/fixtures/full_v3_${PROFILE}_train_tensor_top${FILTER_TOP_K}.npz}"
TRAIN_TAIL_TENSOR="${TRAIN_TAIL_TENSOR:-cascadiav3/fixtures/full_v3_${PROFILE}_train_tensor_top${FILTER_TOP_K}_relation_tail.npz}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-cascadiav3/fixtures/full_v3_${PROFILE}_train_manifest.json}"
VAL_ROOTS="${VAL_ROOTS:-cascadiav3/fixtures/full_v3_${PROFILE}_val_roots.jsonl}"
VAL_TENSOR="${VAL_TENSOR:-cascadiav3/fixtures/full_v3_${PROFILE}_val_tensor.npz}"
VAL_FILTERED_TENSOR="${VAL_FILTERED_TENSOR:-cascadiav3/fixtures/full_v3_${PROFILE}_val_tensor_top${FILTER_TOP_K}.npz}"
VAL_TAIL_TENSOR="${VAL_TAIL_TENSOR:-cascadiav3/fixtures/full_v3_${PROFILE}_val_tensor_top${FILTER_TOP_K}_relation_tail.npz}"
VAL_MANIFEST="${VAL_MANIFEST:-cascadiav3/fixtures/full_v3_${PROFILE}_val_manifest.json}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-cascadiav3/checkpoints/full_v3_${PROFILE}}"
REPORT="${REPORT:-cascadiav3/reports/full_v3_${PROFILE}_train.json}"
METRICS="${METRICS:-cascadiav3/reports/full_v3_${PROFILE}_metrics.jsonl}"
RUNBOOK_REPORT="${RUNBOOK_REPORT:-cascadiav3/reports/full_v3_${PROFILE}_runbook.json}"

sync_sources() {
  cd "$LOCAL_ROOT"
  local source_status
  source_status="$(git status --porcelain --untracked-files=all -- \
    Cargo.toml Cargo.lock crates/cascadia-game crates/cascadia-sim cascadiav3 docs/v3)"
  if [ -n "$source_status" ]; then
    echo "refusing to sync a dirty training source tree for revision $SOURCE_REVISION" >&2
    echo "$source_status" >&2
    return 1
  fi
  ssh -p "$SSH_PORT" "$REMOTE" "mkdir -p '$REMOTE_ROOT/crates' '$REMOTE_ROOT/docs' '$REMOTE_LOG_DIR'"
  rsync -az -e "$RSYNC_SSH" Cargo.toml Cargo.lock "$REMOTE:$REMOTE_ROOT/"
  rsync -az -e "$RSYNC_SSH" --delete --exclude 'target/' \
    crates/cascadia-game/ "$REMOTE:$REMOTE_ROOT/crates/cascadia-game/"
  rsync -az -e "$RSYNC_SSH" --delete --exclude 'target/' \
    crates/cascadia-sim/ "$REMOTE:$REMOTE_ROOT/crates/cascadia-sim/"
  rsync -az -e "$RSYNC_SSH" --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'fixtures/' \
    --exclude 'reports/' \
    --exclude 'checkpoints/' \
    --exclude 'logs/' \
    --exclude 'target/' \
    cascadiav3/ "$REMOTE:$REMOTE_ROOT/cascadiav3/"
  rsync -az -e "$RSYNC_SSH" --delete docs/v3/ "$REMOTE:$REMOTE_ROOT/docs/v3/"
  ssh -p "$SSH_PORT" "$REMOTE" "mkdir -p '$REMOTE_LOG_DIR' '$REMOTE_ROOT/cascadiav3/fixtures' '$REMOTE_ROOT/cascadiav3/reports' '$REMOTE_ROOT/cascadiav3/checkpoints'"
}

write_remote_job() {
  ssh -p "$SSH_PORT" "$REMOTE" "mkdir -p '$REMOTE_LOG_DIR'"
  ssh -p "$SSH_PORT" "$REMOTE" "cat > '$REMOTE_JOB'" <<REMOTE_JOB
#!/usr/bin/env bash
set -euo pipefail
cd '$REMOTE_ROOT'
. ~/.cargo/env 2>/dev/null || true
export BLAKE3_NO_ASM=1
export RAYON_NUM_THREADS='$RAYON_THREADS'
export OMP_NUM_THREADS='$RAYON_THREADS'
export MKL_NUM_THREADS='$RAYON_THREADS'
if [ -x /home/john0/.local/bin/zig-cc ]; then
  export CC=/home/john0/.local/bin/zig-cc
  export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/home/john0/.local/bin/zig-cc
fi

echo "[full-v3] started \$(date -Is)"
echo "[full-v3] profile=$PROFILE train_seeds=$TRAIN_SEED_COUNT val_seeds=$VAL_SEED_COUNT plies=$PLIES_PER_SEED rollouts_per_action=$ROLLOUTS_PER_ACTION rollout_top_k=$ROLLOUT_TOP_K expert_tensor_mode=$EXPERT_TENSOR_MODE"
echo "[full-v3] source_revision=$SOURCE_REVISION"
echo "[full-v3] model_size=$MODEL_SIZE steps=$TRAIN_STEPS batch_size=$BATCH_SIZE grad_accum=$GRAD_ACCUM lr=$LR val_max_batches=$VAL_MAX_BATCHES eval_every_steps=$EVAL_EVERY_STEPS min_selection_greedy_top1=$MIN_SELECTION_GREEDY_TOP1 early_stop_guard_failures=$EARLY_STOP_SELECTION_GUARD_FAILURES early_stop_after_step=$EARLY_STOP_AFTER_STEP filter_top_k=$FILTER_TOP_K filter_mode=$FILTER_MODE objective=$OBJECTIVE selection=$SELECTION_MODE:$SELECTION_METRIC"
echo "[full-v3] note: phase0 writes packed expert_tensor_shard.v1 NPZ directly, filters to top-K, then materializes fixed relation-tail tensors for GPU training"
echo "[full-v3] init_manifest=$INIT_MANIFEST"
echo "[full-v3] resume_manifest=$RESUME_MANIFEST"
echo "[full-v3] train_source_weights=$TRAIN_SOURCE_WEIGHTS"
echo "[full-v3] extra_train_tail_tensors=$EXTRA_TRAIN_TAIL_TENSORS"

cargo test --manifest-path cascadiav3/real-root-exporter/Cargo.toml
cargo build --release --manifest-path cascadiav3/real-root-exporter/Cargo.toml

. '$REMOTE_VENV/bin/activate'
export LD_LIBRARY_PATH=/usr/lib/wsl/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=cascadiav3/src

python -m unittest discover -s cascadiav3/tests -v
python -m cascadiav3.validate_schema_registry --include-legacy --include-expert

'$BINARY' --chance-mcts-dry-run --allow-model-fallback \
  --first-seed 2026063000 --seed-count 2 --plies-per-seed 2 \
  --rollouts-per-action 1 --rollout-top-k '$ROLLOUT_TOP_K' \
  --rayon-threads '$RAYON_THREADS' \
  --out cascadiav3/fixtures/expert_tiny.jsonl \
  --manifest cascadiav3/fixtures/expert_tiny_manifest.json
'$BINARY' --validate-expert-reconstruction \
  --in cascadiav3/fixtures/expert_tiny.jsonl \
  --manifest cascadiav3/fixtures/expert_tiny_manifest.json
python -m cascadiav3.validate_public_boundary --roots cascadiav3/fixtures/expert_tiny.jsonl --deny-hidden-fields
'$BINARY' --validate-hidden-redetermination --first-seed 2026063000 --seed-count 10 --out cascadiav3/reports/full_v3_hidden_redetermination.json
python -m cascadiav3.validate_d6_roundtrip --roots cascadiav3/fixtures/expert_tiny.jsonl
python -m cascadiav3.validate_category_targets --roots cascadiav3/fixtures/expert_tiny.jsonl

generate_tensor_roots() {
  local label="\$1"
  local first_seed="\$2"
  local seed_count="\$3"
  local tensor_path="\$4"
  local manifest_path="\$5"
  local started="\$(date +%s)"
  if [ '$REGENERATE_ROOTS' = '1' ] || [ ! -s "\$tensor_path" ]; then
    echo "[full-v3] generating \$label packed expert tensor roots first_seed=\$first_seed seed_count=\$seed_count"
    local mode_args=()
    case '$EXPERT_TENSOR_MODE' in
      expert)
        mode_args=(--expert-tensor-corpus --allow-model-fallback)
        ;;
      greedy)
        mode_args=(--greedy-expert-tensor-corpus)
        ;;
      greedy_search_bootstrap)
        mode_args=(--greedy-state-search-bootstrap-tensor-corpus)
        ;;
      model_state_search_bootstrap)
        mode_args=(--model-state-search-bootstrap-tensor-corpus)
        if [ -n '$MODEL_SERVICE' ]; then
          mode_args+=(--model-service '$MODEL_SERVICE')
        fi
        if [ -n '$MODEL_MANIFEST' ]; then
          mode_args+=(--model-manifest '$MODEL_MANIFEST')
        fi
        if [ '$ALLOW_MODEL_FALLBACK' = '1' ]; then
          mode_args+=(--allow-model-fallback)
        fi
        mode_args+=(--model-timeout-ms '$MODEL_TIMEOUT_MS')
        ;;
      gumbel_selfplay)
        mode_args=(--gumbel-selfplay-tensor-corpus)
        if [ -n '$MODEL_SERVICE' ]; then
          mode_args+=(--model-service '$MODEL_SERVICE')
        fi
        if [ -n '$MODEL_MANIFEST' ]; then
          mode_args+=(--model-manifest '$MODEL_MANIFEST')
        fi
        if [ '$ALLOW_MODEL_FALLBACK' = '1' ]; then
          mode_args+=(--allow-model-fallback)
        fi
        mode_args+=(--model-timeout-ms '$MODEL_TIMEOUT_MS')
        mode_args+=(--gumbel-n-simulations '$GUMBEL_N_SIMULATIONS')
        mode_args+=(--gumbel-top-m '$GUMBEL_TOP_M')
        mode_args+=(--gumbel-depth-rounds '$GUMBEL_DEPTH_ROUNDS')
        mode_args+=(--gumbel-determinizations '$GUMBEL_DETERMINIZATIONS')
        mode_args+=(--gumbel-market-decision-samples '$GUMBEL_MARKET_DECISION_SAMPLES')
        mode_args+=(--gumbel-exact-endgame-turns '$GUMBEL_EXACT_ENDGAME_TURNS')
        mode_args+=(--source-revision '$SOURCE_REVISION')
        mode_args+=(--gumbel-blend-weight '$GUMBEL_BLEND_WEIGHT')
        mode_args+=(--k-interior '$GUMBEL_K_INTERIOR')
        if [ -n '$GUMBEL_MAX_ROOT_ACTIONS' ]; then
          mode_args+=(--gumbel-max-root-actions '$GUMBEL_MAX_ROOT_ACTIONS')
        fi
        if [ -n '$MODEL_SESSIONS' ]; then
          mode_args+=(--model-sessions '$MODEL_SESSIONS')
        fi
        if [ '$SHARED_MODEL_SESSION' = '1' ]; then
          mode_args+=(--shared-model-session)
        fi
        ;;
      *)
        echo "[full-v3] unknown EXPERT_TENSOR_MODE=$EXPERT_TENSOR_MODE" >&2
        exit 2
        ;;
    esac
    '$BINARY' "\${mode_args[@]}" \
      --first-seed "\$first_seed" \
      --seed-count "\$seed_count" \
      --plies-per-seed '$PLIES_PER_SEED' \
      --max-actions '$MAX_ACTIONS' \
      --rollouts-per-action '$ROLLOUTS_PER_ACTION' \
      --rollout-top-k '$ROLLOUT_TOP_K' \
      --tensor-compression stored \
      --rayon-threads '$RAYON_THREADS' \
      --out "\$tensor_path" \
      --manifest "\$manifest_path"
  else
    echo "[full-v3] reusing existing \$label packed tensor roots \$tensor_path"
  fi
  python -m cascadiav3.expert_tensor_shards \
    --summarize-shard "\$tensor_path" \
    --report "cascadiav3/reports/full_v3_${PROFILE}_\${label}_tensor_summary.json"
  python -m cascadiav3.validate_expert_tensor_invariants \
    --shard "\$tensor_path" \
    --require-selected-action-dropped-count 0 \
    --require-q-equals-afterstate-plus-score-to-go \
    --report "cascadiav3/reports/full_v3_${PROFILE}_\${label}_tensor_invariants.json"
  local elapsed="\$(( \$(date +%s) - started ))"
  if [ "\$label" = train ]; then
    TRAIN_GENERATE_SECONDS="\$elapsed"
  else
    VAL_GENERATE_SECONDS="\$elapsed"
  fi
}

generate_tensor_roots train '$TRAIN_FIRST_SEED' '$TRAIN_SEED_COUNT' '$TRAIN_TENSOR' '$TRAIN_MANIFEST'
generate_tensor_roots val '$VAL_FIRST_SEED' '$VAL_SEED_COUNT' '$VAL_TENSOR' '$VAL_MANIFEST'

filter_tensor_roots() {
  local label="\$1"
  local raw_path="\$2"
  local filtered_path="\$3"
  if [ '$REGENERATE_ROOTS' = '1' ] || [ ! -s "\$filtered_path" ]; then
    echo "[full-v3] filtering \$label tensor roots to top-$FILTER_TOP_K retained actions"
    local filter_extra_args=()
    if [ -n '$FILTER_GREEDY_PREFIX_K' ]; then
      filter_extra_args=(--greedy-prefix-k '$FILTER_GREEDY_PREFIX_K')
    fi
    python -m cascadiav3.expert_tensor_shards \
      --filter-shard "\$raw_path" \
      --top-k '$FILTER_TOP_K' \
      --filter-mode '$FILTER_MODE' \
      "\${filter_extra_args[@]}" \
      --out "\$filtered_path" \
      --report "cascadiav3/reports/full_v3_${PROFILE}_\${label}_tensor_top${FILTER_TOP_K}_summary.json"
  else
    echo "[full-v3] reusing existing \$label filtered tensor roots \$filtered_path"
  fi
  python -m cascadiav3.validate_expert_tensor_invariants \
    --shard "\$filtered_path" \
    --require-selected-action-dropped-count 0 \
    --require-q-equals-afterstate-plus-score-to-go \
    --report "cascadiav3/reports/full_v3_${PROFILE}_\${label}_tensor_top${FILTER_TOP_K}_invariants.json"
}

filter_tensor_roots train '$TRAIN_TENSOR' '$TRAIN_FILTERED_TENSOR'
filter_tensor_roots val '$VAL_TENSOR' '$VAL_FILTERED_TENSOR'

materialize_relation_tail() {
  local label="\$1"
  local filtered_path="\$2"
  local tail_path="\$3"
  if [ '$REGENERATE_ROOTS' = '1' ] || [ ! -s "\$tail_path" ]; then
    echo "[full-v3] materializing \$label relation-tail tensor cache"
    python -m cascadiav3.expert_tensor_shards \
      --materialize-relation-tail "\$filtered_path" \
      --out "\$tail_path" \
      --report "cascadiav3/reports/full_v3_${PROFILE}_\${label}_tensor_top${FILTER_TOP_K}_relation_tail_summary.json"
  else
    echo "[full-v3] reusing existing \$label relation-tail tensor cache \$tail_path"
  fi
}

materialize_relation_tail train '$TRAIN_FILTERED_TENSOR' '$TRAIN_TAIL_TENSOR'
materialize_relation_tail val '$VAL_FILTERED_TENSOR' '$VAL_TAIL_TENSOR'

/usr/lib/wsl/lib/nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,temperature.gpu,power.draw,power.limit --format=csv
TRAINING_STARTED="\$(date +%s)"
TRAIN_INPUT='$TRAIN_TAIL_TENSOR'
if [ -n '$EXTRA_TRAIN_TAIL_TENSORS' ]; then
  TRAIN_INPUT="\$TRAIN_INPUT,$EXTRA_TRAIN_TAIL_TENSORS"
fi
IFS=',' read -r -a TRAIN_INPUT_PATHS <<< "\$TRAIN_INPUT"
for train_input_path in "\${TRAIN_INPUT_PATHS[@]}"; do
  test -s "\$train_input_path"
done
echo "[full-v3] trainer_train=\$TRAIN_INPUT"
TRAINER_INIT_ARGS=()
if [ -n '$RESUME_MANIFEST' ]; then
  TRAINER_INIT_ARGS=(--resume '$RESUME_MANIFEST')
elif [ -n '$INIT_MANIFEST' ]; then
  TRAINER_INIT_ARGS=(--init-manifest '$INIT_MANIFEST')
fi
TRAINER_MIX_ARGS=()
if [ -n '$TRAIN_SOURCE_WEIGHTS' ]; then
  TRAINER_MIX_ARGS=(--train-source-weights '$TRAIN_SOURCE_WEIGHTS')
fi
python -m cascadiav3.torch_train_cascadiaformer \
  --model-size '$MODEL_SIZE' \
  --train "\$TRAIN_INPUT" \
  --val '$VAL_TAIL_TENSOR' \
  --train-format npz \
  --val-format npz \
  --steps '$TRAIN_STEPS' \
  --batch-size '$BATCH_SIZE' \
  --grad-accum '$GRAD_ACCUM' \
  --lr '$LR' \
  --weight-decay '$WEIGHT_DECAY' \
  --warmup-fraction '$WARMUP_FRACTION' \
  --device cuda \
  --seed '$SEED' \
  --objective '$OBJECTIVE' \
  --max-example-passes '$MAX_EXAMPLE_PASSES' \
  --selection-metric '$SELECTION_METRIC' \
  --selection-mode '$SELECTION_MODE' \
  --val-max-batches '$VAL_MAX_BATCHES' \
  --eval-every-steps '$EVAL_EVERY_STEPS' \
  --min-selection-greedy-top1 '$MIN_SELECTION_GREEDY_TOP1' \
  --early-stop-selection-guard-failures '$EARLY_STOP_SELECTION_GUARD_FAILURES' \
  --early-stop-after-step '$EARLY_STOP_AFTER_STEP' \
  --swa-fraction '$SWA_FRACTION' \
  --checkpoint-dir '$CHECKPOINT_DIR' \
  --metrics-jsonl '$METRICS' \
  --out '$REPORT' \
  "\${TRAINER_INIT_ARGS[@]}" \
  "\${TRAINER_MIX_ARGS[@]}" \
  $TRAINER_EXTRA_ARGS
TRAINING_SECONDS="\$(( \$(date +%s) - TRAINING_STARTED ))"

python -m cascadiav3.torch_inference_bridge \
  --self-test-manifest-resolution
python -m cascadiav3.torch_inference_bridge \
  --self-test-inference-request cascadiav3/fixtures/expert_tiny.jsonl
python -m cascadiav3.validate_q_serving_semantics

TRAIN_GENERATE_SECONDS="\$TRAIN_GENERATE_SECONDS" \
VAL_GENERATE_SECONDS="\$VAL_GENERATE_SECONDS" \
TRAINING_SECONDS="\$TRAINING_SECONDS" \
TRAIN_INPUT="\$TRAIN_INPUT" \
python - <<'PY'
import json
import os
from pathlib import Path
report_path = Path('$RUNBOOK_REPORT')
train_manifest = json.loads(Path('$TRAIN_MANIFEST').read_text())
val_manifest = json.loads(Path('$VAL_MANIFEST').read_text())
train_report = json.loads(Path('$REPORT').read_text())
train_tensor_summary = json.loads(Path('cascadiav3/reports/full_v3_${PROFILE}_train_tensor_summary.json').read_text())
val_tensor_summary = json.loads(Path('cascadiav3/reports/full_v3_${PROFILE}_val_tensor_summary.json').read_text())
train_tail_summary = json.loads(Path('cascadiav3/reports/full_v3_${PROFILE}_train_tensor_top${FILTER_TOP_K}_relation_tail_summary.json').read_text())
val_tail_summary = json.loads(Path('cascadiav3/reports/full_v3_${PROFILE}_val_tensor_top${FILTER_TOP_K}_relation_tail_summary.json').read_text())
train_generate_seconds = max(1.0, float(os.environ.get("TRAIN_GENERATE_SECONDS", "0") or 0))
val_generate_seconds = max(1.0, float(os.environ.get("VAL_GENERATE_SECONDS", "0") or 0))
training_seconds = max(1.0, float(os.environ.get("TRAINING_SECONDS", "0") or 0))
root_count = float(train_tensor_summary["record_count"] + val_tensor_summary["record_count"])
generation_seconds = train_generate_seconds + val_generate_seconds
rollout_evals = root_count * float('$MAX_ACTIONS') * float('$ROLLOUTS_PER_ACTION')
report = {
    "status": "pass",
    "profile": "$PROFILE",
    "runbook": "docs/v3/TRAINING_PIPELINE.md",
    "source_revision": "$SOURCE_REVISION",
    "scale_note": "phase0 bootstrap uses packed expert_tensor_shard.v1 NPZ; JSONL is used only for tiny audit gates",
    "train_roots": train_manifest,
    "val_roots": val_manifest,
    "filter_top_k": $FILTER_TOP_K,
    "filter_mode": "$FILTER_MODE",
    "expert_tensor_mode": "$EXPERT_TENSOR_MODE",
    "objective": "$OBJECTIVE",
    "init_manifest": "$INIT_MANIFEST",
    "resume_manifest": "$RESUME_MANIFEST",
    "train_source_weights": "$TRAIN_SOURCE_WEIGHTS",
    "extra_train_tail_tensors": "$EXTRA_TRAIN_TAIL_TENSORS",
    "selection_metric": "$SELECTION_METRIC",
    "selection_mode": "$SELECTION_MODE",
    "eval_every_steps": $EVAL_EVERY_STEPS,
    "min_selection_greedy_top1": $MIN_SELECTION_GREEDY_TOP1,
    "early_stop_selection_guard_failures": $EARLY_STOP_SELECTION_GUARD_FAILURES,
    "early_stop_after_step": $EARLY_STOP_AFTER_STEP,
    "train_filtered_tensor": "$TRAIN_FILTERED_TENSOR",
    "val_filtered_tensor": "$VAL_FILTERED_TENSOR",
    "generated_train_tensor": "$TRAIN_TAIL_TENSOR",
    "train_tensor": os.environ.get("TRAIN_INPUT", "$TRAIN_TAIL_TENSOR"),
    "val_tensor": "$VAL_TAIL_TENSOR",
    "relation_tail_materialized": True,
    "training_report": train_report,
    "performance": {
        "train_generate_seconds": train_generate_seconds,
        "val_generate_seconds": val_generate_seconds,
        "generation_seconds": generation_seconds,
        "training_seconds": training_seconds,
        "roots_per_second": root_count / generation_seconds,
        "rollout_evals_per_second": rollout_evals / generation_seconds,
        "bytes_per_record": float(train_tail_summary["bytes_per_record"]),
        "train_step_seconds": training_seconds / max(1.0, float('$TRAIN_STEPS')),
        "train_record_count": train_tensor_summary["record_count"],
        "val_record_count": val_tensor_summary["record_count"],
        "train_tail_bytes_per_record": train_tail_summary["bytes_per_record"],
        "val_tail_bytes_per_record": val_tail_summary["bytes_per_record"],
    },
}
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n")
print(json.dumps({"status": "pass", "runbook_report": str(report_path)}, sort_keys=True))
PY

python -m cascadiav3.validate_runbook_performance \
  --runbook '$RUNBOOK_REPORT' \
  --require-positive roots_per_second,rollout_evals_per_second,bytes_per_record,train_step_seconds

/usr/lib/wsl/lib/nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv
echo "[full-v3] completed \$(date -Is)"
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
ps -eo pid=,ppid=,psr=,nlwp=,pcpu=,pmem=,etime=,args= | awk '
  /full_v3_training_pipeline/ && !/awk/ {print \"matching full_v3 pid \" \$1 \" ppid \" \$2 \" psr \" \$3 \" threads \" \$4 \" cpu \" \$5 \" mem \" \$6 \" elapsed \" \$7}
  /torch_train_cascadiaformer/ && !/awk/ {print \"matching trainer pid \" \$1 \" ppid \" \$2 \" psr \" \$3 \" threads \" \$4 \" cpu \" \$5 \" mem \" \$6 \" elapsed \" \$7}
  /cascadiav3-real-root-exporter/ && /(chance-mcts-dry-run|expert-tensor-corpus|greedy-state-search-bootstrap-tensor-corpus|model-state-search-bootstrap-tensor-corpus)/ && !/awk/ {print \"matching expert exporter pid \" \$1 \" ppid \" \$2 \" psr \" \$3 \" threads \" \$4 \" cpu \" \$5 \" mem \" \$6 \" elapsed \" \$7}
'
for f in '$TRAIN_ROOTS' '$VAL_ROOTS' '$TRAIN_TENSOR' '$VAL_TENSOR' '$TRAIN_FILTERED_TENSOR' '$VAL_FILTERED_TENSOR' '$TRAIN_TAIL_TENSOR' '$VAL_TAIL_TENSOR' '$TRAIN_MANIFEST' '$VAL_MANIFEST' '$REPORT' '$METRICS' '$RUNBOOK_REPORT'; do
  [ -e \"\$f\" ] && ls -lh \"\$f\"
done
if [ -n '$EXTRA_TRAIN_TAIL_TENSORS' ]; then
  IFS=',' read -r -a EXTRA_TRAIN_INPUT_PATHS <<< '$EXTRA_TRAIN_TAIL_TENSORS'
  for f in \"\${EXTRA_TRAIN_INPUT_PATHS[@]}\"; do
    [ -e \"\$f\" ] && ls -lh \"\$f\"
  done
fi
tail -n 120 '$REMOTE_LOG' 2>/dev/null || true"
}

fetch_artifacts() {
  cd "$LOCAL_ROOT"
  mkdir -p cascadiav3/reports cascadiav3/logs cascadiav3/checkpoints cascadiav3/fixtures
  fetch_file() {
    local rel="$1"
    if ssh -p "$SSH_PORT" "$REMOTE" "[ -e '$REMOTE_ROOT/$rel' ]"; then
      mkdir -p "$(dirname "$rel")"
      rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/$rel" "$(dirname "$rel")/"
    fi
  }
  fetch_dir() {
    local rel="$1"
    if ssh -p "$SSH_PORT" "$REMOTE" "[ -d '$REMOTE_ROOT/$rel' ]"; then
      mkdir -p "$rel"
      rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/$rel/" "$rel/"
    fi
  }

  fetch_file "$REPORT"
  fetch_file "$METRICS"
  fetch_file "$RUNBOOK_REPORT"
  fetch_file "cascadiav3/reports/full_v3_${PROFILE}_train_tensor_summary.json"
  fetch_file "cascadiav3/reports/full_v3_${PROFILE}_val_tensor_summary.json"
  fetch_file "cascadiav3/reports/full_v3_${PROFILE}_train_tensor_invariants.json"
  fetch_file "cascadiav3/reports/full_v3_${PROFILE}_val_tensor_invariants.json"
  fetch_file "cascadiav3/reports/full_v3_${PROFILE}_train_tensor_top${FILTER_TOP_K}_summary.json"
  fetch_file "cascadiav3/reports/full_v3_${PROFILE}_val_tensor_top${FILTER_TOP_K}_summary.json"
  fetch_file "cascadiav3/reports/full_v3_${PROFILE}_train_tensor_top${FILTER_TOP_K}_invariants.json"
  fetch_file "cascadiav3/reports/full_v3_${PROFILE}_val_tensor_top${FILTER_TOP_K}_invariants.json"
  fetch_file "cascadiav3/reports/full_v3_${PROFILE}_train_tensor_top${FILTER_TOP_K}_relation_tail_summary.json"
  fetch_file "cascadiav3/reports/full_v3_${PROFILE}_val_tensor_top${FILTER_TOP_K}_relation_tail_summary.json"
  fetch_file "cascadiav3/reports/cascadiaformer_game_benchmark_summary.md"
  fetch_file "$TRAIN_MANIFEST"
  fetch_file "$VAL_MANIFEST"
  fetch_file "cascadiav3/logs/${JOB_SLUG}_job.log"
  fetch_dir "$CHECKPOINT_DIR"
}

stop_job() {
  ssh -p "$SSH_PORT" "$REMOTE" "set -euo pipefail
if [ -s '$REMOTE_PID' ]; then
  pid=\"\$(cat '$REMOTE_PID')\"
  if kill -0 \"\$pid\" 2>/dev/null; then
    kill -TERM -\"\$pid\" 2>/dev/null || kill -TERM \"\$pid\" 2>/dev/null || true
  fi
fi
rm -f '$REMOTE_PID'
echo stopped"
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
  stop)
    stop_job
    ;;
  *)
    echo "usage: $0 {launch|status|fetch|stop}" >&2
    exit 2
    ;;
esac
