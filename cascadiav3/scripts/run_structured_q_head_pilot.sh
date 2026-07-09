#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
INCUMBENT_MANIFEST="${INCUMBENT_MANIFEST:-cascadiav3/checkpoints/full_v3_gumbel_selfplay_cycle4/best_locked_val.manifest.json}"
FIT_TENSOR="${FIT_TENSOR:-cascadiav3/reports/structured_q_v4_20260709/train_a.npz}"
SELECTION_TENSOR="${SELECTION_TENSOR:-cascadiav3/reports/structured_q_v4_20260709/train_b.npz}"
VERDICT_TENSOR="${VERDICT_TENSOR:-cascadiav3/reports/structured_q_v4_20260709/val.npz}"
: "${FIT_SHA256:?set FIT_SHA256 to the immutable raw v4 shard hash}"
: "${SELECTION_SHA256:?set SELECTION_SHA256 to the immutable raw v4 shard hash}"
: "${VERDICT_SHA256:?set VERDICT_SHA256 to the immutable raw v4 shard hash}"

OUT_ROOT="${OUT_ROOT:-cascadiav3/reports/structured_q_head_pilot_20260709}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-cascadiav3/checkpoints/structured_q_head_pilot_20260709}"
FILTER_TOP_K="${FILTER_TOP_K:-64}"
STEPS="${STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-25}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-20260709}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=cascadiav3/src
mkdir -p "$OUT_ROOT" "$CHECKPOINT_ROOT"

check_hash() {
  local path="$1"
  local expected="$2"
  test -s "$path"
  printf '%s  %s\n' "$expected" "$path" | sha256sum --check --status
}

check_hash "$FIT_TENSOR" "$FIT_SHA256"
check_hash "$SELECTION_TENSOR" "$SELECTION_SHA256"
check_hash "$VERDICT_TENSOR" "$VERDICT_SHA256"
test -s "$INCUMBENT_MANIFEST"

prepare_tensor() {
  local label="$1"
  local raw="$2"
  local filtered="$OUT_ROOT/${label}_top${FILTER_TOP_K}.npz"
  local tail="$OUT_ROOT/${label}_top${FILTER_TOP_K}_relation_tail.npz"
  "$PYTHON" -m cascadiav3.expert_tensor_shards \
    --summarize-shard "$raw" \
    --report "$OUT_ROOT/${label}_raw_summary.json" >&2
  "$PYTHON" -m cascadiav3.validate_expert_tensor_invariants \
    --shard "$raw" \
    --require-q-equals-afterstate-plus-score-to-go \
    --report "$OUT_ROOT/${label}_raw_invariants.json" >&2
  if [ ! -s "$filtered" ]; then
    "$PYTHON" -m cascadiav3.expert_tensor_shards \
      --filter-shard "$raw" \
      --top-k "$FILTER_TOP_K" \
      --filter-mode top-q-with-selected \
      --out "$filtered" \
      --report "$OUT_ROOT/${label}_filter.json" >&2
  fi
  if [ ! -s "$tail" ]; then
    "$PYTHON" -m cascadiav3.expert_tensor_shards \
      --materialize-relation-tail "$filtered" \
      --out "$tail" \
      --report "$OUT_ROOT/${label}_tail.json" >&2
  fi
  "$PYTHON" - "$raw" "$filtered" "$tail" <<'PY'
import hashlib
import sys
from pathlib import Path
from cascadiav3.expert_tensor_shards import ExpertTensorShard

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

raw, filtered, tail = map(Path, sys.argv[1:])
filtered_shard = ExpertTensorShard(filtered)
try:
    assert filtered_shard.metadata["filter"]["source_sha256"] == sha256(raw)
finally:
    filtered_shard.close()
tail_shard = ExpertTensorShard(tail)
try:
    assert tail_shard.metadata["relation_tail"]["source_sha256"] == sha256(filtered)
finally:
    tail_shard.close()
PY
  "$PYTHON" -m cascadiav3.validate_expert_tensor_invariants \
    --shard "$tail" \
    --require-selected-action-dropped-count 0 \
    --require-q-equals-afterstate-plus-score-to-go \
    --report "$OUT_ROOT/${label}_tail_invariants.json" >&2
  printf '%s\n' "$tail"
}

FIT_TAIL="$(prepare_tensor fit "$FIT_TENSOR")"
SELECTION_TAIL="$(prepare_tensor selection "$SELECTION_TENSOR")"
VERDICT_TAIL="$(prepare_tensor verdict "$VERDICT_TENSOR")"

labels=(lr3e4 lr1e3 lr3e3)
learning_rates=(0.0003 0.001 0.003)
selection_args=()
for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  lr="${learning_rates[$index]}"
  checkpoint_dir="$CHECKPOINT_ROOT/$label"
  report="$OUT_ROOT/${label}_train.json"
  metrics="$OUT_ROOT/${label}_metrics.jsonl"
  if [ -e "$report" ] || [ -d "$checkpoint_dir" ]; then
    echo "refusing to mix a fresh arm with existing output: $label" >&2
    exit 2
  fi
  "$PYTHON" -m cascadiav3.torch_train_cascadiaformer \
    --model-size M \
    --q-decomposition \
    --q-decomposition-head-only \
    --train "$FIT_TAIL" \
    --val "$SELECTION_TAIL" \
    --train-format npz \
    --val-format npz \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum 1 \
    --lr "$lr" \
    --weight-decay 0 \
    --warmup-fraction 0.05 \
    --device "$DEVICE" \
    --seed "$SEED" \
    --objective gumbel-selfplay-structured-q \
    --max-example-passes 4 \
    --selection-metric locked_val_q_decomposition \
    --selection-mode min \
    --val-max-batches 0 \
    --eval-every-steps "$EVAL_EVERY_STEPS" \
    --swa-fraction 0.20 \
    --init-manifest "$INCUMBENT_MANIFEST" \
    --init-skip-mismatched \
    --autocast off \
    --checkpoint-dir "$checkpoint_dir" \
    --metrics-jsonl "$metrics" \
    --out "$report"
  selection_args+=(--arm "$label=$report")
done

SELECTION_REPORT="$OUT_ROOT/selection.json"
"$PYTHON" -m cascadiav3.select_structured_q_candidate \
  "${selection_args[@]}" \
  --out "$SELECTION_REPORT"
CANDIDATE_MANIFEST="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["chosen"]["manifest"])' "$SELECTION_REPORT")"

set +e
"$PYTHON" -m cascadiav3.torch_structured_q_probe \
  --candidate-manifest "$CANDIDATE_MANIFEST" \
  --incumbent-manifest "$INCUMBENT_MANIFEST" \
  --shards "$VERDICT_TAIL" \
  --device "$DEVICE" \
  --batch-size 8 \
  --seed "$SEED" \
  --out "$OUT_ROOT/heldout_verdict.json" \
  --markdown "$OUT_ROOT/heldout_verdict.md"
probe_status=$?
set -e
if [ "$probe_status" -eq 0 ]; then
  echo "[structured-q] held-out gate passed; no gameplay was launched"
elif [ "$probe_status" -eq 1 ] && "$PYTHON" -c \
  'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "fail"' \
  "$OUT_ROOT/heldout_verdict.json"; then
  echo "[structured-q] held-out gate failed scientifically; route closed before gameplay"
else
  echo "[structured-q] held-out probe crashed or produced no valid verdict" >&2
  exit "$probe_status"
fi
