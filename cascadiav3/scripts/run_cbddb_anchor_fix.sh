#!/usr/bin/env bash
set -euo pipefail

# CBDDB recipe fix: trust-region anchored fine-tune. Stage 2's naive
# warm-start fine-tune REGRESSED (98.75 vs zero-shot 99.4675) — the same
# continuity-leak seen in D1. This retrains on the EXISTING Stage-2
# corpus (no regeneration) with the anchor terms that pin the fine-tune
# to the incumbent (AAAAA champion = the 99.47 zero-shot model).
#
# Two hypothesis-driven arms (both backstopped by the zero-shot floor:
# an over-strong anchor merely reproduces ~99.47, never worse):
#   A "vonly": value/score-head anchor only (kl=0) — tests the diagnosis
#              that the VALUE head drifted and hurt search-time blending.
#   B "both" : policy KL + value L2 — full trust region.
# Plus the missing control: zero-shot (champion, no fine-tune) at
# n1024/d16 x30 on the same block, so the Stage-2 ft n1024/d16 number
# finally has a paired champion-grade reference.
#
# Screen block 2027190000..99 (paired vs zero-shot 99.4675 @ n256/d4).

ROOT="${ROOT:-/home/john0/cascadia}"
SOURCE_REVISION="${SOURCE_REVISION:?set SOURCE_REVISION}"
BINARY="${BINARY:-cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter}"
PYTHON="${PYTHON:-python3}"
JOBS="${JOBS:-12}"
RULESET_ID="cascadia_research_cbddb_4p_no_habitat_bonus_rules_2026_07_19"
INCUMBENT="${INCUMBENT:-cascadiav3/checkpoints/full_v3_gumbel_selfplay_cycle4/best_locked_val.manifest.json}"
REPORT_DIR="${REPORT_DIR:-cascadiav3/reports}"
LOG_DIR="${LOG_DIR:-cascadiav3/logs}"
FIX="${FIX:-cascadiav3/fixtures}"
TRAIN_NPZ="$FIX/cbddb_s2_train_tensor.npz"
VAL_NPZ="$FIX/cbddb_s2_val_tensor.npz"
EVAL_FIRST_SEED=2027190000
KL_WEIGHT="${KL_WEIGHT:-2.0}"
L2_WEIGHT="${L2_WEIGHT:-2.0}"

export PATH="$HOME/.cargo/bin:$PATH:/usr/lib/wsl/lib"
if [ -x "$HOME/.local/bin/zig-cc" ] && ! command -v cc >/dev/null 2>&1; then
  export BLAKE3_NO_ASM=1 CC="$HOME/.local/bin/zig-cc"
  export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="$HOME/.local/bin/zig-cc"
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="cascadiav3/src"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CASCADIA_CGAB_FUSED=1
export CASCADIA_EVAL_CELL_BUDGET="${CASCADIA_EVAL_CELL_BUDGET:-16777216}"

cd "$ROOT"
mkdir -p "$REPORT_DIR" "$LOG_DIR"

hb(){ echo "[$(date "+%F %T")] [cbddb-anchor] $*"; }

grep -q 'rules_2026_07_19' cascadiav3/real-root-exporter/src/main.rs
grep -q 'anchor-policy-kl-weight' cascadiav3/src/cascadiav3/torch_train_cascadiaformer.py
test -s "$TRAIN_NPZ"; test -s "$VAL_NPZ"; test -s "$INCUMBENT"

hb "start rev=$SOURCE_REVISION kl=$KL_WEIGHT l2=$L2_WEIGHT corpus=$TRAIN_NPZ (reused)"

cargo build --release --manifest-path cascadiav3/real-root-exporter/Cargo.toml
if [ -f /home/john0/venvs/torch/bin/activate ]; then
  # shellcheck disable=SC1091
  source /home/john0/venvs/torch/bin/activate
fi

train_arm() {
  local arm="$1"; local kl="$2"; local l2="$3"
  local ckpt="cascadiav3/checkpoints/cbddb_anchor_${arm}"
  if [ -s "$ckpt/best_locked_val.manifest.json" ]; then
    hb "TRAIN $arm reuse"
    return
  fi
  hb "TRAIN $arm starting (kl=$kl l2=$l2)"
  if python3 -m cascadiav3.torch_train_cascadiaformer \
    --model-size M \
    --train "$TRAIN_NPZ" --val "$VAL_NPZ" \
    --train-format npz --val-format npz \
    --steps 2500 --batch-size 192 --grad-accum 1 \
    --lr 0.0001 --weight-decay 0.05 --warmup-fraction 0.02 \
    --device cuda --seed 20260630 \
    --objective gumbel-selfplay \
    --max-example-passes 4 \
    --q-quantiles 8 --init-skip-mismatched \
    --selection-metric locked_val_final_q_regret --selection-mode min \
    --val-max-batches 8 --eval-every-steps 250 \
    --swa-fraction 0.20 \
    --init-manifest "$INCUMBENT" \
    --anchor-manifest "$INCUMBENT" \
    --anchor-policy-kl-weight "$kl" \
    --anchor-value-l2-weight "$l2" \
    --data-workers 0 --tf32 --fused-optimizer --cgab-fused \
    --checkpoint-dir "$ckpt" \
    --metrics-jsonl "$REPORT_DIR/cbddb_anchor_${arm}_metrics.jsonl" \
    --out "$REPORT_DIR/cbddb_anchor_${arm}_train.json" \
    >> "$LOG_DIR/cbddb_anchor_${arm}_train.log" 2>&1; then
    hb "TRAIN $arm COMPLETE"
  else
    hb "TRAIN $arm FAILED"; return 1
  fi
}

report_matches() {
  local report="$1"
  [ -s "$report" ] && "$PYTHON" - "$report" "$RULESET_ID" "$SOURCE_REVISION" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if r.get("status")=="pass" and r.get("ruleset_id")==sys.argv[2] and r.get("source_revision")==sys.argv[3] else 1)
PY
}

run_eval() {
  local tag="$1"; local manifest="$2"; local sims="$3"; local dets="$4"; local games="$5"
  local report="$REPORT_DIR/cbddb_anchor_${tag}.json"
  if report_matches "$report"; then hb "EVAL $tag reuse"; return; fi
  hb "EVAL $tag starting (n${sims}/d${dets} x ${games})"
  "$PYTHON" -m cascadiav3.torch_cascadiaformer_gumbel_benchmark \
    --binary "$BINARY" --manifest "$manifest" --scoring-cards cbddb \
    --device cuda --first-seed "$EVAL_FIRST_SEED" --games "$games" --jobs "$JOBS" \
    --batch-runner \
    --gumbel-n-simulations "$sims" --gumbel-top-m 16 --gumbel-depth-rounds 1 \
    --gumbel-determinizations "$dets" --gumbel-market-decision-samples 8 \
    --gumbel-blend-weight 0.5 --k-interior 16 --control none \
    --model-timeout-ms 300000 --source-revision "$SOURCE_REVISION" \
    --experiment-id "cbddb_anchor_${tag}" \
    --out "$report" \
    --decisions-out "$REPORT_DIR/cbddb_anchor_${tag}_decisions.jsonl" \
    --games-out "$REPORT_DIR/cbddb_anchor_${tag}_games.jsonl" \
    --summary-out "$REPORT_DIR/cbddb_anchor_${tag}.md"
  hb "EVAL $tag DONE"
}

# Arm A: value/score anchor only. Arm B: full trust region.
train_arm vonly 0.0 "$L2_WEIGHT"
train_arm both "$KL_WEIGHT" "$L2_WEIGHT"

run_eval vonly_n256_d4 "cascadiav3/checkpoints/cbddb_anchor_vonly/best_locked_val.manifest.json" 256 4 100
run_eval both_n256_d4  "cascadiav3/checkpoints/cbddb_anchor_both/best_locked_val.manifest.json"  256 4 100

# Missing control: zero-shot champion at champion grade on the same block.
run_eval zeroshot_n1024_d16 "$INCUMBENT" 1024 16 30

hb "CBDDB ANCHOR FIX COMPLETE"
