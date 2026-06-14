#!/bin/bash
# Bench the 5 new features from items 1,2,4,7,8.
# Baseline: expanded + pf8 + 200r (current champion = 96.4/101.3).
# Each variant adds ONE feature on top of baseline for clean attribution.
set -u
GAMES=${GAMES:-30}
WEIGHTS=${WEIGHTS:-nnue_weights_v9_iter14.bin}
ROLLOUTS=${ROLLOUTS:-200}
OUT_DIR=${OUT_DIR:-overnight}
mkdir -p "$OUT_DIR"

run_bench() {
  local name="$1"
  shift
  local logf="$OUT_DIR/${name}.log"
  echo "[$(date +%H:%M:%S)] Running $name..."
  local start=$(date +%s)
  # Use `env -i` to clear inherited env, then re-apply PATH and per-variant vars
  env "$@" ./target/release/cascadia-cli "$GAMES" \
    --nnue-rollout-mce --candidates expanded --prefilter-k 8 --alloc halving \
    --rollouts "$ROLLOUTS" --weights "$WEIGHTS" > "$logf" 2>&1
  local elapsed=$(($(date +%s) - start))
  local base=$(grep -A1 "Base Score" "$logf" | grep "Mean:" | awk '{print $2}' | head -1)
  local bonus=$(grep -A1 "With Habitat Bonus" "$logf" | grep "Mean:" | awk '{print $2}' | head -1)
  echo "  → base=$base bonus=$bonus (${elapsed}s)"
}

run_bench_alloc() {
  local name="$1"
  local alloc="$2"
  shift 2
  local logf="$OUT_DIR/${name}.log"
  echo "[$(date +%H:%M:%S)] Running $name (alloc=$alloc)..."
  local start=$(date +%s)
  env "$@" ./target/release/cascadia-cli "$GAMES" \
    --nnue-rollout-mce --candidates expanded --prefilter-k 8 --alloc "$alloc" \
    --rollouts "$ROLLOUTS" --weights "$WEIGHTS" > "$logf" 2>&1
  local elapsed=$(($(date +%s) - start))
  local base=$(grep -A1 "Base Score" "$logf" | grep "Mean:" | awk '{print $2}' | head -1)
  local bonus=$(grep -A1 "With Habitat Bonus" "$logf" | grep "Mean:" | awk '{print $2}' | head -1)
  echo "  → base=$base bonus=$bonus (${elapsed}s)"
}

echo "═══ New Features: expanded + pf8 + 200r base ═══"
echo

# Baseline (reference point — already known 96.4/101.3)
run_bench "30_baseline_ref"

# #1: Control variate blending
run_bench "31_cv_alpha_0.85" MCE_CV_ALPHA=0.85
run_bench "32_cv_alpha_0.70" MCE_CV_ALPHA=0.70

# #2: LMR tiered budget
run_bench "33_lmr_only" MCE_LMR=1

# #8: Strategic commitment bias
run_bench "34_strategy_bias" MCE_STRATEGY_BIAS=1

# Combined: CV + LMR + strategy
run_bench "35_cv_lmr_strat" MCE_CV_ALPHA=0.85 MCE_LMR=1 MCE_STRATEGY_BIAS=1

# #7: Successive Rejects allocator
run_bench_alloc "36_successive_rejects" sr

# #4: Progressive widening allocator
run_bench_alloc "37_halving_pw" halving-pw

# Combined: best allocator (SR) + all features
run_bench_alloc "38_sr_cv_lmr_strat" sr MCE_CV_ALPHA=0.85 MCE_LMR=1 MCE_STRATEGY_BIAS=1

echo
echo "═══ NEW FEATURES SUMMARY ═══"
printf "%-28s  %8s  %8s\n" "Variant" "Base" "Bonus"
for name in 30_baseline_ref 31_cv_alpha_0.85 32_cv_alpha_0.70 33_lmr_only 34_strategy_bias \
             35_cv_lmr_strat 36_successive_rejects 37_halving_pw 38_sr_cv_lmr_strat; do
  f="$OUT_DIR/${name}.log"
  base=$(grep -A1 "Base Score" "$f" | grep "Mean:" | awk '{print $2}' | head -1)
  bonus=$(grep -A1 "With Habitat Bonus" "$f" | grep "Mean:" | awk '{print $2}' | head -1)
  printf "%-28s  %8s  %8s\n" "$name" "${base:-?}" "${bonus:-?}"
done
