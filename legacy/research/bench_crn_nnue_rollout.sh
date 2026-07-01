#!/bin/bash
# Second-round bench: CRN variance reduction + NNUE-guided rollouts.
# Baselines: B_halving (from previous bench).

set -u
GAMES=${GAMES:-50}
ROLLOUTS=${ROLLOUTS:-300}
OUT_DIR=${OUT_DIR:-bench_variants}
NNUE_WEIGHTS=${NNUE_WEIGHTS:-nnue_weights_v9_iter14.bin}
mkdir -p "$OUT_DIR"

echo "═══ CRN + NNUE-rollout bench: GAMES=$GAMES, ROLLOUTS=$ROLLOUTS ═══"
echo

run_bench() {
  local name="$1"
  shift
  local logf="$OUT_DIR/${name}.log"
  echo "[$(date +%H:%M:%S)] Running $name..."
  local start=$(date +%s)
  ./target/release/cascadia-cli "$GAMES" "$@" > "$logf" 2>&1
  local elapsed=$(($(date +%s) - start))
  local base=$(grep -A1 "Base Score" "$logf" | grep "Mean:" | awk '{print $2}' | head -1)
  local bonus=$(grep -A1 "With Habitat Bonus" "$logf" | grep "Mean:" | awk '{print $2}' | head -1)
  echo "  → base=$base bonus=$bonus (${elapsed}s)"
}

run_bench "F_crn"              --greedy-mce --rollouts "$ROLLOUTS" --alloc crn
run_bench "G_halving_crn"      --greedy-mce --rollouts "$ROLLOUTS" --alloc halving-crn
run_bench "H_nnue_halving"     --nnue-rollout-mce --rollouts "$ROLLOUTS" --alloc halving --weights "$NNUE_WEIGHTS"
run_bench "I_nnue_crn"         --nnue-rollout-mce --rollouts "$ROLLOUTS" --alloc crn --weights "$NNUE_WEIGHTS"
run_bench "J_nnue_halving_crn" --nnue-rollout-mce --rollouts "$ROLLOUTS" --alloc halving-crn --weights "$NNUE_WEIGHTS"

echo
echo "═══ SUMMARY ═══"
printf "%-20s  %6s  %6s\n" "variant" "base" "bonus"
for f in "$OUT_DIR"/*.log; do
  name=$(basename "$f" .log)
  base=$(grep -A1 "Base Score" "$f" | grep "Mean:" | awk '{print $2}' | head -1)
  bonus=$(grep -A1 "With Habitat Bonus" "$f" | grep "Mean:" | awk '{print $2}' | head -1)
  printf "%-20s  %6s  %6s\n" "$name" "${base:-?}" "${bonus:-?}"
done
