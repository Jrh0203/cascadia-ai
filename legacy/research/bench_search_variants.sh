#!/bin/bash
# Head-to-head bench of greedy MCE variants.
# All runs use deterministic seeds for fair comparison.
#
# Variants compared:
#   A. uniform              (baseline — equal rollouts per candidate)
#   B. halving              (sequential halving)
#   C. ucb                  (UCB1 adaptive budget)
#   D. halving + expanded   (seq halving + larger candidate pool)
#   E. ucb + expanded       (UCB + larger candidate pool)

set -u
GAMES=${GAMES:-50}
ROLLOUTS=${ROLLOUTS:-300}
OUT_DIR=${OUT_DIR:-bench_variants}
mkdir -p "$OUT_DIR"

echo "═══ Head-to-head bench: GAMES=$GAMES, ROLLOUTS=$ROLLOUTS ═══"
echo

run_bench() {
  local name="$1"
  shift
  local logf="$OUT_DIR/${name}.log"
  echo "[$(date +%H:%M:%S)] Running $name..."
  local start=$(date +%s)
  ./target/release/cascadia-cli "$GAMES" --greedy-mce --rollouts "$ROLLOUTS" "$@" > "$logf" 2>&1
  local elapsed=$(($(date +%s) - start))
  local base=$(grep -A1 "Base Score" "$logf" | grep "Mean:" | awk '{print $2}' | head -1)
  local bonus=$(grep -A1 "With Habitat Bonus" "$logf" | grep "Mean:" | awk '{print $2}' | head -1)
  echo "  → base=$base bonus=$bonus (${elapsed}s)"
}

run_bench "A_uniform"        --alloc uniform
run_bench "B_halving"        --alloc halving
run_bench "C_ucb"            --alloc ucb
run_bench "D_halving_exp"    --alloc halving --candidates expanded
run_bench "E_ucb_exp"        --alloc ucb     --candidates expanded

echo
echo "═══ SUMMARY ═══"
printf "%-20s  %6s  %6s\n" "variant" "base" "bonus"
for f in "$OUT_DIR"/*.log; do
  name=$(basename "$f" .log)
  base=$(grep -A1 "Base Score" "$f" | grep "Mean:" | awk '{print $2}' | head -1)
  bonus=$(grep -A1 "With Habitat Bonus" "$f" | grep "Mean:" | awk '{print $2}' | head -1)
  printf "%-20s  %6s  %6s\n" "$name" "${base:-?}" "${bonus:-?}"
done
