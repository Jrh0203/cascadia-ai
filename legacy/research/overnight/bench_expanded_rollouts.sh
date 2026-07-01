#!/bin/bash
# The big finding is: expanded candidates + prefilter k=8 = 96.4 / 101.4
# Does it scale with more rollouts? Test 500/750 rollouts.
set -u
GAMES=${GAMES:-30}
WEIGHTS=${WEIGHTS:-nnue_weights_v9_iter14.bin}
OUT_DIR=${OUT_DIR:-overnight}
mkdir -p "$OUT_DIR"

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

echo "═══ Expanded + Prefilter-8: Rollout Sweep ═══"

run_bench "20_expanded_pf8_r200" --nnue-rollout-mce --rollouts 200 --alloc halving --candidates expanded --prefilter-k 8 --weights "$WEIGHTS"
run_bench "21_expanded_pf8_r500" --nnue-rollout-mce --rollouts 500 --alloc halving --candidates expanded --prefilter-k 8 --weights "$WEIGHTS"
run_bench "22_expanded_pf8_r750" --nnue-rollout-mce --rollouts 750 --alloc halving --candidates expanded --prefilter-k 8 --weights "$WEIGHTS"

echo
echo "═══ SUMMARY ═══"
for name in 20_expanded_pf8_r200 21_expanded_pf8_r500 22_expanded_pf8_r750; do
  f="$OUT_DIR/${name}.log"
  base=$(grep -A1 "Base Score" "$f" | grep "Mean:" | awk '{print $2}' | head -1)
  bonus=$(grep -A1 "With Habitat Bonus" "$f" | grep "Mean:" | awk '{print $2}' | head -1)
  printf "%-26s  base=%6s  bonus=%6s\n" "$name" "${base:-?}" "${bonus:-?}"
done
