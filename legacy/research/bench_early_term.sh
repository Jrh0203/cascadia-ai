#!/bin/bash
# Compare sequential halving with and without early termination.
set -u
GAMES=${GAMES:-30}
ROLLOUTS=${ROLLOUTS:-300}
OUT_DIR=${OUT_DIR:-bench_variants}
mkdir -p "$OUT_DIR"

echo "═══ Halving vs Halving+Early-Term: GAMES=$GAMES, ROLLOUTS=$ROLLOUTS ═══"

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

run_bench "K_halving_baseline"  --greedy-mce --rollouts "$ROLLOUTS" --alloc halving
run_bench "L_halving_et"        --greedy-mce --rollouts "$ROLLOUTS" --alloc halving-et

echo
echo "═══ SUMMARY ═══"
for name in K_halving_baseline L_halving_et; do
  f="$OUT_DIR/${name}.log"
  base=$(grep -A1 "Base Score" "$f" | grep "Mean:" | awk '{print $2}' | head -1)
  bonus=$(grep -A1 "With Habitat Bonus" "$f" | grep "Mean:" | awk '{print $2}' | head -1)
  printf "%-24s  base=%6s  bonus=%6s\n" "$name" "${base:-?}" "${bonus:-?}"
done
