#!/bin/bash
# Validation bench: re-run best variant from primary/advanced batches
# with more games for tight confidence intervals.
#
# After other benches complete, run this with WINNER set, e.g.:
#   WINNER='--prefilter-k 8 --exact-endgame 2' bench_validate_winner.sh
set -u
GAMES=${GAMES:-100}
WEIGHTS=${WEIGHTS:-nnue_weights_v9_iter14.bin}
ROLLOUTS=${ROLLOUTS:-300}
OUT_DIR=${OUT_DIR:-overnight}
WINNER=${WINNER:---prefilter-k 8 --exact-endgame 2}
mkdir -p "$OUT_DIR"

echo "═══ Validation: GAMES=$GAMES, ROLLOUTS=$ROLLOUTS ═══"
echo "  Variant: nnue-rollout-mce halving $WINNER"
echo

# Re-run baseline at same game count for tight comparison
echo "[$(date +%H:%M:%S)] Running baseline_validation..."
start=$(date +%s)
./target/release/cascadia-cli "$GAMES" --nnue-rollout-mce --rollouts "$ROLLOUTS" --alloc halving \
  --weights "$WEIGHTS" > "$OUT_DIR/V0_baseline_${GAMES}g.log" 2>&1
elapsed=$(($(date +%s) - start))
base=$(grep -A1 "Base Score" "$OUT_DIR/V0_baseline_${GAMES}g.log" | grep "Mean:" | awk '{print $2}' | head -1)
bonus=$(grep -A1 "With Habitat Bonus" "$OUT_DIR/V0_baseline_${GAMES}g.log" | grep "Mean:" | awk '{print $2}' | head -1)
echo "  baseline → base=$base bonus=$bonus (${elapsed}s)"

# Winner with same scenarios
echo "[$(date +%H:%M:%S)] Running winner..."
start=$(date +%s)
./target/release/cascadia-cli "$GAMES" --nnue-rollout-mce --rollouts "$ROLLOUTS" --alloc halving \
  $WINNER --weights "$WEIGHTS" > "$OUT_DIR/V1_winner_${GAMES}g.log" 2>&1
elapsed=$(($(date +%s) - start))
base=$(grep -A1 "Base Score" "$OUT_DIR/V1_winner_${GAMES}g.log" | grep "Mean:" | awk '{print $2}' | head -1)
bonus=$(grep -A1 "With Habitat Bonus" "$OUT_DIR/V1_winner_${GAMES}g.log" | grep "Mean:" | awk '{print $2}' | head -1)
echo "  winner   → base=$base bonus=$bonus (${elapsed}s)"

echo
echo "═══ VALIDATION SUMMARY ═══"
for name in V0_baseline V1_winner; do
  f="$OUT_DIR/${name}_${GAMES}g.log"
  base=$(grep -A1 "Base Score" "$f" | grep "Mean:" | awk '{print $2}' | head -1)
  bonus=$(grep -A1 "With Habitat Bonus" "$f" | grep "Mean:" | awk '{print $2}' | head -1)
  printf "%-22s  base=%6s  bonus=%6s\n" "$name" "${base:-?}" "${bonus:-?}"
done
