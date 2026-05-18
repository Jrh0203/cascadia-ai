#!/usr/bin/env bash
# Exp #6 — Cross-turn MCTS tree reuse benchmark suite.
#
# Validates the AUTONOMOUS_RESEARCH_REPORT.md "Exp #6 — Cross-turn tree
# reuse" hypothesis by running matched pairs at the same R=600 budget.
# Same C_UCB, same MAX_TREE_DEPTH, same parallel forest, same seed regime —
# only difference is whether the tree persists across turns.
#
# Usage: ./run_bench.sh [N=20]
set -euo pipefail
cd "$(dirname "$0")/../.."

N="${1:-20}"
BIN=./target-mid-v4/release/cascadia-cli
WEIGHTS=nnue_weights_v4opp_modal_iter3.bin
OUT="experiments/exp6_mcts_tree/results.log"
: >"$OUT"

run_bench() {
  local label="$1"; shift
  echo "=== $label (N=$N) ===" | tee -a "$OUT"
  { time "$BIN" "$N" "$@"; } 2>&1 | tee -a "$OUT"
  echo | tee -a "$OUT"
}

# A: cross-turn MCTS, greedy rollouts (the new mechanism)
run_bench "A: mcts-tree (cross-turn, greedy rollouts) R=600 parallel" \
    --mcts-tree --simulations 600 --parallel

# B: flat single-turn UCT, greedy rollouts (no cross-turn reuse — control)
run_bench "B: uct-mcts (flat, greedy rollouts) R=600 parallel" \
    --uct-mcts --simulations 600 --parallel

# C: cross-turn MCTS, NNUE rollouts (champion weights)
run_bench "C: mcts-tree (cross-turn, NNUE rollouts) R=600 parallel" \
    --mcts-tree --simulations 600 --parallel --weights "$WEIGHTS"

# D: flat MCE at same R=600 (the existing baseline shape — for context)
run_bench "D: greedy-mce (flat halving, no tree) R=600" \
    --greedy-mce --rollouts 600 --alloc halving --candidates expanded

echo "Done. Results: $OUT"
