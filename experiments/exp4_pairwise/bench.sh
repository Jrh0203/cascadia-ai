#!/bin/bash
set -uo pipefail
BIN=/Users/johnherrick/cascadia/target-mid-v4/release/cascadia-cli
W_BASE=/Users/johnherrick/cascadia/nnue_weights_v4opp_modal_iter3.bin
W_PAIR=/Users/johnherrick/cascadia/experiments/exp4_pairwise/nnue_pairwise_v2.bin
N=15; R=300
COMMON_ENV="MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 MCE_MUTATE_EXPAND=24"
COMMON_ARGS="--nnue-rollout-mce --candidates expanded --prefilter-k 8 --rollouts $R --alloc halving"
ts() { date +"%H:%M:%S"; }
run_one() { local label="$1"; shift; local weights="$1"; shift
  echo "[$(ts)] === $label ===" | tee -a results.log
  env $COMMON_ENV $BIN $N $COMMON_ARGS --weights $weights 2>&1 \
    | grep -E "Mean:|Wildlife:|Habitat:|Tokens:|Bear|Elk|Salmon|Hawk|Fox" | tee -a results.log
  echo "[$(ts)] done $label" | tee -a results.log
}
echo "=== Exp #4 pairwise N=$N R=$R Card A ===" > results.log
run_one "baseline_halving" $W_BASE
run_one "pairwise_v2_halving" $W_PAIR
echo "=== ALL DONE ===" | tee -a results.log
