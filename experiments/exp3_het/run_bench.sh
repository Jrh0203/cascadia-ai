#!/bin/bash
set -uo pipefail
BIN=/Users/johnherrick/cascadia/target-mid-v4/release/cascadia-cli
W_BASE=/Users/johnherrick/cascadia/nnue_weights_v4opp_modal_iter3.bin
W_HET=/Users/johnherrick/cascadia/experiments/exp3_het/nnue_het_v1.bin
N=15
R=300
COMMON_ENV="MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 MCE_MUTATE_EXPAND=24"
COMMON_ARGS="--nnue-rollout-mce --candidates expanded --prefilter-k 8 --rollouts $R"
ts() { date +"%H:%M:%S"; }
run_one() {
  local label="$1"; shift; local extra_env="$1"; shift; local alloc="$1"; shift; local weights="$1"; shift
  echo "[$(ts)] === $label ===" | tee -a results.log
  env $extra_env $COMMON_ENV $BIN $N $COMMON_ARGS --alloc $alloc --weights $weights 2>&1 \
    | grep -E "Mean:|Wildlife:|Habitat:|Tokens:|Bear|Elk|Salmon|Hawk|Fox" | tee -a results.log
  echo "[$(ts)] done $label" | tee -a results.log
}
echo "=== Exp #3 + cv-fix N=$N R=$R Card A (rerun) ===" > results.log
run_one "baseline_halving"          ""                                                halving         $W_BASE
run_one "het_v1_halving"            ""                                                halving         $W_HET
run_one "cv_fixed_halving_basenet"  "MCE_CONTROL_VARIATES=1"                          halving         $W_BASE
run_one "cv_fixed_halving_hetnet"   "MCE_CONTROL_VARIATES=1"                          halving         $W_HET
echo "=== ALL DONE ===" | tee -a results.log
