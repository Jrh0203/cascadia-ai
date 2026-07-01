#!/bin/bash
set -uo pipefail
BIN=/Users/johnherrick/cascadia/target-mid-v4/release/cascadia-cli
W=/Users/johnherrick/cascadia/nnue_weights_v4opp_modal_iter3.bin
N=20
R=300
COMMON_ENV="MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 MCE_MUTATE_EXPAND=24"
COMMON_ARGS="--nnue-rollout-mce --candidates expanded --prefilter-k 8 --rollouts $R --weights $W"
ts() { date +"%H:%M:%S"; }
run_one() {
  local label="$1"; shift
  local extra_env="$1"; shift
  local alloc="$1"; shift
  echo "[$(ts)] === $label ===" | tee -a results.log
  env $extra_env $COMMON_ENV $BIN $N $COMMON_ARGS --alloc $alloc 2>&1 \
    | grep -E "Mean:|Wildlife:|Habitat:|Tokens:|Bear|Elk|Salmon|Hawk|Fox" \
    | tee -a results.log
  echo "[$(ts)] done $label" | tee -a results.log
}
echo "=== Exp #1 N=$N R=$R Card A ===" > results.log
run_one "baseline_halving"   ""                      halving
run_one "halving_ci_z0.5"    "MCE_HALVING_CI_Z=0.5"  halving-ci
run_one "halving_ci_z1.0"    "MCE_HALVING_CI_Z=1.0"  halving-ci
run_one "halving_ci_z1.5_floor" "MCE_HALVING_CI_Z=1.5 MCE_HALVING_CI_FLOOR=1" halving-ci
run_one "halving_hetero"     ""                      halving-hetero
echo "=== ALL DONE ===" | tee -a results.log
