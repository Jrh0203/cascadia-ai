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
  local label="$1"; shift; local extra_env="$1"; shift; local alloc="$1"; shift
  echo "[$(ts)] === $label ===" | tee -a results.log
  env $extra_env $COMMON_ENV $BIN $N $COMMON_ARGS --alloc $alloc 2>&1 \
    | grep -E "Mean:|Wildlife:|Habitat:|Tokens:|Bear|Elk|Salmon|Hawk|Fox" | tee -a results.log
  echo "[$(ts)] done $label" | tee -a results.log
}
echo "=== Exp #2 N=$N R=$R Card A — control variates ===" > results.log
# Re-baseline with NEW binary (cv overhead always present, env off = same as exp1 baseline)
run_one "baseline_halving_newbin"        ""                                   halving
# CV applied at decision time, halving allocator
run_one "halving_cv_on"                  "MCE_CONTROL_VARIATES=1"             halving
# CV with halving-hetero (exp #1 + exp #2 compounded)
run_one "hetero_cv_on"                   "MCE_CONTROL_VARIATES=1"             halving-hetero
# CV with tighter beta cap
run_one "halving_cv_on_betacap1"         "MCE_CONTROL_VARIATES=1 MCE_CV_BETA_CAP=1.0"  halving
# CV with mid-eval at turn 4 (later = more variance to soak up)
run_one "halving_cv_on_turn4"            "MCE_CONTROL_VARIATES=1 MCE_CV_AT_TURN=4"     halving
echo "=== ALL DONE ===" | tee -a results.log
