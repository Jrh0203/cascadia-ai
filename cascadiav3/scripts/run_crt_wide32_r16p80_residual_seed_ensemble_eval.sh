#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-john0}"
SSH_PORT="${SSH_PORT:-2222}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/john0/cascadia}"
REMOTE_VENV="${REMOTE_VENV:-/home/john0/venvs/torch}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RSYNC_SSH="ssh -p ${SSH_PORT}"

INPUTS="${INPUTS:-cascadiav3/reports/crt_wide32_r16p80_semantic_residual_attention_prefilter_eval_roots.jsonl,cascadiav3/reports/crt_wide32_r16p80_semantic_residual_attention_seed31_prefilter_eval_roots.jsonl}"
WEIGHTS="${WEIGHTS:-}"
K_VALUES="${K_VALUES:-4,8,16,24,32}"
MIN_RECALL="${MIN_RECALL:-0.75}"
MAX_ORACLE_REGRET="${MAX_ORACLE_REGRET:-0.25}"
EXPERIMENT_ID="${EXPERIMENT_ID:-crt-wide32-r16p80-residual-seed-ensemble-v1}"
REPORT="${REPORT:-cascadiav3/reports/crt_wide32_r16p80_residual_seed_ensemble_eval.json}"
PER_ROOT_OUT="${PER_ROOT_OUT:-cascadiav3/reports/crt_wide32_r16p80_residual_seed_ensemble_eval_roots.jsonl}"
SUMMARY_OUT="${SUMMARY_OUT:-cascadiav3/reports/crt_wide32_r16p80_residual_seed_ensemble_eval_summary.md}"

cd "$LOCAL_ROOT"

ssh -p "$SSH_PORT" "$REMOTE" "mkdir -p '$REMOTE_ROOT/cascadiav3'"
rsync -az -e "$RSYNC_SSH" \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'target/' \
  cascadiav3/ "$REMOTE:$REMOTE_ROOT/cascadiav3/"

ssh -p "$SSH_PORT" "$REMOTE" "set -euo pipefail
cd '$REMOTE_ROOT'
. '$REMOTE_VENV/bin/activate'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m unittest discover -s cascadiav3/tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m cascadiav3.torch_prefilter_seed_ensemble_eval \
  --inputs '$INPUTS' \
  ${WEIGHTS:+--weights '$WEIGHTS'} \
  --k-values '$K_VALUES' \
  --min-recall '$MIN_RECALL' \
  --max-oracle-regret '$MAX_ORACLE_REGRET' \
  --experiment-id '$EXPERIMENT_ID' \
  --out '$REPORT' \
  --per-root-out '$PER_ROOT_OUT' \
  --summary-out '$SUMMARY_OUT'
"

rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/$REPORT" cascadiav3/reports/
rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/$PER_ROOT_OUT" cascadiav3/reports/
rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/$SUMMARY_OUT" cascadiav3/reports/
