#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-john0}"
SSH_PORT="${SSH_PORT:-2222}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/john0/cascadia}"
REMOTE_VENV="${REMOTE_VENV:-/home/john0/venvs/torch}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RSYNC_SSH="ssh -p ${SSH_PORT}"

VAL="${VAL:-cascadiav3/fixtures/crt_wide32_r16p80x2_semantic_val.jsonl}"
SEEDS="${SEEDS:-20260640,20260641,20260642}"
CHECKPOINT_PREFIX="${CHECKPOINT_PREFIX:-cascadiav3/checkpoints/crt_wide32_r16p80x2_semantic_residual_attention}"
CHECKPOINT_MEMBER="${CHECKPOINT_MEMBER:-mlp}"
REPORT_PREFIX="${REPORT_PREFIX:-cascadiav3/reports/crt_wide32_r16p80x2_semantic_${CHECKPOINT_MEMBER}_member}"
BATCH_SIZE="${BATCH_SIZE:-12}"
K_VALUES="${K_VALUES:-4,8,16,24,32}"
MIN_RECALL="${MIN_RECALL:-0.75}"
MAX_ORACLE_REGRET="${MAX_ORACLE_REGRET:-0.25}"
ENSEMBLE_REPORT="${ENSEMBLE_REPORT:-cascadiav3/reports/crt_wide32_r16p80x2_${CHECKPOINT_MEMBER}_seed_ensemble_3x_eval.json}"
ENSEMBLE_PER_ROOT_OUT="${ENSEMBLE_PER_ROOT_OUT:-cascadiav3/reports/crt_wide32_r16p80x2_${CHECKPOINT_MEMBER}_seed_ensemble_3x_eval_roots.jsonl}"
ENSEMBLE_SUMMARY_OUT="${ENSEMBLE_SUMMARY_OUT:-cascadiav3/reports/crt_wide32_r16p80x2_${CHECKPOINT_MEMBER}_seed_ensemble_3x_eval_summary.md}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-crt-wide32-r16p80x2-semantic-${CHECKPOINT_MEMBER}-member}"

cd "$LOCAL_ROOT"

ssh -p "$SSH_PORT" "$REMOTE" "mkdir -p '$REMOTE_ROOT/cascadiav3'"
rsync -az -e "$RSYNC_SSH" --delete \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  cascadiav3/src/ "$REMOTE:$REMOTE_ROOT/cascadiav3/src/"
rsync -az -e "$RSYNC_SSH" --delete \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  cascadiav3/tests/ "$REMOTE:$REMOTE_ROOT/cascadiav3/tests/"
rsync -az -e "$RSYNC_SSH" cascadiav3/scripts/ "$REMOTE:$REMOTE_ROOT/cascadiav3/scripts/"

ssh -p "$SSH_PORT" "$REMOTE" "set -euo pipefail
cd '$REMOTE_ROOT'
. '$REMOTE_VENV/bin/activate'
export LD_LIBRARY_PATH=/usr/lib/wsl/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
/usr/lib/wsl/lib/nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,temperature.gpu,power.draw,power.limit --format=csv
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m unittest discover -s cascadiav3/tests -v
inputs=''
for seed in \$(echo '$SEEDS' | tr ',' ' '); do
  checkpoint='${CHECKPOINT_PREFIX}_seed'\${seed}'_pilot.pt'
  report='${REPORT_PREFIX}_seed'\${seed}'_prefilter_eval.json'
  per_root='${REPORT_PREFIX}_seed'\${seed}'_prefilter_eval_roots.jsonl'
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m cascadiav3.torch_prefilter_eval \
    --val '$VAL' \
    --checkpoint \"\$checkpoint\" \
    --checkpoint-member '$CHECKPOINT_MEMBER' \
    --skip-baselines \
    --batch-size '$BATCH_SIZE' \
    --k-values '$K_VALUES' \
    --min-recall '$MIN_RECALL' \
    --max-oracle-regret '$MAX_ORACLE_REGRET' \
    --experiment-id '${EXPERIMENT_PREFIX}-seed'\${seed}'-prefilter-v1' \
    --out \"\$report\" \
    --per-root-out \"\$per_root\"
  if [ -z \"\$inputs\" ]; then
    inputs=\"\$per_root\"
  else
    inputs=\"\$inputs,\$per_root\"
  fi
done
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m cascadiav3.torch_prefilter_seed_ensemble_eval \
  --inputs \"\$inputs\" \
  --k-values '$K_VALUES' \
  --min-recall '$MIN_RECALL' \
  --max-oracle-regret '$MAX_ORACLE_REGRET' \
  --experiment-id '${EXPERIMENT_PREFIX}-seed-ensemble-3x-v1' \
  --out '$ENSEMBLE_REPORT' \
  --per-root-out '$ENSEMBLE_PER_ROOT_OUT' \
  --summary-out '$ENSEMBLE_SUMMARY_OUT'
"

rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/cascadiav3/reports/" cascadiav3/reports/
