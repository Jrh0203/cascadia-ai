#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-john0}"
SSH_PORT="${SSH_PORT:-2222}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/john0/cascadia}"
REMOTE_VENV="${REMOTE_VENV:-/home/john0/venvs/torch}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RSYNC_SSH="ssh -p ${SSH_PORT}"

VAL="${VAL:-cascadiav3/fixtures/crt_wide32_sampled_teacher_val.jsonl}"
CHECKPOINT="${CHECKPOINT:-cascadiav3/checkpoints/crt_wide32_sampled_teacher_relation_bias_pilot.pt}"
REPORT="${REPORT:-cascadiav3/reports/crt_wide32_prefilter_eval.json}"
PER_ROOT_OUT="${PER_ROOT_OUT:-cascadiav3/reports/crt_wide32_prefilter_eval_roots.jsonl}"

BATCH_SIZE="${BATCH_SIZE:-12}"
K_VALUES="${K_VALUES:-4,8,16,24,32}"
MIN_RECALL="${MIN_RECALL:-0.75}"
MAX_ORACLE_REGRET="${MAX_ORACLE_REGRET:-0.25}"
EXPERIMENT_ID="${EXPERIMENT_ID:-crt-wide32-relation-bias-prefilter-eval-v1}"

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
export LD_LIBRARY_PATH=/usr/lib/wsl/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
/usr/lib/wsl/lib/nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,temperature.gpu,power.draw,power.limit --format=csv
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m unittest discover -s cascadiav3/tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m cascadiav3.torch_prefilter_eval \
  --val '$VAL' \
  --checkpoint '$CHECKPOINT' \
  --batch-size '$BATCH_SIZE' \
  --k-values '$K_VALUES' \
  --min-recall '$MIN_RECALL' \
  --max-oracle-regret '$MAX_ORACLE_REGRET' \
  --experiment-id '$EXPERIMENT_ID' \
  --out '$REPORT' \
  --per-root-out '$PER_ROOT_OUT'
/usr/lib/wsl/lib/nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,power.draw --format=csv
"

rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/$REPORT" cascadiav3/reports/
rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/$PER_ROOT_OUT" cascadiav3/reports/
