#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-john0}"
SSH_PORT="${SSH_PORT:-2222}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/john0/cascadia}"
REMOTE_VENV="${REMOTE_VENV:-/home/john0/venvs/torch}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RSYNC_SSH="ssh -p ${SSH_PORT}"

TRAIN="${TRAIN:-cascadiav3/fixtures/crt_wide32_r16_sampled_teacher_train.jsonl}"
VAL="${VAL:-cascadiav3/fixtures/crt_wide32_r16_sampled_teacher_val.jsonl}"
CHECKPOINT="${CHECKPOINT:-cascadiav3/checkpoints/crt_wide32_r16_sampled_teacher_relation_bias_pilot.pt}"
REPORT="${REPORT:-cascadiav3/reports/crt_wide32_r16_prefilter_blend_eval.json}"
BATCH_SIZE="${BATCH_SIZE:-12}"
GRID_STEP="${GRID_STEP:-0.1}"
TARGET_K="${TARGET_K:-16}"
K_VALUES="${K_VALUES:-4,8,16,24,32}"
EXPERIMENT_ID="${EXPERIMENT_ID:-crt-wide32-r16-prefilter-blend-eval-v1}"

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
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m cascadiav3.torch_prefilter_blend_eval \
  --train '$TRAIN' \
  --val '$VAL' \
  --checkpoint '$CHECKPOINT' \
  --batch-size '$BATCH_SIZE' \
  --grid-step '$GRID_STEP' \
  --target-k '$TARGET_K' \
  --k-values '$K_VALUES' \
  --experiment-id '$EXPERIMENT_ID' \
  --out '$REPORT'
/usr/lib/wsl/lib/nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,power.draw --format=csv
"

rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/$REPORT" cascadiav3/reports/
