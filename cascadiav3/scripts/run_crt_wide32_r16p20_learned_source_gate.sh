#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-john0}"
SSH_PORT="${SSH_PORT:-2222}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/john0/cascadia}"
REMOTE_VENV="${REMOTE_VENV:-/home/john0/venvs/torch}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RSYNC_SSH="ssh -p ${SSH_PORT}"

TRAIN="${TRAIN:-cascadiav3/fixtures/crt_wide32_r16p20_semantic_train.jsonl}"
VAL="${VAL:-cascadiav3/fixtures/crt_wide32_r16p20_semantic_val.jsonl}"
CHECKPOINT="${CHECKPOINT:-cascadiav3/checkpoints/crt_wide32_r16p20_semantic_action_set_pilot.pt}"
REPORT="${REPORT:-cascadiav3/reports/crt_wide32_r16p20_semantic_learned_source_gate.json}"
SUMMARY_OUT="${SUMMARY_OUT:-cascadiav3/reports/crt_wide32_r16p20_semantic_learned_source_gate_summary.md}"
EXPERIMENT_ID="${EXPERIMENT_ID:-crt-wide32-r16p20-semantic-learned-source-gate-v1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
STEPS="${STEPS:-2500}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
LR="${LR:-0.001}"
DROPOUT="${DROPOUT:-0.10}"
PAIRWISE_WEIGHT="${PAIRWISE_WEIGHT:-0.50}"
K="${K:-16}"

cd "$LOCAL_ROOT"

ssh -p "$SSH_PORT" "$REMOTE" "mkdir -p '$REMOTE_ROOT'"
rsync -az -e "$RSYNC_SSH" --delete \
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
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m cascadiav3.torch_prefilter_gate_eval \
  --train '$TRAIN' \
  --val '$VAL' \
  --checkpoint '$CHECKPOINT' \
  --batch-size '$BATCH_SIZE' \
  --steps '$STEPS' \
  --hidden-dim '$HIDDEN_DIM' \
  --lr '$LR' \
  --dropout '$DROPOUT' \
  --pairwise-weight '$PAIRWISE_WEIGHT' \
  --k '$K' \
  --experiment-id '$EXPERIMENT_ID' \
  --out '$REPORT' \
  --summary-out '$SUMMARY_OUT'
"

rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/cascadiav3/reports/" cascadiav3/reports/
