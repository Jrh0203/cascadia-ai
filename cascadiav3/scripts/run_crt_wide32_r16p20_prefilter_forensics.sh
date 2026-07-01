#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-john0}"
SSH_PORT="${SSH_PORT:-2222}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/john0/cascadia}"
REMOTE_VENV="${REMOTE_VENV:-/home/john0/venvs/torch}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RSYNC_SSH="ssh -p ${SSH_PORT}"

VAL="${VAL:-cascadiav3/fixtures/crt_wide32_r16p20_semantic_val.jsonl}"
CHECKPOINT="${CHECKPOINT:-cascadiav3/checkpoints/crt_wide32_r16p20_semantic_action_set_pilot.pt}"
REPORT="${REPORT:-cascadiav3/reports/crt_wide32_r16p20_semantic_prefilter_forensics.json}"
SUMMARY_OUT="${SUMMARY_OUT:-cascadiav3/reports/crt_wide32_r16p20_semantic_prefilter_forensics_summary.md}"
EXPERIMENT_ID="${EXPERIMENT_ID:-crt-wide32-r16p20-semantic-prefilter-forensics-v1}"
BATCH_SIZE="${BATCH_SIZE:-12}"
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
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m cascadiav3.torch_prefilter_forensics \
  --val '$VAL' \
  --checkpoint '$CHECKPOINT' \
  --batch-size '$BATCH_SIZE' \
  --k '$K' \
  --experiment-id '$EXPERIMENT_ID' \
  --out '$REPORT' \
  --summary-out '$SUMMARY_OUT'
"

rsync -az -e "$RSYNC_SSH" "$REMOTE:$REMOTE_ROOT/cascadiav3/reports/" cascadiav3/reports/
