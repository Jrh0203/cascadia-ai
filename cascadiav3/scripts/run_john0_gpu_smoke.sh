#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-john0}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/john0/cascadia}"
REMOTE_VENV="${REMOTE_VENV:-/home/john0/venvs/torch}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$LOCAL_ROOT"

rsync -az --delete \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'target/' \
  cascadiav3/ "$REMOTE:$REMOTE_ROOT/cascadiav3/"

ssh "$REMOTE" "set -euo pipefail
cd '$REMOTE_ROOT'
. '$REMOTE_VENV/bin/activate'
export LD_LIBRARY_PATH=/usr/lib/wsl/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
/usr/lib/wsl/lib/nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,temperature.gpu,power.draw,power.limit --format=csv
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m unittest discover -s cascadiav3/tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m cascadiav3.validate --write-artifacts
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m cascadiav3.torch_gpu_smoke
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m cascadiav3.torch_train_tiny --steps 300
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m cascadiav3.torch_train_replay --steps 400 --batch-size 2
if [ -f cascadiav3/fixtures/real_roots.jsonl ]; then
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python -m cascadiav3.torch_train_replay \
    --replay cascadiav3/fixtures/real_roots.jsonl \
    --steps 500 \
    --batch-size 2 \
    --out cascadiav3/reports/real_replay_train.json \
    --checkpoint cascadiav3/checkpoints/real_replay_train.pt
fi
/usr/lib/wsl/lib/nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,power.draw --format=csv
"

rsync -az "$REMOTE:$REMOTE_ROOT/cascadiav3/reports/" cascadiav3/reports/
rsync -az "$REMOTE:$REMOTE_ROOT/cascadiav3/checkpoints/" cascadiav3/checkpoints/
