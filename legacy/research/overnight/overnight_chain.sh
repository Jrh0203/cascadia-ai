#!/bin/bash
# Overnight chain orchestrator:
#   1. Wait for iter 30 HH (already running) to complete (50 games)
#   2. Run iters 31-40 training (LR 3e-6 → 1e-6)
#   3. Find best iter (lowest RMSE), launch HH on it
#
# Designed to run unattended (nohup). All steps log progress with timestamps.

set -uo pipefail   # don't fail on grep no-match

LOGFILE="overnight/v5sh/overnight_chain.log"
mkdir -p overnight/v5sh

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

log "=== Overnight chain started ==="

# STEP 1: Wait for iter 30 HH to complete
log "Step 1: Waiting for iter 30 HH to reach 50 games..."
HH_FILE="overnight/v5sh/hh_iter30_vs_v4.jsonl"
while true; do
  count=$(wc -l < "$HH_FILE" 2>/dev/null || echo 0)
  count=$(echo "$count" | tr -d ' ')
  if [ "${count:-0}" -ge 50 ]; then
    log "  iter 30 HH complete: $count games"
    break
  fi
  sleep 60
done

# Wait an extra 30 sec to ensure file is fully flushed and process is done
sleep 30

# STEP 2: Run iters 31-40 training
log "Step 2: Launching iters 31-40 training..."
./overnight/train_v5sh_continue3.sh > overnight/v5sh/continue_31_40.log 2>&1
log "  iters 31-40 training complete"

# STEP 3: Find best iter (lowest RMSE)
BEST_ITER=$(python3 - <<'PY'
import re
with open('overnight/v5sh/continue_31_40.log') as f:
    log = f.read()
iters = []
# Each iter prints "=== Iteration N ===" then later "Trained. RMSE=X"
for m in re.finditer(r'=== Iteration (\d+) ===.*?RMSE=([\d.]+)', log, re.DOTALL):
    iters.append((int(m.group(1)), float(m.group(2))))
if iters:
    best = min(iters, key=lambda x: x[1])
    print(best[0])
else:
    print(40)  # fallback
PY
)
log "  Best iter (by RMSE): $BEST_ITER"

# STEP 4: Launch HH for best iter
HH_OUT="overnight/v5sh/hh_iter${BEST_ITER}_vs_v4.jsonl"
HH_LOG="overnight/v5sh/hh_iter${BEST_ITER}.log"
log "Step 4: Launching HH on iter ${BEST_ITER}..."
nohup python3 overnight/hh_local_v5.py \
  --strategy-a mce_wide_v1 --weights-a "nnue_weights_v5sh_iter${BEST_ITER}.bin" \
  --strategy-b mce_wide_v1_b --weights-b nnue_weights_v4opp_modal_iter3.bin \
  --binary target-mid-v5/release/cascadia-cli \
  --num-games 50 --parallel 1 \
  --jsonl-out "$HH_OUT" \
  > "$HH_LOG" 2>&1 &
HH_PID=$!
log "  HH launched (PID $HH_PID), output: $HH_OUT"
log "=== Overnight chain handed off to HH; will run until 50 games done ==="
