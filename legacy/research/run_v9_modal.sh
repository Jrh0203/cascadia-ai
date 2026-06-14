#!/bin/bash
# Resilient wrapper for v9 Modal training.
# Auto-detects latest completed iter from Volume and resumes.
# Retries on network/heartbeat failures.
#
# Usage: ./run_v9_modal.sh [MAX_ITER]   (default 15)

set -u
MAX_ITER=${1:-15}
RUN_NAME=${RUN_NAME:-v9}
SP_GAMES=${SP_GAMES:-100000}
NUM_WORKERS=${NUM_WORKERS:-20}
EPOCHS=${EPOCHS:-15}
LR=${LR:-0.0001}
BATCH=${BATCH:-4096}
H1=${H1:-512}
H2=${H2:-64}
NF=${NF:-45260}
EPS=${EPS:-0.1}
BG=${BG:-50}
RETRY_BACKOFF=60   # seconds between retry attempts

echo "═══ Resilient v9 runner ═══"
echo "  Target: $MAX_ITER total iterations"
echo "  Run name: $RUN_NAME"
echo

while true; do
  # Find the highest completed iter by looking at local weight files
  LAST_ITER=0
  for i in $(seq $MAX_ITER -1 1); do
    if [ -f "nnue_weights_${RUN_NAME}_iter${i}.bin" ]; then
      LAST_ITER=$i
      break
    fi
  done

  if [ "$LAST_ITER" -ge "$MAX_ITER" ]; then
    echo "═══ All $MAX_ITER iterations complete. Exiting. ═══"
    break
  fi

  REMAINING=$((MAX_ITER - LAST_ITER))
  INIT_ARG=""
  if [ "$LAST_ITER" -gt 0 ]; then
    INIT_FILE="nnue_weights_${RUN_NAME}_iter${LAST_ITER}.bin"
    INIT_ARG="--init-weights $INIT_FILE"
  fi

  echo "[$(date +%H:%M:%S)] Last completed iter: $LAST_ITER"
  echo "[$(date +%H:%M:%S)] Launching $REMAINING more iters (iters $((LAST_ITER+1))-$MAX_ITER)"

  python3 -m modal run modal_collect.py::train_modal \
    --run-name "$RUN_NAME" \
    --iterations "$REMAINING" \
    --iter-offset "$LAST_ITER" \
    $INIT_ARG \
    --self-play-games "$SP_GAMES" \
    --num-workers "$NUM_WORKERS" \
    --epochs-per-iter "$EPOCHS" \
    --lr "$LR" \
    --batch-size "$BATCH" \
    --hidden1 "$H1" \
    --hidden2 "$H2" \
    --num-features "$NF" \
    --epsilon "$EPS" \
    --benchmark-games "$BG" 2>&1 | tee -a "train_${RUN_NAME}_modal.log"

  RC=${PIPESTATUS[0]}
  echo "[$(date +%H:%M:%S)] modal run exited with code $RC"

  # Try to salvage any completed iter that's in the Volume but not local
  echo "[$(date +%H:%M:%S)] Checking Volume for newly completed iters..."
  for i in $(seq $((LAST_ITER+1)) $MAX_ITER); do
    LOCAL="nnue_weights_${RUN_NAME}_iter${i}.bin"
    if [ ! -f "$LOCAL" ]; then
      # Try to download from Volume (quietly — OK if it doesn't exist)
      PADDED=$(printf "%02d" $i)
      if python3 -m modal run modal_collect.py::download_checkpoint \
          --path "iter${PADDED}/weights.bin" --out "$LOCAL" 2>/dev/null; then
        echo "  Recovered $LOCAL from Volume"
      fi
    fi
  done

  # If modal run succeeded, we're done
  if [ "$RC" -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] modal run succeeded cleanly."
    # Loop will verify all iters are present and exit
    continue
  fi

  # Failure — backoff and retry
  echo "[$(date +%H:%M:%S)] Backing off ${RETRY_BACKOFF}s before retry..."
  sleep "$RETRY_BACKOFF"
done
