#!/bin/bash
# Restart train_hybrid continuation with the CORRECT lr=3e-5 (SGD).
#
# My initial continuation used default lr=0.001 which is 33x too high
# (confirmed from train_10x_phase3.log: iter1-20 used lr=0.000030).
#
# This script:
# 1. Kills the current train_hybrid.py + its self-play/training children
# 2. Starts a fresh run from iter20 with --lr 0.00003 and --iter-offset 20
# 3. Uses a different --iter-prefix to avoid overwriting iter21 (if any)
#
# Usage: ./restart_training_correct_lr.sh

set -e

echo "[$(date +%H:%M:%S)] Looking for train_hybrid.py processes..."
pids=$(pgrep -f "train_hybrid.py" || true)
if [ -n "$pids" ]; then
    echo "[$(date +%H:%M:%S)] Found PIDs: $pids"
    echo "[$(date +%H:%M:%S)] Killing train_hybrid.py and child processes..."
    for pid in $pids; do
        # Kill child processes first (cascadia-cli self-play, train_pytorch)
        pkill -TERM -P "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in $pids; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 1
    for pid in $pids; do
        kill -KILL "$pid" 2>/dev/null || true
    done
else
    echo "[$(date +%H:%M:%S)] No train_hybrid.py running."
fi

# Also kill orphan cascadia-cli self-play if still running
selfplay_pids=$(pgrep -f "cascadia-cli.*--self-play" || true)
if [ -n "$selfplay_pids" ]; then
    echo "[$(date +%H:%M:%S)] Killing orphan self-play: $selfplay_pids"
    kill -TERM $selfplay_pids 2>/dev/null || true
fi

echo "[$(date +%H:%M:%S)] Starting fresh training with correct lr=3e-5..."
echo "[$(date +%H:%M:%S)] Iter offset: 20 (produces iter21-30)"
echo "[$(date +%H:%M:%S)] Init: nnue_weights_hybrid_iter20.bin"

nohup python3 train_hybrid.py \
    --iterations 10 \
    --self-play-games 100000 \
    --epochs-per-iter 15 \
    --lr 0.00003 \
    --init-weights nnue_weights_hybrid_iter20.bin \
    --iter-offset 20 \
    --iter-prefix nnue_weights_hybrid_v2_iter \
    --out nnue_weights_hybrid_continued_v2.bin \
    --benchmark-games 0 \
    > train_continue_iter21_v2.log 2>&1 &

echo "[$(date +%H:%M:%S)] Started pid $!"
echo "[$(date +%H:%M:%S)] Log: train_continue_iter21_v2.log"
