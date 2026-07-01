#!/bin/bash
# Fine-tune iter20 with Adam + cosine on SELF-PLAY data.
#
# Rationale:
# - iter1-20 used SGD lr=3e-5 on self-play data: slow but reliable (+0.7 over 20 iters)
# - small_v1 used Adam lr=1e-3 + cosine on MCE cache only: REGRESSED -4.1 pts (distribution mismatch)
#
# This test: does Adam+cosine (fast optimizer) work on SELF-PLAY data (correct distribution)?
# Key: same distribution as inference → gains should translate to play quality.
#
# Uses training_merged_iter9.bin (565MB) which has self-play + MCE mixed data
# consistent with how iter1-20 were trained.

set -e

DATA="${DATA:-training_merged_iter9.bin}"
LR="${LR:-0.0003}"
EPOCHS="${EPOCHS:-20}"
OUT="${OUT:-nnue_weights_iter20_adam.bin}"

echo "[$(date +%H:%M:%S)] Fine-tune iter20 with Adam+cosine on $DATA"
echo "[$(date +%H:%M:%S)] LR=$LR EPOCHS=$EPOCHS OUT=$OUT"

python3 train_pytorch.py value \
    --samples "$DATA" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --optimizer adam \
    --hidden1 512 --hidden2 64 \
    --init-weights nnue_weights_hybrid_iter20.bin \
    --out "$OUT" \
    --no-augment \
    2>&1 | tee "train_${OUT%.bin}.log"

echo "[$(date +%H:%M:%S)] Done. Benching..."
./target/release/cascadia-cli 200 --nnue --weights "$OUT" \
    > "bench_results/${OUT%.bin}_200g.log" 2>&1
mean=$(grep -m1 "Mean:" "bench_results/${OUT%.bin}_200g.log" | awk '{print $2}')
echo "[$(date +%H:%M:%S)] $OUT NNUE mean=$mean (iter20 baseline=90.9)"
