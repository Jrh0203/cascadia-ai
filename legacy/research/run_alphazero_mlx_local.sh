#!/usr/bin/env bash
set -euo pipefail

# Incremental local AlphaZero/MLX training loop.
# Rust collects exact self-play data; MLX trains on Apple Silicon; every shard
# emits a fresh checkpoint so interrupted runs keep their progress.

PYTHON_BIN="${PYTHON_BIN:-/Users/johnherrick/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3}"
RUN_DIR="${RUN_DIR:-alphazero_mlx_run}"
FEATURES="${FEATURES:-v4-opp,v5-feat,czero-feat}"
BIN="${BIN:-./target/release/cascadia-cli}"

BOOTSTRAP_GAMES="${BOOTSTRAP_GAMES:-128}"
SHARDS="${SHARDS:-8}"
GAMES_PER_SHARD="${GAMES_PER_SHARD:-12}"
SIMS="${SIMS:-24}"
THREADS="${THREADS:-4}"
MIN_SIMS_PER_THREAD="${MIN_SIMS_PER_THREAD:-6}"
CHANNELS="${CHANNELS:-16}"
BLOCKS="${BLOCKS:-2}"
HIDDEN="${HIDDEN:-64}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BOOTSTRAP_EPOCHS="${BOOTSTRAP_EPOCHS:-4}"
SHARD_EPOCHS="${SHARD_EPOCHS:-4}"
LR_BOOTSTRAP="${LR_BOOTSTRAP:-0.0003}"
LR_SHARD="${LR_SHARD:-0.0002}"
SEED="${SEED:-7000}"
BENCH_GAMES="${BENCH_GAMES:-5}"

mkdir -p "$RUN_DIR"

if [[ ! -x "$BIN" ]]; then
  RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}" \
    cargo build --release --features "$FEATURES" --bin cascadia-cli
fi

if [[ ! -f "$RUN_DIR/bootstrap.azd" ]]; then
  "$BIN" "$BOOTSTRAP_GAMES" --az-collect \
    --out "$RUN_DIR/bootstrap.azd" \
    --seed "$SEED"
fi

if [[ ! -f "$RUN_DIR/az_iter000.azr" ]]; then
  "$PYTHON_BIN" train_alphazero_mlx.py \
    --samples "$RUN_DIR/bootstrap.azd" \
    --out "$RUN_DIR/az_iter000.azr" \
    --channels "$CHANNELS" --blocks "$BLOCKS" --hidden "$HIDDEN" \
    --epochs "$BOOTSTRAP_EPOCHS" --batch-size "$BATCH_SIZE" \
    --lr "$LR_BOOTSTRAP" --save-each-epoch
fi

prev="$RUN_DIR/az_iter000.azr"
for iter in $(seq 1 "$SHARDS"); do
  shard=$(printf "%03d" "$iter")
  data="$RUN_DIR/iter${shard}.azd"
  out="$RUN_DIR/az_iter${shard}.azr"
  if [[ ! -f "$data" ]]; then
    "$BIN" "$GAMES_PER_SHARD" --az-collect \
      --az-weights "$prev" \
      --az-sims "$SIMS" \
      --temperature 1.0 \
      --az-parallel --az-threads "$THREADS" \
      --az-min-sims-per-thread "$MIN_SIMS_PER_THREAD" \
      --out "$data" \
      --seed "$((SEED + iter))"
  fi
  if [[ ! -f "$out" ]]; then
    "$PYTHON_BIN" train_alphazero_mlx.py \
      --samples "$RUN_DIR"/bootstrap.azd "$RUN_DIR"/iter*.azd \
      --init "$prev" \
      --out "$out" \
      --epochs "$SHARD_EPOCHS" --batch-size "$BATCH_SIZE" \
      --lr "$LR_SHARD" --save-each-epoch
  fi
  "$BIN" "$BENCH_GAMES" --az \
    --weights "$out" \
    --az-sims "$SIMS" \
    --az-parallel --az-threads "$THREADS" \
    --az-min-sims-per-thread "$MIN_SIMS_PER_THREAD"
  prev="$out"
done
