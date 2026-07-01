# Cascadia AlphaZero MLX Runbook

This pipeline uses the AlphaGo Zero outer loop, not the old NNUE-only loop:

1. Rust generates exact legal candidates and PUCT/self-play data.
2. MLX trains the policy/value network on Apple Silicon.
3. MLX writes `AZR1` weights that Rust can load for PUCT search with `--az`.

The model is a compact AlphaZero-style residual CNN over a dense 21x21 hex-grid
tensor. The input has 65 planes: board occupancy, terrain, wildlife, allowed
wildlife, market state, bag state, own progress, opponent progress, nature-token
context, and turn/player context. The policy head scores each legal candidate by
factoring it into tile cell, wildlife cell, market slot, wildlife market slot,
and skip-wildlife terms. This keeps search fast enough for local PUCT while still
training the heavy math on the Apple GPU.

## Architecture Choice

A transformer-only model is not the default here. Cascadia has both spatial
locality and set/race structure:

- Spatial adjacency drives most wildlife and habitat scoring, so a ResNet has the
  right inductive bias and is much faster per PUCT node.
- Market, bag, opponent races, and nature-token decisions are more set-like, so a
  future top model should probably be a hybrid: residual board trunk plus
  set-attention blocks over market/candidates/opponents.
- The immediate local system keeps the AlphaGo Zero-style ResNet core because
  PUCT calls the network thousands of times. On a Mac mini, inference latency is
  a first-class constraint.

In short: AlphaGo Zero is the right training/search loop. A hybrid ResNet plus
set-attention is probably the right long-term architecture. The current MLX
pipeline is the first exact, fast, locally trainable version of that loop.

## Python Runtime

The system Python may not have a working `pip`. The Codex bundled Python has
working NumPy and was used to install MLX:

```bash
/Users/johnherrick/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  -m pip install --user mlx
```

Verify MLX is using the GPU:

```bash
/Users/johnherrick/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  - <<'PY'
import mlx.core as mx
print(mx.default_device())
PY
```

Expected: `Device(gpu, 0)`.

## Smoke

```bash
cargo test --features v4-opp,v5-feat,czero-feat alphazero -- --nocapture

cargo run --release --features v4-opp,v5-feat,czero-feat --bin cascadia-cli -- \
  2 --az-collect --out /tmp/az_bootstrap.azd --seed 123

/Users/johnherrick/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  train_alphazero_mlx.py \
  --samples /tmp/az_bootstrap.azd \
  --out /tmp/az_mlx_smoke.azr \
  --channels 4 --blocks 1 --hidden 8 \
  --epochs 1 --batch-size 32 --lr 0.001 --save-each-epoch

cargo run --release --features v4-opp,v5-feat,czero-feat --bin cascadia-cli -- \
  1 --az --weights /tmp/az_mlx_smoke.azr --az-sims 4
```

## Local Play Speed

The default `--az` search is serial PUCT. For local play and data collection,
enable root-parallel PUCT:

```bash
cargo run --release --features v4-opp,v5-feat,czero-feat --bin cascadia-cli -- \
  5 --az --weights alphazero_mlx_run/az_iter001.azr \
  --az-sims 12 --az-parallel --az-threads 4
```

The parallel path preserves exact legal move generation and exact move execution.
It changes search scheduling from one shared tree to root-parallel trees whose
root visits are merged, so use more simulations when comparing strength:

```bash
cargo run --release --features v4-opp,v5-feat,czero-feat --bin cascadia-cli -- \
  5 --az --weights alphazero_mlx_run/az_iter001.azr \
  --az-sims 32 --az-parallel --az-threads 4 --az-min-sims-per-thread 8
```

On the first local smoke after adding these speedups, `5` games at `12` sims
dropped from about `10.9s` before optimization to about `7.0s` serial and
`3.2s` root-parallel. A `2` game self-play collection smoke at `12` sims took
`8.6s` with `--az-parallel --az-threads 4`.

## Incremental Local Training

Start with greedy bootstrapping, then add self-play shards. Every stage emits a
checkpoint, so the run is never all-or-nothing.

```bash
mkdir -p alphazero_mlx_run

cargo run --release --features v4-opp,v5-feat,czero-feat --bin cascadia-cli -- \
  32 --az-collect --out alphazero_mlx_run/bootstrap.azd --seed 4242

/Users/johnherrick/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  train_alphazero_mlx.py \
  --samples alphazero_mlx_run/bootstrap.azd \
  --out alphazero_mlx_run/az_iter000.azr \
  --channels 16 --blocks 2 --hidden 64 \
  --epochs 4 --batch-size 128 --lr 0.0003 --save-each-epoch

cargo run --release --features v4-opp,v5-feat,czero-feat --bin cascadia-cli -- \
  8 --az-collect --az-weights alphazero_mlx_run/az_iter000.azr \
  --az-sims 16 --temperature 1.0 --az-parallel --az-threads 4 \
  --out alphazero_mlx_run/iter001.azd --seed 4243

/Users/johnherrick/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  train_alphazero_mlx.py \
  --samples alphazero_mlx_run/bootstrap.azd alphazero_mlx_run/iter001.azd \
  --init alphazero_mlx_run/az_iter000.azr \
  --out alphazero_mlx_run/az_iter001.azr \
  --epochs 4 --batch-size 128 --lr 0.0002 --save-each-epoch

cargo run --release --features v4-opp,v5-feat,czero-feat --bin cascadia-cli -- \
  10 --az --weights alphazero_mlx_run/az_iter001.azr --az-sims 16
```

Scale by increasing bootstrap games, self-play games, and simulations. A serious
overnight Mac-mini pass should use about `200-500` bootstrap games, then repeated
`25-50` game self-play shards at `32-64` PUCT simulations, with checkpoints after
every shard.
