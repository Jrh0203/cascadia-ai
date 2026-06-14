# AlphaZero MLX Recent Status

Last updated: May 18, 2026.

## Goal

Build a local AlphaGo Zero-style Cascadia system that trains from scratch on the
Mac mini, uses greedy play only as the bootstrap prior, then improves through
policy/value self-play and PUCT search. The target is not to imitate the old
NNUE/MCE champion forever; the target is a stronger search-guided policy/value
system whose neural training runs on Apple Silicon through MLX.

## Methodology

- Keep Rust responsible for exact game rules: legal move generation, market and
bag state, scoring, move execution, PUCT collection, and benchmark play.
- Use MLX for neural training on Apple Silicon. Rust writes exact `AZD1` replay
files; MLX trains the policy/value network and writes Rust-loadable `AZR1`
weights.
- Train incrementally: every bootstrap/self-play shard produces durable data and
every training pass produces a checkpoint, so the run is never all-or-nothing.
- Use greedy bootstrap only to make the random policy sane. After that, collect
AlphaZero self-play labels from PUCT visit distributions.
- Optimize local collection without changing rule correctness. Current local
speedups are precomputed convolution neighborhoods, exact top-K candidate
selection, and optional root-parallel PUCT.

## Architecture

- Current model: compact AlphaZero-style residual CNN over a dense `21x21` hex
grid tensor.
- Input: `65` planes covering board occupancy, terrain, wildlife, allowed
wildlife, market state, bag state, own progress, opponent progress, nature-token
context, and turn/player context.
- Policy head: candidate-factorized scoring over legal moves using tile cell,
wildlife cell, market slot, wildlife market slot, and skip-wildlife logits.
- Value head: scalar with-bonus final score target normalized by `120`.
- Design stance: the AlphaGo Zero loop is the right outer algorithm. The likely
future best model is a hybrid board ResNet plus set-attention over market,
candidate, bag, and opponent/race entities.

## Recent Changes

- Added `crates/cascadia-ai/src/alphazero.rs`:
  - dense AlphaZero encoder,
  - residual CNN policy/value network,
  - PUCT search,
  - greedy bootstrap data collection,
  - self-play data collection,
  - `AZD1` replay and `AZR1` weight formats,
  - unit tests for encoder shape, policy distribution, PUCT legality,
    save/load, replay roundtrip, and root-parallel legality.
- Added CLI support in `crates/cascadia-cli/src/main.rs`:
  - `--az` / `--alphazero` for play,
  - `--az-collect` / `--collect-az` for replay collection,
  - `--az-train` CPU fallback,
  - `--az-parallel`, `--az-threads`, and `--az-min-sims-per-thread`.
- Added `train_alphazero_mlx.py`:
  - reads one or more `AZD1` files,
  - trains the policy/value net on MLX GPU,
  - validates policy top-1,
  - saves epoch checkpoints and final `AZR1` weights.
- Added `run_alphazero_mlx_local.sh`:
  - resumable local loop,
  - bootstrap collection,
  - MLX bootstrap training,
  - repeated self-play collection,
  - continuation training,
  - small benchmark after each shard.
- Updated `.gitignore` so generated `.azd`, `.azr`, `.czr`, and `.policy`
  artifacts stay local.
- Killed the old `czero_stage2_wide` tmux training run after explicit approval
  and moved active local work to AlphaZero/MLX.

## Current Local Run

Run directory: `alphazero_mlx_run_fast`.

Completed:

- Greedy bootstrap: `64` games, `5120` samples, `562MB`.
- Bootstrap MLX training: `4` epochs, final validation top-1 about `0.438`.
- AlphaZero self-play shards: `8` shards.
- Each shard: `12` games, `24` PUCT sims, root-parallel with `4` threads and
  `6` minimum sims per worker.
- Total replay after shard 8: `12800` samples.
- Latest checkpoint: `alphazero_mlx_run_fast/az_iter008.azr`.
- Latest MLX validation: about `0.75` top-1 on the held-out shard mix.

Latest tiny benchmark from `az_iter008`:

- Command shape: `5` games, `--az-sims 24`, root-parallel.
- Base mean: `75.4`.
- With-bonus mean: `77.8`.
- Runtime: `3.8s` for `5` games.

This is not yet a strength result. The checkpoint is still very early and is
learning a policy prior, not competing with the champion. The useful signal is
that the local loop is now stable, checkpointed, and fast enough to iterate.

## Verification

Passed:

```bash
cargo test --features v4-opp,v5-feat,czero-feat alphazero -- --nocapture
cargo build --release --features v4-opp,v5-feat,czero-feat --bin cascadia-cli
python3 -m py_compile train_alphazero_mlx.py
```

MLX verified on Apple GPU with `Device(gpu, 0)` using the bundled Codex Python.

## Next Methodology Step

Do not judge promotion strength from the tiny 5-game shard benchmarks. Use them
only as smoke tests. The next meaningful loop is:

1. Increase self-play data volume while keeping checkpoints incremental.
2. Keep `4` threads for interactive use; try `6` threads for overnight runs.
3. Raise PUCT sims from `24` to `48-64` once data collection volume is stable.
4. Track benchmark trend every shard, but only run larger comparisons after the
   policy prior is clearly above greedy quality.
5. After the ResNet loop is stable, add set-attention for market/candidate/bag
   entities rather than replacing the board trunk outright.
