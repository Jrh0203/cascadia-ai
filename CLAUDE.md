# Cascadia AI

## Goal
Build a superhuman Cascadia board game AI that consistently scores 95+ in 4-player games (Card A scoring, no habitat bonuses).

## Current Best: MCE(50) + mulligan-aware pre-move = **92.9 mean, P90=101**

**Saved as `nnue_weights_mce93.bin` (10MB, larger 512→64 NNUE)**

Pre-move optimization uses MCE (full strategy) to evaluate whether to:
1. Take the free 3-of-a-kind replacement (when available)
2. Spend nature tokens on paid mulligans (multi-mulligan supported, deferred returns dig deep into bag)
Decisions sample K=3 possible post-mulligan markets to estimate expected value.

**Methodology for best result (saved as `nnue_weights_mce88.bin`):**

### Step 1: Joint tile+wildlife evaluation (`eval.rs`)
- Changed `best_move_with_potential` from single-best-habitat tile selection to **top-8 tile placements by habitat**, each evaluated jointly with all wildlife placement options.
- This finds moves where a slightly worse habitat placement enables much better wildlife patterns.
- Greedy baseline improved from 76.6 → 78.7 from this change alone.

### Step 2: NNUE network (`nnue.rs`, 5197 features → 512 → 64 → 1)
Sparse binary features:
- **Per-cell** (441 cells × 11): wildlife type (5) + tile-no-wildlife (1) + terrain (5)
- **Game-phase** (110): turn number (21), nature tokens (9), wildlife count per type (30), largest habitat per terrain (50)
- **Pairwise adjacency** (147): wildlife pairs in 3 hex line directions (3 × 7 × 7)
- **Wildlife patterns** (89): bear pair count (5), elk line lengths top-4 (20), salmon run lengths top-3 (24), isolated hawk count (9), fox diversity (6), empty slots per type (25)

### Step 3: NNUE training (`nnue_train.rs`)
- 5 self-play iterations, 100K games each, 15 epochs per iteration
- Iteration 1: greedy data (fast, ~240s). Iterations 2-5: NNUE-guided self-play + epsilon=0.1 exploration
- **Delta labels**: each afterstate labeled with `final_score - current_score` (remaining points to gain). Turn 1 gets ~70, turn 19 gets ~3. This is critical — the old approach (same final score for all positions) couldn't distinguish good from bad early-game boards.
- Mini-batch SGD, lr=0.0001, batch_size=256
- Move selection uses `actual_score + NNUE(remaining_value)` = estimated final score
- **High-score cache**: games scoring 90+ are appended to `training_cache_90plus.bin` for future training
- Command: `cargo run --release --bin cascadia-cli -- 100000 --nnue-train --lr 0.0001 --epochs 15 --iterations 5 --epsilon 0.1`

### Step 4: Monte Carlo Evaluation (`mce.rs`)
- Generate top-15 candidates from: greedy + candidate_moves + wildlife_strategic_candidates
- **Wildlife demand scoring**: compute demand for each wildlife type based on board state (isolated bears, extendable elk lines 2-3, extendable salmon runs, safe hawk slots). Boost candidates that supply high-demand wildlife.
- For each candidate, run 50 rollouts: execute the move, then play 6 AI turns using NNUE + greedy opponents
- Each rollout shuffles the bag for different random futures
- If game ends, use actual score. Otherwise: actual_score + NNUE remaining value estimate
- Average all rollout scores, pick candidate with highest average
- **Parallelized**: all rollouts distributed across CPU cores (10 cores = ~6.5x speedup)
- Command: `cargo run --release --bin cascadia-cli -- 30 --mce --rollouts 50`

### Turn-dependent potential (`eval.rs`)
- `potential_scale = min(1.0, ai_turns_remaining / 10.0)`
- Full potential weight early game, zero on last turns
- Prevents the AI from chasing setup value when there's no time to realize it

## Benchmark Results (4-player, Card A, no habitat bonuses)

| Strategy | Mean | P90 | Wildlife | Bear | Elk | Salmon | Hawk | Fox | Habitat | Tokens |
|----------|------|-----|----------|------|-----|--------|------|-----|---------|--------|
| Greedy (old, top-1 tile) | 76.6 | 83 | 46.4 | 3.8 | 10.2 | 9.5 | 10.7 | 12.1 | 28.1 | 2.1 |
| Greedy (top-8 joint) | 78.7 | 84 | 47.7 | 2.9 | 11.8 | 11.3 | 11.4 | 12.0 | 27.6 | 2.0 |
| NNUE (pure override) | 83.0 | 89 | 49.6 | 8.5 | 9.2 | 10.1 | 8.9 | 12.8 | 28.7 | 4.8 |
| MCE(10 rollouts) | 84.8 | 92 | 52.1 | 6.5 | 10.0 | 12.8 | 9.2 | 13.6 | 28.8 | 3.9 |
| MCE(20 rollouts) | 86.7 | 92 | 52.4 | 10.0 | 9.5 | 11.9 | 8.8 | 12.2 | 29.5 | 4.8 |
| MCE(50, old labels) | 88.6 | 96 | 53.6 | 11.6 | 9.8 | 10.3 | 8.4 | 13.4 | 30.1 | 4.9 |
| MCE(50, +wildlife cands) | 89.4 | 95 | 53.9 | 11.1 | 7.7 | 11.0 | 10.0 | 14.0 | 29.8 | 5.7 |
| MCE(50, delta labels) | 90.8 | 97 | 54.8 | 11.9 | 8.8 | 11.3 | 9.1 | 13.6 | 29.2 | 6.8 |
| MCE(50, 512-NNUE+demand) | 91.1 | 97 | 54.5 | 13.0 | 8.8 | 8.7 | 10.0 | 13.9 | 29.6 | 7.0 |
| MCE(50, +greedy premove) | 92.2 | 97 | 55.4 | 7.8 | 10.4 | 12.3 | 10.1 | 14.8 | 29.8 | 7.0 |
| **MCE(50, +MCE premove)** | **92.9** | **101** | **55.9** | **12.1** | **8.1** | **12.2** | **9.7** | **13.7** | **29.7** | **7.3** |

## Key Findings

1. **Joint tile+wildlife eval is critical** — evaluating top-8 tiles (not just habitat-best) added +2.6 to greedy alone.
2. **NNUE as pure override works** — using `nval` directly instead of `score * 1000 + nval` lets the network drive decisions. Requires well-trained weights.
3. **MCE is the breakthrough** — simulating 6 AI turns ahead with NNUE-guided play captures multi-turn pattern building (bear pairs, salmon runs). Added +5.6 over plain NNUE.
4. **Beam search doesn't work in stochastic games** — deeper depth = worse scores. Noisy pruning from random futures.
5. **Parallel rollouts scale linearly** — 10 cores = ~6.5x speedup for MCE.
6. **MCE plateaus around 50 rollouts** — 100 rollouts didn't improve over 50.
8. **Delta labels are critical** — labeling each position with `final_score - current_score` (remaining points) instead of the same final score for every position. This lets the NNUE distinguish "good early board with lots of potential" from "bad early board." Added +1.4 points over same-label approach.
9. **Wildlife-strategic candidates** — explicitly generating pattern-extending moves (bear pair setups, hawk isolation, etc.) adds +0.8 on top of greedy candidates alone.
10. **High-score game cache** — games scoring 90+ are cached to `training_cache_90plus.bin` for future training on expert-quality data.
7. **Training/eval opponent consistency matters** — training with greedy opponents but benchmarking with greedy opponents works. Random opponents for training hurts.
8. **1p pre-training doesn't transfer well to 4p** — learned values don't account for market competition.

## Remaining Gap to 95 (need ~5 points)

Reliable range: 89-91 mean, P90=93-97 across multiple runs with best weights.

Wildlife at ~54 needs to reach ~60:
- **Elk: 9 → 13** (need line of 4, biggest gap)
- **Salmon: 10-11 → 15** (need run of 5)
- **Hawk: 9 → 14** (need 5 isolated)
- Bear at ~11 and Fox at ~13 are good
- Tokens at ~6.5 — may be hoarding too many

### What we tried that DIDN'T close the gap
- **ROI weights** (salmon×1.3, bear×0.7): helped salmon +1.9 but hurt hawk -1.3, net negative
- **Tile marginal labels**: taught great salmon (15.2!) but over-rotated away from other patterns, net -3
- **Blended labels** (70% delta + 30% marginal): didn't beat pure delta
- **Deeper MCE rollouts** (12 turns): NNUE errors accumulate, net -1
- **Greedy rollouts**: better elk (10.4) but worse bear/fox, net -2
- **Hybrid rollouts** (2 NNUE + 4 greedy): between the two, no improvement
- **1p pre-training**: learned patterns don't transfer to 4p market competition
- **More MCE rollouts** (100 vs 50): plateaued, no improvement

### Most promising remaining approaches
1. **Market drafting strategy** — the AI needs to draft elk/salmon/hawk tiles more often, even if the immediate score is lower. Currently it over-drafts bear/fox tiles.
2. **Larger NNUE** (512→64→1) — more capacity to learn complex pattern interactions
3. **Wildlife allocation planning** — commit to a strategy per game (e.g., "2 bear pairs, 1 elk line of 4, 1 salmon run of 5, 5 hawks, 2 foxes") and draft accordingly
4. **Nature token spending** — proactively spend tokens on independent drafts to get exactly the wildlife needed for pattern completion

### Most promising next steps
1. **Wildlife-strategic candidates** — explicitly generate candidates that extend elk lines, salmon runs, hawk isolation. Add these to the top-10 pool for MCE to evaluate.
2. **Deeper rollouts** — increase from 6 to 12+ AI turns so MCE can plan complete elk lines (4 turns) and salmon runs (5 turns).
3. **Retrain NNUE on MCE-quality games** — current NNUE trained on 79-level play. MCE plays at 88-level. Better training data → better rollout play.
4. **More candidates** — expand from top-10 to top-20 or add pattern-targeted candidates.

## File Map

| File | Purpose |
|------|---------|
| `crates/cascadia-ai/src/eval.rs` | Greedy move evaluation (top-8 joint tile+wildlife, turn-dependent potential) |
| `crates/cascadia-ai/src/nnue.rs` | NNUE network: 5108→256→32→1, feature extraction, forward/backprop, save/load |
| `crates/cascadia-ai/src/nnue_train.rs` | NNUE training: parallel data gen, self-play iterations, MCE(5) for self-play moves |
| `crates/cascadia-ai/src/mce.rs` | Monte Carlo Evaluation: parallel rollouts with NNUE-guided play |
| `crates/cascadia-ai/src/search.rs` | Candidate generation (candidate_moves), wildlife setup bonuses, execute_scored_move |
| `crates/cascadia-ai/src/potential.rs` | Hand-crafted board potential (pattern-aware marginal values) |
| `crates/cascadia-ai/src/ntuple.rs` | N-tuple network (legacy, superseded by NNUE) |
| `crates/cascadia-ai/src/train.rs` | N-tuple TD training (legacy, superseded by nnue_train) |
| `crates/cascadia-cli/src/main.rs` | CLI: --nnue-train, --nnue, --mce, --collect-mce, --train-mce-policy, --train, --ntuple, --all |
| `nnue_weights_mce88.bin` | Saved NNUE weights for MCE(50)=88.6 benchmark (old labels) |
| `nnue_weights_mce91.bin` | Saved NNUE weights for MCE(50)=90.8 (delta labels, 256→32) |
| `nnue_weights_mce91_5.bin` | Saved NNUE weights for MCE(50)=91.1 (larger 512→64 + demand scoring) |
| `nnue_weights_mce93.bin` | **Current best** — MCE(50)+mulligan premove = 92.9 mean, P90=101 |
| `training_cache_90plus.bin` | Cached training data from games scoring 90+ |
| `mce_policy_samples.bin` | **Growing dataset** — MCE-labeled afterstate samples for policy distillation. Append-only, never delete without archiving. |

## Commands

```bash
# Train NNUE (5 self-play iterations, ~35 min)
cargo run --release --bin cascadia-cli -- 100000 --nnue-train --lr 0.0001 --epochs 10 --iterations 5 --epsilon 0.1

# Benchmark with MCE (50 rollouts, ~7 min for 20 games)
# NOTE: MCE benchmarks auto-append samples to mce_policy_samples.bin
cargo run --release --bin cascadia-cli -- 20 --mce --rollouts 50

# Benchmark with NNUE only (fast, ~30s for 500 games)
cargo run --release --bin cascadia-cli -- 500 --nnue

# Benchmark greedy baseline
cargo run --release --bin cascadia-cli -- 2000

# Load specific weights
cargo run --release --bin cascadia-cli -- 50 --mce --rollouts 50 --weights nnue_weights_mce88.bin

# === Policy Distillation Pipeline ===

# Step 1: Collect MCE-labeled training data
# Local (~68s/game with NNUE opponents, MCE(300)):
cargo run --release --bin cascadia-cli -- 100 --collect-mce --weights nnue_weights_mce93.bin --rollouts 300 --random-seed --out mce_policy_samples.bin

# Cloud (Modal — 10 workers, ~30s/game, 100 games in ~5 min for ~$0.50):
modal run modal_collect.py collect --num-workers 10 --games-per-worker 10

# Step 2: Train policy-distilled NNUE (75× augmentation: rotation + translation)
# Checkpoints saved after every epoch. ~29K raw samples → ~2.1M augmented.
cargo run --release --bin cascadia-cli -- 0 --train-mce-policy --samples mce_policy_samples.bin --epochs 30 --lr 0.00003 --init-weights nnue_weights_mce93.bin --weights nnue_weights_v2.bin

# Step 3: Benchmark the policy net (same architecture, no rollouts needed)
cargo run --release --bin cascadia-cli -- 500 --nnue --weights nnue_weights_v2.bin

# === Distributed Benchmarking (Modal) ===
# NNUE benchmark (fast — 500 games across 10 workers):
modal run modal_collect.py benchmark --num-workers 10 --games-per-worker 50

# MCE benchmark (stronger — 100 games across 10 workers):
modal run modal_collect.py benchmark --num-workers 10 --games-per-worker 10 --strategy mce --rollouts 750

# Custom weights:
modal run modal_collect.py benchmark --num-workers 10 --games-per-worker 50 --weights nnue_weights_v2.bin
```

## Modal Setup

```bash
pip3 install modal
python3 -m modal setup   # authenticate via browser
# Then use: modal run modal_collect.py collect/benchmark ...
```

Modal compiles the Rust binary in a remote container (cached after first build).
Each worker uses random seeds (--random-seed) so no coordination needed.
Results download automatically when all workers finish.
Cost: ~$0.50 per 100 games (8 vCPUs per worker).
```
