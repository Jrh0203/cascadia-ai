# Cascadia AI

## Goal
Build a superhuman Cascadia board game AI that consistently scores 95+ in 4-player games (Card A scoring, no habitat bonuses).

## Current Champion (Apr 17, 2026): v4-opp NNUE + MCE wide_v1 = **95.94 mean (100.85 bonus) in 4-player HH**

**Weights: `nnue_weights_v4opp_modal_iter3.bin` (~23MB)**
**Binary: `target-mid-v4/release/cascadia-cli` (compiled with `--features mid-features,v4-opp`)**

**Head-to-head vs prior baseline (100 games, 200 seat-games, Modal):**
- v4opp_modal_iter3 + mce_wide_v1: **95.94 ± 0.21**, 33.5% win rate
- mid_fsp_iter10 baseline:  94.61 ± 0.21, 16.5% win rate (which itself tied mce93)
- **Delta: +1.33 pts, p < 0.001, win rate doubles**

### The step-function change: per-opponent detail features

Prior NNUE had ~55 opponent features (max habitat per terrain only). v4-opp appends
**369 new features per position** — for each of 3 opponents:
- 5 wildlife type counts × 11 bins (55)
- 5 habitat sizes × 11 bins (55)
- Nature tokens × 9 bins (9)
- Pattern signals: bear-singleton, elk-line-3+, salmon-run-4+, isolated-hawk-5+ (4)
= 123 features per opponent × 3 opponents = 369.

Backward compatible: old weights load fine via zero-padding of the new columns. Feature gate: `v4-opp` cargo feature; set `OPP_DETAILED_BASE` conditional on `mid-features` (10862) vs full v3 (45260).

### Why it works

The opponent features let the value head condition its estimate on what opponents are threatening. Wildlife breakdown shows the mechanism: the new network takes **+2.3 more bear points per game** than baseline (opponents no longer "hide" in its blind spot). Slight loss on salmon (-1.6) and elk (-0.4); net wildlife +0.7; rest from habitat/tokens.

Consistent with the literature precedent (Pluribus, AlphaStar, Cicero) — multi-agent superhuman play needs opponent modeling, and until now the NNUE had essentially none.

### Training recipe (what shipped)

Started from `nnue_weights_mid_fsp_iter10.bin` (v3 champion with zero-padded v4 cols).
FSP training: 3 iterations × 30K games × 10 epochs × LR 3e-5 → 1e-5 × ε=0.1.

**Data generation parallelized on Modal:**
- `--selfplay-pool` CLI mode: generates MCV3-format samples with opponent pool from `CASCADIA_TRAIN_OPP_POOL` env.
- 100 Modal workers × 300 games each, ~7-10 min wall per iter.
- Shards concatenated (magic-stripped) into single cache locally.
- `--cache-train` locally: loads MCV3, 10 epochs SGD, ~2 min.
- FSP pool: `random,scarcity,preference,mce93,mid_fsp_iter10,v4opp_fsp_iter3` + all prior-iter checkpoints.
- RMSE across iters: 4.84 → 4.79 → 4.81 (converged).

**Critical discovery**: fine-tuning LR=1e-4 DIVERGES when starting from trained weights. LR=3e-5 is stable. 1e-4 was fine for from-scratch training (original mce93 recipe) but not for fine-tuning.

### Champion command (CORRECTED 2026-05-21)

```bash
MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 MCE_MUTATE_EXPAND=24 \
./target-mid-v4/release/cascadia-cli N --nnue-rollout-mce \
  --candidates expanded --prefilter-k 8 --alloc halving \
  --rollouts 600 --weights nnue_weights_v4opp_modal_iter3.bin
```

**Earlier `CASCADIA_MCE_TRUNC=3` recommendation REVERTED**. At MCE(600), TRUNC=3
regresses mean score by **−2.0 pt** (20 games × 4 seats base score: 94.91 → 92.90).
The original validation was at MCE(100) where high rollout variance hid the
truncation bias. At champion-scale rollouts the NNUE tail-estimate bias dominates.

### MCE env-var status

| Env var | Effect at MCE(100) | Effect at MCE(600) | Recommended use |
|---|---|---|---|
| `CASCADIA_MCE_TRUNC=3` | -0.03 (null, 5.7× speedup) | **−2.0 pt regression** | DATA COLLECTION ONLY (where 5.8× more games beats per-label quality) |
| `CASCADIA_MCE_DECOUPLE_OPP=1` | -0.16 base score (1.29× extra speedup) | Untested at MCE(600), likely also regresses | Same as TRUNC — collection only |

**Key learning**: at MCE(100) high rollout variance (~2 pt) hides NNUE tail
approximation error. At MCE(600+) the variance averages out and the bias
becomes the bottleneck. **Validate any speedup at champion-scale rollouts, not
just MCE(100).**

### When TRUNC IS useful

For training data generation (e.g., `collect_hybrid_pairwise`), TRUNC=3 +
DECOUPLE_OPP gives 7.4× more games per cloud-dollar. The slight per-label
quality cost (~0.15pt at base score) is acceptable because distillation
training averages over thousands of labels.

For actual gameplay benches and H2H, use the corrected champion command above
with TRUNC=0.

Or via tag:
```bash
CASCADIA_SEAT_STRATEGIES="mce_wide_v1:mce_wide_v1:mce_wide_v1:mce_wide_v1" \
./target-mid-v4/release/cascadia-cli N --nnue --weights nnue_weights_v4opp_modal_iter3.bin
```

The `mce_wide_v1` tag (in `pick_move_by_tag`) uses K=32 prefilter, R=600 halving, MCE_LMR=1. It's also the strategy that powers 50/50 mixed HHs.

### Runtime optimization (landed earlier this session)

`pick_best_move_nnue` in `crates/cascadia-ai/src/nnue_train.rs` rewritten to replace
15 board clones per decision with place/undo on a single outer board. Saves ~225KB
of memory copies per decision × ~2 decisions per rollout. Verified: baseline score
matches pre-opt within noise (94.0 vs 94.1).

### Other previous bests (for reference)

- `nnue_weights_mce93.bin` (Apr before) — 92.9 mean + P90=101 with MCE(50) + mulligans
- `nnue_weights_mid_fsp_iter10.bin` (Apr 16) — ~95 mean HH, tied mce93
- `nnue_weights_v4opp_fsp_iter3.bin` (Apr 17 local training, 3 iter × 20K games) — +1.30 HH, essentially same as v4opp_modal_iter3

### Experiments log after v4opp shipped (Apr 17 afternoon) — ALL NULL OR NEGATIVE

Full log: `memory/failed_experiments_apr17.md`

| Experiment | Best result | Verdict |
|---|---|---|
| Gumbel halving (σ=3, 0.5) | -22 to -6 pts | Math doesn't apply at score scale |
| Per-ply expectimax rollouts | -4 pts, 4× slower | Too expensive, trajectory bias |
| MctsPW allocator | -1.1 to -2.4 pts | UCB asymptotic advantage not active at R=600 |
| On-policy MCE(50) training, 15K games, 1 iter | +0.1 (null) | MCE(50) too weak; proper test needs ~$100+ |
| Temporal ensemble (modal iter1+2+3) | -0.9 | Correlated errors |
| **Diverse ensemble (modal_iter3+mid_fsp_iter10+v4opp_fsp_iter3)** | 10g: +0.6 / **100g HH: +0.02 (null)** | Draft shifts but score-neutral |

**Consistent pattern**: search-time changes don't compound at this budget/regime. Value-function **feature signal** is the only axis that has moved the needle this year (v4-opp gave +1.33). Probable step-function levers for a future session: **OPP×MARKET cross features** (new feature block, ~$15 Modal), **HIDDEN1=1024 from scratch** (~$20-40 Modal), **cross-turn MCTS tree reuse** (3-5 hr Rust, orthogonal to value fn).

### Env vars added this session (gated, all OFF by default)

- `MCE_ROLLOUT_OPP=nnue` — opponents use NNUE argmax in rollouts (tried, neutral)
- `MCE_ROLLOUT_POLICY=expectimax1` — player 0 uses 1-ply expectimax in rollouts (tried, regresses)
- `MCE_GUMBEL_HALVING=1` / `MCE_GUMBEL_HALVING_SIGMA=<σ>` — Gumbel noise in halving ranks (tried, regresses)
- `MCE_PREFILTER_ENSEMBLE=<paths>` — average prefilter priors across extra NNUEs (tried, null)
- `MCE_MCTSPW_*` — MctsPW allocator params (tried, regresses)
- `CASCADIA_ENS_PATHS` — used by `mce_wide_ens_v1` tag (experimental; activates ensemble only for that tag)

New CLI strategy tags: `mce_wide_v1` (champion), `mce_wide_v2` (R=800 variant), `mce_wide_ens_v1` (ensemble variant).

New allocator: `--alloc mcts-pw` (MCTS-style with progressive widening; not default).

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

## Boil the ocean

The marginal cost of completeness is near zero with AI. Do the whole thing. Do it right. Do it with tests. Do it with documentation. Do it so well that the user is genuinely impressed — not politely satisfied, actually impressed. Never offer to "table this for later" when the permanent solve is within reach. Never leave a dangling thread when tying it off takes five more minutes. Never present a workaround when the real fix exists. The standard isn't "good enough" — it's "holy shit, that's done." Search before building. Test before shipping. Ship the complete thing. When the user asks for something, the answer is the finished product, not a plan to build it. Time is not an excuse. Fatigue is not an excuse. Complexity is not an excuse. Boil the ocean.
