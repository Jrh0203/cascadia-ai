# Overnight Report — 2026-04-14 → 2026-04-15

## TL;DR

Trained NNUE models with frozen-opponent self-play to address greedy-exploitation bias. **Both approaches failed to beat mce93** in validation. Infrastructure is solid; learning signal with current recipe (10K games × 10 epochs × 1e-4 LR) is insufficient.

**Key finding**: Head-to-head tournament validation reveals mce93 (the simplest, oldest NNUE model) is the most robust — winning 5/13 games vs v3/v9/hybrid in our earlier NNUE-model bake-off. It's a hard bar.

## Infrastructure shipped

1. **`CASCADIA_TRAIN_OPP_WEIGHTS`** env var — frozen opponent net during training (prevents co-adaptation collapse).
2. **`CASCADIA_TRAIN_SEED`** env var — per-iter seed variance.
3. **`--init-weights`** flag — separate input from output weights path (checkpoint-style training).
4. **`train_iterative_selfplay.sh`** — bootstrap-then-iterate with promote/reject gates.
5. **`train_iterative_scratch.sh`** — random-init iterate against fixed anchor.
6. **Collapse watcher** — auto-triggers fallback on RMSE > 10 OR 2 consecutive REJECTs.

## Experiment 1: Bootstrap from mce93 (sym_v2)

Train iter N from iter N-1 weights, opponents = mce93 frozen.

| Iter | Trained RMSE | Win rate vs mce93 | Decision |
|------|-------------:|------------------:|----------|
| 1 | 3.74 | 16.7% | REJECT |
| 2 | 3.71 | 8.3% | REJECT |

**Verdict**: Small LR perturbations to mce93's already-tuned weights moved the model AWAY from the optimum. Bootstrapping hurt. Confirmed the "can't easily fine-tune a well-tuned model" problem. Watcher fired fallback after 2 REJECTs.

## Experiment 2: From scratch with mce93 anchor (sym_v3)

Random init iter 1, carry weights forward, opponents = mce93 fixed.

| Iter | Trained RMSE | Win rate vs mce93 |
|------|-------------:|------------------:|
| 1 | 4.33 | 0% |
| 2 | 4.04 | 0% |
| 3 | 4.02 | 0% |
| 4 | 3.89 | 0% |
| 5 | 3.82 | 0% |

**Verdict**: RMSE improving monotonically (4.33 → 3.82) but play strength never crossed mce93. 0/48 games won through 4 iters. Iter 5 may land similarly.

## Why both failed

**Bootstrap (sym_v2)**: near-optimal starting point + small LR = random noise perturbation. Each iter moves away from sweet spot.

**From-scratch (sym_v3)**: starting from random, 10K games × 10 epochs × 1e-4 LR is not enough gradient signal to match a model that was trained on 500K+ games across multiple iterations.

**MCE validation amplifies differences**: small weight errors compound across 20 turns of rollouts. A 0.5 RMSE gap translates to near-0 win rate.

## Data shipped (for future work)

- `nnue_weights_sym_v2_iter{0..2}.bin` — bootstrap experiments
- `nnue_weights_sym_v3_iter{1..5}.bin` — from-scratch checkpoints
- Full training + validation logs in `overnight/`

## What would actually work (for next attempt)

Based on the diagnosis:

1. **Much longer training per iter**: 100K+ games, 30+ epochs. We did 10K × 10. mce93 was trained on ~500K games.
2. **Higher initial LR with decay**: 1e-3 → 1e-5 over training. 1e-4 is both too high to stay near mce93 and too low to find a new optimum from scratch.
3. **Population-based opponents**: rotate anchor through mce93/v3/v9/hybrid for diverse opponent styles. Current approach trains model to beat mce93 specifically.
4. **MCE-based self-play data**: Use MCE rollouts as training targets (distillation), not NNUE-direct outcomes. We have `mce_policy_samples.bin` (29K MCE samples) — could mix these in.
5. **Better training code**: current Rust SGD may have stability issues. PyTorch pipeline (train_pytorch.py) is more battle-tested.

## Recommendation

**Don't try to beat mce93 in head-to-head with quick training runs.** Three paths forward:

### Path A: Accept mce93 as ceiling for NNUE
- Use mce93 for all benchmarks going forward
- Focus remaining work on SEARCH improvements (where our overnight success was — 94.2 → 96.4 via expanded+pf8)

### Path B: Modal-funded large training run
- $10 Modal credit available
- Run 5 iters × 100K games × 30 epochs with population opponents
- Budget estimate: ~$20-30 (over budget, would need approval)

### Path C: Shift to policy distillation
- Generate 100K MCE games on Modal (~$5)
- Train NNUE on MCE outputs (supervised, not self-play)
- Historically documented: policy distillation adds ~0.5 pts (weak, but anything)

## Final tally (through iter 4)

Best operational model: **still mce93** (or hybrid_iter20 for head-to-head).
Search-side winner: **expanded + prefilter-k 8 + halving + 200r** (96.4 base, 101.3 bonus, LOCAL N=30).
