# Morning Status — 2026-04-10 ~10:15

## 🚨 lr mismatch in continuation training (blocking)

`train_continue_iter21.log` (pid 71051) is running with DEFAULT `lr=0.001`.
Per `train_10x_phase3.log`, iter1-20 all used `lr=0.000030` (3e-5) — **33× lower**.

My continuation will likely produce a worse iter21 from the high lr overshoot.

**Plan:** let iter21 finish (sunk cost), bench it, then run
`./restart_training_correct_lr.sh` to kill the continuation and restart
from iter20 with `--lr 0.00003 --iter-offset 20 --iter-prefix nnue_weights_hybrid_continued_v2_iter`.

## iter_history trajectory (NNUE-only, 200g, stderr ±0.28) — COMPLETE

| Iter | Mean | Δ | Bear | Elk | Salmon | Hawk | Fox |
|---|---|---|---|---|---|---|---|
| iter1 | 90.2 | — | 4.7 | 11.7 | 13.6 | 13.4 | 13.7 |
| iter4 | 90.4 | +0.2 | 3.8 | 11.4 | 14.4 | 14.1 | 13.8 |
| iter8 | 90.5 | +0.1 | 4.1 | 11.8 | 14.2 | 13.9 | 13.4 |
| iter12 | 90.8 | +0.3 | 4.2 | 11.6 | 14.6 | 13.9 | 13.2 |
| iter16 | 90.7 | -0.1 | 4.4 | 11.4 | 14.2 | 13.7 | 13.6 |
| iter20 | 90.9 | +0.2 | 4.1 | 11.5 | 15.7 | 13.2 | 13.4 |

**Key pattern: only salmon is improving (13.6 → 15.7, +2.1).** Bear regresses (4.7 → 4.1).
Elk/hawk/fox basically flat. Training has converged on salmon strategy and stopped
improving elsewhere.

**Gain rate: ~+0.03/iter, diminishing.** At 130+ iters to reach NNUE=95, this training
pipeline is unworkable as a path to the 95+ goal.

## 🚨 small_v1 = catastrophic regression (confirms distribution mismatch)

Trained 512→64 from scratch on `mce_policy_samples.bin` with Adam+cosine lr=1e-3, 30
epochs. Final training RMSE 2.22 — much lower than iter20's ~4.78 on self-play data.
**Despite the lower training loss, bench regresses 4 points: 86.8 vs 90.9.**

Wildlife: 56.0 vs iter20's 57.9 (-1.9). All dimensions worse.

**Lesson (second time confirmed):** training on MCE cache alone overfits to MCE play
distribution and doesn't transfer to NNUE inference distribution. The value network
must train on self-play data (same distribution as its deployment).

Implication: the Adam+cosine speedup is promising but must be applied to self-play data
not MCE cache. Script ready: `./train_adam_continuation.sh` (uses `training_merged_iter9.bin`).

## Currently running

| Task | Started | ETA | Purpose |
|---|---|---|---|
| iter20_c20_200g | ~09:55 | ~10:15 | More candidates |
| baseline_iter20_500g | 09:52 | ~10:25 | Tighter mean |
| policy_mce_iter20_100g | ~09:57 | ~10:15 | PolicyMCE with iter20 |
| iter_history (iter12→20) | ~10:12 | ~10:28 | Per-iter trajectory |
| small_v1 training | 09:50 | ~10:30 | 512→64 from scratch on cache |
| train_hybrid iter21+ | 09:51 | iter21 ~10:30 | 🚨 WRONG lr — restart after iter21 |

## Confirmed findings (200g, stderr ±0.28)

| Config | Mean | Δ vs iter4 baseline |
|---|---|---|
| baseline_iter20 (default) | **95.7** | **+0.4** |
| baseline_iter4 | 95.3 | 0 (anchor) |
| iter20 + 1500 rollouts | 95.6 | +0.3 |
| iter20 + LEAF1 | 95.4 | +0.1 |
| iter20 + depth=4 | 95.3 | 0.0 |
| LEAF1 only (iter4) | 95.4 | +0.1 |
| LEAF1 + depth=4 (iter4) | 95.3 | 0 |

**Conclusion: iter20 default config is optimal. Every variation tested makes it worse or no different.**

## Failed experiments
- 1024→128 from scratch on cache (large_v1): 93.5 — much worse
- LEAF1 with iter18: -0.5 (LEAF1 hurts new weights)
- LEAF1 with iter20 (200g): -0.3 (confirmed)
- All LEAF1 variants (depth, candidates, rollouts) on iter4: within noise of baseline
- NRPA L=2 N=15: 77.6 (catastrophic, 5h bench)
- Standalone Gumbel-MCTS: 90.2

## Cache state
- `mce_policy_samples.bin`: ~371 MB (grew from ~250 MB at session start, ~120 MB of new MCE samples)
- New samples are ~50% iter20-quality (collected during morning), ~50% iter4-quality (from earlier benches)
- Used by `train_hybrid.py` automatically when training

## Next milestones
- ~10:10: iter21 from training pipeline
- ~10:15: iter20_c20 result
- ~11:00: 500g iter20 baseline (tighter mean)
- ~13:00: iter25 from training pipeline
- ~14:00: iter30 from training pipeline (final from this run)
