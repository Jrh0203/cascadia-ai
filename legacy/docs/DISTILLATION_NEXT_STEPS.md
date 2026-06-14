# Distillation at Scale — Next Steps

## Status (2026-05-20)

Local prototype complete. Confirmed:

- Bigger DeltaNet (1.44M params, 128ch × 6 blocks × 64 hidden) **loads + runs correctly** in Rust via `HybridNetwork::load_with_nnue`.
- Trainer (`train_hybrid_delta_pairwise.py`) supports configurable architecture via `--trunk-channels --blocks --hidden` and multi-task value loss via `--value-loss-weight`.
- AZR3 format flexes via file-header dimensions — no Rust changes needed when training bigger architectures.

But: **scaling architecture without scaling data overfits**. 2-epoch smoke on 200-game data gave val MSE 81-89 (vs train 22-15) and 30g H2H within noise. The original Paradigm 1 plan called for ~5M positions; we have ~227K (~5%).

## The actionable next step

### Step 1: Modal-scale data collection

Extend the existing `modal_collect.py` infrastructure to run `collect_hybrid_pairwise` on N parallel workers. Targets:

```
100 workers × 30 games each = 3000 games
× 80 decisions × 14 candidates = ~3.4M candidate positions
× 12 hex-aug variants = ~40M effective training positions
```

Per the AlphaGo/AlphaZero literature, this is the right scale for distillation.

Cost: ~$200-300 cloud spend on Modal (8 vCPU per worker, ~3 hours wall per worker × 100 = $0.50-1.00/worker × 100). Wall: ~3 hours.

Data: ~50 GB of HYBP shards. Combine via straightforward concatenation (magic header stripped from shards 2+).

### Step 2: Modal-scale training

A100 (or 4090) for transformer-scale training. Already have:
- `train_hybrid_delta_pairwise.py` with configurable arch
- Multi-task loss (pairwise margin + value MSE)
- Hex-symmetry augmentation
- AdamW + dropout

New step: dial up batch size (1024+) and epoch count (30-50). With 3M training pairs and proper batching, each epoch is ~5 min on A100. Full training: 2.5-4 hours.

Cost: ~$50 cloud spend.

### Step 3: Bench

Use existing `mce_perf_bench` with `BENCH_SCORE_MODE=base` for clean comparisons. 100 games × 4 seats = 400 seat-games per setting:

| Setting | Expected mean |
|---|---:|
| Current champion (NNUE+MCE wide_v1) | 92.06 (base) |
| + α=0.3 distilled Δ (paragdigm 1) | **~93.0-94.0 expected if distillation works** |
| + α=1.0 (pure distilled, no NNUE) | TBD, could be similar |

## Architecture sizing recommendation

Based on the smoke result (1.44M overfits on 200 games), the actual right size for 3000 games is probably **~3-5M params** to balance capacity vs. data:

```bash
python3 train_hybrid_delta_pairwise.py \
  --hybp <combined_3000_games.hybp> \
  --out distilled_v1.azr3 \
  --trunk-channels 192 --blocks 6 --hidden 128 \
  --epochs 30 --pairs-per-batch 1024 --lr 3e-4 \
  --weight-decay 1e-3 --dropout 0.20 --hex-aug \
  --loss margin --margin 1.5 --value-loss-weight 0.5 \
  --save-each-epoch
```

Use `--save-each-epoch` and bench the best-val checkpoint (typically epoch 10-15 with proper regularization at this scale).

## Why this should work

Per the literature (AlphaGo Zero, Stockfish NNUE, Player of Games):

- **Distillation from oracle** (MCE) → **direct NN evaluation** has worked in every game tested.
- The student typically achieves 90-95% of teacher strength at <1% inference cost.
- For Cascadia: teacher is MCE(600) at ~800ms/decision; student would be NN forward at ~3ms/decision → 250× faster.
- The student can then be COMBINED with MCE for even stronger play (NN-as-prefilter, NN-as-rollout-policy).

The current bench shows TRUNC=3 already gives 5.74× MCE speedup. Distillation gives another likely 3-10× (depends on whether we use NN pure vs NN-as-prefilter). Stacked: ~30-50× compute headroom = MCE(6000-18000) at original champion wall.

## Risk assessment

| Risk | Probability | Mitigation |
|---|---:|---|
| Distillation underperforms teacher significantly | 30% | Use NN-as-prefilter mode (NN + MCE) instead of pure NN |
| Modal compute exceeds budget | 20% | Pre-flight test with 10 workers first |
| Trained model gives noise-bounded H2H result | 40% | Pivot to Paradigm 2 (MuZero) — but at that point we've ruled out the easier path |
| Bench wins but champion regression in production | 10% | Verify with 500+ game H2H before promoting |

Expected outcome: **+1-2pt mean score over current champion** if everything works. This is the +3 path's most concrete remaining lever.

## Decision required from user

This is a paid-cloud-compute step (~$300 total). Recommendations:

1. **If you want to proceed**: kick off modal_collect.py extension; I can write the integration code in next session and run an end-to-end pilot ($30 budget, 100 games).
2. **If pilot positive**: scale to 3000 games + training run + bench.
3. **If you prefer to defer**: this doc captures the plan for resumption later.

The pipeline is built and validated. Data is the only remaining blocker.
