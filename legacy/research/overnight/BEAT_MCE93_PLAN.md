# Plan: Beat mce93 with Extended Self-Play Training

## Current state → target

| Metric | Current best (sym_v3 iter 5) | Target (beat mce93 head-to-head) |
|--------|-----------------------------:|---------------------------------:|
| Mean base score | 90.33 | ≥ 95.3 (mce93 average) |
| Win rate vs mce93 | 0% | ≥ 55% |
| RMSE | 3.82 | ≤ 3.0 (estimated) |
| Gap to close | 4.7 base points | — |
| Trajectory observed | +0.7 pts/iter (slowing) | Need to sustain ≥0.7 for 7+ iters |

## Diagnosis of why sym_v3 didn't get there

1. **Too little compute per iter**: 10K games × 10 epochs gets diminishing returns by iter 3-4
2. **LR too low once past random**: 1e-4 moves weights too slowly after iter 2
3. **Narrow opponent distribution**: always mce93 → model may specifically over-adapt
4. **Validation is expensive** (~40 min per iter) → wastes cycles confirming known-lost games

## Plan: Three-stage training progression

### Stage 1 — Fast climb (games, iters matter more than anything)
**Goal**: Get model from 90 base to ~93 base. Break through the "can't match mce93 on quantity" barrier.

**Config:**
- Start from `nnue_weights_sym_v3_iter5.bin` (already at 90.33)
- **Games per iter**: 30K (3× current)
- **Epochs**: 15 (same as mce93's original recipe)
- **LR schedule**: 5e-4 → 1e-4 decay over training (higher early, anneal)
- **Opponent**: mce93 frozen (same anchor)
- **Iters**: 5
- **Validation**: skip intermediate; just run at end

**Cost estimate**:
- Per iter training: 30K games × 4 players × ~100ms = ~20 min locally
- 5 iters training: ~1.7 hours
- Final validation (1×): ~45 min
- **Total: ~2.5 hours local, $0 Modal**

**Success gate**: at end, validate N=50 (not 12) head-to-head vs mce93. Target: ≥40% win rate (needs to be CLOSE to mce93, not beat it yet).

### Stage 2 — Sharper competition (diverse opponents)
**Goal**: Push from 93 → 95 base. Avoid over-specializing to mce93.

**Config:**
- Start from Stage 1 end weights
- **Games per iter**: 30K
- **Epochs**: 15
- **LR**: 1e-4 (fine-tune)
- **Opponents**: POPULATION — randomly sample per game from `[mce93, v3, hybrid]` (diverse anchors, all strong)
- **Iters**: 3-5
- **Validation**: every 2 iters, N=20 vs mce93

**Cost**: Similar ~2-3 hours.

**Success gate**: ≥50% vs mce93 in head-to-head.

**Needs**: implement population opponents in training code. Small Rust change (~30 min). Sample opponent per game from a list of 3 weight files.

### Stage 3 — Final polish (if Stage 2 crossed 50%)
**Goal**: Cross 55% threshold, make it statistically significant.

**Config:**
- From Stage 2 weights
- **Games per iter**: 50K (more precision)
- **Epochs**: 20
- **LR**: 5e-5 (tight polish)
- **Opponents**: population + include Stage 2 model itself
- **Iters**: 2-3
- **Validation**: N=100 on Modal (~$5)

**Success gate**: ≥60% win rate vs mce93 at N=100 (statistically tight).

### Stage 4 — MCE distillation (independent parallel experiment)
**Goal**: Hedge bet with a different approach.

**Config:**
- Collect 50K MCE(200) games on Modal (~$5)
- Train NNUE on MCE outputs (supervised, not self-play) for 20 epochs
- Compare head-to-head vs mce93

**Rationale**: documented to add ~0.5 pts. Complements self-play. Different failure modes.

## Ranked by expected ROI

| Stage | Expected gain | Time | Cost | Risk |
|-------|--------------|------|------|------|
| **Stage 1** (continue sym_v3 with more compute) | +2-3 pts | 2.5 hr | $0 | Low — trajectory established |
| **Stage 2** (population opponents) | +1-2 pts | 2-3 hr | $0 | Medium — new code required |
| **Stage 3** (polish) | +0.5-1 pt | 2 hr + Modal | ~$5 | Low once Stages 1+2 succeed |
| **Stage 4** (MCE distillation) | +0.5-1 pt | ~1 hr + Modal | ~$5 | Medium — different approach |

## Recommended execution order

1. **Stage 1 first** (no code changes, just parameters). 2.5 hour run.
2. **Assess**: did it reach ~93 base? If yes, continue. If not, diagnose.
3. **Stage 2 in parallel with Stage 4** (different codepaths). ~3 hours.
4. **Stage 3 only if Stage 2 crossed 50%** — don't waste Modal on losing positions.

## Total if all stages: ~7-10 hours compute, ~$10 Modal

Well within remaining budget ($7.60). If Stage 1 alone succeeds, the rest become optional.

## Code changes needed

1. **LR schedule**: add env var `CASCADIA_TRAIN_LR_DECAY` (peak → end). ~15 min.
   - Modify `train_nnue` in `crates/cascadia-ai/src/nnue_train.rs`
   - Linear decay from peak to end across epochs
2. **Population opponents**: accept comma-separated paths for `CASCADIA_TRAIN_OPP_WEIGHTS`, sample per-game. ~30 min.
   - Modify `generate_samples_with_mode` in nnue_train.rs
   - Load vec of Arc<NNUENetwork>, random sample per game
3. **Validation-N flag** in head_to_head.py: already supports this via `--game-samples` multiplier. No change.

## Minimal first experiment (Stage 1 only — do this first)

If we only want to try ONE thing:

```bash
# Continue sym_v3 from iter 5 with 3× more games, 15 epochs, higher LR
CASCADIA_TRAIN_OPP_WEIGHTS=nnue_weights_mce93.bin \
CASCADIA_TRAIN_SEED=12345 \
./target/release/cascadia-cli 30000 --nnue-train \
  --lr 5e-4 --epochs 15 \
  --init-weights nnue_weights_sym_v3_iter5.bin \
  --weights nnue_weights_sym_v4_iter1.bin

# Validate with 5 samples × 4 rotations = 20 games vs mce93
python3 -u overnight/head_to_head.py \
  --strategies "mce_new,mce_anchor,mce_anchor,mce_anchor" \
  --strategy-weights "mce_new=nnue_weights_sym_v4_iter1.bin,mce_anchor=nnue_weights_mce93.bin" \
  --game-samples 5
```

**This is the cleanest first test.** Takes ~1 hour total. If it closes the gap meaningfully (say, 92+ base / 20%+ win rate), we know the recipe scales. If not, we need a different approach.

## Status at time of plan

- **All 5 iters of sym_v3 complete**. Last weights: `nnue_weights_sym_v3_iter5.bin` (90.33 base).
- **Infrastructure ready**: `CASCADIA_TRAIN_OPP_WEIGHTS`, `CASCADIA_TRAIN_SEED`, `--init-weights` all shipped.
- **No running processes** — all tournaments and training complete.
- **Modal budget remaining**: ~$7.60 of $20 total.

## Ready to execute when given the green light.
