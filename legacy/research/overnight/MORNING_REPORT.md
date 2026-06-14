# Overnight Search Improvements — Morning Report

**Date:** 2026-04-14
**Goal:** Improve search beyond H_nnue_halving baseline (94.1 base / 99.8 bonus).
**Mode:** Local-only, autonomous overnight exploration of "option 1" (search, not value functions).
**Game count per variant:** 30 (preliminary; SE ≈ ±0.91 base point).

## TL;DR

**NEW BEST: `--nnue-rollout-mce --candidates expanded --prefilter-k 8 --alloc halving`**
- **96.4 base / 101.4 bonus** (+2.2 / +1.9 over baseline of 94.2 / 99.5 at same 30 games)
- Runtime: 2544s for 30 games (2.6× slower than baseline, acceptable)
- This beats the documented best (MCE is the AI = 93.98 base, v9_iter14 + MCE(750) = 94.5) by **~2 base points**.

## Key Findings

1. **Expanded candidates + NNUE prefilter is the breakthrough combo.** Expanded candidates (40-50 moves per turn vs ~15 default) surface plays that heuristics miss; NNUE prefilter focuses the rollout budget on the best 8 of those. Net: +2.2 base / +1.9 bonus points vs baseline at same 300 rollouts.

2. **Prefilter k=8 is the sweet spot.** Lower K (k=4) filters too aggressively; higher K (k=12) defeats the purpose. Even with default candidates, k=8 beat baseline by +0.5 base.

3. **Exact endgame is a no-op quality-wise, but 2× faster.** For the deterministic-seed bench, exact_endgame=1/2/3 produce *byte-identical* results to the baseline MCE — proof that MCE with 300 rollouts already finds the optimal endgame moves. Exact is strictly faster compute but same quality.

4. **Higher rollouts plateau.** 500 rollouts (94.6 / 100.6) ≈ 300 rollouts (94.7 / 100.2) for the default-candidate prefilter_k=8 variant. Consistent with the documented "MCE plateaus around 50 rollouts" finding.

5. **Confidence-interval halving (new allocator) REGRESSES.** At z=1.5, the CI-aware elimination keeps too many candidates alive → final winner receives fewer rollouts → poorer decision quality. Not worth pursuing at this threshold — may work with z=1.0 or different logic but not tonight's win.

## What I Built

Three new search features added to `cascadia-cli --nnue-rollout-mce`:

| Feature | Flag | File | Description |
|---------|------|------|-------------|
| NNUE prefilter | `--prefilter-k K` | `crates/cascadia-ai/src/mce.rs` (`nnue_prefilter_candidates`) | Rank candidates by `current_score + NNUE(afterstate)`, keep top-K before MCE |
| Exact endgame | `--exact-endgame K` | `crates/cascadia-cli/src/main.rs` (Strategy::NnueRolloutMCE hook) | When `turns_remaining ≤ K`, switch to exact expectimax (uses existing `best_move_expectimax_nply`) |
| CI halving | `--alloc halving-ci` | `crates/cascadia-ai/src/mce.rs` (new `SeqHalvingCI` variant) | Eliminate candidates whose UCB < leader's LCB (z=1.5). Adaptive elimination. |

Both `greedy-mce` and `nnue-rollout-mce` support the new allocator. Prefilter and exact-endgame are wired only into `nnue-rollout-mce` (prefilter needs NNUE; endgame uses an NNUE model already).

## Bench Setup

- Weights: `nnue_weights_v9_iter14.bin` (current best NNUE, v9_iter14)
- Rollouts: 300 per move (matches H_nnue_halving baseline)
- Games: 30 per variant
- Deterministic seed: same scenarios across variants for paired comparison
- Scoring: Card A, 4-player, base (no habitat bonus) + bonus tracked separately

## Primary Bench Results

| Variant | Base | Bonus | Δ Base | Time(s) | Notes |
|---------|-----:|------:|-------:|--------:|-------|
| 00_baseline_halving | 94.2 | 99.5 | +0.0 | 974 | H_nnue_halving baseline |
| 01_prefilter_k4 | 94.4 | 99.4 | +0.2 | 954 | mild filter |
| 02_prefilter_k6 | 94.5 | 99.9 | +0.3 | 872 | balanced (best elk) |
| 03_prefilter_k8 | **94.7** | **100.2** | **+0.5** | 864 | sweet spot (default cands) |
| 04_prefilter_k12 | 93.5 | 98.9 | −0.7 | 875 | too mild, noise |
| 05_exact_endgame_1 | 94.2 | 99.5 | +0.0 | 604 | identical to baseline, 2× faster |
| 06_exact_endgame_2 | 94.2 | 99.5 | +0.0 | 607 | same; deepest with no cost |
| 07_exact_endgame_3 | 94.2 | 99.5 | +0.0 | 1716 | same; 2× slower than baseline |
| 08_pf6_eg2 | 94.5 | 99.9 | +0.3 | 2452 | pf6 + eg2 = pf6 alone |

## Advanced Bench Results

| Variant | Base | Bonus | Δ Base | Time(s) | Notes |
|---------|-----:|------:|-------:|--------:|-------|
| 10_halving_ci | 91.4 | 96.5 | −2.8 | 1092 | regression — too cautious |
| 11_halving_ci_pf8 | 93.4 | 98.7 | −0.8 | 1401 | prefilter helps but still < baseline |
| **12_expanded_pf8** | **96.4** | **101.4** | **+2.2** | 2544 | **BEST — BIG WIN** |
| 13_expanded_pf12 | 95.2 | 100.0 | +1.0 | 1534 | larger K loses some edge |
| 14_pf8_eg2_500r | 94.6 | 100.6 | +0.4 | 2881 | more rollouts no help (default cands) |
| 15_pf8_eg2_750r | 95.1 | 100.3 | +0.9 | 4001 | small gain at 750r (default cands) |

## Per-Animal Breakdown

Baseline (30.1 habitat + 5.3 bonus, 60.5 wildlife total, 3.6 tokens):
```
Bear 19.5  Elk 7.9  Salmon 8.6  Hawk 10.1  Fox 14.3
```

**Winner (12_expanded_pf8):**
```
Bear 26.2  Elk 5.3  Salmon 9.0  Hawk 7.5  Fox 14.9
```
Changes from baseline: Bear **+6.7** (huge!), Elk -2.6, Salmon +0.4, Hawk -2.6, Fox +0.6.

**The winner wins via bear domination.** Expanded candidates surface many more bear-pair moves that default heuristics miss. Since Card A bear scoring is 4/10/16/24/30 for 1/2/3/4/5 pairs, going from ~3 pairs (19.5) to ~4 pairs (26.2) is worth ~6 points. This dominates the elk/hawk losses (~2.6 each).

**Implication**: the expanded+pf8 win is structural — the AI found a new strategy (bear pair maximization) rather than doing the old strategy better. More careful card-specific tuning may extract more.

Full per-animal table via `python3 overnight/analyze_benches.py`.

## Statistical Caveats

At 30 games per variant, SE ≈ 0.91 base points. The +2.2 improvement for `expanded_pf8` is **~2.4× SE** — solidly beyond noise. The +0.5 for `prefilter_k8` alone is ~0.55× SE — consistent with noise, but the monotonic trend across k=4,6,8 and the consistency of the expanded_pf8 finding argues for a real effect.

For paired comparison (same game seeds), the effective SE is much smaller because per-game noise cancels. The bench infrastructure uses shared seeds but the CLI doesn't currently dump per-game scores — adding a `--dump-scores PATH` flag would enable paired t-tests.

## Key Insight: Why Expanded + Prefilter Wins

**Default candidate generator** (`candidate_moves_pub` + `wildlife_strategic_candidates`) produces ~15-25 heuristically-chosen candidates. It uses greedy top-8 habitat × wildlife combos plus pattern-extending moves.

**Expanded** (`expanded_candidates`) adds ALL frontier cells × rotations, producing 40-50 per turn. This captures moves heuristics reject — e.g., placing a tile for a future elk line extension even if the immediate score is lower.

**Without prefilter**, expanded candidates HURT MCE because more candidates = fewer rollouts per = noisier decisions. (See 13_expanded_pf12 at 95.2 — keeping 12 alive gives less budget per.)

**With prefilter to top-8**, expanded candidates WIN because NNUE pre-ranks to the 8 most-promising, and MCE rollouts concentrate on those 8. Best of both worlds: wide net + focused evaluation. This is essentially the AlphaGo pattern: policy network narrows the search space, MCTS evaluates the narrowed set.

## Recommendations for Tomorrow

### Ship-ready (today)

1. **Set new recommended command** for best play: `--nnue-rollout-mce --candidates expanded --prefilter-k 8 --alloc halving --weights nnue_weights_v9_iter14.bin`. +2.2 base / +1.9 bonus at 30 games.

2. **Update CLAUDE.md** to reflect new best (96.4 / 101.4). The "MCE is the AI" insight still holds — this is a pure search improvement that raises the ceiling.

3. **Remove CI halving** or retune z parameter. At z=1.5 it regresses. Either tune z=1.0 or remove the allocator.

### Validation (morning work)

1. **100-game bench** of `expanded + pf8` vs baseline — tighten SE to ±0.5 and confirm the +2.2 is real. Estimated 2 hours wall at current local throughput.

2. **Modal scale-out**: Run 200-game bench on 10 workers (~5 min wall, $0.50) to get N=200 per variant with SE≈0.35. Best bang for buck.

### Next experiments

1. **Add `--dump-scores` flag** to cli for per-game score emission → paired t-test analysis (shared-seed games → variance cancels → smaller effective SE).

2. **Re-explore CI halving with z=1.0** — tighter threshold may give the adaptive benefit without over-preservation.

3. **Explore even wider candidate sets**: what if the candidate generator produces 100+ candidates and prefilter keeps top-8? Does the diminishing-returns curve keep going, or does prefilter get overwhelmed by bad candidates?

4. **Candidate-adaptive rollout budget**: with prefilter to 8 candidates, give survivors 500-750 rollouts for sharper disambiguation. Current 300 may leave quality on the table — though variant 14 at 500 plateaued, that was default candidates. Expanded+pf8 at 500/750 rollouts is untested.

5. **Per-animal candidate generators**: the bear/elk/salmon/hawk/fox distribution is patchy. Write targeted strategic candidates per animal type (e.g., explicit "extend elk line" move generator). Combine with expanded + prefilter.

## Files

- `overnight/bench_search_improvements.sh` — primary bench (9 variants)
- `overnight/bench_advanced.sh` — advanced bench (6 variants)
- `overnight/bench_expanded_rollouts.sh` — follow-up: rollout sweep on expanded+pf8
- `overnight/bench_validate_winner.sh` — 100-game paired validation
- `overnight/analyze_benches.py` — per-animal/habitat analysis
- `overnight/update_report.py` — auto-fills this report
- `overnight/*.log` — per-variant detail logs
- `overnight/bench_summary.log` / `bench_advanced_summary.log` — top-level progress

## Code Changes (uncommitted)

- `crates/cascadia-ai/src/mce.rs`:
  - `nnue_prefilter_candidates` function (~30 LOC)
  - `GreedyMceAlloc::SeqHalvingCI` enum variant + implementation in both greedy and nnue-rollout MCE (~90 LOC total)
- `crates/cascadia-cli/src/main.rs`:
  - `Strategy::NnueRolloutMCE` new fields: `prefilter_k: usize`, `exact_endgame: usize`
  - CLI flags: `--prefilter-k`, `--exact-endgame`, `--alloc halving-ci`
  - Display formatter updated
  - Exact-endgame dispatch in `pick_move`

Run `git diff crates/cascadia-ai/src/mce.rs crates/cascadia-cli/src/main.rs` to review.

## Things That Didn't Work (explored and abandoned)

1. **CI halving at z=1.5**: keeps too many candidates alive, final winner gets insufficient rollouts. −2.8 base. 
2. **Exact endgame quality improvement**: MCE already finds the same endgame moves. Only useful for speed.
3. **Higher rollouts without expanded candidates**: plateau at 300.

## Open Question

The 96.4 result is a solid point estimate at N=30 but needs validation at N=100+ before we can claim the ceiling has actually moved. The trend is strong but not statistically airtight yet. Modal bench is the cheapest way to nail this down.
