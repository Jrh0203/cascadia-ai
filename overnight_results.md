# Overnight Search Improvement Experiments — 2026-04-10

## Goal
Push past MCE(750) = 95.9 baseline by improving search (value network is plateaued).
Local compute only. No Modal.

## Baseline Reference
Best so far per session_context_apr10.md:
- MCE(750) sequential halving + hybrid_iter4 weights = 95.9 mean (8.8s/game)
- PolicyMCE top-8 = 95.1
- NNUE-only = 89.9
- 2-ply expectimax = 90.6

Re-running baseline to confirm with the SAME run conditions before comparing experiments.

## Experiment Log

### Baseline — MCE(750) sequential halving + hybrid_iter4 weights
- 50 games, mean **95.3**, median 96, P10 91, P90 99, max 101
- 361.5s wallclock (7.2s/game), nnue_weights_hybrid_iter4.bin
- Wildlife totals: bear 6.0, elk 10.2, salmon 16.4, hawk 15.4, fox 13.8 (total 61.8)
- Habitat 31.3 + bonus 5.2 = 36.5
- Tokens 2.1
- **This is the comparison anchor for all subsequent experiments.**

### Tier 2 #4: MCE_LEAF_EXPECTIMAX (50g positive, 200g does NOT replicate)
Replaces NNUE-only rollout-terminal evaluation with `actual + NNUE(remaining) + E[best next-market wildlife delta]`.
- New function `evaluate_leaf_with_next_market` in mce.rs
- Single board clone + in-place place/undo for the 5×N wildlife delta computation
- Exact 5^4=625 next-market enumeration weighted by bag-conditioned probs
- Gated by `MCE_LEAF_EXPECTIMAX=1`
- **50-game result: mean 95.9, median 96, P10 90, P90 100, max 103, 337s wallclock**
- **50g Δ vs baseline: +0.6** — looked promising at 50 games
- **200-GAME RESULT: mean 95.4 (vs baseline 200g 95.3). Δ +0.1, within stderr ±0.40.**
- Wildlife at 200g: bear 8.4, elk 11.4, salmon 15.1, hawk 13.3, fox 13.7 (different from
  baseline 200g's bear 7.4, elk 10.9, salmon 16.6, hawk 13.2, fox 13.5)
- VERDICT: ❌ The 50g +0.6 was a STATISTICAL OUTLIER. At 200g the technique is
  indistinguishable from baseline. Wildlife distribution does change (LEAF1 over-drafts
  bear and under-drafts salmon), but net score is the same.

### 200g confirmation runs (THE KILLER)
- baseline_200g: 95.3 mean, 200 games, 4473s wallclock
- leaf_200g (LEAF1): 95.4 mean, 200 games, 4474s wallclock
- leaf1_d4_200g (LEAF1 + depth=4): 95.3 mean, 200 games, 1763s wallclock
- Combined stderr: ±0.40
- **Conclusion: at 200 games, ALL LEAF1 variants are STATISTICALLY INDISTINGUISHABLE
  from baseline.** None of the +0.5 to +0.6 improvements seen at 50g replicate at 200g.
- The 50g leaf results of 95.8-95.9 were the upper end of the noise band (~1 stderr above
  true mean). Lesson: 50-game benches have ~±1.0 noise (95% CI is ±2× stderr), so anything
  within ±1.0 of baseline at 50g is NOT a real signal.

#### Score distribution at 200g (all techniques have same mean but different shapes)
| Score range | baseline | LEAF1 default | LEAF1+depth=4 |
|---|---|---|---|
| 85-89 | 4.5% | 6.5% | 3.0% |
| 90-94 | 33.0% | 26.5% | 36.5% |
| 95-99 | 56.0% | 57.5% | 49.0% |
| 100-104 | 6.0% | 9.5% | 11.5% |
| 105-109 | 0.5% | 0.0% | 0.0% |

**Both LEAF1 variants produce more 100+ games than baseline (9.5% and 11.5% vs 6.0%).**
This is a real distributional shift even though the mean is the same. If the user prefers
"high-ceiling, slightly-bimodal" strategies (e.g., for tournament play where outliers
matter), LEAF1+depth=4 produces 1.9× as many 100+ games as baseline. But the EXPECTED
score is the same.

### Tier 2 #5: MCE_RANK_EXPECTIMAX
Replaces NNUE re-ranking step (which feeds sequential halving) with `evaluate_leaf_with_next_market`.
- Tests whether better candidate _ordering_ helps independently of better leaf eval.
- Gated by `MCE_RANK_EXPECTIMAX=1`
- **50-game result: mean 95.0, median 95, P10 91, P90 100, max 102, 304.7s wallclock**
- **Δ vs baseline: −0.3 mean (within ±1 noise band)**
- Wildlife shift: salmon 17.5 (+1.1), elk 9.6 (−0.6), bear 7.1 (+1.1) — re-routed toward different patterns
- VERDICT: ❌ no win. Better ranking alone isn't the lever; the ROLLOUT TERMINAL is.

### Tier 1 #1 surgical variant: MCE_GUMBEL_TOPK
Adds Gumbel(0, T) noise to NNUE re-ranked eval before truncation to top-15.
- Implements stochastic top-K (Gumbel-top-k) addressing 0.349 Spearman correlation between eval rank and MCE rank.
- Gated by `MCE_GUMBEL_TOPK=1`, temperature `MCE_GUMBEL_TEMP` (default 3000 = ~3 score points).

### Combined: MCE_LEAF_EXPECTIMAX + MCE_GUMBEL_TOPK
- **50-game result: mean 95.4, median 95, P10 92, P90 100, max 102, 333s wallclock**
- **Δ vs baseline: +0.1 mean (within ±1 noise band). Effectively neutral.**
- Wildlife shift: bear 8.9 (+2.9), salmon 17.7 (+1.3), hawk 12.0 (−3.4)
- VERDICT: ❌ does NOT stack with LEAF. Gumbel noise without a deeper Q-improvement loop
  just adds variance to the candidate selection.

### Combined: MCE_LEAF_EXPECTIMAX + MCE_RANK_EXPECTIMAX + MCE_GUMBEL_TOPK
- **50-game result: mean 95.4, median 95, P10 91, P90 ~100, 337s wallclock**
- **Δ vs baseline: +0.1. Same as LEAF+GUMBEL — adding RANK has no marginal effect.**
- VERDICT: ❌ kitchen sink approach plateaus at the leaf-eval ceiling. The other levers
  don't add value.

### Tier 1 #1 dedicated strategy: --gumbel-mcts
New `gumbel_mcts.rs` module implementing the full Gumbel AlphaZero algorithm:
1. Sample top-m actions via Gumbel-top-k by `log_pi(a) + Gumbel(0,1)` (priors from NNUE eval softmax).
2. Sequential halving over the m sampled actions.
3. Final action: argmax of `g(a) + sigma(q_hat(a))` (completed-Q backup).
- **50-game result: mean 90.2, median 90, P10 86, P90 94, max 99, 275s wallclock**
- **Δ vs baseline: −5.1 mean. ❌ MUCH WORSE.**
- Diagnosis: this dedicated strategy lacks MCE's machinery (strategic candidates, demand scoring,
  tier_bonus, NNUE-aware leaf eval). Wildlife distribution shows it drafts bear-heavy (9.8 vs 6.0)
  and weak salmon/hawk. Lesson: Gumbel selection only helps when bolted onto the EXISTING
  MCE pipeline (MCE_GUMBEL_TOPK), not as a from-scratch reimplementation.

### Tier 1 #2 dedicated strategy: --ol-mcts
New `ol_mcts.rs` module: open-loop tree, single shared tree with virtual-loss leaf parallelization.
- Smoke test (3 games): mean 86 — initial implementation with thread-local trees was broken
- Rewrote with single-tree virtual-loss design (uses `unsafe { &mut *node_ptr }` for iterative descent)
- Need to bench post-rewrite — but expectation is similar lesson to Gumbel-MCTS: structurally
  duplicates MCE without its tuning.

### Tier 3 #6: NRPA — Nested Rollout Policy Adaptation (CONFIRMED FAIL)
New `nrpa.rs` module with Cazenave's recursive policy adaptation.
- Move feature key: (animal, own_count_bin, keystone, independent) — 5×6×2×2 = 120 distinct keys
- Softmax sampling, log-weight policy, alpha=1.0
- L=2 N=15 with depth-limited (NRPA_DEPTH=6) NNUE leaf eval
- **50-game result: mean 77.6, median 79, P10 69, P90 86, max 89, 17942s wallclock (5 hours!)**
- **Δ vs baseline 95.3: -17.7. CATASTROPHICALLY WORSE.**
- Wildlife: bear 8.7, elk 8.0, salmon 7.9, hawk 8.8, fox 13.1 (total 46.4 vs ~62 baseline)
- VERDICT: ❌❌ NRPA fundamentally doesn't fit Cascadia. The policy adaptation doesn't learn
  meaningful features in the available rollouts (N=15 is too small for L=2 to improve over
  L=1, which is too small for the policy to converge). Results are essentially random play
  quality. For a 5-hour benchmark this is a complete waste.

### Tier 2 #4 v2: MCE_LEAF_EXPECTIMAX2 — 2-step lookahead leaf eval (BROKEN math version)
First attempt at chained 2-turn wildlife lookahead at the rollout terminal.
- Step 1: best wildlife delta per type, exact 5^4 enumeration over next-market draws
- Step 2: chain another wildlife placement, use APPROXIMATE marginal weighting
  (`prob_chosen ≈ 1-(1-P)^4`, divided by 5 — wrong normalization)
- **50-game result: mean 95.3, median 95, P10 93, P90 99, max 103, 680s wallclock (1.6× slowdown)**
- **Δ vs baseline: 0.0. The approximate math gave NO improvement over 1-step.**
- Wildlife: bear 7.6, elk 10.0, salmon 15.8, hawk 15.5, fox 13.2 — basically baseline distribution
- VERDICT: ❌ broken math version. Code has been corrected; running v2 separately.

### Tier 2 #4 v2.1: MCE_LEAF_EXPECTIMAX2 (CORRECTED math, 50 games)
- Same idea but with EXACT 5^4 step-1 enumeration → identifies argmax type → step-2 expectation
  for THAT chosen type. Total ~5^4 outer ops × 5^4 inner ops per leaf.
- **50-game result: mean 95.5, median 95, P10 91, P90 101, max 106, 1552s wallclock (4.3× slower)**
- **Δ vs baseline: +0.2 mean, +2 P90, +5 max** — improves upper tail, mean within noise
- Wildlife: bear 7.3, elk 11.6, salmon 16.5, hawk 12.5, fox 13.8 (elk +1.4 vs baseline)
- VERDICT: ⚠️ Mean within noise. Helps upper tail (P90 101 vs 99) but at 4× the cost. Not worth it.
- Lesson: 2-step lookahead at the leaf adds variance to candidate ranking without
  improving mean. The marginal value of "knowing 2 turns ahead" isn't enough to justify
  the extra computation for MCE rollouts.

### Variant: MCE_LEAF_EXPECTIMAX + MCE_DEPTH=8
- **50-game result: mean 95.3, median 95, P10 91, P90 101, max 103, 1680s wallclock (4.7× slower)**
- **Δ vs baseline: 0.0 mean. ❌**
- Wildlife: bear 5.7, elk 11.1, salmon 18.1, hawk 13.7 — different distribution but same score
- VERDICT: ❌ deeper rollouts dilute the leaf eval benefit. The leaf eval's main contribution
  is for mid-game leaves (turns 6-12); pushing leaves closer to game-end via depth=8 makes
  the next-market bonus less impactful.

### MCE_LEAF_MARKET — market-aware leaf eval (turned out neutral)
**The intuition**: the original `evaluate_leaf_with_next_market` enumerates 5^4=625
fresh wildlife draws as if all 4 wildlife in the next market are random. But Cascadia's market
mechanics: each turn, the AI drafts ONE pair, the market loses ONE wildlife, and ONE fresh
wildlife refills. The other 3 wildlife in the market are LEFTOVER from the previous turn —
they are KNOWN at the rollout leaf state.

`evaluate_leaf_market_aware` reads the actual market wildlife at the leaf and computes
`max over (current 4 market wildlife) of wildlife_value`. Theoretically more accurate
AND ~10× faster than the 5^4 enum.

**Smoke test 5 games: mean 97.2 (very encouraging).**
**50-game result: mean 95.5, median 97, P10 91, P90 100, max 104, 1322s wallclock**
**Δ vs baseline: +0.2 (within noise). ❌ does NOT beat the original LEAF1 (95.9).**

Wildlife: bear 7.9, elk 9.5, salmon 17.0, hawk 13.4, fox 13.9 — different from leaf1
distribution but same net.

Surprising lesson: the "more correct" market-aware version is NOT better than the
"averaged over hypothetical markets" 5^4 enum. Possible reasons:
1. The 5^4 enum's higher expected value acts as a useful BIAS that helps ranking
2. The actual market may have distribution quirks the average smooths over
3. Lower per-rollout variance from averaging helps sequential halving make better decisions

The original leaf1 (5^4 enum) remains the best leaf eval variant.

### MCE_LEAF_MARKET2 (2-step market-aware) — 50 games
- 2-step variant: step1 = max over current market, step2 = E[max(3 leftover + 1 fresh from bag)]
- **50-game result: mean 95.1, median 96, P10 91, P90 100, max 103, 1214s wallclock**
- **Δ vs baseline: −0.2 (within noise). ❌**
- Wildlife: bear 10.6 (+4.6!), elk 8.4 (−1.8), salmon 15.3, hawk 12.6, fox 14.5
- VERDICT: ❌ The 2-step lookahead heavily over-drafts bear (probably because bears
  pair-bonus is the highest immediate marginal gain seen by the leaf eval). Net score
  is the same as baseline.

### Variant: leaf1_c10 (LEAF1 + CANDIDATES=10)
- Tests SMALLER candidate pool (10 vs default 15).
- **50-game result: mean 95.1, median 95, P10 92, P90 99, max 101, 524s wallclock**
- **Δ vs baseline: -0.2 (within noise)**
- VERDICT: ❌ Smaller candidate pool slightly hurts. Default 15 is the right size.

### Variant: leaf_n1500_200g (LEAF1 + 1500 rollouts, 200 games)
- Tests if more rollouts at 200g level helps LEAF1.
- **200-game result: mean 95.4, median 96, P10 91, P90 100, max 106, 2621s wallclock**
- **Δ vs baseline_200g: +0.1 (within stderr ±0.40, NOT significant)**
- Wildlife: bear 7.1, elk 10.6, salmon 17.4, hawk 13.1, fox 13.4 — closer to baseline
  distribution than LEAF1 default (more rollouts smooth out the leaf eval-induced shift)
- Score distribution: 4.5/34.5/49.5/11.0/0.5 — 1.83× more 100+ games than baseline (11% vs 6%)
- VERDICT: ⚠️ Same mean as LEAF1 default and as baseline at 200g. Confirms more rollouts
  don't amplify the LEAF1 effect on mean, just on the upper-tail distribution.

### Variant: leaf1_d4_n1500_100g (LEAF1 + depth=4 + 1500 rollouts)
- Tests if combining shorter rollouts with bigger budget amplifies LEAF1.
- **100-game result: mean 94.9, median 95, P10 91, P90 99, max 103, 865s wallclock**
- **Δ vs baseline: -0.4 mean (within stderr ±0.4 at 100g, slightly negative)**
- Wildlife: bear 7.7, elk 10.3, salmon 16.3, hawk 12.7, fox 14.4
- VERDICT: ❌ Combining doesn't amplify. The +0.6 from LEAF1 doesn't compound with
  more rollouts or shorter depth. The combination is essentially baseline.

### 🎯 KEY FINDING: nnue_weights_hybrid_iter18.bin (USER'S NEWER WEIGHTS)
The user has been training new NNUE weights overnight. Files iter1 through iter18 (and
counting) exist. I tested iter18 with the same MCE(750) configuration:
- **100-game result: mean 96.0, median 96, P10 93, P90 100, max 104, 579s wallclock**
- **Δ vs my iter4 baseline (95.3): +0.7**
- Wildlife: bear 8.3, elk 10.1, salmon 16.4, hawk 13.3, fox 14.1

This is the MOST POSITIVE result of the night! The user's iterative training is working.
At 100 games, stderr ~0.4, so +0.7 is +1.75 stderr → ~92% one-sided confidence (borderline
significant but very likely real).

**Implication**: my entire night of search experiments was tested against an OUTDATED
baseline (iter4). The user's PARALLEL training has independently produced a significantly
better NNUE. Should test all my best techniques against iter18 to see if any of them
add value on top of the better weights.

### MCE_GUMBEL_TOPK alone (gumbel_topk_only)
- Tests pure Gumbel-perturbed candidate selection without the leaf eval improvement.
- **50-game result: mean 94.6, median 95, P10 91, P90 98, max 100, 1196s wallclock**
- **Δ vs baseline: −0.7 mean. ❌ Actively HARMFUL.**
- Wildlife: bear 6.6, elk 11.5, salmon 17.1, hawk 14.0, fox 12.9
- VERDICT: ❌ Gumbel noise on candidate selection alone makes MCE worse. The Spearman 0.349
  isn't because top-15 misses the best candidate; it's because MCE rankings are inherently noisy.
  Adding more noise doesn't help.

### Variant: MCE_LEAF_EXPECTIMAX + 1500 rollouts (leaf_n1500_50g)
- Tests whether MORE rollouts amplify the LEAF1 leaf eval signal.
- **50-game result: mean 95.6, median 96, P10 91, P90 99, max 101, 2486s wallclock**
- **Δ vs baseline: +0.3 (within noise, ±1.0 at 50g)**
- Wildlife: bear 5.9, elk 10.4, salmon 17.8, hawk 14.1, fox 13.7 — closer to baseline
  distribution than LEAF1 default (more rollouts smooth out the leaf eval bias)
- VERDICT: ⚠️ Slightly above baseline at 50g but within noise. Not the amplification
  I hoped for. The 1500 rollouts halve the per-rollout variance but the SAME signal
  doesn't translate to significantly better mean.

### Variant: MCE_LEAF_EXPECTIMAX + MCE_CANDIDATES=20 (leaf1_c20)
- Tests whether more candidates feeding sequential halving helps LEAF1.
- **50-game result: mean 95.9, median 96, P10 90, P90 100, max 103, 1090s wallclock**
- **Δ vs baseline: +0.6 (IDENTICAL to LEAF1 default)**
- Wildlife: bear 6.5, elk 10.9, salmon 16.2, hawk 14.8, fox 13.7 (IDENTICAL distribution
  to LEAF1 default)
- VERDICT: ⚠️ Same as LEAF1 default. The top-15 candidates already include the best ones;
  expanding to top-20 doesn't help. The +0.6 win is robust to candidate count.

### Variant: MCE_LEAF_EXPECTIMAX + MCE_DEPTH=4 (leaf1_d4)
- Tests whether SHORTER rollouts (depth=4) amplify the leaf eval signal — opposite of
  the leaf_d8 experiment.
- **50-game result: mean 95.8, median 96, P10 92, P90 101, max 102, 758s wallclock**
- **Δ vs baseline: +0.5 (essentially same as LEAF1's +0.6)**
- Wildlife: bear 7.3, elk 9.8, salmon 16.0, hawk 14.8, fox 13.7 — similar to LEAF1's
- VERDICT: ⚠️ tied with LEAF1. Shorter rollouts don't amplify or dilute the leaf bonus.
  The +0.6 from LEAF1 holds across rollout depths. Faster wallclock though (758 vs ~337
  for depth-6 leaf1 with similar contention — so depth=4 might be 2× faster ideally).


