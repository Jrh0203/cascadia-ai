# Hypothesis: Why NNUE doesn't beat greedy by as much under alt rules as under Card A

## The numbers

- **Card A**: NNUE (v4opp) = 95.94, greedy-MCE-750 = 93.98, **NNUE adds +1.96**
- **Alt rules** (Bear C, Elk B, Salmon D, Hawk D, Fox B): NNUE (cards_alt iter15 + alt candidates) = 97.2, greedy-MCE-750 = 96.5 (or 96.6 with alt cands), **NNUE adds +0.7**

NNUE delivers ~35% as much value-add under alt rules as under Card A.

## Three layered explanations, in order of importance

### 1. Card D Hawk has fundamentally higher score variance per state

Bear pairs / elk lines / salmon runs / hawk isolation (Card A) are all **incremental, additive, locally-determined patterns**. Adding one piece changes the pattern by ±0-9 pts. Score-from-state is smooth in the placement decision.

Card D Hawk pair scoring is **discrete, matching-problem-shaped, catastrophic-error-mode**:

- One hawk in the wrong spot can BLOCK an existing LOS pair (lose 4-9 pts)
- One hawk in the right spot can ENABLE a pair worth 9 pts
- The same hawk can be a member of multiple potential pairs but only counts in one (Hungarian algorithm)
- Adding a non-hawk wildlife between two hawks transitions a pair from 0 pts → 4 pts → 7 pts → 9 pts based on type-count
- A whole sequence of moves can collectively enable or destroy 30+ pts

The variance of "score-from-this-state" is 50%+ higher for alt rules than Card A. The NNUE training labels reflect this — RMSE 6.07-6.14 across three training runs vs v4opp's 4.81. **Direct mathematical consequence**: NNUE's L2-fit can't sharpen below the variance floor.

MCE-750 doesn't suffer from this because it doesn't predict — it samples 750 times and averages. **High-variance scoring favors sampling-based estimation over function-approximation estimation.** That's the deepest reason.

### 2. The alt-aware candidate generators were the real lift, not the value function

The candidate-fix lifted NNUE +1.3 (95.9 → 97.2) while only lifting greedy-MCE +0.1. So **most of the apparent NNUE-over-greedy gap under alt rules came from a CANDIDATE GENERATION fix, not the network itself.**

If you decompose the 97.2 NNUE result:
- 96.5 from greedy-MCE-750's search
- +0.7 from the value-function residual

That's the actual NNUE contribution: ~0.7. Card A's NNUE contribution was ~1.96. Ratio: 36%. This matches the variance/RMSE ratio cleanly (4.81/6.10 = 79% of Card A's prediction quality, but the marginal value drops faster than RMSE because high-variance regions have less room for value-function info anyway).

### 3. Greedy with the new alt-aware potential is unusually strong

The hand-crafted `board_potential` per-card dispatch (Bear C, Elk B, Salmon D, Hawk D, Fox B) raised greedy alone from 74.2 → 81.0 (+6.8 pts). This made the **greedy baseline that NNUE has to beat MORE competitive**.

For Card A, greedy with potential ≈ 85, MCE-750 = 93.98. There's a gap MCE-750 fills that NNUE can incrementally improve.

For alt rules, greedy with potential = 81, MCE-750 = 96.5. Larger gap (15.5 pts vs Card A's 9 pts), and MCE-750's averaging is doing 95% of the work to close it. The remaining 5% gap (= ~0.7 pts) is what NNUE can squeeze.

## Synthesis

The NNUE-over-greedy gap is governed by three things compounding:

1. Card D variance puts a hard ceiling on RMSE (~6.0)
2. With high-variance scoring, MCE-750 sampling extracts more value than NNUE-prediction can
3. The new alt-aware potential made greedy-MCE-750 already strong, so NNUE has less to improve

**Alt rules' inherent NNUE-over-greedy ceiling appears to be ~+0.7-1.3 pts.** We're at it. No realistic feature/training/architecture investment will push past it without first addressing Card D's variance source (which would require either: a hawk-specific value head with separate variance budget, or a fundamentally different approach like MCTS with NNUE policy + value).

The honest takeaway from this saga: **for high-variance scoring rules, search dominates so completely that value-function investments yield deeply diminishing returns.** The alt-aware candidate generation is the highest-leverage lever and it's now fixed.
