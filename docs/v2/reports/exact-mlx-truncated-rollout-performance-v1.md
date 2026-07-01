# Exact MLX Truncated-Rollout Performance V1

Date: 2026-06-14

## Result

Two-focal-turn MLX bootstrapping is qualified as a high-throughput screening
tier, not as a replacement for full terminal R600 search.

The 30-game confirmation used raw seeds 34200-34229 across john1, john2, and
john3. Each host ran ten games with the qualified parent model, R600
sequential halving, and `after-focal-move` leaf timing.

| Metric | Result |
|---|---:|
| Treatment mean | 93.775 |
| Paired control mean | 92.342 |
| Mean paired delta | +1.433 |
| Paired standard deviation | 1.995 |
| Standard error | 0.364 |
| 95% paired CI | `[+0.720,+2.147]` |
| Treatment seconds per game | 12.237 |
| Current full-terminal reference | 137.910 s/game |
| Effective search speedup | **11.27x** |
| Three-node wall time | 161 s |
| Effective cluster throughput | about 671 games/hour |
| Neural rows | 19,458,628 |
| Rollout samples | 1,379,892 |
| Bootstrapped samples | 1,310,879 |

## Strength Boundary

The directly matched seeds 34099, 34100, 34103, and 34106 produced:

| Evaluator | Mean |
|---|---:|
| Full terminal R600 | 96.875 |
| Two-turn, immediate afterstate | 93.188 |
| Two-turn, after opponent round | 93.000 |

The immediate-afterstate screen therefore lost 3.688 points on this small
paired diagnostic. It must not support champion, qualification, or promotion
claims without a terminal-rollout rerun.

## Leaf Models

The search can use separate MLX processes for policy decisions and leaf
evaluation. This preserves the qualified parent for root priors and rollout
actions while testing a specialized bootstrap model.

The existing absolute rollout-return checkpoint regressed. The existing joint
return/ranking checkpoint improved the three-seed shallow mean by 0.333 points
but remained 4.250 points below full terminal search. Neither checkpoint is a
qualified leaf evaluator.

## Contract

- No rollout limit means full terminal behavior.
- `--rollout-turns 2 --rollout-leaf-timing after-focal-move` is the canonical
  high-throughput screen.
- `--rollout-leaf-timing after-opponent-round` is available for controlled
  comparisons but did not improve the matched result.
- `--leaf-model-dir` changes only bootstrap evaluation.
- Every survivor must be rerun on common seeds without `--rollout-turns`.
