# Exact Opponent Greedy Decision Reuse Rejection

Status: **rejected before implementation; diagnostic removed**

Date: 2026-06-15

## Hypothesis

Full-terminal rollouts share random seeds across root candidates. If many
rollout states reached the same acting-opponent board and public market, one
exact greedy decision could be computed for the group and then applied to
each state's independent hidden bags.

The proposed key was the existing collision-free `CandidateCacheKey`, which
stores every public value read by the qualified greedy policy. Hidden bag
order is deliberately absent because it affects only the market refill after
the shared move is applied.

## Audit

A diagnostic-only build advanced rollout states in synchronized opponent
plies, after each required overflow replacement. It counted total greedy
requests and exact unique decision keys, but still computed and applied every
state's greedy move independently. The production path was unchanged unless
the temporary diagnostic flag was enabled.

The frozen seed-34400 K32/R600 run produced identical results on john2 and
john3:

| Metric | Result |
|---|---:|
| Opponent plies observed | 22,544 |
| Greedy move requests | 1,390,050 |
| Exact unique decision states | 1,382,936 |
| Reusable requests | 7,114 |
| Reuse rate | **0.512%** |
| Mean requests per unique state | 1.0051 |
| Largest equivalence class | 14 |

Both hosts reproduced scores `[102,96,92,95]`, 3,920 neural batches,
6,121,807 logical rows, 5,062,305 physical rows, 3,716 rollout waves,
46,207 rollout samples, zero bootstraps, zero policy fallbacks, and clean
shutdown.

## Performance Ceiling

The accepted john3 stage trace attributes 3,156.583 ms to all opponent
advancement. Even under the unrealistic assumption that every reusable
request costs the mean request time and grouping is free, eliminating 0.512%
of those calls can save only about **16 ms** per game.

That is less than 2% of the remaining 0.916-second Phase 0 gap. The temporary
synchronization and keying path instead raised opponent advancement to
5,694.061 ms on john2 and 5,726.424 ms on john3.

## Verdict

Reject before implementing decision reuse. The exact equivalence classes are
far too sparse to justify synchronization, grouping, movement fan-out, or a
new production cache. All diagnostic code and its environment flag were
removed.

The next Phase 0 experiment must reduce the shared candidate-generation core
used by both rollout templates and greedy opponents, where a double-digit
improvement can still clear the 10x gate.

Machine-readable evidence:
`docs/v2/reports/exact-opponent-greedy-decision-reuse-rejection-v1.json`.

Raw host evidence is archived under
`artifacts/performance/exact-opponent-greedy-decision-reuse-audit-v1/`.
