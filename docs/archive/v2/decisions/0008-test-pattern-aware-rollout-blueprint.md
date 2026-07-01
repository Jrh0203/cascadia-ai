# ADR 0008: Test Pattern-Aware Rollout Policy

Status: rejected after preregistered pilot on 2026-06-10.

## Context

The confirmed H6 search has a useful K8+H6 root frontier, but its future plies
use exact immediate-score greedy. Root-frontier widening and longer greedy
rollouts have not improved it. The newly confirmed pattern-aware policy is
stronger than both greedy and K8 while remaining deterministic, local, and
public-information-only.

Search literature supports treating the rollout policy as a separate source of
value quality. Rollout-based game-tree pruning relies on informative simulated
continuations, Expert Iteration improves an apprentice from search-generated
experience, and Monte Carlo search quality depends on the policy producing its
trajectories. Work on partially observed team games likewise demonstrates the
importance of search policies that respect each player's information state.

Primary sources:

- [Pruning Game Tree by Rollouts](https://cdn.aaai.org/ojs/9371/9371-13-12899-1-2-20201228.pdf)
- [Expert Iteration via Experience Distribution](https://arxiv.org/abs/2006.00283)
- [Monte Carlo Tree Search and Reinforcement Learning](https://www.jair.org/index.php/jair/article/download/11099/26289/20632)
- [SPARTA: Search in Partially Observable Team Games](https://ojs.aaai.org/index.php/AAAI/article/view/6208/6064)

## Decision

Test one isolated change: retain H6's K8+H6 root candidates, four common
determinizations, four future plies, and exact acting-seat base score at the
leaf, but replace greedy future actions with frozen
`pattern-aware-v1-k8-h6-b8-m4`.

The full configuration must first complete one paired game with treatment
runtime at or below 60 seconds. A ten-game pilot then requires at least +0.25
paired points, habitat and wildlife deltas each at least -0.5, and the same
runtime ceiling. Only a passing pilot may advance unchanged to a disjoint
50-game confirmation whose paired 95% confidence interval must exclude zero.

No parameter search is permitted between stages. A rejection remains useful:
it distinguishes root evaluation error from trajectory-policy error and
prevents a stronger standalone policy from being assumed to be a stronger
rollout policy.

## Outcome

The mandatory smoke passed at 15.408 treatment seconds per game. The disjoint
ten-game pilot then scored 90.525 versus H6 at 91.075, a paired delta of
-0.550 with 95% CI -1.796 to 0.696 and a 4-0-6 record. Habitat was -0.050,
aggregate wildlife -0.400, and Nature Tokens -0.100. Runtime remained within
the registered ceiling at 7.803 seconds per game.

The primary +0.25 advancement gate failed, so no 50-game confirmation was
run. The four-ply exact-score leaf does not reliably realize the standalone
policy's longer-term setup value. Keep the implementation as reproducible
research infrastructure, but do not use it as a teacher or product policy.
