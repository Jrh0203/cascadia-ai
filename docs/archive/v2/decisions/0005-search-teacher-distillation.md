# ADR 0005: Distill Search Preferences, Not Final Outcomes

Status: accepted on 2026-06-10.

## Context

The first fresh MLX model predicted final score with 2.817 held-out MAE but only
0.163 correlation. It failed as an unconstrained policy at 41.25 mean and
regressed exact greedy by 2.688 points even behind a top-8 filter.

Fair public-information search, by contrast, improved greedy by 2.555 points in
a disjoint 50-game confirmation. Expanding candidate breadth also exposed a
measured top-4 value-recall deficit.

The relevant primary methods separate planning from generalization:

- [Expert Iteration](https://arxiv.org/abs/1705.08439) trains an apprentice from
  policies improved by search.
- [Policy Distillation](https://arxiv.org/abs/1511.06295) transfers an expert's
  action preferences into a smaller policy.
- [AlphaZero](https://arxiv.org/abs/1712.01815) repeatedly trains policy/value
  heads from search-guided self-play, though its exact MCTS recipe is not
  presumed suitable here.

## Decision

The next neural baseline will learn candidate action ranking from fresh v2
search labels. It will not reuse historical weights or treat narrow final-score
regression accuracy as evidence of policy quality.

The dataset will preserve grouped candidates from one public state, teacher
means and uncertainty, immediate ranks, exact afterstates, seed provenance, and
teacher configuration. MLX will own the ranking model, loss, optimization, and
Apple GPU execution. Rust will own rules, fair search, collection, validation,
and gameplay.

Promotion requires:

- held-out top-1 accuracy and rank correlation;
- calibration against teacher value differences;
- inference latency and throughput measurements;
- paired complete-game strength against the policy it replaces; and
- a disjoint confirmation before any production use.

## Consequences

Search-quality labels are more expensive than outcome labels, so collection
must be resumable and checksummed. A student can initially match but not
magically exceed its teacher; stronger play must come from using the faster
student inside larger search or from repeated search-guided policy iteration.
