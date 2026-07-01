# ADR 0007: Promote Rules-Derived Opportunity Evaluation

Status: accepted on 2026-06-10.

## Context

Immediate-score greedy and shallow greedy-rollout search consistently
underbuilt Bear pairs. Adding Bear candidates alone merely transferred points
away from other wildlife. A policy needed to value setup without category
weights or hidden information.

`pattern-aware-v1` computes each retained action's exact post-action base score
and adds the expected best legal one-token marginal from four draws without
replacement over public unplaced wildlife supply. Immediate, habitat, and Bear
frontiers are computed in one shared legal-action pass.

## Evidence

Against greedy on 50 disjoint games, pattern-aware improved by 4.890 points
with 95% CI 4.296-5.484 and won all 50 games. Against promoted K8 on another
50-game suite, it scored 91.890 versus 90.775. The reported K8-minus-pattern
delta was -1.115 with 95% CI -1.696 to -0.534.

The shared-frontier implementation reproduced the original pilot exactly and
reduced runtime by 4.53x. In the product control it was 14.49x faster than K8.

## Decision

Promote `pattern-aware-v1-k8-h6-b8-m4` to the interactive API and web tier.
Keep exact greedy as the instant tier. Retain determinized K8/K16 and H6 as
research controls rather than production defaults.

Future search and self-play should use pattern-aware as the rollout blueprint.
MLX models still require independent held-out gameplay promotion.
