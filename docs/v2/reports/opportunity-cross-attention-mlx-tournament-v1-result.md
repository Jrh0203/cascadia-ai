# Exact-R2 Opportunity Query Factorial Result

- Experiment: `opportunity-cross-attention-mlx-tournament-v1`
- Decision: ADR 0166, with repairs in ADR 0168, ADR 0172, ADR 0173, and ADR 0177
- Classification: `opportunity_query_factorial_null`
- Selected arm: none
- Aggregate ID: `cd6431f6546c8ae8d4c8a2e6581506ad9c18dc28ad0d8ae63dcd7a051e13b884`

## Result

Candidate-conditioned attention over exact semantic supply and exact frontier
memory did not produce a promotable successor to the parent-conditioned
Exact-R2 control.

All four matched 2,000-step arms completed over the same open panel of 240
complete decisions and 860,203 actions. Forward and reverse aggregation were
byte-identical. No arm changed the champion, authorized paired gameplay, or
established progress toward the 100-point target.

## Arm Evidence

| Arm | Global top-64 recall | Strategic top-64 recall | Top-64 regret | P99 | Peak RSS | Eligible |
|---|---:|---:|---:|---:|---:|---|
| C0 parent query | 0.7250 | 0.6995 | 0.1087 | 4,360.8 ms | 3.94 GB | No |
| T1 supply query | 0.7250 | 0.6995 | 0.1087 | 4,311.9 ms | 4.55 GB | No |
| T2 frontier query | 0.7292 | 0.6995 | 0.1019 | 4,360.8 ms | 4.64 GB | No |
| T3 combined query | 0.7250 | 0.6995 | 0.1063 | 4,362.3 ms | 4.62 GB | No |

T2 carried the only directional signal: global recall increased by 0.00417 and
regret fell by 0.00674 against the trained parent. The bootstrap probability of
lower regret was 0.9508. This did not transfer to strategic recall, whose delta
and probability of improvement were both zero, and the global-recall
probability of improvement was only 0.6334.

Every arm failed the frozen 250 ms absolute latency gate. Every treatment also
exceeded the 4 GiB RSS gate. The result is therefore null on both decision
quality and deployability, not merely blocked by one conservative threshold.

## Interpretation

The exact opportunity features remain valid information, but this
candidate-query adapter did not convert them into a useful complete-action
ranking advantage at the tested capacity and optimization budget. The slight
T2 frontier effect is too small, strategically flat, and far too expensive to
justify another search-time qualification.

The next experiments should change the learning target or proposal structure,
not repeat this cross-attention geometry:

1. distributional opportunity supervision over shared-seed counterfactual
   continuations;
2. exact sparse hierarchical pointer proposals with selected-prefix state; and
3. only after an offline gate passes, complete-action gameplay qualification.

## Artifacts

- Aggregate: `artifacts/experiments/opportunity-cross-attention-mlx-tournament-v1/launches/adr0172-relaunch-v1/aggregate-forward.json`
- Order proof: `artifacts/experiments/opportunity-cross-attention-mlx-tournament-v1/launches/adr0172-relaunch-v1/order-proof.json`
- Production collection: `artifacts/experiments/opportunity-cross-attention-mlx-tournament-v1/launches/adr0172-relaunch-v1/reports/production-collection.json`
- Classifier repair: `docs/v2/decisions/0177-opportunity-checkpoint-transport-classifier-repair.md`
