# Complete-Action Frontier Fit/Interference Audit V1 Preregistration

ADR 0102 freezes a four-host open-train diagnostic before another full model
treatment is allowed.

## Question

Does scale-16 expected-rank fit fail because individual groups cannot be
locally fit, because finite shared capacity collapses with group count,
because gradients from different groups conflict, or because capacity and
interference are both material?

## Frozen Campaign

- Experiment:
  `complete-action-frontier-fit-interference-audit-v1`.
- Data: the exact 560-group ADR 0101 train dataset and canonical scale-16
  cache.
- Model state: common seed `2026061630` or the exact selected ADR 0101
  checkpoint, as specified per arm.
- Objective: unchanged scale-16 expected-rank cross entropy.
- Optimizer: AdamW `1e-4`, weight decay `1e-4`.
- Selection: deterministic phase-by-width interleaved cohort.
- Domains: open train only; sealed test and gameplay remain closed.

## Four Independent Arms

| Host | Arm | Frozen work |
|---|---|---|
| john1 | nested fit scaling | 1/4/16/64 groups, 60 updates per group |
| john2 | capacity scaling | widths 96/192/288, same 32 groups and exposure |
| john3 | gradient conflict | exact 32-group gradients at init and selected model |
| john4 | empirical interference | independent vs shared adaptation on 24 groups |

Every fit comparison has equal optimizer updates and exact uniform rotation
exposure per included group. Duplicate training is zero.

## Decision Gates

- Local fit requires at least 95% recall and one exact set at size 1, at least
  90% recall and 75% exact sets at size 4, and at least 90% recall plus 75%
  exact sets under independent selected-checkpoint adaptation.
- Scaling collapse requires a size-64 loss of at least 15 recall points or 25
  exact-set points versus size 4, with size-64 recall below 80%.
- Capacity requires monotonic recall within two points and width 288 gains of
  at least 8 recall points and 10 exact-set points over width 192.
- Gradient interference requires at least 30% negative cosine to the sum of
  other gradients, median cosine at most -0.02, and at least 20% of
  off-diagonal pairs at cosine at most -0.10.
- Empirical interference requires independent adaptation to beat shared
  adaptation by at least 15 recall points and 25 exact-set points.

Pipeline invalidity has precedence. Otherwise the frozen outcomes are local
optimization/representation insufficiency, shared capacity bottleneck,
cross-group gradient interference, mixed capacity/interference, unresolved
shared scaling failure, or no material fit-scaling failure.

## Prohibitions

No second seed, extra width, changed optimizer, changed target scale, warm
start outside the specified selected-checkpoint arm, full 560-group trainer,
new teacher compute, validation-driven treatment selection, sealed test,
gameplay, cloud, Modal, or external compute.

Full algorithms, classification precedence, resource gates, source identity,
and throughput accounting are authoritative in ADR 0102.
