# Complete-Action Frontier Calibrated Monotone AdamW V1 Preregistration

ADR 0107 tests the single optimizer mechanism authorized by ADR 0106.

## Mechanism

- AdamW moments, epsilon, no bias correction, and weight decay remain frozen.
- Maximum rate:
  `2 * atanh(0.999) / 1200 = 0.006333668612083666`.
- Deterministic same-batch monotone backtracking halves rejected steps.
- At most 16 trials, minimum rate `1e-8`, loss tolerance `1e-12`.
- The identical implementation serves free residuals and full neural fit.

## Stages

1. Twenty-four free-residual groups, 1,200 updates each, with cross-host
   replay. Stop if aggregate recall is below 95% or exact sets below 75%.
2. Only after Stage 1 passes, four neural groups, 1,200 exposures each, with
   cross-host replay.

One MLX optimizer process runs per host. Dynamic one-group scheduling gives
origin work priority and uses replay tasks for backfill without a host barrier.

## Decision

Stage outcomes mechanically select optimizer insufficiency, public-observable
representation insufficiency, local budget insufficiency, local failure not
reproduced, or a confirmed local optimizer mechanism. Only the corresponding
single successor is authorized.

## Prohibitions

No optimizer sweep, second learning rate, changed weight decay, objective,
representation, group, budget, seed, full trainer, validation treatment,
sealed test, gameplay, cloud, Modal, or external compute.

ADR 0107 is authoritative for all constants, gates, stages, scheduling, and
maximum compute.
