# ADR 0110: Frontier Neural Minimum-Rate Forensic

Status: completed as `public_observable_representation_insufficient`; one
public-observable representation treatment authorized.

Date: 2026-06-16

Experiment ID: `complete-action-frontier-neural-minimum-rate-forensic-v1`

## Context

ADR 0109 closed as `calibrated_optimizer_pipeline_invalid` because neural
group 2 failed the numerical-completion rule after eight accepted updates.
The origin and cross-host replay were bit-identical. The retained report says:

- accepted updates: 8;
- total accepted-step backtracks: 13;
- maximum accepted-step backtracks: 4;
- maximum accepted rate:
  `0.006333668612083666`;
- minimum accepted rate:
  `0.00002474089301595182`;
- mean accepted rate:
  `0.0009154130415902173`;
- nonfinite rejections: 0;
- moments and scores finite; and
- failure: `monotone AdamW could not accept an update`.

The runtime did not retain the failed step's convergence diagnostic. This ADR
uses only frozen bookkeeping and source logic to identify the rejected
subcondition. It performs no model execution or training.

## Frozen Evidence

- ADR 0109 combined report BLAKE3:
  `f269476798c1df773b71d58599da593213b24b041caba528d5bc46d4f19a5b32`.
- ADR 0109 source bundle BLAKE3:
  `8f5bd5ee0e85952f0a3d486fc348243c9ad42c922c16331a7e056944ff461580`.
- ADR 0109 group 2 origin/replay scientific BLAKE3:
  `1d6ee91568ecadd3eece723fa2b4e059960def1bc18db1929e3634061b935839`.
- Frozen AdamW rate recurrence, 16 trials, `0.5` backtrack factor, `2.0`
  regrowth, `1e-8` minimum accepted rate, `1e-7` convergence threshold, and
  `1e-12` loss tolerance.

No neural group may rerun. No source, threshold, optimizer, model, objective,
representation, or metric may change.

## Audit

Enumerate every accepted-step backtrack sequence consistent with all retained
group 2 summary statistics. For each sequence, reconstruct every accepted
rate, the next failed-step starting rate, and all 16 attempted rates.

The audit proves `minimum_rate_completion_conflict` only when:

- at least one and no more than 1,000 consistent sequences exist;
- every consistent sequence has the same failed-step starting rate;
- every consistent sequence attempts a rate below `1e-7`;
- zero nonfinite rejections plus the frozen source path prove all failed-step
  proposals were finite;
- the generic failure message proves current loss was finite;
- finite proposals, moments, and scores prove current parameters, next
  moments, and direction were finite; and
- therefore the only remaining failed convergence condition is a candidate
  improvement greater than `1e-12` at a rate below the frozen `1e-8`
  acceptance floor.

## Domain-Consistent Reclassification

The optimizer cannot accept a proposal below `1e-8`. Such a diagnostic
proposal must not invalidate numerical convergence in the optimizer's
eligible update domain.

If the audit proves `minimum_rate_completion_conflict`, reclassify frozen group
2 as numerically converged after eight accepted updates without changing its
model or metrics. Recombine the four frozen ADR 0109 terminal reports.

Mechanical outcomes:

1. `neural_minimum_rate_forensic_invalid`
   - any identity, enumeration, uniqueness, finite-state, or logical proof
     gate fails.
2. `public_observable_representation_insufficient`
   - the domain-consistent pipeline passes but terminal recall is below 90% or
     exact sets below 75%.
3. `local_failure_not_reproduced`
   - the corrected pipeline and terminal strength both pass.

The 120-exposure checkpoint remains unobserved and may not be fabricated.
Only `public_observable_representation_insufficient` authorizes one
public-observable representation treatment. No outcome authorizes a full
trainer directly.

## Maximum Compute

One deterministic CPU audit, focused/full tests, one report, and source/evidence
hash checks. No MLX model execution, neural training, group replay, threshold
change, optimizer treatment, full trainer, validation treatment, sealed test,
gameplay, cloud, Modal, or external compute.

## Result

The deterministic audit found six accepted-rate histories consistent with
every frozen group 2 summary statistic. All six produce:

- failed-step starting rate `9.896357206380728e-5`; and
- smallest of 16 attempted rates `3.020128541986306e-9`.

The rate threshold therefore passed. The frozen report and source prove all
proposals, current loss, parameters, moments, and direction were finite, and
that every eligible finite loss-nonincreasing proposal would have been
accepted. The only remaining failed condition was a greater-than-`1e-12`
candidate improvement below the optimizer's frozen `1e-8` acceptance floor.

That below-floor diagnostic proposal is outside the optimizer's eligible
update domain. Group 2 is therefore domain-consistently reclassified as
numerically converged after eight accepted updates without changing its model
or metrics.

The corrected four-group pipeline passes. Terminal recall remains 32.39% with
zero exact sets; the 120-exposure checkpoint remains unobserved. The mechanical
classification is `public_observable_representation_insufficient`.

No MLX model execution, gradients, training, replay, cloud, or external
compute was used. One separately preregistered public-observable
representation treatment is authorized. A full trainer is not.

Machine-readable result:
`artifacts/experiments/complete-action-frontier-neural-minimum-rate-forensic-v1/reports/combined.json`.

Human-readable result:
`docs/v2/reports/complete-action-frontier-neural-minimum-rate-forensic-v1-result.md`.
