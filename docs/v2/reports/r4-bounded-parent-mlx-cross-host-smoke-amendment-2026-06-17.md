# ADR 0156 Cross-Host Smoke Amendment

Date: 2026-06-17

Experiment: `r4-bounded-quotient-mlx-comparison-v1`

Protocol: `r4-bounded-parent-mlx-matched-comparison-v1`

## Decision

Before production authorization, run one bounded ten-step numerical-parity
smoke for `q3-affordance-parent` on john1 and john4.

Q3 is the registered passing quotient with the richest active parent payload,
so it is the strictest of the three bounded arms for testing cache
materialization, universal-class projection, optimizer parity, and checkpoint
serialization. Passing Q3 does not substitute for the four production arms.

## Frozen Smoke Contract

- hosts: john1 and john4;
- arm: `q3-affordance-parent`;
- steps: 10;
- seed: `2026061710`;
- groups per step: 4;
- optimizer and objective: exactly the ADR 0156 production definitions;
- parent cache: one content-addressed prefix with at least 80 train and 80
  validation groups, exported once and checksum-fanned out;
- R3 candidate cache and S1 exact-supply cache: the accepted complete caches;
- no production authorization and no resume;
- source bundle, model graph, batch hashes, candidate counts, initialization,
  action identities, and checkpoint tensor names must agree.

## Numerical Gates

```text
loss maximum absolute difference <= 1e-4
loss maximum relative difference <= 1e-5
checkpoint parameter maximum absolute difference <= 1e-4
checkpoint parameter mean absolute difference <= 1e-6
prediction score maximum absolute difference <= 1e-4
prediction standard-error maximum absolute difference <= 1e-5
stable prediction-panel ranking identical
```

Every check must pass. The content-addressed proof classification is
`r4_bounded_parent_mlx_cross_host_smoke_pass`. Any malformed report, identity
drift, missing checkpoint, batch mismatch, or tolerance failure is invalid and
blocks authorization.

## Claim Boundary

The smoke establishes only cross-host numerical reproducibility for the frozen
training path. It does not measure final offline quality, serving efficiency,
gameplay strength, or progress toward the 100-point target.
