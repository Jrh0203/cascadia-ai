# Counterfactual Public-Value Target Audit

ADR 0074 tested whether repeated complete H6 continuations create a stable and
useful public-state expected-return target.

## Protocol

- two fresh validation games, indices 65,000-65,001;
- all 80 pre-action public states per game;
- sixteen domain-separated hidden redeterminations per state;
- frozen H6 K8/H6/R4/D4 source and continuation policy;
- 160 states and 2,560 complete terminal continuations;
- no model training, test split, gameplay, retry, or threshold change.

## Estimator Stability

| Metric | Result | Gate | Pass |
|---|---:|---:|---|
| Mean R16 standard error | 0.508 | <=1.50 | yes |
| R8 MAE to R16 | 0.487 | <=1.25 | yes |
| R8 P90 drift to R16 | 1.131 | descriptive | |
| R8 pairwise accuracy | 91.14% | >=70% | yes |
| R8 pairwise log loss | 0.583 | below R1 | yes |
| R1 pairwise log loss | 0.751 | baseline | |
| Projected 256-game R8 corpus | 13.86 h | <=24 h | yes |

R8 is sufficient for this continuation policy. Its error falls monotonically
with phase: mean drift is 0.856 points on personal turns 1-5, 0.497 on 6-10,
0.391 on 11-15, and 0.205 on 16-20.

## Target Width

| Metric | Result | Gate | Pass |
|---|---:|---:|---|
| R16 state-mean standard deviation | 1.945 | >=2.0 | no |
| Mean within-state standard deviation | 2.032 | descriptive | |
| Factual trajectory MAE to R16 | 1.335 | descriptive | |
| Factual trajectory correlation to R16 | 0.685 | descriptive | |

The repeated sampler removes substantial realization noise, but the resulting
absolute expected totals remain narrowly distributed. The only failed gate is
the one intended to prevent another value model from learning mostly phase
and game-level offsets.

## Decision

Rejected as an absolute state-value training target. No 256-game corpus or
model training is authorized.

The counterfactual sampler is qualified: it is deterministic, public-safe,
checksummed, fast enough at R8, and preserves all raw terminal samples. The
next experiment should apply it to multiple candidate afterstates from the
same decision and learn centered counterfactual advantage. That formulation
cancels the compressed game-level offset instead of trying to regress it.

## Artifacts

- raw dataset:
  `artifacts/datasets/counterfactual-public-value-audit-v1-validation`;
- machine-readable audit:
  `docs/v2/reports/counterfactual-public-value-target-audit-v1.json`;
- implementation smoke:
  `docs/v2/reports/counterfactual-public-value-target-audit-v1-implementation-smoke.json`.
