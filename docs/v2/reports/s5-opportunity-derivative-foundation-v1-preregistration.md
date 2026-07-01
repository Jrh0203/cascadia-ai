# S5 Opportunity-Derivative Foundation V1 Preregistration

Date: 2026-06-17

ADR: 0160

Experiment: `s5-opportunity-derivative-foundation-v1`

Protocol: `s5-exact-counterfactual-derivative-census-v1`

Status: frozen before production

## Question

Can complete actions be represented by an exact, target-free 154-field
counterfactual derivative with authoritative replay and a complete robust
normalization contract?

## Frozen Corpus

```text
host: john4
first seed: 5,410,000
games: 20
positions: 1,600
sampled actions per position: 64
sampled actions: 102,400
rayon threads: 10
rules: four-player AAAAA, no habitat bonus
```

Seed `5,400,000` was used only for implementation calibration and is excluded.

## Frozen Feature Families

- immediate score anatomy;
- habitat topology;
- Bear, Elk, Salmon, Hawk, and Fox motif opportunities;
- frontier affordance;
- lost and newly opened future placements;
- semantic supply compatibility and remaining mass; and
- selected-object and total opponent market access.

## Frozen Gates

```text
feature field count == 154
feature scale count for every field == sampled actions
normalization divisor for every field >= 1
exact replay failures == 0
score delta failures == 0
exact replay checks == sampled actions
score delta checks == sampled actions
```

The raw P99 scale ratio is reported diagnostically and is not a promotion
threshold.

## Frozen Normalization

- all-zero field: identity, divisor one;
- ordinary field: divide by `max(P99 absolute, 1)`;
- heavy-tail field: signed `log1p`, then robust division when
  `max absolute > 16 * P99 absolute`.

## Predictions

1. All 102,400 sampled actions will replay exactly.
2. Immediate score deltas will match R3 exactly.
3. Raw feature scales will span multiple orders of magnitude.
4. The normalization contract will provide one explicit transform and
   divisor for every field.

## Invalidators

- source bundle or executable mismatch;
- fewer than 64 samples at any normal production position;
- target-, score-label-, or teacher-dependent feature construction;
- missing feature fields;
- scientific hash mismatch; or
- changing the normalization rule after launch.

## Claim Boundary

A pass authorizes learned derivative ablations. It does not prove that the
features improve rankings or gameplay.
