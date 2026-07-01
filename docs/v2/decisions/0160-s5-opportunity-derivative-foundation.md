# ADR 0160: S5 Opportunity-Derivative Foundation

Status: completed; foundation passed

Date: 2026-06-17

Experiment: `s5-opportunity-derivative-foundation-v1`

Protocol: `s5-exact-counterfactual-derivative-census-v1`

Research-plan item: S5

## Context

Raw state features describe what exists. Strong drafting also requires the
counterfactual change caused by an action:

- score gained now;
- opportunities created or destroyed;
- habitat topology changed;
- future placements opened or consumed;
- supply compatibility changed; and
- selected market objects denied to opponents.

These quantities live on very different numerical scales. Adding them without
an exact factual definition and normalization contract would create another
opaque hand-engineered evaluator.

## Decision

For up to 64 deterministic complete legal actions at every position, derive
one exact 154-field `OpportunityDerivative` from independently constructed
before and authoritative after states.

### Immediate score

Twelve fields:

- five habitat deltas;
- five wildlife deltas;
- nature-token delta; and
- base-total delta.

### Habitat topology

For each terrain:

- component-count delta;
- internal-edge delta;
- open-boundary delta;
- cycle-rank delta;
- bridge delta;
- articulation delta; and
- merge-frontier delta.

### Wildlife motifs

Include eligible empty cells, Bear singleton/pair/oversize transitions, Elk
line and extension transitions, Salmon validity/endpoints/branch/continuation
transitions, Hawk conflicts/isolation/opportunities, and Fox
diversity/missing/compatible-cell transitions.

### Frontier and future placement

Include frontier count and degree, habitat bridge and repeated-contact
changes, resulting-size changes, and wildlife-specific lost/new future
placements.

### Supply and market competition

Include:

- wildlife-bag changes;
- selected tile archetype before and after;
- destination and frontier semantic-supply compatibility deltas;
- remaining tile/frontier match mass;
- remaining wildlife-slot mass;
- selected tile and wildlife opponent access; and
- total opponent market-access deltas.

No strategic coefficient or learned target is included.

## Sampling

The action set is sorted deterministically and sampled by stable hash to at
most 64 actions per position. The complete action count remains reported. In
the frozen AAAAA corpus every position has at least 64 actions, producing
exactly `positions * 64` derivatives.

## Normalization Contract

For every one of the 154 fields, report:

- observation count;
- nonzero count;
- minimum and maximum;
- P99 absolute value;
- maximum absolute value;
- transform; and
- divisor.

The transform is:

- identity for all-zero fields;
- robust division by `max(P99 absolute, 1)` normally; or
- signed `log1p` followed by robust division when the maximum exceeds 16
  times nonzero P99.

This contract is factual preprocessing, not learned weighting.

## Production Corpus

| Variable | Value |
|---|---:|
| Host | john4 |
| First seed | `5,410,000` |
| Games | 20 |
| Positions | 1,600 |
| Sampled actions | 102,400 |
| Feature observations | 15,769,600 |
| Rayon threads | 10 |

Calibration seed `5,400,000` is excluded from production evidence.

## Promotion Rule

Classify `s5_exact_opportunity_derivatives_promoted` only when:

- the feature schema contains exactly 154 fields;
- every field has one observation per sampled action;
- every divisor is at least one;
- every authoritative afterstate hash matches R3;
- every immediate score delta matches R3; and
- replay and score checks equal sampled actions.

## Consequences

A pass authorizes capacity-controlled learned ablations:

1. no derivatives;
2. immediate score only;
3. topology and motif derivatives;
4. supply and opponent-access derivatives;
5. the complete derivative vector.

A learned promotion still requires aggregate and protected-slice R4800
noninferiority, material ranking gain, and bounded serving cost.

## Claim Boundary

This ADR establishes exact counterfactual features and scaling only. It makes
no claim that the features improve ranking, gameplay, or the 100-point mean.

## Outcome

The production run classified S5 as
`s5_exact_opportunity_derivatives_promoted`.

- 1,600 positions exposed 2,682,245 complete actions;
- 102,400 deterministic actions were sampled;
- all 102,400 authoritative replay checks passed;
- all 102,400 immediate score-delta checks passed;
- all 154 schema fields received 102,400 observations and valid divisors;
- 150 fields were nonzero on the production corpus; and
- the raw P99 scale ratio was 1,218x, validating the need for explicit
  per-field normalization.

The four zero-observation fields remain schema-stable negative evidence:
frontier degree zero, Elk line bins zero and four, and six-edge destination
match. S5 advances to capacity-controlled learned ablations. See
`docs/v2/reports/s5-opportunity-derivative-foundation-v1-result.md`.
