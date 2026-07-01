# Public Action Equivalence Foundation V1 Preregistration

Date: 2026-06-17

ADR: 0162

Experiment: `s7-public-action-equivalence-foundation-v1`

Protocol: `s7-exact-semantic-transition-v1`

Status: invalid; stopped before any accepted shard report

The source and executable were frozen correctly, but post-launch review found
that a serving-safe class of size `n` split into `k` exact-public subclasses
was recorded as `n - k` semantic collapses beyond exact identity. The correct
quantity is `k - 1`. All three processes were terminated before they emitted
shard reports. Protocol V2 supersedes this attempt without changing the
hypothesis, corpus, partition, or promotion thresholds.

## Question

Can differently serialized complete legal actions be collapsed into exact
rules-equivalent public transition classes without heuristic pruning?

## Correction To Prior Evidence

The S4 `equivalent_afterstate` relation is not the S7 target. Its source is an
R3 action-local token multiset with the market reset to the parent market.
The observed 17.73% train and 5.61% validation linked-candidate rates are
hypothesis-generating only. No S7 success claim may cite them as full-state
compression.

## Immutable Inputs

```text
ruleset: four-player AAAAA, no habitat bonuses
train decisions: 560
train actions: 2,135,111
validation decisions: 240
validation actions: 860,203
total decisions: 800
total actions: 2,995,314
```

Only open train and validation are permitted. Test, final, gameplay, hidden
future order, excluded tile identity, and future refill labels are forbidden.

## Exact Key Hierarchy

The census measures four nested notions:

1. `semantic-state-supply`: full public semantic afterstate plus exact semantic
   supply, intentionally ignoring hidden-effect order;
2. `serving-safe`: semantic state and supply plus ordered free replacement,
   paid wipes, draft slots, and wildlife-return behavior;
3. `exact-public-within-safe`: exact serialized `PublicGameState` inside each
   serving-safe class; and
4. `exact-hidden-successor-within-safe`: exact post-refill `GameState` inside
   each serving-safe class.

Only level 2 is the primary compression claim. Level 1 quantifies tempting
near matches that are explicitly rejected. Levels 3 and 4 diagnose whether
the remaining reduction comes from semantically irrelevant identity or from
strictly identical transitions.

## Required Checks

For every selected decision:

```text
graded PositionRecord == replayed parent
public-state hash == dataset
public supply == dataset
every canonical action hash == dataset
every action matched by grouped exact R3 enumeration
every R3 edit applies without truncation
```

For every serving-safe duplicate candidate:

```text
authoritative public record == R3-applied record
authoritative exact semantic supply == R3-applied supply
semantic post-refill successor == every class sibling
```

For every exact-public duplicate subclass:

```text
full hidden post-refill successor == every subclass sibling
```

All class accounting is exact. Every action is either a representative or a
member of exactly one class, and selected/champion action hashes remain
recoverable from their complete class.

## Adversarial Suite

Before production:

- exhaustively enumerate every default-prelude legal action in three seeded
  synthetic four-player states;
- inject one exact duplicate witness per state and prove exact public and
  hidden-successor parity;
- prove ordered paid-wipe traces differ when wipe order is reversed;
- verify unambiguous length-framed key material; and
- fail closed on malformed source digests or incomplete shards.

## Cluster Allocation

```text
john1: ADR 0161 exact-R2 control only
john2: S7 shard 0/3
john3: S7 shard 1/3
john4: S7 shard 2/3
```

Each S7 host replays both open splits but analyzes only its disjoint row
modulo. One immutable source digest and one identical release executable are
required across all three reports.

## Frozen Classification

```text
public_action_equivalence_invalid
public_action_equivalence_proof_only_futile
public_action_equivalence_promising
```

`promising` requires every validity check plus either:

```text
validation median serving-safe reduction >= 20,000 ppm
```

or:

```text
validation P90 serving-safe reduction >= 50,000 ppm
validation P90 absolute collapsed actions >= 128
```

Forward and reverse shard order must produce byte-identical scientific
classification.

## Predictions

1. The exact serving-safe rate will be much smaller than the S4 local-token
   relation.
2. Most rejected near matches will differ in market-prelude or draft trace.
3. Any surviving reduction will primarily quotient semantically identical
   standard tile IDs.
4. If median reduction is below 2%, S7 should remain a proof and diagnostic
   tool rather than enter the serving hot path.

## Invalidators

- production execution before this document is frozen;
- any test/final/gameplay access;
- class keys that omit ordered wipe or draft effects;
- action sampling rather than complete legal-set coverage;
- post-hoc promotion thresholds;
- duplicate primary shards;
- different source or executable identity across hosts; or
- order-dependent aggregation.

## Claim Boundary

This experiment can authorize an exact action-class serving design. It cannot
claim model-quality improvement, gameplay-score improvement, a new champion,
or achievement of the 100-point objective.
