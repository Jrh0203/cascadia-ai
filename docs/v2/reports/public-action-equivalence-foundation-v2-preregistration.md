# Public Action Equivalence Foundation V2 Preregistration

Date: 2026-06-17

ADR: 0163

Experiment: `s7-public-action-equivalence-foundation-v2`

Protocol: `s7-exact-semantic-transition-v2`

Status: frozen before V2 production census execution

## Question

Can differently serialized complete legal actions be collapsed into exact
rules-equivalent public transition classes without heuristic pruning?

V1 is scientifically invalid and supplies no answer. V2 changes only the
class-accounting implementation and its preproduction regression coverage.

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

1. `semantic-state-supply`: full public semantic afterstate plus exact semantic
   supply, intentionally ignoring hidden-effect order;
2. `serving-safe`: semantic state and supply plus ordered free replacement,
   paid wipes, draft slots, and wildlife-return behavior;
3. `exact-public-within-safe`: exact serialized public state inside each
   serving-safe class; and
4. `exact-hidden-successor-within-safe`: exact post-refill game state inside
   each serving-safe class.

Only level 2 is the primary compression claim. Level 1 records rejected near
matches. Levels 3 and 4 distinguish semantic identity from strict transition
identity.

## Exact Accounting Contract

For every decision and every hierarchy level:

```text
candidate count = sum(class sizes)
unique class count = number of classes
collapsed candidates = candidate count - unique class count
```

For each serving-safe class of size `n` containing `k` exact-public
subclasses:

```text
semantic collapses beyond exact-public identity = k - 1
```

The implementation rejects `n < 2`, `k = 0`, or `k > n` at this boundary.
The aggregate must equal the difference between serving-safe and exact-public
collapsed-candidate totals.

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

## Preproduction Gates

- formatting and all-target checks pass;
- all unit tests pass;
- clippy passes with warnings denied;
- the three-state adversarial transition suite passes;
- ordered paid-wipe traces remain distinct;
- malformed source digests fail closed;
- a real open-data smoke completes; and
- a production-accounting duplicate witness exercises `k - 1` accounting.

## Cluster Allocation

```text
john1: ADR 0161 exact-R2 control only
john2: V2 shard 0/3
john3: V2 shard 1/3
john4: V2 shard 2/3
```

Each host replays both open splits and analyzes only its disjoint row modulo.
One immutable source digest and one identical release executable are required.

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
4. If median reduction is below 2%, S7 remains a proof and diagnostic tool.

## Invalidators

- any V2 production execution before this document and bundle are frozen;
- any V1 shard treated as V2 evidence;
- test, final, or gameplay access;
- class keys omitting ordered wipe or draft effects;
- sampled rather than complete legal-set coverage;
- post-hoc promotion thresholds;
- duplicate primary shards;
- different source or executable identities across hosts; or
- order-dependent aggregation.

## Claim Boundary

V2 can authorize an exact action-class serving design. It cannot claim model
quality, gameplay score, a new champion, or achievement of the 100-point
objective.
