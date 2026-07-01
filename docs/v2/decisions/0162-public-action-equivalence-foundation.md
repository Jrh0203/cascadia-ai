# ADR 0162: Exact Public-Action Equivalence Foundation

Status: superseded by ADR 0163; V1 invalid before accepted evidence

Date: 2026-06-17

Experiment: `s7-public-action-equivalence-foundation-v1`

Protocol: `s7-exact-semantic-transition-v1`

V1 was stopped on 2026-06-17 before any shard report was produced. A review
found incorrect accounting for semantic collapses beyond exact-public
identity. See ADR 0163 and the V1 invalid-attempt report.

Research-plan item: S7

## Context

S4 reported a substantial `equivalent_afterstate` relation, including 5.61%
of validation candidates linked to a 256-anchor collision. That label was too
broad for action elimination. The underlying R3 cache field is the BLAKE3 of
an action-local token multiset after restoring the parent market, not the
canonical full public successor. It is useful relational context, but it is
not proof that two complete legal actions have the same transition.

The S7 hypothesis remains worthwhile because Cascadia contains repeated
semantic tile archetypes and actions whose serialized identities can differ
without changing rules-relevant state. Any serving optimization must also
account for market-prelude side effects. Paid wildlife wipes draw and return
hidden tokens in order, and drafting or returning a wildlife token changes
the hidden bag transition. A state-only collision is therefore insufficient.

## Decision

Build a separate Rust census that never modifies the frozen ADR 0161
tournament crate or binary.

For every open graded-oracle action, reconstruct the authoritative
`TurnAction`, then use the accepted exact R3 apply path to derive:

- the complete semantic public afterstate `PositionRecord`;
- the exact semantic tile and wildlife supply;
- the ordered free-replacement and paid-wipe trace;
- the exact draft kind and market slots; and
- whether the drafted wildlife is placed or returned.

The primary serving-safe key is:

```text
full semantic public afterstate
+ exact semantic supply
+ ordered hidden-effect trace
```

`TileId` is intentionally quotiented only after the exact semantic supply has
been included. Tile identity has no scoring or rule behavior, while terrain,
edge orientation, wildlife support, keystone status, market slot, and supply
multiplicity remain exact.

For every primary key containing more than one action, reconstruct each
authoritative public afterstate and full successor. The census must prove:

1. exact R3 record parity;
2. exact semantic-supply parity;
3. semantic post-refill successor parity for the whole primary class; and
4. full hidden-successor parity inside every exact serialized-public-state
   subclass.

Actions with the same semantic state and supply but different ordered traces
are recorded as rejected near matches, never collapsed.

## Corpus And Distribution

Only the immutable open train and validation graded-oracle datasets are
allowed:

```text
train:      560 decisions, 2,135,111 complete legal actions
validation: 240 decisions,   860,203 complete legal actions
```

The test and final splits, gameplay qualification artifacts, future labels,
and hidden teacher state are forbidden.

The production census uses three disjoint modulo shards:

| Host | Shard |
|---|---|
| `john2` | split row modulo 3 equals 0 |
| `john3` | split row modulo 3 equals 1 |
| `john4` | split row modulo 3 equals 2 |

`john1` remains dedicated to the active ADR 0161 exact-R2 control. S7 is
dependency-independent backfill and may not delay tournament closeout.

Every remote uses one frozen source bundle and one identical release binary.
Forward and reverse shard aggregation must produce byte-identical scientific
reports.

## Frozen Success And Futility Gates

Evidence is valid only if:

- the adversarial suite passes;
- all 800 open decisions appear exactly once;
- all 2,995,314 legal actions are accounted for;
- invariant failures are zero;
- every primary duplicate class has semantic successor parity; and
- every exact-public duplicate subclass has full hidden-successor parity.

S7 is promising enough for a serving design only if validation satisfies
either:

```text
median complete-legal-set reduction >= 2%
```

or both:

```text
P90 complete-legal-set reduction >= 5%
P90 absolute collapsed actions >= 128
```

Otherwise the result is `public_action_equivalence_proof_only_futile`. The
proof tool remains useful, but serving is not complicated for immaterial
compression.

## Consequences

A promising result authorizes a separate serving implementation with
representative selection and exact action-hash expansion. It does not itself
change candidate quality, select a model, qualify gameplay, or claim progress
toward a 100-point mean.

A futile result closes exact public-action equivalence as a material
performance lever and prevents the earlier local-token collision rate from
being misreported as available legal-set compression.
