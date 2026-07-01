# P1 Relational Hierarchical Pointer Foundation V1 Preregistration

Date: 2026-06-16

ADR: 0174

Experiment: `p1-relational-hierarchical-pointer-foundation-v1`

## Question

Can the oracle-proven complete-action hierarchy be expressed exactly as
selected-prefix pointers over the accepted sparse R2 state, without 441 dense
cells or flattened pointwise tile rows?

## Frozen Representation

```text
draft/prelude object
-> active-board R2 frontier token + rotation
-> occupied token, selected-prefix new tile, or no-placement
```

The active state is encoded once. Tile placement points to frontier objects.
Wildlife placement points into the exact post-tile destination set. Rust-owned
legality remains authoritative in the eventual serving path.

## Production Evidence

The complete train and validation factor caches are audited against the exact
R3/R2 parent cache. Each split runs on two distinct hosts from one
content-addressed source bundle.

The audit must cover all:

- 800 open groups;
- 2,995,314 complete source actions;
- draft, tile, and wildlife factor items;
- R3-retained cross-cache action hashes; and
- 12 D6 transform/inverse pairs for every spatial pointer.

## Success

- exact group and action coverage;
- zero factor-prefix hash mismatch;
- zero missing or ambiguous pointer;
- zero complete-action pointer collision;
- zero D6 failure;
- at most 121 exact sparse tokens on any active board;
- at most 20 draft objects;
- at most 31 frontier pointer objects; and
- at most 25 wildlife destination objects.

## Failure

Any mismatch blocks pointer training and requires a representation repair.
Post-hoc widening, clipping, coordinate fallback, dense-441 fallback, or
ignoring a failed action family is forbidden.

## Authorized Successor

Only `p1_relational_pointer_foundation_passed` authorizes one matched MLX
selected-prefix pointer pilot. The pilot must compare against the historical
hierarchical scorer on unchanged open targets and retain the P1 proposal gates:

- target recall greater than 98%;
- winner retention greater than 98%;
- mean proposals at most 1,024 initially;
- no phase or action-family guardrail failure; and
- complete-action rescoring before any gameplay claim.

## Claim Boundary

This foundation measures exact representability and serving support only.
Learned quality, latency, gameplay, and the 100-point target remain open.
