# Exact D6 Transform Contract V1 Preregistration

Date: 2026-06-16

Experiment ID: `exact-d6-transform-contract-v1`

ADR 0130 freezes the first production full-dihedral transform owned by
`cascadia-game`. This is an F3 correctness contract, not a gameplay-strength
experiment and not an authorization to launch model training.

## Frozen Semantics

- transforms: `T(k,f)=R^k S^f`;
- rotation: `R(q,r)=(q+r,-q)`;
- reflection: `S(q,r)=(q+r,-r)`;
- stable IDs: rotations `0..5`, reflected transforms `6..11`;
- edge order: `E, NE, NW, W, SW, SE`;
- rotated edge: `e+k mod 6`;
- reflected edge: `k-e mod 6`;
- dual-terrain rotation: `r+k` or `k-r-2 mod 6`;
- single-terrain rotation: always zero;
- composition: left transform after right transform.

The implementation domain is complete V2 `Board`, `PublicGameState`,
`GameState`, staged `TurnAction`, and legal-action row permutations. Market
slots and all non-geometric state retain identity. Hidden tile and wildlife
orders are preserved exactly and never inspected to construct model features.

## Frozen Validation Domain

The Rust suite includes:

- every group element and every ordered composition pair;
- every ordered associativity triple;
- a complete radius-8 coordinate disk;
- every direction, opposite direction, tile rotation, tile edge, and D6
  transform;
- a mixed dual-terrain, wildlife, keystone, and nature-token board;
- all scoring-card variants;
- deterministic initial and generated game states;
- a free three-of-a-kind prelude;
- a paid wildlife-wipe prelude;
- generated legal sets and complete transitions;
- policy vectors with unique argmax values;
- an adversarial coordinate that exceeds the finite backing grid after
  transformation.

The cross-language suite includes:

- a fresh invocation of the production Rust exporter;
- byte equality between fresh output and the permanent bundled artifact;
- strict Python schema and frozen-hash validation;
- every radius-eight coordinate under every transform;
- every direction and tile rotation under every transform;
- all inverse and ordered composition entries;
- reflected scalar and MLX transforms;
- dual- and single-terrain orientation semantics;
- legacy C6 API compatibility;
- stale-artifact and malformed-artifact rejection.

No corpus, queue task, cloud host, Python augmentation, sealed split, or
future refill information is part of this preregistration.

## Required Assertions

1. All 12 elements have unique stable IDs.
2. Identity, inverse, composition tables, and associativity are exact.
3. Coordinates round trip and preserve radius, distance, and adjacency.
4. Direction and opposite-edge relations transform exactly.
5. Tile-edge covariance holds for every rotation, edge, and transform.
6. Single-terrain and keystone orientations canonicalize to zero.
7. Boards and states round trip without changing resource identity or order.
8. Transformed frontiers equal the exact image of source frontiers.
9. Unsupported transformed coordinates return errors; no clipping occurs.
10. Habitat analysis and all score components are invariant.
11. Legal sets have equal cardinality and exact value-based bijections.
12. `T(apply(a)) = apply(T(a))` for ordinary, free-replacement, and paid-wipe
    actions.
13. Policy permutations round trip, compose according to the group table, and
    move argmax to the mapped legal row.
14. Metadata serializes with schema version one and scientific BLAKE3
    `db6ac2f9f6ebe2daaa2db603c6c16183512b5d989aed6979e1991e167737633f`.
15. Fresh Rust exporter output is byte-identical to the bundled Python
    artifact.
16. Python validates all generated tables and uses them for scalar and MLX
    C6/D6 transforms without owning separate formulas.
17. Every radius-eight coordinate, direction, and orientation agrees for all
    12 transforms, including inverse and composition behavior.
18. Single-terrain orientations canonicalize to zero and the legacy C6 API
    retains its prior outputs and padding semantics.
19. Artifact drift and malformed metadata fail closed.

## Success Classification

Classify `exact_d6_transform_contract_complete` only if:

- every assertion above passes in `cascadia-game`;
- format, test, and warning-denying clippy gates pass;
- action transforms resolve the post-prelude draft tile from state;
- legal rows are regenerated and matched by complete action value;
- full state transformation preserves hidden order and rule identity;
- the serializable metadata tables match executable behavior;
- the production artifact checker and cross-language Python tests pass;
- the final classification is recorded only after Rust, Python, and Ruff
  gates complete.

Otherwise classify `exact_d6_transform_contract_incomplete`. A partial
rotation-only layer, Python fallback, stale-row mapping, silent clipping, or
state transform that changes hidden order is a hard failure.

The experiment ledger begins in `planned` state for dashboard visibility. No
queue launch is authorized by this document.
