# ADR 0130: Exact D6 Transform Contract

Status: accepted

Date: 2026-06-16

Experiment ID: `exact-d6-transform-contract-v1`

## Context

The V2 rules engine previously had no authoritative full-dihedral transform.
Rotation-only augmentation existed outside the rules ownership boundary, while
reflections, tile orientation, complete actions, legal rows, hidden-state
preservation, and group composition had no shared executable contract.

That is insufficient for F3. A model-side transform can silently disagree with
the rules engine about edge order, reflected tile orientation, staged market
actions, or legal-row identity. A correct contract must transform complete
rules objects and prove transition equivariance before MLX consumes it.

## Decision

`cascadia-game` owns the exact D6 contract in `src/symmetry.rs`.

The group element is:

```text
T(k, f) = R^k S^f
R(q, r) = (q + r, -q)
S(q, r) = (q + r, -r)
```

Stable IDs are `0..5` for `R^0..R^5` and `6..11` for
`R^0S..R^5S`. `D6Transform::ALL` is in that order and identity has ID zero.
Composition is left action after right action:

```text
left.compose(right)(x) = left(right(x))
```

The canonical directed-edge order is `E, NE, NW, W, SW, SE`. Rotations map
edge `e` to `e + k`; reflected transforms map it to `k - e`, modulo six.
Opposite edges remain separated by three.

For dual-terrain tiles, orientation maps to `r + k` under rotations and
`k - r - 2` under reflected transforms, modulo six. Single-terrain and
keystone tiles always canonicalize to rotation zero. Board insertion also
enforces that canonical representation.

## Board And State Semantics

`Board::transformed` preserves tile identity, wildlife identity, occupied
insertion order, and nature-token count. It checks every transformed occupied
coordinate and every transformed frontier coordinate against the finite V2
backing grid. It never clips. The returned board's frontier must equal the
exact transformed source frontier.

`GameState::transformed` transforms every player board while preserving:

- player and board order;
- current player and completed turn count;
- game configuration, scoring cards, habitat-bonus rule, seed, and schema;
- market slots and contents;
- tile stack, wildlife bag, excluded and discarded resources in exact order;
- wildlife return counter.

`PublicGameState::transformed` follows the same public-state semantics.

`GameState::transform_turn_action` is state-aware. It stages the action's
`MarketPrelude`, resolves its `DraftChoice`, and uses the resolved tile to
transform orientation. Prelude flags, wipe slot order, draft slots, and token
semantics are unchanged. Placement and wildlife coordinates transform.
No bare action-only orientation API is authoritative.

## Legal-Row Contract

`LegalActionPermutation::new(state, prelude, transform)` regenerates both
legal sets from their respective states. It transforms each source action by
value and looks up that complete value in the transformed legal set.

The contract exposes:

- source-row to transformed-row `forward_rows`;
- transformed-row to source-row `inverse_rows`;
- explicit duplicate, cardinality, missing-value, and non-bijection errors;
- forward and inverse helpers for policy-like vectors;
- exact length validation.

Row indices and hashes are never copied from the source legal set.

## Metadata

`d6_contract_metadata()` returns a Serde-serializable schema containing:

- schema version and contract ID;
- stable transform IDs and names;
- edge order;
- 2x2 axial coordinate matrices;
- direction tables;
- dual- and single-terrain rotation tables;
- inverse table;
- composition table;
- scientific BLAKE3.

The V1 scientific hash is:

```text
db6ac2f9f6ebe2daaa2db603c6c16183512b5d989aed6979e1991e167737633f
```

The hash is BLAKE3 over deterministic Postcard serialization of every
scientific metadata field except the hash itself. Timestamps and paths are not
included.

## Rust-To-Python Binding

`crates/cascadia-game/src/bin/d6_contract_metadata.rs` is the production
serialization boundary. It calls `d6_contract_metadata()`, rejects any Rust
metadata whose scientific hash differs from the frozen V1 value, and emits
deterministic JSON. It can atomically regenerate or byte-check the permanent
artifact:

`python/cascadia_mlx/d6_contract_metadata.v1.json`.

`python/cascadia_mlx/d6_contract.py` loads that bundled artifact without
starting a subprocess. It validates the exact root schema, schema version,
contract ID, scientific hash, 12 stable transform descriptors, table
dimensions and ranges, coordinate-matrix uniqueness and unimodularity,
identity, inverse, composition, associativity, and the coordinate, direction,
dual-orientation, and single-orientation group actions.

`python/cascadia_mlx/hex_symmetry.py` contains no independent coordinate,
reflection, or orientation formulas. Scalar and MLX APIs select only from the
Rust-generated matrices and tables. The legacy `rotation_steps`,
`rotate_axial`, and `rotate_one_hot` APIs remain compatible as the stable-ID
`0..5` C6 subset. The D6 APIs also expose reflected coordinate and direction
transforms, inverse and composition IDs, dual-terrain orientation, and
single-terrain orientation canonicalized to zero.

## Regeneration And Drift Check

The artifact must never be hand-edited. Regenerate it only from the Rust
contract:

```bash
cargo run -p cascadia-game --bin d6_contract_metadata -- \
  --output python/cascadia_mlx/d6_contract_metadata.v1.json
```

CI and local verification must byte-check the checked-in artifact:

```bash
cargo run -p cascadia-game --bin d6_contract_metadata -- \
  --check python/cascadia_mlx/d6_contract_metadata.v1.json
```

The cross-language test also captures fresh exporter stdout and requires exact
byte equality with the bundled artifact before comparing every Python table.
Any Rust semantic drift, stale artifact, altered formatting, or hand edit
therefore fails CI.

## Required Proofs

The crate test suite must cover:

- 12 unique elements, stable IDs, identity, inverse, composition, and
  associativity;
- coordinate round trip, radius, distance, adjacency, edges, and opposites;
- tile-edge covariance for every tile orientation, edge, and transform;
- single-terrain canonicalization;
- board and complete-state round trips;
- exact frontier equality and explicit finite-grid overflow;
- habitat, wildlife, nature-token, and whole-game score invariance;
- legal-set bijection on initial and generated states;
- free three-of-a-kind and paid-wipe preludes;
- transition equivariance;
- policy permutation round trip, composition, and argmax identity;
- stable, serializable metadata and frozen scientific hash.

The cross-language suite must additionally cover:

- fresh Rust exporter bytes equal the bundled artifact;
- Python validates the complete generated schema and scientific hash;
- every coordinate in the radius-eight disk under all 12 transforms;
- every direction and tile rotation under all 12 transforms;
- all inverse and ordered composition-table entries;
- reflected MLX coordinate, direction, and orientation transforms;
- single-terrain orientation canonicalization;
- legacy C6 public API compatibility;
- explicit artifact-drift and malformed-metadata failures.

## Success Gates

F3 is complete only when:

- `cargo fmt --check` passes;
- `cargo test -p cascadia-game` passes;
- `cargo clippy -p cascadia-game --all-targets -- -D warnings` passes;
- the Rust artifact check passes;
- focused Python contract and existing symmetry-caller tests pass;
- Ruff format and lint pass for the F3-owned Python files;
- all required proofs above execute in Rust;
- Python/MLX consumes only the bundled Rust-generated contract for this shared
  symmetry module;
- no queue, sealed data, future hidden information, or external compute is
  used.

Any semantic change to IDs, formulas, edge order, tile rotation, composition
order, row mapping, or metadata payload requires a schema-version change, a
new scientific hash, and an ADR amendment.

## Consequences

All later D6 augmentation and exact group averaging can consume one rules-owned
contract. Model and data code may serialize or bind this metadata, but may not
redefine the geometry. The transform allocates only for copied state, frontier
verification, legal sets, row maps, and returned policy vectors; primitive
group operations are constant-time and table-backed where identity matters.
