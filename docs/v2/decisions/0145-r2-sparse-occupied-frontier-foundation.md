# ADR 0145: R2 Sparse Occupied-Plus-Frontier Foundation

Status: accepted

Date: 2026-06-17

Experiment ID: `r2-sparse-occupied-frontier-foundation-v1`

Schema: `r2-sparse-public-token-state-v1`

## Context

R0 established that exact occupied entities are faster to construct than
materialized 61-, 91-, 127-, or 441-cell spatial supports. The accepted
60,000-position corpus contains only 51.5 occupied entities on average across
all four relative boards. Compact dense disks remained lossless only by
retaining exact overflow and still failed the preregistered state-build
throughput gate.

R2 therefore needs an exact public substrate that represents the entities and
affordances that exist. It must not choose another finite disk, clip unusual
legal boards, duplicate hidden state, or turn derived summaries into an
independent source of truth.

## Decision

The R2 V1 foundation is a standalone Rust crate at
`tools/r2_sparse_entity_census`. It opts out of the parent Cargo workspace and
depends on the existing `cascadia-data` and `cascadia-game` crates by path.
The foundation does not modify either dependency.

The authoritative packed state contains:

1. exact public global metadata;
2. exact per-player public metadata in relative-seat order;
3. exactly four public market slots;
4. one exact occupied-tile entity for every placed tile; and
5. an optional supplied tile archetype used only to derive frontier rotation
   compatibility.

The following layers are deterministic projections regenerated from the
authoritative payload after every decode and D6 transform:

- legal frontier tokens;
- exact habitat-component tokens; and
- minimal wildlife-motif anchor tokens.

Terminal score targets are neither serialized nor hashed.

## Coordinate Contract

Every occupied and frontier token carries a signed axial coordinate. The
representation has no radius, square, local window, overflow stream, or fixed
token capacity. Its packed format uses canonical signed variable-length
integers.

Input coordinates must still be valid values in the source
`compact-entity-v2` rules domain. This is source validation, not clipping.
Every accepted source coordinate is retained exactly, including legal
23-tile chains extending beyond radius 6.

## Occupied-Tile Tokens

Each occupied token retains:

- relative seat;
- signed axial coordinate;
- primary and optional secondary terrain;
- canonical tile rotation;
- all six directed edge terrains after rotation;
- wildlife eligibility mask;
- placed wildlife, if any; and
- keystone status.

Tokens are ordered strictly by `(relative_seat, q, r)`. Duplicate coordinates
and noncanonical source order fail closed.

The six directed edge terrains are derived from the exact game-owned
`Tile::terrain_on_edge` contract. They are present in the expanded token
layer, but are not redundantly serialized.

## Legal-Frontier Tokens

For every relative board, the legal frontier is the exact set of empty
neighbors of occupied tiles. Each token contains:

- relative seat and coordinate;
- six neighbor-presence bits;
- the six neighboring terrains facing the candidate cell;
- adjacent wildlife counts;
- the number of circular occupied-neighbor runs;
- opposite-neighbor pair bits;
- every touched habitat component, its exact size, and contact-edge bits;
- the exact resulting component size for each terrain if all matching
  contacts of that terrain are joined;
- terrain bits where placement can bridge distinct habitat components;
- terrain bits where one existing component is contacted on multiple edges;
  and
- optional supplied-tile rotation compatibility.

Habitat bridge and repeated-contact fields are exact local graph facts. They
are articulation-relevant, but are not predictions of future global value.

When a supplied tile is requested, every canonical rotation reports:

- matching directed-edge bits and count;
- whether every present neighboring edge matches; and
- exact touched component IDs and resulting habitat size by tile terrain.

`terrain_compatible_rotations` means rotations with at least one matching
habitat edge. Cascadia rules still permit every canonical rotation.

## Habitat-Component Tokens

Components are reconstructed separately for each relative seat and terrain.
Two tile members connect exactly when their facing directed edges carry the
same terrain.

Each component token contains:

- stable deterministic component ID;
- relative seat and terrain;
- sorted exact member coordinates;
- member count;
- matching internal edge count;
- open habitat-boundary edge count; and
- unique legal-frontier contact count.

IDs are assigned after sorting components by terrain and exact membership.
Frontier references use those IDs. The production union-find result must
equal an independent breadth-first graph oracle.

## Wildlife-Motif Boundary

V1 emits one exact motif anchor per placed wildlife:

- relative seat;
- coordinate and wildlife species;
- six directional neighboring wildlife values;
- adjacent wildlife multiset counts; and
- same-species neighbor bits.

This layer reconstructs the complete represented wildlife entity set exactly.
It is deliberately not described as a complete Card A scoring quotient.
Bear-pair, Elk-line, Salmon-path, Hawk-conflict, and Fox-diversity quotient
objects remain later learned-representation work.

## Public Metadata

Global metadata retains:

- game index;
- completed turn;
- perspective absolute seat;
- derived current absolute and relative seats;
- player count and total turns;
- scoring cards; and
- habitat-bonus mode.

Per-player metadata retains:

- relative and absolute seat;
- turns taken and turns until next action;
- occupied count;
- Nature Tokens;
- wildlife counts; and
- largest habitats.

Market tokens retain exact visible tile semantics and visible wildlife.
Hidden tile order, hidden wildlife order, future refills, future actions, and
terminal targets are absent.

## Validation

Extraction fails closed on:

- invalid player, turn, scoring-card, or inactive-seat metadata;
- board counts inconsistent with public turn order;
- more than 23 occupied tiles;
- invalid terrain, rotation, wildlife, mask, or keystone codes;
- noncanonical single-terrain rotation;
- impossible single- or dual-terrain semantics;
- placed wildlife outside the tile eligibility mask;
- coordinates outside the source rules domain;
- duplicate or noncanonical occupied coordinates;
- nonempty padding rows;
- disconnected boards;
- wildlife-count or largest-habitat metadata disagreement;
- Nature Tokens exceeding wildlife-bearing keystone opportunities;
- malformed or incomplete preterminal market slots;
- malformed packed magic, schema, flags, varints, ordering, or trailing bytes;
  and
- any frontier, component, motif, pack, reconstruction, or D6 oracle mismatch.

## D6 Contract

`D6Transform`, `transform_coord`, `transform_edge`, and
`transform_tile_rotation` from `cascadia-game` are authoritative.

Transforms preserve:

- relative seat;
- exact transformed coordinates;
- tile semantic composition;
- directed edge terrains under the transformed direction permutation;
- component and frontier semantics after deterministic regeneration; and
- wildlife identity.

Every transform must pass exact inverse reconstruction. No independently
invented coordinate or reflection convention exists in this tool.

## Packed Format

The canonical format is `CSR2SP1`:

- 8-byte magic `CSR2SP1\0`;
- 16-bit schema version;
- 16-bit flags;
- global public metadata;
- one player block per active relative seat;
- exact occupied rows grouped by relative seat;
- four raw validated market entities; and
- optional supplied-tile semantics.

Occupied coordinates use canonical zigzag LEB128 over signed 16-bit values.
Derived frontier, component, and motif layers are regenerated after decode.
Decode followed by encode must be byte-identical.

## Scientific Output

The census JSON contains a `scientific` payload and a BLAKE3 over compact JSON
serialization of that payload. Paths, timestamps, hostnames, timings, and
output locations are absent. Dataset identity uses content hashes and
manifested IDs, so byte-identical roots at different paths produce the same
scientific result.

## Consequences

This decision creates a mechanically exact substrate for R2 model work. It
does not select Set Transformer, graph, or Perceiver architecture; establish
MLX throughput; improve offline regret; or claim gameplay strength.

Any later model that truncates these tokens must be a separately named,
preregistered treatment. The foundation itself has no silent truncation path.
