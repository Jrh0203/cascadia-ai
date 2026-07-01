# ADR 0143: Exact Semantic Supply Foundation

Status: accepted and production census complete

Date: 2026-06-17

Research-plan item: S1

Experiment ID: `exact-semantic-supply-v1`

## Context

The existing public supply representation contains five wildlife-bag counts and
25 tile marginals:

- terrain capacity by terrain;
- wildlife eligibility capacity by species;
- keystone count by terrain; and
- dual-terrain count by unordered terrain pair.

Those values are useful summaries, but they do not identify the semantic
multiset of unseen tiles. Different pools can have the same marginals while
inducing different refill distributions and different compatibility with a
frontier. S1 requires the exact public distribution without reading the hidden
tile order or assigning strategy weights.

In a four-player game, all 85 standard habitat tiles are shuffled before the
83-tile stack is separated from two excluded tiles. Four market tiles are then
drawn. The public does not know either the stack order or which two tiles were
excluded.

## Decision

`cascadia-data` owns one canonical, factual semantic-supply representation for
standard multiplayer games.

### Canonical tile archetype

`CanonicalTileArchetype` contains every rule-relevant public fact attached to a
standard habitat tile:

- the two terrain identities, sorted into canonical order;
- the six directed edge terrains;
- the directed edge ring normalized to its lexicographically smallest rotation;
- the complete wildlife-eligibility mask; and
- keystone status.

The stable physical tile ID is not part of the archetype. It is a serialization
identifier, not a rule behavior. Physical multiplicity is retained separately
as `standard_tile_count`.

Catalog entries are sorted by canonical archetype bytes and assigned contiguous
`SemanticArchetypeId` values. The official 85-tile catalog produces 75 semantic
archetypes. Its frozen schema-v1 BLAKE3 is:

```text
362a1f090066f537fc29398fdc464f667b7e106889feff8a77607e35dd015c19
```

A tile reference also records the rotation mapping between the physical tile's
orientation and the canonical edge ring. This preserves exact directed-edge
meaning without making orientation part of archetype identity.

### Exact public counts

`ExactSemanticSupply::from_public_state` starts from official catalog
multiplicities and subtracts every publicly visible standard tile on:

- all player boards; and
- all current market slots.

It validates stable tile ID, official tile semantics, duplicate visible IDs,
archetype underflow, and total count conservation. The remaining count vector
is therefore the exact public multiset of hidden stack plus excluded tiles.

The supply separately records the number of physically drawable stack tiles.
For standard multiplayer, the initial stack contains `total_turns + 3` tiles:
four enter the opening market and one refill follows every turn except the
last. Subtracting publicly visible standard tile IDs therefore gives the exact
remaining drawable count. The difference between unseen and drawable counts is
the known number of still-hidden setup exclusions.

Wildlife-bag counts are reconstructed from the official 20 tokens per species
minus publicly visible placed and market wildlife. A staged afterstate in which
a drafted token is not placed correctly returns that token to the inferred bag.

The implementation supports standard multiplayer only. Solo play publicly
discards habitat tiles and wildlife, but `PublicGameState` does not retain that
discard history. Solo therefore fails closed instead of fabricating an exact
pool.

### Why excluded tiles remain in the refill belief

The setup performs one uniform shuffle before splitting stack and excluded
positions. Conditional on all public observations, every remaining physical
tile is exchangeable over every still-hidden position. The next stack position
is therefore uniform over the complete publicly unseen pool, including tiles
that might occupy an excluded position.

After a public draw, the same argument applies to the remaining hidden
positions. Thus the next `k` draws, for `k <= 4`, are an ordered sample without
replacement from the public unseen semantic multiset, provided `k` does not
exceed the public drawable count. No hidden stack order or excluded-set identity
is needed or permitted.

### Exact refill law

For archetype counts `c_i`, total unseen count `N`, and ordered archetype
sequence `a_1, ..., a_k`, the implementation returns:

```text
weight(a_1, ..., a_k)
  = product over j of
    (c_[a_j] - earlier occurrences of a_j)

P(a_1, ..., a_k) = weight(a_1, ..., a_k) / (N)_k
```

where `(N)_k = N * (N - 1) * ... * (N - k + 1)`.

For an unordered request with multiplicities `r_i`:

```text
P({r_i}) = product_i choose(c_i, r_i) / choose(N, k)
```

Probabilities are reduced exact integer fractions. The API supports:

- one-slot probabilities;
- ordered and unordered multi-slot queries;
- conditional distributions after an observed prefix;
- bounded exhaustive ordered-outcome enumeration; and
- horizons from one through four market slots.

### Deterministic identity

Schema v1 defines three canonical little-endian byte encodings:

- `CSSCAT1\0`: sorted catalog, archetypes, and physical multiplicities;
- `CSSSUP1\0`: catalog hash, wildlife counts, unseen total, drawable total, and
  archetype counts;
- `CSSRFL1\0`: horizon, catalog hash, and archetype counts.

All parsers reject unknown versions, wrong catalog identity, invalid lengths,
trailing bytes, impossible counts, and invalid horizons. BLAKE3 of canonical
bytes is the stable identity for catalogs, supplies, and refill laws.

### Consumer links

The foundation exposes factual links without introducing a strategy:

- each occupied market slot maps to an archetype and exact rotation reference;
- each frontier requirement stores the six public neighbor-facing terrains;
- compatibility reports matching-edge counts for all six rotations;
- a mask identifies rotations matching every present neighboring edge; and
- D6 transforms preserve compatibility covariantly.

These are compatibility primitives for future frontier, component, motif,
opponent, and planning consumers. S1 does not score or rank them.

### Census and distributed execution

`exact_semantic_supply_census` exports every public position of complete
four-player AAAAA games. Each position verifies:

- physical and semantic count conservation;
- drawable plus excluded conservation and feasible refill horizons;
- exact parity with the existing 30 public marginals;
- canonical serialization round trips;
- one- through four-slot refill identities and normalization;
- hidden redeterminization invariance;
- invariance under all 12 D6 transforms; and
- market-to-archetype links.

The exporter accepts only open train and validation splits. It uses deterministic
modulo ownership so separate hosts cannot overlap.

`s1_semantic_supply_queue.py` creates a reviewed-only 13-task graph:

- one immutable source/binary fanout;
- four disjoint train shards;
- four disjoint validation shards;
- one checksum-verified collection task;
- forward and reverse frozen merges; and
- one byte-identical merge-order proof.

The default corpus is 400 train games and 100 validation games, for 40,000
positions total. The graph is generated but is not installed unless `--apply`
is explicit.

## Rejected Alternatives

- Reading `tile_stack`, `excluded_tiles`, or their order would leak hidden
  information.
- Treating excluded tiles as known absent would overstate public knowledge.
- Keeping only the 25 tile marginals cannot recover exact refill semantics.
- Assigning hand-authored scarcity or compatibility weights would mix strategy
  into a factual foundation.
- Learning supply semantics in Python would duplicate authoritative game rules
  and permit Rust/MLX drift.
- Supporting solo without public discard history would make the word "exact"
  false.

## Consequences

The representation is larger than the legacy marginal vector, but remains tiny:
75 bounded integer tile counts, five wildlife counts, and exact unseen/drawable
totals. It is deterministic, fully local, cheap to derive, and suitable for
sparse, set, graph, or token consumers.

Schema changes now require a new version and catalog hash. A changed official
tile definition, canonicalization rule, or serialization order will fail the
pinned hash test and every source-frozen merge.

This ADR establishes semantic truth only. It does not claim a score gain. The
future learned S1 experiment must retain the 30-marginal representation as the
explicit control and measure retrieval and gameplay effects separately.

## Production Census Result

The source-frozen four-host census completed on 2026-06-17:

| Measurement | Result |
|---|---:|
| Games | 500 |
| Positions | 40,000 |
| Unique exact supply states | 40,000 |
| Train / validation positions | 32,000 / 8,000 |
| Accepted shards | 8 / 8 |
| Minimum / maximum unseen tiles | 2 / 81 |
| Minimum / maximum drawable tiles | 0 / 79 |
| Hidden setup exclusions | exactly 2 at every position |
| Forward/reverse merge identity | byte-identical |
| Scientific BLAKE3 | `44f2d8b6f6ab4d6f2f6920f2846ea4115693415f332b390a6f8f0cf4c45f589d` |

Classification:

```text
exact_semantic_supply_census_complete
```

The complete result is recorded in
[`exact-semantic-supply-v1-result.md`](../reports/exact-semantic-supply-v1-result.md).

## Verification

The accepted implementation includes:

- deterministic catalog and hash tests;
- public count and legacy-parity tests;
- exact drawable and setup-exclusion conservation through stack exhaustion;
- board-permutation and staged-afterstate tests;
- hidden-order and complete D6 tests;
- exact probability and conditional-law tests;
- exhaustive two-tile legacy-collision separation over the official inventory;
- market/frontier rotation and D6-covariance tests;
- fail-closed solo coverage;
- complete-game census tests; and
- queue, merger, tamper, gap, overlap, source-identity, and merge-order tests.

The preregistered operational contract is
[`exact-semantic-supply-v1-preregistration.md`](../reports/exact-semantic-supply-v1-preregistration.md).
