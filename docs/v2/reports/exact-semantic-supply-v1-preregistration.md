# Exact Semantic Supply V1 Preregistration

Date: 2026-06-17

Experiment ID: `exact-semantic-supply-v1`

Research-plan item: S1

Ruleset: four-player AAAAA, no habitat bonuses

Compute boundary: john1, john2, john3, and john4; local Apple Silicon only

Status: distributed source-frozen census complete

## Research Question

Can Cascadia V2 replace aliased public supply marginals with an exact,
legally observable semantic tile multiset and exact refill law, creating a
lossless foundation for later drafting, frontier, opponent, and planning
experiments?

The foundation question is factual. It does not ask whether a learned player is
stronger yet.

## Hypotheses

### H1: Exactness

The official standard catalog, public boards, and current market are sufficient
to reconstruct the exact public semantic multiset of all unseen habitat tiles.

### H2: Refill law

Because hidden stack and excluded positions are exchangeable after the setup
shuffle, exact one- through four-slot refill probabilities can be derived from
semantic counts alone, without inspecting future order.

### H3: Alias removal

The existing 25 tile marginals merge semantically different unseen pools. A
canonical archetype-count vector separates every such pair whenever the
semantic multisets differ.

### H4: Consumer compatibility

The same canonical archetypes can identify market tiles and compute exact
rotation-aware frontier terrain compatibility without hand-authored value
weights.

## Frozen Information Boundary

Permitted inputs:

- `GameConfig`, including standard multiplayer mode and ruleset;
- every public tile and placed wildlife on all boards;
- every visible market tile and wildlife token;
- official standard catalog definitions and multiplicities; and
- public afterstates before a hidden refill.

Forbidden inputs:

- hidden tile-stack order;
- hidden wildlife-bag order;
- identities of setup-excluded tiles;
- future refill realizations;
- game seed as a feature;
- future actions or terminal outcomes; and
- policy, value, teacher, or score labels.

The implementation must return the same supply and refill hashes after any
hidden redeterminization consistent with the public state.

Solo is outside this schema version because solo discard history is not present
in `PublicGameState`. It must fail closed.

## Frozen Semantic Schema

Schema ID: `canonical-public-tile-archetype-v1`

Each archetype contains:

| Field | Type | Meaning |
|---|---|---|
| primary terrain | enum | Lower canonical terrain identity |
| secondary terrain | optional enum | Higher canonical terrain identity |
| directed edges | six enums | Lexicographically minimum rotational edge ring |
| wildlife | five-bit mask | Complete eligibility set |
| keystone | boolean | Official keystone behavior |

Archetypes exclude physical tile ID and include physical multiplicity in the
catalog definition.

Frozen catalog facts:

```text
official physical tiles: 85
canonical semantic archetypes: 75
schema version: 1
catalog BLAKE3:
362a1f090066f537fc29398fdc464f667b7e106889feff8a77607e35dd015c19
```

Any catalog hash change invalidates all schema-v1 evidence until explained by a
new ADR and schema version.

## Frozen Supply Schema

Schema ID: `exact-semantic-supply-v1`

Each supply contains:

- five exact public wildlife-bag counts;
- one count for each of the 75 canonical tile archetypes;
- the conserved unseen tile total; and
- the exact number of physically drawable stack tiles;
- the exact number of hidden setup exclusions, derived as unseen minus
  drawable; and
- the exact catalog hash.

For standard multiplayer:

```text
unseen archetype counts
  = official catalog multiplicities
  - archetypes of every board tile
  - archetypes of every market tile
```

The implementation also validates physical standard tile IDs. A visible tile
cannot be counted twice or silently replaced by a different tile with similar
features.

For standard multiplayer:

```text
initial drawable inventory = total_turns + 3
remaining drawable tiles
  = initial drawable inventory - visible standard tile IDs
hidden setup exclusions = unseen tiles - drawable tiles
```

In the four-player ruleset, unseen counts run from 81 to 2 while drawable counts
run from 79 to 0. The two setup-excluded identities remain unknown, but their
count is public and constant.

The legacy 30-value `PublicSupply` must be exactly derivable from this richer
state at every settled public game position.

## Frozen Probability Contract

Let `c_i` be unseen count of archetype `i`, `N = sum_i c_i`, `D` the public
drawable count, and `k` the refill horizon. A refill law exists only when
`1 <= k <= min(4, D)`.

One slot:

```text
P(i) = c_i / N
```

Ordered sequence `a_1, ..., a_k`:

```text
P(a_1, ..., a_k)
  = product_j (c_[a_j] - previous occurrences of a_j)
    / (N * (N - 1) * ... * (N - k + 1))
```

Unordered multiplicities `r_i`:

```text
P({r_i}) = product_i choose(c_i, r_i) / choose(N, k)
```

All results use reduced integer fractions. Floating-point approximation is a
consumer convenience only and cannot define artifact identity.

The supported horizon is exactly `1..=4`. Requests beyond market width, beyond
the physically drawable stack, or with impossible prefixes fail closed. The
probability denominator still uses all `N` unseen tiles because excluded
identities are hidden and exchangeable with stack identities.

## Frozen Serialization Contract

Canonical encodings use:

| Object | Magic | Bound content |
|---|---|---|
| catalog | `CSSCAT1\0` | version, ordered definitions, multiplicities |
| supply | `CSSSUP1\0` | version, catalog hash, wildlife, unseen, drawable, counts |
| refill | `CSSRFL1\0` | version, horizon, catalog hash, counts |

Integers are little-endian. Catalog IDs are contiguous and canonical-byte
sorted. Parsers reject truncation, trailing bytes, impossible multiplicity,
unknown versions, and catalog mismatch.

The BLAKE3 of canonical bytes is the only supply or refill identity accepted by
the corpus merger.

## Frozen Collision Witness

The following two physical two-tile pools share the same legacy tile marginals:

```text
left:  tile 0  = Mountain / Hawk keystone
       tile 23 = River / Bear keystone

right: tile 2  = Mountain / Bear keystone
       tile 20 = River / Hawk keystone
```

They map to different archetype multisets:

```text
left archetypes:  [26, 72]
right archetypes: [24, 74]
```

This witness must remain in every census shard. In addition, the Rust suite
exhaustively enumerates every distinct two-tile physical pool from the official
85-tile inventory. Every pair sharing legacy marginals must either have the
same exact semantic multiset or different canonical supply bytes and hashes.

## Control And Treatment

### C0: Legacy marginals

The explicit control for future learned work is the current 30 values:

- five wildlife-bag counts; and
- 25 tile marginals.

C0 remains supported through `ExactSemanticSupply::to_legacy_public_supply`.

### T1: Exact semantic supply

T1 receives:

- the same five wildlife counts;
- 75 archetype counts;
- catalog identity; and
- exact market/frontier archetype links when the consumer requires them.

This foundation task implements C0 parity and T1 truth. It does not train a
model, assign hand weights, change search, or run gameplay qualification.

## Foundation Acceptance Gates

All gates are mandatory:

1. The official 85 tiles canonicalize to exactly 75 sorted archetypes.
2. Catalog construction is independent of source iteration order.
3. Swapping the two terrain fields of a dual tile does not change identity.
4. Every physical tile rotation maps exactly to and from canonical rotation.
5. The catalog hash equals the frozen schema-v1 hash.
6. Exact counts conserve the official physical inventory.
7. Drawable plus hidden-excluded counts exactly conserve unseen tiles.
8. A refill request beyond the remaining drawable stack fails closed.
9. Exact counts reproduce the legacy public marginals at every tested settled
   position.
10. Board order does not change exact supply.
11. Public afterstate staging preserves tile and drawable counts and exactly
    handles returned unplaced wildlife.
12. Every hidden redeterminization produces identical supply and refill hashes.
13. Every one of the 12 D6 transforms produces identical supply.
14. Frontier compatibility is D6-covariant for every standard tile and all six
    rotations.
15. One-slot probability mass is exactly one.
16. Exhaustively enumerated ordered support sums to the falling-factorial
    denominator for one through four slots.
17. Ordered, unordered, and conditional queries agree with exact combinatorics.
18. Supply and refill canonical bytes round-trip exactly and reject drift.
19. Every official two-tile legacy collision is separated when semantics differ.
20. Solo fails closed without public discard history.
21. Every census game emits exactly turns `0..79`.
22. The merger rejects byte tampering, refill-hash tampering, gaps, overlaps,
    source drift, executable drift, catalog drift, and noncanonical partitions.
23. Forward and reverse shard merge orders produce byte-identical reports.
24. The generated graph validates under the production research-queue schema.

Failure of gates 1 through 20 is an implementation defect, not a negative
research result.

## Open Census Corpus

The first distributed census is frozen as:

| Split | Game interval | Games | Positions | Strategy |
|---|---:|---:|---:|---|
| train | `[320000, 320400)` | 400 | 32,000 | `pattern-aware-v1` |
| validation | `[321000, 321100)` | 100 | 8,000 | `pattern-aware-v1` |

Each split uses four modulo shards:

```text
(game_index - first_game_index) % 4 == shard_index
```

Test and final split identifiers are rejected by the exporter. This foundation
census cannot consume sealed strength domains.

Every position records:

- game index, turn, and active player;
- canonical public-state hash;
- canonical semantic-supply bytes and hash;
- exact wildlife, archetype, unseen, drawable, and excluded counts;
- market archetype IDs; and
- exact refill-distribution hashes for all feasible horizons from one to four.

Every complete game contributes exactly 80 pre-decision public positions.

## Per-Position Runtime Audits

The exporter recomputes and fails on:

- semantic count conservation;
- drawable plus excluded conservation and stack-horizon validity;
- legacy marginal parity;
- supply serialization round trip;
- refill serialization round trip;
- refill probability normalization;
- one independent hidden redeterminization;
- refill-law invariance after redeterminization;
- all 12 D6 transforms; and
- market-link validity.

The report summary counts each passed audit. A count is not accepted merely
because the exporter says it passed; the merger reparses canonical bytes,
recomputes BLAKE3 values, validates count bounds, and recomputes all refill
hashes.

## Source-Frozen Cluster Plan

The queue generator creates 13 tasks:

| Wave | Tasks | Host allocation |
|---|---:|---|
| immutable bundle fanout | 1 | john1 coordinator |
| train census | 4 | one disjoint shard per host |
| validation census | 4 | one disjoint shard per host |
| checksum collection | 1 | john1 coordinator |
| deterministic merge | 2 | john1, forward and reverse |
| merge-order proof | 1 | john1 |

Train and validation tasks are pinned one per host and may not duplicate seed
ownership. The production queue permits only one running claim per host, so the
two waves remain work-conserving without oversubscribing a Mac.

The queue specification is reviewed-only by default. No task is installed until
the generated JSON, immutable bundle ID, source roots, seed intervals, and host
paths are reviewed and the command is repeated with `--apply`.

## Source-Frozen Build Commands

Build the census binary:

```bash
cargo build --release -p cascadia-data \
  --bin exact_semantic_supply_census
```

Create one content-addressed source and executable bundle:

```bash
.venv/bin/python tools/rust_experiment_bundle.py \
  --repository . \
  --experiment-id exact-semantic-supply-v1 \
  --include CASCADIA_V2_GOAL.txt \
  --include Cargo.lock \
  --include Cargo.toml \
  --include Makefile \
  --include pyproject.toml \
  --include uv.lock \
  --include apps/web/src \
  --include crates/cascadia-api \
  --include crates/cascadia-cli-v2 \
  --include crates/cascadia-data \
  --include crates/cascadia-differential \
  --include crates/cascadia-eval \
  --include crates/cascadia-game \
  --include crates/cascadia-model \
  --include crates/cascadia-provenance \
  --include crates/cascadia-search \
  --include crates/cascadia-sim \
  --include legacy/crates/cascadia-ai \
  --include legacy/crates/cascadia-core \
  --include python/cascadia_mlx \
  --include tools/s1_semantic_supply_merge.py \
  --binary target/release/exact_semantic_supply_census \
  --output-root artifacts/experiments/exact-semantic-supply-v1/bundles
```

Generate, but do not install, the queue specification:

```bash
.venv/bin/python tools/s1_semantic_supply_queue.py \
  --repository . \
  --bundle artifacts/experiments/exact-semantic-supply-v1/bundles/<bundle-id> \
  --output artifacts/experiments/exact-semantic-supply-v1/queue/generated.json
```

After review, install the exact same graph explicitly:

```bash
.venv/bin/python tools/s1_semantic_supply_queue.py \
  --repository . \
  --bundle artifacts/experiments/exact-semantic-supply-v1/bundles/<bundle-id> \
  --output artifacts/experiments/exact-semantic-supply-v1/queue/applied.json \
  --apply
```

## Local Verification Commands

```bash
cargo test -p cascadia-data semantic_supply --lib --no-fail-fast

cargo test -p cascadia-data \
  --bin exact_semantic_supply_census \
  --no-fail-fast

.venv/bin/pytest -q \
  tools/test_s1_semantic_supply_queue.py \
  tools/test_s1_semantic_supply_merge.py

.venv/bin/ruff check \
  tools/s1_semantic_supply_queue.py \
  tools/s1_semantic_supply_merge.py \
  tools/test_s1_semantic_supply_queue.py \
  tools/test_s1_semantic_supply_merge.py
```

## Smoke Receipt

A non-authoritative local one-game smoke run on 2026-06-17 emitted:

```text
game index: 399999
strategy: random-v1
positions: 80
unique supply states: 80
unseen tile range: 81 down to 2
drawable tile range: 79 down to 0
hidden excluded tiles: 2
all exact checks: 80 / 80
catalog BLAKE3:
362a1f090066f537fc29398fdc464f667b7e106889feff8a77607e35dd015c19
scientific BLAKE3:
ba93f24bccb250247fd02482b608dc1335e59da61bab5922c51079674387c82d
```

This receipt validates the local path only. It is not a substitute for the
source-frozen four-host corpus.

An additional source-frozen local graph smoke exercised all eight modulo shard
roles through the bundled release executable and frozen merger:

```text
bundle ID:
f3cbc25526e15bb71a9949633ce1ceef872c167df817a73027a052011c01a720
shards: 8
games: 8
positions: 640
unique supply states: 640
classification: exact_semantic_supply_census_complete
source BLAKE3:
a3494f04fa8f907201a57f2eccda638975a14b6828d8893b2ff55c6ed26867ff
executable BLAKE3:
09afc254d763e813b224e1ac8f4abdb4fababb6d2828eae1276de403f3e3f789
aggregate scientific BLAKE3:
45eee515aa52a1f8670da2327c6f3fd58ebb66fa98eb938886aac0e4d9b82c5e
forward/reverse merge: byte-identical
queue status: generated-not-applied
```

This is a tooling proof, not the preregistered 500-game scientific census.

## Foundation Success Classification

The distributed foundation advances only as:

```text
exact_semantic_supply_census_complete
```

That classification requires:

- all eight expected shard reports;
- exact train and validation interval coverage;
- one common source hash, executable hash, catalog, and collision witness;
- all canonical supply bytes and refill hashes independently validated;
- all 40,000 positions present;
- every exact audit passed; and
- byte-identical forward and reverse aggregate reports.

## Failure Classifications

- `invalid_semantic_catalog`: official count, order, field, or hash mismatch.
- `invalid_public_reconstruction`: count conservation or legacy parity fails.
- `hidden_information_leakage`: redeterminization changes supply or refill law.
- `invalid_d6_contract`: supply invariance or frontier covariance fails.
- `invalid_probability_law`: exact normalization or combinatorics fail.
- `invalid_serialization`: canonical bytes fail round trip or identity checks.
- `legacy_alias_not_separated`: different semantic multisets collide.
- `invalid_shard_partition`: game ownership is missing, overlapping, or wrong.
- `invalid_source_identity`: shards disagree on source or executable identity.
- `invalid_merge_determinism`: forward and reverse aggregates differ.

An invalid foundation is repaired and rerun under a new receipt. It is not
interpreted as evidence that marginal supply is sufficient.

## Future Learned S1 Experiment

Only after the foundation classification is complete may a learned S1 ablation
begin.

The control and treatment must use the same:

- compact spatial substrate;
- train and validation examples;
- target, optimizer, model capacity, and schedule;
- complete-action candidate set;
- search budget and random streams; and
- opponent population.

Primary offline outcomes:

- exact next-tile distribution decoding above 99.99%;
- conditional tile-stage top-64 target recall;
- complete-action confidence-set coverage;
- cross-entropy and calibration of refill predictions; and
- results split by low-supply and independent-draft states.

The exact foundation itself decodes the next-tile law at 100% by construction.
The 99.99% threshold applies to any learned consumer.

Gameplay begins only after offline sufficiency. It must use paired seeds,
balanced seats, equal search budget, score anatomy, and the common strength
gates in `RESEARCH_IMPLEMENTATION_PLAN_TO_100.md`.

No score gain is claimed by this preregistration.
