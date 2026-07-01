# ADR 0149: Corrected Mid-Tail Activation Census

Status: accepted; production execution complete

Date: 2026-06-17

Experiment ID: `corrected-mid-tail-activation-census-v1`

Production authorization:

- immutable bundle:
  `6aabd4db4a88605779d8a8837865227a4af581feca3057ee36e6068631b4f368`;
- queue graph: 13 reviewed tasks in
  `artifacts/experiments/corrected-mid-tail-activation-census-v1/queue-spec.json`;
- allocation: shard `i` runs on `john(i+1)`, with no duplicated game index;
- launch date: 2026-06-17.

Production classification:
`corrected_mid_tail_activation_census_complete`.

All 301 corrected rows activated naturally across 81,920 representative
public states. The two count blocks reached 150/150 channels each, row
`11230` activated in 25,903 natural rows, and the separate reachable witness
also activated row `11230`. Forward and reverse aggregation were
byte-identical.

Parent experiment: `corrected-mid-tail-v1`

Research-plan item: F5 corrected-schema activation gate

## Context

ADR 0137 introduced the 11,231-row
`legacy-mid-v4-fixed-v1` schema:

| Range | Width | Meaning |
|---|---:|---|
| `0..10561` | 10,561 | Frozen historical v2 base |
| `10561..10930` | 369 | Migrated opponent detail |
| `10930..11080` | 150 | Extended tile-bag terrain counts |
| `11080..11230` | 150 | Extended tile-bag wildlife-capacity counts |
| `11230..11231` | 1 | Overflow-used-this-turn |

ADR 0144 then proved exact C0/T1 prediction parity on the immutable
200,000-row historical activation corpus. That corpus cannot establish
corrected-tail activation. Its records contain only sparse features emitted by
the historical `mid-features,v4-opp` extractor, whose 301-row tail had
different semantics and was inactive by construction.

The next F5 gate needs new records generated through the actual corrected Rust
feature gate. Reconstructing the rows from V2 marginals or documentation in
Python would not be authoritative. The evidence must also separate naturally
observed state frequencies from an adversarial overflow witness; blending a
fixture into natural-play statistics would overstate representativeness.

## Decision

Add the standalone crate:

```text
tools/f5_corrected_tail_activation_census
```

Its direct dependency is:

```toml
cascadia-ai = {
  path = "../../legacy/crates/cascadia-ai",
  features = ["legacy-mid-v4-fixed-v1"]
}
```

The crate is intentionally its own Cargo workspace. It requires no root
workspace, production binary, queue, dashboard, status, roadmap, or plan edit.

The command surface is:

```text
generate-shard  generate one immutable public-state corpus shard
census-shard    replay and census one shard through the actual Rust extractor
aggregate       combine exactly four disjoint shard reports
verify-order    require two aggregate files to be byte-identical
```

## Corrected-Record Corpus

The production corpus is frozen as:

| Field | Value |
|---|---:|
| Dataset ID | `corrected-mid-tail-public-state-corpus-v1` |
| Games | 1,024 |
| Players | 4 |
| Decisions per game | 80 |
| Representative rows | 81,920 |
| Shards | 4 |
| Games per shard | 256 |
| Ruleset | AAAAA |
| Score or habitat-bonus labels | excluded |

Game ownership is:

```text
(game_index - first_game_index) mod 4
```

with `first_game_index=0`. Shards therefore have no shared game, state, or
record. Each shard stores its exact sorted game-index list and a content hash
of that list. Aggregation requires the union to equal `0..1024` exactly once.

## Public-State Boundary

Every representative record is captured immediately before the active
player's draft, after applying the legal free three-of-a-kind replacement
once when available.

The serialized state contains only:

- absolute focal seat, personal turn, phase, and turns remaining;
- all four public boards;
- tile placement order, coordinates, rotation, terrain, allowed wildlife,
  keystone status, placed wildlife, and starter-tile status;
- public nature-token counts;
- the four ordered visible market pairs;
- public tile-supply marginals;
- wildlife remaining as defined by the historical `BagInfo` public-state
  calculation;
- whether the free overflow replacement was applied this turn; and
- stable record, source, and feature receipts.

The record schema contains no field for:

- tile-bag order;
- wildlife-bag order;
- future refill realization;
- future action;
- score, terminal target, or habitat-bonus label;
- teacher output; or
- raw RNG seed.

The raw seed is replaced by a one-way commitment. Deterministic seed domains
and game indices remain documented for reproducibility.

`TrajectoryPolicy` is experimental corpus provenance, not a public game fact.
It is stored in `RecordProvenance`, outside `PublicState`, and therefore does
not enter the per-record public-state BLAKE3. It remains part of the
record-identity stream and continues to define the policy coverage slices.

The pre-launch record, manifest, shard-report, and aggregate artifact schema
version is `2`. Version `1` smoke artifacts are invalid after this boundary
correction and cannot be mixed with version `2`.

## Public Tile-Supply Proof

The corrected count rows depend on tile-bag composition. The corpus does not
serialize hidden bag order.

Instead, the Rust reader reconstructs the exact remaining tile multiset from:

```text
official 85-tile multiset
- every public non-starter tile on all four boards
- every visible market tile
```

Starter tiles are explicitly marked and excluded because they do not come
from the 85-tile draft bag.

The reader derives:

- five terrain counts;
- five wildlife-capacity counts;
- the joint terrain-by-wildlife counts; and
- remaining tile count.

Generation fails if that public reconstruction differs from
`GameState.tile_bag.feature_distributions()` or the live tile-bag length.
Replay fails if conservation, multiplicity, or any public marginal differs.

## Actual Extractor Replay

Generation and census both call:

```text
BagInfo::from_game_for_player
extract_features_with_bag
```

from `legacy/crates/cascadia-ai/src/nnue.rs`, compiled with
`legacy-mid-v4-fixed-v1`.

For replay, all boards and the market are rebuilt from the public record.
`BagInfo::from_game_for_player` computes the public base and opponent context.
The publicly reconstructed tile-supply fields replace only the temporary
template game's irrelevant bag values.

Each record freezes:

- public-state BLAKE3;
- raw extractor emission count;
- normalized sparse feature count;
- normalized full-feature BLAKE3;
- exact corrected-tail feature list; and
- corrected-tail BLAKE3.

`census-shard` recomputes every value. Any mismatch is an invalid corpus, not
a negative research result.

## Representative Trajectories

The policy assignment is:

```text
(game_index + absolute_seat) mod 4
```

with:

1. greedy;
2. random draft plus greedy placement;
3. scarcity draft plus greedy placement; and
4. sampled wildlife preference plus greedy placement.

Every game assigns each policy to exactly one seat. Across 1,024 games this
gives exactly 20,480 rows per policy and 20,480 rows per seat.

Frozen phase totals are:

| Phase | Rows |
|---|---:|
| Opening | 4,096 |
| Early | 16,384 |
| Middle | 32,768 |
| Late | 28,672 |

The policy mixture is not a strength benchmark. It broadens public state and
supply coverage without adding teacher compute or hidden information.

## Separate Overflow Witness

Natural trajectories are the only source of representativeness statistics.

Shard zero additionally stores one separately labeled
`reachable_overflow_witness`. It is the first deterministic setup seed, under
a distinct frozen domain, whose opening market legally contains a
three-of-a-kind. The real `GameState::replace_overflow` transition is applied,
the resulting public state is recorded, and the corrected extractor must emit
row `11230`.

The witness:

- is reachable through the authoritative legacy rules engine;
- uses the actual corrected extractor;
- has independent state and feature hashes;
- is excluded from game, row, phase, seat, policy, channel-frequency, and
  natural overflow-rate totals; and
- cannot rescue a representative corpus in which natural overflow never
  occurs.

Production success requires both natural activation of row `11230` and the
separate witness.

## Activation Report

Every corrected row receives:

- absolute feature index;
- block;
- terrain or wildlife owner and count bin when applicable;
- total representative activations;
- phase slice;
- absolute focal-seat slice;
- trajectory-policy slice; and
- overflow-used/not-used slice.

Block summaries report active channels and activation totals for:

- `10930..11080`;
- `11080..11230`; and
- `11230..11231`.

The fixture never contributes to these totals.

## Source Freeze

Each manifest content-hashes:

- this crate's Cargo manifest, lockfile, build script, source, and tests;
- this ADR and the preregistration;
- the complete legacy `cascadia-ai` source tree and Cargo manifest; and
- the complete legacy `cascadia-core` source tree and Cargo manifest.

The sorted file list, byte sizes, per-file BLAKE3 values, source-bundle BLAKE3,
extractor-file BLAKE3, and Git revision are recorded.

`build.rs` resolves the Git object ID at compile time from the checkout or the
explicit `F5_SOURCE_GIT_REVISION` build environment. A missing or malformed
revision is a build error. The binary embeds that revision, so runtime source
identity never depends on a `.git` directory and never records
`git_revision=unavailable`.

Generation captures the source identity before and after writing a shard and
fails if it changes. Census requires the current source identity to match the
manifest exactly.

Build products, `target`, `.git`, and `__pycache__` are excluded.

The integration suite copies the complete immutable source-root set without
`.git`, runs the compiled binary from that bundle for generation and census,
and requires the embedded revision to survive exactly.

## Strict JSON Boundary

Every scientific JSON reader first parses through one reusable recursive
visitor that rejects duplicate object keys at any nesting depth and rejects
trailing JSON values. Only then is the duplicate-free value deserialized into
the typed artifact.

The boundary applies to:

- corpus manifests;
- every JSONL public-state record;
- the separate overflow-witness record;
- shard reports; and
- aggregate reports.

Duplicate keys are foundation-invalid even when the attacker recomputes outer
file or scientific hashes. Ordinary `serde_json` last-key-wins parsing is not
used at a scientific ingress.

## Aggregation

Aggregation fails without writing a passing report when:

- the report count is not exactly four;
- shard indices are duplicated, missing, or out of range;
- game indices overlap, have gaps, or leave `0..1024`;
- any scientific hash does not recompute;
- source, extractor, schema, corpus, or shard contracts disagree;
- any record or payload is malformed or corrupt;
- any scientific JSON object contains a duplicate key at any depth;
- phase, seat, policy, or row totals drift;
- the separately labeled witness is absent, duplicated, or blended into
  representative counts; or
- channel metadata differs across shards.

Inputs are sorted by shard index before merging. Aggregate operational input
paths are also sorted. Forward and reverse invocations must therefore produce
byte-identical JSON.

## Success Classification

Use:

```text
corrected_mid_tail_activation_census_complete
```

only when all of the following hold:

- four valid source-identical production shards;
- 1,024 games and 81,920 representative rows exactly once;
- exact phase, seat, and policy totals;
- both natural overflow slices are nonempty;
- all 150 terrain-count channels activate naturally;
- all 150 wildlife-capacity channels activate naturally;
- row `11230` activates naturally;
- the separate authoritative overflow witness activates row `11230`; and
- every scientific input has a content receipt.

A valid production corpus that misses any activation gate is classified:

```text
corrected_mid_tail_activation_census_incomplete
```

It is a negative activation result and does not authorize F5 fine-tuning.

Malformed inputs, duplicate JSON keys, source drift, extractor mismatch,
overlap, gap, or corrupt hashes are foundation-invalid and produce no
research classification.

## Consequences

This closes the implementation gap between exact migration parity and
corrected-tail learning experiments.

The implementation does not launch production, fine-tune a network, mutate
the cluster queue, change the dashboard, or make a score claim. A completed
production activation census authorizes only the next preregistered offline
F5 training gate.
