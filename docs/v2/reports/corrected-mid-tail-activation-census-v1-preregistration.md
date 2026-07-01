# Corrected Mid-Tail Activation Census V1 Preregistration

Date: 2026-06-17

Experiment ID: `corrected-mid-tail-activation-census-v1`

Parent experiment: `corrected-mid-tail-v1`

Schema: `legacy-mid-v4-fixed-v1`

Status: completed; all production gates passed

Authorization is limited to the reviewed 13-task queue graph in
`artifacts/experiments/corrected-mid-tail-activation-census-v1/queue-spec.json`
and immutable bundle
`6aabd4db4a88605779d8a8837865227a4af581feca3057ee36e6068631b4f368`.
No training task is authorized by this census.

Result:
`corrected_mid_tail_activation_census_complete`.

The frozen four-host corpus contained 1,024 games and 81,920 representative
rows. Natural coverage was 150/150 terrain-count channels, 150/150
wildlife-capacity channels, and 1/1 overflow channel. The overflow row was
active in 25,903 natural records, and the separately labeled reachable
witness independently activated the same row.

## Question

Does a deterministic, source-frozen corpus of representative public
four-player AAAAA states naturally exercise every corrected historical
mid-tail channel, including the overflow-used bit, when features are emitted
by the actual `legacy-mid-v4-fixed-v1` Rust extractor?

## Hypotheses

H1. Every representative row emits exactly five terrain-count rows and five
wildlife-capacity rows.

H2. Across 1,024 mixed-policy games, every bin in both 150-row count blocks is
observed at least once.

H3. Natural legal free-overflow replacements occur in the representative
corpus, so row `11230` activates without synthetic augmentation.

H4. A separately labeled, authoritative reachable-overflow witness also
activates row `11230`, independently proving the extractor's boolean path.

H5. Four modulo-owned shards aggregate without overlap, gap, host dependence,
or input-order dependence.

## Frozen Implementation

Manifest:

```text
tools/f5_corrected_tail_activation_census/Cargo.toml
```

Required dependency feature:

```text
cascadia-ai/legacy-mid-v4-fixed-v1
```

The implementation may call only the actual Rust feature path. No Python
feature formula, V2 marginal inference, corrected-index reconstruction, or
tolerance-based feature comparison is permitted.

Artifact schema version: `2`.

Version `1` smoke artifacts are invalidated by the pre-launch correction that
moves experimental trajectory policy out of `PublicState` and into
`RecordProvenance`.

## Frozen Corpus

| Field | Value |
|---|---:|
| Dataset ID | `corrected-mid-tail-public-state-corpus-v1` |
| First game index | 0 |
| Total games | 1,024 |
| Players | 4 |
| Decisions per game | 80 |
| Representative rows | 81,920 |
| Shards | 4 |
| Games per shard | 256 |
| Scoring cards | AAAAA |
| Habitat bonus labels | excluded |
| Score/teacher/terminal labels | excluded |

Each shard owns:

```text
(game_index - 0) mod 4 == shard_index
```

The union must be exactly `0..1024`.

## State Timing

Capture occurs:

```text
after a legal free three-of-a-kind replacement, if available
before the active player's draft
```

Paid wildlife mulligans are not part of the corpus generator.

The overflow flag therefore means:

```text
the free replacement was actually applied in this represented turn
```

## Public Information Contract

Allowed:

- public boards and tile placement order;
- public tile and wildlife attributes;
- public nature tokens;
- ordered visible market;
- current player, turn, and phase;
- public-inferable remaining tile multiset marginals;
- public-inferable wildlife remaining under historical `BagInfo` semantics;
- overflow-used-this-turn;
- source, state, payload, and feature content receipts.

Forbidden:

- tile-bag order;
- wildlife-bag order;
- future refill or action;
- terminal score;
- habitat bonus;
- teacher output;
- target or selected-action label;
- raw RNG seed.

The remaining tile multiset is reconstructed from the official 85-tile
catalog minus non-starter board tiles and market tiles. Generation requires
that reconstruction to equal the live legacy tile bag exactly.

Trajectory policy is experimental provenance, not public game state. It is
serialized only in `RecordProvenance`, is excluded from `public_state_blake3`,
and remains included in record-identity hashing and policy-slice accounting.

## Deterministic Trajectory Mix

Policy assignment:

```text
(game_index + absolute_seat) mod 4
```

Policies:

| ID | Policy |
|---:|---|
| 0 | greedy |
| 1 | random draft, greedy placement |
| 2 | scarcity draft, greedy placement |
| 3 | sampled wildlife preference, greedy placement |

Every game has one seat per policy. Expected aggregate totals:

| Slice | Rows |
|---|---:|
| Each focal seat | 20,480 |
| Each policy | 20,480 |
| Opening | 4,096 |
| Early | 16,384 |
| Middle | 32,768 |
| Late | 28,672 |

## Separate Authoritative Overflow Fixture

Shard zero includes one extra file, not one extra representative row.

The fixture is produced by:

1. searching deterministic seeds under the frozen witness domain in ascending
   counter order;
2. choosing the first opening state with a legal three-of-a-kind market;
3. applying `GameState::replace_overflow`;
4. serializing only the resulting public state; and
5. calling the corrected Rust extractor.

It must activate `11230`.

The fixture is excluded from:

- 81,920 representative rows;
- phase, seat, and policy totals;
- natural overflow counts;
- all per-channel activation frequencies; and
- count-bin coverage.

Natural representative activation of `11230` remains an independent success
gate.

## Per-Channel Measurements

For all 301 rows report:

- feature index;
- block;
- semantic owner;
- bin;
- activation count;
- phase counts;
- focal-seat counts;
- policy counts; and
- overflow-used/not-used counts.

For each block report:

- declared range and channel count;
- active channel count;
- total activations; and
- representative rows with any activation.

## Content Identities

Every shard manifest must contain:

- exact source file list, byte sizes, and BLAKE3 values;
- source-bundle BLAKE3;
- `nnue.rs` BLAKE3;
- nonempty build-time embedded Git object ID;
- exact game-index list and BLAKE3;
- record payload byte size and BLAKE3;
- public-state stream BLAKE3;
- normalized full-feature stream BLAKE3;
- corrected-tail stream BLAKE3;
- representative record-identity BLAKE3; and
- separate overflow-witness payload and semantic receipts on shard zero.

Operational host, path, elapsed time, and executable identity do not enter the
scientific BLAKE3.

All manifest, JSONL record, witness, shard-report, and aggregate readers must
use the shared strict recursive JSON boundary. Duplicate object keys at any
depth and trailing JSON values are invalid before typed deserialization.
Recomputed outer hashes do not make a duplicate-key artifact admissible.

The production binary must not discover Git metadata at runtime. `build.rs`
must embed a 40-64 character hexadecimal revision resolved from the build
checkout or `F5_SOURCE_GIT_REVISION`; unresolved revisions fail the build.
Bundle-style generation and census from a source tree with no `.git`
directory are required verification gates.

## Production Gates

Classification
`corrected_mid_tail_activation_census_complete` requires:

1. four valid and source-identical reports;
2. exact shard IDs `0,1,2,3`;
3. exact game IDs `0..1024` once each;
4. 81,920 representative rows;
5. exact phase, seat, and policy totals;
6. nonempty natural overflow-used and not-used slices;
7. 150/150 active terrain-count channels;
8. 150/150 active wildlife-capacity channels;
9. natural activation of row `11230`;
10. separate authoritative witness activation of row `11230`;
11. exact actual-extractor replay for every record; and
12. byte-identical forward/reverse aggregate output.

The fixture cannot satisfy gates 6 or 9.

## Failure Classes

Input-invalid, no scientific classification:

- `invalid_source_drift`
- `invalid_schema_or_extractor`
- `invalid_public_state_replay`
- `invalid_payload_corruption`
- `invalid_duplicate_json_key`
- `invalid_shard_overlap`
- `invalid_shard_gap`
- `invalid_order_dependence`
- `invalid_fixture_blending`

Valid negative activation result:

```text
corrected_mid_tail_activation_census_incomplete
```

This applies when the corpus is valid but one or more corrected channels do
not activate naturally.

## Four-Host Allocation

| Host | Shard |
|---|---:|
| john1 | 0 |
| john2 | 1 |
| john3 | 2 |
| john4 | 3 |

All four generators and censuses are independent and may run concurrently.
No host repeats another host's game index.

Suggested production roots:

```text
artifacts/experiments/corrected-mid-tail-activation-census-v1/corpus/shard-0
artifacts/experiments/corrected-mid-tail-activation-census-v1/corpus/shard-1
artifacts/experiments/corrected-mid-tail-activation-census-v1/corpus/shard-2
artifacts/experiments/corrected-mid-tail-activation-census-v1/corpus/shard-3
```

## Exact Commands

Build and verify:

```bash
cargo test \
  --manifest-path tools/f5_corrected_tail_activation_census/Cargo.toml

cargo clippy \
  --manifest-path tools/f5_corrected_tail_activation_census/Cargo.toml \
  --all-targets --no-deps -- -D warnings

cargo fmt \
  --manifest-path tools/f5_corrected_tail_activation_census/Cargo.toml \
  -- --check
```

Generate one production shard:

```bash
cargo run --release \
  --manifest-path tools/f5_corrected_tail_activation_census/Cargo.toml -- \
  generate-shard \
  --output-root \
    artifacts/experiments/corrected-mid-tail-activation-census-v1/corpus/shard-0 \
  --shard-index 0 \
  --shard-count 4 \
  --first-game-index 0 \
  --total-games 1024
```

Census that shard:

```bash
cargo run --release \
  --manifest-path tools/f5_corrected_tail_activation_census/Cargo.toml -- \
  census-shard \
  --corpus-root \
    artifacts/experiments/corrected-mid-tail-activation-census-v1/corpus/shard-0 \
  --output \
    artifacts/experiments/corrected-mid-tail-activation-census-v1/reports/shard-0.json
```

Aggregate forward:

```bash
cargo run --release \
  --manifest-path tools/f5_corrected_tail_activation_census/Cargo.toml -- \
  aggregate \
  --report artifacts/experiments/corrected-mid-tail-activation-census-v1/reports/shard-0.json \
  --report artifacts/experiments/corrected-mid-tail-activation-census-v1/reports/shard-1.json \
  --report artifacts/experiments/corrected-mid-tail-activation-census-v1/reports/shard-2.json \
  --report artifacts/experiments/corrected-mid-tail-activation-census-v1/reports/shard-3.json \
  --require-shards 4 \
  --output artifacts/experiments/corrected-mid-tail-activation-census-v1/reports/aggregate-forward.json
```

Run the same command with report arguments reversed and write
`aggregate-reverse.json`, then require:

```bash
cargo run --release \
  --manifest-path tools/f5_corrected_tail_activation_census/Cargo.toml -- \
  verify-order \
  --left artifacts/experiments/corrected-mid-tail-activation-census-v1/reports/aggregate-forward.json \
  --right artifacts/experiments/corrected-mid-tail-activation-census-v1/reports/aggregate-reverse.json
```

## Authorization Boundary

This preregistration and implementation do not authorize production launch.
They do not mutate the shared queue or dashboard.

A reviewed production aggregate may authorize the next F5 offline
fine-tuning gate. It does not authorize gameplay or a score claim by itself.
