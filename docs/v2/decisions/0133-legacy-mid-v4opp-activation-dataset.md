# ADR 0133: Legacy Mid-V4Opp Activation Dataset

Status: accepted

Date: 2026-06-16

Dataset ID: `legacy-mid-v4opp-activation-v1`

## Context

ADR 0129 freezes the `legacy-mid-v4opp-11231` schema and its activation
census, but the repository did not contain a production input stream for that
historical representation. Reconstructing the 11,231 columns in Python or
from documentation would be scientifically invalid: the champion build has a
known historical layout defect in `10561..10862`, and its opponent slots are
emitted in the exact order implemented by the old Rust `BagInfo`.

The missing artifact must therefore call the historical Rust extractor
directly, preserve its emitted semantics, expose every focal seat and phase,
and be reproducible without teacher rollouts, sealed data, gameplay scoring,
cloud execution, or external compute.

## Decision

Add the standalone `legacy_activation_export` Rust binary under
`cascadia-differential`. It is compiled with the existing `legacy-teacher`
feature, whose `cascadia-ai` dependency enables exactly
`mid-features,v4-opp`.

The exporter fails closed unless all of these runtime constants match:

```text
NUM_FEATURES          = 11231
NUM_FEATURES_MID      = 10862
NUM_FEATURES_MID_V4   = 11231
OPP_DETAILED_BASE     = 10862
```

Every row calls:

```text
BagInfo::from_game_for_player
extract_features_with_bag
```

from `legacy/crates/cascadia-ai/src/nnue.rs`. The exporter does not reproduce,
translate, or approximate a feature formula.

## Frozen Corpus

The production corpus is:

| Field | Value |
|---|---:|
| Split | `train` |
| Players | 4 |
| Scoring cards | AAAAA |
| Games | 2,500 |
| Decisions per game | 80 |
| Rows | 200,000 |
| Shards | 10 |
| Games per shard | 250 |
| Rows per shard | 20,000 |

The 200,000-row size is intentionally below the 2.995 million complete-action
candidate corpus. It still gives a channel at the ADR 0129 rare threshold of
`1e-4` an expected support boundary of 20 observations, while retaining exact
coverage of all 20 personal turns, all four phases, and all four focal seats
in every game.

No score, terminal result, habitat bonus, teacher value, selected label,
expected rank, rollout statistic, or sealed split enters the dataset.

## State Timing

Rows are observed immediately before the active player's draft. The free
three-of-a-kind market replacement is applied first when legal, matching the
historical production prelude. Paid wildlife wipes are not performed.

`personal_turn` is `1..20`. The phase mapping is the ADR 0129 mapping:

```text
opening = 1
early   = 2..5
middle  = 6..13
late    = 14..20
```

Each JSONL row records:

- stable `game_index` and `decision_index`;
- sorted, unique `features`;
- the extractor's raw emission count before deduplication;
- absolute `focal_seat`;
- `personal_turn` and `phase`;
- the deterministic trajectory policy; and
- whether the free overflow prelude fired.

Duplicate sparse emissions are removed because ADR 0129 defines binary
activation. Their count remains auditable through `raw_feature_count`.

## Deterministic Seeds And Policy Mix

Game seeds are the little-endian first eight bytes of:

```text
BLAKE3("legacy-mid-v4opp-activation-v1/game-seed/v1" || game_index_le_u64)
```

Policy RNGs use a separate domain plus game seed, seat, and policy ID. Policy
assignment is:

```text
(game_index + focal_seat) mod 4
```

with IDs:

```text
0 greedy
1 random_draft
2 scarcity_draft
3 preference_draft
```

This rotation gives every seat exactly 625 games under every policy. Each
policy uses the historical Rust move generator and placement evaluator.
Preference vectors are sampled once per game and seat from the policy RNG.
The policy mix exists only to broaden open-state activation support; this is
not a strength benchmark.

## Manifest And Integrity

The permanent root is:

```text
artifacts/datasets/legacy-mid-v4opp-activation-v1
```

Its `manifest.json` satisfies `scan_legacy_root` in
`tools/feature_schema_activation_census.py` and additionally records:

- exact shard byte sizes, row counts, game ranges, seeds, and BLAKE3 hashes;
- aggregate payload BLAKE3;
- phase, seat, policy, feature-emission, and overflow statistics;
- a deterministic scientific BLAKE3;
- exact source-file identities for the exporter, historical extractor,
  trajectory policy, rules engine, Cargo manifests, and lockfile;
- repository source snapshot identity; and
- executable byte size and BLAKE3.

Generation writes every payload and the manifest under an unpublished sibling
directory, validates the complete dataset, fsyncs files, and atomically
renames the directory into place. Existing output roots are never overwritten.

The validator rejects:

- a wrong schema, split, feature count, row count, or shard sequence;
- missing, extra, corrupt, or size-drifted payload shards;
- malformed JSON rows;
- noncontiguous games or decisions;
- incorrect seats, turns, phases, policies, or seed ranges;
- empty, duplicate, unsorted, or out-of-range feature indices;
- aggregate-statistic or scientific-identity drift;
- relevant source-bundle drift; and
- executable drift unless explicitly running content-only validation.

## Production Command

```bash
cargo run --release -p cascadia-differential \
  --features legacy-teacher \
  --bin legacy_activation_export -- generate \
  --output artifacts/datasets/legacy-mid-v4opp-activation-v1 \
  --games 2500 \
  --shard-games 250 \
  --first-game-index 0
```

Strict validation:

```bash
target/release/legacy_activation_export validate \
  --root artifacts/datasets/legacy-mid-v4opp-activation-v1
```

## Success Gates

The dataset is complete only when:

- focused exporter tests pass;
- formatting and Clippy with warnings denied pass;
- generation produces exactly 200,000 rows in ten shards;
- strict Rust validation passes;
- a smoke ADR 0129 scan passes;
- a full no-row-limit ADR 0129 scan owns all ten shards and passes;
- all four seats and phases have their exact expected row counts;
- the manifest and every payload hash validate;
- repeated validation is deterministic; and
- no test split, teacher rollout, score comparison, queue, cloud, or external
  compute is used.

## Consequences

F1 can now measure the real historical champion representation instead of
leaving it unmeasurable or substituting an approximate reimplementation. Any
future legacy schema needs a distinct dataset ID, feature schema, and
scientific identity; this artifact remains immutable.
