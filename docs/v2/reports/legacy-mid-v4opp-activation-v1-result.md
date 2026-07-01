# Legacy Mid-V4Opp Activation Dataset V1 Result

Date: 2026-06-16

Final classification:
**`legacy_mid_v4opp_activation_dataset_complete`**

Dataset root:
`artifacts/datasets/legacy-mid-v4opp-activation-v1`

There is no remaining blocker. The repository now contains a deterministic,
manifest-backed sparse activation corpus produced by the actual historical
Rust `mid-features,v4-opp` extractor. No feature formula was reimplemented in
Python or approximated from documentation.

## Final Identity

| Identity | BLAKE3 |
|---|---|
| Manifest file | `7ade2ca310c976c5db9a0e5a840399e226ad8c650e6a4342da845fbb501e0996` |
| Dataset scientific identity | `193da520e3ccf3f440dd0f657996d486c1144737abcef1f8399b12ee8b34be92` |
| Aggregate payload | `433ebf13b88f6133efa41f42f3225e13278052b82e3f23a7735401427b5019d8` |
| Relevant source bundle | `2ace06a212bce7f5e915f38331cdb5c98f025bce0bbe3a8de96b0e6ebb79141f` |
| Historical `nnue.rs` extractor | `8b7dbacc3827489df297306891c0748347d9cdfb7208104727c37a1fe9143552` |
| Release executable | `ac68f8a6f79b05ca78f01aee7ee643b3d7419d1ab009b4f4d91c058c970e48ba` |
| Full F1 scanner scientific identity | `d97af97e7b7f8fa8dde5e15e8720c6ecf5d7713b39487f641c19dc3b41598047` |
| Four-owner merged F1 identity | `9c3d5d149d068a3ae7c4b8e136df99ae3be9bd11f8b6f8ce95a6670dc3105b3e` |

The scanner's scientific manifest identity for this root is
`8b7c269932cc15c0fd5e46ef123af847b61258521ea30f1ee396da8703066c01`.

## Materialized Corpus

| Field | Result |
|---|---:|
| Feature schema | `legacy-mid-v4opp-11231` |
| Split | `train` |
| Games | 2,500 |
| Rows | 200,000 |
| Shards | 10 |
| Rows per shard | 20,000 |
| On-disk root size including validation | 284 MB |
| Generation plus strict validation | 1.40 seconds |
| Full F1 scan | 15.08 seconds |
| Unique features per row | 165 minimum, 317 maximum |
| Raw feature emissions | 49,849,871 |
| Unique binary activations | 46,833,451 |
| Duplicate emissions removed and recorded | 3,016,420 |
| Free overflow preludes applied | 61,006 |

Every sparse row is nonempty, strictly sorted, unique, and bounded below
11,231. Every game contains exactly 80 contiguous decisions.

## Cohort Coverage

| Cohort | Rows |
|---|---:|
| Opening | 10,000 |
| Early | 40,000 |
| Middle | 80,000 |
| Late | 70,000 |
| Focal seat 0 | 50,000 |
| Focal seat 1 | 50,000 |
| Focal seat 2 | 50,000 |
| Focal seat 3 | 50,000 |
| Greedy | 50,000 |
| Random draft | 50,000 |
| Scarcity draft | 50,000 |
| Preference draft | 50,000 |

The full F1 scan reports the v4 opponent block active on every row in every
seat and phase cohort. For each focal seat, the exact phase counts are 2,500
opening, 10,000 early, 20,000 middle, and 17,500 late.

## Scanner Verification

The bounded smoke scanned 1,000 rows successfully.

The full no-row-limit scanner:

- validated the manifest and all ten payload BLAKE3 hashes;
- owned and scanned all 200,000 rows;
- exposed all four seats and all four frozen phases;
- retained all closed-domain flags as false; and
- emitted 16,428 deterministic channel-detail rows.

The full report and detail JSONL were generated twice. Both pairs were
byte-for-byte identical:

| Artifact | File BLAKE3 |
|---|---|
| Full scanner JSON | `d349067fe93ce91e5e7d7a32c317cf0fb8e8d0aede1ca8da07afee84262cfe98` |
| Full scanner details | `13019a56c4d9d67ca09d1aa7370e6ab69f69b7c65851c65fb24c57c639414211` |

Four local ownership scans were also run with exact shard IDs `0/4`, `1/4`,
`2/4`, and `3/4`. Their evidence distribution was 0, 2, 4, and 4 payload
shards respectively. The empty `0/4` ownership report is valid and explicit;
the other reports contain every payload exactly once. `merge
--require-shards 4` accepted the exact set, scanned 200,000 rows from ten
unique evidence IDs, and preserved every closed-domain flag as false.

The four-owner merge was repeated with reverse input order. Its JSON and
16,428-row detail stream were byte-for-byte identical:

| Artifact | File BLAKE3 |
|---|---|
| Four-owner merged JSON | `041e4f13ec97fabd4ef42679a7830400ccbb8d529920b02d60f68503bcc97bf5` |
| Four-owner merged details | `13019a56c4d9d67ca09d1aa7370e6ab69f69b7c65851c65fb24c57c639414211` |

## Initial Activation Finding

The corpus reproduces the frozen 11,231-column layout and confirms a concrete
consequence of its historical defect:

- `legacy.mid_tail_historical_adjacency_prefix` (`10561..10862`) is 301/301
  dead on the observed domain;
- `legacy.v4opp` (`10862..11231`) is active, with 31 dead and 4 rare channels
  at the frozen `1e-4` threshold.

The dead accidental prefix is consistent with its ownership of the first
three complete absolute-grid cells plus 67 channels of the fourth cell:
ordinary centered Cascadia boards never occupy those far-corner cells. This
is an activation result, not a correction to historical checkpoint semantics.

## Verification Commands

Focused exporter tests:

```bash
cargo test -p cascadia-differential \
  --features legacy-teacher \
  --bin legacy_activation_export
```

Result: 5 passed, 0 failed.

Scoped Clippy:

```bash
cargo clippy -p cascadia-differential \
  --features legacy-teacher \
  --bin legacy_activation_export \
  --no-deps -- \
  -D warnings \
  -A clippy::repeat-vec-with-capacity \
  -A clippy::collapsible-if \
  -A clippy::large-enum-variant
```

Result: passed. The three allowances cover pre-existing diagnostics in
concurrently owned shared modules; the exporter itself has no denied warning.

Formatting and fallback build:

```bash
rustfmt --edition 2024 --check \
  crates/cascadia-differential/src/bin/legacy_activation_export.rs

cargo check -p cascadia-differential \
  --bin legacy_activation_export
```

Result: both passed. The no-feature binary fails closed at runtime with an
instruction to enable `legacy-teacher`.

Production generation:

```bash
target/release/legacy_activation_export generate \
  --output artifacts/datasets/legacy-mid-v4opp-activation-v1 \
  --games 2500 \
  --shard-games 250 \
  --first-game-index 0
```

Repeated strict validation:

```bash
target/release/legacy_activation_export validate \
  --root artifacts/datasets/legacy-mid-v4opp-activation-v1
```

Result: valid, with matching relevant source and executable identities.

F1 scanner regression suite:

```bash
uv run pytest -q tools/test_feature_schema_activation_census.py
```

Result: 7 passed, 0 failed.

Smoke and full scanner:

```bash
uv run python tools/feature_schema_activation_census.py census \
  --legacy-root artifacts/datasets/legacy-mid-v4opp-activation-v1 \
  --row-limit 1000 \
  --output artifacts/datasets/legacy-mid-v4opp-activation-v1/validation/f1-scanner-smoke.json

uv run python tools/feature_schema_activation_census.py census \
  --legacy-root artifacts/datasets/legacy-mid-v4opp-activation-v1 \
  --output artifacts/datasets/legacy-mid-v4opp-activation-v1/validation/f1-scanner-full.json \
  --details-jsonl artifacts/datasets/legacy-mid-v4opp-activation-v1/validation/f1-scanner-full-details.jsonl
```

Four-owner merge:

```bash
uv run python tools/feature_schema_activation_census.py merge \
  --input artifacts/datasets/legacy-mid-v4opp-activation-v1/validation/shards/f1-owner-0-of-4.json \
  --input artifacts/datasets/legacy-mid-v4opp-activation-v1/validation/shards/f1-owner-1-of-4.json \
  --input artifacts/datasets/legacy-mid-v4opp-activation-v1/validation/shards/f1-owner-2-of-4.json \
  --input artifacts/datasets/legacy-mid-v4opp-activation-v1/validation/shards/f1-owner-3-of-4.json \
  --require-shards 4 \
  --output artifacts/datasets/legacy-mid-v4opp-activation-v1/validation/f1-scanner-four-owner-merge.json \
  --details-jsonl artifacts/datasets/legacy-mid-v4opp-activation-v1/validation/f1-scanner-four-owner-merge-details.jsonl
```

## Changed And Generated Files

Implementation and documentation:

- `crates/cascadia-differential/src/bin/legacy_activation_export.rs`
- `docs/v2/decisions/0133-legacy-mid-v4opp-activation-dataset.md`
- `docs/v2/reports/legacy-mid-v4opp-activation-v1-result.md`

Generated dataset:

- `artifacts/datasets/legacy-mid-v4opp-activation-v1/manifest.json`
- `artifacts/datasets/legacy-mid-v4opp-activation-v1/part-00000.jsonl`
  through `part-00009.jsonl`
- `artifacts/datasets/legacy-mid-v4opp-activation-v1/validation/`

ADR 0129, its preregistration, the F1 ledger and manifest, cluster queue,
dashboard, and all sealed/test data were left unchanged. No teacher rollout,
score benchmark, ML training, cloud execution, or external compute was used.
