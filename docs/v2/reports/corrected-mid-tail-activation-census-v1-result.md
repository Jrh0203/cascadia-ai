# Corrected Mid-Tail Activation Census V1 Result

Date: 2026-06-17

Experiment ID: `corrected-mid-tail-activation-census-v1`

Classification: **`corrected_mid_tail_activation_census_complete`**

Outcome: **Passed**

## Executive Result

The source-frozen four-host production corpus naturally activated every one
of the 301 corrected historical mid-tail rows:

- tile-bag terrain counts: 150/150 channels;
- tile-bag wildlife-capacity counts: 150/150 channels; and
- free-overflow-used: 1/1 channel.

The result uses 81,920 representative public states from 1,024 deterministic
AAAAA games. Row `11230` activated in 25,903 natural trajectory records. The
separate reachable overflow witness also activated row `11230`, but it is
excluded from every representative statistic.

This closes the corrected-tail activation gate. It does not make a gameplay,
strength, or score claim.

## Frozen Identities

| Identity | BLAKE3 |
|---|---|
| Source bundle | `935911658b8e466e2534668756424d2db7efa800d4511ae9618acb2e526dea9d` |
| Corrected extractor `nnue.rs` | `c636d9649ab1665be8d54d5a359b8bc1c049b8e2795f6c9ef27ea71294caa0ab` |
| Aggregate scientific result | `8f6f1a243b5c6ce583dde50445252bbacd2a5af6a2224605e2c58444bff34658` |
| Representative public-state stream | `518c0e908a3731d72465dc718c1d8dce6dce442818d7d5ab181bc1119fb0a642` |
| Normalized feature stream | `14c3b77fca780f1464cbc8257ed4809e105028aeb371fbe6ec350f31e99f433d` |
| Corrected-tail stream | `4e243ed204af8a85605ef95d1fa986a0f154809c6885d038f310753e1659a4f9` |
| Record identity stream | `72947e6db77c0396c58b5c1fc9bee1eaa17312f6db3ee5668c136f215904c471` |

The complete forward and reverse aggregate files are byte-identical and have
SHA-256
`a9e51ef9d60d279e9b3c1efa22de7f20f60af4b7e7094cae59514cd9febefd5a`.

## Representative Corpus

| Measurement | Result |
|---|---:|
| Shards | 4 / 4 |
| Games | 1,024 / 1,024 |
| Representative rows | 81,920 / 81,920 |
| Rows per shard | 20,480 |
| Opening | 4,096 / 4,096 |
| Early | 16,384 / 16,384 |
| Middle | 32,768 / 32,768 |
| Late | 28,672 / 28,672 |
| Seat 0 / 1 / 2 / 3 | 20,480 each |
| Greedy / random / scarcity / preference | 20,480 each |
| Natural overflow-used rows | 25,903 |
| Natural overflow-not-used rows | 56,017 |

Every game index in `0..1024` appears exactly once. Shard ownership is
`game_index mod 4`, mapped to `john1`, `john2`, `john3`, and `john4`.

## Corrected Blocks

| Block | Range | Active channels | Activations | Verdict |
|---|---|---:|---:|---|
| Tile-bag terrain counts | `10930..11080` | 150 / 150 | 409,600 | Pass |
| Tile-bag wildlife capacity | `11080..11230` | 150 / 150 | 409,600 | Pass |
| Overflow used | `11230..11231` | 1 / 1 | 25,903 | Pass |

Every representative row activates exactly five terrain-count rows and five
wildlife-capacity rows. The least-supported corrected channel is row `11019`,
wetland count bin 29, with 468 natural activations. It appears across every
seat and trajectory policy and in both overflow slices.

## Separate Overflow Witness

| Field | Result |
|---|---|
| Source | Separate reachable opening-market adversarial fixture |
| Search counter | 10 |
| Public-state BLAKE3 | `9d134da7a9c05d4df0d15fa3f6bc9151347964faff2e437e5173ba252eb241cb` |
| Full-feature BLAKE3 | `bd493ab1ac40b633f5e49ace7086d7b331d450f89ac6b4dc29da18703cf95086` |
| Corrected-tail BLAKE3 | `e43dfd84b764fc510670c86a18f1444456afea7483c7b16e105af8e56306bfc2` |
| Row `11230` active | Yes |
| Included in representative statistics | **No** |

The witness proves extractor reachability independently. The natural corpus
passes the overflow gate without it.

## Integrity Gates

| Gate | Result |
|---|---|
| Exact shard IDs `0..3` | Pass |
| No game overlap or gap | Pass |
| Exact game IDs `0..1024` | Pass |
| One source and extractor identity | Pass |
| Exact phase, seat, and policy coverage | Pass |
| Natural overflow-used and not-used slices | Pass |
| Every record replayed through the actual Rust extractor | Pass |
| Every scientific input content-hashed | Pass |
| Separate witness excluded from representative totals | Pass |
| Forward/reverse aggregate byte identity | Pass |

The first launch attempts for shard 0 and shard 2 failed before scientific
execution because the coordinator host temporarily exhausted its process
table. After the completed agent process was closed and resources returned,
only those two tasks were retried. Their successful attempts used the same
immutable bundle and frozen task definitions. No partial scientific artifact
from either failed launch was admitted.

## Decision

The corrected 301-row tail is demonstrably live and sufficiently exercised
to proceed to a separately preregistered F5 fine-tuning comparison. That
successor must retain the frozen corrected schema, use stable fine-tuning
learning rates, and compare against the current champion under paired seeds.

This census itself authorizes no model promotion.

## Production Artifacts

- `artifacts/experiments/corrected-mid-tail-activation-census-v1/queue-spec.json`
- `artifacts/experiments/corrected-mid-tail-activation-census-v1/corpus/shard-{0,1,2,3}/manifest.json`
- `artifacts/experiments/corrected-mid-tail-activation-census-v1/corpus/shard-{0,1,2,3}/records.jsonl`
- `artifacts/experiments/corrected-mid-tail-activation-census-v1/reports/shard-{0,1,2,3}.json`
- `artifacts/experiments/corrected-mid-tail-activation-census-v1/reports/remote-collection.json`
- `artifacts/experiments/corrected-mid-tail-activation-census-v1/reports/aggregate-forward.json`
- `artifacts/experiments/corrected-mid-tail-activation-census-v1/reports/aggregate-reverse.json`
- `artifacts/experiments/corrected-mid-tail-activation-census-v1/reports/source-frozen-bundle-fanout.json`
