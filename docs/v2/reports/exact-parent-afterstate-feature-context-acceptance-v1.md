# Exact Parent Afterstate Feature Context Acceptance

Status: **accepted**

Date: 2026-06-15

## Change

The qualified `mid-features,v4-opp` extractor previously rebuilt every sparse
feature block after applying each candidate tile and wildlife placement.
The accepted implementation constructs one `MidV4AfterstateFeatureContext`
per rollout policy state and reuses:

- invariant cell, allowed-wildlife, and secondary-terrain state;
- bag, market, and opponent feature blocks;
- a parent `MidV4PatternSnapshot`;
- wildlife-domain metrics unaffected by the candidate placement.

The child path still emits the historical ordered `Vec<u16>`, including the
truncated mid-feature adjacency range and v4 opponent block. The general
extractor remains an independent debug/test oracle and the fallback for every
other feature configuration. There is no experiment switch in production.

## Exactness

The focused oracle test covered eight complete four-player AAAAA games and all
640 turns. Every prepared candidate row matched
`extract_features_with_bag` byte for byte. Coverage included ordinary and
independent drafts, same-cell and different-cell wildlife placement,
keystones, no-wildlife moves, and all six tile rotations.

The complete library suites passed separately:

- default: 84 passed;
- `mid-features,v4-opp`: 85 passed.

All source, profile-training, and PGO executions reproduced:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches;
- 6,121,807 logical and 5,062,305 physical rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps and zero policy fallbacks;
- clean service shutdown.

## Mechanism

Stage timing confirmed that the intended work disappeared:

| Host | Previous candidate prep | Parent context | Reduction |
|---|---:|---:|---:|
| john2 | 1,528.318840 ms | 1,252.633045 ms | 18.039% |
| john3 | 1,537.515084 ms | 1,268.272850 ms | 17.512% |

## Source Screen

Matched non-PGO binaries were crossed in opposite balanced orders with two
measurements per binary per host.

| Host | Control | Treatment | Improvement |
|---|---:|---:|---:|
| john2 | 15.761652 s | 15.323042 s | 2.783% |
| john3 | 15.509282 s | 15.161453 s | 2.243% |
| Combined | **15.635467 s** | **15.242247 s** | **2.515%** |

Mean peak RSS fell by 0.036%, and the treatment global peak was 16,384 bytes
lower. The source result cleared every preregistered advancement gate.

## Fresh PGO

One complete R600 profile was collected on each worker with
`RAYON_NUM_THREADS=1`. Both profiles contained 5,476 functions and 120,121
blocks. Their total counts differed by 113,437 out of 116.38 billion.
Only those two runtime profiles were merged.

The fresh profile-use LTO binary was crossed against the accepted
elk-potential PGO champion:

| Host | Elk PGO | Parent-context PGO | Improvement |
|---|---:|---:|---:|
| john2 | 15.333361 s | 15.211514 s | 0.795% |
| john3 | 15.146230 s | 14.826227 s | 2.113% |
| Combined | **15.239795 s** | **15.018871 s** | **1.450%** |

The final RSS measurements varied by less than 0.4% across all runs. The
treatment mean was 0.12% higher and its global peak was 180,224 bytes higher,
which is within the observed run-to-run spread and is not a material
operational regression.

## Verdict

Accept. The optimization is exact, removes the measured feature-construction
bottleneck, and survives production PGO with the same sign on both workers.

The frozen path is now **9.390x** faster than the 141.027296-second reference:

- accepted time: `15.01887077075` seconds;
- 10x threshold: `14.1027296` seconds;
- remaining gap: `0.91614117075` seconds, or 6.100%.

Machine-readable evidence:
`docs/v2/reports/exact-parent-afterstate-feature-context-acceptance-v1.json`.

The complete local evidence archive is preserved under
`artifacts/performance/exact-parent-afterstate-feature-context-v1/`.
