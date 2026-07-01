# Exact Direct Rollout Template Preparation Acceptance

Status: **accepted**

Date: 2026-06-15

## Change

The exact pipelined rollout policy no longer constructs and groups complete
public-state cache keys before candidate generation. It now builds one
uncached candidate template for each active rollout state and immediately
consumes that local template while preparing the state-specific NNUE
afterstates.

The scalar and synchronous candidate APIs retain their exact single-entry
cache. The qualified production pipeline has no experiment switch and no
obsolete grouping branch.

## Exactness

The direct generator matched the cached candidate set and fallback across four
complete seeded four-player AAAAA games. The pipelined parity suite preserved
ordered sparse rows, predictions, selected actions, rollout traces, and logical
diagnostics.

The complete sequential test suites passed:

- default workspace libraries: `cascadia-ai` 85, `cascadia-core` 125,
  `cascadia-search` 61, and all other workspace libraries;
- `mid-features,v4-opp`: `cascadia-ai` 86, `cascadia-core` 125,
  `cascadia-search` 61, and all other workspace libraries;
- focused Python exact client/service tests: 15 passed.

Every source, production-parity, profile-training, and PGO run reproduced:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches;
- 6,121,807 logical and 5,062,305 physical rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps and zero policy fallbacks;
- clean shutdown.

## Mechanism

The reuse audit observed 440,239 template requests but 440,227 unique exact
public states: only 12 reusable requests, or 0.002726%. Removing key
construction, hashing, indirection, inner cache-key reconstruction, and
template cloning reduced combined key/template/candidate preparation:

| Host | Previous path | Direct production path | Reduction |
|---|---:|---:|---:|
| john2 | 5,106.105738 ms | 4,506.035685 ms | 11.752% |
| john3 | 5,120.997131 ms | 4,524.078099 ms | 11.656% |

Production keying is now diagnostic-only and measured about 0.08 ms for the
complete game when diagnostics were disabled during formal timing.

## Source Screen

The same non-PGO treatment-capable binary was crossed with the treatment
switch off and on in opposite balanced orders, with two measurements per mode
per host:

| Host | Control | Treatment | Improvement |
|---|---:|---:|---:|
| john2 | 15.266181 s | 14.751673 s | 3.370% |
| john3 | 15.025241 s | 14.443337 s | 3.873% |
| Combined | **15.145711 s** | **14.597505 s** | **3.620%** |

Mean maximum RSS fell 0.032%. The treatment-capable experimental binary showed
a short-lived footprint increase while both branches were linked, but the
production-only source binary removed that effect and measured 56.9-58.2 MB
in the two parity runs.

## Fresh PGO

One complete R600 profile was collected on each worker with
`RAYON_NUM_THREADS=1`. Both profiles contained 5,557 functions and 120,282
blocks. Their total counts differed by only 15,423 out of 116.17 billion per
host. Only those two runtime profiles were merged.

The production profile-use LTO binary was crossed against the accepted
parent-afterstate PGO champion:

| Host | Parent-context PGO | Direct-template PGO | Improvement |
|---|---:|---:|---:|
| john2 | 15.031477 s | 14.499902 s | 3.536% |
| john3 | 14.869270 s | 14.166401 s | 4.727% |
| Combined | **14.950374 s** | **14.333151 s** | **4.128%** |

Mean maximum RSS changed by only +0.008%, while mean peak physical footprint
fell 8.879%. The final binary is faster on both workers with no operational
regression.

## Verdict

Accept. Exact public-state template reuse is effectively absent in the
qualified workload, and removing the unused machinery produces a large,
cross-host gain that survives fresh race-free PGO.

The frozen path is now **9.839x** faster than the 141.027296-second reference:

- accepted time: `14.3331512395` seconds;
- 10x threshold: `14.1027296` seconds;
- remaining gap: `0.2304216395` seconds, or 1.608%.

Machine-readable evidence:
`docs/v2/reports/exact-direct-rollout-template-preparation-acceptance-v1.json`.

The complete local evidence archive is preserved under
`artifacts/performance/exact-direct-rollout-template-preparation-v1/`.
