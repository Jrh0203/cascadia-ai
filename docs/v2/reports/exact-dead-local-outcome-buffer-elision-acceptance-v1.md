# Exact Dead Local Outcome Buffer Elision Acceptance

Status: **accepted**

Date: 2026-06-15

## Change

The exact production candidate generator no longer allocates, zero-fills, and
frees a local `Option<RotationInvariantOutcome>` vector that the shared-cache
path can never read. The local vector remains only in the test/reference
monomorphization that disables shared outcome reuse.

The experiment switch and eager production monomorphization were removed after
acceptance. `candidate_move_set()` now calls the single qualified shared-cache
implementation directly. No search input, cache key, candidate, ordering rule,
feature row, random stream, or budget changed.

## Exactness

Complete ordered candidate sets were compared against the non-sharing oracle
across seeded games and adversarial board conditions. The local-reference path
still proves that a correctly sized local outcome buffer is available when
sharing is disabled.

Post-promotion verification passed:

- default workspace tests, including `cascadia-ai` 87, `cascadia-core` 125,
  and `cascadia-search` 61;
- `cascadia-ai` with `mid-features,v4-opp`: 88 tests;
- focused Python exact service/client tests: 15;
- formatting and patch-integrity checks.

Every mechanism, source, production-parity, profile-training, PGO, and final
diagnostic run reproduced:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches;
- 6,121,807 logical and 5,062,305 physical neural rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps and zero policy fallbacks;
- clean shutdown.

## Mechanism

Optimized LLVM IR for the treatment removed one dynamic
`frontier.len() * 12` allocation at four-byte alignment. Production IR
contains exactly one matching allocation: the shared cache buffer that is
actually used.

Matched diagnostics showed the intended reduction on both workers:

| Host | Eager local buffer | Elided local buffer | Reduction |
|---|---:|---:|---:|
| john2 template preparation | 4,542.286273 ms | 4,452.052259 ms | 1.987% |
| john3 template preparation | 4,497.884070 ms | 4,476.285531 ms | 0.480% |
| john2 retired instructions | 1,087,604,873,751 | 1,082,900,943,846 | 0.433% |
| john3 retired instructions | 1,088,150,435,236 | 1,082,970,687,422 | 0.476% |

Allocator peak footprint fell on both workers in the mechanism runs.

## Source Screen

One treatment-capable non-PGO binary was crossed in opposite balanced orders,
with two measurements per mode per host:

| Host | Control | Treatment | Improvement |
|---|---:|---:|---:|
| john2 | 14.802802 s | 14.541419 s | 1.766% |
| john3 | 14.466847 s | 14.387534 s | 0.548% |
| Combined | **14.634825 s** | **14.464476 s** | **1.164%** |

Mean maximum RSS fell 0.060%. Mean allocator peak footprint rose 2.684% in
this noisy source screen, while the targeted mechanism runs reduced footprint
on both hosts and the final PGO screen reduced it materially.

## Fresh PGO

One complete R600 production profile was collected per worker with
`RAYON_NUM_THREADS=1`. Each profile contained 5,556 functions and 120,240
blocks. Their total counts differed by only 16,222 out of about 115.99 billion
per host. Only those two profiles were merged.

The fresh production PGO binary was crossed against the accepted bounded-slice
PGO champion:

| Host | Bounded-slice PGO | Dead-buffer-elision PGO | Improvement |
|---|---:|---:|---:|
| john2 | 14.390736 s | 14.211937 s | 1.242% |
| john3 | 14.090791 s | 13.980756 s | 0.781% |
| Combined | **14.240764 s** | **14.096347 s** | **1.014%** |

Mean maximum RSS fell 0.107% and mean allocator peak footprint fell 4.026%.
Final treatment diagnostics attributed 4.455-4.458 seconds to template
preparation, 3.087-3.096 seconds to opponent advancement, and 8.834-8.987
seconds to MLX evaluation.

## Verdict

Accept. The change removes provably dead work from the hottest exact template
path, improves both workers before and after fresh PGO, and preserves the
entire frozen contract.

The production path has cleared the mandatory Phase 0 gate:

- frozen reference: `141.027296` seconds;
- 10x threshold: `14.1027296` seconds;
- accepted time: `14.096346521` seconds;
- total speedup: `10.004528179689x`;
- threshold margin: `0.006383079` seconds.

Machine-readable evidence:
`docs/v2/reports/exact-dead-local-outcome-buffer-elision-acceptance-v1.json`.

The complete local evidence archive is preserved under
`artifacts/performance/exact-dead-local-outcome-buffer-elision-v1/`.
