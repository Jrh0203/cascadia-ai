# Exact Candidate Placement Metadata Packing Rejection

Status: **rejected and removed**

Date: 2026-06-15

## Hypothesis

`candidate_move_set` generated an eight-byte transient placement record for
every frontier coordinate and tile rotation. The record eagerly duplicated
axial coordinates even though coordinates are a pure function of the board
index and do not participate in placement ranking.

The treatment packed board index and rotation into one `u16`, retained habitat
score in a second `u16`, and derived `HexCoord` only when a placement became a
selected candidate. This reduced each placement record from eight bytes to
four while preserving vector growth, insertion order, stable sorting,
top-128 truncation, and all downstream iteration order.

## Exactness

Dedicated tests established the eight-byte and four-byte record sizes and
round-tripped all 441 board indices across all six legal rotations. A
test-only expanded oracle then compared complete ordered `CandidateMoveSet`
values across all-A and mixed-card games, ordinary and forced-Nature-Token
states, duplicate wildlife, overflow replacement, and dense habitat ties.

Before timing, the complete default workspace suite passed with 87
`cascadia-ai`, 125 `cascadia-core`, and 61 `cascadia-search` tests. The
`mid-features,v4-opp` suite passed 88 `cascadia-ai` tests, and all 15 focused
Python exact client/service tests passed.

All four mechanism diagnostics and all eight source-screen runs reproduced:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches;
- 6,121,807 logical and 5,062,305 physical neural rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps and zero policy fallbacks;
- clean shutdown.

## Mechanism

The packed representation improved the intended stage on both workers:

| Host | Expanded record | Packed record | Reduction |
|---|---:|---:|---:|
| john2 template preparation | 4,550.042955 ms | 4,468.522717 ms | 1.792% |
| john3 template preparation | 4,557.435520 ms | 4,508.457936 ms | 1.075% |
| john2 retired instructions | 1,087,404,289,721 | 1,087,018,154,780 | 0.0355% |
| john3 retired instructions | 1,087,597,711,415 | 1,087,305,914,700 | 0.0268% |

Mean maximum RSS rose 0.043%, and mean allocator peak footprint rose 0.222%.
The maxima for both treatment memory measures remained below their respective
control maxima. The preregistered mechanism gate therefore passed.

## Source Screen

One treatment-capable non-PGO binary, SHA-256
`e4c13d3d71cf99416c75978266a94afa1c4b9f781c2e272da6c35823d7c9dfae`,
was crossed in opposite balanced orders:

- john2: treatment, control, control, treatment;
- john3: control, treatment, treatment, control.

| Host | Control mean | Packed mean | Result |
|---|---:|---:|---:|
| john2 | 14.653264 s | 14.745721 s | 0.631% slower |
| john3 | 14.457369 s | 14.474092 s | 0.116% slower |
| Combined | **14.555317 s** | **14.609906 s** | **0.375% slower** |

Mean maximum RSS fell 0.127%. Mean allocator peak footprint fell 4.622%, and
mean retired instructions fell 0.024%. Those resource reductions did not
offset the end-to-end wall-time regression.

## Verdict

Reject. The source gate required both hosts to improve and the combined mean
to improve by more than 0.25%. Both hosts instead regressed, so production
conversion and fresh PGO were not authorized.

The packed representation, source-screen environment switch, second release
monomorphization, and temporary oracle tests were removed. The restored
`search.rs` SHA-256 is
`18544722241e587de471dcbc211ae3ac688fadde8fb36ecfe581370a8723caab`.
The complete post-removal suite passed with 85 `cascadia-ai`, 125
`cascadia-core`, and 61 `cascadia-search` tests; the feature-gated suite
passed 86 `cascadia-ai` tests; all 15 focused Python tests passed.

The accepted bounded-slice PGO champion remains unchanged at
14.16305453125 seconds, or 9.957x versus the 141.027296-second reference. The
official Phase 0 gap remains 0.06032493125 seconds.

Machine-readable evidence:
`docs/v2/reports/exact-candidate-placement-metadata-packing-rejection-v1.json`.

Raw evidence is archived under
`artifacts/performance/exact-candidate-placement-metadata-packing-v1/`.
