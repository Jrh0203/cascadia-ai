# Exact Persistent Evaluator Worker Rejection

Status: **rejected and removed**

Date: 2026-06-15

## Hypothesis

The accepted exact pipeline created capacity-one request/response channels and
spawned one scoped evaluator thread for every sequential-halving batch. A K32
search normally has five rounds, even though the evaluator process and MLX
model remain alive for the complete game.

The treatment retained one evaluator thread and channel pair for every
halving round in a search. Request contents, request order, response order,
96-state cohorts, rollout allocation, random streams, and MLX evaluation were
unchanged.

## Exactness

The treatment matched the per-batch oracle's complete ordered evaluator
request stream, estimates, rollout-value samples, diagnostics, selected
actions, and terminal scores. Dedicated tests also established equivalent
evaluator-error, invalid-width, non-finite-output, disconnect, and panic
behavior.

Before timing, the complete default workspace suite passed with 87
`cascadia-ai`, 125 `cascadia-core`, and 61 `cascadia-search` tests. The
`mid-features,v4-opp` suite passed 88 `cascadia-ai` tests, and all 15 focused
Python exact client/service tests passed.

All 8 source runs, 2 production-parity runs, 2 profile-training runs, and 8
formal PGO runs reproduced:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches;
- 6,121,807 logical and 5,062,305 physical neural rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps and zero policy fallbacks;
- clean shutdown.

## Mechanism

Native samples confirmed that the treatment removed the intended lifecycle
work:

| Host | Accepted worker identities | Treatment | Reduction |
|---|---:|---:|---:|
| john2 | 104 | 21 | 79.808% |
| john3 | 107 | 21 | 80.374% |

The remaining treatment identities were the main thread, the long-lived Rayon
pool, the single evaluator worker, and service/runtime threads. The source
screen also improved both hosts and reduced mean allocator footprint.

## Source Screen

One treatment-capable non-PGO binary, SHA-256
`cbdaf760266343fe80c457bf7e6375ca95db51f2ed68f2e454613a10b86681f8`,
was crossed in opposite balanced orders:

- john2: treatment, control, control, treatment;
- john3: control, treatment, treatment, control.

| Host | Control mean | Treatment mean | Improvement |
|---|---:|---:|---:|
| john2 | 14.631387 s | 14.515858 s | 0.790% |
| john3 | 14.375427 s | 14.324509 s | 0.354% |
| Combined | **14.503407 s** | **14.420184 s** | **0.574%** |

Mean maximum RSS fell 0.040%. Mean allocator peak footprint fell 1.986%.
This passed the preregistered source gate, so the treatment was made
unconditional and rebuilt with fresh PGO.

## Fresh PGO

The production source binary preserved exact parity on both workers. One
complete R600 profile was then collected per host with
`RAYON_NUM_THREADS=1`. Each profile contained 5,596 functions and 121,182
blocks. Host total counts differed by only 24,065 out of roughly 116.17
billion per host, and only those profiles were merged.

The fresh production PGO binary, SHA-256
`d7e5ddde222e733f766d2e05f08f52396a955c8f57d9b63d93f80c66443c3e44`,
was crossed against the accepted bounded-slice PGO champion:

| Host | Accepted control | Persistent worker | Result |
|---|---:|---:|---:|
| john2 | 14.458092 s | 14.228214 s | 1.590% faster |
| john3 | 13.997023 s | 14.096939 s | 0.714% slower |
| Combined | **14.227557 s** | **14.162576 s** | **0.457% faster** |

Mean maximum RSS rose 0.076%. Mean allocator peak footprint fell 2.571%.

One initial warmup pair is explicitly excluded from the formal design. The
john2 treatment warmup completed, but its intended john3 control counterpart
failed immediately because the command contained an incorrect model path. A
correct john3 control warmup was then run. The two successful warmup artifacts
remain archived as `warmup-john2-treatment` and `warmup-john3-control`; neither
contributes to any reported mean.

## Verdict

Reject. The final production-PGO gate required improvement on both workers and
a crossed mean at or below 14.1027296 seconds. John3 regressed 0.714%, and the
treatment mean was 14.16257636475 seconds, 0.05984676475 seconds above the 10x
threshold. The combined gain therefore cannot override the failed host and
absolute-threshold gates.

The persistent worker, source-screen environment switch, dual worker-lifetime
monomorphization, and temporary parity/error tests were removed. The complete
post-removal suite passed with 85 `cascadia-ai`, 125 `cascadia-core`, and 61
`cascadia-search` tests; the feature-gated suite passed 86 `cascadia-ai`
tests; all 15 focused Python tests passed.

The accepted bounded-slice PGO champion remains unchanged at
14.16305453125 seconds, or 9.957x versus the 141.027296-second reference. The
official Phase 0 gap remains 0.06032493125 seconds.

Machine-readable evidence:
`docs/v2/reports/exact-persistent-evaluator-worker-rejection-v1.json`.

Raw evidence is archived under
`artifacts/performance/exact-persistent-evaluator-worker-v1/`.
