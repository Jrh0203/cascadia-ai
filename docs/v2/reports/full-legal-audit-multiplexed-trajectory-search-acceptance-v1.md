# Full-Legal Multiplexed Trajectory Search Acceptance

Status: **accepted**

Date: 2026-06-15

## Change

Realized-hidden terminal diagnostics now advance all eight finalist searches
in deterministic lockstep. The scheduler preserves an independent K32/R600
search state for every trajectory while sharing the process-global Rayon pool
and one MLX evaluator.

At each barrier, every unfinished search either completes or submits one owned
sparse request. Requests are concatenated in stable search-index order,
evaluated once, and split back by their exact ranges. No cross-request row
deduplication occurs, so the isolated mechanism is coordinated CPU work and
request multiplexing.

The qualification switch and serial production branch were removed.

## Exactness

All qualification, crossover, and production paths reproduced:

- terminal scores `[96,99,92,102]`;
- terminal state
  `7b3f520d5441aa2ae9c3d97d87e0cd08299d546a7d1e8398f76fdadedb53fa7d`;
- 11,594 ordered legal actions;
- 33,260 logical neural batches and 55,710,626 logical rows;
- 44,903,952 physical neural rows;
- 29,151 rollout waves and 549,517 rollout samples;
- zero bootstrap samples, zero policy fallbacks, and clean shutdown.

After removing provenance, timing, service, cache, bridge, and batch
diagnostics, every report has the frozen semantic BLAKE3
`f46ae73349d53d1baa3c69c0f8a3efab5766ed68ef91b6636ad65a3dea340c75`.

Owned-request range tests, deterministic serial-versus-multiplex search tests,
all 23 feature-enabled differential library tests, all 10 NNUE batch tests,
formatting, binary checks, report validation, and patch-integrity checks
passed.

## Mechanism

| Metric | Result |
|---|---:|
| Multi-search cohorts | 120 |
| Exact searches multiplexed | 907 |
| Logical evaluator requests | 58,157 |
| Physical evaluator batches | 7,588 |
| Service-batch reduction | **86.953%** |
| Coalesced evaluator batches | 7,586 / 7,588 |
| Coalescing rate | **99.974%** |
| Evaluator rows | 39,065,421 |
| Maximum requests per batch | 8 |
| Maximum rows per batch | 10,148 |

The experiment exceeded the registered 700-search, 90% coalescing, and 50%
service-reduction gates. Stable range splitting and independent search tests
proved that no request or prediction was reordered.

## Balanced Confirmation

One treatment-capable binary ran the registered opposite-order crossover:

| Host | Control | Treatment | Improvement |
|---|---:|---:|---:|
| john2 | 153.799921 s | 128.772081 s | **16.273%** |
| john3 | 151.379278 s | 129.437743 s | **14.494%** |
| Combined | **152.589600 s** | **129.104912 s** | **15.391%** |

Realized-hidden timing improved independently and in aggregate:

| Host | Control | Treatment | Improvement |
|---|---:|---:|---:|
| john2 | 115.246562 s | 90.439940 s | **21.525%** |
| john3 | 113.311097 s | 91.295086 s | **19.430%** |
| Combined | **114.278829 s** | **90.867513 s** | **20.486%** |

All eight runs reported zero swaps. Maximum treatment RSS was 970,457,088
bytes and maximum allocator peak was 252,297,840 bytes, both well below the
1.5 GiB gate. The higher RSS is the expected bounded cost of keeping eight
independent exact searches resident together.

## Full Contract

The final switch-free production path produced:

| Metric | Parent | Production | Change |
|---|---:|---:|---:|
| Complete report wall | 162.045309 s | **143.775461 s** | **-11.275%** |
| Realized hidden | 121.252030 s | **103.119389 s** | **-14.955%** |
| Paid wipe | 30.970525 s | 30.726967 s | -0.786% |
| Maximum RSS | 249,987,072 B | 907,378,688 B | +262.970% |
| Allocator peak | 149,373,408 B | 152,093,200 B | +1.821% |

The production report validated successfully, preserved the frozen semantic
digest, used zero swap, and remained below the memory ceiling. Its BLAKE3 is
`98c554753d42904877db44d473389068022e6160afe45a5658f8c56ab31514bc`.

The switch-free production binary has SHA-256
`f19f30a16c0c0dc870176dadd568c585a49479639ccc895b6d8966ff8640cc18`
and BLAKE3
`6639d060edd1c9faa83acf17addc2230c979440bdf49204ab9b8a5cabeb2e19f`.

## Verdict

Accept. Exact lockstep scheduling combines contemporaneous sparse inference
requests and shares CPU preparation without changing any search, action,
diagnostic, score, or hidden-information boundary. It improves both remote
workers, exceeds the combined wall and realized-hidden gates, and remains
positive after the qualification branch is removed.

The Phase 1 teacher performance position is now:

- frozen reference: `242.433050` seconds;
- accepted production: `143.775461333` seconds;
- total speedup: `1.686192x`;
- required threshold: `24.243305` seconds;
- remaining factor: `5.931x`.

Machine-readable evidence:
`docs/v2/reports/full-legal-audit-multiplexed-trajectory-search-acceptance-v1.json`.

The complete archive is under
`artifacts/performance/full-legal-audit-multiplexed-trajectory-search-v1/`.
