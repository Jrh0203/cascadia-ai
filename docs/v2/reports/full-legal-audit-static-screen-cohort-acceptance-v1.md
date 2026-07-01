# Full-Legal Static-Screen Cohort Acceptance

Status: **accepted**

Date: 2026-06-15

## Change

Complete-action prior and hidden screens now submit up to **4,096 rows** per
exact MLX request. The accepted rollout pipeline remains fixed at 96 states;
its ordering, overlap, random streams, and numerical path did not change.

The experimental environment switch was removed after qualification.
`EXACT_MLX_STATIC_SCREEN_CHUNK_ROWS` is a production constant, so an ambient
environment variable cannot silently alter the audit contract.

## Exactness

The largest candidate first passed the root and paid-wipe qualification
reports byte for byte:

- root report SHA-256:
  `e3210d624f8da22c1a4fb0c61ec15478aec03ab2b079b75bf4e46207aa7bd6be`;
- paid-wipe report SHA-256:
  `dc866e7fa52fbfc09701bc2a78bbd74e5064f88ac676fece39f27e1c8ed2e348`.

Every sweep, balanced-confirmation, mechanism, and post-promotion report
matched. The complete early/middle/late report retained:

- terminal scores `[96,99,92,102]`;
- terminal state
  `7b3f520d5441aa2ae9c3d97d87e0cd08299d546a7d1e8398f76fdadedb53fa7d`;
- 11,594 ordered legal actions and all action records;
- 66,274,677 logical and 56,095,463 physical neural rows;
- 32,701 rollout waves and 617,722 rollout samples;
- zero bootstraps and zero policy fallbacks.

After removing provenance, elapsed timings, and the intentionally changed
physical request counters, the reference and treatment reports were byte
identical with BLAKE3
`7fd0764edd33ac608f18d011808716679a088fad540b263a58428bad4b8f5d91`.

Focused verification passed 19 Rust tests, formatting, bin checks, report
validation, clean shutdown, and patch-integrity checks.

## Sweep

One treatment-capable binary ran the frozen turn-16 paid-wipe qualification
in ascending order on john2 and descending order on john3:

| Cohort | john2 | john3 | Combined mean |
|---:|---:|---:|---:|
| 512 | 20.22 s | 19.57 s | 19.895 s |
| 1,024 | 18.68 s | 18.66 s | 18.670 s |
| 2,048 | 18.51 s | 18.31 s | 18.410 s |
| **4,096** | **18.52 s** | **18.19 s** | **18.355 s** |
| 8,192 | 18.56 s | 18.50 s | 18.530 s |

The 4,096-row winner then ran in balanced opposite crossover order:

| Host | 96-row control | 4,096-row treatment | Improvement |
|---|---:|---:|---:|
| john2 | 24.665 s | 18.515 s | 24.934% |
| john3 | 23.900 s | 18.375 s | 23.117% |
| Combined | **24.2825 s** | **18.445 s** | **24.040%** |

All eight confirmation reports matched the frozen SHA-256.

## Mechanism

The treatment changed request shape, not work or values:

| Host | Metric | 96 rows | 4,096 rows | Reduction |
|---|---|---:|---:|---:|
| john2 | service requests | 23,115 | 4,519 | 80.450% |
| john2 | MLX evaluation | 11,196.234 ms | 5,757.182 ms | 48.579% |
| john2 | service total | 11,896.911 ms | 6,057.478 ms | 49.084% |
| john3 | service requests | 23,115 | 4,519 | 80.450% |
| john3 | MLX evaluation | 10,700.568 ms | 5,619.818 ms | 47.481% |
| john3 | service total | 11,388.152 ms | 5,928.680 ms | 47.940% |

Maximum RSS rose by about 52 MiB to 161 MiB on each 16 GiB worker, below 1%
of node memory. Peak allocator footprint changed by +0.672% on john2 and
+10.787% on john3. The fixed 8 MiB shared mapping never fell back; its largest
observed request contained 3,948 rows and 817,196 sparse features.

## Full Contract

The promoted constant was measured on the complete frozen seed-60999
early/middle/late audit:

| Stage | Reference | Treatment | Change |
|---|---:|---:|---:|
| Complete wall | 242.433050 s | 217.302694 s | **-10.366%** |
| Paid wipe | 96.281382 s | 70.168686 s | **-27.121%** |
| Complete screening | 0.125168 s | 0.088073 s | -29.636% |
| Realized hidden | 125.548470 s | 126.341884 s | +0.632% |
| Neural requests | 125,861 | 38,695 | **-69.256%** |

The realized-hidden delta is ordinary run noise and confirms that the
treatment remained isolated. The complete report validated successfully.

## Verdict

Accept. Larger static cohorts remove repeated small-request overhead, improve
both workers materially, and preserve the frozen semantic contract.

This becomes the Phase 1 performance baseline, but it does **not** clear the
separate 10x gate:

- frozen reference: `242.43305` seconds;
- accepted static-screen baseline: `217.302693625` seconds;
- required threshold: `24.243305` seconds;
- remaining factor: `8.963x`.

Machine-readable evidence:
`docs/v2/reports/full-legal-audit-static-screen-cohort-acceptance-v1.json`.

The complete local archive is under
`artifacts/performance/full-legal-audit-static-screen-cohort-v1/`.
