# Full-Legal Static-Row Deduplication Acceptance

Status: **accepted**

Date: 2026-06-15

## Change

Every complete-prior screen now uses the rollout pipeline's collision-checked
exact sparse-row deduplicator. Byte-identical rows are submitted once to MLX
and their values are scattered back into original logical action order.

The production path retains no experiment switch. Complete hidden-vector
screens and the accepted 96-state rollout pipeline are unchanged.

## Exactness

The treatment and promoted path reproduced:

- root qualification SHA-256
  `e3210d624f8da22c1a4fb0c61ec15478aec03ab2b079b75bf4e46207aa7bd6be`;
- paid-wipe qualification SHA-256
  `dc866e7fa52fbfc09701bc2a78bbd74e5064f88ac676fece39f27e1c8ed2e348`;
- terminal scores `[96,99,92,102]`;
- terminal state
  `7b3f520d5441aa2ae9c3d97d87e0cd08299d546a7d1e8398f76fdadedb53fa7d`;
- 11,594 ordered legal actions, 66,274,677 logical rows, 32,701 rollout
  waves, and 617,722 rollout samples;
- zero bootstraps, zero policy fallbacks, and clean shutdown.

After removing provenance, elapsed timings, and intentionally changed physical
row counters, the accepted static-screen baseline, treatment, and promoted
reports are byte identical with BLAKE3
`100e8959ad71086bb5986e8004abc8f78229e690e6e80062c957d5a4b01f9a60`.

The collision test forces all fingerprints into one bucket and still recovers
the exact unique rows. Focused verification passed 19 differential tests, the
dedup/scatter legacy test, formatting, bin checks, report validation, and
patch-integrity checks.

## Mechanism

The frozen turn-16 qualification changed:

| Metric | Control | Deduplicated | Reduction |
|---|---:|---:|---:|
| Physical rows | 3,664,806 | 2,822,718 | **22.978%** |
| Sparse features | 855,088,343 | 680,599,071 | **20.406%** |
| Service requests | 4,519 | 4,519 | 0% |

The request count is intentionally stable: deduplication changes row payload,
not the accepted screen boundaries.

| Host | Control MLX | Treatment MLX | Reduction |
|---|---:|---:|---:|
| john2 | 5,757.182 ms | 5,292.152 ms | 8.077% |
| john3 | 5,619.818 ms | 4,997.406 ms | 11.075% |

Total service time fell 7.832% on john2 and 10.741% on john3.

## Source Screen

One treatment-capable binary ran opposite balanced crossover:

| Host | Control | Treatment | Improvement |
|---|---:|---:|---:|
| john2 | 18.665 s | 17.855 s | 4.340% |
| john3 | 18.145 s | 17.910 s | 1.295% |
| Combined | **18.405 s** | **17.8825 s** | **2.839%** |

All eight reports matched the frozen SHA-256. Mean maximum RSS fell 18.943%
on john2 and 19.070% on john3. Mean allocator peak footprint rose 3.367% and
9.780%, respectively, without changing the operating envelope or reliability.

## Full Contract

The fixed production path produced:

| Stage | Static-cohort baseline | Dedup production | Change |
|---|---:|---:|---:|
| Complete wall | 217.302694 s | 212.191376 s | **-2.352%** |
| Paid wipe | 70.168686 s | 64.736415 s | **-7.742%** |
| Realized hidden | 126.341884 s | 126.461304 s | +0.095% |
| Physical rows | 56,095,463 | 52,512,624 | **-6.387%** |

Maximum RSS fell 3.942% and allocator peak footprint fell 0.363%. The
realized-hidden delta is noise-level and confirms isolation.

## Verdict

Accept. Exact sparse-row deduplication removes measurable duplicate inference
work, improves both workers and the complete contract, reduces memory, and
preserves every semantic output.

The Phase 1 performance position is now:

- frozen reference: `242.43305` seconds;
- accepted production: `212.191376166` seconds;
- total speedup: `1.142521x`;
- required threshold: `24.243305` seconds;
- remaining factor: `8.753x`.

Machine-readable evidence:
`docs/v2/reports/full-legal-audit-static-row-dedup-acceptance-v1.json`.

The complete local archive is under
`artifacts/performance/full-legal-audit-static-row-dedup-v1/`.
