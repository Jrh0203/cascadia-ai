# Full-Legal Paid-Screen Cache Acceptance

Status: **accepted**

Date: 2026-06-15

## Change

Each paid-wipe diagnostic now owns one collision-checked complete-screen
cache. The canonical public-state BLAKE3 selects a bucket, but a hit requires
exact equality of the complete public state.

The production representation interns exact invariant contexts
(`GameConfig`, boards, current player, and completed turns) and stores only
the exact market plus selected result in each entry. The result contains the
canonical action hash, original `f32` screen value promoted to `f64`, and
nature-token return flag. The cache never crosses a diagnostic boundary.

The experiment switch and uncached production branch were removed.

## Exactness

The treatment and production paths reproduced:

- root qualification SHA-256
  `e3210d624f8da22c1a4fb0c61ec15478aec03ab2b079b75bf4e46207aa7bd6be`;
- paid-wipe qualification SHA-256
  `dc866e7fa52fbfc09701bc2a78bbd74e5064f88ac676fece39f27e1c8ed2e348`;
- terminal scores `[96,99,92,102]`;
- terminal state
  `7b3f520d5441aa2ae9c3d97d87e0cd08299d546a7d1e8398f76fdadedb53fa7d`;
- 11,594 ordered legal actions, 32,701 rollout waves, and 617,722
  rollout samples;
- zero bootstraps, zero policy fallbacks, and clean shutdown.

After removing elapsed timing metadata, the prior production report and the
cache production report have identical complete game payloads with BLAKE3
`010f743fd375d1785455cf5434e745bf82526e12359c530ee4842f03faeaf1ae`.

The focused cache test proves that hidden reorderings reuse an exact public
entry and that a forced canonical-hash bucket collision still requires full
state equality. All 20 differential library tests, formatting, bin checks,
report validation, and patch-integrity checks passed.

## Mechanism

The frozen turn-16 diagnostic made 842 complete-screen requests:

- 402 exact public states were evaluated;
- 440 requests hit the cache;
- complete-screen evaluations fell **52.257%**;
- three invariant contexts and 402 exact market entries were retained.

The complete qualification workload changed:

| Metric | Control | Cached | Reduction |
|---|---:|---:|---:|
| Physical rows | 2,822,718 | 2,308,670 | **18.211%** |
| Sparse features | 680,599,071 | 573,927,011 | **15.673%** |
| Service requests | 4,519 | 3,639 | **19.473%** |

The first correct implementation retained one complete public state per
entry and reached the same hit rate, but its allocator peak was about 80 MB.
Interning invariant contexts and storing exact markets reduced that peak to
about 44 MB without changing a result.

## Balanced Confirmation

One compact treatment-capable binary ran opposite balanced crossover:

| Host | Control | Treatment | Improvement |
|---|---:|---:|---:|
| john2 | 18.105 s | 11.410 s | **36.979%** |
| john3 | 17.595 s | 11.450 s | **34.925%** |
| Combined | **17.850 s** | **11.430 s** | **35.966%** |

All eight reports matched the frozen paid-wipe SHA-256. Maximum RSS was flat
within 0.11% on each host. Compact allocator footprint fell 2.863% on john2
and rose 6.809% on john3, a 2.77 MB increase at roughly 43 MB that creates no
operational disadvantage.

## Full Contract

The fixed production path produced:

| Stage | Previous production | Cache production | Change |
|---|---:|---:|---:|
| Complete wall | 212.191376 s | 177.686057 s | **-16.261%** |
| Paid wipe | 64.736415 s | 30.911261 s | **-52.251%** |
| Realized hidden | 126.461304 s | 125.949535 s | -0.405% |
| Logical rows | 66,274,677 | 61,772,615 | **-6.793%** |
| Physical rows | 52,512,624 | 49,899,439 | **-4.976%** |
| Neural batches | 38,695 | 37,160 | **-3.967%** |

Maximum RSS fell 6.282%. Allocator peak footprint rose only 0.386%. The
realized-hidden delta is noise-level and confirms that the change is isolated
to paid-wipe screening.

## Verdict

Accept. Exact public-state memoization eliminates repeated complete screens,
improves both independent workers and the full contract by large margins,
keeps memory bounded, and preserves every semantic output.

The Phase 1 teacher performance position is now:

- frozen reference: `242.43305` seconds;
- accepted production: `177.686057166` seconds;
- total speedup: `1.364390x`;
- required threshold: `24.243305` seconds;
- remaining factor: `7.329x`.

Machine-readable evidence:
`docs/v2/reports/full-legal-audit-paid-screen-cache-acceptance-v1.json`.

The complete local archive is under
`artifacts/performance/full-legal-audit-paid-screen-cache-v1/`.
