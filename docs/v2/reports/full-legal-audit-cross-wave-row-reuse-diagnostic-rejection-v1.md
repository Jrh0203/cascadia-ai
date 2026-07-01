# Full-Legal Cross-Wave Sparse-Row Reuse Diagnostic Rejection

Status: **rejected**

Date: 2026-06-15

## Result

The exact late-turn diagnostic observed every physical sparse row produced by
the rollout pipeline within each independent K32/R600 search. A fingerprint
selected a lookup bucket, but complete `Vec<u16>` equality was required before
a row counted as repeated.

| Metric | Result |
|---|---:|
| Physical neural rows in report | 7,198,144 |
| Physical rows observed by cross-wave trackers | 6,116,501 |
| Exact rows repeated in a later wave or round | **0** |
| Exact reuse rate | **0.000%** |
| Rollout waves | 4,355 |
| Independent searches measured | 104 |

The diagnostic exceeded the 100,000-row observation floor by more than 61x
but missed the 5% advance gate completely. No full-contract diagnostic and no
prediction cache are authorized.

## Correctness

The forced-collision unit test proves that distinct rows sharing one
fingerprint are stored and compared independently. All 11 NNUE batch tests and
all 24 feature-enabled differential library tests passed.

The report validated and reproduced:

- terminal scores `[96,99,92,102]`;
- terminal state
  `7b3f520d5441aa2ae9c3d97d87e0cd08299d546a7d1e8398f76fdadedb53fa7d`;
- 9,288,014 logical rows and 7,198,144 physical rows;
- 4,355 rollout waves and 104,615 rollout samples;
- zero bootstrap samples, zero policy fallbacks, and zero bridge fallbacks.

The normalized turn-66 semantic BLAKE3 is
`6f19d82622bab6a5a45c6cdf6e1152f99791630436d1a0354d9f629f95089863`,
identical to the frozen turn-66 treatment. The preregistration initially named
the three-turn digest; that clerical scope mismatch is recorded in its outcome
and does not change this rejection.

## Diagnostic Cost

| Metric | Uninstrumented | Exact tracker | Change |
|---|---:|---:|---:|
| Complete wall | 31.881540 s | 32.636196 s | +2.367% |
| Maximum RSS | 365,674,496 B | 504,102,912 B | +37.856% |
| Allocator peak | 145,408,504 B | 303,203,000 B | +108.518% |

The process used zero swap and remained below 1.5 GiB. The additional memory
is diagnostic-only storage for complete sparse rows and is never active in
production.

## Verdict

Reject a per-search cross-wave prediction cache. The exact feature rows are
effectively unique across sequential-halving rounds and rollout waves, so
such a cache would add hashing, equality checks, storage, and scatter work
without removing one inference row in the measured screen.

This does not test duplicate rows between different searches submitted to the
same multiplexed evaluator batch. That narrower, bounded opportunity remains
eligible for a separate diagnostic because request multiplexing intentionally
preserved all cross-request rows.

Machine-readable evidence:
`docs/v2/reports/full-legal-audit-cross-wave-row-reuse-diagnostic-rejection-v1.json`.

The local archive is under
`artifacts/performance/full-legal-audit-cross-wave-row-reuse-diagnostic-v1/`.
