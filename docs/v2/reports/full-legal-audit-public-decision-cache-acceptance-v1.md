# Full-Legal Public Decision Cache Acceptance

Status: **accepted**

Date: 2026-06-15

## Change

The full-legal audit now owns one exact public-decision cache for each game.
Canonical public-state BLAKE3 selects a bucket, but every hit requires exact
equality of the canonical `PublicGameState` bytes. Entries retain the selected
canonical `TurnAction`.

Ordinary champion play may reuse an exact cached decision. Audited champion
decisions still execute the complete R600 search and validate the selected
action against any existing entry. Every realized-hidden finalist is still
played to the terminal state, but only the branch rooted at the outer
champion action retains decisions because only that branch can reappear on
the subsequent outer trajectory.

The qualification switch and disabled production branch were removed.

## Exactness

All qualification and production paths reproduced:

- terminal scores `[96,99,92,102]`;
- terminal state
  `7b3f520d5441aa2ae9c3d97d87e0cd08299d546a7d1e8398f76fdadedb53fa7d`;
- 11,594 ordered legal actions;
- the complete champion, substantial, high-confidence, paid-wipe, and
  realized-hidden records;
- zero bootstrap samples, zero policy fallbacks, and clean shutdown.

After removing provenance, timing, cache, bridge, and batch diagnostics, the
frozen parent, all eight crossover reports, the local qualification, and the
switch-free production report have the identical semantic BLAKE3
`f46ae73349d53d1baa3c69c0f8a3efab5766ed68ef91b6636ad65a3dea340c75`.

The focused cache test proves that hidden reorderings reuse one public entry,
that a forced digest collision still requires exact state equality, and that
conflicting actions for one exact state fail. All 21 feature-enabled
differential library tests, three trusted fixtures, formatting, bin checks,
report validation, and patch-integrity checks passed.

## Mechanism

The frozen game makes 1,040 public policy requests:

| Metric | Control | Cache | Reduction |
|---|---:|---:|---:|
| Policy evaluations | 1,040 | 922 | **11.346%** |
| Neural batches | 37,160 | 33,260 | **10.495%** |
| Logical rows | 61,772,615 | 55,710,626 | **9.813%** |
| Physical rows | 49,899,439 | 44,903,952 | **10.011%** |
| Rollout waves | 32,701 | 29,151 | **10.856%** |
| Rollout samples | 617,722 | 549,517 | **11.041%** |

The production cache records 118 hits and retains only 80 exact entries.
The first correct qualification retained every finalist continuation and
reached the same 118 hits, but kept 920 entries. Restricting retention to the
only reusable champion-root branch removed that unnecessary footprint without
changing one result.

## Balanced Confirmation

One treatment-capable binary ran the preregistered opposite-order crossover:

| Host | Control | Treatment | Improvement |
|---|---:|---:|---:|
| john2 | 168.170 s | 153.720 s | **8.592%** |
| john3 | 165.445 s | 151.285 s | **8.559%** |
| Combined | **166.8075 s** | **152.5025 s** | **8.576%** |

Every treatment run recorded exactly 118 hits, 922 evaluations, and 80
entries. Every control run recorded 1,040 evaluations and zero hits or
entries.

Mean per-run maximum RSS changed +7.311% on john2 and -3.931% on john3.
Mean allocator peak footprint changed +4.019% on john2 and -8.114% on john3.
All eight runs reported zero swaps and stayed inside the preregistered 10%
memory gates.

## Full Contract

The switch-free production path produced:

| Metric | Parent | Production | Change |
|---|---:|---:|---:|
| Complete report wall | 177.686057 s | **162.045309 s** | **-8.802%** |
| Realized hidden | 125.949535 s | 121.252030 s | -3.730% |
| Paid wipe | 30.911261 s | 30.970525 s | +0.192% |
| Logical rows | 61,772,615 | 55,710,626 | **-9.813%** |
| Physical rows | 49,899,439 | 44,903,952 | **-10.011%** |
| Neural batches | 37,160 | 33,260 | **-10.495%** |

Maximum RSS changed +2.782% and allocator peak changed +0.231% versus the
parent production run. The production report validated successfully and has
BLAKE3
`c789a981327516e7e0ecafcf3b71b4e73ffd0dfe0444d1c5f094d67f2a15e898`.

## Verdict

Accept. Exact game-scoped public-decision reuse removes searches that the
audit previously repeated across champion continuations and outer play. It
improves both independent workers, exceeds the combined 7.5% gate, keeps
memory bounded, and preserves every semantic output.

The Phase 1 teacher performance position is now:

- frozen reference: `242.433050` seconds;
- accepted production: `162.04530875` seconds;
- total speedup: `1.496082x`;
- required threshold: `24.243305` seconds;
- remaining factor: `6.684x`.

Machine-readable evidence:
`docs/v2/reports/full-legal-audit-public-decision-cache-acceptance-v1.json`.

The complete local archive is under
`artifacts/performance/full-legal-audit-public-decision-cache-v1/`.
