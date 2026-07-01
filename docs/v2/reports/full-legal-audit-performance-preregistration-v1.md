# Full-Legal Audit 10x Performance Preregistration

Status: **active**

Date: 2026-06-15

## Decision

The Phase 1 audit is a new dominant workload and must clear its own exact 10x
single-Mac gate before the 13-game corpus starts.

The frozen seed-60999 early/middle/late run took **242.43305 seconds**.
Only **57.627%** of audited-decision time remained in the previously accepted
full-terminal rollout pipeline. Paid-wipe analysis took 96.281382 seconds and
realized-hidden continuations took 125.548470 seconds, so the preregistered
80% inheritance condition failed.

The new acceptance threshold is therefore **24.243305 seconds**.

Machine-readable reference:
[`full-legal-audit-performance-reference-v1.json`](full-legal-audit-performance-reference-v1.json).

## Frozen Contract

- seed `60999`, four-player AAAAA, no habitat bonuses;
- accepted exact MLX K32/R600 champion in every seat;
- audited completed turns `12,39,66`;
- complete legal screen, top 64, 16 sentinels, R1200 substantial evaluation;
- top eight plus required anchors at R4800;
- paid wipe D8, followup D2, width 3, all 15 first wipe masks;
- realized-hidden terminal continuation for every high-confidence finalist;
- identical model, numerical path, random domains, action order, reports, and
  service lifecycle.

The reference produced:

- terminal scores `[96,99,92,102]`;
- 11,594 complete actions;
- 66,274,677 logical and 56,095,463 physical neural rows;
- 32,701 rollout waves and 617,722 rollout samples;
- zero bootstraps and zero policy fallbacks;
- a 12,703,794-byte validated report with BLAKE3
  `1fea91a9243f49b1bd93b50ffb249c0bb4b9e1658749a52d70e87159cfef77bf`.

## Optimization Program

The work proceeds by measured impact, with each treatment isolated and
rejected unless exact:

1. **Factor complete-screen preparation by post-draft context.** Thousands of
   placement actions currently clone and execute a complete legacy game even
   though market refill and bag context depend only on draft slots and whether
   wildlife was placed. Build each distinct post-draft context once, then use
   board place/undo plus the accepted mid-v4 feature context for its actions.
2. **Deduplicate exact complete-screen rows.** Evaluate one copy of every
   byte-identical sparse row and restore logical action order exactly.
3. **Batch or share repeated paid-wipe screens.** Cache by complete public
   state and evaluate independent chance branches in deterministic cohorts
   without changing any sample or contingent decision.
4. **Parallelize realized-hidden finalists.** Run independent finalist
   continuations concurrently while preserving each continuation's exact
   service, RNG, action trace, terminal state, and output order.
5. **Reuse exact continuation work where complete state identity proves it
   valid.** No digest-only cache or approximate transposition is allowed.
6. **Collect fresh PGO only after source-level treatments clear correctness
   and mechanism gates.**

## Correctness Gates

Every treatment must preserve:

1. every ordered complete legal action and canonical hash;
2. every ordered sparse row, immediate score, and MLX prediction;
3. champion, substantial, and high-confidence estimates bit for bit;
4. all paid-wipe options, values, contingent counts, and selected masks;
5. every realized-hidden action trace, terminal score, and terminal hash;
6. terminal game scores and final state hash;
7. all batch and bridge diagnostics except explicitly identified physical-work
   counters reduced by exact deduplication;
8. zero fallback, clean shutdown, validation, and atomic report writing.

Reference implementations remain available in tests. Any action, value,
random-stream, score, report, or hidden-information difference rejects the
treatment.

## Performance Gates

For each mechanism:

- first demonstrate the intended work reduction with stage counters;
- run matched source builds in balanced order on john2 and john3;
- require the same sign on both hosts and a material combined improvement;
- reject peak-memory or reliability regressions;
- retain only accepted changes before testing the next mechanism.

The final source candidate receives fresh race-free PGO profiles from john2
and john3. Phase 1 collection is authorized only after repeated uncontended
single-Mac runs of the complete frozen contract average no more than
**24.243305 seconds**, with the full correctness vector intact.
