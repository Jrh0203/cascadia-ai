# Full-Legal Cross-Wave Sparse-Row Reuse Diagnostic Preregistration

Status: **closed - rejected**

Date: 2026-06-15

## Question

After request multiplexing, the frozen full-legal teacher still evaluates
44,903,952 physical sparse rows. Within one independent K32/R600 search,
sequential-halving rounds and rollout waves may revisit exactly identical
NNUE rows that the existing within-wave deduplicator cannot reuse.

How much exact cross-wave row recurrence exists, and is it large enough to
justify a bounded collision-checked prediction cache?

## Frozen Diagnostic

The diagnostic changes no model value, search decision, row order, rollout
allocation, random stream, or evaluator request. It only:

- enables `CASCADIA_NNUE_ROW_REUSE_DIAGNOSTICS=1`;
- retains exact sparse rows already sent for inference within each independent
  teacher search;
- uses the existing 64-bit fingerprint only as a lookup bucket;
- requires complete `Vec<u16>` equality before recording a repeat;
- resets the tracker between independent searches;
- records physical rows observed and exact rows repeated across later waves or
  halving rounds;
- serializes both counters in `batch_diagnostics`.

No prediction is reused in this diagnostic.

## Qualification Scope

Run the frozen seed-60999 late decision:

- audited completed turn: 66;
- realized-hidden completed turn: 66;
- exact K32/R600 champion;
- exact R1200/R4800 confirmation;
- unchanged paid-wipe diagnostic;
- multiplexed realized-hidden continuations;
- Card AAAAA, four players, no habitat bonuses.

The late decision is the cheapest complete screen of the mechanism. If it
passes the advance gate, confirm the rate on the complete turns 12/39/66
contract before implementing a cache.

## Correctness Gates

1. A forced fingerprint-collision test proves distinct rows are not counted
   as repeats.
2. The report validates and preserves scores `[96,99,92,102]`, terminal state
   `7b3f520d5441aa2ae9c3d97d87e0cd08299d546a7d1e8398f76fdadedb53fa7d`,
   exact logical diagnostics, zero bootstrap samples, and zero fallbacks.
3. The normalized semantic report remains identical to the frozen turn-66
   treatment report.
4. The process shuts down cleanly with zero swaps and stays below 1.5 GiB.

## Advance Gate

Advance to a complete-contract diagnostic only if:

1. at least 100,000 physical rows are observed by the tracker;
2. exact repeated rows are at least 5% of observed rows;
3. no correctness or memory gate fails.

Advance from the complete diagnostic to a cache implementation only if:

1. exact repeated rows are at least 8% of observed rows;
2. the absolute repeated-row count is at least 2,000,000;
3. projected neural work reduction is material against the remaining 5.931x
   teacher gap.

Any later cache must use collision-checked complete-row equality, remain
bounded per search, preserve stable prediction order, and pass its own
preregistered two-host performance gates.

## Outcome

Rejected at the focused diagnostic. The tracker observed 6,116,501 physical
rows and found zero exact cross-wave repeats. The normalized focused report
was
`6f19d82622bab6a5a45c6cdf6e1152f99791630436d1a0354d9f629f95089863`,
the existing frozen turn-66 digest. The earlier full-contract digest in this
gate was a scope mismatch and has been corrected above; it does not affect
the zero-reuse rejection.

Full evidence:
[`full-legal-audit-cross-wave-row-reuse-diagnostic-rejection-v1.md`](full-legal-audit-cross-wave-row-reuse-diagnostic-rejection-v1.md).
