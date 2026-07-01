# Full-Legal Cross-Request Sparse-Row Reuse Diagnostic Preregistration

Status: **closed - rejected**

Date: 2026-06-15

## Question

Multiplexed trajectory search combines up to eight contemporaneous evaluator
requests but intentionally preserves every request row. The preceding exact
diagnostic found no sparse-row recurrence across waves within one search.

Do different searches in the same multiplexed evaluator batch submit exactly
identical sparse rows that one bounded global deduplication pass could remove?

## Frozen Diagnostic

The diagnostic changes no model value, request row, prediction, search
decision, rollout allocation, random stream, or response range. It only:

- enables `CASCADIA_MULTIPLEX_ROW_REUSE_DIAGNOSTICS=1`;
- examines batches containing at least two search requests;
- uses a temporary `HashMap<&[u16], request_index>` whose hash collisions are
  resolved by complete slice equality;
- counts all rows in coalesced batches;
- counts a duplicate only when the row's first occurrence belongs to another
  request;
- discards the map after each bounded evaluator batch;
- serializes observed and duplicate counts in `batch_diagnostics`.

No row is removed and no prediction is reused in this diagnostic.

## Qualification Scope

Run the frozen seed-60999 late decision:

- audited completed turn: 66;
- realized-hidden completed turn: 66;
- exact K32/R600 champion;
- exact R1200/R4800 confirmation;
- unchanged paid-wipe diagnostic;
- multiplexed realized-hidden continuations;
- Card AAAAA, four players, no habitat bonuses.

## Correctness Gates

1. A unit test proves local duplicates are ignored and exact duplicates from
   another request are counted.
2. The report validates and preserves scores `[96,99,92,102]`, terminal state
   `7b3f520d5441aa2ae9c3d97d87e0cd08299d546a7d1e8398f76fdadedb53fa7d`,
   exact logical diagnostics, zero bootstrap samples, and zero fallbacks.
3. The normalized turn-66 semantic BLAKE3 remains
   `6f19d82622bab6a5a45c6cdf6e1152f99791630436d1a0354d9f629f95089863`.
4. The process shuts down cleanly with zero swaps and stays below 1.5 GiB.

## Advance Gate

Advance to an exact deduplication treatment only if:

1. at least 500,000 coalesced rows are observed;
2. at least 50,000 exact cross-request duplicates are found;
3. duplicates are at least 5% of observed rows;
4. no correctness or memory gate fails.

Any treatment must preserve stable request and row order through an explicit
scatter map, use complete-row equality, remain bounded to one evaluator batch,
and pass separate source, two-host crossover, and switch-free production
gates.

## Outcome

Rejected at the focused diagnostic. All 891,486 rows in coalesced evaluator
batches were observed, but only 247 were exact duplicates from another
request: `0.027707%`, or one duplicate per 3,609 rows. No deduplication
treatment is authorized.

Full evidence:
[`full-legal-audit-cross-request-row-reuse-diagnostic-rejection-v1.md`](full-legal-audit-cross-request-row-reuse-diagnostic-rejection-v1.md).
