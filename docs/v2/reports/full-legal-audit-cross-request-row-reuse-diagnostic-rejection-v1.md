# Full-Legal Cross-Request Sparse-Row Reuse Diagnostic Rejection

Status: **rejected**

Date: 2026-06-15

## Result

The exact late-turn diagnostic examined every row in each multiplexed
evaluator batch containing at least two search requests. A borrowed
`HashMap<&[u16], request_index>` used complete slice hashing and equality; a
row counted only when its first occurrence belonged to another request.

| Metric | Result |
|---|---:|
| Coalesced evaluator batches | 189 / 189 |
| Exact searches multiplexed | 104 |
| Coalesced rows observed | 891,486 |
| Exact duplicates from another request | **247** |
| Exact reuse rate | **0.027707%** |
| Rows per duplicate | 3,609 |

The diagnostic exceeded the 500,000-row observation gate but missed the
50,000-row absolute gate by 49,753 rows and the 5% rate gate by 4.972
percentage points. No cross-request deduplication treatment is authorized.

## Correctness

The focused unit test proves that repeats local to one request are ignored and
that exact rows from another request are counted. All 25 feature-enabled
differential library tests passed.

The report validated and reproduced:

- terminal scores `[96,99,92,102]`;
- terminal state
  `7b3f520d5441aa2ae9c3d97d87e0cd08299d546a7d1e8398f76fdadedb53fa7d`;
- 9,288,014 logical rows and 7,198,144 physical rows;
- 4,355 rollout waves and 104,615 rollout samples;
- zero bootstrap samples, zero policy fallbacks, and zero bridge fallbacks.

The normalized turn-66 semantic BLAKE3 is
`6f19d82622bab6a5a45c6cdf6e1152f99791630436d1a0354d9f629f95089863`.
The process used zero swap and 365,641,728 bytes maximum RSS.

## Diagnostic Cost

The instrumented report completed in 31.857941 seconds versus 31.881540
seconds for the uninstrumented treatment, a timing-noise change of -0.074%.
Maximum RSS changed -0.009%. Allocator peak rose 11.346% to 161,907,216
bytes, still far below the 1.5 GiB ceiling.

## Verdict

Reject cross-request sparse-row deduplication. Removing 247 rows from 891,486
would save approximately 0.028% of multiplexed inference work while adding a
hash table, global row map, prediction scatter, and another correctness
surface to every batch.

Both exact row-reuse directions are now closed:

- within one search across waves and halving rounds: 0 / 6,116,501;
- between requests in one multiplexed batch: 247 / 891,486.

The next performance work must optimize the inference kernel, feature and
template preparation, opponent simulation, or search workload itself rather
than cache effectively unique rows.

Machine-readable evidence:
`docs/v2/reports/full-legal-audit-cross-request-row-reuse-diagnostic-rejection-v1.json`.

The local archive is under
`artifacts/performance/full-legal-audit-cross-request-row-reuse-diagnostic-v1/`.
