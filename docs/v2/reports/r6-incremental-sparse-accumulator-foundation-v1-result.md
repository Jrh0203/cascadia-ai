# R6 Incremental Sparse Accumulator Foundation V1 Result

Date: 2026-06-17

ADR: 0158

Classification: `r6_incremental_apply_undo_promoted`

Outcome: passed

## Result

R6 passed exact apply, exact undo, and throughput gates.

| Metric | Result |
|---|---:|
| Positions | 320 |
| Complete actions | 506,425 |
| Exact apply checks / failures | 506,425 / 0 |
| Exact undo checks / failures | 506,425 / 0 |
| Authoritative apply time | 127.256724255 s |
| Incremental apply plus undo time | 2.161846753 s |
| Speed ratio | 58.864821x |
| Accumulator median / P99 bytes | 3,092 / 5,262 |
| Actions per position median / P99 | 864 / 11,016 |

## Interpretation

R3 action edits contain enough information to maintain the active sparse
state exactly without reconstructing every sibling afterstate. Stable
member-derived component keys remove the only false mismatch discovered
during calibration: traversal-local component-number renumbering.

The 58.9x ratio is a pure transition-mechanics gain. It includes undo cost and
uses identical complete-action sets. Exact structural parity was tested
outside the timing loop for every action.

## Decision

Promote the accumulator into the R6 serving path. The next matched benchmark
must measure:

- incremental feature updates;
- sparse NNUE delta evaluation;
- relational rescoring over retained candidates;
- complete-decision latency; and
- end-to-end action throughput.

## Provenance

- bundle:
  `3d7f8a563319ba6636a3fb06d3db33fd846f8dbfdd70beeabd3d5a0d509eeee3`
- scientific report:
  `4a73bc76cb7d4d129bc3a875fe148523ec85eb64991a06411490916cf55bc8c7`
- host: john2
- seeds: `5,210,000-5,210,003`

## Claim Boundary

This result is mechanical leverage, not a score claim.
