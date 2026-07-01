# Exact Greedy Habitat Rank Cache Rejection

Date: 2026-06-15

Status: **rejected and removed**

## Hypothesis

The qualified greedy rollout opponent evaluates the same drafted tile across
paired and independent wildlife combinations. Its top-eight habitat scan
appears pure, so computing that ranking once per market tile could remove most
of the hottest `top_habitat_placements` work.

## Result

The focused greedy parity tests passed, and the seed-34400 score vector remained
`[102, 96, 92, 95]`. The full exact trace did not:

| Diagnostic | Accepted path | Rank cache | Delta |
|---|---:|---:|---:|
| Logical neural rows | 6,121,807 | 6,122,242 | +435 |
| Physical neural rows | 5,062,305 | 5,062,471 | +166 |
| Rollout waves | 3,716 | 3,716 | 0 |
| Rollout samples | 46,207 | 46,207 | 0 |
| Policy fallbacks | 0 | 0 | 0 |

Unprofiled treatment time was 15.071514 seconds on john2 and 15.503854
seconds on john3. Runtime is irrelevant to acceptance because the trace
already failed exactness.

## Explanation

Temporary tile place/undo operations restore explicit union-find merge records
but intentionally retain some path-compression history. A habitat ranking is
mathematically unchanged by that history, yet moving the scan across those
mutation boundaries changes later stable ties. That changes candidate
afterstates and neural row counts even when the final four scores happen to
match.

## Decision

Reject and remove the cache. Future exact optimization of this hotspot must
reduce work *inside each original scan boundary* and preserve every replay,
request, and tie in its original order.
