# ADR 0167: Exact-R2 Preverified Vectorized Materialization

Status: accepted; implementation qualification in progress

Date: 2026-06-17

Experiment: `exact-r2-preverified-vectorized-materialization-v1`

Protocol: `exact-r2-vectorized-materialization-parity-v1`

Research-plan items: common performance gates, R2, R6

## Context

The final exact-R2 C0 run from ADR 0161 isolated a severe implementation
bottleneck:

```text
model-only complete-decision P99        183.137 ms
feature-materialization P99           4,087.113 ms
combined model plus materialization   4,270.355 ms
exact R6 apply/undo P99                  79.741 ms
```

The model is not rebuilding a 441-cell tensor. The delay is in the Python
construction of sparse exact-R2 candidate afterstates. For every candidate,
the legacy loader independently:

1. copies the parent sparse token stream;
2. applies the candidate's D6 transform;
3. recenters every coordinate;
4. removes changed parent objects;
5. appends exact added objects;
6. sorts and hashes the resulting multiset; and
7. expands the compact stream into an 80-channel float tensor.

Full validation contains 860,203 actions over 240 decisions. A decision has
306 to 9,108 actions, but only 19 to 156 distinct `(D6 transform, center)`
frames. Repeating the parent transform once per candidate is therefore pure
bookkeeping waste.

Two explicitly disclosed development rows, 69 and 225, were used only to
profile and validate feasibility before freezing this protocol. On those rows,
the prototype was feature-exact and was 50.99x and 19.90x faster than the
fully verified legacy path. It was 19.68x and 16.78x faster than a legacy path
that was already allowed to skip redundant hashes. The remaining 238
validation rows and the complete train split were unopened by the prototype
comparison before this ADR was frozen.

## Decision

Add a distinct `preverified-vectorized` materialization mode while retaining
`verified-per-candidate` as the independent oracle.

The optimized mode is legal only when the bound dataset carries the exhaustive
open-data verification proof. It cannot be combined with per-candidate hash
verification and cannot be selected for a compact R3 treatment arm.

For each decision it:

1. groups candidates by exact `(transform_id, center_q, center_r)`;
2. transforms and recenters the parent once per unique frame;
3. constructs all removal masks with vectorized indexed writes;
4. computes compact kept-token destinations with cumulative sums;
5. places all exact additions with vectorized ragged offsets;
6. writes the same type, operation, and payload channels directly into the
   padded float tensor; and
7. preserves original candidate and token order.

No model parameter, feature value, action order, target, random stream, or
search rule changes.

## Trust Boundary

Skipping repeated hashes is authorized only by all of:

- a valid 64-hex open-data proof ID;
- exhaustive dataset-to-R3 action identity verification;
- exhaustive dataset-to-S1 candidate identity verification;
- the immutable R3 cache's full mechanical checks; and
- exact treatment-versus-oracle tensor parity in this experiment.

Callers without the proof continue to use the verified oracle. The oracle is
not removed after promotion.

## Frozen Qualification

The experiment must cover:

```text
open train groups                  560
retained train actions        280,012
open validation groups             240
validation actions             860,203
total compared actions        1,140,215
all 12 D6 transforms exercised
```

For every group compare:

- candidate token features;
- candidate token masks;
- candidate token counts;
- canonical transform IDs;
- base legal-action hashes and order;
- selected and champion indices; and
- all noncandidate public inputs.

The final C0 checkpoint is then replayed over complete validation using both
paths. Candidate scores, uncertainties, selected ranks, and the complete
decision panel must agree within the frozen float32 tolerance.

## Performance Protocol

Measure both legacy-preverified and vectorized paths in fresh processes.
Use all 240 validation decisions and report per-decision samples, action
counts, token counts, P50, P95, P99, maximum, actions/s, process RSS, active
MLX memory, and swap delta.

Cross order on two hosts:

| Host | First pass | Second pass |
|---|---|---|
| john1 | legacy-preverified | vectorized |
| john2 | vectorized | legacy-preverified |

john3 owns complete C0 prediction parity. john4 owns deterministic training
schedule parity and peak-memory confirmation. These jobs may run only when
their already-authorized opportunity-query work is not waiting for the host.

## Gates

Promotion requires every condition:

```text
feature parity failures = 0
action identity failures = 0
C0 prediction max absolute error <= 1e-6
C0 selected-rank disagreements = 0
validation materialization P99 speedup >= 10.0x on john1
validation materialization P99 speedup >= 10.0x on john2
vectorized materialization P99 <= 410 ms
peak process RSS <= 4 GiB
system swap delta <= 0
```

The 410 ms limit is the rounded 10x boundary from the frozen 4,087.113 ms C0
materialization P99. Passing this ADR removes one bottleneck; it does not waive
the stricter 250 ms complete-decision serving limit in ADR 0166.

## Consequences

A pass authorizes the vectorized loader for subsequent exact-R2 research and
serving qualification. It also establishes unique-frame materialization as
the reference input path for later compact categorical projection and
duplicate-afterstate spatial-encoding experiments.

A parity failure rejects the treatment immediately. A speed failure retains
the implementation only as diagnostic evidence and leaves the verified path
as the sole authorized loader.

