# Exact Card A Wildlife Score Context Rejection

Status: **rejected and removed**

Date: 2026-06-15

## Hypothesis

The qualified AAAAA search repeatedly rescored a complete wildlife category
after hypothetical placements. The treatment constructed one exact immutable
Card A score context per parent board and updated only the affected Bear, Elk,
Salmon, Hawk, and Fox facts for each placement.

Candidate enumeration and order, habitat and potential arithmetic, feature
rows, row deduplication, MLX requests, random streams, search allocation, and
the K32/R600 full-terminal benchmark contract were unchanged. Non-AAAAA
scoring continued through the general scorer.

## Exactness

The specialized scorer matched the independent complete scorer across an
adversarial component-merging fixture and every legal existing or newly placed
wildlife slot on every intermediate board in 16 complete seeded four-player
AAAAA games. The comparison covered the placed category and the Fox side
effect for all five wildlife types, including Bear pair changes, Elk line
ties, valid and branching Salmon runs, Hawk isolation, Fox masks, keystones,
disconnected components, and no-wildlife outcomes.

The complete default and `mid-features,v4-opp` library suites passed before
timing. Every diagnostic and crossed source run reproduced:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches;
- 6,121,807 logical and 5,062,305 physical neural rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps, zero policy fallbacks, and clean shutdown.

After removal, the complete workspace library suites passed again. The legacy
AI contributed 84 default tests and 85 `mid-features,v4-opp` tests; legacy core
contributed 125 tests in each configuration. A fresh release build is
byte-for-byte identical to the retained pre-experiment parent-context source
binary, SHA-256
`786351ea84e4b2674e81f2ade87d0596e47a8a3b21be2f336dc9e6ff62c4cd94`.

## Source-Level Screen

Matched non-PGO binaries were crossed in balanced opposite orders with two
measurements per binary on john2 and john3.

| Host | Control mean | Treatment mean | Speedup |
|---|---:|---:|---:|
| john2 | 15.207208 s | 15.169689 s | 1.00247x |
| john3 | 15.067608 s | 14.931556 s | 1.00911x |
| Combined | **15.137408 s** | **15.050622 s** | **1.00577x** |

The treatment improved both hosts, but its combined improvement was only
0.573%, below the preregistered 1.00% source-promotion floor.

The targeted diagnostic stages did show the intended mechanism:

| Host | Control aggregate | Treatment aggregate | Improvement |
|---|---:|---:|---:|
| john2 | 8,200.603 ms | 8,104.891 ms | 1.167% |
| john3 | 8,266.984 ms | 8,050.482 ms | 2.619% |

However, john2 candidate preparation regressed by 0.213%, violating the
registered no-individual-stage-regression gate. Combined maximum resident set
size was effectively flat at 102.969 MB control versus 102.949 MB treatment,
but the allocator-reported peak memory footprint increased 9.162%.

## Verdict

Reject before PGO and remove. The exact specialization reduces repeated
wildlife scoring, but the end-to-end gain is too small, one target substage
regresses on john2, and its retained context increases allocator footprint.
Fresh PGO would spend promotion effort on a treatment that already failed its
registered source gate.

The accepted parent-afterstate PGO champion remains unchanged at 15.018871
seconds, or 9.390x versus the 141.027296-second reference. The 10x threshold
remains 14.102730 seconds, leaving 0.916141 seconds or 6.100%.

Machine-readable evidence:
[`exact-card-a-wildlife-score-context-rejection-v1.json`](exact-card-a-wildlife-score-context-rejection-v1.json).

