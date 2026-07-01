# Exact-R2 Preverified Vectorized Materialization V1 Interim Result

Date: 2026-06-17

Status: qualification in progress

Current classification:
`complete_validation_speed_gate_passed_full_qualification_pending`

## What Is Established

The first complete john1 validation replay compared the legacy-preverified and
preverified-vectorized materializers over:

- 240 complete decisions;
- 860,203 legal actions; and
- 50,567,539 candidate tokens.

Every frozen parity field produced the same complete digest. There were zero
feature, mask, count, action-identity, transform, or membership failures.

| Metric | Legacy-preverified | Vectorized | Result |
|---|---:|---:|---:|
| Complete elapsed time | 209.670 s | 15.653 s | 13.40x faster |
| Actions per second | 4,103 | 54,956 | 13.40x faster |
| P99 decision latency | 3,187.6 ms | 206.5 ms | 15.44x faster |
| Maximum decision latency | 4,032.1 ms | 266.4 ms | 15.13x faster |
| Exact digest | `3aef145d0c71...` | `3aef145d0c71...` | identical |

This clears the frozen single-machine speed gate and the 410 ms vectorized-P99
gate on the complete validation corpus.

## Memory Measurement Repair

The first complete verifier reported 4,508,139,520 bytes of peak process RSS,
roughly 213 MB over the 4 GiB gate. ADR 0170 identified that the harness held
both complete batches and created additional whole-array byte copies while
hashing. The measurement therefore included avoidable proof-harness residency,
not just either serving path.

The repaired verifier:

1. keeps only one materialized batch resident;
2. hashes each field through a zero-copy contiguous byte view;
3. compares complete field-digest maps; and
4. rematerializes only a mismatching row for element-level diagnostics.

The repaired two-row pilot retained exact parity and measured:

| Metric | Result |
|---|---:|
| Actions | 16,236 |
| Candidate tokens | 1,205,393 |
| P99 speedup | 15.06x |
| Vectorized P99 | 144.8 ms |
| Peak process RSS | 1,858,453,504 bytes |
| Process swaps | 0 |
| System swap delta | 0 bytes |

This pilot validates the repaired verifier design. It does not satisfy the
full-corpus memory gate by itself.

## Qualification Still Required

Promotion remains blocked until all preregistered checks complete:

- complete john1 validation replay under the streaming verifier;
- crossed john2 vectorized-first validation replay;
- complete train-split feature and action-identity parity;
- frozen-C0 prediction parity with maximum absolute error at most `1e-6`;
- zero selected-rank disagreement; and
- at most 4 GiB RSS with nonpositive system swap growth on qualifying runs.

The opportunity-query factorial currently occupies the cluster. These
remaining checks are scheduled as independent work immediately after its
production arms release john2 through john4.

## Research Meaning

The measured serving bottleneck was not intrinsic to exact-R2 information. A
lossless implementation change reduced complete feature materialization P99 by
more than 15x without changing one compared byte. This creates leverage for
every downstream ranker, search experiment, and representation ablation that
uses the exact-R2 substrate.

This experiment is performance-only. It does not establish a score gain,
change the champion, or count as progress toward the 100-point target until a
stronger policy uses the qualified path.
