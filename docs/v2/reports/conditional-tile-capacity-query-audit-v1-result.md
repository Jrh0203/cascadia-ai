# Conditional Tile Capacity and Query Audit V1 Result

Date: 2026-06-16

Experiment ID: `conditional-tile-capacity-query-audit-v1`

Classification: **`full_data_scale_or_optimization_insufficient`**

## Memorization Arms

| Arm | Reference recall | Best recall | Best exact | Steps | Peak RSS |
|---|---:|---:|---:|---:|---:|
| Baseline 16 | 75.40% | 100.00% | 100.00% | 400 | 1.91 GiB |
| Baseline 256 | 73.81% | 100.00% | 100.00% | 3,400 | 1.96 GiB |
| Attention 256 | 73.81% | 99.95% | 98.83% | 4,000 | 1.96 GiB |

## Frozen Error Anatomy

| Split / width | ADR 0115 | ADR 0116 | Delta |
|---|---:|---:|---:|
| Train overall | 72.60% | 77.21% | +4.61% |
| Validation overall | 66.57% | 70.59% | +4.02% |
| Validation 33-64 | 63.33% | 64.03% | +0.69% |
| Validation 65-96 | 61.15% | 66.82% | +5.67% |
| Validation 97-128 | 57.30% | 62.34% | +5.04% |
| Validation 129+ | 68.04% | 70.74% | +2.71% |

## Input Sensitivity

Validation recall drop from the intact ADR 0116 checkpoint:

- permuted query context: `+0.59%`;
- permuted parent state: `+0.00%`;
- zero tile factor: `+0.47%`;
- zero local geometry: `-0.50%`;
- zero descendant summaries: `+27.47%`.

## Integrity

- The 16-query cohort is the exact prefix of both 256-query arms.
- Baseline and attention medium arms used identical queries.
- All four arms ran on distinct Macs with zero duplicate discovery compute.
- Sealed test, gameplay, validation-driven selection, new teacher compute,
  cloud, and external compute remained closed.

## Failed Gates

- None.

## Decision

The unchanged ranker fits 256 representative hard queries. The next treatment should target full-data sampling, schedule, or scale.
