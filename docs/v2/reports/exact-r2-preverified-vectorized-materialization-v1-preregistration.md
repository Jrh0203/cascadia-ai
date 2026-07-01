# Exact-R2 Preverified Vectorized Materialization V1 Preregistration

Date: 2026-06-17

ADR: 0167

Experiment: `exact-r2-preverified-vectorized-materialization-v1`

Protocol: `exact-r2-vectorized-materialization-parity-v1`

Status: frozen before complete-corpus parity and performance measurement

## Question

Can exact-R2 candidate features be materialized at least ten times faster by
factoring repeated coordinate frames and vectorizing exact edits, without
changing one input value or one C0 decision?

## Frozen Inputs

```text
R3 cache:
  0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156
S1 cache:
  2323ead43b1bff7a506ecef4b8bd4793cebe4d53c6f8940b03404573ca5e6c15
relational cache:
  d4f8e2eb83db237b136fd478b73802544938c36adf77db0bf40f2b3276181bef
open-data proof:
  795b10243af6f87f7eb2f333547f53a83a4a5fd938615d61f6ab5d9e3c0fdeee
C0 checkpoint model:
  eadcfbd5d0f02d642e7003431809b9ae8c41f0c3faf12c57d6da84a18acc5b89
C0 checkpoint manifest:
  a7e31e2713a2afd642f7143fee3d9071c9776ee88ca7bbed61564d6e7b12b9d3
```

The open train and validation datasets, candidate cohorts, labels, model
weights, and exact R6 implementation are unchanged.

## Disclosed Pilot

Rows 69 and 225 were development-only profiling rows.

| Row | Actions | Verified legacy | Preverified legacy | Vectorized | Exact |
|---|---:|---:|---:|---:|---|
| 69 | 7,128 | 7.579 s | 2.926 s | 0.149 s | yes |
| 225 | 9,108 | 5.844 s | 4.927 s | 0.294 s | yes |

These measurements are not the production result and cannot satisfy a gate.

## Exactness Matrix

Complete train and validation parity compares:

1. float candidate feature tensors bit for bit;
2. masks, counts, and canonical transforms bit for bit;
3. legal action hashes and source indices bit for bit;
4. selected and champion membership bit for bit;
5. every unchanged parent, supply, market, player, and global tensor; and
6. all C0 predictions and uncertainties within `1e-6`.

Any mismatch records split, group row, candidate, token, channel, expected
value, and observed value before terminating classification.

## Performance Matrix

The production comparison uses the preverified legacy path as the conservative
control. Fully verified legacy timing is retained from the frozen C0 report as
an operational diagnostic.

```text
rows: all 240 validation decisions
candidate coverage: all 860,203 actions
process isolation: fresh
precision: float32
MLX device: Apple GPU
per-candidate hashes: disabled for both compared arms
open-data proof: required for treatment
```

john1 runs control then treatment. john2 runs treatment then control. Host
reports remain separate and are also aggregated by complete decision.

## Success

```text
zero feature and action-identity failures
zero C0 selected-rank disagreements
maximum prediction error <= 1e-6
P99 speedup >= 10x on both crossed hosts
treatment P99 <= 410 ms
RSS <= 4 GiB
swap delta <= 0
```

The experiment is a pure-performance qualification. It cannot claim a score
gain or progress toward 100 points by itself.

## Classification

```text
exact_r2_vectorized_materialization_promoted
exact_r2_vectorized_materialization_parity_failure
exact_r2_vectorized_materialization_speed_failure
exact_r2_vectorized_materialization_memory_failure
exact_r2_vectorized_materialization_cross_host_inconsistent
exact_r2_vectorized_materialization_structurally_invalid
```

