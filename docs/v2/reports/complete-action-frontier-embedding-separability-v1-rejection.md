# Complete-Action Frontier Embedding Separability V1 Rejection

Status: **complete; rejected**

Date: 2026-06-16

Experiment ID: `complete-action-frontier-embedding-separability-v1`

## Verdict

The selected ADR 0089 model's final 192-dimensional candidate representation
does not contain enough separable target structure for either a new linear
head or a shallow nonlinear head. The preregistered classification is
`frozen_representation_insufficient`.

Another loss swap, learning-rate treatment, or output-head-only pilot is
closed. The next experiment must bypass or replace the collapsed
representation while keeping the trunk frozen long enough to localize where
the usable observable signal is lost.

## Results

| Probe | Train recall | Train exact sets | Validation recall | Validation exact sets | Gate |
|---|---:|---:|---:|---:|---|
| Linear `192 -> 1` | 22.48% | 0% | 17.28% | 0% | failed |
| Nonlinear `192 -> 128 -> 1` | 24.67% | 0% | 19.91% | 0% | failed |

The linear gate required 60% train recall and 5% exact train sets. The
nonlinear gate required 80% train recall and 25% exact train sets. Neither was
close. The nonlinear probe improved validation exact-winner recall to 61.67%
and retained regret to 0.131922, but those secondary metrics do not rescue a
probe that cannot fit the open training target.

## Integrity

- john2 exported all 2,135,111 train actions once in 25.97 seconds.
- john3 exported all 860,203 validation actions once in 10.46 seconds.
- No host repeated trunk inference.
- john1 and john4 independently reconstructed both original heads bit-for-bit
  from the exported embedding on the 10,854-action maximum-width decision.
- john4 reopened and hashed both transferred caches, then reproduced both
  probe reports exactly.
- All four hosts used the same 94-file MLX source bundle.
- Maximum probe RSS was 3.43 GB; process swaps were zero.
- Sealed test, gameplay, new teacher compute, cloud, and external compute
  remained closed.

## Throughput

The full cluster audit completed in 294.24 seconds wall-clock. Cache extraction
itself took 26.39 seconds wall and the two probes ran on separate hosts. The
primary idle source was the one-time 2.26 GiB cache relay between hosts, not
training. The next experiment must reuse these caches or transfer one compact
raw-observable sidecar rather than repeat trunk inference.

john4's first portability-test command failed before testing because the new
test file had not been synchronized. The file was synchronized immediately
and all eight focused tests passed on the registered rerun. No scientific
artifact depended on the failed launch.

## Next Mechanism

Run a frozen-trunk raw-observable bypass audit. Reuse the exact cached
192-dimensional embedding and append a compact sidecar containing the
original observable action and prior features. Independently test:

- a raw-observable probe, which asks whether the target is separable without
  the learned trunk; and
- an embedding-plus-raw bypass probe, which asks whether board context from
  the trunk becomes useful once the original action/rank signal cannot be
  discarded.

Only a probe that materially fits train may authorize a deployable bypass
head. If the combined probe also fails, the next step is a new trunk
representation rather than another head or objective treatment.

Machine-readable result:
`artifacts/experiments/complete-action-frontier-embedding-separability-v1/reports/combined.json`.
