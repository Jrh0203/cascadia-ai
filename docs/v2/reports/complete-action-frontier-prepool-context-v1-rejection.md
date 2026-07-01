# Complete-Action Frontier Pre-Pool Context V1

Status: **complete and rejected**

Date: 2026-06-16

Experiment ID: `complete-action-frontier-prepool-context-v1`

## Verdict

The selected ranker's 192-dimensional vector immediately after
`candidate_projection` does not preserve enough information to learn the
open frontier target. Adding exact legacy pooling, richer global moments, or
observable screen-top64 landmarks cannot recover the missing signal.

The preregistered classification is
`candidate_projection_insufficient`. Post-projection head, pooling, and
candidate-relative context treatments are closed.

## Results

| Host | Probe | Best epoch | Train recall | Train exact | Validation recall | Validation exact | Wall |
|---|---|---:|---:|---:|---:|---:|---:|
| john1 | candidate only | 9 | 28.61% | 0.18% | 24.38% | 0.00% | 1,068.26 s |
| john2 | exact legacy context | 18 | 29.07% | 0.18% | 24.02% | 0.00% | 247.71 s |
| john3 | rich moment context | 17 | 29.03% | 0.18% | 24.13% | 0.00% | 704.98 s |
| john4 | screen-top64 context | 16 | 28.93% | 0.18% | 24.09% | 0.00% | 814.42 s |

The frozen train gate required at least 80% target-positive recall and 25%
exact target sets. Every arm failed both requirements. Validation transfer
therefore cannot classify, although all four validation recalls also remained
near 24% and no validation target set was recovered exactly.

## Integrity

- All four hosts used the same 96-file MLX runtime bundle with SHA-256
  `b63b446e1453c889549ced170c6dc7ab5eee2c037ad993a9ff4c3bb0828fd553`.
- Independently generated train and validation caches were bit-identical
  across all four hosts.
- Every probe scored 2,135,111 train candidates and 860,203 validation
  candidates exactly once per evaluation.
- All outputs were finite.
- Maximum process RSS was 2.11 GB and process swaps were zero.
- john1 and john4 passed bit-exact maximum-width reconstruction at 10,854
  candidates.
- The ring replays reproduced every train and validation scientific metric
  exactly.
- Sealed test, gameplay, new teacher compute, cloud, and external compute
  remained closed.

## Throughput

The four distinct probes launched concurrently. The probe-and-replay phase
finished in 1,111.89 seconds and resolved four independent hypotheses, equal
to 12.95 hypotheses per wall-clock hour.

| Host | Probe and replay productive time | Dependency-blocked idle |
|---|---:|---:|
| john1 | 1,106.84 s | 5.05 s |
| john2 | 253.13 s | 832.84 s |
| john3 | 712.21 s | 42.94 s |
| john4 | 837.53 s | 4.72 s |

The aggregate probe-phase productive fraction was 65.42%. No host had
compatible queued work while idle. john2 finished the fastest training arm,
then correctly waited for john1's candidate-only artifact because duplicate
training, extra seeds, and sweeps were prohibited. The remotes generated
their local caches in 34.85-36.64 seconds versus 83.50 seconds on john1,
which is actionable scheduler evidence: future heavy MLX arms should prefer
the faster remotes, while john1 takes lighter work and coordination.

## Consequence

The next experiment must move before the lossy 1,344-to-192 candidate
compression. Compare distinct candidate-factor integration architectures
across the four Macs, one treatment per host. Duplicate seeds remain closed
until an architecture materially fits the open train target and transfers to
validation.

Machine-readable result:
`artifacts/experiments/complete-action-frontier-prepool-context-v1/reports/combined.json`.
