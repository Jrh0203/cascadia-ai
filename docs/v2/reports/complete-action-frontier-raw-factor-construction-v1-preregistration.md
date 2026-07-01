# Complete-Action Frontier Raw Factor Construction V1

Status: preregistered before implementation or metrics.

Date: 2026-06-16

Decision: `docs/v2/decisions/0098-frontier-raw-factor-construction.md`

## Scientific Question

Can any fresh construction from complete raw public state materially fit the
open frontier target before the selected model's insufficient 192-dimensional
candidate factors are formed?

## Frozen Fork

| Host | Arm | Seed | Distinct mechanism |
|---|---|---:|---|
| john1 | fresh entity cross-attention | 2026061624 | candidate-conditioned raw board and staged-market attention |
| john2 | complete raw flat | 2026061621 | unrestricted dense full-state plus candidate construction |
| john3 | exact local board relation | 2026061622 | rotation-canonical tile and wildlife neighborhood relations |
| john4 | explicit market transition | 2026061623 | current-to-staged market and public-supply deltas/interactions |

Every arm uses the same 384-dimensional output width, observable complete-set
and screen-top64 context scorer, target, loss, selector, 20 epochs, AdamW
`3e-4`, and weight decay `1e-4`. Differences are confined to construction from
raw public observables.

## Frozen Gates

- Train: recall at least 80% and exact sets at least 25%.
- Validation: recall at least 50% and exact sets at least 1%.
- Maximum-width forward/backward, finite-score, exact coverage, padding,
  permutation, save/load, source identity, replay identity, memory, cache, and
  no-swap gates must all pass.
- The test split, gameplay, new teacher compute, and external compute remain
  closed.

## Throughput Contract

The four arms are independent experiments, not training replicas. Each Mac
runs one arm; ring replay begins whenever its dependency is ready. No node may
train a duplicate merely to raise utilization. The experiment reports
hypotheses resolved per wall-clock hour, useful device time, idle time with
compatible work queued, and duplicate compute fraction.

The full architecture, selection order, stop rule, and maximum-compute
boundary are frozen in ADR 0098.
