# Realized-Hidden Concurrency Ceiling Preregistration

Status: **closed: rejected**

Date: 2026-06-15

## Question

The accepted full-legal audit spends 125.949535 of 177.686057 seconds in
eight independent realized-hidden terminal continuations. A native sample
shows substantial Rayon and evaluator waiting, while the complete process
averages fewer than four CPU cores.

Before building a multi-trajectory search engine, measure whether independent
exact R600 processes can use otherwise idle Apple CPU/GPU capacity.

## Frozen Diagnostic

Use the accepted exact full-terminal K32/R600 workload on john2 and john3:

- `MCE_LMR=1`;
- `MCE_DIVERSE_PREFILTER=1`;
- four-player AAAAA, no habitat bonuses;
- exact MLX model `legacy-nnue-v4opp-mlx-v1`;
- weights `nnue_weights_v4opp_modal_iter3.bin`;
- one game per process;
- distinct consecutive raw seeds beginning at 34400;
- no rollout truncation, learned leaf, candidate reduction, or fallback.

Measure cohorts of one, two, and four simultaneously launched processes.
Record cohort makespan, sum of individual wall times, games per minute,
maximum RSS, allocator footprint, score vectors, neural work, fallback count,
and clean shutdown. john2 runs cohort order `1,2,4`; john3 runs `4,2,1`.

This is a mechanism diagnostic only. It changes no production code and cannot
be accepted as the Phase 1 optimization by itself.

## Interpretation Gates

Advance an independent-worker or process-pool design only if:

1. every process preserves the exact R600 contract and shuts down cleanly;
2. two-process aggregate throughput is at least 1.50x the single-process
   throughput on both hosts;
3. four-process aggregate throughput is at least 2.50x on both hosts;
4. no host swaps or exceeds 12 GiB aggregate maximum RSS;
5. normalized results agree in direction across john2 and john3.

If four-process throughput is below 1.50x on either host, reject independent
process parallelism as a primary direction and prioritize shared batching,
inference-work elimination, and exact multi-trajectory computation.

Intermediate results authorize no production change and no audit collection.

## Outcome

All 14 processes preserved the exact R600 contract, reproduced the frozen
score and neural-work vectors for their seed, used zero bootstrap samples and
zero policy fallbacks, and shut down cleanly. Neither host swapped, and the
largest conservative sum of per-process maximum RSS was under 0.66 GB.

The throughput gates failed decisively:

| Host | 1 process | 2 processes | 2-process gain | 4 processes | 4-process gain |
|---|---:|---:|---:|---:|---:|
| john2 | 33.963929 s | 58.449131 s | 1.162171x | 97.096545 s | 1.399182x |
| john3 | 34.230646 s | 58.740134 s | 1.165494x | 97.737026 s | 1.400928x |

The registered gates required at least 1.50x at two processes and 2.50x at
four. Four-way throughput was below the explicit 1.50x rejection boundary on
both hosts. Independent same-Mac process parallelism is therefore rejected as
a primary direction. The next exact performance work must share batching and
computation across trajectories rather than duplicate evaluator and rollout
pipelines.

Full result:
[`full-legal-audit-realized-hidden-concurrency-rejection-v1.md`](full-legal-audit-realized-hidden-concurrency-rejection-v1.md).
