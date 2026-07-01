# Realized-Hidden Concurrency Ceiling Rejection

Status: **rejected**

Date: 2026-06-15

## Experiment

The accepted full-legal audit spends 125.949535 of 177.686057 seconds in
eight independent realized-hidden terminal continuations. Before building a
shared multi-trajectory engine, a preregistered diagnostic measured whether
otherwise idle Apple CPU/GPU capacity could be recovered by launching one,
two, or four complete exact R600 processes on a single Mac.

john2 ran cohorts in `1,2,4` order. john3 ran `4,2,1`. Every process used a
distinct seed beginning at 34400, the same production binary, exact
K32/R600 search, `MCE_LMR=1`, `MCE_DIVERSE_PREFILTER=1`, the same MLX model
and weights, and no rollout truncation, learned leaf, candidate reduction, or
fallback.

## Exactness

All 14 runs passed:

- exact R600 search and clean evaluator shutdown;
- zero bootstrap samples and zero policy fallbacks;
- identical score, diagnostic, and neural-work vectors for every repeated
  seed across host and cohort size;
- identical executable, source, model, and weights digests;
- zero swaps.

The four frozen treatment score vectors were:

| Seed | Scores |
|---:|---|
| 34400 | `[102,96,92,95]` |
| 34401 | `[97,94,97,98]` |
| 34402 | `[92,97,95,97]` |
| 34403 | `[94,95,91,94]` |

## Result

| Host | Processes | Cohort wall | Games/minute | Throughput vs single | Aggregate max RSS |
|---|---:|---:|---:|---:|---:|
| john2 | 1 | 33.963929 s | 1.766580 | 1.000000x | 231,194,624 B |
| john2 | 2 | 58.449131 s | 2.053067 | 1.162171x | 374,079,488 B |
| john2 | 4 | 97.096545 s | 2.471767 | 1.399182x | 656,900,096 B |
| john3 | 1 | 34.230646 s | 1.752815 | 1.000000x | 231,489,536 B |
| john3 | 2 | 58.740134 s | 2.042896 | 1.165494x | 374,685,696 B |
| john3 | 4 | 97.737026 s | 2.455569 | 1.400928x | 656,769,024 B |

The preregistered gates required at least 1.50x aggregate throughput at two
processes and 2.50x at four on both hosts. Both gates failed. Four-process
throughput was also below the explicit 1.50x rejection boundary on both
hosts, with only about 35% parallel efficiency.

Memory was not the limiting resource: the conservative sum of per-process
maximum RSS remained below 0.66 GB, far under the 12 GiB guardrail, and every
`/usr/bin/time -l` record reported zero swaps. Instead, each game's wall time
inflated from about 34 seconds alone to 58 seconds at two-way concurrency and
95-98 seconds at four-way concurrency. The processes contend for the same
native rollout and MLX execution capacity.

## Verdict

Reject independent same-Mac process parallelism as a primary route to the
teacher's 10x gate. Continue distributing independent games across john1,
john2, and john3, but keep one exact search process per Mac for this workload.

The next performance design must share work across the eight trajectories:
one coordinated native scheduler, one evaluator, cross-trajectory MLX
batches, and exact inference-work elimination where identical ordered sparse
prefixes or states permit it. No production code changed in this experiment.

Machine-readable evidence:
[`full-legal-audit-realized-hidden-concurrency-rejection-v1.json`](full-legal-audit-realized-hidden-concurrency-rejection-v1.json).

The complete raw evidence archive is preserved under
`artifacts/performance/full-legal-audit-realized-hidden-concurrency-v1/`.
