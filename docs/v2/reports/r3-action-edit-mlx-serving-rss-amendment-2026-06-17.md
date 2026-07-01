# R3 Action-Edit MLX Serving-RSS Amendment

Date: 2026-06-17  
Experiment: `r3-action-edit-mlx-comparison-v1`  
Protocol: `r3-action-edit-mlx-matched-comparison-v1`  
ADR: `0150`  
Status: frozen before production optimization

## Why this amendment exists

The complete R3 cache was exported and verified before production. The first
full-corpus campaign-process check then exposed a measurement error:

- the R3 sidecar contains about 8.3 GiB of memory-mapped tensors;
- exhaustive checksum, semantic, source-action, and S1 identity verification
  intentionally walks those tensors;
- the verification process reached about 4.8 GiB RSS; and
- `resource.getrusage(RUSAGE_SELF).ru_maxrss` is a lifetime high-water mark.

The original in-process benchmark therefore would have reported the maximum
RSS of cache verification, training, validation, and serving combined. That is
not the serving RSS named by ADR 0150, and it could fail the 4 GiB gate even
when the serving implementation used far less memory.

No production optimizer had started when this was discovered. The cache,
dataset, smoke, model, optimizer, schedule, quality metrics, promotion
thresholds, and host assignments remain unchanged.

## Frozen correction

### Exhaustive verification remains mandatory

Every production host preflight must still:

1. checksum every R3 and S1 tensor;
2. run every sidecar semantic invariant;
3. bind every open train and validation group to its public-state identity;
4. compare every required action hash against the graded-oracle source;
5. verify the S1 full-group candidate identities;
6. verify the immutable source bundle and exporter identity; and
7. emit a content-addressed preflight report.

The cluster reserves 10 GiB for each preflight. All four hosts have 16 GiB of
physical memory. Preflight is an integrity
workload, not a serving-memory measurement.

### Proof-aware production loading

The optimizer process may skip repeated tensor checksums and full semantic
scans only after it validates all of the following:

- the production authorization content address;
- its assigned host preflight content address;
- the immutable source identity;
- the R3 and S1 cache content addresses and manifest hashes;
- both open dataset identities and manifest hashes;
- the frozen protocol and arm-to-host assignment; and
- the shared open-data verification content address.

Even in this mode, the loader validates every manifest envelope, tensor path,
tensor size, group header, candidate count, selected/champion index, public
state hash, and complete group coverage. It retains copied group metadata and
one shard mapping, not one action-hash view per decision.

The exhaustive path remains the default. Proof-aware loading is unavailable
without a valid 64-character verification content address and a validated
production launch control.

### Isolated serving measurement

After training and complete open-validation scoring, each arm must:

1. save and checksum its final checkpoint;
2. create a content-addressed serving-benchmark request binding the checkpoint,
   arm, open-data proof, candidate chunk, warmup count, steady count, and
   complete-decision rows;
3. start a fresh Python process;
4. revalidate all content-addressed manifests and checkpoint bytes;
5. load the exact final model checkpoint on the MLX GPU;
6. bind validation through the verified lightweight loader;
7. run the frozen 256-action and 20-decision benchmark; and
8. emit a content-addressed result.

`peak_process_rss_bytes` is the fresh worker's lifetime `ru_maxrss`. It includes
Python/MLX startup, lightweight dataset mappings, checkpoint loading, compile,
warmup, and steady serving. It excludes cache export, exhaustive preflight,
optimizer state history, training batches, and full validation.

Every production performance report must include:

- `measurement.isolated_process == true`;
- serving request and result content addresses;
- final checkpoint model BLAKE3;
- open-data verification content address;
- `verification_source == "cluster-preflight"`; and
- worker runtime identity.

The report classifier rejects older in-process RSS evidence.

## Empirical verification

Full open-corpus loader measurements on john1:

| Path | Peak RSS | Bind time |
|---|---:|---:|
| Previous full source-action binding, semantic scan disabled | 1,126,694,912 B | 2.61 s |
| Proof-aware header binding | 365,527,040 B | 0.38 s |

The first campaign observation caught about 4.8 GiB resident while the
verification was still running. A subsequent instrumented exhaustive run
recorded 6,953,697,280 B after R3 semantic verification and 7,595,130,880 B
after both complete source-action bindings. That behavior remains acceptable
inside the 10 GiB preflight reservation and confirms that it must not be
reported as serving RSS.

One-step radius-1 smoke with a fresh serving worker:

| Metric | Result |
|---|---:|
| Peak process RSS | 361,316,352 B |
| Peak active MLX memory | 134,075,974 B |
| Process swaps | 0 |
| Fixed-chunk throughput | 40,760.55 scores/s |
| Complete-decision throughput | 27,804.09 scores/s |
| Complete-decision P99 | 39.50 ms |

Artifact:

`artifacts/experiments/r3-action-edit-mlx-comparison-v1/smoke-runs/rss-boundary-v2c-report.json`

The smoke passes every absolute serving gate with substantial margin.

## Invariants and nonchanges

- The 4 GiB active-memory and RSS gates do not change.
- The 20,000 scores/second and 250 ms P99 gates do not change.
- All quality-noninferiority and material-efficiency thresholds do not change.
- All four arms still use identical data, schedules, optimization, capacity,
  losses, and evaluation.
- The sealed test split and gameplay remain unopened.
- This amendment makes no strength, score, promotion, or 100-point claim.
