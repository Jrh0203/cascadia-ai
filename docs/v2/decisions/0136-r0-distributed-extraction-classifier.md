# ADR 0136: R0 Distributed Extraction Classifier

Status: accepted

Date: 2026-06-16

Benchmark ID: `r0-spatial-representation-extraction-v1`

Aggregate ID: `r0-spatial-representation-extraction-aggregate-v2`

Aggregate schema version: `2`

## Context

R0 compares five lossless spatial encodings before any learned-model
tournament:

1. `exact-entity-control`;
2. `hex-radius-6-127`;
3. `hex-radius-5-91`;
4. `hex-radius-4-61`; and
5. `historical-square-21x21-441`.

The preregistered timing protocol requires three independent process
invocations for every host partition. Iterations inside one process are
repeated operations, not independent timing samples. A four-shard run
therefore has 12 classifier inputs, not four.

The distributed result cannot be trusted if reports mix benchmark versions,
datasets, iteration counts, partitions, replica identities, arms, or semantic
payloads. It also cannot treat the three process invocations as three copies
of the source rows. The rows are scientific coverage once; the processes are
timing replication.

This stage measures extraction, packed serialization, deserialization, bytes,
rows, and overflow. It does not train an MLX model and does not evaluate
player strength.

## Decision

`tools/spatial_representation_benchmark_report.py` is the sole merger and
terminal classifier for preregistered R0 extraction reports. Its CLI accepts
one repeated `--report` for each `(shard_index, replicate_index)` pair and
writes one deterministic aggregate JSON:

```bash
.venv/bin/python tools/spatial_representation_benchmark_report.py \
  --report shard-0-replicate-0.json \
  --report shard-0-replicate-1.json \
  --report shard-0-replicate-2.json \
  --report shard-1-replicate-0.json \
  --report shard-1-replicate-1.json \
  --report shard-1-replicate-2.json \
  --report shard-2-replicate-0.json \
  --report shard-2-replicate-1.json \
  --report shard-2-replicate-2.json \
  --report shard-3-replicate-0.json \
  --report shard-3-replicate-1.json \
  --report shard-3-replicate-2.json \
  --output aggregate.json
```

The required replica count is frozen at three. The only accepted indices for
each shard are exactly `0`, `1`, and `2`, each appearing once.

Malformed JSON, missing required fields, non-finite values, invalid digests,
and impossible local accounting are rejected rather than inferred.

## Frozen Rust Contract

Each input uses benchmark schema version `1` and contains:

- `benchmark_id`, `record_count`, `iterations`, `replicate_index`, and
  `selected_arms`;
- `shard.shard_index`, `shard_count`, the exact ordinal rule, record limit,
  total manifested rows, total eligible rows, and loaded rows;
- execution hostname, OS, architecture, logical parallelism, CPU brand,
  memory, and normalized hardware description;
- an ordered dataset list with dataset/schema/split identity, global ordinal
  range, per-shard eligible and loaded rows, and manifest BLAKE3;
- source semantic BLAKE3; and
- one arm record with round-trip status, semantic BLAKE3, timings, packed
  bytes, local and exact rows, overflow rows, and dense-slot accounting.

Dataset roots are host-local provenance and are deliberately excluded from
scientific identity. The ordered dataset manifest BLAKE3 vector is the source
identity. Dataset ID, feature schema, split, game count, row count, and global
ordinal range form the separate dataset identity.

## Replica Identity

The three reports for one shard must have identical non-timing identity:

- benchmark schema and ID;
- record count, iteration count, selected arms, and full shard metadata;
- execution host and hardware provenance;
- source semantic BLAKE3;
- normalized dataset identities and per-shard accounting; and
- every arm field except measured timing fields.

`replicate_index`, validation/read latency, extraction latency, serialization
latency, deserialization latency, and their derived rates are allowed to
differ. Everything else must match exactly. The aggregate records a BLAKE3 of
this timing-free identity for each replica so drift is auditable.

This rule proves that the replicas are repeated measurements of the same
scientific work on the same host partition. Source or arm drift that still
passes each report's local semantic check remains structural incompleteness,
not a timing sample.

## Exact Merge Gates

The aggregate is structurally complete only when:

- every report has schema version `1` and the frozen benchmark ID;
- source-manifest identity and dataset identity are exact across all reports;
- every report uses the same positive iteration count;
- shard count, record limit, total manifested rows, and ordinal rule match;
- reports cover exactly every pair in
  `shard_index in 0..shard_count-1` and `replicate_index in 0..2`;
- no shard/replicate pair is missing or duplicated;
- all three timing-free identities are exact within every shard;
- independently recomputed modulo eligibility matches every dataset and
  shard count;
- loaded rows match the deterministic post-partition record limit;
- every report contains exactly the five required arms, with no duplicate or
  extra arm; and
- every arm's row and iteration counts equal its containing report.

The non-overlap proof is the frozen partition:

```text
global_ordinal % shard_count == shard_index
```

combined with exact dataset order, ordinal ranges, complete shard coverage,
and exact replica groups.

## Semantic Gate

Every arm in every process report must have:

```text
round_trip_verified == true
arm.semantic_blake3 == report.source_semantic_blake3
```

Any violation is a semantic failure, regardless of timing or compression.
Missing arms or replica coverage are structural incompleteness, not evidence
of semantic equivalence.

## Median Invocation

One whole process invocation is selected per shard for absolute aggregation.
The frozen selection criterion is ascending:

```text
exact-entity-control.extraction_ns_per_record
```

With three replicas, rank `1` is the median. Equal criterion values are broken
by ascending `replicate_index`, then report scientific BLAKE3. Selection is
process-wide: all five arms come from the same selected invocation. Choosing
each arm's fastest or median replica independently would break paired
same-process comparisons and is forbidden.

## Aggregation

Semantic and static coverage counts each source row once. Replica reports do
not triple row, byte, overflow, or operation coverage.

For a complete timing result, byte, row, and overflow metrics are taken from
the selected report for each disjoint shard and weighted by represented rows:

```text
weighted_mean = sum(selected_shard_mean * shard_rows) / sum(shard_rows)
```

Overflow positions are summed exactly and divided by unique rows. Local
occupancy is recomputed from weighted active and capacity totals.

Absolute throughput also uses only the selected invocation from each shard:

```text
operations = unique_rows * iterations
records_per_second = sum(selected_operations) / sum(selected_seconds)
ns_per_record = sum(selected_seconds) * 1e9 / sum(selected_operations)
```

If all scientific gates pass but any process timing is invalid, static metrics
remain available from replica `0`, whose timing-free identity is proven equal
to replicas `1` and `2`. Throughput, median selection, variance summaries, and
combined ratios remain unavailable.

## Ratios And Variance

Every process report retains its raw timing fields and its within-process
ratios against both `exact-entity-control` and
`historical-square-21x21-441`:

```text
speedup = reference_ns_per_record / arm_ns_per_record
packed_bytes_fraction = arm_mean_bytes / reference_mean_bytes
```

For every shard and arm, the aggregate reports the distribution of all timing
fields and all within-process ratios across replicas `0`, `1`, and `2`.
Variance is the sample variance with denominator `n - 1`; mean, median,
minimum, maximum, sample standard deviation, and coefficient of variation are
also recorded.

Absolute timing variance is never pooled across unlike hosts. The
cross-shard ratio summary uses only the selected median process from each
shard and computes an operation-weighted geometric mean. Each shard's unique
operations contribute once. The other two replicas establish variance; they
do not masquerade as new row evidence.

Timing evidence is sufficient only when all three phases are positive and
internally reconcile with rows, iterations, nanoseconds per record, and
records per second. Overflow and local-occupancy fractions must also reconcile
with their underlying counts.

## Classification

Classification precedence is fail-closed:

1. `r0_extraction_benchmark_semantic_failure` when any present arm fails
   round-trip or semantic-hash equality;
2. `r0_extraction_benchmark_incomplete` when identity, replica coverage,
   shard coverage, accounting, or required-arm gates fail;
3. `r0_extraction_benchmark_insufficient_performance_evidence` when the
   scientific merge is complete but any process timing is zero or internally
   inconsistent; and
4. `r0_extraction_benchmark_complete` when every gate passes.

The CLI exits `0`, `2`, `3`, or `4` respectively.

`complete` means only that the extraction benchmark is scientifically
mergeable and its timing evidence is valid. The output explicitly records
that no learned model was evaluated, no gameplay strength was measured, no
representation is promoted, and no progress toward 100 mean is claimed.

## Determinism

Reports are normalized and sorted by shard index, replica index, and report
BLAKE3 before validation or arithmetic. Floating-point sums use deterministic
ordered `math.fsum`. Input paths, host-local dataset roots, wall-clock creation
times, and CLI order are excluded from the scientific payload. The complete
payload receives a canonical JSON BLAKE3.

Focused tests cover:

- the complete 4-shard by 3-replica merge;
- unique row accounting without triple counting;
- missing and duplicate replicas;
- source and arm identity drift between replicas;
- source-manifest drift;
- semantic-hash and round-trip failures;
- a missing required arm and iteration drift;
- deterministic median selection and tie breaking;
- exact all-invocation sample variance;
- insufficient timing evidence;
- malformed digests;
- host-local remount invariance;
- deterministic CLI output; and
- byte-identical shuffled input order.
