# ADR 0144: Corrected Mid-Tail Frozen Parity Campaign

Status: accepted, implemented, and completed

Date: 2026-06-17

Experiment ID: `corrected-mid-tail-frozen-parity-v1`

Research-plan item: F5, C0/T1 pre-training parity

## Context

ADR 0137 introduced the corrected `legacy-mid-v4-fixed-v1` feature schema.
Foundation gates 12 and 13 then published the immutable migrated champion and
proved exact MLX parity on the existing 80-decision Rust fixture.

The F5 preregistration requires a larger and independent closure gate before
training: historical control C0 and migrated treatment T1 must produce
byte-identical float32 predictions on all 200,000 states in the immutable
ADR 0133 activation corpus.

The campaign must be distributable across john1, john2, john3, and john4
without allowing host identity, paths, batch size, invocation order, or timing
to change the scientific result. It must also fail closed. A partial,
overlapping, reordered, malformed, drifted, or numerically approximate
campaign is not evidence.

## Decision

Add a standalone parity implementation in:

```text
python/cascadia_mlx/corrected_mid_tail_parity.py
```

The implementation imports the existing exact Rust-order MLX evaluator and
historical-to-corrected feature remapping from
`python/cascadia_mlx/legacy_nnue.py`. It does not reimplement either neural
forward propagation or the corrected row mapping.

The command-line boundary is:

```text
tools/corrected_mid_tail_parity.py
```

The immutable, non-installing cluster specification generator is:

```text
tools/corrected_mid_tail_parity_queue_spec.py
```

The generator has no `--apply` option and does not import the shared queue
mutation API.

Before launch, review found that the first generated graph referenced mutable
repository source paths on each host. That 14-task graph was invalidated
without execution. The accepted generator now creates a content-addressed
five-file Python source bundle, makes its files and directories read-only,
executes it with bytecode generation disabled, and adds a whole-tree-verified
fanout prerequisite. The executed graph therefore contains 15 tasks.

## Frozen Inputs

### Corpus

| Field | Frozen value |
|---|---|
| Dataset | `legacy-mid-v4opp-activation-v1` |
| Rows | 200,000 |
| Shards | 10 x 20,000 |
| Manifest file BLAKE3 | `7ade2ca310c976c5db9a0e5a840399e226ad8c650e6a4342da845fbb501e0996` |
| Manifest scientific BLAKE3 | `193da520e3ccf3f440dd0f657996d486c1144737abcef1f8399b12ee8b34be92` |
| Payload BLAKE3 | `433ebf13b88f6133efa41f42f3225e13278052b82e3f23a7735401427b5019d8` |

Every shard runner validates the exact manifest file, declared shard set,
selected payload byte count, selected payload BLAKE3, row count, contiguous
game and decision identities, seat, personal turn, phase, policy assignment,
strictly increasing features, and raw-versus-unique feature counts.

### Checkpoints

| Arm | Container | Bytes | BLAKE3 |
|---|---|---:|---|
| C0 historical champion | `NNUE` | 23,134,992 | `9e1d568693274fc537ac4f6d6f729abb1ee8da8330a78d1f78a1f62b733de400` |
| T1 corrected migration | `NNUC` | 23,135,024 | `a3e72314adeb4d62077e43ff071f95b27f979aba17f5026699118b19600263d0` |

Before inference, each process independently proves:

1. C0 rows `0..10561` equal T1 rows `0..10561` byte for byte.
2. C0 rows `10862..11231` equal T1 rows `10561..10930` byte for byte.
3. T1 rows `10930..11231` are IEEE-754 signed zero.
4. Every non-first-layer tensor is byte-identical.
5. Both checkpoint hashes, sizes, dimensions, containers, and head versions
   match the frozen contract.

## Per-Row Evaluation

For every historical sparse row:

1. Reject any activation in `10561..10862`.
2. Preserve base features in `0..10561`.
3. Remap opponent features from `10862..11231` to `10561..10930`.
4. Reject any out-of-range, duplicate, unsorted, malformed, or non-integer
   feature.
5. Evaluate C0 and T1 through `LegacyRustExactSparseNnue`.
6. Reject any non-finite output.
7. Compare the little-endian float32 prediction bytes, not a tolerance.
8. On a mismatch, report the first game, decision, C0 bits, and T1 bits and
   exit nonzero without producing a passing shard report.

Predictions and both sparse streams receive domain-separated incremental
BLAKE3 receipts. Batching is operational only; changing `--batch-rows` must not
change a scientific digest.

## Report Contract

Every report has three top-level concerns:

```text
scientific
scientific_blake3
operational
```

The `scientific` object contains only frozen identities, coverage, exact
mapping receipts, corpus statistics, activation counts, prediction receipts,
and pass gates. Recursive validation rejects scientific keys containing path,
host, device, timestamp, wall-time, seconds, or throughput semantics.

The `operational` object records:

- host and MLX device;
- input and output paths;
- batch size;
- C0, T1, paired-inference, and whole-command timing; and
- rows per second.

Timing is never a parity gate and never enters `scientific_blake3`.

A bounded `--row-limit` run is classified as a smoke and sets
`aggregate_eligible` to false. Only a complete 20,000-row shard can enter the
production aggregate.

## Aggregation

The aggregator requires exactly ten reports and then:

1. verifies every report's scientific hash;
2. rejects partial or smoke reports;
3. requires shard indices `0..9` exactly once;
4. requires contiguous game and row intervals with no overlap or gap;
5. requires one implementation identity, corpus identity, checkpoint pair,
   and mapping receipt across all shards;
6. recomputes aggregate phase, seat, policy, overflow, raw-emission, unique
   activation, duplicate-removal, minimum-width, and maximum-width statistics;
7. requires those totals to equal the immutable ADR 0133 manifest;
8. requires zero discarded-row and corrected-tail activations;
9. requires 200,000 finite, bit-identical predictions; and
10. emits an order-independent aggregate scientific receipt.

Production executes the aggregate twice with forward and reverse report
arguments. The two complete JSON reports must be byte-identical.

## Production Result

The four-host campaign completed on 2026-06-17:

| Measurement | Result |
|---|---:|
| Complete shards | 10 / 10 |
| Covered rows | 200,000 / 200,000 |
| Discarded-row activations | 0 |
| Corrected-tail activations | 0 |
| Non-finite C0 / T1 predictions | 0 / 0 |
| Bit-identical C0/T1 predictions | 200,000 |
| Prediction mismatches | 0 |
| Aggregate prediction receipt, both arms | `6d12482cc25615aad9a64b4a17ff477b9161618aa728e25f1c867587af35b53e` |
| Aggregate scientific BLAKE3 | `ea2c89c36d8fe727ade42b8f17bece1e55b7d220b598d6cef784a61ec6a156bb` |
| Implementation BLAKE3 | `82eff4668432f9acd6ced6268547e70f30f08f1669d88c3605ef340d2dd1ff8a` |
| Forward/reverse aggregate identity | byte-identical |

Classification:

```text
corrected_mid_tail_frozen_parity_complete
```

One initial source-fanout attempt encountered a transient local process-limit
failure before any shard ran. The task was retried after the process table
recovered, then whole-tree verification passed on all four hosts. This is
operational history only and does not enter the scientific result.

## Cluster Allocation

The ten shard tasks use `shard_index mod 4`:

| Host | Shards | Count |
|---|---|---:|
| john1 | 0, 4, 8 | 3 |
| john2 | 1, 5, 9 | 3 |
| john3 | 2, 6 | 2 |
| john4 | 3, 7 | 2 |

This permits four independent MLX jobs immediately. The generated graph then:

1. collects the seven non-john1 reports with remote/local checksum proof;
2. aggregates reports in forward order;
3. aggregates reports in reverse order; and
4. compares aggregate bytes.

The generated artifact is review-only. It does not mutate
`artifacts/cluster/research-queue-v1.json` and does not launch any task.

## Commands

One complete shard:

```bash
.venv/bin/python tools/corrected_mid_tail_parity.py shard \
  --shard-index 0 \
  --corpus-root artifacts/datasets/legacy-mid-v4opp-activation-v1 \
  --historical-checkpoint nnue_weights_v4opp_modal_iter3.bin \
  --corrected-checkpoint \
    artifacts/experiments/corrected-mid-tail-v1/models/blake3/a3e72314adeb4d62077e43ff071f95b27f979aba17f5026699118b19600263d0/nnue_weights_legacy_mid_v4_fixed_v1_init.bin \
  --batch-rows 512 \
  --output \
    artifacts/experiments/corrected-mid-tail-v1/frozen-parity-v1/reports/shard-00000.json
```

All-shard aggregation after reports have been collected:

```bash
reports=()
for shard in {0..9}; do
  printf -v report \
    'artifacts/experiments/corrected-mid-tail-v1/frozen-parity-v1/reports/shard-%05d.json' \
    "$shard"
  reports+=(--report "$report")
done

.venv/bin/python tools/corrected_mid_tail_parity.py aggregate \
  "${reports[@]}" \
  --output \
    artifacts/experiments/corrected-mid-tail-v1/frozen-parity-v1/reports/aggregate-forward.json
```

Generate the immutable review-only queue specification:

```bash
.venv/bin/python tools/corrected_mid_tail_parity_queue_spec.py \
  --output \
    artifacts/experiments/corrected-mid-tail-v1/frozen-parity-v1/queue-spec.json
```

## Implementation Evidence

A complete local run of shard 0 was performed without the cluster:

| Measurement | Result |
|---|---:|
| Rows | 20,000 / 20,000 |
| Game interval | 0 through 249 |
| Discarded-row activations | 0 |
| Corrected-tail activations | 0 |
| Bit-identical predictions | 20,000 |
| Prediction BLAKE3, both arms | `08bf625a33725000d80bca5dce3cbf5cbf7e73b7e8d1f7b842c064436009f662` |
| Scientific BLAKE3 | `19802b453df0e29de9d52c19810547085996940089945ce448da4142569355f6` |
| Paired inference throughput | about 179,613 rows/s |
| Whole-command wall time | about 2.90 seconds |

Repeating all 20,000 rows with batch sizes 512 and 777 produced the same
scientific BLAKE3. Timing differed, as expected, and remained operational.

This is implementation evidence, not the final 200,000-row scientific result.

## Consequences

F5's pre-training C0/T1 parity gate is closed. Fine-tuning may now proceed only
after a corrected-schema activation census proves the new 301-row tail is
actually exercised on source-frozen corrected records. This parity result
proves migration neutrality; it does not prove that the corrected features are
useful or authorize a score claim.
