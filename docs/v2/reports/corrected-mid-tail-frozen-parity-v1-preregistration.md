# Corrected Mid-Tail Frozen Parity V1 Preregistration

Date: 2026-06-17

Experiment ID: `corrected-mid-tail-frozen-parity-v1`

Parent experiment: `corrected-mid-tail-v1`

Status: production campaign complete

## Question

Does the immutable migrated T1 checkpoint produce exactly the same float32
prediction bytes as historical C0 on every state in the frozen 200,000-row
ADR 0133 corpus?

This is a foundation validity test, not a strength experiment. The only valid
positive result is exact parity on all rows. A tolerance, statistical
equivalence, or near-zero error is a failure.

## Hypothesis

The historical accidental range `10561..10862` is absent from every frozen
row. C0 base rows are copied exactly, C0 opponent rows are remapped exactly,
the new T1 tail begins at signed zero, and all downstream tensors are
unchanged. Therefore C0 and T1 should produce identical float32 bytes for all
200,000 states.

## Frozen Inputs

### Corpus

```text
root:
  artifacts/datasets/legacy-mid-v4opp-activation-v1
rows:
  200000
shards:
  10
manifest file BLAKE3:
  7ade2ca310c976c5db9a0e5a840399e226ad8c650e6a4342da845fbb501e0996
manifest scientific BLAKE3:
  193da520e3ccf3f440dd0f657996d486c1144737abcef1f8399b12ee8b34be92
payload BLAKE3:
  433ebf13b88f6133efa41f42f3225e13278052b82e3f23a7735401427b5019d8
```

Expected aggregate statistics:

| Statistic | Frozen value |
|---|---:|
| Games | 2,500 |
| Rows | 200,000 |
| Opening / early / middle / late | 10,000 / 40,000 / 80,000 / 70,000 |
| Rows per focal seat | 50,000 |
| Rows per policy | 50,000 |
| Free overflow preludes | 61,006 |
| Raw feature emissions | 49,849,871 |
| Unique feature activations | 46,833,451 |
| Duplicate emissions removed | 3,016,420 |
| Minimum / maximum unique features | 165 / 317 |

### C0

```text
checkpoint:
  nnue_weights_v4opp_modal_iter3.bin
schema:
  legacy-mid-v4opp-11231
container:
  NNUE
bytes:
  23134992
BLAKE3:
  9e1d568693274fc537ac4f6d6f729abb1ee8da8330a78d1f78a1f62b733de400
```

### T1

```text
checkpoint:
  artifacts/experiments/corrected-mid-tail-v1/models/blake3/a3e72314adeb4d62077e43ff071f95b27f979aba17f5026699118b19600263d0/nnue_weights_legacy_mid_v4_fixed_v1_init.bin
schema:
  legacy-mid-v4-fixed-v1
container:
  NNUC version 1
bytes:
  23135024
BLAKE3:
  a3e72314adeb4d62077e43ff071f95b27f979aba17f5026699118b19600263d0
```

## Implementation Contract

Use only:

- `LegacyRustExactSparseNnue`;
- `remap_historical_features_to_corrected`; and
- the strict `NNUE` / `NNUC` parser

from `python/cascadia_mlx/legacy_nnue.py`.

The campaign layer may validate, batch, hash, compare, aggregate, and report.
It may not introduce a second forward pass, a second remapping formula, a
tolerance, or a repaired historical extractor.

The implementation identity is a domain-separated BLAKE3 over the package
initializer, parity module, exact evaluator module, and command-line boundary.
Every production shard must use the same content-addressed identity frozen into
the reviewed queue specification.

## Shard Protocol

Each shard contains 20,000 rows and 250 complete games. A valid shard report
must prove:

1. exact manifest, payload, checkpoint, and implementation identities;
2. exact row count and contiguous game/decision interval;
3. strict JSON with no duplicate keys or non-finite constants;
4. exact seat, personal-turn, phase, and policy schedule;
5. sorted, unique, in-range sparse features;
6. no activation in historical rows `10561..10862`;
7. exact base and opponent checkpoint mapping;
8. signed-zero T1 tail and unchanged downstream tensors;
9. finite C0 and T1 outputs; and
10. one bit-identical float32 prediction per row.

A `--row-limit` report is a smoke. It can validate implementation behavior
but cannot enter the final aggregate.

## Aggregate Protocol

The aggregate is valid only with shard indices `0..9` exactly once. It must
reject:

- fewer or more than ten reports;
- duplicate, overlapping, gapped, or out-of-range shard identities;
- any report whose scientific hash does not recompute;
- any bounded smoke or incomplete shard;
- implementation, corpus, checkpoint, or mapping disagreement;
- malformed or non-finite counts;
- aggregate corpus statistics that differ from ADR 0133;
- any discarded-row activation;
- any non-finite output; or
- any C0/T1 prediction receipt mismatch.

Report arguments are supplied once in ascending order and once in descending
order. Both complete aggregate JSON files must be byte-identical.

## Scientific And Operational Separation

The scientific hash includes:

- implementation source identities;
- corpus and shard content identities;
- checkpoint identities and exact migration mapping;
- row coverage;
- corpus statistics and activation counts;
- sparse-stream receipts;
- prediction receipts; and
- pass gates.

It excludes:

- paths;
- host and device;
- batch size;
- timestamps;
- wall time;
- inference seconds; and
- throughput.

Operational evidence is still mandatory and reports C0, T1, paired, and
whole-command timing separately. No timing threshold can change the parity
classification.

## Cluster Allocation

| Host | Shards |
|---|---|
| john1 | 0, 4, 8 |
| john2 | 1, 5, 9 |
| john3 | 2, 6 |
| john4 | 3, 7 |

At most four shard tasks run concurrently. Every task is independent and
nonduplicative. The seven remote reports are checksum-collected to john1
before the two aggregates run.

## Success Classification

Use:

```text
corrected_mid_tail_frozen_parity_complete
```

only when:

- all ten immutable shards are present exactly once;
- all 200,000 row identities are covered exactly once;
- all frozen corpus statistics match;
- discarded-row activation count is zero;
- every prediction is finite; and
- all 200,000 C0/T1 float32 prediction bytes are identical.

The classification closes the T1 pre-training parity gate. It does not claim
that corrected tail features are active or useful; the historical corpus
cannot activate the new tail by construction.

## Failure Handling

Input, implementation, row, remapping, numerical, or aggregation failures are
foundation-invalid results. They must be repaired and rerun. They are not
negative research findings.

No passing aggregate is written when validation fails.

## Exact Commands

Single production shard:

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

Bounded real-corpus smoke:

```bash
.venv/bin/python tools/corrected_mid_tail_parity.py shard \
  --shard-index 0 \
  --row-limit 80 \
  --output /tmp/corrected-mid-tail-frozen-parity-v1-smoke.json
```

Aggregate all ten reports:

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

Generate the reviewed, immutable, non-applied task graph:

```bash
.venv/bin/python tools/corrected_mid_tail_parity_queue_spec.py \
  --output \
    artifacts/experiments/corrected-mid-tail-v1/frozen-parity-v1/queue-spec.json
```

## Pre-Launch Implementation Evidence

One complete real shard was run locally:

```text
shard:
  0
rows:
  20000
first / last row:
  (0, 0) / (249, 79)
discarded-row activations:
  0
corrected-tail activations:
  0
bit-identical predictions:
  20000
prediction BLAKE3, C0 and T1:
  08bf625a33725000d80bca5dce3cbf5cbf7e73b7e8d1f7b842c064436009f662
scientific BLAKE3:
  19802b453df0e29de9d52c19810547085996940089945ce448da4142569355f6
```

The full shard was repeated with batch sizes 512 and 777. Both runs produced
the same scientific BLAKE3. This establishes deterministic production-scale
shard behavior but does not substitute for the unlaunched ten-shard campaign.

## Final Report

| Field | Result |
|---|---|
| Classification | `corrected_mid_tail_frozen_parity_complete` |
| Aggregate scientific BLAKE3 | `ea2c89c36d8fe727ade42b8f17bece1e55b7d220b598d6cef784a61ec6a156bb` |
| Implementation BLAKE3 | `82eff4668432f9acd6ced6268547e70f30f08f1669d88c3605ef340d2dd1ff8a` |
| Complete shards | 10 / 10 |
| Rows covered | 200,000 / 200,000 |
| Discarded-row activations | 0 |
| Corrected-tail activations | 0 |
| Non-finite C0 / T1 | 0 / 0 |
| Prediction mismatches | 0 |
| C0 aggregate prediction receipt | `6d12482cc25615aad9a64b4a17ff477b9161618aa728e25f1c867587af35b53e` |
| T1 aggregate prediction receipt | `6d12482cc25615aad9a64b4a17ff477b9161618aa728e25f1c867587af35b53e` |
| Forward/reverse aggregate byte identity | pass |
| john1 mean paired rows/s | 164,063.6 |
| john2 mean paired rows/s | 146,900.2 |
| john3 mean paired rows/s | 156,113.2 |
| john4 mean paired rows/s | 138,497.4 |

All preregistered scientific fields pass. The next F5 gate is corrected-schema
activation on source-frozen records; gameplay and score claims remain closed.
