# Corrected Mid-Tail Frozen Parity V1 Result

Date: 2026-06-17

Experiment ID: `corrected-mid-tail-frozen-parity-v1`

Classification: **`corrected_mid_tail_frozen_parity_complete`**

## Result

Historical C0 and migrated corrected-schema T1 produced byte-identical
little-endian float32 predictions for every one of the frozen 200,000 states.

| Gate | Result |
|---|---:|
| Shards | 10 / 10 |
| Rows | 200,000 / 200,000 |
| Complete contiguous game interval | pass |
| Frozen corpus statistics | exact |
| Historical discarded-row activations | 0 |
| Corrected-tail activations | 0 |
| Non-finite C0 / T1 predictions | 0 / 0 |
| Prediction mismatches | 0 |
| Forward/reverse aggregate bytes | identical |

Aggregate scientific BLAKE3:

```text
ea2c89c36d8fe727ade42b8f17bece1e55b7d220b598d6cef784a61ec6a156bb
```

Both prediction streams have aggregate receipt:

```text
6d12482cc25615aad9a64b4a17ff477b9161618aa728e25f1c867587af35b53e
```

## Provenance

Every shard executed the same content-addressed implementation:

```text
82eff4668432f9acd6ced6268547e70f30f08f1669d88c3605ef340d2dd1ff8a
```

The immutable source bundle was whole-tree verified on john1, john2, john3,
and john4 before inference. The original unlaunched 14-task draft was
invalidated during review because it referenced mutable repository paths. The
executed 15-task graph added explicit source-bundle fanout.

## Decision

Close the F5 C0/T1 migration-neutrality gate. The migrated checkpoint is an
exact functional replacement for the historical checkpoint on the full frozen
corpus.

This does not show that the corrected 301-row tail is useful: those rows are
zero-initialized and absent from the historical corpus. Corrected-schema
activation, fine-tuning, offline qualification, and paired gameplay remain
separate gates.

## Artifacts

- `artifacts/experiments/corrected-mid-tail-v1/frozen-parity-v1/reports/aggregate-forward.json`
- `artifacts/experiments/corrected-mid-tail-v1/frozen-parity-v1/reports/aggregate-reverse.json`
- `artifacts/experiments/corrected-mid-tail-v1/frozen-parity-v1/reports/source-bundle-fanout.json`
- `artifacts/experiments/corrected-mid-tail-v1/frozen-parity-v1/queue-spec.json`
