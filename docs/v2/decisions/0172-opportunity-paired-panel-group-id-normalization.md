# ADR 0172: Opportunity Paired-Panel Group-ID Normalization

- Status: accepted
- Date: 2026-06-17
- Experiment: `opportunity-cross-attention-mlx-tournament-v1`
- Scope: paired-evidence serialization and validation
- Model, labels, optimizer, and measurements affected: none

## Context

The untouched-C0 control completed its expensive scoring pass and then failed
while validating the paired decision panel:

```text
OverflowError: Python integer -5482088856184735585 out of bounds for uint64
```

The dataset exposes an opaque 64-bit group hash through a signed `i64` storage
surface. Negative JSON integers therefore represent valid high-bit-set `u64`
patterns. Older NumPy versions silently wrapped a direct `uint64` conversion;
NumPy 2.4 rejects it.

The three treatment trainers use the same paired-panel path after their final
checkpoint, so the boundary bug could also prevent terminal report emission.

## Decision

Paired-panel validation now normalizes each group ID explicitly:

1. accept only integer values;
2. accept the full signed-`i64` through unsigned-`u64` range;
3. reinterpret valid values with `value & ((1 << 64) - 1)`; and
4. construct the final NumPy array from those normalized bit patterns.

Values below `-2^63`, above `2^64 - 1`, booleans, and non-integers fail
closed.

Signed and unsigned representations of the same 64-bit pattern align
identically for panel comparison. The JSON panel and its canonical identity
remain unchanged, so this is a compatibility repair rather than a scientific
change.

## Recovery

The failed C0-control attempt is retained. It produced no accepted artifact.
The original treatment processes were stopped after completing step 400. Their
run directories, traces, and checkpoints are retained as invalid-launch audit
evidence, but they are not resumed or accepted scientifically. All four arms
restart from the common exact-R2 warm start in isolated run directories under a
new immutable bundle after:

- unit coverage of signed, unsigned, and invalid boundaries;
- direct replay of the previously failing group ID; and
- cross-host smoke identity verification.

This deliberately spends the first 400 optimizer steps again. It avoids mixing
source identities inside a run manifest and keeps every accepted treatment
fully reproducible from one immutable source tree.
