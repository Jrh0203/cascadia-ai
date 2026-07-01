# ADR 0170: Streaming Materialization Parity Verifier

- Status: accepted
- Date: 2026-06-17
- Experiment: `exact-r2-preverified-vectorized-materialization-v1`
- Scope: qualification-harness memory only
- Does not change: either materialization implementation, timing boundaries,
  parity fields, corpus, thresholds, or classification rules

## Context

The first complete john1 comparison proved exact digest equality over 240
decisions, 860,203 actions, and 50,567,539 candidate tokens. It achieved a
15.44x P99 speedup, but the verifier's peak RSS was 4,508,139,520 bytes, above
the frozen 4 GiB qualification cap.

The verifier retained both complete per-decision batches simultaneously for
`np.array_equal`. It also converted every large contiguous array to `bytes`
before hashing, creating another transient copy. The measured memory therefore
described the proof harness rather than either materialization path.

## Decision

For each decision and each materialization mode:

1. materialize one batch;
2. bind every parity field's label, dtype, shape, and raw contiguous bytes into
   an independent BLAKE3 digest;
3. fold those field digests into the mode-wide digest;
4. release the batch and clear caches;
5. materialize the other mode;
6. compare the complete field-digest maps.

Hashing uses a zero-copy byte `memoryview` for contiguous arrays. If any field
digest differs, the verifier rematerializes that single row under both paths
and runs the original exact `np.array_equal` diagnostic to identify the first
shape or value difference.

The digest contract is:
`blake3-over-label-dtype-shape-and-contiguous-bytes-v1`.

## Rationale

The successful path never holds two large batches or an additional full byte
copy. The proof still covers every byte of every frozen parity field. The
failure path retains the exact element-level diagnostic, where higher memory is
acceptable because qualification has already failed.

## Evidence Handling

The first full report is retained as
`validation-john1-legacy-first-memory-attempt-1.json`. It remains valid speed
and parity evidence but cannot pass the frozen memory gate.

All promotion evidence must be regenerated with the ADR 0170 source hash.
