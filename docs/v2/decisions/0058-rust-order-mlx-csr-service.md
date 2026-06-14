# ADR 0058: Rust-Order MLX CSR Service

Status: passed on 2026-06-12. A second rollout-wave integration may now be
preregistered; gameplay remains closed.

## Context

ADR 0057 proved that evaluator-independent rollout waves preserve native
search exactly, then rejected the ADR 0056 neural operation. Two measured
causes were independent:

1. standard MLX reduction and matrix-multiplication order differed from the
   qualified Rust scalar forward by up to `4.196e-5`, which can flip a strict
   near tie and change a rollout trajectory;
2. message type 5 interleaves one row length with each row payload, forcing
   Python to parse and repack every row before inference.

An exploratory train-domain fixture probe used three MLX custom Metal kernels:
one thread per first-layer activation, one per second-layer activation, and one
per output. Each thread accumulates in the same order as Rust, with floating
point contraction disabled for multiply-add layers. All 80 frozen fixture
outputs were bit-identical to Rust and the direct 80-row batch took about
0.7 milliseconds after JIT compilation. This probe selects the method but is
not qualifying evidence.

## Decision

Add a distinct, backward-compatible protocol operation:

- request `6`, response `0x8006`;
- frame `count` remains the number of sparse rows;
- payload is `u32 total_features`, `(count + 1)` little-endian `u32` CSR
  offsets, then `total_features` little-endian `u16` feature indices;
- offsets must begin at zero, end at `total_features`, be monotonic, and encode
  no row wider than 4,096;
- duplicate feature indices and empty rows are preserved;
- every feature index must be below 11,231;
- Rust serializes the complete payload with one buffered write and reads the
  complete response with one buffered read;
- Python validates the CSR arrays without rebuilding padded rows;
- MLX custom Metal kernels preserve the exact qualified Rust operation order;
- message type 5 and its passed ADR 0056 behavior remain unchanged.

Rust continues to own feature extraction. The service owns all arithmetic.
NumPy may parse and validate host buffers, but it may not compute predictions.

## Frozen Protocol

- Model and fixture: immutable ADR 0055 artifacts.
- Hardware: Mac mini M4, MLX `0.31.2`, Apple GPU.
- Direct fixture: all 80 records, including duplicate-feature multiplicity.
- Service fixture: all 80 records twice through one long-lived process.
- Performance batches: 1, 32, and 256 rows, 200 timed calls after five warmups.
- Existing malformed-frame, bounds, duplicate, empty-row, and clean-shutdown
  tests remain required.

Every gate must pass:

- direct custom-kernel and service outputs are bit-identical to all 80 Rust
  fixture values;
- repeat service output is bit-identical;
- every response has exact width and finite values;
- malformed CSR offsets, widths, bounds, and truncation fail explicitly;
- clean shutdown succeeds;
- batch-32 median throughput is at least 10,000 evaluations per second;
- batch-32 P99 latency is at most 10 milliseconds;
- focused Python and `cascadia-model` tests pass.

Passing authorizes only a separately preregistered second rollout-wave
integration. It does not authorize gameplay, validation/test seeds, training,
or model promotion.

## Maximum Compute

One qualifying service parity/throughput run after focused tests. No search,
gameplay, training, model sweep, or held-out seed domain.

## Result

Passed every frozen gate:

- all 80 service values were bit-identical to the Rust fixture;
- repeated service output was bit-identical;
- focused Python tests passed 11/11 and `cascadia-model` passed 5/5;
- startup took 154.17 ms and shutdown was clean;
- batch 1: 2,473 evaluations/s, 0.899 ms P99;
- batch 32: 75,176 evaluations/s, 0.698 ms P99;
- batch 256: 382,494 evaluations/s, 1.021 ms P99.

The packed operation is 9.9x faster than ADR 0056 at batch 32 while removing
its `4.196e-5` arithmetic difference completely. The qualifying report is
`docs/v2/reports/legacy-nnue-v4opp-mlx-exact-csr-service-v1.json`, BLAKE3
`1e1481a0b3a6c975d99185a3e06b585056e70d4626b88119b748de964b112e8b`.

This result qualifies only request type 6 and authorizes a separately frozen
rollout-wave parity retry. It makes no playing-strength claim.
