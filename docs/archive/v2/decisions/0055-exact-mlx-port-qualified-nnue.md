# ADR 0055: Exact MLX Port of the Qualified NNUE

Status: passed on 2026-06-12. Batched search integration is authorized only
through a separate preregistered decision.

## Context

The qualified canonical action teacher scored 96.350, but four successive
shared-action apprentices failed to concentrate its selected action near the
top. ADR 0054 then proved that 99.544% of absolute continuation-residual
variance was action-independent within a decision. A centered development
screen improved value geometry but still regressed exact top-one.

The teacher's neural component is not intrinsically difficult to reproduce.
It is a sparse binary NNUE:

- 11,231 input features;
- 512-unit ReLU first layer;
- 64-unit ReLU second layer;
- one scalar value head;
- 5,783,681 value parameters;
- version-one `NNUE` binary source at 23,134,992 bytes.

The source weights are
`nnue_weights_v4opp_modal_iter3.bin`, BLAKE3
`9e1d568693274fc537ac4f6d6f729abb1ee8da8330a78d1f78a1f62b733de400`.
Reusing those parameters directly in MLX avoids another lossy policy
distillation step while satisfying the V2 requirement that Apple neural
execution use MLX.

## Decision

Build an exact, read-only MLX representation of the qualified NNUE:

1. Parse the historical binary with strict magic, version, dimensions, byte
   count, finite-value, and trailing-section validation.
2. Preserve first-layer row orientation exactly as
   `[feature, hidden1]`.
3. Evaluate sparse batches by gathering active first-layer rows, masking
   padding, summing with the bias, then applying the original two ReLU layers
   and scalar head.
4. Store parameters in `model.safetensors` plus a checksummed JSON manifest
   that binds the source file, dimensions, architecture, and artifact bytes.
5. Refuse silent feature truncation, zero-padding, non-finite parameters,
   out-of-range sparse indices, lost feature multiplicity, or manifest drift.
6. Add a Rust fixture command that emits active feature indices and native
   `NNUENetwork::forward` values from every decision in one complete
   canonical V2 trajectory translated through the qualified bridge.
7. Add deterministic synthetic parity and Apple-GPU batch-throughput probes.

This ADR ports only the neural evaluator. It does not authorize gameplay,
promotion, weight mutation, retraining, or a second implementation of
canonical rules.

## Frozen Protocol

- Source model: the exact weights and checksum above.
- Compiled feature set: `mid-features,v4-opp`.
- Expected file version: 1.
- Expected dimensions: 11,231 -> 512 -> 64 -> 1.
- Real-state fixture: one complete four-player AAAAA game, no habitat bonuses,
  train split index 92,000, pattern-aware canonical trajectory.
- Expected fixture size: 80 acting-seat positions.
- Synthetic probe: fixed seed 20260619, including empty, singleton, dense,
  boundary-index, and 256 random sparse feature sets.
- Device: `Device(gpu, 0)`.
- Throughput batches: 1, 32, and 256 after compilation warmup.
- Source and fixture generation must occur from an unchanged V2 source digest.

Every gate must pass:

- exact source checksum, file size, version, and dimensions;
- every parameter finite and every sparse index in range;
- preserve native feature order and multiplicity exactly, including repeated
  indices emitted by the historical extractor;
- artifact load reproduces its manifest and safetensors checksums;
- repeated MLX calls are bit-deterministic within one process;
- synthetic maximum absolute error at most `1e-3`;
- real-state maximum absolute error at most `1e-3`;
- real-state P99 absolute error at most `5e-4`;
- real-state mean absolute error at most `1e-4`;
- all 80 Rust fixture records evaluate and remain finite;
- warmed batch-32 throughput at least 2,000 evaluations per second.

Passing authorizes a separate ADR for batched search integration. Failure
closes direct parameter reuse before any gameplay domain is opened.

## Maximum Compute

One conversion, one 80-state fixture, one synthetic parity run, and one local
Apple-GPU throughput run. No external compute, training, weight changes,
hyperparameter sweep, gameplay benchmark, or promotion is authorized.

## Result

Passed every frozen integrity, parity, determinism, and throughput gate.

The strict converter produced:

- `artifacts/models/legacy-nnue-v4opp-mlx-v1/model.json`, BLAKE3
  `dd3ea3bbbff0187107695132531a56c09a1da18b58fac4bacacf66960fd7ff0d`;
- `model.safetensors`, 23,135,528 bytes, BLAKE3
  `3f8f2609b1440396720aa48adabf9561a4a172d006f77011a9516baa0b06ba65`.

The Rust fixture contains all 80 decisions from train game 92,000 and is bound
to V2 source digest
`ecc837dd21804fada7d60ec4c115eece95e310335a0f2dd757a901cd1f45674f`
and executable digest
`d65038b2790f7b203378af096d7721e6fd38c58adafb97fe7adc90618cb02d44`.
Its BLAKE3 is
`1e1a89d4ca2a540587793a0fe681b11de80e661f6d419328c59f31e910797238`.

The historical extractor emitted repeated feature indices in every real
record: 1,170 duplicate occurrences, with maximum multiplicity five. The MLX
packer preserves those repeats exactly. This is part of the compatibility
contract, not data cleanup.

On `Device(gpu, 0)`:

| Domain | Records | Maximum absolute error | P99 | Mean |
|---|---:|---:|---:|---:|
| Synthetic | 260 | 0.0000104904 | 0.0000094509 | 0.0000025135 |
| Rust fixture | 80 | 0.0000419617 | 0.0000389481 | 0.0000148922 |

Repeated real-batch calls were bit-identical and every output was finite.
Warmed throughput was 3,926.7 evaluations/second at batch one, 40,569.4 at
batch 32, and 58,449.2 at batch 256. Batch-32 performance exceeded the 2,000
evaluation/second gate by 20.3x.

The checksummed machine reports are
`docs/v2/reports/legacy-nnue-v4opp-mlx-v1-parity.json` (BLAKE3
`923f1f5d3eb2144d319524edad685b194ab081b3b3dd086bf8ec806678eb28c2`)
and `docs/v2/reports/legacy-nnue-v4opp-mlx-v1-benchmark.json` (BLAKE3
`1c04afcfc6c364137c9d7f1b64f7db4457c1c438ba02dfc69d334dfcc5be1e98`).
No training, test split, gameplay benchmark, or promotion occurred.
