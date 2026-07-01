# ADR 0056: Batched MLX NNUE Service Boundary

Status: passed on 2026-06-12. Deterministic rollout-wave search integration is
authorized only through a separate preregistered decision.

## Context

ADR 0055 proved that the qualified historical NNUE can execute directly in
MLX with maximum real-state error below 0.000042 points and more than 40,000
batch-32 evaluations per second. Search cannot use that result safely through
one process invocation or one pipe exchange per scalar evaluation. The
qualified MCE policy evaluates many sparse afterstates per decision and needs
a long-lived, explicitly batched boundary.

V2 already owns a framed local `CMLX` protocol and typed Rust process client.
The new evaluator has variable-length sparse inputs, so it requires a distinct
message type rather than overloading fixed-size entity or action records.

## Decision

Add one protocol operation for qualified sparse NNUE inference:

- request type `5`;
- response type `0x8005`;
- the existing 16-byte `CMLX` version-one frame header;
- header `count` equal to the number of sparse rows;
- each row encoded as a little-endian `u16` feature count followed by that many
  little-endian `u16` feature indices;
- one little-endian finite `f32` response per row.

The Rust client and Python service must reject zero or oversized batches,
truncated rows, rows above the explicit feature-count ceiling, out-of-range
indices, non-finite predictions, response type/count/request-ID drift, model
manifest drift, and abnormal process exit. They must preserve row order,
feature order, and repeated feature indices exactly.

This ADR implements and verifies only the service boundary. It does not modify
the MCE search, call the service from gameplay, authorize a gameplay seed, or
promote a strategy.

The service target consumes the immutable ADR 0055 model and fixture. It must
never regenerate or overwrite ADR 0055 evidence as a side effect.

## Frozen Protocol

- Model artifact:
  `artifacts/models/legacy-nnue-v4opp-mlx-v1/model.json`, BLAKE3
  `dd3ea3bbbff0187107695132531a56c09a1da18b58fac4bacacf66960fd7ff0d`.
- Safetensors BLAKE3:
  `3f8f2609b1440396720aa48adabf9561a4a172d006f77011a9516baa0b06ba65`.
- Real fixture:
  `artifacts/fixtures/legacy-nnue-v4opp-mlx-v1-rust.json`, BLAKE3
  `1e1a89d4ca2a540587793a0fe681b11de80e661f6d419328c59f31e910797238`.
- Protocol version: 1.
- Maximum rows per request: 65,536.
- Maximum sparse features per row: 4,096.
- Real parity batch: all 80 fixture rows in original order.
- Synthetic protocol cases: empty sparse row, repeated indices, boundary
  indices, malformed length, truncated payload, out-of-range index, invalid
  request type, and clean shutdown.
- Throughput batches: 1, 32, and 256, using fixture-derived row lengths and
  fixed seed 20260620 after service and MLX warmup.
- Device: `Device(gpu, 0)`.

Every gate must pass:

- all Python and Rust protocol tests;
- service startup loads and verifies the exact ADR 0055 artifact;
- all 80 fixture outputs are finite and returned in original order;
- maximum absolute service-versus-Rust error at most `1e-3`;
- maximum absolute service-versus-direct-MLX error at most `1e-6`;
- repeated service calls are bit-identical;
- repeated sparse indices change the result exactly as direct MLX predicts;
- malformed frames return a typed error and terminate when stream alignment
  cannot be recovered;
- clean shutdown exits with success;
- warmed end-to-end batch-32 throughput at least 2,000 evaluations per second;
- warmed batch-32 P99 latency at most 25 milliseconds.

Passing authorizes a separate ADR to restructure qualified search into
deterministic rollout waves that call this service in batches. Failure closes
pipe-based integration before any gameplay path is modified.

## Maximum Compute

One implementation smoke, one 80-row end-to-end parity run, and one local
Apple-GPU IPC benchmark at batches 1, 32, and 256. No training, external
compute, search modification, gameplay benchmark, test split, or promotion is
authorized.

## Result

Passed every frozen service, parity, determinism, error-handling, shutdown, and
throughput gate.

The implementation adds request type `5` and response type `0x8005` to the
existing version-one `CMLX` framing. Rust validates batch size, row width,
feature bounds, response identity, finite outputs, and process status. Python
validates the same row contract, preserves empty rows and repeated indices,
returns typed error frames, and terminates when malformed input destroys
stream alignment.

Across all 80 immutable ADR 0055 fixture rows:

- service versus direct MLX maximum, P99, and mean absolute error were exactly
  `0.0`; outputs were bit-identical;
- service versus native Rust maximum error was `0.0000419617` and mean error
  was `0.0000148922`;
- repeated calls were bit-identical and every output was finite;
- startup through the first response took 103.20 milliseconds;
- shutdown exited successfully.

End-to-end Rust-to-process throughput, including sparse serialization, pipe
I/O, MLX inference, response parsing, and validation:

| Batch | P50 milliseconds | P99 milliseconds | Evaluations/second |
|---:|---:|---:|---:|
| 1 | 0.3652 | 0.9096 | 2,738.2 |
| 32 | 4.2166 | 4.6977 | 7,589.0 |
| 256 | 37.2453 | 39.0206 | 6,873.4 |

Batch 32 exceeded the 2,000 evaluations/second gate by 3.79x and remained well
below the 25-millisecond P99 ceiling.

The direct protocol report is
`docs/v2/reports/legacy-nnue-v4opp-mlx-service-v1-direct.json`, BLAKE3
`dc3bc0c176b4cb91c36aa1853cbd4d0e25fc34986b520e20956a194eaf257984`.
The Rust IPC report is
`docs/v2/reports/legacy-nnue-v4opp-mlx-service-v1.json`, BLAKE3
`9a98d5b40bcef99ad70eebb619806ead3b7aca95bdc53a61d285df496bec9e81`.
No search behavior, gameplay seed, test split, training run, or promotion was
opened.
