# Qualified NNUE MLX Service

ADR 0056 passed the complete framed-service boundary on 2026-06-12.

## Contract

The existing 16-byte version-one `CMLX` header now reserves:

- request type `5` for sparse NNUE rows;
- response type `0x8005` for one scalar per row.

Each row contains a little-endian `u16` length followed by ordered
little-endian `u16` feature indices. The contract permits empty rows, preserves
duplicates, caps rows at 4,096 features, and caps requests at 65,536 rows.

Both ends reject malformed headers, zero or oversized batches, oversized or
truncated rows, out-of-range features, response identity drift, and non-finite
outputs. Stream-corrupting errors return a typed error frame and terminate.

## Parity

The immutable ADR 0055 fixture contains 80 real sparse rows. Service output was
bit-identical to direct MLX:

| Comparison | Maximum absolute error | P99 | Mean |
|---|---:|---:|---:|
| Service vs direct MLX | 0.0 | 0.0 | 0.0 |
| Service vs native Rust | 0.0000419617 | n/a | 0.0000148922 |

Repeated service calls were bit-identical, all values were finite, and clean
shutdown returned success.

## IPC Performance

| Batch | P50 milliseconds | P99 milliseconds | Evaluations/second |
|---:|---:|---:|---:|
| 1 | 0.3652 | 0.9096 | 2,738.2 |
| 32 | 4.2166 | 4.6977 | 7,589.0 |
| 256 | 37.2453 | 39.0206 | 6,873.4 |

The batch-32 result cleared the frozen 2,000 evaluations/second gate by 3.79x
and the 25-millisecond P99 ceiling by more than 5x.

## Reproduce

```bash
make legacy-nnue-mlx-service
```

This target consumes the immutable ADR 0055 model and fixture. It does not
regenerate or overwrite the earlier conversion evidence.
