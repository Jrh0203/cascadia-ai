# Qualified Legacy NNUE MLX Port

ADR 0055 passed every frozen conversion, forward-parity, determinism, and
Apple-GPU throughput gate on 2026-06-12.

## Artifact

| Item | Value |
|---|---|
| Source | `nnue_weights_v4opp_modal_iter3.bin` |
| Source BLAKE3 | `9e1d568693274fc537ac4f6d6f729abb1ee8da8330a78d1f78a1f62b733de400` |
| Architecture | 11,231 sparse features -> 512 ReLU -> 64 ReLU -> 1 |
| Manifest BLAKE3 | `dd3ea3bbbff0187107695132531a56c09a1da18b58fac4bacacf66960fd7ff0d` |
| Safetensors BLAKE3 | `3f8f2609b1440396720aa48adabf9561a4a172d006f77011a9516baa0b06ba65` |
| Device | `Device(gpu, 0)` |

The converter verifies magic, version, byte count, dimensions, source
checksum, tensor shapes, finite parameters, trailing bytes, and the generated
safetensors checksum. Existing artifact directories are validated rather than
silently overwritten.

## Parity

| Domain | Records | Maximum absolute error | P99 | Mean |
|---|---:|---:|---:|---:|
| Synthetic | 260 | 0.0000104904 | 0.0000094509 | 0.0000025135 |
| Rust fixture | 80 | 0.0000419617 | 0.0000389481 | 0.0000148922 |

Every output was finite. Repeating the complete real batch returned
bit-identical float32 values.

The real fixture is one complete canonical V2 AAAAA trajectory at train index
92,000. It is bound to V2 source and executable checksums. All 80 records
contain repeated feature indices: 1,170 duplicate occurrences with maximum
multiplicity five. The exact MLX representation preserves feature order and
multiplicity rather than treating the extractor output as a set.

## Throughput

| Batch | P50 milliseconds | Evaluations/second |
|---:|---:|---:|
| 1 | 0.2547 | 3,926.7 |
| 32 | 0.7888 | 40,569.4 |
| 256 | 4.3799 | 58,449.2 |

Batch 32 exceeded the frozen 2,000 evaluations/second gate by 20.3x.

## Reproduce

```bash
make legacy-nnue-mlx-port
```

The command converts or verifies the artifact, rebuilds the Rust fixture,
runs synthetic and real parity on MLX, and records warmed throughput. It does
not train, mutate weights, open a test split, benchmark gameplay, or promote a
strategy.
