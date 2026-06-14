# Rust-Order MLX CSR Service

Experiment:
`qualified-legacy-nnue-mlx-exact-csr-service-v1-parity-20260612`

## Result

Passed. The packed CSR service reproduced all 80 qualified Rust NNUE fixture
values bit for bit and repeated deterministically.

| Batch | Median latency | P99 latency | Throughput |
|---:|---:|---:|---:|
| 1 | 0.404 ms | 0.899 ms | 2,473 eval/s |
| 32 | 0.426 ms | 0.698 ms | 75,176 eval/s |
| 256 | 0.669 ms | 1.021 ms | 382,494 eval/s |

Startup was 154.17 ms and shutdown was clean. Focused Python tests passed
11/11 and Rust client tests passed 5/5.

## Interpretation

Message type 6 removes both causes of ADR 0057's failed smoke. CSR offsets
eliminate per-row parsing and padded repacking, while three custom Metal
kernels preserve Rust's first-layer, second-layer, and output accumulation
order. NumPy only validates host buffers; every prediction is computed by MLX
on the Apple GPU.

This report authorizes only a separately preregistered deterministic search
integration. It does not establish gameplay strength.

Machine-readable report:
`legacy-nnue-v4opp-mlx-exact-csr-service-v1.json`

BLAKE3:
`1e1481a0b3a6c975d99185a3e06b585056e70d4626b88119b748de964b112e8b`
