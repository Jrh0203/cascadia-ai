# Local Hardware And Toolchain

Captured: 2026-06-10

## Machine

| Item | Value |
|---|---|
| Model | Mac mini (`Mac16,10`, `MU9D3LL/A`) |
| Chip | Apple M4 |
| CPU | 10 cores: 4 performance, 6 efficiency |
| GPU | 10 integrated cores |
| Unified memory | 16 GB |
| Architecture | arm64 |
| Metal | Supported |
| macOS | 26.2 (`25C56`) |
| Darwin | 25.2.0 |

The 16 GB memory ceiling is a first-class design constraint. Training defaults
must bound replay windows, batches, prefetching, and model size instead of
assuming workstation-class memory.

## Toolchain

| Tool | Version |
|---|---|
| Rust | 1.94.1 (`aarch64-apple-darwin`) |
| Cargo | 1.94.1 |
| LLVM | 21.1.8 |
| Apple Clang | 21.0.0 |
| Homebrew Python | 3.14.4; unusable because of system `libexpat` mismatch |
| Project Python | uv-managed CPython 3.12.13 |
| uv | 0.11.20 |
| MLX | 0.31.2 |

## MLX

The project environment is created by `uv sync --all-groups` from
`pyproject.toml` and `uv.lock`. The checked-in `cascadia-mlx-device` probe
evaluates a one-million-element sum-of-squares workload rather than merely
importing the package.

Verified result on 2026-06-10:

| Item | Result |
|---|---|
| Device | `Device(gpu, 0)` |
| Workload time | 0.184 seconds, including first-use setup |
| Relative numeric error | `7.56e-8` |
| Probe tests | 2 passed |

The Homebrew 3.14 and 3.12 bottles both load `/usr/lib/libexpat.1.dylib` but
expect `_XML_SetAllocTrackerActivationThreshold`, which is absent on this
macOS build. The uv-managed Python distribution avoids that host linkage defect
and is the supported v2 runtime.

CI and non-Apple development paths may validate schemas and pure logic, but
neural training and promoted inference measurements are Apple/MLX-only.

## Storage

The current repository is approximately 110 GB and mixes code with generated
weights, replay files, logs, and build trees. V2 artifacts will live under a
single ignored artifact root with manifests and checksums. Existing artifacts
will not be deleted during migration.
