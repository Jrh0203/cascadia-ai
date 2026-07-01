# ADR 0002: MLX Owns Trainable Neural Compute

Status: Accepted  
Date: 2026-06-10

## Context

V1 implements neural forward and training logic in several Rust and Python
paths. This duplicates architecture definitions and creates format drift. The
target machine is Apple Silicon and the user requires MLX for Apple/neural work.

## Decision

All trainable neural models, losses, optimization, checkpointing, and inference
execute in the Python `cascadia_mlx` package using MLX. Rust communicates with a
long-lived local batched model service through a versioned protocol. Checkpoints
are MLX-compatible safetensors plus manifests; Rust does not parse weights.

## Consequences

- One implementation owns each architecture.
- Apple GPU behavior is explicit and measurable.
- The service boundary must be benchmarked and may need shared-memory
  optimization later.
- Non-Apple environments can test schemas and rules but cannot certify neural
  performance.

