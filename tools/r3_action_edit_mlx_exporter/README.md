# R3 Action-Edit MLX Exporter

Rust-authoritative cache exporter for ADR 0150.

It replays the open complete-action graded-oracle train and validation games,
encodes one accepted R2 parent trunk per decision, and writes:

- an exact compact control delta for every retained R2 afterstate;
- the lossless canonical R3 radius-3 action-token stream;
- deterministic train-cohort indices; and
- complete source, action, and tensor identities.

The production cache contains the frozen at-most-512 train cohort and every
validation action. It never opens sealed test or gameplay data and never reads
hidden refill order beyond replaying the already-frozen source game seed.

```bash
cargo run --release -- \
  --train-dataset ../../artifacts/datasets/complete-action-graded-oracle-v1-train \
  --validation-dataset ../../artifacts/datasets/complete-action-graded-oracle-v1-validation \
  --output-root ../../artifacts/experiments/r3-action-edit-mlx-comparison-v1/cache \
  --receipt ../../artifacts/experiments/r3-action-edit-mlx-comparison-v1/cache-export.json
```

Use `--max-groups-per-split N` only for smoke caches. Bounded caches are marked
incomplete and are rejected by production preflight.

## Python Boundary

`python/cascadia_mlx/r3_action_edit_mlx_cache.py` independently verifies the
content address, every tensor checksum and shape, train-cohort identities,
graded-oracle action hashes, and the complete S1 candidate identity before
MLX sees a row.

For the control arm, the loader:

1. applies the Rust-owned D6 transform to the cached parent;
2. translates the active board into the selected-action frame;
3. applies the exact removal/addition multiset delta;
4. checks the reconstructed afterstate BLAKE3; and
5. maps the result onto the shared 80-wide candidate-token surface.

For R3 arms, radius 2 and radius 1 are exact crops of radius-3 local-patch
tokens. Every non-patch token remains byte-identical.

## Bounded Training Smoke

A smoke cache must include slice members for the fourth training slot. A full
80-decision game is the smallest convenient cache:

```bash
cargo run --release -- \
  --train-dataset ../../artifacts/datasets/complete-action-graded-oracle-v1-train \
  --validation-dataset ../../artifacts/datasets/complete-action-graded-oracle-v1-validation \
  --output-root ../../artifacts/experiments/r3-action-edit-mlx-comparison-v1/smoke-cache-full-game \
  --receipt ../../artifacts/experiments/r3-action-edit-mlx-comparison-v1/smoke-cache-full-game-export.json \
  --max-groups-per-split 80
```

Run at most ten optimizer steps without production authorization:

```bash
PYTHONPATH=python .venv/bin/python -m \
  cascadia_mlx.r3_action_edit_mlx_train \
  --train-dataset artifacts/datasets/complete-action-graded-oracle-v1-train \
  --validation-dataset artifacts/datasets/complete-action-graded-oracle-v1-validation \
  --cache artifacts/experiments/r3-action-edit-mlx-comparison-v1/smoke-cache-full-game/CACHE_ID \
  --s1-cache artifacts/experiments/exact-semantic-supply-learned-comparison-v1/cache/2323ead43b1bff7a506ecef4b8bd4793cebe4d53c6f8940b03404573ca5e6c15 \
  --run-dir artifacts/experiments/r3-action-edit-mlx-comparison-v1/smoke-runs/c0 \
  --output artifacts/experiments/r3-action-edit-mlx-comparison-v1/smoke-runs/c0-report.json \
  --arm c0-full-r2-afterstate \
  --smoke-steps 10
```

Production omits `--smoke-steps` and requires content-bound
`--authorization` and per-host `--preflight` files.
