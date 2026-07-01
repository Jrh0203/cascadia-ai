# Cascadia Exact-R2 Runtime

Reusable workspace implementation of the exact
`r2-sparse-public-token-state-v1` foundation and the R2-MAP streaming bridge.
The historical `tools/r2_sparse_entity_census` manifest is now a compatibility
facade over this crate, so runtime and research tools compile one authority.

The tool reads validated `compact-entity-v2` PositionRecord datasets and
builds:

- exact occupied-tile tokens;
- exact legal-frontier tokens;
- exact directed-edge habitat components;
- exact per-wildlife motif anchors; and
- separate public global, player, and market metadata.

It has no dense board support, clipping path, or hidden-state input. Terminal
targets are intentionally excluded.

## Commands

```bash
cd /Users/johnherrick/cascadia

# Inspect one public state and optionally write its canonical packed form.
cargo run --release -p cascadia-r2 -- tokenize \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-train-part-0 \
  --ordinal 0 \
  --output /tmp/r2-state.json \
  --packed-output /tmp/r2-state.bin

# Include exact per-frontier compatibility for a supplied dual tile.
cargo run --release -p cascadia-r2 -- tokenize \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-train-part-0 \
  --ordinal 0 \
  --supplied-tile forest,river,0x1f,false

# Decode and validate canonical CSR2SP1 bytes.
cargo run --release -p cascadia-r2 -- decode \
  --input /tmp/r2-state.bin

# Export the frozen R2 MLX cache after ADR 0146 corpus authorization.
cargo run --release -p cascadia-r2 -- export-mlx \
  --corpus-lock ../../artifacts/experiments/r2-sparse-mlx-architecture-tournament-v1/control/corpus-lock.json \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-train-part-0 \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-train-part-1 \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-train-part-2 \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-train-part-3 \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-validation-part-0 \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-validation-part-1 \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-validation-part-2 \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-validation-part-3 \
  --output-root ../../artifacts/experiments/r2-sparse-mlx-architecture-tournament-v1/caches
```

The complete accepted-corpus command is in
`docs/v2/reports/r2-sparse-occupied-frontier-preregistration.md`.
The matched MLX protocol and launch gates are in ADR 0146 and
`docs/v2/reports/r2-sparse-mlx-architecture-tournament-v1-preregistration.md`.

The MLX export is exact and content-addressed. `CSR2SP1` remains authoritative
for occupied/public state; Rust regenerates frontier, component, motif, and
graph projections, validates them, and freezes the resulting training tensors
before Python can open the cache. Python does not rederive those semantics.

The cache uses four board-local blocks of 92 rows, for 368 padded rows per
position. It preserves explicit relative-seat ownership, reports active and
padding counts plus per-type accounting, forbids truncation, and verifies
every D6 transform in Rust before finalizing the cache. The accepted
foundation has per-board P99 83, per-board maximum 92, and position maximum
340.

R2-MAP does not persist that padded cache. It compacts active token rows,
streams checksummed parent/selected-afterstate frames from validated `.r2sh`
replays, and lets Python pad only the current batch. See
`docs/v2/R2_MAP_DATASET_BRIDGE.md`.

## Supplied Tile Syntax

```text
TERRAIN_A,TERRAIN_B_OR_NONE,WILDLIFE_MASK,KEYSTONE
```

Terrains accept names or codes `0..4`. The mask accepts decimal or `0x`
hexadecimal. Examples:

```text
forest,river,0x1f,false
mountain,none,1,true
```

Terrain-compatible rotations have at least one matching directed habitat
edge. Every canonical rotation remains rules-legal.

## Verification

```bash
cargo fmt --all -- --check
cargo test -p cascadia-r2 --all-targets
cargo clippy -p cascadia-r2 --all-targets -- -D warnings
```

The corpus census validates every row through public reconstruction, packed
round-trip, independent frontier and habitat oracles, target mutation, and all
12 D6 transform/inverse pairs. Parallel row evaluation preserves input order,
so scientific output is deterministic across Rayon thread counts.
