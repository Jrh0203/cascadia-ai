# Exact D6 Transform Contract V1 Result

Date: 2026-06-16

Final classification:
**`exact_d6_transform_contract_complete`**

Scientific BLAKE3:
`db6ac2f9f6ebe2daaa2db603c6c16183512b5d989aed6979e1991e167737633f`

The exact D6 contract is now shared from Rust through Python/MLX. Rust remains
authoritative, the permanent metadata artifact is generated and byte-checked
by a production Rust command, and Python validates and consumes that artifact
without import-time subprocesses or independent geometry formulas in the
shared symmetry module.

| Gate | Result |
|---|---:|
| Stable D6 elements | 12 / 12 |
| `cascadia-game` tests | 70 passed, 0 failed |
| Focused Python tests | 13 passed, 0 failed |
| Radius-eight coordinates | 217 / 217 across 12 transforms |
| Direction cases | 72 / 72 |
| Tile-orientation cases | 72 dual + 72 single |
| Ordered transform compositions | 144 / 144 |
| Rust artifact byte check | passed |
| Rust formatting | passed |
| Clippy with warnings denied | passed |
| F3 Python Ruff format and lint | passed |
| Silent clipping | none |
| Import-time subprocesses | none |

The suite proves group identity, inverse, composition, and associativity;
coordinate and edge covariance; exact dual- and single-terrain rotation;
board, public-state, and full-state round trips; frontier and score invariance;
state-aware staged-action transformation; complete legal-action bijection;
transition equivariance; and policy permutation round trip, composition, and
argmax identity.

The cross-language suite additionally proves fresh Rust exporter output is
byte-identical to
`python/cascadia_mlx/d6_contract_metadata.v1.json`; Python tables equal the
fresh Rust tables; all 217 radius-eight coordinates, all directions, all tile
rotations, all 12 transforms, inverse and composition tables, reflected MLX
operations, single-terrain canonicalization, and legacy C6 API outputs agree.
Corrupted metadata and artifact drift fail closed.

## Deterministic Workflow

Regenerate only from Rust:

```bash
cargo run -p cascadia-game --bin d6_contract_metadata -- \
  --output python/cascadia_mlx/d6_contract_metadata.v1.json
```

Check for drift:

```bash
cargo run -p cascadia-game --bin d6_contract_metadata -- \
  --check python/cascadia_mlx/d6_contract_metadata.v1.json
```

Final verification:

```bash
cargo fmt --all -- --check
cargo test -p cascadia-game
cargo clippy -p cascadia-game --all-targets -- -D warnings
uv run pytest -q python/tests/test_d6_contract.py \
  python/tests/test_graded_oracle_local_geometry_model.py
uv run ruff format --check python/cascadia_mlx/d6_contract.py \
  python/cascadia_mlx/hex_symmetry.py python/tests/test_d6_contract.py
uv run ruff check --no-cache python/cascadia_mlx/d6_contract.py \
  python/cascadia_mlx/hex_symmetry.py python/tests/test_d6_contract.py
```

No gameplay, sealed data, ML training, cloud execution, or queue launch was
performed. A repository-wide Ruff sweep was also attempted but is currently
red on unrelated concurrent files; the F3-owned files pass both Ruff gates.
