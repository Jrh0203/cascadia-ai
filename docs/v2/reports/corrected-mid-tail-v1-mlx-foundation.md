# Corrected Mid-Tail V1 MLX Foundation

Date: 2026-06-17

Experiment ID: `corrected-mid-tail-v1`

Schema ID: `legacy-mid-v4-fixed-v1`

Gate: F5 foundation gate 13

Status: passed

## Result

The corrected historical NNUE schema now has an exact MLX implementation.
Python validates the schema-tagged `NNUC` version-one container, preserves the
frozen historical `NNUE` path, exposes the corrected row contract, supports
NNUE head-format versions 1 through 4, and packages corrected checkpoints as
integrity-checked MLX artifacts.

The production proof used:

- the qualified historical champion;
- a checkpoint produced by the Rust
  `migrate_legacy_mid_v4_weights` implementation;
- the existing 80-decision Rust prediction fixture; and
- a corrected MLX safetensors artifact loaded through the production artifact
  validator.

All first-layer, downstream-tensor, artifact, and prediction gates passed.
Rust and MLX predictions were bit-identical on all 80 records.

## Implemented Contract

### Fail-closed container parsing

`python/cascadia_mlx/legacy_nnue.py` now distinguishes:

- historical `NNUE`, whose first-layer meaning is inferred only from a
  recognized historical width; and
- corrected `NNUC`, whose 40-byte header must contain container version `1`,
  head version `1..4`, schema tag `MIDTAIL-CORR-V1\0`, and the requested
  feature and hidden dimensions.

The corrected parser rejects unknown magic, unsupported container or head
versions, schema-tag mismatch, dimension mismatch, malformed payload size,
truncation, trailing bytes, incomplete optional heads, and non-finite tensors.

### Corrected first-layer layout

| Block | Range | Width |
|---|---:|---:|
| Historical v2 base | `0..10561` | 10,561 |
| Opponent detail | `10561..10930` | 369 |
| Extended tile-terrain counts | `10930..11080` | 150 |
| Extended tile-wildlife capacity | `11080..11230` | 150 |
| Overflow used | `11230..11231` | 1 |

The module exports this layout as named ranges plus deterministic mapping
functions for all six Rust-recognized historical widths:

```text
5197, 5566, 7670, 10561, 10862, 11231
```

Order and duplicate multiplicity are preserved when sparse feature lists are
remapped. Activations in a discarded historical defect range fail closed by
default.

### Head-version support

The parser and MLX artifact retain:

- version 1: value and policy heads;
- version 2: two split value heads;
- version 3: eleven split value heads; and
- version 4: the heteroscedastic head.

Standard MLX inference selects the same value-head priority as Rust. The
Rust-order MLX path has dedicated Metal output kernels for the two-head and
eleven-head formats, preserving operation order instead of approximating these
formats through a generic matrix reduction.

### Corrected MLX artifact

The new artifact schema records:

- corrected architecture and schema IDs;
- `NNUC` container, schema-tag, and head versions;
- exact dimensions and row ranges;
- source checkpoint bytes and BLAKE3; and
- safetensors bytes and BLAKE3.

It deliberately omits source paths and filenames. Converting byte-identical
checkpoints from different names and paths produced byte-identical manifests
and safetensors, so the artifact identity is host-independent.

Existing qualified historical artifacts remain schema versions 1 and 2 and
retain their prior validation contract. The service required no protocol
change because corrected and historical production models share the same
11,231-row sparse index width; artifact loading now provides the schema
boundary.

## Production Parity Evidence

### Inputs and artifacts

| Item | Bytes | BLAKE3 |
|---|---:|---|
| Historical champion | 23,134,992 | `9e1d568693274fc537ac4f6d6f729abb1ee8da8330a78d1f78a1f62b733de400` |
| Rust-migrated corrected checkpoint | 23,135,024 | `a3e72314adeb4d62077e43ff071f95b27f979aba17f5026699118b19600263d0` |
| Corrected MLX safetensors | 23,135,528 | `2cdff5de26ce019df1f25742b681bf67ae8216600eb83264e3aad7b13eca3356` |
| Corrected MLX manifest | n/a | `e52f0fd074ab07fb7071d2a6ee1c7418fadfb6b9c1d32f5b54ec5387e5791e9c` |
| Rust prediction fixture | 295,258 | `1e1a89d4ca2a540587793a0fe681b11de80e661f6d419328c59f31e910797238` |

### First-layer audit

| Check | Result | BLAKE3 |
|---|---|---|
| 10,561 base rows copied byte-for-byte | pass | `d5ad2fda3e14faaf4cbbae423d7de6eaeb442c879ae0ca8fecb00ed293ad937e` |
| 369 opponent rows remapped byte-for-byte | pass | `bd083250b3a97f0aeb889d5eab603f8ecad3a73e614b939cf4fe9de366848f2c` |
| 301 corrected-tail rows are signed zero | pass | `aa77ec47bd1f7022e5c7061ddf110d00e72c84759c1646e5223c8fb86525ed4f` |
| Every non-first-layer tensor is byte-identical | pass | exact byte comparison |

### Prediction audit

The Rust fixture contains no activation in historical rows `10561..10862`.
Its opponent rows were remapped into corrected rows `10561..10930` without
changing order or multiplicity.

| Comparison | Max error | P99 error | Mean error | Bit-identical |
|---|---:|---:|---:|---|
| Historical Python Rust-order reference vs Rust fixture | 0.0 | 0.0 | 0.0 | yes |
| Corrected reference vs historical reference | 0.0 | 0.0 | 0.0 | yes |
| Direct corrected MLX vs Rust fixture | 0.0 | 0.0 | 0.0 | yes |
| Packaged corrected MLX artifact vs Rust fixture | 0.0 | 0.0 | 0.0 | yes |

The common 80-prediction byte stream has BLAKE3:

```text
072068d4284a32b3f5f26232aca99597609d1951c5d2ee9464c3d8751536cfe0
```

## Permanent Verification

Focused tests:

```text
41 passed
```

Coverage includes:

- valid `NNUC` containers for head versions 1, 2, 3, and 4;
- corruption of every corrected header field;
- truncation, trailing bytes, and non-finite tensors;
- exact migration of all six recognized historical widths;
- policy-less historical checkpoint compatibility;
- unknown historical-width rejection;
- corrected sparse-feature mapping and defect-range rejection;
- Rust-order MLX parity for split-two and split-eleven value heads;
- corrected artifact conversion, loading, and manifest corruption; and
- the existing historical NNUE and sparse-service regression suites.

Ruff lint and formatting checks pass for the changed implementation and test
files. `git diff --check` reports no whitespace errors.

## Reproduction

```bash
cargo run -p cascadia-ai \
  --example migrate_legacy_mid_v4_weights \
  --features legacy-mid-v4-fixed-v1 \
  -- nnue_weights_v4opp_modal_iter3.bin corrected.bin

uv run cascadia-mlx-legacy-nnue convert-corrected \
  --source corrected.bin \
  --output corrected-mlx

uv run cascadia-mlx-legacy-nnue corrected-parity \
  --historical-source nnue_weights_v4opp_modal_iter3.bin \
  --corrected-source corrected.bin \
  --model-dir corrected-mlx \
  --fixture artifacts/fixtures/legacy-nnue-v4opp-mlx-v1-rust.json \
  --output corrected-parity.json

uv run pytest -q \
  python/tests/test_legacy_nnue.py \
  python/tests/test_legacy_nnue_corrected.py \
  python/tests/test_legacy_nnue_serve.py
```

## Ownership Boundary

The production proof generated its corrected checkpoint and MLX artifact in a
temporary directory. Their hashes are recorded above, but this task did not
publish them into the repository because immutable checkpoint publication is
foundation gate 12 and was outside the assigned file ownership.

Gate 13 itself is complete. Apple-cluster neural work can use this
implementation once the gate-12 artifact owner publishes and independently
audits the immutable corrected checkpoint.
