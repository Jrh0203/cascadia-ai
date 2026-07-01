# Corrected Mid-Tail V1 Champion Migration

Date: 2026-06-17

Experiment ID: `corrected-mid-tail-v1`

Schema ID: `legacy-mid-v4-fixed-v1`

Foundation gate: 12

Verdict: **PASS**

## Result

The production champion was migrated exactly once through the existing Rust
`migrate_legacy_mid_v4_weights` example compiled with
`legacy-mid-v4-fixed-v1`. The result passed a structurally independent Python
binary audit and was published as a read-only, content-addressed artifact.

Production source:

```text
path: nnue_weights_v4opp_modal_iter3.bin
bytes: 23,134,992
BLAKE3: 9e1d568693274fc537ac4f6d6f729abb1ee8da8330a78d1f78a1f62b733de400
SHA-256: f40627623d3686d7d2d6a2f8f109445f54e449f0d7045552ebe831f955a58f48
container: NNUE
head-format version: 1
architecture: 11,231 -> 512 -> 64
```

Corrected artifact:

```text
model ID: blake3:a3e72314adeb4d62077e43ff071f95b27f979aba17f5026699118b19600263d0
bytes: 23,135,024
BLAKE3: a3e72314adeb4d62077e43ff071f95b27f979aba17f5026699118b19600263d0
SHA-256: 6bd42185cf0eed8513c3295ce3cc6cc3a781e054d4486bb468f3fb8913128e0a
container: NNUC version 1
schema tag: MIDTAIL-CORR-V1\0
head-format version: 1
architecture: 11,231 -> 512 -> 64
```

Artifact directory:

[`artifacts/experiments/corrected-mid-tail-v1/models/blake3/a3e72314adeb4d62077e43ff071f95b27f979aba17f5026699118b19600263d0`](../../../artifacts/experiments/corrected-mid-tail-v1/models/blake3/a3e72314adeb4d62077e43ff071f95b27f979aba17f5026699118b19600263d0)

The directory contains:

- `nnue_weights_legacy_mid_v4_fixed_v1_init.bin`
- `manifest.json`
- `manifest.blake3`
- `audit.json`

All four files are mode `0444`; the content-addressed directory is mode
`0555`.

## Source Discovery

Repository-wide discovery found exactly one file named
`nnue_weights_v4opp_modal_iter3.bin`, at the repository root. The file is
ignored by the repository's `*.bin` rule, so its identity is established by
both BLAKE3 and SHA-256 rather than Git object identity.

The migration publisher hard-codes the production source size and both hashes.
Any source drift is rejected before Cargo is invoked.

## Migration Invocation

The publisher invoked this command contract:

```bash
cargo run --locked --quiet \
  -p cascadia-ai \
  --example migrate_legacy_mid_v4_weights \
  --features legacy-mid-v4-fixed-v1 \
  -- nnue_weights_v4opp_modal_iter3.bin <staging-output>
```

The exact migration implementation, auditor, Cargo manifests and lockfile,
ADR, and preregistration are recorded in the manifest. Their combined source
identity is:

```text
BLAKE3: c33dec02e75dd33b90e315341b1825948d82ac1896c374ba21c9640fd9c3c6ff
files: 9
```

No existing Rust, MLX, scheduler, dashboard, or queue source was changed by
this gate.

## Independent Audit

The audit is implemented in
`tools/corrected_mid_tail_champion_audit.py`. It does not call or import the
Rust NNUE loader. It parses the documented `NNUE` and `NNUC` layouts directly,
compares raw byte regions, validates IEEE-754 zero bit patterns, and requires
exact file consumption.

Immutable audit receipt:

```text
verdict: pass
bytes: 6,591
BLAKE3: 173d1d982ea8843ead759dc8626cfaf807e336532b4da431567efcaaf56100eb
SHA-256: 2c16a01734b498d49fa02653ef649d729b021820f4ce142e9fcf5ee2a6c83d78
```

### Header

The corrected 40-byte header is exactly:

```text
4e4e5543 01000000 01000000 4d49445441494c2d434f52522d563100
df2b0000 00020000 40000000
```

Decoded:

| Field | Value |
|---|---|
| Magic | `NNUC` |
| Container version | `1` |
| Head-format version | `1` |
| Schema tag | `MIDTAIL-CORR-V1\0` |
| Feature count | `11,231` |
| Hidden layer 1 | `512` |
| Hidden layer 2 | `64` |

### First Layer

| Contract | Bytes | Audit |
|---|---:|---|
| Source `0..10561` to destination `0..10561` | 21,628,928 | Byte-identical |
| Source `10561..10862` | 616,448 | Discarded; no destination |
| Source `10862..11231` to destination `10561..10930` | 755,712 | Byte-identical |
| Destination `10930..11231` | 616,448 | Exact signed zero |

Block receipts:

```text
base BLAKE3:
d5ad2fda3e14faaf4cbbae423d7de6eaeb442c879ae0ca8fecb00ed293ad937e

discarded defect BLAKE3:
e52f708360e6b9207366a1939d0bfb0e2d853740fb0116f28e6d8a137b2dae7c

opponent detail BLAKE3:
bd083250b3a97f0aeb889d5eab603f8ecad3a73e614b939cf4fe9de366848f2c

corrected tail BLAKE3:
aa77ec47bd1f7022e5c7061ddf110d00e72c84759c1646e5223c8fb86525ed4f
```

The historical defect block contains 154,112 nonzero float bit patterns. This
is expected: the feature rows were never activated, but their initialized
weights were not zero. None are retained or reinterpreted.

The corrected tail contains exactly:

```text
positive zero: 154,112
negative zero: 0
nonzero: 0
```

### Downstream Tensors

All 133,896 downstream bytes are byte-identical:

```text
BLAKE3: af2e09a962cc8aa1e2efd2b34876d80150ad4f894c9fbf5cc3489cc12add2cb7
```

The independent audit compares each tensor separately:

| Tensor | Floats | Bytes | Result |
|---|---:|---:|---|
| `b1` | 512 | 2,048 | Byte-identical |
| `w2` | 32,768 | 131,072 | Byte-identical |
| `b2` | 64 | 256 | Byte-identical |
| `w3` | 64 | 256 | Byte-identical |
| `b3` | 1 | 4 | Byte-identical |
| `w3_policy` | 64 | 256 | Byte-identical |
| `b3_policy` | 1 | 4 | Byte-identical |

Both source and corrected files end exactly after `b3_policy`; neither has
trailing bytes.

## Immutability And Idempotence

Publication uses a staging directory under the model store, obtains an
exclusive repository-local lock, and publishes only after the independent
audit passes. The final directory name is the corrected model's BLAKE3 digest.

Existing artifacts are never overwritten. On reuse, the publisher verifies:

- the current production source identity;
- the exact immutable file set;
- model, audit, and manifest hashes;
- the detached manifest BLAKE3 receipt;
- read-only permissions;
- the content-address/directory match; and
- a fresh independent audit equal to the stored audit.

The first production run reported:

```text
reused: false
audit_verdict: pass
```

An immediate second run reported:

```text
reused: true
audit_verdict: pass
```

The second run did not invoke the Rust migration command.

## Verification

Focused tooling verification:

```text
17 passed
ruff: all checks passed
trailing-whitespace scan: clean
```

The 17 tests cover exact mapping, every corrupted payload class, signed
negative zero, source drift, malformed magic, trailing bytes, read-only
publication, idempotent reuse, failed-audit cleanup, model tampering, manifest
tampering, unexpected files, symlink rejection, and the exact Cargo command
contract.

Rust foundation verification:

```text
cargo test -p cascadia-ai --lib \
  --features legacy-mid-v4-fixed-v1 \
  -- --test-threads=1

94 passed; 0 failed
```

Historical-layout regression:

```text
cargo test -p cascadia-ai --lib \
  --features mid-features,v4-opp \
  historical_champion_layout_remains_frozen \
  -- --test-threads=1

1 passed; 0 failed
```

Migration example build:

```text
cargo check --locked -p cascadia-ai \
  --example migrate_legacy_mid_v4_weights \
  --features legacy-mid-v4-fixed-v1

passed
```

Existing compiler warnings remain outside this gate's ownership; no new Rust
warnings or failures were introduced.

## Gate Status

Foundation acceptance gate 12 is complete.

This artifact is an untrained corrected-schema initialization. It is not a new
gameplay-strength claim and must not be evaluated with the historical
extractor.

Gate 13 remains required before Apple-cluster neural training: port the
corrected extractor and checkpoint to MLX and prove byte-for-byte first-layer
row and prediction parity.
