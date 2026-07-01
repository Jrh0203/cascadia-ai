# R2 Sparse Occupied-Plus-Frontier Foundation Preregistration

Date: 2026-06-17

Experiment ID: `r2-sparse-occupied-frontier-foundation-v1`

Contract: ADR 0145

## Question

Can exact occupied, legal-frontier, habitat-component, and minimal wildlife
tokens represent the accepted public PositionRecord corpus without fixed
spatial support, hidden information, or a serving-shape outlier that blocks a
matched MLX prototype?

This is a representation-foundation experiment. It does not test learned
decision quality or gameplay.

## Frozen Corpus

The primary result must use the accepted R0 corpus in this exact order:

| Order | Dataset ID | Split | Rows | Manifest BLAKE3 |
|---:|---|---|---:|---|
| 0 | `pattern-aware-v1-k8-h6-b8-m4-train-200000` | train | 12,560 | `57f86b3f6ae06bee782974995aa6b8d3cad6f637e68d5ef8aac7ffd8112d4244` |
| 1 | `pattern-aware-v1-k8-h6-b8-m4-train-200157` | train | 12,480 | `79bcceebd52144f8c39130de15404f0f2820b695111f2f1e9004dcac5f33c555` |
| 2 | `pattern-aware-v1-k8-h6-b8-m4-train-200313` | train | 12,480 | `fbddc7aa1794b753fcbd3d8f030b51dcc4456051f61f7914eab541e9658db666` |
| 3 | `pattern-aware-v1-k8-h6-b8-m4-train-200469` | train | 12,480 | `8ab6d2a9229f3cfe8bf1567c3a9d110b9268e322a0c96cf30ba131c937435849` |
| 4 | `pattern-aware-v1-k8-h6-b8-m4-validation-210000` | validation | 2,560 | `a991d05962965d61a31d40fe0b8572c743cff04a12d1e948be9e2fa3e6a871d4` |
| 5 | `pattern-aware-v1-k8-h6-b8-m4-validation-210032` | validation | 2,480 | `adf3903a59d9d522fbb9fab2bb3c8a9370c7f2d46c3aa74ac85b6879b80efddc` |
| 6 | `pattern-aware-v1-k8-h6-b8-m4-validation-210063` | validation | 2,480 | `9bfeed300489ac6610313dd2bf032c809197be92cfeac43b357a4cb8aca14803` |
| 7 | `pattern-aware-v1-k8-h6-b8-m4-validation-210094` | validation | 2,480 | `7491212c5a524f954414402661a6aa064161a16cfe23755e051d80886b257186` |
| | **Total** | | **60,000** | |

Every manifest and shard must pass the existing `cascadia-data` validator.
The tool must reject a missing, reordered, substituted, or content-different
root when `--require-r0-corpus` is set.

## Frozen Token Semantics

The exact token and validation contract is ADR 0145. In particular:

- occupied tokens are authoritative;
- frontier, component, and motif layers are deterministic;
- no dense disk or overflow partition exists;
- targets and hidden supply order are excluded;
- all coordinates are exact source-domain coordinates;
- component membership comes from directed terrain matches;
- the wildlife layer is an exact anchor layer, not full scoring compression;
  and
- supplied-tile compatibility is optional and does not change token counts.

## Mechanical Gates

All 60,000 rows must pass:

1. exact public PositionRecord reconstruction when original targets are
   supplied only to the reconstruction API;
2. canonical pack/unpack equality;
3. byte-identical encode-after-decode;
4. independent frontier-set oracle equality;
5. independent habitat-component graph oracle equality;
6. exact wildlife entity projection;
7. all 12 D6 transforms followed by exact inverse;
8. directed edge terrain permutation under D6;
9. target mutation independence; and
10. deterministic corpus and scientific hashes.

Any failure invalidates promotion.

## Frozen Census Measurements

Report nearest-rank median, P90, P99, and maximum for:

- occupied tokens;
- legal-frontier tokens;
- habitat-component tokens;
- wildlife-motif tokens;
- total spatial tokens; and
- canonical packed bytes.

Report the distributions both per public position and per relative board.

For 61, 91, 127, and 441 cells per board, report occupied-token and total-token
fractions in parts per million, with `1,000,000` equal to the corresponding
dense row capacity across active players.

Also report:

- public normalized-position BLAKE3;
- packed-state stream BLAKE3;
- scientific JSON BLAKE3;
- D6 checks performed;
- habitat-bridge frontier count;
- repeated-component-contact frontier count; and
- packed-byte fraction versus the 864-byte `compact-entity-v2` record.

## Preregistered MLX-Foundation Promotion Gate

R2 may proceed to a matched local MLX prototype only if:

| Criterion | Threshold |
|---|---|
| Semantic mechanics | 100% of reconstruction, pack, frontier, component, motif, and D6 checks pass |
| Public-only input | Mutating all terminal targets changes neither tokens nor packed bytes |
| P99 serving shape | Total spatial tokens per position <= 512 |
| Hard outlier shape | Maximum total spatial tokens per position <= 640 |
| Serialized footprint | P99 packed bytes <= 864 |
| Truncation | None |

The token thresholds reflect the plan's first-pass 200-400 typical-token
target while reserving headroom for P99 and hard outliers. They are frozen
before the full census.

Passing authorizes only matched R2 MLX implementation work. It does not claim:

- better training throughput;
- lower inference latency;
- better target recall or retained regret;
- better value calibration;
- better search; or
- any score or gameplay improvement.

Those require separate frozen-data, equal-capacity comparisons against the
best qualified dense and exact-entity controls.

## Determinism Rule

Run the complete census twice with the same ordered content roots and different
output paths. The two JSON files and their BLAKE3 hashes must be byte-identical.
No timestamp, hostname, duration, root path, or output path may enter the
scientific payload.

## Primary Command

```bash
cd /Users/johnherrick/cascadia/tools/r2_sparse_entity_census

cargo run --release -- census \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-train-part-0 \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-train-part-1 \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-train-part-2 \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-train-part-3 \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-validation-part-0 \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-validation-part-1 \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-validation-part-2 \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-validation-part-3 \
  --require-r0-corpus \
  --output ../../artifacts/experiments/r2-sparse-occupied-frontier-foundation-v1/census.json
```
