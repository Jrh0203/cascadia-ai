# R2 Sparse Occupied-Plus-Frontier Foundation Result

Date: 2026-06-17

Experiment ID: `r2-sparse-occupied-frontier-foundation-v1`

Contract: ADR 0145

Verdict: **PASS - authorize a matched local MLX prototype**

Scientific BLAKE3:
`186ad8934287ef0a74a166ed00cc9ebe857dcded20faa01a264974e1eb7081e6`

## Executive Result

The exact sparse public substrate passed every mechanical and serving-shape
gate on the accepted 60,000-row R0 corpus.

Across all four relative boards, one position contains:

| Token layer | Median | P90 | P99 | Maximum |
|---|---:|---:|---:|---:|
| Occupied tiles | 51 | 83 | 91 | 91 |
| Legal frontier | 71 | 90 | 97 | 107 |
| Habitat components | 37 | 54 | 61 | 69 |
| Wildlife motif anchors | 39 | 71 | 79 | 79 |
| **Total spatial tokens** | **199** | **299** | **323** | **340** |

The preregistered limits were P99 <= 512 and maximum <= 640. The observed
P99 uses 63.1% of its budget; the maximum uses 53.1% of its hard budget.
No state is clipped, overflowed, pooled away, or truncated.

Canonical packed states have:

| Packed bytes | Value |
|---|---:|
| Mean | 522 |
| Median | 518 |
| P90 | 774 |
| P99 | 838 |
| Maximum | 838 |

The original `compact-entity-v2` record is 864 bytes. Median sparse packed
state is 59.95% of that size, while P99 and maximum are 96.99%. The packed
state also excludes all 11 terminal targets.

This result authorizes R2 MLX representation work. It does not establish
training throughput, inference latency, decision quality, search quality, or
gameplay strength.

## Corpus And Exactness

The run used the eight frozen train and validation roots in the exact
preregistered order:

- 50,000 train positions;
- 10,000 validation positions;
- 60,000 total public positions; and
- 240,000 relative boards.

Every row passed:

- source manifest and shard validation;
- exact occupied-entity reconstruction;
- exact public PositionRecord reconstruction;
- canonical `CSR2SP1` pack/unpack;
- byte-identical encode after decode;
- independently computed legal-frontier set equality;
- independent directed-edge habitat graph equality;
- exact wildlife entity projection;
- all 12 D6 transform/inverse pairs;
- directed edge terrain permutation under D6; and
- mutation of all terminal target fields without any token or byte change.

The adversarial suite also places a legal 23-tile chain against the rules-grid
boundary and proves that occupied coordinate 24 is retained while impossible
frontier coordinate 25 is excluded from the legal set.

The corpus executed 720,000 D6 transform/inverse pairs and 60,000 target
independence checks. No failure or skipped row occurred.

## Per-Board Shape

The individual relative-board distributions are:

| Token layer | Median | P90 | P99 | Maximum |
|---|---:|---:|---:|---:|
| Occupied tiles | 13 | 21 | 23 | 23 |
| Legal frontier | 18 | 23 | 26 | 31 |
| Habitat components | 9 | 14 | 17 | 22 |
| Wildlife motif anchors | 10 | 18 | 20 | 20 |
| **Total spatial tokens** | **50** | **75** | **83** | **92** |

This is a useful architecture fact. A model can batch one shared four-board
state at roughly 200 typical tokens, or preserve board ownership explicitly
with fewer than 100 tokens on every observed board.

## Dense-Capacity Comparison

Fractions compare token count with the corresponding dense cell count across
four boards.

### Occupied tokens only

| Dense support | Median fraction | P99 fraction | Maximum fraction |
|---|---:|---:|---:|
| 61 cells | 20.90% | 37.30% | 37.30% |
| 91 cells | 14.01% | 25.00% | 25.00% |
| 127 cells | 10.04% | 17.91% | 17.91% |
| 441 cells | 2.89% | 5.16% | 5.16% |

### All four sparse spatial layers

| Dense support | Median fraction | P99 fraction | Maximum fraction |
|---|---:|---:|---:|
| 61 cells | 81.56% | 132.38% | 139.34% |
| 91 cells | 54.67% | 88.74% | 93.41% |
| 127 cells | 39.17% | 63.58% | 66.93% |
| 441 cells | 11.28% | 18.31% | 19.27% |

The 61-cell comparison is allowed to exceed 100% because R2 explicitly
represents frontier, component, and motif objects instead of only grid cells.
That is not overflow: token arrays are variable-length and exact.

The important comparison is information density. Even after adding three
first-class relational layers, R2's maximum token count is only 66.9% of a
four-board radius-6 / 127-cell tensor and 19.3% of the historical 441-cell
shape.

## Frontier And Component Findings

The legal frontier is not a small side channel. It averages 69.27 tokens per
position, compared with 51.5 occupied tiles. Any model that omits frontier
objects asks the network to reconstruct the largest explicit spatial set from
indirect evidence.

The corpus contains:

- 65,500 frontier tokens that can bridge distinct habitat components for at
  least one terrain, 1.58% of all frontier tokens; and
- 1,014,488 frontier tokens contacting the same habitat component on multiple
  edges, 24.41% of all frontier tokens.

Those exact local facts are common enough to justify first-class fields.
Their strategic value remains a learned-model question.

## Wildlife Motif Boundary

The V1 motif layer emits one exact anchor per placed wildlife and carries
directional neighboring wildlife. It reconstructs the complete represented
wildlife set and reaches a maximum of 79 tokens per position.

It is not a complete Card A quotient. The foundation does not yet emit exact
Bear pair objects, alternative Elk lines, Salmon path alternatives, Hawk
conflict graphs, or Fox completion sets. Those remain a separate R2/S3
extension and must preserve the occupied anchors until their own losslessness
is proved.

## Optional Supplied-Tile Compatibility

When `--supplied-tile` is provided, every frontier token deterministically
adds all canonical rotations, matching edge bits, all-present-match status,
and exact component merge results. Single-terrain keystones have one
canonical rotation; dual-terrain tiles have six.

Compatibility is derived and is not redundantly serialized. Decode and D6
transforms regenerate it from the authoritative occupied state and supplied
tile semantics.

## Determinism

The complete census was run twice:

1. Rayon default thread policy;
2. `RAYON_NUM_THREADS=3`.

The outputs are byte-identical:

```text
file BLAKE3:
5cced4cc82f6203577a2cec90114b9171b782784ecc31ae0f883e83221980f2c

scientific BLAKE3:
186ad8934287ef0a74a166ed00cc9ebe857dcded20faa01a264974e1eb7081e6
```

Paths, timestamps, hostnames, timings, and thread counts are absent from the
scientific payload.

## Promotion Assessment

| Gate | Result |
|---|---|
| Exact public reconstruction | Pass |
| Exact canonical codec | Pass |
| Frontier oracle equality | Pass |
| Habitat graph oracle equality | Pass |
| Exact D6 inverse | Pass |
| Target independence | Pass |
| P99 total tokens <= 512 | Pass, 323 |
| Maximum total tokens <= 640 | Pass, 340 |
| P99 packed bytes <= 864 | Pass, 838 |
| Silent truncation | None |
| Learned-quality claim | Not made |
| Gameplay claim | Not made |

The next authorized experiment is a matched local MLX substrate comparison:

1. padded Set Transformer;
2. directional graph message passing plus global attention; and
3. Perceiver-style fixed latents.

All arms must consume the same exact R2 tokens, targets, D6 schedule, optimizer
budget, and legal actions. The state trunk must be encoded once per decision.

## Reproduction

```bash
cd /Users/johnherrick/cascadia/tools/r2_sparse_entity_census

cargo fmt --manifest-path Cargo.toml -- --check
cargo test
cargo clippy --all-targets -- -D warnings

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

## Evidence

- Census:
  `artifacts/experiments/r2-sparse-occupied-frontier-foundation-v1/census.json`
- Independent repeat:
  `artifacts/experiments/r2-sparse-occupied-frontier-foundation-v1/census-repeat.json`
- Repeat proof:
  `artifacts/experiments/r2-sparse-occupied-frontier-foundation-v1/repeat-proof.json`
- Verification summary:
  `artifacts/experiments/r2-sparse-occupied-frontier-foundation-v1/verification.json`
- ADR:
  `docs/v2/decisions/0145-r2-sparse-occupied-frontier-foundation.md`
- Preregistration:
  `docs/v2/reports/r2-sparse-occupied-frontier-preregistration.md`

## Scientific Limitations

- The corpus contains pre-move turns 0 through 79, not terminal turn-80
  PositionRecords.
- Source coordinates remain bounded by the existing rules engine and
  `compact-entity-v2` coordinate scalar domain. R2 removes representation
  support bounds; it does not redefine game legality.
- Component IDs are deterministic within a state. They are regenerated after
  D6 transforms rather than claimed to be transform-invariant labels.
- The habitat bridge fields are exact local graph facts, not strategic value.
- The wildlife layer is lossless for represented entities but incomplete as a
  scoring quotient.
- No MLX kernel, training loop, decision metric, search benchmark, or game was
  run in this foundation experiment.

## Files Added

- `tools/r2_sparse_entity_census/Cargo.toml`
- `tools/r2_sparse_entity_census/Cargo.lock`
- `tools/r2_sparse_entity_census/README.md`
- `tools/r2_sparse_entity_census/src/lib.rs`
- `tools/r2_sparse_entity_census/src/model.rs`
- `tools/r2_sparse_entity_census/src/codec.rs`
- `tools/r2_sparse_entity_census/src/census.rs`
- `tools/r2_sparse_entity_census/src/main.rs`
- `tools/r2_sparse_entity_census/tests/common/mod.rs`
- `tools/r2_sparse_entity_census/tests/foundation.rs`
- `tools/r2_sparse_entity_census/tests/adversarial.rs`
- `docs/v2/decisions/0145-r2-sparse-occupied-frontier-foundation.md`
- `docs/v2/reports/r2-sparse-occupied-frontier-preregistration.md`
- `docs/v2/reports/r2-sparse-occupied-frontier-result.md`
- `artifacts/experiments/r2-sparse-occupied-frontier-foundation-v1/census.json`
- `artifacts/experiments/r2-sparse-occupied-frontier-foundation-v1/census-repeat.json`
- `artifacts/experiments/r2-sparse-occupied-frontier-foundation-v1/repeat-proof.json`
- `artifacts/experiments/r2-sparse-occupied-frontier-foundation-v1/verification.json`

No existing Rust or Python source file, workspace manifest, queue, dashboard,
ledger, or shared data module was modified.
