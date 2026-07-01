# V3 Architecture

## Code boundaries

- `crates/cascadia-v3-nnue`: exact feature schema, overflow/D6 handling, virtual opportunity catalog, accumulators, quantized inference, legal-action ranking, terminal search, and packed records.
- `python/cascadia_v3_mlx`: native CSR streaming, deterministic sparse gradients, exact-QAT MLX graph, resumable training, profiling, quantized export, and cross-backend verification.
- `crates/cascadia-differential/src/bin/v3_campaign_worker.rs`: immutable-container collection and validation worker using the proven V2 rules and qualified V1 policy loader.
- `tools/v3_campaign.py`: checksum-chained two-part campaign state machine, storage guard, dashboard projection, readiness qualification, and authorization gate.
- `tools/v3_promotion.py`: cycles 2–10 robust bounded paired promotion test.
- `tools/v3_promotion_v1.py`: immutable Cycle-1 decision rule retained for exact reproduction.
- `tools/v3_final_report.py`: protected and all-V3 final aggregation.

## Hot path and overflow

Radius-7 rows are incrementally addressable through precomputed coordinate and D6 tables. Overflow is rebuilt only when a transition crosses the radius-7 boundary. Overflow slots are sorted by exact absolute coordinate, so replay, augmentation, and apply/undo have one canonical encoding. An outside-radius legal destination follows the same afterstate encoder and is never clipped or rejected.

## Opportunity virtual features

Inference rows are exact catalog conjunctions. The native loader expands each active inference row into 3–7 train-only factor rows and consolidates counts. MLX learns both tables. At serving export, each inference vector becomes

```text
inference_row + sum(virtual_factor_rows_for_inference_row)
```

and the factor table is discarded. The schema checksum binds the complete row-to-factor map.

## Deterministic training

The CSR forward kernel performs fused embedding-bag accumulation without materializing `[batch, active_features, 1024]`. Backward uses a stable feature-sorted reduction: one Metal workgroup lane deterministically sums all occurrences for a feature. Floating-point atomics are forbidden because their ordering breaks exact restart. Checkpoints bind model, optimizer, loader cursor, dataset checksums, binary checksum, seed, D6 schedule, and origin identity.

The run manifest also hashes every V3 MLX source module, the shared atomic-checkpoint implementation, `pyproject.toml`, `uv.lock`, and the Python/MLX runtime identity. Exported serving manifests carry that run-manifest checksum, so a numerically valid weight file cannot become detached from the code that produced it.

Sparse transformer sums are retained exactly as int32 values through incremental apply/undo. The activation boundary clips to `[0, feature_scale]` and narrows to int16 before product pooling, matching the QAT graph exactly. Training adds a deterministic headroom penalty above 64 float units so accumulator magnitude is controlled as an optimization objective rather than by lossy runtime saturation.

## Search

Every legal action is enumerated before scoring. The optimized path reuses public-state, market-draft, opportunity-graph, and board-transition work, but a retained reference implementation reconstructs each afterstate independently. Fixed-corpus tests require identical legal sets, outputs, ordering, and selected moves.

The network predicts score-to-go. Policy ranking therefore adds the exact current afterstate score in integer output units before sorting; truncated rollouts do the same. This keeps training labels and serving semantics identical, including positions where a later wildlife placement can reduce a previously scored motif.

K32/R64 and K32/R600 use deterministic sequential halving over the direct top 32. Hidden futures are redeterminized only after applying the observed root action. Teacher packets retain eliminated-candidate statistics so training provenance does not silently collapse to the winner.
