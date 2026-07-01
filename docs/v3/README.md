# Cascadia V3

Cascadia v3 is the transformer-based training and search stack for pushing
four-player Cascadia beyond the previous neural/search plateau. The live path is
not an NNUE campaign and not a v2 MLX continuation; it is CascadiaFormer over
packed expert tensors with search-supervised action values.

## Canonical Docs

- [Architecture](ARCHITECTURE.md): model shape, tokenization, relation bias,
  public-boundary rules, and literature basis.
- [Training Pipeline](TRAINING_PIPELINE.md): data generation, objectives,
  expert iteration, checkpointing, and promotion gates.
- [Operations](OPERATIONS.md): local, john0 GPU, and Bacalhau worker workflows.
- [Performance](PERFORMANCE.md): measured loader/training/gameplay facts.

The implementation package lives in
[cascadiav3/README.md](../../cascadiav3/README.md).

## Current Scientific State

- Real training data should use packed `.npz` tensor shards.
- JSONL is retained only for tiny audit fixtures.
- The default CascadiaFormer board fast path is radius 6: 127 canonical cells
  plus exact overflow entities.
- The model must learn score-to-go, and serving must rank by
  `exact_afterstate_score_active + predicted_score_to_go`.
- Current transformer baselines have reached the greedy neighborhood, but have
  not yet surpassed greedy in gameplay.
- The next meaningful strength run is EI-0: greedy-state search bootstrap with
  selected-preserving K32 tensors, corrected Q semantics, and
  search-improved greedy retention.

## Historical Recovery

The pre-cleanup v1/v2 archive, older planning memos, v2 MLX package, web app,
legacy teacher bridge, and rejected experiment attic were removed from `main`.
Recover them from:

```bash
git show archive/pre-v3-repo-cleanup-2026-07-01:<path>
```
