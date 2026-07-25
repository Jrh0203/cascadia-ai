# Cascadia V3 Implementation

This directory contains the active transformer implementation:

- Python/PyTorch CascadiaFormer models and trainers;
- the Rust real-root and packed-tensor exporter;
- self-play, training, evaluation, and fleet scripts;
- focused correctness and performance tests.

See:

- [Training pipeline](../docs/v3/TRAINING_PIPELINE.md)
- [Infrastructure](../docs/v3/INFRASTRUCTURE.md)
- [Strength-first loop](../docs/v3/RESEARCH_PIPELINE_GUIDE.md)

## Common commands

```bash
cargo build --release \
  --manifest-path cascadiav3/real-root-exporter/Cargo.toml

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src \
  uv run python -m unittest discover -s cascadiav3/tests -v

MODEL_MANIFEST=<checkpoint.manifest.json> \
PROFILE=<name> \
bash cascadiav3/scripts/run_gumbel_selfplay_cycle.sh launch
```

Direct checkpoint or search comparison:

```bash
MATCHUP_NAME=<name> \
BASE_MANIFEST=<baseline.manifest.json> \
CAND_MANIFEST=<candidate.manifest.json> \
GAMES=32 \
bash cascadiav3/scripts/run_paired_gate.sh
```

The historical filename is retained for compatibility; it now runs a visible
comparison, not a promotion gate.

## Artifacts

Large generated data lives under `fixtures/`, `reports/`, `checkpoints/`,
`fleet/`, and `logs/`. New runs use ordinary paths, configs, metrics, and
checkpoints. Hashes, receipts, source identities, and scientific-eligibility
labels in older files are ignored by active readers.
