# Cascadia V3

- [Research specification](CASCADIA_V3_RESEARCH_SPEC.md)
- [Stockfish NNUE teacher-search research report](STOCKFISH_NNUE_TEACHER_SEARCH_REPORT.md)
- [Architecture](ARCHITECTURE.md)
- [Operations](OPERATIONS.md)
- [Full v3 training pipeline](FULL_V3_TRAINING_PIPELINE.md)
- `PERFORMANCE.md` records the two optimization passes and measured readiness rates.
- `READINESS.md` is generated from the sealed Part 1 manifest.

The live campaign state and large artifacts are intentionally outside the repository at `/Users/johnherrick/cascadia-bench/v3-nnue`.

The `tools/v3_*.py` contracts cover the checksum-chained controller, topology-free Phase 2 work and training schedules, promotion and final-report statistics, capacity projection, canonical Docker identity, live worker-retry drill, and Part 1 infrastructure qualification.
