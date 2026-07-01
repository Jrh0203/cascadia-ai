# V3 Operations

## Part 1 status and root

All V3 artifacts live under `/Users/johnherrick/cascadia-bench/v3-nnue`. The authoritative state is `control/campaign-state.json`; immutable transitions are under `control/transitions/`. The dashboard source is `control/dashboard-status.json`.

Inspect without mutating:

```bash
uv run python tools/v3_campaign.py status
```

Part 2 is sealed unless status is `awaiting_phase2_approval` and the supplied checksum exactly matches `reports/part1-readiness.json`.

## Phase 2 authorization

After reviewing readiness, John may authorize with:

```bash
uv run python tools/v3_campaign.py authorize-phase2 \
  --readiness-sha256 '<exact readiness_sha256>' \
  --approved-by John
```

This does not itself schedule bootstrap work. Every subsequent state transition requires a passing evidence manifest and its SHA-256. Red readiness additionally requires the explicit `--accept-red-readiness` flag; there is no implicit override.

After authorization, `tools/v3_phase2_jobs.py` emits topology-free, digest-pinned collection work and `tools/v3_training_schedule.py` freezes the three-origin bootstrap plus all ten two-origin expert-cycle schedules. Scientific workers and the native training loader independently verify the checksum-chained state and exact phase; the controller is not the only safety boundary.

## Failure recovery

- Resume MLX only with the same run directory and `--resume`; a changed dataset, binary, optimizer, seed, batch, D6 schedule, or origin is refused.
- The training run binds the exact V3 MLX/checkpoint sources and runtime. Any source change requires a new run origin; it cannot silently resume an older optimizer state.
- Resume CPU work through the durable Bacalhau request ID. Never manually reassign hosts or seed parity.
- Import only checksummed artifacts from the canonical image digest.
- Preserve the incumbent when a cycle candidate is rejected or inconclusive, then continue the next cycle.
- Pause before a planned write would exceed 40 GiB or leave less than 50 GiB free.

## Topology

- John1: source, image build, artifact authority, MLX, aggregation.
- John2/John3: fungible Docker workers.
- John4: dashboard only.

CPU work is expressed as independent scheduler items. During native MLX, John1 is reserved and John2/John3 run only one-iteration-behind benchmarks.
