# Cascadia V2 Architecture

## Principles

1. Rules are deterministic and independent of AI, networking, UI, and MLX.
2. Every complete turn is represented by one canonical typed action.
3. Configuration is explicit data, never ambient environment state.
4. Neural ownership belongs to the Python MLX package.
5. Benchmarks execute the same policy path used by the product.
6. Experiments are registered artifacts, not permanent production branches.
7. V2 does not import v1 code.

## Target Layout

```text
crates/
  cascadia-game/       deterministic rules, scoring, actions, replay
  cascadia-sim/        seeded setup, matches, strategies, parallel simulation
  cascadia-eval/       benchmark protocol, paired statistics, reports
  cascadia-data/       compact records, seed splits, collection, integrity
  cascadia-model/      typed local MLX process protocol
  cascadia-provenance/ shared source, executable, and dirty-tree identity
  cascadia-search/     learned action ranking and future search
  cascadia-api/        shared API DTOs and server application services
  cascadia-cli-v2/     typed user and research commands
apps/
  web/                 TypeScript frontend
python/
  cascadia_mlx/        MLX models, training, inference service, artifact tools
artifacts/             ignored local datasets, runs, models, reports
experiments/           registered experiment specifications and summaries
docs/v2/               architecture, ADRs, runbooks, current status
legacy/                isolated v1 crates, research scripts, and reports
```

The v1 crates now live under `legacy/`. The test-only differential crate is the
sole dependency boundary into them; production v2 crates remain independent.

## `cascadia-game`

Owns:

- hex coordinates and topology,
- tiles, wildlife, cards, bags, market, players, and game state,
- canonical `TurnAction`,
- pure `legal_actions(state)` and `validate(state, action)`,
- transactional `apply(state, action) -> Transition`,
- exact scoring,
- stable state snapshots and replay events,
- deterministic state hash.

It does not own RNG, strategy selection, model evaluation, clocks, or I/O.

`TurnAction` includes all decisions required for a turn:

- optional free overflow replacement,
- zero or more paid wildlife mulligans,
- paired or independent market draft,
- tile coordinate and rotation,
- optional wildlife placement.

Application validates the whole action before mutation. Undo uses a complete
transition delta and must round-trip state and hash exactly.

## `cascadia-sim`

Owns:

- seeded game construction,
- independent RNG streams for setup, each strategy, and stochastic search,
- `Strategy` trait,
- random, legal-greedy, and pattern-aware non-neural baselines,
- complete symmetric match execution,
- parallel game scheduling,
- replay capture.

Strategies receive only observable state. Hidden bag order is unavailable
unless a deliberately labeled oracle experiment requests it.

## `cascadia-eval`

Owns the benchmark contract:

- deterministic seed suites,
- common-random-number paired comparisons,
- all-seat score capture,
- base and bonus metrics as distinct fields,
- category breakdowns,
- mean, SD, SE, confidence intervals, quantiles, and paired deltas,
- machine-readable JSON plus human-readable Markdown reports,
- hardware, commit, config, model, and artifact provenance.

## Neural Boundary

All trainable neural code lives in `python/cascadia_mlx`.

MLX owns:

- encoders,
- models,
- losses,
- optimizers,
- training and resumption,
- checkpointing,
- batch inference,
- calibration and validation metrics.

Rust owns exact rules, simulation, legal actions, and benchmark accounting.

`cascadia-data` is the explicit handoff. Rust writes fixed-width compact entity
records directly from canonical states. Python verifies the manifest and every
shard checksum, memory-maps records, and vectorizes them into MLX tensors.

A long-lived local model service will expose versioned batched inference over a
fixed-header local protocol. Requests carry batches of the same compact records
used for training; responses carry eleven little-endian float32 score
components. Shared-memory transport is allowed only after profiling shows
protocol overhead is material.

Model checkpoints use MLX-compatible safetensors plus a JSON manifest. Rust
does not parse neural weights.

## Artifact System

Every dataset, run, checkpoint, and benchmark has:

- a content-derived artifact ID,
- schema version,
- creation time,
- git commit and dirty-tree digest,
- exact typed configuration,
- parent artifact IDs,
- seed split ID,
- file checksums,
- hardware/toolchain summary,
- status: incomplete, complete, rejected, or promoted.

Writers use temporary paths and atomic rename. Interrupted runs remain
detectable and resumable. Rust datasets and benchmark reports share
`cascadia-provenance`; continuation is rejected when the source or executable
identity differs from the manifest that owns earlier shards.

## Local Experiment Orchestration

Cluster experiment orchestration is intentionally outside production Rust and
the frozen MLX model package. The ADR 0078/0079 handoff is split by ownership:

- `tools/adr0078_cluster_runtime.py`: immutable paths, identities, SSH/process
  execution, state, checksums, locks, and manifest contracts;
- `tools/adr0078_cluster_transport.py`: identity-pinned SSH and rsync endpoint
  construction, with Tailscale primary routes and a verified john2 LAN
  fallback;
- `tools/adr0078_artifact_handoff.py`: producer ownership checks,
  byte-identical strict-prefix recovery, archival evidence, and atomic dataset
  installation;
- `tools/adr0078_collection.py`: train/validation monitoring, producer
  validation, aggregation, and john3 provisioning;
- `tools/adr0078_training.py`: the single resumable MLX run, validation
  evaluation, and artifact retrieval;
- `tools/adr0079_cluster_handoff.py`: post-pass authorization, sealed-test
  collection, transfer, evaluation, and validation replay;
- `tools/adr0078_cluster_supervisor.py`: a small one-shot launchd entrypoint.

No orchestration module owns game, model, or statistical semantics. Those stay
in versioned binaries, manifests, ADRs, and the frozen `cascadia_mlx` source.

## Source Ownership

The v2 CLI entrypoint contains only typed parsing, top-level dispatch, and four
small direct commands. Dataset collection, model operations, policy workflows,
lookahead, oracle probes, counterfactual analysis, and reporting live in
separate command modules.

Search similarly separates public lookahead, MLX value evaluation, ranking
rollouts, prefiltering, prediction validation, policy improvement, and oracle
experiments. Pattern strategy execution and finite-market opportunity
arithmetic are separate simulation modules. Public APIs, command names,
serialized schemas, strategy IDs, and deterministic behavior remain stable.

The source-structure test caps the CLI entrypoint at 300 lines and every active
v2 Rust production module at 1,500 lines.

The complete typed command reference is generated from Clap into
[`CLI_REFERENCE.md`](CLI_REFERENCE.md). `make cli-docs-check` fails when the
checked-in reference drifts from the executable schema.

## Web Product

The web backend uses stateless `cascadia-api` game services and never
duplicates rules. The frontend is a standalone React/TypeScript application.
It supports local persistence through versioned replay documents. Cluster
operations are deliberately stateful: john1 retains a bounded seven-day JSONL
telemetry journal and serves downsampled 1D/7D CPU and memory series without
putting monitoring state on john2 or john3.

Each request reconstructs and validates a canonical `GameState` from the
replay. Market preparation and legal-placement queries use explicit engine
preview methods; only a complete `TurnAction` advances the replay. The browser
never receives future bag order.

The promoted product modes are:

- instant: exact immediate-score greedy with a strict latency cap,
- interactive: bounded public-information pattern-aware search and the
  strongest currently promoted product policy.

The final-five R8 c90 terminal operator, MLX policies, and unrestricted search
remain explicit research surfaces until they clear their own held-out
strength, allocation, reproducibility, and latency gates.

## Dependency Direction

```text
cascadia-game
    ↑
cascadia-sim ← cascadia-eval
    ↑               ↑
cascadia-data   cascadia-cli-v2
    ↑
cascadia-model ← cascadia-search
    ↑                 ↑
python/cascadia_mlx
    ↑
apps/web

python/cascadia_mlx communicates through schemas/protocols, not Rust imports.
```

No lower layer depends on UI, CLI, experiments, or MLX.

## Testing Strategy

- Unit tests for local invariants.
- Property tests for apply/undo, serialization, hashing, topology, and scoring.
- Fixture tests from independently verified rule examples.
- Differential tests against v1 only for fixtures already judged correct.
- Replay determinism tests across thread counts.
- Protocol and artifact round trips.
- Browser tests against the real API.
- Performance budgets tracked separately from correctness tests.
