# V1 Audit

This audit records facts about the live implementation. Historical claims are
not accepted as evidence of strength.

## Structure

| Component | Current role | Main concern |
|---|---|---|
| `legacy/crates/cascadia-core` | Board, market, game state, scoring | Rules and mutable transition semantics are coupled |
| `legacy/crates/cascadia-ai` | Heuristics, NNUE, training, MCE, MCTS, AlphaZero, experiments | Production and abandoned experiments share one crate |
| `legacy/crates/cascadia-cli` | Play, benchmark, collection, training, daemon, reports | Approximately 6,800 lines and manually parsed flags |
| `legacy/crates/cascadia-web` | Axum API and embedded frontend | Backend and a monolithic HTML application are coupled |

The current source surface is approximately 44,000 Rust lines. The largest
modules are `main.rs`, `mce.rs`, `nnue.rs`, `nnue_train.rs`, and
`alphazero_v2.rs`.

## Rules Engine

### Strengths

- Packed board cells and a fixed hex grid are efficient.
- Habitat connectivity uses union-find.
- Board placement supports apply/undo for search.
- Official tile and wildlife bags are represented.
- Wildlife cards A-D have broad unit-test coverage.
- The current suite contains 120 core tests and 61 default AI tests.

### Risks And Required V2 Fixes

- `GameState::execute_move` drafts from the market before validating tile
  placement. A rejected placement can leave state mutated.
- `execute_independent_move` spends a token and drafts before placement
  validation, with the same transactional risk.
- Legal action generation and action application are separate informal
  contracts. Search normally supplies legal moves, but API callers can violate
  the assumption.
- `PlayerMove` does not encode pre-move actions, independent drafts, or a
  complete turn as one canonical action type.
- RNG ownership is implicit. Game setup, search randomness, and simulated
  futures are not represented as explicit independent streams.
- Stable public state/replay serialization is absent.
- Zobrist hashing exists, but v2 needs a documented, versioned state identity
  and replay contract.

V2 must make actions transactional: validation is pure, and applying a valid
action cannot partially fail.

## AI And Configuration

- The AI crate exports roughly 30 modules spanning promoted and failed ideas.
- Cargo features select mutually incompatible feature maps and network shapes.
- AI and CLI code contain 105 runtime environment-variable reads.
- Strategy tags mutate process-wide environment variables around calls.
- Neural architecture, feature extraction, training, and binary compatibility
  logic are concentrated in very large modules.
- Old weights may be accepted through truncation or zero-padding. A file can
  load successfully while being semantically mismatched with the binary.

V2 will use typed, serializable configuration passed explicitly through APIs.
Experiments may have parameters, but production behavior cannot depend on
ambient process state.

## Neural And Dataset Formats

V1 contains many independent formats and magic values, including NNUE versions,
`MCEP`, `MCV2-4`, `MCP2`, `CZR1`, `PLCY`, `CZP1`, `AZD1-2`, `AZR1-3`, `HYBR`,
and `HYBP`. Metadata and compatibility rules are distributed across source
files. Generated files use names as provenance.

V2 needs:

- one artifact manifest schema,
- one sharded sample container contract,
- explicit task/model schema IDs,
- checksums,
- parent artifact IDs,
- command/config capture,
- train/validation/test split identity,
- atomic checkpoints and resumption state.

## Benchmark Semantics

The current benchmark paths are not interchangeable:

- The default CLI benchmark scores player 0 while opponents often use a
  different, weaker policy.
- Symmetric seat tags always take free overflow replacement but do not run the
  same paid-mulligan pre-move optimization as the normal strategy path.
- `CASCADIA_OPPONENTS_SAME=1` is closer to symmetric play, but results are
  emitted partly through diagnostic stderr and only player 0 enters the normal
  summary.
- `mce_perf_bench` runs the same picker on all seats and reports all seats, but
  omits champion pre-move optimization.
- Base and with-bonus scoring can be selected through different flags and
  environment variables.

Therefore no existing command is accepted as the v2 canonical strength metric.
The v1 baseline will be reproduced only after the new benchmark runner applies
the same complete turn policy to every seat.

## Web Product

The web tool provides useful interaction ideas: playable boards, suggestions,
strength controls, undo, mulligans, overflow replacement, and score displays.
However, the backend is about 1,100 lines and embeds a large frontend through
`include_str!`. It has no maintainable frontend build, shared generated API
types, browser test suite, or persistent replay contract.

V2 will preserve the product behavior and visual spirit while replacing the
implementation with a typed API and a standalone responsive frontend.

## Repository And Developer Experience

- No root README existed before v2 work began.
- Generated data, models, logs, reports, scripts, and source share the root.
- The repository occupies approximately 110 GB.
- Multiple build output directories exist for feature combinations.
- Compiler output contains a substantial unused-code/import warning backlog.
- There is no single setup, check, benchmark, train, resume, or promote command.

## V1 Reference Policy

V1 remains available for:

- independently verified rule fixtures,
- differential tests,
- UX behavior inventory,
- baseline strategy reproduction.

V2 code must not depend on v1 crates. Historical models cannot be promoted into
v2; they may only serve as explicitly labeled baseline opponents.

The source, historical scripts, and historical reports now live under
`legacy/`. Only `cascadia-differential` may depend on v1, and only for tests or
explicitly feature-gated baseline reproduction.
