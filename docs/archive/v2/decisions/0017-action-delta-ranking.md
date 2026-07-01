# ADR 0017: Action-Delta Ranking

Status: rejected before gameplay on 2026-06-11.

## Context

The hidden-state-safe terminal ranker learned broad ordering but failed
best-action fidelity. Its independent scorer receives a complete candidate
afterstate without knowing which tile or wildlife was just placed. Candidates
within a decision differ by one small action inside four otherwise identical
boards, so the model must relearn absolute state value to recover a delta.

The terminal datasets preserve every candidate action hash and deterministic
game identity. Exact actions can therefore be reconstructed without rerunning
the R8 teacher: replay the game, regenerate the frozen pattern frontier, match
every serialized action hash, verify the stored observable afterstate, and
apply the teacher-selected tied action with the original deterministic
selection schedule.

## Decision

Add a separate versioned action-ranking format. Each record contains:

- the existing observable `compact-entity-v2` afterstate;
- paired or independent draft identity and market slots;
- drafted tile terrain, wildlife compatibility, and keystone flag;
- drafted wildlife;
- tile coordinate and rotation;
- optional wildlife coordinate;
- free replacement and paid-wipe summary;
- signed immediate deltas for five habitats, five wildlife cards, and Nature
  Tokens.

MLX appends changed-tile and changed-wildlife markers to the acting board
entities and projects the explicit action vector alongside board, market, and
global context. The architecture remains compact: hidden 96, four attention
heads, two board blocks, one market block, and feed-forward multiplier three.

The frozen binary contract is:

- shard magic `CSD2ARK\0`, schema version 1, and a 112-byte header;
- feature schema `compact-action-delta-v1`;
- target schema `search-action-ranking-v1`;
- 52 raw action bytes, including eight reserved zero bytes;
- 916-byte inference inputs: one 864-byte `compact-entity-v2` position plus
  one 52-byte action;
- 972-byte grouped records: 56 bytes of teacher/group metadata plus the
  916-byte inference input.

The Python decoder produces 33-dimensional board entities by appending the
changed-tile and changed-wildlife markers only to the acting board. The
normalized action projection has 63 dimensions. It contains draft kind and
slots, tile and wildlife identity, coordinates, rotation, placement presence,
market-prelude costs, eleven signed immediate score deltas, immediate rank,
and immediate score.

Enrichment is deterministic and fail-closed. It validates the source dataset,
reconstructs the exact frozen K8+H6+B8 pattern frontier, matches every action
by canonical JSON BLAKE3, verifies immediate rank and score, checks byte
identity of the observable pre-refill afterstate, and replays the recorded
teacher winner with the original tie-selection RNG. The action manifest embeds
the absolute source path, dataset ID, source manifest checksum, schemas,
record size, game range, and candidate totals.

The frozen model is `action-delta-ranker-v1`: hidden 96, four heads, two board
attention blocks, one market attention block, feed-forward multiplier three,
and an uncertainty-weighted grouped listwise loss. Training uses AdamW,
learning rate `1e-4`, weight decay `1e-4`, 16 complete decision groups per
batch, seed 20260611, at most 20 epochs, and validation patience five.

## Evaluation

Existing train and validation labels are enriched deterministically. Sixteen
new R8 games from the untouched test namespace are collected and enriched
after the architecture is frozen. Validation selects checkpoints; test alone
controls advancement.

The untouched test must achieve regret at most 0.75, pairwise accuracy at
least 0.65, value-difference correlation at least 0.30, and tie-aware top-one
recall at least 0.45. Only then may gameplay use seeds 25700 through 25709,
followed conditionally by seeds 25800 through 25849.

Promotion additionally requires the selected validation loss to improve
strictly over the initialization checkpoint. `test-report.json` must name the
exact selected checkpoint and pass all four gates. The promotion command
refuses any other run kind, failed or stale test report, non-improving best
checkpoint, checksum mismatch, or existing output directory.

## Implementation Evidence

`make action-ranking-smoke` completed the complete path with a tiny disposable
teacher configuration: terminal collection, exact strategy metadata,
deterministic enrichment, Rust and Python validation, MLX training, untouched
test evaluation, and Rust-to-MLX gameplay inference. Its one-epoch model kept
the initialization checkpoint and failed the deliberately underpowered test
gates. That is expected smoke behavior and is not registered strength
evidence. No substantive test record was read during implementation.

## Result

The frozen run stopped after 12 epochs and 3,840 optimizer steps. Epoch 7
improved validation selection loss from 2.664025 to 2.559858. On the untouched
16-game test split it achieved:

- mean top-one regret 0.967773, failing the 0.75 gate;
- pairwise accuracy 0.670384, passing the 0.65 gate;
- value-difference correlation 0.495073, passing the 0.30 gate;
- tie-aware top-one recall 0.272656, failing the 0.45 gate.

The promoter refused the failed test report. No model artifact or gameplay
result was produced. Explicit action identity improved broad learning from
initialization but did not resolve top-choice ambiguity enough to justify a
policy trial.
