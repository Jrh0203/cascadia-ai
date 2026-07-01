# R3 Action Edit Census

Standalone Rust implementation of the
`r3-exact-action-local-patch-global-edit-v1` foundation.

Experiment ID: `r3-action-edit-foundation-v1`

R3 encodes one public state trunk per decision and one variable-length edit per
complete legal action. The edit is an exact program over the public observation
boundary: applying it to the trunk reproduces
`GameState::preview_public_afterstate` after normalization through
`PositionRecord::observe_public_for_seat`.

Production corpus work has not been launched. The immutable four-host campaign,
five-game parallel shard executor, and deterministic aggregate path are
implemented and tested. A pre-production smoke found and repaired one false D6
failure caused by comparing transient habitat-component numbers as semantics;
the invalid run and exact witness are recorded in
`docs/v2/reports/r3-action-edit-foundation-v1-invalid-smoke-1.md`.

## R2 Authority Adapter

`r2_public_adapter` compiles ADR 0145's authoritative `model.rs` and `codec.rs`
directly from `tools/r2_sparse_entity_census`. It does not copy or fork those
semantics, and it deliberately excludes unrelated R2 census and MLX-export
modules. This keeps the standalone R3 foundation reviewable while concurrent
R2 model work proceeds independently.

## Extraction Path

`PublicStateTrunk::prepare_action_edits` creates the reusable decision handle;
`PreparedPublicStateTrunk::observe_legal_actions` is the primary batch API:

1. validate and hash the parent `PublicStateTrunk` once;
2. realize the selected market prelude once;
3. enumerate the complete legal screen with
   `GameState::evaluate_legal_turn_actions_with_context`;
4. derive board, frontier, component, motif, and score edits while each
   authoritative place/undo candidate board is borrowed; and
5. attach exact visible market, supply, player, and turn deltas without
   cloning or applying a whole `GameState` per action.

The prepared handle retains the one packed trunk and its canonical hash across
the canonical screen and every paid-prelude sentinel. Every action carries that
same hash. Generated tests compare applied edits with
`preview_public_afterstate`, so the optimized extraction path is checked
against the independent public successor oracle.

Frontier equality canonicalizes habitat-component references by terrain and
sorted membership before deciding whether context changed. Raw numeric
component IDs are retained only in exact world edits; they are never treated as
stable D6 semantics.

## Exact Boundary

The state trunk contains:

- the accepted ADR 0145 R2 sparse public state;
- occupied, frontier, habitat-component, and wildlife-motif tokens;
- public global, player, and market metadata; and
- ADR 0143 exact semantic supply counts.

The per-action edit contains:

- ordered free and paid market-prelude factors;
- exact visible market changes caused by the prelude;
- the paired or independent selected market objects;
- exact tile destination, canonical rotation, and six directed edges;
- optional wildlife destination;
- active-board additions and wildlife updates;
- active-player metadata and Nature Token deltas;
- staged-to-afterstate market changes;
- exact public supply deltas;
- frontier additions, removals, and updates;
- habitat-component additions, removals, and updates;
- wildlife-motif additions, removals, and updates;
- immediate public score anatomy;
- global object references; and
- radius-1, radius-2, and radius-3 direct-coordinate coverage.

The authoritative successor stops before hidden market refill. The schema never
contains:

- tile-stack order;
- excluded-tile identity;
- wildlife-bag order;
- RNG seed or return position;
- future refill realization;
- future action or opponent choice;
- terminal score targets; or
- learned labels.

Physical tile IDs are not represented. ADR 0143 and ADR 0145 establish that
they are serialization identities, not distinct public rule semantics. Exact
successor parity therefore means byte equality of the normalized public
`PositionRecord`, plus exact public semantic-supply equality.

## Canonical Action Frame

The complete action retains world coordinates for exact application. Its MLX
view is canonicalized separately:

1. translate the proposed tile destination to axial origin;
2. consider every authoritative D6 transform that maps the selected tile to
   rotation zero;
3. transform the full radius-3 patch and every spatial edit object;
4. replace habitat-component IDs with transformed membership identities; and
5. select the lexicographically smallest canonical serialized view.

The result is invariant across all 12 rotations/reflections while the reusable
state trunk remains a single equivariant encoding. The trunk is never copied
into each action record.

## No-Truncation Contract

All action-dependent collections use variable-length canonical vectors.
There is no fixed maximum for:

- wildlife-wipe sequence length;
- touched components;
- changed frontier objects;
- changed motif objects;
- global references; or
- legal action count.

Radius 1/2/3 are measured local views, not clipping boundaries. Any consequence
outside a local radius remains present through the exact global edit.

## Commands

```bash
cd /Users/johnherrick/cascadia/tools/r3_action_edit_census

# Inspect one generated complete action.
cargo run --release -- inspect \
  --seed 137 \
  --turns 0 \
  --action-index 0 \
  --output /tmp/r3-action.json \
  --packed-trunk /tmp/r3-trunk.bin \
  --packed-edit /tmp/r3-edit.bin

# Focused verification, including the R2 authority adapter.
cargo fmt --all -- --check
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo build --release

# Record the reviewed source-bundle and executable identity.
target/release/r3-action-edit-census identity \
  --output ../../artifacts/experiments/r3-action-edit-foundation-v1/control/runtime-identity.json
```

Every host preflight supplies the expected identities and fails before corpus
work if either differs:

```bash
target/release/r3-action-edit-census identity \
  --expected-source-bundle-blake3 EXPECTED_SOURCE_BLAKE3 \
  --expected-executable-blake3 EXPECTED_EXECUTABLE_BLAKE3 \
  --output ../../artifacts/experiments/r3-action-edit-foundation-v1/reports/preflight-john1.json
```

The preregistered production command is intentionally not run while another
campaign owns the cluster. Each host runs one explicit shard with five Rayon
workers, one for each owned game:

```bash
RAYON_NUM_THREADS=5 \
target/release/r3-action-edit-census census \
  --train-first-seed 3300000 \
  --train-games 16 \
  --validation-first-seed 3400000 \
  --validation-games 4 \
  --paid-wipe-sentinels true \
  --d6-sentinel-per-position true \
  --shard-index HOST_INDEX \
  --shard-count 4 \
  --output ../../artifacts/experiments/r3-action-edit-foundation-v1/reports/shard-HOST_INDEX.json
```

The generated campaign graph is the production authority:

```bash
.venv/bin/python -B tools/r3_action_edit_campaign.py queue-spec \
  --repository . \
  --bundle artifacts/experiments/r3-action-edit-foundation-v1/bundles/BUNDLE_ID \
  --output artifacts/experiments/r3-action-edit-foundation-v1/queue-spec.json

python3 tools/cluster_research_queue.py install-spec \
  --spec artifacts/experiments/r3-action-edit-foundation-v1/queue-spec.json
```

It contains exactly 13 tasks: immutable fanout, four identity preflights, four
nonoverlapping shards, checksum collection, forward and reverse aggregation,
and one terminal byte-order proof.

The census enumerates every canonical complete action at every position, adds
paid-wipe sentinels, independently compares every decoded replay with
`preview_public_afterstate` and exact semantic supply, proves codec round
trips, performs one all-D6 sentinel per decision, and reports deterministic
token, byte, action count, radius coverage, and state-trunk-encoding
measurements.

After collecting shards `0..3`, run `aggregate` once in forward order and once
in reverse order, then run `prove-order`. The aggregate rejects missing,
duplicated, tampered, source-mismatched, or non-production shards; merges exact
histogram bins before recomputing quantiles; and emits path- and
order-independent scientific JSON.

## MLX Consumer Contract

The eventual MLX path must:

1. encode `PublicStateTrunk` exactly once per decision;
2. batch `CanonicalActionView` rows for all legal actions;
3. resolve global references against cached trunk objects;
4. preserve variable action counts with masks or packed offsets; and
5. retain the exact Rust codec as the semantic oracle.

Any fixed-width truncation, dense afterstate materialization per candidate, or
hidden-refill feature is a separately named treatment and cannot claim R3
schema compatibility.
