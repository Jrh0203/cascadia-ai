# R3 Exact Action Local-Patch Plus Global-Edit Foundation Preregistration

Status: completed; all production gates passed; matched MLX prototype authorized

Date: 2026-06-17

Experiment ID: `r3-action-edit-foundation-v1`

Contract: ADR 0148

Schema: `r3-exact-action-local-patch-global-edit-v1`

## Question

Can Cascadia complete actions be represented as exact variable-length edits
over one reusable sparse public state trunk, with a small canonical local patch
and explicit global object changes, without reading hidden refill order or
materializing a full dense afterstate per candidate?

This is a representation-foundation experiment. It does not test a learned
model or gameplay strength.

## Frozen Open Corpus

The production census uses deterministic four-player Card A games with habitat
bonuses disabled:

| Split | Raw seed range | Games | Positions |
|---|---:|---:|---:|
| Train | `3,300,000..3,300,016` | 16 | 1,280 |
| Validation | `3,400,000..3,400,004` | 4 | 320 |
| **Total** | | **20** | **1,600** |

Every game is generated from `GameSeed::from_u64(raw_seed)`. No model,
checkpoint, external data, or hidden-state artifact enters corpus identity.

At each position:

1. apply the feasible free three-of-a-kind prelude selected by the
   authoritative engine;
2. encode and hash one parent state trunk;
3. enumerate every legal complete action for that realized public market
   through the engine's place/undo candidate-board context;
4. encode, decode, and apply every action edit;
5. compare the result with the authoritative public afterstate;
6. select one deterministic action by BLAKE3 of `(raw_seed, completed_turns)`
   to advance the trajectory; and
7. continue for all 80 decisions.

The action-count distribution is the complete canonical screen after the
feasible free prelude.

## Paid-Prelude Sentinels

When the active player has Nature Tokens after the free prelude:

- enumerate all 15 nonempty paid-wipe masks;
- for each feasible one-wipe branch, test the first, median, and last legal
  complete actions;
- when at least two Nature Tokens remain, test the lexicographically first
  feasible two-wipe sequence; and
- retain the ordered wipe masks and exact visible staged market.

Paid-prelude sentinels are reported separately from canonical action counts.
They test schema completeness and do not redefine the action-count
distribution.

## Frozen Semantic Boundary

The authoritative target is:

```text
GameState::preview_public_afterstate(action)
  -> PositionRecord::observe_public_for_seat(acting_seat)
  -> terminal targets forced to zero
```

Exact parity requires:

1. normalized public `PositionRecord` byte equality;
2. exact semantic-supply equality;
3. regenerated frontier/component/motif edit equality; and
4. regenerated canonical action-view equality.

The representation may use visible market outcomes after a chosen prelude. It
may not use the hidden order that produced those outcomes.

Forbidden inputs:

- tile-stack order;
- excluded-tile identity;
- wildlife-bag order;
- wildlife return insertion position;
- RNG seed;
- future refill realization;
- future actions or opponent responses; and
- terminal targets or teacher labels.

## Frozen Action Representation

Each decision has one `PublicStateTrunk`.

Each action has:

- complete prelude/draft/placement factors;
- selected tile and wildlife semantics;
- exact world placement edit;
- exact market and semantic-supply edits;
- exact active-player metadata changes;
- exact frontier/component/motif changes;
- one 37-cell canonical radius-3 patch;
- exact global object references;
- immediate score-component deltas; and
- radius-1/2/3 direct-coordinate coverage.

All collections are variable-length. Any skipped object, capped list, clipped
coordinate, or silent overflow is a mechanical failure.

## Radius Coverage Definition

The direct changed-coordinate set is the union of:

- tile destination;
- wildlife destination, when present;
- frontier additions;
- frontier removals;
- frontier context updates; and
- wildlife-motif additions, removals, and updates.

For each radius `r` in `{1, 2, 3}`, report:

- total direct changed coordinates;
- coordinates at hex distance `<= r` from the tile destination;
- covered-coordinate fraction; and
- fraction of actions with complete direct-coordinate coverage.

Habitat components and other long-range objects are deliberately not required
to fit inside the patch. They must remain exact in the global edit. No local
radius can promote as a standalone representation merely because its direct
coverage is high.

## D6 Gate

For one deterministic canonical action at every position, run all 12
authoritative D6 transforms.

Each transform must prove:

1. transformed state/action legality;
2. transformed edit application parity;
3. byte-identical canonical local patch;
4. byte-identical canonical global edit;
5. equal selected market semantics;
6. equal score anatomy; and
7. equal radius coverage.

That is 19,200 transform checks on the frozen 1,600-position corpus.

Component numbers emitted by the sparse substrate are transient traversal
identities. Frontier equality and the D6 gate therefore compare component
references after normalization to terrain plus sorted exact membership.
Treating raw component numbers as semantic identity is a mechanical failure.

## Frozen Measurements

Report nearest-rank median, P90, P99, and maximum for:

- legal complete actions per canonical decision;
- public state-trunk tokens;
- packed state-trunk bytes;
- per-action edit tokens; and
- packed per-action edit bytes.

Also report:

- train and validation position counts;
- state-trunk encodings, which must equal canonical decisions;
- canonical actions verified;
- paid-wipe sentinel actions verified;
- exact apply checks;
- authoritative normalized public-successor checks;
- exact semantic-supply parity checks;
- regenerated global-edit checks;
- codec round trips;
- D6 checks;
- maximum observed wipe-sequence length;
- radius-1/2/3 coverage; and
- deterministic scientific BLAKE3.

Token accounting is structural:

- one token per global/player/market/supply object in the trunk;
- one token per local patch cell;
- one token per edit object;
- one token per wipe mask; and
- one token per sparse supply delta.

It does not estimate neural FLOPs.

## Frozen Promotion Gates

R3 may proceed to a matched local MLX prototype only if:

| Criterion | Threshold |
|---|---|
| Authoritative public successor parity | 100% |
| Supply-delta parity | 100% |
| Regenerated global-edit parity | 100% |
| Canonical codec round trip | 100% |
| D6 canonical action-view parity | 100% |
| Silent truncation or clipping | None |
| State trunk encodings | Exactly one per canonical decision |
| Median action-edit tokens | `<= 128` |
| P99 action-edit tokens | `<= 256` |
| Maximum action-edit tokens | `<= 384` |
| P99 packed action-edit bytes | `<= 8,192` |

Radius coverage is descriptive, not a promotion gate. Radius 1 or 2 may become
an MLX ablation only with the exact global edit retained.

Passing authorizes only a matched MLX representation prototype. It does not
claim:

- better extraction throughput;
- better inference latency;
- better action recall or regret;
- better search; or
- a gameplay score increase.

## Deterministic Scientific Output

Each shard scientific JSON and the merged scientific JSON contain no:

- path;
- hostname;
- timestamp;
- process ID;
- thread count;
- wall-clock timing; or
- output location.

The scientific BLAKE3 is computed over compact JSON serialization of the
scientific payload. Every production shard binds:

- the complete reviewed source bundle;
- the exact census executable;
- its deterministic modulo-owned train and validation seeds; and
- its exact histogram and verification evidence.

The five owned games in each shard run concurrently with
`RAYON_NUM_THREADS=5`. Every game produces a private exact accumulator.
Accumulators are merged in canonical seed order, so thread scheduling cannot
enter scientific output. The four hosts therefore expose up to 20 independent
game workers without duplicating a seed.

The aggregate accepts exactly four shards with indices `0..3`, verifies each
shard against the running executable and source bundle, rejects duplicate or
unknown JSON keys, recomputes all quantiles from merged exact histograms, and
proves complete frozen-corpus coverage. Input paths and input order are not
scientific fields.

Two aggregates are produced from forward and reverse shard order. They must be
byte-identical. A separate content-bound order proof records the aggregate
scientific BLAKE3, file BLAKE3, and file length.

## Distributed Production Execution

Do not launch while another preregistered campaign owns the cluster. Production
uses one immutable source bundle and one copied release executable on all four
hosts.

```bash
cd /Users/johnherrick/cascadia/tools/r3_action_edit_census

# Before bundling, record the reviewed source and executable identity.
target/release/r3-action-edit-census identity \
  --output ../../artifacts/experiments/r3-action-edit-foundation-v1/control/runtime-identity.json

# Run exactly one deterministic shard per host.
RAYON_NUM_THREADS=5 target/release/r3-action-edit-census census \
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

The fixed assignment is:

| Host | Shard |
|---|---:|
| `john1` | 0 |
| `john2` | 1 |
| `john3` | 2 |
| `john4` | 3 |

The reviewed queue is generated by
`tools/r3_action_edit_campaign.py`. It has exactly 13 tasks:

1. fan out and whole-tree checksum the immutable bundle;
2. run one source/executable identity preflight on each host;
3. run one nonoverlapping production shard on each host;
4. checksum-collect the four shard reports;
5. aggregate in forward and reverse shard order; and
6. prove byte-identical aggregate output.

Every identity preflight passes both
`--expected-source-bundle-blake3` and
`--expected-executable-blake3`; no shard depends on fewer than all four
preflights.

After all four reports are collected to `john1`:

```bash
target/release/r3-action-edit-census aggregate \
  --input ../../artifacts/experiments/r3-action-edit-foundation-v1/reports/shard-0.json \
  --input ../../artifacts/experiments/r3-action-edit-foundation-v1/reports/shard-1.json \
  --input ../../artifacts/experiments/r3-action-edit-foundation-v1/reports/shard-2.json \
  --input ../../artifacts/experiments/r3-action-edit-foundation-v1/reports/shard-3.json \
  --output ../../artifacts/experiments/r3-action-edit-foundation-v1/reports/aggregate-forward.json

target/release/r3-action-edit-census aggregate \
  --input ../../artifacts/experiments/r3-action-edit-foundation-v1/reports/shard-3.json \
  --input ../../artifacts/experiments/r3-action-edit-foundation-v1/reports/shard-2.json \
  --input ../../artifacts/experiments/r3-action-edit-foundation-v1/reports/shard-1.json \
  --input ../../artifacts/experiments/r3-action-edit-foundation-v1/reports/shard-0.json \
  --output ../../artifacts/experiments/r3-action-edit-foundation-v1/reports/aggregate-reverse.json

target/release/r3-action-edit-census prove-order \
  --forward ../../artifacts/experiments/r3-action-edit-foundation-v1/reports/aggregate-forward.json \
  --reverse ../../artifacts/experiments/r3-action-edit-foundation-v1/reports/aggregate-reverse.json \
  --output ../../artifacts/experiments/r3-action-edit-foundation-v1/reports/aggregate-order-proof.json
```

## Invalid Pre-Production Smoke

The first john4 smoke of immutable bundle
`7c7dceb0c6c1273ec5986fa84df367f946fe01b9216c0cb37fb6bcdc5bbec68c`
failed the D6 gate before any production launch. At raw seed `4,100,003`, turn
zero, transform `2`, a frontier token appeared changed only because the
underlying habitat component had a different transient numeric ID.

The run is invalid evidence, not a negative scientific result. The correction
normalizes component references by semantic content before frontier equality,
retains raw IDs only for exact world application, and adds
`d6_regression_seed_4100003_turn_zero_is_exact`. Full details and immutable
identities are in
`docs/v2/reports/r3-action-edit-foundation-v1-invalid-smoke-1.md`.

## Focused Pre-Production Verification

These commands are authorized now:

```bash
cd /Users/johnherrick/cascadia/tools/r3_action_edit_census

cargo fmt --all -- --check
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo build --release

cargo run --release -- inspect \
  --seed 137 \
  --turns 0 \
  --action-index 0 \
  --output /tmp/r3-action.json \
  --packed-trunk /tmp/r3-trunk.bin \
  --packed-edit /tmp/r3-edit.bin
```

Focused tests must include:

- target and hidden-order independence;
- one parent-trunk hash reused across a complete legal-action batch;
- paired and independent draft screens;
- paid market preludes;
- exact public successor replay;
- all 12 D6 transforms;
- codec corruption/truncation/trailing-byte rejection;
- adversarial same-local-patch/different-long-range-component fixtures;
- adversarial same-local-patch/different-long-range-motif fixtures; and
- variable-length no-truncation stress.

## Production Completion

All launch blockers closed before execution:

1. S1 released cluster capacity.
2. The reviewed source and executable were frozen in immutable bundle
   `24416fe767223fee6ca9e9cf2748fde45650bccd7526b78601f43c8490459b58`.
3. A one-thread john4 replay and two-thread john1 replay were byte-identical.
4. The 13-task graph was installed atomically and completed without task
   failure.
5. All four nonoverlapping shard reports were checksum-collected.
6. Forward and reverse aggregate files were byte-identical.

The aggregate scientific BLAKE3 is
`9a3075bf4b9abb0ce05efad1856ce951163d04f41e619f83acdf77ee78130424`.
Every frozen promotion gate passed, authorizing only the matched MLX
prototype described in the result report:

`docs/v2/reports/r3-action-edit-foundation-v1-result.md`
