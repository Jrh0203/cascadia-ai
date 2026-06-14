# Dataset Format

## Contract

`cascadia-data` writes the only production training format. The format is
little-endian, fixed-width, versioned, and directly derived from canonical v2
`GameState` values.

Each dataset directory contains:

```text
dataset.json
shard-00000.csd
shard-00001.csd
...
```

The manifest records the schema IDs, split, strategy, game-index range, totals,
and every shard's byte length and BLAKE3 checksum. Writers flush and fsync a
temporary shard before atomic rename. The manifest is updated only after the
new shard validates.

Collection resumes only when the existing schema, split, strategy, and first
game index exactly match. Existing shard sizes, headers, and checksums are
revalidated before new work begins.

Collection provenance includes the Git revision, dirty-tree digest, complete
v2 source digest, executable checksum, and local hardware summary. Resuming a
partial dataset also requires the original source and executable identities;
source drift starts a new dataset instead of mixing record semantics across
shards.

Counterfactual-advantage teachers additionally bind
`stabilization_conditioning`. ADR 0078 requires
`reject-unstable-market-trajectories-v1`: attempt zero uses the recorded
sample seed exactly, and only a complete trajectory ending in
`WildlifeBagEmpty` is retried from the same public state with a
domain-separated hidden seed. The strict MLX decoder rejects a substantive
R12 manifest that omits or changes this contract.

## Seed Partitions

The seed is a BLAKE3 domain-separated hash of `(split, game_index)`.

- `train`: model fitting and self-play generation
- `validation`: architecture and hyperparameter decisions
- `test`: infrequent pre-promotion confirmation
- `final`: the locked 1,000-game strength claim

The same numeric index in different splits produces unrelated game seeds.
Training commands reject any dataset not marked `train`.

## Shards

Each shard has an 80-byte header followed by 864-byte records. The header
contains:

- magic `CSD2REC\0`;
- schema, header, record, and target dimensions;
- record count, game count, and first game index;
- split, strategy, player count, scoring-card, and habitat-bonus metadata;
- the BLAKE3 hash of `compact-entity-v2`.

Each record stores one public position from the acting player's perspective.
Value datasets use complete pre-action positions. Ranking datasets use
observable post-action positions after the deterministic placement and market
depletion, but before the chance refill:

- game index, turn, active seat, and game length;
- relative board tile counts and Nature Tokens;
- per-board wildlife counts and largest habitats;
- up to 23 compact tile entities for each of four relative seats;
- four market tile/token entities, including empty or independently depleted
  partial slots when a ranking afterstate is awaiting refill;
- eleven final base-score targets: five habitats, five wildlife cards, and
  unused Nature Tokens.

Board entities store coordinates, terrain, rotation, wildlife compatibility,
placed wildlife, and keystone state. Hidden stack and bag order are never
encoded. Candidate records never encode the actual post-draft refill: that
draw is unknown when the action is selected and belongs inside a chance model
or teacher expectation, not the model input.

## MLX Decoding

`cascadia_mlx.dataset.Dataset` verifies integrity, memory-maps one shard at a
time, and vectorizes records into:

- board entities: `(batch, 4, 23, 31)`;
- board mask: `(batch, 4, 23)`;
- market entities: `(batch, 4, 31)`;
- market mask: `(batch, 4)`;
- global features: `(batch, 96)`;
- decomposed targets: `(batch, 11)`.

The loader does not materialize a complete dataset in RAM.

## Canonical Action-Imitation Shards

`canonical-action-imitation-v1` uses `.cim` shards because one decision has a
shared state and a variable candidate count. Its 112-byte shard header records
an 880-byte group-header width, a 68-byte candidate width, candidate and group
totals, split, game range, and feature/target schema hashes.

Each decision group stores:

- group ID, candidate count, and exact selected index;
- one 864-byte observable pre-action `compact-entity-v2` state;
- one row per candidate containing immediate rank and score, a 32-byte action
  hash, and 32 bytes of `compact-state-action-v1` identity features.

The action row encodes draft kind and slots, drafted tile and wildlife,
placement coordinates, rotation, optional wildlife placement, free overflow
replacement, paid-wipe summary, immediate rank, and immediate score. Training
groups contain up to 64 structured legal candidates. Gameplay inference is
not capped at 64 and scores the complete canonical legal set.

The grouped layout stores the state once rather than once per candidate. A
full 80-decision smoke game with 5,120 candidates occupies 418,672 bytes.
Rust and Python independently verify group totals, one exact positive,
nonzero ranks, action metadata parity, unique hashes, headers, checksums, and
teacher provenance.

## Full-Frontier MCE Evidence

`canonical-action-mce-evidence-v1` pairs a fresh grouped `.cim` source with a
compact `.imv` evidence sidecar. It exists because winner-only imitation
discarded most of the qualified teacher's rollout distribution, while
duplicating the 864-byte shared state for every target would waste space.

Each `.imv` shard has a 112-byte header and fixed 56-byte rows:

- group ID, candidate index, and candidate count;
- the exact 32-byte canonical action hash from the source `.cim`;
- rollout mean and rollout standard deviation as float32;
- rollout sample count;
- source flags for teacher frontier, pattern frontier, immediate top set, and
  deterministic negative sampling;
- the exact teacher-selected bit.

A zero sample count means the retained legal action was not evaluated by the
K32 teacher. Its mean and standard deviation must both be zero. Every scored
row is retained from the teacher frontier, and the selected row must be a
scored maximum.

The paired collector writes one game per source and target shard. It retains
the selected action, the complete K32 teacher frontier, the complete V2
pattern frontier, immediate top 16, and deterministic BLAKE3 negatives up to
96 actions. If mandatory frontiers exceed that bound, collection fails rather
than silently dropping evidence.

The target manifest records both the total teacher estimate count and aligned
estimate count. Production collection requires them to be identical. Rust and
Python independently verify source path and dataset identity, split, teacher,
game range, shard count, group count, candidate count, candidate indices,
action hashes, selected bits, row widths, checksums, and complete shard
alignment.

Source identity deliberately excludes the source manifest's mutable checksum
so source and target can stream together. It includes the absolute path,
dataset ID, schemas, first game index, and requested game count; every
completed target shard is still matched exactly to its corresponding source
shard. Resume permits the source to be one game ahead after a crash only when
replaying that game reproduces the stored source bytes exactly.

```bash
make imitation-evidence-parity
make collect-imitation-evidence
```

The first command proves the instrumented per-candidate path selects exactly
the same canonical actions as the historical teacher before substantive
collection. The second writes fresh disjoint train and validation domains and
validates both paired datasets after every durable append.

## Commands

```bash
cargo run --release -p cascadia-cli-v2 -- collect \
  --output artifacts/datasets/greedy-train \
  --games 1000 --split train --strategy greedy --shard-games 64

cargo run --release -p cascadia-cli-v2 -- validate-dataset \
  --dataset artifacts/datasets/greedy-train
```

Confirmed search policies use the same value-record schema through the
reusable manifest writer. The H6 collector records every pre-action public
state and attaches the acting seat's final decomposed score after the complete
symmetric game:

```bash
target/release/cascadia-v2 collect-search \
  --output artifacts/datasets/h6-value-train \
  --games 256 --split train --shard-games 8 --resume \
  --candidates 8 --habitat-candidates 6 \
  --determinizations 4 --greedy-plies 4
```

## Score-To-Go Dataset

Signed score-to-go experiments use a dedicated `.stg` format instead of
changing the semantics of existing value records. Each shard has magic
`CSD2STG\0`, a 128-byte header, and fixed 908-byte records:

- one complete 864-byte `compact-entity-v2` public position;
- eleven unsigned 16-bit exact current score components;
- eleven signed 16-bit `final - current` components.

The position's existing target field remains the exact final component vector.
Rust and Python readers require `current + residual = final` for every
component. This preserves negative Nature Token residuals after token spending
without weakening auditability.

The manifest freezes target schema `signed-score-to-go-components-v1`, the H6
teacher identity and configuration, split, game range, source digest,
executable checksum, and every shard checksum. Collection batches games across
CPU cores independently of shard size, so the frozen experiment retains
one-game atomic resume boundaries without serializing generation.

```bash
make collect-score-to-go

target/release/cascadia-v2 validate-score-to-go-dataset \
  --dataset artifacts/datasets/score-to-go-h6-train
```

## Search-Ranking Dataset

Search distillation uses a separate grouped format rather than overloading the
final-score records.

Each `.csr` shard has a 112-byte header and fixed 920-byte candidate records.
Every record contains:

- a deterministic decision-group ID;
- candidate index and candidate count;
- exact immediate-score rank and score;
- teacher rollout mean and standard deviation;
- a BLAKE3 action hash; and
- the complete canonical 864-byte observable afterstate record.

The afterstate contains the acting board after placement, unchanged opponent
boards, the next turn index, and the publicly known market with drafted
components removed. A paired draft leaves one empty slot. An independent
draft can leave a tile-only slot and a wildlife-only slot. The later random
refill is deliberately absent.

Groups are never split across training batches. The manifest freezes the
teacher strategy, candidate family (`bear`, `habitat`, or `pattern`),
specialized-candidate widths, and search configuration. It records group and
candidate totals and applies the same atomic-write, checksum, split, and
exact-resume rules as the value dataset. Manifests written before candidate
families were introduced remain valid and deserialize as Bear teachers.

Terminal teachers set `greedy_plies` to zero and record an explicit
`terminal_continuation_strategy_id`. Fixed-ply teachers require a positive
horizon and omit that field. Resume rejects either kind of horizon drift. The
qualified R8 collector uses the complete K8+H6+B8 `pattern` frontier, shared
public-information redeterminizations, and terminal acting-seat base score.

Policy-iteration datasets additionally contain a `trajectory` object with the
apprentice strategy ID, canonical `model.json` path, and BLAKE3 checksum. The
teacher still defines every target; the trajectory policy only defines which
public states are visited. Exact resume rejects apprentice model drift.

```bash
target/release/cascadia-v2 collect-ranking \
  --teacher habitat \
  --output artifacts/datasets/ranking-h6-train \
  --games 128 --split train --shard-games 8 \
  --candidates 8 --habitat-candidates 6 \
  --determinizations 4 --greedy-plies 4

target/release/cascadia-v2 validate-ranking-dataset \
  --dataset artifacts/datasets/ranking-h6-train
```

```bash
target/release/cascadia-v2 collect-ranking-iteration \
  --model-dir artifacts/models/entity-ranker-v1-h6 \
  --output artifacts/datasets/ranking-h6-iteration1-train \
  --games 64 --first-game-index 128 --split train --shard-games 8 \
  --candidates 8 --habitat-candidates 6 \
  --determinizations 4 --greedy-plies 4
```

```bash
target/release/cascadia-v2 collect-terminal-ranking \
  --output artifacts/datasets/ranking-terminal-r8-observable-train \
  --games 64 --first-game-index 64 --split train --shard-games 1 --resume \
  --determinizations 8 \
  --policy-candidates 8 --policy-habitat-candidates 6 \
  --policy-bear-candidates 8 --policy-market-draws 4
```

One-game shards make the expensive local R8 collection durably resumable
without mixing source, executable, teacher, or continuation identities.

## Action-Ranking Dataset

The action-delta experiment enriches an existing search-ranking dataset rather
than rerunning its teacher. Each `.car` shard uses magic `CSD2ARK\0`, schema
version 1, a 112-byte header, and fixed 972-byte records:

- 56 bytes of group ID, candidate metadata, teacher moments, and action hash;
- one 864-byte `compact-entity-v2` observable pre-refill afterstate;
- 52 bytes of `compact-action-delta-v1` raw action features.

The 52 action bytes store paired or independent draft identity and slots,
drafted tile terrain/compatibility/keystone fields, drafted wildlife, tile
coordinate and rotation, optional wildlife coordinate, free replacement,
paid-wipe count/union mask/total slots, immediate rank and score, eleven
signed score-component deltas, and eight reserved zero bytes. Inference omits
the 56-byte teacher prefix and sends a 916-byte position-plus-action record.

The target schema is `search-action-ranking-v1`. The manifest carries the
complete teacher and trajectory contracts plus a source block containing the
absolute source path, dataset ID, manifest BLAKE3, schemas, record size, game
range, group count, and candidate count. Resume revalidates both datasets and
rejects source, schema, provenance, game-range, or teacher drift.

Enrichment reconstructs each game and frozen candidate frontier. It must
match every source action hash, immediate rank, immediate score, and complete
observable afterstate byte for byte before writing a candidate. It also
replays the recorded winner with the original deterministic tie RNG, so the
next state cannot drift from the source trajectory.

The MLX decoder memory-maps complete groups and produces:

- board entities `(batch, candidates, 4, 23, 33)`, with changed-tile and
  changed-wildlife markers appended only to acting-seat entities;
- the existing market entities, masks, and 96 global features;
- a normalized 63-dimensional explicit action vector;
- candidate masks and complete teacher/group metadata.

Production commands:

```bash
make enrich-action-ranking
make collect-action-ranking-test

target/release/cascadia-v2 validate-action-ranking-dataset \
  --dataset artifacts/datasets/action-ranking-terminal-r8-train
```

`collect-action-ranking-test` first collects 16 terminal R8 games from test
indices 0-15 into one-game resumable source shards, then enriches and validates
them. The architecture, decoder, loss, optimizer, and gates are frozen before
that command is run.

## Exact MLX Rollout-Value Dataset

ADR 0065 records value evidence already produced by complete exact-MLX search
rollouts. Each `.nnv` file is one atomic game shard with magic `CSD2NNV\0`,
schema version 1, a 160-byte header, and variable-length records:

- a fixed 40-byte prefix containing record kind, decision and personal-turn
  indices, selected bit, feature count, sample count, game index, rollout
  seed, immediate score, score-to-go target, and root uncertainty;
- the exact ordered sparse `u16` feature multiset, including duplicates.

Trajectory records contain one selected focal-player afterstate and exact
`terminal base score - immediate base score`. Root records contain every
K32 candidate's R600 mean, standard deviation, sample count, selected bit,
and exact sparse afterstate. Dataset validation checks checksums, teacher and
schema hashes, feature bounds, all 80 decisions, exactly one selected root per
decision, turn ordering, game sequencing, and trailing bytes.

The Python decoder indexes variable record offsets without materializing the
dataset, memory-maps each one-game shard, and emits padded differentiable
batches or packed CSR batches for exact Rust-order validation inference.

```bash
make rollout-value-smoke
make collect-rollout-value
```

The substantive ADR 0065 split is frozen at train indices 94,000-94,003 and
validation indices 94,000-94,001, both with R600 and trace modulus eight.
