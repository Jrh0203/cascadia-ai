# Model Format And Inference

## Run Checkpoints

Training checkpoints are immutable directories containing:

- `model.safetensors`;
- `optimizer.safetensors`;
- `state.json` with epoch, next batch, global step, and best metric;
- `checkpoint.json` with architecture and per-file BLAKE3 checksums.

`latest.json` points to the exact resumable cursor. `best.json` points to the
checkpoint with the lowest held-out total-score MAE and embeds that
checkpoint's validation report.

## Promoted Model

`cascadia-mlx-promote` creates:

```text
model.json
model.safetensors
```

`model.json` includes:

- schema and architecture versions;
- weight size and BLAKE3 checksum;
- source run and checkpoint;
- best and final validation metrics;
- dataset manifest hashes;
- MLX, Python, hardware, git, dirty-tree, and v2 source provenance.

Promotion is an atomic rename and refuses to overwrite an existing artifact.
Ranking models use the same two-file artifact shape, add
`kind: "action-ranking"`, and record best listwise validation metrics.
Explicit action models use `kind: "action-delta-ranking"` and additionally
embed the exact untouched-test report that authorized promotion.
Full-legal imitation models use `kind: "canonical-action-imitation"`. Their
promotion command likewise requires a passing untouched-test report,
validation improvement over initialization, and exact checkpoint integrity.
Policy-iteration models also preserve initial warm-start metrics, regression
validation metrics, the aggregate checkpoint-selection loss, and the exact
initial model manifest checksum.

## Exact Legacy NNUE Artifact

`legacy-sparse-nnue-v4opp-mlx-v1` is a read-only MLX representation of the
qualified historical value network:

```text
artifacts/models/legacy-nnue-v4opp-mlx-v1/
  model.json
  model.safetensors
```

The manifest binds the exact source path, byte count, BLAKE3, version,
11,231-512-64-1 dimensions, safetensors byte count, and safetensors BLAKE3.
Loading rechecks the manifest contract, file checksum, tensor names, tensor
shapes, and finite values.

Inference receives padded rank-two sparse indices plus a boolean mask. It
gathers first-layer rows, preserves input order and repeated indices, sums the
active rows with the first bias, and applies the original two ReLU layers and
scalar head. Repeated indices are semantically significant because the
historical extractor emitted them in every parity-fixture state.

ADR 0055 proves direct in-process MLX parity. A separate protocol operation is
required before Rust search may call this evaluator through a long-lived
service; existing entity, ranking, action-delta, and imitation message types
must not be overloaded.

ADR 0056 adds that distinct operation without changing the protocol version:

- request type `5`;
- response type `0x8005`;
- header count equal to sparse rows;
- each row encoded as `u16` length plus ordered `u16` indices;
- one finite little-endian `f32` response per row.

Rows may be empty and may repeat indices. Maximum row width is 4,096 and the
existing maximum request count is 65,536. Both Rust and Python validate the
entire contract. End-to-end batch-32 throughput is 7,589 evaluations per
second at 4.70 ms P99.

ADR 0058 adds a second backward-compatible sparse operation:

- request type `6`;
- response type `0x8006`;
- header count equal to sparse rows;
- payload `u32 total_features`, `(count + 1)` CSR `u32` offsets, then ordered
  `u16` indices;
- one finite little-endian `f32` response per row.

Offsets begin at zero, end at `total_features`, remain monotonic, and encode
rows no wider than 4,096. Empty rows and duplicate indices remain meaningful.
This operation feeds three custom MLX Metal kernels whose scalar accumulation
order matches the qualified Rust NNUE exactly. It is bit-identical on all 80
fixture rows and sustains 75,176 batch-32 evaluations per second at 0.698 ms
P99 end to end.

Score-to-go checkpoints use architecture `entity-set-score-to-go-v1`. Their
eleven outputs are signed normalized residual components, not final score
components. Run manifests therefore use kind `signed-score-to-go` and bind the
`signed-score-to-go-components-v1` dataset manifests. The first experiment did
not clear its validation gate, so no standalone promoted artifact or Rust
inference message type was created for it.

`--init-model-dir` and `--resume` have different contracts. Warm start creates
a new run and optimizer from a verified promoted model. Resume restores an
existing run's exact model, optimizer, batch cursor, data identities, runtime,
and source digest. The untouched warm-start checkpoint is written and made
eligible as `best.json` before the first optimization step.

## Local Protocol

`cascadia-mlx-serve` is a long-lived MLX process. Rust owns its stdin/stdout
through `cascadia-model`.

Every 16-byte little-endian frame header contains:

```text
magic[4] = "CMLX"
version: u16
message_type: u16
request_id: u32
count: u32
```

A prediction request appends `count` compact 864-byte position records. A
prediction response appends `count * 11` float32 score components. Request IDs,
batch bounds, message types, payload lengths, and protocol versions are
validated. Service failures return typed error frames.

The ranking service uses the same request frame and returns message type
`0x8002` followed by one float32 action score per record. Rust validates the
response type and width independently from decomposed value predictions.

The action-delta service uses request type `3`. Each request record is exactly
916 bytes: an 864-byte `compact-entity-v2` observable afterstate followed by
52 bytes of `compact-action-delta-v1`. Response type `0x8003` contains one
float32 score per record. Rust validates message type, request ID, count,
payload width, finite output, and child-process lifetime.

The full-legal imitation service uses request type `4` and response type
`0x8004`. A request contains one 864-byte shared pre-action position followed
by `count` 32-byte explicit action records. The MLX model encodes the shared
board, market, opponents, and global features once, broadcasts that
representation over the action embeddings, and returns one float32 score per
action. Rust validates the response count and finite scores before selecting
among the complete canonical legal set.

Full-frontier distributional imitation changes only the training target and
validation report. It uses the same `shared-state-action-imitation-v1`
architecture and request/response protocol. Rollout means, uncertainties,
sample counts, source flags, and selected bits never enter inference features,
so a qualified checkpoint can use the existing full-legal serving boundary
without teacher-only leakage or a second Rust neural implementation.

The qualified sparse NNUE service also defines request type `7` and response
type `0x8007` for exact hidden-state extraction. Requests use the same CSR
sparse payload as type `6`. Each response row contains 64 little-endian
float32 second-layer activations followed by the exact float32 remaining-value
output. Rust validates the request ID, row count, fixed width 65, and every
finite value before constructing a typed prediction.

`canonical-action-parent-hidden-v1` sidecars use 112-byte shard headers and
312-byte records. A record contains group ID, candidate index/count, 32-byte
canonical action hash, exact immediate and remaining float32 values, 64
float32 hidden activations, and four reserved bytes. Manifests bind the source
MCE evidence and exact parent model checksums; shard validation also requires
complete record-for-record action alignment.

The score-residual successor also uses request type `4`. Its model identity is
`shared-state-action-score-residual-v3`; the returned scalar is a point-scale
final-score estimate formed inside MLX as exact immediate score plus a learned
continuation residual. The residual head is zero-initialized, so an untouched
checkpoint reproduces immediate ordering exactly. Rust still receives one
finite float32 score per canonical action and requires no neural logic.

`MlxPatternRankingStrategy` batches the exact K8+H6+B8 afterstate frontier used
by the qualified terminal teacher. The model sees all four relative boards,
the publicly depleted pre-refill market, phase, Nature Tokens, wildlife
counts, habitat sizes, and scoring configuration. Empty and partial market
slots represent paired and independent drafts exactly. It never receives the
actual hidden refill or hidden stack order.

`MlxActionDeltaRankingStrategy` uses that same exact K8+H6+B8 pattern frontier.
Its input adds explicit draft identity, placements, market-prelude cost,
immediate score deltas, and changed-entity markers while preserving the same
hidden-state-safe pre-refill boundary.

The frozen `action-delta-ranker-v1` model decodes acting-board entities at
dimension 33 and a normalized 63-dimensional action vector. Hidden size is
96, with four attention heads, two board blocks, one market block, and
feed-forward multiplier three. Board, market, global, and action summaries are
projected independently and combined by a scalar ranking trunk.

`mlx-r12-counterfactual-advantage-set-ranker-v1` reuses that observable
action-afterstate encoder, adds a separate normalized 30-value public-supply
projection, and applies two four-head self-attention blocks across the full
four-candidate decision set. Its scalar output is
`immediate_score + 4 * tanh(correction)`. The correction head is initialized
to exact zeros, so untouched inference is bit-exactly the immediate-score
baseline. Checkpoints bind both train and validation manifest hashes,
including ADR 0078's stable-market conditioning contract.

## Derived Sparse NNUE

Rollout-return fine-tuning uses artifact schema 2 while retaining architecture
ID `legacy-sparse-nnue-v4opp-mlx-v1`, dimensions 11,231-512-64-1, and the
qualified source identity. Its `derivation` block binds:

- qualified parent manifest and safetensors BLAKE3 values;
- train and validation dataset-manifest BLAKE3 values;
- immutable training run manifest;
- selected checkpoint ID and checkpoint-manifest BLAKE3.

`model.safetensors` contains the selected six value tensors plus the qualified
parent's unchanged policy tensors. Loading validates dimensions, source,
derivation fields, byte count, and checksum. Packaging is atomic and refuses
identity drift or overwrite. The existing exact CSR service accepts both the
read-only schema-1 parent and self-contained schema-2 derived artifacts.

Action-delta promotion is deliberately stricter than packaging a normal
validation winner. The best checkpoint must improve validation selection loss
over initialization; `test-report.json` must evaluate that exact checkpoint
on a split marked `test`; its teacher must equal the training teacher; and all
four preregistered ranking gates must pass. The promoter verifies every
checkpoint checksum and refuses overwrite.

## Learned Value Strategy

`cascadia-search` evaluates all legal afterstates through one batched service
request stream. It reuses one public-state template per turn and replaces only
the active board, so opponent and market encoding is not redundantly rebuilt.
The afterstate removes drafted public market components and stops before
refill chance. Models trained under older `compact-entity-v1` semantics are
schema-incompatible with this boundary.

The strategy remains experimental until a held-out playing-strength comparison
passes the promotion threshold.

## Commands

```bash
cargo run -p cascadia-cli-v2 -- model-smoke \
  --model-dir artifacts/models/entity-value-v1

cargo run --release -p cascadia-cli-v2 -- model-benchmark \
  --model-dir artifacts/models/entity-value-v1 --games 50

cargo run --release -p cascadia-cli-v2 -- ranking-model-benchmark \
  --run-dir artifacts/runs/entity-ranker-v1 --games 50

cargo run --release -p cascadia-cli-v2 -- habitat-ranking-model-h2h \
  --baseline-model-dir artifacts/models/entity-ranker-v1-h6 \
  --treatment-model-dir artifacts/models/entity-ranker-v1-h6-iteration1 \
  --games 50 --first-seed 23800

cargo run --release -p cascadia-cli-v2 -- pattern-ranking-model-compare \
  --model-dir artifacts/models/entity-ranker-v2-terminal-r8-observable \
  --games 10 --first-seed 25100 \
  --policy-candidates 8 --policy-habitat-candidates 6 \
  --policy-bear-candidates 8 --policy-market-draws 4

make evaluate-action-ranking-test
make promote-action-ranking
make evaluate-action-ranking ACTION_RANKING_GAMES=10
```
