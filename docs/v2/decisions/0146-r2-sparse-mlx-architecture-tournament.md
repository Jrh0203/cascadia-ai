# ADR 0146: R2 Sparse MLX Architecture Tournament

Status: accepted; experiment completed; Perceiver fixed latents selected

Date: 2026-06-17

Experiment: `r2-sparse-mlx-architecture-tournament-v1`

Protocol: `r2-sparse-mlx-matched-architecture-v1`

Foundation: ADR 0145 and
`docs/v2/reports/r2-sparse-occupied-frontier-result.md`

## Context

ADR 0145 proved that the exact R2 occupied, legal-frontier,
habitat-component, and wildlife-motif substrate is lossless on the accepted
60,000-position corpus. A position has 199 active spatial tokens at the
median, 323 at P99, and 340 at maximum across all four boards. The maximum
individual board has 92 tokens, and the per-board P99 is 83.

The exact per-position token means are:

| Token class | Mean |
|---|---:|
| Occupied | 51.5000 |
| Legal frontier | 69.2652 |
| Habitat component | 37.6267 |
| Wildlife motif | 39.4323 |
| **Total** | **197.8242** |

Frontier is the largest class. The learned screen must preserve all frontier
objects while preventing a raw count-weighted pooling operation from silently
making token multiplicity the architecture's decision rule.

That result authorizes a learned architecture screen. It does not authorize
changing the corpus, targets, D6 schedule, training budget, or token content.
The experiment must isolate the model trunk while retaining exact public
state and explicit board ownership.

## Decision

Implement one matched local MLX screen with four runs:

| Run role | Architecture | Assigned host |
|---|---|---|
| `set-primary` | padded Set Transformer | john1 |
| `graph-primary` | directional graph message passing plus global attention | john2 |
| `perceiver-primary` | Perceiver-style fixed latents | john3 |
| `set-replay` | independent padded Set Transformer replay | john4 |

The fourth host is a scientific replay, not duplicate production training. It
tests whether the frozen sample, D6, initialization, optimization, and
classification contracts reproduce on an independent Apple Silicon host.

## Exact Cache

The standalone ADR 0145 Rust crate owns MLX export. `CSR2SP1` stores the
authoritative public global, player, market, occupied-tile, and optional
supplied-tile state. Frontier, habitat components, wildlife motifs, and graph
relations are deterministic projections and are never treated as
independently authoritative.

For every accepted row, the Rust exporter:

1. reconstructs the exact source record from authoritative state;
2. round-trips canonical `CSR2SP1` bytes;
3. regenerates every derived token and graph relation;
4. validates the frozen census, board maxima, and graph degree;
5. regenerates and inverse-checks all 12 D6 states; and
6. only then writes the derived token and graph tensors into a
   content-addressed training cache.

Python never invokes frontier, component, motif, or graph constructors. A
batch only slices Rust-authored tensors, applies the frozen rules-owned D6
coordinate/direction/rotation tables, and expands cached CSR edge rows into
the fixed MLX batch shape. It does not discover neighbors, components, motifs,
or relation types during training.

Each position uses four explicit board blocks:

| Dimension | Capacity | Foundation observation |
|---|---:|---:|
| Relative boards | 4 | 4 |
| Rows per board | 92 | P99 83, maximum 92 |
| **Rows per position** | **368** | maximum active 340 |

Within each board, active rows are contiguous in the canonical order
occupied, legal frontier, habitat component, wildlife motif. Class counts are
variable; there are no global fixed layer offsets.

Padding is all zero and truncation is forbidden. Every active token carries:

- a four-way token-type one-hot;
- a four-way relative-seat one-hot, explicitly preserving board ownership;
- the exact 52-wide Rust-authored payload; and
- an exact graph neighborhood when the graph arm consumes it.

The cache reports both the 368-row padded capacity and, for train and
validation independently, active-token total, mean, maximum, maximum per
board, padding total, and per-type total, mean, fraction, and maximum. It
records the ADR 0145 per-board P99 of 83 and maximum of 92.

The graph cache is CSR on disk and bounded at degree 24 only after exhaustive
foundation validation. Relations distinguish occupied adjacency,
occupied-frontier, occupied-component, occupied-motif, frontier-component,
and motif adjacency in both required directions. Six directional bits are
transformed by the rules-owned D6 table.

The cache directory name is the canonical BLAKE3 of its scientific identity.
That identity includes every tensor file checksum, corpus lock, exporter
binary, board-local tensor contract, D6 metadata, semantic digests, and target
digest. Reusing an existing content address rechecks the complete expected
file set, byte counts, and BLAKE3 digests.

## D6 Contract

The cache stores the exact identity representation once. For every one of the
60,000 positions, Rust regenerates and hashes all 12 D6 transforms and proves
their inverses before the cache can be finalized.

Python applies only the frozen coordinate, direction, and tile-rotation
tables exported by the accepted D6 contract. Training transform IDs are a
deterministic function of `(seed, optimizer_step, example_index)`.

## Shared Encoder

Every architecture uses the same:

- 60-to-64 token input adapter;
- 31-to-64 market adapter;
- 23-to-64 per-player adapter;
- 96-to-64 global adapter;
- relative-seat ownership channels;
- four equal-status type-summary tokens per board;
- one explicit player token per board;
- one cross-board attention block over only global, market, and four
  player-board summaries;
- masked mean-plus-max summary projections; and
- 64-to-128-to-11 nonnegative value head.

`R2SparseValueModel.__call__` invokes `encode_state` exactly once. A focused
test instruments the common encoder and requires one invocation per
prediction. Padding invariance is tested independently for all three trunks.
The four type summaries use independent masked means, so frontier multiplicity
does not dominate final board pooling merely because frontier is the largest
token class.

## Architecture Arms

### Padded Set Transformer

For each board independently, two masked self-attention blocks operate on one
player token, four type-summary tokens, and the exact 92-row board block. Only
the five fixed summary tokens are pooled into that board's output.

### Directional Graph Plus Global Attention

For each board independently, two directional message-passing blocks consume
the cached exact graph relations and direction bits. One masked board-local
attention block mixes the resulting tokens with one player and four
type-summary tokens. No raw graph edge or token crosses boards.

### Perceiver Fixed Latents

Each board gets sixteen learned fixed latents. They cross-attend once to that
board's player token, four type summaries, and exact 92-row block, followed by
one latent self-attention block.

All cross-board interaction occurs later through the explicit global, market,
and four player-board summary tokens. No architecture performs all-pairs
attention over the 368 raw rows.

## Matched Capacity

Parameter matching counts every trainable parameter, including input
adapters, embeddings, relation/direction adapters, common encoder, trunk,
summary projection, and value head.

| Architecture | Trainable parameters |
|---|---:|
| Padded Set Transformer | 141,131 |
| Directional graph attention | 143,915 |
| Perceiver fixed latents | 142,283 |

Maximum spread is 1.9726%, below the frozen 3% gate. No dormant branch,
untrained ballast, or ignored parameter is included to manufacture equality.

## Frozen Optimization

| Variable | Value |
|---|---:|
| Seed | 2026061702 |
| Optimizer | AdamW |
| Steps | 500 |
| Batch size | 32 |
| Learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Checkpoint interval | 100 |
| Metric interval | 25 |
| Evaluation batch | 64 |
| Inference batch | 64 |
| D6 policy | Uniform per example over IDs 0 through 11 |
| MLX cache limit | 1 GiB |

Every primary and replay run receives identical training examples, transform
IDs, targets, optimizer steps, and seed. Checkpoints include model,
optimizer, and exact next-step cursor. Resume rejects protocol, runtime,
source, cache, corpus, or authorization drift.

## Evidence

Each run reports:

- complete train and validation component metrics;
- total MAE, RMSE, bias, correlation, and calibration;
- validation masking ablations for occupied, frontier, component, and motif
  tokens, performed by masking cached tensors and incident cached edges
  without semantic regeneration;
- a fixed 256-row validation prediction panel;
- cold compiled invocation;
- warm and steady inference throughput;
- P50, P90, and P99 latency;
- forward-plus-backward throughput;
- MLX active, cache, and peak memory;
- process peak RSS;
- cumulative training throughput; and
- final checkpoint content identity.

One scored public afterstate is one inference action.

## R0 Binding

Production authorization fails closed unless the completed ADR 0142
classification and its order proof are available and byte-identical.

If ADR 0142 selected a compact Stage 2 candidate, that report is the R0 value
reference. If and only if `selected_stage2_candidate` is explicitly null, the
binding selects `exact-entity-control`. Missing or ambiguous R0 evidence does
not imply a fallback.

The current completed R0 result has an explicit null selection, so the
expected binding is the exact-entity control. The binding is content-addressed
and included in production authorization.

## Immutable Execution

Production execution requires:

1. a bundle built and validated by `tools/rust_experiment_bundle.py`;
2. the exact release exporter binary in that bundle;
3. a frozen corpus lock;
4. a completed R0 control binding;
5. explicit parent authorization bound to all preceding identities; and
6. successful Apple Silicon MLX GPU preflight on all four hosts.

Every frozen Python command uses `-B`. The campaign tool emits a
scheduler-compatible task graph but contains no queue-install operation and
does not import the live queue library. It cannot modify the live research
queue or dashboard ledger.

## Production Authorization

John Herrick authorized the complete four-host production tournament through
`CASCADIA_V2_GOAL.txt` on 2026-06-17. The authorization is fail-closed and
binds the exact four run roles, immutable implementation, exporter, corpus,
protocol, and R0 reference:

| Identity | Value |
|---|---|
| Bundle | `65613a7526f292af47740c1ef21bc3bca22efa5454f0a9b3948c037f2db2a962` |
| Authorization | `780455bfd22643c07ec8c9624d4e2c81342f7f3887a49087b7ea1bc759e5b6a3` |
| Corpus lock | `264315f193ea69232c699277f503c50aa1e1026eb1d27fecd7127503ff6ae0f7` |
| R0 control binding | `a314077de5cce28dabd7d25d17bd6520fb830c85d2c6f624688aa7875615e8a6` |
| Exporter BLAKE3 | `a93b723af21c241da8d5813a87fa42004672d96b199b40c14622c21e06c5a974` |
| MLX source BLAKE3 | `dfb035ef7b93ebbd5d230cb80e743894aa493cd90e1879ce4418086e26ba4173` |
| Queue task graph BLAKE3 | `d40000a0c0b790668421a5d5fa001ab5c9bc3359ffb9771b6380e4f1b6e2a11e` |

The authorized launch contains sixteen scheduler tasks: immutable bundle and
control fanout, four independent host preflights, one exact cache export,
byte-identical cache fanout, three primary architecture runs, one independent
Set Transformer replay, report collection, forward and reverse
classification, and a byte-identical order proof.

## Result

The complete four-host production tournament finished on 2026-06-17. All
sixteen scheduler tasks completed successfully, the independent Set
Transformer replay reproduced the primary exactly, and forward/reverse
classification outputs were byte-identical.

The deterministic classifier selected `perceiver-fixed-latents`. It was the
only primary architecture to pass all three frozen R0 value-noninferiority
gates:

| Architecture | Total MAE delta | Total RMSE delta | Mean component MAE delta | Eligible |
|---|---:|---:|---:|---|
| Padded Set Transformer | -0.044343 | -0.043729 | +0.452482 | No |
| Directional graph attention | +0.001391 | -0.013247 | +0.441530 | No |
| Perceiver fixed latents | +0.292230 | +0.340877 | +0.162042 | Yes |

The selected Perceiver reached 10,178.87 inference actions per second, 6.213
ms P50 latency, and 121.8 MB peak active inference memory at batch 64. Its
occupied-token ablation increased total MAE by 0.033931. Frontier, component,
and motif masking did not hurt aggregate total error under this short
value-only protocol, motivating direct relational or token-type supervision
in successor experiments.

Full result:
`docs/v2/reports/r2-sparse-mlx-architecture-tournament-v1-result.md`.

## Consequences

The Perceiver fixed-latent trunk becomes the R2 baseline for later sparse
decision-quality work. The Set Transformer and directional graph trunks do
not advance from this screen because they failed the component-fidelity gate.

The result cannot claim gameplay strength, authorize promotion, or establish
progress to a 100-point mean. Complete-action ranking, retained regret,
realistic serving latency, paired gameplay, and mean score remain downstream
requirements.
