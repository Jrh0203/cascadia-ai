# Local MLX Training

## Environment

```bash
make setup
make mlx-device
```

The locked environment uses uv-managed CPython 3.12 and MLX on
`Device(gpu, 0)`. Neural training and Apple GPU computation live only in
`python/cascadia_mlx`.

## Baseline Model

`entity-set-value-v1` projects compact tile entities, applies masked
self-attention independently to each relative board and the market, pools the
sets, combines them with public summary features, and predicts eleven
decomposed final base-score components.

Targets are normalized per category during optimization and converted back to
points for validation. Reports include per-component MAE/RMSE/bias, total-score
MAE/RMSE/bias, prediction and target means, Pearson correlation, and linear
calibration slope/intercept.

## Train

```bash
uv run cascadia-mlx-train \
  --train-dataset artifacts/datasets/greedy-train \
  --validation-dataset artifacts/datasets/greedy-validation \
  --run-dir artifacts/runs/entity-value-v1 \
  --epochs 10 --batch-size 256
```

Every run writes:

- `run.json`: hyperparameters, dataset manifest hashes, runtime, and device;
- `metrics.jsonl`: append-only epoch metrics;
- `checkpoints/*/model.safetensors`;
- `checkpoints/*/optimizer.safetensors`;
- `checkpoints/*/state.json`;
- checksummed checkpoint manifests and an atomic `latest.json`;
- `final-report.json`.

Validation improvements atomically update `best.json`. The pointer records the
exact checkpoint and metrics from that checkpoint; promotion never assumes the
last epoch was the best epoch.

## Resume

```bash
uv run cascadia-mlx-train \
  --train-dataset artifacts/datasets/greedy-train \
  --validation-dataset artifacts/datasets/greedy-validation \
  --run-dir artifacts/runs/entity-value-v1 \
  --epochs 20 --batch-size 256 --resume
```

The latest complete checkpoint restores model weights, AdamW moments, global
step, epoch, and the exact next batch. Shuffle order is a deterministic
function of the run seed and epoch. Corrupt or partially written checkpoints
are rejected.

Resume is deliberately strict. The trainer rejects changes to the dataset,
model, optimizer, batching, runtime, or source digest; only the requested
epoch ceiling and explicit resume checkpoint may differ. This prevents a
nominal resume from silently becoming a different experiment.

`make train-smoke` performs collection, GPU training, checkpointing, and a
second resumed epoch, promotion, and Rust-to-MLX inference end to end.

## R2-MAP expert-iteration benchmark overlap

During John1 MLX training, only the previous verified checkpoint's fixed
100-game longitudinal panel may use available remote CPU. Every game is an
independent `game-NNNN` work item; Bacalhau owns placement and may use john2 or
john3 without a host or parity assignment. No worker generates the next
training round. The campaign cannot enter the paired candidate gate until both
training and all 100 benchmark receipts are terminal and independently
validated.

The operational commands and artifact schemas are documented in the
`R2-MAP topology-free serving and benchmark work items` section of `CLI_REFERENCE.md` and
the ordinary worker-image workflow is documented in `R2_MAP_DOCKER.md`.
Execution is restart-safe at the checksummed game-work-item boundary.
John1's internal root `/Users/johnherrick/cascadia-bench/r2-map-v1` is the
primary active store. John2 and john3 return checksummed Docker outputs to
john1; john2 retains only dependency-closed cold archives.

John1 runs MLX/Metal training natively and writes loss telemetry, checkpoint
bundles, optimizer state, verification results, and recovery pointers beneath
the primary active root. The former zero-write/remote-storage attestation path
and its D0 trust-chain dependencies are superseded and must not gate training.
Training consumes locally verified, checksummed generation shards and keeps its
compact-window reads bounded. Ordinary checkpoint manifests and atomic pointer
updates provide recovery; no remote CAS or signed publication protocol is
required.

Production training emits hash-chained loss telemetry every 20 optimizer steps
and checkpoints at the configured fixed-step interval, no later than five
minutes after the prior complete checkpoint, plus final completion. MLX's
in-process cache is capped at 1 GiB. Validation selection is incremental and
deterministic, so the runner retains only the current serialized checkpoint
and the best verified checkpoint instead of accumulating every checkpoint
bundle across the 45-minute run. A checkpoint is eligible for
`last_verified` only after exact bundle reload, fixed-panel replay, and
next-batch resume verification. A failed write, reload, or validation stops at
the last verified checkpoint on John1.

Before freezing the production group cap, run
`tools/r2_map_john1_packing_sweep.py` on John1 against the checksummed compact
index in John1's campaign root. It reads disposable windows only into memory,
derives exact steps for each of 12 D6 epochs from replay candidate widths, and
measures representative optimizer steps for group caps `[16, 32, 64, 128]`.
Keep the fastest cap that stays within the configured memory and candidate
budgets. Record the source revision, compact-index checksum, selected cap, and
measured timing in the run directory; these are ordinary reproducibility
metadata, not a distributed trust gate.

The native training command accepts explicit local compact paths below
`/Users/johnherrick/cascadia-bench/r2-map-v1`, validates their checksums and
dataset identity, and binds the selected adapter parameters into every
checkpoint. The console entry point is `cascadia-mlx-r2-map-train`; serving,
verification, and promotion are exposed as the matching
`cascadia-mlx-r2-map-{serve,verify,promote}` commands. Linux Docker workers do
not run MLX training.

The local runner resolves market preparation sequentially from public state.
It scores the free replacement choice, commits it, then scores one paid wipe or
stop at a time, rebuilding after each public reveal, and finally scores every
legal draft action without pruning. Benchmark mode is deterministic argmax at
all stages. The bundled replay action remains the accounting authority for the
free-replacement count, each paid wipe, independent-draft spend, and final
Pinecone conservation.

The W3 integrated smoke checkpoint may exercise the open performance panel,
but that report is explicitly `real-open-checkpoint-performance-only`. It is
not a playing-strength estimate and cannot promote a model. The blinded
20-pair and fixed-250 stages remain unopened until the campaign controller
enters their registered phase.

## Promote

```bash
make promote \
  RUN_DIR=artifacts/runs/entity-value-v1 \
  MODEL_DIR=artifacts/models/entity-value-v1
```

Promotion verifies the best checkpoint and atomically creates a standalone
model artifact. Existing destinations are never overwritten.

## Playing-Strength Evaluation

Run an absolute benchmark:

```bash
make evaluate-model \
  MODEL_DIR=artifacts/models/entity-value-v1 \
  MODEL_GAMES=20
```

Run the preferred paired comparison against greedy on identical held-out game
seeds:

```bash
make compare-model \
  MODEL_DIR=artifacts/models/entity-value-v1 \
  MODEL_GAMES=20
```

The paired report includes the treatment-minus-baseline confidence interval,
game wins/ties/losses, and category deltas.

## Search-Policy Value Data

`make collect-search` creates disjoint 256-game train and 64-game validation
datasets from the H6 research teacher. They use the same fixed records and
decomposed final-score targets as the baseline value pipeline, but the
trajectory distribution comes from the independently measured search policy
rather than greedy play. The teacher's historical product promotion has been
superseded by ADR 0068; these immutable datasets remain valid research
artifacts and do not imply current product status.

```bash
make collect-search
make train-search-value EPOCHS=20
```

This model is intended for held-out leaf/value experiments with an explicit
gameplay gate. It is not promoted merely for improving validation MAE.

## Signed Score To Go

`entity-set-score-to-go-v1` reuses the hidden-96, four-head, two-board-block,
one-market-block public entity encoder and predicts eleven signed normalized
`final - current` components through a linear head. Validation adds the exact
current components back and reports both residual and reconstructed-final
metrics.

```bash
make collect-score-to-go
make train-score-to-go
```

The trainer uses the same atomic model/optimizer/cursor checkpoints and strict
source/data/runtime resume contract as the baseline value trainer. The
implementation proof ran one epoch, resumed into a second epoch on
`Device(gpu, 0)`, and preserved optimizer steps.

The frozen 256/64 H6 experiment selected epoch 13 at 2.568601
reconstructed-final MAE, 0.397451 final correlation, and 0.991700 residual
correlation. Every component guardrail passed, but final correlation missed
the required 0.50, so the model was rejected before promotion or gameplay.

## Search Distillation

`entity-set-ranker-v1` trains on complete candidate groups labeled by fair
public-information search. The loss is uncertainty-weighted listwise
cross-entropy; validation reports tie-aware top-1 value recall, strict
single-index accuracy, top-1 teacher regret, pairwise accuracy, mean rank
correlation, and value-difference correlation.

```bash
make collect-ranking

uv run cascadia-mlx-ranking-train \
  --train-dataset artifacts/datasets/ranking-h6-train \
  --validation-dataset artifacts/datasets/ranking-h6-validation \
  --run-dir artifacts/runs/entity-ranker-v1-h6 \
  --epochs 20 --group-batch-size 16 --validation-patience 5
```

Training checkpoints use the same atomic, checksummed model, optimizer, cursor,
best-pointer, and exact-resume machinery as value training. The consecutive
non-improving validation-epoch count is checkpointed too, so interruption does
not reset the futility budget. Training stops after five non-improving epochs
by default, while inference and promotion continue to select `best.json`.

```bash
make promote-ranking \
  RANKING_RUN_DIR=artifacts/runs/entity-ranker-v1-h6 \
  RANKING_MODEL_DIR=artifacts/models/entity-ranker-v1-h6
```

```bash
target/release/cascadia-v2 ranking-habitat-prefilter-compare \
  --model-dir artifacts/models/entity-ranker-v1-h6 \
  --games 10 --first-seed 22500 \
  --baseline-candidates 8 --baseline-habitat-candidates 6 \
  --candidates 16 --habitat-candidates 8 \
  --immediate-anchors 8 --prefilter-candidates 14 \
  --determinizations 4 --greedy-plies 4
```

This experiment preserves the exact immediate-score K8 frontier, lets MLX
choose six additional actions from a wider K16+H8 union, and then evaluates all
fourteen retained actions with the unchanged fair R4/D4 rollout teacher.

If the fresh H6 ranker clears its held-out gates, a separately preregistered
pilot uses it only for future rollout actions while preserving the exact H6
root candidate set and common determinizations:

```bash
target/release/cascadia-v2 ranking-habitat-rollout-compare \
  --model-dir artifacts/models/entity-ranker-v1-h6 \
  --games 10 --first-seed 22600 \
  --candidates 8 --habitat-candidates 6 \
  --determinizations 4 --rollout-plies 4 \
  --rollout-candidates 8 --rollout-habitat-candidates 6
```

The rollout implementation batches every branch frontier into one MLX request
per ply. The ranker never receives hidden bag order, and exact scoring remains
the root comparison value after the configured horizon. This is an
unrestricted research strategy: a full-configuration one-game runtime smoke
must pass before the preregistered ten-game strength pilot.

### H6 value-leaf search

The H6 trajectory value experiment uses the same decomposed entity-set model,
but its targets come from complete games played by the confirmed H6 teacher.
Every public pre-action state is labeled with the acting seat's final Bear,
Elk, Salmon, Hawk, Fox, five habitat, and Nature Token components.

```bash
make collect-search
make train-search-value EPOCHS=20
make promote-search-value
```

Gameplay evaluation is conditional on held-out total correlation of at least
0.50, total MAE of at most 4.0, and the registered component-error gates:

```bash
make evaluate-value-leaf MODEL_GAMES=10
```

`value-leaf-compare` preserves the exact H6 root union, common hidden-state
determinizations, and four greedy future plies. Terminal branches use exact
base score; all remaining public leaves are batched into one MLX request and
ranked by predicted final decomposed total for the original acting seat. The
model never receives hidden bag order. This isolates leaf-value quality from
candidate generation and rollout policy changes.

The implementation smoke used the previously rejected greedy-trained model at
K2/H1/R1/D1. It completed on MLX GPU in 5.439 treatment seconds with 67.98 ms
mean decision latency and a fully checksummed model manifest. Its negative
single-game score is intentionally not research evidence; it verifies the
batched boundary, report provenance, and sensitivity to model quality.

## Search-Guided Policy Iteration

The iteration collector implements an Expert Iteration/DAgger-style local
loop. A frozen MLX apprentice controls all four seats. At every state, frozen
H6 search evaluates the complete K8+H6 candidate set; those counterfactual
labels are stored before the apprentice chooses its action from that same set.
This moves training onto the apprentice's actual state distribution without
letting the apprentice define its own targets.

```bash
make collect-ranking-iteration
make train-ranking-iteration
make promote-ranking-iteration
make evaluate-ranking-iteration
```

Iteration training:

- warm-starts from an immutable promoted model via `--init-model-dir`;
- aggregates the original teacher-trajectory dataset with apprentice-trajectory
  data via repeated `--additional-train-dataset`;
- evaluates both the new on-policy validation split and the original validation
  split via `--regression-validation-dataset`;
- chooses checkpoints by the mean listwise loss across all validation
  distributions;
- makes the untouched warm-start checkpoint eligible as best, preventing a
  worse first epoch from being promoted;
- records every dataset manifest and the initial model manifest checksum in
  `run.json`.

Warm start and resume are intentionally different. Warm start initializes a
new optimizer and a new experiment from a promoted model. `--resume` restores
the exact model, optimizer, cursor, datasets, runtime, and source identity of
an interrupted run.

`make ranking-iteration-smoke` exercises collection, aggregation, regression
validation, warm start, checkpointing, promotion, and two-model Rust/MLX
inference end to end.

## R12 Counterfactual-Advantage Set Ranker

ADR 0078 trains one four-candidate MLX set ranker from fresh
selected/high/median/low H6 groups:

```bash
make counterfactual-ranker-smoke
make collect-r12-counterfactual-corpus
make train-r12-counterfactual-ranker
make resume-r12-counterfactual-ranker
make evaluate-r12-counterfactual-ranker
```

Each candidate has twelve shared-seed terminal returns, exact public supply,
the observable post-prelude parent and action afterstate, explicit action
features, immediate score, and shallow H6 statistics. Impossible mandatory
market-stabilization trajectories are deterministically rejection-sampled and
the versioned conditioning contract is part of the teacher hash. Training and
resume reject any source, dataset, teacher, optimizer, or runtime drift.

The model starts bit-exactly at immediate score, trains a bounded correction
with uncertainty-weighted centered Huber plus hard-top and soft-listwise
terms, and selects one checkpoint by the frozen validation decision
objective. Validation alone controls whether a separately preregistered test
corpus may be opened.

## Terminal Policy Distillation

The qualified R8 teacher evaluates every K8+H6+B8 root candidate under eight
shared public-information hidden-state samples, then runs frozen pattern-aware
play to terminal acting-seat score. It scored 94.833 in its registered
three-game qualification but requires roughly 186 seconds per game, so its
role is data generation.

```bash
make terminal-ranking-smoke
make collect-terminal-ranking
make train-terminal-ranking
make promote-terminal-ranking
make evaluate-terminal-ranking RANKING_GAMES=10
```

The substantive defaults collect 64 train games and 16 disjoint validation
games into one-game resumable shards. Training uses the existing
`entity-set-ranker-v1` MLX architecture and complete grouped listwise loss.
Promotion remains conditional on held-out ranking gates, and gameplay remains
conditional on a separately registered paired pilot. A model is not promoted
to product merely because terminal teacher labels are stronger.

## Explicit Action-Delta Distillation

The corrected full-afterstate ranker established that terminal ordering signal
exists but hid the one-action difference inside nearly identical boards. The
registered successor enriches those same labels with explicit action identity
and immediate category deltas while preserving the observable pre-refill
boundary.

The substantive workflow is:

```bash
make action-ranking-smoke
make enrich-action-ranking
make collect-action-ranking-test
make train-action-ranking
make evaluate-action-ranking-test
```

`enrich-action-ranking` deterministically converts the existing 64-game train
and 16-game validation terminal datasets. It does not rerun R8. Every action
hash, immediate score, candidate rank, and afterstate byte must match the
source, and each teacher-selected action is replayed under the original tie
schedule.

`collect-action-ranking-test` is intentionally later in the sequence. It
collects 16 new terminal R8 games from test indices 0-15 only after the record
schema, decoder, architecture, loss, optimizer, and gates are frozen. Test
labels are never used for checkpoint selection or model changes.

Training is fixed to `action-delta-ranker-v1`: hidden 96, four heads, two
board blocks, one market block, feed-forward multiplier three, AdamW at
`1e-4`, weight decay `1e-4`, complete-group batch size 16, seed 20260611,
maximum 20 epochs, and validation patience five. The shared ranking trainer
provides immutable checkpoints, exact resume, initialization-as-a-candidate,
and validation-only selection.

The untouched test evaluator writes `test-report.json` and reports all gates
without disguising a failure as a process error:

- mean top-one regret at most 0.75;
- pairwise accuracy at least 0.65;
- value-difference correlation at least 0.30;
- tie-aware top-one value recall at least 0.45.

Only a model passing all four gates and improving validation selection loss
over initialization may be promoted:

```bash
make promote-action-ranking
make evaluate-action-ranking ACTION_RANKING_GAMES=10
```

The ten-game paired pilot uses seeds 25700-25709 and requires at least +0.5
mean score, no worse than -0.5 wildlife, -0.5 habitat, or -1.0 Nature Tokens,
and at most two treatment seconds per game. Only a passing pilot unlocks the
disjoint 50-game confirmation at seeds 25800-25849, whose paired 95% CI lower
bound must exceed zero under the same mechanism and runtime guardrails.

## Full-Legal Teacher Imitation

ADR 0048 proved that the K8+H6+B8 frontier recalls only 51.25% of the
qualified 96.35 teacher's selections. The registered replacement trains on
64 structured candidates per decision but scores every canonical legal action
at gameplay inference:

```bash
make imitation-smoke
make collect-imitation
make train-imitation
make collect-imitation-test
make evaluate-imitation-test
```

The substantive split uses train indices 50,000-50,063, validation indices
50,000-50,015, and test indices 50,000-50,015. Split-domain hashing makes
those three suites disjoint. Collection uses one-game R600 shards and may be
resumed with the unchanged command; provenance validation rejects source,
binary, teacher, weights, schema, or range drift.

The model is `shared-state-action-imitation-v1`: hidden 96, four attention
heads, two board blocks, one market block, AdamW at `1e-4`, weight decay
`1e-4`, group batch 16, seed 20260612, at most 20 epochs, and patience five.
The selected validation checkpoint advances only if untouched-test top-one is
at least 20%, top-five recall at least 55%, MRR at least 0.40, and validation
loss improves over initialization.

Only a passing test report permits:

```bash
make promote-imitation
make evaluate-imitation IMITATION_GAMES=10
```

The paired pilot uses seeds 32700-32709 and requires at least +0.25 mean gain,
no habitat or aggregate wildlife loss worse than -0.50, and at most ten
seconds per game. The promoted artifact and service support both `--run-dir`
and standalone `--model-dir` inference.

The frozen first run was rejected before promotion. Its selected epoch-three
checkpoint improved validation listwise loss from 4.158443 to 2.948818 and
scored 20.078% top-one on the untouched test split, passing that gate.
Top-five recall was 51.094% versus the 55% floor and MRR was 0.347386 versus
0.40, so `make promote-imitation` and the gameplay pilot were not run. The
test dataset is sealed evidence for this rejected model and must not be reused
to select a successor.

## Full-Frontier Distributional Imitation

ADR 0053 replaces winner-only supervision with fresh, full-frontier R600
evidence. The paired collector stores 96 canonical actions per decision and
preserves every K32 rollout mean, standard deviation, and sample allocation:

```bash
make imitation-evidence-parity
make collect-imitation-evidence
make train-imitation-distribution
```

Training uses the same shared-state action ranker from scratch. For every pair
of teacher-scored actions, the soft target is the probability implied by their
rollout-mean difference and combined standard error, with a fixed one-point
variance floor. Confidence weights downweight near-ties. A coefficient-0.25
selected-action listwise term spans all 96 retained actions, including
unscored pattern and deterministic legal negatives.

The run is selected only by validation distributional loss. Its report also
contains selected top-one, top-five, and MRR; whether the predicted action is
inside the scored teacher frontier; conditional teacher regret; scored
top-one recall; pairwise accuracy, log loss, and Brier score; scored rank
correlation; and score-difference correlation.

Checkpoints retain the same atomic model, optimizer, cursor, best pointer, and
patience state as the other grouped trainers. Resume is strict:

```bash
make resume-imitation-distribution
```

The frozen first run uses 64 train games and 16 split-domain validation games
beginning at index 51,000, hidden size 96, four heads, two board blocks, one
market block, AdamW at `1e-4`, weight decay `1e-4`, batch 16, seed 20260616,
20 epochs maximum, and patience five. It may not warm-start or inspect a test
domain. Every ADR 0053 validation gate must pass before a separately
registered fresh test collection is allowed.

The run was rejected on validation. It achieved 1.534834 distributional loss
and 0.444333 scored value-difference correlation, but only 13.750% top-one,
38.438% top-five, 0.269223 MRR, 71.406% predicted teacher coverage, and
67.975% scored pairwise accuracy. No test or gameplay domain was opened.
The complete result is in
`docs/v2/reports/canonical-action-mce-distribution-v1-validation.md`.

## Point-Scale Continuation Residual

ADR 0054 preserves exact immediate score in the final prediction and trains
only the expected remaining-score correction:

```bash
make collect-imitation-score-residual-validation
make train-imitation-score-residual
make resume-imitation-score-residual
```

The untouched residual head is zero, so initialization is exactly the
immediate-score baseline. The model returns
`immediate_score + 100 * residual` in both training and serving. Scored
teacher actions use uncertainty-weighted point Huber regression; a
temperature-five selected-action listwise term spans all retained actions.

The immutable ADR 0053 train corpus is reused. Validation moves to fresh split
indices 51,016-51,031, and every absolute plus improvement-over-initialization
gate in ADR 0054 is frozen before those seeds are opened.

The run was rejected on fresh validation. Its selected checkpoint reduced
anchored loss from 4.984072 to 0.984838, but top-one regressed from 18.906% to
17.188%, top-five reached only 43.516%, MRR reached 0.307109, teacher coverage
fell to 76.641%, value-difference correlation fell to 0.380501, and
conditional regret rose to 1.157315. No test or gameplay domain was opened.

Only 0.456% of validation continuation-residual variance was within action
groups. Absolute point regression is therefore closed: it mostly learns the
state-level remaining-score offset.

A development-only centered screen on this already-open split improved
teacher-frontier coverage to 83.59% and value-difference correlation to
0.5459, but best exact top-one remained below initialization at 17.50%. No
fresh split was opened and the unused training path was removed. The next
neural work is exact MLX conversion of the qualified sparse NNUE, not another
shared-action imitation loss.

## Exact Qualified NNUE Port

ADR 0055 directly converts the immutable qualified historical NNUE instead of
training another apprentice:

```bash
make legacy-nnue-mlx-port
```

The command verifies the 23,134,992-byte source and its BLAKE3, writes
checksummed safetensors, generates an 80-state Rust fixture, checks synthetic
and real float32 parity, and benchmarks warmed Apple-GPU batches. It performs
no optimization and never changes the source parameters.

The port passed with 0.00004197 maximum real-state error, bit-deterministic
repeated calls, and 40,569 batch-32 evaluations per second. Historical feature
indices are a multiset: all 80 fixture records contained repeats, and exact
inference intentionally sums repeated first-layer rows.

The artifact is authorized for a separately preregistered batched service and
search integration. It is not a promoted gameplay model by itself.

The separately frozen service boundary also passes:

```bash
make legacy-nnue-mlx-service
```

It adds a typed variable-length sparse operation to the long-lived `CMLX`
process. Across all 80 real fixture rows, framed service output was
bit-identical to direct MLX and sustained 7,589 batch-32 evaluations per
second at 4.70 ms P99 including serialization and pipe I/O. This target
consumes, and never regenerates, ADR 0055 artifacts.

The exact packed successor and full search qualification are reproducible:

```bash
make legacy-nnue-mlx-exact-service
make legacy-nnue-mlx-rollout-parity
```

Request type 6 uses packed CSR offsets and Rust-order custom Metal kernels. It
is bit-identical to Rust, reaches 75,176 batch-32 evaluations per second, and
reproduces all frozen R32/R600 search outputs exactly.

The fresh gameplay baseline is:

```bash
make legacy-nnue-mlx-gameplay-smoke
make legacy-nnue-mlx-gameplay-confirm
```

The confirmed ten-game mean is 95.800 with every neural forward on MLX. This
artifact is the local research control for experiments toward 100; historical
weights remain non-promotable as the final V2 solution.

## Rollout-Return Fine-Tuning

ADR 0065 fine-tunes the six value tensors of the qualified
11,231-512-64-1 sparse NNUE on fresh complete-rollout returns. The policy head
is not part of the trainable MLX module and is copied byte-for-byte from the
qualified parent when the selected checkpoint is packaged.

```bash
make collect-rollout-value
make train-rollout-value
make resume-rollout-value
```

The trainer enforces the preregistered R600 split and minimum record counts.
It uses Huber loss with a four-point transition, AdamW at `3e-6`, no weight
decay, batch 512, seed 20260620, at most 12 epochs, and patience four.
Checkpoints atomically preserve model tensors, optimizer state, the exact next
batch, best pointer, and patience state. Bounded retention always preserves
the latest checkpoint, the best checkpoint, and recent recovery points so an
overnight local run cannot silently exhaust the machine's disk.

Validation uses the exact Rust-order Metal kernels, not the differentiable
training reduction order. Reports include RMSE, MAE, bias, raw Pearson,
within-personal-turn residual Pearson, four phase-quartile RMSEs, root
pairwise accuracy, selected-action top-one, and teacher regret. A derived
artifact is packaged with immutable parent, dataset, run, checkpoint, and file
checksums; gameplay seeds remain closed unless every ADR 0065 gate passes.

The registered run completed on 2026-06-12 and was rejected before gameplay.
Its selected epoch reduced held-out trajectory RMSE by 41.84% and improved
within-turn residual correlation, but root pairwise accuracy and
selected-action top-one both regressed slightly. The candidate is preserved as
reproducible evidence at `artifacts/models/exact-mlx-rollout-return-v1`; it is
not a promoted gameplay model. See ADR 0065 and
`docs/v2/reports/exact-mlx-rollout-return-finetune-validation.md`.

ADR 0066 added one complete root-group pass per epoch with group-centered
Huber, selected-action listwise, and soft teacher listwise losses. On a fresh
validation split it improved selected-action top-one from 27.50% to 29.38%,
but pairwise accuracy and conditional regret regressed. It was also rejected
before gameplay. See
`docs/v2/reports/exact-mlx-joint-return-ranking-validation.md`.

## Exact-Parent Candidate-Set Residual

ADR 0069 keeps the qualified exact MLX evaluator frozen and learns only a
zero-initialized, permutation-equivariant correction across the complete
candidate set, capped at 96 actions. Parent immediate score and remaining
value are stored in a checksummed `.imp` sidecar aligned by dataset identity,
shard range, group ID, candidate index/count, and canonical action hash.

```bash
make collect-imitation-parent-train-priors
make collect-imitation-parent-validation
make collect-imitation-parent-validation-priors
make train-imitation-parent-residual
```

Collection deterministically replays every source game, byte-compares each
public position, reconstructs every compact action, verifies its JSON BLAKE3,
and advances only with the recorded selected action. Paid wipes are rejected
because the compact v1 action schema intentionally lacks their exact slot
sequence. The frozen source teacher never used a paid prelude.

The model compares candidate embeddings through masked mean, maximum, and
candidate-minus-mean features. Its final score is the masked-standardized
exact parent total plus a learned residual, so initialization preserves exact
parent ordering. Training is one Apple-GPU run with AdamW at `5e-5`, weight
decay `1e-4`, group batch eight, seed 20260622, at most 30 epochs, and
patience six. Validation distributional loss alone selects the checkpoint.

`make imitation-parent-residual-smoke` passed the complete data, replay,
exact-MLX, checkpoint, and gate-report path on implementation-only R2 data.
The substantive train split is immutable; fresh R600 validation, sealed test,
and gameplay domains remain controlled by ADR 0069's frozen gates.

The substantive run was rejected on fresh validation. Epoch 3 reduced
distributional loss from 1.528932 to 1.397396 and improved top-five recall
from 49.453% to 54.688%, but top-one rose only 1.875 points, MRR rose only
0.03061, pairwise accuracy regressed 0.756 points, value-difference
correlation regressed 0.01559, and regret improved only 0.01011. Train top-one
rose 2.695 points versus the required five. No test or gameplay domain was
opened. See
`docs/v2/reports/exact-parent-candidate-set-residual-validation.md`.

## Exact-Parent Hidden-State Residual

ADR 0070 isolates the representation hypothesis left by ADR 0069. The exact
Rust-order MLX service now has a typed request that returns the 64-value second
hidden layer and its exact scalar output in one response. The scalar is
bit-identical to the existing exact operation.

The `.imh` sidecar stores group/candidate identity, canonical action BLAKE3,
exact immediate and remaining values, and 64 float32 hidden activations in a
312-byte aligned record. Collection retains the same deterministic replay,
public-position comparison, source/model identity, checksum, and resumption
guards as the exact-parent prior.

```bash
make imitation-parent-hidden-smoke
make collect-imitation-parent-train-hidden
make collect-imitation-parent-hidden-validation
make collect-imitation-parent-validation-hidden
make train-imitation-parent-hidden
make resume-imitation-parent-hidden
```

`exact-parent-hidden-set-residual-v5` has no public entity or hand-built action
encoder. It projects only normalized exact hidden activations and five parent
scalars, then compares candidates through masked mean, maximum, and
candidate-minus-mean context. Its output layer is zero-initialized, so the
initial score is exactly the standardized parent total. AdamW, split sizes,
loss, checkpoint selection, patience, and advancement gates are unchanged
from ADR 0069.

The R2 implementation smoke passed 100% replay, hash, hidden, checkpoint, and
report integrity on index 90,011.

The single authorized R600 run is complete and rejected. Epoch 30 reduced
fresh-validation distributional loss from 1.522383 to 1.417843, but
selected-action top-one improved only 0.078 percentage point, top-five only
0.703 point, MRR only 0.002335, and regret only 0.001245 point. Train top-one
was exactly unchanged. Six gates failed, so test and gameplay were not opened.
See
`docs/v2/reports/exact-parent-hidden-state-residual-validation.md`.

This result closes residual learning on the fixed historical parent. The next
research step is to measure whether the R600 teacher evidence contains
statistically identifiable action winners before designing a fresh V2
policy/search representation.

## MCE Teacher Identifiability

```bash
make audit-imitation-identifiability
```

The audit streams the checksummed parent-hidden evidence and reports
top-two margins, uncertainty scales, confidence-set sizes, parent/immediate
ranks, and phase breakdowns. On fresh validation, only 18.359% of winners
clear a 95% normal difference test, only 6.953% have non-overlapping 95%
intervals, and the mean 95% confidence set contains 10.140 actions. Opening
turns are least identifiable at 6.563% and 16.309 actions respectively.

See `docs/v2/reports/mce-teacher-identifiability.md`. Exact-action imitation
from this independent-seed teacher is closed. ADR 0071 first tests whether
same-budget common random numbers improve the teacher policy itself.

## Common-Random-Number Teacher

```bash
make exact-mlx-crn-qualification
make exact-mlx-crn-smoke
make exact-mlx-crn-pilot
```

ADR 0071 preserves the exact MLX K32/R600/LMR search and changes only rollout
seed coupling. Within each halving round, alive candidates receive the same
ordered seed prefix. The independent path remained bit-exact against native
Rust; full CRN search replays deterministically.

The smoke passed at +1.25. The three-game pilot scored 97.083 versus 95.917,
or +1.167 paired with 95% CI `[+0.578,+1.756]`, while preserving wildlife,
habitat, token, runtime, integrity, and shutdown gates. The result is
promising, not promoted.

ADR 0072 froze the required confirmation:

```bash
make exact-mlx-crn-confirm
```

It ran exactly 20 fresh paired games on seeds 35,703-35,722 and additionally
required the paired 95% lower bound to remain above zero.

The confirmation rejected CRN: 95.413 versus 95.775 independent, paired
-0.363 with 95% CI `[-1.129,+0.404]`, record 8-1-11. Wildlife, habitat,
tokens, runtime, legality, fallback, rollout accounting, and clean shutdown
all stayed inside guardrails. No CRN training data may be collected. The next
learner must use a fresh MLX-native V2 representation and confidence-set,
preference, or regret-aware targets rather than exact noisy argmax labels.
