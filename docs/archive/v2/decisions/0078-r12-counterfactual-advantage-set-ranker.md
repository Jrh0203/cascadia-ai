# ADR 0078: R12 Counterfactual-Advantage Set Ranker

Status: rejected on validation on 2026-06-14. The corrected fresh train and
validation corpora completed and passed all integrity checks. The single
frozen MLX run stopped after five non-improving epochs, and the untouched
zero-output checkpoint remained best. Six validation gates failed. Test,
inference, gameplay, and promotion remained unopened.

## Context

ADR 0077 qualified the selected/high/median/low counterfactual target on fresh
validation. R12 reproduced R16 with 0.204 centered MAE, 0.968 correlation,
92.19% pairwise accuracy, 78.13% exact winner agreement, and 0.037 mean
winner regret. Mean R16 group range was 2.469 points, and a 160-game R12
corpus projects to 10.33 uncontended local hours.

The target is therefore stable, wide enough to learn, and affordable locally.
The remaining question is whether a complete-candidate-set neural model can
learn the terminal advantage from observable afterstates and exact public
supply while improving on the H6 action that generated each group.

The prior public-beam set ranker is not a warm start. Its teacher, candidate
set, state boundary, and target differ, and ADR 0040 closed further model
changes on that corpus.

## Frozen Data

Collect fresh R12 records with the unchanged ADR 0077 contract:

- symmetric four-player AAAAA with no habitat bonuses;
- source and continuation policy H6 K8/H6/R4/D4;
- canonical public post-prelude decision state;
- selected action plus highest, median, and lowest remaining ranked H6
  alternatives;
- four candidates and sixteen evenly spaced groups per game;
- twelve identical ordered public-redetermination seeds per candidate group;
- deterministic rejection sampling of any sampled trajectory that has no
  legal stabilized wildlife market, recorded as
  `reject-unstable-market-trajectories-v1` in the teacher manifest;
- raw decomposed terminal returns, exact public supply, parent state, action
  afterstates and hashes, immediate values, and shallow H6 statistics;
- local Apple M4 execution only.

Fresh substantive domains are:

- train indices 69,000-69,127: 128 games, 2,048 groups, 8,192 candidates,
  and 98,304 continuations;
- validation indices 70,000-70,031: 32 games, 512 groups, 2,048 candidates,
  and 24,576 continuations.

ADR 0074-0077 datasets may validate compatibility but may not train, select,
or preview this model. Test and final split data are prohibited.

Implementation-only end-to-end smoke may use train index 9,996 and validation
index 9,997 with four groups per game and the same R12 estimator. Smoke data
cannot enter substantive training.

## Frozen Model

`mlx-r12-counterfactual-advantage-set-ranker-v1`:

- encodes each observable action afterstate with the existing action-delta
  board, market, global, and explicit-action encoder;
- separately projects the exact 30-value public supply snapshot and includes
  it in every candidate representation;
- projects to hidden width 96;
- applies two four-head masked self-attention blocks over the complete
  four-candidate set;
- predicts a bounded correction to exact immediate score;
- returns `immediate_score + 4 * tanh(correction)` as the decision score.

Board blocks are two, market blocks one, candidate blocks two, and the
feed-forward multiplier is three. The final correction layer is initialized
to exact zeros, making the untouched model bit-exactly equal to the immediate
score baseline. Shallow H6 means, selected index, terminal samples, and target
uncertainty are labels or evaluation context only and are not model inputs.
There is no warm start, architecture sweep, rotation augmentation, or hidden
state input.

## Frozen Objective

For each candidate, the target is the mean total terminal score over its
twelve shared-seed returns. Candidate uncertainty is the sample standard
error of those twelve totals.

The loss is:

- uncertainty-weighted centered Huber regression;
- plus 0.50 uncertainty-weighted hard-top cross-entropy, uniform over exact
  target ties;
- plus 0.25 uncertainty-weighted soft listwise cross-entropy at teacher
  temperature 0.50.

Training uses AdamW, learning rate `1e-4`, weight decay `1e-4`, group batch
size 32, at most 20 epochs, validation patience 5, checkpoint interval 100,
and seed 20260614. Checkpoint selection minimizes:

`mean regret + 0.25 * (1 - top-value recall) + 0.10 * centered MSE`.

## Frozen Validation Gates

The selected checkpoint qualifies only if every condition holds:

- all schema, header, checksum, provenance, sequence, action-identity,
  public-supply, shared-seed, unused-tail, and finite-target checks pass;
- training and checkpoint resumption run on MLX `Device(gpu, 0)`;
- selected validation objective is at least 10% below exact initialization;
- centered MAE is at most 0.75 points and at least 10% below initialization;
- centered-advantage correlation is at least 0.55;
- tie-aware top-value recall is at least 50% and at least five percentage
  points above the frozen H6-selected-action baseline;
- mean top-action regret is at most 0.40 points and at least 0.05 points below
  the frozen H6-selected-action baseline;
- the selected checkpoint and every report pass integrity verification.

Passing authorizes only a separately preregistered fresh 32-game test corpus
and sealed checkpoint evaluation. It does not authorize promotion, gameplay,
threshold adjustment, another seed, or another architecture. Any failed gate
rejects this model without a retry or validation-driven change.

ADR 0079 preregisters that conditional test before this validation result is
known. Its indices 71,000-71,031 remain sealed unless every gate above passes.

## Required Implementation

- add a strict memory-mapped Python decoder for the counterfactual-advantage
  shard and manifest contract;
- expose grouped MLX batches with all four candidates, R12 target mean and
  standard error, exact public supply, immediate and shallow baselines, source
  selected index, game index, and turn;
- add the frozen ranker, loss, evaluator, resumable trainer, CLI entry point,
  Make targets, and focused corruption, initialization, gradient, and
  end-to-end tests;
- prove a real Apple GPU forward, backward, optimizer update, checkpoint,
  resume, and deterministic evaluation on implementation-only R12 data;
- collect and validate the exact fresh corpus, train once, publish JSON and
  Markdown, then update this ADR, the experiment registry, status, roadmap,
  and score-gap analysis.

## Implementation Evidence

The strict Python reader memory-maps the 6,676-byte group records and validates
the manifest, 160-byte header, feature/target/teacher hashes, shard checksum,
R12 contract, AAAAA configuration, game sequence, group IDs, selected index,
action hashes, parent/afterstate boundary, exact current-score deltas, shared
sample seeds, unused tails, public tile conservation, and finite targets
before exposing a batch.

The frozen MLX model includes the action-afterstate encoder, a separate
30-value public-supply projection, complete four-candidate attention, and a
zero-initialized bounded residual head. The trainer uses the shared atomic
checkpoint and exact-resume infrastructure. The evaluator reloads the selected
checkpoint, recomputes every metric, verifies it against the best pointer and
run source, and applies the frozen validation gates without opening test or
gameplay.

Six focused decoder, corruption, zero-initialization, gradient, checkpoint,
and resume tests passed. The complete Python suite passed 102 tests; Ruff and
format checks passed. The affected Rust packages passed 197 tests, strict
no-dependency Clippy, formatting, whitespace checks, and validation of the
sealed ADR 0077 dataset.

The exact implementation smoke collected train index 9,996 and validation
index 9,997 at four groups and R12: 384 complete continuations in 148.275
seconds. Rust validated both atomic shards, and MLX trained on
`Device(gpu, 0)`, checkpointed, resumed from epoch one to epoch two, reloaded
the selected checkpoint, and reproduced its metrics exactly.

The selected one-step smoke checkpoint reduced validation decision objective
from 0.5933 to 0.4474 and mean regret from 0.3542 to 0.2083. As expected for
only four implementation groups, it did not pass the substantive recall or
relative-MAE gates. This is not a validation result; it proves the frozen
pipeline is executable before expensive data collection.

## Collection Integrity Correction

The first substantive attempt completed all 128 train games, then stopped
during validation game 70,019 after 19 complete validation games. At source
turn 40, candidate 0, sample 3, continuation turn 74, an H6 internal rollout
entered a repeated mandatory four-of-a-kind replacement chain that exhausted
the drawable bag while rejected groups remained set aside.

ADR 0018 already defines that chance branch as having no legal stabilized
market and conditions expectations on successful stabilization. The sampled
H6 and counterfactual collectors had not implemented that conditioning:
instead, one impossible hidden permutation aborted the entire candidate or
corpus.

The permanent correction is deterministic rejection sampling:

- H6 retries a complete internal rollout from its original public state with
  a domain-separated hidden determinization only when the prior rollout
  returns `WildlifeBagEmpty`;
- each R12 candidate sample retries its complete terminal continuation from
  the original post-prelude public state under the same narrow condition;
- attempt zero remains the exact registered sample seed;
- all non-market errors still fail immediately;
- the teacher manifest carries the explicit versioned conditioning contract,
  and the strict Python reader rejects data without it.

The unconditioned 128-game train corpus and 19-game partial validation corpus
were archived under
`artifacts/datasets/invalidated/adr-0078-pre-conditioned-market-20260613/`.
They are prohibited from training, validation, selection, or augmentation.
No substantive target metric was computed and no model was trained from
them.

The exact formerly failing game 70,019 was then recollected at the full
16-group R12 shape. It completed 768 continuations in 225.045 seconds; Rust
and the strict Python memory-mapped reader both validated the resulting
manifest and shard. The corrected implementation smoke recollected indices
9,996 and 9,997, completed 384 continuations, and repeated the same Apple-GPU
checkpoint/resume/evaluation proof. Its decision metrics remain identical
because neither smoke game encountered an impossible branch.

## Distributed Local Execution

The corrected substantive recollection executed in parallel without changing
the frozen statistical protocol:

- john1 owns train indices 69,000-69,127;
- john2 owns disjoint validation indices 70,000-70,031;
- john3 is reserved for the one frozen MLX training run after both checksummed
  datasets validate.

john2 uses the exact release collector checksum from john1. Both worker
checkouts report Git revision `a9918946f66c237a803b23ea299c6a514785ae52`.
john3 ran CPython 3.12.13 and MLX 0.31.2 on `Device(gpu, 0)` and passed all
six focused counterfactual decoder/model tests before receiving statistical
data. The commands, source and executable identities, node allocation,
handoff incident, and completion checklist are recorded in
`docs/v2/reports/adr-0078-distributed-execution.md`.

## Validation Result

The corrected corpus completed exactly as registered:

- train: 128 games, 2,048 groups, 8,192 candidates, and 98,304 continuations;
- validation: 32 games, 512 groups, 2,048 candidates, and 24,576
  continuations;
- train manifest BLAKE3:
  `c4102f016feecb103d2924656a82404fbb27b9f54db39dda66e46ff8ba3737da`;
- validation manifest BLAKE3:
  `02d3603a8ac9e36cd03ba60c160b18296b722d687c72221b2df95d2467bfd392`.

The one authorized MLX run completed 640 optimizer steps over five epochs in
16.318 seconds on `Device(gpu, 0)`, then stopped at the frozen patience limit.
The trained final epoch improved centered MAE from 0.6432 to 0.6002,
correlation from 0.7496 to 0.7881, and pairwise accuracy from 52.48% to
74.64%. It simultaneously worsened the checkpoint-selection objective from
0.7035 to 0.7168, top-value recall from 44.92% to 45.70%, and mean regret from
0.4927 to 0.5199.

The untouched zero-output checkpoint therefore remained the exact selected
checkpoint. It failed six frozen gates:

- validation objective did not improve by at least 10%;
- centered MAE did not improve by at least 10%;
- top-value recall was below 50%;
- top-value recall was not at least five points above H6's 48.24%;
- mean regret exceeded 0.40;
- mean regret was not at least 0.05 below H6's 0.4390.

The selected checkpoint passed GPU, absolute MAE, correlation, source,
checkpoint, and report-integrity gates. The complete machine-readable result
is
`docs/v2/reports/r12-counterfactual-advantage-set-ranker-v1-validation.json`.
Its BLAKE3 is
`c74729dc03e714a2d8c889ec06b7d1ddbc0c6b10c60d4e0ee65439e37ed1cf77`.

This rejects the model without retry. The target remains observable and the
trained network learned broad ordering, but its corrections did not identify
the best action reliably enough to improve the registered decision objective
or H6-relative regret. ADR 0079 and ADR 0080 closed unopened.

## Maximum Compute

The authorized compute is exhausted: two one-game four-group R12
implementation datasets, one corrected 128-game R12 train collection, one
corrected 32-game R12 validation collection, and one frozen MLX run stopped by
patience after five epochs. The invalidated pre-training collection and the
unregistered john1 strict-prefix mirror are retained only as failure evidence.
No external compute, test access, gameplay, model retry, seed sweep,
hyperparameter sweep, warm start, architecture change, threshold change,
candidate change, or extra statistical game is authorized.
