# ADR 0079: R12 Set-Ranker Sealed Test

Status: closed unopened on 2026-06-14. ADR 0078 failed six frozen validation
gates, so no test authorization was written and test indices 71,000-71,031
were never collected or evaluated. Gameplay and promotion remain closed.

## Context

ADR 0077 qualified the R12 rank-stratified counterfactual estimator. ADR 0078
then froze one fresh train corpus, one fresh validation corpus, and one MLX
complete-candidate-set ranker. ADR 0078 explicitly permits only a separately
preregistered fresh 32-game test after a complete validation pass.

This ADR fixes that confirmation before any ADR 0078 validation metric is
known. It does not assume the model will pass. If any ADR 0078 validation gate
fails, no test record may be collected or inspected and this ADR closes
unopened.

## Frozen Test Corpus

- Four-player symmetric AAAAA with habitat bonuses excluded.
- Split: `test`.
- Game indices: 71,000-71,031, exactly 32 games.
- Output:
  `artifacts/datasets/r12-counterfactual-advantage-v1-test-32`.
- Teacher: unchanged H6 K8/H6/R4/D4.
- Candidate selection: selected, highest, median, and lowest remaining ranked
  H6 action.
- Sixteen evenly spaced groups per game.
- Four candidates per group.
- Twelve shared ordered public-redetermination samples per candidate.
- Stable-market conditioning:
  `reject-unstable-market-trajectories-v1`.
- Exact public post-prelude boundary, public supply, opponent boards, action
  afterstates, immediate scores, shallow H6 context, raw decomposed returns,
  sample seeds, hashes, and checksums are unchanged from ADR 0078.

The exact collector executable must have BLAKE3
`183192792323090bac31de9ba8e4327ae466cb066f844447ef6a8c696fc122d1`.
The collector command is:

```bash
target/release/cascadia-v2 collect-counterfactual-advantage \
  --output artifacts/datasets/r12-counterfactual-advantage-v1-test-32 \
  --games 32 \
  --first-game-index 71000 \
  --split test \
  --groups-per-game 16 \
  --samples-per-candidate 12 \
  --candidate-selection stratified \
  --resume
```

No train, validation, audit, implementation-smoke, historical, or gameplay
record may be added to this corpus. Collection must use atomic one-game shards,
resume only missing games, and validate on the producing host and again on
john1 after transfer.

## Frozen Checkpoint

Evaluate exactly the ADR 0078 `best.json` checkpoint selected before test
collection. Its run manifest, checkpoint manifest, model configuration,
optimizer history, validation report, source digest, and tensor checksums must
remain unchanged.

No retraining, resume after completed training, threshold adjustment,
calibration, blending, ensembling, warm start, seed change, architecture
change, or checkpoint reselection is permitted. The test evaluator may add
only split handling and reporting code outside the frozen model/training
source identity.

## Frozen Evaluator

`tools/adr0079_counterfactual_advantage_test.py` is the only authorized test
evaluator. Its BLAKE3 is
`409384795cffd4c9538cd9af498789e823a2832421bb1d10582710e733df0519`.

Before the first test manifest or shard can exist, the cluster supervisor must
write `test-authorization.json` containing the passing validation-report
SHA-256, selected checkpoint identity, authorization timestamp, collector
identity, frozen test indices, and proof that the test path was absent on
john1, john2, and john3. The evaluator rejects a test manifest whose creation
precedes that authorization.

The evaluator:

- loads and integrity-checks the exact ADR 0078 `best` checkpoint;
- constructs the untouched zero-output model for the exact-immediate
  comparison on the same test groups;
- evaluates the selected checkpoint once on the complete test split;
- recomputes the ADR 0078 validation report on john3 and requires exact JSON
  equality with the pre-test report;
- verifies checkpoint, source, authorization, dataset, evaluator, and device
  identity;
- writes complete JSON and Markdown reports without opening gameplay.

Focused tests freeze the collector command and module ownership limits, and
cover pre-authorization data
rejection, authorization ordering, checkpoint drift, validation replay drift,
resume accounting, and stalled-process handling.

## Frozen Test Gates

The checkpoint passes test only if every condition holds:

- ADR 0078 passed every validation gate before the first test record existed;
- all test schema, header, checksum, provenance, sequence, action-identity,
  public-supply, shared-seed, unused-tail, finite-target, run, best-pointer,
  checkpoint, and tensor integrity checks pass;
- the evaluator runs on MLX `Device(gpu, 0)`;
- the evaluated checkpoint is byte-identical to ADR 0078's selected
  checkpoint;
- test decision objective is at least 10% below the untouched exact-immediate
  initialization;
- centered MAE is at most 0.75 points and at least 10% below initialization;
- centered-advantage correlation is at least 0.55;
- tie-aware top-value recall is at least 50% and at least five percentage
  points above the frozen H6-selected-action baseline on the test corpus;
- mean top-action regret is at most 0.40 points and at least 0.05 points below
  the frozen H6-selected-action baseline on the test corpus;
- recomputing the ADR 0078 validation report after test evaluation remains
  bit-exact to the preregistered validation report.

Report all model, exact-immediate, shallow-H6, and selected-H6 metrics even
when a gate fails. Test metrics may not change any threshold or model.

## Consequences

A failed test rejects this model permanently. No retry, alternate test corpus,
post-hoc calibration, gameplay integration, or promotion is authorized.

A complete pass confirms offline generalization only. It authorizes a
separate ADR for canonical Rust/MLX inference integration, latency
qualification, and a fresh paired gameplay protocol. It does not itself
authorize gameplay, product exposure, promotion, a 100-point claim, or final
validation.

ADR 0078 did not reach this stage. After its validation failure, the
supervisor verified that the test dataset path was absent on john1, john2, and
john3 and that `test-authorization.json` did not exist. This ADR therefore
closed without spending a test record, loading the checkpoint against test
data, or opening gameplay.

## Maximum Compute

No test compute was consumed. The conditional 32-game collection, checkpoint
evaluation, and validation replay are permanently unauthorized because ADR
0078 failed validation. No external compute, second model run, test retry,
duplicate corpus, extra statistical game, gameplay seed, or promotion action
is authorized.
