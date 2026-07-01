# O1 Opponent Intent MLX Factorial v1 Preregistration

Experiment: `o1-opponent-intent-mlx-factorial-v1`

ADR: `0187-o1-opponent-intent-mlx-factorial.md`

Status: frozen before production optimization

## Hypothesis

Recent public drafts expose stable opponent demand that improves calibrated
prediction of future market access on policy families absent from training.
Ordered next-draft supervision should sharpen that representation, and explicit
opponent intent tokens should improve the mapping from demand to each current
market tile's survival.

## Frozen Corpus

Train:

- `train-part-0`: 512 games, 38,912 windows;
- `train-part-1`: 512 games, 38,912 windows;
- policy support: Greedy, PatternAware, PatternCommitment only.

Open validation:

- 256 games, 19,456 windows;
- every game contains held-out PatternCompetition.

Sealed test:

- 256 games, 19,456 windows;
- every game contains held-out PatternPortfolio;
- may be opened exactly once and only after validation selects a treatment.

Descriptive final stress:

- 128 games, 9,728 windows;
- every game contains held-out Random;
- cannot select a checkpoint or determine pass/fail.

The corpus classification is
`policy_held_out_draft_survival_corpus_passed`, classification BLAKE3
`3ff49a842b3b070938808ecb7d740ceb22e5ec2e6091ad7610853f2bff778bd1`.

## Frozen Inputs

Each window uses:

- four compact public boards, rotated into focal-seat order;
- exact public market;
- public turn phase, nature tokens, wildlife counts, habitat sizes, and cards;
- up to 12 public actions with age and relative seat.

The decoder materializes at most 23 placed tiles per board. It does not
materialize a 441-cell or 121-cell dense board.

Forbidden model inputs:

- game and dataset identity;
- policy identity;
- physical tile identity;
- final score;
- future opponent actions;
- survival labels;
- hidden supply order or random state.

## Frozen Targets

Primary:

- four initial market slots;
- four disposition classes per slot: opponent one, two, three, or survives.

Shared secondary survival targets:

- whether the original wildlife pairing survives, conditioned on tile survival;
- final market slot, conditioned on tile survival.

Authorized next-draft auxiliaries, ordered by relative opponent:

- tile slot;
- wildlife slot;
- paired versus independent draft;
- drafted wildlife species;
- free three-of-a-kind replacement.

Paid wildlife wipes and strategy switches have no positive support and are not
targets.

## Frozen Graph

- architecture: `compact-opponent-intent-survival-v1`;
- hidden width 64;
- four attention heads;
- one board block;
- one market block;
- two history blocks;
- feed-forward multiplier 2;
- 374,171 parameters;
- identical layout and initial tensor across all arms.

Arm gates:

| Arm | History | Next-draft loss | Intent routed to survival |
|---|---:|---:|---:|
| A0 public state | 0 | 0 | 0 |
| A1 recent history | 1 | 0 | 0 |
| A2 next-draft auxiliary | 1 | 1 | 0 |
| A3 joint intent survival | 1 | 1 | 1 |

Loss:

- disposition cross entropy: weight 1.00;
- pair survival among survivors: weight 0.25;
- final slot among survivors: weight 0.10;
- mean next-draft auxiliary cross entropy: weight 0.25 when enabled.

## Frozen Optimization

- MLX GPU;
- seed `2026061704`;
- 5,120 steps;
- deterministic shard-first batch schedule;
- batch size 128, with exact smaller shard-tail batches;
- 622,592 train examples, exactly eight corpus passes;
- AdamW;
- learning rate `3e-4`;
- weight decay `1e-4`;
- checkpoint every 640 steps;
- metric event every 100 steps;
- fixed final checkpoint;
- no validation during training;
- no early stopping.

## Frozen Metrics

Primary:

- multiclass disposition Brier score.

Guardrails:

- disposition negative log likelihood;
- accuracy and macro F1;
- 15-bin top-label ECE;
- binary survival Brier, NLL, ECE, and AUROC;
- pair-survival and final-slot metrics among survivors;
- opening, early-middle, late-middle, and endgame slices.

Auxiliary:

- NLL, multiclass Brier, accuracy, macro F1, and ECE for every next-draft head;
- relative NLL gain versus Laplace-smoothed train-frequency priors.

Inference:

- paired treatment-minus-control Brier per game;
- 20,000 game-level bootstrap replicates;
- seed `2026061705`.

## Frozen Validation Classification

All eight primary/replay reports, models, and prediction-evidence files are
required. Any mismatch in authorization, data, parameter layout,
initialization, final tensors, serialized model bytes, prediction arrays, or
role-neutral scientific identity makes every treatment ineligible.

Eligibility thresholds and selector order are exactly those in ADR 0187.

Classification:

- `opponent_intent_validation_arm_selected`; or
- `opponent_intent_validation_factorial_null`.

## Frozen Sealed-Test Classification

If validation selects a treatment, compare selected versus A0 once on
PatternPortfolio using the frozen test thresholds in ADR 0187. Then evaluate
Random as descriptive stress.

Terminal classification:

- `opponent_intent_policy_holdout_replication_passed`;
- `opponent_intent_policy_holdout_replication_failed`; or
- `opponent_intent_test_not_opened`.

## Claim Boundary

A pass authorizes a separately preregistered high-regret draft-ranking
integration. It does not establish:

- paid-wipe intent;
- strategy switching;
- v1 champion or learned-policy transfer;
- gameplay strength;
- a score gain;
- progress toward 100.
