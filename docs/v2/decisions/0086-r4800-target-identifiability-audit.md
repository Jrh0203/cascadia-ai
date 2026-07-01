# ADR 0086: R4800 Target Identifiability Audit

Status: closed complete; observable representation or optimization remains
material.

Date: 2026-06-16

Experiment ID: `complete-action-r4800-identifiability-v1`

## Context

ADR 0081's corrected observable-only ranker reduced validation top-64 retained
R4800 regret from 0.113024 to 0.090184, but exact R4800-winner recall moved
only from 71.67% to 73.33%. Cross-host results were bit-identical and every
performance gate passed. The model was correctly rejected before sealed test
or gameplay, but the result does not yet distinguish two mechanisms:

1. the observable model cannot recover decision-relevant action ordering; or
2. a single finite-sample R4800 argmax is an unstable target among actions
   whose expected returns are statistically indistinguishable.

That distinction determines whether the next bounded experiment should change
the representation or change the oracle and learning target. This ADR computes
the answer from already-open train and validation evidence only.

## Frozen Inputs

- Train dataset:
  `artifacts/datasets/complete-action-graded-oracle-v1-train`.
- Validation dataset:
  `artifacts/datasets/complete-action-graded-oracle-v1-validation`.
- Train games: `61000-61002,61005-61006,61009-61010`.
- Validation games: `61003,61007,61011`.
- Sealed-test games `61004,61008,61012` are prohibited.
- Selected ADR 0081 checkpoint:
  `step-000003592-epoch-0008-batch-000000`.
- Selected model BLAKE3:
  `6d2a7bb57fd905e50636a20da012f40017cc3a59c1ebde06eff20f8f974940e8`.
- Corrected observable prior schema: `observable-screen-priors-v1`.

The audit must reject a split other than `train` or `validation`, a changed
dataset manifest, checkpoint, model tensor, source manifest, action count, or
non-finite estimate.

## Frozen Method

For every complete decision:

1. identify the canonical R4800 argmax with the existing mean/hash order;
2. identify the R4800 runner-up among labeled actions;
3. compute each action's raw standard error as
   `stddev / sqrt(max(samples, 1))`;
4. compute the winner-runner-up margin and independent normal combined
   standard error;
5. report whether the winner clears a two-sided 95% normal difference test;
6. report whether its separate 95% interval is above every alternative;
7. count 68% and 95% confidence sets using pairwise combined standard error;
8. compare the R1200 and R4800 argmax on their common labeled set;
9. score every action with the frozen selected model and historical screen;
10. for top-1/8/32/64, report exact-winner recall, retained R4800 regret,
    confidence-set coverage, and distinguishable-winner recall;
11. report the regret and confidence status of every exact-winner miss;
12. repeat the primary metrics by early/middle/late phase, Nature-Token
    availability, independent-draft winner, and source game.

Normal tests are diagnostics, not calibrated guarantees. Adaptive allocation,
shared rollout structure, and common random numbers can violate independence.
The report must state that limitation and must not reinterpret confidence sets
as posterior probabilities.

## Frozen Interpretation Gates

Classify **target ambiguity as dominant** only if validation satisfies all of:

- at most 50% of R4800 winners clear the 95% normal difference test;
- mean R4800 95% confidence-set size is at least 4.0 actions;
- the selected model's top 64 contains at least one member of the R4800 95%
  confidence set in at least 98% of decisions;
- the selected model's top-64 retained mean R4800 regret remains below 0.15;
- at least 95% of selected-model exact-winner misses retain an action within
  the winner's pairwise 95% confidence set;
- no phase has confidence-set coverage below 95%;
- all train/validation identity and complete-action integrity checks pass.

Classify **observable representation or optimization as still material** if
target ambiguity does not pass and either:

- fewer than 90% of statistically distinguishable winners appear in the model
  top 64; or
- model top-64 confidence-set coverage is below 98%.

Report R1200/R4800 argmax agreement and confidence-set containment as
mechanism evidence, not as an independent pass/fail gate.

If target ambiguity is dominant, exact-winner recall is closed as the primary
learning target for future experiments. The next experiment must revise
continuation/oracle targets and retain regret-based safeguards; it may not
retroactively promote ADR 0081. If representation remains material, the next
experiment may revise model inputs or architecture but may not open test or
gameplay.

## Cluster Execution

Run the immutable validation audit independently on john1, john2, and john3.
All three must produce identical scientific metrics and model rankings. Host
performance is reported but is not a strength gate. No host may read the test
dataset.

## Maximum Compute

One train audit and one validation audit on john1, plus byte-identical
validation replays on john2 and john3. No training, new teacher rollout,
gameplay, test access, K2048 screen, alternate checkpoint, threshold change,
or external compute is authorized.

## Outcome

The authorized audit completed exactly once on train and independently on all
three Macs for validation. The validation scientific digest was identical on
john1, john2, and john3:
`b78cd39f53cfcbc97847874b61838d8c073b7e75b40fb80b43947ef12c6796ff`.
All identity, complete-action, finite-score, and sealed-boundary checks passed.

Validation found that only 31.67% of R4800 winners cleared the frozen 95%
difference diagnostic, and the mean 95% confidence set contained 3.23 actions.
R1200 and R4800 selected the same action in 66.67% of decisions; the R1200
winner remained inside the R4800 confidence set in 90.00%.

That ambiguity was not sufficient to explain the ranker's failure:

- selected-model top-64 confidence-set coverage was 86.25%, below 98%;
- distinguishable-winner recall was 85.53%, below 90%;
- only 48.44% of exact-winner misses retained a confidence-equivalent action;
- early, middle, and late confidence-set coverage was 88.10%, 77.38%, and
  94.44%, so every-phase coverage failed;
- retained mean R4800 regret remained acceptable at 0.090184.

The preregistered classification is therefore
`representation_or_optimization_material`. The next bounded experiment may
change the observable model or training objective, but may not open the
sealed test, gameplay, K2048, or a large self-play loop. Exact R4800 argmaxes
must also be treated as noisy targets rather than unquestioned labels.

The complete machine-readable and human-readable results are
`docs/v2/reports/complete-action-r4800-identifiability-v1.json` and
`docs/v2/reports/complete-action-r4800-identifiability-v1.md`. The immutable
execution manifest is
`artifacts/experiments/complete-action-r4800-identifiability-v1/manifest.json`.
