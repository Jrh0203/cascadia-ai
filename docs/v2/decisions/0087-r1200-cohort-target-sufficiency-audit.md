# ADR 0087: R1200 Cohort Target Sufficiency Audit

Status: closed rejected before training.

Date: 2026-06-16

Experiment ID: `complete-action-r1200-target-sufficiency-v1`

## Context

ADR 0086 showed that finite-sample R4800 argmaxes are noisy but that target
ambiguity does not explain ADR 0081's failure. The selected model covered an
R4800 95% confidence-equivalent action in only 86.25% of validation decisions
and recalled only 85.53% of statistically distinguishable winners.

ADR 0081 optimized a hard R4800-winner cross-entropy over every complete legal
action even though dense substantial-budget supervision exists only on the
K1024 cohort. The selected checkpoint learned a mean +3.052 residual on
629,837 validation screen-only actions despite their regularizer. That creates
a specific alternative mechanism: the complete-group objective asks one noisy
positive to outrank thousands of weakly supervised actions, instead of asking
the model to compress the already qualified K1024 cohort to a robust top 64.

This no-training audit tests whether the frozen R1200 cohort contains enough
ordering information to support that compression. It does not authorize a
model change by itself.

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
- Dataset, source, model, and observable-prior identities are identical to
  ADR 0086.

The audit must reject any changed identity, non-finite estimate, missing
action, or split other than train or validation.

## Frozen Method

For each decision:

1. construct the R4800 pairwise 95% confidence set exactly as ADR 0086:
   `winner_mean - action_mean <= 1.959963984540054 *
   hypot(winner_se, action_se)`;
2. construct the analogous R1200 pairwise 95% confidence set over every
   R1200-labeled action;
3. rank the complete action set by the historical screen and selected ADR
   0081 model;
4. construct one non-deployable **R1200 cohort oracle** by placing every
   R1200-labeled action before every unlabeled action, sorting the labeled
   cohort by descending R1200 mean and canonical action hash, then sorting the
   remainder by descending screen value and hash;
5. for top 1, 8, 32, and 64, report exact R4800-winner recall, R4800
   confidence-set coverage, distinguishable-winner recall, and retained R4800
   regret for all three rankings;
6. report whether the R1200 and R4800 95% confidence sets intersect, whether
   the R1200 winner lies in the R4800 set, and their set sizes;
7. report how many model top-64 actions have R600, R1200, and R4800 labels and
   how many are screen-only;
8. repeat primary metrics by early, middle, and late phase, Nature-Token
   availability, independent-draft winner, and source game.

The cohort oracle is an information upper bound, not a deployable player and
not a score claim. Its R1200 mask may not become an inference-time input.
Future deployment may reproduce the cohort only by running the same
observable complete-action K1024 screen that already passed Phase 2 recall.

## Frozen Interpretation Gates

Classify the existing cohort and target as **sufficient for a focused
set-valued proposer** only if validation satisfies all of:

- R1200 cohort-oracle top-64 R4800 confidence-set coverage is at least 99%;
- top-64 distinguishable R4800-winner recall is at least 98%;
- top-64 exact R4800-winner recall is at least 95%;
- top-64 retained mean R4800 regret is below 0.03 points;
- every phase has at least 98% R4800 confidence-set coverage;
- the R1200 and R4800 95% confidence sets intersect in at least 95% of
  decisions;
- every decision has at least 64 R1200-labeled actions or fewer than 64 total
  legal actions;
- all identity, complete-action, finite-value, and sealed-boundary checks
  pass.

If every gate passes, the next experiment must keep the ADR 0081 observable
architecture fixed and change only the proposal domain and objective:

- run the frozen complete-action screen;
- rerank only its R1200/K1024 cohort;
- replace hard single-winner cross-entropy with confidence-set probability
  mass and statistically separated ordering losses;
- select checkpoints on validation confidence-set coverage first, then
  retained regret and distinguishable-winner recall;
- keep test and gameplay closed until the new validation gates pass.

If any substantive gate fails, do not train that proposer. Revisit teacher
allocation or observable representation instead.

## Cluster Execution

Run one train audit on john1 and the immutable validation audit independently
on john1, john2, and john3. All three validation scientific outputs must be
identical. Host performance is descriptive only. No host may read the test
dataset.

## Maximum Compute

One train audit and one validation audit on john1, plus byte-identical
validation replays on john2 and john3. No training, new teacher rollout,
gameplay, test access, K2048 screen, alternate checkpoint, threshold change,
or external compute is authorized.

## Outcome

The corrected audit completed on train and validation within the frozen
compute boundary. Validation scientific output was identical on john1,
john2, and john3 with digest
`b2c5f142aa4d4a9b2ed00a721647efc694e2229caf9f404b1057aed84326e6a3`.
All source, dataset, checkpoint, complete-action, finite-value, and sealed-test
checks passed.

The R1200 cohort oracle materially improved over the selected MLX model but
did not clear its own frozen upper-bound gates:

- top-64 R4800 confidence-set coverage: 97.08%, below 99%;
- distinguishable-winner recall: 90.79%, below 98%;
- exact-winner recall: 95.42%, passing 95%;
- retained mean R4800 regret: 0.020742, passing <0.03;
- early/middle/late confidence-set coverage: 96.43% / 95.24% / 100.00%, so
  the every-phase 98% gate failed;
- R1200/R4800 confidence-set intersection: 97.08%, passing 95%.

Every selected-model top-64 action was already R1200-labeled and none was
screen-only. Therefore weakly supervised complete-action rows were not
displacing the retained top 64, despite the selected model's nonzero residual
on those rows. The decisive limitation is that R1200 ordering itself misses
too many high-confidence R4800 actions, especially early and middle.

The first implementation replay is archived under
`invalid-run-overlapping-label-denominator/`. It incorrectly treated
overlapping R600, R1200, and R4800 label counts as disjoint when computing
composition fractions. Ranking metrics were unaffected, but no artifact from
that run is used as evidence. The corrected implementation and replay retained
the unchanged preregistered method and thresholds.

The proposed fixed-architecture R1200-only proposer is rejected before
training. The next experiment must improve teacher allocation or observable
representation rather than optimize the same R1200 target harder. Sealed test,
gameplay, K2048, and large self-play remain closed.

Results:

- `docs/v2/reports/complete-action-r1200-target-sufficiency-v1.json`;
- `docs/v2/reports/complete-action-r1200-target-sufficiency-v1.md`;
- `artifacts/experiments/complete-action-r1200-target-sufficiency-v1/manifest.json`.
