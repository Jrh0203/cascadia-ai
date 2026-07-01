# O2 Exact Opportunity Matching v1 Preregistration

**Frozen:** 2026-06-17, before production export or treatment analysis  
**Experiment:** `o2-exact-opportunity-matching-v1`  
**Protocol:** `o2-strict-train-top64-foundation-identifiability-v1`  
**Host:** john1 only

## Question

Does exact public demand-supply assignment contain incremental action-ordering
signal beyond the frozen direct exact-R2 ranker?

## Domain And Claim Boundary

The experiment uses only the existing strict open-train cohort:

- 7 games: raw seeds 61000, 61001, 61002, 61005, 61006, 61009, 61010;
- 560 unfiltered decisions;
- 64 exact-R2 roots per decision;
- 35,840 candidate public afterstates;
- the existing partial R4800 labels already frozen by the graded-oracle
  campaign, without generating or imputing new targets; and
- cohort ID
  `aac7a480bd3f73bf15fa09b2314c8efa80cbae01a4ce09f8cf342845c2808512`.

Validation, sealed test, gameplay, score qualification, model promotion, and
progress-to-100 claims are closed. T1 search is not rerun. No cross-host replay
or artifact fanout is authorized during discovery.

## Frozen Inputs

- graded-oracle train dataset manifest BLAKE3:
  `7ed12c943d75a786ccd4ccbe11a6b0146aad4fe5ed40f0cbaf1d652f5ac0bb99`;
- strict cohort base-score tensor BLAKE3:
  `1ec068ae14c2d9e2677c12623fb9caacd5122123ff9921c0b42d5241b6e897e4`;
- strict cohort action-hash tensor BLAKE3:
  `2a638573011c690fa4e0bf35e0fa61de8cb9b5c5a5f0c5fe824da5b147789e68`;
- exact semantic supply schema v1 and catalog identity already promoted by ADR
  0143; and
- `OpportunityGraphV1` and matching semantics frozen by ADR 0192.

Every input digest is verified before export. Any mismatch makes the campaign
invalid.

## F1 Foundation Gate

Production export must prove:

- exactly 560 groups and 35,840 unique action hashes;
- replayed position bytes and public hashes match every graded-oracle group;
- every candidate action reconstructs to its frozen hash;
- every graph and summary validates and round-trips canonically;
- repeated construction is byte-identical;
- hidden redeterminization leaves the pre-action public graph unchanged;
- every group anchor passes all 12 D6 covariance checks;
- no graph or matching field is read from R600/R1200/R4800 targets;
- resource telemetry records wall time, peak RSS, swap delta, rows/s, graph
  sizes, and zero-swap status; and
- all outputs stay under
  `/Volumes/John_1/cascadia-cluster/john1/`.

Any integrity, target-boundary, D6, codec, accounting, or swap failure makes
the experiment invalid. A valid foundation publishes one compact immutable
contract/fixture bundle; it does not publish the row cache.

## Frozen Features

For each candidate, B1 receives the direct exact-R2 score plus only these
deterministic summaries, transformed as `log1p` where nonnegative:

1. demand count;
2. supply count;
3. edge count;
4. matched and unmatched demand counts;
5. wildlife and habitat match counts;
6. current-market and unseen-supply match counts;
7. exact completion value;
8. teacher value in score-point units; and
9. matched-demand, market-match, and wildlife-match fractions with zero-safe
   denominators.

No graph IDs, action hashes, seeds, group indices, teacher targets, raw R4800
values, or post-hoc features enter the treatment.

## F2 Identifiability

Fit a residual ridge probe under outer leave-one-game-out cross-fitting. For
each held-out game, standardization, feature means, and coefficients are fit
on the other six games only. Ridge lambda is selected by an inner
leave-one-game-out loop over the fixed set `{0.01, 0.1, 1, 10, 100}` using
candidate-level residual MSE, with the smaller lambda winning ties.

All bootstrap and deterministic tie domains use seed `2026061702`. Pearson
and pairwise metrics use only labeled candidates. All 20,000 bootstrap draws
resample the seven complete game blocks with replacement.

The residual target on labeled candidates is
`R4800 mean - direct exact-R2 score`. The direct score is retained with
coefficient one; the probe predicts only a correction. Unlabeled candidates
receive predictions but never enter fitting, MSE, or residual correlation.

The exact teacher is identifiable only if all are true:

- cross-fitted residual MSE improves by at least 1.0% over zero correction;
- candidate-level cross-fitted Pearson residual correlation is at least 0.05;
- a 20,000-replicate game-block bootstrap 95% interval for that correlation is
  wholly above zero; and
- at least 50% of groups have nonzero within-group variance in one or more
  matching features.

Failure stops learned O2 arms. Passing F2 may authorize learned O2 even if the
compact B1 selector itself fails, because it establishes incremental signal.

## B0 And B1 Decision Treatment

- **B0:** select the highest frozen direct exact-R2 score, with action hash as
  deterministic tie break.
- **B1:** select the highest cross-fitted direct-plus-residual score, with the
  same action-hash tie break.

Primary metric is retained R4800 regret. For a labeled selection it is the
group-best labeled R4800 mean minus the selected candidate's R4800 mean. For
an unlabeled selection it is conservatively the group-best labeled mean minus
the group-worst labeled mean. This is exactly the frozen T1 missing-label
convention and is applied identically to B0 and B1. Every group must contain
at least one labeled candidate.

B1 is locally material and confirmation-eligible only if all are true:

- mean paired regret improves by at least 0.05;
- a 20,000-replicate game-block bootstrap 95% interval for B1-minus-B0 regret
  is wholly below zero;
- top-1 R4800 recall regresses by no more than 0.01 absolute;
- pairwise R4800 ordering accuracy regresses by no more than 0.005 absolute;
- no protected slice worsens mean regret by more than 0.05; and
- F1 and F2 pass.

## Protected Slices

Slices are fixed before output inspection:

- early turns 0-6, middle 7-13, late 14-19 by personal turn;
- active board has zero versus at least one Nature Token;
- B0 selected action is paired versus independent;
- R4800-best drafted wildlife: Bear, Elk, Salmon, Hawk, or Fox;
- scarce supply: median candidate matching has unseen-match fraction below the
  corpus median computed on the six outer training games only; and
- high competition: median candidate teacher exposure denominator is above
  the outer-training median.

Slices with fewer than 20 held-out groups are reported but not used as a
noninferiority gate. Definitions cannot be changed after production output is
read.

## Terminal Classifications

- `o2_exact_foundation_invalid`: F1 or accounting fails.
- `o2_exact_teacher_unidentifiable`: F1 passes and F2 fails.
- `o2_exact_teacher_signal_b1_null`: F1/F2 pass but B1 fails a material or
  protected-slice gate.
- `o2_exact_teacher_b1_confirmation_eligible`: F1/F2 and every B1 gate pass.

Only the last class permits one later different-host confirmation. Other Macs
continue independent discovery regardless. No all-host synchronization is
authorized by this preregistration.

## Pre-output Correction

The first one-group engineering smoke stopped before writing a group result
when it encountered an unlabeled R4800 candidate. The strict top-64 cohort is
complete for exact-R2 rescoring but intentionally partial for R4800. The
language above was corrected before any graph, matching, treatment, or metric
output was inspected. The correction freezes the already-established T1
conservative missing-label convention; it does not change the cohort, create
targets, or inspect outcomes.
