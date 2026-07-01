# ADR 0119: Conditional Tile Successor Forensics

Status: complete

Date: 2026-06-16

Experiment ID: `conditional-tile-successor-forensics-v1`

## Context

ADR 0118 is a frozen 200-epoch origin on john2. It changes only full-cache
exposure and must finish before its model-strength result is interpreted. The
remaining three Macs must not duplicate that origin or wait without advancing
independent evidence.

This portfolio uses only the open ADR 0115 caches and frozen ADR 0115/0116
weights. It performs no training and cannot override ADR 0118. Its purpose is
to distinguish three successor mechanisms before the origin finishes:

1. uniform query sampling underweights strata containing disproportionate
   target and miss mass;
2. raw draft, tile, and wildlife scores are not comparable across stages; or
3. even oracle factor retrieval needs a complete-action selector rather than a
   fixed factor-rank aggregation.

Sealed test, gameplay, new teacher compute, cloud, and external compute remain
closed.

## Frozen Arms

### Sampling Mass, john3

Replay the ADR 0116 tile checkpoint on the full open train and validation
caches. Partition tile queries into widths `<=32`, `33-64`, `65-96`, `97-128`,
and `129+`. Report query, target, and missed-target shares.

Classify `target_mass_sampling_mismatch` only when the same width stratum has:

- at least 10% of missed targets on train and validation;
- missed-target share at least 1.5 times query share on both splits; and
- target share at least 1.25 times query share on train.

Otherwise classify `uniform_query_sampling_not_explanatory`.

### Score Scale, john4

Replay the frozen draft, ADR 0116 tile, and wildlife checkpoints. For every
query, report score mean, standard deviation, top score, and range by stage.

Classify `cross_stage_score_scale_mismatch` only when both median query
standard-deviation ratio and median query-range ratio are at least 4x across
stages on both train and validation. Otherwise classify
`cross_stage_score_scale_not_dominant`.

### Factor Selector Ceiling, john1

Use oracle factor targets and oracle factor expected ranks to construct the
ADR 0114 proposal. Compare exactly three fixed top-64 selectors:

- sum of negative `log1p` factor ranks;
- sum of within-query factor-rank percentiles; and
- the worst within-query factor-rank percentile.

Select one method using train target recall, winner retention, regret, and
exact-set recovery in that order. Evaluate validation once. Apply the complete
ADR 0115 selector gates, including phase and Nature Token/independent-draft
subsets.

Classify `fixed_factor_selector_sufficient` only if the selected method passes
all train and validation gates. Otherwise classify
`complete_action_selector_required`.

## Pipeline Gates

Every arm must preserve cache identities, score each required query/item or
group/action exactly once, remain finite, write an atomic report, and keep all
closed domains closed. Any violation invalidates only that arm.

## Decision Use

- If ADR 0118 is insufficient and sampling mismatch passes, the one authorized
  successor is a frozen target-mass-aware sampling treatment.
- If ADR 0118 is insufficient and sampling mismatch fails, the successor must
  be an optimizer-schedule treatment rather than another uniform epoch budget.
- If ADR 0118 is sufficient, the factor-selector result determines whether a
  fixed normalized selector is viable or a learned complete-action selector is
  required.
- Score-scale evidence determines whether stage normalization belongs in that
  selector design; it does not authorize a model by itself.

## Cluster Execution

john2 continues the sole ADR 0118 origin. john1, john3, and john4 execute the
three distinct arms concurrently. There is no duplicate discovery compute.

## Maximum Compute

One full open-data pass per arm, focused and full tests, one combined report,
and documentation. No training, second seed, teacher rollout, sealed test,
gameplay, cloud, Modal, or external compute.

## Result

Every pipeline, identity, coverage, numerical, resource, and closed-domain gate
passed.

- john3 classified `uniform_query_sampling_not_explanatory`. The `65-96`
  stratum came closest, but its validation missed-target/query-share ratio was
  1.43x, below the frozen 1.50x threshold.
- john4 classified `cross_stage_score_scale_mismatch`. Median query score
  dispersion differed by 5.64x on train and 9.42x on validation; median score
  range differed by 9.60x and 15.59x.
- john1 classified `complete_action_selector_required`. The train-selected
  oracle-factor `rank_log_sum` selector retained every validation winner with
  zero mean regret, but only 74.72% of the validation target set.

The mechanical successor is now fully determined on either ADR 0118 branch:

- if ADR 0118 is insufficient, run one optimizer-schedule treatment; uniform
  epoch extension and target-mass resampling are closed;
- if ADR 0118 is sufficient, train a normalized complete-action top-64
  selector; fixed factor aggregation is closed.

The three independent decisions completed on john1, john3, and john4 in 31.91
seconds of wall time while john2 continued the sole ADR 0118 origin. Scheduled
process time was 43.01 seconds, duplicate discovery compute was zero, and
decision throughput was 338.5 per wall hour.

Combined scientific BLAKE3:
`c569aff575ab5e300335f35bca2108d9d7a22a679ede8d6b8a80a888cbfeb7eb`.
