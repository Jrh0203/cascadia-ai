# Conditional Tile Successor Forensics V1 Result

Date: 2026-06-16

Experiment ID: `conditional-tile-successor-forensics-v1`

Pipeline: **passed**

## Decisions

| Arm | Classification | Key evidence |
|---|---|---|
| Sampling mass | `uniform_query_sampling_not_explanatory` | No width stratum passed the frozen mismatch gate |
| Score scale | `cross_stage_score_scale_mismatch` | dispersion ratios 5.64x train and 9.42x validation |
| Factor selector | `complete_action_selector_required` | `rank_log_sum` retained 74.72% targets and 100.00% winners |

## Sampling Evidence

| Width | Train miss share / query share | Validation miss share / query share |
|---|---:|---:|
| 33-64 | 1.91x | 2.03x |
| 65-96 | 1.58x | 1.43x |
| 97-128 | 1.37x | 1.38x |
| 129+ | 0.90x | 1.03x |

The `65-96` stratum came closest, but its validation ratio was
1.43x, below the frozen 1.50x threshold. Uniform query
sampling is therefore not the selected explanation.

## Selector Evidence

The train-selected fixed method was `rank_log_sum`. It retained
100.00% of validation winners with 0.000000 mean regret,
but only 74.72% of the validation target set. Even oracle factor
retrieval therefore cannot satisfy the top-64 target-recall gate through a
fixed factor-rank aggregation. A complete-action selector is required.

The stage logits are also materially incomparable: median query standard
deviation differs by 5.64x on train and
9.42x on validation. Any learned complete-action
selector must normalize or learn stage-specific calibration rather than sum
raw logits.

## Mechanical Successors

- If ADR 0118 is insufficient: run one frozen optimizer-schedule treatment,
  not target-mass resampling and not another uniform epoch extension.
- If ADR 0118 is sufficient: train a normalized complete-action top-64 selector;
  fixed factor aggregation is closed.

## Cluster Throughput

The three independent decisions completed on john1, john3, and john4 in
31.91 seconds of wall time and
43.01 scheduled process-seconds. Decision
throughput was 338.5 per wall hour with zero
duplicate discovery compute. john2 continued the sole ADR 0118 origin
throughout.

Sealed test, gameplay, new teacher compute, cloud, and external compute
remained closed.
