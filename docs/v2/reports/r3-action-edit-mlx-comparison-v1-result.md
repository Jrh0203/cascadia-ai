# R3 Action-Edit MLX Comparison V1 Result

Date: 2026-06-17

ADR: 0150

Experiment: `r3-action-edit-mlx-comparison-v1`

Protocol: `r3-action-edit-mlx-matched-comparison-v1`

Status: completed

Classification: `r3_action_edit_mlx_all_treatments_degraded`

Selected representation: none

Promotion authorized: false

## Executive Result

The full exact R2 afterstate control passed every absolute quality, coverage,
latency, throughput, memory, and process-swap gate. None of the three exact
R3 local-patch plus global-edit treatments met the frozen quality
noninferiority contract, and none met a material-efficiency gate.

The control achieved:

- R4800 MAE 1.32023 and RMSE 1.74231;
- 72.50% top-64 stable-winner recall;
- 0.09812 mean top-64 retained regret;
- 97.92% confidence-set coverage;
- 86,208 fixed-chunk action scores per second; and
- 130.95 ms complete-decision P99 latency.

Radius one produced the best compact-arm ranking behavior at 74.58% recall
and 0.10339 regret, but its MAE worsened to 1.48856, its low-supply and
independent-draft recall regressed, and its confidence coverage remained
below the required 99%. It is not selected and does not advance.

This is an offline representation result. It does not measure gameplay,
promote a model, or claim progress to the 100-point mean-score target.

## Immutable Evidence

| Identity | Value |
|---|---|
| Source bundle | `d82198f2ee9b8c7ac92854842128b357d5fc87d2b315bec27ebb5e46f9b75150` |
| Authorization | `deaa2044b6a812f71f9ee638649d4313101fbaba719d52ad6e3d0b6ce3959349` |
| R3 cache | `0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156` |
| Open-data proof | `a056aceadb7f53c01dc87c8a39d95a7866bac6df93b050c45cc860de2b8b87ea` |
| Classification | `49260f87006bf9c49f145cd6de89db131ad916a9532d64b21c578201312404ae` |
| Scientific classification | `4cc7d2deef805bbc3cc4584343f61aab2311253a5462b1aca56bfe6a70a19df9` |
| Order proof | `09a35dc062792159de3ed3fe599b01d93495d68d69e64e8cbb2fa97d6ff30291` |
| Four-arm mechanism report | `462b3c1935a6eabb2853bd39c816a23fc4ef5630f843d151e0239029065725f4` |

Forward and reverse report orders produced byte-identical classification
files.

## Matched Results

Every arm completed exactly 3,000 AdamW steps over the same deterministic
group, candidate, target, and D6 schedule, then scored all 240 validation
decisions and 860,203 legal actions exactly once.

| Arm | MAE | RMSE | Top-64 recall | Top-64 regret | Confidence coverage |
|---|---:|---:|---:|---:|---:|
| Full R2 afterstate | 1.32023 | 1.74231 | 72.50% | 0.09812 | 97.92% |
| R3 radius 3 | 1.37325 | 1.80373 | 69.17% | 0.16959 | 95.42% |
| R3 radius 2 | 1.40308 | 1.86799 | 70.42% | 0.12491 | 96.67% |
| R3 radius 1 | 1.48856 | 1.94580 | 74.58% | 0.10339 | 97.92% |

All compact arms failed MAE and RMSE noninferiority. Radius one was the only
treatment to improve aggregate top-64 recall, but it still failed retained
regret by 0.00028 beyond the allowed delta, failed the 99% confidence gate,
and regressed both protected slices:

| Arm | Low-supply recall | Independent-draft recall | Middle-phase recall |
|---|---:|---:|---:|
| Full R2 afterstate | 91.23% | 80.95% | 62.96% |
| R3 radius 3 | 85.96% | 66.67% | 60.49% |
| R3 radius 2 | 82.46% | 85.71% | 59.26% |
| R3 radius 1 | 82.46% | 76.19% | 69.14% |

The radius ordering is informative. Removing more local-patch tokens improved
compact-arm ranking recall and regret, especially in the middle phase, while
value calibration worsened. The bottleneck is therefore not simply an
insufficient local radius. The independent ranker is struggling to combine
candidate edits with whole-decision context and protected supply situations.

## Four-Arm Failure Mechanism

A post-classification diagnostic scored the same 240 validation decisions
through all four final checkpoints and merged the per-decision failure
atlases. It is classifier-ineligible and does not alter the frozen verdict.

The compact-arm winner rank improved as local radius shrank:

| Pair | Smaller radius better | Larger radius better | Mean smaller-minus-larger rank |
|---|---:|---:|---:|
| Radius 3 vs radius 2 | 115 | 92 | -3.30 |
| Radius 3 vs radius 1 | 113 | 82 | -8.92 |
| Radius 2 vs radius 1 | 116 | 95 | -5.62 |

Radius one passed confidence coverage where radius three failed in seven
decisions; radius three did the reverse in only one. Radius one alone covered
the confidence set in three decisions, while radius three was never the only
compact arm to do so.

This rules against "the patch was too small" as the primary explanation.
Adding local tokens diluted ranking more often than it helped. At the same
time, all compact arms shared 50 decisions where none recalled the R4800
winner in the top 64. The remaining error is substantially common-mode:
candidate-local evidence is insufficient to resolve broad decision-set
competition, which is the exact hypothesis S4 now tests.

## Serving Performance

All arms passed the absolute 20,000 scores/s, 250 ms P99, 4 GiB active/RSS,
and zero process-swap gates. None was materially more efficient than the
control.

| Arm | Fixed-chunk scores/s | Complete P99 | Peak active | Peak RSS |
|---|---:|---:|---:|---:|
| Full R2 afterstate | 86,208 | 130.95 ms | 186.2 MB | 795.4 MB |
| R3 radius 3 | 46,090 | 154.73 ms | 445.5 MB | 2,216.1 MB |
| R3 radius 2 | 52,730 | 158.15 ms | 388.3 MB | 1,621.5 MB |
| R3 radius 1 | 56,037 | 149.52 ms | 350.4 MB | 1,459.4 MB |

The presumed compact path was slower and more memory-intensive in the
measured implementation. Variable edit-token materialization and candidate
Perceiver work outweighed the smaller local crop. The exact full-afterstate
control is already a compact sparse representation, not a duplicated
441-cell dense tensor.

## State-Space Consequence

The earlier frozen state-footprint census already rejected the historical
441-cell lattice. It also established that 121 is not the size of any
complete centered hex disk: the exact capacities are 91 cells at radius 5
and 127 cells at radius 6. Radius 5 covered every open graded-corpus occupied,
frontier, selected, and complete-candidate destination; radius 6 was
empirically lossless over 200,000 generated board observations. Both retain
an exact overflow path for adversarial legal boards.

R3 reinforces the sparse conclusion. The winning full-afterstate candidate
stream averaged 58.79 tokens, reached P99 85, and maxed at 88 tokens. No
441-cell neural surface is needed, and no silent 121-cell crop is authorized.

## Classification

The deterministic classifier resolved:

- all four absolute serving and evidence envelopes: passed;
- radius-3 quality noninferiority: failed;
- radius-2 quality noninferiority: failed;
- radius-1 quality noninferiority: failed;
- every material-efficiency treatment gate: failed;
- forward/reverse order invariance: passed; and
- gameplay, promotion, and 100-point claims withheld: passed.

The terminal verdict is:

```text
r3_action_edit_mlx_all_treatments_degraded
selected_arm = null
promotion_authorized = false
```

## Consequences

1. Keep exact sparse full R2 afterstates as the accepted independent-candidate
   baseline.
2. Do not advance radius 1, 2, or 3 to gameplay under the ADR 0150 protocol.
3. Preserve the exact R3 edit cache as a reusable diagnostic and future
   incremental-serving substrate.
4. Treat radius one's higher aggregate recall as a hypothesis signal, not a
   promotion result.
5. Run S4 as an explicitly preregistered radius-one rescue experiment:
   candidate context must show a material matched-arm effect and recover the
   full-R2 quality envelope before any compact representation can advance.
6. Keep 441 cells closed. Use exact sparse coordinates or the accepted
   recentered 91/127-cell controls with exact overflow.
7. Preserve the classifier-ineligible four-arm mechanism report as the
   diagnostic basis for S4, not as retroactive selection evidence.
