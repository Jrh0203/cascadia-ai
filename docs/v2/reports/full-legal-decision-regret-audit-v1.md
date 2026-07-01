# Full-Legal Decision Regret Audit

- Status: **gate_failed**
- Coverage: 13 games, 1040 decisions, 3,872,079 canonical actions
- Champion corpus mean: 95.346 base points
- Mean champion decision regret: 0.350 [0.316, 0.386] points
- Top-64 recall: 89.904% [88.077, 91.635]
- Top-64 plus champion-frontier union recall: 99.327% [98.846, 99.712]
- Rank-stratified sentinel winner rate: 0.673% [0.288, 1.154]
- Retained top-64 regret: 0.047 [0.036, 0.060] points
- Git revision/status context variants: 2/3. These are recorded context; executable, model, and source-root digests define frozen identity.
- Smallest observed screen width reaching 98% recall: 1024
- Maximum observed high-confidence winner screen rank: 2448

## Screen Recall Curve

| Width | Recall | Game-block bootstrap 95% CI |
|---:|---:|---:|
| 64 | 89.904% | [88.077%, 91.635%] |
| 128 | 94.038% | [92.596%, 95.481%] |
| 256 | 96.250% | [95.000%, 97.500%] |
| 512 | 97.788% | [96.731%, 98.750%] |
| 1,024 | 99.038% | [98.269%, 99.712%] |
| 2,048 | 99.712% | [99.423%, 100.000%] |

## Error Decomposition

| Component | Mean | Game-block bootstrap 95% CI |
|---|---:|---:|
| Champion frontier / proposal | 0.254 | [0.222, 0.288] |
| Within-frontier selection | 0.095 | [0.084, 0.105] |
| Complete-screen top-64 truncation | 0.047 | [0.036, 0.060] |

- Dominant source: `proposal_frontier_regret`
- Dominance supported by non-overlapping game-block intervals: `True`
- First-order 20-turn local headroom: 6.995 points. This is diagnostic, not an online-oracle score claim.

## Nature Tokens

- Token-bearing decisions: 850
- Expected paid-wipe gain over stopping: -0.597 [-0.677, -0.504] points
- Paid wipe preferred probability: 23.603% [19.649, 27.877]
- Positive expected-gain states: 100

## Hindsight Diagnostic

- Public winner hindsight regret: 2.077 [1.359, 2.897]
- Champion hindsight regret: 2.821 [2.026, 3.718]
- Public winner matches realized winner: 35.897% [20.513, 51.282]

## Substantive Gates

| Gate | Passed |
|---|---:|
| `all_games_complete` | `True` |
| `all_decisions_complete` | `True` |
| `every_action_screened` | `True` |
| `single_frozen_executable_model_source_set` | `True` |
| `top64_recall_at_least_98_percent` | `False` |
| `retained_top64_mean_regret_at_most_0_15` | `True` |
| `paid_wipe_diagnostic_complete` | `True` |
| `realized_hidden_diagnostic_complete` | `True` |
| `dominant_error_source_supported_by_game_block_ci` | `True` |

## Host Reproduction

| Host | Decisions | Champion regret | Top-64 recall | Retained regret | Union recall |
|---|---:|---:|---:|---:|---:|
| john1 | 400 | 0.318 | 88.750% | 0.055 | 99.000% |
| john2 | 320 | 0.339 | 91.562% | 0.038 | 99.688% |
| john3 | 320 | 0.400 | 89.688% | 0.046 | 99.375% |

## Host Utilization

| Host | Games | Productive wall | Process wall | Peak RSS | Process swaps | System swap delta |
|---|---:|---:|---:|---:|---:|---:|
| john1 | 5 | 4078.2s | 4082.0s | 1060.9 MiB | 0 | +1179.9 MiB |
| john2 | 4 | 2858.4s | 2860.8s | 1113.9 MiB | 0 | +0.0 MiB |
| john3 | 4 | 2701.9s | 2704.5s | 1130.1 MiB | 0 | +0.0 MiB |

## Highest-Regret Decisions

| Seed | Turn | Phase | Regret | Frontier | Rank | Change | Wildlife |
|---:|---:|---|---:|---:|---:|---|---|
| 61009 | 4 | early | 3.272 | 3.272 | 30 | draft_choice | Bear |
| 61006 | 6 | early | 3.162 | 2.124 | 3 | draft_choice | Bear |
| 61001 | 40 | middle | 2.761 | 2.733 | 12 | tile_placement | Bear |
| 61009 | 8 | early | 2.563 | 2.563 | 3 | draft_choice | Bear |
| 61010 | 15 | early | 2.504 | 2.504 | 14 | draft_choice | Bear |
| 61000 | 5 | early | 2.475 | 2.475 | 1 | draft_choice | Bear |
| 61010 | 30 | middle | 2.472 | 1.972 | 24 | draft_choice | Fox |
| 61009 | 2 | early | 2.471 | 2.471 | 1 | draft_choice | Bear |
| 61009 | 6 | early | 2.468 | 2.468 | 1 | draft_choice | Bear |
| 61000 | 12 | early | 2.459 | 2.194 | 54 | draft_choice | Bear |
| 61000 | 9 | early | 2.350 | 2.182 | 57 | draft_choice | Bear |
| 61008 | 7 | early | 2.093 | 2.093 | 16 | draft_choice | Bear |
| 61012 | 23 | early | 2.084 | 2.084 | 30 | draft_choice | Bear |
| 61002 | 37 | middle | 2.057 | 0.000 | 5 | draft_choice | Fox |
| 61001 | 31 | middle | 2.053 | 2.053 | 8 | draft_choice | Bear |
