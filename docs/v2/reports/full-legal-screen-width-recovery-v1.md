# Full-Legal Screen-Width Recovery

- Status: **complete**
- Coverage: 13 games, 1040 decisions, 3,872,079 canonical actions
- Champion corpus mean: 95.346 base points
- Mean champion decision regret: 0.436 [0.400, 0.472] points
- Top-1,024 recall: 99.423% [98.942, 99.808]
- Top-1,024 plus champion-frontier union recall: 99.808% [99.519, 100.000]
- Rank-stratified sentinel winner rate: 0.192% [0.000, 0.481]
- Retained top-1,024 regret: 0.002 [0.000, 0.004] points
- Git revision/status context variants: 2/4. These are recorded context; executable, model, and source-root digests define frozen identity.
- Smallest observed screen width reaching 98% recall: 1024
- Maximum observed high-confidence winner screen rank: 3181

## Screen Recall Curve

| Width | Recall | Game-block bootstrap 95% CI |
|---:|---:|---:|
| 64 | 70.385% | [67.404%, 73.077%] |
| 128 | 82.500% | [80.288%, 84.712%] |
| 256 | 89.808% | [88.077%, 91.442%] |
| 512 | 95.192% | [94.038%, 96.250%] |
| 1,024 | 99.423% | [98.942%, 99.808%] |
| 2,048 | 99.712% | [99.423%, 100.000%] |

## Error Decomposition

| Component | Mean | Game-block bootstrap 95% CI |
|---|---:|---:|
| Champion frontier / proposal | 0.329 | [0.298, 0.361] |
| Within-frontier selection | 0.107 | [0.094, 0.120] |
| Complete-screen top-1,024 truncation | 0.002 | [0.000, 0.004] |

- Dominant source: `proposal_frontier_regret`
- Dominance supported by non-overlapping game-block intervals: `True`
- First-order 20-turn local headroom: 8.723 points. This is diagnostic, not an online-oracle score claim.

## Nature Tokens

- Token-bearing decisions: 0
- Expected paid-wipe gain over stopping: n/a points
- Paid wipe preferred probability: n/a
- Positive expected-gain states: 0

## Hindsight Diagnostic

- Public winner hindsight regret: n/a
- Champion hindsight regret: n/a
- Public winner matches realized winner: n/a

## Substantive Gates

| Gate | Passed |
|---|---:|
| `all_games_complete` | `True` |
| `all_decisions_complete` | `True` |
| `every_action_screened` | `True` |
| `single_frozen_executable_model_source_set` | `True` |
| `exact_screen_contract_pairing` | `True` |
| `screen_recall_at_least_98_percent` | `True` |
| `retained_screen_mean_regret_at_most_0_15` | `True` |
| `paid_wipe_diagnostic_complete` | `True` |
| `realized_hidden_diagnostic_complete` | `True` |
| `dominant_error_source_supported_by_game_block_ci` | `True` |

## Host Reproduction

| Host | Decisions | Champion regret | Top-1,024 recall | Retained regret | Union recall |
|---|---:|---:|---:|---:|---:|
| john1 | 400 | 0.406 | 99.500% | 0.000 | 99.750% |
| john2 | 320 | 0.422 | 99.062% | 0.005 | 99.688% |
| john3 | 320 | 0.488 | 99.688% | 0.001 | 100.000% |

## Host Utilization

| Host | Games | Productive wall | Process wall | Peak RSS | Process swaps | System swap delta | Telemetry complete |
|---|---:|---:|---:|---:|---:|---:|---:|
| john1 | 5 | 1830.1s | 1479.0s | 718.9 MiB | n/a | +0.0 MiB | `False` |
| john2 | 4 | 1459.0s | 1460.8s | 695.0 MiB | 0 | +0.0 MiB | `True` |
| john3 | 4 | 1457.1s | 1458.8s | 720.4 MiB | 0 | +0.0 MiB | `True` |

## Highest-Regret Decisions

| Seed | Turn | Phase | Regret | Frontier | Rank | Change | Wildlife |
|---:|---:|---|---:|---:|---:|---|---|
| 61004 | 44 | middle | 3.576 | 3.339 | 179 | draft_choice | Bear |
| 61006 | 6 | early | 3.162 | 3.162 | 3 | draft_choice | Bear |
| 61009 | 4 | early | 2.932 | 2.932 | 30 | draft_choice | Bear |
| 61001 | 40 | middle | 2.778 | 2.778 | 12 | tile_placement | Bear |
| 61004 | 12 | early | 2.609 | 1.955 | 26 | draft_choice | Bear |
| 61009 | 8 | early | 2.563 | 2.563 | 3 | draft_choice | Bear |
| 61006 | 17 | early | 2.516 | 2.326 | 565 | draft_choice | Bear |
| 61000 | 9 | early | 2.479 | 2.479 | 55 | draft_choice | Bear |
| 61010 | 15 | early | 2.471 | 2.471 | 14 | draft_choice | Bear |
| 61011 | 13 | early | 2.427 | 1.565 | 803 | draft_choice | Bear |
| 61009 | 6 | early | 2.402 | 2.402 | 1 | draft_choice | Bear |
| 61000 | 5 | early | 2.367 | 2.367 | 1 | draft_choice | Bear |
| 61009 | 2 | early | 2.351 | 2.351 | 1 | draft_choice | Bear |
| 61010 | 6 | early | 2.281 | 0.262 | 9 | tile_placement | Fox |
| 61010 | 30 | middle | 2.267 | 1.849 | 24 | draft_choice | Fox |
