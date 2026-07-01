# Common-Random-Number Sequential Halving Confirmation

ADR 0072 evaluated the promising ADR 0071 pilot on 20 fresh paired games,
seeds 35,703-35,722. Baseline and treatment used the same exact MLX model,
K32 frontier, R600 budget, LMR allocation, rollout and opponent policies,
integer scoring, elimination rule, bridge, and tie order. The only treatment
change was within-round common random numbers.

## Result

| Metric | Independent | CRN | Delta |
|---|---:|---:|---:|
| Mean score | 95.7750 | 95.4125 | -0.3625 |
| 95% paired CI |  |  | `[-1.1286,+0.4036]` |
| Wildlife | 61.1500 | 61.0500 | -0.1000 |
| Habitat | 30.9500 | 30.6000 | -0.3500 |
| Nature Tokens | 3.6750 | 3.7625 | +0.0875 |
| Seconds/game | 150.1772 | 148.3446 | -1.8326 |

CRN won eight games, tied one, and lost eleven. Per-game deltas ranged from
-3.25 to +3.25.

## Integrity

- 1,600 attempted and translated states per arm;
- 1,600 legal selected actions per arm;
- zero bridge fallback and zero rollout-policy fallback;
- 923,320 independent and 923,646 CRN rollout samples;
- both runtime gates passed;
- both MLX services shut down cleanly;
- model manifest:
  `dd3ea3bbbff0187107695132531a56c09a1da18b58fac4bacacf66960fd7ff0d`.

The checksummed JSON report is
`docs/v2/reports/exact-mlx-sequential-halving-crn-v1-confirm20.json`.

## Conclusion

The three-game +1.167 pilot was a small-sample false positive. On the frozen
20-game confirmation, CRN was slightly weaker and statistically unresolved.
Because every systems and category guardrail passed, the rejection is about
gameplay strength rather than implementation quality. Same-budget CRN is
closed; no retry, parameter sweep, or CRN-derived training collection is
authorized.
