# Fresh V2 Baselines

Captured 2026-06-10 on the Apple M4 machine described in `HARDWARE.md`.

Protocol: `cascadia-aaaaa-4p-base-v1`. Each game uses the same complete policy
in all four seats. Scores exclude habitat bonuses. Confidence intervals use
game means as the independent unit.

| Strategy | Games | Seats | Mean | 95% CI | P10 | P50 | P90 | Games/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `random-v1` | 50 | 200 | 33.620 | 32.830-34.410 | 26.0 | 33.0 | 40.1 | 47.87 |
| `greedy-v1` | 50 | 200 | 86.265 | 85.733-86.797 | 81.9 | 86.0 | 91.0 | 1.09 |

## Promoted Interactive Baseline

`pattern-aware-v1-k8-h6-b8-m4` is the current promoted v2 interactive strategy.
It cleared a 50-game confirmation against greedy on seeds 23300-23349:

- pattern-aware mean: 91.525
- greedy mean: 86.635
- paired improvement: +4.890
- 95% paired CI: 4.296-5.484
- game record: 50 wins, 0 ties, 0 losses
- runtime: 0.820 seconds per game

The direct product control on seeds 23400-23449 scored 91.890 for pattern-aware
and 90.775 for K8. The reported K8-minus-pattern delta was -1.115 with 95% CI
-1.696 to -0.534. Pattern-aware ran at 0.292 seconds per game versus K8 at
4.227 seconds, a 14.49x speedup.

The policy uses only public state. It unions exact immediate K8, habitat H6,
and Bear B8 candidates in one shared legal-action pass. Each action is valued
by exact post-action base score plus the expected best legal one-token marginal
from four draws without replacement over public unplaced wildlife supply.

K8 remains a reproduced research control, not the product default.

## Legacy Champion Reference

The frozen v1 `mce_wide_v1` champion was independently reproduced over
deterministic seed offsets 0-49. The adapter parses all four `SYMPLAYER` rows
because the legacy CLI's aggregate summary records seat 0 only.

| Strategy | Games | Seats | Mean | 95% CI | P10 | P50 | P90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 `mce_wide_v1` + v4opp weights | 50 | 200 | 95.895 | 95.480-96.310 | 92.0 | 96.0 | 99.0 |

Mean components were 30.985 habitat, 61.075 wildlife, and 3.835 unused Nature
Tokens. Wildlife means were 11.615 Bear, 11.050 Elk, 12.580 Salmon, 11.110
Hawk, and 14.720 Fox.

This is an independently reproduced legacy reference, not a canonical v2
benchmark: the frozen binary retains v1 rules, state, and accounting. The run
was resumable and checksums both the binary and 23 MB weight artifact. Wall
times were heavily affected by the intentionally parallel local run and
concurrent compilation, so they are preserved for audit but not used as a
latency claim.

The greedy policy evaluates every canonical action after the free
three-of-a-kind replacement, including independent drafts when a Nature Token
is available. It chooses the action with the highest immediate base score and
does not perform lookahead or paid wildlife wipes.

Mean greedy category scores:

| Category | Bear | Elk | Salmon | Hawk | Fox |
|---|---:|---:|---:|---:|---:|
| Wildlife | 3.830 | 12.665 | 12.640 | 13.290 | 14.945 |

Mean habitat corridor sizes are 5.410 mountain, 5.585 forest, 5.485 prairie,
5.365 wetland, and 5.230 river. Mean unused Nature Tokens are 1.820.

Raw reports:

- `../archive/v2/reports/random-v1-50.json`
- `../archive/v2/reports/greedy-v1-50.json`
- `../archive/v2/reports/determinized-lookahead-v2-k4-r4-d4-vs-greedy-50.json`
- `../archive/v2/reports/lookahead-candidate-breadth-k8-confirm50.json`
- `../archive/v2/reports/determinized-lookahead-v2-k8-r4-d4-benchmark-10.json`
- `../archive/v2/reports/pattern-aware-v1-confirm50.json`
- `../archive/v2/reports/pattern-aware-v1-vs-promoted-k8-confirm50.json`
- `../archive/v2/reports/v1-champion-reference-50.json`

Reproduce:

```bash
cargo run --release -p cascadia-cli-v2 -- benchmark \
  --games 50 --first-seed 0 --strategy greedy
```

These are baseline measurements, not promotion-grade strength claims.
