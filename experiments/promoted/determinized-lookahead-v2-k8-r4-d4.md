# Determinized Lookahead V2 K8/R4/D4

Status: promoted on 2026-06-10 as the interactive v2 strategy.

## Hypothesis

The immediate-score top-4 filter excludes actions with stronger multi-turn
value. Expanding only candidate breadth to eight should improve complete-game
score without changing hidden-state sampling or rollout horizon.

## Diagnostic Evidence

On 400 decisions from disjoint K4 trajectories:

- K4 value recall against K8 was 83.25%;
- 67 decisions excluded a strictly higher-valued K8 action;
- misses averaged 0.455 estimated leaf points when they occurred; and
- early-game value recall was lowest at 79.26%.

## Playing Evidence

Exploratory pilot, seeds 20500-20509:

- K4 mean: 89.350
- K8 mean: 90.425
- paired delta: +1.075
- 95% CI: [-0.939, 3.089]
- record: 7-0-3

Disjoint confirmation, seeds 20700-20749:

- K4 mean: 89.305
- K8 mean: 90.270
- paired delta: **+0.965**
- 95% CI: **[+0.418, +1.512]**
- record: 31-4-15
- paired wall time: 253.779 seconds

The confirmation improved Mountain, Forest, Prairie, Wetland, River, Bear,
Salmon, and Hawk on average; Fox declined by 0.095 and Elk was unchanged.
Nature-token use was effectively unchanged.

A standalone 10-game benchmark completed in 43.975 seconds, or 4.397 seconds
per game, meeting the registered interactive runtime gate.

## Command

```bash
target/release/cascadia-v2 lookahead-ablate \
  --games 50 \
  --first-seed 20700 \
  --baseline-candidates 4 \
  --baseline-determinizations 4 \
  --baseline-greedy-plies 4 \
  --treatment-candidates 8 \
  --treatment-determinizations 4 \
  --treatment-greedy-plies 4 \
  --output docs/archive/v2/reports/lookahead-candidate-breadth-k8-confirm50.json
```

The confirmed mean is 9.730 points below the primary 100-point target.
