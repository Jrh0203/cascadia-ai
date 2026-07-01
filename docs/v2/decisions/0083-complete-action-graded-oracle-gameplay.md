# ADR 0083: Complete-Action Graded Oracle Gameplay

Status: closed unopened after ADR 0081 validation failed.

Date: 2026-06-15

## Purpose

Offline recall does not prove playing strength. If and only if ADRs 0081 and
0082 pass, this ADR integrates the frozen ranker as a learned complete-action
screen and tests whether its top 64 candidates convert the K1024 evidence into
a 100-point online player.

## Frozen Treatment

Baseline: `exact-mlx-k32-r600-champion-v1`.

Treatment: `complete-action-graded-oracle-k64-v1`.

At every treatment decision:

1. apply the unchanged canonical free three-of-a-kind prelude;
2. enumerate every canonical legal action;
3. obtain the unchanged historical dense screen features;
4. score every action once with the frozen ADR 0082 checkpoint;
5. retain the learned top 64, all champion-frontier actions, the champion
   action, and 16 canonical rank-stratified sentinels;
6. evaluate the union with exact full-terminal R1200 and common random
   numbers;
7. re-evaluate the best eight R1200 actions, the champion action, and the best
   champion-frontier action with exact full-terminal R4800;
8. play the first canonical-hash maximum R4800 action.

The model does not see hidden bag order, future refill order, realized hidden
trajectories, or rollout results from the current decision. No fallback,
repair, blend, ensemble, alternate checkpoint, extra action, K1024 safety
union, or post-hoc calibration is allowed.

Before gameplay, direct Python and framed Rust/MLX inference must agree within
`1e-5` on every action score and produce identical top-64 sets on 128 fixed
validation groups. The service must load the exact sealed checkpoint, score
every action once, shut down cleanly, and reject identity drift.

## Runtime Smoke

- Seed: `62299`.
- One paired game on john1.
- Exactly 80 legal treatment decisions.
- Zero fallback, bootstrap, process swap, illegal action, non-finite score, or
  dirty shutdown.
- Mean learned-screen inference at most 250 ms per decision.
- Complete treatment mean decision latency at most 3.5 seconds.

The smoke is wiring evidence only.

## Pilot

Twelve fresh paired games:

| Host | Seeds |
|---|---|
| john1 | `62300-62303` |
| john2 | `62304-62307` |
| john3 | `62308-62311` |

The pilot advances only if every integrity gate passes and:

- treatment mean is at least 100.000;
- paired mean improvement is at least +3.000 points;
- no host has a negative paired mean;
- the learned top-64 set recalls the played R4800 winner in at least 98% of
  decisions;
- early, middle, and late paired score contributions are not negative;
- treatment executes exactly 80 decisions per game;
- no healthy node is idle more than five minutes while a compatible shard is
  queued.

Pilot intervals are descriptive. Failure closes the confirmation unopened.

## Confirmation

Forty disjoint paired games, opened only after a complete pilot pass:

| Host | Seeds |
|---|---|
| john1 | `62400-62413` |
| john2 | `62414-62426` |
| john3 | `62427-62439` |

Confirmation passes only if:

- treatment mean is at least 100.000;
- paired mean improvement is at least +3.000 points;
- paired game-block bootstrap 95% confidence lower bound is above zero;
- every host has a positive paired mean;
- every integrity, identity, recall, phase, runtime, shutdown, replay, and
  cluster-utilization gate from the pilot passes.

A pass qualifies a new research baseline and authorizes the next
PLAN_TO_100.md phase. It does not open the final 1,000-game domain or complete
the project.

## Closure

ADR 0081 failed its frozen validation recall gates, so ADR 0082 never opened.
The required precondition for inference integration and gameplay was therefore
not met. No parity fixture, runtime-smoke seed, pilot seed, confirmation seed,
or gameplay artifact was opened under this ADR.

## Maximum Compute

One parity qualification, one runtime smoke, one 12-game pilot, and a
conditional 40-game confirmation. No extra seed, retry, K1024 union, alternate
checkpoint, calibration, threshold change, external compute, or final-domain
game is authorized.
