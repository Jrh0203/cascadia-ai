# Full-Legal Public Oracle K1024 V1 Preregistration

Status: **locked before substantive seed collection**

Date: 2026-06-15

Experiment ID: `full-legal-public-oracle-k1024-v1`

## Purpose

Full-Legal Public Oracle V1 proved that complete-action public-information
search improves the frozen champion, but its K64 plus champion-frontier pilot
stopped at 98.583 mean and +2.375 paired points. It won 11 of 12 paired games,
so the operator is useful, but it missed both advance gates.

The paired K1024 recovery identifies a specific remaining bottleneck. Under the
stronger high-confidence comparison, K64 recalled only 70.385% of winners,
while K1024 recalled 99.423% `[98.942%, 99.808%]` and retained only 0.002
`[0.000, 0.004]` mean regret. This experiment tests whether removing that
measured truncation lifts actual online score above 100.

## Frozen Treatment

The baseline remains the accepted exact MLX K32/R600 champion.

At every treatment decision:

1. run the unchanged K32/R600 champion and retain its action and frontier;
2. enumerate and cheaply score every canonical legal post-prelude action;
3. form the substantial set from the top **1,024** screen actions, all
   champion-frontier actions, the champion action, and 16 deterministic
   rank-stratified sentinels;
4. evaluate the union with exact full-terminal R1200 sequential halving and
   common random numbers within each round;
5. re-evaluate the best eight substantial actions, the champion action, and
   the best substantial champion-frontier action with exact full-terminal
   R4800;
6. play the highest-ranked R4800 action.

The only algorithmic difference from the closed K64 oracle is screen width
`64 -> 1024`. The model, executable, action enumeration, sentinels, rollout
budgets, coupling, ranking, and selection rules remain fixed. No hidden bag
order, future market, or realized trajectory enters selection.

K2048 is not authorized by this experiment. It improved audit recall by only
0.289 percentage point beyond K1024, so failure here closes brute-force width
as the immediate route and advances complete learned proposal/ranking.

## Rules And Identity

- Four-player AAAAA.
- Habitat bonuses disabled.
- The same strategy occupies all four seats.
- Baseline and treatment use the same raw game seed.
- Model: `artifacts/models/legacy-nnue-v4opp-mlx-v1`.
- Source BLAKE3:
  `3d8a378b8b3088141fbc30f3194a84681008c0c339263714b2e94d0ce4f3c40d`.
- Executable SHA-256:
  `b1dee74da6e2288c51358d1f146deb2c73b9d8d64ac88646413fdc4ec85bf7d3`.
- Executable BLAKE3:
  `b666e499cc04d8d74236baedeb10761879d2818f59a1a83ea8d083056d05f0fd`.
- `MCE_LMR=1`.
- `MCE_DIVERSE_PREFILTER=1`.
- Rust owns legality, simulation, scoring, and search.
- MLX owns neural inference.

Any identity mismatch, policy fallback, bootstrapped rollout sample, illegal
action, incomplete game, dirty service shutdown, seed overlap, or process swap
invalidates the affected stage.

## Pilot

The pilot contains 12 fresh paired games:

- john1: seeds `62040-62043`;
- john2: seeds `62044-62047`;
- john3: seeds `62048-62051`.

The pilot advances only if all integrity gates pass and:

- treatment mean is at least 100.000;
- paired mean improvement is at least +3.000 points;
- no host has a negative paired mean;
- early, middle, and late phases contain exactly 28, 28, and 24 decisions per
  game;
- the treatment executes exactly 80 decisions per game.

Pilot confidence intervals are descriptive because 12 games are a screening
domain.

## Confirmation

The confirmation domain remains sealed unless the pilot passes every gate:

- john1: seeds `62200-62213`;
- john2: seeds `62214-62226`;
- john3: seeds `62227-62239`.

Confirmation passes only if:

- treatment mean is at least 102.000;
- paired mean improvement is at least +6.000 points;
- the paired bootstrap 95% confidence lower bound is positive;
- all three hosts reproduce a positive paired mean;
- every integrity and identity gate passes.

Passing establishes a reachable public-information ceiling above 100 and
authorizes complete-action policy/value learning from the K1024 teacher. It
does not promote this expensive oracle as the product player.

## Reporting

The report must include aggregate and per-host means, confidence intervals,
paired deltas, score decomposition, action-change and recall diagnostics,
runtime, all seed pairs, complete identity, integrity telemetry, and checksums.

Raw artifacts live under
`artifacts/experiments/full-legal-public-oracle-k1024-v1/`.
