# Full-Legal Public Oracle V1 Preregistration

Status: **amended before replacement substantive seed collection**

Date: 2026-06-15

Experiment ID: `full-legal-public-oracle-v1`

## Correctness Amendment

The first launch against seeds `62000-62011` is invalid and excluded from
every pilot gate. The shared audit/oracle ladder incorrectly required the
R4800 high-confidence cohort to contain an action from the top complete-screen
prefix. That diagnostic is required by an offline screen-width audit, but it
is not part of the online oracle's selection contract. Seed `62000` reached a
cohort containing only champion-frontier and sentinel actions and aborted with
`high-confidence set contains no retained screen action`.

The correction makes retained-screen regret optional inside the shared ladder.
The offline audit still rejects a missing retained action when materializing an
audit record. The online oracle still plays the highest-ranked R4800 action.
No action enumeration, screen width, sentinel, union, rollout budget, random
coupling, or ranking rule changed.

Before shutdown completed, john2 finished four games and their scores became
observable. Reusing any part of `62000-62011` would therefore compromise a
screening gate. The replacement pilot uses fresh seeds `62020-62031`; all
thresholds and the sealed confirmation domain remain unchanged. The invalid
launch is preserved separately at
`docs/v2/reports/full-legal-public-oracle-v1-invalid-launch.md`.

Before opening the replacement domain, the corrected implementation passed 28
Rust tests, reproduced the frozen three-decision reference with normalized
SHA-256 `b8437c26885f271e88d02fa0927129184cd7d9e2c81968bc298b306fc8a3f7c2`,
and completed all 80 decisions on the old failure seed `62000`. That replay
had exact identity, clean service shutdown, zero fallbacks, zero bootstraps,
and zero process swaps. Its score is correctness-only evidence.

## Purpose

The Full-Legal Decision Regret Audit measured 0.350 points of mean local
champion regret, equivalent to 6.995 points over 20 personal turns as a
first-order diagnostic. This experiment tests whether that local signal
converts into actual online game score.

This is a public-information search ceiling experiment. It is not a promoted
player or a learned model. It uses no hidden bag order, future market, or
realized trajectory information when selecting an action.

## Frozen Player

The baseline is the accepted exact MLX K32/R600 champion.

At every treatment decision:

1. run the unchanged K32/R600 champion and retain its selected action and
   complete frontier;
2. enumerate and cheaply score every canonical legal post-prelude action;
3. form the substantial set from the top 64 screen actions, all champion
   frontier actions, the champion action, and 16 deterministic
   rank-stratified sentinels;
4. evaluate the union with exact full-terminal R1200 sequential halving and
   common random numbers within each round;
5. re-evaluate the best eight substantial actions, the champion action, and
   the best substantial champion-frontier action with exact full-terminal
   R4800;
6. play the highest-ranked R4800 action.

The audit measured 99.327% recall for the top-64 plus champion-frontier union,
with a game-block 95% interval of `[98.846%,99.712%]`. The retained top-64
regret was 0.047 points. Those results justify testing this union online
without representing K64 alone as adequate.

The separately preregistered K1024 recovery completed before the replacement
pilot opened. It recalled 99.423% of measured winners and retained 0.002 mean
regret, passing its complete-screen-only coverage and retained-regret gates.
That result does not change this oracle's already frozen K64 plus
champion-frontier selection contract.

## Rules And Identity

- Four-player AAAAA.
- Habitat bonuses disabled.
- The same strategy occupies all four seats.
- Baseline and treatment use the same raw game seed.
- Model: `artifacts/models/legacy-nnue-v4opp-mlx-v1`.
- `MCE_LMR=1`.
- `MCE_DIVERSE_PREFILTER=1`.
- Rust owns legality, simulation, scoring, and search.
- MLX owns neural inference.

Every shard must record and match the executable, model JSON, model weights,
and v2 source-root digests. Any policy fallback, bootstrapped rollout sample,
illegal action, incomplete game, dirty service shutdown, or seed overlap
invalidates the affected stage.

## Pilot

The replacement pilot contains 12 fresh paired games:

- john1: seeds `62020-62023`;
- john2: seeds `62024-62027`;
- john3: seeds `62028-62031`.

The pilot advances only if all integrity gates pass and:

- treatment mean is at least 100.000;
- paired mean improvement is at least +3.000 points;
- no host has a negative paired mean;
- early, middle, and late phases contain exactly 28, 28, and 24 decisions per
  game respectively, with finite local-regret and action-change diagnostics;
- the treatment executes exactly 80 decisions per game.

Pilot confidence intervals are descriptive because 12 games are a screening
domain. Failure closes this operator as the immediate route to 100 and sends
research toward learning the proposal signal rather than spending the
confirmation domain.

## Confirmation

The confirmation domain remains unopened unless the pilot passes every gate.
It contains 40 fresh paired games:

- john1: seeds `62100-62113`;
- john2: seeds `62114-62126`;
- john3: seeds `62127-62139`.

Confirmation passes only if:

- treatment mean is at least 102.000;
- paired mean improvement is at least +6.000 points;
- the paired 95% confidence lower bound is positive;
- all three hosts reproduce a positive paired mean;
- every integrity and identity gate passes.

Passing establishes a reachable public-information ceiling above 100 and
authorizes complete-action policy/value learning from the full-legal teacher.
It does not promote the oracle itself as the product player.

## Reporting

The report must include:

- baseline and treatment mean, confidence intervals, and paired delta;
- per-host means and deltas;
- habitat, wildlife, and Nature Token decomposition;
- action-change rate, local champion regret, screen recall, and action counts;
- baseline and treatment runtime;
- all seed-level paired game means;
- executable, model, source, and configuration identity;
- service, fallback, bootstrap, memory, and swap integrity;
- checksums for every raw and merged artifact.

Raw artifacts live under
`artifacts/experiments/full-legal-public-oracle-v1/`.
