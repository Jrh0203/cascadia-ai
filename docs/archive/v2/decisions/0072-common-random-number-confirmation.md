# ADR 0072: Common-Random-Number Confirmation

Status: rejected on 2026-06-13.

## Context

ADR 0071 changed only the stochastic coupling inside the qualified exact-MLX
K32/R600/LMR sequential-halving teacher. Its independent path remained
bit-exact, the R600 smoke passed, and the three-game pilot favored CRN in every
pair: 97.0833 versus 95.9167, +1.1667 with 95% CI
`[+0.5778,+1.7556]`.

The pilot is encouraging but too small for promotion, training-label
collection, or a durable strength claim. This ADR freezes the separate
confirmation that ADR 0071 required.

## Decision

Run exactly 20 fresh paired games on seeds 35,703-35,722:

- rules: symmetric four-player AAAAA, base score without habitat bonuses;
- baseline: exact-MLX K32/R600/LMR sequential halving with independent
  per-candidate rollout seeds;
- treatment: the identical search with common random numbers within each
  halving round;
- model: immutable `artifacts/models/legacy-nnue-v4opp-mlx-v1`;
- execution: separate warmed MLX GPU services, arms run sequentially;
- unchanged: model, candidate frontier, rollout budget, LMR allocation,
  rollout/opponent policies, integer scoring, elimination, bridge, and tie
  order.

## Confirmation Gates

The treatment is confirmed only if all conditions hold:

- 1,600 legal actions per arm with zero bridge or neural fallback;
- exact rollout accounting, finite responses, and clean shutdown;
- at most 220 seconds per game for each arm;
- paired CRN-minus-independent mean at least +0.50;
- paired game-block 95% confidence lower bound strictly above zero;
- CRN absolute mean at least 96.0;
- total wildlife delta at least -0.50;
- habitat delta at least -0.50;
- Nature Token delta at least -1.0.

Passing confirms CRN as the stronger local research teacher and authorizes a
separate decision about fresh MLX-native policy data. It does not satisfy the
100-point goal and does not make the historical model a final V2 artifact.
Failure closes same-budget CRN without retries or threshold changes.

## Maximum Compute

One 20-game paired confirmation, local CPU and MLX GPU only. No training,
architecture, budget, frontier, confidence, seed, or parameter changes. No
retry, sweep, test split, external compute, or continuation after an
operationally invalid run without a separately documented root-cause fix.

## Command

```bash
make exact-mlx-crn-confirm
```

## Result

The complete 20-game confirmation was operationally valid:

- 1,600 legal actions per arm;
- zero bridge or policy fallbacks;
- 923,320 independent and 923,646 CRN rollout samples;
- 150.18 and 148.34 seconds per game;
- finite exact-MLX responses and clean shutdown.

Strength did not confirm:

- independent mean: 95.775;
- CRN mean: 95.4125;
- paired delta: -0.3625;
- 95% CI: `[-1.1286,+0.4036]`;
- record: 8-1-11;
- wildlife: -0.100;
- habitat: -0.350;
- Nature Tokens: +0.0875.

CRN failed paired gain, confidence, and absolute-mean gates. Category and
systems guardrails passed. This is a valid strength rejection and closes
same-budget common random numbers without retry or tuning.
