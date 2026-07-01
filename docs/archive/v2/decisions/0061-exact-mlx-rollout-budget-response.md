# ADR 0061: Exact MLX Rollout-Budget Response

Status: rejected on 2026-06-12. Search-budget scaling is closed as the next
lever for this model and allocator.

## Context

ADR 0060 establishes a fresh exact-MLX R600 control at 95.800, 4.2 points
short of the primary target. Historical notes claim search plateaus near much
smaller budgets, but V2 treats that claim as untrusted. The exact MLX service
now makes a clean local budget-response test possible without changing model,
rules, candidate generation, allocator, or arithmetic.

The first new strength question is whether rollout variance remains a material
bottleneck at R600. Doubling only the sequential-halving budget gives a direct
answer. A positive result would justify larger local search and new search-
guided data; a null result would redirect work toward value representation or
candidate recall.

## Decision

Compare two otherwise identical exact-MLX teachers:

- baseline K32/R600;
- treatment K32/R1200;
- same qualified weights, packed CSR operation, canonical bridge, LMR,
  diverse prefilter, greedy opponents, RNG derivation, and tie behavior;
- independent long-lived warmed services for baseline and treatment;
- paired symmetric four-player AAAAA games on identical seeds;
- complete bridge, neural batch, category, latency, runtime, and provenance
  reporting for both arms.

## Frozen Protocol

- Smoke: one paired game at seed 32,799.
- Pilot: three paired games at seeds 32,800-32,802.
- Baseline runtime ceiling: 240 seconds/game.
- Treatment runtime ceiling: 420 seconds/game.

The smoke passes if both arms complete 80 legal decisions with zero bridge or
neural fallback, finite exact-width service output, clean shutdown, and the
runtime ceilings above.

The three-game pilot is promising only if:

- treatment-minus-baseline mean is at least +0.50;
- treatment absolute mean is at least 95.50;
- total wildlife delta is at least -0.50;
- habitat delta is at least -0.50;
- Nature Token delta is at least -1.00;
- all integrity and runtime gates continue to pass.

This is a pilot, not a promotion test. Passing authorizes a separately
preregistered larger confirmation. Failure closes rollout-budget scaling as
the next lever at this model and allocator.

## Maximum Compute

One seed-32,799 smoke followed, only if it passes, by three paired games on
32,800-32,802. No training, parameter sweep, validation/test split, or
promotion.

## Result

The smoke passed every integrity and runtime gate. R600 and R1200 both scored
94.75; R1200 took 298.58 seconds versus 141.02 seconds for R600.

The three-game pilot failed the primary gate:

- R600 mean 97.583;
- R1200 mean 97.750;
- paired gain +0.167 versus the required +0.500;
- 95% CI `[-2.962,+3.296]`;
- record 2-0-1;
- R1200 took 311.43 seconds/game versus 151.19 for R600;
- wildlife +0.750, habitat -0.083, Nature Tokens -0.500;
- both arms completed 240 legal selections with zero fallback.

The pilot report is
`docs/v2/reports/exact-mlx-rollout-budget-r600-r1200-pilot3.json`, BLAKE3
`5a81d4ebbc2a49269316f771c3729c30f068f24bc777289ef2c81890489166ed`.

Doubling rollouts more than doubled neural rows and latency without a material
score response. No larger confirmation is authorized. The next controlled
question changes candidate recall rather than resampling the same retained
frontier.
