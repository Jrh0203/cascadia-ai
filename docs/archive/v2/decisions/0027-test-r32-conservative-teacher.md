# ADR 0027: Test An R32 Conservative Teacher

Status: rejected after pilot on 2026-06-11.

## Context

Two MLX models learned useful lower-bound structure but could not reproduce the
promoted R8 policy's sparse challenger choices. The R8 sample stream is keyed
by game seed, information deliberately absent from the public model input.
Distilling its exact sample-dependent decision is therefore the wrong target.

Before collecting another neural dataset, establish whether reducing terminal
Monte Carlo error improves the policy itself.

## Decision

Compare the promoted final-five R8 c90 policy directly with an otherwise
identical R32 c90 policy:

- original K8+H6+B8 pattern frontier;
- pattern-aware anchor and continuation;
- four future market draws;
- shared public-information samples per candidate;
- one-sided 90% paired Student-t lower bound;
- exact t critical for 31 degrees of freedom;
- same anchor fallback and tie rules.

No cutoff, candidate, continuation, confidence, or scoring parameter changes.

## Frozen Protocol

1. Sequential runtime smoke on seed 28699. R32 must complete in at most
   35 seconds.
2. If runtime passes, ten paired games on seeds 28700-28709.
3. The pilot must achieve:
   - R32 minus R8 mean base score at least +0.20;
   - total wildlife and habitat deltas at least -0.50;
   - Nature Token delta at least -1.00;
   - no rules, determinism, or replay failure.
4. Only a passing pilot may authorize a disjoint 50-game confirmation.

If R32 is null or negative, higher-sample terminal labels are not justified as
the next neural teacher. If it passes, a separate ADR must freeze stable label
collection and MLX training before test data are touched.

## Result

The sequential runtime smoke passed:

- R8: 8.167 seconds;
- R32: 24.454 seconds;
- R32 P90 decision latency: 1,158 ms;
- smoke paired delta: +0.250.

The authorized ten-game pilot on seeds 28700-28709 failed the primary gate:

| Metric | R8 | R32 | Delta |
|---|---:|---:|---:|
| Mean base score | 91.425 | 91.300 | -0.125 |
| Total wildlife | 58.925 | 58.875 | -0.050 |
| Habitat | 28.950 | 28.775 | -0.175 |
| Nature Tokens | 3.550 | 3.650 | +0.100 |

The paired 95% confidence interval was `[-0.616,+0.366]` with a 4-0-6 record.
Category guardrails and runtime passed, but the required +0.20 strength gain
did not. No confirmation or higher-sample neural collection was permitted.
