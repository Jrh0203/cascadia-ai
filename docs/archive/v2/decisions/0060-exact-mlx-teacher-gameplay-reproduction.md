# ADR 0060: Exact MLX Teacher Gameplay Reproduction

Status: passed on 2026-06-12. The exact MLX historical policy is qualified as
a local research baseline, not as the final V2 model.

## Context

The canonical historical teacher was independently qualified at 96.350 over
ten games, but its neural forwards still ran in Rust. ADRs 0055, 0058, and
0059 now prove that the same parameters and complete search execute through
MLX bit for bit at 1.073x native search runtime.

Before using that system as a local research baseline, gameplay must verify
that the exact MLX selector preserves canonical bridge integrity, practical
latency, category balance, and the previously measured strength class on fresh
seeds. Search parity alone cannot substitute for complete-game execution.

## Decision

Add an isolated `ExactMlxLegacyTeacher` under the feature-gated differential
boundary:

- canonical V2 owns game state, rules, scoring, replay, and legality;
- public state is translated through the already-qualified heuristic bridge;
- malformed expanded actions are filtered before the top-32 prefilter;
- all prefilter, LMR-prior, and rollout-policy neural forwards use ADR 0058
  request type 6;
- no native neural evaluation, score tolerance, or selection fallback is used;
- coordinate fallback remains available only under the bridge's existing
  explicit error classification and must stay below 1%;
- one long-lived warmed service is reused across every game;
- the service, bridge, neural batches, latency, score categories, and
  provenance are reported.

## Frozen Protocol

- Rules: symmetric four-player AAAAA, base score without habitat bonuses.
- Model: immutable ADR 0055 artifact.
- Search: K32, R600 sequential halving, LMR and diverse prefilter enabled.
- Baseline: promoted
  `late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`.
- Runtime smoke: one game at seed 32,599.
- Strength pilot: ten paired games at seeds 32,600 through 32,609.
- One service warmup using an empty legal sparse row before timing games.

The smoke passes only if:

- all 80 decisions translate and remain legal;
- zero neural fallback, service error, or non-finite response occurs;
- bridge fallback rate is at most 1%;
- treatment runtime is at most 240 seconds per game;
- service shutdown is clean.

The ten-game pilot additionally requires:

- exact MLX treatment mean at least 95.0;
- paired gain over promoted strong at least +1.5;
- paired 95% confidence lower bound above zero;
- total wildlife delta at least 0.0;
- habitat delta at least 0.0;
- Nature Token delta at least -2.0;
- non-token score delta at least +2.0;
- at least two additional board points per extra token spent.

Passing qualifies an MLX-backed historical baseline for fresh local research.
It does not promote historical weights as the final V2 solution and does not
open validation or test seeds.

## Maximum Compute

One seed-32,599 smoke followed, only if it passes, by one ten-game paired pilot
on seeds 32,600-32,609. No training, sweep, test split, or promotion.

## Result

The smoke and ten-game pilot passed every frozen gate.

Smoke seed 32,599:

- 80/80 translated and legal selected actions;
- zero bridge or neural fallback;
- 146.24 treatment seconds;
- 95.5 treatment mean versus 94.0 for promoted strong;
- clean service shutdown.

Fresh paired seeds 32,600-32,609:

- exact MLX mean 95.800 versus strong at 92.275;
- paired gain +3.525, 95% CI `[+2.388,+4.662]`;
- 10 wins, zero ties, zero losses;
- wildlife +1.700, habitat +1.700, Nature Tokens +0.125;
- non-token score +3.400;
- 800/800 legal selected actions, zero fallbacks;
- 63,217,274 MLX neural rows in 39,886 batches;
- 151.25 treatment seconds per game;
- clean shutdown after all ten games.

The machine-readable confirmation is
`docs/v2/reports/legacy-nnue-v4opp-exact-mlx-gameplay-confirm10.json`, BLAKE3
`79a6ec66fabccfe94e965ed2e3ae35e6050c637cf85d12f8627c4ff1b24fbec9`.

This independently reproduces the historical teacher's 95+ strength class
with every neural forward on MLX. The remaining measured gap to the primary
100-point target is 4.200 points. Historical weights remain a qualified
baseline and research instrument, not the final solution.
