# ADR 0030: Perfect-Information Diverse Frontier

Status: structural candidate-recall hypothesis confirmed on 2026-06-11;
diagnostic only and not promoted.

## Context

The focal perfect-information K8+H6+B8 oracle improved pattern-aware by 1.775
points but reached only 93.150. It shifted +11.325 into Bear while losing
4.975 Elk and 4.050 Hawk. This proves exact future information alone does not
make one-step pattern policy improvement target-level.

Two explanations remain entangled:

1. the base frontier omits species-preserving structural actions;
2. the frozen pattern continuation cannot maintain a balanced portfolio even
   when those actions are available.

Earlier public-information wildlife-diverse search was confounded by
finite-sample maximization. The perfect-information diagnostic removes that
source of winner's curse.

## Decision

Compare two focal-seat perfect-information policies on identical seed blocks:

- baseline: exact K8+H6+B8 pattern frontier;
- treatment: the same frontier plus up to two distinct candidates for each
  wildlife species (`W2`).

Both preserve the true hidden stack and bag, evaluate every candidate by exact
full-game pattern-aware continuation with common deterministic tie randomness,
rotate the focal seat through all four positions, and keep the other three
seats pattern-aware.

Only candidate frontier changes. Continuation, market prelude, tie-breaking,
rules, and reporting remain identical.

## Frozen Protocol

- Rules: canonical four-player AAAAA, no habitat bonuses.
- Runtime smoke: seed 29099.
- Pilot: ten paired four-seat blocks, seeds 29100-29109.
- Base frontier: K8+H6+B8+M4.
- Diverse frontier: K8+H6+B8+W2+M4.
- Full-game exact hidden-state continuation.
- Local CPU only.

The treatment must complete the smoke within 150 seconds per four-rotation
block. Then run the fixed pilot.

Interpretation:

- paired diverse-minus-base gain at least +0.50: structural species candidate
  recall is material and should enter the next public-information planner;
- gain below +0.25: candidate omission is not the primary limit; prioritize a
  stronger multi-turn continuation;
- treatment mean at least 97: the expanded frontier substantially changes the
  reachable one-step ceiling.

This remains diagnostic and non-promotable. Hidden information must never
enter product play, model inputs, or final benchmark claims.

## Result

The runtime smoke completed the treatment's four seat rotations in 14.558
seconds, well inside the 150-second gate. Its single-block score delta was
-0.500, so the fixed pilot proceeded without adaptation.

Across seeds 29100-29109 and 40 focal scores per strategy:

- exact base frontier mean: 92.625;
- exact W2 frontier mean: 93.975;
- paired gain: +1.350, 95% CI `[+0.704,+1.996]`;
- game-block record: 9-0-1;
- treatment runtime: 15.449 seconds per four-rotation block;
- treatment P90 focal decision latency: 337.532 ms.

The category change was Habitat -0.500, Nature Tokens +0.600, and Wildlife
+1.250. Wildlife deltas were Bear -0.050, Elk -0.450, Salmon +0.300, Hawk
-0.325, and Fox +1.775.

The gain exceeds the frozen +0.50 structural-recall threshold with a strictly
positive interval. The base frontier therefore omits materially useful
species-preserving actions. The treatment mean remains below the frozen
97-point ceiling boundary and 6.025 points below the target, so wider
candidates do not remove the continuation limit.

Because the benefit was concentrated in Fox while the other four species
summed to -0.525, the next public-information control should isolate Fox
candidate recall under the promoted confidence-gated terminal operator. That
tests transfer of the exact diagnostic without reopening the previously
rejected all-species W2 allocation failure.
