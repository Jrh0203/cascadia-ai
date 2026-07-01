# ADR 0034: Perfect-Information Portfolio Beam

Status: rejected after pilot on 2026-06-11.

## Context

The exact width-16 focal beam confirmed +0.750 multi-turn continuation value,
but its category movement remained uneven: Bear +0.400 and Elk +0.675 were
partly paid for by Salmon -0.775 and Fox -0.450. The beam retained states by
one scalar pattern heuristic, so competing wildlife portfolios could disappear
before exact terminal scoring.

The first public beam recovery was null and is not a qualified teacher. Before
another public or neural attempt, test whether beam-state retention itself is
discarding useful allocation diversity.

## Decision

Compare two focal-seat perfect-information beams against three frozen
pattern-aware opponents:

- baseline: ADR 0032 scalar width-16 W2 beam;
- treatment: identical search with portfolio-preserving width-16 retention.

At each future focal layer, the treatment scores every child on eight fixed
dimensions:

1. original scalar pattern heuristic;
2. total habitat score;
3. Bear;
4. Elk;
5. Salmon;
6. Hawk;
7. Fox;
8. Nature Tokens.

It retains up to the best two distinct states per dimension in that fixed
order, then fills any unused width by scalar heuristic. Root frontier,
opponents, exact hidden state, common continuation randomness, activation
turn, terminal objective, and total beam width remain unchanged.

This is diagnostic only and non-promotable.

## Frozen Protocol

- Rules: canonical four-player AAAAA, no habitat bonuses.
- Runtime smoke: seed 30299.
- Pilot: seeds 30300-30309.
- Four focal-seat rotations per seed.
- Frontier: K8+H6+B8+W2+M4.
- Beam width: 16.
- Beam activation: final five personal turns.
- Local CPU only.

The treatment must complete the smoke within 180 seconds per four-rotation
block.

Interpretation:

- paired portfolio-minus-scalar gain at least +0.25: scalar pruning discards
  material portfolio value;
- treatment mean at least 97: portfolio retention materially raises the
  diagnostic ceiling;
- treatment mean at least 100: target-level focal play exists under exact
  future information and this frontier;
- gain below +0.10: portfolio retention is not the missing continuation
  mechanism.

Report the paired interval, all score categories, runtime, latency, record,
determinism, and provenance. No width, quota, dimension, order, frontier,
cutoff, or heuristic tuning is permitted between stages.

## Required Tests

- retention never exceeds the configured width;
- every retained state is an evaluated child and no state is duplicated;
- fixed dimension order and two-state quotas are deterministic;
- unused capacity is filled by scalar order;
- width one collapses to scalar retention;
- final-turn selection matches scalar beam when no future pruning occurs;
- complete matches are legal, deterministic, and replayable;
- the CLI compares explicit focal score blocks with complete provenance.

## Result

The runtime smoke tied exactly at 93.750. Portfolio retention completed the
four-seat block in 108.829 seconds, below the frozen 180-second gate, with
3,378 ms P90 decision latency.

Across seeds 30300-30309:

- scalar beam mean: 94.025;
- portfolio beam mean: 94.075;
- paired delta: +0.050, 95% CI `[-0.048,+0.148]`;
- record: 1-9-0;
- treatment runtime: 81.013 seconds per four-seat block;
- treatment P90 decision latency: 3,596 ms.

Habitat gained 0.050 and Nature Tokens gained 0.050, while total wildlife
fell 0.050. Bear was unchanged, Elk fell 0.050, Salmon gained 0.050, Hawk
gained 0.075, and Fox fell 0.125.

The treatment missed the +0.10 mechanism threshold and was an exact tie on
nine of ten seed blocks. Its 94.075 absolute mean remained below the 97
diagnostic boundary and 5.925 points below target. No confirmation is
permitted.

Equal-capacity category retention does not repair the scalar beam. The
remaining exact ceiling is more likely constrained by action recall, search
width, or the continuation heuristic itself than by collapse of wildlife
portfolio diversity during width-16 pruning.
