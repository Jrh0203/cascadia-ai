# ADR 0015: Observable Candidate Afterstates

Status: accepted on 2026-06-11.

## Context

Candidate models must rank an action before the market refill caused by that
action is observed. The original `PositionRecord::afterstate` used a complete
`GameState::transition`, which consumed the real hidden tile and wildlife
stacks. This leaked inaccessible future information and mismatched teachers
that averaged redetermined futures.

Independent drafts also create a public intermediate market with one
tile-only slot and one wildlife-only slot. A complete-pair-only encoding could
not represent that state honestly.

## Decision

`GameState::preview_public_afterstate` now applies the observed prelude,
validates and applies the deterministic draft and placement, removes drafted
market components, advances the public turn metadata, and stops before refill.
It returns `PublicGameState`, which contains no stack, bag, discard, seed, or
RNG state.

`PositionRecord::observable_afterstate` is the only candidate-model boundary.
Paired drafts encode an empty market slot; independent drafts encode partial
slots. Rust and MLX use `compact-entity-v2`, making every older dataset and
model fail schema validation instead of silently mixing semantics.

## Verification

Rust tests prove paired and independent depletion, Nature Token accounting,
and byte-identical afterstates under different hidden stack orders. A search
test proves model-ranking inputs and outputs are hidden-order invariant.
Python tests prove empty, tile-only, wildlife-only, and complete slots decode
with correct masks and features.
