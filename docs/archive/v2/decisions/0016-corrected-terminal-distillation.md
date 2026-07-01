# ADR 0016: Corrected Terminal Distillation

Status: rejected before promotion on 2026-06-11.

## Context

The first R8 terminal-ranker collection was invalidated before training because
candidate records observed the actual hidden refill. ADR 0015 introduced
hidden-order-invariant `PublicGameState` afterstates and
`compact-entity-v2`. The complete corrected smoke passed collection,
validation, MLX training, promotion, and Rust-served gameplay.

Training-data diagnostics from the rejected split also showed frequent exact
teacher ties. Ranking evaluation now separates tie-aware value recall from
strict candidate-index agreement; regret and pairwise fidelity remain the
decision-quality metrics.

## Decision

Recollect from untouched seed ranges:

- 64 train games, indices 64 through 127;
- 16 validation games, indices 16 through 31;
- one atomic shard per game;
- the frozen R8 terminal teacher and K8+H6+B8 frontier.

Train the unchanged 805k-parameter `entity-set-ranker-v1` architecture from
scratch with AdamW at `1e-4`, weight decay `1e-4`, group batch 16, at most 20
epochs, and patience five.

## Gates

Before gameplay:

- best selection loss must improve over initialization;
- mean top-one regret must be at most 0.75;
- pairwise accuracy must be at least 0.65;
- value-difference correlation must be at least 0.30;
- tie-aware top-one value recall must be at least 0.45.

Strict single-index accuracy is reported but is not a gate because equally
valued teacher actions are interchangeable.

Only a complete ranking pass permits the ten-game pilot on seeds 25400 through
25409. It requires +0.5 paired score, no more than 0.5 habitat or wildlife
loss, no more than one Nature Token lost, and at most two seconds per treatment
game. A passing pilot alone permits 50 disjoint games on seeds 25500 through
25549.

## Outcome

The corrected collection completed with 64 train games, 16 validation games,
5,120 train groups, 1,280 validation groups, and 96,070 total candidates.
Every record represented either one empty paired-draft slot or the two partial
slots of an independent draft; zero records contained a hidden refill.

Training stopped after 13 epochs at patience. Epoch 8 improved selection loss
from 2.664595 to 2.563423 and passed pairwise accuracy at 0.680330 plus
value-difference correlation at 0.507718. It failed mean top-one regret at
0.968164 and tie-aware top-one value recall at 0.275781. No model was promoted
and no gameplay was run.

The model receives complete candidate afterstates but no action identity,
newly placed tile marker, or explicit delta from the shared prestate. Phase
analysis showed only a small aggregate improvement over immediate rank 1
(0.968 versus 0.999 regret), concentrated early, with a late-game regression.
The next experiment must represent the action delta directly.
