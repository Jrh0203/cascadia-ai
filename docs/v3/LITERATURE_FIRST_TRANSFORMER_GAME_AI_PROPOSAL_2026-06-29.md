# Literature-First Transformer Game AI Proposal

Date: 2026-06-29

This memo deliberately treats prior Cascadia repo experiments as low-confidence
context. The proposal below is driven by published literature, official project
writeups, and architectural evidence from strong game-playing systems. It assumes
a clean start: we can build the simulator, data format, model, training loop, and
evaluation harness without preserving existing technical debt.

## Executive Recommendation

Build **CascadiaFormer-Zero**: an AlphaZero/ZeusAI-style self-play system with a
domain-aligned transformer, chance-aware MCTS, search-supervised policy/value
training, and structured auxiliary heads.

The core model should be a **sparse entity transformer with dynamic Cascadia
geometry biases**:

- Use tokens for boards, tiles, wildlife, market slots, players, score cards,
  remaining supply, and legal action candidates.
- Use a dynamic attention-bias generator inspired by 2026 Chessformer GAB, but
  adapted to Cascadia's hex geometry, market drafting, turn order, and scoring
  structure.
- Use a legal-action query policy head rather than a giant fixed action index.
- Train from self-play MCTS visit distributions and per-action Q estimates,
  not only final outcomes.
- Predict score distributions and score decompositions, not just win/loss.
- Treat stochastic refills and hidden future supply as explicit chance nodes,
  bounded by progressive widening as in ZeusAI.

The shortest version: **ZeusAI gives us the game-family template, Chessformer
gives us the representation lesson, and searchless chess transformers give us
the target lesson.**

## Source Hierarchy

Trust order for this proposal:

1. Peer-reviewed or major-conference game AI papers and official paper pages.
2. Official project writeups from strong open engines, especially Leela Chess
   Zero.
3. Reproducible open-source implementations and benchmark datasets.
4. Cascadia repo logs and local experiment notes, used only as hypotheses to
   test later.

Important sources:

- ZeusAI, "Learning to Play 7 Wonders Duel Without Human Supervision" (CoG 2024):
  <https://arxiv.org/abs/2406.00741>
- ZeusAI homepage, including 2025/2026 public results:
  <https://sites.google.com/view/zeus-ai/>
- Chessformer, ICLR 2026, "Chessformer: A Unified Architecture for Chess
  Modeling": <https://openreview.net/forum?id=2ltBRzEHyd>
- Chessformer arXiv version with implementation details:
  <https://arxiv.org/abs/2605.19091>
- Searchless chess transformer, "Amortized Planning with Large-Scale
  Transformers: A Case Study on Chess": <https://arxiv.org/abs/2402.04494>
- ResTNet, IJCAI 2025, "Bridging Local and Global Knowledge via Transformer in
  Board Games": <https://www.ijcai.org/proceedings/2025/828>
- Gumbel AlphaZero/MuZero, ICLR 2022, "Policy improvement by planning with
  Gumbel": <https://openreview.net/forum?id=bERaNdoegnO>
- Multiplayer AlphaZero: <https://arxiv.org/abs/1910.13012>
- Perceiver IO: <https://openreview.net/forum?id=fILj7WpI-g>
- AlphaZero, Science 2018: <https://www.science.org/doi/10.1126/science.aar6404>

## What The Literature Actually Says

### ZeusAI Is The Closest Analogue

7 Wonders Duel is not Cascadia, but it is the closest published transformer game
AI analogue I found:

- It is a board/card game, not a spatial perfect-information abstract like chess
  or Go.
- Its state is made of heterogeneous public components: cards, wonders, progress
  tokens, coins, military track, age layout, discards, and player cities.
- It has no hidden information in the current public state, but it has stochastic
  setup and stochastic reveal/continuation events.
- It has multiple strategic objectives and victory modes.
- The state has no simple CNN-friendly grid geometry.

ZeusAI uses AlphaZero-style MCTS plus a transformer encoder. The paper reports a
12-layer, 12-head transformer with model dimension 768, feedforward dimension
3072, and about 92M parameters. The state is represented as learned embeddings
for components and positions. It does not hard-code card costs/effects into the
network; those are learned through self-play. It outputs a value and legal-action
policy.

The search design is especially relevant. ZeusAI explicitly handles stochastic
afterstates caused by random reveals. To control branching, it caps afterstate
children at 11, then gradually relaxes this cap in non-training games. Training
used up to 1,000 MCTS simulations per move; non-training games used up to 5,000.
They warm-started with 35,000 rule-based games, then retrained every 3,000
self-play games on the most recent 100,000 games, for 420,000 total self-play
games.

Interpretation for Cascadia: do **not** start with a CNN or an NNUE clone. Start
with a component transformer and chance-aware MCTS. Cascadia has a meaningful hex
board, unlike 7 Wonders Duel, so we need stronger relational geometry than
ZeusAI, but its component/position encoding and bounded stochastic afterstates
are directly applicable.

### Chessformer Changed The Chess-Transformer Lesson

The older 2024 chess-transformer story was "scale can distill search." That is
still true, but incomplete. The newer, stronger 2026 Chessformer result says:
**domain-aligned tokenization, positional encoding, and action heads can beat
scale.**

Chessformer is an encoder-only transformer that:

- Represents the 64 board squares as tokens.
- Uses a dynamic domain-specific positional bias called Geometric Attention Bias
  (GAB).
- Predicts moves with an attention-based source-destination policy head.

The ICLR 2026 paper reports three important results:

- Maia-3 reaches 57.1% human move-matching accuracy with 79M parameters,
  exceeding prior 355M-parameter methods.
- Chessformer integrated into Leela Chess Zero added over 100 Elo and helped
  Leela configurations beat Stockfish in elite computer chess tournaments.
- The square-token design made attention and activations easier to interpret.

The older Leela/Chessformer preprint and Lc0 writeups add practical detail:
Leela's transformer work found that the chess board cannot be treated as a
generic Euclidean image or linear FEN string. Tokens need fixed semantic
positions, and attention needs chess-specific topology: diagonals, ranks, files,
knight moves, blocked lines, attacks, and dynamic openness.

Interpretation for Cascadia: a generic transformer over serialized game text is
the wrong first architecture. The model should see native Cascadia objects and
relations: hex adjacency, terrain continuity, wildlife connectivity, market slot
coupling, turn order, supply depletion, and legal action structure.

### Searchless Transformers Are Strong But Not The Main Path

DeepMind's 2024 searchless chess work trained transformers up to 270M parameters
on ChessBench: 10 million chess games annotated with Stockfish 16 legal move and
value labels, producing roughly 15 billion state-action data points. The largest
model reached 2895 Lichess blitz Elo without explicit search. The paper also
found that action-value targets are the strongest target family: predicting
per-move values beat pure state-value prediction and behavioral cloning.

This is a huge clue, but not a reason to start searchless for Cascadia.

Chess has Stockfish as an oracle and vast human game databases. Cascadia does
not. If we want searchless CascadiaFormer, we first need a strong search teacher.
Searchless distillation should be phase 2, not the root architecture.

Interpretation for Cascadia: train on **search action-values** whenever possible.
Every MCTS root should record not just the final chosen move and visit counts,
but Q estimates for all legal candidate actions. This gives a far denser and more
useful training signal than one target per position.

### Hybrid Local/Global Models Matter

ResTNet, accepted at IJCAI 2025, interleaves residual CNN blocks and transformer
blocks in AlphaZero-style board-game networks. It improves playing strength in
Go and Hex and improves long-pattern recognition in Go, including circular and
ladder patterns.

The lesson is not "use a CNN for Cascadia." The lesson is that pure global
attention can be wasteful, while pure local convolution can miss global patterns.
Cascadia has both:

- Local: hex adjacency, habitat continuity, empty wildlife slots, isolation,
  salmon/hawk local conflicts.
- Global: scarce wildlife supply, market contention, opponent boards, endgame
  score category tradeoffs, and cross-board draft pressure.

Interpretation for Cascadia: use a transformer trunk, but give it local geometry
through relation biases or a light graph/local module. Do not expect vanilla
attention to discover hex topology efficiently from scratch.

### Multiplayer Requires A Value Vector

Multiplayer AlphaZero extends two-player scalar values into a score/value vector
and backs up each player's corresponding utility through MCTS rather than flipping
signs. That is directly relevant because Cascadia is 4-player and not strictly
zero-sum under the usual score-maximization objective.

Interpretation for Cascadia: the value head should predict at least:

- Own final score distribution.
- Final score vector for all seats.
- Rank distribution.
- Score differential distribution to each opponent.

Even if the objective is "maximize player 0's raw score," opponent modeling is
still part of the state because opponents control market depletion and drafting
pressure.

### Decision Transformers Are Not The Main Tool

Decision Transformer and multi-game Decision Transformer show that sequence
models can learn policies from offline trajectories. That is interesting for
imitation/distillation, but Cascadia needs legal-action awareness, stochastic
rollouts, and a self-improving teacher. A return-conditioned sequence model is
not the right primary architecture. It could later become a game-record model for
analysis or curriculum, but not the first superhuman agent.

## Cascadia-Specific Differences

Cascadia is not chess:

- It is multiplayer and non-zero-sum.
- The objective can be raw score, rank, or differential; those are not identical.
- It has public stochasticity through tile/wildlife supply and market refills.
- It has a sparse, growing hex board for each player rather than a fixed board.
- A move is a compound action: optional market cleanup, optional nature-token
  spending, draft choice, tile placement, wildlife placement, and rotation.
- Strong play depends on both local pattern construction and public-market
  denial/opportunity timing.

Cascadia is not 7 Wonders Duel either:

- It has real spatial geometry.
- It has four simultaneous player boards.
- The tactical action space is much more geometric.
- Score is additive and decomposable rather than victory-mode terminal.

Therefore the right architecture is a **ZeusAI-like component transformer with
Chessformer-like dynamic geometry and action heads**.

## Proposed Architecture: CascadiaFormer-Zero

### 1. Input Schema

Use sparse object tokens, not fixed image planes and not text serialization.

Token groups:

- `game_token`: turn number, active seat, scoring-card id, phase, cleanup state.
- `player_tokens`: one per player with nature tokens, visible score features,
  remaining turns, seat relation to active player, and optional style/version id
  during self-play.
- `tile_tokens`: one per placed habitat tile and one per visible market tile.
  Fields include owner/market slot, tile id, terrain edges, wildlife icons,
  keystone marker, rotation, hex coordinate, placement age, and board-local
  frontier flags.
- `wildlife_tokens`: one per placed wildlife and one per visible market wildlife.
  Fields include owner/market slot, species, coordinate if placed, and pairing
  with market tile if visible.
- `cell/frontier_tokens`: legal empty adjacent cells for each player board,
  capped naturally by game length. These make legal placement queries cheap.
- `supply_tokens`: remaining tile/wildlife counts by observable class, visible
  market composition, bag counts, and uncertainty summaries.
- `score_tokens`: one per scoring category and player, carrying current partial
  score, potential score distribution, and target scoring-card embedding.
- `action_tokens`: one per legal compound action at the current state.

The action-token design is crucial. Cascadia has variable legal actions, and many
are only meaningful after applying game-rule filters. Let the simulator enumerate
legal actions exactly, then let the model score those action tokens.

### 2. Relation And Geometry Bias

Implement **Cascadia Geometric Attention Bias (C-GAB)**:

- Start with static relation templates:
  - same player board
  - same market slot
  - tile paired with wildlife in a market slot
  - adjacent hex direction
  - hex distance bucket
  - same terrain edge/region candidate
  - same wildlife species
  - board token to frontier cell
  - player to opponent seat distance
  - action to its drafted market slot
  - action to its target tile coordinate
  - action to its target wildlife coordinate
- Feed a pooled global state summary into a small generator that mixes these
  relation templates into additive attention biases per layer or layer group.
- Keep relation templates game-rule/topology based, not handcrafted scoring
  bonuses. The model should get the geometry of the domain, not a hidden
  heuristic evaluator.

This is the Cascadia equivalent of Chessformer's GAB. Chessformer teaches that
tokenization, positional encoding, and output structure should match the domain's
actual geometry. For Cascadia, geometry is not just hex distance. It is also
market-slot coupling, turn order, supply, and score-category context.

### 3. Encoder Trunk

Start with a Zeus-scale model:

```text
CascadiaFormer-Zero-M
  transformer encoder layers: 12
  model dimension: 768
  attention heads: 12
  feedforward dimension: 2048 or 3072
  parameters: roughly 80M-120M depending on action/query heads
  precision: bf16/fp16 mixed precision
```

Then train two smaller ablations:

```text
CascadiaFormer-Zero-S
  layers: 8
  d_model: 384 or 512
  heads: 8
  purpose: fast iteration and ablations

CascadiaFormer-Zero-L
  layers: 15
  d_model: 1024
  heads: 16 or 32
  purpose: later scaling once the data engine works
```

NVIDIA's official RTX 5090 page lists 32 GB of GDDR7 memory. That is enough for
serious single-GPU iteration on the S and M models with gradient accumulation,
activation checkpointing, bf16/fp16, and compact token counts. Full self-play
generation will still be CPU/simulator intensive; the GPU should be treated as
the trainer and batched inference engine, not the whole factory.

### 4. Policy Heads

Use a hybrid policy design:

1. **Legal action query head**
   - Encode each fully legal compound action as an action token.
   - Cross-attend action tokens to the state tokens.
   - Output one logit per legal action.
   - This is robust, exact, and easy to train from MCTS visit distributions.

2. **Factor auxiliary heads**
   - Draft slot distribution.
   - Tile-placement coordinate distribution.
   - Rotation distribution.
   - Wildlife-placement distribution.
   - Nature-token spend/cleanup distribution.

The legal action head is the one used by search. The factor heads improve
representation learning and diagnostics.

This mirrors the lesson from Chessformer: the policy head should reflect the
structure of the action space. Chessformer uses source-destination attention
because chess moves are from-to traversals. Cascadia actions are compound
draft-placement assignments, so the head should explicitly represent that
compound structure.

### 5. Value And Auxiliary Heads

Primary value outputs:

- Own final score distribution, e.g. categorical bins from 50 to 130.
- Score vector distribution for all four players.
- Rank distribution.
- Pairwise score differentials versus each opponent.

Auxiliary outputs:

- Wildlife score by category.
- Habitat score by terrain.
- Nature-token score.
- Remaining score-to-go distribution.
- Short-horizon value, e.g. expected score after 1, 3, and 6 own turns.
- Legal-action Q values for every action token when search labels are available.
- Opponent next-draft distribution.
- Future market/supply prediction distribution.
- Uncertainty/error head, following the spirit of modern Leela/Chessformer value
  auxiliaries.

KataGo, Leela, and Chessformer all point in the same direction: auxiliary heads
make training easier and models more useful. For Cascadia, score decomposition is
not optional; it is the natural supervision signal.

### 6. Search

Use chance-aware MCTS:

- PUCT or Gumbel AlphaZero at root.
- Legal action priors from CascadiaFormer.
- Value bootstrap from CascadiaFormer.
- Chance nodes for market refill, wildlife bag draw, and tile-stack reveal.
- Progressive widening for chance nodes.
- Determinized rollouts only as an implementation detail, never as the value
  target definition.

ZeusAI is the model here: stochastic afterstates must be bounded or search
explodes. Gumbel AlphaZero/MuZero is also relevant because Cascadia search budgets
will likely be small relative to the action/chance space. The low-simulation
policy-improvement guarantee is more attractive than classic AlphaZero's
heuristic visit-count policy when root actions are many and not all can be
visited well.

For training games:

- Early: 128-400 simulations per move with Gumbel root selection.
- Main: 800-1,500 simulations for high-quality self-play.
- Evaluation: 1,500-5,000 simulations for slow reference agents.

The exact numbers are hypotheses. The invariant is more important: save the full
root search table so the model learns action-values, not just the chosen move.

### 7. Training Loop

Clean-slate training plan:

1. **Simulator and rules oracle**
   - Deterministic legal action generator.
   - Exact scoring.
   - Reproducible random seeds.
   - State hashing.
   - Symmetry transforms for hex rotations/reflections.

2. **Bootstrap corpus**
   - Generate broad legal random games plus simple non-neural policy games.
   - Train value and score-decomposition heads only.
   - Do not overfit a handcrafted heuristic policy; the goal is basic board and
     score literacy.

3. **Search-guided self-play**
   - Run chance-aware MCTS using the current model.
   - Record for each root:
     - state tokens
     - legal action tokens
     - visit distribution
     - per-action Q estimates
     - selected action
     - chance samples
     - final score vector
     - score decomposition
   - Train on a moving replay window plus a smaller long-term reservoir.

4. **Opponent/version mixing**
   - In 4-player games, sample model versions across seats.
   - Backup value vectors, not only active-player scalar values.
   - Track raw score and rank/differential metrics separately.

5. **Distill a fast policy**
   - Once search is strong, train a searchless CascadiaFormer policy/value model
     from action-value tables.
   - This is the analogue of searchless chess transformers, but it should come
     after the search teacher exists.

6. **Scale**
   - Start with the S model until the data contracts and search labels are
     stable.
   - Move to M as soon as the pipeline is not embarrassing.
   - Only move to L after ablations show C-GAB and action-query heads are working.

### 8. Evaluation

Do not evaluate only by mean score. Use a proper suite:

- 4-player mean score, P50/P90/P95, and low-tail score.
- Rank distribution and win/share-of-first metrics.
- Score differential to each opponent.
- Category score decomposition.
- Calibration of score distribution.
- Searchless policy accuracy against held-out search labels.
- Action-value rank correlation on held-out MCTS roots.
- Chance robustness across common random seeds.
- Ablations:
  - no C-GAB
  - static relation bias only
  - no action tokens
  - scalar value only
  - no score decomposition heads
  - no opponent/version mixing
  - PUCT vs Gumbel root search

The decisive early technical metric is not self-play score. It is whether the
model can predict held-out MCTS action rankings and score distributions. That is
the most literature-supported route from chess transformer results.

## Concrete Architecture Diagram

```text
                    exact simulator
                         |
              legal actions and chance nodes
                         |
        +----------------+----------------+
        |                                 |
  state object tokens                action tokens
        |                                 |
        +---------- CascadiaFormer -------+
                     encoder
              with C-GAB relation bias
                         |
        +----------------+------------------------------+
        |                |                              |
 legal-action policy   value/rank/score heads     auxiliary heads
        |                |                              |
        +----------------+------------------------------+
                         |
                 chance-aware MCTS
                         |
              self-play search labels
                         |
                  replay and training
```

## Why This Beats The Alternatives

### Versus NNUE

NNUE is excellent when the feature interface and update path are known and the
domain is mature. Starting from scratch, it prematurely commits us to handcrafted
feature design. The literature trend in game transformers says the feature design
should move into tokenization, relation bias, and auxiliary supervision rather
than hardcoded sparse features.

### Versus CNN/ResNet AlphaZero

Cascadia has four sparse growing boards and a market/supply system. A fixed image
plane is awkward and will waste capacity. ResTNet argues for local/global mixing,
but not necessarily CNN-first representation. C-GAB gives local geometry without
forcing everything into a dense image.

### Versus Generic LLM/Text Model

Text serialization throws away fixed object identity and native relations.
Chessformer explicitly shows that aligned tokenization and action heads matter.
Cascadia should not be represented as prose or a FEN-like string if we can give
the model game-native tokens.

### Versus Pure Searchless Transformer

Searchless chess works because Stockfish can label billions of action-values.
Cascadia lacks that oracle. Build the search teacher first, then distill.

### Versus Pure MCTS With Handcrafted Rollouts

The action/chance space is too structured and too strategic for unlearned rollouts
to scale. ZeusAI and AlphaZero-style systems use the network to reduce search
breadth and depth. Cascadia should too.

## Open Research Risks

- **Self-play collapse in non-zero-sum scoring.** Raw-score optimization may
  learn odd collusive or opponent-agnostic behavior. Keep rank and differential
  heads even if raw score is the main target.
- **Chance-node explosion.** Cascadia's future supply can branch hard. Use
  progressive widening and common-random-number evaluation from day one.
- **Sparse terminal rewards.** Mitigate with score-decomposition and
  short-horizon auxiliary targets.
- **Action-token scaling.** Full legal action counts may become large. Use exact
  legal generation, batching, and if needed a two-stage candidate policy, but keep
  the training labels full-action where possible.
- **Over-encoding scoring heuristics.** Relation templates should encode legal
  geometry and component identity, not manual "this is worth 3 points" features.

## First Build Milestones

### Milestone A: Clean Simulator Contract

Deliverables:

- Exact legal action enumerator.
- Exact scoring with score decomposition.
- Chance transition API.
- Canonical state serialization.
- Symmetry transforms.
- Golden rule tests.

Exit criterion: random games are reproducible, legal, and exactly scoreable.

### Milestone B: Tokenizer And Model Smoke Test

Deliverables:

- Token schema versioned as a formal spec.
- Action token schema.
- C-GAB relation template builder.
- Small CascadiaFormer-S model.
- Unit tests for token/action invariance under legal symmetries.

Exit criterion: the model can overfit a tiny generated corpus and produce legal
action logits for every root.

### Milestone C: Search Teacher

Deliverables:

- Chance-aware MCTS with PUCT and Gumbel-root variants.
- Root table export with visit counts and per-action Q.
- Replay format.
- Search-vs-random and search-vs-simple-policy gates.

Exit criterion: search labels are stable enough that held-out action-ranking
metrics improve with training.

### Milestone D: First Self-Play Run

Deliverables:

- Self-play worker.
- Trainer with replay window.
- Evaluation suite.
- Model registry and promotion gates.

Exit criterion: CascadiaFormer-S beats non-neural search at equal or lower search
budget and shows calibrated score distributions.

### Milestone E: Zeus-Scale Run

Deliverables:

- CascadiaFormer-M, about 80M-120M params.
- Batched GPU inference server for search.
- RTX 5090 training profile.
- Ablation report for C-GAB, action query head, value vector, and auxiliary
  heads.

Exit criterion: M model materially improves held-out action-value rank and
search-guided play over S, with no regression in calibration.

## Final Proposal

The clean-slate architecture should be:

```text
CascadiaFormer-Zero
  state: sparse game-native object tokens
  geometry: dynamic Cascadia Geometric Attention Bias
  policy: legal action query head plus factor auxiliaries
  value: score/rank/vector distribution heads
  training: AlphaZero/ZeusAI self-play with chance-aware MCTS
  labels: visit distributions, per-action Q values, final score vectors,
          score decompositions, future-market/opponent-response auxiliaries
  deployment: search agent first, searchless distillation second
```

If we are unburdened by the past, this is the architecture I would build first.
It follows the strongest available evidence: ZeusAI for stochastic component
board games, Chessformer for domain-aligned transformer geometry, searchless
chess for action-value supervision, ResTNet for local/global board reasoning, and
Multiplayer AlphaZero for value-vector backups.
