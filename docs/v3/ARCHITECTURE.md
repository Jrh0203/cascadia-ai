# CascadiaFormer Architecture

## Recommendation

Build CascadiaFormer-Zero: a sparse public-state transformer with legal-action
queries, Cascadia-specific relation bias, score-to-go/value heads, policy heads,
opponent/market auxiliary heads, and search-supervised expert iteration.

The core bet is narrow and testable: the remaining score gap is not raw local
pattern recognition alone. It is cross-turn pattern realization, market timing,
stochastic refill handling, and opponent-conditioned access. CascadiaFormer
should model those public entities jointly, then use search to improve and
verify action choice.

## Literature Basis

The architecture is intentionally no-frills:

- AlphaZero supplies the self-play plus search-improved policy/value recipe.
- ZeusAI for 7 Wonders Duel is the closest published board/card-game analogue:
  heterogeneous component tokens, transformer value/policy model, stochastic
  afterstates, and MCTS supervision.
- Chessformer and Leela transformer work show that domain-aligned tokenization,
  geometric bias, and action heads matter more than generic scale.
- Searchless chess transformer results argue for per-action value targets, but
  Cascadia lacks a Stockfish-grade oracle, so searchless distillation comes only
  after a strong Cascadia search teacher exists.
- Set Transformer, Graphormer/GraphGPS, Pointer Networks, Perceiver IO, Gumbel
  AlphaZero, and multiplayer AlphaZero inform the variable entity set,
  graph/geometry bias, dynamic action query, limited-budget search, and
  multi-seat value design.

## Public Input Schema

All model inputs must be public-state legal. Hidden future stack order is never
encoded.

Token groups:

- `game`: turn number, active seat, scoring card, cleanup state, phase bucket.
- `players`: seat-relative player summaries, nature tokens, current scores,
  remaining turns, and optional policy identity during self-play.
- `board_cells`: placed tiles and wildlife on each board with q/r, terrain
  edges, allowed wildlife, placed wildlife, keystone, rotation, owner, and age.
- `frontier_cells`: legal adjacent empty cells, local geometry, and prospective
  connectivity.
- `market`: four tile slots, four wildlife slots, three-of-kind state, and wipe
  affordances.
- `supply`: observable tile/wildlife counts and uncertainty summaries allowed by
  the engine boundary.
- `score`: current and potential category signals per seat.
- `history`: recent public actions only.
- `actions`: one token per legal compound action enumerated by Rust.

The action vocabulary is dynamic. Rust enumerates legal actions exactly; the
model scores those action queries instead of emitting a fixed global action id.

## Geometry And Relation Bias

CascadiaFormer uses a Cascadia geometric attention bias over public entities:

- axial hex distance and direction;
- same board, same component, same frontier, and adjacency;
- D6 orbit and transform identity;
- action uses market tile slot or wildlife slot;
- action touches a tile/wildlife/component already represented in state tokens;
- seat relation and turns until each opponent can affect the market;
- species, terrain, habitat, and scoring-category compatibility.

The default board fast path is a radius-6 hex disk with 127 stable cells. Legal
states outside the disk must remain exact through overflow entities. Overflow is
not clipped or projected.

## Model Shape

The current model family is an entity transformer plus action-query decoder:

```text
public tokens -> state transformer -> state latents
legal action tokens -> action encoder -> action queries
action queries x state latents -> policy, score-to-go Q, value, auxiliary heads
```

Primary heads:

- legal-action policy logits;
- per-action score-to-go Q;
- root value and score distribution;
- rank and score-differential summaries;
- category score decomposition.

Auxiliary heads:

- uncertainty for Q weighting and diagnostics;
- opponent next-draft and market-survival signals;
- pattern portfolio signals for Bear, Elk, Salmon, Hawk, and Fox;
- greedy-retention diagnostics while bootstrapping.

Initial model sizes:

| Model | Layers | Width | Heads | Use |
|---|---:|---:|---:|---|
| CascadiaFormer-S | 10-12 | 384 | 8 | bootstrap, EI-0, ablations |
| CascadiaFormer-M | 14-18 | 512 | 8 | main RTX 5090 target after S passes |
| CascadiaFormer-L | 20-24 | 768 | 12 | only after data and gates justify it |

Use bf16 mixed precision on CUDA, gradient checkpointing when needed, and packed
relation-tail batches rather than Python-built dense relation matrices.

## Serving Semantics

The model's raw Q output is predicted score-to-go, not final score. Any Q-based
serving path must rank actions by:

```text
derived_final_q = exact_afterstate_score_active + predicted_score_to_go
```

This is mandatory because exact immediate score can differ across legal
afterstates. A synthetic rank-flip test must fail if serving ranks by raw
score-to-go alone.

## Search Semantics

The serving-strength search is Gumbel top-m + sequential halving
(`real-root-exporter/src/gumbel.rs`) with the model at both ends: policy
priors select root candidates from the full legal set, and leaf values are
derived final Q from batched model evaluations. Interior plies advance every
seat by its own argmax derived final Q (max^n). A blend weight `w` mixes the
value bootstrap with sampled greedy terminal rollouts while the value head
earns trust; `w = 1.0` removes CPU rollouts entirely.

**No-peek contract:** search may never observe the true hidden tile-stack or
bag order. Every simulation redeterminizes hidden state before the root
action is applied; the legacy rollout path honors the same contract behind
`--rollout-determinize`. Benchmarks that violate this contract are marked
legacy-leaky and are not promotion evidence.

**Optional market-policy contract:** a free three-of-a-kind wildlife wipe is
an accept/decline decision followed by a chance draw and then the ordinary
draft decision. Policy and search value accept from public-hash-derived hidden
samples; they never condition acceptance on the real replacement order.
CascadiaFormer uses separate model rows for sampled accepted markets and, only
after acceptance, for the real revealed market. See
[RULES_CONTRACT.md](RULES_CONTRACT.md).

## Promotion Philosophy

Validation loss, imitation accuracy, and greedy top-1 retention are diagnostics.
Promotion requires paired gameplay evidence with:

- score mean and confidence intervals;
- category and wildlife breakdowns;
- search-retention/regret metrics;
- timing/resource ratios;
- clean provenance and dataset manifests.

The expected first useful mode is search-integrated serving: CascadiaFormer as a
K24/K32/K64 prefilter or value model inside sampled search. A direct no-search
policy is useful only after it proves nonregression.
