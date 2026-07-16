# Beyond CascadiaFormer: a clean-sheet architecture for stochastic board-game AI

**Date:** 2026-07-16  
**Literature cutoff:** 2026-07-16  
**Status:** research synthesis and falsifiable architecture proposal; no Cascadia
strength claim, experiment authorization, rules change, or promotion  
**Objective:** mean seat score **at least 100 over 1,000 four-player games** under
the pinned all-A, no-habitat-bonus Cascadia rules identity in force at the gate

## Executive verdict

The highest-probability clean-sheet challenger is **not another larger
transformer and not a learned world model**. It is a search-first system built
around a much cheaper structured afterstate evaluator:

> **Cascadia-NX:** a D6-symmetry-tied local motif/factor network with NNUE-style
> incremental updates, a small global topology correction, and an explicit
> legal-action value head, embedded in a GPU-resident exact Cascadia planner
> that compares actions on the same counterfactual chance bundles.

The architectural bet is that CascadiaFormer is spending most of its 88.2M
parameters and serving time repeatedly reconstructing facts that the rules
engine can maintain exactly: local pattern deltas, habitat components, legal
compound actions, market/bag counts, and the current score. Cascadia-NX would
make those facts explicit and cheap, then spend the saved budget on the
operation the campaign's own measurements say matters: averaging stochastic
action differences until the best action is identifiable.

The literature provides unusually coherent support for this direction:

1. In stochastic 2048, the strongest published result located is not a general
   sequence model or learned world model. It is a symmetry-shared sparse
   n-tuple **afterstate** evaluator plus exact expectimax: **625,377 average
   score and 72% reaching 32768**, albeit over only 100 games at the six-ply
   setting. A Stochastic MuZero result reported in the same research line is
   about 510,000; the comparison is cross-study and not controlled.
2. A 2025 Azul thesis—the closest game-shape result located—reports that a
   shallow NNUE evaluator inside search beat the strongest handcrafted
   heuristic in **94.07% of 10,218 games**, and that giving the same system more
   search time continued to improve it. It is a two-player MSc result, not a
   peer-reviewed four-player Cascadia result, but it is direct evidence that a
   tiny, fast evaluator can dominate a much richer-looking heuristic in a
   stochastic point-scoring drafting game.
3. Stockfish's official NNUE introduction measured **+92.77 +/- 2.1 Elo** in a
   60,000-game single-thread test. The transfer is not chess's minimax; it is
   the economic principle that an incrementally updated evaluator can buy far
   more exact search.
4. TD-Gammon reached near-parity with a world-class player using a small
   afterstate value network, structural features, self-play TD learning, and
   shallow chance-aware lookahead. DouZero reached first place among 344 bots
   in a three-player stochastic card game using a shared state-action Q model
   and massive direct Monte Carlo training rather than a transformer or tree.
5. Pgx demonstrates that exact board-game simulation can be vectorized on
   accelerators at **10–100x** the throughput of existing Python environments.
   Variance-reduced MCTS shows that common random numbers and control variates
   can be worth roughly **25–60% more simulations** in stochastic games when
   the coupling creates positive covariance.

None of those numbers predicts a Cascadia score. Together they support a much
more specific claim: **when rules are exact, the strongest stochastic-game
systems often win by making evaluation cheap, representing afterstates
directly, and spending compute on exact or carefully coupled planning—not by
learning a large generic dynamics or sequence model.**

This is not a recommendation to stop the authorized R1.4-D1 pipeline. D1 is the
current funded line and was live under the repaired July-16 rules identity at
this report's cutoff. Cascadia-NX is a post-D1 architecture challenger whose
offline design and tests must not displace the live john0 chain or read partial
scientific output.

## 1. Claim boundary and evidence labels

This report uses four evidence labels:

- **Published-direct:** a primary paper or official project report directly
  measured the stated result.
- **Repo-direct:** a result or invariant measured in this repository. Historical
  scores retain their historical rules identity.
- **Transfer:** a mechanism demonstrated elsewhere whose Cascadia effect is
  unknown.
- **Synthesis:** a new combination proposed here. It must not be described as a
  published algorithm or a proven Cascadia improvement.

Important limits:

- No located paper studies Cascadia at this project's exact rules, objective,
  model, or compute budget.
- Scores, Elo, win rates, stable ranks, and leaderboard positions across games
  are not comparable measures of absolute strength.
- The strongest 2048 number uses only 100 evaluation games at its deepest
  search setting. The Azul result is an MSc thesis, uses two-player minimax, and
  has only three games against one top-five human; its large automated matches
  are the useful evidence.
- The historical Cascadia scalar champion is **98.2975** at n1024/d16 and was
  exactly reproduced on a fresh 100-game block under the closed July-9 rules
  identity. The distq arm's 98.3850 was statistically tied, so it did not
  replace the scalar champion. There is **no admissible July-16 canonical
  score yet**. The 1.7025 gap is historical context, not a current-identity
  estimate.
- Every architecture conclusion below is a hypothesis until a current-rules,
  paired, wall-matched Cascadia gate says otherwise.

## 2. What problem should the architecture actually solve?

The current stack is not failing because it cannot encode a board. It is
failing at the narrow decision boundary between several plausible compound
actions.

### 2.1 Measured Cascadia constraints

The proposal must explain all of these repo-direct facts, not just fit the
general literature:

| Fact | Consequence for a challenger |
|---|---|
| Median decision SNR is about 1.06 and roughly 46% of decisions are noise-flippable. | Lower variance in **action differences** matters more than prettier absolute values. |
| Selfish n4096/d16 scaling gained only +0.21, statistically nonsignificant, at about 4x simulations. | More of the same estimator/search axis is not enough. |
| Exact final-personal-turn K1 is score-neutral but about 29x faster at that frontier. | Preserve exact rules and exact score identities wherever possible. |
| CascadiaFormer-L (207M), 3x data, fresh-M, old-label upgrades, and output ensembles were flat. | Do not answer with a bigger same-family sequence model or a checkpoint ensemble. |
| A smaller transformer bought only about 1.9–2.0x CUDA throughput; even more than 3x had already failed the accuracy-for-search trade. | A challenger must change the per-call computation and engine boundary, not merely shrink width. |
| Structured action-conditioned category Q failed its preregistered head-only gate by 17.04% versus a required 10% improvement. | Category heads may regularize a new trunk; they must not be forced into the same load-bearing additive serving contract. |
| Pairwise comparator, wider root menu, LCB, static sigma, Q bias, generic paired rollout RNG, and chance-node leaf expectimax did not clear their bars. | A new system must change representation and simulation, not rename a root-selection tweak. |
| The bridge is within about 5% of its current architectural ceiling. | The systems escape hatch is a GPU-resident engine/search loop, not another IPC micro-optimization. |
| Serving must rank `exact_afterstate_score_active + predicted_score_to_go`. | Exact present score stays outside the learned residual. Own score remains the production objective. |

### 2.2 The latent structure Cascadia gives us for free

Cascadia is unusually factorable:

- one legal action changes a small local patch of one player's hex board;
- exact current score and exact score deltas are computable;
- habitat connectivity can be maintained with incremental component data;
- each wildlife card induces a finite family of local or semi-local motifs;
- scoring-card identity is fixed for the game and can select a small expert;
- the public market and finite bag give an exact chance model;
- legal compound actions are enumerated exactly by Rust;
- each player has a short, fixed horizon; and
- the actual objective is expected **own raw score**, not win probability or
  equilibrium exploitability.

A generic transformer can learn these facts, but it has to rediscover them at
every inference. A factored evaluator can cache them and update only what the
action touched.

## 3. What the best-performing literature actually says

The headline landscape below is deliberately heterogeneous. It answers “what
has performed best?” within each game's own evaluation; it is **not** a league
table and the numbers must never be compared across rows.

| System/domain | Core architecture | Strongest relevant reported result | Transfer limit |
|---|---|---|---|
| Azul NNUE/search | shallow incrementally reused NNUE + minimax | 94.07% wins vs strongest handcrafted heuristic over 10,218 games | MSc thesis; two-player |
| 2048 optimistic TD | symmetry-shared sparse n-tuples + afterstate TD + six-ply expectimax | 625,377 average; 72% +/- 12% reaching 32768 | only 100 games at deepest setting |
| Stochastic MuZero, 2048/backgammon | learned decision-afterstate-chance model + MCTS | about 510k in reported 2048 comparison; matched exact-simulator AlphaZero/GNUbg Grandmaster in its backgammon setup | learned-model cost/error unnecessary with exact Cascadia rules |
| TD-Gammon 2.1 | 80-hidden-unit value net + TD(lambda) + two-ply search | 39–40 over 40 games against Bill Robertie after 1.5M self-play games | tiny human match; two-player |
| Stockfish NNUE | sparse incrementally updated net + alpha-beta/PVS | +92.77 +/- 2.1 Elo in official 60k-game test; Stockfish 12 won at least 10x as many pairs as it lost to Stockfish 11 | deterministic zero-sum chess |
| DouZero | shared state-action Q + direct Monte Carlo + parallel actors | first among 344 Botzone agents; surpassed prior programs in days on four GPUs | massive-data, imperfect-information card game |
| Suphx | supervised pretrain + policy-gradient self-play + reward prediction + privileged training | stable rank 8.74 over 5,760 Tenhou games, reported above 99.99% of active players | hidden information and rank utility, not raw own score |
| Pluribus | CFR blueprint + real-time search | 48 +/- 25 mbb/game with five humans and one AI; 32 +/- 15 with five AI copies and one human | poker equilibrium/exploitation objective |
| AlphaZero | residual CNN + exact MCTS self-play | 155 wins, 6 losses, 839 draws in 1,000 games vs the tested Stockfish version | deterministic games and large compute |
| KataGo | residual/global-pooling net + MCTS + auxiliary targets | surpassed a comparable prior Go result with about 50x less self-play compute | Go-specific combined intervention |

The first five rows are the architectural center of gravity for Cascadia. The
last five establish that transformers are not a prerequisite for superhuman
multi-agent game play, while also showing why poker equilibrium methods,
Mahjong hidden-information tricks, or a giant generic AlphaZero clone are not
automatically the right transfer.

### 3.1 Closest analogue: Azul NNUE plus search

The 2025 thesis [Implementing superhuman AI for Azul board game with a
variation of NNUE](https://jakubkowalski.tech/Supervising/Rzepecki2025ImplementingSuperhuman.pdf)
is the closest located architecture result to Cascadia: drafting from shared
factories, stochastic setup/refill, spatial point construction, and a score
maximization objective.

Its final shallow networks use fully connected shapes such as
`128x16x16x1`, `256x64x64x1`, and `512x16x16x1`; the first layer is engineered
for incremental reuse. The reported final NNUE/search heuristic 350 achieved:

| Match | Result | Games |
|---|---:|---:|
| final NNUE/search vs first heuristic | 98.42% win rate | 10,998 |
| final NNUE/search vs strongest handcrafted heuristic 11 | 94.07% | 10,218 |
| 0.1 s search vs 1 s search | shorter-search arm won 25.06% | 16,356 |
| 1 s vs 5 s | shorter-search arm won 34.02% | 10,528 |
| 5 s vs 20 s | shorter-search arm won 38.27% | 10,086 |

The useful result is not the thesis's small human sample. It is the automated
ablation: **the cheap learned evaluator made search stronger than the hand
engine, and additional search time kept paying.** The direct transfer is
limited because the system is two-player minimax and Azul's state factors are
not Cascadia's.

### 3.2 Strongest stochastic score-game counterexample to “bigger model wins”

[Optimistic Temporal Difference Learning for 2048](https://arxiv.org/abs/2111.11090)
uses afterstate TD, temporal coherence, optimistic initialization, multistage
learning, symmetry sharing, tile downgrading, and six-ply expectimax. Its best
configuration consists of eight six-tuples and 268.4M sparse lookup weights;
most evaluation cost is table lookup and addition rather than dense neural
attention.

The published ablation is unusually informative:

| 2048 system | Average score |
|---|---:|
| base n-tuple learner | 309,208 |
| + optimistic initialization | 361,471 |
| + temporal coherence | 370,194 |
| + two-stage learning | 404,288 |
| + six-ply expectimax | 586,583 |
| + tile downgrading | **625,377** |

At the final setting, 72% +/- 12% of 100 games reached a 32768 tile. The paper
also reports a 0.02% 65536 rate in a larger test. The exact score should not be
over-read: the deepest result is noisy, and a giant sparse table is not the
same memory/computation trade as Cascadia.

The accompanying [2048 reinforcement-learning dissertation](https://arxiv.org/abs/2212.11087)
puts several results in one table: a DNN afterstate TD agent at three-ply
expectimax around 406,927, Stochastic MuZero around 510,000, an earlier n-tuple
agent at 609,104, and the optimistic n-tuple agent at 625,377. These are
cross-paper results with different training and test conditions, so the
ordering is suggestive rather than a controlled head-to-head. It is still the
clearest located warning against assuming a generic learned world model is the
best architecture for an exact stochastic score game.

### 3.3 Explicit chance factorization is valuable; learned dynamics are not

[Stochastic MuZero](https://openreview.net/forum?id=X6D9bAHhBQ1) makes the
correct conceptual split:

```text
decision state --action--> deterministic afterstate
afterstate --chance--> next decision state
```

It performed strongly in 2048 and backgammon, where deterministic MuZero's
single transition model is misspecified. But Cascadia already owns an exact
simulator and exact bag distribution. Learning those dynamics would add model
bias and a second rules implementation without removing any uncertainty.

**Transfer:** copy the afterstate/chance-node ontology.  
**Do not copy:** a learned dynamics model for rules we can execute exactly.

This also argues against prioritizing [DreamerV3](https://www.nature.com/articles/s41586-025-08744-2).
DreamerV3's achievement is broad, stable learning over more than 150 control
tasks and Minecraft from pixels. The located paper does not provide a
comparable exact board-game result; Cascadia does not need to infer physics or
latent rules from observations.

### 3.4 Small evaluator plus lookahead is a recurring winning pattern

- [TD-Gammon](https://doi.org/10.1145/203330.203343) combined a small value
  network, self-play temporal-difference learning, structural features, and
  shallow lookahead in stochastic backgammon. Version 2.1 used 80 hidden units,
  two-ply search, and 1.5M self-play games; it lost by one point over a 40-game
  session against Bill Robertie. That sample is tiny, but the program's broader
  historical strength and strategic influence make the architecture precedent
  important.
- Stockfish's official [NNUE introduction](https://stockfishchess.org/blog/2020/introducing-nnue-evaluation/)
  reports +92.77 +/- 2.1 Elo over 60,000 games at one thread and +89.47 +/- 2.0
  over 40,000 at eight threads. [Stockfish 12](https://stockfishchess.org/blog/2020/stockfish-12/)
  then won at least ten times as many game pairs as it lost against Stockfish
  11. Chess is deterministic and zero-sum; the transferable mechanism is
  incremental evaluation inside much larger exact search.
- [DouZero](https://proceedings.mlr.press/v139/zha21a.html) uses a shared
  state-action Q network and direct Monte Carlo returns for a three-player,
  imperfect-information card game with up to 391 legal actions. It surpassed
  prior programs in days on four GPUs and ranked first among 344 Botzone
  agents. This is strong support for an explicit legal-action Q baseline; its
  enormous data requirement and return variance make it a secondary Cascadia
  bet rather than the lead design.

### 3.5 Local structure needs a global correction

Pure local n-tuples or an old-style sparse NNUE can miss long-range board
structure. The best transfer evidence says to preserve a small explicit global
path:

- [Playing Catan with Cross-dimensional Neural Network](https://arxiv.org/abs/2008.07079)
  couples a spatial hex stream with global scalar/card features. In a simplified
  two-player Catan without trading, its eight-layer agent reached 56.5% over
  10,000 games against jsettler after roughly five weeks. It is not full Catan,
  but the state-shape match—hex geometry plus nonspatial resources—is useful.
- [KataGo](https://arxiv.org/abs/1902.10565) reports a 50x self-play compute
  reduction from a collection of architecture and training improvements. In
  its ablations, removing global pooling cost a 1.60x training factor and
  removing auxiliary ownership/score targets cost 1.65x. Auxiliary component
  targets can therefore improve representation without becoming the serving
  objective.
- [HexaConv](https://arxiv.org/abs/1803.02108) shows that hexagonal and p6m
  group convolutions can share the sixfold geometry directly. Its experiments
  are vision results, not game strength, so it supports the symmetry mechanism
  rather than a Cascadia effect size.
- [AlphaGateau](https://arxiv.org/abs/2410.23753) is a recent preprint using an
  edge-featured GNN for chess state and move representation. Its internal Elo
  results strongly favor the GNN over its CNN baseline at comparable parameter
  counts, but the baseline is shallow and the ratings are internal. Treat it as
  evidence for legal moves as graph edges, not proof that a pure GNN will beat
  CascadiaFormer.
- [Three-Head Neural Network Architecture for MCTS](https://www.ijcai.org/proceedings/2018/523)
  adds an action-value head so search can back up useful estimates without
  eagerly expanding every child. In Hex it significantly beat the two-head
  counterpart and MoHex-CNN with the same core training data. This supports an
  explicit legal-action value path.

The repo supplies a direct negative control: archived ADR 0073's geometry-only
edge-aware hex GNN worsened held-out value correlation from `0.3933` to
`0.3417` and MAE from `2.5415` to `2.7982`; pairwise accuracy rose only 0.65
percentage point. That closes “replace the trunk with a geometry-only GNN.” The
component graph proposed here is narrower: an optional residual over exact
semantic components and legal actions, trained with current v3 targets and
invoked only where its measured benefit pays for its cost.

### 3.6 The simulator and sampling scheme are part of the architecture

[Pgx](https://arxiv.org/abs/2303.17503) reports 10–100x simulator throughput
over existing Python game environments by keeping board-game state transitions
vectorized in JAX on GPU/TPU. It generated 105M 9x9 Go frames in about 8.6
hours on one A100 and demonstrates Gumbel AlphaZero. DeepMind's
[Mctx](https://github.com/google-deepmind/mctx) demonstrates accelerator-native
batched tree search. Neither implements Cascadia, but they establish that the
CPU-engine/GPU-model ping-pong is an engineering choice, not a law.

[Variance Reduction in Monte Carlo Tree Search](https://papers.neurips.cc/paper/4288-variance-reduction-in-monte-carlo-tree-search)
applies common random numbers, antithetic variates, and control variates to Pig,
Can't Stop, and Dominion. Depending on game and method, the reported gains are
roughly equivalent to 25–60% more simulations; some Pig variants beat plain UCT
even when UCT received twice the simulations. The crucial condition is positive
covariance: pairing can make variance worse when the coupled outcomes move in
opposite directions.

That condition explains why the repo's generic R0.2 rollout-RNG pairing is not
contradictory evidence. R0.2 changed only the remaining rollout stream and made
gap variance 4.4% worse. The new proposal must couple the **complete physical
future chance object across root actions**, measure covariance, and fall back to
independent worlds whenever pairing is harmful.

## 4. Proposed system: Cascadia-NX

### 4.1 One-sentence design

Maintain exact, incrementally updatable structural features for each legal
afterstate; evaluate them with a tiny symmetry-tied factor network plus a small
global residual; use that cheap value inside a GPU-resident exact max-n/chance
planner whose root candidates share complete future chance bundles.

### 4.2 Dataflow

```text
exact public state
    |
    +--> exact legal compound actions
    +--> exact current score / category components
    +--> incremental habitat + wildlife motif cache
    +--> exact market and bag state
                 |
                 v
     delta features for every afterstate
                 |
        +--------+---------+
        |                  |
        v                  v
  local motif bank     small global topology encoder
  (all nodes/actions)  (root + search survivors)
        |                  |
        +--------+---------+
                 v
      own score-to-go + policy + auxiliary heads
                 |
                 v
 exact GPU decision/afterstate/chance search
 with paired counterfactual world bundles
                 |
                 v
  exact afterstate score + predicted score-to-go
```

### 4.3 Layer A: exact incremental state compiler

The rules engine should compile the public state into stable integer feature
IDs and exact summaries. An action produces a small add/remove delta rather
than a fresh token sequence.

Feature families:

1. **Local hex motifs:** center tile terrain pair, six directed neighbors,
   wildlife occupancy, allowed wildlife, keystone, frontier shape, and local
   component boundary.
2. **Card-conditioned wildlife motifs:** Bear pair/group states, Elk line/shape
   fragments, Salmon run endpoints/branch violations, Hawk isolation rings,
   and Fox distinct-neighbor sets, keyed by the active scoring card. The design
   should be generated from the scoring-card implementation so feature meaning
   cannot drift silently from the rules.
3. **Habitat topology:** exact component sizes, open boundary counts, merge
   opportunities, articulation/bridge indicators, and per-terrain competition
   summaries. Union-find alone cannot handle hypothetical deletions, but legal
   Cascadia actions only add tiles; persistent or copy-on-write merge state is
   sufficient for candidate afterstates.
4. **Market/bag/phase:** public market slots, legal wipe/refresh affordances,
   remaining public counts, turns remaining by seat, player-relative timing,
   and phase bucket.
5. **Opponent pressure:** public per-seat habitat maxima, visible boards,
   wildlife/card progress, next access to the market, and policy identity only
   where the existing self-play contract permits it.
6. **Action hyperedge:** chosen tile slot, wildlife slot, tile coordinate and
   rotation, wildlife coordinate, token expenditure, refresh decision, and the
   exact feature IDs the action adds/removes.

The first challenger should compile only the five A-card semantics because the
campaign target is all-A. “Card-conditioned” means the feature schema is
versioned against the active scoring implementation and can extend honestly;
it is not a reason to spend the first bakeoff's capacity or data on B–D cards.

Every feature transform needs exact D6 rotation/reflection tests, legality
round-trips, score invariants, and stable schema/version hashes. This is
architectural symmetry tying, not the closed 3x-cost inference TTA direction.

This dependency contract is not optional. The legacy repo attempted an
accumulator at commit `d64d32b6`: it was about 2.5x faster but regressed play
by roughly three points because placing one tile changed pairwise and pattern
features outside the nominally changed cell. It was reverted to clone and
re-extract. In Cascadia-NX every factor must declare its full dependency
footprint, and incremental updates must be property-tested against a slow full
recompute over random legal games, every action type, every D6 transform, and
adversarial component merges. A single mismatch fails closed.

### 4.4 Layer B: symmetry-tied local motif bank

Use a bank of learned n-tuple/factor embeddings rather than a dense attention
stack over every cell. Each active feature ID retrieves a small vector. Feature
vectors are summed into card-, terrain-, wildlife-, phase-, and seat-specific
accumulators, then passed through a shallow nonlinear network.

A plausible initial shape is deliberately small:

```text
feature tables -> 8 x 64-dimensional accumulators
concatenate exact/global summaries
512 -> 256 -> 64 trunk
```

The point is not the exact widths. The contract is:

- cost scales with changed features, not total board tokens;
- D6-equivalent motifs share parameters exactly;
- scoring-card experts share a trunk but retain small card-specific factors;
- action evaluation applies sparse accumulator deltas;
- current exact score is never predicted; and
- the output is remaining own score after the deterministic action.

This combines the table-lookup efficiency of 2048 n-tuples with the nonlinear
interaction capacity of NNUE. It is much smaller than CascadiaFormer-M but is
not merely CascadiaFormer-S: the engine-to-model interface and asymptotic work
per candidate are different.

### 4.5 Layer C: global topology correction

Local factors alone are not enough for habitat corridors, long Salmon chains,
crowding, market timing, or multi-seat contention. Add one small global path,
tested as a strict ablation:

- a D6-equivariant hex residual encoder over the acting board; or
- a sparse factor graph with nodes for habitat components, wildlife motifs,
  market slots, players, and candidate actions; or
- a cross-dimensional two-stream module coupling a compact hex map to global
  scalars.

The recommended first implementation is a **component graph**, not a full-cell
GNN. Component nodes compress exact topology already computed by the engine;
action nodes connect only to touched components, market slots, and wildlife
motifs. Three to five message-passing blocks at width 128 should keep the path
small and auditable.

Use the global correction at two fidelities:

1. **Fast value:** local accumulators + exact summaries, evaluated for every
   legal action and ordinary search node.
2. **Full value:** fast value + component-graph residual, evaluated at the root,
   at leaf batches, or only for sequential-halving survivors.

Stockfish 16.1's official [dual-NNUE release](https://stockfishchess.org/blog/2024/stockfish-16-1/)
provides a modern precedent for using a secondary cheap network on easy
positions. Cascadia's two-fidelity routing must be trained and measured from
scratch; it is not assumed to transfer automatically.

### 4.6 Layer D: explicit legal-action Q

Rust already enumerates the exact legal compound menu. Treat each legal action
as a first-class hyperedge and predict:

```text
Q_remain(s, a) = expected terminal own score
                 - exact own score in afterstate(s, a)

Q_serve(s, a)  = exact own score in afterstate(s, a)
                 + Q_remain(s, a)
```

The action head consumes:

- the root/global accumulator;
- the sparse delta accumulator for that action;
- the touched component embeddings;
- market slot and timing features; and
- exact afterstate score/component deltas as inputs, never targets to relearn.

This borrows DouZero's state-action formulation and the three-head MCTS idea
without reviving the closed pairwise Borda head. Pairwise margins can be an
auxiliary loss on reliable teacher pairs, but serving remains a scalar expected
own score because that is the campaign objective and composes cleanly in
search.

### 4.7 Outputs and losses

Load-bearing outputs:

- per-action own score-to-go mean;
- root own score-to-go mean;
- legal-action policy for search proposal/prior.

Auxiliary-only outputs:

- final score by wildlife/habitat/Nature category;
- four-seat final score vector;
- score distribution or quantiles;
- next-market survival / opponent next draft;
- habitat-component growth and wildlife motif completion;
- teacher uncertainty and action-ranking margins.

The category outputs are representation regularizers in the KataGo sense. They
do **not** replace the scalar mean with a forced additive category head, which
would repeat the closed structured-Q experiment. The four-seat head does not
silently change the selfish own-score objective into table utility.

## 5. Proposed planner: Counterfactual Bundle Search

**Counterfactual Bundle Search (CBS)** is a synthesis proposed here, not the
name of a published algorithm. It combines exact afterstates, common random
numbers, fixed-budget best-arm identification, and accelerator batching.

### 5.1 Complete physical world tapes

At a root, sample `K` complete future worlds. Each world contains:

- a random priority/permutation for every remaining physical tile and wildlife
  token, preserving without-replacement marginals;
- public-state-derived random variables for future market events;
- common Gumbel variables or other explicit random seeds for stochastic
  opponent policy choices; and
- an independent domain-separated stream for any residual stochastic policy
  component.

Evaluate every surviving root action in the **same K worlds**. Different
branches may consume different numbers of objects, but a full physical
priority order remains a valid marginal future for each branch. The estimator
still needs exhaustive empirical validation because cleanup rules and policy
choices can couple consumption to branch state in subtle ways.

For actions `a` and `b`, estimate the paired advantage:

```text
Delta(a, b) = mean_k [return(a, world_k) - return(b, world_k)]
```

When outcomes are positively correlated, chance luck cancels in the
difference. When covariance is zero or negative, CBS must disable pairing for
that stratum rather than force it.

### 5.2 Root allocation

Keep the campaign's proven Gumbel/sequential-halving scaffold initially:

1. enumerate the full legal root menu;
2. score all actions with the fast Cascadia-NX path;
3. retain candidates using the frozen policy/Q rule;
4. give each survivor the same initial paired world bundle;
5. eliminate by paired completed-Q at fixed registered looks;
6. spend additional worlds only on survivors; and
7. use the full global correction for the last survivors or final leaf batch.

Do not introduce OCBA, LCB, or another plug-in allocator in the first test. A
new evaluator and new coupling are already two interventions; sequential
halving supplies a clean control and is the literature-supported incumbent.

### 5.3 Tree semantics

The planner must be a real alternating decision/afterstate/chance system:

```text
seat decision -> exact deterministic placement afterstate
              -> exact public cleanup/chance transition
              -> next seat decision
```

Interior player nodes maximize that seat's own predicted final score (`max^n`),
using the exact active-seat afterstate score plus its own residual. The current
exact-K1 shortcut remains. Any K2 or deeper exact frontier must cross opponent
and chance nodes honestly; repeated one-ply own-score ranking is not an exact
solver.

Exact enumeration is appropriate only when a chance layer is genuinely small.
Otherwise sample full bundles. This is different from the closed generic
chance-node expectimax leaf correction and avoids the old NNUE failure of
adding an explicit future wildlife bonus on top of a network that already
predicted the same future score.

### 5.4 GPU residence

The high-upside systems change is to represent a batch as:

```text
roots x candidate actions x paired worlds x search frontier
```

and keep state transition, legal masks, exact incremental scoring, feature
deltas, and the evaluator on the accelerator. A CPU reference engine remains
the oracle. The GPU implementation must prove bit/exact semantic parity for
integer game state and scoring; floating-point value outputs may use a pinned
tolerance and deterministic reduction order.

This is intentionally larger than another bridge optimization. Pgx suggests
the order of magnitude available when simulation is accelerator-native, while
the repo's R2.4 result says the current request/response architecture has only
about 5% left to tune.

## 6. Training program

### 6.1 Three target sources, each with a different job

1. **Terminal trajectory returns:** broad, unbiased anchors for expected own
   final score under the behavior policy.
2. **Afterstate TD(lambda):** dense credit assignment between a player's
   successive decisions. Rewards are exact score increments; bootstraps predict
   only remaining score. This is the TD-Gammon/2048 lesson.
3. **Exact-search reanalysis:** high-quality per-action policies, paired
   advantages, and completed-Q values on hard states. This is the Expert
   Iteration/D1 lesson.

The current D1 work is therefore complementary: if its current-rules artifacts
pass their own gates, they become the best available teacher corpus for a
Cascadia-NX bakeoff. They do not by themselves prove the new architecture.

### 6.2 Prevent luck leakage

For stochastic targets:

- never label a public action with the real hidden order as if it were its
  expectation;
- average search estimates over exact conditional worlds;
- use paired action advantages only when both actions saw the same valid world;
- keep per-world returns and covariance diagnostics;
- weight absolute-Q regression by effective sample size/uncertainty without
  changing the mean target; and
- retain terminal returns as an anchor against teacher bias.

### 6.3 Phase and card specialization

2048's multistage learner is evidence that one stationary value function can be
an unnecessary compromise. Cascadia-NX should use a shared trunk with small
gated phase experts for opening, middle, and late game, plus scoring-card
experts where rules differ. Gating variables are exact public phase/card IDs,
not a learned mixture that can drift.

This is not a license for dozens of models. Start with a single trunk and three
small residual experts. Require each expert to improve a disjoint phase metric
without regressing aggregate teacher regret.

### 6.4 Search as teacher, network as amortized planner

[Expert Iteration](https://arxiv.org/abs/1705.08439) showed that a neural
apprentice trained from stronger search can substantially outperform vanilla
MCTS and ultimately surpass MoHex. [ReZero](https://arxiv.org/abs/2404.16364)
shows how batched whole-buffer reanalysis can reduce wall time in its tested
domains. The appropriate Cascadia loop is:

1. generate broad exact-rules states;
2. label a registered mixture with CBS/high-budget exact search;
3. train the cheap evaluator/action policy;
4. measure search strength and hard-state regret;
5. refresh labels only after the student changes the visited distribution; and
6. promote only on paired gameplay.

Continuous reanalysis becomes infrastructure only under the already documented
D1 rule: one positive game gate and an independent fresh-cycle replication.

### 6.5 Optional specialist teacher, single serving student

[AZ_db](https://arxiv.org/abs/2308.09175) reports that a latent-conditioned
population solved about twice as many hard chess puzzles as standard AlphaZero
and gained roughly 50 Elo when selecting specialists by opening. A Cascadia
teacher could condition specialists on wildlife-card portfolio, density versus
option value, habitat corridor style, or market-denial tendency.

Use specialists only to propose diverse candidate actions. Evaluate the union
under the same paired chance bundles and distill the result into one serving
student. This is a later diversity tool, not a checkpoint output ensemble and
not a league whose objective becomes opponent exploitation.

## 7. Why this is not the old Cascadia NNUE line

The archive contains a real warning against rediscovering an old idea and
calling it new.

It contains two distinct predecessors. The older n-tuple path had only 9,135
lookup weights over length-2/3/4 straight-line wildlife tuples and length-2/3
terrain tuples, trained by TD(0). It omitted market, bag, opponents,
scoring-card conditioning, and global component structure; the archive labels
it superseded and preserves no qualified strength result. The later NNUE was
much richer but still did not achieve correct incremental production updates.

The qualified pre-v3 evaluator was a sparse binary network:

```text
11,231 inputs -> 512 ReLU -> 64 ReLU -> scalar
5,783,681 parameters
```

Relevant historical evidence, all under older code/rules identities:

- direct NNUE play plateaued around **90.7** in the documented v3/v4 training
  lines; auxiliary heads accelerated convergence but did not lift the ceiling;
- a reproduced v1 K32/R600 policy scored **95.895 over 50 games**;
- the exact MLX port scored **95.800 over 10 fresh games**, versus 92.275 for
  the paired strong control;
- the qualified canonical V2 teacher scored **96.350 over 10 games**;
- doubling K32 search from R600 to R1200 produced only +0.167 over three pilot
  games while more than doubling latency;
- one-ply exact wildlife-market enumeration scored 91.3 versus 90.5 NNUE-only
  over 50 games; a simplified two-ply version scored 92.4; and
- deeper wildlife-only variants fell to 90.1, 87.6, and 89.0 because explicit
  future wildlife gain was added on top of an NNUE residual that already
  contained expected future wildlife value.

Those results close “revive the 11,231-feature NNUE and add more rollouts.”
They do **not** close Cascadia-NX, whose material differences are:

| Legacy line | Cascadia-NX challenger |
|---|---|
| hand-shaped sparse binary features and one scalar residual | generated, scoring-card-conditioned motif factors plus exact topology summaries |
| one board/value path | local incremental path + explicit global component-graph correction |
| implicit action comparison through afterstate scalar | first-class full legal compound-action hyperedges and Q head |
| CPU MCE with model IPC | GPU-resident exact transitions, scoring, and batched search |
| largely independent/random rollout estimates | complete paired physical chance bundles, covariance-gated |
| older self-play/terminal labels | current v3 exact-search reanalysis, paired advantages, and afterstate TD |
| ad hoc explicit future bonus plus full remaining-value NNUE | strict reward accounting: exact increment **or** remaining residual, never both |
| old rules identities and low-90s direct policy | head-to-head challenger against the current-rules v3 incumbent |

The legacy system is prior evidence and a baseline implementation resource,
not a reason to skip the new architecture or to assume it will work.

## 8. Falsifiable bakeoff

No architecture should receive a full training campaign because its diagram is
appealing. The bakeoff should isolate representation, global context, and
search economics in that order.

### 8.1 Frozen arms

| Arm | Evaluator | Purpose |
|---|---|---|
| T | frozen CascadiaFormer-M incumbent | current representation and serving control |
| N0 | local motif/NNUE accumulator only | test the cheap local core |
| N1 | N0 + exact global summaries/MLP | price nonspatial global context |
| N2 | N1 + component-graph residual | test long-range topology |
| N3 | N2 + two-fidelity survivor routing | measure wall-matched planner value |

All arms use the same current-rules states, legal actions, teacher targets,
train/validation/test split, optimizer exposure accounting, and exact
afterstate serving identity. Hyperparameter selection and final verdict blocks
remain disjoint.

### 8.2 Offline metrics

Report by phase, action width, scoring-card family, and teacher uncertainty:

- selected-action and all-action completed-Q RMSE/MAE;
- pairwise accuracy and regret against high-budget teacher Q;
- top-1/top-k teacher action recall on the **full legal menu**;
- calibration of predicted score-to-go, never total score without exact
  grounding;
- error correlation across sibling actions, because shared bias can preserve
  ranking;
- incremental feature parity and D6-equivariance;
- inference rows/second and end-to-end root decisions/second;
- fraction of nodes requiring the full global path; and
- memory footprint and accelerator occupancy.

The architecture only earns search integration if it changes the frontier,
for example either:

- at least 5x end-to-end leaf throughput with teacher regret no worse by more
  than 0.02; or
- at least 2x throughput with teacher regret improved by at least 0.01.

Those example bars are **engineering judgments to preregister**, not published
constants. The exact bar should be frozen before the held-out architecture
outputs are inspected.

### 8.3 Counterfactual Bundle Search preflight

Before gameplay, prove:

1. one million or more sampled market/bag transitions match the CPU engine's
   exact marginal probabilities within preregistered tolerances;
2. paired and independent estimators agree in mean on untouched roots;
3. paired covariance is positive in the strata where CBS is enabled;
4. variance of top-action differences falls materially—suggested bar at least
   20%, matching the earlier R0.2 ambition;
5. action-selection error against a very-high-world reference falls at fixed
   wall time;
6. deterministic replay reproduces every state/action/score digest; and
7. an automatic stratum-level fallback disables harmful coupling.

If the complete bundle does not clear the variance bar, close it. Do not keep
it because common random numbers worked in other games.

### 8.4 Search and gameplay gates

Use two orthogonal comparisons:

1. **Equal-search comparison:** same simulations/worlds to test evaluator
   quality.
2. **Equal-wall comparison:** spend Cascadia-NX's measured savings on more
   exact search to test the actual system thesis.

Only a candidate that survives both the locked puzzle-bank screen and a fresh
current-rules paired gate advances. Promotion still requires at least 100
paired games with the 95% interval excluding zero. The project goal is then
tested separately over 1,000 four-player games under one pinned identity, with
mean seat score at least 100. No validation loss, teacher regret, throughput
gain, or cross-game literature result substitutes for that gate.

## 9. Ranked portfolio of bold alternatives

### Rank 1 — Cascadia-NX + Counterfactual Bundle Search

**Why first:** it simultaneously attacks representation overhead, stale/local
value bias, and stochastic comparison variance while preserving exact rules.
It is the only proposal here that creates a plausible new wall-clock frontier
rather than moving a knob on the existing one.

**Largest risk:** local factors may repeat the old NNUE ceiling and the global
correction may erase the speed gain. The N0/N1/N2 ablation is designed to expose
that immediately.

### Rank 2 — search-free Deep Monte Carlo action-Q

Train a DouZero-style shared state-action Q model directly on terminal own
score, with full legal compound-action encodings and massive parallel actors.
Serve without a tree, or use it only as a cheap rollout policy.

**Why bold:** it deletes search and avoids bootstrapped tree bias.  
**Why second:** Cascadia returns are noisy, the repo already has a strong search
teacher, and DouZero's success depended on enormous data. This is a valuable
radical baseline and potential rollout policy, not the best primary route to
100.

### Rank 3 — pure component/action graph AlphaZero

Represent habitat components, wildlife patterns, market items, players, and
legal actions as a heterogeneous graph; run a small edge-featured GNN and
Gumbel search.

**Evidence:** Catan XdimRes, AlphaGateau, scalable GNN AlphaZero, and the
three-head action-value result.  
**Risk:** message passing can be slower than attention at small graph sizes and
can oversmooth; AlphaGateau's strongest claims are preprint/internal-rating
evidence. Prefer the graph as Cascadia-NX's global correction before making it
the whole evaluator.

### Rank 4 — de-noised regret curriculum

[Regret-Guided Self-play Curriculum](https://rlg.iis.sinica.edu.tw/papers/rgsc/)
reports average gains of 77 Elo over AlphaZero and 89 over Go-Exploit across Go,
Othello, and Hex; in mature 9x9 Go its KataGo win rate increased from 69.3% to
78.2%. For Cascadia, define state priority by high-budget **paired-world**
teacher regret rather than realized outcome regret:

```text
R(s) = Q_teacher(s, best action) - Q_teacher(s, played action)
```

Exclude roots whose apparent regret is dominated by aleatoric world variance.
This is a strong post-D1 data curriculum, but it does not by itself supply a
new evaluator/search architecture.

### Rank 5 — specialist proposal population, single distilled student

Train latent or explicitly conditioned strategic specialists, let them propose
actions, adjudicate the union with paired exact search, and distill one student.
This can break a self-play attractor without paying ensemble serving cost.

### Conditional — Monte Carlo Graph Search

[Monte Carlo Graph Search](https://arxiv.org/abs/2012.11045) reports 30–70%
memory reduction and strength gains in chess/crazyhouse by merging
transpositions. Cascadia's market/bag state may make exact transpositions rare.
Instrument canonical collision and subtree-reuse rates first; build a DAG only
if the measured rate justifies the complexity.

### Moonshots, not first bets

- **Monte Carlo *-Minimax / Star pruning:** designed for densely stochastic
  games and potentially useful with Cascadia score bounds, but requires a
  correct max-n adaptation. Primary source:
  [Monte Carlo *-Minimax Search](https://arxiv.org/abs/1304.6057).
- **Policy Gradient Search:** potentially useful if exact measurement shows
  tree reuse is poor; its strongest located evidence is deterministic Hex.
  [Primary paper](https://arxiv.org/abs/1904.03646).
- **Afterstate novelty:** use a temporary search bonus over controllable board
  motifs, never random outcomes; remove it at final root choice. Evidence is
  promising but indirect.
- **GFlowNet/POMO proposal generation:** generate diverse high-scoring
  completions as teacher proposals, then adjudicate under exact chance worlds.
  Directly optimizing lucky terminal boards would be invalid.

## 10. Approaches not recommended

### Another larger transformer

The repo already measured 88.2M versus 207M, more data, fresh initialization,
and multiple output ensembles without a strength breakthrough. A different
transformer tokenizer could still win, but it is not the clean-sheet bet with
the best combined literature and repo evidence.

### A smaller ordinary transformer plus more search

Closed on the current CUDA serving path: roughly 1.9–2.0x throughput was not
enough, and more than 3x had already failed the accuracy trade. Cascadia-NX is
eligible only because sparse deltas and GPU-resident simulation change the
per-call boundary.

### Learned MuZero/Dreamer dynamics

Cascadia has exact rules, exact score, and an exact finite bag. Learning a
dynamics model would introduce avoidable bias and a shadow rules
implementation. Use Stochastic MuZero's afterstate/chance factorization with
the real simulator.

### Pure local n-tuples or the legacy NNUE

The archive already establishes a low-90s direct ceiling and mid-90s searched
ceiling for that line. A global topology correction and current search
supervision are mandatory parts of the challenger.

### Load-bearing additive category heads

The exact experiment failed. Keep category/component prediction as an
auxiliary representation loss unless fresh, materially different evidence
opens a new serving contract.

### Risk-sensitive or cooperative serving

The goal is mean own raw score. Quantile/CVaR selection and table-total
objectives change the objective or amplify noise and have no current evidence
that they improve it. Distribution/four-seat outputs belong in training and
diagnostics.

### Blind common-random-number pairing

The repo's rollout pairing worsened variance. CBS advances only if complete
world coupling is unbiased and produces positive covariance on current roots.

## 11. Route to 100

The historical July-9 gap from the scalar champion to 100 is 1.7025 points,
but no July-16 canonical score exists yet. A plausible system-level path is not
to demand all 1.7 points from one neural checkpoint:

1. **Better targets:** let D1/current exact-search reanalysis reduce policy and
   value bias.
2. **Cheaper representation:** recover a multi-fold end-to-end leaf/root
   throughput gain without the old small-transformer accuracy loss.
3. **Paired chance planning:** spend the gain on lower-variance action
   differences rather than independent absolute estimates.
4. **Global correction:** restore the long-range structure a pure local
   evaluator would miss.
5. **Curriculum:** feed confirmed high-regret states back into the student only
   after aleatoric luck is removed.
6. **Exact frontiers:** retain K1 and expand exactness only through honest
   max-n/chance semantics.

This decomposition is a strategy, not an additive point forecast. The
interventions can overlap and their gains may not add. The bakeoff is designed
to close the proposal quickly if the new wall-clock frontier does not appear.

## 12. Recommendation

Build **one bounded Cascadia-NX architecture challenger after the live D1
chain reaches its registered boundary**, using the completed current-rules
corpus if admissible. Do not begin with a full GPU rules port. First prove on
frozen current-rules roots that:

1. sparse incremental features reproduce exact CPU afterstates and D6
   transforms;
2. local factors plus a small component graph retain or improve high-budget
   teacher regret;
3. the evaluator changes end-to-end throughput by several-fold rather than
   the 1.9x small-transformer regime; and
4. complete paired chance bundles actually reduce action-difference variance.

If all four pass, the GPU-resident planner is the highest-upside engineering
investment in the portfolio. If representation fails, stop before the port. If
pairing fails, keep the evaluator and use independent exact worlds. If the
equal-wall gameplay gate fails, close the architecture regardless of how
elegant or fast it is.

The core strategic change is simple:

> Stop asking one large network to rediscover the board, the scoring rules,
> and the stochastic comparison on every call. Make structure and chance exact;
> make evaluation incremental; make search the teacher and the variance
> reducer; use learning only for the residual that exact computation cannot
> cheaply settle.

## 13. Primary-source ledger

### Direct stochastic and score-game evidence

| Source | Evidence used |
|---|---|
| [Rzepecki, *Implementing superhuman AI for Azul board game with a variation of NNUE*](https://jakubkowalski.tech/Supervising/Rzepecki2025ImplementingSuperhuman.pdf) | Shallow NNUE shapes, incremental implementation, automated win rates, search-time response; MSc/two-player caveat. |
| [Guei, Chen, Wu, *Optimistic Temporal Difference Learning for 2048*](https://arxiv.org/abs/2111.11090) | Symmetry-shared n-tuples, afterstate TD/TC, multistage learning, expectimax ablation, 625,377 result. |
| [Hung Guei, 2048 RL dissertation](https://arxiv.org/abs/2212.11087) | Cross-study table comparing afterstate DNN, Stochastic MuZero, and n-tuple systems; comparison caveat. |
| [Schrittwieser et al., *Planning in Stochastic Environments with a Learned Model*](https://openreview.net/forum?id=X6D9bAHhBQ1) | Explicit afterstate/chance factorization and stochastic MuZero 2048/backgammon evidence. |
| [Tesauro, *Temporal Difference Learning and TD-Gammon*](https://doi.org/10.1145/203330.203343) | Compact value learner, self-play TD, structural features, shallow stochastic lookahead, human comparison. |
| [Zha et al., *DouZero*](https://proceedings.mlr.press/v139/zha21a.html) | Three-player stochastic game, explicit state-action Q, direct Monte Carlo returns, parallel actors, leaderboard result. |
| [Suphx](https://arxiv.org/abs/2003.13590) | Strong four-player stochastic-game reference; hidden-information/rank objective makes it a weak architecture match for Cascadia. |
| [Brown and Sandholm, *Superhuman AI for multiplayer poker*](https://doi.org/10.1126/science.aay2400) | Six-player stochastic imperfect-information result; CFR/search and mbb/game evidence, with an objective unlike Cascadia's. |

### Efficient evaluator and structured representation

| Source | Evidence used |
|---|---|
| [Stockfish, *Introducing NNUE Evaluation*](https://stockfishchess.org/blog/2020/introducing-nnue-evaluation/) | Incremental evaluator inside exact search; official 60k/40k-game Elo tests. |
| [Stockfish 12](https://stockfishchess.org/blog/2020/stockfish-12/) | Release-level strength result and CPU-efficient NNUE/search architecture. |
| [Stockfish 16.1](https://stockfishchess.org/blog/2024/stockfish-16-1/) | Dual-network precedent for cheap evaluation on easy positions. |
| [Wu, *Accelerating Self-Play Learning in Go*](https://arxiv.org/abs/1902.10565) | Global pooling and auxiliary target ablations; 50x combined compute reduction. |
| [Gendre and Kaneko, *Playing Catan with Cross-dimensional Neural Network*](https://arxiv.org/abs/2008.07079) | Coupled hex-spatial and global-scalar streams in a stochastic board game. |
| [Hoogeboom et al., *HexaConv*](https://arxiv.org/abs/1803.02108) | Hexagonal p6/p6m weight sharing; vision rather than game-strength evidence. |
| [Gao, Muller, Hayward, *Three-Head Neural Network Architecture for MCTS*](https://www.ijcai.org/proceedings/2018/523) | Explicit action-value head and delayed MCTS expansion in Hex. |
| [Rigaux and Kashima, *Enhancing Chess RL with Graph Representation*](https://arxiv.org/abs/2410.23753) | Edge-featured legal-move GNN and internal strength results; recent-preprint caveat. |
| [Ben-Assayag and El-Yaniv, *Train on Small, Play the Large*](https://arxiv.org/abs/2107.08387) | Board graphs, global context, and scalable AlphaZero representation. |

### Search, simulation, and training

| Source | Evidence used |
|---|---|
| [Koyamada et al., *Pgx*](https://arxiv.org/abs/2303.17503) | Accelerator-native exact board-game simulation, 10–100x throughput, Gumbel AlphaZero demonstration. |
| [DeepMind Mctx](https://github.com/google-deepmind/mctx) | Reference accelerator-native batched MCTS implementation. |
| [Veness, Lanctot, Bowling, *Variance Reduction in MCTS*](https://papers.neurips.cc/paper/4288-variance-reduction-in-monte-carlo-tree-search) | Common random numbers, antithetic/control variates, stochastic-game simulation-equivalent gains. |
| [Lanctot et al., *Monte Carlo *-Minimax Search*](https://arxiv.org/abs/1304.6057) | Sparse sampling/search in densely stochastic games. |
| [Czech et al., *Monte Carlo Graph Search*](https://arxiv.org/abs/2012.11045) | Transposition/DAG search; conditional on measured Cascadia state recurrence. |
| [Anthony, Tian, Barber, *Thinking Fast and Slow with Deep Learning and Tree Search*](https://arxiv.org/abs/1705.08439) | Expert Iteration/search distillation and Hex strength. |
| [ReZero](https://arxiv.org/abs/2404.16364) | Batched reanalysis systems precedent; indirect domain transfer. |
| [Regret-Guided Self-play Curriculum](https://rlg.iis.sinica.edu.tw/papers/rgsc/) | Regret-focused state curriculum gains; Cascadia needs paired-world de-noising. |
| [AZ_db](https://arxiv.org/abs/2308.09175) | Diverse latent specialists for teacher proposal and puzzle coverage. |
| [AlphaZero official result summary](https://deepmind.google/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/) | Residual-network/search benchmark context; not evidence that a transformer is required. |
| [DreamerV3](https://www.nature.com/articles/s41586-025-08744-2) | Broad learned-world-model result and the reason not to transfer it uncritically to exact Cascadia. |

### Repository context

- [v3 source of truth](docs/v3/README.md)
- [live campaign state](docs/v3/CAMPAIGN_STATE.md)
- [consolidated verdicts and closed directions](docs/v3/RESEARCH_LOG.md)
- [living research queue](docs/v3/RESEARCH_AGENDA.md)
- [current CascadiaFormer architecture](docs/v3/ARCHITECTURE.md)
- [existing radical directions](docs/v3/RADICAL_DIRECTIONS.md)
- [July 16 research questions](research_questions_7_16.md)
- [July 16 ten-question answer report](research_answers_7_16.md)
- archived pre-v3 recovery point:
  `archive/pre-v3-repo-cleanup-2026-07-01` (tag target
  `f9fe6c97297bcc273de82c633dce9c8fda092979`)
- archived sparse-model contract:
  `docs/archive/v2/decisions/0055-exact-mlx-port-qualified-nnue.md`
- archived gameplay and budget verdicts:
  `docs/archive/v2/decisions/0060-exact-mlx-teacher-gameplay-reproduction.md`
  and `0061-exact-mlx-rollout-budget-response.md`
- archived geometry-only GNN verdict:
  `docs/archive/v2/decisions/0073-edge-aware-hex-graph-value.md`
- exact legacy implementation/result commits: `b51ee6b9` (one-ply),
  `37f6c0e4` (two-ply), `531cfd10` (deeper wildlife), `d64d32b6`
  (failed partial accumulator), and `7947f69e` (v4-opponent NNUE)
