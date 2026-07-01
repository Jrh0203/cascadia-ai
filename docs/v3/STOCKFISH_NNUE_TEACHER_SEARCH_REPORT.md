# Stockfish NNUE Training, Search Teachers, and the Cascadia MCE Tradeoff

Date: 2026-06-20
Status: Research synthesis and methodology guidance; not a campaign authorization or experiment result.

## Executive conclusion

Stockfish does use search while producing training games. It does **not** normally
pay its maximum deployment-time search budget for every training position.
Documented Stockfish data generation instead uses bounded self-play search,
records the searched evaluation already produced while choosing each move, and
trains NNUE to approximate a blend of that searched value and the eventual game
result. At deployment, the inexpensive NNUE becomes the leaf evaluator inside a
much larger search.

The teacher is therefore usually not a separate giant network. The teacher is the
complete search procedure, often using the current NNUE at its leaves. Search
examines alternatives and backs improved values up to the root; the NNUE student
compresses that result into a fast static evaluation.

The direct Cascadia analogue is not `MCE(600)` at every self-play decision. It is:

1. high-volume direct or lightly searched game generation for broad state
   coverage and realized score-to-go targets;
2. bounded search whose estimates are saved whenever search is already used to
   choose a move; and
3. selective deep MCE annotation of informative roots, including counterfactual
   candidate afterstates, with variance-aware loss weighting.

This is materially more compute-efficient than using deep MCE merely to create a
better chosen trajectory. A deep teacher root can label many candidate actions;
a deep self-play decision produces only one chosen continuation unless its
candidate statistics are retained.

## The apparent training/inference mismatch

Cascadia currently has two very different operating regimes:

- **Bulk training-data generation:** direct NNUE inference is fast enough to
  produce many games and afterstates.
- **Strong final play:** the best policy spends a large CPU budget on candidate
  search and Monte Carlo Evaluation (MCE), currently exemplified by a top-32,
  600-rollout configuration.

Direct-only self-play is attractive because it maximizes samples per hour, but it
creates data under the state distribution of the cheaper and weaker policy. Deep
MCE self-play improves the trajectory distribution but can reduce game volume by
orders of magnitude.

This is a standard search-distillation problem: how can a cheap evaluator learn
from an expensive decision procedure without running the expensive procedure on
every training example?

## What Stockfish actually does

### 1. Search is present in documented training-game generation

Stockfish's historical training-data generator plays self-play games and collects
positions and their evaluations along the way. A documented large generation run
used:

- depth-9 search;
- the then-current NNUE evaluation inside search;
- random selection among as many as four sufficiently competitive principal
  variations;
- a shallow-pruning configuration suitable for data generation; and
- an 18-billion-position generation target, producing a cited 16-billion-position
  dataset.

The same documentation identifies fixed 5,000-node searches as another successful
generation regime and describes depth 9 as a historical quality/volume sweet
spot. These are bounded searches, not full tournament-strength searches.

Source: [Stockfish NNUE training datasets](https://github.com/official-stockfish/nnue-pytorch/wiki/Training-datasets).

Stockfish's introduction of NNUE summarized the arrangement directly: NNUE is
trained on position evaluations at moderate search depth, and the resulting
evaluation is later used inside alpha-beta/PVS search.

Source: [Introducing NNUE Evaluation](https://stockfishchess.org/blog/2020/introducing-nnue-evaluation/).

### 2. Search cost is reused as supervision

The search performed to choose a self-play move already returns a root evaluation.
That value can be written into the training record with almost no additional
search cost. Stockfish therefore receives both:

- a stronger actor, because bounded search chooses the move; and
- a dense teacher label, because the backed-up search evaluation is retained.

It does not need to play a game cheaply and then independently re-search every
visited position at maximum strength.

### 3. The target combines searched evaluation and actual result

The official NNUE trainer documentation describes converting an engine evaluation
to bounded win/draw/loss space, then interpolating it with the eventual result:

```text
target = lambda * searched_evaluation + (1 - lambda) * game_result
```

The interpolation can be applied to the target or separately to the loss terms.
The searched value supplies dense local guidance; the game result anchors the
network to realized outcomes. The documentation also notes successful Stockfish
training with a power-loss exponent around 2.6.

Source: [Official NNUE loss-function documentation](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md#loss-functions-and-how-to-apply-them).

### 4. The search teacher can use the student at its leaves

There is no contradiction in using the current NNUE inside the teacher search.
The static evaluator supplies leaf estimates, while search provides policy
improvement by:

- comparing alternative moves;
- extending tactically or strategically important lines;
- applying the game rules over future states; and
- backing the best supported continuation to the root.

The next NNUE is trained to approximate that improved, backed-up value without
performing the search itself. Repeating the process is a form of approximate
policy iteration or expert iteration, although Stockfish development is not a
single rigid AlphaZero-style loop with a canonical number of cycles.

### 5. Data diversity and external teachers matter

The documented Stockfish generator deliberately introduces near-best-move
variation rather than following one deterministic principal variation forever.
Its dataset guidance also says that broad Stockfish-generated data historically
worked well for initial learning, while later retraining on stronger Lc0-derived
data could improve an already capable network. It cautions that stronger labels
do not automatically make a better dataset: quality, coverage, and learnability
trade off and are selected empirically.

The current Stockfish repository acknowledges that its neural networks are
trained on data supplied by the Leela Chess Zero project.

Sources:

- [Stockfish NNUE training datasets](https://github.com/official-stockfish/nnue-pytorch/wiki/Training-datasets)
- [Current Stockfish repository acknowledgement](https://github.com/official-stockfish/Stockfish#acknowledgements)

### 6. Playing strength, not validation loss, decides the winner

The NNUE tooling can continuously export checkpoints and play engine matches to
rank them. This reflects an important methodological separation:

- training loss measures imitation of the selected targets;
- game matches measure whether that imitation produces a stronger search engine.

Source: [Stockfish NNUE PyTorch trainer](https://github.com/official-stockfish/nnue-pytorch#automatically-run-matches-to-determine-the-best-net-generated-by-a-running-training).

## What Stockfish does not imply

Several distinctions prevent a literal copy of the chess recipe.

### Stockfish search is not MCE

Stockfish uses deterministic alpha-beta/PVS search in a two-player,
perfect-information, zero-sum game. Cascadia has chance events, four interacting
players, and policy-dependent rollouts. MCE returns a sample mean with meaningful
Monte Carlo variance. A Cascadia teacher record therefore needs rollout count,
variance, standard error, RNG-domain identity, and opponent-policy identity in
addition to its mean.

### Better games are not the same as better counterfactual labels

An MCE-generated game moves the learner into states selected by a stronger policy,
but terminal supervision only explains the chosen trajectory. It does not directly
say that the rejected candidate was worse or by how much.

Cascadia's prior one-iteration, 15,000-game on-policy MCE(50) experiment was
approximately neutral. That result does not prove that searched training is
useless; it suggests that merely improving trajectories at substantial cost may
be a weak use of teacher compute. Capturing candidate-level estimates is the more
direct distillation signal.

### The deepest available teacher is not automatically optimal

Deeper labels reduce some forms of bias but cost coverage. They may also become
harder for a small NNUE to learn. Stockfish's own documentation describes this as
a quality-versus-learnability tradeoff and historically favored moderate search
for bulk generation.

## The teacher-student formulation for Cascadia

Let:

- `s` be a public game state from the focal player's perspective;
- `a` be one exhaustive legal action;
- `s_a` be the exact afterstate;
- `z(s_a)` be realized final score minus exact score at `s_a`;
- `mu_R(s_a)` be the MCE estimate after `R` rollout samples; and
- `se_R(s_a)` be its standard error.

The NNUE predicts remaining score-to-go:

```text
V_theta(s_a) ~= future score gained after the action
```

Action ranking remains:

```text
exact score at s_a + V_theta(s_a)
```

A blended value objective can be written as:

```text
L_value =
    w_outcome * loss(V_theta(s_a), z(s_a))
  + w_teacher * confidence(se_R) * loss(V_theta(s_a), mu_R(s_a))
```

For roots with multiple teacher-scored candidates, add a ranking or advantage
term:

```text
advantage(a) = mu_R(s_a) - mean_b(mu_R(s_b))
```

Pairwise or listwise ranking supervision teaches the decision boundary even when
absolute MCE estimates share bias. Candidate comparisons should use common random
numbers wherever the simulator permits so that paired differences have lower
variance than independent estimates.

## Recommended compute allocation

### Tier A: broad trajectory corpus

Generate the majority of games with frozen direct NNUE policies plus controlled
exploration and an opponent pool. Retain every focal afterstate with realized
score-to-go. This tier supplies:

- broad coverage;
- rare-state exposure;
- all phases of the game;
- opponent-conditioned outcomes; and
- low-cost terminal anchoring.

Direct games remain useful. They should not be treated as sufficient teacher
supervision by themselves.

### Tier B: bounded-search actor games

On a measured fraction of games, or for the newest focal seat, use a modest MCE
budget. Store all candidate estimates, sample counts, and variances already
computed during move selection. This is the closest analogue to Stockfish's
moderate-depth self-play generator.

The budget should be selected by a throughput/strength experiment rather than by
assuming that `R=50` or any other historical value is optimal. The objective is
the lowest budget that materially changes candidate ordering relative to direct
NNUE while preserving useful game volume.

### Tier C: selective deep teacher roots

Spend the full teacher budget only on a stratified and information-rich subset of
roots. Prioritize:

- small margins between the direct NNUE's top candidates;
- disagreement between direct NNUE and a cheaper search;
- high legal-action width;
- early and middle turns where errors compound;
- rare animal, terrain, market, Nature Token, and Pinecone contexts;
- underrepresented phase/score strata; and
- positions where a pilot allocation indicates that additional rollouts can
  resolve the ranking.

Deep annotation should score multiple candidate afterstates. Sequential allocation
can stop spending on candidates that are clearly inferior and concentrate samples
on unresolved contenders, provided the recorded estimator and selection procedure
remain statistically auditable.

### Suggested priority order

When compute is constrained, the order of value is generally:

1. preserve enough broad games to prevent distribution collapse;
2. retain every search statistic already paid for during generation;
3. deeply label close and consequential candidate sets;
4. add more deep-teacher roots; then
5. increase deep-search quality on already easy or redundant roots.

## Relationship to the Cascadia V3 research specification

The current V3 design already contains the core Stockfish-style separation:

- broad bootstrap and expert-iteration games provide realized score-to-go;
- a stratified subset of roots receives K32/R600 teacher labels;
- teacher records preserve candidate mean, variance, count, and rank;
- the target blends teacher and realized score-to-go;
- NNUE remains the fast inference evaluator inside stronger search; and
- promotion depends on paired games, not training loss alone.

The main methodological refinement to preserve is that teacher compute should be
evaluated by **useful counterfactual labels per CPU-hour**, not merely by the mean
score of the games it generates. If bounded-search actor games are introduced,
their candidate statistics must be persisted and trained on; otherwise much of
the search expense is discarded.

The campaign specification remains the source of truth for exact Phase 2 sample
counts, budgets, gates, and authorization:
[CASCADIA_V3_RESEARCH_SPEC.md](CASCADIA_V3_RESEARCH_SPEC.md).

## Recommended experimental comparison

Before changing a campaign-scale collection policy, compare three compute-matched
arms:

| Arm | Actor | Deep annotation | Purpose |
|---|---|---|---|
| A | Direct NNUE | Stratified roots | Current high-volume baseline |
| B | Light MCE | Same stratified roots | Tests whether better state distribution helps |
| C | Direct NNUE | More counterfactual roots using saved actor compute | Tests whether labels beat trajectory quality |

Hold total CPU time, frozen model, opponents, seed domains, trainer exposures, and
evaluation protocol constant. Compare:

- quantized validation loss on realized and teacher targets separately;
- top-1 and top-K agreement with the deep teacher;
- regret of the direct policy under deep teacher estimates;
- direct-policy playing strength;
- K32/R600 playing strength;
- state-distribution coverage; and
- useful candidate labels produced per CPU-hour.

This experiment answers the actual allocation question: whether the next unit of
compute is better spent improving the games, improving the labels, or increasing
coverage.

## Practical rules

1. **Do not run full MCE merely to obtain a final-score label.** The completed game
   already supplies that label.
2. **Never discard candidate estimates produced during action selection.** They
   are the search teacher's primary supervision.
3. **Use deep MCE mainly for counterfactuals and ambiguous decisions.** This is
   where search adds information unavailable from the realized trajectory.
4. **Freeze actor and teacher identities within a shard.** Record model, opponent
   pool, rollout policy, search parameters, and RNG domains.
5. **Weight teacher labels by uncertainty.** Equal treatment of an `R=8` noisy
   estimate and an `R=600` stable estimate is statistically dishonest.
6. **Keep outcome anchoring.** Search targets can inherit evaluator and rollout
   bias; realized score-to-go prevents pure self-imitation.
7. **Use playing-strength promotion gates.** Lower regression loss does not prove
   a stronger direct policy or a stronger search policy.
8. **Treat actor quality and label quality as separate experimental variables.**
   Changing both simultaneously makes the result uninterpretable.

## Bottom line

Stockfish resolves the search-cost dilemma through **bounded-search generation,
search-value distillation, massive coverage, outcome anchoring, and game-based
promotion**. It does not require production-strength search on every training
position.

For Cascadia, broad NNUE self-play should continue to provide coverage, but the
most valuable use of scarce MCE compute is to create reusable candidate-level
teacher supervision. Light MCE can improve the actor distribution; deep MCE should
be concentrated on informative roots and amortized over many candidate afterstates.
That is the closest faithful translation of the Stockfish teacher paradigm to a
stochastic four-player game.

## References

1. Stockfish. [Introducing NNUE Evaluation](https://stockfishchess.org/blog/2020/introducing-nnue-evaluation/), 2020.
2. Stockfish NNUE PyTorch Wiki. [Training datasets](https://github.com/official-stockfish/nnue-pytorch/wiki/Training-datasets), historical dataset-generation guidance, last modified 2023-01-05 in the indexed wiki snapshot.
3. Stockfish NNUE PyTorch. [NNUE technical and training documentation](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md).
4. Stockfish NNUE PyTorch. [Trainer repository and checkpoint match tooling](https://github.com/official-stockfish/nnue-pytorch).
5. Stockfish. [Current source repository and Lc0 data acknowledgement](https://github.com/official-stockfish/Stockfish).
