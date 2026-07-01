# Cascadia Feature Representation Audit and V3 Research Program

Date: 2026-06-16

Ruleset: four-player AAAAA, no habitat bonuses

Scope: legacy V1 champion, V2 entity and complete-action pipelines, and a
from-first-principles representation program for reaching 100+ mean

Compute boundary: john1, john2, john3, and john4; local Apple Silicon only

## Executive Verdict

The repository does not primarily have a "network too small" problem. It has an
**information organization problem**.

The legacy V1 champion is fast and battle-tested, but it represents the game as
a large hand-bucketed sparse vector. It contains useful Card A concepts and its
only major recent strength gain came from restoring opponent detail, yet it
still:

- omits the playable frontier from the champion schema;
- compresses the unseen supply into marginals;
- encodes geometry through absolute cells and partial hand-built summaries;
- provides no explicit state-action relation;
- does not enforce full hex symmetry; and
- contains a confirmed feature-layout defect in the `mid-features` tail.

V2 made the right foundational move: it stores typed board and market entities,
all four public boards, exact complete actions, staged market context, and MLX
friendly tensors. The raw records are substantially cleaner than V1. The models
then repeatedly compress those records before the decision-critical
relationships are formed:

- boards become mean/max summaries;
- candidate sets become mean/max summaries;
- public supply becomes marginal counts;
- local geometry stops at radius one;
- hierarchical descendants become min/mean/max statistics; and
- complete actions are mostly compared pointwise rather than relationally.

That explains an otherwise confusing body of evidence. Geometry-only models
failed, wider pointwise models failed, and local feature additions often fit
training shortcuts without generalizing. At the same time:

- detailed opponent features added **1.33 points** to V1;
- exact local geometry made the largest measured contribution to conditional
  tile specialization;
- the conditional action hierarchy has a **99.18% validation oracle recall**
  at a mean of 482.4 proposals; and
- the full-legal audit assigns **0.254 points per decision** to proposal regret,
  versus 0.095 to within-frontier selection.

The representation with the best chance of changing the regime is therefore:

> **A typed, multi-resolution Relational Opportunity Graph that preserves exact
> public state, exact semantic supply, frontier affordances, habitat
> components, wildlife scoring motifs, opponent demand, and complete-action
> identity through the final comparison layer.**

The model should encode the state once, retrieve actions hierarchically, and
apply rich candidate-to-state and candidate-to-candidate reasoning only to a
small retained set. Rust remains authoritative for rules, legality, graph
construction, and exact targets. MLX owns batched representation learning and
inference.

This is not a recommendation to discard V1's domain knowledge. The strongest
design combines:

1. V1's useful exact scoring concepts and sparse incremental instincts;
2. V2's typed, lossless public-state and complete-action records; and
3. modern relational, equivariant, set, graph, and value-equivalent learning.

## Current Strength Context

The qualified player scores **95.744** over 1,000 held-out games and 4,000
seats, with a game-block 95% interval of `[95.652, 95.837]`. The measured gap to
100 is 4.256 points. See
[final strength validation](final-strength-validation.md).

Its score anatomy matters:

| Component | Qualified mean | Relative observation |
|---|---:|---|
| Habitat | 30.878 | Strong |
| Bear | 11.363 | Strong |
| Elk | 10.546 | Trails the canonical V2 control by 1.262 |
| Salmon | 12.774 | Modestly ahead |
| Hawk | 11.380 | Trails the canonical V2 control by 0.815 |
| Fox | 14.920 | Strong |
| Nature Tokens | 3.883 | Roughly neutral |

The remaining gap is not "learn the score function." Cascadia's terminal score
is exact and already implemented. The difficult problem is representing:

- which incomplete plans remain feasible;
- how multiple plans compete for the same cells and wildlife;
- what the unseen supply can still deliver;
- what opponents are likely to remove before the next turn;
- which complete actions preserve high-value future options; and
- which apparently different actions are equivalent or dominated.

The full-legal audit reinforces this diagnosis:

| Quantity | Result |
|---|---:|
| Mean champion decision regret | 0.350 |
| Proposal/frontier regret | 0.254 |
| Within-frontier selection regret | 0.095 |
| Current top-64 recall | 89.904% |
| Top-64 plus champion-frontier union recall | 99.327% |
| First observed width exceeding 98% recall | 1,024 |

See [full-legal decision regret audit](full-legal-decision-regret-audit-v1.md).
Representation work should first improve action retrieval and confidence-set
coverage, not merely shave error from a scalar leaf value.

## Method and Evidence Boundary

This report was built from:

- direct inspection of the active Rust and MLX representation paths;
- the qualified gameplay, full-legal, identifiability, local-geometry,
  edge-aware, and hierarchical-retrieval artifacts;
- a complete enumeration of aggregate-supply signatures for all unordered
  distinct pairs in the 85-tile catalog;
- an activation-range audit of the open graded-oracle train and validation
  corpora; and
- primary research papers and official Stockfish documentation.

The sealed graded-oracle test split was not used. Negative local experiments
are interpreted only within their actual architecture, target, and data
regime. Proposed score upside ranges are explicit research priors, not measured
results.

## Audit Standard

This audit evaluates a feature representation against nine questions.

1. **Fidelity:** Can two strategically different public states collide?
2. **Relational access:** Can the model directly see the relation needed for
   the decision, or must it reconstruct it through several compressions?
3. **Action conditioning:** Does the representation explain what a candidate
   changes relative to the parent state?
4. **Symmetry:** Are equivalent rotations and reflections guaranteed or merely
   sampled?
5. **Hierarchy:** Are tiles, components, motifs, plans, players, and actions
   represented at their natural scales?
6. **Stochastic sufficiency:** Does the state determine the correct public
   distribution over refills and future availability?
7. **Opponent sufficiency:** Can it express who wants each market item and when
   they act?
8. **Computational leverage:** Can the state be encoded once and reused across
   thousands of legal actions?
9. **Auditability:** Are schema ownership, index ranges, activation rates,
   invariances, and compatibility explicit and tested?

## Representation Scorecard

| Property | V1 champion | V2 base entity | V2 graded oracle | Proposed V3 |
|---|---|---|---|---|
| Exact occupied boards | Partial hand layout | Yes, all four | Yes, all four | Yes |
| Exact legal action | No | Afterstate dependent | Yes | Yes |
| Exact staged prelude | No | No | Yes | Yes |
| Exact unseen tile multiset | No | No supply | No, marginals | Yes, semantic archetype counts |
| Frontier cells | No champion block | Implicitly derivable | Implicitly derivable | First-class tokens |
| Habitat components | Largest only plus hand summaries | Implicit | Implicit | First-class nodes |
| Wildlife motifs | Hand buckets | Implicit | Radius-one helper only | First-class hyperedges/state machines |
| Opponent boards | Summaries | Full boards | Full boards | Full boards plus demand links |
| Market interaction | Slot buckets | Market entities | Parent/staged entities | Item, demand, survival, and supply graph |
| State-action geometry | Indirect afterstate | Indirect | Learned from q/r cross-attention | Explicit edit and relation edges |
| Candidate-set interaction | None | None | Mean/max context | Relational retained-set attention |
| Full D6 symmetry | No | No | C6 augmentation only | Exact D6 contract |
| Incremental potential | Excellent | Moderate | Moderate | State cache plus local graph deltas |
| Schema auditability | Fragile integer layout | Has schema hash | Has schema hash | Manifest plus activation and invariance census |

## V1: What It Represents

The champion-compatible implementation is in
[`legacy/crates/cascadia-ai/src/nnue.rs`](../../../legacy/crates/cascadia-ai/src/nnue.rs).
The qualified MLX player is an exact port of the legacy `mid-features,v4-opp`
network:

```text
11,231 sparse binary features -> 512 -> 64 -> scalar
```

The first 5,197 features are the original board representation:

| Block | Features | Meaning |
|---|---:|---|
| Per-cell state | 4,851 | 441 cells x wildlife/tile/primary-terrain state |
| Phase and own summary | 110 | Turn, Nature Tokens, wildlife counts, largest habitats |
| Wildlife pair adjacency | 147 | Three hex line directions and wildlife pair states |
| Pattern summaries | 89 | Bear, Elk, Salmon, Hawk, Fox hand-built aggregates |

The V1 extension reaches 7,670 features by adding:

- wildlife bag counts;
- maximum opponent habitat by terrain;
- allowed-wildlife bits on every occupied cell;
- higher-resolution own wildlife counts; and
- terrain edge-pair summaries.

The V2 legacy extension reaches 10,561 with:

| Added block | Features |
|---|---:|
| Secondary terrain per cell | 2,205 |
| Extended habitat buckets | 70 |
| Extended wildlife-count buckets | 55 |
| Wildlife extension-capacity buckets | 40 |
| Pattern V2 summaries | 48 |
| Extended wildlife-bag buckets | 105 |
| Extended maximum-opponent habitat | 70 |
| Market slots | 88 |
| Unseen tile terrain marginals | 105 |
| Unseen tile wildlife-capacity marginals | 105 |

The `v4-opp` block then contributes 369 features:

```text
3 opponents x (
  5 wildlife counts x 11 bins
  + 5 habitat sizes x 11 bins
  + Nature Tokens x 9 bins
  + 4 pattern flags
)
```

### V1 Strengths Worth Preserving

V1 has several virtues that should survive into V3.

1. **Sparse, incremental evaluation.** The representation was designed for
   repeated afterstate scoring and remains a useful systems model.
2. **Rule-aware inductive bias.** Bear singleton, Elk line, Salmon run, Hawk
   isolation, Fox diversity, and habitat concepts are not arbitrary feature
   engineering. They are compact descriptions of the fixed Card A objective.
3. **Opponent evidence.** The 369-feature opponent block produced the only
   recent step-function strength increase, +1.33 points in the historical
   head-to-head.
4. **Backward-compatible loading.** Append-only zero padding made experiments
   cheap, although the same mechanism also hid layout problems.
5. **Cheap exact probes.** A sparse feature can be activation-counted,
   permuted, ablated, and inspected more easily than a latent token.

### Confirmed V1 Layout Defect

`NUM_FEATURES_MID` is documented as:

```text
NUM_FEATURES_V2
+ 150 extended tile-terrain supply features
+ 150 extended tile-wildlife supply features
+ 1 overflow-used feature
= 10,862
```

Extraction does not use that layout. `extract_v2_features` appends the
34,398-feature per-cell adjacency block immediately after feature 10,560. The
extended supply and overflow blocks are placed only after the full adjacency
block, beginning near feature 44,959.

The mid build truncates at 10,862. Its final 301 non-opponent columns are
therefore:

```text
the first 301 per-cell adjacency columns
```

They are not the documented 150 + 150 + 1 supply tail.

Because each adjacency cell occupies 78 columns, the accidental tail covers:

- all adjacency states for grid cells 0, 1, and 2; and
- 67 of 78 states for grid cell 3.

Those cells correspond to the far corner beginning at axial coordinates
`(-10, -10)`. In the open graded-oracle corpus audited for this report:

| Split | Decisions | Candidates | Maximum occupied coordinate magnitude | Maximum action coordinate magnitude |
|---|---:|---:|---:|---:|
| Train | 560 | 2,135,111 | 5 | 6 |
| Validation | 240 | 860,203 | 4 | 5 |

No occupied tile reaches the accidental cells, so the 301 columns are
effectively dead in this corpus. The intended higher-resolution supply and
overflow signals are absent from the champion-compatible schema.

This must **not** be repaired by changing extraction under the existing schema
or checkpoint. That would silently reinterpret trained weights. The correct
fix is a new schema identifier with an explicit remap, zero-initialized new
columns, and retraining or stable low-rate fine-tuning.

### Additional V1 Holes

#### 1. The champion has no explicit frontier

The model describes occupied cells but not the empty legal cells where the next
tile can go. It must infer frontier existence and local affordance from
absolute occupied-cell patterns. Later experimental `v5-feat` code adds
frontier, habitat structure, richer opponent patterns, and joint supply
features, but those blocks are not part of the qualified champion and should
be treated as unvalidated research branches.

#### 2. Supply is represented by marginals

Terrain counts and wildlife-capacity counts do not preserve which properties
co-occur on the same unseen tile. A Mountain/Hawk tile plus River/Bear tile is
not interchangeable with Mountain/Bear plus River/Hawk when current plans need
a particular joint combination.

#### 3. Absolute sparse cells spend capacity on irrelevant coordinates

The 21 x 21 grid offers simple indexing, but games occupy a small moving region.
Most cell rows are cold, translation augmentation is required, and a small
layout mistake can create permanently dead capacity.

#### 4. Symmetry coverage is incomplete

Legacy augmentation implements 0, 120, and 240 degree rotations plus
translations. It does not enforce all six rotations or reflections. The
physical game and Card A scoring are invariant to the full dihedral group D6.

#### 5. Opponent ordering is not actually relative for every perspective

`BagInfo::from_game_for_player` claims to order opponents by relative seat, but
iterates absolute player IDs from zero while skipping the focal player. For
focal seat zero this happens to match relative order. For focal seat one it
produces `[0, 2, 3]` instead of `[2, 3, 0]`. A new schema should fix the order
and explicitly test every focal seat. Old weights must retain historical
semantics.

#### 6. Candidate afterstates inherit parent context

The V1 NNUE candidate path applies a move to a board but commonly reuses parent
`BagInfo` for all candidates. The model sees board and token consequences well,
but not a fully candidate-specific staged market and supply transition. That is
a serious limitation for independent drafts, replacements, and paid wipes.

#### 7. Opponent compression remains severe

The v4 block restored wildlife, habitat, token, and four pattern signals, but
not opponent frontier, motif feasibility, current market demand, or time until
each opponent acts. The historical +1.33 gain is evidence that further
opponent-market structure is still underrepresented.

## V2: What It Represents

V2 has three relevant representation layers rather than one.

1. `compact-entity-v2`: reusable public positions.
2. Action-ranking records: early compact candidate features.
3. Complete-action graded-oracle records: lossless actions, staged context,
   supply, and multifidelity teacher labels.

### V2 Base Position Record

The Rust schema is in
[`crates/cascadia-data/src/lib.rs`](../../../crates/cascadia-data/src/lib.rs)
and the MLX decoder is in
[`python/cascadia_mlx/dataset.py`](../../../python/cascadia_mlx/dataset.py).

Each fixed 864-byte record contains:

- four relative-seat boards;
- up to 23 tiles per board;
- four market entities;
- turn, player count, board counts, and Nature Tokens;
- all players' wildlife counts and largest habitats;
- scoring cards and habitat-bonus flag; and
- eleven exact score-component targets.

Each board entity decodes to 31 values:

```text
q, r
+ primary terrain one-hot
+ secondary terrain or none
+ rotation
+ allowed wildlife mask
+ placed wildlife or none
+ keystone
```

The 96 global values include phase, turns remaining, player count, per-player
counts, per-player largest habitats, market wildlife, scoring cards, bonus
flag, and market-wildlife diversity.

This is a substantial improvement over V1:

- all opponent boards are retained;
- tile terrain, rotation, wildlife eligibility, occupancy, and keystone status
  are explicit;
- boards are ordered relative to the focal seat; and
- serialization is schema-hashed and round-tripped.

The base record still omits unseen public supply. Under the fixed AAAAA
four-player experiment, several global channels are constant and consume model
capacity without adding information.

### V2 Base Encoder

[`python/cascadia_mlx/model.py`](../../../python/cascadia_mlx/model.py) applies
self-attention separately within each board and within the market. Every board
then becomes a concatenated mean and maximum. The market does the same. Four
board summaries, one market summary, and the global projection are
concatenated for the value head.

This creates four losses:

1. no board-to-board attention before pooling;
2. no market-to-board attention before pooling;
3. no explicit hex relation or directional bias; and
4. non-injective mean/max compression.

A simple scalar example proves the last point:

```text
set A = {0.00, 0.50, 1.00}
set B = {0.25, 0.25, 1.00}

mean(A) = mean(B) = 0.50
max(A)  = max(B)  = 1.00
```

The learned entity projection can avoid some collisions, but mean/max does not
guarantee preservation of multiplicity, topology, or higher-order relations.
The Deep Sets and GIN literature reaches the same conclusion from a general
expressivity perspective.

### V2 Edge-Aware Graph Experiment

The edge-aware H6 model added exact adjacency and oriented matching-terrain
relations with four message-passing blocks. It was rejected:

| Metric | Set baseline | Hex graph |
|---|---:|---:|
| Final-score correlation | 0.3933 | 0.3417 |
| Final MAE | 2.5415 | 2.7982 |
| Pairwise accuracy | 64.7406% | 65.3890% |
| Pairwise log loss | 0.7628 | 0.7296 |

See [edge-aware validation](edge-aware-hex-score-to-go-v2-validation.md).

This closes **geometry-only message passing on the existing single-trajectory
H6 target and pooled architecture**. It does not establish that graph
representations are useless. The graph lacked exact supply, frontier nodes,
motif hyperedges, candidate identity, opponent demand, global structural
encodings, and a strong counterfactual target.

### Early Action-Ranking Record

The compact action-ranking schema stores a candidate afterstate and 52 raw
action values. It loses paid-wipe order and grouping by reducing the sequence
to:

- wipe count;
- union of affected slots; and
- total wiped slots.

Different replacement sequences can produce different staged markets and
supply transitions. The later graded-oracle schema correctly supersedes this
for complete-action work.

### Complete-Action Graded-Oracle Record

The lossless schema is implemented in
[`crates/cascadia-data/src/graded_oracle.rs`](../../../crates/cascadia-data/src/graded_oracle.rs)
and decoded by
[`python/cascadia_mlx/graded_oracle_dataset.py`](../../../python/cascadia_mlx/graded_oracle_dataset.py).

It preserves:

- exact ordered wildlife-wipe masks;
- replacement choice;
- same-slot or independent draft;
- market tile and wildlife slots;
- tile semantics and stable tile ID;
- tile coordinate and rotation;
- optional wildlife coordinate;
- staged Nature Tokens;
- exact staged market;
- staged public supply;
- immediate score and component deltas;
- candidate hash; and
- R600, R1200, and R4800 targets with sample uncertainty.

This record is the correct substrate for future action learning. Its storage is
lossless with respect to a complete legal turn. The remaining issues are in the
public-state abstraction and model consumption.

### Public Supply Is Not Injective

The 30 public-supply values encode:

```text
5 wildlife-bag counts
+ 5 unseen terrain capacities
+ 5 unseen wildlife capacities
+ 5 unseen keystones by terrain
+ 10 unseen dual-terrain-pair counts
```

See
[`crates/cascadia-data/src/public_supply.rs`](../../../crates/cascadia-data/src/public_supply.rs).

These are useful sufficient statistics for many heuristics, but not for the
actual refill distribution. They discard the joint identity of unseen tile
properties.

An exhaustive adversarial audit of all 3,570 unordered distinct pairs from the
85-tile catalog found:

| Quantity | Count |
|---|---:|
| Distinct tile pairs | 3,570 |
| Unique 30-value aggregate signatures | 2,492 |
| Signatures with collisions | 851 |
| Tile pairs participating in a collision | 1,929 |
| Disjoint colliding-pair comparisons | 577 |

One exact collision is:

```text
{ID 1: Mountain/Hawk keystone, ID 24: River/Bear keystone}

versus

{ID 2: Mountain/Bear keystone, ID 20: River/Hawk keystone}
```

Another is:

```text
{ID 0: Mountain/Hawk keystone,
 ID 40: Mountain/Wetland with Bear+Salmon}

versus

{ID 2: Mountain/Bear keystone,
 ID 39: Mountain/Wetland with Salmon+Hawk}
```

Each pair has identical terrain capacity, wildlife capacity, keystone-terrain,
and dual-terrain aggregate counts. The next-tile distributions differ.

This pair audit is a combinatorial counterexample, not an estimate of collision
frequency in live positions. It proves that the current representation cannot
in principle recover the exact public refill belief.

The proper representation is a count per **semantic tile archetype**, merging
only tiles that are genuinely identical in terrain, allowed wildlife,
keystone, and orientation behavior. Stable serialization IDs should not be
treated as ordered numerical content.

### Scalar Tile ID Is a Representation Smell

The graded action decoder includes `tile_id / 84`. The catalog explicitly says
IDs are stable serialization identifiers, not semantic ordinals. A linear
distance between tile 4 and tile 5 has no strategic meaning.

The ID can remain in lossless storage and hashing. The model should receive:

- semantic tile attributes;
- an optional categorical archetype embedding; or
- no ID at all when the semantic fields are complete.

### Graded-Oracle Model Compression

[`python/cascadia_mlx/graded_oracle_model.py`](../../../python/cascadia_mlx/graded_oracle_model.py)
is considerably richer than the base encoder:

- board and market self-attention;
- parent public-supply projection;
- per-candidate staged market and supply;
- action queries cross-attending all board tokens;
- action queries cross-attending staged market tokens;
- explicit action x parent interaction; and
- predicted residual and standard error.

The critical bottlenecks remain:

1. Each board is mean/max pooled into the parent before most interactions.
2. The parent market is pooled, not directly cross-attended by actions.
3. Seven 192-dimensional candidate factors are compressed from 1,344 values to
   192 before comparison.
4. Candidate context is only global mean, global maximum, and
   candidate-minus-mean.
5. Candidates never directly attend sibling candidates.
6. State-action geometry is learned indirectly from normalized q/r values.
7. The local helper exposes only radius-one relations.
8. Training augmentation samples six rotations but no reflections.

The broad pairwise objective can improve while exact winner retrieval remains
poor because the representation is adequate for coarse ordering but not for
the small relational distinctions near the top of a huge legal set.

### Local Geometry Evidence

The complete-action local-geometry ranker encoded:

- six tile-neighbor relations;
- the wildlife target cell; and
- six wildlife-neighbor relations.

It reduced retained regret by 17%, from 0.113024 to 0.093757, but achieved only
74.17% exact top-64 winner recall and failed every phase recall gate. See
[local-geometry rejection](complete-action-local-geometry-ranker-v1-rejection.md).

A later specialization attribution found:

| Block | Measured contribution |
|---|---:|
| Local geometry | +0.2446 |
| Descendant summary | +0.1056 |
| Tile factor | +0.0457 |

See
[conditional tile attribution](conditional-tile-specialization-attribution-v1-result.md).

The correct interpretation is:

> Geometry matters, but a radius-one feature patch is not a complete
> representation of habitat components, long wildlife motifs, supply
> feasibility, or candidate alternatives.

### Hierarchical Retrieval Evidence

The exact conditional hierarchy:

```text
prelude/draft -> tile placement -> wildlife placement
```

is structurally sound. Its widest oracle configuration reaches:

- 99.18% validation target recall;
- 95.00% exact validation decisions; and
- 482.4 mean proposals.

See [hierarchical oracle](full-legal-hierarchical-factor-oracle-v1-result.md).

The learned pilot then failed:

| Stage | Validation factor recall |
|---|---:|
| Draft | 92.84% |
| Tile | 66.57% |
| Wildlife | 100.00% |
| Integrated proposal | 72.48% |
| Winner retention | 92.08% |

See
[hierarchical retrieval result](full-legal-hierarchical-factor-retrieval-pilot-v1-result.md).

The hierarchy is not the problem. The tile-stage representation is.

The current implementation flattens the q/r-sorted position into a 3,198-value
parent vector, uses manually selected factor slices, and summarizes descendant
values by min/mean/max. Two descendant distributions can collide:

```text
{0.0, 0.0, 1.0, 1.0}
{0.0, 0.5, 0.5, 1.0}

min, mean, and max are identical.
```

They express different robustness and opportunity distributions. Sorting and
flattening also makes slot identity discontinuous: adding one tile can shift
many later entities to different input positions.

## Ranked Representation Holes

### P0: Correctness and Missing Information

1. **V1 mid-tail layout defect.** The documented supply tail is an accidental,
   effectively dead adjacency prefix.
2. **No exact unseen tile multiset.** V1 and V2 supply marginals alias different
   refill distributions.
3. **No first-class frontier.** The most decision-relevant empty cells must be
   reconstructed from occupied cells.
4. **Lossy pre-comparison compression.** Mean/max and 1,344 -> 192 compression
   occur before the hard distinctions are made.
5. **No relational candidate set.** Thousands of alternatives are scored
   mostly independently.
6. **Indirect state-action geometry.** The model must learn axial arithmetic,
   tile-edge orientation, component merging, and motif effects from generic
   coordinates.

### P1: Missing Structure

7. **No full D6 contract.** V1 uses a three-rotation subgroup; V2 uses six
   rotations without reflections.
8. **Dual-terrain edge semantics are implicit.** Exact shared-edge terrain and
   merge consequences should be provided as relations.
9. **No component hierarchy.** Habitat groups, perimeters, bridge cells, and
   merge opportunities are not persistent objects.
10. **No motif hierarchy.** Bear pair opportunities, Elk line endpoints,
    Salmon path constraints, Hawk conflict sets, and Fox neighborhoods are not
    persistent objects.
11. **Opponent demand is disconnected from market supply.** Full opponent
    boards exist but are not converted into item-specific pressure.
12. **Hierarchical descendants are summarized too aggressively.** Robustness,
    multimodality, and interactions disappear.
13. **Arbitrary tile ID enters as a scalar.**

### P2: Efficiency, Generalization, and Learning Signal

14. Constant ruleset channels consume capacity.
15. Absolute q/r and q/r-sorted flattening create avoidable discontinuities.
16. No public action-history or opponent-policy state is retained.
17. Scalar/top-one targets understate teacher uncertainty and strategic
    equivalence.
18. No explicit future-access window describes how many opponents act before
    the focal player sees a market item again.
19. No representation probe verifies what survives each compression boundary.

## What State-of-the-Art Research Contributes

The relevant lesson is not "use a Transformer." It is to align the
representation with the symmetries, entities, relations, action factorization,
and uncertainty of the domain.

### Stockfish NNUE: Explicit Semantics and Incrementality

Stockfish's NNUE documentation treats feature sets as named contracts with
precise index layouts. HalfKP-style features are perspective and anchor
relative, sparse, and incrementally updated. The useful transfer is:

- name and own every feature block;
- make perspective explicit;
- organize local relations around a meaningful anchor;
- measure activation and update cost; and
- never let checkpoint compatibility depend on undocumented integer
  coincidence.

Primary sources:

- [Stockfish NNUE feature documentation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/features.html)
- [Stockfish NNUE architecture documentation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)

### AlphaStar: Entities, Autoregressive Actions, and Opponents

AlphaStar represents variable entities with self-attention, retains temporal
state, and emits structured actions autoregressively with pointer-like
arguments. Its league exposes policy-specific blind spots.

Cascadia's direct analogue is:

- entity and graph state encoding;
- `prelude -> draft -> tile -> wildlife` proposal;
- pointer distributions over legal factors;
- full complete-action rescoring; and
- explicit opponent intent conditioned on visible boards and policy identity.

Primary source:
[AlphaStar](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf).

### Graphormer and GraphGPS: Structure Before Attention

Graphormer shows that node centrality, shortest-path distance, and edge
information can be injected directly into attention. GraphGPS combines local
message passing, global attention, and structural/positional encodings.

Cascadia needs exactly this combination:

- local directional hex messages;
- global interaction among components, market, supply, and opponents;
- structural distances such as same component, path distance, and shared
  frontier; and
- linear or near-linear state encoding.

Primary sources:

- [Graphormer](https://proceedings.neurips.cc/paper_files/paper/2021/file/f1c1592588411002af340cbaedd6fc33-Paper.pdf)
- [GraphGPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5d4834a159f1547b267a05a4e2b7cf5e-Abstract-Conference.html)

### Set Transformer, Deep Sets, and GIN: Sets Are Not Just Means

Deep Sets formalizes permutation-invariant set functions. Set Transformer adds
learned interactions and inducing points. GIN demonstrates that common graph
aggregators can fail to distinguish simple structures and motivates injective
multiset aggregation.

For Cascadia:

- mean/max alone is too weak for candidate and descendant sets;
- inducing-point attention can summarize thousands of actions without a full
  quadratic matrix; and
- counts or injective multiset encoders should be used where multiplicity
  matters.

Primary sources:

- [Deep Sets](https://papers.neurips.cc/paper/6931-deep-sets)
- [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html)
- [Graph Isomorphism Network](https://openreview.net/forum?id=ryGs6iA5Km)

### DimeNet and Hypergraph Networks: Direction and Higher-Order Relations

DimeNet's directional messages show that edge direction and angular structure
should be first-class. Hypergraph neural networks represent relations involving
more than two entities.

Cascadia's motifs are naturally higher-order:

- a Salmon run is a constrained path, not a sum of pairwise adjacencies;
- Fox value depends on a neighborhood set;
- a habitat merge can connect several components through one tile;
- a complete action jointly chooses prelude, pair, tile placement, rotation,
  and wildlife placement.

Primary sources:

- [DimeNet](https://openreview.net/forum?id=B1eWbxStPH)
- [Hypergraph Neural Networks](https://ojs.aaai.org/index.php/AAAI/article/view/4235)

### Group Equivariance and HexaConv: Exact D6 Symmetry

Group-equivariant networks share weights over known transformations. HexaConv
specializes the idea to hexagonal lattices.

For this ruleset, rotations and reflections do not change strategic value. D6
should be a tested semantic contract, not a hope that augmentation eventually
teaches.

Primary sources:

- [Group Equivariant Convolutional Networks](https://proceedings.mlr.press/v48/cohenc16.html)
- [HexaConv](https://arxiv.org/abs/1803.02108)

### MuZero and Bisimulation: Preserve What Changes Control

MuZero learns a value-equivalent latent model that predicts quantities needed
for planning rather than reconstructing every observation detail. Deep
bisimulation learns invariances tied to reward and transition equivalence.

The important Cascadia distinction is:

- raw public state must be lossless at the Rust boundary;
- the learned latent need not reconstruct arbitrary serialization details;
- it must preserve score anatomy, legal affordances, refill belief, opponent
  response, and action ranking.

Primary sources:

- [MuZero](https://arxiv.org/abs/1911.08265)
- [Deep Bisimulation for Control](https://openreview.net/forum?id=-2FCwDKRREu)

### ReBeL: Public Belief Is a State Object

ReBeL represents public observations and policies as a public belief state for
imperfect-information search. Cascadia is not a two-player zero-sum game, so
its guarantees do not transfer. The decomposition still matters: unseen supply
should be represented as an exact public distribution, not as a lossy bag
summary.

Primary source:
[ReBeL](https://proceedings.neurips.cc/paper/2020/file/c61f571dbd2fb949d3fe5ae1608dd48b-Paper.pdf).

### Pointer Networks, Attention Routing, and Gumbel-Sinkhorn: Structured Choice

Pointer Networks and attention-based routing models construct combinatorial
outputs by selecting input elements. Gumbel-Sinkhorn provides a differentiable
relaxation for matching.

The direct opportunities are:

- point to legal market, frontier, and wildlife destinations rather than
  classify a fixed global vocabulary;
- represent open scoring needs and available supply as a bipartite matching
  problem; and
- learn completion feasibility without collapsing every plan to one scalar.

Primary sources:

- [Pointer Networks](https://papers.nips.cc/paper/5866-pointer-networks)
- [Attention, Learn to Solve Routing Problems](https://openreview.net/forum?id=ByxBFsRqYm)
- [Gumbel-Sinkhorn Networks](https://openreview.net/forum?id=Byt3oJ-0W)

## Proposed V3: Relational Opportunity Graph

The proposed representation is not a single flat vector. It is a typed graph
with multiple resolutions and an explicit complete-action interface.

### 1. Exact Public Substrate

Rust constructs a canonical, schema-versioned public state containing:

- every occupied tile and placed wildlife on all boards;
- every legal frontier cell on the focal board;
- exact wildlife-bag counts;
- exact unseen semantic tile-archetype counts;
- current and staged market pairs;
- Nature Tokens and turn order;
- scoring cards and habitat-bonus mode;
- ordered complete preludes; and
- exact legal action factors and transition deltas.

No learned model should be asked to reverse engineer omitted public
information.

### 2. Typed Tokens

#### Tile tokens

One per occupied tile:

- relative axial coordinate;
- both terrains;
- orientation;
- six directed edge terrains;
- wildlife eligibility;
- placed wildlife;
- keystone;
- owner and relative seat;
- turn placed if history is retained.

#### Frontier tokens

One per legal empty cell on the focal board:

- six neighbor-presence bits;
- six facing edge terrains;
- adjacent wildlife multiset;
- habitat components touchable by terrain;
- component-merge sizes;
- motif completion and conflict signals;
- compatible market and unseen tile archetypes;
- distance to relevant motif endpoints; and
- legal rotations for each tile archetype.

This is the most important new object. Cascadia decisions happen on frontier
cells, not on generic empty space.

#### Habitat-component tokens

One per connected component and terrain:

- size;
- perimeter/frontier size;
- second-largest and rank information;
- open merge cells;
- bridge count and articulation risk;
- number and quality of compatible unseen archetypes;
- owner and opponent rank gap.

Tiles connect to every component they belong to. Frontier cells connect to
every component they could merge.

#### Wildlife-motif tokens

Represent the exact scoring structure as typed, variable-size objects:

- Bear components, singletons, pair slots, and oversize risk;
- maximal Elk lines, endpoints, extension rays, and overlapping alternatives;
- Salmon paths, endpoints, branch conflicts, and legal continuation slots;
- Hawk conflict graph nodes and available isolated placements;
- Fox centers, current neighbor diversity, missing wildlife types, and
  compatible frontier cells.

These should be generated by exact Card A algorithms and used both as tokens
and auxiliary targets. They are not replacements for tile tokens. They are a
higher-resolution view of the same state.

#### Market tokens

One per tile-wildlife pair, plus separate tile and wildlife factor tokens:

- exact tile semantics;
- wildlife token;
- slot identity;
- same-slot and independent-draft eligibility;
- focal utility features;
- opponent-specific demand edges;
- probability of surviving until the focal player's next action; and
- staged identity under each legal prelude.

#### Supply tokens

One token per semantic tile archetype with its exact remaining count, plus one
token per wildlife type:

- terrain and edge semantics;
- allowed wildlife;
- keystone;
- remaining count;
- probability of appearing in the next refill;
- compatibility with each open habitat or motif need; and
- scarcity relative to all players' inferred demand.

#### Player and objective tokens

One per player and one per scoring objective:

- relative seat;
- turns until next action;
- Nature Tokens;
- score anatomy;
- component and motif portfolio;
- inferred shadow prices for terrains and wildlife;
- policy identity or history state when heterogeneous opponents are used.

#### Plan-slot tokens

A small exchangeable set of learned slots represents strategic commitments,
such as:

- complete another Bear pair;
- grow one Elk line to four;
- preserve a Salmon path to five;
- reserve isolated Hawk cells;
- complete a Fox neighborhood;
- merge two habitat components.

Slots must be supervised first with exact motif and feasibility targets. They
should earn the right to become latent plans rather than becoming an
uninterpretable extra bottleneck.

### 3. Typed Relations

The graph should include:

- six directed hex adjacency types;
- exact shared-edge terrain and match/mismatch;
- tile-to-habitat-component membership;
- tile/wildlife-to-motif membership;
- frontier-to-neighbor geometry;
- frontier-to-component merge relations;
- frontier-to-motif completion/conflict relations;
- market-to-supply archetype identity;
- market-to-player demand;
- supply-to-plan compatibility;
- action-selects-market-item;
- action-places-tile-at-frontier;
- action-places-wildlife-at-tile;
- action-adds/removes/merges motif and component objects; and
- candidate-to-candidate relations such as same draft, same frontier, same
  wildlife target, dominance, and equivalent public afterstate.

### 4. Action as an Exact Edit Program

Each complete action should be represented twice:

1. as structured factors:
   `prelude -> draft -> tile coordinate/rotation -> wildlife coordinate`; and
2. as an exact graph edit:
   selected market objects, removed objects, new tile, new wildlife, component
   merges, motif transitions, token delta, staged supply, and immediate score
   anatomy.

The current models mostly ask, "What does this candidate vector look like?"
The edit representation asks the more useful question:

> "What exact opportunities does this action create, destroy, merge, consume,
> expose to opponents, or leave robust to refill variance?"

### 5. Encoder Architecture

A practical MLX encoder can combine:

- directional local message passing over tile/frontier edges;
- motif and component hypergraph messages;
- global attention over typed state tokens;
- Graphormer-style relation biases;
- injective count-aware aggregation where multiplicity matters; and
- exact D6 equivariance through weight sharing or group averaging.

The state trunk should be encoded once per decision. A plausible target is
roughly 200-400 state tokens and 4-8 million parameters, small enough for local
MLX batching.

### 6. Scalable Action Pipeline

Full attention over 3,000-10,000 legal actions is unnecessary and expensive.

Use three stages:

1. **Hierarchical pointer proposal**
   - prelude/draft;
   - tile frontier and rotation conditioned on the selected draft;
   - wildlife placement conditioned on both.
2. **Cheap complete-action retrieval**
   - every legal action receives an edit-aware score from cached state tokens;
   - retain approximately 256 actions.
3. **Relational complete-action rescoring**
   - cross-attend retained actions to raw state, frontier, motif, supply, and
     opponent tokens;
   - allow sibling attention or Set Transformer inducing points;
   - retain 64 for search or directly produce a confidence-set policy.

This preserves the oracle-proven hierarchy without forcing every stage to
independently rediscover the whole game.

### 7. D6 Symmetry Contract

Every representation and prediction must be testable under all 12 rotations
and reflections:

- transformed state graph is isomorphic;
- transformed legal actions remain bijective;
- policy probabilities permute exactly;
- scalar values remain equal;
- component and motif targets transform consistently.

For the retained candidate cross-encoder, an action-centric canonical frame can
place the proposed tile at the origin and rotate its orientation to a standard
direction. The shared state trunk should remain equivariant so it is not
recomputed for every action.

### 8. Multitask Representation Targets

The trunk should be trained to preserve exact, decision-relevant semantics:

- current and final score components;
- one-turn and one-table-rotation score deltas;
- habitat component sizes, perimeters, and merge opportunities;
- motif identity, size, endpoints, conflicts, and completion probability;
- exact refill archetype distribution;
- opponent next-draft distribution;
- market-item survival probability;
- complete-action immediate graph edits;
- teacher Q distribution and standard error;
- confidence-set membership;
- legal factor masks; and
- optional public afterstate latent prediction.

Auxiliary heads are not decoration. Each one is a probe that the trunk has not
discarded a required concept.

## Bold Directions

### 1. Demand-Supply Matching as a First-Class Object

Construct a bipartite graph:

```text
open scoring and habitat needs <-> market plus unseen supply
```

Need nodes include Elk endpoints, Salmon endpoints, Bear pair slots, Hawk
isolation slots, Fox missing-neighbor types, and habitat bridge opportunities.
Supply nodes include current pairs, wildlife counts, and semantic tile
archetypes.

Edge values encode compatibility, arrival probability, number of opponents
before access, and alternative uses. A Sinkhorn-style or attention matching
head estimates whether the board's apparent potential is actually fulfillable.

This directly represents the distinction between:

- a promising shape with abundant compatible supply; and
- the same shape after the required tiles or wildlife have mostly disappeared.

### 2. Opportunity Derivatives

For every frontier cell and market item, compute exact local counterfactual
derivatives:

- immediate score delta;
- component merge vector;
- motif state transition;
- new frontier quality;
- lost future placements;
- opponent denial value; and
- number of remaining compatible supply objects.

These are not hand-tuned evaluation weights. They are exact structured
measurements used as action inputs and auxiliary targets.

### 3. Future-Access Windows

A market item has different value depending on whether zero, one, two, or three
opponents act before the focal player can use it again. Represent:

- relative seat order;
- opponents before next focal action;
- each opponent's demand for the item;
- replacement/wipe probability; and
- probability that an equivalent item reappears.

This is the missing bridge between opponent boards and market drafting.

### 4. Public-State Equivalence Classes

Hash exact public afterstates, staged supply, and turn order. If multiple legal
actions produce the same public transition distribution, collapse them for
proposal and preserve a list of equivalent action hashes for replay.

This can reduce the legal set without heuristic pruning. Equivalence must be
proven by the Rust simulator; paid-wipe order must not be collapsed merely
because final slot unions match.

### 5. Learned Quotient State

Train a latent that predicts:

- reward anatomy;
- legal affordances;
- public transition distributions;
- opponent action distributions; and
- teacher action values.

Then test whether states with equivalent predictions can share a latent even
when their raw serialization differs. This is a Cascadia-specific
value-equivalent or bisimulation representation. It should be attempted only
after the exact graph substrate and probes exist.

### 6. Topological and Spectral Structure

Cheap structural encodings may expose long-range shape without many message
passing layers:

- component count and perimeter;
- articulation and bridge cells;
- shortest paths between motif endpoints;
- cycle and hole counts;
- random-walk or Laplacian encodings with sign-invariant treatment.

These should be ablated individually. They are promising for Elk, Salmon, and
habitat topology, but should not become another uncontrolled feature pile.

### 7. Distributional Opportunity Value

Do not collapse descendants to min/mean/max. Represent a small quantile or
histogram distribution over:

- completion score;
- turns to completion;
- supply arrival;
- opponent theft;
- best descendant action; and
- downside from committing a scarce frontier.

Two actions with equal expected value can differ dramatically in robustness.
The teacher itself is uncertain: only 18.359% of validation winners are
distinguishable at 95%, and the mean 95% confidence set contains 10.140
actions. See [teacher identifiability](mce-teacher-identifiability.md).

## Experiment Program

The experiments below are ordered by information leverage, not novelty. Upside
priors are engineering judgments and are not additive.

| Rank | Experiment | Primary hole | Cost | Non-additive upside prior |
|---:|---|---|---|---:|
| 0 | Schema manifest and activation census | Silent dead/constant features | Low | Enables every later result |
| 1 | Corrected V1 mid-tail schema | Missing intended supply tail | Low | 0.0 to +0.3 |
| 2 | Exact semantic supply tokens | Non-injective refill belief | Low-medium | +0.1 to +0.7 |
| 3 | Frontier affordance and action-edit tokens | Missing decision surface | Medium | +0.3 to +1.2 |
| 4 | Habitat-component and wildlife-motif graph | Missing long-range structure | Medium | +0.4 to +1.5 |
| 5 | Relational retained-candidate set | Mean/max sibling context | Medium | +0.2 to +0.9 |
| 6 | Exact D6 representation | Sample inefficiency and shortcuts | Medium | +0.1 to +0.5 |
| 7 | Opponent-demand and market-survival graph | Competition blindness | Medium | +0.3 to +1.0 |
| 8 | Hierarchical pointer plus complete-action rescoring | Proposal bottleneck | Medium-high | +0.5 to +1.8 |
| 9 | Demand-supply matching head | Plan feasibility blindness | Medium-high | +0.3 to +1.2 |
| 10 | Interpretable plan slots | Cross-turn commitment | High | +0.3 to +1.2 |
| 11 | Value-equivalent multi-horizon latent | Planning representation | High | +0.5 to +2.0 |
| 12 | Integrated Relational Opportunity Graph | Combined ceiling | High | +1.0 to +3.0 |

### Experiment 0: Feature Manifest and Activation Census

Create one generated manifest for every V1 and V2 schema:

```text
name
schema version and hash
index or tensor slice
semantic owner
value domain
expected invariance
perspective convention
incremental update dependency
activation rate by phase
constant/dead/aliased status
checkpoint compatibility
```

Run it over at least one million candidate rows and all four focal seats.

Required tests:

- every documented block lands in its declared range;
- no feature index crosses its schema boundary;
- no active feature is unnamed;
- dead and constant channels are reported, not silently accepted;
- all old checkpoints retain exact historical extraction;
- V1 opponent order is characterized for every focal seat;
- D6 transforms preserve documented semantics.

This experiment should produce tooling, tests, and a report before any new
model is trusted.

### Experiment 1: Corrected V1 Mid-Tail

Define `legacy-mid-v4-fixed-v1` as a new schema. Preserve the first 10,561
columns and the 369 historical opponent columns. Add the intended:

- 150 extended tile-terrain counts;
- 150 extended tile-wildlife counts; and
- overflow-used bit

in a documented new range. Do not reinterpret the old 301 rows.

Train two controlled arms:

1. old champion with exact historical extraction;
2. old weights copied into common columns, corrected rows zero-initialized,
   stable fine-tuning at the known safe low learning rate.

Gate:

- exact parity for arm 1;
- nonzero activation for every corrected block;
- no score regression in 100 paired games;
- continue only if offline loss or paired mean moves materially.

### Experiment 2: Exact Semantic Supply

Replace the 30 marginal tile values with semantic-archetype count tokens while
retaining the 30 values as an ablation control.

Cheap falsification:

- decode the exact next-tile distribution with greater than 99.99% accuracy;
- distinguish every adversarial collision pair;
- improve held-out tile-stage top-64 target recall;
- measure gains specifically in low-supply and independent-draft subsets.

This is the cleanest high-information experiment because it changes no board
encoder and attacks a proven collision.

### Experiment 3: Frontier Affordance and Action Edit

Add exact frontier tokens and explicit action-to-frontier relations. Do not yet
add motif hyperedges.

Features should include:

- six directed neighbor edge states;
- exact terrain component merges;
- wildlife eligibility and occupancy;
- current and resulting frontier count;
- immediate score anatomy;
- staged supply and market delta.

Compare:

1. q/r-only action query;
2. existing radius-one local geometry;
3. frontier token cross-attention;
4. frontier token plus exact edit vector.

Gate:

- greater than 99% exact decoding of local action consequences;
- validation top-64 winner recall above the existing 74.17% local model;
- no train-only specialization collapse under geometry corruption tests.

### Experiment 4: Component and Motif Graph

Add habitat components and wildlife motif tokens generated by exact Rust
algorithms.

Use auxiliary heads to decode:

- component membership and size;
- merge result for every legal tile placement;
- longest/valuable Elk structures;
- Salmon path validity and endpoints;
- Hawk conflict degree and isolation;
- Bear pair completion;
- Fox diversity and missing types.

Four frozen ablations:

1. component only;
2. motif only;
3. component plus motif;
4. component plus motif plus frontier.

Gate:

- greater than 99% semantic decoder accuracy;
- no worse retained R4800 regret;
- material improvement in tile-stage recall;
- gains must appear on Elk, Salmon, and Hawk subsets rather than only Bear.

### Experiment 5: Relational Candidate Context

Keep the state encoder fixed and replace candidate mean/max context with:

- inducing-point Set Transformer context;
- low-rank candidate attention; or
- explicit relation attention within groups sharing draft, frontier, or
  wildlife destination.

Run only on the top 256 or top 128 cheap candidates.

Counterexample tests must distinguish candidate multisets with the same
min/mean/max but different rank boundaries and descendant robustness.

Gate:

- improved confidence-set coverage;
- top-64 winner recall above 98% on the open validation target before gameplay;
- retained R4800 regret below 0.15;
- bounded P99 latency under the local MLX serving budget.

### Experiment 6: Full D6

Implement one exact transform library shared by:

- Rust records;
- MLX tensors;
- actions;
- frontier and motif graphs; and
- targets.

Compare C6 augmentation, D6 augmentation, and exact D6 group averaging or
equivariant layers.

Gate:

- policy permutation error below numerical tolerance;
- value invariance below numerical tolerance;
- no data-dependent transform failures;
- equal or better held-out action recall with fewer unique training positions.

### Experiment 7: Opponent Demand and Market Survival

For every opponent and market item, predict:

- probability the opponent selects it;
- marginal terrain value;
- marginal wildlife value;
- denial value against the focal player; and
- probability the item survives to the focal player's next action.

Use all opponent boards, relative seat order, and exact supply. Add edges from
opponent demand tokens to market and supply tokens.

Gate:

- calibrated next-draft prediction;
- improvement concentrated in early and middle phases;
- improvement on high-regret draft-choice decisions;
- paired gameplay gain at equal search budget.

### Experiment 8: Hierarchical Pointer Proposal

Reuse the oracle-proven factorization:

```text
prelude/draft -> tile -> wildlife
```

Each stage receives the same cached relational state plus selected-prefix
tokens. The tile stage points to a frontier token and rotation, not a flattened
factor row. The wildlife stage points to a legal tile token.

The final retained actions are complete and receive joint rescoring.

Gate:

- proposal target recall greater than 98%;
- proposal winner retention greater than 98%;
- mean proposals at most 1,024 initially, then target 512;
- top-64 confidence-set coverage at least 99%;
- no action-family or phase subset below preregistered floors.

### Experiment 9: Demand-Supply Matching

Add exact need and supply nodes with a matching head. Train on completed-game
realization and counterfactual teacher returns.

Required ablations:

- no matching;
- attention-only matching;
- Sinkhorn relaxation;
- exact small bipartite solver as an auxiliary teacher.

Gate:

- better completion-probability calibration;
- better ranking in scarce-supply states;
- positive Elk, Salmon, and Hawk score movement without Bear collapse.

### Experiment 10: Plan Slots

Initialize a small set of plan slots from exact motif opportunities, then allow
them to evolve through attention.

Before any gameplay, require that slots predict:

- selected objective;
- cells and supply objects claimed by the plan;
- expected completion turn;
- expected score;
- abandonment probability; and
- conflict with other slots.

Slots that cannot be decoded should not be trusted as strategic plans.

### Experiment 11: Value-Equivalent Multi-Horizon Model

Predict exact public afterstate representations at:

- immediate afterstate;
- after market refill;
- after one opponent action;
- after one table rotation; and
- terminal score distribution.

Use the exact simulator for transition targets. The model learns only the
decision-relevant stochastic abstraction.

Gate:

- accurate legal-affordance and score-anatomy prediction;
- improved shallow-search ranking at equal simulations;
- measurable search-depth leverage rather than merely lower training loss.

### Experiment 12: Integrated V3

Integrate only blocks that pass their isolated gates:

- exact supply;
- frontier;
- component/motif graph;
- D6;
- opponent demand;
- hierarchical pointer retrieval;
- relational complete-action rescoring;
- score anatomy and uncertainty heads.

Promotion sequence:

1. open train and validation only;
2. semantic probe suite;
3. R4800 proposal and confidence-set gates;
4. equal-budget 20-game smoke;
5. paired 100-game pilot requiring at least +0.50 mean;
6. paired 500-game confirmation;
7. 1,000-game final validation if the confidence interval remains favorable.

## Information-Preservation Tests

Every model should expose representations at these boundaries:

```text
raw record
-> entity/graph tokens
-> pooled or latent state
-> action factors
-> candidate embedding
-> candidate-set context
-> output
```

Train frozen probes for exact concepts at each boundary:

- tile and wildlife occupancy;
- frontier membership;
- component membership and size;
- motif endpoints and conflicts;
- exact supply archetype counts;
- staged market identity;
- immediate action deltas;
- opponent demand;
- D6 transform identity;
- teacher confidence-set membership.

If a concept is decodable before a boundary and lost after it, that boundary
has identified the representation bottleneck. This is more informative than
training another end-to-end model and inspecting only top-one recall.

## Adversarial Test Suite

The permanent representation test suite should contain synthetic pairs that
look identical to a weak representation but require different decisions.

1. **Supply alias:** same 30 marginals, different semantic tile multiset.
2. **Pooling collision:** same mean/max, different multiplicity or descendant
   distribution.
3. **Radius-one alias:** identical immediate neighborhood, different long
   Salmon path or habitat component.
4. **Opponent-order alias:** same absolute opponent order, different focal
   seat-relative order.
5. **D6 transform:** all rotations and reflections of one decision.
6. **Tile-ID permutation:** semantic identity fixed, arbitrary IDs permuted.
7. **Frontier bridge:** same local terrain count, different component merge.
8. **Motif conflict:** same immediate score, one action destroys a future Hawk
   or Salmon plan.
9. **Market survival:** same focal board and market, different opponent demand
   and seat timing.
10. **Action-equivalence:** different serialized actions with proven identical
    public transition, plus near-matches that differ only after refill.
11. **Hierarchy interaction:** same draft score and tile score separately, but
    different joint wildlife completion.
12. **Teacher ambiguity:** several statistically equivalent actions versus one
    genuinely distinguishable winner.

These tests should run in CI and accompany every schema version.

## Cluster Execution Strategy

The four Macs should maximize independent research throughput, not duplicate a
single training run by default.

### During representation development

- john1: Rust extraction, schema manifests, exact collision and invariance
  tests.
- john2: exact-supply and frontier MLX ablations.
- john3: component/motif and D6 ablations.
- john4: candidate-set, hierarchy, and opponent-demand ablations.

### During training

- run one scientifically distinct arm per host;
- use MLX locally on each Apple GPU;
- use CPU cores for record generation, exact graph construction, and teacher
  replay;
- reserve duplicate seeds for uncertainty estimation only after an arm passes
  its first gate;
- cross-replay selected checkpoints on a different host for portability;
- record Metal occupancy separately because CPU utilization does not describe
  MLX load.

### During confirmation

- shard fresh games across all four hosts;
- use disjoint deterministic seed ranges;
- retain one frozen binary, schema, model digest, and source identity;
- aggregate paired game-block confidence intervals centrally.

## What Not to Repeat

1. Do not append more hand-bucketed columns without a manifest, activation
   census, and collision analysis.
2. Do not interpret the V1 mid-tail under new semantics.
3. Do not add generic graph layers while retaining the same lossy pooling and
   weak target.
4. Do not widen pointwise MLPs and call it relational modeling.
5. Do not use scalar tile IDs as semantic geometry.
6. Do not collapse ordered preludes or descendant distributions to unions and
   moments.
7. Do not full-attend ten thousand actions when an oracle-proven hierarchy can
   retrieve hundreds.
8. Do not train exact top-one imitation against a teacher whose winner is
   usually statistically non-unique.
9. Do not add latent plan slots without exact semantic probes.
10. Do not promote on aggregate loss while tile-stage recall, phase subsets, or
    confidence-set coverage fail.
11. Do not infer that geometry is useless from a geometry-only experiment.
12. Do not infer that hierarchy is useless from a weak factor
    representation.

## Immediate Recommendation

The immediate work should be a representation foundation sprint, not another
large gameplay model:

1. build the schema manifest, activation census, and adversarial tests;
2. publish the corrected V1 mid-tail under a new schema;
3. add exact semantic supply tokens to the V2 graded-oracle path;
4. add first-class frontier tokens and exact action-edit relations;
5. run the exact-supply and frontier ablations in parallel across the cluster;
6. only then introduce component/motif tokens and relational candidate
   comparison.

The first learned experiment I would bet on is:

> **Exact semantic supply + frontier affordance tokens + explicit action-edit
> cross-attention, evaluated on the conditional tile stage and complete-action
> top-64 retrieval.**

It attacks three confirmed holes, preserves the existing hierarchy, is small
enough to falsify quickly, and does not require a speculative world model.

## Final Research Thesis

Cascadia's next four points are unlikely to come from representing more facts
about occupied tiles in isolation. They will come from representing
**opportunities and competition**:

- an empty cell as a set of possible component and motif transitions;
- an unseen bag as an exact distribution of joint tile affordances;
- a market pair as a contested object with a survival window;
- an opponent board as demand for particular future resources;
- a plan as a matching between unfinished structures and stochastic supply;
- and an action as an exact edit to that opportunity system.

V1 discovered that opponent context matters. V2 discovered that complete
actions and hierarchical structure matter. The next representation should
make those discoveries native rather than bolt-ons.

## Primary Sources

### Game AI and structured action

- Schrittwieser et al.,
  [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265).
- Vinyals et al.,
  [Grandmaster Level in StarCraft II Using Multi-Agent Reinforcement Learning](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf).
- Perolat et al.,
  [Mastering the Game of Stratego with Model-Free Multiagent Reinforcement Learning](https://www.science.org/doi/10.1126/science.add4679).
- Schmid et al.,
  [Player of Games](https://arxiv.org/abs/2112.03178).
- Hubert et al.,
  [Learning and Planning in Complex Action Spaces](https://arxiv.org/abs/2104.06303).
- Brown et al.,
  [Combining Deep Reinforcement Learning and Search for Imperfect-Information Games](https://proceedings.neurips.cc/paper/2020/file/c61f571dbd2fb949d3fe5ae1608dd48b-Paper.pdf).
- Silver et al.,
  [Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270).
- Wu,
  [Accelerating Self-Play Learning in Go](https://arxiv.org/abs/1902.10565).

### Graphs, sets, symmetry, and objects

- Ying et al.,
  [Do Transformers Really Perform Bad for Graph Representation?](https://proceedings.neurips.cc/paper_files/paper/2021/file/f1c1592588411002af340cbaedd6fc33-Paper.pdf).
- Rampasek et al.,
  [Recipe for a General, Powerful, Scalable Graph Transformer](https://proceedings.neurips.cc/paper_files/paper/2022/hash/5d4834a159f1547b267a05a4e2b7cf5e-Abstract-Conference.html).
- Lee et al.,
  [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html).
- Zaheer et al.,
  [Deep Sets](https://papers.neurips.cc/paper/6931-deep-sets).
- Xu et al.,
  [How Powerful are Graph Neural Networks?](https://openreview.net/forum?id=ryGs6iA5Km).
- Klicpera et al.,
  [Directional Message Passing for Molecular Graphs](https://openreview.net/forum?id=B1eWbxStPH).
- Feng et al.,
  [Hypergraph Neural Networks](https://ojs.aaai.org/index.php/AAAI/article/view/4235).
- Cohen and Welling,
  [Group Equivariant Convolutional Networks](https://proceedings.mlr.press/v48/cohenc16.html).
- Hoogeboom et al.,
  [HexaConv](https://arxiv.org/abs/1803.02108).
- Locatello et al.,
  [Object-Centric Learning with Slot Attention](https://arxiv.org/abs/2006.15055).
- Lim et al.,
  [Sign and Basis Invariant Networks for Spectral Graph Representation Learning](https://openreview.net/forum?id=zSeoDvsDCe).

### Combinatorial choice and representation learning

- Vinyals et al.,
  [Pointer Networks](https://papers.nips.cc/paper/5866-pointer-networks).
- Kool et al.,
  [Attention, Learn to Solve Routing Problems](https://openreview.net/forum?id=ByxBFsRqYm).
- Mena et al.,
  [Learning Latent Permutations with Gumbel-Sinkhorn Networks](https://openreview.net/forum?id=Byt3oJ-0W).
- Zhang et al.,
  [Deep Bisimulation for Control](https://openreview.net/forum?id=-2FCwDKRREu).
- Jaegle et al.,
  [Perceiver IO](https://arxiv.org/abs/2107.14795).

### Engine representation practice

- [Stockfish NNUE feature documentation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/features.html).
- [Stockfish NNUE architecture documentation](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html).
