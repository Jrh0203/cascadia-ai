# Relational Feature Census

This standalone Rust crate is the exact-mechanics foundation for five
representation hypotheses in the Cascadia V2 plan:

- R5: component-and-motif quotient state;
- R6: incremental sparse apply/undo;
- S3: habitat-component and wildlife-motif graph; and
- S5: counterfactual opportunity derivatives; and
- S6: compact topological, path, random-walk, and spectral structure.

It deliberately does not train a model. It establishes which facts can be
computed exactly, which raw geometry remains necessary, and which operations
are fast enough to place inside future MLX training and search loops.

## Shared Corpus

Every lane generates deterministic four-player `AAAAA` games with no habitat
bonus through `GameConfig::research_aaaaa(4)`. Each game contains 80 decision
positions. At every position the runner consumes:

- the exact R2 occupied-plus-frontier sparse public state;
- the exact R3 complete legal action set;
- exact R3 board, market, supply, frontier, component, motif, score, and turn
  edits; and
- authoritative public afterstates produced by applying those edits.

The lanes use disjoint production seed ranges. Calibration seeds are never
admitted to production evidence.

## Exact Relational Graph

`graph.rs` derives Card A objects for every relative board:

- habitat components with size, internal edges, open boundary, cycle rank,
  bridges, articulations, size rank, frontier contacts, and merge frontiers;
- Bear components, singleton/pair opportunities, and oversize risk;
- Elk lines, endpoints, legal extension cells, and overlapping alternatives;
- Salmon paths, endpoints, branch conflicts, validity, and legal
  continuations;
- Hawk conflict edges, isolated Hawks, and isolated-placement opportunities;
- Fox centers, observed diversity, missing species, and compatible cells; and
- frontier degree, bridge, repeated-contact, and resulting-size summaries.

The graph also reconstructs exact Card A wildlife, habitat, nature-token, and
base scores. It uses no learned weights.

## R5

R5 compares four information surfaces:

1. exact R2 tile and frontier control;
2. component-and-motif quotient only;
3. quotient plus a small action-local geometry patch; and
4. the full sparse-plus-relational hybrid.

The quotient is required to preserve score anatomy while remaining
insufficient for exact legal affordance by itself. The local patch must then
recover every tested legal affordance and immediate score delta exactly.

The model-facing quotient must use at most 80% of the control's median parent
tokens or canonical message bytes. The JSON/postcard audit object is
intentionally verbose; token count is the primary realistic graph-serving
measure.

## R6

R6 creates one mutable active-board accumulator and applies every complete R3
action edit, then undoes it. It checks:

- exact equality with an independently reconstructed authoritative afterstate;
- exact return to the parent digest after undo; and
- at least 2x throughput over authoritative full afterstate application.

R2 component IDs are traversal-local and may renumber unchanged components.
The accumulator therefore maps frontier touches to stable component keys
derived from relative seat, terrain, and sorted component members.

## S3

S3 measures four exact views:

1. component only;
2. motif only;
3. component plus motif; and
4. component plus motif plus frontier.

It validates current-board score anatomy, one deterministic complete-action
score delta per position, all twelve D6 transforms, and corpus coverage for
Elk, Salmon, Hawk, and Bear opportunity objects. Passing authorizes learned
ablation tests; it does not claim that any view improves ranking or gameplay.

## S5

S5 samples up to 64 deterministic legal actions per position and constructs a
154-field factual derivative:

- 12 immediate score-anatomy deltas;
- habitat topology deltas;
- wildlife motif and opportunity deltas;
- frontier deltas;
- lost and newly opened future wildlife placements;
- exact semantic-supply compatibility and remaining mass;
- selected-object opponent access; and
- total market-access deltas.

Every derivative is recomputed from authoritative before/after states. The
runner publishes a per-field robust normalization contract. It does not
inject strategic weights or teacher targets.

## S6

S6 derives a compact, D6-invariant encoding for eleven graphs on every board:
the occupied-tile graph, five directed habitat graphs, and five wildlife
adjacency graphs. Every channel includes exact scalar topology, deterministic
random-walk return moments, and exact combinatorial-Laplacian trace moments.
The board encoding also includes exact geometric hole count and shortest-path
summaries between eligible Elk and Salmon continuation cells.

The census validates the habitat channel against the existing exact S3
component graph, runs a registered synthetic adversarial suite, mines board
pairs that collide under the current S3 scalar signature, and reports
extraction latency plus serialized size. Passing authorizes learned family
ablations; it does not establish ranking or gameplay value.

## Commands

```bash
cargo fmt --manifest-path tools/relational_feature_census/Cargo.toml
cargo clippy --manifest-path tools/relational_feature_census/Cargo.toml \
  --all-targets -- -D warnings
cargo test --manifest-path tools/relational_feature_census/Cargo.toml \
  -- --test-threads=1
cargo build --release \
  --manifest-path tools/relational_feature_census/Cargo.toml
```

Run one lane:

```bash
tools/relational_feature_census/target/release/relational-feature-census \
  --lane r5 \
  --first-seed 5110000 \
  --games 20 \
  --source-bundle-id <64-hex-bundle-id> \
  --host john1 \
  --rayon-threads 6 \
  --output artifacts/experiments/r5-component-motif-quotient-foundation-v1/reports/john1-production.json
```

The production campaign is generated and classified by
`tools/relational_feature_foundation_campaign.py`.

## Claim Boundary

Passing these foundations proves exact mechanics, corpus coverage, and
mechanical serving properties only. It does not establish:

- retained R4800 quality;
- learned action-ranking improvement;
- gameplay-score improvement;
- a selected V2 representation; or
- progress to the 100-point mean by itself.

Those claims require matched MLX ablations and paired gameplay experiments.
