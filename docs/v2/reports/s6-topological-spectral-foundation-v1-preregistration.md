# S6 Topological And Spectral Foundation V1 Preregistration

Date: 2026-06-17

ADR: 0164

Experiment: `s6-topological-spectral-foundation-v1`

Protocol: `s6-exact-topology-collision-census-v1`

Status: frozen before production

## Question

Can compact exact topology, marked-path, random-walk, and spectral encodings
expose long-range board distinctions that collide under the accepted S3
scalar graph surface, while remaining D6 invariant and cheap enough to test in
MLX?

## Encoding

For every relative board, construct:

- one occupied-tile graph;
- five directed habitat graphs;
- five wildlife adjacency graphs;
- exact occupied-polyhex hole count;
- eligible Elk endpoint path summary; and
- legal Salmon continuation path summary.

Graph scalars, fixed-point random-walk returns, and exact combinatorial
Laplacian trace moments are separate feature families. No teacher score,
strategic weight, hidden state, or target-derived feature is permitted.

## Immutable Production Design

```text
ruleset: four-player AAAAA, no habitat bonuses
positions per game: 80
games per host: 10
hosts: john1, john2, john3, john4
total games: 40
total positions: 3,200
total board encodings: 12,800
```

Seed blocks:

```text
john1: 5,610,000 .. 5,610,009
john2: 5,620,000 .. 5,620,009
john3: 5,630,000 .. 5,630,009
john4: 5,640,000 .. 5,640,009
```

Seed 5,600,000 is calibration-only and excluded.

## Per-Shard Gates

Every host must independently pass all gates frozen in ADR 0164:

- exact topology decoders;
- all twelve D6 transforms per position;
- all four adversarial checks;
- at least 128 current-S3 collision pairs;
- at least 128 complete-S6 separations;
- at least one separated long-range collision;
- P99 extraction at or below 2 ms; and
- median four-board encoding at or below 4 KiB.

## Aggregate Gates

The campaign passes only when:

- the source bundle ID and executable digest are identical on all hosts;
- host and seed assignments match this document exactly;
- seed ranges are disjoint;
- every shard reports
  `s6_topological_spectral_features_authorized`; and
- reversing report order produces byte-identical aggregate science.

Aggregate collision counts are the conservative sum of within-shard pairs.
Cross-shard collision pairs are deliberately not reconstructed or counted.

## Predictions

1. Scalar distance and degree structure will distinguish more current-S3
   collisions than holes alone.
2. Random-walk and Laplacian moments will separate nonisomorphic graph shapes
   that share node, edge, bridge, articulation, and degree summaries.
3. Elk and Salmon marked paths will contribute a smaller but strategically
   targeted novelty set.
4. Natural geometric holes will be rare or absent in 20-turn boards.
5. The encoding will fit below 4 KiB, but exact matrix moments may determine
   whether the 2 ms P99 gate passes.

## Invalidators

- production before this preregistration and the immutable bundle are frozen;
- changing gates after observing a production shard;
- overlapping seed ranges;
- hidden refill order or teacher-label access;
- floating-point spectral values whose D6 equality depends on tolerance;
- different source or executable identity between hosts;
- omitting a failed shard from the aggregate; or
- order-dependent classification.

## Follow-On

If authorized, run matched MLX ablations on one selected compact substrate:

1. no S6;
2. scalar topology and holes;
3. marked paths;
4. random-walk returns;
5. Laplacian moments; and
6. all S6 families.

The learned promotion decision remains governed by protected-slice R4800
quality, complete-action retrieval, complete-decision latency, and gameplay.
