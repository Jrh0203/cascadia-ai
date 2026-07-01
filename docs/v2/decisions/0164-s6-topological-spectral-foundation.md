# ADR 0164: S6 Topological And Spectral Foundation

Status: rejected before production; superseded by ADR 0165

Date: 2026-06-17

Experiment: `s6-topological-spectral-foundation-v1`

Protocol: `s6-exact-topology-collision-census-v1`

Research-plan item: S6

## Context

S3 proved that explicit habitat components and wildlife motifs are exact and
compact enough to test. Its current graph tokens still expose mostly local
counts and object membership. A learned model must infer long-range shape,
marked endpoint distance, diffusion behavior, and graph spectrum through
message passing.

S6 tests whether those facts can be derived once in Rust as compact,
deterministic state features.

## Decision

Build one exact encoding for every relative board with eleven graph channels:

1. occupied-tile adjacency;
2. five directed habitat graphs; and
3. five same-species wildlife adjacency graphs.

Every channel contains:

- nodes, edges, components, and boundary half-edges;
- cycle rank, bridges, and articulation count;
- diameter, maximum component radius, reachable pairs, and distance sum;
- degree histogram;
- deterministic fixed-point random-walk return at 2, 3, 4, and 6 steps; and
- the first six exact traces of the combinatorial Laplacian powers.

The Laplacian trace moments are invariant to node order, eigenvector sign, and
degenerate eigenspace basis. They provide a deterministic spectral encoding
without a floating-point eigensolver.

Each board also contains:

- exact geometric hole count for the occupied polyhex;
- shortest-path summaries between eligible Elk line endpoints; and
- shortest-path summaries between legal Salmon continuation cells.

## Existing-Signal Control

The corpus groups boards by the current S3 D6-invariant scalar signature:

- habitat component counts and scalar topology;
- Bear, Elk, Salmon, Hawk, and Fox motif scalars;
- opportunity summary;
- frontier summary; and
- exact current score anatomy.

S6 then measures how often topology, path, random-walk, spectral, or complete
encodings distinguish boards that collide under that existing surface.

This is an information and mechanics census. Teacher labels are not used.

## Exact Validation

At every position:

- validate occupied node count and connectivity;
- validate all five habitat channels against exact S3 component nodes, edges,
  component count, boundary, cycle rank, bridges, and articulations;
- validate all five wildlife channels against exact R2 motif node and edge
  counts;
- transform the complete sparse state through all twelve D6 elements and
  require byte-identical S6 encodings; and
- record encoding bytes and extraction time separately from game generation.

## Adversarial Suite

The production binary must pass four registered checks:

1. a six-cell ring contains one exact geometric hole while a line contains
   none;
2. the same path graph with near and far marked endpoints yields distances one
   and five;
3. two graphs that collide under legacy scalar topology are separated by
   fixed-point random-walk returns; and
4. the same scalar collision is separated by exact Laplacian moments.

## Gates

Each production shard must satisfy:

```text
topology decoder failures = 0
topology decoder checks = boards * 11
D6 failures = 0
D6 checks = positions * 12
adversarial checks = 4
adversarial failures = 0
existing-S3 collision pairs >= 128
complete-S6 separated pairs >= 128
at least one long-range collision is separated
P99 extraction <= 2,000,000 ns per four-board state
median encoding <= 4,096 bytes per four-board state
```

The limits are foundation limits. Any learned serving arm must still satisfy
the stricter common complete-decision latency gate.

## Cluster Design

One immutable source and executable bundle is replayed on four disjoint seed
blocks:

| Host | First seed | Games | Positions |
|---|---:|---:|---:|
| john1 | 5,610,000 | 10 | 800 |
| john2 | 5,620,000 | 10 | 800 |
| john3 | 5,630,000 | 10 | 800 |
| john4 | 5,640,000 | 10 | 800 |

Calibration seed 5,600,000 is excluded. The four production reports are
portability and seed-sensitivity replicates with unique evidence, not duplicate
training.

The aggregate requires:

- byte-identical source and executable identity;
- exactly one report from each registered host and seed block;
- every shard gate to pass;
- no overlapping seed ranges; and
- forward/reverse report-order identity.

## Classification

```text
s6_topological_spectral_features_authorized
s6_topology_decoder_failed
s6_d6_invariance_failed
s6_adversarial_separation_failed
s6_corpus_novelty_futile
s6_long_range_coverage_futile
s6_extraction_latency_failed
s6_encoding_compactness_failed
s6_cross_host_inconsistent
```

A pass authorizes capacity-controlled learned ablations of scalar topology,
marked paths, random walks, Laplacian moments, and their combination on the
selected compact substrate.

## Claim Boundary

This foundation can prove exactness, novelty relative to the current S3
signature, symmetry, portability, and mechanical cost. It cannot claim R4800
ranking gain, gameplay improvement, a selected substrate, or progress to 100
points until matched MLX and gameplay experiments pass.

## Preproduction Outcome

No V1 production shard was launched.

The first excluded calibration run exposed a false novelty count because the
complete-encoding hash included relative-seat identity while the control hash
did not. That implementation defect was fixed and regression tested before any
production evidence existed.

The corrected ten-game excluded calibration then produced:

- 35,200 exact topology checks with zero failures;
- 9,600 D6 checks with zero failures;
- all four adversarial checks passed;
- 5,677 current-S3 collision pairs;
- zero topology, path, random-walk, spectral, or complete-feature separations;
- 913 boards with long-range paths;
- 173 boards with geometric holes;
- 1,681-byte median encoding;
- 469 microsecond median extraction; and
- 3.132 millisecond P99 extraction while ten games competed concurrently.

This invalidated two V1 measurement choices:

1. Current-S3 collision separation measures the prevalence of exact signature
   collisions, not whether S6 features are active, learnable, or useful. The
   observed collision groups were repeated equivalent signatures, so the gate
   could not test the intended hypothesis.
2. Per-position timing inside a ten-way Rayon corpus run measures CPU
   contention, not isolated serving cost.

The exact encoding remains viable. ADR 0165 preserves it and replaces only the
misaligned foundation gates with corpus feature variation and a separate
single-thread timing probe.

See
`docs/v2/reports/s6-topological-spectral-foundation-v1-preproduction-rejection.md`.
