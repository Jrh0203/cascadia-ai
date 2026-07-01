# S6 Topological And Spectral Foundation V2 Preregistration

Date: 2026-06-17

ADR: 0165

Experiment: `s6-topological-spectral-foundation-v2`

Protocol: `s6-exact-topology-activation-census-v2`

Status: frozen before V2 calibration and production

## Question

Are the exact S6 topology, path, random-walk, and Laplacian feature families
active and varied on natural Cascadia states, D6 invariant, and cheap in an
isolated serving-like timing probe?

V2 does not use exact current-S3 collision separation as a promotion gate.
V1 showed that the collision groups were repeated equivalent signatures and
therefore could not test the intended hypothesis.

## Immutable Encoding

The V1 encoding is unchanged. The only scientific changes are:

- feature-family unique-count metrics;
- a feature-variation gate;
- an isolated single-thread timing probe; and
- revised classification names.

Relative-seat identity is excluded from every novelty or variation hash.

## Production Corpus

```text
four-player AAAAA, no habitat bonuses
4 hosts
10 unique production games per host
800 production positions per host
3,200 total production positions
12,800 total board encodings
```

Host, timing-seed, and production-seed assignments are frozen in ADR 0165.

## Per-Host Gates

```text
topology decoder checks = boards * 11
topology decoder failures = 0
D6 checks = positions * 12
D6 failures = 0
adversarial checks = 4
adversarial failures = 0
unique topology encodings >= 128
unique path encodings >= 16
unique random-walk encodings >= 128
unique Laplacian encodings >= 128
unique full encodings >= 256
boards with long-range paths > 0
isolated timing samples = 80
isolated P99 extraction <= 2,000,000 ns
median production encoding <= 4,096 bytes
```

Parallel-corpus extraction timing and current-S3 collision separation remain
reported but cannot promote or reject V2.

## Aggregate Gates

- one report from every registered host;
- exact host and seed assignment;
- disjoint production and timing seeds;
- one immutable bundle and executable digest;
- all four hosts classify
  `s6_topological_spectral_foundation_v2_authorized`; and
- forward/reverse report order produces identical science.

## Predictions

1. Topology, random-walk, Laplacian, and complete encodings will each exceed
   128 unique values per ten-game shard.
2. Marked paths will be less diverse but exceed 16 unique values.
3. The isolated P99 will remain below 2 ms even when the parallel corpus P99
   exceeds 2 ms under CPU saturation.
4. Current-S3 collision separations will remain near zero and will be treated
   as diagnostic evidence only.
5. Median encoding will remain near 1.7 KiB.

## Invalidators

- any V2 calibration or production before this document;
- changing the V1 encoding while claiming a measurement-only version;
- including relative seat in feature-variation identity;
- timing under parallel corpus contention as the isolated gate;
- hidden state, future refill labels, or teacher targets;
- overlapping seed blocks;
- source or executable drift; or
- order-dependent aggregation.

## Claim Boundary

V2 can authorize learned ablations. It cannot claim that any S6 family
improves action ranking, gameplay score, search, or the 100-point objective.
