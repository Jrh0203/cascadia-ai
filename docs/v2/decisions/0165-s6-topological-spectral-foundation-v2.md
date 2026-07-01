# ADR 0165: S6 Topological And Spectral Foundation V2

Status: accepted; foundation authorized

Date: 2026-06-17

Experiment: `s6-topological-spectral-foundation-v2`

Protocol: `s6-exact-topology-activation-census-v2`

Supersedes: ADR 0164 before production

## Context

ADR 0164's excluded calibration proved the S6 encoding exact and D6 invariant,
but rejected its production measurement design:

- exact collisions under the already-rich S3 scalar signature were repeated
  equivalent boards, so collision separation could not measure feature
  activation; and
- timing feature extraction inside ten concurrent games measured contention
  rather than isolated serving cost.

No V1 production process or immutable production bundle existed.

## Decision

Preserve the exact S6 encoding:

- occupied, habitat, and wildlife graph channels;
- scalar topology;
- geometric holes;
- marked Elk and Salmon path summaries;
- deterministic fixed-point random-walk returns; and
- exact combinatorial-Laplacian trace moments.

Change only the foundation measurement:

1. report current-S3 collision counts as negative diagnostic evidence, not a
   promotion gate;
2. count unique natural encodings for each S6 feature family;
3. require meaningful feature variation and long-range corpus coverage;
4. run one isolated single-thread timing game before the parallel corpus; and
5. gate serving cost on that isolated probe, while retaining parallel timing
   only as throughput diagnostics.

## Feature-Variation Gates

Each ten-game production shard must observe at least:

```text
unique topology encodings >= 128
unique marked-path encodings >= 16
unique random-walk encodings >= 128
unique Laplacian encodings >= 128
unique complete S6 encodings >= 256
boards with long-range paths > 0
```

These gates establish that every family is active on natural Cascadia states.
They do not establish predictive value.

## Isolated Timing

Before starting its parallel production games, each host runs one
single-thread 80-position timing game at `first_seed - 1`.

The timing probe:

- constructs the same exact S6 encoding;
- excludes D6 replay and corpus collision bookkeeping from the timed region;
- advances with the same deterministic legal-action policy;
- is excluded from production position and feature counts; and
- must have P99 extraction at or below 2 milliseconds.

The parallel corpus still reports extraction time, but that distribution is
not a serving gate.

## Preserved Gates

- 11 exact topology decoder checks per board;
- all twelve D6 transforms per production position;
- four registered adversarial checks;
- zero exactness, D6, or adversarial failures;
- median four-board encoding at or below 4 KiB;
- immutable source and executable identity;
- disjoint seed blocks;
- all four registered hosts; and
- report-order-invariant aggregation.

## Cluster Design

Production remains:

| Host | Timing seed | Production seeds | Games |
|---|---:|---:|---:|
| john1 | 5,609,999 | 5,610,000-5,610,009 | 10 |
| john2 | 5,619,999 | 5,620,000-5,620,009 | 10 |
| john3 | 5,629,999 | 5,630,000-5,630,009 | 10 |
| john4 | 5,639,999 | 5,640,000-5,640,009 | 10 |

## Classification

```text
s6_topological_spectral_foundation_v2_authorized
s6_topology_decoder_failed
s6_d6_invariance_failed
s6_adversarial_separation_failed
s6_feature_variation_futile
s6_long_range_coverage_futile
s6_isolated_latency_failed
s6_encoding_compactness_failed
s6_cross_host_inconsistent
```

## Consequences

A pass authorizes matched MLX feature-family ablations after the compact
substrate is selected. The learned comparison must still prove R4800 and
protected-slice value at bounded complete-decision latency.

## Resolution

Production completed on all four hosts and classified
`s6_topological_spectral_foundation_v2_authorized`.

- 3,200 positions and 12,800 board encodings;
- 140,800 exact topology-decoder checks with zero failures;
- 38,400 D6-invariance checks with zero failures;
- 16 adversarial checks with zero failures;
- worst-host isolated extraction P99 of 1.927 ms;
- worst-host median four-board encoding of 1,686 bytes; and
- order-invariant aggregate scientific BLAKE3
  `8c82103d1b771317013981913f35d370e52a8dec6721af1992f181856f712196`.

The result authorizes matched learned ablations. It does not establish that
any S6 family improves ranking or gameplay.
