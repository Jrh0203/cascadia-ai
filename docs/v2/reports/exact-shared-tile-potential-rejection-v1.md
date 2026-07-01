# Exact Shared Tile Potential Rejection

Status: **rejected and removed**

Date: 2026-06-15

## Hypothesis

Nature Token candidate combinations can pair one market tile and frontier
coordinate with several wildlife choices. The treatment prepared the exact
tile-only portion of board potential once, then evaluated only the additional
wildlife-placement changes for each distinct pairing.

Candidate coverage and ordering, habitat previews, wildlife selection,
potential arithmetic, feature construction, MLX requests, search allocation,
random streams, and the K32/R600 full-terminal contract were unchanged.

## Exactness

The treatment matched direct incremental potential, complete board-potential
recomputation, and the unshared candidate set across six complete seeded
four-player AAAAA games, every exercised tile rotation, every tile-only
afterstate, and every legal wildlife afterstate. Both library configurations
passed before timing.

Every source and PGO timing run reproduced:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches;
- 6,121,807 logical and 5,062,305 physical neural rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps, zero policy fallbacks, and clean shutdown.

After removing the rejected implementation, the complete default and
`mid-features,v4-opp` suites each passed 84 tests. The rebuilt control's
`__TEXT,__text` section has the same offset, size, and SHA-256
`68334bc49378b86ddcf28034301ad5dcbffb1c52ab0045f60a680fe32bc12873`
as the retained pre-experiment control. Whole-file Mach-O hashes differ only
because the rebuild changed linker UUID and debug metadata.

## Source-Level Screen

Matched non-PGO binaries were crossed across john2 and john3.

| Host | Control mean | Treatment mean | Speedup |
|---|---:|---:|---:|
| john2 | 15.749057 s | 15.596178 s | 1.00980x |
| john3 | 15.605873 s | 15.368633 s | 1.01544x |
| Combined | **15.677465 s** | **15.482406 s** | **1.01260x** |

The 1.244% combined improvement cleared the preregistered 0.25% advancement
floor, so the experiment advanced to fresh PGO.

## Race-Free PGO

The instrumented treatment was trained once per host with
`RAYON_NUM_THREADS=1`. The two profiles differed by only 47,661 counts out of
117.475 billion and were merged into a profile containing 5,545 functions and
119,564 blocks.

The fresh production candidate was then crossed against the accepted
elk-potential PGO champion:

| Host | Control mean | Treatment mean | Speedup |
|---|---:|---:|---:|
| john2 | 15.369275 s | 15.382703 s | 0.99913x |
| john3 | 15.095387 s | 15.025399 s | 1.00466x |
| Combined | **15.232331 s** | **15.204051 s** | **1.00186x** |

PGO reduced the aggregate signal to 0.186%, and john2 regressed by 0.087%.
The treatment therefore failed the registered requirement to be reproducibly
faster on both workers.

## Verdict

Reject and remove. The exact reuse is real at source level, but profile-guided
layout and inlining already absorb almost all of its production benefit. A
host-dependent 0.186% aggregate result is not enough evidence to replace the
accepted binary or to count toward the mandatory 10x gate.

The accepted elk-potential PGO champion remains unchanged. The Phase 0
threshold remains `14.1027296` seconds, and the accepted result remains about
`15.515981` seconds.

Machine-readable evidence:
`docs/v2/reports/exact-shared-tile-potential-rejection-v1.json`.
