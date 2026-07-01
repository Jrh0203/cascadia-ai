# State-Footprint Census V1 Result

Date: 2026-06-16

Final classification:
**`state_footprint_census_complete`**

Scientific BLAKE3:
`c6076545aa93e78902b739eefef1545a23b8f2dbe44770f427a30969511800e5`

## Decision

The historical 441-cell lattice is materially oversized for the measured
AAAAA four-player state distribution. The correct regular compact controls
are complete centered hex disks:

| Radius | Cells | Role after F2 |
|---:|---:|---|
| 4 | 61 | aggressive compressed arm with exact overflow |
| 5 | 91 | near-lossless compressed arm with exact overflow |
| 6 | 127 | empirical lossless compact control with exact overflow |

There is no complete centered 121-cell hex disk. The exact formula is
`1 + 3r(r + 1)`, so radius 5 contains 91 cells and radius 6 contains 127.
Any historical 121-cell representation was an irregular crop or used a
different indexing convention.

The word "lossless" above is deliberately empirical. Both frozen legal
23-tile adversarial boards overflow radius 6 even after optimal integer
recentering. R0-B must therefore retain an exact overflow path; it may not
silently clip merely because no measured corpus state overflowed.

## Evidence

The frozen run combined:

- 625 pattern-aware games on john4, raw seeds 73000 through 73624;
- exactly 50,000 generated pre-move states;
- exactly 200,000 generated board observations;
- all 560 open train and 240 open validation graded-oracle groups;
- 3,200 graded board observations;
- 2,995,314 complete-action candidate rows validated from source datasets;
- 14,091 distinct complete-candidate destinations; and
- complete radius 3 through 8 tables for occupied cells, frontier cells,
  selected destinations, candidate destinations, wildlife and terrain
  firings, habitat components, wildlife adjacencies, and sparse token counts.

The immutable source bundle had V2 source BLAKE3
`6662bfa6d508fe14b415e51c3a9ff41843793d437e62d68fb845de463a503631`.
The exact release binary had BLAKE3
`037259d8dab116094cd829498dd6040ebbcaf4cb3b40947be4decd25662848ee`.
Whole-tree fanout to john4 and returned-report collection were checksum
verified. Merge rejected source or executable drift.

## Generated Games

Best-integer recentering materially changes the conclusion:

| Disk | Occupied retained | Frontier retained | Selected destinations retained | Boards with any overflow |
|---|---:|---:|---:|---:|
| 91 cells, radius 5 | 100.0000% | 99.9751% | 99.9900% | 0.0620% |
| 127 cells, radius 6 | 100.0000% | 100.0000% | 100.0000% | 0.0000% |

At radius 5, all 2,575,000 occupied-cell events, all wildlife firings, all
terrain-edge firings, all allowed-wildlife firings, all habitat components,
and all wildlife adjacencies were retained. The remaining overflow consisted
of 861 frontier-cell events and 5 selected destinations across 124 of 200,000
board observations.

At radius 6, every measured generated event was retained after recentering.
The observed recentered maxima were:

- occupied support: radius 5;
- legal frontier: radius 6; and
- selected action destination: radius 6.

Fixed-origin radius 6 was not lossless: it missed 38 occupied cells, 1,113
frontier cells, and 4 selected destinations across 482 boards. Recentering is
therefore part of the representation contract, not an optional optimization.

## Open Graded Corpus

| Disk | Occupied retained | Frontier retained | Selected retained | Candidate retained | Groups with candidate overflow |
|---|---:|---:|---:|---:|---:|
| 61 cells, radius 4, recentered | 100.0000% | 97.9487% | 99.3750% | 98.1690% | 5.2500% |
| 91 cells, radius 5, recentered | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 0.0000% |

The open train and validation boards had recentered occupied radius at most 4.
Their frontier, selected destination, and complete-candidate destination
support reached radius 5. This independently confirms that board occupancy
alone is an unsafe sizing statistic: legal affordances require a larger
boundary than placed tiles.

## Integrity

All frozen completion gates passed:

- the generated origin was exactly 625 games and 50,000 states;
- all four boards were measured for every state;
- train and validation manifests reconciled to exactly 800 groups and
  2,995,314 candidate rows;
- no duplicate graded group was skipped;
- all radius tables and cohort tables were complete;
- all 482 generated and 15 graded radius-6 outliers were retained without
  truncation;
- both legal adversarial boards overflowed radius 6 as expected;
- all 12 D6 transforms preserved radius;
- merge order was deterministic; and
- sealed test, gameplay evaluation, teacher rollout, MLX training, cloud, and
  external compute remained closed.

The complete distributed run took 6.52 seconds of combined wall time:
5.01 seconds for the generated arm, 1.50 seconds for the validated graded
scan, and 0.006 seconds for the final merge.

## Research Consequences

1. Promote exact untruncated sparse coordinates as R0-A.
2. Promote recentered radius 6 / 127 cells plus exact overflow as R0-B.
3. Keep recentered radius 5 / 91 cells plus exact overflow as R0-C.
4. Keep recentered radius 4 / 61 cells plus exact overflow as R0-D.
5. Treat 441 cells only as a historical diagnostic arm.
6. Do not create a nominal "121-cell disk" arm.
7. Measure throughput and strength separately in R0; this census establishes
   support coverage, not score parity or the 100-point result.

