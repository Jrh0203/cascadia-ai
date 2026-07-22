# Exact AAAAA pure-wildlife optimum

## Result

Under the model John specified, the maximum wildlife-card score is **68**.
The model places exactly 20 wildlife tokens on distinct, connected hexes, uses
at most six of each species, applies scoring cards AAAAA, and ignores habitats,
tile wildlife restrictions, Nature tokens, drafting, achievements, and every
other game mechanic.

One optimal species allocation is:

| Species | Count | Score |
|---|---:|---:|
| Bear | 6 | 19 |
| Elk | 4 | 13 |
| Salmon | 6 | 20 |
| Hawk | 0 | 0 |
| Fox | 4 | 16 |
| **Total** | **20** | **68** |

`B`, `E`, `S`, and `F` denote bear, elk, salmon, and fox. Dots are empty and
indentation shows the axial-hex row offset.

```text
r= 0  . . . E
r= 1   . B F E
r= 2    B S F E
r= 3     . S B E
r= 4      S B F F
r= 5       S S S B
r= 6        . . B .
```

The six bears form three isolated pairs, the four elk form one length-four
line, and the six salmon form one valid run. Every fox touches four distinct
species: bear, elk, salmon, and fox.

## Reproduce the output

Print the certified layout and verify it with the Rust production scorer:

```bash
cargo run -q -p cascadia-game --bin aaaaa_wildlife_solver -- --show-optimum
```

Re-run the exact upper-bound proof (the JSON file is checkpointed after every
count allocation and `--resume` continues a clean infeasibility prefix):

```bash
uv sync --dev
uv run python -m tools.aaaaa_wildlife_exact \
  --minimum-score 69 \
  --time-limit 120 \
  --workers 8 \
  --output target/aaaaa-wildlife-solver/exact-result.json
```

The exact solver prints `PROVED OPTIMAL: 68 wildlife points` only after every
allocation capable of reaching 69 has been proved infeasible. The Rust binary
also contains a stochastic incumbent search; omit `--show-optimum` to run it.

## Why the proof is exhaustive

There are 826 species-count vectors satisfying the 20-token total and the cap
of six. A count-only relaxation awards each non-fox species its best possible
standalone score and lets every fox see every species present. Only 128 vectors
have a relaxed score of at least 69, so the remaining 698 cannot beat the
68-point witness regardless of geometry.

The CP-SAT proof solves those 128 vectors independently. Its geometry uses 20
labeled token coordinates rather than a finite collection of candidate board
cells. Every ≥69 vector contains a fox. The lexicographically first fox is
translated to the origin; because any connected 20-cell polyhex has graph
diameter at most 19, a radius-19 coordinate domain contains every possible
layout. Ordering coordinates within each species removes token permutations
without removing physical layouts.

The model enforces distinct occupied coordinates, hex adjacency, and
single-commodity flow connectivity. It encodes isolated bear pairs, disjoint
straight elk lines, complete connected salmon components with maximum
same-species degree two, isolated hawks, and each fox's distinct adjacent
species. The bundled witness is checked twice: by an independent Python score
implementation and by `cascadia_game::score_board` with `ScoringCards::AAAAA`.

The machine-readable proof record is
`docs/v3/evidence/aaaaa_wildlife_optimum_2026-07-22.json`. It records every
allocation status, per-allocation solver parameters and statistics, the
OR-Tools version, assumptions, and the optimal coordinates.
