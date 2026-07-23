# All wildlife-card rulesets: exact cap-six catalog

Status: implementation and proof-performance calibration in progress. No row
is published as optimal until its certificate is complete.

## Question

For every ordered choice of one A/B/C/D card for Bear, Elk, Salmon, Hawk, and
Fox, find a maximum wildlife-only score and return one board attaining it.
There are `4^5 = 1,024` rulesets. A five-letter ID records cards in
Bear/Elk/Salmon/Hawk/Fox order, so `CBDDB` means Bear C, Elk B, Salmon D,
Hawk D, Fox B.

Every board has exactly twenty wildlife tokens on distinct connected axial
hexes. Each species may occur at most six times. Habitats, tile compatibility,
drafting, Nature tokens, habitat corridors, and all other game mechanics are
outside the problem.

## Correctness contract

Each final catalog row must contain:

- the ruleset ID, optimum, five-species score breakdown, and species counts;
- twenty normalized axial `(q,r)` coordinates and wildlife labels;
- a connected-board and cap-six validation;
- agreement between the independent Python scorer and production Rust
  `score_board`;
- a sound global proof: either the witness meets an all-board upper bound or
  every count/profile branch capable of beating it is proved infeasible;
- source, solver, seed, host, and artifact hashes.

`UNKNOWN`, a timeout, a heuristic incumbent, or an incomplete set of count
branches is never labeled optimal.

## Method

### 1. Independent semantics

`tools/all_wildlife_rules.py` implements all twenty card scorers without using
the Rust scorer. It also enumerates the 826 legal species-count vectors and
computes sound card-aware count bounds. The older AAAAA and CBDDB
implementations are retained as regression oracles.

`all_wildlife_score_oracle` independently constructs the canonical Rust
`Board` and invokes production `score_board`. Four frozen connected boards
crossed with all 1,024 rulesets give 4,096 five-part comparisons. All passed;
the canonical result hash is
`06ee8d41dbd14766291d70022259ac930d6ddbf4fc2d7592be7ce0a9cbbd1bc9`.

### 2. Incumbent search

The release Rust candidate engine uses simulated annealing over both geometry
and species assignment. Mutations preserve twenty distinct connected cells and
the cap-six constraint. One constructed production board is evaluated under
uniform A/B/C/D cards, after which any ruleset selects its five relevant
components. This removes repeated board construction and scores only four
production card bundles per layout.

Candidate search supplies warm starts only. It cannot certify optimality.

### 3. Exact achievement certificates

`tools/all_wildlife_exact.py` is a composable CP-SAT coordinate model.
Species counts are fixed within one branch; token coordinates are distinct,
bounded by the maximum connected-board radius, and optionally connected by a
rooted arborescence.

Rather than trusting a second monolithic score function, each card selects
valid scoring objects:

- full wildlife components for Bear and Salmon;
- disjoint line/shape/component groups for Elk A/B/C;
- an ordered remaining-token construction for Elk D rings;
- isolated/visible hawks and the Hawk D maximum-weight matching;
- Fox neighborhood claims and the Fox D adjacent-pair matching.

Every selected object is a score the represented production board genuinely
earns. Conversely, the production scoring decomposition can select all of its
objects. Thus the maximum certificate equals the real fixed-board score, and
infeasibility at threshold `incumbent+1` excludes all better boards for that
count branch.

The permanent fixed-board verifier checked all 1,024 objectives on a frozen
`(4,4,4,4,4)` board with no mismatch. Its canonical row hash is
`005153bf58a77d32feca858fc225e04db3101d7008b295eabde9fedecb878f2f`.

### 4. Proof filters

Before coordinate solving, each count vector receives a sound upper bound.
Two important non-separable geometric filters are:

- Hawk C: only consecutive hawks on each of the three axial line families
  see one another. Through six hawks the tight visible-edge maxima are
  `0,0,1,3,5,7,9`.
- Fox C: assign every fox to the species whose adjacent count it scores.
  Each assignment group and target species form a simple planar bipartite
  subgraph with maximum degree six, giving an edge ceiling for that group.

The proof runner skips any count branch whose upper bound is at most the
current incumbent. It first tries the stronger disconnected relaxation:
infeasibility there also proves the connected problem infeasible. Only
survivors enter the connected coordinate model.

### 5. Fleet execution

Rulesets and remaining count branches are deterministic independent work
units. Production artifacts are sharded over john1–john4 with atomic
checkpoints, heartbeats, single-use tags, and exact source/input hashes.
Returned boards and ledgers are collected only after every shard is terminal,
then rescored and revalidated on john1 before catalog publication.

## Current measured state

- Independent/production score comparisons: `4,096/4,096` passed.
- Exact fixed-board ruleset comparisons: `1,024/1,024` passed.
- Release candidate rate: about 0.11 CPU-seconds per 10,000 evaluated layouts
  in the initial AAAAA microbenchmark.
- Frozen shallow 64-ruleset pilot after the Hawk-C/Fox-C filters:
  mean global-bound gap `11.1875`; median count branches above the incumbent
  `275.5`, mean `288.016`, maximum `689`.
- Known AAAAA global-leader count `(6,1,6,2,5)` excludes score 69 in the
  generalized disconnected model in 12.4 seconds.
- CBDDB count `(6,0,3,6,5)` did not exclude score 85 in 60 seconds; it is a
  proof-tail calibration failure, not evidence that 85 exists.

The durable chronological configurations, hashes, failures, and decisions are
in `cascadiav3/EXPERIMENT_LOG.md`.
