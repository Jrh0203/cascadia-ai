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
  Each assignment group and target species form a bipartite cross-edge graph.
  All 36 cap-six connected-component size pairs were solved exactly on the
  complete finite hex disk, then a component DP covered disconnected support.
  The resulting 0..6 edge table has maximum 17 rather than the planar
  relaxation's 20 at `(6,6)`.
- Fox A: aggregate common-neighbor capacities over every subset of observed
  non-fox species. A target pair has at most two common hex neighbors; three
  or more distinct target cells have at most one.
- Fox B: every doubled-species qualification uses a pair of target tokens,
  and one target pair can have at most two qualifying fox neighbors. The
  stronger selected table exactly maximizes foxes with at least two neighbors
  of one target class for every cap-six side-size pair, again using connected
  components plus a complete disconnected-support DP.

The proof runner skips any count branch whose upper bound is at most the
current incumbent. Its production mode solves the connected coordinate model
directly. A disconnected relaxation remains available as a deliberately
separate proof screen: infeasibility there also proves the connected problem
infeasible, while a feasible disconnected layout is not a valid board.

### 5. Fleet execution

Rulesets and remaining count branches are deterministic independent work
units. `tools/all_wildlife_proof_plan.py` assigns rulesets by deterministic
longest-processing-time balancing, using the number of count branches still
above each incumbent as weight. Production artifacts are sharded over
john1–john4 with atomic checkpoints, heartbeats, single-use tags, and exact
source/input hashes.
The proof identity separately pins the coordinate model, its shared exact
support module, the global runner, and the sound bound/scorer source; changing
any one cannot silently resume an older identity.
Returned boards and ledgers are collected only after every shard is terminal,
then rescored and revalidated on john1 before catalog publication.
`tools/all_wildlife_proof_catalog.py` refuses duplicate ruleset proofs,
candidate/proof identity mismatches, disconnected incumbents, cap or scoring
mismatches, connectivity-mode identity mismatches, and inconsistent unresolved
branch bookkeeping. It writes the machine-readable JSON catalog and the
1,024-board Markdown catalog atomically. Incomplete rows are visibly labeled
as unproven incumbents rather than optima.
When both modes are present, it validates each ledger independently and unions
only exact infeasibility thresholds; a disconnected relaxation can strengthen
the upper proof but never replace the connected, independently rescored
incumbent witness.
`tools/verify_all_wildlife_candidate_catalog.py` then requires the final
catalog and every row to be proof-complete, recomputes its holistic summary,
and batch-compares every board with the production Rust scorer.

## Current measured state

- Independent/production score comparisons: `4,096/4,096` passed.
- Exact fixed-board ruleset comparisons: `1,024/1,024` passed.
- Release candidate rate: about 0.11 CPU-seconds per 10,000 evaluated layouts
  in the initial AAAAA microbenchmark.
- Frozen shallow 64-ruleset pilot after all four Hawk/Fox filters: mean
  global-bound gap `10.984375`; median
  count branches above the incumbent `268`, mean `278.75`, maximum `689`.
- AAAAA's count ceiling is `72` (down from `73`) and only 108 count
  allocations remain above its certified 68-point incumbent (down from 128).
- CBDDB's count ceiling is `99` (down from `100`) and 309 allocations remain
  above its 84-point incumbent (down from 332).
- Known AAAAA global-leader count `(6,1,6,2,5)` excludes score 69 in the
  generalized disconnected model in 12.4 seconds.
- CBDDB count `(6,0,3,6,5)` did not exclude score 85 in 60 seconds; it is a
  proof-tail calibration failure, not evidence that 85 exists.
- The first eight-rule connected proof calibration completed 2/8 rows within
  five minutes. ACACA 76 and ADACA 77 are certified; the other six remain
  incumbents. This triggered the preregistered rejection band, so the generic
  1,024-row proof shape was not launched.
- The selected exact Fox-C lattice table cuts the frozen Fox-C frontier by
  70.01% and the all-ruleset frontier by 22.33%; its 36/36 optimal component
  derivation is retained at
  `docs/v3/evidence/hex_bipartite_edge_bounds_2026-07-23.json`.
- The selected exact Fox-B qualification table cuts the frozen Fox-B frontier
  by 9.66%, ADCCB 94→71 branches, and CBDDB 309→283. Its 36 component proofs
  are retained at
  `docs/v3/evidence/hex_fox_qualification_bounds_2026-07-23.json`.
- A 216-component exact Fox-A two-target-species derivation matched the
  existing `min(foxes,2ab)` capacity in every entry, giving 0% reduction. It
  is a documented negative, not an additional production table:
  `docs/v3/evidence/hex_dual_observation_bounds_2026-07-23.json`.
- The same-budget specialized connected recalibration remained 2/8 complete.
  A disconnected-relaxation prescreen completed no additional row; exact
  exclusion union reduced the six-row unresolved total only 997→994 (0.30%),
  below its preregistered 20% selection gate. Both generic coordinate shapes
  are rejected for full production, while every exact exclusion remains
  reusable.
- Seven rulesets are currently integrated as certified: AAAAA 68 through its
  complete 128-allocation global certificate; ACACA 76 and ADACA 77 by
  coordinate exclusion; plus bound-matched DCAAC 69, DCCAC 69, DDAAC 72, and
  DDCAC 72.
  Proof-less bound matches are now recognized directly by the collector and
  covered by a regression test.
- The recovered 40× hard-row candidate calibration passed all independent and
  production checks. Its six boards strictly improve 17 rulesets after
  cross-scoring, led by CADAC 66→68 and DDDDD 78→79; the total proof frontier
  falls 119,959→119,273. The holistic candidate remains 85 on the same eight
  rulesets. These are stronger warm starts, not optimum claims:
  `docs/v3/evidence/all_wildlife_candidate_deep_recovery_2026-07-23.json`.
- The fail-closed catalog augmenter recovered both legacy complete coordinate
  shards from their hash-pinned fleet ledger, rebased all 17 deep incumbent
  improvements, imported AAAAA, and production-rescored all 1,024 boards.
  That first integrated catalog had 7 complete rulesets and 119,139
  unresolved count branches. Curated evidence:
  `docs/v3/evidence/all_wildlife_catalog_augmentation_2026-07-23.json`.
- The seven rulesets with one remaining branch each were frozen and run as a
  connected exact 2/2/2/1 fleet pass. All seven threshold queries returned
  `INFEASIBLE` in 1.29–9.40 seconds, with no timeout or unknown. The
  independently and production-rescored catalog is now 14/1,024 exact with
  119,132 branches across 1,010 unresolved rulesets. The score-85 leader on
  eight rows is still the holistic incumbent, not a proof:
  `docs/v3/evidence/all_wildlife_near_complete7_2026-07-23.json`.
- The next eight rulesets had exactly two remaining branches each. All 16
  connected exact threshold queries returned `INFEASIBLE` in 0.77–3.39
  seconds, with no timeout/unknown. Independent and production rescoring
  advanced the integrated catalog 14→22/1,024 exact; 119,116 branches remain
  across 1,002 rows. This is strong near-tail throughput evidence, not a
  whole-frontier extrapolation:
  `docs/v3/evidence/all_wildlife_two_branch8_2026-07-23.json`.
- The complete 3–5-branch slice certified 22/23 rulesets and 93/95 count
  thresholds in 6m34s fleet wall. DCACC 77 proved three exclusions but
  retains two `UNKNOWN` branches after the fixed row cap. The integrated
  catalog advances 22→44/1,024 exact with 119,023 branches across 980 rows.
  This is the first measured transition from seconds-scale rows into the
  timeout tail:
  `docs/v3/evidence/all_wildlife_three_to_five_branch23_2026-07-23.json`.
- A coupled same/cross-edge and degree-six relaxation passed all 846,848
  count/ruleset containment comparisons but equaled the existing bound in
  every cell. It is not selected; pair-edge resource coupling without richer
  geometry is closed:
  `docs/v3/evidence/all_wildlife_coupled_edge_bound_2026-07-23.json`.
- A correct twelve-way fox-anchor dihedral break regressed a known exact
  exclusion from 20.55 seconds to `UNKNOWN` at 30 seconds and resolved no
  hard case. It was removed and the prior exact source restored:
  `docs/v3/evidence/all_wildlife_dihedral_calibration_2026-07-23.json`.
- A reproducible anchor-centroid variant preserving all but one fox ordering
  also regressed the known proof and resolved no hard case. It was removed:
  `docs/v3/evidence/all_wildlife_anchor_centroid_calibration_2026-07-23.json`.
- A discrete in-model score table missed its runtime gate but cut CADAC
  branches 35% and CBDDB 20%. It remains default-off and triggers external
  score-profile sharding for the tractable 2/6/29-profile calibration cases:
  `docs/v3/evidence/all_wildlife_score_profile_calibration_2026-07-23.json`.
- The frozen 37-profile external calibration completed across john1–john4:
  5 profiles were exactly infeasible and 32 remained unknown (known AAAAA
  1/2, hard AAAAA 4/6, CADAC 0/29). It misses both selection conditions and
  is not scaled. Its five exclusions remain reusable; collection evidence:
  `docs/v3/evidence/all_wildlife_profile_shard_calibration_2026-07-23.json`.
- The first canonical component-local bitset engine exactly closed all 266
  registered split-Salmon packing submodels and certified four AAAAA
  count-vector optima. Only one of those four was still in the arbitrary-
  ruleset AAAAA frontier, so its reusable exclusion reduces that row 23→22.
  Formal certificate:
  `docs/v3/evidence/aaaaa_split_salmon_bitset_certificate_2026-07-23.json`.

The durable chronological configurations, hashes, failures, and decisions are
in `cascadiav3/EXPERIMENT_LOG.md`.
