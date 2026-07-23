# Pure-Wildlife Optimal Board Catalogs

## Scope

This analysis asks a deliberately isolated Cascadia question: place exactly
20 wildlife tokens on distinct connected hexes, choose any animal counts up to
a fixed per-species cap, and maximize wildlife-card points. Habitats, habitat
corridors, tile wildlife restrictions, drafting, market composition, Nature
tokens, and every other game mechanic are ignored.

The occupied cells remain connected because a legal Cascadia tile board is
connected. With 20 cells, translating one token to `(0, 0)` guarantees every
other occupied coordinate is within hex distance 19. The exact models search
that complete radius-19 domain; candidate generators use compact boards only
to find lower bounds and never restrict a proof.

Species and all score breakdowns use this fixed order:

1. Bear
2. Elk
3. Salmon
4. Hawk
5. Fox

## Count-vector state space

For cap six, a count vector `(b,e,s,h,f)` satisfies

```text
b + e + s + h + f = 20
0 <= b,e,s,h,f <= 6
```

The number of vectors is the coefficient of `x^20` in
`(1+x+...+x^6)^5`:

```text
C(24,4) - 5*C(17,4) + 10*C(10,4) = 826
```

At cap seven, the analogous coefficient is:

```text
C(24,4) - 5*C(16,4) + 10*C(8,4) = 2,226
```

That is 1,400 additional vectors, or approximately 2.695 times the cap-six
catalog. The cap-seven elementary all-board upper bounds are 74 for AAAAA
after a fox-incidence refinement and 102 for CBDDB. Full derivation:
[WILDLIFE_CAP7_UPPER_BOUNDS.md](WILDLIFE_CAP7_UPPER_BOUNDS.md).

If the cap is raised to eight, the analogous coefficient is:

```text
C(24,4) - 5*C(15,4) + 10*C(6,4) = 3,951
```

That is 3,125 additional count vectors, or **4.783293x** the cap-six
catalog. It also makes individual exact models larger because a species can
have eight labeled tokens rather than six; the total proof cost therefore
grows by more than the count-vector multiplier alone.

## Rulesets

### AAAAA

- Bear A: score isolated bear pairs.
- Elk A: maximize disjoint straight elk groups of length one through four.
- Salmon A: score valid, unbranched salmon components by length.
- Hawk A: score the number of hawks with no adjacent hawk.
- Fox A: for each fox, count distinct adjacent wildlife species.

The independently certified holistic upper bound is 68. Its proof excludes
every score of 69 or more across all legal max-six count vectors. The
published 68-point witness has counts `6/4/6/0/4` and breakdown
`19/13/20/0/16`.

### CBDDB

- Bear C: components of sizes one, two, and three score 2, 5, and 8;
  components of size four or more score zero; add three for having at least
  one component of every scoring size.
- Elk B: maximize a disjoint packing of singles (2), adjacent pairs (5),
  triangles (9), and strict four-elk rhombi (13).
- Salmon D: every valid unbranched component of length at least three scores
  its length plus the number of distinct adjacent non-salmon tokens.
- Hawk D: visible nonadjacent hawk pairs score 4, 7, or 9 for one, two, or at
  least three distinct intervening non-hawk species; maximize a weighted
  matching so each hawk is used at most once.
- Fox B: a fox scores 3, 5, or 7 when one, two, or at least three non-fox
  species each occur at least twice among its six neighbors.

The independent Python scorer is tested against the production Rust
`score_board(..., ScoringCards::CBDDB)` semantics, including strict rhombi,
branched salmon rejection, blocked hawk sight lines, matching conflicts, and
fox thresholds.

## Optimization and proof method

Each count vector is handled in two stages.

### 1. Construct a strong connected incumbent

A parallel Rust simulated annealer searches both animal assignments and
connected 20-cell polyhex shapes. Its custom ruleset scorer is regression
tested against the production Rust scorer. Before an incumbent is retained,
the production scorer checks its complete B/E/S/H/F breakdown.

This stage proves nothing unless the board exactly reaches a sound upper
bound. Its purpose is to make the exact question as narrow as possible:
“does any board score at least incumbent + 1?”

### 2. Exclude every better board exactly

The CP-SAT formulation uses exactly 20 labeled tokens. It models axial
coordinates, non-overlap, per-species label ordering, and—when required—a
rooted connectivity certificate. Pairwise adjacency is reified exactly from
coordinate differences.

Card-specific score variables represent the full ruleset. The runner first
asks a disconnected relaxation. If even that relaxation is infeasible, no
connected board can improve the incumbent. Otherwise it asks the exact
connected model. A feasible witness is independently rescored and becomes the
new incumbent; the threshold advances until either the upper bound is reached
or the next threshold is proved infeasible.

The only accepted per-vector certificates are:

- `witness_matches_count_relaxation`;
- `witness_matches_global_upper_bound` (AAAAA only);
- `disconnected_relaxation_infeasible`;
- `connected_model_infeasible`;
- `standalone_maximum_motif_incompatibility`;
- `zero_hawk_relaxed_local_packing_infeasible`;
- `hawk_one_loss_relaxed_local_packing_infeasible`;
- `gap_one_joint_salmon_local_packing_infeasible`.

`UNKNOWN`, timeout, an incomplete ledger, or a good heuristic score is never
an optimality certificate.

### 3. Exact tail certificates

The coordinate model's hardest cases miss their elementary upper bound by
only one or two points. For those cases, the score target forces specific
maximum card motifs. Separate deterministic certificates enumerate free
unbranched salmon polyhexes, straight elk group partitions, Bear-A pairs,
Fox-A species observations, and—where present—Hawk-A coverage. They then solve
a smaller cell-set-packing problem around the forced salmon/fox neighborhoods.

These are upper-bound proofs, not heuristic shortcuts. Each deliberately drops
constraints such as whole-board connectivity, Bear/Hawk isolation, or remote
noncovering motifs, creating a strict superset of legal boards. If even that
superset cannot beat the incumbent, the legal board cannot. Disconnected
salmon scoring components are enumerated through every relative separation at
which a length-four elk line or shorter motif can interact; all farther
translations share a factorized representative. Every certificate is
source-hashed, rerunnable, and independently checks its incumbent. The base
catalog writer exited naturally at 11:07 EDT; its final ledger is now
immutable input to the certificate union.

A generalized relaxation retaining all 20 coordinates but only forced scoring
motifs was also implemented and containment-tested. Two preregistered screens,
including a symmetry-pruned v2, remained `UNKNOWN` on already-certified
calibrations at 30 seconds. That direction is closed: longer generic coordinate
search is not treated as useful evidence. Tail work uses finite explicit fox
components and local cell-set packing, which removes the free-coordinate
symmetry and has already produced the accepted specialized certificates.

A 2026/2025 primary-literature review then tested three stronger transfers:
layered unique-token anchoring, exact radius-two fox relation tables, and
canonical per-species Fox-A witnesses with common-witness ring geometry. All
passed containment but failed preregistered known-exclusion strength gates.
They produced no catalog proof and were not run over unresolved rows. The
measured conclusion is that canonicality and specialized propagation must be
applied to externally enumerated finite component/profile branches; redundant
tables inside the full coordinate model do not remove enough search. The
sources, measurements, and generalized multi-component implementation
contract are in
[AAAAA_EXACT_TAIL_LITERATURE_REVIEW.md](AAAAA_EXACT_TAIL_LITERATURE_REVIEW.md).

### Why the remaining AAAAA tail is harder

The unresolved tail is dominated by Fox-A geometry: every one of the 98
remaining vectors has five or six foxes, 81 have six, and 96 contain all five
species. The additive count relaxation treats every fox's wildlife-type
observations independently. For six foxes with every type present, it awards
the full 30 fox points without asking whether all six fox neighborhoods can
simultaneously contain Fox, Bear, Elk, Salmon, and Hawk.

On a real hex board those observations share scarce neighboring tiles. At the
same time, Bear A wants isolated pairs, Elk A wants rigid straight groups,
Salmon A wants unbranched components, and Hawk A wants hawks separated from
one another. The strongest board is easy to exhibit; the hard direction is
proving that no one-point-better arrangement exists among many symmetric
near-solutions. The exact coordinate model must couple 20 positions, all 190
pair adjacencies, card-motif subset variables, and a rooted connectivity
certificate. In the first fleet pass, every one of the 102 incomplete rows
left both its disconnected and connected threshold solves `UNKNOWN`, after a
median 1.85 million branches and 410 thousand conflicts. Specialized
high-fox overlap and local-packing certificates attack this missing global
interference more directly than simply extending the generic timeout.

## Independent validation

Every final board must pass all of the following:

1. exactly 20 tokens;
2. the requested count vector and per-species cap;
3. 20 distinct coordinates;
4. one connected occupied component;
5. independent Python scoring equal to the stored breakdown and total;
6. custom Rust scoring equal to the stored breakdown and total;
7. production Rust scoring equal to the stored breakdown and total;
8. a complete exact certificate for that count vector.

Catalog JSON records source hashes, candidate hashes, solver configuration,
per-attempt status/bounds/timing, proof method, and the full coordinate list.
Imported proofs retain the hash and configuration of the exact model that
created them.

## Artifacts

- `docs/v3/evidence/wildlife_cap7_upper_bounds_2026-07-23.json`:
  exhaustive cap-six/cap-seven upper-bound comparison for both rulesets.
- `docs/v3/evidence/aaaaa_wildlife_catalog_2026-07-22.json`: machine-readable
  AAAAA catalog ledger.
- `docs/v3/AAAAA_WILDLIFE_CATALOG.md`: human-readable AAAAA summary and all
  boards.
- `docs/v3/evidence/aaaaa_wildlife_catalog_first_pass_2026-07-23.json`:
  immutable first-pass evidence retained for mixed-proof provenance.
- `docs/v3/evidence/aaaaa_motif_certificate_3_6_6_0_5_2026-07-23.json`:
  exact relaxed-superset certificate that proves `(3,6,6,0,5)` has optimum
  61; retained as independent union evidence.
- `docs/v3/evidence/aaaaa_zero_hawk_certificates_2026-07-23.json`: exact
  relaxed local-packing certificates for `(3,6,5,0,6)=60`,
  `(4,6,4,0,6)=63`, and `(4,5,5,0,6)=64`.
- `docs/v3/evidence/aaaaa_hawk_one_loss_certificates_2026-07-23.json`: exact
  explicit-fox local-packing certificates for `(4,6,4,2,4)=64` and
  `(3,5,4,3,5)=62`.
- `docs/v3/evidence/aaaaa_gap_one_joint_salmon_certificate_2026-07-23.json`:
  exact 113-submodel certificate for `(3,6,3,3,5)=61`; this closes the last
  first-pass raw gap of two or less.
- `docs/v3/evidence/cbddb_wildlife_catalog_2026-07-23.json`: machine-readable
  CBDDB catalog ledger.
- `docs/v3/CBDDB_WILDLIFE_CATALOG.md`: human-readable CBDDB summary, holistic
  maximum, and all boards.
- `docs/v3/evidence/cbddb_wildlife_candidates_2026-07-23.json`: all 826
  independently verified heuristic warm starts; explicitly not exact proof
  evidence.

## Current status

As of 2026-07-23 16:22 EDT, AAAAA has 732/826 formally certified vectors.
The base retry exited naturally with 711/826 exact, adding
`(4,4,2,4,6)=66` beyond its imported 710-row ledger and leaving 115 timeouts.
The first exact fleet pass returned all 115 requested tail rows and added 13
coordinate-model proofs at 60/120-second limits; 102 rows timed out. Three new
proofs overlap the seven frozen specialized certificates, and the base
retry's one new proof was already in the fleet snapshot. A subsequent exact
split-Salmon bitset certificate adds four non-overlapping vectors, leaving 94
unique vectors unresolved after union. Of the prior 98-row tail, 19 were
soundly bounded within two
points of their incumbent; the remaining sound gaps are 3 (30 rows), 4 (31),
5 (15), and 6 (3). Empirically, the deep candidate was never more than one
point low across all 728 certified rows—including 126 certified rows whose
original relaxation gap was at least three—but the unresolved tail is
selection-biased toward the hardest proofs. A timeout remains an incumbent
only, not an optimum claim.

The new bitset certificate exhausts 57/57/57/95 canonical fox-layout and
Bear/Elk/Hawk packing submodels for counts `(4,5,2,3,6)`,
`(5,5,2,2,6)`, `(3,5,2,4,6)`, and `(3,6,2,3,6)`, then combines them with
the already-exact maximum-Salmon branches. Optima are respectively
66/63/62/62. Machine-readable certificate:
`docs/v3/evidence/aaaaa_split_salmon_bitset_certificate_2026-07-23.json`.
An exact performance rerun after subset-dominance indexing retained all four
proofs but failed the requested 20× gate: critical path 618.447 seconds,
4.84× versus the frozen sequential screen. The dominant case generated
14,648,710 candidate covers. Species ordering and identical-query caching then
retained all 266 exact submodels while reducing the fresh four-case critical
path to 106.349 seconds, a selected 28.16× versus the frozen sequential
screen.

The online-literature implementation pass added no new proof: its single-
anchor, radius-two relation, and canonical-witness calibrations all failed
their frozen exact-result gates. Their negative result is durable and narrows
the remaining work to canonical external multi-component enumeration plus
local set packing; longer generic coordinate time and additional static table
wrappers are closed.

CBDDB heuristic staging completed all 826 vectors in 224.244154 seconds with
zero independent-score or connectivity failures; the current 84-point leader
is only an incumbent. Its frozen full taskset is ready for four-host
207/207/206/206 sharding on john1–john4, but the requested AAAAA-then-CBDDB
ordering blocks launch until AAAAA is complete.
