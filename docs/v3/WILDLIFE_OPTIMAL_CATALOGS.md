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
- `connected_model_infeasible`.

`UNKNOWN`, timeout, an incomplete ledger, or a good heuristic score is never
an optimality certificate.

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

- `docs/v3/evidence/aaaaa_wildlife_catalog_2026-07-22.json`: machine-readable
  AAAAA catalog ledger.
- `docs/v3/AAAAA_WILDLIFE_CATALOG.md`: human-readable AAAAA summary and all
  boards.
- `docs/v3/evidence/aaaaa_wildlife_catalog_first_pass_2026-07-23.json`:
  immutable first-pass evidence retained for mixed-proof provenance.
- `docs/v3/evidence/cbddb_wildlife_catalog_2026-07-23.json`: machine-readable
  CBDDB catalog ledger.
- `docs/v3/CBDDB_WILDLIFE_CATALOG.md`: human-readable CBDDB summary, holistic
  maximum, and all boards.

## Current status

As of 2026-07-23 05:30 EDT, AAAAA has 710/826 formally certified vectors and
is running a hash-pinned hinted retry over the remaining 116. The CBDDB exact
model, independent scorer, Rust candidate generator, and production verifier
are implemented and tested; its exhaustive candidate/proof run starts only
after the AAAAA catalog is complete.
