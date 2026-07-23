# Pure-wildlife cap-seven upper bounds

Date: 2026-07-23
Scope: exactly 20 wildlife tokens, no habitats or other game mechanics, at
most seven tokens of each species.

## Result

Raising the per-species cap from six to seven increases the count space from
826 to 2,226 allocations, but increases the elementary holistic score bounds
only slightly:

| Ruleset | Cap-six comparable bound | Cap-seven bound | Change |
|---|---:|---:|---:|
| AAAAA, geometry-free | 73 | 75 | +2 |
| AAAAA, incidence-aware | 73 | **74** | **+1** |
| CBDDB, geometry-free | 100 | **102** | **+2** |

These are sound upper bounds, not achievable scores. The cap-seven AAAAA
optimum is currently bracketed by the existing 68-point cap-six witness and
the new 74-point bound. The CBDDB optimum is bracketed by the existing
84-point heuristic witness and the new 102-point bound; 84 is not yet an exact
CBDDB result.

The previously published AAAAA value 68 is different in kind from the 73 in
the table: 68 is an exact cap-six holistic certificate after geometric
interference is proved, while 73 is the comparable cap-six count relaxation.
Allowing a seventh token invalidates the scope of that exact certificate. It
does not prove that the attainable maximum rises from 68 to 74.

## Count space

The number of five-species allocations summing to 20 with each count at most
seven is the coefficient of `x^20` in `(1+x+...+x^7)^5`:

```text
C(24,4) - 5*C(16,4) + 10*C(8,4) = 2,226
```

This is 1,400 more allocations than cap six, or approximately 2.695 times as
many.

## AAAAA

The direct geometry-free maximum is 75. It is attained by three count
allocations in Bear/Elk/Salmon/Hawk/Fox order:

```text
(1,4,7,1,7)
(2,3,7,1,7)
(4,1,7,1,7)
```

That relaxation assumes every fox can observe every wildlife type whose count
is nonzero. A hex has only six neighbors, so a singleton non-fox token can be
observed by at most six foxes. Applying just that incidence constraint lowers
the sound global bound to **74**, attained by the relaxation at:

```text
(2,2,7,2,7) = 4 + 5 + 25 + 5 + 35
(2,3,7,1,7) = 4 + 9 + 25 + 2 + 34
(2,4,7,1,6) = 4 + 13 + 25 + 2 + 30
```

The breakdowns are independent per-card maxima plus the Fox-A incidence
bound. They do not show that any of these three complete boards exists.

The seventh-token standalone values are Bear A 19, Elk A 22, Salmon A 25,
and Hawk A 22. The production Rust scorer now has a regression fixture
covering all four values.

## CBDDB

The geometry-free maximum is **102**, attained by:

```text
(0,3,6,4,7) = 0 + 9 + 26 + 18 + 49
(0,4,3,6,7) = 0 + 13 + 13 + 27 + 49
```

The seventh-token standalone bounds are Bear C 20, Elk B 22, and Salmon D 29.
Hawk D remains a maximum-weight matching of three pairs with one unused hawk;
Fox B remains at most seven points per fox. The Bear-C and Elk-B seventh-token
values are covered by the production Rust regression fixture. Salmon D 29 is
a deliberate relaxation: split seven salmon into components of lengths three
and four, let each component independently claim its maximum surrounding
non-salmon cells, and allow those cells to overlap between components.

CBDDB's 102 is expected to be loose because its Bear, Elk, Salmon, Hawk, and
Fox standalone maxima compete for the same cells and lines. No cap-seven
coordinate search was run in this analysis.

## Reproduction and validation

Machine-readable evidence:
`docs/v3/evidence/wildlife_cap7_upper_bounds_2026-07-23.json`.

Reproduce:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:python uv run python \
  -m tools.wildlife_cap_upper_bounds --cap 6 --cap 7
```

Validation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:python uv run pytest -q \
  tools/test_wildlife_cap_upper_bounds.py
cargo test -p cascadia-game \
  seventh_token_reference_patterns_match_a_and_cb_tables
```

The Python tests exhaustively match the existing AAAAA and CBDDB cap-six
relaxations on all 826 allocations, independently check the 2,226-vector
cap-seven count, and pin every maximizing allocation. The Rust test exercises
the production scoring implementation at seven tokens.
