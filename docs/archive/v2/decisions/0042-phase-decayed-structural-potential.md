# ADR 0042: Phase-Decayed Structural Potential

Status: rejected at the selection gate on 2026-06-12. Validation and
confirmation were not opened.

## Context

The promoted pattern policy builds a deliberately diverse K8+H6+B8 frontier,
but its final ranking discards the two structural signals used to create that
frontier: newly matched habitat edges and Bear pair-ready slots. It ranks only
by exact immediate base score plus a one-turn optimistic wildlife opportunity.

Recent public beam work is closed: the full R8/B16 operator was weak online,
and two MLX architectures failed to recover a useful policy from its fixed
target. The next experiment therefore changes the planning objective itself.
It tests whether a small, interpretable amount of phase-decayed structural
credit improves full-game allocation without species-specific score weights.

## Frozen Policy Family

For each existing K8+H6+B8 candidate, compute:

- exact immediate base score;
- existing one-turn public wildlife opportunity;
- change in total matching habitat edges caused by the action;
- change in Bear pair-ready slots caused by the action;
- exact personal turns remaining after the action.

Rank by:

`immediate + a * opportunity + phase * (h * habitat_delta + b * bear_delta)`

where `phase = personal_turns_remaining / 19`.

The terminal action therefore receives no structural setup credit. Candidate
generation, market prelude, public information boundary, tie handling, and all
rules remain unchanged. The coefficients are nonnegative and contain no
species-specific reward.

The complete frozen grid is:

- `a`: 0.50, 0.75, 1.00, 1.25, 1.50;
- `h`: 0.00, 0.25, 0.50, 0.75, 1.00;
- `b`: 0.00, 0.25, 0.50, 0.75, 1.00.

This is exactly 125 policies and includes production pattern-aware at
`(1.00, 0.00, 0.00)`. Select the highest train mean, breaking exact ties by
smallest Manhattan distance from the production coefficients and then
lexicographically by `(a,h,b)`.

## Frozen Protocol

### Selection

- Rules: canonical four-player AAAAA, no habitat bonuses.
- Train seeds: `31300-31331`, 32 four-seat blocks per grid point.
- Every grid point uses the same strategy RNG domain and seed suite.
- Local CPU only.
- Advance one selected point only if its train mean exceeds the included
  production baseline by at least 0.40.

### Held-Out Validation

If selection advances, compare the selected point with production
pattern-aware on seeds `31400-31449`.

Validation requires all of:

- paired gain at least +0.30;
- paired 95% confidence-interval lower bound above zero;
- total wildlife delta at least 0.0;
- aggregate Elk+Salmon+Hawk+Fox delta at least -0.25;
- habitat delta at least 0.0;
- Nature Token delta at least -0.50;
- treatment runtime at most 1.5 seconds per block.

### Confirmation

Only a complete validation pass may open seeds `31500-31549`. Promotion
requires paired gain at least +0.25, a positive 95% interval lower bound, the
same category guardrails, and the same runtime ceiling.

No grid value, feature definition, phase function, objective, tie break, seed,
threshold, or candidate frontier may change after the selection result.
Selection data may not be reused as validation or confirmation evidence.

## Required Implementation Evidence

- exact structural deltas agree with apply-and-rescore references;
- `(1,0,0)` reproduces production rankings and complete games bit for bit;
- final-turn structural credit is exactly zero;
- every coefficient is finite, nonnegative, serialized in the strategy ID,
  and covered by validation tests;
- sweep ordering and tie breaking are deterministic;
- CLI reports every grid result and full provenance;
- strict tests and lint pass before selection.

## Commands

```bash
target/release/cascadia-v2 pattern-potential-sweep \
  --games 32 --first-seed 31300 \
  --output docs/v2/reports/pattern-potential-v1-grid125-train32.json
```

The selected coefficients, if authorized by the train gate, are then passed
unchanged to:

```bash
target/release/cascadia-v2 pattern-potential-compare \
  --games 50 --first-seed 31400 --opportunity-weight A \
  --habitat-weight H --bear-weight B \
  --output docs/v2/reports/pattern-potential-v1-validation50.json
```

## Result

The complete 125-policy sweep evaluated 4,000 canonical four-seat blocks in
46.13 seconds. The included production tuple `(1.00, 0.00, 0.00)` scored
91.992. The deterministic winner was:

- opportunity weight: 1.00;
- habitat weight: 0.00;
- Bear-ready weight: 0.75;
- mean: 92.117;
- gain over production: +0.125.

The +0.125 gain missed the frozen +0.40 selection gate by 0.275. No validation
or confirmation seed was opened.

The sweep does confirm that Bear readiness changes the allocation: the winner
raised Bear from 7.766 to 8.789 on the train suite, but lost 0.227 Elk, 0.367
Salmon, 0.289 Hawk, 0.227 Fox, and 0.375 habitat in aggregate, leaving only a
small total gain. The best point containing habitat credit ranked fifth at
91.961. Simple additive structural credit therefore repeats the known
allocation tradeoff and is closed.
