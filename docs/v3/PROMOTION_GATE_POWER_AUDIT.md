# V3 Promotion Gate Power Audit

Date: 2026-06-24

## Finding

Cycle 1 opened under `cascadia-v3-always-valid-promotion-v1`. That rule is a
valid bounded mixture-betting e-process, but it normalizes every paired score
delta by the full legal `[-200, 200]` range. Its registered maximum is 500
pairs per tier and its evidence threshold is 20.

A deterministic power audit proves that the v1 rule cannot recognize the
effects the campaign was designed to measure:

- 500 zero-variance observations at the alternative `+0.15` produce an
  alternative-boundary e-value of only `1.3883`, not 20.
- Even 500 identical `+1.0` deltas produce only `5.2677`.
- With no sampling variance at all, v1 requires a constant delta of at least
  `+1.7200115` to promote and at most `-1.6704663` to reject.
- The original tests used constant deltas of `+20` and `-20`, so they checked
  code reachability but not power at either registered hypothesis.

This is not a worker, seed, search, or model failure. It is a mismatch between
the raw theoretical score bound, the 500-pair budget, and the effect sizes in
the campaign contract.

## Repair boundary

Cycle 1 remains immutable. Its process had already opened the 0–99 and 100–199
domains before this audit, so it continues under the exact v1 rule even after a
controller restart. `tools/v3_promotion_v1.py` preserves that implementation.
No Cycle-1 result is reinterpreted under the repair.

Before any Cycle 2 pair domain opens, cycles 2–10 are pre-registered on
`cascadia-v3-always-valid-promotion-v2`:

- Raw paired deltas remain validated over `[-200, 200]` and are always
  reported.
- The promotion estimand deterministically winsorizes each paired delta to
  `[-25, 25]` before betting.
- Null `-0.10`, alternative `+0.15`, alpha `0.05`, beta `0.05`, the 100-pair
  looks, the 500-pair maximum, four-tier unanimity, and integrity/resource
  gates are unchanged.
- The cap is a model-selection utility bound, not a claim about the legal raw
  score range: a single pair may move the decision by up to one quarter of the
  100-point campaign goal but cannot dominate hundreds of consistent pairs.
- Final protected comparison and the 100-point claim continue to use raw,
  unclipped scores and raw confidence intervals.

The v2 constant-sequence checks now match the registered hypotheses: 500
identical `+0.15` observations cross promotion and 500 identical `-0.10`
observations cross retention. Outlier tests prove that raw reporting and the
decision estimand remain distinct.

## Statistical basis

The repair retains a bounded nonnegative betting process and Ville-style
optional-stopping control. Variance-adaptive and hedged capital processes could
be tighter, but changing the estimator family mid-campaign would add machinery
without resolving the primary scale error. The bounded robust estimand is the
smallest auditable correction.

Relevant primary references:

- Ian Waudby-Smith and Aaditya Ramdas, “Estimating means of bounded random
  variables by betting,” *JRSS B* 86(1), 2024,
  <https://doi.org/10.1093/jrsssb/qkad009>.
- Steven R. Howard, Aaditya Ramdas, Jon McAuliffe, and Jasjeet Sekhon,
  “Time-uniform, nonparametric, nonasymptotic confidence sequences,” *Annals of
  Statistics* 49(2), 2021, <https://doi.org/10.1214/20-AOS1991>.

## Verification

- `tools/test_v3_promotion.py` covers realistic hypothesis crossings, raw vs.
  decision means, resource rejection, and exact v1 reproduction.
- `python/tests/test_v3_cycle_promotion.py` proves Cycle 1 always selects v1
  and cycles 2–10 select v2.
- The worker artifact contract is unchanged; the repair is controller-side and
  does not alter games, legal moves, RNG domains, search, or Docker identity.
