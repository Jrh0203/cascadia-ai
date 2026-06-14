# ADR 0076: Rank-Stratified Counterfactual Contrast Audit

Status: rejected on R8 exact top-action agreement on 2026-06-13. No train
corpus, model, test, or gameplay domain was opened.

## Context

ADR 0075 qualified the decision-local sampling method. R8 reproduced R16
centered advantages with 0.274-point MAE, 0.855 correlation, 89.58% pairwise
accuracy, 81.25% exact winner agreement, and 0.057 mean top-action regret.
Collection projected to 7.72 local hours for 128 train plus 32 validation
games.

The target was rejected because the H6-selected action and its three nearest
ranked alternatives spanned only 1.367 points on average, below the frozen
1.50-point width gate. More samples cannot create contrast absent from the
candidate group. The next isolated question is whether rank-stratified legal
alternatives expose enough decision signal while retaining the same search
teacher, continuation policy, public information boundary, and estimator.

## Decision

Reuse ADR 0075's grouped raw-return format and qualified sampler, but change
only candidate retention after the public market prelude has resolved:

1. Run frozen H6 K8/H6/R4/D4 and retain its selected action first.
2. Remove that action from H6's existing ranked frontier.
3. Retain the highest-ranked remaining action.
4. Retain the median remaining action at index
   `floor((remaining_count - 1) / 2)`.
5. Retain the lowest-ranked remaining action.

The H6 frontier historically contains at least eight distinct actions, and the
collector must still reject any group that cannot supply four unique legal
actions. No action is regenerated, rescored, widened beyond H6's existing
frontier, or selected using terminal audit returns.

All four candidates use the identical ordered public-redetermination seeds.
Every terminal decomposed score, shared seed, public supply, parent public
state, observable action afterstate, action hash, exact immediate score, H6
shallow mean, and H6 shallow uncertainty remains stored.

## Frozen Protocol

- Implementation-only smoke: train split index 9,995, one source game, groups
  at completed turns `0,20,40,60`, four candidates, R2.
- Substantive audit: validation split indices 67,000-67,001, two complete H6
  source games, groups at completed turns `0,5,...,75`, four candidates, R16.
- Rules: symmetric four-player AAAAA, no habitat bonuses.
- Source and continuation policy: frozen H6 K8/H6/R4/D4.
- Decision boundary: canonical public post-prelude state.
- Candidate ordering: selected, highest remaining, median remaining, lowest
  remaining.
- Shared seed domain:
  `cascadia-v2-counterfactual-advantage-v1`, split, game index, completed turn,
  and sample index.
- Local Apple M4 only; output must be scheduling-independent.
- No train corpus beyond the implementation smoke, model training, test/final
  access, gameplay, alternate quantiles, action-count change, threshold
  change, retry, or external compute.

The contrast target advances only if all conditions hold:

- all schema, checksum, provenance, replay, sequence, action-identity,
  public-supply, shared-seed, and determinism checks pass;
- R8 centered-advantage MAE against R16 is at most 0.50 points;
- R8 centered-advantage correlation against R16 is at least 0.80;
- R8 within-group pairwise accuracy against R16 is at least 80%;
- R8 exact top-action agreement with R16 is at least 65%;
- mean R16 regret from choosing the R8 winner is at most 0.50 points;
- mean R16 within-group value range is at least 1.50 points;
- mean R16 centered-advantage standard error is at most 0.75 points;
- projected 160-game R8 collection is at most 12 uncontended local hours.

Passing authorizes only a separately preregistered 128-game train and 32-game
validation R8 corpus plus an MLX complete-candidate-set ranker. Test collection
and gameplay remain closed. Any failed gate rejects this selection policy
without retry or quantile adjustment.

## Required Implementation

- an explicit versioned candidate-selection identifier in the dataset teacher
  contract while preserving ADR 0075 artifact readability;
- typed CLI selection rather than an environment variable or free-form string;
- exact selected/high/median/low retention tests;
- smoke collection, repeat determinism, strict Clippy, and focused tests;
- atomic substantive collection, validation, JSON audit, and Markdown report;
- final ADR, registry, status, roadmap, and score-gap updates.

## Implementation Evidence

The implementation adds the typed CLI modes `nearest` and `stratified`, stores
the explicit `selected-high-median-low-v1` identifier for ADR 0076 datasets,
and keeps ADR 0075's absent-field teacher serialization and dataset identifier
byte-for-byte unchanged. The new retention helper removes the selected action,
deduplicates the remaining ranked frontier, and retains remaining indices
`0`, `floor((n - 1) / 2)`, and `n - 1`.

Focused data and CLI suites passed eight tests, including exact legacy teacher
serialization, exact legacy and stratified dataset identifiers, grouped record
round trips, duplicate-action rejection, and exact nearest and stratified rank
retention. Strict no-dependency Clippy passed for `cascadia-data` and
`cascadia-cli-v2`; formatting and whitespace checks passed. The rebuilt release
binary also validated ADR 0075's sealed two-game, 32-group, 2,048-continuation
dataset without changing its schema or header interpretation.

The one allowed implementation smoke collected four groups and 32
continuations in 12.739 seconds. Its shard hash is
`6382c516157f0b7fa68459e8d4a2ca370d5850cb6a29f1d5ec50041623fb29b9`.
An independent repeat produced the identical hash and compared byte-for-byte.
The smoke report is
`docs/v2/reports/rank-stratified-counterfactual-contrast-audit-v1-implementation-smoke.md`.
Its R2 mean group range of 4.750 points is implementation evidence only and
does not alter, preview, or satisfy any frozen R16 promotion gate.

## Result

The frozen validation collection completed two games, 32 decision groups, 128
candidates, and 2,048 shared-seed terminal continuations in 639.330 seconds.
Both atomic shards passed schema, header, checksum, sequence, action-identity,
public-supply, and raw-return validation.

Rank stratification solved ADR 0075's target-width failure. Mean R16
within-group range increased from 1.367 to 2.803 points, while mean centered
advantage standard error remained 0.446. R8 also passed centered MAE at 0.353,
centered correlation at 0.931, pairwise accuracy at 85.42%, mean winner regret
at 0.145 points, and projected 160-game local collection cost at 7.10 hours.

The sole failed gate was exact top-action agreement: R8 selected the same
winner as R16 in 20 of 32 groups, or 62.50%, below the frozen 65% threshold by
one group. The result is rejected without retry, threshold adjustment, or
post-hoc promotion. The raw evidence supports a separately preregistered fresh
audit of a larger shared-seed prefix, but it does not authorize corpus
collection, MLX training, test access, or gameplay.

## Maximum Compute

One one-game R2 implementation smoke and one two-game R16 substantive audit.
No retry, sweep, extra game, quantile change, threshold change, candidate-count
change, policy change, test access, gameplay, model training, or external
compute.
