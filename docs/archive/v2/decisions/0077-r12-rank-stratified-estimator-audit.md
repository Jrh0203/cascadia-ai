# ADR 0077: R12 Rank-Stratified Estimator Audit

Status: accepted on fresh validation on 2026-06-13. R12 corpus collection and
an MLX ranker experiment are authorized; test and gameplay remain closed.

## Context

ADR 0076 established that rank-stratified H6 candidate groups have adequate
learning signal. Selected/high/median/low groups spanned 2.803 points at R16,
well above the frozen 1.50-point width gate, while mean centered-advantage
standard error was 0.446.

R8 passed centered MAE, correlation, pairwise accuracy, winner regret,
integrity, uncertainty, and cost gates, but exact winner agreement was 20 of
32 groups, or 62.50%, against 65% required. The miss was one decision and mean
R8 winner regret was only 0.145 points. Changing the target, candidate set,
threshold, or teacher would confound the remaining question. The next
isolated variable is the number of shared public redeterminations.

At ADR 0076's measured throughput, a 160-game R12 corpus projects to 10.66
uncontended local hours, inside the unchanged twelve-hour cost ceiling.

## Decision

Audit R12 against R16 on two fresh validation games while freezing every other
ADR 0076 choice:

- symmetric four-player AAAAA with no habitat bonuses;
- source and continuation policy H6 K8/H6/R4/D4;
- canonical public post-prelude decision state;
- selected action plus highest, median, and lowest remaining ranked H6
  alternatives;
- four candidates per group and sixteen groups per game;
- identical ordered public-redetermination seeds for all four candidates;
- raw decomposed terminal returns, public supply, parent public state, action
  afterstates and hashes, exact immediate scores, H6 means, and uncertainty;
- local Apple M4 execution only.

ADR 0076's validation groups must not be used for promotion or to preview R12.
The fresh substantive domain is validation split indices 68,000-68,001.
Collection remains R16 so R12 can be evaluated against the same complete
reference without changing target semantics.

## Frozen Gates

R12 qualifies only if every condition holds:

- all schema, checksum, provenance, replay, sequence, action-identity,
  public-supply, shared-seed, and determinism checks pass;
- R12 centered-advantage MAE against R16 is at most 0.50 points;
- R12 centered-advantage correlation against R16 is at least 0.80;
- R12 within-group pairwise accuracy against R16 is at least 80%;
- R12 exact top-action agreement with R16 is at least 65%;
- R12 exact top-action agreement is strictly greater than R8 agreement on the
  same fresh groups;
- mean R16 regret from choosing the R12 winner is at most 0.50 points;
- R12 mean winner regret is no greater than R8 mean winner regret on the same
  fresh groups;
- mean R16 within-group value range is at least 1.50 points;
- mean R16 centered-advantage standard error is at most 0.75 points;
- projected 160-game R12 collection is at most twelve uncontended local hours.

Passing authorizes only a separately preregistered 128-game train and 32-game
validation R12 corpus plus an MLX complete-candidate-set ranker. Test
collection and gameplay remain closed. Any failed gate rejects R12 without
retry, sample-count sweep, threshold change, or reuse of ADR 0076 groups.

## Required Implementation

- make estimator sample count an explicit typed audit argument with R8 as the
  backward-compatible default;
- compute and publish R12 alongside the existing prefixes without changing
  sealed ADR 0075 or ADR 0076 artifacts;
- make gate names, Markdown labels, and projected collection cost reflect the
  selected estimator count;
- add focused tests for R12 prefix selection and invalid estimator counts;
- pass strict Clippy, focused tests, formatting, dataset validation, and
  registry validation before collection;
- collect exactly two fresh R16 games, validate both atomic shards, publish
  JSON and Markdown, and update the ADR, registry, status, roadmap, and score
  gap.

## Implementation Evidence

The audit command now accepts typed `--estimator-samples` values R8 and R12,
with R8 remaining the default. Reports include R12 in the fixed prefix set,
derive gate names and projected corpus cost from the selected estimator, and
apply ADR 0077's exact-winner and regret dominance checks against R8 on the
same groups. Invalid sample counts and R12 requests against an R8 substantive
dataset are rejected.

Six focused CLI tests passed, including the R8/R12 contract, invalid count
rejection, exact prefix enumeration, deterministic shared seeds, and both
candidate-retention policies. Strict no-dependency Clippy, formatting, and
whitespace checks passed. The release binary validated both sealed ADR 0075
and ADR 0076 datasets without auditing or exposing their R12 prefixes. No
implementation dataset was collected, as frozen.

The first attempt to collect validation game 68,000 stopped before creating a
manifest or shard with `the wildlife bag is unexpectedly empty`. Therefore no
ADR 0077 return, prefix, metric, or partial statistical artifact existed. The
failure exposed an unhandled game-engine transition during a complete
continuation, not an experiment result. Before reproducing the same frozen
index, the collector was amended only to attach source turn, candidate,
sample, continuation turn, and operation context to transition errors. This
diagnostic amendment changes no game rule, seed, candidate, policy, return, or
gate.

The contextual reproduction failed at source turn 60, candidate 3, sample 8,
continuation turn 77 while H6 previewed its canonical free three-of-a-kind
replacement. The valid prelude entered a repeated automatic four-of-a-kind
chain that set aside enough tokens to leave no legal refill. ADR 0018 already
defines such a branch as having no legal stabilized market rather than an
invented outcome. Because the initiating three-of-a-kind replacement is
optional, the engine now exposes a transactional helper that uses the free
replacement only when its complete preview stabilizes; `WildlifeBagEmpty`
declines the optional replacement and preserves the original legal market,
while every other rule error still propagates.

The fix is shared by deterministic lookahead, Bear and habitat candidate
unions, and Nature-wipe lookahead. A direct conservation regression constructs
the exact three-Bear/one-Elk market with only three Elk drawable, proves the
requested replacement fails, then proves the canonical helper declines it
without mutating the state. This is a pre-data completion of an already
documented finite-bag rule edge, not a change to ADR 0077's target, estimator,
candidate selection, seed suite, or gates.

The first corrected reproduction reached the same coordinate before failing
inside H6's internal greedy rollout rather than its root prelude. The rollout
policy still constructed the free-replacement flag directly and bypassed the
new feasibility helper. No artifact was written. The canonical simulation
strategy and `play_greedy_plies` now use the same transactional helper, so
root selection, determinized lookahead, and internal rollout plies share one
finite-bag policy instead of diverging at this edge.

## Result

The corrected frozen collection completed validation games 68,000-68,001,
32 decision groups, 128 candidates, and 2,048 shared-seed terminal
continuations in 620.079 seconds. Both atomic shards passed schema, header,
checksum, sequence, action-identity, public-supply, and raw-return validation.

R12 passed every preregistered gate:

- centered MAE to R16: 0.204;
- centered correlation: 0.968;
- pairwise accuracy: 92.19%;
- exact winner agreement: 25 of 32 groups, or 78.13%;
- mean winner regret: 0.037 points;
- mean R16 group range: 2.469 points;
- mean centered-advantage standard error: 0.429 points;
- projected 160-game R12 collection: 10.33 local hours.

On the same fresh groups, R8 reached only 18 of 32 exact winners, or 56.25%,
with 0.283 mean regret. R12 therefore passed both direct dominance gates as
well as the absolute fidelity, width, uncertainty, integrity, and cost gates.

This result qualifies the selected/high/median/low R12 target and authorizes a
separately preregistered 128-game train plus 32-game validation corpus and MLX
complete-candidate-set ranker. It does not authorize sealed test collection,
promotion, or gameplay comparison.

## Maximum Compute

One two-game R16 validation collection at indices 68,000-68,001 and one R12
audit. No implementation data collection, retry, sweep, extra game,
sample-count change, candidate change, teacher change, threshold change, test
access, gameplay, model training, or external compute.
