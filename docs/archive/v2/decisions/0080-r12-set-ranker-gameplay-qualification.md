# ADR 0080: R12 Set-Ranker Gameplay Qualification

Status: closed unopened on 2026-06-14. ADR 0078 failed validation, so ADR 0079
never opened and no grouped inference service, parity fixture, gameplay seed,
or promotion path was created.

## Context

ADR 0078 trains one fresh MLX ranker to choose among the H6-selected action
and the high, median, and low alternatives from the same complete H6 frontier.
ADR 0079 conditionally tests that checkpoint on a sealed 32-game R12 corpus.

An offline pass does not prove playing strength. Repeated model selection can
also turn gameplay into another validation set. This ADR therefore fixes the
only authorized inference boundary, gameplay seeds, stages, thresholds, and
maximum compute before either offline result is available.

If ADR 0078 or ADR 0079 fails, this ADR closes unopened. No service,
gameplay record, alternate checkpoint, threshold change, or post-hoc pilot is
authorized.

## Frozen Policy

At every canonical public post-prelude decision:

1. Run unchanged H6 K8/H6/R4/D4 ranking and deterministic selection.
2. Remove prelude fields exactly as in ADR 0078.
3. Retain the H6-selected action first, then the highest, median, and lowest
   distinct remaining ranked actions.
4. Encode the same observable action afterstates, explicit action features,
   exact immediate scores, candidate mask, and 30-value public supply used by
   the training decoder.
5. Score all four candidates jointly with the byte-identical ADR 0078 best
   checkpoint and select the first maximum.
6. Reattach the original H6 prelude and require the complete action to pass
   canonical legality before application.

The policy has no hidden bag access, native neural fallback, score tolerance,
candidate repair, calibration, blend, ensemble, warm start, or additional
search. A service error, non-finite score, count mismatch, identity mismatch,
or illegal selection fails the run.

## Frozen Inference Qualification

Implementation is authorized only after a complete ADR 0079 pass.

- Add one grouped Rust/MLX framed request for exactly four candidates.
- Load only the frozen run's `best.json` checkpoint.
- Compare service output with direct Python output on 128 fixed validation
  groups.
- Require maximum absolute score error at most `1e-5`, identical selected
  actions, identical candidate/action hashes, exact public-supply bytes,
  correct request accounting, zero fallback, and clean shutdown.
- Measure warmed batch-32 throughput and require at least 5,000 candidate
  scores per second with P99 service latency at most 5 ms.

Failure rejects integration without gameplay.

## Frozen Gameplay Protocol

Rules are symmetric four-player AAAAA with habitat bonuses excluded. Baseline
is unchanged H6 K8/H6/R4/D4. Treatment is the frozen grouped ranker policy.
Both use identical canonical numeric game seeds and complete pre-move policy.

### Runtime Smoke

- Seed: 36,099.
- Games: one paired game.
- Require 80 legal treatment selections, zero fallback or service error,
  treatment runtime at most 7.0 seconds per game, and clean shutdown.

### Development Screen

Only after the smoke passes:

- Seeds: 36,100-36,119.
- Games: 20 paired games.
- Require treatment-minus-H6 mean at least `+0.25`.
- Require total wildlife at least `-0.25`, habitat at least `-0.25`, Nature
  Tokens at least `-0.50`, and aggregate Elk+Salmon+Hawk+Fox at least `-0.50`.
- Require treatment runtime at most 7.0 seconds per game, zero fallback, and
  complete replay/integrity evidence.

Failure closes the policy. A passing screen is promising, not confirmation.

### Confirmation

Only after the screen passes:

- Seeds: 36,200-36,299.
- Games: 100 paired games.
- Require treatment mean at least 92.0.
- Require paired mean at least `+0.20` and paired 95% confidence lower bound
  above zero.
- Repeat every score-balance, runtime, legality, fallback, replay, checkpoint,
  request-accounting, and shutdown gate from the screen.

Report game-block and seat-score standard deviations, standard error, 95%
confidence intervals, P10/P50/P90, every scoring category, decision latency,
wall time, host, source revision, executable and checkpoint checksums, and all
paired game records.

## Consequences

A confirmation pass qualifies one fresh MLX-backed v2 research policy. It does
not automatically replace the faster product strategy or establish the
100-point claim. Final policy selection must use only preregistered
confirmation evidence; final indices `0-999` remain sealed until that choice
is complete.

A failed smoke, screen, or confirmation rejects the policy permanently. No
retry, new seed, coefficient adjustment, candidate change, model change, or
partial promotion is authorized.

The prerequisite offline validation failed before inference implementation.
All 121 reserved gameplay games, the 128-group parity qualification, and the
final-domain policy-selection path remained unopened.

## Maximum Compute

No inference or gameplay compute was consumed. The conditional parity,
runtime-smoke, 20-game screen, and 100-game confirmation are permanently
unauthorized after ADR 0078's failure. No external compute, extra training,
test evaluation, gameplay sweep, or ranker access to the final suite is
authorized.
