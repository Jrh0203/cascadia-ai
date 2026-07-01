# ADR 0180: Distributional Authorization JSON Normalization

- Status: accepted
- Date: 2026-06-16
- Experiment: `v2-distributional-opportunity-supervision-v1`
- Supersedes execution task prefix: `v2dist-v1`
- Relaunch task prefix: `v2dist-v2`
- Does not change: data, target, graph, initialization, arm definitions,
  optimization, metrics, thresholds, selector order, or claim boundary

## Context

The ADR 0179 setup completed successfully:

- immutable source bundle `10640f4595bd...` matched on all four hosts;
- train and validation trees matched on all four hosts;
- john1 produced authorization
  `357f63571c5843c2cc590ad66b26ecaf502c4751f35937e59cc4d6b511ebaa5d`;
- that exact authorization file matched on all four hosts.

The first three available primary roles then stopped before optimizer step 1.
The unopened john3 role was cancelled before execution.

Independent authorization recomputation on john1, john2, and john4 produced
the same authorization ID, model initialization, data identities, residual
atoms, and reliability audit. The validator nevertheless rejected the
persisted authorization.

`DistributionalOpportunityProtocol.to_dict()` used `dataclasses.asdict`,
which retained `arms` as a Python tuple. JSON serialization correctly wrote
that tuple as an array. Validation compared the reloaded array to a freshly
constructed in-memory tuple with Python structural equality. The values and
canonical JSON were identical, but `list != tuple`.

## Decision

Protocol dictionaries are normalized to JSON-native values before they enter
authorization identity:

- `arms` is emitted explicitly as a list;
- the normalized dictionary is used for hashing, persistence, validation,
  run manifests, reports, and replay identity.

The validator remains a strict complete-object equality check. No scientific
field is ignored or weakened.

## Verification

The permanent regression:

1. builds an authorization from strict train and validation fixtures;
2. writes it through JSON;
3. reloads and validates it against a fresh independent recomputation; and
4. requires complete object equality.

The complete focused decoder, model, experiment, and queue suite must pass,
along with Ruff, formatting, compilation, and immutable-bundle validation.
The relaunch also inserts a host-local authorization-only preflight before
each primary role. A preflight must rebuild and accept the full authorization
without creating a run directory, optimizer, checkpoint, or training metric.

Before relaunch, the old unopened and downstream `v2dist-v1` tasks are
administratively cancelled. The three failed attempts remain preserved. No
checkpoint, training metric, or model artifact exists because all failures
occurred before run-directory creation and optimizer initialization.

## Consequences

ADR 0179 science remains frozen. A new immutable bundle and `v2dist-v2` task
graph repeat setup and run the original experiment without consuming or
repairing any old run artifact.
