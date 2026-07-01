# ADR 0181: Distributional Work-Conserving Replay Dependencies

- Status: accepted
- Date: 2026-06-16
- Experiment: `v2-distributional-opportunity-supervision-v1`
- Scope: scheduler dependencies only
- Does not change: data, targets, graph, initialization, optimizer, training
  order, fixed final checkpoint, validation, metrics, thresholds, replay host,
  classifier, or claim boundary

## Context

The corrected ADR 0179 primary wave started on every free host. Q2 remained
correctly assigned to john3, which was still occupied by the independent P1
wildlife-pointer stage.

The initial queue made every replay depend on all four primary roles. That
global barrier was unnecessary. A replay consumes only:

1. the common immutable bundle, data, and authorization;
2. its replay host's successful authorization preflight; and
3. the completed primary report and model for the same arm.

It does not read another arm's model, report, metric, or validation result.
Training is fixed at step 3,000 with no early stopping or validation-based
selection. Waiting for unrelated primaries therefore cannot protect a
scientific information boundary; it only leaves eligible hosts idle.

## Decision

Each replay depends on exactly:

- the matching primary arm; and
- the replay host's authorization-only preflight.

The final collection retains its barrier over all four primaries and all four
replays. Classification remains impossible until all eight reports and models
are checksum-bound locally.

The live queue is updated through the queue's audited
`set-dependencies` operation. A regenerated work-conserving specification is
stored beside the original launch graph.

## Verification

The queue regression requires every replay to name its corresponding primary
and its own host preflight, while the collection still names all eight run
tasks. Queue validation must pass after the live dependency mutation.

## Consequences

john1 and john2 can execute E3 and C0 rotated replays while john3 continues P1
and john4 remains available for independent work. The scientific result is
unchanged because replay outputs are deterministic functions of frozen inputs,
and terminal selection still waits for the complete matched factorial.
