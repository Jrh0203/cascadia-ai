# S3 Component-and-Motif Graph Foundation V1 Preregistration

Date: 2026-06-17

ADR: 0159

Experiment: `s3-component-motif-graph-foundation-v1`

Protocol: `s3-card-a-semantic-decoder-census-v1`

Status: frozen before production

## Question

Can exact habitat-component and Card A wildlife-motif objects reconstruct
score anatomy and selected action deltas, remain invariant under all D6
symmetries, and exercise every intended opportunity family?

## Frozen Corpus

```text
host: john3
first seed: 5,310,000
games: 14
positions: 1,120
board score checks: 4,480
action-delta checks: 1,120
D6 checks: 13,440
rayon threads: 10
```

Seed `5,300,000` was used only for implementation calibration and is excluded.

## Frozen Views

- component only;
- motif only;
- component plus motif;
- component plus motif plus frontier.

Canonical byte and token distributions are reported for each view.

## Frozen Gates

```text
(board score checks + action delta checks - failures)
  / (board score checks + action delta checks) >= 0.99
D6 failures == 0
D6 checks == positions * 12
boards with Elk extensions > 0
boards with Salmon continuations > 0
boards with Hawk opportunities > 0
boards with Bear pair opportunities > 0
```

## Predictions

1. Score and selected-action delta decoding will be exact.
2. All invariant signatures will survive all twelve D6 transforms.
3. The corpus will contain substantial coverage for all four opportunity
   families.
4. Motif-only state will be compact, while component-plus-frontier will be
   richer than raw R2 in audit bytes.

## Invalidators

- source bundle or executable mismatch;
- fewer than the frozen checks;
- transform testing without inverse/covariance semantics;
- an unobserved opportunity family;
- scientific hash mismatch; or
- learned or target-derived graph fields.

## Claim Boundary

A pass authorizes learned S3 ablations only. It does not prove R4800 or
gameplay gains.
