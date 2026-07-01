# ADR 0156 Parent-Latency Measurement Amendment

Date: 2026-06-17

Experiment: `r4-bounded-quotient-mlx-comparison-v1`

Protocol: `r4-bounded-parent-mlx-matched-comparison-v1`

## Problem

The pre-production benchmark implementation initially measured parent-encode
P50 by repeatedly encoding validation row 0. Row 0 is an unusually small
early-game exact-R2 state. The bounded quotients have a larger fixed summary
floor but fewer parent tokens on average over the registered corpus, so a
row-0-only latency gate would answer a different question and systematically
bias the comparison.

This defect was found during a one-step bounded smoke. No production arm,
authorization, or classifier result existed.

## Corrected Measurement

The promotion field `parent_encode.latency_milliseconds` is the distribution
of steady parent-only encodes over every registered validation decision:

- all 240 validation rows in production;
- the same row order on treatment and host-paired C0;
- cache-to-MLX materialization occurs before the timer;
- model execution and `mx.eval` are inside the timer;
- measurement occurs after the complete-decision pass has exercised the
  decision shapes;
- exactly one measured parent encode per validation row.

The repeated compiled row-0 timing remains in
`fixed_parent_encode.latency_milliseconds` as a diagnostic and is not used by
the classifier.

The material-efficiency gate remains unchanged:

```text
parent_encode P50 <= 0.80 * same-host C0 parent_encode P50
```

## Claim Boundary

This amendment corrects the registered measurement population. It does not
change any arm, model, data, optimizer, objective, threshold, tie-break, or
claim boundary.
