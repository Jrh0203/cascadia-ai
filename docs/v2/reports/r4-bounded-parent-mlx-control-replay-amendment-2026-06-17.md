# ADR 0156 Host-Paired Control Replay Amendment

Date: 2026-06-17

Experiment: `r4-bounded-quotient-mlx-comparison-v1`

Protocol: `r4-bounded-parent-mlx-matched-comparison-v1`

## Problem

After all four production arms completed, the first host-paired C0 replay on
each treatment machine was rejected before a replay report or classification
could be emitted. The replay command omitted the explicit validation-row set,
so the benchmark library used its 20-decision diagnostic default instead of
the registered 240-decision production population. The replay verifier
correctly rejected the resulting 20-decision, 27,958-action measurements.

The same call also wrote its request and result diagnostics into the copied C0
run directory. That did not alter checkpoint, optimizer, metric, latest, or
final-report bytes, but it replaced the two serving-diagnostic files copied
from john1. A serving replay must not mutate the checksum-fanned control run.

No failed attempt produced an accepted paired-control report. No classifier
was run, and no result or promotion decision used the rejected measurements.

## Correction

The host-paired replay now:

1. verifies that the signed C0 report certifies exactly 240 validation groups
   and 860,203 actions, each scored once;
2. derives the ordered row set `0..239` from that certified population;
3. passes those rows explicitly to the isolated serving benchmark;
4. retains the existing post-measurement coverage verifier; and
5. writes request and result diagnostics under a dedicated paired-control
   artifact directory outside the copied C0 run.

Before retrying, the exact C0 run is checksum-fanned to john2, john3, and john4
again so every replay begins from the original immutable tree.

## Frozen Scope

This is an operational correction to the serving-only control harness. It does
not change:

- any production model, checkpoint, parameter, prediction, or quality metric;
- any arm, representation, token schema, candidate stream, data, label, seed,
  optimizer, schedule, objective, or host assignment;
- any serving warmup, steady count, candidate chunk, validation population,
  threshold, ratio, tie-break, or classifier rule; or
- the claim boundary.

The corrected tool is distributed in a new content-addressed patch bundle.
The original production bundle and all four completed production reports
remain immutable evidence.
