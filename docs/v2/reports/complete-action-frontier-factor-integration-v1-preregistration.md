# Complete-Action Frontier Factor Integration V1 Preregistration

Status: **frozen before implementation and real execution**

Date: 2026-06-16

Experiment ID: `complete-action-frontier-factor-integration-v1`

## Decision

Export the seven exact 192-dimensional inputs to `candidate_projection` and
compare four distinct integration architectures:

1. wide concatenation on john1;
2. screen-relative factor context on john2;
3. factor-token attention on john3; and
4. pairwise-gated factors on john4.

This is a four-hypothesis architecture fork, not a four-seed training run.
The heavier screen-relative treatment is assigned to the fastest host
measured in ADR 0096; john1 receives the simplest arm.

## Frozen Gates

- Train fit: at least 80% target recall and 25% exact target sets.
- Validation transfer: at least 50% target recall and 1% exact target sets.
- Complete finite scoring of all 560 train and 240 validation groups.
- Peak process RSS at most 6 GiB and zero process swaps.
- Bit-identical maximum-width reconstruction on john1 and john4.
- Bit-identical four-host source and portable factor-cache identities.
- Bit-identical ring cross-replay scientific payloads.
- At least 24 GiB free per host before factor-cache generation.
- Sealed test, gameplay, new teacher compute, cloud, and external compute
  remain closed.

## Throughput Contract

Each host regenerates its approximately 15.0 GiB factor cache locally rather
than relaying it. Each host then trains a different mechanism. Duplicate
training remains prohibited. A completed host immediately performs any
unblocked incoming ring replay; otherwise its idle interval is recorded as
dependency-blocked.

## Stop Rule

Run exactly 20 epochs per arm. Do not add a seed, resize a network, change the
objective, or continue the apparent leader after seeing metrics. Classify
only after all four reports and ring replays pass integrity checks.
