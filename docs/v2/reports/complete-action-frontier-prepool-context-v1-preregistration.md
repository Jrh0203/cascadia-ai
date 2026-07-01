# Complete-Action Frontier Pre-Pool Context V1 Preregistration

Status: **frozen before real execution**

Date: 2026-06-16

Experiment ID: `complete-action-frontier-prepool-context-v1`

## Decision

Export the selected ADR 0089 model's exact 192-dimensional pre-pool candidate
vector and compare four distinct linear-memory context mechanisms:

1. candidate only on john1;
2. the exact existing mean/max context on john2;
3. richer global moments on john3; and
4. observable screen-top64 context on john4.

All arms train concurrently with one frozen seed. Ring cross-replay begins
only after the relevant independent artifacts exist. The experiment asks
where candidate target signal is lost; it does not test gameplay strength.

## Frozen Gates

- Train fit: at least 80% target recall and 25% exact target sets.
- Validation transfer: at least 50% target recall and 1% exact target sets.
- Complete finite scoring of all 560 train and 240 validation groups.
- Peak RSS at most 4 GiB and zero process swaps.
- Bit-identical maximum-width reconstruction on john1 and john4.
- Bit-identical four-host source and portable cache identities.
- Bit-identical ring cross-replay scientific payloads.
- Sealed test, gameplay, new teacher compute, cloud, and external compute
  remain closed.

## Throughput Contract

All hosts locally regenerate the compact pre-pool cache because prior measured
local extraction is materially faster than transferring the equivalent
multi-gigabyte payload. Each host then trains a different mechanism. No node
runs duplicate training, and no host waits for another trainer unless its
only remaining authorized work is artifact-dependent cross-replay.

## Stop Rule

Run exactly 20 epochs per arm. Do not add a seed, widen a head, alter context
features, or continue the best arm after seeing metrics. Classify by ADR 0096
only after all four reports and ring replays pass integrity checks.
