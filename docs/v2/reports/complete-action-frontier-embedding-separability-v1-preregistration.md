# Complete-Action Frontier Embedding Separability V1 Preregistration

Status: **active preregistered**

Date: 2026-06-16

Experiment ID: `complete-action-frontier-embedding-separability-v1`

The authoritative protocol is
`docs/v2/decisions/0094-frontier-frozen-embedding-separability.md`.

The exact ADR 0089 candidate vectors immediately before the learned heads will
be cached in float32 for every open action. john2 owns train extraction; john3
owns validation extraction. They exchange caches once and train independent
linear and one-hidden-layer probes. john4 cross-replays both saved probes,
while john1 aggregates the frozen classification.

Before extraction, john1 and john4 independently require bit-identical
reconstruction of both original heads from the exported embedding on the
10,854-action maximum-width open decision, with finite outputs, RSS below
4 GiB, and no swap growth.

The linear gate is 60% train target recall and 5% exact train sets. If it
fails, the nonlinear probe must reach 80% train recall and 25% exact train
sets; 50% validation recall and 1% exact validation sets distinguish useful
head capacity from train-only separability. Failure of nonlinear train fit
authorizes a representation change.

No full-network training, sweep, duplicate cache inference, additional seed,
teacher compute, sealed test, gameplay, cloud, or external compute is
authorized.
