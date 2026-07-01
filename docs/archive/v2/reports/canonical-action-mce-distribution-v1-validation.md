# Canonical Action MCE Distribution v1 Validation

Status: rejected on validation on 2026-06-12. No test, promotion, or gameplay
domain was opened.

## Corpus

- Train: 64 games, 5,120 groups, 491,520 actions.
- Validation: 16 games, 1,280 groups, 122,880 actions.
- Train teacher estimates: 153,021/153,021 aligned.
- Validation teacher estimates: 38,641/38,641 aligned.
- Teacher: deterministic K32/R600/LMR historical heuristic under canonical V2
  execution.
- Train manifest BLAKE3:
  `2ecaf1011da7eaaba8e648785955d12fda188de715d7f274b55c5ad7c84b07d8`.
- Validation manifest BLAKE3:
  `8e69ff94272d03135c30122c6c8bf27046124d6a2aaa285d7ff87f6f827f122e`.

Collection took 9,426.0 seconds for train and 2,346.9 seconds for validation.
Every one-game source and target shard passed Rust validation, and the Python
loader independently streamed all 6,400 paired groups.

## Training

- Device: `Device(gpu, 0)`.
- Architecture: `shared-state-action-imitation-v1`, hidden 96, four heads,
  two board blocks, one market block.
- Optimizer: AdamW, learning rate `1e-4`, weight decay `1e-4`.
- Batch: 16 complete groups.
- Seed: 20260616.
- Stop: epoch 11 after five non-improving epochs.
- Training runtime: 74.317 seconds.
- Best checkpoint: `step-000001920-epoch-0006-batch-000000`.
- Best checkpoint manifest BLAKE3:
  `bf7806849623e0e8f3ec6947e7d4c8ff9d51bfaf96eacc302f9ae6e5f75a6517`.

The best checkpoint was reloaded through the checksummed checkpoint path and
reproduced both train and validation metrics. A completed-run resume performed
zero optimizer steps. During that audit, two scale defects were fixed:

- paired shard readers no longer retain one NumPy memmap descriptor per shard;
- completed ranking resumes preserve the authoritative final report and
  cumulative runtime instead of replacing it with a no-op session.

Focused regression tests cover 140 paired shards, exact checkpoint cursor
round trips, and completed-run report preservation.

## Frozen Gates

| Gate | Required | Result | Pass |
|---|---:|---:|:---:|
| Teacher alignment | 100% | 100% train and validation | yes |
| Loss vs initialization | `< 1.832475` | `1.534834` | yes |
| Selected top one | `>= 0.23` | `0.137500` | no |
| Selected top five | `>= 0.58` | `0.384375` | no |
| Selected MRR | `>= 0.40` | `0.269223` | no |
| Predicted teacher coverage | `>= 0.90` | `0.714063` | no |
| Scored pairwise accuracy | `>= 0.75` | `0.679754` | no |
| Value-difference correlation | `>= 0.35` | `0.444333` | yes |

Additional best-checkpoint diagnostics:

- conditional mean regret: 1.139497;
- scored top-one value recall: 0.216406;
- scored rank correlation: 0.470476;
- pairwise log loss: 0.684818;
- pairwise Brier score: 0.059821.

## Diagnosis

The best model improved value geometry but did not identify the selected action
reliably. Train top-one was only 17.46%, so the failure is not mainly
validation overfit.

The teacher's exact winner is usually a near-tie: 94.53% of validation groups
have a top-two rollout-mean margin at most one point, and 95.78% have a margin
within one combined standard-error scale. Even so, the immediate-score
baseline reached 20.63% top-one and 0.2923 MRR, both above the learned model's
13.75% and 0.2692. A diagnostic reciprocal-immediate-rank blend reached
22.81% top-one but only 49.06% top-five and 0.3587 MRR, still below the frozen
gates.

The single unanchored scalar therefore forgets useful exact immediate score
while only partially learning continuation value. It also selects outside the
teacher-scored K32 frontier in 28.59% of groups. The next experiment must make
the point-scale decomposition explicit instead of changing attention shape or
retuning a post-hoc coefficient.

## Conclusion

ADR 0053 is rejected. Distributional evidence is useful and the corpus is
retained, but the unanchored pairwise objective is not a sufficient policy
distillation method. No sealed test, model promotion, or gameplay benchmark is
authorized.
