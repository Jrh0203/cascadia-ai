# Canonical Action Score Residual v3 Validation

Status: rejected on validation on 2026-06-12. No test, promotion, or gameplay
domain was opened.

## Corpus

- Train: immutable ADR 0053 evidence, 64 games, 5,120 groups, 491,520 actions,
  and 153,021 aligned R600 estimates.
- Validation: 16 fresh games at split indices 51,016-51,031, 1,280 groups,
  122,880 actions, and 38,705/38,705 aligned R600 estimates.
- Validation source dataset:
  `canonical-action-imitation-validation-cc2a28ac54be4b71`.
- Validation target dataset: `imitation-targets-18734a81d30a09df`.
- Validation source manifest BLAKE3:
  `26fc500604c4bce4dfd84f9eb096484d02097c8fc4b3e1f063ecb6ec0ae2d8d0`.
- Validation target manifest BLAKE3:
  `8c318b144fcdc8072ffbad5a280ac3fbd84a98c777f618d327956173e2d7096c`.

Collection took 2,353.2 seconds. Rust validation and an independent Python
streaming pass reproduced all 1,280 groups and 38,705 scored actions.

## Training

- Device: `Device(gpu, 0)`.
- Architecture: `shared-state-action-score-residual-v3`, hidden 96, four
  heads, two board blocks, one market block.
- Optimizer: AdamW, learning rate `1e-4`, weight decay `1e-4`.
- Batch: 16 complete groups.
- Seed: 20260617.
- Stop: full 20-epoch budget, 6,400 optimizer steps.
- Training runtime: 133.052 seconds.
- Best checkpoint: `step-000005440-epoch-0017-batch-000000`.
- Best checkpoint manifest BLAKE3:
  `002eba584b1398027a1f2cfcfee0ee540977bda9770a38cd888047cf61c25a21`.
- Model BLAKE3:
  `5106f4d784805f12cbe599a7fe2bf51c1f319288b8f501f61db38b6dabd22bba`.
- Final report BLAKE3:
  `8ef51ff91410b94efe23bba51de6b8a050268576eeee486c795bc8e997f40e25`.

A completed-run resume performed zero optimizer steps and preserved both the
final report and best-pointer hashes exactly.

## Frozen Gates

| Gate | Required | Initial | Selected | Pass |
|---|---:|---:|---:|:---:|
| Teacher alignment | 100% | n/a | 100% | yes |
| Anchored loss | below initial | 4.984072 | 0.984838 | yes |
| Selected top one | >=0.23 and +0.02 | 0.189063 | 0.171875 | no |
| Selected top five | >=0.50 and +0.08 | 0.364844 | 0.435156 | no |
| Selected MRR | >=0.36 and +0.06 | 0.289552 | 0.307109 | no |
| Predicted teacher coverage | >=0.80 | 0.791406 | 0.766406 | no |
| Scored pairwise accuracy | >=0.70 | 0.569946 | 0.662828 | no |
| Value-difference correlation | >=0.45 | 0.567485 | 0.380501 | no |
| Conditional mean regret | <=1.00 | 1.007402 | 1.157315 | no |

The checkpoint improved top-five and MRR modestly, but it regressed exact
top-one, teacher-frontier coverage, value-difference correlation, and regret.

## Diagnosis

The absolute target was dominated by an action-independent state offset. On
the immutable train corpus, only 0.438% of continuation-residual variance was
within action groups. On the fresh validation corpus, only 0.456% was within
groups:

- total continuation-residual variance: 590.302;
- mean within-group variance: 2.692;
- variance of group means: 585.304;
- median scored-action residual range within a decision: 6.117 points.

The model could therefore reduce absolute point loss by learning phase and
state workload while worsening the small differences that choose an action.
This is exactly what happened: loss fell by 80.2%, while value-difference
correlation fell from 0.5675 to 0.3805 and exact selected top-one fell from
18.91% to 17.19%.

The next target must remove the groupwise nuisance constant and train the
decision-local continuation advantage directly. Exact immediate score remains
part of inference, but absolute remaining-score regression is closed.

A disposable development probe tested that correction on this already-open
split. It raised teacher-frontier coverage to 83.59% and value-difference
correlation to 0.5459, but best exact top-one was only 17.50% and conditional
regret was 1.1874. No fresh evidence was collected. The probe closes target
centering alone and points to apprentice representation or direct MLX reuse of
the qualified NNUE as the next lever.

## Conclusion

ADR 0054 is rejected. Point-scale anchoring alone does not solve teacher
distillation because absolute continuation labels are almost entirely
action-independent within each decision. No sealed test, promotion, or
gameplay benchmark is authorized.
