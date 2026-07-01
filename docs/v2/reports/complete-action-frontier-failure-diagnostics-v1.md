# Complete-Action Frontier Failure Diagnostics V1

Status: **complete**

Date: 2026-06-16

Experiment ID: `complete-action-frontier-failure-diagnostics-v1`

## Verdict

The frontier ranker is failing to fit its training target. Exact observable
collisions, ordinary train-to-validation generalization, and one concentrated
error slice are not the primary mechanism.

The next authorized experiment is one single-host target-set pilot that
changes a structural capacity or optimization constraint. Duplicate replicas
remain closed.

## Results

| Host | Diagnostic | Result | Wall |
|---|---|---|---:|
| john1 | train fit | underfit: 29.36% train target recall, 0.18% exact sets | 71.13 s |
| john2 | exact collision | 0 contradictions across 2,995,314 actions | 6.16 s |
| john3 | objective gradient | cosine -0.908; target norm 7.746, auxiliary 1.695 | 10.69 s |
| john4 | error anatomy | broad misses; no frozen concentration gate passed | 10.19 s |

All jobs launched within 0.36 seconds, completed without process swaps, and
used one byte-identical 86-file MLX source bundle. The longest job determined
cluster wall at 71.31 seconds; the other hosts were not assigned duplicate
work after their independent diagnostic completed.

## Train Fit

- Train target-positive recall: `0.293614`.
- Validation target-positive recall: `0.262132`.
- Train exact target-set fraction: `0.001786`.
- Validation exact target-set fraction: `0`.
- Train-to-validation target-recall gap: `0.031482`.
- Train exact winner recall: `0.769643`.
- Validation exact winner recall: `0.766667`.

The nearly identical winner recall and small target-recall gap show that the
model is not overfitting and then failing to generalize. It is underfitting
the much harder complete target-set allocation on train and validation alike.

## Exclusions

The collision audit found 800 unique complete contexts, no repeated context,
no duplicate scored candidate observation within a context, and no
contradictory target occurrence. Exact input aliasing contributes zero
measured irreducible target mass.

The objective audit found a strongly negative target/listwise cosine on the
eight widest groups, but target-set gradient pressure remained 4.57 times the
combined weighted auxiliary norm. This is useful secondary evidence, not a
pass of the frozen primary-conflict gate.

Error anatomy found 6,021 misses among 8,160 validation target slots. The
model retained 71.16% of target actions already ranked in the screen top 64,
then only 8.71% at ranks 65-128, 1.37% at 129-256, and effectively zero after
rank 256. The pattern is broad across wildlife types and phases, pointing
toward model score capacity or optimization rather than a narrow feature
omission.

## Artifacts

- Combined decision:
  `artifacts/experiments/complete-action-frontier-failure-diagnostics-v1/reports/combined.json`.
- Per-host reports:
  `artifacts/experiments/complete-action-frontier-failure-diagnostics-v1/reports/`.
- Source identities:
  `artifacts/experiments/complete-action-frontier-failure-diagnostics-v1/source-identity/`.
- Protocol:
  `docs/v2/decisions/0090-frontier-ranker-failure-classification.md`.

Sealed test, gameplay, new teacher compute, training, cloud, and external
compute remained closed.
