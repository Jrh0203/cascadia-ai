# MCE Teacher Identifiability Audit

Date: 2026-06-13

## Question

Does the fixed K32/R600 sequential-halving teacher identify a stable best
action strongly enough for exact-action policy imitation?

This audit uses only the already-open ADR 0070 train and validation evidence.
It performs no training, opens no test or gameplay domain, and changes no
teacher output.

## Method

For each decision, the selected action is compared with the runner-up
rollout mean among teacher-scored actions. The audit reports:

- top-two mean margin;
- combined standard error under an independence approximation;
- fraction clearing a two-sided 95% normal difference threshold;
- fraction whose separate 95% confidence intervals do not overlap;
- number of actions statistically indistinguishable from the selected action;
- exact-parent and immediate-score rank of the selected action;
- phase results using each player's turns 0-4, 5-9, 10-14, and 15-19.

Adaptive halving and shared game structure can violate the normal independence
approximation. The audit therefore treats these statistics as diagnostics,
not calibrated guarantees.

## Result

| Metric | Train, 5,120 decisions | Validation, 1,280 decisions |
|---|---:|---:|
| Mean scored candidates | 29.887 | 29.879 |
| Mean top-two margin | 0.366 | 0.385 |
| Median top-two margin | 0.253 | 0.278 |
| Margin at most one point | 93.789% | 92.813% |
| Margin within one combined SE | 60.410% | 58.359% |
| Margin within the training loss scale | 95.430% | 94.531% |
| Winner distinguishable at 95% | 16.973% | 18.359% |
| Non-overlapping 95% intervals | 5.859% | 6.953% |
| Mean 68% confidence-set size | 3.838 | 3.788 |
| Mean 95% confidence-set size | 10.137 | 10.140 |
| Exact parent ranks selected first | 20.840% | 21.641% |
| Exact parent ranks selected top five | 47.539% | 46.484% |

The phase result is more severe where long-horizon planning matters most:

| Validation phase | 95%-distinguishable winner | Mean 95% set size |
|---|---:|---:|
| Opening, turns 0-4 | 6.563% | 16.309 |
| Early, turns 5-9 | 12.813% | 11.494 |
| Middle, turns 10-14 | 19.063% | 7.894 |
| Late, turns 15-19 | 35.000% | 4.863 |

The selected action receives a mean 125.7 samples and the runner-up 95.7, so
this is not merely a minimum-sample tail. The adaptive teacher spends most of
its budget distinguishing alternatives whose estimated values remain far
closer than their uncertainty.

## Interpretation

Exact-action top-one is not the right learning objective for this corpus. In
more than four of five validation decisions, the recorded winner does not
clear even a normal 95% difference test, and roughly ten actions remain in the
average 95% confidence set. Opening decisions are especially weakly
identified.

This also explains why ADRs 0069 and 0070 improved distributional loss without
meaningful selected-action gains: the target asks the model to reproduce an
adaptive Monte Carlo argmax that is usually not statistically unique.

The next experiment changes the teacher rather than the apprentice. It tests
common random numbers inside the same fixed-budget sequential-halving search,
pairing stochastic futures across candidates to reduce variance of action
differences. This is a standard simulation-comparison technique, but its
benefit depends on positive induced correlation and must be established in
Cascadia gameplay.

## Artifacts

- Train JSON:
  `docs/v2/reports/mce-teacher-identifiability-train.json`
  (`87d017ecaeeebd1421bb881db2b3fd1147a4d45e16fc91208cd4a7e7ecca09dd`)
- Validation JSON:
  `docs/v2/reports/mce-teacher-identifiability-validation.json`
  (`b06abe66f76b2db518fa233d824fce47ecb6509ea353b80f223cdffa643f43f4`)
- Command: `make audit-imitation-identifiability`

## Literature

- Karnin, Koren, and Somekh, "Almost Optimal Exploration in Multi-Armed
  Bandits": https://proceedings.mlr.press/v28/karnin13.html
- Glasserman and Yao, "Some Guidelines and Guarantees for Common Random
  Numbers":
  https://business.columbia.edu/sites/default/files-efs/pubfiles/4261/glasserman_yao_guidelines.pdf
- Veness et al., "Variance Reduction in Monte-Carlo Tree Search":
  https://webdocs.cs.ualberta.ca/~bowling/papers/11nips-vrmcts.pdf
