# ADR 0099: Frontier Supervision Identifiability

Status: complete; `uncertainty_aware_supervision_sufficient`.

Date: 2026-06-16

Experiment ID: `complete-action-frontier-supervision-identifiability-v1`

## Context

ADR 0098 trained four fresh 6.7M-7.6M parameter constructors directly from
lossless public observables. Their best train target recall was 37.87%, and no
arm recovered one complete train set. ADRs 0091-0097 had already rejected
three target objectives, frozen heads, raw bypasses, set context, projected
factors, and factor integration. Another neural constructor, head, pool, width,
or optimizer treatment is therefore prohibited.

The hard learned target is not a ground-truth policy. For each decision it
keeps deterministic frontier anchors and marks the highest finite-sample R1200
nonfrontier means needed to fill width 64. The R1200 cutoff can turn tiny,
statistically unresolved value differences into opposite binary labels. Exact
observable collisions being zero proves only that labels are unique, not that
they are stable or learnable from the underlying game signal.

This audit tests the supervision directly before another MLX run.

## Frozen Inputs

- Train dataset:
  `artifacts/datasets/complete-action-graded-oracle-v1-train`.
- Validation dataset:
  `artifacts/datasets/complete-action-graded-oracle-v1-validation`.
- Train manifest BLAKE3:
  `7ed12c943d75a786ccd4ccbe11a6b0146aad4fe5ed40f0cbaf1d652f5ac0bb99`.
- Validation manifest BLAKE3:
  `302ceb7a57482b0fb5fb12963521be35aafc121a36f572e6b9f47def1b820a31`.
- The exact ADR 0089 frontier bit, width 64, stable action-hash order, hard
  R1200 target, and R4800 evaluation definitions remain unchanged.
- Standard error is `stddev / sqrt(max(samples, 1))`, matching ADR 0086.

The sealed test, gameplay, new rollout samples, hidden state, K2048, cloud, and
external compute are prohibited. Normal approximations are diagnostics because
adaptive allocation and common random numbers can create unknown covariance.

## Pre-Result Coverage Correction

The first cross-fidelity launch failed before writing a report because at least
one group had fewer R600-labeled nonfrontier actions than the learned quota.
The preregistration phrase "wherever each frozen cohort can fill width 64" did
not define how incomplete groups affected the split gate.

Before any cross-fidelity metric was observed, the correction was frozen:

- audit every group and report the fraction whose R600 cohort can fill the
  unchanged learned quota;
- compute R600/R1200 overlap metrics only on fillable groups;
- require 100% group coverage in addition to the original 80% target-recall
  and 25% exact-set gates; and
- treat any incomplete group as a failed split-level cross-fidelity gate.

The failed event log is preserved under
`invalid-launch-r600-coverage/`. The other first-launch reports are quarantined
with it because the corrected source bundle must be identical across all four
origins. No threshold, result interpretation, or other arm changed.

## Frozen Arms

### Boundary Signal-To-Noise

- Host: john1.
- For every group, identify the weakest nominal R1200 target and strongest
  excluded R1200-labeled nonfrontier action.
- Report raw margin, combined standard error, z-score, and whether the cutoff
  clears the two-sided 95% normal threshold.
- For every nominal target slot, test whether it clears that strongest excluded
  action at the same threshold.
- Hard-boundary stability passes a split only if at least 80% of target slots
  and 25% of complete target sets are statistically separated.

### Cross-Fidelity Target Stability

- Host: john2.
- Rebuild the same frontier-anchored learned quota independently from R600 and
  R1200 means wherever each frozen cohort can fill width 64.
- Report R600-to-R1200 target recall, exact set agreement, Jaccard overlap,
  common-cohort rank correlation, R4800-winner membership, and the fraction of
  groups whose R600 cohort can fill the quota.
- Cross-fidelity stability passes a split only if every group is fillable,
  R600 recovers at least 80% of R1200 target slots on those groups, and at
  least 25% of complete R1200 target sets agree.

### Finite-Teacher Resampling Stability

- Host: john3.
- Seed: `2026061625`.
- Draw exactly 512 independent-normal synthetic R1200 mean realizations per
  group from the recorded mean and standard error.
- Rebuild the complete nonfrontier quota for every draw.
- Report mean nominal-target recall, exact-set reproduction probability,
  Jaccard overlap, per-target inclusion probability, and entropy.
- Resampling stability passes a split only if mean nominal-target recall is at
  least 80% and exact-set reproduction is at least 25%.

### Uncertainty-Aware Expected-Rank Ceiling

- Host: john4.
- For each R1200-labeled nonfrontier action, compute its expected rank under
  independent normal teacher uncertainty:
  `1 + sum_j P(value_j > value_i)`.
- Use a fixed high-accuracy normal-CDF approximation, stable action hashes for
  ties, deterministic frontier anchors, and the lowest expected ranks to fill
  width 64.
- Report nominal-target agreement and the unchanged R4800 exact-winner,
  confidence-set, distinguishable-winner, regret, and phase metrics.
- The soft ordinal ceiling passes only if validation achieves:
  - exact R4800-winner recall strictly above 98%;
  - R4800 95% confidence-set coverage at least 99%;
  - distinguishable-winner recall at least 98%;
  - retained mean R4800 regret below 0.03;
  - every phase exact recall and confidence coverage at least 98%; and
  - every phase retained regret below 0.03.

## Classification

1. `uncertainty_aware_supervision_sufficient` if the expected-rank ceiling
   passes, regardless of whether the hard target is stable.
2. `hard_target_stable_but_soft_ceiling_insufficient` if boundary,
   cross-fidelity, and resampling gates all pass on train and validation but
   the expected-rank ceiling fails.
3. `existing_teacher_supervision_insufficient` if the expected-rank ceiling
   fails and any hard-target stability gate fails.

A sufficient expected-rank ceiling authorizes one separately preregistered,
single-host MLX pilot with uncertainty-aware ordinal supervision. It does not
authorize test or gameplay. Existing-teacher insufficiency requires a new
teacher-allocation or repeated-sample design; it prohibits another model,
loss, width, or optimizer experiment on these labels.

## Correctness Gates

- Unit tests cover exact target reconstruction, stable hash ties, standard
  errors, boundary significance, cross-fidelity overlap, seeded resampling,
  normal-CDF accuracy, expected-rank symmetry, R4800 ceiling metrics, strict
  gate boundaries, and classification.
- Every mode reads each open group and candidate exactly once from the
  immutable grouped binary data and reports finite metrics.
- All four Macs use a byte-identical source bundle and matching dataset
  manifests.
- Each origin report is replayed on the next host in the ring and must produce
  a bit-identical scientific payload.

## Cluster Execution

- john1: boundary signal-to-noise, replayed on john2.
- john2: cross-fidelity target stability, replayed on john3.
- john3: seeded finite-teacher resampling, replayed on john4.
- john4: expected-rank ceiling, replayed on john1.
- All four distinct CPU jobs launch concurrently under host locks and
  `caffeinate`. Each job uses exactly eight ordered worker processes, leaving
  two physical cores per Mac for orchestration, SSH, telemetry, and the
  dashboard while using 32 cluster cores for scientific work.
- Replays launch as soon as both the source report and destination host are
  available.
- No neural training, duplicate discovery arm, same-mode seed replica, or
  intentionally idle mirror process is authorized.

The report records productive time, dependency-blocked time, queued idle,
candidate throughput, peak RSS, swaps, and hypotheses per cluster hour.

## Stop Rule

Run each frozen arm once with eight workers and its deterministic ring replay
once. Do not change the 95% threshold, 80%/25% stability gates, 512 draws,
random seed, worker count, expected-rank definition, width, teacher, target,
split, or ceiling gates after results are visible.

## Maximum Compute

Four one-pass open-data CPU audits, four deterministic ring replays, tests,
source identity checks, and one combined report. No MLX training, new teacher
sample, sealed test, gameplay, K2048, cloud, or external compute is authorized.

## Result

All four corrected origins and all four ring replays completed over every open
group and candidate. Every scientific payload reproduced bit-for-bit across
hosts, peak process RSS stayed near 1.01 GB, and no process swapped.

The hard R1200 top-64 membership target is not statistically stable:

- only 10.38% of validation target slots cleared the frozen 95% boundary test;
- zero of 240 validation target sets were completely separated;
- the median validation cutoff margin and z-score were both zero;
- no train or validation group had enough R600-labeled nonfrontier actions to
  reconstruct the same width-64 learned quota; and
- 512 finite-teacher resamples recovered only 41.20% of nominal validation
  target slots on average and reproduced only 2.50% of complete target sets.

The uncertainty-aware expected-rank target passed every frozen ceiling gate:

- validation nominal hard-target recall: 92.11%;
- validation nominal exact target sets: 27.08%;
- validation R4800-winner recall: 100%;
- validation R4800 95% confidence-set coverage: 100%;
- validation distinguishable-winner recall: 100%;
- validation retained regret: 0.000000; and
- early, middle, and late exact recall, confidence coverage, and regret all
  passed.

Train also passed at 99.46% exact-winner recall, 99.82% confidence coverage,
99.44% distinguishable recall, and 0.001487 regret. The expected-rank origin on
john4 and replay on john1 were scientifically identical.

The corrected eight-job origin/replay matrix finished in 21.79 seconds of
cluster wall time. Each job used eight ordered workers, duplicate discovery
compute remained zero, and all sealed domains remained closed.

The preregistered classification is
`uncertainty_aware_supervision_sufficient`. The next authorized experiment is
one separately frozen, single-host MLX pilot that learns the continuous
expected-rank ordering. Another hard-cutoff learner, neural constructor sweep,
width change, or optimizer sweep is prohibited.

Machine-readable result:
`artifacts/experiments/complete-action-frontier-supervision-identifiability-v1/reports/combined.json`.

Human-readable result:
`docs/v2/reports/complete-action-frontier-supervision-identifiability-v1-result.md`.
