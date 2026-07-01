# ADR 0179: Matched R12 Distributional Opportunity Supervision

- Status: completed with `distributional_opportunity_factorial_null`
- Date: 2026-06-16
- Experiment: `v2-distributional-opportunity-supervision-v1`
- Goal: determine whether calibrated return distributions improve opportunity
  value without weakening expected-score decisions
- Domains opened: qualified ADR 0078 train and validation only
- Domains closed: test, final, paired gameplay, champion replacement

## Terminal Result

All four primary arms and rotated-host replays completed exactly on
2026-06-17. G1 and Q2 improved CRPS, pairwise probability calibration, and
mean top-action regret. Q2 reduced regret by 0.1012 points and improved
top-value recall by 0.0234, but its 95% winner confidence set still contained
3.60 of four candidates on average. G1's contained 3.66. Both failed the
frozen informativeness gate; E3 failed multiple calibration and coverage
gates.

No arm was selected, test and final remained closed, and successor training
was not authorized. See
`reports/v2-distributional-opportunity-supervision-v1-result.md`.

## Context

ADR 0078 established that the qualified R12 counterfactual corpus is
observable, deterministic, checksummed, and learnable at the level of broad
candidate ordering. Its point-estimate set ranker improved centered MAE and
pairwise accuracy but failed the frozen best-action and regret gates. The
dataset remains valid; the rejected model and its research conclusions do not
constrain this new hypothesis.

Each decision group contains four legal public afterstates evaluated under the
same 12 public redeterminations. That shared-seed structure exposes more than a
noisy scalar mean. It contains:

- candidate-relative terminal returns for every common future;
- heteroscedasticity by action and state;
- asymmetric or multimodal tails;
- empirical pairwise win probabilities; and
- uncertainty about which candidate is genuinely best.

The action-relevant target is defined before training as:

`return(candidate, seed) - mean_candidates(return(candidate, seed))`.

This removes shared game difficulty and preserves exactly the within-decision
signal used to choose an action.

## Pre-Outcome Reliability Audit

The strict reader revalidated every manifest, shard, checksum, group, action,
seed, and AAAAA rule contract before the audit.

| Split | Groups | Candidates | Mean split-half correlation | Winner agreement | R6 choice regret under full R12 |
|---|---:|---:|---:|---:|---:|
| Train | 2,048 | 8,192 | 0.6884 | 46.88% | 0.1312 / 0.1268 |
| Validation | 512 | 2,048 | 0.6744 | 47.66% | 0.1198 / 0.1419 |

Candidate-centered returns span `-12.50` to `+10.75` on train and have
standard deviation `2.1653`. Split-half reliability is lower for distribution
shape than for the mean:

- train standard deviation correlation: `0.5487`;
- train 10th / median / 90th percentile correlation:
  `0.6382 / 0.6434 / 0.6215`;
- train 80% width correlation: `0.5430`;
- train pairwise win-probability correlation: `0.5409`.

This is sufficient to test a regularized distributional model, but it is not
sufficient to treat any single R12 winner as ground truth without uncertainty.

## Decision

Run a four-arm, parameter-matched MLX factorial:

1. `c0-homoscedastic-mean`
   - fixed train-only empirical residual atoms;
   - common expected-mean and ranking objective;
   - empirical CRPS against the fixed homoscedastic distribution;
   - control for the benefit of a distribution loss without state-dependent
     uncertainty.
2. `g1-heteroscedastic-gaussian`
   - candidate-specific learned scale;
   - fixed standard-normal atom locations;
   - Gaussian negative log likelihood.
3. `q2-quantile`
   - 12 direct conditional quantiles;
   - pinball loss plus a preregistered crossing penalty.
4. `e3-crps-atoms`
   - 12 unrestricted conditional atoms;
   - empirical CRPS / energy-score objective.

All arms use the same:

- 836,365-parameter graph;
- byte-identical initialization;
- exact action-afterstate, public-supply, and four-candidate attention trunk;
- 13-output head;
- centered expected-score correction;
- mean Huber, hard-top, and soft-listwise loss;
- distribution-loss weight `0.25`;
- 3,000 optimizer steps, batch size 32, AdamW, learning rate `3e-4`,
  weight decay `1e-4`;
- deterministic epoch permutations;
- no validation during training; and
- fixed final-step checkpoint.

The homoscedastic residual atoms are computed once from train only and bound
into the authorization. Validation cannot affect initialization, training,
checkpoint selection, or thresholds.

## No Hidden Risk Preference

The only action objective is predicted expected score. Quantiles, intervals,
variance, confidence sets, and pairwise probabilities are diagnostics and
future search inputs only. No arm may choose an action by lower confidence
bound, upper confidence bound, CVaR, variance penalty, or any other implicit
risk preference in this experiment.

## Cross-Host Design

Primary wave:

| Host | Arm |
|---|---|
| john1 | C0 homoscedastic control |
| john2 | G1 heteroscedastic Gaussian |
| john3 | Q2 quantile |
| john4 | E3 CRPS atoms |

Rotated replay wave:

| Host | Arm |
|---|---|
| john2 | C0 homoscedastic control |
| john3 | G1 heteroscedastic Gaussian |
| john4 | Q2 quantile |
| john1 | E3 CRPS atoms |

Every arm must reproduce its final parameter tensor, serialized model bytes,
validation prediction probe, metrics, and role-neutral scientific identity
exactly on the rotated host.

## Success Gates

A treatment is eligible only if all eight reports pass integrity and the arm:

1. regresses centered mean MAE by no more than `0.03`;
2. regresses mean top-action regret by no more than `0.05`;
3. regresses top-value recall by no more than `0.01`;
4. improves empirical CRPS by at least `0.02`;
5. improves pairwise-probability Brier score by at least `0.005`;
6. improves uncertainty-versus-absolute-mean-error correlation by at least
   `0.05`;
7. covers a full-R12 expected winner in at least 90% of 80% confidence sets;
   and
8. has mean winner-set size at most `3.50` candidates.

Among eligible arms, selection is lexicographic by:

1. lower CRPS;
2. lower pairwise Brier score;
3. lower centered mean MAE;
4. lower mean top-action regret; and
5. stable arm name.

Thresholds are immutable after this ADR.

## Consequences

A selected arm authorizes a separately preregistered action-value successor
study. It does not authorize test, gameplay, champion replacement, or a claim
of progress toward 100.

A null result closes distributional supervision on this graph and corpus. It
does not invalidate the R12 data, the target reliability audit, or future
work on richer relational state representations.
