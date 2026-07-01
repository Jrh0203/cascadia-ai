# V2 Distributional Opportunity Supervision Preregistration

Experiment: `v2-distributional-opportunity-supervision-v1`

ADR: `0179-matched-r12-distributional-opportunity-supervision.md`

Status: frozen before substantive optimization

## Hypothesis

The R12 teacher contains stable, action-relevant uncertainty that a
state-conditional distribution head can learn. A successful head should
improve probabilistic calibration and identify when its expected value is
unreliable while preserving the expected-score action ranking.

## Frozen Data

Train:

- path: `artifacts/datasets/r12-counterfactual-advantage-v1-train-128`;
- manifest BLAKE3:
  `c4102f016feecb103d2924656a82404fbb27b9f54db39dda66e46ff8ba3737da`;
- 128 games, 2,048 groups, 8,192 candidates, 98,304 continuations;
- game indices 69,000 through 69,127.

Validation:

- path: `artifacts/datasets/r12-counterfactual-advantage-v1-validation-32`;
- manifest BLAKE3:
  `02d3603a8ac9e36cd03ba60c160b18296b722d687c72221b2df95d2467bfd392`;
- 32 games, 512 groups, 2,048 candidates, 24,576 continuations;
- game indices 70,000 through 70,031.

Both datasets are four-player Standard Cascadia, AAAAA cards, habitat bonuses
disabled, selected/high/median/low candidate strata, 12 shared public
redeterminations, and deterministic rejection of impossible stabilized-market
branches.

## Frozen Target

For candidate `c` and shared sample seed `s`:

`y[c,s] = terminal_total[c,s] - mean_c terminal_total[c,s]`.

Expected value is `mean_s y[c,s]`. Distribution losses consume all 12
candidate-centered samples. No raw hidden state, future refill order, test
record, or final record is available to the model.

## Frozen Common Model

- action-afterstate encoder: 4 relative-seat boards, exact public market,
  global features, and action features;
- separate 30-value exact public-supply projection;
- complete four-candidate self-attention;
- hidden width 96, four attention heads;
- two board blocks, one market block, two candidate blocks;
- feed-forward multiplier 3;
- shared 13-output head;
- total trainable parameters: 836,365;
- output layer initialized to zero;
- common initial parameter tensor required across all arms.

Output 0 is the bounded expected-value correction. Outputs 1 through 12 have
arm-specific semantics but remain present and trainable in every graph.

## Frozen Arms

`c0-homoscedastic-mean`:

- 12 fixed residual atoms fitted on train only;
- residual atom vector is ordered, zero-centered, and authorization-bound;
- CRPS plus common expected-value objective;
- unused learned auxiliary outputs receive `0.01` square regularization.

`g1-heteroscedastic-gaussian`:

- learned scale `0.25 + softplus(raw_scale)`;
- 12 fixed normal quantile locations;
- Gaussian NLL plus common expected-value objective;
- unused auxiliary outputs receive `0.01` square regularization.

`q2-quantile`:

- 12 learned offsets, mean-centered per candidate;
- quantile levels `(i + 0.5) / 12`;
- pinball loss;
- adjacent crossing penalty weight `0.05`;
- sorted only for evaluation, never to hide training crossings.

`e3-crps-atoms`:

- 12 learned mean-centered atoms;
- empirical CRPS:
  `E|atom - sample| - 0.5 E|atom - atom'|`.

## Frozen Optimization

- seed: `2026061802`;
- steps: 3,000;
- group batch size: 32;
- optimizer: AdamW;
- learning rate: `3e-4`;
- weight decay: `1e-4`;
- common mean/ranking objective:
  Huber + `0.50 * hard-top CE + 0.25 * soft-listwise CE`;
- distribution objective weight: `0.25`;
- checkpoint every 500 steps;
- deterministic full-corpus epoch permutation keyed by seed plus epoch;
- no early stopping;
- no validation during training;
- final step 3,000 is the only selected checkpoint.

## Frozen Metrics

Expected-value quality:

- centered MAE and RMSE;
- centered target correlation;
- pairwise mean-order accuracy;
- top-action agreement;
- top-value recall;
- mean top-action regret;
- immediate-score baseline.

Distribution quality:

- empirical CRPS over all 12 observations;
- empirical 80% interval coverage and mean width;
- predicted-width versus absolute mean-error correlation;
- predicted versus empirical standard-deviation correlation and MAE;
- pairwise probability Brier score, log loss, and correlation;
- 80% winner-confidence-set coverage, mean size, and singleton fraction.

Integrity:

- exact authorization and dataset identity;
- common parameter count, layout, and initialization;
- final tensor BLAKE3;
- serialized model-file BLAKE3;
- fixed 64-group validation prediction-probe BLAKE3;
- complete finite metrics;
- rotated-host role-neutral identity equality.

## Frozen Classification

Eligibility thresholds and selector order are exactly those in ADR 0179.
Failure of any replay or model-byte check makes every treatment ineligible.

Classification strings:

- `distributional_opportunity_arm_selected`; or
- `distributional_opportunity_factorial_null`.

## Claim Boundary

This is an open-validation representation and supervision study. It cannot:

- open test or final data;
- run paired gameplay;
- change the champion;
- claim a score improvement;
- claim progress toward 100; or
- introduce risk-sensitive action selection.

Its only positive terminal action is authorization of a new, separately
preregistered successor experiment.
