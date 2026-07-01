# V1 Score Anatomy Matched R2 Preregistration

Date: 2026-06-16

Experiment: `v1-score-anatomy-matched-r2-v1`

ADR: `0176-v1-score-anatomy-matched-r2.md`

## Question

Does exact component supervision improve final-score calibration and
same-round rank quality when representation, capacity, initialization,
optimization, data, and compute budget are identical?

## Hypothesis

The eleven exact components provide denser credit assignment than a scalar
total. The component arm should preserve total MAE while improving correlation
and pairwise score ordering.

## Representation

Use the accepted exact sparse R2 Perceiver. It has a frozen capacity of 92
sparse objects per board, below the recalled 121-cell reference and far below
the legacy 441-cell lattice. This is not an 11 by 11 crop: it preserves exact
occupied, frontier, habitat-component, and wildlife-motif objects and exact D6
semantics.

## Arms

| Role | Host | Objective |
|---|---|---|
| scalar primary | john2 | final total MSE only |
| anatomy primary | john3 | 11 normalized component MSE plus total consistency |
| scalar replay | john4 | exact replay of scalar primary |
| anatomy replay | john1 | exact replay of anatomy primary |

Every role uses the same model graph and initial parameter tensor.

## Metrics

- total MAE, RMSE, bias, correlation, and calibration line;
- all eleven component MAE, RMSE, bias, and correlation;
- opening, early-middle, late-middle, and endgame total metrics;
- within-game, within-four-turn-round pairwise accuracy and soft log loss;
- fixed 256-row prediction probe digest;
- final parameter tensor digest;
- throughput and peak active/process memory.

The pairwise metric is an offline score-ordering proxy, not an action-policy or
gameplay claim.

## Frozen Promotion Rule

Promote only when all integrity gates pass and:

1. anatomy total MAE is at most scalar MAE plus 0.05;
2. anatomy total correlation is at least scalar correlation plus 0.03;
3. anatomy pairwise log loss is at least 0.005 lower than scalar;
4. all component metrics are finite;
5. exact sparse capacity is at most 121 objects per board.

Thresholds may not be changed after opening the reports. A promoted result
authorizes a subsequent complete-action value test at equal proposal/search
settings. It does not authorize gameplay promotion by itself.
