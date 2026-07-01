# ADR 0188: O1 High-Regret Draft-Ranking Integration

**Status:** Completed with replicated validation null; sealed test not opened  
**Date:** 2026-06-17  
**Experiment:** `o1-high-regret-draft-ranking-integration-v1`

## Context

ADR 0187 established that the A2 opponent-intent model improves calibrated
future-market-access prediction on two policy families absent from training.
That result authorizes one narrow successor: test whether its frozen
probability surface improves complete-action ranking where the accepted
exact-R2 ranker still has meaningful regret.

The integration must respect two timing facts:

1. O1 predicts from the public state immediately after the focal action.
2. The market refill caused by a candidate action is not observable when that
   candidate is ranked.

Using the realized hidden refill would therefore leak future information.
Using the factual continuation of the historical champion for every
counterfactual action would also be invalid. This experiment instead averages
O1 predictions across a frozen public refill proposal distribution derived
only from the exact semantic public supply.

## Decision

Warm-start the accepted ADR 0150 `c0-full-r2-afterstate` ranker at step 3,000
and freeze every existing parameter. Add one identical, zero-initialized
candidate residual adapter to all four arms:

1. `z0-zero-intent`
   - receives an all-zero 81-value intent vector;
   - controls for adapter capacity and additional optimization.
2. `b1-a0-public-state`
   - receives predictions from the frozen ADR 0187 A0 public-state model.
3. `p2-a2-history-auxiliary`
   - receives predictions from the frozen, selected ADR 0187 A2 model.
4. `s3-a2-stratified-shuffle`
   - receives the same A2 vectors as P2, deterministically permuted within
     split, phase, draft-kind, and market-depletion strata;
   - preserves marginal feature distributions while destroying alignment.

The adapter projects the 81-value vector to width 64, combines it with the
frozen candidate encoding and their elementwise interaction, and emits a
zero-initialized width-64 delta. The original residual and uncertainty heads
remain frozen. Every arm has the same graph, trainable parameter count,
initial trainable tensor, optimizer, batches, and labels.

## Candidate Cohort

The frozen ADR 0150 control scores every candidate before this experiment:

- train: the top 63 control-ranked candidates plus the R4800 selected action
  when it is not already present;
- validation: the strict top 64 control-ranked candidates, with no
  label-dependent insertion;
- sealed test: the strict top 64 control-ranked candidates, built only after
  a validation pass.

Stable ties use the canonical action hash. Candidate membership, base scores,
source indices, action hashes, and cohort hashes are immutable sidecars.

The train-only selected-action insertion keeps the frozen listwise objective
well-defined without contaminating validation retrieval. All claims are about
reranking a fixed base top-64 cohort, not discovering actions outside it.

## Public Refill Integration

For every retained candidate:

1. construct `PositionRecord::observable_afterstate`, which contains the
   exact public placement, Nature Token accounting, and depleted market but no
   hidden refill;
2. append the candidate action as age-zero history after at most 11 preceding
   public champion actions;
3. derive tile-archetype probabilities from the accepted S1 exact semantic
   supply and wildlife probabilities from the staged public bag counts;
4. draw eight deterministic refill proposals from a BLAKE3-keyed sampler;
5. fill only the missing tile and wildlife market components;
6. run frozen A0 and A2 MLX inference on every proposal;
7. average probabilities before producing the 81-value feature vector.

The vector contains:

- 16 tile-disposition probabilities;
- four pair-survival probabilities;
- 16 final-slot probabilities;
- 12 ordered opponent tile-slot probabilities;
- 12 ordered opponent wildlife-slot probabilities;
- three independent-draft probabilities;
- 15 ordered drafted-wildlife probabilities;
- three free-replacement probabilities.

No game identity, policy identity, physical tile identity, hidden stack order,
hidden bag order, excluded-tile identity, future action, teacher value, or
realized refill may enter a model feature.

## Optimization

- MLX GPU only;
- frozen base checkpoint and output heads;
- adapter-only AdamW;
- seed `2026061719`;
- 2,000 fixed steps;
- four groups per step;
- learning rate `1e-4`;
- weight decay `1e-4`;
- canonical geometry only, because O1 probabilities are computed in that
  exact frame;
- checkpoint every 250 steps;
- metric event every 100 steps;
- no validation during training;
- no early stopping;
- final checkpoint only.

The loss remains the ADR 0150 graded-oracle objective over the retained
64-action cohort.

## Validation

Primary endpoint:

- mean top-1 retained R4800 regret over the fixed validation top-64 cohort.

Primary inference is paired by decision group and bootstrapped by source game
with 20,000 replicates. Secondary metrics include top-1 R4800-winner recall,
R1200 pairwise ordering, high-regret groups, phase, low-supply,
Nature-Token-available, and independent-draft-winner slices.

A treatment is eligible only if primary and rotated-host artifacts match and:

1. top-1 retained R4800 regret improves by at least 0.05 points versus Z0;
2. its game-clustered paired 95% interval is wholly below zero;
3. groups whose frozen Z0 regret is at least 0.50 improve by at least
   0.10 points;
4. top-1 R4800-winner recall does not regress;
5. mean R1200 pairwise accuracy does not regress by more than 0.5 percentage
   points;
6. all scores are finite and every cohort action is scored exactly once.

Eligible aligned arms are selected by lower primary regret, higher top-1
recall, higher pairwise accuracy, then stable arm name. S3 is never promotable;
it is a negative control. P2 receives the stronger
`history_aligned_intent_supported` interpretation only if it also beats B1
and S3 with paired intervals below zero.

## Sealed Test

If no aligned arm is eligible, test remains unopened.

If validation selects B1 or P2, build the strict base top-64 test cohort and
open it once. Replication requires:

- at least 0.03 points lower mean top-1 retained R4800 regret than Z0;
- paired game-bootstrap interval wholly below zero;
- at least 0.05 points improvement on frozen-Z0-regret-at-least-0.50 groups;
- no top-1 recall regression;
- no R1200 pairwise-accuracy regression greater than 0.5 percentage points.

A pass authorizes a bounded gameplay experiment. It does not establish a
score gain, promotion, or progress to 100.

## Cluster Allocation

Primary wave:

| Host | Arm |
|---|---|
| john1 | Z0 zero-intent control |
| john2 | B1 frozen A0 public-state predictions |
| john3 | P2 frozen A2 history-aware predictions |
| john4 | S3 stratified shuffled-A2 negative control |

Rotated replay wave:

| Host | Arm |
|---|---|
| john2 | Z0 |
| john3 | B1 |
| john4 | P2 |
| john1 | S3 |

Cache construction is partitioned across all four Macs by source game. No
healthy host may remain idle while a compatible export, inference, training,
replay, evaluation, or verification task is queued.

## Claim Boundary

This experiment may establish only that a frozen, publicly marginalized O1
signal improves offline reranking inside an exact-R2 top-64 cohort. It cannot
claim:

- retrieval outside the cohort;
- exact counterfactual opponent behavior;
- paid-wipe or strategy-switch intent;
- gameplay strength;
- champion promotion;
- a score gain;
- progress to 100.

## Result

All four primary arms and all four rotated-host replays completed on 2026-06-17
with exact replication. No aligned treatment passed the frozen validation
gates.

P2 was directionally best, reducing mean retained R4800 regret from `0.506299`
to `0.497156`, an improvement of `0.009142`. Its game-clustered paired 95%
interval was `[-0.018194, 0.000000]`, and its high-regret improvement was
`0.022008`. These values missed the required `0.05`, interval-below-zero, and
`0.10` high-regret gates. B1 was effectively null. S3 regressed overall regret
and top-1 recall.

The terminal validation classification is
`o1_ranking_validation_factorial_null`. The test classification is
`o1_ranking_test_not_opened`; no test feature, score, cohort, model, or metric
was produced.

The permanent result is documented in
`reports/o1-high-regret-draft-ranking-integration-v1-result.md`. The only
authorized O1 successor is a separately preregistered use of frozen A2
opponent probabilities inside one-rotation public-belief search. Another
direct adapter or feature-fusion sweep is not authorized by this result.
