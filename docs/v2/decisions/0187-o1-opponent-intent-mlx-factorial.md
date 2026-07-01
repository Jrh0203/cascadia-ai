# ADR 0187: O1 Opponent Intent MLX Factorial

**Status:** Completed  
**Date:** 2026-06-17  
**Experiment:** `o1-opponent-intent-mlx-factorial-v1`

## Context

ADR 0186 established a policy-held-out sequential corpus with 126,464 exact
decision windows. It authorizes public-state controls, recent public action
history, next-draft auxiliary targets, exact market-survival targets, and
policy-held-out calibration. It does not authorize paid-wipe, strategy-switch,
champion-transfer, gameplay, or score claims.

The next question is causal and narrow: does observable recent behavior add
decision-relevant information about which of the four current market tiles
will be consumed by opponent one, two, or three, or survive until the focal
player acts again?

## Decision

Run one parameter-matched MLX graph under four gates:

1. `a0-public-state`
   - compact public state only;
   - survival, pair-survival, and final-slot heads.
2. `a1-recent-history`
   - A0 plus the last 12 public actions.
3. `a2-next-draft-auxiliary`
   - A1 plus ordered next-draft auxiliary losses for tile slot, wildlife slot,
     draft mode, wildlife species, and free market replacement.
4. `a3-joint-intent-survival`
   - A2 plus explicit per-opponent intent tokens cross-attended by each market
     slot before survival classification.

Every arm has 374,171 trainable parameters, one byte-identical initialization,
and the same state, history, intent, and output modules. Arm identity changes
only three scalar gates: history input, auxiliary loss, and intent-to-survival
routing.

All boards are rotated into focal-seat order before MLX sees them. Policy IDs,
game IDs, physical tile IDs, future actions, survival labels, final scores, and
the unused 441-hex representation are excluded from model inputs.

## Optimization

- 5,120 fixed steps, exactly eight passes over 77,824 train windows;
- deterministic shard-first batches of at most 128 windows;
- AdamW, learning rate `3e-4`, weight decay `1e-4`;
- checkpoint every 640 steps;
- no validation during training;
- no early stopping;
- final step only.

## Primary Endpoint

The primary endpoint is four-way multiclass Brier score over each initial
market tile:

- consumed by opponent one;
- consumed by opponent two;
- consumed by opponent three;
- survives until the focal player's next access.

Inference is paired by decision window and bootstrapped by game. Log loss,
top-label calibration, binary survival Brier score, pair survival, final slot,
phase slices, and next-draft heads are guardrails or auxiliary diagnostics.

## Validation Gate

A noncontrol arm is eligible only if all primary and replay artifacts agree and
the arm:

1. improves disposition Brier by at least 1% relative to A0;
2. has a game-clustered paired 95% interval wholly below zero;
3. regresses disposition log loss by no more than 0.005;
4. regresses top-label ECE by no more than 0.010;
5. does not regress binary survival Brier;
6. when auxiliary heads are enabled, improves their mean log loss over
   train-frequency priors by at least 2% relative.

Eligible arms are selected by lower Brier, log loss, ECE, then stable arm name.

## Sealed Test

If validation selects no treatment, test and final data remain unopened.

If validation selects a treatment, compare it once with A0 on held-out
PatternPortfolio. Replication requires:

- at least 0.5% relative Brier improvement;
- paired game-bootstrap interval wholly below zero;
- log-loss regression no greater than 0.005;
- ECE regression no greater than 0.015;
- no binary survival-Brier regression.

Random final stress is descriptive and cannot reverse or create a pass.

## Cluster Allocation

Primary wave:

| Host | Arm |
|---|---|
| john1 | A0 public-state control |
| john2 | A1 recent history |
| john3 | A2 next-draft auxiliary |
| john4 | A3 joint intent and survival |

Rotated replay wave:

| Host | Arm |
|---|---|
| john2 | A0 |
| john3 | A1 |
| john4 | A2 |
| john1 | A3 |

## Consequences

A policy-held-out test pass authorizes a separate high-regret draft-ranking
integration experiment. It does not authorize gameplay promotion or a score
claim. A null closes these four information pathways on this corpus and graph
without invalidating the exact sequential data.

## Outcome

All eight primary and rotated-host runs completed with exact final model,
parameter, prediction-evidence, and scientific-identity parity.

A2, recent history plus next-draft auxiliary supervision, was selected:

- validation Brier improved 1.959% over A0, with paired 95% CI
  `[-0.014197, -0.010372]`;
- next-draft auxiliary NLL improved 11.675% over train-frequency priors;
- sealed PatternPortfolio Brier improved 1.824%, with paired 95% CI
  `[-0.013081, -0.009649]`;
- validation and sealed-test NLL and binary survival Brier improved;
- every preregistered gate passed.

The terminal classification is
`opponent_intent_policy_holdout_replication_passed`. A high-regret
draft-ranking integration experiment is authorized. Gameplay, score,
paid-wipe, strategy-switch, and champion-transfer claims remain unauthorized.

See
`docs/v2/reports/o1-opponent-intent-mlx-factorial-v1-result.md`.
