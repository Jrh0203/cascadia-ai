# ADR 0185: Policy-Held-Out Sequential Corpus For O1

**Status:** Completed  
**Date:** 2026-06-17  
**Experiment:** `o1-opponent-intent-policy-heldout-corpus-v1`

## Context

ADR 0182 proved that the historical imitation trajectories can recover exact
physical tile survival and next-pick labels. That corpus contains one policy
family and one checkpoint, so it cannot measure generalization to unseen
opponent behavior.

The native v2 simulator already provides six deterministic public policies:
Random, Greedy, PatternAware, PatternCommitment, PatternCompetition, and
PatternPortfolio. It also supports mixed seat policies and exact pre-action
observation. These policies are not equally strong, but they provide distinct
draft, placement, commitment, competition, and portfolio behavior without
external compute.

## Decision

Create `opponent-intent-v1`, a fixed-width sequential corpus with one record for
each focal turn `0..75`.

Each record contains:

- compact public state after the focal action and refill, from the focal
  player's perspective;
- the previous 12 public actions with age and relative actor seat;
- the next actions of all three opponents;
- exact physical identity and survival disposition for all four market tiles;
- exact final scores for analysis;
- seat-policy identity only in provenance and target metadata.

`model_input_bytes` must zero the stored game index and exclude seat-policy
codes, target policy codes, survival outcomes, and final scores. Changing any
policy code or game index must leave model-input bytes unchanged.

## Frozen Policy Split

| Split | Games | Policy pool | Required held-out family |
|---|---:|---|---|
| Train part 0 | 512 | Greedy, PatternAware, PatternCommitment | none |
| Train part 1 | 512 | Greedy, PatternAware, PatternCommitment | none |
| Validation | 256 | train pool + PatternCompetition | PatternCompetition |
| Test | 256 | validation pool + PatternPortfolio | PatternPortfolio |
| Final stress | 128 | all six policies | Random |

The required family is deterministically assigned to at least one seat in
every held-out game. Remaining seats are sampled deterministically from the
declared pool, with mixed-policy play guaranteed whenever the pool has more
than one family.

Split, cohort ID, game index, and a named seed domain all participate in game
seed and seat-policy assignment.

## Cluster Allocation

- john2: train part 0, then test;
- john4: train part 1, then final Random stress;
- john1: held-out validation and campaign coordination;
- john3: remains dedicated to the active P1 wildlife critical path.

This keeps all four Macs doing nonduplicative work.

## Gates

Corpus foundation passes only if:

1. every game yields exactly 76 windows;
2. every shard checksum, header, game range, turn sequence, history age, and
   target-relative seat validates;
3. each held-out game contains its required family;
4. no train game contains PatternCompetition, PatternPortfolio, or Random;
5. policy and game identity are mechanically absent from model inputs;
6. fixed crossed-host calibration games reproduce byte-identical record
   payloads;
7. every next-action and survival class has nonzero support;
8. train, validation, test, and stress model-input hashes have zero exact
   overlap.

## Consequences

A pass authorizes an MLX O1 learnability factorial on policy-held-out
next-action and survival calibration. It does not authorize gameplay,
checkpoint promotion, or a score claim. Strategy-switch prediction remains a
separate successor because all policies in this corpus are stationary within a
game.
