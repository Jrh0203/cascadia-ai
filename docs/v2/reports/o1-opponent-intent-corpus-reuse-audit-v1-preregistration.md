# O1 Opponent-Intent Corpus Reuse Audit v1 Preregistration

**Frozen before full-corpus execution:** 2026-06-16  
**Experiment ID:** `o1-opponent-intent-corpus-reuse-audit-v1`  
**Plan section:** O1, opponent intent and future-access windows

## Question

Can the existing canonical action-imitation trajectories be converted into
exact sequential next-pick and market-survival supervision, despite compact
records omitting physical tile IDs?

## Hypotheses

**H1:** Split seed plus game index and the selected compact actions reproduce
every original game state and candidate action exactly.

**H2:** Deterministic replay recovers unique tile identity and therefore exact
post-action tile-survival labels across all three intervening opponent turns.

**H3:** Train and validation are disjoint even though both begin at numeric game
index 50,000, because split identity participates in seed and group-ID
derivation.

**H4:** Snapshot-only semantic matching is insufficient in a measurable number
of positions because distinct tile IDs share semantic features.

**H5:** The corpus is useful for same-policy O1 foundation work but cannot
satisfy policy-held-out O1 success criteria.

## Frozen Inputs

| Split | Dataset ID | Games | Groups | Candidates | Manifest BLAKE3 |
|---|---|---:|---:|---:|---|
| Train | `canonical-action-imitation-train-a0155b3613e51112` | 64 | 5,120 | 327,680 | `abdf4f01ac8f5673d2de5fcaae8ea7e4edeaf8e0869822965e4e9d1d10690693` |
| Validation | `canonical-action-imitation-validation-4929d2a8a2bb0a0d` | 16 | 1,280 | 81,920 | `606238e7fe1ec0fb57f1102722627d0c563f8df29e0a42cf1e6bf41438451a77` |

Both datasets use teacher strategy
`canonical-action-legacy-heuristic-v1-k32-r600-lmr-no-paid-prelude` and weights
BLAKE3
`9e1d568693274fc537ac4f6d6f729abb1ee8da8330a78d1f78a1f62b733de400`.

No test or final split is opened for this audit.

## Exact Procedure

The source-frozen Rust executable
`opponent_intent_reuse_audit` performs shard-parallel replay.

For each candidate group:

1. Rebuild the current `GameState` from the declared split and game index.
2. Check turn, seat, and all 864 compact position bytes.
3. Reconstruct all 64 retained actions through the rules engine.
4. Serialize each reconstructed `TurnAction` and reproduce its stored BLAKE3.
5. Identify the sole selected candidate from its binary target.
6. Record exact market tile IDs, public wildlife species, semantic collision
   counts, draft slots, independent-draft use, free replacement use, and
   wildlife placement.
7. Apply the selected action and continue until terminal.

For each focal turn `t = 0..75`, derive a post-action window from the market at
`t + 1`. Track all four tile IDs after opponent one, opponent two, opponent
three, and at next focal access `t + 4`. Also record which opponent consumed an
item and whether an exactly surviving tile retains the same public wildlife
species.

## Primary Metrics

- exact position-byte checks / expected positions;
- exact candidate-action hashes / expected candidates;
- exact selected labels / expected positions;
- exact transitions and terminal games;
- positions with four unique tile IDs;
- positions and occurrences with duplicate tile semantics;
- positions and occurrences with duplicate full pair semantics;
- post-action windows and tile-level survival labels;
- survival and consumption rates by opponent offset;
- train/validation overlap counts for group IDs, compact positions, public
  states, and initial hidden states.

## Pass Gates

Foundation reuse passes only if:

- 80 of 80 turns replay for every game;
- every compact position is byte-identical;
- every candidate action hash is exact;
- every game reaches terminal state;
- every active market contains four unique tile IDs;
- every cross-split overlap count is zero;
- john4 primary and john2 replay produce identical scientific BLAKE3 digests.

Any single exact mismatch rejects reuse. There is no tolerance and no repaired
row path.

## Interpretation

A pass classifies the corpus as
`exact_replay_foundation_reusable_policy_holdout_required`.

It authorizes:

- exact same-policy next-pick labels;
- exact selected tile identity and draft-slot labels;
- exact post-action tile-survival labels;
- public recent-draft histories derived from the sequence;
- a cheap O1 representation and learnability pilot.

It does not authorize:

- final O1 training or promotion;
- claims about unseen opponent policies;
- checkpoint-identity shortcut resistance;
- strategy-switch prediction;
- gameplay or score improvement.

The mandatory successor is a multi-policy sequential corpus with policy family
held out by split and policy/checkpoint identity retained only as provenance.

## Cluster Allocation

| Host | Role |
|---|---|
| john1 | Build and fan out immutable bundle and datasets; collect reports |
| john2 | Independent exact replay |
| john3 | Remains on the P1 wildlife critical path |
| john4 | Primary full-corpus audit |

The audit is CPU-parallel across complete-game shards and uses no MLX or
external compute.

