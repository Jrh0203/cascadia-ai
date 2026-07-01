# ADR 0009: Search-Guided MLX Policy Iteration

Status: accepted for preregistered experiment on 2026-06-10.

## Context

The first H6 ranker learned only from states visited by H6 itself. Using that
model as a standalone apprentice changes the state distribution, so held-out
teacher-trajectory metrics do not measure the errors that compound during its
own games. Direct heuristic rollout substitution was also negative.

## Decision

Implement an Expert Iteration/DAgger-style local loop:

1. Freeze a promoted MLX apprentice.
2. Let that apprentice control all four seats.
3. At every visited state, have frozen H6 label the complete K8+H6 candidate
   set before the apprentice chooses from the same actions.
4. Aggregate those records with the original teacher-trajectory data.
5. Warm-start a new MLX run from the apprentice with a fresh optimizer.
6. Select checkpoints by mean listwise loss across apprentice-trajectory and
   original H6 validation distributions.
7. Require both held-out ranking improvement and paired gameplay improvement
   before promotion.

Dataset manifests bind each iteration to the exact apprentice model checksum.
The original warm-start checkpoint is eligible as best, so an iteration can
cleanly produce no model change rather than silently promote regression.

This loop remains local on Apple Silicon. Rust owns rules, search labels,
collection, and gameplay; MLX owns neural optimization and inference.

## Outcome

The complete local loop ran successfully: 64 apprentice-trajectory train games
and 16 validation games produced 6,400 groups and 81,104 candidate labels.
The warm-started aggregate run stopped after seven epochs at patience.

Epoch 4 was best. Balanced loss improved from 2.428105 to 2.425570,
apprentice top-one regret from 0.370703 to 0.352930, and apprentice pairwise
accuracy from 0.777293 to 0.779077. Original H6 validation also improved
slightly, so no forgetting occurred.

The apprentice regret gain was 0.017773, below the preregistered 0.03 gate.
The experiment was rejected before model promotion or gameplay. Keep the
iteration substrate as production research infrastructure; change the target
or representation before another round.
