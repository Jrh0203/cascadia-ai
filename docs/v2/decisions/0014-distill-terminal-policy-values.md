# ADR 0014: Distill Terminal Policy Values

Status: rejected before training on 2026-06-11.

## Context

R8 terminal policy improvement qualified as a research teacher at 94.833 mean,
+1.333 paired versus pattern-aware, +1.750 Bear, +0.333 total wildlife, and
+1.417 habitat. Its roughly 186-second runtime makes direct product use
unacceptable.

The current entity ranker already represents all four boards, current market,
phase, Nature Tokens, wildlife counts, and habitat sizes. Earlier H6
distillation was limited by four-ply exact-score labels. R8 terminal labels
replace that short target while keeping the same public afterstate boundary.

## Decision

Collect:

- 64 terminal R8 train games, split indices 0 through 63;
- 16 terminal R8 validation games, split indices 0 through 15;
- one game per atomic, checksummed, resumable shard;
- complete K8+H6+B8 candidate groups at all 80 decisions.

Train `entity-set-ranker-v1` from scratch on MLX with its frozen default
architecture: hidden width 96, four heads, two board blocks, one market block,
and feed-forward multiplier three. Use AdamW at `1e-4`, weight decay `1e-4`,
group batch 16, at most 20 epochs, and patience five.

## Gates

Before gameplay, the best checkpoint must:

- improve selection loss over initialization;
- achieve validation top-one regret at most 0.75;
- achieve pairwise accuracy at least 0.65;
- achieve value-difference correlation at least 0.30;
- achieve top-one accuracy at least 0.45.

Only then may a ten-game paired pilot use seeds 25100 through 25109. It requires
+0.5 score, at least -0.5 wildlife, at least -0.5 habitat, at least -1.0 Nature
Tokens, and at most two treatment seconds per game. A passing pilot alone may
advance to 50 disjoint games on seeds 25200 through 25249.

## Outcome

Collection completed all 64 training games and nine of 16 validation games
before a representation audit found that `PositionRecord::afterstate` called
the complete game transition. That transition refilled the drafted market slot
from the real hidden stack. The R8 teacher target averaged fair
redeterminizations, while the model feature encoded one inaccessible actual
refill. Runtime inference used the same leak.

The process was stopped immediately. No model was trained, promoted, or
evaluated. The complete train split and partial validation split are preserved
under `artifacts/datasets/rejected/ranking-terminal-r8-hidden-refill-leak-*`.
ADR 0015 replaces the boundary and bumps the feature schema before any
terminal-label collection resumes.
