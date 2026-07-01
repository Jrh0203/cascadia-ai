# ADR 0062: Exact MLX Root-Candidate Recall

Status: rejected on 2026-06-12. Generic K64 root retention is closed as the
next strength lever for this model and allocator.

## Context

ADR 0061 found that doubling rollout budget over the same K32 retained root
added only +0.167 points while doubling runtime. More precise resampling is
not the next lever. A distinct failure mode is prefilter recall: the qualified
teacher generates roughly 40 canonical expanded actions per decision but
retains at most 32, including diversity variants, before rollout search.

Historical breadth conclusions are untrusted. The next controlled test keeps
R600 fixed and changes only the retained root capacity from 32 to 64. The same
batched prefilter and diversity machinery still runs whenever the expanded
frontier exceeds 32; the treatment simply permits more scored actions to
survive.

## Decision

Compare:

- baseline exact MLX K32/R600;
- treatment exact MLX K64/R600;
- identical model, bridge, candidate generator, prefilter scores, diversity
  variants, rollout allocator, opponents, RNG, and tie behavior;
- independent warmed services and paired symmetric games.

## Frozen Protocol

- Smoke: seed 34,099.
- Pilot: seeds 34,100-34,102.
- Runtime ceiling: 240 seconds/game for each arm.
- Full bridge, batch, score-category, latency, and shutdown evidence required.

The smoke passes if both arms complete 80 legal decisions with zero fallback,
finite exact-width responses, clean shutdown, and the runtime ceiling.

The pilot is promising only if:

- K64-minus-K32 mean is at least +0.50;
- K64 absolute mean is at least 95.50;
- wildlife delta is at least -0.50;
- habitat delta is at least -0.50;
- Nature Token delta is at least -1.00;
- all integrity and runtime gates pass.

Passing authorizes a larger confirmation. Failure closes K32 root truncation
as the next strength bottleneck and redirects research to value quality or new
candidate semantics.

## Maximum Compute

One seed-34,099 smoke followed, only if it passes, by three paired games on
34,100-34,102. No training, sweep, validation/test split, or promotion.

## Result

The smoke passed every integrity and runtime gate:

- K32 mean 93.750;
- K64 mean 95.500;
- paired gain +1.750;
- K32 took 141.70 seconds/game;
- K64 took 146.53 seconds/game;
- both arms completed 80 legal selections with zero fallback.

The fresh three-game pilot rejected the smoke signal:

- K32 mean 96.583;
- K64 mean 96.667;
- paired gain +0.083 versus the required +0.500;
- 95% CI `[-5.151,+5.318]`;
- record 2-0-1, with paired deltas -5.25, +2.50, and +3.00;
- wildlife +0.333, habitat -1.500, Nature Tokens +1.250;
- K32 took 150.01 seconds/game;
- K64 took 155.51 seconds/game;
- both arms completed 240 legal selections with zero fallback and clean
  shutdown.

K64 passed its absolute-mean, wildlife, token, runtime, and integrity gates,
but failed the paired-gain and habitat gates. The result is not a simple
variance-reduction opportunity: retaining more actions changed allocation
materially and produced unstable game-level outcomes without improving total
score.

The pilot report is
`docs/v2/reports/exact-mlx-root-candidate-k32-k64-pilot3.json`, BLAKE3
`00bdd3b79663b03680c2a1c7fcb4911d2e9dd7ddeb3d696a7120206d18131c4c`.
The smoke report BLAKE3 is
`ebcc4943d0337bede774a15ea16a0487c8531ed480408df09cbfb0e09ffb2ef5`.

No larger K64 confirmation is authorized. The next experiment must change
candidate semantics, value quality, or planning structure rather than merely
retaining more of the same generic root frontier.
