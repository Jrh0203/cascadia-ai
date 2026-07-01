# ADR 0063: Exact MLX Habitat-Candidate Semantics

Status: rejected on 2026-06-12. Exact H6 root union is closed as the next
strength lever for the qualified value model.

## Context

ADR 0061 rejected doubling rollout budget, and ADR 0062 rejected retaining 64
generic root actions. K64 scored only +0.083 and lost 1.500 habitat, showing
that more of the existing proposal distribution is not the next lever.

The canonical V2 engine can identify actions that maximize real matching
terrain edges and habitat cohesion while retaining distinct draft-and-tile
placements. The exact MLX teacher does not explicitly guarantee those
actions. This experiment tests a semantic candidate change while keeping the
qualified value function, K32 capacity, R600 budget, prefilter, allocator,
opponents, and arithmetic fixed.

## Decision

Compare:

- baseline exact MLX K32/R600;
- treatment exact MLX K32/R600 with up to six canonical V2 habitat-cohesion
  actions unioned into the root frontier before the unchanged MLX prefilter;
- exact typed conversion from canonical `TurnAction` to the isolated legacy
  `ScoredMove`, with round-trip and translated-execution tests;
- duplicate actions removed before prefiltering;
- independent warmed services and paired symmetric games.

The report must record generated, novel, retained, and selected habitat
candidates so a null result can distinguish proposal overlap from value
rejection.

## Frozen Protocol

- Smoke: seed 34,299.
- Pilot: seeds 34,300-34,302.
- Both arms: K32, R600, LMR enabled, diverse prefilter enabled.
- Treatment semantic frontier: H6.
- Runtime ceiling: 240 seconds/game for each arm.
- Full bridge, candidate-source, batch, score-category, latency, provenance,
  and shutdown evidence required.

The smoke passes if both arms complete 80 legal decisions with zero fallback,
finite exact-width responses, clean shutdown, and the runtime ceiling.

The pilot is promising only if:

- H6-minus-baseline mean is at least +0.50;
- H6 absolute mean is at least 95.50;
- wildlife delta is at least -0.50;
- habitat delta is at least +0.25;
- Nature Token delta is at least -1.00;
- all integrity and runtime gates pass.

Passing authorizes a larger confirmation. Failure closes this exact H6 union
and redirects research from candidate semantics to value representation or
multi-turn planning.

## Maximum Compute

One seed-34,299 smoke followed, only if it passes, by three paired games on
34,300-34,302. No training, sweep, validation/test split, or promotion.

## Result

An initial implementation-only smoke opened seed 34,299, but its H6 frontier
contained legal same-slot independent drafts that the legacy bridge cannot
represent. Those actions spend a Nature Token for the same market pair
available through the free paired action and are therefore strictly dominated.
The smoke and the interrupted first pilot process are invalid implementation
evidence, not gameplay evidence.

ADR 0064 preserves the official legality of same-slot independent drafting and
removes the dominated action from every ranked strategy frontier. The invalid
smoke is retained immutably as
`exact-mlx-habitat-candidate-h6-invalid-prefilter-smoke-1.json`, BLAKE3
`64341891d1486f9242e36aee503b5e483120556831b9bdb665e0ce25f15c50c7`.
The corrected implementation reran the frozen smoke and pilot seeds without
changing any gate, budget, or treatment parameter.

The corrected smoke passed:

- K32 mean 96.000;
- K32+H6 mean 96.250;
- paired gain +0.250;
- wildlife +2.000, habitat -0.750, Nature Tokens -1.000;
- nine novel habitat candidates selected;
- 156.92 baseline and 153.72 treatment seconds/game;
- zero bridge or neural fallback and clean shutdown.

The three-game pilot rejected the treatment:

- K32 mean 96.167;
- K32+H6 mean 96.500;
- paired gain +0.333 versus the required +0.500;
- 95% CI `[-1.485,+2.152]`;
- record 2-0-1, with paired deltas +1.50, +1.00, and -1.50;
- wildlife +0.083, habitat +0.250, Nature Tokens +0.000;
- 1,440 generated, 873 novel, 645 retained, and 27 selected habitat
  candidates;
- 151.83 baseline and 150.27 treatment seconds/game;
- all 480 actions legal, zero fallback, and clean shutdown.

H6 passed the absolute-mean, wildlife, habitat, token, integrity, and runtime
gates. It failed only the preregistered paired-gain gate. The mechanism is real
but too small and unstable to justify a confirmation. No larger H6 experiment
is authorized; the next strength experiment must change value representation
or multi-turn planning.

The corrected pilot report is
`docs/v2/reports/exact-mlx-habitat-candidate-h6-pilot3.json`, BLAKE3
`90f9438d119eefad4d01d17889341dec7f729aa75546c6737200be6bfdfe5c65`.
The corrected smoke report BLAKE3 is
`ab91c43acefaf35af08c65f70e909c42044f4300136ad47095fcbb76b0881c13`.
