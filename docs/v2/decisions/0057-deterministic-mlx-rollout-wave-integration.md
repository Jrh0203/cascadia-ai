# ADR 0057: Deterministic MLX Rollout-Wave Integration

Status: failed on 2026-06-12 during the authorized implementation smoke. No
gameplay or held-out domain was opened.

## Context

ADR 0055 proved exact MLX parameter reuse. ADR 0056 proved a typed long-lived
sparse service at 7,589 batch-32 evaluations per second. The qualified teacher
still calls native Rust NNUE evaluation in four places:

1. expanded-root prefilter scoring;
2. diverse wildlife-placement variant scoring;
3. LMR afterstate priors;
4. every future acting-player move inside every rollout.

Replacing those calls one at a time over a pipe would destroy batching and
could alter search ordering. The correct integration is to retain independent
rollout states but advance them in deterministic waves. At each acting-player
wave, Rust prepares every candidate afterstate in stable rollout/candidate
order, sends one sparse batch to MLX, partitions the returned values back into
their original groups, and continues each rollout.

## Decision

Introduce an evaluator-independent batch-search path in the historical AI
crate and keep the existing native path as the reference:

- factor candidate afterstate preparation from scalar NNUE evaluation;
- batch root prefilter, diverse variants, and LMR priors without changing
  candidate identity or tie-breaking;
- initialize rollout states from the exact existing work-item and seed order;
- advance greedy opponent turns on CPU;
- prepare all acting-player candidate groups in stable indexed order;
- call one `SparseNnueEvaluator` batch per wave;
- restore predictions to their exact groups and apply the existing strict
  first-winner tie behavior;
- preserve sequential-halving rounds, LMR multipliers, integer score
  accumulation, sample counts, elimination, and final sort;
- expose diagnostics for neural rows, batches, batch-size distribution,
  rollout waves, fallbacks, and service failures;
- implement the evaluator with ADR 0056's `ModelProcess`.

The batch path must explicitly reject search options it does not implement
rather than silently approximating them. This experiment freezes the qualified
environment: greedy opponents, NNUE acting-player policy, no expectimax,
control variates, truncation, decoupled opponents, ensemble, strategy bias,
Gumbel, or alternate allocator.

No native neural result may be used after the MLX service is opened. Rust owns
rules, state transitions, feature extraction, candidate generation, score
bookkeeping, RNG, and search. MLX owns every neural forward.

## Frozen Protocol

- Model and service: passed ADR 0055 and ADR 0056 artifacts.
- Environment: `MCE_LMR=1`, `MCE_DIVERSE_PREFILTER=1`; all incompatible
  experimental environment variables absent.
- Rules: four-player AAAAA, no habitat bonuses, canonical V2 execution through
  the qualified bridge.
- Development trajectory: train split game index 92,100, pattern-aware
  canonical actions.
- Full-trajectory parity budget: R32 at all 80 decisions.
- Qualified-budget spot checks: R600 at decisions 0, 39, and 79 from that same
  trajectory.
- Native reference: unchanged `NNUENetwork` prefilter and
  `score_nnue_rollout_mce_seq_halving`.
- MLX treatment: deterministic batch path plus the passed sparse service.
- Search RNG domains and seeds: derived exactly as the qualified teacher does
  from each translated public-state digest.
- Runtime measurement: wall time around each complete decision after one
  service warmup; native and MLX run serially on the same machine.

Every gate must pass:

- native evaluator through the new batch abstraction reproduces the unchanged
  native path exactly before MLX is involved;
- root expanded, legal, and retained candidate identities match exactly;
- sequential-halving sample counts match exactly for every retained action;
- R32 selected action matches at all 80 decisions;
- R32 rollout-mean maximum absolute difference at most `0.05` points and mean
  absolute difference at most `0.01`;
- all three R600 spot-check selected actions match;
- R600 rollout-mean maximum absolute difference at most `0.05` points and mean
  absolute difference at most `0.01`;
- zero neural fallbacks and zero native forwards on the MLX treatment path;
- every service output is finite and every returned width matches;
- repeated MLX treatment runs are bit-identical;
- MLX treatment total R32 decision time no more than 2.0x native;
- no canonical action illegality, bridge fallback, score mismatch, test split,
  gameplay seed, or promotion.

Passing authorizes a separately preregistered gameplay runtime smoke and
paired strength pilot. Failure closes the implementation before gameplay.

## Maximum Compute

One implementation smoke, one 80-decision native/native R32 parity pass, one
80-decision native/MLX R32 pass, and three R600 train-domain spot checks. No
training, external compute, test split, gameplay benchmark, hyperparameter
sweep, or promotion is authorized.

## Result

The evaluator-independent search refactor passed its strongest internal check:
the new batch abstraction with the native evaluator reproduced the unchanged
native path exactly over all 80 decisions. Candidate identities, selected
actions, sample counts, and 2,494 rollout estimates all matched bit for bit.

The ADR 0056 treatment failed before the frozen R32/R600 run was authorized:

- one selected action and seven sequential-halving sample allocations diverged;
- maximum rollout-mean error reached 3.4667 points after an early near-tie
  neural decision changed a rollout trajectory;
- the treatment took 462.05 seconds versus 76.43 seconds for native, a 6.05x
  ratio against the 2.0x gate;
- repeat treatment execution was deterministic and there were zero fallbacks.

The smoke used `--rollouts 1`, but the historical sequential-halving floor
allocates at least one sample per live candidate per round, so it still
executed 4,030 neural batches and 1,726,932 neural rows. The report is
`docs/v2/reports/legacy-nnue-v4opp-mlx-rollout-wave-v1-runtime-smoke-1.json`
with BLAKE3
`764d5e4662503c77d5f5b05a5b362f592d4a7d35adf072ff31e74ac7f9269b11`.

Root-cause probes found two independent defects in the attempted treatment:
standard MLX reductions differed from Rust by at most `4.196e-5`, enough to
flip strict near ties, and the variable-row protocol forced per-row parsing
and padded repacking. A separately preregistered operation must prove
Rust-order arithmetic and packed sparse transport before search integration is
retried. ADR 0057 is rejected; its exact native search refactor remains useful
test infrastructure but authorizes no gameplay claim.
