# ADR 0059: Exact MLX Rollout-Wave Integration

Status: passed on 2026-06-12. A separately preregistered gameplay runtime smoke
and paired strength pilot are authorized.

## Context

ADR 0057 separated search from its evaluator and proved the new batched search
path exactly reproduces the unchanged native path. Its standard-MLX treatment
failed because tiny arithmetic differences changed strict near ties and the
variable-row protocol was slow. ADR 0058 independently removed both causes:
request type 6 is bit-identical to Rust and delivers 75,176 batch-32
evaluations per second end to end.

The remaining question is purely integrational: can that exact operation
replace every native neural forward inside the qualified search while
preserving the complete stochastic computation and meeting an interactive
runtime bound?

## Decision

Reuse ADR 0057's evaluator-independent implementation without changing search
semantics:

- root prefilter, diverse placement variants, LMR priors, and all acting-player
  rollout decisions use the same stable batched row order;
- greedy opponent turns, rules, features, RNG, integer score accumulation,
  sequential-halving allocation, elimination, and final ordering remain in
  Rust;
- the treatment evaluator calls only
  `ModelProcess::predict_sparse_nnue_csr_exact`;
- no native neural result or floating-point tie tolerance is permitted on the
  treatment path;
- exact-width, finite-response, fallback, batch, row, and rollout-wave
  diagnostics remain mandatory;
- one-decision smoke runs may stop early and are never qualifying evidence.

## Frozen Protocol

- Model, fixture, and operation: passed ADR 0055 and ADR 0058 artifacts.
- Environment: `MCE_LMR=1`, `MCE_DIVERSE_PREFILTER=1`; incompatible historical
  experiment variables absent.
- Rules: four-player AAAAA, no habitat bonuses, canonical V2 bridge.
- Development trajectory: train split game index 92,100 using pattern-aware
  canonical actions.
- Full trajectory: R32 at all 80 decisions.
- Qualified spots: R600 at decisions 0, 39, and 79.
- Reference: unchanged native NNUE prefilter and historical sequential
  halving.
- Treatment: unchanged ADR 0057 batch search plus exact CSR request type 6.
- Search RNG: exact qualified public-state digest derivation.
- Runtime: complete post-warmup decision wall time, serially measured.

Every gate must pass:

- native evaluator through the batch abstraction remains exact;
- all root candidate identities match;
- every R32 selected action, sample count, and rollout mean matches exactly;
- repeated exact-MLX treatment is bit-identical;
- every R600 spot selected action, sample count, and rollout mean matches
  exactly;
- zero fallback, native treatment forward, bridge failure, illegality, or
  non-finite/width error;
- exact-MLX R32 wall time is no more than 1.5x native;
- the canonical trajectory contains exactly 80 decisions;
- service shutdown is clean.

Passing authorizes a separately preregistered gameplay runtime smoke and paired
strength pilot. It does not itself promote a strategy or open held-out test
seeds.

## Maximum Compute

One decision-0 R32 implementation smoke, one full 80-decision R32 parity run,
and R600 checks at decisions 0, 39, and 79. No training, gameplay, validation,
test split, hyperparameter sweep, or promotion.

## Result

Passed every frozen gate.

Across the complete 80-decision trajectory:

- the native evaluator through the batch abstraction remained exact;
- all 2,494 R32 estimates, candidate sets, selected actions, and sample counts
  matched exactly;
- the repeated MLX treatment matched bit for bit;
- all 87 estimates at the three R600 spots matched exactly;
- there were zero fallbacks over 4,030 R32 neural batches and 1,726,630 rows;
- MLX took 47.783 seconds versus native at 44.550 seconds, a 1.073x ratio;
- the service shut down cleanly.

The one-decision smoke also passed every applicable gate at 1.154x native.
The qualifying report is
`docs/v2/reports/legacy-nnue-v4opp-mlx-exact-rollout-wave-v1.json`, BLAKE3
`ad17c43f0e55006ca16deb141fbafe3b28c219d98af7848877967dfbe41c75d7`.

The qualified historical policy can now run with every neural forward on MLX
without changing its behavior. This establishes an exact local Apple-neural
baseline; it does not establish new playing strength or promote historical
weights as the final solution.
