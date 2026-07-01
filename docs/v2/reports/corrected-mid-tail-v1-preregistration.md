# Corrected Historical Mid-Tail V1 Preregistration

Date: 2026-06-16

Experiment ID: `corrected-mid-tail-v1`

Schema ID: `legacy-mid-v4-fixed-v1`

Research-plan item: F5

Status: foundation implemented; training and gameplay not launched

## Question

Does replacing the historically dead 301-column adjacency prefix with the
intended extended tile-supply and overflow signals improve the qualified
historical NNUE without reducing strength?

The expected score effect is small. The primary purpose is to remove a proven
schema ambiguity, establish an exact checkpoint migration, and create a valid
control for later representation work.

## Upstream Evidence

The F1 activation census measured 200,000 balanced historical states and found:

- all 301 historical mid-tail columns inactive;
- the 369-column opponent-detail block active in every state;
- the existing market and supply blocks live; and
- no evidence that the intended 150 + 150 + 1 tail was present in the champion
  schema.

F1 final classification:

```text
feature_schema_activation_census_complete
```

Classification scientific BLAKE3:

```text
f7f8559431f53a461f9464e14ef4cee2119cf3ddcf0bf4e3dd9126ab8bdd91fb
```

Merged-census scientific BLAKE3:

```text
8906487b91aa0da25f388e2075d15150c8d1499a022cbf2d231987b37f182e65
```

## Frozen Schema

The treatment layout is:

| Block | Range | Width | Initialization |
|---|---|---:|---|
| Historical v2 base | `0..10561` | 10,561 | Exact copy |
| Opponent detail | `10561..10930` | 369 | Exact remap from `10862..11231` |
| Extended tile-terrain counts | `10930..11080` | 150 | Zero |
| Extended tile-wildlife capacity | `11080..11230` | 150 | Zero |
| Overflow used | `11230..11231` | 1 | Zero |

The historical defect at source range `10561..10862` has no destination and
is not reinterpreted.

The treatment weight container is `NNUC` version 1 with schema tag
`MIDTAIL-CORR-V1\0`. Historical controls remain `NNUE`.

## Foundation Acceptance Gates

All gates below must pass before any training job is queued:

1. `legacy-mid-v4-fixed-v1` compiles as a standalone cargo feature and through
   `cascadia-cli`.
2. Existing default and `mid-features,v4-opp` builds compile and retain their
   frozen feature counts and weight container.
3. Corrected extraction activates exactly five terrain-count rows and five
   wildlife-capacity rows for every bag context.
4. The overflow row activates if and only if overflow was used this turn.
5. Every corrected feature remains in `0..11231`.
6. The optimized parent-afterstate context exactly equals the general
   extractor over candidate afterstates.
7. Historical 11,231-row checkpoint migration copies all 10,561 base rows and
   all 369 opponent rows exactly.
8. Every corrected-tail weight is exactly positive or negative zero after
   migration; no random initialization is permitted.
9. Every non-first-layer tensor is byte-equivalent across migration.
10. Corrected save/load is exact for head-format versions 1 through 4.
11. Unknown historical widths and corrupted corrected headers fail closed.
12. The production champion checkpoint is migrated once into a content-hashed
    immutable artifact, then independently audited before training.
13. The corrected extractor and migrated checkpoint are ported to MLX with
    byte-for-byte first-layer row parity before Apple-cluster neural training.

Gates 12 and 13 are intentionally not executed in this foundation task because
the assigned ownership excludes checkpoint artifacts and MLX code.

## Controlled Arms

### Control C0: exact historical champion

```text
cargo features: mid-features,v4-opp
schema: historical legacy-mid-v4opp-11231
weights: nnue_weights_v4opp_modal_iter3.bin
extraction: exact historical adjacency-prefix defect retained
```

The control checkpoint must not be opened by a corrected extractor.

### Treatment T1: corrected migrated champion

```text
cargo features: legacy-mid-v4-fixed-v1
schema: legacy-mid-v4-fixed-v1
weights: deterministic migration of the C0 checkpoint
base rows: exact C0 copy
opponent rows: exact C0 remap
corrected tail: zero
```

Before fine-tuning, T1 must produce the same network output as C0 for every
state in which the historical accidental tail is inactive. The F1 corpus
predicts exact parity over all 200,000 measured states.

### Treatment T2+: fine-tuned corrected seeds

Only after T1 parity is proven, train multiple corrected seeds from the same
migrated checkpoint. No from-scratch arm is part of F5.

Frozen fine-tuning constraints:

- learning rate starts at or below `3e-5`;
- the historical finding that `1e-4` diverges during fine-tuning is binding;
- optimizer, batch size, epochs, sample order, labels, and opponent pool match
  the control comparison;
- no architecture, candidate generator, search budget, rollout policy, or
  scoring rule changes are mixed into F5;
- corrected rows may train immediately, while optional base-row freezing must
  be a separately named ablation; and
- all training on Apple hardware uses the validated MLX implementation.

## Data Contract

Use one frozen, content-hashed corpus for both arms.

Required state fields:

- exact sparse historical v2 base inputs;
- focal-relative ordered opponent detail;
- exact tile-bag terrain marginals;
- exact tile-bag wildlife-capacity marginals;
- overflow-used-this-turn;
- target and sample provenance; and
- phase, focal seat, and ruleset identity.

Forbidden inputs:

- hidden future bag order;
- future refill realization;
- future actions;
- terminal information unavailable at the represented decision;
- labels or teacher outputs inserted as features; and
- any repaired historical feature row in C0.

The initial parity corpus should be the F1 200,000-state historical dataset or
a deterministic export with identical state identities.

## Offline Measurements

Report for C0, T1, and each trained treatment:

- exact prediction parity before training;
- total validation RMSE;
- validation RMSE by opening, early, middle, and late phase;
- RMSE by focal seat;
- RMSE on overflow-used states;
- RMSE by low, medium, and high tile-supply buckets;
- corrected-block activation counts and channel coverage;
- first-layer gradient norm by base, opponent, terrain-tail, wildlife-tail,
  and overflow blocks;
- first-layer weight norm by the same blocks;
- fraction of corrected rows still exactly zero after each epoch;
- checkpoint schema ID and content hash; and
- wall time, examples per second, peak memory, and host identity.

The treatment is not accepted from aggregate RMSE alone. Improvements must
appear in the supply-sensitive subsets that the new features can explain.

## Gameplay Evaluation

Gameplay begins only after offline gates pass.

Use four-player head-to-head, Card A, no habitat bonuses, identical search
configuration, paired seeds, and balanced seats.

Primary comparison:

```text
C0 historical champion vs best preregistered corrected treatment
100 paired games
200 seat-games
```

Report:

- mean score and paired mean delta;
- standard error and 95 percent confidence interval;
- win rate;
- score by seat;
- wildlife, habitat, and nature-token components;
- Bear, Elk, Salmon, Hawk, and Fox subscores;
- overflow-use frequency;
- independent-draft and mulligan frequency; and
- runtime per decision and per game.

No search-budget increase may compensate for a weaker corrected network.

## Success Criteria

Foundation success:

- every foundation acceptance gate passes.

Scientific continuation gate:

- T1 is exactly prediction-equivalent to C0 over the frozen parity corpus;
- every corrected block has nonzero activation;
- at least one trained treatment improves held-out loss materially in
  supply-sensitive or overflow subsets; and
- the best treatment is noninferior in the 100-paired-game score comparison.

F5 is promoted only if the paired-score confidence interval excludes a
material regression. A small positive score delta may justify retaining the
schema even if it does not independently advance the 100-point frontier.

## Failure Classifications

Use one of these terminal classifications:

- `foundation_invalid_layout`: any range, width, or feature-bound failure;
- `foundation_invalid_migration`: any common row or downstream tensor differs;
- `foundation_invalid_parity`: T1 predictions differ before training where the
  historical defect is inactive;
- `foundation_invalid_mlx_port`: Rust and MLX rows or predictions differ;
- `corrected_tail_inactive`: any corrected block remains dead;
- `corrected_tail_no_offline_signal`: activation exists but held-out metrics
  do not improve;
- `corrected_tail_strength_regression`: gameplay mean regresses materially;
- `corrected_tail_null`: offline and gameplay effects are immaterial; or
- `corrected_tail_positive`: preregistered offline and gameplay gates pass.

An invalid foundation result is repaired and rerun under a new implementation
receipt. It is not reported as a negative research result.

## Future Cluster Allocation

After the MLX parity and immutable-checkpoint gates pass, use the four machines
for nonduplicative work:

| Host | Initial responsibility |
|---|---|
| john1 | C0 parity and frozen-control evaluation |
| john2 | Corrected seed A fine-tuning |
| john3 | Corrected seed B or base-freeze ablation |
| john4 | Corrected seed C or supply-sensitive diagnostics |

Once training finishes, all four hosts run disjoint paired-game seed shards.
No healthy host should repeat the same seed, checkpoint, or evaluation shard.
Merged results must verify disjoint seed ranges and exact checkpoint hashes.

## Foundation Commands

Focused corrected-schema tests:

```bash
cargo test -p cascadia-ai --lib \
  --features legacy-mid-v4-fixed-v1 \
  legacy_mid_v4_fixed_v1_tests \
  -- --test-threads=1
```

Extractor/context parity:

```bash
cargo test -p cascadia-ai --lib \
  --features legacy-mid-v4-fixed-v1 \
  mid_v4_parent_afterstate_context_matches_oracle_across_complete_games \
  -- --test-threads=1
```

Historical-layout regression:

```bash
cargo test -p cascadia-ai --lib \
  --features mid-features,v4-opp \
  historical_champion_layout_remains_frozen
```

Explicit checkpoint migration:

```bash
cargo run -p cascadia-ai \
  --example migrate_legacy_mid_v4_weights \
  --features legacy-mid-v4-fixed-v1 \
  -- nnue_weights_v4opp_modal_iter3.bin \
     nnue_weights_legacy_mid_v4_fixed_v1_init.bin
```

The final migration command is documented here but must not be run until its
output path, artifact manifest, content-hash receipt, and MLX parity owner are
assigned.

## Current Blockers Before Cluster Training

1. The production champion checkpoint has not yet been migrated into an
   immutable, content-hashed experiment artifact.
2. The corrected Rust schema has not yet been mirrored and parity-tested in
   MLX.
3. The frozen parity/training corpus and shard manifest have not yet been
   assigned to F5.
4. No cluster queue or dashboard entry exists for F5, by design of this
   foundation-only task.

Until these four items close, no F5 training or gameplay job is valid.

