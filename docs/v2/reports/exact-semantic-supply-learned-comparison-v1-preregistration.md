# Exact Semantic Supply Learned Comparison V1 Preregistration

Status: completed; frozen classification control failed; no promotion

Date: 2026-06-17

ADR: 0147

Experiment ID: `exact-semantic-supply-learned-comparison-v1`

Protocol ID: `s1-exact-semantic-supply-mlx-comparison-v1`

## Question

On the same open complete-action train and validation decisions, does exact
public semantic supply improve calibrated action ranking beyond the frozen
30-value marginal control, and is explicit candidate-to-supply/frontier
attention required to realize that information?

## Hypotheses

### H0: Legacy marginals are sufficient

C0, T1, and T2 remain equivalent within the frozen value, ranking, and regret
gates. The factual collision exists but is too rare or irrelevant to improve
the current complete-action target.

### H1: Exact counts are independently useful

T1 decodes the exact refill law and improves learned validation behavior while
remaining value-calibrated, even without candidate-specific supply attention.

### H2: Relational access is required

T1 proves exact information is present but does not clear action-ranking gates.
T2 clears the top-64 recall, retained-regret, low-supply, and independent-draft
gates because its complete-action query can relate a selected tile and public
frontier to the exact remaining supply.

### H3: Exact supply is too expensive

T1 or T2 improves quality but fails absolute or C0-relative throughput/memory
gates. Such an arm is not accepted by this comparison.

## Frozen Evidence

Only the existing open splits are admitted:

| Split | Complete decisions | Complete legal actions |
|---|---:|---:|
| Train | 560 | 2,135,111 |
| Validation | 240 | 860,203 |

Both must be the existing
`complete-action-graded-oracle-v1` datasets, with unchanged public-state hashes,
action hashes, R600/R1200/R4800 targets, selected winners, and complete legal
sets.

No sealed/test split, alternate teacher, gameplay row, newly generated
example, replay-derived example, high-score cache, or external dataset is
permitted.

## Frozen Arms

### C0: `c0-legacy-marginals`

- Exactly five public wildlife counts and 25 public tile marginals.
- The 30 normalized values remain in their historical field order.
- The fixed 83-wide input has zeros in every unavailable exact slot.
- The fixed 80-token input has no additional state-dependent facts.
- Selected archetype and frontier-relation inputs are zero/state-independent.
- The exact refill target is retained only as a decoding diagnostic.

### T1: `t1-exact-counts`

- Five exact public wildlife counts.
- 75 exact public semantic-archetype counts.
- Exact unseen, drawable, and hidden-exclusion totals.
- Exact one-draw refill target.
- No selected-archetype or frontier relation.
- Candidate supply attention remains parent-state-only.

### T2: `t2-relational-supply`

- Every T1 fact.
- Rust-authored selected public archetype.
- Rust-authored six-edge public frontier requirements and rotation
  compatibility.
- Complete-action queries attend the exact supply/refill token set.

No arm may receive a different board, market, action, prior, target, loss,
initialization seed, optimizer, batch, epoch, D6, or checkpoint schedule.

## Frozen Capacity

Every arm uses
`s1-exact-supply-iso-complete-action-v1` with:

```text
hidden=128
heads=4
board_blocks=2
market_blocks=1
supply_blocks=1
feed_forward_multiplier=3
supply_vector_width=83
supply_token_count=80
supply_token_width=32
trainable_parameters=3,073,101
```

The complete trainable parameter layout hash must be identical across arms.
Fresh models built after `mx.random.seed(2026061707)` must also have identical
initial weight fingerprints. A count or layout mismatch invalidates the
experiment before training.

## Frozen Normalization

```text
legacy wildlife / 20
legacy tile marginals / 81
exact wildlife / 20
exact archetype counts / 2
unseen / 81
drawable / 79
excluded / 2
refill target_i = archetype_count_i / unseen
C0 unavailable exact slots = 0
```

Normalization is recorded in authorization, control lock, arm report, and
aggregate identity. It may not be inferred from observed validation behavior
or changed after any optimizer step.

## Factual Collision Test

Every accepted cache must contain this ADR 0143 witness:

| Side | Physical tile IDs | Semantic archetype IDs |
|---|---|---|
| Left | `[0, 23]` | `[26, 72]` |
| Right | `[2, 20]` | `[24, 74]` |

Required properties:

- the complete 30-value legacy vectors are equal;
- C0 normalized supply vectors are byte-equal;
- exact count vectors differ;
- exact refill numerator vectors differ;
- T1/T2 normalized supply vectors differ; and
- the witness identity is content-addressed.

This test is factual and is not a learned success gate.

## Frozen Training

```text
seed=2026061707
optimizer=AdamW
epochs=30
group_batch_size=64
maximum_actions_per_batch=8192
maximum_group_actions=16384
learning_rate=0.0001
weight_decay=0.0001
checkpoint_steps=250
validation_patience=6
augmentation=uniform full D6 per complete decision group
warm_start=false
additional_train_data=false
```

The complete-action objective is:

```text
L =
    r1200_huber
  + 4.0 * r4800_huber
  + 0.5 * r1200_listwise
  + r4800_winner
  + 0.1 * standard_error_calibration
  + 0.01 * screen_only_regularization
  + 0.25 * refill_cross_entropy
```

Checkpoint selection minimizes validation
`mean_top64_retained_r4800_regret`; top-64 winner recall and R4800 value MAE
remain secondary recorded metrics.

Resume is permitted only from an atomic checkpoint whose immutable control
lock is unchanged. No failed arm may restart from a new seed or receive a
second production run under this preregistration.

## Data And Information Boundary

Rust is authoritative for:

- the 75-archetype catalog and catalog hash;
- exact public counts and drawable/exclusion state;
- `CSSSUP1` canonical bytes;
- selected archetype identity;
- frontier requirements and rotation compatibility;
- collision witness; and
- public-state/action-hash binding.

Python may normalize, batch, apply the frozen D6 contract, train MLX, evaluate,
benchmark, and classify. It may not derive alternative tile semantics.

The cache and every report must state:

```text
hidden_stack_order_read=false
hidden_wildlife_order_read=false
excluded_tile_identities_read=false
future_refills_read=false
sealed_test_opened=false
gameplay_opened=false
```

## Evaluation Protocol

Evaluate the selected checkpoint on all 240 open validation decisions and all
860,203 legal actions exactly once.

Record:

- training objective;
- R4800 count, MAE, RMSE, bias, correlation, calibration slope, intercept;
- top-1, top-8, top-32, top-64 R4800 winner recall;
- mean retained R4800 regret at those widths;
- top-64 95% confidence-set coverage;
- refill mean and p99 total variation;
- refill cross entropy and probability MAE;
- refill mode accuracy and mean fidelity;
- low-supply top-64 recall, confidence coverage, and retained regret;
- independent-draft-winner top-64 recall, confidence coverage, and regret;
- action scores per second;
- mean and p99 decision milliseconds;
- MLX peak active memory;
- process peak RSS;
- process swaps; and
- system swap delta.

Low supply means `unseen <= 20`. An independent-draft winner is a decision
whose frozen R4800 selected action has independent draft kind.

## Success Gates

### Evidence validity

- Exactly one report from C0/john1, T1/john2, and T2/john3.
- One independent replay from john4.
- Matching experiment, protocol, ADR, bundle, authorization, cache,
  normalization, collision, datasets, D6 contract, parameter count, and
  parameter layout.
- Full validation coverage and finite outputs.
- Report content addresses reconstruct from actual report fields.
- Forward and reverse aggregate bytes are identical.

Any failure yields
`exact_supply_learned_comparison_invalid_evidence`.

### Control viability

C0 must achieve:

```text
throughput >= 20,000 action scores/second
p99 <= 250 ms
MLX active memory <= 4 GiB
RSS <= 4 GiB
process swaps == 0
system swap delta <= 0
```

Failure yields `exact_supply_learned_comparison_control_failed`.

### Exact representation viability

For both T1 and T2 versus C0:

```text
R4800 MAE delta <= 0.05
R4800 RMSE delta <= 0.05
calibration slope-error delta <= 0.05
calibration absolute-intercept delta <= 0.25
```

For both T1 and T2:

```text
refill mean fidelity >= 0.9999
all refill probabilities finite
all absolute performance gates pass
```

Failure yields `exact_supply_learned_comparison_exact_representation_failed`.

### Relational success

T2 versus C0 must achieve:

```text
top64 winner recall delta >= 0.02
mean top64 retained-regret reduction >= 0.01
top64 confidence-set coverage >= 0.995
low-supply top64 recall delta >= 0.02
independent-draft-winner top64 recall delta >= 0.02
throughput fraction >= 0.60
MLX active-memory multiplier <= 1.50
RSS multiplier <= 1.50
```

Together with every prior gate and a passing john4 replay, this yields
`exact_supply_learned_comparison_relational_success`.

If exact representation viability passes but one relational gate does not, the
result is `exact_supply_learned_comparison_relational_null`.

## Independent Replay

John4 does not duplicate training. It independently:

- binds every train and validation group to the Rust cache;
- checks all 2,995,314 complete legal action hashes;
- verifies all 12 D6 inverse round trips;
- proves supply and frontier facts are D6-invariant;
- verifies the exact 3,073,101 parameter count;
- verifies identical parameter-layout hashes;
- verifies identical seeded initial weight fingerprints;
- verifies the hidden-information flags; and
- records that no optimizer, gameplay, or sealed/test evaluation ran.

## Four-Host Allocation

| Host | Role |
|---|---|
| john1 | C0 training, collection, deterministic classification |
| john2 | T1 training |
| john3 | T2 training |
| john4 | Independent replay/control |

The generated 17-task graph is inert. The implementation cannot apply it to
the live queue and does not edit the dashboard or research ledger.

## Frozen Result

The graph completed with a passing john4 replay and byte-identical forward and
reverse classification. The frozen classification is
`exact_supply_learned_comparison_control_failed`; no model, gameplay run, or
progress-to-100 claim is authorized. See
`docs/v2/reports/exact-semantic-supply-learned-comparison-v1-result.md`.

## Stop Rules

Stop and classify invalid without training if:

- the complete cache is absent or incomplete;
- a dataset identity or action hash differs;
- the collision witness differs;
- the parameter count/layout differs;
- seeded initialization differs;
- normalization differs;
- a host assignment differs;
- any Python task lacks `-B`;
- a preflight fails; or
- authorization is absent or stale.

After valid launch, do not:

- change a threshold;
- retry a seed;
- add epochs;
- modify patience;
- alter the model;
- warm start;
- add data;
- open sealed/test;
- run gameplay;
- promote a checkpoint; or
- install another queue task under this experiment ID.

## Claims Boundary

This experiment may establish offline learned value, complete-action ranking,
refill-law decoding, and local MLX performance on the open validation split.

It cannot establish gameplay strength, paired score improvement, model
promotion, production readiness, or progress toward the 100-point objective.

## Authorized Production Identity

The parent review, immutable bundle, complete cache, authorization, and inert
queue review completed on 2026-06-17.

```text
bundle:
  2baae4acb5a5375e056ae56e019180e57b98a5d032a7c5825357c93e6d2bf23c
authorization:
  954d0bb2e1bb1d8dca32cf9109f4d21c2525c4664c3344ef51b4435f11e0afef
cache:
  2323ead43b1bff7a506ecef4b8bd4793cebe4d53c6f8940b03404573ca5e6c15
cache manifest BLAKE3:
  a99d1aad79be950eb030fc56a3340205031a7ed0ebabe9980d7b49e3584b16c1
exporter BLAKE3:
  b1cb47bd848632597414b6636b05c6697291b3db19f2df923d26484d57476a84
collision witness:
  b860814dfe1c16ca9f4c17f574b7d0040ab684ed1bfbcb1fe262395ec84af447
parameter layout:
  f3d723afd7b938d01137b6587d98a3abf7b37217507ebc152b4d1d18413bbd2d
queue task graph BLAKE3:
  200cfb6fc1c241cb919a6b6ea01a3247724343e2db0a0c873d7fce96d7849e5d
```

The cache covers exactly 560 train decisions with 2,135,111 legal actions and
240 validation decisions with 860,203 legal actions. The seventeen-task live
graph remains dependent on successful preflight on john1 through john4 before
any optimizer or replay task becomes ready.
