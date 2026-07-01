# ADR 0038: Public Beam-State Continuation Value

Status: accepted on 2026-06-12.

## Context

Fresh exact diagnostics isolate multi-turn continuation as real value, while
candidate breadth, category retention, and beam capacity are null or negative.
The remaining search error is the scalar continuation-state evaluator.

Prior MLX policy targets failed because sample seeds included hidden game
state unavailable to the public encoder. This experiment must estimate an
expected public value, not reproduce one hidden realization.

## Decision

Define `public-beam-state-value-v1`:

- input: existing hidden-state-safe observable action afterstate;
- target: focal terminal base score under final-five W2/B16 continuation,
  averaged over deterministic public redeterminizations derived only from the
  observable state hash and sample index;
- opponents: frozen pattern-aware K8+H6+B8+M4;
- target units: score points, with exact current base score retained for audit;
- model: Apple MLX shared entity encoder with one scalar continuation head.

Before dataset collection or training, run an observability probe on 32
final-five public decision groups. Evaluate every W2 root candidate with two
disjoint batches of eight public redeterminizations.

The frozen probe domain is train-split game indices `40000-40001`. Each game
uses the pattern-aware K8+H6+B8+M4 trajectory and records all four seats at
exactly 5, 4, 3, and 2 personal turns remaining, yielding 16 groups per game.
The root frontier uses only the currently visible market and never performs a
three-of-a-kind replacement or paid wipe. Such stochastic public transitions
may occur inside each redetermined continuation, but concealed refill order
cannot choose a recorded root action.

Reproduction:

```bash
target/release/cascadia-v2 public-beam-value-probe \
  --output artifacts/datasets/public-beam-state-value-observability-v1 \
  --first-game-index 40000 \
  --games 2 \
  --resume \
  --report docs/v2/reports/public-beam-state-value-observability-v1-r8x2-b16-w2.json
```

## Probe Gates

The target advances only if disjoint R8 batches achieve:

- candidate-value correlation at least 0.60;
- within-group centered-advantage correlation at least 0.50;
- top-action agreement at least 50%;
- mean top-action regret at most 0.50 points;
- no hidden refill, stack order, or hidden seed in records or sample seeds;
- deterministic replay, exact group integrity, and checksummed output.

Failure closes this teacher before MLX compute. Passing authorizes a separately
frozen train/validation/test collection and training protocol. No threshold,
sample count, beam width, frontier, or encoder tuning may use probe results.

## Required Implementation

- versioned Rust dataset schema and atomic one-game shards;
- public-state-hash redetermination seeds with explicit domain separation;
- exact current-score and terminal-target audit fields;
- Rust/Python round-trip and checksum validation;
- MLX GPU smoke only after the probe passes;
- complete typed CLI provenance.

## Result

The frozen two-game probe produced 32 groups and 586 candidates in
1,698.895 seconds. Both Rust and Python independently validated the two atomic
shards and their checksums.

| Metric | Result | Gate |
|---|---:|---:|
| Candidate-value correlation | 0.9914 | >= 0.60 |
| Centered-advantage correlation | 0.9365 | >= 0.50 |
| Top-action agreement | 65.625% | >= 50% |
| Mean top-action regret | 0.1133 | <= 0.50 |

Mean within-group value range was 3.3965 points, substantially larger than the
0.3018 mean absolute centered batch difference. The target is repeatable and
decision-discriminative. MLX continuation-value training is authorized under
the separately frozen ADR 0039 protocol.
