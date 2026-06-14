# ADR 0074: Counterfactual Public-Value Target Audit

Status: rejected as an absolute state-value target on 2026-06-13. The
counterfactual sampler is qualified; no model, test, or gameplay domain was
opened.

## Context

ADR 0073 reconstructed exact hex adjacency and oriented terrain edges, but the
fresh graph model regressed final-score correlation and MAE. Its small
pairwise log-loss improvement did not translate into materially better
ordering. The common target remained one realized terminal outcome from each
H6 trajectory.

That target confounds public board quality with one hidden tile order, one
wildlife order, and all subsequent market interactions. A larger encoder
cannot recover randomness absent from its public input. The next question is
therefore upstream of architecture: whether repeated legal continuations from
the same public state produce a stable expected-return target at a locally
affordable sample count.

The distributional reinforcement-learning literature distinguishes expected
value from the full random return induced by stochastic transitions. This
audit does not adopt a Bellman algorithm, but it retains the same essential
evidence: repeated terminal returns from one observable state.

- [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
- [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044)

## Decision

Add a versioned, checksummed counterfactual-value dataset:

1. Generate the frozen H6 source trajectory under
   `habitat-candidate-lookahead-v1-k8-h6-r4-d4`.
2. At every one of the 80 pre-action public states, retain the acting-seat
   `compact-entity-v2` record and exact current decomposed score.
3. Clone that state sixteen times. For each sample, replace only hidden tile
   and wildlife order with a deterministic domain-separated public
   redetermination seed.
4. Continue every clone to game end with the identical frozen H6 strategy for
   all seats.
5. Store all sixteen terminal decomposed scores and their exact seeds, not
   only a pre-averaged target.
6. Retain the factual source-trajectory terminal score separately for direct
   comparison with the counterfactual distribution.
7. Retain public-safe supply summaries invariant to hidden
   redetermination: wildlife bag counts; unseen-tile terrain capacity;
   unseen-tile wildlife capacity; unseen keystones by terrain; and unseen
   dual-terrain pair counts.
8. Validate every shard header, checksum, game/turn sequence, sample count,
   unique seed, score range, source provenance, executable provenance, and
   public-supply invariance.

The audit computes:

- per-state R16 mean, standard deviation, and standard error of total score;
- factual-single-trajectory error against the R16 mean;
- prefix R1, R2, R4, and R8 mean drift against the R16 mean;
- R1/R2/R4/R8 pairwise ordering accuracy and log loss against R16 within each
  four-seat round;
- phase-binned stability over personal turns 1-5, 6-10, 11-15, and 16-20;
- variance of state means versus mean within-state return variance;
- wall time and continuation throughput.

## Frozen Protocol

- Implementation-only smoke: train split index 9,993, one game, two samples
  per state. It may validate wiring and determinism only.
- Substantive audit: validation split indices 65,000-65,001, two complete
  source games, 160 public states, sixteen continuations per state.
- Rules: symmetric four-player AAAAA, no habitat bonuses.
- Source and continuation policy: frozen H6 K8/H6/R4/D4.
- Sample seeds: BLAKE3 domain
  `cascadia-v2-counterfactual-value-v1`, split, game index, completed turn,
  and sample index.
- Execution: local Apple M4 only. Games may run in parallel; each recorded
  result must be independent of scheduling.
- No model training, test split, gameplay comparison, architecture choice,
  sample-count retry, alternate continuation policy, or external compute.

The target is viable for a separately preregistered training experiment only
if all conditions hold:

- every integrity, replay, determinism, and public-supply-invariance check
  passes;
- mean R16 total-score standard error is at most 1.50 points;
- R8 mean absolute drift from R16 is at most 1.25 points;
- R8 within-round pairwise ordering accuracy against R16 is at least 70%;
- R8 within-round pairwise log loss is below R1;
- the standard deviation of R16 state means is at least 2.0 points;
- collection throughput projects a 256-game, R8, 80-state corpus at no more
  than 24 uncontended local hours.

Passing does not choose a model. It authorizes a separate ADR to collect and
train on repeated public returns. If R8 fails but R16 is stable, the result
must report the local compute implication before any larger collection is
authorized. If R16 itself is unstable, this H6 continuation target is closed.

## Implementation Qualification

The one-game R2 smoke at train index 9,993 passed before either substantive
validation index was opened:

- the release collector retained all 80 public states and 160 complete H6
  continuations;
- the fixed 1,804-byte record preserved the public state, current and factual
  decomposed scores, public supply, two exact 32-byte seeds, and two terminal
  decomposed samples;
- every shard header, checksum, game/turn sequence, unique seed, unused sample
  slot, and manifest total passed independent validation;
- public wildlife and unseen-tile supply remained exactly invariant under
  every hidden redetermination;
- a second complete collection produced a byte-identical 144,480-byte shard,
  BLAKE3
  `3a87ea2885cfb8bf35da3a9ea382501b876ac59a3cba49d9518e2605cd68759d`;
- strict Clippy, focused Rust tests, formatting, and diff checks passed;
- collection took 55.01 seconds, 2.91 continuations per second, projecting a
  256-game R8 corpus at 15.65 uncontended hours before optimization.

The smoke's two-sample return mean is not substantive evidence. Its factual
trajectory was 1.92 points MAE from that provisional mean, which only confirms
that the audit is measuring a nontrivial source of variation. The preserved
implementation report is
`docs/v2/reports/counterfactual-public-value-target-audit-v1-implementation-smoke.json`.

## Result

The two authorized validation games completed all 160 public states and 2,560
H6 continuations in 779.78 seconds. Both shards and the aggregate manifest
passed validation.

The repeated-return estimator is stable:

- mean R16 standard error: 0.508 points;
- R8 mean absolute drift from R16: 0.487 points;
- R8 P90 absolute drift: 1.131 points;
- R8 within-round pairwise accuracy: 91.14%;
- R8 pairwise log loss: 0.583, versus 0.751 at R1;
- projected 256-game R8 collection: 13.86 uncontended hours.

The factual single trajectory was materially noisier: 1.335 points MAE and
0.685 correlation against the R16 state mean. Early-game states were hardest,
with 0.735-point mean R16 standard error and 0.856-point R8 drift; both fell
steadily toward the endgame.

The target nevertheless failed its frozen signal-width gate. Across all 160
states, the standard deviation of R16 expected totals was 1.945 points, below
the required 2.0. Every other gate passed.

This is not permission to round the threshold or train anyway. Absolute H6
public-state expected value remains too compressed to justify a 256-game
corpus. The sampler itself is qualified and should be reused for
decision-local candidate advantages, where subtracting a same-decision
baseline removes the narrow game-level offset. That successor requires a new
ADR and fresh evidence.

## Maximum Compute

One one-game R2 implementation smoke and one two-game R16 substantive audit.
No retry, sweep, extra game, changed threshold, changed policy, changed
sampling domain, test access, gameplay, model training, or external compute.
