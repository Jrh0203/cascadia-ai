# O1 High-Regret Draft-Ranking Integration v1 Result

**Completed:** 2026-06-17
**Experiment:** `o1-high-regret-draft-ranking-integration-v1`
**Validation classification:** `o1_ranking_validation_factorial_null`
**Test classification:** `o1_ranking_test_not_opened`
**Aggregate ID:** `9f823c2d495e74b67a2507e8d2bcf4890f8a52084df3d575640cd52107b5a322`

## Verdict

The direct O1 ranking integration is a replicated null.

The selected A2 opponent-intent representation produced a small directional
improvement over the zero-intent adapter, but it did not reach any
preregistered effect-size gate and its game-clustered confidence interval
touched zero. A0 was essentially identical to the control. The stratified
shuffle regressed overall regret and top-1 recall.

No arm was eligible, the sealed test remained unopened, and no gameplay or
score claim is authorized.

The O1 signal remains calibrated on policy-held-out opponent behavior under
ADR 0187. This result says that adding its marginalized probability vector as
a residual feature to the frozen exact-R2 ranker is not a useful policy
improvement mechanism at the tested scale. The next O1 experiment must use
the signal inside public-belief search, where opponent-action probabilities
can alter branch allocation and backed-up values, before O1 is retired.

## Matched Design

All four arms:

- warm-started the accepted exact-R2 step-3,000 checkpoint;
- froze every existing parent, candidate, residual, and uncertainty parameter;
- trained the same 56,384-parameter residual adapter for 2,000 steps;
- used the same group order, labels, optimizer, initialization, and top-64
  candidate cohorts;
- evaluated only the final checkpoint;
- used compact sparse board entities and never materialized a 441-cell state.

The arms differed only in the immutable 81-value input:

| Arm | Input |
|---|---|
| Z0 | all zeros |
| B1 | frozen A0 public-state probabilities |
| P2 | frozen A2 recent-history plus next-draft-auxiliary probabilities |
| S3 | A2 probabilities shuffled within frozen public strata |

For each candidate, A0 and A2 probabilities were averaged over eight
deterministic public refill proposals drawn from exact semantic supply. No
realized hidden refill, hidden order, policy identity, game identity, future
action, or teacher value entered a model feature.

## Validation Result

The strict validation cohort contained 240 decisions and 15,360 candidate
actions. Five groups had no R4800-labeled action inside the frozen top 64, so
the preregistered R4800 endpoints used 235 scorable groups. R1200 pairwise
metrics used all 240 groups.

| Arm | Mean retained R4800 regret | Improvement vs Z0 | Top-1 recall | R1200 pairwise | Eligible |
|---|---:|---:|---:|---:|---|
| Z0 zero intent | 0.506299 | control | 0.157447 | 0.578547 | control |
| B1 A0 public state | 0.506161 | +0.000138 | 0.157447 | 0.579084 | no |
| **P2 A2 history auxiliary** | **0.497156** | **+0.009142** | **0.161702** | 0.577291 | **no** |
| S3 shuffled A2 | 0.507972 | -0.001673 | 0.153191 | 0.577465 | no |

P2 was the best arm directionally:

- treatment-minus-Z0 regret: `-0.009142`;
- game-clustered 95% interval: `[-0.018194, 0.000000]`;
- high-regret-group improvement: `0.022008` over 99 groups;
- top-1 recall delta: `+0.004255`;
- R1200 pairwise delta: `-0.001256`, inside the 0.5 percentage-point
  noninferiority guardrail.

Those values are far below the frozen gates:

- at least `0.05` mean regret improvement;
- confidence interval wholly below zero;
- at least `0.10` improvement on groups where Z0 regret was at least `0.50`.

B1 missed the same gates with an effect of only `0.000138`. S3 additionally
failed the top-1 recall nonregression gate.

P2 did not establish the stronger history-aligned interpretation. Its paired
interval against B1 was `[-0.018194, 0.000000]`, and against S3 was
`[-0.019880, 0.004168]`.

## Exact Replication

Every arm replayed on a different Mac. Primary and rotated runs matched
exactly on final trainable tensors, complete prediction panels, loss traces,
metrics, and scientific identity.

| Arm | Primary host | Replay host | Replication ID |
|---|---|---|---|
| Z0 | john1 | john2 | `d34504388c6e5845840f3c1ac2fe0ec9d173dc4c68ea84c606f1ae05ad4cd` |
| B1 | john2 | john3 | `42042e6621675e087cbfaf6f93145d525aa2436a3774b893418c0947c546b40c` |
| P2 | john3 | john4 | `c71507cf7465e360d129a6f9cb0f6bb6626f62366623d9c06fa150d366c4d122` |
| S3 | john4 | john1 | `64a96a34b25c609c6e57883ab90cd503f8c1507561d7df4a406e5134d14c14e4` |

All four primary preflights and all four rotated preflights passed. The source
tree digest was identical on every host:

`c167cb5f855965789928b6cde6ae1de1e2922192956c194f9887cc5e29c9795d`

## Performance

Validation scored all 15,360 candidates in 3.38 to 4.23 seconds per arm,
approximately 3,631 to 4,550 candidates per second. Peak active MLX memory was
about 138 MiB and no host used swap.

The full primary wave and full rotated replay wave each ran concurrently on
john1 through john4. Compact sparse entities and cached O1 predictions made
the eight-run campaign practical on local Apple Silicon.

## Interpretation

The experiment separates three claims:

1. A2 predicts policy-held-out opponent drafts and market survival better than
   public state alone. ADR 0187 established this.
2. A2 contains a weak directional ranking signal on exact-R2 high-regret
   states. P2's point estimate supports this descriptively.
3. A2 materially improves direct action ranking. This experiment rejects that
   claim at the frozen effect size and confidence gates.

The adapter could only learn a static correction from candidate-level expected
probabilities. It could not use O1 to decide which opponent branches deserve
simulation, update probabilities after observed opponent actions, or back up
the consequences of market consumption. Those are search operations, not
feature-fusion operations.

The authorized successor is therefore a matched one-rotation public-belief
search experiment with four arms:

- the existing public search opponent policy;
- A0 public-state opponent probabilities;
- A2 history-aware opponent probabilities;
- an oracle-opponent diagnostic ceiling that is never promotable.

O1 should be rejected if both the direct policy route and that search route
remain neutral.

## Artifacts

- aggregate:
  `artifacts/experiments/o1-high-regret-draft-ranking-integration-v1/aggregate.json`;
- authorization:
  `artifacts/experiments/o1-high-regret-draft-ranking-integration-v1/control/authorization.json`;
- immutable cohort:
  `artifacts/experiments/o1-high-regret-draft-ranking-integration-v1/cohort/3856f9c4cf73d34c470357cdf220dbf8314a6ddd2a6340ee686a5e2e16254591`;
- public afterstates:
  `artifacts/experiments/o1-high-regret-draft-ranking-integration-v1/afterstates/effc8e4aae6551e6f29c46862cef6703a5f7f6cbb905af9d153593a38c80fb93`;
- O1 intent cache:
  `artifacts/experiments/o1-high-regret-draft-ranking-integration-v1/intent/b0de970601ddabcc7b3430397b07203df36656f810a53943337c450b2f3152f4`;
- eight primary and replay reports:
  `artifacts/experiments/o1-high-regret-draft-ranking-integration-v1/reports`.
