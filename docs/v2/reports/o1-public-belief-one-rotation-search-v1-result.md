# O1 Public-Belief One-Rotation Search v1 Result

**Completed:** 2026-06-17
**Experiment:** `o1-public-belief-one-rotation-search-v1`
**Protocol:** `o1-public-belief-one-rotation-high-regret-v2`
**Classification:** `o1_public_belief_search_validation_null`
**Aggregate ID:** `07acd22b147b8d8fe7cc19a79b2e989625707efb568e87440e00a77fc66fff2d`

## Verdict

The O1 search integration is a replicated validation null.

A2 history-conditioned opponent probabilities were directionally best and
significantly better than the pattern-prior control, but the effect was below
the frozen minimum and did not survive the A0 and shuffled-history mechanism
controls:

- A2 versus C0 regret improvement: `0.034731`;
- paired game-bootstrap interval: `[-0.048020, -0.026082]`;
- A2 versus A0 improvement: `0.019902`;
- paired interval: `[-0.041931, 0.000000]`;
- A2 versus shuffled A2 improvement: `0.004770`;
- paired interval: `[-0.015007, 0.010535]`.

The frozen gate required at least `0.05` improvement against every reference
and a wholly negative interval for every comparison. No arm was eligible.
The sealed test remained unopened, no gameplay ran, and no score or progress
toward 100 claim is authorized.

Together with the null direct-ranking integration, this closes O1 as a
near-term policy-improvement mechanism. The calibrated opponent model remains
a valid behavioral prediction artifact, but neither tested integration
produced a mechanism-specific effect large enough to promote.

## Matched Result

All arms searched the same 99 open-validation high-regret decisions. Each
used all 64 exact-R2 roots, exactly 640 trajectories per decision, the same
post-root hidden-supply determinizations, the same placement policy, the same
qualified MLX leaf, and the same R4800 labels used only after search.

| Arm | Mean regret | Median regret | Top-1 recall | R1200 pairwise | Eligible |
|---|---:|---:|---:|---:|---|
| C0 pattern prior | 0.879504 | 0.870340 | 0.090909 | 0.564900 | control |
| A0 public-state intent | 0.864676 | 0.869940 | 0.101010 | 0.564963 | no |
| **A2 history intent** | **0.844773** | **0.847496** | **0.111111** | **0.565763** | **no** |
| S3 shuffled history | 0.849543 | 0.847496 | 0.101010 | 0.565077 | never |

A2 passed the recall and R1200 pairwise guardrails. It failed all three
minimum-effect gates, the A0 interval gate, and the shuffled-history interval
gate.

## Exact Replication

Every primary reproduced exactly on a different host:

| Arm | Primary | Replay | Scientific result ID |
|---|---|---|---|
| C0 | john1 | john2 | `3cef6e5f93f760e23dcbd862137134ab7c054b120185ccacae77c3ba759604d8` |
| A0 | john2 | john3 | `155d7f443418084429956259a2e677546dddd25c5d8ee9b1662c4648d83bad25` |
| A2 | john3 | john4 | `fda8508a9ba69891276c4d1c46af84f83e486a79daba317cdcbd6f1e22f2f4ee` |
| S3 | john4 | john1 | `5171e8e06c7a08c07806504844d89656cafa0859e6a3a9150e94e8435ef27da9` |

Every report contained:

- 99 completed groups;
- 6,336 exact candidate-hash checks;
- 63,360 trajectories;
- 63,360 MLX leaf rows;
- 190,080 opponent decisions;
- 99 post-root hidden-order invariance checks;
- complete finite accounting.

Primary wall time was 18.10 to 20.12 seconds per arm. The four primary arms
ran concurrently, followed by four concurrent rotated-host replays.

## Protocol Correction

The first protocol-v1 launch produced no report and no scientific result. It
incorrectly redeterminized hidden order before applying a staged complete
root action. ADR 0190 permanently corrected the causal boundary:

1. apply the frozen complete root against exact replay;
2. redeterminize all remaining hidden future;
3. simulate the opponent rotation.

Protocol v2 passed targeted regression, matched smoke, sealed-bundle smoke,
all four production primaries, and all four replays.

## Interpretation

The result separates opponent modeling from search.

History-conditioned probabilities changed selected actions and improved the
point estimate versus the hand-built pattern prior. That signal is real
enough to describe, but not specific enough to aligned history: shuffled A2
was within `0.004770` regret and its interval overlapped both directions.
Most of the gain therefore comes from the probability surface or branch
distribution generally, not the candidate-aligned recent-history information
claimed by O1.

There is a separate descriptive T1 signal. The frozen direct-ranker control
regret on the panel was `1.003134`; one-rotation search regret ranged from
`0.844773` to `0.879504`, an apparent reduction of `0.123630` to `0.158360`.
This was not a frozen gate, and the panel was selected for direct-ranker
regret, so it cannot authorize promotion. It does justify a fresh,
preregistered search-horizon decomposition that removes O1 and tests whether
the gain comes from the qualified leaf, one opponent turn, or the complete
rotation.

## Representation Boundary

The search state remained canonical exact v2 `GameState` with sparse
occupied/frontier structure. The historical 441-coordinate schema appeared
only as sparse indices inside the frozen qualified leaf evaluator. No dense
441-cell search state or learned input was created.

## Artifacts

- aggregate:
  `artifacts/experiments/o1-public-belief-one-rotation-search-v1/aggregate-v2.json`;
- immutable bundle:
  `artifacts/experiments/o1-public-belief-one-rotation-search-v1/bundles/4e1d9f3e863e66bc546c90e6a26c9d613fe445ed95af78db965233f1423ba1f5`;
- authorization:
  `artifacts/experiments/o1-public-belief-one-rotation-search-v1/control/authorization-package-v2/authorization.json`;
- collected reports:
  `artifacts/experiments/o1-public-belief-one-rotation-search-v1/collected-v2/reports`;
- invalid launch record:
  `docs/v2/reports/o1-public-belief-one-rotation-search-v1-invalid-launch-1.md`.
