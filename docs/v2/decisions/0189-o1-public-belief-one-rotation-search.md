# ADR 0189: O1 Public-Belief One-Rotation Search

**Status:** Completed with replicated validation null; sealed test not opened
**Date:** 2026-06-17
**Experiment:** `o1-public-belief-one-rotation-search-v1`
**Protocol:** `o1-public-belief-one-rotation-high-regret-v2`

## Context

ADR 0187 established that recent public action history improves calibrated
opponent next-draft prediction on held-out policy families. ADR 0188 then
showed that directly fusing those frozen probabilities into the exact-R2
ranker was directionally positive but too small to pass any promotion gate:

- P2 versus Z0 mean retained R4800-regret improvement: `0.009142`;
- game-clustered paired 95% interval: `[-0.018194, 0.000000]`;
- high-regret improvement: `0.022008`.

That result rejects another direct adapter sweep. It does not reject the
mechanism for which O1 was designed: changing the distribution of market
states that the focal player expects to face after three opponents act.

The state-of-the-art research agenda identifies an exact stochastic
public-belief tree as the third-highest-upside direction and opponent intent as
the fifth. The feature audit independently identifies future-access windows as
the missing bridge between opponent boards and market drafting. This
experiment composes those two hypotheses once, under a bounded open-validation
protocol, before either is promoted to gameplay.

## Decision

Evaluate every action in the frozen exact-R2 top-64 cohort by one complete
table rotation:

1. reconstruct the exact public validation state;
2. apply the frozen complete focal action, including its observed staged
   market prelude;
3. redeterminize all remaining genuinely hidden supply from the public
   afterstate identity;
4. simulate up to three opponent turns;
5. return to the focal player;
6. evaluate the public leaf with current exact base score plus the qualified
   frozen v4opp NNUE remaining-value model;
7. allocate a fixed 640 trajectories by deterministic sequential halving.

Run four graph-identical search-policy arms:

1. `c0-pattern-prior`
   - samples each opponent's draft key from the frozen pattern heuristic;
   - softmax temperature is exactly `1.0`.
2. `a0-public-state-intent`
   - samples legal draft keys from frozen O1 A0 next-draft probabilities.
3. `a2-history-intent`
   - samples legal draft keys from frozen O1 A2 probabilities conditioned on
     recent public action history.
4. `s3-shuffled-history-intent`
   - uses the frozen stratified A2 donor mapping from ADR 0188;
   - preserves the probability-vector distribution while breaking alignment
     to the candidate afterstate;
   - is a nonpromotable negative control.

After a draft key is selected, all arms use the same best frozen
pattern-policy placement conditional on that draft. The experiment therefore
tests only the branch distribution induced by opponent draft belief, not a
larger learned opponent policy.

## Why No Oracle-Opponent Arm

The implementation plan originally reserved one slot for an oracle-opponent
ceiling. The available graded-oracle corpus labels focal root actions, not the
counterfactual complete actions of all three opponents under every
redeterminization. Treating the historical continuation, hidden future, or
focal R4800 label as an opponent oracle would leak information or answer a
different question.

The frozen R4800 root labels already provide the action-quality reference.
The shuffled-history control is the valid fourth arm because it distinguishes
aligned opponent information from model capacity, calibration, and marginal
probability effects without using unavailable counterfactual truth.

## Frozen Decision Panel

The panel is selected once from the completed ADR 0188 Z0 validation report:

- split: open validation only;
- eligibility: finite, scorable retained R4800 regret at least `0.50`;
- groups: exactly `99`;
- panel ID:
  `9f8aadb8e789a84b3450dcf78bb2e72ff99630dedbae7421a035749b8d986fdd`;
- panel-file BLAKE3:
  `5da22cab160e3a243eb61c1e5decc882fc0d4a993ae38aa83abdc1c387e68130`;
- source Z0 report BLAKE3:
  `06ca96212d4e39cd0b8cf85b15fd0ef0f921c2173d3031c2635b05220083e077`.

Rows, group IDs, source games, turns, and source Z0 regrets are immutable.
The sealed test split remains unopened.

## Root Cohort

Each group uses all 64 actions from the frozen ADR 0188 validation cohort:

- cohort ID:
  `3856f9c4cf73d34c470357cdf220dbf8314a6ddd2a6340ee686a5e2e16254591`;
- cohort-manifest BLAKE3:
  `6636e6c7659e3d5c96dd0e5cd1703fbab16a8605a3fb421eae668a431a2c3780`;
- dataset ID: `graded-oracle-validation-11e6bc9647c68df7`;
- dataset-manifest BLAKE3:
  `302ceb7a57482b0fb5fb12963521be35aafc121a36f572e6b9f47def1b820a31`.

Every source candidate index and canonical action hash must reconstruct
exactly. No search arm may add, remove, reorder, or relabel candidates.
Consequently, any positive result is a within-cohort selection result, not a
proposal-recall result.

## Opponent Belief

The frozen intent cache is:

- intent ID:
  `b0de970601ddabcc7b3430397b07203df36656f810a53943337c450b2f3152f4`;
- manifest BLAKE3:
  `b915ad288710d59c41f8fc8dd45f220bd61c358ea5ed30aa6508299a88796bc6`.

For each opponent and legal draft option, the O1 weight combines:

- paired versus independent-draft probability;
- tile-slot probability;
- wildlife-slot probability;
- drafted-wildlife probability.

The three content probabilities use their geometric mean so that one
calibrated head cannot dominate only because the factorization has more
terms. Every legal option receives a floor of `1e-9`; invalid, nonfinite, or
nonpositive weights fail closed. The same BLAKE3-derived uniform variate is
used by every arm for the same group, root, sample, and opponent offset.

## Search Budget

All 64 roots begin active. Sequential halving is frozen as:

| Stage | Active roots | New trajectories per root | Retained roots |
|---:|---:|---:|---:|
| 1 | 64 | 4 | 32 |
| 2 | 32 | 4 | 16 |
| 3 | 16 | 8 | 8 |
| 4 | 8 | 16 | 1 |

This is exactly `640` trajectories per decision group:

`64*4 + 32*4 + 16*8 + 8*16 = 640`.

Ties use canonical action hash. The sample index is retained across stages.
There is no adaptive stopping, arm-specific budget, validation peeking, or
wall-time substitution.

## Public-Belief And Chance Contract

The authoritative v2 `GameState` owns all search transitions.

- The root starts from an exact replay of the public graded-oracle record.
- The frozen complete root is first applied against the exact replay,
  including any staged three-of-a-kind replacement or paid wipe encoded in
  that action.
- After the root, all remaining hidden tile and wildlife order is sorted and
  redeterminized from a BLAKE3 seed over group ID, root action hash, and
  sample index.
- No post-root hidden order from the recorded game is retained as evidence.
- Public market, public boards, Nature Tokens, scoring cards, player order,
  exact remaining public counts, and legal actions remain exact.
- The same root/sample redeterminization is used across all four arms.
- Each group includes an explicit post-root hidden-order invariance probe.
- Nodes are identified by public state and canonical action hashes, never
  candidate rank.

The root's staged public market is part of the frozen candidate identity.
Conditioning on it is required to evaluate that complete action and is not
future leakage. ADR 0190 records the invalid protocol-v1 launch that exposed
this boundary and freezes the corrected causal order.

This first T1 experiment uses sampled exact simulator trajectories rather than
a persistent transposition tree. Tree reuse, incremental state deltas, and
cross-turn rerooting belong to T2 and are unauthorized unless T1 first
establishes a quality signal.

## Leaf Evaluation

Terminal leaves use exact focal-player base score. Nonterminal leaves use:

`current exact base score + qualified frozen v4opp NNUE remaining value`.

The frozen MLX model is:

- architecture: `legacy-sparse-nnue-v4opp-mlx-v1`;
- feature width: `11,231`;
- manifest BLAKE3:
  `dd3ea3bbbff0187107695132531a56c09a1da18b58fac4bacacf66960fd7ff0d`;
- safetensors BLAKE3:
  `3f8f2609b1440396720aa48adabf9561a4a172d006f77011a9516baa0b06ba65`.

Rust extracts the exact qualified sparse indices. MLX performs only batched
inference. No model parameter is trained or modified.

## Spatial Representation Boundary

The user hypothesis that 121 cells are sufficient is preserved and already
supported by the representation campaign: exact R2 sparse occupied/frontier
state, not a dense 21-by-21 lattice, is the selected v2 substrate. The
historical 441-cell block is not a v2 search state and is not materialized as
a dense tensor by this experiment.

The only 441-coordinate dependency is inside the frozen legacy NNUE feature
index used for qualified leaf parity. It is a sparse compatibility evaluator,
not a new representation choice:

- v2 simulation remains canonical typed `GameState`;
- occupied and frontier structure remains sparse;
- no 441-cell input is trained;
- no 441-cell input is passed through the search policy;
- no decision, chance, transposition, or opponent node is keyed by a dense
  lattice;
- replacing the frozen leaf is a later matched experiment, not a silent
  change here.

This boundary isolates the O1/T1 hypothesis while retaining exact parity with
the current qualified value function. A positive result authorizes a separate
121/sparse relational leaf comparison; it does not endorse 441 cells.

## Metrics

Primary endpoint:

- mean retained R4800 regret of the selected root over the 99 frozen groups.

Secondary endpoints:

- retained R4800-winner recall;
- R1200 pairwise ordering accuracy over all labeled root pairs;
- selected-action agreement between arms;
- search-value standard deviation;
- terminal versus MLX-evaluated leaves;
- opponent decisions and legal draft options;
- trajectories, model rows, candidate hashes, and invariance checks;
- wall time and trajectories per second.

Comparisons are paired by decision group and bootstrapped by source game with
20,000 replicates and seed `2026061721`.

## Validation Gates

A2 is eligible for a sealed-test successor only if all gates pass:

1. every primary report exactly reproduces on its rotated host;
2. every group has 64 roots and exactly 640 trajectories;
3. all candidate hashes and hidden-order checks pass;
4. A2 lowers mean regret by at least `0.05` versus C0;
5. A2 lowers mean regret by at least `0.05` versus A0;
6. A2 lowers mean regret by at least `0.05` versus S3;
7. all three game-clustered paired A2-minus-reference 95% intervals have upper
   bound below zero;
8. A2 top-1 R4800-winner recall is no worse than both C0 and A0;
9. A2 R1200 pairwise accuracy is within `0.005` of the better of C0 and A0.

S3 is never promotable. A0 is diagnostic and cannot be selected under this
protocol because ADR 0187's distinctive claim concerns history-conditioned
intent. Failure of any accounting or replay gate makes the campaign null.

## Cluster Allocation

Primary wave:

| Host | Role |
|---|---|
| john1 | C0 pattern prior |
| john2 | A0 public-state intent |
| john3 | A2 history intent |
| john4 | S3 shuffled-history intent |

Rotated replay wave:

| Host | Role |
|---|---|
| john2 | C0 replay |
| john3 | A0 replay |
| john4 | A2 replay |
| john1 | S3 replay |

Every host receives and verifies the same immutable source bundle. Dataset,
cohort, intent, model, panel, and authorization bytes are separately fanned
out and checked. The queue is work-conserving: all four primary arms run
concurrently, followed by all four rotated replays concurrently.

## Claim Boundary

This experiment can establish only that aligned frozen opponent probabilities
improve one-rotation root selection on an open high-regret validation panel.
It cannot claim:

- sealed-test generalization;
- full-action proposal improvement;
- a gameplay gain;
- champion promotion;
- a score increase;
- progress toward 100;
- a persistent or reusable tree;
- an exact opponent behavioral model;
- superiority of a dense 441-cell representation.

If validation is null, the sealed test remains unopened and O1 is closed as a
near-term direct/search mechanism. If validation passes, a new ADR must freeze
the sealed-test and gameplay protocol before either is run.

## Result

Protocol v2 completed on 2026-06-17 with exact primary and rotated-host
replication. A2 had the lowest mean retained R4800 regret at `0.844773`, versus
`0.879504` for C0, `0.864676` for A0, and `0.849543` for shuffled A2.

A2 improved on C0 by `0.034731` with paired 95% interval
`[-0.048020, -0.026082]`, but missed the frozen `0.05` effect gate. Its
improvement over A0 was `0.019902` with interval
`[-0.041931, 0.000000]`; its improvement over shuffled A2 was only `0.004770`
with interval `[-0.015007, 0.010535]`. All accounting, recall, pairwise, and
cross-host replication checks passed.

The terminal classification is
`o1_public_belief_search_validation_null`. No arm is selected. The sealed test
and gameplay remain closed, and O1 is closed as a near-term integration
direction. The result report is
`reports/o1-public-belief-one-rotation-search-v1-result.md`.
