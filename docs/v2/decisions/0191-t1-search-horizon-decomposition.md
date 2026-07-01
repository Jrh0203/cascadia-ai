# ADR 0191: T1 Search-Horizon Decomposition

**Status:** Completed; development null, no successor authorized
**Date:** 2026-06-17
**Experiment:** `t1-search-horizon-decomposition-v1`
**Protocol:** `t1-strict-train-horizon-decomposition-v1`

## Context

ADR 0189 tested whether O1 opponent-intent probabilities improve a
one-rotation public-belief search. The history-conditioned arm was
directionally better than the pattern-prior control, but the improvement was
too small and did not survive the public-state and shuffled-history mechanism
controls. O1 is therefore closed.

That campaign exposed a separate, non-preregistered observation. On its
selection-conditioned 99-group panel, every one-rotation search arm had lower
retained R4800 regret than the frozen direct exact-R2 ranker by roughly 0.12 to
0.16 points. That observation cannot be promoted because:

- the panel was selected for high direct-ranker regret;
- search versus direct ranking was not a frozen endpoint;
- root-only leaf rescoring was not measured;
- the contribution of each opponent turn was not isolated;
- the open validation domain was already consumed by the O1 comparison.

The next experiment therefore uses the untouched open train split and removes
opponent-intent variation. It asks whether the apparent signal comes from the
qualified leaf alone, from one or two opponent turns, or from a complete table
rotation.

## Decision

Run four matched offline arms over every one of the 560 open-train decision
groups:

1. `h0-root-leaf`
   - apply each complete frozen root;
   - evaluate the immediate public afterstate once;
   - simulate no opponent turns.
2. `h1-one-opponent`
   - apply the root;
   - redeterminize the remaining hidden future;
   - sample one pattern-prior opponent turn;
   - evaluate the focal player.
3. `h2-two-opponents`
   - identical to H1 through the first opponent;
   - sample a second pattern-prior opponent turn;
   - evaluate the focal player.
4. `h3-full-rotation`
   - identical to H2 through the second opponent;
   - sample the third opponent turn;
   - evaluate after one complete table rotation.

The frozen direct exact-R2 rank-0 action is an offline comparator read from the
cohort, not a fifth compute arm.

H1, H2, and H3 use exactly the same:

- 64 complete root actions;
- root-first chance boundary;
- hidden redeterminization for a given group, root, and sample;
- pattern-prior legal draft options;
- opponent uniform variates for shared prefixes;
- sequential-halving schedule;
- qualified MLX leaf;
- simulator, labels, and tie-breaking.

The only treatment variable is the number of opponent turns simulated before
the leaf is evaluated.

## Strict Development Cohort

The experiment uses all 560 groups from:

- dataset ID: `graded-oracle-train-93c7385b769a4bf9`;
- dataset-manifest BLAKE3:
  `7ed12c943d75a786ccd4ccbe11a6b0146aad4fe5ed40f0cbaf1d652f5ac0bb99`;
- game seeds: `61000`, `61001`, `61002`, `61005`, `61006`, `61009`,
  `61010`.

Each group contains exactly the strict top 64 actions under the frozen
exact-R2 direct ranker. The cohort is:

- cohort ID:
  `aac7a480bd3f73bf15fa09b2314c8efa80cbae01a4ce09f8cf342845c2808512`;
- cohort-manifest BLAKE3:
  `01d36dc98d24f83f99a839752e03fdd318560b34bd212aa66687f1ae35d3a827`;
- schema: `t1-strict-exact-r2-top64-cohort-v1`.

The prior O1 cohort had inserted the graded-oracle winner when it fell outside
the direct top 64. Reusing only rows that happened not to be affected would
create a label-dependent filter. The T1 cohort instead:

- reuses 518 rows only where the stored ranks are exactly the set `0..63`;
- fully rescores all source candidates in the other 42 rows;
- selects the strict top 64 by descending frozen exact-R2 score;
- breaks score ties by ascending canonical action BLAKE3;
- records all source indices, direct scores, direct ranks, action hashes, and
  per-row cohort hashes.

The frozen ranker checkpoint provenance is:

- checkpoint-manifest BLAKE3:
  `059268ab444a4a5d0b03190432a1c9cc7332fba55ec67746d6d635ec1fe5d13c`;
- model BLAKE3:
  `63fec159238f09c192cc44861b4c69a9b1be3932ddb6af6e32a0c8a34c2365c1`.

No validation or sealed-test row was used to build the cohort.

## Root And Chance Contract

The authoritative v2 `GameState` owns every transition.

For every root trajectory:

1. replay the exact public train position;
2. reconstruct the complete root from its source graded-oracle record;
3. require the source, cohort, and reconstructed action hashes to match;
4. apply the complete root against the exact replay, including its frozen
   market prelude;
5. only after the root succeeds, sort and redeterminize the remaining hidden
   tile and wildlife order;
6. simulate the requested number of opponent turns;
7. evaluate the focal player.

The determinization key is a BLAKE3 domain over group ID, root action hash, and
sample index. It excludes the horizon arm. Opponent-uniform keys add only the
opponent offset and also exclude the arm. Therefore H1 is an exact prefix of
H2 and H3, and H2 is an exact prefix of H3, for every shared trajectory.

The frozen root is never applied after redeterminization. ADR 0190 remains the
permanent causal-boundary rule.

## Opponent Policy

Every simulated opponent uses the same frozen pattern-prior policy:

- enumerate legal complete actions under the standard prelude policy;
- deduplicate by public draft key;
- retain the highest-valued placement per draft key;
- break equal placement values by canonical action hash;
- softmax the retained heuristic values at temperature `1.0`;
- sample with the common BLAKE3-derived uniform variate.

An opponent maximizes its own pattern heuristic. It is never modeled as
minimizing the focal player. O1 probabilities, policy identity, hidden labels,
and historical continuations are absent.

## Search Budget

H0 evaluates every root exactly once, for 64 leaf evaluations per group and
35,840 evaluations over the corpus.

H1, H2, and H3 each use deterministic sequential halving:

| Stage | Active roots | New trajectories per root | Retained roots |
|---:|---:|---:|---:|
| 1 | 64 | 4 | 32 |
| 2 | 32 | 4 | 16 |
| 3 | 16 | 8 | 8 |
| 4 | 8 | 16 | 1 |

That is exactly 640 trajectories per group and 358,400 trajectories per
searched horizon. Root means use all samples accumulated before elimination.
Ties use ascending canonical action hash. There is no early stopping,
confidence pruning, arm-specific tuning, or wall-time substitution.

## Leaf Evaluation

Terminal leaves use the exact focal-player Card A base score with habitat
bonuses disabled.

Nonterminal leaves use:

`current exact focal base score + frozen v4opp NNUE remaining value`.

The qualified MLX leaf is:

- architecture: `legacy-sparse-nnue-v4opp-mlx-v1`;
- feature width: `11,231`;
- manifest BLAKE3:
  `dd3ea3bbbff0187107695132531a56c09a1da18b58fac4bacacf66960fd7ff0d`;
- safetensors BLAKE3:
  `3f8f2609b1440396720aa48adabf9561a4a172d006f77011a9516baa0b06ba65`.

H1 and H2 can stop while another player owns the turn. The evaluator still
extracts the focal player's board and focal-player `BagInfo`; current-player
identity does not redefine the value target.

## Representation Boundary

The experiment confirms the representation decision already reached by the
foundation campaign:

- v2 search state is the typed sparse `GameState`;
- the selected learned state substrate is exact sparse R2
  occupied/frontier state;
- no dense 441-cell v2 tensor is allocated, trained, hashed, or searched;
- the T1 cohort is selected by the compact exact-R2 ranker;
- candidate identity is a complete action hash, not a lattice rank.

The only 441-coordinate dependency is the frozen legacy NNUE feature-index
compatibility path used to hold leaf strength constant. Those indices are
sparse and are not evidence that 441 cells are required. A future learned leaf
must use the compact v2 representation in a matched experiment.

## Endpoints

Primary:

- mean retained R4800 regret of the selected action over all 560 groups.

Secondary:

- median retained R4800 regret;
- top-1 retained R4800-winner recall;
- R1200 pairwise ordering accuracy;
- selected-action agreement;
- phase- and turn-stratified regret;
- search-value standard deviation;
- terminal and MLX leaf counts;
- opponent decisions and legal option counts;
- trajectories, candidate hashes, invariance checks, wall time, and
  trajectories per second.

The direct comparator uses the cohort action with frozen direct rank zero.
H0 through H3 select by their own frozen search values.

## Statistical Contract

Comparisons are paired by decision group and clustered by source game.

- bootstrap replicates: `20,000`;
- seed: `2026061722`;
- ordinary paired percentile intervals: 95%;
- superiority family: H1, H2, and H3 versus both direct and H0;
- familywise control: one-sided Holm-Bonferroni at alpha `0.05`.

A searched horizon passes its mechanism gate only if:

1. its mean regret improvement versus direct is at least `0.05`;
2. its mean regret improvement versus H0 is at least `0.03`;
3. both one-sided paired superiority tests survive Holm correction;
4. its top-1 recall is no worse than both direct and H0;
5. its R1200 pairwise accuracy is within `0.005` of the better of direct and
   H0.

If multiple searched horizons pass, select the one with the lowest mean
regret. Exact ties select the shorter horizon.

## Classification

`t1_search_horizon_decomposition_development_passed`

- at least one of H1, H2, or H3 passes every mechanism and integrity gate;
- the selected horizon is authorized for a fresh open-validation successor.

`t1_search_horizon_leaf_only`

- no searched horizon passes;
- H0 nevertheless improves on direct by at least `0.05` with an ordinary
  paired 95% interval wholly below zero.

`t1_search_horizon_decomposition_development_null`

- neither condition above holds.

Any accounting, authorization, replay, hash, or invariance failure makes the
campaign invalid rather than null.

## Cluster Allocation

Primary wave:

| Host | Role |
|---|---|
| john1 | H0 root leaf |
| john2 | H1 one opponent |
| john3 | H2 two opponents |
| john4 | H3 full rotation |

Rotated replay wave:

| Host | Role |
|---|---|
| john2 | H0 replay |
| john3 | H1 replay |
| john4 | H2 replay |
| john1 | H3 replay |

All four hosts run concurrently in both waves. Every host verifies the same
immutable bundle, train dataset, strict cohort, MLX model, and authorization.

## Claim Boundary

This is an open-train mechanism experiment. It may identify a horizon worthy
of fresh validation. It cannot claim:

- open-validation or sealed-test generalization;
- gameplay improvement;
- champion promotion;
- a score increase;
- progress toward 100;
- tree reuse;
- a learned compact leaf improvement;
- superiority of a dense 441-cell representation.

No validation, sealed-test, or gameplay run is authorized by this ADR.

## Terminal Result

The immutable production graph completed all 20 tasks and produced aggregate
`817c5d469c59b830b5f7530712ceeacf105f19b2bb0d91850f9606d4e0559d13`.
The classification is:

`t1_search_horizon_decomposition_development_null`

| Arm | Mean R4800 regret | Top-1 recall | R1200 pairwise |
|---|---:|---:|---:|
| Direct exact R2 | **0.562608** | **0.223214** | **0.588746** |
| H0 root leaf | 0.636628 | 0.150000 | 0.573601 |
| H1 one opponent | 0.646858 | 0.153571 | 0.572253 |
| H2 two opponents | 0.640652 | 0.157143 | 0.572918 |
| H3 full rotation | 0.635184 | 0.164286 | 0.572296 |

Every leaf/search arm was worse than direct exact-R2 ranking. H3 improved H0
by only 0.001444 with interval `[-0.025228, +0.021434]`; no searched comparison
survived Holm correction, and every searched arm failed recall and pairwise
guardrails. All primary/replay pairs matched exactly and all accounting and
invariance gates passed, so this is a scientific null rather than an invalid
campaign.

## Consequences

- No T1 arm advances to validation or gameplay.
- T2 cross-turn reuse is blocked because a faster implementation of this
  failed decision mechanism is not decision-changing research.
- The exact sparse R2 direct ranker remains the selected comparator and
  substrate.
- The public-belief implementation and tests remain reusable infrastructure.
- O2 demand-supply matching is the next primary hypothesis; O3 plan-slot
  semantics may proceed independently.
- No score or progress-to-100 claim is authorized.

See
[`t1-search-horizon-decomposition-v1-result.md`](../reports/t1-search-horizon-decomposition-v1-result.md).
