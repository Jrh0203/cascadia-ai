# T1 Search-Horizon Decomposition Preregistration

Experiment: `t1-search-horizon-decomposition-v1`

Protocol: `t1-strict-train-horizon-decomposition-v1`

ADR: `0191-t1-search-horizon-decomposition.md`

Status: completed; development null

Result: `t1-search-horizon-decomposition-v1-result.md`

## Research Question

Does exact stochastic lookahead improve complete-action selection beyond both
the frozen direct exact-R2 ranker and immediate qualified-leaf rescoring, and
at what number of opponent turns does any advantage first appear?

## Frozen Hypotheses

Search hypothesis:

> At equal roots, simulator, opponent policy, hidden futures, random numbers,
> search budget, and leaf evaluator, at least one nonzero opponent horizon
> materially lowers retained R4800 regret relative to both direct exact-R2
> ranking and root-only leaf rescoring.

Horizon hypothesis:

> If the value of stochastic lookahead comes from market survival and opponent
> resource removal, regret should improve as the leaf crosses the opponent
> turns responsible for that removal.

Leaf-only alternative:

> If H0 improves but no searched horizon improves beyond H0, the descriptive
> O1 search signal came from replacing the direct ranker with the qualified
> leaf, not from opponent simulation.

Null:

> No searched horizon passes the frozen effect, uncertainty, recall, pairwise,
> accounting, and crossed-host replication gates.

## Frozen Inputs

| Input | Identity |
|---|---|
| open-train dataset | `graded-oracle-train-93c7385b769a4bf9` |
| dataset manifest | `7ed12c943d75a786ccd4ccbe11a6b0146aad4fe5ed40f0cbaf1d652f5ac0bb99` |
| strict exact-R2 cohort | `aac7a480bd3f73bf15fa09b2314c8efa80cbae01a4ce09f8cf342845c2808512` |
| cohort manifest | `01d36dc98d24f83f99a839752e03fdd318560b34bd212aa66687f1ae35d3a827` |
| exact-R2 checkpoint manifest | `059268ab444a4a5d0b03190432a1c9cc7332fba55ec67746d6d635ec1fe5d13c` |
| exact-R2 checkpoint weights | `63fec159238f09c192cc44861b4c69a9b1be3932ddb6af6e32a0c8a34c2365c1` |
| qualified MLX manifest | `dd3ea3bbbff0187107695132531a56c09a1da18b58fac4bacacf66960fd7ff0d` |
| qualified MLX weights | `3f8f2609b1440396720aa48adabf9561a4a172d006f77011a9516baa0b06ba65` |

The immutable source bundle and authorization IDs are assigned after this
preregistration enters the bundle. Runtime rejects byte drift.

## Frozen Development Domain

- split: open train only;
- games: 7;
- decision groups: 560;
- groups per game: 80;
- rules: four-player AAAAA;
- habitat bonuses: disabled;
- candidate roots per group: 64;
- validation opened: no;
- sealed test opened: no;
- gameplay run: no.

Every group participates. There is no high-regret, phase, disagreement,
confidence, or label-dependent filter.

## Frozen Cohort Construction

The source O1 cohort is reusable only when its stored direct ranks are exactly
the set `0..63`. This condition holds for 518 groups.

For the other 42 groups:

1. bind the complete source group to the frozen R3 exact-R2 cache;
2. score every source candidate with the frozen exact-R2 checkpoint;
3. rank by descending score;
4. break equal scores by ascending canonical action hash;
5. retain exactly the first 64;
6. store candidates in ascending source position while retaining model rank.

Production accounting:

- reused groups: 518;
- rescored groups: 42;
- rescored source candidates: 21,332;
- strict groups: 560 of 560;
- unique direct rank-0 action per group: 560 of 560.

The cohort builder and loader verify all tensor sizes, BLAKE3 checksums,
strict-rank sets, source ordering, direct indices, and per-row cohort hashes.

## Frozen Comparator

`direct-exact-r2`

- select the unique cohort action whose frozen direct rank is zero;
- use stored frozen direct scores for pairwise metrics;
- consume no simulator or MLX inference;
- remain a comparator, not a primary/replay role.

## Frozen Arms

### H0 Root Leaf

- apply each exact complete root;
- simulate zero opponent turns;
- evaluate the immediate focal afterstate once;
- one deterministic sample per root;
- select the highest leaf value.

### H1 One Opponent

- apply each exact complete root;
- redeterminize remaining hidden supply;
- sample and apply one opponent turn;
- evaluate the focal player;
- use the frozen 640-trajectory sequential-halving schedule.

### H2 Two Opponents

- identical to H1 through the first opponent;
- sample and apply the second opponent turn;
- evaluate the focal player;
- use the same 640-trajectory schedule.

### H3 Full Rotation

- identical to H2 through the second opponent;
- sample and apply the third opponent turn;
- evaluate when the focal player is again current, unless terminal;
- use the same 640-trajectory schedule.

No arm changes root width, opponent policy, leaf, temperature, random-number
domains, or allocation schedule.

## Frozen Root Replay

For each decision group:

1. replay the source game from its dataset seed;
2. require completed turns, current player, position bytes, and public-state
   hash to match the graded-oracle record;
3. reconstruct all 64 complete actions by source candidate index;
4. require source candidate, cohort, and reconstructed canonical action hashes
   to match;
5. apply the recorded champion only after the experiment group is evaluated,
   so replay continues along the factual source trajectory.

No candidate may be silently skipped after a legal or hash failure.

## Frozen Chance And Prefix Coupling

The complete root is applied before any future redeterminization.

For H1-H3:

- determinization key:
  `BLAKE3(domain, group_id, root_action_hash, sample_index)`;
- opponent-uniform key:
  `BLAKE3(domain, group_id, root_action_hash, sample_index, opponent_offset)`;
- horizon identity is excluded from both keys;
- the same pattern-policy option construction is rerun at the same public
  prefix;
- one trace hash records all sampled opponent actions.

Required prefix checks:

- H1 trace equals the first-opponent prefix of H2 and H3;
- H2 trace equals the first-two-opponent prefix of H3;
- a pre-root hidden-order perturbation cannot change a post-root trajectory
  once the frozen root has been applied and the registered determinization is
  installed;
- H0 public leaf and focal features are hidden-order invariant.

## Frozen Opponent Policy

For each opponent node:

1. use the standard automatic three-of-a-kind replacement decision;
2. enumerate pattern-aware complete legal actions;
3. map each action to paired or independent draft key;
4. retain one best placement per key;
5. break placement ties by ascending action hash;
6. convert heuristic values to softmax weights at temperature `1.0`;
7. sample one key with the registered common uniform variate.

Frozen pattern configuration:

- immediate candidate limit: 8;
- habitat candidate limit: 6;
- bear candidate limit: 8;
- future market draws: 4.

O1 intent vectors and shuffled donors are not loaded.

## Frozen Search

H0:

- roots: 64;
- samples per root: 1;
- total evaluations per group: 64;
- total production evaluations: 35,840.

H1-H3:

- roots: 64;
- stage samples: `[4, 4, 8, 16]`;
- stage survivors: `[32, 16, 8, 1]`;
- total trajectories per group: 640;
- total production trajectories per arm: 358,400;
- statistic: arithmetic mean leaf value;
- tie-break: ascending canonical action hash.

Forbidden:

- early stopping;
- confidence pruning;
- rank-keyed state merging;
- arm-specific budgets;
- adaptive horizon selection within a run;
- validation-driven tuning;
- wall-time substitution;
- changing the leaf after any result is observed.

## Frozen Leaf

Terminal leaf:

- exact focal Card A base score.

Nonterminal leaf:

- exact current focal base score;
- plus frozen v4opp MLX remaining-value prediction;
- exact sparse legacy-compatible feature extraction in Rust;
- focal-player `BagInfo`, even when another player is current;
- inference batches of at most 4,096 rows;
- no training, calibration, ensemble, or parameter mutation.

The model process must pass an empty-row warmup and return one finite value per
requested row.

## Frozen 121/441 Contract

- simulation state: typed sparse v2 `GameState`;
- learned root representation: exact sparse R2 occupied/frontier;
- candidate representation: complete action plus exact edit identity;
- dense 441-cell tensors: forbidden;
- 441-coordinate legacy indices: allowed only inside the frozen qualified leaf
  compatibility evaluator;
- any future learned replacement leaf must be a separately authorized matched
  experiment.

No T1 result can be interpreted as support for a dense 441-cell state.

## Frozen Per-Group Output

Each group report records:

- cohort row, group ID, game seed, completed turn, and focal player;
- public-state hash;
- arm and horizon;
- all 64 source indices and action hashes;
- search mean, standard deviation, sample count, and elimination stage;
- frozen direct score and rank;
- R600, R1200, and R4800 estimates where available;
- selected cohort/source index and action hash;
- terminal and MLX leaf counts;
- opponent decisions and legal option counts;
- trace-prefix and hidden-order invariance results;
- group scientific result ID.

Partially completed groups may resume only when the frozen run identity matches
exactly.

## Frozen Metrics

Per comparator or arm:

- mean and median retained R4800 regret;
- top-1 retained R4800-winner recall;
- R1200 pairwise accuracy;
- selected action hashes;
- phase- and turn-stratified regret;
- wall seconds;
- trajectories or leaf evaluations;
- MLX rows;
- terminal leaves;
- opponent decisions and legal options.

Paired comparisons:

- H0 versus direct;
- H1 versus direct and H0;
- H2 versus direct and H0;
- H3 versus direct and H0;
- adjacent horizon differences H2-H1 and H3-H2 as diagnostics.

## Frozen Statistical Procedure

All paired differences are treatment regret minus reference regret.

- cluster unit: source game;
- games: 7;
- bootstrap replicates: 20,000;
- bootstrap seed: `2026061722`;
- ordinary interval: percentile 95%;
- one-sided superiority p-value:
  `(1 + bootstrap_count(difference >= 0)) / (replicates + 1)`;
- multiplicity family: six searched-horizon comparisons, each H1-H3 against
  direct and H0;
- correction: Holm-Bonferroni at familywise alpha `0.05`.

The selected searched horizon is the passing horizon with minimum mean regret.
If mean regrets are exactly equal, choose the shorter horizon.

## Frozen Integrity Gates

Every production report must prove:

1. exact authorization, bundle, dataset, cohort, and model identities;
2. 560 expected and completed groups;
3. 64 reconstructed candidate hashes per group;
4. one unique selected root per group;
5. finite root values;
6. H0 exactly 64 evaluations per group;
7. H1-H3 exactly 640 trajectories per group;
8. exact opponent-decision count implied by horizon and terminal states;
9. one hidden-order invariance check per group;
10. exact primary/replay scientific result identity;
11. no validation, sealed-test, or gameplay access.

## Frozen Promotion Gates

A searched horizon is eligible only if all gates pass:

1. its primary and rotated-host replay match exactly;
2. all four arms satisfy complete accounting;
3. mean regret improvement versus direct is at least `0.05`;
4. mean regret improvement versus H0 is at least `0.03`;
5. both superiority comparisons survive Holm correction;
6. top-1 recall is at least the maximum of direct and H0;
7. R1200 pairwise accuracy is no more than `0.005` below the maximum of direct
   and H0.

Pass classification:

`t1_search_horizon_decomposition_development_passed`

Leaf-only classification:

`t1_search_horizon_leaf_only`

Null classification:

`t1_search_horizon_decomposition_development_null`

Invalid classification:

`t1_search_horizon_decomposition_invalid`

A pass authorizes only a fresh validation preregistration using the selected
horizon. It does not authorize sealed test or gameplay.

## Frozen Cluster Schedule

Primary:

| Host | Role |
|---|---|
| john1 | `h0-primary` |
| john2 | `h1-primary` |
| john3 | `h2-primary` |
| john4 | `h3-primary` |

Replay:

| Host | Role |
|---|---|
| john2 | `h0-replay` |
| john3 | `h1-replay` |
| john4 | `h2-replay` |
| john1 | `h3-replay` |

The queue runs all four primaries concurrently, then all four replays
concurrently. Fanout, preflight, collection, and aggregation are explicit
dependencies. No machine is intentionally idle while runnable work exists.

## Claim Boundary

This campaign is an open-train mechanism decomposition. It cannot establish:

- validation or sealed-test generalization;
- gameplay strength;
- champion score;
- mean score progress toward 100;
- a persistent search tree;
- cross-turn reuse;
- superiority of a new value function;
- necessity of a 441-cell representation.

## Terminal Classification

All 20 immutable queue tasks completed. Every primary/replay pair reproduced
exactly, all arms shared frozen roots, and every report was complete and fully
accounted. The terminal aggregate classified the campaign as:

`t1_search_horizon_decomposition_development_null`

Direct exact-R2 mean regret was 0.562608. H0, H1, H2, and H3 reached 0.636628,
0.646858, 0.640652, and 0.635184 respectively. No searched arm passed any
mechanism gate, and the H0 leaf-only alternative regressed materially.

No validation, sealed test, or gameplay domain was opened. See
[`t1-search-horizon-decomposition-v1-result.md`](t1-search-horizon-decomposition-v1-result.md)
for the complete terminal record.
