# O1 Public-Belief One-Rotation Search Preregistration

Experiment: `o1-public-belief-one-rotation-search-v1`

Protocol: `o1-public-belief-one-rotation-high-regret-v2`

ADRs: `0189-o1-public-belief-one-rotation-search.md`,
`0190-frozen-root-prelude-contingency-boundary.md`

Status: completed with replicated validation null; sealed test not opened

## Protocol Amendment

Protocol v1 redeterminized hidden order before applying the frozen complete
root action. A root with a three-of-a-kind prelude is contingent on the
observed staged public market; changing the prelude draw changes the action
being evaluated and can make the frozen placement illegal.

All four v1 primaries failed on the same second group before writing a report.
No arm comparison, replay, sealed-test row, gameplay game, or score result was
observed. The launch is recorded in
`o1-public-belief-one-rotation-search-v1-invalid-launch-1.md`.

Protocol v2 changes only the chance boundary:

1. apply the frozen complete root against exact replay;
2. redeterminize every remaining hidden future;
3. simulate the opponent rotation.

The panel, candidates, arms, random-number coupling, search budget, leaf,
metrics, success gates, and claim boundary are unchanged.

## Research Question

Does the frozen, history-conditioned O1 opponent model improve focal root
selection when used in the place where its probabilities are causally
relevant: sampling which market resources three opponents remove before the
focal player's next turn?

The direct feature-fusion experiment was null. This is the one authorized
search integration. It is not an adapter retry.

## Frozen Hypotheses

Primary:

> At equal roots, simulator, random numbers, trajectory budget, placement
> policy, and leaf evaluator, A2 history-conditioned opponent draft
> probabilities lower retained R4800 root regret by at least 0.05 points
> relative to the pattern-policy control.

Mechanism:

> A2 must also beat A0 and the stratified shuffled-A2 negative control. This
> establishes that recent public history and candidate-aligned intent matter,
> rather than only probability smoothing or a changed draft prior.

Null:

> A2 fails any effect-size, uncertainty, recall, pairwise, accounting, or
> crossed-host replay gate.

## Frozen Inputs

| Input | Identity |
|---|---|
| validation dataset | `graded-oracle-validation-11e6bc9647c68df7` |
| dataset manifest | `302ceb7a57482b0fb5fb12963521be35aafc121a36f572e6b9f47def1b820a31` |
| exact-R2 cohort | `3856f9c4cf73d34c470357cdf220dbf8314a6ddd2a6340ee686a5e2e16254591` |
| cohort manifest | `6636e6c7659e3d5c96dd0e5cd1703fbab16a8605a3fb421eae668a431a2c3780` |
| O1 intent cache | `b0de970601ddabcc7b3430397b07203df36656f810a53943337c450b2f3152f4` |
| intent manifest | `b915ad288710d59c41f8fc8dd45f220bd61c358ea5ed30aa6508299a88796bc6` |
| high-regret panel | `9f8aadb8e789a84b3450dcf78bb2e72ff99630dedbae7421a035749b8d986fdd` |
| panel file | `5da22cab160e3a243eb61c1e5decc882fc0d4a993ae38aa83abdc1c387e68130` |
| legacy MLX manifest | `dd3ea3bbbff0187107695132531a56c09a1da18b58fac4bacacf66960fd7ff0d` |
| legacy MLX weights | `3f8f2609b1440396720aa48adabf9561a4a172d006f77011a9516baa0b06ba65` |

The immutable source bundle and authorization IDs are assigned after this
preregistration enters the bundle. Runtime rejects any byte drift.

## Frozen Panel

The panel contains exactly 99 open-validation decisions selected from the
completed ADR 0188 Z0 report.

Eligibility:

- R4800-scorable;
- finite retained R4800 regret;
- Z0 regret at least `0.50`.

The panel is sorted by original validation row. Selection may not be changed
after any search result is produced. Sealed-test rows are absent.

## Frozen Candidate Set

For every panel decision:

- use all 64 retained exact-R2 validation actions;
- reconstruct each action from its source graded-oracle record;
- require both the source and cohort canonical action hashes to match;
- preserve the original cohort index;
- prohibit label-dependent insertion, retrieval, or deletion.

R600, R1200, and R4800 estimates are read only for reporting and
classification. They never influence trajectories, elimination, or leaf
values.

## Frozen Arms

### C0 Pattern Prior

- legal draft keys are deduplicated from the frozen pattern-policy action
  ranking;
- one best placement is retained for each draft key;
- draft weights are softmax pattern heuristic values;
- temperature is `1.0`.

### A0 Public-State Intent

- use the candidate-aligned frozen A0 81-value vector;
- form a legal-draft probability from kind, tile slot, wildlife slot, and
  drafted-wildlife probabilities;
- use the same conditional placement policy as C0.

### A2 History Intent

- identical to A0 except for the candidate-aligned frozen A2 vector;
- A2 includes the preregistered recent public action history.

### S3 Shuffled History Intent

- identical to A2 except the vector comes from the frozen stratified donor
  index;
- donor strata were fixed before this experiment;
- S3 is never promotable.

All arms use the same legal option set. If an arm assigns zero or invalid
total mass, the run fails; it may not fall back to C0.

## Frozen Simulator

The v2 Rust simulator owns:

- state replay;
- market stabilization;
- legality;
- hidden-supply redeterminization;
- action application;
- terminal detection;
- Card A base scoring.

Each trajectory:

1. clones the exact replayed public state;
2. applies the frozen complete root action, including its recorded staged
   public market prelude;
3. sorts and redeterminizes all remaining hidden order with
   `BLAKE3(domain, group_id, action_hash, sample_index)`;
4. samples and applies up to three opponent actions;
5. evaluates the focal player when the rotation ends or the game terminates.

The arm is deliberately absent from the determinization and uniform-random
keys. This supplies common random numbers without coupling legal option sets.
The observed root prelude is part of the frozen action identity. No recorded
post-root hidden order survives step 3.

## Frozen Search

- root width: 64;
- stages: four;
- new samples per active root: `[4, 4, 8, 16]`;
- survivors: `[32, 16, 8, 1]`;
- total trajectories per group: 640;
- total production trajectories per arm: 63,360;
- tie-break: lower canonical action hash;
- root statistic: arithmetic mean leaf value;
- no early stopping;
- no confidence pruning;
- no transposition reuse;
- no candidate-rank state merge;
- no arm-specific tuning.

The primary and replay of an arm must produce the same scientific result ID.

## Frozen Leaf

Terminal:

- exact focal Card A base score, habitat bonuses disabled.

Nonterminal:

- exact current focal base score;
- plus frozen v4opp MLX remaining-value prediction;
- exact sparse feature extraction in Rust;
- batch rows in groups of at most 4,096;
- no training, dropout, ensemble, or calibration change.

## 121 And 441 Boundary

The v2 representation conclusion remains exact-R2 sparse occupied/frontier
state, consistent with the user's prior observation that a compact
approximately 121-cell envelope is sufficient in practice. The experiment
does not allocate or train a dense 441-cell v2 tensor.

The qualified legacy leaf uses historical sparse feature indices whose schema
contains 441 coordinate slots. That compatibility path is frozen solely to
hold leaf strength constant. It does not own:

- the simulator state;
- the public-belief identity;
- chance sampling;
- opponent branch selection;
- action representation;
- search bookkeeping;
- a learned v2 encoder.

No result may be interpreted as evidence for a 441-cell state representation.

## Frozen Integrity Checks

Each production report must prove:

- exact authorization, bundle, dataset, cohort, intent, panel, and model IDs;
- exact public-state replay for every group;
- exact complete-root application before future redeterminization;
- 64 reconstructed candidate hashes per group;
- 640 trajectories per group;
- one selected root per group;
- finite search values;
- hidden-order invariance for one fixed trajectory per group;
- invariance perturbation beginning only after the frozen root;
- exact opponent-decision and legal-option accounting;
- exact primary/replay scientific identity.

Partially complete resumable group artifacts are allowed only when their
frozen run identity matches exactly.

## Frozen Metrics

Per arm:

- mean and median retained R4800 regret;
- top-1 retained R4800-winner recall;
- R1200 pairwise accuracy;
- selected action hashes;
- trajectories and terminal leaves;
- MLX rows;
- opponent decisions and option counts;
- wall seconds.

Paired comparisons:

- A2 versus C0;
- A2 versus A0;
- A2 versus S3.

For each comparison:

- mean regret improvement;
- paired mean A2-minus-reference regret;
- game-clustered bootstrap 95% interval;
- selected-action agreement.

Bootstrap:

- source-game cluster;
- 20,000 replicates;
- seed `2026061721`;
- percentile interval at 2.5% and 97.5%.

## Frozen Success Gates

Every gate is conjunctive:

1. all four primary/replay pairs match exactly;
2. all reports are complete and fully accounted;
3. A2 regret improvement versus C0 is at least `0.05`;
4. A2 regret improvement versus A0 is at least `0.05`;
5. A2 regret improvement versus S3 is at least `0.05`;
6. each paired A2-minus-reference interval has upper bound below zero;
7. A2 top-1 recall is at least the maximum of C0 and A0;
8. A2 R1200 pairwise accuracy is no more than `0.005` below the maximum of C0
   and A0.

Pass classification:

`o1_public_belief_search_validation_passed`

Null classification:

`o1_public_belief_search_validation_null`

The selected arm is A2 only on a pass. There is no fallback selection.

## Frozen Cluster Schedule

Primary:

| Host | Role |
|---|---|
| john1 | `c0-primary` |
| john2 | `a0-primary` |
| john3 | `a2-primary` |
| john4 | `s3-primary` |

Replay:

| Host | Role |
|---|---|
| john2 | `c0-replay` |
| john3 | `a0-replay` |
| john4 | `a2-replay` |
| john1 | `s3-replay` |

Four production arms run concurrently in each wave. Preflight build,
authorization verification, artifact fanout, collection, and aggregation are
queue dependencies rather than manual conventions.

## Frozen Claim Boundary

Production validation remains open-data offline research. It cannot:

- open the sealed test;
- run gameplay;
- change the qualified player;
- claim a point gain;
- claim progress toward 100;
- authorize T2;
- authorize expert iteration;
- authorize a new leaf representation.

Only a complete validation pass permits a separately preregistered sealed-test
successor.

## Result

All eight production reports completed with exact primary/replay scientific
identity and complete accounting. A2 was directionally best but failed the
minimum-effect and mechanism-specific controls:

- versus C0: `0.034731` improvement, interval
  `[-0.048020, -0.026082]`;
- versus A0: `0.019902` improvement, interval
  `[-0.041931, 0.000000]`;
- versus shuffled A2: `0.004770` improvement, interval
  `[-0.015007, 0.010535]`.

Classification: `o1_public_belief_search_validation_null`.

No test or gameplay successor is authorized. See
`o1-public-belief-one-rotation-search-v1-result.md`.
