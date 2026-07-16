# Cascadia Rival implementation execution plan

**Date:** 2026-07-16

**Parent proposal:**
[Cascadia Rival finalized architecture](cascadia_rival_final_architecture_proposal_7_16.md)

**Campaign source of truth:**
[V3 README](docs/v3/README.md) and
[CAMPAIGN_STATE](docs/v3/CAMPAIGN_STATE.md)

**Implementation state:** the authorized CPU-only P0/P0.5/P1 engineering
machinery is implemented on isolated branch `feat/rival-cpu-machinery`; the
release-scale CPU-1 battery, production-policy integration, scientific
artifacts, seeds, jobs, and every accelerator phase remain unrun

**Current permit state:** `DENIED`

> **STATUS: CPU MACHINERY IMPLEMENTED -- RIVAL GPU EXECUTION IS NOT AUTHORIZED.**
>
> This document is not an experiment preregistration, launch authorization,
> queue instruction, or permission to inspect partial D1 output. It authorizes
> no Rival deployment, CUDA or MPS initialization, GPU benchmark, training,
> generation, gate, seed allocation, john0 access, queue or waiter creation,
> or modification of the existing D1 chain. Implementation commits never
> imply launch permission.

This document now records both the path and the completed first CPU engineering
batch. The implementation deliberately stops at the CPU-only and D1-dependent
wall. Every future GPU phase still requires a new, explicit user instruction,
a phase-specific machine-readable permit, and the ordinary campaign
preregistration and safety checks. The exact implemented contracts and honest
holds are documented in
[`RIVAL_CPU_MACHINERY.md`](docs/v3/RIVAL_CPU_MACHINERY.md).

## 1. Executive implementation decision

Do **not** start by writing CUDA kernels or training RivalNet. The first risk is
not model quality; it is whether the proposed high/low-fidelity estimator can
have enough finite-sample power within the local compute envelope while
preserving the exact incumbent estimand. The second risk is policy identity:
matching Cascadia rules is insufficient if a new implementation changes any
future transformer-plus-Gumbel action.

Proceed in this order:

```text
P0 identity and fail-closed contract
  -> P0.5 bounded-inference and power proof
  -> P1 canonical CPU reference harness
  -> HARD WAIT for the durable D1 boundary
  -> explicit PROBE permit
  -> P2 small production-policy premise probe
  -> P3 exact semantic compiler
  -> explicit BUILD permit
  -> P4 resident exact backend plus shared B_k integration/parity
  -> P5 bounded RivalNet bakeoff
  -> P6 Rival-MF calibration and coverage
  -> explicit SCIENCE permit
  -> P7 shadow and balanced one-seat instrument
  -> P8 one frozen relabel iteration
  -> P9 paired promotion evidence and absolute target battery
```

The first authorized implementation batch contained only P0, P0.5, and the
parts of P1 that do not modify a live D1 dependency. It ends with tested
CPU libraries, schemas, fixtures, a dry-run validator, and a parametric
theorem/power envelope. It adds no Rival accelerator dependency, launch path,
scientific seed, or strength claim.

### 1.1 Why this order changed from the research blueprint

The proposal's broad direction survives, but implementation review found four
places where a literal reading would be unsafe or wasteful:

1. A CPU pattern policy is useful for harness validation, but it is not the
   complete high-fidelity `B_k` policy. Production-fidelity evidence must use
   the frozen transformer-plus-Gumbel implementation after D1.
2. The proposed fixed Hoeffding interval may be powerless under a loose global
   Cascadia score range. Certified remaining-turn or stratum-specific bounds
   and an absolute sample-cost calculation must precede accelerator work.
3. The current exporter contains policy-critical behavior in a large private
   module. It must be extracted behavior-identically under golden trace tests,
   never copied into a parallel Rival implementation.
4. A resident accelerator backend would be new infrastructure, not a small
   bridge change. It is justified only after the terminal-improvement premise
   and complete-policy cost are measured.

## 2. Scope, non-goals, and fixed decision boundaries

### 2.1 Intended outcome

The implementation is successful only if it eventually produces an ordinary,
frozen, non-cooperative policy whose mean seat score is at least `100` over the
single registered 1,000-game, four-player, pre-habitat-bonus target battery.
Rival is a route to that policy, not a replacement definition of success.

### 2.2 Fixed policy class

Every acting seat maximizes only its own expected raw terminal score. The
implementation must reject rather than merely default away from:

- table-total or table-mean utility;
- sacrifice, donation, fairness, or coalition objectives;
- shared cross-seat policy memory;
- coordinated four-board plans or joint genomes;
- private hidden-state input to any policy;
- oracle access to the true hidden order in any production policy, label, or
  training artifact; and
- risk or quantile serving objectives in place of expected score.

Opponent public observations may inform the acting seat's own-value estimate.
They may not turn the objective into cooperation or minimax against an imagined
coalition.

A typed, offline T3 known-world oracle remains allowed for diagnostic upper
bounds. Its output is marked oracle-only and must be structurally barred from
serving, preference labels, and training views.

### 2.3 Explicit non-goals for the first batch

P0--P1 do not:

- choose a post-D1 base policy;
- claim current-rules strength;
- allocate experiment or verdict seeds;
- construct scientific root cohorts;
- contact john0 or any fleet host;
- start or import a CUDA/MPS runtime;
- implement a production resident accelerator;
- train or select a RivalNet;
- deploy a wrapper or change serving behavior;
- modify the D1 chain, queue, waiters, or inputs; or
- read a live arm's partial score.

The execution trust model assumes trusted operators and trusted local runners;
there is no direct attacker trying to corrupt an experiment. Fail-closed
checks target realistic campaign failures: implementation mistakes, stale or
mismatched artifacts, accidental seed reuse, crashes and partial writes,
resource runaway, and analyses that drift from preregistration. The machinery
does not claim to sandbox an administrator or arbitrary code already executing
inside the verifier process.

### 2.4 Values that remain hard-blocked

The following are intentionally not guessed in this plan:

| Field | Current value | Unblock condition |
| --- | --- | --- |
| `B_0` identity | `UNSET -- HARD BLOCK` | D1 reaches its frozen boundary and the ordinary incumbent is resolved. |
| Current-rules baseline `b` | `UNSET -- HARD BLOCK` | A complete admissible baseline battery exists. |
| Target gap `delta_b` | `UNSET -- HARD BLOCK` | Compute from the frozen definition, not a historical July-9 score. |
| Minimum recoverable headroom `h_min` | `UNSET -- HARD BLOCK` | Frozen after baseline and tomography design. |
| Practical appeal margin `epsilon` | `UNSET -- HARD BLOCK` | Frozen before scientific confirmation output. |
| Eligible strata | `UNSET -- HARD BLOCK` | Frozen from design/calibration roles without coverage peeking. |
| `n_H`, `n_L`, and `beta_cv` | `UNSET -- HARD BLOCK` | Bounded power calculation and disjoint calibration. |
| Absolute throughput bars | `UNSET -- HARD BLOCK` | P0.5 power envelope plus the later P2 cost probe. |
| Scientific seed blocks | `UNALLOCATED -- HARD BLOCK` | Post-D1 preregistration and seed-registry update. |

The parent proposal defines the buffered target gap as
`delta_b = 100.10 - b`. Retain that definition unless a deliberate methodology
ruling changes it; only its numeric value is hard-blocked. No rounded
historical score may fill these fields.

## 3. Repository architecture and ownership boundaries

Rival should be a typed protocol and optimization layer over the canonical
game, not a third rules engine.

```text
crates/cascadia-game
    canonical private/public state, chronology, transitions, scoring
              |
              v
crates/cascadia-sim
    CPU ranking/policy helpers and match simulation
              |
              v
crates/cascadia-rival                  NEW, workspace-integrated
    identities, observation boundary, RNG domains, compiler oracle,
    cohorts, ledgers, bounds, CPU tomography and reference orchestration
              |
              +-----------------------------+
              |                             |
              v                             v
crates/cascadia-v3-policy             Python rival package
    POST-D1 shared complete B_k core      estimator, power, coverage,
              |                           schemas and training views
              +---------------+-------------+
                              |
                +-------------+-------------+
                |                           |
                v                           v
real-root-exporter                future rival-engine
    bridge/I/O adapter only           exact backend only, DEFERRED
```

### 3.1 Canonical code that must be reused

| Concern | Canonical source | Rival use |
| --- | --- | --- |
| Rules and chronology | `crates/cascadia-game/src/game.rs` | Sole `GameState`, public projection, market prelude, legal actions, and hidden redetermination. |
| Exact scoring | `crates/cascadia-game/src/scoring.rs` | Sole score formulas; no Python or accelerator-first reimplementation is authoritative. |
| Board deltas | `crates/cascadia-game/src/board.rs` | Exact rollback, habitat analysis, and dependency discovery. |
| CPU simulation | `crates/cascadia-sim/src/lib.rs` | Reference games and cheap harness fixtures. |
| Current Gumbel policy | `cascadiav3/real-root-exporter/src/gumbel.rs` | Frozen high-fidelity search semantics and parity oracle. |
| Model bridge | `cascadiav3/real-root-exporter/src/model_bridge.rs` | Existing complete-model request path for later P2 evidence. |
| Current features | `cascadiav3/real-root-exporter/src/feature_tensors.rs` | Behavioral reference, not assumed to be Rival's optimal compiler. |
| Packed data | `cascadiav3/src/cascadiav3/expert_tensor_shards.py` | Immutable-shard and corpus conventions. |
| Ordinary model/trainer | `cascadiav3/src/cascadiav3/torch_cascadiaformer.py`, `cascadiav3/src/cascadiav3/torch_train_cascadiaformer.py` | Distillation target after refactoring shared utilities; RivalNet remains separate. |
| Provenance | `crates/cascadia-provenance/src/lib.rs` | Complete source and artifact identity. |
| Campaign methods | `docs/v3/INFRASTRUCTURE.md`, `docs/v3/RULES_CONTRACT.md` | Operational and rules-contract authority. |

The historical terminal policy-improvement code in
`crates/cascadia-search/src/policy_improvement.rs` may supply replay and
determinism test ideas. Its pattern policy and legacy one-sided interval are
not production `B_k`, not Rival-MF, and not funding evidence.

### 3.2 Implemented CPU tree and held future tree

```text
crates/cascadia-rival/
    Cargo.toml
    src/
        lib.rs
        observation.rs       # PrivateSimState -> PublicPolicyObs only
        identity.rs          # incompatible policy/artifact identities
        ruleset_identity.rs  # one structured research-ruleset owner
        action_id.rs         # versioned canonical action/menu hashes
        menu.rs              # root-specific legal/incumbent menu composer
        rng.rs               # explicit physical/policy/search/tie domains
        policy.rs            # FrozenPolicy accepts public input only
        scenario.rs          # independent-world reference sampling
        compiler.rs          # implemented dense oracle; optimized variants held
        bounds.rs            # certified score-difference bounds
        dynamic_urn.rs       # coupling proof/orchestration, not bag ownership
        terminal.rs          # forced action and terminal-pair protocol
        tomography.rs        # T0--T4 witnesses and certificates
        ledger.rs            # durable canonical records and replay hashes
    tests/
        cpu_reference_battery.rs  # 125 games / exactly 10,000 transitions
        rival_contract_cli.rs     # bounded verifier input contract

crates/cascadia-game/src/
    game.rs                  # POST-D1 P2b chance injection in draw/return paths
    types.rs                 # simulator-private physical ID types if selected
    lib.rs                   # gated chance API ownership/export
    coupled_chance.rs        # P2b sidecar/oracle, only if P0.5/P2a retain MF

crates/cascadia-v3-policy/   # created/extracted only after the D1 boundary
    Cargo.toml
    src/
        lib.rs
        root.rs              # ordered candidates and exact afterstates
        observation.rs       # public tokens/request construction
        features.rs          # pure feature tensors, packing, eval_row_key
        gumbel.rs            # one shared complete B_k policy core
        trace.rs             # canonical policy trace

cascadiav3/real-root-exporter/src/
    model_bridge.rs          # existing bridge implementation
    rival_adapter.rs         # future thin I/O wrapper over shared B_k core

cascadiav3/src/cascadiav3/rival/
    __init__.py
    schema.py               # additive schemas and canonical artifact I/O
    manifest.py             # externally byte/content-pinned root manifests
    cohorts.py              # allocation registry, cohorts, seed openings
    panel_plan.py           # exact preregistered S/H/L unit schedules
    multifidelity.py        # fixed estimator and coefficient artifact
    bounds.py               # consumes Rust range certificates only
    coverage.py             # registry-derived census and error ledger
    power.py                # symbolic work/memory envelope
    analysis.py             # complete-source-game clustered summaries
    appeals.py              # typed one-look appeal state machines
    appeal_journal.py       # immutable crash-reconstructible journal
    terminal_evidence.py    # pinned Rust verifier and receipt join
    training_view.py        # held-P8 sidecar and expert-data join
    preflight.py            # source-locked accelerator denial

cascadiav3/tests/
    fixtures/rival/
    test_cpu_test_guard.py
    test_rival_appeal_journal.py
    test_rival_bounds.py
    test_rival_schema.py
    test_rival_cohorts.py
    test_rival_multifidelity.py
    test_rival_coverage.py
    test_rival_panel_plan.py
    test_rival_power.py
    test_rival_preflight.py
    test_rival_terminal_evidence.py
    test_rival_training_view.py

cascadiav3/rival-engine/     # after relevant P2/(optional P3) gates + BUILD permit
    Cargo.toml
    src/                     # resident runtime against shared policy/backend traits
    cuda/                    # accelerator feature, disabled by default
```

The tree above records the implemented pre-GPU files. `cascadia-v3-policy`,
`rival-engine`, exporter extraction, optimized compiler implementations, and
trainer integration remain future/held paths; no placeholder `rival/policy.py`
or `rival/train.py` is claimed to exist.

Fixtures belong under `cascadiav3/tests/fixtures/rival/`; the superficially
similar `cascadiav3/fixtures/rival/` path is ignored by the repository.

If either new root crate is added, the same commit must add it to the root
Cargo workspace and to `SOURCE_ROOTS` in
`crates/cascadia-provenance/src/lib.rs`. Otherwise normal tests or the source
digest could silently omit it.

P0 also repairs the current provenance gap: workspace member
`crates/cascadia-api` is not in `SOURCE_ROOTS`. The scientific dependency-
closure digest must cover it, both new root crates, root `Cargo.toml`/lockfile,
and every path able to affect rules, features, policy, training, or serving.
Tests mutate a fixture file under each registered root and require the digest
to change.

`cascadiav3/rival-engine/Cargo.toml` will be a deliberately standalone nested
workspace, matching the exporter's repository boundary, and will contain its
own `[workspace]`. It is therefore built/tested explicitly and is not assumed
to be covered by root `cargo test --workspace`. Its sources remain covered by
the registered `cascadiav3` provenance root.

Feature contracts are fixed before these crates exist:

- `cpu-reference` enables only deterministic CPU code and is the only feature
  used before BUILD authorization; and
- `accelerator` is default-disabled, exists only in the deferred engine, and
  is rejected by every pre-GPU runner.

### 3.3 Extract, do not copy, exporter behavior

The current exporter keeps policy-critical behavior private in `main.rs`,
including candidate afterstates, determinizations, exact afterstate scores,
action IDs, public/action tokens, request packing, and bridge evaluation.
`feature_tensors.rs`, `pack_eval_request`, and `eval_row_key` are also part of
`B_k` identity; the shared policy crate must own their pure request-building
logic so the future engine cannot duplicate model inputs. Device transport and
the live model bridge remain exporter/backend adapters. Before extraction,
specify the trace schema and build synthetic/canonical CPU fixtures. After the
D1 boundary, record production golden traces for:

- `CandidateAfterstate`, including `apply_truncated`;
- `is_rollout_truncation_rule_error`;
- `complete_with_sampled_greedy`;
- the `LeafEvaluator` contract; and
- the private legacy exporter `RULESET_ID` meaning.

Then record:

- ordered `LegacyActionIdV0` values for the incumbent candidate menu plus both
  rules-legal and incumbent-menu hashes;
- serialized public observation hash;
- exact afterstate active-seat scores;
- normalized bridge requests and masks;
- model outputs within a frozen numerical tolerance;
- Gumbel candidates, visits, values, and backups;
- RNG draw counters by domain;
- refresh, exact-K1, tie, timeout, and fallback paths;
- empty-bag/empty-stack truncation, the partially staged state that is scored,
  and its `apply_truncated` flag;
- selected action; and
- final decision-trace hash.

The refactor passes only if every pre-existing integer/legacy-ID field and
selected action is identical. A future action divergence creates a new policy;
it may not be described as incumbent-equivalent. Whether truncated staging is
desirable is irrelevant to parity: changing it is a separately named policy
change.

The extraction moves one behavior-identical policy core into
`crates/cascadia-v3-policy`. The exporter and future engine both call that
core; the engine implements an exact execution backend, not a second Gumbel
policy. `crates/cascadia-rival` owns protocol types and CPU reference
orchestration, not a competing `B_k` implementation.

Because the game, sim, exporter, bridge, trainer, and shared policy code may
all be inside the in-flight D1 source closure, changes to any of them remain in
an isolated feature worktree and must not merge, deploy, or alter live inputs
until D1 has a durable boundary. Before that boundary, no trace generation may
require a production checkpoint, bridge, accelerator, or live artifact.

## 4. Immutable contracts to freeze before implementation expands

### 4.1 Incompatible policy identities

Define four machine-incompatible types:

| Type | Meaning | Permitted use |
| --- | --- | --- |
| `B_k` | Ordinary frozen transformer-plus-Gumbel base | High-fidelity continuation and ordinary serving. |
| `pi_L` | Complete cheap low-fidelity policy | Proposal/control variate only; never called `B_k`. |
| `W_k` | Rival shadow or one-seat research instrument | Research evidence only; no symmetric serving or promotion entrypoint. |
| `M_(k+1)` | Ordinary model distilled from one frozen Rival tranche | The only Rival-produced gameplay promotion candidate. |

Substitution must fail at deserialization and again at launch validation.
Every identity binds:

- `cascadia_game::RULES_SEMANTICS_ID` and one shared structured
  `ResearchRulesetIdentity`;
- source revision and complete source digest;
- executable, model manifest, checkpoint, and weights SHA-256;
- bridge protocol, tensor schema, numerical mode, and precision;
- every Gumbel/search/refresh/exact-endgame knob;
- action-ID and menu-hash versions;
- physical, policy, redetermination, search, and tie RNG contracts;
- public-observation and policy-memory schema versions;
- timeout, incomplete-unit, OOM, and fallback behavior; and
- compiler, simulator, sampler, and candidate-generator identities.

`ResearchRulesetIdentity` has one owner in `cascadia-rival` and binds the
legacy `RULESET_ID` label, `RULES_SEMANTICS_ID`, and a canonical hash of
serialized `GameConfig::research_aaaaa(4)`. The exporter and engine import it;
they never duplicate the string literal.

`table_total`, `table_native_q`, true-hidden peeking, and model fallback must be
explicitly rejected for Rival identities. Absence of the field is not proof of
`false` unless the schema makes the default canonical and validates it.

### 4.2 Public/private information boundary

The Rust API should make leakage difficult to express:

```text
PrivateSimState(GameState)
    -> public_observation(seat, seat_local_memory)
    -> PublicPolicyObs {
           state: PublicGameState,
           supply: PublicSupply,
           seat,
           memory
       }
    + canonical RulesLegalMenu / frozen IncumbentCandidateMenu
    + opaque HonestWorldSampler capability
    -> FrozenPolicy::act(obs, menu, world_sampler, PolicyRng)
```

No policy trait accepts `&GameState`. `PublicSupply` is part of the observation
identity and cannot be omitted merely because it is separate from
`PublicGameState`. Compile-fail tests or doctests must prove a policy cannot
receive `PrivateSimState`.

The canonical engine exposes legality primitives, not one root-independent
complete menu: free-refresh choices, paid wipes, and
`legal_turn_actions(&MarketPrelude)` are separate. A typed root-specific
`MenuComposer` combines those canonical primitives, validates every composed
action through the canonical transition, and emits `RulesLegalMenu`. A policy
may rank a supplied menu but may not reimplement legality.

The complete `B_k` search controller still needs private determinizations on
which to execute exact rules. `PublicGameState + PublicSupply` does not contain
the exact unseen `TileId` multiset, so a trusted simulator constructs
`HonestWorldSampler` from `PrivateSimState`, immediately merges and
redetermines the hidden partitions with an inner RNG, and exposes only opaque
`PolicyWorld` handles. No method exposes the original suffix/partition,
scenario key, or true `PrivateSimState`. Search may apply canonical rules to a
`PolicyWorld`; model kernels receive only public tensors, legal masks, and
seat-local memory. Metamorphic, capability, and compile-fail tests prove that
the original hidden order/partition cannot influence an action except through
the honest sampled public-information distribution.

There are four distinct memory instances. Mutating one seat's memory must not
alter another seat's state, observation, or action distribution.

### 4.3 Root, menu, and chronology contract

Rival v1 appeals only at a `DraftPolicyRoot`, captured after the public market
prelude completes and before the active seat chooses the draft/action. The
complete incumbent also has a distinct `PreludePolicyRoot` for its free-three-
of-a-kind accept/decline decision. `B_k` continuation parity must reproduce
both root kinds even though v1 does not appeal the prelude root.

A trace must distinguish:

```text
public prelude decision -> chance/reveal -> post-prelude policy root
-> chosen action -> wildlife return if applicable -> public refill/reveal
```

Accept/decline three-of-a-kind, paid wipes, chained automatic wipes, wildlife
returns, and refills may not be reordered to simplify batching. Root identity
includes root kind and chronology version.

Keep two menus distinct:

- `RulesLegalMenu`: every action the canonical rules accept; and
- `IncumbentCandidateMenu`: the exact ordered candidates the frozen `B_k`
  actually considers after its policy-specific generation/pruning.

Canonical legality tests use the first. Policy trace parity and Rival root
identity bind both hashes and use the second for incumbent comparisons. If the
current incumbent omits a rules-legal paid-wipe action or treats a prelude
choice separately, parity preserves that behavior. Adding an omitted action is
a new candidate-generator and policy identity, never a transparent fix.

### 4.4 RNG-domain contract

At minimum, derive explicit domains for:

1. source-game and root selection;
2. outer physical coupling group and chance-event index;
3. incumbent/challenger inner branch randomness;
4. low/high inner fidelity randomness;
5. future acting seat;
6. public-policy redetermination;
7. Gumbel/search noise;
8. policy sampling; and
9. tie-breaking.

Inside paired H, one registered outer physical coupling group may intentionally
span high/low and incumbent/challenger evaluations. Branch, fidelity, seat,
redetermination, search, policy, and tie RNGs remain independently keyed.
Outside an explicit coupling group, outer physical keys are independent. This
separation permits common physical randomness without letting an outer
scenario leak through reused inner policy seeds.

The current `GameState::redeterminize_hidden` preserves the state's original
seed while shuffling hidden order, and some rule/policy helpers also derive
randomness from state seed. That behavior is not asserted to be a production
bug, but it is insufficiently explicit for Rival's outer/inner independence
contract. Rival therefore owns a domain-separated RNG interface and tests it
against the canonical engine.

### 4.5 Versioned action and root identity

The current action ID derives from SHA-256 over serialized `TurnAction` JSON.
Moving code or changing serialization can therefore rewrite scientific
identity. Preserve it byte-for-byte as `LegacyActionIdV0` for incumbent golden
traces, then add three distinct Rival identities:

- `ActionContentId`: only canonical ordered action fields, enum discriminants,
  rules identity, and content-schema version;
- `PublicRootId`: acting seat, root kind, chronology version, complete public
  observation including `PublicSupply`, and root-schema version; and
- `RootActionOccurrenceId`: `PublicRootId`, ordered menu hash,
  `ActionContentId`, action index, and occurrence-schema version.

The same legal action content therefore retains one content ID across roots,
while each scientific occurrence remains bound to its exact root and menu.

Migration fixtures emit and compare `LegacyActionIdV0` and the new additive
IDs. Incumbent parity requires the legacy ID to remain identical; the new IDs
are not misrepresented as pre-existing trace fields. Any intentional legacy ID
change creates a new policy identity and version; it is never silently
normalized.

### 4.6 Artifact schemas

Register additive, fail-closed schemas rather than overloading v4 Q/value
fields:

```text
cascadiav3.rival_policy_identity.v1
cascadiav3.rival_root_manifest.v1
cascadiav3.rival_terminal_pair_ledger.v1
cascadiav3.rival_bound_certificate.v1
cascadiav3.rival_power_envelope.v1
cascadiav3.rival_gpu_permit.v1
cascadiav3.rival_preference_shard.v1
cascadiav3.rival_training_view.v1
cascadiav3.rival_terminal_panel_plan.v1
cascadiav3.rival_coefficient_calibration.v1
cascadiav3.rival_potential_root_census.v1
cascadiav3.rival_error_family_ledger.v1
cascadiav3.rival_allocation_registry.v1
```

Register these additively in `cascadiav3/src/cascadiav3/schema.py`.

`cascadiav3.rival_preference_shard.v1` contains:

- root hash and complete ordered action/menu hashes;
- incumbent and exactly one selected challenger index;
- categorical preference (`challenger_over_incumbent` or `unlabeled`) and
  `preference_valid`;
- one preregistered `preference_weight`;
- activation stratum and natural-frequency weight;
- root-cohort role and `S/H/L/A` panel identities;
- coefficient, allocation, bound, and error-ledger identities;
- parent manifest, raw-root, and raw-world ledger hashes; and
- no observed advantage magnitude unless an independent `A` panel exists.

Sampling probability and training loss weight are different fields. Neither
may be inferred from the other.

## 5. Cohorts, panels, and statistical units

Rival has two independent allocation axes, governed together by one externally
byte/content-pinned allocation registry. A root manifest binds exactly one
root-cohort assignment and requires `complete_game_seed_role = null`;
promotion/target complete-game assignments do not masquerade as root fields.
The registry owns both axes, their commitments, and their disjointness.

### 5.1 Root cohorts

- design/tomography;
- coefficient calibration;
- untouched coverage;
- shadow/one-seat;
- relabel selection.

Roots from one source game are clustered observations for aggregate analyses.
Duplicating roots from a game must not shrink uncertainty as though they were
independent.

Promotion and target are separate complete-game seed roles, with one complete
game as the scientific unit. They are not root cohorts and cannot share seeds
with each other or with any earlier complete-game role.

The registry's `root_source_set_sha256` binds the sorted exact
`(root_id, source_game_id, cohort_role)` projection without creating a
manifest/census/error-ledger cycle. A potential-root census carries the same
projection as `source_root_set_sha256` plus the exact
`allocation_registry_identity`, and its eligible root IDs must equal the
registry-derived family universe. Before any run, typed `u64` root and
complete-game seed openings must rehash to every commitment; realized values,
not just commitment strings, must be globally disjoint across both axes.
Any decision-grade source-game-clustered summary must consume that externally
pinned registry and exactly cover the selected family's registered roots and
root-to-source-game mapping; truncated or reassigned rows fail closed.

### 5.2 Within-root panels

| Panel | Role | Inference use |
| --- | --- | --- |
| `S` | Select exactly one challenger. | Discarded for confirmation inference. |
| `H` | Paired observations of `D_H = G_H(a) - G_H(a0)` and `D_L = G_L(a) - G_L(a0)` under registered physical coupling. | High-fidelity estimand plus paired control variate. |
| `L` | Extra `D'_L = G'_L(a) - G'_L(a0)` observations whose panel is independent of H; its two action branches may use registered action-pair coupling. | Estimates the same frozen low-fidelity expectation. |
| `A` | Fresh post-admission high-fidelity observations targeting `Delta_H`. | Disabled in v1. Required before magnitude regression. |

With `A.disabled`, labels are categorical and use a fixed preregistered weight.
Enabling `A` creates a new experiment, preregistration, and training-target
identity. It does not require a new architecture proposal because the parent
proposal already permits this branch.

The validator rejects:

- root overlap between calibration and coverage;
- domain-key reuse outside the coupling matrix;
- any H/L access before S freezes one challenger;
- more than one v1 confirmation challenger;
- a changed candidate mixture, policy, sampler, coefficient, range, or bound;
- missing, duplicate, or unaccounted world rows;
- reused promotion or target seeds; and
- quantitative training targets when `A` is disabled.

A recorded timeout is a completed failed unit for operational accounting. It
remains in throughput, completion, and activation denominators; it makes that
root's fixed panel incomplete and prevents a label. It is never inserted into
an inferential mean unless a preregistered bounded fallback supplies a valid
score.

## 6. P0 -- identity, schema, and fail-closed contract

**Authorization:** the separate CPU-only implementation instruction was
received and executed on the isolated feature branch. Accelerator execution
remains unauthorized.

**Estimated engineering effort:** 3--5 focused engineer-days.

### 6.1 Work

- Add the workspace-integrated `cascadia-rival` crate with no accelerator
  dependency under default features.
- Add all policy, root, panel, ledger, preference, and permit schemas.
- Add `cascadia-rival` to the provenance source roots.
- Implement incompatible `B_k`, `pi_L`, `W_k`, and `M_(k+1)` identities.
- Implement versioned canonical action/menu/root identities.
- Implement the root-kind-aware `MenuComposer` over canonical prelude, wipe,
  and turn-action legality primitives.
- Implement `PrivateSimState -> PublicPolicyObs` and isolated seat memory.
- Implement the domain-separated RNG key derivation contract.
- Add an identity perturbation suite changing every bound field one at a time.
- Add a phase-readiness checker whose default result is `DENIED`.
- Correct the stale July-9 rules statement in
  `docs/v3/TRAINING_PIPELINE.md` before any Rival scientific run, while
  preserving historical artifact labels as historical.
- Add a CPU-only dry-run CLI that rejects `cuda`, `mps`, and `auto`; only an
  explicit `--device cpu` is accepted in pre-GPU phases.

### 6.2 Acceptance tests

- A policy cannot compile when passed private game state.
- Hidden-order or hidden-partition permutations that preserve the entire
  public observation, including `PublicSupply`, do not alter
  `PublicPolicyObs`, tensors, masks, memory transitions, or pre-reveal action
  distribution when public history and policy RNG match.
- A legitimate public reveal may change the action.
- Mutating one seat's memory leaves all other seat memories bit-identical.
- Changing only other seats' terminal scores, while preserving the acting
  seat's terminal score and causal future, cannot change Rival utility or its
  label.
- No policy input contains scenario ID, physical seed, hidden inventory, or
  event priority.
- No Rival label schema contains table-total or table-mean utility.
- Every missing or substituted identity field fails closed.
- Generic JSON admission uses a stable no-follow descriptor and an inclusive
  64 MiB ceiling; pinned scientific JSON additionally requires canonical,
  single-link bytes and an explicit external file hash.
- The phase checker rejects absent, expired, wrong-revision, wrong-command,
  wrong-preregistration, and over-budget permits.
- The P0 preflight uses only the Python standard library and leaves `torch`
  absent from `sys.modules`.
- Later CPU-only entrypoints invoke no CUDA/MPS availability, synchronization,
  allocation, or device-query API; a test-level guard fails on any such call.

### 6.3 Exit artifacts

- JSON schemas and locked golden fixtures;
- CPU-only source-provenance test;
- identity/fail-closed-contract and nonanticipativity report from fixtures;
- `rival_preflight --validate-only` returning `DENIED` without a permit; and
- an identity field matrix documenting every hash and owner.

No strength conclusion is possible at P0.

## 7. P0.5 -- bounded inference and power proof

**Authorization:** CPU implementation and synthetic/enumerable tests only,
after a separate implementation instruction.

**Estimated engineering effort:** 5--8 focused engineer-days.

### 7.1 Fixed estimator

For selected action-difference observations, implement exactly:

```text
mu_hat = mean(D_H - beta_cv * D_L_on_H)
       + mean(beta_cv * D_L_on_L)
```

When `D_L` inside H and `D'_L` inside L have the same frozen law and variance,
and H and L are independent panels, the parent proposal's special-case
population optimum is:

```text
beta_cv = n_L / (n_H + n_L) * Cov(D_H, D_L) / Var(D_L)
```

Here covariance is measured between `D_H` and `D_L` inside H. Implement the
general covariance/variance optimizer and reduce to this formula only after
tests certify those assumptions for the registered design; otherwise use the
general minimizer or reject the design. The coefficient is eventually frozen
on a disjoint calibration cohort. Negative correlation is permitted
mathematically; instability, zero low-fidelity variance, or a
calibration/coverage identity mismatch fails closed.

### 7.2 Fixed first-generation bound

For certified high- and low-fidelity score-difference widths `R_H` and `R_L`,
test and document the general conservative widths:

```text
X   = D_H - beta_cv * D_L
Y   = beta_cv * D'_L
R_X = R_H + abs(beta_cv) * R_L
R_Y = abs(beta_cv) * R_L
```

The parent proposal's `(1 + abs(beta_cv)) * R_D` form is used only when one
certified common width satisfies `R_D >= max(R_H, R_L)`.

The one-sided error ledger must satisfy:

```text
delta_H + delta_L <= delta_root
sum(delta_root(r) for r in every potentially eligible appeal) <= delta_game
```

Maintain separate multiplicity families for the finite training corpus and the
actual one-seat instrument. Perform exactly one inferential computation after
the complete fixed H/L panels exist. Missing H/L rows invalidate the root and
emit no label; recorded failures remain only in throughput, completion, and
activation denominators unless a preregistered bounded fallback yields a valid
score.

### 7.3 Certified range work

Derive score-difference bounds from current rules, not observed samples. Begin
with the global valid range, then tighten it by remaining personal turns and
registered root stratum. A tighter bound is admissible only if it is proven
never to underestimate the exact reachable difference.

For tractable late states, exhaustively enumerate the reachable endpoint range
and compare it with the certified bound. For larger states, use independently
checked monotone relaxations or branch-and-bound certificates. A loose or
unclosed bound is inconclusive.

### 7.4 Test matrix

- `beta_cv = 0` exactly reduces to the high-fidelity mean.
- Swapping incumbent/challenger labels negates the result.
- Positive, zero, stable negative, and shifted covariance cases.
- Exactly enumerable finite distributions prove algebraic unbiasedness.
- General variance formula agrees with enumerated variance.
- Invalid counts, NaN, zero variance, overlap, or incomplete panels reject.
- Row and sample ordering do not change the result.
- Calibration/test identity changes invalidate `beta_cv`.
- Repeated peeking and pre-completion result access reject.
- Hand-computed Hoeffding and error-ledger fixtures match exactly.
- Exact finite fixtures never exceed the declared undercoverage budget.
- For non-enumerable deterministic fixtures, derive the replication count from
  a preregistered material-undercoverage tolerance and confidence level. The
  one-sided binomial upper bound must clear that fixture-specific tolerance;
  finite simulation checks implementation, not the theorem.
- Cluster fixtures prove that duplicating roots inside a source game does not
  produce an iid-root standard error.

### 7.5 Symbolic power calculator and unresolved envelope

Produce a **parametric, non-funding** `rival_power_envelope.json` with, at
minimum:

- certified range by stratum;
- candidate and error-family counts;
- `n_H`, `n_L`, and calibration requirements over a useful grid;
- required roots and complete terminal pairs;
- optimistic, central, and pessimistic roots/hour assumptions;
- memory assumptions;
- symbolic post-D1 john0 GPU-hours with measured-cost fields explicitly
  `UNRESOLVED`; and
- sensitivity to covariance, activation frequency, target gap, and timeout.

P0.5 cannot pass, fund, or kill the program from assumed covariance,
throughput, strata, or target gap. It proves the calculator and maps symbolic
no-go regions. P2a supplies high-fidelity cost and P2b supplies coupled proxy
covariance; together they produce the first decision-grade 3,000-hour verdict
for whether to fund P3,
not a production Rival-MF qualification. Closure requires a defensible lower
bound on required work above the cap; funding requires a preregistered
conservative envelope inside it. A guessed optimistic rate does neither.

Replacing Hoeffding with empirical Bernstein, an e-process, or a confidence
sequence requires a separately reviewed theorem, coverage implementation, and
new identity; it is not a tuning change.

## 8. P1 -- canonical CPU reference harness

**Authorization:** CPU implementation and deterministic fixtures only, after a
separate implementation instruction. Changes inside any D1 source-closure
dependency remain unmerged until the D1 boundary.

**Estimated engineering effort:** 10--20 focused engineer-days.

### 8.1 Work

- Wrap canonical `GameState`; do not recreate rules.
- Capture v1 post-prelude `DraftPolicyRoot` appeals while separately tracing
  `PreludePolicyRoot`; store rules-legal menus and exercise the incumbent-menu
  schema with typed proxy fixtures. Production `B_0` candidate menus remain
  post-D1 policy-trace work.
- Apply a forced incumbent or challenger first action through canonical rules.
- Continue all four selfish seats with isolated public observations/memory.
- Store immutable source-game, root, action, world, and terminal-pair ledgers.
- Replay every record from hashes and reconstruct the complete trajectory.
- Implement independent high/low physical worlds as the marginal/parity
  reference with `beta_cv = 0`; this mode is not called Rival-MF.
- Implement S/H/L/A panel state machines and coupling-matrix validation.
- Implement T0--T4 tomography APIs with explicit `exact`, `best_found`, and
  certified-bound result types.
- Define the semantic compiler trait and a dense full-recomputation oracle
  only. Optimized component and incremental implementations are P3 work and
  do not begin until P2b funds them.

CPU pattern policies may validate chronology, pairing, replay, and estimator
plumbing. They must be labeled proxy policies and cannot produce a high-
fidelity T1a claim or a Rival funding verdict.

### 8.2 Rules, chronology, and replay parity gate

PR-sized gate:

- 100 complete legal games;
- 10,000 randomized reachable transitions;
- every curated rules boundary; and
- zero mismatch.

Pre-GPU release gate:

- 10,000 complete legal games;
- 1,000,000 randomized reachable transitions; and
- zero mismatch in legal menus, transitions, terminal detection, scoring
  categories, total score, conservation, or replay hashes.

These finite registered suites are strong property/replay evidence, not a
claim of universal equivalence. P3 separately proves compiler conformance on
its registered domain; P4 owns accelerator transition and full-game parity.

Every zero-mismatch claim names both sides:

- Rival wrapper state bytes/hash versus direct canonical `GameState`;
- `GameState::transition` versus clone plus `apply`;
- visitor-based action enumeration versus materialized legal actions for the
  same `MarketPrelude`;
- root-specific `MenuComposer` output versus canonical prelude/wipe primitives
  with every composed action accepted by canonical transition;
- preview/fast afterstate paths versus transactional application;
- dense semantic score caches versus `scoring::score_game`; and
- durable ledger replay versus the source trajectory's state/action hashes.

Curated fixtures include:

- decline/return before refill;
- accept/reveal/draft chronology;
- paid and chained automatic wipes;
- wildlife return and redraw;
- Nature transactions;
- forced actions and exact-K1;
- radius-6 and exact overflow;
- Bear regrouping, Elk alternatives, long/branching Salmon, Hawk isolation,
  Fox diversity, and habitat merges.

The P1 dense semantic oracle receives exact fixture and serialization tests,
but comparison with component-local and incremental implementations belongs to
P3.

### 8.3 Existing regressions that become hard contracts

Retain and extend these existing behaviors:

- `public_afterstate_never_exposes_the_hidden_refill`;
- `hidden_redeterminization_preserves_every_public_game_fact`;
- `public_supply_counts_only_publicly_unseen_resources`;
- `unplaced_drafted_wildlife_returns_before_end_of_turn_refill`;
- `legal_generator_only_emits_accepted_complete_turns`;
- `fast_board_preview_matches_the_transactional_transition`;
- `search_is_deterministic_given_seed`;
- `gumbel_policy_can_accept_or_decline_free_three_of_a_kind`;
- `exact_final_personal_turn_bypasses_model_and_search`;
- `triple_root_market_decision_cannot_observe_actual_replacement_order`;
- `search_never_observes_true_hidden_order`;
- `rollout_stream_seed_contracts`;
- `determinized_rollouts_never_observe_true_hidden_order`;
- `golden_rollout_labels_are_stable`;
- `gumbel_selfplay_records_roundtrip_into_v4_shard`;
- `packed_eval_request_roundtrips_feature_arrays`;
- `action_relation_tail_fast_path_matches_reference`; and
- `ledger_replay_reconstructs_policy_game_trajectory`.

Also retain bridge-pipeline, bridge-ensemble, throughput-knob, v4-shard,
existing six-rotation, and overflow round-trip tests. The exporter remains a
separate Cargo workspace and must always be tested explicitly.

### 8.4 Dynamic-urn coupling admission

Independent worlds are the safe reference but will generally provide little
or no useful low/high covariance. Production Rival-MF therefore depends on a
proof-gated common-physical-randomness construction.

Do not let a CPU pilot qualify production coupling. Dynamic-urn admission
requires:

- unique physical item IDs and exact conservation;
- a fresh event index for every sequential draw;
- returned wildlife never reusing an exposed key;
- a universal conditional-uniformity argument for the construction;
- exhaustive enumeration of every eligible subset and priority permutation
  for canonical small bags through eight physical items;
- proof of exact marginals in all high/low by incumbent/challenger branches,
  plus deterministic randomized property coverage on larger reachable states;
- independent-world replication as a control; and
- no policy access to physical IDs, outer keys, or event priority.

This cannot be implemented honestly by shadowing the private bags in
`cascadia-rival`: current bag/draw/return state is private to `GameState`, and
wildlife lacks a unique physical-token ID. If Rival-MF survives P0.5 and P2a,
add a post-D1 P2b simulator-private `ChanceOracle`/physical-ID hook
inside `cascadia-game`. This requires explicit integration in `game.rs` draw,
wipe, refill, return, and transition paths; `types.rs` owns any physical IDs;
`lib.rs` owns the gated API; and `cascadia-rival::dynamic_urn` owns only the
coupling proof, key schedule, and orchestration. The normal canonical sampler
remains the oracle. The new hook receives no public-policy exposure and earns
admission only through identical-marginal and conservation tests. Maintaining
a parallel external bag is prohibited.

Prefer an ephemeral `CoupledGameState` sidecar so ordinary `GameState`
serialization and the default sampler remain byte-identical. If correct
integration requires adding serialized state, explicitly bump
`STATE_SCHEMA_VERSION`, migrate replay fixtures, create a new simulator/policy
identity, and document every canonical-byte/hash change. A `serde(skip)` field
or silent replay-hash drift is not acceptable.

If this proof or its executable tests fail, Rival-MF closes. The project may
retain a high-fidelity Anchor path, but it may not present independently drawn
low-fidelity values as a useful paired control variate.

### 8.5 P1 exit

P1 exits only with zero registered rules/chronology/replay mismatch,
deterministic durable replay, a working CPU-only panel/cohort validator, and no
accelerator-capable runner. It still produces no current `B_k`
terminal-improvement claim.

## 9. Mandatory D1 wait checkpoint

No Rival scientific experiment begins while D1 is live or ambiguous. The
local durable docs read for this plan report that john0 was unreachable and D1
attempt 5 liveness was unknown at their last update. This planning turn did
not recheck the host, read partial output, or touch the chain.

Later work may cross the checkpoint only after durable documentation proves:

- D1 reached its frozen boundary;
- the promoted or retained ordinary incumbent was resolved;
- complete `B_0` identity is reproducible;
- a current-rules absolute baseline exists;
- the actual gap to 100 is computed under the frozen target definition; and
- fresh, disjoint seed roles can be registered.

If the resolved ordinary policy already reaches 100 on a valid completed
1,000-game target battery, stop Rival: the campaign objective is achieved.

## 10. P2 -- split production-policy premise probes

**Authorization:** blocked. Each subphase requires its own later explicit
`PROBE` instruction, post-D1 preregistration, registered fresh roles, and
hash-pinned Permit A capability. P2a never authorizes P2b.

**Estimated engineering effort:** 7--12 engineer-days plus separately capped
probes. A reasonable planning cap is at most 15 john0 GPU-hours for P2a and 10
for P2b, never more than 25 total, excluding any separately required canonical
baseline.

### 10.1 P2a -- high-fidelity and independent-world premise

Use tractable late roots to establish:

- an executable unilateral terminal-improvement witness;
- complete `B_k` terminal-pair cost by remaining-turn band;
- natural activation frequency inside a preregistered eligible late-game
  stratum sampled from natural source games;
- a high-fidelity-only work envelope; and
- an independent-world low-fidelity reference fixed at `beta_cv = 0`.

P2a makes no useful-covariance or Rival-MF claim because the exact coupled-
chance hook does not yet exist. It may fund a high-fidelity-only Anchor branch,
close the terminal premise, or authorize CPU construction/proof of P2b. It
cannot fund RivalNet or qualify production multifidelity inference.

P2a closes or narrows the relevant branch if a frozen rule triggers, including:

- a conservative bound on `delta_b` exceeds an independently measured
  executable-headroom upper bound under game-clustered uncertainty;
- complete `B_k` continuation cannot fit the high-fidelity work envelope; or
- a preregistered one-sided upper envelope rules out useful changed-trajectory
  one-seat improvement, or activation economics cannot support a label
  tranche.

### 10.2 P2b -- exact-coupling proof and proxy covariance

Only if P0.5 and P2a retain Rival-MF:

1. implement the canonical-engine `ChanceOracle`/physical-ID hook on CPU;
2. complete the universal marginal proof and registered exhaustive/property
   tests from §8.4;
3. freeze one explicitly named proxy-pilot S distribution; and
4. under a new Permit A subphase, measure coupled H/L covariance, cost, and
   equal-wall precision on that distribution.

RivalNet does not yet exist. P2b therefore applies only to its proxy generator
and is a feasibility falsifier; it cannot qualify production Rival-MF. P6 must
recalibrate on the final P5 selected-challenger distribution.

P2b closes the multifidelity branch if the chance proof fails, the proxy S
distribution cannot be reproduced without leakage, or measured covariance and
cost cannot improve equal-wall precision under the preregistered one-sided
uncertainty rule. It may fund CPU P3, not P4/P5 or a scientific arm.

### 10.3 Interpretation limits

Neither subphase is a strength gate, training run, simulator funding blank
check, or permission for the next phase. Do not multiply activation frequency
by mean root gain and call the result a game-score gain. One-deviation effects
are non-additive. P2 activation estimates apply only to the named late-game
stratum and do not extrapolate to the full game without a separately valid
sampling/aggregation argument.

T4 may close the global program only if it supplies a trajectory-wide,
natural-frequency aggregate bound across all eligible decisions. A bound on a
small late-root subset cannot close or fund the full policy by extrapolation.

## 11. P3 -- exact semantic compiler

**Authorization:** CPU work may begin only after P2b funds the multifidelity
branch. Accelerator benchmarking remains blocked until a BUILD permit. A
recorded high-fidelity-only Anchor branch skips this RivalNet compiler phase
unless it independently demonstrates a need for it.

**Estimated engineering effort:** 15--25 engineer-days.

### 11.1 Work and selection rule

- Inventory every scoring, board, supply, chronology, and action dependency.
- Create stable integer feature IDs and legal-action hyperedges.
- Implement dense full recomputation as oracle.
- Implement component-local recomputation.
- Implement incremental update/invalidation.
- Measure p50, p95, and maximum invalidation fanout.
- Measure complete-trajectory cost, not isolated evaluator calls.

For every transition, dense, component-local, and incremental compiler outputs
must have exact feature-ID/value equality. Every changed dense feature must be
present in the incremental invalidation set.

The proposal uses D6 symmetry language. The repository must not assume
reflection support from rotation support. Test all 12 dihedral transforms:

- transform/inverse round trips;
- legal-menu equivariance;
- `transform(apply(s,a)) == apply(transform(s), transform(a))`;
- exact score invariance;
- feature and action-pointer equivariance; and
- overflow parity.

Select the fastest exact implementation. Do not assume an NNUE-style delta
path wins. Zero feature mismatch, proven full-dihedral/overflow correctness,
and complete rules coverage are mandatory before speed is considered.

## 12. P4 -- resident exact backend and shared `B_k` integration

**Authorization:** blocked. Requires a later BUILD instruction, exact source
revision, bounded engineering permit, and P0--P2a acceptance. The
multifidelity/RivalNet branch additionally requires P2b and P3 acceptance; a
registered high-fidelity-only branch does not pretend those gates passed.

**Estimated engineering effort:** 30--60 engineer-days; highest-risk phase.

Implement as three independently stoppable slices:

1. exact batched backend with the canonical Rust engine as oracle;
2. integration with the one shared transformer-plus-Gumbel `B_k` core; and
3. throughput, memory, queue, timeout, and OOM envelope.

One resident owner process may use persistent buffers and wavefront queues.
The accelerator remains a backend, never the new rules authority.

Policy parity includes packed rows, masks, numerical mode, candidates, model
refreshes, Gumbel values/visits/backups, exact-K1, future redeterminizations,
ties, RNG draws, timeouts, and fallback at every future public decision for
all four seats. Parity is trace-level, not mean-score similarity.

Exit requires:

- zero transition/game mismatch on the registered suite;
- identical integer state/action traces;
- floating outputs within frozen tolerances with identical actions;
- complete terminal-pair throughput inside the P0.5/P2 absolute envelope;
- safe p99 memory/queue behavior; and
- fail-closed OOM, timeout, or bridge failure.

Any action divergence means the backend implements a new policy, not `B_k`.

## 13. P5 -- bounded RivalNet bakeoff

**Authorization:** blocked until P4 passes and a training permit is issued.

**Estimated engineering effort:** 15--25 engineer-days plus authorized
training.

Freeze at most three shapes:

1. summary-only;
2. component-augmented; and
3. component plus a small global-correction path.

All share one output contract: legal action logits, own score-to-go auxiliary,
terminal action-preference/difference auxiliary, and optional disagreement or
calibration output. RivalNet is a separate model class, not a loose collection
of CascadiaFormer flags.

Architecture selection uses a dedicated development/selection cohort. That
cohort is disjoint from P6 coefficient calibration and untouched coverage and
may never choose shapes, coefficients, or strata from P6 results.

Select one or none using:

- complete-trajectory rate;
- confirmed-challenger recall on a frozen gold subset;
- held-out low/high terminal-difference covariance;
- equal-wall multifidelity precision; and
- memory/queue stability.

The existing repository result that smaller CUDA models reached only roughly
1.9--2.8x speedup is adverse prior evidence. No fourth architecture search is
allowed after the frozen family fails.

## 14. P6 -- Rival-MF calibration and coverage

**Authorization:** blocked until P5 selects one low-fidelity policy and the
multifidelity route remains active.

**Estimated engineering effort:** 10--15 engineer-days plus authorized
calibration.

Use disjoint coefficient-calibration and untouched-coverage root cohorts. The
calibration generator must reproduce the exact selected-challenger
distribution. Freeze, by stratum:

- `beta_cv`, `n_H`, and `n_L`;
- coupling and RNG identities;
- score-difference range and lower bound;
- error allocation;
- timeout and incomplete-unit handling; and
- activation and fallback rules.

Weak/unstable covariance, marginal-distribution failure, undercoverage, or
insufficient absolute throughput closes an individual stratum. It may not be
rescued by peeking at another coefficient or interval on the same coverage
cohort.

The existing `run_paired_gate.sh` is not reused here. It initializes CUDA and
implements paired complete-game score inference; S/H/L root-level
multifidelity inference has different units, independence, boundedness, and
multiplicity.

If P5 selects no low-fidelity policy, or Rival-MF closes, skip P6 entirely. A
separately identified high-fidelity-only branch may retain Anchor: it sets
`beta_cv = 0`, has no L panel or multifidelity claim, and receives its own
fixed-sample power, bounded-inference, artifact, and gameplay preregistration.

## 15. P7 -- shadow and balanced one-seat instrument

**Authorization:** blocked. Requires a new SCIENCE instruction and permit.

**Estimated engineering effort:** 5--10 engineer-days plus authorized
compute.

`W_k` has no symmetric-serving or promotion entrypoint. First run a shadow
instrument that changes no actions. Shadow evidence may pass only frozen
activation, completion, margin-survival, and runtime bars; it cannot establish
score gain.

Before any one-seat output exists, freeze complete-game sample size, balanced
seat rotations, paired seed construction, practical margin, game-level CI or
sanctioned sequential rule, maximum appeals per game, and the per-game error
ledger. Then run fresh balanced-seat games with one `W_k` research seat and
three fixed incumbents.

Funding a label tranche requires natural-frequency activation and actual
active-seat own-score improvement. Counterfactual root gains, throughput,
validation loss, or table-total changes are insufficient.

One-seat games test changed-trajectory local composition. They do not establish
that four simultaneous copies compose safely; only later symmetric gameplay of
the ordinary model can test that claim.

## 16. P8 -- one frozen relabel iteration

**Authorization:** blocked until P7 produces positive one-seat evidence and a
later explicit TRAIN instruction plus phase-specific permit is issued.

**Estimated engineering effort:** 10--20 engineer-days plus authorized
training.

- Freeze broad, hard, and bounded stress-case root exposures.
- Select one challenger in S and discard S from confirmation inference.
- Confirm on fresh H/L panels for the admitted multifidelity branch, or fresh
  H-only panels under the separately identified high-fidelity branch; the two
  artifact identities are non-substitutable.
- Emit categorical labels with a fixed weight because A is disabled in v1.
- Preserve broad-data exposure floors and cap hard-root influence.
- Add the preference target behind an explicit default-off trainer flag.
- Require flag-off bit identity with the existing trainer.
- Train exactly one ordinary `M_(k+1)` candidate under one frozen checkpoint
  rule.
- Recheck ordinary serving-contract integrity plus wrapper forced-anchor and
  fallback parity. `M_(k+1)` is expected to differ from the incumbent.

The current `ExpertTensorShard` v1--v4 loader has no Rival preference schema,
and the current trainer has no per-record policy weight. Do not overload those
fields. Implement:

- `RivalPreferenceShard` parsing/validation in
  `cascadiav3/src/cascadiav3/rival/training_view.py`;
- a hash-checked derived join onto immutable expert roots using root, ordered
  menu, and action-occurrence identities;
- an explicit optional `preference_target`, `preference_valid`, and
  `policy_weight` in the collated training view; and
- a default-off loss path in
  `cascadiav3/src/cascadiav3/torch_train_cascadiaformer.py`.

Tests must round-trip the new shard, reject stale/misaligned menus, mask all
quantitative targets when A is disabled, prove that sampling weight does not
replace loss weight, preserve v1--v4 loading, and establish flag-off bit
identity for batches, losses, optimizer steps, and checkpoints.

No wrapper, shadow instrument, or RivalNet checkpoint is eligible for
promotion. Only the ordinary `M_(k+1)` artifact proceeds.

## 17. P9 -- gameplay promotion and absolute target

**Authorization:** blocked pending a later explicit GATE/TARGET instruction,
phase-specific permit, and scientific preregistration. Champion promotion
remains reserved to John.

1. Run the valid at-least-100-pair promotion gate, or the currently sanctioned
   preregistered group-sequential equivalent, on fresh complete games.
2. Stop for John's promotion decision.
3. Permit a second and final Rival iteration only after positive gameplay
   evidence and remaining work-budget review.
4. Freeze one ordinary policy.
5. Touch the single 1,000-game absolute target block once, with all four seats
   using that one frozen `M` policy. This is not an `M`-versus-`B` battery.

A CI-positive promotion result and a true mean of at least 100 are separate
claims. Neither root-level evidence nor promotion against a weaker base proves
the absolute target.

## 18. First implementation batch as reviewable changes

These were the reviewable implementation slices. Slices 1--8 are implemented
and CPU-tested on the isolated feature branch. Slice 10's reusable validator is
implemented, but its release-scale battery remains explicitly unrun. Slices 9
and 11 remain held at their later evidence boundaries.

### Slice 1 -- contract skeleton

**Files:** new `crates/cascadia-rival`; root `Cargo.toml` and `Cargo.lock`;
`crates/cascadia-provenance`; Python package/schema fixtures; a CPU-only test
guard; `cascadiav3/tests/test_trainer_perf_knobs.py`; and
`cascadiav3/tests/test_bridge_throughput_knobs.py`.

**Done when:** the CPU-only crate imports; provenance covers the current
workspace member `crates/cascadia-api` plus every new path; changing a file in
each new crate changes the scientific source digest; all schemas reject
missing/unknown identity; the test guard rejects/skips MPS tensor creation and
CUDA device requests/queries; and no newly added Rival code has an accelerator
dependency.

### Slice 2 -- policy identities and permit-deny validator

**Files:** `identity.rs`, `rival/schema.py`, `manifest.py`, `preflight.py`.

**Done when:** all four policy types are non-substitutable, each identity field
has a perturbation rejection test, and a missing permit returns `DENIED` before
any device library import.

### Slice 3 -- public observation, seat memory, and RNG domains

**Files:** `observation.rs`, `rng.rs`, `policy.rs` and compile-fail/metamorphic
fixtures.

**Done when:** private state is unrepresentable at the policy boundary,
four-seat memory isolation is proven, `HonestWorldSampler` exposes only
redetermined opaque worlds, and hidden/outer-key metamorphics pass.

### Slice 4 -- canonical action/root identity

**Files:** `action_id.rs`, `menu.rs`, `ledger.rs`, migration fixtures.

**Done when:** action-content, public-root, and root-action-occurrence IDs each
round-trip to a versioned encoding; menu/root kind is bound only at the
occurrence layer; deliberate serialization perturbations are detected; and
replay hashes are deterministic. Root-kind menu composition validates every
prelude/wipe/turn combination through canonical transitions.

### Slice 5 -- cohort and artifact state machine

**Files:** `terminal.rs`, `rival/cohorts.py`, `appeals.py`, schema tests.

**Done when:** S freezes exactly one challenger before H/L access, all overlap
and key-reuse cases reject, A is structurally disabled, and incomplete units
cannot emit a label.

### Slice 6 -- multifidelity algebra and error ledger

**Files:** `multifidelity.py`, `coverage.py`, `analysis.py`.

**Done when:** enumerable-unbiasedness, bounds, multiplicity, clustering, and
power-derived per-fixture coverage checks pass.

### Slice 7 -- certified bounds and power envelope

**Files:** Rust `bounds.rs` authority, Python `rival/bounds.py` certificate
consumer, `power.py`, and exact mini-state fixtures.

**Done when:** every certified range dominates exhaustive truth on tractable
states and the program produces a reproducible parametric
optimistic/central/pessimistic grid with measured-cost fields unresolved. It
maps no-go regions but cannot fund or close the real program before P2a/P2b
supply the required measurements.

Rust is the only rules-aware reachability/range authority. It emits a
hash-pinned certificate; Python verifies and consumes that certificate for
interval algebra. Python never independently derives a Cascadia score bound.

### Slice 8 -- CPU chronology, replay, and tomography reference

**Files:** `scenario.rs`, `terminal.rs`, `tomography.rs`, `ledger.rs`, and the
dense-only `compiler.rs` oracle.

**Done when:** canonical game transitions, durable replay, T0--T4 result types,
and PR-scale rules/chronology suites pass. Proxy policies remain visibly typed
as proxies.

The first authorized batch is Slices 1--8, and those slices are implemented.
The optimized-compiler, release-scale, and exporter work below retain their
separate status rather than being inferred from the PR-sized tests.

### Slice 9 -- semantic compiler parity implementations, held for P3

**Files:** component/incremental `compiler.rs`; Rust/Python conformance
fixtures; and, because reflections do not currently exist, explicit transforms
in `crates/cascadia-game/src/hex.rs`, `board.rs`, `game.rs`, plus action and
rotation handling.

**Done when:** after P2b funds the work, dense/component/incremental exact
parity, invalidation completeness, all 12 symmetries, and overflow cases pass
at PR scale.

### Slice 10 -- release-scale CPU audit, runner implemented; battery unrun

**Files:** the implemented `validate_rival_pre_gpu.sh`, future release-battery
driver/receipt, artifact manifest, test documentation, and this completion
ledger.

**Done when:** the million-transition/10,000-game CPU release suite is clean,
all expected artifacts hash-verify, no Rival accelerator feature exists or can
be selected by the pre-GPU runner, and the project stops at the BUILD
authorization wall.

The ordinary pre-GPU validator is green, but Slice 10 is not done because the
10,000-game/1,000,000-transition battery and its durable engineering receipt
have not run. That battery is independently runnable on CPU; it is not blocked
on P3's optimized compiler implementations and cannot answer the production
policy premise.

### Slice 11 -- exporter extraction, deliberately held for the D1 boundary

Before D1's durable boundary, prepare only trace schemas and synthetic or
canonical CPU fixtures in an isolated worktree. Do not generate traces that
require a production checkpoint, bridge, accelerator, or live artifact. After
the boundary, capture production golden traces, extract behavior identically,
and rerun the exporter plus complete workspace suites. The extraction updates
root `Cargo.toml`/`Cargo.lock` for `cascadia-v3-policy` and the exporter's
independent `Cargo.toml`/`Cargo.lock` for its path dependency in one focused
change.

## 19. Pre-GPU acceptance ladder

| Gate | Required evidence | Result permitted |
| --- | --- | --- |
| CPU-0 contract | Schemas, provenance, CPU-only boundary, identity perturbation suite. | Continue CPU implementation only. |
| CPU-1 reference | Curated rules/chronology suite; 1,000,000 transitions; 10,000 games; overflow; zero leakage/replay mismatch. | Consider the canonical CPU harness complete. |
| CPU-2 inference | Coupling proof if MF retained; estimator algebra; exhaustive unbiasedness; bounded coverage; certified ranges; symbolic power grid. | Validate the calculator; the real power verdict remains `UNRESOLVED`. |
| CPU-3 P2a readiness | D1 resolved; complete `B_0`; `b`/`delta_b`; high-fidelity P2a sample/budget; fresh probe roles; validate-only pass. | Request Permit A/P2a; still no GPU. |
| PROBE-2a authorization | Explicit P2a instruction; safe current john0 status; exact-revision CPU gates; committed P2a preregistration; Permit A/P2a. | Run only the bounded high-fidelity/independent probe. |
| MEASURED-2a premise | Complete P2a ledger; terminal-pair cost, headroom/activation bounds, high-fidelity work envelope. | Close/narrow, retain a high-fidelity branch, or authorize CPU P2b proof work. |
| CPU-3b coupling | P2b chance hook, universal proof, exhaustive small-bag/property tests, frozen proxy S distribution. | Request Permit A/P2b; still no GPU. |
| PROBE-2b authorization | Explicit P2b instruction; new roles; safe current status; committed P2b preregistration; Permit A/P2b. | Run only the bounded coupled-covariance probe. |
| MEASURED-0 premise | Complete P2a/P2b ledgers; measured cost/covariance/activation in named strata; decision-grade 3,000-hour envelope. | Close/narrow multifidelity or fund CPU P3. |
| CPU-4 compiler | P2b funds the route; dense/component/incremental parity; all 12 symmetries; overflow; complete-trajectory CPU economics. | Request a BUILD permit; still no accelerator build. |
| BUILD-0 authorization | Explicit P4 instruction; exact-revision gates; bounded engineering preregistration; Permit B. | Build/measure only the named exact backend. |

No gate is passed by code review alone. No later gate waives an earlier one.

## 20. GPU permit and launch interlock

A future permit is a machine-readable, hash-pinned capability, not a prose
checkbox. Suggested `cascadiav3.rival_gpu_permit.v1` fields:

```text
phase
source_revision
source_digest
allowed_command_hashes
preregistration_sha256
rules_identity
policy_identity
seed_role_hashes
maximum_gpu_hours
maximum_wall_time
artifact_root
expires_at
john_approval_reference
```

The runner validates the permit before importing torch, loading a CUDA shared
library, probing `nvidia-smi`, or contacting a remote host. Any absent,
expired, mismatched, or over-budget field exits nonzero and writes no partial
scientific result. It does write a durable non-scientific denial audit with the
permit hash, time, requested phase, and fail-closed reason.

Permits are deliberately separate:

- **Permit A -- PROBE:** exactly one named P2a or P2b late-root subphase, with
  separately hashed commands/roles and caps totaling at most 25 john0
  GPU-hours. P2a never authorizes P2b. No training, engine build, gate, or
  serving.
- **Permit B -- BUILD:** accelerator-dependent P3 benchmarking and P4 only.
  Ordinary CPU P3 compiler work needs no GPU permit. No scientific panels,
  training, or gameplay gate.
- **Permit C -- SCIENCE:** exactly one named P5, P6, or P7 training,
  calibration, or instrument activity. No promotion or target battery.
- **Permit D -- TRAIN:** exactly one named P8 Rival relabel/training iteration.
  No gameplay gate, promotion, or target battery.
- **Permit E -- GATE/TARGET:** exactly the named P9 complete-game gate or target
  battery, never both unless the permit and preregistration separately identify
  both stages and the promotion stop between them.
- **Promotion authority:** John's existing reserved authority and a fresh
  preregistration; never implied by A--E.

The planning and CPU implementation turns did not run `campaign_status.sh`, `ssh john0`,
`run_paired_gate.sh`, a queue/waiter script, or any command containing
`--device cuda`, `--device mps`, or an auto-selecting device mode. CPU-only
implementation does not need them. Immediately before a later remote/GPU
phase, project rules require a fresh read-only `campaign_status.sh` safety
check; that check occurs only under the new phase instruction and must confirm
that no live job will be displaced.

## 21. CPU-only verification commands

These commands define the engineering gate used by the authorized CPU
implementation. The implementation receipt below records the commands that
were actually run. Every invocation keeps the explicit CPU device contract.

```bash
git diff --check
```

```bash
env CUDA_VISIBLE_DEVICES="" \
  RUSTC="$HOME/.cargo/bin/rustc" \
  RUSTDOC="$HOME/.cargo/bin/rustdoc" \
  "$HOME/.cargo/bin/cargo" check --workspace
```

```bash
env CUDA_VISIBLE_DEVICES="" \
  RUSTC="$HOME/.cargo/bin/rustc" \
  RUSTDOC="$HOME/.cargo/bin/rustdoc" \
  "$HOME/.cargo/bin/cargo" test --workspace
```

```bash
env CUDA_VISIBLE_DEVICES="" \
  RUSTC="$HOME/.cargo/bin/rustc" \
  RUSTDOC="$HOME/.cargo/bin/rustdoc" \
  "$HOME/.cargo/bin/cargo" test \
  --manifest-path cascadiav3/real-root-exporter/Cargo.toml
```

```bash
env CUDA_VISIBLE_DEVICES="" \
  RUSTC="$HOME/.cargo/bin/rustc" \
  RUSTDOC="$HOME/.cargo/bin/rustdoc" \
  "$HOME/.cargo/bin/cargo" test -p cascadia-rival \
  --no-default-features --features cpu-reference
```

After P4 source exists, its CPU reference is tested separately:

```bash
env CUDA_VISIBLE_DEVICES="" \
  RUSTC="$HOME/.cargo/bin/rustc" \
  RUSTDOC="$HOME/.cargo/bin/rustdoc" \
  "$HOME/.cargo/bin/cargo" test \
  --manifest-path cascadiav3/rival-engine/Cargo.toml \
  --no-default-features --features cpu-reference
```

```bash
env -u PYTORCH_CUDA_ALLOC_CONF \
  CUDA_VISIBLE_DEVICES="" \
  CASCADIA_DEVICE=cpu \
  CASCADIA_CPU_ONLY_TESTS=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=cascadiav3/src \
  .venv/bin/python -m unittest discover \
  -s cascadiav3/tests -p 'test_rival_*.py' -v
```

The full Python discovery command is admitted to this CPU-only block only after
`CASCADIA_CPU_ONLY_TESTS=1` is implemented as a checked guard that skips or
rejects all CUDA/MPS-specific tests, including real MPS tensor creation. It
also requires a Python 3.12 environment with Torch installed; the lightweight
development `.venv` intentionally does not satisfy that optional test
dependency.

```bash
test -x "${CPU_TORCH_PYTHON:?set CPU_TORCH_PYTHON to Python 3.12 with Torch}"
env -u PYTORCH_CUDA_ALLOC_CONF \
  CUDA_VISIBLE_DEVICES="" \
  CASCADIA_DEVICE=cpu \
  CASCADIA_CPU_ONLY_TESTS=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=cascadiav3/src \
  "$CPU_TORCH_PYTHON" -m unittest discover -s cascadiav3/tests -v
```

```bash
env CUDA_VISIBLE_DEVICES="" \
  CASCADIA_DEVICE=cpu \
  CASCADIA_CPU_ONLY_TESTS=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=cascadiav3/src \
  .venv/bin/python -m cascadiav3.rival.coverage \
  --fixtures cascadiav3/tests/fixtures/rival \
  --coverage-design cascadiav3/tests/fixtures/rival/coverage_design.json \
  --device cpu \
  --out cascadiav3/reports/rival_pre_gpu/coverage_report.json
```

```bash
env CUDA_VISIBLE_DEVICES="" \
  CASCADIA_DEVICE=cpu \
  CASCADIA_CPU_ONLY_TESTS=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=cascadiav3/src \
  .venv/bin/python -m cascadiav3.rival.preflight \
  --fixture cascadiav3/tests/fixtures/rival/preflight_fixture.json \
  --device cpu \
  --validate-only
```

```bash
env CUDA_VISIBLE_DEVICES="" \
  CASCADIA_DEVICE=cpu \
  CASCADIA_CPU_ONLY_TESTS=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=cascadiav3/src \
  .venv/bin/python -m cascadiav3.rival.cohorts validate \
  --manifest cascadiav3/tests/fixtures/rival/panel_manifest.json \
  --require-panels S,H,L \
  --require-disjoint calibration,coverage \
  --require-a-disabled
```

The high-fidelity-only branch uses a separate manifest and validator mode that
requires `S,H`, forbids `L`, fixes `beta_cv = 0`, and rejects a multifidelity
claim.

The validator script itself receives a syntax check:

```bash
bash -n cascadiav3/scripts/validate_rival_pre_gpu.sh
```

### 21.1 Executed implementation receipt (2026-07-16)

The authorized batch was verified in the isolated
`feat/rival-cpu-machinery` worktree. All Python commands that could reach
Torch set `CUDA_VISIBLE_DEVICES=""`, `CASCADIA_DEVICE=cpu`, and
`CASCADIA_CPU_ONLY_TESTS=1`; the full suite used the repository's existing
Python 3.12/Torch environment via
`/Users/johnherrick/cascadia/venv/bin/python`.

| Executed gate | Result |
| --- | --- |
| `bash cascadiav3/scripts/validate_rival_pre_gpu.sh` | PASS: expected default denial plus 80 Rust library, 2 binary, 1 PR battery, 3 CLI, 2 doc, 4 provenance, 10 CPU-guard, 13 bridge, 15 trainer, and 153 Rival Python tests; Clippy and Ruff passed |
| `cargo check --workspace` | PASS; one pre-existing `cascadia-api` dead-code warning |
| `cargo test --workspace` | PASS: 310 active tests; one pre-existing ignored timing harness |
| `cargo test --manifest-path cascadiav3/real-root-exporter/Cargo.toml` | PASS: 68 tests |
| full guarded Python discovery | PASS: 550 tests, 48 intentional skips, 116.175 seconds |
| static/format gates | PASS: scoped Rustfmt, Ruff check/format, shell syntax, `git diff --check`, and temporary-marker scan |

The 125-game PR battery exercised exactly 10,000 deterministic randomized
reachable transitions with zero mismatches. The full Python suite uncovered
and fixed an order-dependent preflight assertion and a non-hermetic dependency
on an ignored queue example; that parser case now uses tracked test data, not
an operational queue.

No GPU/MPS/CUDA discovery or initialization, remote command, campaign-status
query, scientific seed allocation, run launch, training, generation, gameplay
gate, or partial-result read occurred. D1 was not queried. CPU-1 release scale,
production `B_k`, P2/P3, trainer activation, GPU work, and all scientific
claims remain held exactly as specified above.

## 22. Durable artifact and evidence layout

Use the campaign's durable-first convention:

```text
docs/v3/preregistrations/
    rival_<identity>.md       # tracked canonical prose
    rival_<identity>.json     # tracked canonical machine contract

cascadiav3/reports/rival_<identity>/
    manifest.json
    preregistration_link.json # canonical tracked paths + SHA-256
    preregistration.md        # optional ignored convenience copy
    preregistration.json      # optional ignored convenience copy
    raw_games/
    root_ledgers/
    world_ledgers/
    parity_report.json
    coverage_report.json
    power_report.json
    throughput_report.json
    verdict.md
```

`cascadiav3/reports/` is ignored and therefore cannot be the canonical
preregistration location. The tracked files under
`docs/v3/preregistrations/` are committed and pushed before launch; the report
manifest hash-links them and verifies any convenience copy byte-for-byte.

No summary is publishable unless every expected row exists and every referenced
SHA-256 verifies. Stale raw directories reject. Partial output cannot emit a
verdict. Raw ledgers are the primary artifact; watchers and summaries are not.

When scientific work is eventually authorized:

- preregister before data exists;
- allocate fresh disjoint seed roles in `docs/v3/INFRASTRUCTURE.md`;
- log every run, including failed or invalid runs, immediately in
  `cascadiav3/EXPERIMENT_LOG.md`;
- update `RESEARCH_LOG`, `CAMPAIGN_STATE`, and the README at each material
  direction/state transition; and
- commit and push preregistration, seed-registry, and durable state docs before
  the launch they describe.

Ordinary unit tests are engineering verification, not strength evidence.
Empirical CPU/GPU tomography, calibration, benchmarks, and game runs are
scientific artifacts and receive the full provenance/logging treatment.

## 23. Ownership and review separation

The same person or agent may fill multiple roles at different times, but no
role self-approves its prohibited decision.

| Role | Owns | Must not approve |
| --- | --- | --- |
| Identity steward | Schemas, provenance, phase/permit checker | Scientific strength. |
| Rules owner | Canonical state, transitions, scoring | Statistical coverage. |
| Serving owner | One shared `B_k` policy core and trace adapters | Candidate promotion. |
| Statistics owner | Estimator, bounds, power, coverage, error ledger | Model selection. |
| Model owner | Compiler, RivalNet, trainer | Coverage exceptions. |
| Experiment operator | Preregistration, seed registry, durable artifacts | Champion promotion. |
| Independent correctness reviewer | Contract audit, mismatch/kill decision | Implementation shortcuts. |
| John | Rules-design rulings, promotion, explicit execution authority | None delegated by this plan. |

Required reviews before GPU-0:

1. rules/nonanticipativity review;
2. policy-trace identity review;
3. statistical theorem and coverage review;
4. provenance/artifact replay review;
5. systems memory/timeout/fail-closed review; and
6. independent failure-mode review of every active kill criterion.

## 24. Failure-mode risk register

| Risk | Earliest decisive test | Required response |
| --- | --- | --- |
| D1 already reaches 100 | Valid completed 1,000-game target result | Stop Rival. |
| Fresh baseline makes the gap too large | Post-D1 baseline plus P2 headroom | Close or narrow; do not invent additive effects. |
| Hoeffding is powerless | P0.5 symbolic grid plus P2a/P2b measured envelope | Close/redesign inference before engine work. |
| Full `B_k` terminal pairs are unaffordable | P2a late-root cost probe | Retain only a bounded offline route or close. |
| Proxy correlation is weak/unstable | P2b and disjoint P6 coverage | Disable failing strata or close MF. |
| Dynamic coupling changes marginals | Universal proof plus finite P1 regression tests | Close MF; do not weaken the test. |
| Exporter refactor changes policy | Golden trace extraction test | Revert/fix; divergence is a new policy. |
| Incremental features miss dependencies | Dense/component/incremental parity | Fix root cause; no tolerance. |
| RivalNet is fast but uncorrelated | P5 complete-trajectory/gold tests | Select none; no fourth model. |
| Candidate selection invalidates inference | S/H/L state-machine tests and coverage | Reject the artifact/run. |
| One-deviation gains fail to compose | P7 balanced one-seat games | Do not fund labels. |
| Rare labels wash out or damage broad play | P8 retention and gameplay gate | Reject `M_(k+1)`. |
| Promotion gain still misses 100 | P9 absolute battery | Report honestly; promotion is not target success. |
| GPU utilization rises but science slows | P4 complete-unit throughput/p99 queue | Stop backend work despite attractive utilization. |

No risk is answered by a validation-loss improvement, an isolated kernel
speedup, a busy GPU, or an unregistered post hoc subgroup.

## 25. Effort, critical path, and parallelism

Planning estimates, not commitments:

| Phase | Engineer-days | Critical dependency |
| --- | ---: | --- |
| P0 | 3--5 | None; CPU only. |
| P0.5 | 5--8 | P0 identity skeleton. |
| P1 | 10--20 | P0 contracts; canonical rules. |
| P2 | 7--12 plus two bounded subphase probes | Durable D1 and separate Permit A capabilities. |
| P3 | 15--25 | P2b funds compiler work. |
| P4 | 30--60 | P3 exact path or registered high-fidelity branch, plus Permit B. |
| P5 | 15--25 plus training | P4 economics. |
| P6 | 10--15 plus calibration | P5 selected policy. |
| P7 | 5--10 plus games | P6 or the high-fidelity branch, plus Permit C. |
| P8 | 10--20 plus training | Positive P7 evidence. |
| P9 | Experiment wall time | Valid ordinary candidate. |

A complete successful path is roughly 100--200 engineer-days before
scientific wall time. Agent parallelism can reduce calendar time for schemas,
fixtures, statistics, and compiler work, but cannot parallelize away evidence
dependencies. The exact-engine plus shared-policy integration is the critical
engineering path.

Safe early parallel lanes after CPU implementation authorization:

- identity/schema/provenance;
- estimator/bounds/power;
- public-boundary/RNG/property fixtures; and
- canonical rules/compiler test corpus.

They merge only through the shared locked schemas. Exporter extraction and all
GPU-dependent lanes remain held at their explicit walls.

## 26. Definition of ready and definition of done

### 26.1 Ready to begin CPU implementation

- [x] Explicit user instruction to implement, distinct from this plan-only
  request.
- [x] Clean, synchronized isolated named worktree confirmed.
- [x] No edit overlaps another session's files.
- [x] P0 slice and CPU-only constraint recorded in the working plan.
- [x] No remote/GPU command included in the active task.

### 26.2 First batch done

- [x] P0 identities, schemas, observation boundary, RNG, and default-deny
  preflight pass.
- [x] P0.5 estimator, certified bounds, error ledger, synthetic coverage, and
  symbolic/grid power tooling pass as engineering artifacts; the real power
  verdict remains `UNRESOLVED` until post-D1 P2 measurements.
- [x] P1 CPU chronology, replay, cohort, nonanticipativity, and dense
  semantic-oracle fixtures pass, including a deterministic 125-complete-game /
  exactly 10,000-randomized-reachable-transition PR battery (exceeding the
  100-game minimum).
- [ ] The separate 10,000-game/1,000,000-transition release-scale CPU-1
  battery passes. It was intentionally not run and no CPU-1 claim is made.
- [x] Every exercised fixture/artifact hash and source digest verifies.
- [x] No accelerator feature or dependency is enabled in newly added Rival
  code, and no GPU/remote action occurred.
- [x] Every implemented change inside the D1 source closure remains unmerged
  until the durable D1 boundary. Exporter/shared-policy extraction did not
  occur.
- [ ] README and implementation status are current and committed; if D1 is
  still live, implementation commits are pushed only to the isolated feature
  branch and are not merged into `main`.
- [x] Work stops at the D1/authorization wall.

### 26.3 Whole program done

- [ ] All active premises passed without waived kill tests.
- [ ] One ordinary frozen `M` policy survived valid paired gameplay evidence.
- [ ] John made any promotion decision explicitly.
- [ ] The one registered 1,000-game target battery was touched once.
- [ ] Mean seat score is at least 100 under the exact frozen current-rules,
  four-player, pre-habitat-bonus identity.
- [ ] Complete provenance, raw ledgers, hashes, and verdict are durable.

## 27. Implementation handoff

The authorized CPU batch is complete on the isolated feature branch and stops
at the intended wall. It is not merged or deployed. The next scientifically
meaningful action is not more proxy engineering: it is the durable D1
boundary, followed only by a new explicit P2 instruction, a phase-specific
permit, a fresh read-only campaign safety check, preregistration, and actual
production-policy premise measurements. CPU-1 release scale may be run as a
separate engineering battery, but it cannot answer the production premise.

The correct operational state is:

```text
Rival CPU machinery: IMPLEMENTED ON ISOLATED FEATURE BRANCH
Rival release-scale CPU-1 claim: UNCLAIMED
Rival GPU permit: DENIED
Scientific seeds: UNALLOCATED
D1 chain: UNTOUCHED
```
