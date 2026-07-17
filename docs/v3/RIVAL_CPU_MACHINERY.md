# Cascadia Rival CPU machinery

**Date:** 2026-07-16
**Scope:** pre-GPU experiment contracts, canonical CPU reference execution,
artifact verification, fixed-panel inference, and training-data joins
**Scientific status:** engineering machinery only; no strength result

This document describes the Cascadia Rival machinery that exists on the
isolated CPU implementation branch. It is the implementation companion to the
[Rival execution plan](../../cascadia_rival_implementation_plan_7_16.md), not a
replacement for that plan's phase gates.

The most important boundary is simple:

> A deterministic CPU proxy can prove that the plumbing is honest. It cannot
> prove that a move is better for the production transformer-plus-search
> policy, cannot emit a scientific preference label, and cannot establish a
> path to a 100-point mean score.

No GPU, MPS, remote host, experiment seed, scientific root cohort, training
run, generation run, gameplay gate, or live-campaign process is created by
this machinery. The D1 boundary remains external and unresolved here.

## 1. What is implemented

The branch now contains five implemented contract layers:

1. A Rust CPU-reference crate, [`cascadia-rival`](../../crates/cascadia-rival/README.md),
   that wraps canonical `cascadia-game` transitions and scoring while owning
   Rival identities, public-policy boundaries, RNG domains, immutable replay,
   proxy terminal pairs, Rust-authored bounds, and typed tomography evidence.
   As of 2026-07-16 (build scope WI-2) the tomography types are backed by two
   working CPU optimizers plus a driver: `tomography_repack.rs` (T0 static
   own-seat repacking, deterministic threshold annealing over the frozen
   realized multiset), `tomography_replay.rs` (chronology-preserving
   hindsight-placement beam replay, emitted at the
   `KnownExogenousChanceTape` boundary as kind T3 — labeling by information
   boundary, not by the scope's positional "T1" heading), and the
   `rival-tomography` bin, which consumes a directory of sealed terminal
   trajectory ledgers and emits one deterministic
   `cascadiav3.rival_tomography_summary.v1` document.  Every witness is
   engine-reverified and every summary carries
   `witness_semantics = "lower_bound_only"`; the evidence-domain enum gained
   a validated `incumbent_measured` variant gated on the `incumbent:` policy
   namespace plus complete policy decision traces, so proxy-population
   diagnostics can never be presented as incumbent measurements.
2. A Python contract package,
   [`cascadiav3.rival`](../../cascadiav3/src/cascadiav3/rival/__init__.py), that
   strictly validates externally pinned manifests, root-source and seed
   allocation registries, panel plans, certificates, evidence receipts,
   census-derived error budgets, fixed estimators, power grids, appeal
   journals, and training-sidecar joins.
3. A strict cross-language evidence boundary: Rust replays and authenticates
   terminal-pair semantics; Python rejoins every preregistered identity field
   to an externally pinned manifest and panel unit while separately checking
   the Rust-authenticated outcome and file/executable authorities.
4. An explicit CPU-only execution guard and a source-locked, default-deny
   future accelerator permit checker.
5. An immutable preference training view whose data contract can be tested on
   CPU while the trainer integration remains deliberately disabled at P8.

These layers are not presented as one active label pipeline. The current Rust
adapter emits proxy evidence only, while the preference sidecar is a separate
CPU-tested contract for a future production-evidence adapter. No proxy appeal
can be converted into a training label.

The root workspace registers the Rust crate in
[`Cargo.toml`](../../Cargo.toml), and the scientific source digest includes
both `crates/cascadia-api` and `crates/cascadia-rival` through
[`cascadia-provenance`](../../crates/cascadia-provenance/src/lib.rs). Rival's
additive schemas are registered without overloading existing replay or expert
tensor formats in
[`cascadiav3/schema.py`](../../cascadiav3/src/cascadiav3/schema.py).

## 2. System architecture

The trust flow is deliberately one-way:

```text
canonical cascadia-game state, rules, transitions, scoring
                         |
                         v
            Rust CPU-reference contracts
      public observation / menus / identities / RNG
                         |
                         v
       replay-complete proxy terminal-pair ledger
                         |
          mandatory expected pair + parent pins
                         v
          Rust semantic replay and receipt sealing
                         |
            exact-shape, hash-bound receipt
                         v
 Python manifest + panel-plan + receipt semantic join
                         |
             typed CPU-proxy evidence row
                         v
 immutable appeal journal and one-look state machine
                         |
             audit result, never a proxy label
```

Python does not declare a Cascadia score range, deserialize arbitrary JSON
into a trusted terminal row, or bless a receipt merely because its digest
matches. Rust does not choose statistical error spending, select roots, or
decide whether an observed effect is scientifically admissible. Each language
owns the part it can validate most directly.

### 2.1 Canonical rules remain singular

[`crates/cascadia-rival/src/compiler.rs`](../../crates/cascadia-rival/src/compiler.rs)
delegates transitions and rescoring to `cascadia-game`. Its dense semantic
compiler is a correctness oracle, not a second rules engine and not an
optimized implementation. Legal root construction in
[`menu.rs`](../../crates/cascadia-rival/src/menu.rs) also validates composed
actions with canonical transitions.

The Rust-authored [`ResearchRulesetIdentity`](../../crates/cascadia-rival/src/ruleset_identity.rs)
binds all three rules layers:

- the legacy research label;
- `cascadia_game::RULES_SEMANTICS_ID`; and
- SHA-256 of serialized `GameConfig::research_aaaaa(4)`.

A familiar text label cannot mask a changed configuration or rules semantics.
Rival currently accepts only corrected four-player AAAAA, card A, no habitat
bonus.

### 2.2 Policies receive public capabilities, not private state

[`observation.rs`](../../crates/cascadia-rival/src/observation.rs) keeps
`PrivateSimState` on the trusted simulator side. A `FrozenPolicy` in
[`policy.rs`](../../crates/cascadia-rival/src/policy.rs) receives only:

- `PublicPolicyObs`, containing the public game state, public supply, acting
  seat, and that seat's local memory;
- one exact `RulesLegalMenu`;
- an opaque `HonestWorldSampler`; and
- a domain-specific `PolicyRng`.

There are four independent `SeatLocalMemory` values. A policy call returns the
next memory for the acting seat only. No shared cross-seat memory, private bag
inventory, hidden order, physical chance key, or raw `GameState` crosses the
policy boundary. `FrozenPolicy::fresh_instance` additionally requires a clean
policy instance for each seat and branch; action-affecting recurrent state
must live in explicit seat memory rather than hidden mutable object state.

### 2.3 Root chronology is explicit

The Rust contract distinguishes two public root kinds:

- `PreludePolicyRoot` chooses decline or a currently legal free
  three-of-a-kind replacement.
- `DraftPolicyRoot` chooses one complete draft action or one currently legal
  paid wipe.

Paid wipes are sequential reveal-and-recompose decisions. Future wipes are
never pre-enumerated before their replacement markets are public. A singleton
prelude decline is deterministic orchestration: it consumes no policy call,
memory transition, RNG ordinal, or decision record. A real accept/decline
choice is recorded normally.

`RulesLegalMenu` and `IncumbentCandidateMenu` are incompatible types and have
separate hashes. Candidate menus contain complete post-prelude drafts only;
prelude actions and paid-wipe indices cannot be frozen as candidates.

### 2.4 Identity hierarchy

[`action_id.rs`](../../crates/cascadia-rival/src/action_id.rs) separates
content from occurrence:

- `LegacyActionIdV0` preserves the historical
  `sha256(serde_json::to_vec(TurnAction))` algorithm exactly.
- `ActionContentId` binds canonical action content and the research rules.
- `PublicRootId` binds public observation, acting seat, root kind, chronology,
  local memory, and rules.
- `RootActionOccurrenceId` binds root, ordered legal menu, action content, and
  action index.
- `CandidateActionOccurrenceId` separately binds the frozen incumbent
  candidate menu.

The root manifest then binds the exact ordered candidate occurrence/content
pairs and every per-challenger S allocation. `candidate_set_identity` excludes
the allocation intentionally, while `deployment_design_sha256` includes every
pre-coefficient deployment choice. This split makes both menu identity and
experimental allocation auditable.

`B_k`, `pi_L`, `W_k`, and `M_(k+1)` use distinct Rust types and a shared exact
policy schema with a required `policy_kind`. Their identities include source,
executable, model, checkpoint, weights, bridge, tensor, numerical, precision,
search, Gumbel, refresh, endgame, RNG, memory, compiler, simulator, sampler,
candidate-generator, and failure contracts. Table-total utility,
table-native Q, true-hidden peeking, and model fallback are required fields
whose only admitted value is `false`.

## 3. Artifact and schema map

### 3.1 Additive schemas in the repository registry

[`rival/schema.py`](../../cascadiav3/src/cascadiav3/rival/schema.py) defines the
following exact, versioned records. Validators reject missing fields, unknown
fields, non-finite numbers, ambiguous digest wires, duplicate JSON keys, and
silent type coercions.

| Schema ID | Purpose | Current use |
| --- | --- | --- |
| `cascadiav3.rival_policy_identity.v1` | Non-substitutable identity for `B_k`, `pi_L`, `W_k`, or `M_(k+1)` | Implemented in Rust and Python |
| `cascadiav3.rival_root_manifest.v1` | Frozen root, candidates, roles, panels, inference mode, coefficient, bound, and error identities | Implemented |
| `cascadiav3.rival_terminal_pair_ledger.v1` | Replay-complete incumbent/challenger terminal pair | Implemented for CPU proxy evidence |
| `cascadiav3.rival_bound_certificate.v1` | Rust-authored certified terminal score-difference ranges | Implemented with a locked global certificate |
| `cascadiav3.rival_power_envelope.v1` | Symbolic pre-measurement work and memory grid | Implemented; structurally non-funding |
| `cascadiav3.rival_gpu_permit.v1` | Future accelerator capability | Wire validator implemented; source gate locked off |
| `cascadiav3.rival_preference_shard.v1` | Categorical preference sidecar with source provenance | Implemented for CPU contract tests |
| `cascadiav3.rival_training_view.v1` | Checked join of unchanged expert data and preference sidecar | Implemented; trainer use held |
| `cascadiav3.rival_terminal_panel_plan.v1` | Exact pre-outcome S, H, or L unit schedule | Implemented |
| `cascadiav3.rival_coefficient_calibration.v1` | Frozen control-variate coefficient and calibration provenance | Implemented |
| `cascadiav3.rival_potential_root_census.v1` | Complete eligible-root census for one error family | Implemented |
| `cascadiav3.rival_error_family_ledger.v1` | Census-complete family-wise error allocation | Implemented |
| `cascadiav3.rival_allocation_registry.v1` | Exact root/source/cohort assignments plus committed complete-game seed roles | Implemented with byte pins and seed-opening checks |

### 3.2 Rust-owned and journal-local wires

The CPU reference also owns exact wires that are validated locally rather than
being treated as expert-tensor schemas:

| Schema or contract ID | Authority |
| --- | --- |
| `cascadiav3.research_ruleset_identity.v1` | Canonical Rust rules identity |
| `cascadiav3.rival_public_policy_observation.v1` | Rust public observation |
| `cascadiav3.rival_seat_local_memory.v1` | Rust seat-local memory |
| `cascadiav3.rival_trajectory_ledger.v1` | Rust complete/partial canonical trajectory ledger |
| `cascadiav3.rival_proxy_terminal_trajectory.v1` | Rust forced-action proxy trajectory |
| `cascadiav3.rival_verified_terminal_pair_receipt.v1` | Rust verifier receipt |
| `cascadia-rival.verify-terminal-pair.v1` | Mandatory pinned verifier protocol |
| `cascadiav3.rival_tomography_result.v1` | Typed T0-T4 evidence grade |
| `cascadiav3.rival_tomography_summary.v1` | Rust WI-2 tomography harness summary (lower-bound witnesses only) |
| `cascadiav3.rival_dynamic_urn_proof_contract.v1` | Future coupling obligations; not an admission |
| `cascadiav3.rival_appeal_event.v1` | Python immutable journal event |
| `cascadiav3.rival_appeal_final.v1` | Python immutable one-look final receipt |
| `cascadiav3.rival_expert_root_identity_index.v1` | Python expert-record identity index |
| `cascadiav3.rival_preflight_expectation.v1` | Python phase and future-permit expectation |
| `cascadiav3.rival_candidate_set.v1` | Python candidate-set content identity payload |
| `cascadiav3.rival_deployment_design.v1` | Python deployment-design identity payload |
| `cascadiav3.rival_seed_commitment_payload.v1` | Python typed seed-opening commitment payload |
| `cascadiav3.rival_root_source_set.v1` | Python root/source-game universe identity payload |

The test fixtures under
[`cascadiav3/tests/fixtures/rival`](../../cascadiav3/tests/fixtures/rival)
lock representative rules, policy, manifest, bound, coverage, and preflight
wires. Fixtures are contract evidence only; their proxy scores are not game-AI
evidence.

### 3.3 Validation capabilities

Several validated Python dataclasses carry a process-local, content-bound
capability issued by their strict validator. The authenticator is keyed with a
process secret and covers the runtime fingerprint. This is practical defense
against accidental direct construction, `dataclasses.replace`, and stale
in-memory objects; it is not a sandbox against a malicious caller already
running arbitrary Python in the process.

Semantic manifest validation and artifact admission are distinct. Fixture
code may validate an in-memory mapping, but terminal evidence requires a
manifest loaded from canonical bytes under independent file and content pins.
Allocation registries use the same distinction. A self-carried content hash
alone is never the experiment's external root of trust.

This is defense in depth. Durable authority still comes from exact canonical
bytes and caller-supplied SHA-256 pins; an in-memory capability is never a
replacement for an artifact.

All generic Rival JSON readers impose an inclusive 64 MiB ceiling, reject an
oversized regular file from metadata before allocation, read through one
stable no-follow descriptor, and detect growth or mutation during the read.
Pinned scientific artifacts additionally require exact canonical bytes,
single-link identity, and a caller-supplied file hash. This bound protects
trusted runners from an honest wrong-path or stale-artifact mistake consuming
unbounded memory.

## 4. Panel plans freeze the experiment before outcomes

[`panel_plan.py`](../../cascadiav3/src/cascadiav3/rival/panel_plan.py) closes the
gap between a manifest that names a panel digest and an executable list of
units.

Each `cascadiav3.rival_terminal_panel_plan.v1` record binds:

- manifest, public root, structured rules, and source-game identity;
- candidate-set and incumbent policy identity;
- incumbent candidate occurrence and action content;
- sampler and policy RNG factory;
- exactly one panel kind;
- canonical contiguous unit indices; and
- for every unit, fidelity, target seat, challenger occurrence/action, and
  incumbent/challenger post-forced-action memory SHA-256.

The validator joins every field to a validated `RootManifest`, rejects a plan
whose content digest differs from `manifest.panel_identities[panel]`, and
requires the exact conditional allocation:

- S contains each challenger's individually registered `expected_s` count.
- H contains `expected_h` units per eligible challenger.
- L contains `expected_l` units per eligible challenger.

Required panels must have distinct, non-null identities; forbidden panels are
null. V1 also disables A and quantitative targets. High-fidelity-only designs
require S and H, forbid L and A, use `beta_cv = 0`, and make no multifidelity
claim. Multifidelity plans use low S, paired-high-low H, and independent low L
units, but those wires do not authorize production multifidelity evidence.

A validated plan can issue a content-bound `TerminalUnitExpectation` only for the currently
supported high-fidelity S/H proxy path. The expectation is a plan-issued
capability whose identity includes panel, unit, seat, candidate, action, and
both memory commitments. A caller cannot choose those values after seeing a
terminal result.

## 5. World, memory, and RNG commitments

### 5.1 Domain-separated RNG

[`rng.rs`](../../crates/cascadia-rival/src/rng.rs) derives independent domains
for source roots, outer physical randomness, branch keys, redetermination,
search, policy sampling, and tie breaking. Inner APIs accept one indivisible
`InnerRngCoordinate` containing public root, panel, branch, fidelity, acting
seat, replicate, and sample index. A caller cannot accidentally omit one of
those dimensions while constructing a key.

Policies never receive the outer coupling key. Changing an outer physical
coupling choice cannot alter the inner policy stream. The CPU stream and domain
derivations have golden tests.

### 5.2 Honest-world sampler

An `HonestWorldSampler` owns the root-local redetermination capability. The
policy requests worlds by ordinal and receives an opaque `PolicyWorld`, not a
seed or raw state. Ordinal zero is the initial redetermination; every other
ordinal is separately derived and deterministic.

The current canonical `GameState::redeterminize_hidden` retains the original
game seed, which canonical wildlife-return insertion also uses. Therefore the
implemented [`IndependentScenarioSampler`](../../crates/cascadia-rival/src/scenario.rs)
is conservatively named an **independent hidden-order reference**. It is valid
for replay and high-fidelity-only, beta-zero plumbing. It is not a proven
complete physical-world resampler, a common-random-numbers construction, or a
source of production covariance evidence.

### 5.3 Terminal commitments

A proxy terminal trajectory embeds the source state, source world, public
root, rules and candidate menus, forced action, initial redetermined world,
source and continuation memories, every subsequent compound action and public
decision, canonical state hashes, and final scores. Deserialization reruns
semantic validation and deterministic replay.

The verified receipt binds both branch-local post-action memory digests and
both exact 32-byte world-redetermination-seed commitments. The terminology is
important: these receipt fields commit to redetermination seeds; they are not
being presented as a universal physical-world coupling proof.

Every admitted S/H/L row also carries those exact seed commitments. The appeal
state machine and immutable journal reject reuse within a row and across every
panel of the same root appeal. The separately pinned allocation registry checks
global commitment uniqueness and realized-seed disjointness across registered
root and complete-game axes. Together these checks prevent accidental reuse
without claiming more physical-world semantics than the current sampler
supplies.

## 6. Strict Rust-to-Python terminal evidence

The cross-language join is implemented in
[`terminal_evidence.py`](../../cascadiav3/src/cascadiav3/rival/terminal_evidence.py)
and the Rust contract CLI in
[`rival-contract.rs`](../../crates/cascadia-rival/src/bin/rival-contract.rs).

### 6.1 Mandatory pins

The only receipt-emitting verifier form is:

```bash
rival-contract verify-terminal-pair \
  <pair-ledger.json> \
  'sha256:<expected-pair>' \
  'sha256:<expected-parent-manifest>'
```

Path-only verification is not an evidence form. Both qualified SHA-256 pins
are mandatory. Rust reads the exact ledger bytes, strictly deserializes them,
replays both trajectories, checks cross-branch equality and difference
contracts, recomputes the pair identity, requires the expected pair, requires
the externally frozen parent manifest, and only then seals a receipt.

The receipt is serialize-only in Rust and can be created only by the pinned
pair verifier. It binds:

- verifier contract and running executable SHA-256;
- exact ledger file SHA-256;
- pair and parent-manifest SHA-256;
- structured ruleset identity;
- source game, public root, legal menu, and candidate menu;
- scenario sampler, continuation policy, and policy RNG factory;
- panel ID, unit index, fidelity, and target seat;
- incumbent/challenger candidate occurrences and action content;
- incumbent/challenger post-action memories;
- incumbent/challenger redetermination-seed commitments;
- the challenger branch ordinal, fixed to its candidate-menu index;
- target-seat terminal score difference;
- `proxy_policy = true` and `beta_cv_required = 0`; and
- a canonical receipt content hash.

### 6.2 Process and file hardening

`RustTerminalVerifier` refuses to execute a path merely because it once had
the correct hash. It:

1. opens the source executable without following symlinks;
2. copies and hashes it into a private mode-0700 directory;
3. executes the immutable snapshot with no shell;
4. uses a small sanitized environment with the explicit CPU-only contract;
5. drains stdout and stderr concurrently with hard byte caps;
6. rejects terminal-pair ledgers larger than the inclusive 64 MiB Rust/Python
   contract before hashing or spawning;
7. enforces a bounded timeout, kills the verifier process group, and reaps the
   child so descendants cannot outlive a failed unit;
8. rejects successful commands that write to stderr;
9. rejects non-regular, symlinked, replaced, or changing ledger files; and
10. checks executable and ledger identity again after execution.

Stdout must contain one UTF-8 JSON object with the exact receipt shape;
duplicate keys are rejected and the decoded canonical content hash is checked.
Every preregistered identity field is then rejoined to the externally pinned
root manifest and plan-issued unit expectation. Rust-authenticated outcome
fields and independently pinned ledger/executable fields are checked against
their own authorities. A valid receipt for a different root, seat, menu,
candidate, panel, unit, memory, policy, sampler, or ruleset is rejected.

### 6.3 Evidence domains

Terminal rows carry a typed `EvidenceDomain`:

- `synthetic_contract_test` is available only through explicit fixture
  constructors.
- `cpu_proxy_reference` is available only through the Rust receipt adapter.
- `production_terminal` names the future domain but is structurally rejected
  by current row constructors, journals, and appeal machines.

Direct dataclass construction cannot manufacture a verified numeric row. The
current Rust adapter always returns `cpu_proxy_reference`; it has no code path
that emits `production_terminal`. Even if a proxy fixed bound clears the
margin, the appeal result is `no_label`, with `scientific_evidence = false`.
Enabling production evidence requires a future admitted Rust policy/RNG
adapter and an explicit code change, not merely a different enum value.

## 7. Immutable appeal journals and one-look inference

[`appeal_journal.py`](../../cascadiav3/src/cascadiav3/rival/appeal_journal.py)
makes the statistical state machine crash-reconstructible.

One journal directory contains:

```text
.journal.lock
events/
    0000000000000000.json
    0000000000000001.json
    ...
FINAL.json
```

Each event is canonical JSON, create-new, content hashed, sequence numbered,
and chained to the previous event. Every event repeats the appeal mode, root,
deployment design, and manifest identity. Event names must be contiguous from
zero. Unexpected files, gaps, aliases, symlinks, multiple hard links,
duplicate keys, non-canonical bytes, unknown event types, or broken chain
links fail closed.

An interprocess `flock` serializes append and finalization. Batch additions are
validated against a replayed machine before any event is published. `FINAL.json`
is also create-new and binds the event tip to the deterministic replayed
decision. Concurrent creators or finalizers have exactly one winner; a broken
`FINAL.json` symlink is still a finality fence rather than an invitation to
overwrite it.

The journal never trusts persisted terminal floats. A non-fixture row requires
a resolver bound to the same root, deployment design, and manifest. The
concrete `RustTerminalRowResolver` maps a receipt identity to an immutable
ledger and plan expectation, reruns the Rust verifier on every journal replay,
and compares the reconstructed row to every serialized field. Missing,
moved, changed, or substituted evidence invalidates the journal.

Callers receive a read-only `JournalSnapshot`, not a mutable state machine. A
single `finalize()` consumes the planned look; a second look is rejected.

## 8. Fixed panels, estimator, and categorical decisions

[`appeals.py`](../../cascadiav3/src/cascadiav3/rival/appeals.py) implements two
separate designs.

### 8.1 Multifidelity design

S evaluates every eligible challenger with its frozen allocation and selects
exactly one using `highest_mean_then_lexicographic_action_id`. S is selection
data only and never enters the confirmation estimate. H then records paired
high/low differences for the frozen challenger. L records an independent
low-fidelity panel.

For complete fixed panels, the estimator in
[`multifidelity.py`](../../cascadiav3/src/cascadiav3/rival/multifidelity.py) is

```text
mean(D_H) - beta_cv * mean(D_L on H) + beta_cv * mean(D_L on L).
```

The H-low and L-low expectations must be identical. Equal-law optimization is
used only with explicit matching law identities and a certified variance
match; otherwise the general independent-panel formula is used. Covariance is
checked against Cauchy-Schwarz, and a coefficient is bound to a disjoint
calibration corpus, root index, deployment design, policies, sampler,
allocation, low-law identities, and maximum absolute beta.

### 8.2 High-fidelity-only control

The beta-zero control has a distinct manifest, design type, state machine,
bound type, and result type. It uses S followed by H, structurally forbids L,
and makes no multifidelity claim. This separation prevents a high-only result
from being described as failed or degenerate Rival-MF.

### 8.3 Failures and label semantics

Every planned unit is either `complete`, `timeout`, or `invalid`. A timeout or
invalid unit remains in attempted-unit accounting and makes that fixed panel
produce `no_label`; it is never dropped to shorten the denominator. H cannot
open before S deterministically freezes one challenger, and L cannot open
before every H attempt exists. Unit IDs, receipt IDs, redetermination-seed
commitments, and inner RNG keys cannot be reused.

The future admitted production path may emit only this categorical preference:

```text
challenger_over_incumbent
```

It identifies the tested challenger action content and incumbent action
content. Untested menu actions are not marked as losers. No table utility,
numeric score target, or quantitative advantage is written to a training
label. V1 A panels and quantitative targets remain disabled.

The present CPU proxy path emits no preference at all.

## 9. Bounds, multiplicity, coverage, and analysis

### 9.1 Rust-authored score ranges

[`bounds.rs`](../../crates/cascadia-rival/src/bounds.rs) is the only
rules-aware range authority. Its current global AAAAA certificate is an
analytic relaxation, not an observed sample range. Python's
[`bounds.py`](../../cascadiav3/src/cascadiav3/rival/bounds.py) accepts only the
locked Rust authority, structured rules identity, expected scope, and pinned
certificate content. It never derives a Cascadia range from samples.

For multifidelity inference, certified widths transform to

```text
H corrected width = width_H + abs(beta_cv) * width_L
L correction width = abs(beta_cv) * width_L.
```

The implementation uses a preregistered, one-sided, fixed-sample Hoeffding
lower bound with separately allocated H and L error. The high-only control has
one H error term and no synthetic L expenditure.

### 9.2 Complete error families

[`cohorts.py`](../../cascadiav3/src/cascadiav3/rival/cohorts.py) loads one
canonical externally pinned allocation registry. Its root-source set is the
exact sorted projection of root ID, source game, and cohort role; it also binds
globally unique seed commitments and the separately registered promotion and
target roles. Commitment uniqueness is not called seed disjointness. Immediately
before a future run, typed u64 openings must reproduce every commitment and
prove realized cross-axis seed uniqueness.

[`coverage.py`](../../cascadiav3/src/cascadiav3/rival/coverage.py) derives the
eligible root universe from that registry: `relabel_selection` for the finite
training family and `shadow_one_seat` for the one-seat instrument. A census
must equal the derived set exactly before it can receive a validation
capability. Root IDs are sorted and unique. The error ledger must then account
for the census exactly, and the sum of per-root budgets cannot exceed the
family budget.

Finite-training-corpus and one-seat-instrument families are separate, and
their roots cannot overlap. A root that never activates still consumes its
preregistered potential-root allocation; observed activation cannot be used
to shrink the multiplicity family after the fact.

### 9.3 Coverage checks

The compact exact-coverage runner enumerates finite H and L distributions,
checks that declared widths cover support, verifies that H-low and L-low
marginals match, reproduces estimator unbiasedness, and computes exact
undercoverage. It also supplies dependency-free one-sided Clopper-Pearson
bounds and the replication count required for a zero-failure tolerance.

These are synthetic or exhaustively enumerable CPU checks. They prove the
implementation against their registered finite designs; they do not validate
the unknown production distribution.

### 9.4 Correct analysis unit

[`analysis.py`](../../cascadiav3/src/cascadiav3/rival/analysis.py) averages
roots within each complete source game and treats source games as the
independent top-level units. A decision-grade summary requires the externally
pinned allocation registry, its explicit identity, and one family kind; the
measurement rows must exactly cover that family and preserve every registered
root-to-source-game mapping. Missing, extra, or reassigned roots reject instead
of silently disappearing. Duplicating several roots from one game cannot
artificially shrink uncertainty. An iid-root standard error exists only as an
explicit diagnostic and is not the reported inferential unit.

## 10. Symbolic power and capacity machinery

[`power.py`](../../cascadiav3/src/cascadiav3/rival/power.py) evaluates a full
Cartesian design grid over:

- certified high/low widths by stratum;
- `n_H` and `n_L`;
- covariance and population-variance assumptions;
- target gaps, activation frequencies, and timeout rates;
- optimistic, central, and pessimistic per-unit throughput assumptions; and
- explicit available memory, fixed workspace, and GiB per active root.

The work model counts S units, both high and low work for paired H units, L
units, fixed per-root work, timeout completion probability, and the smaller of
worker capacity and memory capacity. An allocation that cannot clear the
fixed Hoeffding half-width reports
`NO_FINITE_HOURS_AT_FIXED_ROOT_ALLOCATION`; it is not assigned a misleading
large finite number.

The artifact is structurally marked:

```text
status = NON_FUNDING_SYMBOLIC_ONLY
can_fund_program = false
can_close_program = false
```

Every measured-cost field remains `UNRESOLVED`, including real terminal-pair
cost, coupled covariance, activation, timeout, resolved roots per hour, and
post-D1 GPU hours. Replacing any of those values before the P2 measurements
makes validation fail.

The symbolic grid can expose mathematical no-go regions and implementation
mistakes. It cannot fund, close, or rank the real program. Real measured power
has not been established.

## 11. CPU-only boundary and future permit

[`cpu_test_guard.py`](../../cascadiav3/src/cascadiav3/cpu_test_guard.py) exists
because hiding CUDA alone does not disable Apple MPS and does not prevent
availability queries. A strict CPU test session requires all three settings
before a device library is imported:

```bash
export CASCADIA_CPU_ONLY_TESTS=1
export CASCADIA_DEVICE=cpu
export CUDA_VISIBLE_DEVICES=''
```

Under that contract, `auto`, `cuda`, and `mps` requests fail before Torch
import or device discovery. Intentional accelerator tests are skipped before
they create a tensor. The guard is inert outside an explicitly enabled CPU
test session, so it does not silently rewrite ordinary production behavior.

The guard is applied at the entry points of the trainer, inference bridge,
bridge throughput probe, and model throughput benchmark. The complete CPU gate
in [`validate_rival_pre_gpu.sh`](../../cascadiav3/scripts/validate_rival_pre_gpu.sh)
sets the environment rather than inspecting available hardware.

[`preflight.py`](../../cascadiav3/src/cascadiav3/rival/preflight.py) separately
validates the shape of a future permit against an exact caller expectation:
phase, device, source revision and digest, command digest, preregistration
digest, budget, authority, and validity interval. The source constant
`ACCELERATOR_PHASE_ENABLED` is hard-coded `False`. A missing permit returns
`PERMIT_MISSING`; even a valid future permit returns
`PRE_GPU_PHASE_LOCKED`. Invalid contracts exit nonzero instead of masquerading
as the expected lock.

This is a reviewed future capability wire, not current permission to run an
accelerator.

## 12. Training-view machinery and the P8 hold

[`training_view.py`](../../cascadiav3/src/cascadiav3/rival/training_view.py)
implements a two-stage immutable join without changing ExpertTensorShard v1-v4:

1. A root-identity index binds every expert record to the expert shard, raw
   root ledger, source revision, root/menu/action occurrence identities,
   selected action, and exact action-tensor row digests.
2. A preference sidecar binds that index plus the expert shard, raw root and
   world ledgers, policies, manifest, allocation, bound, error family,
   panels, inference mode, and optional coefficient.

Both must be loaded from canonical, caller-pinned, stable-descriptor,
single-link files. The join rechecks source digests, exact tensor rows,
selected action, menu membership, panel identities, and high-only versus
multifidelity structure. Panel identities cannot be reused across records in
one sidecar; global cross-sidecar non-reuse remains a required P8 admission
check. Unlabeled records carry zero preference loss weight, and policy weight
is not treated as a sampling or frequency weight. The sole positive category
wire is `challenger_over_incumbent`, matching the appeal decision contract.

The default-off collator delegates to the existing v1-v4 collation path.
Requesting `--rival-preference-training` in
[`torch_train_cascadiaformer.py`](../../cascadiav3/src/cascadiav3/torch_train_cascadiaformer.py)
raises `RivalTrainerIntegrationHeld` before Torch import and before any input
or output path is touched. The current code does not add a preference loss,
sampler, optimizer state, checkpoint field, or resume behavior.

Removing this hold requires positive evidence through P7, John's explicit
TRAIN instruction, a phase-specific permit, a frozen preregistration, and
flag-off bit-identity tests. CPU validation of the sidecar format does not
satisfy those requirements.

## 13. Dynamic-urn status

[`dynamic_urn.rs`](../../crates/cascadia-rival/src/dynamic_urn.rs) contains the
obligation set and an exhaustive ideal priority-oracle checker for small urns
through eight items. It intentionally contains no `GameState` chance hook and
cannot construct an admitted status from JSON.

Production common-physical-randomness coupling remains denied because:

- canonical bags are private to `cascadia-game`;
- wildlife items do not yet have stable physical-token IDs;
- draw, wipe, refill, return, and replay semantics have no admitted canonical
  chance-injection hook; and
- the present independent hidden-order reference does not prove identical
  high/low and incumbent/challenger physical marginals.

No external shadow bag, `serde(skip)` state, or duplicate chance engine has
been added. If P2a and P2b later justify this line, the canonical hook and
physical identity work must occur under the plan's explicit post-D1 proof
gate. Failure of that proof closes multifidelity; it does not justify weakening
the contract.

## 14. Performance posture

The current code is optimized for trustworthy pre-GPU engineering where
performance does not compromise identity or replay:

- Rust handles rules-heavy state, action, replay, and hashing without an
  accelerator dependency.
- The verifier executes one pinned semantic replay and emits one compact
  receipt; Python performs only the cross-artifact join and statistics.
- Verifier stdout and stderr are streamed concurrently and bounded, preventing
  a pipe deadlock or unbounded memory growth.
- Immutable writers publish atomically without replacing an existing artifact.
- Symbolic power explicitly caps parallel roots by memory, rather than assuming
  worker count equals realizable concurrency.
- The dense semantic compiler recomputes canonical truth and is intentionally
  unoptimized; component-local and incremental compilers are P3 work.

No terminal-pair throughput number in a CPU fixture is decision-grade. The
production `B_k` cost, memory footprint, bridge economics, coupled low-fidelity
cost, and realized parallel-root capacity are all unknown until their gated
measurements. The 10,000-game and 1,000,000-transition release suites have not
been run. Real measured power has not been calculated.

## 15. Threat model

The machinery assumes trusted operators, trusted local runners, and no direct
adversary trying to defeat the experiment. It is designed for the failures
that actually threaten this campaign: implementation bugs, stale or mismatched
artifacts, accidental reuse, races, crashes, partial writes, resource runaway,
and analysis choices that drift from the preregistration. It does not attempt
to sandbox a malicious caller already executing arbitrary code inside the
Python process, or an administrator deliberately rewriting source and every
pinned authority before review.

The defended threats are:

- substitution of rules, policies, menus, roots, actions, panels, allocations,
  coefficients, bounds, ledgers, executables, or source files;
- leakage of private state, hidden chance identities, or one seat's memory to
  another seat;
- adaptive choice of candidate, panel size, seat, memory, beta, subgroup, or
  inference look after outcomes exist;
- fabricated numeric rows, proxy-to-production relabeling, or Python-authored
  Cascadia range claims;
- truncating failed panels or dropping timeouts to improve a result;
- duplicate/reordered JSON keys, non-finite values, coercive booleans, unknown
  fields, stale canonical bytes, or recomputed hashes over semantically invalid
  records;
- symlink, alias, replace, mutation, and executable time-of-check/time-of-use
  mistakes, with single-link requirements on artifact families that require
  unique ownership;
- concurrent append/finalize races and repeated inferential looks; and
- accidental CUDA/MPS initialization during the authorized CPU phase.

The principal residual risks are scientific rather than mechanical: global
bounds may be too loose, full-policy terminal pairs may be too expensive,
low/high covariance may be weak, one-deviation gains may not compose, and
relabeling may not improve gameplay. The current implementation intentionally
does not hide those unknowns behind CPU proxy success.

## 16. Failure matrix

| Failure | Detection point | Disposition | Scientific consequence |
| --- | --- | --- | --- |
| Missing, stale, or wrong manifest/panel pin | Manifest and panel-plan validators | Reject before a unit is admitted | No row, no label |
| Wrong pair or parent-manifest pin | Rust verifier CLI | Nonzero exit, no receipt | No evidence |
| Semantically altered ledger with recomputed JSON hash | Rust deserialization and replay | Reject | No evidence |
| Receipt belongs to another root, menu, seat, policy, memory, or panel | Python exact semantic join | Reject | No row |
| Executable path is symlinked, changed, or replaced | Snapshot and before/after identity checks | Reject or terminate | No row |
| Verifier times out, exits nonzero, overproduces output, or warns on stderr | Bounded subprocess runner | Reject | Operational failure only |
| S unit times out or is invalid | Appeal state machine | Final `no_label`; H/L never open | Attempt stays in accounting |
| H or L unit times out or is invalid | Appeal state machine | Final `no_label`; no shortened panel | Attempt stays in accounting |
| Candidate, unit, redetermination seed, inner RNG key, or receipt reused | State machine and journal replay | Reject transition | No inference |
| H/L low laws or coefficient design differ | Coefficient and design binding | Reject | No MF estimate |
| Error ledger omits a potential root or overspends family delta | Census/ledger validator | Reject family | No bound claim |
| Proxy bound clears practical margin | Evidence-domain gate | Audit `no_label` | Explicitly not scientific evidence |
| Journal gap, mutation, alias, chain break, or forged final | Journal replay | Reject journal | No result |
| Concurrent finalization | File lock plus create-new `FINAL.json` | Exactly one winner | No second look |
| Device is `auto`, CUDA, or MPS under CPU guard | Pre-import device guard | Immediate failure/skip | No device access |
| Valid future permit supplied during pre-GPU source phase | Source-locked preflight | `PRE_GPU_PHASE_LOCKED` | No launch |
| Preference training flag requested | P8 trainer hold | Raise before Torch or filesystem access | No training |
| Dynamic-urn admission attempted without canonical hook | Rust admission type | Reject | No coupling or MF claim |
| Symbolic design cannot clear the bound | Power calculator | `NO_FINITE_HOURS_AT_FIXED_ROOT_ALLOCATION` | Mathematical no-go for that row only |
| Measured field inserted before P2 | Power artifact validator | Reject artifact | Real power remains unresolved |

## 17. Independent failure-mode findings incorporated

The implementation review found and closed several ways a superficially
well-hashed system could still be wrong:

1. **A rules label was not enough.** The fix is the structured Rust ruleset
   identity binding label, semantics, and exact configuration bytes.
2. **A pair path and self-reported digest were not evidence.** The verifier now
   requires external pair and parent-manifest pins on every receipt-emitting
   invocation.
3. **Hash-only verification did not prove semantics.** Rust now strictly
   deserializes and replays both trajectories before sealing the receipt;
   Python rejoins preregistered identities to the manifest and panel unit and
   checks outcome/file fields against their separate authorities.
4. **A panel digest could have been decorative.** Required panel identities now
   resolve to exact pre-outcome plans with contiguous unit schedules and exact
   per-candidate allocations.
5. **Post-action recurrent state was initially undercommitted.** Plans, pairs,
   and receipts now bind separate incumbent and challenger memory digests.
6. **An executable could change between hashing and execution.** The Python
   verifier executes a private immutable snapshot and checks it again after the
   child exits.
7. **Persisted floats could have bypassed the verifier.** Numeric terminal rows
   now require private proof-bearing constructors; journals re-resolve and
   reverify their ledgers on every replay.
8. **An in-memory one-look rule did not survive crashes.** Immutable chained
   events and a create-new final receipt make finality reconstructible.
9. **Equal-law beta formulas were too easy to assume.** The general variance
   formula is the default; the special formula requires exact expectation/law
   identities and verifies reduction to the general form.
10. **Observed appeals could have defined their own multiplicity family.** A
    complete preregistered root census now precedes the immutable error ledger.
11. **Root-level standard errors could overcount one source game.** Analysis now
    clusters at the complete source-game level.
12. **Timeouts could have improved a result by disappearing.** Fixed counts and
    operational accounting turn any incomplete confirmation panel into
    `no_label`.
13. **`CUDA_VISIBLE_DEVICES=''` did not address MPS.** The pre-import guard now
    requires an explicit CPU device and rejects auto-selection across relevant
    trainer and bridge entry points.
14. **A CPU proxy could accidentally look like strength evidence.** Evidence
    domains and label emission are structural: proxy rows cannot emit a
    scientific preference even when their audit bound clears.
15. **A tempting shadow-bag implementation would have created a second chance
    engine.** Dynamic-urn code is limited to proof obligations and a small ideal
    oracle; canonical integration remains held.
16. **A validated sidecar could have silently activated training.** The legacy
    trainer remains default-off and the opt-in flag deliberately fails before
    Torch import or filesystem mutation.
17. **A self-hashed manifest was not an external preregistration pin.** Fixture
    validation and externally byte/content-pinned loading are now distinct;
    terminal evidence requires the latter.
18. **A caller-supplied root list could understate multiplicity.** The potential
    root census is now derived from the externally pinned allocation registry
    and must equal its family-specific root universe exactly.
19. **Unique commitments were not realized seed disjointness.** The registry
    reports commitment uniqueness only; a separate pre-run opening check must
    cover every registered root/game seed and reject actual reuse.
20. **An unconstrained challenger branch permitted accidental seed shopping.**
    Rust now derives the branch ordinal from the frozen candidate-menu index,
    includes it in the receipt, and Python rejoins it to the manifest.
21. **An oversized ledger or verifier descendant could outlive a failed unit.**
    Rust and Python share an inclusive 64 MiB ledger ceiling, Rust caps nested
    trajectory structures, and timeout/output failure kills and reaps the
    verifier process group.
22. **The documented S tie-break used the wrong identity.** Equal selection
    means now break on `action_content_id`, never candidate-occurrence ID, and
    a reversed-order regression prevents the two namespaces from aligning by
    accident.
23. **A clustered summary could omit roots or entire source games.** It now
    requires an externally pinned allocation registry and exact family root-to-
    source-game coverage before reporting a mean or standard error.
24. **A journal helper rehashed a receipt including its own hash.** It now
    validates and returns the Rust-compatible self-hash over all receipt fields
    except `receipt_sha256`.
25. **A wrong JSON path could allocate without limit.** Every generic Rival
    JSON reader now has the stable-descriptor inclusive 64 MiB ceiling described
    above.
26. **Training sidecars had weaker file admission and an error-path descriptor
    leak.** Index and sidecar loads now use the canonical single-link reader,
    and every post-open failure closes the expert shard.

## 18. CPU verification runbook

Run commands from the repository root. These commands establish the CPU-only
contract explicitly; none should probe or initialize an accelerator.

### 18.1 Complete pre-GPU gate

```bash
bash -n cascadiav3/scripts/validate_rival_pre_gpu.sh
bash cascadiav3/scripts/validate_rival_pre_gpu.sh
```

The script checks the guard and expected preflight denial; Rust formatting,
locked build/check/test/clippy, provenance coverage, and the exact contract
binary; the CPU guard plus the existing bridge/trainer suites affected by that
guard; every `test_rival_*.py` test; and Ruff on the new CPU/Rival surface.

### 18.2 Explicit Rust/Python receipt integration

Build the CPU contract binary and force the integration test to use that exact
path so the test cannot skip a missing or stale binary:

```bash
env CUDA_VISIBLE_DEVICES='' \
  cargo build --locked -p cascadia-rival \
  --no-default-features --features cpu-reference \
  --bin rival-contract

env CUDA_VISIBLE_DEVICES='' \
  CASCADIA_DEVICE=cpu \
  CASCADIA_CPU_ONLY_TESTS=1 \
  CASCADIA_RIVAL_CONTRACT_BIN="$PWD/target/debug/rival-contract" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=cascadiav3/src \
  .venv/bin/python -m unittest discover \
  -s cascadiav3/tests -p 'test_rival_terminal_evidence.py' -v
```

### 18.3 Rust workspace checks

```bash
env CUDA_VISIBLE_DEVICES='' \
  cargo fmt --package cascadia-rival -- --check

env CUDA_VISIBLE_DEVICES='' \
  cargo fmt --package cascadia-provenance -- --check

env CUDA_VISIBLE_DEVICES='' \
  cargo check --workspace

env CUDA_VISIBLE_DEVICES='' \
  cargo test --workspace

env CUDA_VISIBLE_DEVICES='' \
  cargo clippy -p cascadia-rival --all-targets \
  --no-default-features --features cpu-reference -- -D warnings

env CUDA_VISIBLE_DEVICES='' \
  cargo test --manifest-path cascadiav3/real-root-exporter/Cargo.toml
```

The exporter is a separate workspace and must be tested explicitly even though
exporter extraction for Rival remains held.

### 18.4 Python checks

```bash
env CUDA_VISIBLE_DEVICES='' \
  CASCADIA_DEVICE=cpu \
  CASCADIA_CPU_ONLY_TESTS=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=cascadiav3/src \
  .venv/bin/python -m unittest discover \
  -s cascadiav3/tests -p 'test_rival_*.py' -v

env CUDA_VISIBLE_DEVICES='' \
  CASCADIA_DEVICE=cpu \
  CASCADIA_CPU_ONLY_TESTS=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=cascadiav3/src \
  .venv/bin/python -m unittest discover \
  -s cascadiav3/tests -p 'test_cpu_test_guard.py' -v

.venv/bin/ruff check \
  cascadiav3/src/cascadiav3/cpu_test_guard.py \
  cascadiav3/src/cascadiav3/rival \
  cascadiav3/tests/test_cpu_test_guard.py \
  cascadiav3/tests/test_rival_*.py
```

The full repository suite additionally needs a Python 3.12 environment with
Torch installed. The environment may contain accelerator-capable Torch, but
the invocation remains CPU-only: the checked guard rejects `cuda`, `mps`, and
`auto` before an accelerator-specific entry point can import Torch, and the
intentional CUDA/MPS tests skip before device discovery.

```bash
test -x "${CPU_TORCH_PYTHON:?set CPU_TORCH_PYTHON to Python 3.12 with Torch}"
env -u PYTORCH_CUDA_ALLOC_CONF \
  CUDA_VISIBLE_DEVICES='' \
  CASCADIA_DEVICE=cpu \
  CASCADIA_CPU_ONLY_TESTS=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=cascadiav3/src \
  "$CPU_TORCH_PYTHON" -m unittest discover -s cascadiav3/tests
```

### 18.5 Executed verification receipt (2026-07-16)

All results below were obtained in the isolated
`feat/rival-cpu-machinery` worktree with `CUDA_VISIBLE_DEVICES=''` and the
CPU-only guard enabled wherever Python could reach Torch.

| Gate | Result |
| --- | --- |
| `validate_rival_pre_gpu.sh` | PASS: expected `PERMIT_MISSING` denial; 80 Rust library, 2 binary, 1 PR battery, 3 CLI, 2 doc, 4 provenance, 10 guard, 13 bridge, 15 trainer, and 153 Rival Python tests; expected skips were 6 bridge and 14 trainer tests; Clippy and Ruff passed |
| `cargo check --workspace` | PASS; only the pre-existing `cascadia-api` dead-code warning |
| `cargo test --workspace` | PASS: 310 active tests across the workspace, including the 125-game/exactly-10,000-transition reference battery; one pre-existing timing harness remained ignored |
| standalone `real-root-exporter` suite | PASS: 68 tests |
| full Python 3.12/Torch suite | PASS: 550 tests, 48 intentional skips, 116.175 seconds |
| formatting and static checks | PASS: scoped Rustfmt, Ruff check/format, shell syntax, `git diff --check`, and temporary-marker scan |
| isolated publication | PASS: `feat/rival-cpu-machinery` pushed to its same-named origin branch; not merged into `main` |

The full-suite run also exposed and closed two order/hermeticity defects: the
preflight CLI now distinguishes a Torch module that was already loaded by a
different test from a new import during its own invocation, and the experiment
queue parser test uses a tracked test fixture rather than an ignored
operational queue. Neither fix changes a live queue or runner.

No device discovery, accelerator initialization, remote command, campaign
status query, scientific seed allocation, experiment launch, training run, or
gameplay gate occurred. The release-scale CPU-1 battery remains deliberately
unrun.

### 18.6 Test tiers and honest interpretation

| Tier | What it covers | What passing means |
| --- | --- | --- |
| Contract unit tests | Exact schemas, identities, public/private boundary, RNG, algebra, ranges, state machines | The local contracts behave as specified |
| Fault-injection tests | Field perturbation, semantic resealing, duplicate JSON, links, races, mutation, timeout, output caps | Known substitution and durability failures fail closed |
| Cross-language receipt test | Real Rust fixture, mandatory pins, Python semantic join | The two implementations agree on one replayed proxy contract |
| Workspace regression | Existing Rust/Python/exporter behavior | The branch has not broken the tested surrounding system |
| PR-scale chronology/replay | Curated cases plus 125 complete games and exactly 10,000 deterministic randomized reachable transitions | Strong finite engineering evidence only |
| Release-scale CPU reference | 12,500 games and 1,000,000 transitions | **CLAIMED 2026-07-16: zero mismatches in 220.8s** ([receipt](../../cascadiav3/reports/rival_cpu1_battery_receipt_20260716.json)) |
| Scientific premise/power | Production `B_k`, measured costs, covariance, activation, complete panels | **Not run; requires D1 and P2 permits** |
| Gameplay strength | Paired current-rules games and the frozen 1,000-game target battery | **Not run and not authorized here** |

## 19. Explicit holds

The following are intentionally absent, not unfinished details to infer away:

- **D1:** no status query, partial read, process change, queue change, or claim
  about its outcome.
- **GPU/MPS:** no initialization, discovery, benchmark, kernel, dependency, or
  runner. The future permit remains denied in source.
- **Remote and scientific operations:** no john0/fleet access, seed allocation,
  preregistration launch, job, waiter, chain, training run, generation run, or
  gameplay gate.
- **Production `B_k`:** no exporter trace extraction and no claim that the
  first-legal CPU fixture reproduces the transformer-plus-Gumbel policy.
- **Exporter extraction:** held until the durable D1 boundary; the existing
  exporter is tested but not refactored for Rival here.
- **Dynamic urn:** no canonical chance hook, physical-token identity, admitted
  coupling, production covariance, or L-row terminal verifier.
- **P2/P3:** no production premise probe, component compiler, incremental
  compiler, 12-symmetry compiler proof, or accelerator economics.
- **Trainer integration:** held at P8; preference sidecars are validation
  artifacts only.
- **Scientific labels:** the current Rust terminal adapter is proxy-only and
  cannot emit one.
- **Release-scale claims:** the 10,000-game/1,000,000-transition suite and real
  measured power are not run.

## 20. Readiness statement

This branch supplies a serious CPU foundation for future experiments: exact
identities, canonical replay, preregistered unit plans, a strict Rust/Python
evidence boundary, immutable single-look journals, bounded fixed-panel
statistics, complete multiplicity accounting, symbolic work/memory analysis,
and a training-data join that cannot activate the trainer.

It does **not** establish that Rival will improve Cascadia play. The next
scientifically meaningful step remains the plan's durable D1 checkpoint,
followed by separately authorized and preregistered production-policy premise
measurements. Until those measurements exist, the only honest verdict is:

```text
CPU contract machinery: implemented for engineering validation
CPU proxy strength evidence: none
Production terminal evidence: none
Real measured power: UNRESOLVED
GPU authorization: DENIED
100-point claim: not evaluated
```

## 21. Golden-trace prep (WI-3, held at the D1 wall)

The preparation half of build-scope Work Item 3 (exporter extraction under
golden traces) is delivered in `crates/cascadia-rival/src/golden_trace.rs`,
sized to the 2026-07-16 budget ruling: only what Rival-Lite's late-game
terminal continuations will need to prove behavioral identity of an extracted
incumbent, and nothing speculative beyond it.

**What exists (CPU-only, synthetic fixtures only):**

- **`cascadiav3.rival_golden_decision_trace.v1`** — a fail-closed,
  content-hashed record of ONE serving decision: the identity triple
  (canonical ruleset, source revision, hash-pinned policy identity), public
  state hash + ply + seat + completed turns, the free three-of-a-kind prelude
  decision (with the revealed market's hash when accepted), a legal-menu
  digest (action count, cap, first/last action ids, SHA-256 of the full
  canonical ordered id list — the list itself is never stored), the complete
  gumbel search configuration (n, top_m, depth_rounds, determinizations,
  market samples, exact endgame turns, blend, menu cap, c_visit, c_scale,
  seed), every model-bridge interaction in order as request-row-count +
  request/response payload SHA-256 digests, the chosen action (id, menu
  index, completed Q, improved-policy mass), and a whole-trace content hash
  over the recursively key-sorted canonical JSON. Floats travel as canonical
  shortest round-trip decimal strings, so traces are byte-deterministic
  across platforms and float equality is bit-exact.
- **`compare_traces(reference, candidate)`** — returns the FIRST divergent
  field as a named `TraceDivergence` carrying both observed values (state
  hash, prelude, each menu and search-config field, the k-th bridge digest,
  chosen action, ...). Byte-identical traces compare equal; any single-field
  mutation is detected and named. This is the identity check the post-D1
  extraction must pass on every captured production trace.
- **`cascadiav3.rival_golden_trace_manifest.v1`** — a sealed trace-set
  container (identity triple, count, strictly seed-sorted per-trace SHA-256
  entries, manifest content hash) with the tomography harness's immutable
  publication pattern (`write_json_immutable` never overwrites). A trace
  declaring any other ruleset, source revision, or policy identity is
  refused, never mixed in.
- **Fifteen synthetic-fixture tests** in the crate's unit suite: wire-form
  lockdown, round-trips, every-leaf-required and every-leaf-perturbation
  fail-closed sweeps, unknown-field rejection per layer, a 30-case
  first-divergence matrix, manifest fault injection, immutable publication,
  cross-identity rejection, and locked golden hashes for both v1 schemas.

**What is deliberately absent (not unfinished — held):**

- No production trace capture: nothing here reads a checkpoint, weights
  manifest, model bridge, or accelerator, and no trace of the real serving
  incumbent exists on this branch.
- No `cascadia-v3-policy` crate: the extraction target crate is not stubbed
  or scaffolded; creating it touches the exporter's dependency closure.
- No change to `cascadiav3/real-root-exporter/**`, `crates/cascadia-game/**`,
  the trainer, the bridge, or anything else the live D1 chain reads.

**Unblock condition:** a durable D1 boundary plus an explicit instruction.
Only then: capture production golden traces on pinned seeds/config (CPU
bridge acceptable), extract the policy-critical exporter paths
behavior-identically into the library crate, and require every captured
trace to compare equal through the library path alongside the full
pre-existing exporter and workspace suites.
