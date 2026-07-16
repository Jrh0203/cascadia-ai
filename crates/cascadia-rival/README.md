# cascadia-rival

`cascadia-rival` is the CPU-reference protocol and scientific-identity layer
for the Rival research program. It is not a rules engine, a serving engine, or
a strength claim. All legal transitions and scoring remain owned by
`cascadia-game`.

## Hard boundaries

- The default and only current feature is `cpu-reference`. This crate has no
  accelerator feature or dependency.
- `PrivateSimState` is accepted only at the trusted simulator boundary. A
  `FrozenPolicy` receives `PublicPolicyObs`, an exact supplied menu, an opaque
  `HonestWorldSampler`, and a domain-specific `PolicyRng`; it never receives a
  `GameState`.
- `PublicPolicyObs` always includes both `PublicGameState` and `PublicSupply`,
  plus one validated seat index and that seat's versioned local memory.
- Rival observations and menus reject every game configuration except the
  corrected four-player AAAAA, no-habitat-bonus research configuration.
- `PolicyWorld` permits canonical branching but has no public raw-state,
  hidden-inventory, physical-key, or scenario-key accessor.

## Chronology

There are two root kinds:

1. `PreludePolicyRoot` chooses decline or the currently legal free
   three-of-a-kind replacement.
2. `DraftPolicyRoot` chooses either a complete draft action or one currently
   legal paid wipe.

Paid wipes are sequential public decisions. `MenuComposer::draft_root` never
pre-enumerates a vector of future wipes: after one wipe is selected, canonical
preview reveals its replacement market, the accumulated `MarketPrelude` is
extended, and a new public observation/root/menu is constructed. Complete
drafts retain the entire accumulated prelude and are revalidated with
`GameState::transition` from the source state.

`RulesLegalMenu` and `IncumbentCandidateMenu` are intentionally different
types. The former is canonical legality; the latter is an ordered frozen-policy
subset containing complete post-prelude drafts only. Prelude and paid-wipe
indices are rejected at candidate freeze time. Their hashes cannot be
substituted silently.

A singleton prelude menu is deterministic orchestration: it must be the
canonical decline, invokes no policy, changes no seat memory, consumes no RNG
ordinal, and produces no decision record. A genuine accept/decline choice is a
normal recorded policy root.

## Identity layers

- `LegacyActionIdV0` exactly preserves
  `sha256(serde_json::to_vec(TurnAction))` and its historical `sha256:` wire
  form.
- `ActionContentId` binds canonical ordered action fields, enum
  discriminants, content schema, and structured research rules.
- `PublicRootId` binds the full public observation (including supply and local
  memory), acting seat, root kind, chronology version, and rules.
- `RootActionOccurrenceId` binds root ID, ordered menu hash, action content,
  and fixed-width action index.
- `BkIdentity`, `PiLIdentity`, `WkIdentity`, and `MNextIdentity` are distinct
  Rust and wire types. All bind complete source/artifact/config/RNG/failure
  identities and explicitly reject table-total utility, table-native Q,
  true-hidden peeking, and model fallback.
- `ResearchRulesetIdentity` is the sole owner of the legacy research label,
  `cascadia_game::RULES_SEMANTICS_ID`, and the locked serialized
  `GameConfig::research_aaaaa(4)` digest.

## Randomness

`RngFactory` derives separate source-root, outer-physical, branch,
redetermination, search, policy-sampling, and tie-break domains. Inner APIs do
not accept an outer coupling key. They do require one indivisible
`InnerRngCoordinate` containing the public root, panel, branch, fidelity,
acting seat, replicate, and sample index. This prevents a call site from
silently omitting an experimental identity dimension while still keeping
outer physical coupling inaccessible. Golden tests lock the derivation and CPU
stream.

`HonestWorldSampler` owns the root-local redetermination capability. Policies
request `sample(0)`, `sample(1)`, and so on; they never receive or construct a
seed. Ordinal zero is the initial world, every other ordinal is deterministically
domain-separated, and repeated ordinals reproduce byte-identical worlds.

The current canonical `GameState::redeterminize_hidden` deliberately preserves
the state's original seed, which canonical wildlife-return helpers also use.
This crate does not reach through private serialization or create a second
chance implementation to rewrite it. Consequently, the P0 sampler proves the
public/hidden-order boundary and independent inner RNG contract, but a coupled
dynamic-urn production claim remains fail-closed until the separately gated
canonical chance-hook work is admitted. `SmallUrnOracleReport` is
recomputed during deserialization; an `exact: true` JSON record cannot be
forged by editing summary counts.

## Terminal-pair evidence

`IndependentScenarioSampler::new` freezes a canonical post-prelude draft root,
all four initial seat memories, the complete source game identity, the rules
menu, and the outer RNG identity. A `ProxyTerminalPairRequest` requires:

- one frozen draft-only incumbent candidate menu;
- distinct incumbent and challenger candidate occurrences;
- matched panel, unit, and fidelity coordinates, with incumbent branch fixed to
  `incumbent` and challenger branch ordinal fixed to the challenger's frozen
  candidate-menu index;
- one frozen continuation-policy prototype, from which the harness requests
  four fresh seat instances per branch; and
- one explicit branch-local post-forced-action memory for the acting seat.

Every trajectory embeds the source world, source public root, both menu hashes,
both root/candidate occurrence IDs, all source and continuation memories, the
independently redetermined initial world, every compound action and public-root
decision, every canonical state hash, and canonical final scores. Deserializing
a trajectory or pair replays all of it. Pair fields are private, and semantic
validation still runs if a caller recomputes JSON content hashes after a
mutation.

The challenger branch ordinal is not a tunable seed coordinate. Both live runs
and deserialized ledgers require
`EvaluationBranch::Challenger(challenger_candidate_index)` exactly. The
verified receipt publishes the resulting `challenger_branch_ordinal` and binds
it into `receipt_sha256`, so a candidate cannot be paired with a conveniently
substituted hidden-world stream.

The `FrozenPolicy::fresh_instance` contract requires behaviorally clean
instances; every action-affecting recurrent value belongs in explicit
`SeatLocalMemory`. Hidden mutable implementation state is permitted only for
semantics-neutral caches.

`TrajectoryLedger`, proxy trajectories, proxy pairs, and verified receipts use
same-directory, no-replace immutable publication. A hard-link publish is
atomic with respect to competing writers; an existing artifact is never
overwritten.

The contract CLI provides an end-to-end Rust authority boundary:

```bash
rival-contract proxy-terminal-pair-fixture > pair.json
rival-contract proxy-terminal-pair-fixture 'sha256:<parent-manifest-sha256>' > pinned-pair.json
rival-contract proxy-terminal-pair-fixture \
  'sha256:<parent-manifest-sha256>' 'sha256:<panel-id>' '<unit-index-u32>' > panel-pair.json
rival-contract verify-terminal-pair pair.json \
  'sha256:<expected-pair>' 'sha256:<expected-parent-manifest>' > receipt.json
```

The fixture command is deterministic and explicitly proxy-only. Its optional
strictly qualified SHA-256 argument replaces only the fixture parent-manifest
pin (and consequently the enclosing pair hash), allowing a cross-language
manifest join without changing either trajectory. Supplying the parent pin,
panel ID, and canonical unsigned-decimal unit index together also binds the
fixture to a preregistered panel unit; two-argument and four-or-more-argument
forms are rejected. The verifier
performs strict deserialization, replay, cross-branch validation, and content
hash checks before emitting anything. Input must be a non-symlink regular file
no larger than 64 MiB (67,108,864 bytes); metadata is checked before allocation
and the read itself is capped. Trajectory deserialization independently caps
candidate entries, turns, state hashes, score rows, and per-turn decision
records before replay. Any violation exits nonzero and emits no receipt. Its
receipt binds the exact ledger file
bytes, pair and parent-manifest identities, source/root/menu/action identities,
experimental coordinate including the candidate-derived challenger branch
ordinal, result, verifier contract, post-action memory
commitments, each branch's exact 32-byte world-redetermination-seed commitment,
and SHA-256 of the running verifier executable. The Rust receipt type is
serialize-only and can be
constructed only by the pinned pair verifier; arbitrary JSON cannot be
deserialized into an authenticated Rust receipt. Both the pair identity and the
externally frozen parent-manifest identity are mandatory; a path-only invocation
emits no receipt.

## CPU verification

```bash
cargo test -p cascadia-rival
cargo test -p cascadia-rival --no-default-features
cargo test -p cascadia-provenance
cargo check -p cascadia-rival --all-targets --all-features
cargo clippy -p cascadia-rival --all-targets --no-default-features --no-deps -- -D warnings
```

The tests include compile-fail private-state checks, hidden-order and hidden-
partition metamorphics, seat-memory isolation, strict schema perturbations,
legacy-ID goldens, sequential-wipe chronology, canonical transition parity,
domain and panel separation, source-digest coverage, draft-only candidate
freezing, singleton-prelude behavior, four-seat continuation isolation,
resealed semantic tampering, immutable-writer races, and receipt fail-closure.
They also include challenger-seed substitution, oversized/symlink/non-regular
CLI ledger inputs, serde-level trajectory bounds, and the no-stdout-on-failure
receipt invariant.

The PR-sized deterministic reference battery completes 125 corrected-rules
four-player AAAAA games (80 turns each), for exactly 10,000 randomized
reachable transition/compiler comparisons. Every transition checks dense
semantics against canonical scoring and `GameState::transition`, clone/apply
parity, canonical hashes and bytes, ledger replay, terminal scoring, and the
no-habitat-bonus research contract. This is finite engineering evidence, not a
gameplay-strength result or the separate 10,000-game/1,000,000-transition
release battery.
