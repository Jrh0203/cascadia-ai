# ADR 0148: R3 Exact Action Local-Patch Plus Global-Edit Foundation

Status: accepted; production foundation passed; matched MLX prototype authorized

Date: 2026-06-17

Research-plan item: R3

Experiment ID: `r3-action-edit-foundation-v1`

Schema: `r3-exact-action-local-patch-global-edit-v1`

## Context

R0 showed that compact dense disks do not automatically improve state-build
throughput. R2 then established an exact sparse public substrate with no fixed
spatial support: occupied tiles, legal frontier, habitat components, wildlife
anchors, public metadata, and exact canonical bytes.

Action ranking still risks recomputing an entire afterstate representation for
every legal candidate. A Cascadia decision can contain thousands of complete
actions even though each action directly changes one tile, at most one
wildlife-bearing cell, a bounded market surface, and the public objects touched
by those changes.

R3 tests the stronger representation claim:

> Encode the public state once, then encode each action as an exact local patch
> and global edit program.

The foundation must be mechanically exact before model architecture or
throughput claims are allowed.

## Decision

R3 is implemented as a standalone Rust crate:

`tools/r3_action_edit_census`

It opts out of the parent Cargo workspace and depends on:

- `cascadia-game` for legal actions, staged preludes, authoritative public
  afterstates, scoring, and D6 transforms;
- `cascadia-data` for public observation records and exact semantic supply;
  and
- the nested `r2_public_adapter` for the accepted R2 public state trunk
  semantics.

No shared game/data source, live queue, dashboard, or experiment ledger is
modified.

The adapter compiles R2's authoritative `model.rs` and `codec.rs` directly. It
does not copy their implementation and does not compile R2's unrelated census
or MLX-export modules. This is the only experiment-specific adapter.

## Candidate Extraction And Trunk Reuse

`PublicStateTrunk::prepare_action_edits` creates one
`PreparedPublicStateTrunk` per decision. Its `observe_legal_actions` method
accepts a market prelude, then:

1. validates and hashes the parent trunk once;
2. realizes the visible prelude once;
3. calls `GameState::evaluate_legal_turn_actions_with_context`;
4. extracts exact candidate board and geometry changes inside the engine's
   authoritative place/undo callback; and
5. assembles market, supply, player, score, and turn edits after enumeration.

The prepared handle retains the single packed trunk and canonical hash across
the canonical screen and paid-prelude sentinels. No complete `GameState` is
cloned or applied per candidate. Every emitted edit contains the same
parent-trunk hash. The focused oracle test applies every tested batch edit and
compares its normalized record with `GameState::preview_public_afterstate`.

## Public State Trunk

`PublicStateTrunk` contains exactly one:

1. ADR 0145 `SparsePublicState`; and
2. exact public semantic-supply snapshot.

The R2 trunk remains authoritative for occupied entities. Frontier, habitat
components, and wildlife motifs are deterministic projections. The supply
snapshot contains:

- five wildlife-bag counts;
- all 75 semantic tile-archetype counts;
- unseen and drawable tile totals; and
- the frozen semantic catalog identity.

Terminal targets are forced to zero before hashing and are absent from the R2
packed representation. Hidden order is not an input.

## Complete Action Factors

Every `ActionEdit` retains the complete factorization:

```text
prelude
  -> paired or independent draft
  -> tile destination and rotation
  -> optional wildlife destination
```

Prelude factors include:

- free three-of-a-kind replacement;
- an ordered, variable-length sequence of paid wipe masks; and
- exact visible before/after market snapshots.

The selected market objects are resolved after the prelude and retain exact
public tile semantics, semantic archetype identity when applicable, and
wildlife identity.

The placed tile retains:

- exact world coordinate;
- canonical game rotation;
- six world-directed edge terrains;
- complete wildlife eligibility; and
- keystone status.

## Exact Edit Program

The edit applies in two verified stages.

### Prelude stage

- verify the input market slots;
- apply every visible market slot change;
- verify and update active-player public metadata;
- apply the exact public semantic-supply delta.

### Placement stage

- add the selected habitat tile;
- update the wildlife destination when wildlife is placed;
- apply staged-to-afterstate market removals;
- update occupied count, wildlife counts, habitat maxima, and Nature Tokens;
- apply the public semantic-supply delta; and
- advance the completed-turn/current-seat metadata.

Board edits are canonical coordinate-keyed additions/removals/updates. The
current game action produces one tile addition and at most one wildlife update,
but the codec is intentionally general and variable-length.

After application, R3:

1. hashes the normalized public `PositionRecord`;
2. compares it with the authoritative
   `GameState::preview_public_afterstate` observation;
3. regenerates R2 geometry from the edited board;
4. recomputes frontier, component, and motif deltas; and
5. recomputes the canonical action view.

Any mismatch fails closed.

## Global Geometry Edits

R3 records exact active-board changes to:

- occupied tile objects;
- legal-frontier additions, removals, and in-place context updates;
- habitat-component additions, removals, and metadata updates; and
- wildlife-motif additions, removals, and neighborhood updates.

Habitat components use a content identity derived from terrain and sorted exact
membership, not the transient R2 numeric component ID. Frontier references are
resolved against those component identities in the canonical action view.
Frontier before/after equality uses the same semantic normalization. Raw R2
component numbers can change under traversal or D6 transformation and are not
stable rule identities.

Global references separately enumerate:

- before/after component keys;
- changed frontier coordinates;
- changed motif coordinates;
- changed market slots; and
- changed supply archetypes.

## Canonical Local Patch

The local patch is a complete radius-3 hex disk of 37 cells centered on the
proposed tile destination. Every cell distinguishes:

- outside the rules backing grid;
- empty nonfrontier;
- empty legal frontier; and
- occupied exact tile/wildlife semantics.

Occupied cells retain canonical local rotation and all six local-directed edge
terrains.

Radius 1 and radius 2 are exact crops of the same representation. They are
measured as potential model arms; they never truncate the global edit.

## D6 Contract

World application coordinates remain covariant under the authoritative
`cascadia-game` D6 contract.

The MLX-facing action view is canonical:

1. translate the tile destination to origin;
2. enumerate every authoritative D6 transform mapping the selected tile to
   rotation zero;
3. transform the local patch, wildlife destination, board edits, frontier
   edits, component memberships, motif neighborhoods, and directed edge bits;
4. sort every set-like edit collection canonically; and
5. choose the lexicographically smallest serialized view.

For all 12 transforms:

- transformed legal actions remain engine-authoritative;
- applying the transformed edit reproduces the transformed successor; and
- the complete canonical action view is byte-identical.

## Canonical Codecs

R3 defines two envelopes:

- `CSR3ST1\0`: one canonical R2 packed trunk plus one exact supply snapshot;
- `CSR3AE1\0`: one canonical variable-length action edit.

Both use a 16-bit schema version, length-delimited payloads, strict trailing
byte rejection, decode/re-encode identity, and BLAKE3 canonical identity.

The action codec has no fixed maximum for prelude length, touched components,
frontier edits, motif edits, global references, or legal action count.
Production verification applies the decoded edit, rather than only comparing
decoded fields.

## Distributed Census Contract

The frozen corpus is partitioned across four hosts by:

```text
(seed - cohort_first_seed) % 4 == shard_index
```

Every shard owns four train games and one validation game. A shard report binds
the complete reviewed source bundle, exact executable, deterministic owned
seed lists, exact histograms, and all verification counters. The CLI requires
an explicit shard index; the former unsharded invocation is not accepted.

The aggregate:

1. parses strict JSON with duplicate and unknown key rejection;
2. requires exactly one shard for every index `0..3`;
3. requires byte-identical runtime and source identity across shards and the
   aggregating executable;
4. validates deterministic seed ownership and frozen corpus coverage;
5. merges exact histogram bins before recomputing quantiles;
6. recomputes every promotion gate from merged evidence; and
7. omits paths, hostnames, timestamps, and input order from scientific output.

Forward and reverse shard-order aggregates must be byte-identical. A separate
order-proof artifact is bound to the aggregate scientific hash and exact file
bytes.

The production execution graph is generated by
`tools/r3_action_edit_campaign.py` and contains exactly 13 tasks: one immutable
fanout, four fail-closed runtime preflights, four fixed nonoverlapping shards,
one checksum collection, two aggregate orders, and one terminal order proof.
Each shard runs its four train games and one validation game through five Rayon
workers. Per-game evidence is private and merged in canonical seed order, so
parallel scheduling cannot affect scientific bytes.

## Pre-Production Smoke Correction

The first john4 smoke failed before production at raw seed `4,100,003`, turn
zero, transform `2`. The implementation compared raw frontier tokens, whose
habitat-component references contained transient numeric IDs. The transformed
board assigned a different number to semantically identical component
membership, creating one false frontier update.

The invalid bundle and witness are recorded in
`docs/v2/reports/r3-action-edit-foundation-v1-invalid-smoke-1.md`. The fix
canonicalizes frontier component references by terrain and sorted membership
before equality, and a permanent regression test freezes the witness. No
scientific result or production shard was emitted by the invalid run.

## Public Information Boundary

The exact successor is the normalized public afterstate before hidden refill.
This boundary includes:

- visible realized prelude outcomes;
- selected public market objects;
- placed public board objects;
- public Nature Tokens;
- visible market removals; and
- exact public semantic-supply belief after the action.

It excludes:

- hidden tile-stack order;
- hidden wildlife-bag order;
- excluded-tile identity;
- wildlife return insertion position;
- future refill realization;
- RNG seed;
- future actions; and
- terminal targets or learned labels.

Physical tile IDs are omitted under ADR 0143/0145 because semantically
identical physical copies have identical public rule behavior. Exact parity is
therefore normalized `PositionRecord` byte equality plus exact semantic-supply
equality, not private `GameState` byte equality.

## MLX Serving Contract

Any R3-compatible MLX model must:

1. encode the state trunk once per decision;
2. batch variable-length action edits;
3. resolve action references against cached trunk objects;
4. keep local radius and global edit as separate ablation axes; and
5. preserve Rust codec parity as a required test.

Materializing one dense afterstate per action defeats the R3 hypothesis and is
not a compatible implementation.

## Consequences

The preregistered 20-game production corpus completed across all four hosts.
The exact aggregate contains 1,600 decisions and 2,679,459 action-edit
applications. Every public-successor, semantic-supply, regenerated-global-edit,
codec, and D6 check passed.

Observed edit sizes were:

- median 55 tokens;
- P99 62 tokens;
- maximum 70 tokens;
- P99 4,915 packed bytes; and
- maximum 6,629 packed bytes.

All frozen compactness gates passed. Forward and reverse shard-order
aggregates were byte-identical with scientific BLAKE3
`9a3075bf4b9abb0ce05efad1856ce951163d04f41e619f83acdf77ee78130424`.

Radius-only locality did not pass as a lossless representation. Radius 1, 2,
and 3 completely covered 12.5015%, 42.5873%, and 58.2432% of actions,
respectively. The accepted R3 contract therefore remains a local patch plus
exact global component, motif, frontier, market, supply, and metadata edits.

R3 now authorizes a matched MLX prototype. It still does not establish:

- end-to-end extraction or inference throughput;
- offline value or action-ranking quality;
- search quality; or
- gameplay strength.

Those claims require a separately preregistered learned comparison. Full
results are in
`docs/v2/reports/r3-action-edit-foundation-v1-result.md`.
