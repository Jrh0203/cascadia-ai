# Feature-Schema Activation Census V1: Base Result

Date: 2026-06-16

Interim classification:
**`feature_schema_activation_census_base_complete_legacy_pending`**

Base scientific BLAKE3:
`60fc2955c5babdfc5b553a9e132fe9899e81fd9e1cfeb069a0ed82a5316041e1`

This is a preregistered intermediate result, not the final F1 verdict. The
modern V2, graded-action, candidate-factor, and hierarchical-cache evidence
is complete. The historical `legacy-nnue-v1-5197` and
`legacy-mid-v4opp-11231` activation stream is still being materialized and
will merge as disjoint supplemental evidence.

## Distributed Execution

One immutable scanner bundle was checksum-verified across john1, john3, and
john4 before production work began. Evidence ownership remained
`BLAKE3(evidence_id) mod 4`:

| Shard | Host | Wall time | Evidence payloads | Report rows scanned |
|---:|---|---:|---:|---:|
| 0/4 | john1 | 26.24 s | 155 | 1,874,208 |
| 1/4 | john1 | 24.08 s | 156 | 3,676,626 |
| 2/4 | john3 | 38.87 s | 142 | 2,486,756 |
| 3/4 | john4 | 35.06 s | 165 | 3,350,106 |

Shards 0, 2, and 3 ran concurrently. Shard 1 began immediately after shard 0.
Remote reports and channel-detail files were collected with byte-exact
SHA-256 verification. The four-shard merge rejected overlap and completed in
0.22 seconds.

## Evidence Coverage

The base sweep validated and scanned:

- all 2,135,111 train and 860,203 validation graded candidates;
- the aligned 2,135,111 train and 860,203 validation candidate-factor rows;
- 5,397,068 hierarchical group, query, or item rows across train and
  validation caches;
- 618 unique manifested evidence payloads;
- all four focal seats;
- opening, early, middle, and late phases; and
- 78 of 78 implemented non-legacy blocks.

The remaining 31 implemented blocks are exactly the seven V1 and 24
mid-v4opp blocks awaiting the historical sparse activation stream. The two
future schemas remain explicitly unimplemented and unmeasurable.

## Whole-Block Findings

Five measured blocks were dead across the entire open corpus:

| Block | Width | Interpretation |
|---|---:|---|
| `v2.market.coordinates_rotation` | 32 | Structural zero padding; pure wasted learned capacity |
| `v2.global.habitat_bonus` | 1 | Correctly zero under the no-habitat-bonus target ruleset |
| `graded.action.wipe_count` | 1 | No wipe action present in the frozen corpus |
| `graded.action.wipe_masks` | 80 | No wipe action present in the frozen corpus |
| `hierarchical.draft_query_context.constant` | 1 | Structural zero channel |

Eight blocks were constant. In addition to the dead blocks above:

- `v2.global.player_count` is constant under frozen four-player play;
- `v2.global.scoring_cards` is constant under AAAAA; and
- `hierarchical.group_state.market_mask` is constant because all four market
  slots are present in every measured group.

These values may remain in a lossless interchange schema, but a
ruleset-specific learned model should not spend trainable input capacity on
them.

## Partial Dead Capacity

The census also found dead or constant subranges inside active blocks:

- compact board coordinates: 2 of 184 channels;
- primary terrain: 7 of 460;
- secondary terrain: 6 of 552;
- rotation: 9 of 552;
- allowed wildlife: 5 of 460;
- placed wildlife: 11 of 552;
- staged market: 36 of 124;
- hierarchical group-state board entities: 41 of 2,852;
- hierarchical group-state global: 19 dead and 25 constant of 96;
- hierarchical draft/tile-query factors: 81 dead of 117 in each block;
- hierarchical staged-public tensors: 36 dead and 40 constant of 158; and
- hierarchical wildlife item local geometry: 5 dead of 210.

Some of these are intentional padding or domain restrictions. Others may be
duplicated or unreachable channels. F1 names them precisely so the R0/R1
representation tournament can remove only proven waste and preserve every
semantic distinction.

Only one measured block contained channels below the preregistered
`1e-4` activation threshold:
`hierarchical.tile_item.descendant_min` had two rare channels out of 20 over
1,198,315 rows.

## Aliases And Collisions

Within each evidence shard, deterministic sampled candidate fingerprints
were byte-verified:

- 46,791 of 2,995,314 graded candidate rows sampled;
- 46,813 of 2,995,314 factor candidate rows sampled;
- zero verified representation collisions; and
- zero cryptographic hash collisions.

Cross-shard candidate collisions remain explicitly unknown because rows split
across source evidence are not compared during merge. Channel alias analysis
did identify expected structural aliases and many deterministic sketch
candidates, especially in padded board tensors and hierarchical cache
blocks. Exact interpretation waits for the final report and, where needed,
targeted byte-level follow-up probes.

## Immediate Research Consequences

1. Keep exact semantic metadata in persisted schemas, but remove frozen
   ruleset constants from learned model inputs.
2. Do not carry the structurally dead 32-channel market coordinate/rotation
   block into a new compact model.
3. Gate wipe features behind evidence that the action family is actually
   present; do not pay 81 channels unconditionally.
4. Audit the hierarchical factor construction before increasing model
   capacity: several blocks contain large proven-dead subranges.
5. Preserve the 91/127-cell footprint tournament independently of these
   channel removals so spatial compression and semantic pruning remain
   separately attributable.
6. Do not finalize F1 or authorize a legacy-compatible replacement until the
   exact 5,197- and 11,231-column historical streams are measured.

## Integrity

The scanner test suite passed all seven focused tests and Ruff. Two local
smoke runs were byte-identical with scientific BLAKE3
`0f40dc49365a0386fe3f7a757a64a54a7c65b758c223d6402ed3cddb7fd6e887`.
The production bundle matched across all participating hosts. Test data,
gameplay evaluation, new teacher compute, hidden teacher values, cloud, and
external compute remained closed.
