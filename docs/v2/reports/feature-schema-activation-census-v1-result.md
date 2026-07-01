# Feature-Schema Activation Census V1: Final Result

Date: 2026-06-16

Final classification:
**`feature_schema_activation_census_complete`**

Classification scientific BLAKE3:
`f7f8559431f53a461f9464e14ef4cee2119cf3ddcf0bf4e3dd9126ab8bdd91fb`

Merged-census scientific BLAKE3:
`8906487b91aa0da25f388e2075d15150c8d1499a022cbf2d231987b37f182e65`

F1 is complete. Every implemented modern and historical feature block was
measured over its preregistered open evidence, forward and reverse merges were
scientifically identical, and all closed domains remained closed.

## Decision

The census supports a clean replacement of the historical 441-cell learned
input with a compact, recentered representation, provided the replacement
retains exact overflow entities and is tested independently from semantic
feature pruning.

It does **not** authorize a lossy 121-cell crop. F2 established that no regular
121-cell hex disk exists: regular radius-5 and radius-6 disks contain 91 and
127 cells. Radius 6 retained every measured event, while legal adversarial
boards still require exact overflow.

The historical feature layouts contain large amounts of proven inactive
capacity:

| Schema | Width | Never-active channels | Remaining active-capacity channels |
|---|---:|---:|---:|
| `legacy-nnue-v1-5197` | 5,197 | 4,297 (82.68%) | 900 |
| `legacy-mid-v4opp-11231` | 11,231 | 8,597 (76.55%) | 2,634 |

These are corpus-wide channel observations, not a claim that every inactive
channel is semantically removable in every ruleset. The R0 tournament must
separate spatial compaction, frozen-ruleset pruning, and architectural changes
so their effects remain attributable.

## Coverage

The final census merged 628 unique evidence payloads:

- 2,135,111 train and 860,203 validation graded candidates;
- the aligned 2,135,111 train and 860,203 validation factor rows;
- 5,397,068 hierarchical group, query, and item rows;
- 200,000 balanced historical sparse activations;
- all four focal seats; and
- opening, early, middle, and late phases.

All 109 implemented blocks were measured: 78 modern blocks and 31 historical
blocks. Proposed-only schemas remained explicitly unimplemented and
unmeasured.

## Historical Findings

### The 441-cell core is mostly inactive

Both historical schemas inherit a 4,851-channel cell core. Across 200,000
balanced states, 4,258 channels (87.77%) never activated. The sparse extractor
still emitted exactly 25 active cell-core features per state, but the fixed
21-by-21 coordinate lattice forced the network to reserve capacity for remote
coordinates that generated play never occupied.

This is direct evidence that the dense 441-cell indexing is a poor learned
representation. It agrees with F2's independent footprint result, while F2
supplies the safety condition: recenter and retain exact overflow rather than
silently truncating legal states.

### The historical mid tail is entirely dead

`legacy.mid_tail_historical_adjacency_prefix` occupies 301 columns in the
11,231-wide champion schema. All 301 columns were zero in all 200,000 states.
The block is the frozen historical adjacency-prefix defect, not the intended
extended supply tail. A successor must not reproduce or train through it.

### Opponent detail is real signal

The 369-column `legacy.v4opp` block activated in every state. Thirty-one
channels were unreachable in this fixed corpus and four were rare, but the
block as a whole is live. This is consistent with the champion's measured
strength gain from detailed opponent modeling. Compact successors should
preserve the semantics and test more expressive relational alternatives,
rather than deleting opponent detail with the dead spatial capacity.

### Market and supply context are live

The historical market, tile-bag terrain marginal, and tile-bag wildlife
marginal blocks had no dead channels. They are not candidates for blind
removal. Their interaction with opponent demand remains a high-priority
representation hypothesis.

## Modern Findings

Six whole implemented blocks were dead in their measured domains:

| Block | Width | Interpretation |
|---|---:|---|
| `v2.market.coordinates_rotation` | 32 | Structural padding with no learned signal |
| `v2.global.habitat_bonus` | 1 | Correctly zero under the target no-bonus ruleset |
| `graded.action.wipe_count` | 1 | Wipe actions absent from the frozen corpus |
| `graded.action.wipe_masks` | 80 | Wipe actions absent from the frozen corpus |
| `hierarchical.draft_query_context.constant` | 1 | Structural zero |
| `legacy.mid_tail_historical_adjacency_prefix` | 301 | Historical extraction defect |

Nine whole blocks were constant. In addition to the dead blocks, the fixed
AAAAA four-player corpus made `v2.global.player_count`,
`v2.global.scoring_cards`, and `hierarchical.group_state.market_mask`
constant.

The census also found substantial dead subranges in active hierarchical
blocks, including 81 of 117 channels in each draft/tile-query factor block.
Those blocks require construction-level ablations before adding model
capacity.

## Alias And Collision Status

The result preserves four distinct statuses:

- `no_channel_alias_detected`;
- `structural_or_empirical_alias`;
- `unknown`; and
- `unknown_hash_candidate`.

The modern candidate sample byte-verified 46,791 graded rows and 46,813 factor
rows with zero representation collisions. Cross-shard candidate collisions
and historical sparse-vector collisions remain explicitly unknown; the census
does not turn absence of evidence into a proof of injectivity.

## Distributed Execution

The modern sweep ran as four deterministic evidence shards across john1,
john3, and john4. The independently generated historical dataset was then
fanned out and scanned as four more shards on the same hosts. Remote artifacts
were collected with tree and file checksum verification.

The complete evidence was merged twice in opposite input orders. Both merges
produced the same scientific hash:
`8906487b91aa0da25f388e2075d15150c8d1499a022cbf2d231987b37f182e65`.
The terminal classifier passed every preregistered gate.

## R0 Requirements

The compact-state tournament should now enforce these independent arms:

1. Historical 21-by-21/441 encoding as a diagnostic control.
2. Recentered radius-6/127 encoding with exact overflow.
3. Recentered radius-5/91 encoding with exact overflow.
4. Recentered radius-4/61 stress arm with exact overflow.
5. Exact entity-list control without a fixed spatial disk.
6. Proven-dead/frozen-channel pruning as a separate ablation.

Every arm must use the same examples, targets, model budget, optimizer,
evaluation seeds, and D6 contract. Success requires exact round-trip behavior,
no score regression, and a material end-to-end throughput gain. A faster arm
that loses legal information or player strength is a rejection.

## Integrity

The scanner and terminal classifier passed 10 focused tests and Ruff. The
manifest scientific BLAKE3 is
`1ebc86d586453548cb6109780f3c86d05867936f61eb1e34690b7bfd086fc9de`.
Test evaluation, gameplay evaluation, new teacher compute, hidden teacher
values, cloud compute, and other external compute remained closed.
