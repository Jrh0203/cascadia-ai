# Cascadia v3 Schema Contracts

All schemas are versioned and must carry a `schema_id`. CPU implementations
should reject missing or incompatible schema ids before reading model, replay, or
search-root content.

## Schema Registry

The implementation keeps the original CPU fixture and greedy tensor contracts
active while adding expert-root contracts beside them:

| Schema ID | Artifact | Status |
|---|---|---|
| `cascadiav3.pre_gpu.v0` | Legacy CPU search-root JSONL fixtures | active |
| `greedy_policy_tensor_shard_v1` | Greedy behavior-cloning tensor shard | active |
| `cascadiav3.expert_root.v1` | Full legal-action expert root JSONL | active |
| `cascadiav3.expert_tensor_shard.v1` | Packed expert training shard | active target |

Validate the registry with:

```bash
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.validate_schema_registry --include-legacy --include-expert
```

## Coordinate Contract

| Schema | Field | Type | Rule |
|---|---|---:|---|
| `CanonicalHexCoord` | `q` | signed int | axial q coordinate |
| `CanonicalHexCoord` | `r` | signed int | axial r coordinate |
| `CanonicalHexCoord` | `s` | signed int | computed as `s = -q - r`; serialized only for checksum/debug if desired |
| `CanonicalHexCoord` | `radius6_member` | bool | true iff `max(abs(q), abs(r), abs(s)) <= 6` |
| `CanonicalHexCoord` | `cell_index` | uint16/null | stable index `0..126` when `radius6_member=true`; null for overflow |
| `OverflowHexCoord` | `q`, `r`, `s` | signed int | exact legal coordinate outside radius 6 |
| `OverflowHexCoord` | `owner_seat` | uint8 | board owner for the overflow entity |
| `OverflowHexCoord` | `placement_id` | uint32 | stable local id for exact joins from tile/wildlife/action records |

Radius 6 is canonical and contains 127 cells. Overflow is exact: legal
out-of-radius states remain addressable by coordinate and owner, and no legal
state may be clipped or remapped into the radius 6 fast path.

## Token Schemas

| Token | Required Fields | Optional Fields | Notes |
|---|---|---|---|
| `GameToken` | `turn_index`, `active_seat`, `phase`, `cleanup_state`, `scoring_card_id`, `schema_id` | `ruleset_id`, `version_mix_id` | One global token per state. |
| `PlayerToken` | `seat`, `relative_seat`, `nature_tokens`, `turns_remaining`, `visible_score_total` | `model_version_id`, `style_id` | One per player; oriented to active seat for value vector backup. |
| `TileToken` | `tile_instance_id`, `owner_kind`, `owner_seat_or_slot`, `terrain_edges`, `wildlife_icons`, `keystone`, `rotation`, `coord_ref`, `placement_age` | `frontier_flags`, `habitat_region_ids` | Represents placed tiles and visible market tiles. |
| `WildlifeToken` | `wildlife_instance_id`, `owner_kind`, `owner_seat_or_slot`, `species`, `coord_ref`, `paired_tile_ref` | `placement_age`, `bag_draw_id` | Represents placed wildlife and visible market wildlife. |
| `FrontierToken` | `owner_seat`, `coord_ref`, `adjacent_tile_refs`, `legal_rotations_mask` | `frontier_age`, `terrain_match_summary` | Legal empty adjacent cells for tile placement. |
| `SupplyToken` | `supply_kind`, `observable_count`, `uncertainty_bucket`, `market_slot_summary` | `bag_count_vector`, `tile_stack_bucket` | Visible market and remaining public/uncertain supply summaries. |
| `ScoreToken` | `seat`, `score_category`, `current_points`, `score_card_id`, `potential_bucket` | `score_to_go_bucket`, `category_rank_bucket` | Supports score decomposition and score/rank/vector auxiliary heads. |
| `ActionToken` | `action_id`, `active_seat`, `cleanup_choice`, `nature_spend`, `draft_slot`, `tile_ref`, `wildlife_ref`, `target_coord_ref`, `rotation`, `wildlife_coord_ref` | `chance_policy_hint`, `factor_labels` | One per exact legal compound action. Legal action identity is simulator-owned. |

`coord_ref` points to either a `CanonicalHexCoord` radius 6 cell index or an
`OverflowHexCoord` entity id. The tokenizer must preserve which representation
was used.

## Relation And C-GAB Templates

| Schema | Relation | Applies To | Value Domain | Purpose |
|---|---|---|---|---|
| `RelationTemplate` | `same_board` | tokens with board owner | bool | Connects objects on the same player board. |
| `RelationTemplate` | `same_market_slot` | market tile/wildlife/action | bool | Connects draft components by visible slot. |
| `RelationTemplate` | `tile_wildlife_pairing` | market tile and wildlife | bool | Marks currently paired market objects. |
| `RelationTemplate` | `adjacent_direction` | board coordinates | `0..5`/none | Encodes hex neighbor direction. |
| `RelationTemplate` | `distance_bucket` | board coordinates | `0,1,2,3,4,5,6,7+` | Encodes local/global geometry without a dense board. |
| `RelationTemplate` | `terrain_continuity` | tile edges/cells | terrain id/bool | Connects compatible habitat continuity candidates. |
| `RelationTemplate` | `same_species` | wildlife/action tokens | species id/bool | Connects wildlife of the same type. |
| `RelationTemplate` | `action_draft_slot` | action to market token | slot id/bool | Links `ActionToken` to its selected draft slot. |
| `RelationTemplate` | `action_target_tile_coordinate` | action to tile/frontier coordinate | coord ref/bool | Links tile placement target. |
| `RelationTemplate` | `action_target_wildlife_coordinate` | action to wildlife coordinate | coord ref/bool | Links wildlife placement target. |
| `CGabTemplate` | `template_id` | all relation templates | uint16 | Stable id for bias lookup/mixing. |
| `CGabTemplate` | `layer_group_mask` | transformer layers | bitset | Identifies where the template may contribute. |
| `CGabTemplate` | `sparsity_format` | token pairs | enum | CPU dry run may use sparse edge lists; GPU implementation may pack differently. |

Templates encode legal geometry and object identity only. They must not encode
manual score bonuses or hidden evaluator heuristics.

## Search Root Record

| Field | Type | Required | Rule |
|---|---|---:|---|
| `schema_id` | string | Yes | Must match tokenizer/action/replay schema. |
| `state_hash` | bytes/string | Yes | Stable canonical state hash before root search. |
| `active_seat` | uint8 | Yes | Seat whose legal action set is represented. |
| `legal_actions` | array `ActionToken` or action ids | Yes | Exact simulator-enumerated legal actions in stable order. |
| `priors` | float array | Yes | One prior per legal action; sums checked with tolerance. |
| `visits` | uint32 array | Yes | One visit count per legal action. |
| `per_action_Q` | float array or vector array | Yes | Per-action Q labels aligned to `legal_actions`. |
| `per_action_Q_variance` | float array | No | Per-action target variance when labels are rollout means. |
| `per_action_Q_count` | uint32 array | No | Number of rollout/search samples contributing to each Q label. |
| `per_action_truncated_count` | uint32 array | No | Number of rollout samples scored from a resource-exhausted truncated continuation. |
| `selected_action` | action id/index | Yes | Action actually played from the root. |
| `chance_samples` | array | Yes | Chance branches sampled or widened from this root. |
| `final_score_vector` | int/float array length 4 | Yes | Final scores by seat. |
| `score_decomposition` | structured map | Yes | Wildlife, habitat, nature-token, and total components by player. |
| `rank_vector` | array length 4 | Yes | Final rank or rank distribution label. |
| `checksum` | bytes/string | Yes | Covers serialized state/action/label payload. |

Search root tables are the primary replay artifact for self-play/search-root
replay labels.

## Expert Root v1 Additions

`cascadiav3.expert_root.v1` extends the base search-root table with:

- `root_replay`: seed, replay prefix, deterministic market prelude, parent and
  staged hashes.
- `ruleset_id`, source hash, binary hash, actor/opponent/model/search/RNG
  identities.
- `action_ids`, `afterstate_hashes`, `afterstate_public_hashes`,
  `exact_afterstate_score_active`, `per_action_score_to_go`, and
  `per_action_Q_valid`, all aligned one-to-one with `legal_actions`.
- `chance_samples` carrying seed, probability/logprob, before/after hashes,
  public delta, and private audit hash outside model observations.

The target identity is:

```text
per_action_Q = active-seat final raw score estimate
per_action_score_to_go = per_action_Q - exact_afterstate_score_active
```

Legal actions with no valid Q label remain in the arrays with
`per_action_Q_valid=false`; Q and pairwise losses must ignore those entries.

## Replay Shard Manifest

| Field | Type | Required | Rule |
|---|---|---:|---|
| `schema_id` | string | Yes | Schema version for every record in the shard. |
| `source_generator` | string | Yes | Program/config identifier that produced the shard. |
| `seed_domain` | string | Yes | Describes deterministic seed derivation. |
| `record_count` | uint64 | Yes | Exact number of records. |
| `checksum` | bytes/string | Yes | Whole-shard checksum. |
| `scientific_eligibility` | enum | Yes | `dry_run`, `debug`, `training_candidate`, `evaluation_locked`. |
| `created_at_utc` | timestamp | Yes | Manifest timestamp. |
| `format` | enum | Yes | `jsonl`, `binary`, or future packed format. |
| `notes` | string | No | Free-form non-authoritative annotation. |

Dry-run JSONL is acceptable before GPU work. Compact binary is expected before
large replay generation.

## Serving Prefilter Eval Report

| Field | Type | Required | Rule |
|---|---|---:|---|
| `experiment_id` | string | Yes | Identifies the replay-only prefilter evaluation. |
| `checkpoint` | path/string | Yes | Checkpoint whose scores produced the ranking. |
| `val` | path/string | Yes | Replay shard evaluated without mutation or training. |
| `k_values` | int array | Yes | Retained widths evaluated, currently including 4, 8, 16, 24, and 32. |
| `models.<name>.metrics.prefilter.<K>.recall` | float | Yes | Fraction of roots where teacher-best action is retained in top K. |
| `models.<name>.metrics.prefilter.<K>.mean_oracle_regret` | float | Yes | Mean gap between teacher-best Q and best teacher Q among retained actions. |
| `models.<name>.serving_decision.recommended_k` | int/null | Yes | Smallest K passing the configured recall/regret gate, or null. |
| `per_root_out` | path/string/null | Yes | Optional JSONL with exact retained action ids by root and K. |

Per-root prefilter JSONL rows must include `state_hash`, `action_count`,
`teacher_best`, `model_selected`, `ranked_action_ids`, `ranked_predicted_q`,
`ranked_teacher_q`, and a `prefilter` map keyed by retained K. Action ids in the
per-root file must be simulator-owned `ActionToken.action_id` values copied from
the replay shard, not regenerated by the evaluator.

## Model Config

| Config | Layers | `d_model` | Heads | FFN | Parameter Target | First Use |
|---|---:|---:|---:|---:|---:|---|
| `CascadiaFormer-Zero-S` | 8 | 384 or 512 | 8 | implementation-selected | small ablation target | First CPU/GPU smoke target; verify tensor shapes/legal logits. |
| `CascadiaFormer-Zero-M` | 12 | 768 | 12 | 2048 or 3072 | roughly 80M-120M | Zeus-scale training after pipeline readiness. |
| `CascadiaFormer-Zero-L` | 15 | 1024 | 16 or 32 | implementation-selected | larger scaling target | Later scaling only after S/M evidence. |

The first model smoke should use the smallest S-compatible shape or a mock CPU
backend sufficient to validate token tensors, C-GAB inputs, legal-action query
logits, value vector outputs, and score/rank/vector auxiliary heads.

## Validation Gate Registry

| Gate ID | Command | Expected Artifact | Pass/Fail Rule | GPU Requirement |
|---|---|---|---|---:|
| `plan-package-present` | `find cascadiav3 -maxdepth 2 -type f | sort` | six documentation files | Exact approved package present | No |
| `schema-fixtures-roundtrip` | future CPU command | fixture output and checksum | Deserialize/serialize without schema or checksum drift | No |
| `radius6-census` | future CPU command | coverage/overflow report | Reports radius 6 membership and exact overflow count; no silent clipping | No |
| `legal-action-golden` | future CPU command | legal action fixture report | Fixed roots match expected legal action ids/order | No |
| `score-decomposition-golden` | future CPU command | score decomposition report | Category totals sum to final scores exactly | No |
| `cgab-template-smoke` | future CPU command | relation template summary | Same inputs produce stable template ids and memory report | No |
| `search-root-roundtrip` | future CPU command | replay shard and manifest | Root table labels align to legal actions and checksum validates | No |
| `tiny-cpu-model-smoke` | future CPU command | tensor shape report | Legal-action logits, value vector, and auxiliary heads have expected dimensions | No |
| `GPU_HANDOFF` | manual checklist | handoff bundle | All CPU prerequisites complete before GPU-only work | No |
