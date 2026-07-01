use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Seek, Write},
    path::{Path, PathBuf},
};

use cascadia_game::{
    DraftChoice, GameConfig, GameState, HexCoord, MarketPrelude, MarketSlot, PublicSupply,
    Rotation, ScoreBreakdown, Tile, TilePlacement, TurnAction, Wildlife, WildlifeWipe, score_board,
};
use serde::{Deserialize, Serialize};

use super::public_supply::{decode_public_supply, encode_public_supply};
use super::{
    CollectionProvenance, DataError, DatasetSplit, FEATURE_SCHEMA, PositionRecord, RECORD_SIZE,
    RankingShardManifest, checksum_file, collection_provenance, collection_provenance_matches,
    read_array, unix_seconds, write_manifest_atomic, write_slice,
};

pub const GRADED_ORACLE_DATASET_SCHEMA_VERSION: u16 = 1;
pub const GRADED_ORACLE_FEATURE_SCHEMA: &str = "complete-action-graded-oracle-v1";
pub const GRADED_ORACLE_TARGET_SCHEMA: &str = "screen-r600-r1200-r4800-graded-v1";
pub const GRADED_ORACLE_SHARD_MAGIC: &[u8; 8] = b"CSD2GOV\0";
pub const GRADED_ORACLE_HEADER_SIZE: usize = 112;
pub const GRADED_ORACLE_GROUP_HEADER_SIZE: usize = 960;
pub const GRADED_ORACLE_ACTION_FEATURE_SIZE: usize = 128;
pub const GRADED_ORACLE_CANDIDATE_RECORD_SIZE: usize = 224;
pub const GRADED_ORACLE_MAX_WILDLIFE_WIPES: usize = 20;
pub const GRADED_ORACLE_STAGED_MARKET_SIZE: usize = 32;

pub const GRADED_SOURCE_TOP_SCREEN: u16 = 1 << 0;
pub const GRADED_SOURCE_CHAMPION_FRONTIER: u16 = 1 << 1;
pub const GRADED_SOURCE_CHAMPION_SELECTED: u16 = 1 << 2;
pub const GRADED_SOURCE_SENTINEL: u16 = 1 << 3;
pub const GRADED_SOURCE_SUBSTANTIAL_TOP: u16 = 1 << 4;
pub const GRADED_SOURCE_BEST_CHAMPION_FRONTIER: u16 = 1 << 5;
pub const GRADED_SOURCE_R600: u16 = 1 << 6;
pub const GRADED_SOURCE_R1200: u16 = 1 << 7;
pub const GRADED_SOURCE_R4800: u16 = 1 << 8;
pub const GRADED_SOURCE_COMPLETE_LEGAL: u16 = 1 << 9;

pub const GRADED_FIDELITY_R600: u16 = 1 << 0;
pub const GRADED_FIDELITY_R1200: u16 = 1 << 1;
pub const GRADED_FIDELITY_R4800: u16 = 1 << 2;

const NONE: u8 = u8::MAX;
const SCORE_COMPONENTS: usize = 11;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GradedOracleAuditInput {
    pub path: String,
    pub blake3: String,
    pub raw_seed: u64,
    pub audit_protocol_id: String,
    pub audit_config_blake3: String,
    pub source_blake3: String,
    pub executable_blake3: String,
    pub model_json_blake3: String,
    pub model_safetensors_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GradedOracleTeacherIdentity {
    pub audit_protocol_id: String,
    pub audit_config_blake3: String,
    pub source_blake3: String,
    pub executable_blake3: String,
    pub model_json_blake3: String,
    pub model_safetensors_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GradedOracleSourceManifest {
    pub path: String,
    pub dataset_id: String,
    pub manifest_blake3: String,
    pub seeds: Vec<u64>,
}

impl From<&GradedOracleAuditInput> for GradedOracleTeacherIdentity {
    fn from(input: &GradedOracleAuditInput) -> Self {
        Self {
            audit_protocol_id: input.audit_protocol_id.clone(),
            audit_config_blake3: input.audit_config_blake3.clone(),
            source_blake3: input.source_blake3.clone(),
            executable_blake3: input.executable_blake3.clone(),
            model_json_blake3: input.model_json_blake3.clone(),
            model_safetensors_blake3: input.model_safetensors_blake3.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GradedOracleDatasetConfig {
    pub output: PathBuf,
    pub split: DatasetSplit,
    pub audit_inputs: Vec<GradedOracleAuditInput>,
    pub resume: bool,
}

impl GradedOracleDatasetConfig {
    fn validate(&self) -> Result<(), DataError> {
        if self.audit_inputs.is_empty() {
            return Err(DataError::InvalidConfig(
                "graded-oracle dataset requires audit inputs",
            ));
        }
        let mut previous_seed = None;
        for input in &self.audit_inputs {
            if previous_seed.is_some_and(|seed| seed >= input.raw_seed)
                || input.path.trim().is_empty()
                || input.blake3.trim().is_empty()
                || input.audit_protocol_id.trim().is_empty()
                || input.audit_config_blake3.trim().is_empty()
                || input.source_blake3.trim().is_empty()
                || input.executable_blake3.trim().is_empty()
                || input.model_json_blake3.trim().is_empty()
                || input.model_safetensors_blake3.trim().is_empty()
            {
                return Err(DataError::InvalidConfig(
                    "graded-oracle audit inputs are incomplete or not strictly ordered",
                ));
            }
            previous_seed = Some(input.raw_seed);
        }
        let first = &self.audit_inputs[0];
        if self.audit_inputs.iter().any(|input| {
            input.audit_protocol_id != first.audit_protocol_id
                || input.audit_config_blake3 != first.audit_config_blake3
                || input.source_blake3 != first.source_blake3
                || input.executable_blake3 != first.executable_blake3
                || input.model_json_blake3 != first.model_json_blake3
                || input.model_safetensors_blake3 != first.model_safetensors_blake3
        }) {
            return Err(DataError::InvalidConfig(
                "graded-oracle audit inputs do not share one frozen identity",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GradedOracleDatasetManifest {
    pub schema_version: u16,
    pub dataset_id: String,
    pub feature_schema: String,
    pub position_feature_schema: String,
    pub target_schema: String,
    pub group_header_size: usize,
    pub candidate_record_size: usize,
    pub action_feature_size: usize,
    pub public_supply_size: usize,
    pub maximum_wildlife_wipes: usize,
    pub game: GameConfig,
    pub split: DatasetSplit,
    pub seeds: Vec<u64>,
    pub requested_games: usize,
    pub completed_games: usize,
    pub total_groups: usize,
    pub total_records: usize,
    pub teacher: GradedOracleTeacherIdentity,
    pub audit_inputs: Vec<GradedOracleAuditInput>,
    #[serde(default)]
    pub source_manifests: Vec<GradedOracleSourceManifest>,
    pub created_unix_seconds: u64,
    pub updated_unix_seconds: u64,
    pub provenance: CollectionProvenance,
    pub shards: Vec<RankingShardManifest>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GradedOracleEstimate {
    pub mean: f32,
    pub stddev: f32,
    pub samples: u16,
}

impl GradedOracleEstimate {
    pub fn validate(self) -> Result<(), DataError> {
        if !self.mean.is_finite()
            || !self.stddev.is_finite()
            || self.stddev < 0.0
            || (self.samples == 0 && (self.mean != 0.0 || self.stddev != 0.0))
        {
            return Err(DataError::InvalidConfig(
                "graded-oracle estimate is invalid",
            ));
        }
        Ok(())
    }

    fn write_to(self, bytes: &mut [u8], offset: &mut usize) {
        write_slice(bytes, offset, &self.mean.to_le_bytes());
        write_slice(bytes, offset, &self.stddev.to_le_bytes());
        write_slice(bytes, offset, &self.samples.to_le_bytes());
        *offset += 2;
    }

    fn read_from(bytes: &[u8], offset: &mut usize) -> Self {
        let mean = f32::from_le_bytes(read_array(bytes, offset));
        let stddev = f32::from_le_bytes(read_array(bytes, offset));
        let samples = u16::from_le_bytes(read_array(bytes, offset));
        *offset += 2;
        Self {
            mean,
            stddev,
            samples,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GradedOracleActionFeatures {
    pub same_slot_independent: u8,
    pub draft_kind: u8,
    pub tile_slot: u8,
    pub wildlife_slot: u8,
    pub tile_id: u8,
    pub tile_terrain_a: u8,
    pub tile_terrain_b: u8,
    pub tile_wildlife_mask: u8,
    pub tile_keystone: u8,
    pub drafted_wildlife: u8,
    pub tile_q: i8,
    pub tile_r: i8,
    pub rotation: u8,
    pub wildlife_present: u8,
    pub wildlife_q: i8,
    pub wildlife_r: i8,
    pub replace_three_of_a_kind: u8,
    pub wipe_count: u8,
    pub wipe_masks: [u8; GRADED_ORACLE_MAX_WILDLIFE_WIPES],
    pub staged_active_nature_tokens: u8,
    pub staged_market_entities: [[u8; 8]; 4],
    pub staged_public_supply: [u8; super::PUBLIC_SUPPLY_SIZE],
    pub immediate_score: u16,
    pub immediate_deltas: [i16; SCORE_COMPONENTS],
}

/// Owned invariant context for encoding actions from an already-staged draft
/// decision. R2-MAP commits each sequential market choice before requesting
/// draft inference, so every candidate shares this public context and the same
/// pre-action score.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GradedOracleStagedActionContext {
    market_tiles: [Option<Tile>; 4],
    market_wildlife: [Option<Wildlife>; 4],
    staged_active_nature_tokens: u8,
    staged_market_entities: [[u8; 8]; 4],
    staged_public_supply: [u8; super::PUBLIC_SUPPLY_SIZE],
    before: ScoreBreakdown,
}

impl GradedOracleStagedActionContext {
    pub fn observe(game: &GameState) -> Self {
        let position = PositionRecord::observe(game, 0);
        Self {
            market_tiles: game.market().tiles,
            market_wildlife: game.market().wildlife,
            staged_active_nature_tokens: position.nature_tokens[0],
            staged_market_entities: position.market_entities,
            staged_public_supply: encode_public_supply(game.public_supply()),
            before: score_board(
                &game.boards()[game.current_player()],
                game.config().scoring_cards,
            ),
        }
    }

    /// Encode one action using an exact score already obtained while the
    /// canonical production enumerator had that afterstate applied.
    pub fn encode_after_score(
        &self,
        action: &TurnAction,
        after: ScoreBreakdown,
    ) -> Result<[u8; GRADED_ORACLE_ACTION_FEATURE_SIZE], DataError> {
        if action.replace_three_of_a_kind || !action.wildlife_wipes.is_empty() {
            return Err(DataError::InvalidConfig(
                "staged action context requires a post-market draft action",
            ));
        }
        let (draft_kind, tile_slot, wildlife_slot, same_slot_independent) = match action.draft {
            DraftChoice::Paired { slot } => (0, slot.index() as u8, slot.index() as u8, 0),
            DraftChoice::Independent {
                tile_slot,
                wildlife_slot,
            } => (
                1,
                tile_slot.index() as u8,
                wildlife_slot.index() as u8,
                u8::from(tile_slot == wildlife_slot),
            ),
        };
        let tile = self.market_tiles[usize::from(tile_slot)].ok_or(DataError::InvalidConfig(
            "staged action tile slot is unavailable",
        ))?;
        let wildlife = self.market_wildlife[usize::from(wildlife_slot)].ok_or(
            DataError::InvalidConfig("staged action wildlife slot is unavailable"),
        )?;
        Ok(GradedOracleActionFeatures {
            same_slot_independent,
            draft_kind,
            tile_slot,
            wildlife_slot,
            tile_id: tile.id.0,
            tile_terrain_a: tile.terrain_a as u8,
            tile_terrain_b: tile.terrain_b.map_or(NONE, |terrain| terrain as u8),
            tile_wildlife_mask: tile.wildlife.bits(),
            tile_keystone: u8::from(tile.keystone),
            drafted_wildlife: wildlife as u8,
            tile_q: action.tile.coord.q,
            tile_r: action.tile.coord.r,
            rotation: action.tile.rotation.get(),
            wildlife_present: u8::from(action.wildlife.is_some()),
            wildlife_q: action.wildlife.map_or(0, |coord| coord.q),
            wildlife_r: action.wildlife.map_or(0, |coord| coord.r),
            replace_three_of_a_kind: 0,
            wipe_count: 0,
            wipe_masks: [0; GRADED_ORACLE_MAX_WILDLIFE_WIPES],
            staged_active_nature_tokens: self.staged_active_nature_tokens,
            staged_market_entities: self.staged_market_entities,
            staged_public_supply: self.staged_public_supply,
            immediate_score: after.base_total,
            immediate_deltas: score_deltas(self.before, after),
        }
        .to_bytes())
    }
}

impl GradedOracleActionFeatures {
    pub fn observe(game: &GameState, action: &TurnAction) -> Result<Self, DataError> {
        if action.wildlife_wipes.len() > GRADED_ORACLE_MAX_WILDLIFE_WIPES {
            return Err(DataError::InvalidConfig(
                "graded-oracle action exceeds the fixed wipe capacity",
            ));
        }
        let acting_seat = game.current_player();
        let staged = game.preview_market_prelude(&action.prelude())?;
        let (draft_kind, tile_slot, wildlife_slot, same_slot_independent) = match action.draft {
            DraftChoice::Paired { slot } => (0, slot.index() as u8, slot.index() as u8, 0),
            DraftChoice::Independent {
                tile_slot,
                wildlife_slot,
            } => (
                1,
                tile_slot.index() as u8,
                wildlife_slot.index() as u8,
                u8::from(tile_slot == wildlife_slot),
            ),
        };
        let tile = staged.market().tiles[usize::from(tile_slot)].ok_or(
            DataError::InvalidConfig("graded-oracle tile slot is unavailable after its prelude"),
        )?;
        let wildlife = staged.market().wildlife[usize::from(wildlife_slot)].ok_or(
            DataError::InvalidConfig(
                "graded-oracle wildlife slot is unavailable after its prelude",
            ),
        )?;
        let mut wipe_masks = [0; GRADED_ORACLE_MAX_WILDLIFE_WIPES];
        for (index, wipe) in action.wildlife_wipes.iter().enumerate() {
            let mut mask = 0u8;
            for slot in &wipe.slots {
                mask |= 1 << slot.index();
            }
            if mask == 0 {
                return Err(DataError::InvalidConfig(
                    "graded-oracle action contains an empty wildlife wipe",
                ));
            }
            wipe_masks[index] = mask;
        }
        let before = score_board(&game.boards()[acting_seat], game.config().scoring_cards);
        let after_board = game.preview_active_board(action)?;
        let after = score_board(&after_board, game.config().scoring_cards);
        let staged_position = PositionRecord::observe(&staged, 0);
        Ok(Self {
            same_slot_independent,
            draft_kind,
            tile_slot,
            wildlife_slot,
            tile_id: tile.id.0,
            tile_terrain_a: tile.terrain_a as u8,
            tile_terrain_b: tile.terrain_b.map_or(NONE, |terrain| terrain as u8),
            tile_wildlife_mask: tile.wildlife.bits(),
            tile_keystone: u8::from(tile.keystone),
            drafted_wildlife: wildlife as u8,
            tile_q: action.tile.coord.q,
            tile_r: action.tile.coord.r,
            rotation: action.tile.rotation.get(),
            wildlife_present: u8::from(action.wildlife.is_some()),
            wildlife_q: action.wildlife.map_or(0, |coord| coord.q),
            wildlife_r: action.wildlife.map_or(0, |coord| coord.r),
            replace_three_of_a_kind: u8::from(action.replace_three_of_a_kind),
            wipe_count: u8::try_from(action.wildlife_wipes.len()).map_err(|_| {
                DataError::InvalidConfig("graded-oracle wipe count exceeds fixed-width storage")
            })?,
            wipe_masks,
            staged_active_nature_tokens: staged_position.nature_tokens[0],
            staged_market_entities: staged_position.market_entities,
            staged_public_supply: encode_public_supply(staged.public_supply()),
            immediate_score: after.base_total,
            immediate_deltas: score_deltas(before, after),
        })
    }

    /// Canonical lossless wire bytes used by both graded-oracle and R2-MAP.
    pub fn to_bytes(&self) -> [u8; GRADED_ORACLE_ACTION_FEATURE_SIZE] {
        let mut bytes = [0; GRADED_ORACLE_ACTION_FEATURE_SIZE];
        let mut offset = 0;
        self.write_to(&mut bytes, &mut offset);
        debug_assert_eq!(offset, GRADED_ORACLE_ACTION_FEATURE_SIZE);
        bytes
    }

    /// Decode the canonical lossless action wire representation.
    pub fn from_bytes(bytes: &[u8; GRADED_ORACLE_ACTION_FEATURE_SIZE]) -> Self {
        let mut offset = 0;
        let value = Self::read_from(bytes, &mut offset);
        debug_assert_eq!(offset, GRADED_ORACLE_ACTION_FEATURE_SIZE);
        value
    }

    pub fn to_game_action(&self, game: &GameState) -> Result<TurnAction, DataError> {
        let wipe_count = usize::from(self.wipe_count);
        if wipe_count > GRADED_ORACLE_MAX_WILDLIFE_WIPES
            || self.wipe_masks[..wipe_count].contains(&0)
            || self.wipe_masks[wipe_count..].iter().any(|mask| *mask != 0)
        {
            return Err(DataError::InvalidConfig(
                "graded-oracle wipe sequence is malformed",
            ));
        }
        let tile_slot = MarketSlot::new(self.tile_slot).ok_or(DataError::InvalidConfig(
            "graded-oracle action has an invalid tile slot",
        ))?;
        let wildlife_slot = MarketSlot::new(self.wildlife_slot).ok_or(DataError::InvalidConfig(
            "graded-oracle action has an invalid wildlife slot",
        ))?;
        let draft = match self.draft_kind {
            0 if tile_slot == wildlife_slot && self.same_slot_independent == 0 => {
                DraftChoice::Paired { slot: tile_slot }
            }
            1 => DraftChoice::Independent {
                tile_slot,
                wildlife_slot,
            },
            _ => {
                return Err(DataError::InvalidConfig(
                    "graded-oracle action has an invalid draft kind",
                ));
            }
        };
        if u8::from(matches!(
            draft,
            DraftChoice::Independent {
                tile_slot,
                wildlife_slot
            } if tile_slot == wildlife_slot
        )) != self.same_slot_independent
        {
            return Err(DataError::InvalidConfig(
                "graded-oracle same-slot independent flag is inconsistent",
            ));
        }
        let wildlife_wipes = self.wipe_masks[..wipe_count]
            .iter()
            .map(|mask| WildlifeWipe {
                slots: MarketSlot::ALL
                    .into_iter()
                    .filter(|slot| mask & (1 << slot.index()) != 0)
                    .collect(),
            })
            .collect::<Vec<_>>();
        let prelude = MarketPrelude {
            replace_three_of_a_kind: self.replace_three_of_a_kind != 0,
            wildlife_wipes,
        };
        let staged = game.preview_market_prelude(&prelude)?;
        let staged_position = PositionRecord::observe(&staged, 0);
        if staged_position.nature_tokens[0] != self.staged_active_nature_tokens
            || staged_position.market_entities != self.staged_market_entities
            || encode_public_supply(staged.public_supply()) != self.staged_public_supply
        {
            return Err(DataError::InvalidConfig(
                "graded-oracle staged public context does not match replay",
            ));
        }
        let tile = staged.market().tiles[tile_slot.index()].ok_or(DataError::InvalidConfig(
            "graded-oracle staged tile slot is unavailable",
        ))?;
        let wildlife = staged.market().wildlife[wildlife_slot.index()].ok_or(
            DataError::InvalidConfig("graded-oracle staged wildlife slot is unavailable"),
        )?;
        if tile.id.0 != self.tile_id
            || tile.terrain_a as u8 != self.tile_terrain_a
            || tile.terrain_b.map_or(NONE, |terrain| terrain as u8) != self.tile_terrain_b
            || tile.wildlife.bits() != self.tile_wildlife_mask
            || u8::from(tile.keystone) != self.tile_keystone
            || wildlife as u8 != self.drafted_wildlife
        {
            return Err(DataError::InvalidConfig(
                "graded-oracle staged draft metadata does not match replay",
            ));
        }
        let rotation = Rotation::new(self.rotation).ok_or(DataError::InvalidConfig(
            "graded-oracle action rotation is outside zero through five",
        ))?;
        let action = TurnAction {
            replace_three_of_a_kind: prelude.replace_three_of_a_kind,
            wildlife_wipes: prelude.wildlife_wipes,
            draft,
            tile: TilePlacement {
                coord: HexCoord::new(self.tile_q, self.tile_r),
                rotation,
            },
            wildlife: (self.wildlife_present != 0)
                .then_some(HexCoord::new(self.wildlife_q, self.wildlife_r)),
        };
        let before = score_board(
            &game.boards()[game.current_player()],
            game.config().scoring_cards,
        );
        let after_board = game.preview_active_board(&action)?;
        let after = score_board(&after_board, game.config().scoring_cards);
        if after.base_total != self.immediate_score
            || score_deltas(before, after) != self.immediate_deltas
        {
            return Err(DataError::InvalidConfig(
                "graded-oracle exact action deltas do not match replay",
            ));
        }
        game.transition(&action)?;
        Ok(action)
    }

    fn write_to(&self, bytes: &mut [u8], offset: &mut usize) {
        write_slice(
            bytes,
            offset,
            &[
                self.same_slot_independent,
                self.draft_kind,
                self.tile_slot,
                self.wildlife_slot,
                self.tile_id,
                self.tile_terrain_a,
                self.tile_terrain_b,
                self.tile_wildlife_mask,
                self.tile_keystone,
                self.drafted_wildlife,
                self.tile_q as u8,
                self.tile_r as u8,
                self.rotation,
                self.wildlife_present,
                self.wildlife_q as u8,
                self.wildlife_r as u8,
                self.replace_three_of_a_kind,
                self.wipe_count,
            ],
        );
        write_slice(bytes, offset, &self.wipe_masks);
        write_slice(bytes, offset, &[self.staged_active_nature_tokens]);
        *offset += 3;
        for entity in self.staged_market_entities {
            write_slice(bytes, offset, &entity);
        }
        write_slice(bytes, offset, &self.staged_public_supply);
        write_slice(bytes, offset, &self.immediate_score.to_le_bytes());
        for delta in self.immediate_deltas {
            write_slice(bytes, offset, &delta.to_le_bytes());
        }
    }

    fn read_from(bytes: &[u8], offset: &mut usize) -> Self {
        let [
            same_slot_independent,
            draft_kind,
            tile_slot,
            wildlife_slot,
            tile_id,
            tile_terrain_a,
            tile_terrain_b,
            tile_wildlife_mask,
            tile_keystone,
            drafted_wildlife,
            tile_q,
            tile_r,
            rotation,
            wildlife_present,
            wildlife_q,
            wildlife_r,
            replace_three_of_a_kind,
            wipe_count,
        ] = read_array(bytes, offset);
        let wipe_masks = read_array(bytes, offset);
        let staged_active_nature_tokens = read_array::<1>(bytes, offset)[0];
        *offset += 3;
        let mut staged_market_entities = [[NONE; 8]; 4];
        for entity in &mut staged_market_entities {
            *entity = read_array(bytes, offset);
        }
        let staged_public_supply = read_array(bytes, offset);
        let immediate_score = u16::from_le_bytes(read_array(bytes, offset));
        let mut immediate_deltas = [0; SCORE_COMPONENTS];
        for delta in &mut immediate_deltas {
            *delta = i16::from_le_bytes(read_array(bytes, offset));
        }
        Self {
            same_slot_independent,
            draft_kind,
            tile_slot,
            wildlife_slot,
            tile_id,
            tile_terrain_a,
            tile_terrain_b,
            tile_wildlife_mask,
            tile_keystone,
            drafted_wildlife,
            tile_q: tile_q as i8,
            tile_r: tile_r as i8,
            rotation,
            wildlife_present,
            wildlife_q: wildlife_q as i8,
            wildlife_r: wildlife_r as i8,
            replace_three_of_a_kind,
            wipe_count,
            wipe_masks,
            staged_active_nature_tokens,
            staged_market_entities,
            staged_public_supply,
            immediate_score,
            immediate_deltas,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GradedOracleCandidate {
    pub action_hash: [u8; 32],
    pub canonical_index: u16,
    pub screen_rank: u16,
    pub source_flags: u16,
    pub fidelity_mask: u16,
    pub model_immediate_score: f32,
    pub model_remaining_value: f32,
    pub screen_value: f32,
    pub uniform_market_survival_proxy: f32,
    pub visible_wildlife_count: u8,
    pub public_bag_wildlife_count: u8,
    pub action: GradedOracleActionFeatures,
    pub r600: GradedOracleEstimate,
    pub r1200: GradedOracleEstimate,
    pub r4800: GradedOracleEstimate,
}

impl GradedOracleCandidate {
    #[allow(clippy::too_many_arguments)]
    pub fn observe(
        game: &GameState,
        action: &TurnAction,
        action_hash: [u8; 32],
        canonical_index: u16,
        screen_rank: u16,
        source_flags: u16,
        model_immediate_score: f32,
        model_remaining_value: f32,
        screen_value: f32,
        uniform_market_survival_proxy: f32,
        visible_wildlife_count: u8,
        public_bag_wildlife_count: u8,
        r600: GradedOracleEstimate,
        r1200: GradedOracleEstimate,
        r4800: GradedOracleEstimate,
    ) -> Result<Self, DataError> {
        if source_flags & GRADED_SOURCE_COMPLETE_LEGAL == 0
            || screen_rank == 0
            || !model_immediate_score.is_finite()
            || !model_remaining_value.is_finite()
            || !screen_value.is_finite()
            || !uniform_market_survival_proxy.is_finite()
            || (model_immediate_score + model_remaining_value - screen_value).abs() > 1e-3
        {
            return Err(DataError::InvalidConfig(
                "graded-oracle candidate metadata is invalid",
            ));
        }
        r600.validate()?;
        r1200.validate()?;
        r4800.validate()?;
        let fidelity_mask = fidelity_mask(r600, r1200, r4800);
        if source_fidelity_mask(source_flags) != fidelity_mask {
            return Err(DataError::InvalidConfig(
                "graded-oracle source and fidelity masks disagree",
            ));
        }
        Ok(Self {
            action_hash,
            canonical_index,
            screen_rank,
            source_flags,
            fidelity_mask,
            model_immediate_score,
            model_remaining_value,
            screen_value,
            uniform_market_survival_proxy,
            visible_wildlife_count,
            public_bag_wildlife_count,
            action: GradedOracleActionFeatures::observe(game, action)?,
            r600,
            r1200,
            r4800,
        })
    }

    pub fn to_bytes(&self) -> [u8; GRADED_ORACLE_CANDIDATE_RECORD_SIZE] {
        let mut bytes = [0; GRADED_ORACLE_CANDIDATE_RECORD_SIZE];
        let mut offset = 0;
        write_slice(&mut bytes, &mut offset, &self.action_hash);
        write_slice(&mut bytes, &mut offset, &self.canonical_index.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.screen_rank.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.source_flags.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.fidelity_mask.to_le_bytes());
        write_slice(
            &mut bytes,
            &mut offset,
            &self.model_immediate_score.to_le_bytes(),
        );
        write_slice(
            &mut bytes,
            &mut offset,
            &self.model_remaining_value.to_le_bytes(),
        );
        write_slice(&mut bytes, &mut offset, &self.screen_value.to_le_bytes());
        write_slice(
            &mut bytes,
            &mut offset,
            &self.uniform_market_survival_proxy.to_le_bytes(),
        );
        write_slice(
            &mut bytes,
            &mut offset,
            &[self.visible_wildlife_count, self.public_bag_wildlife_count],
        );
        offset += 2;
        let action_start = offset;
        self.action.write_to(&mut bytes, &mut offset);
        debug_assert_eq!(offset - action_start, GRADED_ORACLE_ACTION_FEATURE_SIZE);
        self.r600.write_to(&mut bytes, &mut offset);
        self.r1200.write_to(&mut bytes, &mut offset);
        self.r4800.write_to(&mut bytes, &mut offset);
        debug_assert_eq!(offset, GRADED_ORACLE_CANDIDATE_RECORD_SIZE);
        bytes
    }

    pub fn from_bytes(bytes: &[u8; GRADED_ORACLE_CANDIDATE_RECORD_SIZE]) -> Self {
        let mut offset = 0;
        let action_hash = read_array(bytes, &mut offset);
        let canonical_index = u16::from_le_bytes(read_array(bytes, &mut offset));
        let screen_rank = u16::from_le_bytes(read_array(bytes, &mut offset));
        let source_flags = u16::from_le_bytes(read_array(bytes, &mut offset));
        let fidelity_mask = u16::from_le_bytes(read_array(bytes, &mut offset));
        let model_immediate_score = f32::from_le_bytes(read_array(bytes, &mut offset));
        let model_remaining_value = f32::from_le_bytes(read_array(bytes, &mut offset));
        let screen_value = f32::from_le_bytes(read_array(bytes, &mut offset));
        let uniform_market_survival_proxy = f32::from_le_bytes(read_array(bytes, &mut offset));
        let [visible_wildlife_count, public_bag_wildlife_count] = read_array(bytes, &mut offset);
        offset += 2;
        let action_start = offset;
        let action = GradedOracleActionFeatures::read_from(bytes, &mut offset);
        debug_assert_eq!(offset - action_start, GRADED_ORACLE_ACTION_FEATURE_SIZE);
        let r600 = GradedOracleEstimate::read_from(bytes, &mut offset);
        let r1200 = GradedOracleEstimate::read_from(bytes, &mut offset);
        let r4800 = GradedOracleEstimate::read_from(bytes, &mut offset);
        debug_assert_eq!(offset, GRADED_ORACLE_CANDIDATE_RECORD_SIZE);
        Self {
            action_hash,
            canonical_index,
            screen_rank,
            source_flags,
            fidelity_mask,
            model_immediate_score,
            model_remaining_value,
            screen_value,
            uniform_market_survival_proxy,
            visible_wildlife_count,
            public_bag_wildlife_count,
            action,
            r600,
            r1200,
            r4800,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GradedOracleGroup {
    pub group_id: u64,
    pub raw_seed: u64,
    pub completed_turns: u16,
    pub current_player: u8,
    pub personal_turn: u8,
    pub phase: u8,
    pub selected_index: u16,
    pub champion_index: u16,
    pub public_state_hash: [u8; 32],
    pub public_supply: PublicSupply,
    pub position: PositionRecord,
    pub candidates: Vec<GradedOracleCandidate>,
}

pub struct GradedOracleDatasetWriter {
    output: PathBuf,
    manifest_path: PathBuf,
    manifest: GradedOracleDatasetManifest,
}

impl GradedOracleDatasetWriter {
    pub fn open(config: &GradedOracleDatasetConfig) -> Result<Self, DataError> {
        config.validate()?;
        fs::create_dir_all(&config.output)?;
        let manifest_path = config.output.join("dataset.json");
        let manifest = if manifest_path.exists() {
            if !config.resume {
                return Err(DataError::DatasetExists(config.output.clone()));
            }
            let manifest: GradedOracleDatasetManifest =
                serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
            validate_resume(&manifest, config)?;
            validate_graded_oracle_dataset(&config.output, &manifest)?;
            manifest
        } else {
            let now = unix_seconds()?;
            GradedOracleDatasetManifest {
                schema_version: GRADED_ORACLE_DATASET_SCHEMA_VERSION,
                dataset_id: dataset_id(config),
                feature_schema: GRADED_ORACLE_FEATURE_SCHEMA.to_owned(),
                position_feature_schema: FEATURE_SCHEMA.to_owned(),
                target_schema: GRADED_ORACLE_TARGET_SCHEMA.to_owned(),
                group_header_size: GRADED_ORACLE_GROUP_HEADER_SIZE,
                candidate_record_size: GRADED_ORACLE_CANDIDATE_RECORD_SIZE,
                action_feature_size: GRADED_ORACLE_ACTION_FEATURE_SIZE,
                public_supply_size: super::PUBLIC_SUPPLY_SIZE,
                maximum_wildlife_wipes: GRADED_ORACLE_MAX_WILDLIFE_WIPES,
                game: GameConfig::research_aaaaa(4)?,
                split: config.split,
                seeds: config
                    .audit_inputs
                    .iter()
                    .map(|input| input.raw_seed)
                    .collect(),
                requested_games: config.audit_inputs.len(),
                completed_games: 0,
                total_groups: 0,
                total_records: 0,
                teacher: GradedOracleTeacherIdentity::from(&config.audit_inputs[0]),
                audit_inputs: config.audit_inputs.clone(),
                source_manifests: Vec::new(),
                created_unix_seconds: now,
                updated_unix_seconds: now,
                provenance: collection_provenance()?,
                shards: Vec::new(),
            }
        };
        Ok(Self {
            output: config.output.clone(),
            manifest_path,
            manifest,
        })
    }

    pub fn manifest(&self) -> &GradedOracleDatasetManifest {
        &self.manifest
    }

    pub fn root(&self) -> &Path {
        &self.output
    }

    pub fn append_game(
        &mut self,
        raw_seed: u64,
        groups: &[GradedOracleGroup],
    ) -> Result<(), DataError> {
        let expected_seed = self
            .manifest
            .audit_inputs
            .get(self.manifest.completed_games)
            .map(|input| input.raw_seed);
        if expected_seed != Some(raw_seed)
            || self.manifest.completed_games >= self.manifest.requested_games
            || groups.is_empty()
            || groups.iter().any(|group| group.raw_seed != raw_seed)
        {
            return Err(DataError::InvalidConfig(
                "graded-oracle game range is invalid",
            ));
        }
        let record_count = validate_groups(groups)?;
        let file_name = format!("seed-{raw_seed}.gov");
        let path = self.output.join(&file_name);
        if path.exists() {
            return Err(DataError::DatasetExists(path));
        }
        write_shard(&path, self.manifest.split, raw_seed, groups, record_count)?;
        let metadata = fs::metadata(&path)?;
        self.manifest.shards.push(RankingShardManifest {
            file: file_name,
            first_game_index: raw_seed,
            game_count: 1,
            group_count: groups.len(),
            record_count,
            byte_count: metadata.len(),
            blake3: checksum_file(&path)?,
        });
        self.manifest.completed_games += 1;
        self.manifest.total_groups += groups.len();
        self.manifest.total_records += record_count;
        self.manifest.updated_unix_seconds = unix_seconds()?;
        write_manifest_atomic(&self.manifest_path, &self.manifest)
    }
}

pub fn validate_graded_oracle_dataset(
    root: &Path,
    manifest: &GradedOracleDatasetManifest,
) -> Result<(), DataError> {
    if manifest.schema_version != GRADED_ORACLE_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != GRADED_ORACLE_FEATURE_SCHEMA
        || manifest.position_feature_schema != FEATURE_SCHEMA
        || manifest.target_schema != GRADED_ORACLE_TARGET_SCHEMA
        || manifest.group_header_size != GRADED_ORACLE_GROUP_HEADER_SIZE
        || manifest.candidate_record_size != GRADED_ORACLE_CANDIDATE_RECORD_SIZE
        || manifest.action_feature_size != GRADED_ORACLE_ACTION_FEATURE_SIZE
        || manifest.public_supply_size != super::PUBLIC_SUPPLY_SIZE
        || manifest.maximum_wildlife_wipes != GRADED_ORACLE_MAX_WILDLIFE_WIPES
        || manifest.game != GameConfig::research_aaaaa(4)?
        || manifest.requested_games != manifest.audit_inputs.len()
        || manifest.shards.len() != manifest.completed_games
        || manifest.seeds
            != manifest
                .audit_inputs
                .iter()
                .map(|input| input.raw_seed)
                .collect::<Vec<_>>()
        || manifest
            .audit_inputs
            .first()
            .map(GradedOracleTeacherIdentity::from)
            != Some(manifest.teacher.clone())
        || manifest
            .audit_inputs
            .iter()
            .any(|input| GradedOracleTeacherIdentity::from(input) != manifest.teacher)
        || manifest.source_manifests.iter().any(|source| {
            source.path.trim().is_empty()
                || source.dataset_id.trim().is_empty()
                || source.manifest_blake3.trim().is_empty()
                || source.seeds.is_empty()
        })
    {
        return Err(DataError::InvalidManifest(
            "graded-oracle schema identifiers do not match",
        ));
    }
    let mut games = 0;
    let mut groups = 0;
    let mut records = 0;
    for (index, shard) in manifest.shards.iter().enumerate() {
        let expected_seed =
            manifest
                .seeds
                .get(index)
                .copied()
                .ok_or(DataError::InvalidManifest(
                    "graded-oracle manifest has more shards than seeds",
                ))?;
        if shard.first_game_index != expected_seed || shard.game_count != 1 {
            return Err(DataError::InvalidManifest(
                "graded-oracle shard seed does not match its manifest position",
            ));
        }
        let path = root.join(&shard.file);
        if fs::metadata(&path)?.len() != shard.byte_count {
            return Err(DataError::InvalidManifest(
                "graded-oracle shard byte count mismatch",
            ));
        }
        if checksum_file(&path)? != shard.blake3 {
            return Err(DataError::ChecksumMismatch(path));
        }
        let shard_groups = read_graded_oracle_shard(root, manifest.split, shard)?;
        if shard_groups.len() != shard.group_count
            || shard_groups
                .iter()
                .map(|group| group.candidates.len())
                .sum::<usize>()
                != shard.record_count
            || shard_groups
                .iter()
                .any(|group| group.raw_seed != expected_seed)
        {
            return Err(DataError::InvalidManifest(
                "graded-oracle shard totals do not match",
            ));
        }
        games += shard.game_count;
        groups += shard.group_count;
        records += shard.record_count;
    }
    if games != manifest.completed_games
        || groups != manifest.total_groups
        || records != manifest.total_records
        || manifest.completed_games > manifest.requested_games
    {
        return Err(DataError::InvalidManifest(
            "graded-oracle manifest totals do not match shards",
        ));
    }
    Ok(())
}

pub fn merge_graded_oracle_datasets(
    inputs: &[PathBuf],
    output: &Path,
) -> Result<GradedOracleDatasetManifest, DataError> {
    if inputs.is_empty() {
        return Err(DataError::InvalidConfig(
            "graded-oracle merge requires source datasets",
        ));
    }
    if output.exists() {
        return Err(DataError::DatasetExists(output.to_path_buf()));
    }

    let mut sources = Vec::with_capacity(inputs.len());
    for root in inputs {
        let manifest_path = root.join("dataset.json");
        let manifest: GradedOracleDatasetManifest =
            serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
        validate_graded_oracle_dataset(root, &manifest)?;
        if manifest.completed_games != manifest.requested_games {
            return Err(DataError::InvalidManifest(
                "graded-oracle merge input is incomplete",
            ));
        }
        sources.push((root.clone(), checksum_file(&manifest_path)?, manifest));
    }
    sources.sort_by_key(|(_, _, manifest)| manifest.seeds[0]);
    let frozen = &sources[0].2;
    if sources.iter().any(|(_, _, manifest)| {
        manifest.schema_version != frozen.schema_version
            || manifest.feature_schema != frozen.feature_schema
            || manifest.position_feature_schema != frozen.position_feature_schema
            || manifest.target_schema != frozen.target_schema
            || manifest.group_header_size != frozen.group_header_size
            || manifest.candidate_record_size != frozen.candidate_record_size
            || manifest.action_feature_size != frozen.action_feature_size
            || manifest.public_supply_size != frozen.public_supply_size
            || manifest.maximum_wildlife_wipes != frozen.maximum_wildlife_wipes
            || manifest.game != frozen.game
            || manifest.split != frozen.split
            || manifest.teacher != frozen.teacher
    }) {
        return Err(DataError::InvalidManifest(
            "graded-oracle merge inputs have incompatible identities",
        ));
    }

    let mut entries = Vec::new();
    let mut source_manifests = Vec::with_capacity(sources.len());
    for (root, manifest_blake3, manifest) in &sources {
        source_manifests.push(GradedOracleSourceManifest {
            path: root.display().to_string(),
            dataset_id: manifest.dataset_id.clone(),
            manifest_blake3: manifest_blake3.clone(),
            seeds: manifest.seeds.clone(),
        });
        for ((seed, audit_input), shard) in manifest
            .seeds
            .iter()
            .zip(&manifest.audit_inputs)
            .zip(&manifest.shards)
        {
            if *seed != audit_input.raw_seed || *seed != shard.first_game_index {
                return Err(DataError::InvalidManifest(
                    "graded-oracle merge input ordering drifted",
                ));
            }
            entries.push((*seed, audit_input.clone(), shard.clone(), root.clone()));
        }
    }
    entries.sort_by_key(|(seed, _, _, _)| *seed);
    if entries.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
        return Err(DataError::InvalidManifest(
            "graded-oracle merge inputs overlap",
        ));
    }

    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let file_name = output
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("graded-oracle");
    let temp = parent.join(format!(
        ".{file_name}.merge-{}-{}.tmp",
        std::process::id(),
        unix_seconds()?
    ));
    if temp.exists() {
        fs::remove_dir_all(&temp)?;
    }
    fs::create_dir(&temp)?;

    let result = (|| {
        let now = unix_seconds()?;
        let audit_inputs = entries
            .iter()
            .map(|(_, input, _, _)| input.clone())
            .collect::<Vec<_>>();
        let shards = entries
            .iter()
            .map(|(_, _, shard, _)| shard.clone())
            .collect::<Vec<_>>();
        for (_, _, shard, root) in &entries {
            let destination = temp.join(&shard.file);
            if destination.exists() {
                return Err(DataError::InvalidManifest(
                    "graded-oracle merge shard name collides",
                ));
            }
            fs::copy(root.join(&shard.file), &destination)?;
            File::open(destination)?.sync_all()?;
        }
        let manifest = GradedOracleDatasetManifest {
            schema_version: frozen.schema_version,
            dataset_id: dataset_id_from_inputs(frozen.split, &audit_inputs),
            feature_schema: frozen.feature_schema.clone(),
            position_feature_schema: frozen.position_feature_schema.clone(),
            target_schema: frozen.target_schema.clone(),
            group_header_size: frozen.group_header_size,
            candidate_record_size: frozen.candidate_record_size,
            action_feature_size: frozen.action_feature_size,
            public_supply_size: frozen.public_supply_size,
            maximum_wildlife_wipes: frozen.maximum_wildlife_wipes,
            game: frozen.game,
            split: frozen.split,
            seeds: entries.iter().map(|(seed, _, _, _)| *seed).collect(),
            requested_games: entries.len(),
            completed_games: entries.len(),
            total_groups: shards.iter().map(|shard| shard.group_count).sum(),
            total_records: shards.iter().map(|shard| shard.record_count).sum(),
            teacher: frozen.teacher.clone(),
            audit_inputs,
            source_manifests,
            created_unix_seconds: now,
            updated_unix_seconds: now,
            provenance: collection_provenance()?,
            shards,
        };
        write_manifest_atomic(&temp.join("dataset.json"), &manifest)?;
        validate_graded_oracle_dataset(&temp, &manifest)?;
        fs::rename(&temp, output)?;
        Ok(manifest)
    })();
    if result.is_err() {
        fs::remove_dir_all(&temp).ok();
    }
    result
}

pub fn read_graded_oracle_shard(
    root: &Path,
    split: DatasetSplit,
    shard: &RankingShardManifest,
) -> Result<Vec<GradedOracleGroup>, DataError> {
    let path = root.join(&shard.file);
    validate_shard_header(&path, split, shard)?;
    let mut reader = BufReader::new(File::open(path)?);
    reader.seek(std::io::SeekFrom::Start(GRADED_ORACLE_HEADER_SIZE as u64))?;
    let mut groups = Vec::with_capacity(shard.group_count);
    for _ in 0..shard.group_count {
        let mut header = [0; GRADED_ORACLE_GROUP_HEADER_SIZE];
        reader.read_exact(&mut header)?;
        let mut offset = 0;
        let group_id = u64::from_le_bytes(read_array(&header, &mut offset));
        let raw_seed = u64::from_le_bytes(read_array(&header, &mut offset));
        let candidate_count = u16::from_le_bytes(read_array(&header, &mut offset));
        let selected_index = u16::from_le_bytes(read_array(&header, &mut offset));
        let champion_index = u16::from_le_bytes(read_array(&header, &mut offset));
        let completed_turns = u16::from_le_bytes(read_array(&header, &mut offset));
        let [current_player, personal_turn, phase] = read_array(&header, &mut offset);
        offset += 1;
        let public_state_hash = read_array(&header, &mut offset);
        let position_bytes: [u8; RECORD_SIZE] = read_array(&header, &mut offset);
        let public_supply = decode_public_supply(read_array(&header, &mut offset));
        offset += 6;
        debug_assert_eq!(offset, GRADED_ORACLE_GROUP_HEADER_SIZE);
        let mut candidates = Vec::with_capacity(usize::from(candidate_count));
        for _ in 0..candidate_count {
            let mut bytes = [0; GRADED_ORACLE_CANDIDATE_RECORD_SIZE];
            reader.read_exact(&mut bytes)?;
            candidates.push(GradedOracleCandidate::from_bytes(&bytes));
        }
        groups.push(GradedOracleGroup {
            group_id,
            raw_seed,
            completed_turns,
            current_player,
            personal_turn,
            phase,
            selected_index,
            champion_index,
            public_state_hash,
            public_supply,
            position: PositionRecord::from_bytes(&position_bytes),
            candidates,
        });
    }
    validate_groups(&groups)?;
    Ok(groups)
}

fn validate_groups(groups: &[GradedOracleGroup]) -> Result<usize, DataError> {
    let mut group_ids = BTreeSet::new();
    let mut turns = BTreeSet::new();
    let mut record_count = 0;
    for group in groups {
        let count = group.candidates.len();
        if count < 2
            || count > usize::from(u16::MAX)
            || usize::from(group.selected_index) >= count
            || usize::from(group.champion_index) >= count
            || group.current_player >= 4
            || !(1..=20).contains(&group.personal_turn)
            || group.phase > 2
            || group.position.game_index != group.raw_seed
            || u16::from(group.position.turn) != group.completed_turns
            || group.position.active_seat != group.current_player
            || !group_ids.insert(group.group_id)
            || !turns.insert(group.completed_turns)
        {
            return Err(DataError::InvalidConfig(
                "graded-oracle candidate group is inconsistent",
            ));
        }
        let mut hashes = BTreeSet::new();
        let mut canonical_indices = BTreeSet::new();
        let mut screen_ranks = BTreeSet::new();
        for candidate in &group.candidates {
            candidate.r600.validate()?;
            candidate.r1200.validate()?;
            candidate.r4800.validate()?;
            if candidate.source_flags & GRADED_SOURCE_COMPLETE_LEGAL == 0
                || !candidate.model_immediate_score.is_finite()
                || !candidate.model_remaining_value.is_finite()
                || !candidate.screen_value.is_finite()
                || !candidate.uniform_market_survival_proxy.is_finite()
                || (candidate.model_immediate_score + candidate.model_remaining_value
                    - candidate.screen_value)
                    .abs()
                    > 1e-3
                || candidate.screen_rank == 0
                || candidate.fidelity_mask
                    != fidelity_mask(candidate.r600, candidate.r1200, candidate.r4800)
                || source_fidelity_mask(candidate.source_flags) != candidate.fidelity_mask
                || !hashes.insert(candidate.action_hash)
                || !canonical_indices.insert(candidate.canonical_index)
                || !screen_ranks.insert(candidate.screen_rank)
                || usize::from(candidate.action.wipe_count) > GRADED_ORACLE_MAX_WILDLIFE_WIPES
                || candidate.action.staged_public_supply
                    != encode_public_supply(decode_public_supply(
                        candidate.action.staged_public_supply,
                    ))
            {
                return Err(DataError::InvalidConfig(
                    "graded-oracle candidate metadata is inconsistent",
                ));
            }
        }
        if canonical_indices != (0..count as u16).collect()
            || screen_ranks != (1..=count as u16).collect()
            || group.candidates[usize::from(group.champion_index)].source_flags
                & GRADED_SOURCE_CHAMPION_SELECTED
                == 0
        {
            return Err(DataError::InvalidConfig(
                "graded-oracle candidate coverage is incomplete",
            ));
        }
        let selected = &group.candidates[usize::from(group.selected_index)];
        let best_r4800 = group
            .candidates
            .iter()
            .filter(|candidate| candidate.r4800.samples > 0)
            .map(|candidate| candidate.r4800.mean)
            .fold(f32::NEG_INFINITY, f32::max);
        let stable_winner = group
            .candidates
            .iter()
            .filter(|candidate| {
                candidate.r4800.samples > 0
                    && (candidate.r4800.mean - best_r4800).abs() <= f32::EPSILON
            })
            .min_by_key(|candidate| candidate.action_hash)
            .map(|candidate| candidate.action_hash);
        if selected.r4800.samples == 0
            || (selected.r4800.mean - best_r4800).abs() > f32::EPSILON
            || stable_winner != Some(selected.action_hash)
        {
            return Err(DataError::InvalidConfig(
                "graded-oracle selected action is not an R4800 maximum",
            ));
        }
        record_count += count;
    }
    Ok(record_count)
}

fn write_shard(
    path: &Path,
    split: DatasetSplit,
    raw_seed: u64,
    groups: &[GradedOracleGroup],
    record_count: usize,
) -> Result<(), DataError> {
    let temp_path = path.with_extension("gov.tmp");
    let mut writer = BufWriter::new(File::create(&temp_path)?);
    writer.write_all(GRADED_ORACLE_SHARD_MAGIC)?;
    writer.write_all(&GRADED_ORACLE_DATASET_SCHEMA_VERSION.to_le_bytes())?;
    writer.write_all(&(GRADED_ORACLE_HEADER_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(GRADED_ORACLE_GROUP_HEADER_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(GRADED_ORACLE_CANDIDATE_RECORD_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(record_count as u32).to_le_bytes())?;
    writer.write_all(&(groups.len() as u32).to_le_bytes())?;
    writer.write_all(&1u32.to_le_bytes())?;
    writer.write_all(&[split.code(), 4, 0, 0])?;
    writer.write_all(&raw_seed.to_le_bytes())?;
    writer.write_all(blake3::hash(GRADED_ORACLE_FEATURE_SCHEMA.as_bytes()).as_bytes())?;
    writer.write_all(blake3::hash(GRADED_ORACLE_TARGET_SCHEMA.as_bytes()).as_bytes())?;
    writer.write_all(&[0; 8])?;
    for group in groups {
        writer.write_all(&group.group_id.to_le_bytes())?;
        writer.write_all(&group.raw_seed.to_le_bytes())?;
        writer.write_all(&(group.candidates.len() as u16).to_le_bytes())?;
        writer.write_all(&group.selected_index.to_le_bytes())?;
        writer.write_all(&group.champion_index.to_le_bytes())?;
        writer.write_all(&group.completed_turns.to_le_bytes())?;
        writer.write_all(&[group.current_player, group.personal_turn, group.phase, 0])?;
        writer.write_all(&group.public_state_hash)?;
        writer.write_all(&group.position.to_bytes())?;
        writer.write_all(&encode_public_supply(group.public_supply))?;
        writer.write_all(&[0; 6])?;
        for candidate in &group.candidates {
            writer.write_all(&candidate.to_bytes())?;
        }
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(temp_path, path)?;
    Ok(())
}

fn validate_shard_header(
    path: &Path,
    split: DatasetSplit,
    shard: &RankingShardManifest,
) -> Result<(), DataError> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut header = [0; GRADED_ORACLE_HEADER_SIZE];
    reader.read_exact(&mut header)?;
    if &header[..8] != GRADED_ORACLE_SHARD_MAGIC
        || u16::from_le_bytes([header[8], header[9]]) != GRADED_ORACLE_DATASET_SCHEMA_VERSION
        || u16::from_le_bytes([header[10], header[11]]) as usize != GRADED_ORACLE_HEADER_SIZE
        || u16::from_le_bytes([header[12], header[13]]) as usize != GRADED_ORACLE_GROUP_HEADER_SIZE
        || u16::from_le_bytes([header[14], header[15]]) as usize
            != GRADED_ORACLE_CANDIDATE_RECORD_SIZE
        || u32::from_le_bytes(header[16..20].try_into().expect("fixed header")) as usize
            != shard.record_count
        || u32::from_le_bytes(header[20..24].try_into().expect("fixed header")) as usize
            != shard.group_count
        || u32::from_le_bytes(header[24..28].try_into().expect("fixed header")) != 1
        || header[28] != split.code()
        || u64::from_le_bytes(header[32..40].try_into().expect("fixed header"))
            != shard.first_game_index
        || &header[40..72] != blake3::hash(GRADED_ORACLE_FEATURE_SCHEMA.as_bytes()).as_bytes()
        || &header[72..104] != blake3::hash(GRADED_ORACLE_TARGET_SCHEMA.as_bytes()).as_bytes()
    {
        return Err(DataError::InvalidShard("incompatible graded-oracle header"));
    }
    let expected = GRADED_ORACLE_HEADER_SIZE as u64
        + shard.group_count as u64 * GRADED_ORACLE_GROUP_HEADER_SIZE as u64
        + shard.record_count as u64 * GRADED_ORACLE_CANDIDATE_RECORD_SIZE as u64;
    if fs::metadata(path)?.len() != expected {
        return Err(DataError::InvalidShard(
            "graded-oracle shard size does not match records",
        ));
    }
    Ok(())
}

fn validate_resume(
    manifest: &GradedOracleDatasetManifest,
    config: &GradedOracleDatasetConfig,
) -> Result<(), DataError> {
    let current = collection_provenance()?;
    if manifest.schema_version != GRADED_ORACLE_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != GRADED_ORACLE_FEATURE_SCHEMA
        || manifest.position_feature_schema != FEATURE_SCHEMA
        || manifest.target_schema != GRADED_ORACLE_TARGET_SCHEMA
        || manifest.split != config.split
        || manifest.seeds
            != config
                .audit_inputs
                .iter()
                .map(|input| input.raw_seed)
                .collect::<Vec<_>>()
        || manifest.requested_games != config.audit_inputs.len()
        || manifest.audit_inputs != config.audit_inputs
        || !collection_provenance_matches(&manifest.provenance, &current)
    {
        return Err(DataError::ResumeMismatch);
    }
    Ok(())
}

fn dataset_id(config: &GradedOracleDatasetConfig) -> String {
    dataset_id_from_inputs(config.split, &config.audit_inputs)
}

fn dataset_id_from_inputs(split: DatasetSplit, audit_inputs: &[GradedOracleAuditInput]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"cascadia-v2-graded-oracle-dataset");
    hasher.update(split.id().as_bytes());
    for input in audit_inputs {
        hasher.update(&input.raw_seed.to_le_bytes());
        hasher.update(input.blake3.as_bytes());
    }
    let digest = hasher.finalize().to_hex().to_string();
    format!("graded-oracle-{}-{}", split.id(), &digest[..16])
}

fn fidelity_mask(
    r600: GradedOracleEstimate,
    r1200: GradedOracleEstimate,
    r4800: GradedOracleEstimate,
) -> u16 {
    (u16::from(r600.samples > 0) * GRADED_FIDELITY_R600)
        | (u16::from(r1200.samples > 0) * GRADED_FIDELITY_R1200)
        | (u16::from(r4800.samples > 0) * GRADED_FIDELITY_R4800)
}

fn source_fidelity_mask(source_flags: u16) -> u16 {
    (u16::from(source_flags & GRADED_SOURCE_R600 != 0) * GRADED_FIDELITY_R600)
        | (u16::from(source_flags & GRADED_SOURCE_R1200 != 0) * GRADED_FIDELITY_R1200)
        | (u16::from(source_flags & GRADED_SOURCE_R4800 != 0) * GRADED_FIDELITY_R4800)
}

fn score_deltas(before: ScoreBreakdown, after: ScoreBreakdown) -> [i16; SCORE_COMPONENTS] {
    let mut deltas = [0; SCORE_COMPONENTS];
    for index in 0..5 {
        deltas[index] = after.habitat[index] as i16 - before.habitat[index] as i16;
        deltas[5 + index] = after.wildlife[index] as i16 - before.wildlife[index] as i16;
    }
    deltas[10] = after.nature_tokens as i16 - before.nature_tokens as i16;
    deltas
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_game::GameSeed;

    #[test]
    fn candidate_round_trip_preserves_lossless_action_and_fidelity_tiers() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(61_000),
        )
        .unwrap();
        let prelude = MarketPrelude {
            replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
            wildlife_wipes: Vec::new(),
        };
        let action = game.legal_turn_actions(&prelude).unwrap().remove(0);
        let candidate = GradedOracleCandidate::observe(
            &game,
            &action,
            [7; 32],
            3,
            4,
            GRADED_SOURCE_COMPLETE_LEGAL
                | GRADED_SOURCE_TOP_SCREEN
                | GRADED_SOURCE_R1200
                | GRADED_SOURCE_R4800,
            5.0,
            86.5,
            91.5,
            0.25,
            2,
            20,
            GradedOracleEstimate::default(),
            GradedOracleEstimate {
                mean: 94.0,
                stddev: 2.0,
                samples: 1_200,
            },
            GradedOracleEstimate {
                mean: 95.0,
                stddev: 1.5,
                samples: 4_800,
            },
        )
        .unwrap();
        let decoded = GradedOracleCandidate::from_bytes(&candidate.to_bytes());
        assert_eq!(decoded, candidate);
        assert_eq!(decoded.action.to_game_action(&game).unwrap(), action);
    }

    #[test]
    fn ordered_wipe_masks_have_fixed_lossless_storage() {
        let mut action = GradedOracleActionFeatures {
            same_slot_independent: 1,
            draft_kind: 1,
            tile_slot: 0,
            wildlife_slot: 0,
            tile_id: 1,
            tile_terrain_a: 0,
            tile_terrain_b: NONE,
            tile_wildlife_mask: 1,
            tile_keystone: 1,
            drafted_wildlife: 0,
            tile_q: -2,
            tile_r: 3,
            rotation: 5,
            wildlife_present: 1,
            wildlife_q: -1,
            wildlife_r: 2,
            replace_three_of_a_kind: 1,
            wipe_count: 2,
            wipe_masks: [0; GRADED_ORACLE_MAX_WILDLIFE_WIPES],
            staged_active_nature_tokens: 3,
            staged_market_entities: [[NONE; 8]; 4],
            staged_public_supply: [0; super::super::PUBLIC_SUPPLY_SIZE],
            immediate_score: 91,
            immediate_deltas: [0; SCORE_COMPONENTS],
        };
        action.wipe_masks[0] = 0b0011;
        action.wipe_masks[1] = 0b1100;
        let mut bytes = [0; GRADED_ORACLE_ACTION_FEATURE_SIZE];
        let mut offset = 0;
        action.write_to(&mut bytes, &mut offset);
        assert_eq!(offset, GRADED_ORACLE_ACTION_FEATURE_SIZE);
        let mut offset = 0;
        assert_eq!(
            GradedOracleActionFeatures::read_from(&bytes, &mut offset),
            action
        );
    }
}
