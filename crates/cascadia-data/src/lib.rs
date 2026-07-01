//! Versioned, checksummed training data for Cascadia AI v2.

mod action_ranking;
mod conservative_advantage;
mod counterfactual_advantage;
mod counterfactual_value;
mod imitation;
mod imitation_parent_prior;
mod imitation_targets;
mod public_beam_value;
mod public_supply;
mod ranking;
mod rollout_value;
mod score_to_go;

use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use blake3::Hasher;
use cascadia_game::{
    Board, GameConfig, GameSeed, GameState, Market, MarketSlot, PublicGameState, ScoreBreakdown,
    ScoringVariant, Terrain, Wildlife,
};
use cascadia_provenance::source_provenance;
use cascadia_sim::{MatchConfig, SimulationError, StrategyKind, play_match_observed};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use action_ranking::{
    ACTION_FEATURE_SCHEMA, ACTION_FEATURE_SIZE, ACTION_POSITION_RECORD_SIZE,
    ACTION_RANKING_DATASET_SCHEMA_VERSION, ACTION_RANKING_HEADER_SIZE, ACTION_RANKING_RECORD_SIZE,
    ACTION_RANKING_SHARD_MAGIC, ACTION_RANKING_TARGET_SCHEMA, ActionFeatures, ActionPositionRecord,
    ActionRankingDatasetConfig, ActionRankingDatasetManifest, ActionRankingDatasetWriter,
    ActionRankingRecord, ActionRankingSourceManifest, validate_action_ranking_dataset,
};
pub use conservative_advantage::{
    CONSERVATIVE_ADVANTAGE_DATASET_SCHEMA_VERSION, CONSERVATIVE_ADVANTAGE_FEATURE_SCHEMA,
    CONSERVATIVE_ADVANTAGE_HEADER_SIZE, CONSERVATIVE_ADVANTAGE_RECORD_SIZE,
    CONSERVATIVE_ADVANTAGE_SHARD_MAGIC, CONSERVATIVE_ADVANTAGE_TARGET_SCHEMA,
    ConservativeAdvantageDatasetConfig, ConservativeAdvantageDatasetManifest,
    ConservativeAdvantageDatasetWriter, ConservativeAdvantageRecord,
    ConservativeAdvantageTeacherConfig, read_conservative_advantage_shard_records,
    validate_conservative_advantage_dataset,
};
pub use counterfactual_advantage::{
    COUNTERFACTUAL_ADVANTAGE_CANDIDATE_SIZE, COUNTERFACTUAL_ADVANTAGE_DATASET_SCHEMA_VERSION,
    COUNTERFACTUAL_ADVANTAGE_FEATURE_SCHEMA, COUNTERFACTUAL_ADVANTAGE_HEADER_SIZE,
    COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES, COUNTERFACTUAL_ADVANTAGE_MAX_SAMPLES,
    COUNTERFACTUAL_ADVANTAGE_NEAREST_SELECTION, COUNTERFACTUAL_ADVANTAGE_RECORD_SIZE,
    COUNTERFACTUAL_ADVANTAGE_SHARD_MAGIC, COUNTERFACTUAL_ADVANTAGE_STABILIZATION_CONDITIONING,
    COUNTERFACTUAL_ADVANTAGE_STRATIFIED_SELECTION, COUNTERFACTUAL_ADVANTAGE_TARGET_SCHEMA,
    CounterfactualAdvantageCandidate, CounterfactualAdvantageDatasetConfig,
    CounterfactualAdvantageDatasetManifest, CounterfactualAdvantageDatasetWriter,
    CounterfactualAdvantageRecord, CounterfactualAdvantageTeacherConfig,
    read_counterfactual_advantage_shard_records, validate_counterfactual_advantage_dataset,
};
pub use counterfactual_value::{
    COUNTERFACTUAL_VALUE_DATASET_SCHEMA_VERSION, COUNTERFACTUAL_VALUE_FEATURE_SCHEMA,
    COUNTERFACTUAL_VALUE_HEADER_SIZE, COUNTERFACTUAL_VALUE_MAX_SAMPLES,
    COUNTERFACTUAL_VALUE_RECORD_SIZE, COUNTERFACTUAL_VALUE_SHARD_MAGIC,
    COUNTERFACTUAL_VALUE_TARGET_SCHEMA, CounterfactualValueDatasetConfig,
    CounterfactualValueDatasetManifest, CounterfactualValueDatasetWriter,
    CounterfactualValueRecord, CounterfactualValueTeacherConfig,
    read_counterfactual_value_shard_records, validate_counterfactual_value_dataset,
};
pub use imitation::{
    IMITATION_CANDIDATE_RECORD_SIZE, IMITATION_DATASET_SCHEMA_VERSION, IMITATION_FEATURE_SCHEMA,
    IMITATION_GROUP_HEADER_SIZE, IMITATION_HEADER_SIZE, IMITATION_SHARD_MAGIC,
    IMITATION_TARGET_SCHEMA, ImitationCandidateConfig, ImitationDatasetConfig,
    ImitationDatasetManifest, ImitationDatasetWriter, ImitationRecord, ImitationTeacherConfig,
    PROPOSAL_ACTION_FEATURE_SIZE, ProposalActionFeatures, ProposalPositionRecord,
    read_imitation_shard_records, validate_imitation_dataset,
};
pub use imitation_parent_prior::{
    IMITATION_PARENT_HIDDEN_DATASET_SCHEMA_VERSION, IMITATION_PARENT_HIDDEN_DIM,
    IMITATION_PARENT_HIDDEN_FEATURE_SCHEMA, IMITATION_PARENT_HIDDEN_HEADER_SIZE,
    IMITATION_PARENT_HIDDEN_RECORD_SIZE, IMITATION_PARENT_HIDDEN_SHARD_MAGIC,
    IMITATION_PARENT_HIDDEN_TARGET_SCHEMA, IMITATION_PARENT_PRIOR_DATASET_SCHEMA_VERSION,
    IMITATION_PARENT_PRIOR_FEATURE_SCHEMA, IMITATION_PARENT_PRIOR_HEADER_SIZE,
    IMITATION_PARENT_PRIOR_RECORD_SIZE, IMITATION_PARENT_PRIOR_SHARD_MAGIC,
    IMITATION_PARENT_PRIOR_TARGET_SCHEMA, ImitationParentHiddenDatasetConfig,
    ImitationParentHiddenDatasetManifest, ImitationParentHiddenDatasetWriter,
    ImitationParentHiddenModelManifest, ImitationParentHiddenRecord,
    ImitationParentHiddenSourceManifest, ImitationParentPriorDatasetConfig,
    ImitationParentPriorDatasetManifest, ImitationParentPriorDatasetWriter,
    ImitationParentPriorModelManifest, ImitationParentPriorRecord,
    ImitationParentPriorSourceManifest, read_imitation_parent_hidden_shard_records,
    read_imitation_parent_prior_shard_records, validate_imitation_parent_hidden_dataset,
    validate_imitation_parent_prior_dataset,
};
pub use imitation_targets::{
    IMITATION_TARGETS_DATASET_SCHEMA_VERSION, IMITATION_TARGETS_FEATURE_SCHEMA,
    IMITATION_TARGETS_HEADER_SIZE, IMITATION_TARGETS_RECORD_SIZE, IMITATION_TARGETS_SHARD_MAGIC,
    IMITATION_TARGETS_TARGET_SCHEMA, ImitationTargetRecord, ImitationTargetsDatasetConfig,
    ImitationTargetsDatasetManifest, ImitationTargetsDatasetWriter, ImitationTargetsSourceManifest,
    SOURCE_DETERMINISTIC_NEGATIVE, SOURCE_IMMEDIATE_TOP, SOURCE_PATTERN_FRONTIER,
    SOURCE_TEACHER_FRONTIER, read_imitation_target_shard_records,
    validate_imitation_targets_dataset,
};
pub use public_beam_value::{
    PUBLIC_BEAM_VALUE_DATASET_SCHEMA_VERSION, PUBLIC_BEAM_VALUE_FEATURE_SCHEMA,
    PUBLIC_BEAM_VALUE_HEADER_SIZE, PUBLIC_BEAM_VALUE_RECORD_SIZE, PUBLIC_BEAM_VALUE_SHARD_MAGIC,
    PUBLIC_BEAM_VALUE_TARGET_SCHEMA, PublicBeamValueDatasetConfig, PublicBeamValueDatasetManifest,
    PublicBeamValueDatasetWriter, PublicBeamValueRecord, PublicBeamValueTeacherConfig,
    read_public_beam_value_shard_records, validate_public_beam_value_dataset,
};
pub use public_supply::PUBLIC_SUPPLY_SIZE;
pub use ranking::{
    RANKING_DATASET_SCHEMA_VERSION, RANKING_HEADER_SIZE, RANKING_RECORD_SIZE, RANKING_SHARD_MAGIC,
    RANKING_TARGET_SCHEMA, RankingCandidateFamily, RankingDatasetConfig, RankingDatasetManifest,
    RankingDatasetWriter, RankingRecord, RankingShardManifest, RankingTeacherConfig,
    RankingTrajectoryConfig, read_ranking_shard_records, validate_ranking_dataset,
};
pub use rollout_value::{
    ROLLOUT_VALUE_DATASET_SCHEMA_VERSION, ROLLOUT_VALUE_FEATURE_SCHEMA, ROLLOUT_VALUE_HEADER_SIZE,
    ROLLOUT_VALUE_RECORD_PREFIX_SIZE, ROLLOUT_VALUE_SHARD_MAGIC, ROLLOUT_VALUE_TARGET_SCHEMA,
    RolloutValueDatasetConfig, RolloutValueDatasetManifest, RolloutValueDatasetWriter,
    RolloutValueRecord, RolloutValueRecordKind, RolloutValueTeacherConfig,
    read_rollout_value_shard_records, validate_rollout_value_dataset,
};
pub use score_to_go::{
    SCORE_TO_GO_DATASET_SCHEMA_VERSION, SCORE_TO_GO_FEATURE_SCHEMA, SCORE_TO_GO_HEADER_SIZE,
    SCORE_TO_GO_RECORD_SIZE, SCORE_TO_GO_SHARD_MAGIC, SCORE_TO_GO_TARGET_SCHEMA,
    ScoreToGoDatasetConfig, ScoreToGoDatasetManifest, ScoreToGoDatasetWriter, ScoreToGoRecord,
    ScoreToGoTeacherConfig, validate_score_to_go_dataset,
};

pub const DATASET_SCHEMA_VERSION: u16 = 1;
pub const FEATURE_SCHEMA: &str = "compact-entity-v2";
pub const TARGET_SCHEMA: &str = "base-score-components-v1";
pub const SHARD_MAGIC: &[u8; 8] = b"CSD2REC\0";
pub const SHARD_HEADER_SIZE: usize = 80;
pub const RECORD_SIZE: usize = 864;
pub const BOARD_SLOTS: usize = 4;
pub const MAX_BOARD_TILES: usize = 23;
pub const BOARD_ENTITY_SIZE: usize = 8;
pub const MARKET_ENTITY_SIZE: usize = 8;
pub const TARGET_DIM: usize = 11;
const NONE: u8 = u8::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DatasetSplit {
    Train,
    Validation,
    Test,
    Final,
}

impl DatasetSplit {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "validation",
            Self::Test => "test",
            Self::Final => "final",
        }
    }

    const fn code(self) -> u8 {
        match self {
            Self::Train => 0,
            Self::Validation => 1,
            Self::Test => 2,
            Self::Final => 3,
        }
    }

    pub fn game_seed(self, game_index: u64) -> GameSeed {
        let mut hasher = Hasher::new();
        hasher.update(b"cascadia-v2-dataset-seed");
        hasher.update(self.id().as_bytes());
        hasher.update(&game_index.to_le_bytes());
        GameSeed(*hasher.finalize().as_bytes())
    }
}

#[derive(Debug, Clone)]
pub struct CollectConfig {
    pub output: PathBuf,
    pub split: DatasetSplit,
    pub first_game_index: u64,
    pub games: usize,
    pub shard_games: usize,
    pub strategy: StrategyKind,
    pub resume: bool,
}

impl CollectConfig {
    fn validate(&self) -> Result<(), DataError> {
        if self.games == 0 {
            return Err(DataError::InvalidConfig("games must be positive"));
        }
        if self.shard_games == 0 {
            return Err(DataError::InvalidConfig("shard_games must be positive"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct DatasetWriterConfig {
    pub output: PathBuf,
    pub split: DatasetSplit,
    pub first_game_index: u64,
    pub games: usize,
    pub strategy_id: String,
    pub resume: bool,
}

impl DatasetWriterConfig {
    fn validate(&self) -> Result<(), DataError> {
        if self.games == 0 {
            return Err(DataError::InvalidConfig("games must be positive"));
        }
        if self.strategy_id.trim().is_empty() {
            return Err(DataError::InvalidConfig("strategy ID cannot be empty"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetManifest {
    pub schema_version: u16,
    pub dataset_id: String,
    pub feature_schema: String,
    pub target_schema: String,
    pub record_size: usize,
    pub game: GameConfig,
    pub split: DatasetSplit,
    pub strategy: String,
    pub first_game_index: u64,
    pub requested_games: usize,
    pub completed_games: usize,
    pub total_records: usize,
    pub created_unix_seconds: u64,
    pub updated_unix_seconds: u64,
    pub provenance: CollectionProvenance,
    pub shards: Vec<ShardManifest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectionProvenance {
    pub collector_version: String,
    pub git_revision: String,
    pub git_dirty: bool,
    #[serde(default)]
    pub git_status_blake3: String,
    #[serde(default)]
    pub v2_source_blake3: String,
    pub executable_blake3: String,
    pub hardware: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardManifest {
    pub file: String,
    pub first_game_index: u64,
    pub game_count: usize,
    pub record_count: usize,
    pub byte_count: u64,
    pub blake3: String,
}

pub struct DatasetWriter {
    output: PathBuf,
    manifest_path: PathBuf,
    manifest: DatasetManifest,
    split: DatasetSplit,
    strategy_code: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PositionRecord {
    pub game_index: u64,
    pub turn: u8,
    pub active_seat: u8,
    pub player_count: u8,
    pub total_turns: u8,
    pub board_counts: [u8; BOARD_SLOTS],
    pub nature_tokens: [u8; BOARD_SLOTS],
    pub scoring_cards: [u8; 5],
    pub habitat_bonuses: bool,
    pub wildlife_counts: [[u8; 5]; BOARD_SLOTS],
    pub habitat_sizes: [[u8; 5]; BOARD_SLOTS],
    pub board_entities: [[[u8; BOARD_ENTITY_SIZE]; MAX_BOARD_TILES]; BOARD_SLOTS],
    pub market_entities: [[u8; MARKET_ENTITY_SIZE]; 4],
    pub targets: [u16; TARGET_DIM],
}

impl PositionRecord {
    pub fn observe(game: &GameState, game_index: u64) -> Self {
        Self::observe_for_seat(game, game_index, game.current_player())
    }

    pub fn observe_for_seat(game: &GameState, game_index: u64, perspective_seat: usize) -> Self {
        Self::observe_components(
            game.config(),
            game.completed_turns(),
            game.boards(),
            game.market(),
            game_index,
            perspective_seat,
        )
    }

    pub fn observe_public_for_seat(
        game: &PublicGameState,
        game_index: u64,
        perspective_seat: usize,
    ) -> Self {
        Self::observe_components(
            game.config(),
            game.completed_turns(),
            game.boards(),
            game.market(),
            game_index,
            perspective_seat,
        )
    }

    fn observe_components(
        config: GameConfig,
        completed_turns: u16,
        boards: &[Board],
        market: &Market,
        game_index: u64,
        perspective_seat: usize,
    ) -> Self {
        assert!(
            perspective_seat < usize::from(config.player_count),
            "perspective seat must exist in the game"
        );
        let mut record = Self {
            game_index,
            turn: completed_turns as u8,
            active_seat: perspective_seat as u8,
            player_count: config.player_count,
            total_turns: (20 * u16::from(config.player_count)) as u8,
            board_counts: [0; BOARD_SLOTS],
            nature_tokens: [0; BOARD_SLOTS],
            scoring_cards: scoring_card_codes(config),
            habitat_bonuses: config.habitat_bonuses,
            wildlife_counts: [[0; 5]; BOARD_SLOTS],
            habitat_sizes: [[0; 5]; BOARD_SLOTS],
            board_entities: [[[NONE; BOARD_ENTITY_SIZE]; MAX_BOARD_TILES]; BOARD_SLOTS],
            market_entities: [[NONE; MARKET_ENTITY_SIZE]; 4],
            targets: [0; TARGET_DIM],
        };

        for relative in 0..usize::from(record.player_count) {
            let absolute = (perspective_seat + relative) % usize::from(record.player_count);
            record.write_board(relative, &boards[absolute]);
        }

        record.write_market(market);
        record
    }

    fn write_market(&mut self, market: &Market) {
        for slot in MarketSlot::ALL {
            let tile = market.tiles[slot.index()];
            let wildlife = market.wildlife[slot.index()];
            if tile.is_some() || wildlife.is_some() {
                self.market_entities[slot.index()] = [
                    tile.map_or(NONE, |value| value.terrain_a as u8),
                    tile.and_then(|value| value.terrain_b)
                        .map_or(NONE, |terrain| terrain as u8),
                    tile.map_or(0, |value| value.wildlife.bits()),
                    wildlife.map_or(NONE, |value| value as u8),
                    tile.map_or(0, |value| u8::from(value.keystone)),
                    0,
                    0,
                    0,
                ];
            }
        }
    }

    pub fn observable_afterstate(
        game: &GameState,
        action: &cascadia_game::TurnAction,
        game_index: u64,
    ) -> Result<Self, cascadia_game::RuleError> {
        let acting_seat = game.current_player();
        let afterstate = game.preview_public_afterstate(action)?;
        Ok(Self::observe_public_for_seat(
            &afterstate,
            game_index,
            acting_seat,
        ))
    }

    pub fn set_target(&mut self, score: ScoreBreakdown) {
        self.targets[..5].copy_from_slice(&score.habitat);
        self.targets[5..10].copy_from_slice(&score.wildlife);
        self.targets[10] = score.nature_tokens;
    }

    pub fn to_bytes(&self) -> [u8; RECORD_SIZE] {
        let mut bytes = [0u8; RECORD_SIZE];
        let mut offset = 0;
        write_slice(&mut bytes, &mut offset, &self.game_index.to_le_bytes());
        write_slice(
            &mut bytes,
            &mut offset,
            &[
                self.turn,
                self.active_seat,
                self.player_count,
                self.total_turns,
            ],
        );
        write_slice(&mut bytes, &mut offset, &self.board_counts);
        write_slice(&mut bytes, &mut offset, &self.nature_tokens);
        write_slice(&mut bytes, &mut offset, &self.scoring_cards);
        write_slice(&mut bytes, &mut offset, &[u8::from(self.habitat_bonuses)]);
        offset += 6;
        for counts in self.wildlife_counts {
            write_slice(&mut bytes, &mut offset, &counts);
        }
        for sizes in self.habitat_sizes {
            write_slice(&mut bytes, &mut offset, &sizes);
        }
        for board in self.board_entities {
            for entity in board {
                write_slice(&mut bytes, &mut offset, &entity);
            }
        }
        for entity in self.market_entities {
            write_slice(&mut bytes, &mut offset, &entity);
        }
        for target in self.targets {
            write_slice(&mut bytes, &mut offset, &target.to_le_bytes());
        }
        offset += 2;
        debug_assert_eq!(offset, RECORD_SIZE);
        bytes
    }

    fn write_board(&mut self, relative: usize, board: &Board) {
        self.board_counts[relative] = board.tile_count() as u8;
        self.nature_tokens[relative] = board.nature_tokens();
        self.wildlife_counts[relative] = [0; 5];
        self.habitat_sizes[relative] = [0; 5];
        self.board_entities[relative] = [[NONE; BOARD_ENTITY_SIZE]; MAX_BOARD_TILES];
        for wildlife in Wildlife::ALL {
            self.wildlife_counts[relative][wildlife as usize] =
                board.wildlife_positions(wildlife).len() as u8;
        }
        for terrain in Terrain::ALL {
            self.habitat_sizes[relative][terrain as usize] = board.largest_habitat(terrain);
        }
        let mut tiles: Vec<_> = board.placed_tiles().collect();
        tiles.sort_unstable_by_key(|(coord, _)| (coord.q, coord.r));
        for (index, (coord, placed)) in tiles.into_iter().enumerate() {
            self.board_entities[relative][index] = [
                coord.q as u8,
                coord.r as u8,
                placed.tile.terrain_a as u8,
                placed.tile.terrain_b.map_or(NONE, |terrain| terrain as u8),
                placed.rotation.get(),
                placed.tile.wildlife.bits(),
                placed.wildlife.map_or(NONE, |wildlife| wildlife as u8),
                u8::from(placed.tile.keystone),
            ];
        }
    }

    pub(crate) fn from_bytes(bytes: &[u8; RECORD_SIZE]) -> Self {
        let mut offset = 0;
        let game_index = u64::from_le_bytes(read_array(bytes, &mut offset));
        let [turn, active_seat, player_count, total_turns] = read_array(bytes, &mut offset);
        let board_counts = read_array(bytes, &mut offset);
        let nature_tokens = read_array(bytes, &mut offset);
        let scoring_cards = read_array(bytes, &mut offset);
        let habitat_bonuses = read_array::<1>(bytes, &mut offset)[0] != 0;
        offset += 6;
        let mut wildlife_counts = [[0; 5]; BOARD_SLOTS];
        for counts in &mut wildlife_counts {
            *counts = read_array(bytes, &mut offset);
        }
        let mut habitat_sizes = [[0; 5]; BOARD_SLOTS];
        for sizes in &mut habitat_sizes {
            *sizes = read_array(bytes, &mut offset);
        }
        let mut board_entities = [[[NONE; BOARD_ENTITY_SIZE]; MAX_BOARD_TILES]; BOARD_SLOTS];
        for board in &mut board_entities {
            for entity in board {
                *entity = read_array(bytes, &mut offset);
            }
        }
        let mut market_entities = [[NONE; MARKET_ENTITY_SIZE]; 4];
        for entity in &mut market_entities {
            *entity = read_array(bytes, &mut offset);
        }
        let mut targets = [0; TARGET_DIM];
        for target in &mut targets {
            *target = u16::from_le_bytes(read_array(bytes, &mut offset));
        }
        offset += 2;
        debug_assert_eq!(offset, RECORD_SIZE);
        Self {
            game_index,
            turn,
            active_seat,
            player_count,
            total_turns,
            board_counts,
            nature_tokens,
            scoring_cards,
            habitat_bonuses,
            wildlife_counts,
            habitat_sizes,
            board_entities,
            market_entities,
            targets,
        }
    }
}

impl DatasetWriter {
    pub fn open(config: &DatasetWriterConfig) -> Result<Self, DataError> {
        config.validate()?;
        fs::create_dir_all(&config.output)?;
        let manifest_path = config.output.join("dataset.json");
        let mut manifest = if manifest_path.exists() {
            if !config.resume {
                return Err(DataError::DatasetExists(config.output.clone()));
            }
            let manifest: DatasetManifest =
                serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
            validate_writer_resume(&manifest, config)?;
            validate_dataset(&config.output, &manifest)?;
            manifest
        } else {
            let now = unix_seconds()?;
            DatasetManifest {
                schema_version: DATASET_SCHEMA_VERSION,
                dataset_id: format!(
                    "{}-{}-{}",
                    config.strategy_id,
                    config.split.id(),
                    config.first_game_index
                ),
                feature_schema: FEATURE_SCHEMA.to_owned(),
                target_schema: TARGET_SCHEMA.to_owned(),
                record_size: RECORD_SIZE,
                game: GameConfig::research_aaaaa(4)?,
                split: config.split,
                strategy: config.strategy_id.clone(),
                first_game_index: config.first_game_index,
                requested_games: config.games,
                completed_games: 0,
                total_records: 0,
                created_unix_seconds: now,
                updated_unix_seconds: now,
                provenance: collection_provenance()?,
                shards: Vec::new(),
            }
        };
        manifest.requested_games = config.games;
        let strategy_code = dataset_strategy_code(&config.strategy_id);
        Ok(Self {
            output: config.output.clone(),
            manifest_path,
            manifest,
            split: config.split,
            strategy_code,
        })
    }

    pub fn manifest(&self) -> &DatasetManifest {
        &self.manifest
    }

    pub fn append_shard(
        &mut self,
        first_game_index: u64,
        game_count: usize,
        records: &[PositionRecord],
    ) -> Result<(), DataError> {
        if game_count == 0 || records.is_empty() {
            return Err(DataError::InvalidConfig(
                "dataset shard must contain games and records",
            ));
        }
        let expected_first = self.manifest.first_game_index + self.manifest.completed_games as u64;
        if first_game_index != expected_first {
            return Err(DataError::InvalidConfig(
                "dataset shard game range is not contiguous",
            ));
        }
        if self.manifest.completed_games + game_count > self.manifest.requested_games {
            return Err(DataError::InvalidConfig(
                "dataset shard exceeds requested game count",
            ));
        }
        let shard_index = self.manifest.shards.len();
        let file_name = format!("shard-{shard_index:05}.csd");
        let path = self.output.join(&file_name);
        write_shard(
            &path,
            self.split,
            self.strategy_code,
            first_game_index,
            game_count,
            records,
        )?;
        let metadata = fs::metadata(&path)?;
        self.manifest.shards.push(ShardManifest {
            file: file_name,
            first_game_index,
            game_count,
            record_count: records.len(),
            byte_count: metadata.len(),
            blake3: checksum_file(&path)?,
        });
        self.manifest.completed_games += game_count;
        self.manifest.total_records += records.len();
        self.manifest.updated_unix_seconds = unix_seconds()?;
        write_manifest_atomic(&self.manifest_path, &self.manifest)?;
        Ok(())
    }
}

pub fn collect_dataset(config: &CollectConfig) -> Result<DatasetManifest, DataError> {
    config.validate()?;
    let writer_config = DatasetWriterConfig {
        output: config.output.clone(),
        split: config.split,
        first_game_index: config.first_game_index,
        games: config.games,
        strategy_id: config.strategy.id().to_owned(),
        resume: config.resume,
    };
    let mut writer = DatasetWriter::open(&writer_config)?;

    while writer.manifest().completed_games < config.games {
        let game_count = config
            .shard_games
            .min(config.games - writer.manifest().completed_games);
        let first_game_index = config.first_game_index + writer.manifest().completed_games as u64;
        let game_indices: Vec<_> =
            (first_game_index..first_game_index + game_count as u64).collect();
        let mut game_records: Vec<_> = game_indices
            .par_iter()
            .map(|game_index| collect_game(config, *game_index))
            .collect::<Result<_, _>>()?;
        game_records.sort_unstable_by_key(|(game_index, _)| *game_index);
        let records: Vec<_> = game_records
            .into_iter()
            .flat_map(|(_, records)| records)
            .collect();
        writer.append_shard(first_game_index, game_count, &records)?;
    }
    Ok(writer.manifest)
}

pub fn validate_dataset(root: &Path, manifest: &DatasetManifest) -> Result<(), DataError> {
    if manifest.schema_version != DATASET_SCHEMA_VERSION {
        return Err(DataError::InvalidManifest("unsupported schema version"));
    }
    if manifest.feature_schema != FEATURE_SCHEMA || manifest.target_schema != TARGET_SCHEMA {
        return Err(DataError::InvalidManifest(
            "schema identifiers do not match",
        ));
    }
    if manifest.record_size != RECORD_SIZE || manifest.game != GameConfig::research_aaaaa(4)? {
        return Err(DataError::InvalidManifest(
            "record or game configuration does not match",
        ));
    }
    let mut games = 0usize;
    let mut records = 0usize;
    for shard in &manifest.shards {
        let path = root.join(&shard.file);
        let metadata = fs::metadata(&path)?;
        if metadata.len() != shard.byte_count {
            return Err(DataError::InvalidManifest("shard byte count mismatch"));
        }
        if checksum_file(&path)? != shard.blake3 {
            return Err(DataError::ChecksumMismatch(path));
        }
        validate_shard_header(&path, shard)?;
        games += shard.game_count;
        records += shard.record_count;
    }
    if games != manifest.completed_games || records != manifest.total_records {
        return Err(DataError::InvalidManifest(
            "manifest totals do not match shards",
        ));
    }
    Ok(())
}

fn collect_game(
    config: &CollectConfig,
    game_index: u64,
) -> Result<(u64, Vec<PositionRecord>), DataError> {
    let game = GameConfig::research_aaaaa(4)?;
    let match_config =
        MatchConfig::symmetric(game, config.split.game_seed(game_index), config.strategy);
    let mut records = Vec::with_capacity(80);
    let result = play_match_observed(&match_config, |state, _| {
        records.push(PositionRecord::observe(state, game_index));
    })?;
    for record in &mut records {
        record.set_target(result.scores[usize::from(record.active_seat)]);
    }
    Ok((game_index, records))
}

fn write_shard(
    path: &Path,
    split: DatasetSplit,
    strategy_code: u8,
    first_game_index: u64,
    game_count: usize,
    records: &[PositionRecord],
) -> Result<(), DataError> {
    let temp_path = path.with_extension("csd.tmp");
    let mut writer = BufWriter::new(File::create(&temp_path)?);
    writer.write_all(SHARD_MAGIC)?;
    writer.write_all(&DATASET_SCHEMA_VERSION.to_le_bytes())?;
    writer.write_all(&(SHARD_HEADER_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(RECORD_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(TARGET_DIM as u16).to_le_bytes())?;
    writer.write_all(&(records.len() as u32).to_le_bytes())?;
    writer.write_all(&(game_count as u32).to_le_bytes())?;
    writer.write_all(&first_game_index.to_le_bytes())?;
    writer.write_all(&[split.code(), strategy_code, 4, 0])?;
    writer.write_all(&[0, 0, 0, 0, 0])?;
    writer.write_all(&[0; 7])?;
    writer.write_all(blake3::hash(FEATURE_SCHEMA.as_bytes()).as_bytes())?;
    for record in records {
        writer.write_all(&record.to_bytes())?;
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(temp_path, path)?;
    Ok(())
}

fn validate_shard_header(path: &Path, shard: &ShardManifest) -> Result<(), DataError> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut header = [0u8; SHARD_HEADER_SIZE];
    reader.read_exact(&mut header)?;
    if &header[..8] != SHARD_MAGIC {
        return Err(DataError::InvalidShard("invalid magic"));
    }
    if u16::from_le_bytes([header[8], header[9]]) != DATASET_SCHEMA_VERSION
        || u16::from_le_bytes([header[10], header[11]]) as usize != SHARD_HEADER_SIZE
        || u16::from_le_bytes([header[12], header[13]]) as usize != RECORD_SIZE
        || u16::from_le_bytes([header[14], header[15]]) as usize != TARGET_DIM
    {
        return Err(DataError::InvalidShard("incompatible header"));
    }
    let record_count =
        u32::from_le_bytes(header[16..20].try_into().expect("fixed header")) as usize;
    let game_count = u32::from_le_bytes(header[20..24].try_into().expect("fixed header")) as usize;
    let first_game_index = u64::from_le_bytes(header[24..32].try_into().expect("fixed header"));
    if record_count != shard.record_count
        || game_count != shard.game_count
        || first_game_index != shard.first_game_index
    {
        return Err(DataError::InvalidShard("header and manifest disagree"));
    }
    let expected_size = SHARD_HEADER_SIZE as u64 + record_count as u64 * RECORD_SIZE as u64;
    if fs::metadata(path)?.len() != expected_size {
        return Err(DataError::InvalidShard(
            "file size does not match record count",
        ));
    }
    Ok(())
}

fn validate_writer_resume(
    manifest: &DatasetManifest,
    config: &DatasetWriterConfig,
) -> Result<(), DataError> {
    let current_provenance = collection_provenance()?;
    if manifest.schema_version != DATASET_SCHEMA_VERSION
        || manifest.feature_schema != FEATURE_SCHEMA
        || manifest.target_schema != TARGET_SCHEMA
        || manifest.record_size != RECORD_SIZE
        || manifest.game != GameConfig::research_aaaaa(4)?
        || manifest.split != config.split
        || manifest.strategy != config.strategy_id
        || manifest.first_game_index != config.first_game_index
        || !collection_provenance_matches(&manifest.provenance, &current_provenance)
    {
        return Err(DataError::ResumeMismatch);
    }
    if config.games < manifest.completed_games {
        return Err(DataError::InvalidConfig(
            "requested games are less than completed games",
        ));
    }
    Ok(())
}

fn collection_provenance_matches(
    original: &CollectionProvenance,
    current: &CollectionProvenance,
) -> bool {
    original.executable_blake3 == current.executable_blake3
        && (original.v2_source_blake3.is_empty()
            || original.v2_source_blake3 == current.v2_source_blake3)
}

fn write_manifest_atomic(path: &Path, manifest: &impl Serialize) -> Result<(), DataError> {
    let temp_path = path.with_extension("json.tmp");
    let mut writer = BufWriter::new(File::create(&temp_path)?);
    serde_json::to_writer_pretty(&mut writer, manifest)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(temp_path, path)?;
    Ok(())
}

fn checksum_file(path: &Path) -> Result<String, DataError> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Hasher::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn collection_provenance() -> Result<CollectionProvenance, DataError> {
    let executable = std::env::current_exe()?;
    let source = source_provenance()?;
    let chip = command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
        .unwrap_or_else(|| std::env::consts::ARCH.to_owned());
    let memory =
        command_output("sysctl", &["-n", "hw.memsize"]).unwrap_or_else(|| "unknown".into());
    let os = command_output("sw_vers", &["-productVersion"])
        .unwrap_or_else(|| std::env::consts::OS.to_owned());
    Ok(CollectionProvenance {
        collector_version: env!("CARGO_PKG_VERSION").to_owned(),
        git_revision: source.git_revision,
        git_dirty: source.git_dirty,
        git_status_blake3: source.git_status_blake3,
        v2_source_blake3: source.v2_source_blake3,
        executable_blake3: checksum_file(&executable)?,
        hardware: format!("{chip}; memory_bytes={memory}; os={os}"),
    })
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn scoring_card_codes(config: GameConfig) -> [u8; 5] {
    let cards = config.scoring_cards;
    [
        scoring_variant_code(cards.bear),
        scoring_variant_code(cards.elk),
        scoring_variant_code(cards.salmon),
        scoring_variant_code(cards.hawk),
        scoring_variant_code(cards.fox),
    ]
}

const fn scoring_variant_code(variant: ScoringVariant) -> u8 {
    match variant {
        ScoringVariant::A => 0,
        ScoringVariant::B => 1,
        ScoringVariant::C => 2,
        ScoringVariant::D => 3,
    }
}

fn dataset_strategy_code(strategy_id: &str) -> u8 {
    match strategy_id {
        "random-v1" => 0,
        "greedy-v1" => 1,
        _ => u8::MAX,
    }
}

fn unix_seconds() -> Result<u64, DataError> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| DataError::ClockBeforeEpoch)?
        .as_secs())
}

fn write_slice(output: &mut [u8], offset: &mut usize, value: &[u8]) {
    output[*offset..*offset + value.len()].copy_from_slice(value);
    *offset += value.len();
}

fn read_array<const N: usize>(input: &[u8], offset: &mut usize) -> [u8; N] {
    let value = input[*offset..*offset + N]
        .try_into()
        .expect("record field has fixed width");
    *offset += N;
    value
}

#[derive(Debug, Error)]
pub enum DataError {
    #[error("invalid collection configuration: {0}")]
    InvalidConfig(&'static str),
    #[error("dataset already exists at {0}; pass resume to continue it")]
    DatasetExists(PathBuf),
    #[error("existing dataset does not match the requested collection")]
    ResumeMismatch,
    #[error("dataset manifest is invalid: {0}")]
    InvalidManifest(&'static str),
    #[error("dataset shard is invalid: {0}")]
    InvalidShard(&'static str),
    #[error("dataset shard checksum does not match: {0}")]
    ChecksumMismatch(PathBuf),
    #[error("system clock is before the Unix epoch")]
    ClockBeforeEpoch,
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Rules(#[from] cascadia_game::RuleError),
    #[error(transparent)]
    Simulation(#[from] SimulationError),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record() -> PositionRecord {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            DatasetSplit::Train.game_seed(7),
        )
        .unwrap();
        let mut record = PositionRecord::observe(&game, 7);
        record.set_target(ScoreBreakdown {
            habitat: [4, 5, 6, 7, 8],
            wildlife: [9, 10, 11, 12, 13],
            nature_tokens: 3,
            habitat_bonus: [0; 5],
            base_total: 88,
            total: 88,
        });
        record
    }

    #[test]
    fn split_seed_namespaces_are_stable_and_disjoint() {
        assert_eq!(
            DatasetSplit::Train.game_seed(42),
            DatasetSplit::Train.game_seed(42)
        );
        assert_ne!(
            DatasetSplit::Train.game_seed(42),
            DatasetSplit::Validation.game_seed(42)
        );
        assert_ne!(
            DatasetSplit::Validation.game_seed(42),
            DatasetSplit::Test.game_seed(42)
        );
    }

    #[test]
    fn fixed_record_round_trip_is_exact() {
        let expected = sample_record();
        let bytes = expected.to_bytes();
        assert_eq!(bytes.len(), RECORD_SIZE);
        assert_eq!(PositionRecord::from_bytes(&bytes), expected);
    }

    #[test]
    fn record_uses_active_player_relative_board_order() {
        let record = sample_record();
        assert_eq!(record.board_counts[0], 3);
        assert_eq!(record.board_counts[1], 3);
        assert!(record.board_entities[0][0][0] != NONE);
        assert!(
            record.board_entities[0][3]
                .iter()
                .all(|value| *value == NONE)
        );
    }

    #[test]
    fn observable_afterstate_excludes_the_hidden_market_refill() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            DatasetSplit::Train.game_seed(8),
        )
        .unwrap();
        let action = game
            .legal_turn_actions(&cascadia_game::MarketPrelude::default())
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let cascadia_game::DraftChoice::Paired { slot } = action.draft else {
            panic!("the first default action uses a paired draft");
        };
        let original = PositionRecord::observe(&game, 8);
        let transitioned = game.transition(&action).unwrap();
        let afterstate = PositionRecord::observable_afterstate(&game, &action, 8).unwrap();
        let mut redetermined = game.clone();
        redetermined.redeterminize_hidden(GameSeed::from_u64(9));
        let redetermined_afterstate =
            PositionRecord::observable_afterstate(&redetermined, &action, 8).unwrap();

        assert_eq!(afterstate.turn, original.turn + 1);
        assert_eq!(afterstate.board_counts[0], original.board_counts[0] + 1);
        assert_eq!(afterstate.board_entities[1], original.board_entities[1]);
        assert_ne!(afterstate.market_entities, original.market_entities);
        assert_eq!(afterstate.active_seat, 0);
        assert!(
            afterstate.market_entities[slot.index()]
                .iter()
                .all(|value| *value == NONE)
        );
        assert_eq!(afterstate, redetermined_afterstate);
        assert_ne!(
            afterstate,
            PositionRecord::observe_for_seat(&transitioned, 8, 0)
        );
    }

    #[test]
    fn collection_writes_valid_resumable_shards() {
        let root = std::env::temp_dir().join(format!("cascadia-data-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let mut config = CollectConfig {
            output: root.clone(),
            split: DatasetSplit::Train,
            first_game_index: 0,
            games: 1,
            shard_games: 1,
            strategy: StrategyKind::Random,
            resume: false,
        };
        let first = collect_dataset(&config).unwrap();
        assert_eq!(first.completed_games, 1);
        assert_eq!(first.total_records, 80);
        assert_eq!(first.shards.len(), 1);
        validate_dataset(&root, &first).unwrap();

        config.games = 2;
        config.resume = true;
        let resumed = collect_dataset(&config).unwrap();
        assert_eq!(resumed.completed_games, 2);
        assert_eq!(resumed.total_records, 160);
        assert_eq!(resumed.shards.len(), 2);
        validate_dataset(&root, &resumed).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reusable_writer_preserves_custom_strategy_identity() {
        let root =
            std::env::temp_dir().join(format!("cascadia-writer-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let config = DatasetWriterConfig {
            output: root.clone(),
            split: DatasetSplit::Train,
            first_game_index: 10,
            games: 2,
            strategy_id: "teacher-search-v1".to_owned(),
            resume: false,
        };
        let mut writer = DatasetWriter::open(&config).unwrap();
        writer.append_shard(10, 1, &[sample_record()]).unwrap();
        drop(writer);

        let resumed = DatasetWriter::open(&DatasetWriterConfig {
            resume: true,
            ..config
        })
        .unwrap();
        assert_eq!(resumed.manifest().strategy, "teacher-search-v1");
        assert_eq!(resumed.manifest().completed_games, 1);
        assert_eq!(resumed.manifest().requested_games, 2);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn collection_resume_rejects_changed_provenance() {
        let root =
            std::env::temp_dir().join(format!("cascadia-data-provenance-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let config = CollectConfig {
            output: root.clone(),
            split: DatasetSplit::Train,
            first_game_index: 0,
            games: 1,
            shard_games: 1,
            strategy: StrategyKind::Random,
            resume: false,
        };
        collect_dataset(&config).unwrap();
        let manifest_path = root.join("dataset.json");
        let mut manifest: DatasetManifest =
            serde_json::from_reader(File::open(&manifest_path).unwrap()).unwrap();
        manifest.provenance.v2_source_blake3 = "different-source".to_owned();
        write_manifest_atomic(&manifest_path, &manifest).unwrap();

        let error = collect_dataset(&CollectConfig {
            games: 2,
            resume: true,
            ..config
        })
        .unwrap_err();
        assert!(matches!(error, DataError::ResumeMismatch));
        fs::remove_dir_all(root).unwrap();
    }
}
