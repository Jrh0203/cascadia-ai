use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use cascadia_game::{DraftChoice, GameConfig, GameState, ScoreBreakdown, TurnAction, score_board};
use serde::{Deserialize, Serialize};

use super::{
    CollectionProvenance, DataError, DatasetSplit, FEATURE_SCHEMA, PositionRecord,
    RankingDatasetManifest, RankingShardManifest, RankingTeacherConfig, RankingTrajectoryConfig,
    checksum_file, collection_provenance, collection_provenance_matches, read_array, unix_seconds,
    validate_ranking_dataset, write_manifest_atomic, write_slice,
};

pub const ACTION_RANKING_DATASET_SCHEMA_VERSION: u16 = 1;
pub const ACTION_FEATURE_SCHEMA: &str = "compact-action-delta-v1";
pub const ACTION_RANKING_TARGET_SCHEMA: &str = "search-action-ranking-v1";
pub const ACTION_RANKING_SHARD_MAGIC: &[u8; 8] = b"CSD2ARK\0";
pub const ACTION_RANKING_HEADER_SIZE: usize = 112;
pub const ACTION_FEATURE_SIZE: usize = 52;
pub const ACTION_POSITION_RECORD_SIZE: usize = super::RECORD_SIZE + ACTION_FEATURE_SIZE;
pub const ACTION_RANKING_RECORD_SIZE: usize = 56 + ACTION_POSITION_RECORD_SIZE;

const NONE: u8 = u8::MAX;
const SCORE_COMPONENTS: usize = 11;

#[derive(Debug, Clone)]
pub struct ActionRankingDatasetConfig {
    pub output: PathBuf,
    pub source_root: PathBuf,
    pub source_manifest: RankingDatasetManifest,
    pub resume: bool,
}

impl ActionRankingDatasetConfig {
    fn validate(&self) -> Result<(), DataError> {
        validate_ranking_dataset(&self.source_root, &self.source_manifest)?;
        if self.source_manifest.completed_games == 0
            || self.source_manifest.total_groups == 0
            || self.source_manifest.total_records == 0
        {
            return Err(DataError::InvalidConfig(
                "action-ranking source dataset must contain completed games",
            ));
        }
        if self.source_manifest.feature_schema != FEATURE_SCHEMA {
            return Err(DataError::InvalidConfig(
                "action-ranking source feature schema is unsupported",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionRankingSourceManifest {
    pub path: String,
    pub dataset_id: String,
    pub manifest_blake3: String,
    pub feature_schema: String,
    pub target_schema: String,
    pub record_size: usize,
    pub first_game_index: u64,
    pub completed_games: usize,
    pub total_groups: usize,
    pub total_records: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionRankingDatasetManifest {
    pub schema_version: u16,
    pub dataset_id: String,
    pub feature_schema: String,
    pub position_feature_schema: String,
    pub target_schema: String,
    pub record_size: usize,
    pub action_feature_size: usize,
    pub game: GameConfig,
    pub split: DatasetSplit,
    pub teacher: RankingTeacherConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trajectory: Option<RankingTrajectoryConfig>,
    pub source: ActionRankingSourceManifest,
    pub first_game_index: u64,
    pub requested_games: usize,
    pub completed_games: usize,
    pub total_groups: usize,
    pub total_records: usize,
    pub created_unix_seconds: u64,
    pub updated_unix_seconds: u64,
    pub provenance: CollectionProvenance,
    pub shards: Vec<RankingShardManifest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionFeatures {
    pub draft_kind: u8,
    pub tile_slot: u8,
    pub wildlife_slot: u8,
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
    pub paid_wipe_count: u8,
    pub paid_wipe_slot_mask: u8,
    pub paid_wipe_total_slots: u8,
    pub immediate_rank: u16,
    pub immediate_score: u16,
    pub immediate_deltas: [i16; SCORE_COMPONENTS],
}

impl ActionFeatures {
    pub fn from_game_action(
        game: &GameState,
        action: &TurnAction,
        immediate_rank: u16,
        immediate_score: u16,
    ) -> Result<Self, DataError> {
        let acting_seat = game.current_player();
        let staged = game.preview_market_prelude(&action.prelude())?;
        let (draft_kind, tile_slot, wildlife_slot) = match action.draft {
            DraftChoice::Paired { slot } => (0, slot.index() as u8, slot.index() as u8),
            DraftChoice::Independent {
                tile_slot,
                wildlife_slot,
            } => (1, tile_slot.index() as u8, wildlife_slot.index() as u8),
        };
        let tile = staged.market().tiles[usize::from(tile_slot)].ok_or(
            DataError::InvalidConfig("action-ranking tile slot is unavailable after its prelude"),
        )?;
        let wildlife = staged.market().wildlife[usize::from(wildlife_slot)].ok_or(
            DataError::InvalidConfig(
                "action-ranking wildlife slot is unavailable after its prelude",
            ),
        )?;

        let before = score_board(&game.boards()[acting_seat], game.config().scoring_cards);
        let after_board = game.preview_active_board(action)?;
        let after = score_board(&after_board, game.config().scoring_cards);
        if immediate_rank == 0 || immediate_score != after.base_total {
            return Err(DataError::InvalidConfig(
                "action-ranking immediate metadata does not match the canonical afterstate",
            ));
        }
        let immediate_deltas = score_deltas(before, after);

        let mut paid_wipe_slot_mask = 0u8;
        let mut paid_wipe_total_slots = 0usize;
        for wipe in &action.wildlife_wipes {
            paid_wipe_total_slots += wipe.slots.len();
            for slot in &wipe.slots {
                paid_wipe_slot_mask |= 1 << slot.index();
            }
        }

        Ok(Self {
            draft_kind,
            tile_slot,
            wildlife_slot,
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
            paid_wipe_count: u8::try_from(action.wildlife_wipes.len()).map_err(|_| {
                DataError::InvalidConfig("action-ranking wipe count exceeds fixed-width storage")
            })?,
            paid_wipe_slot_mask,
            paid_wipe_total_slots: u8::try_from(paid_wipe_total_slots).map_err(|_| {
                DataError::InvalidConfig(
                    "action-ranking wiped-slot count exceeds fixed-width storage",
                )
            })?,
            immediate_rank,
            immediate_score,
            immediate_deltas,
        })
    }

    pub fn to_bytes(&self) -> [u8; ACTION_FEATURE_SIZE] {
        let mut bytes = [0; ACTION_FEATURE_SIZE];
        let mut offset = 0;
        write_slice(
            &mut bytes,
            &mut offset,
            &[
                self.draft_kind,
                self.tile_slot,
                self.wildlife_slot,
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
                self.paid_wipe_count,
                self.paid_wipe_slot_mask,
                self.paid_wipe_total_slots,
            ],
        );
        write_slice(&mut bytes, &mut offset, &self.immediate_rank.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.immediate_score.to_le_bytes());
        for delta in self.immediate_deltas {
            write_slice(&mut bytes, &mut offset, &delta.to_le_bytes());
        }
        offset += 8;
        debug_assert_eq!(offset, ACTION_FEATURE_SIZE);
        bytes
    }

    pub fn from_bytes(bytes: &[u8; ACTION_FEATURE_SIZE]) -> Self {
        let mut offset = 0;
        let [
            draft_kind,
            tile_slot,
            wildlife_slot,
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
            paid_wipe_count,
            paid_wipe_slot_mask,
            paid_wipe_total_slots,
        ] = read_array(bytes, &mut offset);
        let immediate_rank = u16::from_le_bytes(read_array(bytes, &mut offset));
        let immediate_score = u16::from_le_bytes(read_array(bytes, &mut offset));
        let mut immediate_deltas = [0; SCORE_COMPONENTS];
        for delta in &mut immediate_deltas {
            *delta = i16::from_le_bytes(read_array(bytes, &mut offset));
        }
        offset += 8;
        debug_assert_eq!(offset, ACTION_FEATURE_SIZE);
        Self {
            draft_kind,
            tile_slot,
            wildlife_slot,
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
            paid_wipe_count,
            paid_wipe_slot_mask,
            paid_wipe_total_slots,
            immediate_rank,
            immediate_score,
            immediate_deltas,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionPositionRecord {
    pub position: PositionRecord,
    pub action: ActionFeatures,
}

impl ActionPositionRecord {
    pub fn observe(
        game: &GameState,
        action: &TurnAction,
        game_index: u64,
        immediate_rank: u16,
        immediate_score: u16,
    ) -> Result<Self, DataError> {
        Ok(Self {
            position: PositionRecord::observable_afterstate(game, action, game_index)?,
            action: ActionFeatures::from_game_action(
                game,
                action,
                immediate_rank,
                immediate_score,
            )?,
        })
    }

    pub fn to_bytes(&self) -> [u8; ACTION_POSITION_RECORD_SIZE] {
        let mut bytes = [0; ACTION_POSITION_RECORD_SIZE];
        let mut offset = 0;
        write_slice(&mut bytes, &mut offset, &self.position.to_bytes());
        write_slice(&mut bytes, &mut offset, &self.action.to_bytes());
        debug_assert_eq!(offset, ACTION_POSITION_RECORD_SIZE);
        bytes
    }

    pub fn from_bytes(bytes: &[u8; ACTION_POSITION_RECORD_SIZE]) -> Self {
        let mut offset = 0;
        let position_bytes: [u8; super::RECORD_SIZE] = read_array(bytes, &mut offset);
        let action_bytes: [u8; ACTION_FEATURE_SIZE] = read_array(bytes, &mut offset);
        debug_assert_eq!(offset, ACTION_POSITION_RECORD_SIZE);
        Self {
            position: PositionRecord::from_bytes(&position_bytes),
            action: ActionFeatures::from_bytes(&action_bytes),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActionRankingRecord {
    pub group_id: u64,
    pub candidate_index: u16,
    pub candidate_count: u16,
    pub immediate_rank: u16,
    pub immediate_score: u16,
    pub teacher_mean: f32,
    pub teacher_stddev: f32,
    pub action_hash: [u8; 32],
    pub input: ActionPositionRecord,
}

impl ActionRankingRecord {
    pub fn to_bytes(&self) -> [u8; ACTION_RANKING_RECORD_SIZE] {
        let mut bytes = [0; ACTION_RANKING_RECORD_SIZE];
        let mut offset = 0;
        write_slice(&mut bytes, &mut offset, &self.group_id.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.candidate_index.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.candidate_count.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.immediate_rank.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.immediate_score.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.teacher_mean.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.teacher_stddev.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.action_hash);
        write_slice(&mut bytes, &mut offset, &self.input.to_bytes());
        debug_assert_eq!(offset, ACTION_RANKING_RECORD_SIZE);
        bytes
    }

    pub fn from_bytes(bytes: &[u8; ACTION_RANKING_RECORD_SIZE]) -> Self {
        let mut offset = 0;
        let group_id = u64::from_le_bytes(read_array(bytes, &mut offset));
        let candidate_index = u16::from_le_bytes(read_array(bytes, &mut offset));
        let candidate_count = u16::from_le_bytes(read_array(bytes, &mut offset));
        let immediate_rank = u16::from_le_bytes(read_array(bytes, &mut offset));
        let immediate_score = u16::from_le_bytes(read_array(bytes, &mut offset));
        let teacher_mean = f32::from_le_bytes(read_array(bytes, &mut offset));
        let teacher_stddev = f32::from_le_bytes(read_array(bytes, &mut offset));
        let action_hash = read_array(bytes, &mut offset);
        let input_bytes: [u8; ACTION_POSITION_RECORD_SIZE] = read_array(bytes, &mut offset);
        debug_assert_eq!(offset, ACTION_RANKING_RECORD_SIZE);
        Self {
            group_id,
            candidate_index,
            candidate_count,
            immediate_rank,
            immediate_score,
            teacher_mean,
            teacher_stddev,
            action_hash,
            input: ActionPositionRecord::from_bytes(&input_bytes),
        }
    }
}

pub struct ActionRankingDatasetWriter {
    output: PathBuf,
    manifest_path: PathBuf,
    manifest: ActionRankingDatasetManifest,
}

impl ActionRankingDatasetWriter {
    pub fn open(config: &ActionRankingDatasetConfig) -> Result<Self, DataError> {
        config.validate()?;
        fs::create_dir_all(&config.output)?;
        let source = source_manifest(config)?;
        let manifest_path = config.output.join("dataset.json");
        let manifest = if manifest_path.exists() {
            if !config.resume {
                return Err(DataError::DatasetExists(config.output.clone()));
            }
            let manifest: ActionRankingDatasetManifest =
                serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
            validate_resume(&manifest, config, &source)?;
            validate_action_ranking_dataset(&config.output, &manifest)?;
            manifest
        } else {
            let now = unix_seconds()?;
            ActionRankingDatasetManifest {
                schema_version: ACTION_RANKING_DATASET_SCHEMA_VERSION,
                dataset_id: format!("action-{}", config.source_manifest.dataset_id),
                feature_schema: ACTION_FEATURE_SCHEMA.to_owned(),
                position_feature_schema: FEATURE_SCHEMA.to_owned(),
                target_schema: ACTION_RANKING_TARGET_SCHEMA.to_owned(),
                record_size: ACTION_RANKING_RECORD_SIZE,
                action_feature_size: ACTION_FEATURE_SIZE,
                game: config.source_manifest.game,
                split: config.source_manifest.split,
                teacher: config.source_manifest.teacher.clone(),
                trajectory: config.source_manifest.trajectory.clone(),
                source,
                first_game_index: config.source_manifest.first_game_index,
                requested_games: config.source_manifest.completed_games,
                completed_games: 0,
                total_groups: 0,
                total_records: 0,
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

    pub fn manifest(&self) -> &ActionRankingDatasetManifest {
        &self.manifest
    }

    pub fn append_shard(
        &mut self,
        first_game_index: u64,
        game_count: usize,
        records: &[ActionRankingRecord],
    ) -> Result<(), DataError> {
        if game_count == 0 || records.is_empty() {
            return Err(DataError::InvalidConfig(
                "action-ranking shard must contain games and records",
            ));
        }
        let expected_first = self.manifest.first_game_index + self.manifest.completed_games as u64;
        if first_game_index != expected_first {
            return Err(DataError::InvalidConfig(
                "action-ranking shard game range is not contiguous",
            ));
        }
        if self.manifest.completed_games + game_count > self.manifest.requested_games {
            return Err(DataError::InvalidConfig(
                "action-ranking shard exceeds requested game count",
            ));
        }
        let group_count = validate_record_groups(records)?;
        let shard_index = self.manifest.shards.len();
        let file_name = format!("shard-{shard_index:05}.car");
        let path = self.output.join(&file_name);
        write_shard(
            &path,
            self.manifest.split,
            first_game_index,
            game_count,
            group_count,
            records,
        )?;
        let metadata = fs::metadata(&path)?;
        self.manifest.shards.push(RankingShardManifest {
            file: file_name,
            first_game_index,
            game_count,
            group_count,
            record_count: records.len(),
            byte_count: metadata.len(),
            blake3: checksum_file(&path)?,
        });
        self.manifest.completed_games += game_count;
        self.manifest.total_groups += group_count;
        self.manifest.total_records += records.len();
        self.manifest.updated_unix_seconds = unix_seconds()?;
        write_manifest_atomic(&self.manifest_path, &self.manifest)?;
        Ok(())
    }
}

pub fn validate_action_ranking_dataset(
    root: &Path,
    manifest: &ActionRankingDatasetManifest,
) -> Result<(), DataError> {
    if manifest.schema_version != ACTION_RANKING_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != ACTION_FEATURE_SCHEMA
        || manifest.position_feature_schema != FEATURE_SCHEMA
        || manifest.target_schema != ACTION_RANKING_TARGET_SCHEMA
        || manifest.record_size != ACTION_RANKING_RECORD_SIZE
        || manifest.action_feature_size != ACTION_FEATURE_SIZE
        || manifest.game != GameConfig::research_aaaaa(4)?
    {
        return Err(DataError::InvalidManifest(
            "action-ranking schema identifiers do not match",
        ));
    }
    let mut games = 0;
    let mut groups = 0;
    let mut records = 0;
    for shard in &manifest.shards {
        let path = root.join(&shard.file);
        let metadata = fs::metadata(&path)?;
        if metadata.len() != shard.byte_count {
            return Err(DataError::InvalidManifest(
                "action-ranking shard byte count mismatch",
            ));
        }
        if checksum_file(&path)? != shard.blake3 {
            return Err(DataError::ChecksumMismatch(path));
        }
        validate_shard_header(&path, manifest.split, shard)?;
        games += shard.game_count;
        groups += shard.group_count;
        records += shard.record_count;
    }
    if games != manifest.completed_games
        || groups != manifest.total_groups
        || records != manifest.total_records
    {
        return Err(DataError::InvalidManifest(
            "action-ranking manifest totals do not match shards",
        ));
    }
    Ok(())
}

fn source_manifest(
    config: &ActionRankingDatasetConfig,
) -> Result<ActionRankingSourceManifest, DataError> {
    let manifest_path = config.source_root.join("dataset.json");
    Ok(ActionRankingSourceManifest {
        path: config.source_root.canonicalize()?.display().to_string(),
        dataset_id: config.source_manifest.dataset_id.clone(),
        manifest_blake3: checksum_file(&manifest_path)?,
        feature_schema: config.source_manifest.feature_schema.clone(),
        target_schema: config.source_manifest.target_schema.clone(),
        record_size: config.source_manifest.record_size,
        first_game_index: config.source_manifest.first_game_index,
        completed_games: config.source_manifest.completed_games,
        total_groups: config.source_manifest.total_groups,
        total_records: config.source_manifest.total_records,
    })
}

fn validate_resume(
    manifest: &ActionRankingDatasetManifest,
    config: &ActionRankingDatasetConfig,
    source: &ActionRankingSourceManifest,
) -> Result<(), DataError> {
    let current_provenance = collection_provenance()?;
    if manifest.schema_version != ACTION_RANKING_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != ACTION_FEATURE_SCHEMA
        || manifest.position_feature_schema != FEATURE_SCHEMA
        || manifest.target_schema != ACTION_RANKING_TARGET_SCHEMA
        || manifest.record_size != ACTION_RANKING_RECORD_SIZE
        || manifest.action_feature_size != ACTION_FEATURE_SIZE
        || manifest.game != config.source_manifest.game
        || manifest.split != config.source_manifest.split
        || manifest.teacher != config.source_manifest.teacher
        || manifest.trajectory != config.source_manifest.trajectory
        || manifest.source != *source
        || manifest.first_game_index != config.source_manifest.first_game_index
        || manifest.requested_games != config.source_manifest.completed_games
        || !collection_provenance_matches(&manifest.provenance, &current_provenance)
    {
        return Err(DataError::ResumeMismatch);
    }
    Ok(())
}

fn validate_record_groups(records: &[ActionRankingRecord]) -> Result<usize, DataError> {
    let mut groups: BTreeMap<u64, Vec<&ActionRankingRecord>> = BTreeMap::new();
    for record in records {
        groups.entry(record.group_id).or_default().push(record);
    }
    for group in groups.values_mut() {
        group.sort_unstable_by_key(|record| record.candidate_index);
        let count = group.len();
        if count > usize::from(u16::MAX)
            || group.iter().enumerate().any(|(index, record)| {
                usize::from(record.candidate_count) != count
                    || usize::from(record.candidate_index) != index
                    || record.immediate_rank == 0
                    || record.input.action.immediate_rank != record.immediate_rank
                    || record.input.action.immediate_score != record.immediate_score
                    || record.input.action.draft_kind > 1
                    || record.input.action.tile_slot > 3
                    || record.input.action.wildlife_slot > 3
            })
        {
            return Err(DataError::InvalidConfig(
                "action-ranking candidate group is inconsistent",
            ));
        }
    }
    Ok(groups.len())
}

fn write_shard(
    path: &Path,
    split: DatasetSplit,
    first_game_index: u64,
    game_count: usize,
    group_count: usize,
    records: &[ActionRankingRecord],
) -> Result<(), DataError> {
    let temp_path = path.with_extension("car.tmp");
    let mut writer = BufWriter::new(File::create(&temp_path)?);
    writer.write_all(ACTION_RANKING_SHARD_MAGIC)?;
    writer.write_all(&ACTION_RANKING_DATASET_SCHEMA_VERSION.to_le_bytes())?;
    writer.write_all(&(ACTION_RANKING_HEADER_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(ACTION_RANKING_RECORD_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(ACTION_FEATURE_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(records.len() as u32).to_le_bytes())?;
    writer.write_all(&(group_count as u32).to_le_bytes())?;
    writer.write_all(&(game_count as u32).to_le_bytes())?;
    writer.write_all(&[split.code(), 4, 0, 0])?;
    writer.write_all(&first_game_index.to_le_bytes())?;
    writer.write_all(blake3::hash(ACTION_FEATURE_SCHEMA.as_bytes()).as_bytes())?;
    writer.write_all(blake3::hash(ACTION_RANKING_TARGET_SCHEMA.as_bytes()).as_bytes())?;
    writer.write_all(&[0; 8])?;
    for record in records {
        writer.write_all(&record.to_bytes())?;
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
    let mut header = [0; ACTION_RANKING_HEADER_SIZE];
    reader.read_exact(&mut header)?;
    if &header[..8] != ACTION_RANKING_SHARD_MAGIC
        || u16::from_le_bytes([header[8], header[9]]) != ACTION_RANKING_DATASET_SCHEMA_VERSION
        || u16::from_le_bytes([header[10], header[11]]) as usize != ACTION_RANKING_HEADER_SIZE
        || u16::from_le_bytes([header[12], header[13]]) as usize != ACTION_RANKING_RECORD_SIZE
        || u16::from_le_bytes([header[14], header[15]]) as usize != ACTION_FEATURE_SIZE
        || header[28] != split.code()
        || &header[40..72] != blake3::hash(ACTION_FEATURE_SCHEMA.as_bytes()).as_bytes()
        || &header[72..104] != blake3::hash(ACTION_RANKING_TARGET_SCHEMA.as_bytes()).as_bytes()
    {
        return Err(DataError::InvalidShard(
            "incompatible action-ranking header",
        ));
    }
    let record_count =
        u32::from_le_bytes(header[16..20].try_into().expect("fixed header")) as usize;
    let group_count = u32::from_le_bytes(header[20..24].try_into().expect("fixed header")) as usize;
    let game_count = u32::from_le_bytes(header[24..28].try_into().expect("fixed header")) as usize;
    let first_game_index = u64::from_le_bytes(header[32..40].try_into().expect("fixed header"));
    if record_count != shard.record_count
        || group_count != shard.group_count
        || game_count != shard.game_count
        || first_game_index != shard.first_game_index
    {
        return Err(DataError::InvalidShard(
            "action-ranking header and manifest disagree",
        ));
    }
    let expected_size =
        ACTION_RANKING_HEADER_SIZE as u64 + record_count as u64 * ACTION_RANKING_RECORD_SIZE as u64;
    if fs::metadata(path)?.len() != expected_size {
        return Err(DataError::InvalidShard(
            "action-ranking shard file size does not match records",
        ));
    }
    Ok(())
}

fn score_deltas(before: ScoreBreakdown, after: ScoreBreakdown) -> [i16; SCORE_COMPONENTS] {
    let mut deltas = [0; SCORE_COMPONENTS];
    for (index, delta) in deltas[..5].iter_mut().enumerate() {
        *delta = signed_delta(after.habitat[index], before.habitat[index]);
    }
    for (index, delta) in deltas[5..10].iter_mut().enumerate() {
        *delta = signed_delta(after.wildlife[index], before.wildlife[index]);
    }
    deltas[10] = signed_delta(after.nature_tokens, before.nature_tokens);
    deltas
}

fn signed_delta(after: u16, before: u16) -> i16 {
    i16::try_from(i32::from(after) - i32::from(before))
        .expect("Cascadia score deltas fit signed 16-bit storage")
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameSeed, MarketPrelude, MarketSlot, Rotation};

    use super::*;

    fn sample_input(immediate_rank: u16) -> ActionPositionRecord {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(81),
        )
        .unwrap();
        let coord = game.boards()[0].frontier()[0];
        let action = TurnAction::paired(MarketSlot::ZERO, coord, Rotation::ZERO);
        let immediate_score = score_board(
            &game.preview_active_board(&action).unwrap(),
            game.config().scoring_cards,
        )
        .base_total;
        ActionPositionRecord::observe(&game, &action, 4, immediate_rank, immediate_score).unwrap()
    }

    fn sample_record(index: u16) -> ActionRankingRecord {
        let input = sample_input(index + 1);
        ActionRankingRecord {
            group_id: 9,
            candidate_index: index,
            candidate_count: 2,
            immediate_rank: index + 1,
            immediate_score: input.action.immediate_score,
            teacher_mean: 52.0 - f32::from(index),
            teacher_stddev: 1.25,
            action_hash: [index as u8; 32],
            input,
        }
    }

    #[test]
    fn action_features_capture_identity_and_exact_score_deltas() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(91),
        )
        .unwrap();
        let prelude = MarketPrelude {
            replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
            wildlife_wipes: Vec::new(),
        };
        let staged = game.preview_market_prelude(&prelude).unwrap();
        let coord = staged.boards()[0].frontier()[0];
        let mut action = TurnAction::paired(MarketSlot::ZERO, coord, Rotation::ZERO);
        action.replace_three_of_a_kind = prelude.replace_three_of_a_kind;
        let immediate_score = score_board(
            &game.preview_active_board(&action).unwrap(),
            game.config().scoring_cards,
        )
        .base_total;
        let features =
            ActionFeatures::from_game_action(&game, &action, 3, immediate_score).unwrap();

        assert_eq!(features.draft_kind, 0);
        assert_eq!(features.tile_slot, 0);
        assert_eq!(features.wildlife_slot, 0);
        assert_eq!(features.tile_q, coord.q);
        assert_eq!(features.tile_r, coord.r);
        assert_eq!(features.wildlife_present, 0);
        assert_eq!(features.immediate_rank, 3);
        assert_eq!(features.immediate_score, immediate_score);
        assert_eq!(features.immediate_deltas[10], 0);
    }

    #[test]
    fn action_ranking_record_round_trip_is_exact() {
        let expected = sample_record(1);
        let bytes = expected.to_bytes();
        assert_eq!(bytes.len(), ACTION_RANKING_RECORD_SIZE);
        assert_eq!(ActionRankingRecord::from_bytes(&bytes), expected);
    }

    #[test]
    fn action_record_group_validation_rejects_missing_candidates() {
        let mut invalid = sample_record(0);
        invalid.candidate_count = 3;
        assert!(matches!(
            validate_record_groups(&[invalid, sample_record(1)]),
            Err(DataError::InvalidConfig(_))
        ));
    }
}
