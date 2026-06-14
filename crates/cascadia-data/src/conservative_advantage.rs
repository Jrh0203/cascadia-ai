use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Seek, Write},
    path::{Path, PathBuf},
};

use cascadia_game::GameConfig;
use serde::{Deserialize, Serialize};

use super::{
    ACTION_POSITION_RECORD_SIZE, ActionPositionRecord, CollectionProvenance, DataError,
    DatasetSplit, RankingShardManifest, checksum_file, collection_provenance,
    collection_provenance_matches, read_array, unix_seconds, write_manifest_atomic, write_slice,
};

pub const CONSERVATIVE_ADVANTAGE_DATASET_SCHEMA_VERSION: u16 = 1;
pub const CONSERVATIVE_ADVANTAGE_FEATURE_SCHEMA: &str = "paired-observable-action-afterstates-v1";
pub const CONSERVATIVE_ADVANTAGE_TARGET_SCHEMA: &str = "paired-c90-lower-bound-v1";
pub const CONSERVATIVE_ADVANTAGE_SHARD_MAGIC: &[u8; 8] = b"CSD2CAV\0";
pub const CONSERVATIVE_ADVANTAGE_HEADER_SIZE: usize = 128;
pub const CONSERVATIVE_ADVANTAGE_RECORD_SIZE: usize = 92 + 2 * ACTION_POSITION_RECORD_SIZE;

#[derive(Debug, Clone)]
pub struct ConservativeAdvantageDatasetConfig {
    pub output: PathBuf,
    pub split: DatasetSplit,
    pub first_game_index: u64,
    pub games: usize,
    pub teacher: ConservativeAdvantageTeacherConfig,
    pub resume: bool,
}

impl ConservativeAdvantageDatasetConfig {
    fn validate(&self) -> Result<(), DataError> {
        if self.games == 0 {
            return Err(DataError::InvalidConfig("games must be positive"));
        }
        self.teacher.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConservativeAdvantageTeacherConfig {
    pub strategy_id: String,
    pub final_personal_turns: u16,
    pub determinizations: usize,
    pub immediate_candidates: usize,
    pub habitat_candidates: usize,
    pub bear_candidates: usize,
    pub future_market_draws: usize,
    pub confidence_percent: u8,
    pub anchor_strategy_id: String,
    pub continuation_strategy_id: String,
}

impl ConservativeAdvantageTeacherConfig {
    fn validate(&self) -> Result<(), DataError> {
        if self.strategy_id.trim().is_empty()
            || self.anchor_strategy_id.trim().is_empty()
            || self.continuation_strategy_id.trim().is_empty()
        {
            return Err(DataError::InvalidConfig(
                "conservative-advantage strategy IDs cannot be empty",
            ));
        }
        if !(1..=20).contains(&self.final_personal_turns)
            || self.determinizations != 8
            || self.immediate_candidates == 0
            || self.habitat_candidates == 0
            || self.bear_candidates == 0
            || self.future_market_draws == 0
            || self.confidence_percent != 90
        {
            return Err(DataError::InvalidConfig(
                "conservative-advantage teacher must use the frozen final-turn, R8, c90 configuration",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConservativeAdvantageDatasetManifest {
    pub schema_version: u16,
    pub dataset_id: String,
    pub feature_schema: String,
    pub target_schema: String,
    pub record_size: usize,
    pub action_position_record_size: usize,
    pub game: GameConfig,
    pub split: DatasetSplit,
    pub teacher: ConservativeAdvantageTeacherConfig,
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

#[derive(Debug, Clone, PartialEq)]
pub struct ConservativeAdvantageRecord {
    pub group_id: u64,
    pub candidate_index: u16,
    pub candidate_count: u16,
    pub selected: bool,
    pub mean_advantage: f32,
    pub advantage_standard_error: f32,
    pub lower_bound: f32,
    pub anchor_hash: [u8; 32],
    pub candidate_hash: [u8; 32],
    pub anchor: ActionPositionRecord,
    pub candidate: ActionPositionRecord,
}

impl ConservativeAdvantageRecord {
    pub fn to_bytes(&self) -> [u8; CONSERVATIVE_ADVANTAGE_RECORD_SIZE] {
        let mut bytes = [0; CONSERVATIVE_ADVANTAGE_RECORD_SIZE];
        let mut offset = 0;
        write_slice(&mut bytes, &mut offset, &self.group_id.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.candidate_index.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.candidate_count.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &[u8::from(self.selected), 0, 0, 0]);
        write_slice(&mut bytes, &mut offset, &self.mean_advantage.to_le_bytes());
        write_slice(
            &mut bytes,
            &mut offset,
            &self.advantage_standard_error.to_le_bytes(),
        );
        write_slice(&mut bytes, &mut offset, &self.lower_bound.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.anchor_hash);
        write_slice(&mut bytes, &mut offset, &self.candidate_hash);
        write_slice(&mut bytes, &mut offset, &self.anchor.to_bytes());
        write_slice(&mut bytes, &mut offset, &self.candidate.to_bytes());
        debug_assert_eq!(offset, CONSERVATIVE_ADVANTAGE_RECORD_SIZE);
        bytes
    }

    pub fn from_bytes(bytes: &[u8; CONSERVATIVE_ADVANTAGE_RECORD_SIZE]) -> Self {
        let mut offset = 0;
        let group_id = u64::from_le_bytes(read_array(bytes, &mut offset));
        let candidate_index = u16::from_le_bytes(read_array(bytes, &mut offset));
        let candidate_count = u16::from_le_bytes(read_array(bytes, &mut offset));
        let [selected, _, _, _] = read_array(bytes, &mut offset);
        let mean_advantage = f32::from_le_bytes(read_array(bytes, &mut offset));
        let advantage_standard_error = f32::from_le_bytes(read_array(bytes, &mut offset));
        let lower_bound = f32::from_le_bytes(read_array(bytes, &mut offset));
        let anchor_hash = read_array(bytes, &mut offset);
        let candidate_hash = read_array(bytes, &mut offset);
        let anchor_bytes: [u8; ACTION_POSITION_RECORD_SIZE] = read_array(bytes, &mut offset);
        let candidate_bytes: [u8; ACTION_POSITION_RECORD_SIZE] = read_array(bytes, &mut offset);
        debug_assert_eq!(offset, CONSERVATIVE_ADVANTAGE_RECORD_SIZE);
        Self {
            group_id,
            candidate_index,
            candidate_count,
            selected: selected != 0,
            mean_advantage,
            advantage_standard_error,
            lower_bound,
            anchor_hash,
            candidate_hash,
            anchor: ActionPositionRecord::from_bytes(&anchor_bytes),
            candidate: ActionPositionRecord::from_bytes(&candidate_bytes),
        }
    }
}

pub struct ConservativeAdvantageDatasetWriter {
    output: PathBuf,
    manifest_path: PathBuf,
    manifest: ConservativeAdvantageDatasetManifest,
}

impl ConservativeAdvantageDatasetWriter {
    pub fn open(config: &ConservativeAdvantageDatasetConfig) -> Result<Self, DataError> {
        config.validate()?;
        fs::create_dir_all(&config.output)?;
        let manifest_path = config.output.join("dataset.json");
        let mut manifest = if manifest_path.exists() {
            if !config.resume {
                return Err(DataError::DatasetExists(config.output.clone()));
            }
            let manifest: ConservativeAdvantageDatasetManifest =
                serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
            validate_resume(&manifest, config)?;
            validate_conservative_advantage_dataset(&config.output, &manifest)?;
            manifest
        } else {
            let now = unix_seconds()?;
            ConservativeAdvantageDatasetManifest {
                schema_version: CONSERVATIVE_ADVANTAGE_DATASET_SCHEMA_VERSION,
                dataset_id: format!(
                    "conservative-advantage-{}-{}-{}",
                    config.teacher.strategy_id,
                    config.split.id(),
                    config.first_game_index
                ),
                feature_schema: CONSERVATIVE_ADVANTAGE_FEATURE_SCHEMA.to_owned(),
                target_schema: CONSERVATIVE_ADVANTAGE_TARGET_SCHEMA.to_owned(),
                record_size: CONSERVATIVE_ADVANTAGE_RECORD_SIZE,
                action_position_record_size: ACTION_POSITION_RECORD_SIZE,
                game: GameConfig::research_aaaaa(4)?,
                split: config.split,
                teacher: config.teacher.clone(),
                first_game_index: config.first_game_index,
                requested_games: config.games,
                completed_games: 0,
                total_groups: 0,
                total_records: 0,
                created_unix_seconds: now,
                updated_unix_seconds: now,
                provenance: collection_provenance()?,
                shards: Vec::new(),
            }
        };
        manifest.requested_games = config.games;
        Ok(Self {
            output: config.output.clone(),
            manifest_path,
            manifest,
        })
    }

    pub fn manifest(&self) -> &ConservativeAdvantageDatasetManifest {
        &self.manifest
    }

    pub fn append_shard(
        &mut self,
        first_game_index: u64,
        game_count: usize,
        records: &[ConservativeAdvantageRecord],
    ) -> Result<(), DataError> {
        if game_count == 0 || records.is_empty() {
            return Err(DataError::InvalidConfig(
                "conservative-advantage shard must contain games and records",
            ));
        }
        let expected_first = self.manifest.first_game_index + self.manifest.completed_games as u64;
        if first_game_index != expected_first
            || self.manifest.completed_games + game_count > self.manifest.requested_games
        {
            return Err(DataError::InvalidConfig(
                "conservative-advantage shard game range is invalid",
            ));
        }
        let group_count = validate_record_groups(records)?;
        let shard_index = self.manifest.shards.len();
        let file_name = format!("shard-{shard_index:05}.cav");
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
        write_manifest_atomic(&self.manifest_path, &self.manifest)
    }
}

pub fn read_conservative_advantage_shard_records(
    root: &Path,
    split: DatasetSplit,
    shard: &RankingShardManifest,
) -> Result<Vec<ConservativeAdvantageRecord>, DataError> {
    let path = root.join(&shard.file);
    validate_shard_header(&path, split, shard)?;
    let mut reader = BufReader::new(File::open(path)?);
    reader.seek(std::io::SeekFrom::Start(
        CONSERVATIVE_ADVANTAGE_HEADER_SIZE as u64,
    ))?;
    let mut records = Vec::with_capacity(shard.record_count);
    for _ in 0..shard.record_count {
        let mut bytes = [0; CONSERVATIVE_ADVANTAGE_RECORD_SIZE];
        reader.read_exact(&mut bytes)?;
        records.push(ConservativeAdvantageRecord::from_bytes(&bytes));
    }
    validate_record_groups(&records)?;
    Ok(records)
}

pub fn validate_conservative_advantage_dataset(
    root: &Path,
    manifest: &ConservativeAdvantageDatasetManifest,
) -> Result<(), DataError> {
    if manifest.schema_version != CONSERVATIVE_ADVANTAGE_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != CONSERVATIVE_ADVANTAGE_FEATURE_SCHEMA
        || manifest.target_schema != CONSERVATIVE_ADVANTAGE_TARGET_SCHEMA
        || manifest.record_size != CONSERVATIVE_ADVANTAGE_RECORD_SIZE
        || manifest.action_position_record_size != ACTION_POSITION_RECORD_SIZE
        || manifest.game != GameConfig::research_aaaaa(4)?
    {
        return Err(DataError::InvalidManifest(
            "conservative-advantage schema identifiers do not match",
        ));
    }
    manifest.teacher.validate()?;
    let mut games = 0;
    let mut groups = 0;
    let mut records = 0;
    for shard in &manifest.shards {
        let path = root.join(&shard.file);
        if fs::metadata(&path)?.len() != shard.byte_count {
            return Err(DataError::InvalidManifest(
                "conservative-advantage shard byte count mismatch",
            ));
        }
        if checksum_file(&path)? != shard.blake3 {
            return Err(DataError::ChecksumMismatch(path));
        }
        validate_shard_header(&path, manifest.split, shard)?;
        let shard_records = read_conservative_advantage_shard_records(root, manifest.split, shard)?;
        if validate_record_groups(&shard_records)? != shard.group_count {
            return Err(DataError::InvalidManifest(
                "conservative-advantage shard group count mismatch",
            ));
        }
        games += shard.game_count;
        groups += shard.group_count;
        records += shard.record_count;
    }
    if games != manifest.completed_games
        || groups != manifest.total_groups
        || records != manifest.total_records
    {
        return Err(DataError::InvalidManifest(
            "conservative-advantage manifest totals do not match shards",
        ));
    }
    Ok(())
}

fn validate_resume(
    manifest: &ConservativeAdvantageDatasetManifest,
    config: &ConservativeAdvantageDatasetConfig,
) -> Result<(), DataError> {
    let current_provenance = collection_provenance()?;
    if manifest.schema_version != CONSERVATIVE_ADVANTAGE_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != CONSERVATIVE_ADVANTAGE_FEATURE_SCHEMA
        || manifest.target_schema != CONSERVATIVE_ADVANTAGE_TARGET_SCHEMA
        || manifest.record_size != CONSERVATIVE_ADVANTAGE_RECORD_SIZE
        || manifest.action_position_record_size != ACTION_POSITION_RECORD_SIZE
        || manifest.game != GameConfig::research_aaaaa(4)?
        || manifest.split != config.split
        || manifest.teacher != config.teacher
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

fn validate_record_groups(records: &[ConservativeAdvantageRecord]) -> Result<usize, DataError> {
    let mut groups: BTreeMap<u64, Vec<&ConservativeAdvantageRecord>> = BTreeMap::new();
    for record in records {
        if !record.mean_advantage.is_finite()
            || !record.advantage_standard_error.is_finite()
            || !record.lower_bound.is_finite()
            || record.advantage_standard_error < 0.0
            || record.anchor_hash == record.candidate_hash
        {
            return Err(DataError::InvalidConfig(
                "conservative-advantage record contains invalid target data",
            ));
        }
        groups.entry(record.group_id).or_default().push(record);
    }
    for group in groups.values_mut() {
        group.sort_unstable_by_key(|record| record.candidate_index);
        let count = group.len();
        if count == 0
            || count > usize::from(u16::MAX)
            || group.iter().enumerate().any(|(index, record)| {
                usize::from(record.candidate_count) != count
                    || usize::from(record.candidate_index) != index
                    || record.anchor_hash != group[0].anchor_hash
                    || record.anchor != group[0].anchor
            })
            || group.iter().filter(|record| record.selected).count() > 1
            || group
                .iter()
                .any(|record| record.selected && record.lower_bound <= 0.0)
        {
            return Err(DataError::InvalidConfig(
                "conservative-advantage candidate group is inconsistent",
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
    records: &[ConservativeAdvantageRecord],
) -> Result<(), DataError> {
    let temp_path = path.with_extension("cav.tmp");
    let mut writer = BufWriter::new(File::create(&temp_path)?);
    writer.write_all(CONSERVATIVE_ADVANTAGE_SHARD_MAGIC)?;
    writer.write_all(&CONSERVATIVE_ADVANTAGE_DATASET_SCHEMA_VERSION.to_le_bytes())?;
    writer.write_all(&(CONSERVATIVE_ADVANTAGE_HEADER_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(CONSERVATIVE_ADVANTAGE_RECORD_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(ACTION_POSITION_RECORD_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(records.len() as u32).to_le_bytes())?;
    writer.write_all(&(group_count as u32).to_le_bytes())?;
    writer.write_all(&(game_count as u32).to_le_bytes())?;
    writer.write_all(&[split.code(), 4, 0, 0])?;
    writer.write_all(&first_game_index.to_le_bytes())?;
    writer.write_all(blake3::hash(CONSERVATIVE_ADVANTAGE_FEATURE_SCHEMA.as_bytes()).as_bytes())?;
    writer.write_all(blake3::hash(CONSERVATIVE_ADVANTAGE_TARGET_SCHEMA.as_bytes()).as_bytes())?;
    writer.write_all(&[0; 24])?;
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
    let mut header = [0; CONSERVATIVE_ADVANTAGE_HEADER_SIZE];
    reader.read_exact(&mut header)?;
    if &header[..8] != CONSERVATIVE_ADVANTAGE_SHARD_MAGIC
        || u16::from_le_bytes([header[8], header[9]])
            != CONSERVATIVE_ADVANTAGE_DATASET_SCHEMA_VERSION
        || u16::from_le_bytes([header[10], header[11]]) as usize
            != CONSERVATIVE_ADVANTAGE_HEADER_SIZE
        || u16::from_le_bytes([header[12], header[13]]) as usize
            != CONSERVATIVE_ADVANTAGE_RECORD_SIZE
        || u16::from_le_bytes([header[14], header[15]]) as usize != ACTION_POSITION_RECORD_SIZE
        || header[28] != split.code()
        || &header[40..72]
            != blake3::hash(CONSERVATIVE_ADVANTAGE_FEATURE_SCHEMA.as_bytes()).as_bytes()
        || &header[72..104]
            != blake3::hash(CONSERVATIVE_ADVANTAGE_TARGET_SCHEMA.as_bytes()).as_bytes()
    {
        return Err(DataError::InvalidShard(
            "incompatible conservative-advantage header",
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
            "conservative-advantage header and manifest disagree",
        ));
    }
    let expected_size = CONSERVATIVE_ADVANTAGE_HEADER_SIZE as u64
        + record_count as u64 * CONSERVATIVE_ADVANTAGE_RECORD_SIZE as u64;
    if fs::metadata(path)?.len() != expected_size {
        return Err(DataError::InvalidShard(
            "conservative-advantage shard file size does not match records",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameSeed, GameState, MarketSlot, Rotation, TurnAction, score_board};

    use super::*;

    fn sample_input(game_index: u64, rank: u16) -> ActionPositionRecord {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(99),
        )
        .unwrap();
        let coord = game.boards()[0].frontier()[usize::from(rank - 1)];
        let action = TurnAction::paired(MarketSlot::ZERO, coord, Rotation::ZERO);
        let score = score_board(
            &game.preview_active_board(&action).unwrap(),
            game.config().scoring_cards,
        )
        .base_total;
        ActionPositionRecord::observe(&game, &action, game_index, rank, score).unwrap()
    }

    #[test]
    fn conservative_advantage_record_round_trip_is_exact() {
        let anchor = sample_input(7, 1);
        let candidate = sample_input(7, 2);
        let expected = ConservativeAdvantageRecord {
            group_id: 42,
            candidate_index: 0,
            candidate_count: 1,
            selected: true,
            mean_advantage: 1.25,
            advantage_standard_error: 0.5,
            lower_bound: 0.542_538,
            anchor_hash: [1; 32],
            candidate_hash: [2; 32],
            anchor,
            candidate,
        };
        let bytes = expected.to_bytes();
        assert_eq!(bytes.len(), CONSERVATIVE_ADVANTAGE_RECORD_SIZE);
        assert_eq!(ConservativeAdvantageRecord::from_bytes(&bytes), expected);
    }

    #[test]
    fn conservative_advantage_groups_reject_multiple_selected_challengers() {
        let anchor = sample_input(7, 1);
        let candidate = sample_input(7, 2);
        let make = |index| ConservativeAdvantageRecord {
            group_id: 42,
            candidate_index: index,
            candidate_count: 2,
            selected: true,
            mean_advantage: 1.0,
            advantage_standard_error: 0.1,
            lower_bound: 0.8,
            anchor_hash: [1; 32],
            candidate_hash: [index as u8 + 2; 32],
            anchor: anchor.clone(),
            candidate: candidate.clone(),
        };
        assert!(matches!(
            validate_record_groups(&[make(0), make(1)]),
            Err(DataError::InvalidConfig(_))
        ));
    }
}
