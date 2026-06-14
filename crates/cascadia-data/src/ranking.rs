use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use super::{
    CollectionProvenance, DataError, DatasetSplit, FEATURE_SCHEMA, PositionRecord, checksum_file,
    collection_provenance, collection_provenance_matches, unix_seconds, write_manifest_atomic,
    write_slice,
};

pub const RANKING_DATASET_SCHEMA_VERSION: u16 = 1;
pub const RANKING_TARGET_SCHEMA: &str = "search-ranking-v1";
pub const RANKING_SHARD_MAGIC: &[u8; 8] = b"CSD2RKG\0";
pub const RANKING_HEADER_SIZE: usize = 112;
pub const RANKING_RECORD_SIZE: usize = 920;

#[derive(Debug, Clone)]
pub struct RankingDatasetConfig {
    pub output: PathBuf,
    pub split: DatasetSplit,
    pub first_game_index: u64,
    pub games: usize,
    pub teacher: RankingTeacherConfig,
    pub trajectory: Option<RankingTrajectoryConfig>,
    pub resume: bool,
}

impl RankingDatasetConfig {
    fn validate(&self) -> Result<(), DataError> {
        if self.games == 0 {
            return Err(DataError::InvalidConfig("games must be positive"));
        }
        if self.teacher.strategy_id.is_empty() {
            return Err(DataError::InvalidConfig(
                "ranking teacher strategy id cannot be empty",
            ));
        }
        if self.teacher.immediate_candidates == 0
            || self.teacher.determinizations == 0
            || match self.teacher.candidate_family {
                RankingCandidateFamily::Bear => self.teacher.bear_candidates == 0,
                RankingCandidateFamily::Habitat => self.teacher.habitat_candidates == 0,
                RankingCandidateFamily::Pattern => {
                    self.teacher.bear_candidates == 0 || self.teacher.habitat_candidates == 0
                }
            }
        {
            return Err(DataError::InvalidConfig(
                "ranking teacher search limits must be positive",
            ));
        }
        if self
            .teacher
            .terminal_continuation_strategy_id
            .as_ref()
            .is_some_and(|strategy_id| strategy_id.trim().is_empty())
        {
            return Err(DataError::InvalidConfig(
                "terminal continuation strategy ID cannot be empty",
            ));
        }
        if self.teacher.terminal_continuation_strategy_id.is_some()
            != (self.teacher.greedy_plies == 0)
        {
            return Err(DataError::InvalidConfig(
                "terminal teachers require zero fixed plies; fixed-ply teachers require a positive horizon",
            ));
        }
        if self.trajectory.as_ref().is_some_and(|trajectory| {
            trajectory.strategy_id.trim().is_empty()
                || trajectory.model_manifest.trim().is_empty()
                || trajectory.model_manifest_blake3.trim().is_empty()
        }) {
            return Err(DataError::InvalidConfig(
                "ranking trajectory policy metadata cannot be empty",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankingTeacherConfig {
    pub strategy_id: String,
    pub immediate_candidates: usize,
    #[serde(default)]
    pub candidate_family: RankingCandidateFamily,
    #[serde(default)]
    pub bear_candidates: usize,
    #[serde(default)]
    pub habitat_candidates: usize,
    pub determinizations: usize,
    pub greedy_plies: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal_continuation_strategy_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankingTrajectoryConfig {
    pub strategy_id: String,
    pub model_manifest: String,
    pub model_manifest_blake3: String,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RankingCandidateFamily {
    #[default]
    Bear,
    Habitat,
    Pattern,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankingDatasetManifest {
    pub schema_version: u16,
    pub dataset_id: String,
    pub feature_schema: String,
    pub target_schema: String,
    pub record_size: usize,
    pub game: cascadia_game::GameConfig,
    pub split: DatasetSplit,
    pub teacher: RankingTeacherConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trajectory: Option<RankingTrajectoryConfig>,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankingShardManifest {
    pub file: String,
    pub first_game_index: u64,
    pub game_count: usize,
    pub group_count: usize,
    pub record_count: usize,
    pub byte_count: u64,
    pub blake3: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RankingRecord {
    pub group_id: u64,
    pub candidate_index: u16,
    pub candidate_count: u16,
    pub immediate_rank: u16,
    pub immediate_score: u16,
    pub teacher_mean: f32,
    pub teacher_stddev: f32,
    pub action_hash: [u8; 32],
    pub position: PositionRecord,
}

impl RankingRecord {
    pub fn to_bytes(&self) -> [u8; RANKING_RECORD_SIZE] {
        let mut bytes = [0; RANKING_RECORD_SIZE];
        let mut offset = 0;
        write_slice(&mut bytes, &mut offset, &self.group_id.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.candidate_index.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.candidate_count.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.immediate_rank.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.immediate_score.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.teacher_mean.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.teacher_stddev.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.action_hash);
        write_slice(&mut bytes, &mut offset, &self.position.to_bytes());
        debug_assert_eq!(offset, RANKING_RECORD_SIZE);
        bytes
    }

    fn from_bytes(bytes: &[u8; RANKING_RECORD_SIZE]) -> Self {
        let mut offset = 0;
        let group_id = u64::from_le_bytes(super::read_array(bytes, &mut offset));
        let candidate_index = u16::from_le_bytes(super::read_array(bytes, &mut offset));
        let candidate_count = u16::from_le_bytes(super::read_array(bytes, &mut offset));
        let immediate_rank = u16::from_le_bytes(super::read_array(bytes, &mut offset));
        let immediate_score = u16::from_le_bytes(super::read_array(bytes, &mut offset));
        let teacher_mean = f32::from_le_bytes(super::read_array(bytes, &mut offset));
        let teacher_stddev = f32::from_le_bytes(super::read_array(bytes, &mut offset));
        let action_hash = super::read_array(bytes, &mut offset);
        let position_bytes: [u8; super::RECORD_SIZE] = super::read_array(bytes, &mut offset);
        debug_assert_eq!(offset, RANKING_RECORD_SIZE);
        Self {
            group_id,
            candidate_index,
            candidate_count,
            immediate_rank,
            immediate_score,
            teacher_mean,
            teacher_stddev,
            action_hash,
            position: PositionRecord::from_bytes(&position_bytes),
        }
    }
}

pub fn read_ranking_shard_records(
    root: &Path,
    split: DatasetSplit,
    shard: &RankingShardManifest,
) -> Result<Vec<RankingRecord>, DataError> {
    let path = root.join(&shard.file);
    validate_ranking_shard_header(&path, split, shard)?;
    let mut reader = BufReader::new(File::open(path)?);
    reader.seek_relative(RANKING_HEADER_SIZE as i64)?;
    let mut records = Vec::with_capacity(shard.record_count);
    for _ in 0..shard.record_count {
        let mut bytes = [0; RANKING_RECORD_SIZE];
        reader.read_exact(&mut bytes)?;
        records.push(RankingRecord::from_bytes(&bytes));
    }
    validate_record_groups(&records)?;
    Ok(records)
}

pub struct RankingDatasetWriter {
    output: PathBuf,
    manifest_path: PathBuf,
    manifest: RankingDatasetManifest,
}

impl RankingDatasetWriter {
    pub fn open(config: &RankingDatasetConfig) -> Result<Self, DataError> {
        config.validate()?;
        fs::create_dir_all(&config.output)?;
        let manifest_path = config.output.join("dataset.json");
        let mut manifest = if manifest_path.exists() {
            if !config.resume {
                return Err(DataError::DatasetExists(config.output.clone()));
            }
            let manifest: RankingDatasetManifest =
                serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
            validate_resume(&manifest, config)?;
            validate_ranking_dataset(&config.output, &manifest)?;
            manifest
        } else {
            let now = unix_seconds()?;
            RankingDatasetManifest {
                schema_version: RANKING_DATASET_SCHEMA_VERSION,
                dataset_id: ranking_dataset_id(config),
                feature_schema: FEATURE_SCHEMA.to_owned(),
                target_schema: RANKING_TARGET_SCHEMA.to_owned(),
                record_size: RANKING_RECORD_SIZE,
                game: cascadia_game::GameConfig::research_aaaaa(4)?,
                split: config.split,
                teacher: config.teacher.clone(),
                trajectory: config.trajectory.clone(),
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

    pub fn manifest(&self) -> &RankingDatasetManifest {
        &self.manifest
    }

    pub fn append_shard(
        &mut self,
        first_game_index: u64,
        game_count: usize,
        records: &[RankingRecord],
    ) -> Result<(), DataError> {
        if game_count == 0 || records.is_empty() {
            return Err(DataError::InvalidConfig(
                "ranking shard must contain games and records",
            ));
        }
        let expected_first = self.manifest.first_game_index + self.manifest.completed_games as u64;
        if first_game_index != expected_first {
            return Err(DataError::InvalidConfig(
                "ranking shard game range is not contiguous",
            ));
        }
        if self.manifest.completed_games + game_count > self.manifest.requested_games {
            return Err(DataError::InvalidConfig(
                "ranking shard exceeds requested game count",
            ));
        }
        let group_count = validate_record_groups(records)?;
        let shard_index = self.manifest.shards.len();
        let file_name = format!("shard-{shard_index:05}.csr");
        let path = self.output.join(&file_name);
        write_ranking_shard(
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

pub fn validate_ranking_dataset(
    root: &Path,
    manifest: &RankingDatasetManifest,
) -> Result<(), DataError> {
    if manifest.schema_version != RANKING_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != FEATURE_SCHEMA
        || manifest.target_schema != RANKING_TARGET_SCHEMA
        || manifest.record_size != RANKING_RECORD_SIZE
        || manifest.game != cascadia_game::GameConfig::research_aaaaa(4)?
    {
        return Err(DataError::InvalidManifest(
            "ranking schema identifiers do not match",
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
                "ranking shard byte count mismatch",
            ));
        }
        if checksum_file(&path)? != shard.blake3 {
            return Err(DataError::ChecksumMismatch(path));
        }
        validate_ranking_shard_header(&path, manifest.split, shard)?;
        games += shard.game_count;
        groups += shard.group_count;
        records += shard.record_count;
    }
    if games != manifest.completed_games
        || groups != manifest.total_groups
        || records != manifest.total_records
    {
        return Err(DataError::InvalidManifest(
            "ranking manifest totals do not match shards",
        ));
    }
    Ok(())
}

fn validate_resume(
    manifest: &RankingDatasetManifest,
    config: &RankingDatasetConfig,
) -> Result<(), DataError> {
    let current_provenance = collection_provenance()?;
    if manifest.schema_version != RANKING_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != FEATURE_SCHEMA
        || manifest.target_schema != RANKING_TARGET_SCHEMA
        || manifest.record_size != RANKING_RECORD_SIZE
        || manifest.game != cascadia_game::GameConfig::research_aaaaa(4)?
        || manifest.split != config.split
        || manifest.teacher != config.teacher
        || manifest.trajectory != config.trajectory
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

fn ranking_dataset_id(config: &RankingDatasetConfig) -> String {
    match &config.trajectory {
        Some(trajectory) => format!(
            "ranking-{}-on-{}-{}-{}",
            config.teacher.strategy_id,
            trajectory.strategy_id,
            config.split.id(),
            config.first_game_index
        ),
        None => format!(
            "ranking-{}-{}-{}",
            config.teacher.strategy_id,
            config.split.id(),
            config.first_game_index
        ),
    }
}

fn validate_record_groups(records: &[RankingRecord]) -> Result<usize, DataError> {
    let mut groups: BTreeMap<u64, Vec<&RankingRecord>> = BTreeMap::new();
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
            })
        {
            return Err(DataError::InvalidConfig(
                "ranking candidate group is inconsistent",
            ));
        }
    }
    Ok(groups.len())
}

fn write_ranking_shard(
    path: &Path,
    split: DatasetSplit,
    first_game_index: u64,
    game_count: usize,
    group_count: usize,
    records: &[RankingRecord],
) -> Result<(), DataError> {
    let temp_path = path.with_extension("csr.tmp");
    let mut writer = BufWriter::new(File::create(&temp_path)?);
    writer.write_all(RANKING_SHARD_MAGIC)?;
    writer.write_all(&RANKING_DATASET_SCHEMA_VERSION.to_le_bytes())?;
    writer.write_all(&(RANKING_HEADER_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(RANKING_RECORD_SIZE as u16).to_le_bytes())?;
    writer.write_all(&0u16.to_le_bytes())?;
    writer.write_all(&(records.len() as u32).to_le_bytes())?;
    writer.write_all(&(group_count as u32).to_le_bytes())?;
    writer.write_all(&(game_count as u32).to_le_bytes())?;
    writer.write_all(&[split.code(), 4, 0, 0])?;
    writer.write_all(&first_game_index.to_le_bytes())?;
    writer.write_all(blake3::hash(FEATURE_SCHEMA.as_bytes()).as_bytes())?;
    writer.write_all(blake3::hash(RANKING_TARGET_SCHEMA.as_bytes()).as_bytes())?;
    writer.write_all(&[0; 8])?;
    for record in records {
        writer.write_all(&record.to_bytes())?;
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(temp_path, path)?;
    Ok(())
}

fn validate_ranking_shard_header(
    path: &Path,
    split: DatasetSplit,
    shard: &RankingShardManifest,
) -> Result<(), DataError> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut header = [0; RANKING_HEADER_SIZE];
    reader.read_exact(&mut header)?;
    if &header[..8] != RANKING_SHARD_MAGIC
        || u16::from_le_bytes([header[8], header[9]]) != RANKING_DATASET_SCHEMA_VERSION
        || u16::from_le_bytes([header[10], header[11]]) as usize != RANKING_HEADER_SIZE
        || u16::from_le_bytes([header[12], header[13]]) as usize != RANKING_RECORD_SIZE
        || header[28] != split.code()
        || &header[40..72] != blake3::hash(FEATURE_SCHEMA.as_bytes()).as_bytes()
        || &header[72..104] != blake3::hash(RANKING_TARGET_SCHEMA.as_bytes()).as_bytes()
    {
        return Err(DataError::InvalidShard("incompatible ranking header"));
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
            "ranking header and manifest disagree",
        ));
    }
    let expected_size =
        RANKING_HEADER_SIZE as u64 + record_count as u64 * RANKING_RECORD_SIZE as u64;
    if fs::metadata(path)?.len() != expected_size {
        return Err(DataError::InvalidShard(
            "ranking shard file size does not match records",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameConfig, GameSeed, GameState};

    use super::*;

    fn sample_record(group_id: u64, candidate_index: u16, candidate_count: u16) -> RankingRecord {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(5),
        )
        .unwrap();
        RankingRecord {
            group_id,
            candidate_index,
            candidate_count,
            immediate_rank: candidate_index + 1,
            immediate_score: 12 + candidate_index,
            teacher_mean: 20.5 + f32::from(candidate_index),
            teacher_stddev: 1.25,
            action_hash: [candidate_index as u8; 32],
            position: PositionRecord::observe(&game, 0),
        }
    }

    #[test]
    fn ranking_record_round_trip_is_exact() {
        let expected = sample_record(17, 1, 3);
        let bytes = expected.to_bytes();
        assert_eq!(bytes.len(), RANKING_RECORD_SIZE);
        assert_eq!(RankingRecord::from_bytes(&bytes), expected);
    }

    #[test]
    fn legacy_bear_teacher_manifest_uses_backward_compatible_defaults() {
        let teacher: RankingTeacherConfig = serde_json::from_value(serde_json::json!({
            "strategy_id": "bear-candidate-lookahead-v1-k8-b8-r4-d4",
            "immediate_candidates": 8,
            "bear_candidates": 8,
            "determinizations": 4,
            "greedy_plies": 4
        }))
        .unwrap();

        assert_eq!(teacher.candidate_family, RankingCandidateFamily::Bear);
        assert_eq!(teacher.bear_candidates, 8);
        assert_eq!(teacher.habitat_candidates, 0);
        assert_eq!(teacher.terminal_continuation_strategy_id, None);
    }

    #[test]
    fn terminal_pattern_teacher_requires_structured_continuation_metadata() {
        let config = RankingDatasetConfig {
            output: PathBuf::from("/tmp/unused-terminal-ranking-test"),
            split: DatasetSplit::Train,
            first_game_index: 0,
            games: 1,
            teacher: RankingTeacherConfig {
                strategy_id: "terminal-policy-improvement-v1-r8-k8-h6-b8-m4".to_owned(),
                immediate_candidates: 8,
                candidate_family: RankingCandidateFamily::Pattern,
                bear_candidates: 8,
                habitat_candidates: 6,
                determinizations: 8,
                greedy_plies: 0,
                terminal_continuation_strategy_id: Some("pattern-aware-v1-k8-h6-b8-m4".to_owned()),
            },
            trajectory: None,
            resume: false,
        };
        assert!(config.validate().is_ok());

        let mut fixed_plies = config.clone();
        fixed_plies.teacher.greedy_plies = 4;
        assert!(matches!(
            fixed_plies.validate(),
            Err(DataError::InvalidConfig(_))
        ));

        let mut missing_bear_frontier = config;
        missing_bear_frontier.teacher.bear_candidates = 0;
        assert!(matches!(
            missing_bear_frontier.validate(),
            Err(DataError::InvalidConfig(_))
        ));
    }

    #[test]
    fn ranking_dataset_writes_resumes_and_validates() {
        let root =
            std::env::temp_dir().join(format!("cascadia-ranking-data-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let config = RankingDatasetConfig {
            output: root.clone(),
            split: DatasetSplit::Train,
            first_game_index: 0,
            games: 2,
            teacher: RankingTeacherConfig {
                strategy_id: "teacher-v1".to_owned(),
                immediate_candidates: 2,
                candidate_family: RankingCandidateFamily::Bear,
                bear_candidates: 1,
                habitat_candidates: 0,
                determinizations: 2,
                greedy_plies: 1,
                terminal_continuation_strategy_id: None,
            },
            trajectory: None,
            resume: false,
        };
        let mut writer = RankingDatasetWriter::open(&config).unwrap();
        writer
            .append_shard(0, 1, &[sample_record(1, 0, 2), sample_record(1, 1, 2)])
            .unwrap();
        validate_ranking_dataset(&root, writer.manifest()).unwrap();
        drop(writer);

        let mut resumed = RankingDatasetWriter::open(&RankingDatasetConfig {
            resume: true,
            ..config
        })
        .unwrap();
        resumed
            .append_shard(1, 1, &[sample_record(2, 0, 1)])
            .unwrap();
        assert_eq!(resumed.manifest().completed_games, 2);
        assert_eq!(resumed.manifest().total_groups, 2);
        assert_eq!(resumed.manifest().total_records, 3);
        validate_ranking_dataset(&root, resumed.manifest()).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ranking_resume_rejects_changed_provenance() {
        let root = std::env::temp_dir().join(format!(
            "cascadia-ranking-provenance-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let config = RankingDatasetConfig {
            output: root.clone(),
            split: DatasetSplit::Train,
            first_game_index: 0,
            games: 2,
            teacher: RankingTeacherConfig {
                strategy_id: "teacher-v1".to_owned(),
                immediate_candidates: 2,
                candidate_family: RankingCandidateFamily::Bear,
                bear_candidates: 1,
                habitat_candidates: 0,
                determinizations: 2,
                greedy_plies: 1,
                terminal_continuation_strategy_id: None,
            },
            trajectory: None,
            resume: false,
        };
        let mut writer = RankingDatasetWriter::open(&config).unwrap();
        writer
            .append_shard(0, 1, &[sample_record(1, 0, 1)])
            .unwrap();
        drop(writer);
        let manifest_path = root.join("dataset.json");
        let mut manifest: RankingDatasetManifest =
            serde_json::from_reader(File::open(&manifest_path).unwrap()).unwrap();
        manifest.provenance.executable_blake3 = "different-executable".to_owned();
        write_manifest_atomic(&manifest_path, &manifest).unwrap();

        let error = RankingDatasetWriter::open(&RankingDatasetConfig {
            resume: true,
            ..config
        })
        .err()
        .expect("changed provenance must reject resume");
        assert!(matches!(error, DataError::ResumeMismatch));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ranking_resume_rejects_changed_trajectory_policy() {
        let root = std::env::temp_dir().join(format!(
            "cascadia-ranking-trajectory-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        let config = RankingDatasetConfig {
            output: root.clone(),
            split: DatasetSplit::Train,
            first_game_index: 8,
            games: 2,
            teacher: RankingTeacherConfig {
                strategy_id: "teacher-v1".to_owned(),
                immediate_candidates: 2,
                candidate_family: RankingCandidateFamily::Habitat,
                bear_candidates: 0,
                habitat_candidates: 1,
                determinizations: 1,
                greedy_plies: 1,
                terminal_continuation_strategy_id: None,
            },
            trajectory: Some(RankingTrajectoryConfig {
                strategy_id: "apprentice-v1".to_owned(),
                model_manifest: "/models/apprentice/model.json".to_owned(),
                model_manifest_blake3: "abc123".to_owned(),
            }),
            resume: false,
        };
        let mut writer = RankingDatasetWriter::open(&config).unwrap();
        writer
            .append_shard(8, 1, &[sample_record(1, 0, 1)])
            .unwrap();
        drop(writer);

        let error = RankingDatasetWriter::open(&RankingDatasetConfig {
            trajectory: Some(RankingTrajectoryConfig {
                strategy_id: "apprentice-v2".to_owned(),
                model_manifest: "/models/apprentice/model.json".to_owned(),
                model_manifest_blake3: "different".to_owned(),
            }),
            resume: true,
            ..config
        })
        .err()
        .expect("changed trajectory must reject resume");
        assert!(matches!(error, DataError::ResumeMismatch));
        fs::remove_dir_all(root).unwrap();
    }
}
