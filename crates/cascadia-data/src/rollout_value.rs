use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use super::{
    CollectionProvenance, DataError, DatasetSplit, ShardManifest, checksum_file,
    collection_provenance, collection_provenance_matches, unix_seconds, write_manifest_atomic,
};

pub const ROLLOUT_VALUE_DATASET_SCHEMA_VERSION: u16 = 1;
pub const ROLLOUT_VALUE_FEATURE_SCHEMA: &str = "legacy-mid-v4opp-sparse-u16-v1";
pub const ROLLOUT_VALUE_TARGET_SCHEMA: &str = "terminal-base-score-to-go-v1";
pub const ROLLOUT_VALUE_SHARD_MAGIC: &[u8; 8] = b"CSD2NNV\0";
pub const ROLLOUT_VALUE_HEADER_SIZE: usize = 160;
pub const ROLLOUT_VALUE_RECORD_PREFIX_SIZE: usize = 40;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RolloutValueRecordKind {
    Trajectory,
    RootEstimate,
}

impl RolloutValueRecordKind {
    const fn code(self) -> u8 {
        match self {
            Self::Trajectory => 0,
            Self::RootEstimate => 1,
        }
    }

    fn from_code(code: u8) -> Result<Self, DataError> {
        match code {
            0 => Ok(Self::Trajectory),
            1 => Ok(Self::RootEstimate),
            _ => Err(DataError::InvalidShard(
                "rollout-value record kind is invalid",
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RolloutValueRecord {
    pub kind: RolloutValueRecordKind,
    pub game_index: u64,
    pub decision_index: u8,
    pub personal_turn: u8,
    pub selected: bool,
    pub rollout_seed: u64,
    pub immediate_score: f32,
    pub target_remaining: f32,
    pub target_stddev: f32,
    pub samples: u32,
    pub features: Vec<u16>,
}

impl RolloutValueRecord {
    pub fn validate(&self, feature_count: usize) -> Result<(), DataError> {
        if self.decision_index >= 80
            || !(1..=20).contains(&self.personal_turn)
            || self.features.is_empty()
            || self.features.len() > u16::MAX as usize
            || self
                .features
                .iter()
                .any(|&feature| feature as usize >= feature_count)
            || !self.immediate_score.is_finite()
            || !self.target_remaining.is_finite()
            || !self.target_stddev.is_finite()
            || self.target_stddev < 0.0
            || self.samples == 0
        {
            return Err(DataError::InvalidConfig(
                "rollout-value record violates its schema",
            ));
        }
        match self.kind {
            RolloutValueRecordKind::Trajectory => {
                if self.target_stddev != 0.0 || self.samples != 1 || !self.selected {
                    return Err(DataError::InvalidConfig(
                        "trajectory rollout-value record has invalid target metadata",
                    ));
                }
            }
            RolloutValueRecordKind::RootEstimate => {
                if self.rollout_seed != 0 {
                    return Err(DataError::InvalidConfig(
                        "root rollout-value record cannot carry a rollout seed",
                    ));
                }
            }
        }
        Ok(())
    }

    fn write_to(&self, writer: &mut impl Write) -> Result<(), DataError> {
        writer.write_all(&[
            self.kind.code(),
            self.decision_index,
            self.personal_turn,
            u8::from(self.selected),
        ])?;
        writer.write_all(&(self.features.len() as u16).to_le_bytes())?;
        writer.write_all(&0u16.to_le_bytes())?;
        writer.write_all(&self.samples.to_le_bytes())?;
        writer.write_all(&self.game_index.to_le_bytes())?;
        writer.write_all(&self.rollout_seed.to_le_bytes())?;
        writer.write_all(&self.immediate_score.to_le_bytes())?;
        writer.write_all(&self.target_remaining.to_le_bytes())?;
        writer.write_all(&self.target_stddev.to_le_bytes())?;
        for feature in &self.features {
            writer.write_all(&feature.to_le_bytes())?;
        }
        Ok(())
    }

    fn read_from(reader: &mut impl Read) -> Result<Self, DataError> {
        let mut prefix = [0; ROLLOUT_VALUE_RECORD_PREFIX_SIZE];
        reader.read_exact(&mut prefix)?;
        let feature_len = u16::from_le_bytes([prefix[4], prefix[5]]) as usize;
        let mut features = vec![0; feature_len];
        for feature in &mut features {
            let mut bytes = [0; 2];
            reader.read_exact(&mut bytes)?;
            *feature = u16::from_le_bytes(bytes);
        }
        Ok(Self {
            kind: RolloutValueRecordKind::from_code(prefix[0])?,
            decision_index: prefix[1],
            personal_turn: prefix[2],
            selected: prefix[3] != 0,
            samples: u32::from_le_bytes(prefix[8..12].try_into().expect("fixed prefix")),
            game_index: u64::from_le_bytes(prefix[12..20].try_into().expect("fixed prefix")),
            rollout_seed: u64::from_le_bytes(prefix[20..28].try_into().expect("fixed prefix")),
            immediate_score: f32::from_le_bytes(prefix[28..32].try_into().expect("fixed prefix")),
            target_remaining: f32::from_le_bytes(prefix[32..36].try_into().expect("fixed prefix")),
            target_stddev: f32::from_le_bytes(prefix[36..40].try_into().expect("fixed prefix")),
            features,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RolloutValueTeacherConfig {
    pub strategy_id: String,
    pub parent_model_manifest_blake3: String,
    pub weights_blake3: String,
    pub feature_count: usize,
    pub candidate_limit: usize,
    pub rollouts: usize,
    pub trace_modulus: u64,
    pub lmr: bool,
    pub diverse_prefilter: bool,
}

impl RolloutValueTeacherConfig {
    fn validate(&self) -> Result<(), DataError> {
        if self.strategy_id.trim().is_empty()
            || self.parent_model_manifest_blake3.len() != 64
            || self.weights_blake3.len() != 64
            || self.feature_count == 0
            || self.feature_count > u16::MAX as usize
            || self.candidate_limit != 32
            || self.rollouts == 0
            || self.trace_modulus == 0
            || !self.lmr
            || !self.diverse_prefilter
        {
            return Err(DataError::InvalidConfig(
                "rollout-value teacher configuration is invalid",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct RolloutValueDatasetConfig {
    pub output: PathBuf,
    pub split: DatasetSplit,
    pub first_game_index: u64,
    pub games: usize,
    pub teacher: RolloutValueTeacherConfig,
    pub resume: bool,
}

impl RolloutValueDatasetConfig {
    fn validate(&self) -> Result<(), DataError> {
        if self.games == 0 {
            return Err(DataError::InvalidConfig("games must be positive"));
        }
        self.teacher.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RolloutValueDatasetManifest {
    pub schema_version: u16,
    pub dataset_id: String,
    pub feature_schema: String,
    pub target_schema: String,
    pub record_prefix_size: usize,
    pub split: DatasetSplit,
    pub teacher: RolloutValueTeacherConfig,
    pub first_game_index: u64,
    pub requested_games: usize,
    pub completed_games: usize,
    pub total_records: usize,
    pub trajectory_records: usize,
    pub root_estimate_records: usize,
    pub created_unix_seconds: u64,
    pub updated_unix_seconds: u64,
    pub provenance: CollectionProvenance,
    pub shards: Vec<ShardManifest>,
}

pub struct RolloutValueDatasetWriter {
    output: PathBuf,
    manifest_path: PathBuf,
    manifest: RolloutValueDatasetManifest,
}

impl RolloutValueDatasetWriter {
    pub fn open(config: &RolloutValueDatasetConfig) -> Result<Self, DataError> {
        config.validate()?;
        fs::create_dir_all(&config.output)?;
        let manifest_path = config.output.join("dataset.json");
        let mut manifest = if manifest_path.exists() {
            if !config.resume {
                return Err(DataError::DatasetExists(config.output.clone()));
            }
            let manifest: RolloutValueDatasetManifest =
                serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
            validate_resume(&manifest, config)?;
            validate_rollout_value_dataset(&config.output, &manifest)?;
            manifest
        } else {
            let now = unix_seconds()?;
            RolloutValueDatasetManifest {
                schema_version: ROLLOUT_VALUE_DATASET_SCHEMA_VERSION,
                dataset_id: format!(
                    "rollout-value-{}-{}-{}",
                    config.teacher.strategy_id,
                    config.split.id(),
                    config.first_game_index
                ),
                feature_schema: ROLLOUT_VALUE_FEATURE_SCHEMA.to_owned(),
                target_schema: ROLLOUT_VALUE_TARGET_SCHEMA.to_owned(),
                record_prefix_size: ROLLOUT_VALUE_RECORD_PREFIX_SIZE,
                split: config.split,
                teacher: config.teacher.clone(),
                first_game_index: config.first_game_index,
                requested_games: config.games,
                completed_games: 0,
                total_records: 0,
                trajectory_records: 0,
                root_estimate_records: 0,
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

    pub fn manifest(&self) -> &RolloutValueDatasetManifest {
        &self.manifest
    }

    pub fn append_game(
        &mut self,
        game_index: u64,
        records: &[RolloutValueRecord],
    ) -> Result<(), DataError> {
        if records.is_empty() {
            return Err(DataError::InvalidConfig(
                "rollout-value game shard cannot be empty",
            ));
        }
        let expected = self.manifest.first_game_index + self.manifest.completed_games as u64;
        if game_index != expected || self.manifest.completed_games >= self.manifest.requested_games
        {
            return Err(DataError::InvalidConfig(
                "rollout-value shard game range is invalid",
            ));
        }
        for record in records {
            if record.game_index != game_index {
                return Err(DataError::InvalidConfig(
                    "rollout-value shard contains another game index",
                ));
            }
            record.validate(self.manifest.teacher.feature_count)?;
        }
        validate_game_records(records, game_index, self.manifest.teacher.feature_count)?;
        let trajectory_records = records
            .iter()
            .filter(|record| record.kind == RolloutValueRecordKind::Trajectory)
            .count();
        let root_estimate_records = records.len() - trajectory_records;
        if trajectory_records == 0 || root_estimate_records == 0 {
            return Err(DataError::InvalidConfig(
                "rollout-value shard requires trajectory and root evidence",
            ));
        }
        let shard_index = self.manifest.shards.len();
        let file_name = format!("shard-{shard_index:05}.nnv");
        let path = self.output.join(&file_name);
        write_shard(
            &path,
            self.manifest.split,
            &self.manifest.teacher,
            game_index,
            records,
            trajectory_records,
            root_estimate_records,
        )?;
        let metadata = fs::metadata(&path)?;
        self.manifest.shards.push(ShardManifest {
            file: file_name,
            first_game_index: game_index,
            game_count: 1,
            record_count: records.len(),
            byte_count: metadata.len(),
            blake3: checksum_file(&path)?,
        });
        self.manifest.completed_games += 1;
        self.manifest.total_records += records.len();
        self.manifest.trajectory_records += trajectory_records;
        self.manifest.root_estimate_records += root_estimate_records;
        self.manifest.updated_unix_seconds = unix_seconds()?;
        write_manifest_atomic(&self.manifest_path, &self.manifest)
    }
}

pub fn validate_rollout_value_dataset(
    root: &Path,
    manifest: &RolloutValueDatasetManifest,
) -> Result<(), DataError> {
    if manifest.schema_version != ROLLOUT_VALUE_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != ROLLOUT_VALUE_FEATURE_SCHEMA
        || manifest.target_schema != ROLLOUT_VALUE_TARGET_SCHEMA
        || manifest.record_prefix_size != ROLLOUT_VALUE_RECORD_PREFIX_SIZE
    {
        return Err(DataError::InvalidManifest(
            "rollout-value schema identifiers do not match",
        ));
    }
    manifest.teacher.validate()?;
    if manifest.completed_games > manifest.requested_games
        || manifest.shards.len() != manifest.completed_games
    {
        return Err(DataError::InvalidManifest(
            "rollout-value manifest game totals are invalid",
        ));
    }
    let mut games = 0;
    let mut records = 0;
    let mut trajectory_records = 0;
    let mut root_estimate_records = 0;
    for (shard_index, shard) in manifest.shards.iter().enumerate() {
        if shard.first_game_index != manifest.first_game_index + shard_index as u64
            || shard.game_count != 1
        {
            return Err(DataError::InvalidManifest(
                "rollout-value shard game sequence is invalid",
            ));
        }
        let path = root.join(&shard.file);
        if fs::metadata(&path)?.len() != shard.byte_count {
            return Err(DataError::InvalidManifest(
                "rollout-value shard byte count mismatch",
            ));
        }
        if checksum_file(&path)? != shard.blake3 {
            return Err(DataError::ChecksumMismatch(path));
        }
        let shard_records = read_shard(&path, manifest.split, &manifest.teacher, shard)?;
        for record in &shard_records {
            record.validate(manifest.teacher.feature_count)?;
        }
        validate_game_records(
            &shard_records,
            shard.first_game_index,
            manifest.teacher.feature_count,
        )?;
        trajectory_records += shard_records
            .iter()
            .filter(|record| record.kind == RolloutValueRecordKind::Trajectory)
            .count();
        root_estimate_records += shard_records.len()
            - shard_records
                .iter()
                .filter(|record| record.kind == RolloutValueRecordKind::Trajectory)
                .count();
        games += shard.game_count;
        records += shard.record_count;
    }
    if games != manifest.completed_games
        || records != manifest.total_records
        || trajectory_records != manifest.trajectory_records
        || root_estimate_records != manifest.root_estimate_records
    {
        return Err(DataError::InvalidManifest(
            "rollout-value manifest totals do not match shards",
        ));
    }
    Ok(())
}

pub fn read_rollout_value_shard_records(
    root: &Path,
    manifest: &RolloutValueDatasetManifest,
    shard: &ShardManifest,
) -> Result<Vec<RolloutValueRecord>, DataError> {
    read_shard(
        &root.join(&shard.file),
        manifest.split,
        &manifest.teacher,
        shard,
    )
}

fn validate_resume(
    manifest: &RolloutValueDatasetManifest,
    config: &RolloutValueDatasetConfig,
) -> Result<(), DataError> {
    let current_provenance = collection_provenance()?;
    if manifest.schema_version != ROLLOUT_VALUE_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != ROLLOUT_VALUE_FEATURE_SCHEMA
        || manifest.target_schema != ROLLOUT_VALUE_TARGET_SCHEMA
        || manifest.record_prefix_size != ROLLOUT_VALUE_RECORD_PREFIX_SIZE
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

fn teacher_hash(teacher: &RolloutValueTeacherConfig) -> Result<[u8; 32], DataError> {
    let encoded = serde_json::to_vec(teacher)?;
    Ok(*blake3::hash(&encoded).as_bytes())
}

fn validate_game_records(
    records: &[RolloutValueRecord],
    game_index: u64,
    feature_count: usize,
) -> Result<(), DataError> {
    let mut root_counts = [0u16; 80];
    let mut selected_root_counts = [0u8; 80];
    for record in records {
        record.validate(feature_count)?;
        if record.game_index != game_index {
            return Err(DataError::InvalidShard(
                "rollout-value record game index does not match its shard",
            ));
        }
        let decision = record.decision_index as usize;
        let root_personal_turn = (decision / 4 + 1) as u8;
        match record.kind {
            RolloutValueRecordKind::Trajectory => {
                if record.personal_turn < root_personal_turn {
                    return Err(DataError::InvalidShard(
                        "trajectory record predates its root decision",
                    ));
                }
            }
            RolloutValueRecordKind::RootEstimate => {
                if record.personal_turn != root_personal_turn {
                    return Err(DataError::InvalidShard(
                        "root estimate personal turn does not match its decision",
                    ));
                }
                root_counts[decision] = root_counts[decision].saturating_add(1);
                selected_root_counts[decision] =
                    selected_root_counts[decision].saturating_add(u8::from(record.selected));
            }
        }
    }
    if root_counts.contains(&0) || selected_root_counts.iter().any(|&count| count != 1) {
        return Err(DataError::InvalidShard(
            "rollout-value game requires every root decision and one selected root each",
        ));
    }
    Ok(())
}

fn write_shard(
    path: &Path,
    split: DatasetSplit,
    teacher: &RolloutValueTeacherConfig,
    game_index: u64,
    records: &[RolloutValueRecord],
    trajectory_records: usize,
    root_estimate_records: usize,
) -> Result<(), DataError> {
    let temp_path = path.with_extension("nnv.tmp");
    let mut writer = BufWriter::new(File::create(&temp_path)?);
    writer.write_all(ROLLOUT_VALUE_SHARD_MAGIC)?;
    writer.write_all(&ROLLOUT_VALUE_DATASET_SCHEMA_VERSION.to_le_bytes())?;
    writer.write_all(&(ROLLOUT_VALUE_HEADER_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(ROLLOUT_VALUE_RECORD_PREFIX_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(teacher.feature_count as u16).to_le_bytes())?;
    writer.write_all(&(records.len() as u32).to_le_bytes())?;
    writer.write_all(&(trajectory_records as u32).to_le_bytes())?;
    writer.write_all(&(root_estimate_records as u32).to_le_bytes())?;
    writer.write_all(&1u32.to_le_bytes())?;
    writer.write_all(&[split.code(), 4, 0, 0, 0, 0, 0, 0])?;
    writer.write_all(&game_index.to_le_bytes())?;
    writer.write_all(blake3::hash(ROLLOUT_VALUE_FEATURE_SCHEMA.as_bytes()).as_bytes())?;
    writer.write_all(blake3::hash(ROLLOUT_VALUE_TARGET_SCHEMA.as_bytes()).as_bytes())?;
    writer.write_all(&teacher_hash(teacher)?)?;
    writer.write_all(&[0; 16])?;
    for record in records {
        record.write_to(&mut writer)?;
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(temp_path, path)?;
    Ok(())
}

fn read_shard(
    path: &Path,
    split: DatasetSplit,
    teacher: &RolloutValueTeacherConfig,
    shard: &ShardManifest,
) -> Result<Vec<RolloutValueRecord>, DataError> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut header = [0; ROLLOUT_VALUE_HEADER_SIZE];
    reader.read_exact(&mut header)?;
    if &header[..8] != ROLLOUT_VALUE_SHARD_MAGIC
        || u16::from_le_bytes([header[8], header[9]]) != ROLLOUT_VALUE_DATASET_SCHEMA_VERSION
        || u16::from_le_bytes([header[10], header[11]]) as usize != ROLLOUT_VALUE_HEADER_SIZE
        || u16::from_le_bytes([header[12], header[13]]) as usize != ROLLOUT_VALUE_RECORD_PREFIX_SIZE
        || u16::from_le_bytes([header[14], header[15]]) as usize != teacher.feature_count
        || header[32] != split.code()
        || &header[48..80] != blake3::hash(ROLLOUT_VALUE_FEATURE_SCHEMA.as_bytes()).as_bytes()
        || &header[80..112] != blake3::hash(ROLLOUT_VALUE_TARGET_SCHEMA.as_bytes()).as_bytes()
        || &header[112..144] != teacher_hash(teacher)?.as_slice()
    {
        return Err(DataError::InvalidShard("incompatible rollout-value header"));
    }
    let record_count =
        u32::from_le_bytes(header[16..20].try_into().expect("fixed header")) as usize;
    let trajectory_count =
        u32::from_le_bytes(header[20..24].try_into().expect("fixed header")) as usize;
    let root_count = u32::from_le_bytes(header[24..28].try_into().expect("fixed header")) as usize;
    let game_count = u32::from_le_bytes(header[28..32].try_into().expect("fixed header")) as usize;
    let game_index = u64::from_le_bytes(header[40..48].try_into().expect("fixed header"));
    if record_count != shard.record_count
        || trajectory_count + root_count != record_count
        || game_count != 1
        || shard.game_count != 1
        || game_index != shard.first_game_index
    {
        return Err(DataError::InvalidShard(
            "rollout-value header and manifest disagree",
        ));
    }
    let mut records = Vec::with_capacity(record_count);
    for _ in 0..record_count {
        records.push(RolloutValueRecord::read_from(&mut reader)?);
    }
    let mut trailing = [0; 1];
    if reader.read(&mut trailing)? != 0 {
        return Err(DataError::InvalidShard(
            "rollout-value shard has trailing bytes",
        ));
    }
    if records
        .iter()
        .filter(|record| record.kind == RolloutValueRecordKind::Trajectory)
        .count()
        != trajectory_count
    {
        return Err(DataError::InvalidShard(
            "rollout-value record-kind totals disagree",
        ));
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variable_record_round_trip_preserves_duplicate_features() {
        let record = RolloutValueRecord {
            kind: RolloutValueRecordKind::Trajectory,
            game_index: 94_000,
            decision_index: 17,
            personal_turn: 4,
            selected: true,
            rollout_seed: 72,
            immediate_score: 38.0,
            target_remaining: 57.5,
            target_stddev: 0.0,
            samples: 1,
            features: vec![1, 4, 4, 11_230],
        };
        record.validate(11_231).unwrap();
        let mut encoded = Vec::new();
        record.write_to(&mut encoded).unwrap();
        let decoded = RolloutValueRecord::read_from(&mut encoded.as_slice()).unwrap();
        assert_eq!(decoded, record);
    }

    fn complete_game_records(game_index: u64) -> Vec<RolloutValueRecord> {
        let mut records = Vec::new();
        records.push(RolloutValueRecord {
            kind: RolloutValueRecordKind::Trajectory,
            game_index,
            decision_index: 0,
            personal_turn: 1,
            selected: true,
            rollout_seed: 8,
            immediate_score: 12.0,
            target_remaining: 80.0,
            target_stddev: 0.0,
            samples: 1,
            features: vec![1, 4, 4, 11_230],
        });
        for decision_index in 0..80 {
            records.push(RolloutValueRecord {
                kind: RolloutValueRecordKind::RootEstimate,
                game_index,
                decision_index,
                personal_turn: decision_index / 4 + 1,
                selected: true,
                rollout_seed: 0,
                immediate_score: 12.0,
                target_remaining: 80.0,
                target_stddev: 2.0,
                samples: 4,
                features: vec![2, 2, 7],
            });
        }
        records
    }

    #[test]
    fn game_validation_requires_all_roots_and_one_selection() {
        let records = complete_game_records(94_000);
        validate_game_records(&records, 94_000, 11_231).unwrap();

        let mut missing = records.clone();
        missing.retain(|record| {
            record.kind != RolloutValueRecordKind::RootEstimate || record.decision_index != 79
        });
        assert!(validate_game_records(&missing, 94_000, 11_231).is_err());

        let mut duplicate_selected = records;
        duplicate_selected.push(RolloutValueRecord {
            kind: RolloutValueRecordKind::RootEstimate,
            game_index: 94_000,
            decision_index: 0,
            personal_turn: 1,
            selected: true,
            rollout_seed: 0,
            immediate_score: 12.0,
            target_remaining: 79.0,
            target_stddev: 2.0,
            samples: 4,
            features: vec![3],
        });
        assert!(validate_game_records(&duplicate_selected, 94_000, 11_231).is_err());
    }
}
