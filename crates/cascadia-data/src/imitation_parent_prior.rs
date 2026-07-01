use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Seek, Write},
    path::{Path, PathBuf},
};

use cascadia_game::GameConfig;
use serde::{Deserialize, Serialize};

use super::{
    CollectionProvenance, DataError, DatasetSplit, ImitationTargetsDatasetManifest,
    ImitationTeacherConfig, RankingShardManifest, checksum_file, collection_provenance,
    collection_provenance_matches, read_array, read_imitation_target_shard_records, unix_seconds,
    validate_imitation_targets_dataset, write_manifest_atomic, write_slice,
};

pub const IMITATION_PARENT_PRIOR_DATASET_SCHEMA_VERSION: u16 = 1;
pub const IMITATION_PARENT_PRIOR_FEATURE_SCHEMA: &str = "canonical-action-parent-prior-v1";
pub const IMITATION_PARENT_PRIOR_TARGET_SCHEMA: &str = "exact-mlx-afterstate-value-v1";
pub const IMITATION_PARENT_PRIOR_SHARD_MAGIC: &[u8; 8] = b"CSD2IMP\0";
pub const IMITATION_PARENT_PRIOR_HEADER_SIZE: usize = 112;
pub const IMITATION_PARENT_PRIOR_RECORD_SIZE: usize = 56;
pub const IMITATION_PARENT_HIDDEN_DATASET_SCHEMA_VERSION: u16 = 1;
pub const IMITATION_PARENT_HIDDEN_FEATURE_SCHEMA: &str = "canonical-action-parent-hidden-v1";
pub const IMITATION_PARENT_HIDDEN_TARGET_SCHEMA: &str = "exact-mlx-afterstate-hidden64-value-v1";
pub const IMITATION_PARENT_HIDDEN_SHARD_MAGIC: &[u8; 8] = b"CSD2IMH\0";
pub const IMITATION_PARENT_HIDDEN_HEADER_SIZE: usize = 112;
pub const IMITATION_PARENT_HIDDEN_DIM: usize = 64;
pub const IMITATION_PARENT_HIDDEN_RECORD_SIZE: usize = 312;

#[derive(Debug, Clone)]
pub struct ImitationParentPriorDatasetConfig {
    pub output: PathBuf,
    pub source_root: PathBuf,
    pub source_manifest: ImitationTargetsDatasetManifest,
    pub model_dir: PathBuf,
    pub resume: bool,
}

#[derive(Debug, Clone)]
pub struct ImitationParentHiddenDatasetConfig {
    pub output: PathBuf,
    pub source_root: PathBuf,
    pub source_manifest: ImitationTargetsDatasetManifest,
    pub model_dir: PathBuf,
    pub resume: bool,
}

impl ImitationParentHiddenDatasetConfig {
    fn validate(&self) -> Result<(), DataError> {
        validate_imitation_targets_dataset(&self.source_root, &self.source_manifest)?;
        if self.source_manifest.completed_games != self.source_manifest.requested_games
            || self.source_manifest.teacher_estimates
                != self.source_manifest.aligned_teacher_estimates
        {
            return Err(DataError::InvalidConfig(
                "parent-hidden source evidence must be complete and fully aligned",
            ));
        }
        model_identity(&self.model_dir)?;
        Ok(())
    }
}

impl ImitationParentPriorDatasetConfig {
    fn validate(&self) -> Result<(), DataError> {
        validate_imitation_targets_dataset(&self.source_root, &self.source_manifest)?;
        if self.source_manifest.completed_games != self.source_manifest.requested_games
            || self.source_manifest.teacher_estimates
                != self.source_manifest.aligned_teacher_estimates
        {
            return Err(DataError::InvalidConfig(
                "parent-prior source evidence must be complete and fully aligned",
            ));
        }
        model_identity(&self.model_dir)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImitationParentPriorSourceManifest {
    pub path: String,
    pub dataset_id: String,
    pub feature_schema: String,
    pub target_schema: String,
    pub action_dataset_id: String,
    pub first_game_index: u64,
    pub requested_games: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImitationParentPriorModelManifest {
    pub path: String,
    pub architecture: String,
    pub manifest_blake3: String,
    pub tensors_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImitationParentPriorDatasetManifest {
    pub schema_version: u16,
    pub dataset_id: String,
    pub feature_schema: String,
    pub target_schema: String,
    pub record_size: usize,
    pub game: GameConfig,
    pub split: DatasetSplit,
    pub teacher: ImitationTeacherConfig,
    pub source: ImitationParentPriorSourceManifest,
    pub model: ImitationParentPriorModelManifest,
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

pub type ImitationParentHiddenSourceManifest = ImitationParentPriorSourceManifest;
pub type ImitationParentHiddenModelManifest = ImitationParentPriorModelManifest;
pub type ImitationParentHiddenDatasetManifest = ImitationParentPriorDatasetManifest;

#[derive(Debug, Clone, PartialEq)]
pub struct ImitationParentPriorRecord {
    pub group_id: u64,
    pub candidate_index: u16,
    pub candidate_count: u16,
    pub action_hash: [u8; 32],
    pub parent_immediate: f32,
    pub parent_remaining: f32,
}

impl ImitationParentPriorRecord {
    pub fn parent_total(&self) -> f32 {
        self.parent_immediate + self.parent_remaining
    }

    pub fn to_bytes(&self) -> [u8; IMITATION_PARENT_PRIOR_RECORD_SIZE] {
        let mut bytes = [0; IMITATION_PARENT_PRIOR_RECORD_SIZE];
        let mut offset = 0;
        write_slice(&mut bytes, &mut offset, &self.group_id.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.candidate_index.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.candidate_count.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.action_hash);
        write_slice(
            &mut bytes,
            &mut offset,
            &self.parent_immediate.to_le_bytes(),
        );
        write_slice(
            &mut bytes,
            &mut offset,
            &self.parent_remaining.to_le_bytes(),
        );
        offset += 4;
        debug_assert_eq!(offset, IMITATION_PARENT_PRIOR_RECORD_SIZE);
        bytes
    }

    pub fn from_bytes(bytes: &[u8; IMITATION_PARENT_PRIOR_RECORD_SIZE]) -> Self {
        let mut offset = 0;
        let group_id = u64::from_le_bytes(read_array(bytes, &mut offset));
        let candidate_index = u16::from_le_bytes(read_array(bytes, &mut offset));
        let candidate_count = u16::from_le_bytes(read_array(bytes, &mut offset));
        let action_hash = read_array(bytes, &mut offset);
        let parent_immediate = f32::from_le_bytes(read_array(bytes, &mut offset));
        let parent_remaining = f32::from_le_bytes(read_array(bytes, &mut offset));
        offset += 4;
        debug_assert_eq!(offset, IMITATION_PARENT_PRIOR_RECORD_SIZE);
        Self {
            group_id,
            candidate_index,
            candidate_count,
            action_hash,
            parent_immediate,
            parent_remaining,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImitationParentHiddenRecord {
    pub group_id: u64,
    pub candidate_index: u16,
    pub candidate_count: u16,
    pub action_hash: [u8; 32],
    pub parent_immediate: f32,
    pub parent_remaining: f32,
    pub parent_hidden: [f32; IMITATION_PARENT_HIDDEN_DIM],
}

impl ImitationParentHiddenRecord {
    pub fn parent_total(&self) -> f32 {
        self.parent_immediate + self.parent_remaining
    }

    pub fn to_bytes(&self) -> [u8; IMITATION_PARENT_HIDDEN_RECORD_SIZE] {
        let mut bytes = [0; IMITATION_PARENT_HIDDEN_RECORD_SIZE];
        let mut offset = 0;
        write_slice(&mut bytes, &mut offset, &self.group_id.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.candidate_index.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.candidate_count.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.action_hash);
        write_slice(
            &mut bytes,
            &mut offset,
            &self.parent_immediate.to_le_bytes(),
        );
        write_slice(
            &mut bytes,
            &mut offset,
            &self.parent_remaining.to_le_bytes(),
        );
        for value in self.parent_hidden {
            write_slice(&mut bytes, &mut offset, &value.to_le_bytes());
        }
        offset += 4;
        debug_assert_eq!(offset, IMITATION_PARENT_HIDDEN_RECORD_SIZE);
        bytes
    }

    pub fn from_bytes(bytes: &[u8; IMITATION_PARENT_HIDDEN_RECORD_SIZE]) -> Self {
        let mut offset = 0;
        let group_id = u64::from_le_bytes(read_array(bytes, &mut offset));
        let candidate_index = u16::from_le_bytes(read_array(bytes, &mut offset));
        let candidate_count = u16::from_le_bytes(read_array(bytes, &mut offset));
        let action_hash = read_array(bytes, &mut offset);
        let parent_immediate = f32::from_le_bytes(read_array(bytes, &mut offset));
        let parent_remaining = f32::from_le_bytes(read_array(bytes, &mut offset));
        let mut parent_hidden = [0.0; IMITATION_PARENT_HIDDEN_DIM];
        for value in &mut parent_hidden {
            *value = f32::from_le_bytes(read_array(bytes, &mut offset));
        }
        offset += 4;
        debug_assert_eq!(offset, IMITATION_PARENT_HIDDEN_RECORD_SIZE);
        Self {
            group_id,
            candidate_index,
            candidate_count,
            action_hash,
            parent_immediate,
            parent_remaining,
            parent_hidden,
        }
    }
}

pub struct ImitationParentPriorDatasetWriter {
    output: PathBuf,
    manifest_path: PathBuf,
    manifest: ImitationParentPriorDatasetManifest,
}

pub struct ImitationParentHiddenDatasetWriter {
    output: PathBuf,
    manifest_path: PathBuf,
    manifest: ImitationParentHiddenDatasetManifest,
}

impl ImitationParentHiddenDatasetWriter {
    pub fn open(config: &ImitationParentHiddenDatasetConfig) -> Result<Self, DataError> {
        config.validate()?;
        fs::create_dir_all(&config.output)?;
        let source = source_identity(&config.source_root, &config.source_manifest)?;
        let model = model_identity(&config.model_dir)?;
        let manifest_path = config.output.join("dataset.json");
        let manifest = if manifest_path.exists() {
            if !config.resume {
                return Err(DataError::DatasetExists(config.output.clone()));
            }
            let manifest: ImitationParentHiddenDatasetManifest =
                serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
            validate_hidden_resume(&manifest, config, &source, &model)?;
            validate_imitation_parent_hidden_dataset(&config.output, &manifest)?;
            manifest
        } else {
            let now = unix_seconds()?;
            ImitationParentHiddenDatasetManifest {
                schema_version: IMITATION_PARENT_HIDDEN_DATASET_SCHEMA_VERSION,
                dataset_id: hidden_dataset_id(&source, &model),
                feature_schema: IMITATION_PARENT_HIDDEN_FEATURE_SCHEMA.to_owned(),
                target_schema: IMITATION_PARENT_HIDDEN_TARGET_SCHEMA.to_owned(),
                record_size: IMITATION_PARENT_HIDDEN_RECORD_SIZE,
                game: config.source_manifest.game,
                split: config.source_manifest.split,
                teacher: config.source_manifest.teacher.clone(),
                source,
                model,
                first_game_index: config.source_manifest.first_game_index,
                requested_games: config.source_manifest.requested_games,
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

    pub fn manifest(&self) -> &ImitationParentHiddenDatasetManifest {
        &self.manifest
    }

    pub fn root(&self) -> &Path {
        &self.output
    }

    pub fn append_shard(
        &mut self,
        first_game_index: u64,
        game_count: usize,
        records: &[ImitationParentHiddenRecord],
    ) -> Result<(), DataError> {
        if game_count == 0 || records.is_empty() {
            return Err(DataError::InvalidConfig(
                "parent-hidden shard must contain games and records",
            ));
        }
        let expected_first = self.manifest.first_game_index + self.manifest.completed_games as u64;
        if first_game_index != expected_first
            || self.manifest.completed_games + game_count > self.manifest.requested_games
        {
            return Err(DataError::InvalidConfig(
                "parent-hidden shard game range is invalid",
            ));
        }
        let group_count = validate_hidden_record_groups(records)?;
        let shard_index = self.manifest.shards.len();
        let file_name = format!("shard-{shard_index:05}.imh");
        let path = self.output.join(&file_name);
        write_hidden_shard(
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

impl ImitationParentPriorDatasetWriter {
    pub fn open(config: &ImitationParentPriorDatasetConfig) -> Result<Self, DataError> {
        config.validate()?;
        fs::create_dir_all(&config.output)?;
        let source = source_identity(&config.source_root, &config.source_manifest)?;
        let model = model_identity(&config.model_dir)?;
        let manifest_path = config.output.join("dataset.json");
        let manifest = if manifest_path.exists() {
            if !config.resume {
                return Err(DataError::DatasetExists(config.output.clone()));
            }
            let manifest: ImitationParentPriorDatasetManifest =
                serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
            validate_resume(&manifest, config, &source, &model)?;
            validate_imitation_parent_prior_dataset(&config.output, &manifest)?;
            manifest
        } else {
            let now = unix_seconds()?;
            ImitationParentPriorDatasetManifest {
                schema_version: IMITATION_PARENT_PRIOR_DATASET_SCHEMA_VERSION,
                dataset_id: dataset_id(&source, &model),
                feature_schema: IMITATION_PARENT_PRIOR_FEATURE_SCHEMA.to_owned(),
                target_schema: IMITATION_PARENT_PRIOR_TARGET_SCHEMA.to_owned(),
                record_size: IMITATION_PARENT_PRIOR_RECORD_SIZE,
                game: config.source_manifest.game,
                split: config.source_manifest.split,
                teacher: config.source_manifest.teacher.clone(),
                source,
                model,
                first_game_index: config.source_manifest.first_game_index,
                requested_games: config.source_manifest.requested_games,
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

    pub fn manifest(&self) -> &ImitationParentPriorDatasetManifest {
        &self.manifest
    }

    pub fn root(&self) -> &Path {
        &self.output
    }

    pub fn append_shard(
        &mut self,
        first_game_index: u64,
        game_count: usize,
        records: &[ImitationParentPriorRecord],
    ) -> Result<(), DataError> {
        if game_count == 0 || records.is_empty() {
            return Err(DataError::InvalidConfig(
                "parent-prior shard must contain games and records",
            ));
        }
        let expected_first = self.manifest.first_game_index + self.manifest.completed_games as u64;
        if first_game_index != expected_first
            || self.manifest.completed_games + game_count > self.manifest.requested_games
        {
            return Err(DataError::InvalidConfig(
                "parent-prior shard game range is invalid",
            ));
        }
        let group_count = validate_record_groups(records)?;
        let shard_index = self.manifest.shards.len();
        let file_name = format!("shard-{shard_index:05}.imp");
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

pub fn read_imitation_parent_prior_shard_records(
    root: &Path,
    split: DatasetSplit,
    shard: &RankingShardManifest,
) -> Result<Vec<ImitationParentPriorRecord>, DataError> {
    let path = root.join(&shard.file);
    validate_shard_header(&path, split, shard)?;
    let mut reader = BufReader::new(File::open(path)?);
    reader.seek(std::io::SeekFrom::Start(
        IMITATION_PARENT_PRIOR_HEADER_SIZE as u64,
    ))?;
    let mut records = Vec::with_capacity(shard.record_count);
    for _ in 0..shard.record_count {
        let mut bytes = [0; IMITATION_PARENT_PRIOR_RECORD_SIZE];
        reader.read_exact(&mut bytes)?;
        records.push(ImitationParentPriorRecord::from_bytes(&bytes));
    }
    validate_record_groups(&records)?;
    Ok(records)
}

pub fn read_imitation_parent_hidden_shard_records(
    root: &Path,
    split: DatasetSplit,
    shard: &RankingShardManifest,
) -> Result<Vec<ImitationParentHiddenRecord>, DataError> {
    let path = root.join(&shard.file);
    validate_hidden_shard_header(&path, split, shard)?;
    let mut reader = BufReader::new(File::open(path)?);
    reader.seek(std::io::SeekFrom::Start(
        IMITATION_PARENT_HIDDEN_HEADER_SIZE as u64,
    ))?;
    let mut records = Vec::with_capacity(shard.record_count);
    for _ in 0..shard.record_count {
        let mut bytes = [0; IMITATION_PARENT_HIDDEN_RECORD_SIZE];
        reader.read_exact(&mut bytes)?;
        records.push(ImitationParentHiddenRecord::from_bytes(&bytes));
    }
    validate_hidden_record_groups(&records)?;
    Ok(records)
}

pub fn validate_imitation_parent_prior_dataset(
    root: &Path,
    manifest: &ImitationParentPriorDatasetManifest,
) -> Result<(), DataError> {
    if manifest.schema_version != IMITATION_PARENT_PRIOR_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != IMITATION_PARENT_PRIOR_FEATURE_SCHEMA
        || manifest.target_schema != IMITATION_PARENT_PRIOR_TARGET_SCHEMA
        || manifest.record_size != IMITATION_PARENT_PRIOR_RECORD_SIZE
        || manifest.game != GameConfig::research_aaaaa(4)?
    {
        return Err(DataError::InvalidManifest(
            "parent-prior schema identifiers do not match",
        ));
    }
    let source_root = Path::new(&manifest.source.path);
    let source_manifest: ImitationTargetsDatasetManifest = serde_json::from_reader(
        BufReader::new(File::open(source_root.join("dataset.json"))?),
    )?;
    validate_imitation_targets_dataset(source_root, &source_manifest)?;
    if source_identity(source_root, &source_manifest)? != manifest.source
        || model_identity(Path::new(&manifest.model.path))? != manifest.model
        || source_manifest.split != manifest.split
        || source_manifest.teacher != manifest.teacher
    {
        return Err(DataError::InvalidManifest(
            "parent-prior source or model identity does not match",
        ));
    }

    let mut games = 0;
    let mut groups = 0;
    let mut records = 0;
    if manifest.shards.len() > source_manifest.shards.len() {
        return Err(DataError::InvalidManifest(
            "parent-prior has more shards than its source",
        ));
    }
    for (index, shard) in manifest.shards.iter().enumerate() {
        let path = root.join(&shard.file);
        if fs::metadata(&path)?.len() != shard.byte_count {
            return Err(DataError::InvalidManifest(
                "parent-prior shard byte count mismatch",
            ));
        }
        if checksum_file(&path)? != shard.blake3 {
            return Err(DataError::ChecksumMismatch(path));
        }
        let prior_records = read_imitation_parent_prior_shard_records(root, manifest.split, shard)?;
        let source_shard = &source_manifest.shards[index];
        if source_shard.first_game_index != shard.first_game_index
            || source_shard.game_count != shard.game_count
            || source_shard.group_count != shard.group_count
            || source_shard.record_count != shard.record_count
        {
            return Err(DataError::InvalidManifest(
                "parent-prior shard range does not match source shard",
            ));
        }
        let source_records =
            read_imitation_target_shard_records(source_root, manifest.split, source_shard)?;
        if source_records.len() != prior_records.len()
            || source_records
                .iter()
                .zip(&prior_records)
                .any(|(source, prior)| {
                    source.group_id != prior.group_id
                        || source.candidate_index != prior.candidate_index
                        || source.candidate_count != prior.candidate_count
                        || source.action_hash != prior.action_hash
                })
        {
            return Err(DataError::InvalidManifest(
                "parent-prior records are not aligned with source evidence",
            ));
        }
        if validate_record_groups(&prior_records)? != shard.group_count {
            return Err(DataError::InvalidManifest(
                "parent-prior shard group count mismatch",
            ));
        }
        games += shard.game_count;
        groups += shard.group_count;
        records += shard.record_count;
    }
    if games != manifest.completed_games
        || groups != manifest.total_groups
        || records != manifest.total_records
        || manifest.completed_games > source_manifest.completed_games
        || manifest.completed_games > manifest.requested_games
    {
        return Err(DataError::InvalidManifest(
            "parent-prior manifest totals do not match shards",
        ));
    }
    Ok(())
}

pub fn validate_imitation_parent_hidden_dataset(
    root: &Path,
    manifest: &ImitationParentHiddenDatasetManifest,
) -> Result<(), DataError> {
    if manifest.schema_version != IMITATION_PARENT_HIDDEN_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != IMITATION_PARENT_HIDDEN_FEATURE_SCHEMA
        || manifest.target_schema != IMITATION_PARENT_HIDDEN_TARGET_SCHEMA
        || manifest.record_size != IMITATION_PARENT_HIDDEN_RECORD_SIZE
        || manifest.game != GameConfig::research_aaaaa(4)?
    {
        return Err(DataError::InvalidManifest(
            "parent-hidden schema identifiers do not match",
        ));
    }
    let source_root = Path::new(&manifest.source.path);
    let source_manifest: ImitationTargetsDatasetManifest = serde_json::from_reader(
        BufReader::new(File::open(source_root.join("dataset.json"))?),
    )?;
    validate_imitation_targets_dataset(source_root, &source_manifest)?;
    if source_identity(source_root, &source_manifest)? != manifest.source
        || model_identity(Path::new(&manifest.model.path))? != manifest.model
        || source_manifest.split != manifest.split
        || source_manifest.teacher != manifest.teacher
    {
        return Err(DataError::InvalidManifest(
            "parent-hidden source or model identity does not match",
        ));
    }

    let mut games = 0;
    let mut groups = 0;
    let mut records = 0;
    if manifest.shards.len() > source_manifest.shards.len() {
        return Err(DataError::InvalidManifest(
            "parent-hidden has more shards than its source",
        ));
    }
    for (index, shard) in manifest.shards.iter().enumerate() {
        let path = root.join(&shard.file);
        if fs::metadata(&path)?.len() != shard.byte_count {
            return Err(DataError::InvalidManifest(
                "parent-hidden shard byte count mismatch",
            ));
        }
        if checksum_file(&path)? != shard.blake3 {
            return Err(DataError::ChecksumMismatch(path));
        }
        let hidden_records =
            read_imitation_parent_hidden_shard_records(root, manifest.split, shard)?;
        let source_shard = &source_manifest.shards[index];
        if source_shard.first_game_index != shard.first_game_index
            || source_shard.game_count != shard.game_count
            || source_shard.group_count != shard.group_count
            || source_shard.record_count != shard.record_count
        {
            return Err(DataError::InvalidManifest(
                "parent-hidden shard range does not match source shard",
            ));
        }
        let source_records =
            read_imitation_target_shard_records(source_root, manifest.split, source_shard)?;
        if source_records.len() != hidden_records.len()
            || source_records
                .iter()
                .zip(&hidden_records)
                .any(|(source, hidden)| {
                    source.group_id != hidden.group_id
                        || source.candidate_index != hidden.candidate_index
                        || source.candidate_count != hidden.candidate_count
                        || source.action_hash != hidden.action_hash
                })
        {
            return Err(DataError::InvalidManifest(
                "parent-hidden records are not aligned with source evidence",
            ));
        }
        if validate_hidden_record_groups(&hidden_records)? != shard.group_count {
            return Err(DataError::InvalidManifest(
                "parent-hidden shard group count mismatch",
            ));
        }
        games += shard.game_count;
        groups += shard.group_count;
        records += shard.record_count;
    }
    if games != manifest.completed_games
        || groups != manifest.total_groups
        || records != manifest.total_records
        || manifest.completed_games > source_manifest.completed_games
        || manifest.completed_games > manifest.requested_games
    {
        return Err(DataError::InvalidManifest(
            "parent-hidden manifest totals do not match shards",
        ));
    }
    Ok(())
}

fn validate_resume(
    manifest: &ImitationParentPriorDatasetManifest,
    config: &ImitationParentPriorDatasetConfig,
    source: &ImitationParentPriorSourceManifest,
    model: &ImitationParentPriorModelManifest,
) -> Result<(), DataError> {
    let current_provenance = collection_provenance()?;
    if manifest.schema_version != IMITATION_PARENT_PRIOR_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != IMITATION_PARENT_PRIOR_FEATURE_SCHEMA
        || manifest.target_schema != IMITATION_PARENT_PRIOR_TARGET_SCHEMA
        || manifest.record_size != IMITATION_PARENT_PRIOR_RECORD_SIZE
        || manifest.game != config.source_manifest.game
        || manifest.split != config.source_manifest.split
        || manifest.teacher != config.source_manifest.teacher
        || manifest.source != *source
        || manifest.model != *model
        || manifest.first_game_index != config.source_manifest.first_game_index
        || manifest.requested_games != config.source_manifest.requested_games
        || !collection_provenance_matches(&manifest.provenance, &current_provenance)
    {
        return Err(DataError::ResumeMismatch);
    }
    Ok(())
}

fn validate_hidden_resume(
    manifest: &ImitationParentHiddenDatasetManifest,
    config: &ImitationParentHiddenDatasetConfig,
    source: &ImitationParentHiddenSourceManifest,
    model: &ImitationParentHiddenModelManifest,
) -> Result<(), DataError> {
    let current_provenance = collection_provenance()?;
    if manifest.schema_version != IMITATION_PARENT_HIDDEN_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != IMITATION_PARENT_HIDDEN_FEATURE_SCHEMA
        || manifest.target_schema != IMITATION_PARENT_HIDDEN_TARGET_SCHEMA
        || manifest.record_size != IMITATION_PARENT_HIDDEN_RECORD_SIZE
        || manifest.game != config.source_manifest.game
        || manifest.split != config.source_manifest.split
        || manifest.teacher != config.source_manifest.teacher
        || manifest.source != *source
        || manifest.model != *model
        || manifest.first_game_index != config.source_manifest.first_game_index
        || manifest.requested_games != config.source_manifest.requested_games
        || !collection_provenance_matches(&manifest.provenance, &current_provenance)
    {
        return Err(DataError::ResumeMismatch);
    }
    Ok(())
}

fn validate_record_groups(records: &[ImitationParentPriorRecord]) -> Result<usize, DataError> {
    let mut groups: BTreeMap<u64, Vec<&ImitationParentPriorRecord>> = BTreeMap::new();
    for record in records {
        if !record.parent_immediate.is_finite()
            || !record.parent_remaining.is_finite()
            || !record.parent_total().is_finite()
        {
            return Err(DataError::InvalidConfig(
                "parent-prior record contains a non-finite value",
            ));
        }
        groups.entry(record.group_id).or_default().push(record);
    }
    for group in groups.values_mut() {
        group.sort_unstable_by_key(|record| record.candidate_index);
        let count = group.len();
        let hashes = group
            .iter()
            .map(|record| record.action_hash)
            .collect::<BTreeSet<_>>();
        if count < 2
            || count > usize::from(u16::MAX)
            || hashes.len() != count
            || group.iter().enumerate().any(|(index, record)| {
                usize::from(record.candidate_count) != count
                    || usize::from(record.candidate_index) != index
            })
        {
            return Err(DataError::InvalidConfig(
                "parent-prior candidate group is inconsistent",
            ));
        }
    }
    Ok(groups.len())
}

fn validate_hidden_record_groups(
    records: &[ImitationParentHiddenRecord],
) -> Result<usize, DataError> {
    let mut groups: BTreeMap<u64, Vec<&ImitationParentHiddenRecord>> = BTreeMap::new();
    for record in records {
        if !record.parent_immediate.is_finite()
            || !record.parent_remaining.is_finite()
            || !record.parent_total().is_finite()
            || record.parent_hidden.iter().any(|value| !value.is_finite())
        {
            return Err(DataError::InvalidConfig(
                "parent-hidden record contains a non-finite value",
            ));
        }
        groups.entry(record.group_id).or_default().push(record);
    }
    for group in groups.values_mut() {
        group.sort_unstable_by_key(|record| record.candidate_index);
        let count = group.len();
        let hashes = group
            .iter()
            .map(|record| record.action_hash)
            .collect::<BTreeSet<_>>();
        if count < 2
            || count > usize::from(u16::MAX)
            || hashes.len() != count
            || group.iter().enumerate().any(|(index, record)| {
                usize::from(record.candidate_count) != count
                    || usize::from(record.candidate_index) != index
            })
        {
            return Err(DataError::InvalidConfig(
                "parent-hidden candidate group is inconsistent",
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
    records: &[ImitationParentPriorRecord],
) -> Result<(), DataError> {
    let temp_path = path.with_extension("imp.tmp");
    let mut writer = BufWriter::new(File::create(&temp_path)?);
    writer.write_all(IMITATION_PARENT_PRIOR_SHARD_MAGIC)?;
    writer.write_all(&IMITATION_PARENT_PRIOR_DATASET_SCHEMA_VERSION.to_le_bytes())?;
    writer.write_all(&(IMITATION_PARENT_PRIOR_HEADER_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(IMITATION_PARENT_PRIOR_RECORD_SIZE as u16).to_le_bytes())?;
    writer.write_all(&0u16.to_le_bytes())?;
    writer.write_all(&(records.len() as u32).to_le_bytes())?;
    writer.write_all(&(group_count as u32).to_le_bytes())?;
    writer.write_all(&(game_count as u32).to_le_bytes())?;
    writer.write_all(&[split.code(), 4, 0, 0])?;
    writer.write_all(&first_game_index.to_le_bytes())?;
    writer.write_all(blake3::hash(IMITATION_PARENT_PRIOR_FEATURE_SCHEMA.as_bytes()).as_bytes())?;
    writer.write_all(blake3::hash(IMITATION_PARENT_PRIOR_TARGET_SCHEMA.as_bytes()).as_bytes())?;
    writer.write_all(&[0; 8])?;
    for record in records {
        writer.write_all(&record.to_bytes())?;
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(temp_path, path)?;
    Ok(())
}

fn write_hidden_shard(
    path: &Path,
    split: DatasetSplit,
    first_game_index: u64,
    game_count: usize,
    group_count: usize,
    records: &[ImitationParentHiddenRecord],
) -> Result<(), DataError> {
    let temp_path = path.with_extension("imh.tmp");
    let mut writer = BufWriter::new(File::create(&temp_path)?);
    writer.write_all(IMITATION_PARENT_HIDDEN_SHARD_MAGIC)?;
    writer.write_all(&IMITATION_PARENT_HIDDEN_DATASET_SCHEMA_VERSION.to_le_bytes())?;
    writer.write_all(&(IMITATION_PARENT_HIDDEN_HEADER_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(IMITATION_PARENT_HIDDEN_RECORD_SIZE as u16).to_le_bytes())?;
    writer.write_all(&0u16.to_le_bytes())?;
    writer.write_all(&(records.len() as u32).to_le_bytes())?;
    writer.write_all(&(group_count as u32).to_le_bytes())?;
    writer.write_all(&(game_count as u32).to_le_bytes())?;
    writer.write_all(&[split.code(), 4, 0, 0])?;
    writer.write_all(&first_game_index.to_le_bytes())?;
    writer.write_all(blake3::hash(IMITATION_PARENT_HIDDEN_FEATURE_SCHEMA.as_bytes()).as_bytes())?;
    writer.write_all(blake3::hash(IMITATION_PARENT_HIDDEN_TARGET_SCHEMA.as_bytes()).as_bytes())?;
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
    let mut header = [0; IMITATION_PARENT_PRIOR_HEADER_SIZE];
    reader.read_exact(&mut header)?;
    if &header[..8] != IMITATION_PARENT_PRIOR_SHARD_MAGIC
        || u16::from_le_bytes([header[8], header[9]])
            != IMITATION_PARENT_PRIOR_DATASET_SCHEMA_VERSION
        || u16::from_le_bytes([header[10], header[11]]) as usize
            != IMITATION_PARENT_PRIOR_HEADER_SIZE
        || u16::from_le_bytes([header[12], header[13]]) as usize
            != IMITATION_PARENT_PRIOR_RECORD_SIZE
        || header[28] != split.code()
        || &header[40..72]
            != blake3::hash(IMITATION_PARENT_PRIOR_FEATURE_SCHEMA.as_bytes()).as_bytes()
        || &header[72..104]
            != blake3::hash(IMITATION_PARENT_PRIOR_TARGET_SCHEMA.as_bytes()).as_bytes()
    {
        return Err(DataError::InvalidShard("incompatible parent-prior header"));
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
            "parent-prior header and manifest disagree",
        ));
    }
    let expected_size = IMITATION_PARENT_PRIOR_HEADER_SIZE as u64
        + record_count as u64 * IMITATION_PARENT_PRIOR_RECORD_SIZE as u64;
    if fs::metadata(path)?.len() != expected_size {
        return Err(DataError::InvalidShard(
            "parent-prior shard size does not match records",
        ));
    }
    Ok(())
}

fn validate_hidden_shard_header(
    path: &Path,
    split: DatasetSplit,
    shard: &RankingShardManifest,
) -> Result<(), DataError> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut header = [0; IMITATION_PARENT_HIDDEN_HEADER_SIZE];
    reader.read_exact(&mut header)?;
    if &header[..8] != IMITATION_PARENT_HIDDEN_SHARD_MAGIC
        || u16::from_le_bytes([header[8], header[9]])
            != IMITATION_PARENT_HIDDEN_DATASET_SCHEMA_VERSION
        || u16::from_le_bytes([header[10], header[11]]) as usize
            != IMITATION_PARENT_HIDDEN_HEADER_SIZE
        || u16::from_le_bytes([header[12], header[13]]) as usize
            != IMITATION_PARENT_HIDDEN_RECORD_SIZE
        || header[28] != split.code()
        || &header[40..72]
            != blake3::hash(IMITATION_PARENT_HIDDEN_FEATURE_SCHEMA.as_bytes()).as_bytes()
        || &header[72..104]
            != blake3::hash(IMITATION_PARENT_HIDDEN_TARGET_SCHEMA.as_bytes()).as_bytes()
    {
        return Err(DataError::InvalidShard("incompatible parent-hidden header"));
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
            "parent-hidden header and manifest disagree",
        ));
    }
    let expected_size = IMITATION_PARENT_HIDDEN_HEADER_SIZE as u64
        + record_count as u64 * IMITATION_PARENT_HIDDEN_RECORD_SIZE as u64;
    if fs::metadata(path)?.len() != expected_size {
        return Err(DataError::InvalidShard(
            "parent-hidden shard size does not match records",
        ));
    }
    Ok(())
}

fn source_identity(
    root: &Path,
    manifest: &ImitationTargetsDatasetManifest,
) -> Result<ImitationParentPriorSourceManifest, DataError> {
    let root = root.canonicalize()?;
    Ok(ImitationParentPriorSourceManifest {
        path: root.display().to_string(),
        dataset_id: manifest.dataset_id.clone(),
        feature_schema: manifest.feature_schema.clone(),
        target_schema: manifest.target_schema.clone(),
        action_dataset_id: manifest.source.dataset_id.clone(),
        first_game_index: manifest.first_game_index,
        requested_games: manifest.requested_games,
    })
}

fn model_identity(model_dir: &Path) -> Result<ImitationParentPriorModelManifest, DataError> {
    let model_dir = model_dir.canonicalize()?;
    let manifest_path = model_dir.join("model.json");
    let tensors_path = model_dir.join("model.safetensors");
    let manifest: serde_json::Value =
        serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
    let architecture = manifest
        .get("architecture")
        .and_then(serde_json::Value::as_str)
        .ok_or(DataError::InvalidConfig(
            "parent-prior model manifest has no architecture",
        ))?;
    Ok(ImitationParentPriorModelManifest {
        path: model_dir.display().to_string(),
        architecture: architecture.to_owned(),
        manifest_blake3: checksum_file(&manifest_path)?,
        tensors_blake3: checksum_file(&tensors_path)?,
    })
}

fn dataset_id(
    source: &ImitationParentPriorSourceManifest,
    model: &ImitationParentPriorModelManifest,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"cascadia-v2-exact-parent-prior");
    hasher.update(source.dataset_id.as_bytes());
    hasher.update(model.manifest_blake3.as_bytes());
    hasher.update(model.tensors_blake3.as_bytes());
    hasher.update(IMITATION_PARENT_PRIOR_TARGET_SCHEMA.as_bytes());
    let digest = hasher.finalize().to_hex().to_string();
    format!("imitation-parent-prior-{}", &digest[..16])
}

fn hidden_dataset_id(
    source: &ImitationParentHiddenSourceManifest,
    model: &ImitationParentHiddenModelManifest,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"cascadia-v2-exact-parent-hidden");
    hasher.update(source.dataset_id.as_bytes());
    hasher.update(model.manifest_blake3.as_bytes());
    hasher.update(model.tensors_blake3.as_bytes());
    hasher.update(IMITATION_PARENT_HIDDEN_TARGET_SCHEMA.as_bytes());
    let digest = hasher.finalize().to_hex().to_string();
    format!("imitation-parent-hidden-{}", &digest[..16])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(index: u16) -> ImitationParentPriorRecord {
        ImitationParentPriorRecord {
            group_id: 7,
            candidate_index: index,
            candidate_count: 2,
            action_hash: [index as u8; 32],
            parent_immediate: 25.0 + f32::from(index),
            parent_remaining: 70.0 - f32::from(index),
        }
    }

    #[test]
    fn parent_prior_record_round_trip_is_exact() {
        let expected = record(0);
        assert_eq!(
            ImitationParentPriorRecord::from_bytes(&expected.to_bytes()),
            expected
        );
        assert_eq!(expected.parent_total(), 95.0);
    }

    #[test]
    fn parent_prior_groups_require_unique_complete_indices() {
        assert_eq!(validate_record_groups(&[record(0), record(1)]).unwrap(), 1);
        let duplicate = [record(0), record(0)];
        assert!(matches!(
            validate_record_groups(&duplicate),
            Err(DataError::InvalidConfig(_))
        ));
    }

    fn hidden_record(index: u16) -> ImitationParentHiddenRecord {
        let mut hidden = [0.0; IMITATION_PARENT_HIDDEN_DIM];
        for (hidden_index, value) in hidden.iter_mut().enumerate() {
            *value = f32::from(index) + hidden_index as f32 / 100.0;
        }
        ImitationParentHiddenRecord {
            group_id: 9,
            candidate_index: index,
            candidate_count: 2,
            action_hash: [index as u8; 32],
            parent_immediate: 24.0 + f32::from(index),
            parent_remaining: 71.0 - f32::from(index),
            parent_hidden: hidden,
        }
    }

    #[test]
    fn parent_hidden_record_round_trip_is_exact() {
        let expected = hidden_record(0);
        assert_eq!(
            ImitationParentHiddenRecord::from_bytes(&expected.to_bytes()),
            expected
        );
        assert_eq!(expected.parent_total(), 95.0);
    }

    #[test]
    fn parent_hidden_groups_reject_non_finite_activations() {
        assert_eq!(
            validate_hidden_record_groups(&[hidden_record(0), hidden_record(1)]).unwrap(),
            1
        );
        let mut invalid = hidden_record(1);
        invalid.parent_hidden[17] = f32::NAN;
        assert!(matches!(
            validate_hidden_record_groups(&[hidden_record(0), invalid]),
            Err(DataError::InvalidConfig(_))
        ));
    }
}
