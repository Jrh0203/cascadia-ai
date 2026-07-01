use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Seek, Write},
    path::{Path, PathBuf},
};

use cascadia_game::GameConfig;
use serde::{Deserialize, Serialize};

use super::{
    CollectionProvenance, DataError, DatasetSplit, ImitationDatasetManifest,
    ImitationTeacherConfig, RankingShardManifest, checksum_file, collection_provenance,
    collection_provenance_matches, read_array, read_imitation_shard_records, unix_seconds,
    validate_imitation_dataset, write_manifest_atomic, write_slice,
};

pub const IMITATION_TARGETS_DATASET_SCHEMA_VERSION: u16 = 1;
pub const IMITATION_TARGETS_FEATURE_SCHEMA: &str = "canonical-action-hash-alignment-v1";
pub const IMITATION_TARGETS_TARGET_SCHEMA: &str = "mce-score-distribution-v1";
pub const IMITATION_TARGETS_SHARD_MAGIC: &[u8; 8] = b"CSD2IMV\0";
pub const IMITATION_TARGETS_HEADER_SIZE: usize = 112;
pub const IMITATION_TARGETS_RECORD_SIZE: usize = 56;

pub const SOURCE_TEACHER_FRONTIER: u8 = 1 << 0;
pub const SOURCE_PATTERN_FRONTIER: u8 = 1 << 1;
pub const SOURCE_IMMEDIATE_TOP: u8 = 1 << 2;
pub const SOURCE_DETERMINISTIC_NEGATIVE: u8 = 1 << 3;

#[derive(Debug, Clone)]
pub struct ImitationTargetsDatasetConfig {
    pub output: PathBuf,
    pub source_root: PathBuf,
    pub source_manifest: ImitationDatasetManifest,
    pub resume: bool,
}

impl ImitationTargetsDatasetConfig {
    fn validate(&self) -> Result<(), DataError> {
        validate_imitation_dataset(&self.source_root, &self.source_manifest)?;
        if self.source_manifest.requested_games == 0 {
            return Err(DataError::InvalidConfig(
                "imitation-target source dataset must request at least one game",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImitationTargetsSourceManifest {
    pub path: String,
    pub dataset_id: String,
    pub feature_schema: String,
    pub target_schema: String,
    pub first_game_index: u64,
    pub requested_games: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImitationTargetsDatasetManifest {
    pub schema_version: u16,
    pub dataset_id: String,
    pub feature_schema: String,
    pub target_schema: String,
    pub record_size: usize,
    pub game: GameConfig,
    pub split: DatasetSplit,
    pub teacher: ImitationTeacherConfig,
    pub source: ImitationTargetsSourceManifest,
    pub first_game_index: u64,
    pub requested_games: usize,
    pub completed_games: usize,
    pub total_groups: usize,
    pub total_records: usize,
    pub teacher_estimates: usize,
    pub aligned_teacher_estimates: usize,
    pub created_unix_seconds: u64,
    pub updated_unix_seconds: u64,
    pub provenance: CollectionProvenance,
    pub shards: Vec<RankingShardManifest>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImitationTargetRecord {
    pub group_id: u64,
    pub candidate_index: u16,
    pub candidate_count: u16,
    pub action_hash: [u8; 32],
    pub teacher_mean: f32,
    pub teacher_stddev: f32,
    pub teacher_samples: u16,
    pub source_flags: u8,
    pub selected: bool,
}

impl ImitationTargetRecord {
    pub fn to_bytes(&self) -> [u8; IMITATION_TARGETS_RECORD_SIZE] {
        let mut bytes = [0; IMITATION_TARGETS_RECORD_SIZE];
        let mut offset = 0;
        write_slice(&mut bytes, &mut offset, &self.group_id.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.candidate_index.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.candidate_count.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.action_hash);
        write_slice(&mut bytes, &mut offset, &self.teacher_mean.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.teacher_stddev.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.teacher_samples.to_le_bytes());
        write_slice(
            &mut bytes,
            &mut offset,
            &[self.source_flags, u8::from(self.selected)],
        );
        debug_assert_eq!(offset, IMITATION_TARGETS_RECORD_SIZE);
        bytes
    }

    pub fn from_bytes(bytes: &[u8; IMITATION_TARGETS_RECORD_SIZE]) -> Self {
        let mut offset = 0;
        let group_id = u64::from_le_bytes(read_array(bytes, &mut offset));
        let candidate_index = u16::from_le_bytes(read_array(bytes, &mut offset));
        let candidate_count = u16::from_le_bytes(read_array(bytes, &mut offset));
        let action_hash = read_array(bytes, &mut offset);
        let teacher_mean = f32::from_le_bytes(read_array(bytes, &mut offset));
        let teacher_stddev = f32::from_le_bytes(read_array(bytes, &mut offset));
        let teacher_samples = u16::from_le_bytes(read_array(bytes, &mut offset));
        let [source_flags, selected] = read_array(bytes, &mut offset);
        debug_assert_eq!(offset, IMITATION_TARGETS_RECORD_SIZE);
        Self {
            group_id,
            candidate_index,
            candidate_count,
            action_hash,
            teacher_mean,
            teacher_stddev,
            teacher_samples,
            source_flags,
            selected: selected != 0,
        }
    }
}

pub struct ImitationTargetsDatasetWriter {
    output: PathBuf,
    manifest_path: PathBuf,
    manifest: ImitationTargetsDatasetManifest,
}

impl ImitationTargetsDatasetWriter {
    pub fn open(config: &ImitationTargetsDatasetConfig) -> Result<Self, DataError> {
        config.validate()?;
        fs::create_dir_all(&config.output)?;
        let source = source_manifest(config)?;
        let manifest_path = config.output.join("dataset.json");
        let manifest = if manifest_path.exists() {
            if !config.resume {
                return Err(DataError::DatasetExists(config.output.clone()));
            }
            let manifest: ImitationTargetsDatasetManifest =
                serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
            validate_resume(&manifest, config, &source)?;
            validate_imitation_targets_dataset(&config.output, &manifest)?;
            manifest
        } else {
            let now = unix_seconds()?;
            ImitationTargetsDatasetManifest {
                schema_version: IMITATION_TARGETS_DATASET_SCHEMA_VERSION,
                dataset_id: dataset_id(&source),
                feature_schema: IMITATION_TARGETS_FEATURE_SCHEMA.to_owned(),
                target_schema: IMITATION_TARGETS_TARGET_SCHEMA.to_owned(),
                record_size: IMITATION_TARGETS_RECORD_SIZE,
                game: config.source_manifest.game,
                split: config.source_manifest.split,
                teacher: config.source_manifest.teacher.clone(),
                source,
                first_game_index: config.source_manifest.first_game_index,
                requested_games: config.source_manifest.requested_games,
                completed_games: 0,
                total_groups: 0,
                total_records: 0,
                teacher_estimates: 0,
                aligned_teacher_estimates: 0,
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

    pub fn manifest(&self) -> &ImitationTargetsDatasetManifest {
        &self.manifest
    }

    pub fn root(&self) -> &Path {
        &self.output
    }

    pub fn append_shard(
        &mut self,
        first_game_index: u64,
        game_count: usize,
        records: &[ImitationTargetRecord],
        teacher_estimates: usize,
    ) -> Result<(), DataError> {
        if game_count == 0 || records.is_empty() {
            return Err(DataError::InvalidConfig(
                "imitation-target shard must contain games and records",
            ));
        }
        let expected_first = self.manifest.first_game_index + self.manifest.completed_games as u64;
        if first_game_index != expected_first
            || self.manifest.completed_games + game_count > self.manifest.requested_games
        {
            return Err(DataError::InvalidConfig(
                "imitation-target shard game range is invalid",
            ));
        }
        let group_count = validate_record_groups(records)?;
        let aligned_teacher_estimates = records
            .iter()
            .filter(|record| record.teacher_samples > 0)
            .count();
        if teacher_estimates < aligned_teacher_estimates {
            return Err(DataError::InvalidConfig(
                "teacher estimate count cannot be below aligned evidence",
            ));
        }
        let shard_index = self.manifest.shards.len();
        let file_name = format!("shard-{shard_index:05}.imv");
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
        self.manifest.teacher_estimates += teacher_estimates;
        self.manifest.aligned_teacher_estimates += aligned_teacher_estimates;
        self.manifest.updated_unix_seconds = unix_seconds()?;
        write_manifest_atomic(&self.manifest_path, &self.manifest)
    }
}

pub fn read_imitation_target_shard_records(
    root: &Path,
    split: DatasetSplit,
    shard: &RankingShardManifest,
) -> Result<Vec<ImitationTargetRecord>, DataError> {
    let path = root.join(&shard.file);
    validate_shard_header(&path, split, shard)?;
    let mut reader = BufReader::new(File::open(path)?);
    reader.seek(std::io::SeekFrom::Start(
        IMITATION_TARGETS_HEADER_SIZE as u64,
    ))?;
    let mut records = Vec::with_capacity(shard.record_count);
    for _ in 0..shard.record_count {
        let mut bytes = [0; IMITATION_TARGETS_RECORD_SIZE];
        reader.read_exact(&mut bytes)?;
        records.push(ImitationTargetRecord::from_bytes(&bytes));
    }
    validate_record_groups(&records)?;
    Ok(records)
}

pub fn validate_imitation_targets_dataset(
    root: &Path,
    manifest: &ImitationTargetsDatasetManifest,
) -> Result<(), DataError> {
    if manifest.schema_version != IMITATION_TARGETS_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != IMITATION_TARGETS_FEATURE_SCHEMA
        || manifest.target_schema != IMITATION_TARGETS_TARGET_SCHEMA
        || manifest.record_size != IMITATION_TARGETS_RECORD_SIZE
        || manifest.game != GameConfig::research_aaaaa(4)?
    {
        return Err(DataError::InvalidManifest(
            "imitation-target schema identifiers do not match",
        ));
    }
    let source_path = Path::new(&manifest.source.path);
    let source_manifest: ImitationDatasetManifest = serde_json::from_reader(BufReader::new(
        File::open(source_path.join("dataset.json"))?,
    ))?;
    validate_imitation_dataset(source_path, &source_manifest)?;
    if source_identity(source_path, &source_manifest)? != manifest.source
        || source_manifest.teacher != manifest.teacher
        || source_manifest.split != manifest.split
    {
        return Err(DataError::InvalidManifest(
            "imitation-target source identity does not match",
        ));
    }

    let mut games = 0;
    let mut groups = 0;
    let mut records = 0;
    let mut aligned_teacher_estimates = 0;
    if manifest.shards.len() > source_manifest.shards.len() {
        return Err(DataError::InvalidManifest(
            "imitation-target has more shards than its source",
        ));
    }
    for (shard_index, shard) in manifest.shards.iter().enumerate() {
        let path = root.join(&shard.file);
        if fs::metadata(&path)?.len() != shard.byte_count {
            return Err(DataError::InvalidManifest(
                "imitation-target shard byte count mismatch",
            ));
        }
        if checksum_file(&path)? != shard.blake3 {
            return Err(DataError::ChecksumMismatch(path));
        }
        let shard_records = read_imitation_target_shard_records(root, manifest.split, shard)?;
        let source_shard = &source_manifest.shards[shard_index];
        if source_shard.first_game_index != shard.first_game_index
            || source_shard.game_count != shard.game_count
            || source_shard.group_count != shard.group_count
            || source_shard.record_count != shard.record_count
        {
            return Err(DataError::InvalidManifest(
                "imitation-target shard range does not match source shard",
            ));
        }
        let source_records =
            read_imitation_shard_records(source_path, source_manifest.split, source_shard)?;
        if source_records.len() != shard_records.len()
            || source_records
                .iter()
                .zip(&shard_records)
                .any(|(source, target)| {
                    source.group_id != target.group_id
                        || source.candidate_index != target.candidate_index
                        || source.candidate_count != target.candidate_count
                        || source.action_hash != target.action_hash
                        || (source.teacher_mean == 1.0) != target.selected
                })
        {
            return Err(DataError::InvalidManifest(
                "imitation-target records are not aligned with source actions",
            ));
        }
        if validate_record_groups(&shard_records)? != shard.group_count {
            return Err(DataError::InvalidManifest(
                "imitation-target shard group count mismatch",
            ));
        }
        games += shard.game_count;
        groups += shard.group_count;
        records += shard.record_count;
        aligned_teacher_estimates += shard_records
            .iter()
            .filter(|record| record.teacher_samples > 0)
            .count();
    }
    if games != manifest.completed_games
        || groups != manifest.total_groups
        || records != manifest.total_records
        || aligned_teacher_estimates != manifest.aligned_teacher_estimates
        || manifest.aligned_teacher_estimates > manifest.teacher_estimates
        || manifest.completed_games > source_manifest.completed_games
        || manifest.completed_games > manifest.requested_games
    {
        return Err(DataError::InvalidManifest(
            "imitation-target manifest totals do not match shards",
        ));
    }
    Ok(())
}

fn validate_resume(
    manifest: &ImitationTargetsDatasetManifest,
    config: &ImitationTargetsDatasetConfig,
    source: &ImitationTargetsSourceManifest,
) -> Result<(), DataError> {
    let current_provenance = collection_provenance()?;
    if manifest.schema_version != IMITATION_TARGETS_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != IMITATION_TARGETS_FEATURE_SCHEMA
        || manifest.target_schema != IMITATION_TARGETS_TARGET_SCHEMA
        || manifest.record_size != IMITATION_TARGETS_RECORD_SIZE
        || manifest.game != config.source_manifest.game
        || manifest.split != config.source_manifest.split
        || manifest.teacher != config.source_manifest.teacher
        || manifest.source != *source
        || manifest.first_game_index != config.source_manifest.first_game_index
        || manifest.requested_games != config.source_manifest.requested_games
        || manifest.teacher_estimates < manifest.aligned_teacher_estimates
        || !collection_provenance_matches(&manifest.provenance, &current_provenance)
    {
        return Err(DataError::ResumeMismatch);
    }
    Ok(())
}

fn validate_record_groups(records: &[ImitationTargetRecord]) -> Result<usize, DataError> {
    let mut groups: BTreeMap<u64, Vec<&ImitationTargetRecord>> = BTreeMap::new();
    for record in records {
        if record.source_flags == 0
            || !record.teacher_mean.is_finite()
            || !record.teacher_stddev.is_finite()
            || record.teacher_stddev < 0.0
            || (record.teacher_samples == 0
                && (record.teacher_mean != 0.0 || record.teacher_stddev != 0.0))
        {
            return Err(DataError::InvalidConfig(
                "imitation-target record contains invalid evidence",
            ));
        }
        groups.entry(record.group_id).or_default().push(record);
    }
    for group in groups.values_mut() {
        group.sort_unstable_by_key(|record| record.candidate_index);
        let count = group.len();
        let selected = group.iter().filter(|record| record.selected).count();
        let hashes = group
            .iter()
            .map(|record| record.action_hash)
            .collect::<BTreeSet<_>>();
        let best_mean = group
            .iter()
            .filter(|record| record.teacher_samples > 0)
            .map(|record| record.teacher_mean)
            .fold(f32::NEG_INFINITY, f32::max);
        if count < 2
            || count > usize::from(u16::MAX)
            || selected != 1
            || hashes.len() != count
            || group.iter().enumerate().any(|(index, record)| {
                usize::from(record.candidate_count) != count
                    || usize::from(record.candidate_index) != index
                    || (record.selected
                        && (record.teacher_samples == 0
                            || (record.teacher_mean - best_mean).abs() > f32::EPSILON))
            })
        {
            return Err(DataError::InvalidConfig(
                "imitation-target candidate group is inconsistent",
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
    records: &[ImitationTargetRecord],
) -> Result<(), DataError> {
    let temp_path = path.with_extension("imv.tmp");
    let mut writer = BufWriter::new(File::create(&temp_path)?);
    writer.write_all(IMITATION_TARGETS_SHARD_MAGIC)?;
    writer.write_all(&IMITATION_TARGETS_DATASET_SCHEMA_VERSION.to_le_bytes())?;
    writer.write_all(&(IMITATION_TARGETS_HEADER_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(IMITATION_TARGETS_RECORD_SIZE as u16).to_le_bytes())?;
    writer.write_all(&0u16.to_le_bytes())?;
    writer.write_all(&(records.len() as u32).to_le_bytes())?;
    writer.write_all(&(group_count as u32).to_le_bytes())?;
    writer.write_all(&(game_count as u32).to_le_bytes())?;
    writer.write_all(&[split.code(), 4, 0, 0])?;
    writer.write_all(&first_game_index.to_le_bytes())?;
    writer.write_all(blake3::hash(IMITATION_TARGETS_FEATURE_SCHEMA.as_bytes()).as_bytes())?;
    writer.write_all(blake3::hash(IMITATION_TARGETS_TARGET_SCHEMA.as_bytes()).as_bytes())?;
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
    let mut header = [0; IMITATION_TARGETS_HEADER_SIZE];
    reader.read_exact(&mut header)?;
    if &header[..8] != IMITATION_TARGETS_SHARD_MAGIC
        || u16::from_le_bytes([header[8], header[9]]) != IMITATION_TARGETS_DATASET_SCHEMA_VERSION
        || u16::from_le_bytes([header[10], header[11]]) as usize != IMITATION_TARGETS_HEADER_SIZE
        || u16::from_le_bytes([header[12], header[13]]) as usize != IMITATION_TARGETS_RECORD_SIZE
        || header[28] != split.code()
        || &header[40..72] != blake3::hash(IMITATION_TARGETS_FEATURE_SCHEMA.as_bytes()).as_bytes()
        || &header[72..104] != blake3::hash(IMITATION_TARGETS_TARGET_SCHEMA.as_bytes()).as_bytes()
    {
        return Err(DataError::InvalidShard(
            "incompatible imitation-target header",
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
            "imitation-target header and manifest disagree",
        ));
    }
    let expected_size = IMITATION_TARGETS_HEADER_SIZE as u64
        + record_count as u64 * IMITATION_TARGETS_RECORD_SIZE as u64;
    if fs::metadata(path)?.len() != expected_size {
        return Err(DataError::InvalidShard(
            "imitation-target shard size does not match records",
        ));
    }
    Ok(())
}

fn source_manifest(
    config: &ImitationTargetsDatasetConfig,
) -> Result<ImitationTargetsSourceManifest, DataError> {
    let root = config.source_root.canonicalize()?;
    source_identity(&root, &config.source_manifest)
}

fn source_identity(
    root: &Path,
    manifest: &ImitationDatasetManifest,
) -> Result<ImitationTargetsSourceManifest, DataError> {
    Ok(ImitationTargetsSourceManifest {
        path: root.display().to_string(),
        dataset_id: manifest.dataset_id.clone(),
        feature_schema: manifest.feature_schema.clone(),
        target_schema: manifest.target_schema.clone(),
        first_game_index: manifest.first_game_index,
        requested_games: manifest.requested_games,
    })
}

fn dataset_id(source: &ImitationTargetsSourceManifest) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"cascadia-v2-imitation-target-enrichment");
    hasher.update(source.dataset_id.as_bytes());
    hasher.update(&source.requested_games.to_le_bytes());
    hasher.update(IMITATION_TARGETS_TARGET_SCHEMA.as_bytes());
    let digest = hasher.finalize().to_hex().to_string();
    format!("imitation-targets-{}", &digest[..16])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(index: u16, selected: bool, scored: bool) -> ImitationTargetRecord {
        ImitationTargetRecord {
            group_id: 7,
            candidate_index: index,
            candidate_count: 2,
            action_hash: [index as u8; 32],
            teacher_mean: if scored { 90.0 - f32::from(index) } else { 0.0 },
            teacher_stddev: if scored { 3.0 } else { 0.0 },
            teacher_samples: if scored { 12 } else { 0 },
            source_flags: if scored {
                SOURCE_TEACHER_FRONTIER
            } else {
                SOURCE_DETERMINISTIC_NEGATIVE
            },
            selected,
        }
    }

    #[test]
    fn target_record_round_trip_is_exact() {
        let expected = record(0, true, true);
        assert_eq!(
            ImitationTargetRecord::from_bytes(&expected.to_bytes()),
            expected
        );
    }

    #[test]
    fn target_groups_require_one_scored_maximum_selection() {
        let valid = [record(0, true, true), record(1, false, false)];
        assert_eq!(validate_record_groups(&valid).unwrap(), 1);

        let invalid = [record(0, false, true), record(1, true, false)];
        assert!(matches!(
            validate_record_groups(&invalid),
            Err(DataError::InvalidConfig(_))
        ));
    }
}
