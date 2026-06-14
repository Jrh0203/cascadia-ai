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

pub const PUBLIC_BEAM_VALUE_DATASET_SCHEMA_VERSION: u16 = 1;
pub const PUBLIC_BEAM_VALUE_FEATURE_SCHEMA: &str = "observable-action-afterstate-v1";
pub const PUBLIC_BEAM_VALUE_TARGET_SCHEMA: &str = "public-redetermined-b16-w2-r8x2-terminal-v1";
pub const PUBLIC_BEAM_VALUE_SHARD_MAGIC: &[u8; 8] = b"CSD2PBV\0";
pub const PUBLIC_BEAM_VALUE_HEADER_SIZE: usize = 128;
pub const PUBLIC_BEAM_VALUE_RECORD_SIZE: usize = 96 + ACTION_POSITION_RECORD_SIZE;

#[derive(Debug, Clone)]
pub struct PublicBeamValueDatasetConfig {
    pub output: PathBuf,
    pub split: DatasetSplit,
    pub first_game_index: u64,
    pub games: usize,
    pub teacher: PublicBeamValueTeacherConfig,
    pub resume: bool,
}

impl PublicBeamValueDatasetConfig {
    fn validate(&self) -> Result<(), DataError> {
        if self.games == 0 {
            return Err(DataError::InvalidConfig("games must be positive"));
        }
        self.teacher.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicBeamValueTeacherConfig {
    pub strategy_id: String,
    pub trajectory_strategy_id: String,
    pub final_personal_turns: u16,
    pub recorded_personal_turns: Vec<u16>,
    pub determinizations_per_batch: usize,
    pub batches: usize,
    pub immediate_candidates: usize,
    pub habitat_candidates: usize,
    pub bear_candidates: usize,
    pub wildlife_candidates: usize,
    pub future_market_draws: usize,
    pub beam_width: usize,
    pub seed_schema: String,
}

impl PublicBeamValueTeacherConfig {
    fn validate(&self) -> Result<(), DataError> {
        if self.strategy_id.trim().is_empty()
            || self.trajectory_strategy_id.trim().is_empty()
            || self.seed_schema != "public-state-hash-domain-separated-v1"
            || self.final_personal_turns != 5
            || self.recorded_personal_turns != [5, 4, 3, 2]
            || self.determinizations_per_batch != 8
            || self.batches != 2
            || self.immediate_candidates != 8
            || self.habitat_candidates != 6
            || self.bear_candidates != 8
            || self.wildlife_candidates != 2
            || self.future_market_draws != 4
            || self.beam_width != 16
        {
            return Err(DataError::InvalidConfig(
                "public beam value teacher must use the frozen R8x2 B16 W2 probe",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicBeamValueDatasetManifest {
    pub schema_version: u16,
    pub dataset_id: String,
    pub feature_schema: String,
    pub target_schema: String,
    pub record_size: usize,
    pub action_position_record_size: usize,
    pub game: GameConfig,
    pub split: DatasetSplit,
    pub teacher: PublicBeamValueTeacherConfig,
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
pub struct PublicBeamValueRecord {
    pub group_id: u64,
    pub candidate_index: u16,
    pub candidate_count: u16,
    pub current_base_score: u16,
    pub batch_a_mean: f32,
    pub batch_b_mean: f32,
    pub batch_a_stddev: f32,
    pub batch_b_stddev: f32,
    pub public_position_hash: [u8; 32],
    pub action_hash: [u8; 32],
    pub input: ActionPositionRecord,
}

impl PublicBeamValueRecord {
    pub fn to_bytes(&self) -> [u8; PUBLIC_BEAM_VALUE_RECORD_SIZE] {
        let mut bytes = [0; PUBLIC_BEAM_VALUE_RECORD_SIZE];
        let mut offset = 0;
        write_slice(&mut bytes, &mut offset, &self.group_id.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.candidate_index.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.candidate_count.to_le_bytes());
        write_slice(
            &mut bytes,
            &mut offset,
            &self.current_base_score.to_le_bytes(),
        );
        write_slice(&mut bytes, &mut offset, &[0, 0]);
        write_slice(&mut bytes, &mut offset, &self.batch_a_mean.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.batch_b_mean.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.batch_a_stddev.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.batch_b_stddev.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.public_position_hash);
        write_slice(&mut bytes, &mut offset, &self.action_hash);
        write_slice(&mut bytes, &mut offset, &self.input.to_bytes());
        debug_assert_eq!(offset, PUBLIC_BEAM_VALUE_RECORD_SIZE);
        bytes
    }

    pub fn from_bytes(bytes: &[u8; PUBLIC_BEAM_VALUE_RECORD_SIZE]) -> Self {
        let mut offset = 0;
        let group_id = u64::from_le_bytes(read_array(bytes, &mut offset));
        let candidate_index = u16::from_le_bytes(read_array(bytes, &mut offset));
        let candidate_count = u16::from_le_bytes(read_array(bytes, &mut offset));
        let current_base_score = u16::from_le_bytes(read_array(bytes, &mut offset));
        let _: [u8; 2] = read_array(bytes, &mut offset);
        let batch_a_mean = f32::from_le_bytes(read_array(bytes, &mut offset));
        let batch_b_mean = f32::from_le_bytes(read_array(bytes, &mut offset));
        let batch_a_stddev = f32::from_le_bytes(read_array(bytes, &mut offset));
        let batch_b_stddev = f32::from_le_bytes(read_array(bytes, &mut offset));
        let public_position_hash = read_array(bytes, &mut offset);
        let action_hash = read_array(bytes, &mut offset);
        let input_bytes: [u8; ACTION_POSITION_RECORD_SIZE] = read_array(bytes, &mut offset);
        debug_assert_eq!(offset, PUBLIC_BEAM_VALUE_RECORD_SIZE);
        Self {
            group_id,
            candidate_index,
            candidate_count,
            current_base_score,
            batch_a_mean,
            batch_b_mean,
            batch_a_stddev,
            batch_b_stddev,
            public_position_hash,
            action_hash,
            input: ActionPositionRecord::from_bytes(&input_bytes),
        }
    }

    fn validate(&self) -> Result<(), DataError> {
        let derived_current = i32::from(self.input.action.immediate_score)
            - self
                .input
                .action
                .immediate_deltas
                .iter()
                .map(|delta| i32::from(*delta))
                .sum::<i32>();
        if self.candidate_count == 0
            || self.candidate_index >= self.candidate_count
            || derived_current != i32::from(self.current_base_score)
            || !self.batch_a_mean.is_finite()
            || !self.batch_b_mean.is_finite()
            || !self.batch_a_stddev.is_finite()
            || !self.batch_b_stddev.is_finite()
            || self.batch_a_stddev < 0.0
            || self.batch_b_stddev < 0.0
            || self.input.action.replace_three_of_a_kind != 0
            || self.input.action.paid_wipe_count != 0
            || self.input.action.paid_wipe_total_slots != 0
        {
            return Err(DataError::InvalidShard("invalid public beam value record"));
        }
        Ok(())
    }
}

pub struct PublicBeamValueDatasetWriter {
    output: PathBuf,
    manifest_path: PathBuf,
    manifest: PublicBeamValueDatasetManifest,
}

impl PublicBeamValueDatasetWriter {
    pub fn open(config: &PublicBeamValueDatasetConfig) -> Result<Self, DataError> {
        config.validate()?;
        fs::create_dir_all(&config.output)?;
        let manifest_path = config.output.join("dataset.json");
        let mut manifest = if manifest_path.exists() {
            if !config.resume {
                return Err(DataError::DatasetExists(config.output.clone()));
            }
            let manifest: PublicBeamValueDatasetManifest =
                serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
            validate_resume(&manifest, config)?;
            validate_public_beam_value_dataset(&config.output, &manifest)?;
            manifest
        } else {
            let now = unix_seconds()?;
            PublicBeamValueDatasetManifest {
                schema_version: PUBLIC_BEAM_VALUE_DATASET_SCHEMA_VERSION,
                dataset_id: format!(
                    "public-beam-value-{}-{}-{}",
                    config.teacher.strategy_id,
                    config.split.id(),
                    config.first_game_index
                ),
                feature_schema: PUBLIC_BEAM_VALUE_FEATURE_SCHEMA.to_owned(),
                target_schema: PUBLIC_BEAM_VALUE_TARGET_SCHEMA.to_owned(),
                record_size: PUBLIC_BEAM_VALUE_RECORD_SIZE,
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

    pub fn manifest(&self) -> &PublicBeamValueDatasetManifest {
        &self.manifest
    }

    pub fn append_shard(
        &mut self,
        first_game_index: u64,
        game_count: usize,
        records: &[PublicBeamValueRecord],
    ) -> Result<(), DataError> {
        let expected_first = self.manifest.first_game_index + self.manifest.completed_games as u64;
        if game_count == 0
            || records.is_empty()
            || first_game_index != expected_first
            || self.manifest.completed_games + game_count > self.manifest.requested_games
        {
            return Err(DataError::InvalidConfig(
                "public beam value shard range is invalid",
            ));
        }
        let group_count = validate_record_groups(records)?;
        let shard_index = self.manifest.shards.len();
        let file_name = format!("shard-{shard_index:05}.pbv");
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

pub fn read_public_beam_value_shard_records(
    root: &Path,
    split: DatasetSplit,
    shard: &RankingShardManifest,
) -> Result<Vec<PublicBeamValueRecord>, DataError> {
    let path = root.join(&shard.file);
    validate_shard_header(&path, split, shard)?;
    let mut reader = BufReader::new(File::open(path)?);
    reader.seek(std::io::SeekFrom::Start(
        PUBLIC_BEAM_VALUE_HEADER_SIZE as u64,
    ))?;
    let mut records = Vec::with_capacity(shard.record_count);
    for _ in 0..shard.record_count {
        let mut bytes = [0; PUBLIC_BEAM_VALUE_RECORD_SIZE];
        reader.read_exact(&mut bytes)?;
        records.push(PublicBeamValueRecord::from_bytes(&bytes));
    }
    validate_record_groups(&records)?;
    Ok(records)
}

pub fn validate_public_beam_value_dataset(
    root: &Path,
    manifest: &PublicBeamValueDatasetManifest,
) -> Result<(), DataError> {
    if manifest.schema_version != PUBLIC_BEAM_VALUE_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != PUBLIC_BEAM_VALUE_FEATURE_SCHEMA
        || manifest.target_schema != PUBLIC_BEAM_VALUE_TARGET_SCHEMA
        || manifest.record_size != PUBLIC_BEAM_VALUE_RECORD_SIZE
        || manifest.action_position_record_size != ACTION_POSITION_RECORD_SIZE
        || manifest.game != GameConfig::research_aaaaa(4)?
    {
        return Err(DataError::InvalidManifest(
            "public beam value schema identifiers do not match",
        ));
    }
    manifest.teacher.validate()?;
    let mut games = 0;
    let mut groups = 0;
    let mut records = 0;
    for shard in &manifest.shards {
        let path = root.join(&shard.file);
        if fs::metadata(&path)?.len() != shard.byte_count || checksum_file(&path)? != shard.blake3 {
            return Err(DataError::ChecksumMismatch(path));
        }
        let shard_records = read_public_beam_value_shard_records(root, manifest.split, shard)?;
        if validate_record_groups(&shard_records)? != shard.group_count {
            return Err(DataError::InvalidManifest(
                "public beam value shard group count mismatch",
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
            "public beam value manifest totals do not match shards",
        ));
    }
    Ok(())
}

fn validate_resume(
    manifest: &PublicBeamValueDatasetManifest,
    config: &PublicBeamValueDatasetConfig,
) -> Result<(), DataError> {
    let provenance = collection_provenance()?;
    if manifest.schema_version != PUBLIC_BEAM_VALUE_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != PUBLIC_BEAM_VALUE_FEATURE_SCHEMA
        || manifest.target_schema != PUBLIC_BEAM_VALUE_TARGET_SCHEMA
        || manifest.record_size != PUBLIC_BEAM_VALUE_RECORD_SIZE
        || manifest.action_position_record_size != ACTION_POSITION_RECORD_SIZE
        || manifest.game != GameConfig::research_aaaaa(4)?
        || manifest.split != config.split
        || manifest.teacher != config.teacher
        || manifest.first_game_index != config.first_game_index
        || !collection_provenance_matches(&manifest.provenance, &provenance)
    {
        return Err(DataError::ResumeMismatch);
    }
    Ok(())
}

fn validate_record_groups(records: &[PublicBeamValueRecord]) -> Result<usize, DataError> {
    let mut groups = BTreeMap::<u64, Vec<&PublicBeamValueRecord>>::new();
    for record in records {
        record.validate()?;
        groups.entry(record.group_id).or_default().push(record);
    }
    for group in groups.values() {
        let count = group.len();
        if group.iter().any(|record| {
            usize::from(record.candidate_count) != count
                || record.public_position_hash != group[0].public_position_hash
                || record.current_base_score != group[0].current_base_score
        }) {
            return Err(DataError::InvalidShard(
                "inconsistent public beam value group",
            ));
        }
        let mut action_hashes = group
            .iter()
            .map(|record| record.action_hash)
            .collect::<Vec<_>>();
        action_hashes.sort_unstable();
        action_hashes.dedup();
        if action_hashes.len() != count {
            return Err(DataError::InvalidShard(
                "public beam value group contains duplicate actions",
            ));
        }
        let mut indexes = group
            .iter()
            .map(|record| record.candidate_index)
            .collect::<Vec<_>>();
        indexes.sort_unstable();
        if indexes
            != (0..u16::try_from(count).map_err(|_| {
                DataError::InvalidShard("public beam value candidate count exceeds u16")
            })?)
                .collect::<Vec<_>>()
        {
            return Err(DataError::InvalidShard(
                "public beam value candidate indexes are not contiguous",
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
    records: &[PublicBeamValueRecord],
) -> Result<(), DataError> {
    let temp_path = path.with_extension("pbv.tmp");
    let mut writer = BufWriter::new(File::create(&temp_path)?);
    writer.write_all(PUBLIC_BEAM_VALUE_SHARD_MAGIC)?;
    writer.write_all(&PUBLIC_BEAM_VALUE_DATASET_SCHEMA_VERSION.to_le_bytes())?;
    writer.write_all(&(PUBLIC_BEAM_VALUE_HEADER_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(PUBLIC_BEAM_VALUE_RECORD_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(ACTION_POSITION_RECORD_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(records.len() as u32).to_le_bytes())?;
    writer.write_all(&(group_count as u32).to_le_bytes())?;
    writer.write_all(&(game_count as u32).to_le_bytes())?;
    writer.write_all(&[split.code(), 4, 0, 0])?;
    writer.write_all(&first_game_index.to_le_bytes())?;
    writer.write_all(blake3::hash(PUBLIC_BEAM_VALUE_FEATURE_SCHEMA.as_bytes()).as_bytes())?;
    writer.write_all(blake3::hash(PUBLIC_BEAM_VALUE_TARGET_SCHEMA.as_bytes()).as_bytes())?;
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
    let mut header = [0; PUBLIC_BEAM_VALUE_HEADER_SIZE];
    reader.read_exact(&mut header)?;
    if &header[..8] != PUBLIC_BEAM_VALUE_SHARD_MAGIC
        || u16::from_le_bytes([header[8], header[9]]) != PUBLIC_BEAM_VALUE_DATASET_SCHEMA_VERSION
        || u16::from_le_bytes([header[10], header[11]]) as usize != PUBLIC_BEAM_VALUE_HEADER_SIZE
        || u16::from_le_bytes([header[12], header[13]]) as usize != PUBLIC_BEAM_VALUE_RECORD_SIZE
        || u16::from_le_bytes([header[14], header[15]]) as usize != ACTION_POSITION_RECORD_SIZE
        || header[28] != split.code()
        || &header[40..72] != blake3::hash(PUBLIC_BEAM_VALUE_FEATURE_SCHEMA.as_bytes()).as_bytes()
        || &header[72..104] != blake3::hash(PUBLIC_BEAM_VALUE_TARGET_SCHEMA.as_bytes()).as_bytes()
    {
        return Err(DataError::InvalidShard(
            "incompatible public beam value header",
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
            "public beam value header and manifest disagree",
        ));
    }
    let expected = PUBLIC_BEAM_VALUE_HEADER_SIZE as u64
        + record_count as u64 * PUBLIC_BEAM_VALUE_RECORD_SIZE as u64;
    if fs::metadata(path)?.len() != expected {
        return Err(DataError::InvalidShard(
            "public beam value shard size does not match records",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameSeed, GameState, score_board};

    use super::*;

    #[test]
    fn record_round_trip_preserves_public_batch_targets() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(501),
        )
        .unwrap();
        let action = game
            .legal_turn_actions(&Default::default())
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let immediate_score = score_board(
            &game.preview_active_board(&action).unwrap(),
            game.config().scoring_cards,
        )
        .base_total;
        let current_base_score = score_board(
            &game.boards()[game.current_player()],
            game.config().scoring_cards,
        )
        .base_total;
        let record = PublicBeamValueRecord {
            group_id: 9,
            candidate_index: 0,
            candidate_count: 1,
            current_base_score,
            batch_a_mean: 91.25,
            batch_b_mean: 92.0,
            batch_a_stddev: 1.5,
            batch_b_stddev: 2.0,
            public_position_hash: [3; 32],
            action_hash: [4; 32],
            input: ActionPositionRecord::observe(&game, &action, 501, 1, immediate_score).unwrap(),
        };
        assert_eq!(
            PublicBeamValueRecord::from_bytes(&record.to_bytes()),
            record
        );
        record.validate().unwrap();
    }

    #[test]
    fn writer_round_trip_validates_header_records_and_checksums() {
        let output = std::env::temp_dir().join(format!(
            "cascadia-public-beam-value-{}-{}",
            std::process::id(),
            unix_seconds().unwrap()
        ));
        let teacher = PublicBeamValueTeacherConfig {
            strategy_id: "public-beam-state-value-observability-v1-r8x2-b16-w2".to_owned(),
            trajectory_strategy_id: "pattern-aware-k8-h6-b8-m4".to_owned(),
            final_personal_turns: 5,
            recorded_personal_turns: vec![5, 4, 3, 2],
            determinizations_per_batch: 8,
            batches: 2,
            immediate_candidates: 8,
            habitat_candidates: 6,
            bear_candidates: 8,
            wildlife_candidates: 2,
            future_market_draws: 4,
            beam_width: 16,
            seed_schema: "public-state-hash-domain-separated-v1".to_owned(),
        };
        let config = PublicBeamValueDatasetConfig {
            output: output.clone(),
            split: DatasetSplit::Train,
            first_game_index: 40_000,
            games: 1,
            teacher,
            resume: false,
        };
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            DatasetSplit::Train.game_seed(40_000),
        )
        .unwrap();
        let action = game
            .legal_turn_actions(&Default::default())
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let immediate_score = score_board(
            &game.preview_active_board(&action).unwrap(),
            game.config().scoring_cards,
        )
        .base_total;
        let current_base_score = score_board(
            &game.boards()[game.current_player()],
            game.config().scoring_cards,
        )
        .base_total;
        let record = PublicBeamValueRecord {
            group_id: 11,
            candidate_index: 0,
            candidate_count: 1,
            current_base_score,
            batch_a_mean: 90.0,
            batch_b_mean: 90.5,
            batch_a_stddev: 1.0,
            batch_b_stddev: 1.25,
            public_position_hash: [5; 32],
            action_hash: [6; 32],
            input: ActionPositionRecord::observe(&game, &action, 40_000, 1, immediate_score)
                .unwrap(),
        };

        let mut writer = PublicBeamValueDatasetWriter::open(&config).unwrap();
        writer
            .append_shard(40_000, 1, std::slice::from_ref(&record))
            .unwrap();
        let manifest = writer.manifest().clone();
        validate_public_beam_value_dataset(&output, &manifest).unwrap();
        let actual =
            read_public_beam_value_shard_records(&output, manifest.split, &manifest.shards[0])
                .unwrap();
        assert_eq!(actual, vec![record]);
        assert_eq!(
            fs::metadata(output.join(&manifest.shards[0].file))
                .unwrap()
                .len(),
            PUBLIC_BEAM_VALUE_HEADER_SIZE as u64 + PUBLIC_BEAM_VALUE_RECORD_SIZE as u64
        );
        fs::remove_dir_all(output).unwrap();
    }
}
