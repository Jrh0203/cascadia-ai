use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Seek, Write},
    path::{Path, PathBuf},
};

use cascadia_game::{GameConfig, ScoreBreakdown};
use serde::{Deserialize, Serialize};

use super::{
    CollectionProvenance, DataError, DatasetSplit, FEATURE_SCHEMA, PositionRecord, RECORD_SIZE,
    ShardManifest, TARGET_DIM, checksum_file, collection_provenance, collection_provenance_matches,
    read_array, unix_seconds, write_manifest_atomic, write_slice,
};

pub const SCORE_TO_GO_DATASET_SCHEMA_VERSION: u16 = 1;
pub const SCORE_TO_GO_FEATURE_SCHEMA: &str = FEATURE_SCHEMA;
pub const SCORE_TO_GO_TARGET_SCHEMA: &str = "signed-score-to-go-components-v1";
pub const SCORE_TO_GO_SHARD_MAGIC: &[u8; 8] = b"CSD2STG\0";
pub const SCORE_TO_GO_HEADER_SIZE: usize = 128;
pub const SCORE_TO_GO_RECORD_SIZE: usize = RECORD_SIZE + TARGET_DIM * 4;

#[derive(Debug, Clone)]
pub struct ScoreToGoDatasetConfig {
    pub output: PathBuf,
    pub split: DatasetSplit,
    pub first_game_index: u64,
    pub games: usize,
    pub teacher: ScoreToGoTeacherConfig,
    pub resume: bool,
}

impl ScoreToGoDatasetConfig {
    fn validate(&self) -> Result<(), DataError> {
        if self.games == 0 {
            return Err(DataError::InvalidConfig("games must be positive"));
        }
        self.teacher.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScoreToGoTeacherConfig {
    pub strategy_id: String,
    pub immediate_candidates: usize,
    pub habitat_candidates: usize,
    pub determinizations: usize,
    pub greedy_plies: usize,
}

impl ScoreToGoTeacherConfig {
    fn validate(&self) -> Result<(), DataError> {
        if self.strategy_id.trim().is_empty()
            || self.immediate_candidates != 8
            || self.habitat_candidates != 6
            || self.determinizations != 4
            || self.greedy_plies != 4
        {
            return Err(DataError::InvalidConfig(
                "score-to-go teacher must use the frozen H6 K8/H6/R4/D4 configuration",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScoreToGoDatasetManifest {
    pub schema_version: u16,
    pub dataset_id: String,
    pub feature_schema: String,
    pub target_schema: String,
    pub record_size: usize,
    pub position_record_size: usize,
    pub target_dim: usize,
    pub game: GameConfig,
    pub split: DatasetSplit,
    pub teacher: ScoreToGoTeacherConfig,
    pub first_game_index: u64,
    pub requested_games: usize,
    pub completed_games: usize,
    pub total_records: usize,
    pub created_unix_seconds: u64,
    pub updated_unix_seconds: u64,
    pub provenance: CollectionProvenance,
    pub shards: Vec<ShardManifest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScoreToGoRecord {
    pub position: PositionRecord,
    pub current: [u16; TARGET_DIM],
    pub residual: [i16; TARGET_DIM],
}

impl ScoreToGoRecord {
    pub fn new(
        mut position: PositionRecord,
        current: ScoreBreakdown,
        final_score: ScoreBreakdown,
    ) -> Result<Self, DataError> {
        let current = score_components(current);
        let final_components = score_components(final_score);
        position.targets = final_components;
        let mut residual = [0; TARGET_DIM];
        for index in 0..TARGET_DIM {
            residual[index] =
                i16::try_from(i32::from(final_components[index]) - i32::from(current[index]))
                    .map_err(|_| DataError::InvalidConfig("score-to-go residual exceeds i16"))?;
        }
        let record = Self {
            position,
            current,
            residual,
        };
        record.validate()?;
        Ok(record)
    }

    pub fn validate(&self) -> Result<(), DataError> {
        for index in 0..TARGET_DIM {
            if i32::from(self.current[index]) + i32::from(self.residual[index])
                != i32::from(self.position.targets[index])
            {
                return Err(DataError::InvalidConfig(
                    "score-to-go current, residual, and final targets disagree",
                ));
            }
        }
        Ok(())
    }

    pub fn to_bytes(&self) -> [u8; SCORE_TO_GO_RECORD_SIZE] {
        let mut bytes = [0; SCORE_TO_GO_RECORD_SIZE];
        let mut offset = 0;
        write_slice(&mut bytes, &mut offset, &self.position.to_bytes());
        for value in self.current {
            write_slice(&mut bytes, &mut offset, &value.to_le_bytes());
        }
        for value in self.residual {
            write_slice(&mut bytes, &mut offset, &value.to_le_bytes());
        }
        debug_assert_eq!(offset, SCORE_TO_GO_RECORD_SIZE);
        bytes
    }

    pub fn from_bytes(bytes: &[u8; SCORE_TO_GO_RECORD_SIZE]) -> Self {
        let mut offset = 0;
        let position_bytes: [u8; RECORD_SIZE] = read_array(bytes, &mut offset);
        let mut current = [0; TARGET_DIM];
        for value in &mut current {
            *value = u16::from_le_bytes(read_array(bytes, &mut offset));
        }
        let mut residual = [0; TARGET_DIM];
        for value in &mut residual {
            *value = i16::from_le_bytes(read_array(bytes, &mut offset));
        }
        debug_assert_eq!(offset, SCORE_TO_GO_RECORD_SIZE);
        Self {
            position: PositionRecord::from_bytes(&position_bytes),
            current,
            residual,
        }
    }
}

pub struct ScoreToGoDatasetWriter {
    output: PathBuf,
    manifest_path: PathBuf,
    manifest: ScoreToGoDatasetManifest,
}

impl ScoreToGoDatasetWriter {
    pub fn open(config: &ScoreToGoDatasetConfig) -> Result<Self, DataError> {
        config.validate()?;
        fs::create_dir_all(&config.output)?;
        let manifest_path = config.output.join("dataset.json");
        let mut manifest = if manifest_path.exists() {
            if !config.resume {
                return Err(DataError::DatasetExists(config.output.clone()));
            }
            let manifest: ScoreToGoDatasetManifest =
                serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
            validate_resume(&manifest, config)?;
            validate_score_to_go_dataset(&config.output, &manifest)?;
            manifest
        } else {
            let now = unix_seconds()?;
            ScoreToGoDatasetManifest {
                schema_version: SCORE_TO_GO_DATASET_SCHEMA_VERSION,
                dataset_id: format!(
                    "score-to-go-{}-{}-{}",
                    config.teacher.strategy_id,
                    config.split.id(),
                    config.first_game_index
                ),
                feature_schema: SCORE_TO_GO_FEATURE_SCHEMA.to_owned(),
                target_schema: SCORE_TO_GO_TARGET_SCHEMA.to_owned(),
                record_size: SCORE_TO_GO_RECORD_SIZE,
                position_record_size: RECORD_SIZE,
                target_dim: TARGET_DIM,
                game: GameConfig::research_aaaaa(4)?,
                split: config.split,
                teacher: config.teacher.clone(),
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
        Ok(Self {
            output: config.output.clone(),
            manifest_path,
            manifest,
        })
    }

    pub fn manifest(&self) -> &ScoreToGoDatasetManifest {
        &self.manifest
    }

    pub fn append_shard(
        &mut self,
        first_game_index: u64,
        game_count: usize,
        records: &[ScoreToGoRecord],
    ) -> Result<(), DataError> {
        if game_count == 0 || records.is_empty() {
            return Err(DataError::InvalidConfig(
                "score-to-go shard must contain games and records",
            ));
        }
        let expected_first = self.manifest.first_game_index + self.manifest.completed_games as u64;
        if first_game_index != expected_first
            || self.manifest.completed_games + game_count > self.manifest.requested_games
        {
            return Err(DataError::InvalidConfig(
                "score-to-go shard game range is invalid",
            ));
        }
        for record in records {
            record.validate()?;
        }
        let shard_index = self.manifest.shards.len();
        let file_name = format!("shard-{shard_index:05}.stg");
        let path = self.output.join(&file_name);
        write_shard(
            &path,
            self.manifest.split,
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
        write_manifest_atomic(&self.manifest_path, &self.manifest)
    }
}

pub fn validate_score_to_go_dataset(
    root: &Path,
    manifest: &ScoreToGoDatasetManifest,
) -> Result<(), DataError> {
    if manifest.schema_version != SCORE_TO_GO_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != SCORE_TO_GO_FEATURE_SCHEMA
        || manifest.target_schema != SCORE_TO_GO_TARGET_SCHEMA
        || manifest.record_size != SCORE_TO_GO_RECORD_SIZE
        || manifest.position_record_size != RECORD_SIZE
        || manifest.target_dim != TARGET_DIM
        || manifest.game != GameConfig::research_aaaaa(4)?
    {
        return Err(DataError::InvalidManifest(
            "score-to-go schema identifiers do not match",
        ));
    }
    manifest.teacher.validate()?;
    let mut games = 0;
    let mut records = 0;
    for shard in &manifest.shards {
        let path = root.join(&shard.file);
        if fs::metadata(&path)?.len() != shard.byte_count {
            return Err(DataError::InvalidManifest(
                "score-to-go shard byte count mismatch",
            ));
        }
        if checksum_file(&path)? != shard.blake3 {
            return Err(DataError::ChecksumMismatch(path));
        }
        validate_shard_header(&path, manifest.split, shard)?;
        for record in read_shard_records(root, manifest.split, shard)? {
            record.validate()?;
        }
        games += shard.game_count;
        records += shard.record_count;
    }
    if games != manifest.completed_games || records != manifest.total_records {
        return Err(DataError::InvalidManifest(
            "score-to-go manifest totals do not match shards",
        ));
    }
    Ok(())
}

fn read_shard_records(
    root: &Path,
    split: DatasetSplit,
    shard: &ShardManifest,
) -> Result<Vec<ScoreToGoRecord>, DataError> {
    let path = root.join(&shard.file);
    validate_shard_header(&path, split, shard)?;
    let mut reader = BufReader::new(File::open(path)?);
    reader.seek(std::io::SeekFrom::Start(SCORE_TO_GO_HEADER_SIZE as u64))?;
    let mut records = Vec::with_capacity(shard.record_count);
    for _ in 0..shard.record_count {
        let mut bytes = [0; SCORE_TO_GO_RECORD_SIZE];
        reader.read_exact(&mut bytes)?;
        records.push(ScoreToGoRecord::from_bytes(&bytes));
    }
    Ok(records)
}

fn validate_resume(
    manifest: &ScoreToGoDatasetManifest,
    config: &ScoreToGoDatasetConfig,
) -> Result<(), DataError> {
    let current_provenance = collection_provenance()?;
    if manifest.schema_version != SCORE_TO_GO_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != SCORE_TO_GO_FEATURE_SCHEMA
        || manifest.target_schema != SCORE_TO_GO_TARGET_SCHEMA
        || manifest.record_size != SCORE_TO_GO_RECORD_SIZE
        || manifest.position_record_size != RECORD_SIZE
        || manifest.target_dim != TARGET_DIM
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

fn write_shard(
    path: &Path,
    split: DatasetSplit,
    first_game_index: u64,
    game_count: usize,
    records: &[ScoreToGoRecord],
) -> Result<(), DataError> {
    let temp_path = path.with_extension("stg.tmp");
    let mut writer = BufWriter::new(File::create(&temp_path)?);
    writer.write_all(SCORE_TO_GO_SHARD_MAGIC)?;
    writer.write_all(&SCORE_TO_GO_DATASET_SCHEMA_VERSION.to_le_bytes())?;
    writer.write_all(&(SCORE_TO_GO_HEADER_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(SCORE_TO_GO_RECORD_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(RECORD_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(TARGET_DIM as u16).to_le_bytes())?;
    writer.write_all(&(records.len() as u32).to_le_bytes())?;
    writer.write_all(&(game_count as u32).to_le_bytes())?;
    writer.write_all(&[split.code(), 4, 0, 0])?;
    writer.write_all(&first_game_index.to_le_bytes())?;
    writer.write_all(blake3::hash(SCORE_TO_GO_FEATURE_SCHEMA.as_bytes()).as_bytes())?;
    writer.write_all(blake3::hash(SCORE_TO_GO_TARGET_SCHEMA.as_bytes()).as_bytes())?;
    writer.write_all(&[0; 26])?;
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
    shard: &ShardManifest,
) -> Result<(), DataError> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut header = [0; SCORE_TO_GO_HEADER_SIZE];
    reader.read_exact(&mut header)?;
    if &header[..8] != SCORE_TO_GO_SHARD_MAGIC
        || u16::from_le_bytes([header[8], header[9]]) != SCORE_TO_GO_DATASET_SCHEMA_VERSION
        || u16::from_le_bytes([header[10], header[11]]) as usize != SCORE_TO_GO_HEADER_SIZE
        || u16::from_le_bytes([header[12], header[13]]) as usize != SCORE_TO_GO_RECORD_SIZE
        || u16::from_le_bytes([header[14], header[15]]) as usize != RECORD_SIZE
        || u16::from_le_bytes([header[16], header[17]]) as usize != TARGET_DIM
        || header[26] != split.code()
        || &header[38..70] != blake3::hash(SCORE_TO_GO_FEATURE_SCHEMA.as_bytes()).as_bytes()
        || &header[70..102] != blake3::hash(SCORE_TO_GO_TARGET_SCHEMA.as_bytes()).as_bytes()
    {
        return Err(DataError::InvalidShard("incompatible score-to-go header"));
    }
    let record_count =
        u32::from_le_bytes(header[18..22].try_into().expect("fixed header")) as usize;
    let game_count = u32::from_le_bytes(header[22..26].try_into().expect("fixed header")) as usize;
    let first_game_index = u64::from_le_bytes(header[30..38].try_into().expect("fixed header"));
    if record_count != shard.record_count
        || game_count != shard.game_count
        || first_game_index != shard.first_game_index
    {
        return Err(DataError::InvalidShard(
            "score-to-go header and manifest disagree",
        ));
    }
    let expected_size =
        SCORE_TO_GO_HEADER_SIZE as u64 + record_count as u64 * SCORE_TO_GO_RECORD_SIZE as u64;
    if fs::metadata(path)?.len() != expected_size {
        return Err(DataError::InvalidShard(
            "score-to-go shard file size does not match records",
        ));
    }
    Ok(())
}

fn score_components(score: ScoreBreakdown) -> [u16; TARGET_DIM] {
    let mut components = [0; TARGET_DIM];
    components[..5].copy_from_slice(&score.habitat);
    components[5..10].copy_from_slice(&score.wildlife);
    components[10] = score.nature_tokens;
    components
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameSeed, GameState, score_board};

    use super::*;

    #[test]
    fn record_round_trip_preserves_signed_target_identity() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(41),
        )
        .unwrap();
        let current = score_board(&game.boards()[0], game.config().scoring_cards);
        let mut final_score = current;
        let decreased = current
            .habitat
            .iter()
            .position(|value| *value > 0)
            .expect("a starter board has habitat score");
        final_score.habitat[decreased] -= 1;
        final_score.wildlife[1] += 4;
        let record = ScoreToGoRecord::new(
            PositionRecord::observe_for_seat(&game, 7, 0),
            current,
            final_score,
        )
        .unwrap();

        let decoded = ScoreToGoRecord::from_bytes(&record.to_bytes());

        assert_eq!(decoded, record);
        assert_eq!(decoded.residual[decreased], -1);
        assert_eq!(decoded.residual[6], 4);
        decoded.validate().unwrap();
    }
}
