use std::{
    collections::HashSet,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Seek, Write},
    path::{Path, PathBuf},
};

use cascadia_game::{GameConfig, GameSeed, PublicSupply, ScoreBreakdown};
use serde::{Deserialize, Serialize};

use super::public_supply::{PUBLIC_SUPPLY_SIZE, decode_public_supply, encode_public_supply};
use super::{
    CollectionProvenance, DataError, DatasetSplit, FEATURE_SCHEMA, PositionRecord, RECORD_SIZE,
    ShardManifest, TARGET_DIM, checksum_file, collection_provenance, collection_provenance_matches,
    read_array, unix_seconds, write_manifest_atomic, write_slice,
};

pub const COUNTERFACTUAL_VALUE_DATASET_SCHEMA_VERSION: u16 = 1;
pub const COUNTERFACTUAL_VALUE_FEATURE_SCHEMA: &str = FEATURE_SCHEMA;
pub const COUNTERFACTUAL_VALUE_TARGET_SCHEMA: &str = "public-redetermined-terminal-components-v1";
pub const COUNTERFACTUAL_VALUE_SHARD_MAGIC: &[u8; 8] = b"CSD2CFV\0";
pub const COUNTERFACTUAL_VALUE_HEADER_SIZE: usize = 160;
pub const COUNTERFACTUAL_VALUE_MAX_SAMPLES: usize = 16;
pub const COUNTERFACTUAL_VALUE_RECORD_SIZE: usize = RECORD_SIZE
    + TARGET_DIM * 4
    + PUBLIC_SUPPLY_SIZE
    + 2
    + COUNTERFACTUAL_VALUE_MAX_SAMPLES * 32
    + COUNTERFACTUAL_VALUE_MAX_SAMPLES * TARGET_DIM * 2;

#[derive(Debug, Clone)]
pub struct CounterfactualValueDatasetConfig {
    pub output: PathBuf,
    pub split: DatasetSplit,
    pub first_game_index: u64,
    pub games: usize,
    pub teacher: CounterfactualValueTeacherConfig,
    pub resume: bool,
}

impl CounterfactualValueDatasetConfig {
    fn validate(&self) -> Result<(), DataError> {
        if self.games == 0 {
            return Err(DataError::InvalidConfig("games must be positive"));
        }
        self.teacher.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CounterfactualValueTeacherConfig {
    pub strategy_id: String,
    pub immediate_candidates: usize,
    pub habitat_candidates: usize,
    pub determinizations: usize,
    pub greedy_plies: usize,
    pub samples_per_state: usize,
    pub sample_seed_domain: String,
}

impl CounterfactualValueTeacherConfig {
    fn validate(&self) -> Result<(), DataError> {
        if self.strategy_id.trim().is_empty()
            || self.immediate_candidates != 8
            || self.habitat_candidates != 6
            || self.determinizations != 4
            || self.greedy_plies != 4
            || !(1..=COUNTERFACTUAL_VALUE_MAX_SAMPLES).contains(&self.samples_per_state)
            || self.sample_seed_domain != "cascadia-v2-counterfactual-value-v1"
        {
            return Err(DataError::InvalidConfig(
                "counterfactual-value teacher contract is invalid",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CounterfactualValueDatasetManifest {
    pub schema_version: u16,
    pub dataset_id: String,
    pub feature_schema: String,
    pub target_schema: String,
    pub record_size: usize,
    pub position_record_size: usize,
    pub target_dim: usize,
    pub maximum_samples: usize,
    pub game: GameConfig,
    pub split: DatasetSplit,
    pub teacher: CounterfactualValueTeacherConfig,
    pub first_game_index: u64,
    pub requested_games: usize,
    pub completed_games: usize,
    pub total_records: usize,
    pub total_continuations: usize,
    pub collection_milliseconds: u64,
    pub created_unix_seconds: u64,
    pub updated_unix_seconds: u64,
    pub provenance: CollectionProvenance,
    pub shards: Vec<ShardManifest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CounterfactualValueRecord {
    pub position: PositionRecord,
    pub current: [u16; TARGET_DIM],
    pub factual_final: [u16; TARGET_DIM],
    pub public_supply: PublicSupply,
    pub sample_count: u8,
    pub sample_seeds: [GameSeed; COUNTERFACTUAL_VALUE_MAX_SAMPLES],
    pub sample_finals: [[u16; TARGET_DIM]; COUNTERFACTUAL_VALUE_MAX_SAMPLES],
}

impl CounterfactualValueRecord {
    pub fn new(
        mut position: PositionRecord,
        current: ScoreBreakdown,
        factual_final: ScoreBreakdown,
        public_supply: PublicSupply,
        samples: &[(GameSeed, ScoreBreakdown)],
    ) -> Result<Self, DataError> {
        if samples.is_empty() || samples.len() > COUNTERFACTUAL_VALUE_MAX_SAMPLES {
            return Err(DataError::InvalidConfig(
                "counterfactual-value sample count is invalid",
            ));
        }
        let current = score_components(current);
        let factual_final = score_components(factual_final);
        position.targets = factual_final;
        let mut sample_seeds = [GameSeed([0; 32]); COUNTERFACTUAL_VALUE_MAX_SAMPLES];
        let mut sample_finals = [[0; TARGET_DIM]; COUNTERFACTUAL_VALUE_MAX_SAMPLES];
        for (index, (seed, score)) in samples.iter().enumerate() {
            sample_seeds[index] = *seed;
            sample_finals[index] = score_components(*score);
        }
        let record = Self {
            position,
            current,
            factual_final,
            public_supply,
            sample_count: samples.len() as u8,
            sample_seeds,
            sample_finals,
        };
        record.validate(samples.len())?;
        Ok(record)
    }

    pub fn validate(&self, expected_samples: usize) -> Result<(), DataError> {
        if usize::from(self.sample_count) != expected_samples
            || !(1..=COUNTERFACTUAL_VALUE_MAX_SAMPLES).contains(&expected_samples)
            || self.position.targets != self.factual_final
            || self.position.turn >= 80
            || self.position.active_seat != self.position.turn % self.position.player_count
        {
            return Err(DataError::InvalidConfig(
                "counterfactual-value record metadata is invalid",
            ));
        }
        let expected_unseen_tiles = 81usize.saturating_sub(usize::from(self.position.turn));
        let unseen_tiles = self
            .public_supply
            .unseen_keystones_by_terrain
            .iter()
            .map(|value| usize::from(*value))
            .sum::<usize>()
            + self
                .public_supply
                .unseen_dual_terrain_pairs
                .iter()
                .map(|value| usize::from(*value))
                .sum::<usize>();
        if unseen_tiles != expected_unseen_tiles {
            return Err(DataError::InvalidConfig(
                "counterfactual-value public tile supply is invalid",
            ));
        }
        let mut seeds = HashSet::with_capacity(expected_samples);
        for index in 0..expected_samples {
            if !seeds.insert(self.sample_seeds[index].0) {
                return Err(DataError::InvalidConfig(
                    "counterfactual-value sample seeds are not unique",
                ));
            }
        }
        for index in expected_samples..COUNTERFACTUAL_VALUE_MAX_SAMPLES {
            if self.sample_seeds[index] != GameSeed([0; 32])
                || self.sample_finals[index] != [0; TARGET_DIM]
            {
                return Err(DataError::InvalidConfig(
                    "counterfactual-value unused sample slots are not zero",
                ));
            }
        }
        Ok(())
    }

    pub fn to_bytes(&self) -> [u8; COUNTERFACTUAL_VALUE_RECORD_SIZE] {
        let mut bytes = [0; COUNTERFACTUAL_VALUE_RECORD_SIZE];
        let mut offset = 0;
        write_slice(&mut bytes, &mut offset, &self.position.to_bytes());
        for value in self.current {
            write_slice(&mut bytes, &mut offset, &value.to_le_bytes());
        }
        for value in self.factual_final {
            write_slice(&mut bytes, &mut offset, &value.to_le_bytes());
        }
        write_slice(
            &mut bytes,
            &mut offset,
            &encode_public_supply(self.public_supply),
        );
        write_slice(&mut bytes, &mut offset, &[self.sample_count, 0]);
        for seed in self.sample_seeds {
            write_slice(&mut bytes, &mut offset, &seed.0);
        }
        for score in self.sample_finals {
            for value in score {
                write_slice(&mut bytes, &mut offset, &value.to_le_bytes());
            }
        }
        debug_assert_eq!(offset, COUNTERFACTUAL_VALUE_RECORD_SIZE);
        bytes
    }

    pub fn from_bytes(bytes: &[u8; COUNTERFACTUAL_VALUE_RECORD_SIZE]) -> Self {
        let mut offset = 0;
        let position_bytes: [u8; RECORD_SIZE] = read_array(bytes, &mut offset);
        let current = read_components(bytes, &mut offset);
        let factual_final = read_components(bytes, &mut offset);
        let public_supply = decode_public_supply(read_array(bytes, &mut offset));
        let [sample_count, _reserved] = read_array(bytes, &mut offset);
        let mut sample_seeds = [GameSeed([0; 32]); COUNTERFACTUAL_VALUE_MAX_SAMPLES];
        for seed in &mut sample_seeds {
            seed.0 = read_array(bytes, &mut offset);
        }
        let mut sample_finals = [[0; TARGET_DIM]; COUNTERFACTUAL_VALUE_MAX_SAMPLES];
        for score in &mut sample_finals {
            *score = read_components(bytes, &mut offset);
        }
        debug_assert_eq!(offset, COUNTERFACTUAL_VALUE_RECORD_SIZE);
        Self {
            position: PositionRecord::from_bytes(&position_bytes),
            current,
            factual_final,
            public_supply,
            sample_count,
            sample_seeds,
            sample_finals,
        }
    }
}

pub struct CounterfactualValueDatasetWriter {
    output: PathBuf,
    manifest_path: PathBuf,
    manifest: CounterfactualValueDatasetManifest,
}

impl CounterfactualValueDatasetWriter {
    pub fn open(config: &CounterfactualValueDatasetConfig) -> Result<Self, DataError> {
        config.validate()?;
        fs::create_dir_all(&config.output)?;
        let manifest_path = config.output.join("dataset.json");
        let mut manifest = if manifest_path.exists() {
            if !config.resume {
                return Err(DataError::DatasetExists(config.output.clone()));
            }
            let manifest: CounterfactualValueDatasetManifest =
                serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
            validate_resume(&manifest, config)?;
            validate_counterfactual_value_dataset(&config.output, &manifest)?;
            manifest
        } else {
            let now = unix_seconds()?;
            CounterfactualValueDatasetManifest {
                schema_version: COUNTERFACTUAL_VALUE_DATASET_SCHEMA_VERSION,
                dataset_id: format!(
                    "counterfactual-value-{}-r{}-{}-{}",
                    config.teacher.strategy_id,
                    config.teacher.samples_per_state,
                    config.split.id(),
                    config.first_game_index
                ),
                feature_schema: COUNTERFACTUAL_VALUE_FEATURE_SCHEMA.to_owned(),
                target_schema: COUNTERFACTUAL_VALUE_TARGET_SCHEMA.to_owned(),
                record_size: COUNTERFACTUAL_VALUE_RECORD_SIZE,
                position_record_size: RECORD_SIZE,
                target_dim: TARGET_DIM,
                maximum_samples: COUNTERFACTUAL_VALUE_MAX_SAMPLES,
                game: GameConfig::research_aaaaa(4)?,
                split: config.split,
                teacher: config.teacher.clone(),
                first_game_index: config.first_game_index,
                requested_games: config.games,
                completed_games: 0,
                total_records: 0,
                total_continuations: 0,
                collection_milliseconds: 0,
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

    pub fn manifest(&self) -> &CounterfactualValueDatasetManifest {
        &self.manifest
    }

    pub fn append_game(
        &mut self,
        game_index: u64,
        records: &[CounterfactualValueRecord],
    ) -> Result<(), DataError> {
        let expected = self.manifest.first_game_index + self.manifest.completed_games as u64;
        if game_index != expected
            || self.manifest.completed_games >= self.manifest.requested_games
            || records.len() != 80
        {
            return Err(DataError::InvalidConfig(
                "counterfactual-value game shard range is invalid",
            ));
        }
        validate_game_records(records, game_index, self.manifest.teacher.samples_per_state)?;
        let shard_index = self.manifest.shards.len();
        let file_name = format!("shard-{shard_index:05}.cfv");
        let path = self.output.join(&file_name);
        write_shard(
            &path,
            self.manifest.split,
            &self.manifest.teacher,
            game_index,
            records,
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
        self.manifest.total_continuations +=
            records.len() * self.manifest.teacher.samples_per_state;
        self.manifest.updated_unix_seconds = unix_seconds()?;
        write_manifest_atomic(&self.manifest_path, &self.manifest)
    }

    pub fn set_collection_milliseconds(&mut self, milliseconds: u64) -> Result<(), DataError> {
        if milliseconds < self.manifest.collection_milliseconds {
            return Err(DataError::InvalidConfig(
                "counterfactual-value collection time cannot decrease",
            ));
        }
        self.manifest.collection_milliseconds = milliseconds;
        self.manifest.updated_unix_seconds = unix_seconds()?;
        write_manifest_atomic(&self.manifest_path, &self.manifest)
    }
}

pub fn validate_counterfactual_value_dataset(
    root: &Path,
    manifest: &CounterfactualValueDatasetManifest,
) -> Result<(), DataError> {
    if manifest.schema_version != COUNTERFACTUAL_VALUE_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != COUNTERFACTUAL_VALUE_FEATURE_SCHEMA
        || manifest.target_schema != COUNTERFACTUAL_VALUE_TARGET_SCHEMA
        || manifest.record_size != COUNTERFACTUAL_VALUE_RECORD_SIZE
        || manifest.position_record_size != RECORD_SIZE
        || manifest.target_dim != TARGET_DIM
        || manifest.maximum_samples != COUNTERFACTUAL_VALUE_MAX_SAMPLES
        || manifest.game != GameConfig::research_aaaaa(4)?
    {
        return Err(DataError::InvalidManifest(
            "counterfactual-value schema identifiers do not match",
        ));
    }
    manifest.teacher.validate()?;
    let mut games = 0;
    let mut records = 0;
    for shard in &manifest.shards {
        let path = root.join(&shard.file);
        if fs::metadata(&path)?.len() != shard.byte_count {
            return Err(DataError::InvalidManifest(
                "counterfactual-value shard byte count mismatch",
            ));
        }
        if checksum_file(&path)? != shard.blake3 {
            return Err(DataError::ChecksumMismatch(path));
        }
        let shard_records = read_counterfactual_value_shard_records(
            root,
            manifest.split,
            &manifest.teacher,
            shard,
        )?;
        validate_game_records(
            &shard_records,
            shard.first_game_index,
            manifest.teacher.samples_per_state,
        )?;
        games += shard.game_count;
        records += shard.record_count;
    }
    if games != manifest.completed_games
        || records != manifest.total_records
        || manifest.total_continuations
            != manifest.total_records * manifest.teacher.samples_per_state
    {
        return Err(DataError::InvalidManifest(
            "counterfactual-value manifest totals do not match shards",
        ));
    }
    Ok(())
}

pub fn read_counterfactual_value_shard_records(
    root: &Path,
    split: DatasetSplit,
    teacher: &CounterfactualValueTeacherConfig,
    shard: &ShardManifest,
) -> Result<Vec<CounterfactualValueRecord>, DataError> {
    let path = root.join(&shard.file);
    validate_shard_header(&path, split, teacher, shard)?;
    let mut reader = BufReader::new(File::open(path)?);
    reader.seek(std::io::SeekFrom::Start(
        COUNTERFACTUAL_VALUE_HEADER_SIZE as u64,
    ))?;
    let mut records = Vec::with_capacity(shard.record_count);
    for _ in 0..shard.record_count {
        let mut bytes = [0; COUNTERFACTUAL_VALUE_RECORD_SIZE];
        reader.read_exact(&mut bytes)?;
        records.push(CounterfactualValueRecord::from_bytes(&bytes));
    }
    Ok(records)
}

fn validate_game_records(
    records: &[CounterfactualValueRecord],
    game_index: u64,
    samples_per_state: usize,
) -> Result<(), DataError> {
    if records.len() != 80 {
        return Err(DataError::InvalidConfig(
            "counterfactual-value game must contain 80 states",
        ));
    }
    for (turn, record) in records.iter().enumerate() {
        record.validate(samples_per_state)?;
        if record.position.game_index != game_index || usize::from(record.position.turn) != turn {
            return Err(DataError::InvalidConfig(
                "counterfactual-value game sequence is invalid",
            ));
        }
    }
    Ok(())
}

fn validate_resume(
    manifest: &CounterfactualValueDatasetManifest,
    config: &CounterfactualValueDatasetConfig,
) -> Result<(), DataError> {
    let current_provenance = collection_provenance()?;
    if manifest.schema_version != COUNTERFACTUAL_VALUE_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != COUNTERFACTUAL_VALUE_FEATURE_SCHEMA
        || manifest.target_schema != COUNTERFACTUAL_VALUE_TARGET_SCHEMA
        || manifest.record_size != COUNTERFACTUAL_VALUE_RECORD_SIZE
        || manifest.position_record_size != RECORD_SIZE
        || manifest.target_dim != TARGET_DIM
        || manifest.maximum_samples != COUNTERFACTUAL_VALUE_MAX_SAMPLES
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
    teacher: &CounterfactualValueTeacherConfig,
    game_index: u64,
    records: &[CounterfactualValueRecord],
) -> Result<(), DataError> {
    let temp_path = path.with_extension("cfv.tmp");
    let teacher_hash = blake3::hash(&serde_json::to_vec(teacher)?);
    let mut writer = BufWriter::new(File::create(&temp_path)?);
    writer.write_all(COUNTERFACTUAL_VALUE_SHARD_MAGIC)?;
    writer.write_all(&COUNTERFACTUAL_VALUE_DATASET_SCHEMA_VERSION.to_le_bytes())?;
    writer.write_all(&(COUNTERFACTUAL_VALUE_HEADER_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(COUNTERFACTUAL_VALUE_RECORD_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(RECORD_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(TARGET_DIM as u16).to_le_bytes())?;
    writer.write_all(&(COUNTERFACTUAL_VALUE_MAX_SAMPLES as u16).to_le_bytes())?;
    writer.write_all(&(records.len() as u32).to_le_bytes())?;
    writer.write_all(&1u32.to_le_bytes())?;
    writer.write_all(&[split.code(), 4, teacher.samples_per_state as u8, 0])?;
    writer.write_all(&game_index.to_le_bytes())?;
    writer.write_all(blake3::hash(COUNTERFACTUAL_VALUE_FEATURE_SCHEMA.as_bytes()).as_bytes())?;
    writer.write_all(blake3::hash(COUNTERFACTUAL_VALUE_TARGET_SCHEMA.as_bytes()).as_bytes())?;
    writer.write_all(teacher_hash.as_bytes())?;
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
    teacher: &CounterfactualValueTeacherConfig,
    shard: &ShardManifest,
) -> Result<(), DataError> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut header = [0; COUNTERFACTUAL_VALUE_HEADER_SIZE];
    reader.read_exact(&mut header)?;
    let teacher_hash = blake3::hash(&serde_json::to_vec(teacher)?);
    if &header[..8] != COUNTERFACTUAL_VALUE_SHARD_MAGIC
        || u16::from_le_bytes([header[8], header[9]]) != COUNTERFACTUAL_VALUE_DATASET_SCHEMA_VERSION
        || u16::from_le_bytes([header[10], header[11]]) as usize != COUNTERFACTUAL_VALUE_HEADER_SIZE
        || u16::from_le_bytes([header[12], header[13]]) as usize != COUNTERFACTUAL_VALUE_RECORD_SIZE
        || u16::from_le_bytes([header[14], header[15]]) as usize != RECORD_SIZE
        || u16::from_le_bytes([header[16], header[17]]) as usize != TARGET_DIM
        || u16::from_le_bytes([header[18], header[19]]) as usize != COUNTERFACTUAL_VALUE_MAX_SAMPLES
        || header[28] != split.code()
        || header[29] != 4
        || usize::from(header[30]) != teacher.samples_per_state
        || &header[40..72]
            != blake3::hash(COUNTERFACTUAL_VALUE_FEATURE_SCHEMA.as_bytes()).as_bytes()
        || &header[72..104]
            != blake3::hash(COUNTERFACTUAL_VALUE_TARGET_SCHEMA.as_bytes()).as_bytes()
        || &header[104..136] != teacher_hash.as_bytes()
    {
        return Err(DataError::InvalidShard(
            "incompatible counterfactual-value header",
        ));
    }
    let record_count =
        u32::from_le_bytes(header[20..24].try_into().expect("fixed header")) as usize;
    let game_count = u32::from_le_bytes(header[24..28].try_into().expect("fixed header")) as usize;
    let first_game_index = u64::from_le_bytes(header[32..40].try_into().expect("fixed header"));
    if record_count != shard.record_count
        || game_count != 1
        || shard.game_count != 1
        || first_game_index != shard.first_game_index
    {
        return Err(DataError::InvalidShard(
            "counterfactual-value header and manifest disagree",
        ));
    }
    let expected_size = COUNTERFACTUAL_VALUE_HEADER_SIZE as u64
        + record_count as u64 * COUNTERFACTUAL_VALUE_RECORD_SIZE as u64;
    if fs::metadata(path)?.len() != expected_size {
        return Err(DataError::InvalidShard(
            "counterfactual-value shard size does not match records",
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

fn read_components(
    bytes: &[u8; COUNTERFACTUAL_VALUE_RECORD_SIZE],
    offset: &mut usize,
) -> [u16; TARGET_DIM] {
    std::array::from_fn(|_| u16::from_le_bytes(read_array(bytes, offset)))
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameSeed, GameState, score_board};

    use super::*;

    #[test]
    fn counterfactual_record_round_trip_preserves_raw_samples() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(47),
        )
        .unwrap();
        let current = score_board(&game.boards()[0], game.config().scoring_cards);
        let mut left = current;
        left.wildlife[0] += 3;
        let mut right = current;
        right.habitat[2] += 2;
        let record = CounterfactualValueRecord::new(
            PositionRecord::observe_for_seat(&game, 11, 0),
            current,
            left,
            game.public_supply(),
            &[
                (GameSeed::from_u64(100), left),
                (GameSeed::from_u64(101), right),
            ],
        )
        .unwrap();

        let decoded = CounterfactualValueRecord::from_bytes(&record.to_bytes());

        assert_eq!(decoded, record);
        decoded.validate(2).unwrap();
    }
}
