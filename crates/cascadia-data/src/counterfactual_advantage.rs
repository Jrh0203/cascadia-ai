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
    ACTION_POSITION_RECORD_SIZE, ActionPositionRecord, CollectionProvenance, DataError,
    DatasetSplit, PositionRecord, RECORD_SIZE, ShardManifest, TARGET_DIM, checksum_file,
    collection_provenance, collection_provenance_matches, read_array, unix_seconds,
    write_manifest_atomic, write_slice,
};

pub const COUNTERFACTUAL_ADVANTAGE_DATASET_SCHEMA_VERSION: u16 = 1;
pub const COUNTERFACTUAL_ADVANTAGE_FEATURE_SCHEMA: &str =
    "grouped-observable-action-afterstates-with-public-supply-v1";
pub const COUNTERFACTUAL_ADVANTAGE_TARGET_SCHEMA: &str =
    "shared-public-redetermined-centered-terminal-components-v1";
pub const COUNTERFACTUAL_ADVANTAGE_SHARD_MAGIC: &[u8; 8] = b"CSD2CFA\0";
pub const COUNTERFACTUAL_ADVANTAGE_HEADER_SIZE: usize = 160;
pub const COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES: usize = 4;
pub const COUNTERFACTUAL_ADVANTAGE_MAX_SAMPLES: usize = 16;
pub const COUNTERFACTUAL_ADVANTAGE_NEAREST_SELECTION: &str = "selected-nearest-three-v1";
pub const COUNTERFACTUAL_ADVANTAGE_STRATIFIED_SELECTION: &str = "selected-high-median-low-v1";
pub const COUNTERFACTUAL_ADVANTAGE_STABILIZATION_CONDITIONING: &str =
    "reject-unstable-market-trajectories-v1";
pub const COUNTERFACTUAL_ADVANTAGE_CANDIDATE_SIZE: usize =
    40 + ACTION_POSITION_RECORD_SIZE + COUNTERFACTUAL_ADVANTAGE_MAX_SAMPLES * TARGET_DIM * 2;
pub const COUNTERFACTUAL_ADVANTAGE_RECORD_SIZE: usize = 16
    + TARGET_DIM * 2
    + PUBLIC_SUPPLY_SIZE
    + RECORD_SIZE
    + COUNTERFACTUAL_ADVANTAGE_MAX_SAMPLES * 32
    + COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES * COUNTERFACTUAL_ADVANTAGE_CANDIDATE_SIZE;

#[derive(Debug, Clone)]
pub struct CounterfactualAdvantageDatasetConfig {
    pub output: PathBuf,
    pub split: DatasetSplit,
    pub first_game_index: u64,
    pub games: usize,
    pub teacher: CounterfactualAdvantageTeacherConfig,
    pub resume: bool,
}

impl CounterfactualAdvantageDatasetConfig {
    fn validate(&self) -> Result<(), DataError> {
        if self.games == 0 {
            return Err(DataError::InvalidConfig("games must be positive"));
        }
        self.teacher.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CounterfactualAdvantageTeacherConfig {
    pub strategy_id: String,
    pub immediate_candidates: usize,
    pub habitat_candidates: usize,
    pub determinizations: usize,
    pub greedy_plies: usize,
    pub candidate_count: usize,
    pub groups_per_game: usize,
    pub samples_per_candidate: usize,
    pub sample_seed_domain: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_selection: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stabilization_conditioning: Option<String>,
}

impl CounterfactualAdvantageTeacherConfig {
    fn validate(&self) -> Result<(), DataError> {
        let candidate_selection = self.candidate_selection_id();
        if self.strategy_id.trim().is_empty()
            || self.immediate_candidates != 8
            || self.habitat_candidates != 6
            || self.determinizations != 4
            || self.greedy_plies != 4
            || self.candidate_count != COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES
            || self.groups_per_game == 0
            || self.groups_per_game > 80
            || !80usize.is_multiple_of(self.groups_per_game)
            || !(1..=COUNTERFACTUAL_ADVANTAGE_MAX_SAMPLES).contains(&self.samples_per_candidate)
            || self.sample_seed_domain != "cascadia-v2-counterfactual-advantage-v1"
            || !matches!(
                candidate_selection,
                COUNTERFACTUAL_ADVANTAGE_NEAREST_SELECTION
                    | COUNTERFACTUAL_ADVANTAGE_STRATIFIED_SELECTION
            )
            || !matches!(
                self.stabilization_conditioning.as_deref(),
                None | Some(COUNTERFACTUAL_ADVANTAGE_STABILIZATION_CONDITIONING)
            )
        {
            return Err(DataError::InvalidConfig(
                "counterfactual-advantage teacher contract is invalid",
            ));
        }
        Ok(())
    }

    pub fn decision_stride(&self) -> usize {
        80 / self.groups_per_game
    }

    pub fn candidate_selection_id(&self) -> &str {
        self.candidate_selection
            .as_deref()
            .unwrap_or(COUNTERFACTUAL_ADVANTAGE_NEAREST_SELECTION)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CounterfactualAdvantageDatasetManifest {
    pub schema_version: u16,
    pub dataset_id: String,
    pub feature_schema: String,
    pub target_schema: String,
    pub record_size: usize,
    pub action_position_record_size: usize,
    pub target_dim: usize,
    pub maximum_candidates: usize,
    pub maximum_samples: usize,
    pub game: GameConfig,
    pub split: DatasetSplit,
    pub teacher: CounterfactualAdvantageTeacherConfig,
    pub first_game_index: u64,
    pub requested_games: usize,
    pub completed_games: usize,
    pub total_groups: usize,
    pub total_candidates: usize,
    pub total_continuations: usize,
    pub collection_milliseconds: u64,
    pub created_unix_seconds: u64,
    pub updated_unix_seconds: u64,
    pub provenance: CollectionProvenance,
    pub shards: Vec<ShardManifest>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CounterfactualAdvantageCandidate {
    pub action_hash: [u8; 32],
    pub shallow_mean: f32,
    pub shallow_stddev: f32,
    pub input: ActionPositionRecord,
    pub sample_finals: [[u16; TARGET_DIM]; COUNTERFACTUAL_ADVANTAGE_MAX_SAMPLES],
}

impl CounterfactualAdvantageCandidate {
    pub fn new(
        action_hash: [u8; 32],
        shallow_mean: f64,
        shallow_stddev: f64,
        input: ActionPositionRecord,
        samples: &[ScoreBreakdown],
    ) -> Result<Self, DataError> {
        if samples.is_empty() || samples.len() > COUNTERFACTUAL_ADVANTAGE_MAX_SAMPLES {
            return Err(DataError::InvalidConfig(
                "counterfactual-advantage candidate sample count is invalid",
            ));
        }
        let mut sample_finals = [[0; TARGET_DIM]; COUNTERFACTUAL_ADVANTAGE_MAX_SAMPLES];
        for (index, score) in samples.iter().copied().enumerate() {
            sample_finals[index] = score_components(score);
        }
        let candidate = Self {
            action_hash,
            shallow_mean: shallow_mean as f32,
            shallow_stddev: shallow_stddev as f32,
            input,
            sample_finals,
        };
        candidate.validate(samples.len())?;
        Ok(candidate)
    }

    fn validate(&self, sample_count: usize) -> Result<(), DataError> {
        if self.action_hash == [0; 32]
            || !self.shallow_mean.is_finite()
            || !self.shallow_stddev.is_finite()
            || self.shallow_stddev < 0.0
            || self.input.action.immediate_rank == 0
            || self.input.position.targets != [0; TARGET_DIM]
            || self.sample_finals[sample_count..]
                .iter()
                .any(|score| *score != [0; TARGET_DIM])
        {
            return Err(DataError::InvalidConfig(
                "counterfactual-advantage candidate is invalid",
            ));
        }
        Ok(())
    }

    fn to_bytes(&self) -> [u8; COUNTERFACTUAL_ADVANTAGE_CANDIDATE_SIZE] {
        let mut bytes = [0; COUNTERFACTUAL_ADVANTAGE_CANDIDATE_SIZE];
        let mut offset = 0;
        write_slice(&mut bytes, &mut offset, &self.action_hash);
        write_slice(&mut bytes, &mut offset, &self.shallow_mean.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.shallow_stddev.to_le_bytes());
        write_slice(&mut bytes, &mut offset, &self.input.to_bytes());
        for score in self.sample_finals {
            for value in score {
                write_slice(&mut bytes, &mut offset, &value.to_le_bytes());
            }
        }
        debug_assert_eq!(offset, COUNTERFACTUAL_ADVANTAGE_CANDIDATE_SIZE);
        bytes
    }

    fn from_bytes(bytes: &[u8; COUNTERFACTUAL_ADVANTAGE_CANDIDATE_SIZE]) -> Self {
        let mut offset = 0;
        let action_hash = read_array(bytes, &mut offset);
        let shallow_mean = f32::from_le_bytes(read_array(bytes, &mut offset));
        let shallow_stddev = f32::from_le_bytes(read_array(bytes, &mut offset));
        let input = ActionPositionRecord::from_bytes(&read_array::<ACTION_POSITION_RECORD_SIZE>(
            bytes,
            &mut offset,
        ));
        let mut sample_finals = [[0; TARGET_DIM]; COUNTERFACTUAL_ADVANTAGE_MAX_SAMPLES];
        for score in &mut sample_finals {
            for value in score {
                *value = u16::from_le_bytes(read_array(bytes, &mut offset));
            }
        }
        debug_assert_eq!(offset, COUNTERFACTUAL_ADVANTAGE_CANDIDATE_SIZE);
        Self {
            action_hash,
            shallow_mean,
            shallow_stddev,
            input,
            sample_finals,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CounterfactualAdvantageRecord {
    pub group_id: u64,
    pub selected_index: u8,
    pub sample_count: u8,
    pub current: [u16; TARGET_DIM],
    pub public_supply: PublicSupply,
    pub parent: PositionRecord,
    pub sample_seeds: [GameSeed; COUNTERFACTUAL_ADVANTAGE_MAX_SAMPLES],
    pub candidates: [CounterfactualAdvantageCandidate; COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES],
}

impl CounterfactualAdvantageRecord {
    pub fn new(
        group_id: u64,
        selected_index: usize,
        current: ScoreBreakdown,
        public_supply: PublicSupply,
        parent: PositionRecord,
        sample_seeds: &[GameSeed],
        candidates: Vec<CounterfactualAdvantageCandidate>,
    ) -> Result<Self, DataError> {
        if group_id == 0
            || selected_index >= COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES
            || sample_seeds.is_empty()
            || sample_seeds.len() > COUNTERFACTUAL_ADVANTAGE_MAX_SAMPLES
            || candidates.len() != COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES
        {
            return Err(DataError::InvalidConfig(
                "counterfactual-advantage group shape is invalid",
            ));
        }
        let mut stored_seeds = [GameSeed([0; 32]); COUNTERFACTUAL_ADVANTAGE_MAX_SAMPLES];
        stored_seeds[..sample_seeds.len()].copy_from_slice(sample_seeds);
        let record = Self {
            group_id,
            selected_index: selected_index as u8,
            sample_count: sample_seeds.len() as u8,
            current: score_components(current),
            public_supply,
            parent,
            sample_seeds: stored_seeds,
            candidates: candidates.try_into().map_err(|_| {
                DataError::InvalidConfig("counterfactual-advantage candidate count is invalid")
            })?,
        };
        record.validate(sample_seeds.len())?;
        Ok(record)
    }

    pub fn validate(&self, expected_samples: usize) -> Result<(), DataError> {
        if self.group_id == 0
            || usize::from(self.selected_index) >= COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES
            || usize::from(self.sample_count) != expected_samples
            || !(1..=COUNTERFACTUAL_ADVANTAGE_MAX_SAMPLES).contains(&expected_samples)
            || self.parent.targets != [0; TARGET_DIM]
            || self.parent.player_count != 4
            || self.parent.total_turns != 80
            || self.parent.turn >= 80
            || self.parent.active_seat != self.parent.turn % 4
        {
            return Err(DataError::InvalidConfig(
                "counterfactual-advantage group metadata is invalid",
            ));
        }
        let expected_unseen_tiles = 81usize.saturating_sub(usize::from(self.parent.turn));
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
                "counterfactual-advantage public tile supply is invalid",
            ));
        }
        let mut seeds = HashSet::with_capacity(expected_samples);
        for seed in &self.sample_seeds[..expected_samples] {
            if *seed == GameSeed([0; 32]) || !seeds.insert(seed.0) {
                return Err(DataError::InvalidConfig(
                    "counterfactual-advantage sample seeds are invalid",
                ));
            }
        }
        if self.sample_seeds[expected_samples..]
            .iter()
            .any(|seed| *seed != GameSeed([0; 32]))
        {
            return Err(DataError::InvalidConfig(
                "counterfactual-advantage unused sample seeds are not zero",
            ));
        }
        let mut actions = HashSet::with_capacity(COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES);
        for candidate in &self.candidates {
            candidate.validate(expected_samples)?;
            if !actions.insert(candidate.action_hash)
                || candidate.input.position.game_index != self.parent.game_index
                || candidate.input.position.turn != self.parent.turn + 1
                || candidate.input.position.active_seat != self.parent.active_seat
            {
                return Err(DataError::InvalidConfig(
                    "counterfactual-advantage candidate group is inconsistent",
                ));
            }
        }
        Ok(())
    }

    pub fn to_bytes(&self) -> [u8; COUNTERFACTUAL_ADVANTAGE_RECORD_SIZE] {
        let mut bytes = [0; COUNTERFACTUAL_ADVANTAGE_RECORD_SIZE];
        let mut offset = 0;
        write_slice(&mut bytes, &mut offset, &self.group_id.to_le_bytes());
        write_slice(
            &mut bytes,
            &mut offset,
            &[
                self.selected_index,
                COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES as u8,
                self.sample_count,
                0,
                0,
                0,
                0,
                0,
            ],
        );
        for value in self.current {
            write_slice(&mut bytes, &mut offset, &value.to_le_bytes());
        }
        write_slice(
            &mut bytes,
            &mut offset,
            &encode_public_supply(self.public_supply),
        );
        write_slice(&mut bytes, &mut offset, &self.parent.to_bytes());
        for seed in self.sample_seeds {
            write_slice(&mut bytes, &mut offset, &seed.0);
        }
        for candidate in &self.candidates {
            write_slice(&mut bytes, &mut offset, &candidate.to_bytes());
        }
        debug_assert_eq!(offset, COUNTERFACTUAL_ADVANTAGE_RECORD_SIZE);
        bytes
    }

    pub fn from_bytes(bytes: &[u8; COUNTERFACTUAL_ADVANTAGE_RECORD_SIZE]) -> Self {
        let mut offset = 0;
        let group_id = u64::from_le_bytes(read_array(bytes, &mut offset));
        let [
            selected_index,
            _candidate_count,
            sample_count,
            _,
            _,
            _,
            _,
            _,
        ] = read_array(bytes, &mut offset);
        let current = std::array::from_fn(|_| u16::from_le_bytes(read_array(bytes, &mut offset)));
        let public_supply = decode_public_supply(read_array(bytes, &mut offset));
        let parent = PositionRecord::from_bytes(&read_array::<RECORD_SIZE>(bytes, &mut offset));
        let mut sample_seeds = [GameSeed([0; 32]); COUNTERFACTUAL_ADVANTAGE_MAX_SAMPLES];
        for seed in &mut sample_seeds {
            seed.0 = read_array(bytes, &mut offset);
        }
        let candidates = std::array::from_fn(|_| {
            CounterfactualAdvantageCandidate::from_bytes(&read_array::<
                COUNTERFACTUAL_ADVANTAGE_CANDIDATE_SIZE,
            >(bytes, &mut offset))
        });
        debug_assert_eq!(offset, COUNTERFACTUAL_ADVANTAGE_RECORD_SIZE);
        Self {
            group_id,
            selected_index,
            sample_count,
            current,
            public_supply,
            parent,
            sample_seeds,
            candidates,
        }
    }
}

pub struct CounterfactualAdvantageDatasetWriter {
    output: PathBuf,
    manifest_path: PathBuf,
    manifest: CounterfactualAdvantageDatasetManifest,
}

impl CounterfactualAdvantageDatasetWriter {
    pub fn open(config: &CounterfactualAdvantageDatasetConfig) -> Result<Self, DataError> {
        config.validate()?;
        fs::create_dir_all(&config.output)?;
        let manifest_path = config.output.join("dataset.json");
        let mut manifest = if manifest_path.exists() {
            if !config.resume {
                return Err(DataError::DatasetExists(config.output.clone()));
            }
            let manifest: CounterfactualAdvantageDatasetManifest =
                serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
            validate_resume(&manifest, config)?;
            validate_counterfactual_advantage_dataset(&config.output, &manifest)?;
            manifest
        } else {
            let now = unix_seconds()?;
            CounterfactualAdvantageDatasetManifest {
                schema_version: COUNTERFACTUAL_ADVANTAGE_DATASET_SCHEMA_VERSION,
                dataset_id: counterfactual_advantage_dataset_id(config),
                feature_schema: COUNTERFACTUAL_ADVANTAGE_FEATURE_SCHEMA.to_owned(),
                target_schema: COUNTERFACTUAL_ADVANTAGE_TARGET_SCHEMA.to_owned(),
                record_size: COUNTERFACTUAL_ADVANTAGE_RECORD_SIZE,
                action_position_record_size: ACTION_POSITION_RECORD_SIZE,
                target_dim: TARGET_DIM,
                maximum_candidates: COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES,
                maximum_samples: COUNTERFACTUAL_ADVANTAGE_MAX_SAMPLES,
                game: GameConfig::research_aaaaa(4)?,
                split: config.split,
                teacher: config.teacher.clone(),
                first_game_index: config.first_game_index,
                requested_games: config.games,
                completed_games: 0,
                total_groups: 0,
                total_candidates: 0,
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

    pub fn manifest(&self) -> &CounterfactualAdvantageDatasetManifest {
        &self.manifest
    }

    pub fn append_game(
        &mut self,
        game_index: u64,
        records: &[CounterfactualAdvantageRecord],
    ) -> Result<(), DataError> {
        let expected = self.manifest.first_game_index + self.manifest.completed_games as u64;
        if game_index != expected
            || self.manifest.completed_games >= self.manifest.requested_games
            || records.len() != self.manifest.teacher.groups_per_game
        {
            return Err(DataError::InvalidConfig(
                "counterfactual-advantage game shard range is invalid",
            ));
        }
        validate_game_records(records, game_index, &self.manifest.teacher)?;
        let shard_index = self.manifest.shards.len();
        let file_name = format!("shard-{shard_index:05}.cfa");
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
        self.manifest.total_groups += records.len();
        self.manifest.total_candidates += records.len() * COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES;
        self.manifest.total_continuations += records.len()
            * COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES
            * self.manifest.teacher.samples_per_candidate;
        self.manifest.updated_unix_seconds = unix_seconds()?;
        write_manifest_atomic(&self.manifest_path, &self.manifest)
    }

    pub fn set_collection_milliseconds(&mut self, milliseconds: u64) -> Result<(), DataError> {
        if milliseconds < self.manifest.collection_milliseconds {
            return Err(DataError::InvalidConfig(
                "counterfactual-advantage collection time cannot decrease",
            ));
        }
        self.manifest.collection_milliseconds = milliseconds;
        self.manifest.updated_unix_seconds = unix_seconds()?;
        write_manifest_atomic(&self.manifest_path, &self.manifest)
    }
}

fn counterfactual_advantage_dataset_id(config: &CounterfactualAdvantageDatasetConfig) -> String {
    let selection = config.teacher.candidate_selection_id();
    let selection_suffix = if selection == COUNTERFACTUAL_ADVANTAGE_NEAREST_SELECTION {
        String::new()
    } else {
        format!("-{selection}")
    };
    let conditioning_suffix = config
        .teacher
        .stabilization_conditioning
        .as_deref()
        .map_or_else(String::new, |conditioning| format!("-{conditioning}"));
    format!(
        "counterfactual-advantage-{}{}{}-k{}-g{}-r{}-{}-{}",
        config.teacher.strategy_id,
        selection_suffix,
        conditioning_suffix,
        config.teacher.candidate_count,
        config.teacher.groups_per_game,
        config.teacher.samples_per_candidate,
        config.split.id(),
        config.first_game_index,
    )
}

pub fn validate_counterfactual_advantage_dataset(
    root: &Path,
    manifest: &CounterfactualAdvantageDatasetManifest,
) -> Result<(), DataError> {
    if manifest.schema_version != COUNTERFACTUAL_ADVANTAGE_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != COUNTERFACTUAL_ADVANTAGE_FEATURE_SCHEMA
        || manifest.target_schema != COUNTERFACTUAL_ADVANTAGE_TARGET_SCHEMA
        || manifest.record_size != COUNTERFACTUAL_ADVANTAGE_RECORD_SIZE
        || manifest.action_position_record_size != ACTION_POSITION_RECORD_SIZE
        || manifest.target_dim != TARGET_DIM
        || manifest.maximum_candidates != COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES
        || manifest.maximum_samples != COUNTERFACTUAL_ADVANTAGE_MAX_SAMPLES
        || manifest.game != GameConfig::research_aaaaa(4)?
    {
        return Err(DataError::InvalidManifest(
            "counterfactual-advantage schema identifiers do not match",
        ));
    }
    manifest.teacher.validate()?;
    let mut games = 0;
    let mut groups = 0;
    for shard in &manifest.shards {
        let path = root.join(&shard.file);
        if fs::metadata(&path)?.len() != shard.byte_count {
            return Err(DataError::InvalidManifest(
                "counterfactual-advantage shard byte count mismatch",
            ));
        }
        if checksum_file(&path)? != shard.blake3 {
            return Err(DataError::ChecksumMismatch(path));
        }
        let shard_records = read_counterfactual_advantage_shard_records(
            root,
            manifest.split,
            &manifest.teacher,
            shard,
        )?;
        validate_game_records(&shard_records, shard.first_game_index, &manifest.teacher)?;
        games += shard.game_count;
        groups += shard.record_count;
    }
    let expected_candidates = groups * COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES;
    let expected_continuations = expected_candidates * manifest.teacher.samples_per_candidate;
    if games != manifest.completed_games
        || groups != manifest.total_groups
        || manifest.total_candidates != expected_candidates
        || manifest.total_continuations != expected_continuations
    {
        return Err(DataError::InvalidManifest(
            "counterfactual-advantage manifest totals do not match shards",
        ));
    }
    Ok(())
}

pub fn read_counterfactual_advantage_shard_records(
    root: &Path,
    split: DatasetSplit,
    teacher: &CounterfactualAdvantageTeacherConfig,
    shard: &ShardManifest,
) -> Result<Vec<CounterfactualAdvantageRecord>, DataError> {
    let path = root.join(&shard.file);
    validate_shard_header(&path, split, teacher, shard)?;
    let mut reader = BufReader::new(File::open(path)?);
    reader.seek(std::io::SeekFrom::Start(
        COUNTERFACTUAL_ADVANTAGE_HEADER_SIZE as u64,
    ))?;
    let mut records = Vec::with_capacity(shard.record_count);
    for _ in 0..shard.record_count {
        let mut bytes = [0; COUNTERFACTUAL_ADVANTAGE_RECORD_SIZE];
        reader.read_exact(&mut bytes)?;
        records.push(CounterfactualAdvantageRecord::from_bytes(&bytes));
    }
    Ok(records)
}

fn validate_game_records(
    records: &[CounterfactualAdvantageRecord],
    game_index: u64,
    teacher: &CounterfactualAdvantageTeacherConfig,
) -> Result<(), DataError> {
    if records.len() != teacher.groups_per_game {
        return Err(DataError::InvalidConfig(
            "counterfactual-advantage game has the wrong group count",
        ));
    }
    let stride = teacher.decision_stride();
    for (index, record) in records.iter().enumerate() {
        record.validate(teacher.samples_per_candidate)?;
        if record.parent.game_index != game_index
            || usize::from(record.parent.turn) != index * stride
        {
            return Err(DataError::InvalidConfig(
                "counterfactual-advantage game sequence is invalid",
            ));
        }
    }
    Ok(())
}

fn validate_resume(
    manifest: &CounterfactualAdvantageDatasetManifest,
    config: &CounterfactualAdvantageDatasetConfig,
) -> Result<(), DataError> {
    let current_provenance = collection_provenance()?;
    if manifest.schema_version != COUNTERFACTUAL_ADVANTAGE_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != COUNTERFACTUAL_ADVANTAGE_FEATURE_SCHEMA
        || manifest.target_schema != COUNTERFACTUAL_ADVANTAGE_TARGET_SCHEMA
        || manifest.record_size != COUNTERFACTUAL_ADVANTAGE_RECORD_SIZE
        || manifest.action_position_record_size != ACTION_POSITION_RECORD_SIZE
        || manifest.target_dim != TARGET_DIM
        || manifest.maximum_candidates != COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES
        || manifest.maximum_samples != COUNTERFACTUAL_ADVANTAGE_MAX_SAMPLES
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
    teacher: &CounterfactualAdvantageTeacherConfig,
    game_index: u64,
    records: &[CounterfactualAdvantageRecord],
) -> Result<(), DataError> {
    let temp_path = path.with_extension("cfa.tmp");
    let teacher_hash = blake3::hash(&serde_json::to_vec(teacher)?);
    let mut writer = BufWriter::new(File::create(&temp_path)?);
    writer.write_all(COUNTERFACTUAL_ADVANTAGE_SHARD_MAGIC)?;
    writer.write_all(&COUNTERFACTUAL_ADVANTAGE_DATASET_SCHEMA_VERSION.to_le_bytes())?;
    writer.write_all(&(COUNTERFACTUAL_ADVANTAGE_HEADER_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(COUNTERFACTUAL_ADVANTAGE_RECORD_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(ACTION_POSITION_RECORD_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(TARGET_DIM as u16).to_le_bytes())?;
    writer.write_all(&(COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES as u16).to_le_bytes())?;
    writer.write_all(&(COUNTERFACTUAL_ADVANTAGE_MAX_SAMPLES as u16).to_le_bytes())?;
    writer.write_all(&(PUBLIC_SUPPLY_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(records.len() as u32).to_le_bytes())?;
    writer.write_all(&1u32.to_le_bytes())?;
    writer.write_all(&[
        split.code(),
        4,
        teacher.candidate_count as u8,
        teacher.samples_per_candidate as u8,
    ])?;
    writer.write_all(&(teacher.groups_per_game as u32).to_le_bytes())?;
    writer.write_all(&game_index.to_le_bytes())?;
    writer
        .write_all(blake3::hash(COUNTERFACTUAL_ADVANTAGE_FEATURE_SCHEMA.as_bytes()).as_bytes())?;
    writer.write_all(blake3::hash(COUNTERFACTUAL_ADVANTAGE_TARGET_SCHEMA.as_bytes()).as_bytes())?;
    writer.write_all(teacher_hash.as_bytes())?;
    writer.write_all(&[0; 16])?;
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
    teacher: &CounterfactualAdvantageTeacherConfig,
    shard: &ShardManifest,
) -> Result<(), DataError> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut header = [0; COUNTERFACTUAL_ADVANTAGE_HEADER_SIZE];
    reader.read_exact(&mut header)?;
    let teacher_hash = blake3::hash(&serde_json::to_vec(teacher)?);
    if &header[..8] != COUNTERFACTUAL_ADVANTAGE_SHARD_MAGIC
        || u16::from_le_bytes([header[8], header[9]])
            != COUNTERFACTUAL_ADVANTAGE_DATASET_SCHEMA_VERSION
        || u16::from_le_bytes([header[10], header[11]]) as usize
            != COUNTERFACTUAL_ADVANTAGE_HEADER_SIZE
        || u16::from_le_bytes([header[12], header[13]]) as usize
            != COUNTERFACTUAL_ADVANTAGE_RECORD_SIZE
        || u16::from_le_bytes([header[14], header[15]]) as usize != ACTION_POSITION_RECORD_SIZE
        || u16::from_le_bytes([header[16], header[17]]) as usize != TARGET_DIM
        || u16::from_le_bytes([header[18], header[19]]) as usize
            != COUNTERFACTUAL_ADVANTAGE_MAX_CANDIDATES
        || u16::from_le_bytes([header[20], header[21]]) as usize
            != COUNTERFACTUAL_ADVANTAGE_MAX_SAMPLES
        || u16::from_le_bytes([header[22], header[23]]) as usize != PUBLIC_SUPPLY_SIZE
        || header[32] != split.code()
        || header[33] != 4
        || usize::from(header[34]) != teacher.candidate_count
        || usize::from(header[35]) != teacher.samples_per_candidate
        || u32::from_le_bytes(header[36..40].try_into().expect("fixed header")) as usize
            != teacher.groups_per_game
        || &header[48..80]
            != blake3::hash(COUNTERFACTUAL_ADVANTAGE_FEATURE_SCHEMA.as_bytes()).as_bytes()
        || &header[80..112]
            != blake3::hash(COUNTERFACTUAL_ADVANTAGE_TARGET_SCHEMA.as_bytes()).as_bytes()
        || &header[112..144] != teacher_hash.as_bytes()
    {
        return Err(DataError::InvalidShard(
            "incompatible counterfactual-advantage header",
        ));
    }
    let record_count =
        u32::from_le_bytes(header[24..28].try_into().expect("fixed header")) as usize;
    let game_count = u32::from_le_bytes(header[28..32].try_into().expect("fixed header")) as usize;
    let first_game_index = u64::from_le_bytes(header[40..48].try_into().expect("fixed header"));
    if record_count != shard.record_count
        || game_count != 1
        || shard.game_count != 1
        || first_game_index != shard.first_game_index
    {
        return Err(DataError::InvalidShard(
            "counterfactual-advantage header and manifest disagree",
        ));
    }
    let expected_size = COUNTERFACTUAL_ADVANTAGE_HEADER_SIZE as u64
        + record_count as u64 * COUNTERFACTUAL_ADVANTAGE_RECORD_SIZE as u64;
    if fs::metadata(path)?.len() != expected_size {
        return Err(DataError::InvalidShard(
            "counterfactual-advantage shard size does not match records",
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
    use std::path::PathBuf;

    use cascadia_game::{GameSeed, GameState, MarketSlot, Rotation, TurnAction, score_board};

    use super::*;

    fn candidate(
        game: &GameState,
        game_index: u64,
        rank: u16,
        sample_count: usize,
    ) -> CounterfactualAdvantageCandidate {
        let coord = game.boards()[0].frontier()[usize::from(rank - 1)];
        let action = TurnAction::paired(MarketSlot::ZERO, coord, Rotation::ZERO);
        let score = score_board(
            &game.preview_active_board(&action).unwrap(),
            game.config().scoring_cards,
        );
        CounterfactualAdvantageCandidate::new(
            [rank as u8; 32],
            f64::from(score.base_total),
            0.5,
            ActionPositionRecord::observe(game, &action, game_index, rank, score.base_total)
                .unwrap(),
            &vec![score; sample_count],
        )
        .unwrap()
    }

    fn teacher(candidate_selection: Option<&str>) -> CounterfactualAdvantageTeacherConfig {
        CounterfactualAdvantageTeacherConfig {
            strategy_id: "habitat-candidate-lookahead-v1-k8-h6-r4-d4".to_owned(),
            immediate_candidates: 8,
            habitat_candidates: 6,
            determinizations: 4,
            greedy_plies: 4,
            candidate_count: 4,
            groups_per_game: 16,
            samples_per_candidate: 16,
            sample_seed_domain: "cascadia-v2-counterfactual-advantage-v1".to_owned(),
            candidate_selection: candidate_selection.map(str::to_owned),
            stabilization_conditioning: None,
        }
    }

    #[test]
    fn legacy_teacher_serialization_and_dataset_id_remain_exact() {
        let teacher = teacher(None);
        assert_eq!(
            serde_json::to_string(&teacher).unwrap(),
            r#"{"strategy_id":"habitat-candidate-lookahead-v1-k8-h6-r4-d4","immediate_candidates":8,"habitat_candidates":6,"determinizations":4,"greedy_plies":4,"candidate_count":4,"groups_per_game":16,"samples_per_candidate":16,"sample_seed_domain":"cascadia-v2-counterfactual-advantage-v1"}"#
        );
        assert_eq!(
            teacher.candidate_selection_id(),
            COUNTERFACTUAL_ADVANTAGE_NEAREST_SELECTION
        );
        let config = CounterfactualAdvantageDatasetConfig {
            output: PathBuf::from("/tmp/legacy-counterfactual-advantage-id"),
            split: DatasetSplit::Validation,
            first_game_index: 66_000,
            games: 2,
            teacher,
            resume: false,
        };
        assert_eq!(
            counterfactual_advantage_dataset_id(&config),
            "counterfactual-advantage-habitat-candidate-lookahead-v1-k8-h6-r4-d4-k4-g16-r16-validation-66000"
        );
    }

    #[test]
    fn stratified_teacher_has_an_explicit_versioned_dataset_id() {
        let config = CounterfactualAdvantageDatasetConfig {
            output: PathBuf::from("/tmp/stratified-counterfactual-advantage-id"),
            split: DatasetSplit::Validation,
            first_game_index: 67_000,
            games: 2,
            teacher: teacher(Some(COUNTERFACTUAL_ADVANTAGE_STRATIFIED_SELECTION)),
            resume: false,
        };
        assert_eq!(
            counterfactual_advantage_dataset_id(&config),
            "counterfactual-advantage-habitat-candidate-lookahead-v1-k8-h6-r4-d4-selected-high-median-low-v1-k4-g16-r16-validation-67000"
        );
    }

    #[test]
    fn conditioned_teacher_has_an_explicit_versioned_dataset_id() {
        let mut teacher = teacher(Some(COUNTERFACTUAL_ADVANTAGE_STRATIFIED_SELECTION));
        teacher.stabilization_conditioning =
            Some(COUNTERFACTUAL_ADVANTAGE_STABILIZATION_CONDITIONING.to_owned());
        let config = CounterfactualAdvantageDatasetConfig {
            output: PathBuf::from("/tmp/conditioned-counterfactual-advantage-id"),
            split: DatasetSplit::Train,
            first_game_index: 69_000,
            games: 128,
            teacher,
            resume: false,
        };
        assert_eq!(
            counterfactual_advantage_dataset_id(&config),
            "counterfactual-advantage-habitat-candidate-lookahead-v1-k8-h6-r4-d4-selected-high-median-low-v1-reject-unstable-market-trajectories-v1-k4-g16-r16-train-69000"
        );
    }

    #[test]
    fn counterfactual_advantage_record_round_trip_is_exact() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(47),
        )
        .unwrap();
        let current = score_board(&game.boards()[0], game.config().scoring_cards);
        let candidates = (1..=4).map(|rank| candidate(&game, 11, rank, 2)).collect();
        let record = CounterfactualAdvantageRecord::new(
            42,
            1,
            current,
            game.public_supply(),
            PositionRecord::observe(&game, 11),
            &[GameSeed::from_u64(100), GameSeed::from_u64(101)],
            candidates,
        )
        .unwrap();

        let decoded = CounterfactualAdvantageRecord::from_bytes(&record.to_bytes());

        assert_eq!(decoded, record);
        decoded.validate(2).unwrap();
        assert_eq!(
            record.to_bytes().len(),
            COUNTERFACTUAL_ADVANTAGE_RECORD_SIZE
        );
    }

    #[test]
    fn counterfactual_advantage_rejects_duplicate_action_hashes() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(48),
        )
        .unwrap();
        let current = score_board(&game.boards()[0], game.config().scoring_cards);
        let mut candidates = (1..=4)
            .map(|rank| candidate(&game, 12, rank, 2))
            .collect::<Vec<_>>();
        candidates[1].action_hash = candidates[0].action_hash;
        assert!(
            CounterfactualAdvantageRecord::new(
                43,
                0,
                current,
                game.public_supply(),
                PositionRecord::observe(&game, 12),
                &[GameSeed::from_u64(102), GameSeed::from_u64(103)],
                candidates,
            )
            .is_err()
        );
    }
}
