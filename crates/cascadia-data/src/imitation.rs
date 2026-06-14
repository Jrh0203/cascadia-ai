use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use cascadia_game::{
    DraftChoice, GameConfig, GameState, HexCoord, MarketPrelude, MarketSlot, Rotation,
    TilePlacement, TurnAction,
};
use serde::{Deserialize, Serialize};

use super::{
    CollectionProvenance, DataError, DatasetSplit, FEATURE_SCHEMA, PositionRecord,
    RankingShardManifest, checksum_file, collection_provenance, collection_provenance_matches,
    read_array, unix_seconds, write_manifest_atomic, write_slice,
};

pub const IMITATION_DATASET_SCHEMA_VERSION: u16 = 1;
pub const IMITATION_FEATURE_SCHEMA: &str = "compact-state-action-v1";
pub const IMITATION_TARGET_SCHEMA: &str = "canonical-action-imitation-v1";
pub const IMITATION_SHARD_MAGIC: &[u8; 8] = b"CSD2IMT\0";
pub const IMITATION_HEADER_SIZE: usize = 112;
pub const IMITATION_GROUP_HEADER_SIZE: usize = 16 + super::RECORD_SIZE;
pub const IMITATION_CANDIDATE_RECORD_SIZE: usize = 36 + PROPOSAL_ACTION_FEATURE_SIZE;
pub const PROPOSAL_ACTION_FEATURE_SIZE: usize = 32;

const NONE: u8 = u8::MAX;

#[derive(Debug, Clone)]
pub struct ImitationDatasetConfig {
    pub output: PathBuf,
    pub split: DatasetSplit,
    pub first_game_index: u64,
    pub games: usize,
    pub teacher: ImitationTeacherConfig,
    pub candidates: ImitationCandidateConfig,
    pub resume: bool,
}

impl ImitationDatasetConfig {
    fn validate(&self) -> Result<(), DataError> {
        if self.games == 0 {
            return Err(DataError::InvalidConfig("imitation games must be positive"));
        }
        self.teacher.validate()?;
        self.candidates.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImitationTeacherConfig {
    pub strategy_id: String,
    pub rollouts: usize,
    pub prefilter_candidates: usize,
    pub weights_path: String,
    pub weights_bytes: u64,
    pub weights_blake3: String,
}

impl ImitationTeacherConfig {
    pub fn from_weights(
        strategy_id: impl Into<String>,
        rollouts: usize,
        prefilter_candidates: usize,
        weights: &Path,
    ) -> Result<Self, DataError> {
        let weights = weights.canonicalize()?;
        Ok(Self {
            strategy_id: strategy_id.into(),
            rollouts,
            prefilter_candidates,
            weights_path: weights.display().to_string(),
            weights_bytes: fs::metadata(&weights)?.len(),
            weights_blake3: checksum_file(&weights)?,
        })
    }

    fn validate(&self) -> Result<(), DataError> {
        if self.strategy_id.trim().is_empty()
            || self.rollouts == 0
            || self.prefilter_candidates == 0
            || self.weights_path.trim().is_empty()
            || self.weights_bytes == 0
            || self.weights_blake3.trim().is_empty()
        {
            return Err(DataError::InvalidConfig(
                "imitation teacher metadata is incomplete",
            ));
        }
        let weights = Path::new(&self.weights_path);
        if fs::metadata(weights)?.len() != self.weights_bytes
            || checksum_file(weights)? != self.weights_blake3
        {
            return Err(DataError::InvalidConfig(
                "imitation teacher weights failed integrity validation",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImitationCandidateConfig {
    pub group_limit: usize,
    pub immediate_limit: usize,
    pub pattern_immediate_limit: usize,
    pub pattern_habitat_limit: usize,
    pub pattern_bear_limit: usize,
    pub pattern_market_draws: usize,
    pub deterministic_sampler: String,
}

impl ImitationCandidateConfig {
    fn validate(&self) -> Result<(), DataError> {
        if self.group_limit < 2
            || self.group_limit > usize::from(u16::MAX)
            || self.immediate_limit == 0
            || self.pattern_immediate_limit == 0
            || self.pattern_habitat_limit == 0
            || self.pattern_bear_limit == 0
            || self.pattern_market_draws == 0
            || self.deterministic_sampler.trim().is_empty()
        {
            return Err(DataError::InvalidConfig(
                "imitation candidate configuration is invalid",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProposalActionFeatures {
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
}

impl ProposalActionFeatures {
    pub fn from_game_action(
        game: &GameState,
        action: &TurnAction,
        immediate_rank: u16,
        immediate_score: u16,
    ) -> Result<Self, DataError> {
        if immediate_rank == 0 {
            return Err(DataError::InvalidConfig(
                "imitation immediate rank must be positive",
            ));
        }
        let staged = game.preview_market_prelude(&action.prelude())?;
        let (draft_kind, tile_slot, wildlife_slot) = match action.draft {
            DraftChoice::Paired { slot } => (0, slot.index() as u8, slot.index() as u8),
            DraftChoice::Independent {
                tile_slot,
                wildlife_slot,
            } => (1, tile_slot.index() as u8, wildlife_slot.index() as u8),
        };
        let tile = staged.market().tiles[usize::from(tile_slot)].ok_or(
            DataError::InvalidConfig("imitation tile slot is unavailable"),
        )?;
        let wildlife = staged.market().wildlife[usize::from(wildlife_slot)].ok_or(
            DataError::InvalidConfig("imitation wildlife slot is unavailable"),
        )?;
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
                DataError::InvalidConfig("imitation wipe count exceeds fixed-width storage")
            })?,
            paid_wipe_slot_mask,
            paid_wipe_total_slots: u8::try_from(paid_wipe_total_slots).map_err(|_| {
                DataError::InvalidConfig("imitation wiped slots exceed fixed-width storage")
            })?,
            immediate_rank,
            immediate_score,
        })
    }

    pub fn to_bytes(&self) -> [u8; PROPOSAL_ACTION_FEATURE_SIZE] {
        let mut bytes = [0; PROPOSAL_ACTION_FEATURE_SIZE];
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
        offset += 10;
        debug_assert_eq!(offset, PROPOSAL_ACTION_FEATURE_SIZE);
        bytes
    }

    pub fn from_bytes(bytes: &[u8; PROPOSAL_ACTION_FEATURE_SIZE]) -> Self {
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
        offset += 10;
        debug_assert_eq!(offset, PROPOSAL_ACTION_FEATURE_SIZE);
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
        }
    }

    pub fn to_game_action(&self, game: &GameState) -> Result<TurnAction, DataError> {
        if self.paid_wipe_count != 0
            || self.paid_wipe_slot_mask != 0
            || self.paid_wipe_total_slots != 0
        {
            return Err(DataError::InvalidConfig(
                "compact imitation action cannot reconstruct paid wildlife wipes",
            ));
        }
        let tile_slot = MarketSlot::new(self.tile_slot).ok_or(DataError::InvalidConfig(
            "imitation action has an invalid tile slot",
        ))?;
        let wildlife_slot = MarketSlot::new(self.wildlife_slot).ok_or(DataError::InvalidConfig(
            "imitation action has an invalid wildlife slot",
        ))?;
        let draft = match self.draft_kind {
            0 if tile_slot == wildlife_slot => DraftChoice::Paired { slot: tile_slot },
            1 => DraftChoice::Independent {
                tile_slot,
                wildlife_slot,
            },
            _ => {
                return Err(DataError::InvalidConfig(
                    "imitation action has an invalid draft kind",
                ));
            }
        };
        let prelude = MarketPrelude {
            replace_three_of_a_kind: self.replace_three_of_a_kind != 0,
            wildlife_wipes: Vec::new(),
        };
        let staged = game.preview_market_prelude(&prelude)?;
        let tile = staged.market().tiles[tile_slot.index()].ok_or(DataError::InvalidConfig(
            "imitation action tile slot is unavailable after its prelude",
        ))?;
        let wildlife =
            staged.market().wildlife[wildlife_slot.index()].ok_or(DataError::InvalidConfig(
                "imitation action wildlife slot is unavailable after its prelude",
            ))?;
        if tile.terrain_a as u8 != self.tile_terrain_a
            || tile.terrain_b.map_or(NONE, |terrain| terrain as u8) != self.tile_terrain_b
            || tile.wildlife.bits() != self.tile_wildlife_mask
            || u8::from(tile.keystone) != self.tile_keystone
            || wildlife as u8 != self.drafted_wildlife
        {
            return Err(DataError::InvalidConfig(
                "imitation action market metadata does not match replay state",
            ));
        }
        let rotation = Rotation::new(self.rotation).ok_or(DataError::InvalidConfig(
            "imitation action rotation is outside zero through five",
        ))?;
        let action = TurnAction {
            replace_three_of_a_kind: prelude.replace_three_of_a_kind,
            wildlife_wipes: Vec::new(),
            draft,
            tile: TilePlacement {
                coord: HexCoord::new(self.tile_q, self.tile_r),
                rotation,
            },
            wildlife: (self.wildlife_present != 0)
                .then_some(HexCoord::new(self.wildlife_q, self.wildlife_r)),
        };
        game.transition(&action)?;
        Ok(action)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProposalPositionRecord {
    pub position: PositionRecord,
    pub action: ProposalActionFeatures,
}

impl ProposalPositionRecord {
    pub fn observe(
        game: &GameState,
        action: &TurnAction,
        game_index: u64,
        immediate_rank: u16,
        immediate_score: u16,
    ) -> Result<Self, DataError> {
        Ok(Self {
            position: PositionRecord::observe(game, game_index),
            action: ProposalActionFeatures::from_game_action(
                game,
                action,
                immediate_rank,
                immediate_score,
            )?,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImitationRecord {
    pub group_id: u64,
    pub candidate_index: u16,
    pub candidate_count: u16,
    pub immediate_rank: u16,
    pub immediate_score: u16,
    pub teacher_mean: f32,
    pub teacher_stddev: f32,
    pub action_hash: [u8; 32],
    pub input: ProposalPositionRecord,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImitationDatasetManifest {
    pub schema_version: u16,
    pub dataset_id: String,
    pub feature_schema: String,
    pub position_feature_schema: String,
    pub target_schema: String,
    pub group_header_size: usize,
    pub candidate_record_size: usize,
    pub action_feature_size: usize,
    pub game: GameConfig,
    pub split: DatasetSplit,
    pub teacher: ImitationTeacherConfig,
    pub candidates: ImitationCandidateConfig,
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

pub struct ImitationDatasetWriter {
    output: PathBuf,
    manifest_path: PathBuf,
    manifest: ImitationDatasetManifest,
}

impl ImitationDatasetWriter {
    pub fn open(config: &ImitationDatasetConfig) -> Result<Self, DataError> {
        config.validate()?;
        fs::create_dir_all(&config.output)?;
        let manifest_path = config.output.join("dataset.json");
        let mut manifest = if manifest_path.exists() {
            if !config.resume {
                return Err(DataError::DatasetExists(config.output.clone()));
            }
            let manifest: ImitationDatasetManifest =
                serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
            validate_resume(&manifest, config)?;
            validate_imitation_dataset(&config.output, &manifest)?;
            manifest
        } else {
            let now = unix_seconds()?;
            ImitationDatasetManifest {
                schema_version: IMITATION_DATASET_SCHEMA_VERSION,
                dataset_id: dataset_id(config)?,
                feature_schema: IMITATION_FEATURE_SCHEMA.to_owned(),
                position_feature_schema: FEATURE_SCHEMA.to_owned(),
                target_schema: IMITATION_TARGET_SCHEMA.to_owned(),
                group_header_size: IMITATION_GROUP_HEADER_SIZE,
                candidate_record_size: IMITATION_CANDIDATE_RECORD_SIZE,
                action_feature_size: PROPOSAL_ACTION_FEATURE_SIZE,
                game: GameConfig::research_aaaaa(4)?,
                split: config.split,
                teacher: config.teacher.clone(),
                candidates: config.candidates.clone(),
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
        if manifest.completed_games > config.games {
            return Err(DataError::ResumeMismatch);
        }
        manifest.requested_games = config.games;
        Ok(Self {
            output: config.output.clone(),
            manifest_path,
            manifest,
        })
    }

    pub fn manifest(&self) -> &ImitationDatasetManifest {
        &self.manifest
    }

    pub fn root(&self) -> &Path {
        &self.output
    }

    pub fn append_shard(
        &mut self,
        first_game_index: u64,
        game_count: usize,
        records: &[ImitationRecord],
    ) -> Result<(), DataError> {
        if game_count == 0 || records.is_empty() {
            return Err(DataError::InvalidConfig(
                "imitation shard must contain games and records",
            ));
        }
        let expected_first = self.manifest.first_game_index + self.manifest.completed_games as u64;
        if first_game_index != expected_first {
            return Err(DataError::InvalidConfig(
                "imitation shard game range is not contiguous",
            ));
        }
        if self.manifest.completed_games + game_count > self.manifest.requested_games {
            return Err(DataError::InvalidConfig(
                "imitation shard exceeds requested game count",
            ));
        }
        let group_count = validate_record_groups(records, self.manifest.candidates.group_limit)?;
        let shard_index = self.manifest.shards.len();
        let file_name = format!("shard-{shard_index:05}.cim");
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

pub fn validate_imitation_dataset(
    root: &Path,
    manifest: &ImitationDatasetManifest,
) -> Result<(), DataError> {
    manifest.teacher.validate()?;
    manifest.candidates.validate()?;
    if manifest.schema_version != IMITATION_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != IMITATION_FEATURE_SCHEMA
        || manifest.position_feature_schema != FEATURE_SCHEMA
        || manifest.target_schema != IMITATION_TARGET_SCHEMA
        || manifest.group_header_size != IMITATION_GROUP_HEADER_SIZE
        || manifest.candidate_record_size != IMITATION_CANDIDATE_RECORD_SIZE
        || manifest.action_feature_size != PROPOSAL_ACTION_FEATURE_SIZE
        || manifest.game != GameConfig::research_aaaaa(4)?
    {
        return Err(DataError::InvalidManifest(
            "imitation schema identifiers do not match",
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
                "imitation shard byte count mismatch",
            ));
        }
        if checksum_file(&path)? != shard.blake3 {
            return Err(DataError::ChecksumMismatch(path));
        }
        let shard_records = read_imitation_shard_records(root, manifest.split, shard)?;
        let shard_groups = validate_record_groups(&shard_records, manifest.candidates.group_limit)?;
        if shard_groups != shard.group_count {
            return Err(DataError::InvalidManifest(
                "imitation shard group count mismatch",
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
            "imitation manifest totals do not match shards",
        ));
    }
    Ok(())
}

fn validate_resume(
    manifest: &ImitationDatasetManifest,
    config: &ImitationDatasetConfig,
) -> Result<(), DataError> {
    let current_provenance = collection_provenance()?;
    if manifest.schema_version != IMITATION_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != IMITATION_FEATURE_SCHEMA
        || manifest.position_feature_schema != FEATURE_SCHEMA
        || manifest.target_schema != IMITATION_TARGET_SCHEMA
        || manifest.group_header_size != IMITATION_GROUP_HEADER_SIZE
        || manifest.candidate_record_size != IMITATION_CANDIDATE_RECORD_SIZE
        || manifest.action_feature_size != PROPOSAL_ACTION_FEATURE_SIZE
        || manifest.game != GameConfig::research_aaaaa(4)?
        || manifest.split != config.split
        || manifest.teacher != config.teacher
        || manifest.candidates != config.candidates
        || manifest.first_game_index != config.first_game_index
        || !collection_provenance_matches(&manifest.provenance, &current_provenance)
    {
        return Err(DataError::ResumeMismatch);
    }
    Ok(())
}

fn validate_record_groups(
    records: &[ImitationRecord],
    group_limit: usize,
) -> Result<usize, DataError> {
    let groups = record_groups(records)?;
    for group in &groups {
        let count = group.len();
        let positive_count = group
            .iter()
            .filter(|record| record.teacher_mean == 1.0)
            .count();
        let action_hashes = group
            .iter()
            .map(|record| record.action_hash)
            .collect::<BTreeSet<_>>();
        let state = group[0].input.position.to_bytes();
        if count < 2
            || count > group_limit
            || positive_count != 1
            || action_hashes.len() != count
            || group.iter().enumerate().any(|(index, record)| {
                usize::from(record.candidate_count) != count
                    || usize::from(record.candidate_index) != index
                    || record.immediate_rank == 0
                    || record.input.action.immediate_rank != record.immediate_rank
                    || record.input.action.immediate_score != record.immediate_score
                    || record.input.position.to_bytes() != state
                    || !matches!(record.teacher_mean, 0.0 | 1.0)
                    || record.teacher_stddev != 0.0
            })
        {
            return Err(DataError::InvalidConfig(
                "imitation candidate group is inconsistent",
            ));
        }
    }
    Ok(groups.len())
}

fn record_groups(records: &[ImitationRecord]) -> Result<Vec<&[ImitationRecord]>, DataError> {
    let mut groups = Vec::new();
    let mut seen = BTreeSet::new();
    let mut start = 0;
    while start < records.len() {
        let group_id = records[start].group_id;
        if !seen.insert(group_id) {
            return Err(DataError::InvalidConfig(
                "imitation group records must be contiguous",
            ));
        }
        let end = records[start..]
            .iter()
            .position(|record| record.group_id != group_id)
            .map_or(records.len(), |offset| start + offset);
        groups.push(&records[start..end]);
        start = end;
    }
    Ok(groups)
}

fn dataset_id(config: &ImitationDatasetConfig) -> Result<String, DataError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"cascadia-v2-canonical-action-imitation-dataset");
    hasher.update(config.split.id().as_bytes());
    hasher.update(&config.first_game_index.to_le_bytes());
    hasher.update(&serde_json::to_vec(&config.teacher)?);
    hasher.update(&serde_json::to_vec(&config.candidates)?);
    let digest = hasher.finalize().to_hex().to_string();
    Ok(format!(
        "canonical-action-imitation-{}-{}",
        config.split.id(),
        &digest[..16]
    ))
}

fn write_shard(
    path: &Path,
    split: DatasetSplit,
    first_game_index: u64,
    game_count: usize,
    group_count: usize,
    records: &[ImitationRecord],
) -> Result<(), DataError> {
    let groups = record_groups(records)?;
    if groups.len() != group_count {
        return Err(DataError::InvalidConfig(
            "imitation group count changed before serialization",
        ));
    }
    let temp_path = path.with_extension("cim.tmp");
    let mut writer = BufWriter::new(File::create(&temp_path)?);
    writer.write_all(IMITATION_SHARD_MAGIC)?;
    writer.write_all(&IMITATION_DATASET_SCHEMA_VERSION.to_le_bytes())?;
    writer.write_all(&(IMITATION_HEADER_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(IMITATION_GROUP_HEADER_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(IMITATION_CANDIDATE_RECORD_SIZE as u16).to_le_bytes())?;
    writer.write_all(&(records.len() as u32).to_le_bytes())?;
    writer.write_all(&(group_count as u32).to_le_bytes())?;
    writer.write_all(&(game_count as u32).to_le_bytes())?;
    writer.write_all(&[split.code(), 4, 0, 0])?;
    writer.write_all(&first_game_index.to_le_bytes())?;
    writer.write_all(blake3::hash(IMITATION_FEATURE_SCHEMA.as_bytes()).as_bytes())?;
    writer.write_all(blake3::hash(IMITATION_TARGET_SCHEMA.as_bytes()).as_bytes())?;
    writer.write_all(&[0; 8])?;
    for group in groups {
        let selected_index = group
            .iter()
            .position(|record| record.teacher_mean == 1.0)
            .ok_or(DataError::InvalidConfig(
                "imitation group has no selected action",
            ))?;
        writer.write_all(&group[0].group_id.to_le_bytes())?;
        writer.write_all(&(group.len() as u16).to_le_bytes())?;
        writer.write_all(&(selected_index as u16).to_le_bytes())?;
        writer.write_all(&[0; 4])?;
        writer.write_all(&group[0].input.position.to_bytes())?;
        for record in group {
            writer.write_all(&record.immediate_rank.to_le_bytes())?;
            writer.write_all(&record.immediate_score.to_le_bytes())?;
            writer.write_all(&record.action_hash)?;
            writer.write_all(&record.input.action.to_bytes())?;
        }
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
    let mut header = [0; IMITATION_HEADER_SIZE];
    reader.read_exact(&mut header)?;
    if &header[..8] != IMITATION_SHARD_MAGIC
        || u16::from_le_bytes([header[8], header[9]]) != IMITATION_DATASET_SCHEMA_VERSION
        || u16::from_le_bytes([header[10], header[11]]) as usize != IMITATION_HEADER_SIZE
        || u16::from_le_bytes([header[12], header[13]]) as usize != IMITATION_GROUP_HEADER_SIZE
        || u16::from_le_bytes([header[14], header[15]]) as usize != IMITATION_CANDIDATE_RECORD_SIZE
        || header[28] != split.code()
        || &header[40..72] != blake3::hash(IMITATION_FEATURE_SCHEMA.as_bytes()).as_bytes()
        || &header[72..104] != blake3::hash(IMITATION_TARGET_SCHEMA.as_bytes()).as_bytes()
    {
        return Err(DataError::InvalidShard("incompatible imitation header"));
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
            "imitation header and manifest disagree",
        ));
    }
    let expected_size = IMITATION_HEADER_SIZE as u64
        + group_count as u64 * IMITATION_GROUP_HEADER_SIZE as u64
        + record_count as u64 * IMITATION_CANDIDATE_RECORD_SIZE as u64;
    if fs::metadata(path)?.len() != expected_size {
        return Err(DataError::InvalidShard(
            "imitation shard file size does not match records",
        ));
    }
    Ok(())
}

pub fn read_imitation_shard_records(
    root: &Path,
    split: DatasetSplit,
    shard: &RankingShardManifest,
) -> Result<Vec<ImitationRecord>, DataError> {
    let path = root.join(&shard.file);
    validate_shard_header(&path, split, shard)?;
    let mut reader = BufReader::new(File::open(path)?);
    reader.seek_relative(IMITATION_HEADER_SIZE as i64)?;
    let mut records = Vec::with_capacity(shard.record_count);
    for _ in 0..shard.group_count {
        let mut group_header = [0; 16];
        reader.read_exact(&mut group_header)?;
        let group_id = u64::from_le_bytes(group_header[..8].try_into().expect("fixed header"));
        let candidate_count =
            u16::from_le_bytes(group_header[8..10].try_into().expect("fixed header"));
        let selected_index =
            u16::from_le_bytes(group_header[10..12].try_into().expect("fixed header"));
        if candidate_count < 2 || selected_index >= candidate_count {
            return Err(DataError::InvalidShard(
                "imitation group header is inconsistent",
            ));
        }
        let mut position_bytes = [0; super::RECORD_SIZE];
        reader.read_exact(&mut position_bytes)?;
        let position = PositionRecord::from_bytes(&position_bytes);
        for candidate_index in 0..candidate_count {
            let mut candidate_bytes = [0; IMITATION_CANDIDATE_RECORD_SIZE];
            reader.read_exact(&mut candidate_bytes)?;
            let mut offset = 0;
            let immediate_rank = u16::from_le_bytes(read_array(&candidate_bytes, &mut offset));
            let immediate_score = u16::from_le_bytes(read_array(&candidate_bytes, &mut offset));
            let action_hash = read_array(&candidate_bytes, &mut offset);
            let action_bytes = read_array(&candidate_bytes, &mut offset);
            debug_assert_eq!(offset, IMITATION_CANDIDATE_RECORD_SIZE);
            records.push(ImitationRecord {
                group_id,
                candidate_index,
                candidate_count,
                immediate_rank,
                immediate_score,
                teacher_mean: if candidate_index == selected_index {
                    1.0
                } else {
                    0.0
                },
                teacher_stddev: 0.0,
                action_hash,
                input: ProposalPositionRecord {
                    position: position.clone(),
                    action: ProposalActionFeatures::from_bytes(&action_bytes),
                },
            });
        }
    }
    if records.len() != shard.record_count {
        return Err(DataError::InvalidShard(
            "imitation candidate count does not match header",
        ));
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameSeed, MarketSlot, Rotation, score_board};

    use super::*;

    fn sample_record(index: u16, selected: bool) -> ImitationRecord {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(7),
        )
        .unwrap();
        let action = TurnAction::paired(
            MarketSlot::ZERO,
            game.boards()[0].frontier()[usize::from(index)],
            Rotation::ZERO,
        );
        let immediate_score = score_board(
            &game.preview_active_board(&action).unwrap(),
            game.config().scoring_cards,
        )
        .base_total;
        ImitationRecord {
            group_id: 1,
            candidate_index: index,
            candidate_count: 2,
            immediate_rank: index + 1,
            immediate_score,
            teacher_mean: if selected { 1.0 } else { 0.0 },
            teacher_stddev: 0.0,
            action_hash: [index as u8; 32],
            input: ProposalPositionRecord::observe(
                &game,
                &action,
                90_000,
                index + 1,
                immediate_score,
            )
            .unwrap(),
        }
    }

    #[test]
    fn proposal_action_round_trip_is_exact() {
        let expected = sample_record(0, true).input.action;
        assert_eq!(
            ProposalActionFeatures::from_bytes(&expected.to_bytes()),
            expected
        );
    }

    #[test]
    fn compact_action_reconstructs_and_validates_the_canonical_action() {
        let record = sample_record(0, true);
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(7),
        )
        .unwrap();
        let reconstructed = record.input.action.to_game_action(&game).unwrap();
        assert_eq!(
            reconstructed,
            TurnAction::paired(
                MarketSlot::ZERO,
                game.boards()[0].frontier()[0],
                Rotation::ZERO
            )
        );
    }

    #[test]
    fn compact_action_rejects_unreconstructable_paid_wipes() {
        let mut action = sample_record(0, true).input.action;
        action.paid_wipe_count = 1;
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(7),
        )
        .unwrap();
        assert!(matches!(
            action.to_game_action(&game),
            Err(DataError::InvalidConfig(_))
        ));
    }

    #[test]
    fn imitation_groups_require_one_exact_positive_and_shared_state() {
        let valid = [sample_record(0, true), sample_record(1, false)];
        assert_eq!(validate_record_groups(&valid, 64).unwrap(), 1);

        let invalid = [sample_record(0, true), sample_record(1, true)];
        assert!(matches!(
            validate_record_groups(&invalid, 64),
            Err(DataError::InvalidConfig(_))
        ));
    }

    #[test]
    fn grouped_shard_round_trip_stores_the_position_once() {
        let root = std::env::temp_dir().join(format!(
            "cascadia-imitation-grouped-shard-{}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("shard-00000.cim");
        let records = [sample_record(0, true), sample_record(1, false)];
        write_shard(&path, DatasetSplit::Train, 90_000, 1, 1, &records).unwrap();
        let shard = RankingShardManifest {
            file: "shard-00000.cim".to_owned(),
            first_game_index: 90_000,
            game_count: 1,
            group_count: 1,
            record_count: records.len(),
            byte_count: fs::metadata(&path).unwrap().len(),
            blake3: checksum_file(&path).unwrap(),
        };

        assert_eq!(
            fs::metadata(&path).unwrap().len() as usize,
            IMITATION_HEADER_SIZE
                + IMITATION_GROUP_HEADER_SIZE
                + records.len() * IMITATION_CANDIDATE_RECORD_SIZE
        );
        assert_eq!(
            read_imitation_shard_records(&root, DatasetSplit::Train, &shard).unwrap(),
            records
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn candidate_configuration_rejects_degenerate_groups() {
        let config = ImitationCandidateConfig {
            group_limit: 1,
            immediate_limit: 16,
            pattern_immediate_limit: 8,
            pattern_habitat_limit: 6,
            pattern_bear_limit: 8,
            pattern_market_draws: 4,
            deterministic_sampler: "blake3-action-v1".to_owned(),
        };
        assert!(matches!(
            config.validate(),
            Err(DataError::InvalidConfig(_))
        ));
    }
}
