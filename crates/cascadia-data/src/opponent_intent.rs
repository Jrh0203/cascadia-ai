use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use blake3::Hasher;
use cascadia_game::{DraftChoice, GameConfig, GameSeed, GameState, MarketSlot, TurnAction};
use cascadia_sim::{MatchConfig, StrategyKind, play_match_observed};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::{
    CollectionProvenance, DataError, DatasetSplit, PositionRecord, RankingShardManifest,
    checksum_file, collection_provenance, collection_provenance_matches, read_array, unix_seconds,
    write_manifest_atomic, write_slice,
};

pub const OPPONENT_INTENT_DATASET_SCHEMA_VERSION: u16 = 1;
pub const OPPONENT_INTENT_FEATURE_SCHEMA: &str =
    "compact-public-state-recent-actions-no-policy-id-v1";
pub const OPPONENT_INTENT_TARGET_SCHEMA: &str = "three-opponent-next-action-four-tile-survival-v1";
pub const OPPONENT_INTENT_SHARD_MAGIC: &[u8; 8] = b"CSD2O1I\0";
pub const OPPONENT_INTENT_HEADER_SIZE: usize = 64;
pub const OPPONENT_INTENT_HISTORY_LENGTH: usize = 12;
pub const PUBLIC_ACTION_RECORD_SIZE: usize = 24;
pub const OPPONENT_INTENT_HISTORY_ENTRY_SIZE: usize = 3 + PUBLIC_ACTION_RECORD_SIZE;
pub const OPPONENT_ACTION_TARGET_SIZE: usize = 3 + PUBLIC_ACTION_RECORD_SIZE;
pub const TILE_SURVIVAL_TARGET_SIZE: usize = 5;
pub const OPPONENT_INTENT_RECORD_SIZE: usize = 8
    + 1
    + 1
    + 4
    + super::RECORD_SIZE
    + 1
    + OPPONENT_INTENT_HISTORY_LENGTH * OPPONENT_INTENT_HISTORY_ENTRY_SIZE
    + 3 * OPPONENT_ACTION_TARGET_SIZE
    + 4 * TILE_SURVIVAL_TARGET_SIZE
    + 8;

const NONE: u8 = u8::MAX;
const WINDOWS_PER_GAME: usize = 76;
const SURVIVED_TO_NEXT_ACCESS: u8 = 4;

#[derive(Debug, Clone)]
pub struct OpponentIntentDatasetConfig {
    pub output: PathBuf,
    pub split: DatasetSplit,
    pub first_game_index: u64,
    pub games: usize,
    pub shard_games: usize,
    pub cohort: OpponentIntentCohort,
    pub resume: bool,
}

impl OpponentIntentDatasetConfig {
    fn validate(&self) -> Result<(), DataError> {
        if self.games == 0 || self.shard_games == 0 {
            return Err(DataError::InvalidConfig(
                "opponent-intent games and shard size must be positive",
            ));
        }
        self.cohort.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpponentIntentCohort {
    pub cohort_id: String,
    pub policy_pool: Vec<StrategyKind>,
    pub required_policy: Option<StrategyKind>,
}

impl OpponentIntentCohort {
    fn validate(&self) -> Result<(), DataError> {
        if self.cohort_id.trim().is_empty() || self.policy_pool.is_empty() {
            return Err(DataError::InvalidConfig(
                "opponent-intent cohort metadata is incomplete",
            ));
        }
        let unique = self
            .policy_pool
            .iter()
            .map(|policy| policy.id())
            .collect::<BTreeSet<_>>();
        if unique.len() != self.policy_pool.len() {
            return Err(DataError::InvalidConfig(
                "opponent-intent policy pool contains duplicates",
            ));
        }
        if self
            .required_policy
            .is_some_and(|required| !self.policy_pool.contains(&required))
        {
            return Err(DataError::InvalidConfig(
                "required opponent-intent policy is absent from the pool",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpponentIntentDatasetManifest {
    pub schema_version: u16,
    pub dataset_id: String,
    pub feature_schema: String,
    pub target_schema: String,
    pub record_size: usize,
    pub game: GameConfig,
    pub split: DatasetSplit,
    pub first_game_index: u64,
    pub requested_games: usize,
    pub completed_games: usize,
    pub total_records: usize,
    pub windows_per_game: usize,
    pub history_length: usize,
    pub cohort: OpponentIntentCohort,
    pub policy_assignment_domain: String,
    pub policy_identity_observable: bool,
    pub game_index_observable: bool,
    pub strategy_switch_targets_available: bool,
    pub created_unix_seconds: u64,
    pub updated_unix_seconds: u64,
    pub provenance: CollectionProvenance,
    pub shards: Vec<RankingShardManifest>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PublicActionRecord {
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
}

impl PublicActionRecord {
    pub fn observe(game: &GameState, action: &TurnAction) -> Result<Self, DataError> {
        let staged = game.preview_market_prelude(&action.prelude())?;
        let (draft_kind, tile_slot, wildlife_slot) = match action.draft {
            DraftChoice::Paired { slot } => (0, slot.index() as u8, slot.index() as u8),
            DraftChoice::Independent {
                tile_slot,
                wildlife_slot,
            } => (1, tile_slot.index() as u8, wildlife_slot.index() as u8),
        };
        let tile = staged.market().tiles[usize::from(tile_slot)].ok_or(
            DataError::InvalidConfig("opponent-intent action tile slot is unavailable"),
        )?;
        let wildlife = staged.market().wildlife[usize::from(wildlife_slot)].ok_or(
            DataError::InvalidConfig("opponent-intent action wildlife slot is unavailable"),
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
                DataError::InvalidConfig("opponent-intent wipe count exceeds fixed storage")
            })?,
            paid_wipe_slot_mask,
            paid_wipe_total_slots: u8::try_from(paid_wipe_total_slots).map_err(|_| {
                DataError::InvalidConfig("opponent-intent wiped slots exceed fixed storage")
            })?,
        })
    }

    pub fn to_bytes(self) -> [u8; PUBLIC_ACTION_RECORD_SIZE] {
        let mut bytes = [0; PUBLIC_ACTION_RECORD_SIZE];
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
        offset += 6;
        debug_assert_eq!(offset, PUBLIC_ACTION_RECORD_SIZE);
        bytes
    }

    pub fn from_bytes(bytes: &[u8; PUBLIC_ACTION_RECORD_SIZE]) -> Self {
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
        offset += 6;
        debug_assert_eq!(offset, PUBLIC_ACTION_RECORD_SIZE);
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
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OpponentIntentHistoryEntry {
    pub valid: u8,
    pub age: u8,
    pub relative_seat: u8,
    pub action: PublicActionRecord,
}

impl OpponentIntentHistoryEntry {
    fn to_bytes(self) -> [u8; OPPONENT_INTENT_HISTORY_ENTRY_SIZE] {
        let mut bytes = [0; OPPONENT_INTENT_HISTORY_ENTRY_SIZE];
        bytes[0] = self.valid;
        bytes[1] = self.age;
        bytes[2] = self.relative_seat;
        bytes[3..3 + PUBLIC_ACTION_RECORD_SIZE].copy_from_slice(&self.action.to_bytes());
        bytes
    }

    fn from_bytes(bytes: &[u8; OPPONENT_INTENT_HISTORY_ENTRY_SIZE]) -> Self {
        Self {
            valid: bytes[0],
            age: bytes[1],
            relative_seat: bytes[2],
            action: PublicActionRecord::from_bytes(
                bytes[3..3 + PUBLIC_ACTION_RECORD_SIZE]
                    .try_into()
                    .expect("history action has fixed width"),
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OpponentActionTarget {
    pub relative_seat: u8,
    pub policy_code: u8,
    pub selected_tile_id: u8,
    pub action: PublicActionRecord,
}

impl OpponentActionTarget {
    fn to_bytes(self) -> [u8; OPPONENT_ACTION_TARGET_SIZE] {
        let mut bytes = [0; OPPONENT_ACTION_TARGET_SIZE];
        bytes[..3].copy_from_slice(&[self.relative_seat, self.policy_code, self.selected_tile_id]);
        bytes[3..].copy_from_slice(&self.action.to_bytes());
        bytes
    }

    fn from_bytes(bytes: &[u8; OPPONENT_ACTION_TARGET_SIZE]) -> Self {
        Self {
            relative_seat: bytes[0],
            policy_code: bytes[1],
            selected_tile_id: bytes[2],
            action: PublicActionRecord::from_bytes(
                bytes[3..]
                    .try_into()
                    .expect("opponent target action has fixed width"),
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TileSurvivalTarget {
    pub initial_tile_id: u8,
    pub initial_wildlife: u8,
    pub disposition: u8,
    pub pair_survives: u8,
    pub final_slot: u8,
}

impl TileSurvivalTarget {
    fn to_bytes(self) -> [u8; TILE_SURVIVAL_TARGET_SIZE] {
        [
            self.initial_tile_id,
            self.initial_wildlife,
            self.disposition,
            self.pair_survives,
            self.final_slot,
        ]
    }

    fn from_bytes(bytes: &[u8; TILE_SURVIVAL_TARGET_SIZE]) -> Self {
        Self {
            initial_tile_id: bytes[0],
            initial_wildlife: bytes[1],
            disposition: bytes[2],
            pair_survives: bytes[3],
            final_slot: bytes[4],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpponentIntentRecord {
    pub game_index: u64,
    pub focal_turn: u8,
    pub focal_seat: u8,
    pub seat_policy_codes: [u8; 4],
    pub position: PositionRecord,
    pub history_count: u8,
    pub history: [OpponentIntentHistoryEntry; OPPONENT_INTENT_HISTORY_LENGTH],
    pub opponent_targets: [OpponentActionTarget; 3],
    pub survival_targets: [TileSurvivalTarget; 4],
    pub final_scores: [u16; 4],
}

impl OpponentIntentRecord {
    pub fn to_bytes(&self) -> [u8; OPPONENT_INTENT_RECORD_SIZE] {
        let mut bytes = [0; OPPONENT_INTENT_RECORD_SIZE];
        let mut offset = 0;
        write_slice(&mut bytes, &mut offset, &self.game_index.to_le_bytes());
        write_slice(
            &mut bytes,
            &mut offset,
            &[
                self.focal_turn,
                self.focal_seat,
                self.seat_policy_codes[0],
                self.seat_policy_codes[1],
                self.seat_policy_codes[2],
                self.seat_policy_codes[3],
            ],
        );
        write_slice(&mut bytes, &mut offset, &self.position.to_bytes());
        write_slice(&mut bytes, &mut offset, &[self.history_count]);
        for entry in self.history {
            write_slice(&mut bytes, &mut offset, &entry.to_bytes());
        }
        for target in self.opponent_targets {
            write_slice(&mut bytes, &mut offset, &target.to_bytes());
        }
        for target in self.survival_targets {
            write_slice(&mut bytes, &mut offset, &target.to_bytes());
        }
        for score in self.final_scores {
            write_slice(&mut bytes, &mut offset, &score.to_le_bytes());
        }
        debug_assert_eq!(offset, OPPONENT_INTENT_RECORD_SIZE);
        bytes
    }

    pub fn from_bytes(bytes: &[u8; OPPONENT_INTENT_RECORD_SIZE]) -> Self {
        let mut offset = 0;
        let game_index = u64::from_le_bytes(read_array(bytes, &mut offset));
        let [focal_turn, focal_seat, p0, p1, p2, p3] = read_array(bytes, &mut offset);
        let position =
            PositionRecord::from_bytes(&read_array::<{ super::RECORD_SIZE }>(bytes, &mut offset));
        let history_count = read_array::<1>(bytes, &mut offset)[0];
        let mut history = [OpponentIntentHistoryEntry::default(); OPPONENT_INTENT_HISTORY_LENGTH];
        for entry in &mut history {
            *entry = OpponentIntentHistoryEntry::from_bytes(&read_array(bytes, &mut offset));
        }
        let mut opponent_targets = [OpponentActionTarget::default(); 3];
        for target in &mut opponent_targets {
            *target = OpponentActionTarget::from_bytes(&read_array(bytes, &mut offset));
        }
        let mut survival_targets = [TileSurvivalTarget::default(); 4];
        for target in &mut survival_targets {
            *target = TileSurvivalTarget::from_bytes(&read_array(bytes, &mut offset));
        }
        let mut final_scores = [0; 4];
        for score in &mut final_scores {
            *score = u16::from_le_bytes(read_array(bytes, &mut offset));
        }
        debug_assert_eq!(offset, OPPONENT_INTENT_RECORD_SIZE);
        Self {
            game_index,
            focal_turn,
            focal_seat,
            seat_policy_codes: [p0, p1, p2, p3],
            position,
            history_count,
            history,
            opponent_targets,
            survival_targets,
            final_scores,
        }
    }

    pub fn model_input_bytes(&self) -> Vec<u8> {
        let mut position = self.position.clone();
        position.game_index = 0;
        position.targets = [0; super::TARGET_DIM];
        let mut bytes = Vec::with_capacity(
            super::RECORD_SIZE
                + 1
                + OPPONENT_INTENT_HISTORY_LENGTH * OPPONENT_INTENT_HISTORY_ENTRY_SIZE,
        );
        bytes.extend_from_slice(&position.to_bytes());
        bytes.push(self.history_count);
        for entry in self.history {
            bytes.extend_from_slice(&entry.to_bytes());
        }
        bytes
    }
}

pub struct OpponentIntentDatasetWriter {
    output: PathBuf,
    manifest_path: PathBuf,
    manifest: OpponentIntentDatasetManifest,
}

impl OpponentIntentDatasetWriter {
    pub fn open(config: &OpponentIntentDatasetConfig) -> Result<Self, DataError> {
        config.validate()?;
        fs::create_dir_all(&config.output)?;
        let manifest_path = config.output.join("dataset.json");
        let manifest = if manifest_path.exists() {
            if !config.resume {
                return Err(DataError::DatasetExists(config.output.clone()));
            }
            let manifest: OpponentIntentDatasetManifest =
                serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
            validate_resume(&manifest, config)?;
            validate_opponent_intent_dataset(&config.output, &manifest)?;
            manifest
        } else {
            let now = unix_seconds()?;
            OpponentIntentDatasetManifest {
                schema_version: OPPONENT_INTENT_DATASET_SCHEMA_VERSION,
                dataset_id: dataset_id(config)?,
                feature_schema: OPPONENT_INTENT_FEATURE_SCHEMA.to_owned(),
                target_schema: OPPONENT_INTENT_TARGET_SCHEMA.to_owned(),
                record_size: OPPONENT_INTENT_RECORD_SIZE,
                game: GameConfig::research_aaaaa(4)?,
                split: config.split,
                first_game_index: config.first_game_index,
                requested_games: config.games,
                completed_games: 0,
                total_records: 0,
                windows_per_game: WINDOWS_PER_GAME,
                history_length: OPPONENT_INTENT_HISTORY_LENGTH,
                cohort: config.cohort.clone(),
                policy_assignment_domain: "cascadia-v2-o1-seat-policy-v1".to_owned(),
                policy_identity_observable: false,
                game_index_observable: false,
                strategy_switch_targets_available: false,
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

    pub fn root(&self) -> &Path {
        &self.output
    }

    pub fn manifest(&self) -> &OpponentIntentDatasetManifest {
        &self.manifest
    }

    pub fn append_shard(
        &mut self,
        first_game_index: u64,
        game_count: usize,
        records: &[OpponentIntentRecord],
    ) -> Result<(), DataError> {
        if game_count == 0 || records.len() != game_count * WINDOWS_PER_GAME {
            return Err(DataError::InvalidConfig(
                "opponent-intent shard has the wrong record count",
            ));
        }
        let expected_first = self.manifest.first_game_index + self.manifest.completed_games as u64;
        if first_game_index != expected_first
            || self.manifest.completed_games + game_count > self.manifest.requested_games
        {
            return Err(DataError::InvalidConfig(
                "opponent-intent shard game range is not contiguous",
            ));
        }
        let file = format!("shard-{:05}.o1i", self.manifest.shards.len());
        let path = self.output.join(&file);
        write_shard(
            &path,
            self.manifest.split,
            first_game_index,
            game_count,
            records,
        )?;
        let metadata = fs::metadata(&path)?;
        self.manifest.shards.push(RankingShardManifest {
            file,
            first_game_index,
            game_count,
            group_count: records.len(),
            record_count: records.len(),
            byte_count: metadata.len(),
            blake3: checksum_file(&path)?,
        });
        self.manifest.completed_games += game_count;
        self.manifest.total_records += records.len();
        self.manifest.updated_unix_seconds = unix_seconds()?;
        write_manifest_atomic(&self.manifest_path, &self.manifest)?;
        Ok(())
    }
}

#[derive(Debug)]
struct TurnTrace {
    active_seat: usize,
    position_for_previous_focal: Option<PositionRecord>,
    tile_ids: [u8; 4],
    wildlife: [u8; 4],
    selected_tile_id: u8,
    action: PublicActionRecord,
}

pub fn collect_opponent_intent_dataset(
    config: &OpponentIntentDatasetConfig,
) -> Result<OpponentIntentDatasetManifest, DataError> {
    config.validate()?;
    let mut writer = OpponentIntentDatasetWriter::open(config)?;
    while writer.manifest.completed_games < config.games {
        let game_count = config
            .shard_games
            .min(config.games - writer.manifest.completed_games);
        let first_game_index = config.first_game_index + writer.manifest.completed_games as u64;
        let game_indices =
            (first_game_index..first_game_index + game_count as u64).collect::<Vec<_>>();
        let mut games = game_indices
            .par_iter()
            .map(|game_index| collect_game(config, *game_index))
            .collect::<Result<Vec<_>, _>>()?;
        games.sort_unstable_by_key(|(game_index, _)| *game_index);
        let records = games
            .into_iter()
            .flat_map(|(_, records)| records)
            .collect::<Vec<_>>();
        writer.append_shard(first_game_index, game_count, &records)?;
    }
    validate_opponent_intent_dataset(writer.root(), writer.manifest())?;
    Ok(writer.manifest)
}

fn collect_game(
    config: &OpponentIntentDatasetConfig,
    game_index: u64,
) -> Result<(u64, Vec<OpponentIntentRecord>), DataError> {
    let policies = opponent_intent_seat_policies(config.split, game_index, &config.cohort)?;
    let seat_policy_codes = policies.map(opponent_intent_policy_code);
    let match_config = MatchConfig {
        game: GameConfig::research_aaaaa(4)?,
        seed: opponent_intent_game_seed(config.split, game_index, &config.cohort.cohort_id),
        seats: policies.to_vec(),
    };
    let mut traces = Vec::with_capacity(80);
    let mut trace_error = None;
    let result = play_match_observed(&match_config, |state, action| {
        if trace_error.is_some() {
            return;
        }
        match observe_turn(state, action, game_index) {
            Ok(trace) => traces.push(trace),
            Err(error) => trace_error = Some(error),
        }
    })?;
    if let Some(error) = trace_error {
        return Err(error);
    }
    if traces.len() != 80 || result.turns != 80 {
        return Err(DataError::InvalidShard(
            "opponent-intent game did not contain 80 turns",
        ));
    }
    let final_scores: [u16; 4] = result
        .scores
        .iter()
        .map(|score| score.base_total)
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| DataError::InvalidShard("opponent-intent game score count is not four"))?;
    let mut records = Vec::with_capacity(WINDOWS_PER_GAME);
    for focal_turn in 0..WINDOWS_PER_GAME {
        records.push(build_record(
            game_index,
            focal_turn,
            seat_policy_codes,
            &traces,
            final_scores,
        )?);
    }
    Ok((game_index, records))
}

fn observe_turn(
    game: &GameState,
    action: &TurnAction,
    game_index: u64,
) -> Result<TurnTrace, DataError> {
    let active_seat = game.current_player();
    let position_for_previous_focal = (game.completed_turns() != 0).then(|| {
        let previous_focal = (active_seat + 3) % 4;
        PositionRecord::observe_for_seat(game, game_index, previous_focal)
    });
    let tile_ids = game
        .market()
        .tiles
        .map(|tile| tile.expect("active standard market has four tiles").id.0);
    let wildlife = game
        .market()
        .wildlife
        .map(|wildlife| wildlife.expect("active standard market has four wildlife") as u8);
    let staged = game.preview_market_prelude(&action.prelude())?;
    let tile_slot = match action.draft {
        DraftChoice::Paired { slot } => slot,
        DraftChoice::Independent { tile_slot, .. } => tile_slot,
    };
    let selected_tile_id = staged.market().tiles[tile_slot.index()]
        .ok_or(DataError::InvalidShard(
            "selected opponent-intent tile is unavailable",
        ))?
        .id
        .0;
    Ok(TurnTrace {
        active_seat,
        position_for_previous_focal,
        tile_ids,
        wildlife,
        selected_tile_id,
        action: PublicActionRecord::observe(game, action)?,
    })
}

fn build_record(
    game_index: u64,
    focal_turn: usize,
    seat_policy_codes: [u8; 4],
    traces: &[TurnTrace],
    final_scores: [u16; 4],
) -> Result<OpponentIntentRecord, DataError> {
    let focal_seat = traces[focal_turn].active_seat;
    let source = &traces[focal_turn + 1];
    let position = source
        .position_for_previous_focal
        .clone()
        .ok_or(DataError::InvalidShard(
            "opponent-intent source position is unavailable",
        ))?;
    if usize::from(position.turn) != focal_turn + 1
        || usize::from(position.active_seat) != focal_seat
    {
        return Err(DataError::InvalidShard(
            "opponent-intent source perspective is inconsistent",
        ));
    }

    let history_start = (focal_turn + 1).saturating_sub(OPPONENT_INTENT_HISTORY_LENGTH);
    let mut history = [OpponentIntentHistoryEntry::default(); OPPONENT_INTENT_HISTORY_LENGTH];
    for (slot, trace_index) in (history_start..=focal_turn).enumerate() {
        let trace = &traces[trace_index];
        history[slot] = OpponentIntentHistoryEntry {
            valid: 1,
            age: (focal_turn - trace_index) as u8,
            relative_seat: ((trace.active_seat + 4 - focal_seat) % 4) as u8,
            action: trace.action,
        };
    }
    let history_count = u8::try_from(focal_turn + 1 - history_start)
        .map_err(|_| DataError::InvalidShard("opponent-intent history count overflow"))?;

    let mut opponent_targets = [OpponentActionTarget::default(); 3];
    for (offset, target) in opponent_targets.iter_mut().enumerate() {
        let trace = &traces[focal_turn + 1 + offset];
        let relative_seat = ((trace.active_seat + 4 - focal_seat) % 4) as u8;
        if relative_seat != (offset + 1) as u8 {
            return Err(DataError::InvalidShard(
                "opponent-intent opponent order is inconsistent",
            ));
        }
        *target = OpponentActionTarget {
            relative_seat,
            policy_code: seat_policy_codes[trace.active_seat],
            selected_tile_id: trace.selected_tile_id,
            action: trace.action,
        };
    }

    let next_focal = &traces[focal_turn + 4];
    let mut survival_targets = [TileSurvivalTarget::default(); 4];
    for slot in MarketSlot::ALL {
        let tile_id = source.tile_ids[slot.index()];
        let initial_wildlife = source.wildlife[slot.index()];
        let consumed = opponent_targets
            .iter()
            .position(|target| target.selected_tile_id == tile_id);
        survival_targets[slot.index()] = if let Some(opponent) = consumed {
            TileSurvivalTarget {
                initial_tile_id: tile_id,
                initial_wildlife,
                disposition: (opponent + 1) as u8,
                pair_survives: 0,
                final_slot: NONE,
            }
        } else {
            let final_slot = next_focal
                .tile_ids
                .iter()
                .position(|candidate| *candidate == tile_id)
                .ok_or(DataError::InvalidShard(
                    "opponent-intent tile neither consumed nor available",
                ))?;
            TileSurvivalTarget {
                initial_tile_id: tile_id,
                initial_wildlife,
                disposition: SURVIVED_TO_NEXT_ACCESS,
                pair_survives: u8::from(next_focal.wildlife[final_slot] == initial_wildlife),
                final_slot: final_slot as u8,
            }
        };
    }

    let record = OpponentIntentRecord {
        game_index,
        focal_turn: focal_turn as u8,
        focal_seat: focal_seat as u8,
        seat_policy_codes,
        position,
        history_count,
        history,
        opponent_targets,
        survival_targets,
        final_scores,
    };
    validate_record(&record)?;
    Ok(record)
}

pub fn opponent_intent_game_seed(
    split: DatasetSplit,
    game_index: u64,
    cohort_id: &str,
) -> GameSeed {
    let mut hasher = Hasher::new();
    hasher.update(b"cascadia-v2-o1-game-seed-v1");
    hasher.update(split.id().as_bytes());
    hasher.update(cohort_id.as_bytes());
    hasher.update(&game_index.to_le_bytes());
    GameSeed(*hasher.finalize().as_bytes())
}

pub fn opponent_intent_seat_policies(
    split: DatasetSplit,
    game_index: u64,
    cohort: &OpponentIntentCohort,
) -> Result<[StrategyKind; 4], DataError> {
    cohort.validate()?;
    let mut hasher = Hasher::new();
    hasher.update(b"cascadia-v2-o1-seat-policy-v1");
    hasher.update(split.id().as_bytes());
    hasher.update(cohort.cohort_id.as_bytes());
    hasher.update(&game_index.to_le_bytes());
    let digest = hasher.finalize();
    let bytes = digest.as_bytes();
    let mut policies = std::array::from_fn(|seat| {
        cohort.policy_pool[usize::from(bytes[seat]) % cohort.policy_pool.len()]
    });
    let required_seat = usize::from(bytes[4]) % 4;
    if let Some(required) = cohort.required_policy {
        policies[required_seat] = required;
    }
    if cohort.policy_pool.len() > 1 && policies.iter().all(|policy| *policy == policies[0]) {
        let replacement_seat = (required_seat + 1) % 4;
        let current = cohort
            .policy_pool
            .iter()
            .position(|policy| *policy == policies[replacement_seat])
            .unwrap_or(0);
        policies[replacement_seat] = cohort.policy_pool[(current + 1) % cohort.policy_pool.len()];
    }
    Ok(policies)
}

pub const fn opponent_intent_policy_code(policy: StrategyKind) -> u8 {
    match policy {
        StrategyKind::Random => 0,
        StrategyKind::Greedy => 1,
        StrategyKind::PatternAware => 2,
        StrategyKind::PatternCommitment => 3,
        StrategyKind::PatternCompetition => 4,
        StrategyKind::PatternPortfolio => 5,
    }
}

pub const fn opponent_intent_policy_from_code(code: u8) -> Option<StrategyKind> {
    match code {
        0 => Some(StrategyKind::Random),
        1 => Some(StrategyKind::Greedy),
        2 => Some(StrategyKind::PatternAware),
        3 => Some(StrategyKind::PatternCommitment),
        4 => Some(StrategyKind::PatternCompetition),
        5 => Some(StrategyKind::PatternPortfolio),
        _ => None,
    }
}

pub fn validate_opponent_intent_dataset(
    root: &Path,
    manifest: &OpponentIntentDatasetManifest,
) -> Result<(), DataError> {
    if manifest.schema_version != OPPONENT_INTENT_DATASET_SCHEMA_VERSION
        || manifest.feature_schema != OPPONENT_INTENT_FEATURE_SCHEMA
        || manifest.target_schema != OPPONENT_INTENT_TARGET_SCHEMA
        || manifest.record_size != OPPONENT_INTENT_RECORD_SIZE
        || manifest.game != GameConfig::research_aaaaa(4)?
        || manifest.windows_per_game != WINDOWS_PER_GAME
        || manifest.history_length != OPPONENT_INTENT_HISTORY_LENGTH
        || manifest.policy_assignment_domain != "cascadia-v2-o1-seat-policy-v1"
        || manifest.policy_identity_observable
        || manifest.game_index_observable
        || manifest.strategy_switch_targets_available
    {
        return Err(DataError::InvalidManifest(
            "opponent-intent manifest contract does not match",
        ));
    }
    manifest.cohort.validate()?;
    let mut games = 0usize;
    let mut records = 0usize;
    for shard in &manifest.shards {
        let path = root.join(&shard.file);
        if fs::metadata(&path)?.len() != shard.byte_count {
            return Err(DataError::InvalidManifest(
                "opponent-intent shard byte count mismatch",
            ));
        }
        if checksum_file(&path)? != shard.blake3 {
            return Err(DataError::ChecksumMismatch(path));
        }
        let shard_records = read_opponent_intent_shard_records(root, manifest.split, shard)?;
        if shard_records.len() != shard.record_count
            || shard.group_count != shard.record_count
            || shard.record_count != shard.game_count * WINDOWS_PER_GAME
        {
            return Err(DataError::InvalidManifest(
                "opponent-intent shard record count mismatch",
            ));
        }
        validate_shard_records(manifest, shard, &shard_records)?;
        games += shard.game_count;
        records += shard.record_count;
    }
    if games != manifest.completed_games
        || records != manifest.total_records
        || records != games * WINDOWS_PER_GAME
    {
        return Err(DataError::InvalidManifest(
            "opponent-intent manifest totals do not match",
        ));
    }
    Ok(())
}

pub fn read_opponent_intent_shard_records(
    root: &Path,
    split: DatasetSplit,
    shard: &RankingShardManifest,
) -> Result<Vec<OpponentIntentRecord>, DataError> {
    let path = root.join(&shard.file);
    let mut reader = BufReader::new(File::open(path)?);
    let mut header = [0; OPPONENT_INTENT_HEADER_SIZE];
    reader.read_exact(&mut header)?;
    validate_header(&header, split, shard)?;
    let mut records = Vec::with_capacity(shard.record_count);
    for _ in 0..shard.record_count {
        let mut bytes = [0; OPPONENT_INTENT_RECORD_SIZE];
        reader.read_exact(&mut bytes)?;
        records.push(OpponentIntentRecord::from_bytes(&bytes));
    }
    if reader.read(&mut [0; 1])? != 0 {
        return Err(DataError::InvalidShard(
            "opponent-intent shard has trailing bytes",
        ));
    }
    Ok(records)
}

fn validate_record(record: &OpponentIntentRecord) -> Result<(), DataError> {
    if usize::from(record.focal_turn) >= WINDOWS_PER_GAME
        || record.focal_seat >= 4
        || record.position.game_index != record.game_index
        || record.position.turn != record.focal_turn + 1
        || record.position.active_seat != record.focal_seat
        || record.history_count == 0
        || usize::from(record.history_count) > OPPONENT_INTENT_HISTORY_LENGTH
        || record
            .seat_policy_codes
            .iter()
            .any(|code| opponent_intent_policy_from_code(*code).is_none())
    {
        return Err(DataError::InvalidShard(
            "opponent-intent record metadata is invalid",
        ));
    }
    let valid_history = record
        .history
        .iter()
        .filter(|entry| entry.valid != 0)
        .count();
    let expected_history_count =
        (usize::from(record.focal_turn) + 1).min(OPPONENT_INTENT_HISTORY_LENGTH);
    if valid_history != usize::from(record.history_count)
        || valid_history != expected_history_count
        || record.history.iter().enumerate().any(|(index, entry)| {
            if index < valid_history {
                entry.valid != 1
                    || entry.relative_seat > 3
                    || usize::from(entry.age) != valid_history - 1 - index
            } else {
                entry.valid != 0
            }
        })
        || record.history[valid_history - 1].relative_seat != 0
    {
        return Err(DataError::InvalidShard(
            "opponent-intent history is invalid",
        ));
    }
    for (index, target) in record.opponent_targets.iter().enumerate() {
        let absolute_seat =
            (usize::from(record.focal_seat) + usize::from(target.relative_seat)) % 4;
        if target.relative_seat != (index + 1) as u8
            || opponent_intent_policy_from_code(target.policy_code).is_none()
            || target.policy_code != record.seat_policy_codes[absolute_seat]
        {
            return Err(DataError::InvalidShard(
                "opponent-intent action target is invalid",
            ));
        }
    }
    let unique_tiles = record
        .survival_targets
        .iter()
        .map(|target| target.initial_tile_id)
        .collect::<BTreeSet<_>>();
    if unique_tiles.len() != 4 {
        return Err(DataError::InvalidShard(
            "opponent-intent survival targets repeat tile identity",
        ));
    }
    let mut disposition_counts = [0usize; 5];
    for target in record.survival_targets {
        if !(1..=SURVIVED_TO_NEXT_ACCESS).contains(&target.disposition)
            || target.pair_survives > 1
            || (target.disposition == SURVIVED_TO_NEXT_ACCESS && target.final_slot >= 4)
            || (target.disposition != SURVIVED_TO_NEXT_ACCESS
                && (target.final_slot != NONE || target.pair_survives != 0))
        {
            return Err(DataError::InvalidShard(
                "opponent-intent survival target is invalid",
            ));
        }
        disposition_counts[usize::from(target.disposition)] += 1;
    }
    if disposition_counts[1..4].iter().any(|count| *count > 1) {
        return Err(DataError::InvalidShard(
            "opponent-intent opponent consumes multiple initial tiles",
        ));
    }
    Ok(())
}

fn validate_shard_records(
    manifest: &OpponentIntentDatasetManifest,
    shard: &RankingShardManifest,
    records: &[OpponentIntentRecord],
) -> Result<(), DataError> {
    let policy_pool = manifest
        .cohort
        .policy_pool
        .iter()
        .copied()
        .map(opponent_intent_policy_code)
        .collect::<BTreeSet<_>>();
    let required_policy = manifest
        .cohort
        .required_policy
        .map(opponent_intent_policy_code);
    for (game_offset, game_records) in records.chunks_exact(WINDOWS_PER_GAME).enumerate() {
        let expected_game_index = shard.first_game_index + game_offset as u64;
        let seat_policy_codes = game_records[0].seat_policy_codes;
        if game_records.iter().enumerate().any(|(turn, record)| {
            record.game_index != expected_game_index
                || usize::from(record.focal_turn) != turn
                || record.seat_policy_codes != seat_policy_codes
        }) || seat_policy_codes
            .iter()
            .any(|code| !policy_pool.contains(code))
            || required_policy.is_some_and(|required| !seat_policy_codes.contains(&required))
        {
            return Err(DataError::InvalidShard(
                "opponent-intent game sequence or policy cohort is invalid",
            ));
        }
        for record in game_records {
            validate_record(record)?;
        }
    }
    Ok(())
}

fn dataset_id(config: &OpponentIntentDatasetConfig) -> Result<String, DataError> {
    let mut hasher = Hasher::new();
    hasher.update(b"cascadia-v2-opponent-intent-dataset-v1");
    hasher.update(config.split.id().as_bytes());
    hasher.update(&config.first_game_index.to_le_bytes());
    hasher.update(&(config.games as u64).to_le_bytes());
    hasher.update(&serde_json::to_vec(&config.cohort)?);
    Ok(format!(
        "opponent-intent-{}-{}-{}",
        config.split.id(),
        config.cohort.cohort_id,
        &hasher.finalize().to_hex()[..16]
    ))
}

fn validate_resume(
    manifest: &OpponentIntentDatasetManifest,
    config: &OpponentIntentDatasetConfig,
) -> Result<(), DataError> {
    if manifest.schema_version != OPPONENT_INTENT_DATASET_SCHEMA_VERSION
        || manifest.split != config.split
        || manifest.first_game_index != config.first_game_index
        || manifest.requested_games != config.games
        || manifest.cohort != config.cohort
        || !collection_provenance_matches(&manifest.provenance, &collection_provenance()?)
    {
        return Err(DataError::ResumeMismatch);
    }
    Ok(())
}

fn write_shard(
    path: &Path,
    split: DatasetSplit,
    first_game_index: u64,
    game_count: usize,
    records: &[OpponentIntentRecord],
) -> Result<(), DataError> {
    let temporary = path.with_extension("o1i.tmp");
    let mut writer = BufWriter::new(File::create(&temporary)?);
    let header = shard_header(split, first_game_index, game_count, records.len());
    writer.write_all(&header)?;
    for record in records {
        writer.write_all(&record.to_bytes())?;
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    fs::rename(temporary, path)?;
    Ok(())
}

fn shard_header(
    split: DatasetSplit,
    first_game_index: u64,
    game_count: usize,
    record_count: usize,
) -> [u8; OPPONENT_INTENT_HEADER_SIZE] {
    let mut header = [0; OPPONENT_INTENT_HEADER_SIZE];
    let mut offset = 0;
    write_slice(&mut header, &mut offset, OPPONENT_INTENT_SHARD_MAGIC);
    write_slice(
        &mut header,
        &mut offset,
        &OPPONENT_INTENT_DATASET_SCHEMA_VERSION.to_le_bytes(),
    );
    write_slice(
        &mut header,
        &mut offset,
        &(OPPONENT_INTENT_RECORD_SIZE as u32).to_le_bytes(),
    );
    write_slice(&mut header, &mut offset, &[split_code(split)]);
    write_slice(&mut header, &mut offset, &first_game_index.to_le_bytes());
    write_slice(&mut header, &mut offset, &(game_count as u32).to_le_bytes());
    write_slice(
        &mut header,
        &mut offset,
        &(record_count as u64).to_le_bytes(),
    );
    header
}

fn validate_header(
    header: &[u8; OPPONENT_INTENT_HEADER_SIZE],
    split: DatasetSplit,
    shard: &RankingShardManifest,
) -> Result<(), DataError> {
    let mut offset = 0;
    if &read_array::<8>(header, &mut offset) != OPPONENT_INTENT_SHARD_MAGIC
        || u16::from_le_bytes(read_array(header, &mut offset))
            != OPPONENT_INTENT_DATASET_SCHEMA_VERSION
        || u32::from_le_bytes(read_array(header, &mut offset)) as usize
            != OPPONENT_INTENT_RECORD_SIZE
        || read_array::<1>(header, &mut offset)[0] != split_code(split)
        || u64::from_le_bytes(read_array(header, &mut offset)) != shard.first_game_index
        || u32::from_le_bytes(read_array(header, &mut offset)) as usize != shard.game_count
        || u64::from_le_bytes(read_array(header, &mut offset)) as usize != shard.record_count
    {
        return Err(DataError::InvalidShard(
            "opponent-intent shard header does not match",
        ));
    }
    Ok(())
}

const fn split_code(split: DatasetSplit) -> u8 {
    match split {
        DatasetSplit::Train => 0,
        DatasetSplit::Validation => 1,
        DatasetSplit::Test => 2,
        DatasetSplit::Final => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cohort() -> OpponentIntentCohort {
        OpponentIntentCohort {
            cohort_id: "unit-mixed".to_owned(),
            policy_pool: vec![
                StrategyKind::Random,
                StrategyKind::Greedy,
                StrategyKind::PatternAware,
            ],
            required_policy: Some(StrategyKind::PatternAware),
        }
    }

    #[test]
    fn seat_schedule_is_deterministic_mixed_and_contains_required_policy() {
        let first = opponent_intent_seat_policies(DatasetSplit::Validation, 7, &cohort()).unwrap();
        let second = opponent_intent_seat_policies(DatasetSplit::Validation, 7, &cohort()).unwrap();
        assert_eq!(first, second);
        assert!(first.contains(&StrategyKind::PatternAware));
        assert!(first.iter().any(|policy| *policy != first[0]));
    }

    #[test]
    fn cohort_and_split_participate_in_game_seed() {
        assert_ne!(
            opponent_intent_game_seed(DatasetSplit::Train, 1, "a"),
            opponent_intent_game_seed(DatasetSplit::Validation, 1, "a")
        );
        assert_ne!(
            opponent_intent_game_seed(DatasetSplit::Train, 1, "a"),
            opponent_intent_game_seed(DatasetSplit::Train, 1, "b")
        );
    }

    #[test]
    fn model_input_excludes_game_and_policy_identity() {
        let config = OpponentIntentDatasetConfig {
            output: PathBuf::new(),
            split: DatasetSplit::Train,
            first_game_index: 0,
            games: 1,
            shard_games: 1,
            cohort: OpponentIntentCohort {
                cohort_id: "unit-random".to_owned(),
                policy_pool: vec![StrategyKind::Random],
                required_policy: None,
            },
            resume: false,
        };
        let (_, records) = collect_game(&config, 0).unwrap();
        let mut changed = records[0].clone();
        changed.game_index = 99;
        changed.position.game_index = 99;
        changed.seat_policy_codes = [5, 4, 3, 2];
        for target in &mut changed.opponent_targets {
            target.policy_code = 5;
        }
        assert_eq!(records[0].model_input_bytes(), changed.model_input_bytes());
        assert_ne!(records[0].to_bytes(), changed.to_bytes());
    }

    #[test]
    fn random_game_yields_exact_complete_windows_and_roundtrips() {
        let config = OpponentIntentDatasetConfig {
            output: PathBuf::new(),
            split: DatasetSplit::Train,
            first_game_index: 0,
            games: 1,
            shard_games: 1,
            cohort: OpponentIntentCohort {
                cohort_id: "unit-random".to_owned(),
                policy_pool: vec![StrategyKind::Random],
                required_policy: None,
            },
            resume: false,
        };
        let (_, records) = collect_game(&config, 0).unwrap();
        assert_eq!(records.len(), WINDOWS_PER_GAME);
        for (turn, record) in records.iter().enumerate() {
            assert_eq!(usize::from(record.focal_turn), turn);
            validate_record(record).unwrap();
            assert_eq!(
                OpponentIntentRecord::from_bytes(&record.to_bytes()),
                *record
            );
        }
    }
}
