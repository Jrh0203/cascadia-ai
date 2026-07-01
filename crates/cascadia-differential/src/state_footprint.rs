//! Deterministic state-footprint census for compact Cascadia representations.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use cascadia_data::{
    BOARD_SLOTS, DatasetManifest, GradedOracleDatasetManifest, GradedOracleGroup, PositionRecord,
    PositionShardReader, RECORD_SIZE, validate_dataset, validate_graded_oracle_dataset,
};
use cascadia_game::{
    Board, GRID_DIM, GRID_RADIUS, GRID_SIZE, GameConfig, GameSeed, HexCoord, Rotation,
    STANDARD_TILES, Terrain, TurnAction, Wildlife,
};
use cascadia_provenance::{SourceProvenance, checksum_file, source_provenance};
use cascadia_sim::{MatchConfig, StrategyKind, play_match_observed};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const STATE_FOOTPRINT_SCHEMA_VERSION: u16 = 1;
pub const STATE_FOOTPRINT_EXPERIMENT_ID: &str = "state-footprint-census-v1";
pub const STATE_FOOTPRINT_RADII: [u8; 6] = [3, 4, 5, 6, 7, 8];
pub const STATE_FOOTPRINT_OUTLIER_RADIUS: u8 = 6;
pub const PREREGISTERED_FIRST_SEED: u64 = 73_000;
pub const PREREGISTERED_GAMES: usize = 625;
pub const PREREGISTERED_STATES: u64 = 50_000;
pub const PREREGISTERED_BOARD_OBSERVATIONS: u64 = 200_000;

const NONE: u8 = u8::MAX;
const DENSE_CHANNEL_ESTIMATE: u64 = 11;
const SCIENTIFIC_HASH_ALGORITHM: &str =
    "blake3(compact-serde-json; ordered-struct-fields; btree-map-keys; btree-set-values)-v1";

#[derive(Debug, Error)]
pub enum StateFootprintError {
    #[error("invalid census configuration: {0}")]
    InvalidConfig(String),
    #[error("census invariant failed: {0}")]
    Invariant(String),
    #[error(transparent)]
    Data(#[from] cascadia_data::DataError),
    #[error(transparent)]
    Rules(#[from] cascadia_game::RuleError),
    #[error(transparent)]
    Board(#[from] cascadia_game::BoardError),
    #[error(transparent)]
    Simulation(#[from] cascadia_sim::SimulationError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone)]
pub struct StateFootprintConfig {
    pub first_seed: u64,
    pub games: usize,
    pub strategy: StrategyKind,
    pub position_dataset_roots: Vec<PathBuf>,
    pub graded_dataset_roots: Vec<PathBuf>,
    pub outlier_cap: usize,
}

impl StateFootprintConfig {
    pub fn validate(&self) -> Result<(), StateFootprintError> {
        if self.games == 0
            && self.position_dataset_roots.is_empty()
            && self.graded_dataset_roots.is_empty()
        {
            return Err(StateFootprintError::InvalidConfig(
                "games may be zero only when at least one position or graded dataset root is supplied"
                    .to_owned(),
            ));
        }
        if self.outlier_cap == 0 {
            return Err(StateFootprintError::InvalidConfig(
                "outlier cap must be positive".to_owned(),
            ));
        }
        let mut roots = BTreeSet::new();
        for root in self
            .position_dataset_roots
            .iter()
            .chain(&self.graded_dataset_roots)
        {
            let canonical = fs::canonicalize(root).map_err(|error| {
                StateFootprintError::InvalidConfig(format!(
                    "cannot canonicalize dataset root {}: {error}",
                    root.display()
                ))
            })?;
            if !roots.insert(canonical) {
                return Err(StateFootprintError::InvalidConfig(format!(
                    "dataset root {} was supplied more than once",
                    root.display()
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateFootprintReport {
    pub schema_version: u16,
    pub experiment_id: String,
    pub created_unix_seconds: u64,
    pub output_path: String,
    pub provenance: ReportProvenance,
    pub runtime: CensusRuntime,
    pub scientific_hash_algorithm: String,
    pub scientific_hash: String,
    pub scientific: ScientificPayload,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReportProvenance {
    pub source: SourceProvenance,
    pub executable_path: String,
    pub executable_blake3: String,
    pub current_v2_grid_radius: i8,
    pub current_v2_grid_dim: usize,
    pub current_v2_grid_size: usize,
    pub historical_legacy_nnue_cell_shape: usize,
    pub dataset_paths: Vec<DatasetPathProvenance>,
    pub merged_inputs: Vec<MergedInputProvenance>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetPathProvenance {
    pub kind: String,
    pub dataset_id: String,
    pub root: String,
    pub manifest_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergedInputProvenance {
    pub path: String,
    pub report_blake3: String,
    pub scientific_hash: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CensusRuntime {
    pub total_wall_seconds: f64,
    pub generated: CorpusRuntime,
    pub position_datasets: CorpusRuntime,
    pub graded_datasets: CorpusRuntime,
    pub merge_wall_seconds: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CorpusRuntime {
    pub parallel_wall_seconds: f64,
    pub summed_source_wall_seconds: f64,
    pub validation_seconds: f64,
    pub read_seconds: f64,
    pub extraction_build_seconds: f64,
    pub simulation_excluding_extraction_seconds: f64,
}

impl CorpusRuntime {
    fn merge_from(&mut self, other: &Self) {
        self.parallel_wall_seconds += other.parallel_wall_seconds;
        self.summed_source_wall_seconds += other.summed_source_wall_seconds;
        self.validation_seconds += other.validation_seconds;
        self.read_seconds += other.read_seconds;
        self.extraction_build_seconds += other.extraction_build_seconds;
        self.simulation_excluding_extraction_seconds +=
            other.simulation_excluding_extraction_seconds;
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificPayload {
    pub schema_version: u16,
    pub experiment_id: String,
    pub ruleset: RulesetIdentity,
    pub configuration: ScientificConfiguration,
    pub definitions: BTreeMap<String, String>,
    pub invariants: GeometryInvariants,
    pub generated: Option<CorpusScientific>,
    pub position_datasets: Option<CorpusScientific>,
    pub graded_oracle: Option<CorpusScientific>,
    pub adversarial_cases: Vec<AdversarialCaseReport>,
    pub completion: CompletionAssessment,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RulesetIdentity {
    pub player_count: u8,
    pub scoring_cards: String,
    pub habitat_bonuses: bool,
    pub turns: u16,
    pub current_v2_grid_radius: i8,
    pub current_v2_grid_dim: usize,
    pub current_v2_grid_size: usize,
    pub current_control_support: String,
    pub historical_legacy_nnue_cell_shape: usize,
    pub historical_legacy_nnue_role: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScientificConfiguration {
    pub radii: Vec<u8>,
    pub outlier_radius: u8,
    pub outlier_cap: usize,
    pub generated_origins: Vec<GeneratedOrigin>,
    pub position_datasets: Vec<PositionDatasetIdentity>,
    pub graded_datasets: Vec<GradedDatasetIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GeneratedOrigin {
    pub first_seed: u64,
    pub games: usize,
    pub strategy_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FileIdentity {
    pub file: String,
    pub byte_count: u64,
    pub blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PositionDatasetIdentity {
    pub dataset_id: String,
    pub manifest_blake3: String,
    pub split: String,
    pub strategy_id: String,
    pub first_game_index: u64,
    pub completed_games: usize,
    pub total_records: usize,
    pub shards: Vec<FileIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GradedDatasetIdentity {
    pub dataset_id: String,
    pub manifest_blake3: String,
    pub split: String,
    pub completed_games: usize,
    pub total_groups: usize,
    pub total_candidate_rows: usize,
    pub seeds: Vec<u64>,
    pub shards: Vec<FileIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeometryInvariants {
    pub centered_hex_capacity_formula: String,
    pub radius_4_capacity: u64,
    pub radius_5_capacity: u64,
    pub radius_6_capacity: u64,
    pub complete_121_cell_disk_exists: bool,
    pub d6_transform_count: usize,
    pub d6_radius_invariant: bool,
    pub recentering_is_integer_exact_and_invertible: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletionAssessment {
    pub classification: String,
    pub complete: bool,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorpusScientific {
    pub merge_accumulator: CorpusAccumulator,
    pub derived: CorpusDerived,
}

impl CorpusScientific {
    fn new(accumulator: CorpusAccumulator) -> Self {
        let derived = CorpusDerived::from_accumulator(&accumulator);
        Self {
            merge_accumulator: accumulator,
            derived,
        }
    }

    fn validate(&self) -> Result<(), StateFootprintError> {
        self.merge_accumulator.validate()?;
        let expected = CorpusDerived::from_accumulator(&self.merge_accumulator);
        if self.derived != expected {
            return Err(StateFootprintError::Invariant(
                "derived corpus report does not match its merge accumulator".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistogramAccumulator {
    pub bins: BTreeMap<u16, u64>,
}

impl HistogramAccumulator {
    fn observe(&mut self, value: usize) -> Result<(), StateFootprintError> {
        let value = u16::try_from(value).map_err(|_| {
            StateFootprintError::Invariant(format!("histogram value {value} exceeds u16"))
        })?;
        *self.bins.entry(value).or_default() += 1;
        Ok(())
    }

    fn merge_from(&mut self, other: &Self) {
        for (value, count) in &other.bins {
            *self.bins.entry(*value).or_default() += count;
        }
    }

    fn observations(&self) -> u64 {
        self.bins.values().sum()
    }

    fn sum(&self) -> u64 {
        self.bins
            .iter()
            .map(|(value, count)| u64::from(*value) * count)
            .sum()
    }

    fn maximum(&self) -> Option<u16> {
        self.bins.last_key_value().map(|(value, _)| *value)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetentionAccumulator {
    pub total: u64,
    pub retained: u64,
    pub overflow: u64,
}

impl RetentionAccumulator {
    fn observe_flags(&mut self, retained: u64, overflow: u64) {
        self.total += retained + overflow;
        self.retained += retained;
        self.overflow += overflow;
    }

    fn observe_coordinates(&mut self, coordinates: &[HexCoord], center: HexCoord, radius: u8) {
        let retained = coordinates
            .iter()
            .filter(|coord| coord.distance(center) <= radius)
            .count() as u64;
        self.observe_flags(retained, coordinates.len() as u64 - retained);
    }

    fn merge_from(&mut self, other: Self) {
        self.total += other.total;
        self.retained += other.retained;
        self.overflow += other.overflow;
    }

    fn validate(self, label: &str) -> Result<(), StateFootprintError> {
        if self.retained + self.overflow != self.total {
            return Err(StateFootprintError::Invariant(format!(
                "{label} retention counts do not sum"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundaryAccumulator {
    pub retention: RetentionAccumulator,
    pub crossing: u64,
    pub fully_outside: u64,
}

impl BoundaryAccumulator {
    fn observe_sets(&mut self, sets: &[Vec<HexCoord>], center: HexCoord, radius: u8) {
        for set in sets {
            let inside = set
                .iter()
                .filter(|coord| coord.distance(center) <= radius)
                .count();
            if inside == set.len() {
                self.retention.observe_flags(1, 0);
            } else {
                self.retention.observe_flags(0, 1);
                if inside == 0 {
                    self.fully_outside += 1;
                } else {
                    self.crossing += 1;
                }
            }
        }
    }

    fn merge_from(&mut self, other: Self) {
        self.retention.merge_from(other.retention);
        self.crossing += other.crossing;
        self.fully_outside += other.fully_outside;
    }

    fn validate(self, label: &str) -> Result<(), StateFootprintError> {
        self.retention.validate(label)?;
        if self.crossing + self.fully_outside != self.retention.overflow {
            return Err(StateFootprintError::Invariant(format!(
                "{label} crossing and fully-outside counts do not sum to overflow"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RadiusAccumulator {
    pub occupied_cells: RetentionAccumulator,
    pub frontier_cells: RetentionAccumulator,
    pub selected_action_destinations: RetentionAccumulator,
    pub complete_candidate_destinations: RetentionAccumulator,
    pub wildlife_firings: RetentionAccumulator,
    pub terrain_edge_firings: RetentionAccumulator,
    pub allowed_wildlife_firings: RetentionAccumulator,
    pub habitat_components: BoundaryAccumulator,
    pub wildlife_adjacencies: BoundaryAccumulator,
    pub sparse_occupied_plus_frontier_tokens: RetentionAccumulator,
    pub board_observations: u64,
    pub boards_with_any_overflow: u64,
    pub states_with_selected_destination: u64,
    pub states_with_selected_destination_overflow: u64,
    pub groups_with_candidate_destinations: u64,
    pub groups_with_candidate_destination_overflow: u64,
}

impl RadiusAccumulator {
    fn merge_from(&mut self, other: Self) {
        self.occupied_cells.merge_from(other.occupied_cells);
        self.frontier_cells.merge_from(other.frontier_cells);
        self.selected_action_destinations
            .merge_from(other.selected_action_destinations);
        self.complete_candidate_destinations
            .merge_from(other.complete_candidate_destinations);
        self.wildlife_firings.merge_from(other.wildlife_firings);
        self.terrain_edge_firings
            .merge_from(other.terrain_edge_firings);
        self.allowed_wildlife_firings
            .merge_from(other.allowed_wildlife_firings);
        self.habitat_components.merge_from(other.habitat_components);
        self.wildlife_adjacencies
            .merge_from(other.wildlife_adjacencies);
        self.sparse_occupied_plus_frontier_tokens
            .merge_from(other.sparse_occupied_plus_frontier_tokens);
        self.board_observations += other.board_observations;
        self.boards_with_any_overflow += other.boards_with_any_overflow;
        self.states_with_selected_destination += other.states_with_selected_destination;
        self.states_with_selected_destination_overflow +=
            other.states_with_selected_destination_overflow;
        self.groups_with_candidate_destinations += other.groups_with_candidate_destinations;
        self.groups_with_candidate_destination_overflow +=
            other.groups_with_candidate_destination_overflow;
    }

    fn validate(&self, label: &str) -> Result<(), StateFootprintError> {
        self.occupied_cells
            .validate(&format!("{label} occupied cells"))?;
        self.frontier_cells
            .validate(&format!("{label} frontier cells"))?;
        self.selected_action_destinations
            .validate(&format!("{label} selected destinations"))?;
        self.complete_candidate_destinations
            .validate(&format!("{label} candidate destinations"))?;
        self.wildlife_firings
            .validate(&format!("{label} wildlife firings"))?;
        self.terrain_edge_firings
            .validate(&format!("{label} terrain-edge firings"))?;
        self.allowed_wildlife_firings
            .validate(&format!("{label} allowed-wildlife firings"))?;
        self.habitat_components
            .validate(&format!("{label} habitat components"))?;
        self.wildlife_adjacencies
            .validate(&format!("{label} wildlife adjacencies"))?;
        self.sparse_occupied_plus_frontier_tokens
            .validate(&format!("{label} sparse tokens"))?;
        if self.boards_with_any_overflow > self.board_observations
            || self.states_with_selected_destination_overflow
                > self.states_with_selected_destination
            || self.groups_with_candidate_destination_overflow
                > self.groups_with_candidate_destinations
        {
            return Err(StateFootprintError::Invariant(format!(
                "{label} observation overflow counts exceed their totals"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RadiusSetAccumulator {
    pub rows: [RadiusAccumulator; STATE_FOOTPRINT_RADII.len()],
}

impl Default for RadiusSetAccumulator {
    fn default() -> Self {
        Self {
            rows: std::array::from_fn(|_| RadiusAccumulator::default()),
        }
    }
}

impl RadiusSetAccumulator {
    fn merge_from(&mut self, other: Self) {
        for (target, source) in self.rows.iter_mut().zip(other.rows) {
            target.merge_from(source);
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DualRadiusHistograms {
    pub fixed_origin: HistogramAccumulator,
    pub best_integer_recentered: HistogramAccumulator,
}

impl DualRadiusHistograms {
    fn observe(&mut self, fixed: usize, recentered: usize) -> Result<(), StateFootprintError> {
        self.fixed_origin.observe(fixed)?;
        self.best_integer_recentered.observe(recentered)
    }

    fn merge_from(&mut self, other: &Self) {
        self.fixed_origin.merge_from(&other.fixed_origin);
        self.best_integer_recentered
            .merge_from(&other.best_integer_recentered);
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupportAccumulator {
    pub occupied_radius: DualRadiusHistograms,
    pub frontier_radius: DualRadiusHistograms,
    pub selected_action_destination_radius: DualRadiusHistograms,
    pub complete_candidate_destination_radius: DualRadiusHistograms,
    pub fixed_origin: RadiusSetAccumulator,
    pub best_integer_recentered: RadiusSetAccumulator,
}

impl SupportAccumulator {
    fn merge_from(&mut self, other: Self) {
        self.occupied_radius.merge_from(&other.occupied_radius);
        self.frontier_radius.merge_from(&other.frontier_radius);
        self.selected_action_destination_radius
            .merge_from(&other.selected_action_destination_radius);
        self.complete_candidate_destination_radius
            .merge_from(&other.complete_candidate_destination_radius);
        self.fixed_origin.merge_from(other.fixed_origin);
        self.best_integer_recentered
            .merge_from(other.best_integer_recentered);
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageAccumulator {
    pub occupied_cells_per_board: HistogramAccumulator,
    pub frontier_cells_per_board: HistogramAccumulator,
    pub sparse_tokens_per_board: HistogramAccumulator,
    pub canonical_public_state_bytes: HistogramAccumulator,
    pub serialized_position_record_bytes: HistogramAccumulator,
}

impl StorageAccumulator {
    fn merge_from(&mut self, other: &Self) {
        self.occupied_cells_per_board
            .merge_from(&other.occupied_cells_per_board);
        self.frontier_cells_per_board
            .merge_from(&other.frontier_cells_per_board);
        self.sparse_tokens_per_board
            .merge_from(&other.sparse_tokens_per_board);
        self.canonical_public_state_bytes
            .merge_from(&other.canonical_public_state_bytes);
        self.serialized_position_record_bytes
            .merge_from(&other.serialized_position_record_bytes);
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateCohortsAccumulator {
    pub phase: BTreeMap<String, u64>,
    pub current_player: BTreeMap<String, u64>,
}

impl StateCohortsAccumulator {
    fn merge_from(&mut self, other: &Self) {
        merge_u64_map(&mut self.phase, &other.phase);
        merge_u64_map(&mut self.current_player, &other.current_player);
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardCohortAccumulator {
    pub board_observations: u64,
    pub occupied_radius: DualRadiusHistograms,
    pub frontier_radius: DualRadiusHistograms,
    pub selected_destination_radius: DualRadiusHistograms,
    pub fixed_origin_boards_with_any_overflow: [u64; STATE_FOOTPRINT_RADII.len()],
    pub recentered_boards_with_any_overflow: [u64; STATE_FOOTPRINT_RADII.len()],
}

impl BoardCohortAccumulator {
    fn observe(&mut self, sample: &BoardCohortSample) -> Result<(), StateFootprintError> {
        self.board_observations += 1;
        self.occupied_radius.observe(
            usize::from(sample.fixed_occupied_radius),
            usize::from(sample.recentered_occupied_radius),
        )?;
        self.frontier_radius.observe(
            usize::from(sample.fixed_frontier_radius),
            usize::from(sample.recentered_frontier_radius),
        )?;
        if let (Some(fixed), Some(recentered)) = (
            sample.fixed_selected_destination_radius,
            sample.recentered_selected_destination_radius,
        ) {
            self.selected_destination_radius
                .observe(usize::from(fixed), usize::from(recentered))?;
        }
        for index in 0..STATE_FOOTPRINT_RADII.len() {
            self.fixed_origin_boards_with_any_overflow[index] +=
                u64::from(sample.fixed_overflow[index]);
            self.recentered_boards_with_any_overflow[index] +=
                u64::from(sample.recentered_overflow[index]);
        }
        Ok(())
    }

    fn merge_from(&mut self, other: &Self) {
        self.board_observations += other.board_observations;
        self.occupied_radius.merge_from(&other.occupied_radius);
        self.frontier_radius.merge_from(&other.frontier_radius);
        self.selected_destination_radius
            .merge_from(&other.selected_destination_radius);
        for index in 0..STATE_FOOTPRINT_RADII.len() {
            self.fixed_origin_boards_with_any_overflow[index] +=
                other.fixed_origin_boards_with_any_overflow[index];
            self.recentered_boards_with_any_overflow[index] +=
                other.recentered_boards_with_any_overflow[index];
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardCohortsAccumulator {
    pub phase: BTreeMap<String, BoardCohortAccumulator>,
    pub absolute_seat: BTreeMap<String, BoardCohortAccumulator>,
    pub focal_relative_seat: BTreeMap<String, BoardCohortAccumulator>,
    pub final_score_band: BTreeMap<String, BoardCohortAccumulator>,
    pub exact_final_score: BTreeMap<String, BoardCohortAccumulator>,
    pub fixed_origin_width: BTreeMap<String, BoardCohortAccumulator>,
    pub recentered_width: BTreeMap<String, BoardCohortAccumulator>,
}

impl BoardCohortsAccumulator {
    fn merge_from(&mut self, other: &Self) {
        merge_cohort_map(&mut self.phase, &other.phase);
        merge_cohort_map(&mut self.absolute_seat, &other.absolute_seat);
        merge_cohort_map(&mut self.focal_relative_seat, &other.focal_relative_seat);
        merge_cohort_map(&mut self.final_score_band, &other.final_score_band);
        merge_cohort_map(&mut self.exact_final_score, &other.exact_final_score);
        merge_cohort_map(&mut self.fixed_origin_width, &other.fixed_origin_width);
        merge_cohort_map(&mut self.recentered_width, &other.recentered_width);
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CensusCohortsAccumulator {
    pub states: StateCohortsAccumulator,
    pub boards: BoardCohortsAccumulator,
}

impl CensusCohortsAccumulator {
    fn merge_from(&mut self, other: &Self) {
        self.states.merge_from(&other.states);
        self.boards.merge_from(&other.boards);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutlierAccumulator {
    pub cap: usize,
    pub total: u64,
    pub records: BTreeSet<OverflowOutlier>,
    pub truncated: bool,
}

impl OutlierAccumulator {
    fn new(cap: usize) -> Self {
        Self {
            cap,
            total: 0,
            records: BTreeSet::new(),
            truncated: false,
        }
    }

    fn observe(&mut self, outlier: OverflowOutlier) {
        self.total += 1;
        self.records.insert(outlier);
        if self.records.len() > self.cap {
            self.records.pop_last();
        }
        self.truncated = self.total > self.records.len() as u64;
    }

    fn merge_from(&mut self, other: Self) -> Result<(), StateFootprintError> {
        if self.cap != other.cap {
            return Err(StateFootprintError::Invariant(format!(
                "cannot merge outlier caps {} and {}",
                self.cap, other.cap
            )));
        }
        self.total += other.total;
        for record in other.records {
            self.records.insert(record);
            if self.records.len() > self.cap {
                self.records.pop_last();
            }
        }
        self.truncated = self.total > self.records.len() as u64;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct OverflowOutlier {
    pub source_kind: String,
    pub dataset_id: Option<String>,
    pub seed_or_game_index: u64,
    pub turn: u16,
    pub decision_group_id: Option<u64>,
    pub current_player: u8,
    pub focal_relative_seat: u8,
    pub absolute_seat: u8,
    pub public_state_hash: String,
    pub recenter_q: i16,
    pub recenter_r: i16,
    pub original_coordinate_bounds: CoordinateBounds,
    pub recentered_coordinate_bounds: CoordinateBounds,
    pub fixed_origin_radii: SupportRadii,
    pub recentered_radii: SupportRadii,
    pub fixed_origin_overflow_at_radius_6: OverflowCoordinates,
    pub recentered_overflow_at_radius_6: OverflowCoordinates,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CoordinateBounds {
    pub min_q: i16,
    pub max_q: i16,
    pub min_r: i16,
    pub max_r: i16,
    pub min_s: i16,
    pub max_s: i16,
    pub max_absolute_q: i16,
    pub max_absolute_r: i16,
    pub max_absolute_s: i16,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SupportRadii {
    pub occupied: u8,
    pub frontier: u8,
    pub selected_destination: Option<u8>,
    pub complete_candidate_destination: Option<u8>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct OverflowCoordinates {
    pub occupied: Vec<HexCoord>,
    pub frontier: Vec<HexCoord>,
    pub selected_destinations: Vec<HexCoord>,
    pub complete_candidate_destinations: Vec<HexCoord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusAccumulator {
    pub state_count: u64,
    pub board_observation_count: u64,
    pub selected_destination_available_states: u64,
    pub selected_destination_unavailable_states: u64,
    pub complete_candidate_rows: u64,
    pub distinct_complete_candidate_destinations: u64,
    pub duplicate_decision_groups_skipped: u64,
    pub support: SupportAccumulator,
    pub storage: StorageAccumulator,
    pub cohorts: CensusCohortsAccumulator,
    pub outliers: OutlierAccumulator,
}

impl CorpusAccumulator {
    fn new(outlier_cap: usize) -> Self {
        Self {
            state_count: 0,
            board_observation_count: 0,
            selected_destination_available_states: 0,
            selected_destination_unavailable_states: 0,
            complete_candidate_rows: 0,
            distinct_complete_candidate_destinations: 0,
            duplicate_decision_groups_skipped: 0,
            support: SupportAccumulator::default(),
            storage: StorageAccumulator::default(),
            cohorts: CensusCohortsAccumulator::default(),
            outliers: OutlierAccumulator::new(outlier_cap),
        }
    }

    fn merge_from(&mut self, other: Self) -> Result<(), StateFootprintError> {
        self.state_count += other.state_count;
        self.board_observation_count += other.board_observation_count;
        self.selected_destination_available_states += other.selected_destination_available_states;
        self.selected_destination_unavailable_states +=
            other.selected_destination_unavailable_states;
        self.complete_candidate_rows += other.complete_candidate_rows;
        self.distinct_complete_candidate_destinations +=
            other.distinct_complete_candidate_destinations;
        self.duplicate_decision_groups_skipped += other.duplicate_decision_groups_skipped;
        self.support.merge_from(other.support);
        self.storage.merge_from(&other.storage);
        self.cohorts.merge_from(&other.cohorts);
        self.outliers.merge_from(other.outliers)
    }

    fn validate(&self) -> Result<(), StateFootprintError> {
        if self.board_observation_count != self.support.occupied_radius.fixed_origin.observations()
            || self.board_observation_count
                != self
                    .support
                    .occupied_radius
                    .best_integer_recentered
                    .observations()
            || self.board_observation_count != self.storage.occupied_cells_per_board.observations()
            || self.board_observation_count != self.storage.frontier_cells_per_board.observations()
            || self.board_observation_count != self.storage.sparse_tokens_per_board.observations()
        {
            return Err(StateFootprintError::Invariant(
                "board observation counts disagree with board histograms".to_owned(),
            ));
        }
        if self.selected_destination_available_states + self.selected_destination_unavailable_states
            != self.state_count
        {
            return Err(StateFootprintError::Invariant(
                "selected destination availability does not cover every state".to_owned(),
            ));
        }
        for (index, radius) in STATE_FOOTPRINT_RADII.into_iter().enumerate() {
            self.support.fixed_origin.rows[index]
                .validate(&format!("fixed-origin radius {radius}"))?;
            self.support.best_integer_recentered.rows[index]
                .validate(&format!("recentered radius {radius}"))?;
        }
        if self.outliers.records.len() > self.outliers.cap
            || self.outliers.records.len() as u64 > self.outliers.total
            || self.outliers.truncated != (self.outliers.total > self.outliers.records.len() as u64)
        {
            return Err(StateFootprintError::Invariant(
                "outlier accumulator is inconsistent".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorpusDerived {
    pub counts: CorpusCounts,
    pub radius_histograms: RadiusHistogramReport,
    pub radius_tables: DualRadiusTableReport,
    pub storage: StorageReport,
    pub cohorts: CensusCohortsAccumulator,
    pub outliers: OutlierSummary,
}

impl CorpusDerived {
    fn from_accumulator(accumulator: &CorpusAccumulator) -> Self {
        Self {
            counts: CorpusCounts {
                states: accumulator.state_count,
                board_observations: accumulator.board_observation_count,
                selected_destination_available_states: accumulator
                    .selected_destination_available_states,
                selected_destination_unavailable_states: accumulator
                    .selected_destination_unavailable_states,
                complete_candidate_rows: accumulator.complete_candidate_rows,
                distinct_complete_candidate_destinations: accumulator
                    .distinct_complete_candidate_destinations,
                duplicate_decision_groups_skipped: accumulator.duplicate_decision_groups_skipped,
            },
            radius_histograms: RadiusHistogramReport::from_support(&accumulator.support),
            radius_tables: DualRadiusTableReport {
                fixed_origin: build_radius_table(&accumulator.support.fixed_origin, accumulator),
                best_integer_recentered: build_radius_table(
                    &accumulator.support.best_integer_recentered,
                    accumulator,
                ),
            },
            storage: StorageReport::from_storage(
                &accumulator.storage,
                accumulator.board_observation_count,
            ),
            cohorts: accumulator.cohorts.clone(),
            outliers: OutlierSummary {
                radius: STATE_FOOTPRINT_OUTLIER_RADIUS,
                cap: accumulator.outliers.cap,
                total: accumulator.outliers.total,
                retained: accumulator.outliers.records.len(),
                truncated: accumulator.outliers.truncated,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusCounts {
    pub states: u64,
    pub board_observations: u64,
    pub selected_destination_available_states: u64,
    pub selected_destination_unavailable_states: u64,
    pub complete_candidate_rows: u64,
    pub distinct_complete_candidate_destinations: u64,
    pub duplicate_decision_groups_skipped: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistogramReport {
    pub observations: u64,
    pub maximum: Option<u16>,
    pub bins: BTreeMap<u16, u64>,
}

impl From<&HistogramAccumulator> for HistogramReport {
    fn from(histogram: &HistogramAccumulator) -> Self {
        Self {
            observations: histogram.observations(),
            maximum: histogram.maximum(),
            bins: histogram.bins.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DualHistogramReport {
    pub fixed_origin: HistogramReport,
    pub best_integer_recentered: HistogramReport,
}

impl From<&DualRadiusHistograms> for DualHistogramReport {
    fn from(histograms: &DualRadiusHistograms) -> Self {
        Self {
            fixed_origin: HistogramReport::from(&histograms.fixed_origin),
            best_integer_recentered: HistogramReport::from(&histograms.best_integer_recentered),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RadiusHistogramReport {
    pub occupied: DualHistogramReport,
    pub frontier: DualHistogramReport,
    pub selected_action_destinations: DualHistogramReport,
    pub complete_candidate_destinations: DualHistogramReport,
}

impl RadiusHistogramReport {
    fn from_support(support: &SupportAccumulator) -> Self {
        Self {
            occupied: DualHistogramReport::from(&support.occupied_radius),
            frontier: DualHistogramReport::from(&support.frontier_radius),
            selected_action_destinations: DualHistogramReport::from(
                &support.selected_action_destination_radius,
            ),
            complete_candidate_destinations: DualHistogramReport::from(
                &support.complete_candidate_destination_radius,
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DualRadiusTableReport {
    pub fixed_origin: Vec<RadiusTableRow>,
    pub best_integer_recentered: Vec<RadiusTableRow>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RadiusTableRow {
    pub radius: u8,
    pub exact_centered_hex_capacity: u64,
    pub occupied_cells: RetentionReport,
    pub frontier_cells: RetentionReport,
    pub selected_action_destinations: RetentionReport,
    pub complete_candidate_destinations: RetentionReport,
    pub wildlife_firings: RetentionReport,
    pub terrain_edge_firings: RetentionReport,
    pub allowed_wildlife_firings: RetentionReport,
    pub habitat_components: BoundaryReport,
    pub wildlife_adjacencies: BoundaryReport,
    pub sparse_occupied_plus_frontier_tokens: RetentionReport,
    pub board_observations: u64,
    pub boards_with_any_overflow: u64,
    pub boards_with_any_overflow_fraction: Option<f64>,
    pub states_with_selected_destination: u64,
    pub states_with_selected_destination_overflow: u64,
    pub states_with_selected_destination_overflow_fraction: Option<f64>,
    pub groups_with_candidate_destinations: u64,
    pub groups_with_candidate_destination_overflow: u64,
    pub groups_with_candidate_destination_overflow_fraction: Option<f64>,
    pub dense_estimates: DenseEstimateReport,
    pub canonical_public_state_bytes_total: u64,
    pub serialized_position_record_bytes_total: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RetentionReport {
    pub total: u64,
    pub retained: u64,
    pub overflow: u64,
    pub retained_fraction: Option<f64>,
    pub overflow_fraction: Option<f64>,
}

impl From<RetentionAccumulator> for RetentionReport {
    fn from(value: RetentionAccumulator) -> Self {
        Self {
            total: value.total,
            retained: value.retained,
            overflow: value.overflow,
            retained_fraction: ratio(value.retained, value.total),
            overflow_fraction: ratio(value.overflow, value.total),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundaryReport {
    pub total: u64,
    pub fully_retained: u64,
    pub overflow: u64,
    pub crossing: u64,
    pub fully_outside: u64,
    pub fully_retained_fraction: Option<f64>,
    pub overflow_fraction: Option<f64>,
    pub crossing_fraction: Option<f64>,
}

impl From<BoundaryAccumulator> for BoundaryReport {
    fn from(value: BoundaryAccumulator) -> Self {
        Self {
            total: value.retention.total,
            fully_retained: value.retention.retained,
            overflow: value.retention.overflow,
            crossing: value.crossing,
            fully_outside: value.fully_outside,
            fully_retained_fraction: ratio(value.retention.retained, value.retention.total),
            overflow_fraction: ratio(value.retention.overflow, value.retention.total),
            crossing_fraction: ratio(value.crossing, value.retention.total),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DenseEstimateReport {
    pub cell_slots: u64,
    pub bytes_at_one_u8_per_cell: u64,
    pub bytes_at_one_f32_per_cell: u64,
    pub bytes_at_eleven_u8_channels_per_cell: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StorageReport {
    pub occupied_cells_per_board: HistogramReport,
    pub frontier_cells_per_board: HistogramReport,
    pub sparse_tokens_per_board: HistogramReport,
    pub canonical_public_state_bytes: SizeSummary,
    pub serialized_position_record_bytes: SizeSummary,
    pub current_v2_2401_cell_backing_grid_estimate: DenseEstimateReport,
    pub historical_441_cell_diagnostic_estimate: DenseEstimateReport,
}

impl StorageReport {
    fn from_storage(storage: &StorageAccumulator, board_observations: u64) -> Self {
        Self {
            occupied_cells_per_board: HistogramReport::from(&storage.occupied_cells_per_board),
            frontier_cells_per_board: HistogramReport::from(&storage.frontier_cells_per_board),
            sparse_tokens_per_board: HistogramReport::from(&storage.sparse_tokens_per_board),
            canonical_public_state_bytes: SizeSummary::from_histogram(
                &storage.canonical_public_state_bytes,
            ),
            serialized_position_record_bytes: SizeSummary::from_histogram(
                &storage.serialized_position_record_bytes,
            ),
            current_v2_2401_cell_backing_grid_estimate: dense_estimate(
                GRID_SIZE as u64,
                board_observations,
            ),
            historical_441_cell_diagnostic_estimate: dense_estimate(441, board_observations),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SizeSummary {
    pub observations: u64,
    pub total_bytes: u64,
    pub mean_bytes: Option<f64>,
    pub maximum_bytes: Option<u16>,
}

impl SizeSummary {
    fn from_histogram(histogram: &HistogramAccumulator) -> Self {
        let observations = histogram.observations();
        let total_bytes = histogram.sum();
        Self {
            observations,
            total_bytes,
            mean_bytes: (observations > 0).then_some(total_bytes as f64 / observations as f64),
            maximum_bytes: histogram.maximum(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutlierSummary {
    pub radius: u8,
    pub cap: usize,
    pub total: u64,
    pub retained: usize,
    pub truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdversarialCaseReport {
    pub name: String,
    pub legal_placed_tile_count: usize,
    pub occupied_fixed_origin_radius: u8,
    pub occupied_recentered_radius: u8,
    pub frontier_fixed_origin_radius: u8,
    pub frontier_recentered_radius: u8,
    pub recenter_q: i16,
    pub recenter_r: i16,
    pub radius_6_occupied_overflow: u64,
    pub radius_6_frontier_overflow: u64,
    pub overflows_radius_6: bool,
}

#[derive(Debug, Clone)]
struct CellView {
    coord: HexCoord,
    terrain_a: Terrain,
    terrain_b: Option<Terrain>,
    rotation: Rotation,
    allowed_wildlife_bits: u8,
    wildlife: Option<Wildlife>,
}

impl CellView {
    fn terrain_on_edge(&self, edge: usize) -> Terrain {
        let Some(terrain_b) = self.terrain_b else {
            return self.terrain_a;
        };
        let offset = (edge + 6 - usize::from(self.rotation.get())) % 6;
        if offset < 3 {
            self.terrain_a
        } else {
            terrain_b
        }
    }

    fn contains_terrain(&self, terrain: Terrain) -> bool {
        self.terrain_a == terrain || self.terrain_b == Some(terrain)
    }
}

#[derive(Debug, Clone)]
struct BoardView {
    cells: Vec<CellView>,
}

#[derive(Debug, Clone, Copy)]
struct BoardCohortSample {
    fixed_occupied_radius: u8,
    recentered_occupied_radius: u8,
    fixed_frontier_radius: u8,
    recentered_frontier_radius: u8,
    fixed_selected_destination_radius: Option<u8>,
    recentered_selected_destination_radius: Option<u8>,
    fixed_overflow: [bool; STATE_FOOTPRINT_RADII.len()],
    recentered_overflow: [bool; STATE_FOOTPRINT_RADII.len()],
}

#[derive(Debug, Clone)]
struct OutlierContext {
    source_kind: &'static str,
    dataset_id: Option<String>,
    seed_or_game_index: u64,
    turn: u16,
    decision_group_id: Option<u64>,
    current_player: u8,
    focal_relative_seat: u8,
    absolute_seat: u8,
    public_state_hash: String,
}

#[derive(Debug)]
struct GeneratedGameCensus {
    raw_seed: u64,
    accumulator: CorpusAccumulator,
    runtime: CorpusRuntime,
}

#[derive(Debug)]
struct ArmBuild<T> {
    corpus: CorpusScientific,
    identities: Vec<T>,
    paths: Vec<DatasetPathProvenance>,
    runtime: CorpusRuntime,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct GradedDecisionKey {
    raw_seed: u64,
    completed_turns: u16,
    group_id: u64,
    public_state_hash: [u8; 32],
}

pub const fn centered_hex_capacity(radius: u8) -> u64 {
    let radius = radius as u64;
    1 + 3 * radius * (radius + 1)
}

pub fn best_integer_center(coordinates: &[HexCoord]) -> HexCoord {
    if coordinates.is_empty() {
        return HexCoord::ORIGIN;
    }
    let mut min_q = i16::MAX;
    let mut max_q = i16::MIN;
    let mut min_r = i16::MAX;
    let mut max_r = i16::MIN;
    let mut min_s = i16::MAX;
    let mut max_s = i16::MIN;
    for coord in coordinates {
        let q = i16::from(coord.q);
        let r = i16::from(coord.r);
        let s = -q - r;
        min_q = min_q.min(q);
        max_q = max_q.max(q);
        min_r = min_r.min(r);
        max_r = max_r.max(r);
        min_s = min_s.min(s);
        max_s = max_s.max(s);
    }
    let lower_bound = [max_q - min_q, max_r - min_r, max_s - min_s]
        .into_iter()
        .max()
        .map_or(0, |span| (span + 1) / 2);
    for radius in lower_bound..=48 {
        let q_low = max_q - radius;
        let q_high = min_q + radius;
        let r_low = max_r - radius;
        let r_high = min_r + radius;
        let s_low = max_s - radius;
        let s_high = min_s + radius;
        if q_low > q_high
            || r_low > r_high
            || s_low > s_high
            || q_low + r_low + s_low > 0
            || q_high + r_high + s_high < 0
        {
            continue;
        }
        for q in q_low..=q_high {
            let minimum_r = r_low.max(-s_high - q);
            let maximum_r = r_high.min(-s_low - q);
            if minimum_r <= maximum_r {
                return HexCoord::new(q as i8, minimum_r as i8);
            }
        }
    }
    unreachable!("supported board coordinates always have an integer minimax center")
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
}

fn merge_u64_map(target: &mut BTreeMap<String, u64>, source: &BTreeMap<String, u64>) {
    for (key, value) in source {
        *target.entry(key.clone()).or_default() += value;
    }
}

fn merge_cohort_map(
    target: &mut BTreeMap<String, BoardCohortAccumulator>,
    source: &BTreeMap<String, BoardCohortAccumulator>,
) {
    for (key, value) in source {
        target.entry(key.clone()).or_default().merge_from(value);
    }
}

fn build_radius_table(
    set: &RadiusSetAccumulator,
    accumulator: &CorpusAccumulator,
) -> Vec<RadiusTableRow> {
    STATE_FOOTPRINT_RADII
        .into_iter()
        .zip(&set.rows)
        .map(|(radius, row)| {
            let capacity = centered_hex_capacity(radius);
            RadiusTableRow {
                radius,
                exact_centered_hex_capacity: capacity,
                occupied_cells: row.occupied_cells.into(),
                frontier_cells: row.frontier_cells.into(),
                selected_action_destinations: row.selected_action_destinations.into(),
                complete_candidate_destinations: row.complete_candidate_destinations.into(),
                wildlife_firings: row.wildlife_firings.into(),
                terrain_edge_firings: row.terrain_edge_firings.into(),
                allowed_wildlife_firings: row.allowed_wildlife_firings.into(),
                habitat_components: row.habitat_components.into(),
                wildlife_adjacencies: row.wildlife_adjacencies.into(),
                sparse_occupied_plus_frontier_tokens: row
                    .sparse_occupied_plus_frontier_tokens
                    .into(),
                board_observations: row.board_observations,
                boards_with_any_overflow: row.boards_with_any_overflow,
                boards_with_any_overflow_fraction: ratio(
                    row.boards_with_any_overflow,
                    row.board_observations,
                ),
                states_with_selected_destination: row.states_with_selected_destination,
                states_with_selected_destination_overflow: row
                    .states_with_selected_destination_overflow,
                states_with_selected_destination_overflow_fraction: ratio(
                    row.states_with_selected_destination_overflow,
                    row.states_with_selected_destination,
                ),
                groups_with_candidate_destinations: row.groups_with_candidate_destinations,
                groups_with_candidate_destination_overflow: row
                    .groups_with_candidate_destination_overflow,
                groups_with_candidate_destination_overflow_fraction: ratio(
                    row.groups_with_candidate_destination_overflow,
                    row.groups_with_candidate_destinations,
                ),
                dense_estimates: dense_estimate(capacity, accumulator.board_observation_count),
                canonical_public_state_bytes_total: accumulator
                    .storage
                    .canonical_public_state_bytes
                    .sum(),
                serialized_position_record_bytes_total: accumulator
                    .storage
                    .serialized_position_record_bytes
                    .sum(),
            }
        })
        .collect()
}

fn dense_estimate(cells_per_board: u64, board_observations: u64) -> DenseEstimateReport {
    let cell_slots = cells_per_board * board_observations;
    DenseEstimateReport {
        cell_slots,
        bytes_at_one_u8_per_cell: cell_slots,
        bytes_at_one_f32_per_cell: cell_slots * 4,
        bytes_at_eleven_u8_channels_per_cell: cell_slots * DENSE_CHANNEL_ESTIMATE,
    }
}

impl BoardView {
    fn from_board(board: &Board) -> Self {
        let mut cells = board
            .placed_tiles()
            .map(|(coord, placed)| CellView {
                coord,
                terrain_a: placed.tile.terrain_a,
                terrain_b: placed.tile.terrain_b,
                rotation: placed.rotation,
                allowed_wildlife_bits: placed.tile.wildlife.bits(),
                wildlife: placed.wildlife,
            })
            .collect::<Vec<_>>();
        cells.sort_by_key(|cell| cell.coord);
        Self { cells }
    }

    fn from_position(
        record: &PositionRecord,
        relative_seat: usize,
    ) -> Result<Self, StateFootprintError> {
        if relative_seat >= usize::from(record.player_count)
            || relative_seat >= BOARD_SLOTS
            || record.player_count != 4
            || record.total_turns != 80
        {
            return Err(StateFootprintError::Invariant(format!(
                "position record {} turn {} has invalid board metadata",
                record.game_index, record.turn
            )));
        }
        let count = usize::from(record.board_counts[relative_seat]);
        if count > record.board_entities[relative_seat].len() {
            return Err(StateFootprintError::Invariant(format!(
                "position record {} turn {} board {} exceeds fixed tile capacity",
                record.game_index, record.turn, relative_seat
            )));
        }
        let mut cells = Vec::with_capacity(count);
        let mut seen = BTreeSet::new();
        for entity in &record.board_entities[relative_seat][..count] {
            let coord = HexCoord::new(entity[0] as i8, entity[1] as i8);
            if coord.to_index().is_none() || !seen.insert(coord) {
                return Err(StateFootprintError::Invariant(format!(
                    "position record {} turn {} board {} contains an invalid or duplicate coordinate {:?}",
                    record.game_index, record.turn, relative_seat, coord
                )));
            }
            let terrain_a = decode_terrain(entity[2])?;
            let terrain_b = if entity[3] == NONE {
                None
            } else {
                Some(decode_terrain(entity[3])?)
            };
            let rotation = Rotation::new(entity[4]).ok_or_else(|| {
                StateFootprintError::Invariant(format!(
                    "position record {} turn {} contains rotation {}",
                    record.game_index, record.turn, entity[4]
                ))
            })?;
            if entity[5] & !0b1_1111 != 0 {
                return Err(StateFootprintError::Invariant(format!(
                    "position record {} turn {} contains invalid wildlife mask {}",
                    record.game_index, record.turn, entity[5]
                )));
            }
            let wildlife = if entity[6] == NONE {
                None
            } else {
                Some(decode_wildlife(entity[6])?)
            };
            cells.push(CellView {
                coord,
                terrain_a,
                terrain_b,
                rotation,
                allowed_wildlife_bits: entity[5],
                wildlife,
            });
        }
        cells.sort_by_key(|cell| cell.coord);
        Ok(Self { cells })
    }

    fn coordinates(&self) -> Vec<HexCoord> {
        self.cells.iter().map(|cell| cell.coord).collect()
    }

    fn frontier(&self) -> Vec<HexCoord> {
        let occupied = self.coordinates().into_iter().collect::<BTreeSet<_>>();
        let mut frontier = BTreeSet::new();
        for coord in &occupied {
            for neighbor in coord.neighbors() {
                if neighbor.to_index().is_some() && !occupied.contains(&neighbor) {
                    frontier.insert(neighbor);
                }
            }
        }
        frontier.into_iter().collect()
    }

    fn habitat_components(&self) -> Vec<Vec<HexCoord>> {
        let by_coord = self
            .cells
            .iter()
            .enumerate()
            .map(|(index, cell)| (cell.coord, index))
            .collect::<BTreeMap<_, _>>();
        let mut components = Vec::new();
        for terrain in Terrain::ALL {
            let mut seen = BTreeSet::new();
            for start in &self.cells {
                if seen.contains(&start.coord) || !start.contains_terrain(terrain) {
                    continue;
                }
                let mut stack = vec![start.coord];
                seen.insert(start.coord);
                let mut component = Vec::new();
                while let Some(coord) = stack.pop() {
                    component.push(coord);
                    let cell = &self.cells[by_coord[&coord]];
                    for edge in 0..6 {
                        if cell.terrain_on_edge(edge) != terrain {
                            continue;
                        }
                        let neighbor_coord = coord.neighbor(edge);
                        let Some(&neighbor_index) = by_coord.get(&neighbor_coord) else {
                            continue;
                        };
                        let neighbor = &self.cells[neighbor_index];
                        if neighbor.terrain_on_edge((edge + 3) % 6) == terrain
                            && seen.insert(neighbor_coord)
                        {
                            stack.push(neighbor_coord);
                        }
                    }
                }
                component.sort();
                components.push(component);
            }
        }
        components
    }

    fn wildlife_adjacencies(&self) -> Vec<Vec<HexCoord>> {
        let wildlife = self
            .cells
            .iter()
            .filter(|cell| cell.wildlife.is_some())
            .map(|cell| cell.coord)
            .collect::<BTreeSet<_>>();
        let mut pairs = Vec::new();
        for coord in &wildlife {
            for neighbor in coord.neighbors() {
                if *coord < neighbor && wildlife.contains(&neighbor) {
                    pairs.push(vec![*coord, neighbor]);
                }
            }
        }
        pairs
    }
}

fn decode_terrain(value: u8) -> Result<Terrain, StateFootprintError> {
    Terrain::ALL
        .get(usize::from(value))
        .copied()
        .ok_or_else(|| {
            StateFootprintError::Invariant(format!("invalid serialized terrain code {value}"))
        })
}

fn decode_wildlife(value: u8) -> Result<Wildlife, StateFootprintError> {
    Wildlife::ALL
        .get(usize::from(value))
        .copied()
        .ok_or_else(|| {
            StateFootprintError::Invariant(format!("invalid serialized wildlife code {value}"))
        })
}

fn maximum_radius(coordinates: &[HexCoord], center: HexCoord) -> u8 {
    coordinates
        .iter()
        .map(|coord| coord.distance(center))
        .max()
        .unwrap_or(0)
}

fn translated_coordinate(coord: HexCoord, center: HexCoord) -> HexCoord {
    HexCoord::new(coord.q - center.q, coord.r - center.r)
}

fn coordinate_bounds(coordinates: &[HexCoord]) -> CoordinateBounds {
    if coordinates.is_empty() {
        return CoordinateBounds::default();
    }
    let mut bounds = CoordinateBounds {
        min_q: i16::MAX,
        max_q: i16::MIN,
        min_r: i16::MAX,
        max_r: i16::MIN,
        min_s: i16::MAX,
        max_s: i16::MIN,
        max_absolute_q: 0,
        max_absolute_r: 0,
        max_absolute_s: 0,
    };
    for coord in coordinates {
        let q = i16::from(coord.q);
        let r = i16::from(coord.r);
        let s = -q - r;
        bounds.min_q = bounds.min_q.min(q);
        bounds.max_q = bounds.max_q.max(q);
        bounds.min_r = bounds.min_r.min(r);
        bounds.max_r = bounds.max_r.max(r);
        bounds.min_s = bounds.min_s.min(s);
        bounds.max_s = bounds.max_s.max(s);
        bounds.max_absolute_q = bounds.max_absolute_q.max(q.abs());
        bounds.max_absolute_r = bounds.max_absolute_r.max(r.abs());
        bounds.max_absolute_s = bounds.max_absolute_s.max(s.abs());
    }
    bounds
}

fn overflow_coordinates(
    occupied: &[HexCoord],
    frontier: &[HexCoord],
    selected: &[HexCoord],
    candidates: &[HexCoord],
    center: HexCoord,
) -> OverflowCoordinates {
    let outside = |coord: &&HexCoord| coord.distance(center) > STATE_FOOTPRINT_OUTLIER_RADIUS;
    OverflowCoordinates {
        occupied: occupied.iter().filter(outside).copied().collect(),
        frontier: frontier.iter().filter(outside).copied().collect(),
        selected_destinations: selected.iter().filter(outside).copied().collect(),
        complete_candidate_destinations: candidates.iter().filter(outside).copied().collect(),
    }
}

fn observe_board(
    accumulator: &mut CorpusAccumulator,
    board: &BoardView,
    selected_destinations: &[HexCoord],
    complete_candidate_destinations: &[HexCoord],
    context: &OutlierContext,
) -> Result<BoardCohortSample, StateFootprintError> {
    let occupied = board.coordinates();
    if occupied.is_empty() {
        return Err(StateFootprintError::Invariant(
            "census encountered an empty Cascadia board".to_owned(),
        ));
    }
    let frontier = board.frontier();
    let wildlife = board
        .cells
        .iter()
        .filter(|cell| cell.wildlife.is_some())
        .map(|cell| cell.coord)
        .collect::<Vec<_>>();
    let components = board.habitat_components();
    let wildlife_adjacencies = board.wildlife_adjacencies();
    let recentered = best_integer_center(&occupied);

    let fixed_radii = SupportRadii {
        occupied: maximum_radius(&occupied, HexCoord::ORIGIN),
        frontier: maximum_radius(&frontier, HexCoord::ORIGIN),
        selected_destination: (!selected_destinations.is_empty())
            .then(|| maximum_radius(selected_destinations, HexCoord::ORIGIN)),
        complete_candidate_destination: (!complete_candidate_destinations.is_empty())
            .then(|| maximum_radius(complete_candidate_destinations, HexCoord::ORIGIN)),
    };
    let recentered_radii = SupportRadii {
        occupied: maximum_radius(&occupied, recentered),
        frontier: maximum_radius(&frontier, recentered),
        selected_destination: (!selected_destinations.is_empty())
            .then(|| maximum_radius(selected_destinations, recentered)),
        complete_candidate_destination: (!complete_candidate_destinations.is_empty())
            .then(|| maximum_radius(complete_candidate_destinations, recentered)),
    };

    accumulator.support.occupied_radius.observe(
        usize::from(fixed_radii.occupied),
        usize::from(recentered_radii.occupied),
    )?;
    accumulator.support.frontier_radius.observe(
        usize::from(fixed_radii.frontier),
        usize::from(recentered_radii.frontier),
    )?;
    for coord in selected_destinations {
        accumulator
            .support
            .selected_action_destination_radius
            .observe(
                usize::from(coord.distance(HexCoord::ORIGIN)),
                usize::from(coord.distance(recentered)),
            )?;
    }
    for coord in complete_candidate_destinations {
        accumulator
            .support
            .complete_candidate_destination_radius
            .observe(
                usize::from(coord.distance(HexCoord::ORIGIN)),
                usize::from(coord.distance(recentered)),
            )?;
    }

    accumulator
        .storage
        .occupied_cells_per_board
        .observe(occupied.len())?;
    accumulator
        .storage
        .frontier_cells_per_board
        .observe(frontier.len())?;
    accumulator
        .storage
        .sparse_tokens_per_board
        .observe(occupied.len() + frontier.len())?;

    let mut fixed_overflow = [false; STATE_FOOTPRINT_RADII.len()];
    let mut recentered_overflow = [false; STATE_FOOTPRINT_RADII.len()];
    for (index, radius) in STATE_FOOTPRINT_RADII.into_iter().enumerate() {
        fixed_overflow[index] = observe_radius_row(
            &mut accumulator.support.fixed_origin.rows[index],
            board,
            &occupied,
            &frontier,
            selected_destinations,
            complete_candidate_destinations,
            &wildlife,
            &components,
            &wildlife_adjacencies,
            HexCoord::ORIGIN,
            radius,
        );
        recentered_overflow[index] = observe_radius_row(
            &mut accumulator.support.best_integer_recentered.rows[index],
            board,
            &occupied,
            &frontier,
            selected_destinations,
            complete_candidate_destinations,
            &wildlife,
            &components,
            &wildlife_adjacencies,
            recentered,
            radius,
        );
    }

    if support_overflows(fixed_radii, STATE_FOOTPRINT_OUTLIER_RADIUS)
        || support_overflows(recentered_radii, STATE_FOOTPRINT_OUTLIER_RADIUS)
    {
        let mut all_coordinates = occupied.clone();
        all_coordinates.extend(&frontier);
        all_coordinates.extend(selected_destinations);
        all_coordinates.extend(complete_candidate_destinations);
        all_coordinates.sort();
        all_coordinates.dedup();
        let translated = all_coordinates
            .iter()
            .map(|coord| translated_coordinate(*coord, recentered))
            .collect::<Vec<_>>();
        accumulator.outliers.observe(OverflowOutlier {
            source_kind: context.source_kind.to_owned(),
            dataset_id: context.dataset_id.clone(),
            seed_or_game_index: context.seed_or_game_index,
            turn: context.turn,
            decision_group_id: context.decision_group_id,
            current_player: context.current_player,
            focal_relative_seat: context.focal_relative_seat,
            absolute_seat: context.absolute_seat,
            public_state_hash: context.public_state_hash.clone(),
            recenter_q: i16::from(recentered.q),
            recenter_r: i16::from(recentered.r),
            original_coordinate_bounds: coordinate_bounds(&all_coordinates),
            recentered_coordinate_bounds: coordinate_bounds(&translated),
            fixed_origin_radii: fixed_radii,
            recentered_radii,
            fixed_origin_overflow_at_radius_6: overflow_coordinates(
                &occupied,
                &frontier,
                selected_destinations,
                complete_candidate_destinations,
                HexCoord::ORIGIN,
            ),
            recentered_overflow_at_radius_6: overflow_coordinates(
                &occupied,
                &frontier,
                selected_destinations,
                complete_candidate_destinations,
                recentered,
            ),
        });
    }

    accumulator.board_observation_count += 1;
    Ok(BoardCohortSample {
        fixed_occupied_radius: fixed_radii.occupied,
        recentered_occupied_radius: recentered_radii.occupied,
        fixed_frontier_radius: fixed_radii.frontier,
        recentered_frontier_radius: recentered_radii.frontier,
        fixed_selected_destination_radius: fixed_radii.selected_destination,
        recentered_selected_destination_radius: recentered_radii.selected_destination,
        fixed_overflow,
        recentered_overflow,
    })
}

#[allow(clippy::too_many_arguments)]
fn observe_radius_row(
    row: &mut RadiusAccumulator,
    board: &BoardView,
    occupied: &[HexCoord],
    frontier: &[HexCoord],
    selected_destinations: &[HexCoord],
    complete_candidate_destinations: &[HexCoord],
    wildlife: &[HexCoord],
    components: &[Vec<HexCoord>],
    wildlife_adjacencies: &[Vec<HexCoord>],
    center: HexCoord,
    radius: u8,
) -> bool {
    row.board_observations += 1;
    row.occupied_cells
        .observe_coordinates(occupied, center, radius);
    row.frontier_cells
        .observe_coordinates(frontier, center, radius);
    row.selected_action_destinations
        .observe_coordinates(selected_destinations, center, radius);
    row.complete_candidate_destinations.observe_coordinates(
        complete_candidate_destinations,
        center,
        radius,
    );
    row.wildlife_firings
        .observe_coordinates(wildlife, center, radius);

    let retained_cells = board
        .cells
        .iter()
        .filter(|cell| cell.coord.distance(center) <= radius)
        .collect::<Vec<_>>();
    row.terrain_edge_firings.observe_flags(
        retained_cells.len() as u64 * 6,
        (board.cells.len() - retained_cells.len()) as u64 * 6,
    );
    let retained_allowed = retained_cells
        .iter()
        .map(|cell| u64::from(cell.allowed_wildlife_bits.count_ones()))
        .sum::<u64>();
    let total_allowed = board
        .cells
        .iter()
        .map(|cell| u64::from(cell.allowed_wildlife_bits.count_ones()))
        .sum::<u64>();
    row.allowed_wildlife_firings
        .observe_flags(retained_allowed, total_allowed - retained_allowed);

    row.habitat_components
        .observe_sets(components, center, radius);
    row.wildlife_adjacencies
        .observe_sets(wildlife_adjacencies, center, radius);
    row.sparse_occupied_plus_frontier_tokens.observe_flags(
        row_count_inside(occupied, center, radius) + row_count_inside(frontier, center, radius),
        row_count_outside(occupied, center, radius) + row_count_outside(frontier, center, radius),
    );

    if !selected_destinations.is_empty() {
        row.states_with_selected_destination += 1;
        if selected_destinations
            .iter()
            .any(|coord| coord.distance(center) > radius)
        {
            row.states_with_selected_destination_overflow += 1;
        }
    }
    if !complete_candidate_destinations.is_empty() {
        row.groups_with_candidate_destinations += 1;
        if complete_candidate_destinations
            .iter()
            .any(|coord| coord.distance(center) > radius)
        {
            row.groups_with_candidate_destination_overflow += 1;
        }
    }

    let any_overflow = occupied
        .iter()
        .chain(frontier)
        .chain(selected_destinations)
        .chain(complete_candidate_destinations)
        .any(|coord| coord.distance(center) > radius);
    row.boards_with_any_overflow += u64::from(any_overflow);
    any_overflow
}

fn row_count_inside(coordinates: &[HexCoord], center: HexCoord, radius: u8) -> u64 {
    coordinates
        .iter()
        .filter(|coord| coord.distance(center) <= radius)
        .count() as u64
}

fn row_count_outside(coordinates: &[HexCoord], center: HexCoord, radius: u8) -> u64 {
    coordinates.len() as u64 - row_count_inside(coordinates, center, radius)
}

fn support_overflows(radii: SupportRadii, radius: u8) -> bool {
    radii.occupied > radius
        || radii.frontier > radius
        || radii
            .selected_destination
            .is_some_and(|value| value > radius)
        || radii
            .complete_candidate_destination
            .is_some_and(|value| value > radius)
}

fn observe_state_cohort(
    accumulator: &mut CorpusAccumulator,
    personal_turn: u8,
    current_player: u8,
) {
    *accumulator
        .cohorts
        .states
        .phase
        .entry(phase_label(personal_turn).to_owned())
        .or_default() += 1;
    *accumulator
        .cohorts
        .states
        .current_player
        .entry(current_player.to_string())
        .or_default() += 1;
}

fn observe_board_cohorts(
    accumulator: &mut CorpusAccumulator,
    sample: &BoardCohortSample,
    personal_turn: u8,
    absolute_seat: u8,
    relative_seat: u8,
) -> Result<(), StateFootprintError> {
    accumulator
        .cohorts
        .boards
        .phase
        .entry(phase_label(personal_turn).to_owned())
        .or_default()
        .observe(sample)?;
    accumulator
        .cohorts
        .boards
        .absolute_seat
        .entry(absolute_seat.to_string())
        .or_default()
        .observe(sample)?;
    accumulator
        .cohorts
        .boards
        .focal_relative_seat
        .entry(relative_seat.to_string())
        .or_default()
        .observe(sample)?;
    let fixed_width = if sample.fixed_occupied_radius > STATE_FOOTPRINT_OUTLIER_RADIUS {
        "wide_gt_radius_6"
    } else {
        "compact_le_radius_6"
    };
    accumulator
        .cohorts
        .boards
        .fixed_origin_width
        .entry(fixed_width.to_owned())
        .or_default()
        .observe(sample)?;
    let recentered_width = if sample.recentered_occupied_radius > STATE_FOOTPRINT_OUTLIER_RADIUS {
        "wide_gt_radius_6"
    } else {
        "compact_le_radius_6"
    };
    accumulator
        .cohorts
        .boards
        .recentered_width
        .entry(recentered_width.to_owned())
        .or_default()
        .observe(sample)?;
    Ok(())
}

fn observe_final_score_cohort(
    accumulator: &mut CorpusAccumulator,
    sample: &BoardCohortSample,
    final_score: u16,
) -> Result<(), StateFootprintError> {
    let band = match final_score {
        0..=89 => "under_90",
        90..=99 => "90_to_99",
        _ => "100_plus",
    };
    accumulator
        .cohorts
        .boards
        .final_score_band
        .entry(band.to_owned())
        .or_default()
        .observe(sample)?;
    accumulator
        .cohorts
        .boards
        .exact_final_score
        .entry(final_score.to_string())
        .or_default()
        .observe(sample)
}

fn phase_label(personal_turn: u8) -> &'static str {
    match personal_turn {
        1..=5 => "opening",
        6..=10 => "early",
        11..=15 => "middle",
        _ => "late",
    }
}

fn collect_generated_arm(
    config: &StateFootprintConfig,
) -> Result<Option<(CorpusScientific, GeneratedOrigin, CorpusRuntime)>, StateFootprintError> {
    if config.games == 0 {
        return Ok(None);
    }
    let wall_started = Instant::now();
    let mut games = (0..config.games)
        .into_par_iter()
        .map(|offset| {
            let raw_seed = config
                .first_seed
                .checked_add(offset as u64)
                .ok_or_else(|| {
                    StateFootprintError::InvalidConfig(
                        "generated seed range overflows u64".to_owned(),
                    )
                })?;
            collect_generated_game(raw_seed, config.strategy, config.outlier_cap)
        })
        .collect::<Result<Vec<_>, _>>()?;
    games.sort_by_key(|game| game.raw_seed);

    let mut accumulator = CorpusAccumulator::new(config.outlier_cap);
    let mut runtime = CorpusRuntime {
        parallel_wall_seconds: wall_started.elapsed().as_secs_f64(),
        ..CorpusRuntime::default()
    };
    for game in games {
        accumulator.merge_from(game.accumulator)?;
        runtime.merge_from(&game.runtime);
    }
    accumulator.validate()?;
    Ok(Some((
        CorpusScientific::new(accumulator),
        GeneratedOrigin {
            first_seed: config.first_seed,
            games: config.games,
            strategy_id: config.strategy.id().to_owned(),
        },
        runtime,
    )))
}

fn collect_generated_game(
    raw_seed: u64,
    strategy: StrategyKind,
    outlier_cap: usize,
) -> Result<GeneratedGameCensus, StateFootprintError> {
    let game_config = GameConfig::research_aaaaa(4)?;
    let match_config = MatchConfig::symmetric(game_config, GameSeed::from_u64(raw_seed), strategy);
    let game_started = Instant::now();
    let mut extraction_seconds = 0.0;
    let mut extraction_error = None;
    let mut accumulator = CorpusAccumulator::new(outlier_cap);
    let mut pending_final_cohorts = Vec::<(u8, BoardCohortSample)>::with_capacity(320);
    let result = play_match_observed(&match_config, |state, action| {
        let extraction_started = Instant::now();
        if extraction_error.is_none()
            && let Err(error) = observe_generated_state(
                &mut accumulator,
                &mut pending_final_cohorts,
                raw_seed,
                state,
                action,
            )
        {
            extraction_error = Some(error);
        }
        extraction_seconds += extraction_started.elapsed().as_secs_f64();
    })?;
    let source_wall_seconds = game_started.elapsed().as_secs_f64();
    if let Some(error) = extraction_error {
        return Err(error);
    }

    for (absolute_seat, sample) in pending_final_cohorts {
        observe_final_score_cohort(
            &mut accumulator,
            &sample,
            result.scores[usize::from(absolute_seat)].base_total,
        )?;
    }
    if accumulator.state_count != u64::from(result.turns)
        || accumulator.board_observation_count != u64::from(result.turns) * 4
    {
        return Err(StateFootprintError::Invariant(format!(
            "generated seed {raw_seed} produced {} states and {} board observations for {} turns",
            accumulator.state_count, accumulator.board_observation_count, result.turns
        )));
    }
    accumulator.validate()?;
    Ok(GeneratedGameCensus {
        raw_seed,
        accumulator,
        runtime: CorpusRuntime {
            summed_source_wall_seconds: source_wall_seconds,
            extraction_build_seconds: extraction_seconds,
            simulation_excluding_extraction_seconds: (source_wall_seconds - extraction_seconds)
                .max(0.0),
            ..CorpusRuntime::default()
        },
    })
}

fn observe_generated_state(
    accumulator: &mut CorpusAccumulator,
    pending_final_cohorts: &mut Vec<(u8, BoardCohortSample)>,
    raw_seed: u64,
    state: &cascadia_game::GameState,
    action: &TurnAction,
) -> Result<(), StateFootprintError> {
    let current_player = state.current_player() as u8;
    let turn = state.completed_turns();
    let personal_turn = (turn / 4 + 1) as u8;
    let public = state.public_state();
    let public_bytes = public.canonical_bytes();
    let public_hash = public.canonical_hash().to_hex().to_string();

    accumulator.state_count += 1;
    accumulator.selected_destination_available_states += 1;
    accumulator
        .storage
        .canonical_public_state_bytes
        .observe(public_bytes.len())?;
    observe_state_cohort(accumulator, personal_turn, current_player);

    for relative_seat in 0..4u8 {
        let absolute_seat = (current_player + relative_seat) % 4;
        let selected = (relative_seat == 0)
            .then_some(action.tile.coord)
            .into_iter()
            .collect::<Vec<_>>();
        let context = OutlierContext {
            source_kind: "generated",
            dataset_id: None,
            seed_or_game_index: raw_seed,
            turn,
            decision_group_id: None,
            current_player,
            focal_relative_seat: relative_seat,
            absolute_seat,
            public_state_hash: public_hash.clone(),
        };
        let board = BoardView::from_board(&state.boards()[usize::from(absolute_seat)]);
        let sample = observe_board(accumulator, &board, &selected, &[], &context)?;
        observe_board_cohorts(
            accumulator,
            &sample,
            personal_turn,
            absolute_seat,
            relative_seat,
        )?;
        pending_final_cohorts.push((absolute_seat, sample));
    }
    Ok(())
}

fn scan_position_datasets(
    roots: &[PathBuf],
    outlier_cap: usize,
) -> Result<Option<ArmBuild<PositionDatasetIdentity>>, StateFootprintError> {
    if roots.is_empty() {
        return Ok(None);
    }
    let wall_started = Instant::now();
    let mut accumulator = CorpusAccumulator::new(outlier_cap);
    let mut identities = Vec::with_capacity(roots.len());
    let mut paths = Vec::with_capacity(roots.len());
    let mut runtime = CorpusRuntime::default();
    let mut seen_states = BTreeMap::<(String, u64, u8), [u8; 32]>::new();

    for root in roots {
        let manifest_path = root.join("dataset.json");
        let manifest: DatasetManifest =
            serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
        let manifest_blake3 = checksum_file(&manifest_path)?;

        let validation_started = Instant::now();
        validate_dataset(root, &manifest)?;
        runtime.validation_seconds += validation_started.elapsed().as_secs_f64();

        let identity = PositionDatasetIdentity {
            dataset_id: manifest.dataset_id.clone(),
            manifest_blake3: manifest_blake3.clone(),
            split: manifest.split.id().to_owned(),
            strategy_id: manifest.strategy.clone(),
            first_game_index: manifest.first_game_index,
            completed_games: manifest.completed_games,
            total_records: manifest.total_records,
            shards: manifest
                .shards
                .iter()
                .map(|shard| FileIdentity {
                    file: shard.file.clone(),
                    byte_count: shard.byte_count,
                    blake3: shard.blake3.clone(),
                })
                .collect(),
        };
        paths.push(DatasetPathProvenance {
            kind: "position_dataset".to_owned(),
            dataset_id: identity.dataset_id.clone(),
            root: fs::canonicalize(root)?.display().to_string(),
            manifest_blake3,
        });

        let source_started = Instant::now();
        let mut extraction_seconds = 0.0;
        let mut current_game = Vec::new();
        let mut previous_game_index = None;
        for shard in &manifest.shards {
            let reader = PositionShardReader::open(root, shard)?;
            for record in reader {
                let record = record?;
                if previous_game_index.is_some_and(|previous| record.game_index < previous) {
                    return Err(StateFootprintError::Invariant(format!(
                        "position dataset {} is not ordered by game index",
                        manifest.dataset_id
                    )));
                }
                if previous_game_index.is_some_and(|previous| record.game_index != previous) {
                    let extraction_started = Instant::now();
                    process_position_game(
                        &manifest,
                        &current_game,
                        &mut accumulator,
                        &mut seen_states,
                    )?;
                    extraction_seconds += extraction_started.elapsed().as_secs_f64();
                    current_game.clear();
                }
                previous_game_index = Some(record.game_index);
                current_game.push(record);
            }
        }
        if !current_game.is_empty() {
            let extraction_started = Instant::now();
            process_position_game(&manifest, &current_game, &mut accumulator, &mut seen_states)?;
            extraction_seconds += extraction_started.elapsed().as_secs_f64();
        }
        let source_seconds = source_started.elapsed().as_secs_f64();
        runtime.read_seconds += (source_seconds - extraction_seconds).max(0.0);
        runtime.extraction_build_seconds += extraction_seconds;
        identities.push(identity);
    }
    identities.sort();
    paths.sort_by(|left, right| {
        left.kind
            .cmp(&right.kind)
            .then_with(|| left.dataset_id.cmp(&right.dataset_id))
    });
    runtime.parallel_wall_seconds = wall_started.elapsed().as_secs_f64();
    accumulator.validate()?;
    Ok(Some(ArmBuild {
        corpus: CorpusScientific::new(accumulator),
        identities,
        paths,
        runtime,
    }))
}

fn process_position_game(
    manifest: &DatasetManifest,
    records: &[PositionRecord],
    accumulator: &mut CorpusAccumulator,
    seen_states: &mut BTreeMap<(String, u64, u8), [u8; 32]>,
) -> Result<(), StateFootprintError> {
    if records.is_empty() {
        return Ok(());
    }
    let game_index = records[0].game_index;
    for (index, record) in records.iter().enumerate() {
        if record.game_index != game_index
            || (index > 0 && record.turn <= records[index - 1].turn)
            || record.active_seat >= 4
        {
            return Err(StateFootprintError::Invariant(format!(
                "position dataset {} has malformed game {} ordering",
                manifest.dataset_id, game_index
            )));
        }
        let public_hash = position_public_hash(record);
        let key = (
            manifest.split.id().to_owned(),
            record.game_index,
            record.turn,
        );
        if let Some(existing) = seen_states.get(&key) {
            if existing != &public_hash {
                return Err(StateFootprintError::Invariant(format!(
                    "position datasets disagree on split {} game {} turn {}",
                    manifest.split.id(),
                    record.game_index,
                    record.turn
                )));
            }
            accumulator.duplicate_decision_groups_skipped += 1;
            continue;
        }
        seen_states.insert(key, public_hash);

        let selected_destination = if let Some(next) = records.get(index + 1) {
            infer_exact_selected_destination(record, next)?
        } else {
            None
        };
        observe_position_record(
            accumulator,
            record,
            selected_destination,
            &[],
            "position_dataset",
            Some(manifest.dataset_id.clone()),
            None,
            hex_bytes(&public_hash),
        )?;
        accumulator
            .storage
            .serialized_position_record_bytes
            .observe(RECORD_SIZE)?;
    }
    Ok(())
}

fn infer_exact_selected_destination(
    current: &PositionRecord,
    next: &PositionRecord,
) -> Result<Option<HexCoord>, StateFootprintError> {
    if next.game_index != current.game_index || next.turn != current.turn.saturating_add(1) {
        return Ok(None);
    }
    let player_count = usize::from(current.player_count);
    if player_count != 4
        || next.player_count != current.player_count
        || next.active_seat as usize != (usize::from(current.active_seat) + 1) % player_count
    {
        return Ok(None);
    }
    let acting_absolute = usize::from(current.active_seat);
    let mut inferred = None;
    for absolute in 0..player_count {
        let current_relative =
            (absolute + player_count - usize::from(current.active_seat)) % player_count;
        let next_relative =
            (absolute + player_count - usize::from(next.active_seat)) % player_count;
        let before = position_board_coordinates(current, current_relative)?;
        let after = position_board_coordinates(next, next_relative)?;
        if absolute == acting_absolute {
            if !before.is_subset(&after) || after.len() != before.len() + 1 {
                return Ok(None);
            }
            inferred = after.difference(&before).next().copied();
        } else if before != after {
            return Ok(None);
        }
    }
    Ok(inferred)
}

fn position_board_coordinates(
    record: &PositionRecord,
    relative_seat: usize,
) -> Result<BTreeSet<HexCoord>, StateFootprintError> {
    Ok(BoardView::from_position(record, relative_seat)?
        .cells
        .into_iter()
        .map(|cell| cell.coord)
        .collect())
}

fn position_public_hash(record: &PositionRecord) -> [u8; 32] {
    let mut public = record.clone();
    public.targets.fill(0);
    *blake3::hash(&public.to_bytes()).as_bytes()
}

fn scan_graded_datasets(
    roots: &[PathBuf],
    outlier_cap: usize,
) -> Result<Option<ArmBuild<GradedDatasetIdentity>>, StateFootprintError> {
    if roots.is_empty() {
        return Ok(None);
    }
    let wall_started = Instant::now();
    let mut accumulator = CorpusAccumulator::new(outlier_cap);
    let mut identities = Vec::with_capacity(roots.len());
    let mut paths = Vec::with_capacity(roots.len());
    let mut runtime = CorpusRuntime::default();
    let mut seen_groups = BTreeSet::<GradedDecisionKey>::new();
    let mut seen_turns = BTreeMap::<(u64, u16), [u8; 32]>::new();

    for root in roots {
        let manifest_path = root.join("dataset.json");
        let manifest: GradedOracleDatasetManifest =
            serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
        let manifest_blake3 = checksum_file(&manifest_path)?;

        let validation_started = Instant::now();
        validate_graded_oracle_dataset(root, &manifest)?;
        runtime.validation_seconds += validation_started.elapsed().as_secs_f64();

        let identity = GradedDatasetIdentity {
            dataset_id: manifest.dataset_id.clone(),
            manifest_blake3: manifest_blake3.clone(),
            split: manifest.split.id().to_owned(),
            completed_games: manifest.completed_games,
            total_groups: manifest.total_groups,
            total_candidate_rows: manifest.total_records,
            seeds: manifest.seeds.clone(),
            shards: manifest
                .shards
                .iter()
                .map(|shard| FileIdentity {
                    file: shard.file.clone(),
                    byte_count: shard.byte_count,
                    blake3: shard.blake3.clone(),
                })
                .collect(),
        };
        paths.push(DatasetPathProvenance {
            kind: "graded_oracle".to_owned(),
            dataset_id: identity.dataset_id.clone(),
            root: fs::canonicalize(root)?.display().to_string(),
            manifest_blake3,
        });

        let source_started = Instant::now();
        let mut extraction_seconds = 0.0;
        for shard in &manifest.shards {
            for group in cascadia_data::read_graded_oracle_shard(root, manifest.split, shard)? {
                let key = GradedDecisionKey {
                    raw_seed: group.raw_seed,
                    completed_turns: group.completed_turns,
                    group_id: group.group_id,
                    public_state_hash: group.public_state_hash,
                };
                if !seen_groups.insert(key) {
                    accumulator.duplicate_decision_groups_skipped += 1;
                    continue;
                }
                let turn_key = (group.raw_seed, group.completed_turns);
                if let Some(existing) = seen_turns.insert(turn_key, group.public_state_hash)
                    && existing != group.public_state_hash
                {
                    return Err(StateFootprintError::Invariant(format!(
                        "graded datasets disagree on seed {} turn {}",
                        group.raw_seed, group.completed_turns
                    )));
                }
                let extraction_started = Instant::now();
                observe_graded_group(&mut accumulator, &manifest.dataset_id, &group)?;
                extraction_seconds += extraction_started.elapsed().as_secs_f64();
            }
        }
        let source_seconds = source_started.elapsed().as_secs_f64();
        runtime.read_seconds += (source_seconds - extraction_seconds).max(0.0);
        runtime.extraction_build_seconds += extraction_seconds;
        identities.push(identity);
    }
    identities.sort();
    paths.sort_by(|left, right| {
        left.kind
            .cmp(&right.kind)
            .then_with(|| left.dataset_id.cmp(&right.dataset_id))
    });
    runtime.parallel_wall_seconds = wall_started.elapsed().as_secs_f64();
    accumulator.validate()?;
    Ok(Some(ArmBuild {
        corpus: CorpusScientific::new(accumulator),
        identities,
        paths,
        runtime,
    }))
}

fn observe_graded_group(
    accumulator: &mut CorpusAccumulator,
    dataset_id: &str,
    group: &GradedOracleGroup,
) -> Result<(), StateFootprintError> {
    let selected = group
        .candidates
        .get(usize::from(group.selected_index))
        .ok_or_else(|| {
            StateFootprintError::Invariant(format!(
                "graded group {} selected index is unavailable",
                group.group_id
            ))
        })?;
    let selected_destination = HexCoord::new(selected.action.tile_q, selected.action.tile_r);
    let candidate_destinations = group
        .candidates
        .iter()
        .map(|candidate| HexCoord::new(candidate.action.tile_q, candidate.action.tile_r))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    accumulator.complete_candidate_rows += group.candidates.len() as u64;
    accumulator.distinct_complete_candidate_destinations += candidate_destinations.len() as u64;
    observe_position_record(
        accumulator,
        &group.position,
        Some(selected_destination),
        &candidate_destinations,
        "graded_oracle",
        Some(dataset_id.to_owned()),
        Some(group.group_id),
        hex_bytes(&group.public_state_hash),
    )?;
    accumulator
        .storage
        .serialized_position_record_bytes
        .observe(RECORD_SIZE)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn observe_position_record(
    accumulator: &mut CorpusAccumulator,
    record: &PositionRecord,
    selected_destination: Option<HexCoord>,
    candidate_destinations: &[HexCoord],
    source_kind: &'static str,
    dataset_id: Option<String>,
    decision_group_id: Option<u64>,
    public_state_hash: String,
) -> Result<(), StateFootprintError> {
    if record.player_count != 4 || record.active_seat >= 4 || record.turn >= record.total_turns {
        return Err(StateFootprintError::Invariant(format!(
            "position record {} turn {} does not describe a four-player pre-move state",
            record.game_index, record.turn
        )));
    }
    let current_player = record.active_seat;
    let personal_turn = u16::from(record.turn) / 4 + 1;
    let personal_turn = personal_turn as u8;

    accumulator.state_count += 1;
    if selected_destination.is_some() {
        accumulator.selected_destination_available_states += 1;
    } else {
        accumulator.selected_destination_unavailable_states += 1;
    }
    observe_state_cohort(accumulator, personal_turn, current_player);

    for relative_seat in 0..4u8 {
        let absolute_seat = (current_player + relative_seat) % 4;
        let selected = (relative_seat == 0)
            .then_some(selected_destination)
            .flatten()
            .into_iter()
            .collect::<Vec<_>>();
        let candidates = if relative_seat == 0 {
            candidate_destinations
        } else {
            &[]
        };
        let context = OutlierContext {
            source_kind,
            dataset_id: dataset_id.clone(),
            seed_or_game_index: record.game_index,
            turn: u16::from(record.turn),
            decision_group_id,
            current_player,
            focal_relative_seat: relative_seat,
            absolute_seat,
            public_state_hash: public_state_hash.clone(),
        };
        let board = BoardView::from_position(record, usize::from(relative_seat))?;
        let sample = observe_board(accumulator, &board, &selected, candidates, &context)?;
        observe_board_cohorts(
            accumulator,
            &sample,
            personal_turn,
            absolute_seat,
            relative_seat,
        )?;
    }
    Ok(())
}

fn hex_bytes(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn run_state_footprint_census(
    config: &StateFootprintConfig,
    output: &Path,
) -> Result<StateFootprintReport, StateFootprintError> {
    config.validate()?;
    let total_started = Instant::now();

    let generated = collect_generated_arm(config)?;
    let position = scan_position_datasets(&config.position_dataset_roots, config.outlier_cap)?;
    let graded = scan_graded_datasets(&config.graded_dataset_roots, config.outlier_cap)?;
    let adversarial_cases = adversarial_cases()?;
    let invariants = geometry_invariants()?;

    let mut generated_origins = Vec::new();
    let mut position_identities = Vec::new();
    let mut graded_identities = Vec::new();
    let mut dataset_paths = Vec::new();
    let mut runtime = CensusRuntime::default();

    let generated_corpus = generated.map(|(corpus, origin, arm_runtime)| {
        generated_origins.push(origin);
        runtime.generated = arm_runtime;
        corpus
    });
    let position_corpus = position.map(|arm| {
        position_identities = arm.identities;
        dataset_paths.extend(arm.paths);
        runtime.position_datasets = arm.runtime;
        arm.corpus
    });
    let graded_corpus = graded.map(|arm| {
        graded_identities = arm.identities;
        dataset_paths.extend(arm.paths);
        runtime.graded_datasets = arm.runtime;
        arm.corpus
    });

    let mut scientific = ScientificPayload {
        schema_version: STATE_FOOTPRINT_SCHEMA_VERSION,
        experiment_id: STATE_FOOTPRINT_EXPERIMENT_ID.to_owned(),
        ruleset: ruleset_identity(),
        configuration: ScientificConfiguration {
            radii: STATE_FOOTPRINT_RADII.to_vec(),
            outlier_radius: STATE_FOOTPRINT_OUTLIER_RADIUS,
            outlier_cap: config.outlier_cap,
            generated_origins,
            position_datasets: position_identities,
            graded_datasets: graded_identities,
        },
        definitions: scientific_definitions(),
        invariants,
        generated: generated_corpus,
        position_datasets: position_corpus,
        graded_oracle: graded_corpus,
        adversarial_cases,
        completion: CompletionAssessment {
            classification: String::new(),
            complete: false,
            reasons: Vec::new(),
        },
    };
    scientific.configuration.generated_origins =
        normalize_generated_origins(&scientific.configuration.generated_origins)?;
    scientific.completion = assess_completion(&scientific);
    validate_scientific_payload(&scientific)?;

    runtime.total_wall_seconds = total_started.elapsed().as_secs_f64();
    let provenance = current_report_provenance(dataset_paths, Vec::new())?;
    let scientific_hash = scientific_hash(&scientific)?;
    Ok(StateFootprintReport {
        schema_version: STATE_FOOTPRINT_SCHEMA_VERSION,
        experiment_id: STATE_FOOTPRINT_EXPERIMENT_ID.to_owned(),
        created_unix_seconds: unix_seconds()?,
        output_path: output.display().to_string(),
        provenance,
        runtime,
        scientific_hash_algorithm: SCIENTIFIC_HASH_ALGORITHM.to_owned(),
        scientific_hash,
        scientific,
    })
}

pub fn write_state_footprint_report_atomic(
    output: &Path,
    report: &StateFootprintReport,
) -> Result<(), StateFootprintError> {
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let extension = output
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("json");
    let temporary = output.with_extension(format!("{extension}.tmp-{}", std::process::id()));
    let result = (|| {
        let mut writer = BufWriter::new(File::create(&temporary)?);
        serde_json::to_writer_pretty(&mut writer, report)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        fs::rename(&temporary, output)?;
        if let Some(parent) = output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            File::open(parent)?.sync_all()?;
        }
        Ok::<(), StateFootprintError>(())
    })();
    if result.is_err() {
        fs::remove_file(&temporary).ok();
    }
    result
}

pub fn merge_state_footprint_report_files(
    inputs: &[PathBuf],
    output: &Path,
) -> Result<StateFootprintReport, StateFootprintError> {
    if inputs.len() < 2 {
        return Err(StateFootprintError::InvalidConfig(
            "merge requires at least two input reports".to_owned(),
        ));
    }
    let merge_started = Instant::now();
    let mut loaded = Vec::with_capacity(inputs.len());
    for path in inputs {
        let report: StateFootprintReport =
            serde_json::from_reader(BufReader::new(File::open(path)?))?;
        validate_report(&report)?;
        loaded.push((
            fs::canonicalize(path)?.display().to_string(),
            checksum_file(path)?,
            report,
        ));
    }
    loaded.sort_by(|left, right| {
        left.2
            .scientific_hash
            .cmp(&right.2.scientific_hash)
            .then_with(|| left.0.cmp(&right.0))
    });
    let frozen_source_blake3 = loaded[0].2.provenance.source.v2_source_blake3.clone();
    let frozen_executable_blake3 = loaded[0].2.provenance.executable_blake3.clone();
    for (_, _, report) in &loaded[1..] {
        validate_merge_implementation_identity(
            &frozen_source_blake3,
            &frozen_executable_blake3,
            &report.provenance,
        )?;
    }
    let contract = loaded[0].2.scientific.clone();
    let first = &contract;
    let outlier_cap = contract.configuration.outlier_cap;
    for (_, _, report) in &loaded[1..] {
        let candidate = &report.scientific;
        if candidate.schema_version != first.schema_version
            || candidate.experiment_id != first.experiment_id
            || candidate.ruleset != first.ruleset
            || candidate.definitions != first.definitions
            || candidate.invariants != first.invariants
            || candidate.adversarial_cases != first.adversarial_cases
            || candidate.configuration.radii != first.configuration.radii
            || candidate.configuration.outlier_radius != first.configuration.outlier_radius
            || candidate.configuration.outlier_cap != outlier_cap
        {
            return Err(StateFootprintError::Invariant(
                "merge inputs have incompatible scientific contracts".to_owned(),
            ));
        }
    }

    let mut generated = None;
    let mut position = None;
    let mut graded = None;
    let mut origins = Vec::new();
    let mut position_identities = Vec::new();
    let mut graded_identities = Vec::new();
    let mut runtime = CensusRuntime::default();
    let mut dataset_paths = Vec::new();
    let mut merged_inputs = Vec::new();

    for (path, report_blake3, report) in loaded {
        merge_optional_corpus(&mut generated, report.scientific.generated, outlier_cap)?;
        merge_optional_corpus(
            &mut position,
            report.scientific.position_datasets,
            outlier_cap,
        )?;
        merge_optional_corpus(&mut graded, report.scientific.graded_oracle, outlier_cap)?;
        origins.extend(report.scientific.configuration.generated_origins);
        position_identities.extend(report.scientific.configuration.position_datasets);
        graded_identities.extend(report.scientific.configuration.graded_datasets);
        runtime.generated.merge_from(&report.runtime.generated);
        runtime
            .position_datasets
            .merge_from(&report.runtime.position_datasets);
        runtime
            .graded_datasets
            .merge_from(&report.runtime.graded_datasets);
        dataset_paths.extend(report.provenance.dataset_paths);
        merged_inputs.push(MergedInputProvenance {
            path,
            report_blake3,
            scientific_hash: report.scientific_hash,
        });
    }

    let origins = normalize_generated_origins(&origins)?;
    normalize_position_identities(&mut position_identities)?;
    normalize_graded_identities(&mut graded_identities)?;
    dataset_paths.sort_by(|left, right| {
        left.kind
            .cmp(&right.kind)
            .then_with(|| left.dataset_id.cmp(&right.dataset_id))
            .then_with(|| left.manifest_blake3.cmp(&right.manifest_blake3))
    });
    dataset_paths.dedup();
    merged_inputs.sort_by(|left, right| {
        left.scientific_hash
            .cmp(&right.scientific_hash)
            .then_with(|| left.path.cmp(&right.path))
    });

    let mut scientific = ScientificPayload {
        schema_version: first.schema_version,
        experiment_id: first.experiment_id.clone(),
        ruleset: first.ruleset.clone(),
        configuration: ScientificConfiguration {
            radii: first.configuration.radii.clone(),
            outlier_radius: first.configuration.outlier_radius,
            outlier_cap,
            generated_origins: origins,
            position_datasets: position_identities,
            graded_datasets: graded_identities,
        },
        definitions: first.definitions.clone(),
        invariants: first.invariants.clone(),
        generated: generated.map(CorpusScientific::new),
        position_datasets: position.map(CorpusScientific::new),
        graded_oracle: graded.map(CorpusScientific::new),
        adversarial_cases: first.adversarial_cases.clone(),
        completion: CompletionAssessment {
            classification: String::new(),
            complete: false,
            reasons: Vec::new(),
        },
    };
    scientific.completion = assess_completion(&scientific);
    validate_scientific_payload(&scientific)?;

    runtime.merge_wall_seconds = merge_started.elapsed().as_secs_f64();
    runtime.total_wall_seconds = runtime.generated.parallel_wall_seconds
        + runtime.position_datasets.parallel_wall_seconds
        + runtime.graded_datasets.parallel_wall_seconds
        + runtime.merge_wall_seconds;
    let provenance = current_report_provenance(dataset_paths, merged_inputs)?;
    validate_merge_implementation_identity(
        &frozen_source_blake3,
        &frozen_executable_blake3,
        &provenance,
    )?;
    let scientific_hash = scientific_hash(&scientific)?;
    Ok(StateFootprintReport {
        schema_version: STATE_FOOTPRINT_SCHEMA_VERSION,
        experiment_id: STATE_FOOTPRINT_EXPERIMENT_ID.to_owned(),
        created_unix_seconds: unix_seconds()?,
        output_path: output.display().to_string(),
        provenance,
        runtime,
        scientific_hash_algorithm: SCIENTIFIC_HASH_ALGORITHM.to_owned(),
        scientific_hash,
        scientific,
    })
}

fn validate_merge_implementation_identity(
    expected_source_blake3: &str,
    expected_executable_blake3: &str,
    candidate: &ReportProvenance,
) -> Result<(), StateFootprintError> {
    if candidate.source.v2_source_blake3 != expected_source_blake3 {
        return Err(StateFootprintError::Invariant(format!(
            "merge input v2_source_blake3 drift: expected {expected_source_blake3}, found {}",
            candidate.source.v2_source_blake3
        )));
    }
    if candidate.executable_blake3 != expected_executable_blake3 {
        return Err(StateFootprintError::Invariant(format!(
            "merge input executable_blake3 drift: expected {expected_executable_blake3}, found {}",
            candidate.executable_blake3
        )));
    }
    Ok(())
}

fn merge_optional_corpus(
    target: &mut Option<CorpusAccumulator>,
    source: Option<CorpusScientific>,
    outlier_cap: usize,
) -> Result<(), StateFootprintError> {
    let Some(source) = source else {
        return Ok(());
    };
    source.validate()?;
    if let Some(target) = target {
        target.merge_from(source.merge_accumulator)?;
    } else {
        let mut accumulator = CorpusAccumulator::new(outlier_cap);
        accumulator.merge_from(source.merge_accumulator)?;
        *target = Some(accumulator);
    }
    Ok(())
}

fn normalize_generated_origins(
    origins: &[GeneratedOrigin],
) -> Result<Vec<GeneratedOrigin>, StateFootprintError> {
    let mut origins = origins.to_vec();
    origins.sort();
    let mut normalized = Vec::<GeneratedOrigin>::new();
    for origin in origins {
        let end = origin
            .first_seed
            .checked_add(origin.games as u64)
            .ok_or_else(|| {
                StateFootprintError::Invariant("generated origin range overflows u64".to_owned())
            })?;
        if let Some(previous) = normalized.last_mut() {
            let previous_end = previous.first_seed + previous.games as u64;
            if origin.first_seed < previous_end {
                return Err(StateFootprintError::Invariant(format!(
                    "generated origins overlap at seed {}",
                    origin.first_seed
                )));
            }
            if origin.first_seed == previous_end && origin.strategy_id == previous.strategy_id {
                previous.games += origin.games;
                continue;
            }
        }
        if end < origin.first_seed {
            return Err(StateFootprintError::Invariant(
                "generated origin has an invalid range".to_owned(),
            ));
        }
        normalized.push(origin);
    }
    Ok(normalized)
}

fn normalize_position_identities(
    identities: &mut [PositionDatasetIdentity],
) -> Result<(), StateFootprintError> {
    identities.sort();
    if identities
        .windows(2)
        .any(|pair| pair[0].manifest_blake3 == pair[1].manifest_blake3)
    {
        return Err(StateFootprintError::Invariant(
            "position report merge contains a duplicate manifest".to_owned(),
        ));
    }
    let mut ranges = BTreeMap::<String, Vec<(u64, u64)>>::new();
    for identity in identities.iter() {
        let end = identity.first_game_index + identity.completed_games as u64;
        ranges
            .entry(identity.split.clone())
            .or_default()
            .push((identity.first_game_index, end));
    }
    for (split, ranges) in &mut ranges {
        ranges.sort();
        if ranges.windows(2).any(|pair| pair[1].0 < pair[0].1) {
            return Err(StateFootprintError::Invariant(format!(
                "position report merge overlaps {split} game-index ranges"
            )));
        }
    }
    Ok(())
}

fn normalize_graded_identities(
    identities: &mut [GradedDatasetIdentity],
) -> Result<(), StateFootprintError> {
    identities.sort();
    if identities
        .windows(2)
        .any(|pair| pair[0].manifest_blake3 == pair[1].manifest_blake3)
    {
        return Err(StateFootprintError::Invariant(
            "graded report merge contains a duplicate manifest".to_owned(),
        ));
    }
    let mut seeds = BTreeSet::new();
    for identity in identities.iter() {
        for seed in &identity.seeds {
            if !seeds.insert(*seed) {
                return Err(StateFootprintError::Invariant(format!(
                    "graded report merge overlaps raw seed {seed}"
                )));
            }
        }
    }
    Ok(())
}

fn validate_report(report: &StateFootprintReport) -> Result<(), StateFootprintError> {
    if report.schema_version != STATE_FOOTPRINT_SCHEMA_VERSION
        || report.experiment_id != STATE_FOOTPRINT_EXPERIMENT_ID
        || report.scientific_hash_algorithm != SCIENTIFIC_HASH_ALGORITHM
        || report.scientific_hash != scientific_hash(&report.scientific)?
        || report.provenance.current_v2_grid_radius != GRID_RADIUS
        || report.provenance.current_v2_grid_dim != GRID_DIM
        || report.provenance.current_v2_grid_size != GRID_SIZE
        || report.provenance.historical_legacy_nnue_cell_shape != 441
        || !is_blake3_digest(&report.provenance.source.v2_source_blake3)
        || !is_blake3_digest(&report.provenance.executable_blake3)
    {
        return Err(StateFootprintError::Invariant(
            "input report identity, provenance, or scientific hash is invalid".to_owned(),
        ));
    }
    validate_scientific_payload(&report.scientific)
}

fn is_blake3_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_scientific_payload(scientific: &ScientificPayload) -> Result<(), StateFootprintError> {
    if scientific.schema_version != STATE_FOOTPRINT_SCHEMA_VERSION
        || scientific.experiment_id != STATE_FOOTPRINT_EXPERIMENT_ID
        || scientific.ruleset != ruleset_identity()
        || scientific.configuration.radii != STATE_FOOTPRINT_RADII
        || scientific.configuration.outlier_radius != STATE_FOOTPRINT_OUTLIER_RADIUS
        || scientific.configuration.outlier_cap == 0
        || scientific.definitions != scientific_definitions()
        || scientific.invariants != geometry_invariants()?
        || scientific.adversarial_cases != adversarial_cases()?
    {
        return Err(StateFootprintError::Invariant(
            "scientific payload has an incompatible identity".to_owned(),
        ));
    }
    if let Some(corpus) = &scientific.generated {
        corpus.validate()?;
    }
    if let Some(corpus) = &scientific.position_datasets {
        corpus.validate()?;
    }
    if let Some(corpus) = &scientific.graded_oracle {
        corpus.validate()?;
    }
    let expected_completion = assess_completion(scientific);
    if scientific.completion != expected_completion {
        return Err(StateFootprintError::Invariant(
            "completion assessment does not match scientific contents".to_owned(),
        ));
    }
    Ok(())
}

pub fn scientific_hash(scientific: &ScientificPayload) -> Result<String, StateFootprintError> {
    let bytes = serde_json::to_vec(scientific)?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn current_report_provenance(
    dataset_paths: Vec<DatasetPathProvenance>,
    merged_inputs: Vec<MergedInputProvenance>,
) -> Result<ReportProvenance, StateFootprintError> {
    let executable = std::env::current_exe()?;
    Ok(ReportProvenance {
        source: source_provenance()?,
        executable_path: executable.display().to_string(),
        executable_blake3: checksum_file(&executable)?,
        current_v2_grid_radius: GRID_RADIUS,
        current_v2_grid_dim: GRID_DIM,
        current_v2_grid_size: GRID_SIZE,
        historical_legacy_nnue_cell_shape: 441,
        dataset_paths,
        merged_inputs,
    })
}

fn unix_seconds() -> Result<u64, StateFootprintError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|_| {
            StateFootprintError::Invariant("system clock is before the Unix epoch".to_owned())
        })
}

fn ruleset_identity() -> RulesetIdentity {
    RulesetIdentity {
        player_count: 4,
        scoring_cards: "AAAAA".to_owned(),
        habitat_bonuses: false,
        turns: 80,
        current_v2_grid_radius: GRID_RADIUS,
        current_v2_grid_dim: GRID_DIM,
        current_v2_grid_size: GRID_SIZE,
        current_control_support:
            "exact untruncated coordinate/entity support on the sparse-indexed V2 backing grid"
                .to_owned(),
        historical_legacy_nnue_cell_shape: 441,
        historical_legacy_nnue_role:
            "historical diagnostic shape only; any 441-cell arm requires explicit overflow"
                .to_owned(),
    }
}

fn scientific_definitions() -> BTreeMap<String, String> {
    [
        (
            "action_destination_availability",
            "Generated and graded states have exact selected tile destinations. Compact PositionRecord states infer one only when the next public record is the same game's next turn, exactly one coordinate was added to the acting board, and every other absolute board is unchanged; otherwise unavailable is counted.",
        ),
        (
            "allowed_wildlife_firing",
            "One firing per allowed-wildlife bit on each occupied tile; retained exactly when the tile anchor lies inside the disk.",
        ),
        (
            "best_integer_recenter",
            "The lexicographically smallest integer axial center (q, then r) minimizing maximum occupied-cell hex distance. Translated coordinates are original minus center and invert as translated plus center.",
        ),
        (
            "board_observation",
            "One serialized or generated absolute board viewed at one focal-relative seat in one pre-move state; four board observations per four-player state.",
        ),
        (
            "boards_with_any_overflow",
            "A board observation overflows when any occupied cell, legal frontier cell, exact selected destination, or distinct complete-candidate destination associated with that board lies outside the disk.",
        ),
        (
            "candidate_destination",
            "A distinct tile coordinate within one complete-action graded decision group. Candidate rows sharing a destination contribute one destination event and never additional states.",
        ),
        (
            "canonical_public_state_bytes",
            "Length of PublicGameState::canonical_bytes for directly generated states. Serialized PositionRecord sources report their fixed record bytes separately and do not fabricate canonical PublicGameState bytes.",
        ),
        (
            "completion_gate",
            "Formal completion requires exactly the preregistered generated origin first_seed=73000, games=625, strategy=pattern-aware, exactly 50,000 generated states and 200,000 generated board observations, plus exactly one validated graded-oracle train manifest and one validation manifest whose observed groups, candidate rows, and four boards per group exactly reconcile with manifest totals with zero skipped duplicate groups. PositionRecord datasets are supplementary and cannot substitute for either required arm.",
        ),
        (
            "current_v2_backing_grid",
            "The current V2 rules engine uses GRID_RADIUS=24, GRID_DIM=49, and GRID_SIZE=2401 with sparse occupied indices. Exact untruncated coordinate/entity support is the control.",
        ),
        (
            "centered_hex_capacity",
            "cells(r) = 1 + 3*r*(r+1) for integer radius r.",
        ),
        (
            "dense_byte_estimates",
            "Capacity times board observations, reported for one u8 per cell, one f32 per cell, and eleven u8 channels per cell. These are explicit layout estimates, not measured allocator bytes.",
        ),
        (
            "final_score_cohorts",
            "Generated-game board observations only: under_90, 90_to_99, and 100_plus, plus exact base-score bins, attached after simulation. No future hidden state enters any spatial measurement.",
        ),
        (
            "fixed_origin_radius",
            "Maximum axial hex distance from (0,0); event histograms use each event's distance rather than a board maximum.",
        ),
        (
            "frontier",
            "Every in-grid unoccupied neighbor of an occupied tile, deduplicated exactly as the rules engine's legal tile frontier.",
        ),
        (
            "habitat_component",
            "One terrain-specific component connected through matching terrain half-edges. It is fully retained when all cells are inside, crossing when both inside and outside cells exist, and fully outside when no cell is inside.",
        ),
        (
            "hidden_state_usage",
            "Spatial extraction uses public boards, public market-state serialization, turn, seats, and selected public actions only. Hidden tile or wildlife order is never inspected; direct generated final scores are used only for post-game cohort labels.",
        ),
        (
            "historical_441_shape",
            "The 441-cell 21x21 shape belongs to the historical legacy NNUE. It is diagnostic only and is never described or treated as the current V2 control; any 441-cell comparison requires explicit overflow.",
        ),
        (
            "merge_contract",
            "The merge_accumulator fields are additive integer counters, BTreeMap histograms/cohorts, and deterministic lexicographic top-K outliers. Derived fractions and the scientific hash must be rebuilt after merging disjoint origins. Every input and the merge executable must have identical provenance.source.v2_source_blake3 and executable_blake3; git revision and dirty status remain descriptive rather than decisive.",
        ),
        (
            "outlier",
            "One board observation whose occupied, frontier, selected-destination, or candidate-destination support exceeds radius 6 under fixed origin or best integer recentering. Records are the lexicographically first K identities and total remains exact.",
        ),
        (
            "phase",
            "Personal turns 1-5 opening, 6-10 early, 11-15 middle, and 16-20 late.",
        ),
        (
            "position_record_state",
            "One compact-entity-v2 PositionRecord is one pre-move state. All four serialized relative boards are inspected once; target fields are ignored.",
        ),
        (
            "scientific_hash",
            "BLAKE3 over compact serde_json bytes of the scientific object. Struct field order is fixed, maps use BTreeMap key order, and sets use BTreeSet value order. Timestamps, output paths, host/runtime telemetry, executable paths, and merge-input paths are outside the hashed object.",
        ),
        (
            "sparse_tokens",
            "Occupied-cell count plus deduplicated legal-frontier count; the sets are disjoint.",
        ),
        (
            "terrain_edge_firing",
            "Six directed terrain half-edge firings per occupied tile; all six are retained exactly when the tile anchor lies inside the disk.",
        ),
        (
            "wildlife_adjacency",
            "One undirected pair for adjacent occupied tiles that both contain wildlife. It is fully retained when both endpoints are inside, crossing when exactly one is inside, and fully outside when neither is inside.",
        ),
        (
            "wildlife_firing",
            "One firing per placed wildlife token, anchored at its occupied tile coordinate.",
        ),
    ]
    .into_iter()
    .map(|(key, value)| (key.to_owned(), value.to_owned()))
    .collect()
}

fn geometry_invariants() -> Result<GeometryInvariants, StateFootprintError> {
    let invariant = GeometryInvariants {
        centered_hex_capacity_formula: "1 + 3*r*(r+1)".to_owned(),
        radius_4_capacity: centered_hex_capacity(4),
        radius_5_capacity: centered_hex_capacity(5),
        radius_6_capacity: centered_hex_capacity(6),
        complete_121_cell_disk_exists: (0..=64).any(|radius| centered_hex_capacity(radius) == 121),
        d6_transform_count: 12,
        d6_radius_invariant: verify_d6_radius_invariant(),
        recentering_is_integer_exact_and_invertible: true,
    };
    if invariant.radius_4_capacity != 61
        || invariant.radius_5_capacity != 91
        || invariant.radius_6_capacity != 127
        || invariant.complete_121_cell_disk_exists
        || !invariant.d6_radius_invariant
    {
        return Err(StateFootprintError::Invariant(
            "geometry invariant self-check failed".to_owned(),
        ));
    }
    Ok(invariant)
}

fn d6_transform(mut coord: HexCoord, transform: usize) -> HexCoord {
    if transform >= 6 {
        coord = HexCoord::new(coord.q, -coord.q - coord.r);
    }
    for _ in 0..transform % 6 {
        coord = HexCoord::new(-coord.r, coord.q + coord.r);
    }
    coord
}

fn verify_d6_radius_invariant() -> bool {
    for q in -24..=24 {
        for r in -24..=24 {
            let coord = HexCoord::new(q, r);
            let radius = coord.distance(HexCoord::ORIGIN);
            for transform in 0..12 {
                if d6_transform(coord, transform).distance(HexCoord::ORIGIN) != radius {
                    return false;
                }
            }
        }
    }
    true
}

fn adversarial_cases() -> Result<Vec<AdversarialCaseReport>, StateFootprintError> {
    let mut straight = vec![HexCoord::ORIGIN];
    straight.extend((1..=11).map(|q| HexCoord::new(q, 0)));
    straight.extend((1..=11).map(|q| HexCoord::new(-q, 0)));
    let mut bent = (0..=11).map(|q| HexCoord::new(q, 0)).collect::<Vec<_>>();
    bent.extend((1..=11).map(|r| HexCoord::new(11, r)));
    Ok(vec![
        adversarial_case_report("straight_23_tile_chain", &straight)?,
        adversarial_case_report("bent_23_tile_chain", &bent)?,
    ])
}

fn adversarial_case_report(
    name: &str,
    coordinates: &[HexCoord],
) -> Result<AdversarialCaseReport, StateFootprintError> {
    if coordinates.len() != 23 {
        return Err(StateFootprintError::Invariant(format!(
            "adversarial case {name} does not contain 23 cells"
        )));
    }
    let mut board = Board::empty();
    for (index, coord) in coordinates.iter().enumerate() {
        board.place_tile(*coord, STANDARD_TILES[index], Rotation::ZERO)?;
    }
    board.validate().map_err(|message| {
        StateFootprintError::Invariant(format!("adversarial case {name} is invalid: {message}"))
    })?;
    let view = BoardView::from_board(&board);
    let occupied = view.coordinates();
    let frontier = view.frontier();
    let center = best_integer_center(&occupied);
    let occupied_fixed = maximum_radius(&occupied, HexCoord::ORIGIN);
    let occupied_recentered = maximum_radius(&occupied, center);
    let frontier_fixed = maximum_radius(&frontier, HexCoord::ORIGIN);
    let frontier_recentered = maximum_radius(&frontier, center);
    let occupied_overflow = row_count_outside(&occupied, center, STATE_FOOTPRINT_OUTLIER_RADIUS);
    let frontier_overflow = row_count_outside(&frontier, center, STATE_FOOTPRINT_OUTLIER_RADIUS);
    Ok(AdversarialCaseReport {
        name: name.to_owned(),
        legal_placed_tile_count: board.tile_count(),
        occupied_fixed_origin_radius: occupied_fixed,
        occupied_recentered_radius: occupied_recentered,
        frontier_fixed_origin_radius: frontier_fixed,
        frontier_recentered_radius: frontier_recentered,
        recenter_q: i16::from(center.q),
        recenter_r: i16::from(center.r),
        radius_6_occupied_overflow: occupied_overflow,
        radius_6_frontier_overflow: frontier_overflow,
        overflows_radius_6: occupied_overflow > 0 || frontier_overflow > 0,
    })
}

fn assess_completion(scientific: &ScientificPayload) -> CompletionAssessment {
    let mut reasons = Vec::new();
    let exact_generated_origin = scientific.configuration.generated_origins
        == [GeneratedOrigin {
            first_seed: PREREGISTERED_FIRST_SEED,
            games: PREREGISTERED_GAMES,
            strategy_id: StrategyKind::PatternAware.id().to_owned(),
        }];
    if !exact_generated_origin {
        reasons.push(
            "generated origin must be exactly first_seed=73000, games=625, strategy=pattern-aware"
                .to_owned(),
        );
    }
    let generated_complete = scientific.generated.as_ref().is_some_and(|corpus| {
        corpus.derived.counts.states == PREREGISTERED_STATES
            && corpus.derived.counts.board_observations == PREREGISTERED_BOARD_OBSERVATIONS
    });
    if !generated_complete {
        reasons.push(
            "generated measurements must contain exactly 50,000 states and 200,000 board observations"
                .to_owned(),
        );
    }

    let graded_identities = &scientific.configuration.graded_datasets;
    let graded_splits = graded_identities
        .iter()
        .map(|identity| identity.split.as_str())
        .collect::<Vec<_>>();
    if graded_identities.len() != 2
        || graded_splits
            .iter()
            .filter(|split| **split == "train")
            .count()
            != 1
        || graded_splits
            .iter()
            .filter(|split| **split == "validation")
            .count()
            != 1
    {
        reasons.push(
            "graded evidence must contain exactly one train manifest and one validation manifest"
                .to_owned(),
        );
    }
    let expected_graded_groups = graded_identities.iter().try_fold(0u64, |total, identity| {
        total.checked_add(u64::try_from(identity.total_groups).ok()?)
    });
    let expected_candidate_rows = graded_identities.iter().try_fold(0u64, |total, identity| {
        total.checked_add(u64::try_from(identity.total_candidate_rows).ok()?)
    });
    let expected_graded_boards = expected_graded_groups.and_then(|groups| groups.checked_mul(4));
    match (
        scientific.graded_oracle.as_ref(),
        expected_graded_groups,
        expected_graded_boards,
        expected_candidate_rows,
    ) {
        (Some(corpus), Some(groups), Some(boards), Some(candidate_rows))
            if corpus.derived.counts.states == groups
                && corpus.derived.counts.board_observations == boards
                && corpus.derived.counts.complete_candidate_rows == candidate_rows
                && corpus.derived.counts.duplicate_decision_groups_skipped == 0 => {}
        (Some(_), _, _, _) => reasons.push(
            "graded measurements must exactly match manifest group and candidate-row totals, four boards per group, with zero skipped duplicate groups"
                .to_owned(),
        ),
        (None, _, _, _) => {
            reasons.push("graded-oracle corpus measurements are absent".to_owned());
        }
    }

    for (label, corpus) in [
        ("generated", scientific.generated.as_ref()),
        ("position", scientific.position_datasets.as_ref()),
        ("graded", scientific.graded_oracle.as_ref()),
    ] {
        if let Some(corpus) = corpus {
            if corpus.merge_accumulator.outliers.truncated {
                reasons.push(format!("{label} radius-6 outlier list is truncated"));
            }
            if corpus.derived.radius_tables.fixed_origin.len() != STATE_FOOTPRINT_RADII.len()
                || corpus.derived.radius_tables.best_integer_recentered.len()
                    != STATE_FOOTPRINT_RADII.len()
            {
                reasons.push(format!("{label} radius tables are incomplete"));
            }
        }
    }
    if scientific.adversarial_cases.len() != 2
        || scientific
            .adversarial_cases
            .iter()
            .any(|case| !case.overflows_radius_6)
    {
        reasons.push("straight and bent adversarial boards must both overflow radius 6".to_owned());
    }
    if !scientific.invariants.d6_radius_invariant
        || scientific.invariants.complete_121_cell_disk_exists
    {
        reasons.push("geometry invariants are not satisfied".to_owned());
    }
    reasons.sort();
    reasons.dedup();
    CompletionAssessment {
        classification: if reasons.is_empty() {
            "state_footprint_census_complete"
        } else {
            "state_footprint_census_incomplete"
        }
        .to_owned(),
        complete: reasons.is_empty(),
        reasons,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centered_hex_capacity_matches_exact_disks_and_excludes_121() {
        assert_eq!(centered_hex_capacity(4), 61);
        assert_eq!(centered_hex_capacity(5), 91);
        assert_eq!(centered_hex_capacity(6), 127);
        assert!((0..=64).all(|radius| centered_hex_capacity(radius) != 121));
    }

    #[test]
    fn report_identifies_current_v2_grid_and_historical_441_diagnostic() {
        assert_eq!(GRID_RADIUS, 24);
        assert_eq!(GRID_DIM, 49);
        assert_eq!(GRID_SIZE, 2_401);

        let ruleset = ruleset_identity();
        assert_eq!(ruleset.current_v2_grid_radius, 24);
        assert_eq!(ruleset.current_v2_grid_dim, 49);
        assert_eq!(ruleset.current_v2_grid_size, 2_401);
        assert!(
            ruleset
                .current_control_support
                .contains("exact untruncated")
        );
        assert_eq!(ruleset.historical_legacy_nnue_cell_shape, 441);
        assert!(ruleset.historical_legacy_nnue_role.contains("diagnostic"));

        let storage = StorageReport::from_storage(&StorageAccumulator::default(), 2);
        assert_eq!(
            storage
                .current_v2_2401_cell_backing_grid_estimate
                .cell_slots,
            4_802
        );
        assert_eq!(
            storage.historical_441_cell_diagnostic_estimate.cell_slots,
            882
        );
    }

    #[test]
    fn exact_integer_recentering_minimizes_shifted_radius_and_round_trips() {
        let coordinates = (10..=14).map(|q| HexCoord::new(q, -3)).collect::<Vec<_>>();
        let center = best_integer_center(&coordinates);
        assert_eq!(center, HexCoord::new(12, -3));
        assert_eq!(maximum_radius(&coordinates, HexCoord::ORIGIN), 14);
        assert_eq!(maximum_radius(&coordinates, center), 2);
        for original in coordinates {
            let translated = translated_coordinate(original, center);
            assert_eq!(
                HexCoord::new(translated.q + center.q, translated.r + center.r),
                original
            );
        }
    }

    #[test]
    fn boundary_accounting_distinguishes_retained_crossing_and_outside() {
        let sets = vec![
            vec![HexCoord::ORIGIN],
            vec![HexCoord::ORIGIN, HexCoord::new(1, 0)],
            vec![HexCoord::new(2, 0)],
        ];
        let mut accumulator = BoundaryAccumulator::default();
        accumulator.observe_sets(&sets, HexCoord::ORIGIN, 0);
        assert_eq!(accumulator.retention.total, 3);
        assert_eq!(accumulator.retention.retained, 1);
        assert_eq!(accumulator.retention.overflow, 2);
        assert_eq!(accumulator.crossing, 1);
        assert_eq!(accumulator.fully_outside, 1);
        accumulator.validate("test boundary").unwrap();
    }

    #[test]
    fn all_twelve_d6_transforms_preserve_radius() {
        assert!(verify_d6_radius_invariant());
        let coordinate = HexCoord::new(7, -3);
        let transformed = (0..12)
            .map(|transform| d6_transform(coordinate, transform))
            .collect::<BTreeSet<_>>();
        assert_eq!(transformed.len(), 12);
        assert!(
            transformed
                .iter()
                .all(|value| value.distance(HexCoord::ORIGIN)
                    == coordinate.distance(HexCoord::ORIGIN))
        );
    }

    #[test]
    fn adversarial_straight_and_bent_boards_require_overflow() {
        let cases = adversarial_cases().unwrap();
        assert_eq!(cases.len(), 2);
        for case in cases {
            assert_eq!(case.legal_placed_tile_count, 23);
            assert!(case.occupied_recentered_radius > 6);
            assert!(case.frontier_recentered_radius > 6);
            assert!(case.radius_6_occupied_overflow > 0);
            assert!(case.radius_6_frontier_overflow > 0);
            assert!(case.overflows_radius_6);
        }
    }

    #[test]
    fn adjacent_position_records_infer_only_an_exact_public_tile_addition() {
        let mut game = cascadia_game::GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(800),
        )
        .unwrap();
        let before = PositionRecord::observe(&game, 800);
        let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible().unwrap();
        let action = game.legal_turn_actions(&prelude).unwrap().remove(0);
        game.apply(&action).unwrap();
        let after = PositionRecord::observe(&game, 800);
        assert_eq!(
            infer_exact_selected_destination(&before, &after).unwrap(),
            Some(action.tile.coord)
        );
        assert_eq!(
            infer_exact_selected_destination(&before, &before).unwrap(),
            None
        );
    }

    #[test]
    fn deterministic_merge_and_hash_are_order_independent() {
        let left = sample_accumulator(11, 0);
        let right = sample_accumulator(12, 1);

        let mut left_then_right = CorpusAccumulator::new(8);
        left_then_right.merge_from(left.clone()).unwrap();
        left_then_right.merge_from(right.clone()).unwrap();
        let mut right_then_left = CorpusAccumulator::new(8);
        right_then_left.merge_from(right).unwrap();
        right_then_left.merge_from(left).unwrap();
        assert_eq!(left_then_right, right_then_left);

        let first = test_scientific_payload(
            left_then_right,
            vec![
                GeneratedOrigin {
                    first_seed: 10,
                    games: 1,
                    strategy_id: StrategyKind::Random.id().to_owned(),
                },
                GeneratedOrigin {
                    first_seed: 11,
                    games: 1,
                    strategy_id: StrategyKind::Random.id().to_owned(),
                },
            ],
        );
        let second = test_scientific_payload(
            right_then_left,
            vec![
                GeneratedOrigin {
                    first_seed: 11,
                    games: 1,
                    strategy_id: StrategyKind::Random.id().to_owned(),
                },
                GeneratedOrigin {
                    first_seed: 10,
                    games: 1,
                    strategy_id: StrategyKind::Random.id().to_owned(),
                },
            ],
        );
        assert_eq!(first, second);
        assert_eq!(
            scientific_hash(&first).unwrap(),
            scientific_hash(&second).unwrap()
        );
        assert_eq!(
            first.configuration.generated_origins,
            vec![GeneratedOrigin {
                first_seed: 10,
                games: 2,
                strategy_id: StrategyKind::Random.id().to_owned(),
            }]
        );
    }

    #[test]
    fn merge_rejects_source_or_executable_drift() {
        let root = unique_test_directory("state-footprint-merge-drift");
        fs::create_dir_all(&root).unwrap();
        let first_path = root.join("first.json");
        let second_path = root.join("second.json");
        let output = root.join("merged.json");
        let scientific = test_scientific_payload(
            sample_accumulator(21, 0),
            vec![GeneratedOrigin {
                first_seed: 21,
                games: 1,
                strategy_id: StrategyKind::Random.id().to_owned(),
            }],
        );
        let source_a = "1".repeat(64);
        let source_b = "2".repeat(64);
        let executable_a = "a".repeat(64);
        let executable_b = "b".repeat(64);

        write_test_report(
            &first_path,
            &test_report(&scientific, &source_a, &executable_a),
        );
        write_test_report(
            &second_path,
            &test_report(&scientific, &source_b, &executable_a),
        );
        let source_error =
            merge_state_footprint_report_files(&[first_path.clone(), second_path.clone()], &output)
                .unwrap_err()
                .to_string();
        assert!(source_error.contains("v2_source_blake3 drift"));

        write_test_report(
            &second_path,
            &test_report(&scientific, &source_a, &executable_b),
        );
        let executable_error =
            merge_state_footprint_report_files(&[first_path, second_path], &output)
                .unwrap_err()
                .to_string();
        assert!(executable_error.contains("executable_blake3 drift"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn truncated_outlier_classification_is_never_complete() {
        let mut accumulator = CorpusAccumulator::new(1);
        accumulator
            .merge_from(sample_accumulator_with_cap(31, 0, 1))
            .unwrap();
        accumulator
            .merge_from(sample_accumulator_with_cap(32, 1, 1))
            .unwrap();
        assert!(accumulator.outliers.truncated);

        let payload = test_scientific_payload(
            accumulator,
            vec![GeneratedOrigin {
                first_seed: 31,
                games: 2,
                strategy_id: StrategyKind::Random.id().to_owned(),
            }],
        );
        assert!(!payload.completion.complete);
        assert!(
            payload
                .completion
                .reasons
                .iter()
                .any(|reason| reason.contains("outlier list is truncated"))
        );
    }

    #[test]
    fn frozen_completion_requires_exact_generated_origin() {
        let payload = completion_candidate_payload();
        assert!(assess_completion(&payload).complete);

        let mut extra_game = payload.clone();
        extra_game.configuration.generated_origins[0].games = 626;
        let assessment = assess_completion(&extra_game);
        assert!(!assessment.complete);
        assert!(
            assessment
                .reasons
                .iter()
                .any(|reason| reason.contains("games=625"))
        );
    }

    #[test]
    fn position_dataset_cannot_substitute_for_frozen_generated_arm() {
        let mut payload = completion_candidate_payload();
        payload.position_datasets = payload.generated.take();
        payload.configuration.generated_origins.clear();
        payload.configuration.position_datasets = vec![PositionDatasetIdentity {
            dataset_id: "position-supplement".to_owned(),
            manifest_blake3: "6".repeat(64),
            split: "train".to_owned(),
            strategy_id: StrategyKind::PatternAware.id().to_owned(),
            first_game_index: 0,
            completed_games: PREREGISTERED_GAMES,
            total_records: PREREGISTERED_STATES as usize,
            shards: Vec::new(),
        }];

        let assessment = assess_completion(&payload);
        assert!(!assessment.complete);
        assert!(
            assessment
                .reasons
                .iter()
                .any(|reason| reason.contains("generated measurements"))
        );
    }

    #[test]
    fn frozen_completion_rejects_extra_missing_or_duplicate_graded_groups() {
        let payload = completion_candidate_payload();

        let mut extra = payload.clone();
        let extra_counts = &mut extra.graded_oracle.as_mut().unwrap().derived.counts;
        extra_counts.states += 1;
        extra_counts.board_observations += 4;
        assert!(!assess_completion(&extra).complete);

        let mut missing = payload.clone();
        let missing_counts = &mut missing.graded_oracle.as_mut().unwrap().derived.counts;
        missing_counts.states -= 1;
        missing_counts.board_observations -= 4;
        assert!(!assess_completion(&missing).complete);

        let mut duplicate = payload;
        duplicate
            .graded_oracle
            .as_mut()
            .unwrap()
            .derived
            .counts
            .duplicate_decision_groups_skipped = 1;
        assert!(!assess_completion(&duplicate).complete);
    }

    #[test]
    fn frozen_completion_reconciles_graded_rows_boards_and_splits() {
        let payload = completion_candidate_payload();

        let mut wrong_rows = payload.clone();
        wrong_rows
            .graded_oracle
            .as_mut()
            .unwrap()
            .derived
            .counts
            .complete_candidate_rows += 1;
        assert!(!assess_completion(&wrong_rows).complete);

        let mut wrong_boards = payload.clone();
        wrong_boards
            .graded_oracle
            .as_mut()
            .unwrap()
            .derived
            .counts
            .board_observations += 1;
        assert!(!assess_completion(&wrong_boards).complete);

        let mut extra_split = payload;
        extra_split
            .configuration
            .graded_datasets
            .push(test_graded_identity("test", 1, 1));
        assert!(!assess_completion(&extra_split).complete);
    }

    #[test]
    fn games_zero_requires_a_dataset_root() {
        let invalid = StateFootprintConfig {
            first_seed: 0,
            games: 0,
            strategy: StrategyKind::Random,
            position_dataset_roots: Vec::new(),
            graded_dataset_roots: Vec::new(),
            outlier_cap: 1,
        };
        assert!(invalid.validate().is_err());

        let valid = StateFootprintConfig {
            position_dataset_roots: vec![PathBuf::from(".")],
            ..invalid
        };
        assert!(valid.validate().is_ok());
    }

    #[test]
    fn tiny_generated_run_writes_a_valid_machine_readable_report() {
        let output = std::env::temp_dir().join(format!(
            "state-footprint-tiny-{}-{}.json",
            std::process::id(),
            unix_seconds().unwrap()
        ));
        let report = run_state_footprint_census(
            &StateFootprintConfig {
                first_seed: 91_000,
                games: 1,
                strategy: StrategyKind::Random,
                position_dataset_roots: Vec::new(),
                graded_dataset_roots: Vec::new(),
                outlier_cap: 1_000,
            },
            &output,
        )
        .unwrap();
        let generated = report.scientific.generated.as_ref().unwrap();
        assert_eq!(generated.derived.counts.states, 80);
        assert_eq!(generated.derived.counts.board_observations, 320);
        assert_eq!(
            generated
                .merge_accumulator
                .storage
                .canonical_public_state_bytes
                .observations(),
            80
        );
        assert_eq!(
            generated
                .merge_accumulator
                .cohorts
                .boards
                .final_score_band
                .values()
                .map(|cohort| cohort.board_observations)
                .sum::<u64>(),
            320
        );
        write_state_footprint_report_atomic(&output, &report).unwrap();
        let decoded: StateFootprintReport =
            serde_json::from_reader(BufReader::new(File::open(&output).unwrap())).unwrap();
        validate_report(&decoded).unwrap();
        assert_eq!(decoded.scientific_hash, report.scientific_hash);
        fs::remove_file(output).unwrap();
    }

    fn sample_accumulator(seed: u64, translation: i8) -> CorpusAccumulator {
        sample_accumulator_with_cap(seed, translation, 8)
    }

    fn sample_accumulator_with_cap(
        seed: u64,
        translation: i8,
        outlier_cap: usize,
    ) -> CorpusAccumulator {
        let mut board = Board::empty();
        let coordinates = (0..=8)
            .map(|offset| HexCoord::new(translation + offset, 0))
            .collect::<Vec<_>>();
        for (index, coord) in coordinates.iter().enumerate() {
            board
                .place_tile(*coord, STANDARD_TILES[index], Rotation::ZERO)
                .unwrap();
        }
        let mut accumulator = CorpusAccumulator::new(outlier_cap);
        accumulator.state_count = 1;
        accumulator.selected_destination_available_states = 1;
        accumulator
            .storage
            .canonical_public_state_bytes
            .observe(128)
            .unwrap();
        observe_state_cohort(&mut accumulator, 1, 0);
        let selected = [HexCoord::new(translation + 9, 0)];
        let sample = observe_board(
            &mut accumulator,
            &BoardView::from_board(&board),
            &selected,
            &[],
            &OutlierContext {
                source_kind: "generated",
                dataset_id: None,
                seed_or_game_index: seed,
                turn: 0,
                decision_group_id: None,
                current_player: 0,
                focal_relative_seat: 0,
                absolute_seat: 0,
                public_state_hash: format!("{seed:064x}"),
            },
        )
        .unwrap();
        observe_board_cohorts(&mut accumulator, &sample, 1, 0, 0).unwrap();
        accumulator.validate().unwrap();
        accumulator
    }

    fn test_report(
        scientific: &ScientificPayload,
        source_blake3: &str,
        executable_blake3: &str,
    ) -> StateFootprintReport {
        StateFootprintReport {
            schema_version: STATE_FOOTPRINT_SCHEMA_VERSION,
            experiment_id: STATE_FOOTPRINT_EXPERIMENT_ID.to_owned(),
            created_unix_seconds: 0,
            output_path: "test.json".to_owned(),
            provenance: ReportProvenance {
                source: SourceProvenance {
                    git_revision: "test".to_owned(),
                    git_dirty: true,
                    git_status_blake3: "0".repeat(64),
                    v2_source_blake3: source_blake3.to_owned(),
                },
                executable_path: "state-footprint-test".to_owned(),
                executable_blake3: executable_blake3.to_owned(),
                current_v2_grid_radius: GRID_RADIUS,
                current_v2_grid_dim: GRID_DIM,
                current_v2_grid_size: GRID_SIZE,
                historical_legacy_nnue_cell_shape: 441,
                dataset_paths: Vec::new(),
                merged_inputs: Vec::new(),
            },
            runtime: CensusRuntime::default(),
            scientific_hash_algorithm: SCIENTIFIC_HASH_ALGORITHM.to_owned(),
            scientific_hash: scientific_hash(scientific).unwrap(),
            scientific: scientific.clone(),
        }
    }

    fn write_test_report(path: &Path, report: &StateFootprintReport) {
        serde_json::to_writer(BufWriter::new(File::create(path).unwrap()), report).unwrap();
    }

    fn unique_test_directory(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{}-{nanos}", std::process::id()))
    }

    fn completion_candidate_payload() -> ScientificPayload {
        let mut payload = test_scientific_payload(
            sample_accumulator(73_000, 0),
            vec![GeneratedOrigin {
                first_seed: PREREGISTERED_FIRST_SEED,
                games: PREREGISTERED_GAMES,
                strategy_id: StrategyKind::PatternAware.id().to_owned(),
            }],
        );
        let generated_counts = &mut payload.generated.as_mut().unwrap().derived.counts;
        generated_counts.states = PREREGISTERED_STATES;
        generated_counts.board_observations = PREREGISTERED_BOARD_OBSERVATIONS;

        payload.configuration.graded_datasets = vec![
            test_graded_identity("train", 3, 30),
            test_graded_identity("validation", 2, 20),
        ];
        let mut graded = CorpusScientific::new(sample_accumulator(80_000, 0));
        graded.derived.counts.states = 5;
        graded.derived.counts.board_observations = 20;
        graded.derived.counts.selected_destination_available_states = 5;
        graded
            .derived
            .counts
            .selected_destination_unavailable_states = 0;
        graded.derived.counts.complete_candidate_rows = 50;
        graded.derived.counts.duplicate_decision_groups_skipped = 0;
        payload.graded_oracle = Some(graded);
        payload.completion = assess_completion(&payload);
        payload
    }

    fn test_graded_identity(
        split: &str,
        total_groups: usize,
        total_candidate_rows: usize,
    ) -> GradedDatasetIdentity {
        GradedDatasetIdentity {
            dataset_id: format!("graded-{split}"),
            manifest_blake3: match split {
                "train" => "3".repeat(64),
                "validation" => "4".repeat(64),
                _ => "5".repeat(64),
            },
            split: split.to_owned(),
            completed_games: 1,
            total_groups,
            total_candidate_rows,
            seeds: vec![match split {
                "train" => 1,
                "validation" => 2,
                _ => 3,
            }],
            shards: Vec::new(),
        }
    }

    fn test_scientific_payload(
        accumulator: CorpusAccumulator,
        origins: Vec<GeneratedOrigin>,
    ) -> ScientificPayload {
        let mut payload = ScientificPayload {
            schema_version: STATE_FOOTPRINT_SCHEMA_VERSION,
            experiment_id: STATE_FOOTPRINT_EXPERIMENT_ID.to_owned(),
            ruleset: ruleset_identity(),
            configuration: ScientificConfiguration {
                radii: STATE_FOOTPRINT_RADII.to_vec(),
                outlier_radius: STATE_FOOTPRINT_OUTLIER_RADIUS,
                outlier_cap: accumulator.outliers.cap,
                generated_origins: normalize_generated_origins(&origins).unwrap(),
                position_datasets: Vec::new(),
                graded_datasets: Vec::new(),
            },
            definitions: scientific_definitions(),
            invariants: geometry_invariants().unwrap(),
            generated: Some(CorpusScientific::new(accumulator)),
            position_datasets: None,
            graded_oracle: None,
            adversarial_cases: adversarial_cases().unwrap(),
            completion: CompletionAssessment {
                classification: String::new(),
                complete: false,
                reasons: Vec::new(),
            },
        };
        payload.completion = assess_completion(&payload);
        payload
    }
}
