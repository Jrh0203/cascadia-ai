//! Isolated public-state adapter for evaluating the historical v1 teacher.
//!
//! This module is feature-gated and belongs to the differential research
//! boundary. Production v2 crates never depend on v1.

use std::{
    collections::HashMap,
    path::Path,
    sync::{
        OnceLock,
        mpsc::{self, SyncSender},
    },
    thread,
};

use blake3::Hasher;
use cascadia_ai::{
    eval::ScoredMove,
    mce::{
        GreedyMceAlloc, best_move_nnue_rollout_mce, expanded_candidates, nnue_prefilter_candidates,
        rank_candidates_nnue_direct, score_nnue_rollout_mce_seq_halving,
        score_nnue_rollout_mce_seq_halving_exact,
    },
    nnue::NNUENetwork,
    nnue_batch::{
        BatchedNnueDiagnostics, BatchedRolloutConfig, BatchedRolloutLeafTiming,
        RolloutSeedCoupling, RolloutValueSample, SparseNnueAfterstate, SparseNnueEvaluator,
        evaluate_sparse_rows_deduplicated, nnue_prefilter_candidates_batched,
        prepare_sparse_nnue_afterstates,
        score_nnue_rollout_mce_seq_halving_batched_with_config_and_coupling,
        score_nnue_rollout_mce_seq_halving_batched_with_leaf_config_and_coupling,
        score_nnue_rollout_mce_seq_halving_batched_with_samples_config_and_coupling,
    },
    search::execute_scored_move,
};
use cascadia_core as v1;
use cascadia_game as v2;
use cascadia_model::{
    DEFAULT_SPARSE_NNUE_SHARED_MEMORY_BYTES, ExactNnueHiddenPrediction, ModelError, ModelProcess,
};
use cascadia_sim::{
    PATTERN_AWARE_STRATEGY_ID, PatternAwareConfig, SimulationError, rank_habitat_setup_actions,
    rank_pattern_actions, select_pattern_action, strategy_rng,
};
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const LEGACY_TEACHER_STRATEGY_ID: &str =
    "isolated-retained-legacy-main-policy-v1-k32-r600-lmr-no-paid-prelude";
pub const FILTERED_LEGACY_TEACHER_STRATEGY_ID: &str =
    "canonical-filtered-legacy-main-policy-v1-k32-r600-lmr-no-paid-prelude";
pub const HEURISTIC_LEGACY_TEACHER_STRATEGY_ID: &str =
    "canonical-action-legacy-heuristic-v1-k32-r600-lmr-no-paid-prelude";
pub const DETERMINISTIC_EVIDENCE_TEACHER_STRATEGY_ID: &str =
    "canonical-action-legacy-heuristic-deterministic-v2-k32-r600-lmr-no-paid-prelude";
pub const EXACT_MLX_LEGACY_TEACHER_STRATEGY_ID: &str =
    "canonical-action-legacy-exact-mlx-v1-k32-r600-lmr-no-paid-prelude";
pub const LEGACY_DIRECT_POLICY_STRATEGY_ID: &str =
    "canonical-action-qualified-v1-direct-nnue-v4opp-v1";
pub const DEFAULT_EXACT_MLX_PIPELINE_CHUNK_STATES: usize = 96;
pub const EXACT_MLX_STATIC_SCREEN_CHUNK_ROWS: usize = 4_096;

const ALLOWED_LEGACY_ENVIRONMENT: &[(&str, &str)] =
    &[("MCE_LMR", "1"), ("MCE_DIVERSE_PREFILTER", "1")];
const DIAGNOSTIC_LEGACY_ENVIRONMENT: &[(&str, &str)] = &[
    ("CASCADIA_NNUE_STAGE_TIMINGS", "1"),
    ("CASCADIA_NNUE_ROW_REUSE_DIAGNOSTICS", "1"),
    ("CASCADIA_NNUE_TEMPLATE_REUSE_DIAGNOSTICS", "1"),
    ("CASCADIA_MULTIPLEX_ROW_REUSE_DIAGNOSTICS", "1"),
    ("CASCADIA_MLX_STAGE_TIMINGS", "1"),
    ("CASCADIA_MLX_ACTIVATION_DIAGNOSTICS", "1"),
];
// These variables belong to the authenticated container transport envelope,
// not to the legacy evaluator.  Their values describe artifact publication
// and provenance only; they cannot alter candidate generation, search, RNG,
// or scoring.  Keep this whitelist explicit so an unregistered CASCADIA_* or
// MCE_* knob still fails the frozen-environment gate.
const TRANSPORT_LEGACY_ENVIRONMENT: &[&str] = &[
    "CASCADIA_APPLICATION_METADATA_JSON",
    "CASCADIA_OUTPUT_ROOT",
    "CASCADIA_PROTOCOL_VERSION",
    "CASCADIA_RETRYABLE_EXIT_CODES",
];

fn permitted_legacy_environment_entry(key: &str, value: &str) -> bool {
    ALLOWED_LEGACY_ENVIRONMENT
        .iter()
        .any(|(allowed, expected)| *allowed == key && *expected == value)
        || DIAGNOSTIC_LEGACY_ENVIRONMENT
            .iter()
            .any(|(allowed, expected)| *allowed == key && *expected == value)
        || TRANSPORT_LEGACY_ENVIRONMENT.contains(&key)
}

pub fn exact_mlx_pipeline_chunk_states() -> usize {
    let Ok(value) = std::env::var("LEGACY_TEACHER_MLX_PIPELINE_CHUNK_STATES") else {
        return DEFAULT_EXACT_MLX_PIPELINE_CHUNK_STATES;
    };
    value
        .parse::<usize>()
        .ok()
        .filter(|&parsed| parsed > 0)
        .unwrap_or_else(|| {
            panic!(
                "LEGACY_TEACHER_MLX_PIPELINE_CHUNK_STATES must be a positive integer, found {value}"
            )
        })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizedScore {
    pub habitat: [u16; 5],
    pub wildlife: [u16; 5],
    pub nature_tokens: u16,
    pub base_total: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranslationEvidence {
    pub public_state_blake3: String,
    pub reconstructed_inventory_blake3: String,
    pub unseen_tiles: usize,
    pub unseen_wildlife: usize,
    pub checked_boards: usize,
    pub maximum_absolute_coordinate: i8,
    pub elk_score_mismatch_boards: usize,
    pub elk_v2_minus_v1_total: u64,
    pub elk_v2_minus_v1_max: u16,
}

pub struct LegacyTranslation {
    pub game: v1::game::GameState,
    pub evidence: TranslationEvidence,
    coordinate_frames: [LegacyCoordinateFrame; 4],
}

/// Lossless coordinate-frame adapter between the canonical V2 board and the
/// historical V1 evaluator's fixed 21x21 storage window.
///
/// Cascadia scoring and legality are translation invariant.  Keeping an
/// explicit frame per player therefore preserves the exact position while
/// allowing naturally translated legal boards (for example, q=11) to be
/// evaluated by V1.  Zero is retained whenever the board already fits so the
/// established V1 identity is unchanged on the qualified domain.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct LegacyCoordinateFrame {
    q_offset: i8,
    r_offset: i8,
}

impl LegacyCoordinateFrame {
    const ZERO: Self = Self {
        q_offset: 0,
        r_offset: 0,
    };

    fn for_board(board: &v2::Board) -> Result<Self, BridgeError> {
        let mut coordinates = board.placed_tiles().map(|(coord, _)| coord);
        let Some(first) = coordinates.next() else {
            return Ok(Self::ZERO);
        };
        let (mut min_q, mut max_q, mut min_r, mut max_r) = (first.q, first.q, first.r, first.r);
        for coord in coordinates {
            min_q = min_q.min(coord.q);
            max_q = max_q.max(coord.q);
            min_r = min_r.min(coord.r);
            max_r = max_r.max(coord.r);
        }
        Ok(Self {
            q_offset: Self::axis_offset('q', min_q, max_q)?,
            r_offset: Self::axis_offset('r', min_r, max_r)?,
        })
    }

    fn axis_offset(axis: char, minimum: i8, maximum: i8) -> Result<i8, BridgeError> {
        let minimum = i16::from(minimum);
        let maximum = i16::from(maximum);
        let lower = -10i16 - minimum;
        let upper = 10i16 - maximum;
        if lower > upper {
            return Err(BridgeError::CoordinateSpanOutOfRange {
                axis,
                minimum: minimum as i8,
                maximum: maximum as i8,
            });
        }
        Ok(0i16.clamp(lower, upper) as i8)
    }

    fn to_legacy(self, coord: v2::HexCoord) -> Result<v1::hex::HexCoord, BridgeError> {
        let q = coord
            .q
            .checked_add(self.q_offset)
            .ok_or(BridgeError::CoordinateOutOfRange {
                q: coord.q,
                r: coord.r,
            })?;
        let r = coord
            .r
            .checked_add(self.r_offset)
            .ok_or(BridgeError::CoordinateOutOfRange {
                q: coord.q,
                r: coord.r,
            })?;
        let translated = v1::hex::HexCoord::new(q, r);
        if translated.to_index().is_none() {
            return Err(BridgeError::CoordinateOutOfRange {
                q: coord.q,
                r: coord.r,
            });
        }
        Ok(translated)
    }

    fn to_v2(self, q: i8, r: i8) -> Result<v2::HexCoord, BridgeError> {
        let q = q
            .checked_sub(self.q_offset)
            .ok_or(BridgeError::CoordinateOutOfRange { q, r })?;
        let r = r
            .checked_sub(self.r_offset)
            .ok_or(BridgeError::CoordinateOutOfRange { q, r })?;
        Ok(v2::HexCoord::new(q, r))
    }
}

impl LegacyTranslation {
    fn current_player_frame(&self) -> LegacyCoordinateFrame {
        self.coordinate_frames[self.game.current_player]
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BridgeDiagnostics {
    pub states_attempted: usize,
    pub states_translated: usize,
    pub checked_boards: usize,
    pub expanded_candidates: usize,
    pub expanded_candidates_legal: usize,
    pub expanded_candidates_illegal: usize,
    pub prefiltered_candidates: usize,
    pub prefiltered_candidates_legal: usize,
    pub prefiltered_candidates_illegal: usize,
    pub selected_actions: usize,
    pub selected_actions_legal: usize,
    pub fallbacks: usize,
    pub maximum_absolute_coordinate: i8,
    pub first_errors: Vec<String>,
    pub malformed_expanded_examples: Vec<String>,
    pub elk_score_mismatch_boards: usize,
    pub elk_v2_minus_v1_total: u64,
    pub elk_v2_minus_v1_max: u16,
    pub pattern_frontier_candidates: usize,
    pub selected_actions_in_pattern_frontier: usize,
    pub selected_independent_actions: usize,
    pub selected_independent_actions_in_pattern_frontier: usize,
    pub selected_actions_by_phase: [usize; 3],
    pub selected_actions_in_pattern_frontier_by_phase: [usize; 3],
    pub pattern_frontier_miss_examples: Vec<String>,
    pub habitat_candidates_generated: usize,
    pub habitat_candidates_novel: usize,
    pub habitat_candidates_retained: usize,
    pub selected_novel_habitat_candidates: usize,
}

impl BridgeDiagnostics {
    fn extend_examples(target: &mut Vec<String>, source: Vec<String>) {
        let remaining = 8usize.saturating_sub(target.len());
        target.extend(source.into_iter().take(remaining));
    }

    pub fn merge_from(&mut self, source: Self) {
        self.states_attempted = self
            .states_attempted
            .saturating_add(source.states_attempted);
        self.states_translated = self
            .states_translated
            .saturating_add(source.states_translated);
        self.checked_boards = self.checked_boards.saturating_add(source.checked_boards);
        self.expanded_candidates = self
            .expanded_candidates
            .saturating_add(source.expanded_candidates);
        self.expanded_candidates_legal = self
            .expanded_candidates_legal
            .saturating_add(source.expanded_candidates_legal);
        self.expanded_candidates_illegal = self
            .expanded_candidates_illegal
            .saturating_add(source.expanded_candidates_illegal);
        self.prefiltered_candidates = self
            .prefiltered_candidates
            .saturating_add(source.prefiltered_candidates);
        self.prefiltered_candidates_legal = self
            .prefiltered_candidates_legal
            .saturating_add(source.prefiltered_candidates_legal);
        self.prefiltered_candidates_illegal = self
            .prefiltered_candidates_illegal
            .saturating_add(source.prefiltered_candidates_illegal);
        self.selected_actions = self
            .selected_actions
            .saturating_add(source.selected_actions);
        self.selected_actions_legal = self
            .selected_actions_legal
            .saturating_add(source.selected_actions_legal);
        self.fallbacks = self.fallbacks.saturating_add(source.fallbacks);
        self.maximum_absolute_coordinate = self
            .maximum_absolute_coordinate
            .max(source.maximum_absolute_coordinate);
        Self::extend_examples(&mut self.first_errors, source.first_errors);
        Self::extend_examples(
            &mut self.malformed_expanded_examples,
            source.malformed_expanded_examples,
        );
        self.elk_score_mismatch_boards = self
            .elk_score_mismatch_boards
            .saturating_add(source.elk_score_mismatch_boards);
        self.elk_v2_minus_v1_total = self
            .elk_v2_minus_v1_total
            .saturating_add(source.elk_v2_minus_v1_total);
        self.elk_v2_minus_v1_max = self.elk_v2_minus_v1_max.max(source.elk_v2_minus_v1_max);
        self.pattern_frontier_candidates = self
            .pattern_frontier_candidates
            .saturating_add(source.pattern_frontier_candidates);
        self.selected_actions_in_pattern_frontier = self
            .selected_actions_in_pattern_frontier
            .saturating_add(source.selected_actions_in_pattern_frontier);
        self.selected_independent_actions = self
            .selected_independent_actions
            .saturating_add(source.selected_independent_actions);
        self.selected_independent_actions_in_pattern_frontier = self
            .selected_independent_actions_in_pattern_frontier
            .saturating_add(source.selected_independent_actions_in_pattern_frontier);
        for phase in 0..3 {
            self.selected_actions_by_phase[phase] = self.selected_actions_by_phase[phase]
                .saturating_add(source.selected_actions_by_phase[phase]);
            self.selected_actions_in_pattern_frontier_by_phase[phase] = self
                .selected_actions_in_pattern_frontier_by_phase[phase]
                .saturating_add(source.selected_actions_in_pattern_frontier_by_phase[phase]);
        }
        Self::extend_examples(
            &mut self.pattern_frontier_miss_examples,
            source.pattern_frontier_miss_examples,
        );
        self.habitat_candidates_generated = self
            .habitat_candidates_generated
            .saturating_add(source.habitat_candidates_generated);
        self.habitat_candidates_novel = self
            .habitat_candidates_novel
            .saturating_add(source.habitat_candidates_novel);
        self.habitat_candidates_retained = self
            .habitat_candidates_retained
            .saturating_add(source.habitat_candidates_retained);
        self.selected_novel_habitat_candidates = self
            .selected_novel_habitat_candidates
            .saturating_add(source.selected_novel_habitat_candidates);
    }

    pub fn fallback_rate(&self) -> f64 {
        if self.states_attempted == 0 {
            0.0
        } else {
            self.fallbacks as f64 / self.states_attempted as f64
        }
    }

    pub fn pattern_frontier_recall(&self) -> f64 {
        ratio(
            self.selected_actions_in_pattern_frontier,
            self.selected_actions,
        )
    }

    pub fn independent_pattern_frontier_recall(&self) -> f64 {
        ratio(
            self.selected_independent_actions_in_pattern_frontier,
            self.selected_independent_actions,
        )
    }

    pub fn pattern_frontier_recall_by_phase(&self) -> [f64; 3] {
        std::array::from_fn(|phase| {
            ratio(
                self.selected_actions_in_pattern_frontier_by_phase[phase],
                self.selected_actions_by_phase[phase],
            )
        })
    }

    fn record_error(&mut self, error: &BridgeError) {
        if self.first_errors.len() < 8 {
            self.first_errors.push(error.to_string());
        }
    }

    pub fn record_external_error(&mut self, error: &BridgeError) {
        self.record_error(error);
    }

    fn record_malformed_expanded(&mut self, error: &BridgeError) {
        if self.malformed_expanded_examples.len() < 8 {
            self.malformed_expanded_examples.push(error.to_string());
        }
    }

    fn record_translation(&mut self, evidence: &TranslationEvidence) {
        self.states_translated += 1;
        self.checked_boards += evidence.checked_boards;
        self.maximum_absolute_coordinate = self
            .maximum_absolute_coordinate
            .max(evidence.maximum_absolute_coordinate);
        self.elk_score_mismatch_boards += evidence.elk_score_mismatch_boards;
        self.elk_v2_minus_v1_total += evidence.elk_v2_minus_v1_total;
        self.elk_v2_minus_v1_max = self.elk_v2_minus_v1_max.max(evidence.elk_v2_minus_v1_max);
    }

    fn record_pattern_frontier_recall(
        &mut self,
        game: &v2::GameState,
        action: &v2::TurnAction,
        frontier: &[cascadia_sim::PatternCandidate],
    ) {
        self.pattern_frontier_candidates += frontier.len();
        let recalled = frontier.iter().any(|candidate| candidate.action == *action);
        let phase = match game.completed_turns() {
            0..=26 => 0,
            27..=53 => 1,
            _ => 2,
        };
        self.selected_actions_by_phase[phase] += 1;
        if recalled {
            self.selected_actions_in_pattern_frontier += 1;
            self.selected_actions_in_pattern_frontier_by_phase[phase] += 1;
        } else if self.pattern_frontier_miss_examples.len() < 8 {
            self.pattern_frontier_miss_examples.push(
                serde_json::to_string(action)
                    .expect("serializing an in-memory canonical action cannot fail"),
            );
        }
        if matches!(action.draft, v2::DraftChoice::Independent { .. }) {
            self.selected_independent_actions += 1;
            if recalled {
                self.selected_independent_actions_in_pattern_frontier += 1;
            }
        }
    }
}

#[derive(Debug, Error)]
pub enum BridgeError {
    #[error("legacy teacher supports exactly four players, received {0}")]
    PlayerCount(u8),
    #[error("v2 coordinate {q},{r} is outside the legacy board")]
    CoordinateOutOfRange { q: i8, r: i8 },
    #[error(
        "v2 board {axis}-coordinate span [{minimum},{maximum}] cannot fit the legacy 21-cell axis"
    )]
    CoordinateSpanOutOfRange {
        axis: char,
        minimum: i8,
        maximum: i8,
    },
    #[error("failed to place translated tile at {q},{r}")]
    TilePlacement { q: i8, r: i8 },
    #[error("failed to place translated wildlife at {q},{r}")]
    WildlifePlacement { q: i8, r: i8 },
    #[error(
        "translated score mismatch for player {player}: v2={v2:?}, v1={v1:?}, wildlife={wildlife:?}"
    )]
    ScoreMismatch {
        player: usize,
        v2: NormalizedScore,
        v1: NormalizedScore,
        wildlife: Vec<(v2::HexCoord, v2::Wildlife)>,
    },
    #[error("public market slot {0} is incomplete")]
    IncompleteMarket(usize),
    #[error("standard tile id {0} appears more than once in public state")]
    DuplicateTile(u8),
    #[error("standard tile conservation failed: expected 85, reconstructed {0}")]
    TileConservation(usize),
    #[error("wildlife conservation failed for {wildlife:?}: visible count {visible}")]
    WildlifeConservation { wildlife: v2::Wildlife, visible: u8 },
    #[error("legacy candidate has an invalid market slot")]
    InvalidMarketSlot,
    #[error("legacy candidate has only one wildlife coordinate component")]
    PartialWildlifeCoordinate,
    #[error("mapped legacy action is illegal in v2: {0}")]
    IllegalMappedAction(String),
    #[error("legacy teacher produced no candidate")]
    NoCandidate,
    #[error("canonical V2 candidate cannot execute in the translated legacy state: {0}")]
    CanonicalCandidateTranslation(String),
    #[error("legacy environment is not frozen: {0}")]
    LegacyEnvironment(String),
    #[error("v2 rule operation failed: {0}")]
    Rules(String),
    #[error("v2 pattern fallback failed: {0}")]
    Pattern(String),
    #[error("failed to load legacy weights: {0}")]
    Weights(String),
    #[error("exact MLX legacy teacher failed: {0}")]
    ExactMlx(String),
}

impl BridgeError {
    pub fn permits_coordinate_fallback(&self) -> bool {
        matches!(
            self,
            Self::CoordinateOutOfRange { .. } | Self::CoordinateSpanOutOfRange { .. }
        )
    }
}

pub fn validate_legacy_environment() -> Result<(), BridgeError> {
    for (key, expected) in ALLOWED_LEGACY_ENVIRONMENT {
        match std::env::var(key) {
            Ok(actual) if actual == *expected => {}
            Ok(actual) => {
                return Err(BridgeError::LegacyEnvironment(format!(
                    "{key} must equal {expected}, found {actual}"
                )));
            }
            Err(_) => {
                return Err(BridgeError::LegacyEnvironment(format!(
                    "{key} must be set to {expected}"
                )));
            }
        }
    }
    for (key, value) in std::env::vars() {
        if !(key.starts_with("MCE_") || key.starts_with("CASCADIA_")) {
            continue;
        }
        if permitted_legacy_environment_entry(&key, &value) {
            continue;
        }
        return Err(BridgeError::LegacyEnvironment(format!(
            "unexpected {key}={value}"
        )));
    }
    Ok(())
}

pub fn load_legacy_weights(path: &Path) -> Result<NNUENetwork, BridgeError> {
    NNUENetwork::load(path).map_err(|error| BridgeError::Weights(error.to_string()))
}

pub fn canonical_prelude(game: &v2::GameState) -> v2::MarketPrelude {
    v2::MarketPrelude {
        replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
        wildlife_wipes: Vec::new(),
    }
}

pub fn legacy_search_rng(public_state_blake3: &str) -> rand::rngs::StdRng {
    rand::rngs::StdRng::from_seed(domain_hash(
        b"legacy-teacher-search-rng",
        &hex_to_bytes(public_state_blake3),
    ))
}

pub fn legacy_search_rng_in_domain(
    public_state_blake3: &str,
    seed_domain: &[u8],
) -> rand::rngs::StdRng {
    let mut hasher = Hasher::new();
    hasher.update(b"legacy-teacher-search-rng-domain");
    hasher.update(public_state_blake3.as_bytes());
    hasher.update(&(seed_domain.len() as u64).to_le_bytes());
    hasher.update(seed_domain);
    rand::rngs::StdRng::from_seed(*hasher.finalize().as_bytes())
}

pub fn spawn_exact_mlx_process(
    server_program: &str,
    model_dir: &Path,
) -> Result<ModelProcess, BridgeError> {
    let args = [
        std::ffi::OsString::from("run"),
        std::ffi::OsString::from("cascadia-mlx-legacy-nnue-serve"),
        std::ffi::OsString::from("--model-dir"),
        model_dir.as_os_str().to_owned(),
    ];
    let mut process = ModelProcess::spawn_with_sparse_nnue_shared_memory(
        server_program,
        args,
        DEFAULT_SPARSE_NNUE_SHARED_MEMORY_BYTES,
    )
    .map_err(|error| BridgeError::ExactMlx(error.to_string()))?;
    let warmup = process
        .predict_sparse_nnue_csr_exact(&[Vec::new()])
        .map_err(|error| BridgeError::ExactMlx(error.to_string()))?;
    if warmup.len() != 1 || !warmup[0].is_finite() {
        return Err(BridgeError::ExactMlx(
            "exact MLX service warmup returned an invalid value".to_owned(),
        ));
    }
    Ok(process)
}

pub fn translate_public_state(
    public: &v2::PublicGameState,
) -> Result<LegacyTranslation, BridgeError> {
    translate_public_state_with_score_policy(public, ScoreParityPolicy::Exact)
}

pub fn translate_public_state_allowing_legacy_elk_undercount(
    public: &v2::PublicGameState,
) -> Result<LegacyTranslation, BridgeError> {
    translate_public_state_with_score_policy(public, ScoreParityPolicy::AllowLegacyElkUndercount)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScoreParityPolicy {
    Exact,
    AllowLegacyElkUndercount,
}

fn translate_public_state_with_score_policy(
    public: &v2::PublicGameState,
    score_policy: ScoreParityPolicy,
) -> Result<LegacyTranslation, BridgeError> {
    let player_count = public.config().player_count;
    if player_count != 4 {
        return Err(BridgeError::PlayerCount(player_count));
    }
    let public_bytes =
        serde_json::to_vec(public).expect("serializing an in-memory public state cannot fail");
    let public_digest = domain_hash(b"legacy-teacher-public-state", &public_bytes);
    let mut inventory_rng = rand::rngs::StdRng::from_seed(domain_hash(
        b"legacy-teacher-public-inventory",
        &public_digest,
    ));

    let mut maximum_absolute_coordinate = 0i8;
    let mut elk_score_mismatch_boards = 0usize;
    let mut elk_v2_minus_v1_total = 0u64;
    let mut elk_v2_minus_v1_max = 0u16;
    let mut boards = Vec::with_capacity(public.boards().len());
    let mut coordinate_frames = Vec::with_capacity(public.boards().len());
    for (player, board) in public.boards().iter().enumerate() {
        let (translated, coordinate_frame) =
            translate_board(board, &mut maximum_absolute_coordinate)?;
        let v2_score = normalized_v2_score(board, public.config().scoring_cards);
        let v1_score = normalized_v1_score(&translated);
        if v2_score != v1_score {
            let elk_delta = allowed_legacy_elk_undercount(&v2_score, &v1_score);
            if score_policy == ScoreParityPolicy::Exact || elk_delta.is_none() {
                return Err(score_mismatch(player, board, v2_score, v1_score));
            }
            let elk_delta = elk_delta.expect("allowed score mismatch was checked above");
            elk_score_mismatch_boards += 1;
            elk_v2_minus_v1_total += u64::from(elk_delta);
            elk_v2_minus_v1_max = elk_v2_minus_v1_max.max(elk_delta);
        }
        boards.push(translated);
        coordinate_frames.push(coordinate_frame);
    }

    let mut used_tiles = [false; 85];
    for board in public.boards() {
        for (_, placed) in board.placed_tiles() {
            if placed.tile.id.0 < 85 {
                let index = usize::from(placed.tile.id.0);
                if std::mem::replace(&mut used_tiles[index], true) {
                    return Err(BridgeError::DuplicateTile(placed.tile.id.0));
                }
            }
        }
    }
    let mut market_pairs = [None; 4];
    for slot in v2::MarketSlot::ALL {
        let index = slot.index();
        let tile = public.market().tiles[index].ok_or(BridgeError::IncompleteMarket(index))?;
        let wildlife =
            public.market().wildlife[index].ok_or(BridgeError::IncompleteMarket(index))?;
        if tile.id.0 < 85 {
            let tile_index = usize::from(tile.id.0);
            if std::mem::replace(&mut used_tiles[tile_index], true) {
                return Err(BridgeError::DuplicateTile(tile.id.0));
            }
        }
        market_pairs[index] = Some(v1::market::MarketPair {
            tile: translate_tile(tile),
            wildlife: translate_wildlife(wildlife),
        });
    }
    let mut unseen_tiles: Vec<_> = v2::STANDARD_TILES
        .iter()
        .copied()
        .filter(|tile| !used_tiles[usize::from(tile.id.0)])
        .collect();
    if used_tiles.iter().filter(|used| **used).count() + unseen_tiles.len() != 85 {
        return Err(BridgeError::TileConservation(
            used_tiles.iter().filter(|used| **used).count() + unseen_tiles.len(),
        ));
    }
    unseen_tiles.shuffle(&mut inventory_rng);

    let mut visible_wildlife = [0u8; 5];
    for board in public.boards() {
        for (_, placed) in board.placed_tiles() {
            if let Some(wildlife) = placed.wildlife {
                visible_wildlife[wildlife as usize] += 1;
            }
        }
    }
    for wildlife in public.market().wildlife.iter().flatten() {
        visible_wildlife[*wildlife as usize] += 1;
    }
    let mut unseen_wildlife = Vec::new();
    for wildlife in v2::Wildlife::ALL {
        let visible = visible_wildlife[wildlife as usize];
        if visible > 20 {
            return Err(BridgeError::WildlifeConservation { wildlife, visible });
        }
        unseen_wildlife.extend(std::iter::repeat_n(wildlife, usize::from(20 - visible)));
    }
    unseen_wildlife.shuffle(&mut inventory_rng);

    let mut initialization_rng =
        rand::rngs::StdRng::from_seed(domain_hash(b"legacy-teacher-shell-state", &public_digest));
    let mut game =
        v1::game::GameState::new(4, v1::types::ScoringCards::all_a(), &mut initialization_rng);
    game.boards = boards;
    game.market = v1::market::Market {
        pairs: market_pairs,
    };
    while game.tile_bag.draw().is_some() {}
    for tile in &unseen_tiles {
        game.tile_bag.return_tile(translate_tile(*tile));
    }
    while game.wildlife_bag.draw().is_some() {}
    for wildlife in &unseen_wildlife {
        game.wildlife_bag
            .return_token(translate_wildlife(*wildlife));
    }
    game.current_player = public.current_player();
    game.turns_remaining = u8::try_from(public.total_turns() - public.completed_turns())
        .expect("four-player Cascadia has at most 80 remaining turns");
    game.num_players = 4;
    game.overflow_used_this_turn = false;

    let mut inventory_hasher = Hasher::new();
    inventory_hasher.update(b"legacy-teacher-reconstructed-inventory");
    for tile in &unseen_tiles {
        inventory_hasher.update(&[tile.id.0]);
    }
    for wildlife in &unseen_wildlife {
        inventory_hasher.update(&[*wildlife as u8]);
    }
    let evidence = TranslationEvidence {
        public_state_blake3: blake3::Hash::from_bytes(public_digest).to_hex().to_string(),
        reconstructed_inventory_blake3: inventory_hasher.finalize().to_hex().to_string(),
        unseen_tiles: unseen_tiles.len(),
        unseen_wildlife: unseen_wildlife.len(),
        checked_boards: public.boards().len(),
        maximum_absolute_coordinate,
        elk_score_mismatch_boards,
        elk_v2_minus_v1_total,
        elk_v2_minus_v1_max,
    };
    Ok(LegacyTranslation {
        game,
        evidence,
        coordinate_frames: coordinate_frames
            .try_into()
            .expect("four-player validation fixes the coordinate-frame count"),
    })
}

pub fn map_legacy_action(
    original: &v2::GameState,
    prelude: &v2::MarketPrelude,
    candidate: &ScoredMove,
) -> Result<v2::TurnAction, BridgeError> {
    map_legacy_action_in_frame(original, prelude, candidate, LegacyCoordinateFrame::ZERO)
}

fn map_legacy_action_in_frame(
    original: &v2::GameState,
    prelude: &v2::MarketPrelude,
    candidate: &ScoredMove,
    coordinate_frame: LegacyCoordinateFrame,
) -> Result<v2::TurnAction, BridgeError> {
    let tile_slot = v2::MarketSlot::new(
        u8::try_from(candidate.market_index).map_err(|_| BridgeError::InvalidMarketSlot)?,
    )
    .ok_or(BridgeError::InvalidMarketSlot)?;
    let draft = if let Some(wildlife_index) = candidate.wildlife_market_index {
        let wildlife_slot = v2::MarketSlot::new(
            u8::try_from(wildlife_index).map_err(|_| BridgeError::InvalidMarketSlot)?,
        )
        .ok_or(BridgeError::InvalidMarketSlot)?;
        v2::DraftChoice::Independent {
            tile_slot,
            wildlife_slot,
        }
    } else {
        v2::DraftChoice::Paired { slot: tile_slot }
    };
    let wildlife = match (candidate.wildlife_q, candidate.wildlife_r) {
        (Some(q), Some(r)) => Some(coordinate_frame.to_v2(q, r)?),
        (None, None) => None,
        _ => return Err(BridgeError::PartialWildlifeCoordinate),
    };
    let action = v2::TurnAction {
        replace_three_of_a_kind: prelude.replace_three_of_a_kind,
        wildlife_wipes: prelude.wildlife_wipes.clone(),
        draft,
        tile: v2::TilePlacement {
            coord: coordinate_frame.to_v2(candidate.tile_q, candidate.tile_r)?,
            rotation: v2::Rotation::new(candidate.rotation).ok_or(
                BridgeError::IllegalMappedAction(
                    "legacy rotation is outside 0 through 5".to_owned(),
                ),
            )?,
        },
        wildlife,
    };
    original.transition(&action).map_err(|error| {
        BridgeError::IllegalMappedAction(format!("{error}; legacy candidate={candidate:?}"))
    })?;
    Ok(action)
}

fn legacy_move_identity(
    movement: &ScoredMove,
) -> (usize, Option<usize>, i8, i8, u8, Option<i8>, Option<i8>) {
    (
        movement.market_index,
        movement.wildlife_market_index,
        movement.tile_q,
        movement.tile_r,
        movement.rotation,
        movement.wildlife_q,
        movement.wildlife_r,
    )
}

#[cfg(test)]
fn canonical_candidate_to_legacy(candidate: &cascadia_sim::GreedyCandidate) -> ScoredMove {
    canonical_candidate_to_legacy_in_frame(candidate, LegacyCoordinateFrame::ZERO)
}

fn canonical_candidate_to_legacy_in_frame(
    candidate: &cascadia_sim::GreedyCandidate,
    coordinate_frame: LegacyCoordinateFrame,
) -> ScoredMove {
    let (market_index, wildlife_market_index) = match candidate.action.draft {
        v2::DraftChoice::Paired { slot } => (slot.index(), None),
        v2::DraftChoice::Independent {
            tile_slot,
            wildlife_slot,
        } => (tile_slot.index(), Some(wildlife_slot.index())),
    };
    ScoredMove {
        market_index,
        tile_q: candidate.action.tile.coord.q + coordinate_frame.q_offset,
        tile_r: candidate.action.tile.coord.r + coordinate_frame.r_offset,
        rotation: candidate.action.tile.rotation.get(),
        wildlife_q: candidate
            .action
            .wildlife
            .map(|coord| coord.q + coordinate_frame.q_offset),
        wildlife_r: candidate
            .action
            .wildlife
            .map(|coord| coord.r + coordinate_frame.r_offset),
        score: candidate.resulting_base_score,
        eval: i32::from(candidate.resulting_base_score) * 1000,
        wildlife_market_index,
    }
}

pub struct LegacyTeacher {
    net: NNUENetwork,
    rollouts: usize,
    filter_invalid_root: bool,
    allow_legacy_elk_undercount: bool,
    probe_pattern_frontier: bool,
    pub diagnostics: BridgeDiagnostics,
}

#[derive(Debug, Clone)]
pub struct LegacyActionEstimate {
    pub action: v2::TurnAction,
    pub direct_raw_units: i32,
    pub rollout_mean: f64,
    pub rollout_stddev: f64,
    pub samples: u32,
}

#[derive(Debug, Clone)]
pub struct LegacyTeacherDecision {
    pub selected: v2::TurnAction,
    pub estimates: Vec<LegacyActionEstimate>,
}

struct PreparedLegacyDecision {
    prelude: v2::MarketPrelude,
    translation: LegacyTranslation,
    candidates: Vec<ScoredMove>,
}

struct PreparedExpandedDecision {
    prelude: v2::MarketPrelude,
    translation: LegacyTranslation,
    expanded: Vec<ScoredMove>,
    canonical: Vec<ScoredMove>,
}

fn prepare_expanded_decision(
    game: &v2::GameState,
    diagnostics: &mut BridgeDiagnostics,
    allow_legacy_elk_undercount: bool,
) -> Result<PreparedExpandedDecision, BridgeError> {
    let prelude = canonical_prelude(game);
    let staged = game
        .preview_market_prelude(&prelude)
        .map_err(|error| BridgeError::Rules(error.to_string()))?;
    let public = staged.public_state();
    let translation = if allow_legacy_elk_undercount {
        translate_public_state_allowing_legacy_elk_undercount(&public)?
    } else {
        translate_public_state(&public)?
    };
    diagnostics.record_translation(&translation.evidence);
    let coordinate_frame = translation.current_player_frame();

    let expanded = expanded_candidates(&translation.game);
    diagnostics.expanded_candidates += expanded.len();
    let mut canonical = Vec::with_capacity(expanded.len());
    for candidate in &expanded {
        match map_legacy_action_in_frame(game, &prelude, candidate, coordinate_frame) {
            Ok(_) => {
                diagnostics.expanded_candidates_legal += 1;
                canonical.push(*candidate);
            }
            Err(error) => {
                diagnostics.expanded_candidates_illegal += 1;
                diagnostics.record_malformed_expanded(&error);
            }
        }
    }
    Ok(PreparedExpandedDecision {
        prelude,
        translation,
        expanded,
        canonical,
    })
}

fn record_prefiltered_candidates(
    game: &v2::GameState,
    prelude: &v2::MarketPrelude,
    candidates: &[ScoredMove],
    coordinate_frame: LegacyCoordinateFrame,
    diagnostics: &mut BridgeDiagnostics,
) -> Result<(), BridgeError> {
    diagnostics.prefiltered_candidates += candidates.len();
    for candidate in candidates {
        match map_legacy_action_in_frame(game, prelude, candidate, coordinate_frame) {
            Ok(_) => diagnostics.prefiltered_candidates_legal += 1,
            Err(error) => {
                diagnostics.prefiltered_candidates_illegal += 1;
                return Err(error);
            }
        }
    }
    Ok(())
}

impl LegacyTeacher {
    pub fn new(net: NNUENetwork, rollouts: usize) -> Result<Self, BridgeError> {
        validate_legacy_environment()?;
        if rollouts == 0 {
            return Err(BridgeError::LegacyEnvironment(
                "rollouts must be positive".to_owned(),
            ));
        }
        Ok(Self {
            net,
            rollouts,
            filter_invalid_root: false,
            allow_legacy_elk_undercount: false,
            probe_pattern_frontier: false,
            diagnostics: BridgeDiagnostics::default(),
        })
    }

    pub fn new_filtered(net: NNUENetwork, rollouts: usize) -> Result<Self, BridgeError> {
        let mut teacher = Self::new(net, rollouts)?;
        teacher.filter_invalid_root = true;
        Ok(teacher)
    }

    pub fn new_heuristic(net: NNUENetwork, rollouts: usize) -> Result<Self, BridgeError> {
        let mut teacher = Self::new_filtered(net, rollouts)?;
        teacher.allow_legacy_elk_undercount = true;
        Ok(teacher)
    }

    pub fn new_heuristic_with_pattern_frontier_probe(
        net: NNUENetwork,
        rollouts: usize,
    ) -> Result<Self, BridgeError> {
        let mut teacher = Self::new_heuristic(net, rollouts)?;
        teacher.probe_pattern_frontier = true;
        Ok(teacher)
    }

    pub fn select_action(&mut self, game: &v2::GameState) -> Result<v2::TurnAction, BridgeError> {
        self.diagnostics.states_attempted += 1;
        let result = self.select_action_inner(game);
        if let Err(error) = &result {
            self.diagnostics.record_error(error);
        }
        result
    }

    pub fn select_action_with_estimates(
        &mut self,
        game: &v2::GameState,
    ) -> Result<LegacyTeacherDecision, BridgeError> {
        self.diagnostics.states_attempted += 1;
        let result = self.select_action_with_estimates_inner(game, false);
        if let Err(error) = &result {
            self.diagnostics.record_error(error);
        }
        result
    }

    pub fn select_action_with_exact_budget_estimates(
        &mut self,
        game: &v2::GameState,
    ) -> Result<LegacyTeacherDecision, BridgeError> {
        self.diagnostics.states_attempted += 1;
        let result = self.select_action_with_estimates_inner(game, true);
        if let Err(error) = &result {
            self.diagnostics.record_error(error);
        }
        result
    }

    fn select_action_inner(&mut self, game: &v2::GameState) -> Result<v2::TurnAction, BridgeError> {
        let prepared = self.prepare_decision(game)?;
        let mut search_rng = legacy_search_rng(&prepared.translation.evidence.public_state_blake3);
        let selected = best_move_nnue_rollout_mce(
            &prepared.translation.game,
            &self.net,
            self.rollouts,
            GreedyMceAlloc::SeqHalving,
            prepared.candidates,
            &mut search_rng,
        )
        .ok_or(BridgeError::NoCandidate)?;
        self.diagnostics.selected_actions += 1;
        let action = map_legacy_action_in_frame(
            game,
            &prepared.prelude,
            &selected,
            prepared.translation.current_player_frame(),
        )?;
        self.diagnostics.selected_actions_legal += 1;
        if self.probe_pattern_frontier {
            let pattern_frontier =
                rank_pattern_actions(game, &prepared.prelude, PatternAwareConfig::default())
                    .map_err(|error| BridgeError::Pattern(error.to_string()))?;
            self.diagnostics
                .record_pattern_frontier_recall(game, &action, &pattern_frontier);
        }
        Ok(action)
    }

    fn select_action_with_estimates_inner(
        &mut self,
        game: &v2::GameState,
        exact_budget: bool,
    ) -> Result<LegacyTeacherDecision, BridgeError> {
        let prepared = self.prepare_decision(game)?;
        let direct_candidates = prepared.candidates.clone();
        let mut search_rng = legacy_search_rng(&prepared.translation.evidence.public_state_blake3);
        let estimates = if exact_budget {
            score_nnue_rollout_mce_seq_halving_exact(
                &prepared.translation.game,
                &self.net,
                self.rollouts,
                prepared.candidates,
                &mut search_rng,
            )
        } else {
            score_nnue_rollout_mce_seq_halving(
                &prepared.translation.game,
                &self.net,
                self.rollouts,
                prepared.candidates,
                &mut search_rng,
            )
        };
        let selected = estimates.first().ok_or(BridgeError::NoCandidate)?.movement;
        let coordinate_frame = prepared.translation.current_player_frame();
        let action =
            map_legacy_action_in_frame(game, &prepared.prelude, &selected, coordinate_frame)?;
        let mapped = estimates
            .into_iter()
            .map(|estimate| {
                let direct_raw_units = direct_candidates
                    .iter()
                    .find(|candidate| **candidate == estimate.movement)
                    .map(|candidate| candidate.eval)
                    .ok_or(BridgeError::NoCandidate)?;
                Ok(LegacyActionEstimate {
                    action: map_legacy_action_in_frame(
                        game,
                        &prepared.prelude,
                        &estimate.movement,
                        coordinate_frame,
                    )?,
                    direct_raw_units,
                    rollout_mean: estimate.rollout_mean,
                    rollout_stddev: estimate.rollout_stddev,
                    samples: estimate.samples,
                })
            })
            .collect::<Result<Vec<_>, BridgeError>>()?;
        self.diagnostics.selected_actions += 1;
        self.diagnostics.selected_actions_legal += 1;
        Ok(LegacyTeacherDecision {
            selected: action,
            estimates: mapped,
        })
    }

    fn prepare_decision(
        &mut self,
        game: &v2::GameState,
    ) -> Result<PreparedLegacyDecision, BridgeError> {
        let prepared = prepare_expanded_decision(
            game,
            &mut self.diagnostics,
            self.allow_legacy_elk_undercount,
        )?;
        let mut candidates = if self.filter_invalid_root {
            prepared.canonical
        } else {
            prepared.expanded
        };
        if candidates.is_empty() {
            return Err(BridgeError::NoCandidate);
        }
        if candidates.len() > 32 {
            candidates =
                nnue_prefilter_candidates(&prepared.translation.game, &self.net, candidates, 32);
        }
        record_prefiltered_candidates(
            game,
            &prepared.prelude,
            &candidates,
            prepared.translation.current_player_frame(),
            &mut self.diagnostics,
        )?;
        Ok(PreparedLegacyDecision {
            prelude: prepared.prelude,
            translation: prepared.translation,
            candidates,
        })
    }
}

/// Frozen V1 NNUE argmax projected onto the canonical V2 legal-action domain.
/// This is intentionally distinct from the K32/R600 teacher: bootstrap games
/// need a fast direct policy, while evaluation retains the qualified search.
pub struct LegacyDirectPolicy {
    net: NNUENetwork,
    diagnostics: BridgeDiagnostics,
}

impl LegacyDirectPolicy {
    pub fn new(net: NNUENetwork) -> Result<Self, BridgeError> {
        validate_legacy_environment()?;
        Ok(Self {
            net,
            diagnostics: BridgeDiagnostics::default(),
        })
    }

    pub fn select_action(&mut self, game: &v2::GameState) -> Result<v2::TurnAction, BridgeError> {
        self.diagnostics.states_attempted += 1;
        let result = self.select_action_inner(game);
        if let Err(error) = &result {
            self.diagnostics.record_error(error);
        }
        result
    }

    fn select_action_inner(&mut self, game: &v2::GameState) -> Result<v2::TurnAction, BridgeError> {
        let prepared = prepare_expanded_decision(game, &mut self.diagnostics, true)?;
        let ranked =
            rank_candidates_nnue_direct(&prepared.translation.game, &self.net, prepared.canonical);
        let selected = ranked.first().ok_or(BridgeError::NoCandidate)?;
        let action = map_legacy_action_in_frame(
            game,
            &prepared.prelude,
            selected,
            prepared.translation.current_player_frame(),
        )?;
        self.diagnostics.selected_actions += 1;
        self.diagnostics.selected_actions_legal += 1;
        Ok(action)
    }

    pub fn diagnostics(&self) -> &BridgeDiagnostics {
        &self.diagnostics
    }
}

struct ExactMlxEvaluator {
    process: ModelProcess,
}

impl SparseNnueEvaluator for ExactMlxEvaluator {
    type Error = ModelError;

    fn evaluate_sparse(&mut self, feature_sets: &[Vec<u16>]) -> Result<Vec<f32>, Self::Error> {
        self.process.predict_sparse_nnue_csr_exact(feature_sets)
    }

    fn rollout_pipeline_chunk_states(&self) -> Option<usize> {
        Some(exact_mlx_pipeline_chunk_states())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExactMlxMultiplexDiagnostics {
    pub search_cohorts: u64,
    pub searches: u64,
    pub evaluator_requests: u64,
    pub evaluator_batches: u64,
    pub coalesced_batches: u64,
    pub evaluator_rows: u64,
    pub cross_request_rows_observed: u64,
    pub cross_request_duplicate_rows: u64,
    pub maximum_requests_per_batch: usize,
    pub maximum_rows_per_batch: usize,
}

impl ExactMlxMultiplexDiagnostics {
    fn merge_from(&mut self, source: Self) {
        self.search_cohorts = self.search_cohorts.saturating_add(source.search_cohorts);
        self.searches = self.searches.saturating_add(source.searches);
        self.evaluator_requests = self
            .evaluator_requests
            .saturating_add(source.evaluator_requests);
        self.evaluator_batches = self
            .evaluator_batches
            .saturating_add(source.evaluator_batches);
        self.coalesced_batches = self
            .coalesced_batches
            .saturating_add(source.coalesced_batches);
        self.evaluator_rows = self.evaluator_rows.saturating_add(source.evaluator_rows);
        self.cross_request_rows_observed = self
            .cross_request_rows_observed
            .saturating_add(source.cross_request_rows_observed);
        self.cross_request_duplicate_rows = self
            .cross_request_duplicate_rows
            .saturating_add(source.cross_request_duplicate_rows);
        self.maximum_requests_per_batch = self
            .maximum_requests_per_batch
            .max(source.maximum_requests_per_batch);
        self.maximum_rows_per_batch = self
            .maximum_rows_per_batch
            .max(source.maximum_rows_per_batch);
    }
}

#[derive(Debug, Error)]
#[error("{0}")]
struct MultiplexEvaluationError(String);

struct MultiplexEvaluationRequest {
    search_index: usize,
    rows: Vec<Vec<u16>>,
    response: SyncSender<Result<Vec<f32>, MultiplexEvaluationError>>,
}

struct MultiplexSearchOutput {
    result: Result<ExactMlxCollectedDecision, BridgeError>,
    diagnostics: BridgeDiagnostics,
    batch_diagnostics: BatchedNnueDiagnostics,
}

enum MultiplexEvent {
    Evaluate(MultiplexEvaluationRequest),
    Finished {
        search_index: usize,
        output: MultiplexSearchOutput,
    },
}

struct MultiplexEvaluatorProxy {
    search_index: usize,
    events: mpsc::Sender<MultiplexEvent>,
}

impl SparseNnueEvaluator for MultiplexEvaluatorProxy {
    type Error = MultiplexEvaluationError;

    fn evaluate_sparse(&mut self, feature_sets: &[Vec<u16>]) -> Result<Vec<f32>, Self::Error> {
        self.evaluate_sparse_owned(feature_sets.to_vec())
    }

    fn evaluate_sparse_owned(
        &mut self,
        feature_sets: Vec<Vec<u16>>,
    ) -> Result<Vec<f32>, Self::Error> {
        let (response, result) = mpsc::sync_channel(1);
        self.events
            .send(MultiplexEvent::Evaluate(MultiplexEvaluationRequest {
                search_index: self.search_index,
                rows: feature_sets,
                response,
            }))
            .map_err(|_| {
                MultiplexEvaluationError(
                    "multiplexed exact evaluator request channel disconnected".to_owned(),
                )
            })?;
        result.recv().map_err(|_| {
            MultiplexEvaluationError(
                "multiplexed exact evaluator response channel disconnected".to_owned(),
            )
        })?
    }

    fn rollout_pipeline_chunk_states(&self) -> Option<usize> {
        Some(exact_mlx_pipeline_chunk_states())
    }
}

pub struct ExactMlxLegacyTeacher {
    evaluator: ExactMlxEvaluator,
    leaf_evaluator: Option<ExactMlxEvaluator>,
    rollouts: usize,
    candidate_limit: usize,
    habitat_candidate_limit: usize,
    seed_coupling: RolloutSeedCoupling,
    rollout_config: BatchedRolloutConfig,
    pub diagnostics: BridgeDiagnostics,
    pub batch_diagnostics: BatchedNnueDiagnostics,
    pub multiplex_diagnostics: ExactMlxMultiplexDiagnostics,
}

/// Native evaluator for the same canonical K32/R600 rollout pipeline used by
/// [`ExactMlxLegacyTeacher`]. The frozen rollout-wave parity artifact proves
/// this evaluator and the qualified exact-MLX port select the same actions.
pub struct ExactRustLegacyTeacher {
    evaluator: NNUENetwork,
    rollouts: usize,
    pub diagnostics: BridgeDiagnostics,
    pub batch_diagnostics: BatchedNnueDiagnostics,
}

#[derive(Debug, Clone)]
pub struct ExactMlxRootEstimate {
    pub action: v2::TurnAction,
    pub candidate_index: usize,
    pub features: Vec<u16>,
    pub immediate_score: f32,
    pub rollout_mean: f64,
    pub rollout_stddev: f64,
    pub samples: u32,
    pub selected: bool,
}

#[derive(Debug, Clone)]
pub struct ExactMlxActionPrior {
    pub immediate_score: f32,
    pub remaining_value: f32,
}

#[derive(Debug, Clone)]
pub struct ExactMlxActionHidden {
    pub immediate_score: f32,
    pub remaining_value: f32,
    pub hidden: [f32; 64],
}

#[derive(Debug, Clone)]
pub struct ExactMlxCollectedDecision {
    pub action: v2::TurnAction,
    pub rollout_value_samples: Vec<RolloutValueSample>,
    pub root_estimates: Vec<ExactMlxRootEstimate>,
}

#[derive(Debug, Clone, Copy)]
enum ExactMlxCollection {
    None,
    Estimates,
    Samples(u64),
}

#[derive(Debug, Clone, Copy)]
struct ExactMlxSearchConfig {
    rollouts: usize,
    candidate_limit: usize,
    habitat_candidate_limit: usize,
    seed_coupling: RolloutSeedCoupling,
    rollout_config: BatchedRolloutConfig,
}

struct PreparedExactActions {
    translation: LegacyTranslation,
    candidates: Vec<ScoredMove>,
    afterstates: Vec<SparseNnueAfterstate>,
}

impl ExactMlxLegacyTeacher {
    pub fn new(process: ModelProcess, rollouts: usize) -> Result<Self, BridgeError> {
        Self::new_with_candidate_configuration(
            process,
            rollouts,
            32,
            0,
            RolloutSeedCoupling::Independent,
            BatchedRolloutConfig::full(),
        )
    }

    pub fn new_with_rollout_turn_limit(
        process: ModelProcess,
        rollouts: usize,
        max_focal_turns: usize,
    ) -> Result<Self, BridgeError> {
        let rollout_config = BatchedRolloutConfig::truncated(max_focal_turns)
            .map_err(|error| BridgeError::LegacyEnvironment(error.to_owned()))?;
        Self::new_with_candidate_configuration(
            process,
            rollouts,
            32,
            0,
            RolloutSeedCoupling::Independent,
            rollout_config,
        )
    }

    pub fn new_with_afterstate_rollout_turn_limit(
        process: ModelProcess,
        rollouts: usize,
        max_focal_turns: usize,
    ) -> Result<Self, BridgeError> {
        let rollout_config = BatchedRolloutConfig::truncated_afterstate(max_focal_turns)
            .map_err(|error| BridgeError::LegacyEnvironment(error.to_owned()))?;
        Self::new_with_candidate_configuration(
            process,
            rollouts,
            32,
            0,
            RolloutSeedCoupling::Independent,
            rollout_config,
        )
    }

    pub fn new_with_leaf_rollout_turn_limit(
        process: ModelProcess,
        leaf_process: ModelProcess,
        rollouts: usize,
        max_focal_turns: usize,
        leaf_timing: BatchedRolloutLeafTiming,
    ) -> Result<Self, BridgeError> {
        let rollout_config = match leaf_timing {
            BatchedRolloutLeafTiming::AfterOpponentRound => {
                BatchedRolloutConfig::truncated(max_focal_turns)
            }
            BatchedRolloutLeafTiming::AfterFocalMove => {
                BatchedRolloutConfig::truncated_afterstate(max_focal_turns)
            }
        }
        .map_err(|error| BridgeError::LegacyEnvironment(error.to_owned()))?;
        let mut teacher = Self::new_with_candidate_configuration(
            process,
            rollouts,
            32,
            0,
            RolloutSeedCoupling::Independent,
            rollout_config,
        )?;
        teacher.leaf_evaluator = Some(ExactMlxEvaluator {
            process: leaf_process,
        });
        Ok(teacher)
    }

    pub fn new_with_seed_coupling(
        process: ModelProcess,
        rollouts: usize,
        seed_coupling: RolloutSeedCoupling,
    ) -> Result<Self, BridgeError> {
        Self::new_with_candidate_configuration(
            process,
            rollouts,
            32,
            0,
            seed_coupling,
            BatchedRolloutConfig::full(),
        )
    }

    pub fn new_with_candidate_limit(
        process: ModelProcess,
        rollouts: usize,
        candidate_limit: usize,
    ) -> Result<Self, BridgeError> {
        Self::new_with_candidate_configuration(
            process,
            rollouts,
            candidate_limit,
            0,
            RolloutSeedCoupling::Independent,
            BatchedRolloutConfig::full(),
        )
    }

    pub fn new_with_habitat_candidates(
        process: ModelProcess,
        rollouts: usize,
        habitat_candidate_limit: usize,
    ) -> Result<Self, BridgeError> {
        Self::new_with_candidate_configuration(
            process,
            rollouts,
            32,
            habitat_candidate_limit,
            RolloutSeedCoupling::Independent,
            BatchedRolloutConfig::full(),
        )
    }

    fn new_with_candidate_configuration(
        process: ModelProcess,
        rollouts: usize,
        candidate_limit: usize,
        habitat_candidate_limit: usize,
        seed_coupling: RolloutSeedCoupling,
        rollout_config: BatchedRolloutConfig,
    ) -> Result<Self, BridgeError> {
        validate_legacy_environment()?;
        if rollouts == 0 {
            return Err(BridgeError::LegacyEnvironment(
                "rollouts must be positive".to_owned(),
            ));
        }
        if candidate_limit < 32 {
            return Err(BridgeError::LegacyEnvironment(
                "exact MLX candidate limit must be at least 32".to_owned(),
            ));
        }
        Ok(Self {
            evaluator: ExactMlxEvaluator { process },
            leaf_evaluator: None,
            rollouts,
            candidate_limit,
            habitat_candidate_limit,
            seed_coupling,
            rollout_config,
            diagnostics: BridgeDiagnostics::default(),
            batch_diagnostics: BatchedNnueDiagnostics::default(),
            multiplex_diagnostics: ExactMlxMultiplexDiagnostics::default(),
        })
    }

    pub fn select_action(&mut self, game: &v2::GameState) -> Result<v2::TurnAction, BridgeError> {
        self.diagnostics.states_attempted += 1;
        let result = self
            .select_action_inner(game, ExactMlxCollection::None)
            .map(|decision| decision.action);
        if let Err(error) = &result {
            self.diagnostics.record_error(error);
        }
        result
    }

    pub fn select_actions(
        &mut self,
        games: &[&v2::GameState],
    ) -> Result<Vec<v2::TurnAction>, BridgeError> {
        if games.is_empty() {
            return Ok(Vec::new());
        }
        if games.len() == 1 {
            return self.select_action(games[0]).map(|action| vec![action]);
        }
        if self.leaf_evaluator.is_some() {
            return Err(BridgeError::LegacyEnvironment(
                "multiplexed exact search does not support a separate leaf evaluator".to_owned(),
            ));
        }

        let config = ExactMlxSearchConfig {
            rollouts: self.rollouts,
            candidate_limit: self.candidate_limit,
            habitat_candidate_limit: self.habitat_candidate_limit,
            seed_coupling: self.seed_coupling,
            rollout_config: self.rollout_config,
        };
        let (outputs, multiplex_diagnostics) =
            run_multiplexed_searches(&mut self.evaluator, games, config, ExactMlxCollection::None)?;
        self.multiplex_diagnostics.merge_from(multiplex_diagnostics);

        let mut actions = Vec::with_capacity(outputs.len());
        let mut first_error = None;
        for output in outputs {
            self.diagnostics.merge_from(output.diagnostics);
            self.batch_diagnostics.merge_from(output.batch_diagnostics);
            match output.result {
                Ok(decision) => actions.push(decision.action),
                Err(error) => {
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
            }
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        Ok(actions)
    }

    pub fn select_action_with_estimates(
        &mut self,
        game: &v2::GameState,
    ) -> Result<ExactMlxCollectedDecision, BridgeError> {
        self.diagnostics.states_attempted += 1;
        let result = self.select_action_inner(game, ExactMlxCollection::Estimates);
        if let Err(error) = &result {
            self.diagnostics.record_error(error);
        }
        result
    }

    pub fn select_action_collecting_rollout_values(
        &mut self,
        game: &v2::GameState,
        trace_modulus: u64,
    ) -> Result<ExactMlxCollectedDecision, BridgeError> {
        self.diagnostics.states_attempted += 1;
        let result = self.select_action_inner(game, ExactMlxCollection::Samples(trace_modulus));
        if let Err(error) = &result {
            self.diagnostics.record_error(error);
        }
        result
    }

    pub fn score_action_priors(
        &mut self,
        game: &v2::GameState,
        actions: &[v2::TurnAction],
    ) -> Result<Vec<ExactMlxActionPrior>, BridgeError> {
        let prepared = self.prepare_exact_actions(game, actions)?;
        let feature_sets = prepared
            .afterstates
            .iter()
            .map(|afterstate| afterstate.features.clone())
            .collect::<Vec<_>>();
        let mut remaining = Vec::with_capacity(feature_sets.len());
        for chunk in feature_sets.chunks(EXACT_MLX_STATIC_SCREEN_CHUNK_ROWS) {
            remaining.extend(
                evaluate_sparse_rows_deduplicated(
                    &mut self.evaluator,
                    chunk,
                    &mut self.batch_diagnostics,
                )
                .map_err(|error| BridgeError::ExactMlx(error.to_string()))?,
            );
        }
        if remaining.len() != prepared.afterstates.len() {
            return Err(BridgeError::ExactMlx(
                "candidate-prior evaluator returned the wrong row count".to_owned(),
            ));
        }
        Ok(prepared
            .afterstates
            .into_iter()
            .zip(remaining)
            .map(|(afterstate, remaining_value)| ExactMlxActionPrior {
                immediate_score: afterstate.immediate_score,
                remaining_value,
            })
            .collect())
    }

    pub fn score_action_hidden(
        &mut self,
        game: &v2::GameState,
        actions: &[v2::TurnAction],
    ) -> Result<Vec<ExactMlxActionHidden>, BridgeError> {
        let prepared = self.prepare_exact_actions(game, actions)?;
        let feature_sets = prepared
            .afterstates
            .iter()
            .map(|afterstate| afterstate.features.clone())
            .collect::<Vec<_>>();
        let mut predictions = Vec::with_capacity(feature_sets.len());
        for chunk in feature_sets.chunks(EXACT_MLX_STATIC_SCREEN_CHUNK_ROWS) {
            predictions.extend(
                self.evaluator
                    .process
                    .predict_sparse_nnue_csr_exact_hidden(chunk)
                    .map_err(|error| BridgeError::ExactMlx(error.to_string()))?,
            );
            self.batch_diagnostics.record_batch(chunk.len());
        }
        if predictions.len() != prepared.afterstates.len() {
            return Err(BridgeError::ExactMlx(
                "candidate-hidden evaluator returned the wrong row count".to_owned(),
            ));
        }
        Ok(prepared
            .afterstates
            .into_iter()
            .zip(predictions)
            .map(
                |(
                    afterstate,
                    ExactNnueHiddenPrediction {
                        hidden,
                        value: remaining_value,
                    },
                )| ExactMlxActionHidden {
                    immediate_score: afterstate.immediate_score,
                    remaining_value,
                    hidden,
                },
            )
            .collect())
    }

    pub fn score_actions_with_rollouts(
        &mut self,
        game: &v2::GameState,
        actions: &[v2::TurnAction],
        rollouts: usize,
    ) -> Result<Vec<ExactMlxRootEstimate>, BridgeError> {
        self.score_actions_with_rollouts_in_domain(game, actions, rollouts, &[])
    }

    pub fn score_actions_with_rollouts_in_domain(
        &mut self,
        game: &v2::GameState,
        actions: &[v2::TurnAction],
        rollouts: usize,
        seed_domain: &[u8],
    ) -> Result<Vec<ExactMlxRootEstimate>, BridgeError> {
        self.score_actions_with_rollouts_in_domain_and_coupling(
            game,
            actions,
            rollouts,
            seed_domain,
            self.seed_coupling,
        )
    }

    pub fn score_actions_with_rollouts_in_domain_and_coupling(
        &mut self,
        game: &v2::GameState,
        actions: &[v2::TurnAction],
        rollouts: usize,
        seed_domain: &[u8],
        seed_coupling: RolloutSeedCoupling,
    ) -> Result<Vec<ExactMlxRootEstimate>, BridgeError> {
        if rollouts == 0 {
            return Err(BridgeError::LegacyEnvironment(
                "action-evaluation rollouts must be positive".to_owned(),
            ));
        }
        let prepared = self.prepare_exact_actions(game, actions)?;
        let mut search_rng = if seed_domain.is_empty() {
            legacy_search_rng(&prepared.translation.evidence.public_state_blake3)
        } else {
            legacy_search_rng_in_domain(
                &prepared.translation.evidence.public_state_blake3,
                seed_domain,
            )
        };
        let estimates = score_nnue_rollout_mce_seq_halving_batched_with_config_and_coupling(
            &prepared.translation.game,
            &mut self.evaluator,
            rollouts,
            prepared.candidates.clone(),
            &mut search_rng,
            &mut self.batch_diagnostics,
            self.rollout_config,
            seed_coupling,
        )
        .map_err(|error| BridgeError::ExactMlx(error.to_string()))?;
        map_exact_mlx_estimates(
            game,
            actions,
            &prepared.candidates,
            &prepared.afterstates,
            estimates,
            prepared.translation.current_player_frame(),
        )
    }

    fn prepare_exact_actions(
        &mut self,
        game: &v2::GameState,
        actions: &[v2::TurnAction],
    ) -> Result<PreparedExactActions, BridgeError> {
        if actions.is_empty() {
            return Err(BridgeError::NoCandidate);
        }
        self.diagnostics.states_attempted += 1;
        let prelude = actions[0].prelude();
        if actions.iter().any(|action| action.prelude() != prelude) {
            return Err(BridgeError::Rules(
                "candidate-prior actions do not share one market prelude".to_owned(),
            ));
        }
        let staged = game
            .preview_market_prelude(&prelude)
            .map_err(|error| BridgeError::Rules(error.to_string()))?;
        let translation =
            translate_public_state_allowing_legacy_elk_undercount(&staged.public_state())?;
        self.diagnostics.record_translation(&translation.evidence);
        let coordinate_frame = translation.current_player_frame();
        let candidates = actions
            .iter()
            .map(|action| canonical_action_to_legacy_in_frame(game, action, coordinate_frame))
            .collect::<Result<Vec<_>, _>>()?;
        let afterstates = prepare_sparse_nnue_afterstates(&translation.game, &candidates);
        if afterstates.len() != actions.len() {
            return Err(BridgeError::ExactMlx(
                "one or more canonical candidate priors could not construct an afterstate"
                    .to_owned(),
            ));
        }
        if afterstates
            .iter()
            .zip(&candidates)
            .any(|(afterstate, candidate)| {
                legacy_move_identity(&afterstate.movement) != legacy_move_identity(candidate)
            })
        {
            return Err(BridgeError::ExactMlx(
                "candidate-prior afterstates changed canonical action order".to_owned(),
            ));
        }
        Ok(PreparedExactActions {
            translation,
            candidates,
            afterstates,
        })
    }

    fn select_action_inner(
        &mut self,
        game: &v2::GameState,
        collection: ExactMlxCollection,
    ) -> Result<ExactMlxCollectedDecision, BridgeError> {
        let config = ExactMlxSearchConfig {
            rollouts: self.rollouts,
            candidate_limit: self.candidate_limit,
            habitat_candidate_limit: self.habitat_candidate_limit,
            seed_coupling: self.seed_coupling,
            rollout_config: self.rollout_config,
        };
        select_action_with_evaluators(
            game,
            collection,
            &mut self.evaluator,
            self.leaf_evaluator.as_mut(),
            config,
            &mut self.diagnostics,
            &mut self.batch_diagnostics,
        )
    }

    pub fn shutdown(self) -> Result<(), BridgeError> {
        let leaf_result = self
            .leaf_evaluator
            .map(|evaluator| evaluator.process.shutdown())
            .transpose();
        let policy_result = self.evaluator.process.shutdown();
        policy_result.map_err(|error| BridgeError::ExactMlx(error.to_string()))?;
        leaf_result
            .map_err(|error| BridgeError::ExactMlx(error.to_string()))?
            .unwrap_or(());
        Ok(())
    }
}

impl ExactRustLegacyTeacher {
    pub fn new(evaluator: NNUENetwork, rollouts: usize) -> Result<Self, BridgeError> {
        validate_legacy_environment()?;
        if rollouts == 0 {
            return Err(BridgeError::LegacyEnvironment(
                "rollouts must be positive".to_owned(),
            ));
        }
        Ok(Self {
            evaluator,
            rollouts,
            diagnostics: BridgeDiagnostics::default(),
            batch_diagnostics: BatchedNnueDiagnostics::default(),
        })
    }

    pub fn select_action(&mut self, game: &v2::GameState) -> Result<v2::TurnAction, BridgeError> {
        self.diagnostics.states_attempted += 1;
        let config = ExactMlxSearchConfig {
            rollouts: self.rollouts,
            candidate_limit: 32,
            habitat_candidate_limit: 0,
            seed_coupling: RolloutSeedCoupling::Independent,
            rollout_config: BatchedRolloutConfig::full(),
        };
        let result = select_action_with_evaluators::<NNUENetwork, NNUENetwork>(
            game,
            ExactMlxCollection::None,
            &mut self.evaluator,
            None,
            config,
            &mut self.diagnostics,
            &mut self.batch_diagnostics,
        )
        .map(|decision| decision.action);
        if let Err(error) = &result {
            self.diagnostics.record_error(error);
        }
        result
    }
}

fn select_action_with_evaluators<E, L>(
    game: &v2::GameState,
    collection: ExactMlxCollection,
    evaluator: &mut E,
    mut leaf_evaluator: Option<&mut L>,
    config: ExactMlxSearchConfig,
    diagnostics: &mut BridgeDiagnostics,
    batch_diagnostics: &mut BatchedNnueDiagnostics,
) -> Result<ExactMlxCollectedDecision, BridgeError>
where
    E: SparseNnueEvaluator,
    L: SparseNnueEvaluator<Error = E::Error>,
    E::Error: std::fmt::Display,
{
    let prepared = prepare_expanded_decision(game, diagnostics, true)?;
    let coordinate_frame = prepared.translation.current_player_frame();
    let mut candidates = prepared.canonical;
    if candidates.is_empty() {
        return Err(BridgeError::NoCandidate);
    }
    let mut novel_habitat_identities = Vec::new();
    if config.habitat_candidate_limit > 0 {
        let habitat = rank_habitat_setup_actions(
            game,
            &prepared.prelude,
            Some(config.habitat_candidate_limit),
        )
        .map_err(|error| BridgeError::Pattern(error.to_string()))?;
        diagnostics.habitat_candidates_generated += habitat.len();
        for canonical in habitat {
            let movement = canonical_candidate_to_legacy_in_frame(&canonical, coordinate_frame);
            let identity = legacy_move_identity(&movement);
            if candidates
                .iter()
                .any(|candidate| legacy_move_identity(candidate) == identity)
            {
                continue;
            }
            let mut translated = prepared.translation.game.clone();
            if !execute_scored_move(&mut translated, &movement) {
                return Err(BridgeError::CanonicalCandidateTranslation(
                    serde_json::to_string(&canonical.action)
                        .expect("serializing an in-memory canonical action cannot fail"),
                ));
            }
            diagnostics.habitat_candidates_novel += 1;
            novel_habitat_identities.push(identity);
            candidates.push(movement);
        }
    }
    if candidates.len() > 32 {
        candidates = nnue_prefilter_candidates_batched(
            &prepared.translation.game,
            evaluator,
            candidates,
            config.candidate_limit,
            batch_diagnostics,
        )
        .map_err(|error| BridgeError::ExactMlx(error.to_string()))?;
    }
    diagnostics.habitat_candidates_retained += candidates
        .iter()
        .filter(|candidate| novel_habitat_identities.contains(&legacy_move_identity(candidate)))
        .count();
    record_prefiltered_candidates(
        game,
        &prepared.prelude,
        &candidates,
        coordinate_frame,
        diagnostics,
    )?;
    let root_afterstates = (!matches!(collection, ExactMlxCollection::None))
        .then(|| prepare_sparse_nnue_afterstates(&prepared.translation.game, &candidates));
    let mut search_rng = legacy_search_rng(&prepared.translation.evidence.public_state_blake3);
    let (estimates, rollout_value_samples) = if let ExactMlxCollection::Samples(trace_modulus) =
        collection
    {
        let result = score_nnue_rollout_mce_seq_halving_batched_with_samples_config_and_coupling(
            &prepared.translation.game,
            evaluator,
            config.rollouts,
            candidates,
            &mut search_rng,
            batch_diagnostics,
            trace_modulus,
            config.rollout_config,
            config.seed_coupling,
        )
        .map_err(|error| BridgeError::ExactMlx(error.to_string()))?;
        (result.estimates, result.rollout_value_samples)
    } else if let Some(leaf_evaluator) = leaf_evaluator.as_mut() {
        (
            score_nnue_rollout_mce_seq_halving_batched_with_leaf_config_and_coupling(
                &prepared.translation.game,
                evaluator,
                *leaf_evaluator,
                config.rollouts,
                candidates,
                &mut search_rng,
                batch_diagnostics,
                config.rollout_config,
                config.seed_coupling,
            )
            .map_err(|error| BridgeError::ExactMlx(error.to_string()))?,
            Vec::new(),
        )
    } else {
        (
            score_nnue_rollout_mce_seq_halving_batched_with_config_and_coupling(
                &prepared.translation.game,
                evaluator,
                config.rollouts,
                candidates,
                &mut search_rng,
                batch_diagnostics,
                config.rollout_config,
                config.seed_coupling,
            )
            .map_err(|error| BridgeError::ExactMlx(error.to_string()))?,
            Vec::new(),
        )
    };
    let selected = estimates.first().ok_or(BridgeError::NoCandidate)?.movement;
    if novel_habitat_identities.contains(&legacy_move_identity(&selected)) {
        diagnostics.selected_novel_habitat_candidates += 1;
    }
    diagnostics.selected_actions += 1;
    let action = map_legacy_action_in_frame(game, &prepared.prelude, &selected, coordinate_frame)?;
    diagnostics.selected_actions_legal += 1;
    let root_estimates = if let Some(root_afterstates) = root_afterstates {
        estimates
            .iter()
            .map(|estimate| {
                let afterstate = root_afterstates
                    .iter()
                    .find(|afterstate| {
                        legacy_move_identity(&afterstate.movement)
                            == legacy_move_identity(&estimate.movement)
                    })
                    .ok_or_else(|| {
                        BridgeError::ExactMlx("root estimate has no sparse afterstate".to_owned())
                    })?;
                Ok(ExactMlxRootEstimate {
                    action: map_legacy_action_in_frame(
                        game,
                        &prepared.prelude,
                        &estimate.movement,
                        coordinate_frame,
                    )?,
                    candidate_index: root_afterstates
                        .iter()
                        .position(|candidate| {
                            legacy_move_identity(&candidate.movement)
                                == legacy_move_identity(&estimate.movement)
                        })
                        .ok_or_else(|| {
                            BridgeError::ExactMlx("root estimate has no candidate index".to_owned())
                        })?,
                    features: afterstate.features.clone(),
                    immediate_score: afterstate.immediate_score,
                    rollout_mean: estimate.rollout_mean,
                    rollout_stddev: estimate.rollout_stddev,
                    samples: estimate.samples,
                    selected: legacy_move_identity(&estimate.movement)
                        == legacy_move_identity(&selected),
                })
            })
            .collect::<Result<Vec<_>, BridgeError>>()?
    } else {
        Vec::new()
    };
    Ok(ExactMlxCollectedDecision {
        action,
        rollout_value_samples,
        root_estimates,
    })
}

fn evaluate_multiplexed_requests<E>(
    evaluator: &mut E,
    pending: &mut Vec<MultiplexEvaluationRequest>,
    diagnostics: &mut ExactMlxMultiplexDiagnostics,
) where
    E: SparseNnueEvaluator,
    E::Error: std::fmt::Display,
{
    pending.sort_by_key(|request| request.search_index);
    let requests = std::mem::take(pending);
    let request_count = requests.len();
    let row_count = requests.iter().map(|request| request.rows.len()).sum();
    diagnostics.evaluator_requests = diagnostics
        .evaluator_requests
        .saturating_add(request_count as u64);
    diagnostics.evaluator_batches = diagnostics.evaluator_batches.saturating_add(1);
    diagnostics.coalesced_batches = diagnostics
        .coalesced_batches
        .saturating_add(u64::from(request_count > 1));
    diagnostics.evaluator_rows = diagnostics.evaluator_rows.saturating_add(row_count as u64);
    diagnostics.maximum_requests_per_batch =
        diagnostics.maximum_requests_per_batch.max(request_count);
    diagnostics.maximum_rows_per_batch = diagnostics.maximum_rows_per_batch.max(row_count);
    if request_count > 1 && cross_request_row_reuse_diagnostics_enabled() {
        diagnostics.cross_request_rows_observed = diagnostics
            .cross_request_rows_observed
            .saturating_add(row_count as u64);
        diagnostics.cross_request_duplicate_rows = diagnostics
            .cross_request_duplicate_rows
            .saturating_add(count_cross_request_duplicate_rows(&requests) as u64);
    }

    let mut combined = Vec::with_capacity(row_count);
    let mut responses = Vec::with_capacity(request_count);
    for mut request in requests {
        let rows = request.rows.len();
        combined.append(&mut request.rows);
        responses.push((request.response, rows));
    }

    let evaluated = evaluator.evaluate_sparse_owned(combined);
    match evaluated {
        Ok(values) if values.len() == row_count => {
            let mut values = values.into_iter();
            for (response, rows) in responses {
                let result = values.by_ref().take(rows).collect::<Vec<_>>();
                let _ = response.send(Ok(result));
            }
            debug_assert!(values.next().is_none());
        }
        Ok(values) => {
            let message = format!(
                "multiplexed exact evaluator returned {} values for {row_count} rows",
                values.len()
            );
            for (response, _) in responses {
                let _ = response.send(Err(MultiplexEvaluationError(message.clone())));
            }
        }
        Err(error) => {
            let message = error.to_string();
            for (response, _) in responses {
                let _ = response.send(Err(MultiplexEvaluationError(message.clone())));
            }
        }
    }
}

fn cross_request_row_reuse_diagnostics_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CASCADIA_MULTIPLEX_ROW_REUSE_DIAGNOSTICS")
            .ok()
            .is_some_and(|value| !value.is_empty() && value != "0")
    })
}

fn count_cross_request_duplicate_rows(requests: &[MultiplexEvaluationRequest]) -> usize {
    let row_count = requests.iter().map(|request| request.rows.len()).sum();
    let mut first_request_by_row = HashMap::<&[u16], usize>::with_capacity(row_count);
    let mut duplicates = 0;
    for (request_index, request) in requests.iter().enumerate() {
        for row in &request.rows {
            match first_request_by_row.get(row.as_slice()).copied() {
                Some(first_request) if first_request != request_index => duplicates += 1,
                Some(_) => {}
                None => {
                    first_request_by_row.insert(row.as_slice(), request_index);
                }
            }
        }
    }
    duplicates
}

fn run_multiplexed_searches<E>(
    evaluator: &mut E,
    games: &[&v2::GameState],
    config: ExactMlxSearchConfig,
    collection: ExactMlxCollection,
) -> Result<(Vec<MultiplexSearchOutput>, ExactMlxMultiplexDiagnostics), BridgeError>
where
    E: SparseNnueEvaluator,
    E::Error: std::fmt::Display,
{
    let search_count = games.len();
    let mut multiplex_diagnostics = ExactMlxMultiplexDiagnostics {
        search_cohorts: 1,
        searches: search_count as u64,
        ..ExactMlxMultiplexDiagnostics::default()
    };

    thread::scope(|scope| {
        let (events, event_receiver) = mpsc::channel::<MultiplexEvent>();
        for (search_index, &game) in games.iter().enumerate() {
            let events = events.clone();
            scope.spawn(move || {
                let mut diagnostics = BridgeDiagnostics {
                    states_attempted: 1,
                    ..BridgeDiagnostics::default()
                };
                let mut batch_diagnostics = BatchedNnueDiagnostics::default();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut evaluator = MultiplexEvaluatorProxy {
                        search_index,
                        events: events.clone(),
                    };
                    select_action_with_evaluators(
                        game,
                        collection,
                        &mut evaluator,
                        None::<&mut MultiplexEvaluatorProxy>,
                        config,
                        &mut diagnostics,
                        &mut batch_diagnostics,
                    )
                }))
                .unwrap_or_else(|_| {
                    Err(BridgeError::ExactMlx(format!(
                        "multiplexed exact search {search_index} panicked"
                    )))
                });
                if let Err(error) = &result {
                    diagnostics.record_error(error);
                }
                let _ = events.send(MultiplexEvent::Finished {
                    search_index,
                    output: MultiplexSearchOutput {
                        result,
                        diagnostics,
                        batch_diagnostics,
                    },
                });
            });
        }
        drop(events);

        let mut active = vec![true; search_count];
        let mut pending = Vec::with_capacity(search_count);
        let mut outputs = std::iter::repeat_with(|| None)
            .take(search_count)
            .collect::<Vec<_>>();
        let mut finished = 0usize;
        while finished < search_count {
            let event = event_receiver.recv().map_err(|_| {
                BridgeError::ExactMlx(
                    "multiplexed exact search event channel disconnected".to_owned(),
                )
            })?;
            match event {
                MultiplexEvent::Evaluate(request) => {
                    if !active[request.search_index]
                        || pending.iter().any(|pending: &MultiplexEvaluationRequest| {
                            pending.search_index == request.search_index
                        })
                    {
                        let _ = request.response.send(Err(MultiplexEvaluationError(
                            "multiplexed search submitted an invalid concurrent request".to_owned(),
                        )));
                    } else {
                        pending.push(request);
                    }
                }
                MultiplexEvent::Finished {
                    search_index,
                    output,
                } => {
                    if !active[search_index] || outputs[search_index].is_some() {
                        return Err(BridgeError::ExactMlx(format!(
                            "multiplexed exact search {search_index} finished more than once"
                        )));
                    }
                    active[search_index] = false;
                    outputs[search_index] = Some(output);
                    finished += 1;
                }
            }

            let active_count = active.iter().filter(|&&is_active| is_active).count();
            if !pending.is_empty() && pending.len() == active_count {
                evaluate_multiplexed_requests(evaluator, &mut pending, &mut multiplex_diagnostics);
            }
        }
        if !pending.is_empty() {
            return Err(BridgeError::ExactMlx(
                "multiplexed exact search completed with pending evaluator requests".to_owned(),
            ));
        }
        outputs
            .into_iter()
            .enumerate()
            .map(|(search_index, output)| {
                output.ok_or_else(|| {
                    BridgeError::ExactMlx(format!(
                        "multiplexed exact search {search_index} produced no output"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()
    })
    .map(|outputs| (outputs, multiplex_diagnostics))
}

fn map_exact_mlx_estimates(
    game: &v2::GameState,
    actions: &[v2::TurnAction],
    candidates: &[ScoredMove],
    afterstates: &[SparseNnueAfterstate],
    estimates: Vec<cascadia_ai::mce::MceMoveEstimate>,
    coordinate_frame: LegacyCoordinateFrame,
) -> Result<Vec<ExactMlxRootEstimate>, BridgeError> {
    if candidates.len() != actions.len() || afterstates.len() != actions.len() {
        return Err(BridgeError::ExactMlx(
            "action-evaluation preparation changed the canonical row count".to_owned(),
        ));
    }
    for (index, candidate) in candidates.iter().enumerate() {
        if candidates[..index]
            .iter()
            .any(|prior| legacy_move_identity(prior) == legacy_move_identity(candidate))
        {
            return Err(BridgeError::ExactMlx(format!(
                "action-evaluation candidate set contains duplicate canonical identity at index {index}"
            )));
        }
    }
    if estimates.len() != actions.len() {
        return Err(BridgeError::ExactMlx(format!(
            "action-evaluation returned {} estimates for {} canonical actions",
            estimates.len(),
            actions.len()
        )));
    }
    let selected_identity = estimates
        .first()
        .map(|estimate| legacy_move_identity(&estimate.movement))
        .ok_or(BridgeError::NoCandidate)?;
    estimates
        .into_iter()
        .map(|estimate| {
            let identity = legacy_move_identity(&estimate.movement);
            let index = candidates
                .iter()
                .position(|candidate| legacy_move_identity(candidate) == identity)
                .ok_or_else(|| {
                    BridgeError::ExactMlx(
                        "action-evaluation estimate has no canonical input action".to_owned(),
                    )
                })?;
            let action = map_legacy_action_in_frame(
                game,
                &actions[index].prelude(),
                &estimate.movement,
                coordinate_frame,
            )?;
            if action != actions[index] {
                return Err(BridgeError::ExactMlx(format!(
                    "action-evaluation estimate changed canonical action identity at index {index}"
                )));
            }
            Ok(ExactMlxRootEstimate {
                action,
                candidate_index: index,
                features: afterstates[index].features.clone(),
                immediate_score: afterstates[index].immediate_score,
                rollout_mean: estimate.rollout_mean,
                rollout_stddev: estimate.rollout_stddev,
                samples: estimate.samples,
                selected: identity == selected_identity,
            })
        })
        .collect()
}

#[cfg(test)]
fn canonical_action_to_legacy(
    game: &v2::GameState,
    action: &v2::TurnAction,
) -> Result<ScoredMove, BridgeError> {
    canonical_action_to_legacy_in_frame(game, action, LegacyCoordinateFrame::ZERO)
}

fn canonical_action_to_legacy_in_frame(
    game: &v2::GameState,
    action: &v2::TurnAction,
    coordinate_frame: LegacyCoordinateFrame,
) -> Result<ScoredMove, BridgeError> {
    game.transition(action)
        .map_err(|error| BridgeError::IllegalMappedAction(error.to_string()))?;
    let (market_index, wildlife_market_index) = match action.draft {
        v2::DraftChoice::Paired { slot } => (slot.index(), None),
        v2::DraftChoice::Independent {
            tile_slot,
            wildlife_slot,
        } => (tile_slot.index(), Some(wildlife_slot.index())),
    };
    Ok(ScoredMove {
        market_index,
        tile_q: coordinate_frame.to_legacy(action.tile.coord)?.q,
        tile_r: coordinate_frame.to_legacy(action.tile.coord)?.r,
        rotation: action.tile.rotation.get(),
        wildlife_q: action
            .wildlife
            .map(|coord| coordinate_frame.to_legacy(coord).map(|coord| coord.q))
            .transpose()?,
        wildlife_r: action
            .wildlife
            .map(|coord| coordinate_frame.to_legacy(coord).map(|coord| coord.r))
            .transpose()?,
        score: 0,
        eval: 0,
        wildlife_market_index,
    })
}

pub fn pattern_fallback(
    game: &v2::GameState,
    rng: &mut ChaCha8Rng,
) -> Result<v2::TurnAction, BridgeError> {
    let prelude = canonical_prelude(game);
    select_pattern_action(game, &prelude, PatternAwareConfig::default(), rng)
        .map_err(|error| BridgeError::Pattern(error.to_string()))
}

pub fn audit_pattern_trajectory(
    game_seed: v2::GameSeed,
    net: &NNUENetwork,
    diagnostics: &mut BridgeDiagnostics,
) -> Result<(), BridgeError> {
    audit_pattern_trajectory_with_gate(
        game_seed,
        net,
        diagnostics,
        ExpandedGate::Strict,
        ScoreParityPolicy::Exact,
    )
}

pub fn audit_retained_pattern_trajectory(
    game_seed: v2::GameSeed,
    net: &NNUENetwork,
    diagnostics: &mut BridgeDiagnostics,
) -> Result<(), BridgeError> {
    audit_pattern_trajectory_with_gate(
        game_seed,
        net,
        diagnostics,
        ExpandedGate::Retained,
        ScoreParityPolicy::Exact,
    )
}

pub fn audit_filtered_pattern_trajectory(
    game_seed: v2::GameSeed,
    net: &NNUENetwork,
    diagnostics: &mut BridgeDiagnostics,
) -> Result<(), BridgeError> {
    audit_pattern_trajectory_with_gate(
        game_seed,
        net,
        diagnostics,
        ExpandedGate::Filter,
        ScoreParityPolicy::Exact,
    )
}

pub fn audit_heuristic_pattern_trajectory(
    game_seed: v2::GameSeed,
    net: &NNUENetwork,
    diagnostics: &mut BridgeDiagnostics,
) -> Result<(), BridgeError> {
    audit_pattern_trajectory_with_gate(
        game_seed,
        net,
        diagnostics,
        ExpandedGate::Filter,
        ScoreParityPolicy::AllowLegacyElkUndercount,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpandedGate {
    Strict,
    Retained,
    Filter,
}

fn audit_pattern_trajectory_with_gate(
    game_seed: v2::GameSeed,
    net: &NNUENetwork,
    diagnostics: &mut BridgeDiagnostics,
    gate: ExpandedGate,
    score_policy: ScoreParityPolicy,
) -> Result<(), BridgeError> {
    let config =
        v2::GameConfig::research_aaaaa(4).map_err(|error| BridgeError::Rules(error.to_string()))?;
    let mut game = v2::GameState::new(config, game_seed)
        .map_err(|error| BridgeError::Rules(error.to_string()))?;
    let mut pattern_rngs = (0..4)
        .map(|seat| strategy_rng(game_seed, seat, PATTERN_AWARE_STRATEGY_ID))
        .collect::<Vec<_>>();
    while !game.is_game_over() {
        diagnostics.states_attempted += 1;
        let prelude = canonical_prelude(&game);
        let staged = game
            .preview_market_prelude(&prelude)
            .map_err(|error| BridgeError::Rules(error.to_string()))?;
        let public = staged.public_state();
        let translation = translate_public_state_with_score_policy(&public, score_policy)?;
        let repeated = translate_public_state_with_score_policy(&public, score_policy)?;
        if translation.evidence != repeated.evidence {
            return Err(BridgeError::Rules(
                "repeated public translation changed its evidence".to_owned(),
            ));
        }
        let mut redetermined = staged.clone();
        redetermined.redeterminize_hidden(v2::GameSeed(domain_hash(
            b"legacy-teacher-hidden-order-audit",
            &hex_to_bytes(&translation.evidence.public_state_blake3),
        )));
        let redetermined_translation =
            translate_public_state_with_score_policy(&redetermined.public_state(), score_policy)?;
        if translation.evidence != redetermined_translation.evidence {
            return Err(BridgeError::Rules(
                "translation changed after hidden-order redetermination".to_owned(),
            ));
        }
        diagnostics.record_translation(&translation.evidence);
        let coordinate_frame = translation.current_player_frame();

        let expanded = expanded_candidates(&translation.game);
        if expanded.is_empty() {
            return Err(BridgeError::NoCandidate);
        }
        diagnostics.expanded_candidates += expanded.len();
        let mut canonical = Vec::with_capacity(expanded.len());
        for candidate in &expanded {
            match map_legacy_action_in_frame(&game, &prelude, candidate, coordinate_frame) {
                Ok(_) => {
                    diagnostics.expanded_candidates_legal += 1;
                    canonical.push(*candidate);
                }
                Err(error) => {
                    diagnostics.expanded_candidates_illegal += 1;
                    diagnostics.record_malformed_expanded(&error);
                    if gate == ExpandedGate::Strict {
                        return Err(error);
                    }
                }
            }
        }
        let candidates = if gate == ExpandedGate::Filter {
            canonical
        } else {
            expanded
        };
        if candidates.is_empty() {
            return Err(BridgeError::NoCandidate);
        }
        let prefiltered = if candidates.len() > 32 {
            nnue_prefilter_candidates(&translation.game, net, candidates, 32)
        } else {
            candidates
        };
        diagnostics.prefiltered_candidates += prefiltered.len();
        for candidate in &prefiltered {
            match map_legacy_action_in_frame(&game, &prelude, candidate, coordinate_frame) {
                Ok(_) => diagnostics.prefiltered_candidates_legal += 1,
                Err(error) => {
                    diagnostics.prefiltered_candidates_illegal += 1;
                    return Err(error);
                }
            }
        }

        let player = game.current_player();
        let action = pattern_fallback(&game, &mut pattern_rngs[player])?;
        game.apply(&action)
            .map_err(|error| BridgeError::Rules(error.to_string()))?;
    }
    Ok(())
}

pub fn simulation_error(error: BridgeError) -> SimulationError {
    SimulationError::Strategy(error.to_string())
}

fn translate_board(
    board: &v2::Board,
    maximum_absolute_coordinate: &mut i8,
) -> Result<(v1::board::Board, LegacyCoordinateFrame), BridgeError> {
    let coordinate_frame = LegacyCoordinateFrame::for_board(board)?;
    let mut translated = v1::board::Board::new();
    for (coord, placed) in board.placed_tiles() {
        *maximum_absolute_coordinate = (*maximum_absolute_coordinate)
            .max(coord.q.saturating_abs())
            .max(coord.r.saturating_abs());
        let legacy_coord = coordinate_frame.to_legacy(coord)?;
        translated
            .place_tile(
                legacy_coord,
                translate_tile(placed.tile),
                placed.rotation.get(),
            )
            .ok_or(BridgeError::TilePlacement {
                q: coord.q,
                r: coord.r,
            })?;
        if let Some(wildlife) = placed.wildlife {
            translated
                .place_wildlife(
                    legacy_coord
                        .to_index()
                        .expect("coordinate range was checked above"),
                    translate_wildlife(wildlife),
                )
                .ok_or(BridgeError::WildlifePlacement {
                    q: coord.q,
                    r: coord.r,
                })?;
        }
    }
    translated.nature_tokens = board.nature_tokens();
    Ok((translated, coordinate_frame))
}

fn translate_tile(tile: v2::Tile) -> v1::types::TileData {
    let wildlife: Vec<_> = tile.wildlife.iter().map(translate_wildlife).collect();
    let allowed = v1::types::WildlifeMask::new(&wildlife);
    match tile.terrain_b {
        Some(terrain_b) => v1::types::TileData::dual(
            translate_terrain(tile.terrain_a),
            translate_terrain(terrain_b),
            allowed,
        ),
        None => v1::types::TileData::single(translate_terrain(tile.terrain_a), allowed),
    }
}

const fn translate_terrain(terrain: v2::Terrain) -> v1::types::Terrain {
    match terrain {
        v2::Terrain::Mountain => v1::types::Terrain::Mountain,
        v2::Terrain::Forest => v1::types::Terrain::Forest,
        v2::Terrain::Prairie => v1::types::Terrain::Prairie,
        v2::Terrain::Wetland => v1::types::Terrain::Wetland,
        v2::Terrain::River => v1::types::Terrain::River,
    }
}

const fn translate_wildlife(wildlife: v2::Wildlife) -> v1::types::Wildlife {
    match wildlife {
        v2::Wildlife::Bear => v1::types::Wildlife::Bear,
        v2::Wildlife::Elk => v1::types::Wildlife::Elk,
        v2::Wildlife::Salmon => v1::types::Wildlife::Salmon,
        v2::Wildlife::Hawk => v1::types::Wildlife::Hawk,
        v2::Wildlife::Fox => v1::types::Wildlife::Fox,
    }
}

fn normalized_v2_score(board: &v2::Board, cards: v2::ScoringCards) -> NormalizedScore {
    let score = v2::score_board(board, cards);
    NormalizedScore {
        habitat: score.habitat,
        wildlife: score.wildlife,
        nature_tokens: score.nature_tokens,
        base_total: score.base_total,
    }
}

fn normalized_v1_score(board: &v1::board::Board) -> NormalizedScore {
    let score =
        v1::scoring::ScoreBreakdown::compute(&mut board.clone(), &v1::types::ScoringCards::all_a());
    NormalizedScore {
        habitat: [
            score.habitat[v1::types::Terrain::Mountain as usize],
            score.habitat[v1::types::Terrain::Forest as usize],
            score.habitat[v1::types::Terrain::Prairie as usize],
            score.habitat[v1::types::Terrain::Wetland as usize],
            score.habitat[v1::types::Terrain::River as usize],
        ],
        wildlife: score.wildlife,
        nature_tokens: score.nature_tokens,
        base_total: score.total,
    }
}

fn allowed_legacy_elk_undercount(
    v2_score: &NormalizedScore,
    v1_score: &NormalizedScore,
) -> Option<u16> {
    if v2_score.habitat != v1_score.habitat
        || v2_score.nature_tokens != v1_score.nature_tokens
        || v2_score.wildlife[0] != v1_score.wildlife[0]
        || v2_score.wildlife[2..] != v1_score.wildlife[2..]
        || v2_score.wildlife[1] < v1_score.wildlife[1]
    {
        return None;
    }
    let elk_delta = v2_score.wildlife[1] - v1_score.wildlife[1];
    if v2_score.base_total != v1_score.base_total.saturating_add(elk_delta) {
        return None;
    }
    Some(elk_delta)
}

fn score_mismatch(
    player: usize,
    board: &v2::Board,
    v2_score: NormalizedScore,
    v1_score: NormalizedScore,
) -> BridgeError {
    BridgeError::ScoreMismatch {
        player,
        v2: v2_score,
        v1: v1_score,
        wildlife: board
            .placed_tiles()
            .filter_map(|(coord, placed)| placed.wildlife.map(|wildlife| (coord, wildlife)))
            .collect(),
    }
}

fn domain_hash(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(domain);
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

fn hex_to_bytes(hex: &str) -> Vec<u8> {
    hex.as_bytes().to_vec()
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        1.0
    } else {
        numerator as f64 / denominator as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::RngCore;

    #[test]
    fn transport_metadata_does_not_modify_the_frozen_legacy_evaluator() {
        for key in TRANSPORT_LEGACY_ENVIRONMENT {
            assert!(permitted_legacy_environment_entry(
                key,
                "arbitrary-envelope-value"
            ));
        }
        assert!(permitted_legacy_environment_entry("MCE_LMR", "1"));
        assert!(!permitted_legacy_environment_entry("MCE_LMR", "0"));
        assert!(!permitted_legacy_environment_entry(
            "CASCADIA_UNREGISTERED_POLICY_KNOB",
            "1"
        ));
    }

    #[derive(Default)]
    struct DeterministicSparseEvaluator {
        owned_requests: Vec<Vec<Vec<u16>>>,
    }

    impl SparseNnueEvaluator for DeterministicSparseEvaluator {
        type Error = std::convert::Infallible;

        fn evaluate_sparse(&mut self, feature_sets: &[Vec<u16>]) -> Result<Vec<f32>, Self::Error> {
            Ok(feature_sets
                .iter()
                .map(|features| deterministic_sparse_value(features))
                .collect())
        }

        fn evaluate_sparse_owned(
            &mut self,
            feature_sets: Vec<Vec<u16>>,
        ) -> Result<Vec<f32>, Self::Error> {
            let values = feature_sets
                .iter()
                .map(|features| deterministic_sparse_value(features))
                .collect();
            self.owned_requests.push(feature_sets);
            Ok(values)
        }

        fn rollout_pipeline_chunk_states(&self) -> Option<usize> {
            Some(8)
        }
    }

    fn deterministic_sparse_value(features: &[u16]) -> f32 {
        let hash = features.iter().fold(2_166_136_261u32, |hash, &feature| {
            hash.wrapping_mul(16_777_619) ^ u32::from(feature)
        });
        (hash % 10_000) as f32 / 100.0
    }

    fn assert_collected_decisions_equal(
        left: &ExactMlxCollectedDecision,
        right: &ExactMlxCollectedDecision,
    ) {
        assert_eq!(left.action, right.action);
        assert_eq!(left.rollout_value_samples, right.rollout_value_samples);
        assert_eq!(left.root_estimates.len(), right.root_estimates.len());
        for (left, right) in left.root_estimates.iter().zip(&right.root_estimates) {
            assert_eq!(left.action, right.action);
            assert_eq!(left.candidate_index, right.candidate_index);
            assert_eq!(left.features, right.features);
            assert_eq!(
                left.immediate_score.to_bits(),
                right.immediate_score.to_bits()
            );
            assert_eq!(left.rollout_mean.to_bits(), right.rollout_mean.to_bits());
            assert_eq!(
                left.rollout_stddev.to_bits(),
                right.rollout_stddev.to_bits()
            );
            assert_eq!(left.samples, right.samples);
            assert_eq!(left.selected, right.selected);
        }
    }

    #[test]
    fn multiplexed_owned_requests_preserve_order_values_and_response_ranges() {
        let (response_two, result_two) = mpsc::sync_channel(1);
        let (response_zero, result_zero) = mpsc::sync_channel(1);
        let (response_one, result_one) = mpsc::sync_channel(1);
        let mut pending = vec![
            MultiplexEvaluationRequest {
                search_index: 2,
                rows: vec![vec![20], vec![21, 22]],
                response: response_two,
            },
            MultiplexEvaluationRequest {
                search_index: 0,
                rows: vec![vec![1, 2, 3]],
                response: response_zero,
            },
            MultiplexEvaluationRequest {
                search_index: 1,
                rows: vec![vec![10], vec![11], vec![12, 13, 14]],
                response: response_one,
            },
        ];
        let expected_combined = vec![
            vec![1, 2, 3],
            vec![10],
            vec![11],
            vec![12, 13, 14],
            vec![20],
            vec![21, 22],
        ];
        let expected_values = expected_combined
            .iter()
            .map(|features| deterministic_sparse_value(features))
            .collect::<Vec<_>>();
        let mut evaluator = DeterministicSparseEvaluator::default();
        let mut diagnostics = ExactMlxMultiplexDiagnostics::default();

        evaluate_multiplexed_requests(&mut evaluator, &mut pending, &mut diagnostics);

        assert!(pending.is_empty());
        assert_eq!(evaluator.owned_requests, vec![expected_combined]);
        assert_eq!(result_zero.recv().unwrap().unwrap(), expected_values[0..1]);
        assert_eq!(result_one.recv().unwrap().unwrap(), expected_values[1..4]);
        assert_eq!(result_two.recv().unwrap().unwrap(), expected_values[4..6]);
        assert_eq!(
            diagnostics,
            ExactMlxMultiplexDiagnostics {
                evaluator_requests: 3,
                evaluator_batches: 1,
                coalesced_batches: 1,
                evaluator_rows: 6,
                maximum_requests_per_batch: 3,
                maximum_rows_per_batch: 6,
                ..ExactMlxMultiplexDiagnostics::default()
            }
        );
    }

    #[test]
    fn cross_request_duplicate_counter_uses_exact_rows_and_ignores_local_repeats() {
        let (response_zero, _result_zero) = mpsc::sync_channel(1);
        let (response_one, _result_one) = mpsc::sync_channel(1);
        let requests = vec![
            MultiplexEvaluationRequest {
                search_index: 0,
                rows: vec![vec![1, 2], vec![1, 2], vec![3]],
                response: response_zero,
            },
            MultiplexEvaluationRequest {
                search_index: 1,
                rows: vec![vec![1, 2], vec![4], vec![1, 2], vec![3]],
                response: response_one,
            },
        ];

        assert_eq!(count_cross_request_duplicate_rows(&requests), 3);
    }

    #[test]
    fn multiplexed_search_matches_serial_actions_estimates_and_logical_work() {
        let config = v2::GameConfig::research_aaaaa(4).unwrap();
        let games = [61_101, 61_102]
            .map(|seed| v2::GameState::new(config, v2::GameSeed::from_u64(seed)).unwrap());
        let search_config = ExactMlxSearchConfig {
            rollouts: 8,
            candidate_limit: 32,
            habitat_candidate_limit: 0,
            seed_coupling: RolloutSeedCoupling::Independent,
            rollout_config: BatchedRolloutConfig::truncated(2).unwrap(),
        };
        let mut serial_outputs = Vec::new();
        for game in &games {
            let mut evaluator = DeterministicSparseEvaluator::default();
            let mut diagnostics = BridgeDiagnostics {
                states_attempted: 1,
                ..BridgeDiagnostics::default()
            };
            let mut batch_diagnostics = BatchedNnueDiagnostics::default();
            let result = select_action_with_evaluators(
                game,
                ExactMlxCollection::Estimates,
                &mut evaluator,
                None::<&mut DeterministicSparseEvaluator>,
                search_config,
                &mut diagnostics,
                &mut batch_diagnostics,
            )
            .unwrap();
            serial_outputs.push((result, diagnostics, batch_diagnostics));
        }

        let mut evaluator = DeterministicSparseEvaluator::default();
        let game_refs = games.iter().collect::<Vec<_>>();
        let (multiplexed, multiplex_diagnostics) = run_multiplexed_searches(
            &mut evaluator,
            &game_refs,
            search_config,
            ExactMlxCollection::Estimates,
        )
        .unwrap();

        assert_eq!(multiplexed.len(), serial_outputs.len());
        for (multiplexed, (serial, diagnostics, batch_diagnostics)) in
            multiplexed.iter().zip(&serial_outputs)
        {
            assert_collected_decisions_equal(multiplexed.result.as_ref().unwrap(), serial);
            assert_eq!(&multiplexed.diagnostics, diagnostics);
            assert_eq!(&multiplexed.batch_diagnostics, batch_diagnostics);
        }
        assert_eq!(multiplex_diagnostics.search_cohorts, 1);
        assert_eq!(multiplex_diagnostics.searches, 2);
        assert!(multiplex_diagnostics.evaluator_requests > 0);
        assert!(multiplex_diagnostics.evaluator_batches > 0);
        assert_eq!(
            multiplex_diagnostics.evaluator_rows,
            evaluator.owned_requests.iter().map(Vec::len).sum::<usize>() as u64
        );
    }

    #[test]
    fn public_translation_is_hidden_order_invariant() {
        let config = v2::GameConfig::research_aaaaa(4).unwrap();
        let game = v2::GameState::new(config, v2::GameSeed::from_u64(41)).unwrap();
        let left = translate_public_state(&game.public_state()).unwrap();
        assert_eq!(
            left.coordinate_frames,
            [LegacyCoordinateFrame::ZERO; 4],
            "the qualified in-range domain must retain its historical coordinates"
        );
        let mut redetermined = game;
        redetermined.redeterminize_hidden(v2::GameSeed::from_u64(42));
        let right = translate_public_state(&redetermined.public_state()).unwrap();
        assert_eq!(left.evidence, right.evidence);
    }

    #[test]
    fn translated_board_outside_the_legacy_origin_window_is_lossless() {
        let mut board = v2::Board::empty();
        board
            .place_tile(
                v2::HexCoord::new(11, -4),
                v2::STANDARD_TILES[0],
                v2::Rotation::new(0).unwrap(),
            )
            .unwrap();
        let mut maximum_absolute_coordinate = 0;
        let (translated, frame) =
            translate_board(&board, &mut maximum_absolute_coordinate).unwrap();

        assert_eq!(frame.q_offset, -1);
        assert_eq!(frame.r_offset, 0);
        assert_eq!(maximum_absolute_coordinate, 11);
        assert!(
            translated
                .grid
                .get_coord(v1::hex::HexCoord::new(10, -4))
                .is_some_and(|cell| cell.is_present())
        );
        assert_eq!(
            normalized_v2_score(&board, v2::ScoringCards::AAAAA),
            normalized_v1_score(&translated)
        );
    }

    #[test]
    fn coordinate_frame_round_trips_complete_actions() {
        let config = v2::GameConfig::research_aaaaa(4).unwrap();
        let game = v2::GameState::new(config, v2::GameSeed::from_u64(40_003)).unwrap();
        let prelude = canonical_prelude(&game);
        let action = game.legal_turn_actions(&prelude).unwrap().remove(0);
        let frame = LegacyCoordinateFrame {
            q_offset: -1,
            r_offset: 2,
        };
        let movement = canonical_action_to_legacy_in_frame(&game, &action, frame).unwrap();
        assert_eq!(
            map_legacy_action_in_frame(&game, &prelude, &movement, frame).unwrap(),
            action
        );
    }

    #[test]
    fn coordinate_span_larger_than_legacy_storage_is_rejected_without_clipping() {
        let mut board = v2::Board::empty();
        for q in -10..=11 {
            board
                .place_tile(
                    v2::HexCoord::new(q, 0),
                    v2::STANDARD_TILES[0],
                    v2::Rotation::new(0).unwrap(),
                )
                .unwrap();
        }
        let error = LegacyCoordinateFrame::for_board(&board).unwrap_err();
        assert!(matches!(
            error,
            BridgeError::CoordinateSpanOutOfRange {
                axis: 'q',
                minimum: -10,
                maximum: 11,
            }
        ));
    }

    #[test]
    fn domain_separated_search_rng_is_reproducible_and_distinct() {
        let public_hash = "0123456789abcdef";
        let mut left = legacy_search_rng_in_domain(public_hash, b"substantial-r1200");
        let mut repeat = legacy_search_rng_in_domain(public_hash, b"substantial-r1200");
        let mut different = legacy_search_rng_in_domain(public_hash, b"high-confidence-r4800");
        assert_eq!(left.next_u64(), repeat.next_u64());
        assert_ne!(left.next_u64(), different.next_u64());
    }

    #[test]
    fn every_initial_canonical_action_round_trips_through_the_legacy_search_move() {
        let config = v2::GameConfig::research_aaaaa(4).unwrap();
        let game = v2::GameState::new(config, v2::GameSeed::from_u64(40_001)).unwrap();
        let prelude = canonical_prelude(&game);
        let actions = game.legal_turn_actions(&prelude).unwrap();
        assert!(!actions.is_empty());

        let mut identities = std::collections::BTreeSet::new();
        for action in actions {
            let movement = canonical_action_to_legacy(&game, &action).unwrap();
            assert!(identities.insert(legacy_move_identity(&movement)));
            assert_eq!(
                map_legacy_action(&game, &prelude, &movement).unwrap(),
                action
            );
        }
    }

    #[test]
    fn exact_action_estimate_mapping_preserves_sorted_canonical_identity() {
        let config = v2::GameConfig::research_aaaaa(4).unwrap();
        let game = v2::GameState::new(config, v2::GameSeed::from_u64(40_002)).unwrap();
        let prelude = canonical_prelude(&game);
        let actions = game
            .legal_turn_actions(&prelude)
            .unwrap()
            .into_iter()
            .take(8)
            .collect::<Vec<_>>();
        let staged = game.preview_market_prelude(&prelude).unwrap();
        let translated =
            translate_public_state_allowing_legacy_elk_undercount(&staged.public_state()).unwrap();
        let candidates = actions
            .iter()
            .map(|action| canonical_action_to_legacy(&game, action).unwrap())
            .collect::<Vec<_>>();
        let afterstates = prepare_sparse_nnue_afterstates(&translated.game, &candidates);
        let raw = candidates
            .iter()
            .rev()
            .enumerate()
            .map(|(index, movement)| cascadia_ai::mce::MceMoveEstimate {
                movement: *movement,
                rollout_mean: 100.0 - index as f64,
                rollout_stddev: index as f64 / 10.0,
                samples: 12 + index as u32,
            })
            .collect();

        let mapped = map_exact_mlx_estimates(
            &game,
            &actions,
            &candidates,
            &afterstates,
            raw,
            LegacyCoordinateFrame::ZERO,
        )
        .unwrap();
        assert_eq!(mapped.len(), actions.len());
        assert_eq!(mapped[0].action, *actions.last().unwrap());
        assert!(mapped[0].selected);
        assert!(mapped.iter().skip(1).all(|estimate| !estimate.selected));
        for (mapped, expected) in mapped.iter().zip(actions.iter().rev()) {
            assert_eq!(&mapped.action, expected);
        }
    }

    #[test]
    fn complete_pattern_game_retains_score_parity_at_every_state() {
        let config = v2::GameConfig::research_aaaaa(4).unwrap();
        let seed = v2::GameSeed::from_u64(43);
        let mut game = v2::GameState::new(config, seed).unwrap();
        let mut rngs = (0..4)
            .map(|seat| strategy_rng(seed, seat, PATTERN_AWARE_STRATEGY_ID))
            .collect::<Vec<_>>();
        let mut checked = 0;
        while !game.is_game_over() {
            let prelude = canonical_prelude(&game);
            let staged = game.preview_market_prelude(&prelude).unwrap();
            translate_public_state(&staged.public_state()).unwrap();
            checked += 1;
            let player = game.current_player();
            let action = pattern_fallback(&game, &mut rngs[player]).unwrap();
            game.apply(&action).unwrap();
        }
        assert_eq!(checked, 80);
    }

    #[test]
    fn heuristic_translation_measures_only_the_known_elk_undercount() {
        let config = v2::GameConfig::research_aaaaa(4).unwrap();
        let seed = v2::GameSeed::from_u64(32_000);
        let mut game = v2::GameState::new(config, seed).unwrap();
        let mut rngs = (0..4)
            .map(|seat| strategy_rng(seed, seat, PATTERN_AWARE_STRATEGY_ID))
            .collect::<Vec<_>>();
        let mut found = false;
        while !game.is_game_over() {
            let prelude = canonical_prelude(&game);
            let staged = game.preview_market_prelude(&prelude).unwrap();
            let public = staged.public_state();
            if matches!(
                translate_public_state(&public),
                Err(BridgeError::ScoreMismatch { .. })
            ) {
                let relaxed =
                    translate_public_state_allowing_legacy_elk_undercount(&public).unwrap();
                assert_eq!(relaxed.evidence.elk_score_mismatch_boards, 1);
                assert_eq!(relaxed.evidence.elk_v2_minus_v1_total, 1);
                assert_eq!(relaxed.evidence.elk_v2_minus_v1_max, 1);
                found = true;
                break;
            }
            let player = game.current_player();
            let action = pattern_fallback(&game, &mut rngs[player]).unwrap();
            game.apply(&action).unwrap();
        }
        assert!(
            found,
            "seed 32000 must preserve the discovered Elk mismatch"
        );
    }

    #[test]
    fn heuristic_score_policy_rejects_non_elk_differences_and_v1_overcounts() {
        let baseline = NormalizedScore {
            habitat: [1, 2, 3, 4, 5],
            wildlife: [6, 7, 8, 9, 10],
            nature_tokens: 2,
            base_total: 57,
        };
        let mut allowed = baseline.clone();
        allowed.wildlife[1] += 2;
        allowed.base_total += 2;
        assert_eq!(allowed_legacy_elk_undercount(&allowed, &baseline), Some(2));

        let mut bear_difference = allowed.clone();
        bear_difference.wildlife[0] += 1;
        bear_difference.base_total += 1;
        assert_eq!(
            allowed_legacy_elk_undercount(&bear_difference, &baseline),
            None
        );

        assert_eq!(allowed_legacy_elk_undercount(&baseline, &allowed), None);
    }

    #[test]
    fn pattern_frontier_recall_uses_complete_typed_actions() {
        let config = v2::GameConfig::research_aaaaa(4).unwrap();
        let game = v2::GameState::new(config, v2::GameSeed::from_u64(45)).unwrap();
        let prelude = canonical_prelude(&game);
        let frontier =
            rank_pattern_actions(&game, &prelude, PatternAwareConfig::default()).unwrap();
        let action = frontier[0].action.clone();
        let mut diagnostics = BridgeDiagnostics {
            selected_actions: 1,
            ..BridgeDiagnostics::default()
        };
        diagnostics.record_pattern_frontier_recall(&game, &action, &frontier);
        assert_eq!(diagnostics.selected_actions_in_pattern_frontier, 1);
        assert_eq!(diagnostics.selected_actions_by_phase, [1, 0, 0]);
        assert_eq!(diagnostics.pattern_frontier_recall(), 1.0);
    }

    #[test]
    fn every_initial_legacy_candidate_maps_to_a_legal_v2_action() {
        let config = v2::GameConfig::research_aaaaa(4).unwrap();
        let game = v2::GameState::new(config, v2::GameSeed::from_u64(44)).unwrap();
        let prelude = canonical_prelude(&game);
        let staged = game.preview_market_prelude(&prelude).unwrap();
        let translated = translate_public_state(&staged.public_state()).unwrap();
        let candidates = expanded_candidates(&translated.game);
        assert!(!candidates.is_empty());
        for candidate in candidates {
            map_legacy_action(&game, &prelude, &candidate).unwrap();
        }
    }

    #[test]
    fn canonical_habitat_candidates_round_trip_through_the_legacy_search_move() {
        let config = v2::GameConfig::research_aaaaa(4).unwrap();
        let game = v2::GameState::new(config, v2::GameSeed::from_u64(46)).unwrap();
        let prelude = canonical_prelude(&game);
        let staged = game.preview_market_prelude(&prelude).unwrap();
        let translated = translate_public_state(&staged.public_state()).unwrap();
        let candidates = rank_habitat_setup_actions(&game, &prelude, Some(6)).unwrap();
        assert_eq!(candidates.len(), 6);
        for candidate in candidates {
            let movement = canonical_candidate_to_legacy(&candidate);
            assert_eq!(
                map_legacy_action(&game, &prelude, &movement).unwrap(),
                candidate.action
            );
            let mut legacy_afterstate = translated.game.clone();
            assert!(execute_scored_move(&mut legacy_afterstate, &movement));
        }
    }

    #[test]
    fn same_slot_independent_action_round_trips_through_the_legacy_search_move() {
        let config = v2::GameConfig::research_aaaaa(4).unwrap();
        let seed = v2::GameSeed::from_u64(5_001);
        let mut game = v2::GameState::new(config, seed).unwrap();
        let mut rngs = (0..4)
            .map(|seat| strategy_rng(seed, seat, PATTERN_AWARE_STRATEGY_ID))
            .collect::<Vec<_>>();

        while !game.is_game_over() {
            let player = game.current_player();
            if game.boards()[player].nature_tokens() > 0 {
                let prelude = canonical_prelude(&game);
                let action = game
                    .legal_turn_actions(&prelude)
                    .unwrap()
                    .into_iter()
                    .find(|action| {
                        matches!(
                            action.draft,
                            v2::DraftChoice::Independent {
                                tile_slot,
                                wildlife_slot,
                            } if tile_slot == wildlife_slot
                        )
                    })
                    .expect("a player with a Nature Token has a same-slot independent action");
                let staged = game.preview_market_prelude(&prelude).unwrap();
                let translated =
                    translate_public_state_allowing_legacy_elk_undercount(&staged.public_state())
                        .unwrap();
                let movement = canonical_action_to_legacy(&game, &action).unwrap();
                assert_eq!(movement.wildlife_market_index, Some(movement.market_index));
                assert_eq!(
                    map_legacy_action(&game, &prelude, &movement).unwrap(),
                    action
                );

                let legacy_player = translated.game.current_player;
                let before_tokens = translated.game.boards[legacy_player].nature_tokens;
                let mut after = translated.game;
                assert!(execute_scored_move(&mut after, &movement));
                assert_eq!(after.boards[legacy_player].nature_tokens + 1, before_tokens);
                return;
            }

            let action = pattern_fallback(&game, &mut rngs[player]).unwrap();
            game.apply(&action).unwrap();
        }

        panic!("seed 5001 must award a Nature Token before the game ends");
    }

    #[test]
    fn legacy_executor_silently_drops_the_first_malformed_wildlife_record() {
        let config = v2::GameConfig::research_aaaaa(4).unwrap();
        let seed = v2::GameSeed::from_u64(31_600);
        let mut game = v2::GameState::new(config, seed).unwrap();
        let mut rngs = (0..4)
            .map(|seat| strategy_rng(seed, seat, PATTERN_AWARE_STRATEGY_ID))
            .collect::<Vec<_>>();
        let mut demonstrated = false;
        while !game.is_game_over() && !demonstrated {
            let prelude = canonical_prelude(&game);
            let staged = game.preview_market_prelude(&prelude).unwrap();
            let translated = translate_public_state(&staged.public_state()).unwrap();
            for candidate in expanded_candidates(&translated.game) {
                if map_legacy_action(&game, &prelude, &candidate).is_ok() {
                    continue;
                }
                let wildlife_slot = candidate
                    .wildlife_market_index
                    .unwrap_or(candidate.market_index);
                let drafted = translated.game.market.pairs[wildlife_slot]
                    .as_ref()
                    .unwrap()
                    .wildlife;
                let wildlife_coord = v1::hex::HexCoord::new(
                    candidate.wildlife_q.unwrap(),
                    candidate.wildlife_r.unwrap(),
                );
                let before_turns = translated.game.turns_remaining;
                let mut executed = translated.game.clone();
                assert!(cascadia_ai::search::execute_scored_move(
                    &mut executed,
                    &candidate
                ));
                assert_eq!(executed.turns_remaining, before_turns - 1);
                assert_ne!(
                    executed.boards[translated.game.current_player]
                        .grid
                        .get(wildlife_coord.to_index().unwrap())
                        .placed_wildlife(),
                    Some(drafted)
                );
                demonstrated = true;
                break;
            }
            if !demonstrated {
                let player = game.current_player();
                let action = pattern_fallback(&game, &mut rngs[player]).unwrap();
                game.apply(&action).unwrap();
            }
        }
        assert!(demonstrated);
    }
}
