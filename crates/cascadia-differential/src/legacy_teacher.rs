//! Isolated public-state adapter for evaluating the historical v1 teacher.
//!
//! This module is feature-gated and belongs to the differential research
//! boundary. Production v2 crates never depend on v1.

use std::path::Path;

use blake3::Hasher;
use cascadia_ai::{
    eval::ScoredMove,
    mce::{
        GreedyMceAlloc, best_move_nnue_rollout_mce, expanded_candidates, nnue_prefilter_candidates,
        score_nnue_rollout_mce_seq_halving,
    },
    nnue::NNUENetwork,
    nnue_batch::{
        BatchedNnueDiagnostics, RolloutSeedCoupling, RolloutValueSample, SparseNnueAfterstate,
        SparseNnueEvaluator, nnue_prefilter_candidates_batched, prepare_sparse_nnue_afterstates,
        score_nnue_rollout_mce_seq_halving_batched_with_coupling,
        score_nnue_rollout_mce_seq_halving_batched_with_samples_and_coupling,
    },
    search::execute_scored_move,
};
use cascadia_core as v1;
use cascadia_game as v2;
use cascadia_model::{ExactNnueHiddenPrediction, ModelError, ModelProcess};
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

const ALLOWED_LEGACY_ENVIRONMENT: &[(&str, &str)] =
    &[("MCE_LMR", "1"), ("MCE_DIVERSE_PREFILTER", "1")];

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
        matches!(self, Self::CoordinateOutOfRange { .. })
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
        if ALLOWED_LEGACY_ENVIRONMENT
            .iter()
            .any(|(allowed, _)| *allowed == key)
        {
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
    for (player, board) in public.boards().iter().enumerate() {
        let translated = translate_board(board, &mut maximum_absolute_coordinate)?;
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
    Ok(LegacyTranslation { game, evidence })
}

pub fn map_legacy_action(
    original: &v2::GameState,
    prelude: &v2::MarketPrelude,
    candidate: &ScoredMove,
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
        (Some(q), Some(r)) => Some(v2::HexCoord::new(q, r)),
        (None, None) => None,
        _ => return Err(BridgeError::PartialWildlifeCoordinate),
    };
    let action = v2::TurnAction {
        replace_three_of_a_kind: prelude.replace_three_of_a_kind,
        wildlife_wipes: prelude.wildlife_wipes.clone(),
        draft,
        tile: v2::TilePlacement {
            coord: v2::HexCoord::new(candidate.tile_q, candidate.tile_r),
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

fn canonical_candidate_to_legacy(candidate: &cascadia_sim::GreedyCandidate) -> ScoredMove {
    let (market_index, wildlife_market_index) = match candidate.action.draft {
        v2::DraftChoice::Paired { slot } => (slot.index(), None),
        v2::DraftChoice::Independent {
            tile_slot,
            wildlife_slot,
        } => (tile_slot.index(), Some(wildlife_slot.index())),
    };
    ScoredMove {
        market_index,
        tile_q: candidate.action.tile.coord.q,
        tile_r: candidate.action.tile.coord.r,
        rotation: candidate.action.tile.rotation.get(),
        wildlife_q: candidate.action.wildlife.map(|coord| coord.q),
        wildlife_r: candidate.action.wildlife.map(|coord| coord.r),
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

    let expanded = expanded_candidates(&translation.game);
    diagnostics.expanded_candidates += expanded.len();
    let mut canonical = Vec::with_capacity(expanded.len());
    for candidate in &expanded {
        match map_legacy_action(game, &prelude, candidate) {
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
    diagnostics: &mut BridgeDiagnostics,
) -> Result<(), BridgeError> {
    diagnostics.prefiltered_candidates += candidates.len();
    for candidate in candidates {
        match map_legacy_action(game, prelude, candidate) {
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
        let result = self.select_action_with_estimates_inner(game);
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
        let action = map_legacy_action(game, &prepared.prelude, &selected)?;
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
    ) -> Result<LegacyTeacherDecision, BridgeError> {
        let prepared = self.prepare_decision(game)?;
        let mut search_rng = legacy_search_rng(&prepared.translation.evidence.public_state_blake3);
        let estimates = score_nnue_rollout_mce_seq_halving(
            &prepared.translation.game,
            &self.net,
            self.rollouts,
            prepared.candidates,
            &mut search_rng,
        );
        let selected = estimates.first().ok_or(BridgeError::NoCandidate)?.movement;
        let action = map_legacy_action(game, &prepared.prelude, &selected)?;
        let mapped = estimates
            .into_iter()
            .map(|estimate| {
                Ok(LegacyActionEstimate {
                    action: map_legacy_action(game, &prepared.prelude, &estimate.movement)?,
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
        record_prefiltered_candidates(game, &prepared.prelude, &candidates, &mut self.diagnostics)?;
        Ok(PreparedLegacyDecision {
            prelude: prepared.prelude,
            translation: prepared.translation,
            candidates,
        })
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
}

pub struct ExactMlxLegacyTeacher {
    evaluator: ExactMlxEvaluator,
    rollouts: usize,
    candidate_limit: usize,
    habitat_candidate_limit: usize,
    seed_coupling: RolloutSeedCoupling,
    pub diagnostics: BridgeDiagnostics,
    pub batch_diagnostics: BatchedNnueDiagnostics,
}

#[derive(Debug, Clone)]
pub struct ExactMlxRootEstimate {
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

impl ExactMlxLegacyTeacher {
    pub fn new(process: ModelProcess, rollouts: usize) -> Result<Self, BridgeError> {
        Self::new_with_candidate_configuration(
            process,
            rollouts,
            32,
            0,
            RolloutSeedCoupling::Independent,
        )
    }

    pub fn new_with_seed_coupling(
        process: ModelProcess,
        rollouts: usize,
        seed_coupling: RolloutSeedCoupling,
    ) -> Result<Self, BridgeError> {
        Self::new_with_candidate_configuration(process, rollouts, 32, 0, seed_coupling)
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
        )
    }

    fn new_with_candidate_configuration(
        process: ModelProcess,
        rollouts: usize,
        candidate_limit: usize,
        habitat_candidate_limit: usize,
        seed_coupling: RolloutSeedCoupling,
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
            rollouts,
            candidate_limit,
            habitat_candidate_limit,
            seed_coupling,
            diagnostics: BridgeDiagnostics::default(),
            batch_diagnostics: BatchedNnueDiagnostics::default(),
        })
    }

    pub fn select_action(&mut self, game: &v2::GameState) -> Result<v2::TurnAction, BridgeError> {
        self.diagnostics.states_attempted += 1;
        let result = self
            .select_action_inner(game, None)
            .map(|decision| decision.action);
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
        let result = self.select_action_inner(game, Some(trace_modulus));
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
        let afterstates = self.prepare_action_prior_afterstates(game, actions)?;
        let feature_sets = afterstates
            .iter()
            .map(|afterstate| afterstate.features.clone())
            .collect::<Vec<_>>();
        let remaining = self
            .evaluator
            .process
            .predict_sparse_nnue_csr_exact(&feature_sets)
            .map_err(|error| BridgeError::ExactMlx(error.to_string()))?;
        self.batch_diagnostics.record_batch(feature_sets.len());
        if remaining.len() != afterstates.len() {
            return Err(BridgeError::ExactMlx(
                "candidate-prior evaluator returned the wrong row count".to_owned(),
            ));
        }
        Ok(afterstates
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
        let afterstates = self.prepare_action_prior_afterstates(game, actions)?;
        let feature_sets = afterstates
            .iter()
            .map(|afterstate| afterstate.features.clone())
            .collect::<Vec<_>>();
        let predictions = self
            .evaluator
            .process
            .predict_sparse_nnue_csr_exact_hidden(&feature_sets)
            .map_err(|error| BridgeError::ExactMlx(error.to_string()))?;
        self.batch_diagnostics.record_batch(feature_sets.len());
        if predictions.len() != afterstates.len() {
            return Err(BridgeError::ExactMlx(
                "candidate-hidden evaluator returned the wrong row count".to_owned(),
            ));
        }
        Ok(afterstates
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

    fn prepare_action_prior_afterstates(
        &mut self,
        game: &v2::GameState,
        actions: &[v2::TurnAction],
    ) -> Result<Vec<SparseNnueAfterstate>, BridgeError> {
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
        let candidates = actions
            .iter()
            .map(|action| canonical_action_to_legacy(game, action))
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
        Ok(afterstates)
    }

    fn select_action_inner(
        &mut self,
        game: &v2::GameState,
        trace_modulus: Option<u64>,
    ) -> Result<ExactMlxCollectedDecision, BridgeError> {
        let prepared = prepare_expanded_decision(game, &mut self.diagnostics, true)?;
        let mut candidates = prepared.canonical;
        if candidates.is_empty() {
            return Err(BridgeError::NoCandidate);
        }
        let mut novel_habitat_identities = Vec::new();
        if self.habitat_candidate_limit > 0 {
            let habitat = rank_habitat_setup_actions(
                game,
                &prepared.prelude,
                Some(self.habitat_candidate_limit),
            )
            .map_err(|error| BridgeError::Pattern(error.to_string()))?;
            self.diagnostics.habitat_candidates_generated += habitat.len();
            for canonical in habitat {
                let movement = canonical_candidate_to_legacy(&canonical);
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
                self.diagnostics.habitat_candidates_novel += 1;
                novel_habitat_identities.push(identity);
                candidates.push(movement);
            }
        }
        if candidates.len() > 32 {
            candidates = nnue_prefilter_candidates_batched(
                &prepared.translation.game,
                &mut self.evaluator,
                candidates,
                self.candidate_limit,
                &mut self.batch_diagnostics,
            )
            .map_err(|error| BridgeError::ExactMlx(error.to_string()))?;
        }
        self.diagnostics.habitat_candidates_retained += candidates
            .iter()
            .filter(|candidate| novel_habitat_identities.contains(&legacy_move_identity(candidate)))
            .count();
        record_prefiltered_candidates(game, &prepared.prelude, &candidates, &mut self.diagnostics)?;
        let root_afterstates = trace_modulus
            .map(|_| prepare_sparse_nnue_afterstates(&prepared.translation.game, &candidates));
        let mut search_rng = legacy_search_rng(&prepared.translation.evidence.public_state_blake3);
        let (estimates, rollout_value_samples) = if let Some(trace_modulus) = trace_modulus {
            let result = score_nnue_rollout_mce_seq_halving_batched_with_samples_and_coupling(
                &prepared.translation.game,
                &mut self.evaluator,
                self.rollouts,
                candidates,
                &mut search_rng,
                &mut self.batch_diagnostics,
                trace_modulus,
                self.seed_coupling,
            )
            .map_err(|error| BridgeError::ExactMlx(error.to_string()))?;
            (result.estimates, result.rollout_value_samples)
        } else {
            (
                score_nnue_rollout_mce_seq_halving_batched_with_coupling(
                    &prepared.translation.game,
                    &mut self.evaluator,
                    self.rollouts,
                    candidates,
                    &mut search_rng,
                    &mut self.batch_diagnostics,
                    self.seed_coupling,
                )
                .map_err(|error| BridgeError::ExactMlx(error.to_string()))?,
                Vec::new(),
            )
        };
        let selected = estimates.first().ok_or(BridgeError::NoCandidate)?.movement;
        if novel_habitat_identities.contains(&legacy_move_identity(&selected)) {
            self.diagnostics.selected_novel_habitat_candidates += 1;
        }
        self.diagnostics.selected_actions += 1;
        let action = map_legacy_action(game, &prepared.prelude, &selected)?;
        self.diagnostics.selected_actions_legal += 1;
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
                            BridgeError::ExactMlx(
                                "root estimate has no sparse afterstate".to_owned(),
                            )
                        })?;
                    Ok(ExactMlxRootEstimate {
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

    pub fn shutdown(self) -> Result<(), BridgeError> {
        self.evaluator
            .process
            .shutdown()
            .map_err(|error| BridgeError::ExactMlx(error.to_string()))
    }
}

fn canonical_action_to_legacy(
    game: &v2::GameState,
    action: &v2::TurnAction,
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
        tile_q: action.tile.coord.q,
        tile_r: action.tile.coord.r,
        rotation: action.tile.rotation.get(),
        wildlife_q: action.wildlife.map(|coord| coord.q),
        wildlife_r: action.wildlife.map(|coord| coord.r),
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

        let expanded = expanded_candidates(&translation.game);
        if expanded.is_empty() {
            return Err(BridgeError::NoCandidate);
        }
        diagnostics.expanded_candidates += expanded.len();
        let mut canonical = Vec::with_capacity(expanded.len());
        for candidate in &expanded {
            match map_legacy_action(&game, &prelude, candidate) {
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
            match map_legacy_action(&game, &prelude, candidate) {
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
) -> Result<v1::board::Board, BridgeError> {
    let mut translated = v1::board::Board::new();
    for (coord, placed) in board.placed_tiles() {
        *maximum_absolute_coordinate = (*maximum_absolute_coordinate)
            .max(coord.q.saturating_abs())
            .max(coord.r.saturating_abs());
        let legacy_coord = v1::hex::HexCoord::new(coord.q, coord.r);
        if legacy_coord.to_index().is_none() {
            return Err(BridgeError::CoordinateOutOfRange {
                q: coord.q,
                r: coord.r,
            });
        }
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
    Ok(translated)
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

    #[test]
    fn public_translation_is_hidden_order_invariant() {
        let config = v2::GameConfig::research_aaaaa(4).unwrap();
        let game = v2::GameState::new(config, v2::GameSeed::from_u64(41)).unwrap();
        let left = translate_public_state(&game.public_state()).unwrap();
        let mut redetermined = game;
        redetermined.redeterminize_hidden(v2::GameSeed::from_u64(42));
        let right = translate_public_state(&redetermined.public_state()).unwrap();
        assert_eq!(left.evidence, right.evidence);
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
