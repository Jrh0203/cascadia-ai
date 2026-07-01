//! Full-legal public-information decision-regret audit for the accepted exact MLX champion.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs, io,
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use cascadia_ai::nnue_batch::{
    BatchedNnueDiagnostics, BatchedNnueStageTimings, RolloutSeedCoupling,
};
use cascadia_game::{
    Board, DraftChoice, GameConfig, GameSeed, GameState, Market, MarketPrelude, MarketSlot,
    PublicGameState, PublicSupply, ScoreBreakdown, TurnAction, Wildlife, WildlifeWipe,
    rescore_after_tile_with_habitat_analysis, rescore_after_wildlife_placement,
    rescore_with_wildlife_scores, score_board, score_game,
};
use cascadia_provenance::SourceProvenance;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::legacy_teacher::{
    BridgeDiagnostics, BridgeError, ExactMlxCollectedDecision, ExactMlxLegacyTeacher,
    ExactMlxMultiplexDiagnostics, ExactMlxRootEstimate, canonical_prelude,
};

pub const FULL_LEGAL_AUDIT_PROTOCOL_ID: &str = "full-legal-decision-regret-audit-v1";
pub const FROZEN_CHAMPION_ROLLOUTS: usize = 600;
pub const FROZEN_SCREEN_LIMIT: usize = 64;
pub const FROZEN_SENTINEL_COUNT: usize = 16;
pub const FROZEN_SUBSTANTIAL_ROLLOUTS: usize = 1_200;
pub const FROZEN_HIGH_CONFIDENCE_LIMIT: usize = 8;
pub const FROZEN_HIGH_CONFIDENCE_ROLLOUTS: usize = 4_800;
pub const FROZEN_REALIZED_HIDDEN_TURNS: [u16; 3] = [12, 39, 66];
pub const FROZEN_PAID_WIPE_DETERMINIZATIONS: usize = 8;
pub const FROZEN_PAID_WIPE_FOLLOWUP_DETERMINIZATIONS: usize = 2;
pub const FROZEN_PAID_WIPE_FOLLOWUP_WIDTH: usize = 3;

#[derive(Debug, Error)]
pub enum FullLegalAuditError {
    #[error("legacy bridge failed: {0}")]
    Bridge(#[from] BridgeError),
    #[error("game rule operation failed: {0}")]
    Rule(#[from] cascadia_game::RuleError),
    #[error("I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("JSON failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("audit invariant failed: {0}")]
    Invariant(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FullLegalAuditConfig {
    pub protocol_id: String,
    pub champion_rollouts: usize,
    pub screen_limit: usize,
    pub sentinel_count: usize,
    pub substantial_rollouts: usize,
    pub high_confidence_limit: usize,
    pub high_confidence_rollouts: usize,
    pub audited_completed_turns: Option<Vec<u16>>,
    pub realized_hidden_completed_turns: Vec<u16>,
    pub paid_wipe_determinizations: usize,
    pub paid_wipe_followup_determinizations: usize,
    pub paid_wipe_followup_width: usize,
}

impl Default for FullLegalAuditConfig {
    fn default() -> Self {
        Self {
            protocol_id: FULL_LEGAL_AUDIT_PROTOCOL_ID.to_owned(),
            champion_rollouts: FROZEN_CHAMPION_ROLLOUTS,
            screen_limit: FROZEN_SCREEN_LIMIT,
            sentinel_count: FROZEN_SENTINEL_COUNT,
            substantial_rollouts: FROZEN_SUBSTANTIAL_ROLLOUTS,
            high_confidence_limit: FROZEN_HIGH_CONFIDENCE_LIMIT,
            high_confidence_rollouts: FROZEN_HIGH_CONFIDENCE_ROLLOUTS,
            audited_completed_turns: None,
            realized_hidden_completed_turns: FROZEN_REALIZED_HIDDEN_TURNS.to_vec(),
            paid_wipe_determinizations: FROZEN_PAID_WIPE_DETERMINIZATIONS,
            paid_wipe_followup_determinizations: FROZEN_PAID_WIPE_FOLLOWUP_DETERMINIZATIONS,
            paid_wipe_followup_width: FROZEN_PAID_WIPE_FOLLOWUP_WIDTH,
        }
    }
}

impl FullLegalAuditConfig {
    pub fn validate(&self) -> Result<(), FullLegalAuditError> {
        if self.protocol_id != FULL_LEGAL_AUDIT_PROTOCOL_ID {
            return Err(FullLegalAuditError::Invariant(format!(
                "unexpected protocol id {}",
                self.protocol_id
            )));
        }
        if self.champion_rollouts == 0
            || self.screen_limit == 0
            || self.substantial_rollouts == 0
            || self.high_confidence_limit == 0
            || self.high_confidence_rollouts == 0
        {
            return Err(FullLegalAuditError::Invariant(
                "all action and rollout limits must be positive".to_owned(),
            ));
        }
        if self.high_confidence_limit > self.screen_limit {
            return Err(FullLegalAuditError::Invariant(
                "high-confidence limit cannot exceed the complete-screen limit".to_owned(),
            ));
        }
        if let Some(turns) = &self.audited_completed_turns {
            if turns.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(FullLegalAuditError::Invariant(
                    "audited completed turns must be strictly increasing".to_owned(),
                ));
            }
            if turns.iter().any(|turn| *turn >= 80) {
                return Err(FullLegalAuditError::Invariant(
                    "audited completed turns must be in 0 through 79".to_owned(),
                ));
            }
        }
        if self
            .realized_hidden_completed_turns
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
            || self
                .realized_hidden_completed_turns
                .iter()
                .any(|turn| *turn >= 80)
        {
            return Err(FullLegalAuditError::Invariant(
                "realized-hidden completed turns must be strictly increasing in 0 through 79"
                    .to_owned(),
            ));
        }
        if self.paid_wipe_determinizations > 0
            && (self.paid_wipe_followup_determinizations == 0
                || !(1..=15).contains(&self.paid_wipe_followup_width))
        {
            return Err(FullLegalAuditError::Invariant(
                "enabled paid-wipe diagnostics require positive followup determinizations and a \
                 followup width from 1 through 15"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    pub fn audits_turn(&self, completed_turns: u16) -> bool {
        self.audited_completed_turns
            .as_ref()
            .is_none_or(|turns| turns.binary_search(&completed_turns).is_ok())
    }

    pub fn audits_realized_hidden(&self, completed_turns: u16) -> bool {
        self.realized_hidden_completed_turns
            .binary_search(&completed_turns)
            .is_ok()
    }

    pub fn audits_paid_wipes(&self) -> bool {
        self.paid_wipe_determinizations > 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditProvenance {
    pub worker: String,
    pub source: SourceProvenance,
    pub executable_blake3: String,
    pub model_json_blake3: String,
    pub model_safetensors_blake3: String,
    pub started_unix_seconds: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionPhase {
    Early,
    Middle,
    Late,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedScoreBreakdown {
    pub habitat: [i16; 5],
    pub wildlife: [i16; 5],
    pub nature_tokens: i16,
    pub base_total: i16,
}

impl SignedScoreBreakdown {
    fn between(before: ScoreBreakdown, after: ScoreBreakdown) -> Self {
        Self {
            habitat: std::array::from_fn(|index| {
                after.habitat[index] as i16 - before.habitat[index] as i16
            }),
            wildlife: std::array::from_fn(|index| {
                after.wildlife[index] as i16 - before.wildlife[index] as i16
            }),
            nature_tokens: after.nature_tokens as i16 - before.nature_tokens as i16,
            base_total: after.base_total as i16 - before.base_total as i16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RolloutEstimateRecord {
    pub mean: f64,
    pub stddev: f64,
    pub samples: u32,
}

impl From<&ExactMlxRootEstimate> for RolloutEstimateRecord {
    fn from(estimate: &ExactMlxRootEstimate) -> Self {
        Self {
            mean: estimate.rollout_mean,
            stddev: estimate.rollout_stddev,
            samples: estimate.samples,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionSources {
    pub top_complete_screen: bool,
    pub champion_frontier: bool,
    pub champion_selected: bool,
    pub rank_stratified_sentinel: bool,
    pub substantial_top: bool,
    pub best_champion_frontier: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FullLegalActionRecord {
    pub canonical_index: usize,
    pub canonical_hash: String,
    pub action: TurnAction,
    pub drafted_tile_id: u8,
    pub drafted_wildlife: Wildlife,
    pub same_slot_independent: bool,
    pub exact_resulting_score: ScoreBreakdown,
    pub exact_score_delta: SignedScoreBreakdown,
    pub model_immediate_score: f32,
    pub model_remaining_value: f32,
    pub screen_value: f32,
    pub screen_rank: usize,
    pub visible_wildlife_count: u8,
    pub public_bag_wildlife_count: u8,
    pub uniform_market_survival_proxy: f64,
    pub sources: ActionSources,
    pub champion_frontier_r600: Option<RolloutEstimateRecord>,
    pub substantial_r1200: Option<RolloutEstimateRecord>,
    pub high_confidence_r4800: Option<RolloutEstimateRecord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RegretEstimate {
    pub points: f64,
    pub standard_error_upper_bound: f64,
    pub confidence_95: [f64; 2],
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct AuditStageTimings {
    pub champion_seconds: f64,
    pub enumeration_seconds: f64,
    pub screening_seconds: f64,
    pub substantial_seconds: f64,
    pub high_confidence_seconds: f64,
    pub paid_wipe_seconds: f64,
    pub realized_hidden_seconds: f64,
    pub total_seconds: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RealizedHiddenActionResult {
    pub canonical_hash: String,
    pub final_score: ScoreBreakdown,
    pub terminal_state_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RealizedHiddenFutureDiagnostic {
    pub label: String,
    pub input_hidden_state_blake3: String,
    pub public_winner_hash: String,
    pub realized_winner_hash: String,
    pub actions: Vec<RealizedHiddenActionResult>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PaidWipeOptionResult {
    pub mask: u8,
    pub slots: Vec<MarketSlot>,
    pub slot_count: usize,
    pub wiped_wildlife: [u8; 5],
    pub expected_value: f64,
    pub value_stddev: f64,
    pub samples: usize,
    pub preferred_over_stop_probability: f64,
    pub expected_total_wipes: f64,
    pub expected_token_return_probability: f64,
    pub mean_post_first_wipe_market: [f64; 5],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PaidWipeDiagnostic {
    pub label: String,
    pub initial_nature_tokens: u8,
    pub determinizations: usize,
    pub followup_determinizations: usize,
    pub followup_width: usize,
    pub contingent_policy_calls: usize,
    pub followup_decision_nodes: usize,
    pub followup_options_evaluated: usize,
    pub maximum_wipe_ordinal_considered: usize,
    pub recursive_followup_exercised: bool,
    pub stop_action_hash: String,
    pub stop_value: f64,
    pub options: Vec<PaidWipeOptionResult>,
    pub best_option_mask: u8,
    pub best_expected_value: f64,
    pub expected_gain_over_stop: f64,
    pub paid_wipe_preferred_probability: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FullLegalDecisionAudit {
    pub raw_seed: u64,
    pub completed_turns: u16,
    pub current_player: usize,
    pub personal_turn: u16,
    pub phase: DecisionPhase,
    pub public_state_blake3: String,
    pub staged_public_state_blake3: String,
    pub prelude: MarketPrelude,
    pub current_score: ScoreBreakdown,
    pub public_supply: PublicSupply,
    pub opponent_eligible_wildlife_slots: [u16; 5],
    pub opponent_placed_wildlife: [u16; 5],
    pub action_count: usize,
    pub champion_frontier_count: usize,
    pub substantial_count: usize,
    pub high_confidence_count: usize,
    pub champion_action_hash: String,
    pub best_champion_frontier_hash: String,
    pub best_complete_screen_hash: String,
    pub top_screen_recalled_winner: bool,
    pub champion_regret: RegretEstimate,
    pub champion_frontier_regret: RegretEstimate,
    pub retained_screen_regret: RegretEstimate,
    pub paid_wipe_diagnostic: Option<PaidWipeDiagnostic>,
    pub realized_hidden_future: Option<RealizedHiddenFutureDiagnostic>,
    pub timings: AuditStageTimings,
    pub actions: Vec<FullLegalActionRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FullLegalGameAudit {
    pub raw_seed: u64,
    pub decisions: Vec<FullLegalDecisionAudit>,
    pub final_scores: Vec<ScoreBreakdown>,
    pub final_state_blake3: String,
    #[serde(default)]
    pub public_decision_cache: PublicDecisionCacheDiagnostics,
    pub elapsed_seconds: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicDecisionCacheDiagnostics {
    pub enabled: bool,
    pub requests: usize,
    pub evaluations: usize,
    pub hits: usize,
    pub entries: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SerializableBatchDiagnostics {
    pub neural_batches: u64,
    pub neural_rows: u64,
    pub physical_neural_rows: u64,
    #[serde(default)]
    pub reuse_observed_physical_rows: u64,
    #[serde(default)]
    pub reuse_repeated_physical_rows: u64,
    pub rollout_waves: u64,
    pub rollout_samples: u64,
    pub bootstrapped_samples: u64,
    pub policy_fallbacks: u64,
    pub minimum_batch_rows: usize,
    pub maximum_batch_rows: usize,
    #[serde(default)]
    pub multiplex_search_cohorts: u64,
    #[serde(default)]
    pub multiplex_searches: u64,
    #[serde(default)]
    pub evaluator_requests: u64,
    #[serde(default)]
    pub evaluator_batches: u64,
    #[serde(default)]
    pub evaluator_coalesced_batches: u64,
    #[serde(default)]
    pub evaluator_rows: u64,
    #[serde(default)]
    pub cross_request_rows_observed: u64,
    #[serde(default)]
    pub cross_request_duplicate_rows: u64,
    #[serde(default)]
    pub maximum_evaluator_requests_per_batch: usize,
    #[serde(default)]
    pub maximum_evaluator_rows_per_batch: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage_timings: Option<SerializableBatchStageTimings>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SerializableBatchStageTimings {
    pub rollout_state_initialization_ns: u64,
    pub opponent_advance_ns: u64,
    pub candidate_keying_ns: u64,
    pub template_preparation_ns: u64,
    pub candidate_preparation_ns: u64,
    pub row_assembly_ns: u64,
    pub row_deduplication_ns: u64,
    pub row_materialization_ns: u64,
    pub neural_evaluation_ns: u64,
    pub prediction_postprocess_ns: u64,
    pub action_selection_ns: u64,
    pub terminal_collection_ns: u64,
    pub total_ns: u64,
}

impl From<BatchedNnueStageTimings> for SerializableBatchStageTimings {
    fn from(value: BatchedNnueStageTimings) -> Self {
        Self {
            rollout_state_initialization_ns: value.rollout_state_initialization_ns,
            opponent_advance_ns: value.opponent_advance_ns,
            candidate_keying_ns: value.candidate_keying_ns,
            template_preparation_ns: value.template_preparation_ns,
            candidate_preparation_ns: value.candidate_preparation_ns,
            row_assembly_ns: value.row_assembly_ns,
            row_deduplication_ns: value.row_deduplication_ns,
            row_materialization_ns: value.row_materialization_ns,
            neural_evaluation_ns: value.neural_evaluation_ns,
            prediction_postprocess_ns: value.prediction_postprocess_ns,
            action_selection_ns: value.action_selection_ns,
            terminal_collection_ns: value.terminal_collection_ns,
            total_ns: value.total_ns(),
        }
    }
}

impl From<BatchedNnueDiagnostics> for SerializableBatchDiagnostics {
    fn from(value: BatchedNnueDiagnostics) -> Self {
        let stage_timings = (value.stage_timings.total_ns() != 0)
            .then(|| SerializableBatchStageTimings::from(value.stage_timings));
        Self {
            neural_batches: value.neural_batches,
            neural_rows: value.neural_rows,
            physical_neural_rows: value.physical_neural_rows,
            reuse_observed_physical_rows: value.reuse_observed_physical_rows,
            reuse_repeated_physical_rows: value.reuse_repeated_physical_rows,
            rollout_waves: value.rollout_waves,
            rollout_samples: value.rollout_samples,
            bootstrapped_samples: value.bootstrapped_samples,
            policy_fallbacks: value.policy_fallbacks,
            minimum_batch_rows: value.minimum_batch_rows,
            maximum_batch_rows: value.maximum_batch_rows,
            multiplex_search_cohorts: 0,
            multiplex_searches: 0,
            evaluator_requests: 0,
            evaluator_batches: 0,
            evaluator_coalesced_batches: 0,
            evaluator_rows: 0,
            cross_request_rows_observed: 0,
            cross_request_duplicate_rows: 0,
            maximum_evaluator_requests_per_batch: 0,
            maximum_evaluator_rows_per_batch: 0,
            stage_timings,
        }
    }
}

impl SerializableBatchDiagnostics {
    fn record_multiplex(&mut self, value: ExactMlxMultiplexDiagnostics) {
        self.multiplex_search_cohorts = value.search_cohorts;
        self.multiplex_searches = value.searches;
        self.evaluator_requests = value.evaluator_requests;
        self.evaluator_batches = value.evaluator_batches;
        self.evaluator_coalesced_batches = value.coalesced_batches;
        self.evaluator_rows = value.evaluator_rows;
        self.cross_request_rows_observed = value.cross_request_rows_observed;
        self.cross_request_duplicate_rows = value.cross_request_duplicate_rows;
        self.maximum_evaluator_requests_per_batch = value.maximum_requests_per_batch;
        self.maximum_evaluator_rows_per_batch = value.maximum_rows_per_batch;
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FullLegalAuditSummary {
    pub games: usize,
    pub decisions: usize,
    pub actions_screened: usize,
    pub top_screen_recall: f64,
    pub mean_champion_regret: f64,
    pub mean_champion_frontier_regret: f64,
    pub mean_retained_screen_regret: f64,
    pub elapsed_seconds: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FullLegalAuditShard {
    pub schema_version: u32,
    pub config: FullLegalAuditConfig,
    pub provenance: AuditProvenance,
    pub first_seed: u64,
    pub games_requested: usize,
    pub games: Vec<FullLegalGameAudit>,
    pub summary: FullLegalAuditSummary,
    pub bridge_diagnostics: BridgeDiagnostics,
    pub batch_diagnostics: SerializableBatchDiagnostics,
    pub completed_unix_seconds: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FullLegalAuditShardManifest {
    pub provenance: AuditProvenance,
    pub first_seed: u64,
    pub games: usize,
    pub summary: FullLegalAuditSummary,
    pub bridge_diagnostics: BridgeDiagnostics,
    pub batch_diagnostics: SerializableBatchDiagnostics,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FullLegalAuditMerged {
    pub schema_version: u32,
    pub config: FullLegalAuditConfig,
    pub shards: Vec<FullLegalAuditShardManifest>,
    pub games: Vec<FullLegalGameAudit>,
    pub summary: FullLegalAuditSummary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExactRootQualificationReport {
    pub protocol_id: String,
    pub raw_seed: u64,
    pub completed_turns: u16,
    pub champion_rollouts: usize,
    pub legal_action_count: usize,
    pub champion_frontier_count: usize,
    pub public_state_blake3: String,
    pub original_hidden_state_blake3: String,
    pub redetermined_hidden_state_blake3: String,
    pub complete_prior_hidden_invariant: bool,
    pub champion_path_exact_parity: bool,
    pub arbitrary_root_hidden_invariant: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PaidWipeQualificationReport {
    pub protocol_id: String,
    pub raw_seed: u64,
    pub requested_completed_turns: Option<u16>,
    pub completed_turns: u16,
    pub current_player: usize,
    pub minimum_nature_tokens: u8,
    pub public_state_blake3: String,
    pub original_hidden_state_blake3: String,
    pub redetermined_hidden_state_blake3: String,
    pub diagnostic_blake3: String,
    pub hidden_invariant: bool,
    pub diagnostic: PaidWipeDiagnostic,
}

struct EnumeratedAction {
    action: TurnAction,
    score: ScoreBreakdown,
}

pub fn qualify_exact_root_evaluation(
    teacher: &mut ExactMlxLegacyTeacher,
    raw_seed: u64,
    completed_turns: u16,
    champion_rollouts: usize,
) -> Result<ExactRootQualificationReport, FullLegalAuditError> {
    if completed_turns >= 80 {
        return Err(FullLegalAuditError::Invariant(
            "qualification completed turn must be in 0 through 79".to_owned(),
        ));
    }
    let mut game = GameState::new(GameConfig::research_aaaaa(4)?, GameSeed::from_u64(raw_seed))?;
    while game.completed_turns() < completed_turns {
        let action = teacher.select_action(&game)?;
        game.apply(&action)?;
    }
    let public_state_blake3 = game.public_state().canonical_hash().to_hex().to_string();
    let original_hidden_state_blake3 = game.canonical_hash().to_hex().to_string();
    let mut redetermined = game.clone();
    redetermined.redeterminize_hidden(GameSeed::from_u64(raw_seed ^ 0xa53c_9e71_d4b8_260f));
    let redetermined_hidden_state_blake3 = redetermined.canonical_hash().to_hex().to_string();
    if redetermined.public_state().canonical_hash() != game.public_state().canonical_hash()
        || redetermined_hidden_state_blake3 == original_hidden_state_blake3
    {
        return Err(FullLegalAuditError::Invariant(
            "hidden redetermination did not preserve exactly the public state".to_owned(),
        ));
    }

    let prelude = canonical_prelude(&game);
    let legal_actions = game.legal_turn_actions(&prelude)?;
    let original_priors = teacher.score_action_priors(&game, &legal_actions)?;
    let redetermined_priors = teacher.score_action_priors(&redetermined, &legal_actions)?;
    let complete_prior_hidden_invariant =
        original_priors
            .iter()
            .zip(&redetermined_priors)
            .all(|(left, right)| {
                left.immediate_score.to_bits() == right.immediate_score.to_bits()
                    && left.remaining_value.to_bits() == right.remaining_value.to_bits()
            })
            && original_priors.len() == redetermined_priors.len();
    if !complete_prior_hidden_invariant {
        return Err(FullLegalAuditError::Invariant(
            "complete legal prior screen changed after hidden redetermination".to_owned(),
        ));
    }

    let champion = teacher.select_action_with_estimates(&game)?;
    let mut candidate_order = champion.root_estimates.iter().collect::<Vec<_>>();
    candidate_order.sort_by_key(|estimate| estimate.candidate_index);
    if candidate_order
        .iter()
        .enumerate()
        .any(|(index, estimate)| estimate.candidate_index != index)
    {
        return Err(FullLegalAuditError::Invariant(
            "champion root estimates do not cover a contiguous candidate order".to_owned(),
        ));
    }
    let frontier_actions = candidate_order
        .iter()
        .map(|estimate| estimate.action.clone())
        .collect::<Vec<_>>();
    let replayed =
        teacher.score_actions_with_rollouts(&game, &frontier_actions, champion_rollouts)?;
    let redetermined_replayed =
        teacher.score_actions_with_rollouts(&redetermined, &frontier_actions, champion_rollouts)?;
    let champion_path_exact_parity =
        exact_estimate_sets_equal(&champion.root_estimates, &replayed)?;
    let arbitrary_root_hidden_invariant =
        exact_estimate_sets_equal(&replayed, &redetermined_replayed)?;
    if !champion_path_exact_parity {
        return Err(FullLegalAuditError::Invariant(
            "arbitrary-root evaluation differs from the champion path".to_owned(),
        ));
    }
    if !arbitrary_root_hidden_invariant {
        return Err(FullLegalAuditError::Invariant(
            "arbitrary-root evaluation changed after hidden redetermination".to_owned(),
        ));
    }
    Ok(ExactRootQualificationReport {
        protocol_id: FULL_LEGAL_AUDIT_PROTOCOL_ID.to_owned(),
        raw_seed,
        completed_turns,
        champion_rollouts,
        legal_action_count: legal_actions.len(),
        champion_frontier_count: champion.root_estimates.len(),
        public_state_blake3,
        original_hidden_state_blake3,
        redetermined_hidden_state_blake3,
        complete_prior_hidden_invariant,
        champion_path_exact_parity,
        arbitrary_root_hidden_invariant,
    })
}

pub fn qualify_paid_wipe_hidden_invariance(
    teacher: &mut ExactMlxLegacyTeacher,
    config: &FullLegalAuditConfig,
    raw_seed: u64,
    requested_completed_turns: Option<u16>,
    minimum_nature_tokens: u8,
) -> Result<PaidWipeQualificationReport, FullLegalAuditError> {
    config.validate()?;
    if requested_completed_turns.is_some_and(|turn| turn >= 80) {
        return Err(FullLegalAuditError::Invariant(
            "paid-wipe qualification turn must be in 0 through 79".to_owned(),
        ));
    }
    if minimum_nature_tokens == 0 {
        return Err(FullLegalAuditError::Invariant(
            "paid-wipe qualification requires at least one Nature Token".to_owned(),
        ));
    }

    let mut game = GameState::new(GameConfig::research_aaaaa(4)?, GameSeed::from_u64(raw_seed))?;
    let staged = loop {
        if game.is_game_over() {
            return Err(FullLegalAuditError::Invariant(format!(
                "seed {raw_seed} never reached a qualifying state with at least \
                 {minimum_nature_tokens} Nature Tokens"
            )));
        }
        let at_requested_turn =
            requested_completed_turns.is_none_or(|turn| turn == game.completed_turns());
        if at_requested_turn {
            let prelude = canonical_prelude(&game);
            let staged = game.preview_market_prelude(&prelude)?;
            let tokens = staged.boards()[staged.current_player()].nature_tokens();
            if tokens >= minimum_nature_tokens {
                break staged;
            }
            if requested_completed_turns.is_some() {
                return Err(FullLegalAuditError::Invariant(format!(
                    "seed {raw_seed} turn {} has {tokens} Nature Tokens, fewer than the requested \
                     minimum {minimum_nature_tokens}",
                    game.completed_turns()
                )));
            }
        }
        let action = teacher.select_action(&game)?;
        game.apply(&action)?;
    };

    let public_state_blake3 = staged.public_state().canonical_hash().to_hex().to_string();
    let original_hidden_state_blake3 = staged.canonical_hash().to_hex().to_string();
    let mut redetermined = staged.clone();
    redetermined.redeterminize_hidden(chance_seed(
        &staged,
        b"paid-wipe-hidden-invariance-qualification",
        &[raw_seed, u64::from(staged.completed_turns())],
    ));
    let redetermined_hidden_state_blake3 = redetermined.canonical_hash().to_hex().to_string();
    if redetermined.public_state().canonical_hash() != staged.public_state().canonical_hash()
        || redetermined_hidden_state_blake3 == original_hidden_state_blake3
    {
        return Err(FullLegalAuditError::Invariant(
            "paid-wipe hidden redetermination did not preserve exactly the public state".to_owned(),
        ));
    }

    let diagnostic = evaluate_paid_wipe_diagnostic(teacher, config, &staged)?;
    let redetermined_diagnostic = evaluate_paid_wipe_diagnostic(teacher, config, &redetermined)?;
    let hidden_invariant = diagnostic == redetermined_diagnostic;
    if !hidden_invariant {
        return Err(FullLegalAuditError::Invariant(
            "paid-wipe diagnostic changed after hidden redetermination".to_owned(),
        ));
    }
    let diagnostic_blake3 = {
        let bytes = serde_json::to_vec(&diagnostic)?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"cascadia-v2-paid-wipe-diagnostic-v1");
        hasher.update(&bytes);
        hasher.finalize().to_hex().to_string()
    };

    Ok(PaidWipeQualificationReport {
        protocol_id: FULL_LEGAL_AUDIT_PROTOCOL_ID.to_owned(),
        raw_seed,
        requested_completed_turns,
        completed_turns: staged.completed_turns(),
        current_player: staged.current_player(),
        minimum_nature_tokens,
        public_state_blake3,
        original_hidden_state_blake3,
        redetermined_hidden_state_blake3,
        diagnostic_blake3,
        hidden_invariant,
        diagnostic,
    })
}

fn exact_estimate_sets_equal(
    left: &[ExactMlxRootEstimate],
    right: &[ExactMlxRootEstimate],
) -> Result<bool, FullLegalAuditError> {
    if left.len() != right.len() {
        return Ok(false);
    }
    let mut right_by_hash = BTreeMap::new();
    for estimate in right {
        let hash = canonical_action_hash(&estimate.action)?;
        if right_by_hash.insert(hash.clone(), estimate).is_some() {
            return Err(FullLegalAuditError::Invariant(format!(
                "duplicate estimate identity {hash}"
            )));
        }
    }
    for estimate in left {
        let hash = canonical_action_hash(&estimate.action)?;
        let Some(other) = right_by_hash.get(&hash) else {
            return Ok(false);
        };
        if estimate.candidate_index != other.candidate_index
            || estimate.features != other.features
            || estimate.immediate_score.to_bits() != other.immediate_score.to_bits()
            || estimate.rollout_mean.to_bits() != other.rollout_mean.to_bits()
            || estimate.rollout_stddev.to_bits() != other.rollout_stddev.to_bits()
            || estimate.samples != other.samples
            || estimate.selected != other.selected
        {
            return Ok(false);
        }
    }
    Ok(true)
}

pub fn collect_full_legal_audit_shard(
    teacher: &mut ExactMlxLegacyTeacher,
    config: FullLegalAuditConfig,
    provenance: AuditProvenance,
    first_seed: u64,
    games: usize,
) -> Result<FullLegalAuditShard, FullLegalAuditError> {
    config.validate()?;
    if games == 0 {
        return Err(FullLegalAuditError::Invariant(
            "audit shard must contain at least one game".to_owned(),
        ));
    }
    let started = Instant::now();
    let mut collected = Vec::with_capacity(games);
    for offset in 0..games {
        let raw_seed = first_seed
            .checked_add(offset as u64)
            .ok_or_else(|| FullLegalAuditError::Invariant("seed range overflowed".to_owned()))?;
        collected.push(collect_game(teacher, &config, raw_seed)?);
    }
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let summary = summarize_games(&collected, elapsed_seconds);
    let mut batch_diagnostics: SerializableBatchDiagnostics = teacher.batch_diagnostics.into();
    batch_diagnostics.record_multiplex(teacher.multiplex_diagnostics);
    Ok(FullLegalAuditShard {
        schema_version: 1,
        config,
        provenance,
        first_seed,
        games_requested: games,
        games: collected,
        summary,
        bridge_diagnostics: teacher.diagnostics.clone(),
        batch_diagnostics,
        completed_unix_seconds: unix_seconds(),
    })
}

fn collect_game(
    teacher: &mut ExactMlxLegacyTeacher,
    config: &FullLegalAuditConfig,
    raw_seed: u64,
) -> Result<FullLegalGameAudit, FullLegalAuditError> {
    let started = Instant::now();
    let game_seed = GameSeed::from_u64(raw_seed);
    let mut game = GameState::new(GameConfig::research_aaaaa(4)?, game_seed)?;
    let mut decisions = Vec::new();
    let mut public_decision_cache = PublicDecisionCache::default();
    while !game.is_game_over() {
        let champion_started = Instant::now();
        let champion = if config.audits_turn(game.completed_turns()) {
            let champion = teacher.select_action_with_estimates(&game)?;
            public_decision_cache.record_evaluation(&game, &champion.action)?;
            champion
        } else {
            ExactMlxCollectedDecision {
                action: public_decision_cache.select_action(teacher, &game)?,
                rollout_value_samples: Vec::new(),
                root_estimates: Vec::new(),
            }
        };
        let champion_seconds = champion_started.elapsed().as_secs_f64();
        if config.audits_turn(game.completed_turns()) {
            decisions.push(audit_decision(
                teacher,
                config,
                raw_seed,
                &game,
                &champion,
                champion_seconds,
                &mut public_decision_cache,
            )?);
        }
        game.apply(&champion.action)?;
    }
    if game.completed_turns() != 80 {
        return Err(FullLegalAuditError::Invariant(format!(
            "seed {raw_seed} completed {} turns instead of 80",
            game.completed_turns()
        )));
    }
    let public_decision_cache = public_decision_cache.diagnostics();
    if std::env::var("CASCADIA_NNUE_STAGE_TIMINGS")
        .ok()
        .is_some_and(|value| !value.is_empty() && value != "0")
    {
        eprintln!(
            "{}",
            serde_json::json!({
                "event": "full_legal_public_decision_cache_diagnostics",
                "enabled": public_decision_cache.enabled,
                "requests": public_decision_cache.requests,
                "evaluations": public_decision_cache.evaluations,
                "hits": public_decision_cache.hits,
                "entries": public_decision_cache.entries,
            })
        );
    }
    Ok(FullLegalGameAudit {
        raw_seed,
        decisions,
        final_scores: score_game(&game),
        final_state_blake3: game.canonical_hash().to_hex().to_string(),
        public_decision_cache,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    })
}

struct PublicDecisionLadder {
    prelude: MarketPrelude,
    staged: GameState,
    current_score: ScoreBreakdown,
    public_supply: PublicSupply,
    actions: Vec<FullLegalActionRecord>,
    champion_action_hash: String,
    best_frontier_hash: String,
    best_complete_screen_hash: String,
    winner_index: usize,
    champion_frontier_count: usize,
    substantial_count: usize,
    high: Vec<ExactMlxRootEstimate>,
    champion_regret: RegretEstimate,
    champion_frontier_regret: RegretEstimate,
    retained_screen_regret: Option<RegretEstimate>,
    enumeration_seconds: f64,
    screening_seconds: f64,
    substantial_seconds: f64,
    high_confidence_seconds: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FullLegalOracleDecision {
    pub action: TurnAction,
    pub champion_action: TurnAction,
    pub champion_action_hash: String,
    pub selected_action_hash: String,
    pub action_count: usize,
    pub champion_frontier_count: usize,
    pub substantial_count: usize,
    pub high_confidence_count: usize,
    pub top_screen_recalled_winner: bool,
    pub champion_regret: RegretEstimate,
    pub champion_seconds: f64,
    pub enumeration_seconds: f64,
    pub screening_seconds: f64,
    pub substantial_seconds: f64,
    pub high_confidence_seconds: f64,
    pub total_seconds: f64,
}

pub fn select_full_legal_oracle_action(
    teacher: &mut ExactMlxLegacyTeacher,
    config: &FullLegalAuditConfig,
    game: &GameState,
) -> Result<FullLegalOracleDecision, FullLegalAuditError> {
    config.validate()?;
    let started = Instant::now();
    let champion_started = Instant::now();
    let champion = teacher.select_action_with_estimates(game)?;
    let champion_seconds = champion_started.elapsed().as_secs_f64();
    let ladder = evaluate_public_decision_ladder(teacher, config, game, &champion)?;
    let action = ladder.actions[ladder.winner_index].action.clone();
    Ok(FullLegalOracleDecision {
        action,
        champion_action: champion.action,
        champion_action_hash: ladder.champion_action_hash,
        selected_action_hash: ladder.best_complete_screen_hash,
        action_count: ladder.actions.len(),
        champion_frontier_count: ladder.champion_frontier_count,
        substantial_count: ladder.substantial_count,
        high_confidence_count: ladder.high.len(),
        top_screen_recalled_winner: ladder.actions[ladder.winner_index]
            .sources
            .top_complete_screen,
        champion_regret: ladder.champion_regret,
        champion_seconds,
        enumeration_seconds: ladder.enumeration_seconds,
        screening_seconds: ladder.screening_seconds,
        substantial_seconds: ladder.substantial_seconds,
        high_confidence_seconds: ladder.high_confidence_seconds,
        total_seconds: started.elapsed().as_secs_f64(),
    })
}

fn evaluate_public_decision_ladder(
    teacher: &mut ExactMlxLegacyTeacher,
    config: &FullLegalAuditConfig,
    game: &GameState,
    champion: &ExactMlxCollectedDecision,
) -> Result<PublicDecisionLadder, FullLegalAuditError> {
    let prelude = canonical_prelude(game);
    let staged = game.preview_market_prelude(&prelude)?;
    let current_score = score_board(
        &game.boards()[game.current_player()],
        game.config().scoring_cards,
    );

    let enumeration_started = Instant::now();
    let enumerated = enumerate_exact_actions(game, &prelude)?;
    let enumeration_seconds = enumeration_started.elapsed().as_secs_f64();
    if enumerated.is_empty() {
        return Err(FullLegalAuditError::Invariant(
            "active decision produced no legal actions".to_owned(),
        ));
    }

    let screening_started = Instant::now();
    let canonical_actions = enumerated
        .iter()
        .map(|candidate| candidate.action.clone())
        .collect::<Vec<_>>();
    let priors = teacher.score_action_priors(game, &canonical_actions)?;
    if priors.len() != enumerated.len() {
        return Err(FullLegalAuditError::Invariant(format!(
            "screen returned {} rows for {} legal actions",
            priors.len(),
            enumerated.len()
        )));
    }
    let public_supply = game.public_supply();
    let mut actions = Vec::with_capacity(enumerated.len());
    let mut by_hash = BTreeMap::new();
    for (canonical_index, (enumerated, prior)) in enumerated.into_iter().zip(priors).enumerate() {
        let canonical_hash = canonical_action_hash(&enumerated.action)?;
        if by_hash
            .insert(canonical_hash.clone(), canonical_index)
            .is_some()
        {
            return Err(FullLegalAuditError::Invariant(format!(
                "duplicate canonical action hash {canonical_hash}"
            )));
        }
        let (tile_id, wildlife) = drafted_components(&staged, &enumerated.action)?;
        let visible_wildlife_count = staged
            .market()
            .wildlife
            .iter()
            .flatten()
            .filter(|shown| **shown == wildlife)
            .count() as u8;
        let public_bag_wildlife_count = public_supply.wildlife_bag[wildlife as usize];
        actions.push(FullLegalActionRecord {
            canonical_index,
            canonical_hash,
            same_slot_independent: is_same_slot_independent(&enumerated.action),
            exact_score_delta: SignedScoreBreakdown::between(current_score, enumerated.score),
            exact_resulting_score: enumerated.score,
            action: enumerated.action,
            drafted_tile_id: tile_id,
            drafted_wildlife: wildlife,
            model_immediate_score: prior.immediate_score,
            model_remaining_value: prior.remaining_value,
            screen_value: prior.immediate_score + prior.remaining_value,
            screen_rank: 0,
            visible_wildlife_count,
            public_bag_wildlife_count,
            uniform_market_survival_proxy: uniform_market_survival_proxy(
                visible_wildlife_count,
                public_bag_wildlife_count,
                public_supply
                    .wildlife_bag
                    .iter()
                    .map(|count| u16::from(*count))
                    .sum(),
            ),
            sources: ActionSources::default(),
            champion_frontier_r600: None,
            substantial_r1200: None,
            high_confidence_r4800: None,
        });
    }
    let ranked_indices = screen_ranked_indices(&actions);
    for (rank, index) in ranked_indices.iter().copied().enumerate() {
        actions[index].screen_rank = rank + 1;
        if rank < config.screen_limit {
            actions[index].sources.top_complete_screen = true;
        }
    }
    let sentinel_indices = stratified_sentinel_indices(
        ranked_indices.len(),
        config.screen_limit,
        config.sentinel_count,
    );
    for rank_index in &sentinel_indices {
        actions[ranked_indices[*rank_index]]
            .sources
            .rank_stratified_sentinel = true;
    }
    let champion_action_hash = canonical_action_hash(&champion.action)?;
    let champion_index = action_index(&by_hash, &champion_action_hash)?;
    actions[champion_index].sources.champion_selected = true;
    for estimate in &champion.root_estimates {
        let hash = canonical_action_hash(&estimate.action)?;
        let index = action_index(&by_hash, &hash)?;
        actions[index].sources.champion_frontier = true;
        actions[index].champion_frontier_r600 = Some(estimate.into());
    }
    if champion.root_estimates.is_empty()
        || champion
            .root_estimates
            .iter()
            .filter(|estimate| estimate.selected)
            .count()
            != 1
    {
        return Err(FullLegalAuditError::Invariant(
            "audited champion decision must expose one selected root estimate".to_owned(),
        ));
    }
    if champion
        .root_estimates
        .iter()
        .find(|estimate| estimate.selected)
        .is_none_or(|estimate| estimate.action != champion.action)
    {
        return Err(FullLegalAuditError::Invariant(
            "champion root estimate selection disagrees with played action".to_owned(),
        ));
    }
    let screening_seconds = screening_started.elapsed().as_secs_f64();

    let substantial_indices = substantial_union_indices(
        &actions,
        &ranked_indices,
        config.screen_limit,
        &sentinel_indices,
        champion_index,
    );
    let substantial_actions = substantial_indices
        .iter()
        .map(|index| actions[*index].action.clone())
        .collect::<Vec<_>>();
    let substantial_started = Instant::now();
    let substantial = teacher.score_actions_with_rollouts_in_domain_and_coupling(
        game,
        &substantial_actions,
        config.substantial_rollouts,
        b"full-legal-audit-substantial-r1200",
        RolloutSeedCoupling::CommonWithinRound,
    )?;
    attach_estimates(
        &mut actions,
        &by_hash,
        &substantial,
        EstimateStage::Substantial,
    )?;
    let substantial_seconds = substantial_started.elapsed().as_secs_f64();

    let substantial_best_frontier_hash = substantial
        .iter()
        .filter_map(|estimate| {
            let hash = canonical_action_hash(&estimate.action).ok()?;
            let index = *by_hash.get(&hash)?;
            actions[index]
                .sources
                .champion_frontier
                .then_some((hash, estimate.rollout_mean))
        })
        .max_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| right.0.cmp(&left.0))
        })
        .map(|(hash, _)| hash)
        .ok_or_else(|| {
            FullLegalAuditError::Invariant(
                "substantial set contains no champion-frontier action".to_owned(),
            )
        })?;
    let substantial_best_frontier_index = action_index(&by_hash, &substantial_best_frontier_hash)?;

    let mut high_indices = Vec::new();
    for estimate in substantial.iter().take(config.high_confidence_limit) {
        let hash = canonical_action_hash(&estimate.action)?;
        let index = action_index(&by_hash, &hash)?;
        actions[index].sources.substantial_top = true;
        push_unique(&mut high_indices, index);
    }
    push_unique(&mut high_indices, champion_index);
    push_unique(&mut high_indices, substantial_best_frontier_index);
    let high_actions = high_indices
        .iter()
        .map(|index| actions[*index].action.clone())
        .collect::<Vec<_>>();
    let high_started = Instant::now();
    let high = teacher.score_actions_with_rollouts_in_domain_and_coupling(
        game,
        &high_actions,
        config.high_confidence_rollouts,
        b"full-legal-audit-high-confidence-r4800",
        RolloutSeedCoupling::CommonWithinRound,
    )?;
    attach_estimates(&mut actions, &by_hash, &high, EstimateStage::HighConfidence)?;
    let high_confidence_seconds = high_started.elapsed().as_secs_f64();

    let winner = high.first().ok_or_else(|| {
        FullLegalAuditError::Invariant("high-confidence evaluation returned no action".to_owned())
    })?;
    let best_complete_screen_hash = canonical_action_hash(&winner.action)?;
    let winner_index = action_index(&by_hash, &best_complete_screen_hash)?;
    let champion_high = high_estimate_for_hash(&high, &champion_action_hash)?;
    let (best_frontier_hash, frontier_high) = high
        .iter()
        .filter_map(|estimate| {
            let hash = canonical_action_hash(&estimate.action).ok()?;
            let index = *by_hash.get(&hash)?;
            actions[index]
                .sources
                .champion_frontier
                .then_some((hash, estimate))
        })
        .max_by(|left, right| {
            left.1
                .rollout_mean
                .total_cmp(&right.1.rollout_mean)
                .then_with(|| right.0.cmp(&left.0))
        })
        .ok_or_else(|| {
            FullLegalAuditError::Invariant(
                "high-confidence set contains no champion-frontier action".to_owned(),
            )
        })?;
    let best_frontier_index = action_index(&by_hash, &best_frontier_hash)?;
    actions[best_frontier_index].sources.best_champion_frontier = true;
    let retained_high = high
        .iter()
        .filter_map(|estimate| {
            let hash = canonical_action_hash(&estimate.action).ok()?;
            let index = *by_hash.get(&hash)?;
            actions[index]
                .sources
                .top_complete_screen
                .then_some(estimate)
        })
        .max_by(|left, right| left.rollout_mean.total_cmp(&right.rollout_mean));
    let champion_regret = regret_estimate(winner, champion_high);
    let champion_frontier_regret = regret_estimate(winner, frontier_high);
    let retained_screen_regret = optional_regret_estimate(winner, retained_high);

    Ok(PublicDecisionLadder {
        prelude,
        staged,
        current_score,
        public_supply,
        actions,
        champion_action_hash,
        best_frontier_hash,
        best_complete_screen_hash,
        winner_index,
        champion_frontier_count: champion.root_estimates.len(),
        substantial_count: substantial.len(),
        high,
        champion_regret,
        champion_frontier_regret,
        retained_screen_regret,
        enumeration_seconds,
        screening_seconds,
        substantial_seconds,
        high_confidence_seconds,
    })
}

fn audit_decision(
    teacher: &mut ExactMlxLegacyTeacher,
    config: &FullLegalAuditConfig,
    raw_seed: u64,
    game: &GameState,
    champion: &ExactMlxCollectedDecision,
    champion_seconds: f64,
    public_decision_cache: &mut PublicDecisionCache,
) -> Result<FullLegalDecisionAudit, FullLegalAuditError> {
    let total_started = Instant::now();
    let ladder = evaluate_public_decision_ladder(teacher, config, game, champion)?;

    let paid_wipe_started = Instant::now();
    let paid_wipe_diagnostic = if config.audits_paid_wipes()
        && ladder.staged.boards()[ladder.staged.current_player()].nature_tokens() > 0
    {
        Some(evaluate_paid_wipe_diagnostic(
            teacher,
            config,
            &ladder.staged,
        )?)
    } else {
        None
    };
    let paid_wipe_seconds = paid_wipe_started.elapsed().as_secs_f64();

    let hidden_started = Instant::now();
    let realized_hidden_future = if config.audits_realized_hidden(game.completed_turns()) {
        Some(realized_hidden_future_diagnostic(
            teacher,
            game,
            &ladder.high,
            &ladder.best_complete_screen_hash,
            &ladder.champion_action_hash,
            public_decision_cache,
        )?)
    } else {
        None
    };
    let realized_hidden_seconds = hidden_started.elapsed().as_secs_f64();

    let PublicDecisionLadder {
        prelude,
        staged,
        current_score,
        public_supply,
        actions,
        champion_action_hash,
        best_frontier_hash,
        best_complete_screen_hash,
        winner_index,
        champion_frontier_count,
        substantial_count,
        high,
        champion_regret,
        champion_frontier_regret,
        retained_screen_regret,
        enumeration_seconds,
        screening_seconds,
        substantial_seconds,
        high_confidence_seconds,
    } = ladder;
    let (opponent_slots, opponent_wildlife) = opponent_wildlife_metrics(game);
    let personal_turn = game.completed_turns() / game.boards().len() as u16 + 1;
    Ok(FullLegalDecisionAudit {
        raw_seed,
        completed_turns: game.completed_turns(),
        current_player: game.current_player(),
        personal_turn,
        phase: decision_phase(personal_turn),
        public_state_blake3: game.public_state().canonical_hash().to_hex().to_string(),
        staged_public_state_blake3: staged.public_state().canonical_hash().to_hex().to_string(),
        prelude,
        current_score,
        public_supply,
        opponent_eligible_wildlife_slots: opponent_slots,
        opponent_placed_wildlife: opponent_wildlife,
        action_count: actions.len(),
        champion_frontier_count,
        substantial_count,
        high_confidence_count: high.len(),
        champion_action_hash,
        best_champion_frontier_hash: best_frontier_hash,
        best_complete_screen_hash,
        top_screen_recalled_winner: actions[winner_index].sources.top_complete_screen,
        champion_regret,
        champion_frontier_regret,
        retained_screen_regret: retained_screen_regret.ok_or_else(|| {
            FullLegalAuditError::Invariant(
                "audited high-confidence set contains no retained screen action".to_owned(),
            )
        })?,
        paid_wipe_diagnostic,
        realized_hidden_future,
        timings: AuditStageTimings {
            champion_seconds,
            enumeration_seconds,
            screening_seconds,
            substantial_seconds,
            high_confidence_seconds,
            paid_wipe_seconds,
            realized_hidden_seconds,
            total_seconds: total_started.elapsed().as_secs_f64() + champion_seconds,
        },
        actions,
    })
}

#[derive(Debug, Clone, PartialEq)]
struct CompleteScreenChoice {
    action_hash: String,
    value: f64,
    earns_nature_token: bool,
}

#[derive(Debug, Clone)]
struct ContingentWipeOutcome {
    choice: CompleteScreenChoice,
    additional_wipes: usize,
}

#[derive(Debug, Clone)]
struct CachedCompleteScreen {
    context_index: usize,
    market: Market,
    choice: CompleteScreenChoice,
}

#[derive(Debug, Clone)]
struct CachedPublicDecision {
    public_state_bytes: Vec<u8>,
    action: TurnAction,
}

struct PendingPublicDecision {
    hash: [u8; 32],
    public_state_bytes: Vec<u8>,
}

enum PublicDecisionLookup {
    Hit(TurnAction),
    Miss(PendingPublicDecision),
}

#[derive(Debug, Default)]
struct PublicDecisionCache {
    requests: usize,
    evaluations: usize,
    hits: usize,
    buckets: HashMap<[u8; 32], Vec<CachedPublicDecision>>,
}

impl PublicDecisionCache {
    fn encoded_public_state(public_state: &PublicGameState) -> ([u8; 32], Vec<u8>) {
        let bytes = public_state.canonical_bytes();
        (*blake3::hash(&bytes).as_bytes(), bytes)
    }

    fn lookup_bucket(&self, hash: &[u8; 32], public_state_bytes: &[u8]) -> Option<TurnAction> {
        self.buckets.get(hash).and_then(|bucket| {
            bucket
                .iter()
                .find(|entry| entry.public_state_bytes == public_state_bytes)
                .map(|entry| entry.action.clone())
        })
    }

    fn insert_or_validate(
        &mut self,
        hash: [u8; 32],
        public_state_bytes: Vec<u8>,
        action: &TurnAction,
    ) -> Result<(), FullLegalAuditError> {
        let bucket = self.buckets.entry(hash).or_default();
        if let Some(entry) = bucket
            .iter()
            .find(|entry| entry.public_state_bytes == public_state_bytes)
        {
            if entry.action != *action {
                return Err(FullLegalAuditError::Invariant(
                    "exact public-state decision cache observed two different actions".to_owned(),
                ));
            }
            return Ok(());
        }
        bucket.push(CachedPublicDecision {
            public_state_bytes,
            action: action.clone(),
        });
        Ok(())
    }

    fn record_evaluation(
        &mut self,
        game: &GameState,
        action: &TurnAction,
    ) -> Result<(), FullLegalAuditError> {
        self.requests += 1;
        self.evaluations += 1;
        let public_state = game.public_state();
        let (hash, bytes) = Self::encoded_public_state(&public_state);
        self.insert_or_validate(hash, bytes, action)
    }

    fn request_cacheable(&mut self, game: &GameState) -> PublicDecisionLookup {
        self.requests += 1;
        let public_state = game.public_state();
        let (hash, public_state_bytes) = Self::encoded_public_state(&public_state);
        if let Some(action) = self.lookup_bucket(&hash, &public_state_bytes) {
            self.hits += 1;
            PublicDecisionLookup::Hit(action)
        } else {
            PublicDecisionLookup::Miss(PendingPublicDecision {
                hash,
                public_state_bytes,
            })
        }
    }

    fn request_without_storage(&mut self) {
        self.requests += 1;
    }

    fn record_requested_evaluation(
        &mut self,
        pending: Option<PendingPublicDecision>,
        action: &TurnAction,
    ) -> Result<(), FullLegalAuditError> {
        self.evaluations += 1;
        if let Some(pending) = pending {
            self.insert_or_validate(pending.hash, pending.public_state_bytes, action)?;
        }
        Ok(())
    }

    fn select_action(
        &mut self,
        teacher: &mut ExactMlxLegacyTeacher,
        game: &GameState,
    ) -> Result<TurnAction, FullLegalAuditError> {
        match self.request_cacheable(game) {
            PublicDecisionLookup::Hit(action) => Ok(action),
            PublicDecisionLookup::Miss(pending) => {
                let action = teacher.select_action(game)?;
                self.record_requested_evaluation(Some(pending), &action)?;
                Ok(action)
            }
        }
    }

    fn entries(&self) -> usize {
        self.buckets.values().map(Vec::len).sum()
    }

    fn diagnostics(&self) -> PublicDecisionCacheDiagnostics {
        PublicDecisionCacheDiagnostics {
            enabled: true,
            requests: self.requests,
            evaluations: self.evaluations,
            hits: self.hits,
            entries: self.entries(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CompleteScreenContext {
    config: GameConfig,
    boards: Vec<Board>,
    current_player: usize,
    completed_turns: u16,
}

impl CompleteScreenContext {
    fn from_public_state(public_state: &PublicGameState) -> Self {
        Self {
            config: public_state.config(),
            boards: public_state.boards().to_vec(),
            current_player: public_state.current_player(),
            completed_turns: public_state.completed_turns(),
        }
    }

    fn matches(&self, public_state: &PublicGameState) -> bool {
        self.config == public_state.config()
            && self.boards.as_slice() == public_state.boards()
            && self.current_player == public_state.current_player()
            && self.completed_turns == public_state.completed_turns()
    }
}

#[derive(Debug, Default)]
struct CompleteScreenCache {
    requests: usize,
    evaluations: usize,
    hits: usize,
    contexts: Vec<CompleteScreenContext>,
    buckets: HashMap<String, Vec<CachedCompleteScreen>>,
}

impl CompleteScreenCache {
    fn lookup_bucket(
        &self,
        hash: &str,
        public_state: &PublicGameState,
    ) -> Option<CompleteScreenChoice> {
        let context_index = self
            .contexts
            .iter()
            .position(|context| context.matches(public_state))?;
        self.buckets.get(hash).and_then(|bucket| {
            bucket
                .iter()
                .find(|entry| {
                    entry.context_index == context_index && entry.market == *public_state.market()
                })
                .map(|entry| entry.choice.clone())
        })
    }

    fn insert(
        &mut self,
        hash: String,
        public_state: PublicGameState,
        choice: CompleteScreenChoice,
    ) {
        let context_index = self
            .contexts
            .iter()
            .position(|context| context.matches(&public_state))
            .unwrap_or_else(|| {
                self.contexts
                    .push(CompleteScreenContext::from_public_state(&public_state));
                self.contexts.len() - 1
            });
        self.buckets
            .entry(hash)
            .or_default()
            .push(CachedCompleteScreen {
                context_index,
                market: public_state.market().clone(),
                choice,
            });
    }

    fn entries(&self) -> usize {
        self.buckets.values().map(Vec::len).sum()
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct PaidWipeInstrumentation {
    contingent_policy_calls: usize,
    followup_decision_nodes: usize,
    followup_options_evaluated: usize,
    maximum_wipe_ordinal_considered: usize,
}

fn evaluate_paid_wipe_diagnostic(
    teacher: &mut ExactMlxLegacyTeacher,
    config: &FullLegalAuditConfig,
    staged_after_free_prelude: &GameState,
) -> Result<PaidWipeDiagnostic, FullLegalAuditError> {
    let mut complete_screen_cache = CompleteScreenCache::default();
    let initial_nature_tokens = staged_after_free_prelude.boards()
        [staged_after_free_prelude.current_player()]
    .nature_tokens();
    let stop = best_complete_screen_choice_cached(
        teacher,
        staged_after_free_prelude,
        &mut complete_screen_cache,
    )?;
    let wipes = staged_after_free_prelude.legal_wildlife_wipes();
    if wipes.len() != 15 {
        return Err(FullLegalAuditError::Invariant(format!(
            "token-bearing state exposed {} paid wipes instead of 15",
            wipes.len()
        )));
    }
    let mut option_values = vec![Vec::with_capacity(config.paid_wipe_determinizations); 15];
    let mut option_total_wipes = vec![Vec::with_capacity(config.paid_wipe_determinizations); 15];
    let mut option_token_returns = vec![Vec::with_capacity(config.paid_wipe_determinizations); 15];
    let mut option_market_totals = vec![[0_u64; 5]; 15];
    let mut instrumentation = PaidWipeInstrumentation::default();

    for sample in 0..config.paid_wipe_determinizations {
        let mut redetermined = staged_after_free_prelude.clone();
        redetermined.redeterminize_hidden(chance_seed(
            staged_after_free_prelude,
            b"paid-wipe-root-common",
            &[sample as u64],
        ));
        for (option_index, wipe) in wipes.iter().enumerate() {
            let after_first = redetermined.preview_market_prelude(&MarketPrelude {
                replace_three_of_a_kind: false,
                wildlife_wipes: vec![wipe.clone()],
            })?;
            for wildlife in after_first.market().wildlife.iter().flatten() {
                option_market_totals[option_index][*wildlife as usize] += 1;
            }
            let context = ((sample as u64) << 8) | u64::from(wipe_mask(wipe));
            let outcome = contingent_wipe_policy(
                teacher,
                config,
                &after_first,
                context,
                0,
                &mut instrumentation,
                &mut complete_screen_cache,
            )?;
            option_values[option_index].push(outcome.choice.value);
            option_total_wipes[option_index].push(1 + outcome.additional_wipes);
            option_token_returns[option_index].push(outcome.choice.earns_nature_token);
        }
    }

    let mut options = Vec::with_capacity(15);
    for (option_index, wipe) in wipes.iter().enumerate() {
        let values = &option_values[option_index];
        let expected_value = mean(values);
        let value_stddev = population_stddev(values, expected_value);
        let mut wiped_wildlife = [0_u8; 5];
        for slot in &wipe.slots {
            let wildlife =
                staged_after_free_prelude.market().wildlife[slot.index()].ok_or_else(|| {
                    FullLegalAuditError::Invariant(
                        "paid wipe references an empty wildlife slot".to_owned(),
                    )
                })?;
            wiped_wildlife[wildlife as usize] += 1;
        }
        options.push(PaidWipeOptionResult {
            mask: wipe_mask(wipe),
            slots: wipe.slots.clone(),
            slot_count: wipe.slots.len(),
            wiped_wildlife,
            expected_value,
            value_stddev,
            samples: values.len(),
            preferred_over_stop_probability: ratio_count(
                values.iter().filter(|value| **value > stop.value).count(),
                values.len(),
            ),
            expected_total_wipes: option_total_wipes[option_index]
                .iter()
                .map(|count| *count as f64)
                .sum::<f64>()
                / option_total_wipes[option_index].len() as f64,
            expected_token_return_probability: ratio_count(
                option_token_returns[option_index]
                    .iter()
                    .filter(|returned| **returned)
                    .count(),
                option_token_returns[option_index].len(),
            ),
            mean_post_first_wipe_market: std::array::from_fn(|wildlife| {
                option_market_totals[option_index][wildlife] as f64
                    / config.paid_wipe_determinizations as f64
            }),
        });
    }
    options.sort_by_key(|option| option.mask);
    let best = options
        .iter()
        .max_by(|left, right| {
            left.expected_value
                .total_cmp(&right.expected_value)
                .then_with(|| right.mask.cmp(&left.mask))
        })
        .ok_or_else(|| {
            FullLegalAuditError::Invariant("paid-wipe diagnostic produced no options".to_owned())
        })?;
    let paid_wipe_preferred_probability = (0..config.paid_wipe_determinizations)
        .filter(|sample| {
            option_values
                .iter()
                .any(|values| values[*sample] > stop.value)
        })
        .count() as f64
        / config.paid_wipe_determinizations as f64;
    let best_option_mask = best.mask;
    let best_expected_value = best.expected_value;
    let diagnostic = PaidWipeDiagnostic {
        label: "public-chance-paid-wipe-contingent-screen-v1".to_owned(),
        initial_nature_tokens,
        determinizations: config.paid_wipe_determinizations,
        followup_determinizations: config.paid_wipe_followup_determinizations,
        followup_width: config.paid_wipe_followup_width,
        contingent_policy_calls: instrumentation.contingent_policy_calls,
        followup_decision_nodes: instrumentation.followup_decision_nodes,
        followup_options_evaluated: instrumentation.followup_options_evaluated,
        maximum_wipe_ordinal_considered: instrumentation.maximum_wipe_ordinal_considered,
        recursive_followup_exercised: instrumentation.followup_decision_nodes > 0,
        stop_action_hash: stop.action_hash,
        stop_value: stop.value,
        options,
        best_option_mask,
        best_expected_value,
        expected_gain_over_stop: best_expected_value - stop.value,
        paid_wipe_preferred_probability,
    };
    if std::env::var("CASCADIA_NNUE_STAGE_TIMINGS")
        .ok()
        .is_some_and(|value| !value.is_empty() && value != "0")
    {
        eprintln!(
            "{}",
            serde_json::json!({
                "event": "paid_screen_cache_diagnostics",
                "requests": complete_screen_cache.requests,
                "evaluations": complete_screen_cache.evaluations,
                "hits": complete_screen_cache.hits,
                "entries": complete_screen_cache.entries(),
                "contexts": complete_screen_cache.contexts.len(),
            })
        );
    }
    Ok(diagnostic)
}

fn contingent_wipe_policy(
    teacher: &mut ExactMlxLegacyTeacher,
    config: &FullLegalAuditConfig,
    state: &GameState,
    context: u64,
    depth: usize,
    instrumentation: &mut PaidWipeInstrumentation,
    complete_screen_cache: &mut CompleteScreenCache,
) -> Result<ContingentWipeOutcome, FullLegalAuditError> {
    instrumentation.contingent_policy_calls += 1;
    let stop = best_complete_screen_choice_cached(teacher, state, complete_screen_cache)?;
    if state.boards()[state.current_player()].nature_tokens() == 0 {
        return Ok(ContingentWipeOutcome {
            choice: stop,
            additional_wipes: 0,
        });
    }
    let candidates = ranked_followup_wipes(state, config.paid_wipe_followup_width)?;
    instrumentation.followup_decision_nodes += 1;
    instrumentation.followup_options_evaluated += candidates.len();
    instrumentation.maximum_wipe_ordinal_considered = instrumentation
        .maximum_wipe_ordinal_considered
        .max(depth + 2);
    let mut expected = Vec::with_capacity(candidates.len());
    for wipe in &candidates {
        let mut values = Vec::with_capacity(config.paid_wipe_followup_determinizations);
        for sample in 0..config.paid_wipe_followup_determinizations {
            let mut planning = state.clone();
            planning.redeterminize_hidden(chance_seed(
                state,
                b"paid-wipe-followup-planning",
                &[
                    context,
                    depth as u64,
                    u64::from(wipe_mask(wipe)),
                    sample as u64,
                ],
            ));
            let after = planning.preview_market_prelude(&MarketPrelude {
                replace_three_of_a_kind: false,
                wildlife_wipes: vec![wipe.clone()],
            })?;
            values.push(
                best_complete_screen_choice_cached(teacher, &after, complete_screen_cache)?.value,
            );
        }
        expected.push((wipe, mean(&values)));
    }
    let Some((chosen, chosen_value)) = expected.into_iter().max_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| wipe_mask(right.0).cmp(&wipe_mask(left.0)))
    }) else {
        return Ok(ContingentWipeOutcome {
            choice: stop,
            additional_wipes: 0,
        });
    };
    if chosen_value <= stop.value {
        return Ok(ContingentWipeOutcome {
            choice: stop,
            additional_wipes: 0,
        });
    }

    let mut realization = state.clone();
    realization.redeterminize_hidden(chance_seed(
        state,
        b"paid-wipe-followup-realization",
        &[context, depth as u64, u64::from(wipe_mask(chosen))],
    ));
    let after = realization.preview_market_prelude(&MarketPrelude {
        replace_three_of_a_kind: false,
        wildlife_wipes: vec![chosen.clone()],
    })?;
    let next_context = context
        .rotate_left(17)
        .wrapping_add(u64::from(wipe_mask(chosen)))
        .wrapping_add(depth as u64);
    let mut continued = contingent_wipe_policy(
        teacher,
        config,
        &after,
        next_context,
        depth + 1,
        instrumentation,
        complete_screen_cache,
    )?;
    continued.additional_wipes += 1;
    Ok(continued)
}

fn best_complete_screen_choice(
    teacher: &mut ExactMlxLegacyTeacher,
    state: &GameState,
) -> Result<CompleteScreenChoice, FullLegalAuditError> {
    let prelude = MarketPrelude::default();
    let enumerated = enumerate_exact_actions(state, &prelude)?;
    let actions = enumerated
        .iter()
        .map(|candidate| candidate.action.clone())
        .collect::<Vec<_>>();
    let priors = teacher.score_action_priors(state, &actions)?;
    let mut best: Option<(usize, String, f32)> = None;
    for (index, (candidate, prior)) in enumerated.iter().zip(priors).enumerate() {
        let hash = canonical_action_hash(&candidate.action)?;
        let value = prior.immediate_score + prior.remaining_value;
        if best.as_ref().is_none_or(|(_, best_hash, best_value)| {
            value > *best_value || (value == *best_value && hash < *best_hash)
        }) {
            best = Some((index, hash, value));
        }
    }
    let (index, action_hash, value) = best.ok_or_else(|| {
        FullLegalAuditError::Invariant(
            "complete screen produced no action for a paid-wipe branch".to_owned(),
        )
    })?;
    Ok(CompleteScreenChoice {
        action_hash,
        value: f64::from(value),
        earns_nature_token: action_earns_nature_token(state, &enumerated[index].action)?,
    })
}

fn best_complete_screen_choice_cached(
    teacher: &mut ExactMlxLegacyTeacher,
    state: &GameState,
    cache: &mut CompleteScreenCache,
) -> Result<CompleteScreenChoice, FullLegalAuditError> {
    cache.requests += 1;
    let public_state = state.public_state();
    let hash = public_state.canonical_hash().to_hex().to_string();
    if let Some(choice) = cache.lookup_bucket(&hash, &public_state) {
        cache.hits += 1;
        return Ok(choice);
    }

    let choice = best_complete_screen_choice(teacher, state)?;
    cache.evaluations += 1;
    cache.insert(hash, public_state, choice.clone());
    Ok(choice)
}

fn action_earns_nature_token(
    state: &GameState,
    action: &TurnAction,
) -> Result<bool, FullLegalAuditError> {
    let tile_slot = match action.draft {
        DraftChoice::Paired { slot } => slot,
        DraftChoice::Independent { tile_slot, .. } => tile_slot,
    };
    let tile = state.market().tiles[tile_slot.index()].ok_or_else(|| {
        FullLegalAuditError::Invariant(
            "screen choice references an empty tile market slot".to_owned(),
        )
    })?;
    Ok(tile.keystone && action.wildlife == Some(action.tile.coord))
}

fn ranked_followup_wipes(
    state: &GameState,
    width: usize,
) -> Result<Vec<WildlifeWipe>, FullLegalAuditError> {
    let board = &state.boards()[state.current_player()];
    let demand = std::array::from_fn::<_, 5, _>(|index| {
        board.wildlife_placements(Wildlife::ALL[index]).len() as f64
    });
    let supply = state.public_supply();
    let bag_total = supply
        .wildlife_bag
        .iter()
        .map(|count| u16::from(*count))
        .sum::<u16>();
    let expected_replacement_demand = if bag_total == 0 {
        0.0
    } else {
        Wildlife::ALL
            .into_iter()
            .map(|wildlife| {
                demand[wildlife as usize] * f64::from(supply.wildlife_bag[wildlife as usize])
            })
            .sum::<f64>()
            / f64::from(bag_total)
    };
    let mut ranked = state
        .legal_wildlife_wipes()
        .into_iter()
        .map(|wipe| {
            let proxy = wipe
                .slots
                .iter()
                .map(|slot| {
                    let wildlife = state.market().wildlife[slot.index()].ok_or_else(|| {
                        FullLegalAuditError::Invariant(
                            "followup wipe references an empty wildlife slot".to_owned(),
                        )
                    })?;
                    Ok(expected_replacement_demand - demand[wildlife as usize])
                })
                .sum::<Result<f64, FullLegalAuditError>>()?;
            Ok((wipe, proxy))
        })
        .collect::<Result<Vec<_>, FullLegalAuditError>>()?;
    ranked.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| wipe_mask(&left.0).cmp(&wipe_mask(&right.0)))
    });
    ranked.truncate(width);
    Ok(ranked.into_iter().map(|(wipe, _)| wipe).collect())
}

fn chance_seed(state: &GameState, domain: &[u8], values: &[u64]) -> GameSeed {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"cascadia-v2-full-legal-audit-chance-v1");
    hasher.update(state.public_state().canonical_hash().as_bytes());
    hasher.update(&(domain.len() as u64).to_le_bytes());
    hasher.update(domain);
    for value in values {
        hasher.update(&value.to_le_bytes());
    }
    GameSeed(*hasher.finalize().as_bytes())
}

fn wipe_mask(wipe: &WildlifeWipe) -> u8 {
    wipe.slots
        .iter()
        .fold(0_u8, |mask, slot| mask | (1 << slot.index()))
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn population_stddev(values: &[f64], mean: f64) -> f64 {
    (values
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

fn ratio_count(numerator: usize, denominator: usize) -> f64 {
    numerator as f64 / denominator as f64
}

struct RealizedHiddenContinuation {
    canonical_hash: String,
    game: GameState,
    reusable: bool,
}

fn realized_hidden_future_diagnostic(
    teacher: &mut ExactMlxLegacyTeacher,
    game: &GameState,
    finalists: &[ExactMlxRootEstimate],
    public_winner_hash: &str,
    reusable_root_hash: &str,
    public_decision_cache: &mut PublicDecisionCache,
) -> Result<RealizedHiddenFutureDiagnostic, FullLegalAuditError> {
    let focal_player = game.current_player();
    let mut reusable_roots = 0;
    let mut continuations = finalists
        .iter()
        .map(|finalist| {
            let canonical_hash = canonical_action_hash(&finalist.action)?;
            let reusable = canonical_hash == reusable_root_hash;
            reusable_roots += usize::from(reusable);
            Ok(RealizedHiddenContinuation {
                canonical_hash,
                game: game.transition(&finalist.action)?,
                reusable,
            })
        })
        .collect::<Result<Vec<_>, FullLegalAuditError>>()?;
    if reusable_roots != 1 {
        return Err(FullLegalAuditError::Invariant(format!(
            "realized-hidden diagnostic expected one reusable champion root, found {reusable_roots}"
        )));
    }

    while continuations
        .iter()
        .any(|continuation| !continuation.game.is_game_over())
    {
        let mut pending_indices = Vec::new();
        let mut pending_cache_entries = Vec::new();
        for (index, continuation) in continuations.iter_mut().enumerate() {
            if continuation.game.is_game_over() {
                continue;
            }
            if continuation.reusable {
                match public_decision_cache.request_cacheable(&continuation.game) {
                    PublicDecisionLookup::Hit(action) => continuation.game.apply(&action)?,
                    PublicDecisionLookup::Miss(pending) => {
                        pending_indices.push(index);
                        pending_cache_entries.push(Some(pending));
                    }
                }
            } else {
                public_decision_cache.request_without_storage();
                pending_indices.push(index);
                pending_cache_entries.push(None);
            }
        }
        if pending_indices.is_empty() {
            continue;
        }
        let pending_games = pending_indices
            .iter()
            .map(|&index| &continuations[index].game)
            .collect::<Vec<_>>();
        let selected = teacher.select_actions(&pending_games)?;
        if selected.len() != pending_indices.len() {
            return Err(FullLegalAuditError::Invariant(format!(
                "multiplexed teacher returned {} actions for {} states",
                selected.len(),
                pending_indices.len()
            )));
        }
        for ((index, pending), action) in pending_indices
            .into_iter()
            .zip(pending_cache_entries)
            .zip(selected)
        {
            public_decision_cache.record_requested_evaluation(pending, &action)?;
            continuations[index].game.apply(&action)?;
        }
    }

    let actions = continuations
        .into_iter()
        .map(|continuation| RealizedHiddenActionResult {
            canonical_hash: continuation.canonical_hash,
            final_score: score_game(&continuation.game)[focal_player],
            terminal_state_blake3: continuation.game.canonical_hash().to_hex().to_string(),
        })
        .collect::<Vec<_>>();
    let realized_winner_hash = actions
        .iter()
        .max_by(|left, right| {
            left.final_score
                .base_total
                .cmp(&right.final_score.base_total)
                .then_with(|| right.canonical_hash.cmp(&left.canonical_hash))
        })
        .map(|result| result.canonical_hash.clone())
        .ok_or_else(|| {
            FullLegalAuditError::Invariant(
                "realized-hidden diagnostic received no finalists".to_owned(),
            )
        })?;
    Ok(RealizedHiddenFutureDiagnostic {
        label: "realized-hidden-future-never-public-selection".to_owned(),
        input_hidden_state_blake3: game.canonical_hash().to_hex().to_string(),
        public_winner_hash: public_winner_hash.to_owned(),
        realized_winner_hash,
        actions,
    })
}

fn enumerate_exact_actions(
    game: &GameState,
    prelude: &MarketPrelude,
) -> Result<Vec<EnumeratedAction>, FullLegalAuditError> {
    let cards = game.config().scoring_cards;
    let active_board = &game.boards()[game.current_player()];
    let baseline = score_board(active_board, cards);
    let habitat = active_board.habitat_analysis();
    let mut wildlife_score_cache = HashMap::new();
    let evaluated = game.evaluate_legal_turn_actions_with_tile_context(
        prelude,
        |board, placement, tile| {
            rescore_after_tile_with_habitat_analysis(
                board, cards, baseline, &habitat, placement, tile,
            )
        },
        |board, after_tile, placed_wildlife| {
            placed_wildlife.map_or(*after_tile, |placed_wildlife| {
                let wildlife_scores =
                    *wildlife_score_cache
                        .entry(placed_wildlife)
                        .or_insert_with(|| {
                            rescore_after_wildlife_placement(
                                board,
                                cards,
                                *after_tile,
                                placed_wildlife.0,
                            )
                            .wildlife
                        });
                rescore_with_wildlife_scores(board, *after_tile, wildlife_scores)
            })
        },
    )?;
    let direct = game.legal_turn_actions(prelude)?;
    if evaluated.len() != direct.len() {
        return Err(FullLegalAuditError::Invariant(format!(
            "evaluated action count {} differs from canonical legal count {}",
            evaluated.len(),
            direct.len()
        )));
    }
    let evaluated_actions = evaluated
        .iter()
        .map(|(action, _)| canonical_action_hash(action))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let direct_actions = direct
        .iter()
        .map(canonical_action_hash)
        .collect::<Result<BTreeSet<_>, _>>()?;
    if evaluated_actions != direct_actions {
        return Err(FullLegalAuditError::Invariant(
            "evaluated action set differs from GameState::legal_turn_actions".to_owned(),
        ));
    }
    Ok(evaluated
        .into_iter()
        .map(|(action, score)| EnumeratedAction { action, score })
        .collect())
}

fn screen_ranked_indices(actions: &[FullLegalActionRecord]) -> Vec<usize> {
    let mut indices = (0..actions.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        actions[*right]
            .screen_value
            .total_cmp(&actions[*left].screen_value)
            .then_with(|| {
                actions[*left]
                    .canonical_hash
                    .cmp(&actions[*right].canonical_hash)
            })
    });
    indices
}

pub fn stratified_sentinel_indices(
    action_count: usize,
    retained: usize,
    sentinel_count: usize,
) -> Vec<usize> {
    if action_count <= retained || sentinel_count == 0 {
        return Vec::new();
    }
    let remainder = action_count - retained;
    let count = sentinel_count.min(remainder);
    (0..count)
        .map(|index| retained + ((2 * index + 1) * remainder) / (2 * count))
        .collect()
}

fn substantial_union_indices(
    actions: &[FullLegalActionRecord],
    ranked_indices: &[usize],
    screen_limit: usize,
    sentinel_rank_indices: &[usize],
    champion_index: usize,
) -> Vec<usize> {
    let mut union = Vec::new();
    for index in ranked_indices.iter().copied().take(screen_limit) {
        push_unique(&mut union, index);
    }
    let mut champion_frontier = actions
        .iter()
        .enumerate()
        .filter_map(|(index, action)| {
            action
                .champion_frontier_r600
                .map(|estimate| (index, estimate.mean))
        })
        .collect::<Vec<_>>();
    champion_frontier.sort_by(|left, right| {
        right.1.total_cmp(&left.1).then_with(|| {
            actions[left.0]
                .canonical_hash
                .cmp(&actions[right.0].canonical_hash)
        })
    });
    for (index, _) in champion_frontier {
        push_unique(&mut union, index);
    }
    push_unique(&mut union, champion_index);
    for rank_index in sentinel_rank_indices {
        push_unique(&mut union, ranked_indices[*rank_index]);
    }
    union
}

#[derive(Debug, Clone, Copy)]
enum EstimateStage {
    Substantial,
    HighConfidence,
}

fn attach_estimates(
    actions: &mut [FullLegalActionRecord],
    by_hash: &BTreeMap<String, usize>,
    estimates: &[ExactMlxRootEstimate],
    stage: EstimateStage,
) -> Result<(), FullLegalAuditError> {
    let mut seen = BTreeSet::new();
    for estimate in estimates {
        let hash = canonical_action_hash(&estimate.action)?;
        if !seen.insert(hash.clone()) {
            return Err(FullLegalAuditError::Invariant(format!(
                "duplicate estimate for canonical action {hash}"
            )));
        }
        let index = action_index(by_hash, &hash)?;
        let record = RolloutEstimateRecord::from(estimate);
        match stage {
            EstimateStage::Substantial => actions[index].substantial_r1200 = Some(record),
            EstimateStage::HighConfidence => actions[index].high_confidence_r4800 = Some(record),
        }
    }
    Ok(())
}

fn high_estimate_for_hash<'a>(
    estimates: &'a [ExactMlxRootEstimate],
    hash: &str,
) -> Result<&'a ExactMlxRootEstimate, FullLegalAuditError> {
    estimates
        .iter()
        .find(|estimate| {
            canonical_action_hash(&estimate.action)
                .is_ok_and(|candidate_hash| candidate_hash == hash)
        })
        .ok_or_else(|| {
            FullLegalAuditError::Invariant(format!(
                "high-confidence set is missing required action {hash}"
            ))
        })
}

fn regret_estimate(
    best: &ExactMlxRootEstimate,
    comparator: &ExactMlxRootEstimate,
) -> RegretEstimate {
    if best.action == comparator.action {
        return RegretEstimate {
            points: 0.0,
            standard_error_upper_bound: 0.0,
            confidence_95: [0.0, 0.0],
        };
    }
    let regret = best.rollout_mean - comparator.rollout_mean;
    let best_se = best.rollout_stddev / f64::from(best.samples).sqrt();
    let comparator_se = comparator.rollout_stddev / f64::from(comparator.samples).sqrt();
    let standard_error = (best_se * best_se + comparator_se * comparator_se).sqrt();
    RegretEstimate {
        points: regret,
        standard_error_upper_bound: standard_error,
        confidence_95: [
            regret - 1.96 * standard_error,
            regret + 1.96 * standard_error,
        ],
    }
}

fn optional_regret_estimate(
    best: &ExactMlxRootEstimate,
    comparator: Option<&ExactMlxRootEstimate>,
) -> Option<RegretEstimate> {
    comparator.map(|comparator| regret_estimate(best, comparator))
}

fn drafted_components(
    staged: &GameState,
    action: &TurnAction,
) -> Result<(u8, Wildlife), FullLegalAuditError> {
    let (tile_slot, wildlife_slot) = match action.draft {
        DraftChoice::Paired { slot } => (slot, slot),
        DraftChoice::Independent {
            tile_slot,
            wildlife_slot,
        } => (tile_slot, wildlife_slot),
    };
    let tile = staged.market().tiles[tile_slot.index()].ok_or_else(|| {
        FullLegalAuditError::Invariant("canonical action references an empty tile slot".to_owned())
    })?;
    let wildlife = staged.market().wildlife[wildlife_slot.index()].ok_or_else(|| {
        FullLegalAuditError::Invariant(
            "canonical action references an empty wildlife slot".to_owned(),
        )
    })?;
    Ok((tile.id.0, wildlife))
}

fn is_same_slot_independent(action: &TurnAction) -> bool {
    matches!(
        action.draft,
        DraftChoice::Independent {
            tile_slot,
            wildlife_slot,
        } if tile_slot == wildlife_slot
    )
}

fn uniform_market_survival_proxy(
    visible_count: u8,
    public_bag_count: u8,
    public_bag_total: u16,
) -> f64 {
    let mut expected = f64::from(visible_count.saturating_sub(1));
    let replacement_probability = if public_bag_total == 0 {
        0.0
    } else {
        f64::from(public_bag_count) / f64::from(public_bag_total)
    };
    expected += replacement_probability;
    for _ in 0..3 {
        expected = expected * 0.75 + replacement_probability;
    }
    (expected / 4.0).clamp(0.0, 1.0)
}

fn opponent_wildlife_metrics(game: &GameState) -> ([u16; 5], [u16; 5]) {
    let mut eligible = [0; 5];
    let mut placed = [0; 5];
    for (seat, board) in game.boards().iter().enumerate() {
        if seat == game.current_player() {
            continue;
        }
        for wildlife in Wildlife::ALL {
            eligible[wildlife as usize] += board.wildlife_placements(wildlife).len() as u16;
        }
        for (_, tile) in board.placed_tiles() {
            if let Some(wildlife) = tile.wildlife {
                placed[wildlife as usize] += 1;
            }
        }
    }
    (eligible, placed)
}

fn decision_phase(personal_turn: u16) -> DecisionPhase {
    match personal_turn {
        1..=7 => DecisionPhase::Early,
        8..=14 => DecisionPhase::Middle,
        _ => DecisionPhase::Late,
    }
}

fn canonical_action_hash(action: &TurnAction) -> Result<String, serde_json::Error> {
    let bytes = serde_json::to_vec(action)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"cascadia-v2-full-legal-action-v1");
    hasher.update(&bytes);
    Ok(hasher.finalize().to_hex().to_string())
}

fn action_index(
    by_hash: &BTreeMap<String, usize>,
    hash: &str,
) -> Result<usize, FullLegalAuditError> {
    by_hash.get(hash).copied().ok_or_else(|| {
        FullLegalAuditError::Invariant(format!(
            "evaluated action {hash} is absent from the complete legal screen"
        ))
    })
}

fn push_unique(values: &mut Vec<usize>, value: usize) {
    if !values.contains(&value) {
        values.push(value);
    }
}

fn summarize_games(games: &[FullLegalGameAudit], elapsed_seconds: f64) -> FullLegalAuditSummary {
    let decisions = games
        .iter()
        .flat_map(|game| &game.decisions)
        .collect::<Vec<_>>();
    let count = decisions.len();
    let mean = |select: fn(&FullLegalDecisionAudit) -> f64| {
        if count == 0 {
            0.0
        } else {
            decisions
                .iter()
                .map(|decision| select(decision))
                .sum::<f64>()
                / count as f64
        }
    };
    FullLegalAuditSummary {
        games: games.len(),
        decisions: count,
        actions_screened: decisions.iter().map(|decision| decision.action_count).sum(),
        top_screen_recall: if count == 0 {
            0.0
        } else {
            decisions
                .iter()
                .filter(|decision| decision.top_screen_recalled_winner)
                .count() as f64
                / count as f64
        },
        mean_champion_regret: mean(|decision| decision.champion_regret.points),
        mean_champion_frontier_regret: mean(|decision| decision.champion_frontier_regret.points),
        mean_retained_screen_regret: mean(|decision| decision.retained_screen_regret.points),
        elapsed_seconds,
    }
}

pub fn validate_full_legal_audit_shard(
    shard: &FullLegalAuditShard,
) -> Result<(), FullLegalAuditError> {
    shard.config.validate()?;
    if shard.schema_version != 1 {
        return Err(FullLegalAuditError::Invariant(format!(
            "unsupported audit schema version {}",
            shard.schema_version
        )));
    }
    if shard.games.len() != shard.games_requested {
        return Err(FullLegalAuditError::Invariant(format!(
            "shard contains {} games but requested {}",
            shard.games.len(),
            shard.games_requested
        )));
    }
    let mut seeds = BTreeSet::new();
    for game in &shard.games {
        if !seeds.insert(game.raw_seed) {
            return Err(FullLegalAuditError::Invariant(format!(
                "duplicate game seed {}",
                game.raw_seed
            )));
        }
        let cache = game.public_decision_cache;
        if cache.requests != cache.evaluations + cache.hits
            || cache.entries > cache.evaluations
            || (!cache.enabled && (cache.hits != 0 || cache.entries != 0))
        {
            return Err(FullLegalAuditError::Invariant(format!(
                "seed {} has inconsistent public decision cache diagnostics",
                game.raw_seed
            )));
        }
        let expected_decisions = shard
            .config
            .audited_completed_turns
            .as_ref()
            .map_or(80, Vec::len);
        if game.decisions.len() != expected_decisions {
            return Err(FullLegalAuditError::Invariant(format!(
                "seed {} has {} audited decisions, expected {}",
                game.raw_seed,
                game.decisions.len(),
                expected_decisions
            )));
        }
        for decision in &game.decisions {
            if decision.actions.len() != decision.action_count {
                return Err(FullLegalAuditError::Invariant(format!(
                    "seed {} turn {} action count mismatch",
                    game.raw_seed, decision.completed_turns
                )));
            }
            let hashes = decision
                .actions
                .iter()
                .map(|action| &action.canonical_hash)
                .collect::<BTreeSet<_>>();
            if hashes.len() != decision.actions.len() {
                return Err(FullLegalAuditError::Invariant(format!(
                    "seed {} turn {} has duplicate action hashes",
                    game.raw_seed, decision.completed_turns
                )));
            }
            for required in [
                &decision.champion_action_hash,
                &decision.best_champion_frontier_hash,
                &decision.best_complete_screen_hash,
            ] {
                if !hashes.contains(required) {
                    return Err(FullLegalAuditError::Invariant(format!(
                        "seed {} turn {} is missing required action {}",
                        game.raw_seed, decision.completed_turns, required
                    )));
                }
            }
            if decision
                .actions
                .iter()
                .any(|action| !action.screen_value.is_finite())
            {
                return Err(FullLegalAuditError::Invariant(format!(
                    "seed {} turn {} contains a non-finite screen value",
                    game.raw_seed, decision.completed_turns
                )));
            }
            let expects_realized_hidden = shard
                .config
                .audits_realized_hidden(decision.completed_turns);
            if decision.realized_hidden_future.is_some() != expects_realized_hidden {
                return Err(FullLegalAuditError::Invariant(format!(
                    "seed {} turn {} realized-hidden diagnostic presence is incorrect",
                    game.raw_seed, decision.completed_turns
                )));
            }
            if let Some(realized) = &decision.realized_hidden_future {
                if realized.public_winner_hash != decision.best_complete_screen_hash
                    || realized.actions.len() != decision.high_confidence_count
                    || !realized
                        .actions
                        .iter()
                        .any(|action| action.canonical_hash == realized.realized_winner_hash)
                {
                    return Err(FullLegalAuditError::Invariant(format!(
                        "seed {} turn {} realized-hidden diagnostic is internally inconsistent",
                        game.raw_seed, decision.completed_turns
                    )));
                }
            }
            let expects_paid_wipe =
                shard.config.audits_paid_wipes() && decision.current_score.nature_tokens > 0;
            if decision.paid_wipe_diagnostic.is_some() != expects_paid_wipe {
                return Err(FullLegalAuditError::Invariant(format!(
                    "seed {} turn {} paid-wipe diagnostic presence is incorrect",
                    game.raw_seed, decision.completed_turns
                )));
            }
            if let Some(paid) = &decision.paid_wipe_diagnostic {
                if paid.options.len() != 15
                    || paid.initial_nature_tokens == 0
                    || paid.recursive_followup_exercised != (paid.followup_decision_nodes > 0)
                    || paid.followup_options_evaluated
                        != paid.followup_decision_nodes * shard.config.paid_wipe_followup_width
                    || paid.maximum_wipe_ordinal_considered
                        > usize::from(paid.initial_nature_tokens)
                    || paid.options.iter().any(|option| {
                        option.samples != shard.config.paid_wipe_determinizations
                            || !option.expected_value.is_finite()
                            || !option.value_stddev.is_finite()
                    })
                    || !paid.stop_value.is_finite()
                    || !paid.best_expected_value.is_finite()
                {
                    return Err(FullLegalAuditError::Invariant(format!(
                        "seed {} turn {} paid-wipe diagnostic is incomplete or non-finite",
                        game.raw_seed, decision.completed_turns
                    )));
                }
            }
        }
    }
    let summary = summarize_games(&shard.games, shard.summary.elapsed_seconds);
    if summary != shard.summary {
        return Err(FullLegalAuditError::Invariant(format!(
            "stored shard summary does not reproduce from game records: {}",
            summary_mismatches(&shard.summary, &summary).join(", ")
        )));
    }
    if shard.bridge_diagnostics.fallbacks != 0 || shard.batch_diagnostics.policy_fallbacks != 0 {
        return Err(FullLegalAuditError::Invariant(
            "audit shard contains a policy fallback".to_owned(),
        ));
    }
    Ok(())
}

fn summary_mismatches(
    stored: &FullLegalAuditSummary,
    recomputed: &FullLegalAuditSummary,
) -> Vec<String> {
    let mut mismatches = Vec::new();
    macro_rules! integer_field {
        ($field:ident) => {
            if stored.$field != recomputed.$field {
                mismatches.push(format!(
                    "{} stored={} recomputed={}",
                    stringify!($field),
                    stored.$field,
                    recomputed.$field
                ));
            }
        };
    }
    macro_rules! float_field {
        ($field:ident) => {
            if stored.$field.to_bits() != recomputed.$field.to_bits() {
                mismatches.push(format!(
                    "{} stored={:.17}({:016x}) recomputed={:.17}({:016x})",
                    stringify!($field),
                    stored.$field,
                    stored.$field.to_bits(),
                    recomputed.$field,
                    recomputed.$field.to_bits()
                ));
            }
        };
    }
    integer_field!(games);
    integer_field!(decisions);
    integer_field!(actions_screened);
    float_field!(top_screen_recall);
    float_field!(mean_champion_regret);
    float_field!(mean_champion_frontier_regret);
    float_field!(mean_retained_screen_regret);
    float_field!(elapsed_seconds);
    mismatches
}

pub fn write_full_legal_audit_shard(
    path: &Path,
    shard: &FullLegalAuditShard,
) -> Result<(), FullLegalAuditError> {
    validate_full_legal_audit_shard(shard)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!(
        "{}.tmp-{}",
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("json"),
        std::process::id()
    ));
    let bytes = serde_json::to_vec(shard)?;
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, path)?;
    Ok(())
}

pub fn read_full_legal_audit_shard(
    path: &Path,
) -> Result<FullLegalAuditShard, FullLegalAuditError> {
    let shard = serde_json::from_slice::<FullLegalAuditShard>(&fs::read(path)?)?;
    validate_full_legal_audit_shard(&shard)?;
    Ok(shard)
}

pub fn merge_full_legal_audit_shards(
    mut shards: Vec<FullLegalAuditShard>,
) -> Result<FullLegalAuditMerged, FullLegalAuditError> {
    if shards.is_empty() {
        return Err(FullLegalAuditError::Invariant(
            "merge requires at least one audit shard".to_owned(),
        ));
    }
    for shard in &shards {
        validate_full_legal_audit_shard(shard)?;
    }
    shards.sort_by(|left, right| {
        left.first_seed
            .cmp(&right.first_seed)
            .then_with(|| left.provenance.worker.cmp(&right.provenance.worker))
    });
    let config = shards[0].config.clone();
    let model_json = shards[0].provenance.model_json_blake3.clone();
    let model_weights = shards[0].provenance.model_safetensors_blake3.clone();
    let source = shards[0].provenance.source.v2_source_blake3.clone();
    for shard in &shards[1..] {
        if shard.config != config {
            return Err(FullLegalAuditError::Invariant(
                "audit shards use different frozen configurations".to_owned(),
            ));
        }
        if shard.provenance.model_json_blake3 != model_json
            || shard.provenance.model_safetensors_blake3 != model_weights
        {
            return Err(FullLegalAuditError::Invariant(
                "audit shards use different MLX model artifacts".to_owned(),
            ));
        }
        if shard.provenance.source.v2_source_blake3 != source {
            return Err(FullLegalAuditError::Invariant(
                "audit shards use different source trees".to_owned(),
            ));
        }
    }
    let elapsed_seconds = shards
        .iter()
        .map(|shard| shard.summary.elapsed_seconds)
        .sum();
    let manifests = shards
        .iter()
        .map(|shard| FullLegalAuditShardManifest {
            provenance: shard.provenance.clone(),
            first_seed: shard.first_seed,
            games: shard.games.len(),
            summary: shard.summary.clone(),
            bridge_diagnostics: shard.bridge_diagnostics.clone(),
            batch_diagnostics: shard.batch_diagnostics,
        })
        .collect();
    let mut games = shards
        .into_iter()
        .flat_map(|shard| shard.games)
        .collect::<Vec<_>>();
    games.sort_by_key(|game| game.raw_seed);
    if games
        .windows(2)
        .any(|pair| pair[0].raw_seed == pair[1].raw_seed)
    {
        return Err(FullLegalAuditError::Invariant(
            "audit shards overlap on at least one game seed".to_owned(),
        ));
    }
    let summary = summarize_games(&games, elapsed_seconds);
    let merged = FullLegalAuditMerged {
        schema_version: 1,
        config,
        shards: manifests,
        games,
        summary,
    };
    validate_full_legal_audit_merged(&merged)?;
    Ok(merged)
}

pub fn validate_full_legal_audit_merged(
    merged: &FullLegalAuditMerged,
) -> Result<(), FullLegalAuditError> {
    merged.config.validate()?;
    if merged.schema_version != 1 {
        return Err(FullLegalAuditError::Invariant(format!(
            "unsupported merged audit schema version {}",
            merged.schema_version
        )));
    }
    if merged.shards.is_empty() || merged.games.is_empty() {
        return Err(FullLegalAuditError::Invariant(
            "merged audit must contain shards and games".to_owned(),
        ));
    }
    if merged
        .games
        .windows(2)
        .any(|pair| pair[0].raw_seed >= pair[1].raw_seed)
    {
        return Err(FullLegalAuditError::Invariant(
            "merged audit games must be strictly seed-sorted".to_owned(),
        ));
    }
    let expected_games = merged.shards.iter().map(|shard| shard.games).sum::<usize>();
    if expected_games != merged.games.len() {
        return Err(FullLegalAuditError::Invariant(format!(
            "merged manifest declares {expected_games} games but contains {}",
            merged.games.len()
        )));
    }
    let elapsed_seconds = merged
        .shards
        .iter()
        .map(|shard| shard.summary.elapsed_seconds)
        .sum();
    if summarize_games(&merged.games, elapsed_seconds) != merged.summary {
        return Err(FullLegalAuditError::Invariant(
            "stored merged summary does not reproduce from game records".to_owned(),
        ));
    }
    Ok(())
}

pub fn write_full_legal_audit_merged(
    path: &Path,
    merged: &FullLegalAuditMerged,
) -> Result<(), FullLegalAuditError> {
    validate_full_legal_audit_merged(merged)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!(
        "{}.tmp-{}",
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("json"),
        std::process::id()
    ));
    fs::write(&temporary, serde_json::to_vec(merged)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

pub fn read_full_legal_audit_merged(
    path: &Path,
) -> Result<FullLegalAuditMerged, FullLegalAuditError> {
    let merged = serde_json::from_slice::<FullLegalAuditMerged>(&fs::read(path)?)?;
    validate_full_legal_audit_merged(&merged)?;
    Ok(merged)
}

pub fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must not precede the Unix epoch")
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_stages_can_be_disabled_without_weakening_action_evaluation() {
        let config = FullLegalAuditConfig {
            realized_hidden_completed_turns: Vec::new(),
            paid_wipe_determinizations: 0,
            ..FullLegalAuditConfig::default()
        };

        config.validate().unwrap();
        assert!(!config.audits_realized_hidden(12));
        assert!(!config.audits_paid_wipes());
        assert_eq!(config.screen_limit, FROZEN_SCREEN_LIMIT);
        assert_eq!(config.substantial_rollouts, FROZEN_SUBSTANTIAL_ROLLOUTS);
        assert_eq!(
            config.high_confidence_rollouts,
            FROZEN_HIGH_CONFIDENCE_ROLLOUTS
        );
    }

    #[test]
    fn online_oracle_allows_high_confidence_sets_without_retained_screen_actions() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(62_000),
        )
        .unwrap();
        let action = game
            .legal_turn_actions(&canonical_prelude(&game))
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let winner = ExactMlxRootEstimate {
            action,
            candidate_index: 0,
            features: Vec::new(),
            immediate_score: 0.0,
            rollout_mean: 100.0,
            rollout_stddev: 1.0,
            samples: 4_800,
            selected: true,
        };

        assert_eq!(optional_regret_estimate(&winner, None), None);
    }

    #[test]
    fn enabled_paid_wipe_diagnostics_still_require_valid_followup_geometry() {
        let config = FullLegalAuditConfig {
            paid_wipe_followup_determinizations: 0,
            ..FullLegalAuditConfig::default()
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn sentinel_indices_are_deterministic_unique_and_stratified() {
        let indices = stratified_sentinel_indices(1_064, 64, 16);
        assert_eq!(indices.len(), 16);
        assert!(indices.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(indices.iter().all(|index| (64..1_064).contains(index)));
        assert_eq!(indices[0], 95);
        assert_eq!(*indices.last().unwrap(), 1_032);
    }

    #[test]
    fn sentinel_indices_shrink_cleanly_for_short_remainders() {
        assert_eq!(stratified_sentinel_indices(66, 64, 16), vec![64, 65]);
        assert!(stratified_sentinel_indices(64, 64, 16).is_empty());
    }

    #[test]
    fn complete_enumerator_matches_canonical_legal_actions_with_independent_drafts() {
        let config = GameConfig::research_aaaaa(4).unwrap();
        let seed = GameSeed::from_u64(50_001);
        let mut game = GameState::new(config, seed).unwrap();
        let mut rngs = (0..4)
            .map(|seat| {
                cascadia_sim::strategy_rng(seed, seat, cascadia_sim::PATTERN_AWARE_STRATEGY_ID)
            })
            .collect::<Vec<_>>();
        for _ in 0..80 {
            let prelude = canonical_prelude(&game);
            if game.boards()[game.current_player()].nature_tokens() > 0 {
                let enumerated = enumerate_exact_actions(&game, &prelude).unwrap();
                let canonical = game.legal_turn_actions(&prelude).unwrap();
                assert_eq!(enumerated.len(), canonical.len());
                assert!(
                    enumerated
                        .iter()
                        .any(|candidate| { is_same_slot_independent(&candidate.action) })
                );
                return;
            }
            let player = game.current_player();
            let action = crate::legacy_teacher::pattern_fallback(&game, &mut rngs[player]).unwrap();
            game.apply(&action).unwrap();
        }
        panic!("test trajectory did not earn a Nature Token");
    }

    #[test]
    fn uniform_survival_proxy_is_bounded_and_supply_sensitive() {
        let scarce = uniform_market_survival_proxy(1, 1, 60);
        let plentiful = uniform_market_survival_proxy(3, 15, 60);
        assert!((0.0..=1.0).contains(&scarce));
        assert!((0.0..=1.0).contains(&plentiful));
        assert!(plentiful > scarce);
    }

    #[test]
    fn batch_diagnostics_preserve_cross_wave_row_reuse_counts() {
        let mut diagnostics: SerializableBatchDiagnostics = BatchedNnueDiagnostics {
            neural_batches: 3,
            neural_rows: 11,
            physical_neural_rows: 7,
            reuse_observed_physical_rows: 7,
            reuse_repeated_physical_rows: 2,
            ..BatchedNnueDiagnostics::default()
        }
        .into();

        assert_eq!(diagnostics.neural_batches, 3);
        assert_eq!(diagnostics.neural_rows, 11);
        assert_eq!(diagnostics.physical_neural_rows, 7);
        assert_eq!(diagnostics.reuse_observed_physical_rows, 7);
        assert_eq!(diagnostics.reuse_repeated_physical_rows, 2);
        assert!(diagnostics.stage_timings.is_none());

        diagnostics.record_multiplex(ExactMlxMultiplexDiagnostics {
            cross_request_rows_observed: 13,
            cross_request_duplicate_rows: 5,
            ..ExactMlxMultiplexDiagnostics::default()
        });
        assert_eq!(diagnostics.cross_request_rows_observed, 13);
        assert_eq!(diagnostics.cross_request_duplicate_rows, 5);

        let timed: SerializableBatchDiagnostics = BatchedNnueDiagnostics {
            stage_timings: BatchedNnueStageTimings {
                opponent_advance_ns: 17,
                neural_evaluation_ns: 23,
                ..BatchedNnueStageTimings::default()
            },
            ..BatchedNnueDiagnostics::default()
        }
        .into();
        assert_eq!(
            timed.stage_timings,
            Some(SerializableBatchStageTimings {
                opponent_advance_ns: 17,
                neural_evaluation_ns: 23,
                total_ns: 40,
                ..SerializableBatchStageTimings::default()
            })
        );
    }

    #[test]
    fn chance_seed_depends_only_on_public_state_and_domain() {
        let config = GameConfig::research_aaaaa(4).unwrap();
        let game = GameState::new(config, GameSeed::from_u64(50_002)).unwrap();
        let mut hidden_variant = game.clone();
        hidden_variant.redeterminize_hidden(GameSeed::from_u64(50_003));
        assert_ne!(game.canonical_hash(), hidden_variant.canonical_hash());
        assert_eq!(
            chance_seed(&game, b"paid", &[1, 2, 3]),
            chance_seed(&hidden_variant, b"paid", &[1, 2, 3])
        );
        assert_ne!(
            chance_seed(&game, b"paid", &[1, 2, 3]),
            chance_seed(&game, b"paid", &[1, 2, 4])
        );
    }

    #[test]
    fn complete_screen_cache_reuses_hidden_reorders_and_checks_hash_collisions() {
        let config = GameConfig::research_aaaaa(4).unwrap();
        let game = GameState::new(config, GameSeed::from_u64(61_500)).unwrap();
        let mut redetermined = game.clone();
        redetermined.redeterminize_hidden(chance_seed(
            &game,
            b"complete-screen-cache-hidden-reorder",
            &[1],
        ));
        assert_eq!(game.public_state(), redetermined.public_state());

        let choice = CompleteScreenChoice {
            action_hash: "cached-choice".to_owned(),
            value: 91.25,
            earns_nature_token: true,
        };
        let public = game.public_state();
        let hash = public.canonical_hash().to_hex().to_string();
        let mut cache = CompleteScreenCache::default();
        cache.insert(hash.clone(), public.clone(), choice.clone());
        assert_eq!(
            cache.lookup_bucket(&hash, &redetermined.public_state()),
            Some(choice)
        );

        let mut advanced = game.clone();
        let prelude = canonical_prelude(&advanced);
        let action = advanced
            .legal_turn_actions(&prelude)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        advanced.apply(&action).unwrap();
        assert_ne!(public, advanced.public_state());
        assert_eq!(cache.lookup_bucket(&hash, &advanced.public_state()), None);
    }

    #[test]
    fn public_decision_cache_requires_exact_state_and_validates_actions() {
        let config = GameConfig::research_aaaaa(4).unwrap();
        let game = GameState::new(config, GameSeed::from_u64(61_501)).unwrap();
        let mut redetermined = game.clone();
        redetermined.redeterminize_hidden(chance_seed(
            &game,
            b"public-decision-cache-hidden-reorder",
            &[1],
        ));
        assert_eq!(game.public_state(), redetermined.public_state());

        let prelude = canonical_prelude(&game);
        let actions = game.legal_turn_actions(&prelude).unwrap();
        assert!(actions.len() > 1);
        let public = game.public_state();
        let (hash, bytes) = PublicDecisionCache::encoded_public_state(&public);
        let mut cache = PublicDecisionCache::default();
        cache
            .insert_or_validate(hash, bytes.clone(), &actions[0])
            .unwrap();
        let redetermined_bytes = redetermined.public_state().canonical_bytes();
        assert_eq!(
            cache.lookup_bucket(&hash, &redetermined_bytes),
            Some(actions[0].clone())
        );
        assert!(cache.insert_or_validate(hash, bytes, &actions[1]).is_err());

        let mut advanced = game.clone();
        advanced.apply(&actions[0]).unwrap();
        assert_ne!(public, advanced.public_state());
        assert_eq!(
            cache.lookup_bucket(&hash, &advanced.public_state().canonical_bytes()),
            None
        );
    }

    #[test]
    fn wipe_masks_cover_every_nonempty_market_subset() {
        let wipes = (1_u8..16)
            .map(|mask| WildlifeWipe {
                slots: MarketSlot::ALL
                    .into_iter()
                    .filter(|slot| mask & (1 << slot.index()) != 0)
                    .collect(),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            wipes.iter().map(wipe_mask).collect::<Vec<_>>(),
            (1_u8..16).collect::<Vec<_>>()
        );
    }

    #[test]
    fn shard_merge_is_input_order_independent() {
        let left = empty_test_shard("john1", 61_000);
        let right = empty_test_shard("john2", 61_001);
        let forward = merge_full_legal_audit_shards(vec![left.clone(), right.clone()]).unwrap();
        let reverse = merge_full_legal_audit_shards(vec![right, left]).unwrap();
        assert_eq!(forward, reverse);
        assert_eq!(forward.games[0].raw_seed, 61_000);
        assert_eq!(forward.games[1].raw_seed, 61_001);
    }

    fn empty_test_shard(worker: &str, raw_seed: u64) -> FullLegalAuditShard {
        let game = FullLegalGameAudit {
            raw_seed,
            decisions: Vec::new(),
            final_scores: Vec::new(),
            final_state_blake3: format!("state-{raw_seed}"),
            public_decision_cache: PublicDecisionCacheDiagnostics::default(),
            elapsed_seconds: 1.0,
        };
        let games = vec![game];
        FullLegalAuditShard {
            schema_version: 1,
            config: FullLegalAuditConfig {
                audited_completed_turns: Some(Vec::new()),
                ..FullLegalAuditConfig::default()
            },
            provenance: AuditProvenance {
                worker: worker.to_owned(),
                source: SourceProvenance {
                    git_revision: "test".to_owned(),
                    git_dirty: false,
                    git_status_blake3: "status".to_owned(),
                    v2_source_blake3: "source".to_owned(),
                },
                executable_blake3: "executable".to_owned(),
                model_json_blake3: "model-json".to_owned(),
                model_safetensors_blake3: "model-weights".to_owned(),
                started_unix_seconds: 1,
            },
            first_seed: raw_seed,
            games_requested: 1,
            summary: summarize_games(&games, 1.0),
            games,
            bridge_diagnostics: BridgeDiagnostics::default(),
            batch_diagnostics: SerializableBatchDiagnostics::default(),
            completed_unix_seconds: 2,
        }
    }
}
