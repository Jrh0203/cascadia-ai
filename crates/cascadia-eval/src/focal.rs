//! Versioned focal-seat benchmark statistics for R2-MAP expert iteration.
//!
//! This module deliberately separates the scientific benchmark contract from
//! model serving. Runners emit one [`FocalGameRecord`] per physical game; the
//! order-independent aggregators below validate pair identity before exposing
//! any strength statistic.

use std::collections::{BTreeMap, HashMap};

use cascadia_game::{
    DraftChoice, GameConfig, GameSeed, GameState, ReplayError, RuleError, ScoreBreakdown,
    score_game,
};
use cascadia_sim::MatchResult;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const FOCAL_BENCHMARK_SCHEMA_VERSION: u16 = 1;
pub const FOCAL_BENCHMARK_PROTOCOL_ID: &str = "r2-map-focal-paired-v1";
pub const SMOKE_PAIR_COUNT: usize = 20;
pub const DEVELOPMENT_PAIR_COUNT: usize = 250;
pub const DEVELOPMENT_TARGET_MEAN: f64 = 100.0;
pub const FOCAL_MAX_RSS_BYTES: u64 = 4 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BenchmarkStage {
    StrengthBlindedSmoke,
    Development,
}

impl BenchmarkStage {
    pub const fn expected_pairs(self) -> usize {
        match self {
            Self::StrengthBlindedSmoke => SMOKE_PAIR_COUNT,
            Self::Development => DEVELOPMENT_PAIR_COUNT,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PairArm {
    Candidate,
    Control,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpponentIdentity {
    pub seat: u8,
    pub checkpoint_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FocalRecordIdentity {
    pub stage: BenchmarkStage,
    pub pair_index: usize,
    pub arm: PairArm,
    pub focal_checkpoint_id: String,
    pub opponents: Vec<OpponentIdentity>,
    pub field_manifest_id: String,
    pub inference_settings_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FocalRuntimeObservation {
    pub checkpoint_load_seconds: f64,
    pub peak_rss_bytes: u64,
    pub swap_delta_bytes: i64,
    pub clean_shutdown: bool,
}

impl Default for FocalRuntimeObservation {
    fn default() -> Self {
        Self {
            checkpoint_load_seconds: 0.0,
            peak_rss_bytes: 0,
            swap_delta_bytes: 0,
            clean_shutdown: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PineconeObservation {
    pub earned: u16,
    pub independent_draft_spend: u16,
    pub paid_wipe_spend: u16,
    pub total_spend: u16,
    pub remaining: u16,
    pub free_replacements: u16,
}

impl PineconeObservation {
    pub fn conservation_holds(self) -> bool {
        let spent = self
            .independent_draft_spend
            .checked_add(self.paid_wipe_spend);
        let accounted = spent.and_then(|spent| spent.checked_add(self.remaining));
        spent == Some(self.total_spend) && accounted == Some(self.earned)
    }
}

#[derive(Debug, Default)]
struct FocalPineconeAccumulator {
    earned: u16,
    independent_draft_spend: u16,
    paid_wipe_spend: u16,
    free_replacements: u16,
}

impl FocalPineconeAccumulator {
    fn observe_turn(
        &mut self,
        pair_index: usize,
        turn_index: usize,
        before: u8,
        after: u8,
        action: &cascadia_game::TurnAction,
    ) -> Result<(), FocalBenchmarkError> {
        let independent = u16::from(matches!(action.draft, DraftChoice::Independent { .. }));
        let paid_wipes = u16::try_from(action.wildlife_wipes.len())
            .map_err(|_| FocalBenchmarkError::PineconeCounterOverflow)?;
        self.independent_draft_spend = self
            .independent_draft_spend
            .checked_add(independent)
            .ok_or(FocalBenchmarkError::PineconeCounterOverflow)?;
        self.paid_wipe_spend = self
            .paid_wipe_spend
            .checked_add(paid_wipes)
            .ok_or(FocalBenchmarkError::PineconeCounterOverflow)?;
        self.free_replacements = self
            .free_replacements
            .checked_add(u16::from(action.replace_three_of_a_kind))
            .ok_or(FocalBenchmarkError::PineconeCounterOverflow)?;
        let spend = independent
            .checked_add(paid_wipes)
            .ok_or(FocalBenchmarkError::PineconeCounterOverflow)?;
        let turn_earned = i32::from(after) + i32::from(spend) - i32::from(before);
        if !(0..=1).contains(&turn_earned) {
            return Err(FocalBenchmarkError::InvalidPineconeTransition {
                pair_index,
                turn_index,
                before,
                spend,
                after,
            });
        }
        self.earned = self
            .earned
            .checked_add(turn_earned as u16)
            .ok_or(FocalBenchmarkError::PineconeCounterOverflow)?;
        Ok(())
    }

    fn finish(self, remaining: u16) -> Result<PineconeObservation, FocalBenchmarkError> {
        let total_spend = self
            .independent_draft_spend
            .checked_add(self.paid_wipe_spend)
            .ok_or(FocalBenchmarkError::PineconeCounterOverflow)?;
        Ok(PineconeObservation {
            earned: self.earned,
            independent_draft_spend: self.independent_draft_spend,
            paid_wipe_spend: self.paid_wipe_spend,
            total_spend,
            remaining,
            free_replacements: self.free_replacements,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FocalGameRecord {
    pub schema_version: u16,
    pub protocol_id: String,
    pub identity: FocalRecordIdentity,
    pub game_seed: GameSeed,
    pub focal_seat: u8,
    pub final_state_hash: [u8; 32],
    pub replay_blake3: String,
    pub score: ScoreBreakdown,
    pub pinecones: PineconeObservation,
    pub focal_decision_seconds: Vec<f64>,
    pub elapsed_seconds: f64,
    pub runtime: FocalRuntimeObservation,
}

impl FocalGameRecord {
    /// Derives focal-only accounting from the sealed replay instead of trusting
    /// runner-maintained counters.
    pub fn from_match(
        identity: FocalRecordIdentity,
        focal_seat: u8,
        runtime: FocalRuntimeObservation,
        result: &MatchResult,
    ) -> Result<Self, FocalBenchmarkError> {
        validate_record_identity_shape(&identity, focal_seat)?;
        let expected_config = GameConfig::research_aaaaa(4)?;
        if result.replay.config != expected_config {
            return Err(FocalBenchmarkError::WrongGameConfig);
        }
        if result.seed != result.replay.seed {
            return Err(FocalBenchmarkError::SeedMismatch);
        }
        if result.scores.len() != 4 || usize::from(focal_seat) >= result.scores.len() {
            return Err(FocalBenchmarkError::InvalidFocalSeat(focal_seat));
        }
        if result.replay.turns.len() != result.decision_seconds.len() {
            return Err(FocalBenchmarkError::DecisionCountMismatch {
                turns: result.replay.turns.len(),
                decisions: result.decision_seconds.len(),
            });
        }

        // Verify the seal before deriving scientific telemetry.
        result.replay.play()?;
        let final_state_hash = result
            .replay
            .final_state_hash
            .ok_or(FocalBenchmarkError::MissingFinalStateHash)?;
        let replay_blake3 = blake3::hash(&serde_json::to_vec(&result.replay)?)
            .to_hex()
            .to_string();
        let mut game = GameState::new(result.replay.config, result.replay.seed)?;
        let mut pinecone_accumulator = FocalPineconeAccumulator::default();
        let mut focal_decision_seconds = Vec::new();

        for (turn_index, action) in result.replay.turns.iter().enumerate() {
            let player = game.current_player();
            let focal_turn = player == usize::from(focal_seat);
            let before = game.boards()[player].nature_tokens();
            game.apply(action)?;
            if focal_turn {
                focal_decision_seconds.push(result.decision_seconds[turn_index]);
                let after = game.boards()[player].nature_tokens();
                pinecone_accumulator.observe_turn(
                    identity.pair_index,
                    turn_index,
                    before,
                    after,
                    action,
                )?;
            }
        }

        let recomputed_scores = score_game(&game);
        if recomputed_scores != result.scores {
            return Err(FocalBenchmarkError::ScoreMismatch);
        }
        let score = result.scores[usize::from(focal_seat)];
        let pinecones = pinecone_accumulator.finish(score.nature_tokens)?;
        if !pinecones.conservation_holds() {
            return Err(FocalBenchmarkError::PineconeConservation {
                pair_index: identity.pair_index,
                arm: identity.arm,
                pinecones,
            });
        }

        Ok(Self {
            schema_version: FOCAL_BENCHMARK_SCHEMA_VERSION,
            protocol_id: FOCAL_BENCHMARK_PROTOCOL_ID.to_owned(),
            identity,
            game_seed: result.seed,
            focal_seat,
            final_state_hash,
            replay_blake3,
            score,
            pinecones,
            focal_decision_seconds,
            elapsed_seconds: result.elapsed_seconds,
            runtime,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntegerDistribution {
    pub count: usize,
    pub mean: f64,
    pub standard_deviation: f64,
    pub standard_error: f64,
    pub confidence_95: [f64; 2],
    pub p10: f64,
    pub p50: f64,
    pub p90: f64,
    pub min: i32,
    pub max: i32,
    pub histogram: BTreeMap<i32, usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FloatDistribution {
    pub count: usize,
    pub mean: f64,
    pub standard_deviation: f64,
    pub standard_error: f64,
    pub confidence_95: [f64; 2],
    pub p10: f64,
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
    pub min: f64,
    pub max: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnimalDistributions {
    pub bear: IntegerDistribution,
    pub elk: IntegerDistribution,
    pub salmon: IntegerDistribution,
    pub hawk: IntegerDistribution,
    pub fox: IntegerDistribution,
    pub aggregate_wildlife: IntegerDistribution,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TerrainDistributions {
    pub mountain: IntegerDistribution,
    pub forest: IntegerDistribution,
    pub prairie: IntegerDistribution,
    pub wetland: IntegerDistribution,
    pub river: IntegerDistribution,
    pub aggregate_habitat: IntegerDistribution,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PineconeDistributions {
    pub earned: IntegerDistribution,
    pub independent_draft_spend: IntegerDistribution,
    pub paid_wipe_spend: IntegerDistribution,
    pub total_spend: IntegerDistribution,
    pub remaining: IntegerDistribution,
    pub free_replacements: IntegerDistribution,
    pub conservation_valid_games: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FocalStatistics {
    pub base_total: IntegerDistribution,
    pub animals: AnimalDistributions,
    pub terrains: TerrainDistributions,
    pub pinecones: PineconeDistributions,
    pub focal_decision_latency_milliseconds: FloatDistribution,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedFocalDeltaStatistics {
    pub base_total: IntegerDistribution,
    pub animals: AnimalDistributions,
    pub terrains: TerrainDistributions,
    pub pinecones: PineconeDistributions,
    /// Per-pair difference between candidate and control focal mean latency.
    pub focal_decision_latency_milliseconds: FloatDistribution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PromotionClassification {
    Promote,
    Reject,
    Inconclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromotionGates {
    pub resource_gates_pass: bool,
    pub preregistered_guardrails_pass: bool,
}

impl Default for PromotionGates {
    fn default() -> Self {
        Self {
            resource_gates_pass: true,
            preregistered_guardrails_pass: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StrengthBlindedSmokeReport {
    pub schema_version: u16,
    pub protocol_id: String,
    pub stage: BenchmarkStage,
    pub strength_outputs_blinded: bool,
    pub pairs: usize,
    pub physical_games: usize,
    pub wall_seconds: f64,
    pub games_per_second: f64,
    pub peak_rss_bytes: u64,
    pub maximum_swap_delta_bytes: i64,
    pub all_clean_shutdowns: bool,
    pub all_pinecone_conservation_checks_passed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DevelopmentComparisonReport {
    pub schema_version: u16,
    pub protocol_id: String,
    pub stage: BenchmarkStage,
    pub strength_outputs_blinded: bool,
    pub pairs: usize,
    pub physical_games: usize,
    pub candidate_checkpoint_id: String,
    pub control_checkpoint_id: String,
    pub candidate: FocalStatistics,
    pub control: FocalStatistics,
    pub paired_delta: PairedFocalDeltaStatistics,
    pub candidate_wins: usize,
    pub ties: usize,
    pub candidate_losses: usize,
    pub candidate_distance_from_100: f64,
    pub control_distance_from_100: f64,
    pub candidate_checkpoint_load_seconds: FloatDistribution,
    pub control_checkpoint_load_seconds: FloatDistribution,
    pub wall_seconds: f64,
    pub games_per_second: f64,
    pub classification: PromotionClassification,
}

#[derive(Debug)]
struct ValidatedPair<'a> {
    candidate: &'a FocalGameRecord,
    control: &'a FocalGameRecord,
}

pub fn aggregate_strength_blinded_smoke(
    records: &[FocalGameRecord],
    wall_seconds: f64,
) -> Result<StrengthBlindedSmokeReport, FocalBenchmarkError> {
    let pairs = validate_and_pair(BenchmarkStage::StrengthBlindedSmoke, records)?;
    validate_wall_seconds(wall_seconds)?;
    let peak_rss_bytes = records
        .iter()
        .map(|record| record.runtime.peak_rss_bytes)
        .max()
        .unwrap_or(0);
    let maximum_swap_delta_bytes = records
        .iter()
        .map(|record| record.runtime.swap_delta_bytes)
        .max()
        .unwrap_or(0);
    Ok(StrengthBlindedSmokeReport {
        schema_version: FOCAL_BENCHMARK_SCHEMA_VERSION,
        protocol_id: FOCAL_BENCHMARK_PROTOCOL_ID.to_owned(),
        stage: BenchmarkStage::StrengthBlindedSmoke,
        strength_outputs_blinded: true,
        pairs: pairs.len(),
        physical_games: records.len(),
        wall_seconds,
        games_per_second: records.len() as f64 / wall_seconds,
        peak_rss_bytes,
        maximum_swap_delta_bytes,
        all_clean_shutdowns: records.iter().all(|record| record.runtime.clean_shutdown),
        all_pinecone_conservation_checks_passed: records
            .iter()
            .all(|record| record.pinecones.conservation_holds()),
    })
}

pub fn aggregate_development_comparison(
    records: &[FocalGameRecord],
    wall_seconds: f64,
    gates: PromotionGates,
) -> Result<DevelopmentComparisonReport, FocalBenchmarkError> {
    let pairs = validate_and_pair(BenchmarkStage::Development, records)?;
    validate_wall_seconds(wall_seconds)?;
    let candidate_records = pairs.iter().map(|pair| pair.candidate).collect::<Vec<_>>();
    let control_records = pairs.iter().map(|pair| pair.control).collect::<Vec<_>>();
    let candidate = focal_statistics(&candidate_records);
    let control = focal_statistics(&control_records);
    let paired_delta = paired_delta_statistics(&pairs);
    let classification = classify_fixed_development(&paired_delta.base_total, gates);
    let candidate_checkpoint_id = candidate_records[0].identity.focal_checkpoint_id.clone();
    let control_checkpoint_id = control_records[0].identity.focal_checkpoint_id.clone();
    let mut candidate_wins = 0;
    let mut ties = 0;
    let mut candidate_losses = 0;
    for pair in &pairs {
        match pair
            .candidate
            .score
            .base_total
            .cmp(&pair.control.score.base_total)
        {
            std::cmp::Ordering::Greater => candidate_wins += 1,
            std::cmp::Ordering::Equal => ties += 1,
            std::cmp::Ordering::Less => candidate_losses += 1,
        }
    }
    Ok(DevelopmentComparisonReport {
        schema_version: FOCAL_BENCHMARK_SCHEMA_VERSION,
        protocol_id: FOCAL_BENCHMARK_PROTOCOL_ID.to_owned(),
        stage: BenchmarkStage::Development,
        strength_outputs_blinded: false,
        pairs: pairs.len(),
        physical_games: records.len(),
        candidate_checkpoint_id,
        control_checkpoint_id,
        candidate_distance_from_100: candidate.base_total.mean - DEVELOPMENT_TARGET_MEAN,
        control_distance_from_100: control.base_total.mean - DEVELOPMENT_TARGET_MEAN,
        candidate_checkpoint_load_seconds: float_distribution(
            &candidate_records
                .iter()
                .map(|record| record.runtime.checkpoint_load_seconds)
                .collect::<Vec<_>>(),
        ),
        control_checkpoint_load_seconds: float_distribution(
            &control_records
                .iter()
                .map(|record| record.runtime.checkpoint_load_seconds)
                .collect::<Vec<_>>(),
        ),
        candidate,
        control,
        paired_delta,
        candidate_wins,
        ties,
        candidate_losses,
        wall_seconds,
        games_per_second: records.len() as f64 / wall_seconds,
        classification,
    })
}

/// Aggregate absolute focal-seat telemetry after independently validating each
/// replay-derived record. This is the shared statistics boundary used by the
/// non-comparative 100-game longitudinal panel.
pub fn aggregate_focal_statistics(
    records: &[FocalGameRecord],
) -> Result<FocalStatistics, FocalBenchmarkError> {
    if records.is_empty() {
        return Err(FocalBenchmarkError::MissingPair(0));
    }
    for record in records {
        validate_focal_record(record)?;
    }
    Ok(focal_statistics(&records.iter().collect::<Vec<_>>()))
}

fn validate_and_pair(
    stage: BenchmarkStage,
    records: &[FocalGameRecord],
) -> Result<Vec<ValidatedPair<'_>>, FocalBenchmarkError> {
    let expected_pairs = stage.expected_pairs();
    let expected_records = expected_pairs * 2;
    if records.len() != expected_records {
        return Err(FocalBenchmarkError::WrongPhysicalGameCount {
            expected: expected_records,
            actual: records.len(),
        });
    }
    let mut by_pair: HashMap<usize, [Option<&FocalGameRecord>; 2]> = HashMap::new();
    for record in records {
        validate_record(record, stage, expected_pairs)?;
        let slots = by_pair.entry(record.identity.pair_index).or_default();
        let arm_index = match record.identity.arm {
            PairArm::Candidate => 0,
            PairArm::Control => 1,
        };
        if slots[arm_index].replace(record).is_some() {
            return Err(FocalBenchmarkError::DuplicateArm {
                pair_index: record.identity.pair_index,
                arm: record.identity.arm,
            });
        }
    }

    let mut pairs = Vec::with_capacity(expected_pairs);
    let mut candidate_checkpoint_id: Option<&str> = None;
    let mut control_checkpoint_id: Option<&str> = None;
    let mut field_manifest_id: Option<&str> = None;
    let mut inference_settings_id: Option<&str> = None;
    for pair_index in 0..expected_pairs {
        let slots = by_pair
            .get(&pair_index)
            .ok_or(FocalBenchmarkError::MissingPair(pair_index))?;
        let candidate = slots[0].ok_or(FocalBenchmarkError::MissingArm {
            pair_index,
            arm: PairArm::Candidate,
        })?;
        let control = slots[1].ok_or(FocalBenchmarkError::MissingArm {
            pair_index,
            arm: PairArm::Control,
        })?;
        validate_pair_identity(candidate, control)?;
        require_stable_identity(
            "candidate checkpoint",
            &mut candidate_checkpoint_id,
            &candidate.identity.focal_checkpoint_id,
        )?;
        require_stable_identity(
            "control checkpoint",
            &mut control_checkpoint_id,
            &control.identity.focal_checkpoint_id,
        )?;
        require_stable_identity(
            "field manifest",
            &mut field_manifest_id,
            &candidate.identity.field_manifest_id,
        )?;
        require_stable_identity(
            "inference settings",
            &mut inference_settings_id,
            &candidate.identity.inference_settings_id,
        )?;
        pairs.push(ValidatedPair { candidate, control });
    }
    Ok(pairs)
}

fn validate_record(
    record: &FocalGameRecord,
    stage: BenchmarkStage,
    expected_pairs: usize,
) -> Result<(), FocalBenchmarkError> {
    if record.schema_version != FOCAL_BENCHMARK_SCHEMA_VERSION {
        return Err(FocalBenchmarkError::UnsupportedSchema(
            record.schema_version,
        ));
    }
    if record.replay_blake3.len() != 64
        || !record
            .replay_blake3
            .bytes()
            .all(|value| value.is_ascii_hexdigit())
    {
        return Err(FocalBenchmarkError::InvalidReplayDigest(
            record.identity.pair_index,
        ));
    }
    if record.protocol_id != FOCAL_BENCHMARK_PROTOCOL_ID {
        return Err(FocalBenchmarkError::ProtocolMismatch(
            record.protocol_id.clone(),
        ));
    }
    if record.identity.stage != stage {
        return Err(FocalBenchmarkError::StageMismatch);
    }
    if record.identity.pair_index >= expected_pairs {
        return Err(FocalBenchmarkError::ExtraPair(record.identity.pair_index));
    }
    validate_record_identity_shape(&record.identity, record.focal_seat)?;
    let expected_base_total = record.score.habitat.iter().sum::<u16>()
        + record.score.wildlife.iter().sum::<u16>()
        + record.score.nature_tokens;
    if record.score.base_total != expected_base_total
        || record.score.total != expected_base_total
        || record.score.habitat_bonus != [0; 5]
    {
        return Err(FocalBenchmarkError::ScoreArithmeticMismatch {
            pair_index: record.identity.pair_index,
            arm: record.identity.arm,
            expected_base_total,
            score: record.score,
        });
    }
    if !record.pinecones.conservation_holds() {
        return Err(FocalBenchmarkError::PineconeConservation {
            pair_index: record.identity.pair_index,
            arm: record.identity.arm,
            pinecones: record.pinecones,
        });
    }
    if record.focal_decision_seconds.len() != 20 {
        return Err(FocalBenchmarkError::WrongFocalDecisionCount {
            pair_index: record.identity.pair_index,
            expected: 20,
            actual: record.focal_decision_seconds.len(),
        });
    }
    if record
        .focal_decision_seconds
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || !record.elapsed_seconds.is_finite()
        || record.elapsed_seconds < 0.0
        || !record.runtime.checkpoint_load_seconds.is_finite()
        || record.runtime.checkpoint_load_seconds < 0.0
    {
        return Err(FocalBenchmarkError::InvalidLatency(
            record.identity.pair_index,
        ));
    }
    Ok(())
}

/// Validates one persisted focal record independently of aggregation.
///
/// Campaign runners use this at the atomic receipt boundary; the fixed-size
/// aggregators repeat the same checks before exposing strength statistics.
pub fn validate_focal_record(record: &FocalGameRecord) -> Result<(), FocalBenchmarkError> {
    validate_record(
        record,
        record.identity.stage,
        record.identity.stage.expected_pairs(),
    )
}

/// Validates the identity relation between one candidate/control pair.
pub fn validate_focal_pair(
    candidate: &FocalGameRecord,
    control: &FocalGameRecord,
) -> Result<(), FocalBenchmarkError> {
    if candidate.identity.arm != PairArm::Candidate || control.identity.arm != PairArm::Control {
        return Err(FocalBenchmarkError::PairArmOrder {
            pair_index: candidate.identity.pair_index,
        });
    }
    validate_focal_record(candidate)?;
    validate_focal_record(control)?;
    validate_pair_identity(candidate, control)
}

fn validate_record_identity_shape(
    identity: &FocalRecordIdentity,
    focal_seat: u8,
) -> Result<(), FocalBenchmarkError> {
    let expected_seat = (identity.pair_index % 4) as u8;
    if focal_seat != expected_seat {
        return Err(FocalBenchmarkError::WrongRotatingFocalSeat {
            pair_index: identity.pair_index,
            expected: expected_seat,
            actual: focal_seat,
        });
    }
    if identity.opponents.len() != 3 {
        return Err(FocalBenchmarkError::WrongOpponentCount(
            identity.opponents.len(),
        ));
    }
    let mut seats = identity
        .opponents
        .iter()
        .map(|opponent| opponent.seat)
        .collect::<Vec<_>>();
    seats.sort_unstable();
    let expected_seats = (0..4)
        .filter(|seat| *seat != focal_seat)
        .collect::<Vec<_>>();
    if seats != expected_seats {
        return Err(FocalBenchmarkError::WrongOpponentSeats {
            focal_seat,
            actual: seats,
        });
    }
    Ok(())
}

fn validate_pair_identity(
    candidate: &FocalGameRecord,
    control: &FocalGameRecord,
) -> Result<(), FocalBenchmarkError> {
    let pair_index = candidate.identity.pair_index;
    if control.identity.pair_index != pair_index
        || candidate.game_seed != control.game_seed
        || candidate.focal_seat != control.focal_seat
        || candidate.identity.opponents != control.identity.opponents
        || candidate.identity.field_manifest_id != control.identity.field_manifest_id
        || candidate.identity.inference_settings_id != control.identity.inference_settings_id
    {
        return Err(FocalBenchmarkError::PairIdentityDrift(pair_index));
    }
    if candidate
        .identity
        .opponents
        .iter()
        .any(|opponent| opponent.checkpoint_id == candidate.identity.focal_checkpoint_id)
    {
        return Err(FocalBenchmarkError::CandidateInOpponentSeat(pair_index));
    }
    Ok(())
}

fn require_stable_identity<'a>(
    label: &'static str,
    expected: &mut Option<&'a str>,
    actual: &'a str,
) -> Result<(), FocalBenchmarkError> {
    match expected {
        Some(expected) if *expected != actual => {
            Err(FocalBenchmarkError::CrossPairIdentityDrift(label))
        }
        Some(_) => Ok(()),
        None => {
            *expected = Some(actual);
            Ok(())
        }
    }
}

fn validate_wall_seconds(wall_seconds: f64) -> Result<(), FocalBenchmarkError> {
    if !wall_seconds.is_finite() || wall_seconds <= 0.0 {
        Err(FocalBenchmarkError::InvalidWallSeconds(wall_seconds))
    } else {
        Ok(())
    }
}

fn focal_statistics(records: &[&FocalGameRecord]) -> FocalStatistics {
    let score_values = ScoreValues::from_records(records);
    let pinecone_values = PineconeValues::from_records(records);
    FocalStatistics {
        base_total: integer_distribution(&score_values.base_total),
        animals: score_values.animals(),
        terrains: score_values.terrains(),
        pinecones: pinecone_values.distributions(records.len()),
        focal_decision_latency_milliseconds: float_distribution(
            &records
                .iter()
                .flat_map(|record| record.focal_decision_seconds.iter())
                .map(|seconds| seconds * 1_000.0)
                .collect::<Vec<_>>(),
        ),
    }
}

fn paired_delta_statistics(pairs: &[ValidatedPair<'_>]) -> PairedFocalDeltaStatistics {
    let candidate = pairs.iter().map(|pair| pair.candidate).collect::<Vec<_>>();
    let control = pairs.iter().map(|pair| pair.control).collect::<Vec<_>>();
    let score_values = ScoreValues::deltas(
        &ScoreValues::from_records(&candidate),
        &ScoreValues::from_records(&control),
    );
    let pinecone_values = PineconeValues::deltas(
        &PineconeValues::from_records(&candidate),
        &PineconeValues::from_records(&control),
    );
    let latency_deltas = pairs
        .iter()
        .map(|pair| {
            mean(&pair.candidate.focal_decision_seconds) * 1_000.0
                - mean(&pair.control.focal_decision_seconds) * 1_000.0
        })
        .collect::<Vec<_>>();
    PairedFocalDeltaStatistics {
        base_total: integer_distribution(&score_values.base_total),
        animals: score_values.animals(),
        terrains: score_values.terrains(),
        pinecones: pinecone_values.distributions(pairs.len()),
        focal_decision_latency_milliseconds: float_distribution(&latency_deltas),
    }
}

fn classify_fixed_development(
    delta: &IntegerDistribution,
    gates: PromotionGates,
) -> PromotionClassification {
    if !gates.resource_gates_pass
        || !gates.preregistered_guardrails_pass
        || delta.confidence_95[1] <= 0.0
    {
        PromotionClassification::Reject
    } else if delta.mean > 0.0 && delta.confidence_95[0] > 0.0 {
        PromotionClassification::Promote
    } else {
        PromotionClassification::Inconclusive
    }
}

#[derive(Default)]
struct ScoreValues {
    base_total: Vec<i32>,
    wildlife: [Vec<i32>; 5],
    aggregate_wildlife: Vec<i32>,
    habitat: [Vec<i32>; 5],
    aggregate_habitat: Vec<i32>,
}

impl ScoreValues {
    fn from_records(records: &[&FocalGameRecord]) -> Self {
        let mut values = Self::default();
        for record in records {
            values.base_total.push(i32::from(record.score.base_total));
            let mut wildlife_total = 0;
            let mut habitat_total = 0;
            for index in 0..5 {
                let wildlife = i32::from(record.score.wildlife[index]);
                let habitat = i32::from(record.score.habitat[index]);
                values.wildlife[index].push(wildlife);
                values.habitat[index].push(habitat);
                wildlife_total += wildlife;
                habitat_total += habitat;
            }
            values.aggregate_wildlife.push(wildlife_total);
            values.aggregate_habitat.push(habitat_total);
        }
        values
    }

    fn deltas(candidate: &Self, control: &Self) -> Self {
        Self {
            base_total: subtract_vectors(&candidate.base_total, &control.base_total),
            wildlife: std::array::from_fn(|index| {
                subtract_vectors(&candidate.wildlife[index], &control.wildlife[index])
            }),
            aggregate_wildlife: subtract_vectors(
                &candidate.aggregate_wildlife,
                &control.aggregate_wildlife,
            ),
            habitat: std::array::from_fn(|index| {
                subtract_vectors(&candidate.habitat[index], &control.habitat[index])
            }),
            aggregate_habitat: subtract_vectors(
                &candidate.aggregate_habitat,
                &control.aggregate_habitat,
            ),
        }
    }

    fn animals(&self) -> AnimalDistributions {
        AnimalDistributions {
            bear: integer_distribution(&self.wildlife[0]),
            elk: integer_distribution(&self.wildlife[1]),
            salmon: integer_distribution(&self.wildlife[2]),
            hawk: integer_distribution(&self.wildlife[3]),
            fox: integer_distribution(&self.wildlife[4]),
            aggregate_wildlife: integer_distribution(&self.aggregate_wildlife),
        }
    }

    fn terrains(&self) -> TerrainDistributions {
        TerrainDistributions {
            mountain: integer_distribution(&self.habitat[0]),
            forest: integer_distribution(&self.habitat[1]),
            prairie: integer_distribution(&self.habitat[2]),
            wetland: integer_distribution(&self.habitat[3]),
            river: integer_distribution(&self.habitat[4]),
            aggregate_habitat: integer_distribution(&self.aggregate_habitat),
        }
    }
}

#[derive(Default)]
struct PineconeValues {
    earned: Vec<i32>,
    independent_draft_spend: Vec<i32>,
    paid_wipe_spend: Vec<i32>,
    total_spend: Vec<i32>,
    remaining: Vec<i32>,
    free_replacements: Vec<i32>,
}

impl PineconeValues {
    fn from_records(records: &[&FocalGameRecord]) -> Self {
        let mut values = Self::default();
        for record in records {
            values.earned.push(i32::from(record.pinecones.earned));
            values
                .independent_draft_spend
                .push(i32::from(record.pinecones.independent_draft_spend));
            values
                .paid_wipe_spend
                .push(i32::from(record.pinecones.paid_wipe_spend));
            values
                .total_spend
                .push(i32::from(record.pinecones.total_spend));
            values.remaining.push(i32::from(record.pinecones.remaining));
            values
                .free_replacements
                .push(i32::from(record.pinecones.free_replacements));
        }
        values
    }

    fn deltas(candidate: &Self, control: &Self) -> Self {
        Self {
            earned: subtract_vectors(&candidate.earned, &control.earned),
            independent_draft_spend: subtract_vectors(
                &candidate.independent_draft_spend,
                &control.independent_draft_spend,
            ),
            paid_wipe_spend: subtract_vectors(&candidate.paid_wipe_spend, &control.paid_wipe_spend),
            total_spend: subtract_vectors(&candidate.total_spend, &control.total_spend),
            remaining: subtract_vectors(&candidate.remaining, &control.remaining),
            free_replacements: subtract_vectors(
                &candidate.free_replacements,
                &control.free_replacements,
            ),
        }
    }

    fn distributions(&self, conservation_valid_games: usize) -> PineconeDistributions {
        PineconeDistributions {
            earned: integer_distribution(&self.earned),
            independent_draft_spend: integer_distribution(&self.independent_draft_spend),
            paid_wipe_spend: integer_distribution(&self.paid_wipe_spend),
            total_spend: integer_distribution(&self.total_spend),
            remaining: integer_distribution(&self.remaining),
            free_replacements: integer_distribution(&self.free_replacements),
            conservation_valid_games,
        }
    }
}

fn subtract_vectors(candidate: &[i32], control: &[i32]) -> Vec<i32> {
    candidate
        .iter()
        .zip(control)
        .map(|(candidate, control)| candidate - control)
        .collect()
}

fn integer_distribution(values: &[i32]) -> IntegerDistribution {
    debug_assert!(!values.is_empty());
    let floats = values
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let mean = mean(&floats);
    let standard_deviation = sample_standard_deviation(&floats, mean);
    let standard_error = standard_deviation / (values.len() as f64).sqrt();
    let margin = 1.96 * standard_error;
    let mut histogram = BTreeMap::new();
    for value in values {
        *histogram.entry(*value).or_insert(0) += 1;
    }
    IntegerDistribution {
        count: values.len(),
        mean,
        standard_deviation,
        standard_error,
        confidence_95: [mean - margin, mean + margin],
        p10: percentile(&floats, 0.10),
        p50: percentile(&floats, 0.50),
        p90: percentile(&floats, 0.90),
        min: *values.iter().min().expect("non-empty values"),
        max: *values.iter().max().expect("non-empty values"),
        histogram,
    }
}

fn float_distribution(values: &[f64]) -> FloatDistribution {
    debug_assert!(!values.is_empty());
    let mean = mean(values);
    let standard_deviation = sample_standard_deviation(values, mean);
    let standard_error = standard_deviation / (values.len() as f64).sqrt();
    let margin = 1.96 * standard_error;
    FloatDistribution {
        count: values.len(),
        mean,
        standard_deviation,
        standard_error,
        confidence_95: [mean - margin, mean + margin],
        p10: percentile(values, 0.10),
        p50: percentile(values, 0.50),
        p90: percentile(values, 0.90),
        p99: percentile(values, 0.99),
        min: values.iter().copied().fold(f64::INFINITY, f64::min),
        max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_standard_deviation(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    (values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64)
        .sqrt()
}

fn percentile(values: &[f64], quantile: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let position = quantile * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = position - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

#[derive(Debug, Error)]
pub enum FocalBenchmarkError {
    #[error("focal benchmark schema version {0} is unsupported")]
    UnsupportedSchema(u16),
    #[error("focal benchmark protocol mismatch: {0}")]
    ProtocolMismatch(String),
    #[error("benchmark stage does not match the aggregator")]
    StageMismatch,
    #[error("expected {expected} physical games, found {actual}")]
    WrongPhysicalGameCount { expected: usize, actual: usize },
    #[error("pair index {0} is missing")]
    MissingPair(usize),
    #[error("pair index {0} is outside the frozen sample")]
    ExtraPair(usize),
    #[error("pair {pair_index} is missing its {arm:?} arm")]
    MissingArm { pair_index: usize, arm: PairArm },
    #[error("pair {pair_index} repeats its {arm:?} arm")]
    DuplicateArm { pair_index: usize, arm: PairArm },
    #[error("pair {0} candidate/control identities drift")]
    PairIdentityDrift(usize),
    #[error("pair {pair_index} does not contain candidate then control arms")]
    PairArmOrder { pair_index: usize },
    #[error("{0} changes across pairs")]
    CrossPairIdentityDrift(&'static str),
    #[error("pair {0} places the candidate checkpoint in an opponent seat")]
    CandidateInOpponentSeat(usize),
    #[error("pair {pair_index} focal seat should be {expected}, found {actual}")]
    WrongRotatingFocalSeat {
        pair_index: usize,
        expected: u8,
        actual: u8,
    },
    #[error("expected three opponent identities, found {0}")]
    WrongOpponentCount(usize),
    #[error("focal seat {focal_seat} has invalid opponent seats {actual:?}")]
    WrongOpponentSeats { focal_seat: u8, actual: Vec<u8> },
    #[error("invalid focal seat {0}")]
    InvalidFocalSeat(u8),
    #[error("focal benchmark requires the four-player Card A no-bonus config")]
    WrongGameConfig,
    #[error("match and replay seeds differ")]
    SeedMismatch,
    #[error("replay has {turns} turns but match has {decisions} decision timings")]
    DecisionCountMismatch { turns: usize, decisions: usize },
    #[error("recomputed final scores differ from the match result")]
    ScoreMismatch,
    #[error("completed match replay is missing its final state hash")]
    MissingFinalStateHash,
    #[error("pair {0} has an invalid replay BLAKE3 digest")]
    InvalidReplayDigest(usize),
    #[error(
        "pair {pair_index} {arm:?} score arithmetic mismatch: expected base {expected_base_total}, found {score:?}"
    )]
    ScoreArithmeticMismatch {
        pair_index: usize,
        arm: PairArm,
        expected_base_total: u16,
        score: ScoreBreakdown,
    },
    #[error("pair {pair_index} should contain {expected} focal decisions, found {actual}")]
    WrongFocalDecisionCount {
        pair_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error("pair {0} has invalid focal decision latency")]
    InvalidLatency(usize),
    #[error("Pinecone counter overflow")]
    PineconeCounterOverflow,
    #[error(
        "pair {pair_index} turn {turn_index} has invalid Pinecone transition: before={before}, spend={spend}, after={after}"
    )]
    InvalidPineconeTransition {
        pair_index: usize,
        turn_index: usize,
        before: u8,
        spend: u16,
        after: u8,
    },
    #[error("pair {pair_index} {arm:?} violates Pinecone conservation: {pinecones:?}")]
    PineconeConservation {
        pair_index: usize,
        arm: PairArm,
        pinecones: PineconeObservation,
    },
    #[error("wall time must be finite and positive, found {0}")]
    InvalidWallSeconds(f64),
    #[error(transparent)]
    Replay(#[from] ReplayError),
    #[error(transparent)]
    Rules(#[from] RuleError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_game::{
        HexCoord, MarketSlot, Rotation, ScoringCards, TilePlacement, TurnAction, WildlifeWipe,
    };
    use cascadia_sim::{MatchConfig, StrategyKind, play_match};

    fn opponents(focal_seat: u8) -> Vec<OpponentIdentity> {
        (0..4)
            .filter(|seat| *seat != focal_seat)
            .map(|seat| OpponentIdentity {
                seat,
                checkpoint_id: format!("historical-{seat}"),
            })
            .collect()
    }

    fn synthetic_record(
        stage: BenchmarkStage,
        pair_index: usize,
        arm: PairArm,
        base_total: u16,
    ) -> FocalGameRecord {
        let focal_seat = (pair_index % 4) as u8;
        let checkpoint = match arm {
            PairArm::Candidate => "candidate-v1",
            PairArm::Control => "control-v1",
        };
        let remaining = base_total % 3;
        let habitat = [4; 5];
        let wildlife_total = base_total - habitat.iter().sum::<u16>() - remaining;
        let mut wildlife = [wildlife_total / 5; 5];
        for value in wildlife.iter_mut().take(usize::from(wildlife_total % 5)) {
            *value += 1;
        }
        let independent_draft_spend = u16::from(base_total.is_multiple_of(2));
        let paid_wipe_spend = u16::from(base_total.is_multiple_of(5));
        let total_spend = independent_draft_spend + paid_wipe_spend;
        FocalGameRecord {
            schema_version: FOCAL_BENCHMARK_SCHEMA_VERSION,
            protocol_id: FOCAL_BENCHMARK_PROTOCOL_ID.to_owned(),
            identity: FocalRecordIdentity {
                stage,
                pair_index,
                arm,
                focal_checkpoint_id: checkpoint.to_owned(),
                opponents: opponents(focal_seat),
                field_manifest_id: "field-v1".to_owned(),
                inference_settings_id: "argmax-v1".to_owned(),
            },
            game_seed: GameSeed::from_u64(10_000 + pair_index as u64),
            focal_seat,
            final_state_hash: [base_total as u8; 32],
            replay_blake3: format!("{base_total:064x}"),
            score: ScoreBreakdown {
                habitat,
                wildlife,
                nature_tokens: remaining,
                habitat_bonus: [0; 5],
                base_total,
                total: base_total,
            },
            pinecones: PineconeObservation {
                earned: total_spend + remaining,
                independent_draft_spend,
                paid_wipe_spend,
                total_spend,
                remaining,
                free_replacements: u16::from(base_total.is_multiple_of(7)),
            },
            focal_decision_seconds: vec![0.001; 20],
            elapsed_seconds: 0.1,
            runtime: FocalRuntimeObservation {
                checkpoint_load_seconds: 0.02,
                peak_rss_bytes: 123,
                swap_delta_bytes: 0,
                clean_shutdown: true,
            },
        }
    }

    fn records(stage: BenchmarkStage, candidate_delta: i16) -> Vec<FocalGameRecord> {
        let mut records = Vec::new();
        for pair_index in 0..stage.expected_pairs() {
            let control_score = 90 + (pair_index % 5) as u16;
            let candidate_score =
                u16::try_from(i32::from(control_score) + i32::from(candidate_delta)).unwrap();
            records.push(synthetic_record(
                stage,
                pair_index,
                PairArm::Candidate,
                candidate_score,
            ));
            records.push(synthetic_record(
                stage,
                pair_index,
                PairArm::Control,
                control_score,
            ));
        }
        records
    }

    #[test]
    fn sealed_match_derives_focal_only_pinecones_and_latency() {
        let game = GameConfig::research_aaaaa(4).unwrap();
        let result = play_match(&MatchConfig::symmetric(
            game,
            GameSeed::from_u64(42),
            StrategyKind::Random,
        ))
        .unwrap();
        let identity = FocalRecordIdentity {
            stage: BenchmarkStage::StrengthBlindedSmoke,
            pair_index: 1,
            arm: PairArm::Candidate,
            focal_checkpoint_id: "candidate-v1".to_owned(),
            opponents: opponents(1),
            field_manifest_id: "field-v1".to_owned(),
            inference_settings_id: "argmax-v1".to_owned(),
        };
        let record =
            FocalGameRecord::from_match(identity, 1, FocalRuntimeObservation::default(), &result)
                .unwrap();
        assert_eq!(record.focal_decision_seconds.len(), 20);
        assert!(record.pinecones.conservation_holds());
        assert_eq!(record.score, result.scores[1]);
    }

    #[test]
    fn pinecone_accounting_counts_every_paid_wipe_and_preserves_conservation() {
        let paired = TurnAction::paired(MarketSlot::ZERO, HexCoord::new(0, 0), Rotation::ZERO);
        let spending_action = TurnAction {
            replace_three_of_a_kind: true,
            wildlife_wipes: vec![
                WildlifeWipe {
                    slots: vec![MarketSlot::ZERO],
                },
                WildlifeWipe {
                    slots: vec![MarketSlot::ONE],
                },
            ],
            draft: DraftChoice::Independent {
                tile_slot: MarketSlot::TWO,
                wildlife_slot: MarketSlot::THREE,
            },
            tile: TilePlacement {
                coord: HexCoord::new(0, 0),
                rotation: Rotation::ZERO,
            },
            wildlife: None,
        };
        let mut accumulator = FocalPineconeAccumulator::default();
        accumulator.observe_turn(7, 0, 0, 1, &paired).unwrap();
        accumulator.observe_turn(7, 4, 1, 2, &paired).unwrap();
        accumulator.observe_turn(7, 8, 2, 3, &paired).unwrap();
        accumulator
            .observe_turn(7, 12, 3, 1, &spending_action)
            .unwrap();

        let observation = accumulator.finish(1).unwrap();
        assert_eq!(observation.earned, 4);
        assert_eq!(observation.independent_draft_spend, 1);
        assert_eq!(observation.paid_wipe_spend, 2);
        assert_eq!(observation.total_spend, 3);
        assert_eq!(observation.remaining, 1);
        assert_eq!(observation.free_replacements, 1);
        assert!(observation.conservation_holds());
    }

    #[test]
    fn integer_distributions_have_interpolated_tails_and_exact_histograms() {
        let distribution = integer_distribution(&[1, 1, 2, 4, 7]);
        assert_eq!(distribution.count, 5);
        assert_eq!(distribution.mean, 3.0);
        assert_eq!(distribution.p10, 1.0);
        assert_eq!(distribution.p50, 2.0);
        assert!((distribution.p90 - 5.8).abs() < 1e-12);
        assert_eq!(distribution.min, 1);
        assert_eq!(distribution.max, 7);
        assert_eq!(
            distribution.histogram,
            BTreeMap::from([(1, 2), (2, 1), (4, 1), (7, 1)])
        );
    }

    #[test]
    fn smoke_serialization_is_strength_blinded() {
        let report = aggregate_strength_blinded_smoke(
            &records(BenchmarkStage::StrengthBlindedSmoke, 20),
            4.0,
        )
        .unwrap();
        let json = serde_json::to_string(&report).unwrap();
        assert!(report.strength_outputs_blinded);
        assert_eq!(report.pairs, SMOKE_PAIR_COUNT);
        assert_eq!(report.physical_games, SMOKE_PAIR_COUNT * 2);
        assert!(!json.contains("base_total"));
        assert!(!json.contains("candidate_checkpoint"));
        assert!(!json.contains("paired_delta"));
    }

    #[test]
    fn fixed_250_protocol_classifies_promote_reject_and_inconclusive() {
        let promoted = aggregate_development_comparison(
            &records(BenchmarkStage::Development, 2),
            50.0,
            PromotionGates::default(),
        )
        .unwrap();
        assert_eq!(promoted.classification, PromotionClassification::Promote);
        assert_eq!(promoted.pairs, DEVELOPMENT_PAIR_COUNT);
        assert_eq!(promoted.physical_games, DEVELOPMENT_PAIR_COUNT * 2);
        assert_eq!(
            promoted.paired_delta.base_total.histogram,
            BTreeMap::from([(2, 250)])
        );

        let rejected = aggregate_development_comparison(
            &records(BenchmarkStage::Development, -2),
            50.0,
            PromotionGates::default(),
        )
        .unwrap();
        assert_eq!(rejected.classification, PromotionClassification::Reject);

        let tied = aggregate_development_comparison(
            &records(BenchmarkStage::Development, 0),
            50.0,
            PromotionGates::default(),
        )
        .unwrap();
        assert_eq!(tied.classification, PromotionClassification::Reject);
    }

    #[test]
    fn aggregation_is_byte_identical_in_forward_and_reverse_order() {
        let forward = records(BenchmarkStage::Development, 2);
        let mut reverse = forward.clone();
        reverse.reverse();
        let left =
            aggregate_development_comparison(&forward, 50.0, PromotionGates::default()).unwrap();
        let right =
            aggregate_development_comparison(&reverse, 50.0, PromotionGates::default()).unwrap();
        assert_eq!(
            serde_json::to_vec(&left).unwrap(),
            serde_json::to_vec(&right).unwrap()
        );
    }

    #[test]
    fn duplicate_missing_tampered_and_wrong_identity_records_are_rejected() {
        let mut duplicate = records(BenchmarkStage::StrengthBlindedSmoke, 1);
        duplicate[1] = duplicate[0].clone();
        assert!(matches!(
            aggregate_strength_blinded_smoke(&duplicate, 1.0),
            Err(FocalBenchmarkError::DuplicateArm { .. })
        ));

        let mut tampered = records(BenchmarkStage::StrengthBlindedSmoke, 1);
        tampered[1].game_seed = GameSeed::from_u64(999);
        assert!(matches!(
            aggregate_strength_blinded_smoke(&tampered, 1.0),
            Err(FocalBenchmarkError::PairIdentityDrift(0))
        ));

        let mut missing = records(BenchmarkStage::StrengthBlindedSmoke, 1);
        missing.pop();
        assert!(matches!(
            aggregate_strength_blinded_smoke(&missing, 1.0),
            Err(FocalBenchmarkError::WrongPhysicalGameCount { .. })
        ));
    }

    #[test]
    fn opponent_scores_cannot_enter_focal_statistics() {
        let mut focal = synthetic_record(BenchmarkStage::Development, 0, PairArm::Candidate, 101);
        focal.score.wildlife = [10, 11, 12, 13, 14];
        let stats = focal_statistics(&[&focal]);
        assert_eq!(stats.base_total.mean, 101.0);
        assert_eq!(stats.animals.bear.mean, 10.0);
        assert_eq!(stats.animals.aggregate_wildlife.mean, 60.0);
    }

    #[test]
    fn game_config_contract_remains_card_a_without_habitat_bonuses() {
        let config = GameConfig::research_aaaaa(4).unwrap();
        assert_eq!(config.scoring_cards, ScoringCards::AAAAA);
        assert!(!config.habitat_bonuses);
    }
}
