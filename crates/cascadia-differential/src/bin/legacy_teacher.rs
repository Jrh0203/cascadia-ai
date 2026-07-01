use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use cascadia_ai::{
    eval::ScoredMove,
    mce::{MceMoveEstimate, nnue_prefilter_candidates, score_nnue_rollout_mce_seq_halving},
    nnue::{BagInfo, HIDDEN1, HIDDEN2, NUM_FEATURES, extract_features_with_bag},
    nnue_batch::{
        BatchedNnueDiagnostics, RolloutSeedCoupling, SparseNnueEvaluator,
        nnue_prefilter_candidates_batched, score_nnue_rollout_mce_seq_halving_batched,
    },
};
use cascadia_data::{
    DatasetSplit, ImitationCandidateConfig, ImitationDatasetConfig, ImitationDatasetManifest,
    ImitationDatasetWriter, ImitationParentHiddenDatasetConfig,
    ImitationParentHiddenDatasetManifest, ImitationParentHiddenDatasetWriter,
    ImitationParentHiddenRecord, ImitationParentPriorDatasetConfig,
    ImitationParentPriorDatasetManifest, ImitationParentPriorDatasetWriter,
    ImitationParentPriorRecord, ImitationRecord, ImitationTargetRecord,
    ImitationTargetsDatasetConfig, ImitationTargetsDatasetManifest, ImitationTargetsDatasetWriter,
    ImitationTeacherConfig, PositionRecord, ProposalPositionRecord, RolloutValueDatasetConfig,
    RolloutValueDatasetManifest, RolloutValueDatasetWriter, RolloutValueRecord,
    RolloutValueRecordKind, RolloutValueTeacherConfig, SOURCE_DETERMINISTIC_NEGATIVE,
    SOURCE_IMMEDIATE_TOP, SOURCE_PATTERN_FRONTIER, SOURCE_TEACHER_FRONTIER,
    read_imitation_shard_records, read_imitation_target_shard_records, validate_imitation_dataset,
    validate_imitation_parent_hidden_dataset, validate_imitation_parent_prior_dataset,
    validate_imitation_targets_dataset, validate_rollout_value_dataset,
};
use cascadia_differential::legacy_teacher::{
    BridgeDiagnostics, DETERMINISTIC_EVIDENCE_TEACHER_STRATEGY_ID,
    EXACT_MLX_LEGACY_TEACHER_STRATEGY_ID, ExactMlxLegacyTeacher,
    FILTERED_LEGACY_TEACHER_STRATEGY_ID, HEURISTIC_LEGACY_TEACHER_STRATEGY_ID,
    LEGACY_TEACHER_STRATEGY_ID, LegacyTeacher, audit_filtered_pattern_trajectory,
    audit_heuristic_pattern_trajectory, audit_pattern_trajectory,
    audit_retained_pattern_trajectory, canonical_prelude, legacy_search_rng, load_legacy_weights,
    map_legacy_action, pattern_fallback, simulation_error,
    translate_public_state_allowing_legacy_elk_undercount, validate_legacy_environment,
};
use cascadia_eval::{ComparisonReport, summarize_paired_match_results};
use cascadia_game::{GameConfig, GameSeed, GameState, ScoreBreakdown, TurnAction};
use cascadia_model::{ModelError, ModelProcess};
use cascadia_provenance::{SourceProvenance, checksum_file, source_provenance};
use cascadia_search::{
    LateConservativeBasePolicyImprovementConfig, LateConservativeBasePolicyImprovementStrategy,
};
use cascadia_sim::{
    GreedyCandidate, MatchResult, PATTERN_AWARE_STRATEGY_ID, PatternAwareConfig,
    play_match_with_selector, rank_greedy_actions, rank_pattern_actions, strategy_rng,
};
use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};

type ImitationEvidence = (Vec<ImitationRecord>, Vec<ImitationTargetRecord>, usize);
type PairedMatchResults = Vec<(u64, MatchResult, MatchResult)>;

#[derive(Debug, Parser)]
#[command(about = "Isolated public-state evaluation of the historical v1 teacher")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Compatibility {
        #[arg(long)]
        games: usize,
        #[arg(long)]
        first_seed: u64,
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    RetainedCompatibility {
        #[arg(long)]
        games: usize,
        #[arg(long)]
        first_seed: u64,
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    FilteredCompatibility {
        #[arg(long)]
        games: usize,
        #[arg(long)]
        first_seed: u64,
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    HeuristicCompatibility {
        #[arg(long)]
        games: usize,
        #[arg(long)]
        first_seed: u64,
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Compare {
        #[arg(long)]
        games: usize,
        #[arg(long)]
        first_seed: u64,
        #[arg(long, default_value_t = 600)]
        rollouts: usize,
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    RetainedCompare {
        #[arg(long)]
        games: usize,
        #[arg(long)]
        first_seed: u64,
        #[arg(long, default_value_t = 600)]
        rollouts: usize,
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    FilteredCompare {
        #[arg(long)]
        games: usize,
        #[arg(long)]
        first_seed: u64,
        #[arg(long, default_value_t = 600)]
        rollouts: usize,
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    HeuristicCompare {
        #[arg(long)]
        games: usize,
        #[arg(long)]
        first_seed: u64,
        #[arg(long, default_value_t = 600)]
        rollouts: usize,
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    ProductiveTokenCompare {
        #[arg(long)]
        games: usize,
        #[arg(long)]
        first_seed: u64,
        #[arg(long, default_value_t = 600)]
        rollouts: usize,
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    ExactMlxProductiveTokenCompare {
        #[arg(long, default_value = "uv")]
        server_program: String,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        games: usize,
        #[arg(long)]
        first_seed: u64,
        #[arg(long, value_enum)]
        split: Option<SplitArg>,
        #[arg(long, default_value_t = 600)]
        rollouts: usize,
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    ExactMlxRolloutBudgetCompare {
        #[arg(long, default_value = "uv")]
        server_program: String,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        games: usize,
        #[arg(long)]
        first_seed: u64,
        #[arg(long, default_value_t = 600)]
        baseline_rollouts: usize,
        #[arg(long, default_value_t = 1200)]
        treatment_rollouts: usize,
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    ExactMlxCrnCompare {
        #[arg(long, default_value = "uv")]
        server_program: String,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        games: usize,
        #[arg(long)]
        first_seed: u64,
        #[arg(long, default_value_t = 600)]
        rollouts: usize,
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    ExactMlxCrnConfirm {
        #[arg(long, default_value = "uv")]
        server_program: String,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        games: usize,
        #[arg(long)]
        first_seed: u64,
        #[arg(long, default_value_t = 600)]
        rollouts: usize,
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    ExactMlxCandidateLimitCompare {
        #[arg(long, default_value = "uv")]
        server_program: String,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        games: usize,
        #[arg(long)]
        first_seed: u64,
        #[arg(long, default_value_t = 32)]
        baseline_candidate_limit: usize,
        #[arg(long, default_value_t = 64)]
        treatment_candidate_limit: usize,
        #[arg(long, default_value_t = 600)]
        rollouts: usize,
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    ExactMlxHabitatCandidateCompare {
        #[arg(long, default_value = "uv")]
        server_program: String,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        games: usize,
        #[arg(long)]
        first_seed: u64,
        #[arg(long, default_value_t = 6)]
        habitat_candidates: usize,
        #[arg(long, default_value_t = 600)]
        rollouts: usize,
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    CollectExactMlxRolloutValues {
        #[arg(long, default_value = "uv")]
        server_program: String,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_game_index: u64,
        #[arg(long, value_enum, default_value_t = SplitArg::Train)]
        split: SplitArg,
        #[arg(long)]
        resume: bool,
        #[arg(long, default_value_t = 600)]
        rollouts: usize,
        #[arg(long, default_value_t = 8)]
        trace_modulus: u64,
        #[arg(long)]
        weights: PathBuf,
    },
    ValidateExactMlxRolloutValues {
        #[arg(long)]
        dataset: PathBuf,
    },
    FrontierRecallProbe {
        #[arg(long)]
        games: usize,
        #[arg(long)]
        first_seed: u64,
        #[arg(long, default_value_t = 600)]
        rollouts: usize,
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    CollectImitation {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_game_index: u64,
        #[arg(long, value_enum)]
        split: SplitArg,
        #[arg(long, default_value_t = 1)]
        shard_games: usize,
        #[arg(long)]
        resume: bool,
        #[arg(long, default_value_t = 64)]
        group_limit: usize,
        #[arg(long, default_value_t = 16)]
        immediate_limit: usize,
        #[arg(long, default_value_t = 600)]
        rollouts: usize,
        #[arg(long)]
        weights: PathBuf,
    },
    ValidateImitationDataset {
        #[arg(long)]
        dataset: PathBuf,
    },
    TeacherEstimateParity {
        #[arg(long, default_value_t = 1)]
        games: usize,
        #[arg(long, default_value_t = 90000)]
        first_game_index: u64,
        #[arg(long, value_enum, default_value_t = SplitArg::Train)]
        split: SplitArg,
        #[arg(long, default_value_t = 600)]
        rollouts: usize,
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    EnrichImitationTargets {
        #[arg(long)]
        source_dataset: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        resume: bool,
        #[arg(long)]
        weights: PathBuf,
    },
    CollectImitationEvidence {
        #[arg(long)]
        source_output: PathBuf,
        #[arg(long)]
        targets_output: PathBuf,
        #[arg(long)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_game_index: u64,
        #[arg(long, value_enum)]
        split: SplitArg,
        #[arg(long)]
        resume: bool,
        #[arg(long, default_value_t = 96)]
        group_limit: usize,
        #[arg(long, default_value_t = 16)]
        immediate_limit: usize,
        #[arg(long, default_value_t = 600)]
        rollouts: usize,
        #[arg(long)]
        weights: PathBuf,
    },
    ValidateImitationTargets {
        #[arg(long)]
        dataset: PathBuf,
    },
    CollectImitationParentPriors {
        #[arg(long, default_value = "uv")]
        server_program: String,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        source_dataset: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        resume: bool,
    },
    ValidateImitationParentPriors {
        #[arg(long)]
        dataset: PathBuf,
    },
    CollectImitationParentHidden {
        #[arg(long, default_value = "uv")]
        server_program: String,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        source_dataset: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        resume: bool,
    },
    ValidateImitationParentHidden {
        #[arg(long)]
        dataset: PathBuf,
    },
    NnueParityFixture {
        #[arg(long, default_value_t = 1)]
        games: usize,
        #[arg(long, default_value_t = 92000)]
        first_game_index: u64,
        #[arg(long, value_enum, default_value_t = SplitArg::Train)]
        split: SplitArg,
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    NnueServiceParity {
        #[arg(long, default_value = "uv")]
        server_program: String,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        fixture: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 200)]
        iterations: usize,
    },
    NnueExactServiceParity {
        #[arg(long, default_value = "uv")]
        server_program: String,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        fixture: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 200)]
        iterations: usize,
    },
    NnueRolloutWaveParity {
        #[arg(long, default_value = "uv")]
        server_program: String,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        fixture: PathBuf,
        #[arg(long)]
        weights: PathBuf,
        #[arg(long, default_value_t = 92100)]
        game_index: u64,
        #[arg(long, default_value_t = 32)]
        rollouts: usize,
        #[arg(long, value_delimiter = ',', default_value = "0,39,79")]
        spot_decisions: Vec<usize>,
        #[arg(long, default_value_t = 600)]
        spot_rollouts: usize,
        #[arg(long)]
        output: PathBuf,
    },
    NnueExactRolloutWaveParity {
        #[arg(long, default_value = "uv")]
        server_program: String,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        fixture: PathBuf,
        #[arg(long)]
        weights: PathBuf,
        #[arg(long, default_value_t = 92100)]
        game_index: u64,
        #[arg(long, default_value_t = 32)]
        rollouts: usize,
        #[arg(long, value_delimiter = ',', default_value = "0,39,79")]
        spot_decisions: Vec<usize>,
        #[arg(long, default_value_t = 600)]
        spot_rollouts: usize,
        #[arg(long)]
        max_decisions: Option<usize>,
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SplitArg {
    Train,
    Validation,
    Test,
    Final,
}

impl From<SplitArg> for DatasetSplit {
    fn from(value: SplitArg) -> Self {
        match value {
            SplitArg::Train => Self::Train,
            SplitArg::Validation => Self::Validation,
            SplitArg::Test => Self::Test,
            SplitArg::Final => Self::Final,
        }
    }
}

#[derive(Debug, Serialize)]
struct ArtifactProvenance {
    source: SourceProvenance,
    executable_path: PathBuf,
    executable_blake3: String,
    weights_path: PathBuf,
    weights_blake3: String,
    legacy_environment: Vec<(String, String)>,
}

#[derive(Debug, Serialize)]
struct CompatibilityReport {
    schema_version: u16,
    experiment_id: &'static str,
    status: &'static str,
    games: usize,
    first_seed: u64,
    last_seed: u64,
    expected_states: usize,
    diagnostics: BridgeDiagnostics,
    expanded_malformed_rate: f64,
    passed: bool,
    failure: Option<String>,
    elapsed_seconds: f64,
    provenance: ArtifactProvenance,
}

#[derive(Debug, Serialize)]
struct StrengthGates {
    bridge_integrity_passed: bool,
    runtime_passed: bool,
    smoke_passed: bool,
    treatment_mean_passed: bool,
    paired_gain_passed: bool,
    paired_confidence_passed: bool,
    wildlife_passed: bool,
    habitat_passed: bool,
    nature_tokens_passed: bool,
    non_token_score_passed: bool,
    token_efficiency_passed: bool,
    frontier_recall_passed: bool,
    qualification_passed: bool,
}

#[derive(Debug, Serialize)]
struct StrengthReport {
    schema_version: u16,
    experiment_id: &'static str,
    status: &'static str,
    rollouts: usize,
    diagnostics: BridgeDiagnostics,
    fallback_rate: f64,
    expanded_malformed_rate: f64,
    total_wildlife_delta: f64,
    habitat_delta: f64,
    non_token_score_delta: f64,
    token_spend: f64,
    board_points_per_token: Option<f64>,
    pattern_frontier_recall: f64,
    independent_pattern_frontier_recall: f64,
    pattern_frontier_recall_by_phase: [f64; 3],
    gates: StrengthGates,
    comparison: ComparisonReport,
    provenance: ArtifactProvenance,
}

#[derive(Debug, Serialize)]
struct ExactMlxStrengthReport {
    schema_version: u16,
    experiment_id: &'static str,
    status: &'static str,
    seed_domain: &'static str,
    rollouts: usize,
    diagnostics: BridgeDiagnostics,
    batch_diagnostics: SearchBatchDiagnostics,
    fallback_rate: f64,
    expanded_malformed_rate: f64,
    total_wildlife_delta: f64,
    habitat_delta: f64,
    non_token_score_delta: f64,
    token_spend: f64,
    board_points_per_token: Option<f64>,
    service_startup_milliseconds: f64,
    clean_shutdown: bool,
    gates: StrengthGates,
    comparison: ComparisonReport,
    game_records: Vec<ExactMlxGameRecord>,
    model_manifest_path: PathBuf,
    model_manifest_blake3: String,
    model_safetensors_blake3: String,
    provenance: ArtifactProvenance,
}

#[derive(Debug, Serialize)]
struct ExactMlxGameRecord {
    seed: u64,
    game_seed: GameSeed,
    baseline_scores: Vec<ScoreBreakdown>,
    treatment_scores: Vec<ScoreBreakdown>,
    baseline_decision_seconds: Vec<f64>,
    treatment_decision_seconds: Vec<f64>,
    baseline_elapsed_seconds: f64,
    treatment_elapsed_seconds: f64,
}

#[derive(Debug, Serialize)]
struct ExactMlxRolloutBudgetGates {
    baseline_integrity: bool,
    treatment_integrity: bool,
    baseline_runtime: bool,
    treatment_runtime: bool,
    clean_shutdown: bool,
    smoke_passed: bool,
    paired_gain: bool,
    paired_confidence: bool,
    treatment_mean: bool,
    wildlife: bool,
    habitat: bool,
    nature_tokens: bool,
    pilot_promising: bool,
}

#[derive(Debug, Serialize)]
struct ExactMlxRolloutBudgetReport {
    schema_version: u16,
    experiment_id: &'static str,
    status: &'static str,
    baseline_rollouts: usize,
    treatment_rollouts: usize,
    baseline_diagnostics: BridgeDiagnostics,
    treatment_diagnostics: BridgeDiagnostics,
    baseline_batch_diagnostics: SearchBatchDiagnostics,
    treatment_batch_diagnostics: SearchBatchDiagnostics,
    baseline_startup_milliseconds: f64,
    treatment_startup_milliseconds: f64,
    baseline_clean_shutdown: bool,
    treatment_clean_shutdown: bool,
    total_wildlife_delta: f64,
    habitat_delta: f64,
    gates: ExactMlxRolloutBudgetGates,
    comparison: ComparisonReport,
    model_manifest_path: PathBuf,
    model_manifest_blake3: String,
    model_safetensors_blake3: String,
    provenance: ArtifactProvenance,
}

#[derive(Debug, Serialize)]
struct ExactMlxCrnReport {
    schema_version: u16,
    experiment_id: &'static str,
    status: &'static str,
    rollouts: usize,
    baseline_seed_coupling: &'static str,
    treatment_seed_coupling: &'static str,
    baseline_diagnostics: BridgeDiagnostics,
    treatment_diagnostics: BridgeDiagnostics,
    baseline_batch_diagnostics: SearchBatchDiagnostics,
    treatment_batch_diagnostics: SearchBatchDiagnostics,
    baseline_startup_milliseconds: f64,
    treatment_startup_milliseconds: f64,
    baseline_clean_shutdown: bool,
    treatment_clean_shutdown: bool,
    total_wildlife_delta: f64,
    habitat_delta: f64,
    gates: ExactMlxRolloutBudgetGates,
    comparison: ComparisonReport,
    model_manifest_path: PathBuf,
    model_manifest_blake3: String,
    model_safetensors_blake3: String,
    provenance: ArtifactProvenance,
}

#[derive(Debug, Serialize)]
struct ExactMlxCandidateLimitReport {
    schema_version: u16,
    experiment_id: &'static str,
    status: &'static str,
    baseline_candidate_limit: usize,
    treatment_candidate_limit: usize,
    rollouts: usize,
    baseline_diagnostics: BridgeDiagnostics,
    treatment_diagnostics: BridgeDiagnostics,
    baseline_batch_diagnostics: SearchBatchDiagnostics,
    treatment_batch_diagnostics: SearchBatchDiagnostics,
    baseline_startup_milliseconds: f64,
    treatment_startup_milliseconds: f64,
    baseline_clean_shutdown: bool,
    treatment_clean_shutdown: bool,
    total_wildlife_delta: f64,
    habitat_delta: f64,
    gates: ExactMlxRolloutBudgetGates,
    comparison: ComparisonReport,
    model_manifest_path: PathBuf,
    model_manifest_blake3: String,
    model_safetensors_blake3: String,
    provenance: ArtifactProvenance,
}

#[derive(Debug, Serialize)]
struct ExactMlxHabitatCandidateReport {
    schema_version: u16,
    experiment_id: &'static str,
    status: &'static str,
    candidate_limit: usize,
    habitat_candidates: usize,
    rollouts: usize,
    baseline_diagnostics: BridgeDiagnostics,
    treatment_diagnostics: BridgeDiagnostics,
    baseline_batch_diagnostics: SearchBatchDiagnostics,
    treatment_batch_diagnostics: SearchBatchDiagnostics,
    baseline_startup_milliseconds: f64,
    treatment_startup_milliseconds: f64,
    baseline_clean_shutdown: bool,
    treatment_clean_shutdown: bool,
    total_wildlife_delta: f64,
    habitat_delta: f64,
    gates: ExactMlxRolloutBudgetGates,
    comparison: ComparisonReport,
    model_manifest_path: PathBuf,
    model_manifest_blake3: String,
    model_safetensors_blake3: String,
    provenance: ArtifactProvenance,
}

#[derive(Debug, Serialize)]
struct TeacherEstimateParityReport {
    schema_version: u16,
    games: usize,
    first_game_index: u64,
    split: DatasetSplit,
    rollouts: usize,
    states: usize,
    estimates: usize,
    minimum_samples: u32,
    maximum_samples: u32,
    passed: bool,
    elapsed_seconds: f64,
    provenance: ArtifactProvenance,
}

#[derive(Debug, Serialize)]
struct NnueParityRecord {
    game_index: u64,
    decision_index: usize,
    active_seat: usize,
    features: Vec<u16>,
    rust_value: f32,
}

#[derive(Debug, Deserialize)]
struct NnueParityFixtureInput {
    records: Vec<NnueParityRecordInput>,
}

#[derive(Debug, Deserialize)]
struct NnueParityRecordInput {
    features: Vec<u16>,
    rust_value: f32,
}

#[derive(Debug, Serialize)]
struct NnueParityFixture {
    schema_version: u16,
    feature_schema: &'static str,
    split: DatasetSplit,
    first_game_index: u64,
    games: usize,
    feature_count: usize,
    hidden1: usize,
    hidden2: usize,
    records_with_duplicate_features: usize,
    duplicate_feature_occurrences: usize,
    maximum_feature_multiplicity: usize,
    records: Vec<NnueParityRecord>,
    provenance: ArtifactProvenance,
}

#[derive(Debug, Serialize)]
struct NnueServiceBenchmark {
    batch_size: usize,
    iterations: usize,
    p50_milliseconds: f64,
    p90_milliseconds: f64,
    p99_milliseconds: f64,
    evaluations_per_second: f64,
}

#[derive(Debug, Serialize)]
struct NnueServiceGates {
    fixture_records: bool,
    finite: bool,
    maximum_rust_error: bool,
    deterministic_repeat: bool,
    clean_shutdown: bool,
    batch32_throughput: bool,
    batch32_p99_latency: bool,
}

#[derive(Debug, Serialize)]
struct NnueServiceReport {
    schema_version: u16,
    experiment_id: &'static str,
    operation: &'static str,
    device_service: &'static str,
    fixture_records: usize,
    maximum_absolute_error_vs_rust: f64,
    mean_absolute_error_vs_rust: f64,
    deterministic_repeat: bool,
    startup_milliseconds: f64,
    benchmarks: Vec<NnueServiceBenchmark>,
    gates: NnueServiceGates,
    passed: bool,
    model_manifest_path: PathBuf,
    model_manifest_blake3: String,
    model_safetensors_blake3: String,
    fixture_path: PathBuf,
    fixture_blake3: String,
    source: SourceProvenance,
    executable_path: PathBuf,
    executable_blake3: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
struct MoveIdentity {
    market_index: usize,
    wildlife_market_index: Option<usize>,
    tile_q: i8,
    tile_r: i8,
    rotation: u8,
    wildlife_q: Option<i8>,
    wildlife_r: Option<i8>,
}

#[derive(Debug, Default, Serialize)]
struct RolloutParityMetrics {
    decisions: usize,
    candidate_identity_mismatches: usize,
    selected_action_mismatches: usize,
    sample_count_mismatches: usize,
    estimate_count: usize,
    maximum_rollout_mean_absolute_error: f64,
    mean_rollout_mean_absolute_error: f64,
    total_seconds: f64,
}

#[derive(Debug, Default, Serialize)]
struct SearchBatchDiagnostics {
    neural_batches: u64,
    neural_rows: u64,
    minimum_batch_rows: usize,
    maximum_batch_rows: usize,
    rollout_waves: u64,
    rollout_samples: u64,
    policy_fallbacks: u64,
}

#[derive(Debug, Serialize)]
struct NnueRolloutWaveGates {
    new_native_exact: bool,
    mlx_r32_candidates: bool,
    mlx_r32_selected_actions: bool,
    mlx_r32_samples: bool,
    mlx_r32_maximum_error: bool,
    mlx_r32_mean_error: bool,
    mlx_repeat_deterministic: bool,
    mlx_r600_candidates: bool,
    mlx_r600_selected_actions: bool,
    mlx_r600_samples: bool,
    mlx_r600_maximum_error: bool,
    mlx_r600_mean_error: bool,
    mlx_zero_fallbacks: bool,
    mlx_runtime_ratio: bool,
    canonical_trajectory: bool,
    clean_shutdown: bool,
}

#[derive(Debug, Serialize)]
struct NnueRolloutWaveReport {
    schema_version: u16,
    experiment_id: &'static str,
    operation: &'static str,
    game_index: u64,
    rollouts: usize,
    spot_decisions: Vec<usize>,
    spot_rollouts: usize,
    trajectory_decisions: usize,
    new_native: RolloutParityMetrics,
    mlx: RolloutParityMetrics,
    mlx_repeat: RolloutParityMetrics,
    mlx_spot: RolloutParityMetrics,
    native_r32_seconds: f64,
    mlx_r32_seconds: f64,
    mlx_native_runtime_ratio: f64,
    new_native_diagnostics: SearchBatchDiagnostics,
    mlx_diagnostics: SearchBatchDiagnostics,
    mlx_repeat_diagnostics: SearchBatchDiagnostics,
    mlx_spot_diagnostics: SearchBatchDiagnostics,
    gates: NnueRolloutWaveGates,
    passed: bool,
    model_manifest_blake3: String,
    fixture_blake3: String,
    weights_blake3: String,
    source: SourceProvenance,
    executable_path: PathBuf,
    executable_blake3: String,
}

#[derive(Debug, Clone, Copy)]
struct SampledCandidate<'a> {
    candidate: &'a GreedyCandidate,
    action_hash: [u8; 32],
    source_flags: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompatibilityMode {
    Strict,
    Retained,
    Filtered,
    Heuristic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TeacherMode {
    Legacy,
    Filtered,
    Heuristic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StrengthGateProfile {
    Legacy,
    Canonical,
    ProductiveToken,
    FrontierRecall,
}

#[derive(Debug, Clone, Copy)]
struct StrengthComparisonConfig<'a> {
    games: usize,
    first_seed: u64,
    rollouts: usize,
    weights: &'a Path,
    output: &'a Path,
    experiment_id: &'static str,
    teacher_mode: TeacherMode,
    gate_profile: StrengthGateProfile,
}

#[derive(Debug)]
struct ExactMlxPairedGamesConfig<'a> {
    game: GameConfig,
    games: usize,
    first_seed: u64,
    baseline_id: &'a str,
    treatment_id: &'a str,
    baseline_label: String,
    treatment_label: String,
    progress_label: &'static str,
}

#[derive(Debug, Clone, Copy)]
struct ExactMlxRolloutBudgetConfig<'a> {
    server_program: &'a str,
    model_dir: &'a Path,
    games: usize,
    first_seed: u64,
    baseline_rollouts: usize,
    treatment_rollouts: usize,
    weights: &'a Path,
    output: &'a Path,
}

#[derive(Debug, Clone, Copy)]
struct ExactMlxCandidateLimitConfig<'a> {
    server_program: &'a str,
    model_dir: &'a Path,
    games: usize,
    first_seed: u64,
    baseline_candidate_limit: usize,
    treatment_candidate_limit: usize,
    rollouts: usize,
    weights: &'a Path,
    output: &'a Path,
}

#[derive(Debug, Clone, Copy)]
struct ExactMlxHabitatCandidateConfig<'a> {
    server_program: &'a str,
    model_dir: &'a Path,
    games: usize,
    first_seed: u64,
    habitat_candidates: usize,
    rollouts: usize,
    weights: &'a Path,
    output: &'a Path,
}

#[derive(Debug, Clone, Copy)]
struct ExactMlxCrnConfig<'a> {
    server_program: &'a str,
    model_dir: &'a Path,
    games: usize,
    expected_games: Option<usize>,
    first_seed: u64,
    rollouts: usize,
    weights: &'a Path,
    output: &'a Path,
    experiment_id: &'static str,
    progress_label: &'static str,
    thresholds: ExactMlxPairThresholds,
    success_status: &'static str,
    rejection_message: &'static str,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("legacy-teacher error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Command::Compatibility {
            games,
            first_seed,
            weights,
            output,
        } => run_compatibility(
            games,
            first_seed,
            &weights,
            &output,
            CompatibilityMode::Strict,
        ),
        Command::RetainedCompatibility {
            games,
            first_seed,
            weights,
            output,
        } => run_compatibility(
            games,
            first_seed,
            &weights,
            &output,
            CompatibilityMode::Retained,
        ),
        Command::FilteredCompatibility {
            games,
            first_seed,
            weights,
            output,
        } => run_compatibility(
            games,
            first_seed,
            &weights,
            &output,
            CompatibilityMode::Filtered,
        ),
        Command::HeuristicCompatibility {
            games,
            first_seed,
            weights,
            output,
        } => run_compatibility(
            games,
            first_seed,
            &weights,
            &output,
            CompatibilityMode::Heuristic,
        ),
        Command::Compare {
            games,
            first_seed,
            rollouts,
            weights,
            output,
        } => run_comparison(StrengthComparisonConfig {
            games,
            first_seed,
            rollouts,
            weights: &weights,
            output: &output,
            experiment_id: "isolated-legacy-teacher-bridge-v1-r600-20260612",
            teacher_mode: TeacherMode::Legacy,
            gate_profile: StrengthGateProfile::Legacy,
        }),
        Command::RetainedCompare {
            games,
            first_seed,
            rollouts,
            weights,
            output,
        } => run_comparison(StrengthComparisonConfig {
            games,
            first_seed,
            rollouts,
            weights: &weights,
            output: &output,
            experiment_id: "retained-frontier-legacy-teacher-v1-r600-20260612",
            teacher_mode: TeacherMode::Legacy,
            gate_profile: StrengthGateProfile::Legacy,
        }),
        Command::FilteredCompare {
            games,
            first_seed,
            rollouts,
            weights,
            output,
        } => run_comparison(StrengthComparisonConfig {
            games,
            first_seed,
            rollouts,
            weights: &weights,
            output: &output,
            experiment_id: "canonical-filtered-legacy-teacher-v1-r600-20260612",
            teacher_mode: TeacherMode::Filtered,
            gate_profile: StrengthGateProfile::Canonical,
        }),
        Command::HeuristicCompare {
            games,
            first_seed,
            rollouts,
            weights,
            output,
        } => run_comparison(StrengthComparisonConfig {
            games,
            first_seed,
            rollouts,
            weights: &weights,
            output: &output,
            experiment_id: "canonical-action-legacy-heuristic-v1-r600-20260612",
            teacher_mode: TeacherMode::Heuristic,
            gate_profile: StrengthGateProfile::Canonical,
        }),
        Command::ProductiveTokenCompare {
            games,
            first_seed,
            rollouts,
            weights,
            output,
        } => run_comparison(StrengthComparisonConfig {
            games,
            first_seed,
            rollouts,
            weights: &weights,
            output: &output,
            experiment_id: "canonical-action-legacy-productive-token-confirm10-20260612",
            teacher_mode: TeacherMode::Heuristic,
            gate_profile: StrengthGateProfile::ProductiveToken,
        }),
        Command::ExactMlxProductiveTokenCompare {
            server_program,
            model_dir,
            games,
            first_seed,
            split,
            rollouts,
            weights,
            output,
        } => run_exact_mlx_comparison(
            &server_program,
            &model_dir,
            games,
            first_seed,
            split.map(Into::into),
            rollouts,
            &weights,
            &output,
        ),
        Command::ExactMlxRolloutBudgetCompare {
            server_program,
            model_dir,
            games,
            first_seed,
            baseline_rollouts,
            treatment_rollouts,
            weights,
            output,
        } => run_exact_mlx_rollout_budget_comparison(ExactMlxRolloutBudgetConfig {
            server_program: &server_program,
            model_dir: &model_dir,
            games,
            first_seed,
            baseline_rollouts,
            treatment_rollouts,
            weights: &weights,
            output: &output,
        }),
        Command::ExactMlxCrnCompare {
            server_program,
            model_dir,
            games,
            first_seed,
            rollouts,
            weights,
            output,
        } => run_exact_mlx_crn_comparison(ExactMlxCrnConfig {
            server_program: &server_program,
            model_dir: &model_dir,
            games,
            expected_games: None,
            first_seed,
            rollouts,
            weights: &weights,
            output: &output,
            experiment_id: "exact-mlx-sequential-halving-crn-v1-20260613",
            progress_label: "common-random-number pilot",
            thresholds: ExactMlxPairThresholds {
                baseline_runtime_seconds: 220.0,
                treatment_runtime_seconds: 220.0,
                paired_gain: 0.50,
                minimum_confidence_95_lower: None,
                treatment_mean: 96.0,
                wildlife_delta: -0.50,
                habitat_delta: -0.50,
                nature_token_delta: -1.00,
            },
            success_status: "promising",
            rejection_message: "exact MLX CRN comparison failed its frozen gates",
        }),
        Command::ExactMlxCrnConfirm {
            server_program,
            model_dir,
            games,
            first_seed,
            rollouts,
            weights,
            output,
        } => run_exact_mlx_crn_comparison(ExactMlxCrnConfig {
            server_program: &server_program,
            model_dir: &model_dir,
            games,
            expected_games: Some(20),
            first_seed,
            rollouts,
            weights: &weights,
            output: &output,
            experiment_id: "exact-mlx-sequential-halving-crn-confirm20-v1-20260613",
            progress_label: "common-random-number confirmation",
            thresholds: ExactMlxPairThresholds {
                baseline_runtime_seconds: 220.0,
                treatment_runtime_seconds: 220.0,
                paired_gain: 0.50,
                minimum_confidence_95_lower: Some(0.0),
                treatment_mean: 96.0,
                wildlife_delta: -0.50,
                habitat_delta: -0.50,
                nature_token_delta: -1.00,
            },
            success_status: "confirmed",
            rejection_message: "exact MLX CRN confirmation failed its frozen gates",
        }),
        Command::ExactMlxCandidateLimitCompare {
            server_program,
            model_dir,
            games,
            first_seed,
            baseline_candidate_limit,
            treatment_candidate_limit,
            rollouts,
            weights,
            output,
        } => run_exact_mlx_candidate_limit_comparison(ExactMlxCandidateLimitConfig {
            server_program: &server_program,
            model_dir: &model_dir,
            games,
            first_seed,
            baseline_candidate_limit,
            treatment_candidate_limit,
            rollouts,
            weights: &weights,
            output: &output,
        }),
        Command::ExactMlxHabitatCandidateCompare {
            server_program,
            model_dir,
            games,
            first_seed,
            habitat_candidates,
            rollouts,
            weights,
            output,
        } => run_exact_mlx_habitat_candidate_comparison(ExactMlxHabitatCandidateConfig {
            server_program: &server_program,
            model_dir: &model_dir,
            games,
            first_seed,
            habitat_candidates,
            rollouts,
            weights: &weights,
            output: &output,
        }),
        Command::CollectExactMlxRolloutValues {
            server_program,
            model_dir,
            output,
            games,
            first_game_index,
            split,
            resume,
            rollouts,
            trace_modulus,
            weights,
        } => run_collect_exact_mlx_rollout_values(
            &server_program,
            &model_dir,
            &output,
            games,
            first_game_index,
            split.into(),
            resume,
            rollouts,
            trace_modulus,
            &weights,
        ),
        Command::ValidateExactMlxRolloutValues { dataset } => {
            let manifest: RolloutValueDatasetManifest = serde_json::from_reader(
                std::io::BufReader::new(fs::File::open(dataset.join("dataset.json"))?),
            )?;
            validate_rollout_value_dataset(&dataset, &manifest)?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
            Ok(())
        }
        Command::FrontierRecallProbe {
            games,
            first_seed,
            rollouts,
            weights,
            output,
        } => run_comparison(StrengthComparisonConfig {
            games,
            first_seed,
            rollouts,
            weights: &weights,
            output: &output,
            experiment_id: "canonical-action-legacy-pattern-frontier-recall2-20260612",
            teacher_mode: TeacherMode::Heuristic,
            gate_profile: StrengthGateProfile::FrontierRecall,
        }),
        Command::CollectImitation {
            output,
            games,
            first_game_index,
            split,
            shard_games,
            resume,
            group_limit,
            immediate_limit,
            rollouts,
            weights,
        } => collect_imitation_dataset(ImitationCollectionConfig {
            output,
            games,
            first_game_index,
            split: split.into(),
            shard_games,
            resume,
            group_limit,
            immediate_limit,
            rollouts,
            weights,
        }),
        Command::ValidateImitationDataset { dataset } => {
            let manifest: ImitationDatasetManifest =
                serde_json::from_reader(fs::File::open(dataset.join("dataset.json"))?)?;
            validate_imitation_dataset(&dataset, &manifest)?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
            Ok(())
        }
        Command::TeacherEstimateParity {
            games,
            first_game_index,
            split,
            rollouts,
            weights,
            output,
        } => run_teacher_estimate_parity(
            games,
            first_game_index,
            split.into(),
            rollouts,
            &weights,
            &output,
        ),
        Command::EnrichImitationTargets {
            source_dataset,
            output,
            resume,
            weights,
        } => enrich_imitation_targets(&source_dataset, &output, resume, &weights),
        Command::CollectImitationEvidence {
            source_output,
            targets_output,
            games,
            first_game_index,
            split,
            resume,
            group_limit,
            immediate_limit,
            rollouts,
            weights,
        } => collect_imitation_evidence(ImitationEvidenceCollectionConfig {
            source_output,
            targets_output,
            games,
            first_game_index,
            split: split.into(),
            resume,
            group_limit,
            immediate_limit,
            rollouts,
            weights,
        }),
        Command::ValidateImitationTargets { dataset } => {
            let manifest: ImitationTargetsDatasetManifest =
                serde_json::from_reader(fs::File::open(dataset.join("dataset.json"))?)?;
            validate_imitation_targets_dataset(&dataset, &manifest)?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
            Ok(())
        }
        Command::CollectImitationParentPriors {
            server_program,
            model_dir,
            source_dataset,
            output,
            resume,
        } => collect_imitation_parent_priors(
            &server_program,
            &model_dir,
            &source_dataset,
            &output,
            resume,
        ),
        Command::ValidateImitationParentPriors { dataset } => {
            let manifest: ImitationParentPriorDatasetManifest =
                serde_json::from_reader(fs::File::open(dataset.join("dataset.json"))?)?;
            validate_imitation_parent_prior_dataset(&dataset, &manifest)?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
            Ok(())
        }
        Command::CollectImitationParentHidden {
            server_program,
            model_dir,
            source_dataset,
            output,
            resume,
        } => collect_imitation_parent_hidden(
            &server_program,
            &model_dir,
            &source_dataset,
            &output,
            resume,
        ),
        Command::ValidateImitationParentHidden { dataset } => {
            let manifest: ImitationParentHiddenDatasetManifest =
                serde_json::from_reader(fs::File::open(dataset.join("dataset.json"))?)?;
            validate_imitation_parent_hidden_dataset(&dataset, &manifest)?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
            Ok(())
        }
        Command::NnueParityFixture {
            games,
            first_game_index,
            split,
            weights,
            output,
        } => write_nnue_parity_fixture(games, first_game_index, split.into(), &weights, &output),
        Command::NnueServiceParity {
            server_program,
            model_dir,
            fixture,
            output,
            iterations,
        } => run_nnue_service_parity(
            &server_program,
            &model_dir,
            &fixture,
            &output,
            iterations,
            false,
        ),
        Command::NnueExactServiceParity {
            server_program,
            model_dir,
            fixture,
            output,
            iterations,
        } => run_nnue_service_parity(
            &server_program,
            &model_dir,
            &fixture,
            &output,
            iterations,
            true,
        ),
        Command::NnueRolloutWaveParity {
            server_program,
            model_dir,
            fixture,
            weights,
            game_index,
            rollouts,
            spot_decisions,
            spot_rollouts,
            output,
        } => run_nnue_rollout_wave_parity(NnueRolloutWaveConfig {
            experiment_id: "qualified-legacy-nnue-mlx-rollout-wave-v1-parity-20260612",
            exact: false,
            server_program,
            model_dir,
            fixture,
            weights,
            game_index,
            rollouts,
            spot_decisions,
            spot_rollouts,
            max_decisions: None,
            output,
        }),
        Command::NnueExactRolloutWaveParity {
            server_program,
            model_dir,
            fixture,
            weights,
            game_index,
            rollouts,
            spot_decisions,
            spot_rollouts,
            max_decisions,
            output,
        } => run_nnue_rollout_wave_parity(NnueRolloutWaveConfig {
            experiment_id: "qualified-legacy-nnue-mlx-exact-rollout-wave-v1-parity-20260612",
            exact: true,
            server_program,
            model_dir,
            fixture,
            weights,
            game_index,
            rollouts,
            spot_decisions,
            spot_rollouts,
            max_decisions,
            output,
        }),
    }
}

struct MlxSparseEvaluator {
    process: ModelProcess,
    exact: bool,
}

impl SparseNnueEvaluator for MlxSparseEvaluator {
    type Error = ModelError;

    fn evaluate_sparse(&mut self, feature_sets: &[Vec<u16>]) -> Result<Vec<f32>, Self::Error> {
        if self.exact {
            self.process.predict_sparse_nnue_csr_exact(feature_sets)
        } else {
            self.process.predict_sparse_nnue(feature_sets)
        }
    }
}

impl MlxSparseEvaluator {
    fn shutdown(self) -> Result<(), ModelError> {
        self.process.shutdown()
    }
}

struct NnueRolloutWaveConfig {
    experiment_id: &'static str,
    exact: bool,
    server_program: String,
    model_dir: PathBuf,
    fixture: PathBuf,
    weights: PathBuf,
    game_index: u64,
    rollouts: usize,
    spot_decisions: Vec<usize>,
    spot_rollouts: usize,
    max_decisions: Option<usize>,
    output: PathBuf,
}

fn move_identity(movement: &ScoredMove) -> MoveIdentity {
    MoveIdentity {
        market_index: movement.market_index,
        wildlife_market_index: movement.wildlife_market_index,
        tile_q: movement.tile_q,
        tile_r: movement.tile_r,
        rotation: movement.rotation,
        wildlife_q: movement.wildlife_q,
        wildlife_r: movement.wildlife_r,
    }
}

fn compare_estimates(
    reference_candidates: &[ScoredMove],
    treatment_candidates: &[ScoredMove],
    reference: &[MceMoveEstimate],
    treatment: &[MceMoveEstimate],
    metrics: &mut RolloutParityMetrics,
) {
    metrics.decisions += 1;
    if reference_candidates
        .iter()
        .map(move_identity)
        .ne(treatment_candidates.iter().map(move_identity))
    {
        metrics.candidate_identity_mismatches += 1;
    }
    if reference
        .first()
        .map(|value| move_identity(&value.movement))
        != treatment
            .first()
            .map(|value| move_identity(&value.movement))
    {
        metrics.selected_action_mismatches += 1;
    }
    let reference = reference
        .iter()
        .map(|estimate| (move_identity(&estimate.movement), estimate))
        .collect::<BTreeMap<_, _>>();
    let treatment = treatment
        .iter()
        .map(|estimate| (move_identity(&estimate.movement), estimate))
        .collect::<BTreeMap<_, _>>();
    if reference.len() != treatment.len() || reference.keys().ne(treatment.keys()) {
        metrics.sample_count_mismatches += 1;
        return;
    }
    let mut error_sum = metrics.mean_rollout_mean_absolute_error * metrics.estimate_count as f64;
    for (identity, reference) in reference {
        let treatment = treatment[&identity];
        if reference.samples != treatment.samples {
            metrics.sample_count_mismatches += 1;
        }
        let error = (reference.rollout_mean - treatment.rollout_mean).abs();
        metrics.maximum_rollout_mean_absolute_error =
            metrics.maximum_rollout_mean_absolute_error.max(error);
        error_sum += error;
        metrics.estimate_count += 1;
    }
    metrics.mean_rollout_mean_absolute_error = error_sum / metrics.estimate_count as f64;
}

fn merge_diagnostics(target: &mut SearchBatchDiagnostics, source: BatchedNnueDiagnostics) {
    target.neural_batches += source.neural_batches;
    target.neural_rows += source.neural_rows;
    if source.minimum_batch_rows > 0 {
        target.minimum_batch_rows = if target.minimum_batch_rows == 0 {
            source.minimum_batch_rows
        } else {
            target.minimum_batch_rows.min(source.minimum_batch_rows)
        };
    }
    target.maximum_batch_rows = target.maximum_batch_rows.max(source.maximum_batch_rows);
    target.rollout_waves += source.rollout_waves;
    target.rollout_samples += source.rollout_samples;
    target.policy_fallbacks += source.policy_fallbacks;
}

fn canonical_root_candidates(
    game: &GameState,
    prelude: &cascadia_game::MarketPrelude,
    translated: &cascadia_core::game::GameState,
) -> Vec<ScoredMove> {
    cascadia_ai::mce::expanded_candidates(translated)
        .into_iter()
        .filter(|candidate| map_legacy_action(game, prelude, candidate).is_ok())
        .collect()
}

fn run_nnue_rollout_wave_parity(
    config: NnueRolloutWaveConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    if config.rollouts == 0 || config.spot_rollouts == 0 {
        return Err("rollout-wave parity budgets must be positive".into());
    }
    validate_legacy_environment()?;
    let reference_net = load_legacy_weights(&config.weights)?;
    let fixture: NnueParityFixtureInput =
        serde_json::from_reader(fs::File::open(&config.fixture)?)?;
    let warmup = fixture
        .records
        .first()
        .ok_or("rollout-wave parity fixture is empty")?
        .features
        .clone();
    let mut mlx = MlxSparseEvaluator {
        process: ModelProcess::spawn(
            &config.server_program,
            [
                std::ffi::OsString::from("run"),
                std::ffi::OsString::from("cascadia-mlx-legacy-nnue-serve"),
                std::ffi::OsString::from("--model-dir"),
                config.model_dir.as_os_str().to_owned(),
            ],
        )?,
        exact: config.exact,
    };
    mlx.evaluate_sparse(&[warmup])?;

    let game_seed = DatasetSplit::Train.game_seed(config.game_index);
    let mut game = GameState::new(GameConfig::research_aaaaa(4)?, game_seed)?;
    let mut fallback_rngs = (0..4)
        .map(|seat| strategy_rng(game_seed, seat, PATTERN_AWARE_STRATEGY_ID))
        .collect::<Vec<_>>();
    let mut new_native_metrics = RolloutParityMetrics::default();
    let mut mlx_metrics = RolloutParityMetrics::default();
    let mut mlx_repeat_metrics = RolloutParityMetrics::default();
    let mut mlx_spot_metrics = RolloutParityMetrics::default();
    let mut new_native_diagnostics = SearchBatchDiagnostics::default();
    let mut mlx_diagnostics = SearchBatchDiagnostics::default();
    let mut mlx_repeat_diagnostics = SearchBatchDiagnostics::default();
    let mut mlx_spot_diagnostics = SearchBatchDiagnostics::default();
    let mut native_r32_seconds = 0.0;
    let mut mlx_r32_seconds = 0.0;
    let mut decision_index = 0usize;

    while !game.is_game_over()
        && config
            .max_decisions
            .is_none_or(|limit| decision_index < limit)
    {
        let prelude = canonical_prelude(&game);
        let staged = game.preview_market_prelude(&prelude)?;
        let translation =
            translate_public_state_allowing_legacy_elk_undercount(&staged.public_state())?;
        let expanded = canonical_root_candidates(&game, &prelude, &translation.game);

        let native_started = Instant::now();
        let native_candidates = if expanded.len() > 32 {
            nnue_prefilter_candidates(&translation.game, &reference_net, expanded.clone(), 32)
        } else {
            expanded.clone()
        };
        let mut native_rng = legacy_search_rng(&translation.evidence.public_state_blake3);
        let native_estimates = score_nnue_rollout_mce_seq_halving(
            &translation.game,
            &reference_net,
            config.rollouts,
            native_candidates.clone(),
            &mut native_rng,
        );
        let native_elapsed = native_started.elapsed().as_secs_f64();
        native_r32_seconds += native_elapsed;

        let mut batch_native = reference_net.clone();
        let mut batch_native_diag = BatchedNnueDiagnostics::default();
        let batch_native_started = Instant::now();
        let batch_native_candidates = if expanded.len() > 32 {
            nnue_prefilter_candidates_batched(
                &translation.game,
                &mut batch_native,
                expanded.clone(),
                32,
                &mut batch_native_diag,
            )?
        } else {
            expanded.clone()
        };
        let mut batch_native_rng = legacy_search_rng(&translation.evidence.public_state_blake3);
        let batch_native_estimates = score_nnue_rollout_mce_seq_halving_batched(
            &translation.game,
            &mut batch_native,
            config.rollouts,
            batch_native_candidates.clone(),
            &mut batch_native_rng,
            &mut batch_native_diag,
        )?;
        new_native_metrics.total_seconds += batch_native_started.elapsed().as_secs_f64();
        compare_estimates(
            &native_candidates,
            &batch_native_candidates,
            &native_estimates,
            &batch_native_estimates,
            &mut new_native_metrics,
        );
        merge_diagnostics(&mut new_native_diagnostics, batch_native_diag);

        let mut mlx_diag = BatchedNnueDiagnostics::default();
        let mlx_started = Instant::now();
        let mlx_candidates = if expanded.len() > 32 {
            nnue_prefilter_candidates_batched(
                &translation.game,
                &mut mlx,
                expanded.clone(),
                32,
                &mut mlx_diag,
            )?
        } else {
            expanded.clone()
        };
        let mut mlx_rng = legacy_search_rng(&translation.evidence.public_state_blake3);
        let mlx_estimates = score_nnue_rollout_mce_seq_halving_batched(
            &translation.game,
            &mut mlx,
            config.rollouts,
            mlx_candidates.clone(),
            &mut mlx_rng,
            &mut mlx_diag,
        )?;
        let mlx_elapsed = mlx_started.elapsed().as_secs_f64();
        mlx_r32_seconds += mlx_elapsed;
        mlx_metrics.total_seconds += mlx_elapsed;
        compare_estimates(
            &native_candidates,
            &mlx_candidates,
            &native_estimates,
            &mlx_estimates,
            &mut mlx_metrics,
        );
        merge_diagnostics(&mut mlx_diagnostics, mlx_diag);

        let mut repeat_diag = BatchedNnueDiagnostics::default();
        let repeat_candidates = if expanded.len() > 32 {
            nnue_prefilter_candidates_batched(
                &translation.game,
                &mut mlx,
                expanded.clone(),
                32,
                &mut repeat_diag,
            )?
        } else {
            expanded.clone()
        };
        let mut repeat_rng = legacy_search_rng(&translation.evidence.public_state_blake3);
        let repeat_estimates = score_nnue_rollout_mce_seq_halving_batched(
            &translation.game,
            &mut mlx,
            config.rollouts,
            repeat_candidates.clone(),
            &mut repeat_rng,
            &mut repeat_diag,
        )?;
        compare_estimates(
            &mlx_candidates,
            &repeat_candidates,
            &mlx_estimates,
            &repeat_estimates,
            &mut mlx_repeat_metrics,
        );
        merge_diagnostics(&mut mlx_repeat_diagnostics, repeat_diag);

        if config.spot_decisions.contains(&decision_index) {
            let mut native_spot_rng = legacy_search_rng(&translation.evidence.public_state_blake3);
            let native_spot = score_nnue_rollout_mce_seq_halving(
                &translation.game,
                &reference_net,
                config.spot_rollouts,
                native_candidates.clone(),
                &mut native_spot_rng,
            );
            let mut spot_diag = BatchedNnueDiagnostics::default();
            let mut mlx_spot_rng = legacy_search_rng(&translation.evidence.public_state_blake3);
            let mlx_spot = score_nnue_rollout_mce_seq_halving_batched(
                &translation.game,
                &mut mlx,
                config.spot_rollouts,
                mlx_candidates.clone(),
                &mut mlx_spot_rng,
                &mut spot_diag,
            )?;
            compare_estimates(
                &native_candidates,
                &mlx_candidates,
                &native_spot,
                &mlx_spot,
                &mut mlx_spot_metrics,
            );
            merge_diagnostics(&mut mlx_spot_diagnostics, spot_diag);
        }

        let canonical_seat = game.current_player();
        let action = pattern_fallback(&game, &mut fallback_rngs[canonical_seat])?;
        game = game.transition(&action)?;
        eprintln!(
            "rollout-wave decision {} complete: native {:.3}s, mlx {:.3}s, batches {}, rows {}",
            decision_index,
            native_elapsed,
            mlx_elapsed,
            mlx_diag.neural_batches,
            mlx_diag.neural_rows
        );
        decision_index += 1;
    }
    let clean_shutdown = mlx.shutdown().is_ok();
    let mlx_native_runtime_ratio = mlx_r32_seconds / native_r32_seconds;
    let new_native_exact = new_native_metrics.candidate_identity_mismatches == 0
        && new_native_metrics.selected_action_mismatches == 0
        && new_native_metrics.sample_count_mismatches == 0
        && new_native_metrics.maximum_rollout_mean_absolute_error == 0.0;
    let gates = NnueRolloutWaveGates {
        new_native_exact,
        mlx_r32_candidates: mlx_metrics.candidate_identity_mismatches == 0,
        mlx_r32_selected_actions: mlx_metrics.selected_action_mismatches == 0,
        mlx_r32_samples: mlx_metrics.sample_count_mismatches == 0,
        mlx_r32_maximum_error: mlx_metrics.maximum_rollout_mean_absolute_error
            <= if config.exact { 0.0 } else { 0.05 },
        mlx_r32_mean_error: mlx_metrics.mean_rollout_mean_absolute_error
            <= if config.exact { 0.0 } else { 0.01 },
        mlx_repeat_deterministic: mlx_repeat_metrics.candidate_identity_mismatches == 0
            && mlx_repeat_metrics.selected_action_mismatches == 0
            && mlx_repeat_metrics.sample_count_mismatches == 0
            && mlx_repeat_metrics.maximum_rollout_mean_absolute_error == 0.0,
        mlx_r600_candidates: mlx_spot_metrics.candidate_identity_mismatches == 0,
        mlx_r600_selected_actions: mlx_spot_metrics.selected_action_mismatches == 0,
        mlx_r600_samples: mlx_spot_metrics.sample_count_mismatches == 0,
        mlx_r600_maximum_error: mlx_spot_metrics.maximum_rollout_mean_absolute_error
            <= if config.exact { 0.0 } else { 0.05 },
        mlx_r600_mean_error: mlx_spot_metrics.mean_rollout_mean_absolute_error
            <= if config.exact { 0.0 } else { 0.01 },
        mlx_zero_fallbacks: mlx_diagnostics.policy_fallbacks == 0
            && mlx_repeat_diagnostics.policy_fallbacks == 0
            && mlx_spot_diagnostics.policy_fallbacks == 0,
        mlx_runtime_ratio: mlx_native_runtime_ratio <= if config.exact { 1.5 } else { 2.0 },
        canonical_trajectory: decision_index == 80,
        clean_shutdown,
    };
    let passed = gates.new_native_exact
        && gates.mlx_r32_candidates
        && gates.mlx_r32_selected_actions
        && gates.mlx_r32_samples
        && gates.mlx_r32_maximum_error
        && gates.mlx_r32_mean_error
        && gates.mlx_repeat_deterministic
        && gates.mlx_r600_candidates
        && gates.mlx_r600_selected_actions
        && gates.mlx_r600_samples
        && gates.mlx_r600_maximum_error
        && gates.mlx_r600_mean_error
        && gates.mlx_zero_fallbacks
        && gates.mlx_runtime_ratio
        && gates.canonical_trajectory
        && gates.clean_shutdown;
    let executable_path = std::env::current_exe()?;
    let report = NnueRolloutWaveReport {
        schema_version: 1,
        experiment_id: config.experiment_id,
        operation: if config.exact {
            "packed-csr-rust-order-metal"
        } else {
            "variable-row-standard-mlx"
        },
        game_index: config.game_index,
        rollouts: config.rollouts,
        spot_decisions: config.spot_decisions,
        spot_rollouts: config.spot_rollouts,
        trajectory_decisions: decision_index,
        new_native: new_native_metrics,
        mlx: mlx_metrics,
        mlx_repeat: mlx_repeat_metrics,
        mlx_spot: mlx_spot_metrics,
        native_r32_seconds,
        mlx_r32_seconds,
        mlx_native_runtime_ratio,
        new_native_diagnostics,
        mlx_diagnostics,
        mlx_repeat_diagnostics,
        mlx_spot_diagnostics,
        gates,
        passed,
        model_manifest_blake3: checksum_file(&config.model_dir.join("model.json"))?,
        fixture_blake3: checksum_file(&config.fixture)?,
        weights_blake3: checksum_file(&config.weights)?,
        source: source_provenance()?,
        executable_blake3: checksum_file(&executable_path)?,
        executable_path,
    };
    write_json_atomic(&config.output, &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !passed && config.max_decisions.is_none() {
        return Err("NNUE rollout-wave integration failed a frozen gate".into());
    }
    Ok(())
}

fn run_nnue_service_parity(
    server_program: &str,
    model_dir: &Path,
    fixture_path: &Path,
    output: &Path,
    iterations: usize,
    exact: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if iterations == 0 {
        return Err("NNUE service benchmark iterations must be positive".into());
    }
    let fixture: NnueParityFixtureInput = serde_json::from_reader(fs::File::open(fixture_path)?)?;
    if fixture.records.len() != 80 {
        return Err(format!(
            "NNUE service fixture has {} records, expected 80",
            fixture.records.len()
        )
        .into());
    }
    let feature_sets = fixture
        .records
        .iter()
        .map(|record| record.features.clone())
        .collect::<Vec<_>>();
    let expected = fixture
        .records
        .iter()
        .map(|record| record.rust_value)
        .collect::<Vec<_>>();
    let model_manifest = model_dir.join("model.json");
    let model_safetensors = model_dir.join("model.safetensors");
    let model_arg = model_dir.as_os_str().to_owned();
    let started = Instant::now();
    let mut service = ModelProcess::spawn(
        server_program,
        [
            std::ffi::OsString::from("run"),
            std::ffi::OsString::from("cascadia-mlx-legacy-nnue-serve"),
            std::ffi::OsString::from("--model-dir"),
            model_arg,
        ],
    )?;
    let predict = |service: &mut ModelProcess, batch: &[Vec<u16>]| {
        if exact {
            service.predict_sparse_nnue_csr_exact(batch)
        } else {
            service.predict_sparse_nnue(batch)
        }
    };
    let first = predict(&mut service, &feature_sets[..1])?;
    let startup_milliseconds = started.elapsed().as_secs_f64() * 1000.0;
    if first.len() != 1 {
        return Err("NNUE service startup probe returned the wrong width".into());
    }

    let actual = predict(&mut service, &feature_sets)?;
    let repeated = predict(&mut service, &feature_sets)?;
    let deterministic_repeat = actual
        .iter()
        .zip(&repeated)
        .all(|(left, right)| left.to_bits() == right.to_bits());
    let errors = actual
        .iter()
        .zip(&expected)
        .map(|(value, reference)| f64::from((value - reference).abs()))
        .collect::<Vec<_>>();
    let maximum_absolute_error_vs_rust = errors.iter().copied().fold(0.0, f64::max);
    let mean_absolute_error_vs_rust = errors.iter().sum::<f64>() / errors.len() as f64;
    let finite = actual.iter().all(|value| value.is_finite());

    let mut benchmarks = Vec::new();
    for batch_size in [1usize, 32, 256] {
        let batch = (0..batch_size)
            .map(|index| feature_sets[index % feature_sets.len()].clone())
            .collect::<Vec<_>>();
        for _ in 0..5 {
            predict(&mut service, &batch)?;
        }
        let mut durations = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let iteration_started = Instant::now();
            predict(&mut service, &batch)?;
            durations.push(iteration_started.elapsed().as_secs_f64());
        }
        durations.sort_by(f64::total_cmp);
        let p50 = percentile(&durations, 0.50);
        let p90 = percentile(&durations, 0.90);
        let p99 = percentile(&durations, 0.99);
        benchmarks.push(NnueServiceBenchmark {
            batch_size,
            iterations,
            p50_milliseconds: p50 * 1000.0,
            p90_milliseconds: p90 * 1000.0,
            p99_milliseconds: p99 * 1000.0,
            evaluations_per_second: batch_size as f64 / p50,
        });
    }
    let clean_shutdown = service.shutdown().is_ok();
    let batch32 = benchmarks
        .iter()
        .find(|benchmark| benchmark.batch_size == 32)
        .ok_or("missing batch-32 benchmark")?;
    let gates = NnueServiceGates {
        fixture_records: actual.len() == 80,
        finite,
        maximum_rust_error: maximum_absolute_error_vs_rust <= if exact { 0.0 } else { 1e-3 },
        deterministic_repeat,
        clean_shutdown,
        batch32_throughput: batch32.evaluations_per_second
            >= if exact { 10_000.0 } else { 2_000.0 },
        batch32_p99_latency: batch32.p99_milliseconds <= if exact { 10.0 } else { 25.0 },
    };
    let passed = gates.fixture_records
        && gates.finite
        && gates.maximum_rust_error
        && gates.deterministic_repeat
        && gates.clean_shutdown
        && gates.batch32_throughput
        && gates.batch32_p99_latency;
    let executable_path = std::env::current_exe()?;
    let report = NnueServiceReport {
        schema_version: 1,
        experiment_id: if exact {
            "qualified-legacy-nnue-mlx-exact-csr-service-v1-parity-20260612"
        } else {
            "qualified-legacy-nnue-mlx-service-v1-parity-20260612"
        },
        operation: if exact {
            "packed-csr-rust-order-metal"
        } else {
            "variable-row-standard-mlx"
        },
        device_service: "Device(gpu, 0)",
        fixture_records: actual.len(),
        maximum_absolute_error_vs_rust,
        mean_absolute_error_vs_rust,
        deterministic_repeat,
        startup_milliseconds,
        benchmarks,
        gates,
        passed,
        model_manifest_path: model_manifest.canonicalize()?,
        model_manifest_blake3: checksum_file(&model_manifest)?,
        model_safetensors_blake3: checksum_file(&model_safetensors)?,
        fixture_path: fixture_path.canonicalize()?,
        fixture_blake3: checksum_file(fixture_path)?,
        source: source_provenance()?,
        executable_blake3: checksum_file(&executable_path)?,
        executable_path,
    };
    write_json_atomic(output, &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !passed {
        return Err("NNUE service parity failed a frozen gate".into());
    }
    Ok(())
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * quantile).ceil() as usize;
    sorted[index]
}

fn write_nnue_parity_fixture(
    games: usize,
    first_game_index: u64,
    split: DatasetSplit,
    weights: &Path,
    output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if games == 0 {
        return Err("NNUE parity fixture requires at least one game".into());
    }
    validate_legacy_environment()?;
    let net = load_legacy_weights(weights)?;
    let mut records = Vec::with_capacity(games * 80);
    let mut records_with_duplicate_features = 0;
    let mut duplicate_feature_occurrences = 0;
    let mut maximum_feature_multiplicity = 1;
    for game_offset in 0..games {
        let game_index = first_game_index + game_offset as u64;
        let game_seed = split.game_seed(game_index);
        let mut game = GameState::new(GameConfig::research_aaaaa(4)?, game_seed)?;
        let mut fallback_rngs = (0..4)
            .map(|seat| strategy_rng(game_seed, seat, PATTERN_AWARE_STRATEGY_ID))
            .collect::<Vec<_>>();
        let mut decision_index = 0;
        while !game.is_game_over() {
            let prelude = canonical_prelude(&game);
            let staged = game.preview_market_prelude(&prelude)?;
            let translation =
                translate_public_state_allowing_legacy_elk_undercount(&staged.public_state())?;
            let active_seat = translation.game.current_player;
            let bag = BagInfo::from_game_for_player(&translation.game, active_seat);
            let features =
                extract_features_with_bag(&translation.game.boards[active_seat], Some(&bag));
            if features.iter().any(|&index| index as usize >= NUM_FEATURES) {
                return Err("NNUE parity fixture produced an out-of-range feature".into());
            }
            let mut multiplicities = BTreeMap::new();
            for &feature in &features {
                *multiplicities.entry(feature).or_insert(0usize) += 1;
            }
            let unique = multiplicities.len();
            if unique != features.len() {
                records_with_duplicate_features += 1;
                duplicate_feature_occurrences += features.len() - unique;
            }
            if let Some(record_maximum) = multiplicities.values().copied().max() {
                maximum_feature_multiplicity = maximum_feature_multiplicity.max(record_maximum);
            }
            records.push(NnueParityRecord {
                game_index,
                decision_index,
                active_seat,
                rust_value: net.forward(&features),
                features,
            });
            let canonical_seat = game.current_player();
            let action = pattern_fallback(&game, &mut fallback_rngs[canonical_seat])?;
            game = game.transition(&action)?;
            decision_index += 1;
        }
        if decision_index != 80 {
            return Err(format!(
                "NNUE parity fixture game {game_index} had {decision_index} decisions"
            )
            .into());
        }
    }
    let fixture = NnueParityFixture {
        schema_version: 1,
        feature_schema: "legacy-mid-v4opp-sparse-u16-v1",
        split,
        first_game_index,
        games,
        feature_count: NUM_FEATURES,
        hidden1: HIDDEN1,
        hidden2: HIDDEN2,
        records_with_duplicate_features,
        duplicate_feature_occurrences,
        maximum_feature_multiplicity,
        records,
        provenance: provenance(weights)?,
    };
    write_json_atomic(output, &fixture)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": fixture.schema_version,
            "output": output,
            "games": fixture.games,
            "records": fixture.records.len(),
            "records_with_duplicate_features": fixture.records_with_duplicate_features,
            "duplicate_feature_occurrences": fixture.duplicate_feature_occurrences,
            "maximum_feature_multiplicity": fixture.maximum_feature_multiplicity,
        }))?
    );
    Ok(())
}

struct ImitationCollectionConfig {
    output: PathBuf,
    games: usize,
    first_game_index: u64,
    split: DatasetSplit,
    shard_games: usize,
    resume: bool,
    group_limit: usize,
    immediate_limit: usize,
    rollouts: usize,
    weights: PathBuf,
}

struct ImitationEvidenceCollectionConfig {
    source_output: PathBuf,
    targets_output: PathBuf,
    games: usize,
    first_game_index: u64,
    split: DatasetSplit,
    resume: bool,
    group_limit: usize,
    immediate_limit: usize,
    rollouts: usize,
    weights: PathBuf,
}

#[allow(clippy::too_many_arguments)]
fn run_collect_exact_mlx_rollout_values(
    server_program: &str,
    model_dir: &Path,
    output: &Path,
    games: usize,
    first_game_index: u64,
    split: DatasetSplit,
    resume: bool,
    rollouts: usize,
    trace_modulus: u64,
    weights: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if games == 0 || rollouts == 0 || trace_modulus == 0 {
        return Err("rollout-value collection counts must be positive".into());
    }
    validate_legacy_environment()?;
    let model_manifest_path = model_dir.join("model.json");
    let model_manifest: serde_json::Value = serde_json::from_reader(std::io::BufReader::new(
        fs::File::open(&model_manifest_path)?,
    ))?;
    let model_manifest_blake3 = checksum_file(&model_manifest_path)?;
    let weights_blake3 = checksum_file(weights)?;
    if model_manifest
        .pointer("/source/blake3")
        .and_then(serde_json::Value::as_str)
        != Some(weights_blake3.as_str())
        || model_manifest
            .pointer("/dimensions/features")
            .and_then(serde_json::Value::as_u64)
            != Some(NUM_FEATURES as u64)
    {
        return Err("rollout-value parent model does not match weights or feature count".into());
    }
    let strategy_id = format!("exact-mlx-rollout-value-k32-r{rollouts}-trace{trace_modulus}-v1");
    let config = RolloutValueDatasetConfig {
        output: output.to_path_buf(),
        split,
        first_game_index,
        games,
        teacher: RolloutValueTeacherConfig {
            strategy_id: strategy_id.clone(),
            parent_model_manifest_blake3: model_manifest_blake3,
            weights_blake3,
            feature_count: NUM_FEATURES,
            candidate_limit: 32,
            rollouts,
            trace_modulus,
            lmr: true,
            diverse_prefilter: true,
        },
        resume,
    };
    let mut writer = RolloutValueDatasetWriter::open(&config)?;
    let completed = writer.manifest().completed_games;
    if completed == games {
        validate_rollout_value_dataset(output, writer.manifest())?;
        println!("{}", serde_json::to_string_pretty(writer.manifest())?);
        return Ok(());
    }

    let (mut teacher, startup_milliseconds) =
        spawn_exact_mlx_teacher(server_program, model_dir, rollouts, 32)?;
    eprintln!(
        "rollout-value collector: exact MLX service ready in {:.1} ms",
        startup_milliseconds
    );
    let started = Instant::now();
    for offset in completed..games {
        let game_index = first_game_index + offset as u64;
        let seed = split.game_seed(game_index);
        let mut records = Vec::new();
        let mut decision_index = 0usize;
        play_match_with_selector(
            GameConfig::research_aaaaa(4)?,
            seed,
            &strategy_id,
            |_player, game| {
                let decision = teacher
                    .select_action_collecting_rollout_values(game, trace_modulus)
                    .map_err(simulation_error)?;
                let personal_turn = (decision_index / 4 + 1) as u8;
                for sample in decision.rollout_value_samples {
                    records.push(RolloutValueRecord {
                        kind: RolloutValueRecordKind::Trajectory,
                        game_index,
                        decision_index: decision_index as u8,
                        personal_turn: sample.personal_turn,
                        selected: true,
                        rollout_seed: sample.rollout_seed,
                        immediate_score: sample.immediate_score,
                        target_remaining: sample.target_remaining,
                        target_stddev: 0.0,
                        samples: 1,
                        features: sample.features,
                    });
                }
                let selected_roots = decision
                    .root_estimates
                    .iter()
                    .filter(|estimate| estimate.selected)
                    .count();
                if selected_roots != 1 {
                    return Err(cascadia_sim::SimulationError::Strategy(format!(
                        "rollout-value decision {decision_index} has {selected_roots} selected roots"
                    )));
                }
                for estimate in decision.root_estimates {
                    records.push(RolloutValueRecord {
                        kind: RolloutValueRecordKind::RootEstimate,
                        game_index,
                        decision_index: decision_index as u8,
                        personal_turn,
                        selected: estimate.selected,
                        rollout_seed: 0,
                        immediate_score: estimate.immediate_score,
                        target_remaining: estimate.rollout_mean as f32 - estimate.immediate_score,
                        target_stddev: estimate.rollout_stddev as f32,
                        samples: estimate.samples,
                        features: estimate.features,
                    });
                }
                decision_index += 1;
                if decision_index.is_multiple_of(10) || decision_index == 80 {
                    eprintln!(
                        "rollout-value game {game_index}: {decision_index}/80 decisions, \
                         {} records, {:.1}s elapsed",
                        records.len(),
                        started.elapsed().as_secs_f64()
                    );
                }
                Ok(decision.action)
            },
        )?;
        if decision_index != 80 {
            return Err(format!(
                "rollout-value game {game_index} completed {decision_index} decisions"
            )
            .into());
        }
        writer.append_game(game_index, &records)?;
        eprintln!(
            "rollout-value collector: {}/{} games, {} total records, {:.1}s elapsed",
            offset + 1,
            games,
            writer.manifest().total_records,
            started.elapsed().as_secs_f64()
        );
    }
    teacher.shutdown()?;
    validate_rollout_value_dataset(output, writer.manifest())?;
    println!("{}", serde_json::to_string_pretty(writer.manifest())?);
    Ok(())
}

fn collect_imitation_dataset(
    config: ImitationCollectionConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    if config.games == 0 || config.shard_games == 0 {
        return Err("imitation collection requires positive game and shard counts".into());
    }
    validate_legacy_environment()?;
    let pattern = PatternAwareConfig::default();
    let teacher_config = ImitationTeacherConfig::from_weights(
        HEURISTIC_LEGACY_TEACHER_STRATEGY_ID,
        config.rollouts,
        32,
        &config.weights,
    )?;
    let candidate_config = ImitationCandidateConfig {
        group_limit: config.group_limit,
        immediate_limit: config.immediate_limit,
        pattern_immediate_limit: pattern.immediate_candidate_limit,
        pattern_habitat_limit: pattern.habitat_candidate_limit,
        pattern_bear_limit: pattern.bear_candidate_limit,
        pattern_market_draws: pattern.future_market_draws,
        deterministic_sampler: "blake3-action-json-v1".to_owned(),
    };
    let mut writer = ImitationDatasetWriter::open(&ImitationDatasetConfig {
        output: config.output,
        split: config.split,
        first_game_index: config.first_game_index,
        games: config.games,
        teacher: teacher_config,
        candidates: candidate_config.clone(),
        resume: config.resume,
    })?;
    let completed = writer.manifest().completed_games;
    if completed > config.games {
        return Err("imitation dataset already exceeds the requested game count".into());
    }
    let net = load_legacy_weights(&config.weights)?;
    let mut teacher = LegacyTeacher::new_heuristic(net, config.rollouts)?;
    let started = Instant::now();
    for shard_start in (completed..config.games).step_by(config.shard_games) {
        let game_count = config.shard_games.min(config.games - shard_start);
        let first_game_index = config.first_game_index + shard_start as u64;
        let mut records = Vec::new();
        for offset in 0..game_count {
            records.extend(collect_imitation_game(
                &mut teacher,
                config.split,
                first_game_index + offset as u64,
                &candidate_config,
            )?);
        }
        writer.append_shard(first_game_index, game_count, &records)?;
        eprintln!(
            "imitation dataset: {}/{} games, {} groups, {} candidates, {:.1}s elapsed",
            writer.manifest().completed_games,
            writer.manifest().requested_games,
            writer.manifest().total_groups,
            writer.manifest().total_records,
            started.elapsed().as_secs_f64(),
        );
    }
    validate_imitation_dataset(writer.root(), writer.manifest())?;
    println!("{}", serde_json::to_string_pretty(writer.manifest())?);
    Ok(())
}

fn collect_imitation_evidence(
    config: ImitationEvidenceCollectionConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    if config.games == 0 || config.rollouts == 0 {
        return Err("imitation evidence collection requires positive games and rollouts".into());
    }
    if config.source_output == config.targets_output {
        return Err("imitation evidence source and target outputs must be distinct".into());
    }
    validate_legacy_environment()?;
    let pattern = PatternAwareConfig::default();
    let teacher_config = ImitationTeacherConfig::from_weights(
        DETERMINISTIC_EVIDENCE_TEACHER_STRATEGY_ID,
        config.rollouts,
        32,
        &config.weights,
    )?;
    let candidate_config = ImitationCandidateConfig {
        group_limit: config.group_limit,
        immediate_limit: config.immediate_limit,
        pattern_immediate_limit: pattern.immediate_candidate_limit,
        pattern_habitat_limit: pattern.habitat_candidate_limit,
        pattern_bear_limit: pattern.bear_candidate_limit,
        pattern_market_draws: pattern.future_market_draws,
        deterministic_sampler: "teacher-frontier-pattern-immediate-blake3-action-json-v1"
            .to_owned(),
    };
    let mut source_writer = ImitationDatasetWriter::open(&ImitationDatasetConfig {
        output: config.source_output,
        split: config.split,
        first_game_index: config.first_game_index,
        games: config.games,
        teacher: teacher_config,
        candidates: candidate_config.clone(),
        resume: config.resume,
    })?;
    let mut target_writer = ImitationTargetsDatasetWriter::open(&ImitationTargetsDatasetConfig {
        output: config.targets_output,
        source_root: source_writer.root().to_path_buf(),
        source_manifest: source_writer.manifest().clone(),
        resume: config.resume,
    })?;
    let source_completed = source_writer.manifest().completed_games;
    let target_completed = target_writer.manifest().completed_games;
    if source_completed > config.games
        || target_completed > source_completed
        || source_writer.manifest().shards.len() != source_completed
        || target_writer.manifest().shards.len() != target_completed
    {
        return Err("imitation evidence resume state is not on one-game shard boundaries".into());
    }

    let net = load_legacy_weights(&config.weights)?;
    let mut teacher = LegacyTeacher::new_heuristic(net, config.rollouts)?;
    let started = Instant::now();
    for game_offset in target_completed..config.games {
        let game_index = config.first_game_index + game_offset as u64;
        let (source_records, target_records, teacher_estimates) = collect_imitation_evidence_game(
            &mut teacher,
            config.split,
            game_index,
            &candidate_config,
            true,
        )?;
        if game_offset < source_completed {
            let source_shard = &source_writer.manifest().shards[game_offset];
            let existing = read_imitation_shard_records(
                source_writer.root(),
                source_writer.manifest().split,
                source_shard,
            )?;
            if existing != source_records {
                return Err(
                    format!("replayed evidence source differs at game index {game_index}").into(),
                );
            }
        } else {
            source_writer.append_shard(game_index, 1, &source_records)?;
        }
        validate_target_alignment(&source_records, &target_records)?;
        target_writer.append_shard(game_index, 1, &target_records, teacher_estimates)?;
        eprintln!(
            "imitation evidence: {}/{} games, {} groups, {} candidates, {}/{} teacher estimates aligned, {:.1}s elapsed",
            target_writer.manifest().completed_games,
            target_writer.manifest().requested_games,
            target_writer.manifest().total_groups,
            target_writer.manifest().total_records,
            target_writer.manifest().aligned_teacher_estimates,
            target_writer.manifest().teacher_estimates,
            started.elapsed().as_secs_f64(),
        );
    }
    validate_imitation_dataset(source_writer.root(), source_writer.manifest())?;
    validate_imitation_targets_dataset(target_writer.root(), target_writer.manifest())?;
    println!(
        "{}",
        serde_json::to_string_pretty(target_writer.manifest())?
    );
    Ok(())
}

fn collect_imitation_game(
    teacher: &mut LegacyTeacher,
    split: DatasetSplit,
    game_index: u64,
    config: &ImitationCandidateConfig,
) -> Result<Vec<ImitationRecord>, Box<dyn std::error::Error>> {
    let mut game = GameState::new(GameConfig::research_aaaaa(4)?, split.game_seed(game_index))?;
    let pattern = PatternAwareConfig {
        immediate_candidate_limit: config.pattern_immediate_limit,
        habitat_candidate_limit: config.pattern_habitat_limit,
        bear_candidate_limit: config.pattern_bear_limit,
        future_market_draws: config.pattern_market_draws,
    };
    let mut records = Vec::with_capacity(80 * config.group_limit);
    while !game.is_game_over() {
        let turn = game.completed_turns();
        let active_seat = game.current_player();
        let selected = teacher.select_action(&game)?;
        let prelude = canonical_prelude(&game);
        let all = rank_greedy_actions(&game, &prelude, None)?;
        let pattern_candidates = rank_pattern_actions(&game, &prelude, pattern)?;
        let group_id = imitation_group_id(split, game_index, turn, active_seat);
        let selected_hash = action_hash(&selected)?;
        let pattern_hashes = pattern_candidates
            .iter()
            .map(|candidate| action_hash(&candidate.action))
            .collect::<Result<BTreeSet<_>, _>>()?;
        let candidates = sample_imitation_candidates(
            &all,
            selected_hash,
            &pattern_hashes,
            &BTreeSet::new(),
            group_id,
            config,
            false,
        )?;
        let candidate_count = u16::try_from(candidates.len())?;
        for (candidate_index, sampled) in candidates.into_iter().enumerate() {
            let candidate = sampled.candidate;
            let hash = sampled.action_hash;
            let immediate_rank = u16::try_from(candidate.immediate_rank)?;
            records.push(ImitationRecord {
                group_id,
                candidate_index: u16::try_from(candidate_index)?,
                candidate_count,
                immediate_rank,
                immediate_score: candidate.resulting_base_score,
                teacher_mean: if hash == selected_hash { 1.0 } else { 0.0 },
                teacher_stddev: 0.0,
                action_hash: hash,
                input: ProposalPositionRecord::observe(
                    &game,
                    &candidate.action,
                    game_index,
                    immediate_rank,
                    candidate.resulting_base_score,
                )?,
            });
        }
        game.apply(&selected)?;
    }
    Ok(records)
}

fn sample_imitation_candidates<'a>(
    all: &'a [GreedyCandidate],
    selected_hash: [u8; 32],
    pattern_hashes: &BTreeSet<[u8; 32]>,
    teacher_hashes: &BTreeSet<[u8; 32]>,
    group_id: u64,
    config: &ImitationCandidateConfig,
    retain_teacher_frontier: bool,
) -> Result<Vec<SampledCandidate<'a>>, Box<dyn std::error::Error>> {
    let all_with_hashes = all
        .iter()
        .map(|candidate| Ok((candidate, action_hash(&candidate.action)?)))
        .collect::<Result<Vec<_>, serde_json::Error>>()?;
    let immediate_order = all_with_hashes
        .iter()
        .take(config.immediate_limit)
        .map(|(_, hash)| *hash)
        .collect::<Vec<_>>();
    let immediate_hashes = immediate_order.iter().copied().collect::<BTreeSet<_>>();
    let mut retained = BTreeSet::new();
    retained.insert(selected_hash);
    if retain_teacher_frontier {
        retained.extend(teacher_hashes.iter().copied());
    }
    retained.extend(pattern_hashes.iter().copied());
    if retained.len() > config.group_limit {
        return Err("required teacher and pattern frontiers exceed imitation group limit".into());
    }
    for hash in immediate_order {
        if retained.len() == config.group_limit {
            break;
        }
        retained.insert(hash);
    }
    let mut sampled = all_with_hashes
        .iter()
        .filter_map(|(_, hash)| {
            (!retained.contains(hash)).then_some((sample_hash(group_id, hash), *hash))
        })
        .collect::<Vec<_>>();
    sampled.sort_unstable();
    let mut deterministic_hashes = BTreeSet::new();
    for (_, hash) in sampled {
        if retained.len() == config.group_limit {
            break;
        }
        retained.insert(hash);
        deterministic_hashes.insert(hash);
    }

    let candidates = all_with_hashes
        .iter()
        .filter(|(_, hash)| retained.contains(hash))
        .map(|(candidate, hash)| {
            let mut source_flags = 0;
            if teacher_hashes.contains(hash) {
                source_flags |= SOURCE_TEACHER_FRONTIER;
            }
            if pattern_hashes.contains(hash) {
                source_flags |= SOURCE_PATTERN_FRONTIER;
            }
            if immediate_hashes.contains(hash) {
                source_flags |= SOURCE_IMMEDIATE_TOP;
            }
            if deterministic_hashes.contains(hash) {
                source_flags |= SOURCE_DETERMINISTIC_NEGATIVE;
            }
            SampledCandidate {
                candidate,
                action_hash: *hash,
                source_flags,
            }
        })
        .collect::<Vec<_>>();
    let expected = config.group_limit.min(all.len());
    if candidates.len() != expected
        || !candidates
            .iter()
            .any(|candidate| candidate.action_hash == selected_hash)
    {
        return Err("imitation sampler failed to retain the selected canonical action".into());
    }
    Ok(candidates)
}

fn imitation_group_id(split: DatasetSplit, game_index: u64, turn: u16, active_seat: usize) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"cascadia-v2-canonical-action-imitation-group");
    hasher.update(split.id().as_bytes());
    hasher.update(&game_index.to_le_bytes());
    hasher.update(&turn.to_le_bytes());
    hasher.update(&(active_seat as u64).to_le_bytes());
    u64::from_le_bytes(
        hasher.finalize().as_bytes()[..8]
            .try_into()
            .expect("BLAKE3 output contains eight bytes"),
    )
}

fn action_hash(action: &TurnAction) -> Result<[u8; 32], serde_json::Error> {
    Ok(*blake3::hash(&serde_json::to_vec(action)?).as_bytes())
}

fn sample_hash(group_id: u64, action_hash: &[u8; 32]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"cascadia-v2-canonical-action-imitation-negative");
    hasher.update(&group_id.to_le_bytes());
    hasher.update(action_hash);
    *hasher.finalize().as_bytes()
}

fn run_teacher_estimate_parity(
    games: usize,
    first_game_index: u64,
    split: DatasetSplit,
    rollouts: usize,
    weights: &Path,
    output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if games == 0 || rollouts == 0 {
        return Err("teacher estimate parity requires positive games and rollouts".into());
    }
    validate_legacy_environment()?;
    let net = load_legacy_weights(weights)?;
    let mut original = LegacyTeacher::new_heuristic(net.clone(), rollouts)?;
    let mut instrumented = LegacyTeacher::new_heuristic(net, rollouts)?;
    let started = Instant::now();
    let mut states = 0;
    let mut estimates = 0;
    let mut minimum_samples = u32::MAX;
    let mut maximum_samples = 0;

    for offset in 0..games {
        let game_index = first_game_index + offset as u64;
        let mut game = GameState::new(GameConfig::research_aaaaa(4)?, split.game_seed(game_index))?;
        while !game.is_game_over() {
            let selected = original.select_action(&game)?;
            let decision = instrumented.select_action_with_estimates(&game)?;
            if selected != decision.selected
                || decision.estimates.first().map(|estimate| &estimate.action)
                    != Some(&decision.selected)
                || decision.estimates.iter().any(|estimate| {
                    !estimate.rollout_mean.is_finite()
                        || !estimate.rollout_stddev.is_finite()
                        || estimate.rollout_stddev < 0.0
                        || estimate.samples == 0
                })
                || decision
                    .estimates
                    .windows(2)
                    .any(|pair| pair[0].rollout_mean < pair[1].rollout_mean)
            {
                return Err(format!(
                    "teacher estimate parity failed at game {game_index}, turn {}",
                    game.completed_turns()
                )
                .into());
            }
            let hashes = decision
                .estimates
                .iter()
                .map(|estimate| action_hash(&estimate.action))
                .collect::<Result<BTreeSet<_>, _>>()?;
            if hashes.len() != decision.estimates.len() {
                return Err("teacher estimate path produced duplicate canonical actions".into());
            }
            states += 1;
            estimates += decision.estimates.len();
            for estimate in &decision.estimates {
                minimum_samples = minimum_samples.min(estimate.samples);
                maximum_samples = maximum_samples.max(estimate.samples);
            }
            game.apply(&decision.selected)?;
        }
    }

    let report = TeacherEstimateParityReport {
        schema_version: 1,
        games,
        first_game_index,
        split,
        rollouts,
        states,
        estimates,
        minimum_samples,
        maximum_samples,
        passed: true,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        provenance: provenance(weights)?,
    };
    write_json_atomic(output, &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn enrich_imitation_targets(
    source_root: &Path,
    output: &Path,
    resume: bool,
    weights: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    validate_legacy_environment()?;
    let source_manifest: ImitationDatasetManifest =
        serde_json::from_reader(fs::File::open(source_root.join("dataset.json"))?)?;
    validate_imitation_dataset(source_root, &source_manifest)?;
    if checksum_file(weights)? != source_manifest.teacher.weights_blake3
        || fs::metadata(weights)?.len() != source_manifest.teacher.weights_bytes
    {
        return Err("enrichment weights do not match the immutable source teacher".into());
    }
    let mut writer = ImitationTargetsDatasetWriter::open(&ImitationTargetsDatasetConfig {
        output: output.to_path_buf(),
        source_root: source_root.to_path_buf(),
        source_manifest: source_manifest.clone(),
        resume,
    })?;
    let completed = writer.manifest().completed_games;
    let prefix_games = source_manifest
        .shards
        .iter()
        .take(writer.manifest().shards.len())
        .map(|shard| shard.game_count)
        .sum::<usize>();
    if completed != prefix_games {
        return Err("imitation-target resume point is not a source-shard boundary".into());
    }

    let net = load_legacy_weights(weights)?;
    let mut teacher = LegacyTeacher::new_heuristic(net, source_manifest.teacher.rollouts)?;
    let started = Instant::now();
    for source_shard in source_manifest
        .shards
        .iter()
        .skip(writer.manifest().shards.len())
    {
        let mut target_records = Vec::with_capacity(source_shard.record_count);
        let mut teacher_estimates = 0;
        for offset in 0..source_shard.game_count {
            let game_index = source_shard.first_game_index + offset as u64;
            let (mut game_records, game_estimates) = collect_imitation_target_game(
                &mut teacher,
                source_manifest.split,
                game_index,
                &source_manifest.candidates,
            )?;
            target_records.append(&mut game_records);
            teacher_estimates += game_estimates;
        }
        let source_records =
            read_imitation_shard_records(source_root, source_manifest.split, source_shard)?;
        validate_target_alignment(&source_records, &target_records)?;
        writer.append_shard(
            source_shard.first_game_index,
            source_shard.game_count,
            &target_records,
            teacher_estimates,
        )?;
        eprintln!(
            "imitation targets: {}/{} games, {}/{} teacher estimates aligned ({:.2}%), {:.1}s elapsed",
            writer.manifest().completed_games,
            writer.manifest().requested_games,
            writer.manifest().aligned_teacher_estimates,
            writer.manifest().teacher_estimates,
            100.0 * writer.manifest().aligned_teacher_estimates as f64
                / writer.manifest().teacher_estimates.max(1) as f64,
            started.elapsed().as_secs_f64(),
        );
    }
    validate_imitation_targets_dataset(writer.root(), writer.manifest())?;
    println!("{}", serde_json::to_string_pretty(writer.manifest())?);
    Ok(())
}

fn collect_imitation_target_game(
    teacher: &mut LegacyTeacher,
    split: DatasetSplit,
    game_index: u64,
    config: &ImitationCandidateConfig,
) -> Result<(Vec<ImitationTargetRecord>, usize), Box<dyn std::error::Error>> {
    let (_, targets, estimate_count) =
        collect_imitation_evidence_game(teacher, split, game_index, config, false)?;
    Ok((targets, estimate_count))
}

fn collect_imitation_evidence_game(
    teacher: &mut LegacyTeacher,
    split: DatasetSplit,
    game_index: u64,
    config: &ImitationCandidateConfig,
    retain_teacher_frontier: bool,
) -> Result<ImitationEvidence, Box<dyn std::error::Error>> {
    let mut game = GameState::new(GameConfig::research_aaaaa(4)?, split.game_seed(game_index))?;
    let pattern = PatternAwareConfig {
        immediate_candidate_limit: config.pattern_immediate_limit,
        habitat_candidate_limit: config.pattern_habitat_limit,
        bear_candidate_limit: config.pattern_bear_limit,
        future_market_draws: config.pattern_market_draws,
    };
    let mut source_records = Vec::with_capacity(80 * config.group_limit);
    let mut target_records = Vec::with_capacity(80 * config.group_limit);
    let mut teacher_estimate_count = 0;
    while !game.is_game_over() {
        let turn = game.completed_turns();
        let active_seat = game.current_player();
        let decision = teacher.select_action_with_estimates(&game)?;
        let selected_hash = action_hash(&decision.selected)?;
        let prelude = canonical_prelude(&game);
        let all = rank_greedy_actions(&game, &prelude, None)?;
        let pattern_candidates = rank_pattern_actions(&game, &prelude, pattern)?;
        let group_id = imitation_group_id(split, game_index, turn, active_seat);
        let pattern_hashes = pattern_candidates
            .iter()
            .map(|candidate| action_hash(&candidate.action))
            .collect::<Result<BTreeSet<_>, _>>()?;
        let mut estimates = BTreeMap::new();
        for estimate in decision.estimates {
            let hash = action_hash(&estimate.action)?;
            if estimates.insert(hash, estimate).is_some() {
                return Err("teacher returned duplicate canonical action estimates".into());
            }
        }
        teacher_estimate_count += estimates.len();
        let teacher_hashes = estimates.keys().copied().collect::<BTreeSet<_>>();
        let candidates = sample_imitation_candidates(
            &all,
            selected_hash,
            &pattern_hashes,
            &teacher_hashes,
            group_id,
            config,
            retain_teacher_frontier,
        )?;
        if retain_teacher_frontier
            && candidates
                .iter()
                .filter(|candidate| teacher_hashes.contains(&candidate.action_hash))
                .count()
                != teacher_hashes.len()
        {
            return Err("full-frontier evidence sampler dropped a teacher candidate".into());
        }
        let candidate_count = u16::try_from(candidates.len())?;
        for (candidate_index, sampled) in candidates.into_iter().enumerate() {
            let estimate = estimates.get(&sampled.action_hash);
            let immediate_rank = u16::try_from(sampled.candidate.immediate_rank)?;
            source_records.push(ImitationRecord {
                group_id,
                candidate_index: u16::try_from(candidate_index)?,
                candidate_count,
                immediate_rank,
                immediate_score: sampled.candidate.resulting_base_score,
                teacher_mean: if sampled.action_hash == selected_hash {
                    1.0
                } else {
                    0.0
                },
                teacher_stddev: 0.0,
                action_hash: sampled.action_hash,
                input: ProposalPositionRecord::observe(
                    &game,
                    &sampled.candidate.action,
                    game_index,
                    immediate_rank,
                    sampled.candidate.resulting_base_score,
                )?,
            });
            target_records.push(ImitationTargetRecord {
                group_id,
                candidate_index: u16::try_from(candidate_index)?,
                candidate_count,
                action_hash: sampled.action_hash,
                teacher_mean: estimate.map_or(0.0, |value| value.rollout_mean as f32),
                teacher_stddev: estimate.map_or(0.0, |value| value.rollout_stddev as f32),
                teacher_samples: estimate
                    .map(|value| u16::try_from(value.samples))
                    .transpose()?
                    .unwrap_or(0),
                source_flags: sampled.source_flags,
                selected: sampled.action_hash == selected_hash,
            });
        }
        game.apply(&decision.selected)?;
    }
    Ok((source_records, target_records, teacher_estimate_count))
}

fn validate_target_alignment(
    source: &[ImitationRecord],
    targets: &[ImitationTargetRecord],
) -> Result<(), Box<dyn std::error::Error>> {
    if source.len() != targets.len() {
        return Err(format!(
            "imitation-target record count {} does not match source {}",
            targets.len(),
            source.len()
        )
        .into());
    }
    for (index, (source, target)) in source.iter().zip(targets).enumerate() {
        if source.group_id != target.group_id
            || source.candidate_index != target.candidate_index
            || source.candidate_count != target.candidate_count
            || source.action_hash != target.action_hash
            || (source.teacher_mean == 1.0) != target.selected
        {
            return Err(format!(
                "imitation-target row {index} mismatch: source group={} candidate={}/{} hash={} selected={}; target group={} candidate={}/{} hash={} selected={}",
                source.group_id,
                source.candidate_index,
                source.candidate_count,
                blake3::Hash::from_bytes(source.action_hash),
                source.teacher_mean == 1.0,
                target.group_id,
                target.candidate_index,
                target.candidate_count,
                blake3::Hash::from_bytes(target.action_hash),
                target.selected,
            )
            .into());
        }
    }
    Ok(())
}

fn collect_imitation_parent_priors(
    server_program: &str,
    model_dir: &Path,
    source_dataset: &Path,
    output: &Path,
    resume: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    validate_legacy_environment()?;
    let source_manifest: ImitationTargetsDatasetManifest = serde_json::from_reader(
        std::io::BufReader::new(fs::File::open(source_dataset.join("dataset.json"))?),
    )?;
    validate_imitation_targets_dataset(source_dataset, &source_manifest)?;
    let action_root = PathBuf::from(&source_manifest.source.path);
    let action_manifest: ImitationDatasetManifest = serde_json::from_reader(
        std::io::BufReader::new(fs::File::open(action_root.join("dataset.json"))?),
    )?;
    validate_imitation_dataset(&action_root, &action_manifest)?;
    let mut writer = ImitationParentPriorDatasetWriter::open(&ImitationParentPriorDatasetConfig {
        output: output.to_owned(),
        source_root: source_dataset.to_owned(),
        source_manifest: source_manifest.clone(),
        model_dir: model_dir.to_owned(),
        resume,
    })?;
    if writer.manifest().completed_games == writer.manifest().requested_games {
        validate_imitation_parent_prior_dataset(writer.root(), writer.manifest())?;
        println!("{}", serde_json::to_string_pretty(writer.manifest())?);
        return Ok(());
    }

    let (mut teacher, startup_milliseconds) =
        spawn_exact_mlx_teacher(server_program, model_dir, 1, 32)?;
    let started = Instant::now();
    let start_shard = writer.manifest().shards.len();
    for shard_index in start_shard..source_manifest.shards.len() {
        let source_shard = &source_manifest.shards[shard_index];
        let action_shard = action_manifest
            .shards
            .get(shard_index)
            .ok_or("parent-prior source action dataset has fewer shards than target evidence")?;
        if source_shard.first_game_index != action_shard.first_game_index
            || source_shard.game_count != action_shard.game_count
            || source_shard.group_count != action_shard.group_count
            || source_shard.record_count != action_shard.record_count
        {
            return Err("parent-prior source action and target shards do not align".into());
        }
        let target_records = read_imitation_target_shard_records(
            source_dataset,
            source_manifest.split,
            source_shard,
        )?;
        let action_records =
            read_imitation_shard_records(&action_root, action_manifest.split, action_shard)?;
        let records = collect_imitation_parent_prior_shard(
            &mut teacher,
            source_manifest.split,
            &action_records,
            &target_records,
            source_shard.game_count,
        )?;
        writer.append_shard(
            source_shard.first_game_index,
            source_shard.game_count,
            &records,
        )?;
        eprintln!(
            "parent priors: {}/{} games, {} groups, {} actions, {:.1}s elapsed",
            writer.manifest().completed_games,
            writer.manifest().requested_games,
            writer.manifest().total_groups,
            writer.manifest().total_records,
            started.elapsed().as_secs_f64(),
        );
    }
    teacher.shutdown()?;
    validate_imitation_parent_prior_dataset(writer.root(), writer.manifest())?;
    eprintln!(
        "exact MLX parent-prior service startup: {:.1} ms",
        startup_milliseconds
    );
    println!("{}", serde_json::to_string_pretty(writer.manifest())?);
    Ok(())
}

fn collect_imitation_parent_hidden(
    server_program: &str,
    model_dir: &Path,
    source_dataset: &Path,
    output: &Path,
    resume: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    validate_legacy_environment()?;
    let source_manifest: ImitationTargetsDatasetManifest = serde_json::from_reader(
        std::io::BufReader::new(fs::File::open(source_dataset.join("dataset.json"))?),
    )?;
    validate_imitation_targets_dataset(source_dataset, &source_manifest)?;
    let action_root = PathBuf::from(&source_manifest.source.path);
    let action_manifest: ImitationDatasetManifest = serde_json::from_reader(
        std::io::BufReader::new(fs::File::open(action_root.join("dataset.json"))?),
    )?;
    validate_imitation_dataset(&action_root, &action_manifest)?;
    let mut writer =
        ImitationParentHiddenDatasetWriter::open(&ImitationParentHiddenDatasetConfig {
            output: output.to_owned(),
            source_root: source_dataset.to_owned(),
            source_manifest: source_manifest.clone(),
            model_dir: model_dir.to_owned(),
            resume,
        })?;
    if writer.manifest().completed_games == writer.manifest().requested_games {
        validate_imitation_parent_hidden_dataset(writer.root(), writer.manifest())?;
        println!("{}", serde_json::to_string_pretty(writer.manifest())?);
        return Ok(());
    }

    let (mut teacher, startup_milliseconds) =
        spawn_exact_mlx_teacher(server_program, model_dir, 1, 32)?;
    let started = Instant::now();
    let start_shard = writer.manifest().shards.len();
    for shard_index in start_shard..source_manifest.shards.len() {
        let source_shard = &source_manifest.shards[shard_index];
        let action_shard = action_manifest
            .shards
            .get(shard_index)
            .ok_or("parent-hidden source action dataset has fewer shards than target evidence")?;
        if source_shard.first_game_index != action_shard.first_game_index
            || source_shard.game_count != action_shard.game_count
            || source_shard.group_count != action_shard.group_count
            || source_shard.record_count != action_shard.record_count
        {
            return Err("parent-hidden source action and target shards do not align".into());
        }
        let target_records = read_imitation_target_shard_records(
            source_dataset,
            source_manifest.split,
            source_shard,
        )?;
        let action_records =
            read_imitation_shard_records(&action_root, action_manifest.split, action_shard)?;
        let records = collect_imitation_parent_hidden_shard(
            &mut teacher,
            source_manifest.split,
            &action_records,
            &target_records,
            source_shard.game_count,
        )?;
        writer.append_shard(
            source_shard.first_game_index,
            source_shard.game_count,
            &records,
        )?;
        eprintln!(
            "parent hidden: {}/{} games, {} groups, {} actions, {:.1}s elapsed",
            writer.manifest().completed_games,
            writer.manifest().requested_games,
            writer.manifest().total_groups,
            writer.manifest().total_records,
            started.elapsed().as_secs_f64(),
        );
    }
    teacher.shutdown()?;
    validate_imitation_parent_hidden_dataset(writer.root(), writer.manifest())?;
    eprintln!(
        "exact MLX parent-hidden service startup: {:.1} ms",
        startup_milliseconds
    );
    println!("{}", serde_json::to_string_pretty(writer.manifest())?);
    Ok(())
}

fn collect_imitation_parent_prior_shard(
    teacher: &mut ExactMlxLegacyTeacher,
    split: DatasetSplit,
    actions: &[ImitationRecord],
    targets: &[ImitationTargetRecord],
    expected_games: usize,
) -> Result<Vec<ImitationParentPriorRecord>, Box<dyn std::error::Error>> {
    replay_imitation_parent_shard(
        teacher,
        split,
        actions,
        targets,
        expected_games,
        |teacher, state, reconstructed, action_group| {
            let priors = teacher.score_action_priors(state, reconstructed)?;
            if priors.len() != action_group.len() {
                return Err("parent-prior evaluator returned the wrong candidate count".into());
            }
            Ok(action_group
                .iter()
                .zip(priors)
                .map(|(record, prior)| ImitationParentPriorRecord {
                    group_id: record.group_id,
                    candidate_index: record.candidate_index,
                    candidate_count: record.candidate_count,
                    action_hash: record.action_hash,
                    parent_immediate: prior.immediate_score,
                    parent_remaining: prior.remaining_value,
                })
                .collect())
        },
    )
}

fn collect_imitation_parent_hidden_shard(
    teacher: &mut ExactMlxLegacyTeacher,
    split: DatasetSplit,
    actions: &[ImitationRecord],
    targets: &[ImitationTargetRecord],
    expected_games: usize,
) -> Result<Vec<ImitationParentHiddenRecord>, Box<dyn std::error::Error>> {
    replay_imitation_parent_shard(
        teacher,
        split,
        actions,
        targets,
        expected_games,
        |teacher, state, reconstructed, action_group| {
            let predictions = teacher.score_action_hidden(state, reconstructed)?;
            if predictions.len() != action_group.len() {
                return Err("parent-hidden evaluator returned the wrong candidate count".into());
            }
            Ok(action_group
                .iter()
                .zip(predictions)
                .map(|(record, prediction)| ImitationParentHiddenRecord {
                    group_id: record.group_id,
                    candidate_index: record.candidate_index,
                    candidate_count: record.candidate_count,
                    action_hash: record.action_hash,
                    parent_immediate: prediction.immediate_score,
                    parent_remaining: prediction.remaining_value,
                    parent_hidden: prediction.hidden,
                })
                .collect())
        },
    )
}

fn replay_imitation_parent_shard<Record>(
    teacher: &mut ExactMlxLegacyTeacher,
    split: DatasetSplit,
    actions: &[ImitationRecord],
    targets: &[ImitationTargetRecord],
    expected_games: usize,
    mut evaluate: impl FnMut(
        &mut ExactMlxLegacyTeacher,
        &GameState,
        &[TurnAction],
        &[ImitationRecord],
    ) -> Result<Vec<Record>, Box<dyn std::error::Error>>,
) -> Result<Vec<Record>, Box<dyn std::error::Error>> {
    if actions.len() != targets.len() || actions.is_empty() {
        return Err("parent-sidecar source action and target record counts differ".into());
    }
    let mut output = Vec::with_capacity(actions.len());
    let mut offset = 0usize;
    let mut game: Option<(u64, GameState)> = None;
    let mut completed_games = 0usize;
    while offset < actions.len() {
        let candidate_count = usize::from(actions[offset].candidate_count);
        let end = offset + candidate_count;
        if candidate_count < 2 || end > actions.len() || end > targets.len() {
            return Err("parent-sidecar source contains a truncated candidate group".into());
        }
        let action_group = &actions[offset..end];
        let target_group = &targets[offset..end];
        let group_id = action_group[0].group_id;
        if action_group.iter().enumerate().any(|(index, record)| {
            record.group_id != group_id
                || usize::from(record.candidate_index) != index
                || usize::from(record.candidate_count) != candidate_count
        }) || target_group
            .iter()
            .zip(action_group)
            .any(|(target, action)| {
                target.group_id != action.group_id
                    || target.candidate_index != action.candidate_index
                    || target.candidate_count != action.candidate_count
                    || target.action_hash != action.action_hash
            })
        {
            return Err("parent-sidecar source candidate group is not aligned".into());
        }
        let game_index = action_group[0].input.position.game_index;
        if game
            .as_ref()
            .is_none_or(|(current, _)| *current != game_index)
        {
            if let Some((_, previous)) = game.take() {
                if !previous.is_game_over() {
                    return Err(
                        "parent-sidecar replay changed game before its terminal state".into(),
                    );
                }
                completed_games += 1;
            }
            game = Some((
                game_index,
                GameState::new(GameConfig::research_aaaaa(4)?, split.game_seed(game_index))?,
            ));
        }
        let state = &mut game.as_mut().expect("game was initialized").1;
        let expected_position = PositionRecord::observe(state, game_index);
        if action_group.iter().any(|record| {
            record.input.position.to_bytes() != expected_position.to_bytes()
                || record.input.position.turn != state.completed_turns() as u8
                || record.input.position.active_seat != state.current_player() as u8
        }) {
            return Err("parent-sidecar replay state does not match the recorded position".into());
        }
        let reconstructed = action_group
            .iter()
            .map(|record| {
                let action = record.input.action.to_game_action(state)?;
                let hash = *blake3::hash(&serde_json::to_vec(&action)?).as_bytes();
                if hash != record.action_hash {
                    return Err::<TurnAction, Box<dyn std::error::Error>>(
                        "parent-sidecar reconstructed action hash does not match source".into(),
                    );
                }
                Ok(action)
            })
            .collect::<Result<Vec<_>, _>>()?;
        output.extend(evaluate(teacher, state, &reconstructed, action_group)?);
        let selected = target_group
            .iter()
            .position(|record| record.selected)
            .ok_or("parent-sidecar source group has no selected action")?;
        if target_group.iter().filter(|record| record.selected).count() != 1 {
            return Err("parent-sidecar source group has multiple selected actions".into());
        }
        state.apply(&reconstructed[selected])?;
        offset = end;
    }
    if let Some((_, final_game)) = game {
        if !final_game.is_game_over() {
            return Err("parent-sidecar replay ended before the source game".into());
        }
        completed_games += 1;
    }
    if completed_games != expected_games {
        return Err("parent-sidecar replay game count does not match the source shard".into());
    }
    Ok(output)
}

fn run_compatibility(
    games: usize,
    first_seed: u64,
    weights: &Path,
    output: &Path,
    mode: CompatibilityMode,
) -> Result<(), Box<dyn std::error::Error>> {
    if games == 0 {
        return Err("compatibility audit requires at least one game".into());
    }
    validate_legacy_environment()?;
    let net = load_legacy_weights(weights)?;
    let provenance = provenance(weights)?;
    let started = Instant::now();
    let mut diagnostics = BridgeDiagnostics::default();
    let mut failure = None;
    for offset in 0..games {
        let seed = first_seed + offset as u64;
        let audit = match mode {
            CompatibilityMode::Strict => {
                audit_pattern_trajectory(GameSeed::from_u64(seed), &net, &mut diagnostics)
            }
            CompatibilityMode::Retained => {
                audit_retained_pattern_trajectory(GameSeed::from_u64(seed), &net, &mut diagnostics)
            }
            CompatibilityMode::Filtered => {
                audit_filtered_pattern_trajectory(GameSeed::from_u64(seed), &net, &mut diagnostics)
            }
            CompatibilityMode::Heuristic => {
                audit_heuristic_pattern_trajectory(GameSeed::from_u64(seed), &net, &mut diagnostics)
            }
        };
        if let Err(error) = audit {
            diagnostics.record_external_error(&error);
            failure = Some(format!("seed {seed}: {error}"));
            break;
        }
    }
    let expected_states = games * 80;
    let expanded_malformed_rate = if diagnostics.expanded_candidates == 0 {
        0.0
    } else {
        diagnostics.expanded_candidates_illegal as f64 / diagnostics.expanded_candidates as f64
    };
    let passed = failure.is_none()
        && diagnostics.states_attempted == expected_states
        && diagnostics.states_translated == expected_states
        && diagnostics.checked_boards == expected_states * 4
        && diagnostics.expanded_candidates > 0
        && match mode {
            CompatibilityMode::Strict => {
                diagnostics.expanded_candidates == diagnostics.expanded_candidates_legal
            }
            CompatibilityMode::Retained
            | CompatibilityMode::Filtered
            | CompatibilityMode::Heuristic => expanded_malformed_rate <= 0.10,
        }
        && diagnostics.prefiltered_candidates > 0
        && diagnostics.prefiltered_candidates == diagnostics.prefiltered_candidates_legal
        && diagnostics.first_errors.is_empty();
    let report = CompatibilityReport {
        schema_version: 1,
        experiment_id: match mode {
            CompatibilityMode::Strict => "isolated-legacy-teacher-bridge-v1-r600-20260612",
            CompatibilityMode::Retained => "retained-frontier-legacy-teacher-v1-r600-20260612",
            CompatibilityMode::Filtered => "canonical-filtered-legacy-teacher-v1-r600-20260612",
            CompatibilityMode::Heuristic => "canonical-action-legacy-heuristic-v1-r600-20260612",
        },
        status: if passed { "complete" } else { "rejected" },
        games,
        first_seed,
        last_seed: first_seed + games.saturating_sub(1) as u64,
        expected_states,
        diagnostics,
        expanded_malformed_rate,
        passed,
        failure,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        provenance,
    };
    write_json_atomic(output, &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !passed {
        return Err("compatibility audit failed its frozen gates".into());
    }
    Ok(())
}

fn run_comparison(run: StrengthComparisonConfig<'_>) -> Result<(), Box<dyn std::error::Error>> {
    let StrengthComparisonConfig {
        games,
        first_seed,
        rollouts,
        weights,
        output,
        experiment_id,
        teacher_mode,
        gate_profile,
    } = run;
    if games == 0 {
        return Err("strength comparison requires at least one game".into());
    }
    validate_legacy_environment()?;
    let net = load_legacy_weights(weights)?;
    let provenance = provenance(weights)?;
    let game = GameConfig::research_aaaaa(4)?;
    let strong = LateConservativeBasePolicyImprovementStrategy::new(
        LateConservativeBasePolicyImprovementConfig::default(),
    )?;
    let mut teacher = match teacher_mode {
        TeacherMode::Legacy => LegacyTeacher::new(net, rollouts)?,
        TeacherMode::Filtered => LegacyTeacher::new_filtered(net, rollouts)?,
        TeacherMode::Heuristic if gate_profile == StrengthGateProfile::FrontierRecall => {
            LegacyTeacher::new_heuristic_with_pattern_frontier_probe(net, rollouts)?
        }
        TeacherMode::Heuristic => LegacyTeacher::new_heuristic(net, rollouts)?,
    };
    let treatment_strategy_id = match teacher_mode {
        TeacherMode::Legacy => LEGACY_TEACHER_STRATEGY_ID,
        TeacherMode::Filtered => FILTERED_LEGACY_TEACHER_STRATEGY_ID,
        TeacherMode::Heuristic => HEURISTIC_LEGACY_TEACHER_STRATEGY_ID,
    };
    let started = Instant::now();
    let mut results: Vec<(u64, MatchResult, MatchResult)> = Vec::with_capacity(games);
    for offset in 0..games {
        let seed_value = first_seed + offset as u64;
        let seed = GameSeed::from_u64(seed_value);
        let baseline = strong.play_match(game, seed)?;
        let mut fallback_rngs = (0..4)
            .map(|seat| strategy_rng(seed, seat, PATTERN_AWARE_STRATEGY_ID))
            .collect::<Vec<_>>();
        let treatment = play_match_with_selector(
            game,
            seed,
            treatment_strategy_id,
            |player, game| match teacher.select_action(game) {
                Ok(action) => Ok(action),
                Err(error) if error.permits_coordinate_fallback() => {
                    teacher.diagnostics.fallbacks += 1;
                    pattern_fallback(game, &mut fallback_rngs[player])
                        .map_err(simulation_error)
                        .map_err(|fallback_error| {
                            cascadia_sim::SimulationError::Strategy(format!(
                                "legacy bridge failed ({error}); fallback failed ({fallback_error})"
                            ))
                        })
                }
                Err(error) => Err(simulation_error(error)),
            },
        )?;
        results.push((seed_value, baseline, treatment));
    }
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let comparison = summarize_paired_match_results(
        strong.strategy_id(),
        treatment_strategy_id,
        first_seed,
        &results,
        elapsed_seconds,
    );
    let fallback_rate = teacher.diagnostics.fallback_rate();
    let expanded_malformed_rate = if teacher.diagnostics.expanded_candidates == 0 {
        0.0
    } else {
        teacher.diagnostics.expanded_candidates_illegal as f64
            / teacher.diagnostics.expanded_candidates as f64
    };
    let total_wildlife_delta = comparison.mean_breakdown_delta.wildlife.iter().sum::<f64>();
    let habitat_delta = comparison.mean_breakdown_delta.habitat.iter().sum::<f64>();
    let nature_token_delta = comparison.mean_breakdown_delta.nature_tokens;
    let token_spend = (-nature_token_delta).max(0.0);
    let non_token_score_delta = comparison.mean_paired_delta - nature_token_delta;
    let board_points_per_token = if token_spend > 0.0 {
        Some(non_token_score_delta / token_spend)
    } else {
        None
    };
    let pattern_frontier_recall = teacher.diagnostics.pattern_frontier_recall();
    let independent_pattern_frontier_recall =
        teacher.diagnostics.independent_pattern_frontier_recall();
    let pattern_frontier_recall_by_phase = teacher.diagnostics.pattern_frontier_recall_by_phase();
    let bridge_integrity_passed = teacher.diagnostics.states_attempted == games * 80
        && teacher.diagnostics.states_translated + teacher.diagnostics.fallbacks
            == teacher.diagnostics.states_attempted
        && expanded_malformed_rate <= 0.10
        && teacher.diagnostics.prefiltered_candidates
            == teacher.diagnostics.prefiltered_candidates_legal
        && teacher.diagnostics.prefiltered_candidates_illegal == 0
        && teacher.diagnostics.selected_actions == teacher.diagnostics.selected_actions_legal
        && fallback_rate <= 0.01;
    let runtime_passed = comparison.treatment_seconds_per_game <= 2_400.0;
    let smoke_passed = bridge_integrity_passed && runtime_passed;
    let (
        treatment_mean_passed,
        paired_gain_passed,
        paired_confidence_passed,
        wildlife_passed,
        habitat_passed,
        nature_tokens_passed,
        non_token_score_passed,
        token_efficiency_passed,
        frontier_recall_passed,
    ) = match gate_profile {
        StrengthGateProfile::Legacy => (
            comparison.treatment_mean >= 94.50,
            comparison.mean_paired_delta >= 1.50,
            true,
            total_wildlife_delta >= 0.50,
            habitat_delta >= -0.50,
            nature_token_delta >= -1.00,
            true,
            true,
            true,
        ),
        StrengthGateProfile::Canonical => (
            comparison.treatment_mean >= 94.00,
            comparison.mean_paired_delta >= 1.25,
            true,
            total_wildlife_delta >= 0.25,
            habitat_delta >= -0.50,
            nature_token_delta >= -1.00,
            true,
            true,
            true,
        ),
        StrengthGateProfile::ProductiveToken => (
            comparison.treatment_mean >= 95.00,
            comparison.mean_paired_delta >= 1.50,
            comparison.confidence_95[0] > 0.0,
            total_wildlife_delta >= 0.0,
            habitat_delta >= 0.0,
            nature_token_delta >= -2.00,
            non_token_score_delta >= 2.00,
            board_points_per_token.is_none()
                || board_points_per_token.is_some_and(|efficiency| efficiency >= 2.00),
            true,
        ),
        StrengthGateProfile::FrontierRecall => (
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            pattern_frontier_recall >= 0.80
                && pattern_frontier_recall_by_phase
                    .iter()
                    .all(|recall| *recall >= 0.65),
        ),
    };
    let qualification_passed = smoke_passed
        && treatment_mean_passed
        && paired_gain_passed
        && paired_confidence_passed
        && wildlife_passed
        && habitat_passed
        && nature_tokens_passed
        && non_token_score_passed
        && token_efficiency_passed
        && frontier_recall_passed;
    let gates = StrengthGates {
        bridge_integrity_passed,
        runtime_passed,
        smoke_passed,
        treatment_mean_passed,
        paired_gain_passed,
        paired_confidence_passed,
        wildlife_passed,
        habitat_passed,
        nature_tokens_passed,
        non_token_score_passed,
        token_efficiency_passed,
        frontier_recall_passed,
        qualification_passed,
    };
    let status = if games == 1 {
        if smoke_passed {
            "smoke-passed"
        } else {
            "rejected"
        }
    } else if qualification_passed {
        "qualified"
    } else {
        "rejected"
    };
    let report = StrengthReport {
        schema_version: 2,
        experiment_id,
        status,
        rollouts,
        diagnostics: teacher.diagnostics,
        fallback_rate,
        expanded_malformed_rate,
        total_wildlife_delta,
        habitat_delta,
        non_token_score_delta,
        token_spend,
        board_points_per_token,
        pattern_frontier_recall,
        independent_pattern_frontier_recall,
        pattern_frontier_recall_by_phase,
        gates,
        comparison,
        provenance,
    };
    write_json_atomic(output, &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if status == "rejected" {
        return Err("strength comparison failed its applicable frozen gates".into());
    }
    Ok(())
}

fn run_exact_mlx_comparison(
    server_program: &str,
    model_dir: &Path,
    games: usize,
    first_seed: u64,
    seed_split: Option<DatasetSplit>,
    rollouts: usize,
    weights: &Path,
    output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if games == 0 || rollouts == 0 {
        return Err("exact MLX strength comparison requires positive games and rollouts".into());
    }
    validate_legacy_environment()?;
    let provenance = provenance(weights)?;
    let model_manifest = model_dir.join("model.json");
    let model_safetensors = model_dir.join("model.safetensors");
    let service_started = Instant::now();
    let mut process = ModelProcess::spawn(
        server_program,
        [
            std::ffi::OsString::from("run"),
            std::ffi::OsString::from("cascadia-mlx-legacy-nnue-serve"),
            std::ffi::OsString::from("--model-dir"),
            model_dir.as_os_str().to_owned(),
        ],
    )?;
    let warmup = process.predict_sparse_nnue_csr_exact(&[Vec::new()])?;
    if warmup.len() != 1 || !warmup[0].is_finite() {
        return Err("exact MLX service warmup returned an invalid value".into());
    }
    let service_startup_milliseconds = service_started.elapsed().as_secs_f64() * 1000.0;

    let game = GameConfig::research_aaaaa(4)?;
    let strong = LateConservativeBasePolicyImprovementStrategy::new(
        LateConservativeBasePolicyImprovementConfig::default(),
    )?;
    let mut teacher = ExactMlxLegacyTeacher::new(process, rollouts)?;
    let started = Instant::now();
    let mut results: Vec<(u64, MatchResult, MatchResult)> = Vec::with_capacity(games);
    for offset in 0..games {
        let seed_value = first_seed + offset as u64;
        let seed = seed_split.map_or_else(
            || GameSeed::from_u64(seed_value),
            |split| split.game_seed(seed_value),
        );
        let baseline = strong.play_match(game, seed)?;
        let mut fallback_rngs = (0..4)
            .map(|seat| strategy_rng(seed, seat, PATTERN_AWARE_STRATEGY_ID))
            .collect::<Vec<_>>();
        let mut decisions = 0usize;
        let treatment = play_match_with_selector(
            game,
            seed,
            EXACT_MLX_LEGACY_TEACHER_STRATEGY_ID,
            |player, game| {
                let action = match teacher.select_action(game) {
                    Ok(action) => Ok(action),
                    Err(error) if error.permits_coordinate_fallback() => {
                        teacher.diagnostics.fallbacks += 1;
                        pattern_fallback(game, &mut fallback_rngs[player])
                            .map_err(simulation_error)
                            .map_err(|fallback_error| {
                                cascadia_sim::SimulationError::Strategy(format!(
                                    "exact MLX bridge failed ({error}); fallback failed \
                                     ({fallback_error})"
                                ))
                            })
                    }
                    Err(error) => Err(simulation_error(error)),
                }?;
                decisions += 1;
                if decisions.is_multiple_of(10) || decisions == 80 {
                    eprintln!(
                        "exact MLX gameplay seed {seed_value}: {decisions}/80 decisions, {:.1}s elapsed",
                        started.elapsed().as_secs_f64()
                    );
                }
                Ok(action)
            },
        )?;
        results.push((seed_value, baseline, treatment));
        eprintln!(
            "exact MLX gameplay: {}/{} games complete, {:.1}s elapsed",
            offset + 1,
            games,
            started.elapsed().as_secs_f64()
        );
    }

    let diagnostics = teacher.diagnostics.clone();
    let batch = teacher.batch_diagnostics;
    let clean_shutdown = teacher.shutdown().is_ok();
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let comparison = summarize_paired_match_results(
        strong.strategy_id(),
        EXACT_MLX_LEGACY_TEACHER_STRATEGY_ID,
        first_seed,
        &results,
        elapsed_seconds,
    );
    let game_records = results
        .iter()
        .map(|(game_index, baseline, treatment)| ExactMlxGameRecord {
            seed: *game_index,
            game_seed: baseline.seed,
            baseline_scores: baseline.scores.clone(),
            treatment_scores: treatment.scores.clone(),
            baseline_decision_seconds: baseline.decision_seconds.clone(),
            treatment_decision_seconds: treatment.decision_seconds.clone(),
            baseline_elapsed_seconds: baseline.elapsed_seconds,
            treatment_elapsed_seconds: treatment.elapsed_seconds,
        })
        .collect();
    let batch_diagnostics = SearchBatchDiagnostics {
        neural_batches: batch.neural_batches,
        neural_rows: batch.neural_rows,
        minimum_batch_rows: batch.minimum_batch_rows,
        maximum_batch_rows: batch.maximum_batch_rows,
        rollout_waves: batch.rollout_waves,
        rollout_samples: batch.rollout_samples,
        policy_fallbacks: batch.policy_fallbacks,
    };
    let fallback_rate = diagnostics.fallback_rate();
    let expanded_malformed_rate = if diagnostics.expanded_candidates == 0 {
        0.0
    } else {
        diagnostics.expanded_candidates_illegal as f64 / diagnostics.expanded_candidates as f64
    };
    let total_wildlife_delta = comparison.mean_breakdown_delta.wildlife.iter().sum::<f64>();
    let habitat_delta = comparison.mean_breakdown_delta.habitat.iter().sum::<f64>();
    let nature_token_delta = comparison.mean_breakdown_delta.nature_tokens;
    let token_spend = (-nature_token_delta).max(0.0);
    let non_token_score_delta = comparison.mean_paired_delta - nature_token_delta;
    let board_points_per_token = if token_spend > 0.0 {
        Some(non_token_score_delta / token_spend)
    } else {
        None
    };
    let bridge_integrity_passed = diagnostics.states_attempted == games * 80
        && diagnostics.states_translated + diagnostics.fallbacks == diagnostics.states_attempted
        && expanded_malformed_rate <= 0.10
        && diagnostics.prefiltered_candidates == diagnostics.prefiltered_candidates_legal
        && diagnostics.prefiltered_candidates_illegal == 0
        && diagnostics.selected_actions == diagnostics.selected_actions_legal
        && fallback_rate <= 0.01
        && batch_diagnostics.policy_fallbacks == 0;
    let runtime_passed = comparison.treatment_seconds_per_game <= 240.0;
    let smoke_passed = bridge_integrity_passed && runtime_passed && clean_shutdown;
    let treatment_mean_passed = comparison.treatment_mean >= 95.0;
    let paired_gain_passed = comparison.mean_paired_delta >= 1.5;
    let paired_confidence_passed = comparison.confidence_95[0] > 0.0;
    let wildlife_passed = total_wildlife_delta >= 0.0;
    let habitat_passed = habitat_delta >= 0.0;
    let nature_tokens_passed = nature_token_delta >= -2.0;
    let non_token_score_passed = non_token_score_delta >= 2.0;
    let token_efficiency_passed = board_points_per_token.is_none_or(|efficiency| efficiency >= 2.0);
    let qualification_passed = smoke_passed
        && treatment_mean_passed
        && paired_gain_passed
        && paired_confidence_passed
        && wildlife_passed
        && habitat_passed
        && nature_tokens_passed
        && non_token_score_passed
        && token_efficiency_passed;
    let gates = StrengthGates {
        bridge_integrity_passed,
        runtime_passed,
        smoke_passed,
        treatment_mean_passed,
        paired_gain_passed,
        paired_confidence_passed,
        wildlife_passed,
        habitat_passed,
        nature_tokens_passed,
        non_token_score_passed,
        token_efficiency_passed,
        frontier_recall_passed: true,
        qualification_passed,
    };
    let status = if games == 1 {
        if smoke_passed {
            "smoke-passed"
        } else {
            "rejected"
        }
    } else if qualification_passed {
        "qualified"
    } else {
        "rejected"
    };
    let report = ExactMlxStrengthReport {
        schema_version: 1,
        experiment_id: "qualified-legacy-nnue-exact-mlx-gameplay-reproduction-v1-20260612",
        status,
        seed_domain: seed_split.map_or("raw-u64", DatasetSplit::id),
        rollouts,
        diagnostics,
        batch_diagnostics,
        fallback_rate,
        expanded_malformed_rate,
        total_wildlife_delta,
        habitat_delta,
        non_token_score_delta,
        token_spend,
        board_points_per_token,
        service_startup_milliseconds,
        clean_shutdown,
        gates,
        comparison,
        game_records,
        model_manifest_path: model_manifest.canonicalize()?,
        model_manifest_blake3: checksum_file(&model_manifest)?,
        model_safetensors_blake3: checksum_file(&model_safetensors)?,
        provenance,
    };
    write_json_atomic(output, &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if status == "rejected" {
        return Err("exact MLX strength comparison failed its frozen gates".into());
    }
    Ok(())
}

fn spawn_exact_mlx_teacher(
    server_program: &str,
    model_dir: &Path,
    rollouts: usize,
    candidate_limit: usize,
) -> Result<(ExactMlxLegacyTeacher, f64), Box<dyn std::error::Error>> {
    spawn_exact_mlx_teacher_with_candidates(server_program, model_dir, rollouts, candidate_limit, 0)
}

fn spawn_exact_mlx_teacher_with_seed_coupling(
    server_program: &str,
    model_dir: &Path,
    rollouts: usize,
    seed_coupling: RolloutSeedCoupling,
) -> Result<(ExactMlxLegacyTeacher, f64), Box<dyn std::error::Error>> {
    let (process, startup_milliseconds) = spawn_exact_mlx_process(server_program, model_dir)?;
    let teacher = ExactMlxLegacyTeacher::new_with_seed_coupling(process, rollouts, seed_coupling)?;
    Ok((teacher, startup_milliseconds))
}

fn spawn_exact_mlx_teacher_with_candidates(
    server_program: &str,
    model_dir: &Path,
    rollouts: usize,
    candidate_limit: usize,
    habitat_candidate_limit: usize,
) -> Result<(ExactMlxLegacyTeacher, f64), Box<dyn std::error::Error>> {
    if habitat_candidate_limit > 0 && candidate_limit != 32 {
        return Err("habitat candidate injection currently requires canonical K32".into());
    }
    let (process, startup_milliseconds) = spawn_exact_mlx_process(server_program, model_dir)?;
    let teacher = if habitat_candidate_limit == 0 {
        ExactMlxLegacyTeacher::new_with_candidate_limit(process, rollouts, candidate_limit)?
    } else {
        ExactMlxLegacyTeacher::new_with_habitat_candidates(
            process,
            rollouts,
            habitat_candidate_limit,
        )?
    };
    Ok((teacher, startup_milliseconds))
}

fn spawn_exact_mlx_process(
    server_program: &str,
    model_dir: &Path,
) -> Result<(ModelProcess, f64), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let mut process = ModelProcess::spawn(
        server_program,
        [
            std::ffi::OsString::from("run"),
            std::ffi::OsString::from("cascadia-mlx-legacy-nnue-serve"),
            std::ffi::OsString::from("--model-dir"),
            model_dir.as_os_str().to_owned(),
        ],
    )?;
    let warmup = process.predict_sparse_nnue_csr_exact(&[Vec::new()])?;
    if warmup.len() != 1 || !warmup[0].is_finite() {
        return Err("exact MLX service warmup returned an invalid value".into());
    }
    let startup_milliseconds = started.elapsed().as_secs_f64() * 1000.0;
    Ok((process, startup_milliseconds))
}

fn play_exact_mlx_match(
    teacher: &mut ExactMlxLegacyTeacher,
    config: GameConfig,
    seed: GameSeed,
    strategy_id: &str,
    label: &str,
    experiment_started: Instant,
) -> Result<MatchResult, Box<dyn std::error::Error>> {
    let mut fallback_rngs = (0..4)
        .map(|seat| strategy_rng(seed, seat, PATTERN_AWARE_STRATEGY_ID))
        .collect::<Vec<_>>();
    let mut decisions = 0usize;
    Ok(play_match_with_selector(
        config,
        seed,
        strategy_id,
        |player, game| {
            let action = match teacher.select_action(game) {
                Ok(action) => Ok(action),
                Err(error) if error.permits_coordinate_fallback() => {
                    teacher.diagnostics.fallbacks += 1;
                    pattern_fallback(game, &mut fallback_rngs[player])
                        .map_err(simulation_error)
                        .map_err(|fallback_error| {
                            cascadia_sim::SimulationError::Strategy(format!(
                                "exact MLX bridge failed ({error}); fallback failed \
                                 ({fallback_error})"
                            ))
                        })
                }
                Err(error) => Err(simulation_error(error)),
            }?;
            decisions += 1;
            if decisions.is_multiple_of(10) || decisions == 80 {
                eprintln!(
                    "{label}: {decisions}/80 decisions, {:.1}s elapsed",
                    experiment_started.elapsed().as_secs_f64()
                );
            }
            Ok(action)
        },
    )?)
}

fn play_exact_mlx_paired_games(
    baseline: &mut ExactMlxLegacyTeacher,
    treatment: &mut ExactMlxLegacyTeacher,
    run: &ExactMlxPairedGamesConfig<'_>,
) -> Result<(PairedMatchResults, f64), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let mut results = Vec::with_capacity(run.games);
    for offset in 0..run.games {
        let seed_value = run.first_seed + offset as u64;
        let seed = GameSeed::from_u64(seed_value);
        let baseline_match = play_exact_mlx_match(
            baseline,
            run.game,
            seed,
            run.baseline_id,
            &format!("{} seed {seed_value}", run.baseline_label),
            started,
        )?;
        let treatment_match = play_exact_mlx_match(
            treatment,
            run.game,
            seed,
            run.treatment_id,
            &format!("{} seed {seed_value}", run.treatment_label),
            started,
        )?;
        results.push((seed_value, baseline_match, treatment_match));
        eprintln!(
            "{}: {}/{} pairs complete, {:.1}s elapsed",
            run.progress_label,
            offset + 1,
            run.games,
            started.elapsed().as_secs_f64()
        );
    }
    Ok((results, started.elapsed().as_secs_f64()))
}

fn search_batch_diagnostics(batch: BatchedNnueDiagnostics) -> SearchBatchDiagnostics {
    SearchBatchDiagnostics {
        neural_batches: batch.neural_batches,
        neural_rows: batch.neural_rows,
        minimum_batch_rows: batch.minimum_batch_rows,
        maximum_batch_rows: batch.maximum_batch_rows,
        rollout_waves: batch.rollout_waves,
        rollout_samples: batch.rollout_samples,
        policy_fallbacks: batch.policy_fallbacks,
    }
}

fn exact_teacher_integrity(
    diagnostics: &BridgeDiagnostics,
    batch: &SearchBatchDiagnostics,
    games: usize,
) -> bool {
    let malformed_rate = if diagnostics.expanded_candidates == 0 {
        0.0
    } else {
        diagnostics.expanded_candidates_illegal as f64 / diagnostics.expanded_candidates as f64
    };
    diagnostics.states_attempted == games * 80
        && diagnostics.states_translated + diagnostics.fallbacks == diagnostics.states_attempted
        && malformed_rate <= 0.10
        && diagnostics.prefiltered_candidates == diagnostics.prefiltered_candidates_legal
        && diagnostics.prefiltered_candidates_illegal == 0
        && diagnostics.selected_actions == diagnostics.selected_actions_legal
        && diagnostics.fallback_rate() <= 0.01
        && batch.rollout_samples > 0
        && batch.policy_fallbacks == 0
}

#[derive(Debug, Clone, Copy)]
struct ExactMlxPairThresholds {
    baseline_runtime_seconds: f64,
    treatment_runtime_seconds: f64,
    paired_gain: f64,
    minimum_confidence_95_lower: Option<f64>,
    treatment_mean: f64,
    wildlife_delta: f64,
    habitat_delta: f64,
    nature_token_delta: f64,
}

#[derive(Debug, Clone, Copy)]
struct ExactMlxPairAssessment<'a> {
    games: usize,
    baseline_diagnostics: &'a BridgeDiagnostics,
    treatment_diagnostics: &'a BridgeDiagnostics,
    baseline_batch: &'a SearchBatchDiagnostics,
    treatment_batch: &'a SearchBatchDiagnostics,
    baseline_clean_shutdown: bool,
    treatment_clean_shutdown: bool,
    comparison: &'a ComparisonReport,
    thresholds: ExactMlxPairThresholds,
    success_status: &'static str,
}

fn assess_exact_mlx_pair(
    assessment: ExactMlxPairAssessment<'_>,
) -> (f64, f64, ExactMlxRolloutBudgetGates, &'static str) {
    let ExactMlxPairAssessment {
        games,
        baseline_diagnostics,
        treatment_diagnostics,
        baseline_batch,
        treatment_batch,
        baseline_clean_shutdown,
        treatment_clean_shutdown,
        comparison,
        thresholds,
        success_status,
    } = assessment;
    let total_wildlife_delta = comparison.mean_breakdown_delta.wildlife.iter().sum::<f64>();
    let habitat_delta = comparison.mean_breakdown_delta.habitat.iter().sum::<f64>();
    let baseline_integrity = exact_teacher_integrity(baseline_diagnostics, baseline_batch, games);
    let treatment_integrity =
        exact_teacher_integrity(treatment_diagnostics, treatment_batch, games);
    let baseline_runtime =
        comparison.baseline_seconds_per_game <= thresholds.baseline_runtime_seconds;
    let treatment_runtime =
        comparison.treatment_seconds_per_game <= thresholds.treatment_runtime_seconds;
    let clean_shutdown = baseline_clean_shutdown && treatment_clean_shutdown;
    let smoke_passed = baseline_integrity
        && treatment_integrity
        && baseline_runtime
        && treatment_runtime
        && clean_shutdown;
    let paired_gain = comparison.mean_paired_delta >= thresholds.paired_gain;
    let paired_confidence = thresholds
        .minimum_confidence_95_lower
        .is_none_or(|minimum| comparison.confidence_95[0] > minimum);
    let treatment_mean = comparison.treatment_mean >= thresholds.treatment_mean;
    let wildlife = total_wildlife_delta >= thresholds.wildlife_delta;
    let habitat = habitat_delta >= thresholds.habitat_delta;
    let nature_tokens =
        comparison.mean_breakdown_delta.nature_tokens >= thresholds.nature_token_delta;
    let pilot_promising = smoke_passed
        && paired_gain
        && paired_confidence
        && treatment_mean
        && wildlife
        && habitat
        && nature_tokens;
    let gates = ExactMlxRolloutBudgetGates {
        baseline_integrity,
        treatment_integrity,
        baseline_runtime,
        treatment_runtime,
        clean_shutdown,
        smoke_passed,
        paired_gain,
        paired_confidence,
        treatment_mean,
        wildlife,
        habitat,
        nature_tokens,
        pilot_promising,
    };
    let status = if games == 1 {
        if smoke_passed {
            "smoke-passed"
        } else {
            "rejected"
        }
    } else if pilot_promising {
        success_status
    } else {
        "rejected"
    };
    (total_wildlife_delta, habitat_delta, gates, status)
}

fn run_exact_mlx_rollout_budget_comparison(
    run: ExactMlxRolloutBudgetConfig<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let ExactMlxRolloutBudgetConfig {
        server_program,
        model_dir,
        games,
        first_seed,
        baseline_rollouts,
        treatment_rollouts,
        weights,
        output,
    } = run;
    if games == 0 || baseline_rollouts == 0 || treatment_rollouts == 0 {
        return Err("exact MLX rollout-budget comparison requires positive inputs".into());
    }
    if baseline_rollouts >= treatment_rollouts {
        return Err("treatment rollout budget must exceed baseline".into());
    }
    validate_legacy_environment()?;
    let provenance = provenance(weights)?;
    let model_manifest = model_dir.join("model.json");
    let model_safetensors = model_dir.join("model.safetensors");
    let (mut baseline, baseline_startup_milliseconds) =
        spawn_exact_mlx_teacher(server_program, model_dir, baseline_rollouts, 32)?;
    let (mut treatment, treatment_startup_milliseconds) =
        spawn_exact_mlx_teacher(server_program, model_dir, treatment_rollouts, 32)?;
    let game = GameConfig::research_aaaaa(4)?;
    let baseline_id = format!("canonical-action-legacy-exact-mlx-k32-r{baseline_rollouts}");
    let treatment_id = format!("canonical-action-legacy-exact-mlx-k32-r{treatment_rollouts}");
    let (results, elapsed_seconds) = play_exact_mlx_paired_games(
        &mut baseline,
        &mut treatment,
        &ExactMlxPairedGamesConfig {
            game,
            games,
            first_seed,
            baseline_id: &baseline_id,
            treatment_id: &treatment_id,
            baseline_label: format!("R{baseline_rollouts}"),
            treatment_label: format!("R{treatment_rollouts}"),
            progress_label: "rollout-budget pilot",
        },
    )?;

    let baseline_diagnostics = baseline.diagnostics.clone();
    let treatment_diagnostics = treatment.diagnostics.clone();
    let baseline_batch_diagnostics = search_batch_diagnostics(baseline.batch_diagnostics);
    let treatment_batch_diagnostics = search_batch_diagnostics(treatment.batch_diagnostics);
    let baseline_clean_shutdown = baseline.shutdown().is_ok();
    let treatment_clean_shutdown = treatment.shutdown().is_ok();
    let comparison = summarize_paired_match_results(
        &baseline_id,
        &treatment_id,
        first_seed,
        &results,
        elapsed_seconds,
    );
    let (total_wildlife_delta, habitat_delta, gates, status) =
        assess_exact_mlx_pair(ExactMlxPairAssessment {
            games,
            baseline_diagnostics: &baseline_diagnostics,
            treatment_diagnostics: &treatment_diagnostics,
            baseline_batch: &baseline_batch_diagnostics,
            treatment_batch: &treatment_batch_diagnostics,
            baseline_clean_shutdown,
            treatment_clean_shutdown,
            comparison: &comparison,
            thresholds: ExactMlxPairThresholds {
                baseline_runtime_seconds: 240.0,
                treatment_runtime_seconds: 420.0,
                paired_gain: 0.50,
                minimum_confidence_95_lower: None,
                treatment_mean: 95.50,
                wildlife_delta: -0.50,
                habitat_delta: -0.50,
                nature_token_delta: -1.00,
            },
            success_status: "promising",
        });
    let report = ExactMlxRolloutBudgetReport {
        schema_version: 1,
        experiment_id: "exact-mlx-rollout-budget-r600-r1200-pilot-v1-20260612",
        status,
        baseline_rollouts,
        treatment_rollouts,
        baseline_diagnostics,
        treatment_diagnostics,
        baseline_batch_diagnostics,
        treatment_batch_diagnostics,
        baseline_startup_milliseconds,
        treatment_startup_milliseconds,
        baseline_clean_shutdown,
        treatment_clean_shutdown,
        total_wildlife_delta,
        habitat_delta,
        gates,
        comparison,
        model_manifest_path: model_manifest.canonicalize()?,
        model_manifest_blake3: checksum_file(&model_manifest)?,
        model_safetensors_blake3: checksum_file(&model_safetensors)?,
        provenance,
    };
    write_json_atomic(output, &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if status == "rejected" {
        return Err("exact MLX rollout-budget comparison failed its frozen gates".into());
    }
    Ok(())
}

fn run_exact_mlx_crn_comparison(
    run: ExactMlxCrnConfig<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let ExactMlxCrnConfig {
        server_program,
        model_dir,
        games,
        expected_games,
        first_seed,
        rollouts,
        weights,
        output,
        experiment_id,
        progress_label,
        thresholds,
        success_status,
        rejection_message,
    } = run;
    if games == 0 || rollouts == 0 {
        return Err("exact MLX CRN comparison requires positive inputs".into());
    }
    if expected_games.is_some_and(|expected| games != expected) {
        return Err(format!(
            "exact MLX CRN confirmation requires exactly {} games",
            expected_games.expect("checked as present")
        )
        .into());
    }
    validate_legacy_environment()?;
    let provenance = provenance(weights)?;
    let model_manifest = model_dir.join("model.json");
    let model_safetensors = model_dir.join("model.safetensors");
    let (mut baseline, baseline_startup_milliseconds) = spawn_exact_mlx_teacher_with_seed_coupling(
        server_program,
        model_dir,
        rollouts,
        RolloutSeedCoupling::Independent,
    )?;
    let (mut treatment, treatment_startup_milliseconds) =
        spawn_exact_mlx_teacher_with_seed_coupling(
            server_program,
            model_dir,
            rollouts,
            RolloutSeedCoupling::CommonWithinRound,
        )?;
    let game = GameConfig::research_aaaaa(4)?;
    let baseline_id = format!("canonical-action-legacy-exact-mlx-k32-r{rollouts}-independent");
    let treatment_id = format!("canonical-action-legacy-exact-mlx-k32-r{rollouts}-crn");
    let (results, elapsed_seconds) = play_exact_mlx_paired_games(
        &mut baseline,
        &mut treatment,
        &ExactMlxPairedGamesConfig {
            game,
            games,
            first_seed,
            baseline_id: &baseline_id,
            treatment_id: &treatment_id,
            baseline_label: "independent".to_owned(),
            treatment_label: "CRN".to_owned(),
            progress_label,
        },
    )?;

    let baseline_diagnostics = baseline.diagnostics.clone();
    let treatment_diagnostics = treatment.diagnostics.clone();
    let baseline_batch_diagnostics = search_batch_diagnostics(baseline.batch_diagnostics);
    let treatment_batch_diagnostics = search_batch_diagnostics(treatment.batch_diagnostics);
    let baseline_clean_shutdown = baseline.shutdown().is_ok();
    let treatment_clean_shutdown = treatment.shutdown().is_ok();
    let comparison = summarize_paired_match_results(
        &baseline_id,
        &treatment_id,
        first_seed,
        &results,
        elapsed_seconds,
    );
    let (total_wildlife_delta, habitat_delta, gates, status) =
        assess_exact_mlx_pair(ExactMlxPairAssessment {
            games,
            baseline_diagnostics: &baseline_diagnostics,
            treatment_diagnostics: &treatment_diagnostics,
            baseline_batch: &baseline_batch_diagnostics,
            treatment_batch: &treatment_batch_diagnostics,
            baseline_clean_shutdown,
            treatment_clean_shutdown,
            comparison: &comparison,
            thresholds,
            success_status,
        });
    let report = ExactMlxCrnReport {
        schema_version: 2,
        experiment_id,
        status,
        rollouts,
        baseline_seed_coupling: "independent",
        treatment_seed_coupling: "common-within-round",
        baseline_diagnostics,
        treatment_diagnostics,
        baseline_batch_diagnostics,
        treatment_batch_diagnostics,
        baseline_startup_milliseconds,
        treatment_startup_milliseconds,
        baseline_clean_shutdown,
        treatment_clean_shutdown,
        total_wildlife_delta,
        habitat_delta,
        gates,
        comparison,
        model_manifest_path: model_manifest.canonicalize()?,
        model_manifest_blake3: checksum_file(&model_manifest)?,
        model_safetensors_blake3: checksum_file(&model_safetensors)?,
        provenance,
    };
    write_json_atomic(output, &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if status == "rejected" {
        return Err(rejection_message.into());
    }
    Ok(())
}

fn run_exact_mlx_candidate_limit_comparison(
    run: ExactMlxCandidateLimitConfig<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let ExactMlxCandidateLimitConfig {
        server_program,
        model_dir,
        games,
        first_seed,
        baseline_candidate_limit,
        treatment_candidate_limit,
        rollouts,
        weights,
        output,
    } = run;
    if games == 0 || rollouts == 0 {
        return Err("exact MLX candidate-limit comparison requires positive inputs".into());
    }
    if baseline_candidate_limit < 32 || treatment_candidate_limit <= baseline_candidate_limit {
        return Err("candidate limits must satisfy 32 <= baseline < treatment".into());
    }
    validate_legacy_environment()?;
    let provenance = provenance(weights)?;
    let model_manifest = model_dir.join("model.json");
    let model_safetensors = model_dir.join("model.safetensors");
    let (mut baseline, baseline_startup_milliseconds) = spawn_exact_mlx_teacher(
        server_program,
        model_dir,
        rollouts,
        baseline_candidate_limit,
    )?;
    let (mut treatment, treatment_startup_milliseconds) = spawn_exact_mlx_teacher(
        server_program,
        model_dir,
        rollouts,
        treatment_candidate_limit,
    )?;
    let game = GameConfig::research_aaaaa(4)?;
    let baseline_id =
        format!("canonical-action-legacy-exact-mlx-k{baseline_candidate_limit}-r{rollouts}");
    let treatment_id =
        format!("canonical-action-legacy-exact-mlx-k{treatment_candidate_limit}-r{rollouts}");
    let (results, elapsed_seconds) = play_exact_mlx_paired_games(
        &mut baseline,
        &mut treatment,
        &ExactMlxPairedGamesConfig {
            game,
            games,
            first_seed,
            baseline_id: &baseline_id,
            treatment_id: &treatment_id,
            baseline_label: format!("K{baseline_candidate_limit}"),
            treatment_label: format!("K{treatment_candidate_limit}"),
            progress_label: "candidate-limit pilot",
        },
    )?;

    let baseline_diagnostics = baseline.diagnostics.clone();
    let treatment_diagnostics = treatment.diagnostics.clone();
    let baseline_batch_diagnostics = search_batch_diagnostics(baseline.batch_diagnostics);
    let treatment_batch_diagnostics = search_batch_diagnostics(treatment.batch_diagnostics);
    let baseline_clean_shutdown = baseline.shutdown().is_ok();
    let treatment_clean_shutdown = treatment.shutdown().is_ok();
    let comparison = summarize_paired_match_results(
        &baseline_id,
        &treatment_id,
        first_seed,
        &results,
        elapsed_seconds,
    );
    let (total_wildlife_delta, habitat_delta, gates, status) =
        assess_exact_mlx_pair(ExactMlxPairAssessment {
            games,
            baseline_diagnostics: &baseline_diagnostics,
            treatment_diagnostics: &treatment_diagnostics,
            baseline_batch: &baseline_batch_diagnostics,
            treatment_batch: &treatment_batch_diagnostics,
            baseline_clean_shutdown,
            treatment_clean_shutdown,
            comparison: &comparison,
            thresholds: ExactMlxPairThresholds {
                baseline_runtime_seconds: 240.0,
                treatment_runtime_seconds: 240.0,
                paired_gain: 0.50,
                minimum_confidence_95_lower: None,
                treatment_mean: 95.50,
                wildlife_delta: -0.50,
                habitat_delta: -0.50,
                nature_token_delta: -1.00,
            },
            success_status: "promising",
        });
    let report = ExactMlxCandidateLimitReport {
        schema_version: 1,
        experiment_id: "exact-mlx-root-candidate-k32-k64-pilot-v1-20260612",
        status,
        baseline_candidate_limit,
        treatment_candidate_limit,
        rollouts,
        baseline_diagnostics,
        treatment_diagnostics,
        baseline_batch_diagnostics,
        treatment_batch_diagnostics,
        baseline_startup_milliseconds,
        treatment_startup_milliseconds,
        baseline_clean_shutdown,
        treatment_clean_shutdown,
        total_wildlife_delta,
        habitat_delta,
        gates,
        comparison,
        model_manifest_path: model_manifest.canonicalize()?,
        model_manifest_blake3: checksum_file(&model_manifest)?,
        model_safetensors_blake3: checksum_file(&model_safetensors)?,
        provenance,
    };
    write_json_atomic(output, &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if status == "rejected" {
        return Err("exact MLX candidate-limit comparison failed its frozen gates".into());
    }
    Ok(())
}

fn run_exact_mlx_habitat_candidate_comparison(
    run: ExactMlxHabitatCandidateConfig<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let ExactMlxHabitatCandidateConfig {
        server_program,
        model_dir,
        games,
        first_seed,
        habitat_candidates,
        rollouts,
        weights,
        output,
    } = run;
    if games == 0 || habitat_candidates == 0 || rollouts == 0 {
        return Err("exact MLX habitat-candidate comparison requires positive inputs".into());
    }
    validate_legacy_environment()?;
    let provenance = provenance(weights)?;
    let model_manifest = model_dir.join("model.json");
    let model_safetensors = model_dir.join("model.safetensors");
    let (mut baseline, baseline_startup_milliseconds) =
        spawn_exact_mlx_teacher(server_program, model_dir, rollouts, 32)?;
    let (mut treatment, treatment_startup_milliseconds) = spawn_exact_mlx_teacher_with_candidates(
        server_program,
        model_dir,
        rollouts,
        32,
        habitat_candidates,
    )?;
    let game = GameConfig::research_aaaaa(4)?;
    let baseline_id = format!("canonical-action-legacy-exact-mlx-k32-r{rollouts}");
    let treatment_id =
        format!("canonical-action-legacy-exact-mlx-k32-h{habitat_candidates}-r{rollouts}");
    let (results, elapsed_seconds) = play_exact_mlx_paired_games(
        &mut baseline,
        &mut treatment,
        &ExactMlxPairedGamesConfig {
            game,
            games,
            first_seed,
            baseline_id: &baseline_id,
            treatment_id: &treatment_id,
            baseline_label: "K32".to_owned(),
            treatment_label: format!("K32+H{habitat_candidates}"),
            progress_label: "habitat-candidate pilot",
        },
    )?;

    let baseline_diagnostics = baseline.diagnostics.clone();
    let treatment_diagnostics = treatment.diagnostics.clone();
    let baseline_batch_diagnostics = search_batch_diagnostics(baseline.batch_diagnostics);
    let treatment_batch_diagnostics = search_batch_diagnostics(treatment.batch_diagnostics);
    let baseline_clean_shutdown = baseline.shutdown().is_ok();
    let treatment_clean_shutdown = treatment.shutdown().is_ok();
    let comparison = summarize_paired_match_results(
        &baseline_id,
        &treatment_id,
        first_seed,
        &results,
        elapsed_seconds,
    );
    let (total_wildlife_delta, habitat_delta, gates, status) =
        assess_exact_mlx_pair(ExactMlxPairAssessment {
            games,
            baseline_diagnostics: &baseline_diagnostics,
            treatment_diagnostics: &treatment_diagnostics,
            baseline_batch: &baseline_batch_diagnostics,
            treatment_batch: &treatment_batch_diagnostics,
            baseline_clean_shutdown,
            treatment_clean_shutdown,
            comparison: &comparison,
            thresholds: ExactMlxPairThresholds {
                baseline_runtime_seconds: 240.0,
                treatment_runtime_seconds: 240.0,
                paired_gain: 0.50,
                minimum_confidence_95_lower: None,
                treatment_mean: 95.50,
                wildlife_delta: -0.50,
                habitat_delta: 0.25,
                nature_token_delta: -1.00,
            },
            success_status: "promising",
        });
    let report = ExactMlxHabitatCandidateReport {
        schema_version: 1,
        experiment_id: "exact-mlx-habitat-candidate-h6-pilot-v1-20260612",
        status,
        candidate_limit: 32,
        habitat_candidates,
        rollouts,
        baseline_diagnostics,
        treatment_diagnostics,
        baseline_batch_diagnostics,
        treatment_batch_diagnostics,
        baseline_startup_milliseconds,
        treatment_startup_milliseconds,
        baseline_clean_shutdown,
        treatment_clean_shutdown,
        total_wildlife_delta,
        habitat_delta,
        gates,
        comparison,
        model_manifest_path: model_manifest.canonicalize()?,
        model_manifest_blake3: checksum_file(&model_manifest)?,
        model_safetensors_blake3: checksum_file(&model_safetensors)?,
        provenance,
    };
    write_json_atomic(output, &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if status == "rejected" {
        return Err("exact MLX habitat-candidate comparison failed its frozen gates".into());
    }
    Ok(())
}

fn provenance(weights: &Path) -> Result<ArtifactProvenance, Box<dyn std::error::Error>> {
    let executable_path = std::env::current_exe()?;
    Ok(ArtifactProvenance {
        source: source_provenance()?,
        executable_blake3: checksum_file(&executable_path)?,
        executable_path,
        weights_blake3: checksum_file(weights)?,
        weights_path: weights.canonicalize()?,
        legacy_environment: vec![
            ("MCE_LMR".to_owned(), "1".to_owned()),
            ("MCE_DIVERSE_PREFILTER".to_owned(), "1".to_owned()),
        ],
    })
}

fn write_json_atomic(
    path: &Path,
    value: &impl Serialize,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("json")
    ));
    fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}
