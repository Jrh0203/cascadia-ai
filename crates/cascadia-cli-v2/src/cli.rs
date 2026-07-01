use std::path::PathBuf;

use cascadia_data::DatasetSplit;
use cascadia_game::Wildlife;
use cascadia_sim::StrategyKind;
use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;

use crate::counterfactual_advantage::CounterfactualCandidateSelectionArg;

#[derive(Debug, Parser)]
#[command(name = "cascadia-v2", version, about = "Cascadia AI v2")]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Debug, Serialize, Subcommand)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum Command {
    /// Run the canonical four-player AAAAA base-score benchmark.
    Benchmark {
        #[arg(long, default_value_t = 4)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long, value_enum, default_value_t = StrategyArg::Random)]
        strategy: StrategyArg,
        #[arg(long)]
        sequential: bool,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare two strategies on the same deterministic game seeds.
    Compare {
        #[arg(long, default_value_t = 20)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long, value_enum)]
        baseline: StrategyArg,
        #[arg(long, value_enum)]
        treatment: StrategyArg,
        #[arg(long)]
        sequential: bool,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Select phase-decayed habitat and Bear structural potential on a frozen grid.
    PatternPotentialSweep {
        #[arg(long, default_value_t = 32)]
        games: usize,
        #[arg(long, default_value_t = 31300)]
        first_seed: u64,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare production pattern-aware with one registered structural-potential point.
    PatternPotentialCompare {
        #[arg(long, default_value_t = 50)]
        games: usize,
        #[arg(long, default_value_t = 31400)]
        first_seed: u64,
        #[arg(long)]
        opportunity_weight: f64,
        #[arg(long)]
        habitat_weight: f64,
        #[arg(long)]
        bear_weight: f64,
        #[arg(long)]
        sequential: bool,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Collect versioned, checksummed MLX training data from canonical games.
    Collect {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_game_index: u64,
        #[arg(long, value_enum, default_value_t = SplitArg::Train)]
        split: SplitArg,
        #[arg(long, value_enum, default_value_t = StrategyArg::Greedy)]
        strategy: StrategyArg,
        #[arg(long, default_value_t = 64)]
        shard_games: usize,
        #[arg(long)]
        resume: bool,
    },
    /// Collect replay-authoritative all-greedy bootstrap games for R2-MAP.
    CollectR2MapBootstrap {
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "r2-map-expert-iteration-v1")]
        campaign_id: String,
        #[arg(long, default_value_t = 0)]
        iteration: u32,
        #[arg(long)]
        host: String,
        #[arg(long)]
        first_game_index: u64,
        #[arg(long)]
        games: u64,
        #[arg(long, default_value_t = 256)]
        shard_games: usize,
        #[arg(long)]
        collector_hash: String,
        #[arg(long)]
        source_hash: String,
        #[arg(long)]
        serving_protocol_hash: String,
        #[arg(long)]
        resume: bool,
    },
    /// Collect one newest-seat expert-iteration shard against a frozen local field.
    CollectR2MapIteration {
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "r2-map-expert-iteration-v1")]
        campaign_id: String,
        #[arg(long)]
        iteration: u32,
        #[arg(long)]
        host: String,
        #[arg(long)]
        first_game_index: u64,
        #[arg(long)]
        games: u64,
        #[arg(long, default_value_t = 64)]
        shard_games: usize,
        #[arg(long, default_value_t = 1_000_000)]
        temperature_parts_per_million: u32,
        #[arg(long)]
        collector_hash: String,
        #[arg(long)]
        source_hash: String,
        #[arg(long)]
        serving_protocol_hash: String,
        #[arg(long)]
        bundle: Option<PathBuf>,
        #[arg(long)]
        newest_manifest_identity: Option<String>,
        #[arg(long = "historical-manifest-identity")]
        historical_manifest_identities: Vec<String>,
        /// Use the deterministic immediate-score predictor for pipeline smoke tests only.
        #[arg(long)]
        exact_score_reference: bool,
        #[arg(long, default_value = ".venv/bin/python")]
        python: PathBuf,
        #[arg(long, default_value = "python")]
        python_path: PathBuf,
        #[arg(long)]
        resume: bool,
    },
    /// Validate an R2-MAP collector manifest and every replay shard.
    ValidateR2MapCollector {
        #[arg(long)]
        dataset: PathBuf,
    },
    /// Extract compact-index identities and candidate widths from validated replay shards.
    InspectR2MapIndexMetadata {
        #[arg(long = "shard", required = true)]
        shards: Vec<PathBuf>,
    },
    /// Stream exact-R2 R2-MAP groups without a persistent padded tensor cache.
    ExportR2MapDataset {
        #[arg(long = "shard", required = true)]
        shards: Vec<PathBuf>,
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        stream: PathBuf,
        #[arg(long, value_enum)]
        mode: R2MapDatasetModeArg,
        #[arg(long, default_value_t = 0)]
        epoch: u64,
        #[arg(long, default_value_t = 0)]
        sampler_seed: u64,
        #[arg(long, default_value_t = 0)]
        fixed_panel_games: usize,
        /// Restrict this disposable stream to whole games at these global indices.
        #[arg(long = "game-index")]
        game_indices: Vec<u64>,
        /// Aggregate receipt proving the replay shards passed full semantic validation.
        #[arg(long, requires_all = ["validated_compact_index", "validated_packing_receipt"])]
        validated_aggregate_receipt: Option<PathBuf>,
        /// Full compact index whose SHA-256 is bound by the packing receipt.
        #[arg(long, requires_all = ["validated_aggregate_receipt", "validated_packing_receipt"])]
        validated_compact_index: Option<PathBuf>,
        /// Packing receipt binding the aggregate receipt, compact index, and dataset.
        #[arg(long, requires_all = ["validated_aggregate_receipt", "validated_compact_index"])]
        validated_packing_receipt: Option<PathBuf>,
    },
    /// Serve exact focal-seat packed batches over a backpressured binary pipe.
    ServeR2MapPackedBatches {
        #[arg(long)]
        shard: PathBuf,
        #[arg(long, value_enum)]
        mode: R2MapDatasetModeArg,
        #[arg(long, default_value_t = 0)]
        epoch: u64,
        #[arg(long)]
        sampler_seed: u64,
        #[arg(long, default_value_t = 128)]
        group_batch_size: usize,
        #[arg(long, default_value_t = 16_384)]
        maximum_candidates_per_batch: usize,
        /// Retain all focal value targets but omit mediocre greedy-policy CE screens.
        #[arg(long)]
        bootstrap_value_only: bool,
        #[arg(long = "game-index", required = true)]
        ordered_game_indices: Vec<u64>,
        #[arg(long, default_value_t = 0)]
        start_game_offset: usize,
        #[arg(long, default_value_t = 0)]
        start_turn_offset: usize,
        #[arg(long, default_value_t = 0)]
        start_batch_index: u64,
        #[arg(long)]
        validated_aggregate_receipt: PathBuf,
        #[arg(long)]
        validated_compact_index: PathBuf,
        #[arg(long)]
        validated_packing_receipt: PathBuf,
    },
    /// Build and verify a compact local multi-checkpoint R2-MAP serving bundle.
    PrepareR2MapServingBundle {
        #[arg(long)]
        host: String,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        collector_hash: String,
        #[arg(long)]
        source_hash: String,
        #[arg(long)]
        serving_protocol_hash: String,
        #[arg(long = "checkpoint", required = true)]
        checkpoints: Vec<PathBuf>,
    },
    /// Initialize the open fixed-100 longitudinal panel from the frozen W0 manifest.
    InitR2MapLongitudinalOpenPanel {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        reference_panel_manifest: PathBuf,
        #[arg(long)]
        reference_panel_registration: PathBuf,
        #[arg(long, default_value = "r2-map-expert-iteration-v1")]
        campaign_id: String,
        #[arg(long)]
        benchmark_id: String,
        #[arg(long, default_value_t = 0)]
        iteration: u32,
        #[arg(long)]
        focal_checkpoint_id: String,
        #[arg(long)]
        field_manifest_id: String,
        #[arg(long = "historical-checkpoint", required = true)]
        historical_checkpoints: Vec<String>,
        #[arg(long, default_value = "r2-map-reference-argmax-v1")]
        inference_settings_id: String,
    },
    /// Initialize a controller-provisioned fixed-100 longitudinal campaign.
    InitR2MapLongitudinalCampaign {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        contract: PathBuf,
        #[arg(long)]
        historical_field: PathBuf,
    },
    /// Initialize a pre-provisioned 20-pair smoke or fixed-250 focal campaign.
    InitR2MapFocalCampaign {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        contract: PathBuf,
        #[arg(long)]
        opponent_field: PathBuf,
    },
    /// Run one restart-safe scheduler-managed longitudinal R2-MAP game.
    RunR2MapLongitudinalWorkItem {
        #[arg(long)]
        root: PathBuf,
        #[arg(long = "work-item")]
        work_item: String,
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long, default_value = ".venv/bin/python")]
        python: PathBuf,
        #[arg(long, default_value = "python")]
        python_path: PathBuf,
    },
    /// Aggregate the complete fixed-100 longitudinal benchmark and projections.
    AggregateR2MapLongitudinal {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        wall_seconds: f64,
    },
    /// Run one restart-safe scheduler-managed focal benchmark pair.
    RunR2MapFocalWorkItem {
        #[arg(long)]
        root: PathBuf,
        #[arg(long = "work-item")]
        work_item: String,
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long, default_value = ".venv/bin/python")]
        python: PathBuf,
        #[arg(long, default_value = "python")]
        python_path: PathBuf,
    },
    /// Aggregate a complete paired focal campaign and emit promotion/dashboard feeds.
    AggregateR2MapFocal {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        wall_seconds: f64,
    },
    /// Collect final-score value targets from the confirmed H6 search teacher.
    CollectSearch {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_game_index: u64,
        #[arg(long, value_enum, default_value_t = SplitArg::Train)]
        split: SplitArg,
        #[arg(long, default_value_t = 8)]
        shard_games: usize,
        #[arg(long)]
        resume: bool,
        #[arg(long, default_value_t = 8)]
        candidates: usize,
        #[arg(long, default_value_t = 6)]
        habitat_candidates: usize,
        #[arg(long, default_value_t = 4)]
        determinizations: usize,
        #[arg(long, default_value_t = 4)]
        greedy_plies: usize,
    },
    /// Collect signed score-to-go targets from the frozen H6 teacher.
    CollectScoreToGo {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_game_index: u64,
        #[arg(long, value_enum, default_value_t = SplitArg::Train)]
        split: SplitArg,
        #[arg(long, default_value_t = 1)]
        shard_games: usize,
        #[arg(long)]
        resume: bool,
    },
    /// Collect repeated public-redetermination terminal returns from H6 states.
    CollectCounterfactualValue {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_game_index: u64,
        #[arg(long, value_enum, default_value_t = SplitArg::Train)]
        split: SplitArg,
        #[arg(long, default_value_t = 16)]
        samples_per_state: usize,
        #[arg(long)]
        resume: bool,
    },
    /// Collect shared-seed same-decision counterfactual action returns.
    CollectCounterfactualAdvantage {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_game_index: u64,
        #[arg(long, value_enum, default_value_t = SplitArg::Train)]
        split: SplitArg,
        #[arg(long, default_value_t = 16)]
        groups_per_game: usize,
        #[arg(long, default_value_t = 16)]
        samples_per_candidate: usize,
        #[arg(
            long,
            value_enum,
            default_value_t = CounterfactualCandidateSelectionArg::Nearest
        )]
        candidate_selection: CounterfactualCandidateSelectionArg,
        #[arg(long)]
        resume: bool,
    },
    /// Verify a dataset manifest, every shard header, size, and checksum.
    ValidateDataset {
        #[arg(long)]
        dataset: PathBuf,
    },
    /// Verify a signed score-to-go dataset and every target identity.
    ValidateScoreToGoDataset {
        #[arg(long)]
        dataset: PathBuf,
    },
    /// Verify a counterfactual-value dataset and every retained sample.
    ValidateCounterfactualValueDataset {
        #[arg(long)]
        dataset: PathBuf,
    },
    /// Verify a grouped counterfactual-advantage dataset and every raw return.
    ValidateCounterfactualAdvantageDataset {
        #[arg(long)]
        dataset: PathBuf,
    },
    /// Audit counterfactual target stability and projected collection cost.
    AuditCounterfactualValueDataset {
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Audit centered action-advantage stability and projected collection cost.
    AuditCounterfactualAdvantageDataset {
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        markdown_output: PathBuf,
        #[arg(long, default_value_t = 8)]
        estimator_samples: usize,
    },
    /// Collect grouped counterfactual action labels from the confirmed search teacher.
    CollectRanking {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_game_index: u64,
        #[arg(long, value_enum, default_value_t = SplitArg::Train)]
        split: SplitArg,
        #[arg(long, default_value_t = 8)]
        shard_games: usize,
        #[arg(long)]
        resume: bool,
        #[arg(long, value_enum, default_value_t = RankingTeacherArg::Bear)]
        teacher: RankingTeacherArg,
        #[arg(long, default_value_t = 8)]
        candidates: usize,
        #[arg(long, default_value_t = 8)]
        bear_candidates: usize,
        #[arg(long, default_value_t = 6)]
        habitat_candidates: usize,
        #[arg(long, default_value_t = 4)]
        determinizations: usize,
        #[arg(long, default_value_t = 4)]
        greedy_plies: usize,
    },
    /// Collect terminal R8 action values from the qualified policy-improvement teacher.
    CollectTerminalRanking {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_game_index: u64,
        #[arg(long, value_enum, default_value_t = SplitArg::Train)]
        split: SplitArg,
        #[arg(long, default_value_t = 1)]
        shard_games: usize,
        #[arg(long)]
        resume: bool,
        #[arg(long, default_value_t = 8)]
        determinizations: usize,
        #[arg(long, default_value_t = 8)]
        policy_candidates: usize,
        #[arg(long, default_value_t = 6)]
        policy_habitat_candidates: usize,
        #[arg(long, default_value_t = 8)]
        policy_bear_candidates: usize,
        #[arg(long, default_value_t = 4)]
        policy_market_draws: usize,
    },
    /// Collect paired c90 anchor/challenger targets from promoted strong trajectories.
    CollectConservativeAdvantage {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_game_index: u64,
        #[arg(long, value_enum, default_value_t = SplitArg::Train)]
        split: SplitArg,
        #[arg(long, default_value_t = 1)]
        shard_games: usize,
        #[arg(long)]
        resume: bool,
        #[arg(long, default_value_t = 5)]
        terminal_turns: u16,
        #[arg(long, default_value_t = 8)]
        policy_candidates: usize,
        #[arg(long, default_value_t = 6)]
        policy_habitat_candidates: usize,
        #[arg(long, default_value_t = 8)]
        policy_bear_candidates: usize,
        #[arg(long, default_value_t = 4)]
        policy_market_draws: usize,
    },
    /// Collect H6 labels on states visited by a frozen MLX habitat apprentice.
    CollectRankingIteration {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_game_index: u64,
        #[arg(long, value_enum, default_value_t = SplitArg::Train)]
        split: SplitArg,
        #[arg(long, default_value_t = 8)]
        shard_games: usize,
        #[arg(long)]
        resume: bool,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value = ".venv/bin/cascadia-mlx-ranking-serve")]
        server: PathBuf,
        #[arg(long, default_value_t = 8)]
        candidates: usize,
        #[arg(long, default_value_t = 6)]
        habitat_candidates: usize,
        #[arg(long, default_value_t = 4)]
        determinizations: usize,
        #[arg(long, default_value_t = 4)]
        greedy_plies: usize,
    },
    /// Verify a grouped action-ranking dataset and every shard checksum.
    ValidateRankingDataset {
        #[arg(long)]
        dataset: PathBuf,
    },
    /// Enrich frozen terminal-ranking labels with explicit, replay-verified action deltas.
    EnrichActionRanking {
        #[arg(long)]
        source_dataset: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        resume: bool,
        #[arg(long, default_value_t = 4)]
        policy_market_draws: usize,
    },
    /// Verify an action-delta ranking dataset and every shard checksum.
    ValidateActionRankingDataset {
        #[arg(long)]
        dataset: PathBuf,
    },
    /// Verify a paired conservative-advantage dataset and every shard checksum.
    ValidateConservativeAdvantageDataset {
        #[arg(long)]
        dataset: PathBuf,
    },
    /// Collect frozen public-redetermination beam values for MLX training.
    CollectPublicBeamValue {
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
    },
    /// Collect and evaluate the frozen public beam-state value observability probe.
    PublicBeamValueProbe {
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 40_000)]
        first_game_index: u64,
        #[arg(long, default_value_t = 2)]
        games: usize,
        #[arg(long)]
        resume: bool,
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Verify a public beam-state value dataset and every shard checksum.
    ValidatePublicBeamValueDataset {
        #[arg(long)]
        dataset: PathBuf,
    },
    /// Verify the complete Rust-to-MLX batch inference boundary.
    ModelSmoke {
        #[arg(long)]
        run_dir: Option<PathBuf>,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, default_value = ".venv/bin/cascadia-mlx-serve")]
        server: PathBuf,
    },
    /// Benchmark a promoted or in-progress MLX value model.
    ModelBenchmark {
        #[arg(long, default_value_t = 4)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long)]
        run_dir: Option<PathBuf>,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, default_value = ".venv/bin/cascadia-mlx-serve")]
        server: PathBuf,
        /// Restrict model ranking to the top K exact immediate-score actions.
        #[arg(long)]
        prefilter_k: Option<usize>,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare an MLX value model with a baseline on identical game seeds.
    ModelCompare {
        #[arg(long, default_value_t = 20)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long, value_enum, default_value = "greedy")]
        baseline: StrategyArg,
        #[arg(long)]
        run_dir: Option<PathBuf>,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, default_value = ".venv/bin/cascadia-mlx-serve")]
        server: PathBuf,
        /// Restrict model ranking to the top K exact immediate-score actions.
        #[arg(long)]
        prefilter_k: Option<usize>,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Benchmark an MLX ranker over the confirmed K8+B8 candidate union.
    RankingModelBenchmark {
        #[arg(long, default_value_t = 4)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long)]
        run_dir: Option<PathBuf>,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, default_value = ".venv/bin/cascadia-mlx-ranking-serve")]
        server: PathBuf,
        #[arg(long, default_value_t = 8)]
        candidates: usize,
        #[arg(long, default_value_t = 8)]
        bear_candidates: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare an MLX ranker against K8 or its search teacher.
    RankingModelCompare {
        #[arg(long, default_value_t = 20)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long)]
        run_dir: Option<PathBuf>,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, default_value = ".venv/bin/cascadia-mlx-ranking-serve")]
        server: PathBuf,
        #[arg(long, value_enum, default_value_t = RankingBaselineArg::K8)]
        baseline: RankingBaselineArg,
        #[arg(long, default_value_t = 8)]
        candidates: usize,
        #[arg(long, default_value_t = 8)]
        bear_candidates: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Benchmark an MLX ranker over the matching H6 K+H candidate union.
    HabitatRankingModelBenchmark {
        #[arg(long, default_value_t = 4)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value = ".venv/bin/cascadia-mlx-ranking-serve")]
        server: PathBuf,
        #[arg(long, default_value_t = 8)]
        candidates: usize,
        #[arg(long, default_value_t = 6)]
        habitat_candidates: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare an MLX H6 apprentice with pattern-aware or the frozen H6 teacher.
    HabitatRankingModelCompare {
        #[arg(long, default_value_t = 10)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value = ".venv/bin/cascadia-mlx-ranking-serve")]
        server: PathBuf,
        #[arg(long, value_enum, default_value_t = HabitatRankingBaselineArg::PatternAware)]
        baseline: HabitatRankingBaselineArg,
        #[arg(long, default_value_t = 8)]
        candidates: usize,
        #[arg(long, default_value_t = 6)]
        habitat_candidates: usize,
        #[arg(long, default_value_t = 4)]
        determinizations: usize,
        #[arg(long, default_value_t = 4)]
        greedy_plies: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare an MLX terminal-label ranker with pattern-aware on identical games.
    PatternRankingModelCompare {
        #[arg(long, default_value_t = 10)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long, default_value = ".venv/bin/cascadia-mlx-ranking-serve")]
        server: PathBuf,
        #[arg(long, default_value_t = 8)]
        policy_candidates: usize,
        #[arg(long, default_value_t = 6)]
        policy_habitat_candidates: usize,
        #[arg(long, default_value_t = 8)]
        policy_bear_candidates: usize,
        #[arg(long, default_value_t = 4)]
        policy_market_draws: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare an explicit action-delta MLX ranker with pattern-aware play.
    ActionRankingModelCompare {
        #[arg(long, default_value_t = 10)]
        games: usize,
        #[arg(long, default_value_t = 25700)]
        first_seed: u64,
        #[arg(long)]
        run_dir: Option<PathBuf>,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, default_value = ".venv/bin/cascadia-mlx-action-ranking-serve")]
        server: PathBuf,
        #[arg(long, default_value_t = 8)]
        policy_candidates: usize,
        #[arg(long, default_value_t = 6)]
        policy_habitat_candidates: usize,
        #[arg(long, default_value_t = 8)]
        policy_bear_candidates: usize,
        #[arg(long, default_value_t = 4)]
        policy_market_draws: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare full-legal MLX imitation against promoted pattern-aware.
    FullActionImitationCompare {
        #[arg(long, default_value_t = 1)]
        games: usize,
        #[arg(long, default_value_t = 32700)]
        first_seed: u64,
        #[arg(long)]
        run_dir: Option<PathBuf>,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, default_value = ".venv/bin/cascadia-mlx-imitation-serve")]
        server: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Smoke-test the public beam-value model through the Rust/MLX boundary.
    PublicBeamValueModelSmoke {
        #[arg(long)]
        run_dir: Option<PathBuf>,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, default_value = ".venv/bin/cascadia-mlx-public-beam-value-serve")]
        server: PathBuf,
    },
    /// Compare the qualified public beam-value policy with promoted strong.
    PublicBeamValueModelCompare {
        #[arg(long, default_value_t = 10)]
        games: usize,
        #[arg(long, default_value_t = 31_000)]
        first_seed: u64,
        #[arg(long)]
        run_dir: Option<PathBuf>,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, default_value = ".venv/bin/cascadia-mlx-public-beam-value-serve")]
        server: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare two frozen MLX H6 apprentices on identical games.
    HabitatRankingModelH2h {
        #[arg(long, default_value_t = 10)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long)]
        baseline_model_dir: PathBuf,
        #[arg(long)]
        treatment_model_dir: PathBuf,
        #[arg(long, default_value = ".venv/bin/cascadia-mlx-ranking-serve")]
        server: PathBuf,
        #[arg(long, default_value_t = 8)]
        candidates: usize,
        #[arg(long, default_value_t = 6)]
        habitat_candidates: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare MLX-prefiltered rollout search with immediate-score K8.
    RankingPrefilterCompare {
        #[arg(long, default_value_t = 10)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long)]
        run_dir: Option<PathBuf>,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, default_value = ".venv/bin/cascadia-mlx-ranking-serve")]
        server: PathBuf,
        #[arg(long, default_value_t = 8)]
        candidates: usize,
        #[arg(long, default_value_t = 8)]
        bear_candidates: usize,
        #[arg(long, default_value_t = 0)]
        immediate_anchors: usize,
        #[arg(long, default_value_t = 8)]
        prefilter_candidates: usize,
        #[arg(long, default_value_t = 4)]
        determinizations: usize,
        #[arg(long, default_value_t = 4)]
        greedy_plies: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare H6 with an MLX-prefiltered wider habitat candidate frontier.
    RankingHabitatPrefilterCompare {
        #[arg(long, default_value_t = 10)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long)]
        run_dir: Option<PathBuf>,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, default_value = ".venv/bin/cascadia-mlx-ranking-serve")]
        server: PathBuf,
        #[arg(long, default_value_t = 8)]
        baseline_candidates: usize,
        #[arg(long, default_value_t = 6)]
        baseline_habitat_candidates: usize,
        #[arg(long, default_value_t = 16)]
        candidates: usize,
        #[arg(long, default_value_t = 8)]
        habitat_candidates: usize,
        #[arg(long, default_value_t = 8)]
        immediate_anchors: usize,
        #[arg(long, default_value_t = 14)]
        prefilter_candidates: usize,
        #[arg(long, default_value_t = 4)]
        determinizations: usize,
        #[arg(long, default_value_t = 4)]
        greedy_plies: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare H6 greedy rollouts with batched MLX H6-policy rollouts.
    RankingHabitatRolloutCompare {
        #[arg(long, default_value_t = 10)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long)]
        run_dir: Option<PathBuf>,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, default_value = ".venv/bin/cascadia-mlx-ranking-serve")]
        server: PathBuf,
        #[arg(long, default_value_t = 8)]
        candidates: usize,
        #[arg(long, default_value_t = 6)]
        habitat_candidates: usize,
        #[arg(long, default_value_t = 4)]
        determinizations: usize,
        #[arg(long, default_value_t = 4)]
        rollout_plies: usize,
        #[arg(long, default_value_t = 8)]
        rollout_candidates: usize,
        #[arg(long, default_value_t = 6)]
        rollout_habitat_candidates: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare H6 with an MLX policy only on the acting seat's next rollout turn.
    RankingSelfRolloutCompare {
        #[arg(long, default_value_t = 10)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long)]
        run_dir: Option<PathBuf>,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, default_value = ".venv/bin/cascadia-mlx-ranking-serve")]
        server: PathBuf,
        #[arg(long, default_value_t = 8)]
        candidates: usize,
        #[arg(long, default_value_t = 6)]
        habitat_candidates: usize,
        #[arg(long, default_value_t = 4)]
        determinizations: usize,
        #[arg(long, default_value_t = 4)]
        rollout_plies: usize,
        #[arg(long, default_value_t = 8)]
        policy_candidates: usize,
        #[arg(long, default_value_t = 6)]
        policy_habitat_candidates: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare H6 search with an MLX final-score value model at rollout leaves.
    ValueLeafCompare {
        #[arg(long, default_value_t = 10)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long)]
        run_dir: Option<PathBuf>,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, default_value = ".venv/bin/cascadia-mlx-serve")]
        server: PathBuf,
        #[arg(long, default_value_t = 8)]
        candidates: usize,
        #[arg(long, default_value_t = 6)]
        habitat_candidates: usize,
        #[arg(long, default_value_t = 4)]
        determinizations: usize,
        #[arg(long, default_value_t = 4)]
        greedy_plies: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Benchmark fair hidden-state lookahead with greedy rollout policies.
    LookaheadBenchmark {
        #[arg(long, default_value_t = 1)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long, default_value_t = 4)]
        candidates: usize,
        #[arg(long, default_value_t = 4)]
        determinizations: usize,
        #[arg(long, default_value_t = 4)]
        greedy_plies: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare fair hidden-state lookahead with a baseline on identical seeds.
    LookaheadCompare {
        #[arg(long, default_value_t = 10)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long, value_enum, default_value = "greedy")]
        baseline: StrategyArg,
        #[arg(long, default_value_t = 4)]
        candidates: usize,
        #[arg(long, default_value_t = 4)]
        determinizations: usize,
        #[arg(long, default_value_t = 4)]
        greedy_plies: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare two fair hidden-state lookahead configurations on identical seeds.
    LookaheadAblate {
        #[arg(long, default_value_t = 10)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long, default_value_t = 4)]
        baseline_candidates: usize,
        #[arg(long, default_value_t = 4)]
        baseline_determinizations: usize,
        #[arg(long, default_value_t = 4)]
        baseline_greedy_plies: usize,
        #[arg(long, default_value_t = 4)]
        treatment_candidates: usize,
        #[arg(long, default_value_t = 4)]
        treatment_determinizations: usize,
        #[arg(long, default_value_t = 4)]
        treatment_greedy_plies: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Measure top-K candidate recall against a wider search on baseline trajectories.
    LookaheadRecall {
        #[arg(long, default_value_t = 5)]
        games: usize,
        #[arg(long, default_value_t = 20600)]
        first_seed: u64,
        #[arg(long, default_value_t = 4)]
        retained_candidates: usize,
        #[arg(long, default_value_t = 8)]
        expanded_candidates: usize,
        #[arg(long, default_value_t = 4)]
        determinizations: usize,
        #[arg(long, default_value_t = 4)]
        greedy_plies: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare promoted lookahead with fair one-wipe Nature Token planning.
    NatureWipeCompare {
        #[arg(long, default_value_t = 5)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long, default_value_t = 8)]
        candidates: usize,
        #[arg(long, default_value_t = 4)]
        determinizations: usize,
        #[arg(long, default_value_t = 4)]
        greedy_plies: usize,
        #[arg(long, default_value_t = 4)]
        prelude_candidates: usize,
        #[arg(long, default_value_t = 2)]
        prelude_determinizations: usize,
        #[arg(long, default_value_t = 4)]
        prelude_greedy_plies: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare promoted lookahead with a Bear-specific candidate union.
    BearCandidateCompare {
        #[arg(long, default_value_t = 10)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        /// Immediate-score candidate count used by the baseline.
        #[arg(long)]
        baseline_candidates: Option<usize>,
        #[arg(long, default_value_t = 8)]
        candidates: usize,
        #[arg(long, default_value_t = 8)]
        bear_candidates: usize,
        #[arg(long, default_value_t = 4)]
        determinizations: usize,
        #[arg(long, default_value_t = 4)]
        greedy_plies: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare promoted lookahead with a habitat-cohesion candidate union.
    HabitatCandidateCompare {
        #[arg(long, default_value_t = 10)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        /// Immediate-score candidate count used by the baseline.
        #[arg(long)]
        baseline_candidates: Option<usize>,
        #[arg(long, default_value_t = 8)]
        candidates: usize,
        #[arg(long, default_value_t = 8)]
        habitat_candidates: usize,
        #[arg(long, default_value_t = 4)]
        determinizations: usize,
        #[arg(long, default_value_t = 4)]
        greedy_plies: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare H6 with a combined habitat- and Bear-aware candidate frontier.
    BearHabitatCandidateCompare {
        #[arg(long, default_value_t = 10)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long, default_value_t = 8)]
        candidates: usize,
        #[arg(long, default_value_t = 6)]
        habitat_candidates: usize,
        #[arg(long, default_value_t = 8)]
        bear_candidates: usize,
        #[arg(long, default_value_t = 4)]
        determinizations: usize,
        #[arg(long, default_value_t = 4)]
        greedy_plies: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare two habitat-cohesion lookahead configurations on identical seeds.
    HabitatCandidateAblate {
        #[arg(long, default_value_t = 10)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long, default_value_t = 8)]
        baseline_candidates: usize,
        #[arg(long, default_value_t = 6)]
        baseline_habitat_candidates: usize,
        #[arg(long, default_value_t = 4)]
        baseline_determinizations: usize,
        #[arg(long, default_value_t = 4)]
        baseline_greedy_plies: usize,
        #[arg(long, default_value_t = 8)]
        treatment_candidates: usize,
        #[arg(long, default_value_t = 6)]
        treatment_habitat_candidates: usize,
        #[arg(long, default_value_t = 4)]
        treatment_determinizations: usize,
        #[arg(long, default_value_t = 8)]
        treatment_greedy_plies: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare H6 with the same root frontier using pattern-aware rollout plies.
    PatternBlueprintCompare {
        #[arg(long, default_value_t = 10)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long, default_value_t = 8)]
        candidates: usize,
        #[arg(long, default_value_t = 6)]
        habitat_candidates: usize,
        #[arg(long, default_value_t = 4)]
        determinizations: usize,
        #[arg(long, default_value_t = 4)]
        rollout_plies: usize,
        #[arg(long, default_value_t = 8)]
        policy_candidates: usize,
        #[arg(long, default_value_t = 6)]
        policy_habitat_candidates: usize,
        #[arg(long, default_value_t = 8)]
        policy_bear_candidates: usize,
        #[arg(long, default_value_t = 4)]
        policy_market_draws: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Measure the K8+H6+B8 frontier with a diagnostic true-hidden-state oracle.
    PerfectInformationOracleCompare {
        #[arg(long, default_value_t = 1)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long, default_value_t = 8)]
        policy_candidates: usize,
        #[arg(long, default_value_t = 6)]
        policy_habitat_candidates: usize,
        #[arg(long, default_value_t = 8)]
        policy_bear_candidates: usize,
        #[arg(long, default_value_t = 4)]
        policy_market_draws: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare exact-hidden-state base and wildlife-diverse focal frontiers.
    PerfectInformationOracleFrontierCompare {
        #[arg(long, default_value_t = 1)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long, default_value_t = 8)]
        policy_candidates: usize,
        #[arg(long, default_value_t = 6)]
        policy_habitat_candidates: usize,
        #[arg(long, default_value_t = 8)]
        policy_bear_candidates: usize,
        #[arg(long, default_value_t = 2)]
        wildlife_candidates: usize,
        #[arg(long, default_value_t = 4)]
        policy_market_draws: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare exact one-step W2 with final-turn exact focal beam planning.
    PerfectInformationFocalBeamCompare {
        #[arg(long, default_value_t = 1)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long, default_value_t = 8)]
        policy_candidates: usize,
        #[arg(long, default_value_t = 6)]
        policy_habitat_candidates: usize,
        #[arg(long, default_value_t = 8)]
        policy_bear_candidates: usize,
        #[arg(long, default_value_t = 2)]
        wildlife_candidates: usize,
        #[arg(long, default_value_t = 4)]
        policy_market_draws: usize,
        #[arg(long, default_value_t = 16)]
        beam_width: usize,
        #[arg(long, default_value_t = 5)]
        terminal_turns: u16,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare W2 and W4 wildlife frontiers under the same exact focal beam.
    PerfectInformationFocalFrontierCompare {
        #[arg(long, default_value_t = 1)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long, default_value_t = 8)]
        policy_candidates: usize,
        #[arg(long, default_value_t = 6)]
        policy_habitat_candidates: usize,
        #[arg(long, default_value_t = 8)]
        policy_bear_candidates: usize,
        #[arg(long, default_value_t = 2)]
        baseline_wildlife_candidates: usize,
        #[arg(long, default_value_t = 4)]
        treatment_wildlife_candidates: usize,
        #[arg(long, default_value_t = 4)]
        policy_market_draws: usize,
        #[arg(long, default_value_t = 16)]
        beam_width: usize,
        #[arg(long, default_value_t = 5)]
        terminal_turns: u16,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare width-16 and width-32 exact W2 focal beams.
    PerfectInformationBeamCapacityCompare {
        #[arg(long, default_value_t = 1)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long, default_value_t = 8)]
        policy_candidates: usize,
        #[arg(long, default_value_t = 6)]
        policy_habitat_candidates: usize,
        #[arg(long, default_value_t = 8)]
        policy_bear_candidates: usize,
        #[arg(long, default_value_t = 2)]
        wildlife_candidates: usize,
        #[arg(long, default_value_t = 4)]
        policy_market_draws: usize,
        #[arg(long, default_value_t = 16)]
        baseline_beam_width: usize,
        #[arg(long, default_value_t = 32)]
        treatment_beam_width: usize,
        #[arg(long, default_value_t = 5)]
        terminal_turns: u16,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare W2 with root-only W4 under W2 future focal layers.
    PerfectInformationRootDiverseBeamCompare {
        #[arg(long, default_value_t = 1)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long, default_value_t = 8)]
        policy_candidates: usize,
        #[arg(long, default_value_t = 6)]
        policy_habitat_candidates: usize,
        #[arg(long, default_value_t = 8)]
        policy_bear_candidates: usize,
        #[arg(long, default_value_t = 2)]
        baseline_root_wildlife_candidates: usize,
        #[arg(long, default_value_t = 4)]
        treatment_root_wildlife_candidates: usize,
        #[arg(long, default_value_t = 2)]
        future_wildlife_candidates: usize,
        #[arg(long, default_value_t = 4)]
        policy_market_draws: usize,
        #[arg(long, default_value_t = 16)]
        beam_width: usize,
        #[arg(long, default_value_t = 5)]
        terminal_turns: u16,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare scalar and portfolio-preserving exact focal beam retention.
    PerfectInformationPortfolioBeamCompare {
        #[arg(long, default_value_t = 1)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long, default_value_t = 8)]
        policy_candidates: usize,
        #[arg(long, default_value_t = 6)]
        policy_habitat_candidates: usize,
        #[arg(long, default_value_t = 8)]
        policy_bear_candidates: usize,
        #[arg(long, default_value_t = 2)]
        wildlife_candidates: usize,
        #[arg(long, default_value_t = 4)]
        policy_market_draws: usize,
        #[arg(long, default_value_t = 16)]
        beam_width: usize,
        #[arg(long, default_value_t = 5)]
        terminal_turns: u16,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare promoted strong with a public redetermined focal-beam teacher.
    PublicFocalBeamCompare {
        #[arg(long, default_value_t = 1)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long, default_value_t = 5)]
        terminal_turns: u16,
        #[arg(long, default_value_t = 4)]
        determinizations: usize,
        #[arg(long, default_value_t = 4)]
        beam_width: usize,
        #[arg(long, default_value_t = 8)]
        policy_candidates: usize,
        #[arg(long, default_value_t = 6)]
        policy_habitat_candidates: usize,
        #[arg(long, default_value_t = 8)]
        policy_bear_candidates: usize,
        #[arg(long, default_value_t = 2)]
        wildlife_candidates: usize,
        #[arg(long, default_value_t = 4)]
        policy_market_draws: usize,
        #[arg(long, default_value_t = false)]
        sequential: bool,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare promoted strong with public open-loop focal tree search.
    PublicFocalTreeCompare {
        #[arg(long, default_value_t = 1)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long, default_value_t = 5)]
        terminal_turns: u16,
        #[arg(long, default_value_t = 128)]
        simulations: usize,
        #[arg(long, default_value_t = 16)]
        root_candidates: usize,
        #[arg(long, default_value_t = 2000)]
        exploration_milli: u16,
        #[arg(long, default_value_t = 8)]
        policy_candidates: usize,
        #[arg(long, default_value_t = 6)]
        policy_habitat_candidates: usize,
        #[arg(long, default_value_t = 8)]
        policy_bear_candidates: usize,
        #[arg(long, default_value_t = 2)]
        wildlife_candidates: usize,
        #[arg(long, default_value_t = 4)]
        policy_market_draws: usize,
        #[arg(long, default_value_t = false)]
        sequential: bool,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare pattern-aware with full-game one-step policy improvement.
    TerminalPolicyImprovementCompare {
        #[arg(long, default_value_t = 1)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long, default_value_t = 2)]
        determinizations: usize,
        #[arg(long, default_value_t = 8)]
        policy_candidates: usize,
        #[arg(long, default_value_t = 6)]
        policy_habitat_candidates: usize,
        #[arg(long, default_value_t = 8)]
        policy_bear_candidates: usize,
        #[arg(long, default_value_t = 4)]
        policy_market_draws: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare pattern-aware with R8 terminal search on only the final personal turns.
    LateTerminalPolicyImprovementCompare {
        #[arg(long, default_value_t = 1)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long, default_value_t = 4)]
        terminal_turns: u16,
        #[arg(long, default_value_t = 8)]
        determinizations: usize,
        #[arg(long, default_value_t = 8)]
        policy_candidates: usize,
        #[arg(long, default_value_t = 6)]
        policy_habitat_candidates: usize,
        #[arg(long, default_value_t = 8)]
        policy_bear_candidates: usize,
        #[arg(long, default_value_t = 4)]
        policy_market_draws: usize,
        #[arg(long, default_value_t = false)]
        sequential: bool,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare pattern-aware with a wildlife-diverse final-turn R8 frontier.
    LateWildlifeDiversePolicyImprovementCompare {
        #[arg(long, default_value_t = 1)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long, default_value_t = 5)]
        terminal_turns: u16,
        #[arg(long, default_value_t = 8)]
        determinizations: usize,
        #[arg(long, default_value_t = 8)]
        policy_candidates: usize,
        #[arg(long, default_value_t = 6)]
        policy_habitat_candidates: usize,
        #[arg(long, default_value_t = 8)]
        policy_bear_candidates: usize,
        #[arg(long, default_value_t = 2)]
        wildlife_candidates: usize,
        #[arg(long, default_value_t = 4)]
        policy_market_draws: usize,
        #[arg(long, default_value_t = false)]
        sequential: bool,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare pattern-aware with confidence-gated final-turn R8 improvement.
    LateConservativePolicyImprovementCompare {
        #[arg(long, default_value_t = 1)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long, default_value_t = 5)]
        terminal_turns: u16,
        #[arg(long, default_value_t = 8)]
        policy_candidates: usize,
        #[arg(long, default_value_t = 6)]
        policy_habitat_candidates: usize,
        #[arg(long, default_value_t = 8)]
        policy_bear_candidates: usize,
        #[arg(long, default_value_t = 2)]
        wildlife_candidates: usize,
        #[arg(long, default_value_t = 4)]
        policy_market_draws: usize,
        #[arg(long, default_value_t = false)]
        sequential: bool,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare pattern-aware with confidence-gated R8 on the original frontier.
    LateConservativeBasePolicyImprovementCompare {
        #[arg(long, default_value_t = 1)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long, default_value_t = 5)]
        terminal_turns: u16,
        #[arg(long, default_value_t = 8)]
        policy_candidates: usize,
        #[arg(long, default_value_t = 6)]
        policy_habitat_candidates: usize,
        #[arg(long, default_value_t = 8)]
        policy_bear_candidates: usize,
        #[arg(long, default_value_t = 4)]
        policy_market_draws: usize,
        #[arg(long, default_value_t = false)]
        sequential: bool,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare promoted strong with confidence-gated focused-species coverage.
    LateConservativeWildlifeFocusedPolicyImprovementCompare {
        #[arg(long, default_value_t = 1)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long, default_value_t = 5)]
        terminal_turns: u16,
        #[arg(long, default_value_t = 8)]
        determinizations: usize,
        #[arg(long, default_value_t = 8)]
        policy_candidates: usize,
        #[arg(long, default_value_t = 6)]
        policy_habitat_candidates: usize,
        #[arg(long, default_value_t = 8)]
        policy_bear_candidates: usize,
        #[arg(long, value_enum)]
        wildlife: WildlifeArg,
        #[arg(long, default_value_t = 2)]
        wildlife_candidates: usize,
        #[arg(long, default_value_t = 4)]
        policy_market_draws: usize,
        #[arg(long, default_value_t = false)]
        sequential: bool,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare conservative final-five policies at two supported sample counts.
    ConservativeSampleCountCompare {
        #[arg(long, default_value_t = 1)]
        games: usize,
        #[arg(long, default_value_t = 0)]
        first_seed: u64,
        #[arg(long, default_value_t = 5)]
        terminal_turns: u16,
        #[arg(long, default_value_t = 8)]
        baseline_determinizations: usize,
        #[arg(long, default_value_t = 32)]
        treatment_determinizations: usize,
        #[arg(long, default_value_t = 8)]
        policy_candidates: usize,
        #[arg(long, default_value_t = 6)]
        policy_habitat_candidates: usize,
        #[arg(long, default_value_t = 8)]
        policy_bear_candidates: usize,
        #[arg(long, default_value_t = 4)]
        policy_market_draws: usize,
        #[arg(long, default_value_t = false)]
        sequential: bool,
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

#[derive(Debug, Clone, Copy, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum StrategyArg {
    Random,
    Greedy,
    PatternAware,
    PatternCommitment,
    PatternCompetition,
    PatternPortfolio,
}

#[derive(Debug, Clone, Copy, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum WildlifeArg {
    Bear,
    Elk,
    Salmon,
    Hawk,
    Fox,
}

#[derive(Debug, Clone, Copy, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum SplitArg {
    Train,
    Validation,
    Test,
    Final,
}

#[derive(Debug, Clone, Copy, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum R2MapDatasetModeArg {
    Train,
    Validation,
    FixedPanel,
}

#[derive(Debug, Clone, Copy, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum RankingBaselineArg {
    K8,
    BearTeacher,
}

#[derive(Debug, Clone, Copy, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum RankingTeacherArg {
    Bear,
    Habitat,
}

#[derive(Debug, Clone, Copy, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum HabitatRankingBaselineArg {
    PatternAware,
    HabitatTeacher,
}

impl From<StrategyArg> for StrategyKind {
    fn from(value: StrategyArg) -> Self {
        match value {
            StrategyArg::Random => Self::Random,
            StrategyArg::Greedy => Self::Greedy,
            StrategyArg::PatternAware => Self::PatternAware,
            StrategyArg::PatternCommitment => Self::PatternCommitment,
            StrategyArg::PatternCompetition => Self::PatternCompetition,
            StrategyArg::PatternPortfolio => Self::PatternPortfolio,
        }
    }
}

impl From<WildlifeArg> for Wildlife {
    fn from(value: WildlifeArg) -> Self {
        match value {
            WildlifeArg::Bear => Self::Bear,
            WildlifeArg::Elk => Self::Elk,
            WildlifeArg::Salmon => Self::Salmon,
            WildlifeArg::Hawk => Self::Hawk,
            WildlifeArg::Fox => Self::Fox,
        }
    }
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
