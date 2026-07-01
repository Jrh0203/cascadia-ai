mod basic_data_commands;
mod candidate_recall;
mod cli;
mod counterfactual;
mod counterfactual_advantage;
mod lookahead_commands;
mod oracle_commands;
mod pattern_potential;
mod policy_commands;
mod public_beam_data_commands;
mod public_beam_probe;
mod r2_map_commands;
mod ranking_data;
mod ranking_data_commands;
mod ranking_model_commands;
mod report;
mod value_model_commands;

use std::time::Instant;

use cascadia_eval::{
    BenchmarkConfig, ComparisonConfig, run_benchmark, run_comparison,
    summarize_paired_match_results,
};
use cascadia_game::{GameConfig, GameSeed};
use cascadia_sim::{
    MatchConfig, PATTERN_AWARE_STRATEGY_ID, PatternAwareConfig, PatternPotentialConfig,
    PatternPotentialStrategy, StrategyKind, play_match,
};
use clap::Parser;
use rayon::prelude::*;

use crate::cli::{Cli, Command};
use crate::pattern_potential::run_pattern_potential_sweep;
use crate::report::{ReportContext, write_report};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let report_context = ReportContext::capture(serde_json::to_value(&cli.command)?)?;
    match cli.command {
        Command::Benchmark {
            games,
            first_seed,
            strategy,
            sequential,
            output,
        } => {
            let report = run_benchmark(BenchmarkConfig {
                games,
                first_seed,
                strategy: strategy.into(),
                parallel: !sequential,
            })?;
            let json = report_context.to_json(&report)?;
            if let Some(path) = output {
                write_report(&path, &json, &report.to_markdown())?;
            }
            println!("{json}");
        }
        Command::Compare {
            games,
            first_seed,
            baseline,
            treatment,
            sequential,
            output,
        } => {
            let report = run_comparison(ComparisonConfig {
                games,
                first_seed,
                baseline: baseline.into(),
                treatment: treatment.into(),
                parallel: !sequential,
            })?;
            let json = report_context.to_json(&report)?;
            if let Some(path) = output {
                write_report(&path, &json, &report.to_markdown())?;
            }
            println!("{json}");
        }
        Command::PatternPotentialSweep {
            games,
            first_seed,
            output,
        } => {
            if games == 0 {
                return Err("pattern-potential-sweep requires at least one game".into());
            }
            let report = run_pattern_potential_sweep(games, first_seed)?;
            let json = report_context.to_json(&report)?;
            if let Some(path) = output {
                write_report(&path, &json, &report.to_markdown())?;
            }
            println!("{json}");
        }
        Command::PatternPotentialCompare {
            games,
            first_seed,
            opportunity_weight,
            habitat_weight,
            bear_weight,
            sequential,
            output,
        } => {
            if games == 0 {
                return Err("pattern-potential-compare requires at least one game".into());
            }
            let treatment = PatternPotentialStrategy::new(PatternPotentialConfig::from_weights(
                PatternAwareConfig::default(),
                opportunity_weight,
                habitat_weight,
                bear_weight,
            )?)?;
            let game_config = GameConfig::research_aaaaa(4)?;
            let run_pair = |index: usize| {
                let numeric_seed = first_seed + index as u64;
                let seed = GameSeed::from_u64(numeric_seed);
                let baseline_result = play_match(&MatchConfig::symmetric(
                    game_config,
                    seed,
                    StrategyKind::PatternAware,
                ))
                .map_err(|error| error.to_string())?;
                let treatment_result = treatment
                    .play_match(game_config, seed)
                    .map_err(|error| error.to_string())?;
                Ok::<_, String>((numeric_seed, baseline_result, treatment_result))
            };
            let started = Instant::now();
            let pairs = if sequential {
                (0..games).map(run_pair).collect::<Result<Vec<_>, _>>()
            } else {
                (0..games)
                    .into_par_iter()
                    .map(run_pair)
                    .collect::<Result<Vec<_>, _>>()
            }
            .map_err(std::io::Error::other)?;
            let report = summarize_paired_match_results(
                PATTERN_AWARE_STRATEGY_ID,
                treatment.strategy_id(),
                first_seed,
                &pairs,
                started.elapsed().as_secs_f64(),
            );
            let json = report_context.to_json(&report)?;
            if let Some(path) = output {
                write_report(&path, &json, &report.to_markdown())?;
            }
            println!("{json}");
        }
        command @ (Command::Collect { .. }
        | Command::CollectSearch { .. }
        | Command::CollectScoreToGo { .. }
        | Command::CollectCounterfactualValue { .. }
        | Command::CollectCounterfactualAdvantage { .. }
        | Command::ValidateDataset { .. }
        | Command::ValidateScoreToGoDataset { .. }
        | Command::ValidateCounterfactualValueDataset { .. }
        | Command::ValidateCounterfactualAdvantageDataset { .. }
        | Command::AuditCounterfactualValueDataset { .. }
        | Command::AuditCounterfactualAdvantageDataset { .. }) => {
            basic_data_commands::run(command)?;
        }
        command @ (Command::CollectR2MapBootstrap { .. }
        | Command::CollectR2MapIteration { .. }
        | Command::ValidateR2MapCollector { .. }
        | Command::InspectR2MapIndexMetadata { .. }
        | Command::ExportR2MapDataset { .. }
        | Command::ServeR2MapPackedBatches { .. }
        | Command::PrepareR2MapServingBundle { .. }
        | Command::InitR2MapLongitudinalOpenPanel { .. }
        | Command::InitR2MapLongitudinalCampaign { .. }
        | Command::InitR2MapFocalCampaign { .. }
        | Command::RunR2MapLongitudinalWorkItem { .. }
        | Command::AggregateR2MapLongitudinal { .. }
        | Command::RunR2MapFocalWorkItem { .. }
        | Command::AggregateR2MapFocal { .. }) => {
            r2_map_commands::run(command)?;
        }
        command @ (Command::CollectRanking { .. }
        | Command::CollectTerminalRanking { .. }
        | Command::CollectConservativeAdvantage { .. }
        | Command::CollectRankingIteration { .. }
        | Command::ValidateRankingDataset { .. }
        | Command::EnrichActionRanking { .. }
        | Command::ValidateActionRankingDataset { .. }
        | Command::ValidateConservativeAdvantageDataset { .. }) => {
            ranking_data_commands::run(command)?;
        }
        command @ (Command::CollectPublicBeamValue { .. }
        | Command::PublicBeamValueProbe { .. }
        | Command::ValidatePublicBeamValueDataset { .. }) => {
            public_beam_data_commands::run(command, &report_context)?;
        }
        command @ (Command::ModelSmoke { .. }
        | Command::ModelBenchmark { .. }
        | Command::ModelCompare { .. }
        | Command::PublicBeamValueModelSmoke { .. }
        | Command::PublicBeamValueModelCompare { .. }
        | Command::ValueLeafCompare { .. }) => {
            value_model_commands::run(command, &report_context)?;
        }
        command @ (Command::RankingModelBenchmark { .. }
        | Command::RankingModelCompare { .. }
        | Command::HabitatRankingModelBenchmark { .. }
        | Command::HabitatRankingModelCompare { .. }
        | Command::PatternRankingModelCompare { .. }
        | Command::ActionRankingModelCompare { .. }
        | Command::FullActionImitationCompare { .. }
        | Command::HabitatRankingModelH2h { .. }
        | Command::RankingPrefilterCompare { .. }
        | Command::RankingHabitatPrefilterCompare { .. }
        | Command::RankingHabitatRolloutCompare { .. }
        | Command::RankingSelfRolloutCompare { .. }) => {
            ranking_model_commands::run(command, &report_context)?;
        }
        command @ (Command::LookaheadBenchmark { .. }
        | Command::LookaheadCompare { .. }
        | Command::LookaheadAblate { .. }
        | Command::LookaheadRecall { .. }
        | Command::NatureWipeCompare { .. }
        | Command::BearCandidateCompare { .. }
        | Command::HabitatCandidateCompare { .. }
        | Command::BearHabitatCandidateCompare { .. }
        | Command::HabitatCandidateAblate { .. }
        | Command::PatternBlueprintCompare { .. }) => {
            lookahead_commands::run(command, &report_context)?;
        }
        command @ (Command::PerfectInformationOracleCompare { .. }
        | Command::PerfectInformationOracleFrontierCompare { .. }
        | Command::PerfectInformationFocalBeamCompare { .. }
        | Command::PerfectInformationFocalFrontierCompare { .. }
        | Command::PerfectInformationBeamCapacityCompare { .. }
        | Command::PerfectInformationRootDiverseBeamCompare { .. }
        | Command::PerfectInformationPortfolioBeamCompare { .. }
        | Command::PublicFocalBeamCompare { .. }
        | Command::PublicFocalTreeCompare { .. }) => {
            oracle_commands::run(command, &report_context)?;
        }
        command @ (Command::TerminalPolicyImprovementCompare { .. }
        | Command::LateTerminalPolicyImprovementCompare { .. }
        | Command::LateWildlifeDiversePolicyImprovementCompare { .. }
        | Command::LateConservativePolicyImprovementCompare { .. }
        | Command::LateConservativeBasePolicyImprovementCompare { .. }
        | Command::LateConservativeWildlifeFocusedPolicyImprovementCompare { .. }
        | Command::ConservativeSampleCountCompare { .. }) => {
            policy_commands::run(command, &report_context)?;
        }
    }
    Ok(())
}
