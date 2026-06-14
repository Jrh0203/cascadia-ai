//! Canonical benchmark runner for Cascadia v2.

use std::time::Instant;

use cascadia_game::{GameConfig, GameSeed, ScoreBreakdown};
use cascadia_sim::{MatchConfig, MatchResult, SimulationError, StrategyKind, play_match};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const PROTOCOL_ID: &str = "cascadia-aaaaa-4p-base-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub games: usize,
    pub first_seed: u64,
    pub strategy: StrategyKind,
    pub parallel: bool,
}

impl BenchmarkConfig {
    pub const fn smoke(strategy: StrategyKind) -> Self {
        Self {
            games: 4,
            first_seed: 0,
            strategy,
            parallel: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub protocol_id: String,
    pub strategy_id: String,
    pub games: usize,
    pub seat_games: usize,
    pub first_seed: u64,
    pub mean_score: f64,
    pub game_mean_stddev: f64,
    pub seat_score_stddev: f64,
    pub standard_error: f64,
    pub confidence_95: [f64; 2],
    pub percentiles: Percentiles,
    pub min_score: f64,
    pub max_score: f64,
    pub mean_breakdown: MeanBreakdown,
    #[serde(default)]
    pub decision_latency: DecisionLatency,
    pub elapsed_seconds: f64,
    pub games_per_second: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Percentiles {
    pub p10: f64,
    pub p50: f64,
    pub p90: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeanBreakdown {
    pub habitat: [f64; 5],
    pub wildlife: [f64; 5],
    pub nature_tokens: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DecisionLatency {
    pub decisions: usize,
    pub mean_milliseconds: f64,
    pub p50_milliseconds: f64,
    pub p90_milliseconds: f64,
    pub p99_milliseconds: f64,
    pub max_milliseconds: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StrategyStatistics {
    pub mean_score: f64,
    pub game_mean_stddev: f64,
    pub seat_score_stddev: f64,
    pub standard_error: f64,
    pub confidence_95: [f64; 2],
    pub percentiles: Percentiles,
    pub min_score: f64,
    pub max_score: f64,
    pub mean_breakdown: MeanBreakdown,
    #[serde(default)]
    pub decision_latency: DecisionLatency,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComparisonConfig {
    pub games: usize,
    pub first_seed: u64,
    pub baseline: StrategyKind,
    pub treatment: StrategyKind,
    pub parallel: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub protocol_id: String,
    pub baseline_id: String,
    pub treatment_id: String,
    pub games: usize,
    pub seat_games_per_strategy: usize,
    pub first_seed: u64,
    pub baseline_mean: f64,
    pub treatment_mean: f64,
    pub baseline_statistics: StrategyStatistics,
    pub treatment_statistics: StrategyStatistics,
    pub mean_paired_delta: f64,
    pub paired_delta_stddev: f64,
    pub standard_error: f64,
    pub confidence_95: [f64; 2],
    pub game_wins: usize,
    pub game_ties: usize,
    pub game_losses: usize,
    pub baseline_breakdown: MeanBreakdown,
    pub treatment_breakdown: MeanBreakdown,
    pub mean_breakdown_delta: MeanBreakdown,
    #[serde(default)]
    pub baseline_decision_latency: DecisionLatency,
    #[serde(default)]
    pub treatment_decision_latency: DecisionLatency,
    pub baseline_elapsed_seconds: f64,
    pub treatment_elapsed_seconds: f64,
    pub baseline_seconds_per_game: f64,
    pub treatment_seconds_per_game: f64,
    pub elapsed_seconds: f64,
    pub pairs: Vec<PairedGame>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedGame {
    pub seed: u64,
    pub baseline_mean: f64,
    pub treatment_mean: f64,
    pub delta: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationBlock {
    pub scores: Vec<ScoreBreakdown>,
    pub decision_seconds: Vec<f64>,
    pub elapsed_seconds: f64,
}

impl From<&MatchResult> for EvaluationBlock {
    fn from(result: &MatchResult) -> Self {
        Self {
            scores: result.scores.clone(),
            decision_seconds: result.decision_seconds.clone(),
            elapsed_seconds: result.elapsed_seconds,
        }
    }
}

impl BenchmarkReport {
    pub fn to_markdown(&self) -> String {
        let mut output = format!(
            "# Cascadia Benchmark\n\n\
             - Protocol: `{}`\n\
             - Strategy: `{}`\n\
             - Games: {} ({} seat scores)\n\
             - Seeds: {} through {}\n\
             - Mean base score: **{:.3}**\n\
             - 95% CI: [{:.3}, {:.3}]\n\
             - Game-mean SD / SE: {:.3} / {:.3}\n\
             - Seat-score SD: {:.3}\n\
             - P10 / P50 / P90: {:.1} / {:.1} / {:.1}\n\
             - Min / max: {:.1} / {:.1}\n\
             - Decision latency mean / P50 / P90 / P99 / max: {:.2} / {:.2} / {:.2} / {:.2} / {:.2} ms ({} decisions)\n\
             - Runtime: {:.3}s ({:.3} games/s)\n\n\
             ## Mean Breakdown\n\n\
             | Category | Mountain | Forest | Prairie | Wetland | River |\n\
             |---|---:|---:|---:|---:|---:|\n\
             | Habitat | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |\n\
             | Wildlife | Bear {:.3} | Elk {:.3} | Salmon {:.3} | Hawk {:.3} | Fox {:.3} |\n\n\
             Mean remaining nature tokens: {:.3}\n",
            self.protocol_id,
            self.strategy_id,
            self.games,
            self.seat_games,
            self.first_seed,
            self.first_seed + self.games.saturating_sub(1) as u64,
            self.mean_score,
            self.confidence_95[0],
            self.confidence_95[1],
            self.game_mean_stddev,
            self.standard_error,
            self.seat_score_stddev,
            self.percentiles.p10,
            self.percentiles.p50,
            self.percentiles.p90,
            self.min_score,
            self.max_score,
            self.decision_latency.mean_milliseconds,
            self.decision_latency.p50_milliseconds,
            self.decision_latency.p90_milliseconds,
            self.decision_latency.p99_milliseconds,
            self.decision_latency.max_milliseconds,
            self.decision_latency.decisions,
            self.elapsed_seconds,
            self.games_per_second,
            self.mean_breakdown.habitat[0],
            self.mean_breakdown.habitat[1],
            self.mean_breakdown.habitat[2],
            self.mean_breakdown.habitat[3],
            self.mean_breakdown.habitat[4],
            self.mean_breakdown.wildlife[0],
            self.mean_breakdown.wildlife[1],
            self.mean_breakdown.wildlife[2],
            self.mean_breakdown.wildlife[3],
            self.mean_breakdown.wildlife[4],
            self.mean_breakdown.nature_tokens,
        );
        output.retain(|character| character != '\r');
        output
    }
}

impl ComparisonReport {
    pub fn to_markdown(&self) -> String {
        format!(
            "# Cascadia Paired Comparison\n\n\
             - Protocol: `{}`\n\
             - Baseline: `{}`\n\
             - Treatment: `{}`\n\
             - Games: {} ({} seat scores per strategy)\n\
             - Baseline mean: {:.3}\n\
             - Treatment mean: {:.3}\n\
             - Baseline P10 / P50 / P90: {:.1} / {:.1} / {:.1}\n\
             - Treatment P10 / P50 / P90: {:.1} / {:.1} / {:.1}\n\
             - Baseline seat SD / range: {:.3} / {:.1}-{:.1}\n\
             - Treatment seat SD / range: {:.3} / {:.1}-{:.1}\n\
             - Paired delta: **{:+.3}**\n\
             - 95% CI: [{:+.3}, {:+.3}]\n\
             - Paired SD / SE: {:.3} / {:.3}\n\
             - Game wins / ties / losses: {} / {} / {}\n\
             - Baseline decision latency mean / P50 / P90 / P99 / max: {:.2} / {:.2} / {:.2} / {:.2} / {:.2} ms\n\
             - Treatment decision latency mean / P50 / P90 / P99 / max: {:.2} / {:.2} / {:.2} / {:.2} / {:.2} ms\n\
             - Baseline runtime: {:.3}s ({:.3}s/game)\n\
             - Treatment runtime: {:.3}s ({:.3}s/game)\n\
             - Combined wall time: {:.3}s\n\n\
             ## Mean Breakdown\n\n\
             | Habitat | Mountain | Forest | Prairie | Wetland | River |\n\
             |---|---:|---:|---:|---:|---:|\n\
             | Baseline | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |\n\
             | Treatment | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |\n\
             | Treatment - baseline | {:+.3} | {:+.3} | {:+.3} | {:+.3} | {:+.3} |\n\n\
             | Component | Bear | Elk | Salmon | Hawk | Fox | Tokens |\n\
             |---|---:|---:|---:|---:|---:|---:|\n\
             | Baseline | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |\n\
             | Treatment | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |\n\
             | Treatment - baseline | {:+.3} | {:+.3} | {:+.3} | {:+.3} | {:+.3} | {:+.3} |\n",
            self.protocol_id,
            self.baseline_id,
            self.treatment_id,
            self.games,
            self.seat_games_per_strategy,
            self.baseline_mean,
            self.treatment_mean,
            self.baseline_statistics.percentiles.p10,
            self.baseline_statistics.percentiles.p50,
            self.baseline_statistics.percentiles.p90,
            self.treatment_statistics.percentiles.p10,
            self.treatment_statistics.percentiles.p50,
            self.treatment_statistics.percentiles.p90,
            self.baseline_statistics.seat_score_stddev,
            self.baseline_statistics.min_score,
            self.baseline_statistics.max_score,
            self.treatment_statistics.seat_score_stddev,
            self.treatment_statistics.min_score,
            self.treatment_statistics.max_score,
            self.mean_paired_delta,
            self.confidence_95[0],
            self.confidence_95[1],
            self.paired_delta_stddev,
            self.standard_error,
            self.game_wins,
            self.game_ties,
            self.game_losses,
            self.baseline_decision_latency.mean_milliseconds,
            self.baseline_decision_latency.p50_milliseconds,
            self.baseline_decision_latency.p90_milliseconds,
            self.baseline_decision_latency.p99_milliseconds,
            self.baseline_decision_latency.max_milliseconds,
            self.treatment_decision_latency.mean_milliseconds,
            self.treatment_decision_latency.p50_milliseconds,
            self.treatment_decision_latency.p90_milliseconds,
            self.treatment_decision_latency.p99_milliseconds,
            self.treatment_decision_latency.max_milliseconds,
            self.baseline_elapsed_seconds,
            self.baseline_seconds_per_game,
            self.treatment_elapsed_seconds,
            self.treatment_seconds_per_game,
            self.elapsed_seconds,
            self.baseline_breakdown.habitat[0],
            self.baseline_breakdown.habitat[1],
            self.baseline_breakdown.habitat[2],
            self.baseline_breakdown.habitat[3],
            self.baseline_breakdown.habitat[4],
            self.treatment_breakdown.habitat[0],
            self.treatment_breakdown.habitat[1],
            self.treatment_breakdown.habitat[2],
            self.treatment_breakdown.habitat[3],
            self.treatment_breakdown.habitat[4],
            self.mean_breakdown_delta.habitat[0],
            self.mean_breakdown_delta.habitat[1],
            self.mean_breakdown_delta.habitat[2],
            self.mean_breakdown_delta.habitat[3],
            self.mean_breakdown_delta.habitat[4],
            self.baseline_breakdown.wildlife[0],
            self.baseline_breakdown.wildlife[1],
            self.baseline_breakdown.wildlife[2],
            self.baseline_breakdown.wildlife[3],
            self.baseline_breakdown.wildlife[4],
            self.baseline_breakdown.nature_tokens,
            self.treatment_breakdown.wildlife[0],
            self.treatment_breakdown.wildlife[1],
            self.treatment_breakdown.wildlife[2],
            self.treatment_breakdown.wildlife[3],
            self.treatment_breakdown.wildlife[4],
            self.treatment_breakdown.nature_tokens,
            self.mean_breakdown_delta.wildlife[0],
            self.mean_breakdown_delta.wildlife[1],
            self.mean_breakdown_delta.wildlife[2],
            self.mean_breakdown_delta.wildlife[3],
            self.mean_breakdown_delta.wildlife[4],
            self.mean_breakdown_delta.nature_tokens,
        )
    }
}

pub fn run_benchmark(config: BenchmarkConfig) -> Result<BenchmarkReport, BenchmarkError> {
    if config.games == 0 {
        return Err(BenchmarkError::NoGames);
    }
    let game_config = GameConfig::research_aaaaa(4)?;
    let started = Instant::now();
    let run_game = |index: usize| {
        let seed = config.first_seed + index as u64;
        play_match(&MatchConfig::symmetric(
            game_config,
            GameSeed::from_u64(seed),
            config.strategy,
        ))
    };
    let results: Vec<MatchResult> = if config.parallel {
        (0..config.games)
            .into_par_iter()
            .map(run_game)
            .collect::<Result<_, _>>()?
    } else {
        (0..config.games).map(run_game).collect::<Result<_, _>>()?
    };
    let elapsed_seconds = started.elapsed().as_secs_f64();
    Ok(summarize(config, &results, elapsed_seconds))
}

pub fn run_comparison(config: ComparisonConfig) -> Result<ComparisonReport, BenchmarkError> {
    if config.games == 0 {
        return Err(BenchmarkError::NoGames);
    }
    let game_config = GameConfig::research_aaaaa(4)?;
    let started = Instant::now();
    let run_pair = |index: usize| {
        let seed = config.first_seed + index as u64;
        let game_seed = GameSeed::from_u64(seed);
        let baseline = play_match(&MatchConfig::symmetric(
            game_config,
            game_seed,
            config.baseline,
        ))?;
        let treatment = play_match(&MatchConfig::symmetric(
            game_config,
            game_seed,
            config.treatment,
        ))?;
        Ok::<_, SimulationError>((seed, baseline, treatment))
    };
    let results: Vec<_> = if config.parallel {
        (0..config.games)
            .into_par_iter()
            .map(run_pair)
            .collect::<Result<_, _>>()?
    } else {
        (0..config.games).map(run_pair).collect::<Result<_, _>>()?
    };
    let elapsed_seconds = started.elapsed().as_secs_f64();
    Ok(summarize_comparison(config, &results, elapsed_seconds))
}

fn summarize(
    config: BenchmarkConfig,
    results: &[MatchResult],
    elapsed_seconds: f64,
) -> BenchmarkReport {
    summarize_match_results(
        config.strategy.id(),
        config.games,
        config.first_seed,
        results,
        elapsed_seconds,
    )
}

pub fn summarize_match_results(
    strategy_id: &str,
    games: usize,
    first_seed: u64,
    results: &[MatchResult],
    elapsed_seconds: f64,
) -> BenchmarkReport {
    let blocks = results
        .iter()
        .map(EvaluationBlock::from)
        .collect::<Vec<_>>();
    let statistics = strategy_statistics(&blocks);

    BenchmarkReport {
        protocol_id: PROTOCOL_ID.to_owned(),
        strategy_id: strategy_id.to_owned(),
        games,
        seat_games: blocks.iter().map(|result| result.scores.len()).sum(),
        first_seed,
        mean_score: statistics.mean_score,
        game_mean_stddev: statistics.game_mean_stddev,
        seat_score_stddev: statistics.seat_score_stddev,
        standard_error: statistics.standard_error,
        confidence_95: statistics.confidence_95,
        percentiles: statistics.percentiles,
        min_score: statistics.min_score,
        max_score: statistics.max_score,
        mean_breakdown: statistics.mean_breakdown,
        decision_latency: statistics.decision_latency,
        elapsed_seconds,
        games_per_second: games as f64 / elapsed_seconds,
    }
}

fn summarize_comparison(
    config: ComparisonConfig,
    results: &[(u64, MatchResult, MatchResult)],
    elapsed_seconds: f64,
) -> ComparisonReport {
    summarize_paired_match_results(
        config.baseline.id(),
        config.treatment.id(),
        config.first_seed,
        results,
        elapsed_seconds,
    )
}

pub fn summarize_paired_match_results(
    baseline_id: &str,
    treatment_id: &str,
    first_seed: u64,
    results: &[(u64, MatchResult, MatchResult)],
    elapsed_seconds: f64,
) -> ComparisonReport {
    let blocks = results
        .iter()
        .map(|(seed, baseline, treatment)| {
            (
                *seed,
                EvaluationBlock::from(baseline),
                EvaluationBlock::from(treatment),
            )
        })
        .collect::<Vec<_>>();
    summarize_paired_evaluation_blocks(
        baseline_id,
        treatment_id,
        first_seed,
        &blocks,
        elapsed_seconds,
    )
}

pub fn summarize_paired_evaluation_blocks(
    baseline_id: &str,
    treatment_id: &str,
    first_seed: u64,
    results: &[(u64, EvaluationBlock, EvaluationBlock)],
    elapsed_seconds: f64,
) -> ComparisonReport {
    let pairs: Vec<_> = results
        .iter()
        .map(|(seed, baseline, treatment)| {
            let baseline_mean = match_mean(baseline);
            let treatment_mean = match_mean(treatment);
            PairedGame {
                seed: *seed,
                baseline_mean,
                treatment_mean,
                delta: treatment_mean - baseline_mean,
            }
        })
        .collect();
    let deltas: Vec<_> = pairs.iter().map(|pair| pair.delta).collect();
    let mean_paired_delta = mean(&deltas);
    let paired_delta_stddev = sample_stddev(&deltas, mean_paired_delta);
    let games = results.len();
    let standard_error = paired_delta_stddev / (games as f64).sqrt();
    let margin = 1.96 * standard_error;
    let baseline_results: Vec<_> = results
        .iter()
        .map(|(_, baseline, _)| baseline.clone())
        .collect();
    let treatment_results: Vec<_> = results
        .iter()
        .map(|(_, _, treatment)| treatment.clone())
        .collect();
    let baseline_breakdown = mean_breakdown(&baseline_results);
    let treatment_breakdown = mean_breakdown(&treatment_results);
    let baseline_statistics = strategy_statistics(&baseline_results);
    let treatment_statistics = strategy_statistics(&treatment_results);
    let baseline_elapsed_seconds = baseline_results
        .iter()
        .map(|result| result.elapsed_seconds)
        .sum::<f64>();
    let treatment_elapsed_seconds = treatment_results
        .iter()
        .map(|result| result.elapsed_seconds)
        .sum::<f64>();
    let baseline_decision_latency = baseline_statistics.decision_latency.clone();
    let treatment_decision_latency = treatment_statistics.decision_latency.clone();
    let seat_games_per_strategy = baseline_results
        .iter()
        .map(|result| result.scores.len())
        .sum();

    ComparisonReport {
        protocol_id: PROTOCOL_ID.to_owned(),
        baseline_id: baseline_id.to_owned(),
        treatment_id: treatment_id.to_owned(),
        games,
        seat_games_per_strategy,
        first_seed,
        baseline_mean: baseline_statistics.mean_score,
        treatment_mean: treatment_statistics.mean_score,
        baseline_statistics,
        treatment_statistics,
        mean_paired_delta,
        paired_delta_stddev,
        standard_error,
        confidence_95: [mean_paired_delta - margin, mean_paired_delta + margin],
        game_wins: deltas.iter().filter(|delta| **delta > 0.0).count(),
        game_ties: deltas.iter().filter(|delta| **delta == 0.0).count(),
        game_losses: deltas.iter().filter(|delta| **delta < 0.0).count(),
        baseline_breakdown: baseline_breakdown.clone(),
        treatment_breakdown: treatment_breakdown.clone(),
        mean_breakdown_delta: subtract_breakdowns(treatment_breakdown, baseline_breakdown),
        baseline_decision_latency,
        treatment_decision_latency,
        baseline_elapsed_seconds,
        treatment_elapsed_seconds,
        baseline_seconds_per_game: baseline_elapsed_seconds / games as f64,
        treatment_seconds_per_game: treatment_elapsed_seconds / games as f64,
        elapsed_seconds,
        pairs,
    }
}

fn strategy_statistics(results: &[EvaluationBlock]) -> StrategyStatistics {
    let game_means: Vec<_> = results.iter().map(match_mean).collect();
    let seat_scores: Vec<_> = results
        .iter()
        .flat_map(|result| result.scores.iter())
        .map(|score| f64::from(score.base_total))
        .collect();
    let mean_score = mean(&game_means);
    let game_mean_stddev = sample_stddev(&game_means, mean_score);
    let seat_score_mean = mean(&seat_scores);
    let seat_score_stddev = sample_stddev(&seat_scores, seat_score_mean);
    let standard_error = game_mean_stddev / (game_means.len() as f64).sqrt();
    let margin = 1.96 * standard_error;
    StrategyStatistics {
        mean_score,
        game_mean_stddev,
        seat_score_stddev,
        standard_error,
        confidence_95: [mean_score - margin, mean_score + margin],
        percentiles: Percentiles {
            p10: percentile(&seat_scores, 0.10),
            p50: percentile(&seat_scores, 0.50),
            p90: percentile(&seat_scores, 0.90),
        },
        min_score: seat_scores.iter().copied().fold(f64::INFINITY, f64::min),
        max_score: seat_scores
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
        mean_breakdown: mean_breakdown(results),
        decision_latency: decision_latency(results),
    }
}

fn decision_latency(results: &[EvaluationBlock]) -> DecisionLatency {
    let milliseconds: Vec<_> = results
        .iter()
        .flat_map(|result| result.decision_seconds.iter())
        .map(|seconds| seconds * 1_000.0)
        .collect();
    if milliseconds.is_empty() {
        return DecisionLatency {
            decisions: 0,
            mean_milliseconds: 0.0,
            p50_milliseconds: 0.0,
            p90_milliseconds: 0.0,
            p99_milliseconds: 0.0,
            max_milliseconds: 0.0,
        };
    }
    DecisionLatency {
        decisions: milliseconds.len(),
        mean_milliseconds: mean(&milliseconds),
        p50_milliseconds: percentile(&milliseconds, 0.50),
        p90_milliseconds: percentile(&milliseconds, 0.90),
        p99_milliseconds: percentile(&milliseconds, 0.99),
        max_milliseconds: milliseconds
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
    }
}

fn match_mean(result: &EvaluationBlock) -> f64 {
    result
        .scores
        .iter()
        .map(|score| f64::from(score.base_total))
        .sum::<f64>()
        / result.scores.len() as f64
}

fn subtract_breakdowns(treatment: MeanBreakdown, baseline: MeanBreakdown) -> MeanBreakdown {
    MeanBreakdown {
        habitat: std::array::from_fn(|index| treatment.habitat[index] - baseline.habitat[index]),
        wildlife: std::array::from_fn(|index| treatment.wildlife[index] - baseline.wildlife[index]),
        nature_tokens: treatment.nature_tokens - baseline.nature_tokens,
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_stddev(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    variance.sqrt()
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

fn mean_breakdown(results: &[EvaluationBlock]) -> MeanBreakdown {
    let scores: Vec<&ScoreBreakdown> = results
        .iter()
        .flat_map(|result| result.scores.iter())
        .collect();
    let count = scores.len() as f64;
    let mut habitat = [0.0; 5];
    let mut wildlife = [0.0; 5];
    let mut nature_tokens = 0.0;
    for score in scores {
        for index in 0..5 {
            habitat[index] += f64::from(score.habitat[index]);
            wildlife[index] += f64::from(score.wildlife[index]);
        }
        nature_tokens += f64::from(score.nature_tokens);
    }
    for index in 0..5 {
        habitat[index] /= count;
        wildlife[index] /= count;
    }
    MeanBreakdown {
        habitat,
        wildlife,
        nature_tokens: nature_tokens / count,
    }
}

#[derive(Debug, Error)]
pub enum BenchmarkError {
    #[error("benchmark must contain at least one game")]
    NoGames,
    #[error(transparent)]
    Rules(#[from] cascadia_game::RuleError),
    #[error(transparent)]
    Simulation(#[from] SimulationError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_smoke_report_is_reproducible() {
        let config = BenchmarkConfig {
            games: 2,
            first_seed: 55,
            strategy: StrategyKind::Random,
            parallel: false,
        };
        let left = run_benchmark(config).unwrap();
        let right = run_benchmark(config).unwrap();

        assert_eq!(left.mean_score, right.mean_score);
        assert_eq!(left.mean_breakdown, right.mean_breakdown);
        assert_eq!(left.seat_games, 8);
        assert_eq!(left.decision_latency.decisions, 160);
        assert_eq!(left.protocol_id, PROTOCOL_ID);
        assert!(left.to_markdown().contains("Mean base score"));
    }

    #[test]
    fn paired_comparison_of_identical_strategies_is_exactly_zero() {
        let report = run_comparison(ComparisonConfig {
            games: 2,
            first_seed: 77,
            baseline: StrategyKind::Random,
            treatment: StrategyKind::Random,
            parallel: true,
        })
        .unwrap();
        assert_eq!(report.mean_paired_delta, 0.0);
        assert_eq!(report.confidence_95, [0.0, 0.0]);
        assert_eq!(report.game_ties, 2);
        assert_eq!(report.pairs.len(), 2);
        assert_eq!(report.baseline_decision_latency.decisions, 160);
        assert_eq!(report.treatment_decision_latency.decisions, 160);
        assert!(report.baseline_seconds_per_game > 0.0);
        assert!(report.treatment_seconds_per_game > 0.0);
        assert!(report.to_markdown().contains("Paired delta"));
    }

    #[test]
    fn reports_without_latency_fields_remain_readable() {
        let report = run_comparison(ComparisonConfig {
            games: 1,
            first_seed: 78,
            baseline: StrategyKind::Random,
            treatment: StrategyKind::Random,
            parallel: false,
        })
        .unwrap();
        let mut value = serde_json::to_value(report).unwrap();
        let object = value.as_object_mut().unwrap();
        object.remove("baseline_decision_latency");
        object.remove("treatment_decision_latency");
        object["baseline_statistics"]
            .as_object_mut()
            .unwrap()
            .remove("decision_latency");
        object["treatment_statistics"]
            .as_object_mut()
            .unwrap()
            .remove("decision_latency");

        let decoded: ComparisonReport = serde_json::from_value(value).unwrap();
        assert_eq!(
            decoded.baseline_decision_latency,
            DecisionLatency::default()
        );
        assert_eq!(
            decoded.treatment_decision_latency,
            DecisionLatency::default()
        );
        assert_eq!(
            decoded.baseline_statistics.decision_latency,
            DecisionLatency::default()
        );
    }

    #[test]
    fn explicit_evaluation_blocks_preserve_seat_rotation_statistics() {
        let score = |total| ScoreBreakdown {
            habitat: [0; 5],
            wildlife: [total, 0, 0, 0, 0],
            nature_tokens: 0,
            habitat_bonus: [0; 5],
            base_total: total,
            total,
        };
        let results = vec![(
            91,
            EvaluationBlock {
                scores: vec![score(80), score(81), score(82), score(83)],
                decision_seconds: vec![0.001; 80],
                elapsed_seconds: 1.0,
            },
            EvaluationBlock {
                scores: vec![score(84), score(85), score(86), score(87)],
                decision_seconds: vec![0.002; 80],
                elapsed_seconds: 4.0,
            },
        )];

        let report =
            summarize_paired_evaluation_blocks("baseline", "focal-treatment", 91, &results, 5.0);

        assert_eq!(report.games, 1);
        assert_eq!(report.seat_games_per_strategy, 4);
        assert_eq!(report.baseline_mean, 81.5);
        assert_eq!(report.treatment_mean, 85.5);
        assert_eq!(report.mean_paired_delta, 4.0);
        assert_eq!(report.treatment_decision_latency.decisions, 80);
        assert_eq!(report.treatment_seconds_per_game, 4.0);
    }
}
