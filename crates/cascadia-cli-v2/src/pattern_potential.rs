use std::{cmp::Ordering, fmt::Write as _, time::Instant};

use cascadia_eval::{BenchmarkReport, PROTOCOL_ID, summarize_match_results};
use cascadia_game::{GameConfig, GameSeed};
use cascadia_sim::{PatternAwareConfig, PatternPotentialConfig, PatternPotentialStrategy};
use rayon::prelude::*;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
struct PatternPotentialPoint {
    opportunity_weight: f64,
    habitat_weight: f64,
    bear_weight: f64,
}

impl PatternPotentialPoint {
    fn from_config(config: PatternPotentialConfig) -> Self {
        Self {
            opportunity_weight: config.opportunity_weight(),
            habitat_weight: config.habitat_weight(),
            bear_weight: config.bear_weight(),
        }
    }

    fn config(self) -> PatternPotentialConfig {
        PatternPotentialConfig::from_weights(
            PatternAwareConfig::default(),
            self.opportunity_weight,
            self.habitat_weight,
            self.bear_weight,
        )
        .expect("registered pattern-potential grid points are valid")
    }

    fn production_distance_quarters(self) -> u8 {
        let opportunity = (self.opportunity_weight * 4.0).round() as u8;
        let habitat = (self.habitat_weight * 4.0).round() as u8;
        let bear = (self.bear_weight * 4.0).round() as u8;
        opportunity.abs_diff(4) + habitat + bear
    }
}

#[derive(Debug, Clone, Serialize)]
struct PatternPotentialSweepEntry {
    point: PatternPotentialPoint,
    benchmark: BenchmarkReport,
}

#[derive(Debug, Serialize)]
pub(crate) struct PatternPotentialSweepReport {
    protocol_id: String,
    games_per_policy: usize,
    seat_games_per_policy: usize,
    first_seed: u64,
    policies: usize,
    baseline_strategy_id: String,
    baseline_mean: f64,
    selected_strategy_id: String,
    selected_point: PatternPotentialPoint,
    selected_mean: f64,
    selected_gain: f64,
    selection_gate_minimum_gain: f64,
    selection_gate_passed: bool,
    elapsed_seconds: f64,
    ranked: Vec<PatternPotentialSweepEntry>,
}

impl PatternPotentialSweepReport {
    pub(crate) fn to_markdown(&self) -> String {
        let mut output = format!(
            "# Pattern Potential Grid Selection\n\n\
             - Protocol: `{}`\n\
             - Policies: {}\n\
             - Games per policy: {} ({} seat scores)\n\
             - Seeds: {} through {}\n\
             - Production baseline: `{}` = {:.3}\n\
             - Selected: `{}` = {:.3}\n\
             - Selected gain: **{:+.3}**\n\
             - Selection gate: {:+.3} ({})\n\
             - Wall time: {:.3}s\n\n\
             ## Top Policies\n\n\
             | Rank | Opportunity | Habitat | Bear | Mean | Gain |\n\
             |---:|---:|---:|---:|---:|---:|\n",
            self.protocol_id,
            self.policies,
            self.games_per_policy,
            self.seat_games_per_policy,
            self.first_seed,
            self.first_seed + self.games_per_policy.saturating_sub(1) as u64,
            self.baseline_strategy_id,
            self.baseline_mean,
            self.selected_strategy_id,
            self.selected_mean,
            self.selected_gain,
            self.selection_gate_minimum_gain,
            if self.selection_gate_passed {
                "passed"
            } else {
                "failed"
            },
            self.elapsed_seconds,
        );
        for (index, entry) in self.ranked.iter().take(15).enumerate() {
            writeln!(
                output,
                "| {} | {:.2} | {:.2} | {:.2} | {:.3} | {:+.3} |",
                index + 1,
                entry.point.opportunity_weight,
                entry.point.habitat_weight,
                entry.point.bear_weight,
                entry.benchmark.mean_score,
                entry.benchmark.mean_score - self.baseline_mean,
            )
            .expect("writing to String cannot fail");
        }
        output
    }
}

pub(crate) fn run_pattern_potential_sweep(
    games: usize,
    first_seed: u64,
) -> Result<PatternPotentialSweepReport, Box<dyn std::error::Error>> {
    let game_config = GameConfig::research_aaaaa(4)?;
    let points = (2..=6)
        .flat_map(|opportunity_quarters| {
            (0..=4).flat_map(move |habitat_quarters| {
                (0..=4).map(move |bear_quarters| PatternPotentialConfig {
                    blueprint: PatternAwareConfig::default(),
                    opportunity_quarters,
                    habitat_quarters,
                    bear_quarters,
                })
            })
        })
        .map(PatternPotentialPoint::from_config)
        .collect::<Vec<_>>();
    let started = Instant::now();
    let entries = points
        .into_par_iter()
        .map(|point| {
            let strategy = PatternPotentialStrategy::new(point.config())?;
            let results = (0..games)
                .map(|index| {
                    strategy.play_match(game_config, GameSeed::from_u64(first_seed + index as u64))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let elapsed_seconds = results
                .iter()
                .map(|result| result.elapsed_seconds)
                .sum::<f64>();
            Ok::<_, cascadia_sim::SimulationError>(PatternPotentialSweepEntry {
                point,
                benchmark: summarize_match_results(
                    strategy.strategy_id(),
                    games,
                    first_seed,
                    &results,
                    elapsed_seconds,
                ),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut ranked = entries;
    ranked.sort_by(|left, right| {
        compare_pattern_potential_points(
            left.point,
            left.benchmark.mean_score,
            right.point,
            right.benchmark.mean_score,
        )
    });
    let baseline = ranked
        .iter()
        .find(|entry| {
            entry.point
                == (PatternPotentialPoint {
                    opportunity_weight: 1.0,
                    habitat_weight: 0.0,
                    bear_weight: 0.0,
                })
        })
        .expect("registered grid contains production pattern-aware");
    let selected = ranked
        .first()
        .expect("registered pattern-potential grid is non-empty");
    let selected_gain = selected.benchmark.mean_score - baseline.benchmark.mean_score;

    Ok(PatternPotentialSweepReport {
        protocol_id: PROTOCOL_ID.to_owned(),
        games_per_policy: games,
        seat_games_per_policy: games * usize::from(game_config.player_count),
        first_seed,
        policies: ranked.len(),
        baseline_strategy_id: baseline.benchmark.strategy_id.clone(),
        baseline_mean: baseline.benchmark.mean_score,
        selected_strategy_id: selected.benchmark.strategy_id.clone(),
        selected_point: selected.point,
        selected_mean: selected.benchmark.mean_score,
        selected_gain,
        selection_gate_minimum_gain: 0.40,
        selection_gate_passed: selected_gain >= 0.40,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        ranked,
    })
}

fn compare_pattern_potential_points(
    left: PatternPotentialPoint,
    left_mean: f64,
    right: PatternPotentialPoint,
    right_mean: f64,
) -> Ordering {
    right_mean
        .total_cmp(&left_mean)
        .then_with(|| {
            left.production_distance_quarters()
                .cmp(&right.production_distance_quarters())
        })
        .then_with(|| left.opportunity_weight.total_cmp(&right.opportunity_weight))
        .then_with(|| left.habitat_weight.total_cmp(&right.habitat_weight))
        .then_with(|| left.bear_weight.total_cmp(&right.bear_weight))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_potential_selection_prefers_simpler_then_lexicographic_ties() {
        let production = PatternPotentialPoint {
            opportunity_weight: 1.0,
            habitat_weight: 0.0,
            bear_weight: 0.0,
        };
        let structural = PatternPotentialPoint {
            opportunity_weight: 1.0,
            habitat_weight: 0.25,
            bear_weight: 0.0,
        };
        assert_eq!(
            compare_pattern_potential_points(production, 92.0, structural, 92.0),
            Ordering::Less
        );

        let lower_opportunity = PatternPotentialPoint {
            opportunity_weight: 0.75,
            habitat_weight: 0.25,
            bear_weight: 0.0,
        };
        let higher_opportunity = PatternPotentialPoint {
            opportunity_weight: 1.25,
            habitat_weight: 0.25,
            bear_weight: 0.0,
        };
        assert_eq!(
            lower_opportunity.production_distance_quarters(),
            higher_opportunity.production_distance_quarters()
        );
        assert_eq!(
            compare_pattern_potential_points(lower_opportunity, 92.0, higher_opportunity, 92.0,),
            Ordering::Less
        );
    }
}
