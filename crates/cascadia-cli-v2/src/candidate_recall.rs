use std::time::Instant;

use cascadia_eval::PROTOCOL_ID;
use cascadia_game::{GameConfig, GameSeed, GameState, score_game};
use cascadia_search::{DeterminizedLookaheadConfig, DeterminizedLookaheadStrategy};
use serde::Serialize;

pub fn run_lookahead_recall(
    games: usize,
    first_seed: u64,
    retained_candidates: usize,
    expanded_candidates: usize,
    determinizations: usize,
    greedy_plies: usize,
) -> Result<CandidateRecallReport, Box<dyn std::error::Error>> {
    if games == 0 {
        return Err("lookahead-recall requires at least one game".into());
    }
    if expanded_candidates <= retained_candidates {
        return Err("expanded candidate count must exceed retained candidate count".into());
    }
    let baseline = DeterminizedLookaheadStrategy::new(DeterminizedLookaheadConfig {
        candidate_limit: retained_candidates,
        determinizations,
        greedy_plies,
    })?;
    let expanded = DeterminizedLookaheadStrategy::new(DeterminizedLookaheadConfig {
        candidate_limit: expanded_candidates,
        determinizations,
        greedy_plies,
    })?;
    let game_config = GameConfig::research_aaaaa(4)?;
    let started = Instant::now();
    let mut decisions = 0;
    let mut outside_retained = 0;
    let mut strict_misses = 0;
    let mut regret_sum = 0.0;
    let mut missed_regret_sum = 0.0;
    let mut max_regret = 0.0_f64;
    let mut rank_histogram = vec![0; expanded_candidates + 1];
    let mut phases = [RecallAccumulator::default(); 3];
    let mut trajectory_score_sum = 0.0;

    for index in 0..games {
        let seed = GameSeed::from_u64(first_seed + index as u64);
        let mut game = GameState::new(game_config, seed)?;
        while !game.is_game_over() {
            let expanded_ranked = expanded.rank_actions_deterministic(&game)?;
            let (baseline_ranked, action) = baseline.rank_and_select_deterministic(&game)?;
            let expanded_best = &expanded_ranked[0];
            let retained_best = expanded_ranked
                .iter()
                .filter(|candidate| candidate.immediate_rank <= retained_candidates)
                .max_by(|left, right| left.mean_leaf_score.total_cmp(&right.mean_leaf_score))
                .ok_or("expanded search did not contain a retained candidate")?;
            if baseline_ranked[0].mean_leaf_score != retained_best.mean_leaf_score {
                return Err("shared search samples produced inconsistent retained values".into());
            }

            let regret = (expanded_best.mean_leaf_score - retained_best.mean_leaf_score).max(0.0);
            let outside = expanded_best.immediate_rank > retained_candidates;
            let missed = regret > 0.0;
            decisions += 1;
            outside_retained += usize::from(outside);
            strict_misses += usize::from(missed);
            regret_sum += regret;
            if missed {
                missed_regret_sum += regret;
            }
            max_regret = max_regret.max(regret);
            rank_histogram[expanded_best.immediate_rank] += 1;
            let phase = ((usize::from(game.completed_turns()) * 3)
                / usize::from(game.total_turns()))
            .min(2);
            phases[phase].record(outside, missed, regret);
            game.apply(&action)?;
        }
        trajectory_score_sum += score_game(&game)
            .iter()
            .map(|score| f64::from(score.base_total))
            .sum::<f64>();
    }

    Ok(CandidateRecallReport {
        protocol_id: PROTOCOL_ID.to_owned(),
        trajectory_strategy_id: baseline.strategy_id().to_owned(),
        expanded_strategy_id: expanded.strategy_id().to_owned(),
        games,
        seat_games: games * usize::from(game_config.player_count),
        first_seed,
        decisions,
        retained_candidates,
        expanded_candidates,
        expanded_best_outside_retained: outside_retained,
        strict_value_misses: strict_misses,
        selection_coverage: 1.0 - outside_retained as f64 / decisions as f64,
        value_recall: 1.0 - strict_misses as f64 / decisions as f64,
        mean_estimated_regret: regret_sum / decisions as f64,
        mean_estimated_regret_when_missed: if strict_misses == 0 {
            0.0
        } else {
            missed_regret_sum / strict_misses as f64
        },
        max_estimated_regret: max_regret,
        immediate_rank_histogram: rank_histogram[1..].to_vec(),
        phase: phases.map(RecallAccumulator::finish),
        trajectory_mean_score: trajectory_score_sum
            / (games * usize::from(game_config.player_count)) as f64,
        elapsed_seconds: started.elapsed().as_secs_f64(),
    })
}

#[derive(Debug, Clone, Copy, Default)]
struct RecallAccumulator {
    decisions: usize,
    outside: usize,
    misses: usize,
    regret_sum: f64,
}

impl RecallAccumulator {
    fn record(&mut self, outside: bool, missed: bool, regret: f64) {
        self.decisions += 1;
        self.outside += usize::from(outside);
        self.misses += usize::from(missed);
        self.regret_sum += regret;
    }

    fn finish(self) -> PhaseRecall {
        PhaseRecall {
            decisions: self.decisions,
            expanded_best_outside_retained: self.outside,
            strict_value_misses: self.misses,
            value_recall: if self.decisions == 0 {
                1.0
            } else {
                1.0 - self.misses as f64 / self.decisions as f64
            },
            mean_estimated_regret: if self.decisions == 0 {
                0.0
            } else {
                self.regret_sum / self.decisions as f64
            },
        }
    }
}

#[derive(Debug, Serialize)]
pub struct CandidateRecallReport {
    protocol_id: String,
    trajectory_strategy_id: String,
    expanded_strategy_id: String,
    games: usize,
    seat_games: usize,
    first_seed: u64,
    decisions: usize,
    retained_candidates: usize,
    expanded_candidates: usize,
    expanded_best_outside_retained: usize,
    strict_value_misses: usize,
    selection_coverage: f64,
    value_recall: f64,
    mean_estimated_regret: f64,
    mean_estimated_regret_when_missed: f64,
    max_estimated_regret: f64,
    immediate_rank_histogram: Vec<usize>,
    phase: [PhaseRecall; 3],
    trajectory_mean_score: f64,
    elapsed_seconds: f64,
}

impl CandidateRecallReport {
    pub fn to_markdown(&self) -> String {
        format!(
            "# Candidate Recall Diagnostic\n\n\
             - Protocol: `{}`\n\
             - Trajectory policy: `{}`\n\
             - Expanded evaluator: `{}`\n\
             - Games / decisions: {} / {}\n\
             - Selection coverage at K={}: {:.2}%\n\
             - Value recall at K={}: {:.2}%\n\
             - Strict value misses: {}\n\
             - Mean estimated regret: {:.3}\n\
             - Mean regret when missed: {:.3}\n\
             - Maximum estimated regret: {:.3}\n\
             - Trajectory mean score: {:.3}\n\
             - Runtime: {:.3}s\n\n\
             ## Phase Breakdown\n\n\
             | Phase | Decisions | Outside K | Strict misses | Value recall | Mean regret |\n\
             |---|---:|---:|---:|---:|---:|\n\
             | Early | {} | {} | {} | {:.2}% | {:.3} |\n\
             | Middle | {} | {} | {} | {:.2}% | {:.3} |\n\
             | Late | {} | {} | {} | {:.2}% | {:.3} |\n\n\
             Immediate-rank histogram: `{:?}`\n",
            self.protocol_id,
            self.trajectory_strategy_id,
            self.expanded_strategy_id,
            self.games,
            self.decisions,
            self.retained_candidates,
            self.selection_coverage * 100.0,
            self.retained_candidates,
            self.value_recall * 100.0,
            self.strict_value_misses,
            self.mean_estimated_regret,
            self.mean_estimated_regret_when_missed,
            self.max_estimated_regret,
            self.trajectory_mean_score,
            self.elapsed_seconds,
            self.phase[0].decisions,
            self.phase[0].expanded_best_outside_retained,
            self.phase[0].strict_value_misses,
            self.phase[0].value_recall * 100.0,
            self.phase[0].mean_estimated_regret,
            self.phase[1].decisions,
            self.phase[1].expanded_best_outside_retained,
            self.phase[1].strict_value_misses,
            self.phase[1].value_recall * 100.0,
            self.phase[1].mean_estimated_regret,
            self.phase[2].decisions,
            self.phase[2].expanded_best_outside_retained,
            self.phase[2].strict_value_misses,
            self.phase[2].value_recall * 100.0,
            self.phase[2].mean_estimated_regret,
            self.immediate_rank_histogram,
        )
    }
}

#[derive(Debug, Serialize)]
struct PhaseRecall {
    decisions: usize,
    expanded_best_outside_retained: usize,
    strict_value_misses: usize,
    value_recall: f64,
    mean_estimated_regret: f64,
}
