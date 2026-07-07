//! Gumbel AlphaZero-style root search with neural-network leaf values.
//!
//! Replaces the flat one-ply greedy-rollout teacher. Key properties:
//!
//! - Root candidates come from the FULL legal action set (or a model-side cap),
//!   never a greedy-ranked truncation, so non-greedy plans stay reachable.
//! - Root action selection is Gumbel top-m + sequential halving over model
//!   policy logits, with completed-Q values from determinized simulations.
//! - Every simulation runs on a hidden-redeterminized clone, sampled BEFORE the
//!   root action's market refill, so search is public-information-legal by
//!   construction (no peeking at the true tile/bag order).
//! - Interior plies advance every seat by argmax of its own derived final Q
//!   (`exact_afterstate_score_active + predicted score-to-go`), which is max^n
//!   play under the model's value estimates.
//! - Leaf value for the root seat is `w * max-Q bootstrap + (1-w) * sampled
//!   greedy terminal rollout`; `w` ramps toward 1.0 as the value model earns
//!   trust. Simulations advance in lockstep so every ply of every live
//!   simulation lands in one batched model evaluation.

use anyhow::{Context, Result, bail};
use cascadia_game::{GameSeed, GameState, MarketPrelude, score_game};
use cascadia_sim::rank_greedy_actions;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{CandidateAfterstate, candidate_afterstates, complete_with_sampled_greedy};

const GUMBEL_NOISE_SALT: u64 = 0x6a3b_1e55_9d2c_4f01;
const DETERMINIZATION_STREAM_SALT: u64 = 0x0d5e_ed12_37ab_44c9;
const ROLLOUT_STREAM_SALT: u64 = 0x77c0_ffee_4bad_5eed;

#[derive(Debug, Clone)]
pub struct GumbelConfig {
    /// Total simulation budget across all root actions.
    pub n_simulations: usize,
    /// Number of root actions sampled without replacement via Gumbel top-m.
    pub top_m: usize,
    /// Optional cap on root candidates (model-ranked by prior after the root
    /// eval, NOT greedy-ranked). `None` keeps the full legal set.
    pub max_root_actions: Option<usize>,
    /// How many times the root seat re-enters before a simulation resolves to
    /// a leaf value. 1 = apply root action, walk opponent plies, value the
    /// state where the root seat moves again.
    pub depth_rounds: usize,
    /// Distinct hidden-order determinizations cycled across each action's
    /// simulations (common random numbers across actions).
    pub determinization_samples: usize,
    /// Leaf value = w * value bootstrap + (1-w) * sampled greedy terminal
    /// rollout. w=1.0 disables CPU rollouts entirely.
    pub rollout_blend_weight: f64,
    /// Sampled-greedy rollout parameters for the (1-w) branch.
    pub rollout_max_actions: usize,
    pub rollout_top_k: usize,
    /// Interior (non-root) ply menu cap.
    pub k_interior: usize,
    /// Gumbel exploration noise at the root (self-play data generation on,
    /// deterministic evaluation off).
    pub exploration: bool,
    /// sigma(q) = (c_visit + max_visits) * c_scale * minmax_norm(q).
    pub c_visit: f64,
    pub c_scale: f64,
    pub search_seed: u64,
    /// Hidden-determinization stream seed. `None` derives it from
    /// `search_seed`; pin it for common-random-number paired comparisons.
    pub determinization_seed: Option<u64>,
    /// ORACLE MODE: simulate on the true hidden state instead of
    /// redeterminizing. Leaks hidden information — only valid for
    /// measuring the information ceiling, never for honest gates or
    /// training labels.
    pub peek_true_hidden: bool,
}

impl Default for GumbelConfig {
    fn default() -> Self {
        Self {
            n_simulations: 64,
            top_m: 16,
            max_root_actions: None,
            depth_rounds: 1,
            determinization_samples: 4,
            rollout_blend_weight: 0.5,
            rollout_max_actions: 8,
            rollout_top_k: 4,
            k_interior: 16,
            exploration: false,
            c_visit: 50.0,
            c_scale: 1.0,
            search_seed: 0,
            determinization_seed: None,
            peek_true_hidden: false,
        }
    }
}

/// One state whose legal-action menu needs a model evaluation.
pub struct EvalRow {
    pub staged: GameState,
    pub prelude: MarketPrelude,
    pub afterstates: Vec<CandidateAfterstate>,
}

/// Per-row model outputs, action-aligned with `EvalRow::afterstates`.
#[derive(Debug, Clone)]
pub struct EvalOut {
    /// Normalized policy priors.
    pub priors: Vec<f64>,
    /// Derived final Q per action: exact afterstate score + score-to-go.
    pub derived_final_q: Vec<f64>,
}

pub trait LeafEvaluator {
    fn evaluate_batch(&mut self, rows: &[EvalRow]) -> Result<Vec<EvalOut>>;
}

#[derive(Debug, Clone)]
pub struct GumbelSearchResult {
    /// Per root action: simulation-averaged value for visited actions, model
    /// derived final Q for unvisited ones. Aligned with the root menu.
    pub completed_q: Vec<f64>,
    /// Population variance of simulation values per action (0 when unvisited).
    pub value_variance: Vec<f64>,
    /// softmax(logits + sigma(completed_q)) over the full root menu.
    pub improved_policy: Vec<f64>,
    pub visit_counts: Vec<u32>,
    /// Model root priors (normalized), aligned with the root menu.
    pub root_priors: Vec<f64>,
    /// Improved-policy-weighted mean of completed Q.
    pub root_value: f64,
    pub chosen_index: usize,
    pub simulations_run: usize,
}

/// Builds an `EvalRow` for a state, or `None` when the game is over.
pub fn eval_row_for_state(game: &GameState, menu_limit: Option<usize>) -> Result<Option<EvalRow>> {
    if game.is_game_over() {
        return Ok(None);
    }
    let (prelude, staged) = game.preview_free_three_of_a_kind_if_feasible()?;
    let candidates = match rank_greedy_actions(&staged, &MarketPrelude::default(), menu_limit) {
        Ok(candidates) => candidates,
        Err(cascadia_sim::SimulationError::Rules(error))
            if crate::is_rollout_truncation_rule_error(&error) =>
        {
            // Empty bag/stack truncation: value the state as its own terminal.
            return Ok(None);
        }
        Err(error) => return Err(error).context("ranking gumbel menu actions"),
    };
    if candidates.is_empty() {
        bail!(
            "no legal candidates for non-terminal state at turn {}",
            game.completed_turns()
        );
    }
    let active_seat = staged.current_player();
    let afterstates = candidate_afterstates(&staged, &candidates, active_seat)?;
    Ok(Some(EvalRow {
        staged,
        prelude,
        afterstates,
    }))
}

struct Simulation {
    /// Root-menu index of the action this simulation credits.
    action_index: usize,
    state: GameState,
    rounds_left: usize,
    rollout_rng: ChaCha8Rng,
    value: Option<f64>,
}

pub(crate) fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn gumbel_noise(rng: &mut ChaCha8Rng) -> f64 {
    let uniform: f64 = rng.gen_range(f64::EPSILON..1.0);
    -(-uniform.ln()).ln()
}

fn sigma(q_normalized: f64, max_visits: u32, c_visit: f64, c_scale: f64) -> f64 {
    (c_visit + f64::from(max_visits)) * c_scale * q_normalized
}

fn minmax_normalize(values: &[f64]) -> Vec<f64> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &value in values {
        min = min.min(value);
        max = max.max(value);
    }
    if !min.is_finite() || !max.is_finite() || (max - min).abs() < 1e-12 {
        return vec![0.0; values.len()];
    }
    values.iter().map(|value| (value - min) / (max - min)).collect()
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|logit| (logit - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|exp| exp / sum).collect()
}

/// Advances every live simulation by one ply (batched model eval), resolving
/// simulations that hit terminal states or leaf conditions.
fn advance_simulations(
    simulations: &mut [Simulation],
    root_seat: usize,
    evaluator: &mut dyn LeafEvaluator,
    cfg: &GumbelConfig,
) -> Result<()> {
    loop {
        let mut live_indexes = Vec::new();
        let mut rows = Vec::new();
        for (sim_index, simulation) in simulations.iter().enumerate() {
            if simulation.value.is_some() {
                continue;
            }
            match eval_row_for_state(&simulation.state, Some(cfg.k_interior))? {
                Some(row) => {
                    live_indexes.push(sim_index);
                    rows.push(row);
                }
                None => {
                    // Terminal: exact final score for the root seat.
                    live_indexes.push(sim_index);
                    rows.push(EvalRow {
                        staged: simulation.state.clone(),
                        prelude: MarketPrelude::default(),
                        afterstates: Vec::new(),
                    });
                }
            }
        }
        if live_indexes.is_empty() {
            return Ok(());
        }

        // Terminal rows resolve without a model call.
        let mut model_rows = Vec::new();
        let mut model_row_sims = Vec::new();
        for (&sim_index, row) in live_indexes.iter().zip(rows.into_iter()) {
            if row.afterstates.is_empty() {
                let scores = score_game(&simulations[sim_index].state);
                simulations[sim_index].value = Some(f64::from(scores[root_seat].total));
            } else {
                model_rows.push(row);
                model_row_sims.push(sim_index);
            }
        }
        if model_rows.is_empty() {
            continue;
        }

        let evals = evaluator.evaluate_batch(&model_rows)?;
        if evals.len() != model_rows.len() {
            bail!(
                "leaf evaluator returned {} results for {} rows",
                evals.len(),
                model_rows.len()
            );
        }
        for ((sim_index, row), eval) in model_row_sims
            .iter()
            .cloned()
            .zip(model_rows.into_iter())
            .zip(evals.into_iter())
        {
            if eval.derived_final_q.len() != row.afterstates.len() {
                bail!("evaluator derived_final_q length mismatch");
            }
            let active_seat = row.staged.current_player();
            let simulation = &mut simulations[sim_index];
            let best_index = eval
                .derived_final_q
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| {
                    left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(index, _)| index)
                .context("empty derived_final_q")?;
            if active_seat == root_seat {
                simulation.rounds_left = simulation.rounds_left.saturating_sub(1);
                if simulation.rounds_left == 0 {
                    // Leaf: blend max-Q bootstrap with an optional terminal rollout.
                    let bootstrap = eval.derived_final_q[best_index];
                    let w = cfg.rollout_blend_weight.clamp(0.0, 1.0);
                    let value = if w >= 1.0 {
                        bootstrap
                    } else {
                        let (terminal, _truncated) = complete_with_sampled_greedy(
                            row.staged.clone(),
                            cfg.rollout_max_actions,
                            cfg.rollout_top_k,
                            &mut simulation.rollout_rng,
                            None,
                        )?;
                        let rollout = f64::from(score_game(&terminal)[root_seat].total);
                        w * bootstrap + (1.0 - w) * rollout
                    };
                    simulation.value = Some(value);
                    continue;
                }
            }
            // Advance by the active seat's own argmax derived final Q. The
            // afterstate clone is reused as the next simulation state.
            let mut afterstates = row.afterstates;
            let chosen = afterstates.swap_remove(best_index);
            simulation.state = chosen.state;
        }
    }
}

pub fn gumbel_search(
    root: &EvalRow,
    evaluator: &mut dyn LeafEvaluator,
    cfg: &GumbelConfig,
) -> Result<GumbelSearchResult> {
    if root.afterstates.is_empty() {
        bail!("gumbel_search requires at least one root action");
    }
    let root_seat = root.staged.current_player();

    // Root model evaluation: priors + initial per-action Q estimates.
    let root_eval = evaluator
        .evaluate_batch(std::slice::from_ref(root))?
        .into_iter()
        .next()
        .context("evaluator returned no root output")?;
    if root_eval.priors.len() != root.afterstates.len()
        || root_eval.derived_final_q.len() != root.afterstates.len()
    {
        bail!("root evaluator output misaligned with root menu");
    }
    let action_count = root.afterstates.len();
    let logits: Vec<f64> = root_eval
        .priors
        .iter()
        .map(|prior| prior.max(1e-12).ln())
        .collect();

    // Optional model-ranked root cap (by prior, never greedy rank).
    let mut candidate_indexes: Vec<usize> = (0..action_count).collect();
    if let Some(cap) = cfg.max_root_actions {
        candidate_indexes.sort_by(|&left, &right| {
            root_eval.priors[right]
                .partial_cmp(&root_eval.priors[left])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidate_indexes.truncate(cap.max(1));
    }

    // Gumbel top-m over logits (noise-free when exploration is off).
    let mut gumbel_rng = ChaCha8Rng::seed_from_u64(cfg.search_seed ^ GUMBEL_NOISE_SALT);
    let mut gumbels = vec![0.0_f64; action_count];
    if cfg.exploration {
        for index in 0..action_count {
            gumbels[index] = gumbel_noise(&mut gumbel_rng);
        }
    }
    let top_m = cfg.top_m.max(1).min(candidate_indexes.len());
    candidate_indexes.sort_by(|&left, &right| {
        let left_score = gumbels[left] + logits[left];
        let right_score = gumbels[right] + logits[right];
        right_score
            .partial_cmp(&left_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut survivors: Vec<usize> = candidate_indexes.into_iter().take(top_m).collect();

    let mut visit_counts = vec![0_u32; action_count];
    let mut value_sums = vec![0.0_f64; action_count];
    let mut value_sq_sums = vec![0.0_f64; action_count];
    let mut simulations_run = 0_usize;

    let phase_count = (top_m.max(2) as f64).log2().ceil() as usize;
    let budget = cfg.n_simulations.max(top_m);

    while survivors.len() > 1 || (simulations_run == 0 && !survivors.is_empty()) {
        let per_action = (budget / (phase_count.max(1) * survivors.len())).max(1);
        let mut simulations = Vec::with_capacity(per_action * survivors.len());
        for &action_index in &survivors {
            for _ in 0..per_action {
                if simulations_run + simulations.len() >= budget && visit_counts[action_index] > 0 {
                    break;
                }
                let visit_index = visit_counts[action_index] as usize
                    + simulations.iter().filter(|s: &&Simulation| s.action_index == action_index).count();
                let det_index = (visit_index % cfg.determinization_samples.max(1)) as u64;
                let det_stream = cfg.determinization_seed.unwrap_or(cfg.search_seed);
                let det_seed = splitmix64(
                    det_stream ^ DETERMINIZATION_STREAM_SALT ^ splitmix64(det_index),
                );
                let mut state = root.staged.clone();
                if !cfg.peek_true_hidden {
                    state.redeterminize_hidden(GameSeed::from_u64(det_seed));
                }
                let action = &root.afterstates[action_index].candidate.action;
                let rollout_seed = splitmix64(
                    cfg.search_seed
                        ^ ROLLOUT_STREAM_SALT
                        ^ splitmix64(action_index as u64)
                        ^ splitmix64(0x1_0000 + visit_index as u64),
                );
                let mut simulation = Simulation {
                    action_index,
                    state,
                    rounds_left: cfg.depth_rounds.max(1),
                    rollout_rng: ChaCha8Rng::seed_from_u64(rollout_seed),
                    value: None,
                };
                if let Err(error) = simulation.state.apply(action) {
                    if crate::is_rollout_truncation_rule_error(&error) {
                        let scores = score_game(&simulation.state);
                        simulation.value = Some(f64::from(scores[root_seat].total));
                    } else {
                        return Err(error).context("applying root action in gumbel simulation");
                    }
                }
                simulations.push(simulation);
            }
        }
        advance_simulations(&mut simulations, root_seat, evaluator, cfg)?;
        for simulation in &simulations {
            let value = simulation
                .value
                .context("simulation resolved without a value")?;
            visit_counts[simulation.action_index] += 1;
            value_sums[simulation.action_index] += value;
            value_sq_sums[simulation.action_index] += value * value;
        }
        simulations_run += simulations.len();

        // Halve survivors by g + logits + sigma(mean value).
        if survivors.len() == 1 {
            break;
        }
        let mean_values: Vec<f64> = survivors
            .iter()
            .map(|&action_index| {
                if visit_counts[action_index] == 0 {
                    root_eval.derived_final_q[action_index]
                } else {
                    value_sums[action_index] / f64::from(visit_counts[action_index])
                }
            })
            .collect();
        let normalized = minmax_normalize(&mean_values);
        let max_visits = survivors
            .iter()
            .map(|&action_index| visit_counts[action_index])
            .max()
            .unwrap_or(0);
        let mut scored: Vec<(usize, f64)> = survivors
            .iter()
            .enumerate()
            .map(|(position, &action_index)| {
                let score = gumbels[action_index]
                    + logits[action_index]
                    + sigma(normalized[position], max_visits, cfg.c_visit, cfg.c_scale);
                (action_index, score)
            })
            .collect();
        scored.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let keep = (survivors.len() + 1) / 2;
        survivors = scored.into_iter().take(keep).map(|(index, _)| index).collect();
    }

    let chosen_index = *survivors.first().context("no surviving root action")?;

    // Completed Q over the full root menu.
    let completed_q: Vec<f64> = (0..action_count)
        .map(|action_index| {
            if visit_counts[action_index] == 0 {
                root_eval.derived_final_q[action_index]
            } else {
                value_sums[action_index] / f64::from(visit_counts[action_index])
            }
        })
        .collect();
    let value_variance: Vec<f64> = (0..action_count)
        .map(|action_index| {
            let visits = visit_counts[action_index];
            if visits == 0 {
                0.0
            } else {
                let count = f64::from(visits);
                let mean = value_sums[action_index] / count;
                (value_sq_sums[action_index] / count - mean * mean).max(0.0)
            }
        })
        .collect();
    let normalized_q = minmax_normalize(&completed_q);
    let max_visits = visit_counts.iter().cloned().max().unwrap_or(0);
    let improved_logits: Vec<f64> = (0..action_count)
        .map(|action_index| {
            logits[action_index]
                + sigma(
                    normalized_q[action_index],
                    max_visits,
                    cfg.c_visit,
                    cfg.c_scale,
                )
        })
        .collect();
    let improved_policy = softmax(&improved_logits);
    let root_value = improved_policy
        .iter()
        .zip(completed_q.iter())
        .map(|(weight, q)| weight * q)
        .sum();
    let root_priors = root_eval
        .priors
        .iter()
        .map(|prior| prior.max(0.0))
        .collect();

    Ok(GumbelSearchResult {
        completed_q,
        value_variance,
        improved_policy,
        visit_counts,
        root_priors,
        root_value,
        chosen_index,
        simulations_run,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_game::GameConfig;

    /// Deterministic evaluator: priors proportional to softmax of exact
    /// afterstate scores; per-action Q = exact afterstate score + 1.
    struct MockEvaluator {
        calls: usize,
        rows_seen: usize,
    }

    impl MockEvaluator {
        fn new() -> Self {
            Self {
                calls: 0,
                rows_seen: 0,
            }
        }
    }

    impl LeafEvaluator for MockEvaluator {
        fn evaluate_batch(&mut self, rows: &[EvalRow]) -> Result<Vec<EvalOut>> {
            self.calls += 1;
            self.rows_seen += rows.len();
            Ok(rows
                .iter()
                .map(|row| {
                    // Scale-consistent final-score proxy: exact afterstate
                    // score plus a per-remaining-turn allowance, mirroring how
                    // the real q head predicts *final* score, not immediate.
                    let remaining = row.staged.turns_remaining() as f64;
                    let exact: Vec<f64> = row
                        .afterstates
                        .iter()
                        .map(|afterstate| afterstate.exact_score_active)
                        .collect();
                    let priors = softmax(&exact.iter().map(|value| value / 4.0).collect::<Vec<_>>());
                    let derived_final_q = exact
                        .iter()
                        .map(|value| value + remaining.max(0.0))
                        .collect::<Vec<_>>();
                    EvalOut {
                        priors,
                        derived_final_q,
                    }
                })
                .collect())
        }
    }

    fn test_state(seed_u64: u64, plies: usize) -> GameState {
        let config = GameConfig::research_aaaaa(4).expect("4p config");
        let mut game = GameState::new(config, GameSeed::from_u64(seed_u64)).expect("game");
        let mut rng = ChaCha8Rng::seed_from_u64(seed_u64 ^ 0x1234);
        for _ in 0..plies {
            if game.is_game_over() {
                break;
            }
            let (next, _) =
                complete_with_sampled_greedy(game, 4, 2, &mut rng, Some(1)).expect("advance");
            game = next;
        }
        game
    }

    fn test_config(seed: u64) -> GumbelConfig {
        GumbelConfig {
            n_simulations: 16,
            top_m: 4,
            determinization_samples: 2,
            rollout_blend_weight: 1.0,
            k_interior: 6,
            search_seed: seed,
            ..GumbelConfig::default()
        }
    }

    #[test]
    fn search_budget_and_outputs_are_well_formed() {
        let game = test_state(2_026_070_100, 4);
        let root = eval_row_for_state(&game, None)
            .expect("root row")
            .expect("non-terminal root");
        let action_count = root.afterstates.len();
        let mut evaluator = MockEvaluator::new();
        let result =
            gumbel_search(&root, &mut evaluator, &test_config(7)).expect("search completes");

        assert_eq!(result.completed_q.len(), action_count);
        assert_eq!(result.improved_policy.len(), action_count);
        assert_eq!(result.visit_counts.len(), action_count);
        assert!(result.chosen_index < action_count);
        assert!(result.simulations_run >= 1);
        // Budget overrun is bounded by one per-action allotment per phase.
        assert!(result.simulations_run <= 16 + 4);
        let policy_sum: f64 = result.improved_policy.iter().sum();
        assert!((policy_sum - 1.0).abs() < 1e-9, "policy sums to 1");
        assert!(result.visit_counts[result.chosen_index] > 0);
        // improved[a] is proportional to prior[a] * exp(sigma(q_norm[a])), so
        // the relative upweighting improved/prior must peak exactly at the
        // action with the highest completed Q.
        let best_completed = (0..action_count)
            .max_by(|&left, &right| {
                result.completed_q[left]
                    .partial_cmp(&result.completed_q[right])
                    .unwrap()
            })
            .unwrap();
        let best_ratio = (0..action_count)
            .max_by(|&left, &right| {
                let left_ratio = result.improved_policy[left] / result.root_priors[left].max(1e-12);
                let right_ratio =
                    result.improved_policy[right] / result.root_priors[right].max(1e-12);
                left_ratio.partial_cmp(&right_ratio).unwrap()
            })
            .unwrap();
        assert_eq!(
            best_ratio, best_completed,
            "improved policy must upweight the highest completed-Q action most"
        );
    }

    #[test]
    fn search_is_deterministic_given_seed() {
        let game = test_state(2_026_070_200, 5);
        let root = eval_row_for_state(&game, None)
            .expect("root row")
            .expect("non-terminal root");
        let run = |seed: u64| {
            let mut evaluator = MockEvaluator::new();
            gumbel_search(&root, &mut evaluator, &test_config(seed)).expect("search")
        };
        let first = run(42);
        let second = run(42);
        assert_eq!(first.chosen_index, second.chosen_index);
        assert_eq!(first.visit_counts, second.visit_counts);
        assert_eq!(first.completed_q, second.completed_q);
        assert_eq!(first.improved_policy, second.improved_policy);
    }

    #[test]
    fn search_never_observes_true_hidden_order() {
        let game = test_state(2_026_070_300, 6);
        let (_prelude, staged) = game
            .preview_free_three_of_a_kind_if_feasible()
            .expect("staged");
        let mut permuted_game = staged.clone();
        permuted_game.redeterminize_hidden(GameSeed::from_u64(0xfeed_f00d));
        assert_eq!(
            staged.public_state().canonical_hash(),
            permuted_game.public_state().canonical_hash(),
            "redeterminization must preserve public state"
        );

        let run = |state: &GameState| {
            let root = eval_row_for_state(state, None)
                .expect("root row")
                .expect("non-terminal root");
            let mut evaluator = MockEvaluator::new();
            gumbel_search(&root, &mut evaluator, &test_config(9)).expect("search")
        };
        // Both a blended and a pure-bootstrap config must be invariant.
        let original = run(&staged);
        let permuted = run(&permuted_game);
        assert_eq!(original.chosen_index, permuted.chosen_index);
        assert_eq!(original.completed_q, permuted.completed_q);
        assert_eq!(original.improved_policy, permuted.improved_policy);

        let run_blended = |state: &GameState| {
            let root = eval_row_for_state(state, None)
                .expect("root row")
                .expect("non-terminal root");
            let mut evaluator = MockEvaluator::new();
            let mut cfg = test_config(9);
            cfg.rollout_blend_weight = 0.5;
            gumbel_search(&root, &mut evaluator, &cfg).expect("search")
        };
        let original_blended = run_blended(&staged);
        let permuted_blended = run_blended(&permuted_game);
        assert_eq!(original_blended.completed_q, permuted_blended.completed_q);
    }

    #[test]
    fn peek_mode_observes_true_hidden_order() {
        // Inverse of the no-peek invariance test: with peek_true_hidden the
        // rollout leaves run on the true hidden order, so permuting it must
        // change the search result. Guards against the flag silently becoming
        // a no-op (which would invalidate ceiling measurements).
        let game = test_state(2_026_070_300, 6);
        let (_prelude, staged) = game
            .preview_free_three_of_a_kind_if_feasible()
            .expect("staged");
        let mut permuted_game = staged.clone();
        permuted_game.redeterminize_hidden(GameSeed::from_u64(0xfeed_f00d));

        let run_peek = |state: &GameState| {
            let root = eval_row_for_state(state, None)
                .expect("root row")
                .expect("non-terminal root");
            let mut evaluator = MockEvaluator::new();
            let mut cfg = test_config(9);
            cfg.rollout_blend_weight = 0.0; // leaf = pure rollout on the (peeked) state
            cfg.peek_true_hidden = true;
            gumbel_search(&root, &mut evaluator, &cfg).expect("search")
        };
        let original = run_peek(&staged);
        let permuted = run_peek(&permuted_game);
        assert_ne!(
            original.completed_q, permuted.completed_q,
            "peek search must depend on the true hidden order"
        );
    }

    #[test]
    fn exploration_off_consumes_no_gumbel_noise() {
        let game = test_state(2_026_070_400, 4);
        let root = eval_row_for_state(&game, None)
            .expect("root row")
            .expect("non-terminal root");
        let run = |search_seed: u64, exploration: bool| {
            let mut evaluator = MockEvaluator::new();
            let mut cfg = test_config(search_seed);
            cfg.exploration = exploration;
            // Pin the determinization and rollout inputs so the only varying
            // stream across search seeds is the Gumbel noise itself.
            cfg.determinization_seed = Some(99);
            cfg.rollout_blend_weight = 1.0;
            gumbel_search(&root, &mut evaluator, &cfg).expect("search")
        };
        // With exploration off, different search seeds produce identical
        // results because the noise stream is never sampled.
        let first = run(1, false);
        let second = run(2, false);
        assert_eq!(first.chosen_index, second.chosen_index);
        assert_eq!(first.visit_counts, second.visit_counts);
        assert_eq!(first.completed_q, second.completed_q);
        assert_eq!(first.improved_policy, second.improved_policy);
    }

    #[test]
    fn near_terminal_roots_still_resolve() {
        // Advance deep into the game, then search close to the end.
        let mut game = test_state(2_026_070_500, 70);
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        while !game.is_game_over() && game.turns_remaining() > 2 {
            let (next, _) =
                complete_with_sampled_greedy(game, 4, 1, &mut rng, Some(1)).expect("advance");
            game = next;
        }
        if game.is_game_over() {
            return; // seed ended early; nothing to assert
        }
        let root = eval_row_for_state(&game, None).expect("root row");
        let Some(root) = root else {
            return;
        };
        let mut evaluator = MockEvaluator::new();
        let mut cfg = test_config(11);
        cfg.rollout_blend_weight = 0.5;
        let result = gumbel_search(&root, &mut evaluator, &cfg).expect("search near terminal");
        assert!(result.chosen_index < root.afterstates.len());
        assert!(result.simulations_run >= 1);
    }
}
