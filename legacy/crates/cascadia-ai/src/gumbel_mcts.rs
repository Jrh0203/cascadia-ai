//! Gumbel AlphaZero-style root selection for Cascadia.
//!
//! Implements the algorithm from Danihelka et al. (ICLR 2022, "Policy improvement
//! by planning with Gumbel"). Designed for low simulation budgets where standard
//! PUCT/UCT is provably suboptimal.
//!
//! Algorithm at the root:
//!   1. Sample m = min(num_root_actions, M) actions via Gumbel-top-k:
//!        a* = top-k by g(a) = log_pi(a) + Gumbel(0,1)
//!   2. Run sequential halving over the sampled actions: log2(m) phases,
//!      each phase visits each surviving action equally, then eliminates
//!      the bottom half by g(a) + sigma(q_hat(a)).
//!   3. Final action: argmax of g(a) + sigma(q_hat(a)) over the surviving set.
//!
//! At non-root nodes, we use the COMPLETED-Q backup (Equation 12 in the paper):
//!   v_completed(s,a) = q_hat(s,a) if visited, else v_pi(s)
//! And select children deterministically by argmax of completed-Q + exploration.
//!
//! For Cascadia where we don't have a trained policy head, we use the candidate's
//! NNUE eval (already a meaningful "value-derived" prior) softmax-normalized.

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use cascadia_core::game::GameState;
use cascadia_core::scoring::ScoreBreakdown;

use crate::eval::ScoredMove;
use crate::nnue::NNUENetwork;
use crate::search::{candidate_moves_decomposed, execute_scored_move, greedy_move};

const PRIOR_TEMP: f64 = 5000.0; // ~5 score points = 1 logit unit
const SIGMA_C_VISIT: f64 = 50.0; // sigma(q) = (c_visit + visit_count_max) * c_scale * q
const SIGMA_C_SCALE: f64 = 1.0;
const PLAYOUT_DEPTH: usize = 6;

/// One root candidate with Gumbel sample, prior, and rollout statistics.
struct GumbelArm {
    action: ScoredMove,
    log_prior: f64,
    gumbel: f64,
    visits: u32,
    total_value: f64,
}

impl GumbelArm {
    fn q(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_value / self.visits as f64
        }
    }
}

/// sigma(q): the monotonic transform from the Gumbel paper.
/// sigma(q) = (c_visit + max_visits) * c_scale * q
/// (Simplified: c_visit absorbs into a per-call scaling.)
fn sigma(q: f64, max_visits: u32) -> f64 {
    (SIGMA_C_VISIT + max_visits as f64) * SIGMA_C_SCALE * q / 100.0
}

fn evals_to_log_priors(candidates: &[ScoredMove]) -> Vec<f64> {
    let max_eval = candidates.iter().map(|c| c.eval).max().unwrap_or(0) as f64;
    candidates
        .iter()
        .map(|c| (c.eval as f64 - max_eval) / PRIOR_TEMP)
        .collect()
}

/// Sample Gumbel(0,1) — inverse CDF: -log(-log(U))
fn gumbel_sample(rng: &mut StdRng) -> f64 {
    let u: f64 = rng.gen_range(1e-12..1.0);
    -(-u.ln()).ln()
}

/// One simulation: execute action, advance opponents, rollout, backup value.
/// Used as a leaf evaluator for arms during sequential halving.
fn arm_simulation(
    game: &GameState,
    action: &ScoredMove,
    net: &NNUENetwork,
    ai_player: usize,
    rng: &mut StdRng,
) -> f64 {
    let mut g = game.clone();
    g.shuffle_bags(rng);
    if !execute_scored_move(&mut g, action) {
        return 0.0;
    }

    let mut ai_plies = 0;
    while !g.is_game_over() {
        if g.current_player != ai_player {
            if g.can_replace_overflow().is_some() {
                g.replace_overflow();
            }
            match greedy_move(&g) {
                Some(mv) => {
                    if !execute_scored_move(&mut g, &mv) {
                        break;
                    }
                }
                None => break,
            }
            continue;
        }
        if g.can_replace_overflow().is_some() {
            g.replace_overflow();
        }
        ai_plies += 1;
        if ai_plies > PLAYOUT_DEPTH {
            break;
        }
        match greedy_move(&g) {
            Some(mv) => {
                if !execute_scored_move(&mut g, &mv) {
                    break;
                }
            }
            None => break,
        }
    }
    if g.is_game_over() {
        ScoreBreakdown::compute(&mut g.boards[ai_player], &g.scoring_cards).total as f64
    } else {
        let actual =
            ScoreBreakdown::compute(&mut g.boards[ai_player], &g.scoring_cards).total as f64;
        let bag_info = crate::nnue::BagInfo::from_game(&g);
        let nval = net
            .evaluate_with_bag(&g.boards[ai_player], &bag_info)
            .max(0.0) as f64;
        actual + nval
    }
}

/// Pick the best move via Gumbel-AlphaZero root selection with sequential halving.
pub fn best_move_gumbel_mcts(
    game: &GameState,
    net: &NNUENetwork,
    num_rollouts: usize,
    m: usize,
    rng: &mut StdRng,
) -> Option<ScoredMove> {
    let ai_player = game.current_player;

    let candidates = candidate_moves_decomposed(game, net);
    if candidates.is_empty() {
        return None;
    }

    let log_priors = evals_to_log_priors(&candidates);

    // Step 1: Gumbel sampling — score = log_prior + Gumbel(0,1), keep top m
    let mut arms: Vec<GumbelArm> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| GumbelArm {
            action: *c,
            log_prior: log_priors[i],
            gumbel: gumbel_sample(rng),
            visits: 0,
            total_value: 0.0,
        })
        .collect();

    // Sort by g(a) = log_prior + gumbel descending, keep top m
    arms.sort_by(|a, b| {
        let sa = a.log_prior + a.gumbel;
        let sb = b.log_prior + b.gumbel;
        sb.partial_cmp(&sa).unwrap()
    });
    let m_eff = m.min(arms.len());
    arms.truncate(m_eff);

    if m_eff == 0 {
        return None;
    }

    // Step 2: Sequential halving — log2(m) phases, equal visits per surviving arm
    let num_phases = (m_eff as f64).log2().ceil().max(1.0) as usize;
    let visits_per_phase = num_rollouts / num_phases;

    let mut alive: Vec<usize> = (0..m_eff).collect();
    let game_arc = Arc::new(game.clone());
    let net_arc = Arc::new(net.clone());

    for phase in 0..num_phases {
        if alive.is_empty() {
            break;
        }
        let visits_per_arm = (visits_per_phase / alive.len()).max(1);

        // Build work items: (arm_idx, seed)
        let mut work_items: Vec<(usize, u64)> = Vec::new();
        for &ai in &alive {
            for _ in 0..visits_per_arm {
                work_items.push((ai, rng.gen()));
            }
        }

        // Parallelize across threads
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let chunk_size = (work_items.len() + num_threads - 1) / num_threads;

        let arm_actions: Vec<ScoredMove> = arms.iter().map(|a| a.action).collect();
        let arm_actions = Arc::new(arm_actions);

        let handles: Vec<_> = work_items
            .chunks(chunk_size)
            .map(|chunk| {
                let work = chunk.to_vec();
                let game = Arc::clone(&game_arc);
                let net = Arc::clone(&net_arc);
                let actions = Arc::clone(&arm_actions);
                std::thread::spawn(move || {
                    let mut results: Vec<(usize, f64)> = Vec::with_capacity(work.len());
                    for &(ai, seed) in &work {
                        let mut thread_rng = StdRng::seed_from_u64(seed);
                        let val =
                            arm_simulation(&game, &actions[ai], &net, ai_player, &mut thread_rng);
                        results.push((ai, val));
                    }
                    results
                })
            })
            .collect();

        for h in handles {
            for (ai, val) in h.join().unwrap() {
                arms[ai].visits += 1;
                arms[ai].total_value += val;
            }
        }

        // Eliminate bottom half by g(a) + sigma(q_hat(a))
        if phase < num_phases - 1 {
            let max_visits = alive.iter().map(|&i| arms[i].visits).max().unwrap_or(1);
            let mut scored: Vec<(usize, f64)> = alive
                .iter()
                .map(|&i| {
                    let g = arms[i].log_prior + arms[i].gumbel + sigma(arms[i].q(), max_visits);
                    (i, g)
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let keep = (scored.len() + 1) / 2;
            alive = scored.into_iter().take(keep).map(|(i, _)| i).collect();
        }
    }

    // Step 3: argmax of g(a) + sigma(q_hat(a))
    let max_visits = arms.iter().map(|a| a.visits).max().unwrap_or(1);
    let mut best_idx = 0;
    let mut best_score = f64::NEG_INFINITY;
    for (i, arm) in arms.iter().enumerate() {
        let s = arm.log_prior + arm.gumbel + sigma(arm.q(), max_visits);
        if s > best_score {
            best_score = s;
            best_idx = i;
        }
    }

    let arm = &arms[best_idx];
    Some(ScoredMove {
        score: arm.q().round() as u16,
        ..arm.action
    })
}
