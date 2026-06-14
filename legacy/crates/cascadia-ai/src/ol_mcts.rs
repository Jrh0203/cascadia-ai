//! Open-Loop Monte Carlo Tree Search for Cascadia.
//!
//! Tree nodes represent action sequences (not states). Chance is sampled per
//! rollout: at each chance node (bag refill, opponent moves) we re-sample, but
//! the tree structure persists across rollouts so each new rollout benefits
//! from the accumulated Q-values of prior rollouts.
//!
//! Key differences from MCE:
//!   - PUCT-style adaptive budget allocation (instead of sequential halving)
//!   - Multi-ply tree (each rollout descends into the tree, not just the root)
//!   - Cross-rollout information sharing — Q estimates inform later rollouts
//!
//! The candidate generation at depth>0 happens lazily on first visit. To handle
//! Cascadia's stochastic action set (legal moves change with each market refill),
//! children are NOT shared across chance outcomes — each rollout that visits a
//! depth-2 node may see different candidates. The tree records the FIRST set of
//! candidates seen and reuses that node only on rollouts where the same actions
//! happen to be legal again.
//!
//! Practical depth: 3 plies of search. Beyond that the value of the tree
//! structure diminishes vs simple heuristic playouts.

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use cascadia_core::game::GameState;
use cascadia_core::scoring::ScoreBreakdown;

use crate::eval::ScoredMove;
use crate::nnue::NNUENetwork;
use crate::search::{candidate_moves_decomposed, execute_scored_move, greedy_move};

const C_PUCT: f64 = 1.4;
const MAX_TREE_DEPTH: usize = 3;
const PLAYOUT_DEPTH: usize = 6;

struct OLNode {
    visits: u32,
    children: Vec<OLEdge>,
}

struct OLEdge {
    action: ScoredMove,
    visits: u32,
    total_value: f64,
    prior: f64,
    child: Option<Box<OLNode>>,
}

impl OLEdge {
    fn q(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_value / self.visits as f64
        }
    }
    fn puct(&self, parent_visits: u32) -> f64 {
        let q = self.q();
        let u = C_PUCT * self.prior * (parent_visits as f64).sqrt() / (1.0 + self.visits as f64);
        q + u
    }
}

impl OLNode {
    fn new() -> Self {
        OLNode {
            visits: 0,
            children: Vec::new(),
        }
    }
}

/// Compute softmax priors from candidate evals (already includes NNUE afterstate score).
fn evals_to_priors(candidates: &[ScoredMove]) -> Vec<f64> {
    if candidates.is_empty() {
        return Vec::new();
    }
    let max_eval = candidates.iter().map(|c| c.eval).max().unwrap() as f64;
    let temperature = 5000.0; // ~5 score points
    let exps: Vec<f64> = candidates
        .iter()
        .map(|c| ((c.eval as f64 - max_eval) / temperature).exp())
        .collect();
    let z: f64 = exps.iter().sum();
    if z > 0.0 {
        exps.iter().map(|&e| e / z).collect()
    } else {
        vec![1.0 / candidates.len() as f64; candidates.len()]
    }
}

/// Heuristic playout from `g` to depth `play_depth` AI plies, returning leaf value.
fn playout_value(g: &mut GameState, net: &NNUENetwork, ai_player: usize, play_depth: usize) -> f64 {
    let mut ai_plies = 0;
    while !g.is_game_over() {
        if g.current_player != ai_player {
            if g.can_replace_overflow().is_some() {
                g.replace_overflow();
            }
            match greedy_move(g) {
                Some(mv) => {
                    if !execute_scored_move(g, &mv) {
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
        if ai_plies > play_depth {
            break;
        }
        match greedy_move(g) {
            Some(mv) => {
                if !execute_scored_move(g, &mv) {
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
        let bag_info = crate::nnue::BagInfo::from_game(g);
        let nval = net
            .evaluate_with_bag(&g.boards[ai_player], &bag_info)
            .max(0.0) as f64;
        actual + nval
    }
}

/// One simulation: descend the tree picking PUCT-best child, then playout.
/// Returns the value backed up to the root.
fn simulate(
    node: &mut OLNode,
    g: &mut GameState,
    net: &NNUENetwork,
    ai_player: usize,
    depth: usize,
    rng: &mut StdRng,
) -> f64 {
    // Terminal or beyond max tree depth: leaf eval via playout
    if g.is_game_over() || depth >= MAX_TREE_DEPTH {
        let value = playout_value(&mut g.clone(), net, ai_player, PLAYOUT_DEPTH);
        node.visits += 1;
        return value;
    }

    // Expand if first visit
    if node.children.is_empty() {
        let cands = candidate_moves_decomposed(g, net);
        if cands.is_empty() {
            let value = playout_value(&mut g.clone(), net, ai_player, PLAYOUT_DEPTH);
            node.visits += 1;
            return value;
        }
        let priors = evals_to_priors(&cands);
        for (i, c) in cands.iter().enumerate() {
            node.children.push(OLEdge {
                action: *c,
                visits: 0,
                total_value: 0.0,
                prior: priors[i],
                child: None,
            });
        }
    }

    if node.children.is_empty() {
        let value = playout_value(&mut g.clone(), net, ai_player, PLAYOUT_DEPTH);
        node.visits += 1;
        return value;
    }

    // PUCT select
    let parent_visits = node.visits.max(1);
    let mut best_idx = 0;
    let mut best_score = f64::NEG_INFINITY;
    for (i, edge) in node.children.iter().enumerate() {
        let s = edge.puct(parent_visits);
        if s > best_score {
            best_score = s;
            best_idx = i;
        }
    }

    // Apply the action and advance state with chance sampling
    let action = node.children[best_idx].action;
    if !execute_scored_move(g, &action) {
        node.visits += 1;
        return 0.0;
    }
    g.shuffle_bags(rng); // chance node — fresh sample per visit

    // Advance opponents (greedy with free-replace)
    crate::search::advance_opponents(g, ai_player);

    // Recurse
    let edge = &mut node.children[best_idx];
    if edge.child.is_none() {
        edge.child = Some(Box::new(OLNode::new()));
    }
    let value = simulate(
        edge.child.as_mut().unwrap(),
        g,
        net,
        ai_player,
        depth + 1,
        rng,
    );

    // Backup
    edge.visits += 1;
    edge.total_value += value;
    node.visits += 1;
    value
}

/// Pick the best move via Open-Loop MCTS with single shared tree + leaf parallelization.
///
/// Design: a single tree shared via leaf parallelization (with virtual loss). The root
/// is expanded once. Each iteration: SELECT best leaf via PUCT, EXPAND it, then submit
/// the new leaf to a thread pool for playout. Backups are applied as playouts complete.
///
/// Simpler v2: do ROLLOUT-level parallelism with virtual-loss-style PUCT to keep the
/// tree growing. Each "batch" picks K leaves in sequence with virtual loss, dispatches
/// playouts to threads, then back-up at completion. Repeat until budget exhausted.
pub fn best_move_ol_mcts(
    game: &GameState,
    net: &NNUENetwork,
    num_rollouts: usize,
    rng: &mut StdRng,
) -> Option<ScoredMove> {
    let ai_player = game.current_player;
    let mut root = OLNode::new();

    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let net_arc = Arc::new(net.clone());

    let mut rollouts_done = 0;
    while rollouts_done < num_rollouts {
        // Collect a batch of leaf jobs by sequential PUCT selection with virtual loss
        let batch_size = num_threads.min(num_rollouts - rollouts_done);
        let mut jobs: Vec<(Vec<usize>, GameState)> = Vec::new(); // (path indices, leaf game)

        for _ in 0..batch_size {
            let mut g = game.clone();
            let mut path: Vec<usize> = Vec::new();
            let mut node_ptr: *mut OLNode = &mut root;
            let mut depth = 0;
            // Descend until reaching an unexpanded or terminal node
            loop {
                let node = unsafe { &mut *node_ptr };
                if g.is_game_over() || depth >= MAX_TREE_DEPTH {
                    break;
                }
                if node.children.is_empty() {
                    // Expand
                    let cands = candidate_moves_decomposed(&g, net);
                    if cands.is_empty() {
                        break;
                    }
                    let priors = evals_to_priors(&cands);
                    for (i, c) in cands.iter().enumerate() {
                        node.children.push(OLEdge {
                            action: *c,
                            visits: 0,
                            total_value: 0.0,
                            prior: priors[i],
                            child: None,
                        });
                    }
                }
                if node.children.is_empty() {
                    break;
                }
                let parent_visits = node.visits.max(1);
                let mut best_idx = 0usize;
                let mut best_score = f64::NEG_INFINITY;
                for (i, e) in node.children.iter().enumerate() {
                    let s = e.puct(parent_visits);
                    if s > best_score {
                        best_score = s;
                        best_idx = i;
                    }
                }
                // Apply virtual loss: increment visits (it'll be undone when real result comes)
                node.children[best_idx].visits += 1;
                node.visits += 1;
                path.push(best_idx);
                let action = node.children[best_idx].action;
                if !execute_scored_move(&mut g, &action) {
                    break;
                }
                g.shuffle_bags(rng);
                crate::search::advance_opponents(&mut g, ai_player);
                if node.children[best_idx].child.is_none() {
                    node.children[best_idx].child = Some(Box::new(OLNode::new()));
                }
                node_ptr = node.children[best_idx].child.as_mut().unwrap().as_mut();
                depth += 1;
            }
            jobs.push((path, g));
        }

        // Dispatch playouts in parallel
        let net = Arc::clone(&net_arc);
        let handles: Vec<_> = jobs
            .iter()
            .map(|(_path, g)| {
                let net = Arc::clone(&net);
                let g = g.clone();
                std::thread::spawn(move || {
                    playout_value(&mut g.clone(), &net, ai_player, PLAYOUT_DEPTH)
                })
            })
            .collect();

        let values: Vec<f64> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // Backup with virtual-loss correction (subtract the +1 we added during selection)
        for ((path, _), value) in jobs.into_iter().zip(values.into_iter()) {
            // Walk the path and update edges
            let mut node_ptr: *mut OLNode = &mut root;
            for &idx in &path {
                let node = unsafe { &mut *node_ptr };
                // Virtual loss already incremented visits — keep them, but add real value
                node.children[idx].total_value += value;
                if let Some(ref mut child) = node.children[idx].child {
                    node_ptr = child.as_mut();
                }
            }
        }
        rollouts_done += batch_size;
    }

    // Pick the most-visited root child
    root.children.sort_by(|a, b| b.visits.cmp(&a.visits));
    root.children.into_iter().next().map(|edge| {
        let avg = if edge.visits > 0 {
            edge.total_value / edge.visits as f64
        } else {
            0.0
        };
        ScoredMove {
            score: avg.round() as u16,
            ..edge.action
        }
    })
}
