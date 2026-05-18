//! Multi-turn UCT MCTS for Cascadia (pure greedy, no value network).
//!
//! Classical UCT (Kocsis & Szepesvári 2006) with open-loop bag resampling:
//! - Tree nodes represent action sequences (not states). Each node stores
//!   N (visit count) and Q (mean backed-up value).
//! - Selection: UCB1 descent — pick child maximizing Q + c*sqrt(ln(N_parent)/N_child).
//! - Expansion: on reaching an unexpanded leaf, generate candidate moves.
//! - Simulation: greedy rollout to game end from new leaf.
//! - Backup: propagate leaf's final AI score up the full path.
//!
//! Chance handling: each descent re-shuffles the bag at chance points
//! (open-loop MCTS). Different descents see different futures, which is
//! correct for stochastic environments like Cascadia.
//!
//! Rollouts use pure greedy for all players. Opponents play greedy with
//! free-replace (matching our standard benchmark behavior).

use std::sync::{Arc, Mutex};
use std::thread;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use cascadia_core::game::GameState;
use cascadia_core::scoring::ScoreBreakdown;

use crate::eval::ScoredMove;
use crate::search::{execute_scored_move, greedy_move};

const C_UCB: f64 = 14.0;  // exploration constant tuned for Cascadia's score range
const MAX_TREE_DEPTH: usize = 4;  // AI plies of tree search before switching to rollouts

/// Tree node. Each node owns its children and tracks N, Q.
struct Node {
    visits: u32,
    total_value: u64,
    /// Edge to each child: action + child subtree. Built lazily on first visit.
    edges: Vec<Edge>,
    expanded: bool,
}

struct Edge {
    action: ScoredMove,
    visits: u32,
    total_value: u64,
    /// Child subtree. Lazy-init on second visit (first visit is a rollout).
    child: Option<Box<Node>>,
}

impl Node {
    fn new() -> Self {
        Node { visits: 0, total_value: 0, edges: Vec::new(), expanded: false }
    }
}

impl Edge {
    fn q(&self) -> f64 {
        if self.visits == 0 { 0.0 } else { self.total_value as f64 / self.visits as f64 }
    }

    fn ucb(&self, parent_visits: u32) -> f64 {
        if self.visits == 0 { return f64::INFINITY; }
        let q = self.q();
        let explore = C_UCB * ((parent_visits as f64).ln() / self.visits as f64).sqrt();
        q + explore
    }
}

/// Advance the game through opponents' turns (all greedy with free-replace).
fn advance_opponents(game: &mut GameState, ai_player: usize) {
    while !game.is_game_over() && game.current_player != ai_player {
        if game.can_replace_overflow().is_some() {
            game.replace_overflow();
        }
        match greedy_move(game) {
            Some(mv) => { if !execute_scored_move(game, &mv) { break; } }
            None => break,
        }
    }
}

/// Pure greedy rollout to game end. Returns AI's final score.
fn greedy_rollout(mut g: GameState, ai_player: usize) -> u64 {
    while !g.is_game_over() {
        if g.can_replace_overflow().is_some() {
            g.replace_overflow();
        }
        match greedy_move(&g) {
            Some(mv) => { if !execute_scored_move(&mut g, &mv) { break; } }
            None => break,
        }
    }
    ScoreBreakdown::compute(
        &mut g.boards[ai_player], &g.scoring_cards,
    ).total as u64
}

/// Generate candidates for AI decisions at an interior tree node.
/// Uses the same candidate pool as greedy MCE for parity.
fn root_candidates(game: &GameState) -> Vec<ScoredMove> {
    crate::mce::default_greedy_mce_candidates(game)
}

/// One simulation: descend tree via UCB, expand new leaf, rollout, backup.
/// Returns the leaf value (AI's final score).
fn simulate(
    node: &mut Node,
    game: &GameState,
    ai_player: usize,
    rng: &mut StdRng,
    depth: usize,
) -> u64 {
    // Terminal
    if game.is_game_over() {
        let score = ScoreBreakdown::compute(
            &mut game.boards[ai_player].clone(), &game.scoring_cards,
        ).total as u64;
        node.visits += 1;
        node.total_value += score;
        return score;
    }

    // Beyond tree depth: just rollout
    if depth >= MAX_TREE_DEPTH {
        let score = greedy_rollout(game.clone(), ai_player);
        node.visits += 1;
        node.total_value += score;
        return score;
    }

    // Expand if first time
    if !node.expanded {
        let cands = root_candidates(game);
        if cands.is_empty() {
            let score = greedy_rollout(game.clone(), ai_player);
            node.visits += 1;
            node.total_value += score;
            return score;
        }
        for mv in cands {
            node.edges.push(Edge {
                action: mv, visits: 0, total_value: 0, child: None,
            });
        }
        node.expanded = true;
    }

    if node.edges.is_empty() {
        let score = greedy_rollout(game.clone(), ai_player);
        node.visits += 1;
        node.total_value += score;
        return score;
    }

    // UCB1 selection
    let parent_visits = node.visits.max(1);
    let mut best_idx = 0;
    let mut best_ucb = f64::NEG_INFINITY;
    for (i, e) in node.edges.iter().enumerate() {
        let u = e.ucb(parent_visits);
        if u > best_ucb { best_ucb = u; best_idx = i; }
    }

    // Apply chosen action + opponents + bag shuffle (open-loop chance node)
    let action = node.edges[best_idx].action;
    let mut g = game.clone();
    g.shuffle_bags(rng);
    if !execute_scored_move(&mut g, &action) {
        // Infeasible: mark this edge as visited with rollout value, don't expand further
        let score = greedy_rollout(game.clone(), ai_player);
        let edge = &mut node.edges[best_idx];
        edge.visits += 1;
        edge.total_value += score;
        node.visits += 1;
        node.total_value += score;
        return score;
    }
    advance_opponents(&mut g, ai_player);

    // Recurse or rollout based on first visit to this edge
    let value = {
        let edge = &mut node.edges[best_idx];
        if edge.visits == 0 {
            // First visit: rollout from here (don't expand child yet)
            greedy_rollout(g, ai_player)
        } else {
            // Subsequent visits: create/descend into child node
            if edge.child.is_none() {
                edge.child = Some(Box::new(Node::new()));
            }
            simulate(edge.child.as_mut().unwrap(), &g, ai_player, rng, depth + 1)
        }
    };

    // Backup
    let edge = &mut node.edges[best_idx];
    edge.visits += 1;
    edge.total_value += value;
    node.visits += 1;
    node.total_value += value;
    value
}

/// Pick the best root move via open-loop UCT MCTS with greedy rollouts.
///
/// `num_simulations` is the total simulation budget. Tree grows up to
/// MAX_TREE_DEPTH AI plies; leaves use greedy rollout to game end.
///
/// Single-threaded tree, but can be called from multiple threads for
/// "root parallelism" if desired. Currently single-tree single-thread.
pub fn best_move_uct_mcts(
    game: &GameState,
    num_simulations: usize,
    rng: &mut StdRng,
) -> Option<ScoredMove> {
    let ai_player = game.current_player;
    let mut root = Node::new();

    for _ in 0..num_simulations {
        simulate(&mut root, game, ai_player, rng, 0);
    }

    if root.edges.is_empty() { return None; }

    // Select most-visited root child (robust choice, not highest-Q)
    let mut best_idx = 0;
    let mut best_visits = 0u32;
    let mut best_q = f64::NEG_INFINITY;
    for (i, e) in root.edges.iter().enumerate() {
        if e.visits > best_visits || (e.visits == best_visits && e.q() > best_q) {
            best_visits = e.visits;
            best_q = e.q();
            best_idx = i;
        }
    }

    let e = &root.edges[best_idx];
    Some(ScoredMove { score: e.q().round() as u16, ..e.action })
}

/// Root-parallelized UCT: run K independent trees, pick the move most-voted
/// as best across them. Simpler than tree-parallelism (no locks needed).
/// Each tree gets `num_simulations / K` sims.
pub fn best_move_uct_mcts_parallel(
    game: &GameState,
    num_simulations: usize,
    rng: &mut StdRng,
) -> Option<ScoredMove> {
    let num_threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let sims_per_thread = (num_simulations + num_threads - 1) / num_threads;

    let game_arc = Arc::new(game.clone());
    let ai_player = game.current_player;

    let handles: Vec<_> = (0..num_threads).map(|t| {
        let g = Arc::clone(&game_arc);
        let seed = rng.gen::<u64>().wrapping_add(t as u64);
        thread::spawn(move || {
            let mut thread_rng = StdRng::seed_from_u64(seed);
            let mut root = Node::new();
            for _ in 0..sims_per_thread {
                simulate(&mut root, &g, ai_player, &mut thread_rng, 0);
            }
            // Return (action, visits, total_value) for each root edge
            root.edges.into_iter()
                .map(|e| (e.action, e.visits, e.total_value))
                .collect::<Vec<_>>()
        })
    }).collect();

    // Aggregate stats across trees: sum visits and totals per unique action
    use std::collections::HashMap;
    let mut agg: HashMap<(usize, Option<usize>, i8, i8, Option<i8>, Option<i8>, u8),
                         (ScoredMove, u64, u64)> = HashMap::new();
    for h in handles {
        let tree_edges = h.join().unwrap();
        for (action, v, tv) in tree_edges {
            let key = (action.market_index, action.wildlife_market_index,
                       action.tile_q, action.tile_r,
                       action.wildlife_q, action.wildlife_r, action.rotation);
            let entry = agg.entry(key).or_insert((action, 0, 0));
            entry.1 += v as u64;
            entry.2 += tv;
        }
    }

    // Pick action with highest visit count; tiebreak by average value
    let mut best: Option<(ScoredMove, u64, f64)> = None;
    for (_, (action, v, tv)) in agg {
        let q = if v > 0 { tv as f64 / v as f64 } else { 0.0 };
        match &best {
            None => best = Some((action, v, q)),
            Some((_, bv, bq)) => {
                if v > *bv || (v == *bv && q > *bq) {
                    best = Some((action, v, q));
                }
            }
        }
    }

    best.map(|(mv, _v, q)| ScoredMove { score: q.round() as u16, ..mv })
}
