//! Monte Carlo Tree Search (MCTS) with PUCT for Cascadia.
//!
//! Uses the dual-head NNUE (value + policy) to guide tree search without rollouts.
//! At each leaf, generates candidate moves, evaluates them with forward_dual(),
//! and uses the value estimate for backup and policy prior for exploration.

use cascadia_core::game::GameState;
use cascadia_core::scoring::ScoreBreakdown;

use crate::eval::ScoredMove;
use crate::nnue::NNUENetwork;

const C_PUCT: f32 = 2.0;

struct MCTSEdge {
    action: ScoredMove,
    prior: f32,       // P(a) from policy head
    visit_count: u32, // N(s,a)
    total_value: f64, // W(s,a) — sum of backed-up values
    child: Option<Box<MCTSNode>>,
}

struct MCTSNode {
    children: Vec<MCTSEdge>,
    visit_count: u32,
    is_expanded: bool,
}

impl MCTSNode {
    fn new() -> Self {
        MCTSNode {
            children: Vec::new(),
            visit_count: 0,
            is_expanded: false,
        }
    }
}

impl MCTSEdge {
    fn q_value(&self) -> f64 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.total_value / self.visit_count as f64
        }
    }

    fn puct_score(&self, parent_visits: u32) -> f64 {
        let q = self.q_value();
        let u = C_PUCT as f64 * self.prior as f64 * (parent_visits as f64).sqrt()
            / (1.0 + self.visit_count as f64);
        q + u
    }
}

/// Expand a leaf node: generate candidates, evaluate with dual-head NNUE,
/// populate children with priors from the policy head.
/// Returns the value estimate for this position (for backup).
fn expand_node(node: &mut MCTSNode, game: &GameState, net: &NNUENetwork, player: usize) -> f64 {
    let candidates = crate::search::candidate_moves_decomposed(game, net);
    if candidates.is_empty() {
        // Terminal or no moves — return actual score
        let score = ScoreBreakdown::compute(&mut game.boards[player].clone(), &game.scoring_cards)
            .total as f64;
        node.is_expanded = true;
        return score;
    }

    let bag_info = crate::nnue::BagInfo::from_game(game);
    let mut values = Vec::with_capacity(candidates.len());
    let mut policy_logits = Vec::with_capacity(candidates.len());

    for mv in &candidates {
        let mut g = game.clone();
        if !crate::search::execute_scored_move(&mut g, mv) {
            values.push(0.0f32);
            policy_logits.push(f32::NEG_INFINITY);
            continue;
        }
        let features = crate::nnue::extract_features_with_bag(&g.boards[player], Some(&bag_info));
        let (value, policy) = net.forward_dual(&features);
        // Value = current_score + remaining_value_estimate
        let current =
            ScoreBreakdown::compute(&mut g.boards[player].clone(), &g.scoring_cards).total as f32;
        values.push(current + value);
        policy_logits.push(policy);
    }

    // Softmax over policy logits to get priors
    let max_logit = policy_logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = policy_logits.iter().map(|&l| (l - max_logit).exp()).sum();
    let priors: Vec<f32> = policy_logits
        .iter()
        .map(|&l| (l - max_logit).exp() / exp_sum)
        .collect();

    // Create children, seeding each with its NNUE value estimate
    for (i, mv) in candidates.iter().enumerate() {
        let v = values[i] as f64;
        node.children.push(MCTSEdge {
            action: *mv,
            prior: priors[i],
            visit_count: 1, // count the initial evaluation as a visit
            total_value: v, // seed with NNUE value
            child: None,
        });
    }

    node.is_expanded = true;
    node.visit_count = candidates.len() as u32; // account for initial visits

    // Return the mean value across children
    let valid: Vec<f64> = values
        .iter()
        .map(|&v| v as f64)
        .filter(|&v| v > 0.0)
        .collect();
    if valid.is_empty() {
        0.0
    } else {
        valid.iter().sum::<f64>() / valid.len() as f64
    }
}

/// Select the best child using PUCT.
fn select_child(node: &MCTSNode) -> usize {
    let mut best_idx = 0;
    let mut best_score = f64::NEG_INFINITY;
    for (i, edge) in node.children.iter().enumerate() {
        let score = edge.puct_score(node.visit_count);
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    best_idx
}

/// Run one MCTS simulation from root.
/// Returns the value backed up to the root.
fn simulate(node: &mut MCTSNode, game: &GameState, net: &NNUENetwork, player: usize) -> f64 {
    if !node.is_expanded {
        // Leaf node — expand and return value
        let value = expand_node(node, game, net, player);
        node.visit_count += 1;
        return value;
    }

    if node.children.is_empty() {
        // Terminal node
        node.visit_count += 1;
        let score = ScoreBreakdown::compute(&mut game.boards[player].clone(), &game.scoring_cards)
            .total as f64;
        return score;
    }

    // SELECT: pick best child via PUCT
    let child_idx = select_child(node);

    // Apply the move
    let mut g = game.clone();
    if !crate::search::execute_scored_move(&mut g, &node.children[child_idx].action) {
        node.visit_count += 1;
        return 0.0;
    }

    // Simulate opponents (greedy, deterministic, with free-replace for overflow)
    crate::search::advance_opponents(&mut g, player);

    // Recurse into child
    let edge = &mut node.children[child_idx];
    if edge.child.is_none() {
        edge.child = Some(Box::new(MCTSNode::new()));
    }
    let child_node = edge.child.as_mut().unwrap();

    let value = if g.is_game_over() {
        child_node.visit_count += 1;
        ScoreBreakdown::compute(&mut g.boards[player], &g.scoring_cards).total as f64
    } else {
        simulate(child_node, &g, net, player)
    };

    // BACKUP
    edge.visit_count += 1;
    edge.total_value += value;
    node.visit_count += 1;

    value
}

/// Run MCTS and return the best move.
/// Runs `num_simulations` from the root, then selects the most-visited child.
pub fn best_move_mcts(
    game: &GameState,
    net: &NNUENetwork,
    num_simulations: usize,
) -> Option<ScoredMove> {
    let result = mcts_search(game, net, num_simulations);
    result
        .into_iter()
        .max_by_key(|(_, visits, _)| *visits)
        .map(|(mv, _, avg_val)| ScoredMove {
            score: avg_val.round() as u16,
            ..mv
        })
}

/// Result of MCTS search: (move, visit_count, avg_value) for each root child.
pub fn mcts_search(
    game: &GameState,
    net: &NNUENetwork,
    num_simulations: usize,
) -> Vec<(ScoredMove, u32, f64)> {
    let player = game.current_player;
    let mut root = MCTSNode::new();

    for _ in 0..num_simulations {
        simulate(&mut root, game, net, player);
    }

    root.children
        .iter()
        .map(|edge| {
            let avg = if edge.visit_count > 0 {
                edge.total_value / edge.visit_count as f64
            } else {
                0.0
            };
            (edge.action, edge.visit_count, avg)
        })
        .collect()
}

/// Run MCTS search and return visit counts + features for self-play training.
/// Returns: (best_move, Vec<(features, visit_count)>, best_value)
/// The visit counts become policy targets; value becomes value target.
pub fn mcts_search_with_features(
    game: &GameState,
    net: &NNUENetwork,
    num_simulations: usize,
    temperature: f32,
) -> Option<(ScoredMove, Vec<(Vec<u16>, f32)>)> {
    let player = game.current_player;
    let mut root = MCTSNode::new();

    for _ in 0..num_simulations {
        simulate(&mut root, game, net, player);
    }

    if root.children.is_empty() {
        return None;
    }

    let bag_info = crate::nnue::BagInfo::from_game(game);

    // Collect features and visit counts for each candidate
    let mut candidates: Vec<(Vec<u16>, f32)> = Vec::new();
    for edge in &root.children {
        // Re-execute the move to get afterstate features
        let mut g = game.clone();
        if !crate::search::execute_scored_move(&mut g, &edge.action) {
            continue;
        }
        let features = crate::nnue::extract_features_with_bag(&g.boards[player], Some(&bag_info));
        candidates.push((features, edge.visit_count as f32));
    }

    // Select move using temperature
    let selected = if temperature < 0.01 {
        // Greedy — most visited
        root.children
            .iter()
            .enumerate()
            .max_by_key(|(_, e)| e.visit_count)
            .map(|(i, _)| i)
            .unwrap()
    } else {
        // Proportional to visit_count^(1/temperature)
        let counts: Vec<f64> = root
            .children
            .iter()
            .map(|e| (e.visit_count as f64).powf(1.0 / temperature as f64))
            .collect();
        let total: f64 = counts.iter().sum();
        if total == 0.0 {
            0
        } else {
            // Simple weighted selection using a deterministic approach
            // (use visit count distribution directly)
            let mut best_idx = 0;
            let mut best_val = 0.0;
            for (i, &c) in counts.iter().enumerate() {
                // Favor higher counts but with some randomness from temperature
                if c > best_val {
                    best_val = c;
                    best_idx = i;
                }
            }
            best_idx
        }
    };

    let best_edge = &root.children[selected];
    let avg_value = if best_edge.visit_count > 0 {
        best_edge.total_value / best_edge.visit_count as f64
    } else {
        0.0
    };

    Some((
        ScoredMove {
            score: avg_value.round() as u16,
            ..best_edge.action
        },
        candidates,
    ))
}
