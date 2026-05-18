//! Persistent cross-turn open-loop UCT MCTS for Cascadia (Exp #6).
//!
//! Implements the "cross-turn tree reuse" experiment from
//! `AUTONOMOUS_RESEARCH_REPORT.md`. The single-turn variant lives in
//! `uct_mcts.rs`; this module adds *persistence across an AI's turns within
//! a single game*: after AI plays move M, the chosen edge's child subtree is
//! promoted to root and sibling subtrees are dropped. Search budget compounds
//! across turns rather than restarting from zero.
//!
//! Why cross-turn reuse is principled here. The tree uses **open-loop**
//! semantics (Cazenave 2014): edges are *action sequences*, not state
//! sequences. Chance — bag shuffle + opponent moves — is averaged over
//! per-descent realizations. The Q-value at `root.children[M].child` is the
//! expected AI return when AI commits to playing M, averaging over all
//! possible chance realizations the simulator might produce. After AI plays
//! M in the real game, that same posterior is exactly what we want for
//! AI's *next* decision — so promoting the child to root is sound. Visit
//! counts and Q-values carry over.
//!
//! Stat-preserving candidate regeneration. The tree was built from a
//! *simulated* state at the prior turn; the actual realized state at AI's
//! next turn may have a different market (bag draws diverge between
//! simulation and reality). When the new root is "fresh" from promotion,
//! the next pick_move call regenerates candidates from the REAL game state
//! and matches them against the promoted root's existing edges by a stable
//! action key. Matching actions inherit their accumulated visits/Q; new
//! candidates start fresh; old infeasible edges are dropped. Parent-node
//! visit/total counters are recomputed from the surviving edges so UCB1's
//! bookkeeping stays consistent.
//!
//! Concurrency. Root parallelism via `PersistentUctForest`: K independent
//! trees, all advanced in lockstep through the same realized AI action.
//! No locks. Aggregation across trees uses sum-of-visits voting (Cazenave
//! & Saffidine 2010 root-parallel MCTS).
//!
//! Tunable env vars (read at construction):
//! - `MCTS_TREE_C` — UCB1 exploration constant (default 14.0, matching `uct_mcts.rs`).
//! - `MCTS_TREE_DEPTH` — tree-search ply cap before switching to rollout (default 4).
//! - `MCTS_TREE_REGEN_ON_ADVANCE` — regenerate root candidates against the real
//!   game state after each advance (default 1; set 0 for pure open-loop).

use std::sync::Arc;
use std::thread;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use cascadia_core::game::GameState;
use cascadia_core::scoring::ScoreBreakdown;

use crate::eval::ScoredMove;
use crate::nnue::NNUENetwork;
use crate::search::{advance_opponents, execute_scored_move, greedy_move};

/// Default UCB1 exploration constant. Cascadia AI scores range ~50–110;
/// uct_mcts.rs uses 14.0, which is well-calibrated for this scale.
const DEFAULT_C_UCB: f64 = 14.0;

/// Default cap on AI plies in the tree before switching to rollout. Cascadia
/// is a 20-move game; 4 AI plies covers near-future planning while keeping
/// the tree manageable.
const DEFAULT_MAX_TREE_DEPTH: usize = 4;

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name).ok()
        .map(|s| !s.is_empty() && s != "0" && s.to_ascii_lowercase() != "false")
        .unwrap_or(default)
}

/// Stable identity key for a `ScoredMove`, ignoring the transient `score`/
/// `eval` fields. Two moves with identical placement intent compare equal
/// regardless of evaluation noise.
type ActionKey = (usize, Option<usize>, i8, i8, Option<i8>, Option<i8>, u8);

fn action_key(m: &ScoredMove) -> ActionKey {
    (m.market_index, m.wildlife_market_index,
     m.tile_q, m.tile_r, m.wildlife_q, m.wildlife_r, m.rotation)
}

/// Run-time tuning shared across a single `PersistentUctTree`. Read from env
/// vars on tree construction so a long-running benchmark can tweak without
/// recompiling, and so each game-loop creation captures the active config.
#[derive(Clone, Copy)]
struct UctConfig {
    c_ucb: f64,
    max_depth: usize,
    regen_on_advance: bool,
}

impl UctConfig {
    fn from_env() -> Self {
        UctConfig {
            c_ucb: env_f64("MCTS_TREE_C", DEFAULT_C_UCB),
            max_depth: env_usize("MCTS_TREE_DEPTH", DEFAULT_MAX_TREE_DEPTH),
            regen_on_advance: env_bool("MCTS_TREE_REGEN_ON_ADVANCE", true),
        }
    }
}

/// Tree node owned by exactly one position in the tree.
///
/// `visits` and `total_value` are the standard UCT bookkeeping. `total_value_sq`
/// is collected for diagnostic reporting (variance estimates) but is not used
/// by the selection rule.
struct UctNode {
    visits: u32,
    total_value: u64,
    total_value_sq: u64,
    edges: Vec<UctEdge>,
    expanded: bool,
}

struct UctEdge {
    action: ScoredMove,
    visits: u32,
    total_value: u64,
    /// Lazily allocated next-AI-decision subtree; created on the second visit
    /// to this edge (the first visit just produces a rollout value).
    child: Option<Box<UctNode>>,
}

impl UctNode {
    fn new() -> Self {
        UctNode {
            visits: 0,
            total_value: 0,
            total_value_sq: 0,
            edges: Vec::new(),
            expanded: false,
        }
    }

    fn count_nodes(&self) -> usize {
        1 + self.edges.iter()
            .filter_map(|e| e.child.as_ref())
            .map(|c| c.count_nodes())
            .sum::<usize>()
    }
}

impl UctEdge {
    fn q(&self) -> f64 {
        if self.visits == 0 { 0.0 } else { self.total_value as f64 / self.visits as f64 }
    }

    fn ucb(&self, parent_visits: u32, c_ucb: f64) -> f64 {
        if self.visits == 0 { return f64::INFINITY; }
        let q = self.q();
        let explore = c_ucb * ((parent_visits as f64).ln() / self.visits as f64).sqrt();
        q + explore
    }
}

/// Plain greedy rollout to game end. AI and opponents both move greedily.
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
    ScoreBreakdown::compute(&mut g.boards[ai_player], &g.scoring_cards).total as u64
}

/// NNUE-guided rollout to game end. AI player uses NNUE-greedy move selection;
/// opponents use plain greedy (matching MCE's `run_nnue_rollout` pattern).
/// Falls back to plain greedy if `net` is None.
fn rollout(g: GameState, ai_player: usize, net: Option<&NNUENetwork>) -> u64 {
    match net {
        None => greedy_rollout(g, ai_player),
        Some(net) => nnue_rollout(g, ai_player, net),
    }
}

fn nnue_rollout(mut g: GameState, ai_player: usize, net: &NNUENetwork) -> u64 {
    while !g.is_game_over() {
        if g.can_replace_overflow().is_some() {
            g.replace_overflow();
        }
        let mv = if g.current_player == ai_player {
            crate::nnue_train::pick_best_move_nnue(&g, net)
                .or_else(|| greedy_move(&g))
        } else {
            greedy_move(&g)
        };
        match mv {
            Some(m) => { if !execute_scored_move(&mut g, &m) { break; } }
            None => break,
        }
    }
    ScoreBreakdown::compute(&mut g.boards[ai_player], &g.scoring_cards).total as u64
}

/// One simulation: descend tree via UCB1, expand on first visit, rollout, backup.
/// Returns the leaf value (AI's final score in this descent).
fn simulate(
    node: &mut UctNode,
    game: &GameState,
    ai_player: usize,
    net: Option<&NNUENetwork>,
    rng: &mut StdRng,
    depth: usize,
    cfg: &UctConfig,
) -> u64 {
    if game.is_game_over() {
        let score = ScoreBreakdown::compute(
            &mut game.boards[ai_player].clone(), &game.scoring_cards,
        ).total as u64;
        node.visits += 1;
        node.total_value += score;
        node.total_value_sq += score * score;
        return score;
    }

    if depth >= cfg.max_depth {
        let score = rollout(game.clone(), ai_player, net);
        node.visits += 1;
        node.total_value += score;
        node.total_value_sq += score * score;
        return score;
    }

    if !node.expanded {
        let cands = crate::mce::default_greedy_mce_candidates(game);
        if cands.is_empty() {
            let score = rollout(game.clone(), ai_player, net);
            node.visits += 1;
            node.total_value += score;
            node.total_value_sq += score * score;
            return score;
        }
        for mv in cands {
            node.edges.push(UctEdge {
                action: mv,
                visits: 0,
                total_value: 0,
                child: None,
            });
        }
        node.expanded = true;
    }

    if node.edges.is_empty() {
        let score = rollout(game.clone(), ai_player, net);
        node.visits += 1;
        node.total_value += score;
        node.total_value_sq += score * score;
        return score;
    }

    // UCB1 selection.
    let parent_visits = node.visits.max(1);
    let mut best_idx = 0;
    let mut best_ucb = f64::NEG_INFINITY;
    for (i, e) in node.edges.iter().enumerate() {
        let u = e.ucb(parent_visits, cfg.c_ucb);
        if u > best_ucb { best_ucb = u; best_idx = i; }
    }

    let action = node.edges[best_idx].action;
    let mut g = game.clone();
    g.shuffle_bags(rng);
    if !execute_scored_move(&mut g, &action) {
        // Action infeasible at this realized state. Take a rollout from the
        // pre-action state so the edge accumulates a reasonable signal; UCB1
        // will deprioritize it next round.
        let score = rollout(game.clone(), ai_player, net);
        let edge = &mut node.edges[best_idx];
        edge.visits += 1;
        edge.total_value += score;
        node.visits += 1;
        node.total_value += score;
        node.total_value_sq += score * score;
        return score;
    }
    advance_opponents(&mut g, ai_player);

    let value = {
        let edge = &mut node.edges[best_idx];
        if edge.visits == 0 {
            // First visit to this edge: rollout, don't expand the child yet.
            rollout(g, ai_player, net)
        } else {
            if edge.child.is_none() {
                edge.child = Some(Box::new(UctNode::new()));
            }
            simulate(edge.child.as_mut().unwrap(), &g, ai_player, net, rng, depth + 1, cfg)
        }
    };

    let edge = &mut node.edges[best_idx];
    edge.visits += 1;
    edge.total_value += value;
    node.visits += 1;
    node.total_value += value;
    node.total_value_sq += value * value;
    value
}

/// Stat-preserving candidate regeneration at the current root.
///
/// Generates candidates from the REAL current game state, then matches each
/// against the promoted root's existing edges by stable action key. Edges with
/// surviving actions inherit their accumulated stats (visits, total_value,
/// child subtree). New candidates start with zero stats. Old edges with no
/// matching action in the new candidate set are dropped — their visits no
/// longer represent a valid AI move at this realized state.
///
/// Parent-node visit/total counters are recomputed as the sum over surviving
/// edges, so UCB1's bookkeeping stays consistent (`node.visits == sum(edge.visits)`).
fn regenerate_root_preserve_stats(node: &mut UctNode, game: &GameState) {
    let new_cands = crate::mce::default_greedy_mce_candidates(game);
    if new_cands.is_empty() {
        node.edges.clear();
        node.expanded = false;
        node.visits = 0;
        node.total_value = 0;
        node.total_value_sq = 0;
        return;
    }

    // Build an index from action key → old edge position so we can pop survivors
    // out of the old edges Vec without quadratic scanning.
    use std::collections::HashMap;
    let mut old_by_key: HashMap<ActionKey, usize> = HashMap::with_capacity(node.edges.len());
    for (i, e) in node.edges.iter().enumerate() {
        old_by_key.insert(action_key(&e.action), i);
    }
    // To safely move edges out by index without disturbing other indices, drain.
    let old_edges: Vec<UctEdge> = std::mem::take(&mut node.edges);
    let mut old_edges: Vec<Option<UctEdge>> = old_edges.into_iter().map(Some).collect();

    let mut new_edges = Vec::with_capacity(new_cands.len());
    let mut surviving_visits: u32 = 0;
    let mut surviving_total: u64 = 0;
    let mut surviving_total_sq: u64 = 0;

    for cand in new_cands {
        let key = action_key(&cand);
        match old_by_key.get(&key) {
            Some(&idx) => {
                if let Some(mut e) = old_edges[idx].take() {
                    // Refresh action's transient score/eval to current evaluation,
                    // but keep accumulated stats and child subtree.
                    e.action = cand;
                    surviving_visits += e.visits;
                    surviving_total += e.total_value;
                    // Variance bookkeeping: edges don't track sq separately;
                    // approximate node-level total_value_sq with sum of squares
                    // of edge means × edge visits. Slight bias but harmless for
                    // diagnostics.
                    let mean = if e.visits > 0 { e.total_value / e.visits as u64 } else { 0 };
                    surviving_total_sq += mean * mean * e.visits as u64;
                    new_edges.push(e);
                } else {
                    // Action key in old_by_key but already taken? Shouldn't happen
                    // (each cand key is unique), but defensively start fresh.
                    new_edges.push(UctEdge {
                        action: cand,
                        visits: 0,
                        total_value: 0,
                        child: None,
                    });
                }
            }
            None => {
                new_edges.push(UctEdge {
                    action: cand,
                    visits: 0,
                    total_value: 0,
                    child: None,
                });
            }
        }
    }

    node.edges = new_edges;
    node.expanded = true;
    node.visits = surviving_visits;
    node.total_value = surviving_total;
    node.total_value_sq = surviving_total_sq;
}

/// A single tree, persistent across an AI's `pick_move` calls within one game.
///
/// The optional `net` is used by both rollouts (NNUE-greedy for AI player)
/// and tree expansion (still uses the existing greedy candidate set —
/// candidate generation is unchanged from the greedy-rollout path so that
/// move-key stability across turns is preserved). When `net` is None the
/// rollout is plain greedy (matches `uct_mcts.rs`).
pub struct PersistentUctTree {
    root: Box<UctNode>,
    ai_player: usize,
    net: Option<Arc<NNUENetwork>>,
    rng: StdRng,
    cfg: UctConfig,
    /// Set when the root was just promoted via `advance` and hasn't been
    /// reconciled against a real game state yet. The next `pick_move` will
    /// run stat-preserving candidate regeneration before its first sim.
    needs_regen: bool,
}

impl PersistentUctTree {
    pub fn new(ai_player: usize, seed: u64) -> Self {
        Self::new_with_net(ai_player, seed, None)
    }

    pub fn new_with_net(
        ai_player: usize,
        seed: u64,
        net: Option<Arc<NNUENetwork>>,
    ) -> Self {
        PersistentUctTree {
            root: Box::new(UctNode::new()),
            ai_player,
            net,
            rng: StdRng::seed_from_u64(seed),
            cfg: UctConfig::from_env(),
            needs_regen: false,
        }
    }

    /// Run `additional_sims` simulations from the current root and return the
    /// most-visited root edge's action.
    ///
    /// The root may already carry visits from prior turns; new sims add to
    /// those. If the root was just promoted from a child subtree, candidates
    /// are first regenerated against `game` (stat-preserving) so action sets
    /// match the realized state.
    pub fn pick_move(&mut self, game: &GameState, additional_sims: usize) -> Option<ScoredMove> {
        if self.needs_regen && self.cfg.regen_on_advance {
            regenerate_root_preserve_stats(&mut self.root, game);
        }
        self.needs_regen = false;

        let cfg = self.cfg;
        let net_ref = self.net.as_deref();
        for _ in 0..additional_sims {
            simulate(&mut self.root, game, self.ai_player, net_ref, &mut self.rng, 0, &cfg);
        }

        if self.root.edges.is_empty() { return None; }

        // Robust selection: most-visited root child, tie-break by mean Q.
        let mut best_idx = 0;
        let mut best_visits = 0u32;
        let mut best_q = f64::NEG_INFINITY;
        for (i, e) in self.root.edges.iter().enumerate() {
            if e.visits > best_visits || (e.visits == best_visits && e.q() > best_q) {
                best_visits = e.visits;
                best_q = e.q();
                best_idx = i;
            }
        }
        let e = &self.root.edges[best_idx];
        Some(ScoredMove { score: e.q().round() as u16, ..e.action })
    }

    /// Promote the child subtree corresponding to `played_move` to root.
    ///
    /// If the move's edge isn't present (move never expanded) or its child is
    /// `None` (only one visit so far → no subtree built yet), the tree resets
    /// to a fresh root. Cross-turn benefit accrues only on edges with ≥2 visits
    /// at the time of advancement, which is the typical case for any
    /// reasonably explored move.
    pub fn advance(&mut self, played_move: &ScoredMove) {
        let target_key = action_key(played_move);
        let idx = self.root.edges.iter().position(|e| action_key(&e.action) == target_key);
        if let Some(idx) = idx {
            let edge = &mut self.root.edges[idx];
            if let Some(child) = edge.child.take() {
                self.root = child;
                self.needs_regen = true;
                return;
            }
        }
        self.root = Box::new(UctNode::new());
        self.needs_regen = false;
    }

    /// Drop all accumulated state and start fresh. Useful between independent
    /// games or after a structural state change (e.g., game restart).
    pub fn reset(&mut self) {
        self.root = Box::new(UctNode::new());
        self.needs_regen = false;
    }

    pub fn root_visits(&self) -> u32 { self.root.visits }

    pub fn node_count(&self) -> usize { self.root.count_nodes() }

    pub fn root_edges_len(&self) -> usize { self.root.edges.len() }
}

/// Root-parallel persistent forest: K independent trees, advanced in lockstep
/// through the same realized AI action.
///
/// Each `pick_move` call distributes `total_sims/K` sims to each tree,
/// running them concurrently on `K` OS threads. Aggregation across trees
/// uses sum-of-visits voting: for each unique action key across all trees,
/// sum visits and total_value, pick the action with the highest visit sum.
///
/// Trees that, after the chosen action, have no expanded subtree for it will
/// reset on advance — so root parallelism's cross-turn benefit is partially
/// realized: most trees retain useful stats, a few fall back to fresh.
pub struct PersistentUctForest {
    trees: Vec<PersistentUctTree>,
    ai_player: usize,
}

impl PersistentUctForest {
    pub fn new(ai_player: usize, num_trees: usize, root_seed: u64) -> Self {
        Self::new_with_net(ai_player, num_trees, root_seed, None)
    }

    pub fn new_with_net(
        ai_player: usize,
        num_trees: usize,
        root_seed: u64,
        net: Option<Arc<NNUENetwork>>,
    ) -> Self {
        let trees = (0..num_trees.max(1))
            .map(|i| {
                PersistentUctTree::new_with_net(
                    ai_player,
                    root_seed.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                    net.as_ref().map(Arc::clone),
                )
            })
            .collect();
        PersistentUctForest { trees, ai_player }
    }

    pub fn pick_move(&mut self, game: &GameState, total_sims: usize) -> Option<ScoredMove> {
        let n = self.trees.len();
        let sims_per_tree = (total_sims + n - 1) / n;

        // Move trees out, run in parallel, move back. Each thread owns one tree.
        let game_arc = Arc::new(game.clone());
        let trees_taken: Vec<_> = std::mem::take(&mut self.trees);

        let handles: Vec<_> = trees_taken.into_iter()
            .map(|mut tree| {
                let g = Arc::clone(&game_arc);
                thread::spawn(move || {
                    tree.pick_move(&g, sims_per_tree);
                    tree
                })
            })
            .collect();

        let processed: Vec<PersistentUctTree> =
            handles.into_iter().map(|h| h.join().unwrap()).collect();
        self.trees = processed;

        // Aggregate root edges across trees by action key.
        use std::collections::HashMap;
        let mut agg: HashMap<ActionKey, (ScoredMove, u64, u64)> = HashMap::new();
        for t in &self.trees {
            for e in &t.root.edges {
                let entry = agg.entry(action_key(&e.action))
                    .or_insert((e.action, 0, 0));
                entry.1 += e.visits as u64;
                entry.2 += e.total_value;
            }
        }

        if agg.is_empty() { return None; }

        let mut best: Option<(ScoredMove, u64, f64)> = None;
        for (_, (action, v, tv)) in &agg {
            let q = if *v > 0 { *tv as f64 / *v as f64 } else { 0.0 };
            match &best {
                None => best = Some((*action, *v, q)),
                Some((_, bv, bq)) => {
                    if v > bv || (v == bv && q > *bq) {
                        best = Some((*action, *v, q));
                    }
                }
            }
        }

        best.map(|(mv, _, q)| ScoredMove { score: q.round() as u16, ..mv })
    }

    /// Advance every tree through the realized action.
    pub fn advance(&mut self, played_move: &ScoredMove) {
        for tree in &mut self.trees {
            tree.advance(played_move);
        }
    }

    pub fn reset(&mut self) {
        for tree in &mut self.trees { tree.reset(); }
    }

    pub fn aggregate_root_visits(&self) -> u32 {
        self.trees.iter().map(|t| t.root_visits()).sum()
    }

    pub fn aggregate_node_count(&self) -> usize {
        self.trees.iter().map(|t| t.node_count()).sum()
    }

    pub fn ai_player(&self) -> usize { self.ai_player }

    pub fn num_trees(&self) -> usize { self.trees.len() }
}

/// One-shot pick_move that does NOT preserve a tree across calls. Useful as a
/// pick_move-style fallback when the caller doesn't thread persistent state.
pub fn best_move_one_shot(
    game: &GameState,
    num_simulations: usize,
    rng: &mut StdRng,
) -> Option<ScoredMove> {
    let ai_player = game.current_player;
    let seed = rng.gen::<u64>();
    let mut tree = PersistentUctTree::new(ai_player, seed);
    tree.pick_move(game, num_simulations)
}

/// One-shot root-parallel pick_move (no cross-turn persistence).
pub fn best_move_one_shot_parallel(
    game: &GameState,
    num_simulations: usize,
    rng: &mut StdRng,
) -> Option<ScoredMove> {
    let ai_player = game.current_player;
    let n = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let seed = rng.gen::<u64>();
    let mut forest = PersistentUctForest::new(ai_player, n, seed);
    forest.pick_move(game, num_simulations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_core::types::ScoringCards;

    fn fresh_game(seed: u64) -> GameState {
        let mut rng = StdRng::seed_from_u64(seed);
        GameState::new(4, ScoringCards::all_a(), &mut rng)
    }

    #[test]
    fn one_shot_returns_a_move() {
        let game = fresh_game(0);
        let mut rng = StdRng::seed_from_u64(1);
        let mv = best_move_one_shot(&game, 32, &mut rng);
        assert!(mv.is_some(), "MCTS one-shot should always return some move");
    }

    #[test]
    fn persistent_tree_grows_across_calls() {
        let mut game = fresh_game(2);
        let mut tree = PersistentUctTree::new(0, 42);

        // First call from fresh root.
        let mv1 = tree.pick_move(&game, 16).expect("must return move");
        let visits_after_t1 = tree.root_visits();
        assert!(visits_after_t1 >= 16, "root visits should reflect sim count, got {}", visits_after_t1);

        // Apply move, advance opponents, advance tree.
        assert!(execute_scored_move(&mut game, &mv1));
        advance_opponents(&mut game, 0);
        tree.advance(&mv1);

        // After advance, tree may have inherited child stats. Run more sims.
        let mv2 = tree.pick_move(&game, 16).expect("must return move");
        // Different state, different best move usually.
        let _ = mv2;
    }

    #[test]
    fn advance_to_unknown_move_resets_to_fresh_root() {
        let game = fresh_game(3);
        let mut tree = PersistentUctTree::new(0, 99);
        tree.pick_move(&game, 8);

        // Construct a synthetic move that can't possibly be in the tree's edges
        // (market index 999 — out of range).
        let bogus = ScoredMove {
            market_index: 999, wildlife_market_index: None,
            tile_q: 99, tile_r: 99, rotation: 0,
            wildlife_q: None, wildlife_r: None,
            score: 0, eval: 0,
        };
        tree.advance(&bogus);

        // Tree should be fresh after advancing on an unknown move.
        assert_eq!(tree.root_visits(), 0);
        assert_eq!(tree.node_count(), 1);
    }

    #[test]
    fn forest_runs_parallel_and_aggregates() {
        let game = fresh_game(7);
        let mut forest = PersistentUctForest::new(0, 4, 1234);
        let mv = forest.pick_move(&game, 32).expect("must return move");
        assert!(forest.aggregate_root_visits() >= 32, "all trees together should have >=32 visits");
        let _ = mv;
    }
}
