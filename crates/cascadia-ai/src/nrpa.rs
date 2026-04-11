//! Nested Rollout Policy Adaptation (NRPA) for Cascadia.
//!
//! Cazenave 2009; world records on Morpion Solitaire, SameGame, Crossword Construction.
//!
//! Algorithm:
//! ```text
//! function NRPA(level, policy):
//!     if level == 0:
//!         return playout(policy)        # one rollout
//!     best_score = -inf
//!     best_seq   = []
//!     for i in 1..=N:
//!         score, seq = NRPA(level-1, policy)
//!         if score > best_score: { best_score, best_seq = score, seq }
//!     policy = adapt(policy, best_seq)  # softmax policy gradient toward best seq
//!     return best_score, best_seq
//! ```
//!
//! For Cascadia we use a small move-feature policy keyed by
//! (drafted_animal × wildlife_count_bin × placement_class), which is small enough to learn
//! online from a few hundred rollouts but expressive enough to capture pattern-building.
//!
//! Stochasticity: each playout shuffles the bag once at the start. Generalized NRPA
//! semantics — chance is fixed within a playout, varied across playouts.

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use cascadia_core::game::GameState;
use cascadia_core::scoring::ScoreBreakdown;
use cascadia_core::types::Wildlife;

use crate::eval::{best_move_with_potential, ScoredMove};
use crate::nnue::NNUENetwork;
use crate::search::{candidate_moves_decomposed, execute_scored_move, greedy_move};

/// Fast candidate generator for NRPA playouts: one greedy move per market combo.
/// Returns at most num_market_pairs candidates (typically 4) — much cheaper than
/// candidate_moves_decomposed (~4 vs ~15-21 NNUE evals per move).
fn fast_candidates(game: &GameState) -> Vec<ScoredMove> {
    let player = game.current_player;
    let mp: Vec<_> = game.market.available()
        .map(|(i, p)| (i, p.tile, p.wildlife)).collect();
    if mp.is_empty() { return Vec::new(); }
    let cards = game.scoring_cards;
    let turns = game.turns_remaining;
    let board = &game.boards[player];

    // For each market slot (and optionally each independent draft), produce one
    // greedy "best" move that uses ONLY that combo.
    let mut candidates = Vec::with_capacity(mp.len() * 2);
    for &(idx, tile, wl) in &mp {
        let restricted = vec![(idx, tile, wl)];
        let mut b = board.clone();
        if let Some(mv) = best_move_with_potential(&mut b, &restricted, &cards, turns) {
            candidates.push(mv);
        }
    }
    // Independent drafts: only if affordable AND there's variety
    if board.nature_tokens > 0 && mp.len() >= 2 {
        for &(ti, tile, _) in &mp {
            for &(wi, _, wl) in &mp {
                if ti == wi { continue; }
                let restricted = vec![(ti, tile, wl)];
                let mut b = board.clone();
                if let Some(mut mv) = best_move_with_potential(&mut b, &restricted, &cards, turns) {
                    mv.wildlife_market_index = Some(wi);
                    candidates.push(mv);
                }
            }
        }
    }
    candidates
}

const NRPA_LEVEL: usize = 2;
const NRPA_N: usize = 30;
const NRPA_ALPHA: f64 = 1.0;

/// A compact move-feature key. Two moves with the same key are treated as
/// interchangeable for policy adaptation purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MoveFeature {
    /// Wildlife type drafted (0-4) or 5 if no wildlife placed
    animal: u8,
    /// Number of own placements of this wildlife so far, binned 0-4 then 5+
    own_count_bin: u8,
    /// Whether the placement is on a keystone tile
    keystone: bool,
    /// Whether the move uses a nature token (independent draft)
    independent: bool,
}

impl MoveFeature {
    fn from_move(game: &GameState, mv: &ScoredMove) -> Self {
        let player = game.current_player;
        let board = &game.boards[player];
        let market = &game.market;

        let drafted = market.pairs[mv.market_index]
            .as_ref()
            .map(|p| p.wildlife);
        let placed_animal = mv.wildlife_q.is_some()
            .then(|| drafted)
            .flatten();

        let animal = placed_animal.map(|w| w as u8).unwrap_or(5);
        let own_count = if let Some(w) = placed_animal {
            board.wildlife_positions[w as usize].len() as u8
        } else {
            0
        };
        let own_count_bin = own_count.min(5);

        let keystone = if let (Some(wq), Some(wr)) = (mv.wildlife_q, mv.wildlife_r) {
            cascadia_core::hex::HexCoord::new(wq, wr)
                .to_index()
                .map(|idx| board.grid.get(idx).is_keystone())
                .unwrap_or(false)
        } else {
            false
        };

        MoveFeature {
            animal,
            own_count_bin,
            keystone,
            independent: mv.wildlife_market_index.is_some(),
        }
    }
}

/// NRPA policy: maps move feature → log-weight. Looked up by hash for fast access.
#[derive(Clone, Default)]
pub struct NrpaPolicy {
    weights: HashMap<MoveFeature, f64>,
}

impl NrpaPolicy {
    fn weight(&self, feat: &MoveFeature) -> f64 {
        self.weights.get(feat).copied().unwrap_or(0.0)
    }

    fn add(&mut self, feat: &MoveFeature, delta: f64) {
        *self.weights.entry(*feat).or_insert(0.0) += delta;
    }
}

/// One step in a playout: which move was taken, and its feature key.
#[derive(Clone)]
struct PlayoutStep {
    chosen: ScoredMove,
    chosen_feat: MoveFeature,
    /// Features of all candidates considered (for policy gradient denominator)
    legal_feats: Vec<MoveFeature>,
}

/// Result of one playout: final score from the AI player's perspective + the action sequence.
#[derive(Clone)]
struct PlayoutResult {
    score: f64,
    seq: Vec<PlayoutStep>,
}

/// Run one stochastic playout from `start_game`, sampling moves with the given policy.
/// Depth-limited: stops after `max_ai_plies` AI ply, returns leaf eval (actual + NNUE).
fn playout(
    start_game: &GameState,
    net: &NNUENetwork,
    policy: &NrpaPolicy,
    ai_player: usize,
    rng: &mut StdRng,
) -> PlayoutResult {
    let max_ai_plies: usize = std::env::var("NRPA_DEPTH").ok().and_then(|s| s.parse().ok()).unwrap_or(6);

    let mut g = start_game.clone();
    g.shuffle_bags(rng);
    let mut seq: Vec<PlayoutStep> = Vec::with_capacity(max_ai_plies);
    let mut ai_plies = 0usize;

    while !g.is_game_over() {
        if g.current_player != ai_player {
            // Opponents play greedy with free-replace
            if g.can_replace_overflow().is_some() {
                g.replace_overflow();
            }
            match greedy_move(&g) {
                Some(mv) => { if !execute_scored_move(&mut g, &mv) { break; } }
                None => break,
            }
            continue;
        }
        if ai_plies >= max_ai_plies { break; }
        // Free overflow
        if g.can_replace_overflow().is_some() {
            g.replace_overflow();
        }

        let cands = if std::env::var("NRPA_FAST").is_ok() {
            fast_candidates(&g)
        } else {
            candidate_moves_decomposed(&g, net)
        };
        if cands.is_empty() { break; }

        let legal_feats: Vec<MoveFeature> = cands.iter()
            .map(|mv| MoveFeature::from_move(&g, mv))
            .collect();

        // Softmax sample
        let weights: Vec<f64> = legal_feats.iter().map(|f| policy.weight(f)).collect();
        let max_w = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = weights.iter().map(|w| (w - max_w).exp()).collect();
        let z: f64 = exps.iter().sum();

        let r: f64 = rng.gen_range(0.0..z);
        let mut acc = 0.0;
        let mut chosen_idx = 0;
        for (i, &e) in exps.iter().enumerate() {
            acc += e;
            if acc >= r { chosen_idx = i; break; }
        }

        let chosen = cands[chosen_idx];
        let chosen_feat = legal_feats[chosen_idx];
        if !execute_scored_move(&mut g, &chosen) { break; }
        ai_plies += 1;

        seq.push(PlayoutStep {
            chosen,
            chosen_feat,
            legal_feats,
        });
    }

    // Score = actual + NNUE remaining (matches MCE leaf semantics)
    let score = if g.is_game_over() {
        ScoreBreakdown::compute(&mut g.boards[ai_player], &g.scoring_cards).total as f64
    } else {
        let actual = ScoreBreakdown::compute(&mut g.boards[ai_player], &g.scoring_cards).total as f64;
        let bag_info = crate::nnue::BagInfo::from_game(&g);
        let nval = net.evaluate_with_bag(&g.boards[ai_player], &bag_info).max(0.0) as f64;
        actual + nval
    };
    PlayoutResult { score, seq }
}

/// Policy gradient update — push the policy toward the actions taken in `seq`.
fn adapt_policy(policy: &mut NrpaPolicy, seq: &[PlayoutStep], alpha: f64) {
    for step in seq {
        // For each step, recompute z over legal_feats then apply gradient
        let weights: Vec<f64> = step.legal_feats.iter().map(|f| policy.weight(f)).collect();
        let max_w = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = weights.iter().map(|w| (w - max_w).exp()).collect();
        let z: f64 = exps.iter().sum();

        for (feat, e) in step.legal_feats.iter().zip(exps.iter()) {
            let p = e / z;
            let mut delta = -alpha * p;
            if *feat == step.chosen_feat {
                delta += alpha;
            }
            policy.add(feat, delta);
        }
    }
}

/// Recursive NRPA: returns (best_score, best_sequence) discovered at this level.
fn nrpa(
    level: usize,
    n: usize,
    policy: &mut NrpaPolicy,
    start_game: &GameState,
    net: &NNUENetwork,
    ai_player: usize,
    rng: &mut StdRng,
) -> PlayoutResult {
    if level == 0 {
        return playout(start_game, net, policy, ai_player, rng);
    }
    let mut best = PlayoutResult { score: f64::NEG_INFINITY, seq: Vec::new() };
    for _ in 0..n {
        let mut sub_policy = policy.clone();
        let r = nrpa(level - 1, n, &mut sub_policy, start_game, net, ai_player, rng);
        if r.score > best.score {
            best = r;
        }
        // Adapt the parent policy toward the best sequence found so far
        adapt_policy(policy, &best.seq, NRPA_ALPHA);
    }
    best
}

/// Pick the best ROOT move using NRPA. Runs nrpa(LEVEL, N) which produces a full
/// best-sequence — we return the first action of that sequence.
pub fn best_move_nrpa(
    game: &GameState,
    net: &NNUENetwork,
    rng: &mut StdRng,
) -> Option<ScoredMove> {
    let level: usize = std::env::var("NRPA_LEVEL").ok().and_then(|s| s.parse().ok()).unwrap_or(NRPA_LEVEL);
    let n: usize = std::env::var("NRPA_N").ok().and_then(|s| s.parse().ok()).unwrap_or(NRPA_N);
    let ai_player = game.current_player;
    let mut policy = NrpaPolicy::default();
    let result = nrpa(level, n, &mut policy, game, net, ai_player, rng);
    result.seq.into_iter().next().map(|s| {
        ScoredMove { score: result.score.round() as u16, ..s.chosen }
    })
}
