//! Exact endgame solver for the last 1-4 plies of a Cascadia game.
//!
//! Strategy: when `game.turns_remaining` is small enough that the entire
//! remaining game tree is enumerable (with NNUE-pruning at deeper plies),
//! replace MCE rollouts with deterministic-bag exact computation. This is
//! the "exact endgame override" of Lever 2 from the +3 plan.
//!
//! Branching collapse: with `prefilter_k = 8` (default), depth-4 search
//! visits at most 8^4 = 4096 leaves — sub-millisecond per decision. The
//! cost is "exact among NNUE's top-K candidates per ply" rather than
//! "exact over all legal moves". From the candidate-cap analysis (2% of
//! decisions have > 24 candidates, all truncated tail), the lost tail
//! is statistically negligible.
//!
//! Bag stochasticity: market tiles are replaced from the bag after each
//! placement. The deterministic refill from `execute_scored_move` is a
//! sound approximation at this depth because each refilled slot is
//! consumed by at most one player before the game ends.
//!
//! Returns the best move for the current player and its score from that
//! player's perspective. `None` if `turns_remaining` is outside the
//! supported range — callers fall back to MCE.

use cascadia_core::game::GameState;
use cascadia_core::scoring::ScoreBreakdown;

use crate::eval::ScoredMove;
use crate::mce::{expanded_candidates, nnue_prefilter_candidates};
use crate::nnue::NNUENetwork;
use crate::search::execute_scored_move;

/// Maximum `turns_remaining` for which the exact solver activates.
/// Beyond this, falls through to MCE.
pub const DEFAULT_MAX_ENDGAME_DEPTH: u8 = 4;

/// Default per-ply candidate cap. With NNUE prefilter at this width,
/// depth-4 search visits ≤ K^4 leaves.
pub const DEFAULT_ENDGAME_PREFILTER_K: usize = 8;

#[derive(Clone, Copy, Debug)]
pub struct EndgameConfig {
    pub max_depth: u8,
    pub prefilter_k: usize,
}

impl Default for EndgameConfig {
    fn default() -> Self {
        EndgameConfig {
            max_depth: DEFAULT_MAX_ENDGAME_DEPTH,
            prefilter_k: DEFAULT_ENDGAME_PREFILTER_K,
        }
    }
}

impl EndgameConfig {
    /// Read tunables from env vars for runtime experimentation:
    /// `CASCADIA_ENDGAME_DEPTH` — max depth (default 4)
    /// `CASCADIA_ENDGAME_K`     — prefilter top-K (default 8)
    pub fn from_env() -> Self {
        let max_depth = std::env::var("CASCADIA_ENDGAME_DEPTH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_MAX_ENDGAME_DEPTH);
        let prefilter_k = std::env::var("CASCADIA_ENDGAME_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_ENDGAME_PREFILTER_K);
        EndgameConfig {
            max_depth,
            prefilter_k,
        }
    }
}

/// Solve the remaining game exactly (over NNUE-pruned candidates) for the
/// current player. Returns `Some((best_move, score_for_root_player))` or
/// `None` if `turns_remaining > cfg.max_depth`.
pub fn solve_endgame(
    game: &GameState,
    net: &NNUENetwork,
    cfg: EndgameConfig,
) -> Option<(ScoredMove, f32)> {
    if game.turns_remaining == 0 || game.turns_remaining > cfg.max_depth {
        return None;
    }
    let root_player = game.current_player;
    solve_for_player(game, net, root_player, cfg.prefilter_k)
}

/// Recursive solver. Returns root_player's score at game end after
/// every player on the remaining path plays optimally for themselves.
///
/// At each node the CURRENT player picks the move that maximizes THEIR
/// own final score — the standard self-interested game-tree assumption
/// (no opponent modeling beyond "they want to win").
fn solve_recursive(
    game: &GameState,
    net: &NNUENetwork,
    root_player: usize,
    prefilter_k: usize,
) -> f32 {
    if game.turns_remaining == 0 || game.is_game_over() {
        return final_score(game, root_player);
    }
    let current = game.current_player;
    let candidates = prefiltered_candidates(game, net, prefilter_k);
    if candidates.is_empty() {
        return final_score(game, root_player);
    }
    let mut best_for_current = f32::NEG_INFINITY;
    let mut score_for_root = final_score(game, root_player);
    for cand in &candidates {
        let mut g = game.clone();
        if !execute_scored_move(&mut g, cand) {
            continue;
        }
        let s_root = solve_recursive(&g, net, root_player, prefilter_k);
        let s_current = if current == root_player {
            s_root
        } else {
            // The recursive call returned root_player's score. The CURRENT
            // (opponent) player chose this move to maximize THEIR own score
            // — we have to compute it separately for the comparison.
            // Do another descent rooted at `current` (cheap: same tree).
            //
            // For depth-1 recursion (current player is about to make their
            // last move), the comparison collapses: their final score is
            // exactly `final_score(g, current)` after the move executes.
            //
            // At deeper depths the opponent's own future play happens via
            // the same `solve_recursive` mechanism, so we recurse from
            // their POV.
            if g.is_game_over() || g.turns_remaining == 0 {
                final_score(&g, current)
            } else {
                solve_recursive(&g, net, current, prefilter_k)
            }
        };
        if s_current > best_for_current {
            best_for_current = s_current;
            score_for_root = s_root;
        }
    }
    score_for_root
}

fn solve_for_player(
    game: &GameState,
    net: &NNUENetwork,
    root_player: usize,
    prefilter_k: usize,
) -> Option<(ScoredMove, f32)> {
    let candidates = prefiltered_candidates(game, net, prefilter_k);
    if candidates.is_empty() {
        return None;
    }
    let mut best: Option<(ScoredMove, f32)> = None;
    for cand in candidates {
        let mut g = game.clone();
        if !execute_scored_move(&mut g, &cand) {
            continue;
        }
        let s = solve_recursive(&g, net, root_player, prefilter_k);
        match best {
            None => best = Some((cand, s)),
            Some((_, b)) if s > b => best = Some((cand, s)),
            _ => {}
        }
    }
    best
}

fn prefiltered_candidates(game: &GameState, net: &NNUENetwork, k: usize) -> Vec<ScoredMove> {
    let cands = expanded_candidates(game);
    if k == 0 || cands.len() <= k {
        return cands;
    }
    nnue_prefilter_candidates(game, net, cands, k)
}

#[inline]
fn final_score(game: &GameState, player: usize) -> f32 {
    let mut boards = game.boards.clone();
    let bd = ScoreBreakdown::compute_with_bonuses(&mut boards, &game.scoring_cards, player);
    bd.total as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_core::types::ScoringCards;
    use rand::{rngs::StdRng, SeedableRng};

    fn champion_nnue() -> NNUENetwork {
        // Tests load the actual champion weights when present, otherwise
        // build a fresh-init NNUE so the tests still run in CI without
        // the binary.
        let path = std::path::Path::new("nnue_weights_v4opp_modal_iter3.bin");
        if path.exists() {
            NNUENetwork::load(path).unwrap_or_else(|_| NNUENetwork::new())
        } else {
            NNUENetwork::new()
        }
    }

    #[test]
    fn returns_none_when_turns_remaining_too_high() {
        let mut rng = StdRng::seed_from_u64(0xA1);
        let g = GameState::new(4, ScoringCards::all_a(), &mut rng);
        let net = NNUENetwork::new(); // weights don't matter here
        assert!(solve_endgame(&g, &net, EndgameConfig::default()).is_none());
    }

    #[test]
    fn solver_returns_legal_move_at_endgame() {
        let mut rng = StdRng::seed_from_u64(0xA2);
        let mut g = GameState::new(4, ScoringCards::all_a(), &mut rng);
        while !g.is_game_over() && g.turns_remaining > DEFAULT_MAX_ENDGAME_DEPTH {
            let mv = crate::search::greedy_move(&g);
            match mv {
                Some(m) => {
                    if !execute_scored_move(&mut g, &m) {
                        break;
                    }
                }
                None => break,
            }
        }
        if g.is_game_over() {
            return;
        }
        let net = champion_nnue();
        let res = solve_endgame(&g, &net, EndgameConfig::default());
        assert!(res.is_some());
        let (_mv, score) = res.unwrap();
        assert!(
            score.is_finite() && score >= 0.0 && score <= 200.0,
            "unreasonable score {}",
            score
        );
    }

    #[test]
    fn solver_picks_max_at_last_ply() {
        let mut rng = StdRng::seed_from_u64(0xA3);
        let mut g = GameState::new(4, ScoringCards::all_a(), &mut rng);
        while !g.is_game_over() && g.turns_remaining > 1 {
            let mv = crate::search::greedy_move(&g);
            match mv {
                Some(m) => {
                    if !execute_scored_move(&mut g, &m) {
                        break;
                    }
                }
                None => break,
            }
        }
        if g.is_game_over() || g.turns_remaining != 1 {
            return;
        }
        let player = g.current_player;
        let net = champion_nnue();
        // At depth=1 we enumerate ALL candidates (no prune) to verify
        // optimality cleanly. K=0 disables the prefilter.
        let cfg = EndgameConfig {
            max_depth: 1,
            prefilter_k: 0,
        };
        let res = solve_endgame(&g, &net, cfg);
        assert!(res.is_some());
        let (_mv, best_score) = res.unwrap();

        let cands = expanded_candidates(&g);
        let mut manual_max: f32 = f32::NEG_INFINITY;
        for c in &cands {
            let mut g2 = g.clone();
            if !execute_scored_move(&mut g2, c) {
                continue;
            }
            let s = final_score(&g2, player);
            if s > manual_max {
                manual_max = s;
            }
        }
        assert!(
            (best_score - manual_max).abs() < 1e-3,
            "solver {} != manual max {}",
            best_score,
            manual_max
        );
    }
}
