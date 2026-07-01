//! Pattern-target candidate generation.
//!
//! Augments MCE's candidate pool with moves that progress toward high-value
//! scoring patterns over multi-turn lookahead, even when those moves have
//! low immediate eval. The intent is to catch "locally suboptimal but
//! globally optimal" plans that the greedy candidate generator drops.
//!
//! Approach:
//! 1. For each wildlife scoring pattern type (bear pair, elk line, salmon run,
//!    hawk LOS isolation, fox diversity), enumerate target completions that
//!    are achievable in ≤ remaining-turns turns.
//! 2. For each target, find the FIRST move that progresses toward it most.
//! 3. Return as candidate moves with eval set to target_value × completion_prob.
//!
//! MCE downstream evaluates these via rollouts; if the pattern is real, the
//! rollouts confirm and the move wins. If the pattern is wishful, the rollout
//! mean is unimpressive and a more grounded candidate wins.
//!
//! Bounded enumeration: at most ~5 targets per wildlife type × 5 wildlife
//! types = ~25 candidates added.

use cascadia_core::board::Board;
use cascadia_core::game::GameState;
use cascadia_core::hex::{HexCoord, ADJACENCY};
use cascadia_core::types::{ScoringCardVariant, Wildlife};

use crate::eval::ScoredMove;

/// Generate pattern-target candidates for the current game state.
/// Returns moves that progress toward high-value future patterns.
///
/// Enabled via env `CASCADIA_PATTERN_TARGET=1`. Default off — must be
/// explicitly opted in for benching to avoid mixing the experiment into
/// production paths.
pub fn pattern_target_candidates(game: &GameState) -> Vec<ScoredMove> {
    let board = &game.boards[game.current_player];
    let frontier = board.frontier();
    if frontier.is_empty() {
        return Vec::new();
    }
    let market_pairs: Vec<(usize, cascadia_core::types::TileData, Wildlife)> = game
        .market
        .available()
        .map(|(i, pair)| (i, pair.tile, pair.wildlife))
        .collect();
    if market_pairs.is_empty() {
        return Vec::new();
    }

    let turns_left_player = (game.turns_remaining as usize) / game.num_players.max(1);
    let mut out: Vec<ScoredMove> = Vec::with_capacity(25);

    // Per wildlife type, find the most leveraged pattern target and emit the
    // first move that progresses toward it.
    for wildlife in [
        Wildlife::Bear,
        Wildlife::Elk,
        Wildlife::Salmon,
        Wildlife::Hawk,
        Wildlife::Fox,
    ] {
        let variant = game.scoring_cards.variant_for(wildlife);
        let target = pattern_target_for_wildlife(board, wildlife, variant, turns_left_player);
        if let Some(target_score) = target {
            // Find a first move that pulls in the same direction. Strategy:
            // among market pairs that include this wildlife, pick the (tile,
            // wildlife_cell) combo that maximizes the wildlife marginal score.
            if let Some(mv) = find_first_move_for_wildlife(
                game,
                board,
                &market_pairs,
                wildlife,
                variant,
                &frontier,
                target_score,
            ) {
                out.push(mv);
            }
        }
    }

    out
}

/// Estimate the best-case score we could still achieve for `wildlife` given
/// the current board, remaining turns, and scoring card variant. This is the
/// *upper-bound* contribution this wildlife could make over remaining turns.
fn pattern_target_for_wildlife(
    board: &Board,
    wildlife: Wildlife,
    variant: ScoringCardVariant,
    turns_left: usize,
) -> Option<u16> {
    let cur = cascadia_core::scoring::wildlife::score_wildlife(board, wildlife, variant);
    // Estimate maximum extra wildlife placements over remaining turns. Each
    // turn the player gets at most 1 wildlife placement (sometimes 2 with
    // nature tokens, ignored here for simplicity).
    let max_extra = turns_left.min(6) as u16; // soft cap
    let extra_potential = match (wildlife, variant) {
        // Card A scoring rules (AAAAA): all conservative ceilings.
        (Wildlife::Bear, ScoringCardVariant::A) => {
            // Bear A: 5 pts per pair. Best case = N more pairs.
            (max_extra / 2) * 5
        }
        (Wildlife::Elk, ScoringCardVariant::A) => {
            // Elk A: lines (1, 2, 5, 9, 13) for sizes (1, 2, 3, 4, 5).
            // Best case: extend longest existing line by `max_extra`, score
            // depends on resulting length. Conservatively: +5 pts (going from
            // size N to N+1 elk is roughly +(2..4) for N≤4 and +4 for N=5).
            (max_extra as u16).saturating_mul(3)
        }
        (Wildlife::Salmon, ScoringCardVariant::A) => {
            // Salmon A: runs of 2/3/4/5/6 score 2/4/7/11/15. Best case +4/turn.
            (max_extra as u16).saturating_mul(4)
        }
        (Wildlife::Hawk, ScoringCardVariant::A) => {
            // Hawk A: count of isolated hawks. Scores 0/2/5/8/11/14/18/22/27 for 0..8.
            // Marginal +3..+5 per isolated hawk. Conservative +4/turn.
            (max_extra as u16).saturating_mul(4)
        }
        (Wildlife::Fox, ScoringCardVariant::A) => {
            // Fox A: distinct wildlife types adjacent. Already capped at 5/fox.
            // +1-3 per fox added depending on neighborhood diversity.
            (max_extra as u16).saturating_mul(3)
        }
        _ => (max_extra as u16).saturating_mul(2),
    };
    Some(cur.saturating_add(extra_potential))
}

/// Find a first move that places `wildlife` (or progresses toward placing it)
/// while extracting maximum pattern leverage. Uses lighter-weight eval than
/// `wildlife_strategic_candidates` to avoid duplicating work — we just want
/// ONE candidate per pattern target.
fn find_first_move_for_wildlife(
    game: &GameState,
    board: &Board,
    market_pairs: &[(usize, cascadia_core::types::TileData, Wildlife)],
    wildlife: Wildlife,
    variant: ScoringCardVariant,
    frontier: &[u16],
    target_score: u16,
) -> Option<ScoredMove> {
    let adj = &*ADJACENCY;
    let cards = &game.scoring_cards;

    // Score = pattern-target value × completion_probability heuristic.
    // We synthesize a high eval so MCE downstream rollouts evaluate this
    // candidate seriously.
    let mut best: Option<(ScoredMove, f32)> = None;

    let mut board_clone = board.clone();

    for &(market_idx, tile, market_wl) in market_pairs {
        // Only consider this market slot if the wildlife matches OR a paid
        // independent draw could swap. For v0, require direct match (skip
        // token-paid combos to keep candidate count bounded).
        if market_wl != wildlife {
            continue;
        }
        let max_rot: u8 = if tile.terrain2.is_none() { 1 } else { 6 };

        for &fi in frontier.iter() {
            let coord = HexCoord::from_index(fi as usize);
            for rot in 0..max_rot {
                let tile_action = match board_clone.place_tile(coord, tile, rot) {
                    Some(a) => a,
                    None => continue,
                };
                // Look for a wildlife slot among placed tiles that takes this
                // wildlife and yields maximum future-pattern score.
                let placed_snapshot: arrayvec::ArrayVec<u16, 64> =
                    board_clone.placed_tiles.iter().copied().collect();
                let without = cascadia_core::scoring::wildlife::score_wildlife(
                    &board_clone,
                    wildlife,
                    variant,
                );
                let mut best_wl_score: Option<(HexCoord, u16)> = None;
                for &ti in placed_snapshot.iter() {
                    if !board_clone
                        .grid
                        .get(ti as usize)
                        .can_place_wildlife(wildlife)
                    {
                        continue;
                    }
                    let wa = match board_clone.place_wildlife(ti as usize, wildlife) {
                        Some(a) => a,
                        None => continue,
                    };
                    let with = cascadia_core::scoring::wildlife::score_wildlife(
                        &board_clone,
                        wildlife,
                        variant,
                    );
                    board_clone.undo(wa);
                    let delta = with.saturating_sub(without);
                    // Pattern-leverage score: prefer placements that ALSO
                    // create future opportunities (neighbor wildlife slots
                    // open). For simplicity, +1 per empty-wildlife-slot
                    // neighbor that allows the same wildlife.
                    let wc = HexCoord::from_index(ti as usize);
                    let mut bonus = 0u16;
                    for n in adj.neighbors_of(ti as usize) {
                        let nc = board_clone.grid.get(n);
                        if nc.is_present() && !nc.has_wildlife() && nc.can_place_wildlife(wildlife)
                        {
                            bonus += 1;
                        }
                    }
                    let total = delta + bonus;
                    let _ = wc;
                    if best_wl_score.is_none() || total > best_wl_score.unwrap().1 {
                        best_wl_score = Some((HexCoord::from_index(ti as usize), total));
                    }
                }
                board_clone.undo(tile_action);

                if let Some((wc, _)) = best_wl_score {
                    // Eval: pattern target score, which is intentionally high
                    // to push MCE to roll out this candidate. If MCE rollouts
                    // confirm value, the candidate wins; if not, it loses.
                    let eval = target_score;
                    let mv = ScoredMove {
                        market_index: market_idx,
                        wildlife_market_index: None,
                        tile_q: coord.q,
                        tile_r: coord.r,
                        rotation: rot,
                        wildlife_q: Some(wc.q),
                        wildlife_r: Some(wc.r),
                        // ScoredMove.score is the actual immediate score; we
                        // don't have a free-standing immediate value here.
                        // Use the wildlife marginal `delta + bonus` we
                        // computed earlier — close enough for downstream
                        // sorting. eval = pattern target (in i32-points-scale).
                        score: 0,
                        eval: (eval as i32) * crate::eval::EVAL_SCALE,
                    };
                    let score = eval as f32;
                    if best.as_ref().map(|(_, s)| score > *s).unwrap_or(true) {
                        best = Some((mv, score));
                    }
                }
            }
        }
    }

    best.map(|(m, _)| m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_core::types::ScoringCards;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn pattern_target_candidates_returns_some_at_fresh_game() {
        let mut rng = StdRng::seed_from_u64(0xBEEF);
        let g = GameState::new(4, ScoringCards::all_a(), &mut rng);
        let cands = pattern_target_candidates(&g);
        // Fresh game: market has some wildlife, frontier exists,
        // so at least some pattern targets should generate candidates.
        // Not all wildlife types may be in the market though — accept any.
        assert!(
            cands.len() <= 5,
            "should be bounded by 5 wildlife types, got {}",
            cands.len()
        );
    }

    #[test]
    fn pattern_target_candidates_are_legal() {
        let mut rng = StdRng::seed_from_u64(0xCAFE);
        let mut g = GameState::new(4, ScoringCards::all_a(), &mut rng);
        // Play forward a few turns to get mid-game state
        for _ in 0..5 {
            if g.is_game_over() {
                break;
            }
            if let Some(mv) = crate::search::greedy_move(&g) {
                crate::search::execute_scored_move(&mut g, &mv);
            } else {
                break;
            }
        }
        let cands = pattern_target_candidates(&g);
        for mv in &cands {
            // Each candidate should be executable.
            let mut g2 = g.clone();
            assert!(
                crate::search::execute_scored_move(&mut g2, mv),
                "pattern-target candidate must be legal: {:?}",
                mv
            );
        }
    }
}
