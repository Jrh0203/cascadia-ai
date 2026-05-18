//! Drafting opponents for training and benchmarking.
//!
//! These exist to create training opponents that can't be exploited the way
//! a single frozen strong NNUE can — the learner sees unpredictable drafts
//! from a variety of opponent "shapes" rather than learning the specific
//! drafting pattern of one opponent.
//!
//! All three opponents draft by choosing a market pair, then place the tile
//! and wildlife greedily (via `eval::best_move_with_potential` constrained to
//! the chosen market slot). They differ only in HOW the market pair is chosen.

use cascadia_core::game::GameState;
use cascadia_core::types::Wildlife;
use rand::rngs::StdRng;
use rand::Rng;

use crate::eval::{best_move_with_potential, ScoredMove};

/// Greedy placement constrained to a single market slot.
fn greedy_move_for_market_index(game: &GameState, market_idx: usize) -> Option<ScoredMove> {
    let mp: Vec<_> = game
        .market
        .available()
        .filter(|(i, _)| *i == market_idx)
        .map(|(i, p)| (i, p.tile, p.wildlife))
        .collect();
    if mp.is_empty() {
        return None;
    }
    let cards = game.scoring_cards;
    let turns = game.turns_remaining;
    let mut board = game.boards[game.current_player].clone();
    best_move_with_potential(&mut board, &mp, &cards, turns)
}

/// Count wildlife tokens placed across all boards, by type.
///
/// Publicly observable: every placed token is visible on someone's board.
/// Used by scarcity opponent (more placed = fewer remaining in bag).
fn count_placed_wildlife(game: &GameState) -> [usize; 5] {
    let mut counts = [0usize; 5];
    for board in &game.boards {
        for w in 0..5 {
            counts[w] += board.wildlife_positions[w].len();
        }
    }
    counts
}

/// List of (market_index, wildlife) pairs for currently-available slots.
fn available_pairs(game: &GameState) -> Vec<(usize, Wildlife)> {
    game.market
        .pairs
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.as_ref().map(|p| (i, p.wildlife)))
        .collect()
}

/// Random-draft opponent: picks a market slot uniformly at random, then
/// places tile + wildlife greedily.
///
/// Unexploitable by construction — no pattern for the learner to learn.
/// But markets stay lush (random draft doesn't grab the best animals),
/// so this is a low-competitive-pressure opponent.
pub fn random_draft_move(game: &GameState, rng: &mut StdRng) -> Option<ScoredMove> {
    let available = available_pairs(game);
    if available.is_empty() {
        return None;
    }
    let (idx, _) = available[rng.gen_range(0..available.len())];
    greedy_move_for_market_index(game, idx)
}

/// Scarcity opponent: picks the market slot whose wildlife type is most
/// scarce in the bag (publicly observable: fewer remaining = more already
/// placed on boards). Ties broken uniformly at random.
///
/// Models a reasonable human heuristic ("grab what's disappearing") while
/// staying unpredictable enough that a learner can't fully exploit it.
pub fn scarcity_draft_move(game: &GameState, rng: &mut StdRng) -> Option<ScoredMove> {
    let available = available_pairs(game);
    if available.is_empty() {
        return None;
    }
    let placed = count_placed_wildlife(game);
    // Higher placed count = fewer left in bag = more scarce.
    let max_placed = available
        .iter()
        .map(|(_, w)| placed[*w as usize])
        .max()
        .unwrap();
    let top: Vec<usize> = available
        .iter()
        .filter(|(_, w)| placed[*w as usize] == max_placed)
        .map(|(i, _)| *i)
        .collect();
    let idx = top[rng.gen_range(0..top.len())];
    greedy_move_for_market_index(game, idx)
}

/// Sample a per-game preference vector over the 5 wildlife types.
///
/// Returns a probability distribution (sums to 1) formed by normalizing
/// 5 uniform samples in [0,1]. Flat enough that any animal can still be
/// picked, skewed enough that each game has a real "character."
pub fn sample_preferences(rng: &mut StdRng) -> [f32; 5] {
    let mut p = [0.0f32; 5];
    let mut sum = 0.0f32;
    for v in p.iter_mut() {
        *v = rng.gen_range(0.0f32..1.0);
        sum += *v;
    }
    // Guard against degenerate draw (probability ~0).
    if sum <= 0.0 {
        return [0.2; 5];
    }
    for v in p.iter_mut() {
        *v /= sum;
    }
    p
}

/// Preference opponent: weights each market slot by its wildlife's preference
/// score, then samples proportionally.
///
/// The preference vector is sampled once per game and held constant — the
/// opponent has a consistent drafting "character" across all 20 turns.
/// With a uniform preference, this is equivalent to `random_draft_move`.
pub fn preference_draft_move(
    game: &GameState,
    prefs: &[f32; 5],
    rng: &mut StdRng,
) -> Option<ScoredMove> {
    let available = available_pairs(game);
    if available.is_empty() {
        return None;
    }
    let weights: Vec<f32> = available.iter().map(|(_, w)| prefs[*w as usize]).collect();
    let total: f32 = weights.iter().sum();
    if total <= 0.0 {
        let (idx, _) = available[rng.gen_range(0..available.len())];
        return greedy_move_for_market_index(game, idx);
    }
    let mut r = rng.gen_range(0.0..total);
    for (i, &w) in weights.iter().enumerate() {
        if r < w {
            return greedy_move_for_market_index(game, available[i].0);
        }
        r -= w;
    }
    // Floating-point fallback — shouldn't usually trigger.
    greedy_move_for_market_index(game, available.last().unwrap().0)
}
