//! Monte Carlo Evaluation (MCE): for each candidate move, simulate the
//! rest of the game N times using NNUE-guided play, average final scores.
//! Rollouts are parallelized across all CPU cores.

use std::sync::Arc;
use std::thread;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use cascadia_core::game::GameState;
use cascadia_core::scoring::ScoreBreakdown;

use crate::nnue::extract_features;

use cascadia_core::board::Board;
use cascadia_core::hex::{HexCoord, ADJACENCY};
use cascadia_core::types::Wildlife;

use crate::eval::{best_move_with_potential, ScoredMove};
use crate::nnue::NNUENetwork;
use crate::nnue_train::pick_best_move_nnue;
use crate::search::{candidate_moves_pub, execute_scored_move, greedy_move};
use crate::wildlife_candidates::wildlife_strategic_candidates;

/// Result of the enumerated mulligan evaluation.
#[derive(Debug, Clone)]
pub struct MulliganAnalysis {
    /// MCE score for each (tile_slot, animal_type) — 4 slots × 5 types
    pub score_matrix: [[f64; 5]; 4],
    /// Best score from the current market (no mulligan)
    pub current_best: f64,
    /// Expected best score after mulliganing (probability-weighted over all draws)
    pub mulligan_ev: f64,
    /// Expected best score after mulligan + pinecone (cross-slot pairing)
    pub mulligan_pinecone_ev: f64,
    /// Whether to mulligan (mulligan_ev - 1 > current_best)
    pub should_mulligan: bool,
    /// Whether to mulligan + pinecone (mulligan_pinecone_ev - 2 > both others)
    pub should_mulligan_pinecone: bool,
}

/// Fast variant: compute score matrix using NNUE (not MCE) for instant evaluation.
/// ~20ms for all 20 pairs vs minutes for MCE variant.
pub fn analyze_mulligan_fast(
    game: &GameState,
    net: &NNUENetwork,
) -> MulliganAnalysis {
    let player = game.current_player;
    let cards = game.scoring_cards;
    let turns = game.turns_remaining;
    let bag_info = crate::nnue::BagInfo::from_game(game);
    let mut score_matrix = [[0.0f64; 5]; 4];

    // For each (slot, animal), find best placement and score with NNUE
    for slot in 0..4 {
        let pair = match &game.market.pairs[slot] {
            Some(p) => p,
            None => continue,
        };
        let tile = pair.tile;

        for animal_idx in 0..5 {
            let animal = Wildlife::from_u8(animal_idx as u8).unwrap();
            let fake_market = vec![(slot, tile, animal)];
            let mut board = game.boards[player].clone();
            let best = best_move_with_potential(&mut board, &fake_market, &cards, turns);

            if let Some(mv) = best {
                // Evaluate the afterstate with NNUE
                let mut eval_board = game.boards[player].clone();
                let coord = cascadia_core::hex::HexCoord::new(mv.tile_q, mv.tile_r);
                if eval_board.place_tile(coord, tile, mv.rotation).is_some() {
                    if let (Some(wq), Some(wr)) = (mv.wildlife_q, mv.wildlife_r) {
                        let wcoord = cascadia_core::hex::HexCoord::new(wq, wr);
                        if let Some(widx) = wcoord.to_index() {
                            eval_board.place_wildlife(widx, animal);
                        }
                    }
                    let actual = cascadia_core::scoring::ScoreBreakdown::compute(
                        &mut eval_board, &cards).total as f64;
                    let remaining = net.evaluate_with_bag(&eval_board, &bag_info) as f64;
                    score_matrix[slot][animal_idx] = actual + remaining;
                }
            }
        }
    }

    // Current market best
    let mut current_best = 0.0f64;
    for slot in 0..4 {
        if let Some(ref pair) = game.market.pairs[slot] {
            let score = score_matrix[slot][pair.wildlife as usize];
            if score > current_best { current_best = score; }
        }
    }

    // Bag composition + enumeration (same as full version)
    let bag_counts: [u32; 5] = [
        bag_info.remaining[0] as u32, bag_info.remaining[1] as u32,
        bag_info.remaining[2] as u32, bag_info.remaining[3] as u32,
        bag_info.remaining[4] as u32,
    ];
    let bag_total: u32 = bag_counts.iter().sum();
    let active_slots: Vec<usize> = (0..4)
        .filter(|&i| game.market.pairs[i].is_some())
        .collect();
    let n_active = active_slots.len();

    if bag_total < n_active as u32 || n_active == 0 {
        return MulliganAnalysis {
            score_matrix, current_best,
            mulligan_ev: 0.0, mulligan_pinecone_ev: 0.0,
            should_mulligan: false, should_mulligan_pinecone: false,
        };
    }

    let mut mulligan_ev = 0.0f64;
    let mut mulligan_pinecone_ev = 0.0f64;
    let mut total_prob = 0.0f64;
    enumerate_draws(
        &active_slots, &bag_counts, bag_total, &score_matrix,
        &mut mulligan_ev, &mut mulligan_pinecone_ev, &mut total_prob,
        &mut [0u8; 4], 0, 1.0,
    );
    if total_prob > 0.0 {
        mulligan_ev /= total_prob;
        mulligan_pinecone_ev /= total_prob;
    }

    let nature_tokens = game.boards[game.current_player].nature_tokens;
    let should_mulligan = nature_tokens >= 1 && mulligan_ev - 1.0 > current_best;
    let should_mulligan_pinecone = nature_tokens >= 2 && mulligan_pinecone_ev - 2.0 > current_best
        && mulligan_pinecone_ev - 2.0 > mulligan_ev - 1.0;

    MulliganAnalysis {
        score_matrix, current_best,
        mulligan_ev, mulligan_pinecone_ev,
        should_mulligan, should_mulligan_pinecone,
    }
}

/// Compute the 4×5 score matrix: MCE score for each (tile_slot, animal_type) pair.
/// Then enumerate all possible post-mulligan draws to compute exact expected values.
pub fn analyze_mulligan(
    game: &GameState,
    net: &NNUENetwork,
    rollouts_per_pair: usize,
    rng: &mut StdRng,
) -> MulliganAnalysis {
    let mut score_matrix = [[0.0f64; 5]; 4];

    // Compute MCE score for each of 20 (slot, animal) pairs
    for slot in 0..4 {
        if game.market.pairs[slot].is_none() { continue; }
        for animal_idx in 0..5 {
            let animal = Wildlife::from_u8(animal_idx as u8).unwrap();
            // Create modified game with only this slot, having the target animal
            let mut modified = game.clone();
            if let Some(ref mut pair) = modified.market.pairs[slot] {
                pair.wildlife = animal;
            }
            // Clear other slots so MCE only evaluates this pairing
            for i in 0..4 {
                if i != slot { modified.market.pairs[i] = None; }
            }
            score_matrix[slot][animal_idx] = best_move_mce(&modified, net, rollouts_per_pair, rng)
                .map(|m| m.score as f64)
                .unwrap_or(0.0);
        }
    }

    // Current market best: for each slot, look up score with its current animal
    let mut current_best = 0.0f64;
    for slot in 0..4 {
        if let Some(ref pair) = game.market.pairs[slot] {
            let score = score_matrix[slot][pair.wildlife as usize];
            if score > current_best { current_best = score; }
        }
    }

    // Bag composition for probability calculation
    let bag_info = crate::nnue::BagInfo::from_game(game);
    let bag_counts: [u32; 5] = [
        bag_info.remaining[0] as u32,
        bag_info.remaining[1] as u32,
        bag_info.remaining[2] as u32,
        bag_info.remaining[3] as u32,
        bag_info.remaining[4] as u32,
    ];
    let bag_total: u32 = bag_counts.iter().sum();

    // Which slots are active (have tiles)?
    let active_slots: Vec<usize> = (0..4)
        .filter(|&i| game.market.pairs[i].is_some())
        .collect();
    let n_active = active_slots.len();

    if bag_total < n_active as u32 || n_active == 0 {
        return MulliganAnalysis {
            score_matrix,
            current_best,
            mulligan_ev: 0.0,
            mulligan_pinecone_ev: 0.0,
            should_mulligan: false,
            should_mulligan_pinecone: false,
        };
    }

    // Enumerate all possible n_active-animal draws from the bag
    // For 4 active slots: 5^4 = 625 type combinations
    let mut mulligan_ev = 0.0f64;
    let mut mulligan_pinecone_ev = 0.0f64;
    let mut total_prob = 0.0f64;

    enumerate_draws(
        &active_slots,
        &bag_counts,
        bag_total,
        &score_matrix,
        &mut mulligan_ev,
        &mut mulligan_pinecone_ev,
        &mut total_prob,
        &mut [0u8; 4], // draw buffer
        0,
        1.0, // running probability
    );

    // Normalize (total_prob should be ~1.0, but floating point)
    if total_prob > 0.0 {
        mulligan_ev /= total_prob;
        mulligan_pinecone_ev /= total_prob;
    }

    let nature_tokens = game.boards[game.current_player].nature_tokens;
    let should_mulligan = nature_tokens >= 1 && mulligan_ev - 1.0 > current_best;
    let should_mulligan_pinecone = nature_tokens >= 2 && mulligan_pinecone_ev - 2.0 > current_best
        && mulligan_pinecone_ev - 2.0 > mulligan_ev - 1.0;

    MulliganAnalysis {
        score_matrix,
        current_best,
        mulligan_ev,
        mulligan_pinecone_ev,
        should_mulligan,
        should_mulligan_pinecone,
    }
}

/// Recursively enumerate all possible draws from the bag.
fn enumerate_draws(
    active_slots: &[usize],
    bag_counts: &[u32; 5],
    bag_total: u32,
    score_matrix: &[[f64; 5]; 4],
    mulligan_ev: &mut f64,
    mulligan_pinecone_ev: &mut f64,
    total_prob: &mut f64,
    draw: &mut [u8; 4],
    depth: usize,
    prob: f64,
) {
    if depth == active_slots.len() {
        // All slots filled — compute best scores for this draw
        let n = active_slots.len();

        // Best paired score: max over slots of score[slot][drawn_animal]
        let mut best_paired = 0.0f64;
        for i in 0..n {
            let slot = active_slots[i];
            let animal = draw[i] as usize;
            let score = score_matrix[slot][animal];
            if score > best_paired { best_paired = score; }
        }

        // Best pinecone score: max over all (tile_slot, animal_from_any_slot)
        let mut best_pinecone = best_paired; // pinecone is optional, compare with paired
        for i in 0..n {
            let tile_slot = active_slots[i];
            for j in 0..n {
                let animal = draw[j] as usize;
                // Pinecone cost: -1 point (only if using cross-slot)
                let score = if i == j {
                    score_matrix[tile_slot][animal] // no pinecone needed
                } else {
                    score_matrix[tile_slot][animal] - 1.0 // pinecone cost
                };
                if score > best_pinecone { best_pinecone = score; }
            }
        }

        *mulligan_ev += prob * best_paired;
        *mulligan_pinecone_ev += prob * best_pinecone;
        *total_prob += prob;
        return;
    }

    // Try each wildlife type for this draw position
    let mut remaining_counts = *bag_counts;
    // Adjust for previous draws in this sequence
    for i in 0..depth {
        remaining_counts[draw[i] as usize] -= 1;
    }
    let remaining_total = bag_total - depth as u32;

    for animal in 0..5u8 {
        if remaining_counts[animal as usize] == 0 { continue; }
        let p = remaining_counts[animal as usize] as f64 / remaining_total as f64;
        draw[depth] = animal;
        enumerate_draws(
            active_slots, bag_counts, bag_total, score_matrix,
            mulligan_ev, mulligan_pinecone_ev, total_prob,
            draw, depth + 1, prob * p,
        );
    }
}

/// 1-ply exact expectimax: for each candidate, simulate opponents then enumerate
/// all possible next-turn wildlife draws weighted by bag probabilities.
/// Uses decomposed evaluation for the next turn: wildlife value per type is
/// pre-computed once, then 625 draw combinations are just array lookups.
///
/// Cost: ~25 NNUE evals per candidate × 15 candidates = ~375 evals (~37ms)
/// vs MCE(750): ~405K evals (~50s). That's 1000× faster.
pub fn best_move_expectimax_1ply(
    game: &GameState,
    net: &NNUENetwork,
) -> Option<ScoredMove> {
    let player = game.current_player;
    let cards = game.scoring_cards;

    // Generate candidates using decomposed approach
    let candidates = crate::search::candidate_moves_decomposed(game, net);
    if candidates.is_empty() { return None; }

    let mut best: Option<(ScoredMove, f64)> = None;

    for mv in &candidates {
        // Execute this candidate move on a clone
        let mut g = game.clone();
        if !crate::search::execute_scored_move(&mut g, mv) { continue; }

        // Simulate opponents (deterministic, greedy)
        crate::search::advance_opponents(&mut g, player);

        if g.is_game_over() {
            // Game ended — use final score
            let score = ScoreBreakdown::compute(&mut g.boards[player], &g.scoring_cards).total;
            let s = score as f64;
            if best.is_none() || s > best.as_ref().unwrap().1 {
                best = Some((*mv, s));
            }
            continue;
        }

        // Current actual score after our move + opponents
        let actual = ScoreBreakdown::compute(&mut g.boards[player], &g.scoring_cards).total as f64;

        // NNUE remaining value (captures future tile expectations + long-term patterns)
        let bag_info = crate::nnue::BagInfo::from_game(&g);
        let nnue_remaining = net.evaluate_with_bag(&g.boards[player], &bag_info) as f64;

        // Pre-compute wildlife value per type on current board
        // "If I get to place animal type T, what's the best delta?"
        let board = &g.boards[player];
        let mut wildlife_value = [0.0f64; 5];
        for animal_idx in 0..5 {
            let animal = Wildlife::from_u8(animal_idx as u8).unwrap();
            let variant = cards.variant_for(animal);
            let base_wl = cascadia_core::scoring::wildlife::score_wildlife(board, animal, variant);

            let mut best_delta = 0.0f64;
            for &ti in board.placed_tiles.iter() {
                let idx = ti as usize;
                if !board.grid.get(idx).can_place_wildlife(animal) { continue; }
                let mut b = board.clone();
                if let Some(wa) = b.place_wildlife(idx, animal) {
                    let with_wl = cascadia_core::scoring::wildlife::score_wildlife(&b, animal, variant);
                    let delta = (with_wl as f64) - (base_wl as f64);
                    // Also add nature token bonus for keystone
                    let keystone_bonus = if b.grid.get(idx).is_keystone() { 1.0 } else { 0.0 };
                    let total_delta = delta + keystone_bonus;
                    if total_delta > best_delta { best_delta = total_delta; }
                    b.undo(wa);
                }
            }
            wildlife_value[animal_idx] = best_delta;
        }

        // Enumerate all possible next-turn wildlife draws (625 combinations)
        let bag_counts: [u32; 5] = [
            bag_info.remaining[0] as u32, bag_info.remaining[1] as u32,
            bag_info.remaining[2] as u32, bag_info.remaining[3] as u32,
            bag_info.remaining[4] as u32,
        ];
        let bag_total: u32 = bag_counts.iter().sum();

        let expected_next_wildlife = if bag_total >= 4 {
            let mut ev = 0.0f64;
            let mut total_prob = 0.0f64;

            // Enumerate 4 draws from the bag (the 4 market refill animals)
            // For each draw, the best wildlife value = max over the 4 drawn types
            for t0 in 0..5u8 {
                let c0 = bag_counts[t0 as usize];
                if c0 == 0 { continue; }
                let p0 = c0 as f64 / bag_total as f64;
                for t1 in 0..5u8 {
                    let c1 = bag_counts[t1 as usize] - if t1 == t0 { 1 } else { 0 };
                    if c1 == 0 { continue; }
                    let p1 = c1 as f64 / (bag_total - 1) as f64;
                    for t2 in 0..5u8 {
                        let c2 = bag_counts[t2 as usize]
                            - if t2 == t0 { 1 } else { 0 }
                            - if t2 == t1 { 1 } else { 0 };
                        if c2 == 0 { continue; }
                        let p2 = c2 as f64 / (bag_total - 2) as f64;
                        for t3 in 0..5u8 {
                            let c3 = bag_counts[t3 as usize]
                                - if t3 == t0 { 1 } else { 0 }
                                - if t3 == t1 { 1 } else { 0 }
                                - if t3 == t2 { 1 } else { 0 };
                            if c3 == 0 { continue; }
                            let p3 = c3 as f64 / (bag_total - 3) as f64;

                            let prob = p0 * p1 * p2 * p3;
                            // Best wildlife value from the 4 drawn types
                            let best_wl = wildlife_value[t0 as usize]
                                .max(wildlife_value[t1 as usize])
                                .max(wildlife_value[t2 as usize])
                                .max(wildlife_value[t3 as usize]);
                            ev += prob * best_wl;
                            total_prob += prob;
                        }
                    }
                }
            }
            if total_prob > 0.0 { ev / total_prob } else { 0.0 }
        } else {
            0.0
        };

        // Total score: actual + NNUE remaining + expected next-turn wildlife bonus
        let total = actual + nnue_remaining + expected_next_wildlife;

        if best.is_none() || total > best.as_ref().unwrap().1 {
            best = Some((*mv, total));
        }
    }

    best.map(|(mv, score)| ScoredMove { score: score.round() as u16, ..mv })
}

/// 2-ply exact expectimax: evaluates current move + enumerates next turn's wildlife draws
/// + for the best next move, enumerates the turn AFTER that.
/// Captures 2-turn pattern building with exact probabilities.
pub fn best_move_expectimax_2ply(
    game: &GameState,
    net: &NNUENetwork,
) -> Option<ScoredMove> {
    let player = game.current_player;
    let cards = game.scoring_cards;

    let candidates = crate::search::candidate_moves_decomposed(game, net);
    if candidates.is_empty() { return None; }

    let mut best: Option<(ScoredMove, f64)> = None;

    for mv in &candidates {
        let mut g = game.clone();
        if !crate::search::execute_scored_move(&mut g, mv) { continue; }

        // Simulate opponents
        crate::search::advance_opponents(&mut g, player);

        if g.is_game_over() {
            let score = ScoreBreakdown::compute(&mut g.boards[player], &g.scoring_cards).total as f64;
            if best.is_none() || score > best.as_ref().unwrap().1 {
                best = Some((*mv, score));
            }
            continue;
        }

        // Ply 1: enumerate next-turn wildlife draws
        let score = evaluate_position_with_enumeration(&g, player, net, 2);

        if best.is_none() || score > best.as_ref().unwrap().1 {
            best = Some((*mv, score));
        }
    }

    best.map(|(mv, score)| ScoredMove { score: score.round() as u16, ..mv })
}

/// Score ALL candidates using 1-ply expectimax. Returns (move, features, score) for each.
/// Used for policy data collection — records what expectimax thinks about every candidate.
pub fn score_all_candidates_expectimax(
    game: &GameState,
    net: &NNUENetwork,
) -> Vec<(ScoredMove, Vec<u16>, f64)> {
    let player = game.current_player;

    let candidates = crate::search::candidate_moves_decomposed(game, net);
    if candidates.is_empty() { return vec![]; }

    let bag_info = crate::nnue::BagInfo::from_game(game);
    let mut results = Vec::with_capacity(candidates.len());

    for mv in &candidates {
        let mut g = game.clone();
        if !crate::search::execute_scored_move(&mut g, mv) { continue; }

        // Extract features of the afterstate
        let features = crate::nnue::extract_features_with_bag(&g.boards[player], Some(&bag_info));

        // Simulate opponents
        crate::search::advance_opponents(&mut g, player);

        let score = if g.is_game_over() {
            ScoreBreakdown::compute(&mut g.boards[player], &g.scoring_cards).total as f64
        } else {
            evaluate_position_with_enumeration(&g, player, net, 1)
        };

        results.push((*mv, features, score));
    }

    results
}

/// Evaluate a position by enumerating future wildlife draws.
/// depth=1: enumerate next market, take best move, use NNUE for remaining.
/// depth=2: enumerate next market, take best move, then enumerate AGAIN for the turn after.
fn evaluate_position_with_enumeration(
    game: &GameState,
    player: usize,
    net: &NNUENetwork,
    depth: usize,
) -> f64 {
    let cards = game.scoring_cards;
    let board = &game.boards[player];

    let actual = ScoreBreakdown::compute(&mut board.clone(), &cards).total as f64;
    let bag_info = crate::nnue::BagInfo::from_game(game);
    let nnue_remaining = net.evaluate_with_bag(board, &bag_info) as f64;

    if depth == 0 {
        return actual + nnue_remaining;
    }

    // Pre-compute wildlife value per type
    let mut wildlife_value = [0.0f64; 5];
    let mut best_wl_placement = [None::<(i8, i8)>; 5]; // track best placement coords

    for animal_idx in 0..5 {
        let animal = Wildlife::from_u8(animal_idx as u8).unwrap();
        let variant = cards.variant_for(animal);
        let base_wl = cascadia_core::scoring::wildlife::score_wildlife(board, animal, variant);

        let mut best_delta = 0.0f64;
        for &ti in board.placed_tiles.iter() {
            let idx = ti as usize;
            if !board.grid.get(idx).can_place_wildlife(animal) { continue; }
            let mut b = board.clone();
            if let Some(wa) = b.place_wildlife(idx, animal) {
                let with_wl = cascadia_core::scoring::wildlife::score_wildlife(&b, animal, variant);
                let delta = (with_wl as f64) - (base_wl as f64);
                let keystone = if board.grid.get(idx).is_keystone() { 1.0 } else { 0.0 };
                if delta + keystone > best_delta {
                    best_delta = delta + keystone;
                    let coord = cascadia_core::hex::HexCoord::from_index(idx);
                    best_wl_placement[animal_idx] = Some((coord.q, coord.r));
                }
                b.undo(wa);
            }
        }
        wildlife_value[animal_idx] = best_delta;
    }

    // For depth >= 2, we need to evaluate the position AFTER placing the best wildlife
    // For each animal type, compute what the board looks like after placing it
    let mut post_wildlife_scores = [0.0f64; 5];

    if depth >= 2 {
        for animal_idx in 0..5 {
            if let Some((wq, wr)) = best_wl_placement[animal_idx] {
                let animal = Wildlife::from_u8(animal_idx as u8).unwrap();
                let mut b = board.clone();
                let wcoord = cascadia_core::hex::HexCoord::new(wq, wr);
                if let Some(widx) = wcoord.to_index() {
                    if let Some(_wa) = b.place_wildlife(widx, animal) {
                        // Simulate opponents after this wildlife placement
                        // (simplified: just use NNUE remaining for the deeper evaluation
                        // since full opponent simulation per type × 625 combos is too expensive)
                        let post_bag = crate::nnue::BagInfo::from_game(game);
                        let post_actual = ScoreBreakdown::compute(&mut b, &cards).total as f64;
                        let post_remaining = net.evaluate_with_bag(&b, &post_bag) as f64;
                        post_wildlife_scores[animal_idx] = post_actual + post_remaining;
                    }
                }
            } else {
                // No valid placement for this type — use base score
                post_wildlife_scores[animal_idx] = actual + nnue_remaining;
            }
        }
    }

    // Enumerate wildlife draws
    let bag_counts: [u32; 5] = [
        bag_info.remaining[0] as u32, bag_info.remaining[1] as u32,
        bag_info.remaining[2] as u32, bag_info.remaining[3] as u32,
        bag_info.remaining[4] as u32,
    ];
    let bag_total: u32 = bag_counts.iter().sum();

    if bag_total < 4 {
        return actual + nnue_remaining;
    }

    let mut ev = 0.0f64;
    let mut total_prob = 0.0f64;

    for t0 in 0..5u8 {
        let c0 = bag_counts[t0 as usize];
        if c0 == 0 { continue; }
        let p0 = c0 as f64 / bag_total as f64;
        for t1 in 0..5u8 {
            let c1 = bag_counts[t1 as usize] - if t1 == t0 { 1 } else { 0 };
            if c1 == 0 { continue; }
            let p1 = c1 as f64 / (bag_total - 1) as f64;
            for t2 in 0..5u8 {
                let c2 = bag_counts[t2 as usize]
                    - if t2 == t0 { 1 } else { 0 }
                    - if t2 == t1 { 1 } else { 0 };
                if c2 == 0 { continue; }
                let p2 = c2 as f64 / (bag_total - 2) as f64;
                for t3 in 0..5u8 {
                    let c3 = bag_counts[t3 as usize]
                        - if t3 == t0 { 1 } else { 0 }
                        - if t3 == t1 { 1 } else { 0 }
                        - if t3 == t2 { 1 } else { 0 };
                    if c3 == 0 { continue; }
                    let p3 = c3 as f64 / (bag_total - 3) as f64;
                    let prob = p0 * p1 * p2 * p3;

                    if depth == 1 {
                        // 1-ply: just use wildlife value
                        let best_wl = wildlife_value[t0 as usize]
                            .max(wildlife_value[t1 as usize])
                            .max(wildlife_value[t2 as usize])
                            .max(wildlife_value[t3 as usize]);
                        ev += prob * (actual + nnue_remaining + best_wl);
                    } else {
                        // 2-ply: use the post-wildlife score which includes NNUE remaining
                        // Find best type among the 4 drawn
                        let types = [t0 as usize, t1 as usize, t2 as usize, t3 as usize];
                        let mut best_score = actual + nnue_remaining; // skip wildlife option
                        for &t in &types {
                            if post_wildlife_scores[t] > best_score {
                                best_score = post_wildlife_scores[t];
                            }
                        }
                        ev += prob * best_score;
                    }

                    total_prob += prob;
                }
            }
        }
    }

    if total_prob > 0.0 { ev / total_prob } else { actual + nnue_remaining }
}

/// N-ply exact expectimax: generalized version that recurses to arbitrary depth.
/// Each ply: execute best move → simulate opponents → enumerate wildlife draws → recurse.
pub fn best_move_expectimax_nply(
    game: &GameState,
    net: &NNUENetwork,
    depth: usize,
) -> Option<ScoredMove> {
    let player = game.current_player;
    let candidates = crate::search::candidate_moves_decomposed(game, net);
    if candidates.is_empty() { return None; }

    let mut best: Option<(ScoredMove, f64)> = None;

    for mv in &candidates {
        let mut g = game.clone();
        if !crate::search::execute_scored_move(&mut g, mv) { continue; }

        // Simulate opponents
        crate::search::advance_opponents(&mut g, player);

        let score = if g.is_game_over() {
            ScoreBreakdown::compute(&mut g.boards[player], &g.scoring_cards).total as f64
        } else {
            evaluate_position_recursive(&g, player, net, depth)
        };

        if best.is_none() || score > best.as_ref().unwrap().1 {
            best = Some((*mv, score));
        }
    }

    best.map(|(mv, score)| ScoredMove { score: score.round() as u16, ..mv })
}

/// Recursively evaluate a position with exact wildlife draw enumeration.
/// At each level: compute wildlife value per type, enumerate 625 draws,
/// for the best draw pick the best wildlife placement, then recurse.
fn evaluate_position_recursive(
    game: &GameState,
    player: usize,
    net: &NNUENetwork,
    depth: usize,
) -> f64 {
    let cards = game.scoring_cards;
    let board = &game.boards[player];

    let actual = ScoreBreakdown::compute(&mut board.clone(), &cards).total as f64;
    let bag_info = crate::nnue::BagInfo::from_game(game);
    let nnue_remaining = net.evaluate_with_bag(board, &bag_info) as f64;

    if depth == 0 {
        return actual + nnue_remaining;
    }

    // Pre-compute best wildlife placement per type + resulting board score
    let mut type_scores = [0.0f64; 5]; // score of board AFTER placing best wildlife of this type
    let mut type_deltas = [0.0f64; 5]; // just the wildlife delta

    for animal_idx in 0..5 {
        let animal = Wildlife::from_u8(animal_idx as u8).unwrap();
        let variant = cards.variant_for(animal);
        let base_wl = cascadia_core::scoring::wildlife::score_wildlife(board, animal, variant);

        let mut best_delta = 0.0f64;
        let mut best_board: Option<cascadia_core::board::Board> = None;

        for &ti in board.placed_tiles.iter() {
            let idx = ti as usize;
            if !board.grid.get(idx).can_place_wildlife(animal) { continue; }
            let mut b = board.clone();
            if let Some(_wa) = b.place_wildlife(idx, animal) {
                let with_wl = cascadia_core::scoring::wildlife::score_wildlife(&b, animal, variant);
                let delta = (with_wl as f64) - (base_wl as f64);
                let keystone = if board.grid.get(idx).is_keystone() { 1.0 } else { 0.0 };
                if delta + keystone > best_delta {
                    best_delta = delta + keystone;
                    best_board = Some(b);
                }
            }
        }
        type_deltas[animal_idx] = best_delta;

        if depth >= 2 {
            if let Some(ref post_board) = best_board {
                // For deeper search: evaluate the post-wildlife board
                let post_actual = ScoreBreakdown::compute(&mut post_board.clone(), &cards).total as f64;
                let post_remaining = net.evaluate_with_bag(post_board, &bag_info) as f64;
                type_scores[animal_idx] = post_actual + post_remaining;
            } else {
                type_scores[animal_idx] = actual + nnue_remaining;
            }
        }
    }

    // Enumerate wildlife draws
    let bag_counts: [u32; 5] = [
        bag_info.remaining[0] as u32, bag_info.remaining[1] as u32,
        bag_info.remaining[2] as u32, bag_info.remaining[3] as u32,
        bag_info.remaining[4] as u32,
    ];
    let bag_total: u32 = bag_counts.iter().sum();

    if bag_total < 4 {
        return actual + nnue_remaining;
    }

    let mut ev = 0.0f64;
    let mut total_prob = 0.0f64;

    for t0 in 0..5u8 {
        let c0 = bag_counts[t0 as usize];
        if c0 == 0 { continue; }
        let p0 = c0 as f64 / bag_total as f64;
        for t1 in 0..5u8 {
            let c1 = bag_counts[t1 as usize] - if t1 == t0 { 1 } else { 0 };
            if c1 == 0 { continue; }
            let p1 = c1 as f64 / (bag_total - 1) as f64;
            for t2 in 0..5u8 {
                let c2 = bag_counts[t2 as usize]
                    - if t2 == t0 { 1 } else { 0 }
                    - if t2 == t1 { 1 } else { 0 };
                if c2 == 0 { continue; }
                let p2 = c2 as f64 / (bag_total - 2) as f64;
                for t3 in 0..5u8 {
                    let c3 = bag_counts[t3 as usize]
                        - if t3 == t0 { 1 } else { 0 }
                        - if t3 == t1 { 1 } else { 0 }
                        - if t3 == t2 { 1 } else { 0 };
                    if c3 == 0 { continue; }
                    let p3 = c3 as f64 / (bag_total - 3) as f64;
                    let prob = p0 * p1 * p2 * p3;

                    let types = [t0 as usize, t1 as usize, t2 as usize, t3 as usize];

                    let draw_score = if depth <= 1 {
                        // 1-ply: wildlife delta only + base
                        let best_wl = types.iter().map(|&t| type_deltas[t]).fold(0.0f64, f64::max);
                        actual + nnue_remaining + best_wl
                    } else {
                        // 2+ ply: use post-wildlife board score
                        let mut best = actual + nnue_remaining;
                        for &t in &types {
                            if type_scores[t] > best { best = type_scores[t]; }
                        }
                        best
                    };

                    ev += prob * draw_score;
                    total_prob += prob;
                }
            }
        }
    }

    if total_prob > 0.0 { ev / total_prob } else { actual + nnue_remaining }
}

/// Wildlife-only deep expectimax: simulates placing animals over multiple turns
/// WITHOUT simulating tile placement or opponents. Terrain is greedy (1-turn),
/// wildlife is where deep lookahead pays off.
///
/// At each ply: pre-compute best placement per animal type on existing open slots,
/// enumerate 625 draws, pick best type, place it, recurse.
/// Ply 1-2: exact enumeration (625 branches each)
/// Ply 3+: greedy single-path (expected best animal, no branching)
pub fn best_move_wildlife_deep(
    game: &GameState,
    net: &NNUENetwork,
    wildlife_depth: usize,
) -> Option<ScoredMove> {
    let player = game.current_player;
    let cards = game.scoring_cards;

    let candidates = crate::search::candidate_moves_decomposed(game, net);
    if candidates.is_empty() { return None; }

    let bag_info = crate::nnue::BagInfo::from_game(game);

    let mut best: Option<(ScoredMove, f64)> = None;

    for mv in &candidates {
        let mut g = game.clone();
        if !crate::search::execute_scored_move(&mut g, mv) { continue; }

        // Simulate opponents for this turn only
        crate::search::advance_opponents(&mut g, player);

        if g.is_game_over() {
            let score = ScoreBreakdown::compute(&mut g.boards[player], &g.scoring_cards).total as f64;
            if best.is_none() || score > best.as_ref().unwrap().1 {
                best = Some((*mv, score));
            }
            continue;
        }

        // Current actual score (habitat is already captured)
        let actual = ScoreBreakdown::compute(&mut g.boards[player], &g.scoring_cards).total as f64;
        let nnue_remaining = net.evaluate_with_bag(&g.boards[player], &bag_info) as f64;

        // Wildlife-only deep lookahead on the current board
        let wildlife_bonus = wildlife_deep_eval(
            &g.boards[player], &cards, &bag_info, wildlife_depth,
        );

        // Score = actual + NNUE remaining (captures terrain/long-term) + wildlife bonus
        let score = actual + nnue_remaining + wildlife_bonus;

        if best.is_none() || score > best.as_ref().unwrap().1 {
            best = Some((*mv, score));
        }
    }

    best.map(|(mv, score)| ScoredMove { score: score.round() as u16, ..mv })
}

/// Evaluate expected wildlife bonus from deep lookahead.
/// Simulates placing animals on existing open slots over multiple turns.
/// Returns the expected wildlife score GAIN from future animal placements.
fn wildlife_deep_eval(
    board: &cascadia_core::board::Board,
    cards: &cascadia_core::types::ScoringCards,
    bag_info: &crate::nnue::BagInfo,
    depth: usize,
) -> f64 {
    if depth == 0 { return 0.0; }

    // Pre-compute: for each animal type, best placement delta on current open slots
    let mut type_delta = [0.0f64; 5];
    let mut best_placement_idx = [usize::MAX; 5]; // grid index of best placement

    for animal_idx in 0..5 {
        let animal = Wildlife::from_u8(animal_idx as u8).unwrap();
        let variant = cards.variant_for(animal);
        let base = cascadia_core::scoring::wildlife::score_wildlife(board, animal, variant) as f64;

        for &ti in board.placed_tiles.iter() {
            let idx = ti as usize;
            if !board.grid.get(idx).can_place_wildlife(animal) { continue; }
            let mut b = board.clone();
            if let Some(_wa) = b.place_wildlife(idx, animal) {
                let with = cascadia_core::scoring::wildlife::score_wildlife(&b, animal, variant) as f64;
                let delta = with - base;
                let keystone = if board.grid.get(idx).is_keystone() { 1.0 } else { 0.0 };
                if delta + keystone > type_delta[animal_idx] {
                    type_delta[animal_idx] = delta + keystone;
                    best_placement_idx[animal_idx] = idx;
                }
            }
        }
    }

    let bag_counts: [u32; 5] = [
        bag_info.remaining[0] as u32, bag_info.remaining[1] as u32,
        bag_info.remaining[2] as u32, bag_info.remaining[3] as u32,
        bag_info.remaining[4] as u32,
    ];
    let bag_total: u32 = bag_counts.iter().sum();
    if bag_total < 4 { return 0.0; }

    if depth <= 2 {
        // Exact enumeration for shallow depths
        let mut ev = 0.0f64;
        let mut total_prob = 0.0f64;

        for t0 in 0..5u8 {
            let c0 = bag_counts[t0 as usize]; if c0 == 0 { continue; }
            let p0 = c0 as f64 / bag_total as f64;
            for t1 in 0..5u8 {
                let c1 = bag_counts[t1 as usize] - if t1 == t0 { 1 } else { 0 };
                if c1 == 0 { continue; }
                let p1 = c1 as f64 / (bag_total - 1) as f64;
                for t2 in 0..5u8 {
                    let c2 = bag_counts[t2 as usize]
                        - if t2 == t0 { 1 } else { 0 } - if t2 == t1 { 1 } else { 0 };
                    if c2 == 0 { continue; }
                    let p2 = c2 as f64 / (bag_total - 2) as f64;
                    for t3 in 0..5u8 {
                        let c3 = bag_counts[t3 as usize]
                            - if t3 == t0 { 1 } else { 0 } - if t3 == t1 { 1 } else { 0 }
                            - if t3 == t2 { 1 } else { 0 };
                        if c3 == 0 { continue; }
                        let p3 = c3 as f64 / (bag_total - 3) as f64;
                        let prob = p0 * p1 * p2 * p3;

                        // Best wildlife from this draw
                        let best_type = [t0, t1, t2, t3].iter()
                            .max_by(|&&a, &&b| type_delta[a as usize]
                                .partial_cmp(&type_delta[b as usize])
                                .unwrap_or(std::cmp::Ordering::Equal))
                            .copied().unwrap() as usize;
                        let immediate = type_delta[best_type];

                        // Recurse: place this animal and look deeper
                        let future = if depth > 1 && best_placement_idx[best_type] != usize::MAX {
                            let mut next_board = board.clone();
                            let animal = Wildlife::from_u8(best_type as u8).unwrap();
                            if let Some(_wa) = next_board.place_wildlife(best_placement_idx[best_type], animal) {
                                // Reduce bag counts for recursion
                                let mut next_bag = bag_info.clone();
                                next_bag.remaining[best_type] = next_bag.remaining[best_type].saturating_sub(1);
                                wildlife_deep_eval(&next_board, cards, &next_bag, depth - 1)
                            } else { 0.0 }
                        } else { 0.0 };

                        ev += prob * (immediate + future);
                        total_prob += prob;
                    }
                }
            }
        }
        if total_prob > 0.0 { ev / total_prob } else { 0.0 }
    } else {
        // Greedy single-path for deep plies (no branching)
        // Find expected best type: highest type_delta × probability_of_appearing
        let mut best_type = 0usize;
        let mut best_expected = 0.0f64;
        for t in 0..5 {
            // Probability of seeing at least one of this type in 4 draws
            let p_none = if bag_total >= 4 {
                let ct = bag_counts[t] as f64;
                let bt = bag_total as f64;
                ((bt - ct) / bt) * ((bt - ct - 1.0).max(0.0) / (bt - 1.0))
                    * ((bt - ct - 2.0).max(0.0) / (bt - 2.0))
                    * ((bt - ct - 3.0).max(0.0) / (bt - 3.0))
            } else { 1.0 };
            let p_at_least_one = 1.0 - p_none;
            let expected = type_delta[t] * p_at_least_one;
            if expected > best_expected {
                best_expected = expected;
                best_type = t;
            }
        }

        let immediate = best_expected;

        // Place the expected best and recurse
        let future = if best_placement_idx[best_type] != usize::MAX {
            let mut next_board = board.clone();
            let animal = Wildlife::from_u8(best_type as u8).unwrap();
            if let Some(_wa) = next_board.place_wildlife(best_placement_idx[best_type], animal) {
                let mut next_bag = bag_info.clone();
                next_bag.remaining[best_type] = next_bag.remaining[best_type].saturating_sub(1);
                wildlife_deep_eval(&next_board, cards, &next_bag, depth - 1)
            } else { 0.0 }
        } else { 0.0 };

        immediate + future
    }
}

/// Hybrid: exact expectimax filters to top-K, then MCE rollouts refine.
/// Combines fast deterministic ranking with deep stochastic search.
pub fn best_move_hybrid(
    game: &GameState,
    net: &NNUENetwork,
    num_rollouts: usize,
    top_k: usize,
    rng: &mut StdRng,
) -> Option<ScoredMove> {
    let player = game.current_player;
    let cards = game.scoring_cards;

    // Step 1: Generate candidates with decomposed eval
    let mut candidates = crate::search::candidate_moves_decomposed(game, net);

    // Also add greedy + strategic candidates
    let mp: Vec<_> = game.market.available()
        .map(|(i, p)| (i, p.tile, p.wildlife)).collect();
    let mut board = game.boards[player].clone();
    let greedy_best = best_move_with_potential(&mut board, &mp, &cards, game.turns_remaining);
    if let Some(ref bm) = greedy_best {
        if !candidates.iter().any(|c| c.tile_q == bm.tile_q && c.tile_r == bm.tile_r
            && c.rotation == bm.rotation && c.wildlife_q == bm.wildlife_q) {
            candidates.push(*bm);
        }
    }
    let strategic = wildlife_strategic_candidates(game);
    for sc in &strategic {
        if !candidates.iter().any(|c| c.tile_q == sc.tile_q && c.tile_r == sc.tile_r
            && c.rotation == sc.rotation && c.wildlife_q == sc.wildlife_q) {
            candidates.push(*sc);
        }
    }

    if candidates.is_empty() {
        return greedy_best;
    }

    // Step 2: Score ALL candidates with 2-ply exact expectimax (fast)
    let bag_info = crate::nnue::BagInfo::from_game(game);
    for mv in candidates.iter_mut() {
        let mut g = game.clone();
        if !crate::search::execute_scored_move(&mut g, mv) { continue; }

        // Simulate opponents
        crate::search::advance_opponents(&mut g, player);

        let score = if g.is_game_over() {
            ScoreBreakdown::compute(&mut g.boards[player], &g.scoring_cards).total as f32
        } else {
            let actual = ScoreBreakdown::compute(&mut g.boards[player], &g.scoring_cards).total as f32;
            let remaining = net.evaluate_with_bag(&g.boards[player], &bag_info);
            actual + remaining
        };
        mv.eval = (score * 1000.0) as i32;
    }

    // Step 3: Keep top-K by exact expectimax score
    candidates.sort_by(|a, b| b.eval.cmp(&a.eval));
    candidates.truncate(top_k);

    // Step 4: Run MCE rollouts on the top-K only (same budget, more per candidate)
    let game_arc = std::sync::Arc::new(game.clone());
    let net_arc = std::sync::Arc::new(net.clone());
    let candidates_arc = std::sync::Arc::new(candidates.clone());

    // Sequential halving on the reduced candidate set
    let n_cands = candidates.len();
    let num_rounds = (n_cands as f64).log2().ceil().max(1.0) as usize;
    let rollouts_per_round = (num_rollouts / num_rounds).max(1);

    let mut totals = vec![0u64; n_cands];
    let mut counts = vec![0u32; n_cands];
    let mut alive: Vec<usize> = (0..n_cands).collect();

    for round in 0..num_rounds {
        if alive.is_empty() { break; }
        let per_candidate = (rollouts_per_round / alive.len()).max(1);

        let mut work_items: Vec<(usize, u64)> = Vec::new();
        for &ci in &alive {
            for _ in 0..per_candidate {
                work_items.push((ci, rng.gen()));
            }
        }

        let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let chunk_size = (work_items.len() + num_threads - 1) / num_threads;

        let handles: Vec<_> = work_items.chunks(chunk_size).map(|chunk| {
            let work = chunk.to_vec();
            let game = std::sync::Arc::clone(&game_arc);
            let net = std::sync::Arc::clone(&net_arc);
            let cands = std::sync::Arc::clone(&candidates_arc);

            std::thread::spawn(move || {
                let mut results: Vec<(usize, u64)> = Vec::with_capacity(work.len());
                for &(ci, seed) in &work {
                    let mv = &cands[ci];
                    let mut g = (*game).clone();
                    let mut rollout_rng = StdRng::seed_from_u64(seed);
                    g.shuffle_bags(&mut rollout_rng);
                    if !crate::search::execute_scored_move(&mut g, mv) { continue; }

                    let depth_limit: usize = std::env::var("MCE_DEPTH").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
                    let mut ai_turns = 0;
                    while !g.is_game_over() {
                        if g.current_player != player {
                            if g.can_replace_overflow().is_some() {
                                g.replace_overflow();
                            }
                            match crate::search::greedy_move(&g) {
                                Some(opp) => { if !crate::search::execute_scored_move(&mut g, &opp) { break; } }
                                None => break,
                            }
                            continue;
                        }
                        ai_turns += 1;
                        if ai_turns > depth_limit { break; }
                        match crate::nnue_train::pick_best_move_nnue(&g, &net) {
                            Some(ai_mv) => { if !crate::search::execute_scored_move(&mut g, &ai_mv) { break; } }
                            None => break,
                        }
                    }

                    let score = if g.is_game_over() {
                        ScoreBreakdown::compute(&mut g.boards[player], &g.scoring_cards).total
                    } else {
                        let actual = ScoreBreakdown::compute(&mut g.boards[player], &g.scoring_cards).total;
                        let nval = net.evaluate(&g.boards[player]);
                        let tier = tier_bonus(&g.boards[player]);
                        (actual as f32 + nval.max(0.0) + tier) as u16
                    };
                    results.push((ci, score as u64));
                }
                results
            })
        }).collect();

        for handle in handles {
            for (ci, score) in handle.join().unwrap() {
                totals[ci] += score;
                counts[ci] += 1;
            }
        }

        if round < num_rounds - 1 {
            let mut alive_scores: Vec<(usize, f64)> = alive.iter()
                .filter_map(|&ci| {
                    if counts[ci] == 0 { return None; }
                    Some((ci, totals[ci] as f64 / counts[ci] as f64))
                }).collect();
            alive_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let keep = (alive_scores.len() + 1) / 2;
            alive = alive_scores.into_iter().take(keep).map(|(ci, _)| ci).collect();
        }
    }

    let mut scored: Vec<(ScoredMove, f64)> = candidates.iter().enumerate()
        .filter_map(|(ci, mv)| {
            if counts[ci] == 0 { return None; }
            Some((*mv, totals[ci] as f64 / counts[ci] as f64))
        }).collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().next().map(|(mv, avg)| ScoredMove { score: avg.round() as u16, ..mv })
}

/// Collect MCE-labeled training samples for the current position.
/// For each candidate that got rollouts, returns the afterstate's NNUE features
/// paired with a delta-style label: (avg_rollout_final_score - afterstate_current_score).
/// This is the training signal needed to distill MCE into a plain NNUE.
pub fn collect_mce_samples(
    game: &GameState,
    net: &NNUENetwork,
    num_rollouts: usize,
    rng: &mut StdRng,
) -> Vec<(Vec<u16>, f32)> {
    let scored = run_mce_candidates(game, net, num_rollouts, rng);
    let mut samples = Vec::with_capacity(scored.len());
    for (mv, avg) in scored {
        let mut g = game.clone();
        if !crate::search::execute_scored_move(&mut g, &mv) { continue; }
        let player = game.current_player;
        let current = ScoreBreakdown::compute(
            &mut g.boards[player], &g.scoring_cards,
        ).total as f32;
        let target = (avg as f32 - current).max(0.0);
        let bag_info = crate::nnue::BagInfo::from_game(&g);
        let features = crate::nnue::extract_features_with_bag(&g.boards[player], Some(&bag_info));
        samples.push((features, target));
    }
    samples
}

/// Pick the best move using the MCE candidate pipeline WITHOUT running rollouts.
/// Uses candidate_moves + wildlife_strategic_candidates + greedy + wildlife demand
/// scoring, then picks the top candidate by `eval`. Fast, strong-but-not-MCE-strong.
pub fn best_move_no_rollouts(game: &GameState) -> Option<ScoredMove> {
    let player = game.current_player;
    let mp: Vec<_> = game.market.available()
        .map(|(i, p)| (i, p.tile, p.wildlife)).collect();
    if mp.is_empty() { return None; }

    let cards = game.scoring_cards;
    let turns = game.turns_remaining;
    let mut board = game.boards[player].clone();
    let greedy_best = best_move_with_potential(&mut board, &mp, &cards, turns);

    let mut candidates = candidate_moves_pub(game);
    if let Some(ref bm) = greedy_best {
        if !candidates.iter().any(|c| c.tile_q == bm.tile_q && c.tile_r == bm.tile_r
            && c.rotation == bm.rotation && c.wildlife_q == bm.wildlife_q) {
            candidates.push(*bm);
        }
    }

    let strategic = wildlife_strategic_candidates(game);
    for sc in &strategic {
        if !candidates.iter().any(|c| c.tile_q == sc.tile_q && c.tile_r == sc.tile_r
            && c.rotation == sc.rotation && c.wildlife_q == sc.wildlife_q) {
            candidates.push(*sc);
        }
    }

    let demand = compute_wildlife_demand(&game.boards[player]);
    for mv in candidates.iter_mut() {
        if mv.wildlife_q.is_some() {
            if let Some(&(_, _, wl)) = mp.iter().find(|&&(i, _, _)| i == mv.market_index) {
                let bonus = (demand[wl as usize] * 3.0) as i32;
                mv.eval += bonus;
            }
        }
    }
    candidates.sort_by(|a, b| b.eval.cmp(&a.eval));
    candidates.into_iter().next().or(greedy_best)
}

/// Return the top N candidate moves (sorted by average MCE score descending)
/// along with their average rollout score. `num_rollouts` is the total rollout
/// budget, distributed via sequential halving across candidates.
pub fn top_moves_mce(
    game: &GameState,
    net: &NNUENetwork,
    num_rollouts: usize,
    rng: &mut StdRng,
    top_n: usize,
) -> Vec<(ScoredMove, f64)> {
    let scored = run_mce_candidates(game, net, num_rollouts, rng);
    scored.into_iter().take(top_n).collect()
}

/// Score all candidates with MCE, returning move + features + scores for policy training.
/// Returns: Vec<(ScoredMove, features, mce_avg_score)> for each candidate, sorted best-first.
pub fn mce_candidates_with_features(
    game: &GameState,
    net: &NNUENetwork,
    num_rollouts: usize,
    rng: &mut StdRng,
) -> Vec<(ScoredMove, Vec<u16>, f32)> {
    let player = game.current_player;
    let bag_info = crate::nnue::BagInfo::from_game(game);
    let scored = run_mce_candidates(game, net, num_rollouts, rng);

    scored.into_iter().map(|(mv, avg)| {
        let mut g = game.clone();
        crate::search::execute_scored_move(&mut g, &mv);
        let features = crate::nnue::extract_features_with_bag(&g.boards[player], Some(&bag_info));
        (mv, features, avg as f32)
    }).collect()
}

/// Pick the best move using parallel Monte Carlo evaluation.
/// `num_rollouts` is the total rollout budget (default 750), distributed
/// via sequential halving across candidates.
pub fn best_move_mce(
    game: &GameState,
    net: &NNUENetwork,
    num_rollouts: usize,
    rng: &mut StdRng,
) -> Option<ScoredMove> {
    let scored = run_mce_candidates(game, net, num_rollouts, rng);
    scored.into_iter().next().map(|(mv, avg)| ScoredMove { score: avg.round() as u16, ..mv })
}

/// MCE with policy-guided candidate pruning.
/// Uses PolicyNetwork to rank candidates, keeps top_k for MCE evaluation.
/// Uses the FULL original MCE pipeline (strategic candidates, demand scoring, NNUE re-ranking)
/// with policy pruning injected after NNUE re-ranking.
pub fn best_move_mce_with_policy(
    game: &GameState,
    net: &NNUENetwork,
    policy_net: &crate::nnue::PolicyNetwork,
    num_rollouts: usize,
    top_k: usize,
    rng: &mut StdRng,
) -> Option<ScoredMove> {
    let scored = run_mce_candidates_impl(game, net, Some(policy_net), top_k, num_rollouts, rng);
    scored.into_iter().next().map(|(mv, avg)| ScoredMove { score: avg.round() as u16, ..mv })
}

/// Run MCE with policy-guided candidate filtering.
fn run_mce_candidates_with_policy(
    game: &GameState,
    net: &NNUENetwork,
    policy_net: &crate::nnue::PolicyNetwork,
    num_rollouts: usize,
    top_k: usize,
    rng: &mut StdRng,
) -> Vec<(ScoredMove, f64)> {
    let player = game.current_player;
    let mp: Vec<_> = game.market.available()
        .map(|(i, p)| (i, p.tile, p.wildlife)).collect();
    if mp.is_empty() { return Vec::new(); }

    let cards = game.scoring_cards;
    let turns = game.turns_remaining;
    let mut board = game.boards[player].clone();
    let greedy_best = best_move_with_potential(&mut board, &mp, &cards, turns);

    let mut candidates = crate::search::candidate_moves_decomposed(game, net);

    // Add greedy best if not already present
    if let Some(ref bm) = greedy_best {
        if !candidates.iter().any(|c| c.tile_q == bm.tile_q && c.tile_r == bm.tile_r
            && c.rotation == bm.rotation && c.wildlife_q == bm.wildlife_q) {
            candidates.push(*bm);
        }
    }

    if candidates.is_empty() { return Vec::new(); }

    // Score all candidates with PolicyNet
    let bag_info = crate::nnue::BagInfo::from_game(game);
    let mut scored_candidates: Vec<(usize, f32)> = candidates.iter().enumerate().map(|(i, mv)| {
        let mut g = game.clone();
        if !crate::search::execute_scored_move(&mut g, mv) {
            return (i, f32::NEG_INFINITY);
        }
        let features = crate::nnue::extract_features_with_bag(&g.boards[player], Some(&bag_info));
        let logit = policy_net.forward(&features);
        (i, logit)
    }).collect();

    // Sort by policy score descending, keep top_k
    scored_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored_candidates.truncate(top_k);

    // Filter candidates to top_k
    let filtered: Vec<ScoredMove> = scored_candidates.iter()
        .map(|&(idx, _)| candidates[idx])
        .collect();

    // Run standard MCE on the filtered set
    run_mce_candidates_on(game, net, &filtered, num_rollouts, rng)
}

/// Run MCE rollouts (sequential halving) on a pre-selected set of candidates.
fn run_mce_candidates_on(
    game: &GameState,
    net: &NNUENetwork,
    candidates: &[ScoredMove],
    num_rollouts: usize,
    rng: &mut StdRng,
) -> Vec<(ScoredMove, f64)> {
    let player = game.current_player;
    if candidates.is_empty() { return Vec::new(); }

    let game_arc = std::sync::Arc::new(game.clone());
    let net_arc = std::sync::Arc::new(net.clone());
    let candidates_arc = std::sync::Arc::new(candidates.to_vec());

    let n_cands = candidates.len();
    let num_rounds = (n_cands as f64).log2().ceil().max(1.0) as usize;
    let rollouts_per_round = (num_rollouts / num_rounds).max(1);

    let mut totals = vec![0u64; n_cands];
    let mut counts = vec![0u32; n_cands];
    let mut alive: Vec<usize> = (0..n_cands).collect();

    for round in 0..num_rounds {
        if alive.is_empty() { break; }
        let per_candidate = (rollouts_per_round / alive.len()).max(1);

        let mut work_items: Vec<(usize, u64)> = Vec::new();
        for &ci in &alive {
            for _ in 0..per_candidate {
                work_items.push((ci, rng.gen()));
            }
        }

        let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let chunk_size = (work_items.len() + num_threads - 1) / num_threads;

        let handles: Vec<_> = work_items.chunks(chunk_size).map(|chunk| {
            let work = chunk.to_vec();
            let game = std::sync::Arc::clone(&game_arc);
            let net = std::sync::Arc::clone(&net_arc);
            let cands = std::sync::Arc::clone(&candidates_arc);

            std::thread::spawn(move || {
                let mut results: Vec<(usize, u64)> = Vec::with_capacity(work.len());
                for &(ci, seed) in &work {
                    let mv = &cands[ci];
                    let mut g = (*game).clone();
                    let mut rollout_rng = StdRng::seed_from_u64(seed);
                    g.shuffle_bags(&mut rollout_rng);
                    if !crate::search::execute_scored_move(&mut g, mv) { continue; }

                    let depth_limit: usize = std::env::var("MCE_DEPTH").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
                    let mut ai_turns = 0;
                    while !g.is_game_over() {
                        if g.current_player != player {
                            if g.can_replace_overflow().is_some() {
                                g.replace_overflow();
                            }
                            match crate::search::greedy_move(&g) {
                                Some(opp) => { if !crate::search::execute_scored_move(&mut g, &opp) { break; } }
                                None => break,
                            }
                            continue;
                        }
                        ai_turns += 1;
                        if ai_turns > depth_limit { break; }
                        match crate::search::greedy_move(&g) {
                            Some(ai_mv) => { if !crate::search::execute_scored_move(&mut g, &ai_mv) { break; } }
                            None => break,
                        }
                    }

                    let score = if g.is_game_over() {
                        ScoreBreakdown::compute(&mut g.boards[player], &g.scoring_cards).total
                    } else {
                        let actual = ScoreBreakdown::compute(&mut g.boards[player], &g.scoring_cards).total;
                        let nval = net.evaluate(&g.boards[player]);
                        let tier = tier_bonus(&g.boards[player]);
                        (actual as f32 + nval.max(0.0) + tier) as u16
                    };
                    results.push((ci, score as u64));
                }
                results
            })
        }).collect();

        for handle in handles {
            for (ci, score) in handle.join().unwrap() {
                totals[ci] += score;
                counts[ci] += 1;
            }
        }

        if round < num_rounds - 1 {
            let mut alive_scores: Vec<(usize, f64)> = alive.iter()
                .filter_map(|&ci| {
                    if counts[ci] == 0 { return None; }
                    Some((ci, totals[ci] as f64 / counts[ci] as f64))
                }).collect();
            alive_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let keep = (alive_scores.len() + 1) / 2;
            alive = alive_scores.into_iter().take(keep).map(|(ci, _)| ci).collect();
        }
    }

    let mut scored: Vec<(ScoredMove, f64)> = candidates.iter().enumerate()
        .filter_map(|(ci, mv)| {
            if counts[ci] == 0 { return None; }
            Some((*mv, totals[ci] as f64 / counts[ci] as f64))
        }).collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
}

/// Candidate source tag for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CandidateSource {
    CandidateMoves,
    Greedy,
    Strategic,
}

/// Diagnostic stats from one MCE run.
#[derive(Debug, Default)]
pub struct MceDiagnostics {
    pub total_candidates: usize,
    pub from_candidate_moves: usize,
    pub from_greedy: usize,
    pub from_strategic: usize,
    pub winner_source: Option<CandidateSource>,
    pub winner_pre_rank: usize, // 0-indexed rank in eval-sorted order before MCE
    pub rank_correlation: f64,  // Spearman correlation between eval rank and MCE rank
}

use std::sync::Mutex;
use std::sync::OnceLock;
static MCE_DIAGNOSTICS: OnceLock<Mutex<Vec<MceDiagnostics>>> = OnceLock::new();

pub fn take_diagnostics() -> Vec<MceDiagnostics> {
    let mutex = MCE_DIAGNOSTICS.get_or_init(|| Mutex::new(Vec::new()));
    std::mem::take(&mut *mutex.lock().unwrap())
}

fn record_diagnostic(diag: MceDiagnostics) {
    let mutex = MCE_DIAGNOSTICS.get_or_init(|| Mutex::new(Vec::new()));
    mutex.lock().unwrap().push(diag);
}

/// Internal: run MCE and return all scored candidates sorted by avg score descending.
fn run_mce_candidates(
    game: &GameState,
    net: &NNUENetwork,
    num_rollouts: usize,
    rng: &mut StdRng,
) -> Vec<(ScoredMove, f64)> {
    run_mce_candidates_impl(game, net, None, 15, num_rollouts, rng)
}

fn run_mce_candidates_impl(
    game: &GameState,
    net: &NNUENetwork,
    policy_net: Option<&crate::nnue::PolicyNetwork>,
    top_k: usize,
    num_rollouts: usize,
    rng: &mut StdRng,
) -> Vec<(ScoredMove, f64)> {
    // Note: pre-move optimization (mulligans) happens in simulate_game before
    // this is called. Here we just pick the best move for the current market.
    let player = game.current_player;

    // Get candidates
    let mp: Vec<_> = game.market.available()
        .map(|(i, p)| (i, p.tile, p.wildlife)).collect();
    if mp.is_empty() { return Vec::new(); }

    let cards = game.scoring_cards;
    let turns = game.turns_remaining;
    let mut board = game.boards[player].clone();
    let greedy_best = best_move_with_potential(&mut board, &mp, &cards, turns);

    let mut candidates = crate::search::candidate_moves_decomposed(game, net);
    let num_candidate_moves = candidates.len();

    let mut greedy_added = false;
    if let Some(ref bm) = greedy_best {
        if !candidates.iter().any(|c| c.tile_q == bm.tile_q && c.tile_r == bm.tile_r
            && c.rotation == bm.rotation && c.wildlife_q == bm.wildlife_q) {
            candidates.push(*bm);
            greedy_added = true;
        }
    }

    // Inject wildlife-strategic candidates (pattern-extending moves greedy misses)
    let strategic = wildlife_strategic_candidates(game);
    let strategic_start = candidates.len();
    for sc in &strategic {
        if !candidates.iter().any(|c| c.tile_q == sc.tile_q && c.tile_r == sc.tile_r
            && c.rotation == sc.rotation && c.wildlife_q == sc.wildlife_q) {
            candidates.push(*sc);
        }
    }
    let num_strategic = candidates.len() - strategic_start;

    // Build source tags before sorting
    let mut sources: Vec<CandidateSource> = Vec::with_capacity(candidates.len());
    for i in 0..candidates.len() {
        if i < num_candidate_moves {
            sources.push(CandidateSource::CandidateMoves);
        } else if greedy_added && i == num_candidate_moves {
            sources.push(CandidateSource::Greedy);
        } else {
            sources.push(CandidateSource::Strategic);
        }
    }

    // Compute wildlife demand — what does the board need most?
    let demand = compute_wildlife_demand(&game.boards[player]);

    // Re-sort candidates by greedy score + demand bonus for the wildlife they supply
    for mv in candidates.iter_mut() {
        if let Some(wq) = mv.wildlife_q {
            // Figure out which wildlife this candidate places
            // by matching market index to wildlife type
            if let Some(&(_, _, wl)) = mp.iter().find(|&&(i, _, _)| i == mv.market_index) {
                let bonus = (demand[wl as usize] * 3.0) as i32; // scale demand to eval units
                mv.eval += bonus;
            }
        }
    }

    // Re-rank with NNUE afterstate evaluation for better initial ordering.
    // If MCE_RANK_EXPECTIMAX is set, use exact 1-ply wildlife enumeration on top of NNUE
    // (more accurate but ~250us vs 10us per candidate).
    let bag_info = crate::nnue::BagInfo::from_game(game);
    let use_rank_expectimax = std::env::var("MCE_RANK_EXPECTIMAX").is_ok();
    for mv in candidates.iter_mut() {
        let coord = cascadia_core::hex::HexCoord::new(mv.tile_q, mv.tile_r);
        let tile = mp.iter().find(|&&(i, _, _)| i == mv.market_index).map(|&(_, t, _)| t);
        let wildlife = mp.iter().find(|&&(i, _, _)| {
            i == mv.wildlife_market_index.unwrap_or(mv.market_index)
        }).map(|&(_, _, w)| w);
        if let Some(tile) = tile {
            // Construct the after-state game so we can call evaluate_leaf_with_next_market
            let mut after_game = game.clone();
            if !crate::search::execute_scored_move(&mut after_game, mv) {
                // Fallback to legacy NNUE-only ranking if execution fails
                let mut eval_board = game.boards[player].clone();
                if eval_board.place_tile(coord, tile, mv.rotation).is_some() {
                    if let (Some(wq), Some(wr), Some(wl)) = (mv.wildlife_q, mv.wildlife_r, wildlife) {
                        let wcoord = cascadia_core::hex::HexCoord::new(wq, wr);
                        if let Some(widx) = wcoord.to_index() {
                            eval_board.place_wildlife(widx, wl);
                        }
                    }
                    let actual = ScoreBreakdown::compute(&mut eval_board, &cards).total as f32;
                    let remaining = net.evaluate_with_bag(&eval_board, &bag_info);
                    mv.eval = ((actual + remaining) * 1000.0) as i32;
                }
                continue;
            }
            if use_rank_expectimax {
                // Need the AI's afterstate, not the post-opponent state. execute_scored_move
                // advances current_player. Use the previous player index (the AI).
                let v = evaluate_leaf_with_next_market(&after_game, player, net);
                mv.eval = (v * 1000.0) as i32;
            } else {
                let mut eval_board = game.boards[player].clone();
                if eval_board.place_tile(coord, tile, mv.rotation).is_some() {
                    if let (Some(wq), Some(wr), Some(wl)) = (mv.wildlife_q, mv.wildlife_r, wildlife) {
                        let wcoord = cascadia_core::hex::HexCoord::new(wq, wr);
                        if let Some(widx) = wcoord.to_index() {
                            eval_board.place_wildlife(widx, wl);
                        }
                    }
                    let actual = ScoreBreakdown::compute(&mut eval_board, &cards).total as f32;
                    let remaining = net.evaluate_with_bag(&eval_board, &bag_info);
                    mv.eval = ((actual + remaining) * 1000.0) as i32;
                }
            }
        }
    }

    // Sort candidates + sources together by NNUE-based eval.
    // If MCE_GUMBEL_TOPK is set, perturb eval with Gumbel(0, T) noise before sorting,
    // implementing Gumbel-top-k stochastic sampling per Danihelka et al. (ICLR 2022).
    // This gives lower-ranked candidates a chance at survival, addressing the
    // observed 0.349 Spearman correlation between eval rank and MCE rank.
    let use_gumbel_topk = std::env::var("MCE_GUMBEL_TOPK").is_ok();
    let gumbel_temp: f64 = std::env::var("MCE_GUMBEL_TEMP")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(3000.0);
    let mut indexed: Vec<(usize, f64)> = if use_gumbel_topk {
        // Sample independent Gumbel(0,1) using inverse-CDF: -log(-log(U))
        candidates.iter().enumerate().map(|(i, c)| {
            let u: f64 = rng.gen_range(1e-12..1.0);
            let g: f64 = -(-u.ln()).ln();
            (i, c.eval as f64 + gumbel_temp * g)
        }).collect()
    } else {
        candidates.iter().enumerate().map(|(i, c)| (i, c.eval as f64)).collect()
    };
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let sorted_candidates: Vec<ScoredMove> = indexed.iter().map(|&(i, _)| candidates[i]).collect();
    let sorted_sources: Vec<CandidateSource> = indexed.iter().map(|&(i, _)| sources[i]).collect();
    candidates = sorted_candidates;
    sources = sorted_sources;

    // Keep top candidates after NNUE re-ranking (configurable via MCE_CANDIDATES env var)
    let max_candidates: usize = std::env::var("MCE_CANDIDATES").ok().and_then(|s| s.parse().ok()).unwrap_or(15);
    candidates.truncate(max_candidates);
    sources.truncate(max_candidates);

    // Optional: policy-guided pruning to top_k (after NNUE re-ranking)
    if let Some(pnet) = policy_net {
        if candidates.len() > top_k {
            let candidate_features: Vec<Vec<u16>> = candidates.iter().map(|mv| {
                let mut g = game.clone();
                if crate::search::execute_scored_move(&mut g, mv) {
                    crate::nnue::extract_features_with_bag(&g.boards[player], Some(&bag_info))
                } else {
                    vec![]
                }
            }).collect();
            let probs = pnet.rank_candidates(&candidate_features);
            let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.truncate(top_k);
            let kept: Vec<usize> = indexed.iter().map(|&(i, _)| i).collect();
            candidates = kept.iter().map(|&i| candidates[i]).collect();
            sources = kept.iter().map(|&i| sources[i]).collect();
        }
    } else {
        // No policy: use top_k from NNUE ranking
        candidates.truncate(top_k);
        sources.truncate(top_k);
    }

    if candidates.is_empty() {
        return greedy_best.into_iter().map(|m| (m, m.score as f64)).collect();
    }

    // === Sequential Halving ===
    // Distribute the total rollout budget across elimination rounds.
    // Each round, run rollouts on surviving candidates, then drop the bottom half.
    // This concentrates compute on the most promising candidates.

    let total_budget = num_rollouts; // total rollouts to spend across all candidates
    let n_cands = candidates.len();

    // Compute sequential halving schedule:
    // rounds = ceil(log2(n_cands)), rollouts_per_round = budget / rounds
    let num_rounds = (n_cands as f64).log2().ceil().max(1.0) as usize;
    let rollouts_per_round = (total_budget / num_rounds).max(1);

    let game_arc = Arc::new(game.clone());
    let net_arc = Arc::new(net.clone());

    // Track cumulative totals/counts across rounds (scores accumulate)
    let mut totals = vec![0u64; n_cands];
    let mut counts = vec![0u32; n_cands];
    // Which candidate indices are still alive
    let mut alive: Vec<usize> = (0..n_cands).collect();

    for round in 0..num_rounds {
        if alive.is_empty() { break; }

        // Distribute rollouts evenly among alive candidates
        let per_candidate = (rollouts_per_round / alive.len()).max(1);

        // Build work items for this round
        let mut work_items: Vec<(usize, u64)> = Vec::new();
        for &ci in &alive {
            for _ in 0..per_candidate {
                work_items.push((ci, rng.gen()));
            }
        }

        // Run rollouts in parallel
        let num_threads = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let chunk_size = (work_items.len() + num_threads - 1) / num_threads;
        let candidates_arc = Arc::new(candidates.clone());

        let handles: Vec<_> = work_items
            .chunks(chunk_size)
            .map(|chunk| {
                let work = chunk.to_vec();
                let game = Arc::clone(&game_arc);
                let net = Arc::clone(&net_arc);
                let cands = Arc::clone(&candidates_arc);
                let player = player;
                let use_expectimax_rollouts = std::env::var("MCE_EXPECTIMAX_ROLLOUTS").is_ok();
                let use_leaf_expectimax = std::env::var("MCE_LEAF_EXPECTIMAX").is_ok();
                let use_leaf_expectimax2 = std::env::var("MCE_LEAF_EXPECTIMAX2").is_ok();
                let use_leaf_market_aware = std::env::var("MCE_LEAF_MARKET").is_ok();
                let use_leaf_market2 = std::env::var("MCE_LEAF_MARKET2").is_ok();

                thread::spawn(move || {
                    let mut results: Vec<(usize, u64)> = Vec::with_capacity(work.len());

                    for &(ci, seed) in &work {
                        let mv = &cands[ci];
                        let mut g = (*game).clone();
                        let mut rollout_rng = StdRng::seed_from_u64(seed);
                        g.shuffle_bags(&mut rollout_rng);

                        if !execute_scored_move(&mut g, mv) {
                            continue;
                        }

                        let depth_limit: usize = std::env::var("MCE_DEPTH").ok().and_then(|s| s.parse().ok()).unwrap_or(6);
                        let mut ai_turns_played = 0;

                        while !g.is_game_over() {
                            if g.current_player != player {
                                // Opponents also take the free 3-of-a-kind replacement
                                if g.can_replace_overflow().is_some() {
                                    g.replace_overflow();
                                }
                                match greedy_move(&g) {
                                    Some(opp_mv) => {
                                        if !execute_scored_move(&mut g, &opp_mv) { break; }
                                    }
                                    None => break,
                                }
                                continue;
                            }

                            ai_turns_played += 1;
                            if ai_turns_played > depth_limit {
                                break;
                            }

                            // Free overflow: always take it in rollouts (fast, no eval needed)
                            if g.can_replace_overflow().is_some() {
                                g.replace_overflow();
                            }

                            // Rollout move policy: greedy is ~20× faster than NNUE candidates
                            // MCE accuracy comes from root candidate selection + rollout count,
                            // not from perfect rollout play.
                            let ai_mv = if use_expectimax_rollouts {
                                best_move_expectimax_1ply(&g, &net)
                            } else {
                                greedy_move(&g)
                            };
                            match ai_mv {
                                Some(ai_mv) => {
                                    if !execute_scored_move(&mut g, &ai_mv) { break; }
                                }
                                None => break,
                            }
                        }

                        let score = if g.is_game_over() {
                            ScoreBreakdown::compute(&mut g.boards[player], &g.scoring_cards).total
                        } else if use_leaf_market2 {
                            // 2-step market-aware leaf
                            let v = evaluate_leaf_market_aware_2step(&g, player, &net);
                            let tier = tier_bonus(&g.boards[player]);
                            (v + tier) as u16
                        } else if use_leaf_market_aware {
                            // Market-aware leaf: max over actual market wildlife
                            let v = evaluate_leaf_market_aware(&g, player, &net);
                            let tier = tier_bonus(&g.boards[player]);
                            (v + tier) as u16
                        } else if use_leaf_expectimax2 {
                            // 2-step lookahead leaf eval (chained wildlife placement)
                            let v = evaluate_leaf_with_next_2_markets(&g, player, &net);
                            let tier = tier_bonus(&g.boards[player]);
                            (v + tier) as u16
                        } else if use_leaf_expectimax {
                            // Stronger leaf eval: NNUE + exact 1-ply wildlife enumeration
                            let v = evaluate_leaf_with_next_market(&g, player, &net);
                            let tier = tier_bonus(&g.boards[player]);
                            (v + tier) as u16
                        } else {
                            let actual = ScoreBreakdown::compute(&mut g.boards[player], &g.scoring_cards).total;
                            let bag_info = crate::nnue::BagInfo::from_game(&g);
                            let nval = net.evaluate_with_bag(&g.boards[player], &bag_info);
                            let tier = tier_bonus(&g.boards[player]);
                            (actual as f32 + nval.max(0.0) + tier) as u16
                        };

                        results.push((ci, score as u64));
                    }

                    results
                })
            })
            .collect();

        // Aggregate this round's results
        for handle in handles {
            for (ci, score) in handle.join().unwrap() {
                totals[ci] += score;
                counts[ci] += 1;
            }
        }

        // Eliminate bottom half (keep top ceil(alive/2))
        if round < num_rounds - 1 {
            let mut alive_scores: Vec<(usize, f64)> = alive.iter()
                .filter_map(|&ci| {
                    if counts[ci] == 0 { return None; }
                    Some((ci, totals[ci] as f64 / counts[ci] as f64))
                })
                .collect();
            alive_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let keep = (alive_scores.len() + 1) / 2; // ceil(n/2)
            alive = alive_scores.into_iter().take(keep).map(|(ci, _)| ci).collect();
        }
    }

    // Collect averages with original eval-sort index
    let mut scored: Vec<(usize, ScoredMove, f64)> = candidates.iter().enumerate()
        .filter_map(|(ci, mv)| {
            if counts[ci] == 0 { return None; }
            let avg = totals[ci] as f64 / counts[ci] as f64;
            Some((ci, *mv, avg))
        })
        .collect();
    scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Record diagnostics
    if !scored.is_empty() {
        let winner_idx = scored[0].0;
        let winner_source = if winner_idx < sources.len() {
            Some(sources[winner_idx])
        } else {
            None
        };

        // Spearman rank correlation: compare eval rank (0..n) with MCE rank
        let n = scored.len();
        let spearman = if n >= 2 {
            // MCE rank order: scored[0] is MCE #1, scored[1] is MCE #2, etc.
            // Each scored[i].0 is its original eval-sort rank
            let mut d2_sum = 0.0f64;
            for (mce_rank, &(eval_rank, _, _)) in scored.iter().enumerate() {
                let d = mce_rank as f64 - eval_rank as f64;
                d2_sum += d * d;
            }
            1.0 - (6.0 * d2_sum) / (n as f64 * (n as f64 * n as f64 - 1.0))
        } else {
            1.0
        };

        record_diagnostic(MceDiagnostics {
            total_candidates: candidates.len(),
            from_candidate_moves: sources.iter().filter(|&&s| s == CandidateSource::CandidateMoves).count(),
            from_greedy: sources.iter().filter(|&&s| s == CandidateSource::Greedy).count(),
            from_strategic: sources.iter().filter(|&&s| s == CandidateSource::Strategic).count(),
            winner_source,
            winner_pre_rank: winner_idx,
            rank_correlation: spearman,
        });
    }

    scored.into_iter().map(|(_, mv, avg)| (mv, avg)).collect()
}

/// 2-step lookahead leaf eval: chains TWO turns of optimal wildlife placement.
///
/// Step 1: For each animal type, compute best placement delta on current board.
/// Step 2: For the BEST type chosen at step 1, place it then recompute step-1
///         on the updated board (with adjusted bag).
/// Returns: actual + NNUE_remaining + E[step1_delta] + E[step2_delta_after_step1].
///
/// Effectively gives 2 turns of optimal wildlife lookahead at the rollout terminal.
fn evaluate_leaf_with_next_2_markets(
    g: &GameState,
    player: usize,
    net: &NNUENetwork,
) -> f32 {
    let cards = g.scoring_cards;
    let actual = ScoreBreakdown::compute(&mut g.boards[player].clone(), &cards).total as f32;
    let bag_info = crate::nnue::BagInfo::from_game(g);
    let nnue_remaining = net.evaluate_with_bag(&g.boards[player], &bag_info).max(0.0);

    // Step 1: best wildlife delta + best placement index per type
    let mut working = g.boards[player].clone();
    let positions: arrayvec::ArrayVec<u16, 64> = working.placed_tiles.iter().copied().collect();
    let mut step1_value = [0.0f32; 5];
    let mut step1_best_idx = [usize::MAX; 5];

    for animal_idx in 0..5 {
        let animal = Wildlife::from_u8(animal_idx as u8).unwrap();
        let variant = cards.variant_for(animal);
        let base_wl = cascadia_core::scoring::wildlife::score_wildlife(&working, animal, variant) as f32;

        let mut best_delta = 0.0f32;
        for &ti in &positions {
            let idx = ti as usize;
            if !working.grid.get(idx).can_place_wildlife(animal) { continue; }
            if let Some(wa) = working.place_wildlife(idx, animal) {
                let with = cascadia_core::scoring::wildlife::score_wildlife(&working, animal, variant) as f32;
                let delta = with - base_wl;
                let keystone = if working.grid.get(idx).is_keystone() { 1.0 } else { 0.0 };
                if delta + keystone > best_delta {
                    best_delta = delta + keystone;
                    step1_best_idx[animal_idx] = idx;
                }
                working.undo(wa);
            }
        }
        step1_value[animal_idx] = best_delta;
    }

    let bag_counts: [u32; 5] = [
        bag_info.remaining[0] as u32, bag_info.remaining[1] as u32,
        bag_info.remaining[2] as u32, bag_info.remaining[3] as u32,
        bag_info.remaining[4] as u32,
    ];
    let bag_total: u32 = bag_counts.iter().sum();

    if bag_total < 8 {
        // Not enough remaining bag for 2-step — fall back to step 1 only
        let step1_ev = expected_best_wildlife_4draws(&step1_value, &bag_counts, bag_total);
        return actual + nnue_remaining + step1_ev;
    }

    // Step 2: for each step1 choice, compute step-1 on the post-step1 board
    let mut step2_value: [[f32; 5]; 5] = [[0.0; 5]; 5];
    for s1_animal in 0..5 {
        if step1_best_idx[s1_animal] == usize::MAX { continue; }
        let animal1 = Wildlife::from_u8(s1_animal as u8).unwrap();
        if let Some(wa) = working.place_wildlife(step1_best_idx[s1_animal], animal1) {
            for s2_animal in 0..5 {
                let animal2 = Wildlife::from_u8(s2_animal as u8).unwrap();
                let variant2 = cards.variant_for(animal2);
                let base_wl2 = cascadia_core::scoring::wildlife::score_wildlife(&working, animal2, variant2) as f32;

                let mut best_delta2 = 0.0f32;
                for &ti in &positions {
                    let idx = ti as usize;
                    if !working.grid.get(idx).can_place_wildlife(animal2) { continue; }
                    if let Some(wa2) = working.place_wildlife(idx, animal2) {
                        let with2 = cascadia_core::scoring::wildlife::score_wildlife(&working, animal2, variant2) as f32;
                        let delta2 = with2 - base_wl2;
                        let keystone2 = if working.grid.get(idx).is_keystone() { 1.0 } else { 0.0 };
                        if delta2 + keystone2 > best_delta2 {
                            best_delta2 = delta2 + keystone2;
                        }
                        working.undo(wa2);
                    }
                }
                step2_value[s1_animal][s2_animal] = best_delta2;
            }
            working.undo(wa);
        }
    }

    // For each (drawn type1, drawn type2), the AI picks the best path
    // Exact joint enumeration: 5^4 step-1 draws × identify best step-1 type → use that type's
    // step-2 deltas → 5^4 step-2 draws.  Total ~390K ops per leaf.
    let mut joint_ev = 0.0f32;
    let mut total_prob = 0.0f32;
    for t0 in 0..5u8 {
        let c0 = bag_counts[t0 as usize]; if c0 == 0 { continue; }
        let p0 = c0 as f32 / bag_total as f32;
        for t1 in 0..5u8 {
            let c1 = bag_counts[t1 as usize] - if t1 == t0 { 1 } else { 0 };
            if c1 == 0 { continue; }
            let p1 = c1 as f32 / (bag_total - 1) as f32;
            for t2 in 0..5u8 {
                let c2 = bag_counts[t2 as usize]
                    - if t2 == t0 { 1 } else { 0 } - if t2 == t1 { 1 } else { 0 };
                if c2 == 0 { continue; }
                let p2 = c2 as f32 / (bag_total - 2) as f32;
                for t3 in 0..5u8 {
                    let c3 = bag_counts[t3 as usize]
                        - if t3 == t0 { 1 } else { 0 } - if t3 == t1 { 1 } else { 0 }
                        - if t3 == t2 { 1 } else { 0 };
                    if c3 == 0 { continue; }
                    let p3 = c3 as f32 / (bag_total - 3) as f32;
                    let prob_step1 = p0 * p1 * p2 * p3;
                    // AI picks the type with the best step-1 wildlife value
                    let types = [t0 as usize, t1 as usize, t2 as usize, t3 as usize];
                    let best_t1 = *types.iter().max_by(|&&a, &&b| {
                        step1_value[a].partial_cmp(&step1_value[b]).unwrap_or(std::cmp::Ordering::Equal)
                    }).unwrap();
                    let immediate1 = step1_value[best_t1];
                    // Step 2: bag has been reduced by the chosen type
                    let mut bag2 = bag_counts;
                    if bag2[best_t1] > 0 { bag2[best_t1] -= 1; }
                    let total2 = bag_total - 1;
                    let step2_ev_inner = expected_best_wildlife_4draws(&step2_value[best_t1], &bag2, total2);
                    joint_ev += prob_step1 * (immediate1 + step2_ev_inner);
                    total_prob += prob_step1;
                }
            }
        }
    }
    let two_step_ev = if total_prob > 0.0 { joint_ev / total_prob } else { 0.0 };

    actual + nnue_remaining + two_step_ev
}

/// Compute E[max wildlife_value over 4 iid draws from bag] using exact 5^4 enumeration.
fn expected_best_wildlife_4draws(
    wildlife_value: &[f32; 5],
    bag_counts: &[u32; 5],
    bag_total: u32,
) -> f32 {
    if bag_total < 4 { return 0.0; }
    let mut ev = 0.0f32;
    let mut total_prob = 0.0f32;
    for t0 in 0..5u8 {
        let c0 = bag_counts[t0 as usize];
        if c0 == 0 { continue; }
        let p0 = c0 as f32 / bag_total as f32;
        for t1 in 0..5u8 {
            let c1 = bag_counts[t1 as usize] - if t1 == t0 { 1 } else { 0 };
            if c1 == 0 { continue; }
            let p1 = c1 as f32 / (bag_total - 1) as f32;
            for t2 in 0..5u8 {
                let c2 = bag_counts[t2 as usize]
                    - if t2 == t0 { 1 } else { 0 }
                    - if t2 == t1 { 1 } else { 0 };
                if c2 == 0 { continue; }
                let p2 = c2 as f32 / (bag_total - 2) as f32;
                for t3 in 0..5u8 {
                    let c3 = bag_counts[t3 as usize]
                        - if t3 == t0 { 1 } else { 0 }
                        - if t3 == t1 { 1 } else { 0 }
                        - if t3 == t2 { 1 } else { 0 };
                    if c3 == 0 { continue; }
                    let p3 = c3 as f32 / (bag_total - 3) as f32;
                    let prob = p0 * p1 * p2 * p3;
                    let best_wl = wildlife_value[t0 as usize]
                        .max(wildlife_value[t1 as usize])
                        .max(wildlife_value[t2 as usize])
                        .max(wildlife_value[t3 as usize]);
                    ev += prob * best_wl;
                    total_prob += prob;
                }
            }
        }
    }
    if total_prob > 0.0 { ev / total_prob } else { 0.0 }
}

/// 2-step market-aware leaf: chains two turns of optimal wildlife placement.
///
/// Step 1: AI picks best wildlife from CURRENT market.
/// Step 2: market loses chosen pair, refills 1 fresh from bag (chance node).
///         AI picks best of (3 leftover + 1 fresh) using post-step1 wildlife values.
///
/// Computes:
///   actual + NNUE(remaining) + step1_value + E[step2_value]
fn evaluate_leaf_market_aware_2step(
    g: &GameState,
    player: usize,
    net: &NNUENetwork,
) -> f32 {
    let cards = g.scoring_cards;
    let actual = ScoreBreakdown::compute(&mut g.boards[player].clone(), &cards).total as f32;
    let bag_info = crate::nnue::BagInfo::from_game(g);
    let nnue_remaining = net.evaluate_with_bag(&g.boards[player], &bag_info).max(0.0);

    // Step 1: compute wildlife_value per type and best placement index per type
    let mut working = g.boards[player].clone();
    let positions: arrayvec::ArrayVec<u16, 64> = working.placed_tiles.iter().copied().collect();
    let mut step1_value = [0.0f32; 5];
    let mut step1_best_idx = [usize::MAX; 5];

    for animal_idx in 0..5 {
        let animal = Wildlife::from_u8(animal_idx as u8).unwrap();
        let variant = cards.variant_for(animal);
        let base_wl = cascadia_core::scoring::wildlife::score_wildlife(&working, animal, variant) as f32;
        let mut best_delta = 0.0f32;
        for &ti in &positions {
            let idx = ti as usize;
            if !working.grid.get(idx).can_place_wildlife(animal) { continue; }
            if let Some(wa) = working.place_wildlife(idx, animal) {
                let with = cascadia_core::scoring::wildlife::score_wildlife(&working, animal, variant) as f32;
                let delta = with - base_wl;
                let keystone = if working.grid.get(idx).is_keystone() { 1.0 } else { 0.0 };
                if delta + keystone > best_delta {
                    best_delta = delta + keystone;
                    step1_best_idx[animal_idx] = idx;
                }
                working.undo(wa);
            }
        }
        step1_value[animal_idx] = best_delta;
    }

    // AI's step-1 choice: best wildlife in CURRENT market
    let market_wildlife: Vec<Wildlife> = g.market.pairs.iter().flatten()
        .map(|p| p.wildlife)
        .collect();
    if market_wildlife.is_empty() {
        return actual + nnue_remaining;
    }
    let step1_chosen = *market_wildlife.iter()
        .max_by(|a, b| step1_value[**a as usize].partial_cmp(&step1_value[**b as usize]).unwrap())
        .unwrap();
    let step1_v = step1_value[step1_chosen as usize];

    // Compute step2_value matrix for the post-step1 board
    let mut step2_value = [0.0f32; 5];
    if step1_best_idx[step1_chosen as usize] != usize::MAX {
        let animal1 = step1_chosen;
        if let Some(wa) = working.place_wildlife(step1_best_idx[animal1 as usize], animal1) {
            for animal_idx in 0..5 {
                let animal2 = Wildlife::from_u8(animal_idx as u8).unwrap();
                let variant2 = cards.variant_for(animal2);
                let base_wl2 = cascadia_core::scoring::wildlife::score_wildlife(&working, animal2, variant2) as f32;
                let mut best_delta2 = 0.0f32;
                for &ti in &positions {
                    let idx = ti as usize;
                    if !working.grid.get(idx).can_place_wildlife(animal2) { continue; }
                    if let Some(wa2) = working.place_wildlife(idx, animal2) {
                        let with2 = cascadia_core::scoring::wildlife::score_wildlife(&working, animal2, variant2) as f32;
                        let delta2 = with2 - base_wl2;
                        let keystone2 = if working.grid.get(idx).is_keystone() { 1.0 } else { 0.0 };
                        if delta2 + keystone2 > best_delta2 {
                            best_delta2 = delta2 + keystone2;
                        }
                        working.undo(wa2);
                    }
                }
                step2_value[animal_idx] = best_delta2;
            }
            working.undo(wa);
        }
    }

    // Step 2: pick best of (3 leftover wildlife + 1 fresh draw)
    let leftover_max = market_wildlife.iter()
        .filter(|&&w| w != step1_chosen)
        .map(|&w| step2_value[w as usize])
        .fold(0.0f32, f32::max);

    // Fresh draw: E[max(leftover_max, step2_value[fresh])]
    let bag_counts: [u32; 5] = [
        bag_info.remaining[0] as u32, bag_info.remaining[1] as u32,
        bag_info.remaining[2] as u32, bag_info.remaining[3] as u32,
        bag_info.remaining[4] as u32,
    ];
    let bag_total: u32 = bag_counts.iter().sum();
    let step2_v = if bag_total > 0 {
        let mut ev = 0.0f32;
        for t in 0..5 {
            let p = bag_counts[t] as f32 / bag_total as f32;
            let v = step2_value[t].max(leftover_max);
            ev += p * v;
        }
        ev
    } else {
        leftover_max
    };

    actual + nnue_remaining + step1_v + step2_v
}

/// Market-aware leaf eval: uses the ACTUAL market wildlife (not bag enumeration).
/// At the rollout leaf, the AI's next turn will draft from the current market —
/// 3 leftover from the rollout's last refill + whatever's in market now. We don't
/// need to enumerate fresh draws, just look at what's actually there.
///
/// Computes: actual + NNUE(remaining) + max_over_market_wildlife(wildlife_value).
fn evaluate_leaf_market_aware(
    g: &GameState,
    player: usize,
    net: &NNUENetwork,
) -> f32 {
    let cards = g.scoring_cards;
    let actual = ScoreBreakdown::compute(&mut g.boards[player].clone(), &cards).total as f32;
    let bag_info = crate::nnue::BagInfo::from_game(g);
    let nnue_remaining = net.evaluate_with_bag(&g.boards[player], &bag_info).max(0.0);

    // Compute best wildlife delta per type (in-place mutation on a single clone)
    let mut working = g.boards[player].clone();
    let positions: arrayvec::ArrayVec<u16, 64> = working.placed_tiles.iter().copied().collect();
    let mut wildlife_value = [0.0f32; 5];
    for animal_idx in 0..5 {
        let animal = Wildlife::from_u8(animal_idx as u8).unwrap();
        let variant = cards.variant_for(animal);
        let base_wl = cascadia_core::scoring::wildlife::score_wildlife(&working, animal, variant) as f32;
        let mut best_delta = 0.0f32;
        for &ti in &positions {
            let idx = ti as usize;
            if !working.grid.get(idx).can_place_wildlife(animal) { continue; }
            if let Some(wa) = working.place_wildlife(idx, animal) {
                let with = cascadia_core::scoring::wildlife::score_wildlife(&working, animal, variant) as f32;
                let delta = with - base_wl;
                let keystone = if working.grid.get(idx).is_keystone() { 1.0 } else { 0.0 };
                if delta + keystone > best_delta { best_delta = delta + keystone; }
                working.undo(wa);
            }
        }
        wildlife_value[animal_idx] = best_delta;
    }

    // The AI's next-turn move: pick best wildlife from the CURRENT market
    let mut next_value = 0.0f32;
    for pair in g.market.pairs.iter().flatten() {
        let v = wildlife_value[pair.wildlife as usize];
        if v > next_value { next_value = v; }
    }

    actual + nnue_remaining + next_value
}

/// Fast leaf evaluation that augments NNUE with exact 1-ply wildlife enumeration.
/// Used inside MCE rollouts when MCE_LEAF_EXPECTIMAX is set.
///
/// Computes:
///   actual_score + NNUE(remaining) + E[best_next_market_wildlife_delta]
///
/// The expectation is over the 5^4 = 625 wildlife refill outcomes in the next market,
/// weighted by exact bag-conditioned probabilities. Wildlife "value" per type is the
/// best placement delta on the current board (in-place place/undo, no clones).
///
/// Cost: ~1 board clone + 5 × N_open_slots × wildlife_score + 625 array ops.
/// This corrects the NNUE leaf bias for short-horizon pattern completion.
fn evaluate_leaf_with_next_market(
    g: &GameState,
    player: usize,
    net: &NNUENetwork,
) -> f32 {
    let cards = g.scoring_cards;
    let actual = ScoreBreakdown::compute(&mut g.boards[player].clone(), &cards).total as f32;
    let bag_info = crate::nnue::BagInfo::from_game(g);
    let nnue_remaining = net.evaluate_with_bag(&g.boards[player], &bag_info).max(0.0);

    // Compute best wildlife delta per type using in-place mutation on a single clone
    let mut working = g.boards[player].clone();
    let positions: arrayvec::ArrayVec<u16, 64> = working.placed_tiles.iter().copied().collect();
    let mut wildlife_value = [0.0f32; 5];

    for animal_idx in 0..5 {
        let animal = Wildlife::from_u8(animal_idx as u8).unwrap();
        let variant = cards.variant_for(animal);
        let base_wl = cascadia_core::scoring::wildlife::score_wildlife(&working, animal, variant) as f32;

        let mut best_delta = 0.0f32;
        for &ti in &positions {
            let idx = ti as usize;
            if !working.grid.get(idx).can_place_wildlife(animal) { continue; }
            if let Some(wa) = working.place_wildlife(idx, animal) {
                let with = cascadia_core::scoring::wildlife::score_wildlife(&working, animal, variant) as f32;
                let delta = with - base_wl;
                let keystone = if working.grid.get(idx).is_keystone() { 1.0 } else { 0.0 };
                if delta + keystone > best_delta {
                    best_delta = delta + keystone;
                }
                working.undo(wa);
            }
        }
        wildlife_value[animal_idx] = best_delta;
    }

    // Exact enumeration of next-market wildlife refill (5^4 = 625 cases)
    let bag_counts: [u32; 5] = [
        bag_info.remaining[0] as u32, bag_info.remaining[1] as u32,
        bag_info.remaining[2] as u32, bag_info.remaining[3] as u32,
        bag_info.remaining[4] as u32,
    ];
    let bag_total: u32 = bag_counts.iter().sum();

    let expected_wildlife_bonus = if bag_total >= 4 {
        let mut ev = 0.0f32;
        let mut total_prob = 0.0f32;
        for t0 in 0..5u8 {
            let c0 = bag_counts[t0 as usize];
            if c0 == 0 { continue; }
            let p0 = c0 as f32 / bag_total as f32;
            for t1 in 0..5u8 {
                let c1 = bag_counts[t1 as usize] - if t1 == t0 { 1 } else { 0 };
                if c1 == 0 { continue; }
                let p1 = c1 as f32 / (bag_total - 1) as f32;
                for t2 in 0..5u8 {
                    let c2 = bag_counts[t2 as usize]
                        - if t2 == t0 { 1 } else { 0 }
                        - if t2 == t1 { 1 } else { 0 };
                    if c2 == 0 { continue; }
                    let p2 = c2 as f32 / (bag_total - 2) as f32;
                    for t3 in 0..5u8 {
                        let c3 = bag_counts[t3 as usize]
                            - if t3 == t0 { 1 } else { 0 }
                            - if t3 == t1 { 1 } else { 0 }
                            - if t3 == t2 { 1 } else { 0 };
                        if c3 == 0 { continue; }
                        let p3 = c3 as f32 / (bag_total - 3) as f32;
                        let prob = p0 * p1 * p2 * p3;
                        let best_wl = wildlife_value[t0 as usize]
                            .max(wildlife_value[t1 as usize])
                            .max(wildlife_value[t2 as usize])
                            .max(wildlife_value[t3 as usize]);
                        ev += prob * best_wl;
                        total_prob += prob;
                    }
                }
            }
        }
        if total_prob > 0.0 { ev / total_prob } else { 0.0 }
    } else {
        0.0
    };

    actual + nnue_remaining + expected_wildlife_bonus
}

/// Tier bonus: reward boards that have reached high-scoring wildlife tiers.
/// These tiers are hard to reach in average play but worth 20+ pts each, and
/// the NNUE (trained on average play) underestimates their value. This bonus
/// biases MCE rollouts toward trajectories that commit to max-scoring patterns.
fn tier_bonus(board: &Board) -> f32 {
    let adj = &*ADJACENCY;
    let mut bonus = 0.0f32;

    // ── Salmon: reward long runs ──
    // Scoring: 1→2, 2→4, 3→7, 4→11, 5→15, 6→20, 7→26
    // Bonus for reaching run of 5+, strong bonus for 6-7
    let salmon_positions = &board.wildlife_positions[Wildlife::Salmon as usize];
    if !salmon_positions.is_empty() {
        let mut visited = [false; 441];
        let mut longest_run = 0usize;
        for &pos in salmon_positions.iter() {
            let idx = pos as usize;
            if visited[idx] { continue; }
            let mut component = arrayvec::ArrayVec::<u16, 24>::new();
            let mut queue = arrayvec::ArrayVec::<u16, 24>::new();
            queue.push(pos);
            visited[idx] = true;
            while let Some(current) = queue.pop() {
                component.push(current);
                for nidx in adj.neighbors_of(current as usize) {
                    if !visited[nidx] && board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Salmon) {
                        visited[nidx] = true;
                        queue.push(nidx as u16);
                    }
                }
            }
            // Valid run: no branching (all degrees <= 2)
            let is_valid = component.iter().all(|&p| {
                adj.neighbors_of(p as usize)
                    .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon))
                    .count() <= 2
            });
            if is_valid && component.len() > longest_run {
                longest_run = component.len();
            }
        }
        bonus += match longest_run {
            5 => 1.0,
            6 => 3.0,  // worth 20 pts, strong signal to reach this
            n if n >= 7 => 6.0, // worth 26 pts, maximum commitment
            _ => 0.0,
        };
    }

    // ── Hawks: reward high isolated counts ──
    // Scoring: 1→2, 2→5, 3→8, 4→11, 5→14, 6→18, 7→22, 8→28
    let hawk_positions = &board.wildlife_positions[Wildlife::Hawk as usize];
    let isolated_hawks = hawk_positions.iter()
        .filter(|&&p| !adj.neighbors_of(p as usize)
            .any(|n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Hawk)))
        .count();
    bonus += match isolated_hawks {
        5 => 1.0,
        6 => 2.0,
        7 => 4.0,
        n if n >= 8 => 6.0,  // worth 28 pts
        _ => 0.0,
    };

    // ── Bears: reward 3+ pairs ──
    // Scoring: 1→4, 2→11, 3→19, 4→27
    let bear_positions = &board.wildlife_positions[Wildlife::Bear as usize];
    if !bear_positions.is_empty() {
        let mut visited = [false; 441];
        let mut pairs = 0usize;
        for &pos in bear_positions.iter() {
            let idx = pos as usize;
            if visited[idx] { continue; }
            let mut size = 0u16;
            let mut queue = arrayvec::ArrayVec::<u16, 24>::new();
            queue.push(pos);
            visited[idx] = true;
            while let Some(current) = queue.pop() {
                size += 1;
                for nidx in adj.neighbors_of(current as usize) {
                    if !visited[nidx] && board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Bear) {
                        visited[nidx] = true;
                        queue.push(nidx as u16);
                    }
                }
            }
            if size == 2 { pairs += 1; }
        }
        bonus += match pairs {
            3 => 2.0,
            n if n >= 4 => 5.0,
            _ => 0.0,
        };
    }

    // ── Elk: reward line of 4 (maxes out the type) ──
    let elk_positions = &board.wildlife_positions[Wildlife::Elk as usize];
    if !elk_positions.is_empty() {
        let mut is_elk = [false; 441];
        for &pos in elk_positions.iter() { is_elk[pos as usize] = true; }
        let mut longest = 0usize;
        for &pos in elk_positions.iter() {
            let coord = HexCoord::from_index(pos as usize);
            for &(dq, dr) in &HexCoord::LINE_DIRECTIONS {
                let mut len = 1usize;
                let mut c = HexCoord::new(coord.q + dq, coord.r + dr);
                while let Some(idx) = c.to_index() {
                    if is_elk[idx] { len += 1; c = HexCoord::new(c.q + dq, c.r + dr); }
                    else { break; }
                }
                c = HexCoord::new(coord.q - dq, coord.r - dr);
                while let Some(idx) = c.to_index() {
                    if is_elk[idx] { len += 1; c = HexCoord::new(c.q - dq, c.r - dr); }
                    else { break; }
                }
                if len > longest { longest = len; }
            }
        }
        if longest >= 4 { bonus += 2.0; }
    }

    bonus
}

/// Compute how much each wildlife type is "needed" based on current board patterns.
/// Returns [bear, elk, salmon, hawk, fox] demand scores (0.0 = no need, 5.0+ = urgent).
/// High demand = placing this type would complete or extend a high-value pattern.
fn compute_wildlife_demand(board: &Board) -> [f32; 5] {
    let adj = &*ADJACENCY;
    let mut demand = [0.0f32; 5];

    // Bear demand: isolated bears waiting for a partner
    let bear_positions = &board.wildlife_positions[Wildlife::Bear as usize];
    for &pos in bear_positions.iter() {
        let bear_neighbors: usize = adj.neighbors_of(pos as usize)
            .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Bear))
            .count();
        if bear_neighbors == 0 {
            // Isolated bear — needs a partner. Check if there's an adjacent bear-accepting slot
            let has_slot = adj.neighbors_of(pos as usize).any(|n| {
                let cell = board.grid.get(n);
                cell.is_present() && cell.can_place_wildlife(Wildlife::Bear)
            });
            if has_slot { demand[Wildlife::Bear as usize] += 3.0; }
        }
    }

    // Elk demand: extendable lines that aren't maxed (line < 4)
    let elk_positions = &board.wildlife_positions[Wildlife::Elk as usize];
    let mut is_elk = [false; 441];
    for &pos in elk_positions.iter() { is_elk[pos as usize] = true; }

    for &pos in elk_positions.iter() {
        let coord = HexCoord::from_index(pos as usize);
        for &(dq, dr) in &HexCoord::LINE_DIRECTIONS {
            // Count line length through this elk
            let mut len = 1u16;
            let mut c = HexCoord::new(coord.q + dq, coord.r + dr);
            while let Some(idx) = c.to_index() {
                if is_elk[idx] { len += 1; c = HexCoord::new(c.q + dq, c.r + dr); }
                else { break; }
            }
            c = HexCoord::new(coord.q - dq, coord.r - dr);
            while let Some(idx) = c.to_index() {
                if is_elk[idx] { len += 1; c = HexCoord::new(c.q - dq, c.r - dr); }
                else { break; }
            }

            if len >= 2 && len < 4 {
                // Line of 2-3 that could become 3-4. Very valuable to extend.
                let value = match len {
                    2 => 3.0, // line of 2 → 3 = +4 pts
                    3 => 5.0, // line of 3 → 4 = +4 pts (max!)
                    _ => 0.0,
                };
                demand[Wildlife::Elk as usize] = demand[Wildlife::Elk as usize].max(value);
            }
        }
    }

    // Salmon demand: extendable run endpoints
    let salmon_positions = &board.wildlife_positions[Wildlife::Salmon as usize];
    let mut visited = [false; 441];
    for &pos in salmon_positions.iter() {
        let idx = pos as usize;
        if visited[idx] { continue; }
        // BFS to find run
        let mut component = arrayvec::ArrayVec::<u16, 24>::new();
        let mut queue = arrayvec::ArrayVec::<u16, 24>::new();
        queue.push(pos);
        visited[idx] = true;
        while let Some(current) = queue.pop() {
            component.push(current);
            for nidx in adj.neighbors_of(current as usize) {
                if !visited[nidx] && board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Salmon) {
                    visited[nidx] = true;
                    queue.push(nidx as u16);
                }
            }
        }
        let run_len = component.len();
        if run_len >= 2 && run_len < 7 {
            // Extendable run — check if any endpoint has an adjacent salmon-accepting slot
            let has_extension = component.iter().any(|&p| {
                let salmon_neighbors = adj.neighbors_of(p as usize)
                    .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon))
                    .count();
                if salmon_neighbors > 1 { return false; } // not an endpoint
                adj.neighbors_of(p as usize).any(|n| {
                    let cell = board.grid.get(n);
                    cell.is_present() && cell.can_place_wildlife(Wildlife::Salmon)
                        && cell.placed_wildlife() != Some(Wildlife::Salmon)
                })
            });
            if has_extension {
                let value = match run_len {
                    2 => 2.0,
                    3 => 3.0,
                    4 => 4.0, // run of 4→5 = +4 pts
                    5 => 5.0, // run of 5→6 = +5 pts
                    6 => 6.0, // run of 6→7 = +6 pts
                    _ => 1.0,
                };
                demand[Wildlife::Salmon as usize] = demand[Wildlife::Salmon as usize].max(value);
            }
        }
    }

    // Hawk demand: always valuable if there are safe isolation spots
    let hawk_positions = &board.wildlife_positions[Wildlife::Hawk as usize];
    let isolated_count = hawk_positions.iter()
        .filter(|&&pos| {
            !adj.neighbors_of(pos as usize)
                .any(|n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Hawk))
        })
        .count();
    // Marginal value of next isolated hawk
    let hawk_value = match isolated_count {
        0..=2 => 3.0,
        3..=4 => 3.5,
        5..=6 => 4.0,
        _ => 2.0,
    };
    demand[Wildlife::Hawk as usize] = hawk_value;

    // Fox demand: low — already scores well via greedy
    demand[Wildlife::Fox as usize] = 1.0;

    demand
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enumerate_draws_probabilities_sum_to_one() {
        // Bag with 3 of each type = 15 total, draw 4
        let bag_counts = [3u32, 3, 3, 3, 3];
        let bag_total = 15;
        let active_slots = vec![0, 1, 2, 3];
        let score_matrix = [[1.0; 5]; 4]; // all equal scores

        let mut mulligan_ev = 0.0;
        let mut pinecone_ev = 0.0;
        let mut total_prob = 0.0;
        let mut draw = [0u8; 4];

        enumerate_draws(
            &active_slots, &bag_counts, bag_total, &score_matrix,
            &mut mulligan_ev, &mut pinecone_ev, &mut total_prob,
            &mut draw, 0, 1.0,
        );

        assert!((total_prob - 1.0).abs() < 1e-10,
            "Probabilities should sum to 1.0, got {}", total_prob);
    }

    #[test]
    fn test_enumerate_draws_with_known_outcome() {
        // Bag with ONLY bears (5 bears, 0 of everything else)
        // Drawing 2 from 2 active slots → guaranteed (bear, bear)
        let bag_counts = [5u32, 0, 0, 0, 0];
        let bag_total = 5;
        let active_slots = vec![0, 1];

        // Score matrix: slot 0 with bear = 90, slot 1 with bear = 85
        let mut score_matrix = [[0.0; 5]; 4];
        score_matrix[0][0] = 90.0; // slot 0 + bear
        score_matrix[1][0] = 85.0; // slot 1 + bear

        let mut mulligan_ev = 0.0;
        let mut pinecone_ev = 0.0;
        let mut total_prob = 0.0;
        let mut draw = [0u8; 4];

        enumerate_draws(
            &active_slots, &bag_counts, bag_total, &score_matrix,
            &mut mulligan_ev, &mut pinecone_ev, &mut total_prob,
            &mut draw, 0, 1.0,
        );

        let ev = mulligan_ev / total_prob;
        assert!((ev - 90.0).abs() < 1e-10,
            "With only bears, best paired should be 90.0, got {}", ev);
    }

    #[test]
    fn test_enumerate_draws_pinecone_can_beat_paired() {
        // Scenario: 2 active slots, bag has only bears and salmon
        let bag_counts = [5u32, 0, 5, 0, 0]; // 5 bear, 5 salmon
        let bag_total = 10;
        let active_slots = vec![0, 1];

        // Slot 0 is great with salmon (95), bad with bear (80)
        // Slot 1 is bad with both (70)
        let mut score_matrix = [[70.0; 5]; 4];
        score_matrix[0][0] = 80.0; // slot 0 + bear
        score_matrix[0][2] = 95.0; // slot 0 + salmon

        let mut mulligan_ev = 0.0;
        let mut pinecone_ev = 0.0;
        let mut total_prob = 0.0;
        let mut draw = [0u8; 4];

        enumerate_draws(
            &active_slots, &bag_counts, bag_total, &score_matrix,
            &mut mulligan_ev, &mut pinecone_ev, &mut total_prob,
            &mut draw, 0, 1.0,
        );

        let paired_ev = mulligan_ev / total_prob;
        let pine_ev = pinecone_ev / total_prob;

        // Pinecone EV should be higher because when draw is (bear, salmon),
        // pinecone can pair slot 0 with salmon from slot 1's draw (95 - 1 = 94)
        // vs paired which only gets slot 0 + bear = 80
        assert!(pine_ev > paired_ev,
            "Pinecone EV ({}) should beat paired EV ({})", pine_ev, paired_ev);
    }

    #[test]
    fn test_enumerate_impossible_draws_skipped() {
        // Bag with 0 of type 0, should never draw it
        let bag_counts = [0u32, 5, 5, 5, 5];
        let bag_total = 20;
        let active_slots = vec![0];

        let mut score_matrix = [[50.0; 5]; 4];
        score_matrix[0][0] = 100.0; // slot 0 + bear (impossible to draw)

        let mut mulligan_ev = 0.0;
        let mut pinecone_ev = 0.0;
        let mut total_prob = 0.0;
        let mut draw = [0u8; 4];

        enumerate_draws(
            &active_slots, &bag_counts, bag_total, &score_matrix,
            &mut mulligan_ev, &mut pinecone_ev, &mut total_prob,
            &mut draw, 0, 1.0,
        );

        let ev = mulligan_ev / total_prob;
        assert!((ev - 50.0).abs() < 1e-10,
            "Bear is impossible to draw, EV should be 50.0, got {}", ev);
    }
}
