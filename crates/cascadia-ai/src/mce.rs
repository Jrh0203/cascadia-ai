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

/// Per-candidate MCE rollout statistics.
#[derive(Clone, Debug)]
pub struct MceStat {
    pub mean: f64,
    pub std: f64,
    pub min: u64,
    pub max: u64,
    pub median: f64,
}

/// Rollout-budget allocation strategy for greedy MCE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GreedyMceAlloc {
    /// Equal rollouts per candidate. Baseline.
    Uniform,
    /// Sequential halving (Karnin 2013): log2(n) rounds, eliminate bottom half each.
    /// Provably optimal for fixed-budget best-arm identification.
    SeqHalving,
    /// UCB1 adaptive allocation. Spend more rollouts on close candidates.
    Ucb,
    /// Common Random Numbers (CRN): all candidates share the same rollout seeds.
    /// Reduces variance of pairwise comparisons — ideal for argmax over candidates.
    UniformCRN,
    /// Sequential halving + CRN (shared seeds within each round).
    SeqHalvingCRN,
    /// Sequential halving with early termination: stop if the top candidate's
    /// mean is > 2σ above the 2nd-best after any round. Saves compute on easy
    /// decisions, can optionally redirect saved budget elsewhere.
    SeqHalvingEarlyTerm,
    /// Confidence-interval-aware halving: instead of halving the alive set by
    /// rank, eliminate any candidate whose UCB (mean + z*stderr) falls below
    /// the leader's LCB (mean - z*stderr). Adaptive — tight matches keep more
    /// candidates alive; clear winners eliminate aggressively.
    /// Z tunable via MCE_HALVING_CI_Z (default 1.5).
    /// MCE_HALVING_CI_FLOOR=1 → also enforce hard top-half-by-mean as floor.
    SeqHalvingCI,
    /// Heteroscedastic-variance-weighted halving (OCBA-inspired).
    /// Within each round, allocate budget proportional to variance/gap-squared
    /// so high-variance candidates near the leader get more samples.
    /// Eliminates by hard halving (top-half-by-mean) each round.
    /// Asymptotically optimal under heterogeneous variance (Audibert+ 2010).
    SeqHalvingHetero,
    /// Successive Rejects (Audibert & Bubeck 2010). K-1 phases; each phase
    /// eliminates the WORST remaining arm. Budget per phase grows to give
    /// more rollouts to harder-to-distinguish pairs late in the tournament.
    /// Provably near-optimal for best-arm identification within factor log(K).
    SuccessiveRejects,
    /// Sequential halving with progressive widening. Starts with a narrow
    /// front (top-K_initial by prefilter), runs partial rollouts, adds
    /// more candidates from the reserve if top candidates are statistically
    /// tied. Adaptive candidate-set expansion.
    SeqHalvingPW,
    /// Thompson Sampling: model each candidate as Normal(mean, var/n).
    /// Each round, sample from posteriors, allocate a batch to the
    /// highest-sampled candidate. Naturally balances exploration (high
    /// uncertainty) vs exploitation (high mean). Works well when candidates
    /// have heterogeneous variance.
    ThompsonSampling,
    /// MCTS-style progressive widening + UCB1.
    /// Start with K_init=4 candidates (top by NNUE prior). Each sim:
    ///   1. If ceil(k·N^α) > expanded_count, add next NNUE-ranked candidate.
    ///   2. Select candidate by UCB1: argmax(mean + c·sqrt(ln(N)/n_i)).
    ///      Normalized-reward form: argmax(mean/100 + c·sqrt(ln(N)/n_i)).
    ///   3. Run 1 rollout on selected, backprop.
    /// Final: argmax by mean (not visit count — visits heavily biased to top).
    /// Addresses halving's weakness: bulk elimination can drop MCE-best moves
    /// that sit at low NNUE rank. PW+UCB keeps discovering candidates while
    /// focusing on promising ones.
    MctsPW,
    /// PUCT (AlphaZero-style): UCB1 with NNUE-prior weighting in the exploration term.
    ///   PUCT(a) = Q(a) + c · P(a) · √N / (1 + n_a)
    /// Where P(a) = softmax over NNUE priors (scaled by `MCE_PUCT_TAU`).
    /// Differs from MctsPW by using prior probability (not visit-rank) for exploration
    /// weighting. Lit: AlphaGo Zero (Silver 2017), AlphaZero (Silver 2018).
    /// Tunables: MCE_PUCT_C (default 2.0), MCE_PUCT_TAU (default 8.0 — softmax temp).
    /// Single-shot batched: explore via PUCT, run small batch per selection.
    Puct,
}

/// Expanded candidate set generator for greedy MCE (used with `--candidates expanded`).
/// In addition to candidate_moves_pub + wildlife_strategic + greedy, enumerates all
/// rotations for each tile placement. Returns up to ~40-50 candidates per turn.
pub fn expanded_candidates(game: &GameState) -> Vec<ScoredMove> {
    use cascadia_core::hex::HexCoord;

    let player = game.current_player;
    let mp: Vec<_> = game.market.available()
        .map(|(i, p)| (i, p.tile, p.wildlife)).collect();
    if mp.is_empty() { return Vec::new(); }

    let board = &game.boards[player];
    let frontier = board.frontier();

    let mut candidates: Vec<ScoredMove> = candidate_moves_pub(game).into_iter()
        .chain(wildlife_strategic_candidates(game).into_iter())
        .collect();

    // For each market slot, also try placing the tile at each frontier cell with each rotation
    // without the potential-filter. This captures candidates the potential-heuristic might miss.
    let has_tokens = board.nature_tokens > 0;
    let combos: Vec<(usize, cascadia_core::types::TileData, cascadia_core::types::Wildlife, Option<usize>)> = {
        let mut c: Vec<_> = mp.iter().map(|&(i, t, w)| (i, t, w, None)).collect();
        if has_tokens {
            for &(ti, tile, _) in &mp {
                for &(wi, _, wl) in &mp {
                    if ti != wi { c.push((ti, tile, wl, Some(wi))); }
                }
            }
        }
        c
    };

    // Cap expanded candidates to a reasonable number — the goal is to cover
    // moves the potential-filter might miss, not to enumerate exhaustively.
    // For each (tile_draft, wildlife_draft) combo: try top-8 frontier positions
    // (by adjacency density). Only 1 rotation per tile to keep count manageable.
    let max_extra: usize = std::env::var("MCE_MAX_EXTRA_CANDS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(20);
    let mut extras: Vec<ScoredMove> = Vec::new();

    // Rank frontier cells by number of placed neighbors (more neighbors = more pattern potential)
    let adj = &*cascadia_core::hex::ADJACENCY;
    let mut frontier_ranked: Vec<(u16, usize)> = frontier.iter().map(|&idx| {
        let n_placed = adj.neighbors_of(idx as usize)
            .filter(|&ni| board.grid.get(ni).is_present())
            .count();
        (idx, n_placed)
    }).collect();
    frontier_ranked.sort_by(|a, b| b.1.cmp(&a.1));
    let frontier_limit: usize = if max_extra > 20 { 16 } else { 8 };
    let top_frontier: Vec<u16> = frontier_ranked.into_iter().take(frontier_limit).map(|(i, _)| i).collect();

    for (tile_mi, tile, drafted_wl, wl_mi) in combos {
        for &tile_idx in &top_frontier {
            let coord = HexCoord::from_index(tile_idx as usize);
            // Wildlife placement: on the new tile if its mask allows, OR on any
            // compatible open slot. Limit to 4 wildlife positions.
            let mut wl_positions: Vec<(i8, i8)> = Vec::new();
            if tile.allowed.contains(drafted_wl) {
                wl_positions.push((coord.q, coord.r));
            }
            for &wp in board.placed_tiles.iter().take(3) {
                let wc = HexCoord::from_index(wp as usize);
                if let Some(wl_idx) = cascadia_core::hex::HexCoord::new(wc.q, wc.r).to_index() {
                    let cell = board.grid.get(wl_idx);
                    if cell.is_present() && !cell.has_wildlife()
                        && cell.allowed_wildlife().contains(drafted_wl) {
                        wl_positions.push((wc.q, wc.r));
                    }
                }
            }
            for (wq, wr) in wl_positions.into_iter().take(4) {
                extras.push(ScoredMove {
                    score: 0,
                    eval: 0,
                    market_index: tile_mi,
                    tile_q: coord.q, tile_r: coord.r,
                    rotation: 0,
                    wildlife_q: Some(wq), wildlife_r: Some(wr),
                    wildlife_market_index: wl_mi,
                });
                if extras.len() >= max_extra { break; }
            }
            if extras.len() >= max_extra { break; }
        }
        if extras.len() >= max_extra { break; }
    }
    candidates.extend(extras);

    // Dedup
    candidates.sort_by_key(|c| (c.market_index, c.wildlife_market_index, c.tile_q, c.tile_r,
                                c.rotation, c.wildlife_q, c.wildlife_r));
    candidates.dedup_by_key(|c| (c.market_index, c.wildlife_market_index, c.tile_q, c.tile_r,
                                 c.rotation, c.wildlife_q, c.wildlife_r));
    candidates
}

/// Run one greedy rollout from the given game state. Returns the final player-0 score.
fn run_greedy_rollout(mut gs: GameState, player: usize, seed: u64, candidate: &ScoredMove) -> u64 {
    let mut rr = StdRng::seed_from_u64(seed);
    gs.shuffle_bags(&mut rr);
    if !execute_scored_move(&mut gs, candidate) { return 0; }
    while !gs.is_game_over() {
        if gs.current_player != player {
            if gs.can_replace_overflow().is_some() {
                gs.replace_overflow();
            }
        } else {
            if gs.can_replace_overflow().is_some() {
                gs.replace_overflow();
            }
        }
        match greedy_move(&gs) {
            Some(mv) => { if !execute_scored_move(&mut gs, &mv) { break; } }
            None => break,
        }
    }
    ScoreBreakdown::compute(
        &mut gs.boards[player], &gs.scoring_cards,
    ).total as u64
}

/// Run a rollout where player 0 uses NNUE for move selection (stronger policy)
/// and opponents continue to play greedy.
/// Motivation (AlphaGo-style): smarter rollout policy → lower variance → more
/// informative comparisons at same budget. Typical NNUE-greedy gap is ~10 pts.
/// Softmax-sampled greedy move: evaluate each market pair independently, then
/// sample proportionally to exp(score / temperature). Lower temperature → more
/// greedy; higher → more random. Used for opponent modeling in rollouts.
fn softmax_greedy_move(game: &GameState, temperature: f32, rng: &mut StdRng) -> Option<ScoredMove> {
    let cards = game.scoring_cards;
    let turns = game.turns_remaining;
    let board = &game.boards[game.current_player];
    let market_pairs: Vec<_> = game.market.available()
        .map(|(i, p)| (i, p.tile, p.wildlife)).collect();
    if market_pairs.is_empty() { return None; }

    let mut options: Vec<(ScoredMove, f64)> = Vec::with_capacity(4);
    for &(mi, tile, wildlife) in &market_pairs {
        let mp_single = vec![(mi, tile, wildlife)];
        let mut b = board.clone();
        if let Some(mv) = best_move_with_potential(&mut b, &mp_single, &cards, turns) {
            options.push((mv, mv.score as f64));
        }
    }
    if options.is_empty() { return None; }
    if options.len() == 1 { return Some(options[0].0); }

    let max_s = options.iter().map(|(_, s)| *s).fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = options.iter()
        .map(|(_, s)| ((s - max_s) / temperature as f64).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    if total <= 0.0 { return Some(options[0].0); }

    let mut r = rng.gen_range(0.0..total);
    for (i, &w) in weights.iter().enumerate() {
        if r < w { return Some(options[i].0); }
        r -= w;
    }
    Some(options.last().unwrap().0)
}

/// Returns (final_score, control_variate). The CV is an NNUE eval taken at a
/// fixed mid-rollout point (after `MCE_CV_AT_TURN` of player 0's plies, default 2).
/// Correlated with final_score; usable for variance reduction at decision time.
/// If MCE_CV_AT_TURN <= 0 or game ends before that turn, CV is set to final_score.
fn run_nnue_rollout(
    mut gs: GameState, player: usize, seed: u64, candidate: &ScoredMove,
    net: &NNUENetwork,
) -> (u64, u64) {
    // MCE_OPP_TEMPERATURE: opponent softmax temperature in rollouts.
    // 0 or unset = greedy (default). >0 = softmax sampling over market options.
    // Pluribus insight: don't assume opponents play argmax.
    let opp_temp: f32 = std::env::var("MCE_OPP_TEMPERATURE").ok()
        .and_then(|s| s.parse().ok())
        .filter(|&t| t > 0.0)
        .unwrap_or(0.0);
    // MCE_ROLLOUT_OPP: opponent policy inside NNUE rollouts.
    //   "greedy" (default) = legacy behaviour, fast, biased.
    //   "nnue" = opponents use the same NNUE as player 0. Self-consistent —
    //   rollout values match the HH reality where all players search.
    //   Cost: ~2-4x slower per rollout (extra NNUE forward passes).
    let rollout_opp_nnue: bool = std::env::var("MCE_ROLLOUT_OPP").ok()
        .map(|s| s.eq_ignore_ascii_case("nnue") || s == "1")
        .unwrap_or(false);
    // MCE_ROLLOUT_POLICY: player 0's policy inside NNUE rollouts.
    //   "nnue" (default) = 1-ply NNUE argmax (pick_best_move_nnue).
    //   "expectimax1" = 1-ply expectimax (enumerate next-market wildlife draws,
    //                                     argmax over decomposed candidates).
    // Rollout quality compounds: 22% top-1 (nnue) → ~40%+ top-1 (expectimax1),
    // giving ~+2 pts standalone per memory (92.4 vs 90.5).  Costs ~3-4× per ply.
    let rollout_policy_expectimax: bool = std::env::var("MCE_ROLLOUT_POLICY").ok()
        .map(|s| s.eq_ignore_ascii_case("expectimax1")
                 || s.eq_ignore_ascii_case("expectimax")
                 || s.eq_ignore_ascii_case("exmx1"))
        .unwrap_or(false);

    // MCE_CV_AT_TURN: take control-variate NNUE eval after this many of player 0's
    // plies. Default 2 — gives enough bag-shuffle variation to be informative.
    // Set to 0 (or unset MCE_CONTROL_VARIATES) to skip the eval entirely (free path).
    let cv_at_turn: usize = std::env::var("MCE_CV_AT_TURN").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(2);
    // Only collect the mid-rollout CV eval if the caller actually plans to use it.
    // Otherwise the extra NNUE forward + ScoreBreakdown::compute per rollout is pure
    // wasted compute and slows down the rollout pipeline by ~25-40% (M1 measurements).
    let cv_collect_enabled: bool = std::env::var("MCE_CONTROL_VARIATES").ok()
        .map(|s| !s.is_empty() && s != "0").unwrap_or(false);

    let mut rr = StdRng::seed_from_u64(seed);
    gs.shuffle_bags(&mut rr);
    if !execute_scored_move(&mut gs, candidate) { return (0, 0); }
    let mut player_turns_done: usize = 1; // candidate move counts as player turn 1
    let mut cv_eval: Option<u64> = None;
    while !gs.is_game_over() {
        if gs.current_player != player {
            if gs.can_replace_overflow().is_some() {
                gs.replace_overflow();
            }
            let opp_mv = if rollout_opp_nnue {
                crate::nnue_train::pick_best_move_nnue(&gs, net)
                    .or_else(|| greedy_move(&gs))
            } else if opp_temp > 0.0 {
                softmax_greedy_move(&gs, opp_temp, &mut rr)
            } else {
                greedy_move(&gs)
            };
            match opp_mv {
                Some(mv) => { if !execute_scored_move(&mut gs, &mv) { break; } }
                None => break,
            }
        } else {
            if gs.can_replace_overflow().is_some() {
                gs.replace_overflow();
            }
            // Take control-variate NNUE eval at the start of player 0's `cv_at_turn`-th turn.
            if cv_collect_enabled && cv_eval.is_none() && cv_at_turn > 0 && player_turns_done >= cv_at_turn {
                let bag = crate::nnue::BagInfo::from_game_for_player(&gs, player);
                let cur = ScoreBreakdown::compute(&mut gs.boards[player].clone(), &gs.scoring_cards).total as f64;
                let remaining = net.evaluate_with_bag(&gs.boards[player], &bag) as f64;
                cv_eval = Some(((cur + remaining).max(0.0)) as u64);
            }
            let pm = if rollout_policy_expectimax {
                best_move_expectimax_1ply(&gs, net)
            } else {
                crate::nnue_train::pick_best_move_nnue(&gs, net)
            };
            match pm.or_else(|| greedy_move(&gs)) {
                Some(mv) => { if !execute_scored_move(&mut gs, &mv) { break; } }
                None => break,
            }
            player_turns_done += 1;
        }
    }
    let final_score = ScoreBreakdown::compute(
        &mut gs.boards[player], &gs.scoring_cards,
    ).total as u64;
    let cv = cv_eval.unwrap_or(final_score);
    (final_score, cv)
}

/// Pure-greedy Monte Carlo Evaluation: no value network at all.
/// For each candidate, runs rollouts, averages final scores, returns argmax.
/// `alloc` controls budget distribution across candidates.
///
/// Rollouts use pure greedy for all players (including opponents with free-replace).
pub fn best_move_greedy_mce_v2(
    game: &GameState,
    num_rollouts: usize,
    alloc: GreedyMceAlloc,
    candidates: Vec<ScoredMove>,
    rng: &mut StdRng,
) -> Option<ScoredMove> {
    let player = game.current_player;
    if candidates.is_empty() { return None; }

    let n_cands = candidates.len();
    let mut totals = vec![0u64; n_cands];
    let mut sumsq = vec![0u64; n_cands];
    let mut counts = vec![0u32; n_cands];

    let game_arc = std::sync::Arc::new(game.clone());
    let cands_arc = std::sync::Arc::new(candidates.clone());

    let run_work = |work_items: Vec<(usize, u64)>,
                    totals: &mut Vec<u64>, sumsq: &mut Vec<u64>, counts: &mut Vec<u32>| {
        let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let chunk_size = ((work_items.len() + num_threads - 1) / num_threads).max(1);
        let handles: Vec<_> = work_items.chunks(chunk_size).map(|chunk| {
            let work = chunk.to_vec();
            let g = std::sync::Arc::clone(&game_arc);
            let c = std::sync::Arc::clone(&cands_arc);
            std::thread::spawn(move || {
                let mut results: Vec<(usize, u64)> = Vec::with_capacity(work.len());
                for &(ci, seed) in &work {
                    let score = run_greedy_rollout((*g).clone(), player, seed, &c[ci]);
                    results.push((ci, score));
                }
                results
            })
        }).collect();
        for h in handles {
            for (ci, score) in h.join().unwrap() {
                totals[ci] += score;
                sumsq[ci] += score * score;
                counts[ci] += 1;
            }
        }
    };

    match alloc {
        GreedyMceAlloc::Uniform => {
            let per = num_rollouts / n_cands.max(1);
            let mut work = Vec::with_capacity(per * n_cands);
            for ci in 0..n_cands {
                for _ in 0..per { work.push((ci, rng.gen())); }
            }
            run_work(work, &mut totals, &mut sumsq, &mut counts);
        }
        GreedyMceAlloc::SeqHalving => {
            let num_rounds = (n_cands as f64).log2().ceil().max(1.0) as usize;
            let mut alive: Vec<usize> = (0..n_cands).collect();
            let budget_per_round = (num_rollouts / num_rounds).max(n_cands);
            for round in 0..num_rounds {
                if alive.is_empty() { break; }
                let per = (budget_per_round / alive.len()).max(1);
                let mut work = Vec::with_capacity(per * alive.len());
                for &ci in &alive {
                    for _ in 0..per { work.push((ci, rng.gen())); }
                }
                run_work(work, &mut totals, &mut sumsq, &mut counts);
                if round < num_rounds - 1 {
                    let mut scored: Vec<(usize, f64)> = alive.iter()
                        .filter_map(|&ci| {
                            if counts[ci] == 0 { return None; }
                            Some((ci, totals[ci] as f64 / counts[ci] as f64))
                        }).collect();
                    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let keep = (scored.len() + 1) / 2;
                    alive = scored.into_iter().take(keep).map(|(i, _)| i).collect();
                }
            }
        }
        GreedyMceAlloc::UniformCRN => {
            let per = num_rollouts / n_cands.max(1);
            // CRN: generate `per` seeds, reuse across all candidates.
            let base_seeds: Vec<u64> = (0..per).map(|_| rng.gen()).collect();
            let mut work = Vec::with_capacity(per * n_cands);
            for ci in 0..n_cands {
                for &seed in &base_seeds { work.push((ci, seed)); }
            }
            run_work(work, &mut totals, &mut sumsq, &mut counts);
        }
        GreedyMceAlloc::SeqHalvingCRN => {
            let num_rounds = (n_cands as f64).log2().ceil().max(1.0) as usize;
            let mut alive: Vec<usize> = (0..n_cands).collect();
            let budget_per_round = (num_rollouts / num_rounds).max(n_cands);
            for round in 0..num_rounds {
                if alive.is_empty() { break; }
                let per = (budget_per_round / alive.len()).max(1);
                // Shared seeds across all alive candidates in this round
                let base_seeds: Vec<u64> = (0..per).map(|_| rng.gen()).collect();
                let mut work = Vec::with_capacity(per * alive.len());
                for &ci in &alive {
                    for &seed in &base_seeds { work.push((ci, seed)); }
                }
                run_work(work, &mut totals, &mut sumsq, &mut counts);
                if round < num_rounds - 1 {
                    let mut scored: Vec<(usize, f64)> = alive.iter()
                        .filter_map(|&ci| {
                            if counts[ci] == 0 { return None; }
                            Some((ci, totals[ci] as f64 / counts[ci] as f64))
                        }).collect();
                    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let keep = (scored.len() + 1) / 2;
                    alive = scored.into_iter().take(keep).map(|(i, _)| i).collect();
                }
            }
        }
        GreedyMceAlloc::SeqHalvingEarlyTerm => {
            let num_rounds = (n_cands as f64).log2().ceil().max(1.0) as usize;
            let mut alive: Vec<usize> = (0..n_cands).collect();
            let budget_per_round = (num_rollouts / num_rounds).max(n_cands);
            for round in 0..num_rounds {
                if alive.is_empty() { break; }
                let per = (budget_per_round / alive.len()).max(1);
                let mut work = Vec::with_capacity(per * alive.len());
                for &ci in &alive {
                    for _ in 0..per { work.push((ci, rng.gen())); }
                }
                run_work(work, &mut totals, &mut sumsq, &mut counts);
                // Early-term check: stop if top > 2.5σ above second
                if alive.len() >= 2 {
                    let mut scored: Vec<(usize, f64, f64)> = alive.iter()
                        .filter_map(|&ci| {
                            if counts[ci] == 0 { return None; }
                            let n = counts[ci] as f64;
                            let mean = totals[ci] as f64 / n;
                            let var = (sumsq[ci] as f64 / n - mean * mean).max(0.0);
                            let stderr = (var / n).sqrt();
                            Some((ci, mean, stderr))
                        }).collect();
                    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    if scored.len() >= 2 {
                        let (_, best_mean, best_se) = scored[0];
                        let (_, second_mean, second_se) = scored[1];
                        let gap = best_mean - second_mean;
                        let combined_se = (best_se.powi(2) + second_se.powi(2)).sqrt();
                        if combined_se > 0.0 && gap > 2.5 * combined_se {
                            break;  // early termination
                        }
                    }
                }
                if round < num_rounds - 1 {
                    let mut scored: Vec<(usize, f64)> = alive.iter()
                        .filter_map(|&ci| {
                            if counts[ci] == 0 { return None; }
                            Some((ci, totals[ci] as f64 / counts[ci] as f64))
                        }).collect();
                    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let keep = (scored.len() + 1) / 2;
                    alive = scored.into_iter().take(keep).map(|(i, _)| i).collect();
                }
            }
        }
        GreedyMceAlloc::Ucb => {
            // Initial small budget per candidate, then UCB1-tuned allocation
            let init_per = ((num_rollouts / 10) / n_cands.max(1)).max(2);
            let mut work = Vec::with_capacity(init_per * n_cands);
            for ci in 0..n_cands {
                for _ in 0..init_per { work.push((ci, rng.gen())); }
            }
            run_work(work, &mut totals, &mut sumsq, &mut counts);
            // Remaining budget via UCB: batched in chunks for parallelism
            let mut remaining = num_rollouts.saturating_sub(init_per * n_cands);
            let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
            while remaining > 0 {
                let batch = remaining.min(num_threads * 4);
                // Pick top-batch candidates by UCB score
                let total_n: u32 = counts.iter().sum();
                let log_total = (total_n.max(1) as f64).ln();
                let mut ucb_work = Vec::with_capacity(batch);
                for _ in 0..batch {
                    // Compute UCB for each candidate, pick argmax
                    let mut best = 0usize;
                    let mut best_ucb = f64::NEG_INFINITY;
                    for ci in 0..n_cands {
                        let n = counts[ci].max(1) as f64;
                        let mean = totals[ci] as f64 / n;
                        // Normalize mean to [0,1] by dividing by 100 (max Cascadia score ~100)
                        let explore = (2.0 * log_total / n).sqrt();
                        // Cascadia-tuned UCB: scale exploration to ~10% of typical score (≈10)
                        let ucb = mean / 100.0 + 0.1 * explore;
                        if ucb > best_ucb { best_ucb = ucb; best = ci; }
                    }
                    ucb_work.push((best, rng.gen()));
                    // Pre-increment counts so next iteration picks differently (optimistic)
                    counts[best] += 1;
                }
                // Undo the optimistic increments — real results will re-increment
                for &(ci, _) in &ucb_work { counts[ci] -= 1; }
                run_work(ucb_work, &mut totals, &mut sumsq, &mut counts);
                remaining = remaining.saturating_sub(batch);
            }
        }
        GreedyMceAlloc::SeqHalvingCI => {
            // Confidence-interval-aware halving (greedy variant).
            // Z tunable via MCE_HALVING_CI_Z (default 1.5).
            // MCE_HALVING_CI_FLOOR=1 → also apply hard halving as a floor:
            //   alive_next = intersect(CI_kept, top-half-by-mean).
            let z: f64 = std::env::var("MCE_HALVING_CI_Z").ok()
                .and_then(|s| s.parse().ok()).unwrap_or(1.5);
            let hard_floor: bool = std::env::var("MCE_HALVING_CI_FLOOR").ok()
                .map(|s| !s.is_empty() && s != "0").unwrap_or(false);
            let num_rounds = (n_cands as f64).log2().ceil().max(1.0) as usize;
            let mut alive: Vec<usize> = (0..n_cands).collect();
            let budget_per_round = (num_rollouts / num_rounds).max(n_cands);
            for round in 0..num_rounds {
                if alive.is_empty() { break; }
                let per = (budget_per_round / alive.len()).max(1);
                let mut work = Vec::with_capacity(per * alive.len());
                for &ci in &alive {
                    for _ in 0..per { work.push((ci, rng.gen())); }
                }
                run_work(work, &mut totals, &mut sumsq, &mut counts);
                if round < num_rounds - 1 && alive.len() >= 2 {
                    let stats: Vec<(usize, f64, f64)> = alive.iter().filter_map(|&ci| {
                        if counts[ci] == 0 { return None; }
                        let n = counts[ci] as f64;
                        let mean = totals[ci] as f64 / n;
                        let var = (sumsq[ci] as f64 / n - mean * mean).max(0.0);
                        let stderr = (var / n).sqrt();
                        Some((ci, mean, stderr))
                    }).collect();
                    if stats.is_empty() { break; }
                    let leader_lcb = stats.iter().fold(f64::NEG_INFINITY, |acc, &(_, m, se)| {
                        let lcb = m - z * se;
                        if lcb > acc { lcb } else { acc }
                    });
                    let mut kept: Vec<usize> = stats.iter().filter_map(|&(ci, m, se)| {
                        if m + z * se >= leader_lcb { Some(ci) } else { None }
                    }).collect();
                    if hard_floor {
                        let mut by_mean: Vec<(usize, f64)> = stats.iter()
                            .map(|&(ci, m, _)| (ci, m)).collect();
                        by_mean.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                        let max_keep = ((alive.len() + 1) / 2).max(2);
                        let top: std::collections::HashSet<usize> = by_mean.into_iter()
                            .take(max_keep).map(|(ci, _)| ci).collect();
                        kept.retain(|ci| top.contains(ci));
                    }
                    if kept.len() < 2 {
                        let mut by_mean: Vec<(usize, f64)> = stats.iter()
                            .map(|&(ci, m, _)| (ci, m)).collect();
                        by_mean.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                        kept = by_mean.into_iter().take(2.min(stats.len())).map(|(ci, _)| ci).collect();
                    }
                    alive = kept;
                }
            }
        }
        GreedyMceAlloc::SeqHalvingHetero => {
            // Heteroscedastic allocation (OCBA-inspired): within each halving
            // round, give MORE rollouts to high-variance candidates close to
            // the leader, FEWER to low-variance or far-from-leader candidates.
            // Asymptotically optimal under heterogeneous variance (Audibert+ 2010).
            // Eliminate by hard halving each round (so it's seq-halving-shaped).
            // Round 0: uniform allocation (no variance estimates yet).
            let num_rounds = (n_cands as f64).log2().ceil().max(1.0) as usize;
            let mut alive: Vec<usize> = (0..n_cands).collect();
            let budget_per_round = (num_rollouts / num_rounds).max(n_cands);
            for round in 0..num_rounds {
                if alive.is_empty() { break; }
                // Compute per-candidate target rollouts for this round
                let total_round = budget_per_round.max(alive.len());
                let pers: Vec<usize> = if round == 0 {
                    // Uniform first round to seed variance estimates
                    let per = (total_round / alive.len()).max(1);
                    alive.iter().map(|_| per).collect()
                } else {
                    // Compute (mean, var) for alive candidates
                    let stats: Vec<(usize, f64, f64)> = alive.iter().filter_map(|&ci| {
                        if counts[ci] == 0 { return None; }
                        let n = counts[ci] as f64;
                        let mean = totals[ci] as f64 / n;
                        let var = (sumsq[ci] as f64 / n - mean * mean).max(1.0);
                        Some((ci, mean, var))
                    }).collect();
                    if stats.is_empty() {
                        let per = (total_round / alive.len()).max(1);
                        alive.iter().map(|_| per).collect()
                    } else {
                        // Find leader by mean
                        let leader = stats.iter().cloned().fold(stats[0],
                            |acc, x| if x.1 > acc.1 { x } else { acc });
                        // OCBA-lite: weight_i = var_i / max(gap_i², ε), leader gets sqrt(sum_others sq_weight)
                        let eps = 1.0_f64;
                        let mut weights: Vec<(usize, f64)> = Vec::with_capacity(alive.len());
                        let mut sum_w_sq = 0.0_f64;
                        for &(ci, m, v) in &stats {
                            if ci == leader.0 { continue; }
                            let gap = (leader.1 - m).abs().max(0.5);
                            let w = v / (gap * gap + eps);
                            weights.push((ci, w));
                            sum_w_sq += w * w / v.max(1.0);
                        }
                        let leader_w = (sum_w_sq.sqrt()).max(1.0);
                        weights.push((leader.0, leader_w));
                        let total_w: f64 = weights.iter().map(|(_, w)| *w).sum();
                        let scale = total_round as f64 / total_w.max(1e-9);
                        // Map weights back to alive order
                        let weight_map: std::collections::HashMap<usize, f64> =
                            weights.into_iter().collect();
                        alive.iter().map(|ci| {
                            let w = *weight_map.get(ci).unwrap_or(&1.0);
                            ((w * scale).round() as usize).max(1)
                        }).collect()
                    }
                };
                let mut work = Vec::with_capacity(pers.iter().sum());
                for (idx, &ci) in alive.iter().enumerate() {
                    for _ in 0..pers[idx] { work.push((ci, rng.gen())); }
                }
                run_work(work, &mut totals, &mut sumsq, &mut counts);
                // Standard halving on means
                if round < num_rounds - 1 {
                    let mut scored: Vec<(usize, f64)> = alive.iter()
                        .filter_map(|&ci| {
                            if counts[ci] == 0 { return None; }
                            Some((ci, totals[ci] as f64 / counts[ci] as f64))
                        }).collect();
                    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let keep = (scored.len() + 1) / 2;
                    alive = scored.into_iter().take(keep).map(|(i, _)| i).collect();
                }
            }
        }
        GreedyMceAlloc::SuccessiveRejects => {
            // Audibert & Bubeck 2010: K-1 phases, eliminate worst each time.
            // Simplified uniform schedule: phase k distributes ~T/(K-1)
            // rollouts equally among (K-k+1) remaining arms.
            let k_total = n_cands;
            if k_total >= 2 {
                let mut alive: Vec<usize> = (0..k_total).collect();
                let phases = k_total - 1;
                let per_phase_budget = num_rollouts / phases.max(1);
                for _phase in 0..phases {
                    if alive.len() < 2 { break; }
                    let per = (per_phase_budget / alive.len()).max(1);
                    let mut work = Vec::with_capacity(per * alive.len());
                    for &ci in &alive {
                        for _ in 0..per { work.push((ci, rng.gen())); }
                    }
                    run_work(work, &mut totals, &mut sumsq, &mut counts);
                    // Eliminate WORST arm
                    let mut scored: Vec<(usize, f64)> = alive.iter()
                        .filter_map(|&ci| {
                            if counts[ci] == 0 { return None; }
                            Some((ci, totals[ci] as f64 / counts[ci] as f64))
                        }).collect();
                    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                    if !scored.is_empty() {
                        let worst = scored[0].0;
                        alive.retain(|&ci| ci != worst);
                    }
                }
            } else {
                // 1 candidate: just allocate all rollouts to it
                let mut work = Vec::with_capacity(num_rollouts);
                for _ in 0..num_rollouts { work.push((0, rng.gen())); }
                run_work(work, &mut totals, &mut sumsq, &mut counts);
            }
        }
        GreedyMceAlloc::SeqHalvingPW => {
            // Progressive widening. Start with a NARROW front (first
            // min(4, n_cands) by input order). Run some rollouts. If top-2
            // are within 1σ, WIDEN by adding more candidates from reserve.
            // Continue halving.
            let initial_k = 4.min(n_cands);
            let mut alive: Vec<usize> = (0..initial_k).collect();
            let mut reserve: Vec<usize> = (initial_k..n_cands).collect();
            // Round 0: initial rollouts on narrow front
            let round0_budget = num_rollouts / 3;
            let per0 = (round0_budget / alive.len()).max(1);
            let mut work = Vec::with_capacity(per0 * alive.len());
            for &ci in &alive {
                for _ in 0..per0 { work.push((ci, rng.gen())); }
            }
            run_work(work, &mut totals, &mut sumsq, &mut counts);
            let mut remaining_budget = num_rollouts.saturating_sub(per0 * alive.len());

            // Widen if top-2 are close (within 1σ)
            let widen = {
                let mut stats: Vec<(usize, f64, f64)> = alive.iter().filter_map(|&ci| {
                    if counts[ci] == 0 { return None; }
                    let n = counts[ci] as f64;
                    let mean = totals[ci] as f64 / n;
                    let var = (sumsq[ci] as f64 / n - mean * mean).max(0.0);
                    let se = (var / n).sqrt();
                    Some((ci, mean, se))
                }).collect();
                stats.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                if stats.len() >= 2 {
                    let (_, m1, se1) = stats[0];
                    let (_, m2, se2) = stats[1];
                    let combined = (se1.powi(2) + se2.powi(2)).sqrt();
                    combined == 0.0 || (m1 - m2) < combined
                } else { false }
            };
            if widen && !reserve.is_empty() {
                let n_add = 4.min(reserve.len());
                alive.extend(reserve.drain(..n_add));
            }
            // Halving on (possibly widened) alive set with remaining budget
            let num_rounds = (alive.len() as f64).log2().ceil().max(1.0) as usize;
            let budget_per_round = (remaining_budget / num_rounds.max(1)).max(alive.len());
            for round in 0..num_rounds {
                if alive.is_empty() { break; }
                let per = (budget_per_round / alive.len()).max(1);
                let mut work = Vec::with_capacity(per * alive.len());
                for &ci in &alive {
                    for _ in 0..per { work.push((ci, rng.gen())); }
                }
                run_work(work, &mut totals, &mut sumsq, &mut counts);
                remaining_budget = remaining_budget.saturating_sub(per * alive.len());
                if round < num_rounds - 1 {
                    let mut scored: Vec<(usize, f64)> = alive.iter()
                        .filter_map(|&ci| {
                            if counts[ci] == 0 { return None; }
                            Some((ci, totals[ci] as f64 / counts[ci] as f64))
                        }).collect();
                    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let keep = (scored.len() + 1) / 2;
                    alive = scored.into_iter().take(keep).map(|(i, _)| i).collect();
                }
            }
        }
        GreedyMceAlloc::ThompsonSampling => {
            let init_per = (num_rollouts / (5 * n_cands)).max(2);
            let mut work = Vec::with_capacity(init_per * n_cands);
            for ci in 0..n_cands {
                for _ in 0..init_per { work.push((ci, rng.gen())); }
            }
            run_work(work, &mut totals, &mut sumsq, &mut counts);
            let mut spent = init_per * n_cands;
            let batch = 4.max(num_rollouts / 50);
            while spent + batch <= num_rollouts {
                let mut best_sample = f64::NEG_INFINITY;
                let mut best_ci = 0;
                for ci in 0..n_cands {
                    if counts[ci] == 0 { continue; }
                    let n = counts[ci] as f64;
                    let mean = totals[ci] as f64 / n;
                    let var = (sumsq[ci] as f64 / n - mean * mean).max(1.0);
                    let stderr = (var / n).sqrt();
                    let u1: f64 = rng.gen_range(0.0001f64..1.0);
                    let u2: f64 = rng.gen_range(0.0f64..std::f64::consts::TAU);
                    let z = (-2.0 * u1.ln()).sqrt() * u2.cos();
                    let sample = mean + stderr * z;
                    if sample > best_sample { best_sample = sample; best_ci = ci; }
                }
                let mut work = Vec::with_capacity(batch);
                for _ in 0..batch { work.push((best_ci, rng.gen())); }
                run_work(work, &mut totals, &mut sumsq, &mut counts);
                spent += batch;
            }
        }
        GreedyMceAlloc::MctsPW => {
            // greedy-MCE path: fallback to Uniform (MctsPW only meaningful
            // for NNUE-rollout path which has the real implementation).
            let per = num_rollouts / n_cands.max(1);
            let mut work = Vec::with_capacity(per * n_cands);
            for ci in 0..n_cands {
                for _ in 0..per { work.push((ci, rng.gen())); }
            }
            run_work(work, &mut totals, &mut sumsq, &mut counts);
        }
        GreedyMceAlloc::Puct => {
            // greedy-MCE path: fallback to Uniform (PUCT requires NNUE priors,
            // not available in greedy-MCE path).
            let per = num_rollouts / n_cands.max(1);
            let mut work = Vec::with_capacity(per * n_cands);
            for ci in 0..n_cands {
                for _ in 0..per { work.push((ci, rng.gen())); }
            }
            run_work(work, &mut totals, &mut sumsq, &mut counts);
        }
    }

    let mut best_idx = 0;
    let mut best_avg = f64::NEG_INFINITY;
    for (ci, (&t, &n)) in totals.iter().zip(counts.iter()).enumerate() {
        if n == 0 { continue; }
        let avg = t as f64 / n as f64;
        if avg > best_avg { best_avg = avg; best_idx = ci; }
    }
    if best_avg < 0.0 { return None; }
    let mv = candidates[best_idx];
    Some(ScoredMove { score: best_avg.round() as u16, ..mv })
}

/// Default candidate set for greedy MCE: same as old MCE.
pub fn default_greedy_mce_candidates(game: &GameState) -> Vec<ScoredMove> {
    let player = game.current_player;
    let mp: Vec<_> = game.market.available()
        .map(|(i, p)| (i, p.tile, p.wildlife)).collect();
    if mp.is_empty() { return Vec::new(); }
    let cards = game.scoring_cards;
    let turns = game.turns_remaining;
    let mut board = game.boards[player].clone();
    let greedy_best = best_move_with_potential(&mut board, &mp, &cards, turns);
    let mut candidates: Vec<ScoredMove> = candidate_moves_pub(game)
        .into_iter()
        .chain(wildlife_strategic_candidates(game).into_iter())
        .collect();
    if let Some(bm) = greedy_best {
        if !candidates.iter().any(|c| c.tile_q == bm.tile_q && c.tile_r == bm.tile_r
                                      && c.wildlife_q == bm.wildlife_q
                                      && c.wildlife_r == bm.wildlife_r
                                      && c.market_index == bm.market_index
                                      && c.wildlife_market_index == bm.wildlife_market_index) {
            candidates.push(bm);
        }
    }
    candidates.sort_by_key(|c| (c.market_index, c.wildlife_market_index, c.tile_q, c.tile_r,
                                c.wildlife_q, c.wildlife_r));
    candidates.dedup_by_key(|c| (c.market_index, c.wildlife_market_index, c.tile_q, c.tile_r,
                                 c.wildlife_q, c.wildlife_r));
    candidates
}

/// NNUE pre-filter: score each candidate by NNUE-predicted total
/// (current_score + NNUE_remaining), keep top-K. Returns input unchanged
/// if `candidates.len() <= k`. Cheap (no rollouts) — typically used
/// before MCE to focus the rollout budget on high-quality candidates.
pub fn nnue_prefilter_candidates(
    game: &GameState,
    net: &NNUENetwork,
    candidates: Vec<ScoredMove>,
    k: usize,
) -> Vec<ScoredMove> {
    let diverse: bool = std::env::var("MCE_DIVERSE_PREFILTER").ok()
        .map(|s| !s.is_empty() && s != "0").unwrap_or(false);
    if !diverse {
        return nnue_prefilter_with_priors(game, net, candidates, k).0;
    }

    // Diverse prefilter v2:
    //   Phase 1: guarantee 1 best candidate per market pair (4 slots)
    //   Phase 2: generate wildlife-placement variants of top-3 candidates
    //   Phase 3: fill remaining with best-overall
    let (all_scored, _) = nnue_prefilter_with_priors(game, net, candidates, 999);
    if all_scored.is_empty() { return all_scored; }

    let player = game.current_player;
    let cards = game.scoring_cards;
    let mut selected: Vec<ScoredMove> = Vec::with_capacity(k + 4);
    let mut used_keys: std::collections::HashSet<(usize, i8, i8, u8, Option<i8>, Option<i8>)> =
        std::collections::HashSet::new();

    let key_of = |mv: &ScoredMove| (mv.market_index, mv.tile_q, mv.tile_r, mv.rotation, mv.wildlife_q, mv.wildlife_r);
    let mut add = |mv: ScoredMove, sel: &mut Vec<ScoredMove>, keys: &mut std::collections::HashSet<_>| -> bool {
        let k = key_of(&mv);
        if keys.contains(&k) { return false; }
        keys.insert(k);
        sel.push(mv);
        true
    };

    // Phase 1: best per market pair
    {
        let mut best_per: std::collections::HashMap<usize, &ScoredMove> = std::collections::HashMap::new();
        for mv in &all_scored {
            best_per.entry(mv.market_index).or_insert(mv);
        }
        let mut picks: Vec<&ScoredMove> = best_per.values().copied().collect();
        picks.sort_by(|a, b| b.eval.cmp(&a.eval));
        for mv in picks {
            add(*mv, &mut selected, &mut used_keys);
        }
    }

    // Phase 2: wildlife-placement variants of top-3 candidates.
    // For each, place the tile, enumerate ALL valid wildlife positions,
    // score each with NNUE, keep the top-2 that differ from the original.
    let top_n = 3.min(all_scored.len());
    for i in 0..top_n {
        let base_mv = &all_scored[i];
        let market_pair = game.market.pairs[base_mv.market_index];
        let market_pair = match market_pair {
            Some(p) => p,
            None => continue,
        };
        let wildlife = if let Some(wmi) = base_mv.wildlife_market_index {
            match game.market.pairs[wmi] { Some(p) => p.wildlife, None => continue }
        } else {
            market_pair.wildlife
        };

        let mut gs = game.clone();
        let tile_coord = cascadia_core::hex::HexCoord::new(base_mv.tile_q, base_mv.tile_r);
        if gs.boards[player].place_tile(tile_coord, market_pair.tile, base_mv.rotation).is_none() {
            continue;
        }

        // Find all valid wildlife placements
        let variant = cards.variant_for(wildlife);
        let without = cascadia_core::scoring::wildlife::score_wildlife(
            &gs.boards[player], wildlife, variant,
        );
        let mut wl_options: Vec<(i8, i8, f32)> = Vec::new();
        let placed: arrayvec::ArrayVec<u16, 64> = gs.boards[player].placed_tiles.iter().copied().collect();
        for &ti in placed.iter() {
            if !gs.boards[player].grid.get(ti as usize).can_place_wildlife(wildlife) { continue; }
            let wa = match gs.boards[player].place_wildlife(ti as usize, wildlife) {
                Some(a) => a, None => continue,
            };
            // Quick NNUE score
            let bag = crate::nnue::BagInfo::from_game_for_player(&gs, player);
            let cur = cascadia_core::scoring::ScoreBreakdown::compute(
                &mut gs.boards[player].clone(), &cards,
            ).total as f32;
            let remaining = net.evaluate_with_bag(&gs.boards[player], &bag);
            let total = cur + remaining;

            let wc = cascadia_core::hex::HexCoord::from_index(ti as usize);
            wl_options.push((wc.q, wc.r, total));
            gs.boards[player].undo(wa);
        }
        wl_options.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Add top-2 alternatives that differ from the base
        let mut added = 0;
        for (wq, wr, score) in &wl_options {
            if added >= 2 { break; }
            if base_mv.wildlife_q == Some(*wq) && base_mv.wildlife_r == Some(*wr) { continue; }
            let variant_mv = ScoredMove {
                wildlife_q: Some(*wq),
                wildlife_r: Some(*wr),
                score: *score as u16,
                eval: (*score * 1000.0) as i32,
                ..*base_mv
            };
            if add(variant_mv, &mut selected, &mut used_keys) { added += 1; }
        }
    }

    // Phase 3: fill remaining with best-overall
    for mv in &all_scored {
        if selected.len() >= k { break; }
        add(*mv, &mut selected, &mut used_keys);
    }

    selected.truncate(k);
    selected
}

/// Prefilter ensemble: extra NNUE checkpoints whose predictions are averaged
/// with the primary net when scoring candidates. Gated by env var
/// `MCE_PREFILTER_ENSEMBLE=<path1,path2,...>`. Loaded lazily per env-var
/// value via a process-wide cache so that per-seat env changes (e.g. HH
/// with different strategies in same process) each see the correct ensemble.
///
/// Empty env / unset = no ensemble. Returns borrow; cost O(N_nets × N_cands)
/// per prefilter call. NB: leaks the loaded nets for 'static lifetime — fine
/// because env-var set is bounded and process is short-lived.
fn prefilter_ensemble_nets() -> &'static Vec<NNUENetwork> {
    use std::sync::{Mutex, OnceLock};
    use std::collections::HashMap;
    static CACHE: OnceLock<Mutex<HashMap<String, &'static Vec<NNUENetwork>>>> = OnceLock::new();
    static EMPTY: OnceLock<Vec<NNUENetwork>> = OnceLock::new();
    let empty_ref: &'static Vec<NNUENetwork> = EMPTY.get_or_init(Vec::new);

    let env_val = std::env::var("MCE_PREFILTER_ENSEMBLE").unwrap_or_default();
    if env_val.trim().is_empty() {
        return empty_ref;
    }
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = cache.lock().unwrap();
    if let Some(&nets) = map.get(&env_val) {
        return nets;
    }
    // Load all paths in env_val, skipping any that fail.
    let mut nets: Vec<NNUENetwork> = Vec::new();
    for p in env_val.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        match NNUENetwork::load(std::path::Path::new(p)) {
            Ok(n) => {
                eprintln!("[prefilter-ensemble] loaded {}", p);
                nets.push(n);
            }
            Err(e) => eprintln!("[prefilter-ensemble] failed {}: {} (skipped)", p, e),
        }
    }
    eprintln!("[prefilter-ensemble] active with {} extra net(s) for key '{}'",
              nets.len(), env_val);
    let leaked: &'static Vec<NNUENetwork> = Box::leak(Box::new(nets));
    map.insert(env_val, leaked);
    leaked
}

/// Same as `nnue_prefilter_candidates` but also returns the NNUE-predicted
/// total score per surviving candidate (in rank order, highest first).
/// Callers can use these priors for control-variate blending or LMR tiering.
///
/// If `MCE_STRATEGY_BIAS` env var is set, adds a committed-strategy bonus
/// to prefilter scores — boosts candidates drafting elk/salmon/hawk when
/// the board is already building such a pattern. Addresses the documented
/// elk/salmon/hawk under-scoring by preserving strategic candidates that
/// NNUE's immediate-score ranking would otherwise discard.
///
/// If `MCE_PREFILTER_ENSEMBLE=<paths>` is set, averages the primary net's
/// `remaining` prediction with the ensemble's for a more stable prior.
pub fn nnue_prefilter_with_priors(
    game: &GameState,
    net: &NNUENetwork,
    candidates: Vec<ScoredMove>,
    k: usize,
) -> (Vec<ScoredMove>, Vec<f32>) {
    let player = game.current_player;
    let cards = game.scoring_cards;
    let use_strategy_bias: bool = std::env::var("MCE_STRATEGY_BIAS").ok()
        .map(|s| !s.is_empty() && s != "0").unwrap_or(false);
    let ensemble = prefilter_ensemble_nets();
    let ensemble_size = ensemble.len();

    // Parallelize across threads for large candidate sets (expanded can be 40-50).
    // Serial path when small (overhead not worth it).
    let mut scored: Vec<(f32, ScoredMove)> = if candidates.len() >= 12 {
        let game_arc = std::sync::Arc::new(game.clone());
        let net_arc = std::sync::Arc::new(net.clone());
        let cands_arc = std::sync::Arc::new(candidates);
        let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let n_cands = cands_arc.len();
        let chunk = ((n_cands + num_threads - 1) / num_threads).max(1);
        let handles: Vec<_> = (0..num_threads).map(|t| {
            let start = t * chunk;
            let end = (start + chunk).min(n_cands);
            if start >= end { return None; }
            let g = std::sync::Arc::clone(&game_arc);
            let n = std::sync::Arc::clone(&net_arc);
            let c = std::sync::Arc::clone(&cands_arc);
            let orig_game = std::sync::Arc::clone(&game_arc);
            Some(std::thread::spawn(move || {
                let ensemble = prefilter_ensemble_nets();
                let mut local: Vec<(f32, ScoredMove)> = Vec::with_capacity(end - start);
                for i in start..end {
                    let mv = c[i];
                    let mut gs = (*g).clone();
                    if !execute_scored_move(&mut gs, &mv) { continue; }
                    let bag = crate::nnue::BagInfo::from_game_for_player(&gs, player);
                    let board = &gs.boards[player];
                    let cur = ScoreBreakdown::compute(&mut board.clone(), &cards).total as f32;
                    let mut remaining = n.evaluate_with_bag(board, &bag);
                    if !ensemble.is_empty() {
                        let mut sum = remaining;
                        for enet in ensemble {
                            sum += enet.evaluate_with_bag(board, &bag);
                        }
                        remaining = sum / (1.0 + ensemble.len() as f32);
                    }
                    let mut total = cur + remaining;
                    if use_strategy_bias {
                        total += strategy_commit_bonus(&orig_game, &mv);
                    }
                    local.push((total, mv));
                }
                local
            }))
        }).collect();
        let mut all: Vec<(f32, ScoredMove)> = Vec::with_capacity(n_cands);
        for opt in handles {
            if let Some(h) = opt {
                all.extend(h.join().unwrap());
            }
        }
        all
    } else {
        // Serial path for small candidate counts
        candidates.into_iter().filter_map(|mv| {
            let mut g = game.clone();
            if !execute_scored_move(&mut g, &mv) { return None; }
            let bag = crate::nnue::BagInfo::from_game_for_player(&g, player);
            let board = &g.boards[player];
            let cur = ScoreBreakdown::compute(&mut board.clone(), &cards).total as f32;
            let mut remaining = net.evaluate_with_bag(board, &bag);
            if ensemble_size > 0 {
                let mut sum = remaining;
                for enet in ensemble {
                    sum += enet.evaluate_with_bag(board, &bag);
                }
                remaining = sum / (1.0 + ensemble_size as f32);
            }
            let mut total = cur + remaining;
            if use_strategy_bias {
                total += strategy_commit_bonus(game, &mv);
            }
            Some((total, mv))
        }).collect()
    };
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let kept: Vec<(f32, ScoredMove)> = scored.into_iter().take(k).collect();
    let mvs: Vec<ScoredMove> = kept.iter().map(|(_, mv)| *mv).collect();
    let priors: Vec<f32> = kept.iter().map(|(p, _)| *p).collect();
    (mvs, priors)
}

/// Committed-strategy bonus: bias prefilter toward elk/salmon/hawk drafts
/// when a pattern is forming. Addresses documented under-scoring of these
/// animals (current 5-10 pts each vs ceiling ~14-26).
fn strategy_commit_bonus(game: &GameState, mv: &ScoredMove) -> f32 {
    // Identify wildlife drafted by this move
    let wl_market_idx = mv.wildlife_market_index.unwrap_or(mv.market_index);
    let pair = match game.market.pairs.get(wl_market_idx).and_then(|p| p.as_ref()) {
        Some(p) => p,
        None => return 0.0,
    };
    let wildlife = pair.wildlife;
    let board = &game.boards[game.current_player];
    let on_board = board.wildlife_positions[wildlife as usize].len() as f32;

    use cascadia_core::types::Wildlife;
    match wildlife {
        // Elk/Salmon/Hawk: big scoring jumps at 4-5+ chain. Bias toward accumulation.
        Wildlife::Elk | Wildlife::Salmon | Wildlife::Hawk => {
            // Ramp: strongest bonus when board already has 2-4 of this type
            // (close to a scoring threshold).
            let ramp = (on_board.min(5.0)) * 0.8 + 1.5;
            ramp  // +1.5 to +5.5 for these animals
        }
        _ => 0.0,
    }
}

/// Backward-compat: uniform-allocation greedy MCE (original implementation).
pub fn best_move_greedy_mce(
    game: &GameState,
    num_rollouts: usize,
    rng: &mut StdRng,
) -> Option<ScoredMove> {
    let candidates = default_greedy_mce_candidates(game);
    best_move_greedy_mce_v2(game, num_rollouts, GreedyMceAlloc::Uniform, candidates, rng)
}

/// NNUE-rollout MCE: rollouts use NNUE for player 0's moves (instead of greedy).
/// Opponents still play greedy (matches all our benchmarks). Allocation can be any
/// of the GreedyMceAlloc variants.
///
/// This is the "policy-guided rollout" analog to AlphaGo's rollout policy.
pub fn best_move_nnue_rollout_mce(
    game: &GameState,
    net: &NNUENetwork,
    num_rollouts: usize,
    alloc: GreedyMceAlloc,
    candidates: Vec<ScoredMove>,
    rng: &mut StdRng,
) -> Option<ScoredMove> {
    let player = game.current_player;
    if candidates.is_empty() { return None; }
    let n_cands = candidates.len();
    let mut totals = vec![0u64; n_cands];
    let mut sumsq = vec![0u64; n_cands];
    let mut counts = vec![0u32; n_cands];

    // MCE_CV_ALPHA: control-variate blend factor in [0, 1].
    //   1.0 = pure rollout mean (default, unchanged behavior)
    //   0.0 = pure NNUE prior (no rollout influence)
    //   0.85 = 85% rollout + 15% NNUE (shrinkage toward stable NNUE baseline)
    let cv_alpha: f32 = std::env::var("MCE_CV_ALPHA").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(1.0);
    // MCE_LMR: if set (any non-empty, non-"0"), use tiered budget by NNUE prior rank.
    let use_lmr: bool = std::env::var("MCE_LMR").ok()
        .map(|s| !s.is_empty() && s != "0").unwrap_or(false);

    // Compute NNUE priors for each candidate if CV or LMR enabled (O(n_cands) forward passes)
    let priors: Vec<f32> = if cv_alpha < 1.0 || use_lmr {
        let cards = game.scoring_cards;
        candidates.iter().map(|mv| {
            let mut g = game.clone();
            if !execute_scored_move(&mut g, mv) { return 0.0; }
            let bag = crate::nnue::BagInfo::from_game_for_player(&g, player);
            let board = &g.boards[player];
            let cur = ScoreBreakdown::compute(&mut board.clone(), &cards).total as f32;
            let remaining = net.evaluate_with_bag(board, &bag);
            cur + remaining
        }).collect()
    } else { Vec::new() };

    // LMR: rank candidates by prior (rank 0 = best).
    let ranks: Vec<usize> = if use_lmr && !priors.is_empty() {
        let mut idx_prior: Vec<(usize, f32)> = priors.iter().enumerate()
            .map(|(i, &p)| (i, p)).collect();
        idx_prior.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut r = vec![0usize; priors.len()];
        for (rank, (i, _)) in idx_prior.iter().enumerate() {
            r[*i] = rank;
        }
        r
    } else { Vec::new() };
    // LMR multiplier: rank-0 gets 2x budget, rank-1 gets 1.5x, rest 1x.
    let lmr_mult = |ci: usize| -> f64 {
        if !use_lmr || ranks.is_empty() { return 1.0; }
        match ranks[ci] {
            0 => 2.0,
            1 => 1.5,
            _ => 1.0,
        }
    };

    let game_arc = std::sync::Arc::new(game.clone());
    let cands_arc = std::sync::Arc::new(candidates.clone());
    let net_arc = std::sync::Arc::new(net.clone());

    // MCE_CONTROL_VARIATES: enable per-rollout NNUE-eval-as-control-variate
    // adjustment at decision time. Subtracts β·(B - meanB) from rollout means
    // to reduce variance. β estimated online per candidate via Cov(R,B)/Var(B).
    let use_cv: bool = std::env::var("MCE_CONTROL_VARIATES").ok()
        .map(|s| !s.is_empty() && s != "0").unwrap_or(false);
    // Accumulators for control variates (always populated; cheap).
    let cv_totals = std::cell::RefCell::new(vec![0u64; n_cands]);
    let cv_sumsq = std::cell::RefCell::new(vec![0u64; n_cands]);
    let cross = std::cell::RefCell::new(vec![0i128; n_cands]);

    let run_work = |work_items: Vec<(usize, u64)>,
                    totals: &mut Vec<u64>, sumsq: &mut Vec<u64>, counts: &mut Vec<u32>| {
        let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let chunk_size = ((work_items.len() + num_threads - 1) / num_threads).max(1);
        let handles: Vec<_> = work_items.chunks(chunk_size).map(|chunk| {
            let work = chunk.to_vec();
            let g = std::sync::Arc::clone(&game_arc);
            let c = std::sync::Arc::clone(&cands_arc);
            let n = std::sync::Arc::clone(&net_arc);
            std::thread::spawn(move || {
                let mut results: Vec<(usize, u64, u64)> = Vec::with_capacity(work.len());
                for &(ci, seed) in &work {
                    let (score, cv) = run_nnue_rollout((*g).clone(), player, seed, &c[ci], &n);
                    results.push((ci, score, cv));
                }
                results
            })
        }).collect();
        let mut cvt = cv_totals.borrow_mut();
        let mut cvs = cv_sumsq.borrow_mut();
        let mut crs = cross.borrow_mut();
        for h in handles {
            for (ci, score, cv) in h.join().unwrap() {
                totals[ci] += score;
                sumsq[ci] += score * score;
                counts[ci] += 1;
                cvt[ci] += cv;
                cvs[ci] += cv * cv;
                crs[ci] += (score as i128) * (cv as i128);
            }
        }
    };

    // MCE_GUMBEL_HALVING: full Gumbel-Top-K + Sequential Halving (Danihelka et al.
    // 2022, "Policy improvement by planning with Gumbel"). Gumbel noise is drawn
    // once per candidate and frozen across halving rounds; survival depends on
    // (rollout_mean + σ·gumbel_prior). Candidates with low NNUE priors can
    // survive via lucky Gumbel draws, giving strict policy improvement at small
    // budgets vs deterministic halving.
    let use_gumbel_halving: bool = std::env::var("MCE_GUMBEL_HALVING").ok()
        .map(|s| !s.is_empty() && s != "0").unwrap_or(false);
    let gumbel_sigma: f64 = std::env::var("MCE_GUMBEL_HALVING_SIGMA").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(3.0);
    let gumbel_priors: Vec<f64> = if use_gumbel_halving {
        // Draw Gumbel(0, 1) per candidate via inverse CDF: -log(-log(U))
        (0..n_cands).map(|_| {
            let u: f64 = rng.gen_range(1e-12..1.0);
            -(-u.ln()).ln()
        }).collect()
    } else {
        Vec::new()
    };

    // Dispatch by allocator (reuse the same logic as greedy MCE)
    match alloc {
        GreedyMceAlloc::Uniform | GreedyMceAlloc::UniformCRN => {
            let per = num_rollouts / n_cands.max(1);
            let base_seeds: Vec<u64> = if alloc == GreedyMceAlloc::UniformCRN {
                (0..per).map(|_| rng.gen()).collect()
            } else { Vec::new() };
            let mut work = Vec::with_capacity(per * n_cands);
            for ci in 0..n_cands {
                for k in 0..per {
                    let seed = if alloc == GreedyMceAlloc::UniformCRN { base_seeds[k] } else { rng.gen() };
                    work.push((ci, seed));
                }
            }
            run_work(work, &mut totals, &mut sumsq, &mut counts);
        }
        GreedyMceAlloc::SeqHalving | GreedyMceAlloc::SeqHalvingCRN => {
            let num_rounds = (n_cands as f64).log2().ceil().max(1.0) as usize;
            let mut alive: Vec<usize> = (0..n_cands).collect();
            let budget_per_round = (num_rollouts / num_rounds).max(n_cands);
            for round in 0..num_rounds {
                if alive.is_empty() { break; }
                let base_per = (budget_per_round / alive.len()).max(1);
                // LMR: per-candidate per via rank multiplier. Normalize so total ≈ base_per * alive.len().
                let raw_pers: Vec<f64> = alive.iter().map(|&ci| (base_per as f64) * lmr_mult(ci)).collect();
                let raw_sum: f64 = raw_pers.iter().sum();
                let target_sum = (base_per as f64) * (alive.len() as f64);
                let scale = if raw_sum > 0.0 { target_sum / raw_sum } else { 1.0 };
                let pers: Vec<usize> = raw_pers.iter().map(|p| ((p * scale).round() as usize).max(1)).collect();
                let max_per = *pers.iter().max().unwrap_or(&base_per);
                let base_seeds: Vec<u64> = if alloc == GreedyMceAlloc::SeqHalvingCRN {
                    (0..max_per).map(|_| rng.gen()).collect()
                } else { Vec::new() };
                let mut work = Vec::with_capacity(pers.iter().sum());
                for (idx, &ci) in alive.iter().enumerate() {
                    for k in 0..pers[idx] {
                        let seed = if alloc == GreedyMceAlloc::SeqHalvingCRN { base_seeds[k] } else { rng.gen() };
                        work.push((ci, seed));
                    }
                }
                run_work(work, &mut totals, &mut sumsq, &mut counts);
                if round < num_rounds - 1 {
                    let mut scored: Vec<(usize, f64)> = alive.iter()
                        .filter_map(|&ci| {
                            if counts[ci] == 0 { return None; }
                            let mean = totals[ci] as f64 / counts[ci] as f64;
                            // Gumbel AZ: add frozen Gumbel prior to rollout mean so low-prior
                            // candidates with lucky samples survive until their evidence either
                            // confirms or rejects them. σ controls exploration strength.
                            let score = if use_gumbel_halving && !gumbel_priors.is_empty() {
                                mean + gumbel_sigma * gumbel_priors[ci]
                            } else {
                                mean
                            };
                            Some((ci, score))
                        }).collect();
                    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let keep = (scored.len() + 1) / 2;
                    alive = scored.into_iter().take(keep).map(|(i, _)| i).collect();
                }
            }
        }
        GreedyMceAlloc::SeqHalvingEarlyTerm => {
            let num_rounds = (n_cands as f64).log2().ceil().max(1.0) as usize;
            let mut alive: Vec<usize> = (0..n_cands).collect();
            let budget_per_round = (num_rollouts / num_rounds).max(n_cands);
            for round in 0..num_rounds {
                if alive.is_empty() { break; }
                let per = (budget_per_round / alive.len()).max(1);
                let mut work = Vec::with_capacity(per * alive.len());
                for &ci in &alive {
                    for _ in 0..per { work.push((ci, rng.gen())); }
                }
                run_work(work, &mut totals, &mut sumsq, &mut counts);
                if alive.len() >= 2 {
                    let mut scored: Vec<(usize, f64, f64)> = alive.iter()
                        .filter_map(|&ci| {
                            if counts[ci] == 0 { return None; }
                            let n = counts[ci] as f64;
                            let mean = totals[ci] as f64 / n;
                            let var = (sumsq[ci] as f64 / n - mean * mean).max(0.0);
                            let stderr = (var / n).sqrt();
                            Some((ci, mean, stderr))
                        }).collect();
                    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    if scored.len() >= 2 {
                        let (_, best_mean, best_se) = scored[0];
                        let (_, second_mean, second_se) = scored[1];
                        let gap = best_mean - second_mean;
                        let combined_se = (best_se.powi(2) + second_se.powi(2)).sqrt();
                        if combined_se > 0.0 && gap > 2.5 * combined_se {
                            break;
                        }
                    }
                }
                if round < num_rounds - 1 {
                    let mut scored: Vec<(usize, f64)> = alive.iter()
                        .filter_map(|&ci| {
                            if counts[ci] == 0 { return None; }
                            Some((ci, totals[ci] as f64 / counts[ci] as f64))
                        }).collect();
                    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let keep = (scored.len() + 1) / 2;
                    alive = scored.into_iter().take(keep).map(|(i, _)| i).collect();
                }
            }
        }
        GreedyMceAlloc::Ucb => {
            let init_per = ((num_rollouts / 10) / n_cands.max(1)).max(2);
            let mut work = Vec::with_capacity(init_per * n_cands);
            for ci in 0..n_cands {
                for _ in 0..init_per { work.push((ci, rng.gen())); }
            }
            run_work(work, &mut totals, &mut sumsq, &mut counts);
            let mut remaining = num_rollouts.saturating_sub(init_per * n_cands);
            let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
            while remaining > 0 {
                let batch = remaining.min(num_threads * 4);
                let total_n: u32 = counts.iter().sum();
                let log_total = (total_n.max(1) as f64).ln();
                let mut ucb_work = Vec::with_capacity(batch);
                for _ in 0..batch {
                    let mut best = 0usize;
                    let mut best_ucb = f64::NEG_INFINITY;
                    for ci in 0..n_cands {
                        let n = counts[ci].max(1) as f64;
                        let mean = totals[ci] as f64 / n;
                        let explore = (2.0 * log_total / n).sqrt();
                        let ucb = mean / 100.0 + 0.1 * explore;
                        if ucb > best_ucb { best_ucb = ucb; best = ci; }
                    }
                    ucb_work.push((best, rng.gen()));
                    counts[best] += 1;
                }
                for &(ci, _) in &ucb_work { counts[ci] -= 1; }
                run_work(ucb_work, &mut totals, &mut sumsq, &mut counts);
                remaining = remaining.saturating_sub(batch);
            }
        }
        GreedyMceAlloc::SeqHalvingCI => {
            // Confidence-interval-aware halving. Each round, eliminate any
            // candidate whose UCB < leader's LCB. Adaptive elimination rate.
            // Z tunable via MCE_HALVING_CI_Z (default 1.5).
            // MCE_HALVING_CI_FLOOR=1 → also enforce hard top-half-by-mean cap.
            let z: f64 = std::env::var("MCE_HALVING_CI_Z").ok()
                .and_then(|s| s.parse().ok()).unwrap_or(1.5);
            let hard_floor: bool = std::env::var("MCE_HALVING_CI_FLOOR").ok()
                .map(|s| !s.is_empty() && s != "0").unwrap_or(false);
            let num_rounds = (n_cands as f64).log2().ceil().max(1.0) as usize;
            let mut alive: Vec<usize> = (0..n_cands).collect();
            let budget_per_round = (num_rollouts / num_rounds).max(n_cands);
            for round in 0..num_rounds {
                if alive.is_empty() { break; }
                let per = (budget_per_round / alive.len()).max(1);
                let mut work = Vec::with_capacity(per * alive.len());
                for &ci in &alive {
                    for _ in 0..per { work.push((ci, rng.gen())); }
                }
                run_work(work, &mut totals, &mut sumsq, &mut counts);

                if round < num_rounds - 1 && alive.len() >= 2 {
                    // Compute (mean, stderr) for alive candidates
                    let stats: Vec<(usize, f64, f64)> = alive.iter().filter_map(|&ci| {
                        if counts[ci] == 0 { return None; }
                        let n = counts[ci] as f64;
                        let mean = totals[ci] as f64 / n;
                        let var = (sumsq[ci] as f64 / n - mean * mean).max(0.0);
                        let stderr = (var / n).sqrt();
                        Some((ci, mean, stderr))
                    }).collect();
                    if stats.is_empty() { break; }
                    let leader_lcb = stats.iter().fold(f64::NEG_INFINITY, |acc, &(_, m, se)| {
                        let lcb = m - z * se;
                        if lcb > acc { lcb } else { acc }
                    });
                    // Keep candidates whose UCB >= leader_lcb
                    let mut kept: Vec<usize> = stats.iter().filter_map(|&(ci, m, se)| {
                        if m + z * se >= leader_lcb { Some(ci) } else { None }
                    }).collect();
                    if hard_floor {
                        let mut by_mean: Vec<(usize, f64)> = stats.iter()
                            .map(|&(ci, m, _)| (ci, m)).collect();
                        by_mean.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                        let max_keep = ((alive.len() + 1) / 2).max(2);
                        let top: std::collections::HashSet<usize> = by_mean.into_iter()
                            .take(max_keep).map(|(ci, _)| ci).collect();
                        kept.retain(|ci| top.contains(ci));
                    }
                    // Safety: always keep at least 2 (so final round has competition)
                    if kept.len() < 2 {
                        let mut by_mean: Vec<(usize, f64)> = stats.iter()
                            .map(|&(ci, m, _)| (ci, m)).collect();
                        by_mean.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                        kept = by_mean.into_iter().take(2.min(stats.len())).map(|(ci, _)| ci).collect();
                    }
                    alive = kept;
                }
            }
        }
        GreedyMceAlloc::SeqHalvingHetero => {
            // Heteroscedastic-variance-weighted halving (NNUE-rollout-MCE path).
            // Within each round, allocate budget proportional to var/gap² so
            // high-variance candidates near the leader get more samples (OCBA).
            // Eliminate by hard halving each round.
            let num_rounds = (n_cands as f64).log2().ceil().max(1.0) as usize;
            let mut alive: Vec<usize> = (0..n_cands).collect();
            let budget_per_round = (num_rollouts / num_rounds).max(n_cands);
            for round in 0..num_rounds {
                if alive.is_empty() { break; }
                let total_round = budget_per_round.max(alive.len());
                let pers: Vec<usize> = if round == 0 {
                    let per = (total_round / alive.len()).max(1);
                    let raw_pers: Vec<f64> = alive.iter().map(|&ci| (per as f64) * lmr_mult(ci)).collect();
                    let raw_sum: f64 = raw_pers.iter().sum();
                    let target_sum = (per as f64) * (alive.len() as f64);
                    let scale = if raw_sum > 0.0 { target_sum / raw_sum } else { 1.0 };
                    raw_pers.iter().map(|p| ((p * scale).round() as usize).max(1)).collect()
                } else {
                    let stats: Vec<(usize, f64, f64)> = alive.iter().filter_map(|&ci| {
                        if counts[ci] == 0 { return None; }
                        let n = counts[ci] as f64;
                        let mean = totals[ci] as f64 / n;
                        let var = (sumsq[ci] as f64 / n - mean * mean).max(1.0);
                        Some((ci, mean, var))
                    }).collect();
                    if stats.is_empty() {
                        let per = (total_round / alive.len()).max(1);
                        alive.iter().map(|_| per).collect()
                    } else {
                        let leader = stats.iter().cloned().fold(stats[0],
                            |acc, x| if x.1 > acc.1 { x } else { acc });
                        let eps = 1.0_f64;
                        let mut weights: Vec<(usize, f64)> = Vec::with_capacity(alive.len());
                        let mut sum_w_sq = 0.0_f64;
                        for &(ci, m, v) in &stats {
                            if ci == leader.0 { continue; }
                            let gap = (leader.1 - m).abs().max(0.5);
                            let w = v / (gap * gap + eps);
                            weights.push((ci, w));
                            sum_w_sq += w * w / v.max(1.0);
                        }
                        let leader_w = (sum_w_sq.sqrt()).max(1.0);
                        weights.push((leader.0, leader_w));
                        let total_w: f64 = weights.iter().map(|(_, w)| *w).sum();
                        let scale = total_round as f64 / total_w.max(1e-9);
                        let weight_map: std::collections::HashMap<usize, f64> =
                            weights.into_iter().collect();
                        alive.iter().map(|ci| {
                            let w = *weight_map.get(ci).unwrap_or(&1.0);
                            ((w * scale).round() as usize).max(1)
                        }).collect()
                    }
                };
                let mut work = Vec::with_capacity(pers.iter().sum());
                for (idx, &ci) in alive.iter().enumerate() {
                    for _ in 0..pers[idx] { work.push((ci, rng.gen())); }
                }
                run_work(work, &mut totals, &mut sumsq, &mut counts);
                if round < num_rounds - 1 {
                    let mut scored: Vec<(usize, f64)> = alive.iter()
                        .filter_map(|&ci| {
                            if counts[ci] == 0 { return None; }
                            Some((ci, totals[ci] as f64 / counts[ci] as f64))
                        }).collect();
                    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let keep = (scored.len() + 1) / 2;
                    alive = scored.into_iter().take(keep).map(|(i, _)| i).collect();
                }
            }
        }
        GreedyMceAlloc::SuccessiveRejects => {
            if n_cands >= 2 {
                let mut alive: Vec<usize> = (0..n_cands).collect();
                let phases = n_cands - 1;
                let per_phase_budget = num_rollouts / phases.max(1);
                for _phase in 0..phases {
                    if alive.len() < 2 { break; }
                    let base_per = (per_phase_budget / alive.len()).max(1);
                    let raw_pers: Vec<f64> = alive.iter().map(|&ci| (base_per as f64) * lmr_mult(ci)).collect();
                    let raw_sum: f64 = raw_pers.iter().sum();
                    let target_sum = (base_per as f64) * (alive.len() as f64);
                    let scale = if raw_sum > 0.0 { target_sum / raw_sum } else { 1.0 };
                    let pers: Vec<usize> = raw_pers.iter().map(|p| ((p * scale).round() as usize).max(1)).collect();
                    let mut work = Vec::with_capacity(pers.iter().sum());
                    for (idx, &ci) in alive.iter().enumerate() {
                        for _ in 0..pers[idx] { work.push((ci, rng.gen())); }
                    }
                    run_work(work, &mut totals, &mut sumsq, &mut counts);
                    let mut scored: Vec<(usize, f64)> = alive.iter()
                        .filter_map(|&ci| {
                            if counts[ci] == 0 { return None; }
                            Some((ci, totals[ci] as f64 / counts[ci] as f64))
                        }).collect();
                    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                    if !scored.is_empty() {
                        let worst = scored[0].0;
                        alive.retain(|&ci| ci != worst);
                    }
                }
            } else {
                let mut work = Vec::with_capacity(num_rollouts);
                for _ in 0..num_rollouts { work.push((0, rng.gen())); }
                run_work(work, &mut totals, &mut sumsq, &mut counts);
            }
        }
        GreedyMceAlloc::SeqHalvingPW => {
            let initial_k = 4.min(n_cands);
            let mut alive: Vec<usize> = (0..initial_k).collect();
            let mut reserve: Vec<usize> = (initial_k..n_cands).collect();
            let round0_budget = num_rollouts / 3;
            let per0 = (round0_budget / alive.len()).max(1);
            let mut work = Vec::with_capacity(per0 * alive.len());
            for &ci in &alive {
                for _ in 0..per0 { work.push((ci, rng.gen())); }
            }
            run_work(work, &mut totals, &mut sumsq, &mut counts);
            let mut remaining_budget = num_rollouts.saturating_sub(per0 * alive.len());

            let widen = {
                let mut stats: Vec<(usize, f64, f64)> = alive.iter().filter_map(|&ci| {
                    if counts[ci] == 0 { return None; }
                    let n = counts[ci] as f64;
                    let mean = totals[ci] as f64 / n;
                    let var = (sumsq[ci] as f64 / n - mean * mean).max(0.0);
                    let se = (var / n).sqrt();
                    Some((ci, mean, se))
                }).collect();
                stats.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                if stats.len() >= 2 {
                    let (_, m1, se1) = stats[0];
                    let (_, m2, se2) = stats[1];
                    let combined = (se1.powi(2) + se2.powi(2)).sqrt();
                    combined == 0.0 || (m1 - m2) < combined
                } else { false }
            };
            if widen && !reserve.is_empty() {
                let n_add = 4.min(reserve.len());
                alive.extend(reserve.drain(..n_add));
            }
            let num_rounds = (alive.len() as f64).log2().ceil().max(1.0) as usize;
            let budget_per_round = (remaining_budget / num_rounds.max(1)).max(alive.len());
            for round in 0..num_rounds {
                if alive.is_empty() { break; }
                let per = (budget_per_round / alive.len()).max(1);
                let mut work = Vec::with_capacity(per * alive.len());
                for &ci in &alive {
                    for _ in 0..per { work.push((ci, rng.gen())); }
                }
                run_work(work, &mut totals, &mut sumsq, &mut counts);
                remaining_budget = remaining_budget.saturating_sub(per * alive.len());
                if round < num_rounds - 1 {
                    let mut scored: Vec<(usize, f64)> = alive.iter()
                        .filter_map(|&ci| {
                            if counts[ci] == 0 { return None; }
                            Some((ci, totals[ci] as f64 / counts[ci] as f64))
                        }).collect();
                    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let keep = (scored.len() + 1) / 2;
                    alive = scored.into_iter().take(keep).map(|(i, _)| i).collect();
                }
            }
        }
        GreedyMceAlloc::ThompsonSampling => {
            // Phase 1: give each candidate a small initial budget for priors.
            let init_per = (num_rollouts / (5 * n_cands)).max(2);
            let mut work = Vec::with_capacity(init_per * n_cands);
            for ci in 0..n_cands {
                for _ in 0..init_per { work.push((ci, rng.gen())); }
            }
            run_work(work, &mut totals, &mut sumsq, &mut counts);
            let mut spent = init_per * n_cands;
            // Phase 2: Thompson Sampling — batch allocation. Each round,
            // sample from Normal(mean, sqrt(var/n)) per candidate, pick
            // the highest sample, give it a batch of rollouts.
            let batch = 4.max(num_rollouts / 50);
            while spent + batch <= num_rollouts {
                // Sample from posterior for each candidate
                let mut best_sample = f64::NEG_INFINITY;
                let mut best_ci = 0;
                for ci in 0..n_cands {
                    if counts[ci] == 0 { continue; }
                    let n = counts[ci] as f64;
                    let mean = totals[ci] as f64 / n;
                    let var = (sumsq[ci] as f64 / n - mean * mean).max(1.0);
                    let stderr = (var / n).sqrt();
                    // Box-Muller for normal sample
                    let u1: f64 = rng.gen_range(0.0001f64..1.0);
                    let u2: f64 = rng.gen_range(0.0f64..std::f64::consts::TAU);
                    let z = (-2.0 * u1.ln()).sqrt() * u2.cos();
                    let sample = mean + stderr * z;
                    if sample > best_sample {
                        best_sample = sample;
                        best_ci = ci;
                    }
                }
                let mut work = Vec::with_capacity(batch);
                for _ in 0..batch { work.push((best_ci, rng.gen())); }
                run_work(work, &mut totals, &mut sumsq, &mut counts);
                spent += batch;
            }
        }
        GreedyMceAlloc::MctsPW => {
            // Candidates are assumed sorted by NNUE prior (best-first) when
            // best_move_nnue_rollout_mce is called with halving-prepped pool.
            // PW params (tunable via env):
            //   MCE_MCTSPW_K_INIT — initial expanded candidates (default 4)
            //   MCE_MCTSPW_K      — widening rate multiplier (default 2)
            //   MCE_MCTSPW_ALPHA  — widening exponent (default 0.5 → sqrt)
            //   MCE_MCTSPW_UCB_C  — UCB1 exploration constant (default 0.25)
            //   MCE_MCTSPW_BATCH  — rollouts per step (default 4; 1 is pure MCTS
            //                        but wastes thread parallelism)
            let k_init: usize = std::env::var("MCE_MCTSPW_K_INIT").ok()
                .and_then(|s| s.parse().ok()).unwrap_or(4);
            let pw_k: f64 = std::env::var("MCE_MCTSPW_K").ok()
                .and_then(|s| s.parse().ok()).unwrap_or(2.0);
            let pw_alpha: f64 = std::env::var("MCE_MCTSPW_ALPHA").ok()
                .and_then(|s| s.parse().ok()).unwrap_or(0.5);
            let ucb_c: f64 = std::env::var("MCE_MCTSPW_UCB_C").ok()
                .and_then(|s| s.parse().ok()).unwrap_or(0.25);
            let batch: usize = std::env::var("MCE_MCTSPW_BATCH").ok()
                .and_then(|s| s.parse().ok()).unwrap_or(4);

            let mut expanded: usize = k_init.min(n_cands);
            // Seed each expanded candidate with 1 rollout so UCB has non-zero counts.
            let mut seed_work = Vec::with_capacity(expanded);
            for ci in 0..expanded {
                seed_work.push((ci, rng.gen()));
            }
            run_work(seed_work, &mut totals, &mut sumsq, &mut counts);
            let mut spent = expanded;

            while spent < num_rollouts {
                let total_visits: u32 = counts.iter().take(expanded).sum();
                // Progressive widening: allow new candidate if k·N^α > expanded.
                let allowed = (pw_k * (total_visits.max(1) as f64).powf(pw_alpha)).ceil() as usize;
                if allowed > expanded && expanded < n_cands {
                    let ci_new = expanded;
                    expanded += 1;
                    // Seed new candidate with 1 rollout to get it a count.
                    let mut w = Vec::with_capacity(1);
                    w.push((ci_new, rng.gen()));
                    run_work(w, &mut totals, &mut sumsq, &mut counts);
                    spent += 1;
                    continue;
                }
                // UCB1 selection over expanded candidates.
                let log_total = (total_visits.max(1) as f64).ln();
                let mut best_ucb = f64::NEG_INFINITY;
                let mut best_ci = 0usize;
                for ci in 0..expanded {
                    let n = counts[ci].max(1) as f64;
                    let mean = totals[ci] as f64 / n;
                    // Normalize reward to [0,1] via /100 to match `c` scale.
                    let explore = (log_total / n).sqrt();
                    let score = mean / 100.0 + ucb_c * explore;
                    if score > best_ucb { best_ucb = score; best_ci = ci; }
                }
                // Dispatch `batch` rollouts to the selected candidate to amortize
                // thread-spawn overhead (pure-MCTS 1-per-step is serial-limited).
                let actual_batch = batch.min(num_rollouts - spent);
                let mut work = Vec::with_capacity(actual_batch);
                for _ in 0..actual_batch { work.push((best_ci, rng.gen())); }
                run_work(work, &mut totals, &mut sumsq, &mut counts);
                spent += actual_batch;
            }
        }
        GreedyMceAlloc::Puct => {
            // PUCT: AlphaZero-style allocator. Selection rule:
            //   PUCT(a) = Q(a) + c · P(a) · √N / (1 + n_a)
            // where Q(a) = empirical mean (normalized to [0,1] via /100 to match c scale),
            // P(a) = softmax over NNUE priors with temperature τ, N = total visits.
            // Differs from MctsPW: prior probability (not visit-rank) drives exploration.
            // Lit: AlphaGo Zero (Silver 2017), AlphaZero (Silver 2018).
            let c_puct: f64 = std::env::var("MCE_PUCT_C").ok()
                .and_then(|s| s.parse().ok()).unwrap_or(2.0);
            let tau: f64 = std::env::var("MCE_PUCT_TAU").ok()
                .and_then(|s| s.parse().ok()).unwrap_or(8.0);
            // Default batch=32 to amortize thread-spawn overhead (~10 cores × 4 rollouts each).
            // PUCT loses much of its theoretical advantage at batch>1, but pure-MCTS-1 is
            // serial-limited and 30x slower. 32 is a pragmatic choice.
            let batch: usize = std::env::var("MCE_PUCT_BATCH").ok()
                .and_then(|s| s.parse().ok()).unwrap_or(32);
            // Compute P(a) from NNUE priors via softmax.
            let priors_f64: Vec<f64> = if !priors.is_empty() {
                priors.iter().map(|&p| p as f64).collect()
            } else {
                vec![0.0; n_cands] // uniform prior fallback
            };
            let max_p = priors_f64.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = priors_f64.iter().map(|&p| ((p - max_p) / tau.max(0.5)).exp()).collect();
            let sum_exp: f64 = exps.iter().sum();
            let p_action: Vec<f64> = exps.iter().map(|e| e / sum_exp.max(1e-9)).collect();
            // Seed each candidate with 1 rollout so empirical means exist.
            let mut seed_work = Vec::with_capacity(n_cands);
            for ci in 0..n_cands {
                seed_work.push((ci, rng.gen()));
            }
            run_work(seed_work, &mut totals, &mut sumsq, &mut counts);
            let mut spent = n_cands;
            while spent < num_rollouts {
                let total_visits: u32 = counts.iter().sum();
                let n_total_sqrt = (total_visits.max(1) as f64).sqrt();
                let mut best_ucb = f64::NEG_INFINITY;
                let mut best_ci = 0usize;
                for ci in 0..n_cands {
                    let n = counts[ci].max(1) as f64;
                    let mean = totals[ci] as f64 / n;
                    let q = mean / 100.0;
                    let u = c_puct * p_action[ci] * n_total_sqrt / (1.0 + n);
                    let score = q + u;
                    if score > best_ucb { best_ucb = score; best_ci = ci; }
                }
                let actual_batch = batch.min(num_rollouts - spent);
                let mut work = Vec::with_capacity(actual_batch);
                for _ in 0..actual_batch { work.push((best_ci, rng.gen())); }
                run_work(work, &mut totals, &mut sumsq, &mut counts);
                spent += actual_batch;
            }
        }
    }

    // Final decision: optionally blend rollout mean with NNUE prior (control variate / shrinkage).
    // If Gumbel halving was active, add the same frozen Gumbel prior to the final scoring
    // for Danihelka 2022 consistency (policy improvement guarantee).
    // If MCE_CONTROL_VARIATES is set, apply per-candidate variance-reduction adjustment.
    // CORRECTED FORM: subtract β_i · (mean_B_i − prior_i) where prior_i is the
    // candidate's afterstate NNUE eval (a deterministic, candidate-specific quantity
    // approximating E[B|state_i]). This removes the bag-noise-induced deviation of
    // mid-rollout B from its expected value, without biasing the comparison across
    // candidates (which the prior subtraction does NOT do — earlier `global_mean_B`
    // version regressed by 17 base points). Falls back to no adjustment if priors empty.
    // MCE_CV_BETA_CAP: clip estimated β to ±cap (default 2.0) for numerical stability.
    let cv_beta_cap: f64 = std::env::var("MCE_CV_BETA_CAP").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(2.0);
    let cvt = cv_totals.borrow();
    let cvs = cv_sumsq.borrow();
    let crs = cross.borrow();
    let mut best_idx = 0;
    let mut best_adj = f64::NEG_INFINITY;
    for (ci, (&t, &n)) in totals.iter().zip(counts.iter()).enumerate() {
        if n == 0 { continue; }
        let nf = n as f64;
        let rollout_mean = t as f64 / nf;
        let cv_adjusted_mean = if use_cv && n >= 4 && !priors.is_empty() {
            let mean_b = cvt[ci] as f64 / nf;
            let var_b = (cvs[ci] as f64 / nf - mean_b * mean_b).max(1.0);
            let cov_rb = crs[ci] as f64 / nf - rollout_mean * mean_b;
            let beta = (cov_rb / var_b).clamp(-cv_beta_cap, cv_beta_cap);
            // Use per-candidate prior (afterstate NNUE eval) as the structural baseline
            // for E[B|state_i]. mean_b - prior_i is the bag-noise-induced deviation.
            let prior_i = priors[ci] as f64;
            rollout_mean - beta * (mean_b - prior_i)
        } else {
            rollout_mean
        };
        let prior = if !priors.is_empty() { priors[ci] as f64 } else { cv_adjusted_mean };
        let mut adj = (cv_alpha as f64) * cv_adjusted_mean + ((1.0 - cv_alpha) as f64) * prior;
        if use_gumbel_halving && !gumbel_priors.is_empty() {
            adj += gumbel_sigma * gumbel_priors[ci];
        }
        if adj > best_adj { best_adj = adj; best_idx = ci; }
    }
    drop(cvt); drop(cvs); drop(crs);
    if best_adj < 0.0 { return None; }
    let mv = candidates[best_idx];
    let rollout_mean_best = if counts[best_idx] > 0 {
        totals[best_idx] as f64 / counts[best_idx] as f64
    } else { 0.0 };
    Some(ScoredMove { score: rollout_mean_best.round() as u16, ..mv })
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

/// Score ALL candidates with uniform MCE rollouts (no elimination).
/// Every candidate gets exactly `rollouts_per_cand` rollouts so we can
/// compare NNUE prefilter rankings against MCE ground-truth rankings.
///
/// Returns: Vec of (candidate, avg_mce_score) for ALL input candidates,
/// sorted by MCE score descending.
pub fn rank_all_candidates_mce(
    game: &GameState,
    net: &NNUENetwork,
    rollouts_per_cand: usize,
    candidates: &[ScoredMove],
    rng: &mut StdRng,
) -> Vec<(ScoredMove, MceStat)> {
    let player = game.current_player;
    if candidates.is_empty() { return Vec::new(); }

    let n_cands = candidates.len();
    let game_arc = Arc::new(game.clone());
    let net_arc = Arc::new(net.clone());
    let candidates_arc = Arc::new(candidates.to_vec());

    // Build uniform work: every candidate gets the same number of rollouts
    let mut work_items: Vec<(usize, u64)> = Vec::with_capacity(n_cands * rollouts_per_cand);
    for ci in 0..n_cands {
        for _ in 0..rollouts_per_cand {
            work_items.push((ci, rng.gen()));
        }
    }

    let num_threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let chunk_size = ((work_items.len() + num_threads - 1) / num_threads).max(1);

    let handles: Vec<_> = work_items.chunks(chunk_size).map(|chunk| {
        let work = chunk.to_vec();
        let game = Arc::clone(&game_arc);
        let net = Arc::clone(&net_arc);
        let cands = Arc::clone(&candidates_arc);

        thread::spawn(move || {
            let mut results: Vec<(usize, u64)> = Vec::with_capacity(work.len());
            for &(ci, seed) in &work {
                let mv = &cands[ci];
                let mut g = (*game).clone();
                let mut rollout_rng = StdRng::seed_from_u64(seed);
                g.shuffle_bags(&mut rollout_rng);
                if !execute_scored_move(&mut g, mv) { continue; }

                let depth_limit: usize = std::env::var("MCE_DEPTH").ok()
                    .and_then(|s| s.parse().ok()).unwrap_or(6);
                let mut ai_turns = 0;
                while !g.is_game_over() {
                    if g.current_player != player {
                        if g.can_replace_overflow().is_some() {
                            g.replace_overflow();
                        }
                        match greedy_move(&g) {
                            Some(opp) => {
                                if !execute_scored_move(&mut g, &opp) { break; }
                            }
                            None => break,
                        }
                        continue;
                    }
                    ai_turns += 1;
                    if ai_turns > depth_limit { break; }
                    if g.can_replace_overflow().is_some() {
                        g.replace_overflow();
                    }
                    match greedy_move(&g) {
                        Some(ai_mv) => {
                            if !execute_scored_move(&mut g, &ai_mv) { break; }
                        }
                        None => break,
                    }
                }

                let score = if g.is_game_over() {
                    ScoreBreakdown::compute(&mut g.boards[player], &g.scoring_cards).total
                } else {
                    let actual = ScoreBreakdown::compute(
                        &mut g.boards[player], &g.scoring_cards).total;
                    let bag_info = crate::nnue::BagInfo::from_game(&g);
                    let nval = net.evaluate_with_bag(&g.boards[player], &bag_info);
                    let tier = tier_bonus(&g.boards[player]);
                    (actual as f32 + nval.max(0.0) + tier) as u16
                };
                results.push((ci, score as u64));
            }
            results
        })
    }).collect();

    let mut totals = vec![0u64; n_cands];
    let mut sumsq = vec![0u64; n_cands];
    let mut counts = vec![0u32; n_cands];
    let mut all_scores: Vec<Vec<u64>> = (0..n_cands).map(|_| Vec::new()).collect();
    for handle in handles {
        for (ci, score) in handle.join().unwrap() {
            totals[ci] += score;
            sumsq[ci] += score * score;
            counts[ci] += 1;
            all_scores[ci].push(score);
        }
    }

    // Return (move, stats) sorted by mean descending
    let mut scored: Vec<(ScoredMove, MceStat)> = candidates.iter().enumerate()
        .filter_map(|(ci, mv)| {
            if counts[ci] == 0 { return None; }
            let n = counts[ci] as f64;
            let mean = totals[ci] as f64 / n;
            let var = (sumsq[ci] as f64 / n - mean * mean).max(0.0);
            let std = var.sqrt();
            let mut s = all_scores[ci].clone();
            s.sort();
            let min = *s.first().unwrap_or(&0);
            let max = *s.last().unwrap_or(&0);
            let median = if s.len() % 2 == 0 && s.len() >= 2 {
                (s[s.len()/2 - 1] + s[s.len()/2]) as f64 / 2.0
            } else if !s.is_empty() { s[s.len()/2] as f64 } else { 0.0 };
            Some((*mv, MceStat { mean, std, min, max, median }))
        }).collect();
    scored.sort_by(|a, b| b.1.mean.partial_cmp(&a.1.mean).unwrap_or(std::cmp::Ordering::Equal));
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
