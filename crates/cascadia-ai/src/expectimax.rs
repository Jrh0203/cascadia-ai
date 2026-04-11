//! Expectimax search with market uncertainty.
//!
//! For each candidate move:
//! 1. Execute the move
//! 2. Sample K possible futures (opponents play, market refills from shuffled bag)
//! 3. For each future, find the best next move via NNUE and evaluate the resulting afterstate
//! 4. Average: expected_value = current_score + avg(next_afterstate_value)
//!
//! This is a 2-ply lookahead with chance nodes. Unlike MCE which does random
//! rollouts many turns deep, expectimax looks shallow but evaluates precisely
//! using the NNUE at leaf states.

use std::sync::Arc;
use std::thread;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use cascadia_core::board::Board;
use cascadia_core::game::GameState;
use cascadia_core::hex::ADJACENCY;
use cascadia_core::scoring::ScoreBreakdown;
use cascadia_core::types::Wildlife;

use crate::eval::{best_move_with_potential, ScoredMove};
use crate::nnue::NNUENetwork;
use crate::nnue_train::pick_best_move_nnue;
use crate::search::{candidate_moves_pub, execute_scored_move, greedy_move};
use crate::wildlife_candidates::wildlife_strategic_candidates;

/// Pick the best move using 2-ply expectimax with chance sampling.
/// `num_samples`: number of chance outcomes to sample per candidate (e.g., 20).
pub fn best_move_expectimax(
    game: &GameState,
    net: &NNUENetwork,
    num_samples: usize,
    rng: &mut StdRng,
) -> Option<ScoredMove> {
    let player = game.current_player;

    // Gather candidates (greedy + strategic + candidate_moves)
    let mp: Vec<_> = game.market.available()
        .map(|(i, p)| (i, p.tile, p.wildlife)).collect();
    if mp.is_empty() { return None; }

    let cards = game.scoring_cards;
    let turns = game.turns_remaining;
    let mut board = game.boards[player].clone();
    let greedy_best = best_move_with_potential(&mut board, &mp, &cards, turns);

    let mut candidates: Vec<ScoredMove> = candidate_moves_pub(game);
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
    candidates.truncate(15);

    if candidates.is_empty() {
        return greedy_best;
    }

    // Build work items: (candidate_index, sample_seed)
    let mut work_items: Vec<(usize, u64)> = Vec::new();
    for ci in 0..candidates.len() {
        for _ in 0..num_samples {
            work_items.push((ci, rng.gen()));
        }
    }

    let num_threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let chunk_size = (work_items.len() + num_threads - 1) / num_threads;

    let game_arc = Arc::new(game.clone());
    let net_arc = Arc::new(net.clone());
    let candidates_arc = Arc::new(candidates.clone());

    let handles: Vec<_> = work_items
        .chunks(chunk_size)
        .map(|chunk| {
            let work = chunk.to_vec();
            let game = Arc::clone(&game_arc);
            let net = Arc::clone(&net_arc);
            let cands = Arc::clone(&candidates_arc);

            thread::spawn(move || {
                let mut results: Vec<(usize, f32)> = Vec::with_capacity(work.len());

                for &(ci, seed) in &work {
                    let mv = &cands[ci];
                    let mut g = (*game).clone();
                    let mut sample_rng = StdRng::seed_from_u64(seed);

                    if !execute_scored_move(&mut g, mv) { continue; }

                    // Chance node: shuffle bag for random refill, advance opponents
                    g.shuffle_bags(&mut sample_rng);
                    if g.num_players > 1 {
                        advance_opponents(&mut g, player);
                    }

                    // Ply 2: pick best next move for our player and evaluate its afterstate
                    let value = if g.is_game_over() {
                        leaf_value(&mut g, &net, player)
                    } else if g.current_player == player {
                        match pick_best_move_nnue(&g, &net) {
                            Some(next_mv) => {
                                let mut g2 = g.clone();
                                if execute_scored_move(&mut g2, &next_mv) {
                                    leaf_value(&mut g2, &net, player)
                                } else {
                                    leaf_value(&mut g, &net, player)
                                }
                            }
                            None => leaf_value(&mut g, &net, player),
                        }
                    } else {
                        leaf_value(&mut g, &net, player)
                    };

                    results.push((ci, value));
                }

                results
            })
        })
        .collect();

    let mut totals = vec![0.0f32; candidates.len()];
    let mut counts = vec![0u32; candidates.len()];

    for handle in handles {
        for (ci, value) in handle.join().unwrap() {
            totals[ci] += value;
            counts[ci] += 1;
        }
    }

    let mut best: Option<(ScoredMove, f32)> = None;
    for (ci, mv) in candidates.iter().enumerate() {
        if counts[ci] == 0 { continue; }
        let avg = totals[ci] / counts[ci] as f32;
        if best.is_none() || avg > best.as_ref().unwrap().1 {
            best = Some((*mv, avg));
        }
    }

    best.map(|(mv, avg)| ScoredMove { score: avg.round() as u16, ..mv })
}

/// Leaf value: actual score + NNUE remaining value + bear half-pair bonus.
/// The bear bonus corrects for NNUE undervaluing half-pairs at deep search.
fn leaf_value(game: &mut GameState, net: &NNUENetwork, player: usize) -> f32 {
    let sc = ScoreBreakdown::compute(&mut game.boards[player], &game.scoring_cards).total as f32;
    let nval = net.evaluate(&game.boards[player]).max(0.0);
    let bear_bonus = bear_halfpair_bonus(&game.boards[player]);
    sc + nval + bear_bonus
}

/// Bear half-pair bonus: reward boards with isolated bears that could complete pairs.
/// The NNUE undervalues half-pairs, so deeper expectimax abandons bears. This fixes it.
fn bear_halfpair_bonus(board: &Board) -> f32 {
    let adj = &*ADJACENCY;
    let bear_positions = &board.wildlife_positions[Wildlife::Bear as usize];
    if bear_positions.is_empty() { return 0.0; }

    // Count current bear pairs (same algorithm as scoring)
    let mut visited = [false; 441];
    let mut pairs = 0usize;
    for &pos in bear_positions.iter() {
        let idx = pos as usize;
        if visited[idx] { continue; }
        let mut component_size = 0u16;
        let mut queue = arrayvec::ArrayVec::<u16, 24>::new();
        queue.push(pos);
        visited[idx] = true;
        while let Some(current) = queue.pop() {
            component_size += 1;
            for nidx in adj.neighbors_of(current as usize) {
                if !visited[nidx] && board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Bear) {
                    visited[nidx] = true;
                    queue.push(nidx as u16);
                }
            }
        }
        if component_size == 2 { pairs += 1; }
    }

    // Count half-pairs: isolated bears with adjacent bear-accepting slots
    let mut half_pairs = 0usize;
    for &pos in bear_positions.iter() {
        let bear_neighbors = adj.neighbors_of(pos as usize)
            .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Bear))
            .count();
        if bear_neighbors == 0 {
            let has_slot = adj.neighbors_of(pos as usize).any(|n| {
                let cell = board.grid.get(n);
                cell.is_present() && cell.can_place_wildlife(Wildlife::Bear)
            });
            if has_slot { half_pairs += 1; }
        }
    }

    // Marginal value of completing the next pair
    let marginal = match pairs {
        0 => 4.0,
        1 => 7.0,
        _ => 8.0,
    };
    // Cap at 2 half-pairs (realistic limit), discount by 0.5 for uncertainty
    let completable = half_pairs.min(2) as f32;
    completable * marginal * 0.5
}

fn advance_opponents(game: &mut GameState, ai_player: usize) {
    // Delegate to the canonical helper which applies the free 3-of-a-kind
    // overflow replacement before each opponent move. See search::advance_opponents.
    crate::search::advance_opponents(game, ai_player)
}

/// Deep expectimax search: look ahead `depth` AI turns with `branching` top
/// candidates at each ply. Uses `num_samples` chance outcomes at the root.
pub fn best_move_expectimax_deep(
    game: &GameState,
    net: &NNUENetwork,
    num_samples: usize,
    depth: usize,
    branching: usize,
    rng: &mut StdRng,
) -> Option<ScoredMove> {
    let player = game.current_player;

    // Gather candidates at the root
    let mp: Vec<_> = game.market.available()
        .map(|(i, p)| (i, p.tile, p.wildlife)).collect();
    if mp.is_empty() { return None; }

    let cards = game.scoring_cards;
    let turns = game.turns_remaining;
    let mut board = game.boards[player].clone();
    let greedy_best = best_move_with_potential(&mut board, &mp, &cards, turns);

    let mut candidates: Vec<ScoredMove> = candidate_moves_pub(game);
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
    candidates.truncate(branching);

    if candidates.is_empty() {
        return greedy_best;
    }

    // Build work items: (candidate_index, sample_seed)
    let mut work_items: Vec<(usize, u64)> = Vec::new();
    for ci in 0..candidates.len() {
        for _ in 0..num_samples {
            work_items.push((ci, rng.gen()));
        }
    }

    let num_threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let chunk_size = (work_items.len() + num_threads - 1) / num_threads;

    let game_arc = Arc::new(game.clone());
    let net_arc = Arc::new(net.clone());
    let candidates_arc = Arc::new(candidates.clone());

    let handles: Vec<_> = work_items
        .chunks(chunk_size)
        .map(|chunk| {
            let work = chunk.to_vec();
            let game = Arc::clone(&game_arc);
            let net = Arc::clone(&net_arc);
            let cands = Arc::clone(&candidates_arc);

            thread::spawn(move || {
                let mut results: Vec<(usize, f32)> = Vec::with_capacity(work.len());

                for &(ci, seed) in &work {
                    let mv = &cands[ci];
                    let mut g = (*game).clone();
                    let mut sample_rng = StdRng::seed_from_u64(seed);
                    g.shuffle_bags(&mut sample_rng);

                    if !execute_scored_move(&mut g, mv) { continue; }
                    if g.num_players > 1 {
                        advance_opponents(&mut g, player);
                    }

                    // Recurse for remaining depth - 1
                    let value = expectimax_value(&g, &net, depth - 1, branching.min(5), player);
                    results.push((ci, value));
                }

                results
            })
        })
        .collect();

    let mut totals = vec![0.0f32; candidates.len()];
    let mut counts = vec![0u32; candidates.len()];

    for handle in handles {
        for (ci, value) in handle.join().unwrap() {
            totals[ci] += value;
            counts[ci] += 1;
        }
    }

    let mut best: Option<(ScoredMove, f32)> = None;
    for (ci, mv) in candidates.iter().enumerate() {
        if counts[ci] == 0 { continue; }
        let avg = totals[ci] / counts[ci] as f32;
        if best.is_none() || avg > best.as_ref().unwrap().1 {
            best = Some((*mv, avg));
        }
    }

    best.map(|(mv, avg)| ScoredMove { score: avg.round() as u16, ..mv })
}

/// Recursive expectimax value: evaluates game state assuming greedy opponents
/// and `depth` remaining AI turns. At leaves, uses score + NNUE estimate.
/// No further sampling — uses the fixed bag state from the root sample.
fn expectimax_value(
    game: &GameState,
    net: &NNUENetwork,
    depth: usize,
    branching: usize,
    ai_player: usize,
) -> f32 {
    if game.is_game_over() || depth == 0 {
        let mut g = game.clone();
        return leaf_value(&mut g, net, ai_player);
    }

    // Advance to our next turn
    let mut g = game.clone();
    if g.current_player != ai_player {
        advance_opponents(&mut g, ai_player);
    }
    if g.is_game_over() {
        return leaf_value(&mut g, net, ai_player);
    }

    // Enumerate top candidates at this ply
    let mp: Vec<_> = g.market.available()
        .map(|(i, p)| (i, p.tile, p.wildlife)).collect();
    if mp.is_empty() {
        return leaf_value(&mut g, net, ai_player);
    }

    let cards = g.scoring_cards;
    let turns = g.turns_remaining;
    let mut board = g.boards[ai_player].clone();
    let greedy_best = best_move_with_potential(&mut board, &mp, &cards, turns);

    let mut candidates: Vec<ScoredMove> = candidate_moves_pub(&g);
    if let Some(ref bm) = greedy_best {
        if !candidates.iter().any(|c| c.tile_q == bm.tile_q && c.tile_r == bm.tile_r
            && c.rotation == bm.rotation && c.wildlife_q == bm.wildlife_q) {
            candidates.push(*bm);
        }
    }
    candidates.truncate(branching);

    if candidates.is_empty() {
        return leaf_value(&mut g, net, ai_player);
    }

    // Max over candidates
    let mut best = f32::NEG_INFINITY;
    for mv in &candidates {
        let mut g2 = g.clone();
        if !execute_scored_move(&mut g2, mv) { continue; }
        let val = expectimax_value(&g2, net, depth - 1, branching, ai_player);
        if val > best { best = val; }
    }

    if best == f32::NEG_INFINITY {
        leaf_value(&mut g, net, ai_player)
    } else {
        best
    }
}
