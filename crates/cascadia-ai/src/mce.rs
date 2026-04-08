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

    let mut candidates = crate::search::candidate_moves_nnue(game, net);
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

    // Re-rank with NNUE afterstate evaluation for better initial ordering
    let bag_info = crate::nnue::BagInfo::from_game(game);
    for mv in candidates.iter_mut() {
        let coord = cascadia_core::hex::HexCoord::new(mv.tile_q, mv.tile_r);
        let tile = mp.iter().find(|&&(i, _, _)| i == mv.market_index).map(|&(_, t, _)| t);
        let wildlife = mp.iter().find(|&&(i, _, _)| {
            i == mv.wildlife_market_index.unwrap_or(mv.market_index)
        }).map(|&(_, _, w)| w);
        if let Some(tile) = tile {
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

    // Sort candidates + sources together by NNUE-based eval
    let mut indexed: Vec<(usize, i32)> = candidates.iter().enumerate().map(|(i, c)| (i, c.eval)).collect();
    indexed.sort_by(|a, b| b.1.cmp(&a.1));
    let sorted_candidates: Vec<ScoredMove> = indexed.iter().map(|&(i, _)| candidates[i]).collect();
    let sorted_sources: Vec<CandidateSource> = indexed.iter().map(|&(i, _)| sources[i]).collect();
    candidates = sorted_candidates;
    sources = sorted_sources;

    // Keep more candidates now that we have diverse sources
    candidates.truncate(15);
    sources.truncate(15);

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

                        let depth_limit = 6;
                        let mut ai_turns_played = 0;

                        while !g.is_game_over() {
                            if g.current_player != player {
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

                            if g.can_replace_overflow().is_some() {
                                let baseline_mv = pick_best_move_nnue(&g, &net);
                                let baseline_score = baseline_mv.as_ref().map(|m| m.score).unwrap_or(0);
                                let mut test = g.clone();
                                test.replace_overflow();
                                let new_mv = pick_best_move_nnue(&test, &net);
                                let new_score = new_mv.as_ref().map(|m| m.score).unwrap_or(0);
                                if new_score > baseline_score {
                                    g.replace_overflow();
                                }
                            }

                            match pick_best_move_nnue(&g, &net) {
                                Some(ai_mv) => {
                                    if !execute_scored_move(&mut g, &ai_mv) { break; }
                                }
                                None => break,
                            }
                        }

                        let score = if g.is_game_over() {
                            ScoreBreakdown::compute(&mut g.boards[player], &g.scoring_cards).total
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
