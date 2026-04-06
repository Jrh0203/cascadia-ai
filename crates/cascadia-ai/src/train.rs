use std::sync::Arc;
use std::thread;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use cascadia_core::board::Board;
use cascadia_core::game::GameState;
use cascadia_core::hex::HexCoord;
use cascadia_core::scoring::ScoreBreakdown;
use cascadia_core::types::ScoringCards;

use crate::eval::ScoredMove;
use crate::ntuple::NTupleNetwork;
use crate::search::{execute_scored_move, greedy_move};

/// Train the N-tuple network using TD(0) afterstate learning.
/// Games are played in parallel batches — each batch uses a frozen snapshot
/// of the weights, collects TD updates, then merges them back.
pub fn train(
    net: &mut NTupleNetwork,
    num_games: usize,
    alpha: f32,
    seed: u64,
) -> TrainStats {
    let num_threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let batch_size = num_threads * 50; // 50 games per thread per batch

    let mut rng = StdRng::seed_from_u64(seed);
    let mut stats = TrainStats::default();
    let mut games_done = 0usize;

    while games_done < num_games {
        let this_batch = batch_size.min(num_games - games_done);
        let games_per_thread = (this_batch + num_threads - 1) / num_threads;

        // Snapshot current weights for all threads to read
        let net_snapshot = Arc::new(net.clone());

        // Generate seeds for each thread's games
        let thread_configs: Vec<(usize, Vec<u64>)> = (0..num_threads)
            .map(|t| {
                let n = if t < num_threads - 1 {
                    games_per_thread.min(this_batch.saturating_sub(t * games_per_thread))
                } else {
                    this_batch.saturating_sub(t * games_per_thread)
                };
                let seeds: Vec<u64> = (0..n).map(|_| rng.gen()).collect();
                (n, seeds)
            })
            .filter(|(n, _)| *n > 0)
            .collect();

        // Run games in parallel
        let handles: Vec<_> = thread_configs
            .into_iter()
            .map(|(_, seeds)| {
                let net_snap = Arc::clone(&net_snapshot);
                let alpha = alpha;
                thread::spawn(move || {
                    let mut local_deltas = NTupleNetwork::new();
                    let mut local_score: u64 = 0;
                    let mut local_count: usize = 0;

                    for game_seed in seeds {
                        let score = play_and_collect_updates(
                            &net_snap, &mut local_deltas, game_seed, alpha,
                        );
                        local_score += score as u64;
                        local_count += 1;
                    }

                    (local_deltas, local_score, local_count)
                })
            })
            .collect();

        // Collect results and merge deltas (scale by 1/num_threads to
        // prevent divergence from accumulated stale updates)
        let scale = 1.0 / handles.len() as f32;
        for handle in handles {
            let (mut deltas, score, count) = handle.join().unwrap();
            deltas.scale(scale);
            net.merge_from(&deltas);
            stats.total_score += score;
            stats.games += count;
        }

        games_done += this_batch;

        if games_done % 1000 < batch_size || games_done >= num_games {
            let avg = stats.total_score as f64 / stats.games as f64;
            eprint!("\r  {}/{} games, avg score: {:.1}    ", games_done, num_games, avg);
        }
    }
    eprintln!();

    stats
}

/// Play one game with greedy moves, compute TD updates, and accumulate
/// weight deltas into `deltas` (does NOT modify the main network).
fn play_and_collect_updates(
    net: &NTupleNetwork,
    deltas: &mut NTupleNetwork,
    seed: u64,
    alpha: f32,
) -> u16 {
    let mut rng = StdRng::seed_from_u64(seed);
    let cards = ScoringCards::all_a();
    let mut game = GameState::new(4, cards, &mut rng);

    struct Snapshot {
        board: Board,
        actual_score: u16,
        ntuple_value: f32,
    }
    let mut snapshots: Vec<Snapshot> = Vec::with_capacity(20);

    while !game.is_game_over() {
        if game.current_player != 0 {
            match greedy_move(&game) {
                Some(mv) => {
                    if !execute_scored_move(&mut game, &mv) { break; }
                }
                None => break,
            }
            continue;
        }

        // Greedy move for stable training data
        let mv = greedy_move(&game);
        match mv {
            Some(mv) => {
                if !execute_scored_move(&mut game, &mv) { break; }
            }
            None => break,
        }

        // Record afterstate
        let actual_score = ScoreBreakdown::compute(
            &mut game.boards[0], &game.scoring_cards,
        ).total;
        let ntuple_value = net.evaluate(&game.boards[0]);
        snapshots.push(Snapshot {
            board: game.boards[0].clone(),
            actual_score,
            ntuple_value,
        });
    }

    let final_score = ScoreBreakdown::compute(
        &mut game.boards[0], &game.scoring_cards,
    ).total;

    // TD(0) backward updates — write to deltas, not the main network
    let n = snapshots.len();
    if n > 0 {
        let delta = 0.0 - snapshots[n - 1].ntuple_value;
        deltas.update(&snapshots[n - 1].board, delta, alpha);

        for t in (0..n - 1).rev() {
            let reward = (snapshots[t + 1].actual_score as f32) - (snapshots[t].actual_score as f32);
            let target = reward + snapshots[t + 1].ntuple_value;
            let delta = target - snapshots[t].ntuple_value;
            deltas.update(&snapshots[t].board, delta, alpha);
        }
    }

    final_score
}

/// Pick the best move for the AI player using greedy score + N-tuple evaluation.
pub fn pick_best_move_ntuple(
    game: &GameState,
    net: &NTupleNetwork,
) -> Option<ScoredMove> {
    let mp: Vec<_> = game.market.available()
        .map(|(i, p)| (i, p.tile, p.wildlife)).collect();
    if mp.is_empty() { return None; }

    let board = &game.boards[game.current_player];
    let cards = &game.scoring_cards;
    let frontier = board.frontier();
    if frontier.is_empty() { return None; }

    let has_tokens = board.nature_tokens > 0;
    let base_wildlife: u16 = cascadia_core::scoring::wildlife::score_all_wildlife(board, cards)
        .iter().sum();

    struct Combo {
        tile_idx: usize,
        tile: cascadia_core::types::TileData,
        wildlife: cascadia_core::types::Wildlife,
        wl_market_idx: Option<usize>,
    }

    let mut combos = Vec::new();
    for &(idx, tile, wl) in &mp {
        combos.push(Combo { tile_idx: idx, tile, wildlife: wl, wl_market_idx: None });
    }
    if has_tokens {
        for &(ti, tile, _) in &mp {
            for &(wi, _, wl) in &mp {
                if ti != wi {
                    combos.push(Combo { tile_idx: ti, tile, wildlife: wl, wl_market_idx: Some(wi) });
                }
            }
        }
    }

    let mut board_clone = board.clone();
    let mut best: Option<(ScoredMove, f32)> = None;

    for combo in &combos {
        let is_independent = combo.wl_market_idx.is_some();
        let effective_nature = if is_independent {
            (board.nature_tokens as u16).saturating_sub(1)
        } else {
            board.nature_tokens as u16
        };
        let max_rot: u8 = if combo.tile.terrain2.is_none() { 1 } else { 6 };

        for &fi in frontier.iter() {
            let coord = HexCoord::from_index(fi as usize);
            for rot in 0..max_rot {
                let tile_action = match board_clone.place_tile(coord, combo.tile, rot) {
                    Some(a) => a,
                    None => continue,
                };

                let hab: u16 = board_clone.largest_group.iter().sum();
                let variant = cards.variant_for(combo.wildlife);
                let without = cascadia_core::scoring::wildlife::score_wildlife(
                    &board_clone, combo.wildlife, variant,
                );

                // Try without wildlife placement
                let skip_score = hab + base_wildlife + effective_nature;
                let skip_nval = net.evaluate(&board_clone);
                let skip_total = skip_score as f32 * 1000.0 + skip_nval;

                if best.is_none() || skip_total > best.as_ref().unwrap().1 {
                    best = Some((ScoredMove {
                        market_index: combo.tile_idx,
                        tile_q: coord.q,
                        tile_r: coord.r,
                        rotation: rot,
                        wildlife_q: None,
                        wildlife_r: None,
                        score: skip_score,
                        eval: skip_total as i32,
                        wildlife_market_index: combo.wl_market_idx,
                    }, skip_total));
                }

                // Try wildlife at each valid position
                let placed_snapshot: arrayvec::ArrayVec<u16, 64> =
                    board_clone.placed_tiles.iter().copied().collect();
                for &ti in placed_snapshot.iter() {
                    if !board_clone.grid.get(ti as usize).can_place_wildlife(combo.wildlife) {
                        continue;
                    }
                    let wa = match board_clone.place_wildlife(ti as usize, combo.wildlife) {
                        Some(a) => a,
                        None => continue,
                    };

                    let with = cascadia_core::scoring::wildlife::score_wildlife(
                        &board_clone, combo.wildlife, variant,
                    );
                    let delta = with.saturating_sub(without);
                    let nat_bonus: u16 = if board_clone.grid.get(ti as usize).is_keystone() { 1 } else { 0 };
                    let score = hab + base_wildlife + delta + effective_nature + nat_bonus;
                    let nval = net.evaluate(&board_clone);
                    let total = score as f32 * 1000.0 + nval;

                    board_clone.undo(wa);

                    if best.is_none() || total > best.as_ref().unwrap().1 {
                        let wc = HexCoord::from_index(ti as usize);
                        best = Some((ScoredMove {
                            market_index: combo.tile_idx,
                            tile_q: coord.q,
                            tile_r: coord.r,
                            rotation: rot,
                            wildlife_q: Some(wc.q),
                            wildlife_r: Some(wc.r),
                            score,
                            eval: total as i32,
                            wildlife_market_index: combo.wl_market_idx,
                        }, total));
                    }
                }

                board_clone.undo(tile_action);
            }
        }
    }

    best.map(|(mv, _)| mv)
}

#[derive(Default)]
pub struct TrainStats {
    pub total_score: u64,
    pub games: usize,
}
