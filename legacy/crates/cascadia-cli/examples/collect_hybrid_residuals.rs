//! Collect residual training data for HybridNet's Δ head.
//!
//! Plays `--games` complete games using NNUE+MCE move selection at strength
//! `--rollouts`. At each player's afterstate (immediately after they place
//! their tile/wildlife, before opponents play), records:
//!   - encoded board (16-plane compact representation)
//!   - NNUE prediction for that afterstate
//!   - current player's score-with-bonus at that moment
//!
//! When the game ends, computes each recorded afterstate's true label as
//! `final_score - current_score_snapshot` (= remaining points actually
//! achieved). Writes records to a HYBR file.
//!
//! Usage:
//!   cargo run --release -p cascadia-cli --features ... \
//!     --example collect_hybrid_residuals -- \
//!     --weights nnue_weights_v4opp_modal_iter3.bin \
//!     --rollouts 50 --games 5 --out /tmp/hybr_smoke.hybr

use std::path::Path;
use std::sync::Mutex;

use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;

use cascadia_ai::hybrid::{append_hybrid_residuals, encode_board_compact, HybridResidualRecord};
use cascadia_ai::mce::{
    best_move_nnue_rollout_mce, default_greedy_mce_candidates, expanded_candidates,
    nnue_prefilter_candidates, GreedyMceAlloc,
};
use cascadia_ai::nnue::{extract_features_with_bag, BagInfo, NNUENetwork};
use cascadia_ai::search::{execute_scored_move, greedy_move};
use cascadia_core::game::GameState;
use cascadia_core::scoring::ScoreBreakdown;
use cascadia_core::types::ScoringCards;

fn parse_arg<T: std::str::FromStr>(args: &[String], key: &str) -> Option<T> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

fn parse_str(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let weights = parse_str(&args, "--weights")
        .unwrap_or_else(|| "nnue_weights_v4opp_modal_iter3.bin".to_string());
    let games: usize = parse_arg(&args, "--games").unwrap_or(5);
    let rollouts: usize = parse_arg(&args, "--rollouts").unwrap_or(50);
    let prefilter_k: usize = parse_arg(&args, "--prefilter-k").unwrap_or(8);
    let seed: u64 = parse_arg(&args, "--seed").unwrap_or(0xC0FFEE);
    let out = parse_str(&args, "--out").unwrap_or_else(|| "/tmp/hybr_smoke.hybr".to_string());

    eprintln!(
        "collect_hybrid_residuals: weights={} games={} rollouts={} prefilter_k={} out={}",
        weights, games, rollouts, prefilter_k, out
    );

    let net = NNUENetwork::load(Path::new(&weights)).unwrap_or_else(|e| {
        eprintln!("Failed to load NNUE weights at {}: {}", weights, e);
        std::process::exit(1);
    });

    let out_path = Path::new(&out);
    // Reset the output file if it already exists — append-only otherwise
    // accumulates across runs which is rarely what a smoke test wants.
    let _ = std::fs::remove_file(out_path);

    let net = std::sync::Arc::new(net);
    let total_records = std::sync::atomic::AtomicUsize::new(0);
    let total_finals_per_seat = Mutex::new([0.0f64; 4]);
    let total_completed = std::sync::atomic::AtomicUsize::new(0);
    let writer = Mutex::new(());

    // Each game gets a distinct seed; rayon distributes across cores.
    // Within-game work is serial (MCE handles its own thread pool
    // internally, so games-in-parallel + games-internally-parallel could
    // oversubscribe — we let rayon handle the outer loop and rely on the
    // per-MCE-call internal parallelism being inexpensive at our scale).
    let game_indices: Vec<usize> = (0..games).collect();
    game_indices.into_par_iter().for_each(|game_idx| {
        let net = std::sync::Arc::clone(&net);
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(game_idx as u64));
        let mut g = GameState::new(4, ScoringCards::all_a(), &mut rng);

        let mut afterstates: Vec<(usize, Vec<f32>, f32, f32)> = Vec::with_capacity(80);

        let mut turn = 0usize;
        while !g.is_game_over() {
            let player = g.current_player;
            let mut cands = expanded_candidates(&g);
            if cands.len() > prefilter_k {
                cands = nnue_prefilter_candidates(&g, &net, cands, prefilter_k);
            }
            let mv = best_move_nnue_rollout_mce(
                &g,
                &net,
                rollouts,
                GreedyMceAlloc::SeqHalving,
                cands,
                &mut rng,
            )
            .or_else(|| {
                let cands = default_greedy_mce_candidates(&g);
                if cands.is_empty() {
                    greedy_move(&g)
                } else {
                    Some(cands[0])
                }
            });
            let mv = match mv {
                Some(m) => m,
                None => break,
            };
            if !execute_scored_move(&mut g, &mv) {
                break;
            }

            let board = &g.boards[player];
            let board_enc = encode_board_compact(board);
            let bag_info = BagInfo::from_game_for_player(&g, player);
            let features = extract_features_with_bag(board, Some(&bag_info));
            let nnue_pred = net.forward(&features);
            let mut boards = g.boards.clone();
            let breakdown =
                ScoreBreakdown::compute_with_bonuses(&mut boards, &g.scoring_cards, player);
            let current_score = breakdown.total as f32;

            afterstates.push((player, board_enc, nnue_pred, current_score));

            turn += 1;
            if turn > 800 {
                eprintln!("WARN: turn cap reached, breaking");
                break;
            }
        }

        let mut final_scores = [0.0f32; 4];
        for p in 0..4 {
            let mut boards = g.boards.clone();
            let bd = ScoreBreakdown::compute_with_bonuses(&mut boards, &g.scoring_cards, p);
            final_scores[p] = bd.total as f32;
        }

        let mut records: Vec<HybridResidualRecord> = Vec::with_capacity(afterstates.len());
        for (player, board, nnue_pred, current) in afterstates {
            let label = final_scores[player] - current;
            records.push(HybridResidualRecord {
                board,
                nnue_pred,
                label,
            });
        }
        let n_rec = records.len();
        {
            let _guard = writer.lock().unwrap();
            append_hybrid_residuals(out_path, &records).unwrap_or_else(|e| {
                eprintln!("Write failed: {}", e);
                std::process::exit(1);
            });
        }
        total_records.fetch_add(n_rec, std::sync::atomic::Ordering::Relaxed);
        {
            let mut sums = total_finals_per_seat.lock().unwrap();
            for p in 0..4 {
                sums[p] += final_scores[p] as f64;
            }
        }
        let completed = total_completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        eprintln!(
            "  [{:3}/{:3}] game {:3}: records={:3}  finals={:?}",
            completed, games, game_idx, n_rec, final_scores
        );
    });

    let total_records = total_records.load(std::sync::atomic::Ordering::Relaxed);
    let mean_finals: Vec<f64> = total_finals_per_seat
        .lock()
        .unwrap()
        .iter()
        .map(|s| s / games as f64)
        .collect();
    eprintln!(
        "Wrote {} records to {} (mean final scores per seat = {:.2?})",
        total_records, out, mean_finals
    );
}
