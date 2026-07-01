//! Pairwise-discrimination training data collection for HybridNet's Δ.
//!
//! For each decision the player faces during a champion-strength game,
//! records ALL candidate moves with their MCE-estimated final scores. The
//! Python trainer then samples pairs within each decision and learns Δ on
//! the pairwise residual difference — which cancels the constant +7 bias
//! and forces Δ to learn discrimination signal only.
//!
//! Usage:
//!   ./target/release/examples/collect_hybrid_pairwise \
//!     --weights nnue_weights_v4opp_modal_iter3.bin \
//!     --rollouts 100 --games 50 --out /tmp/hybp_v1.hybp

use std::path::Path;
use std::sync::{Arc, Mutex};

use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;

use cascadia_ai::hybrid::{append_hybrid_pairwise, encode_board_compact, HybridPairwiseDecision};
use cascadia_ai::mce::{
    best_move_nnue_rollout_mce, default_greedy_mce_candidates, expanded_candidates,
    nnue_prefilter_candidates, run_mce_candidates, GreedyMceAlloc,
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
    let games: usize = parse_arg(&args, "--games").unwrap_or(20);
    let rollouts: usize = parse_arg(&args, "--rollouts").unwrap_or(60);
    let prefilter_k: usize = parse_arg(&args, "--prefilter-k").unwrap_or(8);
    let seed: u64 = parse_arg(&args, "--seed").unwrap_or(0xC0DE);
    let out = parse_str(&args, "--out").unwrap_or_else(|| "/tmp/hybp_v1.hybp".to_string());

    eprintln!(
        "collect_hybrid_pairwise: weights={} games={} rollouts={} prefilter_k={} out={}",
        weights, games, rollouts, prefilter_k, out
    );

    let net = NNUENetwork::load(Path::new(&weights)).unwrap_or_else(|e| {
        eprintln!("Failed to load NNUE: {}", e);
        std::process::exit(1);
    });
    let net = Arc::new(net);
    let out_path = Path::new(&out);
    let _ = std::fs::remove_file(out_path);

    let writer = Mutex::new(());
    let total_decisions = std::sync::atomic::AtomicUsize::new(0);
    let total_candidates = std::sync::atomic::AtomicUsize::new(0);
    let completed = std::sync::atomic::AtomicUsize::new(0);

    let game_indices: Vec<usize> = (0..games).collect();
    game_indices.into_par_iter().for_each(|gi| {
        let net = Arc::clone(&net);
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(gi as u64));
        let mut g = GameState::new(4, ScoringCards::all_a(), &mut rng);

        let mut local_decisions: Vec<HybridPairwiseDecision> = Vec::new();
        let mut steps = 0usize;

        while !g.is_game_over() {
            let player = g.current_player;
            // Run MCE per-candidate to get values. `run_mce_candidates` uses
            // the same expansion + prefilter pipeline as best_move_nnue_rollout_mce
            // does internally (with prefilter_k baked in at top-K=15). Records
            // (move, mean MCE value) for every candidate that was evaluated.
            let cand_values = run_mce_candidates(&g, &net, rollouts, &mut rng);

            // Build the decision record: each candidate gets its post-move
            // board encoding, NNUE prediction of the resulting afterstate,
            // and the MCE-estimated final score (which serves as the
            // pairwise-residual reference).
            let mut cands_record: Vec<(Vec<f32>, f32, f32)> = Vec::with_capacity(cand_values.len());
            for (mv, mce_val) in &cand_values {
                let mut g2 = g.clone();
                if !execute_scored_move(&mut g2, mv) {
                    continue;
                }
                let board = &g2.boards[player];
                let board_enc = encode_board_compact(board);
                let bag_info = BagInfo::from_game_for_player(&g2, player);
                let features = extract_features_with_bag(board, Some(&bag_info));
                let nnue_pred = net.forward(&features);
                cands_record.push((board_enc, nnue_pred, *mce_val as f32));
            }

            if cands_record.len() >= 2 {
                local_decisions.push(HybridPairwiseDecision {
                    candidates: cands_record,
                });
            }

            // Pick the best move from the MCE results and advance.
            let best_mv = cand_values
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(m, _)| *m);
            let mv = best_mv.or_else(|| {
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
            steps += 1;
            if steps > 800 {
                break;
            }
        }

        let n_dec = local_decisions.len();
        let n_cand: usize = local_decisions.iter().map(|d| d.candidates.len()).sum();

        {
            let _guard = writer.lock().unwrap();
            append_hybrid_pairwise(out_path, &local_decisions).unwrap_or_else(|e| {
                eprintln!("Write failed: {}", e);
                std::process::exit(1);
            });
        }
        total_decisions.fetch_add(n_dec, std::sync::atomic::Ordering::Relaxed);
        total_candidates.fetch_add(n_cand, std::sync::atomic::Ordering::Relaxed);
        let c = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;

        let mut final_scores = [0.0f32; 4];
        for p in 0..4 {
            let mut boards = g.boards.clone();
            let bd = ScoreBreakdown::compute_with_bonuses(&mut boards, &g.scoring_cards, p);
            final_scores[p] = bd.total as f32;
        }
        eprintln!(
            "  [{:3}/{:3}] game {:3}: decisions={:3} candidates={:4} finals={:?}",
            c, games, gi, n_dec, n_cand, final_scores
        );
    });

    eprintln!(
        "Wrote {} decisions, {} total candidates to {} ({:.1} candidates/decision avg)",
        total_decisions.load(std::sync::atomic::Ordering::Relaxed),
        total_candidates.load(std::sync::atomic::Ordering::Relaxed),
        out,
        total_candidates.load(std::sync::atomic::Ordering::Relaxed) as f64
            / total_decisions
                .load(std::sync::atomic::Ordering::Relaxed)
                .max(1) as f64,
    );

    // Silence unused-import warnings if any.
    let _ = (
        expanded_candidates,
        nnue_prefilter_candidates,
        best_move_nnue_rollout_mce,
        GreedyMceAlloc::SeqHalving,
    );
}
