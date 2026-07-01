//! Microbench: time the individual operations on the MCE hot path so we can
//! prioritize optimizations honestly. Reports nanoseconds-per-call averages
//! over warmup + measured iterations.

use std::path::Path;
use std::time::Instant;

use rand::{rngs::StdRng, SeedableRng};

use cascadia_ai::nnue::{extract_features, extract_features_with_bag, BagInfo, NNUENetwork};
use cascadia_ai::search::{candidate_moves_pub, execute_scored_move, greedy_move};
use cascadia_core::game::GameState;
use cascadia_core::scoring::ScoreBreakdown;
use cascadia_core::types::ScoringCards;

fn main() {
    let weights = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "nnue_weights_v4opp_modal_iter3.bin".to_string());
    let net = NNUENetwork::load(Path::new(&weights)).expect("load nnue");

    // Set up a realistic mid-game position by playing forward greedy.
    let mut rng = StdRng::seed_from_u64(0xBEAF);
    let mut g = GameState::new(4, ScoringCards::all_a(), &mut rng);
    for _ in 0..40 {
        if g.is_game_over() {
            break;
        }
        if let Some(mv) = greedy_move(&g) {
            execute_scored_move(&mut g, &mv);
        } else {
            break;
        }
    }
    let board = g.boards[g.current_player].clone();
    let bag = BagInfo::from_game_for_player(&g, g.current_player);

    // === Bench 1: extract_features (no bag) ===
    let n = 5_000;
    let _ = extract_features(&board);
    let t0 = Instant::now();
    for _ in 0..n {
        std::hint::black_box(extract_features(&board));
    }
    let extract_simple_us = t0.elapsed().as_nanos() as f64 / n as f64 / 1000.0;

    // === Bench 2: extract_features_with_bag ===
    let _ = extract_features_with_bag(&board, Some(&bag));
    let t0 = Instant::now();
    for _ in 0..n {
        std::hint::black_box(extract_features_with_bag(&board, Some(&bag)));
    }
    let extract_bag_us = t0.elapsed().as_nanos() as f64 / n as f64 / 1000.0;

    // === Bench 3: NNUE.forward over pre-extracted features ===
    let features = extract_features_with_bag(&board, Some(&bag));
    let _ = net.forward(&features);
    let nfwd = 10_000;
    let t0 = Instant::now();
    for _ in 0..nfwd {
        std::hint::black_box(net.forward(&features));
    }
    let forward_us = t0.elapsed().as_nanos() as f64 / nfwd as f64 / 1000.0;

    // === Bench 4: combined evaluate_with_bag (extract + forward) ===
    let _ = net.evaluate_with_bag(&board, &bag);
    let t0 = Instant::now();
    for _ in 0..n {
        std::hint::black_box(net.evaluate_with_bag(&board, &bag));
    }
    let evaluate_us = t0.elapsed().as_nanos() as f64 / n as f64 / 1000.0;

    // === Bench 5: GameState clone ===
    let nclone = 50_000;
    let t0 = Instant::now();
    for _ in 0..nclone {
        std::hint::black_box(g.clone());
    }
    let clone_us = t0.elapsed().as_nanos() as f64 / nclone as f64 / 1000.0;

    // === Bench 6: candidate_moves_pub ===
    let _ = candidate_moves_pub(&g);
    let ncm = 2_000;
    let t0 = Instant::now();
    for _ in 0..ncm {
        std::hint::black_box(candidate_moves_pub(&g));
    }
    let cands_us = t0.elapsed().as_nanos() as f64 / ncm as f64 / 1000.0;

    // === Bench 7: ScoreBreakdown::compute ===
    let nsb = 20_000;
    let t0 = Instant::now();
    for _ in 0..nsb {
        let mut b = board.clone();
        std::hint::black_box(ScoreBreakdown::compute(&mut b, &g.scoring_cards));
    }
    let score_compute_us = t0.elapsed().as_nanos() as f64 / nsb as f64 / 1000.0;

    // === Bench 8: ScoreBreakdown::compute_with_bonuses ===
    let t0 = Instant::now();
    for _ in 0..nsb {
        let mut bs = g.boards.clone();
        std::hint::black_box(ScoreBreakdown::compute_with_bonuses(
            &mut bs,
            &g.scoring_cards,
            g.current_player,
        ));
    }
    let score_bonus_us = t0.elapsed().as_nanos() as f64 / nsb as f64 / 1000.0;

    // === Bench 9: pick_best_move_nnue (a whole single-ply NNUE-greedy pick) ===
    use cascadia_ai::nnue_train::pick_best_move_nnue;
    let _ = pick_best_move_nnue(&g, &net);
    let npicks = 500;
    let t0 = Instant::now();
    for _ in 0..npicks {
        std::hint::black_box(pick_best_move_nnue(&g, &net));
    }
    let pick_best_us = t0.elapsed().as_nanos() as f64 / npicks as f64 / 1000.0;

    println!("\n=== MCE hot-path microbench (mid-game position) ===");
    println!(
        "{:35} {:>10.2} µs",
        "extract_features (no bag)", extract_simple_us
    );
    println!(
        "{:35} {:>10.2} µs",
        "extract_features_with_bag", extract_bag_us
    );
    println!(
        "{:35} {:>10.2} µs",
        "NNUE.forward (pre-extracted)", forward_us
    );
    println!(
        "{:35} {:>10.2} µs",
        "net.evaluate_with_bag (ext+fwd)", evaluate_us
    );
    println!(
        "{:35} {:>10.2} µs",
        "ScoreBreakdown::compute", score_compute_us
    );
    println!(
        "{:35} {:>10.2} µs",
        "ScoreBreakdown::compute_with_bonuses", score_bonus_us
    );
    println!("{:35} {:>10.2} µs", "GameState.clone()", clone_us);
    println!("{:35} {:>10.2} µs", "candidate_moves_pub", cands_us);
    println!(
        "{:35} {:>10.2} µs",
        "pick_best_move_nnue (full ply)", pick_best_us
    );

    let candidates = candidate_moves_pub(&g);
    println!("\n# candidates at this position: {}", candidates.len());
    println!("# features extracted: {}", features.len());
}
