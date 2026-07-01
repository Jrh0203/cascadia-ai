//! Micro-benchmark to identify actual performance hotspots.
//! Compile with: cargo run --release --bin profile_hotspots
//! (Not wired into Cargo.toml by default — see overnight/profile.sh)
//!
//! Measures:
//! - Time per greedy_move
//! - Time per pick_best_move_nnue
//! - Time per full MCE rollout
//! - Time per candidate_moves_pub
//! - Time per execute_scored_move
//! - Time per ScoreBreakdown::compute

use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

use cascadia_ai::nnue::NNUENetwork;
use cascadia_core::game::GameState;
use cascadia_core::types::ScoringCards;

fn main() {
    // Build a mid-game state
    let cards = ScoringCards::all_a();
    let mut rng = StdRng::seed_from_u64(42);
    let mut game = GameState::new(4, cards, &mut rng);

    // Advance to turn 10 so we have a realistic mid-game state
    for _ in 0..40 {
        if game.is_game_over() {
            break;
        }
        if let Some(mv) = cascadia_ai::search::greedy_move(&game) {
            cascadia_ai::search::execute_scored_move(&mut game, &mv);
        } else {
            break;
        }
    }

    let net = NNUENetwork::load(std::path::Path::new("nnue_weights_v9_iter14.bin"))
        .expect("load weights");

    println!("Profiling hotspots on mid-game state...");
    println!("  Player: {}", game.current_player);
    println!("  Turns remaining: {}", game.turns_remaining);
    println!();

    // 1. greedy_move timing
    let n = 1000;
    let t = Instant::now();
    for _ in 0..n {
        std::hint::black_box(cascadia_ai::search::greedy_move(&game));
    }
    let per = t.elapsed().as_micros() as f64 / n as f64;
    println!(
        "  greedy_move:            {:>7.1} μs/call ({} calls)",
        per, n
    );

    // 2. candidate_moves_pub timing
    let n = 1000;
    let t = Instant::now();
    for _ in 0..n {
        std::hint::black_box(cascadia_ai::search::candidate_moves_pub(&game));
    }
    let per = t.elapsed().as_micros() as f64 / n as f64;
    println!(
        "  candidate_moves_pub:    {:>7.1} μs/call ({} calls)",
        per, n
    );

    // 3. pick_best_move_nnue timing
    let n = 200;
    let t = Instant::now();
    for _ in 0..n {
        std::hint::black_box(cascadia_ai::nnue_train::pick_best_move_nnue(&game, &net));
    }
    let per = t.elapsed().as_micros() as f64 / n as f64;
    println!(
        "  pick_best_move_nnue:    {:>7.1} μs/call ({} calls)",
        per, n
    );

    // 4. NNUE forward with bag
    let bag_info = cascadia_ai::nnue::BagInfo::from_game(&game);
    let board = &game.boards[game.current_player];
    let n = 5000;
    let t = Instant::now();
    for _ in 0..n {
        std::hint::black_box(net.evaluate_with_bag(board, &bag_info));
    }
    let per = t.elapsed().as_micros() as f64 / n as f64;
    println!(
        "  NNUE forward (w/ bag):  {:>7.1} μs/call ({} calls)",
        per, n
    );

    // 5. ScoreBreakdown::compute
    let n = 2000;
    let t = Instant::now();
    for _ in 0..n {
        let mut b = board.clone();
        std::hint::black_box(cascadia_core::scoring::ScoreBreakdown::compute(
            &mut b, &cards,
        ));
    }
    let per = t.elapsed().as_micros() as f64 / n as f64;
    println!(
        "  ScoreBreakdown::compute:{:>7.1} μs/call ({} calls)",
        per, n
    );

    // 6. GameState clone
    let n = 2000;
    let t = Instant::now();
    for _ in 0..n {
        std::hint::black_box(game.clone());
    }
    let per = t.elapsed().as_micros() as f64 / n as f64;
    println!(
        "  GameState clone:        {:>7.1} μs/call ({} calls)",
        per, n
    );

    // 7. Board clone
    let n = 5000;
    let t = Instant::now();
    for _ in 0..n {
        std::hint::black_box(board.clone());
    }
    let per = t.elapsed().as_micros() as f64 / n as f64;
    println!(
        "  Board clone:            {:>7.1} μs/call ({} calls)",
        per, n
    );
}
