use cascadia_ai::hybrid::{encode_board_compact, DeltaNet};
use cascadia_core::{game::GameState, types::ScoringCards};
use rand::{rngs::StdRng, SeedableRng};
use std::time::Instant;

fn main() {
    let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
    let g = GameState::new(4, ScoringCards::all_a(), &mut rng);
    let board = &g.boards[0];
    let net = DeltaNet::new(0xC0FFEE);
    // Warm up
    let input = encode_board_compact(board);
    let _ = net.forward(&input);

    let n_enc = 5000;
    let t0 = Instant::now();
    for _ in 0..n_enc {
        std::hint::black_box(encode_board_compact(board));
    }
    let enc_us = t0.elapsed().as_micros() as f64 / n_enc as f64;

    let n_fwd = 5000;
    let t1 = Instant::now();
    for _ in 0..n_fwd {
        std::hint::black_box(net.forward(&input));
    }
    let fwd_us = t1.elapsed().as_micros() as f64 / n_fwd as f64;

    println!("encode_board_compact  : {:.1} µs", enc_us);
    println!(
        "DeltaNet::forward     : {:.1} µs ({:.3} ms)",
        fwd_us,
        fwd_us / 1000.0
    );
    println!("Param count           : {}", net.param_count());
}
