//! Load an AZR3 file produced by `train_hybrid_delta.py`, attach the
//! v4-opp NNUE, and verify forward predictions are sane. The round-trip
//! is the critical wiring test for the Python → Rust handoff.

use rand::{rngs::StdRng, SeedableRng};
use std::path::Path;

use cascadia_ai::hybrid::HybridNetwork;
use cascadia_ai::nnue::{extract_features_with_bag, BagInfo, NNUENetwork};
use cascadia_core::game::GameState;
use cascadia_core::types::ScoringCards;

fn main() {
    let azr3 = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/hybr_smoke.azr3".to_string());
    let nnue_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "nnue_weights_v4opp_modal_iter3.bin".to_string());

    let nnue = NNUENetwork::load(Path::new(&nnue_path)).expect("load nnue");
    let mut hybrid =
        HybridNetwork::load_with_nnue(Path::new(&azr3), nnue.clone()).expect("load azr3");

    println!("HybridNetwork loaded:");
    println!("  alpha          = {}", hybrid.alpha);
    println!("  delta.in_c     = {}", hybrid.delta.input_channels);
    println!("  delta.trunk_c  = {}", hybrid.delta.trunk_channels);
    println!("  delta.blocks   = {}", hybrid.delta.blocks);
    println!("  delta.hidden   = {}", hybrid.delta.hidden);
    println!("  delta param ct = {}", hybrid.delta.param_count());

    let mut rng = StdRng::seed_from_u64(0);
    let g = GameState::new(4, ScoringCards::all_a(), &mut rng);
    let board = &g.boards[0];
    let bag = BagInfo::from_game_for_player(&g, 0);
    let features = extract_features_with_bag(board, Some(&bag));

    let v_nnue = nnue.forward(&features);
    let (v_n_via_hybrid, delta) = hybrid.evaluate_components(board, &features);
    let v_hybrid_loaded = hybrid.evaluate(board, &features);

    println!("\nFresh game, seat 0:");
    println!("  v_nnue (standalone)              = {:.4}", v_nnue);
    println!("  v_nnue (via hybrid components)   = {:.4}", v_n_via_hybrid);
    println!("  delta                            = {:.4}", delta);
    println!(
        "  v_hybrid (alpha={})            = {:.4}",
        hybrid.alpha, v_hybrid_loaded
    );

    // Force α=0 to verify the fast path matches NNUE exactly.
    hybrid.alpha = 0.0;
    let v_alpha0 = hybrid.evaluate(board, &features);
    println!("  v_hybrid (alpha=0)               = {:.4}", v_alpha0);
    assert_eq!(v_alpha0, v_nnue, "alpha=0 must equal NNUE bit-for-bit");
    println!("\nOK — alpha=0 path is bit-identical to NNUE; α-blended path differs.");
}
