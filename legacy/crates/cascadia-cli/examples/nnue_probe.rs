//! Quick probe: load v4-opp NNUE, print predicted "remaining points" at
//! turn 0 of a fresh game. Sane value should be ~95 (champion mean).

use rand::{rngs::StdRng, SeedableRng};
use std::path::Path;

use cascadia_ai::nnue::{
    extract_features, extract_features_with_bag, BagInfo, NNUENetwork, NUM_FEATURES,
};
use cascadia_core::game::GameState;
use cascadia_core::scoring::ScoreBreakdown;
use cascadia_core::types::ScoringCards;

fn main() {
    let weights = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "nnue_weights_v4opp_modal_iter3.bin".to_string());
    println!("NUM_FEATURES at compile time = {}", NUM_FEATURES);
    let net = NNUENetwork::load(Path::new(&weights)).expect("load");
    println!("Loaded {}", weights);

    let mut rng = StdRng::seed_from_u64(0);
    let g = GameState::new(4, ScoringCards::all_a(), &mut rng);
    let board = &g.boards[0];

    let features_simple = extract_features(board);
    let bag = BagInfo::from_game_for_player(&g, 0);
    let features_bag = extract_features_with_bag(board, Some(&bag));

    let pred_simple = net.forward(&features_simple);
    let pred_bag = net.forward(&features_bag);

    let mut boards = g.boards.clone();
    let bd = ScoreBreakdown::compute_with_bonuses(&mut boards, &g.scoring_cards, 0);

    println!("Turn 0 / current score (with bonus) = {}", bd.total);
    println!("NNUE.forward(extract_features)         = {}", pred_simple);
    println!("NNUE.forward(extract_features_with_bag)= {}", pred_bag);
    println!("(Sane v4-opp prediction at turn 0 ≈ 90–100; the NNUE predicts");
    println!(" remaining points, and 4-player AAAAA with-bonus champion ≈ 95.)");
}
