// Play full random 4p games; count overpopulation stalls and time.
use cascadia_game::{GameConfig, GameSeed, GameState};
use std::time::Instant;

#[test]
fn random_full_games_terminate_quickly() {
    let started = Instant::now();
    let mut errors = 0usize;
    for seed in 0..500u64 {
        let config = GameConfig::research_aaaaa(4).unwrap();
        let mut game = GameState::new(config, GameSeed::from_u64(9_000_000 + seed)).unwrap();
        let mut plies = 0usize;
        while !game.is_game_over() && plies < 90 {
            let prelude = game.free_three_of_a_kind_choices().unwrap().into_iter().next().unwrap();
            let actions = game.legal_turn_actions(&prelude).unwrap();
            let pick = (seed as usize + plies * 7) % actions.len();
            if let Err(error) = game.apply(&actions[pick]) {
                eprintln!("seed {seed} ply {plies}: {error:#}");
                errors += 1;
                break;
            }
            plies += 1;
        }
    }
    eprintln!("500 games in {:.2}s, {errors} errors", started.elapsed().as_secs_f64());
    assert_eq!(errors, 0);
}
