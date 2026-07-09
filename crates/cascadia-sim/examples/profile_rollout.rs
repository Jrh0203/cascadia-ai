//! Temporary profiling target: runs sampled-greedy rollouts in a loop so a
//! sampling profiler (macOS `sample`) can attribute time inside a rollout ply.

use cascadia_game::{GameConfig, GameSeed, GameState, MarketPrelude, score_game};
use cascadia_sim::{
    GreedyRankScratch, rank_greedy_actions_with_market_choice, rank_greedy_actions_with_scratch,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const RANK_LIMIT: usize = 16;
const ROLLOUT_TOP_K: usize = 4;

fn advance_sampled_greedy(game: &mut GameState, plies: usize, rng: &mut ChaCha8Rng) {
    let mut scratch = GreedyRankScratch::default();
    for _ in 0..plies {
        if game.is_game_over() {
            return;
        }
        let candidates = if game.market().three_of_a_kind().is_some() {
            rank_greedy_actions_with_market_choice(game, Some(RANK_LIMIT))
        } else {
            rank_greedy_actions_with_scratch(
                game,
                &MarketPrelude::default(),
                Some(RANK_LIMIT),
                &mut scratch,
            )
        }
        .unwrap();
        let sample_limit = ROLLOUT_TOP_K.min(candidates.len());
        let sampled = if sample_limit == 1 {
            0
        } else {
            rng.gen_range(0..sample_limit)
        };
        let action = candidates[sampled].action.clone();
        game.apply(&action).unwrap();
    }
}

fn rollout_to_terminal(mut game: GameState, rng: &mut ChaCha8Rng) -> u16 {
    let mut scratch = GreedyRankScratch::default();
    while !game.is_game_over() {
        let candidates = if game.market().three_of_a_kind().is_some() {
            rank_greedy_actions_with_market_choice(&game, Some(RANK_LIMIT))
        } else {
            rank_greedy_actions_with_scratch(
                &game,
                &MarketPrelude::default(),
                Some(RANK_LIMIT),
                &mut scratch,
            )
        };
        let candidates = match candidates {
            Ok(candidates) => candidates,
            Err(_) => break,
        };
        if candidates.is_empty() {
            break;
        }
        let sample_limit = ROLLOUT_TOP_K.min(candidates.len());
        let sampled = if sample_limit == 1 {
            0
        } else {
            rng.gen_range(0..sample_limit)
        };
        let action = candidates[sampled].action.clone();
        if game.apply(&action).is_err() {
            break;
        }
    }
    score_game(&game).iter().map(|score| score.base_total).sum()
}

fn main() {
    let seconds: f64 = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(20.0);
    let started = std::time::Instant::now();
    let mut total = 0u64;
    let mut rollouts = 0u64;
    let mut seed = 11u64;
    while started.elapsed().as_secs_f64() < seconds {
        let config = GameConfig::research_aaaaa(4).unwrap();
        let mut game = GameState::new(config, GameSeed::from_u64(seed)).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xC0FFEE);
        // Mimic search-leaf rollouts: advance to a mid-game state, then run
        // rollouts to terminal from clones of it.
        advance_sampled_greedy(&mut game, 10, &mut rng);
        for index in 0..24u64 {
            let mut rollout_rng = ChaCha8Rng::seed_from_u64(seed ^ (index * 0x9E37));
            total += u64::from(rollout_to_terminal(game.clone(), &mut rollout_rng));
            rollouts += 1;
        }
        seed += 1;
    }
    println!(
        "rollouts={} avg_score_sum={} elapsed={:.2}s ({:.1} rollouts/s)",
        rollouts,
        total / rollouts.max(1),
        started.elapsed().as_secs_f64(),
        rollouts as f64 / started.elapsed().as_secs_f64()
    );
}
