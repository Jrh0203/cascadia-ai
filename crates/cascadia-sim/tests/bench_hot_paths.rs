//! Timing harness for the self-play/search generation hot paths.
//!
//! Run with:
//! ```bash
//! cargo test --release -p cascadia-sim --test bench_hot_paths -- --ignored --nocapture
//! ```
//!
//! Besides wall-clock numbers, every section prints a blake3 checksum of the
//! produced candidates/scores so behavior parity across optimizations can be
//! verified bit-for-bit.

use std::time::Instant;

use blake3::Hasher;
use cascadia_game::{GameConfig, GameSeed, GameState, MarketPrelude, score_game};
use cascadia_sim::rank_greedy_actions;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const RANK_LIMIT: usize = 16;
const ROLLOUT_TOP_K: usize = 4;

/// Mirrors the exporter's `complete_with_sampled_greedy` advance loop.
fn advance_sampled_greedy(game: &mut GameState, plies: usize, rng: &mut ChaCha8Rng) {
    for _ in 0..plies {
        if game.is_game_over() {
            return;
        }
        let (_prelude, staged) = game.preview_free_three_of_a_kind_if_feasible().unwrap();
        let candidates =
            rank_greedy_actions(&staged, &MarketPrelude::default(), Some(RANK_LIMIT)).unwrap();
        assert!(!candidates.is_empty());
        let sample_limit = ROLLOUT_TOP_K.min(candidates.len());
        let sampled = if sample_limit == 1 {
            0
        } else {
            rng.gen_range(0..sample_limit)
        };
        let action = candidates[sampled].action.clone();
        *game = staged;
        game.apply(&action).unwrap();
    }
}

fn rollout_to_terminal(mut game: GameState, rng: &mut ChaCha8Rng) -> Vec<u16> {
    while !game.is_game_over() {
        let (_prelude, staged) = match game.preview_free_three_of_a_kind_if_feasible() {
            Ok(pair) => pair,
            Err(_) => break,
        };
        let candidates =
            match rank_greedy_actions(&staged, &MarketPrelude::default(), Some(RANK_LIMIT)) {
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
        game = staged;
        if game.apply(&action).is_err() {
            break;
        }
    }
    score_game(&game)
        .into_iter()
        .map(|score| score.base_total)
        .collect()
}

fn time<T>(iters: usize, mut body: impl FnMut() -> T) -> (f64, T) {
    let mut last = None;
    let started = Instant::now();
    for _ in 0..iters {
        last = Some(std::hint::black_box(body()));
    }
    let per_iter = started.elapsed().as_secs_f64() * 1e6 / iters as f64;
    (per_iter, last.unwrap())
}

fn checksum(value: &impl serde::Serialize) -> String {
    let mut hasher = Hasher::new();
    hasher.update(&serde_json::to_vec(value).unwrap());
    hasher.finalize().to_hex()[..16].to_string()
}

#[test]
#[ignore = "timing harness; run explicitly with --ignored --nocapture"]
fn bench_hot_paths() {
    println!();
    println!(
        "{:<28} {:>6} {:>6} {:>12} {:>10}  {}",
        "section", "seed", "plies", "us/op", "iters", "checksum"
    );

    for seed in [11u64, 12, 13] {
        for plies in [10usize, 40, 70] {
            let config = GameConfig::research_aaaaa(4).unwrap();
            let mut game = GameState::new(config, GameSeed::from_u64(seed)).unwrap();
            let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xC0FFEE);
            advance_sampled_greedy(&mut game, plies, &mut rng);
            assert!(!game.is_game_over());
            let prelude = MarketPrelude::default();

            // (a1) rank_greedy_actions with limit Some(16)
            let iters = 60;
            let (us, candidates) = time(iters, || {
                rank_greedy_actions(&game, &prelude, Some(RANK_LIMIT)).unwrap()
            });
            println!(
                "{:<28} {:>6} {:>6} {:>12.1} {:>10}  {}",
                "rank_greedy Some(16)",
                seed,
                plies,
                us,
                iters,
                checksum(&candidates)
            );

            // (a2) rank_greedy_actions with limit None
            let (us, all_candidates) = time(iters, || {
                rank_greedy_actions(&game, &prelude, None).unwrap()
            });
            println!(
                "{:<28} {:>6} {:>6} {:>12.1} {:>10}  {} (n={})",
                "rank_greedy None",
                seed,
                plies,
                us,
                iters,
                checksum(&all_candidates),
                all_candidates.len()
            );

            // (b) score_game
            let iters = 400;
            let (us, scores) = time(iters, || score_game(&game));
            println!(
                "{:<28} {:>6} {:>6} {:>12.1} {:>10}  {}",
                "score_game",
                seed,
                plies,
                us,
                iters,
                checksum(&scores)
            );

            // (d) 16x clone + apply + score_game (candidate afterstates)
            let iters = 30;
            let retained = candidates;
            let (us, afterstate_scores) = time(iters, || {
                retained
                    .iter()
                    .map(|candidate| {
                        let mut next = game.clone();
                        next.apply(&candidate.action).unwrap();
                        score_game(&next)
                            .into_iter()
                            .map(|score| score.base_total)
                            .sum::<u16>()
                    })
                    .collect::<Vec<_>>()
            });
            println!(
                "{:<28} {:>6} {:>6} {:>12.1} {:>10}  {}",
                "16x clone+apply+score",
                seed,
                plies,
                us,
                iters,
                checksum(&afterstate_scores)
            );

            // (c) full sampled-greedy rollout to terminal
            let iters = if plies == 10 { 4 } else { 8 };
            let (us, terminal) = time(iters, || {
                let mut rollout_rng = ChaCha8Rng::seed_from_u64(seed ^ 0xB0BA);
                rollout_to_terminal(game.clone(), &mut rollout_rng)
            });
            println!(
                "{:<28} {:>6} {:>6} {:>12.1} {:>10}  {}",
                "rollout to terminal",
                seed,
                plies,
                us,
                iters,
                checksum(&terminal)
            );
        }
    }
}
