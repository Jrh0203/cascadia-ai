//! Phase 0.8 performance benchmarks.
//!
//! Captures microbenchmarks for the AZ-v2 hot paths so each subphase
//! (A → B → C → D → E) can verify the expected speedup landed and no
//! regression slipped in.
//!
//! Run:
//!   cargo bench --bench az_v2_perf \
//!     --features v4-opp,v5-feat,czero-feat,az-v2
//!
//! Stage targets (relative to Phase 1 baseline measurements):
//!
//! | Bench                                | Pre-A     | Post-B    | Post-C    |
//! |--------------------------------------|----------:|----------:|----------:|
//! | bench_full_forward_v2                | 50–80 ms  | ≤ 10 ms   |           |
//! | bench_full_forward_v2_batch8         |     —     |           | ≤ 30 ms   |
//! | bench_encode_game_local              | ~3.5 ms   |           | ≤ 1.2 ms  |
//! | bench_puct_single_search_32sims      | ~9 s/game |           | ≤ 0.3 s   |
//!
//! (Single-thread baseline; root-parallel multipliers measured separately.)

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::SeedableRng;

use cascadia_ai::alphazero_v2::{
    candidate_moves_v2, encode_game_local, AlphaZeroNetV2, AlphaZeroV2Config,
};
use cascadia_core::game::GameState;
use cascadia_core::types::ScoringCards;

/// Build a 4P AAAAA game played to a mid-game position (~12 moves in) for a
/// realistic forward / encode / search load.
fn mid_game_state(seed: u64) -> GameState {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut game = GameState::new(4, ScoringCards::all_a(), &mut rng);
    use cascadia_ai::search::{execute_scored_move, greedy_move};
    for _ in 0..12 {
        if game.is_game_over() {
            break;
        }
        if game.can_replace_overflow().is_some() {
            game.replace_overflow();
        }
        if let Some(mv) = greedy_move(&game) {
            if !execute_scored_move(&mut game, &mv) {
                break;
            }
        } else {
            break;
        }
    }
    game
}

fn bench_full_forward_v2(c: &mut Criterion) {
    let game = mid_game_state(0xA1);
    let net = AlphaZeroNetV2::new(AlphaZeroV2Config::default(), 0xB1);
    let cands = candidate_moves_v2(&game, net.config().max_candidates);

    c.bench_function("full_forward_v2", |b| {
        b.iter(|| {
            let (val, probs) = net.evaluate(black_box(&game), black_box(&cands));
            black_box((val, probs));
        });
    });
}

fn bench_encode_game_local(c: &mut Criterion) {
    let game = mid_game_state(0xA2);

    c.bench_function("encode_game_local", |b| {
        b.iter(|| {
            let (input, opp_inputs, entities) = encode_game_local(black_box(&game));
            black_box((input, opp_inputs, entities));
        });
    });
}

fn bench_puct_single_search_32sims(c: &mut Criterion) {
    let game = mid_game_state(0xA3);
    let net = AlphaZeroNetV2::new(AlphaZeroV2Config::default(), 0xB3);
    let mut group = c.benchmark_group("puct_single_search_32sims");
    // PUCT search is expensive; reduce sample size so the bench finishes
    // in reasonable time (default 100 iters × ~1s = 100s, too long).
    group.sample_size(20);
    group.bench_function("default", |b| {
        b.iter(|| {
            let mut rng = StdRng::seed_from_u64(0xC3);
            let result = cascadia_ai::alphazero_v2::az_search_v2(
                black_box(&game),
                black_box(&net),
                32,
                0.0,
                &mut rng,
            );
            black_box(result);
        });
    });
    group.finish();
}

criterion_group!(
    az_v2_perf,
    bench_full_forward_v2,
    bench_encode_game_local,
    bench_puct_single_search_32sims,
);
criterion_main!(az_v2_perf);
