use cascadia_game::{GameConfig, GameSeed};
use cascadia_sim::{MatchConfig, StrategyKind, play_match};
use criterion::{Criterion, criterion_group, criterion_main};

fn benchmark_matches(criterion: &mut Criterion) {
    let game = GameConfig::research_aaaaa(4).unwrap();
    criterion.bench_function("simulation/random-four-player-game", |bencher| {
        let mut seed = 0u64;
        bencher.iter(|| {
            let config =
                MatchConfig::symmetric(game, GameSeed::from_u64(seed), StrategyKind::Random);
            seed = seed.wrapping_add(1);
            play_match(&config).unwrap()
        })
    });
}

criterion_group!(benches, benchmark_matches);
criterion_main!(benches);
