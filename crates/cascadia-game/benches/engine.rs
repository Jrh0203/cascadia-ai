use cascadia_game::{GameConfig, GameSeed, GameState, MarketPrelude, score_board};
use criterion::{Criterion, criterion_group, criterion_main};

fn benchmark_engine(criterion: &mut Criterion) {
    let state = GameState::new(
        GameConfig::research_aaaaa(4).unwrap(),
        GameSeed::from_u64(0),
    )
    .unwrap();
    let prelude = MarketPrelude {
        replace_three_of_a_kind: state.market().three_of_a_kind().is_some(),
        wildlife_wipes: Vec::new(),
    };
    let actions = state.legal_turn_actions(&prelude).unwrap();
    let action = actions[0].clone();

    criterion.bench_function("game/legal-actions-initial", |bencher| {
        bencher.iter(|| state.legal_turn_actions(&prelude).unwrap())
    });
    criterion.bench_function("game/score-board-initial", |bencher| {
        bencher.iter(|| {
            score_board(
                &state.boards()[state.current_player()],
                state.config().scoring_cards,
            )
        })
    });
    criterion.bench_function("game/preview-active-board", |bencher| {
        bencher.iter(|| state.preview_active_board(&action).unwrap())
    });
    criterion.bench_function("game/transactional-transition", |bencher| {
        bencher.iter(|| state.transition(&action).unwrap())
    });
}

criterion_group!(benches, benchmark_engine);
criterion_main!(benches);
