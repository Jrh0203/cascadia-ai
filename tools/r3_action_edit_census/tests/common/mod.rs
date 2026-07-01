#![allow(dead_code)]

use cascadia_game::{DraftChoice, GameConfig, GameSeed, GameState, MarketPrelude, TurnAction};

pub fn game(seed: u64) -> GameState {
    GameState::new(
        GameConfig::research_aaaaa(4).unwrap(),
        GameSeed::from_u64(seed),
    )
    .unwrap()
}

pub fn canonical_actions(game: &GameState) -> (MarketPrelude, Vec<TurnAction>) {
    let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible().unwrap();
    let actions = game.legal_turn_actions(&prelude).unwrap();
    assert!(!actions.is_empty());
    (prelude, actions)
}

pub fn state_with_nature_token() -> GameState {
    state_with_nature_tokens(1)
}

pub fn state_with_nature_tokens(required: u8) -> GameState {
    for seed in 0..512 {
        let mut state = game(10_000 + seed);
        for _ in 0..64 {
            if state.boards()[state.current_player()].nature_tokens() >= required {
                return state;
            }
            let (_, actions) = canonical_actions(&state);
            let before = state.boards()[state.current_player()].nature_tokens();
            let selected = actions
                .iter()
                .find(|action| {
                    state
                        .preview_active_board(action)
                        .is_ok_and(|board| board.nature_tokens() > before)
                })
                .or_else(|| actions.iter().find(|action| action.wildlife.is_some()))
                .unwrap_or(&actions[0])
                .clone();
            state.apply(&selected).unwrap();
        }
    }
    panic!("failed to generate a current player with {required} Nature Tokens");
}

pub fn representative_dual_action(game: &GameState) -> TurnAction {
    let (_, actions) = canonical_actions(game);
    actions
        .iter()
        .find(|action| action.tile.rotation.get() != 0 && action.wildlife.is_some())
        .or_else(|| {
            actions
                .iter()
                .find(|action| action.tile.rotation.get() != 0)
        })
        .unwrap()
        .clone()
}

pub fn representative_drafts(game: &GameState) -> (DraftChoice, DraftChoice) {
    let (_, actions) = canonical_actions(game);
    let paired = actions
        .iter()
        .find_map(|action| {
            matches!(action.draft, DraftChoice::Paired { .. }).then_some(action.draft)
        })
        .unwrap();
    let independent = actions
        .iter()
        .find_map(|action| {
            matches!(action.draft, DraftChoice::Independent { .. }).then_some(action.draft)
        })
        .unwrap();
    (paired, independent)
}
