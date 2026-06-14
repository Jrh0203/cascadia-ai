use cascadia_game::{
    Board, GameConfig, GameSeed, HexCoord, Rotation, STANDARD_TILES, ScoringCards,
};
use rand::SeedableRng;

use super::*;
use crate::{
    play_greedy_plies, rank_bear_setup_actions, rank_greedy_actions, rank_habitat_setup_actions,
};

#[test]
fn strategy_id_records_every_pattern_configuration_field() {
    assert_eq!(
        PatternAwareConfig::default().strategy_id(),
        PATTERN_AWARE_STRATEGY_ID
    );
    assert_eq!(
        PatternAwareConfig {
            immediate_candidate_limit: 2,
            habitat_candidate_limit: 1,
            bear_candidate_limit: 1,
            future_market_draws: 2,
        }
        .strategy_id(),
        "pattern-aware-v1-k2-h1-b1-m2"
    );
    assert_eq!(
        PatternPotentialConfig::from_weights(PatternAwareConfig::default(), 1.25, 0.5, 0.75,)
            .unwrap()
            .strategy_id(),
        "pattern-potential-v1-k8-h6-b8-m4-a125-h050-b075"
    );
}

#[test]
fn pattern_potential_rejects_values_outside_the_registered_grid() {
    for weights in [
        (0.25, 0.0, 0.0),
        (1.75, 0.0, 0.0),
        (1.0, -0.25, 0.0),
        (1.0, 1.25, 0.0),
        (1.0, 0.0, 0.1),
        (f64::NAN, 0.0, 0.0),
    ] {
        assert!(matches!(
            PatternPotentialConfig::from_weights(
                PatternAwareConfig::default(),
                weights.0,
                weights.1,
                weights.2,
            ),
            Err(SimulationError::Strategy(_))
        ));
    }
}

#[test]
fn production_pattern_potential_reproduces_rankings_and_complete_games() {
    let game_config = GameConfig::research_aaaaa(4).unwrap();
    let seed = GameSeed::from_u64(709);
    let game = GameState::new(game_config, seed).unwrap();
    let prelude = MarketPrelude {
        replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
        wildlife_wipes: Vec::new(),
    };
    let baseline = rank_pattern_actions(&game, &prelude, PatternAwareConfig::default()).unwrap();
    let potential =
        rank_pattern_potential_actions(&game, &prelude, PatternPotentialConfig::default()).unwrap();
    assert_eq!(potential, baseline);

    let baseline_match = crate::play_match(&crate::MatchConfig::symmetric(
        game_config,
        seed,
        crate::StrategyKind::PatternAware,
    ))
    .unwrap();
    let potential_match = PatternPotentialStrategy::new(PatternPotentialConfig::default())
        .unwrap()
        .play_match(game_config, seed)
        .unwrap();
    assert_eq!(potential_match.scores, baseline_match.scores);
    assert_eq!(potential_match.replay, baseline_match.replay);
}

#[test]
fn pattern_potential_structural_deltas_match_afterstate_references() {
    let game = GameState::new(
        GameConfig::research_aaaaa(4).unwrap(),
        GameSeed::from_u64(710),
    )
    .unwrap();
    let prelude = MarketPrelude {
        replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
        wildlife_wipes: Vec::new(),
    };
    let baseline_board = &game.boards()[game.current_player()];
    let baseline_edges = baseline_board.habitat_analysis().matching_edges();
    let baseline_bear = bear_pair_ready_slots(baseline_board);
    let ranked =
        rank_pattern_potential_actions(&game, &prelude, PatternPotentialConfig::default()).unwrap();

    for candidate in ranked {
        let after = game.preview_public_afterstate(&candidate.action).unwrap();
        let board = &after.boards()[game.current_player()];
        assert_eq!(
            candidate.matching_habitat_edge_delta,
            board.habitat_analysis().matching_edges().cast_signed() - baseline_edges.cast_signed()
        );
        assert_eq!(
            candidate.bear_pair_ready_delta,
            bear_pair_ready_slots(board).cast_signed() - baseline_bear.cast_signed()
        );
    }
}

#[test]
fn final_turn_pattern_potential_has_zero_structural_credit() {
    let mut game = GameState::new(
        GameConfig::research_aaaaa(4).unwrap(),
        GameSeed::from_u64(711),
    )
    .unwrap();
    let mut rollout_rng = ChaCha8Rng::seed_from_u64(13);
    play_greedy_plies(&mut game, 76, &mut rollout_rng).unwrap();
    let prelude = MarketPrelude {
        replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
        wildlife_wipes: Vec::new(),
    };
    let config =
        PatternPotentialConfig::from_weights(PatternAwareConfig::default(), 1.5, 1.0, 1.0).unwrap();
    let ranked = rank_pattern_potential_actions(&game, &prelude, config).unwrap();

    assert!(!ranked.is_empty());
    assert!(ranked.iter().all(|candidate| {
        candidate.personal_turns_remaining == 0
            && candidate.future_market_opportunity == 0.0
            && candidate.heuristic_value == f64::from(candidate.resulting_base_score)
    }));
}

#[test]
fn expected_market_max_respects_without_replacement_supply() {
    let all_bear = expected_max_without_replacement([4, 0, 0, 0, 0], [7.0, 0.0, 0.0, 0.0, 0.0], 4);
    let mixed = expected_max_without_replacement([1, 3, 0, 0, 0], [7.0, 2.0, 0.0, 0.0, 0.0], 4);
    let all_elk = expected_max_without_replacement([0, 3, 0, 0, 0], [7.0, 2.0, 0.0, 0.0, 0.0], 4);

    assert!((all_bear - 7.0).abs() < 1e-12);
    assert!((mixed - 7.0).abs() < 1e-12);
    assert!((all_elk - 2.0).abs() < 1e-12);
}

#[test]
fn two_turn_opportunity_sees_through_a_zero_score_bear_setup() {
    let mut board = Board::empty();
    for (coord, tile) in [
        (HexCoord::new(0, 0), STANDARD_TILES[2]),
        (HexCoord::new(1, 0), STANDARD_TILES[5]),
        (HexCoord::new(2, 0), STANDARD_TILES[0]),
        (HexCoord::new(3, 0), STANDARD_TILES[6]),
        (HexCoord::new(4, 0), STANDARD_TILES[23]),
    ] {
        board.place_tile(coord, tile, Rotation::ZERO).unwrap();
    }
    board
        .place_wildlife(HexCoord::new(0, 0), Wildlife::Bear)
        .unwrap();
    board
        .place_wildlife(HexCoord::new(1, 0), Wildlife::Bear)
        .unwrap();

    let one_turn = future_wildlife_opportunity(&board, ScoringCards::AAAAA, [20, 0, 0, 0, 0], 4, 1);
    let two_turn = future_wildlife_opportunity(&board, ScoringCards::AAAAA, [20, 0, 0, 0, 0], 4, 2);

    assert_eq!(one_turn, 1.0);
    assert_eq!(two_turn, 9.0);
}

#[test]
fn two_turn_opportunity_never_values_less_than_one_turn() {
    let game = GameState::new(
        GameConfig::research_aaaaa(4).unwrap(),
        GameSeed::from_u64(705),
    )
    .unwrap();
    let board = &game.boards()[0];
    let counts = game.unplaced_wildlife_counts();
    let one_turn = future_wildlife_opportunity(board, ScoringCards::AAAAA, counts, 4, 1);
    let two_turn = future_wildlife_opportunity(board, ScoringCards::AAAAA, counts, 4, 2);

    assert!(two_turn >= one_turn);
}

#[test]
fn replacement_process_conserves_probability_and_tokens() {
    let three_of_a_kind = WildlifeMarketState {
        market: [3, 1, 0, 0, 0],
        bag: [5, 7, 8, 9, 10],
    };
    let automatic_four = WildlifeMarketState {
        market: [0, 0, 0, 4, 0],
        bag: [8, 8, 8, 4, 8],
    };

    assert_eq!(market_token_total(three_of_a_kind), 43);
    assert_eq!(market_token_total(automatic_four), 40);
    let actor_values = [1.0; 5];
    let three_value = expected_market_value(three_of_a_kind, 0, &[], actor_values).unwrap();
    let four_value = expected_market_value(automatic_four, 0, &[], actor_values).unwrap();
    assert!((three_value - 1.0).abs() < 1e-12);
    assert!((four_value - 1.0).abs() < 1e-12);

    let mut kernel = ReplacementKernel::default();
    let three_distribution =
        terminal_market_distribution(three_of_a_kind, 0, &[], &mut kernel).unwrap();
    let four_distribution =
        terminal_market_distribution(automatic_four, 0, &[], &mut kernel).unwrap();
    assert!(
        (three_distribution
            .iter()
            .map(|(_, probability)| probability)
            .sum::<f64>()
            - 1.0)
            .abs()
            < 1e-12
    );
    assert!(
        (four_distribution
            .iter()
            .map(|(_, probability)| probability)
            .sum::<f64>()
            - 1.0)
            .abs()
            < 1e-12
    );
}

#[test]
fn closed_form_four_replacement_matches_exhaustive_reference() {
    fn exhaustive(bag: [u8; 5], reach: f64, outcomes: &mut HashMap<[u8; 5], f64>) {
        for (allocation, probability) in draw_allocations(bag, 4) {
            let mut remaining = bag;
            for (index, drawn) in allocation.into_iter().enumerate() {
                remaining[index] -= drawn;
            }
            if allocation.contains(&4) {
                exhaustive(remaining, reach * probability, outcomes);
            } else {
                *outcomes.entry(allocation).or_insert(0.0) += reach * probability;
            }
        }
    }

    let bag = [5, 5, 2, 1, 0];
    let mut exhaustive_outcomes = HashMap::new();
    exhaustive(bag, 1.0, &mut exhaustive_outcomes);
    let mut kernel = ReplacementKernel::default();
    let closed_form = kernel.stable_four_draws(bag);

    assert_eq!(closed_form.len(), exhaustive_outcomes.len());
    for (market, probability) in closed_form {
        let reference = exhaustive_outcomes[&market];
        assert!(
            (probability - reference).abs() < 1e-12,
            "{market:?}: closed form {probability}, exhaustive {reference}"
        );
    }
}

#[test]
fn opponent_drafts_unique_highest_value_species() {
    let state = WildlifeMarketState {
        market: [1, 1, 1, 1, 0],
        bag: [10; 5],
    };
    let choices = opponent_draft_choices(state, [9.0, 2.0, 1.0, 3.0, 0.0]).unwrap();

    assert_eq!(choices, vec![(Wildlife::Bear as usize, 1.0)]);
}

#[test]
fn commitment_ranking_collapses_to_one_turn_when_only_one_future_turn_remains() {
    let mut game = GameState::new(
        GameConfig::research_aaaaa(4).unwrap(),
        GameSeed::from_u64(706),
    )
    .unwrap();
    let mut rollout_rng = ChaCha8Rng::seed_from_u64(11);
    play_greedy_plies(&mut game, 72, &mut rollout_rng).unwrap();

    assert_eq!(game.current_player(), 0);
    assert_eq!(game.turns_remaining_for_player(0), 2);
    let prelude = MarketPrelude {
        replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
        wildlife_wipes: Vec::new(),
    };
    let one_turn = rank_pattern_actions(&game, &prelude, PatternAwareConfig::default()).unwrap();
    let phase_capped =
        rank_pattern_commitment_actions(&game, &prelude, PatternAwareConfig::default()).unwrap();

    assert_eq!(phase_capped, one_turn);
}

#[test]
fn portfolio_ranking_equals_pattern_aware_with_one_future_turn() {
    let mut game = GameState::new(
        GameConfig::research_aaaaa(4).unwrap(),
        GameSeed::from_u64(709),
    )
    .unwrap();
    let mut rollout_rng = ChaCha8Rng::seed_from_u64(13);
    play_greedy_plies(&mut game, 72, &mut rollout_rng).unwrap();

    assert_eq!(game.current_player(), 0);
    assert_eq!(game.turns_remaining_for_player(0), 2);
    let prelude = MarketPrelude {
        replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
        wildlife_wipes: Vec::new(),
    };
    let pattern = rank_pattern_actions(&game, &prelude, PatternAwareConfig::default()).unwrap();
    let portfolio =
        rank_pattern_portfolio_actions(&game, &prelude, PatternAwareConfig::default()).unwrap();

    assert_eq!(portfolio, pattern);
}

#[test]
fn pattern_competition_has_zero_opportunity_on_the_final_personal_turn() {
    let mut game = GameState::new(
        GameConfig::research_aaaaa(4).unwrap(),
        GameSeed::from_u64(708),
    )
    .unwrap();
    let mut rollout_rng = ChaCha8Rng::seed_from_u64(12);
    play_greedy_plies(&mut game, 76, &mut rollout_rng).unwrap();

    assert_eq!(game.current_player(), 0);
    assert_eq!(game.turns_remaining_for_player(0), 1);
    let prelude = MarketPrelude {
        replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
        wildlife_wipes: Vec::new(),
    };
    let ranked =
        rank_pattern_competition_actions(&game, &prelude, PatternAwareConfig::default()).unwrap();
    let portfolio =
        rank_pattern_portfolio_actions(&game, &prelude, PatternAwareConfig::default()).unwrap();

    assert!(!ranked.is_empty());
    assert!(ranked.iter().all(|candidate| {
        candidate.future_market_opportunity == 0.0
            && candidate.heuristic_value == f64::from(candidate.resulting_base_score)
    }));
    assert!(!portfolio.is_empty());
    assert!(portfolio.iter().all(|candidate| {
        candidate.future_market_opportunity == 0.0
            && candidate.heuristic_value == f64::from(candidate.resulting_base_score)
    }));
}

#[test]
fn pattern_ranking_is_legal_reproducible_and_bounded() {
    let game = GameState::new(
        GameConfig::research_aaaaa(4).unwrap(),
        GameSeed::from_u64(701),
    )
    .unwrap();
    let config = PatternAwareConfig {
        immediate_candidate_limit: 3,
        habitat_candidate_limit: 2,
        bear_candidate_limit: 2,
        future_market_draws: 4,
    };
    let left = rank_pattern_actions(&game, &MarketPrelude::default(), config).unwrap();
    let right = rank_pattern_actions(&game, &MarketPrelude::default(), config).unwrap();

    assert_eq!(left, right);
    assert!(!left.is_empty());
    assert!(left.len() <= 7);
    assert!(
        left.windows(2)
            .all(|pair| pair[0].heuristic_value >= pair[1].heuristic_value)
    );
    for candidate in left {
        game.transition(&candidate.action).unwrap();
        assert!(candidate.future_market_opportunity.is_finite());
        assert!(candidate.heuristic_value >= f64::from(candidate.resulting_base_score));
    }
}

#[test]
fn unified_frontier_matches_three_reference_rankings() {
    let mut game = GameState::new(
        GameConfig::research_aaaaa(4).unwrap(),
        GameSeed::from_u64(703),
    )
    .unwrap();
    let mut rollout_rng = ChaCha8Rng::seed_from_u64(10);
    let config = PatternAwareConfig {
        immediate_candidate_limit: 5,
        habitat_candidate_limit: 4,
        bear_candidate_limit: 3,
        future_market_draws: 4,
    };

    for plies in [0, 7, 19] {
        if plies > 0 {
            play_greedy_plies(&mut game, plies, &mut rollout_rng).unwrap();
        }
        let prelude = MarketPrelude {
            replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
            wildlife_wipes: Vec::new(),
        };
        let staged = game.preview_market_prelude(&prelude).unwrap();
        let mut expected = rank_greedy_actions(
            &staged,
            &MarketPrelude::default(),
            Some(config.immediate_candidate_limit),
        )
        .unwrap();
        merge_unique(
            &mut expected,
            rank_habitat_setup_actions(
                &staged,
                &MarketPrelude::default(),
                Some(config.habitat_candidate_limit),
            )
            .unwrap(),
        );
        merge_unique(
            &mut expected,
            rank_bear_setup_actions(
                &staged,
                &MarketPrelude::default(),
                Some(config.bear_candidate_limit),
            )
            .unwrap(),
        );
        let actual =
            rank_pattern_frontier_actions(&staged, &MarketPrelude::default(), config).unwrap();

        assert_eq!(actual, expected);
    }
}

#[test]
fn wildlife_diverse_frontier_is_a_bounded_superset_with_species_coverage() {
    let game = GameState::new(
        GameConfig::research_aaaaa(4).unwrap(),
        GameSeed::from_u64(41),
    )
    .unwrap();
    let prelude = MarketPrelude {
        replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
        wildlife_wipes: Vec::new(),
    };
    let staged = game.preview_market_prelude(&prelude).unwrap();
    let config = PatternAwareConfig::default();
    let baseline =
        rank_pattern_frontier_actions(&staged, &MarketPrelude::default(), config).unwrap();
    let expanded = rank_wildlife_diverse_pattern_frontier_actions(
        &staged,
        &MarketPrelude::default(),
        config,
        2,
    )
    .unwrap();

    assert!(
        baseline
            .iter()
            .all(|candidate| expanded.iter().any(|item| item.action == candidate.action))
    );
    assert!(expanded.len() <= baseline.len() + Wildlife::ALL.len() * 2);

    for wildlife in staged.market().wildlife.iter().flatten() {
        assert!(expanded.iter().any(|candidate| {
            drafted_wildlife(staged.market(), candidate.action.draft) == Some(*wildlife)
        }));
    }
}

#[test]
fn wildlife_diverse_frontier_rejects_zero_species_limit() {
    let game = GameState::new(
        GameConfig::research_aaaaa(4).unwrap(),
        GameSeed::from_u64(42),
    )
    .unwrap();
    assert!(matches!(
        rank_wildlife_diverse_pattern_frontier_actions(
            &game,
            &MarketPrelude::default(),
            PatternAwareConfig::default(),
            0,
        ),
        Err(SimulationError::Strategy(_))
    ));
}

#[test]
fn wildlife_focused_frontier_is_bounded_and_only_adds_selected_species() {
    let game = GameState::new(
        GameConfig::research_aaaaa(4).unwrap(),
        GameSeed::from_u64(41),
    )
    .unwrap();
    let prelude = MarketPrelude {
        replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
        wildlife_wipes: Vec::new(),
    };
    let staged = game.preview_market_prelude(&prelude).unwrap();
    let config = PatternAwareConfig::default();
    let baseline =
        rank_pattern_frontier_actions(&staged, &MarketPrelude::default(), config).unwrap();
    let focused = rank_wildlife_focused_pattern_frontier_actions(
        &staged,
        &MarketPrelude::default(),
        config,
        Wildlife::Fox,
        2,
    )
    .unwrap();

    assert!(
        baseline
            .iter()
            .all(|candidate| focused.iter().any(|item| item.action == candidate.action))
    );
    assert!(focused.len() <= baseline.len() + 2);
    for candidate in focused.iter().filter(|candidate| {
        baseline
            .iter()
            .all(|baseline| baseline.action != candidate.action)
    }) {
        assert_eq!(
            drafted_wildlife(staged.market(), candidate.action.draft),
            Some(Wildlife::Fox)
        );
    }
}

#[test]
fn wildlife_focused_frontier_rejects_zero_species_limit() {
    let game = GameState::new(
        GameConfig::research_aaaaa(4).unwrap(),
        GameSeed::from_u64(42),
    )
    .unwrap();
    assert!(matches!(
        rank_wildlife_focused_pattern_frontier_actions(
            &game,
            &MarketPrelude::default(),
            PatternAwareConfig::default(),
            Wildlife::Fox,
            0,
        ),
        Err(SimulationError::Strategy(_))
    ));
}

#[test]
fn pattern_selection_is_seeded_and_legal() {
    let game = GameState::new(
        GameConfig::research_aaaaa(4).unwrap(),
        GameSeed::from_u64(702),
    )
    .unwrap();
    let ranked = rank_pattern_actions(
        &game,
        &MarketPrelude::default(),
        PatternAwareConfig::default(),
    )
    .unwrap();
    let tied = ranked
        .iter()
        .take_while(|candidate| candidate.heuristic_value == ranked[0].heuristic_value)
        .count();
    let mut reference_rng = ChaCha8Rng::seed_from_u64(9);
    let expected = ranked[reference_rng.gen_range(0..tied)].action.clone();
    let mut left_rng = ChaCha8Rng::seed_from_u64(9);
    let mut right_rng = ChaCha8Rng::seed_from_u64(9);
    let left = select_pattern_action(
        &game,
        &MarketPrelude::default(),
        PatternAwareConfig::default(),
        &mut left_rng,
    )
    .unwrap();
    let right = select_pattern_action(
        &game,
        &MarketPrelude::default(),
        PatternAwareConfig::default(),
        &mut right_rng,
    )
    .unwrap();

    assert_eq!(left, expected);
    assert_eq!(left, right);
    game.transition(&left).unwrap();
    assert_eq!(
        best_pattern_heuristic_value(
            &game,
            &MarketPrelude::default(),
            PatternAwareConfig::default(),
        )
        .unwrap(),
        Some(ranked[0].heuristic_value)
    );
}

#[test]
fn pattern_competition_is_hidden_order_invariant_legal_and_reproducible() {
    let game = GameState::new(
        GameConfig::research_aaaaa(4).unwrap(),
        GameSeed::from_u64(707),
    )
    .unwrap();
    let mut redetermined = game.clone();
    redetermined.redeterminize_hidden(GameSeed::from_u64(999_707));
    let prelude = MarketPrelude {
        replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
        wildlife_wipes: Vec::new(),
    };

    let left =
        rank_pattern_competition_actions(&game, &prelude, PatternAwareConfig::default()).unwrap();
    let right =
        rank_pattern_competition_actions(&redetermined, &prelude, PatternAwareConfig::default())
            .unwrap();
    assert_eq!(left, right);

    let mut left_rng = ChaCha8Rng::seed_from_u64(19);
    let mut right_rng = ChaCha8Rng::seed_from_u64(19);
    let left_action = select_pattern_competition_action(
        &game,
        &prelude,
        PatternAwareConfig::default(),
        &mut left_rng,
    )
    .unwrap();
    let right_action = select_pattern_competition_action(
        &game,
        &prelude,
        PatternAwareConfig::default(),
        &mut right_rng,
    )
    .unwrap();
    assert_eq!(left_action, right_action);
    game.transition(&left_action).unwrap();
}

#[test]
fn pattern_portfolio_is_hidden_order_invariant_legal_and_nonnegative() {
    let game = GameState::new(
        GameConfig::research_aaaaa(4).unwrap(),
        GameSeed::from_u64(707),
    )
    .unwrap();
    let mut redetermined = game.clone();
    redetermined.redeterminize_hidden(GameSeed::from_u64(999_707));
    let prelude = MarketPrelude {
        replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
        wildlife_wipes: Vec::new(),
    };
    assert!(!prelude.replace_three_of_a_kind);

    let pattern = rank_pattern_actions(&game, &prelude, PatternAwareConfig::default()).unwrap();
    let left =
        rank_pattern_portfolio_actions(&game, &prelude, PatternAwareConfig::default()).unwrap();
    let right =
        rank_pattern_portfolio_actions(&redetermined, &prelude, PatternAwareConfig::default())
            .unwrap();
    assert_eq!(left, right);
    for candidate in &left {
        let one_turn = pattern
            .iter()
            .find(|reference| reference.action == candidate.action)
            .expect("portfolio retains the pattern-aware frontier");
        assert!(candidate.future_market_opportunity + 1e-12 >= one_turn.future_market_opportunity);
    }

    let mut left_rng = ChaCha8Rng::seed_from_u64(20);
    let mut right_rng = ChaCha8Rng::seed_from_u64(20);
    let left_action = select_pattern_portfolio_action(
        &game,
        &prelude,
        PatternAwareConfig::default(),
        &mut left_rng,
    )
    .unwrap();
    let right_action = select_pattern_portfolio_action(
        &game,
        &prelude,
        PatternAwareConfig::default(),
        &mut right_rng,
    )
    .unwrap();
    assert_eq!(left_action, right_action);
    game.transition(&left_action).unwrap();
}
