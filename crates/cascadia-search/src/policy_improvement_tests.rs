use cascadia_game::{GameConfig, GameSeed, GameState};

use super::*;

fn tiny_config() -> TerminalPolicyImprovementConfig {
    TerminalPolicyImprovementConfig {
        determinizations: 1,
        blueprint: PatternAwareConfig {
            immediate_candidate_limit: 2,
            habitat_candidate_limit: 1,
            bear_candidate_limit: 1,
            future_market_draws: 2,
        },
    }
}

fn tiny_late_config() -> LateTerminalPolicyImprovementConfig {
    LateTerminalPolicyImprovementConfig {
        final_personal_turns: 4,
        terminal: tiny_config(),
    }
}

fn tiny_wildlife_diverse_config() -> WildlifeDiverseTerminalPolicyImprovementConfig {
    WildlifeDiverseTerminalPolicyImprovementConfig {
        wildlife_candidate_limit: 1,
        terminal: tiny_config(),
    }
}

fn tiny_late_wildlife_diverse_config() -> LateWildlifeDiversePolicyImprovementConfig {
    LateWildlifeDiversePolicyImprovementConfig {
        final_personal_turns: 4,
        terminal: tiny_wildlife_diverse_config(),
    }
}

fn tiny_conservative_config() -> LateConservativePolicyImprovementConfig {
    let mut terminal = tiny_wildlife_diverse_config();
    terminal.terminal.determinizations = 8;
    LateConservativePolicyImprovementConfig {
        final_personal_turns: 1,
        terminal,
    }
}

fn tiny_conservative_base_config() -> LateConservativeBasePolicyImprovementConfig {
    let mut terminal = tiny_config();
    terminal.determinizations = 8;
    LateConservativeBasePolicyImprovementConfig {
        final_personal_turns: 1,
        terminal,
    }
}

fn tiny_conservative_focused_config() -> LateConservativeWildlifeFocusedPolicyImprovementConfig {
    let mut terminal = tiny_config();
    terminal.determinizations = 8;
    LateConservativeWildlifeFocusedPolicyImprovementConfig {
        final_personal_turns: 1,
        terminal: WildlifeFocusedTerminalPolicyImprovementConfig {
            wildlife: Wildlife::Fox,
            wildlife_candidate_limit: 1,
            terminal,
        },
    }
}

fn tiny_conservative_beam_config() -> LateConservativeFocalBeamConfig {
    LateConservativeFocalBeamConfig {
        final_personal_turns: 1,
        determinizations: 4,
        beam_width: 1,
        wildlife_candidate_limit: 1,
        blueprint: PatternAwareConfig {
            immediate_candidate_limit: 1,
            habitat_candidate_limit: 1,
            bear_candidate_limit: 1,
            future_market_draws: 1,
        },
    }
}

#[test]
fn config_rejects_zero_work_and_invalid_blueprint() {
    let mut config = tiny_config();
    config.determinizations = 0;
    assert!(matches!(
        config.validate(),
        Err(SearchError::InvalidConfig(_))
    ));

    let mut config = tiny_config();
    config.blueprint.future_market_draws = 0;
    assert!(matches!(config.validate(), Err(SearchError::Simulation(_))));

    let mut config = tiny_late_config();
    config.final_personal_turns = 0;
    assert!(matches!(
        config.validate(),
        Err(SearchError::InvalidConfig(_))
    ));

    config.final_personal_turns = 21;
    assert!(matches!(
        config.validate(),
        Err(SearchError::InvalidConfig(_))
    ));

    let mut config = tiny_wildlife_diverse_config();
    config.wildlife_candidate_limit = 0;
    assert!(matches!(
        config.validate(),
        Err(SearchError::InvalidConfig(_))
    ));

    let mut config = tiny_late_wildlife_diverse_config();
    config.final_personal_turns = 0;
    assert!(matches!(
        config.validate(),
        Err(SearchError::InvalidConfig(_))
    ));

    let mut config = tiny_conservative_config();
    config.terminal.terminal.determinizations = 7;
    assert!(matches!(
        config.validate(),
        Err(SearchError::InvalidConfig(_))
    ));

    let mut config = tiny_conservative_base_config();
    config.terminal.determinizations = 7;
    assert!(matches!(
        config.validate(),
        Err(SearchError::InvalidConfig(_))
    ));
    config.terminal.determinizations = 32;
    assert!(config.validate().is_ok());

    let mut config = tiny_conservative_focused_config();
    config.terminal.wildlife_candidate_limit = 0;
    assert!(matches!(
        config.validate(),
        Err(SearchError::InvalidConfig(_))
    ));
    config.terminal.wildlife_candidate_limit = 1;
    config.terminal.terminal.determinizations = 7;
    assert!(matches!(
        config.validate(),
        Err(SearchError::InvalidConfig(_))
    ));

    let mut config = tiny_conservative_beam_config();
    config.determinizations = 8;
    assert!(config.validate().is_ok());
    config.determinizations = 32;
    assert!(config.validate().is_ok());
    config.determinizations = 7;
    assert!(matches!(
        config.validate(),
        Err(SearchError::InvalidConfig(_))
    ));
    config = tiny_conservative_beam_config();
    config.beam_width = 0;
    assert!(matches!(
        config.validate(),
        Err(SearchError::InvalidConfig(_))
    ));
}

#[test]
fn strategy_id_captures_sampling_and_blueprint_configuration() {
    assert_eq!(
        tiny_config().strategy_id(),
        "terminal-policy-improvement-v1-r1-k2-h1-b1-m2"
    );
    assert_eq!(
        tiny_late_config().strategy_id(),
        "late-terminal-policy-improvement-v1-t4-r1-k2-h1-b1-m2"
    );
    assert_eq!(
        tiny_wildlife_diverse_config().strategy_id(),
        "wildlife-diverse-terminal-policy-improvement-v1-r1-k2-h1-b1-w1-m2"
    );
    assert_eq!(
        tiny_late_wildlife_diverse_config().strategy_id(),
        "late-wildlife-diverse-policy-improvement-v1-t4-r1-k2-h1-b1-w1-m2"
    );
    assert_eq!(
        tiny_conservative_config().strategy_id(),
        "late-conservative-policy-improvement-v1-t1-r8-k2-h1-b1-w1-m2-c90"
    );
    assert_eq!(
        tiny_conservative_base_config().strategy_id(),
        "late-conservative-base-policy-improvement-v1-t1-r8-k2-h1-b1-m2-c90"
    );
    assert_eq!(
        tiny_conservative_focused_config().strategy_id(),
        "late-conservative-wildlife-focused-policy-improvement-v1-t1-r8-k2-h1-b1-fox1-m2-c90"
    );
    assert_eq!(
        tiny_conservative_beam_config().strategy_id(),
        "late-conservative-focal-beam-v1-t1-r4-b1-k1-h1-b1-w1-m1-c90"
    );
    let mut r8_beam = tiny_conservative_beam_config();
    r8_beam.determinizations = 8;
    r8_beam.beam_width = 16;
    assert_eq!(
        r8_beam.strategy_id(),
        "late-conservative-focal-beam-v1-t1-r8-b16-k1-h1-b1-w1-m1-c90"
    );
    let mut r32 = tiny_conservative_base_config();
    r32.terminal.determinizations = 32;
    assert_eq!(
        r32.strategy_id(),
        "late-conservative-base-policy-improvement-v1-t1-r32-k2-h1-b1-m2-c90"
    );
}

#[test]
fn late_terminal_policy_matches_blueprint_before_cutoff_and_terminal_at_cutoff() {
    let game_seed = GameSeed::from_u64(29);
    let mut game = GameState::new(GameConfig::research_aaaaa(2).unwrap(), game_seed).unwrap();
    let strategy = LateTerminalPolicyImprovementStrategy::new(tiny_late_config()).unwrap();
    let mut expected_rng = strategy_rng(game_seed, 0, PATTERN_AWARE_STRATEGY_ID);
    let mut actual_rng = strategy_rng(game_seed, 0, PATTERN_AWARE_STRATEGY_ID);
    let prelude = MarketPrelude {
        replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
        wildlife_wipes: Vec::new(),
    };
    let expected =
        select_pattern_action(&game, &prelude, tiny_config().blueprint, &mut expected_rng).unwrap();

    assert!(!strategy.uses_terminal_search(&game));
    assert_eq!(
        strategy.select_action(&game, &mut actual_rng).unwrap(),
        expected
    );

    let mut rollout_rng = rollout_rng(GameSeed::from_u64(30));
    play_pattern_plies(&mut game, 32, tiny_config().blueprint, &mut rollout_rng).unwrap();
    let expected = strategy
        .terminal
        .select_action_deterministic(&game)
        .unwrap();

    assert!(strategy.uses_terminal_search(&game));
    assert_eq!(
        strategy.select_action(&game, &mut actual_rng).unwrap(),
        expected
    );
}

#[test]
fn late_terminal_match_is_reproducible_and_replayable() {
    let mut config = tiny_late_config();
    config.final_personal_turns = 1;
    let strategy = LateTerminalPolicyImprovementStrategy::new(config).unwrap();
    let game_config = GameConfig::research_aaaaa(2).unwrap();
    let seed = GameSeed::from_u64(31);

    let left = strategy.play_match(game_config, seed).unwrap();
    let right = strategy.play_match(game_config, seed).unwrap();

    assert_eq!(left.scores, right.scores);
    assert_eq!(left.replay, right.replay);
    left.replay.play().unwrap();
}

#[test]
fn late_wildlife_diverse_policy_matches_blueprint_before_cutoff_and_is_replayable() {
    let game_seed = GameSeed::from_u64(43);
    let game_config = GameConfig::research_aaaaa(2).unwrap();
    let game = GameState::new(game_config, game_seed).unwrap();
    let strategy =
        LateWildlifeDiversePolicyImprovementStrategy::new(tiny_late_wildlife_diverse_config())
            .unwrap();
    let mut expected_rng = strategy_rng(game_seed, 0, PATTERN_AWARE_STRATEGY_ID);
    let mut actual_rng = strategy_rng(game_seed, 0, PATTERN_AWARE_STRATEGY_ID);
    let prelude = MarketPrelude {
        replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
        wildlife_wipes: Vec::new(),
    };
    let expected =
        select_pattern_action(&game, &prelude, tiny_config().blueprint, &mut expected_rng).unwrap();

    assert!(!strategy.uses_terminal_search(&game));
    assert_eq!(
        strategy.select_action(&game, &mut actual_rng).unwrap(),
        expected
    );

    let mut terminal_config = tiny_late_wildlife_diverse_config();
    terminal_config.final_personal_turns = 20;
    let terminal_strategy =
        LateWildlifeDiversePolicyImprovementStrategy::new(terminal_config).unwrap();
    let action = terminal_strategy
        .select_action(&game, &mut actual_rng)
        .unwrap();
    game.transition(&action).unwrap();

    let mut replay_config = tiny_late_wildlife_diverse_config();
    replay_config.final_personal_turns = 1;
    let replay_strategy = LateWildlifeDiversePolicyImprovementStrategy::new(replay_config).unwrap();
    let left = replay_strategy.play_match(game_config, game_seed).unwrap();
    let right = replay_strategy.play_match(game_config, game_seed).unwrap();
    assert_eq!(left.scores, right.scores);
    assert_eq!(left.replay, right.replay);
    left.replay.play().unwrap();
}

#[test]
fn terminal_policy_improvement_is_legal_reproducible_and_terminal_scored() {
    let game = GameState::new(
        GameConfig::research_aaaaa(2).unwrap(),
        GameSeed::from_u64(27),
    )
    .unwrap();
    let strategy = TerminalPolicyImprovementStrategy::new(tiny_config()).unwrap();

    let left = strategy.rank_and_select_deterministic(&game).unwrap();
    let right = strategy.rank_and_select_deterministic(&game).unwrap();
    let replayed = strategy
        .select_from_recorded_ranking_deterministic(&game, &left.0)
        .unwrap();

    assert_eq!(left, right);
    assert_eq!(replayed, left.1);
    assert!((2..=4).contains(&left.0.len()));
    assert!(
        left.0
            .windows(2)
            .all(|pair| pair[0].mean_leaf_score >= pair[1].mean_leaf_score)
    );
    assert!(
        left.0
            .iter()
            .all(|candidate| candidate.mean_leaf_score >= f64::from(candidate.immediate_score))
    );
    game.transition(&left.1).unwrap();
}

#[test]
fn paired_lcb_accepts_consistent_advantage_and_rejects_noisy_mean() {
    let anchor = [100.0; 8];
    let consistent = [101.0; 8];
    let noisy = [112.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0];

    let (consistent_mean, _, consistent_lcb) =
        paired_advantage_lcb90(&consistent, &anchor).unwrap();
    let (noisy_mean, _, noisy_lcb) = paired_advantage_lcb90(&noisy, &anchor).unwrap();

    assert_eq!(consistent_mean, 1.0);
    assert_eq!(consistent_lcb, 1.0);
    assert!(noisy_mean > 0.0);
    assert!(noisy_lcb < 0.0);
}

#[test]
fn paired_lcb_uses_the_registered_r32_student_t_critical() {
    let anchor = [100.0; 32];
    let candidate: [f64; 32] = std::array::from_fn(|index| if index == 0 { 104.0 } else { 101.0 });

    let (mean, standard_error, lower_bound) = paired_advantage_lcb90(&candidate, &anchor).unwrap();

    assert!((lower_bound - (mean - ONE_SIDED_T_90_DF_31 * standard_error)).abs() < 1e-12);
}

#[test]
fn paired_lcb_uses_the_registered_r4_student_t_critical() {
    let anchor = [100.0; 4];
    let candidate = [104.0, 101.0, 101.0, 101.0];

    let (mean, standard_error, lower_bound) = paired_advantage_lcb90(&candidate, &anchor).unwrap();

    assert!((lower_bound - (mean - ONE_SIDED_T_90_DF_3 * standard_error)).abs() < 1e-12);
}

#[test]
fn paired_lcb_uses_the_registered_r8_student_t_critical() {
    let anchor = [100.0; 8];
    let candidate = [104.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0];

    let (mean, standard_error, lower_bound) = paired_advantage_lcb90(&candidate, &anchor).unwrap();

    assert!((lower_bound - (mean - ONE_SIDED_T_90_DF_7 * standard_error)).abs() < 1e-12);
}

#[test]
fn conservative_late_match_is_reproducible_and_replayable() {
    let strategy =
        LateConservativePolicyImprovementStrategy::new(tiny_conservative_config()).unwrap();
    let game_config = GameConfig::research_aaaaa(2).unwrap();
    let seed = GameSeed::from_u64(44);
    let game = GameState::new(game_config, seed).unwrap();
    let prelude = MarketPrelude {
        replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
        wildlife_wipes: Vec::new(),
    };
    let mut expected_rng = strategy_rng(seed, 0, PATTERN_AWARE_STRATEGY_ID);
    let mut actual_rng = strategy_rng(seed, 0, PATTERN_AWARE_STRATEGY_ID);
    let expected = select_pattern_action(
        &game,
        &prelude,
        tiny_conservative_config().terminal.terminal.blueprint,
        &mut expected_rng,
    )
    .unwrap();
    assert!(!strategy.uses_terminal_search(&game));
    assert_eq!(
        strategy.select_action(&game, &mut actual_rng).unwrap(),
        expected
    );

    let left = strategy.play_match(game_config, seed).unwrap();
    let right = strategy.play_match(game_config, seed).unwrap();

    assert_eq!(left.scores, right.scores);
    assert_eq!(left.replay, right.replay);
    left.replay.play().unwrap();
}

#[test]
fn conservative_base_match_is_reproducible_and_replayable() {
    let strategy =
        LateConservativeBasePolicyImprovementStrategy::new(tiny_conservative_base_config())
            .unwrap();
    let game_config = GameConfig::research_aaaaa(2).unwrap();
    let seed = GameSeed::from_u64(45);

    let left = strategy.play_match(game_config, seed).unwrap();
    let right = strategy.play_match(game_config, seed).unwrap();

    assert_eq!(left.scores, right.scores);
    assert_eq!(left.replay, right.replay);
    left.replay.play().unwrap();
}

#[test]
fn conservative_focused_match_is_reproducible_legal_and_replayable() {
    let strategy = LateConservativeWildlifeFocusedPolicyImprovementStrategy::new(
        tiny_conservative_focused_config(),
    )
    .unwrap();
    let game_config = GameConfig::research_aaaaa(2).unwrap();
    let seed = GameSeed::from_u64(46);
    let opening = GameState::new(game_config, seed).unwrap();
    let prelude = MarketPrelude {
        replace_three_of_a_kind: opening.market().three_of_a_kind().is_some(),
        wildlife_wipes: Vec::new(),
    };
    let mut expected_rng = strategy_rng(seed, 0, PATTERN_AWARE_STRATEGY_ID);
    let mut actual_rng = strategy_rng(seed, 0, PATTERN_AWARE_STRATEGY_ID);
    let expected = select_pattern_action(
        &opening,
        &prelude,
        tiny_conservative_focused_config()
            .terminal
            .terminal
            .blueprint,
        &mut expected_rng,
    )
    .unwrap();

    assert!(!strategy.uses_terminal_search(&opening));
    assert_eq!(
        strategy.select_action(&opening, &mut actual_rng).unwrap(),
        expected
    );

    let left = strategy.play_match(game_config, seed).unwrap();
    let right = strategy.play_match(game_config, seed).unwrap();
    assert_eq!(left.scores, right.scores);
    assert_eq!(left.replay, right.replay);
    left.replay.play().unwrap();
}

#[test]
fn conservative_focal_beam_is_hidden_order_invariant_and_replayable() {
    let strategy = LateConservativeFocalBeamStrategy::new(tiny_conservative_beam_config()).unwrap();
    let game_config = GameConfig::research_aaaaa(2).unwrap();
    let (seed, game) = (47..60)
        .find_map(|numeric_seed| {
            let seed = GameSeed::from_u64(numeric_seed);
            let mut game = GameState::new(game_config, seed).unwrap();
            let mut policy_rng = rollout_rng(GameSeed::from_u64(numeric_seed + 100));
            play_pattern_plies(
                &mut game,
                38,
                tiny_conservative_beam_config().blueprint,
                &mut policy_rng,
            )
            .unwrap();
            game.market()
                .three_of_a_kind()
                .is_none()
                .then_some((seed, game))
        })
        .expect("fixture seeds include a settled final-turn market");
    let mut redetermined = game.clone();
    redetermined.redeterminize_hidden(GameSeed::from_u64(149));
    let mut left_rng = strategy_rng(seed, 0, PATTERN_AWARE_STRATEGY_ID);
    let mut right_rng = strategy_rng(seed, 0, PATTERN_AWARE_STRATEGY_ID);

    assert!(strategy.uses_beam(&game));
    assert_eq!(
        strategy.select_action(&game, &mut left_rng).unwrap(),
        strategy
            .select_action(&redetermined, &mut right_rng)
            .unwrap()
    );

    let left = strategy.play_match(game_config, seed).unwrap();
    let right = strategy.play_match(game_config, seed).unwrap();
    assert_eq!(left.scores, right.scores);
    assert_eq!(left.replay, right.replay);
    left.replay.play().unwrap();
}
