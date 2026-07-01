use cascadia_game::{GameConfig, GameSeed, score_game};

use super::*;

fn tiny_strategy() -> PerfectInformationPatternOracleStrategy {
    PerfectInformationPatternOracleStrategy::new(PerfectInformationPatternOracleConfig {
        blueprint: PatternAwareConfig {
            immediate_candidate_limit: 1,
            habitat_candidate_limit: 1,
            bear_candidate_limit: 1,
            future_market_draws: 1,
        },
        wildlife_candidate_limit: None,
    })
    .unwrap()
}

fn tiny_beam_strategy(final_personal_turns: u16) -> PerfectInformationFocalBeamStrategy {
    PerfectInformationFocalBeamStrategy::new(PerfectInformationFocalBeamConfig {
        blueprint: PatternAwareConfig {
            immediate_candidate_limit: 1,
            habitat_candidate_limit: 1,
            bear_candidate_limit: 1,
            future_market_draws: 1,
        },
        wildlife_candidate_limit: 1,
        beam_width: 2,
        final_personal_turns,
    })
    .unwrap()
}

fn tiny_portfolio_beam_strategy(
    final_personal_turns: u16,
) -> PerfectInformationPortfolioBeamStrategy {
    PerfectInformationPortfolioBeamStrategy::new(PerfectInformationPortfolioBeamConfig {
        blueprint: PatternAwareConfig {
            immediate_candidate_limit: 1,
            habitat_candidate_limit: 1,
            bear_candidate_limit: 1,
            future_market_draws: 1,
        },
        wildlife_candidate_limit: 1,
        beam_width: 2,
        final_personal_turns,
    })
    .unwrap()
}

fn tiny_root_diverse_beam_strategy(
    root_wildlife_candidate_limit: usize,
    final_personal_turns: u16,
) -> PerfectInformationRootDiverseBeamStrategy {
    PerfectInformationRootDiverseBeamStrategy::new(PerfectInformationRootDiverseBeamConfig {
        blueprint: PatternAwareConfig {
            immediate_candidate_limit: 1,
            habitat_candidate_limit: 1,
            bear_candidate_limit: 1,
            future_market_draws: 1,
        },
        root_wildlife_candidate_limit,
        future_wildlife_candidate_limit: 1,
        beam_width: 2,
        final_personal_turns,
    })
    .unwrap()
}

#[test]
fn oracle_configuration_names_and_validates_diverse_frontiers() {
    let blueprint = PatternAwareConfig {
        immediate_candidate_limit: 2,
        habitat_candidate_limit: 1,
        bear_candidate_limit: 1,
        future_market_draws: 2,
    };
    let base = PerfectInformationPatternOracleConfig {
        blueprint,
        wildlife_candidate_limit: None,
    };
    let diverse = PerfectInformationPatternOracleConfig {
        blueprint,
        wildlife_candidate_limit: Some(2),
    };

    assert_eq!(
        base.strategy_id(),
        "perfect-information-pattern-oracle-v1-k2-h1-b1-m2"
    );
    assert_eq!(
        diverse.strategy_id(),
        "perfect-information-pattern-oracle-v1-k2-h1-b1-w2-m2"
    );
    assert!(
        PerfectInformationPatternOracleConfig {
            blueprint,
            wildlife_candidate_limit: Some(0),
        }
        .validate()
        .is_err()
    );

    let beam = PerfectInformationFocalBeamConfig {
        blueprint,
        wildlife_candidate_limit: 2,
        beam_width: 16,
        final_personal_turns: 5,
    };
    assert_eq!(
        beam.strategy_id(),
        "perfect-information-focal-beam-v1-t5-b16-k2-h1-b1-w2-m2"
    );
    let mut wider_capacity = beam;
    wider_capacity.beam_width = 32;
    assert_eq!(
        wider_capacity.strategy_id(),
        "perfect-information-focal-beam-v1-t5-b32-k2-h1-b1-w2-m2"
    );
    assert_ne!(beam.strategy_id(), wider_capacity.strategy_id());
    let mut wider_beam = beam;
    wider_beam.wildlife_candidate_limit = 4;
    assert_eq!(
        wider_beam.strategy_id(),
        "perfect-information-focal-beam-v1-t5-b16-k2-h1-b1-w4-m2"
    );
    assert_ne!(beam.strategy_id(), wider_beam.strategy_id());
    let mut invalid = beam;
    invalid.beam_width = 0;
    assert!(invalid.validate().is_err());
    invalid = beam;
    invalid.wildlife_candidate_limit = 0;
    assert!(invalid.validate().is_err());
    invalid = beam;
    invalid.final_personal_turns = 0;
    assert!(invalid.validate().is_err());

    let portfolio = PerfectInformationPortfolioBeamConfig {
        blueprint,
        wildlife_candidate_limit: 2,
        beam_width: 16,
        final_personal_turns: 5,
    };
    assert_eq!(
        portfolio.strategy_id(),
        "perfect-information-portfolio-beam-v1-t5-b16-k2-h1-b1-w2-m2"
    );
    assert!(portfolio.validate().is_ok());

    let root_diverse = PerfectInformationRootDiverseBeamConfig {
        blueprint,
        root_wildlife_candidate_limit: 4,
        future_wildlife_candidate_limit: 2,
        beam_width: 16,
        final_personal_turns: 5,
    };
    assert_eq!(
        root_diverse.strategy_id(),
        "perfect-information-root-diverse-beam-v1-t5-b16-rootw4-futurew2-k2-h1-b1-m2"
    );
    assert!(root_diverse.validate().is_ok());
    let mut invalid_root = root_diverse;
    invalid_root.root_wildlife_candidate_limit = 0;
    assert!(invalid_root.validate().is_err());
    invalid_root = root_diverse;
    invalid_root.future_wildlife_candidate_limit = 0;
    assert!(invalid_root.validate().is_err());
}

#[test]
fn portfolio_retention_is_deterministic_unique_bounded_and_dimension_ordered() {
    let dimensions = (0..24)
        .map(|index| {
            let mut values = [0.0; BEAM_DIMENSIONS];
            values[0] = (100 - index) as f64;
            for (dimension, value) in values.iter_mut().enumerate().skip(1) {
                *value = if index == dimension * 2 || index == dimension * 2 + 1 {
                    1_000.0 - index as f64
                } else {
                    index as f64
                };
            }
            values
        })
        .collect::<Vec<_>>();

    let first = retention_indices(&dimensions, 16, BeamRetention::Portfolio);
    let second = retention_indices(&dimensions, 16, BeamRetention::Portfolio);
    let mut unique = first.clone();
    unique.sort_unstable();
    unique.dedup();

    assert_eq!(first, second);
    assert_eq!(first.len(), 16);
    assert_eq!(unique.len(), first.len());
    assert!(first.iter().all(|index| *index < dimensions.len()));
    assert_eq!(&first[..2], &[0, 1]);
    assert_eq!(&first[2..4], &[2, 3]);
    assert_eq!(&first[4..6], &[4, 5]);
}

#[test]
fn portfolio_retention_fills_unused_capacity_in_scalar_order() {
    let mut dimensions = [[0.0; BEAM_DIMENSIONS]; 5];
    for (index, values) in dimensions.iter_mut().enumerate() {
        values[0] = 9.0 - index as f64;
        for value in values.iter_mut().skip(1) {
            *value = match index {
                0 => 100.0,
                1 => 99.0,
                _ => index as f64,
            };
        }
    }

    assert_eq!(
        retention_indices(&dimensions, 5, BeamRetention::Portfolio),
        vec![0, 1, 2, 3, 4]
    );
}

#[test]
fn width_one_portfolio_retention_matches_scalar_retention() {
    let dimensions = [
        [1.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    assert_eq!(
        retention_indices(&dimensions, 1, BeamRetention::Portfolio),
        retention_indices(&dimensions, 1, BeamRetention::Scalar)
    );
}

#[test]
fn oracle_ranking_is_reproducible_legal_and_terminal_scored() {
    let game = GameState::new(
        GameConfig::research_aaaaa(4).unwrap(),
        GameSeed::from_u64(29),
    )
    .unwrap();
    let strategy = tiny_strategy();

    let first = strategy.rank_actions_deterministic(&game).unwrap();
    let second = strategy.rank_actions_deterministic(&game).unwrap();

    assert_eq!(first, second);
    assert!(!first.is_empty());
    assert!(first.windows(2).all(|pair| {
        pair[0].mean_leaf_score > pair[1].mean_leaf_score
            || (pair[0].mean_leaf_score == pair[1].mean_leaf_score
                && pair[0].immediate_score >= pair[1].immediate_score)
    }));
    for candidate in first {
        assert_eq!(candidate.leaf_score_stddev, 0.0);
        assert_eq!(candidate.mean_leaf_score.fract(), 0.0);
        game.transition(&candidate.action).unwrap();
    }
}

#[test]
fn oracle_selection_is_reproducible_and_uses_ranked_maximum() {
    let game = GameState::new(
        GameConfig::research_aaaaa(4).unwrap(),
        GameSeed::from_u64(31),
    )
    .unwrap();
    let strategy = tiny_strategy();

    let (ranked, action) = strategy.rank_and_select_deterministic(&game).unwrap();
    let repeated = strategy.select_action_deterministic(&game).unwrap();

    assert_eq!(action, repeated);
    assert!(
        ranked
            .iter()
            .take_while(|candidate| { candidate.mean_leaf_score == ranked[0].mean_leaf_score })
            .any(|candidate| candidate.action == action)
    );
    game.transition(&action).unwrap();
}

#[test]
fn focal_beam_is_reproducible_legal_and_uses_ranked_maximum() {
    let game = GameState::new(
        GameConfig::research_aaaaa(2).unwrap(),
        GameSeed::from_u64(32),
    )
    .unwrap();
    let strategy = tiny_beam_strategy(20);

    let left = strategy.rank_and_select_deterministic(&game).unwrap();
    let right = strategy.rank_and_select_deterministic(&game).unwrap();

    assert_eq!(left, right);
    assert!(left.0.windows(2).all(|pair| {
        pair[0].mean_leaf_score > pair[1].mean_leaf_score
            || (pair[0].mean_leaf_score == pair[1].mean_leaf_score
                && pair[0].immediate_score >= pair[1].immediate_score)
    }));
    game.transition(&left.1).unwrap();
}

#[test]
fn focal_beam_matches_one_step_with_one_focal_turn_remaining() {
    let game_seed = GameSeed::from_u64(33);
    let mut game = GameState::new(GameConfig::research_aaaaa(2).unwrap(), game_seed).unwrap();
    let blueprint = PatternAwareConfig {
        immediate_candidate_limit: 1,
        habitat_candidate_limit: 1,
        bear_candidate_limit: 1,
        future_market_draws: 1,
    };
    let mut policy_rng = rollout_rng(GameSeed::from_u64(34));
    play_pattern_plies(&mut game, 38, blueprint, &mut policy_rng).unwrap();
    assert_eq!(game.current_player(), 0);
    assert_eq!(game.turns_remaining_for_player(0), 1);

    let one_step =
        PerfectInformationPatternOracleStrategy::new(PerfectInformationPatternOracleConfig {
            blueprint,
            wildlife_candidate_limit: Some(1),
        })
        .unwrap();
    let beam = tiny_beam_strategy(1);

    assert_eq!(
        one_step.select_action_deterministic(&game).unwrap(),
        beam.select_action_deterministic(&game).unwrap()
    );
    assert_eq!(
        beam.select_action_deterministic(&game).unwrap(),
        tiny_portfolio_beam_strategy(1)
            .select_action_deterministic(&game)
            .unwrap()
    );
}

#[test]
fn portfolio_beam_completes_deterministically_and_replays_legally() {
    let mut initial = GameState::new(
        GameConfig::research_aaaaa(2).unwrap(),
        GameSeed::from_u64(35),
    )
    .unwrap();
    let blueprint = PatternAwareConfig {
        immediate_candidate_limit: 1,
        habitat_candidate_limit: 1,
        bear_candidate_limit: 1,
        future_market_draws: 1,
    };
    let mut policy_rng = rollout_rng(GameSeed::from_u64(36));
    play_pattern_plies(&mut initial, 36, blueprint, &mut policy_rng).unwrap();
    let strategy = tiny_portfolio_beam_strategy(2);

    let play = |mut game: GameState| {
        let mut actions = Vec::new();
        while !game.is_game_over() {
            let action = strategy.select_action_deterministic(&game).unwrap();
            game.apply(&action).unwrap();
            actions.push(action);
        }
        (actions, score_game(&game))
    };
    let (first_actions, first_scores) = play(initial.clone());
    let (second_actions, second_scores) = play(initial.clone());

    assert_eq!(first_actions, second_actions);
    assert_eq!(first_scores, second_scores);
    for action in first_actions {
        initial.apply(&action).unwrap();
    }
    assert!(initial.is_game_over());
    assert_eq!(score_game(&initial), first_scores);
}

#[test]
fn root_diverse_beam_matches_w2_before_cutoff_and_only_widens_cutoff_root() {
    let game_seed = GameSeed::from_u64(37);
    let mut game = GameState::new(GameConfig::research_aaaaa(2).unwrap(), game_seed).unwrap();
    let baseline = tiny_beam_strategy(2);
    let treatment = tiny_root_diverse_beam_strategy(2, 2);

    assert_eq!(
        baseline.select_action_deterministic(&game).unwrap(),
        treatment.select_action_deterministic(&game).unwrap()
    );

    let blueprint = PatternAwareConfig {
        immediate_candidate_limit: 1,
        habitat_candidate_limit: 1,
        bear_candidate_limit: 1,
        future_market_draws: 1,
    };
    let mut policy_rng = rollout_rng(GameSeed::from_u64(38));
    play_pattern_plies(&mut game, 38, blueprint, &mut policy_rng).unwrap();
    let baseline_ranked = baseline.rank_actions_deterministic(&game).unwrap();
    let treatment_ranked = treatment.rank_actions_deterministic(&game).unwrap();

    assert!(baseline_ranked.iter().all(|candidate| {
        treatment_ranked
            .iter()
            .any(|item| item.action == candidate.action)
    }));
    assert!(treatment_ranked.len() >= baseline_ranked.len());
}

#[test]
fn sibling_opponent_replay_matches_direct_policy_execution() {
    let game_seed = GameSeed::from_u64(39);
    let mut game = GameState::new(GameConfig::research_aaaaa(4).unwrap(), game_seed).unwrap();
    let blueprint = PatternAwareConfig {
        immediate_candidate_limit: 4,
        habitat_candidate_limit: 4,
        bear_candidate_limit: 4,
        future_market_draws: 1,
    };
    let mut trajectory_rng = rollout_rng(GameSeed::from_u64(40));
    play_pattern_plies(&mut game, 64, blueprint, &mut trajectory_rng).unwrap();
    let focal_seat = game.current_player();
    let candidates = rank_wildlife_diverse_pattern_frontier_actions(
        &game,
        &MarketPrelude::default(),
        blueprint,
        2,
    )
    .unwrap();
    let strategy = PerfectInformationFocalBeamStrategy::new(PerfectInformationFocalBeamConfig {
        blueprint,
        wildlife_candidate_limit: 2,
        beam_width: 4,
        final_personal_turns: 4,
    })
    .unwrap();
    let policy_rng = rollout_rng(GameSeed::from_u64(41));
    let mut children = candidates
        .iter()
        .map(|candidate| {
            let mut child = BeamNode {
                game: game.clone(),
                policy_rng: policy_rng.clone(),
            };
            child.game.apply(&candidate.action).unwrap();
            child
        })
        .collect::<Vec<_>>();
    let mut pair = None;
    'outer: for left in 0..children.len() {
        for right in left + 1..children.len() {
            if PerfectInformationFocalBeamStrategy::pattern_policy_observation(
                &children[left].game,
                focal_seat,
            ) == PerfectInformationFocalBeamStrategy::pattern_policy_observation(
                &children[right].game,
                focal_seat,
            ) {
                pair = Some((left, right));
                break 'outer;
            }
        }
    }
    let (left, right) = pair.expect("diverse frontier should contain replayable siblings");
    let replay = strategy
        .advance_opponents_recording(&mut children[left], focal_seat)
        .unwrap();
    let mut direct = BeamNode {
        game: children[right].game.clone(),
        policy_rng: policy_rng.clone(),
    };
    strategy.advance_opponents(&mut direct, focal_seat).unwrap();

    assert!(
        PerfectInformationFocalBeamStrategy::replay_opponents(
            &mut children[right],
            focal_seat,
            &replay,
        )
        .unwrap()
    );
    assert_eq!(children[right].game, direct.game);
}

#[test]
fn sibling_replay_preserves_exact_beam_value() {
    let game_seed = GameSeed::from_u64(42);
    let mut game = GameState::new(GameConfig::research_aaaaa(2).unwrap(), game_seed).unwrap();
    let strategy = tiny_beam_strategy(2);
    let mut trajectory_rng = rollout_rng(GameSeed::from_u64(43));
    play_pattern_plies(
        &mut game,
        36,
        strategy.config.blueprint,
        &mut trajectory_rng,
    )
    .unwrap();
    let candidates = rank_wildlife_diverse_pattern_frontier_actions(
        &game,
        &MarketPrelude::default(),
        strategy.config.blueprint,
        strategy.config.wildlife_candidate_limit,
    )
    .unwrap();
    let root = &candidates[0].action;
    let continuation_seed = GameSeed::from_u64(44);

    let replayed = strategy
        .evaluate_root_candidate_with_retention_impl(
            &game,
            root,
            game.current_player(),
            continuation_seed,
            BeamRetention::Scalar,
            true,
        )
        .unwrap();
    let direct = strategy
        .evaluate_root_candidate_with_retention_impl(
            &game,
            root,
            game.current_player(),
            continuation_seed,
            BeamRetention::Scalar,
            false,
        )
        .unwrap();

    assert_eq!(replayed, direct);
}

#[test]
fn public_beam_value_batches_ignore_hidden_order_and_validate_work() {
    let mut game = GameState::new(
        GameConfig::research_aaaaa(2).unwrap(),
        GameSeed::from_u64(45),
    )
    .unwrap();
    let blueprint = PatternAwareConfig {
        immediate_candidate_limit: 1,
        habitat_candidate_limit: 1,
        bear_candidate_limit: 1,
        future_market_draws: 1,
    };
    let mut policy_rng = rollout_rng(GameSeed::from_u64(46));
    play_pattern_plies(&mut game, 38, blueprint, &mut policy_rng).unwrap();
    let config = PublicBeamValueProbeConfig {
        blueprint,
        wildlife_candidate_limit: 1,
        beam_width: 2,
        final_personal_turns: 1,
        determinizations_per_batch: 1,
        batches: 2,
    };
    let mut redetermined = game.clone();
    redetermined.redeterminize_hidden(GameSeed::from_u64(47));

    assert_eq!(
        evaluate_public_beam_value_batches(&game, config).unwrap(),
        evaluate_public_beam_value_batches(&redetermined, config).unwrap()
    );
    let mut invalid = config;
    invalid.determinizations_per_batch = 0;
    assert!(invalid.validate().is_err());
    invalid = config;
    invalid.batches = 1;
    assert!(invalid.validate().is_err());
}
