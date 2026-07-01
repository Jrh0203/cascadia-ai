use cascadia_data::PositionRecord;
use cascadia_game::{GameConfig, GameSeed, GameState, MarketPrelude};
use cascadia_model::Prediction;
use cascadia_sim::rank_greedy_actions;

use super::*;

struct CountPredictor;
struct NonFinitePredictor;
struct CountRankingPredictor;
struct NonFiniteRankingPredictor;
struct ReverseOrderRankingPredictor;

impl Predictor for CountPredictor {
    fn predict(&mut self, records: &[PositionRecord]) -> Result<Vec<Prediction>, SearchError> {
        Ok(records
            .iter()
            .map(|record| {
                let mut prediction = [0.0; 11];
                prediction[0] = f32::from(record.board_counts[0]);
                prediction
            })
            .collect())
    }
}

impl Predictor for NonFinitePredictor {
    fn predict(&mut self, records: &[PositionRecord]) -> Result<Vec<Prediction>, SearchError> {
        Ok(vec![[f32::NAN; 11]; records.len()])
    }
}

impl RankingPredictor for CountRankingPredictor {
    fn predict_scores(&mut self, records: &[PositionRecord]) -> Result<Vec<f32>, SearchError> {
        Ok(records
            .iter()
            .map(|record| f32::from(record.board_counts[0]))
            .collect())
    }
}

impl RankingPredictor for NonFiniteRankingPredictor {
    fn predict_scores(&mut self, records: &[PositionRecord]) -> Result<Vec<f32>, SearchError> {
        Ok(vec![f32::NAN; records.len()])
    }
}

impl RankingPredictor for ReverseOrderRankingPredictor {
    fn predict_scores(&mut self, records: &[PositionRecord]) -> Result<Vec<f32>, SearchError> {
        Ok((0..records.len()).map(|index| index as f32).collect())
    }
}

#[test]
fn model_ranked_action_is_legal_and_deterministic() {
    let seed = GameSeed::from_u64(9);
    let game = GameState::new(GameConfig::research_aaaaa(2).unwrap(), seed).unwrap();
    let mut left = MlxValueStrategy::with_predictor(CountPredictor);
    let mut right = MlxValueStrategy::with_predictor(CountPredictor);
    let mut left_rng = strategy_rng(seed, 0);
    let mut right_rng = strategy_rng(seed, 0);
    let left_action = left.select_action(&game, &mut left_rng).unwrap();
    let right_action = right.select_action(&game, &mut right_rng).unwrap();
    assert_eq!(left_action, right_action);
    game.transition(&left_action).unwrap();
}

#[test]
fn mlx_value_leaf_lookahead_is_legal_reproducible_and_ranked() {
    let seed = GameSeed::from_u64(10);
    let game = GameState::new(GameConfig::research_aaaaa(2).unwrap(), seed).unwrap();
    let config = MlxValueLeafLookaheadConfig {
        immediate_candidate_limit: 3,
        habitat_candidate_limit: 2,
        determinizations: 2,
        greedy_plies: 1,
    };
    let mut left = MlxValueLeafLookaheadStrategy::with_predictor(CountPredictor, config).unwrap();
    let mut right = MlxValueLeafLookaheadStrategy::with_predictor(CountPredictor, config).unwrap();
    let left_ranked = left.rank_actions(&game).unwrap();
    let right_ranked = right.rank_actions(&game).unwrap();

    assert_eq!(left_ranked, right_ranked);
    assert!((3..=5).contains(&left_ranked.len()));
    assert!(
        left_ranked
            .windows(2)
            .all(|pair| pair[0].mean_leaf_score >= pair[1].mean_leaf_score)
    );
    let action = left.select_action(&game).unwrap();
    game.transition(&action).unwrap();
}

#[test]
fn mlx_value_leaf_lookahead_rejects_non_finite_model_output() {
    let game = GameState::new(
        GameConfig::research_aaaaa(2).unwrap(),
        GameSeed::from_u64(12),
    )
    .unwrap();
    let mut strategy = MlxValueLeafLookaheadStrategy::with_predictor(
        NonFinitePredictor,
        MlxValueLeafLookaheadConfig {
            immediate_candidate_limit: 2,
            habitat_candidate_limit: 1,
            determinizations: 1,
            greedy_plies: 1,
        },
    )
    .unwrap();

    assert!(matches!(
        strategy.rank_actions(&game),
        Err(SearchError::NonFinitePrediction { index: 0 })
    ));
}

#[test]
fn determinized_lookahead_is_legal_reproducible_and_ranked() {
    let seed = GameSeed::from_u64(11);
    let game = GameState::new(GameConfig::research_aaaaa(2).unwrap(), seed).unwrap();
    let strategy = DeterminizedLookaheadStrategy::new(DeterminizedLookaheadConfig {
        candidate_limit: 3,
        determinizations: 2,
        greedy_plies: 1,
    })
    .unwrap();
    let left = strategy.rank_actions_deterministic(&game).unwrap();
    let right = strategy.rank_actions_deterministic(&game).unwrap();

    assert_eq!(left, right);
    assert_eq!(left.len(), 3);
    assert!(left.iter().all(|candidate| candidate.immediate_rank <= 3));
    assert!(
        left.windows(2)
            .all(|pair| pair[0].mean_leaf_score >= pair[1].mean_leaf_score)
    );
    for candidate in left {
        game.transition(&candidate.action).unwrap();
    }
}

#[test]
fn bear_candidate_lookahead_is_legal_reproducible_and_bounded() {
    let seed = GameSeed::from_u64(13);
    let game = GameState::new(GameConfig::research_aaaaa(2).unwrap(), seed).unwrap();
    let strategy = BearCandidateLookaheadStrategy::new(BearCandidateLookaheadConfig {
        immediate_candidate_limit: 3,
        bear_candidate_limit: 2,
        determinizations: 2,
        greedy_plies: 1,
    })
    .unwrap();
    let left = strategy.rank_and_select_deterministic(&game).unwrap();
    let right = strategy.rank_and_select_deterministic(&game).unwrap();

    assert_eq!(left, right);
    assert!((3..=5).contains(&left.0.len()));
    assert!(
        left.0
            .windows(2)
            .all(|pair| pair[0].mean_leaf_score >= pair[1].mean_leaf_score)
    );
    game.transition(&left.1).unwrap();
}

#[test]
fn habitat_candidate_lookahead_is_legal_reproducible_and_bounded() {
    let seed = GameSeed::from_u64(14);
    let game = GameState::new(GameConfig::research_aaaaa(2).unwrap(), seed).unwrap();
    let strategy = HabitatCandidateLookaheadStrategy::new(HabitatCandidateLookaheadConfig {
        immediate_candidate_limit: 3,
        habitat_candidate_limit: 2,
        determinizations: 2,
        greedy_plies: 1,
    })
    .unwrap();
    let left = strategy.rank_and_select_deterministic(&game).unwrap();
    let right = strategy.rank_and_select_deterministic(&game).unwrap();

    assert_eq!(left, right);
    assert!((3..=5).contains(&left.0.len()));
    assert!(
        left.0
            .windows(2)
            .all(|pair| pair[0].mean_leaf_score >= pair[1].mean_leaf_score)
    );
    game.transition(&left.1).unwrap();
}

#[test]
fn bear_habitat_candidate_lookahead_is_legal_reproducible_and_bounded() {
    let seed = GameSeed::from_u64(16);
    let game = GameState::new(GameConfig::research_aaaaa(2).unwrap(), seed).unwrap();
    let strategy =
        BearHabitatCandidateLookaheadStrategy::new(BearHabitatCandidateLookaheadConfig {
            immediate_candidate_limit: 3,
            habitat_candidate_limit: 2,
            bear_candidate_limit: 2,
            determinizations: 2,
            greedy_plies: 1,
        })
        .unwrap();
    let left = strategy.rank_and_select_deterministic(&game).unwrap();
    let right = strategy.rank_and_select_deterministic(&game).unwrap();

    assert_eq!(left, right);
    assert!((3..=7).contains(&left.0.len()));
    assert!(
        left.0
            .windows(2)
            .all(|pair| pair[0].mean_leaf_score >= pair[1].mean_leaf_score)
    );
    game.transition(&left.1).unwrap();
}

#[test]
fn mlx_ranking_action_is_legal_and_reproducible() {
    let seed = GameSeed::from_u64(15);
    let game = GameState::new(GameConfig::research_aaaaa(2).unwrap(), seed).unwrap();
    let config = MlxRankingConfig {
        immediate_candidate_limit: 3,
        bear_candidate_limit: 2,
    };
    let mut left = MlxRankingStrategy::with_predictor(CountRankingPredictor, config).unwrap();
    let mut right = MlxRankingStrategy::with_predictor(CountRankingPredictor, config).unwrap();
    let left_action = left.select_action(&game).unwrap();
    let right_action = right.select_action(&game).unwrap();

    assert_eq!(left_action, right_action);
    game.transition(&left_action).unwrap();
}

#[test]
fn mlx_prefiltered_lookahead_is_legal_reproducible_and_bounded() {
    let seed = GameSeed::from_u64(17);
    let game = GameState::new(GameConfig::research_aaaaa(2).unwrap(), seed).unwrap();
    let config = MlxPrefilteredLookaheadConfig {
        immediate_candidate_limit: 3,
        bear_candidate_limit: 2,
        immediate_anchor_limit: 0,
        prefilter_candidate_limit: 2,
        determinizations: 2,
        greedy_plies: 1,
    };
    let mut left =
        MlxPrefilteredLookaheadStrategy::with_predictor(CountRankingPredictor, config).unwrap();
    let mut right =
        MlxPrefilteredLookaheadStrategy::with_predictor(CountRankingPredictor, config).unwrap();
    let left_ranked = left.rank_and_select_deterministic(&game).unwrap();
    let right_ranked = right.rank_and_select_deterministic(&game).unwrap();

    assert_eq!(left_ranked, right_ranked);
    assert_eq!(left_ranked.0.len(), 2);
    assert!(
        left_ranked
            .0
            .windows(2)
            .all(|pair| pair[0].mean_leaf_score >= pair[1].mean_leaf_score)
    );
    game.transition(&left_ranked.1).unwrap();
}

#[test]
fn mlx_prefilter_preserves_immediate_anchors_against_model_order() {
    let game = GameState::new(
        GameConfig::research_aaaaa(2).unwrap(),
        GameSeed::from_u64(18),
    )
    .unwrap();
    let (prelude, _, union) = bear_candidate_union(&game, 3, 2).unwrap();
    let protected: Vec<_> = union
        .iter()
        .take(2)
        .map(|candidate| with_prelude(candidate.action.clone(), &prelude))
        .collect();
    let mut strategy = MlxPrefilteredLookaheadStrategy::with_predictor(
        ReverseOrderRankingPredictor,
        MlxPrefilteredLookaheadConfig {
            immediate_candidate_limit: 3,
            bear_candidate_limit: 2,
            immediate_anchor_limit: 2,
            prefilter_candidate_limit: 3,
            determinizations: 1,
            greedy_plies: 1,
        },
    )
    .unwrap();

    let ranked = strategy.rank_actions(&game).unwrap();
    assert_eq!(ranked.len(), 3);
    assert!(
        protected
            .iter()
            .all(|action| ranked.iter().any(|candidate| &candidate.action == action))
    );
}

#[test]
fn ranking_strategies_reject_non_finite_model_output() {
    let game = GameState::new(
        GameConfig::research_aaaaa(2).unwrap(),
        GameSeed::from_u64(19),
    )
    .unwrap();
    let mut strategy = MlxPrefilteredLookaheadStrategy::with_predictor(
        NonFiniteRankingPredictor,
        MlxPrefilteredLookaheadConfig {
            immediate_candidate_limit: 2,
            bear_candidate_limit: 1,
            immediate_anchor_limit: 0,
            prefilter_candidate_limit: 2,
            determinizations: 1,
            greedy_plies: 1,
        },
    )
    .unwrap();

    assert!(matches!(
        strategy.rank_actions(&game),
        Err(SearchError::NonFinitePrediction { index: 0 })
    ));
}

#[test]
fn mlx_habitat_prefilter_is_legal_reproducible_and_bounded() {
    let seed = GameSeed::from_u64(20);
    let game = GameState::new(GameConfig::research_aaaaa(2).unwrap(), seed).unwrap();
    let config = MlxHabitatPrefilteredLookaheadConfig {
        immediate_candidate_limit: 4,
        habitat_candidate_limit: 3,
        immediate_anchor_limit: 2,
        prefilter_candidate_limit: 4,
        determinizations: 2,
        greedy_plies: 1,
    };
    let mut left =
        MlxHabitatPrefilteredLookaheadStrategy::with_predictor(CountRankingPredictor, config)
            .unwrap();
    let mut right =
        MlxHabitatPrefilteredLookaheadStrategy::with_predictor(CountRankingPredictor, config)
            .unwrap();
    let left_ranked = left.rank_and_select_deterministic(&game).unwrap();
    let right_ranked = right.rank_and_select_deterministic(&game).unwrap();

    assert_eq!(left_ranked, right_ranked);
    assert!(left_ranked.0.len() <= 4);
    game.transition(&left_ranked.1).unwrap();
}

#[test]
fn mlx_habitat_prefilter_preserves_immediate_anchors() {
    let game = GameState::new(
        GameConfig::research_aaaaa(2).unwrap(),
        GameSeed::from_u64(21),
    )
    .unwrap();
    let (prelude, _, union) = habitat_candidate_union(&game, 4, 3).unwrap();
    let protected: Vec<_> = union
        .iter()
        .take(3)
        .map(|candidate| with_prelude(candidate.action.clone(), &prelude))
        .collect();
    let mut strategy = MlxHabitatPrefilteredLookaheadStrategy::with_predictor(
        ReverseOrderRankingPredictor,
        MlxHabitatPrefilteredLookaheadConfig {
            immediate_candidate_limit: 4,
            habitat_candidate_limit: 3,
            immediate_anchor_limit: 3,
            prefilter_candidate_limit: 4,
            determinizations: 1,
            greedy_plies: 1,
        },
    )
    .unwrap();

    let ranked = strategy
        .rank_actions_with_rng(&game, &mut lookahead_decision_rng(&game))
        .unwrap();
    assert_eq!(ranked.len(), 4);
    assert!(
        protected
            .iter()
            .all(|action| ranked.iter().any(|candidate| &candidate.action == action))
    );
}

#[test]
fn mlx_habitat_rollout_is_legal_reproducible_and_ranked() {
    let seed = GameSeed::from_u64(22);
    let game = GameState::new(GameConfig::research_aaaaa(2).unwrap(), seed).unwrap();
    let config = MlxHabitatRolloutLookaheadConfig {
        immediate_candidate_limit: 3,
        habitat_candidate_limit: 2,
        determinizations: 2,
        rollout_plies: 2,
        rollout_immediate_candidate_limit: 3,
        rollout_habitat_candidate_limit: 2,
    };
    let mut left =
        MlxHabitatRolloutLookaheadStrategy::with_predictor(CountRankingPredictor, config).unwrap();
    let mut right =
        MlxHabitatRolloutLookaheadStrategy::with_predictor(CountRankingPredictor, config).unwrap();
    let left_ranked = left.rank_actions(&game).unwrap();
    let right_ranked = right.rank_actions(&game).unwrap();

    assert_eq!(left_ranked, right_ranked);
    assert!((3..=5).contains(&left_ranked.len()));
    assert!(
        left_ranked
            .windows(2)
            .all(|pair| pair[0].mean_leaf_score >= pair[1].mean_leaf_score)
    );
    let action = left.select_action(&game).unwrap();
    game.transition(&action).unwrap();
}

#[test]
fn mlx_habitat_rollout_rejects_non_finite_model_output() {
    let game = GameState::new(
        GameConfig::research_aaaaa(2).unwrap(),
        GameSeed::from_u64(23),
    )
    .unwrap();
    let mut strategy = MlxHabitatRolloutLookaheadStrategy::with_predictor(
        NonFiniteRankingPredictor,
        MlxHabitatRolloutLookaheadConfig {
            immediate_candidate_limit: 2,
            habitat_candidate_limit: 1,
            determinizations: 1,
            rollout_plies: 1,
            rollout_immediate_candidate_limit: 2,
            rollout_habitat_candidate_limit: 1,
        },
    )
    .unwrap();

    assert!(matches!(
        strategy.rank_actions(&game),
        Err(SearchError::NonFinitePrediction { index: 0 })
    ));
}

#[test]
fn mlx_self_rollout_is_legal_reproducible_and_ranked() {
    let seed = GameSeed::from_u64(24);
    let game = GameState::new(GameConfig::research_aaaaa(4).unwrap(), seed).unwrap();
    let config = MlxSelfRolloutLookaheadConfig {
        immediate_candidate_limit: 2,
        habitat_candidate_limit: 1,
        determinizations: 1,
        rollout_plies: 4,
        policy_immediate_candidate_limit: 2,
        policy_habitat_candidate_limit: 1,
    };
    let mut left =
        MlxSelfRolloutLookaheadStrategy::with_predictor(CountRankingPredictor, config).unwrap();
    let mut right =
        MlxSelfRolloutLookaheadStrategy::with_predictor(CountRankingPredictor, config).unwrap();
    let left_ranked = left.rank_actions(&game).unwrap();
    let right_ranked = right.rank_actions(&game).unwrap();

    assert_eq!(left_ranked, right_ranked);
    assert!((2..=3).contains(&left_ranked.len()));
    assert!(
        left_ranked
            .windows(2)
            .all(|pair| pair[0].mean_leaf_score >= pair[1].mean_leaf_score)
    );
    let action = left.select_action(&game).unwrap();
    game.transition(&action).unwrap();
}

#[test]
fn mlx_self_rollout_rejects_non_finite_model_output() {
    let game = GameState::new(
        GameConfig::research_aaaaa(4).unwrap(),
        GameSeed::from_u64(25),
    )
    .unwrap();
    let mut strategy = MlxSelfRolloutLookaheadStrategy::with_predictor(
        NonFiniteRankingPredictor,
        MlxSelfRolloutLookaheadConfig {
            immediate_candidate_limit: 2,
            habitat_candidate_limit: 1,
            determinizations: 1,
            rollout_plies: 4,
            policy_immediate_candidate_limit: 2,
            policy_habitat_candidate_limit: 1,
        },
    )
    .unwrap();

    assert!(matches!(
        strategy.rank_actions(&game),
        Err(SearchError::NonFinitePrediction { index: 0 })
    ));
}

#[test]
fn nature_wipe_lookahead_is_legal_and_reproducible() {
    let game = game_with_active_nature_token();
    let strategy = NatureWipeLookaheadStrategy::new(NatureWipeLookaheadConfig {
        action_search: DeterminizedLookaheadConfig {
            candidate_limit: 2,
            determinizations: 1,
            greedy_plies: 1,
        },
        prelude_candidate_limit: 2,
        prelude_determinizations: 1,
        prelude_greedy_plies: 1,
    })
    .unwrap();
    let left = strategy.select_action_deterministic(&game).unwrap();
    let right = strategy.select_action_deterministic(&game).unwrap();

    assert_eq!(left, right);
    assert!(left.wildlife_wipes.len() <= 1);
    game.transition(&left).unwrap();
}

#[test]
fn stable_market_rollout_rejection_is_deterministic_and_narrowly_classified() {
    let base = GameSeed::from_u64(52);
    assert_eq!(conditioned_rollout_seed(base, 0), base);
    assert_eq!(
        conditioned_rollout_seed(base, 1),
        conditioned_rollout_seed(base, 1)
    );
    assert_ne!(
        conditioned_rollout_seed(base, 1),
        conditioned_rollout_seed(base, 2)
    );
    assert!(SearchError::Rules(RuleError::WildlifeBagEmpty).is_unstable_market_exhaustion());
    assert!(
        SearchError::Simulation(SimulationError::Rules(RuleError::WildlifeBagEmpty))
            .is_unstable_market_exhaustion()
    );
    assert!(!SearchError::Rules(RuleError::GameOver).is_unstable_market_exhaustion());
}

fn game_with_active_nature_token() -> GameState {
    for numeric_seed in 0..64 {
        let mut game = GameState::new(
            GameConfig::research_aaaaa(2).unwrap(),
            GameSeed::from_u64(numeric_seed),
        )
        .unwrap();
        while !game.is_game_over() {
            if game.boards()[game.current_player()].nature_tokens() > 0 {
                return game;
            }
            let prelude = MarketPrelude {
                replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
                wildlife_wipes: Vec::new(),
            };
            let action = rank_greedy_actions(&game, &prelude, Some(1))
                .unwrap()
                .remove(0)
                .action;
            game.apply(&action).unwrap();
        }
    }
    panic!("test search did not produce an active player with a Nature Token");
}
