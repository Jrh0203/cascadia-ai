use std::ffi::OsStr;

use cascadia_data::PositionRecord;
use cascadia_game::{GameConfig, GameSeed, GameState, MarketPrelude, TurnAction};
use cascadia_model::{MAX_BATCH, ModelProcess};
use cascadia_sim::{
    GreedyCandidate, MatchResult, PatternAwareConfig, SimulationError, play_match_with_selector,
    rank_pattern_frontier_actions,
};
use rand::Rng;

use crate::{
    RankingPredictor, RolloutCandidate, SearchError, habitat_candidate_union,
    predict_ranking_scores, ranking_decision_rng, with_prelude,
};

pub const MLX_HABITAT_RANKING_STRATEGY_ID: &str = "mlx-habitat-ranking-v1";
pub const MLX_PATTERN_RANKING_STRATEGY_ID: &str = "mlx-pattern-ranking-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlxHabitatRankingConfig {
    pub immediate_candidate_limit: usize,
    pub habitat_candidate_limit: usize,
}

impl MlxHabitatRankingConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        if self.immediate_candidate_limit == 0 || self.habitat_candidate_limit == 0 {
            return Err(SearchError::InvalidConfig(
                "MLX habitat ranking candidate limits must be positive",
            ));
        }
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{MLX_HABITAT_RANKING_STRATEGY_ID}-k{}-h{}",
            self.immediate_candidate_limit, self.habitat_candidate_limit
        )
    }
}

impl Default for MlxHabitatRankingConfig {
    fn default() -> Self {
        Self {
            immediate_candidate_limit: 8,
            habitat_candidate_limit: 6,
        }
    }
}

pub struct MlxHabitatRankingStrategy<P = ModelProcess> {
    predictor: P,
    config: MlxHabitatRankingConfig,
    strategy_id: String,
}

impl MlxHabitatRankingStrategy<ModelProcess> {
    pub fn spawn<I, S>(
        program: impl AsRef<OsStr>,
        args: I,
        config: MlxHabitatRankingConfig,
    ) -> Result<Self, SearchError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let config = config.validate()?;
        Ok(Self {
            predictor: ModelProcess::spawn(program, args)?,
            strategy_id: config.strategy_id(),
            config,
        })
    }

    pub fn shutdown(self) -> Result<(), SearchError> {
        self.predictor.shutdown()?;
        Ok(())
    }
}

impl<P: RankingPredictor> MlxHabitatRankingStrategy<P> {
    pub fn with_predictor(
        predictor: P,
        config: MlxHabitatRankingConfig,
    ) -> Result<Self, SearchError> {
        let config = config.validate()?;
        Ok(Self {
            predictor,
            strategy_id: config.strategy_id(),
            config,
        })
    }

    pub fn strategy_id(&self) -> &str {
        &self.strategy_id
    }

    pub fn rank_actions(&mut self, game: &GameState) -> Result<Vec<RolloutCandidate>, SearchError> {
        let (prelude, _staged, candidates) = habitat_candidate_union(
            game,
            self.config.immediate_candidate_limit,
            self.config.habitat_candidate_limit,
        )?;
        rank_model_candidates(&mut self.predictor, game, &prelude, candidates)
    }

    pub fn select_from_teacher_candidates(
        &mut self,
        game: &GameState,
        candidates: &[RolloutCandidate],
        game_index: u64,
    ) -> Result<TurnAction, SearchError> {
        if candidates.is_empty() {
            return Err(SearchError::NoLegalActions);
        }
        let records = candidates
            .iter()
            .map(|candidate| {
                PositionRecord::observable_afterstate(game, &candidate.action, game_index)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut scores = Vec::with_capacity(records.len());
        for chunk in records.chunks(MAX_BATCH) {
            scores.extend(self.predictor.predict_scores(chunk)?);
        }
        select_scored_action(game, candidates, &scores)
    }

    pub fn select_action(&mut self, game: &GameState) -> Result<TurnAction, SearchError> {
        let ranked = self.rank_actions(game)?;
        select_ranked_model_action(game, &ranked)
    }

    pub fn play_match(
        &mut self,
        game_config: GameConfig,
        seed: GameSeed,
    ) -> Result<MatchResult, SearchError> {
        let strategy_id = self.strategy_id.clone();
        Ok(play_match_with_selector(
            game_config,
            seed,
            &strategy_id,
            |_, game| {
                self.select_action(game)
                    .map_err(|error| SimulationError::Strategy(error.to_string()))
            },
        )?)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MlxPatternRankingConfig {
    pub blueprint: PatternAwareConfig,
}

impl MlxPatternRankingConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        self.blueprint.validate()?;
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{MLX_PATTERN_RANKING_STRATEGY_ID}-k{}-h{}-b{}-m{}",
            self.blueprint.immediate_candidate_limit,
            self.blueprint.habitat_candidate_limit,
            self.blueprint.bear_candidate_limit,
            self.blueprint.future_market_draws,
        )
    }
}

pub struct MlxPatternRankingStrategy<P = ModelProcess> {
    predictor: P,
    config: MlxPatternRankingConfig,
    strategy_id: String,
}

impl MlxPatternRankingStrategy<ModelProcess> {
    pub fn spawn<I, S>(
        program: impl AsRef<OsStr>,
        args: I,
        config: MlxPatternRankingConfig,
    ) -> Result<Self, SearchError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let config = config.validate()?;
        Ok(Self {
            predictor: ModelProcess::spawn(program, args)?,
            strategy_id: config.strategy_id(),
            config,
        })
    }

    pub fn shutdown(self) -> Result<(), SearchError> {
        self.predictor.shutdown()?;
        Ok(())
    }
}

impl<P: RankingPredictor> MlxPatternRankingStrategy<P> {
    pub fn with_predictor(
        predictor: P,
        config: MlxPatternRankingConfig,
    ) -> Result<Self, SearchError> {
        let config = config.validate()?;
        Ok(Self {
            predictor,
            strategy_id: config.strategy_id(),
            config,
        })
    }

    pub fn strategy_id(&self) -> &str {
        &self.strategy_id
    }

    pub fn rank_actions(&mut self, game: &GameState) -> Result<Vec<RolloutCandidate>, SearchError> {
        let prelude = MarketPrelude {
            replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
            wildlife_wipes: Vec::new(),
        };
        let staged = game.preview_market_prelude(&prelude)?;
        let candidates = rank_pattern_frontier_actions(
            &staged,
            &MarketPrelude::default(),
            self.config.blueprint,
        )?;
        rank_model_candidates(&mut self.predictor, game, &prelude, candidates)
    }

    pub fn select_action(&mut self, game: &GameState) -> Result<TurnAction, SearchError> {
        let ranked = self.rank_actions(game)?;
        select_ranked_model_action(game, &ranked)
    }

    pub fn play_match(
        &mut self,
        game_config: GameConfig,
        seed: GameSeed,
    ) -> Result<MatchResult, SearchError> {
        let strategy_id = self.strategy_id.clone();
        Ok(play_match_with_selector(
            game_config,
            seed,
            &strategy_id,
            |_, game| {
                self.select_action(game)
                    .map_err(|error| SimulationError::Strategy(error.to_string()))
            },
        )?)
    }
}

fn rank_model_candidates<P: RankingPredictor>(
    predictor: &mut P,
    game: &GameState,
    prelude: &MarketPrelude,
    candidates: Vec<GreedyCandidate>,
) -> Result<Vec<RolloutCandidate>, SearchError> {
    let scores = predict_ranking_scores(predictor, game, prelude, &candidates)?;
    let mut ranked = candidates
        .into_iter()
        .zip(scores)
        .map(|(candidate, score)| RolloutCandidate {
            action: with_prelude(candidate.action, prelude),
            immediate_rank: candidate.immediate_rank,
            immediate_score: candidate.resulting_base_score,
            mean_leaf_score: f64::from(score),
            leaf_score_stddev: 0.0,
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .mean_leaf_score
            .total_cmp(&left.mean_leaf_score)
            .then_with(|| right.immediate_score.cmp(&left.immediate_score))
    });
    Ok(ranked)
}

fn select_scored_action(
    game: &GameState,
    candidates: &[RolloutCandidate],
    scores: &[f32],
) -> Result<TurnAction, SearchError> {
    if scores.len() != candidates.len() {
        return Err(SearchError::PredictionCount {
            expected: candidates.len(),
            actual: scores.len(),
        });
    }
    if let Some((index, _)) = scores
        .iter()
        .enumerate()
        .find(|(_, score)| !score.is_finite())
    {
        return Err(SearchError::NonFinitePrediction { index });
    }
    let best_score = scores
        .iter()
        .copied()
        .max_by(f32::total_cmp)
        .ok_or(SearchError::NoLegalActions)?;
    let tied = scores
        .iter()
        .enumerate()
        .filter_map(|(index, score)| (*score == best_score).then_some(index))
        .collect::<Vec<_>>();
    let mut rng = ranking_decision_rng(game);
    Ok(candidates[tied[rng.gen_range(0..tied.len())]]
        .action
        .clone())
}

fn select_ranked_model_action(
    game: &GameState,
    ranked: &[RolloutCandidate],
) -> Result<TurnAction, SearchError> {
    let Some(best) = ranked.first() else {
        return Err(SearchError::NoLegalActions);
    };
    let tied = ranked
        .iter()
        .take_while(|candidate| candidate.mean_leaf_score == best.mean_leaf_score)
        .count();
    let mut rng = ranking_decision_rng(game);
    Ok(ranked[rng.gen_range(0..tied)].action.clone())
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameConfig, GameSeed, GameState};

    use super::*;

    struct OrderedPredictor;

    impl RankingPredictor for OrderedPredictor {
        fn predict_scores(&mut self, records: &[PositionRecord]) -> Result<Vec<f32>, SearchError> {
            Ok((0..records.len()).map(|index| index as f32).collect())
        }
    }

    struct ObservableAfterstatePredictor;

    impl RankingPredictor for ObservableAfterstatePredictor {
        fn predict_scores(&mut self, records: &[PositionRecord]) -> Result<Vec<f32>, SearchError> {
            assert!(records.iter().all(|record| {
                record
                    .market_entities
                    .iter()
                    .any(|slot| slot[0] == u8::MAX || slot[3] == u8::MAX)
            }));
            Ok(records
                .iter()
                .map(|record| {
                    record
                        .to_bytes()
                        .iter()
                        .enumerate()
                        .map(|(index, value)| (index as u64 + 1) * u64::from(*value))
                        .sum::<u64>() as f32
                })
                .collect())
        }
    }

    #[test]
    fn habitat_ranker_is_legal_reproducible_and_bounded() {
        let game = GameState::new(
            GameConfig::research_aaaaa(2).unwrap(),
            GameSeed::from_u64(23),
        )
        .unwrap();
        let config = MlxHabitatRankingConfig {
            immediate_candidate_limit: 3,
            habitat_candidate_limit: 2,
        };
        let mut left = MlxHabitatRankingStrategy::with_predictor(OrderedPredictor, config).unwrap();
        let mut right =
            MlxHabitatRankingStrategy::with_predictor(OrderedPredictor, config).unwrap();

        let left_action = left.select_action(&game).unwrap();
        let right_action = right.select_action(&game).unwrap();

        assert_eq!(left_action, right_action);
        game.transition(&left_action).unwrap();
        assert_eq!(config.strategy_id(), "mlx-habitat-ranking-v1-k3-h2");
    }

    #[test]
    fn teacher_candidate_selection_rejects_prediction_count_mismatch() {
        let game = GameState::new(
            GameConfig::research_aaaaa(2).unwrap(),
            GameSeed::from_u64(29),
        )
        .unwrap();
        let mut strategy = MlxHabitatRankingStrategy::with_predictor(
            OrderedPredictor,
            MlxHabitatRankingConfig::default(),
        )
        .unwrap();
        let ranked = strategy.rank_actions(&game).unwrap();
        let action = strategy
            .select_from_teacher_candidates(&game, &ranked, 0)
            .unwrap();
        game.transition(&action).unwrap();
    }

    #[test]
    fn pattern_ranker_is_legal_reproducible_and_uses_full_frontier() {
        let game = GameState::new(
            GameConfig::research_aaaaa(2).unwrap(),
            GameSeed::from_u64(31),
        )
        .unwrap();
        let config = MlxPatternRankingConfig {
            blueprint: PatternAwareConfig {
                immediate_candidate_limit: 3,
                habitat_candidate_limit: 2,
                bear_candidate_limit: 2,
                future_market_draws: 4,
            },
        };
        let mut left = MlxPatternRankingStrategy::with_predictor(OrderedPredictor, config).unwrap();
        let mut right =
            MlxPatternRankingStrategy::with_predictor(OrderedPredictor, config).unwrap();

        let left_ranked = left.rank_actions(&game).unwrap();
        let right_ranked = right.rank_actions(&game).unwrap();

        assert_eq!(left_ranked, right_ranked);
        assert!((3..=7).contains(&left_ranked.len()));
        game.transition(&left.select_action(&game).unwrap())
            .unwrap();
        assert_eq!(config.strategy_id(), "mlx-pattern-ranking-v1-k3-h2-b2-m4");
    }

    #[test]
    fn pattern_ranker_inference_is_invariant_to_hidden_refill_order() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(230),
        )
        .unwrap();
        assert!(game.market().three_of_a_kind().is_none());
        let mut redetermined = game.clone();
        redetermined.redeterminize_hidden(GameSeed::from_u64(231));
        let config = MlxPatternRankingConfig {
            blueprint: PatternAwareConfig {
                immediate_candidate_limit: 3,
                habitat_candidate_limit: 2,
                bear_candidate_limit: 2,
                future_market_draws: 2,
            },
        };
        let mut left =
            MlxPatternRankingStrategy::with_predictor(ObservableAfterstatePredictor, config)
                .unwrap();
        let mut right =
            MlxPatternRankingStrategy::with_predictor(ObservableAfterstatePredictor, config)
                .unwrap();

        assert_eq!(
            left.rank_actions(&game).unwrap(),
            right.rank_actions(&redetermined).unwrap()
        );
    }
}
