use std::ffi::OsStr;

use cascadia_data::{ActionPositionRecord, PositionRecord, ProposalActionFeatures};
use cascadia_game::{GameConfig, GameSeed, GameState, MarketPrelude, TurnAction};
use cascadia_model::{MAX_BATCH, ModelProcess};
use cascadia_sim::{
    MatchResult, PATTERN_AWARE_STRATEGY_ID, PatternAwareConfig, SimulationError,
    play_match_with_selector, rank_greedy_actions, rank_pattern_frontier_actions,
    rank_wildlife_diverse_pattern_frontier_actions, select_pattern_action, strategy_rng,
};
use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::{RolloutCandidate, SearchError, ranking_decision_rng, with_prelude};

pub const MLX_ACTION_DELTA_RANKING_STRATEGY_ID: &str = "mlx-action-delta-ranking-v1";
pub const MLX_FULL_ACTION_IMITATION_STRATEGY_ID: &str = "mlx-full-action-imitation-v1";
pub const MLX_PUBLIC_BEAM_VALUE_STRATEGY_ID: &str = "mlx-public-beam-value-v1";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MlxActionDeltaRankingConfig {
    pub blueprint: PatternAwareConfig,
}

impl MlxActionDeltaRankingConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        self.blueprint.validate()?;
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{MLX_ACTION_DELTA_RANKING_STRATEGY_ID}-k{}-h{}-b{}-m{}",
            self.blueprint.immediate_candidate_limit,
            self.blueprint.habitat_candidate_limit,
            self.blueprint.bear_candidate_limit,
            self.blueprint.future_market_draws,
        )
    }
}

pub trait ActionRankingPredictor {
    fn predict_action_scores(
        &mut self,
        records: &[ActionPositionRecord],
    ) -> Result<Vec<f32>, SearchError>;
}

pub trait ImitationPredictor {
    fn predict_imitation_scores(
        &mut self,
        position: &PositionRecord,
        actions: &[ProposalActionFeatures],
    ) -> Result<Vec<f32>, SearchError>;
}

impl ActionRankingPredictor for ModelProcess {
    fn predict_action_scores(
        &mut self,
        records: &[ActionPositionRecord],
    ) -> Result<Vec<f32>, SearchError> {
        Ok(ModelProcess::predict_action_scores(self, records)?)
    }
}

impl ImitationPredictor for ModelProcess {
    fn predict_imitation_scores(
        &mut self,
        position: &PositionRecord,
        actions: &[ProposalActionFeatures],
    ) -> Result<Vec<f32>, SearchError> {
        Ok(ModelProcess::predict_imitation_scores(
            self, position, actions,
        )?)
    }
}

pub struct MlxActionDeltaRankingStrategy<P = ModelProcess> {
    predictor: P,
    config: MlxActionDeltaRankingConfig,
    strategy_id: String,
}

pub struct MlxFullActionImitationStrategy<P = ModelProcess> {
    predictor: P,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlxPublicBeamValueConfig {
    pub blueprint: PatternAwareConfig,
    pub wildlife_candidate_limit: usize,
    pub final_personal_turns: u16,
}

impl Default for MlxPublicBeamValueConfig {
    fn default() -> Self {
        Self {
            blueprint: PatternAwareConfig::default(),
            wildlife_candidate_limit: 2,
            final_personal_turns: 5,
        }
    }
}

impl MlxPublicBeamValueConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        self.blueprint.validate()?;
        if self.wildlife_candidate_limit == 0 || !(1..=20).contains(&self.final_personal_turns) {
            return Err(SearchError::InvalidConfig(
                "public beam-value policy requires positive W and a 1-20 turn cutoff",
            ));
        }
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{MLX_PUBLIC_BEAM_VALUE_STRATEGY_ID}-t{}-w{}-k{}-h{}-b{}-m{}",
            self.final_personal_turns,
            self.wildlife_candidate_limit,
            self.blueprint.immediate_candidate_limit,
            self.blueprint.habitat_candidate_limit,
            self.blueprint.bear_candidate_limit,
            self.blueprint.future_market_draws,
        )
    }
}

pub struct MlxPublicBeamValueStrategy<P = ModelProcess> {
    predictor: P,
    config: MlxPublicBeamValueConfig,
    strategy_id: String,
}

impl MlxPublicBeamValueStrategy<ModelProcess> {
    pub fn spawn<I, S>(
        program: impl AsRef<OsStr>,
        args: I,
        config: MlxPublicBeamValueConfig,
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

impl<P: ActionRankingPredictor> MlxPublicBeamValueStrategy<P> {
    pub fn with_predictor(
        predictor: P,
        config: MlxPublicBeamValueConfig,
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

    pub fn uses_model(&self, game: &GameState) -> bool {
        game.turns_remaining_for_player(game.current_player()) <= self.config.final_personal_turns
    }

    pub fn rank_terminal_actions(
        &mut self,
        game: &GameState,
    ) -> Result<Vec<RolloutCandidate>, SearchError> {
        let candidates = rank_wildlife_diverse_pattern_frontier_actions(
            game,
            &MarketPrelude::default(),
            self.config.blueprint,
            self.config.wildlife_candidate_limit,
        )?;
        if candidates.is_empty() {
            return Err(SearchError::NoLegalActions);
        }
        let records = candidates
            .iter()
            .map(|candidate| {
                Ok(ActionPositionRecord::observe(
                    game,
                    &candidate.action,
                    0,
                    u16::try_from(candidate.immediate_rank).map_err(|_| {
                        SearchError::InvalidConfig(
                            "public beam-value immediate rank exceeds fixed-width storage",
                        )
                    })?,
                    candidate.resulting_base_score,
                )?)
            })
            .collect::<Result<Vec<_>, SearchError>>()?;
        let mut scores = Vec::with_capacity(records.len());
        for chunk in records.chunks(MAX_BATCH) {
            scores.extend(self.predictor.predict_action_scores(chunk)?);
        }
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
        let mut ranked = candidates
            .into_iter()
            .zip(scores)
            .map(|(candidate, score)| RolloutCandidate {
                action: candidate.action,
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

    pub fn select_action(&mut self, game: &GameState) -> Result<TurnAction, SearchError> {
        let mut rng = ranking_decision_rng(game);
        self.select_action_with_rng(game, &mut rng)
    }

    pub fn select_action_with_rng(
        &mut self,
        game: &GameState,
        blueprint_rng: &mut ChaCha8Rng,
    ) -> Result<TurnAction, SearchError> {
        if !self.uses_model(game) {
            let prelude = MarketPrelude {
                replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
                wildlife_wipes: Vec::new(),
            };
            return Ok(select_pattern_action(
                game,
                &prelude,
                self.config.blueprint,
                blueprint_rng,
            )?);
        }
        let ranked = self.rank_terminal_actions(game)?;
        let best = ranked.first().ok_or(SearchError::NoLegalActions)?;
        let tied = ranked
            .iter()
            .take_while(|candidate| candidate.mean_leaf_score == best.mean_leaf_score)
            .count();
        let mut rng = ranking_decision_rng(game);
        Ok(ranked[rng.gen_range(0..tied)].action.clone())
    }

    pub fn play_match(
        &mut self,
        game_config: GameConfig,
        seed: GameSeed,
    ) -> Result<MatchResult, SearchError> {
        let strategy_id = self.strategy_id.clone();
        let mut blueprint_rngs = (0..usize::from(game_config.player_count))
            .map(|seat| strategy_rng(seed, seat, PATTERN_AWARE_STRATEGY_ID))
            .collect::<Vec<_>>();
        Ok(play_match_with_selector(
            game_config,
            seed,
            &strategy_id,
            |player, game| {
                self.select_action_with_rng(game, &mut blueprint_rngs[player])
                    .map_err(|error| SimulationError::Strategy(error.to_string()))
            },
        )?)
    }
}

impl MlxActionDeltaRankingStrategy<ModelProcess> {
    pub fn spawn<I, S>(
        program: impl AsRef<OsStr>,
        args: I,
        config: MlxActionDeltaRankingConfig,
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

impl<P: ActionRankingPredictor> MlxActionDeltaRankingStrategy<P> {
    pub fn with_predictor(
        predictor: P,
        config: MlxActionDeltaRankingConfig,
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
        if candidates.is_empty() {
            return Err(SearchError::NoLegalActions);
        }
        let mut records = Vec::with_capacity(candidates.len());
        for candidate in &candidates {
            let action = with_prelude(candidate.action.clone(), &prelude);
            records.push(ActionPositionRecord::observe(
                game,
                &action,
                0,
                u16::try_from(candidate.immediate_rank).map_err(|_| {
                    SearchError::InvalidConfig(
                        "action-delta immediate rank exceeds fixed-width storage",
                    )
                })?,
                candidate.resulting_base_score,
            )?);
        }
        let mut scores = Vec::with_capacity(records.len());
        for chunk in records.chunks(MAX_BATCH) {
            scores.extend(self.predictor.predict_action_scores(chunk)?);
        }
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

        let mut ranked = candidates
            .into_iter()
            .zip(scores)
            .map(|(candidate, score)| RolloutCandidate {
                action: with_prelude(candidate.action, &prelude),
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

    pub fn select_action(&mut self, game: &GameState) -> Result<TurnAction, SearchError> {
        let ranked = self.rank_actions(game)?;
        let best = ranked.first().ok_or(SearchError::NoLegalActions)?;
        let tied = ranked
            .iter()
            .take_while(|candidate| candidate.mean_leaf_score == best.mean_leaf_score)
            .count();
        let mut rng = ranking_decision_rng(game);
        Ok(ranked[rng.gen_range(0..tied)].action.clone())
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

impl MlxFullActionImitationStrategy<ModelProcess> {
    pub fn spawn<I, S>(program: impl AsRef<OsStr>, args: I) -> Result<Self, SearchError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        Ok(Self {
            predictor: ModelProcess::spawn(program, args)?,
        })
    }

    pub fn shutdown(self) -> Result<(), SearchError> {
        self.predictor.shutdown()?;
        Ok(())
    }
}

impl<P: ImitationPredictor> MlxFullActionImitationStrategy<P> {
    pub fn with_predictor(predictor: P) -> Self {
        Self { predictor }
    }

    pub const fn strategy_id(&self) -> &'static str {
        MLX_FULL_ACTION_IMITATION_STRATEGY_ID
    }

    pub fn rank_actions(&mut self, game: &GameState) -> Result<Vec<RolloutCandidate>, SearchError> {
        let prelude = MarketPrelude {
            replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
            wildlife_wipes: Vec::new(),
        };
        let candidates = rank_greedy_actions(game, &prelude, None)?;
        if candidates.is_empty() {
            return Err(SearchError::NoLegalActions);
        }
        let actions = candidates
            .iter()
            .map(|candidate| {
                Ok(ProposalActionFeatures::from_game_action(
                    game,
                    &candidate.action,
                    u16::try_from(candidate.immediate_rank).map_err(|_| {
                        SearchError::InvalidConfig(
                            "full-action immediate rank exceeds fixed-width storage",
                        )
                    })?,
                    candidate.resulting_base_score,
                )?)
            })
            .collect::<Result<Vec<_>, SearchError>>()?;
        let position = PositionRecord::observe(game, 0);
        let mut scores = Vec::with_capacity(actions.len());
        for chunk in actions.chunks(MAX_BATCH) {
            scores.extend(self.predictor.predict_imitation_scores(&position, chunk)?);
        }
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
        let mut ranked = candidates
            .into_iter()
            .zip(scores)
            .map(|(candidate, score)| RolloutCandidate {
                action: candidate.action,
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
                .then_with(|| left.immediate_rank.cmp(&right.immediate_rank))
        });
        Ok(ranked)
    }

    pub fn select_action(&mut self, game: &GameState) -> Result<TurnAction, SearchError> {
        let ranked = self.rank_actions(game)?;
        let best = ranked.first().ok_or(SearchError::NoLegalActions)?;
        let tied = ranked
            .iter()
            .take_while(|candidate| candidate.mean_leaf_score == best.mean_leaf_score)
            .count();
        let mut rng = ranking_decision_rng(game);
        Ok(ranked[rng.gen_range(0..tied)].action.clone())
    }

    pub fn play_match(
        &mut self,
        game_config: GameConfig,
        seed: GameSeed,
    ) -> Result<MatchResult, SearchError> {
        Ok(play_match_with_selector(
            game_config,
            seed,
            MLX_FULL_ACTION_IMITATION_STRATEGY_ID,
            |_, game| {
                self.select_action(game)
                    .map_err(|error| SimulationError::Strategy(error.to_string()))
            },
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ExplicitActionPredictor;

    impl ActionRankingPredictor for ExplicitActionPredictor {
        fn predict_action_scores(
            &mut self,
            records: &[ActionPositionRecord],
        ) -> Result<Vec<f32>, SearchError> {
            assert!(records.iter().all(|record| {
                record
                    .position
                    .market_entities
                    .iter()
                    .any(|slot| slot[0] == u8::MAX || slot[3] == u8::MAX)
            }));
            Ok(records
                .iter()
                .map(|record| {
                    f32::from(record.action.immediate_score)
                        + f32::from(record.action.tile_q) / 100.0
                        + f32::from(record.action.tile_r) / 10_000.0
                })
                .collect())
        }
    }

    struct PublicValuePredictor;

    impl ActionRankingPredictor for PublicValuePredictor {
        fn predict_action_scores(
            &mut self,
            records: &[ActionPositionRecord],
        ) -> Result<Vec<f32>, SearchError> {
            assert!(records.iter().all(|record| {
                record.action.replace_three_of_a_kind == 0
                    && record.action.paid_wipe_count == 0
                    && record.action.paid_wipe_total_slots == 0
            }));
            Ok(records
                .iter()
                .map(|record| f32::from(record.action.immediate_score))
                .collect())
        }
    }

    impl ImitationPredictor for ExplicitActionPredictor {
        fn predict_imitation_scores(
            &mut self,
            _position: &PositionRecord,
            actions: &[ProposalActionFeatures],
        ) -> Result<Vec<f32>, SearchError> {
            Ok(actions
                .iter()
                .map(|action| {
                    f32::from(action.immediate_score)
                        + f32::from(action.tile_q) / 100.0
                        + f32::from(action.tile_r) / 10_000.0
                })
                .collect())
        }
    }

    fn tiny_config() -> MlxActionDeltaRankingConfig {
        MlxActionDeltaRankingConfig {
            blueprint: PatternAwareConfig {
                immediate_candidate_limit: 3,
                habitat_candidate_limit: 2,
                bear_candidate_limit: 2,
                future_market_draws: 2,
            },
        }
    }

    fn tiny_public_config() -> MlxPublicBeamValueConfig {
        MlxPublicBeamValueConfig {
            blueprint: PatternAwareConfig {
                immediate_candidate_limit: 3,
                habitat_candidate_limit: 2,
                bear_candidate_limit: 2,
                future_market_draws: 2,
            },
            wildlife_candidate_limit: 1,
            final_personal_turns: 20,
        }
    }

    #[test]
    fn action_delta_ranker_is_legal_reproducible_and_explicit() {
        let game = GameState::new(
            GameConfig::research_aaaaa(2).unwrap(),
            GameSeed::from_u64(401),
        )
        .unwrap();
        let mut left =
            MlxActionDeltaRankingStrategy::with_predictor(ExplicitActionPredictor, tiny_config())
                .unwrap();
        let mut right =
            MlxActionDeltaRankingStrategy::with_predictor(ExplicitActionPredictor, tiny_config())
                .unwrap();

        let left_action = left.select_action(&game).unwrap();
        let right_action = right.select_action(&game).unwrap();

        assert_eq!(left_action, right_action);
        game.transition(&left_action).unwrap();
        assert_eq!(
            tiny_config().strategy_id(),
            "mlx-action-delta-ranking-v1-k3-h2-b2-m2"
        );
    }

    #[test]
    fn action_delta_inference_is_invariant_to_hidden_refill_order() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(402),
        )
        .unwrap();
        assert!(game.market().three_of_a_kind().is_none());
        let mut redetermined = game.clone();
        redetermined.redeterminize_hidden(GameSeed::from_u64(403));
        let mut left =
            MlxActionDeltaRankingStrategy::with_predictor(ExplicitActionPredictor, tiny_config())
                .unwrap();
        let mut right =
            MlxActionDeltaRankingStrategy::with_predictor(ExplicitActionPredictor, tiny_config())
                .unwrap();

        assert_eq!(
            left.rank_actions(&game).unwrap(),
            right.rank_actions(&redetermined).unwrap()
        );
    }

    #[test]
    fn full_action_imitation_scores_every_canonical_legal_action() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(406),
        )
        .unwrap();
        let prelude = MarketPrelude {
            replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
            wildlife_wipes: Vec::new(),
        };
        let expected = game.legal_turn_actions(&prelude).unwrap().len();
        let mut strategy = MlxFullActionImitationStrategy::with_predictor(ExplicitActionPredictor);

        let ranked = strategy.rank_actions(&game).unwrap();

        assert_eq!(ranked.len(), expected);
        assert!(
            ranked
                .iter()
                .all(|candidate| game.transition(&candidate.action).is_ok())
        );
        assert_eq!(
            strategy.strategy_id(),
            MLX_FULL_ACTION_IMITATION_STRATEGY_ID
        );
    }

    #[test]
    fn public_beam_value_policy_is_public_legal_and_reproducible() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(404),
        )
        .unwrap();
        let mut redetermined = game.clone();
        redetermined.redeterminize_hidden(GameSeed::from_u64(405));
        let mut left =
            MlxPublicBeamValueStrategy::with_predictor(PublicValuePredictor, tiny_public_config())
                .unwrap();
        let mut right =
            MlxPublicBeamValueStrategy::with_predictor(PublicValuePredictor, tiny_public_config())
                .unwrap();

        let left_ranked = left.rank_terminal_actions(&game).unwrap();
        let right_ranked = right.rank_terminal_actions(&redetermined).unwrap();
        assert_eq!(left_ranked, right_ranked);
        let action = left.select_action(&game).unwrap();
        game.transition(&action).unwrap();
        assert!(!action.replace_three_of_a_kind);
        assert_eq!(
            tiny_public_config().strategy_id(),
            "mlx-public-beam-value-v1-t20-w1-k3-h2-b2-m2"
        );
    }
}
