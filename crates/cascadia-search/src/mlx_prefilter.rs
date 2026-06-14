use std::ffi::OsStr;

use cascadia_game::{GameConfig, GameSeed, GameState, TurnAction};
use cascadia_model::ModelProcess;
use cascadia_sim::{GreedyCandidate, MatchResult, SimulationError, play_match_with_selector};
use rand_chacha::ChaCha8Rng;

use super::{
    MLX_HABITAT_PREFILTER_LOOKAHEAD_STRATEGY_ID, MLX_PREFILTER_LOOKAHEAD_STRATEGY_ID,
    RankingPredictor, RolloutCandidate, SearchError, bear_candidate_union, habitat_candidate_union,
    lookahead_decision_rng, predict_ranking_scores, rank_rollout_candidates, select_ranked_action,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlxPrefilteredLookaheadConfig {
    pub immediate_candidate_limit: usize,
    pub bear_candidate_limit: usize,
    pub immediate_anchor_limit: usize,
    pub prefilter_candidate_limit: usize,
    pub determinizations: usize,
    pub greedy_plies: usize,
}

impl MlxPrefilteredLookaheadConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        if self.immediate_candidate_limit == 0
            || self.bear_candidate_limit == 0
            || self.prefilter_candidate_limit == 0
        {
            return Err(SearchError::InvalidConfig(
                "MLX prefilter candidate limits must be positive",
            ));
        }
        if self.immediate_anchor_limit > self.immediate_candidate_limit
            || self.immediate_anchor_limit > self.prefilter_candidate_limit
        {
            return Err(SearchError::InvalidConfig(
                "MLX prefilter anchors must fit within the immediate and retained candidates",
            ));
        }
        if self.determinizations == 0 {
            return Err(SearchError::InvalidConfig(
                "MLX prefilter lookahead must use at least one determinization",
            ));
        }
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        if self.immediate_anchor_limit == 0 {
            format!(
                "{MLX_PREFILTER_LOOKAHEAD_STRATEGY_ID}-k{}-b{}-p{}-r{}-d{}",
                self.immediate_candidate_limit,
                self.bear_candidate_limit,
                self.prefilter_candidate_limit,
                self.determinizations,
                self.greedy_plies
            )
        } else {
            format!(
                "mlx-anchored-prefilter-lookahead-v1-k{}-b{}-a{}-p{}-r{}-d{}",
                self.immediate_candidate_limit,
                self.bear_candidate_limit,
                self.immediate_anchor_limit,
                self.prefilter_candidate_limit,
                self.determinizations,
                self.greedy_plies
            )
        }
    }
}

impl Default for MlxPrefilteredLookaheadConfig {
    fn default() -> Self {
        Self {
            immediate_candidate_limit: 8,
            bear_candidate_limit: 8,
            immediate_anchor_limit: 0,
            prefilter_candidate_limit: 8,
            determinizations: 4,
            greedy_plies: 4,
        }
    }
}

pub struct MlxPrefilteredLookaheadStrategy<P = ModelProcess> {
    predictor: P,
    config: MlxPrefilteredLookaheadConfig,
    strategy_id: String,
}

impl MlxPrefilteredLookaheadStrategy<ModelProcess> {
    pub fn spawn<I, S>(
        program: impl AsRef<OsStr>,
        args: I,
        config: MlxPrefilteredLookaheadConfig,
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

impl<P: RankingPredictor> MlxPrefilteredLookaheadStrategy<P> {
    pub fn with_predictor(
        predictor: P,
        config: MlxPrefilteredLookaheadConfig,
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
        self.rank_actions_with_rng(game, &mut lookahead_decision_rng(game))
    }

    pub(crate) fn rank_actions_with_rng(
        &mut self,
        game: &GameState,
        rng: &mut ChaCha8Rng,
    ) -> Result<Vec<RolloutCandidate>, SearchError> {
        let (prelude, staged, candidates) = bear_candidate_union(
            game,
            self.config.immediate_candidate_limit,
            self.config.bear_candidate_limit,
        )?;
        let scores = predict_ranking_scores(&mut self.predictor, game, &prelude, &candidates)?;
        let retained = retain_prefilter_candidates(
            candidates,
            scores,
            self.config.immediate_anchor_limit,
            self.config.prefilter_candidate_limit,
        );
        rank_rollout_candidates(
            game,
            &staged,
            retained,
            &prelude,
            self.config.determinizations,
            self.config.greedy_plies,
            rng,
        )
    }

    pub fn rank_and_select_deterministic(
        &mut self,
        game: &GameState,
    ) -> Result<(Vec<RolloutCandidate>, TurnAction), SearchError> {
        let mut rng = lookahead_decision_rng(game);
        let ranked = self.rank_actions_with_rng(game, &mut rng)?;
        let action = select_ranked_action(&ranked, &mut rng)?;
        Ok((ranked, action))
    }

    pub fn select_action_deterministic(
        &mut self,
        game: &GameState,
    ) -> Result<TurnAction, SearchError> {
        Ok(self.rank_and_select_deterministic(game)?.1)
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
                self.select_action_deterministic(game)
                    .map_err(|error| SimulationError::Strategy(error.to_string()))
            },
        )?)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlxHabitatPrefilteredLookaheadConfig {
    pub immediate_candidate_limit: usize,
    pub habitat_candidate_limit: usize,
    pub immediate_anchor_limit: usize,
    pub prefilter_candidate_limit: usize,
    pub determinizations: usize,
    pub greedy_plies: usize,
}

impl MlxHabitatPrefilteredLookaheadConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        if self.immediate_candidate_limit == 0
            || self.habitat_candidate_limit == 0
            || self.prefilter_candidate_limit == 0
        {
            return Err(SearchError::InvalidConfig(
                "MLX habitat prefilter candidate limits must be positive",
            ));
        }
        if self.immediate_anchor_limit > self.immediate_candidate_limit
            || self.immediate_anchor_limit > self.prefilter_candidate_limit
        {
            return Err(SearchError::InvalidConfig(
                "MLX habitat prefilter anchors must fit within immediate and retained candidates",
            ));
        }
        if self.determinizations == 0 {
            return Err(SearchError::InvalidConfig(
                "MLX habitat prefilter must use at least one determinization",
            ));
        }
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{MLX_HABITAT_PREFILTER_LOOKAHEAD_STRATEGY_ID}-k{}-h{}-a{}-p{}-r{}-d{}",
            self.immediate_candidate_limit,
            self.habitat_candidate_limit,
            self.immediate_anchor_limit,
            self.prefilter_candidate_limit,
            self.determinizations,
            self.greedy_plies
        )
    }
}

pub struct MlxHabitatPrefilteredLookaheadStrategy<P = ModelProcess> {
    predictor: P,
    config: MlxHabitatPrefilteredLookaheadConfig,
    strategy_id: String,
}

impl MlxHabitatPrefilteredLookaheadStrategy<ModelProcess> {
    pub fn spawn<I, S>(
        program: impl AsRef<OsStr>,
        args: I,
        config: MlxHabitatPrefilteredLookaheadConfig,
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

impl<P: RankingPredictor> MlxHabitatPrefilteredLookaheadStrategy<P> {
    pub fn with_predictor(
        predictor: P,
        config: MlxHabitatPrefilteredLookaheadConfig,
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

    pub(crate) fn rank_actions_with_rng(
        &mut self,
        game: &GameState,
        rng: &mut ChaCha8Rng,
    ) -> Result<Vec<RolloutCandidate>, SearchError> {
        let (prelude, staged, candidates) = habitat_candidate_union(
            game,
            self.config.immediate_candidate_limit,
            self.config.habitat_candidate_limit,
        )?;
        let scores = predict_ranking_scores(&mut self.predictor, game, &prelude, &candidates)?;
        let retained = retain_prefilter_candidates(
            candidates,
            scores,
            self.config.immediate_anchor_limit,
            self.config.prefilter_candidate_limit,
        );
        rank_rollout_candidates(
            game,
            &staged,
            retained,
            &prelude,
            self.config.determinizations,
            self.config.greedy_plies,
            rng,
        )
    }

    pub fn rank_and_select_deterministic(
        &mut self,
        game: &GameState,
    ) -> Result<(Vec<RolloutCandidate>, TurnAction), SearchError> {
        let mut rng = lookahead_decision_rng(game);
        let ranked = self.rank_actions_with_rng(game, &mut rng)?;
        let action = select_ranked_action(&ranked, &mut rng)?;
        Ok((ranked, action))
    }

    pub fn select_action_deterministic(
        &mut self,
        game: &GameState,
    ) -> Result<TurnAction, SearchError> {
        Ok(self.rank_and_select_deterministic(game)?.1)
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
                self.select_action_deterministic(game)
                    .map_err(|error| SimulationError::Strategy(error.to_string()))
            },
        )?)
    }
}

fn retain_prefilter_candidates(
    candidates: Vec<GreedyCandidate>,
    scores: Vec<f32>,
    immediate_anchor_limit: usize,
    prefilter_candidate_limit: usize,
) -> Vec<GreedyCandidate> {
    let retained_limit = prefilter_candidate_limit.min(candidates.len());
    let anchor_limit = immediate_anchor_limit.min(retained_limit);
    let mut ranked_indices: Vec<_> = (anchor_limit..candidates.len()).collect();
    ranked_indices.sort_by(|left, right| {
        scores[*right]
            .total_cmp(&scores[*left])
            .then_with(|| left.cmp(right))
    });

    let mut retained_indices: Vec<_> = (0..anchor_limit).collect();
    retained_indices.extend(
        ranked_indices
            .into_iter()
            .take(retained_limit - anchor_limit),
    );
    retained_indices
        .into_iter()
        .map(|index| candidates[index].clone())
        .collect()
}
