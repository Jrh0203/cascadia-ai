use std::ffi::OsStr;

use cascadia_data::PositionRecord;
use cascadia_game::{GameConfig, GameSeed, GameState, MarketPrelude, TurnAction, score_board};
use cascadia_model::{MAX_BATCH, ModelProcess};
use cascadia_sim::{
    GreedyCandidate, MatchResult, SimulationError, play_greedy_plies, play_match_with_selector,
};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use super::{
    RolloutCandidate, SearchError, bear_candidate_union, habitat_candidate_union,
    lookahead_decision_rng, predict_ranking_scores, ranking_decision_rng, rollout_rng,
    select_ranked_action, with_prelude,
};

pub const MLX_RANKING_STRATEGY_ID: &str = "mlx-ranking-v1";
pub const MLX_PREFILTER_LOOKAHEAD_STRATEGY_ID: &str = "mlx-prefilter-lookahead-v1";
pub const MLX_HABITAT_PREFILTER_LOOKAHEAD_STRATEGY_ID: &str = "mlx-habitat-prefilter-lookahead-v1";
pub const MLX_HABITAT_ROLLOUT_LOOKAHEAD_STRATEGY_ID: &str = "mlx-habitat-rollout-lookahead-v1";
pub const MLX_SELF_ROLLOUT_LOOKAHEAD_STRATEGY_ID: &str = "mlx-self-rollout-lookahead-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlxRankingConfig {
    pub immediate_candidate_limit: usize,
    pub bear_candidate_limit: usize,
}

impl MlxRankingConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        if self.immediate_candidate_limit == 0 || self.bear_candidate_limit == 0 {
            return Err(SearchError::InvalidConfig(
                "MLX ranking candidate limits must be positive",
            ));
        }
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{MLX_RANKING_STRATEGY_ID}-k{}-b{}",
            self.immediate_candidate_limit, self.bear_candidate_limit
        )
    }
}

impl Default for MlxRankingConfig {
    fn default() -> Self {
        Self {
            immediate_candidate_limit: 8,
            bear_candidate_limit: 8,
        }
    }
}

pub trait RankingPredictor {
    fn predict_scores(&mut self, records: &[PositionRecord]) -> Result<Vec<f32>, SearchError>;
}

impl RankingPredictor for ModelProcess {
    fn predict_scores(&mut self, records: &[PositionRecord]) -> Result<Vec<f32>, SearchError> {
        Ok(ModelProcess::predict_scores(self, records)?)
    }
}

pub struct MlxRankingStrategy<P = ModelProcess> {
    predictor: P,
    config: MlxRankingConfig,
    strategy_id: String,
}

impl MlxRankingStrategy<ModelProcess> {
    pub fn spawn<I, S>(
        program: impl AsRef<OsStr>,
        args: I,
        config: MlxRankingConfig,
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

impl<P: RankingPredictor> MlxRankingStrategy<P> {
    pub fn with_predictor(predictor: P, config: MlxRankingConfig) -> Result<Self, SearchError> {
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

    pub fn select_action(&mut self, game: &GameState) -> Result<TurnAction, SearchError> {
        let (prelude, _staged, candidates) = bear_candidate_union(
            game,
            self.config.immediate_candidate_limit,
            self.config.bear_candidate_limit,
        )?;
        let actions: Vec<_> = candidates
            .iter()
            .map(|candidate| with_prelude(candidate.action.clone(), &prelude))
            .collect();
        let scores = predict_ranking_scores(&mut self.predictor, game, &prelude, &candidates)?;
        let best_score = scores
            .iter()
            .copied()
            .max_by(f32::total_cmp)
            .ok_or(SearchError::NoLegalActions)?;
        let tied: Vec<_> = scores
            .iter()
            .enumerate()
            .filter_map(|(index, score)| (*score == best_score).then_some(index))
            .collect();
        let mut rng = ranking_decision_rng(game);
        Ok(actions[tied[rng.gen_range(0..tied.len())]].clone())
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlxHabitatRolloutLookaheadConfig {
    pub immediate_candidate_limit: usize,
    pub habitat_candidate_limit: usize,
    pub determinizations: usize,
    pub rollout_plies: usize,
    pub rollout_immediate_candidate_limit: usize,
    pub rollout_habitat_candidate_limit: usize,
}

impl MlxHabitatRolloutLookaheadConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        if self.immediate_candidate_limit == 0
            || self.habitat_candidate_limit == 0
            || self.rollout_immediate_candidate_limit == 0
            || self.rollout_habitat_candidate_limit == 0
        {
            return Err(SearchError::InvalidConfig(
                "MLX habitat rollout candidate limits must be positive",
            ));
        }
        if self.determinizations == 0 || self.rollout_plies == 0 {
            return Err(SearchError::InvalidConfig(
                "MLX habitat rollout sampling and horizon must be positive",
            ));
        }
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{MLX_HABITAT_ROLLOUT_LOOKAHEAD_STRATEGY_ID}-k{}-h{}-r{}-d{}-rk{}-rh{}",
            self.immediate_candidate_limit,
            self.habitat_candidate_limit,
            self.determinizations,
            self.rollout_plies,
            self.rollout_immediate_candidate_limit,
            self.rollout_habitat_candidate_limit,
        )
    }
}

impl Default for MlxHabitatRolloutLookaheadConfig {
    fn default() -> Self {
        Self {
            immediate_candidate_limit: 8,
            habitat_candidate_limit: 6,
            determinizations: 4,
            rollout_plies: 4,
            rollout_immediate_candidate_limit: 8,
            rollout_habitat_candidate_limit: 6,
        }
    }
}

pub struct MlxHabitatRolloutLookaheadStrategy<P = ModelProcess> {
    predictor: P,
    config: MlxHabitatRolloutLookaheadConfig,
    strategy_id: String,
}

impl MlxHabitatRolloutLookaheadStrategy<ModelProcess> {
    pub fn spawn<I, S>(
        program: impl AsRef<OsStr>,
        args: I,
        config: MlxHabitatRolloutLookaheadConfig,
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

impl<P: RankingPredictor> MlxHabitatRolloutLookaheadStrategy<P> {
    pub fn with_predictor(
        predictor: P,
        config: MlxHabitatRolloutLookaheadConfig,
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
        let (prelude, staged, candidates) = habitat_candidate_union(
            game,
            self.config.immediate_candidate_limit,
            self.config.habitat_candidate_limit,
        )?;
        rank_mlx_rollout_candidates(
            &mut self.predictor,
            game,
            &staged,
            candidates,
            &prelude,
            self.config,
            &mut lookahead_decision_rng(game),
        )
    }

    pub fn select_action(&mut self, game: &GameState) -> Result<TurnAction, SearchError> {
        let ranked = self.rank_actions(game)?;
        select_ranked_action(&ranked, &mut lookahead_decision_rng(game))
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

struct MlxRolloutBranch {
    root_candidate_index: usize,
    state: GameState,
}

fn rank_mlx_rollout_candidates<P: RankingPredictor>(
    predictor: &mut P,
    game: &GameState,
    staged: &GameState,
    candidates: Vec<GreedyCandidate>,
    prelude: &MarketPrelude,
    config: MlxHabitatRolloutLookaheadConfig,
    rng: &mut ChaCha8Rng,
) -> Result<Vec<RolloutCandidate>, SearchError> {
    if candidates.is_empty() {
        return Err(SearchError::NoLegalActions);
    }
    let acting_seat = staged.current_player();
    let mut sample_seeds = Vec::with_capacity(config.determinizations);
    for _ in 0..config.determinizations {
        let mut seed = [0; 32];
        rng.fill(&mut seed);
        sample_seeds.push(GameSeed(seed));
    }
    let mut branches = Vec::with_capacity(candidates.len() * sample_seeds.len());
    for (candidate_index, candidate) in candidates.iter().enumerate() {
        for sample_seed in &sample_seeds {
            let mut state = staged.clone();
            state.redeterminize_hidden(*sample_seed);
            state.apply(&candidate.action)?;
            branches.push(MlxRolloutBranch {
                root_candidate_index: candidate_index,
                state,
            });
        }
    }

    for _ in 0..config.rollout_plies {
        play_batched_mlx_rollout_ply(predictor, &mut branches, config)?;
        if branches.iter().all(|branch| branch.state.is_game_over()) {
            break;
        }
    }

    let cards = game.config().scoring_cards;
    let mut values = vec![Vec::with_capacity(sample_seeds.len()); candidates.len()];
    for branch in branches {
        values[branch.root_candidate_index].push(f64::from(
            score_board(&branch.state.boards()[acting_seat], cards).base_total,
        ));
    }
    let mut ranked = candidates
        .into_iter()
        .zip(values)
        .map(|(candidate, values)| {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = if values.len() > 1 {
                values
                    .iter()
                    .map(|value| (value - mean).powi(2))
                    .sum::<f64>()
                    / (values.len() - 1) as f64
            } else {
                0.0
            };
            RolloutCandidate {
                action: with_prelude(candidate.action, prelude),
                immediate_rank: candidate.immediate_rank,
                immediate_score: candidate.resulting_base_score,
                mean_leaf_score: mean,
                leaf_score_stddev: variance.sqrt(),
            }
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

struct MlxRolloutGroup {
    branch_index: usize,
    actions: Vec<TurnAction>,
    immediate_scores: Vec<u16>,
    prediction_start: usize,
}

fn play_batched_mlx_rollout_ply<P: RankingPredictor>(
    predictor: &mut P,
    branches: &mut [MlxRolloutBranch],
    config: MlxHabitatRolloutLookaheadConfig,
) -> Result<(), SearchError> {
    let mut groups = Vec::new();
    let mut records = Vec::new();
    for (branch_index, branch) in branches.iter().enumerate() {
        if branch.state.is_game_over() {
            continue;
        }
        let (prelude, _, candidates) = habitat_candidate_union(
            &branch.state,
            config.rollout_immediate_candidate_limit,
            config.rollout_habitat_candidate_limit,
        )?;
        let actions: Vec<_> = candidates
            .iter()
            .map(|candidate| with_prelude(candidate.action.clone(), &prelude))
            .collect();
        let prediction_start = records.len();
        records.extend(
            actions
                .iter()
                .map(|action| PositionRecord::observable_afterstate(&branch.state, action, 0))
                .collect::<Result<Vec<_>, _>>()?,
        );
        groups.push(MlxRolloutGroup {
            branch_index,
            actions,
            immediate_scores: candidates
                .iter()
                .map(|candidate| candidate.resulting_base_score)
                .collect(),
            prediction_start,
        });
    }
    if records.is_empty() {
        return Ok(());
    }
    let mut predictions = Vec::with_capacity(records.len());
    for chunk in records.chunks(MAX_BATCH) {
        predictions.extend(predictor.predict_scores(chunk)?);
    }
    if predictions.len() != records.len() {
        return Err(SearchError::PredictionCount {
            expected: records.len(),
            actual: predictions.len(),
        });
    }
    if let Some((index, _)) = predictions
        .iter()
        .enumerate()
        .find(|(_, score)| !score.is_finite())
    {
        return Err(SearchError::NonFinitePrediction { index });
    }
    for group in groups {
        let prediction_end = group.prediction_start + group.actions.len();
        let group_predictions = &predictions[group.prediction_start..prediction_end];
        let best_index = (0..group.actions.len())
            .max_by(|left, right| {
                group_predictions[*left]
                    .total_cmp(&group_predictions[*right])
                    .then_with(|| {
                        group.immediate_scores[*left].cmp(&group.immediate_scores[*right])
                    })
            })
            .ok_or(SearchError::NoLegalActions)?;
        branches[group.branch_index]
            .state
            .apply(&group.actions[best_index])?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlxSelfRolloutLookaheadConfig {
    pub immediate_candidate_limit: usize,
    pub habitat_candidate_limit: usize,
    pub determinizations: usize,
    pub rollout_plies: usize,
    pub policy_immediate_candidate_limit: usize,
    pub policy_habitat_candidate_limit: usize,
}

impl MlxSelfRolloutLookaheadConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        if self.immediate_candidate_limit == 0
            || self.habitat_candidate_limit == 0
            || self.policy_immediate_candidate_limit == 0
            || self.policy_habitat_candidate_limit == 0
        {
            return Err(SearchError::InvalidConfig(
                "MLX self-rollout candidate limits must be positive",
            ));
        }
        if self.determinizations == 0 || self.rollout_plies == 0 {
            return Err(SearchError::InvalidConfig(
                "MLX self-rollout sampling and horizon must be positive",
            ));
        }
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{MLX_SELF_ROLLOUT_LOOKAHEAD_STRATEGY_ID}-k{}-h{}-r{}-d{}-pk{}-ph{}",
            self.immediate_candidate_limit,
            self.habitat_candidate_limit,
            self.determinizations,
            self.rollout_plies,
            self.policy_immediate_candidate_limit,
            self.policy_habitat_candidate_limit,
        )
    }
}

impl Default for MlxSelfRolloutLookaheadConfig {
    fn default() -> Self {
        Self {
            immediate_candidate_limit: 8,
            habitat_candidate_limit: 6,
            determinizations: 4,
            rollout_plies: 4,
            policy_immediate_candidate_limit: 8,
            policy_habitat_candidate_limit: 6,
        }
    }
}

pub struct MlxSelfRolloutLookaheadStrategy<P = ModelProcess> {
    predictor: P,
    config: MlxSelfRolloutLookaheadConfig,
    strategy_id: String,
}

impl MlxSelfRolloutLookaheadStrategy<ModelProcess> {
    pub fn spawn<I, S>(
        program: impl AsRef<OsStr>,
        args: I,
        config: MlxSelfRolloutLookaheadConfig,
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

impl<P: RankingPredictor> MlxSelfRolloutLookaheadStrategy<P> {
    pub fn with_predictor(
        predictor: P,
        config: MlxSelfRolloutLookaheadConfig,
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
        let (prelude, staged, candidates) = habitat_candidate_union(
            game,
            self.config.immediate_candidate_limit,
            self.config.habitat_candidate_limit,
        )?;
        rank_mlx_self_rollout_candidates(
            &mut self.predictor,
            game,
            &staged,
            candidates,
            &prelude,
            self.config,
            &mut lookahead_decision_rng(game),
        )
    }

    pub fn select_action(&mut self, game: &GameState) -> Result<TurnAction, SearchError> {
        let ranked = self.rank_actions(game)?;
        select_ranked_action(&ranked, &mut lookahead_decision_rng(game))
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

struct MlxSelfRolloutBranch {
    root_candidate_index: usize,
    state: GameState,
    rollout_rng: ChaCha8Rng,
}

fn rank_mlx_self_rollout_candidates<P: RankingPredictor>(
    predictor: &mut P,
    game: &GameState,
    staged: &GameState,
    candidates: Vec<GreedyCandidate>,
    prelude: &MarketPrelude,
    config: MlxSelfRolloutLookaheadConfig,
    rng: &mut ChaCha8Rng,
) -> Result<Vec<RolloutCandidate>, SearchError> {
    if candidates.is_empty() {
        return Err(SearchError::NoLegalActions);
    }
    let acting_seat = staged.current_player();
    let mut sample_seeds = Vec::with_capacity(config.determinizations);
    for _ in 0..config.determinizations {
        let mut seed = [0; 32];
        rng.fill(&mut seed);
        sample_seeds.push(GameSeed(seed));
    }
    let mut branches = Vec::with_capacity(candidates.len() * sample_seeds.len());
    for (candidate_index, candidate) in candidates.iter().enumerate() {
        for sample_seed in &sample_seeds {
            let mut state = staged.clone();
            state.redeterminize_hidden(*sample_seed);
            state.apply(&candidate.action)?;
            branches.push(MlxSelfRolloutBranch {
                root_candidate_index: candidate_index,
                state,
                rollout_rng: rollout_rng(*sample_seed),
            });
        }
    }

    for _ in 0..config.rollout_plies {
        play_mlx_self_rollout_ply(predictor, &mut branches, acting_seat, config)?;
        if branches.iter().all(|branch| branch.state.is_game_over()) {
            break;
        }
    }

    let cards = game.config().scoring_cards;
    let mut values = vec![Vec::with_capacity(sample_seeds.len()); candidates.len()];
    for branch in branches {
        values[branch.root_candidate_index].push(f64::from(
            score_board(&branch.state.boards()[acting_seat], cards).base_total,
        ));
    }
    finish_rollout_ranking(candidates, values, prelude)
}

fn play_mlx_self_rollout_ply<P: RankingPredictor>(
    predictor: &mut P,
    branches: &mut [MlxSelfRolloutBranch],
    acting_seat: usize,
    config: MlxSelfRolloutLookaheadConfig,
) -> Result<(), SearchError> {
    let model_turns = branches
        .iter()
        .map(|branch| !branch.state.is_game_over() && branch.state.current_player() == acting_seat)
        .collect::<Vec<_>>();
    let greedy_results = branches
        .par_iter_mut()
        .zip(&model_turns)
        .map(|(branch, model_turn)| {
            if branch.state.is_game_over() || *model_turn {
                return Ok(());
            }
            play_greedy_plies(&mut branch.state, 1, &mut branch.rollout_rng)?;
            Ok::<_, SearchError>(())
        })
        .collect::<Vec<_>>();
    for result in greedy_results {
        result?;
    }
    play_batched_self_policy_ply(predictor, branches, &model_turns, config)
}

fn play_batched_self_policy_ply<P: RankingPredictor>(
    predictor: &mut P,
    branches: &mut [MlxSelfRolloutBranch],
    model_turns: &[bool],
    config: MlxSelfRolloutLookaheadConfig,
) -> Result<(), SearchError> {
    let mut groups = Vec::new();
    let mut records = Vec::new();
    for (branch_index, (branch, model_turn)) in branches.iter().zip(model_turns).enumerate() {
        if !model_turn {
            continue;
        }
        let (prelude, _, candidates) = habitat_candidate_union(
            &branch.state,
            config.policy_immediate_candidate_limit,
            config.policy_habitat_candidate_limit,
        )?;
        let actions = candidates
            .iter()
            .map(|candidate| with_prelude(candidate.action.clone(), &prelude))
            .collect::<Vec<_>>();
        let prediction_start = records.len();
        records.extend(
            actions
                .iter()
                .map(|action| PositionRecord::observable_afterstate(&branch.state, action, 0))
                .collect::<Result<Vec<_>, _>>()?,
        );
        groups.push(MlxRolloutGroup {
            branch_index,
            actions,
            immediate_scores: candidates
                .iter()
                .map(|candidate| candidate.resulting_base_score)
                .collect(),
            prediction_start,
        });
    }
    if records.is_empty() {
        return Ok(());
    }
    let mut predictions = Vec::with_capacity(records.len());
    for chunk in records.chunks(MAX_BATCH) {
        predictions.extend(predictor.predict_scores(chunk)?);
    }
    if predictions.len() != records.len() {
        return Err(SearchError::PredictionCount {
            expected: records.len(),
            actual: predictions.len(),
        });
    }
    if let Some((index, _)) = predictions
        .iter()
        .enumerate()
        .find(|(_, score)| !score.is_finite())
    {
        return Err(SearchError::NonFinitePrediction { index });
    }
    for group in groups {
        let prediction_end = group.prediction_start + group.actions.len();
        let group_predictions = &predictions[group.prediction_start..prediction_end];
        let best_index = (0..group.actions.len())
            .max_by(|left, right| {
                group_predictions[*left]
                    .total_cmp(&group_predictions[*right])
                    .then_with(|| {
                        group.immediate_scores[*left].cmp(&group.immediate_scores[*right])
                    })
            })
            .ok_or(SearchError::NoLegalActions)?;
        branches[group.branch_index]
            .state
            .apply(&group.actions[best_index])?;
    }
    Ok(())
}

pub(crate) fn finish_rollout_ranking(
    candidates: Vec<GreedyCandidate>,
    values: Vec<Vec<f64>>,
    prelude: &MarketPrelude,
) -> Result<Vec<RolloutCandidate>, SearchError> {
    if candidates.len() != values.len() || values.iter().any(Vec::is_empty) {
        return Err(SearchError::InvalidConfig(
            "rollout evaluation produced incomplete candidate values",
        ));
    }
    let mut ranked = candidates
        .into_iter()
        .zip(values)
        .map(|(candidate, values)| {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = if values.len() > 1 {
                values
                    .iter()
                    .map(|value| (value - mean).powi(2))
                    .sum::<f64>()
                    / (values.len() - 1) as f64
            } else {
                0.0
            };
            RolloutCandidate {
                action: with_prelude(candidate.action, prelude),
                immediate_rank: candidate.immediate_rank,
                immediate_score: candidate.resulting_base_score,
                mean_leaf_score: mean,
                leaf_score_stddev: variance.sqrt(),
            }
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
