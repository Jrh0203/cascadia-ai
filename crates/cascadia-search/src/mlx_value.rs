use std::ffi::OsStr;

use cascadia_data::PositionRecord;
use cascadia_game::{GameConfig, GameSeed, GameState, MarketPrelude, TurnAction, score_board};
use cascadia_model::{MAX_BATCH, ModelProcess, Prediction};
use cascadia_sim::{
    GreedyCandidate, MatchResult, SimulationError, play_greedy_plies, play_match_with_selector,
    rank_greedy_actions,
};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use super::{
    MLX_VALUE_LEAF_LOOKAHEAD_STRATEGY_ID, MLX_VALUE_STRATEGY_ID, RolloutCandidate, SearchError,
    habitat_candidate_union, lookahead_decision_rng, rollout_rng, select_ranked_action,
    strategy_rng, with_prelude,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MlxValueConfig {
    pub greedy_prefilter: Option<usize>,
}

impl MlxValueConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        if self.greedy_prefilter == Some(0) {
            return Err(SearchError::InvalidConfig(
                "greedy prefilter must retain at least one candidate",
            ));
        }
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        match self.greedy_prefilter {
            Some(limit) => format!("mlx-value-v1-greedy-top-{limit}"),
            None => MLX_VALUE_STRATEGY_ID.to_owned(),
        }
    }
}

pub trait Predictor {
    fn predict(&mut self, records: &[PositionRecord]) -> Result<Vec<Prediction>, SearchError>;
}

impl Predictor for ModelProcess {
    fn predict(&mut self, records: &[PositionRecord]) -> Result<Vec<Prediction>, SearchError> {
        Ok(ModelProcess::predict(self, records)?)
    }
}

pub struct MlxValueStrategy<P = ModelProcess> {
    predictor: P,
    config: MlxValueConfig,
    strategy_id: String,
}

impl MlxValueStrategy<ModelProcess> {
    pub fn spawn<I, S>(program: impl AsRef<OsStr>, args: I) -> Result<Self, SearchError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        Self::spawn_with_config(program, args, MlxValueConfig::default())
    }

    pub fn spawn_with_config<I, S>(
        program: impl AsRef<OsStr>,
        args: I,
        config: MlxValueConfig,
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

impl<P: Predictor> MlxValueStrategy<P> {
    pub fn with_predictor(predictor: P) -> Self {
        Self::with_predictor_and_config(predictor, MlxValueConfig::default())
            .expect("default MLX value configuration is valid")
    }

    pub fn with_predictor_and_config(
        predictor: P,
        config: MlxValueConfig,
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

    pub fn select_action(
        &mut self,
        game: &GameState,
        rng: &mut ChaCha8Rng,
    ) -> Result<TurnAction, SearchError> {
        let prelude = MarketPrelude {
            replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
            wildlife_wipes: Vec::new(),
        };
        let actions: Vec<_> = match self.config.greedy_prefilter {
            Some(limit) => rank_greedy_actions(game, &prelude, Some(limit))?
                .into_iter()
                .map(|candidate| candidate.action)
                .collect(),
            None => game.legal_turn_actions(&prelude)?,
        };
        if actions.is_empty() {
            return Err(SearchError::NoLegalActions);
        }
        let candidates = actions
            .iter()
            .map(|action| PositionRecord::observable_afterstate(game, action, 0))
            .collect::<Result<Vec<_>, _>>()?;

        let mut predictions = Vec::with_capacity(candidates.len());
        for chunk in candidates.chunks(MAX_BATCH) {
            predictions.extend(self.predictor.predict(chunk)?);
        }
        if predictions.len() != candidates.len() {
            return Err(SearchError::PredictionCount {
                expected: candidates.len(),
                actual: predictions.len(),
            });
        }

        let mut best_value = f32::NEG_INFINITY;
        let mut best_indices = Vec::new();
        for (index, prediction) in predictions.iter().enumerate() {
            let value = prediction.iter().sum::<f32>();
            match value.total_cmp(&best_value) {
                std::cmp::Ordering::Greater => {
                    best_value = value;
                    best_indices.clear();
                    best_indices.push(index);
                }
                std::cmp::Ordering::Equal => best_indices.push(index),
                std::cmp::Ordering::Less => {}
            }
        }
        let chosen = best_indices[rng.gen_range(0..best_indices.len())];
        Ok(actions[chosen].clone())
    }

    pub fn play_match(
        &mut self,
        game_config: GameConfig,
        seed: GameSeed,
    ) -> Result<MatchResult, SearchError> {
        let mut rngs: Vec<_> = (0..game_config.player_count)
            .map(|seat| strategy_rng(seed, seat))
            .collect();
        let strategy_id = self.strategy_id.clone();
        Ok(play_match_with_selector(
            game_config,
            seed,
            &strategy_id,
            |seat, game| {
                self.select_action(game, &mut rngs[seat])
                    .map_err(|error| SimulationError::Strategy(error.to_string()))
            },
        )?)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlxValueLeafLookaheadConfig {
    pub immediate_candidate_limit: usize,
    pub habitat_candidate_limit: usize,
    pub determinizations: usize,
    pub greedy_plies: usize,
}

impl MlxValueLeafLookaheadConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        if self.immediate_candidate_limit == 0 || self.habitat_candidate_limit == 0 {
            return Err(SearchError::InvalidConfig(
                "MLX value-leaf candidate limits must be positive",
            ));
        }
        if self.determinizations == 0 {
            return Err(SearchError::InvalidConfig(
                "MLX value-leaf lookahead must use at least one determinization",
            ));
        }
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{MLX_VALUE_LEAF_LOOKAHEAD_STRATEGY_ID}-k{}-h{}-r{}-d{}",
            self.immediate_candidate_limit,
            self.habitat_candidate_limit,
            self.determinizations,
            self.greedy_plies,
        )
    }
}

impl Default for MlxValueLeafLookaheadConfig {
    fn default() -> Self {
        Self {
            immediate_candidate_limit: 8,
            habitat_candidate_limit: 6,
            determinizations: 4,
            greedy_plies: 4,
        }
    }
}

pub struct MlxValueLeafLookaheadStrategy<P = ModelProcess> {
    predictor: P,
    config: MlxValueLeafLookaheadConfig,
    strategy_id: String,
}

impl MlxValueLeafLookaheadStrategy<ModelProcess> {
    pub fn spawn<I, S>(
        program: impl AsRef<OsStr>,
        args: I,
        config: MlxValueLeafLookaheadConfig,
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

impl<P: Predictor> MlxValueLeafLookaheadStrategy<P> {
    pub fn with_predictor(
        predictor: P,
        config: MlxValueLeafLookaheadConfig,
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
        rank_mlx_value_leaf_candidates(
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

struct ValueLeafBranch {
    root_candidate_index: usize,
    state: GameState,
}

fn rank_mlx_value_leaf_candidates<P: Predictor>(
    predictor: &mut P,
    game: &GameState,
    staged: &GameState,
    candidates: Vec<GreedyCandidate>,
    prelude: &MarketPrelude,
    config: MlxValueLeafLookaheadConfig,
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
    let sample_count = sample_seeds.len();
    let branches: Vec<Result<ValueLeafBranch, SearchError>> = (0..candidates.len() * sample_count)
        .into_par_iter()
        .map(|job| {
            let candidate_index = job / sample_count;
            let sample_seed = sample_seeds[job % sample_count];
            let mut state = staged.clone();
            state.redeterminize_hidden(sample_seed);
            state.apply(&candidates[candidate_index].action)?;
            let mut rollout_rng = rollout_rng(sample_seed);
            play_greedy_plies(&mut state, config.greedy_plies, &mut rollout_rng)?;
            Ok(ValueLeafBranch {
                root_candidate_index: candidate_index,
                state,
            })
        })
        .collect();
    let branches = branches.into_iter().collect::<Result<Vec<_>, _>>()?;
    let mut records = Vec::new();
    let mut model_branch_indices = Vec::new();
    let mut branch_values = vec![None; branches.len()];
    let cards = game.config().scoring_cards;
    for (branch_index, branch) in branches.iter().enumerate() {
        if branch.state.is_game_over() {
            branch_values[branch_index] = Some(f64::from(
                score_board(&branch.state.boards()[acting_seat], cards).base_total,
            ));
        } else {
            records.push(PositionRecord::observe_for_seat(
                &branch.state,
                0,
                acting_seat,
            ));
            model_branch_indices.push(branch_index);
        }
    }
    let mut predictions = Vec::with_capacity(records.len());
    for chunk in records.chunks(MAX_BATCH) {
        predictions.extend(predictor.predict(chunk)?);
    }
    if predictions.len() != records.len() {
        return Err(SearchError::PredictionCount {
            expected: records.len(),
            actual: predictions.len(),
        });
    }
    for (prediction_index, prediction) in predictions.into_iter().enumerate() {
        if prediction.iter().any(|component| !component.is_finite()) {
            return Err(SearchError::NonFinitePrediction {
                index: prediction_index,
            });
        }
        branch_values[model_branch_indices[prediction_index]] =
            Some(f64::from(prediction.iter().sum::<f32>()));
    }

    let mut values = vec![Vec::with_capacity(sample_count); candidates.len()];
    for (branch, value) in branches.into_iter().zip(branch_values) {
        values[branch.root_candidate_index]
            .push(value.expect("every value leaf is exact or model-evaluated"));
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
