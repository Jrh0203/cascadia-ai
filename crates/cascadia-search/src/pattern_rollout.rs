use cascadia_game::{GameConfig, GameSeed, GameState, TurnAction, score_board};
use cascadia_sim::{
    MatchResult, PatternAwareConfig, SimulationError, play_match_with_selector, play_pattern_plies,
};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::{
    RolloutCandidate, SearchError, finish_rollout_ranking, habitat_candidate_union,
    lookahead_decision_rng, rollout_rng, select_ranked_action,
};

pub const PATTERN_BLUEPRINT_LOOKAHEAD_STRATEGY_ID: &str = "pattern-blueprint-lookahead-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PatternBlueprintLookaheadConfig {
    pub immediate_candidate_limit: usize,
    pub habitat_candidate_limit: usize,
    pub determinizations: usize,
    pub rollout_plies: usize,
    pub blueprint: PatternAwareConfig,
}

impl PatternBlueprintLookaheadConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        if self.immediate_candidate_limit == 0 {
            return Err(SearchError::InvalidConfig(
                "pattern-blueprint lookahead must retain an immediate-score candidate",
            ));
        }
        if self.habitat_candidate_limit == 0 {
            return Err(SearchError::InvalidConfig(
                "pattern-blueprint lookahead must retain a habitat candidate",
            ));
        }
        if self.determinizations == 0 {
            return Err(SearchError::InvalidConfig(
                "pattern-blueprint lookahead must use at least one determinization",
            ));
        }
        self.blueprint.validate()?;
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{PATTERN_BLUEPRINT_LOOKAHEAD_STRATEGY_ID}-k{}-h{}-r{}-d{}-pk{}-ph{}-pb{}-pm{}",
            self.immediate_candidate_limit,
            self.habitat_candidate_limit,
            self.determinizations,
            self.rollout_plies,
            self.blueprint.immediate_candidate_limit,
            self.blueprint.habitat_candidate_limit,
            self.blueprint.bear_candidate_limit,
            self.blueprint.future_market_draws,
        )
    }
}

impl Default for PatternBlueprintLookaheadConfig {
    fn default() -> Self {
        Self {
            immediate_candidate_limit: 8,
            habitat_candidate_limit: 6,
            determinizations: 4,
            rollout_plies: 4,
            blueprint: PatternAwareConfig::default(),
        }
    }
}

pub struct PatternBlueprintLookaheadStrategy {
    config: PatternBlueprintLookaheadConfig,
    strategy_id: String,
}

impl PatternBlueprintLookaheadStrategy {
    pub fn new(config: PatternBlueprintLookaheadConfig) -> Result<Self, SearchError> {
        let config = config.validate()?;
        Ok(Self {
            strategy_id: config.strategy_id(),
            config,
        })
    }

    pub fn strategy_id(&self) -> &str {
        &self.strategy_id
    }

    pub fn rank_actions(
        &self,
        game: &GameState,
        rng: &mut ChaCha8Rng,
    ) -> Result<Vec<RolloutCandidate>, SearchError> {
        let (prelude, staged, candidates) = habitat_candidate_union(
            game,
            self.config.immediate_candidate_limit,
            self.config.habitat_candidate_limit,
        )?;
        if candidates.is_empty() {
            return Err(SearchError::NoLegalActions);
        }

        let acting_seat = staged.current_player();
        let cards = game.config().scoring_cards;
        let mut sample_seeds = Vec::with_capacity(self.config.determinizations);
        for _ in 0..self.config.determinizations {
            let mut seed = [0; 32];
            rng.fill(&mut seed);
            sample_seeds.push(GameSeed(seed));
        }

        let sample_count = sample_seeds.len();
        let scores: Vec<Result<(usize, f64), SearchError>> = (0..candidates.len() * sample_count)
            .into_par_iter()
            .map(|job| {
                let candidate_index = job / sample_count;
                let sample_seed = sample_seeds[job % sample_count];
                let mut sample = staged.clone();
                sample.redeterminize_hidden(sample_seed);
                sample.apply(&candidates[candidate_index].action)?;
                let mut rollout_rng = rollout_rng(sample_seed);
                play_pattern_plies(
                    &mut sample,
                    self.config.rollout_plies,
                    self.config.blueprint,
                    &mut rollout_rng,
                )?;
                Ok((
                    candidate_index,
                    f64::from(score_board(&sample.boards()[acting_seat], cards).base_total),
                ))
            })
            .collect();

        let mut values = vec![Vec::with_capacity(sample_count); candidates.len()];
        for score in scores {
            let (candidate_index, score) = score?;
            values[candidate_index].push(score);
        }
        finish_rollout_ranking(candidates, values, &prelude)
    }

    pub fn rank_and_select_deterministic(
        &self,
        game: &GameState,
    ) -> Result<(Vec<RolloutCandidate>, TurnAction), SearchError> {
        let mut rng = lookahead_decision_rng(game);
        let ranked = self.rank_actions(game, &mut rng)?;
        let action = select_ranked_action(&ranked, &mut rng)?;
        Ok((ranked, action))
    }

    pub fn select_action_deterministic(&self, game: &GameState) -> Result<TurnAction, SearchError> {
        Ok(self.rank_and_select_deterministic(game)?.1)
    }

    pub fn play_match(
        &self,
        game_config: GameConfig,
        seed: GameSeed,
    ) -> Result<MatchResult, SearchError> {
        Ok(play_match_with_selector(
            game_config,
            seed,
            &self.strategy_id,
            |_, game| {
                self.select_action_deterministic(game)
                    .map_err(|error| SimulationError::Strategy(error.to_string()))
            },
        )?)
    }
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameConfig, GameSeed, GameState};

    use super::*;

    fn tiny_config() -> PatternBlueprintLookaheadConfig {
        PatternBlueprintLookaheadConfig {
            immediate_candidate_limit: 2,
            habitat_candidate_limit: 1,
            determinizations: 1,
            rollout_plies: 1,
            blueprint: PatternAwareConfig {
                immediate_candidate_limit: 2,
                habitat_candidate_limit: 1,
                bear_candidate_limit: 1,
                future_market_draws: 2,
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
    }

    #[test]
    fn strategy_id_captures_root_and_blueprint_configuration() {
        assert_eq!(
            tiny_config().strategy_id(),
            "pattern-blueprint-lookahead-v1-k2-h1-r1-d1-pk2-ph1-pb1-pm2"
        );
    }

    #[test]
    fn pattern_blueprint_lookahead_is_legal_reproducible_and_ranked() {
        let game = GameState::new(
            GameConfig::research_aaaaa(2).unwrap(),
            GameSeed::from_u64(19),
        )
        .unwrap();
        let strategy = PatternBlueprintLookaheadStrategy::new(tiny_config()).unwrap();

        let left = strategy.rank_and_select_deterministic(&game).unwrap();
        let right = strategy.rank_and_select_deterministic(&game).unwrap();

        assert_eq!(left, right);
        assert!((2..=3).contains(&left.0.len()));
        assert!(
            left.0
                .windows(2)
                .all(|pair| pair[0].mean_leaf_score >= pair[1].mean_leaf_score)
        );
        game.transition(&left.1).unwrap();
    }
}
