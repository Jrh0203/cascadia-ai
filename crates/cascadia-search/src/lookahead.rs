use cascadia_game::{
    GameConfig, GameSeed, GameState, MarketPrelude, TurnAction, WildlifeWipe, score_board,
};
use cascadia_sim::{
    GreedyCandidate, MatchResult, SimulationError, play_greedy_plies, play_match_with_selector,
    rank_bear_setup_actions, rank_greedy_actions, rank_habitat_setup_actions,
};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use super::{
    DETERMINIZED_LOOKAHEAD_STRATEGY_ID, SearchError, conditioned_rollout_seed,
    lookahead_decision_rng, nature_wipe_decision_rng, rollout_rng,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeterminizedLookaheadConfig {
    pub candidate_limit: usize,
    pub determinizations: usize,
    pub greedy_plies: usize,
}

impl DeterminizedLookaheadConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        if self.candidate_limit == 0 {
            return Err(SearchError::InvalidConfig(
                "lookahead must retain at least one candidate",
            ));
        }
        if self.determinizations == 0 {
            return Err(SearchError::InvalidConfig(
                "lookahead must use at least one determinization",
            ));
        }
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{DETERMINIZED_LOOKAHEAD_STRATEGY_ID}-k{}-r{}-d{}",
            self.candidate_limit, self.determinizations, self.greedy_plies
        )
    }
}

impl Default for DeterminizedLookaheadConfig {
    fn default() -> Self {
        Self {
            candidate_limit: 8,
            determinizations: 4,
            greedy_plies: 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RolloutCandidate {
    pub action: TurnAction,
    pub immediate_rank: usize,
    pub immediate_score: u16,
    pub mean_leaf_score: f64,
    pub leaf_score_stddev: f64,
}

pub struct DeterminizedLookaheadStrategy {
    config: DeterminizedLookaheadConfig,
    strategy_id: String,
}

impl DeterminizedLookaheadStrategy {
    pub fn new(config: DeterminizedLookaheadConfig) -> Result<Self, SearchError> {
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
        let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
        self.rank_actions_with_prelude(game, &prelude, rng)
    }

    pub fn rank_actions_with_prelude(
        &self,
        game: &GameState,
        prelude: &MarketPrelude,
        rng: &mut ChaCha8Rng,
    ) -> Result<Vec<RolloutCandidate>, SearchError> {
        let staged = game.preview_market_prelude(prelude)?;
        let candidates = rank_greedy_actions(
            &staged,
            &MarketPrelude::default(),
            Some(self.config.candidate_limit),
        )?;
        rank_rollout_candidates(
            game,
            &staged,
            candidates,
            prelude,
            self.config.determinizations,
            self.config.greedy_plies,
            rng,
        )
    }

    pub fn select_action(
        &self,
        game: &GameState,
        rng: &mut ChaCha8Rng,
    ) -> Result<TurnAction, SearchError> {
        Ok(self.rank_and_select(game, rng)?.1)
    }

    pub fn rank_and_select(
        &self,
        game: &GameState,
        rng: &mut ChaCha8Rng,
    ) -> Result<(Vec<RolloutCandidate>, TurnAction), SearchError> {
        let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
        self.rank_and_select_with_prelude(game, &prelude, rng)
    }

    pub fn rank_and_select_with_prelude(
        &self,
        game: &GameState,
        prelude: &MarketPrelude,
        rng: &mut ChaCha8Rng,
    ) -> Result<(Vec<RolloutCandidate>, TurnAction), SearchError> {
        let ranked = self.rank_actions_with_prelude(game, prelude, rng)?;
        let best_value = ranked[0].mean_leaf_score;
        let tied = ranked
            .iter()
            .take_while(|candidate| candidate.mean_leaf_score == best_value)
            .count();
        let action = ranked[rng.gen_range(0..tied)].action.clone();
        Ok((ranked, action))
    }

    pub fn rank_actions_deterministic(
        &self,
        game: &GameState,
    ) -> Result<Vec<RolloutCandidate>, SearchError> {
        self.rank_actions(game, &mut lookahead_decision_rng(game))
    }

    pub fn select_action_deterministic(&self, game: &GameState) -> Result<TurnAction, SearchError> {
        self.select_action(game, &mut lookahead_decision_rng(game))
    }

    pub fn rank_and_select_deterministic(
        &self,
        game: &GameState,
    ) -> Result<(Vec<RolloutCandidate>, TurnAction), SearchError> {
        self.rank_and_select(game, &mut lookahead_decision_rng(game))
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

pub const BEAR_CANDIDATE_LOOKAHEAD_STRATEGY_ID: &str = "bear-candidate-lookahead-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BearCandidateLookaheadConfig {
    pub immediate_candidate_limit: usize,
    pub bear_candidate_limit: usize,
    pub determinizations: usize,
    pub greedy_plies: usize,
}

impl BearCandidateLookaheadConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        if self.immediate_candidate_limit == 0 {
            return Err(SearchError::InvalidConfig(
                "bear-aware lookahead must retain an immediate-score candidate",
            ));
        }
        if self.bear_candidate_limit == 0 {
            return Err(SearchError::InvalidConfig(
                "bear-aware lookahead must retain a Bear candidate",
            ));
        }
        if self.determinizations == 0 {
            return Err(SearchError::InvalidConfig(
                "bear-aware lookahead must use at least one determinization",
            ));
        }
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{BEAR_CANDIDATE_LOOKAHEAD_STRATEGY_ID}-k{}-b{}-r{}-d{}",
            self.immediate_candidate_limit,
            self.bear_candidate_limit,
            self.determinizations,
            self.greedy_plies
        )
    }
}

impl Default for BearCandidateLookaheadConfig {
    fn default() -> Self {
        Self {
            immediate_candidate_limit: 8,
            bear_candidate_limit: 8,
            determinizations: 4,
            greedy_plies: 4,
        }
    }
}

pub struct BearCandidateLookaheadStrategy {
    config: BearCandidateLookaheadConfig,
    strategy_id: String,
}

impl BearCandidateLookaheadStrategy {
    pub fn new(config: BearCandidateLookaheadConfig) -> Result<Self, SearchError> {
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
        let (prelude, staged, candidates) = bear_candidate_union(
            game,
            self.config.immediate_candidate_limit,
            self.config.bear_candidate_limit,
        )?;
        rank_rollout_candidates(
            game,
            &staged,
            candidates,
            &prelude,
            self.config.determinizations,
            self.config.greedy_plies,
            rng,
        )
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

pub(crate) fn bear_candidate_union(
    game: &GameState,
    immediate_candidate_limit: usize,
    bear_candidate_limit: usize,
) -> Result<(MarketPrelude, GameState, Vec<GreedyCandidate>), SearchError> {
    let (prelude, staged) = game.preview_free_three_of_a_kind_if_feasible()?;
    let mut candidates = rank_greedy_actions(
        &staged,
        &MarketPrelude::default(),
        Some(immediate_candidate_limit),
    )?;
    merge_unique_candidates(
        &mut candidates,
        rank_bear_setup_actions(
            &staged,
            &MarketPrelude::default(),
            Some(bear_candidate_limit),
        )?,
    );
    Ok((prelude, staged, candidates))
}

pub const HABITAT_CANDIDATE_LOOKAHEAD_STRATEGY_ID: &str = "habitat-candidate-lookahead-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HabitatCandidateLookaheadConfig {
    pub immediate_candidate_limit: usize,
    pub habitat_candidate_limit: usize,
    pub determinizations: usize,
    pub greedy_plies: usize,
}

impl HabitatCandidateLookaheadConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        if self.immediate_candidate_limit == 0 {
            return Err(SearchError::InvalidConfig(
                "habitat-aware lookahead must retain an immediate-score candidate",
            ));
        }
        if self.habitat_candidate_limit == 0 {
            return Err(SearchError::InvalidConfig(
                "habitat-aware lookahead must retain a habitat candidate",
            ));
        }
        if self.determinizations == 0 {
            return Err(SearchError::InvalidConfig(
                "habitat-aware lookahead must use at least one determinization",
            ));
        }
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{HABITAT_CANDIDATE_LOOKAHEAD_STRATEGY_ID}-k{}-h{}-r{}-d{}",
            self.immediate_candidate_limit,
            self.habitat_candidate_limit,
            self.determinizations,
            self.greedy_plies
        )
    }
}

impl Default for HabitatCandidateLookaheadConfig {
    fn default() -> Self {
        Self {
            immediate_candidate_limit: 8,
            habitat_candidate_limit: 8,
            determinizations: 4,
            greedy_plies: 4,
        }
    }
}

pub struct HabitatCandidateLookaheadStrategy {
    config: HabitatCandidateLookaheadConfig,
    strategy_id: String,
}

impl HabitatCandidateLookaheadStrategy {
    pub fn new(config: HabitatCandidateLookaheadConfig) -> Result<Self, SearchError> {
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
        rank_rollout_candidates(
            game,
            &staged,
            candidates,
            &prelude,
            self.config.determinizations,
            self.config.greedy_plies,
            rng,
        )
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

pub(crate) fn habitat_candidate_union(
    game: &GameState,
    immediate_candidate_limit: usize,
    habitat_candidate_limit: usize,
) -> Result<(MarketPrelude, GameState, Vec<GreedyCandidate>), SearchError> {
    let (prelude, staged) = game.preview_free_three_of_a_kind_if_feasible()?;
    let mut candidates = rank_greedy_actions(
        &staged,
        &MarketPrelude::default(),
        Some(immediate_candidate_limit),
    )?;
    merge_unique_candidates(
        &mut candidates,
        rank_habitat_setup_actions(
            &staged,
            &MarketPrelude::default(),
            Some(habitat_candidate_limit),
        )?,
    );
    Ok((prelude, staged, candidates))
}

pub const BEAR_HABITAT_CANDIDATE_LOOKAHEAD_STRATEGY_ID: &str =
    "bear-habitat-candidate-lookahead-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BearHabitatCandidateLookaheadConfig {
    pub immediate_candidate_limit: usize,
    pub habitat_candidate_limit: usize,
    pub bear_candidate_limit: usize,
    pub determinizations: usize,
    pub greedy_plies: usize,
}

impl BearHabitatCandidateLookaheadConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        if self.immediate_candidate_limit == 0
            || self.habitat_candidate_limit == 0
            || self.bear_candidate_limit == 0
        {
            return Err(SearchError::InvalidConfig(
                "Bear-habitat lookahead candidate limits must be positive",
            ));
        }
        if self.determinizations == 0 {
            return Err(SearchError::InvalidConfig(
                "Bear-habitat lookahead must use at least one determinization",
            ));
        }
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{BEAR_HABITAT_CANDIDATE_LOOKAHEAD_STRATEGY_ID}-k{}-h{}-b{}-r{}-d{}",
            self.immediate_candidate_limit,
            self.habitat_candidate_limit,
            self.bear_candidate_limit,
            self.determinizations,
            self.greedy_plies,
        )
    }
}

impl Default for BearHabitatCandidateLookaheadConfig {
    fn default() -> Self {
        Self {
            immediate_candidate_limit: 8,
            habitat_candidate_limit: 6,
            bear_candidate_limit: 8,
            determinizations: 4,
            greedy_plies: 4,
        }
    }
}

pub struct BearHabitatCandidateLookaheadStrategy {
    config: BearHabitatCandidateLookaheadConfig,
    strategy_id: String,
}

impl BearHabitatCandidateLookaheadStrategy {
    pub fn new(config: BearHabitatCandidateLookaheadConfig) -> Result<Self, SearchError> {
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
        let (prelude, staged, candidates) = bear_habitat_candidate_union(
            game,
            self.config.immediate_candidate_limit,
            self.config.habitat_candidate_limit,
            self.config.bear_candidate_limit,
        )?;
        rank_rollout_candidates(
            game,
            &staged,
            candidates,
            &prelude,
            self.config.determinizations,
            self.config.greedy_plies,
            rng,
        )
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

fn bear_habitat_candidate_union(
    game: &GameState,
    immediate_candidate_limit: usize,
    habitat_candidate_limit: usize,
    bear_candidate_limit: usize,
) -> Result<(MarketPrelude, GameState, Vec<GreedyCandidate>), SearchError> {
    let (prelude, staged, mut candidates) =
        habitat_candidate_union(game, immediate_candidate_limit, habitat_candidate_limit)?;
    merge_unique_candidates(
        &mut candidates,
        rank_bear_setup_actions(
            &staged,
            &MarketPrelude::default(),
            Some(bear_candidate_limit),
        )?,
    );
    Ok((prelude, staged, candidates))
}

fn merge_unique_candidates(
    retained: &mut Vec<GreedyCandidate>,
    additional: impl IntoIterator<Item = GreedyCandidate>,
) {
    for candidate in additional {
        if !retained
            .iter()
            .any(|existing| existing.action == candidate.action)
        {
            retained.push(candidate);
        }
    }
}

pub const NATURE_WIPE_LOOKAHEAD_STRATEGY_ID: &str = "nature-wipe-lookahead-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NatureWipeLookaheadConfig {
    pub action_search: DeterminizedLookaheadConfig,
    pub prelude_candidate_limit: usize,
    pub prelude_determinizations: usize,
    pub prelude_greedy_plies: usize,
}

impl NatureWipeLookaheadConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        self.action_search.validate()?;
        if self.prelude_candidate_limit == 0 {
            return Err(SearchError::InvalidConfig(
                "nature-wipe search must retain at least one action",
            ));
        }
        if self.prelude_determinizations == 0 {
            return Err(SearchError::InvalidConfig(
                "nature-wipe search must use at least one determinization",
            ));
        }
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{NATURE_WIPE_LOOKAHEAD_STRATEGY_ID}-{}-pk{}-pr{}-pd{}",
            self.action_search.strategy_id(),
            self.prelude_candidate_limit,
            self.prelude_determinizations,
            self.prelude_greedy_plies
        )
    }
}

impl Default for NatureWipeLookaheadConfig {
    fn default() -> Self {
        Self {
            action_search: DeterminizedLookaheadConfig::default(),
            prelude_candidate_limit: 4,
            prelude_determinizations: 2,
            prelude_greedy_plies: 4,
        }
    }
}

pub struct NatureWipeLookaheadStrategy {
    config: NatureWipeLookaheadConfig,
    action_search: DeterminizedLookaheadStrategy,
    strategy_id: String,
}

impl NatureWipeLookaheadStrategy {
    pub fn new(config: NatureWipeLookaheadConfig) -> Result<Self, SearchError> {
        let config = config.validate()?;
        Ok(Self {
            action_search: DeterminizedLookaheadStrategy::new(config.action_search)?,
            strategy_id: config.strategy_id(),
            config,
        })
    }

    pub fn strategy_id(&self) -> &str {
        &self.strategy_id
    }

    pub fn select_action_deterministic(&self, game: &GameState) -> Result<TurnAction, SearchError> {
        let (free_prelude, after_free) = game.preview_free_three_of_a_kind_if_feasible()?;
        let replace_three_of_a_kind = free_prelude.replace_three_of_a_kind;
        let paid_wipe = self.select_paid_wipe(&after_free)?;
        let paid_prelude = MarketPrelude {
            replace_three_of_a_kind: false,
            wildlife_wipes: paid_wipe.iter().cloned().collect(),
        };
        let staged = after_free.preview_market_prelude(&paid_prelude)?;
        let (_, mut action) = self.action_search.rank_and_select_with_prelude(
            &staged,
            &MarketPrelude::default(),
            &mut lookahead_decision_rng(game),
        )?;
        action.replace_three_of_a_kind = replace_three_of_a_kind;
        action.wildlife_wipes = paid_wipe.into_iter().collect();
        Ok(action)
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

    fn select_paid_wipe(&self, game: &GameState) -> Result<Option<WildlifeWipe>, SearchError> {
        let mut options = vec![None];
        options.extend(game.legal_wildlife_wipes().into_iter().map(Some));
        if options.len() == 1 {
            return Ok(None);
        }

        let mut rng = nature_wipe_decision_rng(game);
        let mut sample_seeds = Vec::with_capacity(self.config.prelude_determinizations);
        for _ in 0..self.config.prelude_determinizations {
            let mut seed = [0; 32];
            rng.fill(&mut seed);
            sample_seeds.push(GameSeed(seed));
        }
        let cards = game.config().scoring_cards;
        let acting_seat = game.current_player();
        let sample_count = sample_seeds.len();
        let jobs: Vec<Result<(usize, f64), SearchError>> = (0..options.len() * sample_count)
            .into_par_iter()
            .map(|job| {
                let option_index = job / sample_count;
                let sample_seed = sample_seeds[job % sample_count];
                let prelude = MarketPrelude {
                    replace_three_of_a_kind: false,
                    wildlife_wipes: options[option_index].iter().cloned().collect(),
                };
                let mut sample = game.clone();
                // The wipe choice is valued before its replacement draw becomes public.
                sample.redeterminize_hidden(sample_seed);
                let staged = sample.preview_market_prelude(&prelude)?;
                let candidates = rank_greedy_actions(
                    &staged,
                    &MarketPrelude::default(),
                    Some(self.config.prelude_candidate_limit),
                )?;
                let mut best = f64::NEG_INFINITY;
                for candidate in candidates {
                    let mut branch = staged.clone();
                    branch.apply(&candidate.action)?;
                    let mut rollout_rng = rollout_rng(sample_seed);
                    play_greedy_plies(
                        &mut branch,
                        self.config.prelude_greedy_plies,
                        &mut rollout_rng,
                    )?;
                    best = best.max(f64::from(
                        score_board(&branch.boards()[acting_seat], cards).base_total,
                    ));
                }
                if !best.is_finite() {
                    return Err(SearchError::NoLegalActions);
                }
                Ok((option_index, best))
            })
            .collect();

        let mut values = vec![0.0; options.len()];
        for job in jobs {
            let (option_index, value) = job?;
            values[option_index] += value;
        }
        for value in &mut values {
            *value /= sample_count as f64;
        }
        let mut best_index = 0;
        for index in 1..values.len() {
            if values[index] > values[best_index] {
                best_index = index;
            }
        }
        Ok(options[best_index].clone())
    }
}

pub(crate) fn with_prelude(mut action: TurnAction, prelude: &MarketPrelude) -> TurnAction {
    action.replace_three_of_a_kind = prelude.replace_three_of_a_kind;
    action.wildlife_wipes = prelude.wildlife_wipes.clone();
    action
}

pub(crate) fn rank_rollout_candidates(
    game: &GameState,
    staged: &GameState,
    candidates: Vec<GreedyCandidate>,
    prelude: &MarketPrelude,
    determinizations: usize,
    greedy_plies: usize,
    rng: &mut ChaCha8Rng,
) -> Result<Vec<RolloutCandidate>, SearchError> {
    if candidates.is_empty() {
        return Err(SearchError::NoLegalActions);
    }
    let acting_seat = staged.current_player();
    let mut sample_seeds = Vec::with_capacity(determinizations);
    for _ in 0..determinizations {
        let mut seed = [0; 32];
        rng.fill(&mut seed);
        sample_seeds.push(GameSeed(seed));
    }
    let cards = game.config().scoring_cards;
    let sample_count = sample_seeds.len();
    let scores: Vec<Result<(usize, f64), SearchError>> = (0..candidates.len() * sample_count)
        .into_par_iter()
        .map(|job| {
            let candidate_index = job / sample_count;
            let base_seed = sample_seeds[job % sample_count];
            let mut attempt = 0u64;
            loop {
                let sample_seed = conditioned_rollout_seed(base_seed, attempt);
                let result: Result<(usize, f64), SearchError> = (|| {
                    let mut sample = staged.clone();
                    sample.redeterminize_hidden(sample_seed);
                    sample.apply(&candidates[candidate_index].action)?;
                    let mut rollout_rng = rollout_rng(sample_seed);
                    play_greedy_plies(&mut sample, greedy_plies, &mut rollout_rng)?;
                    Ok((
                        candidate_index,
                        f64::from(score_board(&sample.boards()[acting_seat], cards).base_total),
                    ))
                })();
                match result {
                    Ok(score) => break Ok(score),
                    Err(error) if error.is_unstable_market_exhaustion() => {
                        attempt = attempt.checked_add(1).ok_or(error)?;
                    }
                    Err(error) => break Err(error),
                }
            }
        })
        .collect();
    let mut values = vec![Vec::with_capacity(sample_count); candidates.len()];
    for score in scores {
        let (candidate_index, score) = score?;
        values[candidate_index].push(score);
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

pub(crate) fn select_ranked_action(
    ranked: &[RolloutCandidate],
    rng: &mut ChaCha8Rng,
) -> Result<TurnAction, SearchError> {
    let Some(best) = ranked.first() else {
        return Err(SearchError::NoLegalActions);
    };
    let tied = ranked
        .iter()
        .take_while(|candidate| candidate.mean_leaf_score == best.mean_leaf_score)
        .count();
    Ok(ranked[rng.gen_range(0..tied)].action.clone())
}
