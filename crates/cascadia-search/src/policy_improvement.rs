use cascadia_game::{
    GameConfig, GameSeed, GameState, MarketPrelude, TurnAction, Wildlife, score_board,
};
#[cfg(test)]
use cascadia_sim::select_pattern_action;
use cascadia_sim::{
    MatchResult, PATTERN_AWARE_STRATEGY_ID, PatternAwareConfig, SimulationError,
    choose_pattern_market_prelude, play_match_with_selector, play_pattern_plies,
    rank_pattern_frontier_actions, rank_wildlife_diverse_pattern_frontier_actions,
    rank_wildlife_focused_pattern_frontier_actions, select_pattern_action_with_market_choice,
    strategy_rng,
};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::{
    PerfectInformationFocalBeamConfig, PerfectInformationFocalBeamStrategy, RolloutCandidate,
    SearchError, lookahead_decision_rng, rollout_rng, select_ranked_action, with_prelude,
};

pub const TERMINAL_POLICY_IMPROVEMENT_STRATEGY_ID: &str = "terminal-policy-improvement-v1";
pub const LATE_TERMINAL_POLICY_IMPROVEMENT_STRATEGY_ID: &str =
    "late-terminal-policy-improvement-v1";
pub const WILDLIFE_DIVERSE_TERMINAL_POLICY_IMPROVEMENT_STRATEGY_ID: &str =
    "wildlife-diverse-terminal-policy-improvement-v1";
pub const LATE_WILDLIFE_DIVERSE_POLICY_IMPROVEMENT_STRATEGY_ID: &str =
    "late-wildlife-diverse-policy-improvement-v1";
pub const LATE_CONSERVATIVE_POLICY_IMPROVEMENT_STRATEGY_ID: &str =
    "late-conservative-policy-improvement-v1";
pub const LATE_CONSERVATIVE_BASE_POLICY_IMPROVEMENT_STRATEGY_ID: &str =
    "late-conservative-base-policy-improvement-v1";
pub const LATE_CONSERVATIVE_WILDLIFE_FOCUSED_POLICY_IMPROVEMENT_STRATEGY_ID: &str =
    "late-conservative-wildlife-focused-policy-improvement-v1";
pub const LATE_CONSERVATIVE_FOCAL_BEAM_STRATEGY_ID: &str = "late-conservative-focal-beam-v1";
const ONE_SIDED_T_90_DF_3: f64 = 1.637_744_353_696_209_5;
const ONE_SIDED_T_90_DF_7: f64 = 1.414_923_927_648_858_5;
const ONE_SIDED_T_90_DF_31: f64 = 1.309_464_835_175_997;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TerminalPolicyImprovementConfig {
    pub determinizations: usize,
    pub blueprint: PatternAwareConfig,
}

impl TerminalPolicyImprovementConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        if self.determinizations == 0 {
            return Err(SearchError::InvalidConfig(
                "terminal policy improvement must use at least one determinization",
            ));
        }
        self.blueprint.validate()?;
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{TERMINAL_POLICY_IMPROVEMENT_STRATEGY_ID}-r{}-k{}-h{}-b{}-m{}",
            self.determinizations,
            self.blueprint.immediate_candidate_limit,
            self.blueprint.habitat_candidate_limit,
            self.blueprint.bear_candidate_limit,
            self.blueprint.future_market_draws,
        )
    }
}

impl Default for TerminalPolicyImprovementConfig {
    fn default() -> Self {
        Self {
            determinizations: 2,
            blueprint: PatternAwareConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LateTerminalPolicyImprovementConfig {
    pub final_personal_turns: u16,
    pub terminal: TerminalPolicyImprovementConfig,
}

impl LateTerminalPolicyImprovementConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        if !(1..=20).contains(&self.final_personal_turns) {
            return Err(SearchError::InvalidConfig(
                "late terminal policy improvement requires 1 to 20 final personal turns",
            ));
        }
        self.terminal.validate()?;
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{LATE_TERMINAL_POLICY_IMPROVEMENT_STRATEGY_ID}-t{}-r{}-k{}-h{}-b{}-m{}",
            self.final_personal_turns,
            self.terminal.determinizations,
            self.terminal.blueprint.immediate_candidate_limit,
            self.terminal.blueprint.habitat_candidate_limit,
            self.terminal.blueprint.bear_candidate_limit,
            self.terminal.blueprint.future_market_draws,
        )
    }
}

impl Default for LateTerminalPolicyImprovementConfig {
    fn default() -> Self {
        Self {
            final_personal_turns: 4,
            terminal: TerminalPolicyImprovementConfig {
                determinizations: 8,
                blueprint: PatternAwareConfig::default(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WildlifeDiverseTerminalPolicyImprovementConfig {
    pub wildlife_candidate_limit: usize,
    pub terminal: TerminalPolicyImprovementConfig,
}

impl WildlifeDiverseTerminalPolicyImprovementConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        if self.wildlife_candidate_limit == 0 {
            return Err(SearchError::InvalidConfig(
                "wildlife-diverse terminal policy improvement requires at least one candidate per wildlife",
            ));
        }
        self.terminal.validate()?;
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{WILDLIFE_DIVERSE_TERMINAL_POLICY_IMPROVEMENT_STRATEGY_ID}-r{}-k{}-h{}-b{}-w{}-m{}",
            self.terminal.determinizations,
            self.terminal.blueprint.immediate_candidate_limit,
            self.terminal.blueprint.habitat_candidate_limit,
            self.terminal.blueprint.bear_candidate_limit,
            self.wildlife_candidate_limit,
            self.terminal.blueprint.future_market_draws,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LateWildlifeDiversePolicyImprovementConfig {
    pub final_personal_turns: u16,
    pub terminal: WildlifeDiverseTerminalPolicyImprovementConfig,
}

impl LateWildlifeDiversePolicyImprovementConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        if !(1..=20).contains(&self.final_personal_turns) {
            return Err(SearchError::InvalidConfig(
                "late wildlife-diverse policy improvement requires 1 to 20 final personal turns",
            ));
        }
        self.terminal.validate()?;
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{LATE_WILDLIFE_DIVERSE_POLICY_IMPROVEMENT_STRATEGY_ID}-t{}-r{}-k{}-h{}-b{}-w{}-m{}",
            self.final_personal_turns,
            self.terminal.terminal.determinizations,
            self.terminal.terminal.blueprint.immediate_candidate_limit,
            self.terminal.terminal.blueprint.habitat_candidate_limit,
            self.terminal.terminal.blueprint.bear_candidate_limit,
            self.terminal.wildlife_candidate_limit,
            self.terminal.terminal.blueprint.future_market_draws,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LateConservativePolicyImprovementConfig {
    pub final_personal_turns: u16,
    pub terminal: WildlifeDiverseTerminalPolicyImprovementConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LateConservativeBasePolicyImprovementConfig {
    pub final_personal_turns: u16,
    pub terminal: TerminalPolicyImprovementConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WildlifeFocusedTerminalPolicyImprovementConfig {
    pub wildlife: Wildlife,
    pub wildlife_candidate_limit: usize,
    pub terminal: TerminalPolicyImprovementConfig,
}

impl WildlifeFocusedTerminalPolicyImprovementConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        if self.wildlife_candidate_limit == 0 {
            return Err(SearchError::InvalidConfig(
                "wildlife-focused terminal policy improvement requires at least one candidate",
            ));
        }
        self.terminal.validate()?;
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LateConservativeWildlifeFocusedPolicyImprovementConfig {
    pub final_personal_turns: u16,
    pub terminal: WildlifeFocusedTerminalPolicyImprovementConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LateConservativeFocalBeamConfig {
    pub final_personal_turns: u16,
    pub determinizations: usize,
    pub beam_width: usize,
    pub wildlife_candidate_limit: usize,
    pub blueprint: PatternAwareConfig,
}

impl LateConservativeFocalBeamConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        if !(1..=20).contains(&self.final_personal_turns) {
            return Err(SearchError::InvalidConfig(
                "late conservative focal beam requires 1 to 20 final personal turns",
            ));
        }
        if !matches!(self.determinizations, 4 | 8 | 32) {
            return Err(SearchError::InvalidConfig(
                "late conservative focal beam supports exactly 4, 8, or 32 determinizations",
            ));
        }
        if self.beam_width == 0 || self.wildlife_candidate_limit == 0 {
            return Err(SearchError::InvalidConfig(
                "late conservative focal beam widths must be positive",
            ));
        }
        self.blueprint.validate()?;
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{LATE_CONSERVATIVE_FOCAL_BEAM_STRATEGY_ID}-t{}-r{}-b{}-k{}-h{}-b{}-w{}-m{}-c90",
            self.final_personal_turns,
            self.determinizations,
            self.beam_width,
            self.blueprint.immediate_candidate_limit,
            self.blueprint.habitat_candidate_limit,
            self.blueprint.bear_candidate_limit,
            self.wildlife_candidate_limit,
            self.blueprint.future_market_draws,
        )
    }
}

impl LateConservativeWildlifeFocusedPolicyImprovementConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        if !(1..=20).contains(&self.final_personal_turns) {
            return Err(SearchError::InvalidConfig(
                "late conservative wildlife-focused policy improvement requires 1 to 20 final personal turns",
            ));
        }
        self.terminal.validate()?;
        if !matches!(self.terminal.terminal.determinizations, 8 | 32) {
            return Err(SearchError::InvalidConfig(
                "late conservative wildlife-focused policy improvement supports exactly 8 or 32 determinizations",
            ));
        }
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{LATE_CONSERVATIVE_WILDLIFE_FOCUSED_POLICY_IMPROVEMENT_STRATEGY_ID}-t{}-r{}-k{}-h{}-b{}-{}{}-m{}-c90",
            self.final_personal_turns,
            self.terminal.terminal.determinizations,
            self.terminal.terminal.blueprint.immediate_candidate_limit,
            self.terminal.terminal.blueprint.habitat_candidate_limit,
            self.terminal.terminal.blueprint.bear_candidate_limit,
            wildlife_slug(self.terminal.wildlife),
            self.terminal.wildlife_candidate_limit,
            self.terminal.terminal.blueprint.future_market_draws,
        )
    }
}

fn wildlife_slug(wildlife: Wildlife) -> &'static str {
    match wildlife {
        Wildlife::Bear => "bear",
        Wildlife::Elk => "elk",
        Wildlife::Salmon => "salmon",
        Wildlife::Hawk => "hawk",
        Wildlife::Fox => "fox",
    }
}

impl LateConservativeBasePolicyImprovementConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        if !(1..=20).contains(&self.final_personal_turns) {
            return Err(SearchError::InvalidConfig(
                "late conservative base policy improvement requires 1 to 20 final personal turns",
            ));
        }
        self.terminal.validate()?;
        if !matches!(self.terminal.determinizations, 8 | 32) {
            return Err(SearchError::InvalidConfig(
                "late conservative base policy improvement supports exactly 8 or 32 determinizations",
            ));
        }
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{LATE_CONSERVATIVE_BASE_POLICY_IMPROVEMENT_STRATEGY_ID}-t{}-r{}-k{}-h{}-b{}-m{}-c90",
            self.final_personal_turns,
            self.terminal.determinizations,
            self.terminal.blueprint.immediate_candidate_limit,
            self.terminal.blueprint.habitat_candidate_limit,
            self.terminal.blueprint.bear_candidate_limit,
            self.terminal.blueprint.future_market_draws,
        )
    }
}

impl Default for LateConservativeBasePolicyImprovementConfig {
    fn default() -> Self {
        Self {
            final_personal_turns: 5,
            terminal: TerminalPolicyImprovementConfig {
                determinizations: 8,
                blueprint: PatternAwareConfig::default(),
            },
        }
    }
}

impl LateConservativePolicyImprovementConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        if !(1..=20).contains(&self.final_personal_turns) {
            return Err(SearchError::InvalidConfig(
                "late conservative policy improvement requires 1 to 20 final personal turns",
            ));
        }
        self.terminal.validate()?;
        if self.terminal.terminal.determinizations != 8 {
            return Err(SearchError::InvalidConfig(
                "late conservative policy improvement v1 requires exactly eight determinizations",
            ));
        }
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{LATE_CONSERVATIVE_POLICY_IMPROVEMENT_STRATEGY_ID}-t{}-r8-k{}-h{}-b{}-w{}-m{}-c90",
            self.final_personal_turns,
            self.terminal.terminal.blueprint.immediate_candidate_limit,
            self.terminal.terminal.blueprint.habitat_candidate_limit,
            self.terminal.terminal.blueprint.bear_candidate_limit,
            self.terminal.wildlife_candidate_limit,
            self.terminal.terminal.blueprint.future_market_draws,
        )
    }
}

pub struct TerminalPolicyImprovementStrategy {
    config: TerminalPolicyImprovementConfig,
    strategy_id: String,
}

pub struct LateTerminalPolicyImprovementStrategy {
    config: LateTerminalPolicyImprovementConfig,
    terminal: TerminalPolicyImprovementStrategy,
    strategy_id: String,
}

pub struct WildlifeDiverseTerminalPolicyImprovementStrategy {
    config: WildlifeDiverseTerminalPolicyImprovementConfig,
    strategy_id: String,
}

pub struct LateWildlifeDiversePolicyImprovementStrategy {
    config: LateWildlifeDiversePolicyImprovementConfig,
    terminal: WildlifeDiverseTerminalPolicyImprovementStrategy,
    strategy_id: String,
}

pub struct LateConservativePolicyImprovementStrategy {
    config: LateConservativePolicyImprovementConfig,
    terminal: WildlifeDiverseTerminalPolicyImprovementStrategy,
    strategy_id: String,
}

pub struct LateConservativeBasePolicyImprovementStrategy {
    config: LateConservativeBasePolicyImprovementConfig,
    terminal: TerminalPolicyImprovementStrategy,
    strategy_id: String,
}

struct WildlifeFocusedTerminalPolicyImprovementStrategy {
    config: WildlifeFocusedTerminalPolicyImprovementConfig,
}

pub struct LateConservativeWildlifeFocusedPolicyImprovementStrategy {
    config: LateConservativeWildlifeFocusedPolicyImprovementConfig,
    terminal: WildlifeFocusedTerminalPolicyImprovementStrategy,
    strategy_id: String,
}

pub struct LateConservativeFocalBeamStrategy {
    config: LateConservativeFocalBeamConfig,
    beam: PerfectInformationFocalBeamStrategy,
    strategy_id: String,
}

struct TerminalCandidateValues {
    candidates: Vec<cascadia_sim::GreedyCandidate>,
    values: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConservativeCandidateEvaluation {
    pub action: TurnAction,
    pub immediate_rank: usize,
    pub immediate_score: u16,
    pub terminal_mean: f64,
    pub terminal_stddev: f64,
    pub mean_advantage: f64,
    pub advantage_standard_error: f64,
    pub lower_bound: f64,
    pub is_anchor: bool,
}

impl LateTerminalPolicyImprovementStrategy {
    pub fn new(config: LateTerminalPolicyImprovementConfig) -> Result<Self, SearchError> {
        let config = config.validate()?;
        Ok(Self {
            terminal: TerminalPolicyImprovementStrategy::new(config.terminal)?,
            strategy_id: config.strategy_id(),
            config,
        })
    }

    pub fn strategy_id(&self) -> &str {
        &self.strategy_id
    }

    pub fn uses_terminal_search(&self, game: &GameState) -> bool {
        game.turns_remaining_for_player(game.current_player()) <= self.config.final_personal_turns
    }

    pub fn select_action(
        &self,
        game: &GameState,
        blueprint_rng: &mut ChaCha8Rng,
    ) -> Result<TurnAction, SearchError> {
        if self.uses_terminal_search(game) {
            self.terminal.select_action_deterministic(game)
        } else {
            Ok(select_pattern_action_with_market_choice(
                game,
                self.config.terminal.blueprint,
                blueprint_rng,
            )?)
        }
    }

    pub fn play_match(
        &self,
        game_config: GameConfig,
        seed: GameSeed,
    ) -> Result<MatchResult, SearchError> {
        let mut blueprint_rngs = (0..usize::from(game_config.player_count))
            .map(|seat| strategy_rng(seed, seat, PATTERN_AWARE_STRATEGY_ID))
            .collect::<Vec<_>>();
        Ok(play_match_with_selector(
            game_config,
            seed,
            &self.strategy_id,
            |player, game| {
                self.select_action(game, &mut blueprint_rngs[player])
                    .map_err(|error| SimulationError::Strategy(error.to_string()))
            },
        )?)
    }
}

impl LateWildlifeDiversePolicyImprovementStrategy {
    pub fn new(config: LateWildlifeDiversePolicyImprovementConfig) -> Result<Self, SearchError> {
        let config = config.validate()?;
        Ok(Self {
            terminal: WildlifeDiverseTerminalPolicyImprovementStrategy::new(config.terminal)?,
            strategy_id: config.strategy_id(),
            config,
        })
    }

    pub fn strategy_id(&self) -> &str {
        &self.strategy_id
    }

    pub fn uses_terminal_search(&self, game: &GameState) -> bool {
        game.turns_remaining_for_player(game.current_player()) <= self.config.final_personal_turns
    }

    pub fn select_action(
        &self,
        game: &GameState,
        blueprint_rng: &mut ChaCha8Rng,
    ) -> Result<TurnAction, SearchError> {
        if self.uses_terminal_search(game) {
            self.terminal.select_action_deterministic(game)
        } else {
            Ok(select_pattern_action_with_market_choice(
                game,
                self.config.terminal.terminal.blueprint,
                blueprint_rng,
            )?)
        }
    }

    pub fn play_match(
        &self,
        game_config: GameConfig,
        seed: GameSeed,
    ) -> Result<MatchResult, SearchError> {
        let mut blueprint_rngs = (0..usize::from(game_config.player_count))
            .map(|seat| strategy_rng(seed, seat, PATTERN_AWARE_STRATEGY_ID))
            .collect::<Vec<_>>();
        Ok(play_match_with_selector(
            game_config,
            seed,
            &self.strategy_id,
            |player, game| {
                self.select_action(game, &mut blueprint_rngs[player])
                    .map_err(|error| SimulationError::Strategy(error.to_string()))
            },
        )?)
    }
}

impl LateConservativePolicyImprovementStrategy {
    pub fn new(config: LateConservativePolicyImprovementConfig) -> Result<Self, SearchError> {
        let config = config.validate()?;
        Ok(Self {
            terminal: WildlifeDiverseTerminalPolicyImprovementStrategy::new(config.terminal)?,
            strategy_id: config.strategy_id(),
            config,
        })
    }

    pub fn strategy_id(&self) -> &str {
        &self.strategy_id
    }

    pub fn uses_terminal_search(&self, game: &GameState) -> bool {
        game.turns_remaining_for_player(game.current_player()) <= self.config.final_personal_turns
    }

    pub fn select_action(
        &self,
        game: &GameState,
        blueprint_rng: &mut ChaCha8Rng,
    ) -> Result<TurnAction, SearchError> {
        if !self.uses_terminal_search(game) {
            return Ok(select_pattern_action_with_market_choice(
                game,
                self.config.terminal.terminal.blueprint,
                blueprint_rng,
            )?);
        }

        let anchor = select_pattern_action_with_market_choice(
            game,
            self.config.terminal.terminal.blueprint,
            blueprint_rng,
        )?;
        self.terminal
            .select_conservative_action_deterministic(game, &anchor)
    }

    pub fn play_match(
        &self,
        game_config: GameConfig,
        seed: GameSeed,
    ) -> Result<MatchResult, SearchError> {
        let mut blueprint_rngs = (0..usize::from(game_config.player_count))
            .map(|seat| strategy_rng(seed, seat, PATTERN_AWARE_STRATEGY_ID))
            .collect::<Vec<_>>();
        Ok(play_match_with_selector(
            game_config,
            seed,
            &self.strategy_id,
            |player, game| {
                self.select_action(game, &mut blueprint_rngs[player])
                    .map_err(|error| SimulationError::Strategy(error.to_string()))
            },
        )?)
    }
}

impl LateConservativeWildlifeFocusedPolicyImprovementStrategy {
    pub fn new(
        config: LateConservativeWildlifeFocusedPolicyImprovementConfig,
    ) -> Result<Self, SearchError> {
        let config = config.validate()?;
        Ok(Self {
            terminal: WildlifeFocusedTerminalPolicyImprovementStrategy::new(config.terminal)?,
            strategy_id: config.strategy_id(),
            config,
        })
    }

    pub fn strategy_id(&self) -> &str {
        &self.strategy_id
    }

    pub fn uses_terminal_search(&self, game: &GameState) -> bool {
        game.turns_remaining_for_player(game.current_player()) <= self.config.final_personal_turns
    }

    pub fn select_action(
        &self,
        game: &GameState,
        blueprint_rng: &mut ChaCha8Rng,
    ) -> Result<TurnAction, SearchError> {
        let anchor = select_pattern_action_with_market_choice(
            game,
            self.config.terminal.terminal.blueprint,
            blueprint_rng,
        )?;
        if !self.uses_terminal_search(game) {
            return Ok(anchor);
        }
        self.terminal
            .select_conservative_action_deterministic(game, &anchor)
    }

    pub fn play_match(
        &self,
        game_config: GameConfig,
        seed: GameSeed,
    ) -> Result<MatchResult, SearchError> {
        let mut blueprint_rngs = (0..usize::from(game_config.player_count))
            .map(|seat| strategy_rng(seed, seat, PATTERN_AWARE_STRATEGY_ID))
            .collect::<Vec<_>>();
        Ok(play_match_with_selector(
            game_config,
            seed,
            &self.strategy_id,
            |player, game| {
                self.select_action(game, &mut blueprint_rngs[player])
                    .map_err(|error| SimulationError::Strategy(error.to_string()))
            },
        )?)
    }
}

impl LateConservativeFocalBeamStrategy {
    pub fn new(config: LateConservativeFocalBeamConfig) -> Result<Self, SearchError> {
        let config = config.validate()?;
        Ok(Self {
            beam: PerfectInformationFocalBeamStrategy::new(PerfectInformationFocalBeamConfig {
                blueprint: config.blueprint,
                wildlife_candidate_limit: config.wildlife_candidate_limit,
                beam_width: config.beam_width,
                final_personal_turns: config.final_personal_turns,
            })?,
            strategy_id: config.strategy_id(),
            config,
        })
    }

    pub fn strategy_id(&self) -> &str {
        &self.strategy_id
    }

    pub fn uses_beam(&self, game: &GameState) -> bool {
        game.turns_remaining_for_player(game.current_player()) <= self.config.final_personal_turns
    }

    fn candidate_values(
        &self,
        game: &GameState,
        rng: &mut ChaCha8Rng,
    ) -> Result<TerminalCandidateValues, SearchError> {
        let acting_seat = game.current_player();
        let sample_seeds = terminal_sample_seeds(rng, self.config.determinizations);
        let sample_count = sample_seeds.len();
        let prelude = choose_pattern_market_prelude(game, self.config.blueprint)?;
        let staged = game.preview_market_prelude(&prelude)?;
        let candidates = rank_wildlife_diverse_pattern_frontier_actions(
            &staged,
            &MarketPrelude::default(),
            self.config.blueprint,
            self.config.wildlife_candidate_limit,
        )?;
        let scores: Vec<Result<(usize, f64), SearchError>> = (0..candidates.len() * sample_count)
            .into_par_iter()
            .map(|job| {
                let candidate_index = job / sample_count;
                let sample_seed = sample_seeds[job % sample_count];
                let mut sample = staged.clone();
                sample.redeterminize_hidden(sample_seed);
                let score = self.beam.evaluate_root_candidate(
                    &sample,
                    &candidates[candidate_index].action,
                    acting_seat,
                    sample_seed,
                )?;
                Ok((candidate_index, score))
            })
            .collect();
        let mut values = vec![Vec::with_capacity(sample_count); candidates.len()];
        for score in scores {
            let (candidate_index, score) = score?;
            values[candidate_index].push(score);
        }
        if candidates.is_empty() {
            return Err(SearchError::NoLegalActions);
        }
        Ok(TerminalCandidateValues {
            candidates: candidates
                .into_iter()
                .map(|mut candidate| {
                    candidate.action = with_prelude(candidate.action, &prelude);
                    candidate
                })
                .collect(),
            values,
        })
    }

    pub fn select_action(
        &self,
        game: &GameState,
        blueprint_rng: &mut ChaCha8Rng,
    ) -> Result<TurnAction, SearchError> {
        let anchor =
            select_pattern_action_with_market_choice(game, self.config.blueprint, blueprint_rng)?;
        if !self.uses_beam(game) {
            return Ok(anchor);
        }
        let mut rng = lookahead_decision_rng(game);
        let sampled = self.candidate_values(game, &mut rng)?;
        select_conservative_candidate(&sampled, &anchor, &mut rng)
    }

    pub fn play_match(
        &self,
        game_config: GameConfig,
        seed: GameSeed,
    ) -> Result<MatchResult, SearchError> {
        let mut blueprint_rngs = (0..usize::from(game_config.player_count))
            .map(|seat| strategy_rng(seed, seat, PATTERN_AWARE_STRATEGY_ID))
            .collect::<Vec<_>>();
        Ok(play_match_with_selector(
            game_config,
            seed,
            &self.strategy_id,
            |player, game| {
                self.select_action(game, &mut blueprint_rngs[player])
                    .map_err(|error| SimulationError::Strategy(error.to_string()))
            },
        )?)
    }
}

impl LateConservativeBasePolicyImprovementStrategy {
    pub fn new(config: LateConservativeBasePolicyImprovementConfig) -> Result<Self, SearchError> {
        let config = config.validate()?;
        Ok(Self {
            terminal: TerminalPolicyImprovementStrategy::new(config.terminal)?,
            strategy_id: config.strategy_id(),
            config,
        })
    }

    pub fn strategy_id(&self) -> &str {
        &self.strategy_id
    }

    pub fn uses_terminal_search(&self, game: &GameState) -> bool {
        game.turns_remaining_for_player(game.current_player()) <= self.config.final_personal_turns
    }

    pub fn rank_and_select_terminal_deterministic(
        &self,
        game: &GameState,
        anchor: &TurnAction,
    ) -> Result<(Vec<RolloutCandidate>, TurnAction), SearchError> {
        self.terminal
            .rank_and_select_conservative_deterministic(game, anchor)
    }

    pub fn evaluate_and_select_terminal_deterministic(
        &self,
        game: &GameState,
        anchor: &TurnAction,
    ) -> Result<(Vec<ConservativeCandidateEvaluation>, TurnAction), SearchError> {
        self.terminal
            .evaluate_and_select_conservative_deterministic(game, anchor)
    }

    pub fn select_action(
        &self,
        game: &GameState,
        blueprint_rng: &mut ChaCha8Rng,
    ) -> Result<TurnAction, SearchError> {
        if !self.uses_terminal_search(game) {
            return Ok(select_pattern_action_with_market_choice(
                game,
                self.config.terminal.blueprint,
                blueprint_rng,
            )?);
        }

        let anchor = select_pattern_action_with_market_choice(
            game,
            self.config.terminal.blueprint,
            blueprint_rng,
        )?;
        self.terminal
            .select_conservative_action_deterministic(game, &anchor)
    }

    pub fn play_match(
        &self,
        game_config: GameConfig,
        seed: GameSeed,
    ) -> Result<MatchResult, SearchError> {
        let mut blueprint_rngs = (0..usize::from(game_config.player_count))
            .map(|seat| strategy_rng(seed, seat, PATTERN_AWARE_STRATEGY_ID))
            .collect::<Vec<_>>();
        Ok(play_match_with_selector(
            game_config,
            seed,
            &self.strategy_id,
            |player, game| {
                self.select_action(game, &mut blueprint_rngs[player])
                    .map_err(|error| SimulationError::Strategy(error.to_string()))
            },
        )?)
    }
}

impl TerminalPolicyImprovementStrategy {
    pub fn new(config: TerminalPolicyImprovementConfig) -> Result<Self, SearchError> {
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
        let sampled =
            terminal_candidate_values_across_market_choices(game, self.config, rng, |staged| {
                Ok(rank_pattern_frontier_actions(
                    staged,
                    &MarketPrelude::default(),
                    self.config.blueprint,
                )?)
            })?;
        finish_terminal_ranking(sampled.candidates, sampled.values)
    }

    fn candidate_values(
        &self,
        game: &GameState,
        rng: &mut ChaCha8Rng,
    ) -> Result<TerminalCandidateValues, SearchError> {
        terminal_candidate_values_across_market_choices(game, self.config, rng, |staged| {
            Ok(rank_pattern_frontier_actions(
                staged,
                &MarketPrelude::default(),
                self.config.blueprint,
            )?)
        })
    }

    pub fn select_conservative_action_deterministic(
        &self,
        game: &GameState,
        anchor: &TurnAction,
    ) -> Result<TurnAction, SearchError> {
        Ok(self
            .rank_and_select_conservative_deterministic(game, anchor)?
            .1)
    }

    pub fn rank_and_select_conservative_deterministic(
        &self,
        game: &GameState,
        anchor: &TurnAction,
    ) -> Result<(Vec<RolloutCandidate>, TurnAction), SearchError> {
        let mut rng = lookahead_decision_rng(game);
        let sampled = self.candidate_values(game, &mut rng)?;
        let action = select_conservative_candidate(&sampled, anchor, &mut rng)?;
        let TerminalCandidateValues { candidates, values } = sampled;
        let ranked = finish_terminal_ranking(candidates, values)?;
        Ok((ranked, action))
    }

    pub fn evaluate_and_select_conservative_deterministic(
        &self,
        game: &GameState,
        anchor: &TurnAction,
    ) -> Result<(Vec<ConservativeCandidateEvaluation>, TurnAction), SearchError> {
        let mut rng = lookahead_decision_rng(game);
        let sampled = self.candidate_values(game, &mut rng)?;
        let action = select_conservative_candidate(&sampled, anchor, &mut rng)?;
        let evaluations = conservative_candidate_evaluations(&sampled, anchor)?;
        Ok((evaluations, action))
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

    pub fn select_from_recorded_ranking_deterministic(
        &self,
        game: &GameState,
        ranked: &[RolloutCandidate],
    ) -> Result<TurnAction, SearchError> {
        let mut rng = lookahead_decision_rng(game);
        terminal_sample_seeds(&mut rng, self.config.determinizations);
        select_ranked_action(ranked, &mut rng)
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

impl WildlifeDiverseTerminalPolicyImprovementStrategy {
    pub fn new(
        config: WildlifeDiverseTerminalPolicyImprovementConfig,
    ) -> Result<Self, SearchError> {
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
        let sampled = terminal_candidate_values_across_market_choices(
            game,
            self.config.terminal,
            rng,
            |staged| {
                Ok(rank_wildlife_diverse_pattern_frontier_actions(
                    staged,
                    &MarketPrelude::default(),
                    self.config.terminal.blueprint,
                    self.config.wildlife_candidate_limit,
                )?)
            },
        )?;
        finish_terminal_ranking(sampled.candidates, sampled.values)
    }

    fn candidate_values(
        &self,
        game: &GameState,
        rng: &mut ChaCha8Rng,
    ) -> Result<TerminalCandidateValues, SearchError> {
        terminal_candidate_values_across_market_choices(game, self.config.terminal, rng, |staged| {
            Ok(rank_wildlife_diverse_pattern_frontier_actions(
                staged,
                &MarketPrelude::default(),
                self.config.terminal.blueprint,
                self.config.wildlife_candidate_limit,
            )?)
        })
    }

    pub fn select_conservative_action_deterministic(
        &self,
        game: &GameState,
        anchor: &TurnAction,
    ) -> Result<TurnAction, SearchError> {
        let mut rng = lookahead_decision_rng(game);
        let sampled = self.candidate_values(game, &mut rng)?;
        select_conservative_candidate(&sampled, anchor, &mut rng)
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
}

impl WildlifeFocusedTerminalPolicyImprovementStrategy {
    fn new(config: WildlifeFocusedTerminalPolicyImprovementConfig) -> Result<Self, SearchError> {
        Ok(Self {
            config: config.validate()?,
        })
    }

    fn candidate_values(
        &self,
        game: &GameState,
        rng: &mut ChaCha8Rng,
    ) -> Result<TerminalCandidateValues, SearchError> {
        terminal_candidate_values_across_market_choices(game, self.config.terminal, rng, |staged| {
            Ok(rank_wildlife_focused_pattern_frontier_actions(
                staged,
                &MarketPrelude::default(),
                self.config.terminal.blueprint,
                self.config.wildlife,
                self.config.wildlife_candidate_limit,
            )?)
        })
    }

    fn select_conservative_action_deterministic(
        &self,
        game: &GameState,
        anchor: &TurnAction,
    ) -> Result<TurnAction, SearchError> {
        let mut rng = lookahead_decision_rng(game);
        let sampled = self.candidate_values(game, &mut rng)?;
        select_conservative_candidate(&sampled, anchor, &mut rng)
    }
}

fn terminal_candidate_values_across_market_choices(
    game: &GameState,
    config: TerminalPolicyImprovementConfig,
    rng: &mut ChaCha8Rng,
    mut candidates_for_staged: impl FnMut(
        &GameState,
    )
        -> Result<Vec<cascadia_sim::GreedyCandidate>, SearchError>,
) -> Result<TerminalCandidateValues, SearchError> {
    let sample_seeds = terminal_sample_seeds(rng, config.determinizations);
    let prelude = choose_pattern_market_prelude(game, config.blueprint)?;
    let staged = game.preview_market_prelude(&prelude)?;
    let candidates = candidates_for_staged(&staged)?;
    let values = evaluate_terminal_candidates(game, &staged, &candidates, config, &sample_seeds)?;
    if candidates.is_empty() {
        return Err(SearchError::NoLegalActions);
    }
    Ok(TerminalCandidateValues {
        candidates: candidates
            .into_iter()
            .map(|mut candidate| {
                candidate.action = with_prelude(candidate.action, &prelude);
                candidate
            })
            .collect(),
        values,
    })
}

fn evaluate_terminal_candidates(
    game: &GameState,
    staged: &GameState,
    candidates: &[cascadia_sim::GreedyCandidate],
    config: TerminalPolicyImprovementConfig,
    sample_seeds: &[GameSeed],
) -> Result<Vec<Vec<f64>>, SearchError> {
    if candidates.is_empty() {
        return Err(SearchError::NoLegalActions);
    }

    let acting_seat = staged.current_player();
    let cards = game.config().scoring_cards;
    let sample_count = sample_seeds.len();
    let scores: Vec<Result<(usize, f64), SearchError>> = (0..candidates.len() * sample_count)
        .into_par_iter()
        .map(|job| {
            let candidate_index = job / sample_count;
            let sample_seed = sample_seeds[job % sample_count];
            let mut sample = staged.clone();
            sample.redeterminize_hidden(sample_seed);
            sample.apply(&candidates[candidate_index].action)?;
            let remaining_plies = usize::from(sample.turns_remaining());
            let mut policy_rng = rollout_rng(sample_seed);
            play_pattern_plies(
                &mut sample,
                remaining_plies,
                config.blueprint,
                &mut policy_rng,
            )?;
            debug_assert!(sample.is_game_over());
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
    Ok(values)
}

fn finish_terminal_ranking(
    candidates: Vec<cascadia_sim::GreedyCandidate>,
    values: Vec<Vec<f64>>,
) -> Result<Vec<RolloutCandidate>, SearchError> {
    if candidates.len() != values.len() || values.iter().any(Vec::is_empty) {
        return Err(SearchError::InvalidConfig(
            "terminal evaluation produced incomplete candidate values",
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
                action: candidate.action,
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

fn select_conservative_candidate(
    sampled: &TerminalCandidateValues,
    anchor: &TurnAction,
    rng: &mut ChaCha8Rng,
) -> Result<TurnAction, SearchError> {
    let anchor_index = sampled
        .candidates
        .iter()
        .position(|candidate| candidate.action == *anchor)
        .ok_or(SearchError::InvalidConfig(
            "pattern-aware anchor is absent from terminal frontier",
        ))?;
    let anchor_values = &sampled.values[anchor_index];
    let mut eligible = Vec::new();
    for (index, (candidate, candidate_values)) in
        sampled.candidates.iter().zip(&sampled.values).enumerate()
    {
        let (mean_advantage, _, lower_bound) =
            paired_advantage_lcb90(candidate_values, anchor_values)?;
        if lower_bound > 0.0 {
            eligible.push((index, candidate, mean_advantage, lower_bound));
        }
    }
    if eligible.is_empty() {
        return Ok(anchor.clone());
    }
    eligible.sort_by(|left, right| {
        right
            .3
            .total_cmp(&left.3)
            .then_with(|| right.2.total_cmp(&left.2))
            .then_with(|| {
                right
                    .1
                    .resulting_base_score
                    .cmp(&left.1.resulting_base_score)
            })
    });
    let tied = eligible
        .iter()
        .take_while(|candidate| candidate.3 == eligible[0].3)
        .count();
    let selected = eligible[rng.gen_range(0..tied)].0;
    Ok(sampled.candidates[selected].action.clone())
}

fn conservative_candidate_evaluations(
    sampled: &TerminalCandidateValues,
    anchor: &TurnAction,
) -> Result<Vec<ConservativeCandidateEvaluation>, SearchError> {
    let anchor_index = sampled
        .candidates
        .iter()
        .position(|candidate| candidate.action == *anchor)
        .ok_or(SearchError::InvalidConfig(
            "pattern-aware anchor is absent from terminal frontier",
        ))?;
    let anchor_values = &sampled.values[anchor_index];
    sampled
        .candidates
        .iter()
        .zip(&sampled.values)
        .enumerate()
        .map(|(index, (candidate, values))| {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / (values.len() - 1) as f64;
            let (mean_advantage, advantage_standard_error, lower_bound) =
                paired_advantage_lcb90(values, anchor_values)?;
            Ok(ConservativeCandidateEvaluation {
                action: candidate.action.clone(),
                immediate_rank: candidate.immediate_rank,
                immediate_score: candidate.resulting_base_score,
                terminal_mean: mean,
                terminal_stddev: variance.sqrt(),
                mean_advantage,
                advantage_standard_error,
                lower_bound,
                is_anchor: index == anchor_index,
            })
        })
        .collect()
}

fn paired_advantage_lcb90(
    candidate: &[f64],
    anchor: &[f64],
) -> Result<(f64, f64, f64), SearchError> {
    if candidate.len() != anchor.len() {
        return Err(SearchError::InvalidConfig(
            "paired confidence samples must have equal lengths",
        ));
    }
    let critical = one_sided_t_90_critical(candidate.len())?;
    let differences = candidate
        .iter()
        .zip(anchor)
        .map(|(candidate, anchor)| candidate - anchor)
        .collect::<Vec<_>>();
    let mean = differences.iter().sum::<f64>() / differences.len() as f64;
    let variance = differences
        .iter()
        .map(|difference| (difference - mean).powi(2))
        .sum::<f64>()
        / (differences.len() - 1) as f64;
    let standard_error = variance.sqrt() / (differences.len() as f64).sqrt();
    Ok((mean, standard_error, mean - critical * standard_error))
}

fn one_sided_t_90_critical(sample_count: usize) -> Result<f64, SearchError> {
    match sample_count {
        4 => Ok(ONE_SIDED_T_90_DF_3),
        8 => Ok(ONE_SIDED_T_90_DF_7),
        32 => Ok(ONE_SIDED_T_90_DF_31),
        _ => Err(SearchError::InvalidConfig(
            "paired c90 confidence supports exactly 4, 8, or 32 samples",
        )),
    }
}

fn terminal_sample_seeds(rng: &mut ChaCha8Rng, count: usize) -> Vec<GameSeed> {
    (0..count)
        .map(|_| {
            let mut seed = [0; 32];
            rng.fill(&mut seed);
            GameSeed(seed)
        })
        .collect()
}

#[cfg(test)]
#[path = "policy_improvement_tests.rs"]
mod tests;
