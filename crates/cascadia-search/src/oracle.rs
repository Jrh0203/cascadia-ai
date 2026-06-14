use blake3::Hasher;
use cascadia_data::PositionRecord;
use cascadia_game::{Board, GameSeed, GameState, Market, MarketPrelude, TurnAction, score_board};
use cascadia_sim::{
    PatternAwareConfig, best_pattern_heuristic_value, play_pattern_plies,
    rank_pattern_frontier_actions, rank_wildlife_diverse_pattern_frontier_actions,
    select_pattern_action,
};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::{
    RolloutCandidate, SearchError, lookahead_decision_rng, rollout_rng, select_ranked_action,
    with_prelude,
};

pub const PERFECT_INFORMATION_PATTERN_ORACLE_STRATEGY_ID: &str =
    "perfect-information-pattern-oracle-v1";
pub const PERFECT_INFORMATION_FOCAL_BEAM_STRATEGY_ID: &str = "perfect-information-focal-beam-v1";
pub const PERFECT_INFORMATION_PORTFOLIO_BEAM_STRATEGY_ID: &str =
    "perfect-information-portfolio-beam-v1";
pub const PERFECT_INFORMATION_ROOT_DIVERSE_BEAM_STRATEGY_ID: &str =
    "perfect-information-root-diverse-beam-v1";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PerfectInformationPatternOracleConfig {
    pub blueprint: PatternAwareConfig,
    pub wildlife_candidate_limit: Option<usize>,
}

impl PerfectInformationPatternOracleConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        self.blueprint.validate()?;
        if self.wildlife_candidate_limit == Some(0) {
            return Err(SearchError::InvalidConfig(
                "perfect-information wildlife candidate limit must be positive",
            ));
        }
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        self.wildlife_candidate_limit.map_or_else(
            || {
                format!(
                    "{PERFECT_INFORMATION_PATTERN_ORACLE_STRATEGY_ID}-k{}-h{}-b{}-m{}",
                    self.blueprint.immediate_candidate_limit,
                    self.blueprint.habitat_candidate_limit,
                    self.blueprint.bear_candidate_limit,
                    self.blueprint.future_market_draws,
                )
            },
            |limit| {
                format!(
                    "{PERFECT_INFORMATION_PATTERN_ORACLE_STRATEGY_ID}-k{}-h{}-b{}-w{}-m{}",
                    self.blueprint.immediate_candidate_limit,
                    self.blueprint.habitat_candidate_limit,
                    self.blueprint.bear_candidate_limit,
                    limit,
                    self.blueprint.future_market_draws,
                )
            },
        )
    }
}

pub struct PerfectInformationPatternOracleStrategy {
    config: PerfectInformationPatternOracleConfig,
    strategy_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PerfectInformationFocalBeamConfig {
    pub blueprint: PatternAwareConfig,
    pub wildlife_candidate_limit: usize,
    pub beam_width: usize,
    pub final_personal_turns: u16,
}

impl PerfectInformationFocalBeamConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        self.blueprint.validate()?;
        if self.wildlife_candidate_limit == 0 {
            return Err(SearchError::InvalidConfig(
                "perfect-information focal beam wildlife width must be positive",
            ));
        }
        if self.beam_width == 0 {
            return Err(SearchError::InvalidConfig(
                "perfect-information focal beam width must be positive",
            ));
        }
        if !(1..=20).contains(&self.final_personal_turns) {
            return Err(SearchError::InvalidConfig(
                "perfect-information focal beam requires 1 to 20 final personal turns",
            ));
        }
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{PERFECT_INFORMATION_FOCAL_BEAM_STRATEGY_ID}-t{}-b{}-k{}-h{}-b{}-w{}-m{}",
            self.final_personal_turns,
            self.beam_width,
            self.blueprint.immediate_candidate_limit,
            self.blueprint.habitat_candidate_limit,
            self.blueprint.bear_candidate_limit,
            self.wildlife_candidate_limit,
            self.blueprint.future_market_draws,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PerfectInformationPortfolioBeamConfig {
    pub blueprint: PatternAwareConfig,
    pub wildlife_candidate_limit: usize,
    pub beam_width: usize,
    pub final_personal_turns: u16,
}

impl PerfectInformationPortfolioBeamConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        PerfectInformationFocalBeamConfig {
            blueprint: self.blueprint,
            wildlife_candidate_limit: self.wildlife_candidate_limit,
            beam_width: self.beam_width,
            final_personal_turns: self.final_personal_turns,
        }
        .validate()?;
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{PERFECT_INFORMATION_PORTFOLIO_BEAM_STRATEGY_ID}-t{}-b{}-k{}-h{}-b{}-w{}-m{}",
            self.final_personal_turns,
            self.beam_width,
            self.blueprint.immediate_candidate_limit,
            self.blueprint.habitat_candidate_limit,
            self.blueprint.bear_candidate_limit,
            self.wildlife_candidate_limit,
            self.blueprint.future_market_draws,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PerfectInformationRootDiverseBeamConfig {
    pub blueprint: PatternAwareConfig,
    pub root_wildlife_candidate_limit: usize,
    pub future_wildlife_candidate_limit: usize,
    pub beam_width: usize,
    pub final_personal_turns: u16,
}

impl PerfectInformationRootDiverseBeamConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        if self.root_wildlife_candidate_limit == 0 {
            return Err(SearchError::InvalidConfig(
                "perfect-information root-diverse beam root wildlife width must be positive",
            ));
        }
        PerfectInformationFocalBeamConfig {
            blueprint: self.blueprint,
            wildlife_candidate_limit: self.future_wildlife_candidate_limit,
            beam_width: self.beam_width,
            final_personal_turns: self.final_personal_turns,
        }
        .validate()?;
        Ok(self)
    }

    pub fn strategy_id(self) -> String {
        format!(
            "{PERFECT_INFORMATION_ROOT_DIVERSE_BEAM_STRATEGY_ID}-t{}-b{}-rootw{}-futurew{}-k{}-h{}-b{}-m{}",
            self.final_personal_turns,
            self.beam_width,
            self.root_wildlife_candidate_limit,
            self.future_wildlife_candidate_limit,
            self.blueprint.immediate_candidate_limit,
            self.blueprint.habitat_candidate_limit,
            self.blueprint.bear_candidate_limit,
            self.blueprint.future_market_draws,
        )
    }
}

pub struct PerfectInformationFocalBeamStrategy {
    config: PerfectInformationFocalBeamConfig,
    one_step: PerfectInformationPatternOracleStrategy,
    strategy_id: String,
}

pub struct PerfectInformationPortfolioBeamStrategy {
    beam: PerfectInformationFocalBeamStrategy,
    strategy_id: String,
}

pub struct PerfectInformationRootDiverseBeamStrategy {
    beam: PerfectInformationFocalBeamStrategy,
    root_wildlife_candidate_limit: usize,
    strategy_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublicBeamValueProbeConfig {
    pub blueprint: PatternAwareConfig,
    pub wildlife_candidate_limit: usize,
    pub beam_width: usize,
    pub final_personal_turns: u16,
    pub determinizations_per_batch: usize,
    pub batches: usize,
}

impl PublicBeamValueProbeConfig {
    pub fn validate(self) -> Result<Self, SearchError> {
        PerfectInformationFocalBeamConfig {
            blueprint: self.blueprint,
            wildlife_candidate_limit: self.wildlife_candidate_limit,
            beam_width: self.beam_width,
            final_personal_turns: self.final_personal_turns,
        }
        .validate()?;
        if self.determinizations_per_batch == 0 || self.batches < 2 {
            return Err(SearchError::InvalidConfig(
                "public beam value probe requires positive samples and at least two batches",
            ));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PublicBeamCandidateValue {
    pub action: TurnAction,
    pub immediate_rank: usize,
    pub immediate_score: u16,
    pub batch_means: Vec<f64>,
    pub batch_stddevs: Vec<f64>,
}

struct BeamNode {
    game: GameState,
    policy_rng: ChaCha8Rng,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PatternPolicyObservation {
    current_player: usize,
    completed_turns: u16,
    active_board: Board,
    market: Market,
    unplaced_wildlife_counts: [u8; 5],
}

#[derive(Clone)]
struct OpponentReplayStep {
    observation: PatternPolicyObservation,
    action: TurnAction,
    policy_rng_after: ChaCha8Rng,
}

struct OpponentReplay {
    initial_observation: Option<PatternPolicyObservation>,
    steps: Vec<OpponentReplayStep>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BeamRetention {
    Scalar,
    Portfolio,
}

const BEAM_DIMENSIONS: usize = 8;
const PORTFOLIO_DIMENSION_QUOTA: usize = 2;

impl PerfectInformationPatternOracleStrategy {
    pub fn new(config: PerfectInformationPatternOracleConfig) -> Result<Self, SearchError> {
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
        let prelude = MarketPrelude {
            replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
            wildlife_wipes: Vec::new(),
        };
        let staged = game.preview_market_prelude(&prelude)?;
        let candidates = if let Some(limit) = self.config.wildlife_candidate_limit {
            rank_wildlife_diverse_pattern_frontier_actions(
                &staged,
                &MarketPrelude::default(),
                self.config.blueprint,
                limit,
            )?
        } else {
            rank_pattern_frontier_actions(
                &staged,
                &MarketPrelude::default(),
                self.config.blueprint,
            )?
        };
        if candidates.is_empty() {
            return Err(SearchError::NoLegalActions);
        }

        let acting_seat = staged.current_player();
        let cards = staged.config().scoring_cards;
        let mut continuation_seed = [0; 32];
        rng.fill(&mut continuation_seed);
        let continuation_seed = GameSeed(continuation_seed);
        let scores = candidates
            .par_iter()
            .map(|candidate| {
                let mut sample = staged.clone();
                sample.apply(&candidate.action)?;
                let remaining_plies = usize::from(sample.turns_remaining());
                let mut policy_rng = rollout_rng(continuation_seed);
                play_pattern_plies(
                    &mut sample,
                    remaining_plies,
                    self.config.blueprint,
                    &mut policy_rng,
                )?;
                debug_assert!(sample.is_game_over());
                Ok::<_, SearchError>(f64::from(
                    score_board(&sample.boards()[acting_seat], cards).base_total,
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut ranked = candidates
            .into_iter()
            .zip(scores)
            .map(|(candidate, final_score)| RolloutCandidate {
                action: with_prelude(candidate.action, &prelude),
                immediate_rank: candidate.immediate_rank,
                immediate_score: candidate.resulting_base_score,
                mean_leaf_score: final_score,
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

    pub fn rank_actions_deterministic(
        &self,
        game: &GameState,
    ) -> Result<Vec<RolloutCandidate>, SearchError> {
        self.rank_actions(game, &mut lookahead_decision_rng(game))
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

impl PerfectInformationFocalBeamStrategy {
    pub fn new(config: PerfectInformationFocalBeamConfig) -> Result<Self, SearchError> {
        let config = config.validate()?;
        Ok(Self {
            one_step: PerfectInformationPatternOracleStrategy::new(
                PerfectInformationPatternOracleConfig {
                    blueprint: config.blueprint,
                    wildlife_candidate_limit: Some(config.wildlife_candidate_limit),
                },
            )?,
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

    pub fn rank_actions(
        &self,
        game: &GameState,
        rng: &mut ChaCha8Rng,
    ) -> Result<Vec<RolloutCandidate>, SearchError> {
        self.rank_actions_with_retention(game, rng, BeamRetention::Scalar)
    }

    fn rank_actions_with_retention(
        &self,
        game: &GameState,
        rng: &mut ChaCha8Rng,
        retention: BeamRetention,
    ) -> Result<Vec<RolloutCandidate>, SearchError> {
        self.rank_actions_with_root_limit_and_retention(
            game,
            rng,
            self.config.wildlife_candidate_limit,
            retention,
        )
    }

    fn rank_actions_with_root_limit_and_retention(
        &self,
        game: &GameState,
        rng: &mut ChaCha8Rng,
        root_wildlife_candidate_limit: usize,
        retention: BeamRetention,
    ) -> Result<Vec<RolloutCandidate>, SearchError> {
        let prelude = MarketPrelude {
            replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
            wildlife_wipes: Vec::new(),
        };
        let staged = game.preview_market_prelude(&prelude)?;
        let candidates = rank_wildlife_diverse_pattern_frontier_actions(
            &staged,
            &MarketPrelude::default(),
            self.config.blueprint,
            root_wildlife_candidate_limit,
        )?;
        if candidates.is_empty() {
            return Err(SearchError::NoLegalActions);
        }

        let acting_seat = staged.current_player();
        let mut continuation_seed = [0; 32];
        rng.fill(&mut continuation_seed);
        let continuation_seed = GameSeed(continuation_seed);
        let scores = candidates
            .par_iter()
            .map(|candidate| {
                self.evaluate_root_candidate_with_retention(
                    &staged,
                    &candidate.action,
                    acting_seat,
                    continuation_seed,
                    retention,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut ranked = candidates
            .into_iter()
            .zip(scores)
            .map(|(candidate, final_score)| RolloutCandidate {
                action: with_prelude(candidate.action, &prelude),
                immediate_rank: candidate.immediate_rank,
                immediate_score: candidate.resulting_base_score,
                mean_leaf_score: final_score,
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

    pub(crate) fn evaluate_root_candidate(
        &self,
        staged: &GameState,
        root_action: &TurnAction,
        focal_seat: usize,
        continuation_seed: GameSeed,
    ) -> Result<f64, SearchError> {
        self.evaluate_root_candidate_with_retention(
            staged,
            root_action,
            focal_seat,
            continuation_seed,
            BeamRetention::Scalar,
        )
    }

    fn evaluate_root_candidate_with_retention(
        &self,
        staged: &GameState,
        root_action: &TurnAction,
        focal_seat: usize,
        continuation_seed: GameSeed,
        retention: BeamRetention,
    ) -> Result<f64, SearchError> {
        self.evaluate_root_candidate_with_retention_impl(
            staged,
            root_action,
            focal_seat,
            continuation_seed,
            retention,
            true,
        )
    }

    fn evaluate_root_candidate_with_retention_impl(
        &self,
        staged: &GameState,
        root_action: &TurnAction,
        focal_seat: usize,
        continuation_seed: GameSeed,
        retention: BeamRetention,
        replay_sibling_opponents: bool,
    ) -> Result<f64, SearchError> {
        let mut root = BeamNode {
            game: staged.clone(),
            policy_rng: rollout_rng(continuation_seed),
        };
        root.game.apply(root_action)?;
        self.advance_opponents(&mut root, focal_seat)?;
        let mut beam = vec![root];

        while !beam[0].game.is_game_over() {
            let mut expanded = Vec::new();
            for node in beam {
                debug_assert_eq!(node.game.current_player(), focal_seat);
                let prelude = MarketPrelude {
                    replace_three_of_a_kind: node.game.market().three_of_a_kind().is_some(),
                    wildlife_wipes: Vec::new(),
                };
                let staged = node.game.preview_market_prelude(&prelude)?;
                let candidates = rank_wildlife_diverse_pattern_frontier_actions(
                    &staged,
                    &MarketPrelude::default(),
                    self.config.blueprint,
                    self.config.wildlife_candidate_limit,
                )?;
                let mut opponent_replays = Vec::new();
                for candidate in candidates {
                    let mut child = BeamNode {
                        game: staged.clone(),
                        policy_rng: node.policy_rng.clone(),
                    };
                    child.game.apply(&candidate.action)?;
                    let initial_observation =
                        Self::pattern_policy_observation(&child.game, focal_seat);
                    let replay = replay_sibling_opponents.then(|| {
                        opponent_replays.iter().find(|replay: &&OpponentReplay| {
                            replay.initial_observation == initial_observation
                        })
                    });
                    if let Some(Some(replay)) = replay {
                        if !Self::replay_opponents(&mut child, focal_seat, replay)? {
                            self.advance_opponents(&mut child, focal_seat)?;
                        }
                    } else if replay_sibling_opponents {
                        let replay = self.advance_opponents_recording(&mut child, focal_seat)?;
                        opponent_replays.push(replay);
                    } else {
                        self.advance_opponents(&mut child, focal_seat)?;
                    }
                    expanded.push(child);
                }
            }
            if expanded.is_empty() {
                return Err(SearchError::NoLegalActions);
            }
            let scored = expanded
                .into_iter()
                .map(|node| {
                    let dimensions = self.beam_dimensions(&node.game, focal_seat)?;
                    Ok((node, dimensions))
                })
                .collect::<Result<Vec<_>, SearchError>>()?;
            beam = retain_beam(scored, self.config.beam_width, retention);
        }

        let cards = staged.config().scoring_cards;
        Ok(beam
            .iter()
            .map(|node| f64::from(score_board(&node.game.boards()[focal_seat], cards).base_total))
            .max_by(f64::total_cmp)
            .expect("beam always retains at least one terminal state"))
    }

    fn advance_opponents(&self, node: &mut BeamNode, focal_seat: usize) -> Result<(), SearchError> {
        while !node.game.is_game_over() && node.game.current_player() != focal_seat {
            let prelude = MarketPrelude {
                replace_three_of_a_kind: node.game.market().three_of_a_kind().is_some(),
                wildlife_wipes: Vec::new(),
            };
            let action = select_pattern_action(
                &node.game,
                &prelude,
                self.config.blueprint,
                &mut node.policy_rng,
            )?;
            node.game.apply(&action)?;
        }
        Ok(())
    }

    fn advance_opponents_recording(
        &self,
        node: &mut BeamNode,
        focal_seat: usize,
    ) -> Result<OpponentReplay, SearchError> {
        let initial_observation = Self::pattern_policy_observation(&node.game, focal_seat);
        let mut steps = Vec::new();
        while !node.game.is_game_over() && node.game.current_player() != focal_seat {
            let observation = Self::pattern_policy_observation(&node.game, focal_seat)
                .expect("an unfinished opponent turn has a policy observation");
            let prelude = MarketPrelude {
                replace_three_of_a_kind: node.game.market().three_of_a_kind().is_some(),
                wildlife_wipes: Vec::new(),
            };
            let action = select_pattern_action(
                &node.game,
                &prelude,
                self.config.blueprint,
                &mut node.policy_rng,
            )?;
            steps.push(OpponentReplayStep {
                observation,
                action: action.clone(),
                policy_rng_after: node.policy_rng.clone(),
            });
            node.game.apply(&action)?;
        }
        Ok(OpponentReplay {
            initial_observation,
            steps,
        })
    }

    fn replay_opponents(
        node: &mut BeamNode,
        focal_seat: usize,
        replay: &OpponentReplay,
    ) -> Result<bool, SearchError> {
        for step in &replay.steps {
            if Self::pattern_policy_observation(&node.game, focal_seat)
                != Some(step.observation.clone())
            {
                return Ok(false);
            }
            node.game.apply(&step.action)?;
            node.policy_rng = step.policy_rng_after.clone();
        }
        Ok(node.game.is_game_over() || node.game.current_player() == focal_seat)
    }

    fn pattern_policy_observation(
        game: &GameState,
        focal_seat: usize,
    ) -> Option<PatternPolicyObservation> {
        if game.is_game_over() || game.current_player() == focal_seat {
            return None;
        }
        let current_player = game.current_player();
        Some(PatternPolicyObservation {
            current_player,
            completed_turns: game.completed_turns(),
            active_board: game.boards()[current_player].clone(),
            market: game.market().clone(),
            unplaced_wildlife_counts: game.unplaced_wildlife_counts(),
        })
    }

    fn beam_heuristic(&self, game: &GameState, focal_seat: usize) -> Result<f64, SearchError> {
        let cards = game.config().scoring_cards;
        if game.is_game_over() {
            return Ok(f64::from(
                score_board(&game.boards()[focal_seat], cards).base_total,
            ));
        }
        debug_assert_eq!(game.current_player(), focal_seat);
        let prelude = MarketPrelude {
            replace_three_of_a_kind: game.market().three_of_a_kind().is_some(),
            wildlife_wipes: Vec::new(),
        };
        Ok(
            best_pattern_heuristic_value(game, &prelude, self.config.blueprint)?.unwrap_or_else(
                || f64::from(score_board(&game.boards()[focal_seat], cards).base_total),
            ),
        )
    }

    fn beam_dimensions(
        &self,
        game: &GameState,
        focal_seat: usize,
    ) -> Result<[f64; BEAM_DIMENSIONS], SearchError> {
        let score = score_board(&game.boards()[focal_seat], game.config().scoring_cards);
        Ok([
            self.beam_heuristic(game, focal_seat)?,
            f64::from(score.habitat.iter().sum::<u16>()),
            f64::from(score.wildlife[0]),
            f64::from(score.wildlife[1]),
            f64::from(score.wildlife[2]),
            f64::from(score.wildlife[3]),
            f64::from(score.wildlife[4]),
            f64::from(score.nature_tokens),
        ])
    }

    pub fn rank_actions_deterministic(
        &self,
        game: &GameState,
    ) -> Result<Vec<RolloutCandidate>, SearchError> {
        if self.uses_beam(game) {
            self.rank_actions(game, &mut lookahead_decision_rng(game))
        } else {
            self.one_step.rank_actions_deterministic(game)
        }
    }

    pub fn rank_and_select_deterministic(
        &self,
        game: &GameState,
    ) -> Result<(Vec<RolloutCandidate>, TurnAction), SearchError> {
        let mut rng = lookahead_decision_rng(game);
        let ranked = if self.uses_beam(game) {
            self.rank_actions(game, &mut rng)?
        } else {
            self.one_step.rank_actions(game, &mut rng)?
        };
        let action = select_ranked_action(&ranked, &mut rng)?;
        Ok((ranked, action))
    }

    pub fn select_action_deterministic(&self, game: &GameState) -> Result<TurnAction, SearchError> {
        Ok(self.rank_and_select_deterministic(game)?.1)
    }
}

impl PerfectInformationPortfolioBeamStrategy {
    pub fn new(config: PerfectInformationPortfolioBeamConfig) -> Result<Self, SearchError> {
        let config = config.validate()?;
        Ok(Self {
            beam: PerfectInformationFocalBeamStrategy::new(PerfectInformationFocalBeamConfig {
                blueprint: config.blueprint,
                wildlife_candidate_limit: config.wildlife_candidate_limit,
                beam_width: config.beam_width,
                final_personal_turns: config.final_personal_turns,
            })?,
            strategy_id: config.strategy_id(),
        })
    }

    pub fn strategy_id(&self) -> &str {
        &self.strategy_id
    }

    pub fn rank_actions_deterministic(
        &self,
        game: &GameState,
    ) -> Result<Vec<RolloutCandidate>, SearchError> {
        if self.beam.uses_beam(game) {
            self.beam.rank_actions_with_retention(
                game,
                &mut lookahead_decision_rng(game),
                BeamRetention::Portfolio,
            )
        } else {
            self.beam.one_step.rank_actions_deterministic(game)
        }
    }

    pub fn rank_and_select_deterministic(
        &self,
        game: &GameState,
    ) -> Result<(Vec<RolloutCandidate>, TurnAction), SearchError> {
        let mut rng = lookahead_decision_rng(game);
        let ranked = if self.beam.uses_beam(game) {
            self.beam
                .rank_actions_with_retention(game, &mut rng, BeamRetention::Portfolio)?
        } else {
            self.beam.one_step.rank_actions(game, &mut rng)?
        };
        let action = select_ranked_action(&ranked, &mut rng)?;
        Ok((ranked, action))
    }

    pub fn select_action_deterministic(&self, game: &GameState) -> Result<TurnAction, SearchError> {
        Ok(self.rank_and_select_deterministic(game)?.1)
    }
}

impl PerfectInformationRootDiverseBeamStrategy {
    pub fn new(config: PerfectInformationRootDiverseBeamConfig) -> Result<Self, SearchError> {
        let config = config.validate()?;
        Ok(Self {
            beam: PerfectInformationFocalBeamStrategy::new(PerfectInformationFocalBeamConfig {
                blueprint: config.blueprint,
                wildlife_candidate_limit: config.future_wildlife_candidate_limit,
                beam_width: config.beam_width,
                final_personal_turns: config.final_personal_turns,
            })?,
            root_wildlife_candidate_limit: config.root_wildlife_candidate_limit,
            strategy_id: config.strategy_id(),
        })
    }

    pub fn strategy_id(&self) -> &str {
        &self.strategy_id
    }

    pub fn rank_actions_deterministic(
        &self,
        game: &GameState,
    ) -> Result<Vec<RolloutCandidate>, SearchError> {
        if self.beam.uses_beam(game) {
            self.beam.rank_actions_with_root_limit_and_retention(
                game,
                &mut lookahead_decision_rng(game),
                self.root_wildlife_candidate_limit,
                BeamRetention::Scalar,
            )
        } else {
            self.beam.one_step.rank_actions_deterministic(game)
        }
    }

    pub fn rank_and_select_deterministic(
        &self,
        game: &GameState,
    ) -> Result<(Vec<RolloutCandidate>, TurnAction), SearchError> {
        let mut rng = lookahead_decision_rng(game);
        let ranked = if self.beam.uses_beam(game) {
            self.beam.rank_actions_with_root_limit_and_retention(
                game,
                &mut rng,
                self.root_wildlife_candidate_limit,
                BeamRetention::Scalar,
            )?
        } else {
            self.beam.one_step.rank_actions(game, &mut rng)?
        };
        let action = select_ranked_action(&ranked, &mut rng)?;
        Ok((ranked, action))
    }

    pub fn select_action_deterministic(&self, game: &GameState) -> Result<TurnAction, SearchError> {
        Ok(self.rank_and_select_deterministic(game)?.1)
    }
}

pub fn evaluate_public_beam_value_batches(
    game: &GameState,
    config: PublicBeamValueProbeConfig,
) -> Result<Vec<PublicBeamCandidateValue>, SearchError> {
    let config = config.validate()?;
    let beam = PerfectInformationFocalBeamStrategy::new(PerfectInformationFocalBeamConfig {
        blueprint: config.blueprint,
        wildlife_candidate_limit: config.wildlife_candidate_limit,
        beam_width: config.beam_width,
        final_personal_turns: config.final_personal_turns,
    })?;
    // The root frontier must be a function of the current public market. Applying
    // a replacement here would reveal a concealed refill before redetermination.
    let prelude = MarketPrelude::default();
    let staged = game.clone();
    let candidates = rank_wildlife_diverse_pattern_frontier_actions(
        &staged,
        &MarketPrelude::default(),
        config.blueprint,
        config.wildlife_candidate_limit,
    )?;
    if candidates.is_empty() {
        return Err(SearchError::NoLegalActions);
    }

    let public_bytes = PositionRecord::observe(game, 0).to_bytes();
    let acting_seat = staged.current_player();
    let sample_count = config.determinizations_per_batch;
    let scores = (0..candidates.len() * config.batches * sample_count)
        .into_par_iter()
        .map(|job| {
            let candidate_index = job / (config.batches * sample_count);
            let within_candidate = job % (config.batches * sample_count);
            let batch_index = within_candidate / sample_count;
            let sample_index = within_candidate % sample_count;
            let sample_seed =
                public_beam_sample_seed(&public_bytes, batch_index as u64, sample_index as u64);
            let mut sample = staged.clone();
            sample.redeterminize_hidden(sample_seed);
            let score = beam.evaluate_root_candidate(
                &sample,
                &candidates[candidate_index].action,
                acting_seat,
                sample_seed,
            )?;
            Ok::<_, SearchError>((candidate_index, batch_index, score))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut values = vec![vec![Vec::with_capacity(sample_count); config.batches]; candidates.len()];
    for (candidate_index, batch_index, score) in scores {
        values[candidate_index][batch_index].push(score);
    }
    Ok(candidates
        .into_iter()
        .enumerate()
        .map(|(candidate_index, candidate)| PublicBeamCandidateValue {
            action: with_prelude(candidate.action, &prelude),
            immediate_rank: candidate.immediate_rank,
            immediate_score: candidate.resulting_base_score,
            batch_means: values[candidate_index]
                .iter()
                .map(|batch| batch.iter().sum::<f64>() / batch.len() as f64)
                .collect(),
            batch_stddevs: values[candidate_index]
                .iter()
                .map(|batch| {
                    let mean = batch.iter().sum::<f64>() / batch.len() as f64;
                    (batch
                        .iter()
                        .map(|value| (value - mean).powi(2))
                        .sum::<f64>()
                        / batch.len() as f64)
                        .sqrt()
                })
                .collect(),
        })
        .collect())
}

fn public_beam_sample_seed(
    public_position_bytes: &[u8],
    batch_index: u64,
    sample_index: u64,
) -> GameSeed {
    let mut hasher = Hasher::new();
    hasher.update(b"cascadia-v2/public-beam-state-value-v1");
    hasher.update(public_position_bytes);
    hasher.update(&batch_index.to_le_bytes());
    hasher.update(&sample_index.to_le_bytes());
    GameSeed(*hasher.finalize().as_bytes())
}

fn retain_beam(
    scored: Vec<(BeamNode, [f64; BEAM_DIMENSIONS])>,
    width: usize,
    retention: BeamRetention,
) -> Vec<BeamNode> {
    let dimensions = scored
        .iter()
        .map(|(_, dimensions)| *dimensions)
        .collect::<Vec<_>>();
    let retained = retention_indices(&dimensions, width, retention);
    let mut nodes = scored
        .into_iter()
        .map(|(node, _)| Some(node))
        .collect::<Vec<_>>();
    retained
        .into_iter()
        .map(|index| nodes[index].take().expect("retention indexes are unique"))
        .collect()
}

fn retention_indices(
    dimensions: &[[f64; BEAM_DIMENSIONS]],
    width: usize,
    retention: BeamRetention,
) -> Vec<usize> {
    let limit = width.min(dimensions.len());
    if limit == 0 {
        return Vec::new();
    }
    if retention == BeamRetention::Scalar || limit == 1 {
        return sorted_dimension_indices(dimensions, 0)
            .into_iter()
            .take(limit)
            .collect();
    }

    let mut selected = Vec::with_capacity(limit);
    let mut included = vec![false; dimensions.len()];
    for dimension in 0..BEAM_DIMENSIONS {
        for index in sorted_dimension_indices(dimensions, dimension)
            .into_iter()
            .take(PORTFOLIO_DIMENSION_QUOTA)
        {
            if included[index] {
                continue;
            }
            selected.push(index);
            included[index] = true;
            if selected.len() == limit {
                return selected;
            }
        }
    }

    for index in sorted_dimension_indices(dimensions, 0) {
        if !included[index] {
            selected.push(index);
            included[index] = true;
            if selected.len() == limit {
                break;
            }
        }
    }
    selected
}

fn sorted_dimension_indices(dimensions: &[[f64; BEAM_DIMENSIONS]], dimension: usize) -> Vec<usize> {
    let mut indexes = (0..dimensions.len()).collect::<Vec<_>>();
    indexes.sort_by(|left, right| {
        dimensions[*right][dimension]
            .total_cmp(&dimensions[*left][dimension])
            .then_with(|| dimensions[*right][0].total_cmp(&dimensions[*left][0]))
            .then_with(|| left.cmp(right))
    });
    indexes
}

#[cfg(test)]
#[path = "oracle_tests.rs"]
mod tests;
