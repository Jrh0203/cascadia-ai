//! Search and learned action selection over canonical Cascadia states.

mod action_ranking;
mod habitat_ranking;
mod lookahead;
mod mlx_prefilter;
mod mlx_ranking_rollout;
mod mlx_value;
mod oracle;
mod pattern_rollout;
mod policy_improvement;
mod public_tree;
mod ranking_prediction;

use blake3::Hasher;
use cascadia_data::DataError;
use cascadia_game::{GameSeed, GameState, RuleError};
use cascadia_model::ModelError;
use cascadia_sim::SimulationError;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use thiserror::Error;

pub use action_ranking::{
    ActionRankingPredictor, ImitationPredictor, MLX_ACTION_DELTA_RANKING_STRATEGY_ID,
    MLX_FULL_ACTION_IMITATION_STRATEGY_ID, MLX_PUBLIC_BEAM_VALUE_STRATEGY_ID,
    MlxActionDeltaRankingConfig, MlxActionDeltaRankingStrategy, MlxFullActionImitationStrategy,
    MlxPublicBeamValueConfig, MlxPublicBeamValueStrategy,
};
pub use habitat_ranking::{
    MLX_HABITAT_RANKING_STRATEGY_ID, MLX_PATTERN_RANKING_STRATEGY_ID, MlxHabitatRankingConfig,
    MlxHabitatRankingStrategy, MlxPatternRankingConfig, MlxPatternRankingStrategy,
};
pub use oracle::{
    PERFECT_INFORMATION_FOCAL_BEAM_STRATEGY_ID, PERFECT_INFORMATION_PATTERN_ORACLE_STRATEGY_ID,
    PERFECT_INFORMATION_PORTFOLIO_BEAM_STRATEGY_ID,
    PERFECT_INFORMATION_ROOT_DIVERSE_BEAM_STRATEGY_ID, PerfectInformationFocalBeamConfig,
    PerfectInformationFocalBeamStrategy, PerfectInformationPatternOracleConfig,
    PerfectInformationPatternOracleStrategy, PerfectInformationPortfolioBeamConfig,
    PerfectInformationPortfolioBeamStrategy, PerfectInformationRootDiverseBeamConfig,
    PerfectInformationRootDiverseBeamStrategy, PublicBeamCandidateValue,
    PublicBeamValueProbeConfig, evaluate_public_beam_value_batches,
};
pub use pattern_rollout::{
    PATTERN_BLUEPRINT_LOOKAHEAD_STRATEGY_ID, PatternBlueprintLookaheadConfig,
    PatternBlueprintLookaheadStrategy,
};
pub use policy_improvement::{
    ConservativeCandidateEvaluation, LATE_CONSERVATIVE_BASE_POLICY_IMPROVEMENT_STRATEGY_ID,
    LATE_CONSERVATIVE_FOCAL_BEAM_STRATEGY_ID, LATE_CONSERVATIVE_POLICY_IMPROVEMENT_STRATEGY_ID,
    LATE_CONSERVATIVE_WILDLIFE_FOCUSED_POLICY_IMPROVEMENT_STRATEGY_ID,
    LATE_TERMINAL_POLICY_IMPROVEMENT_STRATEGY_ID,
    LATE_WILDLIFE_DIVERSE_POLICY_IMPROVEMENT_STRATEGY_ID,
    LateConservativeBasePolicyImprovementConfig, LateConservativeBasePolicyImprovementStrategy,
    LateConservativeFocalBeamConfig, LateConservativeFocalBeamStrategy,
    LateConservativePolicyImprovementConfig, LateConservativePolicyImprovementStrategy,
    LateConservativeWildlifeFocusedPolicyImprovementConfig,
    LateConservativeWildlifeFocusedPolicyImprovementStrategy, LateTerminalPolicyImprovementConfig,
    LateTerminalPolicyImprovementStrategy, LateWildlifeDiversePolicyImprovementConfig,
    LateWildlifeDiversePolicyImprovementStrategy, TERMINAL_POLICY_IMPROVEMENT_STRATEGY_ID,
    TerminalPolicyImprovementConfig, TerminalPolicyImprovementStrategy,
    WILDLIFE_DIVERSE_TERMINAL_POLICY_IMPROVEMENT_STRATEGY_ID,
    WildlifeDiverseTerminalPolicyImprovementConfig,
    WildlifeDiverseTerminalPolicyImprovementStrategy,
    WildlifeFocusedTerminalPolicyImprovementConfig,
};
pub use public_tree::{
    PUBLIC_FOCAL_OPEN_LOOP_TREE_STRATEGY_ID, PublicFocalOpenLoopTreeConfig,
    PublicFocalOpenLoopTreeStrategy, PublicTreeAnalysis, PublicTreeRootEvaluation,
};

pub use lookahead::{
    BearCandidateLookaheadConfig, BearCandidateLookaheadStrategy,
    BearHabitatCandidateLookaheadConfig, BearHabitatCandidateLookaheadStrategy,
    DeterminizedLookaheadConfig, DeterminizedLookaheadStrategy, HabitatCandidateLookaheadConfig,
    HabitatCandidateLookaheadStrategy, NatureWipeLookaheadConfig, NatureWipeLookaheadStrategy,
    RolloutCandidate,
};
pub(crate) use lookahead::{
    bear_candidate_union, habitat_candidate_union, rank_rollout_candidates, select_ranked_action,
    with_prelude,
};
pub use mlx_prefilter::{
    MlxHabitatPrefilteredLookaheadConfig, MlxHabitatPrefilteredLookaheadStrategy,
    MlxPrefilteredLookaheadConfig, MlxPrefilteredLookaheadStrategy,
};
pub(crate) use mlx_ranking_rollout::finish_rollout_ranking;
pub use mlx_ranking_rollout::{
    MLX_HABITAT_PREFILTER_LOOKAHEAD_STRATEGY_ID, MLX_HABITAT_ROLLOUT_LOOKAHEAD_STRATEGY_ID,
    MLX_PREFILTER_LOOKAHEAD_STRATEGY_ID, MLX_RANKING_STRATEGY_ID,
    MLX_SELF_ROLLOUT_LOOKAHEAD_STRATEGY_ID, MlxHabitatRolloutLookaheadConfig,
    MlxHabitatRolloutLookaheadStrategy, MlxRankingConfig, MlxRankingStrategy,
    MlxSelfRolloutLookaheadConfig, MlxSelfRolloutLookaheadStrategy, RankingPredictor,
};
pub use mlx_value::{
    MlxValueConfig, MlxValueLeafLookaheadConfig, MlxValueLeafLookaheadStrategy, MlxValueStrategy,
    Predictor,
};
pub(crate) use ranking_prediction::predict_ranking_scores;

pub const MLX_VALUE_STRATEGY_ID: &str = "mlx-value-v1";
pub const MLX_VALUE_LEAF_LOOKAHEAD_STRATEGY_ID: &str = "mlx-value-leaf-lookahead-v1";
pub const DETERMINIZED_LOOKAHEAD_STRATEGY_ID: &str = "determinized-lookahead-v2";

fn strategy_rng(seed: GameSeed, seat: u8) -> ChaCha8Rng {
    let mut hasher = Hasher::new();
    hasher.update(b"cascadia-v2-mlx-value-rng");
    hasher.update(&seed.0);
    hasher.update(&[seat]);
    ChaCha8Rng::from_seed(*hasher.finalize().as_bytes())
}

pub(crate) fn ranking_decision_rng(game: &GameState) -> ChaCha8Rng {
    let mut hasher = Hasher::new();
    hasher.update(b"cascadia-v2-mlx-ranking-decision-rng");
    hasher.update(&game.seed().0);
    hasher.update(&game.completed_turns().to_le_bytes());
    hasher.update(&(game.current_player() as u64).to_le_bytes());
    ChaCha8Rng::from_seed(*hasher.finalize().as_bytes())
}

pub(crate) fn lookahead_decision_rng(game: &GameState) -> ChaCha8Rng {
    let mut hasher = Hasher::new();
    hasher.update(b"cascadia-v2-determinized-lookahead-decision-rng");
    hasher.update(&game.seed().0);
    hasher.update(&game.completed_turns().to_le_bytes());
    hasher.update(&(game.current_player() as u64).to_le_bytes());
    ChaCha8Rng::from_seed(*hasher.finalize().as_bytes())
}

fn nature_wipe_decision_rng(game: &GameState) -> ChaCha8Rng {
    let mut hasher = Hasher::new();
    hasher.update(b"cascadia-v2-nature-wipe-decision-rng");
    hasher.update(&game.seed().0);
    hasher.update(&game.completed_turns().to_le_bytes());
    hasher.update(&(game.current_player() as u64).to_le_bytes());
    ChaCha8Rng::from_seed(*hasher.finalize().as_bytes())
}

pub(crate) fn rollout_rng(seed: GameSeed) -> ChaCha8Rng {
    let mut hasher = Hasher::new();
    hasher.update(b"cascadia-v2-greedy-rollout-rng");
    hasher.update(&seed.0);
    ChaCha8Rng::from_seed(*hasher.finalize().as_bytes())
}

fn conditioned_rollout_seed(base_seed: GameSeed, attempt: u64) -> GameSeed {
    if attempt == 0 {
        return base_seed;
    }
    let mut hasher = Hasher::new();
    hasher.update(b"cascadia-v2-stable-market-rollout-rejection-v1");
    hasher.update(&base_seed.0);
    hasher.update(&attempt.to_le_bytes());
    GameSeed(*hasher.finalize().as_bytes())
}

#[derive(Debug, Error)]
pub enum SearchError {
    #[error("learned strategy found no legal actions")]
    NoLegalActions,
    #[error("invalid learned search configuration: {0}")]
    InvalidConfig(&'static str),
    #[error("model returned {actual} predictions for {expected} candidates")]
    PredictionCount { expected: usize, actual: usize },
    #[error("model returned a non-finite prediction for candidate {index}")]
    NonFinitePrediction { index: usize },
    #[error(transparent)]
    Rules(#[from] RuleError),
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error(transparent)]
    Data(#[from] DataError),
    #[error(transparent)]
    Simulation(#[from] SimulationError),
}

impl SearchError {
    pub fn is_unstable_market_exhaustion(&self) -> bool {
        matches!(
            self,
            Self::Rules(RuleError::WildlifeBagEmpty)
                | Self::Simulation(SimulationError::Rules(RuleError::WildlifeBagEmpty))
        )
    }
}

#[cfg(test)]
#[path = "search_tests.rs"]
mod tests;
