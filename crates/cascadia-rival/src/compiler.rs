//! Dense semantic reference compilation.
//!
//! This module deliberately optimizes nothing.  It delegates every transition
//! and score to `cascadia-game`, then records a stable integer description of
//! the result.  Component-local and incremental compilers must eventually
//! match this oracle exactly; they do not belong in the pre-D1 CPU tranche.

use cascadia_game::{GameConfig, GameState, RuleError, ScoreBreakdown, TurnAction, score_game};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const DENSE_COMPILER_ID: &str = "cascadiav3.rival_dense_semantic_compiler.v1";

/// Signed score-category change caused by one complete canonical turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreDelta {
    pub habitat: [i32; 5],
    pub wildlife: [i32; 5],
    pub nature_tokens: i32,
    pub habitat_bonus: [i32; 5],
    pub base_total: i32,
    pub total: i32,
}

impl ScoreDelta {
    pub fn between(before: ScoreBreakdown, after: ScoreBreakdown) -> Self {
        Self {
            habitat: std::array::from_fn(|index| {
                i32::from(after.habitat[index]) - i32::from(before.habitat[index])
            }),
            wildlife: std::array::from_fn(|index| {
                i32::from(after.wildlife[index]) - i32::from(before.wildlife[index])
            }),
            nature_tokens: i32::from(after.nature_tokens) - i32::from(before.nature_tokens),
            habitat_bonus: std::array::from_fn(|index| {
                i32::from(after.habitat_bonus[index]) - i32::from(before.habitat_bonus[index])
            }),
            base_total: i32::from(after.base_total) - i32::from(before.base_total),
            total: i32::from(after.total) - i32::from(before.total),
        }
    }

    pub fn category_sum(self) -> i32 {
        self.habitat.into_iter().sum::<i32>()
            + self.wildlife.into_iter().sum::<i32>()
            + self.nature_tokens
            + self.habitat_bonus.into_iter().sum::<i32>()
    }
}

/// Full-recomputation semantics for one reachable state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseStateSemantics {
    pub compiler_id: String,
    pub state_hash: [u8; 32],
    pub current_player: usize,
    pub completed_turns: u16,
    pub terminal: bool,
    pub scores: Vec<ScoreBreakdown>,
}

impl DenseStateSemantics {
    pub fn own_score(&self, seat: usize) -> Option<ScoreBreakdown> {
        self.scores.get(seat).copied()
    }
}

/// Full-recomputation semantics for one accepted complete turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseActionSemantics {
    pub compiler_id: String,
    pub actor: usize,
    pub source_state_hash: [u8; 32],
    pub after_state_hash: [u8; 32],
    pub before: ScoreBreakdown,
    pub after: ScoreBreakdown,
    pub own_score_delta: ScoreDelta,
}

/// Contract optimized compilers must implement without changing semantics.
pub trait SemanticCompiler {
    type StateSemantics;
    type ActionSemantics;

    fn compile_state(&self, game: &GameState) -> Result<Self::StateSemantics, CompilerError>;

    fn compile_action(
        &self,
        game: &GameState,
        action: &TurnAction,
    ) -> Result<Self::ActionSemantics, CompilerError>;
}

/// The sole P1 oracle: canonical transition plus canonical full rescoring.
#[derive(Debug, Default, Clone, Copy)]
pub struct DenseSemanticCompiler;

impl SemanticCompiler for DenseSemanticCompiler {
    type StateSemantics = DenseStateSemantics;
    type ActionSemantics = DenseActionSemantics;

    fn compile_state(&self, game: &GameState) -> Result<DenseStateSemantics, CompilerError> {
        validate_source(game)?;
        Ok(DenseStateSemantics {
            compiler_id: DENSE_COMPILER_ID.to_owned(),
            state_hash: *game.canonical_hash().as_bytes(),
            current_player: game.current_player(),
            completed_turns: game.completed_turns(),
            terminal: game.is_game_over(),
            scores: score_game(game),
        })
    }

    fn compile_action(
        &self,
        game: &GameState,
        action: &TurnAction,
    ) -> Result<DenseActionSemantics, CompilerError> {
        validate_source(game)?;
        if game.is_game_over() {
            return Err(CompilerError::TerminalState);
        }
        let actor = game.current_player();
        let before = score_game(game)
            .get(actor)
            .copied()
            .ok_or(CompilerError::MissingActorScore(actor))?;
        let after_state = game.transition(action)?;
        let after = score_game(&after_state)
            .get(actor)
            .copied()
            .ok_or(CompilerError::MissingActorScore(actor))?;
        let own_score_delta = ScoreDelta::between(before, after);
        if own_score_delta.category_sum() != own_score_delta.total {
            return Err(CompilerError::ScoreDecompositionMismatch);
        }
        Ok(DenseActionSemantics {
            compiler_id: DENSE_COMPILER_ID.to_owned(),
            actor,
            source_state_hash: *game.canonical_hash().as_bytes(),
            after_state_hash: *after_state.canonical_hash().as_bytes(),
            before,
            after,
            own_score_delta,
        })
    }
}

fn validate_source(game: &GameState) -> Result<(), CompilerError> {
    if game.config() != GameConfig::research_aaaaa(4).expect("valid research config") {
        return Err(CompilerError::WrongRuleset);
    }
    game.validate().map_err(CompilerError::InvalidState)
}

#[derive(Debug, Error)]
pub enum CompilerError {
    #[error("dense Rival semantics require corrected-rules four-player research AAAAA")]
    WrongRuleset,
    #[error("cannot compile an invalid canonical game state: {0}")]
    InvalidState(&'static str),
    #[error("cannot compile an action from a terminal state")]
    TerminalState,
    #[error("canonical scoring omitted actor seat {0}")]
    MissingActorScore(usize),
    #[error("canonical score category deltas do not sum to the canonical total delta")]
    ScoreDecompositionMismatch,
    #[error(transparent)]
    Rules(#[from] RuleError),
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameConfig, GameSeed, MarketPrelude};

    use super::*;

    fn fixture() -> GameState {
        GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(0x5eed),
        )
        .unwrap()
    }

    #[test]
    fn dense_state_is_exact_canonical_rescoring() {
        let game = fixture();
        let compiled = DenseSemanticCompiler.compile_state(&game).unwrap();
        assert_eq!(compiled.scores, score_game(&game));
        assert_eq!(compiled.state_hash, *game.canonical_hash().as_bytes());
        assert_eq!(compiled.current_player, game.current_player());
        assert!(!compiled.terminal);
    }

    #[test]
    fn dense_action_matches_transactional_transition_and_category_sum() {
        let game = fixture();
        let action = game
            .legal_turn_actions(&MarketPrelude::default())
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let compiled = DenseSemanticCompiler
            .compile_action(&game, &action)
            .unwrap();
        let next = game.transition(&action).unwrap();
        assert_eq!(compiled.after_state_hash, *next.canonical_hash().as_bytes());
        assert_eq!(compiled.after, score_game(&next)[compiled.actor]);
        assert_eq!(
            compiled.own_score_delta,
            ScoreDelta::between(compiled.before, compiled.after)
        );
        assert_eq!(
            compiled.own_score_delta.category_sum(),
            compiled.own_score_delta.total
        );
    }

    #[test]
    fn dense_semantics_roundtrip_without_loss() {
        let game = fixture();
        let compiled = DenseSemanticCompiler.compile_state(&game).unwrap();
        let bytes = postcard::to_allocvec(&compiled).unwrap();
        let decoded: DenseStateSemantics = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, compiled);
    }

    #[test]
    fn dense_compiler_rejects_nonresearch_and_invalid_states() {
        let wrong_config = GameState::new(
            GameConfig::research_aaaaa(2).unwrap(),
            GameSeed::from_u64(7),
        )
        .unwrap();
        assert!(matches!(
            DenseSemanticCompiler.compile_state(&wrong_config),
            Err(CompilerError::WrongRuleset)
        ));
        let action = wrong_config
            .legal_turn_actions(&MarketPrelude::default())
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        assert!(matches!(
            DenseSemanticCompiler.compile_action(&wrong_config, &action),
            Err(CompilerError::WrongRuleset)
        ));

        let mut value = serde_json::to_value(fixture()).unwrap();
        value["current_player"] = serde_json::json!(9);
        let invalid: GameState = serde_json::from_value(value).unwrap();
        assert!(matches!(
            DenseSemanticCompiler.compile_state(&invalid),
            Err(CompilerError::InvalidState(_))
        ));
    }
}
