use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{GameConfig, GameSeed, GameState, RuleError, TurnAction};

const REPLAY_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Replay {
    pub schema_version: u16,
    pub config: GameConfig,
    pub seed: GameSeed,
    pub turns: Vec<TurnAction>,
    pub final_state_hash: Option<[u8; 32]>,
}

impl Replay {
    pub fn new(config: GameConfig, seed: GameSeed) -> Self {
        Self {
            schema_version: REPLAY_SCHEMA_VERSION,
            config,
            seed,
            turns: Vec::new(),
            final_state_hash: None,
        }
    }

    pub fn play(&self) -> Result<GameState, ReplayError> {
        if self.schema_version != REPLAY_SCHEMA_VERSION {
            return Err(ReplayError::UnsupportedSchema(self.schema_version));
        }
        let mut game = GameState::new(self.config, self.seed)?;
        for (turn, action) in self.turns.iter().enumerate() {
            game.apply(action)
                .map_err(|source| ReplayError::InvalidTurn { turn, source })?;
        }
        if let Some(expected) = self.final_state_hash {
            let actual = *game.canonical_hash().as_bytes();
            if actual != expected {
                return Err(ReplayError::HashMismatch { expected, actual });
            }
        }
        Ok(game)
    }

    pub fn seal(&mut self) -> Result<[u8; 32], ReplayError> {
        let hash = *self.play()?.canonical_hash().as_bytes();
        self.final_state_hash = Some(hash);
        Ok(hash)
    }
}

#[derive(Debug, Error)]
pub enum ReplayError {
    #[error("replay schema version {0} is not supported")]
    UnsupportedSchema(u16),
    #[error("replay turn {turn} is invalid: {source}")]
    InvalidTurn {
        turn: usize,
        #[source]
        source: RuleError,
    },
    #[error("replay final hash mismatch")]
    HashMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    #[error(transparent)]
    Setup(#[from] RuleError),
}

#[cfg(test)]
mod tests {
    use crate::{MarketSlot, Rotation, ScoringCards};

    use super::*;

    #[test]
    fn sealed_replay_reconstructs_identical_state() {
        let config = GameConfig::research_aaaaa(2).unwrap();
        let seed = GameSeed::from_u64(42);
        let initial = GameState::new(config, seed).unwrap();
        let action = TurnAction::paired(
            MarketSlot::ZERO,
            initial.boards()[0].frontier()[0],
            Rotation::ZERO,
        );
        let mut replay = Replay::new(config, seed);
        replay.turns.push(action);
        let sealed_hash = replay.seal().unwrap();

        assert_eq!(
            replay.play().unwrap().canonical_hash().as_bytes(),
            &sealed_hash
        );

        let json = serde_json::to_string(&replay).unwrap();
        let decoded: Replay = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, replay);
    }

    #[test]
    fn tampered_replay_is_detected() {
        let config = GameConfig::solo(ScoringCards::AAAAA);
        let seed = GameSeed::from_u64(7);
        let initial = GameState::new(config, seed).unwrap();
        let mut replay = Replay::new(config, seed);
        replay.turns.push(TurnAction::paired(
            MarketSlot::ZERO,
            initial.boards()[0].frontier()[0],
            Rotation::ZERO,
        ));
        replay.seal().unwrap();
        replay.final_state_hash = Some([0; 32]);

        assert!(matches!(
            replay.play(),
            Err(ReplayError::HashMismatch { .. })
        ));
    }
}
