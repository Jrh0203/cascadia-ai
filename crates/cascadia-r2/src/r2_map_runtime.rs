//! Direct in-memory exact-R2 materialization for grouped local inference.

use cascadia_data::{
    GRADED_ORACLE_ACTION_FEATURE_SIZE, GradedOracleActionFeatures, GradedOracleStagedActionContext,
    PositionRecord,
};
use cascadia_game::{
    D6Transform, DraftChoice, GameState, MarketDecision, MarketDecisionStage,
    PUBLIC_MARKET_ACTION_WIRE_SIZE, PublicGameState, ScoreBreakdown, TilePlacement, TurnAction,
};

use crate::{
    BOARD_SLOTS, GLOBAL_FEATURES, R2_MAP_TOKEN_CAPACITY, R2Error, Result, SparsePublicState,
    TOKEN_PAYLOAD_WIDTH, encode_global_features, encode_market_features, encode_player_features,
    mlx_export::encode_r2_map_state_authoritative,
};

pub const R2_MAP_TOKEN_FEATURES: usize = 60;
pub const R2_MAP_ACTION_BYTES: usize = GRADED_ORACLE_ACTION_FEATURE_SIZE;
pub const R2_MAP_MARKET_ACTION_BYTES: usize = PUBLIC_MARKET_ACTION_WIRE_SIZE;
pub const R2_MAP_MARKET_ACTION_SCHEMA_BLAKE3: &str =
    "e9ab2382f20f4ea440591adba6021d85f8be83ff4e483513b05d46e5f285cd38";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum R2MapMarketDecisionKind {
    FreeThreeOfAKind = 0,
    PaidWipes = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum R2MapMarketActionKind {
    Keep = 0,
    Replace = 1,
    Stop = 2,
    PaidWipe = 3,
}

/// Canonical public market-action wire bytes shared by Rust replay/inference
/// and the MLX v1.1 market-decision head.
///
/// Layout is exactly little-endian `<BBBBI>`: schema, decision kind, action
/// kind, four-slot wipe mask, and a zero reserved u32. No refill outcome,
/// hidden order, seed, host, policy, or split metadata is representable.
pub fn encode_r2_map_market_action_bytes(
    stage: MarketDecisionStage,
    decision: &MarketDecision,
) -> Result<[u8; R2_MAP_MARKET_ACTION_BYTES]> {
    decision
        .public_wire_bytes(stage)
        .map_err(|error| R2Error::DatasetContract(error.to_string()))
}

#[derive(Debug, Clone, PartialEq)]
pub struct R2MapPublicTensors {
    pub token_features: Vec<f32>,
    pub token_types: Vec<i32>,
    pub token_mask: Vec<u8>,
    pub market_features: Vec<f32>,
    pub market_mask: [u8; 4],
    pub player_features: Vec<f32>,
    pub player_mask: [u8; BOARD_SLOTS],
    pub global_features: [f32; GLOBAL_FEATURES],
}

/// Reuses the D6-transformed parent state while encoding every sibling action
/// in one exhaustive decision. The transform is exact and action order is
/// owned by the caller; this cache changes neither legality nor enumeration.
pub struct R2MapActionEncoder<'a> {
    source_game: &'a GameState,
    transformed_game: GameState,
    staged_context: GradedOracleStagedActionContext,
    transform: D6Transform,
}

impl<'a> R2MapActionEncoder<'a> {
    pub fn new(source_game: &'a GameState, transform: D6Transform) -> Result<Self> {
        let transformed_game = source_game.transformed(transform)?;
        let staged_context = GradedOracleStagedActionContext::observe(&transformed_game);
        Ok(Self {
            source_game,
            transformed_game,
            staged_context,
            transform,
        })
    }

    pub fn encode(&self, action: &TurnAction) -> Result<[u8; R2_MAP_ACTION_BYTES]> {
        let transformed_action = action.transformed(self.source_game, self.transform)?;
        Ok(
            GradedOracleActionFeatures::observe(&self.transformed_game, &transformed_action)?
                .to_bytes(),
        )
    }

    /// Exact staged-draft fast path. The score is captured while the canonical
    /// enumerator has the candidate applied, so this avoids cloning, replaying,
    /// and rescoring the same afterstate during dense candidate expansion.
    pub fn encode_staged_after_score(
        &self,
        action: &TurnAction,
        after: ScoreBreakdown,
    ) -> Result<[u8; R2_MAP_ACTION_BYTES]> {
        if action.replace_three_of_a_kind || !action.wildlife_wipes.is_empty() {
            return Err(R2Error::DatasetContract(
                "staged R2-MAP action encoder received an unresolved market prelude".to_owned(),
            ));
        }
        let tile_slot = match action.draft {
            DraftChoice::Paired { slot } => slot,
            DraftChoice::Independent { tile_slot, .. } => tile_slot,
        };
        let tile = self.source_game.market().tiles[tile_slot.index()].ok_or_else(|| {
            R2Error::DatasetContract("staged R2-MAP tile slot is unavailable".to_owned())
        })?;
        let transformed_action = TurnAction {
            replace_three_of_a_kind: false,
            wildlife_wipes: Vec::new(),
            draft: action.draft,
            tile: TilePlacement {
                coord: self.transform.transform_coord(action.tile.coord)?,
                rotation: self
                    .transform
                    .transform_tile_rotation(tile, action.tile.rotation),
            },
            wildlife: action
                .wildlife
                .map(|coord| self.transform.transform_coord(coord))
                .transpose()?,
        };
        Ok(self
            .staged_context
            .encode_after_score(&transformed_action, after)?)
    }
}

pub fn encode_r2_map_public_tensors(
    state: &PublicGameState,
    game_index: u64,
    perspective_seat: usize,
    transform: D6Transform,
    selected_afterstate: bool,
) -> Result<R2MapPublicTensors> {
    let record = PositionRecord::observe_public_for_seat(state, game_index, perspective_seat);
    let sparse = if selected_afterstate {
        SparsePublicState::from_selected_afterstate_record(&record, None)?
    } else {
        SparsePublicState::from_position_record(&record, None)?
    };
    let encoded = encode_r2_map_state_authoritative(&sparse, transform)?;
    let mut token_features = vec![0.0; R2_MAP_TOKEN_CAPACITY * R2_MAP_TOKEN_FEATURES];
    let mut token_types = vec![0; R2_MAP_TOKEN_CAPACITY];
    let mut token_mask = vec![0; R2_MAP_TOKEN_CAPACITY];
    for slot in 0..R2_MAP_TOKEN_CAPACITY {
        let token_type = encoded.token_types[slot];
        if token_type == 0 {
            continue;
        }
        if !(1..=4).contains(&token_type) {
            return Err(R2Error::DatasetContract(
                "R2-MAP token type is outside the accepted schema".to_owned(),
            ));
        }
        token_types[slot] = i32::from(token_type);
        token_mask[slot] = 1;
        let feature = slot * R2_MAP_TOKEN_FEATURES;
        token_features[feature + usize::from(token_type - 1)] = 1.0;
        let relative_seat = usize::from(encoded.token_seats[slot]);
        if relative_seat >= BOARD_SLOTS {
            return Err(R2Error::DatasetContract(
                "R2-MAP token seat is outside the accepted schema".to_owned(),
            ));
        }
        token_features[feature + 4 + relative_seat] = 1.0;
        let payload = slot * TOKEN_PAYLOAD_WIDTH;
        for index in 0..TOKEN_PAYLOAD_WIDTH {
            token_features[feature + 8 + index] =
                f32::from(encoded.token_payload[payload + index]) / 64.0;
        }
    }
    let (market_features, market_mask) = encode_market_features(&record)?;
    let (player_features, player_mask) = encode_player_features(&record)?;
    Ok(R2MapPublicTensors {
        token_features,
        token_types,
        token_mask,
        market_features: market_features.to_vec(),
        market_mask,
        player_features: player_features.to_vec(),
        player_mask,
        global_features: encode_global_features(&record)?,
    })
}

pub fn encode_r2_map_action_bytes(
    game: &GameState,
    action: &TurnAction,
    transform: D6Transform,
) -> Result<[u8; R2_MAP_ACTION_BYTES]> {
    R2MapActionEncoder::new(game, transform)?.encode(action)
}

#[cfg(test)]
mod tests {
    use cascadia_data::GradedOracleActionFeatures;
    use cascadia_game::{
        D6Transform, GameConfig, GameSeed, GameState, MarketDecision, MarketDecisionStage,
        MarketSlot, WildlifeWipe,
    };

    use super::*;

    #[test]
    fn canonical_action_bytes_round_trip_under_every_d6_transform() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(0x0052_324d_4150),
        )
        .unwrap();
        let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible().unwrap();
        let action = game.legal_turn_actions(&prelude).unwrap().remove(0);
        for transform in D6Transform::ALL {
            let transformed_game = game.transformed(transform).unwrap();
            let expected = action.transformed(&game, transform).unwrap();
            let bytes = encode_r2_map_action_bytes(&game, &action, transform).unwrap();
            assert_eq!(bytes.len(), R2_MAP_ACTION_BYTES);
            let decoded = GradedOracleActionFeatures::from_bytes(&bytes)
                .to_game_action(&transformed_game)
                .unwrap();
            assert_eq!(decoded, expected);
        }
    }

    #[test]
    fn canonical_market_action_bytes_are_exact_and_reject_stage_drift() {
        assert_eq!(
            encode_r2_map_market_action_bytes(
                MarketDecisionStage::FreeThreeOfAKind,
                &MarketDecision::KeepThreeOfAKind,
            )
            .unwrap(),
            [1, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            encode_r2_map_market_action_bytes(
                MarketDecisionStage::PaidWipes,
                &MarketDecision::PaidWipe(WildlifeWipe {
                    slots: vec![MarketSlot::ZERO, MarketSlot::THREE],
                }),
            )
            .unwrap(),
            [1, 1, 3, 0b1001, 0, 0, 0, 0]
        );
        assert!(
            encode_r2_map_market_action_bytes(
                MarketDecisionStage::PaidWipes,
                &MarketDecision::ReplaceThreeOfAKind,
            )
            .is_err()
        );
    }
}
