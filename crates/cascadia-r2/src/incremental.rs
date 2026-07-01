//! Exact sibling-action materialization over one mutable active board.
//!
//! The canonical game enumerator already applies and undoes each placement on
//! one board.  This module is the production R6 boundary: it materializes the
//! changed board once, reuses the three unchanged relative-board tensor
//! slices, and derives only the selected market/player/global metadata.

use cascadia_data::PositionRecord;
use cascadia_game::{
    Board, D6Transform, DraftChoice, GameState, HexCoord, Market, TurnAction, Wildlife,
};

use crate::{
    R2_MAP_BOARD_TOKEN_CAPACITY, R2_MAP_TOKEN_FEATURES, R2Error, R2MapPublicTensors, Result,
    TOKEN_PAYLOAD_WIDTH, encode_global_features, encode_market_features, encode_player_features,
    encode_r2_map_public_tensors,
    mlx_export::{EncodedBoardTokens, encode_sparse_board_tokens, encode_wildlife_sibling_tokens},
    model::SparseBoardState,
};

/// Board-local result captured while the canonical legal-action enumerator
/// has one sibling applied.  Its fields are private so consumers cannot
/// construct a partial or inconsistent afterstate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct R2MapActiveBoardDelta {
    encoded: EncodedBoardTokens,
    occupied_count: u8,
    nature_tokens: u8,
    wildlife_counts: [u8; 5],
    largest_habitats: [u8; 5],
}

/// Reusable spatial context for all optional-wildlife siblings of one tile
/// placement. Habitat components are identical across those siblings.
#[derive(Debug, Clone)]
pub struct R2MapTileBoardContext {
    sparse: SparseBoardState,
    without_wildlife: R2MapActiveBoardDelta,
}

/// Reusable exact materializer for all draft siblings of one public parent.
#[derive(Debug, Clone)]
pub struct R2MapIncrementalMaterializer {
    parent_record: PositionRecord,
    parent_market: Market,
    parent_tensors: R2MapPublicTensors,
    transform: D6Transform,
}

impl R2MapIncrementalMaterializer {
    pub fn new(game: &GameState, game_index: u64, transform: D6Transform) -> Result<Self> {
        let perspective_seat = game.current_player();
        let public = game.public_state();
        let parent_record =
            PositionRecord::observe_public_for_seat(&public, game_index, perspective_seat);
        let parent_tensors =
            encode_r2_map_public_tensors(&public, game_index, perspective_seat, transform, false)?;
        Ok(Self {
            parent_record,
            parent_market: game.market().clone(),
            parent_tensors,
            transform,
        })
    }

    pub fn parent_tensors(&self) -> &R2MapPublicTensors {
        &self.parent_tensors
    }

    /// Capture the exact active-board delta while the game enumerator has the
    /// candidate applied. Opponent boards and public nonspatial state are not
    /// revisited here.
    pub fn capture_active_board(&self, board: &Board) -> Result<R2MapActiveBoardDelta> {
        let sparse = SparseBoardState::from_board(0, board)?;
        self.capture_sparse_board(board, sparse)
    }

    pub fn capture_tile_board(&self, board: &Board) -> Result<R2MapTileBoardContext> {
        let sparse = SparseBoardState::from_board(0, board)?;
        let without_wildlife = self.capture_sparse_board(board, sparse.clone())?;
        Ok(R2MapTileBoardContext {
            sparse,
            without_wildlife,
        })
    }

    pub fn capture_wildlife_sibling(
        &self,
        board: &Board,
        tile_context: &R2MapTileBoardContext,
        placed_wildlife: Option<(Wildlife, HexCoord)>,
    ) -> Result<R2MapActiveBoardDelta> {
        let Some((wildlife, coord)) = placed_wildlife else {
            return Ok(tile_context.without_wildlife.clone());
        };
        let sparse =
            SparseBoardState::from_wildlife_placement(&tile_context.sparse, coord, wildlife)?;
        let encoded = encode_wildlife_sibling_tokens(
            &tile_context.sparse,
            &tile_context.without_wildlife.encoded,
            &sparse,
            self.transform,
        )?;
        Self::capture_encoded_board(board, sparse, encoded)
    }

    fn capture_sparse_board(
        &self,
        board: &Board,
        sparse: SparseBoardState,
    ) -> Result<R2MapActiveBoardDelta> {
        let encoded = encode_sparse_board_tokens(&sparse, self.transform)?;
        Self::capture_encoded_board(board, sparse, encoded)
    }

    fn capture_encoded_board(
        board: &Board,
        sparse: SparseBoardState,
        encoded: EncodedBoardTokens,
    ) -> Result<R2MapActiveBoardDelta> {
        let occupied_count = u8::try_from(sparse.occupied_tiles.len())
            .map_err(|_| R2Error::DatasetContract("active board count exceeds u8".to_owned()))?;
        let wildlife_counts = sparse.wildlife_counts();
        let largest_habitats = sparse.largest_habitats();
        Ok(R2MapActiveBoardDelta {
            encoded,
            occupied_count,
            nature_tokens: board.nature_tokens(),
            wildlife_counts,
            largest_habitats,
        })
    }

    /// Combine one captured active-board delta with the action's public market
    /// effect. The result is byte-for-byte tensor-equivalent to observing the
    /// canonical selected public afterstate and contains no hidden refill.
    pub fn materialize_afterstate(
        &self,
        delta: &R2MapActiveBoardDelta,
        action: &TurnAction,
    ) -> Result<R2MapPublicTensors> {
        if action.replace_three_of_a_kind || !action.wildlife_wipes.is_empty() {
            return Err(R2Error::DatasetContract(
                "incremental draft action must belong to the staged post-prelude game".to_owned(),
            ));
        }
        let mut record = self.parent_record.clone();
        record.turn = record
            .turn
            .checked_add(1)
            .ok_or_else(|| R2Error::DatasetContract("turn index overflow".to_owned()))?;
        record.board_counts[0] = delta.occupied_count;
        record.nature_tokens[0] = delta.nature_tokens;
        record.wildlife_counts[0] = delta.wildlife_counts;
        record.habitat_sizes[0] = delta.largest_habitats;

        let mut market = self.parent_market.clone();
        match action.draft {
            DraftChoice::Paired { slot } => {
                market.take_paired(slot).ok_or_else(|| {
                    R2Error::DatasetContract("paired draft disappeared".to_owned())
                })?;
            }
            DraftChoice::Independent {
                tile_slot,
                wildlife_slot,
            } => {
                market
                    .take_independent(tile_slot, wildlife_slot)
                    .ok_or_else(|| {
                        R2Error::DatasetContract("independent draft disappeared".to_owned())
                    })?;
            }
        }
        record.overwrite_market(&market);

        let active_feature_width = R2_MAP_BOARD_TOKEN_CAPACITY * R2_MAP_TOKEN_FEATURES;
        // The active-board prefix is replaced wholesale. Build that prefix as
        // zeroes and copy only the three unchanged opponent slices, avoiding
        // the previous clone-then-clear write amplification while preserving
        // the exact fixed-capacity tensor schema.
        let mut token_features = Vec::with_capacity(self.parent_tensors.token_features.len());
        token_features.resize(active_feature_width, 0.0);
        token_features
            .extend_from_slice(&self.parent_tensors.token_features[active_feature_width..]);
        let mut token_types = Vec::with_capacity(self.parent_tensors.token_types.len());
        token_types.resize(R2_MAP_BOARD_TOKEN_CAPACITY, 0);
        token_types
            .extend_from_slice(&self.parent_tensors.token_types[R2_MAP_BOARD_TOKEN_CAPACITY..]);
        let mut token_mask = Vec::with_capacity(self.parent_tensors.token_mask.len());
        token_mask.resize(R2_MAP_BOARD_TOKEN_CAPACITY, 0);
        token_mask
            .extend_from_slice(&self.parent_tensors.token_mask[R2_MAP_BOARD_TOKEN_CAPACITY..]);
        for slot in 0..R2_MAP_BOARD_TOKEN_CAPACITY {
            let token_type = delta.encoded.token_types[slot];
            if token_type == 0 {
                continue;
            }
            if !(1..=4).contains(&token_type) {
                return Err(R2Error::DatasetContract(
                    "incremental token type is outside [1, 4]".to_owned(),
                ));
            }
            token_types[slot] = i32::from(token_type);
            token_mask[slot] = 1;
            let feature = slot * R2_MAP_TOKEN_FEATURES;
            token_features[feature + usize::from(token_type - 1)] = 1.0;
            token_features[feature + 4] = 1.0;
            let payload = slot * TOKEN_PAYLOAD_WIDTH;
            for index in 0..TOKEN_PAYLOAD_WIDTH {
                token_features[feature + 8 + index] =
                    f32::from(delta.encoded.token_payload[payload + index]) / 64.0;
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
}
