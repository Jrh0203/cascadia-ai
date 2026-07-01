//! Deterministic, transactional Cascadia rules for v2.
//!
//! This crate intentionally has no dependency on the v1 implementation. Rules,
//! simulation, AI, and presentation build on this single canonical state model.

mod board;
mod game;
mod hex;
mod market;
mod replay;
mod scoring;
mod symmetry;
mod tile_catalog;
mod types;

pub use board::{Board, BoardDelta, BoardError, HabitatAnalysis, MAX_BOARD_TILES, PlacedTile};
pub use game::{
    BoardRestoreAudit, BoardUndoAudit, DraftChoice, GameConfig, GameMode, GameSeed, GameState,
    MarketDecision, MarketDecisionSession, MarketDecisionStage, MarketDecisionTransition,
    MarketPrelude, PUBLIC_MARKET_ACTION_WIRE_SIZE, PUBLIC_MARKET_ACTION_WIRE_VERSION,
    PublicGameState, PublicSupply, RuleError, TilePlacement, TurnAction, WildlifeWipe,
    public_market_action_identity, public_market_decision_identity,
    public_market_replacement_is_universally_safe, public_market_universally_safe_wipe_masks,
};
pub use hex::{GRID_DIM, GRID_RADIUS, GRID_SIZE, HexCoord, HexDirection};
pub use market::Market;
pub use replay::{Replay, ReplayError};
pub use scoring::{
    ScoreBreakdown, rescore_after_placement, rescore_after_placement_with_habitat_analysis,
    rescore_after_tile_with_habitat_analysis, rescore_after_wildlife_placement,
    rescore_with_wildlife_scores, score_board, score_game, score_wildlife_type,
};
pub use symmetry::{
    D6_CONTRACT_SCHEMA_VERSION, D6ContractMetadata, D6Error, D6Transform, D6TransformMetadata,
    LegalActionPermutation, d6_contract_metadata,
};
pub use tile_catalog::{STANDARD_TILES, STARTER_CLUSTERS, StarterPlacement};
pub use types::{
    MarketSlot, Rotation, ScoringCards, ScoringVariant, Terrain, Tile, TileId, Wildlife,
    WildlifeMask,
};
