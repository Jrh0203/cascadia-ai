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
mod tile_catalog;
mod types;

pub use board::{Board, BoardDelta, BoardError, HabitatAnalysis, MAX_BOARD_TILES, PlacedTile};
pub use game::{
    DraftChoice, GameConfig, GameMode, GameSeed, GameState, MarketPrelude, PublicGameState,
    PublicSupply, RuleError, TilePlacement, TurnAction, WildlifeWipe,
};
pub use hex::{GRID_DIM, GRID_RADIUS, GRID_SIZE, HexCoord};
pub use market::Market;
pub use replay::{Replay, ReplayError};
pub use scoring::{
    ScoreBreakdown, rescore_after_placement, rescore_after_placement_with_habitat_analysis,
    rescore_after_tile_with_habitat_analysis, rescore_after_wildlife_placement,
    rescore_with_wildlife_scores, score_board, score_game,
};
pub use tile_catalog::{STANDARD_TILES, STARTER_CLUSTERS, StarterPlacement};
pub use types::{
    MarketSlot, Rotation, ScoringCards, ScoringVariant, Terrain, Tile, TileId, Wildlife,
    WildlifeMask,
};
