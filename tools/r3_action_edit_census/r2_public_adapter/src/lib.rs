//! Compatibility adapter for ADR 0145 public-token semantics.
//!
//! The authoritative implementation now lives in the reusable workspace
//! `cascadia-r2` crate.  Re-exporting it preserves the historical dependency
//! name without compiling a second copy of the encoder.

pub use cascadia_r2::{
    AxialCoord, FrontierHabitatTouch, FrontierToken, GlobalMetadata, HabitatComponentToken,
    HabitatMerge, MarketToken, OccupiedTileToken, PACKED_MAGIC, PACKED_SCHEMA_VERSION,
    PlayerMetadata, R2Error, RotationCompatibility, SparsePublicState, SuppliedTile,
    SuppliedTileCompatibility, WildlifeMotifToken,
};

pub type Result<T> = std::result::Result<T, R2Error>;
