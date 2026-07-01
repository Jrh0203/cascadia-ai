//! Exact action-centric local-patch plus global-edit representation.
//!
//! R3 separates one reusable public state trunk from variable-length action
//! edits. Applying an edit reproduces the authoritative public afterstate
//! emitted by `GameState::preview_public_afterstate`; hidden refill order and
//! terminal training targets are outside the schema.

mod aggregate;
mod census;
mod codec;
mod mlx_action;
mod model;
mod source;
mod strict_json;

pub use aggregate::{
    AggregateOrderProof, AggregateOrderProofScientific, AggregateReport, AggregateScientific,
    AggregateShardIdentity, R3_AGGREGATE_ARTIFACT_KIND, R3_AGGREGATE_SCHEMA_VERSION,
    R3_ORDER_PROOF_ARTIFACT_KIND, aggregate_census_files, aggregate_census_reports,
    prove_aggregate_order,
};
pub use census::{
    CensusConfig, CensusReport, CorpusContract, DistributionSummary, PromotionAssessment,
    R3_CENSUS_PROTOCOL_ID, R3_EXPERIMENT_ID, ScientificCensus, run_census, write_json_atomic,
};
pub use codec::{
    ACTION_EDIT_MAGIC, ACTION_EDIT_SCHEMA_VERSION, STATE_TRUNK_MAGIC, STATE_TRUNK_SCHEMA_VERSION,
};
pub use mlx_action::{
    MLX_ACTION_ENCODING_SCHEMA_VERSION, MLX_ACTION_OPERATION_COUNT, MLX_ACTION_TOKEN_PAYLOAD_WIDTH,
    MLX_ACTION_TOKEN_TYPE_COUNT, MlxActionEncoding, MlxActionOperation, MlxActionToken,
    MlxActionTokenType,
};
pub use model::{
    ActionEdit, ActionFactors, AppliedPublicState, AxialCoord, BoardObjectChanges, BoardTileToken,
    CanonicalActionView, CanonicalBoardToken, CanonicalComponentObject, CanonicalFrontierToken,
    CanonicalFrontierTouch, CanonicalGlobalEdit, CanonicalLocalPatch, CanonicalMotifObject,
    ComponentChanges, ComponentObject, CoverageByRadius, DraftFactor, FrontierChanges,
    GlobalObjectReferences, ImmediateScoreDelta, LocalPatchCell, MarketSlotEdit, MarketSlotToken,
    MarketSnapshot, MotifChanges, ObjectUpdate, PlacementEdit, PlayerPublicSummary, PreludeEdit,
    PreparedPublicStateTrunk, PublicStateTrunk, SelectedMarketObjects, SupplyCountDelta,
    SupplyDelta, SupplySnapshot, TileSemantic, TurnAdvance, WildlifeMotifObject,
};
pub use source::{
    RuntimeIdentity, SOURCE_IDENTITY_CONTRACT, SourceFileIdentity, SourceIdentity,
    capture_runtime_identity, capture_runtime_identity_checked, capture_source_identity,
    capture_source_identity_at, source_bundle_roots,
};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum R3Error {
    #[error("R3 invariant failed: {0}")]
    Invariant(String),
    #[error("invalid R3 packed magic")]
    InvalidPackedMagic,
    #[error("unsupported R3 packed schema {0}")]
    UnsupportedPackedSchema(u16),
    #[error("R3 packed input ended unexpectedly")]
    UnexpectedPackedEnd,
    #[error("R3 packed input has {0} trailing bytes")]
    TrailingPackedBytes(usize),
    #[error("R3 packed input is noncanonical: {0}")]
    NonCanonicalPacked(String),
    #[error(transparent)]
    Rule(#[from] cascadia_game::RuleError),
    #[error(transparent)]
    D6(#[from] cascadia_game::D6Error),
    #[error(transparent)]
    Data(#[from] cascadia_data::DataError),
    #[error(transparent)]
    Supply(#[from] cascadia_data::SemanticSupplyError),
    #[error(transparent)]
    R2(#[from] r2_sparse_entity_census::R2Error),
    #[error(transparent)]
    Postcard(#[from] postcard::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    IntegerConversion(#[from] std::num::TryFromIntError),
    #[error(transparent)]
    PathPrefix(#[from] std::path::StripPrefixError),
}

pub type Result<T> = std::result::Result<T, R3Error>;

pub(crate) fn canonical_blake3(value: &impl serde::Serialize) -> Result<String> {
    Ok(blake3::hash(&serde_json::to_vec(value)?)
        .to_hex()
        .to_string())
}
