//! Exact focal-nearfield plus far-topology representations for R4.
//!
//! The canonical payload retains every public occupied entity through a
//! local-index or exact-coordinate overflow row. Habitat, wildlife, and
//! frontier topology are deterministic model-visible projections and never
//! participate in authoritative reconstruction.

mod adversarial;
mod bounded;
mod bounded_adversarial;
mod bounded_census;
mod census;
mod codec;
mod model;
mod parent_mlx;

pub use adversarial::{
    ADVERSARIAL_PARITY_SCHEMA, ADVERSARIAL_SCHEMA, AdversarialAblationComparison,
    AdversarialCaseResult, AdversarialFixtureId, AdversarialParityReport,
    AdversarialParityScientific, AdversarialReport, AdversarialScientific,
    adversarial_fixture_pair, compare_adversarial_reports, evaluate_adversarial_fixture,
    run_adversarial_suite, validate_adversarial_report,
};
pub use bounded::{
    BOUNDED_ACTIVE_SCALAR_LIMIT, BOUNDED_BYTE_LIMIT, BOUNDED_MAGIC, BOUNDED_MAX_TOKENS,
    BOUNDED_P99_TOKEN_LIMIT, BOUNDED_PADDED_SCALAR_LIMIT, BOUNDED_SCHEMA_VERSION,
    BOUNDED_THROUGHPUT_RATIO_MINIMUM, BoundedAccounting, BoundedArm, BoundedFeatureView,
    BoundedGlobalView, BoundedMarketView, BoundedPlayerView, BoundedToken, BoundedTokenKind,
};
pub use bounded_adversarial::{
    BOUNDED_ADVERSARIAL_PARITY_SCHEMA, BOUNDED_ADVERSARIAL_SCHEMA, BoundedAdversarialArmResult,
    BoundedAdversarialCase, BoundedAdversarialParityReport, BoundedAdversarialParityScientific,
    BoundedAdversarialReport, BoundedAdversarialScientific, compare_bounded_adversarial_reports,
    run_bounded_adversarial_suite, validate_bounded_adversarial_parity_report,
    validate_bounded_adversarial_report,
};
pub use bounded_census::{
    BOUNDED_AGGREGATE_SCHEMA, BOUNDED_EXPERIMENT_ID, BOUNDED_ORDER_PROOF_SCHEMA,
    BOUNDED_REPORT_SCHEMA, BoundedAggregateReport, BoundedAggregateScientific, BoundedArmAggregate,
    BoundedArmClassification, BoundedArmGateAssessment, BoundedArmOperational, BoundedArmReport,
    BoundedArmScientific, BoundedClassification, BoundedOrderProofReport,
    BoundedOrderProofScientific, BoundedPromotionAssessment, aggregate_bounded_reports,
    aggregate_bounded_reports_with_order_proof, census_bounded_arm,
};
pub use census::{
    AGGREGATE_SCHEMA, AggregateReport, DatasetIdentity, DistributionSummary, EXPERIMENT_ID,
    Histogram, ORDER_PROOF_SCHEMA, OrderProofReport, OrderProofScientific, R4Classification,
    REPORT_SCHEMA, RadiusAggregate, ShardReport, aggregate_reports,
    aggregate_reports_with_order_proof, census_datasets, write_json_atomic,
};
pub use codec::{PACKED_MAGIC, PACKED_SCHEMA_VERSION};
pub use model::{
    ABLATIONS, AdaptiveBoard, AdaptiveFeatureView, AdaptiveMultiResolutionState, FarFrontierBucket,
    FarHabitatComponent, FarWildlifeComponent, FarWildlifeMotifBucket, FeatureAblation,
    FeatureBlockSet, IndexedOccupiedTile, NearCell, NearCellState, NearFieldRadius, RadiusId,
    centered_hex_capacity, deterministic_integer_center, hex_disk_coord, hex_disk_index,
};
pub use parent_mlx::{
    BOUNDED_PARENT_ADR_ID, BOUNDED_PARENT_ARMS, BOUNDED_PARENT_CACHE_SCHEMA,
    BOUNDED_PARENT_CACHE_SCHEMA_VERSION, BOUNDED_PARENT_EXPERIMENT_ID, BOUNDED_PARENT_PROTOCOL_ID,
    UNIVERSAL_PARENT_CLASS_COUNT, UNIVERSAL_PARENT_VALUE_WIDTH, bounded_parent_token_owner,
    bounded_token_universal_class,
};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum R4Error {
    #[error("invalid R4 state: {0}")]
    InvalidState(String),
    #[error("invalid R4 feature view: {0}")]
    InvalidFeatureView(String),
    #[error("invalid bounded quotient view: {0}")]
    InvalidBoundedView(String),
    #[error("invalid bounded quotient envelope: {0}")]
    InvalidBoundedEnvelope(String),
    #[error("invalid packed R4 magic")]
    InvalidPackedMagic,
    #[error("unsupported packed R4 schema {0}")]
    UnsupportedPackedSchema(u16),
    #[error("unsupported packed R4 flags 0x{0:04x}")]
    UnsupportedPackedFlags(u16),
    #[error("invalid radius code {0}")]
    InvalidRadiusCode(u8),
    #[error("packed R4 state ended unexpectedly")]
    UnexpectedPackedEnd,
    #[error("packed R4 state contains a noncanonical variable-length integer")]
    NonCanonicalVarint,
    #[error("packed R4 state contains an out-of-range variable-length integer")]
    VarintOverflow,
    #[error("packed R4 state has {0} trailing bytes")]
    TrailingPackedBytes(usize),
    #[error("dataset contract failed: {0}")]
    DatasetContract(String),
    #[error("aggregate contract failed: {0}")]
    AggregateContract(String),
    #[error(transparent)]
    R2(#[from] r2_sparse_entity_census::R2Error),
    #[error(transparent)]
    Data(#[from] cascadia_data::DataError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, R4Error>;
