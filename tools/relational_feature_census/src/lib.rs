//! Exact relational representation experiments for Cascadia v2.
//!
//! This crate shares one deterministic AAAAA corpus runner across the R5
//! quotient, R6 incremental accumulator, S3 component/motif, S5
//! opportunity-derivative, and S6 topology/spectral foundations. Each lane
//! emits a self-contained scientific report and never treats a learned or
//! hand-weighted heuristic as ground truth.

mod common;
mod graph;
mod mlx_substrate;
mod r5;
mod r6;
mod s3;
mod s5;
mod s6;

pub use common::{
    CommonConfig, DistributionSummary, ExperimentLane, ExperimentReport, ReportEnvelope,
    write_json_atomic,
};
pub use graph::{
    BoardGraph, CardAScoreAnatomy, HabitatComponentGraph, RelationalStateGraph,
    WildlifeOpportunitySummary,
};
pub use mlx_substrate::{
    BEAR_COMPONENT_CLASS, ELK_LINE_CLASS, FOX_CENTER_CLASS, FRONTIER_SUMMARY_CLASS,
    HABITAT_COMPONENT_CLASS, HAWK_POSITION_CLASS, OPPORTUNITY_SUMMARY_CLASS,
    R5_MINIMAL_CLASS_COUNT, RELATIONAL_TOKEN_CLASS_COUNT, RELATIONAL_TOKEN_VALUE_WIDTH,
    RelationalParentToken, SALMON_COMPONENT_CLASS, r5_minimal_token, rich_relational_parent_tokens,
};
pub use r5::{R5Metrics, run_r5};
pub use r6::{IncrementalSparseAccumulator, R6Metrics, run_r6};
pub use s3::{S3Metrics, run_s3};
pub use s5::{
    OpportunityDerivative, OpportunityDerivativeContext, S5Metrics,
    opportunity_derivative_features, opportunity_derivative_values, run_s5,
};
pub use s6::{
    GraphChannelEncoding, MarkedPathSummary, S6Metrics, S6StateEncoding, TopologicalBoardEncoding,
    run_s6, topological_state_encoding,
};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum RelationalError {
    #[error("relational feature invariant failed: {0}")]
    Invariant(String),
    #[error(transparent)]
    Rule(#[from] cascadia_game::RuleError),
    #[error(transparent)]
    R2(#[from] r2_sparse_entity_census::R2Error),
    #[error(transparent)]
    R3(#[from] r3_action_edit_census::R3Error),
    #[error(transparent)]
    Data(#[from] cascadia_data::DataError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Postcard(#[from] postcard::Error),
    #[error(transparent)]
    IntegerConversion(#[from] std::num::TryFromIntError),
}

pub type Result<T> = std::result::Result<T, RelationalError>;

pub(crate) fn invalid(message: impl Into<String>) -> RelationalError {
    RelationalError::Invariant(message.into())
}
