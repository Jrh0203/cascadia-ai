//! Cascadia V3 radius-7 Stockfish-style NNUE substrate.
//!
//! This crate deliberately has no compatibility path for historical NNUE
//! feature indices or weights.  It consumes only the canonical public rules
//! state and the exact public opportunity graph.

mod accumulator;
mod campaign;
mod features;
mod model;
mod schema;
mod search;
mod teacher;
mod training;

pub use accumulator::{AccumulatorUndo, PreparedOwnAccumulator, V3AccumulatorStack};
pub use campaign::{V3_CAMPAIGN_ID, V3_CAMPAIGN_STATE_SCHEMA_ID, V3CampaignState};
pub use features::{
    ActiveFeature, BoardFeatureEncoding, FullOpportunitiesCatalog, OpportunityFeatureSpec,
    OpportunityTrainingFactorSpec, OverflowEntity, PreparedOpportunityEvaluation, V3FeatureContext,
    V3FeatureSet, V3OwnFeatureSet, encode_board_features, encode_public_features,
    transform_feature_set, transformed_overflow,
};
pub use model::{
    InferenceBackend, QuantizedEvaluation, QuantizedEvaluationTrace, QuantizedV3Model,
    V3ModelManifest, V3ModelScales,
};
pub use schema::{
    BASE_FEATURE_ROWS, CORE_SPATIAL_FEATURE_ROWS, GLOBAL_FEATURE_ROWS, HOT_CELL_COUNT, HOT_RADIUS,
    OPPORTUNITY_FEATURE_MAX, OPPORTUNITY_FEATURE_MIN, OVERFLOW_COORD_MAX, OVERFLOW_COORD_MIN,
    OVERFLOW_SLOT_COUNT, V3_FEATURE_SCHEMA_ID, V3FeatureSchemaManifest, hot_coord, hot_index,
};
pub use search::{
    RankedV3Action, TerminalRolloutConfig, V3RankProfile, V3SearchBudget, V3SearchPolicy,
    V3TeacherCandidateEstimate, V3TeacherRootLabel, select_boltzmann_top32,
};
pub use teacher::{
    LABELED_TEACHER_ROOT_SHARD_MAGIC, TEACHER_ROOT_SHARD_MAGIC, V3LabeledTeacherRoot,
    V3LabeledTeacherRootShardReader, V3LabeledTeacherRootShardWriter, V3TeacherRoot,
    V3TeacherRootShardReader, V3TeacherRootShardWriter, V3TeacherSplit, V3TeacherStratum,
    labeled_root_training_entries,
};
pub use training::{
    GAME_SHARD_MAGIC, TRAINING_SHARD_MAGIC, V3GameRecord, V3GameShardReader, V3GameShardWriter,
    V3TrainingEntry, V3TrainingProvenance, V3TrainingShardReader, V3TrainingShardWriter,
    signed_score_to_go,
};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum V3Error {
    #[error("invalid V3 feature state: {0}")]
    InvalidFeature(String),
    #[error("invalid V3 model: {0}")]
    InvalidModel(String),
    #[error("invalid V3 training data: {0}")]
    InvalidTraining(String),
    #[error("V3 integer accumulator overflow")]
    AccumulatorOverflow,
    #[error("V3 artifact checksum mismatch for {0}")]
    ChecksumMismatch(String),
    #[error(transparent)]
    Rules(#[from] cascadia_game::RuleError),
    #[error(transparent)]
    Simulation(#[from] cascadia_sim::SimulationError),
    #[error(transparent)]
    Opportunity(#[from] cascadia_data::OpportunityGraphError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Postcard(#[from] postcard::Error),
}

pub type Result<T> = std::result::Result<T, V3Error>;
