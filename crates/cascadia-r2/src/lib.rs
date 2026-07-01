//! Exact sparse occupied-plus-frontier tokenization for Cascadia public states.
//!
//! The authoritative payload is public metadata plus one exact entity for
//! every occupied tile. Frontier, habitat-component, and wildlife-motif
//! layers are deterministic projections that are regenerated after decoding.
//! Terminal targets are never serialized or hashed.

mod census;
mod codec;
mod incremental;
mod mlx_export;
mod model;
mod r2_map_dataset;
mod r2_map_runtime;

pub use census::{
    ACCEPTED_R0_CORPUS_ROWS, CensusReport, CorpusRequirement, DatasetIdentity, DistributionSummary,
    R2PromotionAssessment, ScientificCensus, census_datasets, read_record_at_ordinal,
    write_json_atomic,
};
pub use codec::{PACKED_MAGIC, PACKED_SCHEMA_VERSION};
pub use incremental::{R2MapActiveBoardDelta, R2MapIncrementalMaterializer, R2MapTileBoardContext};
pub use mlx_export::{
    BOARD_OWNERSHIP_ENCODING, BOARD_SLOTS, BOARD_TOKEN_CAPACITY,
    FOUNDATION_PER_BOARD_MAX_ACTIVE_TOKENS, FOUNDATION_PER_BOARD_P99_ACTIVE_TOKENS,
    GLOBAL_FEATURES, GRAPH_MAX_DEGREE, GRAPH_RELATION_COUNT, MARKET_FEATURES, MLX_CACHE_SCHEMA,
    MLX_CACHE_SCHEMA_VERSION, MLX_CORPUS_LOCK_CONTRACT, MLX_CORPUS_LOCK_SCHEMA_VERSION,
    MLX_EXPERIMENT_ID, MlxCompactEncodedState, MlxCorpusDatasetIdentity, MlxCorpusLock,
    MlxCorpusLockIdentity, MlxEncodedState, MlxExportReceipt, PLAYER_FEATURES,
    R2_MAP_BOARD_TOKEN_CAPACITY, R2_MAP_MAX_LEGAL_FRONTIER_TOKENS,
    R2_MAP_MAX_LEGAL_HABITAT_COMPONENT_TOKENS, R2_MAP_MAX_LEGAL_WILDLIFE_MOTIF_TOKENS,
    R2_MAP_TOKEN_CAPACITY, TOKEN_CAPACITY, TOKEN_PAYLOAD_WIDTH, compact_encoded_state,
    encode_global_features, encode_market_features, encode_player_features, encode_sparse_state,
    export_mlx_cache, transform_encoded_state,
};
pub use model::{
    AxialCoord, FrontierHabitatTouch, FrontierToken, GlobalMetadata, HabitatComponentToken,
    HabitatMerge, MarketToken, OccupiedTileToken, PlayerMetadata, RotationCompatibility,
    SparsePublicState, SuppliedTile, SuppliedTileCompatibility, WildlifeMotifToken,
};
pub use r2_map_dataset::{
    R2_MAP_DATASET_ACTION_BYTES, R2_MAP_DATASET_DRAFT_FRAME_KIND, R2_MAP_DATASET_FRAME_HEADER_SIZE,
    R2_MAP_DATASET_FRAME_PREFIX_SIZE, R2_MAP_DATASET_FRAME_VERSION, R2_MAP_DATASET_HEADER_SIZE,
    R2_MAP_DATASET_MAGIC, R2_MAP_DATASET_MARKET_FIXED_SIZE, R2_MAP_DATASET_MARKET_FRAME_KIND,
    R2_MAP_DATASET_OPPONENT_TARGET_SIZE, R2_MAP_DATASET_OPPONENT_WIPE_MAX,
    R2_MAP_DATASET_PROTOCOL_ID, R2_MAP_DATASET_SCHEMA_VERSION, R2_MAP_DRAFT_IMITATION_SUBSET_ID,
    R2_MAP_DRAFT_IMITATION_SUBSET_PARTS_PER_MILLION, R2MapCompactIndexGameMetadata,
    R2MapCompactIndexMetadata, R2MapDatasetManifest, R2MapDatasetMode, R2MapDatasetRoundIdentity,
    R2MapDatasetSource, R2MapDatasetStreamConfig, R2MapDatasetStreamReceipt,
    R2MapPackedBatchProducerConfig, build_r2_map_compact_index_metadata,
    build_r2_map_dataset_manifest, draft_is_imitation_subset, game_is_validation,
    serve_r2_map_packed_batches, stream_r2_map_dataset,
    stream_r2_map_dataset_after_semantic_validation,
};
pub use r2_map_runtime::{
    R2_MAP_ACTION_BYTES, R2_MAP_MARKET_ACTION_BYTES, R2_MAP_MARKET_ACTION_SCHEMA_BLAKE3,
    R2_MAP_TOKEN_FEATURES, R2MapActionEncoder, R2MapMarketActionKind, R2MapMarketDecisionKind,
    R2MapPublicTensors, encode_r2_map_action_bytes, encode_r2_map_market_action_bytes,
    encode_r2_map_public_tensors,
};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum R2Error {
    #[error("invalid record: {0}")]
    InvalidRecord(String),
    #[error("invalid occupied tile for relative seat {seat} at row {row}: {reason}")]
    InvalidOccupiedTile {
        seat: u8,
        row: usize,
        reason: String,
    },
    #[error("duplicate occupied coordinate {coord:?} for relative seat {seat}")]
    DuplicateCoordinate { seat: u8, coord: AxialCoord },
    #[error("occupied rows for relative seat {seat} are not in canonical coordinate order")]
    NonCanonicalOccupiedOrder { seat: u8 },
    #[error("relative seat {seat} board is disconnected")]
    DisconnectedBoard { seat: u8 },
    #[error("frontier oracle mismatch for relative seat {seat}")]
    FrontierOracleMismatch { seat: u8 },
    #[error("habitat-component oracle mismatch for relative seat {seat}")]
    HabitatOracleMismatch { seat: u8 },
    #[error("wildlife-motif projection mismatch for relative seat {seat}")]
    WildlifeMotifMismatch { seat: u8 },
    #[error("D6 transform {transform_id} cannot represent coordinate {coord:?}: {reason}")]
    D6Coordinate {
        transform_id: u8,
        coord: AxialCoord,
        reason: String,
    },
    #[error("packed representation has invalid magic")]
    InvalidPackedMagic,
    #[error("packed representation schema {0} is unsupported")]
    UnsupportedPackedSchema(u16),
    #[error("packed representation flags contain unsupported bits: 0x{0:04x}")]
    UnsupportedPackedFlags(u16),
    #[error("packed representation ended unexpectedly")]
    UnexpectedPackedEnd,
    #[error("packed representation contains a noncanonical variable-length integer")]
    NonCanonicalVarint,
    #[error("packed representation contains an out-of-range variable-length integer")]
    VarintOverflow,
    #[error("packed representation has {0} trailing bytes")]
    TrailingPackedBytes(usize),
    #[error("decoded representation is not canonical: {0}")]
    NonCanonicalPacked(String),
    #[error("dataset contract failed: {0}")]
    DatasetContract(String),
    #[error("record ordinal {ordinal} is outside the loaded corpus of {total} rows")]
    OrdinalOutOfRange { ordinal: usize, total: usize },
    #[error(transparent)]
    Data(#[from] cascadia_data::DataError),
    #[error(transparent)]
    Experience(#[from] cascadia_data::R2MapExperienceError),
    #[error(transparent)]
    Rule(#[from] cascadia_game::RuleError),
    #[error(transparent)]
    D6(#[from] cascadia_game::D6Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, R2Error>;
