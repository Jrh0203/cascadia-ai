use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    env,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use blake3::Hasher;
use cascadia_data::{
    DatasetManifest, DatasetSplit, PositionRecord, PositionShardReader, TARGET_DIM,
    validate_dataset,
};
use cascadia_game::{D6Transform, MAX_BOARD_TILES};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{AxialCoord, R2Error, Result, SparsePublicState, model::SparseBoardState};

pub const MLX_CACHE_SCHEMA_VERSION: u16 = 1;
pub const MLX_CACHE_SCHEMA: &str = "r2-sparse-board-local-mlx-cache-v1";
pub const MLX_EXPERIMENT_ID: &str = "r2-sparse-mlx-architecture-tournament-v1";
pub const MLX_CORPUS_LOCK_SCHEMA_VERSION: u16 = 1;
pub const MLX_CORPUS_LOCK_CONTRACT: &str = "r2-sparse-mlx-frozen-corpus-v1";
pub const BOARD_SLOTS: usize = 4;
/// Frozen padding used only by the historical R2 sparse-foundation cache.
/// Live R2-MAP inference has a separately versioned, rules-complete capacity.
pub const BOARD_TOKEN_CAPACITY: usize = 92;
pub const TOKEN_CAPACITY: usize = BOARD_SLOTS * BOARD_TOKEN_CAPACITY;
/// A connected `n`-tile hex board has at most `2n + 4` distinct empty
/// frontier cells. Each occupied tile contributes at most two habitat
/// components, and the three wildlife-free starter tiles leave at most
/// `n - 3` wildlife motifs after `n - 3` legal turns. Therefore the complete
/// live representation is bounded by `n + (2n + 4) + 2n + (n - 3) = 6n + 1`.
pub const R2_MAP_MAX_LEGAL_FRONTIER_TOKENS: usize = 2 * MAX_BOARD_TILES + 4;
pub const R2_MAP_MAX_LEGAL_HABITAT_COMPONENT_TOKENS: usize = 2 * MAX_BOARD_TILES;
pub const R2_MAP_MAX_LEGAL_WILDLIFE_MOTIF_TOKENS: usize = MAX_BOARD_TILES - 3;
pub const R2_MAP_BOARD_TOKEN_CAPACITY: usize = MAX_BOARD_TILES
    + R2_MAP_MAX_LEGAL_FRONTIER_TOKENS
    + R2_MAP_MAX_LEGAL_HABITAT_COMPONENT_TOKENS
    + R2_MAP_MAX_LEGAL_WILDLIFE_MOTIF_TOKENS;
pub const R2_MAP_TOKEN_CAPACITY: usize = BOARD_SLOTS * R2_MAP_BOARD_TOKEN_CAPACITY;
pub const TOKEN_PAYLOAD_WIDTH: usize = 52;
pub const BOARD_OWNERSHIP_ENCODING: &str = "relative-seat-one-hot-4";
pub const FOUNDATION_PER_BOARD_P99_ACTIVE_TOKENS: usize = 83;
pub const FOUNDATION_PER_BOARD_MAX_ACTIVE_TOKENS: usize = 92;
pub const GRAPH_MAX_DEGREE: usize = 24;
pub const GRAPH_RELATION_COUNT: usize = 10;
pub const MARKET_FEATURES: usize = 31;
pub const GLOBAL_FEATURES: usize = 96;
pub const PLAYER_FEATURES: usize = 23;

const NONE_CATEGORY: i16 = 5;
const FOUNDATION_EXPERIMENT_ID: &str = "r2-sparse-occupied-frontier-foundation-v1";
const FOUNDATION_SCIENTIFIC_BLAKE3: &str =
    "186ad8934287ef0a74a166ed00cc9ebe857dcded20faa01a264974e1eb7081e6";
const FOUNDATION_PUBLIC_POSITION_BLAKE3: &str =
    "29836be57c6e0529c06b0b628c455b27f06284fe7a8c333e54024174a7e7f003";
const FOUNDATION_PACKED_STATE_BLAKE3: &str =
    "c181be2126a42b668f500666cccf41573ea079a3f2c34ab7bc3989f690fec789";
const EXPECTED_LAYER_MAXIMA: [usize; 5] = [91, 107, 69, 79, 340];
const EXPECTED_TYPE_TOKEN_TOTALS: [usize; 4] = [3_090_000, 4_155_914, 2_257_600, 2_365_940];
const EXPECTED_ACTIVE_TOKENS: usize = 11_869_454;

const TOKEN_TYPE_OCCUPIED: u8 = 1;
const TOKEN_TYPE_FRONTIER: u8 = 2;
const TOKEN_TYPE_COMPONENT: u8 = 3;
const TOKEN_TYPE_MOTIF: u8 = 4;

/// Graph-free, board-local encoding consumed by the R2-MAP serving path.
/// The offline MLX cache retains its full relational graph; live R2-MAP
/// inference consumes only these canonical token rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EncodedBoardTokens {
    pub(crate) token_types: Vec<u8>,
    pub(crate) token_payload: Vec<i8>,
    pub(crate) type_counts: [u16; 4],
}

/// Variable-length graph-free token block persisted by R2-MAP replay streams.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EncodedR2MapCompactTokens {
    pub(crate) token_types: Vec<u8>,
    pub(crate) token_seats: Vec<u8>,
    pub(crate) token_payload: Vec<i8>,
    pub(crate) board_type_counts: [[u16; 4]; BOARD_SLOTS],
}

const REL_OCCUPIED_NEIGHBOR: u8 = 1;
const REL_OCCUPIED_FRONTIER: u8 = 2;
const REL_OCCUPIED_COMPONENT: u8 = 3;
const REL_OCCUPIED_MOTIF: u8 = 4;
const REL_FRONTIER_OCCUPIED: u8 = 5;
const REL_FRONTIER_COMPONENT: u8 = 6;
const REL_COMPONENT_OCCUPIED: u8 = 7;
const REL_MOTIF_OCCUPIED: u8 = 8;
const REL_MOTIF_NEIGHBOR: u8 = 9;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxCorpusDatasetIdentity {
    pub order: usize,
    pub split: String,
    pub root_name: String,
    pub dataset_id: String,
    pub total_records: usize,
    pub manifest_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxCorpusLockIdentity {
    pub foundation_experiment_id: String,
    pub foundation_scientific_blake3: String,
    pub foundation_public_position_blake3: String,
    pub foundation_packed_state_blake3: String,
    pub feature_schema: String,
    pub target_schema: String,
    pub total_records: usize,
    pub train_records: usize,
    pub validation_records: usize,
    pub layer_maxima: [usize; 5],
    pub type_token_totals: [usize; 4],
    pub active_tokens: usize,
    pub per_board_p99_active_tokens: usize,
    pub per_board_max_active_tokens: usize,
    pub datasets: Vec<MlxCorpusDatasetIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxCorpusLock {
    pub schema_version: u16,
    pub contract_id: String,
    pub lock_id: String,
    pub identity: MlxCorpusLockIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MlxEncodedState {
    pub token_types: Vec<u8>,
    pub token_seats: Vec<u8>,
    pub token_payload: Vec<i8>,
    pub board_type_counts: [[u16; 4]; BOARD_SLOTS],
    pub graph_token_offsets: Vec<u32>,
    pub graph_targets: Vec<u16>,
    pub graph_relations: Vec<u8>,
    pub graph_direction_bits: Vec<u8>,
    pub max_degree: usize,
}

/// Variable-length wire form of the historical sparse-foundation cache.
///
/// Tokens remain board-major and type-major, but the unused tail of every
/// frozen 92-token foundation partition is omitted and graph indices are
/// remapped. Live R2-MAP replay uses [`EncodedR2MapCompactTokens`] instead so
/// its rules-complete 139-token bound is independent of the archived cache.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxCompactEncodedState {
    pub token_types: Vec<u8>,
    pub token_seats: Vec<u8>,
    pub token_payload: Vec<i8>,
    pub board_type_counts: [[u16; 4]; BOARD_SLOTS],
    pub graph_token_offsets: Vec<u32>,
    pub graph_targets: Vec<u16>,
    pub graph_relations: Vec<u8>,
    pub graph_direction_bits: Vec<u8>,
    pub max_degree: usize,
}

impl MlxEncodedState {
    pub fn active_tokens(&self) -> usize {
        self.board_type_counts
            .iter()
            .flatten()
            .map(|value| usize::from(*value))
            .sum()
    }

    pub fn type_counts(&self) -> [usize; 4] {
        std::array::from_fn(|token_type| {
            self.board_type_counts
                .iter()
                .map(|counts| usize::from(counts[token_type]))
                .sum()
        })
    }

    pub fn board_counts(&self) -> [usize; BOARD_SLOTS] {
        self.board_type_counts
            .map(|counts| counts.into_iter().map(usize::from).sum())
    }

    fn update_semantic_hash(&self, hasher: &mut Hasher) {
        update_framed_hash(hasher, &self.token_types);
        update_framed_hash(hasher, &self.token_seats);
        update_framed_hash(
            hasher,
            &self
                .token_payload
                .iter()
                .map(|value| *value as u8)
                .collect::<Vec<_>>(),
        );
        let counts = self
            .board_type_counts
            .iter()
            .flatten()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        update_framed_hash(hasher, &counts);
        let offsets = self
            .graph_token_offsets
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        update_framed_hash(hasher, &offsets);
        let targets = self
            .graph_targets
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        update_framed_hash(hasher, &targets);
        update_framed_hash(hasher, &self.graph_relations);
        update_framed_hash(hasher, &self.graph_direction_bits);
    }
}

/// Remove only fixed-capacity padding from the authoritative encoded state.
/// All active token payload bytes and graph edges are preserved exactly.
pub fn compact_encoded_state(source: &MlxEncodedState) -> Result<MlxCompactEncodedState> {
    compact_encoded_state_with_board_capacity(source, BOARD_TOKEN_CAPACITY)
}

fn compact_encoded_state_with_board_capacity(
    source: &MlxEncodedState,
    board_token_capacity: usize,
) -> Result<MlxCompactEncodedState> {
    validate_encoded_state_with_board_capacity(source, board_token_capacity)?;
    let token_capacity = BOARD_SLOTS
        .checked_mul(board_token_capacity)
        .ok_or_else(|| R2Error::DatasetContract("R2 token capacity overflowed".to_owned()))?;
    let board_counts = source.board_counts();
    let active_tokens: usize = board_counts.iter().sum();
    let mut old_to_new = vec![u16::MAX; token_capacity];
    let mut active_slots = Vec::with_capacity(active_tokens);
    for (board, &count) in board_counts.iter().enumerate() {
        for local in 0..count {
            let old = board * board_token_capacity + local;
            let new = u16::try_from(active_slots.len()).map_err(|_| {
                R2Error::DatasetContract("compact R2 token index exceeds u16".to_owned())
            })?;
            old_to_new[old] = new;
            active_slots.push(old);
        }
    }

    let mut token_types = Vec::with_capacity(active_tokens);
    let mut token_seats = Vec::with_capacity(active_tokens);
    let mut token_payload = Vec::with_capacity(active_tokens * TOKEN_PAYLOAD_WIDTH);
    let mut graph_token_offsets = Vec::with_capacity(active_tokens + 1);
    let mut graph_targets = Vec::new();
    let mut graph_relations = Vec::new();
    let mut graph_direction_bits = Vec::new();
    graph_token_offsets.push(0);
    for old in active_slots {
        token_types.push(source.token_types[old]);
        token_seats.push(source.token_seats[old]);
        let payload = old * TOKEN_PAYLOAD_WIDTH;
        token_payload
            .extend_from_slice(&source.token_payload[payload..payload + TOKEN_PAYLOAD_WIDTH]);
        let edge_start = usize::try_from(source.graph_token_offsets[old])
            .map_err(|_| R2Error::DatasetContract("R2 graph offset exceeds usize".to_owned()))?;
        let edge_end = usize::try_from(source.graph_token_offsets[old + 1])
            .map_err(|_| R2Error::DatasetContract("R2 graph offset exceeds usize".to_owned()))?;
        for edge in edge_start..edge_end {
            let old_target = usize::from(source.graph_targets[edge]);
            let target = old_to_new.get(old_target).copied().unwrap_or(u16::MAX);
            if target == u16::MAX {
                return Err(R2Error::DatasetContract(
                    "active R2 graph edge targets padding".to_owned(),
                ));
            }
            graph_targets.push(target);
            graph_relations.push(source.graph_relations[edge]);
            graph_direction_bits.push(source.graph_direction_bits[edge]);
        }
        graph_token_offsets
            .push(u32::try_from(graph_targets.len()).map_err(|_| {
                R2Error::DatasetContract("compact R2 graph exceeds u32".to_owned())
            })?);
    }
    Ok(MlxCompactEncodedState {
        token_types,
        token_seats,
        token_payload,
        board_type_counts: source.board_type_counts,
        graph_token_offsets,
        graph_targets,
        graph_relations,
        graph_direction_bits,
        max_degree: source.max_degree,
    })
}

#[derive(Debug)]
struct ValidatedDataset {
    root: PathBuf,
    manifest: DatasetManifest,
    manifest_blake3: String,
}

#[derive(Debug, Clone, Serialize)]
struct FileIdentity {
    file: String,
    dtype: String,
    shape: Vec<usize>,
    bytes: u64,
    blake3: String,
}

#[derive(Debug, Default, Clone, Serialize)]
struct SplitIntegrity {
    records: usize,
    active_tokens: usize,
    padding_tokens: usize,
    graph_edges: usize,
    max_active_tokens: usize,
    max_active_tokens_per_board: usize,
    max_graph_degree: usize,
    layer_maxima: [usize; 4],
    type_token_totals: [usize; 4],
}

#[derive(Debug, Clone, Serialize)]
struct SplitManifest {
    records: usize,
    files: BTreeMap<String, FileIdentity>,
    integrity: SplitIntegrity,
}

#[derive(Debug, Serialize)]
struct CacheManifest {
    schema_version: u16,
    cache_schema: &'static str,
    experiment_id: &'static str,
    cache_id: String,
    scientific_identity: Value,
    tensor_contract: Value,
    corpus: Value,
    semantic_integrity: Value,
    splits: BTreeMap<String, SplitManifest>,
    exporter: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct MlxExportReceipt {
    pub schema_version: u16,
    pub experiment_id: &'static str,
    pub cache_id: String,
    pub cache_root: String,
    pub cache_manifest: String,
    pub cache_manifest_blake3: String,
}

#[derive(Debug)]
struct HashedWriter {
    path: PathBuf,
    writer: BufWriter<File>,
    hasher: Hasher,
    bytes: u64,
}

impl HashedWriter {
    fn create(path: PathBuf) -> Result<Self> {
        Ok(Self {
            writer: BufWriter::new(File::create(&path)?),
            path,
            hasher: Hasher::new(),
            bytes: 0,
        })
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.writer.write_all(bytes)?;
        self.hasher.update(bytes);
        self.bytes = self.bytes.checked_add(bytes.len() as u64).ok_or_else(|| {
            R2Error::DatasetContract("cache byte accounting overflowed".to_owned())
        })?;
        Ok(())
    }

    fn finish(mut self, dtype: &str, shape: Vec<usize>) -> Result<FileIdentity> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        let expected = shape
            .iter()
            .try_fold(dtype_width(dtype), |bytes, dimension| {
                bytes.checked_mul(*dimension)
            })
            .ok_or_else(|| R2Error::DatasetContract("tensor byte count overflowed".to_owned()))?
            as u64;
        if self.bytes != expected {
            return Err(R2Error::DatasetContract(format!(
                "{} has {} bytes but shape and dtype require {expected}",
                self.path.display(),
                self.bytes
            )));
        }
        Ok(FileIdentity {
            file: self
                .path
                .file_name()
                .and_then(|value| value.to_str())
                .ok_or_else(|| {
                    R2Error::DatasetContract("cache file name must be UTF-8".to_owned())
                })?
                .to_owned(),
            dtype: dtype.to_owned(),
            shape,
            bytes: self.bytes,
            blake3: self.hasher.finalize().to_hex().to_string(),
        })
    }
}

struct SplitWriters {
    split: &'static str,
    token_types: HashedWriter,
    token_seats: HashedWriter,
    token_payload: HashedWriter,
    board_type_counts: HashedWriter,
    graph_record_offsets: HashedWriter,
    graph_token_offsets: HashedWriter,
    graph_targets: HashedWriter,
    graph_relations: HashedWriter,
    graph_direction_bits: HashedWriter,
    market_features: HashedWriter,
    market_mask: HashedWriter,
    player_features: HashedWriter,
    player_mask: HashedWriter,
    global_features: HashedWriter,
    targets: HashedWriter,
    game_index: HashedWriter,
    turn: HashedWriter,
    total_graph_edges: u64,
    integrity: SplitIntegrity,
}

impl SplitWriters {
    fn create(root: &Path, split: &'static str) -> Result<Self> {
        let mut graph_record_offsets =
            HashedWriter::create(root.join(format!("{split}-graph-record-offsets.u64")))?;
        graph_record_offsets.write_bytes(&0u64.to_le_bytes())?;
        Ok(Self {
            split,
            token_types: HashedWriter::create(root.join(format!("{split}-token-types.u8")))?,
            token_seats: HashedWriter::create(root.join(format!("{split}-token-seats.u8")))?,
            token_payload: HashedWriter::create(root.join(format!("{split}-token-payload.i8")))?,
            board_type_counts: HashedWriter::create(
                root.join(format!("{split}-board-type-counts.u16")),
            )?,
            graph_record_offsets,
            graph_token_offsets: HashedWriter::create(
                root.join(format!("{split}-graph-token-offsets.u32")),
            )?,
            graph_targets: HashedWriter::create(root.join(format!("{split}-graph-targets.u16")))?,
            graph_relations: HashedWriter::create(
                root.join(format!("{split}-graph-relations.u8")),
            )?,
            graph_direction_bits: HashedWriter::create(
                root.join(format!("{split}-graph-direction-bits.u8")),
            )?,
            market_features: HashedWriter::create(
                root.join(format!("{split}-market-features.f32")),
            )?,
            market_mask: HashedWriter::create(root.join(format!("{split}-market-mask.u8")))?,
            player_features: HashedWriter::create(
                root.join(format!("{split}-player-features.f32")),
            )?,
            player_mask: HashedWriter::create(root.join(format!("{split}-player-mask.u8")))?,
            global_features: HashedWriter::create(
                root.join(format!("{split}-global-features.f32")),
            )?,
            targets: HashedWriter::create(root.join(format!("{split}-targets.f32")))?,
            game_index: HashedWriter::create(root.join(format!("{split}-game-index.u64")))?,
            turn: HashedWriter::create(root.join(format!("{split}-turn.u8")))?,
            total_graph_edges: 0,
            integrity: SplitIntegrity::default(),
        })
    }

    fn write_record(&mut self, record: &PositionRecord, encoded: &MlxEncodedState) -> Result<()> {
        self.token_types.write_bytes(&encoded.token_types)?;
        self.token_seats.write_bytes(&encoded.token_seats)?;
        self.token_payload.write_bytes(
            &encoded
                .token_payload
                .iter()
                .map(|value| *value as u8)
                .collect::<Vec<_>>(),
        )?;
        for count in encoded.board_type_counts.iter().flatten() {
            self.board_type_counts.write_bytes(&count.to_le_bytes())?;
        }
        for offset in &encoded.graph_token_offsets {
            self.graph_token_offsets
                .write_bytes(&offset.to_le_bytes())?;
        }
        for target in &encoded.graph_targets {
            self.graph_targets.write_bytes(&target.to_le_bytes())?;
        }
        self.graph_relations.write_bytes(&encoded.graph_relations)?;
        self.graph_direction_bits
            .write_bytes(&encoded.graph_direction_bits)?;
        self.total_graph_edges = self
            .total_graph_edges
            .checked_add(encoded.graph_targets.len() as u64)
            .ok_or_else(|| R2Error::DatasetContract("graph edge count overflowed".to_owned()))?;
        self.graph_record_offsets
            .write_bytes(&self.total_graph_edges.to_le_bytes())?;

        let (market, market_mask) = market_features(record)?;
        for value in market {
            self.market_features.write_bytes(&value.to_le_bytes())?;
        }
        self.market_mask.write_bytes(&market_mask)?;
        let (players, player_mask) = player_features(record)?;
        for value in players {
            self.player_features.write_bytes(&value.to_le_bytes())?;
        }
        self.player_mask.write_bytes(&player_mask)?;
        for value in global_features(record)? {
            self.global_features.write_bytes(&value.to_le_bytes())?;
        }
        for target in record.targets {
            self.targets.write_bytes(&(target as f32).to_le_bytes())?;
        }
        self.game_index
            .write_bytes(&record.game_index.to_le_bytes())?;
        self.turn.write_bytes(&[record.turn])?;

        let type_counts = encoded.type_counts();
        let board_counts = encoded.board_counts();
        let active = encoded.active_tokens();
        self.integrity.records += 1;
        self.integrity.active_tokens += active;
        self.integrity.padding_tokens += TOKEN_CAPACITY - active;
        self.integrity.graph_edges += encoded.graph_targets.len();
        self.integrity.max_active_tokens = self.integrity.max_active_tokens.max(active);
        self.integrity.max_active_tokens_per_board = self
            .integrity
            .max_active_tokens_per_board
            .max(*board_counts.iter().max().unwrap_or(&0));
        self.integrity.max_graph_degree = self.integrity.max_graph_degree.max(encoded.max_degree);
        for (index, count) in type_counts.into_iter().enumerate() {
            self.integrity.layer_maxima[index] = self.integrity.layer_maxima[index].max(count);
            self.integrity.type_token_totals[index] += count;
        }
        Ok(())
    }

    fn finish(self) -> Result<SplitManifest> {
        let records = self.integrity.records;
        let edges = usize::try_from(self.total_graph_edges).map_err(|_| {
            R2Error::DatasetContract("graph edge count does not fit usize".to_owned())
        })?;
        let mut files = BTreeMap::new();
        insert_file(
            &mut files,
            "token_types",
            self.token_types
                .finish("|u1", vec![records, BOARD_SLOTS, BOARD_TOKEN_CAPACITY])?,
        );
        insert_file(
            &mut files,
            "token_seats",
            self.token_seats
                .finish("|u1", vec![records, BOARD_SLOTS, BOARD_TOKEN_CAPACITY])?,
        );
        insert_file(
            &mut files,
            "token_payload",
            self.token_payload.finish(
                "|i1",
                vec![
                    records,
                    BOARD_SLOTS,
                    BOARD_TOKEN_CAPACITY,
                    TOKEN_PAYLOAD_WIDTH,
                ],
            )?,
        );
        insert_file(
            &mut files,
            "board_type_counts",
            self.board_type_counts
                .finish("<u2", vec![records, BOARD_SLOTS, 4])?,
        );
        insert_file(
            &mut files,
            "graph_record_offsets",
            self.graph_record_offsets.finish("<u8", vec![records + 1])?,
        );
        insert_file(
            &mut files,
            "graph_token_offsets",
            self.graph_token_offsets
                .finish("<u4", vec![records, TOKEN_CAPACITY + 1])?,
        );
        insert_file(
            &mut files,
            "graph_targets",
            self.graph_targets.finish("<u2", vec![edges])?,
        );
        insert_file(
            &mut files,
            "graph_relations",
            self.graph_relations.finish("|u1", vec![edges])?,
        );
        insert_file(
            &mut files,
            "graph_direction_bits",
            self.graph_direction_bits.finish("|u1", vec![edges])?,
        );
        insert_file(
            &mut files,
            "market_features",
            self.market_features
                .finish("<f4", vec![records, 4, MARKET_FEATURES])?,
        );
        insert_file(
            &mut files,
            "market_mask",
            self.market_mask.finish("|u1", vec![records, 4])?,
        );
        insert_file(
            &mut files,
            "player_features",
            self.player_features
                .finish("<f4", vec![records, BOARD_SLOTS, PLAYER_FEATURES])?,
        );
        insert_file(
            &mut files,
            "player_mask",
            self.player_mask.finish("|u1", vec![records, BOARD_SLOTS])?,
        );
        insert_file(
            &mut files,
            "global_features",
            self.global_features
                .finish("<f4", vec![records, GLOBAL_FEATURES])?,
        );
        insert_file(
            &mut files,
            "targets",
            self.targets.finish("<f4", vec![records, TARGET_DIM])?,
        );
        insert_file(
            &mut files,
            "game_index",
            self.game_index.finish("<u8", vec![records])?,
        );
        insert_file(&mut files, "turn", self.turn.finish("|u1", vec![records])?);

        let expected_tokens = records.checked_mul(TOKEN_CAPACITY).ok_or_else(|| {
            R2Error::DatasetContract("split token accounting overflowed".to_owned())
        })?;
        if self.integrity.active_tokens + self.integrity.padding_tokens != expected_tokens {
            return Err(R2Error::DatasetContract(format!(
                "{} token accounting drifted",
                self.split
            )));
        }
        if self.integrity.type_token_totals.iter().sum::<usize>() != self.integrity.active_tokens
            || self.integrity.max_active_tokens > EXPECTED_LAYER_MAXIMA[4]
            || self.integrity.max_active_tokens_per_board > BOARD_TOKEN_CAPACITY
        {
            return Err(R2Error::DatasetContract(format!(
                "{} type or board-local token accounting drifted",
                self.split
            )));
        }
        Ok(SplitManifest {
            records,
            files,
            integrity: self.integrity,
        })
    }
}

pub fn encode_sparse_state(state: &SparsePublicState) -> Result<MlxEncodedState> {
    encode_sparse_state_with_board_capacity(state, BOARD_TOKEN_CAPACITY)
}

fn encode_sparse_state_with_board_capacity(
    state: &SparsePublicState,
    board_token_capacity: usize,
) -> Result<MlxEncodedState> {
    if state.supplied_tile.is_some() {
        return Err(R2Error::DatasetContract(
            "R2 MLX V1 does not admit supplied-tile compatibility tokens".to_owned(),
        ));
    }
    let token_capacity = BOARD_SLOTS
        .checked_mul(board_token_capacity)
        .ok_or_else(|| R2Error::DatasetContract("R2 token capacity overflowed".to_owned()))?;
    let mut token_types = vec![0; token_capacity];
    let mut token_seats = vec![0; token_capacity];
    let mut token_payload = vec![0; token_capacity * TOKEN_PAYLOAD_WIDTH];
    let mut board_type_counts = [[0u16; 4]; BOARD_SLOTS];
    let mut occupied_slots = HashMap::new();
    let mut frontier_slots = HashMap::new();
    let mut component_slots = HashMap::new();
    let mut motif_slots = HashMap::new();

    for (relative_seat, board_counts) in board_type_counts.iter_mut().enumerate() {
        let seat = relative_seat as u8;
        let counts = [
            state
                .occupied_tiles
                .iter()
                .filter(|token| token.relative_seat == seat)
                .count(),
            state
                .legal_frontier
                .iter()
                .filter(|token| token.relative_seat == seat)
                .count(),
            state
                .habitat_components
                .iter()
                .filter(|token| token.relative_seat == seat)
                .count(),
            state
                .wildlife_motifs
                .iter()
                .filter(|token| token.relative_seat == seat)
                .count(),
        ];
        let board_total = counts.iter().sum::<usize>();
        if board_total > board_token_capacity {
            return Err(R2Error::DatasetContract(format!(
                "relative board {relative_seat} has {board_total} R2 tokens; \
                 board-local capacity is {board_token_capacity}"
            )));
        }
        if board_token_capacity == R2_MAP_BOARD_TOKEN_CAPACITY {
            let occupied = counts[0];
            let legal_layer_maxima = [
                MAX_BOARD_TILES,
                2 * occupied + 4,
                2 * occupied,
                occupied.saturating_sub(3),
            ];
            if counts
                .iter()
                .zip(legal_layer_maxima)
                .any(|(count, maximum)| *count > maximum)
            {
                return Err(R2Error::DatasetContract(format!(
                    "relative board {relative_seat} violates the legal R2-MAP token-layer bounds: counts={counts:?}, maxima={legal_layer_maxima:?}"
                )));
            }
        }
        *board_counts = counts.map(|count| count as u16);

        let mut cursor = relative_seat * board_token_capacity;
        for token in state
            .occupied_tiles
            .iter()
            .filter(|token| token.relative_seat == seat)
        {
            occupied_slots.insert((seat, token.coord), cursor);
            cursor += 1;
        }
        for token in state
            .legal_frontier
            .iter()
            .filter(|token| token.relative_seat == seat)
        {
            frontier_slots.insert((seat, token.coord), cursor);
            cursor += 1;
        }
        for token in state
            .habitat_components
            .iter()
            .filter(|token| token.relative_seat == seat)
        {
            component_slots.insert((seat, token.component_id), cursor);
            cursor += 1;
        }
        for token in state
            .wildlife_motifs
            .iter()
            .filter(|token| token.relative_seat == seat)
        {
            motif_slots.insert((seat, token.coord), cursor);
            cursor += 1;
        }
    }

    for token in &state.occupied_tiles {
        let slot = occupied_slots[&(token.relative_seat, token.coord)];
        let values = [
            token.coord.q,
            token.coord.r,
            token.terrain_a as i16,
            token.terrain_b.map_or(NONE_CATEGORY, |value| value as i16),
            i16::from(token.rotation.get()),
            token.directed_edge_terrains[0] as i16,
            token.directed_edge_terrains[1] as i16,
            token.directed_edge_terrains[2] as i16,
            token.directed_edge_terrains[3] as i16,
            token.directed_edge_terrains[4] as i16,
            token.directed_edge_terrains[5] as i16,
            i16::from(token.wildlife_eligibility.bits()),
            token
                .placed_wildlife
                .map_or(NONE_CATEGORY, |value| value as i16),
            i16::from(token.keystone),
        ];
        write_token(
            &mut token_types,
            &mut token_seats,
            &mut token_payload,
            slot,
            TOKEN_TYPE_OCCUPIED,
            token.relative_seat,
            &values,
        )?;
    }

    for token in &state.legal_frontier {
        let slot = frontier_slots[&(token.relative_seat, token.coord)];
        if token.touched_habitat_components.len() > 6 {
            return Err(R2Error::DatasetContract(
                "frontier token touches more than six habitat components".to_owned(),
            ));
        }
        let mut values = vec![
            token.coord.q,
            token.coord.r,
            i16::from(token.neighbor_presence_bits),
        ];
        values.extend(
            token
                .neighbor_facing_terrains
                .iter()
                .map(|value| value.map_or(NONE_CATEGORY, |terrain| terrain as i16)),
        );
        values.extend(
            token
                .adjacent_wildlife_counts
                .iter()
                .map(|value| i16::from(*value)),
        );
        values.push(i16::from(token.occupied_neighbor_runs));
        values.push(i16::from(token.opposite_neighbor_pair_bits));
        values.push(token.touched_habitat_components.len() as i16);
        for touch in &token.touched_habitat_components {
            let component_slot = *component_slots
                .get(&(token.relative_seat, touch.component_id))
                .ok_or_else(|| {
                    R2Error::DatasetContract(
                        "frontier references an absent habitat component".to_owned(),
                    )
                })?;
            values.extend_from_slice(&[
                touch.terrain as i16,
                (component_slot % board_token_capacity) as i16,
                touch.component_size as i16,
                i16::from(touch.contact_edge_bits),
            ]);
        }
        while values.len() < 41 {
            values.push(0);
        }
        values.extend(
            token
                .resulting_size_by_terrain
                .iter()
                .map(|value| *value as i16),
        );
        values.push(i16::from(token.habitat_bridge_terrain_bits));
        values.push(i16::from(token.repeated_component_contact_terrain_bits));
        write_token(
            &mut token_types,
            &mut token_seats,
            &mut token_payload,
            slot,
            TOKEN_TYPE_FRONTIER,
            token.relative_seat,
            &values,
        )?;
    }

    for token in &state.habitat_components {
        let slot = component_slots[&(token.relative_seat, token.component_id)];
        if token.members.len() > 23 || token.members.len() != usize::from(token.member_count) {
            return Err(R2Error::DatasetContract(
                "habitat component member accounting exceeds the exact board limit".to_owned(),
            ));
        }
        let mut values = vec![
            token.terrain as i16,
            token.component_id as i16,
            token.member_count as i16,
            token.matching_internal_edge_count as i16,
            token.open_boundary_edge_count as i16,
            token.frontier_contact_count as i16,
        ];
        for member in &token.members {
            values.push(member.q);
            values.push(member.r);
        }
        write_token(
            &mut token_types,
            &mut token_seats,
            &mut token_payload,
            slot,
            TOKEN_TYPE_COMPONENT,
            token.relative_seat,
            &values,
        )?;
    }

    for token in &state.wildlife_motifs {
        let slot = motif_slots[&(token.relative_seat, token.coord)];
        let mut values = vec![token.coord.q, token.coord.r, token.wildlife as i16];
        values.extend(
            token
                .neighbor_wildlife
                .iter()
                .map(|value| value.map_or(NONE_CATEGORY, |wildlife| wildlife as i16)),
        );
        values.extend(
            token
                .adjacent_wildlife_counts
                .iter()
                .map(|value| i16::from(*value)),
        );
        values.push(i16::from(token.same_species_neighbor_bits));
        write_token(
            &mut token_types,
            &mut token_seats,
            &mut token_payload,
            slot,
            TOKEN_TYPE_MOTIF,
            token.relative_seat,
            &values,
        )?;
    }

    let mut adjacency = vec![Vec::<GraphEdge>::new(); token_capacity];
    let component_memberships = state
        .habitat_components
        .iter()
        .flat_map(|component| {
            let slot = component_slots[&(component.relative_seat, component.component_id)];
            component
                .members
                .iter()
                .map(move |coord| ((component.relative_seat, *coord), slot))
        })
        .fold(
            HashMap::<(u8, AxialCoord), Vec<usize>>::new(),
            |mut map, (key, slot)| {
                map.entry(key).or_default().push(slot);
                map
            },
        );

    for token in &state.occupied_tiles {
        let source = occupied_slots[&(token.relative_seat, token.coord)];
        for edge in 0..6 {
            let neighbor = token.coord.neighbor(edge);
            if let Some(target) = occupied_slots.get(&(token.relative_seat, neighbor)) {
                adjacency[source].push(GraphEdge::new(*target, REL_OCCUPIED_NEIGHBOR, 1 << edge)?);
            }
            if let Some(target) = frontier_slots.get(&(token.relative_seat, neighbor)) {
                adjacency[source].push(GraphEdge::new(*target, REL_OCCUPIED_FRONTIER, 1 << edge)?);
            }
        }
        if let Some(components) = component_memberships.get(&(token.relative_seat, token.coord)) {
            for target in components {
                adjacency[source].push(GraphEdge::new(*target, REL_OCCUPIED_COMPONENT, 0)?);
            }
        }
        if let Some(target) = motif_slots.get(&(token.relative_seat, token.coord)) {
            adjacency[source].push(GraphEdge::new(*target, REL_OCCUPIED_MOTIF, 0)?);
        }
    }

    for token in &state.legal_frontier {
        let source = frontier_slots[&(token.relative_seat, token.coord)];
        for edge in 0..6 {
            let neighbor = token.coord.neighbor(edge);
            if let Some(target) = occupied_slots.get(&(token.relative_seat, neighbor)) {
                adjacency[source].push(GraphEdge::new(*target, REL_FRONTIER_OCCUPIED, 1 << edge)?);
            }
        }
        for touch in &token.touched_habitat_components {
            let target = component_slots[&(token.relative_seat, touch.component_id)];
            adjacency[source].push(GraphEdge::new(
                target,
                REL_FRONTIER_COMPONENT,
                touch.contact_edge_bits,
            )?);
        }
    }

    for token in &state.habitat_components {
        let source = component_slots[&(token.relative_seat, token.component_id)];
        for member in &token.members {
            let target = occupied_slots[&(token.relative_seat, *member)];
            adjacency[source].push(GraphEdge::new(target, REL_COMPONENT_OCCUPIED, 0)?);
        }
    }

    for token in &state.wildlife_motifs {
        let source = motif_slots[&(token.relative_seat, token.coord)];
        let occupied = occupied_slots[&(token.relative_seat, token.coord)];
        adjacency[source].push(GraphEdge::new(occupied, REL_MOTIF_OCCUPIED, 0)?);
        for edge in 0..6 {
            let neighbor = token.coord.neighbor(edge);
            if let Some(target) = motif_slots.get(&(token.relative_seat, neighbor)) {
                adjacency[source].push(GraphEdge::new(*target, REL_MOTIF_NEIGHBOR, 1 << edge)?);
            }
        }
    }

    let mut graph_token_offsets = Vec::with_capacity(token_capacity + 1);
    let mut graph_targets = Vec::new();
    let mut graph_relations = Vec::new();
    let mut graph_direction_bits = Vec::new();
    let mut max_degree = 0;
    graph_token_offsets.push(0);
    for (source, edges) in adjacency.iter_mut().enumerate() {
        edges.sort_unstable();
        edges.dedup();
        if edges.len() > GRAPH_MAX_DEGREE {
            return Err(R2Error::DatasetContract(format!(
                "R2 graph token {source} has degree {}; hard capacity is {GRAPH_MAX_DEGREE}",
                edges.len()
            )));
        }
        max_degree = max_degree.max(edges.len());
        for edge in edges {
            graph_targets.push(edge.target);
            graph_relations.push(edge.relation);
            graph_direction_bits.push(edge.direction_bits);
        }
        graph_token_offsets.push(graph_targets.len() as u32);
    }

    let encoded = MlxEncodedState {
        token_types,
        token_seats,
        token_payload,
        board_type_counts,
        graph_token_offsets,
        graph_targets,
        graph_relations,
        graph_direction_bits,
        max_degree,
    };
    validate_encoded_state_with_board_capacity(&encoded, board_token_capacity)?;
    Ok(encoded)
}

/// Encode one exact relative board without constructing the relational graph
/// or revisiting any sibling board. Active-row order and payload bytes match
/// [`encode_sparse_state`], while the live padding uses the independently
/// proved 139-token rules bound instead of the foundation corpus maximum.
pub(crate) fn encode_sparse_board_tokens(
    state: &SparseBoardState,
    transform: D6Transform,
) -> Result<EncodedBoardTokens> {
    let seat = state.relative_seat;
    let occupied = state.occupied_tiles.len();
    let maximum_frontier = 2 * occupied + 4;
    let maximum_components = 2 * occupied;
    let maximum_motifs = occupied.saturating_sub(3);
    if occupied > MAX_BOARD_TILES
        || state.legal_frontier.len() > maximum_frontier
        || state.habitat_components.len() > maximum_components
        || state.wildlife_motifs.len() > maximum_motifs
    {
        return Err(R2Error::DatasetContract(format!(
            "relative board {seat} violates the legal R2-MAP token-layer bounds: occupied={occupied}, frontier={}/{maximum_frontier}, components={}/{maximum_components}, motifs={}/{maximum_motifs}",
            state.legal_frontier.len(),
            state.habitat_components.len(),
            state.wildlife_motifs.len(),
        )));
    }
    let counts = [
        occupied,
        state.legal_frontier.len(),
        state.habitat_components.len(),
        state.wildlife_motifs.len(),
    ];
    let board_total = counts.iter().sum::<usize>();
    if board_total > R2_MAP_BOARD_TOKEN_CAPACITY {
        return Err(R2Error::DatasetContract(format!(
            "relative board {seat} has {board_total} R2-MAP tokens; rules-complete board-local capacity is {R2_MAP_BOARD_TOKEN_CAPACITY}"
        )));
    }
    let type_counts = counts.map(|count| count as u16);
    let mut token_types = vec![0; R2_MAP_BOARD_TOKEN_CAPACITY];
    let mut token_seats = vec![0; R2_MAP_BOARD_TOKEN_CAPACITY];
    let mut token_payload = vec![0; R2_MAP_BOARD_TOKEN_CAPACITY * TOKEN_PAYLOAD_WIDTH];
    let mut occupied_slots = HashMap::new();
    let mut frontier_slots = HashMap::new();
    let mut component_slots = HashMap::new();
    let mut motif_slots = HashMap::new();
    let mut cursor = 0usize;
    for token in &state.occupied_tiles {
        occupied_slots.insert(token.coord, cursor);
        cursor += 1;
    }
    for token in &state.legal_frontier {
        frontier_slots.insert(token.coord, cursor);
        cursor += 1;
    }
    for token in &state.habitat_components {
        component_slots.insert(token.component_id, cursor);
        cursor += 1;
    }
    for token in &state.wildlife_motifs {
        motif_slots.insert(token.coord, cursor);
        cursor += 1;
    }

    for token in &state.occupied_tiles {
        let values = [
            token.coord.q,
            token.coord.r,
            token.terrain_a as i16,
            token.terrain_b.map_or(NONE_CATEGORY, |value| value as i16),
            i16::from(token.rotation.get()),
            token.directed_edge_terrains[0] as i16,
            token.directed_edge_terrains[1] as i16,
            token.directed_edge_terrains[2] as i16,
            token.directed_edge_terrains[3] as i16,
            token.directed_edge_terrains[4] as i16,
            token.directed_edge_terrains[5] as i16,
            i16::from(token.wildlife_eligibility.bits()),
            token
                .placed_wildlife
                .map_or(NONE_CATEGORY, |value| value as i16),
            i16::from(token.keystone),
        ];
        write_token(
            &mut token_types,
            &mut token_seats,
            &mut token_payload,
            occupied_slots[&token.coord],
            TOKEN_TYPE_OCCUPIED,
            seat,
            &values,
        )?;
    }

    for token in &state.legal_frontier {
        if token.touched_habitat_components.len() > 6 {
            return Err(R2Error::DatasetContract(
                "frontier token touches more than six habitat components".to_owned(),
            ));
        }
        let mut values = vec![
            token.coord.q,
            token.coord.r,
            i16::from(token.neighbor_presence_bits),
        ];
        values.extend(
            token
                .neighbor_facing_terrains
                .iter()
                .map(|value| value.map_or(NONE_CATEGORY, |terrain| terrain as i16)),
        );
        values.extend(
            token
                .adjacent_wildlife_counts
                .iter()
                .map(|value| i16::from(*value)),
        );
        values.push(i16::from(token.occupied_neighbor_runs));
        values.push(i16::from(token.opposite_neighbor_pair_bits));
        values.push(token.touched_habitat_components.len() as i16);
        for touch in &token.touched_habitat_components {
            let component_slot = *component_slots.get(&touch.component_id).ok_or_else(|| {
                R2Error::DatasetContract(
                    "frontier references an absent habitat component".to_owned(),
                )
            })?;
            values.extend_from_slice(&[
                touch.terrain as i16,
                component_slot as i16,
                touch.component_size as i16,
                i16::from(touch.contact_edge_bits),
            ]);
        }
        while values.len() < 41 {
            values.push(0);
        }
        values.extend(
            token
                .resulting_size_by_terrain
                .iter()
                .map(|value| *value as i16),
        );
        values.push(i16::from(token.habitat_bridge_terrain_bits));
        values.push(i16::from(token.repeated_component_contact_terrain_bits));
        write_token(
            &mut token_types,
            &mut token_seats,
            &mut token_payload,
            frontier_slots[&token.coord],
            TOKEN_TYPE_FRONTIER,
            seat,
            &values,
        )?;
    }

    for token in &state.habitat_components {
        if token.members.len() > 23 || token.members.len() != usize::from(token.member_count) {
            return Err(R2Error::DatasetContract(
                "habitat component member accounting exceeds the exact board limit".to_owned(),
            ));
        }
        let mut values = vec![
            token.terrain as i16,
            token.component_id as i16,
            token.member_count as i16,
            token.matching_internal_edge_count as i16,
            token.open_boundary_edge_count as i16,
            token.frontier_contact_count as i16,
        ];
        for member in &token.members {
            values.push(member.q);
            values.push(member.r);
        }
        write_token(
            &mut token_types,
            &mut token_seats,
            &mut token_payload,
            component_slots[&token.component_id],
            TOKEN_TYPE_COMPONENT,
            seat,
            &values,
        )?;
    }

    for token in &state.wildlife_motifs {
        let mut values = vec![token.coord.q, token.coord.r, token.wildlife as i16];
        values.extend(
            token
                .neighbor_wildlife
                .iter()
                .map(|value| value.map_or(NONE_CATEGORY, |wildlife| wildlife as i16)),
        );
        values.extend(
            token
                .adjacent_wildlife_counts
                .iter()
                .map(|value| i16::from(*value)),
        );
        values.push(i16::from(token.same_species_neighbor_bits));
        write_token(
            &mut token_types,
            &mut token_seats,
            &mut token_payload,
            motif_slots[&token.coord],
            TOKEN_TYPE_MOTIF,
            seat,
            &values,
        )?;
    }

    transform_token_payloads(&token_types, &mut token_payload, transform)?;
    Ok(EncodedBoardTokens {
        token_types,
        token_payload,
        type_counts,
    })
}

/// Apply the one-wildlife sibling delta to an already transformed tile-only
/// board encoding. Habitat/component geometry and slot order are unchanged;
/// only one occupied payload, adjacent frontier wildlife counts, and the
/// canonical motif suffix can differ.
pub(crate) fn encode_wildlife_sibling_tokens(
    tile_parent: &SparseBoardState,
    parent_encoded: &EncodedBoardTokens,
    sibling: &SparseBoardState,
    transform: D6Transform,
) -> Result<EncodedBoardTokens> {
    if sibling.relative_seat != tile_parent.relative_seat
        || sibling.occupied_tiles.len() != tile_parent.occupied_tiles.len()
        || sibling.legal_frontier.len() != tile_parent.legal_frontier.len()
        || sibling.habitat_components != tile_parent.habitat_components
    {
        return Err(R2Error::DatasetContract(
            "wildlife sibling changed board geometry or token order".to_owned(),
        ));
    }
    let mut encoded = parent_encoded.clone();
    for (slot, (candidate, parent)) in sibling
        .occupied_tiles
        .iter()
        .zip(&tile_parent.occupied_tiles)
        .enumerate()
    {
        if candidate.placed_wildlife != parent.placed_wildlife {
            encoded.token_payload[slot * TOKEN_PAYLOAD_WIDTH + 12] = candidate
                .placed_wildlife
                .map_or(NONE_CATEGORY as i8, |wildlife| wildlife as i8);
        }
    }

    let frontier_start = sibling.occupied_tiles.len();
    for (index, frontier) in sibling.legal_frontier.iter().enumerate() {
        let slot = frontier_start + index;
        let payload = slot * TOKEN_PAYLOAD_WIDTH;
        for (wildlife, count) in frontier.adjacent_wildlife_counts.iter().enumerate() {
            encoded.token_payload[payload + 9 + wildlife] = *count as i8;
        }
    }

    let motif_start =
        frontier_start + sibling.legal_frontier.len() + sibling.habitat_components.len();
    encoded.token_types[motif_start..].fill(0);
    encoded.token_payload[motif_start * TOKEN_PAYLOAD_WIDTH..].fill(0);
    let mut token_seats = vec![0; R2_MAP_BOARD_TOKEN_CAPACITY];
    for (index, token) in sibling.wildlife_motifs.iter().enumerate() {
        let slot = motif_start + index;
        if slot >= R2_MAP_BOARD_TOKEN_CAPACITY {
            return Err(R2Error::DatasetContract(format!(
                "wildlife sibling motif suffix exceeds rules-complete board capacity: occupied={}, frontier={}, components={}, motifs={}, total={}, capacity={R2_MAP_BOARD_TOKEN_CAPACITY}",
                sibling.occupied_tiles.len(),
                sibling.legal_frontier.len(),
                sibling.habitat_components.len(),
                sibling.wildlife_motifs.len(),
                motif_start + sibling.wildlife_motifs.len(),
            )));
        }
        let mut values = vec![token.coord.q, token.coord.r, token.wildlife as i16];
        values.extend(
            token
                .neighbor_wildlife
                .iter()
                .map(|value| value.map_or(NONE_CATEGORY, |wildlife| wildlife as i16)),
        );
        values.extend(
            token
                .adjacent_wildlife_counts
                .iter()
                .map(|value| i16::from(*value)),
        );
        values.push(i16::from(token.same_species_neighbor_bits));
        write_token(
            &mut encoded.token_types,
            &mut token_seats,
            &mut encoded.token_payload,
            slot,
            TOKEN_TYPE_MOTIF,
            sibling.relative_seat,
            &values,
        )?;
    }
    transform_token_payloads(
        &encoded.token_types[motif_start..],
        &mut encoded.token_payload[motif_start * TOKEN_PAYLOAD_WIDTH..],
        transform,
    )?;
    encoded.type_counts[3] = sibling.wildlife_motifs.len() as u16;
    Ok(encoded)
}

/// Independently encode the complete public state using the authoritative
/// relational encoder, then apply the exact D6 transform. The historical
/// sparse-foundation entry point remains frozen at 4x92; this live entry point
/// is versioned at the rules-complete 4x139 bound.
pub(crate) fn encode_r2_map_state_authoritative(
    state: &SparsePublicState,
    transform: D6Transform,
) -> Result<MlxEncodedState> {
    let encoded = encode_sparse_state_with_board_capacity(state, R2_MAP_BOARD_TOKEN_CAPACITY)?;
    transform_encoded_state_with_board_capacity(&encoded, transform, R2_MAP_BOARD_TOKEN_CAPACITY)
}

/// Encode the same live 4x139 contract without padding for replay storage.
/// Token rows come from the authoritative relational encoder, not the
/// incremental board encoder used by the serving fast path.
pub(crate) fn encode_compact_r2_map_tokens(
    state: &SparsePublicState,
    transform: D6Transform,
) -> Result<EncodedR2MapCompactTokens> {
    let authoritative = encode_r2_map_state_authoritative(state, transform)?;
    let mut token_types = Vec::new();
    let mut token_seats = Vec::new();
    let mut token_payload = Vec::new();
    for (relative_seat, counts) in authoritative.board_type_counts.iter().enumerate() {
        let active = counts
            .iter()
            .map(|count| usize::from(*count))
            .sum::<usize>();
        let start = relative_seat * R2_MAP_BOARD_TOKEN_CAPACITY;
        let end = start + active;
        token_types.extend_from_slice(&authoritative.token_types[start..end]);
        token_seats.extend_from_slice(&authoritative.token_seats[start..end]);
        token_payload.extend_from_slice(
            &authoritative.token_payload[start * TOKEN_PAYLOAD_WIDTH..end * TOKEN_PAYLOAD_WIDTH],
        );
    }
    Ok(EncodedR2MapCompactTokens {
        token_types,
        token_seats,
        token_payload,
        board_type_counts: authoritative.board_type_counts,
    })
}

/// Apply the accepted D6 tensor transform without changing token slots.
///
/// This is the Rust authority for the Python cache transform used by the
/// Perceiver arm. Set-like token consumers may retain slot order while exact
/// coordinates, rotations, directed values, component members, and graph
/// direction bits remain covariant.
pub fn transform_encoded_state(
    source: &MlxEncodedState,
    transform: D6Transform,
) -> Result<MlxEncodedState> {
    transform_encoded_state_with_board_capacity(source, transform, BOARD_TOKEN_CAPACITY)
}

fn transform_encoded_state_with_board_capacity(
    source: &MlxEncodedState,
    transform: D6Transform,
    board_token_capacity: usize,
) -> Result<MlxEncodedState> {
    validate_encoded_state_with_board_capacity(source, board_token_capacity)?;
    let mut transformed = source.clone();
    transform_token_payloads(
        &transformed.token_types,
        &mut transformed.token_payload,
        transform,
    )?;
    for bits in &mut transformed.graph_direction_bits {
        *bits = transform_direction_bits(*bits, transform)?;
    }
    validate_encoded_state_with_board_capacity(&transformed, board_token_capacity)?;
    Ok(transformed)
}

fn transform_token_payloads(
    token_types: &[u8],
    token_payload: &mut [i8],
    transform: D6Transform,
) -> Result<()> {
    if token_payload.len() != token_types.len() * TOKEN_PAYLOAD_WIDTH {
        return Err(R2Error::DatasetContract(
            "R2 token payload shape drifted".to_owned(),
        ));
    }
    for (slot, &token_type) in token_types.iter().enumerate() {
        if token_type == 0 {
            continue;
        }
        let base = slot * TOKEN_PAYLOAD_WIDTH;
        let payload = &mut token_payload[base..base + TOKEN_PAYLOAD_WIDTH];
        match token_type {
            TOKEN_TYPE_OCCUPIED => {
                transform_payload_coord(payload, 0, 1, transform)?;
                let rotation = payload[4];
                if !(0..=5).contains(&rotation) {
                    return Err(R2Error::DatasetContract(
                        "occupied token rotation is outside [0, 5]".to_owned(),
                    ));
                }
                payload[4] = if i16::from(payload[3]) == NONE_CATEGORY {
                    0
                } else {
                    transform_dual_rotation(rotation as u8, transform) as i8
                };
                permute_direction_values(&mut payload[5..11], transform)?;
            }
            TOKEN_TYPE_FRONTIER => {
                transform_payload_coord(payload, 0, 1, transform)?;
                payload[2] = transform_direction_bits(payload[2] as u8, transform)? as i8;
                permute_direction_values(&mut payload[3..9], transform)?;
                payload[15] = opposite_pair_bits(payload[2] as u8) as i8;
                let touch_count = usize::try_from(payload[16]).map_err(|_| {
                    R2Error::DatasetContract(
                        "frontier touch count is negative in the encoded payload".to_owned(),
                    )
                })?;
                if touch_count > 6 {
                    return Err(R2Error::DatasetContract(
                        "frontier touch count exceeds six".to_owned(),
                    ));
                }
                for touch in 0..touch_count {
                    let bit_slot = 20 + touch * 4;
                    payload[bit_slot] =
                        transform_direction_bits(payload[bit_slot] as u8, transform)? as i8;
                }
            }
            TOKEN_TYPE_COMPONENT => {
                let member_count = usize::try_from(payload[2]).map_err(|_| {
                    R2Error::DatasetContract(
                        "component member count is negative in the encoded payload".to_owned(),
                    )
                })?;
                if member_count > 23 || 6 + member_count * 2 > TOKEN_PAYLOAD_WIDTH {
                    return Err(R2Error::DatasetContract(
                        "component member payload exceeds the exact board bound".to_owned(),
                    ));
                }
                let mut members = Vec::with_capacity(member_count);
                for index in 0..member_count {
                    let q = payload[6 + index * 2];
                    let r = payload[7 + index * 2];
                    members.push(transform_i8_coord(q, r, transform)?);
                }
                members.sort_unstable();
                for (index, (q, r)) in members.into_iter().enumerate() {
                    payload[6 + index * 2] = q;
                    payload[7 + index * 2] = r;
                }
            }
            TOKEN_TYPE_MOTIF => {
                transform_payload_coord(payload, 0, 1, transform)?;
                permute_direction_values(&mut payload[3..9], transform)?;
                payload[14] = transform_direction_bits(payload[14] as u8, transform)? as i8;
            }
            _ => {
                return Err(R2Error::DatasetContract(format!(
                    "encoded token type {token_type} is outside [1, 4]"
                )));
            }
        }
    }
    Ok(())
}

fn transform_payload_coord(
    payload: &mut [i8],
    q_slot: usize,
    r_slot: usize,
    transform: D6Transform,
) -> Result<()> {
    let (q, r) = transform_i8_coord(payload[q_slot], payload[r_slot], transform)?;
    payload[q_slot] = q;
    payload[r_slot] = r;
    Ok(())
}

fn transform_i8_coord(q: i8, r: i8, transform: D6Transform) -> Result<(i8, i8)> {
    let coord = AxialCoord::new(i16::from(q), i16::from(r)).transformed(transform)?;
    Ok((
        i8::try_from(coord.q).map_err(|_| {
            R2Error::DatasetContract("transformed q coordinate exceeds i8".to_owned())
        })?,
        i8::try_from(coord.r).map_err(|_| {
            R2Error::DatasetContract("transformed r coordinate exceeds i8".to_owned())
        })?,
    ))
}

fn transform_dual_rotation(rotation: u8, transform: D6Transform) -> u8 {
    let rotation = i16::from(rotation);
    let steps = i16::from(transform.rotation_steps());
    let value = if transform.is_reflected() {
        steps - rotation - 2
    } else {
        rotation + steps
    };
    value.rem_euclid(6) as u8
}

fn permute_direction_values(values: &mut [i8], transform: D6Transform) -> Result<()> {
    if values.len() != 6 {
        return Err(R2Error::DatasetContract(
            "directional payload slice is not six-wide".to_owned(),
        ));
    }
    let source = values.to_vec();
    for edge in 0..6 {
        values[transform.transform_edge(edge).map_err(|error| {
            R2Error::DatasetContract(format!("D6 edge transform failed: {error}"))
        })?] = source[edge];
    }
    Ok(())
}

fn transform_direction_bits(bits: u8, transform: D6Transform) -> Result<u8> {
    let mut result = 0u8;
    for edge in 0..6 {
        if bits & (1 << edge) != 0 {
            result |= 1
                << transform.transform_edge(edge).map_err(|error| {
                    R2Error::DatasetContract(format!("D6 edge transform failed: {error}"))
                })?;
        }
    }
    Ok(result)
}

fn opposite_pair_bits(presence: u8) -> u8 {
    let mut result = 0u8;
    for pair in 0..3 {
        let present = ((presence >> pair) & 1) & ((presence >> (pair + 3)) & 1);
        result |= present << pair;
    }
    result
}

pub fn export_mlx_cache(
    corpus_lock: &Path,
    dataset_roots: &[PathBuf],
    output_root: &Path,
    receipt: Option<&Path>,
) -> Result<MlxExportReceipt> {
    let lock = read_corpus_lock(corpus_lock)?;
    let datasets = validate_corpus(dataset_roots, &lock)?;
    fs::create_dir_all(output_root)?;
    let temporary = output_root.join(format!(".r2-mlx-export-{}.tmp", std::process::id()));
    if temporary.exists() {
        fs::remove_dir_all(&temporary)?;
    }
    fs::create_dir(&temporary)?;

    let export_result = export_cache_files(&temporary, &datasets, &lock);
    let (splits, semantic_integrity, scientific_identity) = match export_result {
        Ok(value) => value,
        Err(error) => {
            fs::remove_dir_all(&temporary).ok();
            return Err(error);
        }
    };
    let executable = env::current_exe()?;
    let executable_blake3 = hash_file(&executable)?;
    let mut identity = scientific_identity;
    identity
        .as_object_mut()
        .expect("scientific identity is an object")
        .insert(
            "exporter_executable_blake3".to_owned(),
            Value::String(executable_blake3.clone()),
        );
    let cache_id = blake3::hash(&canonical_json_bytes(&identity))
        .to_hex()
        .to_string();
    let manifest = CacheManifest {
        schema_version: MLX_CACHE_SCHEMA_VERSION,
        cache_schema: MLX_CACHE_SCHEMA,
        experiment_id: MLX_EXPERIMENT_ID,
        cache_id: cache_id.clone(),
        scientific_identity: identity,
        tensor_contract: tensor_contract(),
        corpus: serde_json::to_value(&lock)?,
        semantic_integrity,
        splits,
        exporter: json!({
            "executable": executable.file_name().and_then(|value| value.to_str()),
            "executable_blake3": executable_blake3,
            "package_version": env!("CARGO_PKG_VERSION"),
        }),
    };
    let manifest_path = temporary.join("cache.json");
    fs::write(
        &manifest_path,
        [serde_json::to_vec_pretty(&manifest)?.as_slice(), b"\n"].concat(),
    )?;
    let final_root = output_root.join(&cache_id);
    if final_root.exists() {
        let existing = fs::read(final_root.join("cache.json"))?;
        let generated = fs::read(&manifest_path)?;
        if existing != generated {
            fs::remove_dir_all(&temporary).ok();
            return Err(R2Error::DatasetContract(format!(
                "content-address collision or provenance drift at {}",
                final_root.display()
            )));
        }
        validate_existing_cache_tree(&final_root, &manifest)?;
        fs::remove_dir_all(&temporary)?;
    } else {
        fs::rename(&temporary, &final_root)?;
    }
    let final_manifest = final_root.join("cache.json");
    let output = MlxExportReceipt {
        schema_version: 1,
        experiment_id: MLX_EXPERIMENT_ID,
        cache_id,
        cache_root: final_root.display().to_string(),
        cache_manifest: final_manifest.display().to_string(),
        cache_manifest_blake3: hash_file(&final_manifest)?,
    };
    if let Some(path) = receipt {
        write_json_atomic(path, &output)?;
    }
    Ok(output)
}

type ExportResult = (BTreeMap<String, SplitManifest>, Value, Value);

fn export_cache_files(
    root: &Path,
    datasets: &[ValidatedDataset],
    lock: &MlxCorpusLock,
) -> Result<ExportResult> {
    let mut train = SplitWriters::create(root, "train")?;
    let mut validation = SplitWriters::create(root, "validation")?;
    let mut public_hasher = Hasher::new();
    let mut packed_hasher = Hasher::new();
    let mut identity_semantic_hasher = Hasher::new();
    let mut d6_semantic_hasher = Hasher::new();
    let mut target_hasher = Hasher::new();
    let mut d6_checks = 0usize;
    let mut observed_layer_maxima = [0usize; 5];
    let mut observed_board_active =
        Vec::with_capacity(lock.identity.total_records.saturating_mul(BOARD_SLOTS));

    for dataset in datasets {
        for shard in &dataset.manifest.shards {
            for record in PositionShardReader::open(&dataset.root, shard)? {
                let record = record?;
                let state = SparsePublicState::from_position_record(&record, None)?;
                let reconstructed = state.reconstruct_position_record(record.targets)?;
                if reconstructed != record {
                    return Err(R2Error::DatasetContract(
                        "R2 MLX export changed the public source row".to_owned(),
                    ));
                }
                let public = state
                    .reconstruct_position_record([0; TARGET_DIM])?
                    .to_bytes();
                let packed = state.to_packed_bytes()?;
                update_framed_hash(&mut public_hasher, &public);
                update_framed_hash(&mut packed_hasher, &packed);
                let encoded = encode_sparse_state(&state)?;
                encoded.update_semantic_hash(&mut identity_semantic_hasher);
                observed_board_active.extend(encoded.board_counts());
                for target in record.targets {
                    target_hasher.update(&target.to_le_bytes());
                }
                let counts = [
                    state.occupied_tiles.len(),
                    state.legal_frontier.len(),
                    state.habitat_components.len(),
                    state.wildlife_motifs.len(),
                    state.total_spatial_tokens(),
                ];
                for (index, count) in counts.into_iter().enumerate() {
                    observed_layer_maxima[index] = observed_layer_maxima[index].max(count);
                }

                for transform in D6Transform::ALL {
                    let transformed = state.transformed(transform)?;
                    let restored = transformed.transformed(transform.inverse())?;
                    if restored != state {
                        return Err(R2Error::DatasetContract(format!(
                            "R2 MLX export failed D6 inverse for transform {}",
                            transform.id()
                        )));
                    }
                    let transformed_encoded = encode_sparse_state(&transformed)?;
                    d6_semantic_hasher.update(&[transform.id()]);
                    transformed_encoded.update_semantic_hash(&mut d6_semantic_hasher);
                    d6_checks += 1;
                }

                match dataset.manifest.split {
                    DatasetSplit::Train => train.write_record(&record, &encoded)?,
                    DatasetSplit::Validation => validation.write_record(&record, &encoded)?,
                    DatasetSplit::Test | DatasetSplit::Final => {
                        return Err(R2Error::DatasetContract(
                            "R2 MLX cache prohibits test and final data".to_owned(),
                        ));
                    }
                }
            }
        }
    }

    let train = train.finish()?;
    let validation = validation.finish()?;
    if train.records != lock.identity.train_records
        || validation.records != lock.identity.validation_records
        || train.records + validation.records != lock.identity.total_records
    {
        return Err(R2Error::DatasetContract(
            "R2 MLX exported split totals drifted from the corpus lock".to_owned(),
        ));
    }
    if observed_layer_maxima != lock.identity.layer_maxima
        || observed_layer_maxima != EXPECTED_LAYER_MAXIMA
    {
        return Err(R2Error::DatasetContract(format!(
            "R2 MLX layer maxima drifted: observed={observed_layer_maxima:?}"
        )));
    }
    let observed_type_totals = std::array::from_fn(|index| {
        train.integrity.type_token_totals[index] + validation.integrity.type_token_totals[index]
    });
    let observed_active_tokens = train.integrity.active_tokens + validation.integrity.active_tokens;
    observed_board_active.sort_unstable();
    let observed_per_board_p99 = nearest_rank(&observed_board_active, 99);
    let observed_per_board_max = observed_board_active.last().copied().unwrap_or(0);
    if observed_type_totals != lock.identity.type_token_totals
        || observed_type_totals != EXPECTED_TYPE_TOKEN_TOTALS
        || observed_active_tokens != lock.identity.active_tokens
        || observed_active_tokens != EXPECTED_ACTIVE_TOKENS
        || observed_per_board_p99 != lock.identity.per_board_p99_active_tokens
        || observed_per_board_p99 != FOUNDATION_PER_BOARD_P99_ACTIVE_TOKENS
        || observed_per_board_max != lock.identity.per_board_max_active_tokens
        || observed_per_board_max != FOUNDATION_PER_BOARD_MAX_ACTIVE_TOKENS
    {
        return Err(R2Error::DatasetContract(format!(
            "R2 MLX token census drifted: type_totals={observed_type_totals:?}, \
             active={observed_active_tokens}, board_p99={observed_per_board_p99}, \
             board_max={observed_per_board_max}"
        )));
    }
    let public_position_blake3 = public_hasher.finalize().to_hex().to_string();
    let packed_state_blake3 = packed_hasher.finalize().to_hex().to_string();
    if public_position_blake3 != lock.identity.foundation_public_position_blake3
        || packed_state_blake3 != lock.identity.foundation_packed_state_blake3
    {
        return Err(R2Error::DatasetContract(
            "R2 MLX public or packed stream differs from the frozen foundation".to_owned(),
        ));
    }

    let mut splits = BTreeMap::new();
    splits.insert("train".to_owned(), train);
    splits.insert("validation".to_owned(), validation);
    let d6_contract = cascadia_game::d6_contract_metadata();
    let semantic_integrity = json!({
        "exact_public_reconstruction_verified": true,
        "canonical_packed_round_trip_verified": true,
        "exact_no_truncation_verified": true,
        "padding_zero_verified": true,
        "graph_degree_bound_verified": true,
        "board_local_layout_verified": true,
        "derived_tokens_cached_after_regeneration": true,
        "type_token_totals": observed_type_totals,
        "active_tokens": observed_active_tokens,
        "per_board_p99_active_tokens": observed_per_board_p99,
        "per_board_max_active_tokens": observed_per_board_max,
        "identity_encoded_semantic_blake3": identity_semantic_hasher.finalize().to_hex().to_string(),
        "d6_regenerated_semantic_blake3": d6_semantic_hasher.finalize().to_hex().to_string(),
        "d6_transform_inverse_checks": d6_checks,
        "d6_contract_id": d6_contract.contract_id,
        "d6_contract_scientific_blake3": d6_contract.scientific_blake3,
        "public_position_blake3": public_position_blake3,
        "packed_state_blake3": packed_state_blake3,
        "target_blake3": target_hasher.finalize().to_hex().to_string(),
        "test_or_final_data_opened": false,
    });
    let scientific_identity = json!({
        "cache_schema": MLX_CACHE_SCHEMA,
        "corpus_lock_id": lock.lock_id,
        "experiment_id": MLX_EXPERIMENT_ID,
        "files": splits.iter().map(|(split, manifest)| {
            (
                split.clone(),
                manifest.files.iter().map(|(name, file)| {
                    (
                        name.clone(),
                        json!({
                            "blake3": file.blake3,
                            "bytes": file.bytes,
                            "dtype": file.dtype,
                            "file": file.file,
                            "shape": file.shape,
                        }),
                    )
                }).collect::<BTreeMap<_, _>>(),
            )
        }).collect::<BTreeMap<_, _>>(),
        "foundation_scientific_blake3": lock.identity.foundation_scientific_blake3,
        "identity_encoded_semantic_blake3": semantic_integrity["identity_encoded_semantic_blake3"],
        "d6_regenerated_semantic_blake3": semantic_integrity["d6_regenerated_semantic_blake3"],
        "target_blake3": semantic_integrity["target_blake3"],
        "split_records": {
            "train": splits["train"].records,
            "validation": splits["validation"].records,
        },
        "tensor_contract": tensor_contract(),
    });
    Ok((splits, semantic_integrity, scientific_identity))
}

fn validate_encoded_state_with_board_capacity(
    encoded: &MlxEncodedState,
    board_token_capacity: usize,
) -> Result<()> {
    let token_capacity = BOARD_SLOTS
        .checked_mul(board_token_capacity)
        .ok_or_else(|| R2Error::DatasetContract("R2 token capacity overflowed".to_owned()))?;
    if encoded.token_types.len() != token_capacity
        || encoded.token_seats.len() != token_capacity
        || encoded.token_payload.len() != token_capacity * TOKEN_PAYLOAD_WIDTH
        || encoded.graph_token_offsets.len() != token_capacity + 1
        || encoded.graph_targets.len() != encoded.graph_relations.len()
        || encoded.graph_targets.len() != encoded.graph_direction_bits.len()
    {
        return Err(R2Error::DatasetContract(
            "R2 MLX encoded tensor shape drifted".to_owned(),
        ));
    }
    if encoded.graph_token_offsets.first() != Some(&0)
        || usize::try_from(*encoded.graph_token_offsets.last().unwrap()).ok()
            != Some(encoded.graph_targets.len())
    {
        return Err(R2Error::DatasetContract(
            "R2 MLX graph offsets do not span the edge arrays".to_owned(),
        ));
    }
    let active = encoded.active_tokens();
    let mut active_slots = vec![false; token_capacity];
    for (board, counts) in encoded.board_type_counts.iter().enumerate() {
        let board_total = counts
            .iter()
            .map(|value| usize::from(*value))
            .sum::<usize>();
        if board_total > board_token_capacity {
            return Err(R2Error::DatasetContract(
                "R2 MLX board-local token capacity was exceeded".to_owned(),
            ));
        }
        let mut cursor = board * board_token_capacity;
        for (type_index, count) in counts.iter().copied().enumerate() {
            let count = usize::from(count);
            active_slots[cursor..cursor + count].fill(true);
            if encoded.token_types[cursor..cursor + count]
                .iter()
                .any(|value| *value != type_index as u8 + 1)
            {
                return Err(R2Error::DatasetContract(
                    "R2 MLX board-local type ordering drifted".to_owned(),
                ));
            }
            cursor += count;
        }
    }
    if active_slots.iter().filter(|value| **value).count() != active {
        return Err(R2Error::DatasetContract(
            "R2 MLX active-token accounting drifted".to_owned(),
        ));
    }
    for (slot, is_active) in active_slots.iter().copied().enumerate() {
        let board = slot / board_token_capacity;
        let payload =
            &encoded.token_payload[slot * TOKEN_PAYLOAD_WIDTH..(slot + 1) * TOKEN_PAYLOAD_WIDTH];
        if is_active {
            if !(1..=4).contains(&encoded.token_types[slot])
                || usize::from(encoded.token_seats[slot]) != board
            {
                return Err(R2Error::DatasetContract(
                    "R2 MLX active token has invalid type or board ownership".to_owned(),
                ));
            }
        } else if encoded.token_types[slot] != 0
            || encoded.token_seats[slot] != 0
            || payload.iter().any(|value| *value != 0)
        {
            return Err(R2Error::DatasetContract(
                "R2 MLX padding token contains nonzero data".to_owned(),
            ));
        }
        let start = encoded.graph_token_offsets[slot] as usize;
        let end = encoded.graph_token_offsets[slot + 1] as usize;
        if end < start || end > encoded.graph_targets.len() || end - start > GRAPH_MAX_DEGREE {
            return Err(R2Error::DatasetContract(
                "R2 MLX graph offsets exceed the degree contract".to_owned(),
            ));
        }
        if !is_active && end != start {
            return Err(R2Error::DatasetContract(
                "R2 MLX padding token has graph edges".to_owned(),
            ));
        }
        for edge in start..end {
            let target = usize::from(encoded.graph_targets[edge]);
            if target >= token_capacity
                || !active_slots[target]
                || target / board_token_capacity != board
                || encoded.graph_relations[edge] == 0
                || usize::from(encoded.graph_relations[edge]) >= GRAPH_RELATION_COUNT
                || encoded.graph_direction_bits[edge] & !0x3f != 0
            {
                return Err(R2Error::DatasetContract(
                    "R2 MLX graph edge violates its exact schema".to_owned(),
                ));
            }
        }
    }
    Ok(())
}

fn write_token(
    token_types: &mut [u8],
    token_seats: &mut [u8],
    token_payload: &mut [i8],
    slot: usize,
    token_type: u8,
    seat: u8,
    values: &[i16],
) -> Result<()> {
    if token_types.len() != token_seats.len()
        || token_payload.len() != token_types.len() * TOKEN_PAYLOAD_WIDTH
        || slot >= token_types.len()
        || values.len() > TOKEN_PAYLOAD_WIDTH
        || seat >= BOARD_SLOTS as u8
    {
        return Err(R2Error::DatasetContract(
            "R2 MLX token exceeds its exact tensor schema".to_owned(),
        ));
    }
    if token_types[slot] != 0 {
        return Err(R2Error::DatasetContract(
            "R2 MLX token slot was written twice".to_owned(),
        ));
    }
    token_types[slot] = token_type;
    token_seats[slot] = seat;
    let base = slot * TOKEN_PAYLOAD_WIDTH;
    for (index, value) in values.iter().copied().enumerate() {
        token_payload[base + index] = i8::try_from(value).map_err(|_| {
            R2Error::DatasetContract(format!(
                "R2 MLX payload value {value} does not fit the frozen i8 schema"
            ))
        })?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct GraphEdge {
    target: u16,
    relation: u8,
    direction_bits: u8,
}

impl GraphEdge {
    fn new(target: usize, relation: u8, direction_bits: u8) -> Result<Self> {
        let target = u16::try_from(target)
            .map_err(|_| R2Error::DatasetContract("R2 MLX graph target exceeds u16".to_owned()))?;
        if relation == 0
            || usize::from(relation) >= GRAPH_RELATION_COUNT
            || direction_bits & !0x3f != 0
        {
            return Err(R2Error::DatasetContract(
                "R2 MLX graph edge is outside the frozen schema".to_owned(),
            ));
        }
        Ok(Self {
            target,
            relation,
            direction_bits,
        })
    }
}

fn nearest_rank(sorted: &[usize], percentile: usize) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let rank = (percentile * sorted.len()).div_ceil(100).max(1);
    sorted[rank - 1]
}

fn market_features(record: &PositionRecord) -> Result<([f32; 4 * MARKET_FEATURES], [u8; 4])> {
    let mut output = [0.0; 4 * MARKET_FEATURES];
    let mut mask = [0u8; 4];
    for (slot, slot_mask) in mask.iter_mut().enumerate() {
        let raw = record.market_entities[slot];
        let active = raw[0] < 5 || raw[3] < 5;
        *slot_mask = u8::from(active);
        if !active {
            continue;
        }
        let base = slot * MARKET_FEATURES;
        one_hot(&mut output[base + 2..base + 7], raw[0], 5);
        one_hot_with_none(&mut output[base + 7..base + 13], raw[1], 5);
        mask_bits(&mut output[base + 19..base + 24], raw[2], 5);
        one_hot_with_none(&mut output[base + 24..base + 30], raw[3], 5);
        output[base + 30] = f32::from(raw[4]);
    }
    Ok((output, mask))
}

/// Encode the accepted R2 public-market features without exporting a cache.
pub fn encode_market_features(
    record: &PositionRecord,
) -> Result<([f32; 4 * MARKET_FEATURES], [u8; 4])> {
    market_features(record)
}

fn player_features(
    record: &PositionRecord,
) -> Result<([f32; BOARD_SLOTS * PLAYER_FEATURES], [u8; BOARD_SLOTS])> {
    if !(1..=BOARD_SLOTS as u8).contains(&record.player_count) {
        return Err(R2Error::DatasetContract(
            "player count is outside the four-slot MLX schema".to_owned(),
        ));
    }
    let mut output = [0.0; BOARD_SLOTS * PLAYER_FEATURES];
    let mut mask = [0u8; BOARD_SLOTS];
    let current_absolute_seat = record.turn % record.player_count;
    let current_relative_seat =
        (current_absolute_seat + record.player_count - record.active_seat) % record.player_count;
    for relative_seat in 0..usize::from(record.player_count) {
        let relative_seat_u8 = relative_seat as u8;
        let absolute_seat = (record.active_seat + relative_seat_u8) % record.player_count;
        let base = relative_seat * PLAYER_FEATURES;
        mask[relative_seat] = 1;
        output[base + relative_seat] = 1.0;
        output[base + 4 + usize::from(absolute_seat)] = 1.0;
        output[base + 8] = f32::from(relative_seat_u8 == current_relative_seat);
        output[base + 9] = f32::from(
            record.turn / record.player_count + u8::from(absolute_seat < current_absolute_seat),
        ) / 20.0;
        output[base + 10] = f32::from(
            (absolute_seat + record.player_count - current_absolute_seat) % record.player_count,
        ) / 4.0;
        output[base + 11] = f32::from(record.board_counts[relative_seat]) / 23.0;
        output[base + 12] = f32::from(record.nature_tokens[relative_seat]) / 20.0;
        for (index, value) in record.wildlife_counts[relative_seat]
            .into_iter()
            .enumerate()
        {
            output[base + 13 + index] = f32::from(value) / 20.0;
        }
        for (index, value) in record.habitat_sizes[relative_seat].into_iter().enumerate() {
            output[base + 18 + index] = f32::from(value) / 23.0;
        }
    }
    Ok((output, mask))
}

/// Encode the accepted R2 relative-player features without exporting a cache.
pub fn encode_player_features(
    record: &PositionRecord,
) -> Result<([f32; BOARD_SLOTS * PLAYER_FEATURES], [u8; BOARD_SLOTS])> {
    player_features(record)
}

fn global_features(record: &PositionRecord) -> Result<[f32; GLOBAL_FEATURES]> {
    let mut output = [0.0; GLOBAL_FEATURES];
    let mut offset = 0usize;
    let total_turns = f32::from(record.total_turns).max(1.0);
    output[offset] = f32::from(record.turn) / total_turns;
    offset += 1;
    output[offset] = (f32::from(record.total_turns) - f32::from(record.turn)) / total_turns;
    offset += 1;
    if !(1..=4).contains(&record.player_count) {
        return Err(R2Error::DatasetContract(
            "player count is outside the four-slot MLX schema".to_owned(),
        ));
    }
    output[offset + usize::from(record.player_count - 1)] = 1.0;
    offset += 4;
    for value in record.nature_tokens {
        output[offset] = f32::from(value) / 20.0;
        offset += 1;
    }
    for value in record.board_counts {
        output[offset] = f32::from(value) / 23.0;
        offset += 1;
    }
    for board in record.wildlife_counts {
        for value in board {
            output[offset] = f32::from(value) / 20.0;
            offset += 1;
        }
    }
    for board in record.habitat_sizes {
        for value in board {
            output[offset] = f32::from(value) / 23.0;
            offset += 1;
        }
    }
    let mut diversity = [false; 5];
    for market in record.market_entities {
        let wildlife = market[3];
        if wildlife < 5 {
            output[offset + usize::from(wildlife)] = 1.0;
            diversity[usize::from(wildlife)] = true;
        }
        offset += 5;
    }
    for card in record.scoring_cards {
        if card >= 4 {
            return Err(R2Error::DatasetContract(
                "scoring-card code exceeds the four-card schema".to_owned(),
            ));
        }
        output[offset + usize::from(card)] = 1.0;
        offset += 4;
    }
    output[offset] = f32::from(record.habitat_bonuses);
    offset += 1;
    output[offset] = diversity.iter().filter(|value| **value).count() as f32 / 4.0;
    offset += 1;
    if offset != GLOBAL_FEATURES {
        return Err(R2Error::DatasetContract(
            "global feature width drifted".to_owned(),
        ));
    }
    Ok(output)
}

/// Encode the accepted R2 public-global features without exporting a cache.
pub fn encode_global_features(record: &PositionRecord) -> Result<[f32; GLOBAL_FEATURES]> {
    global_features(record)
}

fn one_hot(output: &mut [f32], value: u8, classes: usize) {
    if usize::from(value) < classes {
        output[usize::from(value)] = 1.0;
    }
}

fn one_hot_with_none(output: &mut [f32], value: u8, classes: usize) {
    let index = if usize::from(value) < classes {
        usize::from(value)
    } else {
        classes
    };
    output[index] = 1.0;
}

fn mask_bits(output: &mut [f32], value: u8, bits: usize) {
    for (shift, output) in output.iter_mut().take(bits).enumerate() {
        *output = f32::from((value >> shift) & 1);
    }
}

fn tensor_contract() -> Value {
    let d6 = cascadia_game::d6_contract_metadata();
    json!({
        "token_layout": "board-major-4x92",
        "board_slots": BOARD_SLOTS,
        "board_token_capacity": BOARD_TOKEN_CAPACITY,
        "token_capacity": TOKEN_CAPACITY,
        "token_payload_width": TOKEN_PAYLOAD_WIDTH,
        "board_ownership_encoding": BOARD_OWNERSHIP_ENCODING,
        "foundation_per_board_p99_active_tokens": FOUNDATION_PER_BOARD_P99_ACTIVE_TOKENS,
        "foundation_per_board_max_active_tokens": FOUNDATION_PER_BOARD_MAX_ACTIVE_TOKENS,
        "board_local_type_order": [
            "occupied",
            "frontier",
            "habitat_component",
            "wildlife_motif",
        ],
        "foundation_type_token_totals": EXPECTED_TYPE_TOKEN_TOTALS,
        "foundation_active_tokens": EXPECTED_ACTIVE_TOKENS,
        "token_types": {
            "padding": 0,
            "occupied": TOKEN_TYPE_OCCUPIED,
            "frontier": TOKEN_TYPE_FRONTIER,
            "habitat_component": TOKEN_TYPE_COMPONENT,
            "wildlife_motif": TOKEN_TYPE_MOTIF,
        },
        "graph_max_degree": GRAPH_MAX_DEGREE,
        "graph_relation_count": GRAPH_RELATION_COUNT,
        "market_feature_dim": MARKET_FEATURES,
        "player_feature_dim": PLAYER_FEATURES,
        "global_feature_dim": GLOBAL_FEATURES,
        "target_dim": TARGET_DIM,
        "d6": {
            "contract_id": d6.contract_id,
            "scientific_blake3": d6.scientific_blake3,
            "coordinate_matrices": d6.coordinate_matrices,
            "direction_tables": d6.direction_tables,
            "dual_tile_rotation_tables": d6.dual_tile_rotation_tables,
            "single_tile_rotation_tables": d6.single_tile_rotation_tables,
        },
        "padding": {
            "token_type": 0,
            "token_seat": 0,
            "token_payload": "all-zero",
        },
        "truncation": "forbidden",
    })
}

fn read_corpus_lock(path: &Path) -> Result<MlxCorpusLock> {
    let lock: MlxCorpusLock = serde_json::from_reader(BufReader::new(File::open(path)?))?;
    if lock.schema_version != MLX_CORPUS_LOCK_SCHEMA_VERSION
        || lock.contract_id != MLX_CORPUS_LOCK_CONTRACT
    {
        return Err(R2Error::DatasetContract(
            "unsupported R2 MLX corpus lock".to_owned(),
        ));
    }
    let identity = serde_json::to_value(&lock.identity)?;
    if blake3::hash(&canonical_json_bytes(&identity))
        .to_hex()
        .to_string()
        != lock.lock_id
    {
        return Err(R2Error::DatasetContract(
            "R2 MLX corpus lock content address drifted".to_owned(),
        ));
    }
    if lock.identity.foundation_experiment_id != FOUNDATION_EXPERIMENT_ID
        || lock.identity.foundation_scientific_blake3 != FOUNDATION_SCIENTIFIC_BLAKE3
        || lock.identity.foundation_public_position_blake3 != FOUNDATION_PUBLIC_POSITION_BLAKE3
        || lock.identity.foundation_packed_state_blake3 != FOUNDATION_PACKED_STATE_BLAKE3
        || lock.identity.total_records != 60_000
        || lock.identity.train_records != 50_000
        || lock.identity.validation_records != 10_000
        || lock.identity.layer_maxima != EXPECTED_LAYER_MAXIMA
        || lock.identity.type_token_totals != EXPECTED_TYPE_TOKEN_TOTALS
        || lock.identity.active_tokens != EXPECTED_ACTIVE_TOKENS
        || lock.identity.per_board_p99_active_tokens != FOUNDATION_PER_BOARD_P99_ACTIVE_TOKENS
        || lock.identity.per_board_max_active_tokens != FOUNDATION_PER_BOARD_MAX_ACTIVE_TOKENS
        || lock.identity.datasets.len() != 8
    {
        return Err(R2Error::DatasetContract(
            "R2 MLX corpus lock does not bind the accepted foundation".to_owned(),
        ));
    }
    Ok(lock)
}

fn validate_corpus(roots: &[PathBuf], lock: &MlxCorpusLock) -> Result<Vec<ValidatedDataset>> {
    if roots.len() != lock.identity.datasets.len() {
        return Err(R2Error::DatasetContract(format!(
            "R2 MLX lock requires {} roots; received {}",
            lock.identity.datasets.len(),
            roots.len()
        )));
    }
    let mut datasets = Vec::with_capacity(roots.len());
    let mut train = 0usize;
    let mut validation = 0usize;
    for (order, (root, expected)) in roots.iter().zip(&lock.identity.datasets).enumerate() {
        if expected.order != order {
            return Err(R2Error::DatasetContract(
                "R2 MLX corpus lock order is noncanonical".to_owned(),
            ));
        }
        let root_name = root
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| R2Error::DatasetContract("dataset root must be UTF-8".to_owned()))?;
        if root_name != expected.root_name {
            return Err(R2Error::DatasetContract(format!(
                "dataset root order drifted at {order}: expected {}, found {root_name}",
                expected.root_name
            )));
        }
        let manifest_path = root.join("dataset.json");
        let manifest_blake3 = hash_file(&manifest_path)?;
        if manifest_blake3 != expected.manifest_blake3 {
            return Err(R2Error::DatasetContract(format!(
                "dataset manifest drifted: {}",
                manifest_path.display()
            )));
        }
        let manifest: DatasetManifest =
            serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
        validate_dataset(root, &manifest)?;
        if manifest.dataset_id != expected.dataset_id
            || manifest.total_records != expected.total_records
            || manifest.feature_schema != lock.identity.feature_schema
            || manifest.target_schema != lock.identity.target_schema
            || manifest.split.id() != expected.split
        {
            return Err(R2Error::DatasetContract(format!(
                "dataset identity drifted: {}",
                root.display()
            )));
        }
        match manifest.split {
            DatasetSplit::Train => train += manifest.total_records,
            DatasetSplit::Validation => validation += manifest.total_records,
            DatasetSplit::Test | DatasetSplit::Final => {
                return Err(R2Error::DatasetContract(
                    "R2 MLX corpus contains prohibited test or final data".to_owned(),
                ));
            }
        }
        datasets.push(ValidatedDataset {
            root: root.clone(),
            manifest,
            manifest_blake3,
        });
    }
    if train != lock.identity.train_records || validation != lock.identity.validation_records {
        return Err(R2Error::DatasetContract(
            "R2 MLX corpus split totals drifted".to_owned(),
        ));
    }
    if datasets
        .iter()
        .any(|dataset| dataset.manifest_blake3.is_empty())
    {
        return Err(R2Error::DatasetContract(
            "validated dataset lost its manifest digest".to_owned(),
        ));
    }
    Ok(datasets)
}

fn insert_file(files: &mut BTreeMap<String, FileIdentity>, name: &str, file: FileIdentity) {
    let previous = files.insert(name.to_owned(), file);
    debug_assert!(previous.is_none());
}

fn validate_existing_cache_tree(root: &Path, manifest: &CacheManifest) -> Result<()> {
    let mut expected = BTreeSet::from(["cache.json".to_owned()]);
    for split in manifest.splits.values() {
        for file in split.files.values() {
            let path = root.join(&file.file);
            let metadata = path.metadata().map_err(|error| {
                R2Error::DatasetContract(format!(
                    "content-addressed cache is missing {}: {error}",
                    path.display()
                ))
            })?;
            if !metadata.is_file()
                || metadata.len() != file.bytes
                || hash_file(&path)? != file.blake3
            {
                return Err(R2Error::DatasetContract(format!(
                    "content-addressed cache tensor drifted: {}",
                    path.display()
                )));
            }
            expected.insert(file.file.clone());
        }
    }
    let actual = fs::read_dir(root)?
        .map(|entry| {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                return Err(R2Error::DatasetContract(format!(
                    "content-addressed cache contains a non-file entry: {}",
                    entry.path().display()
                )));
            }
            entry.file_name().into_string().map_err(|_| {
                R2Error::DatasetContract(
                    "content-addressed cache file name must be UTF-8".to_owned(),
                )
            })
        })
        .collect::<Result<BTreeSet<_>>>()?;
    if actual != expected {
        return Err(R2Error::DatasetContract(format!(
            "content-addressed cache file set drifted: expected={expected:?}, actual={actual:?}"
        )));
    }
    Ok(())
}

fn dtype_width(dtype: &str) -> usize {
    match dtype {
        "|i1" | "|u1" => 1,
        "<u2" => 2,
        "<u4" | "<f4" => 4,
        "<u8" => 8,
        _ => panic!("unsupported R2 MLX dtype {dtype}"),
    }
}

fn canonical_json_bytes(value: &Value) -> Vec<u8> {
    serde_json::to_vec(&sort_json(value)).expect("JSON values always serialize")
}

fn sort_json(value: &Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.iter().map(sort_json).collect()),
        Value::Object(values) => {
            let mut sorted = BTreeMap::new();
            for (key, value) in values {
                sorted.insert(key.clone(), sort_json(value));
            }
            serde_json::to_value(sorted).expect("sorted JSON object serializes")
        }
        _ => value.clone(),
    }
}

fn update_framed_hash(hasher: &mut Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn hash_file(path: &Path) -> Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Hasher::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.tmp");
    let mut writer = BufWriter::new(File::create(&temporary)?);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WildlifeMotifToken;
    use cascadia_game::{GameConfig, GameSeed, GameState, Wildlife};

    #[test]
    fn board_local_capacity_covers_the_frozen_foundation() {
        assert_eq!(TOKEN_CAPACITY, 4 * 92);
        assert_eq!(FOUNDATION_PER_BOARD_P99_ACTIVE_TOKENS, 83);
        assert_eq!(FOUNDATION_PER_BOARD_MAX_ACTIVE_TOKENS, BOARD_TOKEN_CAPACITY);
        assert!(EXPECTED_LAYER_MAXIMA[4] <= TOKEN_CAPACITY);
        assert_eq!(
            EXPECTED_TYPE_TOKEN_TOTALS.iter().sum::<usize>(),
            EXPECTED_ACTIVE_TOKENS
        );
    }

    #[test]
    fn live_r2_map_capacity_is_the_rules_complete_23_tile_bound() {
        assert_eq!(R2_MAP_MAX_LEGAL_FRONTIER_TOKENS, 50);
        assert_eq!(R2_MAP_MAX_LEGAL_HABITAT_COMPONENT_TOKENS, 46);
        assert_eq!(R2_MAP_MAX_LEGAL_WILDLIFE_MOTIF_TOKENS, 20);
        assert_eq!(R2_MAP_BOARD_TOKEN_CAPACITY, 139);
        assert_eq!(R2_MAP_TOKEN_CAPACITY, 4 * 139);
        for occupied in 3..=MAX_BOARD_TILES {
            let turns = occupied - 3;
            let upper_bound = occupied + (2 * occupied + 4) + 2 * occupied + turns;
            assert_eq!(upper_bound, 6 * occupied + 1);
            assert!(upper_bound <= R2_MAP_BOARD_TOKEN_CAPACITY);
        }
        assert_eq!(6 * MAX_BOARD_TILES + 1, R2_MAP_BOARD_TOKEN_CAPACITY);
    }

    fn capacity_boundary_state(motif_count: usize) -> SparseBoardState {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(0x5232_4d41_505f_4341),
        )
        .unwrap();
        let mut state = SparseBoardState::from_board(0, &game.boards()[0]).unwrap();
        let occupied = state.occupied_tiles[0].clone();
        state.occupied_tiles.resize(23, occupied);
        let frontier = state.legal_frontier[0].clone();
        state.legal_frontier.resize(50, frontier);
        let component = state.habitat_components[0].clone();
        state.habitat_components.resize(46, component);
        state.wildlife_motifs = (0..motif_count)
            .map(|index| WildlifeMotifToken {
                relative_seat: 0,
                coord: AxialCoord::new(index as i16 - 10, 0),
                wildlife: Wildlife::Bear,
                neighbor_wildlife: [None; 6],
                adjacent_wildlife_counts: [0; 5],
                same_species_neighbor_bits: 0,
            })
            .collect();
        state
    }

    fn capacity_boundary_parent_encoded() -> EncodedBoardTokens {
        let mut token_types = vec![0; R2_MAP_BOARD_TOKEN_CAPACITY];
        token_types[..23].fill(TOKEN_TYPE_OCCUPIED);
        token_types[23..73].fill(TOKEN_TYPE_FRONTIER);
        token_types[73..119].fill(TOKEN_TYPE_COMPONENT);
        EncodedBoardTokens {
            token_types,
            token_payload: vec![0; R2_MAP_BOARD_TOKEN_CAPACITY * TOKEN_PAYLOAD_WIDTH],
            type_counts: [23, 50, 46, 0],
        }
    }

    #[test]
    fn wildlife_suffix_uses_the_last_legal_slot_under_every_d6_without_truncation() {
        let mut tile_parent = capacity_boundary_state(0);
        tile_parent.wildlife_motifs.clear();
        let parent_encoded = capacity_boundary_parent_encoded();
        let sibling = capacity_boundary_state(20);
        for transform in D6Transform::ALL {
            let encoded =
                encode_wildlife_sibling_tokens(&tile_parent, &parent_encoded, &sibling, transform)
                    .unwrap();
            assert_eq!(encoded.type_counts, [23, 50, 46, 20]);
            assert_eq!(encoded.token_types.len(), 139);
            assert_eq!(encoded.token_types[138], TOKEN_TYPE_MOTIF);
            assert_eq!(
                [
                    encoded
                        .token_types
                        .iter()
                        .filter(|value| **value == TOKEN_TYPE_OCCUPIED)
                        .count(),
                    encoded
                        .token_types
                        .iter()
                        .filter(|value| **value == TOKEN_TYPE_FRONTIER)
                        .count(),
                    encoded
                        .token_types
                        .iter()
                        .filter(|value| **value == TOKEN_TYPE_COMPONENT)
                        .count(),
                    encoded
                        .token_types
                        .iter()
                        .filter(|value| **value == TOKEN_TYPE_MOTIF)
                        .count(),
                ],
                [23, 50, 46, 20]
            );
            let final_payload =
                &encoded.token_payload[138 * TOKEN_PAYLOAD_WIDTH..139 * TOKEN_PAYLOAD_WIDTH];
            assert_eq!(final_payload[2], Wildlife::Bear as i8);
            assert!(final_payload.iter().any(|value| *value != 0));
        }
    }

    #[test]
    fn wildlife_suffix_rejects_the_first_impossible_slot_instead_of_truncating() {
        let tile_parent = capacity_boundary_state(0);
        let parent_encoded = capacity_boundary_parent_encoded();
        let impossible = capacity_boundary_state(21);
        let error = encode_wildlife_sibling_tokens(
            &tile_parent,
            &parent_encoded,
            &impossible,
            D6Transform::IDENTITY,
        )
        .unwrap_err();
        assert!(error.to_string().contains("total=140, capacity=139"));
    }

    #[test]
    fn graph_relations_fit_the_frozen_schema() {
        assert_eq!(REL_MOTIF_NEIGHBOR as usize + 1, GRAPH_RELATION_COUNT);
        assert_eq!(GRAPH_MAX_DEGREE, 24);
    }

    #[test]
    fn encoded_d6_transform_is_exactly_invertible_without_slot_reordering() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(91_001),
        )
        .unwrap();
        let record = PositionRecord::observe(&game, 91_001);
        let state = SparsePublicState::from_position_record(&record, None).unwrap();
        let encoded = encode_sparse_state(&state).unwrap();
        for transform in D6Transform::ALL {
            let transformed = transform_encoded_state(&encoded, transform).unwrap();
            let restored = transform_encoded_state(&transformed, transform.inverse()).unwrap();
            assert_eq!(restored, encoded);
        }
    }

    #[test]
    fn authoritative_live_encoder_matches_independent_board_encoder_for_all_d6() {
        let mut game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(0x5232_4d41_505f_5041),
        )
        .unwrap();
        let (prelude, staged) = game.preview_free_three_of_a_kind_if_feasible().unwrap();
        let action = staged
            .legal_turn_actions(&prelude)
            .unwrap()
            .into_iter()
            .find(|action| action.wildlife.is_some())
            .unwrap();
        game.apply(&action).unwrap();
        let record = PositionRecord::observe(&game, 0x5232_4d41_505f_5041);
        let state = SparsePublicState::from_position_record(&record, None).unwrap();

        for transform in D6Transform::ALL {
            let authoritative = encode_r2_map_state_authoritative(&state, transform).unwrap();
            let compact = encode_compact_r2_map_tokens(&state, transform).unwrap();
            let mut compact_cursor = 0;
            for relative_seat in 0..BOARD_SLOTS {
                let board = SparseBoardState::from_sparse_public_state(&state, relative_seat as u8);
                let independent = encode_sparse_board_tokens(&board, transform).unwrap();
                let start = relative_seat * R2_MAP_BOARD_TOKEN_CAPACITY;
                let end = start + R2_MAP_BOARD_TOKEN_CAPACITY;
                assert_eq!(
                    authoritative.token_types[start..end],
                    independent.token_types
                );
                assert_eq!(
                    authoritative.token_payload
                        [start * TOKEN_PAYLOAD_WIDTH..end * TOKEN_PAYLOAD_WIDTH],
                    independent.token_payload
                );
                assert_eq!(
                    authoritative.board_type_counts[relative_seat],
                    independent.type_counts
                );
                let active = independent
                    .type_counts
                    .iter()
                    .map(|count| usize::from(*count))
                    .sum::<usize>();
                assert_eq!(
                    compact.token_types[compact_cursor..compact_cursor + active],
                    independent.token_types[..active]
                );
                assert_eq!(
                    compact.token_payload[compact_cursor * TOKEN_PAYLOAD_WIDTH
                        ..(compact_cursor + active) * TOKEN_PAYLOAD_WIDTH],
                    independent.token_payload[..active * TOKEN_PAYLOAD_WIDTH]
                );
                compact_cursor += active;
            }
            assert_eq!(compact_cursor, compact.token_types.len());
            assert_eq!(compact.token_seats.len(), compact_cursor);
            assert_eq!(compact.board_type_counts, authoritative.board_type_counts);
        }
    }
}
