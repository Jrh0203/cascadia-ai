//! Streaming, replay-authoritative R2-MAP dataset bridge.
//!
//! The compact `.r2sh` replay is the durable dataset. This module validates
//! each shard, regenerates exact R2 tensors lazily, and writes checksummed
//! frames to a caller-provided stream. It never creates a persistent padded
//! tensor cache.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufRead, Write},
    path::{Path, PathBuf},
};

use blake3::Hasher;
use cascadia_data::{
    GRADED_ORACLE_ACTION_FEATURE_SIZE, GradedOracleActionFeatures, PositionRecord,
    PublicActionRecord, R2MapCollectionKind, R2MapGameRecord, R2MapPrimaryExample,
    focal_seat_for_game, r2_map_ordered_action_ids_blake3, read_r2_map_shard,
    read_r2_map_shard_after_semantic_validation,
};
use cascadia_game::{
    D6Transform, GameState, MarketDecisionSession, MarketDecisionStage, MarketSlot, WildlifeWipe,
    public_market_action_identity, score_board,
};
use serde::{Deserialize, Serialize};

use crate::{
    R2Error, Result, SparsePublicState, TOKEN_PAYLOAD_WIDTH, encode_global_features,
    encode_market_features, encode_player_features,
    mlx_export::{EncodedR2MapCompactTokens, encode_compact_r2_map_tokens},
};

pub const R2_MAP_DATASET_SCHEMA_VERSION: u16 = 3;
pub const R2_MAP_DATASET_PROTOCOL_ID: &str = "r2-map-streaming-exact-r2-v3";
pub const R2_MAP_DATASET_MAGIC: &[u8; 8] = b"CSDR2MP\0";
pub const R2_MAP_DATASET_HEADER_SIZE: u16 = 120;
pub const R2_MAP_DATASET_FRAME_HEADER_SIZE: usize = 36;
pub const R2_MAP_DATASET_FRAME_PREFIX_SIZE: usize = 4;
pub const R2_MAP_DATASET_DRAFT_FRAME_KIND: u8 = 0;
pub const R2_MAP_DATASET_MARKET_FRAME_KIND: u8 = 1;
pub const R2_MAP_DATASET_FRAME_VERSION: u8 = 2;
pub const R2_MAP_DATASET_MARKET_FIXED_SIZE: usize = 272;
pub const R2_MAP_DATASET_ACTION_BYTES: usize = GRADED_ORACLE_ACTION_FEATURE_SIZE;
pub const R2_MAP_DATASET_SPLIT_DOMAIN: &str = "r2-map-whole-game-split-v1";
pub const R2_MAP_DATASET_D6_DOMAIN: &str = "r2-map-d6-cyclic-offset-v1";
pub const R2_MAP_DRAFT_IMITATION_SUBSET_ID: &str = "r2-map-draft-imitation-subset-v1";
pub const R2_MAP_DRAFT_IMITATION_SUBSET_PARTS_PER_MILLION: u32 = 10_000;
pub const R2_MAP_DATASET_OPPONENT_WIPE_MAX: usize = 20;
pub const R2_MAP_DATASET_OPPONENT_TARGET_SIZE: usize = 5 + 1 + R2_MAP_DATASET_OPPONENT_WIPE_MAX;
pub const R2_MAP_DATASET_VALIDATION_BUCKETS: u8 = 10;
pub const R2_MAP_DATASET_VALIDATION_BUCKET: u8 = 0;
pub const R2_MAP_PACKED_PIPE_PROTOCOL_ID: &str = "r2-map-packed-batch-pipe-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum R2MapDatasetMode {
    Train,
    Validation,
    FixedPanel,
}

impl R2MapDatasetMode {
    const fn code(self) -> u8 {
        match self {
            Self::Train => 0,
            Self::Validation => 1,
            Self::FixedPanel => 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct R2MapDatasetSource {
    pub file_name: String,
    pub bytes: u64,
    pub blake3: String,
    pub first_game_index: u64,
    pub next_game_index: u64,
    pub game_count: usize,
    pub example_count: usize,
    pub imitation_example_count: usize,
    pub market_decision_count: usize,
    pub market_policy_target_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct R2MapDatasetRoundIdentity {
    pub campaign_id: String,
    pub iteration: u32,
    pub collection_kind: R2MapCollectionKind,
    pub newest_checkpoint_blake3: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct R2MapDatasetManifest {
    pub schema_version: u16,
    pub protocol_id: String,
    pub feature_schema: String,
    pub target_schema: String,
    pub split_schema: String,
    pub d6_schema: String,
    pub imitation_subset_schema: String,
    pub imitation_subset_parts_per_million: u32,
    pub round: R2MapDatasetRoundIdentity,
    pub dataset_blake3: String,
    pub game_count: usize,
    pub example_count: usize,
    pub imitation_example_count: usize,
    pub market_decision_count: usize,
    pub market_policy_target_count: usize,
    pub train_games: usize,
    pub validation_games: usize,
    pub sources: Vec<R2MapDatasetSource>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct R2MapDatasetStreamConfig {
    pub mode: R2MapDatasetMode,
    pub epoch: u64,
    pub sampler_seed: u64,
    pub fixed_panel_games: usize,
    /// Optional receipt-bound subset of whole games for one bounded window.
    pub game_indices: Vec<u64>,
}

impl R2MapDatasetStreamConfig {
    pub fn train(epoch: u64, sampler_seed: u64) -> Self {
        Self {
            mode: R2MapDatasetMode::Train,
            epoch,
            sampler_seed,
            fixed_panel_games: 0,
            game_indices: Vec::new(),
        }
    }

    pub fn validation() -> Self {
        Self {
            mode: R2MapDatasetMode::Validation,
            epoch: 0,
            sampler_seed: 0,
            fixed_panel_games: 0,
            game_indices: Vec::new(),
        }
    }

    pub fn fixed_panel(games: usize) -> Self {
        Self {
            mode: R2MapDatasetMode::FixedPanel,
            epoch: 0,
            sampler_seed: 0,
            fixed_panel_games: games,
            game_indices: Vec::new(),
        }
    }

    fn validate(&self) -> Result<()> {
        if self.mode == R2MapDatasetMode::FixedPanel && self.fixed_panel_games == 0 {
            return invalid("fixed-panel mode requires a nonzero game count");
        }
        if self.mode != R2MapDatasetMode::FixedPanel && self.fixed_panel_games != 0 {
            return invalid("fixed_panel_games is only valid in fixed-panel mode");
        }
        if self.mode != R2MapDatasetMode::Train && (self.epoch != 0 || self.sampler_seed != 0) {
            return invalid("validation and fixed-panel transforms are frozen at identity");
        }
        let unique = self.game_indices.iter().copied().collect::<BTreeSet<_>>();
        if unique.len() != self.game_indices.len() {
            return invalid("bounded R2-MAP window repeats a game index");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct R2MapDatasetStreamReceipt {
    pub schema_version: u16,
    pub protocol_id: String,
    pub dataset_blake3: String,
    pub config_blake3: String,
    pub frames: usize,
    pub draft_frames: usize,
    pub imitation_draft_frames: usize,
    pub selected_only_draft_frames: usize,
    pub draft_candidates: usize,
    pub market_frames: usize,
    pub market_policy_target_frames: usize,
    pub games: usize,
    pub stream_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct R2MapPackedBatchProducerConfig {
    pub mode: R2MapDatasetMode,
    pub epoch: u64,
    pub sampler_seed: u64,
    pub group_batch_size: usize,
    pub maximum_candidates_per_batch: usize,
    pub bootstrap_value_only: bool,
    pub ordered_game_indices: Vec<u64>,
    pub start_game_offset: usize,
    pub start_turn_offset: usize,
    pub start_batch_index: u64,
}

impl R2MapPackedBatchProducerConfig {
    fn validate(&self) -> Result<()> {
        if self.group_batch_size == 0
            || self.maximum_candidates_per_batch == 0
            || self.ordered_game_indices.is_empty()
            || self.start_game_offset >= self.ordered_game_indices.len()
            || self.start_turn_offset >= 20
        {
            return invalid("packed-batch producer configuration is invalid");
        }
        if self.mode == R2MapDatasetMode::FixedPanel
            || (self.mode == R2MapDatasetMode::Validation
                && (self.epoch != 0 || self.sampler_seed != 0))
        {
            return invalid("packed-batch producer mode or transform is invalid");
        }
        let unique = self
            .ordered_game_indices
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if unique.len() != self.ordered_game_indices.len() {
            return invalid("packed-batch producer repeats a game index");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct PackedBatchIdentity<'a> {
    protocol_id: &'static str,
    source_blake3: &'a str,
    epoch: u64,
    sampler_seed: u64,
    batch_index: u64,
    first_game_offset: usize,
    first_turn_offset: usize,
    next_game_offset: usize,
    next_turn_offset: usize,
    groups: usize,
    padded_width: usize,
    payload_bytes: usize,
    payload_blake3: &'a str,
}

/// Produce exact focal-seat packed batches over a backpressured stdin/stdout pipe.
///
/// Each output batch is a self-contained in-memory R2-MAP stream.  The producer
/// waits for an identity acknowledgement before constructing the next batch, so
/// a consumer crash cannot advance durable cursor state or accumulate expanded
/// windows.  Bootstrap focal seat is always `global_game_index mod 4`.
pub fn serve_r2_map_packed_batches<R: BufRead, W: Write>(
    shard: &Path,
    config: &R2MapPackedBatchProducerConfig,
    mut acknowledgements: R,
    mut output: W,
) -> Result<()> {
    config.validate()?;
    let metadata = build_r2_map_compact_index_metadata(&[shard.to_path_buf()])?;
    let manifest = metadata.dataset_manifest;
    let source = manifest
        .sources
        .first()
        .ok_or_else(|| R2Error::DatasetContract("packed producer source is empty".to_owned()))?;
    let records = read_r2_map_shard_after_semantic_validation(shard)?;
    let mut by_index = records
        .into_iter()
        .map(|record| (record.identity.global_game_index, record))
        .collect::<BTreeMap<_, _>>();
    if by_index.len() != source.game_count {
        return invalid("packed producer source game identity differs");
    }
    let mut ordered = Vec::with_capacity(config.ordered_game_indices.len());
    for game_index in &config.ordered_game_indices {
        let record = by_index
            .remove(game_index)
            .ok_or_else(|| R2Error::DatasetContract("packed producer game is absent".to_owned()))?;
        if game_is_validation(record.identity.game_id)
            != (config.mode == R2MapDatasetMode::Validation)
        {
            return invalid("packed producer game differs from the requested split");
        }
        ordered.push(record);
    }
    let producer_identity =
        canonical_hash(&(R2_MAP_PACKED_PIPE_PROTOCOL_ID, &source.blake3, config))?;
    write_json_line(
        &mut output,
        &serde_json::json!({
            "type": "ready",
            "protocol_id": R2_MAP_PACKED_PIPE_PROTOCOL_ID,
            "producer_identity": producer_identity,
            "source": source,
            "manifest": manifest,
            "mode": config.mode,
            "epoch": config.epoch,
            "sampler_seed": config.sampler_seed,
            "start_game_offset": config.start_game_offset,
            "start_turn_offset": config.start_turn_offset,
            "start_batch_index": config.start_batch_index,
            "ordered_game_count": ordered.len(),
            "focal_seat_rule": "global-game-index-mod-4",
            "bootstrap_value_only": config.bootstrap_value_only,
        }),
    )?;

    let mut game_offset = config.start_game_offset;
    let mut turn_offset = config.start_turn_offset;
    let mut batch_index = config.start_batch_index;
    while game_offset < ordered.len() {
        let first_game_offset = game_offset;
        let first_turn_offset = turn_offset;
        let mut positions = Vec::new();
        let mut padded_width = 0usize;
        while positions.len() < config.group_batch_size && game_offset < ordered.len() {
            let record = &ordered[game_offset];
            let focal = focal_seat_for_game(record.identity.global_game_index);
            let decision_index = usize::from(focal) + turn_offset * 4;
            let decision = record.decisions.get(decision_index).ok_or_else(|| {
                R2Error::DatasetContract("packed producer focal turn is absent".to_owned())
            })?;
            let width = if !config.bootstrap_value_only
                && draft_is_imitation_subset(record.collection_kind, decision.draft_decision_id)
            {
                usize::try_from(decision.draft_legal_action_count).map_err(|_| {
                    R2Error::DatasetContract("packed producer width exceeds usize".to_owned())
                })?
            } else {
                1
            };
            let next_width = padded_width.max(width);
            if !packed_batch_can_admit(
                padded_width,
                positions.len(),
                width,
                config.maximum_candidates_per_batch,
            ) {
                break;
            }
            positions.push((game_offset, turn_offset));
            padded_width = next_width;
            turn_offset += 1;
            if turn_offset == 20 {
                game_offset += 1;
                turn_offset = 0;
            }
            if positions.len() == 1 && padded_width > config.maximum_candidates_per_batch {
                break;
            }
        }
        if positions.is_empty() {
            return invalid("packed producer selected an empty batch");
        }

        let mut examples_by_game = BTreeMap::new();
        for (position_game, _) in &positions {
            if !examples_by_game.contains_key(position_game) {
                let record = &ordered[*position_game];
                let focal = focal_seat_for_game(record.identity.global_game_index);
                examples_by_game.insert(*position_game, record.extract_focal_seat_examples(focal)?);
            }
        }
        let mut payloads = Vec::new();
        let mut draft_candidates = 0usize;
        let mut game_indices = Vec::new();
        for (position_game, position_turn) in &positions {
            let record = &ordered[*position_game];
            let example = &examples_by_game[position_game][*position_turn];
            let decision = &record.decisions[usize::from(example.turn_index)];
            let transform = transform_id(
                &R2MapDatasetStreamConfig {
                    mode: config.mode,
                    epoch: config.epoch,
                    sampler_seed: config.sampler_seed,
                    fixed_panel_games: 0,
                    game_indices: Vec::new(),
                },
                example.game_id,
                decision.draft_decision_id,
            );
            payloads.extend(encode_market_examples(record, example, transform)?);
            let draft = encode_example(record, example, transform, !config.bootstrap_value_only)?;
            draft_candidates += draft.candidate_count;
            payloads.push(draft.payload);
            if game_indices.last() != Some(&record.identity.global_game_index) {
                game_indices.push(record.identity.global_game_index);
            }
        }
        let batch_stream_config = R2MapDatasetStreamConfig {
            mode: config.mode,
            epoch: config.epoch,
            sampler_seed: config.sampler_seed,
            fixed_panel_games: 0,
            game_indices: game_indices.clone(),
        };
        let config_blake3 = canonical_hash(&(
            R2_MAP_DATASET_PROTOCOL_ID,
            &manifest.dataset_blake3,
            &batch_stream_config,
        ))?;
        let mut stream = Vec::new();
        write_stream_header(
            &mut stream,
            &manifest.dataset_blake3,
            &config_blake3,
            &batch_stream_config,
            payloads.len(),
            game_indices.len(),
        )?;
        let mut stream_hasher = Hasher::new();
        for payload in payloads {
            write_dataset_frame(&mut stream, &mut stream_hasher, &payload)?;
        }
        let payload_blake3 = blake3::hash(&stream).to_hex().to_string();
        let identity = PackedBatchIdentity {
            protocol_id: R2_MAP_PACKED_PIPE_PROTOCOL_ID,
            source_blake3: &source.blake3,
            epoch: config.epoch,
            sampler_seed: config.sampler_seed,
            batch_index,
            first_game_offset,
            first_turn_offset,
            next_game_offset: game_offset,
            next_turn_offset: turn_offset,
            groups: positions.len(),
            padded_width,
            payload_bytes: stream.len(),
            payload_blake3: &payload_blake3,
        };
        let batch_identity = canonical_hash(&identity)?;
        write_json_line(
            &mut output,
            &serde_json::json!({
                "type": "batch",
                "protocol_id": R2_MAP_PACKED_PIPE_PROTOCOL_ID,
                "producer_identity": producer_identity,
                "batch_identity": batch_identity,
                "batch_index": batch_index,
                "first_game_offset": first_game_offset,
                "first_turn_offset": first_turn_offset,
                "next_game_offset": game_offset,
                "next_turn_offset": turn_offset,
                "groups": positions.len(),
                "padded_width": padded_width,
                "draft_candidates": draft_candidates,
                "game_indices": game_indices,
                "payload_bytes": stream.len(),
                "payload_blake3": payload_blake3,
            }),
        )?;
        output.write_all(&stream)?;
        output.flush()?;

        let mut acknowledgement = String::new();
        if acknowledgements.read_line(&mut acknowledgement)? == 0 {
            return invalid("packed producer consumer closed before acknowledgement");
        }
        let acknowledgement: serde_json::Value = serde_json::from_str(&acknowledgement)?;
        if acknowledgement.as_object().map(|value| value.len()) != Some(1)
            || acknowledgement.get("ack").and_then(|value| value.as_str()) != Some(&batch_identity)
        {
            return invalid("packed producer acknowledgement identity differs");
        }
        batch_index += 1;
    }
    write_json_line(
        &mut output,
        &serde_json::json!({
            "type": "done",
            "protocol_id": R2_MAP_PACKED_PIPE_PROTOCOL_ID,
            "producer_identity": producer_identity,
            "next_game_offset": game_offset,
            "next_turn_offset": turn_offset,
            "next_batch_index": batch_index,
        }),
    )?;
    Ok(())
}

fn packed_batch_can_admit(
    padded_width: usize,
    groups: usize,
    candidate_width: usize,
    maximum_candidates_per_batch: usize,
) -> bool {
    groups == 0
        || padded_width
            .max(candidate_width)
            .checked_mul(groups + 1)
            .is_some_and(|value| value <= maximum_candidates_per_batch)
}

fn write_json_line<W: Write>(writer: &mut W, value: &serde_json::Value) -> Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

/// Compact-index metadata recovered from replay shards that have already
/// passed the collector's full semantic validation gate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct R2MapCompactIndexGameMetadata {
    pub source_file_name: String,
    pub source_blake3: String,
    pub global_game_index: u64,
    pub game_id: String,
    pub example_count: usize,
    pub imitation_example_count: usize,
    pub market_decision_count: usize,
    pub market_policy_target_count: usize,
    pub split: String,
    pub candidate_widths: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct R2MapCompactIndexMetadata {
    pub schema_version: u16,
    pub schema_id: String,
    pub dataset_manifest: R2MapDatasetManifest,
    pub games: Vec<R2MapCompactIndexGameMetadata>,
}

#[derive(Debug, Clone, Serialize)]
struct ManifestIdentity<'a> {
    schema_version: u16,
    protocol_id: &'a str,
    feature_schema: &'a str,
    target_schema: &'a str,
    split_schema: &'a str,
    d6_schema: &'a str,
    imitation_subset_schema: &'a str,
    imitation_subset_parts_per_million: u32,
    round: &'a R2MapDatasetRoundIdentity,
    game_count: usize,
    example_count: usize,
    imitation_example_count: usize,
    market_decision_count: usize,
    market_policy_target_count: usize,
    train_games: usize,
    validation_games: usize,
    sources: &'a [R2MapDatasetSource],
}

/// Validate source shards and bind their content and whole-game split into one
/// order-independent dataset identity.
pub fn build_r2_map_dataset_manifest(paths: &[PathBuf]) -> Result<R2MapDatasetManifest> {
    if paths.is_empty() {
        return invalid("R2-MAP dataset requires at least one .r2sh shard");
    }
    let mut sources = Vec::with_capacity(paths.len());
    let mut seen_games = BTreeSet::new();
    let mut game_count = 0usize;
    let mut example_count = 0usize;
    let mut imitation_example_count = 0usize;
    let mut market_decision_count = 0usize;
    let mut market_policy_target_count = 0usize;
    let mut train_games = 0usize;
    let mut validation_games = 0usize;
    let mut round: Option<R2MapDatasetRoundIdentity> = None;
    for path in paths {
        let records = read_r2_map_shard(path)?;
        let bytes = fs::metadata(path)?.len();
        let file_bytes = fs::read(path)?;
        let first = records
            .first()
            .ok_or_else(|| R2Error::DatasetContract("validated R2 shard is empty".to_owned()))?;
        let last = records.last().expect("validated R2 shard is nonempty");
        let mut source_examples = 0usize;
        let mut source_imitation_examples = 0usize;
        let mut source_market_decisions = 0usize;
        let mut source_market_policy_targets = 0usize;
        for record in &records {
            let observed_round = R2MapDatasetRoundIdentity {
                campaign_id: record.identity.campaign_id.clone(),
                iteration: record.identity.iteration,
                collection_kind: record.collection_kind,
                newest_checkpoint_blake3: record
                    .focal_checkpoint_hash
                    .map(|value| blake3::Hash::from_bytes(value).to_hex().to_string()),
            };
            if let Some(expected) = &round {
                if expected != &observed_round {
                    return invalid(
                        "R2-MAP source shards mix campaign, iteration, collection, or newest checkpoint",
                    );
                }
            } else {
                round = Some(observed_round);
            }
            if !seen_games.insert(record.identity.game_id) {
                return invalid("R2-MAP source shards repeat a game identity");
            }
            let examples = record.extract_primary_examples()?;
            let count = examples.len();
            source_examples = source_examples.checked_add(count).ok_or_else(|| {
                R2Error::DatasetContract("R2-MAP example count overflow".to_owned())
            })?;
            for example in examples {
                let decision = record
                    .decisions
                    .get(usize::from(example.turn_index))
                    .ok_or_else(|| {
                        R2Error::DatasetContract(
                            "R2-MAP example turn is absent from its replay".to_owned(),
                        )
                    })?;
                if draft_is_imitation_subset(record.collection_kind, decision.draft_decision_id) {
                    source_imitation_examples =
                        source_imitation_examples.checked_add(1).ok_or_else(|| {
                            R2Error::DatasetContract(
                                "R2-MAP imitation example count overflow".to_owned(),
                            )
                        })?;
                }
                let decisions = decision.market_decisions.len();
                source_market_decisions = source_market_decisions
                    .checked_add(decisions)
                    .ok_or_else(|| {
                        R2Error::DatasetContract("R2-MAP market decision count overflow".to_owned())
                    })?;
                if record.collection_kind == R2MapCollectionKind::Bootstrap {
                    source_market_policy_targets = source_market_policy_targets
                        .checked_add(decisions)
                        .ok_or_else(|| {
                            R2Error::DatasetContract(
                                "R2-MAP market policy-target count overflow".to_owned(),
                            )
                        })?;
                }
            }
            if game_is_validation(record.identity.game_id) {
                validation_games += 1;
            } else {
                train_games += 1;
            }
        }
        game_count += records.len();
        example_count += source_examples;
        imitation_example_count += source_imitation_examples;
        market_decision_count += source_market_decisions;
        market_policy_target_count += source_market_policy_targets;
        sources.push(R2MapDatasetSource {
            file_name: path
                .file_name()
                .and_then(|value| value.to_str())
                .ok_or_else(|| {
                    R2Error::DatasetContract("R2 shard file name is not UTF-8".to_owned())
                })?
                .to_owned(),
            bytes,
            blake3: blake3::hash(&file_bytes).to_hex().to_string(),
            first_game_index: first.identity.global_game_index,
            next_game_index: last
                .identity
                .global_game_index
                .checked_add(1)
                .ok_or_else(|| R2Error::DatasetContract("R2 game range overflow".to_owned()))?,
            game_count: records.len(),
            example_count: source_examples,
            imitation_example_count: source_imitation_examples,
            market_decision_count: source_market_decisions,
            market_policy_target_count: source_market_policy_targets,
        });
    }
    sources.sort_by_key(|source| (source.first_game_index, source.blake3.clone()));
    for pair in sources.windows(2) {
        if pair[0].next_game_index > pair[1].first_game_index {
            return invalid("R2-MAP source shard game-index ranges overlap");
        }
    }
    let identity = ManifestIdentity {
        schema_version: R2_MAP_DATASET_SCHEMA_VERSION,
        protocol_id: R2_MAP_DATASET_PROTOCOL_ID,
        feature_schema: "exact-r2-staged-public-market-and-draft-v3",
        target_schema: "selected-value-plus-deterministic-full-draft-imitation-v3",
        split_schema: R2_MAP_DATASET_SPLIT_DOMAIN,
        d6_schema: R2_MAP_DATASET_D6_DOMAIN,
        imitation_subset_schema: R2_MAP_DRAFT_IMITATION_SUBSET_ID,
        imitation_subset_parts_per_million: R2_MAP_DRAFT_IMITATION_SUBSET_PARTS_PER_MILLION,
        round: round
            .as_ref()
            .ok_or_else(|| R2Error::DatasetContract("R2-MAP dataset has no round".to_owned()))?,
        game_count,
        example_count,
        imitation_example_count,
        market_decision_count,
        market_policy_target_count,
        train_games,
        validation_games,
        sources: &sources,
    };
    let dataset_blake3 = canonical_hash(&identity)?;
    Ok(R2MapDatasetManifest {
        schema_version: identity.schema_version,
        protocol_id: identity.protocol_id.to_owned(),
        feature_schema: identity.feature_schema.to_owned(),
        target_schema: identity.target_schema.to_owned(),
        split_schema: identity.split_schema.to_owned(),
        d6_schema: identity.d6_schema.to_owned(),
        imitation_subset_schema: identity.imitation_subset_schema.to_owned(),
        imitation_subset_parts_per_million: identity.imitation_subset_parts_per_million,
        round: identity.round.clone(),
        dataset_blake3,
        game_count,
        example_count,
        imitation_example_count,
        market_decision_count,
        market_policy_target_count,
        train_games,
        validation_games,
        sources,
    })
}

/// Build the exact compact-index manifest, per-game split metadata, and
/// candidate widths without regenerating expanded exact-R2 tensors.
///
/// This path is only for shards already admitted by the aggregate collector
/// receipt. It rechecks framed shard integrity and all identity/count fields
/// used by the index, while avoiding the intentionally expensive second
/// replay/materialization pass.
pub fn build_r2_map_compact_index_metadata(paths: &[PathBuf]) -> Result<R2MapCompactIndexMetadata> {
    if paths.is_empty() {
        return invalid("compact-index metadata requires source shards");
    }
    let mut seen_games = BTreeSet::new();
    let mut game_count = 0usize;
    let mut example_count = 0usize;
    let mut imitation_example_count = 0usize;
    let mut market_decision_count = 0usize;
    let mut market_policy_target_count = 0usize;
    let mut train_games = 0usize;
    let mut validation_games = 0usize;
    let mut round: Option<R2MapDatasetRoundIdentity> = None;
    let mut sources = Vec::new();
    let mut games = Vec::new();
    for path in paths {
        let records = read_r2_map_shard_after_semantic_validation(path)?;
        let bytes = fs::metadata(path)?.len();
        let file_bytes = fs::read(path)?;
        let source_blake3 = blake3::hash(&file_bytes).to_hex().to_string();
        let source_file_name = path
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| R2Error::DatasetContract("R2 shard file name is not UTF-8".to_owned()))?
            .to_owned();
        let first = records
            .first()
            .ok_or_else(|| R2Error::DatasetContract("validated R2 shard is empty".to_owned()))?;
        let last = records.last().expect("validated R2 shard is nonempty");
        let mut source_examples = 0usize;
        let mut source_imitation_examples = 0usize;
        let mut source_market_decisions = 0usize;
        let mut source_market_policy_targets = 0usize;
        for record in &records {
            let observed_round = R2MapDatasetRoundIdentity {
                campaign_id: record.identity.campaign_id.clone(),
                iteration: record.identity.iteration,
                collection_kind: record.collection_kind,
                newest_checkpoint_blake3: record
                    .focal_checkpoint_hash
                    .map(|value| blake3::Hash::from_bytes(value).to_hex().to_string()),
            };
            if let Some(expected) = &round {
                if expected != &observed_round {
                    return invalid(
                        "R2-MAP source shards mix campaign, iteration, collection, or newest checkpoint",
                    );
                }
            } else {
                round = Some(observed_round);
            }
            if !seen_games.insert(record.identity.game_id) {
                return invalid("R2-MAP source shards repeat a game identity");
            }
            let retain_all = record.collection_kind == R2MapCollectionKind::Bootstrap;
            let selected = record
                .decisions
                .iter()
                .filter(|decision| retain_all || decision.seat == record.focal_seat)
                .collect::<Vec<_>>();
            let expected_examples = if retain_all { 80 } else { 20 };
            if selected.len() != expected_examples {
                return invalid("compact-index replay has the wrong primary-decision count");
            }
            let mut widths = Vec::with_capacity(selected.len());
            let mut game_imitation = 0usize;
            let mut game_market_decisions = 0usize;
            for decision in selected {
                let imitation =
                    draft_is_imitation_subset(record.collection_kind, decision.draft_decision_id);
                let width = if imitation {
                    usize::try_from(decision.draft_legal_action_count).map_err(|_| {
                        R2Error::DatasetContract(
                            "R2-MAP legal action count exceeds usize".to_owned(),
                        )
                    })?
                } else {
                    1
                };
                if width == 0 {
                    return invalid("compact-index replay has an empty legal action set");
                }
                widths.push(width);
                game_imitation += usize::from(imitation);
                game_market_decisions = game_market_decisions
                    .checked_add(decision.market_decisions.len())
                    .ok_or_else(|| {
                        R2Error::DatasetContract("R2-MAP market decision count overflow".to_owned())
                    })?;
            }
            let game_market_policy_targets = if retain_all { game_market_decisions } else { 0 };
            let validation = game_is_validation(record.identity.game_id);
            train_games += usize::from(!validation);
            validation_games += usize::from(validation);
            source_examples += expected_examples;
            source_imitation_examples += game_imitation;
            source_market_decisions += game_market_decisions;
            source_market_policy_targets += game_market_policy_targets;
            games.push(R2MapCompactIndexGameMetadata {
                source_file_name: source_file_name.clone(),
                source_blake3: source_blake3.clone(),
                global_game_index: record.identity.global_game_index,
                game_id: blake3::Hash::from_bytes(record.identity.game_id)
                    .to_hex()
                    .to_string(),
                example_count: expected_examples,
                imitation_example_count: game_imitation,
                market_decision_count: game_market_decisions,
                market_policy_target_count: game_market_policy_targets,
                split: if validation { "validation" } else { "train" }.to_owned(),
                candidate_widths: widths,
            });
        }
        game_count += records.len();
        example_count += source_examples;
        imitation_example_count += source_imitation_examples;
        market_decision_count += source_market_decisions;
        market_policy_target_count += source_market_policy_targets;
        sources.push(R2MapDatasetSource {
            file_name: source_file_name,
            bytes,
            blake3: source_blake3,
            first_game_index: first.identity.global_game_index,
            next_game_index: last
                .identity
                .global_game_index
                .checked_add(1)
                .ok_or_else(|| R2Error::DatasetContract("R2 game range overflow".to_owned()))?,
            game_count: records.len(),
            example_count: source_examples,
            imitation_example_count: source_imitation_examples,
            market_decision_count: source_market_decisions,
            market_policy_target_count: source_market_policy_targets,
        });
    }
    sources.sort_by_key(|source| (source.first_game_index, source.blake3.clone()));
    games.sort_by_key(|game| (game.global_game_index, game.game_id.clone()));
    for pair in sources.windows(2) {
        if pair[0].next_game_index > pair[1].first_game_index {
            return invalid("R2-MAP source shard game-index ranges overlap");
        }
    }
    let round =
        round.ok_or_else(|| R2Error::DatasetContract("R2-MAP dataset has no round".to_owned()))?;
    let identity = ManifestIdentity {
        schema_version: R2_MAP_DATASET_SCHEMA_VERSION,
        protocol_id: R2_MAP_DATASET_PROTOCOL_ID,
        feature_schema: "exact-r2-staged-public-market-and-draft-v3",
        target_schema: "selected-value-plus-deterministic-full-draft-imitation-v3",
        split_schema: R2_MAP_DATASET_SPLIT_DOMAIN,
        d6_schema: R2_MAP_DATASET_D6_DOMAIN,
        imitation_subset_schema: R2_MAP_DRAFT_IMITATION_SUBSET_ID,
        imitation_subset_parts_per_million: R2_MAP_DRAFT_IMITATION_SUBSET_PARTS_PER_MILLION,
        round: &round,
        game_count,
        example_count,
        imitation_example_count,
        market_decision_count,
        market_policy_target_count,
        train_games,
        validation_games,
        sources: &sources,
    };
    let dataset_blake3 = canonical_hash(&identity)?;
    let dataset_manifest = R2MapDatasetManifest {
        schema_version: identity.schema_version,
        protocol_id: identity.protocol_id.to_owned(),
        feature_schema: identity.feature_schema.to_owned(),
        target_schema: identity.target_schema.to_owned(),
        split_schema: identity.split_schema.to_owned(),
        d6_schema: identity.d6_schema.to_owned(),
        imitation_subset_schema: identity.imitation_subset_schema.to_owned(),
        imitation_subset_parts_per_million: identity.imitation_subset_parts_per_million,
        round,
        dataset_blake3,
        game_count,
        example_count,
        imitation_example_count,
        market_decision_count,
        market_policy_target_count,
        train_games,
        validation_games,
        sources,
    };
    Ok(R2MapCompactIndexMetadata {
        schema_version: 1,
        schema_id: "r2-map-compact-index-metadata-v1".to_owned(),
        dataset_manifest,
        games,
    })
}

/// Regenerate exact tensors one example at a time and stream checksummed
/// variable-length frames. Input shard order and worker scheduling cannot alter
/// frame order, split assignment, transform IDs, or hashes.
pub fn stream_r2_map_dataset<W: Write>(
    paths: &[PathBuf],
    manifest: &R2MapDatasetManifest,
    config: &R2MapDatasetStreamConfig,
    writer: W,
) -> Result<R2MapDatasetStreamReceipt> {
    stream_r2_map_dataset_inner(paths, manifest, config, writer, false)
}

/// Stream shards whose exact bytes were admitted by an independently verified
/// aggregate receipt and compact index. Framing, checksums, decoding, range,
/// and identity uniqueness are still checked; only duplicate replay/scoring
/// validation is skipped.
pub fn stream_r2_map_dataset_after_semantic_validation<W: Write>(
    paths: &[PathBuf],
    manifest: &R2MapDatasetManifest,
    config: &R2MapDatasetStreamConfig,
    writer: W,
) -> Result<R2MapDatasetStreamReceipt> {
    stream_r2_map_dataset_inner(paths, manifest, config, writer, true)
}

fn stream_r2_map_dataset_inner<W: Write>(
    paths: &[PathBuf],
    manifest: &R2MapDatasetManifest,
    config: &R2MapDatasetStreamConfig,
    mut writer: W,
    semantic_validation_is_receipt_bound: bool,
) -> Result<R2MapDatasetStreamReceipt> {
    config.validate()?;
    let observed = if semantic_validation_is_receipt_bound {
        build_r2_map_compact_index_metadata(paths)?.dataset_manifest
    } else {
        build_r2_map_dataset_manifest(paths)?
    };
    if &observed != manifest {
        return invalid("R2-MAP dataset manifest does not match source shards");
    }
    let config_blake3 =
        canonical_hash(&(R2_MAP_DATASET_PROTOCOL_ID, &manifest.dataset_blake3, config))?;
    let mut records = Vec::new();
    for path in paths {
        records.extend(if semantic_validation_is_receipt_bound {
            read_r2_map_shard_after_semantic_validation(path)?
        } else {
            read_r2_map_shard(path)?
        });
    }
    records.sort_by_key(|record| record.identity.global_game_index);
    let selected_games = select_records(&records, config);
    if !config.game_indices.is_empty() && selected_games.len() != config.game_indices.len() {
        return invalid("bounded R2-MAP window game indices differ from the source split");
    }
    let mut expected_draft_frames = 0usize;
    let mut expected_market_frames = 0usize;
    for record in &selected_games {
        let examples = record.extract_primary_examples()?;
        expected_draft_frames = expected_draft_frames
            .checked_add(examples.len())
            .ok_or_else(|| {
                R2Error::DatasetContract("R2-MAP draft frame count overflow".to_owned())
            })?;
        for example in examples {
            expected_market_frames = expected_market_frames
                .checked_add(
                    record.decisions[usize::from(example.turn_index)]
                        .market_decisions
                        .len(),
                )
                .ok_or_else(|| {
                    R2Error::DatasetContract("R2-MAP market frame count overflow".to_owned())
                })?;
        }
    }
    let expected_frames = expected_draft_frames
        .checked_add(expected_market_frames)
        .ok_or_else(|| R2Error::DatasetContract("R2-MAP stream frame count overflow".to_owned()))?;
    write_stream_header(
        &mut writer,
        &manifest.dataset_blake3,
        &config_blake3,
        config,
        expected_frames,
        selected_games.len(),
    )?;
    let mut stream_hasher = Hasher::new();
    let mut frames = 0usize;
    let mut draft_frames = 0usize;
    let mut imitation_draft_frames = 0usize;
    let mut selected_only_draft_frames = 0usize;
    let mut draft_candidates = 0usize;
    let mut market_frames = 0usize;
    let mut market_policy_target_frames = 0usize;
    for record in selected_games {
        let examples = record.extract_primary_examples()?;
        for example in &examples {
            let turn_record = record
                .decisions
                .get(usize::from(example.turn_index))
                .ok_or_else(|| {
                    R2Error::DatasetContract("R2-MAP example turn is absent".to_owned())
                })?;
            let transform_id = transform_id(config, example.game_id, turn_record.draft_decision_id);
            for payload in encode_market_examples(record, example, transform_id)? {
                write_dataset_frame(&mut writer, &mut stream_hasher, &payload)?;
                frames += 1;
                market_frames += 1;
                if record.collection_kind == R2MapCollectionKind::Bootstrap {
                    market_policy_target_frames += 1;
                }
            }
            let encoded = encode_example(record, example, transform_id, true)?;
            write_dataset_frame(&mut writer, &mut stream_hasher, &encoded.payload)?;
            frames += 1;
            draft_frames += 1;
            draft_candidates = draft_candidates
                .checked_add(encoded.candidate_count)
                .ok_or_else(|| {
                    R2Error::DatasetContract("R2-MAP draft candidate count overflow".to_owned())
                })?;
            if encoded.imitation {
                imitation_draft_frames += 1;
            } else {
                selected_only_draft_frames += 1;
            }
        }
    }
    if frames != expected_frames
        || draft_frames != expected_draft_frames
        || market_frames != expected_market_frames
    {
        return invalid("R2-MAP stream emitted the wrong frame count");
    }
    writer.flush()?;
    Ok(R2MapDatasetStreamReceipt {
        schema_version: R2_MAP_DATASET_SCHEMA_VERSION,
        protocol_id: R2_MAP_DATASET_PROTOCOL_ID.to_owned(),
        dataset_blake3: manifest.dataset_blake3.clone(),
        config_blake3,
        frames,
        draft_frames,
        imitation_draft_frames,
        selected_only_draft_frames,
        draft_candidates,
        market_frames,
        market_policy_target_frames,
        games: selected_games_len(&records, config),
        stream_blake3: stream_hasher.finalize().to_hex().to_string(),
    })
}

fn write_dataset_frame<W: Write>(
    writer: &mut W,
    stream_hasher: &mut Hasher,
    payload: &[u8],
) -> Result<()> {
    let payload_hash = blake3::hash(payload);
    let length = u32::try_from(payload.len())
        .map_err(|_| R2Error::DatasetContract("R2-MAP frame exceeds u32".to_owned()))?;
    writer.write_all(&length.to_le_bytes())?;
    writer.write_all(payload_hash.as_bytes())?;
    writer.write_all(payload)?;
    stream_hasher.update(&length.to_le_bytes());
    stream_hasher.update(payload_hash.as_bytes());
    Ok(())
}

fn select_records<'a>(
    records: &'a [R2MapGameRecord],
    config: &R2MapDatasetStreamConfig,
) -> Vec<&'a R2MapGameRecord> {
    let mut selected = records
        .iter()
        .filter(|record| {
            (config.game_indices.is_empty()
                || config
                    .game_indices
                    .contains(&record.identity.global_game_index))
                && match config.mode {
                    R2MapDatasetMode::Train => !game_is_validation(record.identity.game_id),
                    R2MapDatasetMode::Validation | R2MapDatasetMode::FixedPanel => {
                        game_is_validation(record.identity.game_id)
                    }
                }
        })
        .collect::<Vec<_>>();
    if config.mode == R2MapDatasetMode::FixedPanel {
        selected.truncate(config.fixed_panel_games);
    }
    selected
}

fn selected_games_len(records: &[R2MapGameRecord], config: &R2MapDatasetStreamConfig) -> usize {
    select_records(records, config).len()
}

pub fn game_is_validation(game_id: [u8; 32]) -> bool {
    let hash = hash_parts(R2_MAP_DATASET_SPLIT_DOMAIN.as_bytes(), &[&game_id]);
    hash[0] % R2_MAP_DATASET_VALIDATION_BUCKETS == R2_MAP_DATASET_VALIDATION_BUCKET
}

fn transform_id(
    config: &R2MapDatasetStreamConfig,
    game_id: [u8; 32],
    draft_decision_id: [u8; 32],
) -> u8 {
    if config.mode != R2MapDatasetMode::Train {
        return 0;
    }
    let hash = hash_parts(
        R2_MAP_DATASET_D6_DOMAIN.as_bytes(),
        &[
            &game_id,
            &draft_decision_id,
            &config.sampler_seed.to_le_bytes(),
        ],
    );
    (hash[0] % 12 + (config.epoch % 12) as u8) % 12
}

pub fn draft_is_imitation_subset(
    collection_kind: R2MapCollectionKind,
    draft_decision_id: [u8; 32],
) -> bool {
    if collection_kind != R2MapCollectionKind::Bootstrap {
        return false;
    }
    let hash = hash_parts(
        R2_MAP_DRAFT_IMITATION_SUBSET_ID.as_bytes(),
        &[&draft_decision_id],
    );
    let draw = u64::from_le_bytes(hash[..8].try_into().expect("eight-byte digest prefix"));
    u128::from(draw) * 1_000_000
        < u128::from(u64::MAX) * u128::from(R2_MAP_DRAFT_IMITATION_SUBSET_PARTS_PER_MILLION)
}

fn write_stream_header<W: Write>(
    writer: &mut W,
    dataset_blake3: &str,
    config_blake3: &str,
    stream_config: &R2MapDatasetStreamConfig,
    frames: usize,
    games: usize,
) -> Result<()> {
    let dataset = decode_hash(dataset_blake3)?;
    let config = decode_hash(config_blake3)?;
    writer.write_all(R2_MAP_DATASET_MAGIC)?;
    writer.write_all(&R2_MAP_DATASET_SCHEMA_VERSION.to_le_bytes())?;
    writer.write_all(&R2_MAP_DATASET_HEADER_SIZE.to_le_bytes())?;
    writer.write_all(&dataset)?;
    writer.write_all(&config)?;
    writer.write_all(
        &u64::try_from(frames)
            .map_err(|_| R2Error::DatasetContract("frame count exceeds u64".to_owned()))?
            .to_le_bytes(),
    )?;
    writer.write_all(
        &u64::try_from(games)
            .map_err(|_| R2Error::DatasetContract("game count exceeds u64".to_owned()))?
            .to_le_bytes(),
    )?;
    writer.write_all(&[stream_config.mode.code(), 0, 0, 0])?;
    writer.write_all(&stream_config.epoch.to_le_bytes())?;
    writer.write_all(&stream_config.sampler_seed.to_le_bytes())?;
    writer.write_all(
        &u64::try_from(stream_config.fixed_panel_games)
            .map_err(|_| R2Error::DatasetContract("fixed panel count exceeds u64".to_owned()))?
            .to_le_bytes(),
    )?;
    debug_assert_eq!(
        8 + 2 + 2 + 32 + 32 + 8 + 8 + 4 + 8 + 8 + 8,
        usize::from(R2_MAP_DATASET_HEADER_SIZE)
    );
    Ok(())
}

fn game_at_example(record: &R2MapGameRecord, example: &R2MapPrimaryExample) -> Result<GameState> {
    let turn = usize::from(example.turn_index);
    let mut game = GameState::new(record.config, record.seed)?;
    for action in &record.replay.turns[..turn] {
        game = game.transition(action)?;
    }
    if game.current_player() != usize::from(example.seat)
        || record.replay.turns.get(turn) != Some(&example.action)
    {
        return invalid("R2-MAP replay turn differs from primary example");
    }
    Ok(game)
}

fn stage_market_session(
    record: &R2MapGameRecord,
    example: &R2MapPrimaryExample,
    game: &GameState,
) -> Result<MarketDecisionSession> {
    let decision = record
        .decisions
        .get(usize::from(example.turn_index))
        .ok_or_else(|| R2Error::DatasetContract("R2-MAP turn record is absent".to_owned()))?;
    let mut session = MarketDecisionSession::begin(game)?;
    for expected in &decision.market_decisions {
        if session.stage() != expected.stage
            || *session.public_state().canonical_hash().as_bytes() != expected.parent_public_hash
        {
            return invalid("R2-MAP market frame parent or stage differs from replay");
        }
        let legal = session.legal_decisions();
        let action_ids = legal
            .iter()
            .map(|choice| {
                let bytes = choice.public_wire_bytes(expected.stage)?;
                Ok(public_market_action_identity(expected.decision_id, bytes))
            })
            .collect::<std::result::Result<Vec<_>, cascadia_game::RuleError>>()?;
        if expected.legal_action_count as usize != legal.len()
            || usize::from(expected.selected_index) >= legal.len()
            || legal[usize::from(expected.selected_index)] != expected.selected
            || action_ids[usize::from(expected.selected_index)] != expected.selected_action_id
            || r2_map_ordered_action_ids_blake3(&action_ids)?
                != expected.ordered_legal_action_ids_blake3
        {
            return invalid("R2-MAP market frame legal screen differs from replay evidence");
        }
        session.commit(&expected.selected)?;
        if *session.public_state().canonical_hash().as_bytes() != expected.resulting_public_hash {
            return invalid("R2-MAP market frame resulting public hash differs");
        }
    }
    if session.stage() != MarketDecisionStage::Draft {
        return invalid("R2-MAP market replay omitted explicit Stop");
    }
    Ok(session)
}

fn encode_market_examples(
    record: &R2MapGameRecord,
    example: &R2MapPrimaryExample,
    transform_id: u8,
) -> Result<Vec<Vec<u8>>> {
    let game = game_at_example(record, example)?;
    let turn_record = &record.decisions[usize::from(example.turn_index)];
    let transform = D6Transform::from_id(transform_id)
        .ok_or_else(|| R2Error::DatasetContract("D6 transform id exceeds 11".to_owned()))?;
    let mut session = MarketDecisionSession::begin(&game)?;
    let mut frames = Vec::with_capacity(turn_record.market_decisions.len());
    for expected in &turn_record.market_decisions {
        let stage = session.stage();
        let parent = session.public_state();
        if stage != expected.stage
            || *parent.canonical_hash().as_bytes() != expected.parent_public_hash
        {
            return invalid("R2-MAP market frame parent or stage differs from replay");
        }
        let legal = session.legal_decisions();
        let rows = legal
            .iter()
            .map(|choice| {
                let bytes = choice.public_wire_bytes(stage)?;
                let action_id = public_market_action_identity(expected.decision_id, bytes);
                Ok((action_id, bytes))
            })
            .collect::<std::result::Result<Vec<_>, cascadia_game::RuleError>>()?;
        let action_ids = rows
            .iter()
            .map(|(action_id, _)| *action_id)
            .collect::<Vec<_>>();
        let selected_index = usize::from(expected.selected_index);
        if expected.legal_action_count as usize != rows.len()
            || rows.get(selected_index).map(|row| row.0) != Some(expected.selected_action_id)
            || legal.get(selected_index) != Some(&expected.selected)
            || r2_map_ordered_action_ids_blake3(&action_ids)?
                != expected.ordered_legal_action_ids_blake3
        {
            return invalid("R2-MAP market frame legal rows differ from replay evidence");
        }
        let perspective = usize::from(example.seat);
        let exact_current = score_board(
            &session.staged_game().boards()[perspective],
            record.config.scoring_cards,
        )
        .base_total;
        let final_score = record.scores[perspective].base_total;
        let score_to_go = i16::try_from(i32::from(final_score) - i32::from(exact_current))
            .map_err(|_| {
                R2Error::DatasetContract("R2-MAP market score-to-go exceeds i16".to_owned())
            })?;
        let public_nature_tokens = session.staged_game().boards()[perspective].nature_tokens();
        let public_wildlife_bag_counts = session.staged_game().public_supply().wildlife_bag;
        let public_wildlife_bag_total = public_wildlife_bag_counts
            .into_iter()
            .try_fold(0u8, u8::checked_add)
            .ok_or_else(|| {
                R2Error::DatasetContract("public wildlife bag count exceeds u8".to_owned())
            })?;
        let public_market_wildlife = session
            .staged_game()
            .market()
            .wildlife
            .map(|wildlife| wildlife.expect("active market is complete") as u8);

        let mut payload = Vec::new();
        payload.extend_from_slice(&[
            R2_MAP_DATASET_MARKET_FRAME_KIND,
            expected.ordinal,
            stage as u8,
            R2_MAP_DATASET_FRAME_VERSION,
        ]);
        payload.extend_from_slice(&example.game_id);
        payload.extend_from_slice(&example.position_id);
        payload.extend_from_slice(&expected.decision_id);
        payload.extend_from_slice(&expected.selected_action_id);
        payload.extend_from_slice(&expected.parent_public_hash);
        payload.extend_from_slice(&expected.resulting_public_hash);
        payload.extend_from_slice(&expected.ordered_legal_action_ids_blake3);
        payload.extend_from_slice(&record.identity.global_game_index.to_le_bytes());
        payload.extend_from_slice(&example.turn_index.to_le_bytes());
        payload.extend_from_slice(&[
            example.seat,
            u8::from(game_is_validation(example.game_id)),
            transform_id,
            public_nature_tokens,
            public_wildlife_bag_total,
            u8::from(record.collection_kind == R2MapCollectionKind::Bootstrap),
        ]);
        payload.extend_from_slice(&public_wildlife_bag_counts);
        payload.extend_from_slice(&public_market_wildlife);
        payload.extend_from_slice(&[0; 3]);
        payload.extend_from_slice(
            &u32::try_from(rows.len())
                .map_err(|_| {
                    R2Error::DatasetContract("market legal screen exceeds u32".to_owned())
                })?
                .to_le_bytes(),
        );
        payload.extend_from_slice(
            &u32::try_from(selected_index)
                .map_err(|_| {
                    R2Error::DatasetContract("market selected index exceeds u32".to_owned())
                })?
                .to_le_bytes(),
        );
        payload.extend_from_slice(&exact_current.to_le_bytes());
        payload.extend_from_slice(&final_score.to_le_bytes());
        payload.extend_from_slice(&score_to_go.to_le_bytes());
        payload.extend_from_slice(&0u16.to_le_bytes());
        debug_assert_eq!(payload.len(), R2_MAP_DATASET_MARKET_FIXED_SIZE);
        encode_public_state(
            &mut payload,
            &parent,
            record.identity.global_game_index,
            perspective,
            transform,
            false,
        )?;
        for (action_id, bytes) in rows {
            payload.extend_from_slice(&action_id);
            payload.extend_from_slice(&bytes);
        }
        session.commit(&expected.selected)?;
        if *session.public_state().canonical_hash().as_bytes() != expected.resulting_public_hash {
            return invalid("R2-MAP market frame resulting public hash differs");
        }
        frames.push(payload);
    }
    if session.stage() != MarketDecisionStage::Draft {
        return invalid("R2-MAP market frames omitted explicit Stop");
    }
    Ok(frames)
}

struct EncodedDraftFrame {
    payload: Vec<u8>,
    candidate_count: usize,
    imitation: bool,
}

fn encode_example(
    record: &R2MapGameRecord,
    example: &R2MapPrimaryExample,
    transform_id: u8,
    retain_imitation: bool,
) -> Result<EncodedDraftFrame> {
    example.validate()?;
    let turn = usize::from(example.turn_index);
    let game = game_at_example(record, example)?;
    let session = stage_market_session(record, example, &game)?;
    let turn_record = &record.decisions[turn];
    let mut action = example.action.clone();
    action.replace_three_of_a_kind = false;
    action.wildlife_wipes.clear();
    if session.bundle_action(&action)? != example.action
        || turn_record.draft_action_id != cascadia_data::r2_map_draft_action_id(&action)?
    {
        return invalid("R2-MAP staged draft does not reconstruct its bundled replay action");
    }
    let draft_parent = session.public_state();
    if *draft_parent.canonical_hash().as_bytes() != turn_record.draft_parent_public_hash {
        return invalid("R2-MAP staged draft parent hash differs from replay evidence");
    }
    let draft_afterstate = session.staged_game().preview_public_afterstate(&action)?;
    if draft_afterstate != example.afterstate {
        return invalid("R2-MAP staged draft afterstate differs from bundled replay afterstate");
    }
    let auxiliaries = derive_auxiliary_targets(record, turn, &game)?;
    let transform = D6Transform::from_id(transform_id)
        .ok_or_else(|| R2Error::DatasetContract("D6 transform id exceeds 11".to_owned()))?;
    let transformed_game = session.staged_game().transformed(transform)?;
    let imitation = retain_imitation
        && draft_is_imitation_subset(record.collection_kind, turn_record.draft_decision_id);
    let (candidates, selected_index) = if imitation {
        let candidates = session.legal_draft_actions()?;
        let action_ids = candidates
            .iter()
            .map(cascadia_data::r2_map_draft_action_id)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if u32::try_from(candidates.len()).ok() != Some(turn_record.draft_legal_action_count)
            || r2_map_ordered_action_ids_blake3(&action_ids)?
                != turn_record.draft_ordered_action_ids_blake3
        {
            return invalid("R2-MAP imitation draft legal screen differs from replay evidence");
        }
        let selected_index = candidates
            .iter()
            .position(|candidate| candidate == &action)
            .ok_or_else(|| {
                R2Error::DatasetContract(
                    "R2-MAP imitation draft omitted the selected action".to_owned(),
                )
            })?;
        if action_ids[selected_index] != turn_record.draft_action_id {
            return invalid("R2-MAP imitation selected action identity differs");
        }
        (candidates, selected_index)
    } else {
        (vec![action.clone()], 0)
    };

    let mut payload = Vec::new();
    payload.extend_from_slice(&[
        R2_MAP_DATASET_DRAFT_FRAME_KIND,
        u8::try_from(turn_record.market_decisions.len())
            .map_err(|_| R2Error::DatasetContract("draft ordinal exceeds u8".to_owned()))?,
        MarketDecisionStage::Draft as u8,
        R2_MAP_DATASET_FRAME_VERSION,
    ]);
    payload.extend_from_slice(&example.game_id);
    payload.extend_from_slice(&turn_record.draft_decision_id);
    payload.extend_from_slice(&turn_record.draft_action_id);
    payload.extend_from_slice(&record.identity.global_game_index.to_le_bytes());
    payload.extend_from_slice(&example.turn_index.to_le_bytes());
    payload.push(example.seat);
    payload.push(u8::from(game_is_validation(example.game_id)));
    payload.push(transform_id);
    payload.push(auxiliaries.opponent_valid_mask);
    payload.push(u8::from(auxiliaries.market_valid));
    payload.push(u8::from(imitation));
    payload.extend_from_slice(
        &u32::try_from(candidates.len())
            .map_err(|_| R2Error::DatasetContract("draft legal screen exceeds u32".to_owned()))?
            .to_le_bytes(),
    );
    payload.extend_from_slice(
        &u32::try_from(selected_index)
            .map_err(|_| R2Error::DatasetContract("draft selected index exceeds u32".to_owned()))?
            .to_le_bytes(),
    );
    for value in example.current {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    for value in example.residual {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    for value in example.terminal {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    for target in auxiliaries.opponents {
        payload.extend_from_slice(&[
            target.tile_slot,
            target.wildlife_slot,
            target.draft_kind,
            target.drafted_wildlife,
            target.replace_three_of_a_kind,
            target.paid_wipe_count,
        ]);
        payload.extend_from_slice(&target.paid_wipe_masks);
    }
    payload.extend_from_slice(&auxiliaries.disposition);
    payload.extend_from_slice(&auxiliaries.pair_survives);
    payload.extend_from_slice(&auxiliaries.final_slot);
    encode_public_state(
        &mut payload,
        &draft_parent,
        record.identity.global_game_index,
        usize::from(example.seat),
        transform,
        false,
    )?;
    for candidate in &candidates {
        let action_id = cascadia_data::r2_map_draft_action_id(candidate)?;
        let afterstate = session.staged_game().preview_public_afterstate(candidate)?;
        let transformed_action = candidate.transformed(session.staged_game(), transform)?;
        payload.extend_from_slice(&action_id);
        payload.extend_from_slice(
            &GradedOracleActionFeatures::observe(&transformed_game, &transformed_action)?
                .to_bytes(),
        );
        let exact_afterstate_score = score_board(
            &afterstate.boards()[usize::from(example.seat)],
            record.config.scoring_cards,
        )
        .base_total;
        payload.extend_from_slice(&exact_afterstate_score.to_le_bytes());
        encode_public_state(
            &mut payload,
            &afterstate,
            record.identity.global_game_index,
            usize::from(example.seat),
            transform,
            true,
        )?;
    }
    if candidates[selected_index] != action || draft_afterstate != example.afterstate {
        return invalid("R2-MAP encoded selected draft differs from replay afterstate");
    }
    Ok(EncodedDraftFrame {
        payload,
        candidate_count: candidates.len(),
        imitation,
    })
}

#[derive(Debug, Clone, Copy, Default)]
struct OpponentTarget {
    tile_slot: u8,
    wildlife_slot: u8,
    draft_kind: u8,
    drafted_wildlife: u8,
    replace_three_of_a_kind: u8,
    paid_wipe_count: u8,
    paid_wipe_masks: [u8; R2_MAP_DATASET_OPPONENT_WIPE_MAX],
}

#[derive(Debug, Clone, Copy)]
struct AuxiliaryTargets {
    opponent_valid_mask: u8,
    market_valid: bool,
    opponents: [OpponentTarget; 3],
    disposition: [u8; 4],
    pair_survives: [u8; 4],
    final_slot: [u8; 4],
}

fn encode_paid_wipe_sequence(
    wipes: &[WildlifeWipe],
) -> Result<(u8, [u8; R2_MAP_DATASET_OPPONENT_WIPE_MAX])> {
    if wipes.len() > R2_MAP_DATASET_OPPONENT_WIPE_MAX {
        return invalid("opponent paid-wipe sequence exceeds the frozen 20-wipe maximum");
    }
    let mut masks = [0u8; R2_MAP_DATASET_OPPONENT_WIPE_MAX];
    for (ordinal, wipe) in wipes.iter().enumerate() {
        masks[ordinal] = wipe
            .slots
            .iter()
            .fold(0u8, |mask, slot| mask | (1 << slot.index()));
        if masks[ordinal] == 0 {
            return invalid("opponent paid-wipe sequence contains an empty mask");
        }
    }
    Ok((u8::try_from(wipes.len()).expect("maximum is 20"), masks))
}

fn derive_auxiliary_targets(
    record: &R2MapGameRecord,
    turn: usize,
    parent: &GameState,
) -> Result<AuxiliaryTargets> {
    let mut game = parent.transition(&record.replay.turns[turn])?;
    let initial_tiles = game.market().tiles.map(|tile| tile.map(|value| value.id.0));
    let initial_wildlife = game
        .market()
        .wildlife
        .map(|value| value.map(|wildlife| wildlife as u8));
    let mut opponents = [OpponentTarget::default(); 3];
    let mut selected_tile_ids = [u8::MAX; 3];
    let mut opponent_valid_mask = 0u8;
    for offset in 0..3 {
        let action_index = turn + 1 + offset;
        let Some(action) = record.replay.turns.get(action_index) else {
            break;
        };
        let observed = PublicActionRecord::observe(&game, action)?;
        let tile_slot = usize::from(observed.tile_slot);
        selected_tile_ids[offset] = game.market().tiles[tile_slot]
            .ok_or_else(|| R2Error::DatasetContract("opponent selected an absent tile".to_owned()))?
            .id
            .0;
        let (paid_wipe_count, paid_wipe_masks) = encode_paid_wipe_sequence(&action.wildlife_wipes)?;
        opponents[offset] = OpponentTarget {
            tile_slot: observed.tile_slot,
            wildlife_slot: observed.wildlife_slot,
            draft_kind: observed.draft_kind,
            drafted_wildlife: observed.drafted_wildlife,
            replace_three_of_a_kind: observed.replace_three_of_a_kind,
            paid_wipe_count,
            paid_wipe_masks,
        };
        opponent_valid_mask |= 1 << offset;
        game = game.transition(action)?;
    }
    let market_valid = opponent_valid_mask == 0b111;
    let mut disposition = [0u8; 4];
    let mut pair_survives = [0u8; 4];
    let mut final_slot = [0u8; 4];
    if market_valid {
        if initial_tiles.iter().any(Option::is_none) || initial_wildlife.iter().any(Option::is_none)
        {
            return invalid(
                "post-action market is not fully refilled for complete auxiliary horizon",
            );
        }
        for slot in MarketSlot::ALL {
            let initial_tile = initial_tiles[slot.index()].expect("checked full market");
            if let Some(opponent) = selected_tile_ids
                .iter()
                .position(|value| *value == initial_tile)
            {
                disposition[slot.index()] = opponent as u8;
            } else {
                disposition[slot.index()] = 3;
                let surviving_slot = game
                    .market()
                    .tiles
                    .iter()
                    .position(|tile| tile.is_some_and(|value| value.id.0 == initial_tile))
                    .ok_or_else(|| {
                        R2Error::DatasetContract("tile neither consumed nor surviving".to_owned())
                    })?;
                final_slot[slot.index()] = surviving_slot as u8;
                pair_survives[slot.index()] = u8::from(
                    game.market().wildlife[surviving_slot].map(|value| value as u8)
                        == initial_wildlife[slot.index()],
                );
            }
        }
    }
    Ok(AuxiliaryTargets {
        opponent_valid_mask,
        market_valid,
        opponents,
        disposition,
        pair_survives,
        final_slot,
    })
}

fn encode_public_state(
    output: &mut Vec<u8>,
    state: &cascadia_game::PublicGameState,
    game_index: u64,
    perspective_seat: usize,
    transform: D6Transform,
    selected_afterstate: bool,
) -> Result<()> {
    let record = PositionRecord::observe_public_for_seat(state, game_index, perspective_seat);
    let sparse = if selected_afterstate {
        SparsePublicState::from_selected_afterstate_record(&record, None)?
    } else {
        SparsePublicState::from_position_record(&record, None)?
    };
    let compact = encode_compact_r2_map_tokens(&sparse, transform)?;
    write_compact_tokens(output, &compact)?;
    let (market, market_mask) = encode_market_features(&record)?;
    let (players, player_mask) = encode_player_features(&record)?;
    let global = encode_global_features(&record)?;
    write_f32s(output, &market);
    output.extend_from_slice(&market_mask);
    write_f32s(output, &players);
    output.extend_from_slice(&player_mask);
    write_f32s(output, &global);
    Ok(())
}

fn write_compact_tokens(output: &mut Vec<u8>, state: &EncodedR2MapCompactTokens) -> Result<()> {
    let token_count = state.token_types.len();
    if state.token_seats.len() != token_count
        || state.token_payload.len() != token_count * TOKEN_PAYLOAD_WIDTH
        || token_count
            != state
                .board_type_counts
                .iter()
                .flatten()
                .map(|value| usize::from(*value))
                .sum::<usize>()
    {
        return invalid("compact R2 token vectors are inconsistent");
    }
    output.extend_from_slice(
        &u16::try_from(token_count)
            .map_err(|_| R2Error::DatasetContract("compact token count exceeds u16".to_owned()))?
            .to_le_bytes(),
    );
    for count in state.board_type_counts.iter().flatten() {
        output.extend_from_slice(&count.to_le_bytes());
    }
    output.extend_from_slice(&state.token_types);
    output.extend_from_slice(&state.token_seats);
    output.extend(state.token_payload.iter().map(|value| *value as u8));
    Ok(())
}

fn write_f32s<const N: usize>(output: &mut Vec<u8>, values: &[f32; N]) {
    for value in values {
        output.extend_from_slice(&value.to_le_bytes());
    }
}

fn canonical_hash<T: Serialize>(value: &T) -> Result<String> {
    let bytes = serde_json::to_vec(value)?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn hash_parts(domain: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(&(domain.len() as u64).to_le_bytes());
    hasher.update(domain);
    for part in parts {
        hasher.update(&(part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    *hasher.finalize().as_bytes()
}

fn decode_hash(value: &str) -> Result<[u8; 32]> {
    if value.len() != 64 {
        return invalid("R2-MAP hash must contain 64 lowercase hex characters");
    }
    let mut output = [0u8; 32];
    for (index, slot) in output.iter_mut().enumerate() {
        let byte = &value[index * 2..index * 2 + 2];
        *slot = u8::from_str_radix(byte, 16)
            .map_err(|_| R2Error::DatasetContract("R2-MAP hash is not hexadecimal".to_owned()))?;
    }
    Ok(output)
}

fn invalid<T>(message: &str) -> Result<T> {
    Err(R2Error::DatasetContract(message.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_cost_packing_isolates_an_indivisible_oversize_screen() {
        assert!(packed_batch_can_admit(0, 0, 12_972, 4_096));
        assert!(!packed_batch_can_admit(12_972, 1, 1, 4_096));
        assert!(packed_batch_can_admit(441, 9, 1, 4_500));
        assert!(!packed_batch_can_admit(441, 9, 1, 4_096));
    }
    use cascadia_data::{
        R2MapCollectionKind, R2MapExplorationIdentity, R2MapGameIdentity, R2MapPolicyIdentity,
        R2MapProtocolIdentity, R2MapRecordContext, R2MapRngIdentity, R2MapSeedPurpose,
        focal_seat_for_game, r2_map_game_seed, write_r2_map_shard,
    };
    use cascadia_game::GameConfig;
    use cascadia_sim::{MatchConfig, StrategyKind, play_match};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn split_and_transform_are_identity_driven() {
        assert_eq!(game_is_validation([9; 32]), game_is_validation([9; 32]));
        assert_ne!(R2_MAP_DATASET_SPLIT_DOMAIN, R2_MAP_DATASET_D6_DOMAIN);
    }

    #[test]
    fn cyclic_d6_schedule_covers_each_transform_once_and_resumes_exactly() {
        let game_id = [0x31; 32];
        let draft_decision_id = [0x72; 32];
        let transforms = (0..12)
            .map(|epoch| {
                transform_id(
                    &R2MapDatasetStreamConfig::train(epoch, 99),
                    game_id,
                    draft_decision_id,
                )
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(transforms, (0..12).collect());
        assert_eq!(
            transform_id(
                &R2MapDatasetStreamConfig::train(0, 99),
                game_id,
                draft_decision_id,
            ),
            transform_id(
                &R2MapDatasetStreamConfig::train(12, 99),
                game_id,
                draft_decision_id,
            )
        );
        assert_eq!(
            transform_id(
                &R2MapDatasetStreamConfig::validation(),
                game_id,
                draft_decision_id,
            ),
            0
        );
    }

    #[test]
    fn opponent_paid_wipe_wire_preserves_exact_order_and_fixed_padding() {
        let wipes = vec![
            WildlifeWipe {
                slots: vec![MarketSlot::ZERO, MarketSlot::TWO],
            },
            WildlifeWipe {
                slots: vec![MarketSlot::THREE],
            },
        ];
        let (count, masks) = encode_paid_wipe_sequence(&wipes).unwrap();
        assert_eq!(count, 2);
        assert_eq!(&masks[..2], &[0b0101, 0b1000]);
        assert!(masks[2..].iter().all(|mask| *mask == 0));

        let too_many = vec![wipes[0].clone(); R2_MAP_DATASET_OPPONENT_WIPE_MAX + 1];
        assert!(encode_paid_wipe_sequence(&too_many).is_err());
    }

    #[test]
    fn imitation_subset_never_labels_iterative_or_benchmark_actions() {
        let identity = (0u32..100_000)
            .map(|value| hash_parts(b"test-draft-identity", &[&value.to_le_bytes()]))
            .find(|identity| draft_is_imitation_subset(R2MapCollectionKind::Bootstrap, *identity))
            .expect("the deterministic search finds a 1% bootstrap member");
        assert!(!draft_is_imitation_subset(
            R2MapCollectionKind::IterativeTraining,
            identity
        ));
        assert!(!draft_is_imitation_subset(
            R2MapCollectionKind::Benchmark,
            identity
        ));
    }

    #[test]
    fn replay_stream_is_deterministic_and_masks_terminal_horizons() {
        let campaign = "r2-map-dataset-test";
        let config = GameConfig::research_aaaaa(4).unwrap();
        let (_game_index, record) = (0..100)
            .find_map(|game_index| {
                let seed = r2_map_game_seed(campaign, R2MapSeedPurpose::Bootstrap, 0, game_index);
                let identity = R2MapGameIdentity::new(campaign, 0, "john2", game_index, seed);
                if game_is_validation(identity.game_id) {
                    return None;
                }
                let result =
                    play_match(&MatchConfig::symmetric(config, seed, StrategyKind::Greedy))
                        .unwrap();
                let record = R2MapGameRecord::from_match(
                    R2MapRecordContext {
                        collection_kind: R2MapCollectionKind::Bootstrap,
                        identity,
                        seed_purpose: R2MapSeedPurpose::Bootstrap,
                        focal_seat: focal_seat_for_game(game_index),
                        seats: vec![R2MapPolicyIdentity::greedy(); 4],
                        rng: R2MapRngIdentity::default(),
                        exploration: R2MapExplorationIdentity::disabled(),
                        protocols: R2MapProtocolIdentity {
                            collector_hash: [1; 32],
                            source_hash: [2; 32],
                            serving_protocol_hash: [3; 32],
                        },
                    },
                    &result,
                    &[],
                )
                .unwrap();
                record
                    .decisions
                    .iter()
                    .any(|decision| {
                        draft_is_imitation_subset(
                            R2MapCollectionKind::Bootstrap,
                            decision.draft_decision_id,
                        )
                    })
                    .then_some((game_index, record))
            })
            .expect("one of the first 100 deterministic games enters the 1% imitation subset");
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("cascadia-r2-map-{unique}"));
        fs::create_dir_all(&root).unwrap();
        let shard = root.join("one.r2sh");
        write_r2_map_shard(&shard, &[record]).unwrap();
        let paths = vec![shard];
        let manifest = build_r2_map_dataset_manifest(&paths).unwrap();
        let compact_metadata = build_r2_map_compact_index_metadata(&paths).unwrap();
        assert_eq!(compact_metadata.dataset_manifest, manifest);
        assert_eq!(compact_metadata.games.len(), 1);
        assert_eq!(compact_metadata.games[0].candidate_widths.len(), 80);
        assert_eq!(
            compact_metadata.games[0]
                .candidate_widths
                .iter()
                .filter(|width| **width > 1)
                .count(),
            manifest.imitation_example_count
        );
        assert!(manifest.imitation_example_count > 0);
        assert_eq!(
            manifest.market_policy_target_count,
            manifest.market_decision_count
        );
        let config = R2MapDatasetStreamConfig::train(7, 11);
        let mut first = Vec::new();
        let receipt = stream_r2_map_dataset(&paths, &manifest, &config, &mut first).unwrap();
        let mut second = Vec::new();
        let repeated = stream_r2_map_dataset(&paths, &manifest, &config, &mut second).unwrap();
        assert_eq!(first, second);
        assert_eq!(receipt, repeated);
        let mut receipt_bound = Vec::new();
        let receipt_bound_result = stream_r2_map_dataset_after_semantic_validation(
            &paths,
            &manifest,
            &config,
            &mut receipt_bound,
        )
        .unwrap();
        assert_eq!(first, receipt_bound);
        assert_eq!(receipt, receipt_bound_result);
        let mut chunk_config = config.clone();
        chunk_config.game_indices = vec![compact_metadata.games[0].global_game_index];
        let mut chunk = Vec::new();
        let chunk_receipt = stream_r2_map_dataset_after_semantic_validation(
            &paths,
            &manifest,
            &chunk_config,
            &mut chunk,
        )
        .unwrap();
        assert_eq!(chunk_receipt.frames, receipt.frames);
        assert_eq!(
            &chunk[usize::from(R2_MAP_DATASET_HEADER_SIZE)..],
            &first[usize::from(R2_MAP_DATASET_HEADER_SIZE)..]
        );
        chunk_config.game_indices = vec![u64::MAX];
        assert!(
            stream_r2_map_dataset_after_semantic_validation(
                &paths,
                &manifest,
                &chunk_config,
                Vec::new(),
            )
            .is_err()
        );
        assert_eq!(receipt.draft_frames, 80);
        assert_eq!(
            receipt.imitation_draft_frames + receipt.selected_only_draft_frames,
            receipt.draft_frames
        );
        assert!(receipt.draft_candidates >= receipt.draft_frames);
        assert_eq!(
            receipt.imitation_draft_frames,
            manifest.imitation_example_count
        );
        assert_eq!(receipt.market_policy_target_frames, receipt.market_frames);
        assert_eq!(receipt.market_frames, manifest.market_decision_count);
        assert_eq!(receipt.frames, receipt.draft_frames + receipt.market_frames);

        let mut cursor = usize::from(R2_MAP_DATASET_HEADER_SIZE);
        let mut opponent_masks = Vec::new();
        let mut market_masks = Vec::new();
        let mut market_stages = Vec::new();
        let mut imitation_flags = Vec::new();
        for _ in 0..receipt.frames {
            let length = u32::from_le_bytes(first[cursor..cursor + 4].try_into().unwrap()) as usize;
            let payload = cursor + R2_MAP_DATASET_FRAME_HEADER_SIZE;
            match first[payload] {
                R2_MAP_DATASET_DRAFT_FRAME_KIND => {
                    opponent_masks.push(first[payload + 4 + 109]);
                    market_masks.push(first[payload + 4 + 110]);
                    let imitation = first[payload + 115];
                    let candidate_count =
                        u32::from_le_bytes(first[payload + 116..payload + 120].try_into().unwrap());
                    assert!(imitation <= 1);
                    if imitation == 0 {
                        assert_eq!(candidate_count, 1);
                    } else {
                        assert!(candidate_count > 1);
                    }
                    imitation_flags.push(imitation);
                    assert_eq!(first[payload + 2], MarketDecisionStage::Draft as u8);
                }
                R2_MAP_DATASET_MARKET_FRAME_KIND => {
                    assert!(length >= R2_MAP_DATASET_MARKET_FIXED_SIZE);
                    market_stages.push(first[payload + 2]);
                    assert_eq!(first[payload + 243], 1);
                    let legal_count =
                        u32::from_le_bytes(first[payload + 256..payload + 260].try_into().unwrap());
                    let selected_index =
                        u32::from_le_bytes(first[payload + 260..payload + 264].try_into().unwrap());
                    assert!(legal_count >= 1);
                    assert!(selected_index < legal_count);
                }
                other => panic!("unexpected R2-MAP frame kind {other}"),
            }
            cursor = payload + length;
        }
        assert!(market_stages.iter().all(|stage| *stage <= 1));
        assert_eq!(
            imitation_flags.iter().filter(|value| **value == 1).count(),
            receipt.imitation_draft_frames
        );
        assert_eq!(&opponent_masks[76..], &[0b111, 0b011, 0b001, 0]);
        assert_eq!(&market_masks[76..], &[1, 0, 0, 0]);
        assert_eq!(cursor, first.len());

        fn emitted_batches(bytes: &[u8]) -> Vec<(serde_json::Value, Vec<u8>)> {
            let mut cursor = 0usize;
            let mut values = Vec::new();
            while cursor < bytes.len() {
                let newline = bytes[cursor..]
                    .iter()
                    .position(|value| *value == b'\n')
                    .map(|offset| cursor + offset)
                    .expect("control frame is newline terminated");
                let control: serde_json::Value =
                    serde_json::from_slice(&bytes[cursor..newline]).unwrap();
                cursor = newline + 1;
                if control["type"] == "batch" {
                    let length = control["payload_bytes"].as_u64().unwrap() as usize;
                    let payload = bytes[cursor..cursor + length].to_vec();
                    cursor += length;
                    values.push((control, payload));
                }
            }
            values
        }

        let producer = R2MapPackedBatchProducerConfig {
            mode: R2MapDatasetMode::Train,
            epoch: 0,
            sampler_seed: 20260618,
            group_batch_size: 10,
            maximum_candidates_per_batch: 16_384,
            bootstrap_value_only: false,
            ordered_game_indices: vec![compact_metadata.games[0].global_game_index],
            start_game_offset: 0,
            start_turn_offset: 0,
            start_batch_index: 0,
        };
        let mut crashed = Vec::new();
        assert!(
            serve_r2_map_packed_batches(
                &paths[0],
                &producer,
                std::io::Cursor::new(Vec::<u8>::new()),
                &mut crashed,
            )
            .is_err()
        );
        let first_batch = emitted_batches(&crashed).remove(0);
        let resumed = R2MapPackedBatchProducerConfig {
            start_game_offset: first_batch.0["next_game_offset"].as_u64().unwrap() as usize,
            start_turn_offset: first_batch.0["next_turn_offset"].as_u64().unwrap() as usize,
            start_batch_index: first_batch.0["batch_index"].as_u64().unwrap() + 1,
            ..producer.clone()
        };
        let mut resumed_output = Vec::new();
        assert!(
            serve_r2_map_packed_batches(
                &paths[0],
                &resumed,
                std::io::Cursor::new(Vec::<u8>::new()),
                &mut resumed_output,
            )
            .is_err()
        );
        let resumed_batch = emitted_batches(&resumed_output).remove(0);
        let mut repeated_resume = Vec::new();
        assert!(
            serve_r2_map_packed_batches(
                &paths[0],
                &resumed,
                std::io::Cursor::new(Vec::<u8>::new()),
                &mut repeated_resume,
            )
            .is_err()
        );
        assert_eq!(resumed_batch, emitted_batches(&repeated_resume).remove(0));

        let mut tampered = Vec::new();
        let error = serve_r2_map_packed_batches(
            &paths[0],
            &producer,
            std::io::Cursor::new(b"{\"ack\":\"tampered\"}\n".to_vec()),
            &mut tampered,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("acknowledgement identity differs")
        );
        fs::remove_dir_all(root).unwrap();
    }
}
