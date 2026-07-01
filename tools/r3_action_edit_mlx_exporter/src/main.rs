use std::{
    collections::{BTreeMap, HashMap},
    env,
    error::Error,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use blake3::Hasher;
use cascadia_data::{
    DatasetSplit, GradedOracleDatasetManifest, GradedOracleGroup, PositionRecord,
    read_graded_oracle_shard, validate_graded_oracle_dataset,
};
use cascadia_game::{DraftChoice, GameConfig, GameSeed, GameState, TurnAction};
use cascadia_provenance::{checksum_file, source_provenance};
use clap::Parser;
use r2_sparse_entity_census::{
    BOARD_SLOTS, BOARD_TOKEN_CAPACITY, GLOBAL_FEATURES, MARKET_FEATURES, MlxEncodedState,
    PLAYER_FEATURES, SparsePublicState, TOKEN_PAYLOAD_WIDTH, encode_global_features,
    encode_market_features, encode_player_features, encode_sparse_state, transform_encoded_state,
};
use r3_action_edit_census::{
    ActionEdit, MLX_ACTION_TOKEN_PAYLOAD_WIDTH, MlxActionEncoding, PublicStateTrunk,
};
use r3_action_edit_mlx_exporter::{
    ADR_ID, CACHE_SCHEMA, CACHE_SCHEMA_VERSION, ControlDelta, EXPERIMENT_ID, PROTOCOL_ID,
    active_board_tokens_relative, cohort_blake3, control_delta, select_train_candidate_indices,
};
use serde::Serialize;
use serde_json::{Value, json};

const TRAIN_GROUPS: usize = 560;
const VALIDATION_GROUPS: usize = 240;
const VALIDATION_CANDIDATES: usize = 860_203;
const TRAIN_SOURCE_CANDIDATES: usize = 2_135_111;
const CANDIDATE_IDENTITY_DOMAIN: &[u8] = b"r3-mlx-candidate-identity-v1";
const R3_TOKEN_DOMAIN: &[u8] = b"r3-mlx-action-token-stream-v1";

#[derive(Debug, Parser)]
#[command(about = "Export the exact ADR 0150 R3 action-edit MLX cache")]
struct Args {
    #[arg(long)]
    train_dataset: PathBuf,
    #[arg(long)]
    validation_dataset: PathBuf,
    #[arg(long)]
    output_root: PathBuf,
    #[arg(long)]
    receipt: PathBuf,
    #[arg(long)]
    max_groups_per_split: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct FileSpec {
    file: String,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: u64,
    blake3: String,
}

#[derive(Debug, Clone, Default, Serialize)]
struct SplitChecks {
    groups_replayed: usize,
    parent_r2_encodings: usize,
    position_record_checks: usize,
    public_state_hash_checks: usize,
    public_supply_checks: usize,
    graded_action_reconstructions: usize,
    grouped_r3_action_matches: usize,
    r3_apply_checks: usize,
    authoritative_successor_checks: usize,
    canonical_transform_checks: usize,
    r2_afterstate_encodings: usize,
    control_delta_round_trips: usize,
    r3_token_round_trips: usize,
    selected_winner_retained: usize,
    champion_retained: usize,
    source_r600: usize,
    r600_retained: usize,
    source_r1200: usize,
    r1200_retained: usize,
    source_r4800: usize,
    r4800_retained: usize,
    silent_truncations: usize,
    minimum_control_tokens: usize,
    maximum_control_tokens: usize,
    minimum_r3_tokens: usize,
    maximum_r3_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
struct SplitManifest {
    split: &'static str,
    dataset_id: String,
    dataset_manifest_blake3: String,
    groups: usize,
    source_candidates: usize,
    retained_candidates: usize,
    complete_open_split: bool,
    control_removed_tokens: usize,
    control_added_tokens: usize,
    r3_tokens: usize,
    files: BTreeMap<String, FileSpec>,
    checks: SplitChecks,
}

#[derive(Debug, Clone, Serialize)]
struct CacheManifest {
    schema_version: u16,
    cache_schema: &'static str,
    experiment_id: &'static str,
    protocol_id: &'static str,
    adr: &'static str,
    cache_id: String,
    complete_open_corpus: bool,
    exporter: ExporterIdentity,
    tensor_contract: Value,
    hidden_information: HiddenInformationBoundary,
    splits: BTreeMap<String, SplitManifest>,
    scientific_identity: Value,
}

#[derive(Debug, Clone, Serialize)]
struct ExporterIdentity {
    executable_blake3: String,
    source: cascadia_provenance::SourceProvenance,
}

#[derive(Debug, Clone, Serialize)]
struct HiddenInformationBoundary {
    open_train_and_validation_only: bool,
    source_seed_used_for_authoritative_replay: bool,
    hidden_order_exported: bool,
    excluded_tile_identity_exported: bool,
    future_refill_exported: bool,
    sealed_test_opened: bool,
    gameplay_opened: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct DraftBatchKey {
    replace_three_of_a_kind: bool,
    wipe_masks: Vec<u8>,
    draft_kind: u8,
    tile_slot: u8,
    wildlife_slot: u8,
}

struct HashedWriter {
    file_name: String,
    writer: BufWriter<File>,
    hasher: Hasher,
    bytes: u64,
}

impl HashedWriter {
    fn create(root: &Path, file_name: impl Into<String>) -> Result<Self, Box<dyn Error>> {
        let file_name = file_name.into();
        Ok(Self {
            writer: BufWriter::new(File::create(root.join(&file_name))?),
            file_name,
            hasher: Hasher::new(),
            bytes: 0,
        })
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
        self.writer.write_all(bytes)?;
        self.hasher.update(bytes);
        self.bytes += bytes.len() as u64;
        Ok(())
    }

    fn write_u8(&mut self, value: u8) -> Result<(), Box<dyn Error>> {
        self.write_bytes(&[value])
    }

    fn write_i8(&mut self, value: i8) -> Result<(), Box<dyn Error>> {
        self.write_u8(value as u8)
    }

    fn write_u16(&mut self, value: u16) -> Result<(), Box<dyn Error>> {
        self.write_bytes(&value.to_le_bytes())
    }

    fn write_u64(&mut self, value: u64) -> Result<(), Box<dyn Error>> {
        self.write_bytes(&value.to_le_bytes())
    }

    fn write_f32(&mut self, value: f32) -> Result<(), Box<dyn Error>> {
        self.write_bytes(&value.to_le_bytes())
    }

    fn finish(
        mut self,
        dtype: &'static str,
        shape: Vec<usize>,
    ) -> Result<FileSpec, Box<dyn Error>> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        Ok(FileSpec {
            file: self.file_name,
            dtype,
            shape,
            bytes: self.bytes,
            blake3: self.hasher.finalize().to_hex().to_string(),
        })
    }
}

struct SplitWriters {
    group_ids: HashedWriter,
    public_state_hashes: HashedWriter,
    source_candidate_counts: HashedWriter,
    retained_candidate_counts: HashedWriter,
    selected_source_indices: HashedWriter,
    champion_source_indices: HashedWriter,
    cohort_hashes: HashedWriter,
    candidate_identity_hashes: HashedWriter,
    candidate_offsets: HashedWriter,
    parent_token_types: HashedWriter,
    parent_token_seats: HashedWriter,
    parent_token_payload: HashedWriter,
    parent_board_type_counts: HashedWriter,
    parent_market_features: HashedWriter,
    parent_market_mask: HashedWriter,
    parent_player_features: HashedWriter,
    parent_player_mask: HashedWriter,
    parent_global_features: HashedWriter,
    source_candidate_indices: HashedWriter,
    action_hashes: HashedWriter,
    canonical_transform_ids: HashedWriter,
    transformed_centers: HashedWriter,
    control_after_hashes: HashedWriter,
    control_remove_offsets: HashedWriter,
    control_remove_indices: HashedWriter,
    control_add_offsets: HashedWriter,
    control_add_types: HashedWriter,
    control_add_payload: HashedWriter,
    r3_token_offsets: HashedWriter,
    r3_token_types: HashedWriter,
    r3_token_operations: HashedWriter,
    r3_token_payload: HashedWriter,
    retained_candidates: usize,
    control_removed_tokens: usize,
    control_added_tokens: usize,
    r3_tokens: usize,
}

impl SplitWriters {
    fn create(root: &Path, split: &str) -> Result<Self, Box<dyn Error>> {
        let writer = |name: &str| HashedWriter::create(root, format!("{split}-{name}.bin"));
        let mut result = Self {
            group_ids: writer("group-ids")?,
            public_state_hashes: writer("public-state-hashes")?,
            source_candidate_counts: writer("source-candidate-counts")?,
            retained_candidate_counts: writer("retained-candidate-counts")?,
            selected_source_indices: writer("selected-source-indices")?,
            champion_source_indices: writer("champion-source-indices")?,
            cohort_hashes: writer("cohort-hashes")?,
            candidate_identity_hashes: writer("candidate-identity-hashes")?,
            candidate_offsets: writer("candidate-offsets")?,
            parent_token_types: writer("parent-token-types")?,
            parent_token_seats: writer("parent-token-seats")?,
            parent_token_payload: writer("parent-token-payload")?,
            parent_board_type_counts: writer("parent-board-type-counts")?,
            parent_market_features: writer("parent-market-features")?,
            parent_market_mask: writer("parent-market-mask")?,
            parent_player_features: writer("parent-player-features")?,
            parent_player_mask: writer("parent-player-mask")?,
            parent_global_features: writer("parent-global-features")?,
            source_candidate_indices: writer("source-candidate-indices")?,
            action_hashes: writer("action-hashes")?,
            canonical_transform_ids: writer("canonical-transform-ids")?,
            transformed_centers: writer("transformed-centers")?,
            control_after_hashes: writer("control-after-hashes")?,
            control_remove_offsets: writer("control-remove-offsets")?,
            control_remove_indices: writer("control-remove-indices")?,
            control_add_offsets: writer("control-add-offsets")?,
            control_add_types: writer("control-add-types")?,
            control_add_payload: writer("control-add-payload")?,
            r3_token_offsets: writer("r3-token-offsets")?,
            r3_token_types: writer("r3-token-types")?,
            r3_token_operations: writer("r3-token-operations")?,
            r3_token_payload: writer("r3-token-payload")?,
            retained_candidates: 0,
            control_removed_tokens: 0,
            control_added_tokens: 0,
            r3_tokens: 0,
        };
        result.candidate_offsets.write_u64(0)?;
        result.control_remove_offsets.write_u64(0)?;
        result.control_add_offsets.write_u64(0)?;
        result.r3_token_offsets.write_u64(0)?;
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn write_group(
        &mut self,
        group: &GradedOracleGroup,
        retained: &[usize],
        cohort_hash: [u8; 32],
        parent: &MlxEncodedState,
        market_features: &[f32; 4 * MARKET_FEATURES],
        market_mask: &[u8; 4],
        player_features: &[f32; BOARD_SLOTS * PLAYER_FEATURES],
        player_mask: &[u8; BOARD_SLOTS],
        global_features: &[f32; GLOBAL_FEATURES],
    ) -> Result<(), Box<dyn Error>> {
        self.group_ids.write_u64(group.group_id)?;
        self.public_state_hashes
            .write_bytes(&group.public_state_hash)?;
        self.source_candidate_counts
            .write_u16(u16::try_from(group.candidates.len())?)?;
        self.retained_candidate_counts
            .write_u16(u16::try_from(retained.len())?)?;
        self.selected_source_indices
            .write_u16(group.selected_index)?;
        self.champion_source_indices
            .write_u16(group.champion_index)?;
        self.cohort_hashes.write_bytes(&cohort_hash)?;
        self.parent_token_types.write_bytes(&parent.token_types)?;
        self.parent_token_seats.write_bytes(&parent.token_seats)?;
        for value in &parent.token_payload {
            self.parent_token_payload.write_i8(*value)?;
        }
        for count in parent.board_type_counts.iter().flatten() {
            self.parent_board_type_counts.write_u16(*count)?;
        }
        for value in market_features {
            self.parent_market_features.write_f32(*value)?;
        }
        self.parent_market_mask.write_bytes(market_mask)?;
        for value in player_features {
            self.parent_player_features.write_f32(*value)?;
        }
        self.parent_player_mask.write_bytes(player_mask)?;
        for value in global_features {
            self.parent_global_features.write_f32(*value)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn write_candidate(
        &mut self,
        source_index: usize,
        action_hash: [u8; 32],
        transform_id: u8,
        transformed_center: (i8, i8),
        delta: &ControlDelta,
        r3: &MlxActionEncoding,
        group_identity: &mut Hasher,
    ) -> Result<(), Box<dyn Error>> {
        self.source_candidate_indices
            .write_u16(u16::try_from(source_index)?)?;
        self.action_hashes.write_bytes(&action_hash)?;
        self.canonical_transform_ids.write_u8(transform_id)?;
        self.transformed_centers.write_i8(transformed_center.0)?;
        self.transformed_centers.write_i8(transformed_center.1)?;
        self.control_after_hashes
            .write_bytes(&delta.after_multiset_blake3)?;

        self.control_remove_indices
            .write_bytes(&delta.remove_indices)?;
        self.control_removed_tokens += delta.remove_indices.len();
        self.control_remove_offsets
            .write_u64(self.control_removed_tokens as u64)?;

        for token in &delta.added {
            self.control_add_types.write_u8(token.token_type)?;
            for value in token.payload {
                self.control_add_payload.write_i8(value)?;
            }
        }
        self.control_added_tokens += delta.added.len();
        self.control_add_offsets
            .write_u64(self.control_added_tokens as u64)?;

        let r3_hash = r3_token_blake3(r3);
        for token in &r3.tokens {
            self.r3_token_types.write_u8(token.token_type)?;
            self.r3_token_operations.write_u8(token.operation)?;
            for value in token.payload {
                self.r3_token_payload.write_i8(value)?;
            }
        }
        self.r3_tokens += r3.tokens.len();
        self.r3_token_offsets.write_u64(self.r3_tokens as u64)?;
        self.retained_candidates += 1;

        group_identity.update(&u16::try_from(source_index)?.to_le_bytes());
        group_identity.update(&action_hash);
        group_identity.update(&delta.after_multiset_blake3);
        group_identity.update(&r3_hash);
        Ok(())
    }

    fn finish_group(&mut self, identity: [u8; 32]) -> Result<(), Box<dyn Error>> {
        self.candidate_identity_hashes.write_bytes(&identity)?;
        self.candidate_offsets
            .write_u64(self.retained_candidates as u64)?;
        Ok(())
    }

    fn finish(self, groups: usize) -> Result<BTreeMap<String, FileSpec>, Box<dyn Error>> {
        let candidates = self.retained_candidates;
        let removed = self.control_removed_tokens;
        let added = self.control_added_tokens;
        let r3_tokens = self.r3_tokens;
        let mut files = BTreeMap::new();
        macro_rules! finish {
            ($field:ident, $dtype:literal, $shape:expr) => {
                files.insert(
                    stringify!($field).to_owned(),
                    self.$field.finish($dtype, $shape)?,
                );
            };
        }
        finish!(group_ids, "<u8", vec![groups]);
        finish!(public_state_hashes, "|u1", vec![groups, 32]);
        finish!(source_candidate_counts, "<u2", vec![groups]);
        finish!(retained_candidate_counts, "<u2", vec![groups]);
        finish!(selected_source_indices, "<u2", vec![groups]);
        finish!(champion_source_indices, "<u2", vec![groups]);
        finish!(cohort_hashes, "|u1", vec![groups, 32]);
        finish!(candidate_identity_hashes, "|u1", vec![groups, 32]);
        finish!(candidate_offsets, "<u8", vec![groups + 1]);
        finish!(
            parent_token_types,
            "|u1",
            vec![groups, BOARD_SLOTS, BOARD_TOKEN_CAPACITY]
        );
        finish!(
            parent_token_seats,
            "|u1",
            vec![groups, BOARD_SLOTS, BOARD_TOKEN_CAPACITY]
        );
        finish!(
            parent_token_payload,
            "|i1",
            vec![
                groups,
                BOARD_SLOTS,
                BOARD_TOKEN_CAPACITY,
                TOKEN_PAYLOAD_WIDTH
            ]
        );
        finish!(
            parent_board_type_counts,
            "<u2",
            vec![groups, BOARD_SLOTS, 4]
        );
        finish!(
            parent_market_features,
            "<f4",
            vec![groups, 4, MARKET_FEATURES]
        );
        finish!(parent_market_mask, "|u1", vec![groups, 4]);
        finish!(
            parent_player_features,
            "<f4",
            vec![groups, BOARD_SLOTS, PLAYER_FEATURES]
        );
        finish!(parent_player_mask, "|u1", vec![groups, BOARD_SLOTS]);
        finish!(parent_global_features, "<f4", vec![groups, GLOBAL_FEATURES]);
        finish!(source_candidate_indices, "<u2", vec![candidates]);
        finish!(action_hashes, "|u1", vec![candidates, 32]);
        finish!(canonical_transform_ids, "|u1", vec![candidates]);
        finish!(transformed_centers, "|i1", vec![candidates, 2]);
        finish!(control_after_hashes, "|u1", vec![candidates, 32]);
        finish!(control_remove_offsets, "<u8", vec![candidates + 1]);
        finish!(control_remove_indices, "|u1", vec![removed]);
        finish!(control_add_offsets, "<u8", vec![candidates + 1]);
        finish!(control_add_types, "|u1", vec![added]);
        finish!(control_add_payload, "|i1", vec![added, TOKEN_PAYLOAD_WIDTH]);
        finish!(r3_token_offsets, "<u8", vec![candidates + 1]);
        finish!(r3_token_types, "|u1", vec![r3_tokens]);
        finish!(r3_token_operations, "|u1", vec![r3_tokens]);
        finish!(
            r3_token_payload,
            "|i1",
            vec![r3_tokens, MLX_ACTION_TOKEN_PAYLOAD_WIDTH]
        );
        Ok(files)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if args.max_groups_per_split == Some(0) {
        return Err("--max-groups-per-split must be positive".into());
    }
    fs::create_dir_all(&args.output_root)?;
    let temporary = args.output_root.join(format!(
        ".tmp-r3-action-edit-mlx-{}-{}",
        std::process::id(),
        unix_millis()?
    ));
    if temporary.exists() {
        fs::remove_dir_all(&temporary)?;
    }
    fs::create_dir(&temporary)?;

    let result = (|| -> Result<(PathBuf, CacheManifest), Box<dyn Error>> {
        let train = export_split(
            &args.train_dataset,
            DatasetSplit::Train,
            &temporary,
            args.max_groups_per_split,
        )?;
        let validation = export_split(
            &args.validation_dataset,
            DatasetSplit::Validation,
            &temporary,
            args.max_groups_per_split,
        )?;
        let executable_blake3 = checksum_file(&env::current_exe()?)?;
        let exporter = ExporterIdentity {
            executable_blake3,
            source: source_provenance()?,
        };
        let splits = BTreeMap::from([
            ("train".to_owned(), train),
            ("validation".to_owned(), validation),
        ]);
        let complete_open_corpus = splits.values().all(|split| split.complete_open_split);
        let tensor_contract = tensor_contract();
        let hidden_information = HiddenInformationBoundary {
            open_train_and_validation_only: true,
            source_seed_used_for_authoritative_replay: true,
            hidden_order_exported: false,
            excluded_tile_identity_exported: false,
            future_refill_exported: false,
            sealed_test_opened: false,
            gameplay_opened: false,
        };
        let scientific_identity = json!({
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_schema": CACHE_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "protocol_id": PROTOCOL_ID,
            "adr": ADR_ID,
            "complete_open_corpus": complete_open_corpus,
            "exporter": exporter.clone(),
            "tensor_contract": tensor_contract.clone(),
            "hidden_information": hidden_information.clone(),
            "splits": splits.clone(),
        });
        let cache_id = canonical_blake3(&scientific_identity)?;
        let manifest = CacheManifest {
            schema_version: CACHE_SCHEMA_VERSION,
            cache_schema: CACHE_SCHEMA,
            experiment_id: EXPERIMENT_ID,
            protocol_id: PROTOCOL_ID,
            adr: ADR_ID,
            cache_id: cache_id.clone(),
            complete_open_corpus,
            exporter,
            tensor_contract,
            hidden_information,
            splits,
            scientific_identity,
        };
        write_json_atomic(&temporary.join("cache.json"), &manifest)?;
        let final_root = args.output_root.join(&cache_id);
        if final_root.exists() {
            let existing = fs::read(final_root.join("cache.json"))?;
            let generated = fs::read(temporary.join("cache.json"))?;
            if existing != generated {
                return Err(format!(
                    "content-address collision or provenance drift at {}",
                    final_root.display()
                )
                .into());
            }
            fs::remove_dir_all(&temporary)?;
        } else {
            fs::rename(&temporary, &final_root)?;
        }
        Ok((final_root, manifest))
    })();

    match result {
        Ok((cache_root, manifest)) => {
            let receipt = json!({
                "schema_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "protocol_id": PROTOCOL_ID,
                "adr": ADR_ID,
                "cache_id": manifest.cache_id,
                "cache_root": cache_root,
                "complete_open_corpus": manifest.complete_open_corpus,
                "train_groups": manifest.splits["train"].groups,
                "train_candidates": manifest.splits["train"].retained_candidates,
                "validation_groups": manifest.splits["validation"].groups,
                "validation_candidates": manifest.splits["validation"].retained_candidates,
            });
            write_json_atomic(&args.receipt, &receipt)?;
            println!("{}", serde_json::to_string(&receipt)?);
            Ok(())
        }
        Err(error) => {
            fs::remove_dir_all(&temporary).ok();
            Err(error)
        }
    }
}

fn export_split(
    root: &Path,
    expected_split: DatasetSplit,
    output: &Path,
    maximum_groups: Option<usize>,
) -> Result<SplitManifest, Box<dyn Error>> {
    let manifest_path = root.join("dataset.json");
    let manifest_bytes = fs::read(&manifest_path)?;
    let manifest: GradedOracleDatasetManifest = serde_json::from_slice(&manifest_bytes)?;
    validate_graded_oracle_dataset(root, &manifest)?;
    if manifest.split != expected_split {
        return Err(format!(
            "dataset {} has split {:?}, expected {:?}",
            root.display(),
            manifest.split,
            expected_split
        )
        .into());
    }
    if matches!(expected_split, DatasetSplit::Test | DatasetSplit::Final) {
        return Err("R3 MLX exporter prohibits test and final splits".into());
    }

    let split_name = match expected_split {
        DatasetSplit::Train => "train",
        DatasetSplit::Validation => "validation",
        DatasetSplit::Test | DatasetSplit::Final => unreachable!(),
    };
    let mut writers = SplitWriters::create(output, split_name)?;
    let mut checks = SplitChecks {
        minimum_control_tokens: usize::MAX,
        minimum_r3_tokens: usize::MAX,
        ..SplitChecks::default()
    };
    let mut groups = 0usize;
    let mut source_candidates = 0usize;
    let mut stop = false;

    for shard in &manifest.shards {
        if stop {
            break;
        }
        let shard_groups = read_graded_oracle_shard(root, expected_split, shard)?;
        let mut game = GameState::new(
            GameConfig::research_aaaaa(4)?,
            GameSeed::from_u64(shard.first_game_index),
        )?;
        for group in shard_groups {
            if maximum_groups.is_some_and(|maximum| groups >= maximum) {
                stop = true;
                break;
            }
            source_candidates += group.candidates.len();
            process_group(expected_split, &mut game, &group, &mut writers, &mut checks)?;
            let champion = group.candidates[usize::from(group.champion_index)]
                .action
                .to_game_action(&game)?;
            game.apply(&champion)?;
            groups += 1;
        }
        if !stop && !game.is_game_over() {
            return Err(format!(
                "source game {} did not contain 80 complete decisions",
                shard.first_game_index
            )
            .into());
        }
    }

    let complete_open_split = maximum_groups.is_none()
        && match expected_split {
            DatasetSplit::Train => {
                groups == TRAIN_GROUPS && source_candidates == TRAIN_SOURCE_CANDIDATES
            }
            DatasetSplit::Validation => {
                groups == VALIDATION_GROUPS && source_candidates == VALIDATION_CANDIDATES
            }
            DatasetSplit::Test | DatasetSplit::Final => false,
        };
    if maximum_groups.is_none() && !complete_open_split {
        return Err(
            format!("{split_name} export did not cover the complete frozen open split").into(),
        );
    }
    if checks.minimum_control_tokens == usize::MAX {
        checks.minimum_control_tokens = 0;
    }
    if checks.minimum_r3_tokens == usize::MAX {
        checks.minimum_r3_tokens = 0;
    }
    let retained_candidates = writers.retained_candidates;
    let control_removed_tokens = writers.control_removed_tokens;
    let control_added_tokens = writers.control_added_tokens;
    let r3_tokens = writers.r3_tokens;
    let files = writers.finish(groups)?;
    Ok(SplitManifest {
        split: split_name,
        dataset_id: manifest.dataset_id,
        dataset_manifest_blake3: blake3::hash(&manifest_bytes).to_hex().to_string(),
        groups,
        source_candidates,
        retained_candidates,
        complete_open_split,
        control_removed_tokens,
        control_added_tokens,
        r3_tokens,
        files,
        checks,
    })
}

fn process_group(
    split: DatasetSplit,
    game: &mut GameState,
    group: &GradedOracleGroup,
    writers: &mut SplitWriters,
    checks: &mut SplitChecks,
) -> Result<(), Box<dyn Error>> {
    if game.completed_turns() != group.completed_turns
        || game.current_player() != usize::from(group.current_player)
        || PositionRecord::observe(game, group.raw_seed).to_bytes() != group.position.to_bytes()
    {
        return Err(format!("graded replay drifted at group {}", group.group_id).into());
    }
    checks.position_record_checks += 1;
    if *game.public_state().canonical_hash().as_bytes() != group.public_state_hash {
        return Err(format!("public-state hash drifted at group {}", group.group_id).into());
    }
    checks.public_state_hash_checks += 1;
    if game.public_supply() != group.public_supply {
        return Err(format!("public supply drifted at group {}", group.group_id).into());
    }
    checks.public_supply_checks += 1;

    let retained = match split {
        DatasetSplit::Train => select_train_candidate_indices(group)?,
        DatasetSplit::Validation => (0..group.candidates.len()).collect(),
        DatasetSplit::Test | DatasetSplit::Final => unreachable!(),
    };
    let cohort_hash = cohort_blake3(group, &retained)?;
    if retained.contains(&usize::from(group.selected_index)) {
        checks.selected_winner_retained += 1;
    }
    if retained.contains(&usize::from(group.champion_index)) {
        checks.champion_retained += 1;
    }
    let source_r600 = group
        .candidates
        .iter()
        .filter(|candidate| candidate.r600.samples > 0)
        .count();
    let retained_r600 = retained
        .iter()
        .filter(|index| group.candidates[**index].r600.samples > 0)
        .count();
    let source_r1200 = group
        .candidates
        .iter()
        .filter(|candidate| candidate.r1200.samples > 0)
        .count();
    let retained_r1200 = retained
        .iter()
        .filter(|index| group.candidates[**index].r1200.samples > 0)
        .count();
    let source_r4800 = group
        .candidates
        .iter()
        .filter(|candidate| candidate.r4800.samples > 0)
        .count();
    let retained_r4800 = retained
        .iter()
        .filter(|index| group.candidates[**index].r4800.samples > 0)
        .count();
    if retained_r600 != source_r600 || retained_r4800 != source_r4800 {
        return Err(format!(
            "mandatory fidelity labels were omitted at group {}: \
             R600 {retained_r600}/{source_r600}, R4800 {retained_r4800}/{source_r4800}",
            group.group_id
        )
        .into());
    }
    if split == DatasetSplit::Validation && retained_r1200 != source_r1200 {
        return Err(format!(
            "validation R1200 labels were omitted at group {}: {retained_r1200}/{source_r1200}",
            group.group_id
        )
        .into());
    }
    checks.source_r600 += source_r600;
    checks.r600_retained += retained_r600;
    checks.source_r1200 += source_r1200;
    checks.r1200_retained += retained_r1200;
    checks.source_r4800 += source_r4800;
    checks.r4800_retained += retained_r4800;

    let parent_state = SparsePublicState::from_position_record(&group.position, None)?;
    let parent_encoded = encode_sparse_state(&parent_state)?;
    checks.parent_r2_encodings += 1;
    let (market_features, market_mask) = encode_market_features(&group.position)?;
    let (player_features, player_mask) = encode_player_features(&group.position)?;
    let global_features = encode_global_features(&group.position)?;
    writers.write_group(
        group,
        &retained,
        cohort_hash,
        &parent_encoded,
        &market_features,
        &market_mask,
        &player_features,
        &player_mask,
        &global_features,
    )?;

    let trunk = PublicStateTrunk::observe(game, group.raw_seed)?;
    let prepared = trunk.prepare_action_edits()?;
    let mut selected_actions = Vec::with_capacity(retained.len());
    let mut batches = BTreeMap::<DraftBatchKey, Vec<(usize, TurnAction)>>::new();
    for (retained_position, source_index) in retained.iter().copied().enumerate() {
        let candidate = &group.candidates[source_index];
        let action = candidate.action.to_game_action(game)?;
        if canonical_action_hash(&action)? != candidate.action_hash {
            return Err(format!(
                "graded action hash drifted at group {} candidate {}",
                group.group_id, source_index
            )
            .into());
        }
        checks.graded_action_reconstructions += 1;
        batches
            .entry(draft_batch_key(&action))
            .or_default()
            .push((retained_position, action.clone()));
        selected_actions.push(action);
    }

    let mut edits = vec![None; retained.len()];
    for targets in batches.into_values() {
        let prelude = targets[0].1.prelude();
        let draft = targets[0].1.draft;
        let target_positions = targets
            .into_iter()
            .map(|(position, action)| (action, position))
            .collect::<HashMap<_, _>>();
        for (action, edit) in prepared.observe_draft_actions(game, &prelude, draft)? {
            if let Some(position) = target_positions.get(&action).copied() {
                if edits[position].replace(edit).is_some() {
                    return Err("grouped R3 enumeration emitted a duplicate target".into());
                }
                checks.grouped_r3_action_matches += 1;
            }
        }
    }
    if edits.iter().any(Option::is_none) {
        return Err(format!(
            "grouped R3 enumeration missed retained actions in group {}",
            group.group_id
        )
        .into());
    }

    let mut transformed_parents = (0..12)
        .map(|_| None)
        .collect::<Vec<Option<MlxEncodedState>>>();
    let mut group_identity = Hasher::new();
    group_identity.update(CANDIDATE_IDENTITY_DOMAIN);
    group_identity.update(&group.group_id.to_le_bytes());
    group_identity.update(&(retained.len() as u64).to_le_bytes());

    for (retained_position, source_index) in retained.iter().copied().enumerate() {
        let action = &selected_actions[retained_position];
        let edit: ActionEdit = edits[retained_position].take().unwrap();
        let applied = prepared.apply(&edit)?;
        checks.r3_apply_checks += 1;
        let authoritative = game.preview_public_afterstate(action)?;
        let authoritative_record = PositionRecord::observe_public_for_seat(
            &authoritative,
            group.raw_seed,
            game.current_player(),
        );
        if applied.record.to_bytes() != authoritative_record.to_bytes() {
            return Err(format!(
                "R3 public successor drifted in group {} candidate {}",
                group.group_id, source_index
            )
            .into());
        }
        checks.authoritative_successor_checks += 1;

        let transform_id = prepared.canonical_transform_id(&edit)?;
        let transform = cascadia_game::D6Transform::from_id(transform_id)
            .ok_or("R3 canonical transform ID is outside [0, 11]")?;
        let transformed_coord = transform.transform_coord(action.tile.coord)?;
        let center = (transformed_coord.q, transformed_coord.r);
        checks.canonical_transform_checks += 1;

        let transformed_parent =
            if let Some(parent) = &transformed_parents[usize::from(transform_id)] {
                parent
            } else {
                transformed_parents[usize::from(transform_id)] =
                    Some(transform_encoded_state(&parent_encoded, transform)?);
                transformed_parents[usize::from(transform_id)]
                    .as_ref()
                    .unwrap()
            };
        let parent_tokens = active_board_tokens_relative(transformed_parent, center)?;

        let mut geometry_record = applied.record.clone();
        geometry_record.market_entities = group.position.market_entities;
        let after_state = SparsePublicState::from_position_record(&geometry_record, None)?;
        let after_encoded =
            transform_encoded_state(&encode_sparse_state(&after_state)?, transform)?;
        checks.r2_afterstate_encodings += 1;
        let after_tokens = active_board_tokens_relative(&after_encoded, center)?;
        let delta = control_delta(&parent_tokens, &after_tokens)?;
        checks.control_delta_round_trips += 1;
        checks.minimum_control_tokens = checks.minimum_control_tokens.min(after_tokens.len());
        checks.maximum_control_tokens = checks.maximum_control_tokens.max(after_tokens.len());

        let r3 = edit.mlx_action_encoding()?;
        if r3.decode_canonical_view()? != edit.canonical {
            return Err("R3 token stream changed the canonical action view".into());
        }
        checks.r3_token_round_trips += 1;
        checks.minimum_r3_tokens = checks.minimum_r3_tokens.min(r3.tokens.len());
        checks.maximum_r3_tokens = checks.maximum_r3_tokens.max(r3.tokens.len());

        writers.write_candidate(
            source_index,
            group.candidates[source_index].action_hash,
            transform_id,
            center,
            &delta,
            &r3,
            &mut group_identity,
        )?;
    }
    writers.finish_group(*group_identity.finalize().as_bytes())?;
    checks.groups_replayed += 1;
    Ok(())
}

fn draft_batch_key(action: &TurnAction) -> DraftBatchKey {
    let wipe_masks = action
        .wildlife_wipes
        .iter()
        .map(|wipe| {
            wipe.slots
                .iter()
                .fold(0u8, |mask, slot| mask | (1 << slot.index()))
        })
        .collect();
    let (draft_kind, tile_slot, wildlife_slot) = match action.draft {
        DraftChoice::Paired { slot } => (0, slot.index() as u8, slot.index() as u8),
        DraftChoice::Independent {
            tile_slot,
            wildlife_slot,
        } => (1, tile_slot.index() as u8, wildlife_slot.index() as u8),
    };
    DraftBatchKey {
        replace_three_of_a_kind: action.replace_three_of_a_kind,
        wipe_masks,
        draft_kind,
        tile_slot,
        wildlife_slot,
    }
}

fn canonical_action_hash(action: &TurnAction) -> Result<[u8; 32], Box<dyn Error>> {
    let mut hasher = Hasher::new();
    hasher.update(b"cascadia-v2-full-legal-action-v1");
    hasher.update(&serde_json::to_vec(action)?);
    Ok(*hasher.finalize().as_bytes())
}

fn r3_token_blake3(encoding: &MlxActionEncoding) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(R3_TOKEN_DOMAIN);
    hasher.update(&encoding.schema_version.to_le_bytes());
    hasher.update(&(encoding.tokens.len() as u64).to_le_bytes());
    for token in &encoding.tokens {
        hasher.update(&[token.token_type, token.operation]);
        hasher.update(
            &token
                .payload
                .into_iter()
                .map(|value| value as u8)
                .collect::<Vec<_>>(),
        );
    }
    *hasher.finalize().as_bytes()
}

fn tensor_contract() -> Value {
    json!({
        "parent": {
            "boards": BOARD_SLOTS,
            "tokens_per_board": BOARD_TOKEN_CAPACITY,
            "token_payload_width": TOKEN_PAYLOAD_WIDTH,
            "market_feature_dim": MARKET_FEATURES,
            "player_feature_dim": PLAYER_FEATURES,
            "global_feature_dim": GLOBAL_FEATURES,
            "one_parent_encoding_per_group": true,
        },
        "candidate": {
            "train_candidate_cap": 512,
            "validation_is_complete": true,
            "control": "canonical-parent-multiset-removals-plus-exact-additions",
            "r3_payload_width": MLX_ACTION_TOKEN_PAYLOAD_WIDTH,
            "r3_radius_three_cached_once": true,
            "radius_one_and_two": "exact-loader-crop-of-local-patch-tokens",
            "silent_truncation": false,
        },
    })
}

fn canonical_blake3(value: &impl Serialize) -> Result<String, Box<dyn Error>> {
    Ok(blake3::hash(&serde_json::to_vec(value)?)
        .to_hex()
        .to_string())
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    let bytes = [serde_json::to_vec_pretty(value)?.as_slice(), b"\n"].concat();
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn unix_millis() -> Result<u128, Box<dyn Error>> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis())
}
