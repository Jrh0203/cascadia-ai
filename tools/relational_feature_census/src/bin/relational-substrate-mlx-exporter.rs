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
use cascadia_game::{
    D6Transform, DraftChoice, GameConfig, GameSeed, GameState, TurnAction, score_board,
};
use cascadia_provenance::{SourceProvenance, checksum_file, source_provenance};
use clap::Parser;
use r3_action_edit_census::{ActionEdit, PublicStateTrunk};
use rayon::prelude::*;
use relational_feature_census::{
    OpportunityDerivativeContext, RELATIONAL_TOKEN_CLASS_COUNT, RELATIONAL_TOKEN_VALUE_WIDTH,
    RelationalStateGraph, opportunity_derivative_features, opportunity_derivative_values,
    rich_relational_parent_tokens, write_json_atomic,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const CACHE_SCHEMA_VERSION: u16 = 1;
const CACHE_SCHEMA: &str = "relational-substrate-mlx-cache-v1";
const EXPERIMENT_ID: &str = "relational-substrate-mlx-tournament-v1";
const PROTOCOL_ID: &str = "r5-s3-s5-matched-mlx-v1";
const ADR_ID: &str = "0161";
const EXPECTED_R3_CACHE_ID: &str =
    "0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156";
const EXPECTED_R3_CACHE_SCHEMA: &str = "r3-action-edit-mlx-cache-v1";
const TRAIN_GROUPS: usize = 560;
const VALIDATION_GROUPS: usize = 240;
const TRAIN_SOURCE_CANDIDATES: usize = 2_135_111;
const VALIDATION_SOURCE_CANDIDATES: usize = 860_203;
const TRAIN_RETAINED_CANDIDATES: usize = 280_012;
const VALIDATION_RETAINED_CANDIDATES: usize = 860_203;
const D6_TRANSFORMS: usize = 12;
const BOARD_SLOTS: usize = 4;
const S5_FEATURES: usize = 154;

#[derive(Debug, Parser)]
#[command(about = "Export the exact ADR 0161 relational MLX sidecar")]
struct Args {
    #[arg(long)]
    train_dataset: PathBuf,
    #[arg(long)]
    validation_dataset: PathBuf,
    #[arg(long)]
    r3_cache: PathBuf,
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
    r3_group_id_checks: usize,
    r3_public_state_hash_checks: usize,
    r3_candidate_count_checks: usize,
    r3_candidate_identity_checks: usize,
    position_record_checks: usize,
    public_state_hash_checks: usize,
    public_supply_checks: usize,
    current_score_checks: usize,
    current_score_failures: usize,
    parent_views: usize,
    parent_tokens: usize,
    minimum_parent_tokens: usize,
    maximum_parent_tokens: usize,
    retained_action_hash_checks: usize,
    grouped_action_matches: usize,
    afterstate_hash_checks: usize,
    score_delta_checks: usize,
    s5_width_checks: usize,
    i16_storage_checks: usize,
}

#[derive(Debug, Clone, Serialize)]
struct SplitManifest {
    split: &'static str,
    dataset_id: String,
    dataset_manifest_blake3: String,
    groups: usize,
    source_candidates: usize,
    retained_candidates: usize,
    parent_views: usize,
    parent_tokens: usize,
    complete_open_split: bool,
    files: BTreeMap<String, FileSpec>,
    checks: SplitChecks,
}

#[derive(Debug, Clone, Serialize)]
struct NormalizationSpec {
    index: usize,
    name: String,
    count: usize,
    nonzero_count: usize,
    minimum: i16,
    maximum: i16,
    p99_absolute: u16,
    maximum_absolute: u16,
    transform: &'static str,
    divisor: u16,
}

#[derive(Debug, Clone, Serialize)]
struct ExporterIdentity {
    executable_blake3: String,
    source: SourceProvenance,
}

#[derive(Debug, Clone, Serialize)]
struct R3CacheIdentity {
    cache_id: String,
    manifest_blake3: String,
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
    teacher_values_used_for_features: bool,
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
    r3_cache: R3CacheIdentity,
    tensor_contract: Value,
    normalization: Vec<NormalizationSpec>,
    hidden_information: HiddenInformationBoundary,
    splits: BTreeMap<String, SplitManifest>,
    scientific_identity: Value,
}

#[derive(Debug, Deserialize)]
struct R3Manifest {
    cache_id: String,
    cache_schema: String,
    complete_open_corpus: bool,
    splits: BTreeMap<String, R3SplitManifest>,
}

#[derive(Debug, Deserialize)]
struct R3SplitManifest {
    dataset_id: String,
    dataset_manifest_blake3: String,
    groups: usize,
    source_candidates: usize,
    retained_candidates: usize,
    files: BTreeMap<String, R3FileSpec>,
}

#[derive(Debug, Deserialize)]
struct R3FileSpec {
    file: String,
    dtype: String,
    shape: Vec<usize>,
    bytes: u64,
    blake3: String,
}

struct R3SplitBinding {
    dataset_id: String,
    dataset_manifest_blake3: String,
    groups: usize,
    retained_candidates: usize,
    group_ids: Vec<u64>,
    public_state_hashes: Vec<[u8; 32]>,
    source_candidate_counts: Vec<u16>,
    candidate_offsets: Vec<u64>,
    source_candidate_indices: Vec<u16>,
    action_hashes: Vec<[u8; 32]>,
    candidate_identity_hashes: Vec<[u8; 32]>,
}

struct R3Binding {
    identity: R3CacheIdentity,
    splits: BTreeMap<String, R3SplitBinding>,
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

    fn write_u16(&mut self, value: u16) -> Result<(), Box<dyn Error>> {
        self.write_bytes(&value.to_le_bytes())
    }

    fn write_i16(&mut self, value: i16) -> Result<(), Box<dyn Error>> {
        self.write_bytes(&value.to_le_bytes())
    }

    fn write_u64(&mut self, value: u64) -> Result<(), Box<dyn Error>> {
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
    candidate_identity_hashes: HashedWriter,
    opportunity_flags: HashedWriter,
    candidate_offsets: HashedWriter,
    source_candidate_indices: HashedWriter,
    action_hashes: HashedWriter,
    parent_view_offsets: HashedWriter,
    parent_token_classes: HashedWriter,
    parent_token_seats: HashedWriter,
    parent_token_values: HashedWriter,
    parent_view_counts: HashedWriter,
    parent_view_hashes: HashedWriter,
    s5_values: HashedWriter,
    retained_candidates: usize,
    parent_views: usize,
    parent_tokens: usize,
}

impl SplitWriters {
    fn create(root: &Path, split: &str) -> Result<Self, Box<dyn Error>> {
        let mut candidate_offsets =
            HashedWriter::create(root, format!("{split}-candidate-offsets.bin"))?;
        candidate_offsets.write_u64(0)?;
        let mut parent_view_offsets =
            HashedWriter::create(root, format!("{split}-parent-view-offsets.bin"))?;
        parent_view_offsets.write_u64(0)?;
        Ok(Self {
            group_ids: HashedWriter::create(root, format!("{split}-group-ids.bin"))?,
            public_state_hashes: HashedWriter::create(
                root,
                format!("{split}-public-state-hashes.bin"),
            )?,
            candidate_identity_hashes: HashedWriter::create(
                root,
                format!("{split}-candidate-identity-hashes.bin"),
            )?,
            opportunity_flags: HashedWriter::create(
                root,
                format!("{split}-opportunity-flags.bin"),
            )?,
            candidate_offsets,
            source_candidate_indices: HashedWriter::create(
                root,
                format!("{split}-source-candidate-indices.bin"),
            )?,
            action_hashes: HashedWriter::create(root, format!("{split}-action-hashes.bin"))?,
            parent_view_offsets,
            parent_token_classes: HashedWriter::create(
                root,
                format!("{split}-parent-token-classes.bin"),
            )?,
            parent_token_seats: HashedWriter::create(
                root,
                format!("{split}-parent-token-seats.bin"),
            )?,
            parent_token_values: HashedWriter::create(
                root,
                format!("{split}-parent-token-values.bin"),
            )?,
            parent_view_counts: HashedWriter::create(
                root,
                format!("{split}-parent-view-counts.bin"),
            )?,
            parent_view_hashes: HashedWriter::create(
                root,
                format!("{split}-parent-view-hashes.bin"),
            )?,
            s5_values: HashedWriter::create(root, format!("{split}-s5-values.bin"))?,
            retained_candidates: 0,
            parent_views: 0,
            parent_tokens: 0,
        })
    }

    fn write_group_identity(
        &mut self,
        group: &GradedOracleGroup,
        candidate_identity_hash: &[u8; 32],
        opportunity_flags: [u8; 4],
    ) -> Result<(), Box<dyn Error>> {
        self.group_ids.write_u64(group.group_id)?;
        self.public_state_hashes
            .write_bytes(&group.public_state_hash)?;
        self.candidate_identity_hashes
            .write_bytes(candidate_identity_hash)?;
        self.opportunity_flags.write_bytes(&opportunity_flags)?;
        Ok(())
    }

    fn write_parent_view(
        &mut self,
        tokens: &[relational_feature_census::RelationalParentToken],
    ) -> Result<(), Box<dyn Error>> {
        self.parent_view_counts
            .write_u16(u16::try_from(tokens.len())?)?;
        let mut digest = Hasher::new();
        digest.update(b"relational-substrate-parent-view-v1");
        digest.update(&(tokens.len() as u64).to_le_bytes());
        for token in tokens {
            if !(1..=RELATIONAL_TOKEN_CLASS_COUNT as u8).contains(&token.class_id)
                || usize::from(token.relative_seat) >= BOARD_SLOTS
            {
                return Err("relational parent token code is out of range".into());
            }
            self.parent_token_classes.write_u8(token.class_id)?;
            self.parent_token_seats.write_u8(token.relative_seat)?;
            digest.update(&[token.class_id, token.relative_seat]);
            for value in token.values {
                self.parent_token_values.write_i16(value)?;
                digest.update(&value.to_le_bytes());
            }
            self.parent_tokens += 1;
        }
        self.parent_views += 1;
        self.parent_view_offsets
            .write_u64(u64::try_from(self.parent_tokens)?)?;
        self.parent_view_hashes
            .write_bytes(digest.finalize().as_bytes())?;
        Ok(())
    }

    fn write_candidate(
        &mut self,
        source_index: u16,
        action_hash: &[u8; 32],
        values: &[i16],
    ) -> Result<(), Box<dyn Error>> {
        if values.len() != S5_FEATURES {
            return Err("S5 candidate vector width drifted".into());
        }
        self.source_candidate_indices.write_u16(source_index)?;
        self.action_hashes.write_bytes(action_hash)?;
        for value in values {
            self.s5_values.write_i16(*value)?;
        }
        self.retained_candidates += 1;
        Ok(())
    }

    fn finish_group(&mut self) -> Result<(), Box<dyn Error>> {
        self.candidate_offsets
            .write_u64(u64::try_from(self.retained_candidates)?)?;
        Ok(())
    }

    fn finish(self, groups: usize) -> Result<BTreeMap<String, FileSpec>, Box<dyn Error>> {
        let views = groups * D6_TRANSFORMS;
        if self.parent_views != views {
            return Err(format!(
                "relational parent writer emitted {} views instead of {views}",
                self.parent_views
            )
            .into());
        }
        Ok(BTreeMap::from([
            (
                "group_ids".to_owned(),
                self.group_ids.finish("<u8", vec![groups])?,
            ),
            (
                "public_state_hashes".to_owned(),
                self.public_state_hashes.finish("|u1", vec![groups, 32])?,
            ),
            (
                "candidate_identity_hashes".to_owned(),
                self.candidate_identity_hashes
                    .finish("|u1", vec![groups, 32])?,
            ),
            (
                "opportunity_flags".to_owned(),
                self.opportunity_flags.finish("|u1", vec![groups, 4])?,
            ),
            (
                "candidate_offsets".to_owned(),
                self.candidate_offsets.finish("<u8", vec![groups + 1])?,
            ),
            (
                "source_candidate_indices".to_owned(),
                self.source_candidate_indices
                    .finish("<u2", vec![self.retained_candidates])?,
            ),
            (
                "action_hashes".to_owned(),
                self.action_hashes
                    .finish("|u1", vec![self.retained_candidates, 32])?,
            ),
            (
                "parent_view_offsets".to_owned(),
                self.parent_view_offsets.finish("<u8", vec![views + 1])?,
            ),
            (
                "parent_token_classes".to_owned(),
                self.parent_token_classes
                    .finish("|u1", vec![self.parent_tokens])?,
            ),
            (
                "parent_token_seats".to_owned(),
                self.parent_token_seats
                    .finish("|u1", vec![self.parent_tokens])?,
            ),
            (
                "parent_token_values".to_owned(),
                self.parent_token_values.finish(
                    "<i2",
                    vec![self.parent_tokens, RELATIONAL_TOKEN_VALUE_WIDTH],
                )?,
            ),
            (
                "parent_view_counts".to_owned(),
                self.parent_view_counts
                    .finish("<u2", vec![groups, D6_TRANSFORMS])?,
            ),
            (
                "parent_view_hashes".to_owned(),
                self.parent_view_hashes
                    .finish("|u1", vec![groups, D6_TRANSFORMS, 32])?,
            ),
            (
                "s5_values".to_owned(),
                self.s5_values
                    .finish("<i2", vec![self.retained_candidates, S5_FEATURES])?,
            ),
        ]))
    }
}

struct SplitOutcome {
    manifest: SplitManifest,
    feature_names: Vec<String>,
    s5_path: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if args.max_groups_per_split == Some(0) {
        return Err("--max-groups-per-split must be positive".into());
    }
    fs::create_dir_all(&args.output_root)?;
    let r3 = load_r3_binding(&args.r3_cache)?;
    let temporary = args.output_root.join(format!(
        ".tmp-relational-substrate-mlx-{}-{}",
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
            &r3,
            &temporary,
            args.max_groups_per_split,
        )?;
        let validation = export_split(
            &args.validation_dataset,
            DatasetSplit::Validation,
            &r3,
            &temporary,
            args.max_groups_per_split,
        )?;
        if train.feature_names != validation.feature_names
            || train.feature_names.len() != S5_FEATURES
        {
            return Err("train and validation S5 schemas disagree".into());
        }
        let normalization = normalization_from_train(
            &train.s5_path,
            train.manifest.retained_candidates,
            &train.feature_names,
        )?;
        let exporter = ExporterIdentity {
            executable_blake3: checksum_file(&env::current_exe()?)?,
            source: source_provenance()?,
        };
        let splits = BTreeMap::from([
            ("train".to_owned(), train.manifest),
            ("validation".to_owned(), validation.manifest),
        ]);
        let complete_open_corpus = splits.values().all(|split| split.complete_open_split);
        let hidden_information = HiddenInformationBoundary {
            open_train_and_validation_only: true,
            source_seed_used_for_authoritative_replay: true,
            hidden_order_exported: false,
            excluded_tile_identity_exported: false,
            future_refill_exported: false,
            sealed_test_opened: false,
            gameplay_opened: false,
            teacher_values_used_for_features: false,
        };
        let tensor_contract = tensor_contract();
        let scientific_identity = json!({
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_schema": CACHE_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "protocol_id": PROTOCOL_ID,
            "adr": ADR_ID,
            "complete_open_corpus": complete_open_corpus,
            "exporter": exporter.clone(),
            "r3_cache": r3.identity.clone(),
            "tensor_contract": tensor_contract.clone(),
            "normalization": normalization.clone(),
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
            r3_cache: r3.identity.clone(),
            tensor_contract,
            normalization,
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
                "r3_cache_id": manifest.r3_cache.cache_id,
                "complete_open_corpus": manifest.complete_open_corpus,
                "train_groups": manifest.splits["train"].groups,
                "train_candidates": manifest.splits["train"].retained_candidates,
                "validation_groups": manifest.splits["validation"].groups,
                "validation_candidates": manifest.splits["validation"].retained_candidates,
                "s5_features": manifest.normalization.len(),
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
    r3: &R3Binding,
    output: &Path,
    maximum_groups: Option<usize>,
) -> Result<SplitOutcome, Box<dyn Error>> {
    if matches!(expected_split, DatasetSplit::Test | DatasetSplit::Final) {
        return Err("ADR 0161 exporter prohibits test and final splits".into());
    }
    let split_name = split_name(expected_split);
    let r3_split = r3
        .splits
        .get(split_name)
        .ok_or("R3 cache is missing an open split")?;
    let manifest_path = root.join("dataset.json");
    let manifest_bytes = fs::read(&manifest_path)?;
    let manifest: GradedOracleDatasetManifest = serde_json::from_slice(&manifest_bytes)?;
    validate_graded_oracle_dataset(root, &manifest)?;
    if manifest.split != expected_split
        || manifest.dataset_id != r3_split.dataset_id
        || blake3::hash(&manifest_bytes).to_hex().as_str() != r3_split.dataset_manifest_blake3
    {
        return Err(format!("{split_name} dataset identity disagrees with R3").into());
    }

    let mut writers = SplitWriters::create(output, split_name)?;
    let mut checks = SplitChecks {
        minimum_parent_tokens: usize::MAX,
        ..SplitChecks::default()
    };
    let mut feature_names: Option<Vec<String>> = None;
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
            process_group(
                &mut game,
                &group,
                groups,
                r3_split,
                &mut writers,
                &mut checks,
                &mut feature_names,
            )?;
            source_candidates += group.candidates.len();
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
    if checks.minimum_parent_tokens == usize::MAX {
        checks.minimum_parent_tokens = 0;
    }
    let expected = match expected_split {
        DatasetSplit::Train => (
            TRAIN_GROUPS,
            TRAIN_SOURCE_CANDIDATES,
            TRAIN_RETAINED_CANDIDATES,
        ),
        DatasetSplit::Validation => (
            VALIDATION_GROUPS,
            VALIDATION_SOURCE_CANDIDATES,
            VALIDATION_RETAINED_CANDIDATES,
        ),
        DatasetSplit::Test | DatasetSplit::Final => unreachable!(),
    };
    let complete_open_split = maximum_groups.is_none()
        && (groups, source_candidates, writers.retained_candidates) == expected;
    if maximum_groups.is_none() && !complete_open_split {
        return Err(format!(
            "{split_name} export covered ({groups}, {source_candidates}, {}) instead of {expected:?}",
            writers.retained_candidates,
        )
        .into());
    }
    let retained_candidates = writers.retained_candidates;
    let parent_views = writers.parent_views;
    let parent_tokens = writers.parent_tokens;
    let files = writers.finish(groups)?;
    let s5_path = output.join(
        files
            .get("s5_values")
            .ok_or("S5 output file is absent")?
            .file
            .as_str(),
    );
    Ok(SplitOutcome {
        manifest: SplitManifest {
            split: split_name,
            dataset_id: manifest.dataset_id,
            dataset_manifest_blake3: blake3::hash(&manifest_bytes).to_hex().to_string(),
            groups,
            source_candidates,
            retained_candidates,
            parent_views,
            parent_tokens,
            complete_open_split,
            files,
            checks,
        },
        feature_names: feature_names.ok_or("split emitted no S5 feature schema")?,
        s5_path,
    })
}

#[allow(clippy::too_many_arguments)]
fn process_group(
    game: &mut GameState,
    group: &GradedOracleGroup,
    row: usize,
    r3: &R3SplitBinding,
    writers: &mut SplitWriters,
    checks: &mut SplitChecks,
    feature_names: &mut Option<Vec<String>>,
) -> Result<(), Box<dyn Error>> {
    if row >= r3.groups || group.group_id != r3.group_ids[row] {
        return Err(format!("R3 group ID drifted at row {row}").into());
    }
    checks.r3_group_id_checks += 1;
    if group.public_state_hash != r3.public_state_hashes[row] {
        return Err(format!("R3 public-state hash drifted at row {row}").into());
    }
    checks.r3_public_state_hash_checks += 1;
    if group.candidates.len() != usize::from(r3.source_candidate_counts[row]) {
        return Err(format!("R3 source candidate count drifted at row {row}").into());
    }
    checks.r3_candidate_count_checks += 1;
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

    let trunk = PublicStateTrunk::observe(game, group.raw_seed)?;
    let graph = RelationalStateGraph::from_sparse(&trunk.sparse)?;
    for (relative_seat, board) in graph.boards.iter().enumerate() {
        let absolute = (game.current_player() + relative_seat) % game.boards().len();
        let expected = score_board(&game.boards()[absolute], game.config().scoring_cards);
        let actual = board.score_anatomy();
        checks.current_score_checks += 1;
        if actual.habitat != expected.habitat
            || actual.wildlife != expected.wildlife
            || actual.nature_tokens != expected.nature_tokens
            || actual.base_total != expected.base_total
        {
            checks.current_score_failures += 1;
            return Err(format!("score decoder drifted at group {}", group.group_id).into());
        }
    }
    let opportunity = &graph.boards[0].opportunity;
    let opportunity_flags = [
        u8::from(opportunity.elk_eligible_extensions > 0),
        u8::from(opportunity.salmon_legal_continuations > 0),
        u8::from(opportunity.hawk_isolated_opportunities > 0),
        u8::from(opportunity.bear_pair_completion_cells > 0),
    ];
    writers.write_group_identity(group, &r3.candidate_identity_hashes[row], opportunity_flags)?;
    checks.r3_candidate_identity_checks += 1;

    for transform in D6Transform::ALL {
        let tokens = rich_relational_parent_tokens(&trunk.sparse, transform)?;
        checks.parent_views += 1;
        checks.parent_tokens += tokens.len();
        checks.minimum_parent_tokens = checks.minimum_parent_tokens.min(tokens.len());
        checks.maximum_parent_tokens = checks.maximum_parent_tokens.max(tokens.len());
        writers.write_parent_view(&tokens)?;
    }

    let start = usize::try_from(r3.candidate_offsets[row])?;
    let end = usize::try_from(r3.candidate_offsets[row + 1])?;
    if start >= end || end > r3.retained_candidates {
        return Err(format!("R3 retained candidate offsets drifted at row {row}").into());
    }
    let sources = &r3.source_candidate_indices[start..end];
    let expected_hashes = &r3.action_hashes[start..end];
    let prepared = trunk.prepare_action_edits()?;
    let mut selected_actions = Vec::with_capacity(sources.len());
    let mut batches = BTreeMap::<DraftBatchKey, Vec<(usize, TurnAction)>>::new();
    for (retained_position, source_index) in sources.iter().copied().enumerate() {
        let source = usize::from(source_index);
        let candidate = &group.candidates[source];
        let action = candidate.action.to_game_action(game)?;
        if candidate.action_hash != expected_hashes[retained_position] {
            return Err(format!(
                "R3 action hash drifted at group {} candidate {}",
                group.group_id, source
            )
            .into());
        }
        checks.retained_action_hash_checks += 1;
        batches
            .entry(draft_batch_key(&action))
            .or_default()
            .push((retained_position, action.clone()));
        selected_actions.push(action);
    }
    let mut edits = vec![None; sources.len()];
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
                checks.grouped_action_matches += 1;
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
    let edits = edits
        .into_iter()
        .map(Option::unwrap)
        .collect::<Vec<ActionEdit>>();
    let context = OpportunityDerivativeContext::from_trunk(&trunk)?;
    let rows = edits
        .par_iter()
        .map(|edit| -> Result<Vec<i16>, String> {
            let applied = prepared.apply(edit).map_err(|error| error.to_string())?;
            if applied.canonical_record_hash() != edit.expected_public_afterstate_blake3 {
                return Err("R3 afterstate hash drifted".to_owned());
            }
            let derivative = context
                .derive(&trunk, edit, &applied)
                .map_err(|error| error.to_string())?;
            let expected = [
                edit.score_delta.habitat[0],
                edit.score_delta.habitat[1],
                edit.score_delta.habitat[2],
                edit.score_delta.habitat[3],
                edit.score_delta.habitat[4],
                edit.score_delta.wildlife[0],
                edit.score_delta.wildlife[1],
                edit.score_delta.wildlife[2],
                edit.score_delta.wildlife[3],
                edit.score_delta.wildlife[4],
                edit.score_delta.nature_tokens,
                edit.score_delta.base_total,
            ];
            if derivative.immediate_score_delta != expected {
                return Err("S5 immediate score delta drifted".to_owned());
            }
            let values = opportunity_derivative_values(&derivative);
            if values.len() != S5_FEATURES {
                return Err("S5 feature width drifted".to_owned());
            }
            values
                .into_iter()
                .map(|value| {
                    i16::try_from(value)
                        .map_err(|_| "S5 value does not fit signed 16-bit storage".to_owned())
                })
                .collect()
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("group {}: {error}", group.group_id))?;
    if feature_names.is_none() {
        let applied = prepared.apply(&edits[0])?;
        let derivative = context.derive(&trunk, &edits[0], &applied)?;
        *feature_names = Some(
            opportunity_derivative_features(&derivative)
                .into_iter()
                .map(|(name, _)| name)
                .collect(),
        );
    }
    for (position, values) in rows.iter().enumerate() {
        writers.write_candidate(sources[position], &expected_hashes[position], values)?;
        checks.afterstate_hash_checks += 1;
        checks.score_delta_checks += 1;
        checks.s5_width_checks += 1;
        checks.i16_storage_checks += values.len();
    }
    writers.finish_group()?;
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

fn normalization_from_train(
    path: &Path,
    candidates: usize,
    names: &[String],
) -> Result<Vec<NormalizationSpec>, Box<dyn Error>> {
    if names.len() != S5_FEATURES {
        return Err("S5 normalization schema width drifted".into());
    }
    let bytes = fs::read(path)?;
    let expected_bytes = candidates
        .checked_mul(S5_FEATURES)
        .and_then(|values| values.checked_mul(2))
        .ok_or("S5 train byte count overflow")?;
    if bytes.len() != expected_bytes {
        return Err("S5 train tensor byte count drifted".into());
    }
    let mut output = Vec::with_capacity(S5_FEATURES);
    for (feature, name) in names.iter().enumerate() {
        let mut values = Vec::with_capacity(candidates);
        for candidate in 0..candidates {
            let offset = (candidate * S5_FEATURES + feature) * 2;
            values.push(i16::from_le_bytes([bytes[offset], bytes[offset + 1]]));
        }
        values.sort_unstable();
        let minimum = values[0];
        let maximum = *values.last().expect("nonempty S5 feature");
        let nonzero_count = values.iter().filter(|value| **value != 0).count();
        let mut absolute = values
            .iter()
            .map(|value| value.unsigned_abs())
            .collect::<Vec<_>>();
        absolute.sort_unstable();
        let p99_absolute = absolute[(absolute.len() - 1) * 99 / 100];
        let maximum_absolute = *absolute.last().expect("nonempty S5 feature");
        let transform = if maximum_absolute == 0 {
            "identity"
        } else if p99_absolute > 0 && maximum_absolute > p99_absolute.saturating_mul(16) {
            "signed_log1p_robust_divide"
        } else {
            "robust_divide"
        };
        output.push(NormalizationSpec {
            index: feature,
            name: name.clone(),
            count: candidates,
            nonzero_count,
            minimum,
            maximum,
            p99_absolute,
            maximum_absolute,
            transform,
            divisor: p99_absolute.max(1),
        });
    }
    Ok(output)
}

fn load_r3_binding(root: &Path) -> Result<R3Binding, Box<dyn Error>> {
    let manifest_path = root.join("cache.json");
    let manifest_bytes = fs::read(&manifest_path)?;
    let manifest: R3Manifest = serde_json::from_slice(&manifest_bytes)?;
    if manifest.cache_id != EXPECTED_R3_CACHE_ID
        || manifest.cache_schema != EXPECTED_R3_CACHE_SCHEMA
        || !manifest.complete_open_corpus
        || root.file_name().and_then(|name| name.to_str()) != Some(EXPECTED_R3_CACHE_ID)
    {
        return Err("ADR 0161 requires the accepted complete R3 cache".into());
    }
    let mut splits = BTreeMap::new();
    for split in ["train", "validation"] {
        let source = manifest
            .splits
            .get(split)
            .ok_or("R3 cache is missing train or validation")?;
        let expected = if split == "train" {
            (
                TRAIN_GROUPS,
                TRAIN_SOURCE_CANDIDATES,
                TRAIN_RETAINED_CANDIDATES,
            )
        } else {
            (
                VALIDATION_GROUPS,
                VALIDATION_SOURCE_CANDIDATES,
                VALIDATION_RETAINED_CANDIDATES,
            )
        };
        if (
            source.groups,
            source.source_candidates,
            source.retained_candidates,
        ) != expected
        {
            return Err(format!("{split} R3 cache counts drifted").into());
        }
        let group_ids =
            read_u64_tensor(root, required_r3_file(source, "group_ids")?, source.groups)?;
        let public_state_hashes = read_hash_tensor(
            root,
            required_r3_file(source, "public_state_hashes")?,
            source.groups,
        )?;
        let source_candidate_counts = read_u16_tensor(
            root,
            required_r3_file(source, "source_candidate_counts")?,
            source.groups,
        )?;
        let candidate_offsets = read_u64_tensor(
            root,
            required_r3_file(source, "candidate_offsets")?,
            source.groups + 1,
        )?;
        let source_candidate_indices = read_u16_tensor(
            root,
            required_r3_file(source, "source_candidate_indices")?,
            source.retained_candidates,
        )?;
        let action_hashes = read_hash_tensor(
            root,
            required_r3_file(source, "action_hashes")?,
            source.retained_candidates,
        )?;
        let candidate_identity_hashes = read_hash_tensor(
            root,
            required_r3_file(source, "candidate_identity_hashes")?,
            source.groups,
        )?;
        if candidate_offsets.first() != Some(&0)
            || candidate_offsets.last() != Some(&(source.retained_candidates as u64))
            || candidate_offsets.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(format!("{split} R3 candidate offsets drifted").into());
        }
        splits.insert(
            split.to_owned(),
            R3SplitBinding {
                dataset_id: source.dataset_id.clone(),
                dataset_manifest_blake3: source.dataset_manifest_blake3.clone(),
                groups: source.groups,
                retained_candidates: source.retained_candidates,
                group_ids,
                public_state_hashes,
                source_candidate_counts,
                candidate_offsets,
                source_candidate_indices,
                action_hashes,
                candidate_identity_hashes,
            },
        );
    }
    Ok(R3Binding {
        identity: R3CacheIdentity {
            cache_id: manifest.cache_id,
            manifest_blake3: blake3::hash(&manifest_bytes).to_hex().to_string(),
        },
        splits,
    })
}

fn required_r3_file<'a>(
    split: &'a R3SplitManifest,
    name: &str,
) -> Result<&'a R3FileSpec, Box<dyn Error>> {
    split
        .files
        .get(name)
        .ok_or_else(|| format!("R3 cache tensor {name} is absent").into())
}

fn read_u16_tensor(
    root: &Path,
    spec: &R3FileSpec,
    count: usize,
) -> Result<Vec<u16>, Box<dyn Error>> {
    let bytes = read_r3_tensor(root, spec, "<u2", &[count])?;
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn read_u64_tensor(
    root: &Path,
    spec: &R3FileSpec,
    count: usize,
) -> Result<Vec<u64>, Box<dyn Error>> {
    let bytes = read_r3_tensor(root, spec, "<u8", &[count])?;
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| {
            u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect())
}

fn read_hash_tensor(
    root: &Path,
    spec: &R3FileSpec,
    count: usize,
) -> Result<Vec<[u8; 32]>, Box<dyn Error>> {
    let bytes = read_r3_tensor(root, spec, "|u1", &[count, 32])?;
    Ok(bytes
        .chunks_exact(32)
        .map(|chunk| {
            let mut hash = [0u8; 32];
            hash.copy_from_slice(chunk);
            hash
        })
        .collect())
}

fn read_r3_tensor(
    root: &Path,
    spec: &R3FileSpec,
    dtype: &str,
    shape: &[usize],
) -> Result<Vec<u8>, Box<dyn Error>> {
    if spec.dtype != dtype || spec.shape != shape {
        return Err("R3 tensor dtype or shape drifted".into());
    }
    let path = root.join(&spec.file);
    if path.parent() != Some(root) || !path.is_file() {
        return Err("R3 tensor path escapes or is absent".into());
    }
    let bytes = fs::read(&path)?;
    if bytes.len() as u64 != spec.bytes || blake3::hash(&bytes).to_hex().as_str() != spec.blake3 {
        return Err("R3 tensor byte count or checksum drifted".into());
    }
    Ok(bytes)
}

fn tensor_contract() -> Value {
    json!({
        "r3_boundary": {
            "cache_id": EXPECTED_R3_CACHE_ID,
            "group_public_candidate_and_action_alignment": true,
            "candidate_cohorts_and_labels_unchanged": true,
        },
        "parent": {
            "d6_views_per_group": D6_TRANSFORMS,
            "classes": RELATIONAL_TOKEN_CLASS_COUNT,
            "value_width": RELATIONAL_TOKEN_VALUE_WIDTH,
            "relative_boards": BOARD_SLOTS,
            "rich_view_stored_once": true,
            "r5_minimal_view_is_loader_projection": true,
            "silent_truncation": false,
        },
        "candidate": {
            "s5_width": S5_FEATURES,
            "raw_dtype": "signed_i16",
            "normalization_fit": "open_train_retained_candidates_only",
            "validation_fit": false,
            "teacher_values_used": false,
        },
    })
}

fn split_name(split: DatasetSplit) -> &'static str {
    match split {
        DatasetSplit::Train => "train",
        DatasetSplit::Validation => "validation",
        DatasetSplit::Test | DatasetSplit::Final => unreachable!(),
    }
}

fn canonical_blake3(value: &impl Serialize) -> Result<String, Box<dyn Error>> {
    Ok(blake3::hash(&serde_json::to_vec(value)?)
        .to_hex()
        .to_string())
}

fn unix_millis() -> Result<u128, Box<dyn Error>> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis())
}
