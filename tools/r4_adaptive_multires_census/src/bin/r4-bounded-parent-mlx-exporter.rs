use std::{
    collections::BTreeMap,
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
use cascadia_game::{D6Transform, GameConfig, GameSeed, GameState};
use cascadia_provenance::{SourceProvenance, checksum_file, source_provenance};
use clap::Parser;
use r2_sparse_entity_census::SparsePublicState;
use r4_adaptive_multires_census::{
    AdaptiveMultiResolutionState, BOUNDED_PARENT_ADR_ID, BOUNDED_PARENT_ARMS,
    BOUNDED_PARENT_CACHE_SCHEMA, BOUNDED_PARENT_CACHE_SCHEMA_VERSION, BOUNDED_PARENT_EXPERIMENT_ID,
    BOUNDED_PARENT_PROTOCOL_ID, BoundedArm, BoundedFeatureView, NearFieldRadius,
    UNIVERSAL_PARENT_CLASS_COUNT, UNIVERSAL_PARENT_VALUE_WIDTH, bounded_parent_token_owner,
    bounded_token_universal_class,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const EXPECTED_R3_CACHE_ID: &str =
    "0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156";
const EXPECTED_R3_CACHE_SCHEMA: &str = "r3-action-edit-mlx-cache-v1";
const TRAIN_GROUPS: usize = 560;
const VALIDATION_GROUPS: usize = 240;
const TRAIN_SOURCE_CANDIDATES: usize = 2_135_111;
const VALIDATION_SOURCE_CANDIDATES: usize = 860_203;
const BOARD_SLOTS: usize = 4;
const D6_TRANSFORMS: usize = 12;
const BOUNDED_KIND_COUNT: usize = 5;

#[derive(Debug, Parser)]
#[command(about = "Export the exact ADR 0156 bounded-parent MLX sidecar")]
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

#[derive(Debug, Clone, Serialize)]
struct ArmChecks {
    bounded_views: usize,
    bounded_envelope_round_trips: usize,
    source_accounting_checks: usize,
    hard_token_max_checks: usize,
    universal_class_checks: usize,
    minimum_tokens: usize,
    maximum_tokens: usize,
    minimum_active_values: usize,
    maximum_active_values: usize,
}

impl Default for ArmChecks {
    fn default() -> Self {
        Self {
            bounded_views: 0,
            bounded_envelope_round_trips: 0,
            source_accounting_checks: 0,
            hard_token_max_checks: 0,
            universal_class_checks: 0,
            minimum_tokens: usize::MAX,
            maximum_tokens: 0,
            minimum_active_values: usize::MAX,
            maximum_active_values: 0,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
struct SplitChecks {
    groups_replayed: usize,
    r3_group_id_checks: usize,
    r3_public_state_hash_checks: usize,
    r3_candidate_count_checks: usize,
    r3_selected_index_checks: usize,
    r3_champion_index_checks: usize,
    position_record_checks: usize,
    game_public_state_hash_checks: usize,
    public_supply_checks: usize,
    exact_r2_constructions: usize,
    adaptive_state_constructions: usize,
    adaptive_packed_round_trips: usize,
    d6_transform_checks: usize,
    d6_inverse_checks: usize,
}

#[derive(Debug, Clone, Serialize)]
struct ArmManifest {
    arm: &'static str,
    universal_classes: [u8; BOUNDED_KIND_COUNT],
    hard_token_max: usize,
    views: usize,
    tokens: usize,
    active_values: usize,
    files: BTreeMap<String, FileSpec>,
    checks: ArmChecks,
}

#[derive(Debug, Clone, Serialize)]
struct SplitManifest {
    split: &'static str,
    dataset_id: String,
    dataset_manifest_blake3: String,
    groups: usize,
    source_candidates: usize,
    complete_open_split: bool,
    files: BTreeMap<String, FileSpec>,
    arms: BTreeMap<String, ArmManifest>,
    checks: SplitChecks,
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
    group_ids: Vec<u64>,
    public_state_hashes: Vec<[u8; 32]>,
    source_candidate_counts: Vec<u16>,
    selected_source_indices: Vec<u16>,
    champion_source_indices: Vec<u16>,
}

struct R3Binding {
    identity: R3CacheIdentity,
    splits: BTreeMap<String, R3SplitBinding>,
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

struct ArmWriters {
    arm: BoundedArm,
    view_offsets: HashedWriter,
    token_kinds: HashedWriter,
    token_seats: HashedWriter,
    value_offsets: HashedWriter,
    token_values: HashedWriter,
    token_counts: HashedWriter,
    board_class_counts: HashedWriter,
    view_hashes: HashedWriter,
    adaptive_state_hashes: HashedWriter,
    views: usize,
    tokens: usize,
    active_values: usize,
    checks: ArmChecks,
}

impl ArmWriters {
    fn create(root: &Path, split: &str, arm: BoundedArm) -> Result<Self, Box<dyn Error>> {
        let prefix = format!("{split}-{}", arm.id());
        let mut view_offsets = HashedWriter::create(root, format!("{prefix}-view-offsets.bin"))?;
        view_offsets.write_u64(0)?;
        let mut value_offsets = HashedWriter::create(root, format!("{prefix}-value-offsets.bin"))?;
        value_offsets.write_u64(0)?;
        Ok(Self {
            arm,
            view_offsets,
            token_kinds: HashedWriter::create(root, format!("{prefix}-token-kinds.bin"))?,
            token_seats: HashedWriter::create(root, format!("{prefix}-token-seats.bin"))?,
            value_offsets,
            token_values: HashedWriter::create(root, format!("{prefix}-token-values.bin"))?,
            token_counts: HashedWriter::create(root, format!("{prefix}-token-counts.bin"))?,
            board_class_counts: HashedWriter::create(
                root,
                format!("{prefix}-board-class-counts.bin"),
            )?,
            view_hashes: HashedWriter::create(root, format!("{prefix}-view-hashes.bin"))?,
            adaptive_state_hashes: HashedWriter::create(
                root,
                format!("{prefix}-adaptive-state-hashes.bin"),
            )?,
            views: 0,
            tokens: 0,
            active_values: 0,
            checks: ArmChecks::default(),
        })
    }

    fn write_view(
        &mut self,
        view: &BoundedFeatureView,
        adaptive_state_hash: &[u8; 32],
    ) -> Result<(), Box<dyn Error>> {
        if view.arm != self.arm {
            return Err("bounded parent writer received the wrong arm".into());
        }
        let canonical = view.canonical_bytes()?;
        if BoundedFeatureView::from_canonical_bytes(&canonical)? != *view {
            return Err("bounded parent envelope did not round trip".into());
        }
        self.checks.bounded_envelope_round_trips += 1;
        if view.tokens.len() > self.arm.hard_token_max() {
            return Err(format!(
                "{} emitted {} tokens above hard maximum {}",
                self.arm.id(),
                view.tokens.len(),
                self.arm.hard_token_max()
            )
            .into());
        }
        self.checks.hard_token_max_checks += 1;
        if view.accounting.exact_wildlife_buckets != 0
            || view.accounting.exact_wildlife_mass != 0
            || view.accounting.exact_frontier_buckets != 0
            || view.accounting.exact_frontier_mass != 0
            || view.accounting.source_wildlife_buckets
                != view.accounting.summarized_wildlife_buckets
            || view.accounting.source_wildlife_mass != view.accounting.summarized_wildlife_mass
            || view.accounting.source_frontier_buckets
                != view.accounting.summarized_frontier_buckets
            || view.accounting.source_frontier_mass != view.accounting.summarized_frontier_mass
        {
            return Err(format!("{} source accounting is not exact", self.arm.id()).into());
        }
        self.checks.source_accounting_checks += 1;

        let mut board_class_counts = [[0u16; BOUNDED_KIND_COUNT]; BOARD_SLOTS];
        let active_values = view
            .tokens
            .iter()
            .map(|token| token.values.len())
            .sum::<usize>();
        self.token_counts
            .write_u16(u16::try_from(view.tokens.len())?)?;
        for token in &view.tokens {
            let owner = bounded_parent_token_owner(view, token)?;
            let class = bounded_token_universal_class(token.kind)?;
            if class as usize > UNIVERSAL_PARENT_CLASS_COUNT {
                return Err("bounded universal class exceeds the registered schema".into());
            }
            self.checks.universal_class_checks += 1;
            let kind_index = usize::from(token.kind.code() - 1);
            board_class_counts[usize::from(owner)][kind_index] = board_class_counts
                [usize::from(owner)][kind_index]
                .checked_add(1)
                .ok_or("bounded board-class count overflow")?;
            self.token_kinds.write_u8(token.kind.code())?;
            self.token_seats.write_u8(owner)?;
            for value in &token.values {
                self.token_values.write_i16(*value)?;
            }
            self.tokens += 1;
            self.active_values += token.values.len();
            self.value_offsets
                .write_u64(u64::try_from(self.active_values)?)?;
        }
        for board in board_class_counts {
            for count in board {
                self.board_class_counts.write_u16(count)?;
            }
        }
        self.views += 1;
        self.view_offsets.write_u64(u64::try_from(self.tokens)?)?;
        self.view_hashes
            .write_bytes(blake3::hash(&canonical).as_bytes())?;
        self.adaptive_state_hashes
            .write_bytes(adaptive_state_hash)?;
        self.checks.bounded_views += 1;
        self.checks.minimum_tokens = self.checks.minimum_tokens.min(view.tokens.len());
        self.checks.maximum_tokens = self.checks.maximum_tokens.max(view.tokens.len());
        self.checks.minimum_active_values = self.checks.minimum_active_values.min(active_values);
        self.checks.maximum_active_values = self.checks.maximum_active_values.max(active_values);
        Ok(())
    }

    fn finish(mut self, groups: usize) -> Result<ArmManifest, Box<dyn Error>> {
        if self.checks.minimum_tokens == usize::MAX {
            self.checks.minimum_tokens = 0;
        }
        if self.checks.minimum_active_values == usize::MAX {
            self.checks.minimum_active_values = 0;
        }
        let views = groups * D6_TRANSFORMS;
        if self.views != views {
            return Err(format!(
                "{} wrote {} views; expected {views}",
                self.arm.id(),
                self.views
            )
            .into());
        }
        let mut files = BTreeMap::new();
        files.insert(
            "view_offsets".to_owned(),
            self.view_offsets.finish("<u8", vec![views + 1])?,
        );
        files.insert(
            "token_kinds".to_owned(),
            self.token_kinds.finish("|u1", vec![self.tokens])?,
        );
        files.insert(
            "token_seats".to_owned(),
            self.token_seats.finish("|u1", vec![self.tokens])?,
        );
        files.insert(
            "value_offsets".to_owned(),
            self.value_offsets.finish("<u8", vec![self.tokens + 1])?,
        );
        files.insert(
            "token_values".to_owned(),
            self.token_values.finish("<i2", vec![self.active_values])?,
        );
        files.insert(
            "token_counts".to_owned(),
            self.token_counts
                .finish("<u2", vec![groups, D6_TRANSFORMS])?,
        );
        files.insert(
            "board_class_counts".to_owned(),
            self.board_class_counts.finish(
                "<u2",
                vec![groups, D6_TRANSFORMS, BOARD_SLOTS, BOUNDED_KIND_COUNT],
            )?,
        );
        files.insert(
            "view_hashes".to_owned(),
            self.view_hashes
                .finish("|u1", vec![groups, D6_TRANSFORMS, 32])?,
        );
        files.insert(
            "adaptive_state_hashes".to_owned(),
            self.adaptive_state_hashes
                .finish("|u1", vec![groups, D6_TRANSFORMS, 32])?,
        );
        Ok(ArmManifest {
            arm: self.arm.id(),
            universal_classes: [5, 6, 7, 8, 9],
            hard_token_max: self.arm.hard_token_max(),
            views,
            tokens: self.tokens,
            active_values: self.active_values,
            files,
            checks: self.checks,
        })
    }
}

struct SplitWriters {
    group_ids: HashedWriter,
    public_state_hashes: HashedWriter,
    arms: BTreeMap<BoundedArm, ArmWriters>,
}

type FinishedSplitWriters = (BTreeMap<String, FileSpec>, BTreeMap<String, ArmManifest>);

impl SplitWriters {
    fn create(root: &Path, split: &str) -> Result<Self, Box<dyn Error>> {
        let arms = BOUNDED_PARENT_ARMS
            .into_iter()
            .map(|arm| Ok((arm, ArmWriters::create(root, split, arm)?)))
            .collect::<Result<BTreeMap<_, _>, Box<dyn Error>>>()?;
        Ok(Self {
            group_ids: HashedWriter::create(root, format!("{split}-group-ids.bin"))?,
            public_state_hashes: HashedWriter::create(
                root,
                format!("{split}-public-state-hashes.bin"),
            )?,
            arms,
        })
    }

    fn write_group_identity(&mut self, group: &GradedOracleGroup) -> Result<(), Box<dyn Error>> {
        self.group_ids.write_u64(group.group_id)?;
        self.public_state_hashes
            .write_bytes(&group.public_state_hash)?;
        Ok(())
    }

    fn finish(self, groups: usize) -> Result<FinishedSplitWriters, Box<dyn Error>> {
        let files = BTreeMap::from([
            (
                "group_ids".to_owned(),
                self.group_ids.finish("<u8", vec![groups])?,
            ),
            (
                "public_state_hashes".to_owned(),
                self.public_state_hashes.finish("|u1", vec![groups, 32])?,
            ),
        ]);
        let arms = self
            .arms
            .into_iter()
            .map(|(arm, writer)| Ok((arm.id().to_owned(), writer.finish(groups)?)))
            .collect::<Result<BTreeMap<_, _>, Box<dyn Error>>>()?;
        Ok((files, arms))
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if args.max_groups_per_split == Some(0) {
        return Err("--max-groups-per-split must be positive".into());
    }
    fs::create_dir_all(&args.output_root)?;
    let r3 = load_r3_binding(&args.r3_cache)?;
    let temporary = args.output_root.join(format!(
        ".tmp-r4-bounded-parent-mlx-{}-{}",
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
        let exporter = ExporterIdentity {
            executable_blake3: checksum_file(&env::current_exe()?)?,
            source: source_provenance()?,
        };
        let splits = BTreeMap::from([
            ("train".to_owned(), train),
            ("validation".to_owned(), validation),
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
        };
        let tensor_contract = tensor_contract();
        let scientific_identity = json!({
            "schema_version": BOUNDED_PARENT_CACHE_SCHEMA_VERSION,
            "cache_schema": BOUNDED_PARENT_CACHE_SCHEMA,
            "experiment_id": BOUNDED_PARENT_EXPERIMENT_ID,
            "protocol_id": BOUNDED_PARENT_PROTOCOL_ID,
            "adr": BOUNDED_PARENT_ADR_ID,
            "complete_open_corpus": complete_open_corpus,
            "exporter": exporter.clone(),
            "r3_cache": r3.identity.clone(),
            "tensor_contract": tensor_contract.clone(),
            "hidden_information": hidden_information.clone(),
            "splits": splits.clone(),
        });
        let cache_id = canonical_blake3(&scientific_identity)?;
        let manifest = CacheManifest {
            schema_version: BOUNDED_PARENT_CACHE_SCHEMA_VERSION,
            cache_schema: BOUNDED_PARENT_CACHE_SCHEMA,
            experiment_id: BOUNDED_PARENT_EXPERIMENT_ID,
            protocol_id: BOUNDED_PARENT_PROTOCOL_ID,
            adr: BOUNDED_PARENT_ADR_ID,
            cache_id: cache_id.clone(),
            complete_open_corpus,
            exporter,
            r3_cache: r3.identity.clone(),
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
                "experiment_id": BOUNDED_PARENT_EXPERIMENT_ID,
                "protocol_id": BOUNDED_PARENT_PROTOCOL_ID,
                "adr": BOUNDED_PARENT_ADR_ID,
                "cache_id": manifest.cache_id,
                "cache_root": cache_root,
                "r3_cache_id": manifest.r3_cache.cache_id,
                "complete_open_corpus": manifest.complete_open_corpus,
                "train_groups": manifest.splits["train"].groups,
                "validation_groups": manifest.splits["validation"].groups,
                "arms": BOUNDED_PARENT_ARMS.map(|arm| arm.id()),
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
) -> Result<SplitManifest, Box<dyn Error>> {
    if matches!(expected_split, DatasetSplit::Test | DatasetSplit::Final) {
        return Err("ADR 0156 exporter prohibits test and final splits".into());
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
        return Err(format!("{split_name} dataset identity disagrees with the R3 cache").into());
    }

    let mut writers = SplitWriters::create(output, split_name)?;
    let mut checks = SplitChecks::default();
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
    let expected = match expected_split {
        DatasetSplit::Train => (TRAIN_GROUPS, TRAIN_SOURCE_CANDIDATES),
        DatasetSplit::Validation => (VALIDATION_GROUPS, VALIDATION_SOURCE_CANDIDATES),
        DatasetSplit::Test | DatasetSplit::Final => unreachable!(),
    };
    let complete_open_split = maximum_groups.is_none() && (groups, source_candidates) == expected;
    if maximum_groups.is_none() && !complete_open_split {
        return Err(format!(
            "{split_name} export covered {groups}/{}, {source_candidates}/{}",
            expected.0, expected.1
        )
        .into());
    }
    let (files, arms) = writers.finish(groups)?;
    Ok(SplitManifest {
        split: split_name,
        dataset_id: manifest.dataset_id,
        dataset_manifest_blake3: blake3::hash(&manifest_bytes).to_hex().to_string(),
        groups,
        source_candidates,
        complete_open_split,
        files,
        arms,
        checks,
    })
}

fn process_group(
    game: &mut GameState,
    group: &GradedOracleGroup,
    row: usize,
    r3: &R3SplitBinding,
    writers: &mut SplitWriters,
    checks: &mut SplitChecks,
) -> Result<(), Box<dyn Error>> {
    if row >= r3.groups {
        return Err(format!("source contains group row {row} beyond the R3 cache").into());
    }
    if group.group_id != r3.group_ids[row] {
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
    if group.selected_index != r3.selected_source_indices[row] {
        return Err(format!("R3 selected source index drifted at row {row}").into());
    }
    checks.r3_selected_index_checks += 1;
    if group.champion_index != r3.champion_source_indices[row] {
        return Err(format!("R3 champion source index drifted at row {row}").into());
    }
    checks.r3_champion_index_checks += 1;
    if game.completed_turns() != group.completed_turns
        || game.current_player() != usize::from(group.current_player)
        || PositionRecord::observe(game, group.raw_seed).to_bytes() != group.position.to_bytes()
    {
        return Err(format!("graded replay drifted at group {}", group.group_id).into());
    }
    checks.position_record_checks += 1;
    if *game.public_state().canonical_hash().as_bytes() != group.public_state_hash {
        return Err(format!("game public-state hash drifted at group {}", group.group_id).into());
    }
    checks.game_public_state_hash_checks += 1;
    if game.public_supply() != group.public_supply {
        return Err(format!("public supply drifted at group {}", group.group_id).into());
    }
    checks.public_supply_checks += 1;

    let sparse = SparsePublicState::from_position_record(&group.position, None)?;
    checks.exact_r2_constructions += 1;
    let state = AdaptiveMultiResolutionState::from_sparse_state(&sparse, NearFieldRadius::Radius4)?;
    checks.adaptive_state_constructions += 1;
    let packed = state.to_packed_bytes()?;
    if AdaptiveMultiResolutionState::from_packed_bytes(&packed)? != state {
        return Err(format!("CSR4AM1 round trip drifted at group {}", group.group_id).into());
    }
    checks.adaptive_packed_round_trips += 1;
    writers.write_group_identity(group)?;

    for transform in D6Transform::ALL {
        let transformed = state.transformed(transform)?;
        checks.d6_transform_checks += 1;
        if transformed.transformed(transform.inverse())? != state {
            return Err(format!(
                "D6 inverse drifted at group {} transform {}",
                group.group_id,
                transform.id()
            )
            .into());
        }
        checks.d6_inverse_checks += 1;
        let transformed_packed = transformed.to_packed_bytes()?;
        if AdaptiveMultiResolutionState::from_packed_bytes(&transformed_packed)? != transformed {
            return Err(format!(
                "transformed CSR4AM1 round trip drifted at group {} transform {}",
                group.group_id,
                transform.id()
            )
            .into());
        }
        checks.adaptive_packed_round_trips += 1;
        let adaptive_hash = *blake3::hash(&transformed_packed).as_bytes();
        for arm in BOUNDED_PARENT_ARMS {
            let view = BoundedFeatureView::from_state(&transformed, arm)?;
            writers
                .arms
                .get_mut(&arm)
                .ok_or("bounded parent writer is absent")?
                .write_view(&view, &adaptive_hash)?;
        }
    }
    checks.groups_replayed += 1;
    Ok(())
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
        return Err("ADR 0156 requires the accepted complete ADR 0150 R3 cache".into());
    }
    let mut splits = BTreeMap::new();
    for split in ["train", "validation"] {
        let source = manifest
            .splits
            .get(split)
            .ok_or("R3 cache is missing train or validation")?;
        let expected = if split == "train" {
            (TRAIN_GROUPS, TRAIN_SOURCE_CANDIDATES)
        } else {
            (VALIDATION_GROUPS, VALIDATION_SOURCE_CANDIDATES)
        };
        if (source.groups, source.source_candidates) != expected {
            return Err(format!("{split} R3 cache counts drifted").into());
        }
        splits.insert(
            split.to_owned(),
            R3SplitBinding {
                dataset_id: source.dataset_id.clone(),
                dataset_manifest_blake3: source.dataset_manifest_blake3.clone(),
                groups: source.groups,
                group_ids: read_u64_tensor(
                    root,
                    required_r3_file(source, "group_ids")?,
                    source.groups,
                )?,
                public_state_hashes: read_hash_tensor(
                    root,
                    required_r3_file(source, "public_state_hashes")?,
                    source.groups,
                )?,
                source_candidate_counts: read_u16_tensor(
                    root,
                    required_r3_file(source, "source_candidate_counts")?,
                    source.groups,
                )?,
                selected_source_indices: read_u16_tensor(
                    root,
                    required_r3_file(source, "selected_source_indices")?,
                    source.groups,
                )?,
                champion_source_indices: read_u16_tensor(
                    root,
                    required_r3_file(source, "champion_source_indices")?,
                    source.groups,
                )?,
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

fn read_u64_tensor(
    root: &Path,
    spec: &R3FileSpec,
    count: usize,
) -> Result<Vec<u64>, Box<dyn Error>> {
    let bytes = read_r3_tensor(root, spec, "<u8", &[count])?;
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("eight-byte chunk")))
        .collect())
}

fn read_u16_tensor(
    root: &Path,
    spec: &R3FileSpec,
    count: usize,
) -> Result<Vec<u16>, Box<dyn Error>> {
    let bytes = read_r3_tensor(root, spec, "<u2", &[count])?;
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes(chunk.try_into().expect("two-byte chunk")))
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
        .map(|chunk| chunk.try_into().expect("32-byte chunk"))
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
        return Err(format!("R3 tensor checksum or byte count drifted: {}", spec.file).into());
    }
    Ok(bytes)
}

fn tensor_contract() -> Value {
    json!({
        "r3_parent_boundary": {
            "cache_id": EXPECTED_R3_CACHE_ID,
            "group_and_public_hash_alignment": true,
            "candidate_afterstate_stream_unchanged": true,
        },
        "bounded_parent": {
            "arms": BOUNDED_PARENT_ARMS.map(|arm| arm.id()),
            "radius": NearFieldRadius::Radius4.id(),
            "d6_views_per_group": D6_TRANSFORMS,
            "ragged_active_i16_values": true,
            "universal_parent_class_count": UNIVERSAL_PARENT_CLASS_COUNT,
            "universal_parent_value_width": UNIVERSAL_PARENT_VALUE_WIDTH,
            "board_slots": BOARD_SLOTS,
            "bounded_kind_count": BOUNDED_KIND_COUNT,
            "silent_truncation": false,
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
