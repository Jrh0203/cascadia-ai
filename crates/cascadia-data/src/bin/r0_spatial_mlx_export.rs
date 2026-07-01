use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use blake3::Hasher;
use cascadia_data::{
    BOARD_SLOTS, DatasetManifest, DatasetSplit, MAX_BOARD_TILES, PositionRecord,
    PositionShardReader, SpatialArm, SpatialBoardRepresentation, SpatialPositionRepresentation,
    TARGET_DIM, validate_dataset,
};
use cascadia_game::D6Transform;
use cascadia_provenance::{checksum_file, source_provenance};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const CACHE_SCHEMA_VERSION: u16 = 1;
const CACHE_SCHEMA: &str = "r0-spatial-mlx-cache-v1";
const EXPERIMENT_ID: &str = "r0-spatial-mlx-tournament-v1";
const CORPUS_LOCK_SCHEMA_VERSION: u16 = 1;
const CORPUS_LOCK_CONTRACT: &str = "r0-frozen-60000-position-corpus-v1";
const CORPUS_DIGEST_PREFIX: &[u8] = b"R0MLXCORPUS1\0";
const MAX_ENTITIES_PER_BOARD: usize = MAX_BOARD_TILES;
const D6_TRANSFORMS: usize = 12;
const TOKEN_FIELDS: usize = 11;
const MARKET_FEATURES: usize = 31;
const GLOBAL_FEATURES: usize = 96;
const SLOT_SENTINEL: u16 = u16::MAX;
const NONE: u8 = u8::MAX;

const HELP: &str = concat!(
    "Usage: r0_spatial_mlx_export \\\n",
    "  --corpus-lock PATH \\\n",
    "  --dataset-root PATH [--dataset-root PATH ...] \\\n",
    "  --arm ID --output-root PATH [--receipt PATH]\n\n",
    "The corpus lock must name exactly the frozen eight-part 50,000/10,000 R0 corpus.\n",
    "The output is installed under OUTPUT_ROOT/<content-hash>/cache.json."
);

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    corpus_lock: PathBuf,
    dataset_roots: Vec<PathBuf>,
    arm: SpatialArm,
    output_root: PathBuf,
    receipt: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CorpusLock {
    schema_version: u16,
    contract_id: String,
    lock_id: String,
    identity: CorpusLockIdentity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CorpusLockIdentity {
    feature_schema: String,
    target_schema: String,
    total_records: usize,
    train_records: usize,
    validation_records: usize,
    source_v2_blake3: String,
    corpus_blake3: String,
    datasets: Vec<CorpusDatasetIdentity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CorpusDatasetIdentity {
    order: usize,
    split: String,
    part_index: usize,
    root_name: String,
    dataset_id: String,
    first_game_index: u64,
    completed_games: usize,
    total_records: usize,
    manifest_blake3: String,
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
    source_entity_rows: usize,
    exported_active_token_rows: usize,
    exported_padding_token_rows: usize,
    identity_overflow_entity_rows: usize,
    identity_positions_with_overflow: usize,
    d6_overflow_entity_rows: usize,
    d6_positions_with_overflow: usize,
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
    arm: &'static str,
    scientific_identity: Value,
    tensor_contract: Value,
    corpus: Value,
    semantic_integrity: Value,
    overflow_integrity: Value,
    splits: BTreeMap<String, SplitManifest>,
    exporter: Value,
}

#[derive(Debug, Serialize)]
struct ExportReceipt {
    schema_version: u16,
    experiment_id: &'static str,
    arm: &'static str,
    cache_id: String,
    cache_root: String,
    cache_manifest: String,
    cache_manifest_blake3: String,
}

struct HashedWriter {
    path: PathBuf,
    writer: BufWriter<File>,
    hasher: Hasher,
    bytes: u64,
}

impl HashedWriter {
    fn create(path: PathBuf) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            writer: BufWriter::new(File::create(&path)?),
            path,
            hasher: Hasher::new(),
            bytes: 0,
        })
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
        self.writer.write_all(bytes)?;
        self.hasher.update(bytes);
        self.bytes = self
            .bytes
            .checked_add(bytes.len() as u64)
            .ok_or("exported cache exceeds u64 byte accounting")?;
        Ok(())
    }

    fn finish(mut self, dtype: &str, shape: Vec<usize>) -> Result<FileIdentity, Box<dyn Error>> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        let expected_bytes = shape
            .iter()
            .try_fold(dtype_width(dtype), |value, dimension| {
                value.checked_mul(*dimension)
            })
            .ok_or("tensor byte count overflowed usize")? as u64;
        if expected_bytes != self.bytes {
            return Err(format!(
                "{} has {} bytes; shape and dtype require {}",
                self.path.display(),
                self.bytes,
                expected_bytes
            )
            .into());
        }
        Ok(FileIdentity {
            file: self
                .path
                .file_name()
                .and_then(|value| value.to_str())
                .ok_or("cache tensor file name must be UTF-8")?
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
    token_slots: HashedWriter,
    token_features: HashedWriter,
    market_features: HashedWriter,
    market_mask: HashedWriter,
    global_features: HashedWriter,
    targets: HashedWriter,
    game_index: HashedWriter,
    turn: HashedWriter,
    board_counts: HashedWriter,
    integrity: SplitIntegrity,
}

impl SplitWriters {
    fn create(root: &Path, split: &'static str) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            split,
            token_slots: HashedWriter::create(root.join(format!("{split}-token-slots.u16")))?,
            token_features: HashedWriter::create(root.join(format!("{split}-token-features.i8")))?,
            market_features: HashedWriter::create(
                root.join(format!("{split}-market-features.f32")),
            )?,
            market_mask: HashedWriter::create(root.join(format!("{split}-market-mask.u8")))?,
            global_features: HashedWriter::create(
                root.join(format!("{split}-global-features.f32")),
            )?,
            targets: HashedWriter::create(root.join(format!("{split}-targets.f32")))?,
            game_index: HashedWriter::create(root.join(format!("{split}-game-index.u64")))?,
            turn: HashedWriter::create(root.join(format!("{split}-turn.u8")))?,
            board_counts: HashedWriter::create(root.join(format!("{split}-board-counts.u8")))?,
            integrity: SplitIntegrity::default(),
        })
    }

    fn write_record(
        &mut self,
        record: &PositionRecord,
        representation: &SpatialPositionRepresentation,
        arm: SpatialArm,
        d6_hasher: &mut Hasher,
    ) -> Result<(), Box<dyn Error>> {
        let token_capacity = spatial_token_capacity(arm);
        let mut slots = vec![SLOT_SENTINEL; D6_TRANSFORMS * BOARD_SLOTS * MAX_ENTITIES_PER_BOARD];
        let mut features =
            vec![0i8; D6_TRANSFORMS * BOARD_SLOTS * MAX_ENTITIES_PER_BOARD * TOKEN_FIELDS];
        let source_entities: usize = record
            .board_counts
            .iter()
            .map(|value| *value as usize)
            .sum();
        let identity_accounting = representation.accounting();
        self.integrity.records += 1;
        self.integrity.source_entity_rows += source_entities;
        self.integrity.identity_overflow_entity_rows += identity_accounting.overflow_entity_rows;
        self.integrity.identity_positions_with_overflow +=
            usize::from(identity_accounting.overflow_entity_rows > 0);

        for transform in D6Transform::ALL {
            let transformed = representation.transformed(transform)?;
            let transformed_record = transformed.to_position_record()?;
            d6_hasher.update(&transformed_record.to_bytes());
            let recovered = transformed.transformed(transform.inverse())?;
            if recovered.to_position_record()? != *record {
                return Err(format!(
                    "{} failed inverse D6 round trip for transform {}",
                    arm.id(),
                    transform.id()
                )
                .into());
            }
            let accounting = transformed.accounting();
            self.integrity.d6_overflow_entity_rows += accounting.overflow_entity_rows;
            self.integrity.d6_positions_with_overflow +=
                usize::from(accounting.overflow_entity_rows > 0);
            for (board_index, board) in transformed.boards.iter().enumerate() {
                write_board_tokens(
                    &mut slots,
                    &mut features,
                    transform.id() as usize,
                    board_index,
                    token_capacity,
                    board,
                    arm,
                )?;
            }
        }

        let active = slots.iter().filter(|slot| **slot != SLOT_SENTINEL).count();
        let expected_active = source_entities
            .checked_mul(D6_TRANSFORMS)
            .ok_or("active token count overflowed usize")?;
        if active != expected_active {
            return Err(format!(
                "{} exported {active} active tokens; expected {expected_active}",
                arm.id()
            )
            .into());
        }
        validate_sparse_record(&slots, &features, token_capacity, arm)?;
        self.integrity.exported_active_token_rows += active;
        self.integrity.exported_padding_token_rows += slots.len() - active;

        for slot in slots {
            self.token_slots.write_bytes(&slot.to_le_bytes())?;
        }
        let feature_bytes = features
            .into_iter()
            .map(|value| value as u8)
            .collect::<Vec<_>>();
        self.token_features.write_bytes(&feature_bytes)?;

        let (market, market_mask) = market_features(record)?;
        for value in market {
            self.market_features.write_bytes(&value.to_le_bytes())?;
        }
        self.market_mask.write_bytes(&market_mask)?;

        for value in global_features(record)? {
            self.global_features.write_bytes(&value.to_le_bytes())?;
        }
        for target in record.targets {
            self.targets.write_bytes(&(target as f32).to_le_bytes())?;
        }
        self.game_index
            .write_bytes(&record.game_index.to_le_bytes())?;
        self.turn.write_bytes(&[record.turn])?;
        self.board_counts.write_bytes(&record.board_counts)?;
        Ok(())
    }

    fn finish(self, arm: SpatialArm) -> Result<SplitManifest, Box<dyn Error>> {
        let records = self.integrity.records;
        let token_capacity = spatial_token_capacity(arm);
        let mut files = BTreeMap::new();
        insert_file(
            &mut files,
            "token_slots",
            self.token_slots.finish(
                "<u2",
                vec![records, D6_TRANSFORMS, BOARD_SLOTS, MAX_ENTITIES_PER_BOARD],
            )?,
        );
        insert_file(
            &mut files,
            "token_features",
            self.token_features.finish(
                "|i1",
                vec![
                    records,
                    D6_TRANSFORMS,
                    BOARD_SLOTS,
                    MAX_ENTITIES_PER_BOARD,
                    TOKEN_FIELDS,
                ],
            )?,
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
        insert_file(
            &mut files,
            "board_counts",
            self.board_counts
                .finish("|u1", vec![records, BOARD_SLOTS])?,
        );
        let expected_slots = records
            .checked_mul(D6_TRANSFORMS * BOARD_SLOTS * MAX_ENTITIES_PER_BOARD)
            .ok_or("split slot count overflowed usize")?;
        if self.integrity.exported_active_token_rows + self.integrity.exported_padding_token_rows
            != expected_slots
        {
            return Err(format!("{} split token accounting drifted", self.split).into());
        }
        if self.integrity.exported_active_token_rows
            != self.integrity.source_entity_rows * D6_TRANSFORMS
        {
            return Err(format!("{} split entity accounting drifted", self.split).into());
        }
        if token_capacity < MAX_ENTITIES_PER_BOARD {
            return Err("spatial token capacity cannot be below the entity limit".into());
        }
        Ok(SplitManifest {
            records,
            files,
            integrity: self.integrity,
        })
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args(env::args_os().skip(1))?;
    let lock = read_corpus_lock(&args.corpus_lock)?;
    let datasets = validate_corpus(&args.dataset_roots, &lock)?;
    let source = source_provenance()?;
    let executable = env::current_exe()?;
    let executable_blake3 = checksum_file(&executable)?;

    fs::create_dir_all(&args.output_root)?;
    let temporary = args
        .output_root
        .join(format!(".r0-mlx-export-{}.tmp", std::process::id()));
    if temporary.exists() {
        fs::remove_dir_all(&temporary)?;
    }
    fs::create_dir(&temporary)?;

    let export = export_cache(&temporary, &datasets, &lock, args.arm);
    let (splits, semantic, overflow, mut scientific_identity) = match export {
        Ok(value) => value,
        Err(error) => {
            fs::remove_dir_all(&temporary).ok();
            return Err(error);
        }
    };
    let identity = scientific_identity
        .as_object_mut()
        .ok_or("cache scientific identity must be a JSON object")?;
    identity.insert(
        "exporter_executable_blake3".to_owned(),
        Value::String(executable_blake3.clone()),
    );
    identity.insert(
        "exporter_source_v2_blake3".to_owned(),
        Value::String(source.v2_source_blake3.clone()),
    );
    let cache_id = blake3::hash(&canonical_json_bytes(&scientific_identity))
        .to_hex()
        .to_string();
    let tensor_contract = tensor_contract(args.arm);
    let corpus = corpus_manifest_value(&lock);
    let manifest = CacheManifest {
        schema_version: CACHE_SCHEMA_VERSION,
        cache_schema: CACHE_SCHEMA,
        experiment_id: EXPERIMENT_ID,
        cache_id: cache_id.clone(),
        arm: args.arm.id(),
        scientific_identity,
        tensor_contract,
        corpus,
        semantic_integrity: semantic,
        overflow_integrity: overflow,
        splits,
        exporter: json!({
            "source_provenance": source,
            "executable_blake3": executable_blake3,
            "package_version": env!("CARGO_PKG_VERSION"),
        }),
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    fs::write(
        temporary.join("cache.json"),
        [&manifest_bytes[..], b"\n"].concat(),
    )?;

    let final_root = args.output_root.join(&cache_id);
    if final_root.exists() {
        let existing = fs::read(final_root.join("cache.json"))?;
        let generated = fs::read(temporary.join("cache.json"))?;
        if existing != generated {
            fs::remove_dir_all(&temporary).ok();
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

    let manifest_path = final_root.join("cache.json");
    let receipt = ExportReceipt {
        schema_version: 1,
        experiment_id: EXPERIMENT_ID,
        arm: args.arm.id(),
        cache_id,
        cache_root: final_root.display().to_string(),
        cache_manifest: manifest_path.display().to_string(),
        cache_manifest_blake3: checksum_file(&manifest_path)?,
    };
    let encoded = serde_json::to_string_pretty(&receipt)? + "\n";
    if let Some(path) = &args.receipt {
        write_atomically(path, encoded.as_bytes())?;
    }
    print!("{encoded}");
    Ok(())
}

type ExportResult = (BTreeMap<String, SplitManifest>, Value, Value, Value);

fn export_cache(
    root: &Path,
    datasets: &[ValidatedDataset],
    lock: &CorpusLock,
    arm: SpatialArm,
) -> Result<ExportResult, Box<dyn Error>> {
    let mut train = SplitWriters::create(root, "train")?;
    let mut validation = SplitWriters::create(root, "validation")?;
    let mut source_hasher = Hasher::new();
    let mut d6_hasher = Hasher::new();
    let mut target_hasher = Hasher::new();
    let mut packed_round_trips = 0usize;

    for dataset in datasets {
        for shard in &dataset.manifest.shards {
            for record in PositionShardReader::open(&dataset.root, shard)? {
                let record = record?;
                source_hasher.update(&record.to_bytes());
                for target in record.targets {
                    target_hasher.update(&target.to_le_bytes());
                }
                let representation = SpatialPositionRepresentation::from_record(&record, arm)?;
                if representation.to_position_record()? != record {
                    return Err(
                        format!("{} failed the in-memory semantic round trip", arm.id()).into(),
                    );
                }
                let packed = representation.to_packed_bytes()?;
                let decoded = SpatialPositionRepresentation::from_packed_bytes(&packed)?;
                if decoded != representation || decoded.to_position_record()? != record {
                    return Err(
                        format!("{} failed the packed semantic round trip", arm.id()).into(),
                    );
                }
                packed_round_trips += 1;
                match dataset.manifest.split {
                    DatasetSplit::Train => {
                        train.write_record(&record, &representation, arm, &mut d6_hasher)?;
                    }
                    DatasetSplit::Validation => {
                        validation.write_record(&record, &representation, arm, &mut d6_hasher)?;
                    }
                    DatasetSplit::Test | DatasetSplit::Final => {
                        return Err("R0 MLX export prohibits test and final data".into());
                    }
                }
            }
        }
    }

    let source_semantic_blake3 = source_hasher.finalize().to_hex().to_string();
    let d6_semantic_blake3 = d6_hasher.finalize().to_hex().to_string();
    let target_blake3 = target_hasher.finalize().to_hex().to_string();
    let train_manifest = train.finish(arm)?;
    let validation_manifest = validation.finish(arm)?;
    if train_manifest.records != lock.identity.train_records
        || validation_manifest.records != lock.identity.validation_records
        || train_manifest.records + validation_manifest.records != lock.identity.total_records
        || packed_round_trips != lock.identity.total_records
    {
        return Err("exported row totals do not match the frozen corpus lock".into());
    }

    let mut splits = BTreeMap::new();
    splits.insert("train".to_owned(), train_manifest);
    splits.insert("validation".to_owned(), validation_manifest);
    let semantic = json!({
        "identity_round_trip_verified": true,
        "packed_round_trip_verified": true,
        "packed_round_trip_records": packed_round_trips,
        "source_semantic_blake3": source_semantic_blake3,
        "d6_semantic_blake3": d6_semantic_blake3,
        "target_blake3": target_blake3,
        "d6_inverse_round_trip_verified": true,
        "d6_transform_count": D6_TRANSFORMS,
    });
    let overflow = json!({
        "exact_entities_retained": true,
        "identity": {
            "train_overflow_entity_rows": splits["train"].integrity.identity_overflow_entity_rows,
            "train_positions_with_overflow": splits["train"].integrity.identity_positions_with_overflow,
            "validation_overflow_entity_rows": splits["validation"].integrity.identity_overflow_entity_rows,
            "validation_positions_with_overflow": splits["validation"].integrity.identity_positions_with_overflow,
        },
        "all_d6_transforms": {
            "train_overflow_entity_rows": splits["train"].integrity.d6_overflow_entity_rows,
            "train_positions_with_overflow": splits["train"].integrity.d6_positions_with_overflow,
            "validation_overflow_entity_rows": splits["validation"].integrity.d6_overflow_entity_rows,
            "validation_positions_with_overflow": splits["validation"].integrity.d6_positions_with_overflow,
        },
    });
    let scientific_identity = json!({
        "arm": arm.id(),
        "cache_schema": CACHE_SCHEMA,
        "corpus_blake3": lock.identity.corpus_blake3,
        "corpus_lock_id": lock.lock_id,
        "d6_semantic_blake3": semantic["d6_semantic_blake3"],
        "d6_transform_ids": D6Transform::ALL.map(D6Transform::id),
        "experiment_id": EXPERIMENT_ID,
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
        "source_semantic_blake3": semantic["source_semantic_blake3"],
        "spatial_token_capacity": spatial_token_capacity(arm),
        "split_records": {
            "train": splits["train"].records,
            "validation": splits["validation"].records,
        },
        "target_blake3": semantic["target_blake3"],
    });
    Ok((splits, semantic, overflow, scientific_identity))
}

fn write_board_tokens(
    slots: &mut [u16],
    features: &mut [i8],
    transform_index: usize,
    board_index: usize,
    token_capacity: usize,
    board: &SpatialBoardRepresentation,
    arm: SpatialArm,
) -> Result<(), Box<dyn Error>> {
    let center = board.center();
    let mut row = 0usize;
    for (ordinal, entity) in board.exact_entities().iter().enumerate() {
        write_token(
            slots,
            features,
            transform_index,
            board_index,
            row,
            ordinal,
            entity.coord.q,
            entity.coord.r,
            center.q,
            center.r,
            1,
            entity.channels,
            token_capacity,
        )?;
        row += 1;
    }
    for entity in board.local_entities() {
        let coord = arm
            .local_coord(entity.index)
            .ok_or("Rust spatial representation returned an invalid local index")?;
        write_token(
            slots,
            features,
            transform_index,
            board_index,
            row,
            entity.index as usize,
            coord.q,
            coord.r,
            center.q,
            center.r,
            2,
            entity.channels,
            token_capacity,
        )?;
        row += 1;
    }
    let overflow_base = arm.local_capacity();
    for (ordinal, entity) in board.overflow_entities().iter().enumerate() {
        write_token(
            slots,
            features,
            transform_index,
            board_index,
            row,
            overflow_base + ordinal,
            entity.coord.q,
            entity.coord.r,
            center.q,
            center.r,
            3,
            entity.channels,
            token_capacity,
        )?;
        row += 1;
    }
    if row != board.entity_count() {
        return Err("board token export changed the entity count".into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_token(
    slots: &mut [u16],
    features: &mut [i8],
    transform_index: usize,
    board_index: usize,
    entity_row: usize,
    slot: usize,
    q: i8,
    r: i8,
    center_q: i8,
    center_r: i8,
    path_code: i8,
    channels: [u8; 6],
    token_capacity: usize,
) -> Result<(), Box<dyn Error>> {
    if entity_row >= MAX_ENTITIES_PER_BOARD || slot >= token_capacity {
        return Err("exported token index exceeds its frozen tensor shape".into());
    }
    let row_index =
        (transform_index * BOARD_SLOTS + board_index) * MAX_ENTITIES_PER_BOARD + entity_row;
    if slots[row_index] != SLOT_SENTINEL {
        return Err("exported token row was written twice".into());
    }
    slots[row_index] = u16::try_from(slot)?;
    let offset = row_index * TOKEN_FIELDS;
    features[offset..offset + TOKEN_FIELDS].copy_from_slice(&[
        q,
        r,
        center_q,
        center_r,
        path_code,
        i8::try_from(channels[0])?,
        optional_category(channels[1], 5)?,
        i8::try_from(channels[2])?,
        i8::try_from(channels[3])?,
        optional_category(channels[4], 5)?,
        i8::try_from(channels[5])?,
    ]);
    Ok(())
}

fn optional_category(value: u8, none_code: i8) -> Result<i8, Box<dyn Error>> {
    if value == NONE {
        Ok(none_code)
    } else {
        Ok(i8::try_from(value)?)
    }
}

fn validate_sparse_record(
    slots: &[u16],
    features: &[i8],
    token_capacity: usize,
    arm: SpatialArm,
) -> Result<(), Box<dyn Error>> {
    for transform in 0..D6_TRANSFORMS {
        for board in 0..BOARD_SLOTS {
            let mut used = BTreeSet::new();
            for row in 0..MAX_ENTITIES_PER_BOARD {
                let index = (transform * BOARD_SLOTS + board) * MAX_ENTITIES_PER_BOARD + row;
                let feature = &features[index * TOKEN_FIELDS..(index + 1) * TOKEN_FIELDS];
                let slot = slots[index];
                if slot == SLOT_SENTINEL {
                    if feature.iter().any(|value| *value != 0) {
                        return Err("padding token contains nonzero features".into());
                    }
                    continue;
                }
                if usize::from(slot) >= token_capacity || !used.insert(slot) {
                    return Err("active token slot is out of range or duplicated".into());
                }
                let path = feature[4];
                match arm {
                    SpatialArm::ExactEntityControl if path != 1 => {
                        return Err("exact arm emitted a non-exact token".into());
                    }
                    SpatialArm::ExactEntityControl => {}
                    _ if usize::from(slot) < arm.local_capacity() && path != 2 => {
                        return Err("local token path code disagrees with its slot".into());
                    }
                    _ if usize::from(slot) >= arm.local_capacity() && path != 3 => {
                        return Err("overflow token path code disagrees with its slot".into());
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

fn market_features(
    record: &PositionRecord,
) -> Result<([f32; 4 * MARKET_FEATURES], [u8; 4]), Box<dyn Error>> {
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
        output[base + 30] = raw[4] as f32;
    }
    Ok((output, mask))
}

fn global_features(record: &PositionRecord) -> Result<[f32; GLOBAL_FEATURES], Box<dyn Error>> {
    let mut output = [0.0; GLOBAL_FEATURES];
    let mut offset = 0usize;
    let total_turns = f32::from(record.total_turns).max(1.0);
    output[offset] = f32::from(record.turn) / total_turns;
    offset += 1;
    output[offset] = (f32::from(record.total_turns) - f32::from(record.turn)) / total_turns;
    offset += 1;
    if !(1..=4).contains(&record.player_count) {
        return Err("player count is outside the frozen four-slot schema".into());
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
    let mut diversity = BTreeSet::new();
    for market in record.market_entities {
        let wildlife = market[3];
        if wildlife < 5 {
            output[offset + usize::from(wildlife)] = 1.0;
            diversity.insert(wildlife);
        }
        offset += 5;
    }
    for card in record.scoring_cards {
        if card >= 4 {
            return Err("scoring-card code is outside the frozen four-card schema".into());
        }
        output[offset + usize::from(card)] = 1.0;
        offset += 4;
    }
    output[offset] = f32::from(record.habitat_bonuses);
    offset += 1;
    output[offset] = diversity.len() as f32 / 4.0;
    offset += 1;
    if offset != GLOBAL_FEATURES {
        return Err("global feature width drifted".into());
    }
    Ok(output)
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

fn spatial_token_capacity(arm: SpatialArm) -> usize {
    match arm {
        SpatialArm::ExactEntityControl => MAX_ENTITIES_PER_BOARD,
        _ => arm.local_capacity() + MAX_ENTITIES_PER_BOARD,
    }
}

fn tensor_contract(arm: SpatialArm) -> Value {
    json!({
        "board_slots": BOARD_SLOTS,
        "d6_transform_ids": D6Transform::ALL.map(D6Transform::id),
        "global_feature_dim": GLOBAL_FEATURES,
        "local_capacity": arm.local_capacity(),
        "market_feature_dim": MARKET_FEATURES,
        "max_entities_per_board": MAX_ENTITIES_PER_BOARD,
        "padding": {
            "features": "all-zero",
            "mask_rule": "token-slot != 65535",
            "slot_sentinel": SLOT_SENTINEL,
        },
        "spatial_token_capacity": spatial_token_capacity(arm),
        "token_fields": [
            "q",
            "r",
            "center_q",
            "center_r",
            "path_code",
            "terrain_a",
            "terrain_b_or_none_5",
            "rotation",
            "allowed_wildlife_mask",
            "placed_wildlife_or_none_5",
            "keystone",
        ],
        "token_path_codes": {
            "exact": 1,
            "local": 2,
            "overflow": 3,
        },
    })
}

fn corpus_manifest_value(lock: &CorpusLock) -> Value {
    json!({
        "contract_id": lock.contract_id,
        "lock_id": lock.lock_id,
        "identity": lock.identity,
    })
}

fn insert_file(files: &mut BTreeMap<String, FileIdentity>, name: &str, file: FileIdentity) {
    let previous = files.insert(name.to_owned(), file);
    debug_assert!(previous.is_none());
}

fn dtype_width(dtype: &str) -> usize {
    match dtype {
        "|i1" | "|u1" => 1,
        "<u2" => 2,
        "<f4" => 4,
        "<u8" => 8,
        _ => panic!("unsupported frozen cache dtype {dtype}"),
    }
}

fn read_corpus_lock(path: &Path) -> Result<CorpusLock, Box<dyn Error>> {
    let lock: CorpusLock = serde_json::from_slice(&fs::read(path)?)?;
    if lock.schema_version != CORPUS_LOCK_SCHEMA_VERSION || lock.contract_id != CORPUS_LOCK_CONTRACT
    {
        return Err("unsupported R0 MLX corpus lock".into());
    }
    let identity = serde_json::to_value(&lock.identity)?;
    let expected = blake3::hash(&canonical_json_bytes(&identity))
        .to_hex()
        .to_string();
    if expected != lock.lock_id {
        return Err("R0 MLX corpus lock identity hash drifted".into());
    }
    if lock.identity.total_records != 60_000
        || lock.identity.train_records != 50_000
        || lock.identity.validation_records != 10_000
        || lock.identity.datasets.len() != 8
    {
        return Err("R0 MLX corpus lock is not the frozen 60,000-row corpus".into());
    }
    Ok(lock)
}

fn validate_corpus(
    roots: &[PathBuf],
    lock: &CorpusLock,
) -> Result<Vec<ValidatedDataset>, Box<dyn Error>> {
    if roots.len() != lock.identity.datasets.len() {
        return Err(format!(
            "corpus lock requires {} dataset roots; received {}",
            lock.identity.datasets.len(),
            roots.len()
        )
        .into());
    }
    let mut corpus_hasher = Hasher::new();
    corpus_hasher.update(CORPUS_DIGEST_PREFIX);
    let mut validated = Vec::with_capacity(roots.len());
    let mut train_records = 0usize;
    let mut validation_records = 0usize;
    for (order, (root, expected)) in roots.iter().zip(&lock.identity.datasets).enumerate() {
        if expected.order != order {
            return Err("corpus lock dataset order is not canonical".into());
        }
        let root_name = root
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or("dataset root requires a UTF-8 final component")?;
        if root_name != expected.root_name {
            return Err(format!(
                "dataset root order drifted at {order}: expected {}, found {root_name}",
                expected.root_name
            )
            .into());
        }
        let manifest_bytes = fs::read(root.join("dataset.json"))?;
        let manifest_blake3 = blake3::hash(&manifest_bytes).to_hex().to_string();
        if manifest_blake3 != expected.manifest_blake3 {
            return Err(format!("dataset manifest drifted: {}", root.display()).into());
        }
        let manifest: DatasetManifest = serde_json::from_slice(&manifest_bytes)?;
        validate_dataset(root, &manifest)?;
        if manifest.dataset_id != expected.dataset_id
            || manifest.first_game_index != expected.first_game_index
            || manifest.completed_games != expected.completed_games
            || manifest.total_records != expected.total_records
            || manifest.split.id() != expected.split
            || manifest.feature_schema != lock.identity.feature_schema
            || manifest.target_schema != lock.identity.target_schema
            || manifest.provenance.v2_source_blake3 != lock.identity.source_v2_blake3
        {
            return Err(format!("dataset identity drifted: {}", root.display()).into());
        }
        match manifest.split {
            DatasetSplit::Train => train_records += manifest.total_records,
            DatasetSplit::Validation => validation_records += manifest.total_records,
            DatasetSplit::Test | DatasetSplit::Final => {
                return Err("R0 MLX corpus contains prohibited test or final data".into());
            }
        }
        corpus_hasher.update(&(manifest_bytes.len() as u64).to_le_bytes());
        corpus_hasher.update(&manifest_bytes);
        validated.push(ValidatedDataset {
            root: root.clone(),
            manifest,
            manifest_blake3,
        });
    }
    if train_records != lock.identity.train_records
        || validation_records != lock.identity.validation_records
    {
        return Err("corpus split totals drifted from the lock".into());
    }
    let corpus_blake3 = corpus_hasher.finalize().to_hex().to_string();
    if corpus_blake3 != lock.identity.corpus_blake3 {
        return Err("corpus manifest sequence digest drifted".into());
    }
    if validated
        .iter()
        .any(|dataset| dataset.manifest_blake3.is_empty())
    {
        return Err("validated corpus lost a manifest digest".into());
    }
    Ok(validated)
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
            serde_json::to_value(sorted).expect("sorted JSON map serializes")
        }
        _ => value.clone(),
    }
}

fn parse_args(
    arguments: impl IntoIterator<Item = impl Into<std::ffi::OsString>>,
) -> Result<Args, Box<dyn Error>> {
    let mut arguments = arguments
        .into_iter()
        .map(Into::into)
        .collect::<Vec<_>>()
        .into_iter();
    let mut corpus_lock = None;
    let mut dataset_roots = Vec::new();
    let mut arm = None;
    let mut output_root = None;
    let mut receipt = None;
    while let Some(argument) = arguments.next() {
        let argument = argument
            .to_str()
            .ok_or("command-line arguments must be valid UTF-8")?;
        match argument {
            "--corpus-lock" => {
                corpus_lock = Some(PathBuf::from(
                    arguments.next().ok_or("--corpus-lock requires a path")?,
                ));
            }
            "--dataset-root" => {
                dataset_roots.push(PathBuf::from(
                    arguments.next().ok_or("--dataset-root requires a path")?,
                ));
            }
            "--arm" => {
                let value = arguments.next().ok_or("--arm requires an ID")?;
                let id = value.to_str().ok_or("--arm ID must be valid UTF-8")?;
                arm = Some(
                    SpatialArm::from_id(id)
                        .ok_or_else(|| format!("unknown spatial arm ID: {id}"))?,
                );
            }
            "--output-root" => {
                output_root = Some(PathBuf::from(
                    arguments.next().ok_or("--output-root requires a path")?,
                ));
            }
            "--receipt" => {
                receipt = Some(PathBuf::from(
                    arguments.next().ok_or("--receipt requires a path")?,
                ));
            }
            "--help" | "-h" => {
                println!("{HELP}");
                std::process::exit(0);
            }
            unknown => return Err(format!("unknown argument: {unknown}").into()),
        }
    }
    if dataset_roots.is_empty() {
        return Err("at least one --dataset-root is required".into());
    }
    Ok(Args {
        corpus_lock: corpus_lock.ok_or("--corpus-lock is required")?,
        dataset_roots,
        arm: arm.ok_or("--arm is required")?,
        output_root: output_root.ok_or("--output-root is required")?,
        receipt,
    })
}

fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or("output path requires a UTF-8 file name")?;
    let temporary = parent.join(format!(".{name}.{}.tmp", std::process::id()));
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record() -> PositionRecord {
        let mut board_entities = [[[NONE; 8]; MAX_BOARD_TILES]; BOARD_SLOTS];
        board_entities[0][0] = [0, 0, 0, NONE, 0, 0b00001, NONE, 1];
        PositionRecord {
            game_index: 17,
            turn: 4,
            active_seat: 0,
            player_count: 4,
            total_turns: 80,
            board_counts: [1, 0, 0, 0],
            nature_tokens: [1, 2, 3, 4],
            scoring_cards: [0, 0, 0, 0, 0],
            habitat_bonuses: false,
            wildlife_counts: [[0; 5]; BOARD_SLOTS],
            habitat_sizes: [[0; 5]; BOARD_SLOTS],
            board_entities,
            market_entities: [[0, NONE, 0b00001, 0, 1, 0, 0, 0]; 4],
            targets: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        }
    }

    #[test]
    fn parser_requires_the_locked_inputs_and_accepts_one_arm() {
        let args = parse_args([
            "--corpus-lock",
            "lock.json",
            "--dataset-root",
            "train",
            "--arm",
            "hex-radius-5-91",
            "--output-root",
            "cache",
            "--receipt",
            "receipt.json",
        ])
        .unwrap();
        assert_eq!(args.arm, SpatialArm::HexRadius5);
        assert_eq!(args.dataset_roots, [PathBuf::from("train")]);
        assert_eq!(args.receipt, Some(PathBuf::from("receipt.json")));
        assert!(parse_args(["--dataset-root", "train"]).is_err());
    }

    #[test]
    fn token_capacities_are_explicit_and_include_exact_overflow_slots() {
        assert_eq!(spatial_token_capacity(SpatialArm::ExactEntityControl), 23);
        assert_eq!(spatial_token_capacity(SpatialArm::HexRadius6), 150);
        assert_eq!(spatial_token_capacity(SpatialArm::HexRadius5), 114);
        assert_eq!(spatial_token_capacity(SpatialArm::HexRadius4), 84);
        assert_eq!(spatial_token_capacity(SpatialArm::HistoricalSquare21), 464);
    }

    #[test]
    fn rust_owned_token_export_is_sparse_padded_and_round_trippable() {
        let record = sample_record();
        for arm in SpatialArm::ALL {
            let representation = SpatialPositionRepresentation::from_record(&record, arm).unwrap();
            let mut slots =
                vec![SLOT_SENTINEL; D6_TRANSFORMS * BOARD_SLOTS * MAX_ENTITIES_PER_BOARD];
            let mut features =
                vec![0i8; D6_TRANSFORMS * BOARD_SLOTS * MAX_ENTITIES_PER_BOARD * TOKEN_FIELDS];
            for transform in D6Transform::ALL {
                let transformed = representation.transformed(transform).unwrap();
                for (board_index, board) in transformed.boards.iter().enumerate() {
                    write_board_tokens(
                        &mut slots,
                        &mut features,
                        transform.id() as usize,
                        board_index,
                        spatial_token_capacity(arm),
                        board,
                        arm,
                    )
                    .unwrap();
                }
            }
            validate_sparse_record(&slots, &features, spatial_token_capacity(arm), arm).unwrap();
            assert_eq!(
                slots.iter().filter(|slot| **slot != SLOT_SENTINEL).count(),
                D6_TRANSFORMS
            );
            assert!(
                slots
                    .iter()
                    .zip(features.chunks_exact(TOKEN_FIELDS))
                    .all(|(slot, feature)| *slot != SLOT_SENTINEL
                        || feature.iter().all(|value| *value == 0))
            );
        }
    }

    #[test]
    fn nonspatial_feature_shapes_match_the_frozen_v2_decoder() {
        let record = sample_record();
        let (market, mask) = market_features(&record).unwrap();
        let globals = global_features(&record).unwrap();
        assert_eq!(market.len(), 4 * MARKET_FEATURES);
        assert_eq!(mask, [1, 1, 1, 1]);
        assert_eq!(globals.len(), GLOBAL_FEATURES);
        assert_eq!(globals[0], 4.0 / 80.0);
        assert_eq!(globals[1], 76.0 / 80.0);
    }

    #[test]
    fn canonical_json_hash_ignores_object_insertion_order() {
        let left = json!({"b": {"z": 2, "a": 1}, "a": [3, 4]});
        let right = json!({"a": [3, 4], "b": {"a": 1, "z": 2}});
        assert_eq!(canonical_json_bytes(&left), canonical_json_bytes(&right));
        assert_eq!(
            blake3::hash(&canonical_json_bytes(&left)),
            blake3::hash(&canonical_json_bytes(&right))
        );
    }
}
