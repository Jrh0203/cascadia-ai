use std::{
    env,
    error::Error,
    fs,
    hint::black_box,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

use blake3::Hasher;
use cascadia_data::{
    DatasetManifest, PositionRecord, PositionShardReader, RECORD_SIZE, SpatialArm,
    SpatialPositionRepresentation, SpatialRepresentationAccounting, validate_dataset,
};
use serde::Serialize;

const BENCHMARK_SCHEMA_VERSION: u16 = 1;
const BENCHMARK_ID: &str = "r0-spatial-representation-extraction-v1";
const MAX_REPLICATE_INDEX: usize = 2;
const HELP: &str = concat!(
    "Usage: spatial_representation_benchmark \\\n",
    "  --dataset-root PATH [--dataset-root PATH ...] \\\n",
    "  --replicate-index R \\\n",
    "  [--arm ID ...] [--shard-index I --shard-count N] \\\n",
    "  [--records N] [--iterations N] [--output PATH]\n\n",
    "Arm IDs:\n",
    "  exact-entity-control\n",
    "  hex-radius-6-127\n",
    "  hex-radius-5-91\n",
    "  hex-radius-4-61\n",
    "  historical-square-21x21-441\n\n",
    "Repeated --arm selects a canonical subset; default is all arms.\n",
    "--replicate-index is required and must be 0, 1, or 2.\n",
    "--records 0 loads every record eligible for this shard (default: 1024).\n",
    "Partitioning uses global_ordinal % shard_count == shard_index before limiting."
);

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    dataset_roots: Vec<PathBuf>,
    arms: Vec<SpatialArm>,
    replicate_index: usize,
    shard_index: usize,
    shard_count: usize,
    record_limit: usize,
    iterations: usize,
    output: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    schema_version: u16,
    benchmark_id: &'static str,
    record_count: usize,
    iterations: usize,
    replicate_index: usize,
    selected_arms: Vec<&'static str>,
    shard: ShardMetadata,
    execution_provenance: ExecutionProvenance,
    validation_and_read_seconds: f64,
    source_semantic_blake3: String,
    datasets: Vec<DatasetIdentity>,
    arms: Vec<ArmBenchmark>,
}

#[derive(Debug, Serialize)]
struct ExecutionProvenance {
    hostname: String,
    os: &'static str,
    arch: &'static str,
    logical_parallelism: Option<usize>,
    cpu_brand: String,
    memory_bytes: Option<u64>,
    hardware_description: String,
}

#[derive(Debug, Serialize)]
struct ShardMetadata {
    shard_index: usize,
    shard_count: usize,
    ordinal_rule: &'static str,
    record_limit_after_partition: usize,
    total_manifest_records: usize,
    total_eligible_records: usize,
    loaded_records: usize,
}

#[derive(Debug, Serialize)]
struct DatasetIdentity {
    root: String,
    dataset_id: String,
    feature_schema: String,
    split: String,
    completed_games: usize,
    total_records: usize,
    global_ordinal_start: usize,
    global_ordinal_end_exclusive: usize,
    eligible_records: usize,
    loaded_records: usize,
    manifest_blake3: String,
}

#[derive(Debug, Serialize)]
struct ArmBenchmark {
    arm: &'static str,
    records: usize,
    iterations: usize,
    round_trip_verified: bool,
    semantic_blake3: String,
    extraction_seconds: f64,
    extraction_ns_per_record: f64,
    extraction_records_per_second: f64,
    serialization_seconds: f64,
    serialization_ns_per_record: f64,
    deserialization_seconds: f64,
    deserialization_ns_per_record: f64,
    mean_packed_bytes: f64,
    mean_packed_bytes_vs_position_record: f64,
    mean_local_capacity_rows: f64,
    mean_active_local_rows: f64,
    local_occupancy_fraction: Option<f64>,
    mean_exact_entity_rows: f64,
    mean_overflow_entity_rows: f64,
    positions_with_overflow: usize,
    overflow_position_fraction: f64,
    mean_dense_raw_scalar_slots: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args(env::args_os().skip(1))?;
    let execution_provenance = capture_execution_provenance();
    let started = Instant::now();
    let loaded = load_records(&args)?;
    let validation_and_read_seconds = started.elapsed().as_secs_f64();
    let records = loaded.records;
    if records.is_empty() {
        return Err("no PositionRecord rows were loaded".into());
    }

    let source_semantic_blake3 = semantic_digest(records.iter());
    let mut arms = Vec::with_capacity(args.arms.len());
    for arm in &args.arms {
        arms.push(benchmark_arm(&records, *arm, args.iterations)?);
    }
    if arms
        .iter()
        .any(|arm| arm.semantic_blake3 != source_semantic_blake3)
    {
        return Err("one or more spatial arms changed PositionRecord semantics".into());
    }

    let report = BenchmarkReport {
        schema_version: BENCHMARK_SCHEMA_VERSION,
        benchmark_id: BENCHMARK_ID,
        record_count: records.len(),
        iterations: args.iterations,
        replicate_index: args.replicate_index,
        selected_arms: args.arms.iter().map(|arm| arm.id()).collect(),
        shard: ShardMetadata {
            shard_index: args.shard_index,
            shard_count: args.shard_count,
            ordinal_rule: "concatenated CLI dataset-root order, manifest shard order, in-shard row order; global_ordinal % shard_count == shard_index",
            record_limit_after_partition: args.record_limit,
            total_manifest_records: loaded.total_manifest_records,
            total_eligible_records: loaded.total_eligible_records,
            loaded_records: records.len(),
        },
        execution_provenance,
        validation_and_read_seconds,
        source_semantic_blake3,
        datasets: loaded.datasets,
        arms,
    };
    let encoded = serde_json::to_string_pretty(&report)? + "\n";
    if let Some(path) = &args.output {
        write_atomically(path, encoded.as_bytes())?;
    }
    print!("{encoded}");
    Ok(())
}

fn capture_execution_provenance() -> ExecutionProvenance {
    let hostname = command_output("hostname", &["-s"])
        .or_else(|| env::var("HOSTNAME").ok())
        .map(|value| short_hostname(&value))
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown".to_owned());
    let logical_parallelism = std::thread::available_parallelism()
        .ok()
        .map(|value| value.get());
    let cpu_brand = if std::env::consts::OS == "macos" {
        command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "unknown".to_owned())
    } else {
        "unknown".to_owned()
    };
    let memory_bytes = if std::env::consts::OS == "macos" {
        command_output("sysctl", &["-n", "hw.memsize"]).and_then(|value| value.parse::<u64>().ok())
    } else {
        None
    };
    let hardware_description = format!(
        "cpu={cpu_brand}; memory_bytes={}",
        memory_bytes
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unknown".to_owned())
    );
    ExecutionProvenance {
        hostname,
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        logical_parallelism,
        cpu_brand,
        memory_bytes,
        hardware_description,
    }
}

fn command_output(program: &str, arguments: &[&str]) -> Option<String> {
    let output = Command::new(program).args(arguments).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn short_hostname(value: &str) -> String {
    value
        .trim()
        .split_once('.')
        .map_or_else(|| value.trim(), |(short, _)| short)
        .to_owned()
}

fn parse_args(
    arguments: impl IntoIterator<Item = impl Into<std::ffi::OsString>>,
) -> Result<Args, Box<dyn Error>> {
    let mut arguments = arguments
        .into_iter()
        .map(Into::into)
        .collect::<Vec<_>>()
        .into_iter();
    let mut dataset_roots = Vec::new();
    let mut selected_arm_codes = [false; SpatialArm::ALL.len()];
    let mut arm_was_supplied = false;
    let mut replicate_index = None;
    let mut shard_index = 0usize;
    let mut shard_count = 1usize;
    let mut record_limit = 1_024usize;
    let mut iterations = 20usize;
    let mut output = None;

    while let Some(argument) = arguments.next() {
        let argument = argument
            .to_str()
            .ok_or("command-line arguments must be valid UTF-8")?;
        match argument {
            "--dataset-root" => {
                let value = arguments.next().ok_or("--dataset-root requires a path")?;
                dataset_roots.push(PathBuf::from(value));
            }
            "--arm" => {
                let value = arguments.next().ok_or("--arm requires an ID")?;
                let id = value.to_str().ok_or("--arm ID must be valid UTF-8")?;
                let arm = SpatialArm::from_id(id)
                    .ok_or_else(|| format!("unknown spatial arm ID: {id}"))?;
                let code = usize::from(arm.code());
                if selected_arm_codes[code] {
                    return Err(format!("duplicate spatial arm ID: {id}").into());
                }
                selected_arm_codes[code] = true;
                arm_was_supplied = true;
            }
            "--replicate-index" => {
                if replicate_index.is_some() {
                    return Err("--replicate-index may only be supplied once".into());
                }
                let value = arguments
                    .next()
                    .ok_or("--replicate-index requires a value")?;
                let value = value
                    .to_str()
                    .ok_or("--replicate-index must be valid UTF-8")?;
                let parsed = value.parse::<usize>().map_err(|_| {
                    format!("--replicate-index must be an integer in 0..={MAX_REPLICATE_INDEX}")
                })?;
                if parsed > MAX_REPLICATE_INDEX {
                    return Err(format!(
                        "--replicate-index must be an integer in 0..={MAX_REPLICATE_INDEX}"
                    )
                    .into());
                }
                replicate_index = Some(parsed);
            }
            "--shard-index" => {
                let value = arguments.next().ok_or("--shard-index requires a value")?;
                shard_index = value
                    .to_str()
                    .ok_or("--shard-index must be valid UTF-8")?
                    .parse()?;
            }
            "--shard-count" => {
                let value = arguments.next().ok_or("--shard-count requires a value")?;
                shard_count = value
                    .to_str()
                    .ok_or("--shard-count must be valid UTF-8")?
                    .parse()?;
            }
            "--records" => {
                let value = arguments.next().ok_or("--records requires a value")?;
                record_limit = value
                    .to_str()
                    .ok_or("--records must be valid UTF-8")?
                    .parse()?;
            }
            "--iterations" => {
                let value = arguments.next().ok_or("--iterations requires a value")?;
                iterations = value
                    .to_str()
                    .ok_or("--iterations must be valid UTF-8")?
                    .parse()?;
            }
            "--output" => {
                let value = arguments.next().ok_or("--output requires a path")?;
                output = Some(PathBuf::from(value));
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
    let replicate_index = replicate_index.ok_or("--replicate-index is required")?;
    if iterations == 0 {
        return Err("--iterations must be positive".into());
    }
    if shard_count == 0 {
        return Err("--shard-count must be positive".into());
    }
    if shard_index >= shard_count {
        return Err(format!(
            "--shard-index {shard_index} must be less than --shard-count {shard_count}"
        )
        .into());
    }
    let arms = if arm_was_supplied {
        SpatialArm::ALL
            .into_iter()
            .filter(|arm| selected_arm_codes[usize::from(arm.code())])
            .collect()
    } else {
        SpatialArm::ALL.to_vec()
    };
    Ok(Args {
        dataset_roots,
        arms,
        replicate_index,
        shard_index,
        shard_count,
        record_limit,
        iterations,
        output,
    })
}

struct LoadedRecords {
    records: Vec<PositionRecord>,
    datasets: Vec<DatasetIdentity>,
    total_manifest_records: usize,
    total_eligible_records: usize,
}

struct ValidatedDataset {
    root: PathBuf,
    manifest: DatasetManifest,
    manifest_blake3: String,
    global_ordinal_start: usize,
}

fn load_records(args: &Args) -> Result<LoadedRecords, Box<dyn Error>> {
    let mut validated = Vec::with_capacity(args.dataset_roots.len());
    let mut total_manifest_records = 0usize;
    for root in &args.dataset_roots {
        let manifest_path = root.join("dataset.json");
        let manifest_bytes = fs::read(&manifest_path)?;
        let manifest: DatasetManifest = serde_json::from_slice(&manifest_bytes)?;
        validate_dataset(root, &manifest)?;
        let global_ordinal_start = total_manifest_records;
        total_manifest_records = total_manifest_records
            .checked_add(manifest.total_records)
            .ok_or("manifested record count exceeds usize")?;
        validated.push(ValidatedDataset {
            root: root.clone(),
            manifest,
            manifest_blake3: blake3::hash(&manifest_bytes).to_hex().to_string(),
            global_ordinal_start,
        });
    }
    let total_eligible_records = eligible_count_in_range(
        0,
        total_manifest_records,
        args.shard_index,
        args.shard_count,
    );
    let mut records = Vec::new();
    let mut identities = Vec::with_capacity(validated.len());
    let mut limit_reached = false;
    for dataset in validated {
        let dataset_end = dataset
            .global_ordinal_start
            .checked_add(dataset.manifest.total_records)
            .ok_or("dataset ordinal range exceeds usize")?;
        let eligible_records = eligible_count_in_range(
            dataset.global_ordinal_start,
            dataset_end,
            args.shard_index,
            args.shard_count,
        );
        let before = records.len();
        if !limit_reached {
            let mut global_ordinal = dataset.global_ordinal_start;
            'shards: for shard in &dataset.manifest.shards {
                let reader = PositionShardReader::open(&dataset.root, shard)?;
                for record in reader {
                    let record = record?;
                    if record_is_eligible(global_ordinal, args.shard_index, args.shard_count) {
                        records.push(record);
                        if args.record_limit != 0 && records.len() == args.record_limit {
                            limit_reached = true;
                            break 'shards;
                        }
                    }
                    global_ordinal += 1;
                }
            }
        }
        identities.push(DatasetIdentity {
            root: dataset.root.display().to_string(),
            dataset_id: dataset.manifest.dataset_id,
            feature_schema: dataset.manifest.feature_schema,
            split: dataset.manifest.split.id().to_owned(),
            completed_games: dataset.manifest.completed_games,
            total_records: dataset.manifest.total_records,
            global_ordinal_start: dataset.global_ordinal_start,
            global_ordinal_end_exclusive: dataset_end,
            eligible_records,
            loaded_records: records.len() - before,
            manifest_blake3: dataset.manifest_blake3,
        });
    }
    Ok(LoadedRecords {
        records,
        datasets: identities,
        total_manifest_records,
        total_eligible_records,
    })
}

const fn record_is_eligible(global_ordinal: usize, shard_index: usize, shard_count: usize) -> bool {
    global_ordinal % shard_count == shard_index
}

fn eligible_count_in_range(
    start: usize,
    end_exclusive: usize,
    shard_index: usize,
    shard_count: usize,
) -> usize {
    if start >= end_exclusive {
        return 0;
    }
    let start_remainder = start % shard_count;
    let offset = if start_remainder <= shard_index {
        shard_index - start_remainder
    } else {
        shard_count - (start_remainder - shard_index)
    };
    if offset >= end_exclusive - start {
        return 0;
    }
    let first = start + offset;
    1 + (end_exclusive - 1 - first) / shard_count
}

fn benchmark_arm(
    records: &[PositionRecord],
    arm: SpatialArm,
    iterations: usize,
) -> Result<ArmBenchmark, Box<dyn Error>> {
    let mut representations = Vec::with_capacity(records.len());
    let mut packed_records = Vec::with_capacity(records.len());
    let mut accounting = SpatialRepresentationAccounting::default();
    let mut positions_with_overflow = 0usize;
    let mut semantic_hasher = Hasher::new();

    for record in records {
        let representation = SpatialPositionRepresentation::from_record(record, arm)?;
        let round_trip = representation.to_position_record()?;
        if round_trip != *record {
            return Err(format!("{} failed the in-memory round trip", arm.id()).into());
        }
        semantic_hasher.update(&round_trip.to_bytes());
        let packed = representation.to_packed_bytes()?;
        let decoded = SpatialPositionRepresentation::from_packed_bytes(&packed)?;
        if decoded != representation || decoded.to_position_record()? != *record {
            return Err(format!("{} failed the packed round trip", arm.id()).into());
        }
        let current = representation.accounting();
        add_accounting(&mut accounting, current);
        positions_with_overflow += usize::from(current.overflow_entity_rows > 0);
        representations.push(representation);
        packed_records.push(packed);
    }

    for record in records {
        black_box(SpatialPositionRepresentation::from_record(record, arm)?);
    }

    let extraction_started = Instant::now();
    let mut extraction_guard = 0usize;
    for _ in 0..iterations {
        for record in records {
            let representation = SpatialPositionRepresentation::from_record(record, arm)?;
            extraction_guard =
                extraction_guard.wrapping_add(representation.accounting().packed_bytes);
            black_box(representation);
        }
    }
    black_box(extraction_guard);
    let extraction_seconds = extraction_started.elapsed().as_secs_f64();

    let serialization_started = Instant::now();
    let mut serialization_guard = 0usize;
    for _ in 0..iterations {
        for representation in &representations {
            let packed = representation.to_packed_bytes()?;
            serialization_guard = serialization_guard.wrapping_add(packed.len());
            black_box(packed);
        }
    }
    black_box(serialization_guard);
    let serialization_seconds = serialization_started.elapsed().as_secs_f64();

    let deserialization_started = Instant::now();
    let mut deserialization_guard = 0usize;
    for _ in 0..iterations {
        for packed in &packed_records {
            let representation = SpatialPositionRepresentation::from_packed_bytes(packed)?;
            deserialization_guard =
                deserialization_guard.wrapping_add(representation.accounting().packed_bytes);
            black_box(representation);
        }
    }
    black_box(deserialization_guard);
    let deserialization_seconds = deserialization_started.elapsed().as_secs_f64();

    let operations = records.len() * iterations;
    let operations_f64 = operations as f64;
    let records_f64 = records.len() as f64;
    let local_occupancy_fraction = (accounting.local_capacity_rows > 0)
        .then_some(accounting.active_local_rows as f64 / accounting.local_capacity_rows as f64);
    Ok(ArmBenchmark {
        arm: arm.id(),
        records: records.len(),
        iterations,
        round_trip_verified: true,
        semantic_blake3: semantic_hasher.finalize().to_hex().to_string(),
        extraction_seconds,
        extraction_ns_per_record: extraction_seconds * 1_000_000_000.0 / operations_f64,
        extraction_records_per_second: operations_f64 / extraction_seconds,
        serialization_seconds,
        serialization_ns_per_record: serialization_seconds * 1_000_000_000.0 / operations_f64,
        deserialization_seconds,
        deserialization_ns_per_record: deserialization_seconds * 1_000_000_000.0 / operations_f64,
        mean_packed_bytes: accounting.packed_bytes as f64 / records_f64,
        mean_packed_bytes_vs_position_record: accounting.packed_bytes as f64
            / records_f64
            / RECORD_SIZE as f64,
        mean_local_capacity_rows: accounting.local_capacity_rows as f64 / records_f64,
        mean_active_local_rows: accounting.active_local_rows as f64 / records_f64,
        local_occupancy_fraction,
        mean_exact_entity_rows: accounting.exact_entity_rows as f64 / records_f64,
        mean_overflow_entity_rows: accounting.overflow_entity_rows as f64 / records_f64,
        positions_with_overflow,
        overflow_position_fraction: positions_with_overflow as f64 / records_f64,
        mean_dense_raw_scalar_slots: accounting.dense_raw_scalar_slots as f64 / records_f64,
    })
}

fn add_accounting(
    total: &mut SpatialRepresentationAccounting,
    value: SpatialRepresentationAccounting,
) {
    total.packed_bytes += value.packed_bytes;
    total.packed_spatial_bytes += value.packed_spatial_bytes;
    total.local_capacity_rows += value.local_capacity_rows;
    total.active_local_rows += value.active_local_rows;
    total.exact_entity_rows += value.exact_entity_rows;
    total.overflow_entity_rows += value.overflow_entity_rows;
    total.semantic_entity_rows += value.semantic_entity_rows;
    total.dense_raw_scalar_slots += value.dense_raw_scalar_slots;
}

fn semantic_digest<'a>(records: impl IntoIterator<Item = &'a PositionRecord>) -> String {
    let mut hasher = Hasher::new();
    for record in records {
        hasher.update(&record.to_bytes());
    }
    hasher.finalize().to_hex().to_string()
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

    #[test]
    fn parser_accepts_multiple_roots_and_zero_record_limit() {
        let args = parse_args([
            "--dataset-root",
            "one",
            "--dataset-root",
            "two",
            "--replicate-index",
            "2",
            "--records",
            "0",
            "--iterations",
            "3",
            "--output",
            "report.json",
        ])
        .unwrap();
        assert_eq!(
            args.dataset_roots,
            [PathBuf::from("one"), PathBuf::from("two")]
        );
        assert_eq!(args.arms, SpatialArm::ALL);
        assert_eq!(args.replicate_index, 2);
        assert_eq!(args.shard_index, 0);
        assert_eq!(args.shard_count, 1);
        assert_eq!(args.record_limit, 0);
        assert_eq!(args.iterations, 3);
        assert_eq!(args.output, Some(PathBuf::from("report.json")));
    }

    #[test]
    fn hostname_normalization_is_short_and_stable() {
        assert_eq!(short_hostname("john1.local\n"), "john1");
        assert_eq!(short_hostname("john4"), "john4");
        assert_eq!(short_hostname("  "), "");
    }

    #[test]
    fn parser_rejects_missing_roots_and_zero_iterations() {
        assert!(parse_args(["--replicate-index", "0", "--iterations", "1"]).is_err());
        assert!(
            parse_args([
                "--dataset-root",
                "one",
                "--replicate-index",
                "0",
                "--iterations",
                "0",
            ])
            .is_err()
        );
    }

    #[test]
    fn parser_requires_and_bounds_replicate_index() {
        assert_eq!(
            parse_args(["--dataset-root", "one"])
                .unwrap_err()
                .to_string(),
            "--replicate-index is required"
        );
        for replicate_index in ["0", "1", "2"] {
            let args = parse_args([
                "--dataset-root",
                "one",
                "--replicate-index",
                replicate_index,
            ])
            .unwrap();
            assert_eq!(
                args.replicate_index.to_string(),
                replicate_index,
                "replicate {replicate_index} should be accepted"
            );
        }
        for replicate_index in ["-1", "3", "18446744073709551616"] {
            assert_eq!(
                parse_args([
                    "--dataset-root",
                    "one",
                    "--replicate-index",
                    replicate_index,
                ])
                .unwrap_err()
                .to_string(),
                "--replicate-index must be an integer in 0..=2"
            );
        }
        assert_eq!(
            parse_args([
                "--dataset-root",
                "one",
                "--replicate-index",
                "0",
                "--replicate-index",
                "1",
            ])
            .unwrap_err()
            .to_string(),
            "--replicate-index may only be supplied once"
        );
    }

    #[test]
    fn parser_canonicalizes_selected_arms_and_rejects_bad_ids() {
        let args = parse_args([
            "--dataset-root",
            "one",
            "--replicate-index",
            "0",
            "--arm",
            "hex-radius-4-61",
            "--arm",
            "exact-entity-control",
        ])
        .unwrap();
        assert_eq!(
            args.arms,
            [SpatialArm::ExactEntityControl, SpatialArm::HexRadius4]
        );

        let duplicate = parse_args([
            "--dataset-root",
            "one",
            "--replicate-index",
            "0",
            "--arm",
            "hex-radius-5-91",
            "--arm",
            "hex-radius-5-91",
        ])
        .unwrap_err()
        .to_string();
        assert_eq!(duplicate, "duplicate spatial arm ID: hex-radius-5-91");

        let unknown = parse_args([
            "--dataset-root",
            "one",
            "--replicate-index",
            "0",
            "--arm",
            "radius-five-ish",
        ])
        .unwrap_err()
        .to_string();
        assert_eq!(unknown, "unknown spatial arm ID: radius-five-ish");
    }

    #[test]
    fn parser_validates_shard_contract() {
        let args = parse_args([
            "--dataset-root",
            "one",
            "--replicate-index",
            "1",
            "--shard-index",
            "2",
            "--shard-count",
            "4",
        ])
        .unwrap();
        assert_eq!(args.shard_index, 2);
        assert_eq!(args.shard_count, 4);

        assert_eq!(
            parse_args([
                "--dataset-root",
                "one",
                "--replicate-index",
                "1",
                "--shard-index",
                "0",
                "--shard-count",
                "0",
            ])
            .unwrap_err()
            .to_string(),
            "--shard-count must be positive"
        );
        assert_eq!(
            parse_args([
                "--dataset-root",
                "one",
                "--replicate-index",
                "1",
                "--shard-index",
                "4",
                "--shard-count",
                "4",
            ])
            .unwrap_err()
            .to_string(),
            "--shard-index 4 must be less than --shard-count 4"
        );
    }

    #[test]
    fn modulo_shards_are_disjoint_and_union_to_the_unsharded_stream() {
        let total = 103usize;
        let unsharded = (0..total).collect::<Vec<_>>();
        let mut union = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for shard_index in 0..4 {
            let shard = (0..total)
                .filter(|ordinal| record_is_eligible(*ordinal, shard_index, 4))
                .collect::<Vec<_>>();
            assert_eq!(
                shard.len(),
                eligible_count_in_range(0, total, shard_index, 4)
            );
            for ordinal in &shard {
                assert!(seen.insert(*ordinal), "ordinal {ordinal} appeared twice");
            }
            union.extend(shard);
        }
        union.sort_unstable();
        assert_eq!(union, unsharded);
    }

    #[test]
    fn record_limit_is_applied_after_partitioning() {
        let eligible = (0..30)
            .filter(|ordinal| record_is_eligible(*ordinal, 2, 4))
            .collect::<Vec<_>>();
        let limited = eligible.iter().copied().take(3).collect::<Vec<_>>();
        assert_eq!(limited, [2, 6, 10]);
        assert_eq!(eligible_count_in_range(0, 30, 2, 4), eligible.len());
        assert_eq!(eligible_count_in_range(5, 23, 2, 4), 5);
    }
}
