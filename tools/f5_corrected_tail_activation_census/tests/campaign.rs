use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

use f5_corrected_tail_activation_census::{
    GenerateShardConfig, aggregate_reports, census_shard, generate_shard, read_aggregate_report,
    read_manifest, read_shard_report, verify_reports_byte_identical,
};
use tempfile::TempDir;

struct SmokeCampaign {
    _temporary: TempDir,
    roots: Vec<PathBuf>,
    reports: Vec<PathBuf>,
}

impl SmokeCampaign {
    fn generate() -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let mut roots = Vec::new();
        let mut reports = Vec::new();
        for shard_index in 0..4 {
            let root = temporary
                .path()
                .join("corpus")
                .join(format!("shard-{shard_index}"));
            generate_shard(&GenerateShardConfig {
                output_root: root.clone(),
                shard_index,
                shard_count: 4,
                first_game_index: 0,
                total_games: 4,
                threads: 1,
            })
            .unwrap();
            let report = temporary
                .path()
                .join("reports")
                .join(format!("shard-{shard_index}.json"));
            census_shard(&root, &report).unwrap();
            roots.push(root);
            reports.push(report);
        }
        Self {
            _temporary: temporary,
            roots,
            reports,
        }
    }
}

#[test]
fn compiled_binary_runs_from_immutable_bundle_without_git_metadata() {
    let temporary = tempfile::tempdir().unwrap();
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .unwrap();
    let bundle = temporary.path().join("bundle");
    for relative in [
        "CASCADIA_V2_GOAL.txt",
        "tools/f5_corrected_tail_activation_census/Cargo.toml",
        "tools/f5_corrected_tail_activation_census/Cargo.lock",
        "tools/f5_corrected_tail_activation_census/build.rs",
        "tools/f5_corrected_tail_activation_census/src",
        "tools/f5_corrected_tail_activation_census/tests",
        "legacy/crates/cascadia-ai/Cargo.toml",
        "legacy/crates/cascadia-ai/src",
        "legacy/crates/cascadia-core/Cargo.toml",
        "legacy/crates/cascadia-core/src",
        "docs/v2/decisions/0149-corrected-tail-activation-census.md",
        "docs/v2/reports/corrected-mid-tail-activation-census-v1-preregistration.md",
    ] {
        copy_relative(repository, &bundle, relative);
    }
    assert!(!bundle.join(".git").exists());

    let corpus = temporary.path().join("bundle-output").join("shard-0");
    run_bundle_binary(
        &bundle,
        &[
            "generate-shard",
            "--output-root",
            corpus.to_str().unwrap(),
            "--shard-index",
            "0",
            "--shard-count",
            "4",
            "--first-game-index",
            "0",
            "--total-games",
            "4",
            "--threads",
            "1",
        ],
    );
    let manifest = read_manifest(&corpus).unwrap();
    assert_eq!(
        manifest.scientific.source.git_revision,
        env!("F5_SOURCE_GIT_REVISION")
    );
    assert_ne!(manifest.scientific.source.git_revision, "unavailable");
    assert!(
        manifest
            .scientific
            .source
            .files
            .iter()
            .any(|file| file.file == "tools/f5_corrected_tail_activation_census/build.rs")
    );

    let report = temporary.path().join("bundle-output").join("report-0.json");
    run_bundle_binary(
        &bundle,
        &[
            "census-shard",
            "--corpus-root",
            corpus.to_str().unwrap(),
            "--output",
            report.to_str().unwrap(),
        ],
    );
    read_shard_report(&report).unwrap();
}

#[test]
fn four_shard_smoke_is_order_independent_and_fail_closed() {
    let campaign = SmokeCampaign::generate();
    let forward = campaign._temporary.path().join("aggregate-forward.json");
    let reverse = campaign._temporary.path().join("aggregate-reverse.json");
    aggregate_reports(&campaign.reports, 4, &forward).unwrap();
    let reversed = campaign.reports.iter().rev().cloned().collect::<Vec<_>>();
    aggregate_reports(&reversed, 4, &reverse).unwrap();
    verify_reports_byte_identical(&forward, &reverse).unwrap();
    let aggregate = read_aggregate_report(&forward).unwrap();
    assert_eq!(
        aggregate.scientific.classification,
        "corrected_mid_tail_activation_census_smoke_complete"
    );
    assert_eq!(aggregate.scientific.total_games, 4);
    assert_eq!(aggregate.scientific.statistics.rows, 320);
    assert!(
        aggregate
            .scientific
            .overflow_witness
            .excluded_from_representativeness_statistics
    );

    let duplicate_manifest_root = campaign._temporary.path().join("duplicate-manifest");
    copy_tree(&campaign.roots[1], &duplicate_manifest_root);
    duplicate_once(
        &duplicate_manifest_root.join("manifest.json"),
        "    \"schema_version\": 2,",
        "\n",
    );
    assert_duplicate_rejected(read_manifest(&duplicate_manifest_root));

    let duplicate_record_root = campaign._temporary.path().join("duplicate-record");
    copy_tree(&campaign.roots[2], &duplicate_record_root);
    duplicate_once(
        &duplicate_record_root.join("records.jsonl"),
        "\"ruleset\":\"four_player_aaaaa_no_habitat_bonus_labels\",",
        "",
    );
    refresh_manifest_payload(&duplicate_record_root, ManifestPayload::Records);
    assert_duplicate_rejected(census_shard(
        &duplicate_record_root,
        &campaign
            ._temporary
            .path()
            .join("duplicate-record-census.json"),
    ));

    let duplicate_witness_root = campaign._temporary.path().join("duplicate-witness");
    copy_tree(&campaign.roots[0], &duplicate_witness_root);
    duplicate_once(
        &duplicate_witness_root.join("overflow-witness.json"),
        "    \"ruleset\": \"four_player_aaaaa_no_habitat_bonus_labels\",",
        "\n",
    );
    refresh_manifest_payload(&duplicate_witness_root, ManifestPayload::Witness);
    assert_duplicate_rejected(census_shard(
        &duplicate_witness_root,
        &campaign
            ._temporary
            .path()
            .join("duplicate-witness-census.json"),
    ));

    let duplicate_report = campaign._temporary.path().join("duplicate-report.json");
    fs::copy(&campaign.reports[1], &duplicate_report).unwrap();
    duplicate_once(&duplicate_report, "    \"schema_version\": 2,", "\n");
    assert_duplicate_rejected(read_shard_report(&duplicate_report));

    let duplicate_aggregate = campaign._temporary.path().join("duplicate-aggregate.json");
    fs::copy(&forward, &duplicate_aggregate).unwrap();
    duplicate_once(&duplicate_aggregate, "    \"schema_version\": 2,", "\n");
    assert_duplicate_rejected(read_aggregate_report(&duplicate_aggregate));

    let duplicate = vec![
        campaign.reports[0].clone(),
        campaign.reports[0].clone(),
        campaign.reports[2].clone(),
        campaign.reports[3].clone(),
    ];
    assert!(
        aggregate_reports(
            &duplicate,
            4,
            &campaign._temporary.path().join("duplicate.json")
        )
        .is_err()
    );
    assert!(
        aggregate_reports(
            &campaign.reports[..3],
            4,
            &campaign._temporary.path().join("gap.json")
        )
        .is_err()
    );

    let overlap_report = campaign._temporary.path().join("overlap-report.json");
    let mut report = read_shard_report(&campaign.reports[1]).unwrap();
    report.scientific.ownership.owned_game_indices[0] = 0;
    write_rehashed_report(&overlap_report, report);
    let reports_with_overlap = vec![
        campaign.reports[0].clone(),
        overlap_report,
        campaign.reports[2].clone(),
        campaign.reports[3].clone(),
    ];
    assert!(
        aggregate_reports(
            &reports_with_overlap,
            4,
            &campaign._temporary.path().join("forged-overlap.json")
        )
        .is_err()
    );

    let gap_report = campaign._temporary.path().join("gap-report.json");
    let mut report = read_shard_report(&campaign.reports[2]).unwrap();
    report.scientific.ownership.owned_game_indices[0] = 99;
    write_rehashed_report(&gap_report, report);
    let reports_with_gap = vec![
        campaign.reports[0].clone(),
        campaign.reports[1].clone(),
        gap_report,
        campaign.reports[3].clone(),
    ];
    assert!(
        aggregate_reports(
            &reports_with_gap,
            4,
            &campaign._temporary.path().join("forged-gap.json")
        )
        .is_err()
    );

    let corrupt_report = campaign._temporary.path().join("corrupt-report.json");
    fs::copy(&campaign.reports[1], &corrupt_report).unwrap();
    let mut report = read_shard_report(&corrupt_report).unwrap();
    report.scientific_blake3 = "0".repeat(64);
    fs::write(&corrupt_report, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    let reports_with_corruption = vec![
        campaign.reports[0].clone(),
        corrupt_report,
        campaign.reports[2].clone(),
        campaign.reports[3].clone(),
    ];
    assert!(
        aggregate_reports(
            &reports_with_corruption,
            4,
            &campaign._temporary.path().join("corrupt-aggregate.json")
        )
        .is_err()
    );

    let corrupt_root = campaign._temporary.path().join("corrupt-corpus");
    copy_tree(&campaign.roots[2], &corrupt_root);
    let records = corrupt_root.join("records.jsonl");
    let mut file = OpenOptions::new().append(true).open(records).unwrap();
    file.write_all(b"{malformed\n").unwrap();
    file.flush().unwrap();
    assert!(
        census_shard(
            &corrupt_root,
            &campaign._temporary.path().join("corrupt-census.json")
        )
        .is_err()
    );

    let malformed_root = campaign._temporary.path().join("malformed-corpus");
    copy_tree(&campaign.roots[3], &malformed_root);
    let malformed_records = malformed_root.join("records.jsonl");
    let mut file = OpenOptions::new()
        .append(true)
        .open(&malformed_records)
        .unwrap();
    file.write_all(b"{malformed\n").unwrap();
    file.flush().unwrap();
    let payload = fs::read(&malformed_records).unwrap();
    let mut manifest = read_manifest(&malformed_root).unwrap();
    manifest.scientific.records.bytes = payload.len() as u64;
    manifest.scientific.records.blake3 = blake3::hash(&payload).to_hex().to_string();
    manifest.scientific_blake3 = blake3::hash(&serde_json::to_vec(&manifest.scientific).unwrap())
        .to_hex()
        .to_string();
    let mut manifest_bytes = serde_json::to_vec_pretty(&manifest).unwrap();
    manifest_bytes.push(b'\n');
    fs::write(malformed_root.join("manifest.json"), manifest_bytes).unwrap();
    assert!(
        census_shard(
            &malformed_root,
            &campaign._temporary.path().join("malformed-census.json")
        )
        .is_err()
    );
}

#[derive(Clone, Copy)]
enum ManifestPayload {
    Records,
    Witness,
}

fn refresh_manifest_payload(root: &Path, kind: ManifestPayload) {
    let mut manifest = read_manifest(root).unwrap();
    let file = match kind {
        ManifestPayload::Records => manifest.scientific.records.file.clone(),
        ManifestPayload::Witness => manifest
            .scientific
            .overflow_witness
            .as_ref()
            .unwrap()
            .payload
            .file
            .clone(),
    };
    let bytes = fs::read(root.join(file)).unwrap();
    let payload = match kind {
        ManifestPayload::Records => &mut manifest.scientific.records,
        ManifestPayload::Witness => {
            &mut manifest
                .scientific
                .overflow_witness
                .as_mut()
                .unwrap()
                .payload
        }
    };
    payload.bytes = bytes.len() as u64;
    payload.blake3 = blake3::hash(&bytes).to_hex().to_string();
    manifest.scientific_blake3 = canonical_json_blake3(&manifest.scientific);
    let mut manifest_bytes = serde_json::to_vec_pretty(&manifest).unwrap();
    manifest_bytes.push(b'\n');
    fs::write(root.join("manifest.json"), manifest_bytes).unwrap();
}

fn duplicate_once(path: &Path, needle: &str, separator: &str) {
    let original = fs::read_to_string(path).unwrap();
    assert!(
        original.contains(needle),
        "duplicate-key fixture target is absent from {}",
        path.display()
    );
    let replacement = format!("{needle}{separator}{needle}");
    fs::write(path, original.replacen(needle, &replacement, 1)).unwrap();
}

fn assert_duplicate_rejected<T>(result: f5_corrected_tail_activation_census::Result<T>) {
    let error = match result {
        Ok(_) => panic!("duplicate JSON key must fail closed"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("duplicate object key"),
        "unexpected strict-JSON error: {error}"
    );
}

fn write_rehashed_report(
    path: &Path,
    mut report: f5_corrected_tail_activation_census::ShardReport,
) {
    report.scientific.ownership.owned_game_indices_blake3 =
        canonical_json_blake3(&report.scientific.ownership.owned_game_indices);
    report.scientific_blake3 = canonical_json_blake3(&report.scientific);
    let mut bytes = serde_json::to_vec_pretty(&report).unwrap();
    bytes.push(b'\n');
    fs::write(path, bytes).unwrap();
}

fn canonical_json_blake3(value: &impl serde::Serialize) -> String {
    blake3::hash(&serde_json::to_vec(value).unwrap())
        .to_hex()
        .to_string()
}

fn copy_relative(repository: &Path, bundle: &Path, relative: &str) {
    let source = repository.join(relative);
    let destination = bundle.join(relative);
    if source.is_dir() {
        copy_tree(&source, &destination);
    } else {
        fs::create_dir_all(destination.parent().unwrap()).unwrap();
        fs::copy(source, destination).unwrap();
    }
}

fn run_bundle_binary(bundle: &Path, arguments: &[&str]) {
    let output = Command::new(env!("CARGO_BIN_EXE_f5-corrected-tail-activation-census"))
        .current_dir(bundle)
        .args(arguments)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "bundle command failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn copy_tree(source: &Path, destination: &Path) {
    fs::create_dir_all(destination).unwrap();
    for entry in fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        if source_path.is_dir() {
            copy_tree(&source_path, &destination_path);
        } else {
            fs::copy(source_path, destination_path).unwrap();
        }
    }
}
