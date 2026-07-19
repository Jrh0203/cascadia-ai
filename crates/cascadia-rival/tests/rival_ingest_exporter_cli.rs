//! End-to-end test of the real `rival-ingest-exporter` binary: a directory
//! of raw exporter games in, sealed trajectory ledgers plus a deterministic
//! manifest out, and a hard non-zero exit on any per-file failure.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use cascadia_rival::{Sha256Digest, TrajectoryLedger};

const FIXTURE_SEED: u64 = 2027160000;
const DEFAULT_POLICY_ID: &str = "incumbent:cascadia-v3-cycle4-n1024-d16-gate0-20260716";

struct TemporaryDirectory(PathBuf);

impl TemporaryDirectory {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "cascadia-rival-ingest-cli-{label}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create isolated CLI test directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(format!("gumbel_game_seed_{FIXTURE_SEED}.jsonl"))
}

fn run_binary(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rival-ingest-exporter"))
        .args(arguments)
        .output()
        .expect("execute rival-ingest-exporter")
}

#[test]
fn ingests_a_raw_game_directory_skips_strays_and_prints_a_deterministic_manifest() {
    let raw = TemporaryDirectory::new("raw");
    let out = TemporaryDirectory::new("ledgers");
    let raw_game = raw.path().join(format!("gumbel_game_seed_{FIXTURE_SEED}.jsonl"));
    fs::copy(fixture_path(), &raw_game).unwrap();
    // Non-matching names (worker stderr logs, manifests) must be skipped.
    fs::write(
        raw.path()
            .join(format!("gumbel_game_seed_{FIXTURE_SEED}.jsonl.stderr")),
        "worker noise\n",
    )
    .unwrap();
    fs::write(raw.path().join("manifest.json"), "{}\n").unwrap();

    let output = run_binary(&[
        raw.path().to_str().unwrap(),
        "--out-dir",
        out.path().to_str().unwrap(),
    ]);
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let ledger_name = format!("rival_incumbent_ledger_seed_{FIXTURE_SEED}.json");
    let published = fs::read(out.path().join(&ledger_name)).expect("published sealed ledger");
    let ledger = TrajectoryLedger::from_json_slice(&published)
        .expect("published ledger re-verifies from bytes");
    assert_eq!(
        ledger.source_game_id(),
        format!("{DEFAULT_POLICY_ID}-{FIXTURE_SEED}")
    );
    assert_eq!(
        fs::read_dir(out.path()).unwrap().count(),
        1,
        "exactly one sealed ledger and no temporaries"
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    let file_sha256 = Sha256Digest::of_bytes(&published);
    let summary_line = format!(
        "ingested seed={FIXTURE_SEED} decisions=80 scores=99/99/98/99 ledger={ledger_name} {file_sha256}"
    );
    let manifest_header = format!("manifest policy_id={DEFAULT_POLICY_ID} count=1");
    let manifest_line = format!("manifest {ledger_name} {file_sha256}");
    for expected in [&summary_line, &manifest_header, &manifest_line] {
        assert!(
            stdout.lines().any(|line| line == expected.as_str()),
            "stdout is missing {expected:?}:\n{stdout}"
        );
    }

    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("skipping non-game file"), "stderr: {stderr}");
}

#[test]
fn any_bad_game_file_is_a_process_failure() {
    let raw = TemporaryDirectory::new("raw-bad");
    let out = TemporaryDirectory::new("ledgers-bad");
    let contents = fs::read_to_string(fixture_path()).unwrap().replace(
        "cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_16",
        "cascadia_house_rules_2026",
    );
    fs::write(
        raw.path().join(format!("gumbel_game_seed_{FIXTURE_SEED}.jsonl")),
        contents,
    )
    .unwrap();

    let output = run_binary(&[
        raw.path().to_str().unwrap(),
        "--out-dir",
        out.path().to_str().unwrap(),
    ]);
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("ingest failed"), "stderr: {stderr}");
    assert_eq!(
        fs::read_dir(out.path()).unwrap().count(),
        0,
        "a failed run publishes nothing for the failing file"
    );
}

#[test]
fn an_empty_or_gameless_raw_directory_is_a_process_failure() {
    let raw = TemporaryDirectory::new("raw-empty");
    let out = TemporaryDirectory::new("ledgers-empty");
    fs::write(raw.path().join("notes.txt"), "no games here\n").unwrap();
    let output = run_binary(&[
        raw.path().to_str().unwrap(),
        "--out-dir",
        out.path().to_str().unwrap(),
    ]);
    assert!(!output.status.success());

    let invalid_policy = run_binary(&[
        raw.path().to_str().unwrap(),
        "--out-dir",
        out.path().to_str().unwrap(),
        "--policy-id",
        "gate0-not-incumbent",
    ]);
    assert!(!invalid_policy.status.success());
}
