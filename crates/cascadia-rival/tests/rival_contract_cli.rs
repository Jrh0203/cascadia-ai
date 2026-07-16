use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

use cascadia_rival::{MAX_TERMINAL_PAIR_LEDGER_BYTES, Sha256Digest};

static TEMPORARY_DIRECTORY_ORDINAL: AtomicU64 = AtomicU64::new(0);

struct TemporaryDirectory(PathBuf);

impl TemporaryDirectory {
    fn new(label: &str) -> Self {
        let ordinal = TEMPORARY_DIRECTORY_ORDINAL.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "cascadia-rival-{label}-{}-{ordinal}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create isolated CLI test directory");
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

fn run_verifier(path: &Path) -> Output {
    let pair = Sha256Digest::of_bytes(b"expected-pair");
    let parent = Sha256Digest::of_bytes(b"expected-parent");
    Command::new(env!("CARGO_BIN_EXE_rival-contract"))
        .arg("verify-terminal-pair")
        .arg(path)
        .arg(pair.to_string())
        .arg(parent.to_string())
        .output()
        .expect("run rival-contract verifier")
}

#[test]
fn verifier_rejects_oversized_ledger_before_reading_and_prints_no_receipt() {
    let directory = TemporaryDirectory::new("oversized-ledger");
    let ledger = directory.path().join("oversized.json");
    File::create(&ledger)
        .expect("create sparse oversized ledger")
        .set_len(MAX_TERMINAL_PAIR_LEDGER_BYTES + 1)
        .expect("set sparse oversized ledger length");

    let output = run_verifier(&ledger);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty(), "failure must not emit a receipt");
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 verifier error");
    assert!(stderr.contains("hard maximum is 67108864"), "{stderr}");
}

#[cfg(unix)]
#[test]
fn verifier_rejects_symlink_ledger_and_prints_no_receipt() {
    use std::os::unix::fs::symlink;

    let directory = TemporaryDirectory::new("symlink-ledger");
    let target = directory.path().join("target.json");
    fs::write(&target, b"{}").expect("write symlink target");
    let ledger = directory.path().join("ledger.json");
    symlink(&target, &ledger).expect("create ledger symlink");

    let output = run_verifier(&ledger);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty(), "failure must not emit a receipt");
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 verifier error");
    assert!(stderr.contains("must not be a symbolic link"), "{stderr}");
}

#[test]
fn verifier_rejects_non_regular_ledger_and_prints_no_receipt() {
    let directory = TemporaryDirectory::new("directory-ledger");
    let output = run_verifier(directory.path());
    assert!(!output.status.success());
    assert!(output.stdout.is_empty(), "failure must not emit a receipt");
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 verifier error");
    assert!(stderr.contains("must be a regular file"), "{stderr}");
}
