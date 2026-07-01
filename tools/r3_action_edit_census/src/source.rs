use std::{
    env,
    ffi::OsStr,
    fs::{self, File},
    io::{BufReader, Read},
    path::{Component, Path, PathBuf},
};

use blake3::Hasher;
use serde::{Deserialize, Serialize};

use crate::{R3Error, Result};

pub const SOURCE_IDENTITY_CONTRACT: &str =
    "r3-source-bundle-v1: sorted relative path + byte length + file BLAKE3";

const SOURCE_ROOTS: &[&str] = &[
    "CASCADIA_V2_GOAL.txt",
    "Cargo.toml",
    "Cargo.lock",
    "crates/cascadia-data/Cargo.toml",
    "crates/cascadia-data/src",
    "crates/cascadia-game/Cargo.toml",
    "crates/cascadia-game/src",
    "crates/cascadia-provenance/Cargo.toml",
    "crates/cascadia-provenance/src",
    "crates/cascadia-sim/Cargo.toml",
    "crates/cascadia-sim/src",
    "crates/cascadia-r2/Cargo.toml",
    "crates/cascadia-r2/src/codec.rs",
    "crates/cascadia-r2/src/model.rs",
    "tools/r3_action_edit_census/Cargo.toml",
    "tools/r3_action_edit_census/Cargo.lock",
    "tools/r3_action_edit_census/README.md",
    "tools/r3_action_edit_census/r2_public_adapter/Cargo.toml",
    "tools/r3_action_edit_census/r2_public_adapter/src",
    "tools/r3_action_edit_census/src",
    "tools/r3_action_edit_census/tests",
    "tools/r3_action_edit_campaign.py",
    "tools/test_r3_action_edit_campaign.py",
    "tools/cluster_artifact_collect.py",
    "tools/cluster_artifact_fanout.py",
    "tools/cluster_research_queue.py",
    "tools/rust_experiment_bundle.py",
    "docs/v2/decisions/0148-r3-exact-action-local-patch-global-edit-foundation.md",
    "docs/v2/reports/r3-action-edit-foundation-v1-invalid-smoke-1.md",
    "docs/v2/reports/r3-action-edit-foundation-v1-preregistration.md",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceFileIdentity {
    pub file: String,
    pub bytes: u64,
    pub blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceIdentity {
    pub contract: String,
    pub source_bundle_blake3: String,
    pub files: Vec<SourceFileIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeIdentity {
    pub source: SourceIdentity,
    pub executable_blake3: String,
}

pub fn source_bundle_roots() -> &'static [&'static str] {
    SOURCE_ROOTS
}

pub fn capture_source_identity() -> Result<SourceIdentity> {
    capture_source_identity_at(&repository_root()?)
}

pub fn capture_source_identity_at(repository: &Path) -> Result<SourceIdentity> {
    if !repository.join("CASCADIA_V2_GOAL.txt").is_file() {
        return Err(invalid(format!(
            "R3 source root has no CASCADIA_V2_GOAL.txt marker: {}",
            repository.display()
        )));
    }
    let mut paths = Vec::new();
    for relative in SOURCE_ROOTS {
        collect_source_files(&repository.join(relative), &mut paths)?;
    }
    paths.sort();
    paths.dedup();

    let mut files = Vec::with_capacity(paths.len());
    for path in paths {
        let relative = path
            .strip_prefix(repository)?
            .to_string_lossy()
            .replace('\\', "/");
        let metadata = fs::metadata(&path)?;
        files.push(SourceFileIdentity {
            file: relative,
            bytes: metadata.len(),
            blake3: hash_file(&path)?,
        });
    }
    let source_bundle_blake3 = source_bundle_digest(&files)?;
    Ok(SourceIdentity {
        contract: SOURCE_IDENTITY_CONTRACT.to_owned(),
        source_bundle_blake3,
        files,
    })
}

pub fn capture_runtime_identity() -> Result<RuntimeIdentity> {
    Ok(RuntimeIdentity {
        source: capture_source_identity()?,
        executable_blake3: hash_file(&env::current_exe()?)?,
    })
}

pub fn capture_runtime_identity_checked(
    expected_source_bundle_blake3: Option<&str>,
    expected_executable_blake3: Option<&str>,
) -> Result<RuntimeIdentity> {
    let identity = capture_runtime_identity()?;
    if let Some(expected) = expected_source_bundle_blake3 {
        validate_blake3("expected source bundle", expected)?;
        if identity.source.source_bundle_blake3 != expected {
            return Err(invalid(format!(
                "source bundle BLAKE3 mismatch: expected {expected}, found {}",
                identity.source.source_bundle_blake3
            )));
        }
    }
    if let Some(expected) = expected_executable_blake3 {
        validate_blake3("expected executable", expected)?;
        if identity.executable_blake3 != expected {
            return Err(invalid(format!(
                "executable BLAKE3 mismatch: expected {expected}, found {}",
                identity.executable_blake3
            )));
        }
    }
    Ok(identity)
}

pub(crate) fn validate_runtime_identity(identity: &RuntimeIdentity) -> Result<()> {
    validate_source_identity(&identity.source)?;
    validate_blake3("executable", &identity.executable_blake3)
}

pub(crate) fn validate_source_identity(identity: &SourceIdentity) -> Result<()> {
    if identity.contract != SOURCE_IDENTITY_CONTRACT {
        return Err(invalid("source identity contract drifted"));
    }
    if identity.files.is_empty() {
        return Err(invalid("source identity contains no files"));
    }
    for pair in identity.files.windows(2) {
        if pair[0].file >= pair[1].file {
            return Err(invalid(
                "source identity files are duplicated or noncanonical",
            ));
        }
    }
    for file in &identity.files {
        validate_relative_source_path(&file.file)?;
        validate_blake3("source file", &file.blake3)?;
    }
    validate_blake3("source bundle", &identity.source_bundle_blake3)?;
    if source_bundle_digest(&identity.files)? != identity.source_bundle_blake3 {
        return Err(invalid(
            "source bundle BLAKE3 does not match its file table",
        ));
    }
    Ok(())
}

pub(crate) fn validate_blake3(label: &str, value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid(format!("{label} BLAKE3 is not lowercase hex")));
    }
    Ok(())
}

pub(crate) fn hash_file(path: &Path) -> Result<String> {
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

fn source_bundle_digest(files: &[SourceFileIdentity]) -> Result<String> {
    let mut hasher = Hasher::new();
    hasher.update(b"r3-action-edit-foundation-v1/source-bundle/v1");
    for file in files {
        update_framed(&mut hasher, file.file.as_bytes());
        hasher.update(&file.bytes.to_le_bytes());
        update_framed(&mut hasher, file.blake3.as_bytes());
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn repository_root() -> Result<PathBuf> {
    if let Some(root) = env::var_os("R3_SOURCE_ROOT") {
        return Ok(PathBuf::from(root));
    }
    if let Ok(current) = env::current_dir()
        && let Some(root) = marked_ancestor(&current)
    {
        return Ok(root);
    }
    if let Ok(executable) = env::current_exe()
        && let Some(root) = marked_ancestor(&executable)
    {
        return Ok(root);
    }
    marked_ancestor(Path::new(env!("CARGO_MANIFEST_DIR")))
        .ok_or_else(|| invalid("R3 source root not found"))
}

fn marked_ancestor(path: &Path) -> Option<PathBuf> {
    path.ancestors()
        .find(|candidate| candidate.join("CASCADIA_V2_GOAL.txt").is_file())
        .map(Path::to_path_buf)
}

fn collect_source_files(path: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if path.is_file() {
        files.push(path.to_owned());
        return Ok(());
    }
    if !path.is_dir() {
        return Err(invalid(format!(
            "source bundle path does not exist: {}",
            path.display()
        )));
    }
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let child = entry.path();
        if child.components().any(|component| {
            matches!(
                component.as_os_str(),
                value if value == OsStr::new("target")
                    || value == OsStr::new(".git")
                    || value == OsStr::new("__pycache__")
            )
        }) {
            continue;
        }
        collect_source_files(&child, files)?;
    }
    Ok(())
}

fn validate_relative_source_path(value: &str) -> Result<()> {
    let path = Path::new(value);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(invalid(format!(
            "source identity contains a non-relative path: {value}"
        )));
    }
    Ok(())
}

fn update_framed(hasher: &mut Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn invalid(message: impl Into<String>) -> R3Error {
    R3Error::Invariant(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_identity_is_stable_and_git_independent() {
        let left = capture_source_identity().unwrap();
        let right = capture_source_identity().unwrap();
        assert_eq!(left, right);
        assert!(left.files.iter().all(|file| !file.file.contains(".git")));
        validate_source_identity(&left).unwrap();
    }

    #[test]
    fn checked_runtime_identity_fails_closed_on_expected_hash_drift() {
        let actual = capture_runtime_identity().unwrap();
        let checked = capture_runtime_identity_checked(
            Some(&actual.source.source_bundle_blake3),
            Some(&actual.executable_blake3),
        )
        .unwrap();
        assert_eq!(checked, actual);

        let source_error =
            capture_runtime_identity_checked(Some(&"0".repeat(64)), None).unwrap_err();
        assert!(
            source_error
                .to_string()
                .contains("source bundle BLAKE3 mismatch")
        );

        let executable_error =
            capture_runtime_identity_checked(None, Some(&"0".repeat(64))).unwrap_err();
        assert!(
            executable_error
                .to_string()
                .contains("executable BLAKE3 mismatch")
        );
    }

    #[test]
    fn checked_runtime_identity_rejects_malformed_expected_hashes() {
        let error = capture_runtime_identity_checked(Some("not-a-hash"), None).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("expected source bundle BLAKE3 is not lowercase hex")
        );
    }
}
