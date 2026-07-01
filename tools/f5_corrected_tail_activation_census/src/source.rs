use std::{
    env,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
};

use blake3::Hasher;
use serde::{Deserialize, Serialize};

use crate::{Result, hash_file, invalid};

const SOURCE_ROOTS: &[&str] = &[
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
    pub git_revision: String,
    pub source_bundle_blake3: String,
    pub extractor_source_blake3: String,
    pub files: Vec<SourceFileIdentity>,
}

pub fn capture_source_identity() -> Result<SourceIdentity> {
    let repository = repository_root()?;
    let mut paths = Vec::new();
    for relative in SOURCE_ROOTS {
        collect_source_files(&repository.join(relative), &mut paths)?;
    }
    paths.sort();
    paths.dedup();

    let mut files = Vec::with_capacity(paths.len());
    let mut bundle = Hasher::new();
    bundle.update(b"corrected-mid-tail-activation-census-v1/source-bundle/v1");
    for path in paths {
        let relative = path
            .strip_prefix(&repository)?
            .to_string_lossy()
            .replace('\\', "/");
        let metadata = fs::metadata(&path)?;
        let digest = hash_file(&path)?;
        bundle.update(&(relative.len() as u64).to_le_bytes());
        bundle.update(relative.as_bytes());
        bundle.update(&metadata.len().to_le_bytes());
        bundle.update(digest.as_bytes());
        files.push(SourceFileIdentity {
            file: relative,
            bytes: metadata.len(),
            blake3: digest,
        });
    }

    let extractor_source_blake3 = files
        .iter()
        .find(|file| file.file == "legacy/crates/cascadia-ai/src/nnue.rs")
        .map(|file| file.blake3.clone())
        .ok_or_else(|| invalid("corrected Rust extractor source is absent from source bundle"))?;

    Ok(SourceIdentity {
        contract: "sorted relative path + byte length + file BLAKE3, domain separated".to_owned(),
        git_revision: env!("F5_SOURCE_GIT_REVISION").to_owned(),
        source_bundle_blake3: bundle.finalize().to_hex().to_string(),
        extractor_source_blake3,
        files,
    })
}

fn repository_root() -> Result<PathBuf> {
    let current = env::current_dir()?;
    if let Some(root) = current
        .ancestors()
        .find(|path| path.join("CASCADIA_V2_GOAL.txt").is_file())
    {
        return Ok(root.to_owned());
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest
        .ancestors()
        .find(|path| path.join("CASCADIA_V2_GOAL.txt").is_file())
        .map(Path::to_path_buf)
        .ok_or_else(|| invalid("repository root not found"))
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
                    || value == OsStr::new("__pycache__")
                    || value == OsStr::new(".git")
            )
        }) {
            continue;
        }
        collect_source_files(&child, files)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_identity_is_stable_and_covers_the_actual_extractor() {
        let left = capture_source_identity().unwrap();
        let right = capture_source_identity().unwrap();
        assert_eq!(left, right);
        assert!(
            left.files
                .iter()
                .any(|file| file.file == "legacy/crates/cascadia-ai/src/nnue.rs")
        );
        assert!(
            left.files
                .iter()
                .all(|file| !file.file.contains("/target/"))
        );
        assert_eq!(left.git_revision, env!("F5_SOURCE_GIT_REVISION"));
        assert_ne!(left.git_revision, "unavailable");
    }
}
