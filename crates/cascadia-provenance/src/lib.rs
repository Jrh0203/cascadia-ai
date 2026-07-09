//! Reproducible source and executable identity for Cascadia artifacts.

use std::{
    env,
    ffi::OsStr,
    fs::{self, File},
    io::{self, Read},
    path::{Path, PathBuf},
    process::Command,
};

use blake3::Hasher;
use serde::{Deserialize, Serialize};

const SOURCE_ROOTS: &[&str] = &[
    "Cargo.toml",
    "Cargo.lock",
    "Makefile",
    "pyproject.toml",
    "uv.lock",
    "cascadiav3",
    "infra",
    "python/cascadia_cluster",
    "tools",
    "crates/cascadia-game",
    "crates/cascadia-sim",
    "crates/cascadia-data",
    "crates/cascadia-model",
    "crates/cascadia-eval",
    "crates/cascadia-search",
    "crates/cascadia-provenance",
];

/// Recursive fallback exclusions for source archives that are no longer in a
/// Git worktree. The normal Git path uses repository ignore rules directly.
const GENERATED_DIRECTORY_NAMES: &[&str] = &[
    ".git",
    ".venv",
    "__pycache__",
    "checkpoints",
    "logs",
    "node_modules",
    "reports",
    "target",
    "venv",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceProvenance {
    pub git_revision: String,
    pub git_dirty: bool,
    pub git_status_blake3: String,
    #[serde(alias = "v2_source_blake3")]
    pub v2_source_blake3: String,
}

pub fn source_provenance() -> io::Result<SourceProvenance> {
    let repository = repository_root()?;
    let git_status = command_output_in(&repository, "git", &["status", "--porcelain=v1"])
        .unwrap_or_else(|| "unavailable".to_owned());
    Ok(SourceProvenance {
        git_revision: command_output_in(&repository, "git", &["rev-parse", "HEAD"])
            .unwrap_or_else(|| "unavailable".to_owned()),
        git_dirty: git_status == "unavailable" || !git_status.is_empty(),
        git_status_blake3: blake3::hash(git_status.as_bytes()).to_hex().to_string(),
        v2_source_blake3: source_digest(&repository)?,
    })
}

pub fn repository_root() -> io::Result<PathBuf> {
    let current = env::current_dir()?;
    if let Some(root) = current
        .ancestors()
        .find(|path| path.join("Cargo.toml").is_file() && path.join("cascadiav3").is_dir())
    {
        return Ok(root.to_owned());
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest
        .ancestors()
        .find(|path| path.join("Cargo.toml").is_file() && path.join("cascadiav3").is_dir())
        .map(Path::to_path_buf)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "repository root not found"))
}

pub fn checksum_file(path: &Path) -> io::Result<String> {
    let mut hasher = Hasher::new();
    update_file_hash(&mut hasher, path)?;
    Ok(hasher.finalize().to_hex().to_string())
}

fn source_digest(repository: &Path) -> io::Result<String> {
    // Hash tracked files plus untracked, non-ignored source files. Generated
    // checkpoints/reports/build outputs can be enormous and mutable during a
    // run; including them made the purported source digest slow and unstable.
    let mut files = match git_visible_source_files(repository) {
        Some(files) => files,
        None => {
            let mut files = Vec::new();
            for relative in SOURCE_ROOTS {
                collect_files(&repository.join(relative), &mut files)?;
            }
            files
        }
    };
    files.sort_unstable();
    let mut digest = Hasher::new();
    for path in files {
        let relative = path
            .strip_prefix(repository)
            .map_err(io::Error::other)?
            .to_string_lossy()
            .replace('\\', "/");
        let relative = relative.as_bytes();
        digest.update(&(relative.len() as u32).to_le_bytes());
        digest.update(relative);
        update_file_hash(&mut digest, &path)?;
    }
    Ok(digest.finalize().to_hex().to_string())
}

fn git_visible_source_files(repository: &Path) -> Option<Vec<PathBuf>> {
    let output = Command::new("git")
        .current_dir(repository)
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
        ])
        .args(SOURCE_ROOTS)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let paths = String::from_utf8(output.stdout).ok()?;
    Some(
        paths
            .split_terminator('\0')
            .map(|relative| repository.join(relative))
            // A tracked deletion belongs in the Git-status digest but has no
            // file bytes to include in the source-content digest.
            .filter(|path| path.is_file())
            .collect(),
    )
}

fn collect_files(path: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
    if path.is_file() {
        files.push(path.to_owned());
        return Ok(());
    }
    if !path.is_dir() {
        return Ok(());
    }
    if path
        .file_name()
        .and_then(OsStr::to_str)
        .is_some_and(|name| GENERATED_DIRECTORY_NAMES.contains(&name))
    {
        return Ok(());
    }
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let child = entry.path();
        collect_files(&child, files)?;
    }
    Ok(())
}

fn update_file_hash(hasher: &mut Hasher, path: &Path) -> io::Result<()> {
    let mut reader = File::open(path)?;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(())
}

fn command_output_in(directory: &Path, program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program)
        .current_dir(directory)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_digest_is_stable_for_identical_contents() {
        let left = source_provenance().unwrap();
        let right = source_provenance().unwrap();
        assert_eq!(left.v2_source_blake3, right.v2_source_blake3);
        assert_eq!(left.git_status_blake3, right.git_status_blake3);
    }

    #[test]
    fn ignored_generated_outputs_do_not_change_source_digest() {
        let repository = repository_root().unwrap();
        let ignored = repository
            .join("cascadiav3/real-root-exporter/target")
            .join(format!("provenance-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&ignored);
        let before = source_digest(&repository).unwrap();
        fs::create_dir_all(&ignored).unwrap();
        fs::write(ignored.join("mutable-checkpoint.bin"), b"generated output").unwrap();
        let after = source_digest(&repository).unwrap();
        fs::remove_dir_all(&ignored).unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn archive_fallback_skips_generated_directories() {
        let root = env::temp_dir().join(format!("cascadia-provenance-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).unwrap();
        fs::create_dir_all(root.join("target/release")).unwrap();
        fs::create_dir_all(root.join("reports")).unwrap();
        fs::write(root.join("src/lib.rs"), b"source").unwrap();
        fs::write(root.join("target/release/binary"), b"generated").unwrap();
        fs::write(root.join("reports/result.json"), b"generated").unwrap();

        let mut files = Vec::new();
        collect_files(&root, &mut files).unwrap();
        assert_eq!(files, vec![root.join("src/lib.rs")]);
        fs::remove_dir_all(&root).unwrap();
    }
}
