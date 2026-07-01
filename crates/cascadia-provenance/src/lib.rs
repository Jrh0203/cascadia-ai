//! Reproducible source and executable identity for Cascadia v2 artifacts.

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
    "python/cascadia_mlx",
    "apps/web/src",
    "crates/cascadia-game",
    "crates/cascadia-sim",
    "crates/cascadia-data",
    "crates/cascadia-model",
    "crates/cascadia-eval",
    "crates/cascadia-search",
    "crates/cascadia-api",
    "crates/cascadia-cli-v2",
    "crates/cascadia-differential",
    "crates/cascadia-provenance",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceProvenance {
    pub git_revision: String,
    pub git_dirty: bool,
    pub git_status_blake3: String,
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
        .find(|path| path.join("docs/archive/v2/CASCADIA_V2_GOAL.txt").is_file())
    {
        return Ok(root.to_owned());
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest
        .ancestors()
        .find(|path| path.join("docs/archive/v2/CASCADIA_V2_GOAL.txt").is_file())
        .map(Path::to_path_buf)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "repository root not found"))
}

pub fn checksum_file(path: &Path) -> io::Result<String> {
    let mut hasher = Hasher::new();
    update_file_hash(&mut hasher, path)?;
    Ok(hasher.finalize().to_hex().to_string())
}

fn source_digest(repository: &Path) -> io::Result<String> {
    let mut files = Vec::new();
    for relative in SOURCE_ROOTS {
        collect_files(&repository.join(relative), &mut files)?;
    }
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

fn collect_files(path: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
    if path.is_file() {
        files.push(path.to_owned());
        return Ok(());
    }
    if !path.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let child = entry.path();
        if child
            .components()
            .any(|component| component.as_os_str() == OsStr::new("__pycache__"))
        {
            continue;
        }
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
}
