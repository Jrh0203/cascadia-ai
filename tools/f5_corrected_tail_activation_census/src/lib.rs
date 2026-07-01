//! Source-frozen corrected-schema activation census for F5.
//!
//! The corpus stores only public state: boards, market, turn metadata, and
//! public supply marginals. Census replay rebuilds the legacy Rust extractor
//! inputs and calls `extract_features_with_bag` directly. Feature semantics are
//! never reproduced in Python or inferred from documentation.

mod census;
mod corpus;
mod model;
mod source;
mod strict_json;

pub use census::{
    AggregateReport, ShardReport, aggregate_reports, census_shard, read_aggregate_report,
    read_shard_report, verify_reports_byte_identical,
};
pub use corpus::{
    CorpusManifest, GenerateShardConfig, generate_shard, read_manifest, validate_corpus_manifest,
};
pub use model::{
    EXPERIMENT_ID, FEATURE_SCHEMA, PRODUCTION_FIRST_GAME_INDEX, PRODUCTION_SHARD_COUNT,
    PRODUCTION_TOTAL_GAMES, PublicStateRecord, corrected_tail_indices,
    generate_reachable_overflow_witness, replay_record,
};
pub use source::{SourceIdentity, capture_source_identity};

use std::path::Path;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum F5Error {
    #[error("invalid corrected-tail census input: {0}")]
    Invalid(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    StripPrefix(#[from] std::path::StripPrefixError),
    #[error(transparent)]
    Rayon(#[from] rayon::ThreadPoolBuildError),
    #[error(transparent)]
    IntConversion(#[from] std::num::TryFromIntError),
}

pub type Result<T> = std::result::Result<T, F5Error>;

pub(crate) fn invalid(message: impl Into<String>) -> F5Error {
    F5Error::Invalid(message.into())
}

pub(crate) fn hash_file(path: &Path) -> Result<String> {
    use std::{
        fs::File,
        io::{BufReader, Read},
    };

    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = blake3::Hasher::new();
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

pub(crate) fn canonical_blake3<T: serde::Serialize>(value: &T) -> Result<String> {
    Ok(blake3::hash(&serde_json::to_vec(value)?)
        .to_hex()
        .to_string())
}

pub(crate) fn update_framed(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

pub(crate) fn write_json_atomic(path: &Path, value: &impl serde::Serialize) -> Result<()> {
    use std::{
        fs::{self, File},
        io::{BufWriter, Write},
    };

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.tmp");
    let mut writer = BufWriter::new(File::create(&temporary)?);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}
