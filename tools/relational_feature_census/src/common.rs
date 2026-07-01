use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use cascadia_game::{GameConfig, GameSeed, GameState};
use rayon::prelude::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{Result, invalid};

pub const REPORT_SCHEMA_VERSION: u16 = 1;
pub const POSITIONS_PER_GAME: u64 = 80;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributionSummary {
    pub count: u64,
    pub minimum: u64,
    pub mean_milli: u64,
    pub median: u64,
    pub p90: u64,
    pub p99: u64,
    pub maximum: u64,
}

impl DistributionSummary {
    pub fn from_values(mut values: Vec<u64>) -> Result<Self> {
        if values.is_empty() {
            return Err(invalid("cannot summarize an empty distribution"));
        }
        values.sort_unstable();
        let count = values.len() as u64;
        let sum = values.iter().map(|value| u128::from(*value)).sum::<u128>();
        Ok(Self {
            count,
            minimum: values[0],
            mean_milli: u64::try_from(
                sum.checked_mul(1_000)
                    .ok_or_else(|| invalid("distribution mean overflowed"))?
                    / u128::from(count),
            )?,
            median: percentile(&values, 50),
            p90: percentile(&values, 90),
            p99: percentile(&values, 99),
            maximum: *values.last().expect("nonempty values"),
        })
    }
}

fn percentile(values: &[u64], percentile: usize) -> u64 {
    let index = (values.len() - 1) * percentile / 100;
    values[index]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExperimentLane {
    R5Quotient,
    R6Incremental,
    S3ComponentMotif,
    S5Derivatives,
    S6Topology,
}

impl ExperimentLane {
    pub const fn experiment_id(self) -> &'static str {
        match self {
            Self::R5Quotient => "r5-component-motif-quotient-foundation-v1",
            Self::R6Incremental => "r6-incremental-sparse-accumulator-foundation-v1",
            Self::S3ComponentMotif => "s3-component-motif-graph-foundation-v1",
            Self::S5Derivatives => "s5-opportunity-derivative-foundation-v1",
            Self::S6Topology => "s6-topological-spectral-foundation-v2",
        }
    }

    pub const fn protocol_id(self) -> &'static str {
        match self {
            Self::R5Quotient => "r5-exact-decoding-and-compactness-v1",
            Self::R6Incremental => "r6-apply-undo-parity-and-throughput-v1",
            Self::S3ComponentMotif => "s3-card-a-semantic-decoder-census-v1",
            Self::S5Derivatives => "s5-exact-counterfactual-derivative-census-v1",
            Self::S6Topology => "s6-exact-topology-activation-census-v2",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommonConfig {
    pub lane: ExperimentLane,
    pub first_seed: u64,
    pub games: u32,
    pub source_bundle_id: String,
    pub host: String,
    pub rayon_threads: usize,
}

impl CommonConfig {
    pub fn validate(&self) -> Result<()> {
        if self.games == 0 {
            return Err(invalid(
                "the experiment corpus must contain at least one game",
            ));
        }
        if self.source_bundle_id.len() != 64
            || !self
                .source_bundle_id
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(invalid(
                "source bundle ID must be a 64-character hexadecimal digest",
            ));
        }
        if self.host.trim().is_empty() {
            return Err(invalid("host identity must not be empty"));
        }
        if self.rayon_threads == 0 {
            return Err(invalid("rayon thread count must be positive"));
        }
        self.first_seed
            .checked_add(u64::from(self.games))
            .ok_or_else(|| invalid("seed range overflowed"))?;
        Ok(())
    }

    pub fn seeds(&self) -> Vec<u64> {
        (0..u64::from(self.games))
            .map(|offset| self.first_seed + offset)
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusSummary {
    pub first_seed: u64,
    pub games: u32,
    pub positions: u64,
    pub seeds_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionSummary {
    pub host: String,
    pub started_unix_ms: u64,
    pub completed_unix_ms: u64,
    pub elapsed_ms: u64,
    pub executable_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize", deserialize = "T: DeserializeOwned"))]
#[serde(deny_unknown_fields)]
pub struct ExperimentReport<T> {
    pub schema_version: u16,
    pub artifact_kind: String,
    pub experiment_id: String,
    pub protocol_id: String,
    pub source_bundle_id: String,
    pub config: CommonConfig,
    pub corpus: CorpusSummary,
    pub metrics: T,
    pub passed: bool,
    pub classification: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize", deserialize = "T: DeserializeOwned"))]
#[serde(deny_unknown_fields)]
pub struct ReportEnvelope<T> {
    pub scientific: ExperimentReport<T>,
    pub scientific_blake3: String,
    pub execution: ExecutionSummary,
}

impl<T: Serialize> ReportEnvelope<T> {
    pub fn validate_hash(&self) -> Result<()> {
        let expected = canonical_blake3(&self.scientific)?;
        if expected != self.scientific_blake3 {
            return Err(invalid("scientific report BLAKE3 mismatch"));
        }
        Ok(())
    }
}

pub(crate) fn run_games<T: Send>(
    config: &CommonConfig,
    run_seed: impl Fn(u64, GameState) -> Result<T> + Sync,
) -> Result<Vec<(u64, T)>> {
    config.validate()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.rayon_threads)
        .build()
        .map_err(|error| invalid(format!("cannot build Rayon pool: {error}")))?;
    let mut results = pool.install(|| {
        config
            .seeds()
            .into_par_iter()
            .map(|seed| {
                let game =
                    GameState::new(GameConfig::research_aaaaa(4)?, GameSeed::from_u64(seed))?;
                Ok((seed, run_seed(seed, game)?))
            })
            .collect::<Result<Vec<_>>>()
    })?;
    results.sort_unstable_by_key(|(seed, _)| *seed);
    Ok(results)
}

pub(crate) fn corpus_summary(config: &CommonConfig) -> Result<CorpusSummary> {
    let seeds = config.seeds();
    Ok(CorpusSummary {
        first_seed: config.first_seed,
        games: config.games,
        positions: u64::from(config.games)
            .checked_mul(POSITIONS_PER_GAME)
            .ok_or_else(|| invalid("corpus position count overflowed"))?,
        seeds_blake3: canonical_blake3(&seeds)?,
    })
}

pub(crate) fn envelope<T: Serialize>(
    config: CommonConfig,
    metrics: T,
    passed: bool,
    classification: impl Into<String>,
    started_unix_ms: u64,
) -> Result<ReportEnvelope<T>> {
    let lane = config.lane;
    let scientific = ExperimentReport {
        schema_version: REPORT_SCHEMA_VERSION,
        artifact_kind: "relational_feature_census_report".to_owned(),
        experiment_id: lane.experiment_id().to_owned(),
        protocol_id: lane.protocol_id().to_owned(),
        source_bundle_id: config.source_bundle_id.clone(),
        corpus: corpus_summary(&config)?,
        config: config.clone(),
        metrics,
        passed,
        classification: classification.into(),
    };
    let completed_unix_ms = unix_ms()?;
    let result = ReportEnvelope {
        scientific_blake3: canonical_blake3(&scientific)?,
        execution: ExecutionSummary {
            host: config.host,
            started_unix_ms,
            completed_unix_ms,
            elapsed_ms: completed_unix_ms.saturating_sub(started_unix_ms),
            executable_blake3: executable_blake3()?,
        },
        scientific,
    };
    result.validate_hash()?;
    Ok(result)
}

pub(crate) fn canonical_blake3(value: &impl Serialize) -> Result<String> {
    Ok(blake3::hash(&serde_json::to_vec(value)?)
        .to_hex()
        .to_string())
}

pub(crate) fn unix_ms() -> Result<u64> {
    Ok(u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| invalid(format!("system clock precedes Unix epoch: {error}")))?
            .as_millis(),
    )?)
}

fn executable_blake3() -> Result<String> {
    let bytes = fs::read(std::env::current_exe()?)?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

pub fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("json")
    ));
    {
        let mut writer = BufWriter::new(File::create(&temporary)?);
        serde_json::to_writer_pretty(&mut writer, value)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }
    fs::rename(temporary, path)?;
    Ok(())
}

pub(crate) fn deterministic_index(seed: u64, turn: u16, len: usize, domain: &[u8]) -> usize {
    debug_assert!(len > 0);
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"cascadia-relational-feature-census-v1");
    hasher.update(domain);
    hasher.update(&seed.to_le_bytes());
    hasher.update(&turn.to_le_bytes());
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&hasher.finalize().as_bytes()[..8]);
    (u64::from_le_bytes(bytes) % len as u64) as usize
}
