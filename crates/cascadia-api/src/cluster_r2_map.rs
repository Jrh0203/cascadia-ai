//! Compact R2-MAP campaign status mirror for the cluster dashboard.
//!
//! The reader opens exactly one configured JSON file. It never enumerates the
//! campaign root, datasets, checkpoints, receipts, or benchmark artifacts.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs, io,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

pub const R2_MAP_STATUS_SCHEMA_VERSION: u16 = 1;
pub const R2_MAP_STATUS_SCHEMA_ID: &str = "cascadia.r2-map.dashboard-status.v1";
pub const V3_STATUS_SCHEMA_ID: &str = "cascadia.v3.dashboard-status.v1";
const RESPONSE_SCHEMA_VERSION: u16 = 1;
const MAX_STATUS_BYTES: u64 = 1_048_576;
const MAX_SERVING_PROJECTION_BYTES: u64 = 65_536;
const MAX_LOSS_SAMPLES: usize = 512;
const MAX_HOST_DETAIL_BYTES: usize = 512;
const MAX_CLOCK_SKEW_MS: u64 = 60_000;
const SERVING_PROJECTION_SCHEMA_ID: &str = "cascadia.r2-map.dashboard-serving-projection.v2";
const CANONICAL_STATUS_HOST: &str = "john1";
const CANONICAL_STATUS_PATH: &str =
    "/Users/johnherrick/cascadia-bench/r2-map-v1/control/dashboard-status.json";
const V3_CANONICAL_STATUS_PATH: &str =
    "/Users/johnherrick/cascadia-bench/v3-nnue/control/dashboard-status.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum StatusCondition {
    Unconfigured,
    Fresh,
    Stale,
    Invalid,
}

#[derive(Debug, Clone, Serialize)]
pub struct CampaignStatusResponse {
    pub schema_version: u16,
    pub configured: bool,
    pub condition: StatusCondition,
    pub source_path: PathBuf,
    pub observed_unix_ms: u64,
    pub updated_unix_ms: Option<u64>,
    pub age_seconds: Option<f64>,
    pub stale_after_seconds: Option<u64>,
    pub status: Option<CampaignStatus>,
    pub error: Option<String>,
}

impl CampaignStatusResponse {
    fn unavailable(path: &Path, observed_unix_ms: u64) -> Self {
        Self {
            schema_version: RESPONSE_SCHEMA_VERSION,
            configured: false,
            condition: StatusCondition::Unconfigured,
            source_path: path.to_path_buf(),
            observed_unix_ms,
            updated_unix_ms: None,
            age_seconds: None,
            stale_after_seconds: None,
            status: None,
            error: None,
        }
    }

    fn invalid(path: &Path, observed_unix_ms: u64, error: impl Into<String>) -> Self {
        Self {
            schema_version: RESPONSE_SCHEMA_VERSION,
            configured: true,
            condition: StatusCondition::Invalid,
            source_path: path.to_path_buf(),
            observed_unix_ms,
            updated_unix_ms: None,
            age_seconds: None,
            stale_after_seconds: None,
            status: None,
            error: Some(error.into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignStatus {
    pub schema_version: u16,
    pub schema_id: String,
    pub campaign_id: String,
    pub updated_unix_ms: u64,
    pub stale_after_seconds: u64,
    pub phase: String,
    pub legal_next_transitions: Vec<String>,
    pub round_index: Option<u32>,
    pub models: CampaignModels,
    pub hosts: BTreeMap<String, CampaignHostStatus>,
    pub training: TrainingStatus,
    pub benchmark: BenchmarkStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignModels {
    pub incumbent: Option<ModelIdentity>,
    pub candidate: Option<ModelIdentity>,
    pub opponent_pool: Vec<ModelIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelIdentity {
    pub id: String,
    pub blake3: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CampaignHostIntent {
    Control,
    Generate,
    Validate,
    Train,
    Benchmark,
    CandidateGate,
    Idle,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignHostStatus {
    pub intent: CampaignHostIntent,
    pub detail: Option<String>,
    pub generation_games_completed: u64,
    pub generation_games_target: Option<u64>,
    pub generation_seed_prefix: Option<String>,
    pub benchmark_pairs_completed: u64,
    pub benchmark_pairs_total: Option<u64>,
    pub eta_seconds: Option<f64>,
    pub throughput_games_per_second: Option<f64>,
    pub rss_bytes: Option<u64>,
    pub swap_delta_bytes: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingStatus {
    pub active: bool,
    pub latest_verified_checkpoint: Option<ModelIdentity>,
    pub current_step: Option<u64>,
    pub total_steps: Option<u64>,
    pub eta_seconds: Option<f64>,
    pub examples_per_second: Option<f64>,
    pub loss_samples: Vec<LossSample>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LossSample {
    pub step: u64,
    pub train_total: f64,
    pub validation_total: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BenchmarkClassification {
    Pending,
    Promote,
    Reject,
    Inconclusive,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkStatus {
    pub active: bool,
    pub stage: Option<String>,
    pub pairs_completed: u64,
    pub pairs_total: Option<u64>,
    pub eta_seconds: Option<f64>,
    pub throughput_games_per_second: Option<f64>,
    pub peak_rss_bytes: Option<u64>,
    pub swap_delta_bytes: Option<i64>,
    pub focal: Option<FocalScoreStatus>,
    pub paired_delta: Option<PairedDeltaStatus>,
    pub classification: BenchmarkClassification,
    pub scheduler_work_items: Option<SchedulerWorkItemsStatus>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerWorkItemsStatus {
    pub completed: u64,
    pub total: u64,
    pub states: BTreeMap<String, u64>,
    pub retry_attempts: u64,
    pub utilization: Option<SchedulerUtilizationStatus>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerUtilizationStatus {
    pub sample_count: u64,
    pub observed_seconds: f64,
    pub cpu_capacity_min: f64,
    pub cpu_capacity_max: f64,
    pub cpu_allocated_mean: f64,
    pub cpu_allocated_peak: f64,
    pub cpu_utilization_mean: f64,
    pub cpu_utilization_peak: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FocalScoreStatus {
    pub base_total: DistributionSummary,
    pub animals: AnimalSummaries,
    pub habitat: HabitatSummaries,
    pub pinecones: PineconeSummaries,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributionSummary {
    pub mean: f64,
    pub p10: f64,
    pub p50: f64,
    pub p90: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnimalSummaries {
    pub aggregate: DistributionSummary,
    pub bear: DistributionSummary,
    pub elk: DistributionSummary,
    pub salmon: DistributionSummary,
    pub hawk: DistributionSummary,
    pub fox: DistributionSummary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HabitatSummaries {
    pub aggregate: DistributionSummary,
    pub mountain: DistributionSummary,
    pub forest: DistributionSummary,
    pub prairie: DistributionSummary,
    pub wetland: DistributionSummary,
    pub river: DistributionSummary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PineconeSummaries {
    pub earned: DistributionSummary,
    pub independent_draft_spend: DistributionSummary,
    pub paid_wipe_spend: DistributionSummary,
    pub total_spend: DistributionSummary,
    pub remaining: DistributionSummary,
    pub free_replacements: DistributionSummary,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairedDeltaStatus {
    pub mean: f64,
    pub confidence_95: [f64; 2],
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServingProjection {
    schema_version: u16,
    schema_id: String,
    canonical_host: String,
    canonical_path: PathBuf,
    canonical_blake3: String,
    canonical_updated_unix_ms: u64,
    fetched_unix_ms: u64,
    canonical_payload: String,
}

pub fn load(path: &Path, observed_unix_ms: u64) -> CampaignStatusResponse {
    match load_file(path) {
        Ok(status) => {
            if status.updated_unix_ms > observed_unix_ms.saturating_add(MAX_CLOCK_SKEW_MS) {
                return CampaignStatusResponse::invalid(
                    path,
                    observed_unix_ms,
                    "R2-MAP status mirror timestamp is more than 60 seconds in the future",
                );
            }
            let age_ms = observed_unix_ms.saturating_sub(status.updated_unix_ms);
            let stale_after_ms = status.stale_after_seconds.saturating_mul(1_000);
            let condition = if age_ms > stale_after_ms {
                StatusCondition::Stale
            } else {
                StatusCondition::Fresh
            };
            CampaignStatusResponse {
                schema_version: RESPONSE_SCHEMA_VERSION,
                configured: true,
                condition,
                source_path: path.to_path_buf(),
                observed_unix_ms,
                updated_unix_ms: Some(status.updated_unix_ms),
                age_seconds: Some(age_ms as f64 / 1_000.0),
                stale_after_seconds: Some(status.stale_after_seconds),
                status: Some(status),
                error: None,
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            CampaignStatusResponse::unavailable(path, observed_unix_ms)
        }
        Err(error) => CampaignStatusResponse::invalid(path, observed_unix_ms, error.to_string()),
    }
}

fn load_file(path: &Path) -> io::Result<CampaignStatus> {
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "R2-MAP status mirror is not a regular file",
        ));
    }
    if metadata.len() > MAX_STATUS_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "R2-MAP status mirror is {} bytes; maximum is {MAX_STATUS_BYTES}",
                metadata.len()
            ),
        ));
    }
    let bytes = fs::read(path)?;
    let outer: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("R2-MAP status mirror JSON is invalid: {error}"),
        )
    })?;
    let schema_id = outer.get("schema_id").and_then(serde_json::Value::as_str);
    if schema_id == Some("cascadia.r2-map.dashboard-serving-projection.v1") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "retired external-SSD R2-MAP serving projections are rejected",
        ));
    }
    let status = if schema_id == Some(SERVING_PROJECTION_SCHEMA_ID) {
        if metadata.len() > MAX_SERVING_PROJECTION_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "R2-MAP serving projection is {} bytes; maximum is {MAX_SERVING_PROJECTION_BYTES}",
                    metadata.len()
                ),
            ));
        }
        decode_serving_projection(outer)?
    } else {
        serde_json::from_value::<CampaignStatus>(outer).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("R2-MAP status mirror JSON is invalid: {error}"),
            )
        })?
    };
    validate(status).map_err(|message| io::Error::new(io::ErrorKind::InvalidData, message))
}

fn decode_serving_projection(outer: serde_json::Value) -> io::Result<CampaignStatus> {
    let projection: ServingProjection = serde_json::from_value(outer).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("R2-MAP serving projection JSON is invalid: {error}"),
        )
    })?;
    if projection.schema_version != 1
        || projection.schema_id != SERVING_PROJECTION_SCHEMA_ID
        || projection.canonical_host != CANONICAL_STATUS_HOST
        || ![
            Path::new(CANONICAL_STATUS_PATH),
            Path::new(V3_CANONICAL_STATUS_PATH),
        ]
        .contains(&projection.canonical_path.as_path())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "R2-MAP serving projection identity is invalid",
        ));
    }
    if projection.fetched_unix_ms == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "R2-MAP serving projection fetch timestamp is invalid",
        ));
    }
    decode_projection_payload(
        &projection.canonical_blake3,
        projection.canonical_updated_unix_ms,
        &projection.canonical_payload,
    )
}

fn decode_projection_payload(
    canonical_blake3: &str,
    canonical_updated_unix_ms: u64,
    canonical_payload: &str,
) -> io::Result<CampaignStatus> {
    if canonical_blake3.len() != 64
        || !canonical_blake3
            .bytes()
            .all(|value| value.is_ascii_hexdigit())
        || blake3::hash(canonical_payload.as_bytes()).to_hex().as_str() != canonical_blake3
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "R2-MAP serving projection canonical hash does not match its payload",
        ));
    }
    let status: CampaignStatus = serde_json::from_str(canonical_payload).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("R2-MAP serving projection payload is invalid: {error}"),
        )
    })?;
    if status.updated_unix_ms != canonical_updated_unix_ms {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "R2-MAP serving projection timestamp disagrees with its canonical payload",
        ));
    }
    Ok(status)
}

fn validate(status: CampaignStatus) -> Result<CampaignStatus, String> {
    let is_v3 = status.schema_id == V3_STATUS_SCHEMA_ID;
    if status.schema_version != R2_MAP_STATUS_SCHEMA_VERSION
        || !(status.schema_id == R2_MAP_STATUS_SCHEMA_ID || is_v3)
    {
        return Err("unsupported campaign status mirror schema".to_owned());
    }
    require_nonempty(&status.campaign_id, "campaign id")?;
    require_nonempty(&status.phase, "phase")?;
    if !(5..=3_600).contains(&status.stale_after_seconds) {
        return Err("stale_after_seconds must be between 5 and 3600".to_owned());
    }
    require_unique_nonempty(&status.legal_next_transitions, "legal next transitions")?;
    validate_models(&status.models)?;

    let expected_hosts = if is_v3 {
        ["john1", "john2", "john3", "john4"]
            .into_iter()
            .collect::<BTreeSet<_>>()
    } else {
        ["john1", "john2", "john3"]
            .into_iter()
            .collect::<BTreeSet<_>>()
    };
    let actual_hosts = status
        .hosts
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if actual_hosts != expected_hosts {
        return Err(if is_v3 {
            "V3 hosts must contain exactly john1, john2, john3, and john4".to_owned()
        } else {
            "hosts must contain exactly john1, john2, and john3".to_owned()
        });
    }
    for (host, state) in &status.hosts {
        if state
            .detail
            .as_ref()
            .is_some_and(|detail| detail.trim().is_empty() || detail.len() > MAX_HOST_DETAIL_BYTES)
        {
            return Err(format!(
                "{host} detail must be nonempty and at most {MAX_HOST_DETAIL_BYTES} bytes"
            ));
        }
        if let (Some(target), completed) = (
            state.generation_games_target,
            state.generation_games_completed,
        ) && completed > target
        {
            return Err(format!("{host} generation progress exceeds target"));
        }
        if let Some(total) = state.benchmark_pairs_total
            && state.benchmark_pairs_completed > total
        {
            return Err(format!("{host} benchmark progress exceeds total"));
        }
        validate_optional_nonnegative(state.eta_seconds, &format!("{host} ETA"))?;
        validate_optional_nonnegative(
            state.throughput_games_per_second,
            &format!("{host} throughput"),
        )?;
    }
    validate_training(&status.training)?;
    validate_benchmark(&status.benchmark)?;
    Ok(status)
}

fn validate_models(models: &CampaignModels) -> Result<(), String> {
    let mut ids = BTreeSet::new();
    for model in models
        .incumbent
        .iter()
        .chain(models.candidate.iter())
        .chain(models.opponent_pool.iter())
    {
        validate_model(model)?;
        if !ids.insert(model.id.as_str()) {
            return Err(format!("model identity {} is duplicated", model.id));
        }
    }
    Ok(())
}

fn validate_model(model: &ModelIdentity) -> Result<(), String> {
    require_nonempty(&model.id, "model id")?;
    if let Some(digest) = &model.blake3
        && (digest.len() != 64 || !digest.bytes().all(|value| value.is_ascii_hexdigit()))
    {
        return Err(format!("model {} has an invalid BLAKE3 digest", model.id));
    }
    Ok(())
}

fn validate_training(training: &TrainingStatus) -> Result<(), String> {
    if let Some(checkpoint) = &training.latest_verified_checkpoint {
        validate_model(checkpoint)?;
    }
    if let (Some(current), Some(total)) = (training.current_step, training.total_steps)
        && current > total
    {
        return Err("training current step exceeds total steps".to_owned());
    }
    validate_optional_nonnegative(training.eta_seconds, "training ETA")?;
    validate_optional_nonnegative(training.examples_per_second, "training throughput")?;
    if training.loss_samples.len() > MAX_LOSS_SAMPLES {
        return Err(format!(
            "loss sample count exceeds compact mirror maximum {MAX_LOSS_SAMPLES}"
        ));
    }
    let mut previous_step = None;
    for sample in &training.loss_samples {
        if previous_step.is_some_and(|previous| sample.step <= previous) {
            return Err("loss sample steps must be strictly increasing".to_owned());
        }
        validate_nonnegative(sample.train_total, "training loss")?;
        validate_optional_nonnegative(sample.validation_total, "validation loss")?;
        previous_step = Some(sample.step);
    }
    Ok(())
}

fn validate_benchmark(benchmark: &BenchmarkStatus) -> Result<(), String> {
    if let Some(total) = benchmark.pairs_total
        && benchmark.pairs_completed > total
    {
        return Err("benchmark progress exceeds total".to_owned());
    }
    validate_optional_nonnegative(benchmark.eta_seconds, "benchmark ETA")?;
    validate_optional_nonnegative(
        benchmark.throughput_games_per_second,
        "benchmark throughput",
    )?;
    if let Some(focal) = &benchmark.focal {
        for (label, distribution) in focal_distributions(focal) {
            validate_distribution(distribution, label)?;
        }
    }
    if let Some(delta) = benchmark.paired_delta
        && (!delta.mean.is_finite()
            || delta.confidence_95.iter().any(|value| !value.is_finite())
            || delta.confidence_95[0] > delta.confidence_95[1])
    {
        return Err("paired delta is invalid".to_owned());
    }
    if let Some(items) = &benchmark.scheduler_work_items {
        let invalid_counts =
            items.completed > items.total || items.states.values().sum::<u64>() > items.total;
        let benchmark_coupling_required = benchmark.active || benchmark.pairs_total.is_some();
        let invalid_benchmark_coupling = benchmark_coupling_required
            && (items.completed != benchmark.pairs_completed
                || benchmark.pairs_total != Some(items.total));
        if invalid_counts || invalid_benchmark_coupling {
            return Err("benchmark scheduler work-item progress is inconsistent".to_owned());
        }
        if let Some(utilization) = items.utilization {
            let values = [
                utilization.observed_seconds,
                utilization.cpu_capacity_min,
                utilization.cpu_capacity_max,
                utilization.cpu_allocated_mean,
                utilization.cpu_allocated_peak,
                utilization.cpu_utilization_mean,
                utilization.cpu_utilization_peak,
            ];
            if utilization.sample_count == 0
                || values
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
                || utilization.cpu_capacity_min > utilization.cpu_capacity_max
                || utilization.cpu_allocated_mean > utilization.cpu_allocated_peak
                || utilization.cpu_allocated_peak > utilization.cpu_capacity_max
                || utilization.cpu_utilization_mean > utilization.cpu_utilization_peak
                || utilization.cpu_utilization_peak > 1.0
            {
                return Err("benchmark scheduler utilization is invalid".to_owned());
            }
        }
    }
    Ok(())
}

fn focal_distributions(focal: &FocalScoreStatus) -> Vec<(&'static str, DistributionSummary)> {
    vec![
        ("base total", focal.base_total),
        ("wildlife aggregate", focal.animals.aggregate),
        ("bear", focal.animals.bear),
        ("elk", focal.animals.elk),
        ("salmon", focal.animals.salmon),
        ("hawk", focal.animals.hawk),
        ("fox", focal.animals.fox),
        ("habitat aggregate", focal.habitat.aggregate),
        ("mountain", focal.habitat.mountain),
        ("forest", focal.habitat.forest),
        ("prairie", focal.habitat.prairie),
        ("wetland", focal.habitat.wetland),
        ("river", focal.habitat.river),
        ("pinecones earned", focal.pinecones.earned),
        (
            "independent draft spend",
            focal.pinecones.independent_draft_spend,
        ),
        ("paid wipe spend", focal.pinecones.paid_wipe_spend),
        ("total spend", focal.pinecones.total_spend),
        ("pinecones remaining", focal.pinecones.remaining),
        ("free replacements", focal.pinecones.free_replacements),
    ]
}

fn validate_distribution(value: DistributionSummary, label: &str) -> Result<(), String> {
    if [value.mean, value.p10, value.p50, value.p90]
        .iter()
        .any(|value| !value.is_finite())
        || value.p10 > value.p50
        || value.p50 > value.p90
    {
        Err(format!("{label} distribution is invalid"))
    } else {
        Ok(())
    }
}

fn validate_nonnegative(value: f64, label: &str) -> Result<(), String> {
    if !value.is_finite() || value < 0.0 {
        Err(format!("{label} must be finite and nonnegative"))
    } else {
        Ok(())
    }
}

fn validate_optional_nonnegative(value: Option<f64>, label: &str) -> Result<(), String> {
    value.map_or(Ok(()), |value| validate_nonnegative(value, label))
}

fn require_nonempty(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{label} must be nonempty"))
    } else {
        Ok(())
    }
}

fn require_unique_nonempty(values: &[String], label: &str) -> Result<(), String> {
    let mut unique = BTreeSet::new();
    for value in values {
        require_nonempty(value, label)?;
        if !unique.insert(value) {
            return Err(format!("{label} must not contain duplicates"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn temporary_path(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "cascadia-r2-map-status-{label}-{}-{nonce}.json",
            std::process::id()
        ))
    }

    fn distribution(mean: f64) -> serde_json::Value {
        serde_json::json!({"mean": mean, "p10": mean - 1.0, "p50": mean, "p90": mean + 1.0})
    }

    fn golden_fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/r2_map")
            .join(name)
    }

    fn serving_projection(payload: &str, updated_unix_ms: u64) -> serde_json::Value {
        serde_json::json!({
            "schema_version": 1,
            "schema_id": SERVING_PROJECTION_SCHEMA_ID,
            "canonical_host": CANONICAL_STATUS_HOST,
            "canonical_path": CANONICAL_STATUS_PATH,
            "canonical_blake3": blake3::hash(payload.as_bytes()).to_hex().to_string(),
            "canonical_updated_unix_ms": updated_unix_ms,
            "fetched_unix_ms": updated_unix_ms + 500,
            "canonical_payload": payload,
        })
    }

    fn legacy_serving_projection(payload: &str, updated_unix_ms: u64) -> serde_json::Value {
        serde_json::json!({
            "schema_version": 1,
            "schema_id": "cascadia.r2-map.dashboard-serving-projection.v1",
            "canonical_path": "/Volumes/John_1/cascadia-cluster/r2-map-v1/control/dashboard-status.json",
            "canonical_blake3": blake3::hash(payload.as_bytes()).to_hex().to_string(),
            "canonical_updated_unix_ms": updated_unix_ms,
            "canonical_payload": payload,
        })
    }

    fn valid_status(updated_unix_ms: u64) -> serde_json::Value {
        let animals = serde_json::json!({
            "aggregate": distribution(58.0), "bear": distribution(10.0),
            "elk": distribution(11.0), "salmon": distribution(12.0),
            "hawk": distribution(12.0), "fox": distribution(13.0)
        });
        let habitat = serde_json::json!({
            "aggregate": distribution(30.0), "mountain": distribution(6.0),
            "forest": distribution(6.0), "prairie": distribution(6.0),
            "wetland": distribution(6.0), "river": distribution(6.0)
        });
        let pinecones = serde_json::json!({
            "earned": distribution(7.0), "independent_draft_spend": distribution(2.0),
            "paid_wipe_spend": distribution(1.0), "total_spend": distribution(3.0),
            "remaining": distribution(4.0), "free_replacements": distribution(1.0)
        });
        serde_json::json!({
            "schema_version": 1,
            "schema_id": R2_MAP_STATUS_SCHEMA_ID,
            "campaign_id": "r2-map-expert-iteration-v1",
            "updated_unix_ms": updated_unix_ms,
            "stale_after_seconds": 30,
            "phase": "training-on-john1-benchmarking-on-john2-john3",
            "legal_next_transitions": ["candidate-verified-benchmark-complete"],
            "round_index": 1,
            "models": {
                "incumbent": {"id": "c0", "blake3": null},
                "candidate": {"id": "t1", "blake3": null},
                "opponent_pool": [{"id": "greedy-v1", "blake3": null}]
            },
            "hosts": {
                "john1": {"intent": "train", "detail": "MLX", "generation_games_completed": 0, "generation_games_target": null, "generation_seed_prefix": null, "benchmark_pairs_completed": 0, "benchmark_pairs_total": null, "eta_seconds": 20.0, "throughput_games_per_second": null, "rss_bytes": 1000, "swap_delta_bytes": 0},
                "john2": {"intent": "idle", "detail": "Bacalhau scheduler capacity", "generation_games_completed": 0, "generation_games_target": null, "generation_seed_prefix": null, "benchmark_pairs_completed": 0, "benchmark_pairs_total": null, "eta_seconds": null, "throughput_games_per_second": null, "rss_bytes": 2000, "swap_delta_bytes": 0},
                "john3": {"intent": "idle", "detail": "Bacalhau scheduler capacity", "generation_games_completed": 0, "generation_games_target": null, "generation_seed_prefix": null, "benchmark_pairs_completed": 0, "benchmark_pairs_total": null, "eta_seconds": null, "throughput_games_per_second": null, "rss_bytes": 2000, "swap_delta_bytes": 0}
            },
            "training": {"active": true, "latest_verified_checkpoint": {"id": "step-100", "blake3": null}, "current_step": 120, "total_steps": 500, "examples_per_second": 400.0, "loss_samples": [{"step": 100, "train_total": 2.0, "validation_total": 2.1}, {"step": 120, "train_total": 1.8, "validation_total": null}]},
            "benchmark": {"active": true, "stage": "longitudinal", "pairs_completed": 40, "pairs_total": 100, "eta_seconds": 10.0, "throughput_games_per_second": 8.0, "peak_rss_bytes": 2000, "swap_delta_bytes": 0, "focal": {"base_total": distribution(96.0), "animals": animals, "habitat": habitat, "pinecones": pinecones}, "paired_delta": {"mean": 1.2, "confidence_95": [0.2, 2.2]}, "classification": "pending", "scheduler_work_items": {"completed": 40, "total": 100, "states": {"running": 60, "succeeded": 40}, "retry_attempts": 1, "utilization": {"sample_count": 4, "observed_seconds": 60.0, "cpu_capacity_min": 29.0, "cpu_capacity_max": 29.0, "cpu_allocated_mean": 27.0, "cpu_allocated_peak": 28.0, "cpu_utilization_mean": 0.931, "cpu_utilization_peak": 0.966}}}
        })
    }

    #[test]
    fn fresh_and_stale_mirrors_are_distinct() {
        let path = temporary_path("fresh-stale");
        fs::write(&path, serde_json::to_vec(&valid_status(100_000)).unwrap()).unwrap();
        let fresh = load(&path, 120_000);
        assert_eq!(fresh.condition, StatusCondition::Fresh);
        assert_eq!(fresh.age_seconds, Some(20.0));
        let stale = load(&path, 131_000);
        assert_eq!(stale.condition, StatusCondition::Stale);
        assert!(stale.status.is_some());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn missing_mirror_is_explicitly_unconfigured() {
        let response = load(Path::new("/definitely/not/r2-map-status.json"), 1);
        assert_eq!(response.condition, StatusCondition::Unconfigured);
        assert!(!response.configured);
        assert!(response.error.is_none());
    }

    #[test]
    fn malformed_and_semantically_invalid_mirrors_are_visible() {
        let malformed = temporary_path("malformed");
        fs::write(&malformed, b"{broken").unwrap();
        assert_eq!(load(&malformed, 1).condition, StatusCondition::Invalid);
        fs::remove_file(malformed).unwrap();

        let invalid = temporary_path("invalid");
        let mut value = valid_status(100_000);
        value["hosts"]["john4"] = value["hosts"]["john3"].clone();
        fs::write(&invalid, serde_json::to_vec(&value).unwrap()).unwrap();
        let response = load(&invalid, 100_001);
        assert_eq!(response.condition, StatusCondition::Invalid);
        assert!(
            response
                .error
                .unwrap()
                .contains("exactly john1, john2, and john3")
        );
        fs::remove_file(invalid).unwrap();

        let unknown = temporary_path("unknown-field");
        let mut value = valid_status(100_000);
        value["unversioned_extension"] = serde_json::json!(true);
        fs::write(&unknown, serde_json::to_vec(&value).unwrap()).unwrap();
        let response = load(&unknown, 100_001);
        assert_eq!(response.condition, StatusCondition::Invalid);
        assert!(response.error.unwrap().contains("unknown field"));
        fs::remove_file(unknown).unwrap();
    }

    #[test]
    fn scheduler_utilization_must_be_finite_bounded_and_consistent() {
        let path = temporary_path("invalid-scheduler-utilization");
        let mut value = valid_status(100_000);
        value["benchmark"]["scheduler_work_items"]["utilization"]["cpu_utilization_peak"] =
            serde_json::json!(1.01);
        fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        let response = load(&path, 100_001);
        assert_eq!(response.condition, StatusCondition::Invalid);
        assert!(response.error.unwrap().contains("scheduler utilization"));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn nonbenchmark_campaign_phases_may_publish_generic_scheduler_progress() {
        let path = temporary_path("generic-scheduler-progress");
        let mut value = valid_status(100_000);
        value["schema_id"] = serde_json::json!(V3_STATUS_SCHEMA_ID);
        value["campaign_id"] = serde_json::json!("cascadia-v3-radius7-stockfish-nnue-v1");
        value["phase"] = serde_json::json!("bootstrap_labeling");
        value["hosts"]["john4"] = serde_json::json!({
            "intent": "idle", "detail": "dashboard-only", "generation_games_completed": 0,
            "generation_games_target": null, "generation_seed_prefix": null,
            "benchmark_pairs_completed": 0, "benchmark_pairs_total": null,
            "eta_seconds": null, "throughput_games_per_second": null,
            "rss_bytes": null, "swap_delta_bytes": 0
        });
        value["training"]["active"] = serde_json::json!(false);
        value["benchmark"]["active"] = serde_json::json!(false);
        value["benchmark"]["pairs_completed"] = serde_json::json!(0);
        value["benchmark"]["pairs_total"] = serde_json::Value::Null;
        value["benchmark"]["scheduler_work_items"] = serde_json::json!({
            "completed": 0, "total": 120,
            "states": {"running": 29, "queued": 91},
            "retry_attempts": 0, "utilization": null
        });
        fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        let response = load(&path, 100_001);
        assert_eq!(response.condition, StatusCondition::Fresh);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn compact_reader_refuses_oversized_files_without_scanning_siblings() {
        let path = temporary_path("oversized");
        fs::write(&path, vec![b' '; MAX_STATUS_BYTES as usize + 1]).unwrap();
        let response = load(&path, 1);
        assert_eq!(response.condition, StatusCondition::Invalid);
        assert!(response.error.unwrap().contains("maximum"));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn implausibly_future_mirror_is_invalid_instead_of_fresh() {
        let path = temporary_path("future");
        fs::write(&path, serde_json::to_vec(&valid_status(161_001)).unwrap()).unwrap();
        let response = load(&path, 100_000);
        assert_eq!(response.condition, StatusCondition::Invalid);
        assert!(response.error.unwrap().contains("future"));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn python_publisher_golden_fixtures_match_the_rust_reader_contract() {
        let contracts = load(
            &golden_fixture("dashboard-status-v1-contracts-ready.json"),
            1_781_755_201_000,
        );
        assert_eq!(contracts.condition, StatusCondition::Fresh);
        let contracts = contracts.status.unwrap();
        assert_eq!(contracts.phase, "contracts-ready");
        assert_eq!(contracts.hosts.len(), 3);
        assert!(contracts.benchmark.focal.is_none());

        let training = load(
            &golden_fixture("dashboard-status-v1-training.json"),
            1_781_755_201_000,
        );
        assert_eq!(training.condition, StatusCondition::Fresh);
        let training = training.status.unwrap();
        assert_eq!(training.round_index, Some(0));
        assert_eq!(training.training.loss_samples.len(), 2);
        assert_eq!(training.benchmark.pairs_completed, 80);
        assert_eq!(training.benchmark.focal.unwrap().base_total.mean, 96.25);
    }

    #[test]
    fn serving_projection_is_hash_bound_stale_aware_and_fail_visible() {
        let path = temporary_path("serving-projection");
        let payload =
            fs::read_to_string(golden_fixture("dashboard-status-v1-contracts-ready.json")).unwrap();
        let mut projection = serving_projection(&payload, 1_781_755_200_000);
        fs::write(&path, serde_json::to_vec(&projection).unwrap()).unwrap();
        assert_eq!(
            load(&path, 1_781_755_201_000).condition,
            StatusCondition::Fresh
        );
        assert_eq!(
            load(&path, 1_781_755_231_000).condition,
            StatusCondition::Stale
        );

        projection["canonical_updated_unix_ms"] = serde_json::json!(1_781_755_200_001_u64);
        fs::write(&path, serde_json::to_vec(&projection).unwrap()).unwrap();
        let timestamp_drift = load(&path, 1_781_755_201_000);
        assert_eq!(timestamp_drift.condition, StatusCondition::Invalid);
        assert!(timestamp_drift.error.unwrap().contains("timestamp"));

        projection["canonical_updated_unix_ms"] = serde_json::json!(1_781_755_200_000_u64);
        projection["canonical_blake3"] = serde_json::json!("0".repeat(64));
        fs::write(&path, serde_json::to_vec(&projection).unwrap()).unwrap();
        let hash_drift = load(&path, 1_781_755_201_000);
        assert_eq!(hash_drift.condition, StatusCondition::Invalid);
        assert!(hash_drift.error.unwrap().contains("hash"));

        projection = serving_projection(&payload, 1_781_755_200_000);
        projection["canonical_host"] = serde_json::json!("john2");
        fs::write(&path, serde_json::to_vec(&projection).unwrap()).unwrap();
        let source_drift = load(&path, 1_781_755_201_000);
        assert_eq!(source_drift.condition, StatusCondition::Invalid);
        assert!(source_drift.error.unwrap().contains("identity"));

        let legacy = legacy_serving_projection(&payload, 1_781_755_200_000);
        fs::write(&path, serde_json::to_vec(&legacy).unwrap()).unwrap();
        let retired = load(&path, 1_781_755_201_000);
        assert_eq!(retired.condition, StatusCondition::Invalid);
        assert!(retired.error.unwrap().contains("external-SSD"));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn john1_runtime_stage_identity_survives_the_bounded_v2_projection() {
        let path = temporary_path("runtime-stage");
        let mut status = valid_status(1_781_755_200_000);
        let detail = format!(
            "runtime-stage:verified run=bootstrap-0001 sha256={} bytes=67108864 cleanup=pending",
            "a".repeat(64)
        );
        status["hosts"]["john1"]["detail"] = serde_json::json!(detail);
        let payload = serde_json::to_string(&status).unwrap();
        let projection = serving_projection(&payload, 1_781_755_200_000);
        let encoded = serde_json::to_vec(&projection).unwrap();
        assert!(encoded.len() <= MAX_SERVING_PROJECTION_BYTES as usize);
        fs::write(&path, encoded).unwrap();

        let loaded = load(&path, 1_781_755_201_000);
        assert_eq!(loaded.condition, StatusCondition::Fresh);
        assert_eq!(
            loaded.status.unwrap().hosts["john1"].detail.as_deref(),
            Some(detail.as_str())
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn host_detail_is_bounded_before_it_reaches_the_serving_projection() {
        let path = temporary_path("oversized-host-detail");
        let mut status = valid_status(1_781_755_200_000);
        status["hosts"]["john1"]["detail"] =
            serde_json::json!("x".repeat(MAX_HOST_DETAIL_BYTES + 1));
        fs::write(&path, serde_json::to_vec(&status).unwrap()).unwrap();
        let loaded = load(&path, 1_781_755_201_000);
        assert_eq!(loaded.condition, StatusCondition::Invalid);
        assert!(loaded.error.unwrap().contains("at most 512 bytes"));
        fs::remove_file(path).unwrap();
    }
}
