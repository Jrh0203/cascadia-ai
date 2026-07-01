//! Restart-safe, absolute 100-game focal benchmark for the incumbent model.
//!
//! This is intentionally separate from the paired candidate gate. It measures
//! one frozen checkpoint against one frozen historical field while John1 is
//! training the next candidate. Every game is an independent scheduler-owned
//! work item; Bacalhau owns placement, admission, retry, and rescheduling.

use std::{
    collections::{BTreeSet, HashSet},
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use cascadia_game::{GameSeed, ScoreBreakdown};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    focal::{
        BenchmarkStage, FocalGameRecord, FocalRecordIdentity, FocalRuntimeObservation,
        FocalStatistics, OpponentIdentity, PairArm, PineconeObservation,
        aggregate_focal_statistics,
    },
    focal_campaign::{
        FocalCampaignError, FocalGameRequest, canonical_blake3, file_sha256, read_json,
        write_immutable_json,
    },
    r2_map_binding::R2MapImplementationBinding,
};

pub const LONGITUDINAL_SCHEMA_VERSION: u16 = 4;
pub const LONGITUDINAL_GAME_COUNT: usize = 100;
pub const LONGITUDINAL_CONTRACT_SCHEMA_ID: &str = "cascadia.r2-map.longitudinal-contract.v4";
pub const LONGITUDINAL_FIELD_SCHEMA_ID: &str = "cascadia.r2-map.longitudinal-field.v4";
pub const LONGITUDINAL_RECEIPT_SCHEMA_ID: &str = "cascadia.r2-map.longitudinal-game-receipt.v4";
pub const LONGITUDINAL_WORK_ITEM_SCHEMA_ID: &str = "cascadia.r2-map.longitudinal-work-item.v4";
pub const LONGITUDINAL_REPORT_SCHEMA_ID: &str = "cascadia.r2-map.longitudinal-report.v4";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BenchmarkEvidenceClass {
    /// Real engine and model service, but a smoke-trained checkpoint whose
    /// score cannot support a model-strength conclusion.
    RealOpenCheckpointPerformanceOnly,
    /// A verified incumbent measured descriptively one iteration behind
    /// training. This is not the paired promotion gate or final domain.
    RealExpertIterationCheckpointDescriptive,
    /// Deterministic fixtures used only for artifact/integrity tests.
    SyntheticFixture,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LongitudinalBenchmarkPurpose {
    ExpertIterationLongitudinal,
    OpenPerformanceReferenceOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum LongitudinalExecutionPartition {
    SchedulerManagedGames,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LongitudinalBenchmarkContract {
    pub schema_version: u16,
    pub schema_id: String,
    pub campaign_id: String,
    pub benchmark_id: String,
    pub iteration: u32,
    pub game_count: usize,
    pub focal_checkpoint_id: String,
    pub historical_field_manifest_id: String,
    pub historical_pool_checkpoint_ids: Vec<String>,
    pub inference_settings_id: String,
    pub seed_domain_id: String,
    pub execution_partition: LongitudinalExecutionPartition,
    pub purpose: LongitudinalBenchmarkPurpose,
    pub evidence_class: BenchmarkEvidenceClass,
    pub strength_claim_authorized: bool,
    pub implementation_binding: R2MapImplementationBinding,
}

impl LongitudinalBenchmarkContract {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        campaign_id: impl Into<String>,
        benchmark_id: impl Into<String>,
        iteration: u32,
        focal_checkpoint_id: impl Into<String>,
        historical_field_manifest_id: impl Into<String>,
        historical_pool_checkpoint_ids: Vec<String>,
        inference_settings_id: impl Into<String>,
        seed_domain_id: impl Into<String>,
        purpose: LongitudinalBenchmarkPurpose,
        evidence_class: BenchmarkEvidenceClass,
        implementation_binding: R2MapImplementationBinding,
    ) -> Self {
        Self {
            schema_version: LONGITUDINAL_SCHEMA_VERSION,
            schema_id: LONGITUDINAL_CONTRACT_SCHEMA_ID.to_owned(),
            campaign_id: campaign_id.into(),
            benchmark_id: benchmark_id.into(),
            iteration,
            game_count: LONGITUDINAL_GAME_COUNT,
            focal_checkpoint_id: focal_checkpoint_id.into(),
            historical_field_manifest_id: historical_field_manifest_id.into(),
            historical_pool_checkpoint_ids,
            inference_settings_id: inference_settings_id.into(),
            seed_domain_id: seed_domain_id.into(),
            execution_partition: LongitudinalExecutionPartition::SchedulerManagedGames,
            purpose,
            evidence_class,
            strength_claim_authorized: false,
            implementation_binding,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LongitudinalGameAssignment {
    pub game_index: usize,
    pub game_seed: GameSeed,
    pub focal_seat: u8,
    pub opponents: Vec<OpponentIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LongitudinalFieldManifest {
    pub schema_version: u16,
    pub schema_id: String,
    pub manifest_id: String,
    pub assignments: Vec<LongitudinalGameAssignment>,
}

impl LongitudinalFieldManifest {
    pub fn new(
        manifest_id: impl Into<String>,
        assignments: Vec<LongitudinalGameAssignment>,
    ) -> Self {
        Self {
            schema_version: LONGITUDINAL_SCHEMA_VERSION,
            schema_id: LONGITUDINAL_FIELD_SCHEMA_ID.to_owned(),
            manifest_id: manifest_id.into(),
            assignments,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LongitudinalGameRecord {
    pub game_index: usize,
    pub game_seed: GameSeed,
    pub focal_seat: u8,
    pub focal_checkpoint_id: String,
    pub opponents: Vec<OpponentIdentity>,
    pub final_state_hash: [u8; 32],
    pub replay_blake3: String,
    pub score: ScoreBreakdown,
    pub pinecones: PineconeObservation,
    pub focal_decision_seconds: Vec<f64>,
    pub elapsed_seconds: f64,
    pub runtime: FocalRuntimeObservation,
}

impl LongitudinalGameRecord {
    fn from_focal(game_index: usize, value: FocalGameRecord) -> Self {
        Self {
            game_index,
            game_seed: value.game_seed,
            focal_seat: value.focal_seat,
            focal_checkpoint_id: value.identity.focal_checkpoint_id,
            opponents: value.identity.opponents,
            final_state_hash: value.final_state_hash,
            replay_blake3: value.replay_blake3,
            score: value.score,
            pinecones: value.pinecones,
            focal_decision_seconds: value.focal_decision_seconds,
            elapsed_seconds: value.elapsed_seconds,
            runtime: value.runtime,
        }
    }

    fn to_focal(&self, contract: &LongitudinalBenchmarkContract) -> FocalGameRecord {
        FocalGameRecord {
            schema_version: crate::focal::FOCAL_BENCHMARK_SCHEMA_VERSION,
            protocol_id: crate::focal::FOCAL_BENCHMARK_PROTOCOL_ID.to_owned(),
            identity: FocalRecordIdentity {
                stage: BenchmarkStage::Development,
                pair_index: self.game_index,
                arm: PairArm::Control,
                focal_checkpoint_id: self.focal_checkpoint_id.clone(),
                opponents: self.opponents.clone(),
                field_manifest_id: contract.historical_field_manifest_id.clone(),
                inference_settings_id: contract.inference_settings_id.clone(),
            },
            game_seed: self.game_seed,
            focal_seat: self.focal_seat,
            final_state_hash: self.final_state_hash,
            replay_blake3: self.replay_blake3.clone(),
            score: self.score,
            pinecones: self.pinecones,
            focal_decision_seconds: self.focal_decision_seconds.clone(),
            elapsed_seconds: self.elapsed_seconds,
            runtime: self.runtime,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LongitudinalGameReceipt {
    pub schema_version: u16,
    pub schema_id: String,
    pub contract_blake3: String,
    pub field_blake3: String,
    pub contract_sha256: String,
    pub field_sha256: String,
    pub implementation_binding: R2MapImplementationBinding,
    pub payload_blake3: String,
    pub payload: LongitudinalGameRecord,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LongitudinalReceiptReference {
    pub game_index: usize,
    pub receipt_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LongitudinalWorkItem {
    pub schema_version: u16,
    pub schema_id: String,
    pub contract_blake3: String,
    pub field_blake3: String,
    pub contract_sha256: String,
    pub field_sha256: String,
    pub implementation_binding: R2MapImplementationBinding,
    pub work_item_id: String,
    pub games: usize,
    pub receipts: Vec<LongitudinalReceiptReference>,
    pub peak_rss_bytes: u64,
    pub maximum_swap_delta_bytes: i64,
    pub all_clean_shutdowns: bool,
    pub all_pinecone_conservation_checks_passed: bool,
    pub summed_game_seconds: f64,
    pub summed_checkpoint_load_seconds: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LongitudinalBenchmarkReport {
    pub schema_version: u16,
    pub schema_id: String,
    pub campaign_id: String,
    pub benchmark_id: String,
    pub iteration: u32,
    pub checkpoint_id: String,
    pub purpose: LongitudinalBenchmarkPurpose,
    pub evidence_class: BenchmarkEvidenceClass,
    pub strength_claim_authorized: bool,
    pub reference_optimized_pair_complete: bool,
    pub contract_blake3: String,
    pub field_blake3: String,
    pub contract_sha256: String,
    pub field_sha256: String,
    pub implementation_binding: R2MapImplementationBinding,
    pub work_items: Vec<LongitudinalWorkItem>,
    pub games: usize,
    pub focal: FocalStatistics,
    pub distance_from_100: f64,
    pub wall_seconds: f64,
    pub games_per_second: f64,
    pub peak_rss_bytes: u64,
    pub maximum_swap_delta_bytes: i64,
    pub all_clean_shutdowns: bool,
    pub all_pinecone_conservation_checks_passed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LongitudinalRunOutcome {
    pub assigned_games: usize,
    pub executed_games: usize,
    pub resumed_games: usize,
}

pub trait LongitudinalGameExecutor {
    type Error: Error + Send + Sync + 'static;

    fn execute_longitudinal(
        &mut self,
        request: &FocalGameRequest,
    ) -> Result<FocalGameRecord, Self::Error>;
}

impl<F, E> LongitudinalGameExecutor for F
where
    F: FnMut(&FocalGameRequest) -> Result<FocalGameRecord, E>,
    E: Error + Send + Sync + 'static,
{
    type Error = E;

    fn execute_longitudinal(
        &mut self,
        request: &FocalGameRequest,
    ) -> Result<FocalGameRecord, Self::Error> {
        self(request)
    }
}

#[derive(Debug, Clone)]
pub struct LongitudinalLayout {
    root: PathBuf,
}

impl LongitudinalLayout {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn contract_path(&self) -> PathBuf {
        self.root.join("contract.json")
    }

    pub fn field_path(&self) -> PathBuf {
        self.root.join("historical-field.json")
    }

    pub fn receipt_directory(&self, work_item_id: &str) -> PathBuf {
        self.root.join("receipts").join(work_item_id)
    }

    pub fn receipt_path(&self, work_item_id: &str, index: usize) -> PathBuf {
        self.receipt_directory(work_item_id)
            .join(format!("game-{index:04}.json"))
    }

    pub fn work_item_summary_path(&self, work_item_id: &str) -> PathBuf {
        self.root
            .join("work-item-summaries")
            .join(format!("{work_item_id}.json"))
    }

    pub fn report_path(&self) -> PathBuf {
        self.root.join("reports/longitudinal-benchmark.json")
    }

    pub fn dashboard_input_path(&self) -> PathBuf {
        self.root.join("projections/dashboard-benchmark.json")
    }

    pub fn ledger_feed_path(&self) -> PathBuf {
        self.root.join("projections/ledger-experiment.json")
    }
}

pub fn initialize_longitudinal_campaign(
    root: impl Into<PathBuf>,
    contract: &LongitudinalBenchmarkContract,
    field: &LongitudinalFieldManifest,
) -> Result<LongitudinalLayout, LongitudinalError> {
    validate_contract_and_field(contract, field)?;
    let layout = LongitudinalLayout::new(root);
    let mut directories = vec![
        layout.root.clone(),
        layout.root.join("work-item-summaries"),
        layout.root.join("reports"),
        layout.root.join("projections"),
    ];
    directories.extend(
        longitudinal_work_items(contract)
            .into_iter()
            .map(|work_item_id| layout.receipt_directory(&work_item_id)),
    );
    for directory in directories {
        fs::create_dir_all(directory)?;
    }
    write_immutable_json(&layout.contract_path(), contract)?;
    write_immutable_json(&layout.field_path(), field)?;
    Ok(layout)
}

pub fn run_longitudinal_work_item<E: LongitudinalGameExecutor>(
    layout: &LongitudinalLayout,
    work_item_id: &str,
    executor: &mut E,
) -> Result<LongitudinalRunOutcome, LongitudinalError> {
    let game_index = parse_work_item_id(work_item_id)?;
    let (contract, field) = read_inputs(layout)?;
    if game_index >= contract.game_count {
        return Err(LongitudinalError::WorkItem(work_item_id.to_owned()));
    }
    validate_receipt_directory(layout, work_item_id, &field)?;
    let bindings = bindings(layout, &contract, &field)?;
    let assignments = assignments_for_work_item(&field, work_item_id)?;
    let mut executed_games = 0;
    let mut resumed_games = 0;
    for assignment in &assignments {
        let path = layout.receipt_path(work_item_id, assignment.game_index);
        if path.exists() {
            let receipt: LongitudinalGameReceipt = read_json(&path)?;
            validate_receipt(&receipt, &contract, assignment, &bindings)?;
            resumed_games += 1;
            continue;
        }
        let request = game_request(&contract, assignment);
        let focal = executor.execute_longitudinal(&request).map_err(|error| {
            LongitudinalError::Executor {
                game_index: assignment.game_index,
                detail: error.to_string(),
            }
        })?;
        let payload = LongitudinalGameRecord::from_focal(assignment.game_index, focal);
        validate_record(&payload, &contract, assignment)?;
        let receipt = LongitudinalGameReceipt {
            schema_version: LONGITUDINAL_SCHEMA_VERSION,
            schema_id: LONGITUDINAL_RECEIPT_SCHEMA_ID.to_owned(),
            contract_blake3: bindings.contract_blake3.clone(),
            field_blake3: bindings.field_blake3.clone(),
            contract_sha256: bindings.contract_sha256.clone(),
            field_sha256: bindings.field_sha256.clone(),
            implementation_binding: contract.implementation_binding.clone(),
            payload_blake3: canonical_blake3(&payload)?,
            payload,
        };
        write_immutable_json(&path, &receipt)?;
        executed_games += 1;
    }
    validate_receipt_directory(layout, work_item_id, &field)?;
    let summary = build_work_item_summary(layout, work_item_id, &contract, &field, &bindings)?;
    write_immutable_json(&layout.work_item_summary_path(work_item_id), &summary)?;
    Ok(LongitudinalRunOutcome {
        assigned_games: assignments.len(),
        executed_games,
        resumed_games,
    })
}

pub fn aggregate_longitudinal_campaign(
    layout: &LongitudinalLayout,
    wall_seconds: f64,
) -> Result<LongitudinalBenchmarkReport, LongitudinalError> {
    if !wall_seconds.is_finite() || wall_seconds <= 0.0 {
        return Err(LongitudinalError::InvalidWallTime);
    }
    let (contract, field) = read_inputs(layout)?;
    let bindings = bindings(layout, &contract, &field)?;
    let mut work_items = Vec::new();
    let mut records = Vec::with_capacity(LONGITUDINAL_GAME_COUNT);
    for work_item_id in longitudinal_work_items(&contract) {
        validate_receipt_directory(layout, &work_item_id, &field)?;
        let stored: LongitudinalWorkItem =
            read_json(&layout.work_item_summary_path(&work_item_id))?;
        let recomputed =
            build_work_item_summary(layout, &work_item_id, &contract, &field, &bindings)?;
        if stored != recomputed {
            return Err(LongitudinalError::WorkItemSummaryDrift(work_item_id));
        }
        for assignment in assignments_for_work_item(&field, &stored.work_item_id)? {
            let receipt: LongitudinalGameReceipt =
                read_json(&layout.receipt_path(&stored.work_item_id, assignment.game_index))?;
            validate_receipt(&receipt, &contract, assignment, &bindings)?;
            records.push(receipt.payload.to_focal(&contract));
        }
        work_items.push(stored);
    }
    records.sort_by_key(|record| record.identity.pair_index);
    if records.len() != LONGITUDINAL_GAME_COUNT
        || records
            .iter()
            .enumerate()
            .any(|(index, record)| record.identity.pair_index != index)
    {
        return Err(LongitudinalError::Coverage);
    }
    let focal = aggregate_focal_statistics(&records)?;
    work_items.sort_by(|left, right| left.work_item_id.cmp(&right.work_item_id));
    let peak_rss_bytes = work_items
        .iter()
        .map(|item| item.peak_rss_bytes)
        .max()
        .unwrap_or(0);
    let maximum_swap_delta_bytes = work_items
        .iter()
        .map(|item| item.maximum_swap_delta_bytes)
        .max()
        .unwrap_or(0);
    let report = LongitudinalBenchmarkReport {
        schema_version: LONGITUDINAL_SCHEMA_VERSION,
        schema_id: LONGITUDINAL_REPORT_SCHEMA_ID.to_owned(),
        campaign_id: contract.campaign_id,
        benchmark_id: contract.benchmark_id,
        iteration: contract.iteration,
        checkpoint_id: contract.focal_checkpoint_id,
        purpose: contract.purpose,
        evidence_class: contract.evidence_class,
        strength_claim_authorized: contract.strength_claim_authorized,
        reference_optimized_pair_complete: false,
        contract_blake3: bindings.contract_blake3,
        field_blake3: bindings.field_blake3,
        contract_sha256: bindings.contract_sha256,
        field_sha256: bindings.field_sha256,
        implementation_binding: contract.implementation_binding,
        games: records.len(),
        distance_from_100: focal.base_total.mean - 100.0,
        wall_seconds,
        games_per_second: records.len() as f64 / wall_seconds,
        peak_rss_bytes,
        maximum_swap_delta_bytes,
        all_clean_shutdowns: work_items.iter().all(|item| item.all_clean_shutdowns),
        all_pinecone_conservation_checks_passed: work_items
            .iter()
            .all(|item| item.all_pinecone_conservation_checks_passed),
        work_items,
        focal,
    };
    write_immutable_json(&layout.report_path(), &report)?;
    write_immutable_json(
        &layout.dashboard_input_path(),
        &dashboard_benchmark_input(&report),
    )?;
    write_immutable_json(&layout.ledger_feed_path(), &ledger_experiment_feed(&report))?;
    Ok(report)
}

pub fn dashboard_benchmark_input(report: &LongitudinalBenchmarkReport) -> serde_json::Value {
    let compact = |value: &crate::focal::IntegerDistribution| {
        serde_json::json!({
            "mean": value.mean,
            "p10": value.p10,
            "p50": value.p50,
            "p90": value.p90,
        })
    };
    serde_json::json!({
        "active": false,
        "stage": match report.purpose {
            LongitudinalBenchmarkPurpose::ExpertIterationLongitudinal => "longitudinal-100",
            LongitudinalBenchmarkPurpose::OpenPerformanceReferenceOnly => "open-performance-reference-100",
        },
        "pairs_completed": report.games,
        "pairs_total": report.games,
        "eta_seconds": 0.0,
        "throughput_games_per_second": report.games_per_second,
        "peak_rss_bytes": report.peak_rss_bytes,
        "swap_delta_bytes": report.maximum_swap_delta_bytes,
        "focal": {
            "base_total": compact(&report.focal.base_total),
            "animals": {
                "aggregate": compact(&report.focal.animals.aggregate_wildlife),
                "bear": compact(&report.focal.animals.bear),
                "elk": compact(&report.focal.animals.elk),
                "salmon": compact(&report.focal.animals.salmon),
                "hawk": compact(&report.focal.animals.hawk),
                "fox": compact(&report.focal.animals.fox),
            },
            "habitat": {
                "aggregate": compact(&report.focal.terrains.aggregate_habitat),
                "mountain": compact(&report.focal.terrains.mountain),
                "forest": compact(&report.focal.terrains.forest),
                "prairie": compact(&report.focal.terrains.prairie),
                "wetland": compact(&report.focal.terrains.wetland),
                "river": compact(&report.focal.terrains.river),
            },
            "pinecones": {
                "earned": compact(&report.focal.pinecones.earned),
                "independent_draft_spend": compact(&report.focal.pinecones.independent_draft_spend),
                "paid_wipe_spend": compact(&report.focal.pinecones.paid_wipe_spend),
                "total_spend": compact(&report.focal.pinecones.total_spend),
                "remaining": compact(&report.focal.pinecones.remaining),
                "free_replacements": compact(&report.focal.pinecones.free_replacements),
            },
        },
        "paired_delta": null,
        "classification": "pending",
    })
}

pub fn ledger_experiment_feed(report: &LongitudinalBenchmarkReport) -> serde_json::Value {
    let (title, hypothesis, purpose_tag, panel_tag) = match report.purpose {
        LongitudinalBenchmarkPurpose::ExpertIterationLongitudinal => (
            "R2-MAP longitudinal focal benchmark",
            "The frozen incumbent remains measurable against its frozen historical field while candidate training runs independently.",
            "longitudinal",
            "iteration-benchmark",
        ),
        LongitudinalBenchmarkPurpose::OpenPerformanceReferenceOnly => (
            "R2-MAP open reference performance panel",
            "The exhaustive reference serving path completes the frozen open 100-game panel with exact replay and resource accounting.",
            "open-reference-performance",
            "open-panel",
        ),
    };
    serde_json::json!({
        "id": format!("{}-{}", report.campaign_id, report.benchmark_id),
        "title": title,
        "hypothesis": hypothesis,
        "summary": format!(
            "Completed {} open focal games at {:.3} games/s; evidence is {:?} and is not a strength qualification.",
            report.games, report.games_per_second, report.evidence_class
        ),
        "status": "completed",
        "outcome": if report.all_clean_shutdowns && report.all_pinecone_conservation_checks_passed { "passed" } else { "invalid" },
        "verdict": null,
        "plan_section": "W5",
        "started_unix_ms": 0,
        "completed_unix_ms": 0,
        "updated_unix_ms": 0,
        "work_items": report.work_items.iter().map(|item| item.work_item_id.clone()).collect::<Vec<_>>(),
        "tags": ["r2-map", purpose_tag, "focal", panel_tag],
        "task_ids": [],
        "metrics": [
            {"label": "Games", "value": report.games.to_string(), "tone": "neutral"},
            {"label": "Mean focal score", "value": format!("{:.3}", report.focal.base_total.mean), "tone": "neutral"},
            {"label": "Games/second", "value": format!("{:.3}", report.games_per_second), "tone": "neutral"},
            {"label": "Peak RSS bytes", "value": report.peak_rss_bytes.to_string(), "tone": "neutral"},
        ],
        "criteria": [
            {"label": "Complete fixed 100-game coverage", "passed": report.games == LONGITUDINAL_GAME_COUNT, "observed": report.games.to_string()},
            {"label": "Pinecone conservation", "passed": report.all_pinecone_conservation_checks_passed, "observed": report.all_pinecone_conservation_checks_passed.to_string()},
            {"label": "Zero positive swap growth", "passed": report.maximum_swap_delta_bytes <= 0, "observed": report.maximum_swap_delta_bytes.to_string()},
        ],
        "notes": [
            "This feed is a deterministic projection; the controller stamps import time.",
            "strength_claim_authorized=false",
            "reference_optimized_pair_complete=false",
        ],
        "artifacts": [
            {"label": "R2-MAP plan", "path": "docs/v2/R2_MAP_EXPERT_ITERATION_RESEARCH_PLAN.md"}
        ],
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Bindings {
    contract_blake3: String,
    field_blake3: String,
    contract_sha256: String,
    field_sha256: String,
}

fn bindings(
    layout: &LongitudinalLayout,
    contract: &LongitudinalBenchmarkContract,
    field: &LongitudinalFieldManifest,
) -> Result<Bindings, LongitudinalError> {
    Ok(Bindings {
        contract_blake3: canonical_blake3(contract)?,
        field_blake3: canonical_blake3(field)?,
        contract_sha256: file_sha256(&layout.contract_path())?,
        field_sha256: file_sha256(&layout.field_path())?,
    })
}

fn read_inputs(
    layout: &LongitudinalLayout,
) -> Result<(LongitudinalBenchmarkContract, LongitudinalFieldManifest), LongitudinalError> {
    let contract = read_json(&layout.contract_path())?;
    let field = read_json(&layout.field_path())?;
    validate_contract_and_field(&contract, &field)?;
    Ok((contract, field))
}

fn validate_contract_and_field(
    contract: &LongitudinalBenchmarkContract,
    field: &LongitudinalFieldManifest,
) -> Result<(), LongitudinalError> {
    if contract.schema_version != LONGITUDINAL_SCHEMA_VERSION
        || contract.schema_id != LONGITUDINAL_CONTRACT_SCHEMA_ID
        || field.schema_version != LONGITUDINAL_SCHEMA_VERSION
        || field.schema_id != LONGITUDINAL_FIELD_SCHEMA_ID
    {
        return Err(LongitudinalError::Schema);
    }
    if contract.game_count != LONGITUDINAL_GAME_COUNT
        || contract.strength_claim_authorized
        || contract.historical_field_manifest_id != field.manifest_id
        || contract.execution_partition != LongitudinalExecutionPartition::SchedulerManagedGames
    {
        return Err(LongitudinalError::Contract);
    }
    if matches!(
        contract.purpose,
        LongitudinalBenchmarkPurpose::OpenPerformanceReferenceOnly
    ) && contract.evidence_class != BenchmarkEvidenceClass::RealOpenCheckpointPerformanceOnly
    {
        return Err(LongitudinalError::Contract);
    }
    contract
        .implementation_binding
        .validate()
        .map_err(|_| LongitudinalError::ImplementationBinding)?;
    if matches!(
        contract.purpose,
        LongitudinalBenchmarkPurpose::OpenPerformanceReferenceOnly
    ) && contract.seed_domain_id
        != contract
            .implementation_binding
            .open_reference_seed_domain_id
    {
        return Err(LongitudinalError::ImplementationBinding);
    }
    let identities = [
        contract.campaign_id.as_str(),
        contract.benchmark_id.as_str(),
        contract.focal_checkpoint_id.as_str(),
        contract.historical_field_manifest_id.as_str(),
        contract.inference_settings_id.as_str(),
        contract.seed_domain_id.as_str(),
    ];
    if identities.iter().any(|value| value.trim().is_empty())
        || contract.historical_pool_checkpoint_ids.is_empty()
        || contract
            .historical_pool_checkpoint_ids
            .iter()
            .any(|value| value.trim().is_empty())
        || contract
            .historical_pool_checkpoint_ids
            .iter()
            .collect::<BTreeSet<_>>()
            .len()
            != contract.historical_pool_checkpoint_ids.len()
    {
        return Err(LongitudinalError::Identity);
    }
    if field.assignments.len() != LONGITUDINAL_GAME_COUNT {
        return Err(LongitudinalError::Coverage);
    }
    let pool = contract
        .historical_pool_checkpoint_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let mut indices = BTreeSet::new();
    let mut seeds = HashSet::new();
    for assignment in &field.assignments {
        if !indices.insert(assignment.game_index) || !seeds.insert(assignment.game_seed) {
            return Err(LongitudinalError::Duplicate);
        }
        if assignment.focal_seat != (assignment.game_index % 4) as u8
            || assignment.opponents.len() != 3
        {
            return Err(LongitudinalError::Assignment(assignment.game_index));
        }
        let seats = assignment
            .opponents
            .iter()
            .map(|opponent| opponent.seat)
            .collect::<BTreeSet<_>>();
        let expected_seats = (0..4)
            .filter(|seat| *seat != assignment.focal_seat)
            .collect::<BTreeSet<_>>();
        if seats != expected_seats
            || assignment
                .opponents
                .iter()
                .any(|opponent| !pool.contains(opponent.checkpoint_id.as_str()))
        {
            return Err(LongitudinalError::Assignment(assignment.game_index));
        }
    }
    if indices != (0..LONGITUDINAL_GAME_COUNT).collect() {
        return Err(LongitudinalError::Coverage);
    }
    Ok(())
}

fn game_request(
    contract: &LongitudinalBenchmarkContract,
    assignment: &LongitudinalGameAssignment,
) -> FocalGameRequest {
    FocalGameRequest {
        benchmark_id: contract.benchmark_id.clone(),
        implementation_binding: contract.implementation_binding.clone(),
        identity: FocalRecordIdentity {
            // Development permits indices 0..249 and exposes the same complete
            // score/Pinecone validation. The longitudinal artifact itself has
            // its own explicit schema and never persists this adapter label.
            stage: BenchmarkStage::Development,
            pair_index: assignment.game_index,
            arm: PairArm::Control,
            focal_checkpoint_id: contract.focal_checkpoint_id.clone(),
            opponents: assignment.opponents.clone(),
            field_manifest_id: contract.historical_field_manifest_id.clone(),
            inference_settings_id: contract.inference_settings_id.clone(),
        },
        game_seed: assignment.game_seed,
        focal_seat: assignment.focal_seat,
    }
}

fn validate_record(
    record: &LongitudinalGameRecord,
    contract: &LongitudinalBenchmarkContract,
    assignment: &LongitudinalGameAssignment,
) -> Result<(), LongitudinalError> {
    if record.game_index != assignment.game_index
        || record.game_seed != assignment.game_seed
        || record.focal_seat != assignment.focal_seat
        || record.focal_checkpoint_id != contract.focal_checkpoint_id
        || record.opponents != assignment.opponents
    {
        return Err(LongitudinalError::RecordIdentity(assignment.game_index));
    }
    crate::focal::validate_focal_record(&record.to_focal(contract))?;
    Ok(())
}

fn validate_receipt(
    receipt: &LongitudinalGameReceipt,
    contract: &LongitudinalBenchmarkContract,
    assignment: &LongitudinalGameAssignment,
    bindings: &Bindings,
) -> Result<(), LongitudinalError> {
    if receipt.schema_version != LONGITUDINAL_SCHEMA_VERSION
        || receipt.schema_id != LONGITUDINAL_RECEIPT_SCHEMA_ID
        || receipt.contract_blake3 != bindings.contract_blake3
        || receipt.field_blake3 != bindings.field_blake3
        || receipt.contract_sha256 != bindings.contract_sha256
        || receipt.field_sha256 != bindings.field_sha256
        || receipt.implementation_binding != contract.implementation_binding
        || receipt.payload_blake3 != canonical_blake3(&receipt.payload)?
    {
        return Err(LongitudinalError::ReceiptDrift(assignment.game_index));
    }
    validate_record(&receipt.payload, contract, assignment)
}

fn work_item_id(game_index: usize) -> String {
    format!("game-{game_index:04}")
}

fn parse_work_item_id(value: &str) -> Result<usize, LongitudinalError> {
    let Some(raw) = value.strip_prefix("game-") else {
        return Err(LongitudinalError::WorkItem(value.to_owned()));
    };
    if raw.len() != 4 || !raw.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(LongitudinalError::WorkItem(value.to_owned()));
    }
    raw.parse()
        .map_err(|_| LongitudinalError::WorkItem(value.to_owned()))
}

fn longitudinal_work_items(contract: &LongitudinalBenchmarkContract) -> Vec<String> {
    (0..contract.game_count).map(work_item_id).collect()
}

fn assignments_for_work_item<'a>(
    field: &'a LongitudinalFieldManifest,
    work_item_id: &str,
) -> Result<Vec<&'a LongitudinalGameAssignment>, LongitudinalError> {
    let game_index = parse_work_item_id(work_item_id)?;
    let assignments = field
        .assignments
        .iter()
        .filter(|assignment| assignment.game_index == game_index)
        .collect::<Vec<_>>();
    if assignments.len() != 1 {
        return Err(LongitudinalError::WorkItem(work_item_id.to_owned()));
    }
    Ok(assignments)
}

fn build_work_item_summary(
    layout: &LongitudinalLayout,
    work_item_id: &str,
    contract: &LongitudinalBenchmarkContract,
    field: &LongitudinalFieldManifest,
    bindings: &Bindings,
) -> Result<LongitudinalWorkItem, LongitudinalError> {
    let assignments = assignments_for_work_item(field, work_item_id)?;
    let mut receipts = Vec::with_capacity(assignments.len());
    let mut peak_rss_bytes = 0;
    let mut maximum_swap_delta_bytes = i64::MIN;
    let mut all_clean_shutdowns = true;
    let mut all_pinecone_conservation_checks_passed = true;
    let mut summed_game_seconds = 0.0;
    let mut summed_checkpoint_load_seconds = 0.0;
    for assignment in assignments {
        let receipt: LongitudinalGameReceipt =
            read_json(&layout.receipt_path(work_item_id, assignment.game_index))?;
        validate_receipt(&receipt, contract, assignment, bindings)?;
        receipts.push(LongitudinalReceiptReference {
            game_index: assignment.game_index,
            receipt_blake3: canonical_blake3(&receipt)?,
        });
        let record = &receipt.payload;
        peak_rss_bytes = peak_rss_bytes.max(record.runtime.peak_rss_bytes);
        maximum_swap_delta_bytes = maximum_swap_delta_bytes.max(record.runtime.swap_delta_bytes);
        all_clean_shutdowns &= record.runtime.clean_shutdown;
        all_pinecone_conservation_checks_passed &= record.pinecones.conservation_holds();
        summed_game_seconds += record.elapsed_seconds;
        summed_checkpoint_load_seconds += record.runtime.checkpoint_load_seconds;
    }
    Ok(LongitudinalWorkItem {
        schema_version: LONGITUDINAL_SCHEMA_VERSION,
        schema_id: LONGITUDINAL_WORK_ITEM_SCHEMA_ID.to_owned(),
        contract_blake3: bindings.contract_blake3.clone(),
        field_blake3: bindings.field_blake3.clone(),
        contract_sha256: bindings.contract_sha256.clone(),
        field_sha256: bindings.field_sha256.clone(),
        implementation_binding: contract.implementation_binding.clone(),
        work_item_id: work_item_id.to_owned(),
        games: receipts.len(),
        receipts,
        peak_rss_bytes,
        maximum_swap_delta_bytes,
        all_clean_shutdowns,
        all_pinecone_conservation_checks_passed,
        summed_game_seconds,
        summed_checkpoint_load_seconds,
    })
}

fn validate_receipt_directory(
    layout: &LongitudinalLayout,
    work_item_id: &str,
    field: &LongitudinalFieldManifest,
) -> Result<(), LongitudinalError> {
    let expected = assignments_for_work_item(field, work_item_id)?
        .into_iter()
        .map(|assignment| format!("game-{:04}.json", assignment.game_index))
        .collect::<BTreeSet<_>>();
    for entry in fs::read_dir(layout.receipt_directory(work_item_id))? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if !entry.file_type()?.is_file() || !expected.contains(&name) {
            return Err(LongitudinalError::UnexpectedArtifact(entry.path()));
        }
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum LongitudinalError {
    #[error("longitudinal schema differs")]
    Schema,
    #[error("longitudinal contract differs from the fixed 100-game scheduler-managed protocol")]
    Contract,
    #[error("longitudinal W0 v1.1 implementation binding is invalid or drifted")]
    ImplementationBinding,
    #[error("longitudinal identity is empty, duplicated, or inconsistent")]
    Identity,
    #[error("longitudinal field does not cover exactly 100 game indices")]
    Coverage,
    #[error("longitudinal field repeats a game index or seed")]
    Duplicate,
    #[error("longitudinal assignment {0} violates its focal seat or frozen field")]
    Assignment(usize),
    #[error("longitudinal record {0} differs from its frozen assignment")]
    RecordIdentity(usize),
    #[error("longitudinal receipt {0} failed identity or payload integrity")]
    ReceiptDrift(usize),
    #[error("longitudinal work-item summary drifted for {0}")]
    WorkItemSummaryDrift(String),
    #[error("longitudinal work item must be the canonical game-NNNN identity; found {0}")]
    WorkItem(String),
    #[error("longitudinal executor failed at game {game_index}: {detail}")]
    Executor { game_index: usize, detail: String },
    #[error("longitudinal wall time must be finite and positive")]
    InvalidWallTime,
    #[error("unexpected longitudinal receipt artifact: {0}")]
    UnexpectedArtifact(PathBuf),
    #[error(transparent)]
    Campaign(#[from] FocalCampaignError),
    #[error(transparent)]
    Focal(#[from] crate::focal::FocalBenchmarkError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use std::{
        io,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;
    use crate::focal::{FOCAL_BENCHMARK_PROTOCOL_ID, FOCAL_BENCHMARK_SCHEMA_VERSION};

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(label: &str) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "cascadia-r2-map-longitudinal-{label}-{}-{nonce}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn fixture() -> (LongitudinalBenchmarkContract, LongitudinalFieldManifest) {
        let contract = LongitudinalBenchmarkContract::new(
            "campaign-v1",
            "longitudinal-r0",
            0,
            "incumbent-v1",
            "field-v1",
            vec!["historical-v1".to_owned()],
            "argmax-v1",
            "open-longitudinal-v1",
            LongitudinalBenchmarkPurpose::ExpertIterationLongitudinal,
            BenchmarkEvidenceClass::SyntheticFixture,
            implementation_binding_fixture(),
        );
        let assignments = (0..LONGITUDINAL_GAME_COUNT)
            .map(|game_index| {
                let focal_seat = (game_index % 4) as u8;
                LongitudinalGameAssignment {
                    game_index,
                    game_seed: GameSeed::from_u64(800_000 + game_index as u64),
                    focal_seat,
                    opponents: (0..4)
                        .filter(|seat| *seat != focal_seat)
                        .map(|seat| OpponentIdentity {
                            seat,
                            checkpoint_id: "historical-v1".to_owned(),
                        })
                        .collect(),
                }
            })
            .collect();
        (
            contract,
            LongitudinalFieldManifest::new("field-v1", assignments),
        )
    }

    fn implementation_binding_fixture() -> R2MapImplementationBinding {
        R2MapImplementationBinding::new(
            "11".repeat(32),
            "22".repeat(32),
            "33".repeat(32),
            "44".repeat(32),
            "55".repeat(32),
            "66".repeat(32),
            "88".repeat(32),
            "99".repeat(32),
            "aa".repeat(32),
            "bb".repeat(32),
            "cc".repeat(32),
            "77".repeat(32),
            "r2-map-open-reference-performance-100-v1".to_owned(),
        )
        .unwrap()
    }

    fn response(request: &FocalGameRequest) -> FocalGameRecord {
        let total = 80 + (request.identity.pair_index % 21) as u16;
        let remaining = total % 3;
        let habitat = [4; 5];
        let wildlife_total = total - habitat.iter().sum::<u16>() - remaining;
        let mut wildlife = [wildlife_total / 5; 5];
        for value in wildlife.iter_mut().take(usize::from(wildlife_total % 5)) {
            *value += 1;
        }
        FocalGameRecord {
            schema_version: FOCAL_BENCHMARK_SCHEMA_VERSION,
            protocol_id: FOCAL_BENCHMARK_PROTOCOL_ID.to_owned(),
            identity: request.identity.clone(),
            game_seed: request.game_seed,
            focal_seat: request.focal_seat,
            final_state_hash: [request.identity.pair_index as u8; 32],
            replay_blake3: format!("{:064x}", request.identity.pair_index + 1),
            score: ScoreBreakdown {
                habitat,
                wildlife,
                nature_tokens: remaining,
                habitat_bonus: [0; 5],
                base_total: total,
                total,
            },
            pinecones: PineconeObservation {
                earned: remaining,
                independent_draft_spend: 0,
                paid_wipe_spend: 0,
                total_spend: 0,
                remaining,
                free_replacements: 0,
            },
            focal_decision_seconds: vec![0.001; 20],
            elapsed_seconds: 0.02,
            runtime: FocalRuntimeObservation {
                checkpoint_load_seconds: 0.0,
                peak_rss_bytes: 4_096,
                swap_delta_bytes: 0,
                clean_shutdown: true,
            },
        }
    }

    #[test]
    fn independent_work_items_resume_and_aggregate_into_dashboard_and_ledger_inputs() {
        let temporary = TestDirectory::new("aggregate");
        let (contract, field) = fixture();
        let layout = initialize_longitudinal_campaign(&temporary.0, &contract, &field).unwrap();
        let mut executor = |request: &FocalGameRequest| -> Result<FocalGameRecord, io::Error> {
            Ok(response(request))
        };
        for work_item_id in longitudinal_work_items(&contract).into_iter().rev() {
            let first = run_longitudinal_work_item(&layout, &work_item_id, &mut executor).unwrap();
            assert_eq!(first.assigned_games, 1);
            assert_eq!(first.executed_games, 1);
            let resumed =
                run_longitudinal_work_item(&layout, &work_item_id, &mut executor).unwrap();
            assert_eq!(resumed.executed_games, 0);
            assert_eq!(resumed.resumed_games, 1);
        }
        let report = aggregate_longitudinal_campaign(&layout, 2.0).unwrap();
        assert_eq!(report.games, 100);
        assert!(!report.strength_claim_authorized);
        assert_eq!(report.work_items.len(), LONGITUDINAL_GAME_COUNT);
        assert_eq!(report.work_items[0].work_item_id, "game-0000");
        assert_eq!(
            read_json::<serde_json::Value>(&layout.dashboard_input_path()).unwrap()["pairs_completed"],
            100
        );
        assert_eq!(
            read_json::<serde_json::Value>(&layout.ledger_feed_path()).unwrap()["status"],
            "completed"
        );
    }

    #[test]
    fn field_drift_is_rejected_before_writing_artifacts() {
        let temporary = TestDirectory::new("field-drift");
        let (contract, mut field) = fixture();
        field.assignments[2].focal_seat = 3;
        assert!(matches!(
            initialize_longitudinal_campaign(&temporary.0, &contract, &field),
            Err(LongitudinalError::Assignment(2))
        ));
        assert!(!temporary.0.join("contract.json").exists());
    }

    #[test]
    fn payload_tampering_and_extra_receipts_fail_closed() {
        let temporary = TestDirectory::new("tamper");
        let (contract, field) = fixture();
        let layout = initialize_longitudinal_campaign(&temporary.0, &contract, &field).unwrap();
        run_longitudinal_work_item(
            &layout,
            "game-0000",
            &mut |request: &FocalGameRequest| -> Result<FocalGameRecord, io::Error> {
                Ok(response(request))
            },
        )
        .unwrap();
        let receipt_path = layout.receipt_path("game-0000", 0);
        let original_receipt = fs::read(&receipt_path).unwrap();
        let mut binding_receipt: LongitudinalGameReceipt = read_json(&receipt_path).unwrap();
        binding_receipt
            .implementation_binding
            .maximum_width_panel_sha256 = binding_receipt
            .implementation_binding
            .replay_pinecone_panel_sha256
            .clone();
        fs::write(
            &receipt_path,
            serde_json::to_vec_pretty(&binding_receipt).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            run_longitudinal_work_item(
                &layout,
                "game-0000",
                &mut |request: &FocalGameRequest| -> Result<FocalGameRecord, io::Error> {
                    Ok(response(request))
                },
            ),
            Err(LongitudinalError::ReceiptDrift(0))
        ));
        fs::write(&receipt_path, &original_receipt).unwrap();

        let mut receipt: serde_json::Value = read_json(&receipt_path).unwrap();
        receipt["payload"]["score"]["base_total"] = serde_json::json!(999);
        fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
        assert!(matches!(
            run_longitudinal_work_item(
                &layout,
                "game-0000",
                &mut |request: &FocalGameRequest| -> Result<FocalGameRecord, io::Error> {
                    Ok(response(request))
                },
            ),
            Err(LongitudinalError::ReceiptDrift(0))
        ));

        fs::write(
            layout
                .receipt_directory("game-0001")
                .join("unexpected.json"),
            b"{}\n",
        )
        .unwrap();
        assert!(matches!(
            run_longitudinal_work_item(
                &layout,
                "game-0001",
                &mut |request: &FocalGameRequest| -> Result<FocalGameRecord, io::Error> {
                    Ok(response(request))
                },
            ),
            Err(LongitudinalError::UnexpectedArtifact(_))
        ));
    }
}
