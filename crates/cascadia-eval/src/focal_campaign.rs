//! Resumable artifact runner for the R2-MAP focal benchmark.
//!
//! Model serving is injected through [`FocalGameExecutor`]. This module owns
//! immutable campaign identity, pair-level atomicity, exact resume, work-item
//! summaries, and final report artifacts without depending on MLX.

use std::{
    collections::{BTreeSet, HashSet},
    error::Error,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use cascadia_game::GameSeed;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::focal::{
    BenchmarkStage, DevelopmentComparisonReport, FocalBenchmarkError, FocalGameRecord,
    FocalRecordIdentity, OpponentIdentity, PairArm, PromotionGates, StrengthBlindedSmokeReport,
    aggregate_development_comparison, aggregate_strength_blinded_smoke, validate_focal_pair,
    validate_focal_record,
};
use crate::r2_map_binding::R2MapImplementationBinding;

pub const FOCAL_CAMPAIGN_SCHEMA_VERSION: u16 = 4;
pub const FOCAL_CAMPAIGN_CONTRACT_SCHEMA_ID: &str = "cascadia.r2-map.focal-contract.v4";
pub const OPPONENT_FIELD_SCHEMA_ID: &str = "cascadia.r2-map.opponent-field.v4";
pub const PAIR_RECEIPT_SCHEMA_ID: &str = "cascadia.r2-map.focal-pair-receipt.v4";
pub const WORK_ITEM_SCHEMA_ID: &str = "cascadia.r2-map.focal-work-item.v4";
pub const CAMPAIGN_REPORT_SCHEMA_ID: &str = "cascadia.r2-map.focal-report.v4";

pub(crate) static TEMPORARY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum ExecutionPartition {
    SchedulerManagedPairs,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FocalBenchmarkContract {
    pub schema_version: u16,
    pub schema_id: String,
    pub campaign_id: String,
    pub benchmark_id: String,
    pub iteration: u32,
    pub stage: BenchmarkStage,
    pub pair_count: usize,
    pub execution_partition: ExecutionPartition,
    pub candidate_checkpoint_id: String,
    pub control_checkpoint_id: String,
    pub opponent_field_manifest_id: String,
    pub inference_settings_id: String,
    pub implementation_binding: R2MapImplementationBinding,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FocalBenchmarkIdentities {
    pub campaign_id: String,
    pub benchmark_id: String,
    pub candidate_checkpoint_id: String,
    pub control_checkpoint_id: String,
    pub opponent_field_manifest_id: String,
    pub inference_settings_id: String,
}

impl FocalBenchmarkIdentities {
    pub fn new(
        campaign_id: impl Into<String>,
        benchmark_id: impl Into<String>,
        candidate_checkpoint_id: impl Into<String>,
        control_checkpoint_id: impl Into<String>,
        opponent_field_manifest_id: impl Into<String>,
        inference_settings_id: impl Into<String>,
    ) -> Self {
        Self {
            campaign_id: campaign_id.into(),
            benchmark_id: benchmark_id.into(),
            candidate_checkpoint_id: candidate_checkpoint_id.into(),
            control_checkpoint_id: control_checkpoint_id.into(),
            opponent_field_manifest_id: opponent_field_manifest_id.into(),
            inference_settings_id: inference_settings_id.into(),
        }
    }
}

impl FocalBenchmarkContract {
    pub fn new(
        iteration: u32,
        stage: BenchmarkStage,
        identities: FocalBenchmarkIdentities,
        implementation_binding: R2MapImplementationBinding,
    ) -> Self {
        Self {
            schema_version: FOCAL_CAMPAIGN_SCHEMA_VERSION,
            schema_id: FOCAL_CAMPAIGN_CONTRACT_SCHEMA_ID.to_owned(),
            campaign_id: identities.campaign_id,
            benchmark_id: identities.benchmark_id,
            iteration,
            stage,
            pair_count: stage.expected_pairs(),
            execution_partition: ExecutionPartition::SchedulerManagedPairs,
            candidate_checkpoint_id: identities.candidate_checkpoint_id,
            control_checkpoint_id: identities.control_checkpoint_id,
            opponent_field_manifest_id: identities.opponent_field_manifest_id,
            inference_settings_id: identities.inference_settings_id,
            implementation_binding,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FocalPairAssignment {
    pub pair_index: usize,
    pub game_seed: GameSeed,
    pub seed_domain_id: String,
    pub focal_seat: u8,
    pub opponents: Vec<OpponentIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpponentFieldManifest {
    pub schema_version: u16,
    pub schema_id: String,
    pub manifest_id: String,
    pub assignments: Vec<FocalPairAssignment>,
}

impl OpponentFieldManifest {
    pub fn new(manifest_id: impl Into<String>, assignments: Vec<FocalPairAssignment>) -> Self {
        Self {
            schema_version: FOCAL_CAMPAIGN_SCHEMA_VERSION,
            schema_id: OPPONENT_FIELD_SCHEMA_ID.to_owned(),
            manifest_id: manifest_id.into(),
            assignments,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FocalGameRequest {
    pub benchmark_id: String,
    pub implementation_binding: R2MapImplementationBinding,
    pub identity: FocalRecordIdentity,
    pub game_seed: GameSeed,
    pub focal_seat: u8,
}

pub trait FocalGameExecutor {
    type Error: Error + Send + Sync + 'static;

    fn execute(&mut self, request: &FocalGameRequest) -> Result<FocalGameRecord, Self::Error>;

    /// Finalize item-local services before a durable work-item summary exists.
    fn finish(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl<F, E> FocalGameExecutor for F
where
    F: FnMut(&FocalGameRequest) -> Result<FocalGameRecord, E>,
    E: Error + Send + Sync + 'static,
{
    type Error = E;

    fn execute(&mut self, request: &FocalGameRequest) -> Result<FocalGameRecord, Self::Error> {
        self(request)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairReceiptPayload {
    pub pair_index: usize,
    /// The candidate is the treatment arm in persisted protocol terminology.
    #[serde(rename = "treatment")]
    pub candidate: FocalGameRecord,
    pub control: FocalGameRecord,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairReceipt {
    pub schema_version: u16,
    pub schema_id: String,
    pub contract_blake3: String,
    pub opponent_field_blake3: String,
    pub contract_sha256: String,
    pub opponent_field_sha256: String,
    pub implementation_binding: R2MapImplementationBinding,
    pub payload_blake3: String,
    pub payload: PairReceiptPayload,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairReceiptReference {
    pub pair_index: usize,
    pub receipt_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkItemSummary {
    pub schema_version: u16,
    pub schema_id: String,
    pub contract_blake3: String,
    pub opponent_field_blake3: String,
    pub contract_sha256: String,
    pub opponent_field_sha256: String,
    pub implementation_binding: R2MapImplementationBinding,
    pub work_item_id: String,
    pub stage: BenchmarkStage,
    pub pairs: usize,
    pub physical_games: usize,
    pub pair_receipts: Vec<PairReceiptReference>,
    pub peak_rss_bytes: u64,
    pub maximum_swap_delta_bytes: i64,
    pub all_clean_shutdowns: bool,
    pub all_pinecone_conservation_checks_passed: bool,
    pub summed_game_seconds: f64,
    pub summed_checkpoint_load_seconds: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkItemRunOutcome {
    pub assigned_pairs: usize,
    pub executed_pairs: usize,
    pub resumed_pairs: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "statistics", rename_all = "kebab-case")]
pub enum FocalCampaignStatistics {
    StrengthBlindedSmoke(StrengthBlindedSmokeReport),
    Development(Box<DevelopmentComparisonReport>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FocalCampaignReport {
    pub schema_version: u16,
    pub schema_id: String,
    pub campaign_id: String,
    pub benchmark_id: String,
    pub iteration: u32,
    pub contract_blake3: String,
    pub opponent_field_blake3: String,
    pub contract_sha256: String,
    pub opponent_field_sha256: String,
    pub implementation_binding: R2MapImplementationBinding,
    pub work_items: Vec<WorkItemSummary>,
    pub result: FocalCampaignStatistics,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CampaignBindings {
    contract_blake3: String,
    opponent_field_blake3: String,
    contract_sha256: String,
    opponent_field_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CampaignReportArtifacts {
    pub json: PathBuf,
    pub markdown: PathBuf,
}

#[derive(Debug, Clone)]
pub struct FocalCampaignLayout {
    root: PathBuf,
}

impl FocalCampaignLayout {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn contract_path(&self) -> PathBuf {
        self.root.join("contract.json")
    }

    pub fn opponent_field_path(&self) -> PathBuf {
        self.root.join("opponent-field.json")
    }

    pub fn receipt_directory(&self, work_item_id: &str) -> PathBuf {
        self.root.join("receipts").join(work_item_id)
    }

    pub fn pair_receipt_path(&self, work_item_id: &str, pair_index: usize) -> PathBuf {
        self.receipt_directory(work_item_id)
            .join(format!("pair-{pair_index:04}.json"))
    }

    pub fn work_item_summary_path(&self, work_item_id: &str) -> PathBuf {
        self.root
            .join("work-item-summaries")
            .join(format!("{work_item_id}.json"))
    }

    pub fn report_json_path(&self) -> PathBuf {
        self.root.join("reports/focal-benchmark.json")
    }

    pub fn report_markdown_path(&self) -> PathBuf {
        self.root.join("reports/focal-benchmark.md")
    }

    pub fn dashboard_input_path(&self) -> PathBuf {
        self.root.join("projections/dashboard-benchmark.json")
    }

    pub fn ledger_feed_path(&self) -> PathBuf {
        self.root.join("projections/ledger-experiment.json")
    }
}

pub fn initialize_focal_campaign(
    root: impl Into<PathBuf>,
    contract: &FocalBenchmarkContract,
    opponent_field: &OpponentFieldManifest,
) -> Result<FocalCampaignLayout, FocalCampaignError> {
    validate_contract_and_field(contract, opponent_field)?;
    let layout = FocalCampaignLayout::new(root);
    let mut directories = vec![
        layout.root.clone(),
        layout.root.join("work-item-summaries"),
        layout.root.join("reports"),
        layout.root.join("projections"),
    ];
    directories.extend(
        campaign_work_items(contract)
            .into_iter()
            .map(|work_item_id| layout.receipt_directory(&work_item_id)),
    );
    for directory in directories {
        fs::create_dir_all(directory)?;
    }
    write_immutable_json(&layout.contract_path(), contract)?;
    write_immutable_json(&layout.opponent_field_path(), opponent_field)?;
    Ok(layout)
}

pub fn run_focal_work_item<E: FocalGameExecutor>(
    layout: &FocalCampaignLayout,
    work_item_id: &str,
    executor: &mut E,
) -> Result<WorkItemRunOutcome, FocalCampaignError> {
    let pair_index = parse_work_item_id(work_item_id)?;
    let (contract, opponent_field) = read_campaign_inputs(layout)?;
    if pair_index >= contract.pair_count {
        return Err(FocalCampaignError::WorkItem(work_item_id.to_owned()));
    }
    validate_receipt_directory(layout, work_item_id, &opponent_field)?;
    let bindings = campaign_bindings(layout, &contract, &opponent_field)?;
    let assignments = assignments_for_work_item(&opponent_field, work_item_id)?;
    let mut executed_pairs = 0;
    let mut resumed_pairs = 0;

    for assignment in &assignments {
        let path = layout.pair_receipt_path(work_item_id, assignment.pair_index);
        if path.exists() {
            let receipt: PairReceipt = read_json(&path)?;
            validate_pair_receipt(&receipt, &contract, assignment, &bindings)?;
            resumed_pairs += 1;
            continue;
        }

        let candidate_request = game_request(&contract, assignment, PairArm::Candidate);
        let control_request = game_request(&contract, assignment, PairArm::Control);
        // Alternate execution order deterministically so process warm-up cannot
        // remain confounded with one arm. Receipts are normalized to candidate
        // then control regardless of physical execution order.
        let execute = |executor: &mut E,
                       request: &FocalGameRequest|
         -> Result<FocalGameRecord, FocalCampaignError> {
            executor
                .execute(request)
                .map_err(|source| FocalCampaignError::Executor {
                    pair_index: assignment.pair_index,
                    arm: request.identity.arm,
                    detail: source.to_string(),
                })
        };
        let (candidate, control) = if assignment.pair_index.is_multiple_of(2) {
            (
                execute(executor, &candidate_request)?,
                execute(executor, &control_request)?,
            )
        } else {
            let control = execute(executor, &control_request)?;
            let candidate = execute(executor, &candidate_request)?;
            (candidate, control)
        };
        validate_response(&candidate_request, &candidate)?;
        validate_response(&control_request, &control)?;
        validate_focal_pair(&candidate, &control)?;
        let payload = PairReceiptPayload {
            pair_index: assignment.pair_index,
            candidate,
            control,
        };
        let receipt = PairReceipt {
            schema_version: FOCAL_CAMPAIGN_SCHEMA_VERSION,
            schema_id: PAIR_RECEIPT_SCHEMA_ID.to_owned(),
            contract_blake3: bindings.contract_blake3.clone(),
            opponent_field_blake3: bindings.opponent_field_blake3.clone(),
            contract_sha256: bindings.contract_sha256.clone(),
            opponent_field_sha256: bindings.opponent_field_sha256.clone(),
            implementation_binding: contract.implementation_binding.clone(),
            payload_blake3: canonical_blake3(&payload)?,
            payload,
        };
        write_immutable_json(&path, &receipt)?;
        executed_pairs += 1;
    }

    validate_receipt_directory(layout, work_item_id, &opponent_field)?;
    executor
        .finish()
        .map_err(|source| FocalCampaignError::Finalize {
            work_item: work_item_id.to_owned(),
            detail: source.to_string(),
        })?;
    let summary =
        build_work_item_summary(layout, work_item_id, &contract, &opponent_field, &bindings)?;
    write_immutable_json(&layout.work_item_summary_path(work_item_id), &summary)?;
    Ok(WorkItemRunOutcome {
        assigned_pairs: assignments.len(),
        executed_pairs,
        resumed_pairs,
    })
}

pub fn aggregate_focal_campaign(
    layout: &FocalCampaignLayout,
    wall_seconds: f64,
    promotion_gates: PromotionGates,
) -> Result<(FocalCampaignReport, CampaignReportArtifacts), FocalCampaignError> {
    let (contract, opponent_field) = read_campaign_inputs(layout)?;
    let bindings = campaign_bindings(layout, &contract, &opponent_field)?;
    let mut work_items = Vec::new();
    let mut records = Vec::with_capacity(contract.pair_count * 2);

    for work_item_id in campaign_work_items(&contract) {
        validate_receipt_directory(layout, &work_item_id, &opponent_field)?;
        let stored: WorkItemSummary = read_json(&layout.work_item_summary_path(&work_item_id))?;
        let recomputed =
            build_work_item_summary(layout, &work_item_id, &contract, &opponent_field, &bindings)?;
        if stored != recomputed {
            return Err(FocalCampaignError::WorkItemSummaryDrift(work_item_id));
        }
        for assignment in assignments_for_work_item(&opponent_field, &stored.work_item_id)? {
            let receipt: PairReceipt =
                read_json(&layout.pair_receipt_path(&stored.work_item_id, assignment.pair_index))?;
            validate_pair_receipt(&receipt, &contract, assignment, &bindings)?;
            records.push(receipt.payload.candidate);
            records.push(receipt.payload.control);
        }
        work_items.push(stored);
    }
    records.sort_by_key(|record| {
        (
            record.identity.pair_index,
            match record.identity.arm {
                PairArm::Candidate => 0,
                PairArm::Control => 1,
            },
        )
    });

    let result = match contract.stage {
        BenchmarkStage::StrengthBlindedSmoke => FocalCampaignStatistics::StrengthBlindedSmoke(
            aggregate_strength_blinded_smoke(&records, wall_seconds)?,
        ),
        BenchmarkStage::Development => FocalCampaignStatistics::Development(Box::new(
            aggregate_development_comparison(&records, wall_seconds, promotion_gates)?,
        )),
    };
    work_items.sort_by(|left, right| left.work_item_id.cmp(&right.work_item_id));
    let report = FocalCampaignReport {
        schema_version: FOCAL_CAMPAIGN_SCHEMA_VERSION,
        schema_id: CAMPAIGN_REPORT_SCHEMA_ID.to_owned(),
        campaign_id: contract.campaign_id,
        benchmark_id: contract.benchmark_id,
        iteration: contract.iteration,
        contract_blake3: bindings.contract_blake3,
        opponent_field_blake3: bindings.opponent_field_blake3,
        contract_sha256: bindings.contract_sha256,
        opponent_field_sha256: bindings.opponent_field_sha256,
        implementation_binding: contract.implementation_binding,
        work_items,
        result,
    };
    let artifacts = CampaignReportArtifacts {
        json: layout.report_json_path(),
        markdown: layout.report_markdown_path(),
    };
    write_immutable_json(&artifacts.json, &report)?;
    write_immutable_bytes(
        &artifacts.markdown,
        render_campaign_markdown(&report).as_bytes(),
    )?;
    write_immutable_json(
        &layout.dashboard_input_path(),
        &focal_dashboard_input(&report),
    )?;
    write_immutable_json(&layout.ledger_feed_path(), &focal_ledger_feed(&report))?;
    Ok((report, artifacts))
}

/// Exact compact object consumed by the existing dashboard publisher.
pub fn focal_dashboard_input(report: &FocalCampaignReport) -> serde_json::Value {
    let compact = |value: &crate::focal::IntegerDistribution| {
        serde_json::json!({
            "mean": value.mean,
            "p10": value.p10,
            "p50": value.p50,
            "p90": value.p90,
        })
    };
    let resources = match &report.result {
        FocalCampaignStatistics::StrengthBlindedSmoke(value) => (
            value.pairs,
            value.games_per_second,
            value.peak_rss_bytes,
            value.maximum_swap_delta_bytes,
        ),
        FocalCampaignStatistics::Development(value) => (
            value.pairs,
            value.games_per_second,
            report
                .work_items
                .iter()
                .map(|item| item.peak_rss_bytes)
                .max()
                .unwrap_or(0),
            report
                .work_items
                .iter()
                .map(|item| item.maximum_swap_delta_bytes)
                .max()
                .unwrap_or(0),
        ),
    };
    let (focal, paired_delta, classification, stage) = match &report.result {
        FocalCampaignStatistics::StrengthBlindedSmoke(_) => (
            serde_json::Value::Null,
            serde_json::Value::Null,
            "pending",
            "strength-blinded-smoke",
        ),
        FocalCampaignStatistics::Development(value) => {
            let focal = serde_json::json!({
                "base_total": compact(&value.candidate.base_total),
                "animals": {
                    "aggregate": compact(&value.candidate.animals.aggregate_wildlife),
                    "bear": compact(&value.candidate.animals.bear),
                    "elk": compact(&value.candidate.animals.elk),
                    "salmon": compact(&value.candidate.animals.salmon),
                    "hawk": compact(&value.candidate.animals.hawk),
                    "fox": compact(&value.candidate.animals.fox),
                },
                "habitat": {
                    "aggregate": compact(&value.candidate.terrains.aggregate_habitat),
                    "mountain": compact(&value.candidate.terrains.mountain),
                    "forest": compact(&value.candidate.terrains.forest),
                    "prairie": compact(&value.candidate.terrains.prairie),
                    "wetland": compact(&value.candidate.terrains.wetland),
                    "river": compact(&value.candidate.terrains.river),
                },
                "pinecones": {
                    "earned": compact(&value.candidate.pinecones.earned),
                    "independent_draft_spend": compact(&value.candidate.pinecones.independent_draft_spend),
                    "paid_wipe_spend": compact(&value.candidate.pinecones.paid_wipe_spend),
                    "total_spend": compact(&value.candidate.pinecones.total_spend),
                    "remaining": compact(&value.candidate.pinecones.remaining),
                    "free_replacements": compact(&value.candidate.pinecones.free_replacements),
                },
            });
            let delta = serde_json::json!({
                "mean": value.paired_delta.base_total.mean,
                "confidence_95": value.paired_delta.base_total.confidence_95,
            });
            let classification = match value.classification {
                crate::focal::PromotionClassification::Promote => "promote",
                crate::focal::PromotionClassification::Reject => "reject",
                crate::focal::PromotionClassification::Inconclusive => "inconclusive",
            };
            (focal, delta, classification, "fixed-250-development")
        }
    };
    serde_json::json!({
        "active": false,
        "stage": stage,
        "pairs_completed": resources.0,
        "pairs_total": resources.0,
        "eta_seconds": 0.0,
        "throughput_games_per_second": resources.1,
        "peak_rss_bytes": resources.2,
        "swap_delta_bytes": resources.3,
        "focal": focal,
        "paired_delta": paired_delta,
        "classification": classification,
    })
}

/// Deterministic experiment object accepted by `cluster_experiment_ledger.py`.
/// The central controller stamps import time, preserving reproducible items.
pub fn focal_ledger_feed(report: &FocalCampaignReport) -> serde_json::Value {
    let work_items = report
        .work_items
        .iter()
        .map(|item| item.work_item_id.clone())
        .collect::<Vec<_>>();
    let (summary, outcome, metrics, criteria) = match &report.result {
        FocalCampaignStatistics::StrengthBlindedSmoke(value) => (
            format!(
                "Completed {} strength-blinded integrity pairs; score outputs remain sealed.",
                value.pairs
            ),
            if value.all_clean_shutdowns && value.all_pinecone_conservation_checks_passed {
                "passed"
            } else {
                "invalid"
            },
            vec![
                serde_json::json!({"label": "Pairs", "value": value.pairs.to_string(), "tone": "neutral"}),
                serde_json::json!({"label": "Physical games", "value": value.physical_games.to_string(), "tone": "neutral"}),
                serde_json::json!({"label": "Games/second", "value": format!("{:.3}", value.games_per_second), "tone": "neutral"}),
            ],
            vec![
                serde_json::json!({"label": "Strength outputs blinded", "passed": value.strength_outputs_blinded, "observed": value.strength_outputs_blinded.to_string()}),
                serde_json::json!({"label": "Pinecone conservation", "passed": value.all_pinecone_conservation_checks_passed, "observed": value.all_pinecone_conservation_checks_passed.to_string()}),
            ],
        ),
        FocalCampaignStatistics::Development(value) => {
            let classification = match value.classification {
                crate::focal::PromotionClassification::Promote => "promote",
                crate::focal::PromotionClassification::Reject => "reject",
                crate::focal::PromotionClassification::Inconclusive => "inconclusive",
            };
            (
                format!(
                    "Fixed-250 paired gate classified {classification} with mean delta {:+.3}.",
                    value.paired_delta.base_total.mean
                ),
                if classification == "promote" {
                    "passed"
                } else if classification == "reject" {
                    "failed"
                } else {
                    "inconclusive"
                },
                vec![
                    serde_json::json!({"label": "Pairs", "value": value.pairs.to_string(), "tone": "neutral"}),
                    serde_json::json!({"label": "Candidate mean", "value": format!("{:.3}", value.candidate.base_total.mean), "tone": "neutral"}),
                    serde_json::json!({"label": "Paired delta", "value": format!("{:+.3}", value.paired_delta.base_total.mean), "tone": "neutral"}),
                    serde_json::json!({"label": "Classification", "value": classification, "tone": if classification == "promote" { "good" } else if classification == "reject" { "bad" } else { "warn" }}),
                ],
                vec![
                    serde_json::json!({"label": "Complete fixed-250 coverage", "passed": value.pairs == 250, "observed": value.pairs.to_string()}),
                ],
            )
        }
    };
    serde_json::json!({
        "id": format!("r2-map-focal-{}", report.benchmark_id),
        "title": "R2-MAP focal candidate gate",
        "hypothesis": "The candidate improves focal score against the frozen historical field.",
        "summary": summary,
        "status": "completed",
        "outcome": outcome,
        "verdict": null,
        "plan_section": "W5",
        "started_unix_ms": 0,
        "completed_unix_ms": 0,
        "updated_unix_ms": 0,
        "work_items": work_items,
        "tags": ["r2-map", "focal", "candidate-gate"],
        "task_ids": [],
        "metrics": metrics,
        "criteria": criteria,
        "notes": ["The controller stamps import time and binds this feed to the hash-verified focal report."],
        "artifacts": [{"label": "R2-MAP plan", "path": "docs/v2/R2_MAP_EXPERT_ITERATION_RESEARCH_PLAN.md"}],
    })
}

pub fn render_campaign_markdown(report: &FocalCampaignReport) -> String {
    let mut output = format!(
        "# R2-MAP Focal Benchmark\n\n- Campaign: `{}`\n- Benchmark: `{}`\n- Iteration: {}\n- Contract BLAKE3: `{}`\n- Opponent field BLAKE3: `{}`\n- Contract file SHA-256: `{}`\n- Opponent field file SHA-256: `{}`\n- W0 v1.1 registration SHA-256: `{}`\n- Source bundle SHA-256: `{}`\n- Serving protocol schema SHA-256: `{}`\n- Model schema SHA-256: `{}`\n",
        report.campaign_id,
        report.benchmark_id,
        report.iteration,
        report.contract_blake3,
        report.opponent_field_blake3,
        report.contract_sha256,
        report.opponent_field_sha256,
        report.implementation_binding.w0_registration_sha256,
        report.implementation_binding.source_bundle_sha256,
        report.implementation_binding.serving_protocol_schema_sha256,
        report.implementation_binding.model_schema_sha256,
    );
    match &report.result {
        FocalCampaignStatistics::StrengthBlindedSmoke(smoke) => {
            output.push_str(&format!(
                "- Stage: strength-blinded smoke\n- Strength outputs blinded: **{}**\n- Pairs / physical games: {} / {}\n- Runtime: {:.3}s ({:.3} games/s)\n- Peak RSS: {} bytes\n- Maximum swap delta: {} bytes\n- Clean shutdowns: {}\n- Pinecone conservation: {}\n",
                smoke.strength_outputs_blinded,
                smoke.pairs,
                smoke.physical_games,
                smoke.wall_seconds,
                smoke.games_per_second,
                smoke.peak_rss_bytes,
                smoke.maximum_swap_delta_bytes,
                smoke.all_clean_shutdowns,
                smoke.all_pinecone_conservation_checks_passed,
            ));
        }
        FocalCampaignStatistics::Development(development) => {
            let classification = match development.classification {
                crate::focal::PromotionClassification::Promote => "positive",
                crate::focal::PromotionClassification::Reject => "negative",
                crate::focal::PromotionClassification::Inconclusive => "inconclusive",
            };
            let peak_rss_bytes = report
                .work_items
                .iter()
                .map(|item| item.peak_rss_bytes)
                .max()
                .unwrap_or(0);
            let maximum_swap_delta_bytes = report
                .work_items
                .iter()
                .map(|item| item.maximum_swap_delta_bytes)
                .max()
                .unwrap_or(0);
            let clean_shutdowns = report
                .work_items
                .iter()
                .all(|item| item.all_clean_shutdowns);
            let pinecone_conservation = report
                .work_items
                .iter()
                .all(|item| item.all_pinecone_conservation_checks_passed);
            output.push_str(&format!(
                "- Stage: fixed-250 development comparison\n- Classification: **{classification}**\n- Pairs / physical games: {} / {}\n- Candidate/control mean: {:.3} / {:.3}\n- Paired delta: {:+.3}; SE {:.3}; 95% CI [{:+.3}, {:+.3}]\n- Candidate P10/P50/P90: {:.1} / {:.1} / {:.1}\n- Control P10/P50/P90: {:.1} / {:.1} / {:.1}\n- Wins/ties/losses: {} / {} / {}\n- Candidate/control distance from 100: {:+.3} / {:+.3}\n- Runtime: {:.3}s ({:.6} games/s)\n- Peak RSS / maximum swap delta: {} / {} bytes\n- Clean shutdowns / Pinecone conservation: {} / {}\n- Candidate/control/delta decision latency mean: {:.3} / {:.3} / {:+.3} ms\n- Candidate/control checkpoint load mean: {:.6} / {:.6}s\n\n",
                development.pairs,
                development.physical_games,
                development.candidate.base_total.mean,
                development.control.base_total.mean,
                development.paired_delta.base_total.mean,
                development.paired_delta.base_total.standard_error,
                development.paired_delta.base_total.confidence_95[0],
                development.paired_delta.base_total.confidence_95[1],
                development.candidate.base_total.p10,
                development.candidate.base_total.p50,
                development.candidate.base_total.p90,
                development.control.base_total.p10,
                development.control.base_total.p50,
                development.control.base_total.p90,
                development.candidate_wins,
                development.ties,
                development.candidate_losses,
                development.candidate_distance_from_100,
                development.control_distance_from_100,
                development.wall_seconds,
                development.games_per_second,
                peak_rss_bytes,
                maximum_swap_delta_bytes,
                clean_shutdowns,
                pinecone_conservation,
                development.candidate.focal_decision_latency_milliseconds.mean,
                development.control.focal_decision_latency_milliseconds.mean,
                development
                    .paired_delta
                    .focal_decision_latency_milliseconds
                    .mean,
                development.candidate_checkpoint_load_seconds.mean,
                development.control_checkpoint_load_seconds.mean,
            ));
            output.push_str("## Score histograms\n\n");
            output.push_str(&format!(
                "- Candidate: `{}`\n- Control: `{}`\n- Paired delta: `{}`\n\n",
                serde_json::to_string(&development.candidate.base_total.histogram)
                    .expect("integer histogram is JSON-safe"),
                serde_json::to_string(&development.control.base_total.histogram)
                    .expect("integer histogram is JSON-safe"),
                serde_json::to_string(&development.paired_delta.base_total.histogram)
                    .expect("integer histogram is JSON-safe"),
            ));
            output.push_str(
                "## Score anatomy\n\n| Component | Candidate mean | C P10 | C P50 | C P90 | Control mean | N P10 | N P50 | N P90 | Delta mean | Delta SE | Delta 95% CI |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
            );
            let rows = [
                (
                    "Total",
                    &development.candidate.base_total,
                    &development.control.base_total,
                    &development.paired_delta.base_total,
                ),
                (
                    "Bear",
                    &development.candidate.animals.bear,
                    &development.control.animals.bear,
                    &development.paired_delta.animals.bear,
                ),
                (
                    "Elk",
                    &development.candidate.animals.elk,
                    &development.control.animals.elk,
                    &development.paired_delta.animals.elk,
                ),
                (
                    "Salmon",
                    &development.candidate.animals.salmon,
                    &development.control.animals.salmon,
                    &development.paired_delta.animals.salmon,
                ),
                (
                    "Hawk",
                    &development.candidate.animals.hawk,
                    &development.control.animals.hawk,
                    &development.paired_delta.animals.hawk,
                ),
                (
                    "Fox",
                    &development.candidate.animals.fox,
                    &development.control.animals.fox,
                    &development.paired_delta.animals.fox,
                ),
                (
                    "Wildlife",
                    &development.candidate.animals.aggregate_wildlife,
                    &development.control.animals.aggregate_wildlife,
                    &development.paired_delta.animals.aggregate_wildlife,
                ),
                (
                    "Mountain",
                    &development.candidate.terrains.mountain,
                    &development.control.terrains.mountain,
                    &development.paired_delta.terrains.mountain,
                ),
                (
                    "Forest",
                    &development.candidate.terrains.forest,
                    &development.control.terrains.forest,
                    &development.paired_delta.terrains.forest,
                ),
                (
                    "Prairie",
                    &development.candidate.terrains.prairie,
                    &development.control.terrains.prairie,
                    &development.paired_delta.terrains.prairie,
                ),
                (
                    "Wetland",
                    &development.candidate.terrains.wetland,
                    &development.control.terrains.wetland,
                    &development.paired_delta.terrains.wetland,
                ),
                (
                    "River",
                    &development.candidate.terrains.river,
                    &development.control.terrains.river,
                    &development.paired_delta.terrains.river,
                ),
                (
                    "Habitat",
                    &development.candidate.terrains.aggregate_habitat,
                    &development.control.terrains.aggregate_habitat,
                    &development.paired_delta.terrains.aggregate_habitat,
                ),
                (
                    "Pinecones earned",
                    &development.candidate.pinecones.earned,
                    &development.control.pinecones.earned,
                    &development.paired_delta.pinecones.earned,
                ),
                (
                    "Independent spend",
                    &development.candidate.pinecones.independent_draft_spend,
                    &development.control.pinecones.independent_draft_spend,
                    &development.paired_delta.pinecones.independent_draft_spend,
                ),
                (
                    "Paid-wipe spend",
                    &development.candidate.pinecones.paid_wipe_spend,
                    &development.control.pinecones.paid_wipe_spend,
                    &development.paired_delta.pinecones.paid_wipe_spend,
                ),
                (
                    "Total spend",
                    &development.candidate.pinecones.total_spend,
                    &development.control.pinecones.total_spend,
                    &development.paired_delta.pinecones.total_spend,
                ),
                (
                    "Pinecones remaining",
                    &development.candidate.pinecones.remaining,
                    &development.control.pinecones.remaining,
                    &development.paired_delta.pinecones.remaining,
                ),
                (
                    "Free replacements",
                    &development.candidate.pinecones.free_replacements,
                    &development.control.pinecones.free_replacements,
                    &development.paired_delta.pinecones.free_replacements,
                ),
            ];
            for (label, candidate, control, delta) in rows {
                output.push_str(&format!(
                    "| {label} | {:.3} | {:.1} | {:.1} | {:.1} | {:.3} | {:.1} | {:.1} | {:.1} | {:+.3} | {:.3} | [{:+.3}, {:+.3}] |\n",
                    candidate.mean,
                    candidate.p10,
                    candidate.p50,
                    candidate.p90,
                    control.mean,
                    control.p10,
                    control.p50,
                    control.p90,
                    delta.mean,
                    delta.standard_error,
                    delta.confidence_95[0],
                    delta.confidence_95[1],
                ));
            }
            output.push_str("\n## Work-item resources\n\n| Work item | Pairs | Games | Peak RSS bytes | Maximum swap delta bytes | Game seconds | Checkpoint-load seconds | Clean shutdown | Pinecone conservation |\n|---|---:|---:|---:|---:|---:|---:|---|---|\n");
            for item in &report.work_items {
                output.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {:.3} | {:.6} | {} | {} |\n",
                    item.work_item_id,
                    item.pairs,
                    item.physical_games,
                    item.peak_rss_bytes,
                    item.maximum_swap_delta_bytes,
                    item.summed_game_seconds,
                    item.summed_checkpoint_load_seconds,
                    item.all_clean_shutdowns,
                    item.all_pinecone_conservation_checks_passed,
                ));
            }
        }
    }
    output
}

fn read_campaign_inputs(
    layout: &FocalCampaignLayout,
) -> Result<(FocalBenchmarkContract, OpponentFieldManifest), FocalCampaignError> {
    let contract = read_json(&layout.contract_path())?;
    let opponent_field = read_json(&layout.opponent_field_path())?;
    validate_contract_and_field(&contract, &opponent_field)?;
    Ok((contract, opponent_field))
}

fn validate_contract_and_field(
    contract: &FocalBenchmarkContract,
    opponent_field: &OpponentFieldManifest,
) -> Result<(), FocalCampaignError> {
    if contract.schema_version != FOCAL_CAMPAIGN_SCHEMA_VERSION
        || contract.schema_id != FOCAL_CAMPAIGN_CONTRACT_SCHEMA_ID
    {
        return Err(FocalCampaignError::ContractSchema);
    }
    if opponent_field.schema_version != FOCAL_CAMPAIGN_SCHEMA_VERSION
        || opponent_field.schema_id != OPPONENT_FIELD_SCHEMA_ID
    {
        return Err(FocalCampaignError::OpponentFieldSchema);
    }
    if contract.pair_count != contract.stage.expected_pairs() {
        return Err(FocalCampaignError::PairCount {
            expected: contract.stage.expected_pairs(),
            actual: contract.pair_count,
        });
    }
    contract
        .implementation_binding
        .validate()
        .map_err(|_| FocalCampaignError::ImplementationBinding)?;
    if contract.execution_partition != ExecutionPartition::SchedulerManagedPairs {
        return Err(FocalCampaignError::ContractSchema);
    }
    if contract.opponent_field_manifest_id != opponent_field.manifest_id {
        return Err(FocalCampaignError::OpponentFieldIdentityDrift);
    }
    for (label, value) in [
        ("campaign id", contract.campaign_id.as_str()),
        ("benchmark id", contract.benchmark_id.as_str()),
        (
            "candidate checkpoint",
            contract.candidate_checkpoint_id.as_str(),
        ),
        (
            "control checkpoint",
            contract.control_checkpoint_id.as_str(),
        ),
        (
            "opponent field",
            contract.opponent_field_manifest_id.as_str(),
        ),
        (
            "inference settings",
            contract.inference_settings_id.as_str(),
        ),
    ] {
        if value.is_empty() {
            return Err(FocalCampaignError::EmptyIdentity(label));
        }
    }
    if opponent_field.assignments.len() != contract.pair_count {
        return Err(FocalCampaignError::PairCount {
            expected: contract.pair_count,
            actual: opponent_field.assignments.len(),
        });
    }
    let mut indices = BTreeSet::new();
    let mut seeds = HashSet::new();
    let mut seed_domains = HashSet::new();
    for assignment in &opponent_field.assignments {
        if !indices.insert(assignment.pair_index) {
            return Err(FocalCampaignError::DuplicateAssignment(
                assignment.pair_index,
            ));
        }
        if !seeds.insert(assignment.game_seed) {
            return Err(FocalCampaignError::DuplicateGameSeed(assignment.pair_index));
        }
        if assignment.seed_domain_id.is_empty()
            || !seed_domains.insert(assignment.seed_domain_id.as_str())
        {
            return Err(FocalCampaignError::DuplicateSeedDomain(
                assignment.seed_domain_id.clone(),
            ));
        }
        if assignment.focal_seat != (assignment.pair_index % 4) as u8 {
            return Err(FocalCampaignError::AssignmentFocalSeat(
                assignment.pair_index,
            ));
        }
        let request = game_request(contract, assignment, PairArm::Candidate);
        validate_identity_shape(&request.identity, request.focal_seat)?;
        if assignment
            .opponents
            .iter()
            .any(|opponent| opponent.checkpoint_id == contract.candidate_checkpoint_id)
        {
            return Err(FocalCampaignError::CandidateInOpponentSeat(
                assignment.pair_index,
            ));
        }
    }
    let expected = (0..contract.pair_count).collect::<BTreeSet<_>>();
    if indices != expected {
        return Err(FocalCampaignError::AssignmentCoverage);
    }
    Ok(())
}

fn validate_identity_shape(
    identity: &FocalRecordIdentity,
    focal_seat: u8,
) -> Result<(), FocalCampaignError> {
    if identity.opponents.len() != 3 {
        return Err(FocalCampaignError::OpponentCount(identity.opponents.len()));
    }
    let mut seats = identity
        .opponents
        .iter()
        .map(|opponent| opponent.seat)
        .collect::<Vec<_>>();
    seats.sort_unstable();
    let expected = (0..4)
        .filter(|seat| *seat != focal_seat)
        .collect::<Vec<_>>();
    if seats != expected {
        return Err(FocalCampaignError::OpponentSeats(identity.pair_index));
    }
    Ok(())
}

fn work_item_id(pair_index: usize) -> String {
    format!("pair-{pair_index:04}")
}

fn parse_work_item_id(value: &str) -> Result<usize, FocalCampaignError> {
    let Some(raw) = value.strip_prefix("pair-") else {
        return Err(FocalCampaignError::WorkItem(value.to_owned()));
    };
    if raw.len() != 4 || !raw.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(FocalCampaignError::WorkItem(value.to_owned()));
    }
    raw.parse()
        .map_err(|_| FocalCampaignError::WorkItem(value.to_owned()))
}

fn campaign_work_items(contract: &FocalBenchmarkContract) -> Vec<String> {
    (0..contract.pair_count).map(work_item_id).collect()
}

fn assignments_for_work_item<'a>(
    opponent_field: &'a OpponentFieldManifest,
    work_item: &str,
) -> Result<Vec<&'a FocalPairAssignment>, FocalCampaignError> {
    let pair_index = parse_work_item_id(work_item)?;
    let assignments = opponent_field
        .assignments
        .iter()
        .filter(|assignment| assignment.pair_index == pair_index)
        .collect::<Vec<_>>();
    if assignments.len() != 1 {
        return Err(FocalCampaignError::WorkItem(work_item.to_owned()));
    }
    Ok(assignments)
}

fn game_request(
    contract: &FocalBenchmarkContract,
    assignment: &FocalPairAssignment,
    arm: PairArm,
) -> FocalGameRequest {
    let focal_checkpoint_id = match arm {
        PairArm::Candidate => &contract.candidate_checkpoint_id,
        PairArm::Control => &contract.control_checkpoint_id,
    };
    FocalGameRequest {
        benchmark_id: contract.benchmark_id.clone(),
        implementation_binding: contract.implementation_binding.clone(),
        identity: FocalRecordIdentity {
            stage: contract.stage,
            pair_index: assignment.pair_index,
            arm,
            focal_checkpoint_id: focal_checkpoint_id.clone(),
            opponents: assignment.opponents.clone(),
            field_manifest_id: contract.opponent_field_manifest_id.clone(),
            inference_settings_id: contract.inference_settings_id.clone(),
        },
        game_seed: assignment.game_seed,
        focal_seat: assignment.focal_seat,
    }
}

fn validate_response(
    request: &FocalGameRequest,
    response: &FocalGameRecord,
) -> Result<(), FocalCampaignError> {
    validate_focal_record(response)?;
    if response.identity != request.identity
        || response.game_seed != request.game_seed
        || response.focal_seat != request.focal_seat
    {
        return Err(FocalCampaignError::ExecutorIdentityDrift {
            pair_index: request.identity.pair_index,
            arm: request.identity.arm,
        });
    }
    Ok(())
}

fn validate_pair_receipt(
    receipt: &PairReceipt,
    contract: &FocalBenchmarkContract,
    assignment: &FocalPairAssignment,
    bindings: &CampaignBindings,
) -> Result<(), FocalCampaignError> {
    if receipt.schema_version != FOCAL_CAMPAIGN_SCHEMA_VERSION
        || receipt.schema_id != PAIR_RECEIPT_SCHEMA_ID
    {
        return Err(FocalCampaignError::PairReceiptSchema(assignment.pair_index));
    }
    if receipt.contract_blake3 != bindings.contract_blake3
        || receipt.opponent_field_blake3 != bindings.opponent_field_blake3
        || receipt.contract_sha256 != bindings.contract_sha256
        || receipt.opponent_field_sha256 != bindings.opponent_field_sha256
        || receipt.implementation_binding != contract.implementation_binding
    {
        return Err(FocalCampaignError::ReceiptContractDrift(
            assignment.pair_index,
        ));
    }
    if receipt.payload.pair_index != assignment.pair_index
        || receipt.payload_blake3 != canonical_blake3(&receipt.payload)?
    {
        return Err(FocalCampaignError::ReceiptPayloadDrift(
            assignment.pair_index,
        ));
    }
    validate_response(
        &game_request(contract, assignment, PairArm::Candidate),
        &receipt.payload.candidate,
    )?;
    validate_response(
        &game_request(contract, assignment, PairArm::Control),
        &receipt.payload.control,
    )?;
    validate_focal_pair(&receipt.payload.candidate, &receipt.payload.control)?;
    Ok(())
}

fn build_work_item_summary(
    layout: &FocalCampaignLayout,
    work_item_id: &str,
    contract: &FocalBenchmarkContract,
    opponent_field: &OpponentFieldManifest,
    bindings: &CampaignBindings,
) -> Result<WorkItemSummary, FocalCampaignError> {
    let assignments = assignments_for_work_item(opponent_field, work_item_id)?;
    let mut pair_receipts = Vec::with_capacity(assignments.len());
    let mut peak_rss_bytes = 0;
    let mut maximum_swap_delta_bytes = i64::MIN;
    let mut all_clean_shutdowns = true;
    let mut all_pinecone_conservation_checks_passed = true;
    let mut summed_game_seconds = 0.0;
    let mut summed_checkpoint_load_seconds = 0.0;
    for assignment in &assignments {
        let path = layout.pair_receipt_path(work_item_id, assignment.pair_index);
        let receipt: PairReceipt = read_json(&path)?;
        validate_pair_receipt(&receipt, contract, assignment, bindings)?;
        pair_receipts.push(PairReceiptReference {
            pair_index: assignment.pair_index,
            receipt_blake3: canonical_blake3(&receipt)?,
        });
        for record in [&receipt.payload.candidate, &receipt.payload.control] {
            peak_rss_bytes = peak_rss_bytes.max(record.runtime.peak_rss_bytes);
            maximum_swap_delta_bytes =
                maximum_swap_delta_bytes.max(record.runtime.swap_delta_bytes);
            all_clean_shutdowns &= record.runtime.clean_shutdown;
            all_pinecone_conservation_checks_passed &= record.pinecones.conservation_holds();
            summed_game_seconds += record.elapsed_seconds;
            summed_checkpoint_load_seconds += record.runtime.checkpoint_load_seconds;
        }
    }
    Ok(WorkItemSummary {
        schema_version: FOCAL_CAMPAIGN_SCHEMA_VERSION,
        schema_id: WORK_ITEM_SCHEMA_ID.to_owned(),
        contract_blake3: bindings.contract_blake3.clone(),
        opponent_field_blake3: bindings.opponent_field_blake3.clone(),
        contract_sha256: bindings.contract_sha256.clone(),
        opponent_field_sha256: bindings.opponent_field_sha256.clone(),
        implementation_binding: contract.implementation_binding.clone(),
        work_item_id: work_item_id.to_owned(),
        stage: contract.stage,
        pairs: assignments.len(),
        physical_games: assignments.len() * 2,
        pair_receipts,
        peak_rss_bytes,
        maximum_swap_delta_bytes,
        all_clean_shutdowns,
        all_pinecone_conservation_checks_passed,
        summed_game_seconds,
        summed_checkpoint_load_seconds,
    })
}

pub fn load_work_item_summary(
    layout: &FocalCampaignLayout,
    work_item_id: &str,
) -> Result<WorkItemSummary, FocalCampaignError> {
    parse_work_item_id(work_item_id)?;
    read_json(&layout.work_item_summary_path(work_item_id))
}

pub fn load_all_work_item_summaries(
    layout: &FocalCampaignLayout,
) -> Result<Vec<WorkItemSummary>, FocalCampaignError> {
    let (contract, _) = read_campaign_inputs(layout)?;
    campaign_work_items(&contract)
        .into_iter()
        .map(|work_item_id| load_work_item_summary(layout, &work_item_id))
        .collect()
}

fn validate_receipt_directory(
    layout: &FocalCampaignLayout,
    work_item_id: &str,
    opponent_field: &OpponentFieldManifest,
) -> Result<(), FocalCampaignError> {
    let directory = layout.receipt_directory(work_item_id);
    let expected = assignments_for_work_item(opponent_field, work_item_id)?
        .into_iter()
        .map(|assignment| format!("pair-{:04}.json", assignment.pair_index))
        .collect::<BTreeSet<_>>();
    for entry in fs::read_dir(&directory)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if !entry.file_type()?.is_file() || !expected.contains(&name) {
            return Err(FocalCampaignError::UnexpectedReceiptArtifact(entry.path()));
        }
    }
    Ok(())
}

pub(crate) fn canonical_blake3<T: Serialize>(value: &T) -> Result<String, FocalCampaignError> {
    Ok(blake3::hash(&serde_json::to_vec(value)?)
        .to_hex()
        .to_string())
}

fn campaign_bindings(
    layout: &FocalCampaignLayout,
    contract: &FocalBenchmarkContract,
    opponent_field: &OpponentFieldManifest,
) -> Result<CampaignBindings, FocalCampaignError> {
    Ok(CampaignBindings {
        contract_blake3: canonical_blake3(contract)?,
        opponent_field_blake3: canonical_blake3(opponent_field)?,
        contract_sha256: file_sha256(&layout.contract_path())?,
        opponent_field_sha256: file_sha256(&layout.opponent_field_path())?,
    })
}

pub(crate) fn file_sha256(path: &Path) -> Result<String, FocalCampaignError> {
    let bytes = fs::read(path).map_err(|source| FocalCampaignError::ArtifactIo {
        path: path.to_owned(),
        source,
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

pub(crate) fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, FocalCampaignError> {
    let mut file = File::open(path).map_err(|source| FocalCampaignError::ArtifactIo {
        path: path.to_owned(),
        source,
    })?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|source| FocalCampaignError::ArtifactIo {
            path: path.to_owned(),
            source,
        })?;
    serde_json::from_slice(&bytes).map_err(|source| FocalCampaignError::ArtifactJson {
        path: path.to_owned(),
        source,
    })
}

pub(crate) fn write_immutable_json<T>(path: &Path, value: &T) -> Result<(), FocalCampaignError>
where
    T: Serialize + DeserializeOwned + PartialEq,
{
    if path.exists() {
        let existing: T = read_json(path)?;
        if &existing == value {
            return Ok(());
        }
        return Err(FocalCampaignError::ImmutableArtifactDrift(path.to_owned()));
    }
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    write_immutable_bytes(path, &bytes)
}

pub(crate) fn write_immutable_bytes(path: &Path, bytes: &[u8]) -> Result<(), FocalCampaignError> {
    if path.exists() {
        let existing = fs::read(path).map_err(|source| FocalCampaignError::ArtifactIo {
            path: path.to_owned(),
            source,
        })?;
        if existing == bytes {
            return Ok(());
        }
        return Err(FocalCampaignError::ImmutableArtifactDrift(path.to_owned()));
    }
    let parent = path
        .parent()
        .ok_or_else(|| FocalCampaignError::MissingArtifactParent(path.to_owned()))?;
    fs::create_dir_all(parent)?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| FocalCampaignError::NonUtf8Artifact(path.to_owned()))?;
    let sequence = TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(
        ".{file_name}.{}.{}.tmp",
        std::process::id(),
        sequence
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|source| FocalCampaignError::ArtifactIo {
            path: temporary.clone(),
            source,
        })?;
    if let Err(source) = file.write_all(bytes).and_then(|()| file.sync_all()) {
        drop(file);
        let _ = fs::remove_file(&temporary);
        return Err(FocalCampaignError::ArtifactIo {
            path: temporary.clone(),
            source,
        });
    }
    drop(file);
    if let Err(source) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(FocalCampaignError::ArtifactIo {
            path: path.to_owned(),
            source,
        });
    }
    File::open(parent)?
        .sync_all()
        .map_err(|source| FocalCampaignError::ArtifactIo {
            path: parent.to_owned(),
            source,
        })?;
    Ok(())
}

#[derive(Debug, Error)]
pub enum FocalCampaignError {
    #[error("focal benchmark contract schema is unsupported")]
    ContractSchema,
    #[error("focal benchmark implementation binding is invalid or drifted")]
    ImplementationBinding,
    #[error("opponent-field manifest schema is unsupported")]
    OpponentFieldSchema,
    #[error("opponent-field manifest identity differs from the contract")]
    OpponentFieldIdentityDrift,
    #[error("expected {expected} pairs, found {actual}")]
    PairCount { expected: usize, actual: usize },
    #[error("{0} must not be empty")]
    EmptyIdentity(&'static str),
    #[error("pair {0} is assigned more than once")]
    DuplicateAssignment(usize),
    #[error("pair {0} repeats a protected game seed")]
    DuplicateGameSeed(usize),
    #[error("protected seed domain is empty or duplicated: {0}")]
    DuplicateSeedDomain(String),
    #[error("pair {0} violates focal-seat rotation")]
    AssignmentFocalSeat(usize),
    #[error("opponent-field assignments do not cover the frozen pair range exactly")]
    AssignmentCoverage,
    #[error("expected three opponents, found {0}")]
    OpponentCount(usize),
    #[error("pair {0} has invalid opponent seats")]
    OpponentSeats(usize),
    #[error("pair {0} places the candidate in an opponent seat")]
    CandidateInOpponentSeat(usize),
    #[error("focal work item must be the canonical pair-NNNN identity; found {0}")]
    WorkItem(String),
    #[error("executor failed for pair {pair_index} {arm:?}: {detail}")]
    Executor {
        pair_index: usize,
        arm: PairArm,
        detail: String,
    },
    #[error("executor failed to finalize work item {work_item}: {detail}")]
    Finalize { work_item: String, detail: String },
    #[error("executor response identity drift for pair {pair_index} {arm:?}")]
    ExecutorIdentityDrift { pair_index: usize, arm: PairArm },
    #[error("pair {0} receipt schema is unsupported")]
    PairReceiptSchema(usize),
    #[error("pair {0} receipt contract identity drift")]
    ReceiptContractDrift(usize),
    #[error("pair {0} receipt payload digest or index drift")]
    ReceiptPayloadDrift(usize),
    #[error("work-item summary for {0} does not reproduce from its pair receipt")]
    WorkItemSummaryDrift(String),
    #[error("unexpected, duplicate, or partial receipt artifact: {0}")]
    UnexpectedReceiptArtifact(PathBuf),
    #[error("immutable artifact differs from the registered value: {0}")]
    ImmutableArtifactDrift(PathBuf),
    #[error("artifact path has no parent: {0}")]
    MissingArtifactParent(PathBuf),
    #[error("artifact file name is not UTF-8: {0}")]
    NonUtf8Artifact(PathBuf),
    #[error("artifact I/O failed at {path}: {source}")]
    ArtifactIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("artifact JSON failed at {path}: {source}")]
    ArtifactJson {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Benchmark(#[from] FocalBenchmarkError),
}

#[cfg(test)]
mod tests {
    use std::{
        cell::Cell,
        fmt, io,
        time::{SystemTime, UNIX_EPOCH},
    };

    use cascadia_game::ScoreBreakdown;

    use super::*;
    use crate::focal::{
        FOCAL_BENCHMARK_PROTOCOL_ID, FOCAL_BENCHMARK_SCHEMA_VERSION, FocalRuntimeObservation,
        PineconeObservation, PromotionClassification,
    };

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(label: &str) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "cascadia-focal-campaign-{label}-{}-{nonce}",
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

    #[derive(Debug)]
    struct FinishError;

    impl fmt::Display for FinishError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("synthetic finalization failure")
        }
    }

    impl std::error::Error for FinishError {}

    struct FinishFailingExecutor;

    impl FocalGameExecutor for FinishFailingExecutor {
        type Error = FinishError;

        fn execute(&mut self, request: &FocalGameRequest) -> Result<FocalGameRecord, Self::Error> {
            Ok(fake_response(request))
        }

        fn finish(&mut self) -> Result<(), Self::Error> {
            Err(FinishError)
        }
    }

    fn fixture(stage: BenchmarkStage) -> (FocalBenchmarkContract, OpponentFieldManifest) {
        let contract = FocalBenchmarkContract::new(
            0,
            stage,
            FocalBenchmarkIdentities::new(
                "r2-map-expert-iteration-v1",
                format!("benchmark-{stage:?}"),
                "candidate-v1",
                "control-v1",
                "field-v1",
                "argmax-v1",
            ),
            implementation_binding_fixture(),
        );
        let assignments = (0..stage.expected_pairs())
            .map(|pair_index| {
                let focal_seat = (pair_index % 4) as u8;
                FocalPairAssignment {
                    pair_index,
                    game_seed: GameSeed::from_u64(100_000 + pair_index as u64),
                    seed_domain_id: format!("protected-{pair_index:04}"),
                    focal_seat,
                    opponents: (0..4)
                        .filter(|seat| *seat != focal_seat)
                        .map(|seat| OpponentIdentity {
                            seat,
                            checkpoint_id: format!("historical-{seat}"),
                        })
                        .collect(),
                }
            })
            .collect();
        (
            contract,
            OpponentFieldManifest::new("field-v1", assignments),
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

    fn fake_response(request: &FocalGameRequest) -> FocalGameRecord {
        let control = 90 + (request.identity.pair_index % 5) as u16;
        let base_total = match request.identity.arm {
            PairArm::Candidate => control + 2,
            PairArm::Control => control,
        };
        let remaining = base_total % 3;
        let habitat = [4; 5];
        let wildlife_total = base_total - habitat.iter().sum::<u16>() - remaining;
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
            final_state_hash: [base_total as u8; 32],
            replay_blake3: format!("{base_total:064x}"),
            score: ScoreBreakdown {
                habitat,
                wildlife,
                nature_tokens: remaining,
                habitat_bonus: [0; 5],
                base_total,
                total: base_total,
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
            elapsed_seconds: 0.1,
            runtime: FocalRuntimeObservation {
                checkpoint_load_seconds: 0.01,
                peak_rss_bytes: 1_024,
                swap_delta_bytes: 0,
                clean_shutdown: true,
            },
        }
    }

    #[test]
    fn callback_executor_writes_atomic_pairs_and_resumes_only_complete_pairs() {
        let temporary = TestDirectory::new("resume");
        let (contract, field) = fixture(BenchmarkStage::StrengthBlindedSmoke);
        let layout = initialize_focal_campaign(&temporary.0, &contract, &field).unwrap();
        let calls = Cell::new(0usize);
        let mut executor = |request: &FocalGameRequest| -> Result<FocalGameRecord, io::Error> {
            calls.set(calls.get() + 1);
            Ok(fake_response(request))
        };
        let first = run_focal_work_item(&layout, "pair-0001", &mut executor).unwrap();
        assert_eq!(first.assigned_pairs, 1);
        assert_eq!(first.executed_pairs, 1);
        assert_eq!(first.resumed_pairs, 0);
        assert_eq!(calls.get(), 2);

        let second = run_focal_work_item(&layout, "pair-0001", &mut executor).unwrap();
        assert_eq!(second.executed_pairs, 0);
        assert_eq!(second.resumed_pairs, 1);
        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn physical_arm_order_alternates_by_pair_but_receipts_remain_normalized() {
        let temporary = TestDirectory::new("arm-order");
        let (contract, field) = fixture(BenchmarkStage::StrengthBlindedSmoke);
        let layout = initialize_focal_campaign(&temporary.0, &contract, &field).unwrap();
        let mut observed = Vec::new();
        run_focal_work_item(
            &layout,
            "pair-0001",
            &mut |request: &FocalGameRequest| -> Result<FocalGameRecord, io::Error> {
                observed.push((request.identity.pair_index, request.identity.arm));
                Ok(fake_response(request))
            },
        )
        .unwrap();
        assert_eq!(observed[0], (1, PairArm::Control));
        assert_eq!(observed[1], (1, PairArm::Candidate));
        let receipt: PairReceipt = read_json(&layout.pair_receipt_path("pair-0001", 1)).unwrap();
        assert_eq!(receipt.payload.candidate.identity.arm, PairArm::Candidate);
        assert_eq!(receipt.payload.control.identity.arm, PairArm::Control);
    }

    #[test]
    fn smoke_aggregate_emits_strength_blinded_json_and_markdown() {
        let temporary = TestDirectory::new("smoke");
        let (contract, field) = fixture(BenchmarkStage::StrengthBlindedSmoke);
        let layout = initialize_focal_campaign(&temporary.0, &contract, &field).unwrap();
        for work_item in campaign_work_items(&contract) {
            run_focal_work_item(
                &layout,
                &work_item,
                &mut |request: &FocalGameRequest| -> Result<FocalGameRecord, io::Error> {
                    Ok(fake_response(request))
                },
            )
            .unwrap();
        }
        let (report, artifacts) =
            aggregate_focal_campaign(&layout, 4.0, PromotionGates::default()).unwrap();
        let json = fs::read_to_string(artifacts.json).unwrap();
        let markdown = fs::read_to_string(artifacts.markdown).unwrap();
        let dashboard = fs::read_to_string(layout.dashboard_input_path()).unwrap();
        assert!(matches!(
            report.result,
            FocalCampaignStatistics::StrengthBlindedSmoke(_)
        ));
        assert!(!json.contains("base_total"));
        assert!(markdown.contains("Strength outputs blinded: **true**"));
        assert!(!markdown.contains("Candidate/control mean"));
        assert!(!dashboard.contains("base_total"));
        assert!(dashboard.contains("strength-blinded-smoke"));
    }

    #[test]
    fn development_aggregate_is_fixed_250_and_dashboard_ready() {
        let temporary = TestDirectory::new("development");
        let (contract, field) = fixture(BenchmarkStage::Development);
        let layout = initialize_focal_campaign(&temporary.0, &contract, &field).unwrap();
        for work_item in campaign_work_items(&contract).into_iter().rev() {
            run_focal_work_item(
                &layout,
                &work_item,
                &mut |request: &FocalGameRequest| -> Result<FocalGameRecord, io::Error> {
                    Ok(fake_response(request))
                },
            )
            .unwrap();
        }
        let (report, artifacts) =
            aggregate_focal_campaign(&layout, 50.0, PromotionGates::default()).unwrap();
        assert_eq!(
            report.contract_sha256,
            file_sha256(&layout.contract_path()).unwrap()
        );
        assert_eq!(
            report.opponent_field_sha256,
            file_sha256(&layout.opponent_field_path()).unwrap()
        );
        assert!(report.work_items.iter().all(|item| {
            item.contract_sha256 == report.contract_sha256
                && item.opponent_field_sha256 == report.opponent_field_sha256
                && item
                    .pair_receipts
                    .iter()
                    .all(|receipt| receipt.receipt_blake3.len() == 64)
        }));
        let FocalCampaignStatistics::Development(development) = report.result else {
            panic!("expected development report");
        };
        assert_eq!(development.pairs, 250);
        assert_eq!(development.physical_games, 500);
        assert_eq!(development.classification, PromotionClassification::Promote);
        assert_eq!(development.paired_delta.base_total.mean, 2.0);
        let markdown = fs::read_to_string(artifacts.markdown).unwrap();
        assert!(markdown.contains("## Score histograms"));
        assert!(markdown.contains("Paired delta: +2.000; SE"));
        assert!(markdown.contains("Control P10/P50/P90"));
        assert!(markdown.contains("## Score anatomy"));
        assert!(markdown.contains("## Work-item resources"));
        let dashboard: serde_json::Value = read_json(&layout.dashboard_input_path()).unwrap();
        assert_eq!(dashboard["pairs_completed"], 250);
        assert_eq!(dashboard["classification"], "promote");
        let ledger: serde_json::Value = read_json(&layout.ledger_feed_path()).unwrap();
        assert_eq!(ledger["outcome"], "passed");
        assert!(markdown.contains("Pinecones earned"));
    }

    #[test]
    fn work_item_summary_is_not_published_before_executor_finalizes_cleanly() {
        let temporary = TestDirectory::new("finish-failure");
        let (contract, field) = fixture(BenchmarkStage::StrengthBlindedSmoke);
        let layout = initialize_focal_campaign(&temporary.0, &contract, &field).unwrap();
        let error = run_focal_work_item(&layout, "pair-0000", &mut FinishFailingExecutor)
            .expect_err("finalization must gate the summary");
        assert!(matches!(error, FocalCampaignError::Finalize { .. }));
        assert!(!layout.work_item_summary_path("pair-0000").exists());
    }

    #[test]
    fn shared_work_item_golden_freezes_dual_hash_and_receipt_binding_schema() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/r2_map/focal-work-item-provenance-v4.json");
        let item: WorkItemSummary = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        assert_eq!(item.contract_blake3, "a".repeat(64));
        assert_eq!(item.contract_sha256, "b".repeat(64));
        assert_eq!(item.opponent_field_blake3, "c".repeat(64));
        assert_eq!(item.opponent_field_sha256, "d".repeat(64));
        item.implementation_binding.validate().unwrap();
        assert_eq!(item.pair_receipts.len(), 1);
        assert!(
            item.pair_receipts
                .iter()
                .all(|receipt| receipt.receipt_blake3.len() == 64)
        );
    }

    #[test]
    fn immutable_contract_drift_is_rejected() {
        let temporary = TestDirectory::new("contract-drift");
        let (contract, field) = fixture(BenchmarkStage::StrengthBlindedSmoke);
        initialize_focal_campaign(&temporary.0, &contract, &field).unwrap();
        let mut drifted = contract.clone();
        drifted.candidate_checkpoint_id = "different".to_owned();
        assert!(matches!(
            initialize_focal_campaign(&temporary.0, &drifted, &field),
            Err(FocalCampaignError::ImmutableArtifactDrift(_))
        ));
    }

    #[test]
    fn partial_duplicate_and_tampered_receipts_are_rejected() {
        let temporary = TestDirectory::new("partial");
        let (contract, field) = fixture(BenchmarkStage::StrengthBlindedSmoke);
        let layout = initialize_focal_campaign(&temporary.0, &contract, &field).unwrap();
        let partial = layout
            .receipt_directory("pair-0000")
            .join(".pair-0000.partial");
        fs::write(&partial, b"partial").unwrap();
        assert!(matches!(
            run_focal_work_item(
                &layout,
                "pair-0000",
                &mut |request: &FocalGameRequest| -> Result<FocalGameRecord, io::Error> {
                    Ok(fake_response(request))
                }
            ),
            Err(FocalCampaignError::UnexpectedReceiptArtifact(_))
        ));
        fs::remove_file(partial).unwrap();

        let duplicate = layout
            .receipt_directory("pair-0000")
            .join("pair-0000-copy.json");
        fs::write(&duplicate, b"{}").unwrap();
        assert!(matches!(
            run_focal_work_item(
                &layout,
                "pair-0000",
                &mut |request: &FocalGameRequest| -> Result<FocalGameRecord, io::Error> {
                    Ok(fake_response(request))
                }
            ),
            Err(FocalCampaignError::UnexpectedReceiptArtifact(_))
        ));
        fs::remove_file(duplicate).unwrap();

        run_focal_work_item(
            &layout,
            "pair-0000",
            &mut |request: &FocalGameRequest| -> Result<FocalGameRecord, io::Error> {
                Ok(fake_response(request))
            },
        )
        .unwrap();
        let receipt_path = layout.pair_receipt_path("pair-0000", 0);
        let original_receipt = fs::read(&receipt_path).unwrap();
        let mut receipt: PairReceipt = read_json(&receipt_path).unwrap();
        receipt.contract_sha256 = "0".repeat(64);
        fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
        assert!(matches!(
            run_focal_work_item(
                &layout,
                "pair-0000",
                &mut |request: &FocalGameRequest| -> Result<FocalGameRecord, io::Error> {
                    Ok(fake_response(request))
                }
            ),
            Err(FocalCampaignError::ReceiptContractDrift(0))
        ));
        fs::write(&receipt_path, &original_receipt).unwrap();

        let mut receipt: PairReceipt = read_json(&receipt_path).unwrap();
        receipt.implementation_binding.maximum_width_panel_sha256 = receipt
            .implementation_binding
            .replay_pinecone_panel_sha256
            .clone();
        fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
        assert!(matches!(
            run_focal_work_item(
                &layout,
                "pair-0000",
                &mut |request: &FocalGameRequest| -> Result<FocalGameRecord, io::Error> {
                    Ok(fake_response(request))
                }
            ),
            Err(FocalCampaignError::ReceiptContractDrift(0))
        ));
        fs::write(&receipt_path, &original_receipt).unwrap();

        let mut receipt: PairReceipt = read_json(&receipt_path).unwrap();
        receipt.payload.candidate.score.base_total += 1;
        fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
        assert!(matches!(
            run_focal_work_item(
                &layout,
                "pair-0000",
                &mut |request: &FocalGameRequest| -> Result<FocalGameRecord, io::Error> {
                    Ok(fake_response(request))
                }
            ),
            Err(FocalCampaignError::ReceiptPayloadDrift(0))
        ));
    }
}
