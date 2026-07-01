use std::{collections::BTreeMap, path::PathBuf, time::Instant};

use cascadia_data::{PositionRecord, PositionShardReader, TARGET_DIM};
use cascadia_game::D6Transform;
use r2_sparse_entity_census::SparsePublicState;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::census::{
    FROZEN_RECORD_COUNT, ValidatedDataset, dataset_order, scientific_hash, update_framed_hash,
    validate_frozen_aggregate_datasets, validate_inputs,
};
use crate::{
    AdaptiveMultiResolutionState, BOUNDED_ACTIVE_SCALAR_LIMIT, BOUNDED_BYTE_LIMIT,
    BOUNDED_MAX_TOKENS, BOUNDED_P99_TOKEN_LIMIT, BOUNDED_PADDED_SCALAR_LIMIT,
    BOUNDED_THROUGHPUT_RATIO_MINIMUM, BoundedAdversarialParityReport, BoundedAdversarialReport,
    BoundedArm, BoundedFeatureView, BoundedTokenKind, DatasetIdentity, DistributionSummary,
    FeatureAblation, Histogram, NearFieldRadius, R4Error, Result,
    validate_bounded_adversarial_parity_report, validate_bounded_adversarial_report,
};

pub const BOUNDED_EXPERIMENT_ID: &str = "r4-bounded-far-quotient-foundation-v1";
pub const BOUNDED_REPORT_SCHEMA: &str = "r4-bounded-far-quotient-report-v1";
pub const BOUNDED_AGGREGATE_SCHEMA: &str = "r4-bounded-far-quotient-aggregate-v1";
pub const BOUNDED_ORDER_PROOF_SCHEMA: &str = "r4-bounded-far-quotient-order-proof-v1";

const CORPUS_CONTRACT: &str = "accepted-r0-r2-60000-v1";
const EXPECTED_D6_CHECKS_PER_RECORD: u64 = 12;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedArmScientific {
    pub schema: String,
    pub experiment_id: String,
    pub arm_id: String,
    pub hard_token_max: usize,
    pub corpus_contract: String,
    pub datasets: Vec<DatasetIdentity>,
    pub record_count: usize,
    pub source_stream_blake3: String,
    pub exact_state_stream_blake3: String,
    pub full_hwf_stream_blake3: String,
    pub bounded_feature_stream_blake3: String,
    pub exact_codec_round_trip_checks: u64,
    pub r2_semantic_equality_checks: u64,
    pub target_independence_checks: u64,
    pub deterministic_construction_checks: u64,
    pub bounded_codec_round_trip_checks: u64,
    pub source_accounting_checks: u64,
    pub d6_inverse_checks: u64,
    pub d6_shape_covariance_checks: u64,
    pub packed_byte_histogram: Histogram,
    pub full_hwf_token_histogram: Histogram,
    pub full_hwf_byte_histogram: Histogram,
    pub bounded_token_histogram: Histogram,
    pub active_scalar_histogram: Histogram,
    pub padded_scalar_histogram: Histogram,
    pub bounded_byte_histogram: Histogram,
    pub habitat_component_histogram: Histogram,
    pub wildlife_component_histogram: Histogram,
    pub source_wildlife_bucket_histogram: Histogram,
    pub summarized_wildlife_bucket_histogram: Histogram,
    pub exact_wildlife_bucket_histogram: Histogram,
    pub source_frontier_bucket_histogram: Histogram,
    pub summarized_frontier_bucket_histogram: Histogram,
    pub exact_frontier_bucket_histogram: Histogram,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedArmOperational {
    pub elapsed_seconds: f64,
    pub records_per_second: f64,
    pub rayon_threads: usize,
    pub full_hwf_view_nanoseconds: u64,
    pub bounded_view_nanoseconds: u64,
    pub bounded_to_full_hwf_throughput_ratio: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedArmReport {
    pub scientific: BoundedArmScientific,
    pub scientific_blake3: String,
    pub operational: BoundedArmOperational,
    pub operational_blake3: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundedArmClassification {
    Passed,
    InformationFailed,
    SizeFailed,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedArmGateAssessment {
    pub exact_mechanics: bool,
    pub frozen_corpus_complete: bool,
    pub source_accounting_exact: bool,
    pub adversarial_information_passed: bool,
    pub hard_arm_token_max_passed: bool,
    pub global_token_max_passed: bool,
    pub token_p99_passed: bool,
    pub active_scalar_max_passed: bool,
    pub padded_scalar_max_passed: bool,
    pub canonical_byte_max_passed: bool,
    pub paired_throughput_passed: bool,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedArmAggregate {
    pub arm_id: String,
    pub hard_token_max: usize,
    pub record_count: usize,
    pub tokens: DistributionSummary,
    pub active_scalars: DistributionSummary,
    pub padded_scalar_slots: DistributionSummary,
    pub canonical_bytes: DistributionSummary,
    pub full_hwf_tokens: DistributionSummary,
    pub full_hwf_bytes: DistributionSummary,
    pub habitat_components: DistributionSummary,
    pub wildlife_components: DistributionSummary,
    pub source_wildlife_buckets: DistributionSummary,
    pub summarized_wildlife_buckets: DistributionSummary,
    pub exact_wildlife_buckets: DistributionSummary,
    pub source_frontier_buckets: DistributionSummary,
    pub summarized_frontier_buckets: DistributionSummary,
    pub exact_frontier_buckets: DistributionSummary,
    pub bounded_to_full_hwf_throughput_ratio: f64,
    pub full_hwf_view_nanoseconds: u64,
    pub bounded_view_nanoseconds: u64,
    pub gate_assessment: BoundedArmGateAssessment,
    pub classification: BoundedArmClassification,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundedClassification {
    R4BoundedQuotientFoundationPassed,
    R4BoundedQuotientInformationFailed,
    R4BoundedQuotientSizeFailed,
    R4BoundedQuotientInvalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedPromotionAssessment {
    pub four_arm_coverage: bool,
    pub frozen_corpus_complete: bool,
    pub source_streams_identical: bool,
    pub exact_state_streams_identical: bool,
    pub full_hwf_streams_identical: bool,
    pub adversarial_cross_host_parity: bool,
    pub passing_arm_count: usize,
    pub minimal_successor: Option<String>,
    pub richest_successor: Option<String>,
    pub order_invariant: bool,
    pub authorize_matched_mlx: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedAggregateScientific {
    pub schema: String,
    pub experiment_id: String,
    pub corpus_contract: String,
    pub datasets: Vec<DatasetIdentity>,
    pub record_count_per_arm: usize,
    pub source_stream_blake3: String,
    pub exact_state_stream_blake3: String,
    pub full_hwf_stream_blake3: String,
    pub adversarial_scientific_blake3: String,
    pub adversarial_parity_scientific_blake3: String,
    pub arms: Vec<BoundedArmAggregate>,
    pub promotion_assessment: BoundedPromotionAssessment,
    pub classification: BoundedClassification,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedAggregateReport {
    pub scientific: BoundedAggregateScientific,
    pub scientific_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedOrderProofScientific {
    pub schema: String,
    pub experiment_id: String,
    pub forward_arm_order: Vec<String>,
    pub reverse_arm_order: Vec<String>,
    pub forward_aggregate_scientific_blake3: String,
    pub reverse_aggregate_scientific_blake3: String,
    pub forward_document_blake3: String,
    pub reverse_document_blake3: String,
    pub byte_identical: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedOrderProofReport {
    pub scientific: BoundedOrderProofScientific,
    pub scientific_blake3: String,
}

#[derive(Debug)]
struct BoundedRow {
    source_digest: [u8; 32],
    exact_digest: [u8; 32],
    full_hwf_digest: [u8; 32],
    bounded_digest: [u8; 32],
    packed_bytes: u64,
    full_hwf_tokens: u64,
    full_hwf_bytes: u64,
    bounded_tokens: u64,
    active_scalars: u64,
    padded_scalars: u64,
    bounded_bytes: u64,
    habitat_components: u64,
    wildlife_components: u64,
    source_wildlife_buckets: u64,
    summarized_wildlife_buckets: u64,
    exact_wildlife_buckets: u64,
    source_frontier_buckets: u64,
    summarized_frontier_buckets: u64,
    exact_frontier_buckets: u64,
    full_hwf_nanoseconds: u64,
    bounded_nanoseconds: u64,
    d6_checks: u64,
    d6_shape_checks: u64,
}

#[derive(Default)]
struct BoundedAccumulator {
    record_count: usize,
    source_hasher: blake3::Hasher,
    exact_hasher: blake3::Hasher,
    full_hwf_hasher: blake3::Hasher,
    bounded_hasher: blake3::Hasher,
    packed_bytes: Histogram,
    full_hwf_tokens: Histogram,
    full_hwf_bytes: Histogram,
    bounded_tokens: Histogram,
    active_scalars: Histogram,
    padded_scalars: Histogram,
    bounded_bytes: Histogram,
    habitat_components: Histogram,
    wildlife_components: Histogram,
    source_wildlife_buckets: Histogram,
    summarized_wildlife_buckets: Histogram,
    exact_wildlife_buckets: Histogram,
    source_frontier_buckets: Histogram,
    summarized_frontier_buckets: Histogram,
    exact_frontier_buckets: Histogram,
    full_hwf_nanoseconds: u64,
    bounded_nanoseconds: u64,
    d6_checks: u64,
    d6_shape_checks: u64,
}

impl BoundedAccumulator {
    fn observe(&mut self, row: BoundedRow) -> Result<()> {
        self.record_count = self
            .record_count
            .checked_add(1)
            .ok_or_else(|| R4Error::DatasetContract("bounded record count overflow".to_owned()))?;
        update_framed_hash(&mut self.source_hasher, &row.source_digest);
        update_framed_hash(&mut self.exact_hasher, &row.exact_digest);
        update_framed_hash(&mut self.full_hwf_hasher, &row.full_hwf_digest);
        update_framed_hash(&mut self.bounded_hasher, &row.bounded_digest);
        self.packed_bytes.observe(row.packed_bytes);
        self.full_hwf_tokens.observe(row.full_hwf_tokens);
        self.full_hwf_bytes.observe(row.full_hwf_bytes);
        self.bounded_tokens.observe(row.bounded_tokens);
        self.active_scalars.observe(row.active_scalars);
        self.padded_scalars.observe(row.padded_scalars);
        self.bounded_bytes.observe(row.bounded_bytes);
        self.habitat_components.observe(row.habitat_components);
        self.wildlife_components.observe(row.wildlife_components);
        self.source_wildlife_buckets
            .observe(row.source_wildlife_buckets);
        self.summarized_wildlife_buckets
            .observe(row.summarized_wildlife_buckets);
        self.exact_wildlife_buckets
            .observe(row.exact_wildlife_buckets);
        self.source_frontier_buckets
            .observe(row.source_frontier_buckets);
        self.summarized_frontier_buckets
            .observe(row.summarized_frontier_buckets);
        self.exact_frontier_buckets
            .observe(row.exact_frontier_buckets);
        self.full_hwf_nanoseconds = self
            .full_hwf_nanoseconds
            .checked_add(row.full_hwf_nanoseconds)
            .ok_or_else(|| {
                R4Error::DatasetContract("full-HWF timing counter overflow".to_owned())
            })?;
        self.bounded_nanoseconds = self
            .bounded_nanoseconds
            .checked_add(row.bounded_nanoseconds)
            .ok_or_else(|| {
                R4Error::DatasetContract("bounded timing counter overflow".to_owned())
            })?;
        self.d6_checks = self
            .d6_checks
            .checked_add(row.d6_checks)
            .ok_or_else(|| R4Error::DatasetContract("D6 check counter overflow".to_owned()))?;
        self.d6_shape_checks = self
            .d6_shape_checks
            .checked_add(row.d6_shape_checks)
            .ok_or_else(|| R4Error::DatasetContract("D6 shape counter overflow".to_owned()))?;
        Ok(())
    }
}

pub fn census_bounded_arm(
    roots: &[PathBuf],
    arm: BoundedArm,
    require_frozen: bool,
) -> Result<BoundedArmReport> {
    let mut datasets = validate_inputs(roots)?;
    if require_frozen {
        validate_full_frozen_corpus(&datasets)?;
        datasets.sort_unstable_by_key(|dataset| dataset_order(&dataset.identity));
    } else {
        datasets.sort_unstable_by(|left, right| {
            left.identity.dataset_id.cmp(&right.identity.dataset_id)
        });
    }

    let started = Instant::now();
    let mut accumulator = BoundedAccumulator::default();
    for dataset in &datasets {
        let records = load_records(dataset)?;
        let rows = records
            .par_iter()
            .enumerate()
            .map(|(ordinal, record)| {
                process_record(record, arm).map_err(|error| {
                    R4Error::DatasetContract(format!(
                        "dataset {} record {ordinal} failed in {}: {error}",
                        dataset.manifest.dataset_id,
                        arm.id()
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        for row in rows {
            accumulator.observe(row)?;
        }
    }
    if accumulator.record_count == 0 {
        return Err(R4Error::DatasetContract(
            "bounded census selected zero records".to_owned(),
        ));
    }
    if require_frozen && accumulator.record_count != FROZEN_RECORD_COUNT {
        return Err(R4Error::DatasetContract(format!(
            "bounded census processed {} rows; expected {FROZEN_RECORD_COUNT}",
            accumulator.record_count
        )));
    }

    let record_count = accumulator.record_count;
    let scientific = BoundedArmScientific {
        schema: BOUNDED_REPORT_SCHEMA.to_owned(),
        experiment_id: BOUNDED_EXPERIMENT_ID.to_owned(),
        arm_id: arm.id().to_owned(),
        hard_token_max: arm.hard_token_max(),
        corpus_contract: if require_frozen {
            CORPUS_CONTRACT.to_owned()
        } else {
            "validated-compact-entity-v2".to_owned()
        },
        datasets: datasets
            .iter()
            .map(|dataset| dataset.identity.clone())
            .collect(),
        record_count,
        source_stream_blake3: accumulator.source_hasher.finalize().to_hex().to_string(),
        exact_state_stream_blake3: accumulator.exact_hasher.finalize().to_hex().to_string(),
        full_hwf_stream_blake3: accumulator.full_hwf_hasher.finalize().to_hex().to_string(),
        bounded_feature_stream_blake3: accumulator.bounded_hasher.finalize().to_hex().to_string(),
        exact_codec_round_trip_checks: record_count as u64,
        r2_semantic_equality_checks: record_count as u64,
        target_independence_checks: record_count as u64,
        deterministic_construction_checks: record_count as u64,
        bounded_codec_round_trip_checks: record_count as u64,
        source_accounting_checks: record_count as u64,
        d6_inverse_checks: accumulator.d6_checks,
        d6_shape_covariance_checks: accumulator.d6_shape_checks,
        packed_byte_histogram: accumulator.packed_bytes,
        full_hwf_token_histogram: accumulator.full_hwf_tokens,
        full_hwf_byte_histogram: accumulator.full_hwf_bytes,
        bounded_token_histogram: accumulator.bounded_tokens,
        active_scalar_histogram: accumulator.active_scalars,
        padded_scalar_histogram: accumulator.padded_scalars,
        bounded_byte_histogram: accumulator.bounded_bytes,
        habitat_component_histogram: accumulator.habitat_components,
        wildlife_component_histogram: accumulator.wildlife_components,
        source_wildlife_bucket_histogram: accumulator.source_wildlife_buckets,
        summarized_wildlife_bucket_histogram: accumulator.summarized_wildlife_buckets,
        exact_wildlife_bucket_histogram: accumulator.exact_wildlife_buckets,
        source_frontier_bucket_histogram: accumulator.source_frontier_buckets,
        summarized_frontier_bucket_histogram: accumulator.summarized_frontier_buckets,
        exact_frontier_bucket_histogram: accumulator.exact_frontier_buckets,
    };
    let scientific_blake3 = scientific_hash(&scientific)?;
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let throughput_ratio =
        accumulator.full_hwf_nanoseconds as f64 / accumulator.bounded_nanoseconds.max(1) as f64;
    let operational = BoundedArmOperational {
        elapsed_seconds,
        records_per_second: record_count as f64 / elapsed_seconds,
        rayon_threads: rayon::current_num_threads(),
        full_hwf_view_nanoseconds: accumulator.full_hwf_nanoseconds,
        bounded_view_nanoseconds: accumulator.bounded_nanoseconds,
        bounded_to_full_hwf_throughput_ratio: throughput_ratio,
    };
    let operational_blake3 = scientific_hash(&operational)?;
    Ok(BoundedArmReport {
        scientific,
        scientific_blake3,
        operational,
        operational_blake3,
    })
}

fn load_records(dataset: &ValidatedDataset) -> Result<Vec<PositionRecord>> {
    let mut records = Vec::with_capacity(dataset.manifest.total_records);
    for shard in &dataset.manifest.shards {
        records.extend(
            PositionShardReader::open(&dataset.root, shard)?
                .collect::<std::result::Result<Vec<_>, _>>()?,
        );
    }
    if records.len() != dataset.manifest.total_records {
        return Err(R4Error::DatasetContract(format!(
            "dataset {} loaded {} rows; manifest declares {}",
            dataset.manifest.dataset_id,
            records.len(),
            dataset.manifest.total_records
        )));
    }
    Ok(records)
}

fn process_record(record: &PositionRecord, arm: BoundedArm) -> Result<BoundedRow> {
    let mut public_record = record.clone();
    public_record.targets = [0; TARGET_DIM];
    let source_digest = *blake3::hash(&public_record.to_bytes()).as_bytes();
    let sparse = SparsePublicState::from_position_record(&public_record, None)?;
    let state = AdaptiveMultiResolutionState::from_sparse_state(&sparse, NearFieldRadius::Radius4)?;
    let packed = state.to_packed_bytes()?;
    let decoded = AdaptiveMultiResolutionState::from_packed_bytes(&packed)?;
    if decoded != state || decoded.to_sparse_state()? != sparse {
        return Err(R4Error::DatasetContract(
            "bounded census exact codec or R2 semantic round trip changed state".to_owned(),
        ));
    }

    let baseline_first = source_digest[0] & 1 == 0;
    let ((full_hwf_view, full_hwf_bytes), full_hwf_nanoseconds);
    let ((bounded_view, bounded_bytes), bounded_nanoseconds);
    if baseline_first {
        ((full_hwf_view, full_hwf_bytes), full_hwf_nanoseconds) =
            measure(|| build_full_hwf(&state))?;
        ((bounded_view, bounded_bytes), bounded_nanoseconds) =
            measure(|| build_bounded(&state, arm))?;
    } else {
        ((bounded_view, bounded_bytes), bounded_nanoseconds) =
            measure(|| build_bounded(&state, arm))?;
        ((full_hwf_view, full_hwf_bytes), full_hwf_nanoseconds) =
            measure(|| build_full_hwf(&state))?;
    }
    if BoundedFeatureView::from_canonical_bytes(&bounded_bytes)? != bounded_view {
        return Err(R4Error::DatasetContract(
            "bounded feature envelope failed its round trip".to_owned(),
        ));
    }
    if BoundedFeatureView::from_state(&state, arm)?.canonical_bytes()? != bounded_bytes {
        return Err(R4Error::DatasetContract(
            "bounded feature construction is nondeterministic".to_owned(),
        ));
    }

    let mut changed_targets = record.clone();
    for target in &mut changed_targets.targets {
        *target = !*target;
    }
    let changed = AdaptiveMultiResolutionState::from_position_record(
        &changed_targets,
        None,
        NearFieldRadius::Radius4,
    )?;
    if changed.to_packed_bytes()? != packed
        || BoundedFeatureView::from_state(&changed, arm)?.canonical_bytes()? != bounded_bytes
    {
        return Err(R4Error::DatasetContract(
            "target mutation changed exact or bounded bytes".to_owned(),
        ));
    }

    let original_shape = bounded_shape(&bounded_view);
    let mut d6_checks = 0u64;
    let mut d6_shape_checks = 0u64;
    for transform in D6Transform::ALL {
        let transformed = state.transformed(transform)?;
        let transformed_view = BoundedFeatureView::from_state(&transformed, arm)?;
        if bounded_shape(&transformed_view) != original_shape {
            return Err(R4Error::DatasetContract(format!(
                "{} changed bounded shape under D6 transform {}",
                arm.id(),
                transform.id()
            )));
        }
        d6_shape_checks += 1;
        let restored = transformed.transformed(transform.inverse())?;
        if restored != state
            || BoundedFeatureView::from_state(&restored, arm)?.canonical_bytes()? != bounded_bytes
        {
            return Err(R4Error::DatasetContract(format!(
                "{} failed D6 inverse transform {}",
                arm.id(),
                transform.id()
            )));
        }
        d6_checks += 1;
    }

    let habitat_components = state
        .boards
        .iter()
        .map(|board| board.far_habitat_components.len())
        .sum::<usize>();
    let wildlife_components = state
        .boards
        .iter()
        .map(|board| board.far_wildlife_components.len())
        .sum::<usize>();
    Ok(BoundedRow {
        source_digest,
        exact_digest: *blake3::hash(&packed).as_bytes(),
        full_hwf_digest: *blake3::hash(&full_hwf_bytes).as_bytes(),
        bounded_digest: *blake3::hash(&bounded_bytes).as_bytes(),
        packed_bytes: checked_u64(packed.len(), "packed bytes")?,
        full_hwf_tokens: checked_u64(full_hwf_view.spatial_token_count(), "full-HWF tokens")?,
        full_hwf_bytes: checked_u64(full_hwf_bytes.len(), "full-HWF bytes")?,
        bounded_tokens: checked_u64(bounded_view.spatial_token_count(), "bounded tokens")?,
        active_scalars: checked_u64(bounded_view.active_scalar_count(), "active scalars")?,
        padded_scalars: checked_u64(bounded_view.padded_scalar_slots(), "padded scalar slots")?,
        bounded_bytes: checked_u64(bounded_bytes.len(), "bounded bytes")?,
        habitat_components: checked_u64(habitat_components, "habitat components")?,
        wildlife_components: checked_u64(wildlife_components, "wildlife components")?,
        source_wildlife_buckets: u64::from(bounded_view.accounting.source_wildlife_buckets),
        summarized_wildlife_buckets: u64::from(bounded_view.accounting.summarized_wildlife_buckets),
        exact_wildlife_buckets: u64::from(bounded_view.accounting.exact_wildlife_buckets),
        source_frontier_buckets: u64::from(bounded_view.accounting.source_frontier_buckets),
        summarized_frontier_buckets: u64::from(bounded_view.accounting.summarized_frontier_buckets),
        exact_frontier_buckets: u64::from(bounded_view.accounting.exact_frontier_buckets),
        full_hwf_nanoseconds,
        bounded_nanoseconds,
        d6_checks,
        d6_shape_checks,
    })
}

fn build_full_hwf(
    state: &AdaptiveMultiResolutionState,
) -> Result<(crate::AdaptiveFeatureView, Vec<u8>)> {
    let view = state.feature_view(FeatureAblation::AllTopology)?;
    let bytes = view.canonical_bytes()?;
    Ok((view, bytes))
}

fn build_bounded(
    state: &AdaptiveMultiResolutionState,
    arm: BoundedArm,
) -> Result<(BoundedFeatureView, Vec<u8>)> {
    let view = BoundedFeatureView::from_state(state, arm)?;
    let bytes = view.canonical_bytes()?;
    Ok((view, bytes))
}

fn measure<T>(operation: impl FnOnce() -> Result<T>) -> Result<(T, u64)> {
    let started = Instant::now();
    let value = operation()?;
    let nanoseconds = u64::try_from(started.elapsed().as_nanos()).map_err(|_| {
        R4Error::DatasetContract("paired timing exceeded u64 nanoseconds".to_owned())
    })?;
    Ok((value, nanoseconds))
}

fn bounded_shape(
    view: &BoundedFeatureView,
) -> (
    usize,
    usize,
    usize,
    Vec<(BoundedTokenKind, usize)>,
    u32,
    u32,
) {
    let mut kinds = BTreeMap::new();
    for token in &view.tokens {
        *kinds.entry(token.kind).or_insert(0usize) += 1;
    }
    (
        view.spatial_token_count(),
        view.active_scalar_count(),
        view.padded_scalar_slots(),
        kinds.into_iter().collect(),
        view.accounting.exact_wildlife_buckets,
        view.accounting.exact_frontier_buckets,
    )
}

fn validate_full_frozen_corpus(datasets: &[ValidatedDataset]) -> Result<()> {
    let identities = datasets
        .iter()
        .map(|dataset| dataset.identity.clone())
        .collect::<Vec<_>>();
    validate_frozen_aggregate_datasets(&identities)?;
    let rows = datasets
        .iter()
        .map(|dataset| dataset.manifest.total_records)
        .sum::<usize>();
    if rows != FROZEN_RECORD_COUNT {
        return Err(R4Error::DatasetContract(format!(
            "bounded full corpus has {rows} records; expected {FROZEN_RECORD_COUNT}"
        )));
    }
    Ok(())
}

pub fn aggregate_bounded_reports(
    reports: &[BoundedArmReport],
    adversarial: &BoundedAdversarialReport,
    adversarial_parity: &BoundedAdversarialParityReport,
) -> Result<BoundedAggregateReport> {
    validate_bounded_adversarial_report(adversarial)?;
    validate_bounded_adversarial_parity_report(adversarial_parity)?;
    if reports.len() != BoundedArm::ALL.len() {
        return Err(R4Error::AggregateContract(format!(
            "bounded aggregate requires {} arm reports; received {}",
            BoundedArm::ALL.len(),
            reports.len()
        )));
    }

    let mut by_arm = BTreeMap::new();
    for report in reports {
        validate_arm_report(report)?;
        let arm = BoundedArm::from_id(&report.scientific.arm_id).ok_or_else(|| {
            R4Error::AggregateContract("bounded report has unknown arm".to_owned())
        })?;
        if by_arm.insert(arm, report).is_some() {
            return Err(R4Error::AggregateContract(
                "bounded aggregate contains a duplicate arm".to_owned(),
            ));
        }
    }
    if by_arm.keys().copied().collect::<Vec<_>>() != BoundedArm::ALL {
        return Err(R4Error::AggregateContract(
            "bounded aggregate arm coverage is incomplete".to_owned(),
        ));
    }

    let first = by_arm
        .first_key_value()
        .expect("four validated arms are present")
        .1;
    let datasets = first.scientific.datasets.clone();
    validate_frozen_aggregate_datasets(&datasets)?;
    let source_stream = first.scientific.source_stream_blake3.clone();
    let exact_stream = first.scientific.exact_state_stream_blake3.clone();
    let full_hwf_stream = first.scientific.full_hwf_stream_blake3.clone();
    let source_streams_identical = by_arm
        .values()
        .all(|report| report.scientific.source_stream_blake3 == source_stream);
    let exact_state_streams_identical = by_arm
        .values()
        .all(|report| report.scientific.exact_state_stream_blake3 == exact_stream);
    let full_hwf_streams_identical = by_arm
        .values()
        .all(|report| report.scientific.full_hwf_stream_blake3 == full_hwf_stream);
    let datasets_identical = by_arm
        .values()
        .all(|report| report.scientific.datasets == datasets);

    let adversarial_cross_host_parity = adversarial_parity.scientific.report_count == 4
        && adversarial_parity
            .scientific
            .all_scientific_reports_identical
        && adversarial_parity.scientific.all_suites_passed
        && adversarial_parity
            .scientific
            .report_scientific_blake3s
            .iter()
            .all(|hash| hash == &adversarial.scientific_blake3);

    let corpus_common_valid = datasets_identical
        && source_streams_identical
        && exact_state_streams_identical
        && full_hwf_streams_identical
        && by_arm
            .values()
            .all(|report| report.scientific.record_count == FROZEN_RECORD_COUNT);
    let mut arms = Vec::with_capacity(BoundedArm::ALL.len());
    for arm in BoundedArm::ALL {
        let report = by_arm[&arm];
        let adversarial_information_passed = adversarial.scientific.cases.iter().all(|case| {
            case.arms
                .iter()
                .find(|result| result.arm_id == arm.id())
                .is_some_and(|result| result.passed)
        });
        arms.push(aggregate_arm(
            report,
            arm,
            corpus_common_valid,
            adversarial_information_passed,
        )?);
    }

    let passing = arms
        .iter()
        .filter(|arm| arm.gate_assessment.passed)
        .collect::<Vec<_>>();
    let passing_arm_count = passing.len();
    let minimal_successor = passing
        .iter()
        .min_by(|left, right| {
            left.tokens
                .max
                .cmp(&right.tokens.max)
                .then_with(|| left.canonical_bytes.p99.cmp(&right.canonical_bytes.p99))
                .then_with(|| {
                    right
                        .bounded_to_full_hwf_throughput_ratio
                        .total_cmp(&left.bounded_to_full_hwf_throughput_ratio)
                })
        })
        .map(|arm| arm.arm_id.clone());
    let richest_successor = if passing
        .iter()
        .any(|arm| arm.arm_id == BoundedArm::SelectiveExact.id())
    {
        Some(BoundedArm::SelectiveExact.id().to_owned())
    } else {
        passing
            .iter()
            .max_by_key(|arm| arm.hard_token_max)
            .map(|arm| arm.arm_id.clone())
    };

    let four_arm_coverage = by_arm.len() == BoundedArm::ALL.len();
    let frozen_corpus_complete = corpus_common_valid;
    let mechanically_valid = four_arm_coverage
        && frozen_corpus_complete
        && adversarial_cross_host_parity
        && arms
            .iter()
            .all(|arm| arm.classification != BoundedArmClassification::Invalid);
    let any_information = arms
        .iter()
        .any(|arm| arm.gate_assessment.adversarial_information_passed);
    let classification = if !mechanically_valid {
        BoundedClassification::R4BoundedQuotientInvalid
    } else if !any_information {
        BoundedClassification::R4BoundedQuotientInformationFailed
    } else if passing_arm_count == 0 {
        BoundedClassification::R4BoundedQuotientSizeFailed
    } else {
        BoundedClassification::R4BoundedQuotientFoundationPassed
    };
    let authorize_matched_mlx =
        classification == BoundedClassification::R4BoundedQuotientFoundationPassed;
    let scientific = BoundedAggregateScientific {
        schema: BOUNDED_AGGREGATE_SCHEMA.to_owned(),
        experiment_id: BOUNDED_EXPERIMENT_ID.to_owned(),
        corpus_contract: CORPUS_CONTRACT.to_owned(),
        datasets,
        record_count_per_arm: FROZEN_RECORD_COUNT,
        source_stream_blake3: source_stream,
        exact_state_stream_blake3: exact_stream,
        full_hwf_stream_blake3: full_hwf_stream,
        adversarial_scientific_blake3: adversarial.scientific_blake3.clone(),
        adversarial_parity_scientific_blake3: adversarial_parity.scientific_blake3.clone(),
        arms,
        promotion_assessment: BoundedPromotionAssessment {
            four_arm_coverage,
            frozen_corpus_complete,
            source_streams_identical,
            exact_state_streams_identical,
            full_hwf_streams_identical,
            adversarial_cross_host_parity,
            passing_arm_count,
            minimal_successor,
            richest_successor,
            order_invariant: true,
            authorize_matched_mlx,
        },
        classification,
    };
    let scientific_blake3 = scientific_hash(&scientific)?;
    Ok(BoundedAggregateReport {
        scientific,
        scientific_blake3,
    })
}

fn validate_arm_report(report: &BoundedArmReport) -> Result<()> {
    if scientific_hash(&report.scientific)? != report.scientific_blake3
        || scientific_hash(&report.operational)? != report.operational_blake3
        || report.scientific.schema != BOUNDED_REPORT_SCHEMA
        || report.scientific.experiment_id != BOUNDED_EXPERIMENT_ID
        || report.scientific.corpus_contract != CORPUS_CONTRACT
    {
        return Err(R4Error::AggregateContract(
            "bounded arm report contract or hash drifted".to_owned(),
        ));
    }
    let arm = BoundedArm::from_id(&report.scientific.arm_id)
        .ok_or_else(|| R4Error::AggregateContract("bounded report arm is unknown".to_owned()))?;
    if report.scientific.hard_token_max != arm.hard_token_max()
        || report.scientific.datasets.len() != 8
        || report.scientific.record_count != FROZEN_RECORD_COUNT
        || report.operational.full_hwf_view_nanoseconds == 0
        || report.operational.bounded_view_nanoseconds == 0
        || !report
            .operational
            .bounded_to_full_hwf_throughput_ratio
            .is_finite()
    {
        return Err(R4Error::AggregateContract(
            "bounded arm report metadata drifted".to_owned(),
        ));
    }
    validate_frozen_aggregate_datasets(&report.scientific.datasets)?;
    let expected_ratio = report.operational.full_hwf_view_nanoseconds as f64
        / report.operational.bounded_view_nanoseconds as f64;
    if (expected_ratio - report.operational.bounded_to_full_hwf_throughput_ratio).abs()
        > f64::EPSILON * expected_ratio.abs().max(1.0) * 4.0
    {
        return Err(R4Error::AggregateContract(
            "bounded paired throughput ratio is inconsistent".to_owned(),
        ));
    }
    for histogram in report_histograms(&report.scientific) {
        if histogram.count() != FROZEN_RECORD_COUNT as u64 {
            return Err(R4Error::AggregateContract(
                "bounded arm histogram coverage drifted".to_owned(),
            ));
        }
    }
    Ok(())
}

fn report_histograms(scientific: &BoundedArmScientific) -> [&Histogram; 15] {
    [
        &scientific.packed_byte_histogram,
        &scientific.full_hwf_token_histogram,
        &scientific.full_hwf_byte_histogram,
        &scientific.bounded_token_histogram,
        &scientific.active_scalar_histogram,
        &scientific.padded_scalar_histogram,
        &scientific.bounded_byte_histogram,
        &scientific.habitat_component_histogram,
        &scientific.wildlife_component_histogram,
        &scientific.source_wildlife_bucket_histogram,
        &scientific.summarized_wildlife_bucket_histogram,
        &scientific.exact_wildlife_bucket_histogram,
        &scientific.source_frontier_bucket_histogram,
        &scientific.summarized_frontier_bucket_histogram,
        &scientific.exact_frontier_bucket_histogram,
    ]
}

fn aggregate_arm(
    report: &BoundedArmReport,
    arm: BoundedArm,
    corpus_common_valid: bool,
    adversarial_information_passed: bool,
) -> Result<BoundedArmAggregate> {
    let tokens = report.scientific.bounded_token_histogram.summary()?;
    let active_scalars = report.scientific.active_scalar_histogram.summary()?;
    let padded_scalar_slots = report.scientific.padded_scalar_histogram.summary()?;
    let canonical_bytes = report.scientific.bounded_byte_histogram.summary()?;
    let expected_records = report.scientific.record_count as u64;
    let exact_mechanics = report.scientific.exact_codec_round_trip_checks == expected_records
        && report.scientific.r2_semantic_equality_checks == expected_records
        && report.scientific.target_independence_checks == expected_records
        && report.scientific.deterministic_construction_checks == expected_records
        && report.scientific.bounded_codec_round_trip_checks == expected_records
        && report.scientific.d6_inverse_checks == expected_records * EXPECTED_D6_CHECKS_PER_RECORD
        && report.scientific.d6_shape_covariance_checks
            == expected_records * EXPECTED_D6_CHECKS_PER_RECORD;
    let source_accounting_exact = report.scientific.source_accounting_checks == expected_records;
    let hard_arm_token_max_passed = tokens.max <= arm.hard_token_max() as u64;
    let global_token_max_passed = tokens.max <= BOUNDED_MAX_TOKENS as u64;
    let token_p99_passed = tokens.p99 <= BOUNDED_P99_TOKEN_LIMIT;
    let active_scalar_max_passed = active_scalars.max <= BOUNDED_ACTIVE_SCALAR_LIMIT;
    let padded_scalar_max_passed = padded_scalar_slots.max <= BOUNDED_PADDED_SCALAR_LIMIT;
    let canonical_byte_max_passed = canonical_bytes.max <= BOUNDED_BYTE_LIMIT;
    let paired_throughput_passed =
        report.operational.bounded_to_full_hwf_throughput_ratio >= BOUNDED_THROUGHPUT_RATIO_MINIMUM;
    let frozen_corpus_complete =
        corpus_common_valid && report.scientific.record_count == FROZEN_RECORD_COUNT;
    let mechanically_valid = exact_mechanics && frozen_corpus_complete && source_accounting_exact;
    let size_and_runtime = hard_arm_token_max_passed
        && global_token_max_passed
        && token_p99_passed
        && active_scalar_max_passed
        && padded_scalar_max_passed
        && canonical_byte_max_passed
        && paired_throughput_passed;
    let passed = mechanically_valid && adversarial_information_passed && size_and_runtime;
    let classification = if !mechanically_valid {
        BoundedArmClassification::Invalid
    } else if !adversarial_information_passed {
        BoundedArmClassification::InformationFailed
    } else if !size_and_runtime {
        BoundedArmClassification::SizeFailed
    } else {
        BoundedArmClassification::Passed
    };
    Ok(BoundedArmAggregate {
        arm_id: arm.id().to_owned(),
        hard_token_max: arm.hard_token_max(),
        record_count: report.scientific.record_count,
        tokens,
        active_scalars,
        padded_scalar_slots,
        canonical_bytes,
        full_hwf_tokens: report.scientific.full_hwf_token_histogram.summary()?,
        full_hwf_bytes: report.scientific.full_hwf_byte_histogram.summary()?,
        habitat_components: report.scientific.habitat_component_histogram.summary()?,
        wildlife_components: report.scientific.wildlife_component_histogram.summary()?,
        source_wildlife_buckets: report
            .scientific
            .source_wildlife_bucket_histogram
            .summary()?,
        summarized_wildlife_buckets: report
            .scientific
            .summarized_wildlife_bucket_histogram
            .summary()?,
        exact_wildlife_buckets: report
            .scientific
            .exact_wildlife_bucket_histogram
            .summary()?,
        source_frontier_buckets: report
            .scientific
            .source_frontier_bucket_histogram
            .summary()?,
        summarized_frontier_buckets: report
            .scientific
            .summarized_frontier_bucket_histogram
            .summary()?,
        exact_frontier_buckets: report
            .scientific
            .exact_frontier_bucket_histogram
            .summary()?,
        bounded_to_full_hwf_throughput_ratio: report
            .operational
            .bounded_to_full_hwf_throughput_ratio,
        full_hwf_view_nanoseconds: report.operational.full_hwf_view_nanoseconds,
        bounded_view_nanoseconds: report.operational.bounded_view_nanoseconds,
        gate_assessment: BoundedArmGateAssessment {
            exact_mechanics,
            frozen_corpus_complete,
            source_accounting_exact,
            adversarial_information_passed,
            hard_arm_token_max_passed,
            global_token_max_passed,
            token_p99_passed,
            active_scalar_max_passed,
            padded_scalar_max_passed,
            canonical_byte_max_passed,
            paired_throughput_passed,
            passed,
        },
        classification,
    })
}

pub fn aggregate_bounded_reports_with_order_proof(
    reports: &[BoundedArmReport],
    adversarial: &BoundedAdversarialReport,
    adversarial_parity: &BoundedAdversarialParityReport,
) -> Result<(
    BoundedAggregateReport,
    BoundedAggregateReport,
    BoundedOrderProofReport,
)> {
    let mut forward = reports.to_vec();
    forward.sort_unstable_by_key(|report| {
        BoundedArm::from_id(&report.scientific.arm_id)
            .map(BoundedArm::code)
            .unwrap_or(u8::MAX)
    });
    let mut reverse = forward.clone();
    reverse.reverse();
    let forward_aggregate = aggregate_bounded_reports(&forward, adversarial, adversarial_parity)?;
    let reverse_aggregate = aggregate_bounded_reports(&reverse, adversarial, adversarial_parity)?;
    let forward_bytes = serde_json::to_vec(&forward_aggregate)?;
    let reverse_bytes = serde_json::to_vec(&reverse_aggregate)?;
    let scientific = BoundedOrderProofScientific {
        schema: BOUNDED_ORDER_PROOF_SCHEMA.to_owned(),
        experiment_id: BOUNDED_EXPERIMENT_ID.to_owned(),
        forward_arm_order: forward
            .iter()
            .map(|report| report.scientific.arm_id.clone())
            .collect(),
        reverse_arm_order: reverse
            .iter()
            .map(|report| report.scientific.arm_id.clone())
            .collect(),
        forward_aggregate_scientific_blake3: forward_aggregate.scientific_blake3.clone(),
        reverse_aggregate_scientific_blake3: reverse_aggregate.scientific_blake3.clone(),
        forward_document_blake3: blake3::hash(&forward_bytes).to_hex().to_string(),
        reverse_document_blake3: blake3::hash(&reverse_bytes).to_hex().to_string(),
        byte_identical: forward_bytes == reverse_bytes,
    };
    let scientific_blake3 = scientific_hash(&scientific)?;
    Ok((
        forward_aggregate,
        reverse_aggregate,
        BoundedOrderProofReport {
            scientific,
            scientific_blake3,
        },
    ))
}

fn checked_u64(value: usize, field: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| R4Error::DatasetContract(format!("{field} does not fit u64")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::census::frozen_dataset_identities_for_test;
    use crate::{compare_bounded_adversarial_reports, run_bounded_adversarial_suite};

    fn histogram(value: u64) -> Histogram {
        Histogram {
            bins: BTreeMap::from([(value, FROZEN_RECORD_COUNT as u64)]),
        }
    }

    fn synthetic_report(arm: BoundedArm) -> BoundedArmReport {
        let tokens = match arm {
            BoundedArm::SeatMarginal => 160,
            BoundedArm::Directional => 174,
            BoundedArm::Affordance => 170,
            BoundedArm::SelectiveExact => 184,
        };
        let scientific = BoundedArmScientific {
            schema: BOUNDED_REPORT_SCHEMA.to_owned(),
            experiment_id: BOUNDED_EXPERIMENT_ID.to_owned(),
            arm_id: arm.id().to_owned(),
            hard_token_max: arm.hard_token_max(),
            corpus_contract: CORPUS_CONTRACT.to_owned(),
            datasets: frozen_dataset_identities_for_test(),
            record_count: FROZEN_RECORD_COUNT,
            source_stream_blake3: "source".to_owned(),
            exact_state_stream_blake3: "exact".to_owned(),
            full_hwf_stream_blake3: "full-hwf".to_owned(),
            bounded_feature_stream_blake3: arm.id().to_owned(),
            exact_codec_round_trip_checks: FROZEN_RECORD_COUNT as u64,
            r2_semantic_equality_checks: FROZEN_RECORD_COUNT as u64,
            target_independence_checks: FROZEN_RECORD_COUNT as u64,
            deterministic_construction_checks: FROZEN_RECORD_COUNT as u64,
            bounded_codec_round_trip_checks: FROZEN_RECORD_COUNT as u64,
            source_accounting_checks: FROZEN_RECORD_COUNT as u64,
            d6_inverse_checks: FROZEN_RECORD_COUNT as u64 * 12,
            d6_shape_covariance_checks: FROZEN_RECORD_COUNT as u64 * 12,
            packed_byte_histogram: histogram(765),
            full_hwf_token_histogram: histogram(271),
            full_hwf_byte_histogram: histogram(48_000),
            bounded_token_histogram: histogram(tokens),
            active_scalar_histogram: histogram(8_000),
            padded_scalar_histogram: histogram(16_000),
            bounded_byte_histogram: histogram(33_000),
            habitat_component_histogram: histogram(40),
            wildlife_component_histogram: histogram(30),
            source_wildlife_bucket_histogram: histogram(40),
            summarized_wildlife_bucket_histogram: histogram(24),
            exact_wildlife_bucket_histogram: histogram(if arm == BoundedArm::SelectiveExact {
                16
            } else {
                0
            }),
            source_frontier_bucket_histogram: histogram(50),
            summarized_frontier_bucket_histogram: histogram(26),
            exact_frontier_bucket_histogram: histogram(if arm == BoundedArm::SelectiveExact {
                24
            } else {
                0
            }),
        };
        let operational = BoundedArmOperational {
            elapsed_seconds: 120.0,
            records_per_second: 500.0,
            rayon_threads: 10,
            full_hwf_view_nanoseconds: 1_200,
            bounded_view_nanoseconds: 1_000,
            bounded_to_full_hwf_throughput_ratio: 1.2,
        };
        BoundedArmReport {
            scientific_blake3: scientific_hash(&scientific).unwrap(),
            operational_blake3: scientific_hash(&operational).unwrap(),
            scientific,
            operational,
        }
    }

    #[test]
    fn synthetic_four_arm_aggregate_passes_and_selects_successors() {
        let adversarial = run_bounded_adversarial_suite().unwrap();
        let parity = compare_bounded_adversarial_reports(&[
            adversarial.clone(),
            adversarial.clone(),
            adversarial.clone(),
            adversarial.clone(),
        ])
        .unwrap();
        let reports = BoundedArm::ALL.map(synthetic_report);
        let aggregate = aggregate_bounded_reports(&reports, &adversarial, &parity).unwrap();
        assert_eq!(
            aggregate.scientific.classification,
            BoundedClassification::R4BoundedQuotientFoundationPassed
        );
        assert_eq!(
            aggregate
                .scientific
                .promotion_assessment
                .minimal_successor
                .as_deref(),
            Some(BoundedArm::SeatMarginal.id())
        );
        assert_eq!(
            aggregate
                .scientific
                .promotion_assessment
                .richest_successor
                .as_deref(),
            Some(BoundedArm::SelectiveExact.id())
        );
        assert!(
            aggregate
                .scientific
                .promotion_assessment
                .authorize_matched_mlx
        );
    }

    #[test]
    fn bounded_aggregate_is_byte_invariant_to_arm_input_order() {
        let adversarial = run_bounded_adversarial_suite().unwrap();
        let parity = compare_bounded_adversarial_reports(&[
            adversarial.clone(),
            adversarial.clone(),
            adversarial.clone(),
            adversarial.clone(),
        ])
        .unwrap();
        let reports = BoundedArm::ALL.map(synthetic_report);
        let (forward, reverse, proof) =
            aggregate_bounded_reports_with_order_proof(&reports, &adversarial, &parity).unwrap();
        assert_eq!(forward, reverse);
        assert!(proof.scientific.byte_identical);
    }
}
