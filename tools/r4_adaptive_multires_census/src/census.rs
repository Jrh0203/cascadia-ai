use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use cascadia_data::{
    DatasetManifest, DatasetSplit, FEATURE_SCHEMA, PositionRecord, PositionShardReader, TARGET_DIM,
    validate_dataset,
};
use cascadia_game::D6Transform;
use r2_sparse_entity_census::SparsePublicState;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    ABLATIONS, AdaptiveMultiResolutionState, FeatureAblation, NearFieldRadius, R4Error, Result,
};

pub const EXPERIMENT_ID: &str = "r4-adaptive-multires-foundation-v1";
pub const REPORT_SCHEMA: &str = "r4-adaptive-multires-foundation-report-v1";
pub const AGGREGATE_SCHEMA: &str = "r4-adaptive-multires-foundation-aggregate-v1";
pub const ORDER_PROOF_SCHEMA: &str = "r4-adaptive-multires-order-proof-v1";
pub const FROZEN_RECORD_COUNT: usize = 60_000;
pub const FROZEN_SHARD_COUNT: u8 = 4;
const PACKED_P99_LIMIT: u64 = 864;
const RADIUS4_HWF_P99_LIMIT: u64 = 256;
const RADIUS5_HWF_P99_LIMIT: u64 = 288;

const FROZEN_DATASETS: [FrozenDataset; 8] = [
    FrozenDataset {
        part: 0,
        dataset_id: "pattern-aware-v1-k8-h6-b8-m4-train-200000",
        split: "train",
        rows: 12_560,
        manifest_blake3: "57f86b3f6ae06bee782974995aa6b8d3cad6f637e68d5ef8aac7ffd8112d4244",
    },
    FrozenDataset {
        part: 1,
        dataset_id: "pattern-aware-v1-k8-h6-b8-m4-train-200157",
        split: "train",
        rows: 12_480,
        manifest_blake3: "79bcceebd52144f8c39130de15404f0f2820b695111f2f1e9004dcac5f33c555",
    },
    FrozenDataset {
        part: 2,
        dataset_id: "pattern-aware-v1-k8-h6-b8-m4-train-200313",
        split: "train",
        rows: 12_480,
        manifest_blake3: "fbddc7aa1794b753fcbd3d8f030b51dcc4456051f61f7914eab541e9658db666",
    },
    FrozenDataset {
        part: 3,
        dataset_id: "pattern-aware-v1-k8-h6-b8-m4-train-200469",
        split: "train",
        rows: 12_480,
        manifest_blake3: "8ab6d2a9229f3cfe8bf1567c3a9d110b9268e322a0c96cf30ba131c937435849",
    },
    FrozenDataset {
        part: 0,
        dataset_id: "pattern-aware-v1-k8-h6-b8-m4-validation-210000",
        split: "validation",
        rows: 2_560,
        manifest_blake3: "a991d05962965d61a31d40fe0b8572c743cff04a12d1e948be9e2fa3e6a871d4",
    },
    FrozenDataset {
        part: 1,
        dataset_id: "pattern-aware-v1-k8-h6-b8-m4-validation-210032",
        split: "validation",
        rows: 2_480,
        manifest_blake3: "adf3903a59d9d522fbb9fab2bb3c8a9370c7f2d46c3aa74ac85b6879b80efddc",
    },
    FrozenDataset {
        part: 2,
        dataset_id: "pattern-aware-v1-k8-h6-b8-m4-validation-210063",
        split: "validation",
        rows: 2_480,
        manifest_blake3: "9bfeed300489ac6610313dd2bf032c809197be92cfeac43b357a4cb8aca14803",
    },
    FrozenDataset {
        part: 3,
        dataset_id: "pattern-aware-v1-k8-h6-b8-m4-validation-210094",
        split: "validation",
        rows: 2_480,
        manifest_blake3: "7491212c5a524f954414402661a6aa064161a16cfe23755e051d80886b257186",
    },
];

#[derive(Debug, Clone, Copy)]
struct FrozenDataset {
    part: u8,
    dataset_id: &'static str,
    split: &'static str,
    rows: usize,
    manifest_blake3: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetIdentity {
    pub dataset_id: String,
    pub split: String,
    pub total_records: usize,
    pub manifest_blake3: String,
    pub shard_blake3s: Vec<String>,
}

#[derive(Debug)]
pub(crate) struct ValidatedDataset {
    pub(crate) root: PathBuf,
    pub(crate) manifest: DatasetManifest,
    pub(crate) identity: DatasetIdentity,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Histogram {
    pub bins: BTreeMap<u64, u64>,
}

impl Histogram {
    pub fn observe(&mut self, value: u64) {
        *self.bins.entry(value).or_default() += 1;
    }

    pub fn merge(&mut self, other: &Self) {
        for (value, count) in &other.bins {
            *self.bins.entry(*value).or_default() += count;
        }
    }

    pub fn count(&self) -> u64 {
        self.bins.values().sum()
    }

    pub fn sum(&self) -> u64 {
        self.bins
            .iter()
            .map(|(value, count)| value.saturating_mul(*count))
            .sum()
    }

    pub fn summary(&self) -> Result<DistributionSummary> {
        let count = self.count();
        if count == 0 {
            return Err(R4Error::AggregateContract(
                "cannot summarize an empty histogram".to_owned(),
            ));
        }
        Ok(DistributionSummary {
            count,
            sum: self.sum(),
            mean: self.sum() as f64 / count as f64,
            median: self.nearest_rank(50),
            p90: self.nearest_rank(90),
            p99: self.nearest_rank(99),
            max: *self.bins.last_key_value().expect("histogram is nonempty").0,
        })
    }

    fn nearest_rank(&self, percentile: u64) -> u64 {
        let target = (percentile * self.count()).div_ceil(100).max(1);
        let mut cumulative = 0u64;
        for (value, count) in &self.bins {
            cumulative += count;
            if cumulative >= target {
                return *value;
            }
        }
        unreachable!("nonempty histogram reaches its own count")
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributionSummary {
    pub count: u64,
    pub sum: u64,
    pub mean: f64,
    pub median: u64,
    pub p90: u64,
    pub p99: u64,
    pub max: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AblationShard {
    pub id: String,
    pub token_histogram: Histogram,
    pub feature_byte_histogram: Histogram,
    pub feature_stream_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RadiusShard {
    pub radius_id: String,
    pub capacity: usize,
    pub record_count: usize,
    pub codec_round_trip_checks: u64,
    pub r2_semantic_equality_checks: u64,
    pub d6_inverse_checks: u64,
    pub target_independence_checks: u64,
    pub exact_state_stream_blake3: String,
    pub packed_byte_histogram: Histogram,
    pub focal_local_occupied_histogram: Histogram,
    pub focal_overflow_occupied_histogram: Histogram,
    pub opponent_exact_occupied_histogram: Histogram,
    pub habitat_token_histogram: Histogram,
    pub wildlife_component_histogram: Histogram,
    pub wildlife_bucket_histogram: Histogram,
    pub frontier_bucket_histogram: Histogram,
    pub ablations: Vec<AblationShard>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScientificShard {
    pub schema: String,
    pub experiment_id: String,
    pub shard_index: u8,
    pub shard_count: u8,
    pub corpus_contract: String,
    pub datasets: Vec<DatasetIdentity>,
    pub record_count: usize,
    pub source_stream_blake3: String,
    pub radii: Vec<RadiusShard>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationalShard {
    pub elapsed_seconds: f64,
    pub records_per_second: f64,
    pub rayon_threads: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardReport {
    pub scientific: ScientificShard,
    pub scientific_blake3: String,
    pub operational: OperationalShard,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AblationAggregate {
    pub id: String,
    pub token_distribution: DistributionSummary,
    pub feature_byte_distribution: DistributionSummary,
    pub ordered_shard_stream_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RadiusAggregate {
    pub radius_id: String,
    pub capacity: usize,
    pub record_count: usize,
    pub codec_round_trip_checks: u64,
    pub r2_semantic_equality_checks: u64,
    pub d6_inverse_checks: u64,
    pub target_independence_checks: u64,
    pub ordered_exact_state_stream_blake3: String,
    pub packed_bytes: DistributionSummary,
    pub focal_local_occupied: DistributionSummary,
    pub focal_overflow_occupied: DistributionSummary,
    pub opponent_exact_occupied: DistributionSummary,
    pub habitat_tokens: DistributionSummary,
    pub wildlife_components: DistributionSummary,
    pub wildlife_buckets: DistributionSummary,
    pub frontier_buckets: DistributionSummary,
    pub ablations: Vec<AblationAggregate>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum R4Classification {
    Passed,
    InformationInsufficient,
    CompactnessFailed,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionAssessment {
    pub exact_mechanics: bool,
    pub frozen_corpus_complete: bool,
    pub packed_p99_within_864: bool,
    pub radius4_hwf_p99_within_256: bool,
    pub radius5_hwf_p99_within_288: bool,
    pub adversarial_suite_passed: bool,
    pub order_invariant: bool,
    pub authorize_mlx: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScientificAggregate {
    pub schema: String,
    pub experiment_id: String,
    pub corpus_contract: String,
    pub datasets: Vec<DatasetIdentity>,
    pub record_count: usize,
    pub ordered_source_stream_blake3: String,
    pub radii: Vec<RadiusAggregate>,
    pub adversarial_suite_scientific_blake3: Option<String>,
    pub adversarial_suite_passed: bool,
    pub promotion_assessment: PromotionAssessment,
    pub classification: R4Classification,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AggregateReport {
    pub scientific: ScientificAggregate,
    pub scientific_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrderProofScientific {
    pub schema: String,
    pub experiment_id: String,
    pub forward_shard_order: Vec<u8>,
    pub reverse_shard_order: Vec<u8>,
    pub forward_aggregate_scientific_blake3: String,
    pub reverse_aggregate_scientific_blake3: String,
    pub forward_document_blake3: String,
    pub reverse_document_blake3: String,
    pub byte_identical: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrderProofReport {
    pub scientific: OrderProofScientific,
    pub scientific_blake3: String,
}

#[derive(Debug)]
struct RowAblation {
    tokens: u64,
    feature_bytes: u64,
    digest: [u8; 32],
}

#[derive(Debug)]
struct RowRadius {
    packed_bytes: u64,
    exact_digest: [u8; 32],
    focal_local_occupied: u64,
    focal_overflow_occupied: u64,
    opponent_exact_occupied: u64,
    habitat_tokens: u64,
    wildlife_components: u64,
    wildlife_buckets: u64,
    frontier_buckets: u64,
    ablations: Vec<RowAblation>,
    codec_checks: u64,
    semantic_checks: u64,
    d6_checks: u64,
    target_checks: u64,
}

#[derive(Debug)]
struct RowResult {
    source_digest: [u8; 32],
    radii: Vec<RowRadius>,
}

pub fn census_datasets(
    roots: &[PathBuf],
    shard_index: u8,
    shard_count: u8,
    require_frozen: bool,
) -> Result<ShardReport> {
    if shard_count == 0 || shard_index >= shard_count {
        return Err(R4Error::DatasetContract(
            "shard index/count are invalid".to_owned(),
        ));
    }
    if require_frozen && shard_count != FROZEN_SHARD_COUNT {
        return Err(R4Error::DatasetContract(format!(
            "frozen corpus requires shard count {FROZEN_SHARD_COUNT}"
        )));
    }
    let datasets = validate_inputs(roots)?;
    if require_frozen {
        validate_frozen_shard(&datasets, shard_index)?;
    }

    let started = Instant::now();
    let mut source_hasher = blake3::Hasher::new();
    let mut radius_accumulators = NearFieldRadius::ALL.map(RadiusAccumulator::new);
    let mut record_count = 0usize;

    for dataset in &datasets {
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
        let rows = records
            .par_iter()
            .enumerate()
            .map(|(ordinal, record)| {
                process_record(record).map_err(|error| {
                    R4Error::DatasetContract(format!(
                        "dataset {} record {ordinal} failed: {error}",
                        dataset.manifest.dataset_id
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        for row in rows {
            update_framed_hash(&mut source_hasher, &row.source_digest);
            for (accumulator, radius_row) in radius_accumulators.iter_mut().zip(row.radii) {
                accumulator.observe(radius_row);
            }
            record_count += 1;
        }
    }
    if record_count == 0 {
        return Err(R4Error::DatasetContract(
            "R4 census selected zero records".to_owned(),
        ));
    }
    let scientific = ScientificShard {
        schema: REPORT_SCHEMA.to_owned(),
        experiment_id: EXPERIMENT_ID.to_owned(),
        shard_index,
        shard_count,
        corpus_contract: if require_frozen {
            "accepted-r0-r2-60000-v1".to_owned()
        } else {
            "validated-compact-entity-v2".to_owned()
        },
        datasets: datasets
            .iter()
            .map(|dataset| dataset.identity.clone())
            .collect(),
        record_count,
        source_stream_blake3: source_hasher.finalize().to_hex().to_string(),
        radii: radius_accumulators
            .into_iter()
            .map(RadiusAccumulator::finish)
            .collect(),
    };
    let scientific_blake3 = scientific_hash(&scientific)?;
    let elapsed_seconds = started.elapsed().as_secs_f64();
    Ok(ShardReport {
        scientific,
        scientific_blake3,
        operational: OperationalShard {
            elapsed_seconds,
            records_per_second: record_count as f64 / elapsed_seconds,
            rayon_threads: rayon::current_num_threads(),
        },
    })
}

struct RadiusAccumulator {
    radius: NearFieldRadius,
    record_count: usize,
    codec_checks: u64,
    semantic_checks: u64,
    d6_checks: u64,
    target_checks: u64,
    exact_hasher: blake3::Hasher,
    packed_bytes: Histogram,
    focal_local: Histogram,
    focal_overflow: Histogram,
    opponent_exact: Histogram,
    habitat: Histogram,
    wildlife_components: Histogram,
    wildlife_buckets: Histogram,
    frontier: Histogram,
    ablations: Vec<AblationAccumulator>,
}

struct AblationAccumulator {
    arm: FeatureAblation,
    tokens: Histogram,
    feature_bytes: Histogram,
    hasher: blake3::Hasher,
}

impl RadiusAccumulator {
    fn new(radius: NearFieldRadius) -> Self {
        Self {
            radius,
            record_count: 0,
            codec_checks: 0,
            semantic_checks: 0,
            d6_checks: 0,
            target_checks: 0,
            exact_hasher: blake3::Hasher::new(),
            packed_bytes: Histogram::default(),
            focal_local: Histogram::default(),
            focal_overflow: Histogram::default(),
            opponent_exact: Histogram::default(),
            habitat: Histogram::default(),
            wildlife_components: Histogram::default(),
            wildlife_buckets: Histogram::default(),
            frontier: Histogram::default(),
            ablations: ABLATIONS
                .into_iter()
                .map(|arm| AblationAccumulator {
                    arm,
                    tokens: Histogram::default(),
                    feature_bytes: Histogram::default(),
                    hasher: blake3::Hasher::new(),
                })
                .collect(),
        }
    }

    fn observe(&mut self, row: RowRadius) {
        self.record_count += 1;
        self.codec_checks += row.codec_checks;
        self.semantic_checks += row.semantic_checks;
        self.d6_checks += row.d6_checks;
        self.target_checks += row.target_checks;
        update_framed_hash(&mut self.exact_hasher, &row.exact_digest);
        self.packed_bytes.observe(row.packed_bytes);
        self.focal_local.observe(row.focal_local_occupied);
        self.focal_overflow.observe(row.focal_overflow_occupied);
        self.opponent_exact.observe(row.opponent_exact_occupied);
        self.habitat.observe(row.habitat_tokens);
        self.wildlife_components.observe(row.wildlife_components);
        self.wildlife_buckets.observe(row.wildlife_buckets);
        self.frontier.observe(row.frontier_buckets);
        for (accumulator, ablation) in self.ablations.iter_mut().zip(row.ablations) {
            accumulator.tokens.observe(ablation.tokens);
            accumulator.feature_bytes.observe(ablation.feature_bytes);
            update_framed_hash(&mut accumulator.hasher, &ablation.digest);
        }
    }

    fn finish(self) -> RadiusShard {
        RadiusShard {
            radius_id: self.radius.id().to_owned(),
            capacity: self.radius.capacity(),
            record_count: self.record_count,
            codec_round_trip_checks: self.codec_checks,
            r2_semantic_equality_checks: self.semantic_checks,
            d6_inverse_checks: self.d6_checks,
            target_independence_checks: self.target_checks,
            exact_state_stream_blake3: self.exact_hasher.finalize().to_hex().to_string(),
            packed_byte_histogram: self.packed_bytes,
            focal_local_occupied_histogram: self.focal_local,
            focal_overflow_occupied_histogram: self.focal_overflow,
            opponent_exact_occupied_histogram: self.opponent_exact,
            habitat_token_histogram: self.habitat,
            wildlife_component_histogram: self.wildlife_components,
            wildlife_bucket_histogram: self.wildlife_buckets,
            frontier_bucket_histogram: self.frontier,
            ablations: self
                .ablations
                .into_iter()
                .map(|arm| AblationShard {
                    id: arm.arm.id().to_owned(),
                    token_histogram: arm.tokens,
                    feature_byte_histogram: arm.feature_bytes,
                    feature_stream_blake3: arm.hasher.finalize().to_hex().to_string(),
                })
                .collect(),
        }
    }
}

fn process_record(record: &PositionRecord) -> Result<RowResult> {
    let mut public_record = record.clone();
    public_record.targets = [0; TARGET_DIM];
    let source_digest = *blake3::hash(&public_record.to_bytes()).as_bytes();
    let sparse = SparsePublicState::from_position_record(&public_record, None)?;
    let mut radii = Vec::with_capacity(NearFieldRadius::ALL.len());
    for radius in NearFieldRadius::ALL {
        let state = AdaptiveMultiResolutionState::from_sparse_state(&sparse, radius)?;
        let packed = state.to_packed_bytes()?;
        let decoded = AdaptiveMultiResolutionState::from_packed_bytes(&packed)?;
        if decoded != state || decoded.to_sparse_state()? != sparse {
            return Err(R4Error::DatasetContract(
                "R4 codec or R2 semantic round trip changed the state".to_owned(),
            ));
        }

        let mut changed_targets = record.clone();
        for target in &mut changed_targets.targets {
            *target = !*target;
        }
        let changed =
            AdaptiveMultiResolutionState::from_position_record(&changed_targets, None, radius)?;
        if changed.to_packed_bytes()? != packed {
            return Err(R4Error::DatasetContract(
                "target mutation changed R4 authoritative bytes".to_owned(),
            ));
        }

        let mut d6_checks = 0u64;
        for transform in D6Transform::ALL {
            let transformed = state.transformed(transform)?;
            let restored = transformed.transformed(transform.inverse())?;
            if restored != state {
                return Err(R4Error::DatasetContract(format!(
                    "R4 D6 inverse changed {} under transform {}",
                    radius.id(),
                    transform.id()
                )));
            }
            d6_checks += 1;
        }

        let focal = &state.boards[usize::from(state.focal_relative_seat)];
        let opponent_exact_occupied = state
            .boards
            .iter()
            .filter(|board| !board.is_focal)
            .map(|board| board.exact_far.occupied_tiles.len())
            .sum::<usize>();
        let habitat_tokens = state
            .boards
            .iter()
            .map(|board| board.far_habitat_components.len())
            .sum::<usize>();
        let wildlife_components = state
            .boards
            .iter()
            .map(|board| board.far_wildlife_components.len())
            .sum::<usize>();
        let wildlife_buckets = state
            .boards
            .iter()
            .map(|board| board.far_wildlife_motif_buckets.len())
            .sum::<usize>();
        let frontier_buckets = state
            .boards
            .iter()
            .map(|board| board.far_frontier_buckets.len())
            .sum::<usize>();
        let mut ablations = Vec::with_capacity(ABLATIONS.len());
        for arm in ABLATIONS {
            let view = state.feature_view(arm)?;
            let bytes = view.canonical_bytes()?;
            let changed_view = changed.feature_view(arm)?.canonical_bytes()?;
            if bytes != changed_view {
                return Err(R4Error::DatasetContract(format!(
                    "target mutation changed {} {} feature bytes",
                    radius.id(),
                    arm.id()
                )));
            }
            ablations.push(RowAblation {
                tokens: view.spatial_token_count() as u64,
                feature_bytes: bytes.len() as u64,
                digest: *blake3::hash(&bytes).as_bytes(),
            });
        }
        radii.push(RowRadius {
            packed_bytes: packed.len() as u64,
            exact_digest: *blake3::hash(&packed).as_bytes(),
            focal_local_occupied: focal.authority_local_occupied.len() as u64,
            focal_overflow_occupied: focal.authority_overflow_occupied.len() as u64,
            opponent_exact_occupied: opponent_exact_occupied as u64,
            habitat_tokens: habitat_tokens as u64,
            wildlife_components: wildlife_components as u64,
            wildlife_buckets: wildlife_buckets as u64,
            frontier_buckets: frontier_buckets as u64,
            ablations,
            codec_checks: 1,
            semantic_checks: 1,
            d6_checks,
            target_checks: 1,
        });
    }
    Ok(RowResult {
        source_digest,
        radii,
    })
}

pub fn aggregate_reports(
    reports: &[ShardReport],
    adversarial_suite_scientific_blake3: Option<String>,
    adversarial_suite_passed: bool,
) -> Result<AggregateReport> {
    if reports.len() != usize::from(FROZEN_SHARD_COUNT) {
        return Err(R4Error::AggregateContract(format!(
            "expected {FROZEN_SHARD_COUNT} shard reports; received {}",
            reports.len()
        )));
    }
    let mut by_index = BTreeMap::new();
    for report in reports {
        if scientific_hash(&report.scientific)? != report.scientific_blake3 {
            return Err(R4Error::AggregateContract(format!(
                "shard {} scientific hash mismatch",
                report.scientific.shard_index
            )));
        }
        if report.scientific.schema != REPORT_SCHEMA
            || report.scientific.experiment_id != EXPERIMENT_ID
            || report.scientific.shard_count != FROZEN_SHARD_COUNT
            || report.scientific.corpus_contract != "accepted-r0-r2-60000-v1"
        {
            return Err(R4Error::AggregateContract(
                "shard report contract drifted".to_owned(),
            ));
        }
        if by_index
            .insert(report.scientific.shard_index, report)
            .is_some()
        {
            return Err(R4Error::AggregateContract(
                "duplicate shard index".to_owned(),
            ));
        }
    }
    if by_index.keys().copied().collect::<Vec<_>>() != (0..FROZEN_SHARD_COUNT).collect::<Vec<_>>() {
        return Err(R4Error::AggregateContract(
            "shard index coverage is incomplete".to_owned(),
        ));
    }

    let mut datasets = Vec::new();
    let mut record_count = 0usize;
    let mut source_hasher = blake3::Hasher::new();
    let mut radius_merges = NearFieldRadius::ALL.map(RadiusMerge::new);
    for report in by_index.values() {
        datasets.extend(report.scientific.datasets.clone());
        record_count += report.scientific.record_count;
        update_framed_hash(
            &mut source_hasher,
            report.scientific.source_stream_blake3.as_bytes(),
        );
        if report.scientific.radii.len() != NearFieldRadius::ALL.len() {
            return Err(R4Error::AggregateContract(
                "shard radius count drifted".to_owned(),
            ));
        }
        for (merge, radius) in radius_merges.iter_mut().zip(&report.scientific.radii) {
            merge.merge(radius)?;
        }
    }
    validate_frozen_aggregate_datasets(&datasets)?;
    if record_count != FROZEN_RECORD_COUNT {
        return Err(R4Error::AggregateContract(format!(
            "aggregate has {record_count} records; expected {FROZEN_RECORD_COUNT}"
        )));
    }
    datasets.sort_unstable_by_key(dataset_order);
    let radii = radius_merges
        .into_iter()
        .map(RadiusMerge::finish)
        .collect::<Result<Vec<_>>>()?;
    let radius4 = radii
        .iter()
        .find(|radius| radius.radius_id == NearFieldRadius::Radius4.id())
        .expect("radius four aggregate exists");
    let radius5 = radii
        .iter()
        .find(|radius| radius.radius_id == NearFieldRadius::Radius5.id())
        .expect("radius five aggregate exists");
    let radius4_hwf = radius4
        .ablations
        .iter()
        .find(|arm| arm.id == FeatureAblation::AllTopology.id())
        .expect("radius four HWF arm exists");
    let radius5_hwf = radius5
        .ablations
        .iter()
        .find(|arm| arm.id == FeatureAblation::AllTopology.id())
        .expect("radius five HWF arm exists");
    let exact_mechanics = radii.iter().all(|radius| {
        radius.codec_round_trip_checks == record_count as u64
            && radius.r2_semantic_equality_checks == record_count as u64
            && radius.target_independence_checks == record_count as u64
            && radius.d6_inverse_checks == record_count as u64 * 12
    });
    let packed_p99_within_864 = radii
        .iter()
        .all(|radius| radius.packed_bytes.p99 <= PACKED_P99_LIMIT);
    let radius4_hwf_p99_within_256 = radius4_hwf.token_distribution.p99 <= RADIUS4_HWF_P99_LIMIT;
    let radius5_hwf_p99_within_288 = radius5_hwf.token_distribution.p99 <= RADIUS5_HWF_P99_LIMIT;
    let compactness =
        packed_p99_within_864 && radius4_hwf_p99_within_256 && radius5_hwf_p99_within_288;
    let adversarial_evidence_present = adversarial_suite_scientific_blake3.is_some();
    let authorize_mlx = exact_mechanics
        && compactness
        && adversarial_evidence_present
        && adversarial_suite_passed
        && record_count == FROZEN_RECORD_COUNT;
    let classification =
        if !exact_mechanics || record_count != FROZEN_RECORD_COUNT || !adversarial_evidence_present
        {
            R4Classification::Invalid
        } else if !adversarial_suite_passed {
            R4Classification::InformationInsufficient
        } else if !compactness {
            R4Classification::CompactnessFailed
        } else {
            R4Classification::Passed
        };
    let scientific = ScientificAggregate {
        schema: AGGREGATE_SCHEMA.to_owned(),
        experiment_id: EXPERIMENT_ID.to_owned(),
        corpus_contract: "accepted-r0-r2-60000-v1".to_owned(),
        datasets,
        record_count,
        ordered_source_stream_blake3: source_hasher.finalize().to_hex().to_string(),
        radii,
        adversarial_suite_scientific_blake3,
        adversarial_suite_passed,
        promotion_assessment: PromotionAssessment {
            exact_mechanics,
            frozen_corpus_complete: record_count == FROZEN_RECORD_COUNT,
            packed_p99_within_864,
            radius4_hwf_p99_within_256,
            radius5_hwf_p99_within_288,
            adversarial_suite_passed,
            order_invariant: true,
            authorize_mlx,
        },
        classification,
    };
    let scientific_blake3 = scientific_hash(&scientific)?;
    Ok(AggregateReport {
        scientific,
        scientific_blake3,
    })
}

pub fn aggregate_reports_with_order_proof(
    reports: &[ShardReport],
    adversarial_suite_scientific_blake3: Option<String>,
    adversarial_suite_passed: bool,
) -> Result<(AggregateReport, AggregateReport, OrderProofReport)> {
    let mut forward_reports = reports.iter().collect::<Vec<_>>();
    forward_reports.sort_unstable_by_key(|report| report.scientific.shard_index);
    let mut reverse_reports = forward_reports.clone();
    reverse_reports.reverse();
    let forward_owned = forward_reports
        .iter()
        .map(|report| (*report).clone())
        .collect::<Vec<_>>();
    let reverse_owned = reverse_reports
        .iter()
        .map(|report| (*report).clone())
        .collect::<Vec<_>>();
    let forward = aggregate_reports(
        &forward_owned,
        adversarial_suite_scientific_blake3.clone(),
        adversarial_suite_passed,
    )?;
    let reverse = aggregate_reports(
        &reverse_owned,
        adversarial_suite_scientific_blake3,
        adversarial_suite_passed,
    )?;
    let forward_bytes = serde_json::to_vec(&forward)?;
    let reverse_bytes = serde_json::to_vec(&reverse)?;
    let byte_identical = forward_bytes == reverse_bytes;
    let scientific = OrderProofScientific {
        schema: ORDER_PROOF_SCHEMA.to_owned(),
        experiment_id: EXPERIMENT_ID.to_owned(),
        forward_shard_order: forward_reports
            .iter()
            .map(|report| report.scientific.shard_index)
            .collect(),
        reverse_shard_order: reverse_reports
            .iter()
            .map(|report| report.scientific.shard_index)
            .collect(),
        forward_aggregate_scientific_blake3: forward.scientific_blake3.clone(),
        reverse_aggregate_scientific_blake3: reverse.scientific_blake3.clone(),
        forward_document_blake3: blake3::hash(&forward_bytes).to_hex().to_string(),
        reverse_document_blake3: blake3::hash(&reverse_bytes).to_hex().to_string(),
        byte_identical,
    };
    let scientific_blake3 = scientific_hash(&scientific)?;
    Ok((
        forward,
        reverse,
        OrderProofReport {
            scientific,
            scientific_blake3,
        },
    ))
}

struct RadiusMerge {
    radius: NearFieldRadius,
    record_count: usize,
    codec_checks: u64,
    semantic_checks: u64,
    d6_checks: u64,
    target_checks: u64,
    exact_hasher: blake3::Hasher,
    packed: Histogram,
    focal_local: Histogram,
    focal_overflow: Histogram,
    opponent_exact: Histogram,
    habitat: Histogram,
    wildlife_components: Histogram,
    wildlife_buckets: Histogram,
    frontier: Histogram,
    ablations: Vec<AblationMerge>,
}

struct AblationMerge {
    arm: FeatureAblation,
    tokens: Histogram,
    bytes: Histogram,
    hasher: blake3::Hasher,
}

impl RadiusMerge {
    fn new(radius: NearFieldRadius) -> Self {
        Self {
            radius,
            record_count: 0,
            codec_checks: 0,
            semantic_checks: 0,
            d6_checks: 0,
            target_checks: 0,
            exact_hasher: blake3::Hasher::new(),
            packed: Histogram::default(),
            focal_local: Histogram::default(),
            focal_overflow: Histogram::default(),
            opponent_exact: Histogram::default(),
            habitat: Histogram::default(),
            wildlife_components: Histogram::default(),
            wildlife_buckets: Histogram::default(),
            frontier: Histogram::default(),
            ablations: ABLATIONS
                .into_iter()
                .map(|arm| AblationMerge {
                    arm,
                    tokens: Histogram::default(),
                    bytes: Histogram::default(),
                    hasher: blake3::Hasher::new(),
                })
                .collect(),
        }
    }

    fn merge(&mut self, radius: &RadiusShard) -> Result<()> {
        if radius.radius_id != self.radius.id()
            || radius.capacity != self.radius.capacity()
            || radius.ablations.len() != ABLATIONS.len()
        {
            return Err(R4Error::AggregateContract(
                "radius shard schema drifted".to_owned(),
            ));
        }
        self.record_count += radius.record_count;
        self.codec_checks += radius.codec_round_trip_checks;
        self.semantic_checks += radius.r2_semantic_equality_checks;
        self.d6_checks += radius.d6_inverse_checks;
        self.target_checks += radius.target_independence_checks;
        update_framed_hash(
            &mut self.exact_hasher,
            radius.exact_state_stream_blake3.as_bytes(),
        );
        self.packed.merge(&radius.packed_byte_histogram);
        self.focal_local
            .merge(&radius.focal_local_occupied_histogram);
        self.focal_overflow
            .merge(&radius.focal_overflow_occupied_histogram);
        self.opponent_exact
            .merge(&radius.opponent_exact_occupied_histogram);
        self.habitat.merge(&radius.habitat_token_histogram);
        self.wildlife_components
            .merge(&radius.wildlife_component_histogram);
        self.wildlife_buckets
            .merge(&radius.wildlife_bucket_histogram);
        self.frontier.merge(&radius.frontier_bucket_histogram);
        for (merge, arm) in self.ablations.iter_mut().zip(&radius.ablations) {
            if arm.id != merge.arm.id() {
                return Err(R4Error::AggregateContract(
                    "ablation order or ID drifted".to_owned(),
                ));
            }
            merge.tokens.merge(&arm.token_histogram);
            merge.bytes.merge(&arm.feature_byte_histogram);
            update_framed_hash(&mut merge.hasher, arm.feature_stream_blake3.as_bytes());
        }
        Ok(())
    }

    fn finish(self) -> Result<RadiusAggregate> {
        Ok(RadiusAggregate {
            radius_id: self.radius.id().to_owned(),
            capacity: self.radius.capacity(),
            record_count: self.record_count,
            codec_round_trip_checks: self.codec_checks,
            r2_semantic_equality_checks: self.semantic_checks,
            d6_inverse_checks: self.d6_checks,
            target_independence_checks: self.target_checks,
            ordered_exact_state_stream_blake3: self.exact_hasher.finalize().to_hex().to_string(),
            packed_bytes: self.packed.summary()?,
            focal_local_occupied: self.focal_local.summary()?,
            focal_overflow_occupied: self.focal_overflow.summary()?,
            opponent_exact_occupied: self.opponent_exact.summary()?,
            habitat_tokens: self.habitat.summary()?,
            wildlife_components: self.wildlife_components.summary()?,
            wildlife_buckets: self.wildlife_buckets.summary()?,
            frontier_buckets: self.frontier.summary()?,
            ablations: self
                .ablations
                .into_iter()
                .map(|arm| {
                    Ok(AblationAggregate {
                        id: arm.arm.id().to_owned(),
                        token_distribution: arm.tokens.summary()?,
                        feature_byte_distribution: arm.bytes.summary()?,
                        ordered_shard_stream_blake3: arm.hasher.finalize().to_hex().to_string(),
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

pub fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
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

pub(crate) fn validate_inputs(roots: &[PathBuf]) -> Result<Vec<ValidatedDataset>> {
    if roots.is_empty() {
        return Err(R4Error::DatasetContract(
            "at least one dataset root is required".to_owned(),
        ));
    }
    let mut datasets = Vec::with_capacity(roots.len());
    for root in roots {
        let manifest_path = root.join("dataset.json");
        let manifest: DatasetManifest =
            serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
        validate_dataset(root, &manifest)?;
        if manifest.feature_schema != FEATURE_SCHEMA {
            return Err(R4Error::DatasetContract(format!(
                "dataset {} feature schema drifted",
                manifest.dataset_id
            )));
        }
        let identity = DatasetIdentity {
            dataset_id: manifest.dataset_id.clone(),
            split: split_name(manifest.split).to_owned(),
            total_records: manifest.total_records,
            manifest_blake3: hash_file(&manifest_path)?,
            shard_blake3s: manifest
                .shards
                .iter()
                .map(|shard| shard.blake3.clone())
                .collect(),
        };
        datasets.push(ValidatedDataset {
            root: root.clone(),
            manifest,
            identity,
        });
    }
    Ok(datasets)
}

fn validate_frozen_shard(datasets: &[ValidatedDataset], shard_index: u8) -> Result<()> {
    if datasets.len() != 2 {
        return Err(R4Error::DatasetContract(
            "each frozen R4 shard requires one train and one validation root".to_owned(),
        ));
    }
    let mut matched_splits = BTreeSet::new();
    for dataset in datasets {
        let expected = FROZEN_DATASETS
            .iter()
            .find(|expected| {
                expected.part == shard_index && expected.dataset_id == dataset.identity.dataset_id
            })
            .ok_or_else(|| {
                R4Error::DatasetContract(format!(
                    "dataset {} does not belong to frozen shard {shard_index}",
                    dataset.identity.dataset_id
                ))
            })?;
        if dataset.identity.split != expected.split
            || dataset.identity.total_records != expected.rows
            || dataset.identity.manifest_blake3 != expected.manifest_blake3
        {
            return Err(R4Error::DatasetContract(format!(
                "dataset {} identity drifted",
                dataset.identity.dataset_id
            )));
        }
        matched_splits.insert(dataset.identity.split.clone());
    }
    if matched_splits != BTreeSet::from(["train".to_owned(), "validation".to_owned()]) {
        return Err(R4Error::DatasetContract(
            "frozen shard does not contain train and validation".to_owned(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_frozen_aggregate_datasets(datasets: &[DatasetIdentity]) -> Result<()> {
    if datasets.len() != FROZEN_DATASETS.len() {
        return Err(R4Error::AggregateContract(
            "aggregate dataset count is not eight".to_owned(),
        ));
    }
    let mut seen = BTreeSet::new();
    for dataset in datasets {
        if !seen.insert(dataset.dataset_id.clone()) {
            return Err(R4Error::AggregateContract(
                "aggregate contains a duplicate dataset".to_owned(),
            ));
        }
        let expected = FROZEN_DATASETS
            .iter()
            .find(|expected| expected.dataset_id == dataset.dataset_id)
            .ok_or_else(|| R4Error::AggregateContract("unknown frozen dataset".to_owned()))?;
        if dataset.split != expected.split
            || dataset.total_records != expected.rows
            || dataset.manifest_blake3 != expected.manifest_blake3
        {
            return Err(R4Error::AggregateContract(format!(
                "dataset {} identity drifted",
                dataset.dataset_id
            )));
        }
    }
    Ok(())
}

pub(crate) fn dataset_order(dataset: &DatasetIdentity) -> usize {
    FROZEN_DATASETS
        .iter()
        .position(|expected| expected.dataset_id == dataset.dataset_id)
        .expect("aggregate datasets were validated")
}

#[cfg(test)]
pub(crate) fn frozen_dataset_identities_for_test() -> Vec<DatasetIdentity> {
    FROZEN_DATASETS
        .iter()
        .map(|dataset| DatasetIdentity {
            dataset_id: dataset.dataset_id.to_owned(),
            split: dataset.split.to_owned(),
            total_records: dataset.rows,
            manifest_blake3: dataset.manifest_blake3.to_owned(),
            shard_blake3s: Vec::new(),
        })
        .collect()
}

fn split_name(split: DatasetSplit) -> &'static str {
    match split {
        DatasetSplit::Train => "train",
        DatasetSplit::Validation => "validation",
        DatasetSplit::Test => "test",
        DatasetSplit::Final => "final",
    }
}

pub(crate) fn scientific_hash(value: &impl Serialize) -> Result<String> {
    Ok(blake3::hash(&serde_json::to_vec(value)?)
        .to_hex()
        .to_string())
}

pub(crate) fn update_framed_hash(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn hash_file(path: &Path) -> Result<String> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_merges_before_quantiles() {
        let mut left = Histogram::default();
        left.observe(1);
        left.observe(100);
        let mut right = Histogram::default();
        right.observe(2);
        right.observe(3);
        left.merge(&right);
        let summary = left.summary().unwrap();
        assert_eq!(summary.count, 4);
        assert_eq!(summary.median, 2);
        assert_eq!(summary.max, 100);
    }

    #[test]
    fn frozen_dataset_rows_sum_to_sixty_thousand() {
        assert_eq!(
            FROZEN_DATASETS
                .iter()
                .map(|dataset| dataset.rows)
                .sum::<usize>(),
            FROZEN_RECORD_COUNT
        );
    }
}
