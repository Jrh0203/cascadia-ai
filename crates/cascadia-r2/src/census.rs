use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use cascadia_data::{
    DatasetManifest, DatasetSplit, FEATURE_SCHEMA, PositionRecord, PositionShardReader,
    RECORD_SIZE, validate_dataset,
};
use cascadia_game::D6Transform;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{R2Error, Result, SparsePublicState, SuppliedTile};

pub const ACCEPTED_R0_CORPUS_ROWS: usize = 60_000;
const POSITION_TOKEN_P99_BUDGET: u64 = 512;
const POSITION_TOKEN_MAX_BUDGET: u64 = 640;
const PACKED_P99_BUDGET: u64 = RECORD_SIZE as u64;
const RATIO_SCALE: u64 = 1_000_000;
const DENSE_CAPACITIES: [u64; 4] = [61, 91, 127, 441];

const ACCEPTED_R0_DATASETS: [AcceptedDataset; 8] = [
    AcceptedDataset {
        dataset_id: "pattern-aware-v1-k8-h6-b8-m4-train-200000",
        split: "train",
        rows: 12_560,
        manifest_blake3: "57f86b3f6ae06bee782974995aa6b8d3cad6f637e68d5ef8aac7ffd8112d4244",
    },
    AcceptedDataset {
        dataset_id: "pattern-aware-v1-k8-h6-b8-m4-train-200157",
        split: "train",
        rows: 12_480,
        manifest_blake3: "79bcceebd52144f8c39130de15404f0f2820b695111f2f1e9004dcac5f33c555",
    },
    AcceptedDataset {
        dataset_id: "pattern-aware-v1-k8-h6-b8-m4-train-200313",
        split: "train",
        rows: 12_480,
        manifest_blake3: "fbddc7aa1794b753fcbd3d8f030b51dcc4456051f61f7914eab541e9658db666",
    },
    AcceptedDataset {
        dataset_id: "pattern-aware-v1-k8-h6-b8-m4-train-200469",
        split: "train",
        rows: 12_480,
        manifest_blake3: "8ab6d2a9229f3cfe8bf1567c3a9d110b9268e322a0c96cf30ba131c937435849",
    },
    AcceptedDataset {
        dataset_id: "pattern-aware-v1-k8-h6-b8-m4-validation-210000",
        split: "validation",
        rows: 2_560,
        manifest_blake3: "a991d05962965d61a31d40fe0b8572c743cff04a12d1e948be9e2fa3e6a871d4",
    },
    AcceptedDataset {
        dataset_id: "pattern-aware-v1-k8-h6-b8-m4-validation-210032",
        split: "validation",
        rows: 2_480,
        manifest_blake3: "adf3903a59d9d522fbb9fab2bb3c8a9370c7f2d46c3aa74ac85b6879b80efddc",
    },
    AcceptedDataset {
        dataset_id: "pattern-aware-v1-k8-h6-b8-m4-validation-210063",
        split: "validation",
        rows: 2_480,
        manifest_blake3: "9bfeed300489ac6610313dd2bf032c809197be92cfeac43b357a4cb8aca14803",
    },
    AcceptedDataset {
        dataset_id: "pattern-aware-v1-k8-h6-b8-m4-validation-210094",
        split: "validation",
        rows: 2_480,
        manifest_blake3: "7491212c5a524f954414402661a6aa064161a16cfe23755e051d80886b257186",
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorpusRequirement {
    AnyValidatedCompactEntityV2,
    AcceptedR0SixtyThousand,
}

#[derive(Debug, Clone, Copy)]
struct AcceptedDataset {
    dataset_id: &'static str,
    split: &'static str,
    rows: usize,
    manifest_blake3: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardIdentity {
    pub file_name: String,
    pub first_game_index: u64,
    pub game_count: usize,
    pub record_count: usize,
    pub byte_count: u64,
    pub blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetIdentity {
    pub dataset_id: String,
    pub split: String,
    pub feature_schema: String,
    pub total_records: usize,
    pub manifest_blake3: String,
    pub shards: Vec<ShardIdentity>,
}

#[derive(Debug)]
struct ValidatedDataset {
    root: PathBuf,
    manifest: DatasetManifest,
    identity: DatasetIdentity,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistributionSummary {
    pub count: u64,
    pub sum: u64,
    pub mean: f64,
    pub median: u64,
    pub p90: u64,
    pub p99: u64,
    pub max: u64,
}

impl DistributionSummary {
    fn from_values(mut values: Vec<u64>) -> Result<Self> {
        if values.is_empty() {
            return Err(R2Error::DatasetContract(
                "cannot summarize an empty distribution".to_owned(),
            ));
        }
        values.sort_unstable();
        let count = values.len() as u64;
        let sum = values.iter().sum::<u64>();
        Ok(Self {
            count,
            sum,
            mean: sum as f64 / count as f64,
            median: nearest_rank(&values, 50),
            p90: nearest_rank(&values, 90),
            p99: nearest_rank(&values, 99),
            max: *values.last().expect("distribution is nonempty"),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenDistributions {
    pub occupied: DistributionSummary,
    pub frontier: DistributionSummary,
    pub habitat_components: DistributionSummary,
    pub wildlife_motifs: DistributionSummary,
    pub total_spatial_tokens: DistributionSummary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseCapacityComparison {
    pub cells_per_board: u64,
    pub ratio_unit: String,
    pub occupied_token_fraction_ppm: DistributionSummary,
    pub total_spatial_token_fraction_ppm: DistributionSummary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromotionCriterion {
    pub id: String,
    pub threshold: String,
    pub rationale: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct R2PromotionAssessment {
    pub exact_public_round_trip: bool,
    pub exact_pack_round_trip: bool,
    pub frontier_oracle_equality: bool,
    pub habitat_oracle_equality: bool,
    pub exact_d6_inverse: bool,
    pub target_independence: bool,
    pub p99_total_tokens_within_512: bool,
    pub max_total_tokens_within_640: bool,
    pub p99_packed_bytes_within_position_record: bool,
    pub authorize_matched_mlx_prototype: bool,
    pub learned_quality_claim: bool,
    pub gameplay_claim: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificCensus {
    pub experiment_id: String,
    pub schema: String,
    pub schema_version: u16,
    pub packed_schema: String,
    pub feature_schema: String,
    pub corpus_order: String,
    pub accepted_r0_contract_required: bool,
    pub accepted_r0_contract_matched: bool,
    pub datasets: Vec<DatasetIdentity>,
    pub record_count: usize,
    pub board_count: usize,
    pub supplied_tile: Option<SuppliedTile>,
    pub targets_serialized_or_hashed: bool,
    pub public_position_blake3: String,
    pub packed_state_blake3: String,
    pub d6_transform_inverse_checks: u64,
    pub target_independence_checks: u64,
    pub position_tokens: TokenDistributions,
    pub board_tokens: TokenDistributions,
    pub packed_bytes: DistributionSummary,
    pub packed_bytes_fraction_of_compact_entity_v2_ppm: DistributionSummary,
    pub dense_capacity_comparisons: Vec<DenseCapacityComparison>,
    pub frontier_positions_with_habitat_bridge: u64,
    pub frontier_positions_with_repeated_component_contact: u64,
    pub promotion_criteria: Vec<PromotionCriterion>,
    pub promotion_assessment: R2PromotionAssessment,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CensusReport {
    pub scientific: ScientificCensus,
    pub scientific_blake3: String,
}

#[derive(Default)]
struct Counts {
    occupied: Vec<u64>,
    frontier: Vec<u64>,
    components: Vec<u64>,
    motifs: Vec<u64>,
    total: Vec<u64>,
}

impl Counts {
    fn push(&mut self, values: [usize; 5]) {
        self.occupied.push(values[0] as u64);
        self.frontier.push(values[1] as u64);
        self.components.push(values[2] as u64);
        self.motifs.push(values[3] as u64);
        self.total.push(values[4] as u64);
    }

    fn summarize(self) -> Result<TokenDistributions> {
        Ok(TokenDistributions {
            occupied: DistributionSummary::from_values(self.occupied)?,
            frontier: DistributionSummary::from_values(self.frontier)?,
            habitat_components: DistributionSummary::from_values(self.components)?,
            wildlife_motifs: DistributionSummary::from_values(self.motifs)?,
            total_spatial_tokens: DistributionSummary::from_values(self.total)?,
        })
    }
}

struct RowCensus {
    public_bytes: [u8; RECORD_SIZE],
    packed: Vec<u8>,
    player_count: u8,
    position_values: [usize; 5],
    board_values: Vec<[usize; 5]>,
    d6_checks: u64,
    target_independence_checks: u64,
    frontier_positions_with_habitat_bridge: u64,
    frontier_positions_with_repeated_component_contact: u64,
}

pub fn census_datasets(
    roots: &[PathBuf],
    supplied_tile: Option<SuppliedTile>,
    requirement: CorpusRequirement,
) -> Result<CensusReport> {
    let datasets = validate_inputs(roots)?;
    let accepted_r0_contract_matched = matches_accepted_r0(&datasets);
    if requirement == CorpusRequirement::AcceptedR0SixtyThousand && !accepted_r0_contract_matched {
        return Err(R2Error::DatasetContract(
            "dataset identities do not match the frozen accepted R0 corpus in canonical order"
                .to_owned(),
        ));
    }

    let total_manifest_records = datasets
        .iter()
        .map(|dataset| dataset.manifest.total_records)
        .sum::<usize>();
    let mut records = Vec::with_capacity(total_manifest_records);
    for dataset in &datasets {
        for shard in &dataset.manifest.shards {
            records.extend(
                PositionShardReader::open(&dataset.root, shard)?
                    .collect::<std::result::Result<Vec<_>, _>>()?,
            );
        }
    }
    if records.len() != total_manifest_records {
        return Err(R2Error::DatasetContract(format!(
            "loaded {} rows but manifests declare {total_manifest_records}",
            records.len()
        )));
    }
    let rows = records
        .par_iter()
        .enumerate()
        .map(|(ordinal, record)| {
            process_record(record, supplied_tile).map_err(|error| {
                R2Error::DatasetContract(format!("record {ordinal} failed: {error}"))
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut position_counts = Counts::default();
    let mut board_counts = Counts::default();
    let mut packed_bytes_values = Vec::new();
    let mut packed_fraction_values = Vec::new();
    let mut dense_occupied_ratios = DENSE_CAPACITIES.map(|_| Vec::new());
    let mut dense_total_ratios = DENSE_CAPACITIES.map(|_| Vec::new());
    let mut public_hasher = blake3::Hasher::new();
    let mut packed_hasher = blake3::Hasher::new();
    let mut record_count = 0usize;
    let mut board_count = 0usize;
    let mut d6_checks = 0u64;
    let mut target_independence_checks = 0u64;
    let mut frontier_positions_with_habitat_bridge = 0u64;
    let mut frontier_positions_with_repeated_component_contact = 0u64;

    for row in rows {
        update_framed_hash(&mut public_hasher, &row.public_bytes);
        update_framed_hash(&mut packed_hasher, &row.packed);
        position_counts.push(row.position_values);
        packed_bytes_values.push(row.packed.len() as u64);
        packed_fraction_values.push((row.packed.len() as u64 * RATIO_SCALE) / RECORD_SIZE as u64);
        for (index, capacity) in DENSE_CAPACITIES.iter().copied().enumerate() {
            let dense_rows = capacity * u64::from(row.player_count);
            dense_occupied_ratios[index]
                .push(row.position_values[0] as u64 * RATIO_SCALE / dense_rows);
            dense_total_ratios[index]
                .push(row.position_values[4] as u64 * RATIO_SCALE / dense_rows);
        }
        for values in row.board_values {
            board_counts.push(values);
            board_count += 1;
        }
        d6_checks += row.d6_checks;
        target_independence_checks += row.target_independence_checks;
        frontier_positions_with_habitat_bridge += row.frontier_positions_with_habitat_bridge;
        frontier_positions_with_repeated_component_contact +=
            row.frontier_positions_with_repeated_component_contact;
        record_count += 1;
    }
    if record_count == 0 {
        return Err(R2Error::DatasetContract(
            "validated inputs contain no PositionRecord rows".to_owned(),
        ));
    }
    if accepted_r0_contract_matched && record_count != ACCEPTED_R0_CORPUS_ROWS {
        return Err(R2Error::DatasetContract(format!(
            "accepted R0 identities yielded {record_count} rows instead of {ACCEPTED_R0_CORPUS_ROWS}"
        )));
    }

    let position_tokens = position_counts.summarize()?;
    let board_tokens = board_counts.summarize()?;
    let packed_bytes = DistributionSummary::from_values(packed_bytes_values)?;
    let packed_bytes_fraction_of_compact_entity_v2_ppm =
        DistributionSummary::from_values(packed_fraction_values)?;
    let mut dense_capacity_comparisons = Vec::new();
    for (index, capacity) in DENSE_CAPACITIES.iter().copied().enumerate() {
        dense_capacity_comparisons.push(DenseCapacityComparison {
            cells_per_board: capacity,
            ratio_unit: "parts-per-million; 1_000_000 equals the dense row capacity".to_owned(),
            occupied_token_fraction_ppm: DistributionSummary::from_values(std::mem::take(
                &mut dense_occupied_ratios[index],
            ))?,
            total_spatial_token_fraction_ppm: DistributionSummary::from_values(std::mem::take(
                &mut dense_total_ratios[index],
            ))?,
        });
    }

    let promotion_assessment = R2PromotionAssessment {
        exact_public_round_trip: true,
        exact_pack_round_trip: true,
        frontier_oracle_equality: true,
        habitat_oracle_equality: true,
        exact_d6_inverse: true,
        target_independence: target_independence_checks == record_count as u64,
        p99_total_tokens_within_512: position_tokens.total_spatial_tokens.p99
            <= POSITION_TOKEN_P99_BUDGET,
        max_total_tokens_within_640: position_tokens.total_spatial_tokens.max
            <= POSITION_TOKEN_MAX_BUDGET,
        p99_packed_bytes_within_position_record: packed_bytes.p99 <= PACKED_P99_BUDGET,
        authorize_matched_mlx_prototype: false,
        learned_quality_claim: false,
        gameplay_claim: false,
    };
    let mut promotion_assessment = promotion_assessment;
    promotion_assessment.authorize_matched_mlx_prototype = promotion_assessment
        .exact_public_round_trip
        && promotion_assessment.exact_pack_round_trip
        && promotion_assessment.frontier_oracle_equality
        && promotion_assessment.habitat_oracle_equality
        && promotion_assessment.exact_d6_inverse
        && promotion_assessment.target_independence
        && promotion_assessment.p99_total_tokens_within_512
        && promotion_assessment.max_total_tokens_within_640
        && promotion_assessment.p99_packed_bytes_within_position_record;

    let scientific = ScientificCensus {
        experiment_id: "r2-sparse-occupied-frontier-foundation-v1".to_owned(),
        schema: "r2-sparse-public-token-state-v1".to_owned(),
        schema_version: 1,
        packed_schema: "CSR2SP1".to_owned(),
        feature_schema: FEATURE_SCHEMA.to_owned(),
        corpus_order:
            "CLI dataset-root order, manifest shard order, in-shard record order".to_owned(),
        accepted_r0_contract_required: requirement
            == CorpusRequirement::AcceptedR0SixtyThousand,
        accepted_r0_contract_matched,
        datasets: datasets
            .iter()
            .map(|dataset| dataset.identity.clone())
            .collect(),
        record_count,
        board_count,
        supplied_tile,
        targets_serialized_or_hashed: false,
        public_position_blake3: public_hasher.finalize().to_hex().to_string(),
        packed_state_blake3: packed_hasher.finalize().to_hex().to_string(),
        d6_transform_inverse_checks: d6_checks,
        target_independence_checks,
        position_tokens,
        board_tokens,
        packed_bytes,
        packed_bytes_fraction_of_compact_entity_v2_ppm,
        dense_capacity_comparisons,
        frontier_positions_with_habitat_bridge,
        frontier_positions_with_repeated_component_contact,
        promotion_criteria: promotion_criteria(),
        promotion_assessment,
        limitations: vec![
            "Wildlife motif tokens are exact per-wildlife anchors with local adjacency; they are not a complete Card A scoring quotient.".to_owned(),
            "Habitat bridge bits identify exact local component merges; they do not predict future placement sequences or global strategic value.".to_owned(),
            "Terrain-compatible rotations mean at least one directed habitat edge matches. Cascadia rules permit every canonical rotation.".to_owned(),
            "The census establishes representation mechanics and serving-shape feasibility only. It makes no learned-quality, score, search, or gameplay claim.".to_owned(),
            "Hidden stack order, hidden wildlife order, future refills, future actions, and terminal targets are absent.".to_owned(),
        ],
    };
    let scientific_bytes = serde_json::to_vec(&scientific)?;
    let scientific_blake3 = blake3::hash(&scientific_bytes).to_hex().to_string();
    Ok(CensusReport {
        scientific,
        scientific_blake3,
    })
}

fn process_record(
    record: &PositionRecord,
    supplied_tile: Option<SuppliedTile>,
) -> Result<RowCensus> {
    let state = SparsePublicState::from_position_record(record, supplied_tile)?;
    if state.reconstruct_position_record(record.targets)? != *record {
        return Err(R2Error::DatasetContract(
            "public reconstruction changed the source record".to_owned(),
        ));
    }
    let packed = state.to_packed_bytes()?;
    let decoded = SparsePublicState::from_packed_bytes(&packed)?;
    if decoded != state {
        return Err(R2Error::DatasetContract(
            "packed round trip changed the sparse state".to_owned(),
        ));
    }

    let mut d6_checks = 0;
    for transform in D6Transform::ALL {
        let transformed = state.transformed(transform)?;
        let restored = transformed.transformed(transform.inverse())?;
        if restored != state {
            return Err(R2Error::DatasetContract(format!(
                "D6 inverse changed the state under transform {}",
                transform.id()
            )));
        }
        d6_checks += 1;
    }

    let mut changed_targets = record.clone();
    for target in &mut changed_targets.targets {
        *target = !*target;
    }
    let changed = SparsePublicState::from_position_record(&changed_targets, supplied_tile)?;
    if changed != state || changed.to_packed_bytes()? != packed {
        return Err(R2Error::DatasetContract(
            "target mutation changed the public sparse state".to_owned(),
        ));
    }

    let position_values = [
        state.occupied_tiles.len(),
        state.legal_frontier.len(),
        state.habitat_components.len(),
        state.wildlife_motifs.len(),
        state.total_spatial_tokens(),
    ];
    let board_values = (0..state.global.player_count)
        .map(|relative_seat| state.board_token_counts(relative_seat))
        .collect();
    let public_bytes = state.reconstruct_position_record([0; 11])?.to_bytes();
    let frontier_positions_with_habitat_bridge = state
        .legal_frontier
        .iter()
        .filter(|frontier| frontier.habitat_bridge_terrain_bits != 0)
        .count() as u64;
    let frontier_positions_with_repeated_component_contact = state
        .legal_frontier
        .iter()
        .filter(|frontier| frontier.repeated_component_contact_terrain_bits != 0)
        .count() as u64;
    Ok(RowCensus {
        public_bytes,
        packed,
        player_count: state.global.player_count,
        position_values,
        board_values,
        d6_checks,
        target_independence_checks: 1,
        frontier_positions_with_habitat_bridge,
        frontier_positions_with_repeated_component_contact,
    })
}

pub fn read_record_at_ordinal(
    roots: &[PathBuf],
    ordinal: usize,
) -> Result<(DatasetIdentity, PositionRecord)> {
    let datasets = validate_inputs(roots)?;
    let total = datasets
        .iter()
        .map(|dataset| dataset.manifest.total_records)
        .sum::<usize>();
    if ordinal >= total {
        return Err(R2Error::OrdinalOutOfRange { ordinal, total });
    }
    let mut current = 0usize;
    for dataset in datasets {
        for shard in &dataset.manifest.shards {
            let reader = PositionShardReader::open(&dataset.root, shard)?;
            for record in reader {
                if current == ordinal {
                    return Ok((dataset.identity, record?));
                }
                let _ = record?;
                current += 1;
            }
        }
    }
    Err(R2Error::OrdinalOutOfRange { ordinal, total })
}

pub fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temp = path.with_extension("json.tmp");
    let mut writer = BufWriter::new(File::create(&temp)?);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(temp, path)?;
    Ok(())
}

fn validate_inputs(roots: &[PathBuf]) -> Result<Vec<ValidatedDataset>> {
    if roots.is_empty() {
        return Err(R2Error::DatasetContract(
            "at least one --dataset-root is required".to_owned(),
        ));
    }
    let mut datasets = Vec::with_capacity(roots.len());
    for root in roots {
        let manifest_path = root.join("dataset.json");
        let manifest: DatasetManifest =
            serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
        validate_dataset(root, &manifest)?;
        if manifest.feature_schema != FEATURE_SCHEMA {
            return Err(R2Error::DatasetContract(format!(
                "dataset {} uses feature schema {} instead of {}",
                manifest.dataset_id, manifest.feature_schema, FEATURE_SCHEMA
            )));
        }
        if manifest.record_size != RECORD_SIZE {
            return Err(R2Error::DatasetContract(format!(
                "dataset {} record size {} does not equal {RECORD_SIZE}",
                manifest.dataset_id, manifest.record_size
            )));
        }
        let identity = DatasetIdentity {
            dataset_id: manifest.dataset_id.clone(),
            split: split_name(manifest.split).to_owned(),
            feature_schema: manifest.feature_schema.clone(),
            total_records: manifest.total_records,
            manifest_blake3: hash_file(&manifest_path)?,
            shards: manifest
                .shards
                .iter()
                .map(|shard| ShardIdentity {
                    file_name: shard.file.clone(),
                    first_game_index: shard.first_game_index,
                    game_count: shard.game_count,
                    record_count: shard.record_count,
                    byte_count: shard.byte_count,
                    blake3: shard.blake3.clone(),
                })
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

fn matches_accepted_r0(datasets: &[ValidatedDataset]) -> bool {
    datasets.len() == ACCEPTED_R0_DATASETS.len()
        && datasets
            .iter()
            .zip(ACCEPTED_R0_DATASETS)
            .all(|(actual, expected)| {
                actual.identity.dataset_id == expected.dataset_id
                    && actual.identity.split == expected.split
                    && actual.identity.total_records == expected.rows
                    && actual.identity.manifest_blake3 == expected.manifest_blake3
            })
}

fn promotion_criteria() -> Vec<PromotionCriterion> {
    vec![
        PromotionCriterion {
            id: "mechanical-exactness".to_owned(),
            threshold: "100% public reconstruction, packed round-trip, frontier oracle, habitat oracle, and D6 inverse checks".to_owned(),
            rationale: "Any semantic loss blocks learned work.".to_owned(),
        },
        PromotionCriterion {
            id: "target-independence".to_owned(),
            threshold: "Terminal target mutation changes neither tokens nor packed bytes".to_owned(),
            rationale: "The representation must contain public inputs only.".to_owned(),
        },
        PromotionCriterion {
            id: "p99-token-budget".to_owned(),
            threshold: format!("P99 total spatial tokens <= {POSITION_TOKEN_P99_BUDGET}"),
            rationale: "Keeps padded Set Transformer and graph batches within the frozen first-pass serving envelope.".to_owned(),
        },
        PromotionCriterion {
            id: "maximum-token-budget".to_owned(),
            threshold: format!("Maximum total spatial tokens <= {POSITION_TOKEN_MAX_BUDGET}"),
            rationale: "Prevents rare legal states from requiring a silent truncation path.".to_owned(),
        },
        PromotionCriterion {
            id: "packed-byte-budget".to_owned(),
            threshold: format!("P99 packed bytes <= compact-entity-v2 record size ({PACKED_P99_BUDGET})"),
            rationale: "The exact sparse substrate should not increase the normal serialized state footprint.".to_owned(),
        },
        PromotionCriterion {
            id: "claim-boundary".to_owned(),
            threshold: "Passing authorizes only a matched MLX prototype".to_owned(),
            rationale: "Offline quality, throughput, regret, and gameplay remain separate preregistered experiments.".to_owned(),
        },
    ]
}

fn nearest_rank(sorted: &[u64], percentile: usize) -> u64 {
    let rank = (percentile * sorted.len()).div_ceil(100).max(1);
    sorted[rank - 1]
}

fn update_framed_hash(hasher: &mut blake3::Hasher, bytes: &[u8]) {
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

const fn split_name(split: DatasetSplit) -> &'static str {
    match split {
        DatasetSplit::Train => "train",
        DatasetSplit::Validation => "validation",
        DatasetSplit::Test => "test",
        DatasetSplit::Final => "final",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_definition_is_stable() {
        let values = (1..=100).collect::<Vec<_>>();
        assert_eq!(nearest_rank(&values, 50), 50);
        assert_eq!(nearest_rank(&values, 90), 90);
        assert_eq!(nearest_rank(&values, 99), 99);
    }

    #[test]
    fn promotion_criteria_are_frozen_and_non_gameplay() {
        let criteria = promotion_criteria();
        assert_eq!(criteria.len(), 6);
        assert!(
            criteria
                .iter()
                .any(|criterion| criterion.id == "claim-boundary")
        );
    }
}
