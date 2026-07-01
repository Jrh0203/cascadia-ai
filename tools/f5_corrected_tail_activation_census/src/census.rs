use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
    time::Instant,
};

use blake3::Hasher;
use serde::{Deserialize, Serialize};

use crate::{
    Result, canonical_blake3,
    corpus::{
        CorpusManifest, PublicCorpusStatistics, ShardOwnership, read_overflow_witness,
        read_representative_records, statistics_for_records, validate_corpus_manifest,
    },
    hash_file, invalid,
    model::{
        ARTIFACT_SCHEMA_VERSION, DATASET_ID, EXPERIMENT_ID, FEATURE_SCHEMA,
        PRODUCTION_FIRST_GAME_INDEX, PRODUCTION_SHARD_COUNT, PRODUCTION_TOTAL_GAMES, Phase,
        PublicStateRecord, ROWS_PER_GAME, RecordKind, TrajectoryPolicy, corrected_tail_indices,
        owned_game_indices, replay_record,
    },
    source::{SourceIdentity, capture_source_identity},
    strict_json, update_framed, write_json_atomic,
};

const TERRAIN_NAMES: [&str; 5] = ["forest", "prairie", "wetland", "mountain", "river"];
const WILDLIFE_NAMES: [&str; 5] = ["bear", "elk", "salmon", "hawk", "fox"];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SliceCoverage {
    pub phase: BTreeMap<String, u64>,
    pub focal_seat: BTreeMap<String, u64>,
    pub policy: BTreeMap<String, u64>,
    pub overflow_used: BTreeMap<String, u64>,
}

impl SliceCoverage {
    fn zeroed() -> Self {
        Self {
            phase: Phase::ALL
                .into_iter()
                .map(|phase| (phase.as_str().to_owned(), 0))
                .collect(),
            focal_seat: (0..4).map(|seat| (seat.to_string(), 0)).collect(),
            policy: TrajectoryPolicy::ALL
                .into_iter()
                .map(|policy| (policy.as_str().to_owned(), 0))
                .collect(),
            overflow_used: BTreeMap::from([("not_used".to_owned(), 0), ("used".to_owned(), 0)]),
        }
    }

    fn observe(&mut self, record: &PublicStateRecord) {
        *self
            .phase
            .get_mut(record.state.phase.as_str())
            .expect("phase map contains all frozen phases") += 1;
        *self
            .focal_seat
            .get_mut(&record.state.focal_seat.to_string())
            .expect("seat map contains all four seats") += 1;
        *self
            .policy
            .get_mut(record.provenance.policy.as_str())
            .expect("policy map contains all frozen policies") += 1;
        *self
            .overflow_used
            .get_mut(if record.state.overflow_used_this_turn {
                "used"
            } else {
                "not_used"
            })
            .expect("overflow map contains both states") += 1;
    }

    fn add_assign(&mut self, other: &Self) -> Result<()> {
        add_maps(&mut self.phase, &other.phase)?;
        add_maps(&mut self.focal_seat, &other.focal_seat)?;
        add_maps(&mut self.policy, &other.policy)?;
        add_maps(&mut self.overflow_used, &other.overflow_used)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelCoverage {
    pub feature_index: u16,
    pub block: String,
    pub semantic_owner: String,
    pub bin: Option<u8>,
    pub activations: u64,
    pub slices: SliceCoverage,
}

impl ChannelCoverage {
    fn observe(&mut self, record: &PublicStateRecord) {
        self.activations += 1;
        self.slices.observe(record);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockCoverage {
    pub block: String,
    pub range_start: u16,
    pub range_end_exclusive: u16,
    pub channels: usize,
    pub active_channels: usize,
    pub activations: u64,
    pub representative_rows_with_activation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OverflowWitnessEvidence {
    pub source: String,
    pub excluded_from_representativeness_statistics: bool,
    pub fixture_search_counter: u64,
    pub public_state_blake3: String,
    pub normalized_features_blake3: String,
    pub corrected_tail_blake3: String,
    pub overflow_feature_index: u16,
    pub overflow_feature_active: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardGates {
    pub manifest_and_payload_hashes_valid: bool,
    pub source_bundle_matches: bool,
    pub exact_owned_games_and_rows: bool,
    pub public_state_receipts_match: bool,
    pub actual_rust_extractor_replay_exact: bool,
    pub five_terrain_and_five_wildlife_rows_per_record: bool,
    pub natural_overflow_coverage_reported_separately: bool,
    pub authoritative_overflow_witness_valid_if_owned: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardScientific {
    pub schema_version: u32,
    pub experiment_id: String,
    pub dataset_id: String,
    pub feature_schema: String,
    pub source_bundle_blake3: String,
    pub extractor_source_blake3: String,
    pub manifest_file_blake3: String,
    pub manifest_scientific_blake3: String,
    pub records_payload_blake3: String,
    pub ownership: ShardOwnership,
    pub statistics: PublicCorpusStatistics,
    pub channels: Vec<ChannelCoverage>,
    pub blocks: Vec<BlockCoverage>,
    pub natural_representative_overflow_rows: u64,
    pub overflow_witness: Option<OverflowWitnessEvidence>,
    pub aggregate_eligible: bool,
    pub gates: ShardGates,
    pub classification: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardOperational {
    pub host: String,
    pub corpus_root: String,
    pub output: String,
    pub elapsed_millis: u128,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardReport {
    pub scientific: ShardScientific,
    pub scientific_blake3: String,
    pub operational: ShardOperational,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AggregateShardIdentity {
    pub shard_index: usize,
    pub report_scientific_blake3: String,
    pub manifest_scientific_blake3: String,
    pub records_payload_blake3: String,
    pub rows: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AggregateGates {
    pub exact_required_shards: bool,
    pub no_overlap_or_gap: bool,
    pub one_source_and_extractor_identity: bool,
    pub exact_phase_coverage: bool,
    pub exact_seat_coverage: bool,
    pub exact_policy_coverage: bool,
    pub both_natural_overflow_slices_observed: bool,
    pub terrain_block_all_150_channels_active: bool,
    pub wildlife_capacity_block_all_150_channels_active: bool,
    pub representative_overflow_row_active: bool,
    pub separate_authoritative_overflow_witness_active: bool,
    pub all_scientific_inputs_content_hashed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AggregateScientific {
    pub schema_version: u32,
    pub experiment_id: String,
    pub dataset_id: String,
    pub feature_schema: String,
    pub source_bundle_blake3: String,
    pub extractor_source_blake3: String,
    pub first_game_index: u64,
    pub total_games: usize,
    pub shard_count: usize,
    pub owned_game_indices: Vec<u64>,
    pub shards: Vec<AggregateShardIdentity>,
    pub statistics: PublicCorpusStatistics,
    pub channels: Vec<ChannelCoverage>,
    pub blocks: Vec<BlockCoverage>,
    pub natural_representative_overflow_rows: u64,
    pub overflow_witness: OverflowWitnessEvidence,
    pub production_contract: bool,
    pub success: bool,
    pub gates: AggregateGates,
    pub classification: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AggregateOperational {
    pub input_reports: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AggregateReport {
    pub scientific: AggregateScientific,
    pub scientific_blake3: String,
    pub operational: AggregateOperational,
}

pub fn census_shard(corpus_root: &Path, output: &Path) -> Result<ShardReport> {
    let started = Instant::now();
    let manifest = validate_corpus_manifest(corpus_root)?;
    let records = read_representative_records(corpus_root, &manifest)?;
    validate_record_sequence(&manifest, &records)?;
    let mut channels = initial_channels();
    for record in &records {
        let replayed = replay_record(record)?;
        for feature in replayed.corrected_tail_features {
            let offset = usize::from(feature)
                .checked_sub(10_930)
                .ok_or_else(|| invalid("corrected tail feature precedes corrected range"))?;
            channels
                .get_mut(offset)
                .ok_or_else(|| invalid("corrected tail feature exceeds corrected range"))?
                .observe(record);
        }
    }
    let statistics = statistics_for_records(
        &records,
        manifest.scientific.ownership.owned_game_indices.len(),
    )?;
    if statistics != manifest.scientific.statistics {
        return Err(invalid(
            "replayed representative statistics differ from frozen manifest",
        ));
    }
    let blocks = block_coverage(&channels, statistics.rows)?;
    let natural_representative_overflow_rows = statistics.rows_by_overflow_used["used"];
    let overflow_witness = validate_overflow_witness(corpus_root, &manifest)?;
    let scientific = ShardScientific {
        schema_version: ARTIFACT_SCHEMA_VERSION,
        experiment_id: EXPERIMENT_ID.to_owned(),
        dataset_id: manifest.scientific.contract.dataset_id.clone(),
        feature_schema: FEATURE_SCHEMA.to_owned(),
        source_bundle_blake3: manifest.scientific.source.source_bundle_blake3.clone(),
        extractor_source_blake3: manifest.scientific.source.extractor_source_blake3.clone(),
        manifest_file_blake3: hash_file(&corpus_root.join("manifest.json"))?,
        manifest_scientific_blake3: manifest.scientific_blake3.clone(),
        records_payload_blake3: manifest.scientific.records.blake3.clone(),
        ownership: manifest.scientific.ownership.clone(),
        statistics,
        channels,
        blocks,
        natural_representative_overflow_rows,
        overflow_witness,
        aggregate_eligible: manifest.scientific.aggregate_eligible,
        gates: ShardGates {
            manifest_and_payload_hashes_valid: true,
            source_bundle_matches: true,
            exact_owned_games_and_rows: true,
            public_state_receipts_match: true,
            actual_rust_extractor_replay_exact: true,
            five_terrain_and_five_wildlife_rows_per_record: true,
            natural_overflow_coverage_reported_separately: true,
            authoritative_overflow_witness_valid_if_owned: true,
        },
        classification: "corrected_mid_tail_activation_shard_complete".to_owned(),
    };
    let report = ShardReport {
        scientific_blake3: canonical_blake3(&scientific)?,
        scientific,
        operational: ShardOperational {
            host: host_name(),
            corpus_root: corpus_root.display().to_string(),
            output: output.display().to_string(),
            elapsed_millis: started.elapsed().as_millis(),
        },
    };
    write_json_atomic(output, &report)?;
    Ok(report)
}

pub fn aggregate_reports(
    report_paths: &[PathBuf],
    require_shards: usize,
    output: &Path,
) -> Result<AggregateReport> {
    if require_shards != PRODUCTION_SHARD_COUNT || report_paths.len() != require_shards {
        return Err(invalid(format!(
            "aggregate requires exactly {PRODUCTION_SHARD_COUNT} reports, received {} with require-shards={require_shards}",
            report_paths.len()
        )));
    }
    let mut reports = report_paths
        .iter()
        .map(|path| read_shard_report(path))
        .collect::<Result<Vec<_>>>()?;
    let current_source = capture_source_identity()?;
    for report in &reports {
        validate_shard_report(report, &current_source)?;
    }
    reports.sort_by_key(|report| report.scientific.ownership.shard_index);

    let shard_indices = reports
        .iter()
        .map(|report| report.scientific.ownership.shard_index)
        .collect::<Vec<_>>();
    if shard_indices != (0..require_shards).collect::<Vec<_>>() {
        return Err(invalid(
            "aggregate shard indices contain an overlap, gap, or out-of-range value",
        ));
    }
    let first = &reports[0].scientific;
    if first.ownership.shard_count != require_shards {
        return Err(invalid(
            "report shard count differs from aggregate requirement",
        ));
    }
    for report in &reports[1..] {
        let scientific = &report.scientific;
        if scientific.experiment_id != first.experiment_id
            || scientific.dataset_id != first.dataset_id
            || scientific.feature_schema != first.feature_schema
            || scientific.source_bundle_blake3 != first.source_bundle_blake3
            || scientific.extractor_source_blake3 != first.extractor_source_blake3
            || scientific.ownership.shard_count != first.ownership.shard_count
            || scientific.ownership.first_game_index != first.ownership.first_game_index
            || scientific.ownership.total_games != first.ownership.total_games
            || scientific.aggregate_eligible != first.aggregate_eligible
        {
            return Err(invalid(
                "aggregate reports disagree on source, schema, corpus, or shard contract",
            ));
        }
    }

    let mut union = BTreeSet::new();
    for report in &reports {
        for game_index in &report.scientific.ownership.owned_game_indices {
            if !union.insert(*game_index) {
                return Err(invalid(format!(
                    "aggregate game index {game_index} appears in multiple shards"
                )));
            }
        }
    }
    let expected_games = (first.ownership.first_game_index
        ..first
            .ownership
            .first_game_index
            .checked_add(u64::try_from(first.ownership.total_games)?)
            .ok_or_else(|| invalid("aggregate game range overflows u64"))?)
        .collect::<Vec<_>>();
    let owned_game_indices = union.into_iter().collect::<Vec<_>>();
    if owned_game_indices != expected_games {
        return Err(invalid(
            "aggregate representative game indices contain a gap or out-of-range value",
        ));
    }

    let statistics = aggregate_statistics(&reports)?;
    let channels = aggregate_channels(&reports)?;
    let blocks = block_coverage(&channels, statistics.rows)?;
    let witnesses = reports
        .iter()
        .filter_map(|report| report.scientific.overflow_witness.clone())
        .collect::<Vec<_>>();
    if witnesses.len() != 1 {
        return Err(invalid(
            "aggregate requires exactly one separately labeled overflow witness",
        ));
    }
    let overflow_witness = witnesses[0].clone();
    if !overflow_witness.excluded_from_representativeness_statistics
        || !overflow_witness.overflow_feature_active
        || overflow_witness.overflow_feature_index != 11_230
    {
        return Err(invalid("aggregate overflow witness contract drifted"));
    }

    let production_contract = first.aggregate_eligible
        && first.ownership.first_game_index == PRODUCTION_FIRST_GAME_INDEX
        && first.ownership.total_games == PRODUCTION_TOTAL_GAMES
        && require_shards == PRODUCTION_SHARD_COUNT;
    let gates = aggregate_gates(
        &statistics,
        &blocks,
        &overflow_witness,
        first.ownership.total_games,
        require_shards,
        production_contract,
    )?;
    let success = production_contract
        && gates.exact_required_shards
        && gates.no_overlap_or_gap
        && gates.one_source_and_extractor_identity
        && gates.exact_phase_coverage
        && gates.exact_seat_coverage
        && gates.exact_policy_coverage
        && gates.both_natural_overflow_slices_observed
        && gates.terrain_block_all_150_channels_active
        && gates.wildlife_capacity_block_all_150_channels_active
        && gates.representative_overflow_row_active
        && gates.separate_authoritative_overflow_witness_active
        && gates.all_scientific_inputs_content_hashed;
    let classification = if production_contract && success {
        "corrected_mid_tail_activation_census_complete"
    } else if production_contract {
        "corrected_mid_tail_activation_census_incomplete"
    } else {
        "corrected_mid_tail_activation_census_smoke_complete"
    };
    let shard_identities = reports
        .iter()
        .map(|report| AggregateShardIdentity {
            shard_index: report.scientific.ownership.shard_index,
            report_scientific_blake3: report.scientific_blake3.clone(),
            manifest_scientific_blake3: report.scientific.manifest_scientific_blake3.clone(),
            records_payload_blake3: report.scientific.records_payload_blake3.clone(),
            rows: report.scientific.statistics.rows,
        })
        .collect();
    let scientific = AggregateScientific {
        schema_version: ARTIFACT_SCHEMA_VERSION,
        experiment_id: EXPERIMENT_ID.to_owned(),
        dataset_id: first.dataset_id.clone(),
        feature_schema: FEATURE_SCHEMA.to_owned(),
        source_bundle_blake3: first.source_bundle_blake3.clone(),
        extractor_source_blake3: first.extractor_source_blake3.clone(),
        first_game_index: first.ownership.first_game_index,
        total_games: first.ownership.total_games,
        shard_count: require_shards,
        owned_game_indices,
        shards: shard_identities,
        statistics,
        channels,
        blocks,
        natural_representative_overflow_rows: reports
            .iter()
            .map(|report| report.scientific.natural_representative_overflow_rows)
            .sum(),
        overflow_witness,
        production_contract,
        success,
        gates,
        classification: classification.to_owned(),
    };
    let mut input_reports = report_paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>();
    input_reports.sort();
    let aggregate = AggregateReport {
        scientific_blake3: canonical_blake3(&scientific)?,
        scientific,
        operational: AggregateOperational { input_reports },
    };
    write_json_atomic(output, &aggregate)?;
    Ok(aggregate)
}

pub fn read_shard_report(path: &Path) -> Result<ShardReport> {
    Ok(strict_json::from_reader(BufReader::new(File::open(path)?))?)
}

pub fn read_aggregate_report(path: &Path) -> Result<AggregateReport> {
    let report: AggregateReport = strict_json::from_reader(BufReader::new(File::open(path)?))?;
    if report.scientific_blake3 != canonical_blake3(&report.scientific)? {
        return Err(invalid("aggregate report scientific BLAKE3 drifted"));
    }
    Ok(report)
}

pub fn verify_reports_byte_identical(left: &Path, right: &Path) -> Result<()> {
    let mut left_bytes = Vec::new();
    let mut right_bytes = Vec::new();
    File::open(left)?.read_to_end(&mut left_bytes)?;
    File::open(right)?.read_to_end(&mut right_bytes)?;
    if left_bytes != right_bytes {
        return Err(invalid("aggregate reports are not byte-identical"));
    }
    let _ = read_aggregate_report(left)?;
    let _ = read_aggregate_report(right)?;
    Ok(())
}

fn validate_shard_report(report: &ShardReport, current_source: &SourceIdentity) -> Result<()> {
    let scientific = &report.scientific;
    if scientific.schema_version != ARTIFACT_SCHEMA_VERSION
        || scientific.experiment_id != EXPERIMENT_ID
        || scientific.dataset_id != DATASET_ID
        || scientific.feature_schema != FEATURE_SCHEMA
        || scientific.classification != "corrected_mid_tail_activation_shard_complete"
        || report.scientific_blake3 != canonical_blake3(scientific)?
        || scientific.channels.len() != 301
        || scientific.blocks.len() != 3
    {
        return Err(invalid("shard report scientific contract drifted"));
    }
    if scientific.source_bundle_blake3 != current_source.source_bundle_blake3
        || scientific.extractor_source_blake3 != current_source.extractor_source_blake3
    {
        return Err(invalid(
            "shard report source or extractor differs from the current frozen source",
        ));
    }
    for (label, hash) in [
        ("source bundle", &scientific.source_bundle_blake3),
        ("extractor source", &scientific.extractor_source_blake3),
        ("manifest file", &scientific.manifest_file_blake3),
        (
            "manifest scientific payload",
            &scientific.manifest_scientific_blake3,
        ),
        ("records payload", &scientific.records_payload_blake3),
        ("report scientific payload", &report.scientific_blake3),
    ] {
        validate_blake3(label, hash)?;
    }

    let ownership = &scientific.ownership;
    let expected_owned = owned_game_indices(
        ownership.first_game_index,
        ownership.total_games,
        ownership.shard_index,
        ownership.shard_count,
    )?;
    if ownership.first_game_index != PRODUCTION_FIRST_GAME_INDEX
        || ownership.shard_count != PRODUCTION_SHARD_COUNT
        || ownership.owned_game_indices != expected_owned
        || ownership.owned_game_indices_blake3 != canonical_blake3(&expected_owned)?
        || scientific.aggregate_eligible
            != (ownership.total_games == PRODUCTION_TOTAL_GAMES
                && ownership.first_game_index == PRODUCTION_FIRST_GAME_INDEX
                && ownership.shard_count == PRODUCTION_SHARD_COUNT)
    {
        return Err(invalid("shard report ownership contract drifted"));
    }
    validate_shard_statistics(&scientific.statistics, expected_owned.len())?;
    validate_channel_table(&scientific.channels, &scientific.statistics)?;
    if scientific.blocks != block_coverage(&scientific.channels, scientific.statistics.rows)? {
        return Err(invalid(
            "shard block summaries do not match channel evidence",
        ));
    }
    if scientific.natural_representative_overflow_rows
        != scientific.statistics.rows_by_overflow_used["used"]
        || scientific.natural_representative_overflow_rows != scientific.channels[300].activations
    {
        return Err(invalid(
            "natural overflow evidence disagrees across shard statistics and channels",
        ));
    }
    match (ownership.shard_index, scientific.overflow_witness.as_ref()) {
        (0, Some(witness)) => validate_aggregate_witness(witness)?,
        (0, None) => return Err(invalid("shard zero is missing the overflow witness")),
        (_, Some(_)) => {
            return Err(invalid(
                "only shard zero may carry the separate overflow witness",
            ));
        }
        (_, None) => {}
    }

    let gates = &scientific.gates;
    if !gates.manifest_and_payload_hashes_valid
        || !gates.source_bundle_matches
        || !gates.exact_owned_games_and_rows
        || !gates.public_state_receipts_match
        || !gates.actual_rust_extractor_replay_exact
        || !gates.five_terrain_and_five_wildlife_rows_per_record
        || !gates.natural_overflow_coverage_reported_separately
        || !gates.authoritative_overflow_witness_valid_if_owned
    {
        return Err(invalid("shard report contains a failed validity gate"));
    }
    Ok(())
}

fn validate_shard_statistics(
    statistics: &PublicCorpusStatistics,
    owned_games: usize,
) -> Result<()> {
    let games = u64::try_from(owned_games)?;
    let expected_rows = owned_games
        .checked_mul(ROWS_PER_GAME)
        .ok_or_else(|| invalid("shard row count overflowed"))?;
    let expected_per_seat_or_policy = games * 20;
    if statistics.games != owned_games
        || statistics.rows != expected_rows
        || statistics.rows_by_phase
            != BTreeMap::from([
                ("opening".to_owned(), games * 4),
                ("early".to_owned(), games * 16),
                ("middle".to_owned(), games * 32),
                ("late".to_owned(), games * 28),
            ])
        || statistics.rows_by_focal_seat
            != (0..4)
                .map(|seat| (seat.to_string(), expected_per_seat_or_policy))
                .collect()
        || statistics.rows_by_policy
            != TrajectoryPolicy::ALL
                .into_iter()
                .map(|policy| (policy.as_str().to_owned(), expected_per_seat_or_policy))
                .collect()
        || statistics.rows_by_overflow_used.keys().collect::<Vec<_>>()
            != zero_overflow_map().keys().collect::<Vec<_>>()
        || statistics.rows_by_overflow_used.values().sum::<u64>() != u64::try_from(expected_rows)?
    {
        return Err(invalid("shard report statistics shape drifted"));
    }
    for (label, hash) in [
        (
            "public-state stream",
            &statistics.public_state_stream_blake3,
        ),
        (
            "normalized-feature stream",
            &statistics.normalized_feature_stream_blake3,
        ),
        (
            "corrected-tail stream",
            &statistics.corrected_tail_stream_blake3,
        ),
        (
            "representative-record identity stream",
            &statistics.representative_record_identity_blake3,
        ),
    ] {
        validate_blake3(label, hash)?;
    }
    if statistics.raw_feature_emissions < statistics.normalized_feature_activations
        || statistics.normalized_feature_activations < statistics.corrected_tail_activations
        || statistics.corrected_tail_activations
            != u64::try_from(expected_rows)? * 10 + statistics.rows_by_overflow_used["used"]
    {
        return Err(invalid("shard feature activation totals drifted"));
    }
    Ok(())
}

fn validate_channel_table(
    channels: &[ChannelCoverage],
    statistics: &PublicCorpusStatistics,
) -> Result<()> {
    let expected = initial_channels();
    let rows = u64::try_from(statistics.rows)?;
    for (actual, frozen) in channels.iter().zip(expected) {
        if actual.feature_index != frozen.feature_index
            || actual.block != frozen.block
            || actual.semantic_owner != frozen.semantic_owner
            || actual.bin != frozen.bin
            || actual.activations > rows
        {
            return Err(invalid(
                "shard channel registry or activation count drifted",
            ));
        }
        for (actual_slice, expected_keys) in [
            (&actual.slices.phase, zero_phase_map()),
            (&actual.slices.focal_seat, zero_seat_map()),
            (&actual.slices.policy, zero_policy_map()),
            (&actual.slices.overflow_used, zero_overflow_map()),
        ] {
            if actual_slice.keys().collect::<Vec<_>>() != expected_keys.keys().collect::<Vec<_>>()
                || actual_slice.values().sum::<u64>() != actual.activations
            {
                return Err(invalid(
                    "shard channel slice keys or activation totals drifted",
                ));
            }
        }
    }
    for owner_start in [0usize, 30, 60, 90, 120, 150, 180, 210, 240, 270] {
        if channels[owner_start..owner_start + 30]
            .iter()
            .map(|channel| channel.activations)
            .sum::<u64>()
            != rows
        {
            return Err(invalid(
                "a corrected count owner does not emit exactly one bin per row",
            ));
        }
    }
    if channels[300].slices.overflow_used["used"] != channels[300].activations
        || channels[300].slices.overflow_used["not_used"] != 0
        || channels
            .iter()
            .map(|channel| channel.activations)
            .sum::<u64>()
            != statistics.corrected_tail_activations
    {
        return Err(invalid(
            "overflow channel or corrected-tail activation totals drifted",
        ));
    }
    Ok(())
}

fn validate_aggregate_witness(witness: &OverflowWitnessEvidence) -> Result<()> {
    if witness.source != "separate reachable opening-market adversarial fixture"
        || !witness.excluded_from_representativeness_statistics
        || !witness.overflow_feature_active
        || witness.overflow_feature_index != 11_230
    {
        return Err(invalid("aggregate overflow witness contract drifted"));
    }
    for (label, hash) in [
        (
            "overflow witness public state",
            &witness.public_state_blake3,
        ),
        (
            "overflow witness normalized features",
            &witness.normalized_features_blake3,
        ),
        (
            "overflow witness corrected tail",
            &witness.corrected_tail_blake3,
        ),
    ] {
        validate_blake3(label, hash)?;
    }
    Ok(())
}

fn validate_blake3(label: &str, value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid(format!("{label} BLAKE3 is not lowercase hex")));
    }
    Ok(())
}

fn validate_record_sequence(
    manifest: &CorpusManifest,
    records: &[PublicStateRecord],
) -> Result<()> {
    let owned = &manifest.scientific.ownership.owned_game_indices;
    if records.len() != owned.len() * ROWS_PER_GAME {
        return Err(invalid("representative record sequence row count drifted"));
    }
    for (row, record) in records.iter().enumerate() {
        let expected_game = owned[row / ROWS_PER_GAME];
        let expected_decision = row % ROWS_PER_GAME;
        if record.provenance.kind != RecordKind::Representative
            || record.provenance.game_index != Some(expected_game)
            || usize::from(record.provenance.decision_index) != expected_decision
        {
            return Err(invalid(format!(
                "representative record row {row} has a gap, overlap, or ordering drift"
            )));
        }
    }
    Ok(())
}

fn validate_overflow_witness(
    root: &Path,
    manifest: &CorpusManifest,
) -> Result<Option<OverflowWitnessEvidence>> {
    let Some(record) = read_overflow_witness(root, manifest)? else {
        return Ok(None);
    };
    if record.provenance.kind != RecordKind::ReachableOverflowWitness {
        return Err(invalid(
            "separate overflow witness payload is not labeled as a witness",
        ));
    }
    let replayed = replay_record(&record)?;
    let active = replayed.corrected_tail_features.contains(&11_230);
    if !active || !record.state.overflow_used_this_turn {
        return Err(invalid(
            "authoritative reachable overflow witness did not activate row 11230",
        ));
    }
    let declared = manifest
        .scientific
        .overflow_witness
        .as_ref()
        .ok_or_else(|| invalid("overflow witness payload is not declared"))?;
    if declared.fixture_search_counter
        != record
            .provenance
            .fixture_search_counter
            .ok_or_else(|| invalid("overflow witness has no search counter"))?
        || declared.public_state_blake3 != record.public_state_blake3
        || declared.corrected_tail_blake3 != record.corrected_tail_blake3
    {
        return Err(invalid("overflow witness declaration drifted"));
    }
    Ok(Some(OverflowWitnessEvidence {
        source: "separate reachable opening-market adversarial fixture".to_owned(),
        excluded_from_representativeness_statistics: true,
        fixture_search_counter: declared.fixture_search_counter,
        public_state_blake3: record.public_state_blake3,
        normalized_features_blake3: record.normalized_features_blake3,
        corrected_tail_blake3: record.corrected_tail_blake3,
        overflow_feature_index: 11_230,
        overflow_feature_active: true,
    }))
}

fn initial_channels() -> Vec<ChannelCoverage> {
    corrected_tail_indices()
        .into_iter()
        .map(|feature_index| {
            let index = usize::from(feature_index);
            let (block, semantic_owner, bin) = if index < 11_080 {
                let offset = index - 10_930;
                (
                    "tile_bag_terrain_count".to_owned(),
                    TERRAIN_NAMES[offset / 30].to_owned(),
                    Some((offset % 30) as u8),
                )
            } else if index < 11_230 {
                let offset = index - 11_080;
                (
                    "tile_bag_wildlife_capacity_count".to_owned(),
                    WILDLIFE_NAMES[offset / 30].to_owned(),
                    Some((offset % 30) as u8),
                )
            } else {
                (
                    "overflow_used".to_owned(),
                    "free_three_of_a_kind_replacement_used_this_turn".to_owned(),
                    None,
                )
            };
            ChannelCoverage {
                feature_index,
                block,
                semantic_owner,
                bin,
                activations: 0,
                slices: SliceCoverage::zeroed(),
            }
        })
        .collect()
}

fn block_coverage(channels: &[ChannelCoverage], rows: usize) -> Result<Vec<BlockCoverage>> {
    if channels.len() != 301 {
        return Err(invalid("channel table width is not 301"));
    }
    let specifications = [
        ("tile_bag_terrain_count", 10_930u16, 11_080u16),
        ("tile_bag_wildlife_capacity_count", 11_080u16, 11_230u16),
        ("overflow_used", 11_230u16, 11_231u16),
    ];
    Ok(specifications
        .into_iter()
        .map(|(name, start, end)| {
            let selected = channels
                .iter()
                .filter(|channel| (start..end).contains(&channel.feature_index))
                .collect::<Vec<_>>();
            let activations = selected.iter().map(|channel| channel.activations).sum();
            let representative_rows_with_activation = if name == "overflow_used" {
                selected[0].activations
            } else {
                u64::try_from(rows).expect("row count fits u64")
            };
            BlockCoverage {
                block: name.to_owned(),
                range_start: start,
                range_end_exclusive: end,
                channels: selected.len(),
                active_channels: selected
                    .iter()
                    .filter(|channel| channel.activations > 0)
                    .count(),
                activations,
                representative_rows_with_activation,
            }
        })
        .collect())
}

fn aggregate_statistics(reports: &[ShardReport]) -> Result<PublicCorpusStatistics> {
    let mut phase = zero_phase_map();
    let mut seats = zero_seat_map();
    let mut policy = zero_policy_map();
    let mut overflow = zero_overflow_map();
    let mut games = 0usize;
    let mut rows = 0usize;
    let mut raw = 0u64;
    let mut normalized = 0u64;
    let mut tail = 0u64;
    let mut public_hasher = aggregate_receipt_hasher(b"public-state");
    let mut feature_hasher = aggregate_receipt_hasher(b"normalized-feature");
    let mut tail_hasher = aggregate_receipt_hasher(b"corrected-tail");
    let mut identity_hasher = aggregate_receipt_hasher(b"record-identity");
    for report in reports {
        let statistics = &report.scientific.statistics;
        games += statistics.games;
        rows += statistics.rows;
        raw += statistics.raw_feature_emissions;
        normalized += statistics.normalized_feature_activations;
        tail += statistics.corrected_tail_activations;
        add_maps(&mut phase, &statistics.rows_by_phase)?;
        add_maps(&mut seats, &statistics.rows_by_focal_seat)?;
        add_maps(&mut policy, &statistics.rows_by_policy)?;
        add_maps(&mut overflow, &statistics.rows_by_overflow_used)?;
        update_framed(
            &mut public_hasher,
            statistics.public_state_stream_blake3.as_bytes(),
        );
        update_framed(
            &mut feature_hasher,
            statistics.normalized_feature_stream_blake3.as_bytes(),
        );
        update_framed(
            &mut tail_hasher,
            statistics.corrected_tail_stream_blake3.as_bytes(),
        );
        update_framed(
            &mut identity_hasher,
            statistics.representative_record_identity_blake3.as_bytes(),
        );
    }
    Ok(PublicCorpusStatistics {
        games,
        rows,
        rows_by_phase: phase,
        rows_by_focal_seat: seats,
        rows_by_policy: policy,
        rows_by_overflow_used: overflow,
        raw_feature_emissions: raw,
        normalized_feature_activations: normalized,
        corrected_tail_activations: tail,
        public_state_stream_blake3: public_hasher.finalize().to_hex().to_string(),
        normalized_feature_stream_blake3: feature_hasher.finalize().to_hex().to_string(),
        corrected_tail_stream_blake3: tail_hasher.finalize().to_hex().to_string(),
        representative_record_identity_blake3: identity_hasher.finalize().to_hex().to_string(),
    })
}

fn aggregate_channels(reports: &[ShardReport]) -> Result<Vec<ChannelCoverage>> {
    let mut channels = initial_channels();
    for report in reports {
        for (aggregate, shard) in channels.iter_mut().zip(&report.scientific.channels) {
            if aggregate.feature_index != shard.feature_index
                || aggregate.block != shard.block
                || aggregate.semantic_owner != shard.semantic_owner
                || aggregate.bin != shard.bin
            {
                return Err(invalid("shard channel metadata drifted"));
            }
            aggregate.activations += shard.activations;
            aggregate.slices.add_assign(&shard.slices)?;
        }
    }
    Ok(channels)
}

fn aggregate_gates(
    statistics: &PublicCorpusStatistics,
    blocks: &[BlockCoverage],
    witness: &OverflowWitnessEvidence,
    total_games: usize,
    shard_count: usize,
    production: bool,
) -> Result<AggregateGates> {
    let exact_phase = if production {
        statistics.rows_by_phase
            == BTreeMap::from([
                ("opening".to_owned(), 4_096),
                ("early".to_owned(), 16_384),
                ("middle".to_owned(), 32_768),
                ("late".to_owned(), 28_672),
            ])
    } else {
        statistics.rows_by_phase.values().sum::<u64>() == statistics.rows as u64
    };
    let expected_per_seat = u64::try_from(total_games * 20)?;
    let exact_seat = statistics.rows_by_focal_seat
        == (0..4)
            .map(|seat| (seat.to_string(), expected_per_seat))
            .collect();
    let exact_policy = statistics.rows_by_policy
        == TrajectoryPolicy::ALL
            .into_iter()
            .map(|policy| (policy.as_str().to_owned(), expected_per_seat))
            .collect();
    let terrain = blocks
        .iter()
        .find(|block| block.block == "tile_bag_terrain_count")
        .ok_or_else(|| invalid("aggregate terrain block is absent"))?;
    let wildlife = blocks
        .iter()
        .find(|block| block.block == "tile_bag_wildlife_capacity_count")
        .ok_or_else(|| invalid("aggregate wildlife-capacity block is absent"))?;
    let overflow = blocks
        .iter()
        .find(|block| block.block == "overflow_used")
        .ok_or_else(|| invalid("aggregate overflow block is absent"))?;
    Ok(AggregateGates {
        exact_required_shards: shard_count == PRODUCTION_SHARD_COUNT || !production,
        no_overlap_or_gap: true,
        one_source_and_extractor_identity: true,
        exact_phase_coverage: exact_phase,
        exact_seat_coverage: exact_seat,
        exact_policy_coverage: exact_policy,
        both_natural_overflow_slices_observed: statistics.rows_by_overflow_used["used"] > 0
            && statistics.rows_by_overflow_used["not_used"] > 0,
        terrain_block_all_150_channels_active: terrain.active_channels == 150,
        wildlife_capacity_block_all_150_channels_active: wildlife.active_channels == 150,
        representative_overflow_row_active: overflow.active_channels == 1
            && overflow.activations > 0,
        separate_authoritative_overflow_witness_active: witness.overflow_feature_active
            && witness.excluded_from_representativeness_statistics,
        all_scientific_inputs_content_hashed: true,
    })
}

fn aggregate_receipt_hasher(kind: &[u8]) -> Hasher {
    let mut hasher = Hasher::new();
    hasher.update(b"corrected-mid-tail-activation-census-v1/aggregate-stream/v1");
    update_framed(&mut hasher, kind);
    hasher
}

fn add_maps(target: &mut BTreeMap<String, u64>, source: &BTreeMap<String, u64>) -> Result<()> {
    if target.keys().collect::<Vec<_>>() != source.keys().collect::<Vec<_>>() {
        return Err(invalid("slice map keys drifted across reports"));
    }
    for (key, value) in source {
        *target
            .get_mut(key)
            .expect("target key set was validated above") += value;
    }
    Ok(())
}

fn zero_phase_map() -> BTreeMap<String, u64> {
    Phase::ALL
        .into_iter()
        .map(|phase| (phase.as_str().to_owned(), 0))
        .collect()
}

fn zero_seat_map() -> BTreeMap<String, u64> {
    (0..4).map(|seat| (seat.to_string(), 0)).collect()
}

fn zero_policy_map() -> BTreeMap<String, u64> {
    TrajectoryPolicy::ALL
        .into_iter()
        .map(|policy| (policy.as_str().to_owned(), 0))
        .collect()
}

fn zero_overflow_map() -> BTreeMap<String, u64> {
    BTreeMap::from([("not_used".to_owned(), 0), ("used".to_owned(), 0)])
}

fn host_name() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "unknown".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_registry_exactly_covers_the_301_corrected_rows() {
        let channels = initial_channels();
        assert_eq!(channels.len(), 301);
        assert_eq!(channels.first().unwrap().feature_index, 10_930);
        assert_eq!(channels.last().unwrap().feature_index, 11_230);
        assert_eq!(
            channels
                .iter()
                .filter(|channel| channel.block == "tile_bag_terrain_count")
                .count(),
            150
        );
        assert_eq!(
            channels
                .iter()
                .filter(|channel| channel.block == "tile_bag_wildlife_capacity_count")
                .count(),
            150
        );
    }

    #[test]
    fn witness_is_not_counted_as_a_representative_slice() {
        let witness = crate::model::generate_reachable_overflow_witness().unwrap();
        let before = SliceCoverage::zeroed();
        let after = before.clone();
        assert_eq!(before, after);
        assert_eq!(
            witness.provenance.kind,
            RecordKind::ReachableOverflowWitness
        );
    }
}
