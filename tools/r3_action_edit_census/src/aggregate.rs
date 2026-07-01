use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::BufReader,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{
    R3Error, Result, canonical_blake3,
    census::{
        CensusCounters, CensusReport, CorpusContract, ExactDistribution, PRODUCTION_SHARD_COUNT,
        PromotionAssessment, R3_CENSUS_PROTOCOL_ID, R3_CENSUS_SCHEMA_VERSION, R3_EXPERIMENT_ID,
        RadiusCoverageSummary, SHARD_PARTITION_RULE, assess_promotion, merge_radius_coverage,
        validate_census_report, validate_radius_coverage,
    },
    source::{
        RuntimeIdentity, capture_runtime_identity, validate_blake3, validate_runtime_identity,
    },
    strict_json,
};

pub const R3_AGGREGATE_SCHEMA_VERSION: u16 = 1;
pub const R3_AGGREGATE_ARTIFACT_KIND: &str = "r3_action_edit_census_aggregate";
pub const R3_ORDER_PROOF_ARTIFACT_KIND: &str = "r3_action_edit_census_order_proof";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AggregateShardIdentity {
    pub shard_index: usize,
    pub scientific_blake3: String,
    pub train_owned_seeds_blake3: String,
    pub validation_owned_seeds_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AggregateScientific {
    pub schema_version: u16,
    pub artifact_kind: String,
    pub runtime: RuntimeIdentity,
    pub experiment_id: String,
    pub protocol_id: String,
    pub shard_schema_version: u16,
    pub corpus: CorpusContract,
    pub shard_count: usize,
    pub partition_rule: String,
    pub shards: Vec<AggregateShardIdentity>,
    pub production_coverage: bool,
    pub counters: CensusCounters,
    pub action_count: ExactDistribution,
    pub trunk_tokens: ExactDistribution,
    pub trunk_packed_bytes: ExactDistribution,
    pub edit_tokens: ExactDistribution,
    pub edit_packed_bytes: ExactDistribution,
    pub radius_coverage: [RadiusCoverageSummary; 3],
    pub promotion: PromotionAssessment,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AggregateReport {
    pub scientific: AggregateScientific,
    pub scientific_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AggregateOrderProofScientific {
    pub schema_version: u16,
    pub artifact_kind: String,
    pub aggregate_scientific_blake3: String,
    pub aggregate_file_blake3: String,
    pub aggregate_file_bytes: u64,
    pub byte_identical: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AggregateOrderProof {
    pub scientific: AggregateOrderProofScientific,
    pub scientific_blake3: String,
}

pub fn aggregate_census_files(paths: &[PathBuf]) -> Result<AggregateReport> {
    if paths.is_empty() {
        return Err(invalid("R3 aggregate requires shard reports"));
    }
    let reports = paths
        .iter()
        .map(|path| {
            let reader = BufReader::new(File::open(path)?);
            strict_json::from_reader(reader).map_err(R3Error::from)
        })
        .collect::<Result<Vec<CensusReport>>>()?;
    aggregate_census_reports(&reports)
}

pub fn aggregate_census_reports(reports: &[CensusReport]) -> Result<AggregateReport> {
    let runtime_before = capture_runtime_identity()?;
    let mut ordered = reports.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|report| report.scientific.config.shard_index);

    validate_shard_set(&ordered, &runtime_before)?;

    let mut counters = CensusCounters::default();
    for report in &ordered {
        counters.merge(&report.scientific.counters)?;
    }
    let action_count =
        ExactDistribution::merge(ordered.iter().map(|report| &report.scientific.action_count))?;
    let trunk_tokens =
        ExactDistribution::merge(ordered.iter().map(|report| &report.scientific.trunk_tokens))?;
    let trunk_packed_bytes = ExactDistribution::merge(
        ordered
            .iter()
            .map(|report| &report.scientific.trunk_packed_bytes),
    )?;
    let edit_tokens =
        ExactDistribution::merge(ordered.iter().map(|report| &report.scientific.edit_tokens))?;
    let edit_packed_bytes = ExactDistribution::merge(
        ordered
            .iter()
            .map(|report| &report.scientific.edit_packed_bytes),
    )?;
    let radius_coverage = merge_radius_coverage(
        ordered
            .iter()
            .map(|report| report.scientific.radius_coverage.clone()),
    )?;
    let promotion = assess_promotion(&counters, &edit_tokens, &edit_packed_bytes)?;
    let corpus = ordered[0].scientific.config.corpus.clone();
    let shards = ordered
        .iter()
        .map(|report| AggregateShardIdentity {
            shard_index: report.scientific.config.shard_index,
            scientific_blake3: report.scientific_blake3.clone(),
            train_owned_seeds_blake3: report.scientific.ownership.train.owned_seeds_blake3.clone(),
            validation_owned_seeds_blake3: report
                .scientific
                .ownership
                .validation
                .owned_seeds_blake3
                .clone(),
        })
        .collect();
    let scientific = AggregateScientific {
        schema_version: R3_AGGREGATE_SCHEMA_VERSION,
        artifact_kind: R3_AGGREGATE_ARTIFACT_KIND.to_owned(),
        runtime: runtime_before.clone(),
        experiment_id: R3_EXPERIMENT_ID.to_owned(),
        protocol_id: R3_CENSUS_PROTOCOL_ID.to_owned(),
        shard_schema_version: R3_CENSUS_SCHEMA_VERSION,
        corpus,
        shard_count: PRODUCTION_SHARD_COUNT,
        partition_rule: SHARD_PARTITION_RULE.to_owned(),
        shards,
        production_coverage: true,
        counters,
        action_count,
        trunk_tokens,
        trunk_packed_bytes,
        edit_tokens,
        edit_packed_bytes,
        radius_coverage,
        promotion,
    };
    validate_aggregate_scientific(&scientific)?;
    let report = AggregateReport {
        scientific_blake3: canonical_blake3(&scientific)?,
        scientific,
    };
    let runtime_after = capture_runtime_identity()?;
    if runtime_after != runtime_before {
        return Err(invalid(
            "R3 source bundle or executable changed while aggregating shards",
        ));
    }
    validate_aggregate_report(&report, &runtime_after)?;
    Ok(report)
}

pub fn prove_aggregate_order(forward: &Path, reverse: &Path) -> Result<AggregateOrderProof> {
    let forward_bytes = fs::read(forward)?;
    let reverse_bytes = fs::read(reverse)?;
    if forward_bytes != reverse_bytes {
        return Err(invalid(
            "R3 forward and reverse aggregate files are not byte-identical",
        ));
    }
    let forward_report: AggregateReport = strict_json::from_reader(forward_bytes.as_slice())?;
    let reverse_report: AggregateReport = strict_json::from_reader(reverse_bytes.as_slice())?;
    let runtime = capture_runtime_identity()?;
    validate_aggregate_report(&forward_report, &runtime)?;
    validate_aggregate_report(&reverse_report, &runtime)?;
    if forward_report != reverse_report {
        return Err(invalid("R3 forward and reverse aggregate payloads differ"));
    }
    let scientific = AggregateOrderProofScientific {
        schema_version: R3_AGGREGATE_SCHEMA_VERSION,
        artifact_kind: R3_ORDER_PROOF_ARTIFACT_KIND.to_owned(),
        aggregate_scientific_blake3: forward_report.scientific_blake3,
        aggregate_file_blake3: blake3::hash(&forward_bytes).to_hex().to_string(),
        aggregate_file_bytes: u64::try_from(forward_bytes.len())?,
        byte_identical: true,
    };
    Ok(AggregateOrderProof {
        scientific_blake3: canonical_blake3(&scientific)?,
        scientific,
    })
}

fn validate_shard_set(ordered: &[&CensusReport], current_runtime: &RuntimeIdentity) -> Result<()> {
    if ordered.len() != PRODUCTION_SHARD_COUNT {
        return Err(invalid(format!(
            "R3 production aggregate requires exactly {PRODUCTION_SHARD_COUNT} shards"
        )));
    }
    let expected_indices = (0..PRODUCTION_SHARD_COUNT).collect::<Vec<_>>();
    let indices = ordered
        .iter()
        .map(|report| report.scientific.config.shard_index)
        .collect::<Vec<_>>();
    if indices != expected_indices {
        return Err(invalid(
            "R3 production aggregate requires each shard index exactly once",
        ));
    }
    let mut scientific_hashes = BTreeSet::new();
    for report in ordered {
        validate_census_report(report, current_runtime)?;
        if !report.scientific.production_coverage
            || !report.scientific.config.is_production_coverage()
        {
            return Err(invalid(
                "R3 aggregate received a non-production shard configuration",
            ));
        }
        if !scientific_hashes.insert(report.scientific_blake3.as_str()) {
            return Err(invalid("R3 aggregate received duplicate shard evidence"));
        }
    }
    Ok(())
}

fn validate_aggregate_report(
    report: &AggregateReport,
    current_runtime: &RuntimeIdentity,
) -> Result<()> {
    validate_blake3("aggregate scientific", &report.scientific_blake3)?;
    if report.scientific_blake3 != canonical_blake3(&report.scientific)? {
        return Err(invalid("R3 aggregate scientific BLAKE3 drifted"));
    }
    validate_aggregate_scientific(&report.scientific)?;
    if &report.scientific.runtime != current_runtime {
        return Err(invalid(
            "R3 aggregate source bundle or executable differs from the current runtime",
        ));
    }
    Ok(())
}

fn validate_aggregate_scientific(scientific: &AggregateScientific) -> Result<()> {
    if scientific.schema_version != R3_AGGREGATE_SCHEMA_VERSION
        || scientific.artifact_kind != R3_AGGREGATE_ARTIFACT_KIND
        || scientific.experiment_id != R3_EXPERIMENT_ID
        || scientific.protocol_id != R3_CENSUS_PROTOCOL_ID
        || scientific.shard_schema_version != R3_CENSUS_SCHEMA_VERSION
    {
        return Err(invalid("R3 aggregate schema or identity drifted"));
    }
    validate_runtime_identity(&scientific.runtime)?;
    scientific.corpus.validate()?;
    if !scientific.corpus.is_frozen_production()
        || scientific.shard_count != PRODUCTION_SHARD_COUNT
        || scientific.partition_rule != SHARD_PARTITION_RULE
        || !scientific.production_coverage
    {
        return Err(invalid(
            "R3 aggregate does not represent the frozen production corpus",
        ));
    }
    if scientific.shards.len() != PRODUCTION_SHARD_COUNT {
        return Err(invalid("R3 aggregate shard identity count drifted"));
    }
    for (expected, shard) in scientific.shards.iter().enumerate() {
        if shard.shard_index != expected {
            return Err(invalid("R3 aggregate shard identities are noncanonical"));
        }
        validate_blake3("shard scientific", &shard.scientific_blake3)?;
        validate_blake3("train owned seeds", &shard.train_owned_seeds_blake3)?;
        validate_blake3(
            "validation owned seeds",
            &shard.validation_owned_seeds_blake3,
        )?;
    }
    for distribution in [
        &scientific.action_count,
        &scientific.trunk_tokens,
        &scientific.trunk_packed_bytes,
        &scientific.edit_tokens,
        &scientific.edit_packed_bytes,
    ] {
        distribution.validate()?;
    }
    validate_radius_coverage(&scientific.radius_coverage)?;
    validate_aggregate_counter_evidence(scientific)?;
    if scientific.promotion
        != assess_promotion(
            &scientific.counters,
            &scientific.edit_tokens,
            &scientific.edit_packed_bytes,
        )?
    {
        return Err(invalid(
            "R3 aggregate promotion assessment differs from exact evidence",
        ));
    }
    Ok(())
}

fn validate_aggregate_counter_evidence(scientific: &AggregateScientific) -> Result<()> {
    let counters = &scientific.counters;
    let expected_train_positions = u64::from(scientific.corpus.train_games)
        .checked_mul(crate::census::POSITIONS_PER_GAME)
        .ok_or_else(|| invalid("R3 aggregate train position count overflowed"))?;
    let expected_validation_positions = u64::from(scientific.corpus.validation_games)
        .checked_mul(crate::census::POSITIONS_PER_GAME)
        .ok_or_else(|| invalid("R3 aggregate validation position count overflowed"))?;
    let expected_decisions = expected_train_positions
        .checked_add(expected_validation_positions)
        .ok_or_else(|| invalid("R3 aggregate decision count overflowed"))?;
    let expected_d6_checks = expected_decisions
        .checked_mul(12)
        .ok_or_else(|| invalid("R3 aggregate D6 check count overflowed"))?;
    let total_verified_actions = counters.total_verified_actions()?;
    if counters.train_positions != expected_train_positions
        || counters.validation_positions != expected_validation_positions
        || counters.canonical_decisions != expected_decisions
        || counters.state_trunk_encodings != expected_decisions
        || counters.d6_checks != expected_d6_checks
        || counters.exact_apply_checks != total_verified_actions
        || counters.authoritative_public_successor_checks != total_verified_actions
        || counters.supply_delta_parity_checks != total_verified_actions
        || counters.regenerated_global_edit_checks != total_verified_actions
        || counters.codec_round_trip_checks != total_verified_actions
    {
        return Err(invalid(
            "R3 aggregate counters do not match frozen corpus coverage",
        ));
    }
    if scientific.action_count.summary.count != expected_decisions
        || scientific.action_count.histogram.sum()? != u128::from(counters.canonical_actions)
        || scientific.trunk_tokens.summary.count != expected_decisions
        || scientific.trunk_packed_bytes.summary.count != expected_decisions
        || scientific.edit_tokens.summary.count != total_verified_actions
        || scientific.edit_packed_bytes.summary.count != total_verified_actions
        || scientific
            .radius_coverage
            .iter()
            .any(|coverage| coverage.total_actions != total_verified_actions)
    {
        return Err(invalid(
            "R3 aggregate histograms or radius totals do not match counters",
        ));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> R3Error {
    R3Error::Invariant(message.into())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::census::{
        CensusConfig, ExactHistogram, RadiusCoverageSummary, ShardOwnership, build_scientific,
    };
    use tempfile::tempdir;

    use super::*;

    fn distribution(value: u64, count: u64) -> ExactDistribution {
        ExactDistribution::from_histogram(ExactHistogram {
            bins: BTreeMap::from([(value, count)]),
        })
        .unwrap()
    }

    fn shard_reports() -> Vec<CensusReport> {
        let runtime = capture_runtime_identity().unwrap();
        (0..PRODUCTION_SHARD_COUNT)
            .map(|shard_index| {
                let config = CensusConfig::production_shard(shard_index);
                let ownership = ShardOwnership::from_config(&config).unwrap();
                let decisions = 400;
                let counters = CensusCounters {
                    train_positions: 320,
                    validation_positions: 80,
                    canonical_decisions: decisions,
                    state_trunk_encodings: decisions,
                    canonical_actions: decisions,
                    paid_wipe_sentinel_actions: 0,
                    exact_apply_checks: decisions,
                    authoritative_public_successor_checks: decisions,
                    supply_delta_parity_checks: decisions,
                    regenerated_global_edit_checks: decisions,
                    codec_round_trip_checks: decisions,
                    d6_checks: decisions * 12,
                    maximum_wildlife_wipe_sequence: 0,
                };
                let radius = std::array::from_fn(|index| RadiusCoverageSummary {
                    radius: index as u8 + 1,
                    changed_coordinates: decisions,
                    covered_coordinates: decisions,
                    complete_actions: decisions,
                    total_actions: decisions,
                    covered_parts_per_million: 1_000_000,
                    complete_parts_per_million: 1_000_000,
                });
                let scientific = build_scientific(
                    runtime.clone(),
                    config,
                    ownership,
                    counters,
                    distribution(1, decisions),
                    distribution(10, decisions),
                    distribution(100, decisions),
                    distribution(20, decisions),
                    distribution(200, decisions),
                    radius,
                )
                .unwrap();
                CensusReport {
                    scientific_blake3: canonical_blake3(&scientific).unwrap(),
                    scientific,
                }
            })
            .collect()
    }

    #[test]
    fn reverse_input_order_produces_identical_aggregate() {
        let forward = shard_reports();
        let mut reverse = forward.clone();
        reverse.reverse();
        assert_eq!(
            aggregate_census_reports(&forward).unwrap(),
            aggregate_census_reports(&reverse).unwrap()
        );
    }

    #[test]
    fn missing_duplicate_and_tampered_shards_fail_closed() {
        let reports = shard_reports();
        assert!(aggregate_census_reports(&reports[..3]).is_err());

        let mut duplicate = reports.clone();
        duplicate[3] = duplicate[2].clone();
        assert!(aggregate_census_reports(&duplicate).is_err());

        let mut tampered = reports;
        tampered[0].scientific.counters.canonical_actions += 1;
        assert!(aggregate_census_reports(&tampered).is_err());
    }

    #[test]
    fn source_mismatch_fails_even_when_the_shard_hash_is_recomputed() {
        let mut reports = shard_reports();
        reports[0].scientific.runtime.executable_blake3 = "0".repeat(64);
        reports[0].scientific_blake3 = canonical_blake3(&reports[0].scientific).unwrap();
        assert!(aggregate_census_reports(&reports).is_err());
    }

    #[test]
    fn strict_file_loading_rejects_duplicate_and_unknown_keys() {
        let temp = tempdir().unwrap();
        let duplicate = temp.path().join("duplicate.json");
        fs::write(
            &duplicate,
            r#"{"scientific":{},"scientific":{},"scientific_blake3":"00"}"#,
        )
        .unwrap();
        assert!(aggregate_census_files(&[duplicate]).is_err());

        let unknown = temp.path().join("unknown.json");
        let mut value = serde_json::to_value(&shard_reports()[0]).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("unexpected".to_owned(), serde_json::Value::Bool(true));
        fs::write(&unknown, serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(aggregate_census_files(&[unknown]).is_err());
    }

    #[test]
    fn forward_reverse_files_produce_a_content_bound_order_proof() {
        let reports = shard_reports();
        let mut reverse = reports.clone();
        reverse.reverse();
        let forward = aggregate_census_reports(&reports).unwrap();
        let reverse = aggregate_census_reports(&reverse).unwrap();
        let temp = tempdir().unwrap();
        let forward_path = temp.path().join("forward.json");
        let reverse_path = temp.path().join("reverse.json");
        crate::write_json_atomic(&forward_path, &forward).unwrap();
        crate::write_json_atomic(&reverse_path, &reverse).unwrap();

        let proof = prove_aggregate_order(&forward_path, &reverse_path).unwrap();
        assert!(proof.scientific.byte_identical);
        assert_eq!(
            proof.scientific.aggregate_scientific_blake3,
            forward.scientific_blake3
        );

        fs::write(
            &reverse_path,
            [fs::read(&reverse_path).unwrap(), b"\n".to_vec()].concat(),
        )
        .unwrap();
        assert!(prove_aggregate_order(&forward_path, &reverse_path).is_err());
    }
}
