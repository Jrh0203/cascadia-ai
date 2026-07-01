use cascadia_data::TARGET_DIM;
use cascadia_game::D6Transform;
use serde::{Deserialize, Serialize};

use crate::{
    AdaptiveMultiResolutionState, AdversarialFixtureId, BOUNDED_MAGIC, BoundedArm,
    BoundedFeatureView, NearFieldRadius, R4Error, Result, adversarial_fixture_pair,
};

pub const BOUNDED_ADVERSARIAL_SCHEMA: &str = "r4-bounded-far-quotient-adversarial-v1";
pub const BOUNDED_ADVERSARIAL_PARITY_SCHEMA: &str = "r4-bounded-far-quotient-adversarial-parity-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedAdversarialArmResult {
    pub arm_id: String,
    pub left_blake3: String,
    pub right_blake3: String,
    pub distinguishes: bool,
    pub deterministic_construction: bool,
    pub target_independence: bool,
    pub source_accounting_passed: bool,
    pub bounded_codec_round_trip_passed: bool,
    pub d6_inverse_checks: u64,
    pub d6_shape_covariance_checks: u64,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedAdversarialCase {
    pub fixture_id: String,
    pub scalar_controls_match: bool,
    pub exact_authority_distinct: bool,
    pub exact_codec_round_trip_passed: bool,
    pub mechanical_controls_passed: bool,
    pub information_passing_arm_count: usize,
    pub arms: Vec<BoundedAdversarialArmResult>,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedAdversarialScientific {
    pub schema: String,
    pub experiment_id: String,
    pub radius_id: String,
    pub fixture_count: usize,
    pub arm_count: usize,
    pub arm_case_count: usize,
    pub malformed_envelope_checks: u64,
    pub malformed_envelopes_rejected: bool,
    pub information_passing_arm_ids: Vec<String>,
    pub cases: Vec<BoundedAdversarialCase>,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedAdversarialReport {
    pub scientific: BoundedAdversarialScientific,
    pub scientific_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedAdversarialParityScientific {
    pub schema: String,
    pub experiment_id: String,
    pub report_count: usize,
    pub report_scientific_blake3s: Vec<String>,
    pub all_scientific_reports_identical: bool,
    pub all_suites_passed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundedAdversarialParityReport {
    pub scientific: BoundedAdversarialParityScientific,
    pub scientific_blake3: String,
}

pub fn run_bounded_adversarial_suite() -> Result<BoundedAdversarialReport> {
    let mut cases = Vec::with_capacity(AdversarialFixtureId::ALL.len());
    for fixture in AdversarialFixtureId::ALL {
        cases.push(evaluate_fixture(fixture)?);
    }
    let malformed_envelope_checks = malformed_envelope_suite()?;
    let malformed_envelopes_rejected =
        malformed_envelope_checks == BoundedArm::ALL.len() as u64 * 4;
    let information_passing_arm_ids = BoundedArm::ALL
        .into_iter()
        .filter(|arm| {
            cases.iter().all(|case| {
                case.arms
                    .iter()
                    .find(|result| result.arm_id == arm.id())
                    .is_some_and(|result| result.passed)
            })
        })
        .map(|arm| arm.id().to_owned())
        .collect::<Vec<_>>();
    let passed = malformed_envelopes_rejected
        && cases.iter().all(|case| case.mechanical_controls_passed)
        && !information_passing_arm_ids.is_empty();
    let scientific = BoundedAdversarialScientific {
        schema: BOUNDED_ADVERSARIAL_SCHEMA.to_owned(),
        experiment_id: crate::bounded_census::BOUNDED_EXPERIMENT_ID.to_owned(),
        radius_id: NearFieldRadius::Radius4.id().to_owned(),
        fixture_count: AdversarialFixtureId::ALL.len(),
        arm_count: BoundedArm::ALL.len(),
        arm_case_count: AdversarialFixtureId::ALL.len() * BoundedArm::ALL.len(),
        malformed_envelope_checks,
        malformed_envelopes_rejected,
        information_passing_arm_ids,
        cases,
        passed,
    };
    let scientific_blake3 = scientific_hash(&scientific)?;
    Ok(BoundedAdversarialReport {
        scientific,
        scientific_blake3,
    })
}

pub fn validate_bounded_adversarial_report(report: &BoundedAdversarialReport) -> Result<()> {
    if report.scientific.schema != BOUNDED_ADVERSARIAL_SCHEMA
        || report.scientific.experiment_id != crate::bounded_census::BOUNDED_EXPERIMENT_ID
        || report.scientific.radius_id != NearFieldRadius::Radius4.id()
        || report.scientific.fixture_count != AdversarialFixtureId::ALL.len()
        || report.scientific.arm_count != BoundedArm::ALL.len()
        || report.scientific.arm_case_count
            != AdversarialFixtureId::ALL.len() * BoundedArm::ALL.len()
        || report.scientific.cases.len() != AdversarialFixtureId::ALL.len()
        || scientific_hash(&report.scientific)? != report.scientific_blake3
    {
        return Err(R4Error::AggregateContract(
            "bounded adversarial report contract or scientific hash drifted".to_owned(),
        ));
    }
    for (fixture, case) in AdversarialFixtureId::ALL
        .into_iter()
        .zip(&report.scientific.cases)
    {
        let passing_count = case.arms.iter().filter(|arm| arm.passed).count();
        if case.fixture_id != fixture.id()
            || case.arms.len() != BoundedArm::ALL.len()
            || case.mechanical_controls_passed
                != (case.scalar_controls_match
                    && case.exact_authority_distinct
                    && case.exact_codec_round_trip_passed)
            || case.information_passing_arm_count != passing_count
            || case.passed
                != (case.mechanical_controls_passed && case.information_passing_arm_count > 0)
            || case
                .arms
                .iter()
                .zip(BoundedArm::ALL)
                .any(|(arm_result, arm)| arm_result.arm_id != arm.id())
        {
            return Err(R4Error::AggregateContract(
                "bounded adversarial case ordering drifted".to_owned(),
            ));
        }
    }
    let expected_passing_arms = BoundedArm::ALL
        .into_iter()
        .filter(|arm| {
            report.scientific.cases.iter().all(|case| {
                case.arms
                    .iter()
                    .find(|result| result.arm_id == arm.id())
                    .is_some_and(|result| result.passed)
            })
        })
        .map(|arm| arm.id().to_owned())
        .collect::<Vec<_>>();
    if report.scientific.information_passing_arm_ids != expected_passing_arms
        || report.scientific.passed
            != (report.scientific.malformed_envelopes_rejected
                && report
                    .scientific
                    .cases
                    .iter()
                    .all(|case| case.mechanical_controls_passed)
                && !expected_passing_arms.is_empty())
    {
        return Err(R4Error::AggregateContract(
            "bounded adversarial promotion fields drifted".to_owned(),
        ));
    }
    Ok(())
}

pub fn compare_bounded_adversarial_reports(
    reports: &[BoundedAdversarialReport],
) -> Result<BoundedAdversarialParityReport> {
    if reports.is_empty() {
        return Err(R4Error::AggregateContract(
            "bounded adversarial parity requires at least one report".to_owned(),
        ));
    }
    for report in reports {
        validate_bounded_adversarial_report(report)?;
    }
    let report_scientific_blake3s = reports
        .iter()
        .map(|report| report.scientific_blake3.clone())
        .collect::<Vec<_>>();
    let all_scientific_reports_identical = report_scientific_blake3s
        .windows(2)
        .all(|pair| pair[0] == pair[1]);
    let all_suites_passed = reports.iter().all(|report| report.scientific.passed);
    let scientific = BoundedAdversarialParityScientific {
        schema: BOUNDED_ADVERSARIAL_PARITY_SCHEMA.to_owned(),
        experiment_id: crate::bounded_census::BOUNDED_EXPERIMENT_ID.to_owned(),
        report_count: reports.len(),
        report_scientific_blake3s,
        all_scientific_reports_identical,
        all_suites_passed,
    };
    let scientific_blake3 = scientific_hash(&scientific)?;
    Ok(BoundedAdversarialParityReport {
        scientific,
        scientific_blake3,
    })
}

pub fn validate_bounded_adversarial_parity_report(
    report: &BoundedAdversarialParityReport,
) -> Result<()> {
    if report.scientific.schema != BOUNDED_ADVERSARIAL_PARITY_SCHEMA
        || report.scientific.experiment_id != crate::bounded_census::BOUNDED_EXPERIMENT_ID
        || report.scientific.report_count == 0
        || report.scientific.report_scientific_blake3s.len() != report.scientific.report_count
        || scientific_hash(&report.scientific)? != report.scientific_blake3
    {
        return Err(R4Error::AggregateContract(
            "bounded adversarial parity contract or scientific hash drifted".to_owned(),
        ));
    }
    Ok(())
}

fn evaluate_fixture(fixture: AdversarialFixtureId) -> Result<BoundedAdversarialCase> {
    let (left, right) = adversarial_fixture_pair(fixture);
    let scalar_controls_match =
        left.wildlife_counts == right.wildlife_counts && left.habitat_sizes == right.habitat_sizes;
    let left_state =
        AdaptiveMultiResolutionState::from_position_record(&left, None, NearFieldRadius::Radius4)?;
    let right_state =
        AdaptiveMultiResolutionState::from_position_record(&right, None, NearFieldRadius::Radius4)?;
    let left_exact = left_state.to_packed_bytes()?;
    let right_exact = right_state.to_packed_bytes()?;
    let exact_authority_distinct = left_exact != right_exact;
    let exact_codec_round_trip_passed =
        AdaptiveMultiResolutionState::from_packed_bytes(&left_exact)? == left_state
            && AdaptiveMultiResolutionState::from_packed_bytes(&right_exact)? == right_state;

    let mut left_target_mutation = left.clone();
    let mut right_target_mutation = right.clone();
    mutate_targets(&mut left_target_mutation.targets);
    mutate_targets(&mut right_target_mutation.targets);
    let changed_left = AdaptiveMultiResolutionState::from_position_record(
        &left_target_mutation,
        None,
        NearFieldRadius::Radius4,
    )?;
    let changed_right = AdaptiveMultiResolutionState::from_position_record(
        &right_target_mutation,
        None,
        NearFieldRadius::Radius4,
    )?;

    let mut arms = Vec::with_capacity(BoundedArm::ALL.len());
    for arm in BoundedArm::ALL {
        let left_view = BoundedFeatureView::from_state(&left_state, arm)?;
        let right_view = BoundedFeatureView::from_state(&right_state, arm)?;
        let left_bytes = left_view.canonical_bytes()?;
        let right_bytes = right_view.canonical_bytes()?;
        let deterministic_construction = BoundedFeatureView::from_state(&left_state, arm)?
            .canonical_bytes()?
            == left_bytes
            && BoundedFeatureView::from_state(&right_state, arm)?.canonical_bytes()? == right_bytes;
        let target_independence =
            BoundedFeatureView::from_state(&changed_left, arm)?.canonical_bytes()? == left_bytes
                && BoundedFeatureView::from_state(&changed_right, arm)?.canonical_bytes()?
                    == right_bytes
                && changed_left.to_packed_bytes()? == left_exact
                && changed_right.to_packed_bytes()? == right_exact;
        let bounded_codec_round_trip_passed =
            BoundedFeatureView::from_canonical_bytes(&left_bytes)? == left_view
                && BoundedFeatureView::from_canonical_bytes(&right_bytes)? == right_view;
        let source_accounting_passed =
            accounting_is_exact(&left_view) && accounting_is_exact(&right_view);

        let mut d6_inverse_checks = 0u64;
        let mut d6_shape_covariance_checks = 0u64;
        for state in [&left_state, &right_state] {
            let original = BoundedFeatureView::from_state(state, arm)?;
            let original_shape = view_shape(&original);
            let original_bytes = original.canonical_bytes()?;
            for transform in D6Transform::ALL {
                let transformed = state.transformed(transform)?;
                let transformed_view = BoundedFeatureView::from_state(&transformed, arm)?;
                if view_shape(&transformed_view) != original_shape {
                    return Err(R4Error::DatasetContract(format!(
                        "{} {} changed bounded shape under D6 transform {}",
                        fixture.id(),
                        arm.id(),
                        transform.id()
                    )));
                }
                d6_shape_covariance_checks += 1;
                let restored = transformed.transformed(transform.inverse())?;
                if restored != *state
                    || BoundedFeatureView::from_state(&restored, arm)?.canonical_bytes()?
                        != original_bytes
                {
                    return Err(R4Error::DatasetContract(format!(
                        "{} {} failed D6 inverse transform {}",
                        fixture.id(),
                        arm.id(),
                        transform.id()
                    )));
                }
                d6_inverse_checks += 1;
            }
        }
        let distinguishes = left_bytes != right_bytes;
        let passed = distinguishes
            && deterministic_construction
            && target_independence
            && source_accounting_passed
            && bounded_codec_round_trip_passed
            && d6_inverse_checks == 24
            && d6_shape_covariance_checks == 24;
        arms.push(BoundedAdversarialArmResult {
            arm_id: arm.id().to_owned(),
            left_blake3: blake3::hash(&left_bytes).to_hex().to_string(),
            right_blake3: blake3::hash(&right_bytes).to_hex().to_string(),
            distinguishes,
            deterministic_construction,
            target_independence,
            source_accounting_passed,
            bounded_codec_round_trip_passed,
            d6_inverse_checks,
            d6_shape_covariance_checks,
            passed,
        });
    }
    let mechanical_controls_passed =
        scalar_controls_match && exact_authority_distinct && exact_codec_round_trip_passed;
    let information_passing_arm_count = arms.iter().filter(|arm| arm.passed).count();
    let passed = mechanical_controls_passed && information_passing_arm_count > 0;
    Ok(BoundedAdversarialCase {
        fixture_id: fixture.id().to_owned(),
        scalar_controls_match,
        exact_authority_distinct,
        exact_codec_round_trip_passed,
        mechanical_controls_passed,
        information_passing_arm_count,
        arms,
        passed,
    })
}

fn mutate_targets(targets: &mut [u16; TARGET_DIM]) {
    for target in targets {
        *target = !*target;
    }
}

fn accounting_is_exact(view: &BoundedFeatureView) -> bool {
    let accounting = &view.accounting;
    checked_parts_equal(
        accounting.source_wildlife_buckets,
        accounting.summarized_wildlife_buckets,
        accounting.exact_wildlife_buckets,
    ) && checked_parts_equal(
        accounting.source_wildlife_mass,
        accounting.summarized_wildlife_mass,
        accounting.exact_wildlife_mass,
    ) && checked_parts_equal(
        accounting.source_frontier_buckets,
        accounting.summarized_frontier_buckets,
        accounting.exact_frontier_buckets,
    ) && checked_parts_equal(
        accounting.source_frontier_mass,
        accounting.summarized_frontier_mass,
        accounting.exact_frontier_mass,
    )
}

fn checked_parts_equal(total: u32, summarized: u32, exact: u32) -> bool {
    summarized.checked_add(exact) == Some(total)
}

fn view_shape(view: &BoundedFeatureView) -> (usize, usize, usize, usize, u32, u32) {
    (
        view.spatial_token_count(),
        view.active_scalar_count(),
        view.padded_scalar_slots(),
        view.canonical_bytes()
            .expect("validated bounded view encodes")
            .len(),
        view.accounting.exact_wildlife_buckets,
        view.accounting.exact_frontier_buckets,
    )
}

fn malformed_envelope_suite() -> Result<u64> {
    let (record, _) = adversarial_fixture_pair(AdversarialFixtureId::FarLegalFrontier);
    let state = AdaptiveMultiResolutionState::from_position_record(
        &record,
        None,
        NearFieldRadius::Radius4,
    )?;
    let mut checks = 0u64;
    for arm in BoundedArm::ALL {
        let bytes = BoundedFeatureView::from_state(&state, arm)?.canonical_bytes()?;

        let mut bad_magic = bytes.clone();
        bad_magic[..BOUNDED_MAGIC.len()].fill(0);
        require_rejected(&bad_magic, "bad bounded magic")?;
        checks += 1;

        require_rejected(&bytes[..bytes.len() - 1], "truncated bounded envelope")?;
        checks += 1;

        let mut trailing = bytes.clone();
        trailing.push(0);
        require_rejected(&trailing, "bounded trailing byte")?;
        checks += 1;

        let mut bad_accounting = bytes;
        let accounting_offset = accounting_offset(&bad_accounting)?;
        let source = u32::from_le_bytes(
            bad_accounting[accounting_offset..accounting_offset + 4]
                .try_into()
                .expect("four-byte accounting field"),
        );
        bad_accounting[accounting_offset..accounting_offset + 4]
            .copy_from_slice(&source.wrapping_add(1).to_le_bytes());
        require_rejected(&bad_accounting, "bounded accounting mismatch")?;
        checks += 1;
    }
    Ok(checks)
}

fn require_rejected(bytes: &[u8], label: &str) -> Result<()> {
    if BoundedFeatureView::from_canonical_bytes(bytes).is_ok() {
        return Err(R4Error::DatasetContract(format!(
            "malformed envelope was accepted: {label}"
        )));
    }
    Ok(())
}

fn accounting_offset(bytes: &[u8]) -> Result<usize> {
    let mut offset = BOUNDED_MAGIC.len() + 2 + 1 + 4 + 5 + 1;
    let player_count = usize::from(*bytes.get(offset).ok_or_else(|| {
        R4Error::InvalidBoundedEnvelope("bounded envelope ended before players".to_owned())
    })?);
    offset += 1 + player_count * 15;
    let market_count = usize::from(*bytes.get(offset).ok_or_else(|| {
        R4Error::InvalidBoundedEnvelope("bounded envelope ended before market".to_owned())
    })?);
    offset += 1;
    for _ in 0..market_count {
        offset += 1;
        let has_tile = *bytes.get(offset).ok_or_else(|| {
            R4Error::InvalidBoundedEnvelope("bounded market ended early".to_owned())
        })?;
        offset += 1;
        if has_tile == 1 {
            offset += 4;
        }
        offset += 1;
    }
    let has_supplied = *bytes.get(offset).ok_or_else(|| {
        R4Error::InvalidBoundedEnvelope("bounded supplied-tile flag is missing".to_owned())
    })?;
    offset += 1;
    if has_supplied == 1 {
        offset += 4;
    }
    if offset + 48 > bytes.len() {
        return Err(R4Error::InvalidBoundedEnvelope(
            "bounded accounting block is truncated".to_owned(),
        ));
    }
    Ok(offset)
}

fn scientific_hash(value: &impl Serialize) -> Result<String> {
    Ok(blake3::hash(&serde_json::to_vec(value)?)
        .to_hex()
        .to_string())
}
