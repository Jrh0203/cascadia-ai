use r4_adaptive_multires_census::{
    AdversarialFixtureId, compare_adversarial_reports, evaluate_adversarial_fixture,
    run_adversarial_suite,
};

fn assert_fixture(fixture: AdversarialFixtureId) {
    let cases = evaluate_adversarial_fixture(fixture).unwrap();
    assert_eq!(cases.len(), 2);
    for case in cases {
        assert!(
            case.passed,
            "{} {} failed its production adversarial contract: {case:#?}",
            case.fixture_id, case.radius_id
        );
    }
}

#[test]
fn habitat_block_resolves_equal_scalar_different_far_components() {
    assert_fixture(AdversarialFixtureId::FarHabitatComponent);
}

#[test]
fn wildlife_block_resolves_long_salmon_topology() {
    assert_fixture(AdversarialFixtureId::LongSalmonTopology);
}

#[test]
fn wildlife_block_resolves_far_hawk_conflict() {
    assert_fixture(AdversarialFixtureId::FarHawkConflict);
}

#[test]
fn wildlife_block_resolves_far_fox_diversity() {
    assert_fixture(AdversarialFixtureId::FarFoxDiversity);
}

#[test]
fn frontier_block_resolves_same_near_different_far_legal_affordance() {
    assert_fixture(AdversarialFixtureId::FarLegalFrontier);
}

#[test]
fn frontier_block_resolves_overflow_consequence() {
    assert_fixture(AdversarialFixtureId::OverflowConsequence);
}

#[test]
fn relative_opponent_topology_is_not_pooled_away() {
    assert_fixture(AdversarialFixtureId::RelativeOpponentBoard);
}

#[test]
fn complete_report_is_reproducible_and_passes() {
    let first = run_adversarial_suite().unwrap();
    let second = run_adversarial_suite().unwrap();
    assert_eq!(first, second);
    assert!(first.scientific.passed);
    assert_eq!(first.scientific.fixture_count, 7);
    assert_eq!(first.scientific.case_count, 14);
}

#[test]
fn cross_host_parity_requires_identical_passing_scientific_reports() {
    let report = run_adversarial_suite().unwrap();
    let parity =
        compare_adversarial_reports(&[report.clone(), report.clone(), report.clone(), report])
            .unwrap();
    assert_eq!(parity.scientific.report_count, 4);
    assert!(parity.scientific.all_scientific_reports_identical);
    assert!(parity.scientific.all_suites_passed);
}
