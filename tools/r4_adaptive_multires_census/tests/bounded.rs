mod common;

use cascadia_game::D6Transform;
use common::sample_record_after_one_turn;
use r4_adaptive_multires_census::{
    AdaptiveMultiResolutionState, BOUNDED_MAGIC, BoundedArm, BoundedFeatureView, BoundedTokenKind,
    NearFieldRadius, bounded_parent_token_owner, bounded_token_universal_class,
    compare_bounded_adversarial_reports, run_bounded_adversarial_suite,
};

#[test]
fn every_bounded_arm_is_canonical_bounded_and_round_trips() {
    let state = AdaptiveMultiResolutionState::from_position_record(
        &sample_record_after_one_turn(),
        None,
        NearFieldRadius::Radius4,
    )
    .unwrap();
    for arm in BoundedArm::ALL {
        let first = BoundedFeatureView::from_state(&state, arm).unwrap();
        let second = BoundedFeatureView::from_state(&state, arm).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.tokens.len(), first.spatial_token_count());
        assert!(first.spatial_token_count() <= arm.hard_token_max());
        assert_eq!(
            first
                .tokens
                .iter()
                .filter(|token| token.kind == BoundedTokenKind::NearCell)
                .count(),
            61
        );
        assert_eq!(
            first
                .tokens
                .iter()
                .filter(|token| token.kind == BoundedTokenKind::WildlifeSummary)
                .count(),
            20
        );
        let bytes = first.canonical_bytes().unwrap();
        assert_eq!(&bytes[..BOUNDED_MAGIC.len()], BOUNDED_MAGIC);
        assert_eq!(
            BoundedFeatureView::from_canonical_bytes(&bytes).unwrap(),
            first
        );
    }
}

#[test]
fn learned_parent_arms_route_every_token_to_one_registered_board_and_class() {
    let state = AdaptiveMultiResolutionState::from_position_record(
        &sample_record_after_one_turn(),
        None,
        NearFieldRadius::Radius4,
    )
    .unwrap();
    for arm in [
        BoundedArm::SeatMarginal,
        BoundedArm::Directional,
        BoundedArm::Affordance,
    ] {
        let view = BoundedFeatureView::from_state(&state, arm).unwrap();
        for token in &view.tokens {
            let owner = bounded_parent_token_owner(&view, token).unwrap();
            let class = bounded_token_universal_class(token.kind).unwrap();
            assert!(owner < view.global.player_count);
            assert!((5..=9).contains(&class));
            if token.kind == BoundedTokenKind::NearCell {
                assert_eq!(owner, view.global.current_relative_seat);
            } else {
                assert_eq!(i16::from(owner), token.values[0]);
            }
        }
    }
}

#[test]
fn bounded_views_are_target_independent_and_d6_inverse_stable() {
    let left = sample_record_after_one_turn();
    let mut changed_targets = left.clone();
    for target in &mut changed_targets.targets {
        *target = !*target;
    }
    let state =
        AdaptiveMultiResolutionState::from_position_record(&left, None, NearFieldRadius::Radius4)
            .unwrap();
    let changed = AdaptiveMultiResolutionState::from_position_record(
        &changed_targets,
        None,
        NearFieldRadius::Radius4,
    )
    .unwrap();
    for arm in BoundedArm::ALL {
        let original = BoundedFeatureView::from_state(&state, arm)
            .unwrap()
            .canonical_bytes()
            .unwrap();
        assert_eq!(
            BoundedFeatureView::from_state(&changed, arm)
                .unwrap()
                .canonical_bytes()
                .unwrap(),
            original
        );
        for transform in D6Transform::ALL {
            let restored = state
                .transformed(transform)
                .unwrap()
                .transformed(transform.inverse())
                .unwrap();
            assert_eq!(restored, state);
            assert_eq!(
                BoundedFeatureView::from_state(&restored, arm)
                    .unwrap()
                    .canonical_bytes()
                    .unwrap(),
                original
            );
        }
    }
}

#[test]
fn bounded_decoder_rejects_corruption_truncation_and_trailing_bytes() {
    let state = AdaptiveMultiResolutionState::from_position_record(
        &sample_record_after_one_turn(),
        None,
        NearFieldRadius::Radius4,
    )
    .unwrap();
    let mut bytes = BoundedFeatureView::from_state(&state, BoundedArm::Directional)
        .unwrap()
        .canonical_bytes()
        .unwrap();

    let mut bad_magic = bytes.clone();
    bad_magic[..BOUNDED_MAGIC.len()].fill(0);
    assert!(BoundedFeatureView::from_canonical_bytes(&bad_magic).is_err());

    assert!(BoundedFeatureView::from_canonical_bytes(&bytes[..bytes.len() - 1]).is_err());
    bytes.push(0);
    assert!(BoundedFeatureView::from_canonical_bytes(&bytes).is_err());
}

#[test]
fn complete_bounded_adversarial_matrix_is_reproducible_and_passing() {
    let first = run_bounded_adversarial_suite().unwrap();
    let second = run_bounded_adversarial_suite().unwrap();
    assert_eq!(first, second);
    assert!(first.scientific.passed, "{:#?}", first.scientific);
    assert_eq!(first.scientific.fixture_count, 7);
    assert_eq!(first.scientific.arm_count, 4);
    assert_eq!(first.scientific.arm_case_count, 28);
    assert_eq!(first.scientific.malformed_envelope_checks, 16);
    assert_eq!(
        first.scientific.information_passing_arm_ids,
        BoundedArm::ALL
            .into_iter()
            .map(|arm| arm.id().to_owned())
            .collect::<Vec<_>>()
    );
    assert!(
        first
            .scientific
            .cases
            .iter()
            .flat_map(|case| &case.arms)
            .all(|arm| arm.passed)
    );
}

#[test]
fn bounded_adversarial_parity_requires_identical_reports() {
    let report = run_bounded_adversarial_suite().unwrap();
    let parity = compare_bounded_adversarial_reports(&[
        report.clone(),
        report.clone(),
        report.clone(),
        report,
    ])
    .unwrap();
    assert_eq!(parity.scientific.report_count, 4);
    assert!(parity.scientific.all_scientific_reports_identical);
    assert!(parity.scientific.all_suites_passed);
}
