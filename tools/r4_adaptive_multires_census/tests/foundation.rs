mod common;

use cascadia_game::D6Transform;
use common::{BoardSpec, sample_record_after_one_turn, straight_coordinates, terminal_record};
use r2_sparse_entity_census::SparsePublicState;
use r4_adaptive_multires_census::{
    ABLATIONS, AdaptiveMultiResolutionState, AdversarialFixtureId, FeatureAblation,
    NearFieldRadius, PACKED_MAGIC, adversarial_fixture_pair,
};

#[test]
fn exact_state_codec_and_r2_semantics_round_trip() {
    let record = sample_record_after_one_turn();
    let sparse = SparsePublicState::from_position_record(&record, None).unwrap();
    for radius in NearFieldRadius::ALL {
        let state = AdaptiveMultiResolutionState::from_sparse_state(&sparse, radius).unwrap();
        assert_eq!(state.to_sparse_state().unwrap(), sparse);
        assert_eq!(
            state.boards[state.focal_relative_seat as usize]
                .near_cells
                .len(),
            radius.capacity()
        );
        let packed = state.to_packed_bytes().unwrap();
        assert_eq!(&packed[..PACKED_MAGIC.len()], PACKED_MAGIC);
        let decoded = AdaptiveMultiResolutionState::from_packed_bytes(&packed).unwrap();
        assert_eq!(decoded, state);
        assert_eq!(decoded.to_sparse_state().unwrap(), sparse);
    }
}

#[test]
fn legal_elongated_board_uses_exact_overflow() {
    let record = terminal_record(vec![BoardSpec::straight_single(
        cascadia_game::Terrain::River,
        cascadia_game::Wildlife::Salmon,
    )]);
    assert_eq!(straight_coordinates().len(), 23);
    for radius in NearFieldRadius::ALL {
        let state =
            AdaptiveMultiResolutionState::from_position_record(&record, None, radius).unwrap();
        let focal = &state.boards[0];
        assert!(!focal.authority_overflow_occupied.is_empty());
        assert_eq!(
            focal.authority_local_occupied.len() + focal.authority_overflow_occupied.len(),
            23
        );
        assert_eq!(state.to_position_record().unwrap(), record);
    }
}

#[test]
fn all_twelve_d6_transforms_and_carried_centers_round_trip() {
    let (record, _) = adversarial_fixture_pair(AdversarialFixtureId::FarLegalFrontier);
    for radius in NearFieldRadius::ALL {
        let state =
            AdaptiveMultiResolutionState::from_position_record(&record, None, radius).unwrap();
        for transform in D6Transform::ALL {
            let transformed = state.transformed(transform).unwrap();
            let restored = transformed.transformed(transform.inverse()).unwrap();
            assert_eq!(restored, state);
            assert_eq!(
                AdaptiveMultiResolutionState::from_packed_bytes(
                    &transformed.to_packed_bytes().unwrap()
                )
                .unwrap(),
                transformed
            );
        }
    }
}

#[test]
fn target_mutation_is_absent_from_every_view() {
    let mut left = sample_record_after_one_turn();
    let mut right = left.clone();
    for target in &mut right.targets {
        *target = !*target;
    }
    left.targets.fill(0);
    for radius in NearFieldRadius::ALL {
        let left_state =
            AdaptiveMultiResolutionState::from_position_record(&left, None, radius).unwrap();
        let right_state =
            AdaptiveMultiResolutionState::from_position_record(&right, None, radius).unwrap();
        assert_eq!(
            left_state.to_packed_bytes().unwrap(),
            right_state.to_packed_bytes().unwrap()
        );
        for arm in ABLATIONS {
            assert_eq!(
                left_state.feature_view(arm).unwrap(),
                right_state.feature_view(arm).unwrap()
            );
        }
    }
}

#[test]
fn ablation_lattice_exposes_only_named_blocks() {
    let record = sample_record_after_one_turn();
    let state =
        AdaptiveMultiResolutionState::from_position_record(&record, None, NearFieldRadius::Radius4)
            .unwrap();
    let near = state.feature_view(FeatureAblation::NearOnly).unwrap();
    assert!(!near.near_cells.is_empty());
    assert!(near.far_habitat_components.is_empty());
    assert!(near.far_wildlife_components.is_empty());
    assert!(near.far_frontier_buckets.is_empty());
    assert!(near.exact_far.is_empty());

    let all = state.feature_view(FeatureAblation::AllTopology).unwrap();
    assert!(all.exact_far.is_empty());
    assert!(
        !all.far_habitat_components.is_empty()
            || !all.far_wildlife_components.is_empty()
            || !all.far_frontier_buckets.is_empty()
    );
    let exact = state
        .feature_view(FeatureAblation::ExactFarControl)
        .unwrap();
    assert!(!exact.exact_far.is_empty());
    assert!(exact.far_habitat_components.is_empty());
}

#[test]
fn packed_decoder_rejects_corruption_truncation_and_trailing_bytes() {
    let state = AdaptiveMultiResolutionState::from_position_record(
        &sample_record_after_one_turn(),
        None,
        NearFieldRadius::Radius5,
    )
    .unwrap();
    let packed = state.to_packed_bytes().unwrap();

    let mut bad_magic = packed.clone();
    bad_magic[..8].fill(0);
    assert!(AdaptiveMultiResolutionState::from_packed_bytes(&bad_magic).is_err());

    let mut bad_radius = packed.clone();
    bad_radius[12] = 121;
    assert!(AdaptiveMultiResolutionState::from_packed_bytes(&bad_radius).is_err());

    assert!(AdaptiveMultiResolutionState::from_packed_bytes(&packed[..packed.len() - 1]).is_err());
    let mut trailing = packed;
    trailing.push(0);
    assert!(AdaptiveMultiResolutionState::from_packed_bytes(&trailing).is_err());
}
