mod common;

use std::collections::HashSet;

use cascadia_game::{D6Transform, HexCoord, Terrain};
use cascadia_r2::{SparsePublicState, SuppliedTile};
use common::{
    boundary_terminal_record, elongated_terminal_record, placed_wildlife_count,
    sample_record_after_one_turn,
};

#[test]
fn exact_layers_round_trip_public_record_and_packed_bytes() {
    let record = sample_record_after_one_turn();
    let state = SparsePublicState::from_position_record(&record, None).unwrap();
    assert_eq!(
        state.reconstruct_position_record(record.targets).unwrap(),
        record
    );
    let packed = state.to_packed_bytes().unwrap();
    let decoded = SparsePublicState::from_packed_bytes(&packed).unwrap();
    assert_eq!(decoded, state);
    assert_eq!(decoded.to_packed_bytes().unwrap(), packed);
    assert_eq!(state.wildlife_motifs.len(), placed_wildlife_count(&record));
    assert_eq!(
        state.total_spatial_tokens(),
        state.occupied_tiles.len()
            + state.legal_frontier.len()
            + state.habitat_components.len()
            + state.wildlife_motifs.len()
    );
}

#[test]
fn target_values_are_not_part_of_the_public_state() {
    let record = sample_record_after_one_turn();
    let mut changed = record.clone();
    changed.targets = [u16::MAX; 11];
    let original = SparsePublicState::from_position_record(&record, None).unwrap();
    let changed = SparsePublicState::from_position_record(&changed, None).unwrap();
    assert_eq!(original, changed);
    assert_eq!(
        original.to_packed_bytes().unwrap(),
        changed.to_packed_bytes().unwrap()
    );
    assert!(original.global.targets_omitted);
}

#[test]
fn frontier_matches_an_external_neighbor_set_oracle() {
    let record = sample_record_after_one_turn();
    let state = SparsePublicState::from_position_record(&record, None).unwrap();
    for relative_seat in 0..state.global.player_count {
        let occupied = state
            .occupied_tiles
            .iter()
            .filter(|tile| tile.relative_seat == relative_seat)
            .map(|tile| tile.coord)
            .collect::<HashSet<_>>();
        let oracle = occupied
            .iter()
            .flat_map(|coord| coord.neighbors())
            .filter(|coord| !occupied.contains(coord))
            .collect::<HashSet<_>>();
        let actual = state
            .legal_frontier
            .iter()
            .filter(|token| token.relative_seat == relative_seat)
            .map(|token| token.coord)
            .collect::<HashSet<_>>();
        assert_eq!(actual, oracle);
    }
}

#[test]
fn component_membership_is_complete_and_nonoverlapping_per_terrain() {
    let record = sample_record_after_one_turn();
    let state = SparsePublicState::from_position_record(&record, None).unwrap();
    for relative_seat in 0..state.global.player_count {
        for terrain in Terrain::ALL {
            let expected = state
                .occupied_tiles
                .iter()
                .filter(|tile| {
                    tile.relative_seat == relative_seat
                        && (tile.terrain_a == terrain || tile.terrain_b == Some(terrain))
                })
                .map(|tile| tile.coord)
                .collect::<HashSet<_>>();
            let actual = state
                .habitat_components
                .iter()
                .filter(|component| {
                    component.relative_seat == relative_seat && component.terrain == terrain
                })
                .flat_map(|component| component.members.iter().copied())
                .collect::<HashSet<_>>();
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn every_d6_transform_has_an_exact_inverse_and_rotates_directed_edges() {
    let record = sample_record_after_one_turn();
    let state = SparsePublicState::from_position_record(&record, None).unwrap();
    for transform in D6Transform::ALL {
        let transformed = state.transformed(transform).unwrap();
        assert_eq!(transformed.transformed(transform.inverse()).unwrap(), state);
        for source in &state.occupied_tiles {
            let transformed_coord = source.coord.transformed(transform).unwrap();
            let target = transformed
                .occupied_tiles
                .iter()
                .find(|target| {
                    target.relative_seat == source.relative_seat
                        && target.coord == transformed_coord
                })
                .unwrap();
            for edge in 0..6 {
                let target_edge = transform.transform_edge(edge).unwrap();
                assert_eq!(
                    target.directed_edge_terrains[target_edge],
                    source.directed_edge_terrains[edge]
                );
            }
        }
    }
}

#[test]
fn supplied_tile_populates_exact_rotation_compatibility() {
    let record = sample_record_after_one_turn();
    let supplied: SuppliedTile = "forest,river,0x1f,false".parse().unwrap();
    let state = SparsePublicState::from_position_record(&record, Some(supplied)).unwrap();
    assert!(state.legal_frontier.iter().all(|frontier| {
        let compatibility = frontier.supplied_tile_compatibility.as_ref().unwrap();
        compatibility.rotations.len() == 6
            && compatibility
                .rotations
                .iter()
                .all(|rotation| rotation.matching_edge_bits & !frontier.neighbor_presence_bits == 0)
    }));
}

#[test]
fn sparse_coordinates_do_not_clip_to_any_dense_radius() {
    let record = elongated_terminal_record();
    let state = SparsePublicState::from_position_record(&record, None).unwrap();
    assert_eq!(state.occupied_tiles.len(), 23);
    assert!(
        state
            .occupied_tiles
            .iter()
            .any(|tile| tile.coord.q == 22 && tile.coord.r == 0)
    );
    assert_eq!(
        state.reconstruct_position_record(record.targets).unwrap(),
        record
    );
    assert_eq!(
        SparsePublicState::from_packed_bytes(&state.to_packed_bytes().unwrap()).unwrap(),
        state
    );
}

#[test]
fn legal_frontier_excludes_coordinates_outside_the_rules_domain() {
    let record = boundary_terminal_record();
    let state = SparsePublicState::from_position_record(&record, None).unwrap();
    assert!(state.occupied_tiles.iter().any(|tile| tile.coord.q == 24));
    assert!(state.legal_frontier.iter().all(|frontier| {
        let q = i8::try_from(frontier.coord.q).unwrap();
        let r = i8::try_from(frontier.coord.r).unwrap();
        HexCoord::new(q, r).to_index().is_some()
    }));
    assert!(
        !state
            .legal_frontier
            .iter()
            .any(|frontier| frontier.coord.q == 25)
    );
}
