mod common;

use common::{canonical_actions, game};
use r3_action_edit_census::{ActionEdit, AxialCoord, PublicStateTrunk};

#[test]
fn long_range_component_collision_is_visible_beyond_identical_radius_three_patch() {
    let state = game(301);
    let trunk = PublicStateTrunk::observe(&state, 90_000).unwrap();
    let action = canonical_actions(&state)
        .1
        .into_iter()
        .find(|action| action.wildlife.is_some())
        .unwrap();
    let edit = ActionEdit::observe(&state, &trunk, &action).unwrap();
    let left = edit.canonical.clone();
    let mut right = left.clone();
    let component = right
        .global_edit
        .components_added
        .first_mut()
        .expect("placing a tile creates or extends a habitat component");
    component.members.push(AxialCoord::new(11, -7));
    component.members.sort_unstable();
    component.member_count += 1;

    assert_eq!(left.local_patch, right.local_patch);
    assert_ne!(
        postcard::to_allocvec(&left).unwrap(),
        postcard::to_allocvec(&right).unwrap()
    );
}

#[test]
fn long_range_motif_collision_is_visible_beyond_identical_radius_three_patch() {
    let state = game(302);
    let trunk = PublicStateTrunk::observe(&state, 90_001).unwrap();
    let action = canonical_actions(&state)
        .1
        .into_iter()
        .find(|action| action.wildlife.is_some())
        .unwrap();
    let edit = ActionEdit::observe(&state, &trunk, &action).unwrap();
    let left = edit.canonical.clone();
    let mut right = left.clone();
    let motif = right
        .global_edit
        .motifs_added
        .first_mut()
        .expect("the selected action places wildlife");
    motif.offset = AxialCoord::new(12, -8);

    assert_eq!(left.local_patch, right.local_patch);
    assert_ne!(
        postcard::to_allocvec(&left).unwrap(),
        postcard::to_allocvec(&right).unwrap()
    );
}

#[test]
fn variable_length_codec_has_no_silent_truncation_path() {
    let state = game(303);
    let trunk = PublicStateTrunk::observe(&state, 90_002).unwrap();
    let action = canonical_actions(&state).1.remove(0);
    let mut edit = ActionEdit::observe(&state, &trunk, &action).unwrap();

    edit.global_references.frontier_coords = (0..1_024)
        .map(|index| AxialCoord::new(index, -index))
        .collect();
    edit.global_references.supply_archetype_ids = (0..75).collect();
    let packed = edit.to_packed_bytes().unwrap();
    let decoded = ActionEdit::from_packed_bytes(&packed).unwrap();

    assert_eq!(decoded.global_references.frontier_coords.len(), 1_024);
    assert_eq!(decoded.global_references.supply_archetype_ids.len(), 75);
    assert_eq!(decoded, edit);
    assert!(decoded.apply(&trunk).is_err());
}
