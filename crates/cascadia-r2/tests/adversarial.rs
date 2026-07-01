mod common;

use cascadia_data::{BOARD_ENTITY_SIZE, MAX_BOARD_TILES};
use cascadia_r2::{PACKED_MAGIC, SparsePublicState};
use common::{elongated_terminal_record, sample_record_after_one_turn};

const NONE: u8 = u8::MAX;
type RecordMutation = Box<dyn Fn(&mut cascadia_data::PositionRecord)>;

#[test]
fn malformed_entities_fail_closed() {
    let base = sample_record_after_one_turn();
    let mutations: Vec<RecordMutation> = vec![
        Box::new(|record| record.board_entities[0][0][2] = 9),
        Box::new(|record| record.board_entities[0][0][4] = 6),
        Box::new(|record| record.board_entities[0][0][5] = 0),
        Box::new(|record| record.board_entities[0][0][6] = 9),
        Box::new(|record| record.board_entities[0][0][7] = 2),
        Box::new(|record| record.board_entities[0][0][0] = 127),
    ];
    for mutate in mutations {
        let mut record = base.clone();
        mutate(&mut record);
        assert!(SparsePublicState::from_position_record(&record, None).is_err());
    }
}

#[test]
fn duplicate_noncanonical_and_padding_rows_fail_closed() {
    let base = sample_record_after_one_turn();

    let mut duplicate = base.clone();
    let duplicate_coord = [
        duplicate.board_entities[0][0][0],
        duplicate.board_entities[0][0][1],
    ];
    duplicate.board_entities[0][1][..2].copy_from_slice(&duplicate_coord);
    assert!(SparsePublicState::from_position_record(&duplicate, None).is_err());

    let mut unsorted = base.clone();
    unsorted.board_entities[0].swap(0, 1);
    assert!(SparsePublicState::from_position_record(&unsorted, None).is_err());

    let mut padding = base;
    let first_padding = usize::from(padding.board_counts[0]);
    padding.board_entities[0][first_padding] = [0; BOARD_ENTITY_SIZE];
    assert!(SparsePublicState::from_position_record(&padding, None).is_err());
}

#[test]
fn impossible_counts_metadata_and_connectivity_fail_closed() {
    let base = sample_record_after_one_turn();

    let mut board_count = base.clone();
    board_count.board_counts[0] = MAX_BOARD_TILES as u8;
    assert!(SparsePublicState::from_position_record(&board_count, None).is_err());

    let mut wildlife_count = base.clone();
    wildlife_count.wildlife_counts[0][0] = wildlife_count.wildlife_counts[0][0].saturating_add(1);
    assert!(SparsePublicState::from_position_record(&wildlife_count, None).is_err());

    let mut habitat_size = base.clone();
    habitat_size.habitat_sizes[0][0] = habitat_size.habitat_sizes[0][0].saturating_add(1);
    assert!(SparsePublicState::from_position_record(&habitat_size, None).is_err());

    let mut nature_tokens = base.clone();
    nature_tokens.nature_tokens[0] = u8::MAX;
    assert!(SparsePublicState::from_position_record(&nature_tokens, None).is_err());

    let mut detached = base.clone();
    detached.board_entities[0][0][0] = 20;
    detached.board_entities[0][0][1] = 20;
    detached.board_entities[0].sort_unstable_by_key(|entity| (entity[0] as i8, entity[1] as i8));
    assert!(SparsePublicState::from_position_record(&detached, None).is_err());

    let mut inactive = elongated_terminal_record();
    inactive.board_entities[1][0] = [0; BOARD_ENTITY_SIZE];
    assert!(SparsePublicState::from_position_record(&inactive, None).is_err());
}

#[test]
fn malformed_market_fails_closed() {
    let base = sample_record_after_one_turn();

    let mut reserved = base.clone();
    reserved.market_entities[0][5] = 1;
    assert!(SparsePublicState::from_position_record(&reserved, None).is_err());

    let mut partial = base.clone();
    partial.market_entities[0][0] = NONE;
    assert!(SparsePublicState::from_position_record(&partial, None).is_err());
}

#[test]
fn packed_decoder_rejects_corruption_and_noncanonical_bytes() {
    let state =
        SparsePublicState::from_position_record(&sample_record_after_one_turn(), None).unwrap();
    let packed = state.to_packed_bytes().unwrap();

    let mut bad_magic = packed.clone();
    bad_magic[..PACKED_MAGIC.len()].fill(0);
    assert!(SparsePublicState::from_packed_bytes(&bad_magic).is_err());

    let mut bad_schema = packed.clone();
    bad_schema[8..10].copy_from_slice(&2u16.to_le_bytes());
    assert!(SparsePublicState::from_packed_bytes(&bad_schema).is_err());

    let mut bad_flags = packed.clone();
    bad_flags[10..12].copy_from_slice(&0x8000u16.to_le_bytes());
    assert!(SparsePublicState::from_packed_bytes(&bad_flags).is_err());

    assert!(SparsePublicState::from_packed_bytes(&packed[..packed.len() - 1]).is_err());

    let mut trailing = packed;
    trailing.push(0);
    assert!(SparsePublicState::from_packed_bytes(&trailing).is_err());
}

#[test]
fn inactive_padding_sentinel_is_exact() {
    let mut record = sample_record_after_one_turn();
    for row in usize::from(record.board_counts[0])..MAX_BOARD_TILES {
        assert_eq!(record.board_entities[0][row], [NONE; BOARD_ENTITY_SIZE]);
    }
    record.board_entities[0][MAX_BOARD_TILES - 1][0] = 0;
    assert!(SparsePublicState::from_position_record(&record, None).is_err());
}
