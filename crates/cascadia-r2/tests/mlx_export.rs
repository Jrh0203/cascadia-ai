mod common;

use cascadia_r2::{
    BOARD_OWNERSHIP_ENCODING, BOARD_SLOTS, BOARD_TOKEN_CAPACITY,
    FOUNDATION_PER_BOARD_MAX_ACTIVE_TOKENS, GRAPH_MAX_DEGREE, SparsePublicState, TOKEN_CAPACITY,
    TOKEN_PAYLOAD_WIDTH, compact_encoded_state, encode_sparse_state,
};
use common::sample_record_after_one_turn;

#[test]
fn mlx_encoding_preserves_exact_layers_ownership_and_zero_padding() {
    let record = sample_record_after_one_turn();
    let state = SparsePublicState::from_position_record(&record, None).unwrap();
    let encoded = encode_sparse_state(&state).unwrap();

    assert_eq!(BOARD_OWNERSHIP_ENCODING, "relative-seat-one-hot-4");
    assert_eq!(FOUNDATION_PER_BOARD_MAX_ACTIVE_TOKENS, 92);
    assert_eq!(encoded.token_types.len(), TOKEN_CAPACITY);
    assert_eq!(encoded.token_seats.len(), TOKEN_CAPACITY);
    assert_eq!(
        encoded.token_payload.len(),
        TOKEN_CAPACITY * TOKEN_PAYLOAD_WIDTH
    );
    assert_eq!(encoded.graph_token_offsets.len(), TOKEN_CAPACITY + 1);
    assert_eq!(encoded.active_tokens(), state.total_spatial_tokens());
    assert!(encoded.max_degree <= GRAPH_MAX_DEGREE);

    for board in 0..BOARD_SLOTS {
        let counts = encoded.board_type_counts[board];
        let active = counts
            .iter()
            .map(|value| usize::from(*value))
            .sum::<usize>();
        let board_start = board * BOARD_TOKEN_CAPACITY;
        let mut cursor = board_start;
        for (type_index, count) in counts.into_iter().enumerate() {
            let end = cursor + usize::from(count);
            assert!(
                encoded.token_types[cursor..end]
                    .iter()
                    .all(|value| *value == type_index as u8 + 1)
            );
            assert!(
                encoded.token_seats[cursor..end]
                    .iter()
                    .all(|value| usize::from(*value) == board)
            );
            cursor = end;
        }
        assert_eq!(cursor, board_start + active);
        for slot in board_start..board_start + BOARD_TOKEN_CAPACITY {
            let payload = &encoded.token_payload
                [slot * TOKEN_PAYLOAD_WIDTH..(slot + 1) * TOKEN_PAYLOAD_WIDTH];
            if slot >= cursor {
                assert_eq!(encoded.token_types[slot], 0);
                assert_eq!(encoded.token_seats[slot], 0);
                assert!(payload.iter().all(|value| *value == 0));
            }
            let edge_start = encoded.graph_token_offsets[slot] as usize;
            let edge_end = encoded.graph_token_offsets[slot + 1] as usize;
            for target in &encoded.graph_targets[edge_start..edge_end] {
                assert_eq!(
                    usize::from(*target) / BOARD_TOKEN_CAPACITY,
                    board,
                    "graph edges must remain board-local"
                );
            }
        }
    }
}

#[test]
fn compact_encoding_removes_only_padding_and_remaps_graph_targets() {
    let record = sample_record_after_one_turn();
    let state = SparsePublicState::from_position_record(&record, None).unwrap();
    let encoded = encode_sparse_state(&state).unwrap();
    let compact = compact_encoded_state(&encoded).unwrap();
    let counts = encoded.board_counts();
    let active: usize = counts.iter().sum();
    assert_eq!(compact.token_types.len(), active);
    assert_eq!(compact.token_seats.len(), active);
    assert_eq!(compact.token_payload.len(), active * TOKEN_PAYLOAD_WIDTH);
    assert_eq!(compact.graph_token_offsets.len(), active + 1);
    assert_eq!(compact.graph_targets.len(), encoded.graph_targets.len());
    assert!(
        compact
            .graph_targets
            .iter()
            .all(|target| usize::from(*target) < active)
    );

    let mut compact_slot = 0;
    for (board, count) in counts.into_iter().enumerate() {
        for local in 0..count {
            let padded_slot = board * BOARD_TOKEN_CAPACITY + local;
            assert_eq!(
                compact.token_types[compact_slot],
                encoded.token_types[padded_slot]
            );
            assert_eq!(
                compact.token_seats[compact_slot],
                encoded.token_seats[padded_slot]
            );
            let compact_payload = &compact.token_payload
                [compact_slot * TOKEN_PAYLOAD_WIDTH..(compact_slot + 1) * TOKEN_PAYLOAD_WIDTH];
            let padded_payload = &encoded.token_payload
                [padded_slot * TOKEN_PAYLOAD_WIDTH..(padded_slot + 1) * TOKEN_PAYLOAD_WIDTH];
            assert_eq!(compact_payload, padded_payload);
            compact_slot += 1;
        }
    }
}

#[test]
fn mlx_encoding_rejects_optional_supplied_tile_extension() {
    let record = sample_record_after_one_turn();
    let supplied = "forest,river,0x1f,false".parse().unwrap();
    let state = SparsePublicState::from_position_record(&record, Some(supplied)).unwrap();
    assert!(encode_sparse_state(&state).is_err());
}
