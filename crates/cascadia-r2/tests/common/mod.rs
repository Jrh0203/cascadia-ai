use cascadia_data::{
    BOARD_ENTITY_SIZE, BOARD_SLOTS, MARKET_ENTITY_SIZE, MAX_BOARD_TILES, PositionRecord, TARGET_DIM,
};
use cascadia_game::{
    Board, GameConfig, GameSeed, GameState, HexCoord, Rotation, STANDARD_TILES, Terrain, Wildlife,
};

const NONE: u8 = u8::MAX;

pub fn sample_record_after_one_turn() -> PositionRecord {
    let mut game = GameState::new(
        GameConfig::research_aaaaa(4).unwrap(),
        GameSeed::from_u64(0x5eed),
    )
    .unwrap();
    let (prelude, staged) = game.preview_free_three_of_a_kind_if_feasible().unwrap();
    let actions = staged.legal_turn_actions(&prelude).unwrap();
    let action = actions
        .iter()
        .find(|action| action.wildlife.is_some())
        .or_else(|| actions.first())
        .unwrap()
        .clone();
    game.apply(&action).unwrap();
    PositionRecord::observe(&game, 17)
}

#[allow(dead_code)]
pub fn elongated_terminal_record() -> PositionRecord {
    elongated_terminal_record_from(0)
}

#[allow(dead_code)]
pub fn boundary_terminal_record() -> PositionRecord {
    elongated_terminal_record_from(2)
}

fn elongated_terminal_record_from(start_q: i8) -> PositionRecord {
    let mut board = Board::empty();
    for (index, tile) in STANDARD_TILES.iter().copied().take(23).enumerate() {
        board
            .place_tile(
                HexCoord::new(start_q + index as i8, 0),
                tile,
                Rotation::ZERO,
            )
            .unwrap();
    }
    let mut record = PositionRecord {
        game_index: 23,
        turn: 20,
        active_seat: 0,
        player_count: 1,
        total_turns: 20,
        board_counts: [0; BOARD_SLOTS],
        nature_tokens: [0; BOARD_SLOTS],
        scoring_cards: [0; 5],
        habitat_bonuses: false,
        wildlife_counts: [[0; 5]; BOARD_SLOTS],
        habitat_sizes: [[0; 5]; BOARD_SLOTS],
        board_entities: [[[NONE; BOARD_ENTITY_SIZE]; MAX_BOARD_TILES]; BOARD_SLOTS],
        market_entities: [[NONE; MARKET_ENTITY_SIZE]; 4],
        targets: [0; TARGET_DIM],
    };
    record.board_counts[0] = 23;
    for terrain in Terrain::ALL {
        record.habitat_sizes[0][terrain as usize] = board.largest_habitat(terrain);
    }
    let mut placed = board.placed_tiles().collect::<Vec<_>>();
    placed.sort_unstable_by_key(|(coord, _)| (coord.q, coord.r));
    for (row, (coord, tile)) in placed.into_iter().enumerate() {
        record.board_entities[0][row] = [
            coord.q as u8,
            coord.r as u8,
            tile.tile.terrain_a as u8,
            tile.tile.terrain_b.map_or(NONE, |terrain| terrain as u8),
            tile.rotation.get(),
            tile.tile.wildlife.bits(),
            tile.wildlife.map_or(NONE, |wildlife| wildlife as u8),
            u8::from(tile.tile.keystone),
        ];
    }
    record
}

#[allow(dead_code)]
pub fn placed_wildlife_count(record: &PositionRecord) -> usize {
    record.wildlife_counts[..usize::from(record.player_count)]
        .iter()
        .flatten()
        .map(|count| usize::from(*count))
        .sum()
}

#[allow(dead_code)]
pub fn wildlife_code(wildlife: Wildlife) -> u8 {
    wildlife as u8
}
