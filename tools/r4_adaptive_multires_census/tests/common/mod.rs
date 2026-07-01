#![allow(dead_code)]

use std::collections::BTreeSet;

use cascadia_data::{
    BOARD_ENTITY_SIZE, BOARD_SLOTS, MARKET_ENTITY_SIZE, MAX_BOARD_TILES, PositionRecord, TARGET_DIM,
};
use cascadia_game::{
    Board, GameConfig, GameSeed, GameState, HexCoord, Rotation, Terrain, Tile, TileId, Wildlife,
    WildlifeMask,
};

const NONE: u8 = u8::MAX;

#[derive(Debug, Clone)]
pub struct BoardSpec {
    pub coordinates: Vec<HexCoord>,
    pub tiles: Vec<Tile>,
}

impl BoardSpec {
    pub fn straight_single(terrain: Terrain, wildlife: Wildlife) -> Self {
        let coordinates = straight_coordinates();
        let tiles = coordinates
            .iter()
            .enumerate()
            .map(|(index, _)| single_tile(index, terrain, wildlife))
            .collect();
        Self { coordinates, tiles }
    }
}

pub fn sample_record_after_one_turn() -> PositionRecord {
    let mut game = GameState::new(
        GameConfig::research_aaaaa(4).unwrap(),
        GameSeed::from_u64(0x44_55_66),
    )
    .unwrap();
    let (prelude, staged) = game.preview_free_three_of_a_kind_if_feasible().unwrap();
    let action = staged
        .legal_turn_actions(&prelude)
        .unwrap()
        .into_iter()
        .find(|action| action.wildlife.is_some())
        .unwrap();
    game.apply(&action).unwrap();
    PositionRecord::observe(&game, 0x445566)
}

pub fn straight_coordinates() -> Vec<HexCoord> {
    (-11..=11).map(|q| HexCoord::new(q, 0)).collect()
}

pub fn single_tile(index: usize, terrain: Terrain, wildlife: Wildlife) -> Tile {
    Tile {
        id: TileId(index as u8),
        terrain_a: terrain,
        terrain_b: None,
        wildlife: WildlifeMask::one(wildlife),
        keystone: true,
    }
}

pub fn terminal_record(boards: Vec<BoardSpec>) -> PositionRecord {
    let player_count = boards.len() as u8;
    let total_turns = 20 * player_count;
    let mut record = PositionRecord {
        game_index: 0x7777,
        turn: total_turns,
        active_seat: 0,
        player_count,
        total_turns,
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
    for (seat, spec) in boards.into_iter().enumerate() {
        assert_eq!(spec.coordinates.len(), 23);
        assert_eq!(spec.tiles.len(), 23);
        assert_eq!(
            spec.coordinates
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len(),
            23
        );
        let mut board = Board::empty();
        for (coord, tile) in spec
            .coordinates
            .iter()
            .copied()
            .zip(spec.tiles.iter().copied())
        {
            board.place_tile(coord, tile, Rotation::ZERO).unwrap();
        }
        board.validate().unwrap();
        record.board_counts[seat] = 23;
        for terrain in Terrain::ALL {
            record.habitat_sizes[seat][terrain as usize] = board.largest_habitat(terrain);
        }
        let mut placed = board.placed_tiles().collect::<Vec<_>>();
        placed.sort_unstable_by_key(|(coord, _)| (coord.q, coord.r));
        for (row, (coord, tile)) in placed.into_iter().enumerate() {
            record.board_entities[seat][row] = [
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
    }
    record
}
