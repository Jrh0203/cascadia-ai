use crate::board::Board;
use crate::hex::ADJACENCY;
use crate::types::Wildlife;

/// Hawk Card A: Score based on number of hawks not adjacent to any other hawk.
///
/// Scoring:
///   1 isolated hawk = 2 points
///   2 isolated hawks = 5 points
///   3 isolated hawks = 8 points
///   4 isolated hawks = 11 points
///   5 isolated hawks = 14 points
///   6 isolated hawks = 18 points
///   7 isolated hawks = 22 points
///   8+ isolated hawks = 28 points
pub fn score_a(board: &Board) -> u16 {
    let positions = &board.wildlife_positions[Wildlife::Hawk as usize];
    if positions.is_empty() {
        return 0;
    }

    let adj = &*ADJACENCY;
    let mut isolated_count = 0u16;

    for &pos in positions.iter() {
        let has_adjacent_hawk = adj
            .neighbors_of(pos as usize)
            .any(|nidx| board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Hawk));

        if !has_adjacent_hawk {
            isolated_count += 1;
        }
    }

    match isolated_count {
        0 => 0,
        1 => 2,
        2 => 5,
        3 => 8,
        4 => 11,
        5 => 14,
        6 => 18,
        7 => 22,
        _ => 28,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::hex::HexCoord;
    use crate::types::{TileData, Terrain, WildlifeMask};

    fn hawk_tile() -> TileData {
        TileData::dual(
            Terrain::Mountain,
            Terrain::Prairie,
            WildlifeMask::new(&[Wildlife::Hawk]),
        )
    }

    fn place_hawk(board: &mut Board, q: i8, r: i8) {
        board.place_tile(HexCoord::new(q, r), hawk_tile(), 0);
        let idx = HexCoord::new(q, r).to_index().unwrap();
        board.place_wildlife(idx, Wildlife::Hawk);
    }

    #[test]
    fn no_hawks() {
        let board = Board::new();
        assert_eq!(score_a(&board), 0);
    }

    #[test]
    fn one_isolated_hawk() {
        let mut board = Board::new();
        place_hawk(&mut board, 0, 0);
        assert_eq!(score_a(&board), 2);
    }

    #[test]
    fn two_isolated_hawks() {
        let mut board = Board::new();
        place_hawk(&mut board, 0, 0);
        place_hawk(&mut board, 5, 0); // far apart
        assert_eq!(score_a(&board), 5);
    }

    #[test]
    fn adjacent_hawks_dont_score() {
        let mut board = Board::new();
        place_hawk(&mut board, 0, 0);
        place_hawk(&mut board, 1, 0); // adjacent
        assert_eq!(score_a(&board), 0);
    }

    #[test]
    fn mix_isolated_and_adjacent() {
        let mut board = Board::new();
        place_hawk(&mut board, 0, 0);
        place_hawk(&mut board, 1, 0); // adjacent to (0,0)
        place_hawk(&mut board, 5, 0); // isolated
        assert_eq!(score_a(&board), 2); // only 1 isolated
    }
}
