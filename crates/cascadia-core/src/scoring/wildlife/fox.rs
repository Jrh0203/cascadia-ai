use crate::board::Board;
use crate::hex::ADJACENCY;
use crate::types::Wildlife;

/// Fox Card A: Each fox scores based on the number of unique wildlife types
/// adjacent to it (including other foxes).
///
/// Scoring per fox:
///   0 unique types = 0 points
///   1 unique type  = 1 point
///   2 unique types = 2 points
///   3 unique types = 3 points
///   4 unique types = 4 points
///   5 unique types = 5 points
pub fn score_a(board: &Board) -> u16 {
    let positions = &board.wildlife_positions[Wildlife::Fox as usize];
    if positions.is_empty() {
        return 0;
    }

    let adj = &*ADJACENCY;
    let mut total = 0u16;

    for &pos in positions.iter() {
        let mut seen_mask = 0u8; // 5 bits, one per wildlife type

        for nidx in adj.neighbors_of(pos as usize) {
            if let Some(w) = board.grid.get(nidx).placed_wildlife() {
                seen_mask |= 1 << (w as u8);
            }
        }

        total += seen_mask.count_ones() as u16;
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::hex::HexCoord;
    use crate::types::{TileData, Terrain, WildlifeMask};

    fn any_tile() -> TileData {
        TileData::dual(
            Terrain::Forest,
            Terrain::Prairie,
            WildlifeMask::new(&Wildlife::ALL),
        )
    }

    fn place(board: &mut Board, q: i8, r: i8, w: Wildlife) {
        board.place_tile(HexCoord::new(q, r), any_tile(), 0);
        let idx = HexCoord::new(q, r).to_index().unwrap();
        board.place_wildlife(idx, w);
    }

    #[test]
    fn no_foxes() {
        let board = Board::new();
        assert_eq!(score_a(&board), 0);
    }

    #[test]
    fn fox_with_no_neighbors() {
        let mut board = Board::new();
        place(&mut board, 0, 0, Wildlife::Fox);
        assert_eq!(score_a(&board), 0);
    }

    #[test]
    fn fox_with_three_unique_neighbors() {
        let mut board = Board::new();
        place(&mut board, 0, 0, Wildlife::Fox);
        place(&mut board, 1, 0, Wildlife::Bear);
        place(&mut board, -1, 0, Wildlife::Elk);
        place(&mut board, 0, 1, Wildlife::Salmon);
        assert_eq!(score_a(&board), 3);
    }

    #[test]
    fn fox_with_duplicate_neighbors() {
        let mut board = Board::new();
        place(&mut board, 0, 0, Wildlife::Fox);
        place(&mut board, 1, 0, Wildlife::Bear);
        place(&mut board, -1, 0, Wildlife::Bear); // duplicate type
        assert_eq!(score_a(&board), 1); // only 1 unique type
    }

    #[test]
    fn two_foxes() {
        let mut board = Board::new();
        place(&mut board, 0, 0, Wildlife::Fox);
        place(&mut board, 1, 0, Wildlife::Bear);
        // Second fox far away
        place(&mut board, 5, 0, Wildlife::Fox);
        place(&mut board, 6, 0, Wildlife::Hawk);
        place(&mut board, 4, 0, Wildlife::Salmon);
        assert_eq!(score_a(&board), 1 + 2); // fox1=1 unique, fox2=2 unique
    }
}
