use crate::board::Board;
use crate::hex::ADJACENCY;
use crate::types::Wildlife;

/// Bear Card A: Score based on number of bear pairs.
/// A "pair" is exactly 2 bears adjacent to each other, with no other bears
/// adjacent to either bear in the pair.
///
/// Scoring table:
///   1 pair  = 4 points
///   2 pairs = 11 points
///   3 pairs = 19 points
///   4+ pairs = 27 points
pub fn score_a(board: &Board) -> u16 {
    let positions = &board.wildlife_positions[Wildlife::Bear as usize];
    if positions.len() < 2 {
        return 0;
    }

    let adj = &*ADJACENCY;

    // Build connected components of bears using simple BFS on small set
    let mut visited = [false; 441]; // reuse fixed array, small enough
    let mut pairs = 0u16;

    for &pos in positions.iter() {
        let idx = pos as usize;
        if visited[idx] {
            continue;
        }

        // BFS to find this component
        let mut component = arrayvec::ArrayVec::<u16, 24>::new();
        let mut queue = arrayvec::ArrayVec::<u16, 24>::new();
        queue.push(pos);
        visited[idx] = true;

        while let Some(current) = queue.pop() {
            component.push(current);
            for nidx in adj.neighbors_of(current as usize) {
                if !visited[nidx]
                    && board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Bear)
                {
                    visited[nidx] = true;
                    queue.push(nidx as u16);
                }
            }
        }

        // A pair is a component of exactly size 2
        if component.len() == 2 {
            pairs += 1;
        }
    }

    match pairs {
        0 => 0,
        1 => 4,
        2 => 11,
        3 => 19,
        _ => 27,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::hex::HexCoord;
    use crate::types::{TileData, Terrain, WildlifeMask};

    fn bear_tile() -> TileData {
        TileData::single(Terrain::Forest, WildlifeMask::new(&[Wildlife::Bear]))
    }

    fn place_bear(board: &mut Board, q: i8, r: i8) {
        board.place_tile(HexCoord::new(q, r), bear_tile(), 0);
        let idx = HexCoord::new(q, r).to_index().unwrap();
        board.place_wildlife(idx, Wildlife::Bear);
    }

    #[test]
    fn no_bears() {
        let board = Board::new();
        assert_eq!(score_a(&board), 0);
    }

    #[test]
    fn single_bear() {
        let mut board = Board::new();
        place_bear(&mut board, 0, 0);
        assert_eq!(score_a(&board), 0);
    }

    #[test]
    fn one_pair() {
        let mut board = Board::new();
        place_bear(&mut board, 0, 0);
        place_bear(&mut board, 1, 0);
        assert_eq!(score_a(&board), 4);
    }

    #[test]
    fn three_bears_no_pair() {
        let mut board = Board::new();
        place_bear(&mut board, 0, 0);
        place_bear(&mut board, 1, 0);
        place_bear(&mut board, 0, 1); // adjacent to (0,0), making a group of 3
        assert_eq!(score_a(&board), 0); // group of 3, not a pair
    }

    #[test]
    fn two_pairs() {
        let mut board = Board::new();
        // Pair 1
        place_bear(&mut board, 0, 0);
        place_bear(&mut board, 1, 0);
        // Pair 2 (far away)
        place_bear(&mut board, 5, 0);
        place_bear(&mut board, 6, 0);
        assert_eq!(score_a(&board), 11);
    }
}
