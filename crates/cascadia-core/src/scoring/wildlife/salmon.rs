use crate::board::Board;
use crate::hex::ADJACENCY;
use crate::types::Wildlife;

/// Salmon Card A: Score based on runs of salmon.
/// A "run" is a connected group where each salmon has at most 2 salmon neighbors
/// (i.e., the group forms a simple path or cycle, not a branching structure).
/// Runs must not be adjacent to other salmon outside the run.
///
/// Scoring per run:
///   1 salmon = 2 points
///   2 salmon = 4 points
///   3 salmon = 7 points
///   4 salmon = 11 points
///   5 salmon = 15 points
///   6 salmon = 20 points
///   7+ salmon = 26 points
pub fn score_a(board: &Board) -> u16 {
    let positions = &board.wildlife_positions[Wildlife::Salmon as usize];
    if positions.is_empty() {
        return 0;
    }

    let adj = &*ADJACENCY;

    // Find connected components
    let mut visited = [false; 441];
    let mut total_score = 0u16;

    for &pos in positions.iter() {
        let idx = pos as usize;
        if visited[idx] {
            continue;
        }

        // BFS to find component
        let mut component = arrayvec::ArrayVec::<u16, 24>::new();
        let mut queue = arrayvec::ArrayVec::<u16, 24>::new();
        queue.push(pos);
        visited[idx] = true;

        while let Some(current) = queue.pop() {
            component.push(current);
            for nidx in adj.neighbors_of(current as usize) {
                if !visited[nidx]
                    && board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Salmon)
                {
                    visited[nidx] = true;
                    queue.push(nidx as u16);
                }
            }
        }

        // Check if this is a valid run:
        // Each salmon in the component must have at most 2 salmon neighbors
        let is_valid_run = component.iter().all(|&pos| {
            let salmon_neighbors = adj
                .neighbors_of(pos as usize)
                .filter(|&nidx| board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Salmon))
                .count();
            salmon_neighbors <= 2
        });

        if is_valid_run {
            total_score += run_score(component.len() as u16);
        }
        // Invalid runs (branching) score 0
    }

    total_score
}

fn run_score(length: u16) -> u16 {
    match length {
        0 => 0,
        1 => 2,
        2 => 4,
        3 => 7,
        4 => 11,
        5 => 15,
        6 => 20,
        _ => 26,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::hex::HexCoord;
    use crate::types::{TileData, Terrain, WildlifeMask};

    fn salmon_tile() -> TileData {
        TileData::dual(
            Terrain::River,
            Terrain::Wetland,
            WildlifeMask::new(&[Wildlife::Salmon]),
        )
    }

    fn place_salmon(board: &mut Board, q: i8, r: i8) {
        board.place_tile(HexCoord::new(q, r), salmon_tile(), 0);
        let idx = HexCoord::new(q, r).to_index().unwrap();
        board.place_wildlife(idx, Wildlife::Salmon);
    }

    #[test]
    fn no_salmon() {
        let board = Board::new();
        assert_eq!(score_a(&board), 0);
    }

    #[test]
    fn single_salmon() {
        let mut board = Board::new();
        place_salmon(&mut board, 0, 0);
        assert_eq!(score_a(&board), 2);
    }

    #[test]
    fn run_of_three() {
        let mut board = Board::new();
        place_salmon(&mut board, 0, 0);
        place_salmon(&mut board, 1, 0);
        place_salmon(&mut board, 2, 0);
        assert_eq!(score_a(&board), 7);
    }

    #[test]
    fn branching_scores_zero() {
        let mut board = Board::new();
        // Create a Y-shape: center connected to 3 neighbors
        place_salmon(&mut board, 0, 0);
        place_salmon(&mut board, 1, 0);   // E
        place_salmon(&mut board, -1, 0);  // W
        place_salmon(&mut board, 0, -1);  // NW
        // (0,0) has 3 salmon neighbors -> invalid run
        assert_eq!(score_a(&board), 0);
    }

    #[test]
    fn two_separate_runs() {
        let mut board = Board::new();
        // Run 1: length 2
        place_salmon(&mut board, 0, 0);
        place_salmon(&mut board, 1, 0);
        // Run 2: length 3 (far away)
        place_salmon(&mut board, 5, 0);
        place_salmon(&mut board, 6, 0);
        place_salmon(&mut board, 7, 0);
        assert_eq!(score_a(&board), 4 + 7); // 11
    }
}
