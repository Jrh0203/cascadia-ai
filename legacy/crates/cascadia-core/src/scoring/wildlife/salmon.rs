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
///   2 salmon = 5 points
///   3 salmon = 8 points
///   4 salmon = 12 points
///   5 salmon = 16 points
///   6 salmon = 20 points
///   7+ salmon = 25 points
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
        2 => 5,
        3 => 8,
        4 => 12,
        5 => 16,
        6 => 20,
        _ => 25,
    }
}

/// Walk salmon connected components and invoke `on_run` for each VALID run
/// (component where every salmon has ≤ 2 salmon neighbors). Invalid (branching)
/// components are silently skipped — they score 0 on every variant.
fn for_each_valid_run<F>(board: &Board, mut on_run: F)
where
    F: FnMut(&[u16]),
{
    let positions = &board.wildlife_positions[Wildlife::Salmon as usize];
    if positions.is_empty() {
        return;
    }
    let adj = &*ADJACENCY;
    let mut visited = [false; 441];

    for &pos in positions.iter() {
        let idx = pos as usize;
        if visited[idx] {
            continue;
        }

        let mut component = arrayvec::ArrayVec::<u16, 32>::new();
        let mut queue = arrayvec::ArrayVec::<u16, 32>::new();
        queue.push(pos);
        visited[idx] = true;

        while let Some(current) = queue.pop() {
            let _ = component.try_push(current);
            for nidx in adj.neighbors_of(current as usize) {
                if !visited[nidx]
                    && board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Salmon)
                {
                    visited[nidx] = true;
                    let _ = queue.try_push(nidx as u16);
                }
            }
        }

        let valid = component.iter().all(|&p| {
            adj.neighbors_of(p as usize)
                .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon))
                .count()
                <= 2
        });
        if valid {
            on_run(&component);
        }
    }
}

/// Salmon Card B: shorter scoring table, capped at length 5.
///   1=2, 2=4, 3=9, 4=11, 5+=17
pub fn score_b(board: &Board) -> u16 {
    let mut total = 0u16;
    for_each_valid_run(board, |run| {
        total += match run.len() {
            0 => 0,
            1 => 2,
            2 => 4,
            3 => 9,
            4 => 11,
            _ => 17,
        };
    });
    total
}

/// Salmon Card C: minimum run size 3.
///   1-2 = 0, 3=10, 4=12, 5+=15
pub fn score_c(board: &Board) -> u16 {
    let mut total = 0u16;
    for_each_valid_run(board, |run| {
        total += match run.len() {
            0 | 1 | 2 => 0,
            3 => 10,
            4 => 12,
            _ => 15,
        };
    });
    total
}

/// Salmon Card D: 1 point per salmon in the run + 1 point per UNIQUE non-salmon
/// wildlife token adjacent to the run (each adjacent token counted once even if
/// it borders multiple salmon in the run). Minimum run size = 3; runs of length
/// 1 or 2 score 0.
pub fn score_d(board: &Board) -> u16 {
    let adj = &*ADJACENCY;
    let mut total = 0u16;
    for_each_valid_run(board, |run| {
        if run.len() < 3 {
            return;
        }
        // Per-salmon point.
        total += run.len() as u16;

        // Collect unique adjacent non-salmon tokens (by cell index).
        let mut seen = [false; 441];
        let mut bonus = 0u16;
        for &p in run {
            for nidx in adj.neighbors_of(p as usize) {
                if seen[nidx] {
                    continue;
                }
                seen[nidx] = true;
                if let Some(w) = board.grid.get(nidx).placed_wildlife() {
                    if w != Wildlife::Salmon {
                        bonus += 1;
                    }
                }
            }
        }
        total += bonus;
    });
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::hex::HexCoord;
    use crate::types::{Terrain, TileData, WildlifeMask};

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
        assert_eq!(score_a(&board), 8);
    }

    #[test]
    fn branching_scores_zero() {
        let mut board = Board::new();
        // Create a Y-shape: center connected to 3 neighbors
        place_salmon(&mut board, 0, 0);
        place_salmon(&mut board, 1, 0); // E
        place_salmon(&mut board, -1, 0); // W
        place_salmon(&mut board, 0, -1); // NW
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
        assert_eq!(score_a(&board), 5 + 8); // 13
    }

    fn place_other(board: &mut Board, q: i8, r: i8, w: Wildlife) {
        let tile = TileData::dual(
            Terrain::Forest,
            Terrain::Mountain,
            WildlifeMask::new(&Wildlife::ALL),
        );
        board.place_tile(HexCoord::new(q, r), tile, 0);
        let idx = HexCoord::new(q, r).to_index().unwrap();
        board.place_wildlife(idx, w);
    }

    // ---- Salmon B ----

    #[test]
    fn b_single_salmon() {
        let mut board = Board::new();
        place_salmon(&mut board, 0, 0);
        assert_eq!(score_b(&board), 2);
    }

    #[test]
    fn b_run_table() {
        for (len, expected) in [(1u8, 2u16), (2, 4), (3, 9), (4, 11), (5, 17), (6, 17)] {
            let mut board = Board::new();
            for i in 0..len {
                place_salmon(&mut board, i as i8, 0);
            }
            assert_eq!(
                score_b(&board),
                expected,
                "len {} should score {}",
                len,
                expected
            );
        }
    }

    #[test]
    fn b_branching_scores_zero() {
        let mut board = Board::new();
        place_salmon(&mut board, 0, 0);
        place_salmon(&mut board, 1, 0);
        place_salmon(&mut board, -1, 0);
        place_salmon(&mut board, 0, -1);
        assert_eq!(score_b(&board), 0);
    }

    // ---- Salmon C ----

    #[test]
    fn c_short_runs_dont_score() {
        let mut board = Board::new();
        place_salmon(&mut board, 0, 0);
        place_salmon(&mut board, 1, 0); // length 2
        assert_eq!(score_c(&board), 0);
    }

    #[test]
    fn c_run_table() {
        for (len, expected) in [(1u8, 0u16), (2, 0), (3, 10), (4, 12), (5, 15), (6, 15)] {
            let mut board = Board::new();
            for i in 0..len {
                place_salmon(&mut board, i as i8, 0);
            }
            assert_eq!(
                score_c(&board),
                expected,
                "len {} should score {}",
                len,
                expected
            );
        }
    }

    // ---- Salmon D ----

    #[test]
    fn d_run_with_no_adjacent_animals() {
        let mut board = Board::new();
        place_salmon(&mut board, 0, 0);
        place_salmon(&mut board, 1, 0);
        place_salmon(&mut board, 2, 0);
        // 3 salmon, no adjacent non-salmon
        assert_eq!(score_d(&board), 3);
    }

    #[test]
    fn d_run_with_adjacent_animals() {
        let mut board = Board::new();
        place_salmon(&mut board, 0, 0);
        place_salmon(&mut board, 1, 0);
        place_salmon(&mut board, 2, 0);
        // Adjacent to the run: place a bear next to (0,0) and an elk next to (2,0)
        place_other(&mut board, -1, 0, Wildlife::Bear);
        place_other(&mut board, 3, 0, Wildlife::Elk);
        // 3 salmon + 2 unique adjacent animals = 5
        assert_eq!(score_d(&board), 5);
    }

    #[test]
    fn d_animal_adjacent_to_two_salmon_counted_once() {
        let mut board = Board::new();
        // Need ≥ 3 salmon for D to score at all; use a length-3 run.
        place_salmon(&mut board, 0, 0);
        place_salmon(&mut board, 1, 0);
        place_salmon(&mut board, 2, 0);
        // (1,-1) is adjacent to BOTH (0,0) and (1,0). Adjacent to only one of those is
        // already common; we want the same token adjacent to multiple salmon → counted once.
        place_other(&mut board, 1, -1, Wildlife::Bear); // adjacent to (0,0) and (1,0)
                                                        // 3 salmon + 1 unique adjacent animal = 4
        assert_eq!(score_d(&board), 4);
    }

    #[test]
    fn d_short_runs_dont_score() {
        let mut board = Board::new();
        place_salmon(&mut board, 0, 0);
        place_salmon(&mut board, 1, 0); // length 2
        place_other(&mut board, 0, 1, Wildlife::Bear);
        // Even with adjacent animals, length < 3 → 0.
        assert_eq!(score_d(&board), 0);
    }

    #[test]
    fn d_single_salmon_doesnt_score() {
        let mut board = Board::new();
        place_salmon(&mut board, 0, 0);
        place_other(&mut board, 1, 0, Wildlife::Bear);
        place_other(&mut board, -1, 0, Wildlife::Elk);
        assert_eq!(score_d(&board), 0);
    }

    #[test]
    fn d_branching_scores_zero() {
        let mut board = Board::new();
        place_salmon(&mut board, 0, 0);
        place_salmon(&mut board, 1, 0);
        place_salmon(&mut board, -1, 0);
        place_salmon(&mut board, 0, -1);
        place_other(&mut board, 5, 0, Wildlife::Bear);
        // Even with adjacent bears, invalid run scores 0.
        assert_eq!(score_d(&board), 0);
    }
}
