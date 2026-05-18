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

/// Compute connected components of bears, returning the size of each component.
fn bear_component_sizes(board: &Board) -> arrayvec::ArrayVec<u16, 32> {
    let positions = &board.wildlife_positions[Wildlife::Bear as usize];
    let mut sizes = arrayvec::ArrayVec::<u16, 32>::new();
    if positions.is_empty() {
        return sizes;
    }

    let adj = &*ADJACENCY;
    let mut visited = [false; 441];

    for &pos in positions.iter() {
        let idx = pos as usize;
        if visited[idx] {
            continue;
        }
        let mut size = 0u16;
        let mut queue = arrayvec::ArrayVec::<u16, 32>::new();
        queue.push(pos);
        visited[idx] = true;

        while let Some(current) = queue.pop() {
            size += 1;
            for nidx in adj.neighbors_of(current as usize) {
                if !visited[nidx]
                    && board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Bear)
                {
                    visited[nidx] = true;
                    let _ = queue.try_push(nidx as u16);
                }
            }
        }
        let _ = sizes.try_push(size);
    }

    sizes
}

/// Bear Card B: Score 10 points per group of EXACTLY 3 bears.
/// Groups of any other size score 0. Standard bear "no touching"
/// rule means each connected component is one group.
pub fn score_b(board: &Board) -> u16 {
    let sizes = bear_component_sizes(board);
    sizes.iter().filter(|&&s| s == 3).count() as u16 * 10
}

/// Bear Card C: Score by group size:
///   1 bear  = 2 points
///   2 bears = 5 points
///   3 bears = 8 points
/// Groups of size 4+ score 0.
/// Bonus: +3 points if you have at least one group of each of sizes 1, 2, and 3.
pub fn score_c(board: &Board) -> u16 {
    let sizes = bear_component_sizes(board);
    let mut total = 0u16;
    let mut has1 = false;
    let mut has2 = false;
    let mut has3 = false;
    for &s in sizes.iter() {
        match s {
            1 => { total += 2; has1 = true; }
            2 => { total += 5; has2 = true; }
            3 => { total += 8; has3 = true; }
            _ => {}
        }
    }
    if has1 && has2 && has3 {
        total += 3;
    }
    total
}

/// Bear Card D: Score by group size:
///   2 bears = 5 points
///   3 bears = 8 points
///   4 bears = 14 points
/// Groups of size 1 or 5+ score 0.
pub fn score_d(board: &Board) -> u16 {
    let sizes = bear_component_sizes(board);
    sizes.iter().map(|&s| match s {
        2 => 5,
        3 => 8,
        4 => 14,
        _ => 0,
    }).sum()
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

    // ---- Bear B ----

    #[test]
    fn b_no_groups_of_three() {
        let mut board = Board::new();
        place_bear(&mut board, 0, 0);
        place_bear(&mut board, 1, 0); // pair
        assert_eq!(score_b(&board), 0);
    }

    #[test]
    fn b_one_group_of_three() {
        let mut board = Board::new();
        // Triangle of 3
        place_bear(&mut board, 0, 0);
        place_bear(&mut board, 1, 0);
        place_bear(&mut board, 0, 1);
        assert_eq!(score_b(&board), 10);
    }

    #[test]
    fn b_group_of_four_doesnt_score() {
        let mut board = Board::new();
        place_bear(&mut board, 0, 0);
        place_bear(&mut board, 1, 0);
        place_bear(&mut board, 0, 1);
        place_bear(&mut board, 1, 1); // 4-blob
        assert_eq!(score_b(&board), 0);
    }

    #[test]
    fn b_two_groups_of_three() {
        let mut board = Board::new();
        // Triangle 1
        place_bear(&mut board, 0, 0);
        place_bear(&mut board, 1, 0);
        place_bear(&mut board, 0, 1);
        // Triangle 2 (far)
        place_bear(&mut board, 5, 5);
        place_bear(&mut board, 6, 5);
        place_bear(&mut board, 5, 6);
        assert_eq!(score_b(&board), 20);
    }

    // ---- Bear C ----

    #[test]
    fn c_singletons_only() {
        let mut board = Board::new();
        place_bear(&mut board, 0, 0);
        place_bear(&mut board, 5, 0);
        assert_eq!(score_c(&board), 4); // 2 + 2
    }

    #[test]
    fn c_one_of_each_with_bonus() {
        let mut board = Board::new();
        // singleton
        place_bear(&mut board, 0, 0);
        // pair
        place_bear(&mut board, 5, 0);
        place_bear(&mut board, 6, 0);
        // triangle
        place_bear(&mut board, -5, 0);
        place_bear(&mut board, -6, 0);
        place_bear(&mut board, -5, 1);
        assert_eq!(score_c(&board), 2 + 5 + 8 + 3); // 18
    }

    #[test]
    fn c_no_bonus_when_one_size_missing() {
        let mut board = Board::new();
        // singleton + triangle, no pair
        place_bear(&mut board, 0, 0);
        place_bear(&mut board, -5, 0);
        place_bear(&mut board, -6, 0);
        place_bear(&mut board, -5, 1);
        assert_eq!(score_c(&board), 2 + 8); // no +3 bonus
    }

    #[test]
    fn c_group_of_four_doesnt_score() {
        let mut board = Board::new();
        place_bear(&mut board, 0, 0);
        place_bear(&mut board, 1, 0);
        place_bear(&mut board, 0, 1);
        place_bear(&mut board, 1, 1);
        assert_eq!(score_c(&board), 0);
    }

    // ---- Bear D ----

    #[test]
    fn d_pair_and_triangle() {
        let mut board = Board::new();
        // pair
        place_bear(&mut board, 0, 0);
        place_bear(&mut board, 1, 0);
        // triangle
        place_bear(&mut board, 5, 0);
        place_bear(&mut board, 6, 0);
        place_bear(&mut board, 5, 1);
        assert_eq!(score_d(&board), 5 + 8);
    }

    #[test]
    fn d_group_of_four() {
        let mut board = Board::new();
        place_bear(&mut board, 0, 0);
        place_bear(&mut board, 1, 0);
        place_bear(&mut board, 0, 1);
        place_bear(&mut board, 1, 1);
        assert_eq!(score_d(&board), 14);
    }

    #[test]
    fn d_singletons_dont_score() {
        let mut board = Board::new();
        place_bear(&mut board, 0, 0);
        place_bear(&mut board, 5, 0);
        assert_eq!(score_d(&board), 0);
    }
}
