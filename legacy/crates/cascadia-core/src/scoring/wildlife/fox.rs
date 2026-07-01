use crate::board::Board;
use crate::hex::{HexCoord, ADJACENCY};
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

/// Collect a per-wildlife-type count of non-fox wildlife tokens adjacent to `pos`.
/// Index is `Wildlife as usize`; the fox slot is always left at 0 (excluded).
#[inline]
fn adjacent_non_fox_counts(board: &Board, pos: u16) -> [u8; 5] {
    let adj = &*ADJACENCY;
    let mut counts = [0u8; 5];
    for nidx in adj.neighbors_of(pos as usize) {
        if let Some(w) = board.grid.get(nidx).placed_wildlife() {
            if w != Wildlife::Fox {
                counts[w as usize] += 1;
            }
        }
    }
    counts
}

/// Fox Card B: For each fox, count the number of unique non-fox wildlife
/// TYPES that appear ≥ 2 times in its 6 adjacent cells.
///
/// Scoring per fox:
///   0 pair-types = 0
///   1 pair-type  = 3
///   2 pair-types = 5
///   3 pair-types = 7
///   4+ pair-types = 7 (cap; physically limited to 3 since the fox has only 6 neighbors)
pub fn score_b(board: &Board) -> u16 {
    let positions = &board.wildlife_positions[Wildlife::Fox as usize];
    if positions.is_empty() {
        return 0;
    }

    let mut total = 0u16;
    for &pos in positions.iter() {
        let counts = adjacent_non_fox_counts(board, pos);
        let pair_types = counts.iter().filter(|&&c| c >= 2).count();
        total += match pair_types {
            0 => 0,
            1 => 3,
            2 => 5,
            _ => 7,
        };
    }
    total
}

/// Fox Card C: For each fox, score equals the largest count of a single non-fox
/// wildlife type in its 6 adjacent cells (0-6).
pub fn score_c(board: &Board) -> u16 {
    let positions = &board.wildlife_positions[Wildlife::Fox as usize];
    if positions.is_empty() {
        return 0;
    }

    let mut total = 0u16;
    for &pos in positions.iter() {
        let counts = adjacent_non_fox_counts(board, pos);
        let max = *counts.iter().max().unwrap_or(&0);
        total += max as u16;
    }
    total
}

/// Fox Card D: Score per PAIR of adjacent foxes. A pair's score is determined
/// by the number of unique non-fox wildlife types with ≥ 2 tokens across the
/// 8 cells surrounding the pair (= union of each fox's 6 neighbors, minus
/// the 2 fox cells themselves). Each fox is used in at most one pair;
/// choose the pairing that maximises total score.
///
/// Per-pair scoring:
///   0 pair-types = 0
///   1 = 5, 2 = 7, 3 = 9, 4 = 11 (cap at 4)
pub fn score_d(board: &Board) -> u16 {
    let positions = &board.wildlife_positions[Wildlife::Fox as usize];
    let n = positions.len();
    if n < 2 {
        return 0;
    }

    let adj = &*ADJACENCY;

    // Build list of fox-fox pairs that are adjacent, with per-pair weight.
    let mut edges: arrayvec::ArrayVec<(u8, u8, u16), 64> = arrayvec::ArrayVec::new();
    for i in 0..n {
        let pi = positions[i] as usize;
        let ci = HexCoord::from_index(pi);
        for j in (i + 1)..n {
            let pj = positions[j] as usize;
            let cj = HexCoord::from_index(pj);
            let dq = (ci.q - cj.q).abs();
            let dr = (ci.r - cj.r).abs();
            let ds = (ci.q + ci.r - cj.q - cj.r).abs();
            // Hex adjacency: max of axial distance components == 1
            let hex_dist = (dq.max(dr).max(ds)) as u16;
            if hex_dist != 1 {
                continue;
            }

            // Collect the union of non-fox neighbor cells from both foxes.
            let mut surround = [u16::MAX; 12];
            let mut surround_len = 0usize;
            for &center in &[pi, pj] {
                for nidx in adj.neighbors_of(center) {
                    if nidx == pi || nidx == pj {
                        continue;
                    }
                    // dedupe
                    if !surround[..surround_len].contains(&(nidx as u16)) {
                        surround[surround_len] = nidx as u16;
                        surround_len += 1;
                    }
                }
            }

            // Count non-fox wildlife types across surround. Other foxes (whether
            // part of yet another pair or unmatched singletons) are never counted.
            let mut counts = [0u8; 5];
            for &cell in &surround[..surround_len] {
                if let Some(w) = board.grid.get(cell as usize).placed_wildlife() {
                    if w == Wildlife::Fox {
                        continue;
                    }
                    counts[w as usize] += 1;
                }
            }
            let pair_types = counts.iter().filter(|&&c| c >= 2).count();
            let weight: u16 = match pair_types {
                0 => 0,
                1 => 5,
                2 => 7,
                3 => 9,
                _ => 11,
            };
            if weight > 0 {
                let _ = edges.try_push((i as u8, j as u8, weight));
            }
        }
    }

    super::matching::max_weight_matching(n, &edges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::hex::HexCoord;
    use crate::types::{Terrain, TileData, WildlifeMask};

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

    // ---- Fox B ----

    #[test]
    fn b_no_pairs() {
        let mut board = Board::new();
        place(&mut board, 0, 0, Wildlife::Fox);
        place(&mut board, 1, 0, Wildlife::Bear);
        place(&mut board, -1, 0, Wildlife::Elk);
        place(&mut board, 0, 1, Wildlife::Salmon);
        // 3 unique types but no pair → 0
        assert_eq!(score_b(&board), 0);
    }

    #[test]
    fn b_one_pair_type() {
        let mut board = Board::new();
        place(&mut board, 0, 0, Wildlife::Fox);
        place(&mut board, 1, 0, Wildlife::Bear);
        place(&mut board, -1, 0, Wildlife::Bear); // pair of bears
        assert_eq!(score_b(&board), 3);
    }

    #[test]
    fn b_three_pair_types() {
        let mut board = Board::new();
        // Fox surrounded by 6 neighbors: 2 bear, 2 elk, 2 salmon
        place(&mut board, 0, 0, Wildlife::Fox);
        place(&mut board, 1, 0, Wildlife::Bear);
        place(&mut board, -1, 0, Wildlife::Bear);
        place(&mut board, 0, 1, Wildlife::Elk);
        place(&mut board, 0, -1, Wildlife::Elk);
        place(&mut board, 1, -1, Wildlife::Salmon);
        place(&mut board, -1, 1, Wildlife::Salmon);
        assert_eq!(score_b(&board), 7);
    }

    #[test]
    fn b_excludes_fox_neighbors() {
        let mut board = Board::new();
        // Center fox with two adjacent foxes — those don't count as a pair.
        place(&mut board, 0, 0, Wildlife::Fox);
        place(&mut board, 1, 0, Wildlife::Fox);
        place(&mut board, -1, 0, Wildlife::Fox);
        // Center fox sees 0 non-fox pairs.
        // Each adjacent fox sees only the center fox → 0 non-fox pairs each.
        assert_eq!(score_b(&board), 0);
    }

    // ---- Fox C ----

    #[test]
    fn c_max_single_type() {
        let mut board = Board::new();
        place(&mut board, 0, 0, Wildlife::Fox);
        place(&mut board, 1, 0, Wildlife::Bear);
        place(&mut board, -1, 0, Wildlife::Bear);
        place(&mut board, 0, 1, Wildlife::Bear); // 3 bears
        place(&mut board, 0, -1, Wildlife::Elk);
        place(&mut board, 1, -1, Wildlife::Elk); // 2 elk
        assert_eq!(score_c(&board), 3); // max of 3 bear vs 2 elk
    }

    #[test]
    fn c_excludes_foxes() {
        let mut board = Board::new();
        place(&mut board, 0, 0, Wildlife::Fox);
        place(&mut board, 1, 0, Wildlife::Fox);
        place(&mut board, -1, 0, Wildlife::Bear);
        place(&mut board, 0, 1, Wildlife::Bear);
        // Center fox: 2 bears, 2 foxes (excluded). Score 2.
        // Adjacent fox: 1 bear, 1 fox. Score 1.
        assert_eq!(score_c(&board), 2 + 1);
    }

    // ---- Fox D ----

    #[test]
    fn d_no_adjacent_foxes() {
        let mut board = Board::new();
        place(&mut board, 0, 0, Wildlife::Fox);
        place(&mut board, 5, 0, Wildlife::Fox);
        assert_eq!(score_d(&board), 0);
    }

    #[test]
    fn d_one_pair_two_pair_types() {
        let mut board = Board::new();
        // Two adjacent foxes, surrounded by 2 bears and 2 elk in the union.
        place(&mut board, 0, 0, Wildlife::Fox);
        place(&mut board, 1, 0, Wildlife::Fox);
        // Around fox(0,0): NE=(1,-1), NW=(0,-1), W=(-1,0), SW=(-1,1), SE=(0,1)
        place(&mut board, -1, 0, Wildlife::Bear);
        place(&mut board, 0, -1, Wildlife::Bear);
        // Around fox(1,0): E=(2,0), NE=(2,-1), SE=(1,1)
        place(&mut board, 2, 0, Wildlife::Elk);
        place(&mut board, 2, -1, Wildlife::Elk);
        assert_eq!(score_d(&board), 7); // 2 pair-types
    }

    #[test]
    fn d_other_foxes_dont_count_as_animals() {
        let mut board = Board::new();
        // Fox pair at (0,0)-(1,0). Surrounding cells include 2 OTHER foxes.
        // Those other foxes must NOT count as a pair-type for this pair's score.
        place(&mut board, 0, 0, Wildlife::Fox);
        place(&mut board, 1, 0, Wildlife::Fox);
        place(&mut board, -1, 0, Wildlife::Fox); // adjacent to fox(0,0), not the pair
        place(&mut board, 2, 0, Wildlife::Fox); // adjacent to fox(1,0), not the pair
                                                // No non-fox animals around → 0 pair types → pair scores 0.
                                                // The two extra foxes also can't pair (each adj only to a paired fox; paired ones taken).
                                                // Actually max-matching may pair (0,0)-(-1,0) instead; same logic — no animals → 0.
        assert_eq!(score_d(&board), 0);
    }

    #[test]
    fn d_other_foxes_present_with_real_pair_type() {
        let mut board = Board::new();
        // Pair at (0,0)-(1,0) with 2 bears around AND extra non-paired foxes.
        place(&mut board, 0, 0, Wildlife::Fox);
        place(&mut board, 1, 0, Wildlife::Fox);
        place(&mut board, -1, 0, Wildlife::Bear);
        place(&mut board, 0, -1, Wildlife::Bear);
        // Extra fox far away (not adjacent to either pair member).
        place(&mut board, 5, 0, Wildlife::Fox);
        // 1 pair type (bears) → 5 pts. Extra fox doesn't affect score.
        assert_eq!(score_d(&board), 5);
    }

    #[test]
    fn d_chain_of_three_picks_one_pair() {
        let mut board = Board::new();
        // 3 foxes in a row; only one pair allowed (each fox in ≤1 pair).
        place(&mut board, 0, 0, Wildlife::Fox);
        place(&mut board, 1, 0, Wildlife::Fox);
        place(&mut board, 2, 0, Wildlife::Fox);
        place(&mut board, -1, 0, Wildlife::Bear);
        place(&mut board, 0, -1, Wildlife::Bear);
        // Two candidate pairs: (0,1) sees 2 bears → 5 pts; (1,2) sees nothing → 0.
        assert_eq!(score_d(&board), 5);
    }
}
