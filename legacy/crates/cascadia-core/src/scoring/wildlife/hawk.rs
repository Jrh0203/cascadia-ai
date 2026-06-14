use crate::board::Board;
use crate::hex::{HexCoord, ADJACENCY};
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
///   8+ isolated hawks = 26 points
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
        _ => 26,
    }
}

/// For each ordered pair (a, b) of hawks with `hawk_idx[a] < hawk_idx[b]`, invoke
/// `on_pair(idx_a, idx_b, adjacent, non_hawk_types_mask)` if they have a line
/// of sight to each other. LOS = same hex-axis, no OTHER hawk strictly between.
/// Non-hawk wildlife tokens in between are collected as a type bitmask.
fn for_each_los_pair<F>(positions: &[u16], board: &Board, mut on_pair: F)
where
    F: FnMut(u8, u8, bool, u8),
{
    let n = positions.len();
    if n < 2 {
        return;
    }

    let mut hawk_set = [false; 441];
    let mut pos_to_idx = [u8::MAX; 441];
    for (i, &p) in positions.iter().enumerate() {
        hawk_set[p as usize] = true;
        pos_to_idx[p as usize] = i as u8;
    }

    for (i, &start) in positions.iter().enumerate() {
        let coord = HexCoord::from_index(start as usize);
        for &(dq, dr) in &HexCoord::DIRECTIONS {
            let mut cur = HexCoord::new(coord.q + dq, coord.r + dr);
            let mut steps = 1u16;
            let mut types_mask = 0u8;
            loop {
                match cur.to_index() {
                    Some(idx) => {
                        if hawk_set[idx] {
                            // Found a hawk — LOS to it.
                            let j = pos_to_idx[idx];
                            if (i as u8) < j {
                                let adjacent = steps == 1;
                                on_pair(i as u8, j, adjacent, types_mask);
                            }
                            break;
                        }
                        if let Some(w) = board.grid.get(idx).placed_wildlife() {
                            // Hawks block LOS (handled above); anything else is recorded.
                            types_mask |= 1 << (w as u8);
                        }
                    }
                    None => break,
                }
                cur = HexCoord::new(cur.q + dq, cur.r + dr);
                steps += 1;
            }
        }
    }
}

/// Hawk Card B: Count hawks with LOS to at least one other NON-ADJACENT hawk.
/// (Adjacent hawks are excluded — only LOS pairs across a gap qualify.)
/// Scoring by count:
///   0 or 1 = 0
///   2 = 5, 3 = 9, 4 = 12, 5 = 16, 6 = 20, 7 = 24, 8+ = 28
pub fn score_b(board: &Board) -> u16 {
    let positions = &board.wildlife_positions[Wildlife::Hawk as usize];
    if positions.len() < 2 {
        return 0;
    }

    let mut seen = [false; 32];
    for_each_los_pair(positions, board, |i, j, adjacent, _mask| {
        if adjacent {
            return;
        }
        seen[i as usize] = true;
        seen[j as usize] = true;
    });

    let valid = seen.iter().filter(|&&s| s).count() as u16;
    match valid {
        0 | 1 => 0,
        2 => 5,
        3 => 9,
        4 => 12,
        5 => 16,
        6 => 20,
        7 => 24,
        _ => 28,
    }
}

/// Hawk Card C: 3 points per unique LOS pair between NON-adjacent hawks.
pub fn score_c(board: &Board) -> u16 {
    let positions = &board.wildlife_positions[Wildlife::Hawk as usize];
    if positions.len() < 2 {
        return 0;
    }

    let mut pairs = 0u16;
    for_each_los_pair(positions, board, |_i, _j, adjacent, _mask| {
        if !adjacent {
            pairs += 1;
        }
    });
    pairs * 3
}

/// Hawk Card D: Score per PAIR of hawks with LOS (non-adjacent), where the
/// pair's score is determined by the number of unique non-hawk wildlife TYPES
/// in the cells strictly between the two hawks.
///
/// Per-pair score: 0 types = 0, 1 = 4, 2 = 7, 3+ = 9.
///
/// Each hawk participates in at most one pair; choose the pairing that
/// maximises total points.
pub fn score_d(board: &Board) -> u16 {
    let positions = &board.wildlife_positions[Wildlife::Hawk as usize];
    let n = positions.len();
    if n < 2 {
        return 0;
    }

    let mut edges: arrayvec::ArrayVec<(u8, u8, u16), 64> = arrayvec::ArrayVec::new();
    for_each_los_pair(positions, board, |i, j, adjacent, mask| {
        if adjacent {
            return; // adjacent hawks have no intervening cells
        }
        // Mask records non-hawk wildlife types only.
        let unique = mask.count_ones();
        let weight: u16 = match unique {
            0 => 0,
            1 => 4,
            2 => 7,
            _ => 9,
        };
        if weight > 0 {
            let _ = edges.try_push((i, j, weight));
        }
    });

    super::matching::max_weight_matching(n, &edges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::hex::HexCoord;
    use crate::types::{Terrain, TileData, WildlifeMask};

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

    // ---- Hawk B ----

    #[test]
    fn b_two_hawks_with_los() {
        let mut board = Board::new();
        place_hawk(&mut board, 0, 0);
        place_hawk(&mut board, 3, 0); // collinear, 2 cells between
        assert_eq!(score_b(&board), 5);
    }

    #[test]
    fn b_two_hawks_off_axis() {
        let mut board = Board::new();
        place_hawk(&mut board, 0, 0);
        place_hawk(&mut board, 2, 1); // not on hex axis
        assert_eq!(score_b(&board), 0);
    }

    #[test]
    fn b_adjacent_hawks_dont_count() {
        let mut board = Board::new();
        place_hawk(&mut board, 0, 0);
        place_hawk(&mut board, 1, 0); // adjacent → doesn't count for B
        assert_eq!(score_b(&board), 0);
    }

    #[test]
    fn b_mixed_adjacent_and_los() {
        let mut board = Board::new();
        // (0,0)-(1,0) adjacent → neither counts for B from this pair alone.
        // Add (3,0): (1,0)-(3,0) is non-adjacent LOS, validates both (1,0) and (3,0).
        // (0,0)-(3,0) blocked by (1,0).
        // So 2 valid hawks → 5 pts.
        place_hawk(&mut board, 0, 0);
        place_hawk(&mut board, 1, 0);
        place_hawk(&mut board, 3, 0);
        assert_eq!(score_b(&board), 5);
    }

    #[test]
    fn b_blocked_by_third_hawk() {
        let mut board = Board::new();
        place_hawk(&mut board, 0, 0);
        place_hawk(&mut board, 2, 0); // blocker
        place_hawk(&mut board, 4, 0); // blocked from (0,0) but adjacent... actually (2,0)-(4,0) has LOS.
                                      // (0,0) sees (2,0) (adjacent? no, 2 apart, 1 cell between). LOS yes.
                                      // (2,0) sees (0,0) and (4,0) — both 2 apart, 1 cell between, LOS to both.
                                      // (4,0) sees (2,0). (4,0) cannot see (0,0) because (2,0) blocks.
                                      // All 3 are valid → 9 points.
        assert_eq!(score_b(&board), 9);
    }

    #[test]
    fn b_count_table() {
        // Build N hawks all in mutual LOS via 1-cell gaps along a line.
        // Use offsets centred on origin to fit within the 21×21 grid.
        // Spacing = 2 → q range = -(n-1) .. +(n-1) ≤ ±10 when n ≤ 6.
        // For n ≥ 7 we instead use a 7-direction layout (different axes).
        for (n, expected) in [(2u8, 5u16), (3, 9), (4, 12), (5, 16), (6, 20)] {
            let mut board = Board::new();
            let half = (n as i8 - 1);
            for k in 0..n {
                let q = (k as i8) * 2 - half;
                place_hawk(&mut board, q, 0);
            }
            assert_eq!(score_b(&board), expected, "n={}", n);
        }
    }

    #[test]
    fn b_eight_hawks_caps() {
        // Place 8 hawks across two parallel lines with 1-cell gaps:
        // line 1 at r=0:  q ∈ {-7,-5,-3,-1,1,3,5,7}  (8 hawks)
        let mut board = Board::new();
        for q in (-7..=7).step_by(2) {
            place_hawk(&mut board, q, 0);
        }
        assert_eq!(score_b(&board), 28); // 8 valid hawks → cap
    }

    // ---- Hawk C ----

    #[test]
    fn c_no_pairs() {
        let mut board = Board::new();
        place_hawk(&mut board, 0, 0);
        place_hawk(&mut board, 2, 1); // off-axis
        assert_eq!(score_c(&board), 0);
    }

    #[test]
    fn c_adjacent_doesnt_score() {
        let mut board = Board::new();
        place_hawk(&mut board, 0, 0);
        place_hawk(&mut board, 1, 0); // adjacent → not counted
        assert_eq!(score_c(&board), 0);
    }

    #[test]
    fn c_one_pair() {
        let mut board = Board::new();
        place_hawk(&mut board, 0, 0);
        place_hawk(&mut board, 3, 0);
        assert_eq!(score_c(&board), 3);
    }

    #[test]
    fn c_three_in_line() {
        let mut board = Board::new();
        // Hawks at (0,0), (2,0), (4,0). Pairs with LOS:
        //   (0,0)-(2,0) non-adjacent, LOS — count 1
        //   (2,0)-(4,0) non-adjacent, LOS — count 1
        //   (0,0)-(4,0) blocked by (2,0)
        place_hawk(&mut board, 0, 0);
        place_hawk(&mut board, 2, 0);
        place_hawk(&mut board, 4, 0);
        assert_eq!(score_c(&board), 6);
    }

    // ---- Hawk D ----

    #[test]
    fn d_no_pairs() {
        let mut board = Board::new();
        place_hawk(&mut board, 0, 0);
        place_hawk(&mut board, 5, 5); // off-axis
        assert_eq!(score_d(&board), 0);
    }

    #[test]
    fn d_pair_with_one_animal() {
        let mut board = Board::new();
        place_hawk(&mut board, 0, 0);
        place_hawk(&mut board, 3, 0);
        place_other(&mut board, 1, 0, Wildlife::Bear);
        // 1 unique non-hawk type between → 4 pts
        assert_eq!(score_d(&board), 4);
    }

    #[test]
    fn d_pair_with_two_types() {
        let mut board = Board::new();
        place_hawk(&mut board, 0, 0);
        place_hawk(&mut board, 4, 0); // 3 cells between
        place_other(&mut board, 1, 0, Wildlife::Bear);
        place_other(&mut board, 2, 0, Wildlife::Elk);
        place_other(&mut board, 3, 0, Wildlife::Bear); // duplicate type
        assert_eq!(score_d(&board), 7); // 2 unique types
    }

    #[test]
    fn d_pair_with_three_types_caps() {
        let mut board = Board::new();
        place_hawk(&mut board, 0, 0);
        place_hawk(&mut board, 5, 0);
        place_other(&mut board, 1, 0, Wildlife::Bear);
        place_other(&mut board, 2, 0, Wildlife::Elk);
        place_other(&mut board, 3, 0, Wildlife::Salmon);
        place_other(&mut board, 4, 0, Wildlife::Fox);
        // 4 unique types, capped → 9
        assert_eq!(score_d(&board), 9);
    }

    #[test]
    fn d_each_hawk_in_at_most_one_pair() {
        let mut board = Board::new();
        // 3 hawks in a line with animals between.
        // (0,0)-(2,0) has 1 bear between → 4 pts
        // (2,0)-(4,0) has 1 elk between → 4 pts
        // Both share hawk (2,0) → matching picks ONE pair = 4 pts.
        place_hawk(&mut board, 0, 0);
        place_hawk(&mut board, 2, 0);
        place_hawk(&mut board, 4, 0);
        place_other(&mut board, 1, 0, Wildlife::Bear);
        place_other(&mut board, 3, 0, Wildlife::Elk);
        assert_eq!(score_d(&board), 4);
    }

    #[test]
    fn d_two_disjoint_pairs() {
        let mut board = Board::new();
        // Pair 1 on the y-axis area, pair 2 far away
        place_hawk(&mut board, 0, 0);
        place_hawk(&mut board, 3, 0);
        place_other(&mut board, 1, 0, Wildlife::Bear);
        place_other(&mut board, 2, 0, Wildlife::Elk); // 2 types → 7 pts

        place_hawk(&mut board, -3, 5);
        place_hawk(&mut board, 0, 5);
        place_other(&mut board, -2, 5, Wildlife::Salmon); // 1 type → 4 pts
        assert_eq!(score_d(&board), 7 + 4);
    }
}
