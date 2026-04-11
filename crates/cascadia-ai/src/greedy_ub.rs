//! Greedy upper-bound oracle for Cascadia wildlife scoring (Bound 2).
//!
//! Cheap (~microsecond) upper bound on the wildlife score achievable from a
//! given state with R moves remaining. Useful for:
//!   - MCE candidate pruning: if `actual + ub_remaining(R-1) < best - margin`, prune
//!   - Per-game capture-rate diagnostics (loose but instant)
//!
//! Approach: enumerate (b, e, s, h, f) tuples summing to ≤ R cells, scoring
//! each with the optimal pattern partition for each species. Take the max.
//!
//! For R = 20 this is ~80K combinations, total ~10µs.
//!
//! Card A scoring tables verified against scoring/wildlife/*.rs.

/// Best salmon score for n salmon (multi-chain partition optimum).
/// dp[n] = max over partitions of n into chunks ≤7 of sum SALMON_RUN_SCORE.
const SALMON_BEST: [u16; 21] = [
    0,  // 0
    2,  // 1
    4,  // 2
    7,  // 3
    11, // 4
    15, // 5
    20, // 6
    26, // 7
    28, // 8 (7+1)
    30, // 9 (7+2)
    33, // 10 (7+3)
    37, // 11 (7+4)
    41, // 12 (7+5)
    46, // 13 (7+6)
    52, // 14 (7+7)
    54, // 15 (7+7+1)
    56, // 16
    59, // 17
    63, // 18
    67, // 19
    72, // 20 (7+7+6)
];

/// Best elk score for n cells using only lines of length 1-4.
/// Greedy: as many lines of 4 as possible, then leftover scores per Card A.
const ELK_BEST: [u16; 21] = [
    0,  // 0
    2,  // 1
    5,  // 2
    9,  // 3
    13, // 4 (line of 4)
    15, // 5 (4 + 1) = 13 + 2
    18, // 6 (4 + 2) = 13 + 5
    22, // 7 (4 + 3) = 13 + 9
    26, // 8 (4 + 4) = 26
    28, // 9 (4 + 4 + 1)
    31, // 10 (4 + 4 + 2)
    35, // 11 (4 + 4 + 3)
    39, // 12 (4 + 4 + 4)
    41, // 13 (4 + 4 + 4 + 1)
    44, // 14
    48, // 15
    52, // 16 (4 + 4 + 4 + 4)
    54, // 17
    57, // 18
    61, // 19
    65, // 20 (5 lines of 4)
];

const BEAR_SCORE: [u16; 5] = [0, 4, 11, 19, 27];
const HAWK_SCORE: [u16; 9] = [0, 2, 5, 8, 11, 14, 18, 22, 28];

#[inline]
fn bear_score(n_cells: usize) -> u16 {
    let pairs = (n_cells / 2).min(4);
    BEAR_SCORE[pairs]
}

#[inline]
fn salmon_score(n_cells: usize) -> u16 {
    SALMON_BEST[n_cells.min(20)]
}

#[inline]
fn elk_score(n_cells: usize) -> u16 {
    ELK_BEST[n_cells.min(20)]
}

#[inline]
fn hawk_score(n_cells: usize) -> u16 {
    HAWK_SCORE[n_cells.min(8)]
}

/// Tighter fox upper bound: each fox scores at most `num_distinct_species` points
/// where the count INCLUDES fox itself (if there's any fox). Without other species,
/// foxes only see foxes → each scores 1.
#[inline]
fn fox_score(n_cells: usize, num_other_species: usize) -> u16 {
    if n_cells == 0 { return 0; }
    let per_fox = (num_other_species + 1).min(5);  // +1 for fox itself
    (n_cells * per_fox) as u16
}

/// Compute the greedy upper bound on wildlife score for R remaining moves.
///
/// Enumerates all (b, e, s, h, f) with sum ≤ R, scoring each via the optimal
/// per-species partition. Returns the max.
///
/// Per-species caps:
///   bear: 8 cells (4 pairs)
///   elk: 20 cells
///   salmon: 9 cells (per user-specified realistic cap)
///   hawk: 8 cells (8 isolated)
///   fox: R cells (no cap)
///
/// Cost: ~80K iterations for R=20 ≈ 10µs.
pub fn greedy_upper_bound(moves_remaining: usize) -> u16 {
    let r = moves_remaining;
    let mut best: u16 = 0;
    for b in 0..=r.min(8) {
        for s in 0..=(r - b).min(9) {
            let bs_score = bear_score(b) + salmon_score(s);
            for h in 0..=(r - b - s).min(8) {
                let bsh_score = bs_score + hawk_score(h);
                let r_eh = r - b - s - h;
                for e in 0..=r_eh {
                    let f = r_eh - e;
                    let other_species = (b > 0) as usize + (s > 0) as usize
                        + (h > 0) as usize + (e > 0) as usize;
                    let score = bsh_score + elk_score(e) + fox_score(f, other_species);
                    if score > best {
                        best = score;
                    }
                }
            }
        }
    }
    best
}

/// Returns (b, e, s, h, f) cell counts that maximize the greedy UB given the
/// current state. Used as a v5 auxiliary supervision target — the network learns
/// to predict the optimal allocation rather than the AI's actual allocation.
///
/// The returned counts are TOTALS (current + new), so the network learns to
/// predict the END-of-game optimal layout from the CURRENT mid-game state.
pub fn greedy_upper_bound_argmax_from_state(
    current_bear: usize,
    current_elk: usize,
    current_salmon: usize,
    current_hawk: usize,
    current_fox: usize,
    moves_remaining: usize,
) -> (u8, u8, u8, u8, u8) {
    let r = moves_remaining;
    let mut best: u16 = 0;
    let mut best_alloc: (u8, u8, u8, u8, u8) = (
        current_bear as u8,
        current_elk as u8,
        current_salmon as u8,
        current_hawk as u8,
        current_fox as u8,
    );

    for db in 0..=r.min(8) {
        let new_b = current_bear + db;
        for ds in 0..=(r - db).min(9) {
            let new_s = current_salmon + ds;
            for dh in 0..=(r - db - ds).min(8) {
                let new_h = current_hawk + dh;
                let r_ef = r - db - ds - dh;
                for de in 0..=r_ef {
                    let df = r_ef - de;
                    let new_e = current_elk + de;
                    let new_f = current_fox + df;
                    let other_species = (new_b > 0) as usize + (new_e > 0) as usize
                        + (new_s > 0) as usize + (new_h > 0) as usize;
                    let new_score = bear_score(new_b)
                        + elk_score(new_e)
                        + salmon_score(new_s)
                        + hawk_score(new_h)
                        + fox_score(new_f, other_species);
                    if new_score > best {
                        best = new_score;
                        best_alloc = (new_b as u8, new_e as u8, new_s as u8, new_h as u8, new_f as u8);
                    }
                }
            }
        }
    }
    best_alloc
}

/// Compute the greedy UB given a CURRENT state's per-species cell counts and
/// remaining moves. Returns the maximum total achievable on top of the current
/// state's score.
///
/// Conservative: assumes each NEW cell can be allocated to any species, but
/// existing cells are FIXED. Doesn't account for synergies (e.g., extending an
/// existing chain). Used for fast diagnostics.
pub fn greedy_upper_bound_from_state(
    current_bear: usize,
    current_elk: usize,
    current_salmon: usize,
    current_hawk: usize,
    current_fox: usize,
    moves_remaining: usize,
) -> u16 {
    let r = moves_remaining;
    let mut best: u16 = 0;
    let cur_other = (current_bear > 0) as usize + (current_elk > 0) as usize
        + (current_salmon > 0) as usize + (current_hawk > 0) as usize;
    let cur_score = bear_score(current_bear)
        + elk_score(current_elk)
        + salmon_score(current_salmon)
        + hawk_score(current_hawk)
        + fox_score(current_fox, cur_other);

    for db in 0..=r.min(8) {
        let new_b = current_bear + db;
        for ds in 0..=(r - db).min(9) {
            let new_s = current_salmon + ds;
            for dh in 0..=(r - db - ds).min(8) {
                let new_h = current_hawk + dh;
                let r_ef = r - db - ds - dh;
                for de in 0..=r_ef {
                    let df = r_ef - de;
                    let new_e = current_elk + de;
                    let new_f = current_fox + df;
                    let other_species = (new_b > 0) as usize + (new_e > 0) as usize
                        + (new_s > 0) as usize + (new_h > 0) as usize;
                    let new_score = bear_score(new_b)
                        + elk_score(new_e)
                        + salmon_score(new_s)
                        + hawk_score(new_h)
                        + fox_score(new_f, other_species);
                    if new_score > best {
                        best = new_score;
                    }
                }
            }
        }
    }
    best.max(cur_score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_board_20_moves() {
        // Empty state, 20 moves available — what's the max possible?
        let ub = greedy_upper_bound(20);
        // ILP found 71-72 for the same problem, this should be ≥ 72 (looser is OK)
        assert!(ub >= 72, "expected ≥72, got {}", ub);
        // But also not absurdly loose
        assert!(ub <= 100, "loose bound shouldn't exceed 100, got {}", ub);
    }

    #[test]
    fn small_budget() {
        // 4 moves: best is 1 elk line of 4 (=13). With tight fox bound, 4 lone foxes = 4 only.
        let ub = greedy_upper_bound(4);
        assert!(ub >= 13, "expected ≥13 (1 elk line), got {}", ub);
    }

    #[test]
    fn zero_moves() {
        let ub = greedy_upper_bound(0);
        assert_eq!(ub, 0);
    }

    #[test]
    fn matches_expected_at_8_moves() {
        // 8 moves: best should be ≥28 (8 hawks isolated = HAWK_SCORE[8])
        // Or salmon-7 (26) + 1 fox-with-1-other-species (=2) = 28
        // Or 4 bear pairs (27) + nothing = 27
        let ub = greedy_upper_bound(8);
        assert!(ub >= 28, "expected ≥28 (8 hawks or salmon7+1), got {}", ub);
    }

    #[test]
    fn from_state_baseline() {
        // No prior state, 20 moves: same as greedy_upper_bound(20)
        let from_state = greedy_upper_bound_from_state(0, 0, 0, 0, 0, 20);
        let direct = greedy_upper_bound(20);
        assert_eq!(from_state, direct);
    }

    #[test]
    fn from_state_existing_bears() {
        // Existing 4 bear pairs (8 bears) — can't add more bears
        // Plus 12 free moves
        let with_state = greedy_upper_bound_from_state(8, 0, 0, 0, 0, 12);
        // Should include the 27 from the 4 existing bear pairs + best of 12 more cells
        assert!(with_state >= 27, "should include existing 27, got {}", with_state);
    }

    #[test]
    fn argmax_empty_board_20_moves() {
        let (b, e, s, h, f) = greedy_upper_bound_argmax_from_state(0, 0, 0, 0, 0, 20);
        // Sanity: counts must sum to ≤ 20
        let total = b as usize + e as usize + s as usize + h as usize + f as usize;
        assert!(total <= 20, "argmax sum {} > 20", total);
        // Empty board should pick a non-trivial allocation (likely heavy on bears + salmon)
        assert!(b + s > 0, "argmax should pick some bear or salmon, got ({},{},{},{},{})", b,e,s,h,f);
    }

    #[test]
    fn argmax_already_full_bears() {
        // 8 bears (4 pairs), 12 moves remaining → should not add more bears
        let (b, _, _, _, _) = greedy_upper_bound_argmax_from_state(8, 0, 0, 0, 0, 12);
        assert_eq!(b, 8, "should keep 8 bears, got {}", b);
    }
}
