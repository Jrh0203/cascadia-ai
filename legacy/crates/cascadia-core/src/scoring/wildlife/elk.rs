use crate::board::Board;
use crate::hex::{HexCoord, ADJACENCY};
use crate::types::Wildlife;

/// Elk Card A: Score based on straight lines of adjacent elk.
/// Each elk can only count for one line. We find the maximum-scoring
/// assignment of elk to non-overlapping lines.
///
/// Scoring per line:
///   1 elk = 2 points
///   2 elk = 5 points
///   3 elk = 9 points
///   4+ elk = 13 points
pub fn score_a(board: &Board) -> u16 {
    let positions = &board.wildlife_positions[Wildlife::Elk as usize];
    if positions.is_empty() {
        return 0;
    }

    let adj = &*ADJACENCY;

    // Find connected components of elk
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
                if !visited[nidx] && board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Elk) {
                    visited[nidx] = true;
                    queue.push(nidx as u16);
                }
            }
        }

        // For this component, find all maximal lines in each of 3 directions
        // and greedily assign elk to maximize score.
        total_score += score_component(&component, board);
    }

    total_score
}

fn score_component(component: &[u16], _board: &Board) -> u16 {
    if component.len() == 1 {
        return 2; // single elk = 2 points
    }

    // For small components, find all lines and pick best non-overlapping set.
    // A line is a maximal sequence of elk in one of the 3 hex line directions.

    let mut is_elk = [false; 441];
    for &pos in component {
        is_elk[pos as usize] = true;
    }

    // Find all maximal lines in all 3 directions
    let mut lines: arrayvec::ArrayVec<arrayvec::ArrayVec<u16, 24>, 48> = arrayvec::ArrayVec::new();
    let mut in_line = [[false; 441]; 3]; // track which elk have been used as line starts per direction

    for dir in 0..3 {
        let (dq, dr) = HexCoord::LINE_DIRECTIONS[dir];
        for &pos in component {
            let coord = HexCoord::from_index(pos as usize);
            if in_line[dir][pos as usize] {
                continue;
            }

            // Walk backwards to find the start of this line
            let mut start = coord;
            loop {
                let prev = HexCoord::new(start.q - dq, start.r - dr);
                if let Some(pidx) = prev.to_index() {
                    if is_elk[pidx] {
                        start = prev;
                        continue;
                    }
                }
                break;
            }

            // Walk forward to build the line
            let mut line = arrayvec::ArrayVec::<u16, 24>::new();
            let mut current = start;
            loop {
                if let Some(cidx) = current.to_index() {
                    if is_elk[cidx] {
                        line.push(cidx as u16);
                        in_line[dir][cidx] = true;
                        current = HexCoord::new(current.q + dq, current.r + dr);
                        continue;
                    }
                }
                break;
            }

            if !line.is_empty() && !lines.is_full() {
                lines.push(line);
            }
        }
    }

    // Greedy approach: sort lines by length (longest first), assign elk greedily
    // For small components this is near-optimal.
    let mut line_lengths: arrayvec::ArrayVec<(usize, usize), 48> = arrayvec::ArrayVec::new();
    for (i, line) in lines.iter().enumerate() {
        line_lengths.push((line.len(), i));
    }
    line_lengths.sort_unstable_by(|a, b| b.0.cmp(&a.0));

    let mut used = [false; 441];
    let mut score = 0u16;

    for &(_, li) in &line_lengths {
        let line = &lines[li];
        // Count how many unused elk are in this line
        let available: u16 = line.iter().filter(|&&p| !used[p as usize]).count() as u16;
        if available == 0 {
            continue;
        }

        // Find the longest contiguous run of unused elk in this line
        let mut best_run = 0u16;
        let mut current_run = 0u16;
        for &p in line {
            if !used[p as usize] {
                current_run += 1;
                if current_run > best_run {
                    best_run = current_run;
                }
            } else {
                current_run = 0;
            }
        }

        if best_run > 0 {
            // Mark elk in the best run as used
            // Find the start of the best run
            current_run = 0;
            let mut run_start = 0;
            let mut best_start = 0;
            let mut found_best = 0;
            for (i, &p) in line.iter().enumerate() {
                if !used[p as usize] {
                    if current_run == 0 {
                        run_start = i;
                    }
                    current_run += 1;
                    if current_run > found_best {
                        found_best = current_run;
                        best_start = run_start;
                    }
                } else {
                    current_run = 0;
                }
            }

            for i in best_start..(best_start + best_run as usize) {
                used[line[i] as usize] = true;
            }

            score += line_score(best_run);
        }
    }

    // Any remaining unused elk score as singles
    for &pos in component {
        if !used[pos as usize] {
            score += 2; // single elk
        }
    }

    score
}

fn line_score(length: u16) -> u16 {
    match length {
        0 => 0,
        1 => 2,
        2 => 5,
        3 => 9,
        _ => 13,
    }
}

// =================================================================
// Shared helpers for elk variants B/C/D.
// =================================================================

#[inline]
fn hex_adjacent(a: u16, b: u16) -> bool {
    let ca = HexCoord::from_index(a as usize);
    let cb = HexCoord::from_index(b as usize);
    let dq = (ca.q as i16 - cb.q as i16).abs();
    let dr = (ca.r as i16 - cb.r as i16).abs();
    let ds = ((ca.q as i16 + ca.r as i16) - (cb.q as i16 + cb.r as i16)).abs();
    dq.max(dr).max(ds) == 1
}

/// Find connected components of elk on the board. Returns each component as a
/// vector of grid indices.
fn elk_components(board: &Board) -> arrayvec::ArrayVec<arrayvec::ArrayVec<u16, 24>, 24> {
    let positions = &board.wildlife_positions[Wildlife::Elk as usize];
    let adj = &*ADJACENCY;
    let mut visited = [false; 441];
    let mut components: arrayvec::ArrayVec<arrayvec::ArrayVec<u16, 24>, 24> =
        arrayvec::ArrayVec::new();

    for &pos in positions.iter() {
        let idx = pos as usize;
        if visited[idx] {
            continue;
        }
        let mut comp = arrayvec::ArrayVec::<u16, 24>::new();
        let mut queue = arrayvec::ArrayVec::<u16, 24>::new();
        queue.push(pos);
        visited[idx] = true;
        while let Some(c) = queue.pop() {
            let _ = comp.try_push(c);
            for n in adj.neighbors_of(c as usize) {
                if !visited[n] && board.grid.get(n).placed_wildlife() == Some(Wildlife::Elk) {
                    visited[n] = true;
                    let _ = queue.try_push(n as u16);
                }
            }
        }
        let _ = components.try_push(comp);
    }
    components
}

/// Build local adjacency bitmask between elk indices in `component`.
fn local_adjacency(component: &[u16]) -> [u32; 24] {
    let n = component.len();
    let mut adj_local = [0u32; 24];
    for i in 0..n {
        for j in (i + 1)..n {
            if hex_adjacent(component[i], component[j]) {
                adj_local[i] |= 1u32 << j;
                adj_local[j] |= 1u32 << i;
            }
        }
    }
    adj_local
}

/// Test if a bitmask of elk forms a connected subgraph in `adj_local`.
fn is_connected_subset(mask: u32, adj_local: &[u32; 24]) -> bool {
    if mask == 0 {
        return true;
    }
    let start = mask.trailing_zeros() as usize;
    let mut visited = 1u32 << start;
    let mut stack = arrayvec::ArrayVec::<u8, 24>::new();
    let _ = stack.try_push(start as u8);
    while let Some(v) = stack.pop() {
        let mut nbrs = adj_local[v as usize] & mask & !visited;
        while nbrs != 0 {
            let u = nbrs.trailing_zeros();
            nbrs &= nbrs - 1;
            visited |= 1u32 << u;
            let _ = stack.try_push(u as u8);
        }
    }
    visited == mask
}

// =================================================================
// Elk Card B: shape-based scoring.
//
// Valid shapes (each elk in at most one shape; partition to maximise):
//   single elk         = 2
//   any 2 adjacent     = 5
//   triangle of 3      = 9   (3 mutually-adjacent elk forming a triad)
//   triangle + 1 adj   = 13  (cluster of 4 where 3 form a triangle and the
//                             4th is hex-adjacent to ≥ 1 of the triangle)
// Other configurations (e.g. line of 3, line of 4) score by best partition
// into the above shapes.
// =================================================================
pub fn score_b(board: &Board) -> u16 {
    let mut total = 0u16;
    for comp in elk_components(board).iter() {
        total += score_b_component(comp);
    }
    total
}

fn score_b_component(component: &[u16]) -> u16 {
    let n = component.len();
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 2;
    }
    let adj_local = local_adjacency(component);
    let size = 1usize << n;
    let mut dp = vec![0u16; size];

    for state in 1..size {
        let i = (state as u32).trailing_zeros() as usize;
        let bit_i = 1usize << i;
        let without_i = state & !bit_i;

        // Option A: i alone.
        let mut best = 2 + dp[without_i];

        // Option B: i + j (pair).
        let adj_to_i = adj_local[i] as usize & state;
        // Iterate j adjacent to i in `state` (all j > i since i is lowest bit).
        let mut tj = adj_to_i;
        while tj != 0 {
            let j = tj.trailing_zeros() as usize;
            tj &= tj - 1;
            let bit_j = 1usize << j;
            let after_pair = state & !bit_i & !bit_j;
            let val = 5 + dp[after_pair];
            if val > best {
                best = val;
            }

            // Option C: triangle {i, j, k}. k must be adjacent to BOTH i and j.
            let common_ijk = (adj_local[i] & adj_local[j]) as usize & state & !bit_i & !bit_j;
            let mut tk = common_ijk;
            while tk != 0 {
                let k = tk.trailing_zeros() as usize;
                tk &= tk - 1;
                let bit_k = 1usize << k;
                let mask3 = bit_i | bit_j | bit_k;
                let after3 = state & !mask3;
                let val3 = 9 + dp[after3];
                if val3 > best {
                    best = val3;
                }

                // Option D: triangle + 4th (l) adjacent to any of {i, j, k}.
                let triangle_nbrs = (adj_local[i] | adj_local[j] | adj_local[k]) as usize;
                let mut tl = triangle_nbrs & state & !mask3;
                while tl != 0 {
                    let l = tl.trailing_zeros() as usize;
                    tl &= tl - 1;
                    let mask4 = mask3 | (1usize << l);
                    let after4 = state & !mask4;
                    let val4 = 13 + dp[after4];
                    if val4 > best {
                        best = val4;
                    }
                }
            }
        }

        dp[state] = best;
    }
    dp[size - 1]
}

// =================================================================
// Elk Card C: any contiguous group; score by size.
//   1=2, 2=4, 3=7, 4=10, 5=14, 6=18, 7=23, 8+=28 (cap)
// =================================================================
pub fn score_c(board: &Board) -> u16 {
    let mut total = 0u16;
    for comp in elk_components(board).iter() {
        total += score_c_component(comp);
    }
    total
}

#[inline]
fn group_score_c(size: usize) -> u16 {
    match size {
        0 => 0,
        1 => 2,
        2 => 4,
        3 => 7,
        4 => 10,
        5 => 14,
        6 => 18,
        7 => 23,
        _ => 28,
    }
}

fn score_c_component(component: &[u16]) -> u16 {
    let n = component.len();
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 2;
    }
    let adj_local = local_adjacency(component);
    let size = 1usize << n;
    let mut dp = vec![0u16; size];

    for state in 1..size {
        let i = (state as u32).trailing_zeros() as usize;
        let bit_i = 1usize << i;
        let other_bits = (state ^ bit_i) as u32;

        // Enumerate all subsets of state containing i; check connectivity.
        let mut best = 0u16;
        let mut s = other_bits;
        loop {
            let candidate = bit_i as u32 | s;
            if is_connected_subset(candidate, &adj_local) {
                let cnt = candidate.count_ones() as usize;
                let score = group_score_c(cnt);
                let remaining = state & !(candidate as usize);
                let val = score + dp[remaining];
                if val > best {
                    best = val;
                }
            }
            if s == 0 {
                break;
            }
            s = s.wrapping_sub(1) & other_bits;
        }
        dp[state] = best;
    }
    dp[size - 1]
}

// =================================================================
// Elk Card D: ring formations.
// Each elk is "in" at most one ring centred on a hex point that has elk in
// some/all of its 6 adjacent cells. Per ring score: 1=2, 2=5, 3=8, 4=12,
// 5=16, 6=21. Maximise total. Rings can span otherwise-disconnected components
// (a center hex between two non-adjacent elk produces a 2-elk ring).
// =================================================================
pub fn score_d(board: &Board) -> u16 {
    let positions = &board.wildlife_positions[Wildlife::Elk as usize];
    let n = positions.len();
    if n == 0 {
        return 0;
    }
    if n > 24 {
        // Pathological — should never occur. Fall back to ring_score(1) per elk.
        return n as u16 * 2;
    }
    let adj = &*ADJACENCY;

    // Map grid index → elk index in `positions`.
    let mut elk_idx_at_pos = [u8::MAX; 441];
    for (i, &p) in positions.iter().enumerate() {
        elk_idx_at_pos[p as usize] = i as u8;
    }

    // Enumerate every distinct ring (= elk bitmask) reachable from some center
    // hex. Two centers that produce the same elk mask share a single entry.
    let mut rings: arrayvec::ArrayVec<u32, 64> = arrayvec::ArrayVec::new();
    let mut seen_center = [false; 441];
    for &p in positions.iter() {
        for c in adj.neighbors_of(p as usize) {
            if seen_center[c] {
                continue;
            }
            seen_center[c] = true;
            let mut mask = 0u32;
            for nbr in adj.neighbors_of(c) {
                let ei = elk_idx_at_pos[nbr];
                if ei != u8::MAX {
                    mask |= 1u32 << ei;
                }
            }
            if mask != 0 && !rings.contains(&mask) {
                let _ = rings.try_push(mask);
            }
        }
    }

    // For each elk, list of ring masks containing it.
    let mut rings_per_elk: arrayvec::ArrayVec<arrayvec::ArrayVec<u32, 16>, 24> =
        arrayvec::ArrayVec::new();
    for _ in 0..n {
        let _ = rings_per_elk.try_push(arrayvec::ArrayVec::new());
    }
    for &ring in rings.iter() {
        let mut tmp = ring;
        while tmp != 0 {
            let i = tmp.trailing_zeros() as usize;
            tmp &= tmp - 1;
            let _ = rings_per_elk[i].try_push(ring);
        }
    }

    // DP over bitmask of unassigned elk.
    let size = 1usize << n;
    let mut dp = vec![0u16; size];
    for state in 1..size {
        let i = (state as u32).trailing_zeros() as usize;
        let mut best = 0u16;
        for &ring in rings_per_elk[i].iter() {
            let claim = ring & state as u32; // elk in this ring still available
                                             // i is guaranteed in claim because rings_per_elk[i] only stores rings
                                             // containing i, and i is in state.
            let count = claim.count_ones() as usize;
            let score = ring_score_d(count);
            let remaining = state & !(claim as usize);
            let val = score + dp[remaining];
            if val > best {
                best = val;
            }
        }
        dp[state] = best;
    }
    dp[size - 1]
}

#[inline]
fn ring_score_d(size: usize) -> u16 {
    match size {
        0 => 0,
        1 => 2,
        2 => 5,
        3 => 8,
        4 => 12,
        5 => 16,
        _ => 21, // 6 (or above; physically capped at 6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::hex::HexCoord;
    use crate::types::{Terrain, TileData, WildlifeMask};

    fn elk_tile() -> TileData {
        TileData::dual(
            Terrain::Prairie,
            Terrain::Forest,
            WildlifeMask::new(&[Wildlife::Elk]),
        )
    }

    fn place_elk(board: &mut Board, q: i8, r: i8) {
        board.place_tile(HexCoord::new(q, r), elk_tile(), 0);
        let idx = HexCoord::new(q, r).to_index().unwrap();
        board.place_wildlife(idx, Wildlife::Elk);
    }

    #[test]
    fn no_elk() {
        let board = Board::new();
        assert_eq!(score_a(&board), 0);
    }

    #[test]
    fn single_elk() {
        let mut board = Board::new();
        place_elk(&mut board, 0, 0);
        assert_eq!(score_a(&board), 2);
    }

    #[test]
    fn line_of_two() {
        let mut board = Board::new();
        place_elk(&mut board, 0, 0);
        place_elk(&mut board, 1, 0); // E direction
        assert_eq!(score_a(&board), 5);
    }

    #[test]
    fn line_of_three() {
        let mut board = Board::new();
        place_elk(&mut board, 0, 0);
        place_elk(&mut board, 1, 0);
        place_elk(&mut board, 2, 0);
        assert_eq!(score_a(&board), 9);
    }

    #[test]
    fn line_of_four() {
        let mut board = Board::new();
        for i in 0..4 {
            place_elk(&mut board, i, 0);
        }
        assert_eq!(score_a(&board), 13);
    }

    // ---- Elk B ----

    #[test]
    fn b_single_elk() {
        let mut board = Board::new();
        place_elk(&mut board, 0, 0);
        assert_eq!(score_b(&board), 2);
    }

    #[test]
    fn b_pair() {
        let mut board = Board::new();
        place_elk(&mut board, 0, 0);
        place_elk(&mut board, 1, 0);
        assert_eq!(score_b(&board), 5);
    }

    #[test]
    fn b_triangle_scores_nine() {
        let mut board = Board::new();
        // Triad: (0,0), (1,0), (0,1). All mutually adjacent.
        place_elk(&mut board, 0, 0);
        place_elk(&mut board, 1, 0);
        place_elk(&mut board, 0, 1);
        assert_eq!(score_b(&board), 9);
    }

    #[test]
    fn b_line_of_three_partitions() {
        let mut board = Board::new();
        // Line of 3 (no triangle): (0,0), (1,0), (2,0). Best partition:
        // pair {(0,0),(1,0)} = 5, single {(2,0)} = 2. Total 7.
        // Alt: 3 singles = 6.  Take max = 7.
        place_elk(&mut board, 0, 0);
        place_elk(&mut board, 1, 0);
        place_elk(&mut board, 2, 0);
        assert_eq!(score_b(&board), 7);
    }

    #[test]
    fn b_rhombus_scores_thirteen() {
        let mut board = Board::new();
        // Triangle (0,0), (1,0), (0,1) + 4th adjacent to ≥1 of triangle.
        // (1,1) is adjacent to (1,0) and (0,1).
        place_elk(&mut board, 0, 0);
        place_elk(&mut board, 1, 0);
        place_elk(&mut board, 0, 1);
        place_elk(&mut board, 1, 1);
        assert_eq!(score_b(&board), 13);
    }

    #[test]
    fn b_line_of_four_partitions() {
        let mut board = Board::new();
        // Line of 4: best = pair + pair = 5 + 5 = 10.
        for i in 0..4 {
            place_elk(&mut board, i, 0);
        }
        assert_eq!(score_b(&board), 10);
    }

    // ---- Elk C ----

    #[test]
    fn c_table_by_size() {
        // Build a connected blob of size N (use a "snake" so we get 1 component).
        for (n, expected) in [(1u8, 2u16), (2, 4), (3, 7), (4, 10)] {
            let mut board = Board::new();
            for i in 0..n {
                place_elk(&mut board, i as i8, 0);
            }
            assert_eq!(score_c(&board), expected, "n={}", n);
        }
    }

    #[test]
    fn c_super_linear_keeps_blob_intact() {
        let mut board = Board::new();
        // Triangle of 3: per table, 3 elk = 7 vs partition (pair + single = 4+2=6 or 3 singles=6).
        // Triangle is connected → can be one group → 7.
        place_elk(&mut board, 0, 0);
        place_elk(&mut board, 1, 0);
        place_elk(&mut board, 0, 1);
        assert_eq!(score_c(&board), 7);
    }

    #[test]
    fn c_two_disconnected_components() {
        let mut board = Board::new();
        // pair + triangle, far apart
        place_elk(&mut board, 0, 0);
        place_elk(&mut board, 1, 0);
        place_elk(&mut board, 5, 5);
        place_elk(&mut board, 6, 5);
        place_elk(&mut board, 5, 6);
        assert_eq!(score_c(&board), 4 + 7);
    }

    // ---- Elk D ----

    #[test]
    fn d_single_elk_makes_ring_of_one() {
        let mut board = Board::new();
        place_elk(&mut board, 0, 0);
        // Any of the 6 candidate centers gives a ring of 1 → 2 pts.
        assert_eq!(score_d(&board), 2);
    }

    #[test]
    fn d_two_elk_around_shared_center() {
        let mut board = Board::new();
        // Place 2 elk that share a common neighbor center.
        // (0,0) and (2,0) share (1,0) as common neighbor.
        place_elk(&mut board, 0, 0);
        place_elk(&mut board, 2, 0);
        // Best ring: center (1,0) with both elk → 5 pts.
        assert_eq!(score_d(&board), 5);
    }

    #[test]
    fn d_six_elk_around_one_center() {
        let mut board = Board::new();
        // Surround (0,0) with all 6 neighbors.
        // Use a nearby cell that's not (0,0) — e.g. (5,5) so (0,0) stays empty.
        // Place elk at the 6 neighbors of (5,5).
        for &(dq, dr) in &HexCoord::DIRECTIONS {
            place_elk(&mut board, 5 + dq, 5 + dr);
        }
        // 6-elk ring around (5,5) → 21 pts.
        assert_eq!(score_d(&board), 21);
    }

    #[test]
    fn d_disjoint_components_can_share_ring() {
        let mut board = Board::new();
        // Two non-adjacent elk that share a center.
        // (0,0) and (2,0) — center (1,0) is between them.
        place_elk(&mut board, 0, 0);
        place_elk(&mut board, 2, 0);
        assert_eq!(score_d(&board), 5);
    }

    #[test]
    fn d_three_elk_picks_best_partition() {
        let mut board = Board::new();
        // Elk at (0,0), (2,0), (4,0). Possible rings:
        //   center (1,0): {(0,0), (2,0)} → 5
        //   center (3,0): {(2,0), (4,0)} → 5
        //   plus singletons.
        // Both pair-rings share (2,0), so we can only have one of them.
        // Best: 5 + 2 (single ring for the leftover) = 7.
        place_elk(&mut board, 0, 0);
        place_elk(&mut board, 2, 0);
        place_elk(&mut board, 4, 0);
        assert_eq!(score_d(&board), 7);
    }
}
