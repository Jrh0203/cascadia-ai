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
                if !visited[nidx]
                    && board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Elk)
                {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::hex::HexCoord;
    use crate::types::{TileData, Terrain, WildlifeMask};

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
}
