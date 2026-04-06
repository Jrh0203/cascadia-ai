use cascadia_core::board::Board;
use cascadia_core::hex::{HexCoord, ADJACENCY};
use cascadia_core::types::{ScoringCards, Wildlife};

/// Compute a potential bonus estimating future scoring opportunities.
/// Returns a value in "fractional points" (scaled by 10 to avoid floats).
/// Uses actual marginal values from Card A scoring tables.
pub fn board_potential(board: &Board, _cards: &ScoringCards) -> i32 {
    let mut potential: i32 = 0;

    potential += habitat_potential(board);
    potential += bear_potential(board);
    potential += elk_potential(board);
    potential += salmon_potential(board);
    potential += hawk_potential(board);
    potential += fox_potential(board);
    potential += empty_slot_potential(board);

    potential
}

/// Habitat potential: bonus for frontier cells adjacent to same-terrain tiles,
/// especially when they could bridge separate groups.
fn habitat_potential(board: &Board) -> i32 {
    let adj = &*ADJACENCY;
    let frontier = board.frontier();
    let mut potential: i32 = 0;

    for &fi in frontier.iter() {
        let idx = fi as usize;
        // For each terrain, count how many neighboring tiles have this terrain
        for ti in 0..5 {
            let mut terrain_neighbor_count = 0u8;
            for nidx in adj.neighbors_of(idx) {
                let cell = board.grid.get(nidx);
                let has_terrain = cell.primary_terrain().map_or(false, |t| t as usize == ti)
                    || cell.secondary_terrain().map_or(false, |t| t as usize == ti);
                if has_terrain {
                    terrain_neighbor_count += 1;
                }
            }
            // Multiple same-terrain neighbors suggests a bridging opportunity
            if terrain_neighbor_count >= 2 {
                potential += 15; // bridging opportunity = ~1.5 habitat points
            } else if terrain_neighbor_count == 1 {
                potential += 2; // simple growth
            }
        }
    }

    potential
}

/// Bear potential (Card A): reward setups toward completing bear pairs.
/// Uses actual marginal values: 0→1 pair = +4, 1→2 = +7, 2→3 = +8, 3→4+ = +8.
fn bear_potential(board: &Board) -> i32 {
    let positions = &board.wildlife_positions[Wildlife::Bear as usize];
    let adj = &*ADJACENCY;

    // Count current pairs using the same algorithm as scoring
    let current_pairs = count_bear_pairs(board);

    // Marginal value of next pair (in tenths of a point)
    let next_pair_marginal: i32 = match current_pairs {
        0 => 40,  // +4 points
        1 => 70,  // +7 points
        2 => 80,  // +8 points
        _ => 80,  // +8 points
    };

    let mut half_pairs = 0i32;
    let mut penalty = 0i32;

    for &pos in positions.iter() {
        let idx = pos as usize;
        let bear_neighbors: usize = adj
            .neighbors_of(idx)
            .filter(|&nidx| board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Bear))
            .count();

        if bear_neighbors == 0 {
            let bear_slots: usize = adj.neighbors_of(idx)
                .filter(|&nidx| {
                    let cell = board.grid.get(nidx);
                    cell.is_present() && cell.can_place_wildlife(Wildlife::Bear)
                })
                .count();
            if bear_slots >= 1 {
                half_pairs += 1;
            }
        }
        // Bears in groups of 3+ broke a potential pair
        if bear_neighbors >= 2 {
            penalty -= 20;
        }
    }

    let completable_pairs = (half_pairs / 2).min(2);
    completable_pairs * next_pair_marginal / 2 + penalty
}

fn count_bear_pairs(board: &Board) -> u16 {
    let positions = &board.wildlife_positions[Wildlife::Bear as usize];
    let adj = &*ADJACENCY;
    let mut visited = [false; 441];
    let mut pairs = 0u16;

    for &pos in positions.iter() {
        let idx = pos as usize;
        if visited[idx] { continue; }

        let mut size = 0u16;
        let mut queue = arrayvec::ArrayVec::<u16, 24>::new();
        queue.push(pos);
        visited[idx] = true;

        while let Some(current) = queue.pop() {
            size += 1;
            for nidx in adj.neighbors_of(current as usize) {
                if !visited[nidx]
                    && board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Bear)
                {
                    visited[nidx] = true;
                    queue.push(nidx as u16);
                }
            }
        }

        if size == 2 { pairs += 1; }
    }
    pairs
}

/// Elk potential (Card A): reward extendable lines based on actual marginals.
/// Line of 1→2: +3pts, 2→3: +4pts, 3→4+: +4pts. Lines of 4+ have no extension value.
fn elk_potential(board: &Board) -> i32 {
    let positions = &board.wildlife_positions[Wildlife::Elk as usize];
    if positions.is_empty() { return 0; }
    let adj = &*ADJACENCY;
    let mut potential: i32 = 0;

    // Find line lengths for each elk by checking the 3 hex directions
    for &pos in positions.iter() {
        let coord = HexCoord::from_index(pos as usize);

        for &(dq, dr) in &HexCoord::LINE_DIRECTIONS {
            // Measure line length in the forward direction from this elk
            let fwd = HexCoord::new(coord.q + dq, coord.r + dr);
            let has_fwd_elk = fwd.to_index().map_or(false, |idx| {
                board.grid.get(idx).placed_wildlife() == Some(Wildlife::Elk)
            });

            if !has_fwd_elk { continue; }

            // Count the full line length through this elk in this direction
            let mut line_len = 1u16;
            // Forward
            let mut c = HexCoord::new(coord.q + dq, coord.r + dr);
            while let Some(idx) = c.to_index() {
                if board.grid.get(idx).placed_wildlife() == Some(Wildlife::Elk) {
                    line_len += 1;
                    c = HexCoord::new(c.q + dq, c.r + dr);
                } else {
                    break;
                }
            }
            // Backward
            c = HexCoord::new(coord.q - dq, coord.r - dr);
            while let Some(idx) = c.to_index() {
                if board.grid.get(idx).placed_wildlife() == Some(Wildlife::Elk) {
                    line_len += 1;
                    c = HexCoord::new(c.q - dq, c.r - dr);
                } else {
                    break;
                }
            }

            if line_len >= 4 { continue; } // line already maxed out

            // Check if the backward direction can accept an elk (extension point)
            let bwd = HexCoord::new(coord.q - dq, coord.r - dr);
            let can_extend = bwd.to_index().map_or(false, |idx| {
                let cell = board.grid.get(idx);
                cell.is_present() && cell.can_place_wildlife(Wildlife::Elk)
            });

            if can_extend {
                // Marginal value of extending based on current line length
                let marginal = match line_len {
                    1 => 30, // 1→2: +3 points
                    2 => 40, // 2→3: +4 points
                    3 => 40, // 3→4: +4 points
                    _ => 0,
                };
                // Discount for uncertainty
                potential += marginal / 2;
            }
        }
    }

    potential
}

/// Salmon potential (Card A): reward extendable runs based on actual marginals.
/// Run scoring: 1→2: +2, 2→3: +3, 3→4: +4, 4→5: +4, 5→6: +5, 6→7+: +6
fn salmon_potential(board: &Board) -> i32 {
    let positions = &board.wildlife_positions[Wildlife::Salmon as usize];
    if positions.is_empty() { return 0; }
    let adj = &*ADJACENCY;
    let mut potential: i32 = 0;

    // Find connected components and their sizes (same as scorer)
    let mut visited = [false; 441];

    for &pos in positions.iter() {
        let idx = pos as usize;
        if visited[idx] { continue; }

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

        let run_len = component.len() as u16;

        // Check if this is a valid run (no branching)
        let is_valid = component.iter().all(|&p| {
            adj.neighbors_of(p as usize)
                .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon))
                .count() <= 2
        });

        if !is_valid {
            // Branching run — penalty
            potential -= 30;
            continue;
        }

        // Find endpoints (salmon with exactly 0 or 1 salmon neighbor)
        for &p in &component {
            let salmon_neighbors: usize = adj.neighbors_of(p as usize)
                .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon))
                .count();

            let is_endpoint = salmon_neighbors <= 1;
            if !is_endpoint { continue; }

            // Check if endpoint has an adjacent empty slot accepting salmon
            let can_extend = adj.neighbors_of(p as usize).any(|nidx| {
                let cell = board.grid.get(nidx);
                cell.is_present()
                    && cell.can_place_wildlife(Wildlife::Salmon)
                    && cell.placed_wildlife() != Some(Wildlife::Salmon)
            });

            if can_extend {
                // Marginal value based on what the run would become
                let marginal = match run_len {
                    1 => 20,  // 1→2: +2 points
                    2 => 30,  // 2→3: +3
                    3 => 40,  // 3→4: +4
                    4 => 40,  // 4→5: +4
                    5 => 50,  // 5→6: +5
                    6 => 60,  // 6→7: +6
                    _ => 0,   // 7+ already maxed
                };
                // Discount for uncertainty, only count once per run
                potential += marginal / 2;
                break; // only count one endpoint per run
            }
        }
    }

    potential
}

/// Hawk potential (Card A): reward isolated hawks and safe placement opportunities.
/// Marginals: 0→1: +2, 1→2: +3, 2→3: +3, 3→4: +3, 4→5: +3, 5→6: +4, 6→7: +4, 7→8+: +6
fn hawk_potential(board: &Board) -> i32 {
    let positions = &board.wildlife_positions[Wildlife::Hawk as usize];
    let adj = &*ADJACENCY;
    let mut potential: i32 = 0;

    // Count current isolated hawks
    let mut isolated_count = 0u16;
    for &pos in positions.iter() {
        let has_adjacent_hawk = adj.neighbors_of(pos as usize)
            .any(|nidx| board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Hawk));
        if !has_adjacent_hawk {
            isolated_count += 1;
        }
    }

    // Value of maintaining each isolated hawk's safety
    for &pos in positions.iter() {
        let idx = pos as usize;
        let has_adjacent_hawk = adj.neighbors_of(idx)
            .any(|nidx| board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Hawk));

        if has_adjacent_hawk { continue; } // not isolated

        // Count danger slots (empty slots adjacent to this hawk that accept hawks)
        let danger_slots: usize = adj.neighbors_of(idx)
            .filter(|&nidx| {
                let cell = board.grid.get(nidx);
                cell.is_present() && cell.can_place_wildlife(Wildlife::Hawk)
            })
            .count();

        if danger_slots == 0 {
            potential += 10; // perfectly safe isolated hawk
        } else {
            // Some risk of losing isolation
            potential += 5;
        }
    }

    // Bonus for potential to add MORE isolated hawks
    // Count empty tiles that accept hawks and aren't adjacent to any hawk
    let mut safe_hawk_slots = 0i32;
    for &tile_idx in &board.placed_tiles {
        let idx = tile_idx as usize;
        let cell = board.grid.get(idx);
        if cell.has_wildlife() || !cell.can_place_wildlife(Wildlife::Hawk) {
            continue;
        }
        let adjacent_to_hawk = adj.neighbors_of(idx)
            .any(|nidx| board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Hawk));
        if !adjacent_to_hawk {
            safe_hawk_slots += 1;
        }
    }

    // Marginal value of adding the next isolated hawk
    let next_marginal = match isolated_count {
        0 => 20,  // +2
        1..=4 => 30, // +3
        5..=6 => 40, // +4
        _ => 60,     // +6
    };

    // Potential for adding isolated hawks (capped at 2 for uncertainty)
    let addable = safe_hawk_slots.min(2);
    potential += addable * next_marginal / 2;

    potential
}

/// Fox potential (Card A): reward diversity setup. Each unique adjacent type = +1.
fn fox_potential(board: &Board) -> i32 {
    let positions = &board.wildlife_positions[Wildlife::Fox as usize];
    let adj = &*ADJACENCY;
    let mut potential: i32 = 0;

    for &pos in positions.iter() {
        let idx = pos as usize;
        let mut seen_mask = 0u8;
        let mut empty_slots = 0u8;

        for nidx in adj.neighbors_of(idx) {
            let cell = board.grid.get(nidx);
            if let Some(w) = cell.placed_wildlife() {
                seen_mask |= 1 << (w as u8);
            } else if cell.is_present() && cell.allowed_wildlife().count() > 0 {
                empty_slots += 1;
            }
        }

        let current_types = seen_mask.count_ones() as i32;
        let missing_types = 5 - current_types;
        // Each empty slot could potentially add a new type
        let addable = missing_types.min(empty_slots as i32);
        // Each new type = +1 point = 10 in our scale
        potential += addable * 10 / 2; // discount for uncertainty
    }

    potential
}

/// Bonus for wildlife types that have very few remaining placement slots.
fn empty_slot_potential(board: &Board) -> i32 {
    let mut potential: i32 = 0;

    // Count how many tiles accept each wildlife type (empty slots only)
    let mut acceptance_count = [0u16; 5];
    for &tile_idx in &board.placed_tiles {
        let cell = board.grid.get(tile_idx as usize);
        if cell.has_wildlife() { continue; }
        for w in Wildlife::ALL {
            if cell.allowed_wildlife().contains(w) {
                acceptance_count[w as usize] += 1;
            }
        }
    }

    // Urgency bonus: fewer slots = more important to use them wisely
    for w in Wildlife::ALL {
        let count = acceptance_count[w as usize];
        if count == 1 {
            potential += 15; // last slot — very urgent
        } else if count == 2 {
            potential += 5;
        }
    }

    potential
}
