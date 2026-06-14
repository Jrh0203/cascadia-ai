use cascadia_core::board::Board;
use cascadia_core::hex::{HexCoord, ADJACENCY};
use cascadia_core::types::{ScoringCardVariant, ScoringCards, Wildlife};

/// Compute a potential bonus estimating future scoring opportunities.
/// Returns a value in "fractional points" (scaled by 10 to avoid floats).
/// Per-animal heuristics dispatch to the active scoring-card variant.
pub fn board_potential(board: &Board, cards: &ScoringCards) -> i32 {
    let mut potential: i32 = 0;

    potential += habitat_potential(board);
    potential += bear_potential_dispatch(board, cards.variant_for(Wildlife::Bear));
    potential += elk_potential_dispatch(board, cards.variant_for(Wildlife::Elk));
    potential += salmon_potential_dispatch(board, cards.variant_for(Wildlife::Salmon));
    potential += hawk_potential_dispatch(board, cards.variant_for(Wildlife::Hawk));
    potential += fox_potential_dispatch(board, cards.variant_for(Wildlife::Fox));
    potential += empty_slot_potential(board);

    potential
}

#[inline]
fn bear_potential_dispatch(board: &Board, v: ScoringCardVariant) -> i32 {
    match v {
        ScoringCardVariant::A => bear_potential(board),
        ScoringCardVariant::C => bear_potential_c(board),
        // Bear B (groups of exactly 3) and D (sizes 2/3/4) have different sweet
        // spots. For now use Card A's "extend toward small groups" heuristic as
        // a rough proxy — it's directionally correct (more bears → more points).
        _ => bear_potential(board),
    }
}

#[inline]
fn elk_potential_dispatch(board: &Board, v: ScoringCardVariant) -> i32 {
    match v {
        ScoringCardVariant::A => elk_potential(board),
        ScoringCardVariant::B => elk_potential_b(board),
        // Elk C (any contiguous group, super-linear) and D (rings around any hex
        // center) prefer dense clusters too — Card B is the best single proxy.
        ScoringCardVariant::C | ScoringCardVariant::D => elk_potential_b(board),
    }
}

#[inline]
fn salmon_potential_dispatch(board: &Board, v: ScoringCardVariant) -> i32 {
    match v {
        ScoringCardVariant::A => salmon_potential(board),
        ScoringCardVariant::D => salmon_potential_d(board),
        // Salmon B/C use the same chain rule as A, just with different score
        // tables — A's "extend the run" heuristic is directionally correct.
        _ => salmon_potential(board),
    }
}

#[inline]
fn hawk_potential_dispatch(board: &Board, v: ScoringCardVariant) -> i32 {
    match v {
        ScoringCardVariant::A => hawk_potential(board),
        ScoringCardVariant::D => hawk_potential_d(board),
        // Hawk B/C also reward LOS — Card A's "isolate" heuristic is the
        // OPPOSITE of what they want, but we don't have variant-specific
        // potentials yet. Fall back to A and accept the mismatch.
        _ => hawk_potential(board),
    }
}

#[inline]
fn fox_potential_dispatch(board: &Board, v: ScoringCardVariant) -> i32 {
    match v {
        ScoringCardVariant::A => fox_potential(board),
        ScoringCardVariant::B => fox_potential_b(board),
        // Fox C (single-type max count) and D (pair scoring) need their own
        // heuristics; B's "encourage same-type adjacency" is closer to C/D
        // than A's "encourage diversity".
        ScoringCardVariant::C | ScoringCardVariant::D => fox_potential_b(board),
    }
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
        0 => 40, // +4 points
        1 => 70, // +7 points
        2 => 80, // +8 points
        _ => 80, // +8 points
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
            let bear_slots: usize = adj
                .neighbors_of(idx)
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
        if visited[idx] {
            continue;
        }

        let mut size = 0u16;
        let mut queue = arrayvec::ArrayVec::<u16, 24>::new();
        queue.push(pos);
        visited[idx] = true;

        while let Some(current) = queue.pop() {
            size += 1;
            for nidx in adj.neighbors_of(current as usize) {
                if !visited[nidx] && board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Bear)
                {
                    visited[nidx] = true;
                    queue.push(nidx as u16);
                }
            }
        }

        if size == 2 {
            pairs += 1;
        }
    }
    pairs
}

/// Elk potential (Card A): reward extendable lines based on actual marginals.
/// Line of 1→2: +3pts, 2→3: +4pts, 3→4+: +4pts. Lines of 4+ have no extension value.
fn elk_potential(board: &Board) -> i32 {
    let positions = &board.wildlife_positions[Wildlife::Elk as usize];
    if positions.is_empty() {
        return 0;
    }
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

            if !has_fwd_elk {
                continue;
            }

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

            if line_len >= 4 {
                continue;
            } // line already maxed out

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
    if positions.is_empty() {
        return 0;
    }
    let adj = &*ADJACENCY;
    let mut potential: i32 = 0;

    // Find connected components and their sizes (same as scorer)
    let mut visited = [false; 441];

    for &pos in positions.iter() {
        let idx = pos as usize;
        if visited[idx] {
            continue;
        }

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
                .count()
                <= 2
        });

        if !is_valid {
            // Branching run — penalty
            potential -= 30;
            continue;
        }

        // Find endpoints (salmon with exactly 0 or 1 salmon neighbor)
        for &p in &component {
            let salmon_neighbors: usize = adj
                .neighbors_of(p as usize)
                .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon))
                .count();

            let is_endpoint = salmon_neighbors <= 1;
            if !is_endpoint {
                continue;
            }

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
                    1 => 20, // 1→2: +2 points
                    2 => 30, // 2→3: +3
                    3 => 40, // 3→4: +4
                    4 => 40, // 4→5: +4
                    5 => 50, // 5→6: +5
                    6 => 60, // 6→7: +6
                    _ => 0,  // 7+ already maxed
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
        let has_adjacent_hawk = adj
            .neighbors_of(pos as usize)
            .any(|nidx| board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Hawk));
        if !has_adjacent_hawk {
            isolated_count += 1;
        }
    }

    // Value of maintaining each isolated hawk's safety
    for &pos in positions.iter() {
        let idx = pos as usize;
        let has_adjacent_hawk = adj
            .neighbors_of(idx)
            .any(|nidx| board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Hawk));

        if has_adjacent_hawk {
            continue;
        } // not isolated

        // Count danger slots (empty slots adjacent to this hawk that accept hawks)
        let danger_slots: usize = adj
            .neighbors_of(idx)
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
        let adjacent_to_hawk = adj
            .neighbors_of(idx)
            .any(|nidx| board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Hawk));
        if !adjacent_to_hawk {
            safe_hawk_slots += 1;
        }
    }

    // Marginal value of adding the next isolated hawk
    let next_marginal = match isolated_count {
        0 => 20,     // +2
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

/// Bear potential (Card C): reward growing groups toward sizes 1/2/3.
/// Marginals: an empty pair-completion is +5 (2-pt loss as singleton + 5-pt pair = +3 net),
/// a 2→3 extension is +3 (8-pt triple − 5-pt pair). Sizes >3 score 0 so they're penalised.
/// +3 set-completion bonus rewarded if board has size-1 + size-2 components and a 3 is reachable.
fn bear_potential_c(board: &Board) -> i32 {
    let positions = &board.wildlife_positions[Wildlife::Bear as usize];
    let adj = &*ADJACENCY;
    if positions.is_empty() {
        return 0;
    }

    // Per-component sizes (BFS), and per-component "can extend by 1" flag.
    let mut visited = [false; 441];
    let mut sizes = arrayvec::ArrayVec::<u16, 32>::new();
    let mut extendable = arrayvec::ArrayVec::<bool, 32>::new();

    for &pos in positions.iter() {
        let idx = pos as usize;
        if visited[idx] {
            continue;
        }
        let mut comp = arrayvec::ArrayVec::<u16, 32>::new();
        let mut q = arrayvec::ArrayVec::<u16, 32>::new();
        q.push(pos);
        visited[idx] = true;
        while let Some(c) = q.pop() {
            let _ = comp.try_push(c);
            for n in adj.neighbors_of(c as usize) {
                if !visited[n] && board.grid.get(n).placed_wildlife() == Some(Wildlife::Bear) {
                    visited[n] = true;
                    let _ = q.try_push(n as u16);
                }
            }
        }
        let mut can_ext = false;
        for &p in &comp {
            for n in adj.neighbors_of(p as usize) {
                let cell = board.grid.get(n);
                if cell.is_present()
                    && !cell.has_wildlife()
                    && cell.can_place_wildlife(Wildlife::Bear)
                {
                    can_ext = true;
                    break;
                }
            }
            if can_ext {
                break;
            }
        }
        let _ = sizes.try_push(comp.len() as u16);
        let _ = extendable.try_push(can_ext);
    }

    let mut p: i32 = 0;
    let mut has_1 = false;
    let mut has_2 = false;
    let mut has_3 = false;
    for (&s, &ext) in sizes.iter().zip(extendable.iter()) {
        match s {
            1 => {
                has_1 = true;
                if ext {
                    p += 30;
                } // +3 to grow to pair (5 pts vs 2 pts)
            }
            2 => {
                has_2 = true;
                if ext {
                    p += 30;
                } // +3 to grow to triple (8 pts vs 5 pts)
            }
            3 => {
                has_3 = true;
                if ext {
                    p -= 80;
                } // BIG penalty: 4+ scores 0, lose all 8 pts
            }
            _ => {
                p -= 30;
            } // already in dead zone
        }
    }
    // Bonus completion: if we have 2 of the 3 sizes, partial credit toward +3 bonus.
    let sizes_present = (has_1 as i32) + (has_2 as i32) + (has_3 as i32);
    p += sizes_present * 10; // gradient toward the bonus

    p
}

/// Elk potential (Card B): reward DENSE shapes (triangle, rhombus) rather than lines.
/// Marginals: single = 2, pair = 5, triangle = 9, rhombus = 13.
/// "Triangle" = 3 mutually-adjacent elk. "Rhombus" = triangle + 1 adjacent elk.
fn elk_potential_b(board: &Board) -> i32 {
    let positions = &board.wildlife_positions[Wildlife::Elk as usize];
    if positions.is_empty() {
        return 0;
    }
    let adj = &*ADJACENCY;
    let mut p: i32 = 0;

    for &pos in positions.iter() {
        let idx = pos as usize;
        // Count elk neighbors and "elk-able" empty neighbors.
        let mut elk_n = 0u8;
        let mut grow = 0u8;
        for n in adj.neighbors_of(idx) {
            let cell = board.grid.get(n);
            if cell.placed_wildlife() == Some(Wildlife::Elk) {
                elk_n += 1;
            } else if cell.is_present()
                && !cell.has_wildlife()
                && cell.can_place_wildlife(Wildlife::Elk)
            {
                grow += 1;
            }
        }
        // Singleton with growth potential → reward becoming a pair (5 vs 2 = +3).
        if elk_n == 0 && grow > 0 {
            p += 15;
        }
        // Pair-end (1 elk neighbor) with growth → could become triangle (9 vs 5 = +4).
        // Highly valuable if the growth slot is also adjacent to the OTHER elk (forms triangle).
        if elk_n == 1 && grow > 0 {
            p += 20;
        }
        // 2-neighbor elk (already in a triad-like config) with growth → rhombus (13 vs 9 = +4).
        if elk_n == 2 && grow > 0 {
            p += 20;
        }
        // 3+ neighbor elk (already a tight cluster) — diminishing returns.
        if elk_n >= 3 {
            p -= 5;
        }
    }
    p
}

/// Salmon potential (Card D): reward salmon ADJACENT to non-salmon wildlife,
/// and runs of length ≥ 3. (Each adjacent non-salmon token = +1 pt under Card D.)
fn salmon_potential_d(board: &Board) -> i32 {
    let positions = &board.wildlife_positions[Wildlife::Salmon as usize];
    if positions.is_empty() {
        return 0;
    }
    let adj = &*ADJACENCY;
    let mut p: i32 = 0;

    // Component sizes (same as scorer)
    let mut visited = [false; 441];
    for &pos in positions.iter() {
        let idx = pos as usize;
        if visited[idx] {
            continue;
        }
        let mut comp = arrayvec::ArrayVec::<u16, 32>::new();
        let mut q = arrayvec::ArrayVec::<u16, 32>::new();
        q.push(pos);
        visited[idx] = true;
        while let Some(c) = q.pop() {
            let _ = comp.try_push(c);
            for n in adj.neighbors_of(c as usize) {
                if !visited[n] && board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon) {
                    visited[n] = true;
                    let _ = q.try_push(n as u16);
                }
            }
        }
        let len = comp.len();
        // Validity check (≤ 2 salmon neighbors per cell)
        let valid = comp.iter().all(|&p| {
            adj.neighbors_of(p as usize)
                .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon))
                .count()
                <= 2
        });
        if !valid {
            p -= 30;
            continue;
        }

        // Reward growth toward length 3 (the qualifying threshold).
        // Runs at length 1 and 2 score 0 outright but are PROGRESS toward
        // the 3+ payoff. Give a graded reward by distance to threshold,
        // multiplied by the already-accumulated adjacent-animal bonus
        // (which will cash in once len hits 3).
        if len < 3 {
            // Find an endpoint with extension room.
            let mut can_extend = false;
            for &c in &comp {
                let salmon_n: usize = adj
                    .neighbors_of(c as usize)
                    .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon))
                    .count();
                if salmon_n > 1 {
                    continue;
                }
                if adj.neighbors_of(c as usize).any(|n| {
                    let cell = board.grid.get(n);
                    cell.is_present()
                        && !cell.has_wildlife()
                        && cell.can_place_wildlife(Wildlife::Salmon)
                }) {
                    can_extend = true;
                    break;
                }
            }
            // Count unique non-salmon adjacent tokens already next to this seed run
            // — they'll add 1 pt each as soon as the run qualifies.
            let mut seen = [false; 441];
            let mut adj_animals = 0i32;
            for &c in &comp {
                for n in adj.neighbors_of(c as usize) {
                    if seen[n] {
                        continue;
                    }
                    seen[n] = true;
                    if let Some(w) = board.grid.get(n).placed_wildlife() {
                        if w != Wildlife::Salmon {
                            adj_animals += 1;
                        }
                    }
                }
            }
            // Base seed reward: how far along the run is. Length 2 is half-credit
            // toward length-3 qualification; length 1 is quarter-credit. Each is
            // also worth 1 pt directly once qualified (per-salmon scoring).
            let seed_value: i32 = match len {
                1 => 10, // singletons: small nudge to plant a seed
                2 => 25, // 1 step away — strong nudge to complete
                _ => 0,
            };
            // If the run can be extended, the seed-value AND the staged adj-animal
            // bonus (worth ~adj_animals pts at qualification) are realisable.
            if can_extend {
                p += seed_value;
                // Discount: bonus only realised after one more salmon placement,
                // which may not happen — half-credit.
                p += adj_animals * 5;
            } else {
                // Trapped seed (no extension slot) — small consolation, since the
                // adjacent animals at least set up future positioning. But mostly
                // a wasted salmon: penalise modestly so greedy doesn't plant
                // dead-end singletons.
                p += seed_value / 4;
                p -= 10;
            }
        } else {
            // Run already qualifies — count unique non-salmon adjacent tokens (the actual D bonus).
            let mut seen = [false; 441];
            let mut bonus = 0i32;
            for &c in &comp {
                for n in adj.neighbors_of(c as usize) {
                    if seen[n] {
                        continue;
                    }
                    seen[n] = true;
                    if let Some(w) = board.grid.get(n).placed_wildlife() {
                        if w != Wildlife::Salmon {
                            bonus += 10;
                        }
                    }
                }
            }
            // Reward the existing bonus + a small add-one-more-salmon nudge.
            p += bonus / 2; // already partially captured by current scoring
                            // Reward extension: each new salmon adds 1 pt + potentially adj-animal bonuses.
            let mut endpoints_extendable = 0;
            for &c in &comp {
                let salmon_n: usize = adj
                    .neighbors_of(c as usize)
                    .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon))
                    .count();
                if salmon_n > 1 {
                    continue;
                }
                if adj.neighbors_of(c as usize).any(|n| {
                    let cell = board.grid.get(n);
                    cell.is_present()
                        && !cell.has_wildlife()
                        && cell.can_place_wildlife(Wildlife::Salmon)
                }) {
                    endpoints_extendable += 1;
                }
            }
            p += (endpoints_extendable.min(2) as i32) * 15;
        }
    }
    p
}

/// Hawk potential (Card D): reward hawks placed where they could form a non-adjacent
/// LOS pair with another hawk WITH non-hawk wildlife in the cells between.
/// (Per pair: 1 type=4, 2=7, 3+=9. Each hawk in ≤ 1 pair.)
fn hawk_potential_d(board: &Board) -> i32 {
    let positions = &board.wildlife_positions[Wildlife::Hawk as usize];
    if positions.len() < 2 {
        return 0;
    }
    let mut hawk_set = [false; 441];
    for &p in positions.iter() {
        hawk_set[p as usize] = true;
    }
    let mut p: i32 = 0;

    // For each hawk, walk each direction. If we find another hawk at distance ≥ 2
    // (non-adjacent) without an intervening hawk, count unique non-hawk wildlife
    // types in between. Bigger crowd = bigger pair-score.
    let mut paired = [false; 441];
    for (i, &pos) in positions.iter().enumerate() {
        if paired[pos as usize] {
            continue;
        }
        let coord = HexCoord::from_index(pos as usize);
        for &(dq, dr) in &cascadia_core::hex::HexCoord::DIRECTIONS {
            let mut cur = HexCoord::new(coord.q + dq, coord.r + dr);
            let mut steps = 1u16;
            let mut types_mask = 0u8;
            loop {
                match cur.to_index() {
                    Some(idx) => {
                        if hawk_set[idx] {
                            if steps >= 2 && !paired[idx] {
                                let unique =
                                    (types_mask & !(1 << Wildlife::Hawk as u8)).count_ones();
                                let pair_pts: i32 = match unique {
                                    0 => 0,
                                    1 => 40,
                                    2 => 70,
                                    _ => 90,
                                };
                                if pair_pts > 0 {
                                    p += pair_pts / 2; // discount for partial credit
                                    paired[pos as usize] = true;
                                    paired[idx] = true;
                                }
                            }
                            break;
                        }
                        if let Some(w) = board.grid.get(idx).placed_wildlife() {
                            types_mask |= 1 << (w as u8);
                        }
                    }
                    None => break,
                }
                cur = HexCoord::new(cur.q + dq, cur.r + dr);
                steps += 1;
            }
        }
        let _ = i;
    }
    p
}

/// Fox potential (Card B): reward foxes with ≥ 2 of the same non-fox type adjacent
/// (a "pair-type"). Per fox: 1 pair-type=3, 2=5, 3=7.
fn fox_potential_b(board: &Board) -> i32 {
    let positions = &board.wildlife_positions[Wildlife::Fox as usize];
    if positions.is_empty() {
        return 0;
    }
    let adj = &*ADJACENCY;
    let mut p: i32 = 0;

    for &pos in positions.iter() {
        let mut counts = [0u8; 5];
        let mut empty_with_type: [u8; 5] = [0; 5]; // empties that could become this type
        for n in adj.neighbors_of(pos as usize) {
            let cell = board.grid.get(n);
            if let Some(w) = cell.placed_wildlife() {
                if w != Wildlife::Fox {
                    counts[w as usize] += 1;
                }
            } else if cell.is_present() && !cell.has_wildlife() {
                for w in 0..5 {
                    let wl = Wildlife::from_u8(w as u8).unwrap();
                    if wl == Wildlife::Fox {
                        continue;
                    }
                    if cell.can_place_wildlife(wl) {
                        empty_with_type[w] += 1;
                    }
                }
            }
        }
        let pair_types_now = counts.iter().filter(|&&c| c >= 2).count() as i32;
        // Score the achieved state at half-credit (already in the live scoring).
        p += match pair_types_now {
            0 => 0,
            1 => 15,
            2 => 25,
            _ => 35,
        };
        // Reward types that are 1-away from being a pair (i.e., have count = 1
        // AND have an empty neighbor that could host that type).
        for w in 0..5 {
            if counts[w] == 1 && empty_with_type[w] >= 1 {
                p += 10; // could grow to a pair-type
            }
        }
    }
    p
}

/// Bonus for wildlife types that have very few remaining placement slots.
fn empty_slot_potential(board: &Board) -> i32 {
    let mut potential: i32 = 0;

    // Count how many tiles accept each wildlife type (empty slots only)
    let mut acceptance_count = [0u16; 5];
    for &tile_idx in &board.placed_tiles {
        let cell = board.grid.get(tile_idx as usize);
        if cell.has_wildlife() {
            continue;
        }
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
