//! Wildlife-strategic candidate generation.
//! For each wildlife type, find the best tile placement that extends
//! or sets up a valuable pattern, regardless of habitat score.
//! These candidates are injected alongside greedy candidates for MCE to evaluate.

use cascadia_core::board::Board;
use cascadia_core::game::GameState;
use cascadia_core::hex::{HexCoord, ADJACENCY};
use cascadia_core::types::{ScoringCardVariant, Wildlife};

use crate::eval::ScoredMove;

/// Generate wildlife-strategic candidates for the current game state.
/// Returns moves that target pattern completion for each wildlife type.
pub fn wildlife_strategic_candidates(game: &GameState) -> Vec<ScoredMove> {
    let board = &game.boards[game.current_player];
    let cards = &game.scoring_cards;
    let frontier = board.frontier();
    if frontier.is_empty() {
        return Vec::new();
    }

    let market_pairs: Vec<_> = game.market.available()
        .map(|(i, pair)| (i, pair.tile, pair.wildlife))
        .collect();
    if market_pairs.is_empty() {
        return Vec::new();
    }

    let adj = &*ADJACENCY;
    let base_wildlife: u16 = cascadia_core::scoring::wildlife::score_all_wildlife(board, cards)
        .iter().sum();

    let mut candidates: Vec<ScoredMove> = Vec::new();
    let mut board_clone = board.clone();

    let has_tokens = board.nature_tokens > 0;

    // Build combos: normal pairs + independent drafts (if tokens available)
    struct MarketCombo {
        tile_idx: usize,
        tile: cascadia_core::types::TileData,
        wildlife: cascadia_core::types::Wildlife,
        wl_market_idx: Option<usize>,
    }
    let mut combos = Vec::new();
    for &(idx, tile, wl) in &market_pairs {
        combos.push(MarketCombo { tile_idx: idx, tile, wildlife: wl, wl_market_idx: None });
    }
    // Independent drafts: pick tile from one slot, wildlife from another
    if has_tokens {
        for &(ti, tile, _) in &market_pairs {
            for &(wi, _, wl) in &market_pairs {
                if ti != wi {
                    combos.push(MarketCombo { tile_idx: ti, tile, wildlife: wl, wl_market_idx: Some(wi) });
                }
            }
        }
    }

    for combo in &combos {
        let (idx, tile, wildlife) = (combo.tile_idx, combo.tile, combo.wildlife);
        let max_rot: u8 = if tile.terrain2.is_none() { 1 } else { 6 };

        for &fi in frontier.iter() {
            let coord = HexCoord::from_index(fi as usize);
            for rot in 0..max_rot {
                let tile_action = match board_clone.place_tile(coord, tile, rot) {
                    Some(a) => a,
                    None => continue,
                };

                let hab: u16 = board_clone.largest_group.iter().sum();

                // Score every valid wildlife placement by PATTERN VALUE, not just delta
                let variant = cards.variant_for(wildlife);
                let without = cascadia_core::scoring::wildlife::score_wildlife(
                    &board_clone, wildlife, variant,
                );

                let placed_snapshot: arrayvec::ArrayVec<u16, 64> =
                    board_clone.placed_tiles.iter().copied().collect();

                for &ti in placed_snapshot.iter() {
                    if !board_clone.grid.get(ti as usize).can_place_wildlife(wildlife) {
                        continue;
                    }
                    let wa = match board_clone.place_wildlife(ti as usize, wildlife) {
                        Some(a) => a,
                        None => continue,
                    };

                    let with = cascadia_core::scoring::wildlife::score_wildlife(
                        &board_clone, wildlife, variant,
                    );
                    let delta = with.saturating_sub(without);

                    // Compute strategic value: how much does this placement
                    // improve the PATTERN SETUP, not just immediate score?
                    // Dispatched by active scoring card so alt-rules get the
                    // right pattern targets (Hawk D LOS, Fox B pair-types, etc).
                    let strategic_value = pattern_setup_value_dispatch(
                        &board_clone, ti as usize, wildlife, variant, adj,
                    );

                    board_clone.undo(wa);

                    // Only keep moves with significant strategic value
                    // (either immediate delta OR strong setup)
                    if strategic_value >= 3 || delta >= 3 {
                        let nat_bonus: u16 = if board_clone.grid.get(ti as usize).is_keystone() { 1 } else { 0 };
                        let total = hab + base_wildlife + delta + board.nature_tokens as u16 + nat_bonus;
                        let wc = HexCoord::from_index(ti as usize);
                        candidates.push(ScoredMove {
                            market_index: idx,
                            tile_q: coord.q,
                            tile_r: coord.r,
                            rotation: rot,
                            wildlife_q: Some(wc.q),
                            wildlife_r: Some(wc.r),
                            score: total,
                            eval: (total as i32 + strategic_value as i32 * 100),
                            wildlife_market_index: combo.wl_market_idx,
                        });
                    }
                }

                // Also check: does this tile placement CREATE valuable wildlife slots?
                // (even if we don't place wildlife from this pair here)
                let slot_value = new_slot_value_dispatch(&board_clone, coord, cards, adj);
                if slot_value >= 4 {
                    // This tile creates great wildlife slots — add as a candidate
                    // with the best available wildlife placement
                    let best_wl = find_best_wildlife_placement(
                        &mut board_clone, wildlife, variant, without,
                    );
                    let nat_bonus = best_wl.map(|(ti, _)| {
                        if board_clone.grid.get(ti).is_keystone() { 1u16 } else { 0 }
                    }).unwrap_or(0);
                    let delta = best_wl.map(|(_, d)| d).unwrap_or(0);
                    let total = hab + base_wildlife + delta + board.nature_tokens as u16 + nat_bonus;

                    let (wq, wr) = best_wl.map(|(ti, _)| {
                        let wc = HexCoord::from_index(ti);
                        (Some(wc.q), Some(wc.r))
                    }).unwrap_or((None, None));

                    candidates.push(ScoredMove {
                        market_index: idx,
                        tile_q: coord.q,
                        tile_r: coord.r,
                        rotation: rot,
                        wildlife_q: wq,
                        wildlife_r: wr,
                        score: total,
                        eval: (total as i32 + slot_value as i32 * 100),
                        wildlife_market_index: None,
                    });
                }

                board_clone.undo(tile_action);
            }
        }
    }

    // Sort by strategic eval and keep top candidates
    candidates.sort_by(|a, b| b.eval.cmp(&a.eval));
    candidates.dedup_by(|a, b| {
        a.tile_q == b.tile_q && a.tile_r == b.tile_r
            && a.rotation == b.rotation && a.wildlife_q == b.wildlife_q
    });
    candidates.truncate(10);
    candidates
}

/// How much PATTERN SETUP value does placing this wildlife at this position provide?
/// Returns a score 0-10 based on pattern-specific heuristics.
fn pattern_setup_value(
    board: &Board,
    pos: usize,
    wildlife: Wildlife,
    adj: &cascadia_core::hex::AdjacencyTable,
) -> u16 {
    match wildlife {
        Wildlife::Bear => {
            // Value: completing a pair (huge) or setting up a half-pair
            let bear_neighbors: usize = adj.neighbors_of(pos)
                .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Bear))
                .count();

            if bear_neighbors == 1 {
                // Check if this completes an isolated pair (no other bears adjacent to either)
                let neighbor_pos = adj.neighbors_of(pos)
                    .find(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Bear))
                    .unwrap();
                let neighbor_other_bears = adj.neighbors_of(neighbor_pos)
                    .filter(|&n| n != pos && board.grid.get(n).placed_wildlife() == Some(Wildlife::Bear))
                    .count();
                if neighbor_other_bears == 0 { 8 } else { 0 } // valid pair = very valuable
            } else if bear_neighbors == 0 {
                // Half-pair setup: how many adjacent slots accept bears?
                let bear_slots: usize = adj.neighbors_of(pos)
                    .filter(|&n| {
                        let cell = board.grid.get(n);
                        cell.is_present() && cell.can_place_wildlife(Wildlife::Bear)
                    })
                    .count();
                if bear_slots >= 2 { 5 } else if bear_slots == 1 { 3 } else { 0 }
            } else {
                0 // cluster, bad
            }
        }
        Wildlife::Elk => {
            // Value: extending an existing line
            let coord = HexCoord::from_index(pos);
            let mut best = 0u16;

            for &(dq, dr) in &HexCoord::LINE_DIRECTIONS {
                // Count line length in both directions from this position
                let mut len = 1u16;
                // Forward
                let mut c = HexCoord::new(coord.q + dq, coord.r + dr);
                while let Some(idx) = c.to_index() {
                    if board.grid.get(idx).placed_wildlife() == Some(Wildlife::Elk) {
                        len += 1;
                        c = HexCoord::new(c.q + dq, c.r + dr);
                    } else { break; }
                }
                // Backward
                c = HexCoord::new(coord.q - dq, coord.r - dr);
                while let Some(idx) = c.to_index() {
                    if board.grid.get(idx).placed_wildlife() == Some(Wildlife::Elk) {
                        len += 1;
                        c = HexCoord::new(c.q - dq, c.r - dr);
                    } else { break; }
                }

                // Score based on resulting line length
                let value = match len {
                    2 => 4,  // created a pair, potential for line of 3-4
                    3 => 6,  // line of 3, one more for max
                    4 => 8,  // line of 4 = max score (13 pts)
                    _ => 0,  // single or 5+ (already maxed)
                };
                best = best.max(value);
            }
            best
        }
        Wildlife::Salmon => {
            // Value: extending an existing run
            let salmon_neighbors: usize = adj.neighbors_of(pos)
                .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon))
                .count();

            if salmon_neighbors == 1 {
                // Extending a run endpoint — find the run length
                let run_len = count_salmon_run(board, pos, adj);
                match run_len {
                    2 => 3,  // run of 2
                    3 => 5,  // run of 3 → 7 pts
                    4 => 6,  // run of 4 → 11 pts
                    5 => 7,  // run of 5 → 15 pts
                    6 => 8,  // run of 6 → 20 pts
                    _ => 2,
                }
            } else if salmon_neighbors == 0 {
                // Starting a run — check if adjacent to empty salmon-accepting slots
                let slots: usize = adj.neighbors_of(pos)
                    .filter(|&n| {
                        let cell = board.grid.get(n);
                        cell.is_present() && cell.can_place_wildlife(Wildlife::Salmon)
                    })
                    .count();
                if slots >= 2 { 3 } else { 0 }
            } else {
                0 // branching = bad
            }
        }
        Wildlife::Hawk => {
            // Value: isolated placement
            let has_hawk_neighbor = adj.neighbors_of(pos)
                .any(|n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Hawk));
            if !has_hawk_neighbor {
                // Count current isolated hawks to determine marginal value
                let isolated = board.wildlife_positions[Wildlife::Hawk as usize].iter()
                    .filter(|&&p| {
                        !adj.neighbors_of(p as usize)
                            .any(|n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Hawk))
                    })
                    .count();
                match isolated {
                    0..=2 => 4,
                    3..=4 => 5,
                    5..=6 => 6,
                    _ => 3,
                }
            } else {
                0
            }
        }
        Wildlife::Fox => {
            // Fox already scores well via greedy — minimal strategic value needed
            let mut mask = 0u8;
            for nidx in adj.neighbors_of(pos) {
                if let Some(w) = board.grid.get(nidx).placed_wildlife() {
                    mask |= 1 << (w as u8);
                }
            }
            let unique = mask.count_ones() as u16;
            if unique >= 4 { 5 } else { 0 }
        }
    }
}

/// Count the salmon run length that includes position `pos`.
fn count_salmon_run(board: &Board, pos: usize, adj: &cascadia_core::hex::AdjacencyTable) -> u16 {
    let mut visited = [false; 441];
    let mut queue = arrayvec::ArrayVec::<usize, 24>::new();
    queue.push(pos);
    visited[pos] = true;
    let mut len = 0u16;

    while let Some(current) = queue.pop() {
        len += 1;
        for nidx in adj.neighbors_of(current) {
            if !visited[nidx] && board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Salmon) {
                visited[nidx] = true;
                queue.push(nidx);
            }
        }
    }
    len
}

/// How valuable are the new wildlife slots created by this tile placement?
/// Checks if the placed tile creates slots adjacent to existing wildlife patterns.
fn new_slot_value(board: &Board, coord: HexCoord, adj: &cascadia_core::hex::AdjacencyTable) -> u16 {
    let idx = match coord.to_index() {
        Some(i) => i,
        None => return 0,
    };
    let cell = board.grid.get(idx);
    if !cell.is_present() { return 0; }

    let mut value = 0u16;

    // Check if this tile accepts wildlife that would complete patterns
    for w in Wildlife::ALL {
        if !cell.allowed_wildlife().contains(w) { continue; }

        match w {
            Wildlife::Bear => {
                // Does this slot neighbor an isolated bear?
                let adjacent_isolated_bear = adj.neighbors_of(idx).any(|n| {
                    if board.grid.get(n).placed_wildlife() != Some(Wildlife::Bear) { return false; }
                    // Check if that bear is isolated (no other bear neighbors)
                    adj.neighbors_of(n)
                        .filter(|&nn| nn != idx && board.grid.get(nn).placed_wildlife() == Some(Wildlife::Bear))
                        .count() == 0
                });
                if adjacent_isolated_bear { value += 5; }
            }
            Wildlife::Elk => {
                // Does this slot extend an existing elk line?
                let coord = HexCoord::from_index(idx);
                for &(dq, dr) in &HexCoord::LINE_DIRECTIONS {
                    let neighbor = HexCoord::new(coord.q + dq, coord.r + dr);
                    if let Some(nidx) = neighbor.to_index() {
                        if board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Elk) {
                            value += 4;
                            break;
                        }
                    }
                }
            }
            Wildlife::Salmon => {
                // Does this slot extend an existing salmon run endpoint?
                let has_salmon_neighbor = adj.neighbors_of(idx).any(|n| {
                    board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon)
                });
                if has_salmon_neighbor { value += 3; }
            }
            Wildlife::Hawk => {
                // Is this slot safe for hawk isolation?
                let has_hawk_neighbor = adj.neighbors_of(idx).any(|n| {
                    board.grid.get(n).placed_wildlife() == Some(Wildlife::Hawk)
                });
                if !has_hawk_neighbor { value += 2; }
            }
            _ => {}
        }
    }

    value
}

/// Find the best wildlife placement on the current board (for a specific wildlife type).
/// Returns (position_index, score_delta).
fn find_best_wildlife_placement(
    board: &mut Board,
    wildlife: Wildlife,
    variant: cascadia_core::types::ScoringCardVariant,
    without: u16,
) -> Option<(usize, u16)> {
    let mut best: Option<(usize, u16)> = None;
    let placed: arrayvec::ArrayVec<u16, 64> = board.placed_tiles.iter().copied().collect();

    for &ti in placed.iter() {
        if !board.grid.get(ti as usize).can_place_wildlife(wildlife) { continue; }
        let wa = match board.place_wildlife(ti as usize, wildlife) {
            Some(a) => a,
            None => continue,
        };
        let with = cascadia_core::scoring::wildlife::score_wildlife(board, wildlife, variant);
        board.undo(wa);
        let delta = with.saturating_sub(without);
        if best.is_none() || delta > best.unwrap().1 {
            best = Some((ti as usize, delta));
        }
    }
    best
}

// ─────────────────────────────────────────────────────────────────────
// Card-variant-aware dispatch for pattern_setup_value and new_slot_value.
// The original (above) is hardcoded for Card A. Under alt rules the targets
// fundamentally change (e.g. Hawk D wants LOS pairs, Card A wants isolation —
// the OPPOSITE incentive). This dispatch routes each (animal, variant) pair
// to the right scoring intuition so MCE's candidate pool aligns with the
// active scoring rules.
// ─────────────────────────────────────────────────────────────────────

fn pattern_setup_value_dispatch(
    board: &Board,
    pos: usize,
    wildlife: Wildlife,
    variant: ScoringCardVariant,
    adj: &cascadia_core::hex::AdjacencyTable,
) -> u16 {
    use ScoringCardVariant::*;
    match (wildlife, variant) {
        // Card A — keep original logic (well-validated for the v4opp champion).
        (_, A) => pattern_setup_value(board, pos, wildlife, adj),

        (Wildlife::Bear, C) => bear_c_setup(board, pos, adj),
        (Wildlife::Bear, D) => bear_d_setup(board, pos, adj),
        (Wildlife::Bear, B) => bear_b_setup(board, pos, adj),

        (Wildlife::Elk, B) => elk_b_setup(board, pos, adj),
        (Wildlife::Elk, C) => elk_c_setup(board, pos, adj),
        (Wildlife::Elk, D) => elk_d_setup(board, pos, adj),

        (Wildlife::Salmon, D) => salmon_d_setup(board, pos, adj),
        (Wildlife::Salmon, B) | (Wildlife::Salmon, C) => salmon_bc_setup(board, pos, adj, variant),

        (Wildlife::Hawk, D) => hawk_d_setup(board, pos, adj),
        (Wildlife::Hawk, B) => hawk_b_setup(board, pos, adj),
        (Wildlife::Hawk, C) => hawk_c_setup(board, pos, adj),

        (Wildlife::Fox, B) => fox_b_setup(board, pos, adj),
        (Wildlife::Fox, C) => fox_c_setup(board, pos, adj),
        (Wildlife::Fox, D) => fox_d_setup(board, pos, adj),
    }
}

fn new_slot_value_dispatch(
    board: &Board,
    coord: HexCoord,
    cards: &cascadia_core::types::ScoringCards,
    adj: &cascadia_core::hex::AdjacencyTable,
) -> u16 {
    let idx = match coord.to_index() {
        Some(i) => i,
        None => return 0,
    };
    let cell = board.grid.get(idx);
    if !cell.is_present() { return 0; }

    use ScoringCardVariant::*;
    let mut value = 0u16;
    for w in Wildlife::ALL {
        if !cell.allowed_wildlife().contains(w) { continue; }
        let v = cards.variant_for(w);
        // Each (animal, variant) contributes to slot value if a placement here
        // would be tactically valuable under the active rules.
        let contribution: u16 = match (w, v) {
            (_, A) => new_slot_value_card_a(board, idx, w, adj),
            (Wildlife::Bear, C) | (Wildlife::Bear, D) | (Wildlife::Bear, B) => {
                bear_slot_value_alt(board, idx, v, adj)
            }
            (Wildlife::Elk, B) | (Wildlife::Elk, C) | (Wildlife::Elk, D) => {
                elk_slot_value_alt(board, idx, v, adj)
            }
            (Wildlife::Salmon, D) => salmon_slot_value_d(board, idx, adj),
            (Wildlife::Salmon, _) => {
                if adj.neighbors_of(idx).any(|n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon)) { 3 } else { 0 }
            }
            (Wildlife::Hawk, D) | (Wildlife::Hawk, B) | (Wildlife::Hawk, C) => {
                hawk_slot_value_alt(board, idx, v, adj)
            }
            (Wildlife::Fox, B) => fox_slot_value_b(board, idx, adj),
            (Wildlife::Fox, _) => 0,
        };
        value += contribution;
    }
    value
}

// Helper that wraps the Card A version of new_slot_value's per-animal logic
// for use by the dispatcher (since the original new_slot_value iterates over
// all wildlife internally, we replicate per-animal here).
fn new_slot_value_card_a(
    board: &Board,
    idx: usize,
    w: Wildlife,
    adj: &cascadia_core::hex::AdjacencyTable,
) -> u16 {
    match w {
        Wildlife::Bear => {
            let adjacent_isolated_bear = adj.neighbors_of(idx).any(|n| {
                if board.grid.get(n).placed_wildlife() != Some(Wildlife::Bear) { return false; }
                adj.neighbors_of(n)
                    .filter(|&nn| nn != idx && board.grid.get(nn).placed_wildlife() == Some(Wildlife::Bear))
                    .count() == 0
            });
            if adjacent_isolated_bear { 5 } else { 0 }
        }
        Wildlife::Elk => {
            let coord = HexCoord::from_index(idx);
            for &(dq, dr) in &HexCoord::LINE_DIRECTIONS {
                let neighbor = HexCoord::new(coord.q + dq, coord.r + dr);
                if let Some(nidx) = neighbor.to_index() {
                    if board.grid.get(nidx).placed_wildlife() == Some(Wildlife::Elk) {
                        return 4;
                    }
                }
            }
            0
        }
        Wildlife::Salmon => {
            if adj.neighbors_of(idx).any(|n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon)) { 3 } else { 0 }
        }
        Wildlife::Hawk => {
            let has_hawk_neighbor = adj.neighbors_of(idx)
                .any(|n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Hawk));
            if !has_hawk_neighbor { 2 } else { 0 }
        }
        Wildlife::Fox => 0,
    }
}

// ─── Bear C: groups of size 1 (=2pt), 2 (=5pt), 3 (=8pt), +3 if all-three ───
fn bear_c_setup(board: &Board, pos: usize, adj: &cascadia_core::hex::AdjacencyTable) -> u16 {
    let bn: usize = adj.neighbors_of(pos)
        .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Bear))
        .count();
    // Count placed bear sizes already on the board (helper for "all three sizes" bonus).
    let (has1, has2, has3) = bear_sizes_on_board(board, adj);

    if bn == 0 {
        // Singleton placement = +2 immediate. Strategic value: opens a new
        // singleton component (counts toward all-three-sizes bonus if missing).
        let mut v = 4; // baseline "this is a real point earner"
        if !has1 && (has2 || has3) { v += 4; } // unlocks bonus path
        v
    } else if bn == 1 {
        // Joins an existing bear → forms size-2 (pair) IF the neighbor was singleton.
        // Need to check: does the neighbor have other bear neighbors already?
        let nbr = adj.neighbors_of(pos)
            .find(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Bear)).unwrap();
        let nbr_bn: usize = adj.neighbors_of(nbr)
            .filter(|&nn| nn != pos && board.grid.get(nn).placed_wildlife() == Some(Wildlife::Bear))
            .count();
        if nbr_bn == 0 {
            // Going singleton(2pt)→pair(5pt) = +3, AND we lose the singleton.
            // Net delta to score: +3. Plus bonus implications.
            let mut v = 6;
            if !has2 && (has1 || has3) { v += 4; }
            v
        } else if nbr_bn == 1 {
            // Joining a pair → forms size-3 (8pt vs 5pt = +3). Useful.
            let mut v = 6;
            if !has3 && (has1 || has2) { v += 4; }
            v
        } else {
            // Joining a triple → forms size-4 (8→0 = LOSS of 8pts!). BAD.
            0
        }
    } else if bn == 2 {
        // Two bear neighbors. Likely creates a size-3+ component. Need to check
        // if both neighbors are part of the same component or separate.
        let nbrs: arrayvec::ArrayVec<u16, 6> = adj.neighbors_of(pos)
            .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Bear))
            .map(|n| n as u16)
            .collect();
        // Best case: both neighbors are isolated singletons → joining creates a triple (8pt).
        // Worst case: both are already in larger components → 4+ result, 0 score.
        let total_neighbor_bears: usize = nbrs.iter().map(|&n| {
            adj.neighbors_of(n as usize)
                .filter(|&nn| nn != pos && board.grid.get(nn).placed_wildlife() == Some(Wildlife::Bear))
                .count()
        }).sum();
        if total_neighbor_bears == 0 {
            // Two singletons + this = size-3. Worth 8 vs 2+2=4 = +4 net. Solid.
            let mut v = 7;
            if !has3 && (has1 || has2) { v += 3; }
            v
        } else {
            0
        }
    } else {
        0
    }
}

fn bear_sizes_on_board(board: &Board, adj: &cascadia_core::hex::AdjacencyTable) -> (bool, bool, bool) {
    let positions = &board.wildlife_positions[Wildlife::Bear as usize];
    let mut visited = [false; 441];
    let (mut has1, mut has2, mut has3) = (false, false, false);
    for &p in positions.iter() {
        let idx = p as usize;
        if visited[idx] { continue; }
        let mut size = 0u16;
        let mut q = arrayvec::ArrayVec::<u16, 32>::new();
        q.push(p);
        visited[idx] = true;
        while let Some(c) = q.pop() {
            size += 1;
            for n in adj.neighbors_of(c as usize) {
                if !visited[n] && board.grid.get(n).placed_wildlife() == Some(Wildlife::Bear) {
                    visited[n] = true;
                    let _ = q.try_push(n as u16);
                }
            }
        }
        match size { 1 => has1 = true, 2 => has2 = true, 3 => has3 = true, _ => {} }
    }
    (has1, has2, has3)
}

// ─── Bear D: sizes 2 (=5), 3 (=8), 4 (=14) ───
fn bear_d_setup(board: &Board, pos: usize, adj: &cascadia_core::hex::AdjacencyTable) -> u16 {
    let bn: usize = adj.neighbors_of(pos)
        .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Bear))
        .count();
    if bn == 1 {
        let nbr = adj.neighbors_of(pos)
            .find(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Bear)).unwrap();
        let nbr_bn: usize = adj.neighbors_of(nbr)
            .filter(|&nn| nn != pos && board.grid.get(nn).placed_wildlife() == Some(Wildlife::Bear))
            .count();
        match nbr_bn {
            0 => 6, // singleton→pair (0→5=+5)
            1 => 5, // pair→triple (5→8=+3)
            2 => 8, // triple→quad (8→14=+6)
            _ => 0, // quad→5+ = lose 14
        }
    } else if bn == 2 {
        // Bridging two singletons makes a triple (5+0+0 → 8 = +3).
        7
    } else {
        0
    }
}

// ─── Bear B: ONLY size 3 scores (=10pt each) ───
fn bear_b_setup(board: &Board, pos: usize, adj: &cascadia_core::hex::AdjacencyTable) -> u16 {
    let bn: usize = adj.neighbors_of(pos)
        .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Bear))
        .count();
    if bn == 1 {
        let nbr = adj.neighbors_of(pos)
            .find(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Bear)).unwrap();
        let nbr_bn: usize = adj.neighbors_of(nbr)
            .filter(|&nn| nn != pos && board.grid.get(nn).placed_wildlife() == Some(Wildlife::Bear))
            .count();
        match nbr_bn {
            1 => 10, // pair→triple (0→10) — JACKPOT
            _ => 0,  // singleton→pair (0→0), triple→quad (10→0) — bad
        }
    } else if bn == 2 {
        // Both singletons → triple (0+0→10) — also jackpot
        let nbrs: arrayvec::ArrayVec<u16, 6> = adj.neighbors_of(pos)
            .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Bear))
            .map(|n| n as u16)
            .collect();
        let extra: usize = nbrs.iter().map(|&n| {
            adj.neighbors_of(n as usize)
                .filter(|&nn| nn != pos && board.grid.get(nn).placed_wildlife() == Some(Wildlife::Bear))
                .count()
        }).sum();
        if extra == 0 { 9 } else { 0 }
    } else {
        0
    }
}

// ─── Elk B: shapes (single 2, pair 5, triangle 9, rhombus 13) ───
fn elk_b_setup(board: &Board, pos: usize, adj: &cascadia_core::hex::AdjacencyTable) -> u16 {
    let elk_n: arrayvec::ArrayVec<u16, 6> = adj.neighbors_of(pos)
        .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Elk))
        .map(|n| n as u16)
        .collect();
    let n = elk_n.len();
    if n == 0 {
        // New singleton (2pt). Reward extendability that could become triangle.
        let elk_slots: usize = adj.neighbors_of(pos)
            .filter(|&n| {
                let cell = board.grid.get(n);
                cell.is_present() && cell.can_place_wildlife(Wildlife::Elk)
            }).count();
        if elk_slots >= 2 { 3 } else { 0 }
    } else if n == 1 {
        // Forms a pair (5pt). Bonus if there's a 3rd cell mutually adjacent for triangle.
        let nbr = elk_n[0] as usize;
        let could_triangle: bool = adj.neighbors_of(pos).any(|p3| {
            p3 != nbr
                && board.grid.get(p3).is_present() && !board.grid.get(p3).has_wildlife()
                && board.grid.get(p3).can_place_wildlife(Wildlife::Elk)
                && adj.neighbors_of(nbr).any(|n2| n2 == p3)
        });
        if could_triangle { 6 } else { 3 }
    } else if n >= 2 {
        // Multiple elk neighbors. Check if 2 of them are mutually adjacent (triangle completed).
        for i in 0..elk_n.len() {
            for j in (i + 1)..elk_n.len() {
                let a = elk_n[i] as usize;
                let b = elk_n[j] as usize;
                if adj.neighbors_of(a).any(|x| x == b) {
                    // Triangle completed (5→9 = +4). Check if rhombus also possible.
                    let could_rhombus: bool = adj.neighbors_of(pos).chain(adj.neighbors_of(a)).chain(adj.neighbors_of(b))
                        .any(|p4| {
                            p4 != pos && p4 != a && p4 != b
                                && board.grid.get(p4).is_present() && !board.grid.get(p4).has_wildlife()
                                && board.grid.get(p4).can_place_wildlife(Wildlife::Elk)
                        });
                    return if could_rhombus { 9 } else { 8 };
                }
            }
        }
        // Multiple neighbors but no triangle (e.g. line). Lower value.
        2
    } else { 0 }
}

// ─── Elk C: any contiguous group, super-linear (cap at 8) ───
fn elk_c_setup(board: &Board, pos: usize, adj: &cascadia_core::hex::AdjacencyTable) -> u16 {
    let elk_n: usize = adj.neighbors_of(pos)
        .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Elk))
        .count();
    if elk_n >= 1 {
        // Joining/growing a group is good (super-linear scoring favors big groups).
        4 + (elk_n.min(3) as u16)
    } else {
        // Isolated singleton with growth potential
        let slots: usize = adj.neighbors_of(pos)
            .filter(|&n| {
                let c = board.grid.get(n);
                c.is_present() && c.can_place_wildlife(Wildlife::Elk)
            }).count();
        if slots >= 2 { 2 } else { 0 }
    }
}

// ─── Elk D: rings around any hex center (1=2,2=5,3=8,4=12,5=16,6=21) ───
fn elk_d_setup(board: &Board, pos: usize, adj: &cascadia_core::hex::AdjacencyTable) -> u16 {
    // Maximize "elk surrounding a hex center". For each potential center
    // (= neighbor of pos), count how many elk are adjacent to it.
    let mut best: u16 = 0;
    for c in adj.neighbors_of(pos) {
        let elk_around: usize = adj.neighbors_of(c)
            .filter(|&n| n == pos || board.grid.get(n).placed_wildlife() == Some(Wildlife::Elk))
            .count();
        // Marginal value of extending a ring from k to k+1.
        let marginal: u16 = match elk_around {
            1 => 2, // we'd be a singleton ring (worth 2)
            2 => 3, // 1→2 = +3
            3 => 3, // 2→3 = +3
            4 => 4, // 3→4 = +4
            5 => 4, // 4→5 = +4
            6 => 5, // 5→6 = +5 (max)
            _ => 0,
        };
        if marginal > best { best = marginal; }
    }
    best
}

// ─── Salmon D: per-salmon + per unique non-salmon adj animal, min 3 ───
fn salmon_d_setup(board: &Board, pos: usize, adj: &cascadia_core::hex::AdjacencyTable) -> u16 {
    let salmon_n: usize = adj.neighbors_of(pos)
        .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon))
        .count();
    if salmon_n > 2 { return 0; } // would create branching (invalid run)

    // Count unique non-salmon types adjacent to THIS position.
    let mut adj_mask = 0u8;
    for n in adj.neighbors_of(pos) {
        if let Some(w) = board.grid.get(n).placed_wildlife() {
            if w != Wildlife::Salmon { adj_mask |= 1 << (w as u8); }
        }
    }
    let adj_types = adj_mask.count_ones() as u16;

    if salmon_n == 0 {
        // Brand new singleton — doesn't qualify yet (need len ≥ 3).
        // Reward only if rich neighborhood AND extension room.
        let salmon_slots: usize = adj.neighbors_of(pos)
            .filter(|&n| {
                let c = board.grid.get(n);
                c.is_present() && c.can_place_wildlife(Wildlife::Salmon)
            }).count();
        if salmon_slots >= 2 && adj_types >= 1 { 2 + adj_types } else { 0 }
    } else if salmon_n == 1 {
        // Extending a chain. Run length contribution + adj-types-here.
        let run_len = count_salmon_run(board, adj.neighbors_of(pos)
            .find(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon)).unwrap(), adj);
        // Going from len N to N+1: +1 per-salmon + potentially new adj-types.
        let len_bonus = if run_len + 1 >= 3 { 3 } else { 1 };
        len_bonus + adj_types * 2
    } else { // salmon_n == 2 — middle of a run, doesn't extend
        adj_types
    }
}

// ─── Salmon B/C: same chain rule, different scoring tables ───
fn salmon_bc_setup(board: &Board, pos: usize, adj: &cascadia_core::hex::AdjacencyTable, variant: ScoringCardVariant) -> u16 {
    let salmon_n: usize = adj.neighbors_of(pos)
        .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon))
        .count();
    if salmon_n != 1 { return 0; } // only extending a chain endpoint scores
    let nbr = adj.neighbors_of(pos)
        .find(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon)).unwrap();
    let run_len = count_salmon_run(board, nbr, adj);
    use ScoringCardVariant::*;
    match (variant, run_len) {
        // Card B: 1=2/2=4/3=9/4=11/5+=17 — extending to len 3 unlocks +5, to len 5 unlocks +6
        (B, 0) | (B, 1) => 2,  // → 2/3
        (B, 2) => 5,           // 4 → 9 jump
        (B, 3) => 2,           // 9 → 11
        (B, 4) => 6,           // 11 → 17 jump
        (B, _) => 0,           // 5+ already capped
        // Card C: 3=10/4=12/5+=15 — len 2→3 unlocks 10pt, otherwise modest
        (C, 0) | (C, 1) => 1,  // sub-qualifying
        (C, 2) => 8,           // 0 → 10
        (C, 3) => 2,           // 10 → 12
        (C, 4) => 3,           // 12 → 15
        _ => 0,
    }
}

// ─── Hawk D: pairs of hawks with non-adjacent LOS, scored by intervening types ───
fn hawk_d_setup(board: &Board, pos: usize, adj: &cascadia_core::hex::AdjacencyTable) -> u16 {
    let coord = HexCoord::from_index(pos);
    // Adjacent hawk = LOS gets blocked / no pair.
    let has_adj_hawk = adj.neighbors_of(pos)
        .any(|n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Hawk));
    if has_adj_hawk { return 0; }

    let mut best: u16 = 0;
    // Walk each of 6 hex directions looking for an existing hawk on the same axis.
    for &(dq, dr) in &cascadia_core::hex::HexCoord::DIRECTIONS {
        let mut cur = HexCoord::new(coord.q + dq, coord.r + dr);
        let mut steps = 1u16;
        let mut types_mask = 0u8;
        loop {
            match cur.to_index() {
                Some(idx) => {
                    let c = board.grid.get(idx);
                    if c.placed_wildlife() == Some(Wildlife::Hawk) {
                        if steps >= 2 {
                            let unique = (types_mask & !(1 << Wildlife::Hawk as u8)).count_ones();
                            let pair_pts: u16 = match unique {
                                0 => 1, // pair exists but no animals between → 0 pts under D, but still creates LOS pair
                                1 => 5, // 4 pts under D
                                2 => 7, // 7 pts under D
                                _ => 9, // 9 pts under D
                            };
                            if pair_pts > best { best = pair_pts; }
                        }
                        break;
                    }
                    if let Some(w) = c.placed_wildlife() {
                        types_mask |= 1 << (w as u8);
                    }
                }
                None => break,
            }
            cur = HexCoord::new(cur.q + dq, cur.r + dr);
            steps += 1;
        }
    }
    best
}

// ─── Hawk B: count hawks with LOS to non-adjacent partner. 2=5,3=9,...,8=28 ───
fn hawk_b_setup(board: &Board, pos: usize, adj: &cascadia_core::hex::AdjacencyTable) -> u16 {
    // Same as Hawk D but doesn't care about intervening types — just LOS existence.
    let coord = HexCoord::from_index(pos);
    let has_adj_hawk = adj.neighbors_of(pos)
        .any(|n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Hawk));
    if has_adj_hawk { return 0; }
    for &(dq, dr) in &cascadia_core::hex::HexCoord::DIRECTIONS {
        let mut cur = HexCoord::new(coord.q + dq, coord.r + dr);
        let mut steps = 1u16;
        loop {
            match cur.to_index() {
                Some(idx) => {
                    if board.grid.get(idx).placed_wildlife() == Some(Wildlife::Hawk) {
                        if steps >= 2 { return 5; }
                        break;
                    }
                }
                None => break,
            }
            cur = HexCoord::new(cur.q + dq, cur.r + dr);
            steps += 1;
        }
    }
    0
}

// ─── Hawk C: 3 pts per non-adjacent LOS pair ───
fn hawk_c_setup(board: &Board, pos: usize, adj: &cascadia_core::hex::AdjacencyTable) -> u16 {
    // Count pairs we'd form. Just one is +3, two is +6, etc.
    let coord = HexCoord::from_index(pos);
    let has_adj_hawk = adj.neighbors_of(pos)
        .any(|n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Hawk));
    if has_adj_hawk { return 0; }
    let mut pairs = 0u16;
    for &(dq, dr) in &cascadia_core::hex::HexCoord::DIRECTIONS {
        let mut cur = HexCoord::new(coord.q + dq, coord.r + dr);
        let mut steps = 1u16;
        loop {
            match cur.to_index() {
                Some(idx) => {
                    if board.grid.get(idx).placed_wildlife() == Some(Wildlife::Hawk) {
                        if steps >= 2 { pairs += 1; }
                        break;
                    }
                }
                None => break,
            }
            cur = HexCoord::new(cur.q + dq, cur.r + dr);
            steps += 1;
        }
    }
    pairs * 3
}

// ─── Fox B: per fox, count of types with ≥2 of same type adjacent (1=3,2=5,3+=7) ───
fn fox_b_setup(board: &Board, pos: usize, adj: &cascadia_core::hex::AdjacencyTable) -> u16 {
    // Wildlife placement context: when we place a fox here, what's the pair-type count
    // among its neighbors RIGHT NOW.
    let mut counts = [0u8; 5];
    for n in adj.neighbors_of(pos) {
        if let Some(w) = board.grid.get(n).placed_wildlife() {
            if w != Wildlife::Fox { counts[w as usize] += 1; }
        }
    }
    let pair_types = counts.iter().filter(|&&c| c >= 2).count() as u16;
    let single_types: u16 = counts.iter().filter(|&&c| c == 1).count() as u16; // potential if extended
    let immediate_score: u16 = match pair_types { 0 => 0, 1 => 3, 2 => 5, _ => 7 };
    immediate_score + single_types // potential pair-types if neighborhood extends
}

// ─── Fox C: max single-type count adjacent (range 0-6) ───
fn fox_c_setup(board: &Board, pos: usize, adj: &cascadia_core::hex::AdjacencyTable) -> u16 {
    let mut counts = [0u8; 5];
    for n in adj.neighbors_of(pos) {
        if let Some(w) = board.grid.get(n).placed_wildlife() {
            if w != Wildlife::Fox { counts[w as usize] += 1; }
        }
    }
    *counts.iter().max().unwrap_or(&0) as u16
}

// ─── Fox D: pair of foxes scoring by union pair-types in 8 surrounding cells ───
fn fox_d_setup(board: &Board, pos: usize, adj: &cascadia_core::hex::AdjacencyTable) -> u16 {
    // Reward placement adjacent to an existing fox (forms a pair).
    let fox_n = adj.neighbors_of(pos)
        .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Fox))
        .count();
    if fox_n == 0 { return 0; }
    // Union of 8 cells around the pair: nbrs of self + nbrs of fox neighbor minus the two fox cells.
    let other_fox = adj.neighbors_of(pos)
        .find(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Fox)).unwrap();
    let mut counts = [0u8; 5];
    let mut seen = [false; 441];
    for src in [pos, other_fox] {
        for n in adj.neighbors_of(src) {
            if n == pos || n == other_fox { continue; }
            if seen[n] { continue; }
            seen[n] = true;
            if let Some(w) = board.grid.get(n).placed_wildlife() {
                if w != Wildlife::Fox { counts[w as usize] += 1; }
            }
        }
    }
    let pair_types = counts.iter().filter(|&&c| c >= 2).count() as u16;
    match pair_types { 0 => 2, 1 => 5, 2 => 7, 3 => 9, _ => 11 }
}

// ─── Slot-value helpers per variant family ───

fn bear_slot_value_alt(board: &Board, idx: usize, variant: ScoringCardVariant, adj: &cascadia_core::hex::AdjacencyTable) -> u16 {
    use ScoringCardVariant::*;
    // Slot adjacent to a bear → potential to build group. Variant determines which sizes are good.
    let bears: usize = adj.neighbors_of(idx)
        .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Bear)).count();
    match (variant, bears) {
        (C, 1) => 3, (C, 2) => 4, (C, _) => 0,
        (D, 1) => 4, (D, 2) => 5, (D, _) => 0,
        (B, 2) => 6, // pair → triple potential
        (B, _) => 0,
        (A, _) => 0, // shouldn't reach here (caller dispatches)
    }
}

fn elk_slot_value_alt(board: &Board, idx: usize, variant: ScoringCardVariant, adj: &cascadia_core::hex::AdjacencyTable) -> u16 {
    use ScoringCardVariant::*;
    let elks: usize = adj.neighbors_of(idx)
        .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Elk)).count();
    match variant {
        B => if elks >= 2 { 5 } else if elks == 1 { 2 } else { 0 },
        C => if elks >= 1 { 3 + elks.min(3) as u16 } else { 0 },
        D => if elks >= 1 { 4 } else { 0 },
        A => 0,
    }
}

fn salmon_slot_value_d(board: &Board, idx: usize, adj: &cascadia_core::hex::AdjacencyTable) -> u16 {
    // Slot adjacent to salmon AND with non-salmon types nearby = great Salmon D placement target.
    let has_salmon = adj.neighbors_of(idx).any(|n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon));
    if !has_salmon { return 0; }
    let mut mask = 0u8;
    for n in adj.neighbors_of(idx) {
        if let Some(w) = board.grid.get(n).placed_wildlife() {
            if w != Wildlife::Salmon { mask |= 1 << (w as u8); }
        }
    }
    3 + mask.count_ones() as u16
}

fn hawk_slot_value_alt(board: &Board, idx: usize, variant: ScoringCardVariant, adj: &cascadia_core::hex::AdjacencyTable) -> u16 {
    use ScoringCardVariant::*;
    // For Hawk B/C/D, a slot is valuable if it's on a hex axis with an existing hawk
    // (so a hawk placed here would form a LOS pair).
    let coord = HexCoord::from_index(idx);
    let has_adj_hawk = adj.neighbors_of(idx)
        .any(|n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Hawk));
    if has_adj_hawk { return 0; }

    let mut score: u16 = 0;
    for &(dq, dr) in &cascadia_core::hex::HexCoord::DIRECTIONS {
        let mut cur = HexCoord::new(coord.q + dq, coord.r + dr);
        let mut steps = 1u16;
        let mut types_mask = 0u8;
        loop {
            match cur.to_index() {
                Some(c_idx) => {
                    if board.grid.get(c_idx).placed_wildlife() == Some(Wildlife::Hawk) {
                        if steps >= 2 {
                            let bonus: u16 = match variant {
                                D => {
                                    let unique = (types_mask & !(1 << Wildlife::Hawk as u8)).count_ones();
                                    match unique { 0 => 1, 1 => 4, 2 => 6, _ => 8 }
                                }
                                B => 4,
                                C => 3,
                                A => 0,
                            };
                            if bonus > score { score = bonus; }
                        }
                        break;
                    }
                    if let Some(w) = board.grid.get(c_idx).placed_wildlife() {
                        types_mask |= 1 << (w as u8);
                    }
                }
                None => break,
            }
            cur = HexCoord::new(cur.q + dq, cur.r + dr);
            steps += 1;
        }
    }
    score
}

fn fox_slot_value_b(board: &Board, idx: usize, adj: &cascadia_core::hex::AdjacencyTable) -> u16 {
    // Slot is valuable for Fox B if there are pair-type neighbors already (a fox here gets immediate score).
    let mut counts = [0u8; 5];
    for n in adj.neighbors_of(idx) {
        if let Some(w) = board.grid.get(n).placed_wildlife() {
            if w != Wildlife::Fox { counts[w as usize] += 1; }
        }
    }
    let pairs = counts.iter().filter(|&&c| c >= 2).count() as u16;
    match pairs { 0 => 0, 1 => 3, 2 => 5, _ => 7 }
}
