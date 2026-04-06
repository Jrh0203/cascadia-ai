//! Wildlife-strategic candidate generation.
//! For each wildlife type, find the best tile placement that extends
//! or sets up a valuable pattern, regardless of habitat score.
//! These candidates are injected alongside greedy candidates for MCE to evaluate.

use cascadia_core::board::Board;
use cascadia_core::game::GameState;
use cascadia_core::hex::{HexCoord, ADJACENCY};
use cascadia_core::types::Wildlife;

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
                    let strategic_value = pattern_setup_value(
                        &board_clone, ti as usize, wildlife, adj,
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
                let slot_value = new_slot_value(&board_clone, coord, adj);
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
