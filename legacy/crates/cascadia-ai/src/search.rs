use rand::Rng;
use std::hash::Hash;

use cascadia_core::game::GameState;
use cascadia_core::hex::HexCoord;
use cascadia_core::scoring::ScoreBreakdown;
use cascadia_core::types::ScoringCards;

use cascadia_core::board::Board;
use cascadia_core::hex::ADJACENCY;
use cascadia_core::types::Wildlife;

use crate::eval::{best_move_with_potential, ScoredMove, EVAL_SCALE};
use crate::potential::board_potential;

/// Compute a setup bonus for placing wildlife at a given position.
/// This is public so eval.rs can use it too.
/// Returns bonus in actual score points, rewarding placements that build
/// toward high-value patterns even when they don't score immediately.
pub fn wildlife_setup_bonus(board: &Board, pos: usize, wildlife: Wildlife) -> u16 {
    let adj = &*ADJACENCY;

    match wildlife {
        Wildlife::Bear => {
            // Bears: modest bonus for half-pair setup
            let bear_neighbors: usize = adj
                .neighbors_of(pos)
                .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Bear))
                .count();
            if bear_neighbors == 0 {
                let partner_slots: usize = adj
                    .neighbors_of(pos)
                    .filter(|&n| {
                        let cell = board.grid.get(n);
                        cell.is_present() && cell.can_place_wildlife(Wildlife::Bear)
                    })
                    .count();
                if partner_slots >= 1 {
                    1
                } else {
                    0
                }
            } else {
                0
            }
        }
        Wildlife::Elk => {
            // Elk lines: 2/5/9/13 for length 1/2/3/4.
            // Extending a line of 2→3 = +4 pts, 1→2 = +3 pts. Worth investing.
            let coord = HexCoord::from_index(pos);
            let mut bonus = 0u16;

            for &(dq, dr) in &HexCoord::LINE_DIRECTIONS {
                let fwd = HexCoord::new(coord.q + dq, coord.r + dr);
                let bwd = HexCoord::new(coord.q - dq, coord.r - dr);

                let fwd_elk = fwd.to_index().map_or(false, |idx| {
                    board.grid.get(idx).placed_wildlife() == Some(Wildlife::Elk)
                });
                let bwd_elk = bwd.to_index().map_or(false, |idx| {
                    board.grid.get(idx).placed_wildlife() == Some(Wildlife::Elk)
                });

                if fwd_elk && bwd_elk {
                    // In the MIDDLE of a line — very valuable (extends in both directions)
                    bonus = bonus.max(3);
                } else if fwd_elk || bwd_elk {
                    let other_dir = if fwd_elk { bwd } else { fwd };
                    let can_grow = other_dir.to_index().map_or(false, |idx| {
                        let cell = board.grid.get(idx);
                        cell.is_present() && cell.can_place_wildlife(Wildlife::Elk)
                    });
                    if can_grow {
                        bonus = bonus.max(2); // extendable endpoint
                    } else {
                        bonus = bonus.max(1); // line building but capped
                    }
                }
            }
            bonus
        }
        Wildlife::Salmon => {
            // Salmon runs: 2/4/7/11/15/20/26 for length 1-7.
            // Extending runs is very valuable — run of 5→6 = +5, 6→7 = +6.
            let salmon_neighbors: usize = adj
                .neighbors_of(pos)
                .filter(|&n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Salmon))
                .count();

            if salmon_neighbors == 1 {
                // Extending a run endpoint
                let extension_slots: usize = adj
                    .neighbors_of(pos)
                    .filter(|&n| {
                        let cell = board.grid.get(n);
                        cell.is_present()
                            && cell.can_place_wildlife(Wildlife::Salmon)
                            && cell.placed_wildlife() != Some(Wildlife::Salmon)
                    })
                    .count();
                if extension_slots >= 2 {
                    3
                } else if extension_slots >= 1 {
                    2
                } else {
                    1
                }
            } else if salmon_neighbors == 0 {
                // Starting a new run — bonus if room to grow
                let slots: usize = adj
                    .neighbors_of(pos)
                    .filter(|&n| {
                        let cell = board.grid.get(n);
                        cell.is_present() && cell.can_place_wildlife(Wildlife::Salmon)
                    })
                    .count();
                if slots >= 2 {
                    1
                } else {
                    0
                }
            } else {
                0 // 2+ neighbors = internal to run, already scored
            }
        }
        Wildlife::Hawk => {
            // Hawks: 2/5/8/11/14/18/22/28 for 1-8 isolated.
            // Each isolated hawk is worth 2-4 marginal points. Always place isolated.
            let has_hawk_neighbor = adj
                .neighbors_of(pos)
                .any(|n| board.grid.get(n).placed_wildlife() == Some(Wildlife::Hawk));
            if !has_hawk_neighbor {
                2
            } else {
                0
            }
        }
        Wildlife::Fox => {
            // Fox scores per unique adjacent type. No setup bonus needed —
            // the immediate score delta already captures diversity.
            0
        }
    }
}

/// Collect all candidate moves for the current game state.
/// Public accessor for candidate_moves (used by NNUE re-ranking).
///
/// PERF: when env `CASCADIA_MCE_CACHE=1` is set, the result is memoized in a
/// thread-local single-entry cache keyed by a hash of the inputs that affect
/// candidate_moves output. Strict tie-out: same inputs → identical output.
/// Inside MCE rollouts at fixed-state nodes (e.g. the root across all
/// rollouts of one decision) this provides ~14% wall speedup.
pub fn candidate_moves_pub(game: &GameState) -> Vec<ScoredMove> {
    use std::cell::RefCell;
    if !candidate_cache_enabled() {
        return candidate_moves(game);
    }
    thread_local! {
        static CACHE: RefCell<Option<(u64, Vec<ScoredMove>)>> = const { RefCell::new(None) };
    }
    let key = candidate_cache_key(game);
    CACHE.with(|c| {
        {
            let cell = c.borrow();
            if let Some((k, v)) = cell.as_ref() {
                if *k == key {
                    return v.clone();
                }
            }
        }
        let result = candidate_moves(game);
        *c.borrow_mut() = Some((key, result.clone()));
        result
    })
}

#[inline(always)]
fn candidate_cache_enabled() -> bool {
    // Cached once per thread for hot-path speed.
    thread_local! {
        static ENABLED: std::cell::Cell<i8> = const { std::cell::Cell::new(-1) };
    }
    ENABLED.with(|cell| {
        let v = cell.get();
        if v >= 0 {
            return v == 1;
        }
        let enabled = std::env::var("CASCADIA_MCE_CACHE")
            .ok()
            .map(|s| !s.is_empty() && s != "0")
            .unwrap_or(false);
        cell.set(if enabled { 1 } else { 0 });
        enabled
    })
}

/// Hash the subset of game state that affects `candidate_moves` output.
/// candidate_moves reads: current player's board (grid + nature_tokens +
/// placed_tiles), market.pairs, scoring_cards, turns_remaining,
/// current_player, num_players (via has_tokens).
fn candidate_cache_key(game: &GameState) -> u64 {
    use std::hash::Hasher;
    // Simple FxHash-style fold; we don't need cryptographic strength, just
    // collision resistance for typical game states. Inline a small variant.
    let mut h: u64 = 0xcbf29ce484222325; // FNV-1a basis
    let mix = |h: &mut u64, v: u64| {
        *h ^= v;
        *h = h.wrapping_mul(0x100000001b3);
    };
    mix(&mut h, game.current_player as u64);
    mix(&mut h, game.num_players as u64);
    mix(&mut h, game.turns_remaining as u64);
    // Market pairs (4 slots). Each slot has Option<MarketSlotPair { tile, wildlife }>.
    for slot in &game.market.pairs {
        match slot {
            Some(p) => {
                mix(&mut h, 0x5a5a5a5a);
                mix(&mut h, p.tile.terrain1 as u64);
                mix(&mut h, p.tile.terrain2.map(|t| t as u64 + 1).unwrap_or(0));
                mix(&mut h, p.tile.keystone as u64);
                mix(&mut h, p.tile.allowed.0 as u64);
                mix(&mut h, p.wildlife as u64);
            }
            None => {
                mix(&mut h, 0xa5a5a5a5);
            }
        }
    }
    // Board: placed_tiles (positions placed in order matters for can_place check
    // via placed_tiles iteration in candidate_moves) plus per-cell wildlife
    // assignments, plus nature_tokens count.
    let board = &game.boards[game.current_player];
    mix(&mut h, board.nature_tokens as u64);
    for &tile_idx in &board.placed_tiles {
        mix(&mut h, tile_idx as u64);
        // Per-cell wildlife state affects candidate generation via
        // can_place_wildlife checks.
        let cell = board.grid.get(tile_idx as usize);
        mix(&mut h, cell.0 as u64);
    }
    // ScoringCards is constant per game; hash variants directly (5 of them,
    // each fits in a u64).
    for v in &game.scoring_cards.cards {
        mix(&mut h, *v as u64);
    }
    h
}

fn candidate_moves(game: &GameState) -> Vec<ScoredMove> {
    let board = &game.boards[game.current_player];
    let cards = &game.scoring_cards;
    let frontier = board.frontier();
    if frontier.is_empty() {
        return Vec::new();
    }

    let market_pairs: Vec<_> = game
        .market
        .available()
        .map(|(i, pair)| (i, pair.tile, pair.wildlife))
        .collect();
    if market_pairs.is_empty() {
        return Vec::new();
    }

    let has_tokens = board.nature_tokens > 0;
    let base_wildlife: u16 = cascadia_core::scoring::wildlife::score_all_wildlife(board, cards)
        .iter()
        .sum();

    struct Combo {
        tile_idx: usize,
        tile: cascadia_core::types::TileData,
        wildlife: cascadia_core::types::Wildlife,
        wl_market_idx: Option<usize>,
    }

    let mut combos = Vec::new();
    for &(idx, tile, wl) in &market_pairs {
        combos.push(Combo {
            tile_idx: idx,
            tile,
            wildlife: wl,
            wl_market_idx: None,
        });
    }
    if has_tokens {
        for &(ti, tile, _) in &market_pairs {
            for &(wi, _, wl) in &market_pairs {
                if ti != wi {
                    combos.push(Combo {
                        tile_idx: ti,
                        tile,
                        wildlife: wl,
                        wl_market_idx: Some(wi),
                    });
                }
            }
        }
    }

    let mut moves = Vec::new();
    let mut board_clone = board.clone();

    for combo in &combos {
        let is_independent = combo.wl_market_idx.is_some();
        let effective_nature = if is_independent {
            (board.nature_tokens as u16).saturating_sub(1)
        } else {
            board.nature_tokens as u16
        };
        let max_rot: u8 = if combo.tile.terrain2.is_none() { 1 } else { 6 };

        // Evaluate ALL tile placements — don't limit by habitat.
        // Wildlife-valuable placements often aren't habitat-optimal.
        const TOP_K: usize = 128; // effectively unlimited
        struct TilePlacement {
            q: i8,
            r: i8,
            rot: u8,
            hab: u16,
        }
        let mut placements: Vec<TilePlacement> = Vec::new();

        for &fi in frontier.iter() {
            let coord = HexCoord::from_index(fi as usize);
            for rot in 0..max_rot {
                let action = match board_clone.place_tile(coord, combo.tile, rot) {
                    Some(a) => a,
                    None => continue,
                };
                let hab: u16 = board_clone.largest_group.iter().sum();
                placements.push(TilePlacement {
                    q: coord.q,
                    r: coord.r,
                    rot,
                    hab,
                });
                board_clone.undo(action);
            }
        }
        if placements.is_empty() {
            continue;
        }
        placements.sort_by(|a, b| b.hab.cmp(&a.hab));
        placements.truncate(TOP_K);

        // For each top tile placement, jointly find best wildlife placement
        let mut best_total: u16 = 0;
        let mut best_eval: i32 = i32::MIN;
        let mut best_tq: i8 = 0;
        let mut best_tr: i8 = 0;
        let mut best_rot: u8 = 0;
        let mut best_wq: Option<i8> = None;
        let mut best_wr: Option<i8> = None;
        let mut found = false;

        for placement in &placements {
            let action = board_clone
                .place_tile(
                    HexCoord::new(placement.q, placement.r),
                    combo.tile,
                    placement.rot,
                )
                .unwrap();

            let variant = cards.variant_for(combo.wildlife);
            let without = cascadia_core::scoring::wildlife::score_wildlife(
                &board_clone,
                combo.wildlife,
                variant,
            );

            // Score with no wildlife placement
            let skip_score = placement.hab + base_wildlife + effective_nature;
            let mut local_best_total = skip_score;
            let mut local_best_wq: Option<i8> = None;
            let mut local_best_wr: Option<i8> = None;

            let placed_snapshot: arrayvec::ArrayVec<u16, 64> =
                board_clone.placed_tiles.iter().copied().collect();
            for &ti in placed_snapshot.iter() {
                if !board_clone
                    .grid
                    .get(ti as usize)
                    .can_place_wildlife(combo.wildlife)
                {
                    continue;
                }
                let wa = match board_clone.place_wildlife(ti as usize, combo.wildlife) {
                    Some(a) => a,
                    None => continue,
                };
                let with = cascadia_core::scoring::wildlife::score_wildlife(
                    &board_clone,
                    combo.wildlife,
                    variant,
                );
                board_clone.undo(wa);

                let delta = with.saturating_sub(without);
                let nat_bonus: u16 = if board_clone.grid.get(ti as usize).is_keystone() {
                    1
                } else {
                    0
                };
                let total = placement.hab + base_wildlife + delta + effective_nature + nat_bonus;
                if total > local_best_total {
                    local_best_total = total;
                    let wc = HexCoord::from_index(ti as usize);
                    local_best_wq = Some(wc.q);
                    local_best_wr = Some(wc.r);
                }
            }

            // Compute potential while the TILE is still placed.
            // Saves a redundant place_tile+undo cycle vs the original code.
            // Correctness: place+undo+place is equivalent to a single place for
            // board state (Board::undo fully restores merged UF groups).
            let potential = if local_best_wq.is_some() {
                let wc = HexCoord::new(local_best_wq.unwrap(), local_best_wr.unwrap());
                let wa = board_clone.place_wildlife(wc.to_index().unwrap(), combo.wildlife);
                let p = board_potential(&board_clone, cards);
                if let Some(wa) = wa {
                    board_clone.undo(wa);
                }
                p
            } else {
                board_potential(&board_clone, cards)
            };

            board_clone.undo(action);

            let local_eval = (local_best_total as i32) * EVAL_SCALE + potential;

            if !found || local_eval > best_eval {
                best_eval = local_eval;
                best_total = local_best_total;
                best_tq = placement.q;
                best_tr = placement.r;
                best_rot = placement.rot;
                best_wq = local_best_wq;
                best_wr = local_best_wr;
                found = true;
            }
        }
        if !found {
            continue;
        }

        moves.push(ScoredMove {
            market_index: combo.tile_idx,
            tile_q: best_tq,
            tile_r: best_tr,
            rotation: best_rot,
            wildlife_q: best_wq,
            wildlife_r: best_wr,
            score: best_total,
            eval: best_eval,
            wildlife_market_index: combo.wl_market_idx,
        });
    }

    // Sort by eval (score * 100 + potential) to consider setup value
    moves.sort_by(|a, b| b.eval.cmp(&a.eval));
    moves
}

/// Generate candidate moves using NNUE afterstate evaluation.
/// For each market combo, evaluates all placements with actual_score + NNUE(remaining)
/// instead of greedy habitat + wildlife delta.
pub fn candidate_moves_nnue(game: &GameState, net: &crate::nnue::NNUENetwork) -> Vec<ScoredMove> {
    let player = game.current_player;
    let board = &game.boards[player];
    let cards = &game.scoring_cards;
    let frontier = board.frontier();
    if frontier.is_empty() {
        return Vec::new();
    }

    let market_pairs: Vec<_> = game
        .market
        .available()
        .map(|(i, pair)| (i, pair.tile, pair.wildlife))
        .collect();
    if market_pairs.is_empty() {
        return Vec::new();
    }

    let has_tokens = board.nature_tokens > 0;
    let bag_info = crate::nnue::BagInfo::from_game(game);

    struct Combo {
        tile_idx: usize,
        tile: cascadia_core::types::TileData,
        wildlife: cascadia_core::types::Wildlife,
        wl_market_idx: Option<usize>,
    }

    let mut combos = Vec::new();
    for &(idx, tile, wl) in &market_pairs {
        combos.push(Combo {
            tile_idx: idx,
            tile,
            wildlife: wl,
            wl_market_idx: None,
        });
    }
    if has_tokens {
        for &(ti, tile, _) in &market_pairs {
            for &(wi, _, wl) in &market_pairs {
                if ti != wi {
                    combos.push(Combo {
                        tile_idx: ti,
                        tile,
                        wildlife: wl,
                        wl_market_idx: Some(wi),
                    });
                }
            }
        }
    }

    let mut moves = Vec::new();
    let mut board_clone = board.clone();

    for combo in &combos {
        let max_rot: u8 = if combo.tile.terrain2.is_none() { 1 } else { 6 };

        let mut best_eval = f32::NEG_INFINITY;
        let mut best_score: u16 = 0;
        let mut best_tq: i8 = 0;
        let mut best_tr: i8 = 0;
        let mut best_rot: u8 = 0;
        let mut best_wq: Option<i8> = None;
        let mut best_wr: Option<i8> = None;
        let mut found = false;

        for &fi in frontier.iter() {
            let coord = HexCoord::from_index(fi as usize);
            for rot in 0..max_rot {
                let tile_action = match board_clone.place_tile(coord, combo.tile, rot) {
                    Some(a) => a,
                    None => continue,
                };

                // Evaluate with no wildlife placement
                let actual =
                    cascadia_core::scoring::ScoreBreakdown::compute(&mut board_clone, cards).total;
                let remaining = net.evaluate_with_bag(&board_clone, &bag_info);
                let skip_eval = actual as f32 + remaining;

                if !found || skip_eval > best_eval {
                    best_eval = skip_eval;
                    best_score = actual;
                    best_tq = coord.q;
                    best_tr = coord.r;
                    best_rot = rot;
                    best_wq = None;
                    best_wr = None;
                    found = true;
                }

                // Try each wildlife placement
                let placed_snapshot: arrayvec::ArrayVec<u16, 64> =
                    board_clone.placed_tiles.iter().copied().collect();
                for &ti in placed_snapshot.iter() {
                    if !board_clone
                        .grid
                        .get(ti as usize)
                        .can_place_wildlife(combo.wildlife)
                    {
                        continue;
                    }
                    let wa = match board_clone.place_wildlife(ti as usize, combo.wildlife) {
                        Some(a) => a,
                        None => continue,
                    };

                    let wl_actual =
                        cascadia_core::scoring::ScoreBreakdown::compute(&mut board_clone, cards)
                            .total;
                    let wl_remaining = net.evaluate_with_bag(&board_clone, &bag_info);
                    let wl_eval = wl_actual as f32 + wl_remaining;

                    board_clone.undo(wa);

                    if wl_eval > best_eval {
                        best_eval = wl_eval;
                        best_score = wl_actual;
                        best_tq = coord.q;
                        best_tr = coord.r;
                        best_rot = rot;
                        let wc = HexCoord::from_index(ti as usize);
                        best_wq = Some(wc.q);
                        best_wr = Some(wc.r);
                    }
                }

                board_clone.undo(tile_action);
            }
        }

        if !found {
            continue;
        }

        moves.push(ScoredMove {
            market_index: combo.tile_idx,
            tile_q: best_tq,
            tile_r: best_tr,
            rotation: best_rot,
            wildlife_q: best_wq,
            wildlife_r: best_wr,
            score: best_score,
            eval: (best_eval * 1000.0) as i32,
            wildlife_market_index: combo.wl_market_idx,
        });
    }

    moves.sort_by(|a, b| b.eval.cmp(&a.eval));
    moves
}

/// Decomposed candidate generation: evaluate tile and wildlife placements INDEPENDENTLY
/// then combine additively. O(tiles + animals) instead of O(tiles × animals).
pub fn candidate_moves_decomposed(
    game: &GameState,
    net: &crate::nnue::NNUENetwork,
) -> Vec<ScoredMove> {
    let player = game.current_player;
    let board = &game.boards[player];
    let cards = &game.scoring_cards;
    let frontier = board.frontier();
    if frontier.is_empty() {
        return Vec::new();
    }

    let market_pairs: Vec<_> = game
        .market
        .available()
        .map(|(i, pair)| (i, pair.tile, pair.wildlife))
        .collect();
    if market_pairs.is_empty() {
        return Vec::new();
    }

    let has_tokens = board.nature_tokens > 0;
    let bag_info = crate::nnue::BagInfo::from_game(game);

    let base_actual =
        cascadia_core::scoring::ScoreBreakdown::compute(&mut board.clone(), cards).total as f32;
    let base_remaining = net.evaluate_with_bag(board, &bag_info);
    let base_total = base_actual + base_remaining;

    struct Combo {
        tile_idx: usize,
        tile: cascadia_core::types::TileData,
        wildlife: cascadia_core::types::Wildlife,
        wl_market_idx: Option<usize>,
    }

    let mut combos = Vec::new();
    for &(idx, tile, wl) in &market_pairs {
        combos.push(Combo {
            tile_idx: idx,
            tile,
            wildlife: wl,
            wl_market_idx: None,
        });
    }
    if has_tokens {
        for &(ti, tile, _) in &market_pairs {
            for &(wi, _, wl) in &market_pairs {
                if ti != wi {
                    combos.push(Combo {
                        tile_idx: ti,
                        tile,
                        wildlife: wl,
                        wl_market_idx: Some(wi),
                    });
                }
            }
        }
    }

    let mut all_moves = Vec::new();

    for combo in &combos {
        let max_rot: u8 = if combo.tile.terrain2.is_none() { 1 } else { 6 };

        // Phase 1: Score tile placements independently (~60 evals)
        let mut tile_scores: Vec<(i8, i8, u8, f32)> = Vec::new(); // (q, r, rot, delta)
        {
            let mut b = board.clone();
            for &fi in frontier.iter() {
                let coord = HexCoord::from_index(fi as usize);
                for rot in 0..max_rot {
                    let action = match b.place_tile(coord, combo.tile, rot) {
                        Some(a) => a,
                        None => continue,
                    };
                    let actual =
                        cascadia_core::scoring::ScoreBreakdown::compute(&mut b, cards).total as f32;
                    let remaining = net.evaluate_with_bag(&b, &bag_info);
                    tile_scores.push((coord.q, coord.r, rot, (actual + remaining) - base_total));
                    b.undo(action);
                }
            }
        }
        tile_scores.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        tile_scores.truncate(8);

        // Phase 2: Score wildlife placements independently on EXISTING board (~15 evals)
        let mut wl_scores: Vec<(usize, i8, i8, f32)> = Vec::new(); // (idx, q, r, delta)
        {
            let mut b = board.clone();
            for &ti in b.placed_tiles.clone().iter() {
                let idx = ti as usize;
                if !b.grid.get(idx).can_place_wildlife(combo.wildlife) {
                    continue;
                }
                let wa = match b.place_wildlife(idx, combo.wildlife) {
                    Some(a) => a,
                    None => continue,
                };
                let actual =
                    cascadia_core::scoring::ScoreBreakdown::compute(&mut b, cards).total as f32;
                let remaining = net.evaluate_with_bag(&b, &bag_info);
                let coord = HexCoord::from_index(idx);
                wl_scores.push((idx, coord.q, coord.r, (actual + remaining) - base_total));
                b.undo(wa);
            }
        }
        wl_scores.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
        wl_scores.truncate(5);

        // Phase 3: Combine top tiles × top wildlife (additive approximation)
        let mut best_eval = f32::NEG_INFINITY;
        let mut best_move: Option<ScoredMove> = None;

        for &(tq, tr, rot, t_delta) in &tile_scores {
            // Tile only (skip wildlife)
            if t_delta > best_eval {
                best_eval = t_delta;
                best_move = Some(ScoredMove {
                    market_index: combo.tile_idx,
                    tile_q: tq,
                    tile_r: tr,
                    rotation: rot,
                    wildlife_q: None,
                    wildlife_r: None,
                    score: 0,
                    eval: (t_delta * 1000.0) as i32,
                    wildlife_market_index: combo.wl_market_idx,
                });
            }
            // Tile + wildlife on existing slot (additive)
            for &(_widx, wq, wr, w_delta) in &wl_scores {
                let combined = t_delta + w_delta;
                if combined > best_eval {
                    best_eval = combined;
                    best_move = Some(ScoredMove {
                        market_index: combo.tile_idx,
                        tile_q: tq,
                        tile_r: tr,
                        rotation: rot,
                        wildlife_q: Some(wq),
                        wildlife_r: Some(wr),
                        score: 0,
                        eval: (combined * 1000.0) as i32,
                        wildlife_market_index: combo.wl_market_idx,
                    });
                }
            }
        }

        // Phase 4: Interaction term — wildlife on the NEWLY placed tile (~8 evals)
        {
            let mut b = board.clone();
            for &(tq, tr, rot, _t_delta) in &tile_scores {
                let tile_action = match b.place_tile(HexCoord::new(tq, tr), combo.tile, rot) {
                    Some(a) => a,
                    None => continue,
                };
                let tile_idx = HexCoord::new(tq, tr).to_index().unwrap();
                if b.grid.get(tile_idx).can_place_wildlife(combo.wildlife) {
                    if let Some(wa) = b.place_wildlife(tile_idx, combo.wildlife) {
                        let actual = cascadia_core::scoring::ScoreBreakdown::compute(&mut b, cards)
                            .total as f32;
                        let remaining = net.evaluate_with_bag(&b, &bag_info);
                        let combined = (actual + remaining) - base_total;
                        if combined > best_eval {
                            best_eval = combined;
                            let wc = HexCoord::from_index(tile_idx);
                            best_move = Some(ScoredMove {
                                market_index: combo.tile_idx,
                                tile_q: tq,
                                tile_r: tr,
                                rotation: rot,
                                wildlife_q: Some(wc.q),
                                wildlife_r: Some(wc.r),
                                score: 0,
                                eval: (combined * 1000.0) as i32,
                                wildlife_market_index: combo.wl_market_idx,
                            });
                        }
                        b.undo(wa);
                    }
                }
                b.undo(tile_action);
            }
        }

        if let Some(mv) = best_move {
            all_moves.push(mv);
        }
    }

    all_moves.sort_by(|a, b| b.eval.cmp(&a.eval));
    all_moves
}

/// Execute a ScoredMove on a GameState. Returns false if it fails.
pub fn execute_scored_move(game: &mut GameState, mv: &ScoredMove) -> bool {
    let tile_coord = HexCoord::new(mv.tile_q, mv.tile_r);
    let wildlife_placement = match (mv.wildlife_q, mv.wildlife_r) {
        (Some(wq), Some(wr)) => HexCoord::new(wq, wr).to_index(),
        _ => None,
    };

    if let Some(wmi) = mv.wildlife_market_index {
        game.execute_independent_move(
            mv.market_index,
            wmi,
            tile_coord,
            mv.rotation,
            wildlife_placement,
        )
    } else {
        game.execute_move(cascadia_core::game::PlayerMove {
            market_index: mv.market_index,
            tile_coord,
            rotation: mv.rotation,
            wildlife_placement,
        })
    }
}

/// Quick greedy move for any player (used for opponent simulation).
pub fn greedy_move(game: &GameState) -> Option<ScoredMove> {
    let mp: Vec<_> = game
        .market
        .available()
        .map(|(i, p)| (i, p.tile, p.wildlife))
        .collect();
    let cards = game.scoring_cards;
    let turns = game.turns_remaining;
    let mut board = game.boards[game.current_player].clone();
    best_move_with_potential(&mut board, &mp, &cards, turns)
}

/// Advance the game past non-AI players' turns using greedy strategy.
///
/// Each opponent takes the free 3-of-a-kind overflow replacement when available
/// before picking their move. This is strictly an improvement over the current
/// market state and any rational player would take it. Inference-time opponent
/// loops (CLI bench, cascadia-web, MCE rollouts, training data generation) all
/// use this same behavior to stay consistent.
pub fn advance_opponents(game: &mut GameState, ai_player: usize) {
    while !game.is_game_over() && game.current_player != ai_player {
        if game.can_replace_overflow().is_some() {
            game.replace_overflow();
        }
        match greedy_move(game) {
            Some(mv) => {
                if !execute_scored_move(game, &mv) {
                    break;
                }
            }
            None => break,
        }
    }
}

/// Get the final score for a player from a game state.
fn final_score(game: &mut GameState, player: usize) -> u16 {
    ScoreBreakdown::compute(&mut game.boards[player], &game.scoring_cards).total
}

// ─────────────────────────────────────────────────────────────────────
// Board evaluation for search
// ─────────────────────────────────────────────────────────────────────

/// Evaluate a board state for a given player. Uses same scale as EVAL_SCALE (100)
/// so that potential meaningfully contributes to decisions.
fn evaluate_board(game: &mut GameState, player: usize) -> i32 {
    let cards = game.scoring_cards;
    let score = ScoreBreakdown::compute(&mut game.boards[player], &cards).total;
    let potential = board_potential(&game.boards[player], &cards);
    (score as i32) * EVAL_SCALE + potential
}

// ─────────────────────────────────────────────────────────────────────
// Strategy 1: Beam Search with depth-limited lookahead
// ─────────────────────────────────────────────────────────────────────

/// Beam search: at each turn, keep the top-K game states and expand them
/// to depth D. Pick the move from the root that leads to the best state.
/// Shuffles bags at each depth to prevent future-peeking.
/// In multi-player games, simulates opponents with greedy play between AI turns.
///
/// beam_width: how many states to keep at each level (K)
/// depth: how many AI turns to look ahead (D)
pub fn best_move_beam(game: &GameState, beam_width: usize, depth: usize) -> Option<ScoredMove> {
    let player = game.current_player;

    struct BeamEntry {
        root_move: ScoredMove,
        game: GameState,
    }

    let candidates = candidate_moves(game);
    if candidates.is_empty() {
        return None;
    }

    // Create initial beam: execute each top-K candidate on the KNOWN market
    let mut beam: Vec<BeamEntry> = Vec::new();
    for mv in candidates.iter().take(beam_width) {
        let mut g = game.clone();
        if execute_scored_move(&mut g, mv) {
            // In multiplayer, simulate opponents between our turns
            if g.num_players > 1 {
                advance_opponents(&mut g, player);
            }
            beam.push(BeamEntry {
                root_move: *mv,
                game: g,
            });
        }
    }

    // Expand beam for remaining depth (each depth = one AI turn)
    for _d in 1..depth {
        let mut next_beam: Vec<(ScoredMove, GameState, i32)> = Vec::new();

        for entry in &beam {
            if entry.game.is_game_over() {
                let mut g = entry.game.clone();
                let score = final_score(&mut g, player) as i32 * EVAL_SCALE;
                next_beam.push((entry.root_move, entry.game.clone(), score));
                continue;
            }

            let cands = candidate_moves(&entry.game);
            for mv in cands.iter().take(beam_width.max(3)) {
                let mut g = entry.game.clone();
                if execute_scored_move(&mut g, mv) {
                    if g.num_players > 1 {
                        advance_opponents(&mut g, player);
                    }

                    let eval = if g.is_game_over() {
                        final_score(&mut g, player) as i32 * EVAL_SCALE
                    } else {
                        // Evaluate our board state + potential
                        evaluate_board(&mut g, player)
                    };
                    next_beam.push((entry.root_move, g, eval));
                }
            }
        }

        next_beam.sort_by(|a, b| b.2.cmp(&a.2));
        next_beam.truncate(beam_width);

        beam = next_beam
            .into_iter()
            .map(|(root, g, _)| BeamEntry {
                root_move: root,
                game: g,
            })
            .collect();
    }

    // Pick the root move with the best final evaluation
    let mut best: Option<(ScoredMove, i32)> = None;
    for entry in &beam {
        let mut g = entry.game.clone();
        let eval = if g.is_game_over() {
            final_score(&mut g, player) as i32 * EVAL_SCALE
        } else {
            evaluate_board(&mut g, player)
        };
        if best.is_none() || eval > best.unwrap().1 {
            best = Some((entry.root_move, eval));
        }
    }

    best.map(|(mv, eval)| ScoredMove {
        score: (eval / EVAL_SCALE) as u16,
        ..mv
    })
}

// ─────────────────────────────────────────────────────────────────────
// Strategy 2: Monte Carlo Rollouts
// ─────────────────────────────────────────────────────────────────────

/// Monte Carlo rollout: for each candidate move, play out the rest of the
/// game N times with shuffled bags and greedy play, and pick the move with
/// the highest average final score.
///
/// num_rollouts: how many complete games to simulate per candidate move
pub fn best_move_mcts(
    game: &GameState,
    num_rollouts: usize,
    rng: &mut impl Rng,
) -> Option<ScoredMove> {
    let player = game.current_player;
    let candidates = candidate_moves(game);
    if candidates.is_empty() {
        return None;
    }

    let max_candidates = 8;
    let mut best: Option<(ScoredMove, f64)> = None;

    for mv in candidates.iter().take(max_candidates) {
        let mut total_score: u64 = 0;
        let mut count = 0u32;

        for _ in 0..num_rollouts {
            let mut g = game.clone();
            if !execute_scored_move(&mut g, mv) {
                continue;
            }

            // Shuffle bags for each rollout so we explore different futures
            g.shuffle_bags(rng);

            // Play out the rest of the game with greedy moves for all players
            while !g.is_game_over() {
                match greedy_move(&g) {
                    Some(next_mv) => {
                        if !execute_scored_move(&mut g, &next_mv) {
                            break;
                        }
                    }
                    None => break,
                }
            }

            let score = ScoreBreakdown::compute(&mut g.boards[player], &g.scoring_cards).total;
            total_score += score as u64;
            count += 1;
        }

        if count == 0 {
            continue;
        }
        let avg = total_score as f64 / count as f64;

        if best.is_none() || avg > best.unwrap().1 {
            best = Some((*mv, avg));
        }
    }

    best.map(|(mv, avg)| ScoredMove {
        score: avg.round() as u16,
        ..mv
    })
}
