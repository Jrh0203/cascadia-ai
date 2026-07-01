use rand::Rng;

use cascadia_core::board::Board;
use cascadia_core::game::GameState;
use cascadia_core::hex::HexCoord;
use cascadia_core::hex::ADJACENCY;
use cascadia_core::scoring::ScoreBreakdown;
use cascadia_core::types::Wildlife;

use crate::eval::{best_move_with_potential, ScoredMove, EVAL_SCALE};
use crate::potential::{board_potential, board_potential_after_single_move, BoardPotentialContext};

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
/// PERF: the result is memoized in a thread-local single-entry cache keyed by
/// the exact valid-game inputs that affect candidate generation. MCE evaluates
/// many rollout states with identical public boards and markets but different
/// hidden bag order, so this removes repeated candidate generation without
/// changing any move. Set `CASCADIA_MCE_CACHE=0` for performance A/B tests.
pub fn candidate_moves_pub(game: &GameState) -> Vec<ScoredMove> {
    candidate_moves_with_base_pub(game).moves
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CandidateMoveSet {
    pub(crate) moves: Vec<ScoredMove>,
    pub(crate) base_move: Option<ScoredMove>,
}

pub(crate) fn candidate_moves_with_base_pub(game: &GameState) -> CandidateMoveSet {
    use std::cell::RefCell;
    if !candidate_cache_enabled() {
        return candidate_move_set(game);
    }
    thread_local! {
        static CACHE: RefCell<Option<(CandidateCacheKey, CandidateMoveSet)>> =
            const { RefCell::new(None) };
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
        let result = candidate_move_set(game);
        *c.borrow_mut() = Some((key, result.clone()));
        result
    })
}

pub(crate) fn candidate_moves_with_base_uncached_pub(game: &GameState) -> CandidateMoveSet {
    candidate_move_set(game)
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
            .unwrap_or(true);
        cell.set(if enabled { 1 } else { 0 });
        enabled
    })
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct CandidateCacheKey {
    words: arrayvec::ArrayVec<u64, 80>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct CandidateBoardCacheKey {
    words: arrayvec::ArrayVec<u64, 72>,
}

/// Capture the exact subset of a valid game state read by `candidate_moves`.
///
/// This deliberately stores values instead of a digest: a cache collision
/// cannot alter play. Tile rotations are included because habitat connectivity
/// and therefore candidate ordering depend on edge orientation.
pub(crate) fn candidate_cache_key(game: &GameState) -> CandidateCacheKey {
    let mut words = arrayvec::ArrayVec::new();
    words.push(game.current_player as u64);
    words.push(game.num_players as u64);
    words.push(game.turns_remaining as u64);

    for slot in &game.market.pairs {
        match slot {
            Some(p) => {
                let terrain2 = p.tile.terrain2.map(|t| t as u64 + 1).unwrap_or(0);
                words.push(
                    1 | ((p.tile.terrain1 as u64) << 1)
                        | (terrain2 << 4)
                        | ((p.tile.keystone as u64) << 7)
                        | ((p.tile.allowed.0 as u64) << 8)
                        | ((p.wildlife as u64) << 16),
                );
            }
            None => words.push(0),
        }
    }

    words.extend(candidate_board_cache_key(game).words);
    CandidateCacheKey { words }
}

/// Capture the exact board-and-scoring subset of candidate generation.
///
/// This excludes the market so diagnostics can quantify how much expensive
/// board analysis is repeated across stochastic public markets.
pub(crate) fn candidate_board_cache_key(game: &GameState) -> CandidateBoardCacheKey {
    let mut words = arrayvec::ArrayVec::new();
    let board = &game.boards[game.current_player];
    words.push(
        (board.nature_tokens as u64)
            | ((board.tile_count as u64) << 8)
            | ((board.largest_group[0] as u64) << 16)
            | ((board.largest_group[1] as u64) << 24)
            | ((board.largest_group[2] as u64) << 32)
            | ((board.largest_group[3] as u64) << 40)
            | ((board.largest_group[4] as u64) << 48),
    );
    for &tile_idx in &board.placed_tiles {
        let idx = tile_idx as usize;
        words.push(
            (tile_idx as u64)
                | ((board.grid.get(idx).0 as u64) << 16)
                | ((board.rotations[idx] as u64) << 32),
        );
    }
    // Some legacy wildlife scorers use stable, insertion-ordered traversals
    // when equally scoring structures are available. Preserve that order in
    // the cache key even though the visible grid alone contains the same
    // wildlife identities.
    for positions in &board.wildlife_positions {
        words.push(positions.len() as u64);
        for &position in positions {
            words.push(position as u64);
        }
    }

    let mut scoring = 0u64;
    for (index, variant) in game.scoring_cards.cards.iter().enumerate() {
        scoring |= (*variant as u64) << (index * 3);
    }
    words.push(scoring);

    CandidateBoardCacheKey { words }
}

/// Derive the exact frontier after placing `placed_idx` from the frontier
/// before placement. `Board::place_tile` appends the tile to `placed_tiles`,
/// so this also reproduces `Board::frontier` ordering.
#[cfg(test)]
fn frontier_after_tile_placement(
    board: &Board,
    frontier_before: &[u16],
    placed_idx: usize,
) -> arrayvec::ArrayVec<u16, 128> {
    let mut frontier = arrayvec::ArrayVec::new();
    for &idx in frontier_before {
        if idx as usize != placed_idx {
            frontier.push(idx);
        }
    }
    for neighbor in ADJACENCY.neighbors_of(placed_idx) {
        let neighbor = neighbor as u16;
        if !board.grid.get(neighbor as usize).is_present() && !frontier.contains(&neighbor) {
            frontier.push(neighbor);
        }
    }
    frontier
}

fn candidate_moves(game: &GameState) -> Vec<ScoredMove> {
    candidate_move_set(game).moves
}

#[derive(Clone, Copy)]
struct PotentialTileAction {
    index: usize,
}

fn place_potential_tile(
    board: &mut Board,
    index: usize,
    tile: cascadia_core::types::TileData,
) -> PotentialTileAction {
    assert!(
        !board.grid.get(index).is_present(),
        "potential-only tile placement requires an empty cell"
    );
    board.grid.set(index, tile.to_cell());
    board.placed_tiles.push(index as u16);
    PotentialTileAction { index }
}

fn undo_potential_tile(board: &mut Board, action: PotentialTileAction) {
    let removed = board
        .placed_tiles
        .pop()
        .expect("potential-only tile placement was appended");
    assert_eq!(removed as usize, action.index);
    board
        .grid
        .set(action.index, cascadia_core::types::Cell::EMPTY);
}

fn candidate_move_set(game: &GameState) -> CandidateMoveSet {
    candidate_move_set_impl::<true>(game)
}

#[cfg(test)]
fn candidate_move_set_reference(game: &GameState) -> CandidateMoveSet {
    candidate_move_set_impl::<false>(game)
}

#[inline(always)]
fn local_outcome_buffer<T: Copy, const REUSE_SHARED_OUTCOMES: bool>(len: usize) -> Vec<Option<T>> {
    if REUSE_SHARED_OUTCOMES {
        Vec::new()
    } else {
        vec![None; len]
    }
}

fn candidate_move_set_impl<const REUSE_SHARED_OUTCOMES: bool>(
    game: &GameState,
) -> CandidateMoveSet {
    let board = &game.boards[game.current_player];
    let cards = &game.scoring_cards;
    let frontier = board.frontier();
    if frontier.is_empty() {
        return CandidateMoveSet {
            moves: Vec::new(),
            base_move: None,
        };
    }

    let market_pairs: arrayvec::ArrayVec<_, 4> = game
        .market
        .available()
        .map(|(i, pair)| (i, pair.tile, pair.wildlife))
        .collect();
    if market_pairs.is_empty() {
        return CandidateMoveSet {
            moves: Vec::new(),
            base_move: None,
        };
    }

    let has_tokens = board.nature_tokens > 0;
    let base_wildlife_scores = cascadia_core::scoring::wildlife::score_all_wildlife(board, cards);
    let base_wildlife: u16 = base_wildlife_scores.iter().sum();
    let potential_context = BoardPotentialContext::new(board, cards, &frontier);

    #[derive(Clone, Copy)]
    struct Combo {
        tile_idx: usize,
        tile: cascadia_core::types::TileData,
        wildlife: cascadia_core::types::Wildlife,
        wl_market_idx: Option<usize>,
    }

    #[derive(Clone, Copy)]
    struct TilePlacement {
        index: u16,
        q: i8,
        r: i8,
        rot: u8,
        hab: u16,
    }

    #[derive(Clone, Copy)]
    struct ExistingWildlifePlacement {
        value: u16,
        q: i8,
        r: i8,
    }

    #[derive(Clone, Copy)]
    struct RotationInvariantOutcome {
        wildlife_value: u16,
        wildlife_q: Option<i8>,
        wildlife_r: Option<i8>,
        potential: i32,
    }

    struct RotationInvariantOutcomeCache {
        tile_idx: usize,
        wildlife: Wildlife,
        outcomes: Box<[Option<RotationInvariantOutcome>]>,
    }

    let mut combos = arrayvec::ArrayVec::<Combo, 16>::new();
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
    let mut frontier_positions = [u8::MAX; 441];
    for (position, &index) in frontier.iter().enumerate() {
        frontier_positions[index as usize] = position as u8;
    }
    let mut placements_by_market: [Vec<TilePlacement>; 4] = std::array::from_fn(|_| Vec::new());
    for &(market_index, tile, _) in &market_pairs {
        let max_rotation = if tile.terrain2.is_none() { 1 } else { 6 };
        let placements = &mut placements_by_market[market_index];
        for &frontier_index in &frontier {
            let coord = HexCoord::from_index(frontier_index as usize);
            for rotation in 0..max_rotation {
                let habitat_score = board
                    .preview_habitat_total_at_index(frontier_index as usize, tile, rotation)
                    .expect("candidate frontier placement remains legal");
                placements.push(TilePlacement {
                    index: frontier_index,
                    q: coord.q,
                    r: coord.r,
                    rot: rotation,
                    hab: habitat_score,
                });
            }
        }
        placements.sort_by(|left, right| right.hab.cmp(&left.hab));
        placements.truncate(128);
    }

    let mut moves = Vec::with_capacity(combos.len());
    let derive_base_move = !greedy_potential_enabled();
    let mut base_move = None;
    let mut board_clone = board.clone();
    let mut shared_outcome_caches = arrayvec::ArrayVec::<RotationInvariantOutcomeCache, 16>::new();
    let mut best_existing_wildlife: [Option<ExistingWildlifePlacement>; 5] = [None; 5];
    for wildlife in Wildlife::ALL {
        let variant = cards.variant_for(wildlife);
        let without = base_wildlife_scores[wildlife as usize];
        for &tile_index in &board.placed_tiles {
            if !board
                .grid
                .get(tile_index as usize)
                .can_place_wildlife(wildlife)
            {
                continue;
            }
            let with = cascadia_core::scoring::wildlife::score_wildlife_after_placement(
                &mut board_clone,
                wildlife,
                variant,
                tile_index as usize,
            );
            let nature_bonus = u16::from(board.grid.get(tile_index as usize).is_keystone());
            let value = with.saturating_sub(without) + nature_bonus;
            if value == 0
                || best_existing_wildlife[wildlife as usize]
                    .is_some_and(|current| current.value >= value)
            {
                continue;
            }
            let coord = HexCoord::from_index(tile_index as usize);
            best_existing_wildlife[wildlife as usize] = Some(ExistingWildlifePlacement {
                value,
                q: coord.q,
                r: coord.r,
            });
        }
    }

    for combo in &combos {
        let is_independent = combo.wl_market_idx.is_some();
        let effective_nature = if is_independent {
            (board.nature_tokens as u16).saturating_sub(1)
        } else {
            board.nature_tokens as u16
        };

        // Habitat previews depend only on the drafted tile, not the wildlife
        // paired with it. Reuse the exact sorted top-K placements for all
        // normal and independent-draft combinations of the same market tile.
        let placements = &placements_by_market[combo.tile_idx];
        if placements.is_empty() {
            continue;
        }

        // For each top tile placement, jointly find best wildlife placement
        let mut best_total: u16 = 0;
        let mut best_eval: i32 = i32::MIN;
        let mut best_tq: i8 = 0;
        let mut best_tr: i8 = 0;
        let mut best_rot: u8 = 0;
        let mut best_wq: Option<i8> = None;
        let mut best_wr: Option<i8> = None;
        let mut found = false;
        let mut local_outcomes_by_coordinate =
            local_outcome_buffer::<RotationInvariantOutcome, REUSE_SHARED_OUTCOMES>(frontier.len());
        let outcomes_by_coordinate = if REUSE_SHARED_OUTCOMES {
            let cache_index = match shared_outcome_caches.iter().position(|cache| {
                cache.tile_idx == combo.tile_idx && cache.wildlife == combo.wildlife
            }) {
                Some(index) => index,
                None => {
                    shared_outcome_caches.push(RotationInvariantOutcomeCache {
                        tile_idx: combo.tile_idx,
                        wildlife: combo.wildlife,
                        outcomes: vec![None; frontier.len()].into_boxed_slice(),
                    });
                    shared_outcome_caches.len() - 1
                }
            };
            shared_outcome_caches[cache_index].outcomes.as_mut()
        } else {
            &mut local_outcomes_by_coordinate
        };

        for (placement_rank, placement) in placements.iter().enumerate() {
            let tile_idx = placement.index as usize;
            let frontier_position = frontier_positions[tile_idx] as usize;
            debug_assert!(frontier_position < frontier.len());
            let outcome = if let Some(outcome) = outcomes_by_coordinate[frontier_position] {
                outcome
            } else {
                let tile_action = place_potential_tile(&mut board_clone, tile_idx, combo.tile);
                let variant = cards.variant_for(combo.wildlife);
                let without = base_wildlife_scores[combo.wildlife as usize];
                let mut wildlife_value = 0;
                let mut wildlife_q = None;
                let mut wildlife_r = None;

                if let Some(existing) = best_existing_wildlife[combo.wildlife as usize] {
                    wildlife_value = existing.value;
                    wildlife_q = Some(existing.q);
                    wildlife_r = Some(existing.r);
                }

                if board_clone
                    .grid
                    .get(tile_idx)
                    .can_place_wildlife(combo.wildlife)
                {
                    let with = cascadia_core::scoring::wildlife::score_wildlife_after_placement(
                        &mut board_clone,
                        combo.wildlife,
                        variant,
                        tile_idx,
                    );

                    let value = with.saturating_sub(without)
                        + u16::from(board_clone.grid.get(tile_idx).is_keystone());
                    if value > wildlife_value {
                        wildlife_value = value;
                        let wildlife_coord = HexCoord::from_index(tile_idx);
                        wildlife_q = Some(wildlife_coord.q);
                        wildlife_r = Some(wildlife_coord.r);
                    }
                }

                let potential = if let (Some(q), Some(r)) = (wildlife_q, wildlife_r) {
                    let wildlife_index = HexCoord::new(q, r).to_index().unwrap();
                    let wildlife_action =
                        board_clone.place_wildlife(wildlife_index, combo.wildlife);
                    let potential = board_potential_after_single_move(
                        &board_clone,
                        cards,
                        &potential_context,
                        tile_idx,
                        Some((wildlife_index, combo.wildlife)),
                    );
                    if let Some(wildlife_action) = wildlife_action {
                        board_clone.undo(wildlife_action);
                    }
                    potential
                } else {
                    board_potential_after_single_move(
                        &board_clone,
                        cards,
                        &potential_context,
                        tile_idx,
                        None,
                    )
                };
                undo_potential_tile(&mut board_clone, tile_action);

                let outcome = RotationInvariantOutcome {
                    wildlife_value,
                    wildlife_q,
                    wildlife_r,
                    potential,
                };
                outcomes_by_coordinate[frontier_position] = Some(outcome);
                outcome
            };

            let local_best_total =
                placement.hab + base_wildlife + effective_nature + outcome.wildlife_value;
            let local_eval = (local_best_total as i32) * EVAL_SCALE + outcome.potential;

            if derive_base_move && placement_rank < 8 {
                let base_eval = (local_best_total as i32) * EVAL_SCALE;
                if base_move.is_none_or(|current: ScoredMove| base_eval > current.eval) {
                    base_move = Some(ScoredMove {
                        market_index: combo.tile_idx,
                        tile_q: placement.q,
                        tile_r: placement.r,
                        rotation: placement.rot,
                        wildlife_q: outcome.wildlife_q,
                        wildlife_r: outcome.wildlife_r,
                        score: local_best_total,
                        eval: base_eval,
                        wildlife_market_index: combo.wl_market_idx,
                    });
                }
            }

            if !found || local_eval > best_eval {
                best_eval = local_eval;
                best_total = local_best_total;
                best_tq = placement.q;
                best_tr = placement.r;
                best_rot = placement.rot;
                best_wq = outcome.wildlife_q;
                best_wr = outcome.wildlife_r;
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
    let base_move = if derive_base_move {
        base_move
    } else {
        let mut fallback_board = board.clone();
        best_move_with_potential(
            &mut fallback_board,
            &market_pairs,
            cards,
            game.turns_remaining,
        )
    };
    CandidateMoveSet { moves, base_move }
}

fn greedy_potential_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CASCADIA_GREEDY_POTENTIAL")
            .ok()
            .is_some_and(|value| !value.is_empty() && value != "0")
    })
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
    let mp: arrayvec::ArrayVec<_, 4> = game
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

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_core::types::ScoringCards;
    use rand::{rngs::StdRng, SeedableRng};

    fn seeded_game() -> GameState {
        let mut rng = StdRng::seed_from_u64(0xc57ad1a);
        GameState::new(4, ScoringCards::all_a(), &mut rng)
    }

    #[test]
    fn candidate_cache_key_includes_tile_rotations() {
        let game = seeded_game();
        let mut rotated = game.clone();
        let tile_idx = rotated.boards[0].placed_tiles[0] as usize;
        rotated.boards[0].rotations[tile_idx] = (rotated.boards[0].rotations[tile_idx] + 1) % 6;

        assert!(candidate_cache_key(&game) != candidate_cache_key(&rotated));
    }

    #[test]
    fn candidate_cache_key_includes_wildlife_insertion_order() {
        let mut first = seeded_game();
        let positions = &mut first.boards[0].wildlife_positions[Wildlife::Elk as usize];
        positions.push(10);
        positions.push(20);
        let mut reversed = first.clone();
        reversed.boards[0].wildlife_positions[Wildlife::Elk as usize].swap(0, 1);

        assert!(candidate_cache_key(&first) != candidate_cache_key(&reversed));
    }

    #[test]
    fn derived_frontier_and_potential_match_full_recomputation() {
        let game = seeded_game();
        let board = &game.boards[0];
        let frontier = board.frontier();
        let context = BoardPotentialContext::new(board, &game.scoring_cards, &frontier);

        for (_, pair) in game.market.available() {
            let max_rotation = if pair.tile.terrain2.is_some() { 6 } else { 1 };
            for &frontier_idx in &frontier {
                for rotation in 0..max_rotation {
                    let mut placed = board.clone();
                    let coord = HexCoord::from_index(frontier_idx as usize);
                    placed.place_tile(coord, pair.tile, rotation).unwrap();

                    let derived =
                        frontier_after_tile_placement(&placed, &frontier, frontier_idx as usize);
                    let exact = placed.frontier();
                    assert_eq!(derived.as_slice(), exact.as_slice());
                    assert_eq!(
                        crate::potential::board_potential_with_frontier(
                            &placed,
                            &game.scoring_cards,
                            &derived,
                        ),
                        board_potential(&placed, &game.scoring_cards),
                    );
                    assert_eq!(
                        board_potential_after_single_move(
                            &placed,
                            &game.scoring_cards,
                            &context,
                            frontier_idx as usize,
                            None,
                        ),
                        board_potential(&placed, &game.scoring_cards),
                    );
                    for wildlife in Wildlife::ALL {
                        let Some(wildlife_index) = placed
                            .placed_tiles
                            .iter()
                            .copied()
                            .find(|&index| {
                                placed.grid.get(index as usize).can_place_wildlife(wildlife)
                            })
                            .map(usize::from)
                        else {
                            continue;
                        };
                        let action = placed.place_wildlife(wildlife_index, wildlife).unwrap();
                        assert_eq!(
                            board_potential_after_single_move(
                                &placed,
                                &game.scoring_cards,
                                &context,
                                frontier_idx as usize,
                                Some((wildlife_index, wildlife)),
                            ),
                            board_potential(&placed, &game.scoring_cards),
                        );
                        placed.undo(action);
                    }
                }
            }
        }
    }

    #[test]
    fn incremental_potential_matches_full_recomputation_during_play() {
        use cascadia_core::types::ScoringCardVariant::{A, B, C, D};

        let card_sets = [
            ScoringCards::all_a(),
            ScoringCards {
                cards: [D, C, B, D, A],
            },
        ];

        for (card_index, cards) in card_sets.into_iter().enumerate() {
            let mut rng = StdRng::seed_from_u64(0xc57ad1a + card_index as u64);
            let mut game = GameState::new(4, cards, &mut rng);
            for turn_index in 0..16 {
                let board = &game.boards[game.current_player];
                let frontier = board.frontier();
                let context = BoardPotentialContext::new(board, &game.scoring_cards, &frontier);

                for (_, pair) in game.market.available() {
                    let max_rotation = if pair.tile.terrain2.is_some() { 6 } else { 1 };
                    for &frontier_index in frontier.iter().take(3) {
                        for rotation in 0..max_rotation {
                            let mut placed = board.clone();
                            let mut potential_only = board.clone();
                            placed
                                .place_tile(
                                    HexCoord::from_index(frontier_index as usize),
                                    pair.tile,
                                    rotation,
                                )
                                .unwrap();
                            let potential_tile_action = place_potential_tile(
                                &mut potential_only,
                                frontier_index as usize,
                                pair.tile,
                            );
                            assert_eq!(
                                board_potential(&potential_only, &game.scoring_cards),
                                board_potential(&placed, &game.scoring_cards),
                            );
                            assert_eq!(
                                board_potential_after_single_move(
                                    &placed,
                                    &game.scoring_cards,
                                    &context,
                                    frontier_index as usize,
                                    None,
                                ),
                                board_potential(&placed, &game.scoring_cards),
                            );
                            assert_eq!(
                                board_potential_after_single_move(
                                    &potential_only,
                                    &game.scoring_cards,
                                    &context,
                                    frontier_index as usize,
                                    None,
                                ),
                                board_potential(&placed, &game.scoring_cards),
                            );
                            for wildlife in Wildlife::ALL {
                                let Some(wildlife_index) = placed
                                    .placed_tiles
                                    .iter()
                                    .copied()
                                    .find(|&index| {
                                        placed.grid.get(index as usize).can_place_wildlife(wildlife)
                                    })
                                    .map(usize::from)
                                else {
                                    continue;
                                };
                                let action =
                                    placed.place_wildlife(wildlife_index, wildlife).unwrap();
                                let potential_action = potential_only
                                    .place_wildlife(wildlife_index, wildlife)
                                    .unwrap();
                                assert_eq!(
                                    board_potential(&potential_only, &game.scoring_cards),
                                    board_potential(&placed, &game.scoring_cards),
                                );
                                assert_eq!(
                                    board_potential_after_single_move(
                                        &placed,
                                        &game.scoring_cards,
                                        &context,
                                        frontier_index as usize,
                                        Some((wildlife_index, wildlife)),
                                    ),
                                    board_potential(&placed, &game.scoring_cards),
                                    "incremental potential mismatch: card_set={card_index}, turn={turn_index}, frontier={frontier_index}, rotation={rotation}, wildlife={wildlife:?}, wildlife_index={wildlife_index}",
                                );
                                assert_eq!(
                                    board_potential_after_single_move(
                                        &potential_only,
                                        &game.scoring_cards,
                                        &context,
                                        frontier_index as usize,
                                        Some((wildlife_index, wildlife)),
                                    ),
                                    board_potential(&placed, &game.scoring_cards),
                                );
                                placed.undo(action);
                                potential_only.undo(potential_action);
                            }
                            undo_potential_tile(&mut potential_only, potential_tile_action);
                            assert_eq!(potential_only.placed_tiles, board.placed_tiles);
                            assert!(!potential_only
                                .grid
                                .get(frontier_index as usize)
                                .is_present());
                        }
                    }
                }

                let Some(movement) = greedy_move(&game) else {
                    break;
                };
                assert!(execute_scored_move(&mut game, &movement));
            }
        }
    }

    #[test]
    fn cached_candidates_match_uncached_generation() {
        let game = seeded_game();
        let expected = candidate_moves(&game);
        assert_eq!(candidate_moves_pub(&game), expected);
        assert_eq!(candidate_moves_pub(&game), expected);
    }

    #[test]
    fn shared_candidate_path_elides_the_dead_local_outcome_buffer() {
        assert!(local_outcome_buffer::<u8, true>(17).is_empty());
        assert_eq!(local_outcome_buffer::<u8, false>(17).len(), 17);
    }

    #[test]
    fn dead_local_outcome_buffer_elision_preserves_complete_candidate_sets() {
        use cascadia_core::types::ScoringCardVariant::{A, B, C, D};

        let card_sets = [
            ScoringCards::all_a(),
            ScoringCards {
                cards: [D, C, B, A, D],
            },
        ];
        for (card_index, cards) in card_sets.into_iter().enumerate() {
            for seed_offset in 0..2 {
                let mut rng =
                    StdRng::seed_from_u64(0xde1d_c700 + card_index as u64 * 16 + seed_offset);
                let mut game = GameState::new(4, cards, &mut rng);
                for turn in 0..28 {
                    if game.can_replace_overflow().is_some() {
                        game.replace_overflow();
                    }

                    let reference = candidate_move_set_reference(&game);
                    assert_eq!(
                        candidate_move_set(&game),
                        reference,
                        "shared candidate set diverged: cards={card_index}, seed={seed_offset}, turn={turn}"
                    );

                    let mut token_game = game.clone();
                    let player = token_game.current_player;
                    token_game.boards[player].nature_tokens =
                        token_game.boards[player].nature_tokens.max(1);
                    let token_reference = candidate_move_set_reference(&token_game);
                    assert_eq!(
                        candidate_move_set(&token_game),
                        token_reference,
                        "shared token candidate set diverged: cards={card_index}, seed={seed_offset}, turn={turn}"
                    );

                    let Some(movement) = reference
                        .base_move
                        .or_else(|| reference.moves.first().copied())
                    else {
                        break;
                    };
                    assert!(execute_scored_move(&mut game, &movement));
                }
            }
        }
    }

    #[test]
    fn cached_candidate_sets_match_direct_generation_across_complete_games() {
        for game_index in 0..4_u64 {
            let mut rng = StdRng::seed_from_u64(0xD1CE_C700_0000_0000 + game_index);
            let mut game = GameState::new(4, ScoringCards::all_a(), &mut rng);
            while !game.is_game_over() {
                if game.can_replace_overflow().is_some() {
                    game.replace_overflow();
                }
                let direct = candidate_moves_with_base_uncached_pub(&game);
                assert_eq!(candidate_moves_with_base_pub(&game), direct);
                let movement = direct.base_move.or_else(|| direct.moves.first().copied());
                let Some(movement) = movement else {
                    break;
                };
                assert!(execute_scored_move(&mut game, &movement));
            }
        }
    }

    #[test]
    fn combined_candidate_pass_preserves_greedy_base_move() {
        if std::env::var("CASCADIA_GREEDY_POTENTIAL")
            .ok()
            .is_some_and(|value| !value.is_empty() && value != "0")
        {
            return;
        }

        let mut game = seeded_game();
        for _ in 0..24 {
            let market = game
                .market
                .available()
                .map(|(index, pair)| (index, pair.tile, pair.wildlife))
                .collect::<Vec<_>>();
            let mut board = game.boards[game.current_player].clone();
            let expected = best_move_with_potential(
                &mut board,
                &market,
                &game.scoring_cards,
                game.turns_remaining,
            );
            let combined = candidate_move_set(&game);
            assert_eq!(combined.base_move, expected);

            let Some(movement) = expected else {
                break;
            };
            assert!(execute_scored_move(&mut game, &movement));
        }
    }

    #[test]
    fn shared_duplicate_wildlife_outcomes_match_per_combo_reference() {
        use cascadia_core::types::ScoringCardVariant::{A, B, C, D};

        let card_sets = [
            ScoringCards::all_a(),
            ScoringCards {
                cards: [D, C, B, A, D],
            },
        ];
        for (card_index, cards) in card_sets.into_iter().enumerate() {
            for seed_offset in 0..3 {
                let mut rng =
                    StdRng::seed_from_u64(0x5a4ed000 + card_index as u64 * 16 + seed_offset);
                let mut game = GameState::new(4, cards, &mut rng);
                for turn in 0..28 {
                    let expected = candidate_move_set_reference(&game);
                    let actual = candidate_move_set(&game);
                    assert_eq!(
                        actual, expected,
                        "candidate outcome reuse diverged: cards={card_index}, seed={seed_offset}, turn={turn}"
                    );

                    let Some(movement) = expected
                        .base_move
                        .or_else(|| expected.moves.first().copied())
                    else {
                        break;
                    };
                    assert!(execute_scored_move(&mut game, &movement));
                }
            }
        }
    }
}
