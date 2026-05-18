use cascadia_core::board::Board;
use cascadia_core::hex::HexCoord;
use cascadia_core::scoring::wildlife;
use cascadia_core::types::{ScoringCards, Wildlife};

use crate::potential::board_potential;
use crate::search::wildlife_setup_bonus;

/// A fully specified move.
#[derive(Debug, Clone, Copy)]
pub struct ScoredMove {
    pub market_index: usize,
    pub tile_q: i8,
    pub tile_r: i8,
    pub rotation: u8,
    pub wildlife_q: Option<i8>,
    pub wildlife_r: Option<i8>,
    pub score: u16,
    /// Internal evaluation score (score * SCALE + potential). Used for comparison only.
    pub eval: i32,
    /// If set, wildlife comes from a different market slot (costs a nature token).
    pub wildlife_market_index: Option<usize>,
}

/// Scale factor: real score is multiplied by this before adding potential.
/// Higher values = potential is more of a tiebreaker; lower = potential has more influence.
pub const EVAL_SCALE: i32 = 1000;

/// Wildlife ROI weights: how many points per placement each type yields on average
/// when building optimal patterns. Used to bias move selection toward high-ROI types.
/// Applied to wildlife delta in evaluation (not to actual score).
pub fn wildlife_roi_weight(w: Wildlife) -> f32 {
    match w {
        Wildlife::Salmon => 1.3,  // Best scaling: run of 7 = 3.71 pts/each
        Wildlife::Hawk => 1.15,   // Good scaling: 8 isolated = 3.5 pts/each
        Wildlife::Elk => 1.0,     // Decent: line of 4 = 3.25 pts/each
        Wildlife::Fox => 1.0,     // Decent: depends on diversity
        Wildlife::Bear => 0.7,    // Worst ROI unless pair completes: lone bear = 0 pts
    }
}

/// Find the move that maximizes the current player's score.
///
/// Considers both normal paired drafts AND independent drafts (spending a
/// nature token to pick any tile + any wildlife from different market slots).
///
/// Uses decomposed search: habitat and wildlife scoring are independent.
pub fn best_move(
    board: &mut Board,
    market: &[(usize, cascadia_core::types::TileData, Wildlife)],
    cards: &ScoringCards,
) -> Option<ScoredMove> {
    best_move_with_potential(board, market, cards, 20) // default: always use potential
}

/// Find the best move. When `turns_remaining <= 1`, potential is ignored
/// (last move should maximize actual score, not future setup).
pub fn best_move_with_potential(
    board: &mut Board,
    market: &[(usize, cascadia_core::types::TileData, Wildlife)],
    cards: &ScoringCards,
    turns_remaining: u8,
) -> Option<ScoredMove> {
    // Potential weight scales with turns remaining: full weight early, zero on last turn.
    // In 4p, turns_remaining counts ALL players, so AI turns = turns_remaining / 4.
    let ai_turns_left = (turns_remaining as f32 / 4.0).max(0.0);
    // Hand-crafted potential disabled by default — NNUE afterstate scoring
    // handles future-value better when an NNUE is present. For pure-greedy
    // benches (no NNUE, e.g. testing under alternate scoring cards), set
    // `CASCADIA_GREEDY_POTENTIAL=1` to enable the per-card dispatch heuristic.
    let _potential_scale = (ai_turns_left / 10.0).min(1.0);
    let use_potential = std::env::var("CASCADIA_GREEDY_POTENTIAL").ok()
        .map(|s| !s.is_empty() && s != "0").unwrap_or(false);
    let frontier = board.frontier();
    if frontier.is_empty() || market.is_empty() {
        return None;
    }

    let base_wildlife_total: u16 = wildlife::score_all_wildlife(board, cards).iter().sum();
    let base_nature = board.nature_tokens as u16;
    let has_tokens = board.nature_tokens > 0;

    let mut best: Option<ScoredMove> = None;

    // Build all (tile, wildlife, cost) combinations to evaluate.
    // Normal: tile and wildlife from same slot (cost=0)
    // Independent: tile from slot A, wildlife from slot B where A≠B (cost=1 nature token)
    let mut combos: Vec<(usize, cascadia_core::types::TileData, Wildlife, Option<usize>)> =
        Vec::new();

    // Normal paired drafts
    for &(idx, tile, wl) in market {
        combos.push((idx, tile, wl, None));
    }

    // Independent drafts (only if player has nature tokens)
    if has_tokens {
        for &(tile_idx, tile, _) in market {
            for &(wl_idx, _, wl) in market {
                if tile_idx != wl_idx {
                    combos.push((tile_idx, tile, wl, Some(wl_idx)));
                }
            }
        }
    }

    for (tile_market_idx, tile, drafted_wildlife, wl_market_idx) in combos {
        let is_independent = wl_market_idx.is_some();
        let effective_nature = if is_independent {
            base_nature.saturating_sub(1)
        } else {
            base_nature
        };

        let max_rotations: u8 = if tile.terrain2.is_none() { 1 } else { 6 };

        // --- Joint tile+wildlife evaluation ---
        // Use top-8 tile placements by habitat to keep speed reasonable.
        // This covers the habitat-optimal placements AND several alternatives
        // where wildlife placement may be much better.
        struct TilePos { q: i8, r: i8, rot: u8, hab: u16 }
        let mut tile_positions: Vec<TilePos> = Vec::new();
        for &frontier_idx in frontier.iter() {
            let c = HexCoord::from_index(frontier_idx as usize);
            for rot in 0..max_rotations {
                if let Some(a) = board.place_tile(c, tile, rot) {
                    let hab: u16 = board.largest_group.iter().sum();
                    tile_positions.push(TilePos { q: c.q, r: c.r, rot, hab });
                    board.undo(a);
                }
            }
        }
        tile_positions.sort_by(|a, b| b.hab.cmp(&a.hab));
        tile_positions.truncate(8);

        for tp in &tile_positions {
            let coord = HexCoord::new(tp.q, tp.r);
            let rotation = tp.rot;
            let habitat_score = tp.hab;

            let tile_action = board.place_tile(coord, tile, rotation).unwrap();

        // --- Find best wildlife placement for THIS tile position ---
        let skip_score = habitat_score + base_wildlife_total + effective_nature;
        let skip_potential = if use_potential { (board_potential(board, cards) as f32 * _potential_scale) as i32 } else { 0 };
        let skip_eval = (skip_score as i32) * EVAL_SCALE + skip_potential;

        let mut best_total = skip_score;
        let mut best_eval = skip_eval;
        let mut best_wl_q: Option<i8> = None;
        let mut best_wl_r: Option<i8> = None;

        let valid_positions: arrayvec::ArrayVec<u16, 64> = board
            .placed_tiles
            .iter()
            .copied()
            .filter(|&idx| board.grid.get(idx as usize).can_place_wildlife(drafted_wildlife))
            .collect();

        let variant = cards.variant_for(drafted_wildlife);
        let without_score = wildlife::score_wildlife(board, drafted_wildlife, variant);

        for &wl_idx in &valid_positions {
            let wl_action = match board.place_wildlife(wl_idx as usize, drafted_wildlife) {
                Some(a) => a,
                None => continue,
            };

            let with_score = wildlife::score_wildlife(board, drafted_wildlife, variant);
            let potential = if use_potential { (board_potential(board, cards) as f32 * _potential_scale) as i32 } else { 0 };

            board.undo(wl_action);

            let wildlife_delta = with_score.saturating_sub(without_score);
            let nature_bonus: u16 = if board.grid.get(wl_idx as usize).is_keystone() {
                1
            } else {
                0
            };
            let total =
                habitat_score + base_wildlife_total + wildlife_delta + effective_nature + nature_bonus;
            let eval = (total as i32) * EVAL_SCALE + potential;

            if eval > best_eval {
                best_total = total;
                best_eval = eval;
                let wl_coord = HexCoord::from_index(wl_idx as usize);
                best_wl_q = Some(wl_coord.q);
                best_wl_r = Some(wl_coord.r);
            }
        }

        // Check if this tile+wildlife combo is the best so far
        if best.is_none() || best_eval > best.unwrap().eval {
            best = Some(ScoredMove {
                market_index: tile_market_idx,
                tile_q: coord.q,
                tile_r: coord.r,
                rotation,
                wildlife_q: best_wl_q,
                wildlife_r: best_wl_r,
                score: best_total,
                eval: best_eval,
                wildlife_market_index: wl_market_idx,
            });
        }

                board.undo(tile_action);
        } // tile_positions
    } // combos

    best
}

/// Find the best move using 1-turn lookahead.
///
/// For each candidate move, clone the game state, execute the move (which
/// refills the market from the actual bag), then evaluate what the best
/// score would be after the NEXT turn's best greedy move.
///
/// This naturally accounts for how this turn's placement sets up next turn.
pub fn best_move_lookahead(game: &cascadia_core::game::GameState) -> Option<ScoredMove> {
    let board = &game.boards[game.current_player];
    let cards = &game.scoring_cards;
    let frontier = board.frontier();
    if frontier.is_empty() {
        return None;
    }

    // Collect all candidate moves (paired + independent)
    let market_pairs: Vec<_> = game
        .market
        .available()
        .map(|(i, pair)| (i, pair.tile, pair.wildlife))
        .collect();

    if market_pairs.is_empty() {
        return None;
    }

    let has_tokens = board.nature_tokens > 0;

    struct Candidate {
        tile_idx: usize,
        tile: cascadia_core::types::TileData,
        wildlife: Wildlife,
        wl_market_idx: Option<usize>,
    }

    let mut candidates: Vec<Candidate> = Vec::new();

    // Normal paired drafts
    for &(idx, tile, wl) in &market_pairs {
        candidates.push(Candidate { tile_idx: idx, tile, wildlife: wl, wl_market_idx: None });
    }
    // Independent drafts
    if has_tokens {
        for &(ti, tile, _) in &market_pairs {
            for &(wi, _, wl) in &market_pairs {
                if ti != wi {
                    candidates.push(Candidate { tile_idx: ti, tile, wildlife: wl, wl_market_idx: Some(wi) });
                }
            }
        }
    }

    let mut best: Option<ScoredMove> = None;

    for cand in &candidates {
        let is_independent = cand.wl_market_idx.is_some();
        let max_rotations: u8 = if cand.tile.terrain2.is_none() { 1 } else { 6 };

        // Phase 1: find best tile placement by habitat (same as greedy)
        let board = &game.boards[game.current_player];
        let base_wildlife_total: u16 = wildlife::score_all_wildlife(board, cards).iter().sum();

        // We need a mutable board clone just for tile placement evaluation
        let mut board_clone = board.clone();

        let mut best_habitat_score: u16 = 0;
        let mut best_tile_q: i8 = 0;
        let mut best_tile_r: i8 = 0;
        let mut best_rotation: u8 = 0;
        let mut found_tile = false;

        for &frontier_idx in frontier.iter() {
            let coord = HexCoord::from_index(frontier_idx as usize);
            for rotation in 0..max_rotations {
                let action = match board_clone.place_tile(coord, cand.tile, rotation) {
                    Some(a) => a,
                    None => continue,
                };
                let habitat_score: u16 = board_clone.largest_group.iter().sum();
                if !found_tile || habitat_score > best_habitat_score {
                    best_habitat_score = habitat_score;
                    best_tile_q = coord.q;
                    best_tile_r = coord.r;
                    best_rotation = rotation;
                    found_tile = true;
                }
                board_clone.undo(action);
            }
        }

        if !found_tile {
            continue;
        }

        // Phase 2: find best wildlife placement
        let tile_action = board_clone
            .place_tile(HexCoord::new(best_tile_q, best_tile_r), cand.tile, best_rotation)
            .unwrap();

        let effective_nature = if is_independent {
            (board.nature_tokens as u16).saturating_sub(1)
        } else {
            board.nature_tokens as u16
        };

        let variant = cards.variant_for(cand.wildlife);
        let without_score = wildlife::score_wildlife(&board_clone, cand.wildlife, variant);

        let skip_score = best_habitat_score + base_wildlife_total + effective_nature;
        let mut best_imm_total = skip_score;
        let mut best_wl_q: Option<i8> = None;
        let mut best_wl_r: Option<i8> = None;

        let valid_positions: arrayvec::ArrayVec<u16, 64> = board_clone
            .placed_tiles
            .iter()
            .copied()
            .filter(|&idx| board_clone.grid.get(idx as usize).can_place_wildlife(cand.wildlife))
            .collect();

        for &wl_idx in &valid_positions {
            let wl_action = match board_clone.place_wildlife(wl_idx as usize, cand.wildlife) {
                Some(a) => a,
                None => continue,
            };
            let with_score = wildlife::score_wildlife(&board_clone, cand.wildlife, variant);
            board_clone.undo(wl_action);

            let wildlife_delta = with_score.saturating_sub(without_score);
            let nature_bonus: u16 = if board_clone.grid.get(wl_idx as usize).is_keystone() { 1 } else { 0 };
            let total = best_habitat_score + base_wildlife_total + wildlife_delta + effective_nature + nature_bonus;

            if total > best_imm_total {
                best_imm_total = total;
                let wl_coord = HexCoord::from_index(wl_idx as usize);
                best_wl_q = Some(wl_coord.q);
                best_wl_r = Some(wl_coord.r);
            }
        }

        board_clone.undo(tile_action);

        // Phase 3: LOOKAHEAD — clone full game, execute this move, then
        // evaluate the best score achievable on the next turn.
        let mut game_clone = game.clone();
        let tile_coord = HexCoord::new(best_tile_q, best_tile_r);
        let wildlife_placement = match (best_wl_q, best_wl_r) {
            (Some(wq), Some(wr)) => HexCoord::new(wq, wr).to_index(),
            _ => None,
        };

        let executed = if let Some(wmi) = cand.wl_market_idx {
            game_clone.execute_independent_move(
                cand.tile_idx, wmi, tile_coord, best_rotation, wildlife_placement,
            )
        } else {
            game_clone.execute_move(cascadia_core::game::PlayerMove {
                market_index: cand.tile_idx,
                tile_coord,
                rotation: best_rotation,
                wildlife_placement,
            })
        };

        if !executed {
            continue;
        }

        // Evaluate: if game is over, use final score. Otherwise, compute
        // what score the best greedy move next turn would yield for THIS player.
        let this_player = game.current_player;
        let lookahead_score = if game_clone.is_game_over() {
            cascadia_core::scoring::ScoreBreakdown::compute(
                &mut game_clone.boards[this_player],
                &game_clone.scoring_cards,
            ).total
        } else {
            // Evaluate our own board's score after this move + potential
            let next_market: Vec<_> = game_clone
                .market
                .available()
                .map(|(i, pair)| (i, pair.tile, pair.wildlife))
                .collect();
            let next_turns = game_clone.turns_remaining;
            let next_board = &mut game_clone.boards[this_player];
            match best_move_with_potential(next_board, &next_market, &game_clone.scoring_cards, next_turns) {
                Some(next_mv) => next_mv.score,
                None => best_imm_total,
            }
        };

        let is_better = match &best {
            None => true,
            Some(prev) => {
                if lookahead_score > prev.score {
                    true
                } else if lookahead_score == prev.score {
                    !is_independent && prev.wildlife_market_index.is_some()
                } else {
                    false
                }
            }
        };

        if is_better {
            best = Some(ScoredMove {
                market_index: cand.tile_idx,
                tile_q: best_tile_q,
                tile_r: best_tile_r,
                rotation: best_rotation,
                wildlife_q: best_wl_q,
                wildlife_r: best_wl_r,
                score: lookahead_score,
                eval: lookahead_score as i32 * EVAL_SCALE,
                wildlife_market_index: cand.wl_market_idx,
            });
        }
    }

    best
}
