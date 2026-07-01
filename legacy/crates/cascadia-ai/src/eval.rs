use cascadia_core::board::Board;
use cascadia_core::board::HabitatPreviewContext;
use cascadia_core::hex::HexCoord;
use cascadia_core::scoring::wildlife;
use cascadia_core::types::{ScoringCards, Wildlife};

use crate::potential::board_potential;

/// A fully specified move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Clone, Copy)]
struct TilePlacement {
    index: u16,
    preview_index: u8,
    rotation: u8,
    habitat_score: u16,
}

#[derive(Clone, Copy)]
struct DraftCombination {
    tile_market_index: usize,
    tile: cascadia_core::types::TileData,
    wildlife: Wildlife,
    wildlife_market_index: Option<usize>,
}

#[derive(Clone, Copy)]
struct ExistingWildlifePlacement {
    value: u16,
    q: i8,
    r: i8,
}

/// Wildlife ROI weights: how many points per placement each type yields on average
/// when building optimal patterns. Used to bias move selection toward high-ROI types.
/// Applied to wildlife delta in evaluation (not to actual score).
pub fn wildlife_roi_weight(w: Wildlife) -> f32 {
    match w {
        Wildlife::Salmon => 1.3, // Best scaling: run of 7 = 3.71 pts/each
        Wildlife::Hawk => 1.15,  // Good scaling: 8 isolated = 3.5 pts/each
        Wildlife::Elk => 1.0,    // Decent: line of 4 = 3.25 pts/each
        Wildlife::Fox => 1.0,    // Decent: depends on diversity
        Wildlife::Bear => 0.7,   // Worst ROI unless pair completes: lone bear = 0 pts
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
    // Hand-crafted potential disabled by default — NNUE afterstate scoring
    // handles future-value better when an NNUE is present. For pure-greedy
    // benches (no NNUE, e.g. testing under alternate scoring cards), set
    // `CASCADIA_GREEDY_POTENTIAL=1` to enable the per-card dispatch heuristic.
    let use_potential = greedy_potential_enabled();
    let frontier = board.frontier();
    if frontier.is_empty() || market.is_empty() {
        return None;
    }

    if !use_potential {
        if greedy_reference_enabled() {
            return best_move_full_scan(board, market, cards, &frontier, 0.0, false);
        }
        let mut reference_board = greedy_assert_parity_enabled().then(|| board.clone());
        let movement = best_move_without_potential(board, market, cards, &frontier);
        if let Some(reference_board) = reference_board.as_mut() {
            let reference =
                best_move_full_scan(reference_board, market, cards, &frontier, 0.0, false);
            assert_eq!(
                movement,
                reference,
                "optimized greedy mismatch: nature_tokens={}, placed_tiles={}, market={market:?}",
                board.nature_tokens,
                board.placed_tiles.len(),
            );
        }
        return movement;
    }

    // Potential weight scales with turns remaining: full weight early, zero on last turn.
    // In 4p, turns_remaining counts ALL players, so AI turns = turns_remaining / 4.
    let ai_turns_left = (turns_remaining as f32 / 4.0).max(0.0);
    let potential_scale = (ai_turns_left / 10.0).min(1.0);
    best_move_full_scan(board, market, cards, &frontier, potential_scale, true)
}

fn greedy_reference_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("LEGACY_TEACHER_GREEDY_REFERENCE")
            .ok()
            .is_some_and(|value| !value.is_empty() && value != "0")
    })
}

fn greedy_potential_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CASCADIA_GREEDY_POTENTIAL")
            .ok()
            .is_some_and(|value| !value.is_empty() && value != "0")
    })
}

fn greedy_assert_parity_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("LEGACY_TEACHER_GREEDY_ASSERT_PARITY")
            .ok()
            .is_some_and(|value| !value.is_empty() && value != "0")
    })
}

fn draft_combinations(
    market: &[(usize, cascadia_core::types::TileData, Wildlife)],
    has_tokens: bool,
) -> arrayvec::ArrayVec<DraftCombination, 16> {
    let mut combinations = arrayvec::ArrayVec::new();

    for &(market_index, tile, wildlife) in market {
        combinations.push(DraftCombination {
            tile_market_index: market_index,
            tile,
            wildlife,
            wildlife_market_index: None,
        });
    }

    if has_tokens {
        for &(tile_market_index, tile, _) in market {
            for &(wildlife_market_index, _, wildlife) in market {
                if tile_market_index != wildlife_market_index {
                    combinations.push(DraftCombination {
                        tile_market_index,
                        tile,
                        wildlife,
                        wildlife_market_index: Some(wildlife_market_index),
                    });
                }
            }
        }
    }

    combinations
}

/// Retain the stable top-eight habitat placements without allocating or
/// sorting the full placement set.
#[inline(never)]
fn top_habitat_placements(
    board: &Board,
    context: &HabitatPreviewContext,
    tile: cascadia_core::types::TileData,
) -> arrayvec::ArrayVec<TilePlacement, 8> {
    let max_rotations = if tile.terrain2.is_none() { 1 } else { 6 };
    let mut selected = arrayvec::ArrayVec::<(TilePlacement, u16), 8>::new();
    let mut traversal_order = 0u16;
    let mut worst_index = 0usize;

    for (preview_index, frontier_cell) in context.frontier.iter().enumerate() {
        for rotation in 0..max_rotations {
            let habitat_score =
                board.preview_habitat_total_prepared(context, frontier_cell, tile, rotation);
            let placement = TilePlacement {
                index: frontier_cell.index,
                preview_index: preview_index as u8,
                rotation,
                habitat_score,
            };
            if selected.len() < selected.capacity() {
                selected.push((placement, traversal_order));
                if selected.len() == selected.capacity() {
                    worst_index = worst_ranked_placement(&selected);
                }
            } else if habitat_score > selected[worst_index].0.habitat_score {
                selected[worst_index] = (placement, traversal_order);
                worst_index = worst_ranked_placement(&selected);
            }
            traversal_order += 1;
        }
    }

    selected.sort_unstable_by(|left, right| {
        right
            .0
            .habitat_score
            .cmp(&left.0.habitat_score)
            .then_with(|| left.1.cmp(&right.1))
    });
    let mut placements = arrayvec::ArrayVec::<TilePlacement, 8>::new();
    placements.extend(selected.into_iter().map(|(placement, _)| placement));
    placements
}

#[inline]
fn worst_ranked_placement(placements: &arrayvec::ArrayVec<(TilePlacement, u16), 8>) -> usize {
    placements
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            left.0
                .habitat_score
                .cmp(&right.0.habitat_score)
                .then_with(|| right.1.cmp(&left.1))
        })
        .map(|(index, _)| index)
        .expect("top-eight selection only ranks a full set")
}

/// Exact qualified greedy policy used by rollout opponents.
///
/// Existing wildlife placements depend only on the wildlife type, so each type
/// is scored once per board. Habitat placements are recomputed at their
/// original combination boundary because legacy apply/undo history is part of
/// the qualified policy's exact tie-breaking behavior. For a particular tile
/// placement only the newly placed tile can introduce another wildlife option,
/// and that wildlife value is rotation-invariant for a fixed coordinate.
#[inline(never)]
fn best_move_without_potential(
    board: &mut Board,
    market: &[(usize, cascadia_core::types::TileData, Wildlife)],
    cards: &ScoringCards,
    frontier: &[u16],
) -> Option<ScoredMove> {
    let base_wildlife_scores = wildlife::score_all_wildlife(board, cards);
    let base_wildlife_total = base_wildlife_scores.iter().sum::<u16>();
    let base_nature = board.nature_tokens as u16;
    let combinations = draft_combinations(market, board.nature_tokens > 0);
    let habitat_preview = board.habitat_preview_context(frontier);

    let mut best_existing_wildlife: [Option<ExistingWildlifePlacement>; 5] = [None; 5];
    let mut needed_wildlife = [false; 5];
    for combination in &combinations {
        needed_wildlife[combination.wildlife as usize] = true;
    }
    let placed_tiles = board.placed_tiles.clone();
    for wildlife_type in Wildlife::ALL
        .into_iter()
        .filter(|&wildlife_type| needed_wildlife[wildlife_type as usize])
    {
        let without_score = base_wildlife_scores[wildlife_type as usize];
        let variant = cards.variant_for(wildlife_type);
        for &tile_index in &placed_tiles {
            let tile_index = tile_index as usize;
            if !board.grid.get(tile_index).can_place_wildlife(wildlife_type) {
                continue;
            }
            let with_score =
                wildlife::score_wildlife_after_placement(board, wildlife_type, variant, tile_index);

            let value = with_score.saturating_sub(without_score)
                + u16::from(board.grid.get(tile_index).is_keystone());
            if value == 0
                || best_existing_wildlife[wildlife_type as usize]
                    .is_some_and(|current| current.value >= value)
            {
                continue;
            }
            let coord = HexCoord::from_index(tile_index);
            best_existing_wildlife[wildlife_type as usize] = Some(ExistingWildlifePlacement {
                value,
                q: coord.q,
                r: coord.r,
            });
        }
    }

    let mut best = None;
    for combination in combinations {
        let effective_nature = if combination.wildlife_market_index.is_some() {
            base_nature.saturating_sub(1)
        } else {
            base_nature
        };
        let placements = top_habitat_placements(board, &habitat_preview, combination.tile);
        let mut new_tile_wildlife_values = arrayvec::ArrayVec::<(u16, u16), 8>::new();

        for placement in &placements {
            let tile_index = placement.index as usize;
            let coord = HexCoord::from_index(tile_index);
            assert!(
                board.replay_tile_place_undo_habitat_prepared(
                    &habitat_preview.frontier[placement.preview_index as usize],
                    combination.tile,
                    placement.rotation,
                ),
                "previewed frontier placement must remain legal"
            );
            let mut wildlife_value = 0;
            let mut wildlife_q = None;
            let mut wildlife_r = None;
            if let Some(existing) = best_existing_wildlife[combination.wildlife as usize] {
                wildlife_value = existing.value;
                wildlife_q = Some(existing.q);
                wildlife_r = Some(existing.r);
            }

            if combination.tile.allowed.contains(combination.wildlife) {
                let value = if let Some((_, value)) = new_tile_wildlife_values
                    .iter()
                    .find(|&&(index, _)| index == placement.index)
                {
                    *value
                } else {
                    let with_score = wildlife::score_wildlife_after_placement(
                        board,
                        combination.wildlife,
                        cards.variant_for(combination.wildlife),
                        tile_index,
                    );
                    let value = with_score
                        .saturating_sub(base_wildlife_scores[combination.wildlife as usize])
                        + u16::from(combination.tile.keystone);
                    new_tile_wildlife_values.push((placement.index, value));
                    value
                };
                // Existing tiles precede the new tile in `placed_tiles`, so a
                // strict comparison preserves the original tie break.
                if value > wildlife_value {
                    wildlife_value = value;
                    wildlife_q = Some(coord.q);
                    wildlife_r = Some(coord.r);
                }
            }

            let total =
                placement.habitat_score + base_wildlife_total + wildlife_value + effective_nature;
            let eval = (total as i32) * EVAL_SCALE;
            if best.is_none_or(|current: ScoredMove| eval > current.eval) {
                best = Some(ScoredMove {
                    market_index: combination.tile_market_index,
                    tile_q: coord.q,
                    tile_r: coord.r,
                    rotation: placement.rotation,
                    wildlife_q,
                    wildlife_r,
                    score: total,
                    eval,
                    wildlife_market_index: combination.wildlife_market_index,
                });
            }
        }
    }

    best
}

/// Complete reference evaluator retained for potential-enabled play and exact
/// parity tests of the optimized qualified path.
fn best_move_full_scan(
    board: &mut Board,
    market: &[(usize, cascadia_core::types::TileData, Wildlife)],
    cards: &ScoringCards,
    frontier: &[u16],
    potential_scale: f32,
    use_potential: bool,
) -> Option<ScoredMove> {
    let base_wildlife_total: u16 = wildlife::score_all_wildlife(board, cards).iter().sum();
    let base_nature = board.nature_tokens as u16;
    let has_tokens = board.nature_tokens > 0;

    let mut best: Option<ScoredMove> = None;
    let combinations = draft_combinations(market, has_tokens);

    for combination in combinations {
        let is_independent = combination.wildlife_market_index.is_some();
        let effective_nature = if is_independent {
            base_nature.saturating_sub(1)
        } else {
            base_nature
        };

        let max_rotations: u8 = if combination.tile.terrain2.is_none() {
            1
        } else {
            6
        };

        // --- Joint tile+wildlife evaluation ---
        // Use top-8 tile placements by habitat to keep speed reasonable.
        // This covers the habitat-optimal placements AND several alternatives
        // where wildlife placement may be much better.
        let mut tile_positions = Vec::new();
        for &frontier_idx in frontier.iter() {
            let c = HexCoord::from_index(frontier_idx as usize);
            for rot in 0..max_rotations {
                if let Some(habitat_score) = board.preview_habitat_total(c, combination.tile, rot) {
                    tile_positions.push(TilePlacement {
                        index: frontier_idx,
                        preview_index: 0,
                        rotation: rot,
                        habitat_score,
                    });
                }
            }
        }
        tile_positions.sort_by(|a, b| b.habitat_score.cmp(&a.habitat_score));
        tile_positions.truncate(8);

        for tp in &tile_positions {
            let coord = HexCoord::from_index(tp.index as usize);
            let rotation = tp.rotation;
            let habitat_score = tp.habitat_score;

            let tile_action = board.place_tile(coord, combination.tile, rotation).unwrap();

            // --- Find best wildlife placement for THIS tile position ---
            let skip_score = habitat_score + base_wildlife_total + effective_nature;
            let skip_potential = if use_potential {
                (board_potential(board, cards) as f32 * potential_scale) as i32
            } else {
                0
            };
            let skip_eval = (skip_score as i32) * EVAL_SCALE + skip_potential;

            let mut best_total = skip_score;
            let mut best_eval = skip_eval;
            let mut best_wl_q: Option<i8> = None;
            let mut best_wl_r: Option<i8> = None;

            let valid_positions: arrayvec::ArrayVec<u16, 64> = board
                .placed_tiles
                .iter()
                .copied()
                .filter(|&idx| {
                    board
                        .grid
                        .get(idx as usize)
                        .can_place_wildlife(combination.wildlife)
                })
                .collect();

            let variant = cards.variant_for(combination.wildlife);
            let without_score = wildlife::score_wildlife(board, combination.wildlife, variant);

            for &wl_idx in &valid_positions {
                let wl_action = match board.place_wildlife(wl_idx as usize, combination.wildlife) {
                    Some(a) => a,
                    None => continue,
                };

                let with_score = wildlife::score_wildlife(board, combination.wildlife, variant);
                let potential = if use_potential {
                    (board_potential(board, cards) as f32 * potential_scale) as i32
                } else {
                    0
                };

                board.undo(wl_action);

                let wildlife_delta = with_score.saturating_sub(without_score);
                let nature_bonus: u16 = if board.grid.get(wl_idx as usize).is_keystone() {
                    1
                } else {
                    0
                };
                let total = habitat_score
                    + base_wildlife_total
                    + wildlife_delta
                    + effective_nature
                    + nature_bonus;
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
                    market_index: combination.tile_market_index,
                    tile_q: coord.q,
                    tile_r: coord.r,
                    rotation,
                    wildlife_q: best_wl_q,
                    wildlife_r: best_wl_r,
                    score: best_total,
                    eval: best_eval,
                    wildlife_market_index: combination.wildlife_market_index,
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
        candidates.push(Candidate {
            tile_idx: idx,
            tile,
            wildlife: wl,
            wl_market_idx: None,
        });
    }
    // Independent drafts
    if has_tokens {
        for &(ti, tile, _) in &market_pairs {
            for &(wi, _, wl) in &market_pairs {
                if ti != wi {
                    candidates.push(Candidate {
                        tile_idx: ti,
                        tile,
                        wildlife: wl,
                        wl_market_idx: Some(wi),
                    });
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
            .place_tile(
                HexCoord::new(best_tile_q, best_tile_r),
                cand.tile,
                best_rotation,
            )
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
            .filter(|&idx| {
                board_clone
                    .grid
                    .get(idx as usize)
                    .can_place_wildlife(cand.wildlife)
            })
            .collect();

        for &wl_idx in &valid_positions {
            let wl_action = match board_clone.place_wildlife(wl_idx as usize, cand.wildlife) {
                Some(a) => a,
                None => continue,
            };
            let with_score = wildlife::score_wildlife(&board_clone, cand.wildlife, variant);
            board_clone.undo(wl_action);

            let wildlife_delta = with_score.saturating_sub(without_score);
            let nature_bonus: u16 = if board_clone.grid.get(wl_idx as usize).is_keystone() {
                1
            } else {
                0
            };
            let total = best_habitat_score
                + base_wildlife_total
                + wildlife_delta
                + effective_nature
                + nature_bonus;

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
                cand.tile_idx,
                wmi,
                tile_coord,
                best_rotation,
                wildlife_placement,
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
            )
            .total
        } else {
            // Evaluate our own board's score after this move + potential
            let next_market: Vec<_> = game_clone
                .market
                .available()
                .map(|(i, pair)| (i, pair.tile, pair.wildlife))
                .collect();
            let next_turns = game_clone.turns_remaining;
            let next_board = &mut game_clone.boards[this_player];
            match best_move_with_potential(
                next_board,
                &next_market,
                &game_clone.scoring_cards,
                next_turns,
            ) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_core::game::GameState;
    use cascadia_core::types::ScoringCardVariant::{A, B, C, D};
    use rand::{rngs::StdRng, SeedableRng};

    fn market(game: &GameState) -> Vec<(usize, cascadia_core::types::TileData, Wildlife)> {
        game.market
            .available()
            .map(|(index, pair)| (index, pair.tile, pair.wildlife))
            .collect()
    }

    fn reference_top_habitat_placements(
        board: &Board,
        frontier: &[u16],
        tile: cascadia_core::types::TileData,
    ) -> Vec<(i8, i8, u8, u16)> {
        let rotations = if tile.terrain2.is_none() { 1 } else { 6 };
        let mut placements = Vec::new();
        for &frontier_index in frontier {
            let coord = HexCoord::from_index(frontier_index as usize);
            for rotation in 0..rotations {
                if let Some(score) = board.preview_habitat_total(coord, tile, rotation) {
                    placements.push((coord.q, coord.r, rotation, score));
                }
            }
        }
        placements.sort_by(|left, right| right.3.cmp(&left.3));
        placements.truncate(8);
        placements
    }

    fn assert_qualified_move_parity(game: &GameState, force_nature_token: bool) {
        let market = market(game);
        let mut fast_board = game.boards[game.current_player].clone();
        if force_nature_token {
            fast_board.nature_tokens = fast_board.nature_tokens.max(1);
        }
        let mut reference_board = fast_board.clone();
        let frontier = fast_board.frontier();

        let fast =
            best_move_without_potential(&mut fast_board, &market, &game.scoring_cards, &frontier);
        let reference = best_move_full_scan(
            &mut reference_board,
            &market,
            &game.scoring_cards,
            &frontier,
            0.0,
            false,
        );
        assert_eq!(
            fast, reference,
            "qualified greedy mismatch at player {}, {} turns remaining, forced_token={force_nature_token}",
            game.current_player, game.turns_remaining
        );
    }

    #[test]
    fn fixed_top_eight_matches_stable_full_sort() {
        let mut rng = StdRng::seed_from_u64(0x1eaf_cafe);
        let mut game = GameState::new(4, ScoringCards::all_a(), &mut rng);

        for _ in 0..16 {
            let board = &game.boards[game.current_player];
            let frontier = board.frontier();
            let context = board.habitat_preview_context(&frontier);
            for (_, tile, _) in market(&game) {
                let bounded = top_habitat_placements(board, &context, tile)
                    .iter()
                    .map(|placement| {
                        let coord = HexCoord::from_index(placement.index as usize);
                        (
                            coord.q,
                            coord.r,
                            placement.rotation,
                            placement.habitat_score,
                        )
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    bounded,
                    reference_top_habitat_placements(board, &frontier, tile)
                );
            }

            let market = market(&game);
            let mut board = game.boards[game.current_player].clone();
            let frontier = board.frontier();
            let Some(movement) = best_move_full_scan(
                &mut board,
                &market,
                &game.scoring_cards,
                &frontier,
                0.0,
                false,
            ) else {
                break;
            };
            assert!(crate::search::execute_scored_move(&mut game, &movement));
        }
    }

    #[test]
    fn optimized_qualified_greedy_matches_full_scanner() {
        let card_sets = [
            ScoringCards::all_a(),
            ScoringCards {
                cards: [A, B, C, D, A],
            },
            ScoringCards {
                cards: [D, C, B, A, D],
            },
        ];

        for (card_index, cards) in card_sets.into_iter().enumerate() {
            let mut rng = StdRng::seed_from_u64(0x600d_f00d + card_index as u64);
            let mut game = GameState::new(4, cards, &mut rng);
            for _ in 0..16 {
                assert_qualified_move_parity(&game, false);
                assert_qualified_move_parity(&game, true);

                let market = market(&game);
                let mut board = game.boards[game.current_player].clone();
                let frontier = board.frontier();
                let Some(movement) = best_move_full_scan(
                    &mut board,
                    &market,
                    &game.scoring_cards,
                    &frontier,
                    0.0,
                    false,
                ) else {
                    break;
                };
                assert!(crate::search::execute_scored_move(&mut game, &movement));
            }
        }
    }

    #[test]
    fn optimized_qualified_greedy_matches_complete_aaaaa_games() {
        for game_index in 0..16 {
            let mut rng = StdRng::seed_from_u64(0xa11a_0000 + game_index);
            let mut game = GameState::new(4, ScoringCards::all_a(), &mut rng);
            while !game.is_game_over() {
                assert_qualified_move_parity(&game, false);
                assert_qualified_move_parity(&game, true);

                let market = market(&game);
                let mut board = game.boards[game.current_player].clone();
                let frontier = board.frontier();
                let Some(movement) = best_move_full_scan(
                    &mut board,
                    &market,
                    &game.scoring_cards,
                    &frontier,
                    0.0,
                    false,
                ) else {
                    break;
                };
                assert!(crate::search::execute_scored_move(&mut game, &movement));
            }
        }
    }
}
