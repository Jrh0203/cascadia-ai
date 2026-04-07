use std::sync::Mutex;

use axum::{
    extract::State,
    http::StatusCode,
    response::Html,
    routing::{get, post},
    Json, Router,
};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use cascadia_core::game::{GameState, PlayerMove};
use cascadia_core::hex::HexCoord;
use cascadia_core::scoring::ScoreBreakdown;
use cascadia_core::types::*;

struct AppState {
    game: Mutex<GameState>,
    rng: Mutex<StdRng>,
    nnue: Option<std::sync::Arc<cascadia_ai::nnue::NNUENetwork>>,
    solo_sim: Mutex<bool>,
    history: Mutex<Vec<GameState>>,
    events: Mutex<Vec<String>>,
}

// --- JSON response types ---

#[derive(Serialize)]
struct GameView {
    cells: Vec<CellView>,
    market: Vec<MarketPairView>,
    current_player: usize,
    view_player: usize,
    num_players: usize,
    turns_remaining: u8,
    scores: Vec<ScoreView>,
    game_over: bool,
    nature_tokens: Vec<u8>,
    can_mulligan: bool,
    has_overflow: bool,
    events: Vec<String>,
}

#[derive(Serialize)]
struct CellView {
    q: i8,
    r: i8,
    present: bool,
    terrain1: Option<&'static str>,
    terrain2: Option<&'static str>,
    wildlife: Option<&'static str>,
    wildlife_emoji: Option<&'static str>,
    allowed: Vec<&'static str>,
    keystone: bool,
    is_frontier: bool,
    rotation: u8,
}

#[derive(Serialize)]
struct MarketPairView {
    index: usize,
    terrain1: &'static str,
    terrain2: Option<&'static str>,
    allowed: Vec<&'static str>,
    wildlife: &'static str,
    wildlife_emoji: &'static str,
    keystone: bool,
}

#[derive(Serialize)]
struct ScoreView {
    player: usize,
    habitat: [u16; 5],
    wildlife: [u16; 5],
    nature_tokens: u16,
    habitat_bonus: [u8; 5],
    total: u16,
}

#[derive(Deserialize)]
struct MoveRequest {
    market_index: usize,
    q: i8,
    r: i8,
    #[serde(default)]
    rotation: u8,
    wildlife_q: Option<i8>,
    wildlife_r: Option<i8>,
    /// If set, spend a pinecone to pick wildlife from a different market slot.
    wildlife_market_index: Option<usize>,
}

#[derive(Deserialize)]
struct NewGameRequest {
    num_players: Option<usize>,
    solo_sim: Option<bool>,
}

fn terrain_name(t: Terrain) -> &'static str {
    match t {
        Terrain::Forest => "forest",
        Terrain::Prairie => "prairie",
        Terrain::Wetland => "wetland",
        Terrain::Mountain => "mountain",
        Terrain::River => "river",
    }
}

fn wildlife_name(w: Wildlife) -> &'static str {
    match w {
        Wildlife::Bear => "bear",
        Wildlife::Elk => "elk",
        Wildlife::Salmon => "salmon",
        Wildlife::Hawk => "hawk",
        Wildlife::Fox => "fox",
    }
}

fn build_game_view(game: &mut GameState) -> GameView {
    build_game_view_for_with(game, game.current_player, Vec::new())
}

fn build_game_view_with_events(game: &mut GameState, events: Vec<String>) -> GameView {
    build_game_view_for_with(game, game.current_player, events)
}

fn build_game_view_for(game: &mut GameState, view_player: usize) -> GameView {
    build_game_view_for_with(game, view_player, Vec::new())
}

fn build_game_view_for_with(game: &mut GameState, view_player: usize, events: Vec<String>) -> GameView {
    let view_player = view_player.min(game.num_players.saturating_sub(1));
    let board = &game.boards[view_player];
    let frontier = board.frontier();
    let _frontier_set: std::collections::HashSet<u16> = frontier.iter().copied().collect();

    // Collect cells — only emit placed tiles and frontier cells
    let mut cells = Vec::new();

    for &tile_idx in &board.placed_tiles {
        let idx = tile_idx as usize;
        let coord = HexCoord::from_index(idx);
        let cell = board.grid.get(idx);
        cells.push(CellView {
            q: coord.q,
            r: coord.r,
            present: true,
            terrain1: cell.primary_terrain().map(terrain_name),
            terrain2: cell.secondary_terrain().map(terrain_name),
            wildlife: cell.placed_wildlife().map(wildlife_name),
            wildlife_emoji: cell.placed_wildlife().map(|w| w.emoji()),
            allowed: cell.allowed_wildlife().iter().map(wildlife_name).collect(),
            keystone: cell.is_keystone(),
            is_frontier: false,
            rotation: board.rotations[idx],
        });
    }

    for &fidx in &frontier {
        let coord = HexCoord::from_index(fidx as usize);
        cells.push(CellView {
            q: coord.q,
            r: coord.r,
            present: false,
            terrain1: None,
            terrain2: None,
            wildlife: None,
            wildlife_emoji: None,
            allowed: vec![],
            keystone: false,
            is_frontier: true,
            rotation: 0,
        });
    }

    // Market
    let market: Vec<MarketPairView> = game
        .market
        .available()
        .map(|(i, pair)| MarketPairView {
            index: i,
            terrain1: terrain_name(pair.tile.terrain1),
            terrain2: pair.tile.terrain2.map(terrain_name),
            allowed: pair.tile.allowed.iter().map(wildlife_name).collect(),
            wildlife: wildlife_name(pair.wildlife),
            wildlife_emoji: pair.wildlife.emoji(),
            keystone: pair.tile.keystone,
        })
        .collect();

    // Scores for all players
    // Habitat majority bonuses only apply at end of game in multiplayer
    let is_game_over = game.is_game_over();
    let is_multiplayer = game.num_players > 1;
    let scores: Vec<ScoreView> = (0..game.num_players)
        .map(|p| {
            let breakdown = if is_game_over && is_multiplayer {
                ScoreBreakdown::compute_with_bonuses(&mut game.boards, &game.scoring_cards, p)
            } else {
                ScoreBreakdown::compute(&mut game.boards[p], &game.scoring_cards)
            };
            ScoreView {
                player: p,
                habitat: breakdown.habitat,
                wildlife: breakdown.wildlife,
                nature_tokens: breakdown.nature_tokens,
                habitat_bonus: breakdown.habitat_bonus,
                total: breakdown.total,
            }
        })
        .collect();

    let nature_tokens: Vec<u8> = game.boards.iter().map(|b| b.nature_tokens).collect();
    let current_tokens = game.boards[game.current_player].nature_tokens;
    let has_overflow = game.market.has_overflow();

    GameView {
        cells,
        market,
        current_player: game.current_player,
        view_player,
        num_players: game.num_players,
        turns_remaining: game.turns_remaining,
        scores,
        game_over: game.is_game_over(),
        nature_tokens,
        can_mulligan: current_tokens > 0,
        has_overflow,
        events,
    }
}

#[derive(Deserialize)]
struct StateQuery {
    view: Option<usize>,
}

async fn get_state(
    State(state): State<std::sync::Arc<AppState>>,
    axum::extract::Query(q): axum::extract::Query<StateQuery>,
) -> Json<GameView> {
    let mut game = state.game.lock().unwrap();
    let view = q.view.unwrap_or(game.current_player);
    let events = state.events.lock().unwrap().clone();
    Json(build_game_view_for_with(&mut game, view, events))
}

async fn make_move(
    State(state): State<std::sync::Arc<AppState>>,
    Json(req): Json<MoveRequest>,
) -> Result<Json<GameView>, (StatusCode, String)> {
    let mut game = state.game.lock().unwrap();

    if game.is_game_over() {
        return Err((StatusCode::BAD_REQUEST, "Game is over".to_string()));
    }

    // Capture market info BEFORE the move executes (for event log)
    let drafted_tile_info = game.market.pairs[req.market_index].as_ref().map(|p| {
        let terrain = terrain_name(p.tile.terrain1);
        let terrain2 = p.tile.terrain2.map(terrain_name);
        let wildlife = wildlife_name(p.wildlife);
        (terrain.to_string(), terrain2.map(|s| s.to_string()), wildlife.to_string())
    });
    let independent_wildlife = req.wildlife_market_index.and_then(|wmi| {
        if wmi != req.market_index {
            game.market.pairs[wmi].as_ref().map(|p| wildlife_name(p.wildlife).to_string())
        } else {
            None
        }
    });

    // Snapshot for undo
    {
        let mut hist = state.history.lock().unwrap();
        hist.push(game.clone());
        // Cap history at 50 moves to avoid unbounded growth
        if hist.len() > 50 {
            hist.remove(0);
        }
    }

    let tile_coord = HexCoord::new(req.q, req.r);

    // Determine wildlife placement
    let wildlife_placement = match (req.wildlife_q, req.wildlife_r) {
        (Some(wq), Some(wr)) => HexCoord::new(wq, wr).to_index(),
        _ => None,
    };

    // Check if this is an independent draft (pinecone spend to break pairs)
    let success = if let Some(wmi) = req.wildlife_market_index {
        if wmi != req.market_index {
            game.execute_independent_move(
                req.market_index,
                wmi,
                tile_coord,
                req.rotation % 6,
                wildlife_placement,
            )
        } else {
            // Same index — normal draft
            game.execute_move(PlayerMove {
                market_index: req.market_index,
                tile_coord,
                rotation: req.rotation % 6,
                wildlife_placement,
            })
        }
    } else {
        game.execute_move(PlayerMove {
            market_index: req.market_index,
            tile_coord,
            rotation: req.rotation % 6,
            wildlife_placement,
        })
    };

    if !success {
        // Pop the snapshot we pushed earlier since the move failed
        state.history.lock().unwrap().pop();
        return Err((
            StatusCode::BAD_REQUEST,
            "Invalid move".to_string(),
        ));
    }

    // Log the player's move
    {
        let mut events = state.events.lock().unwrap();
        let tile_desc = drafted_tile_info.as_ref().map(|(t1, t2, wl)| {
            let terrain = match t2 {
                Some(t2) => format!("{}/{}", t1, t2),
                None => t1.clone(),
            };
            (terrain, wl.clone())
        });
        let action = match (independent_wildlife, tile_desc) {
            (Some(wl), Some((terrain, _))) => {
                format!("🌲 pinecone draft: {} tile + {} (from slot {})",
                    terrain, wl, req.wildlife_market_index.unwrap())
            }
            (None, Some((terrain, wl))) => {
                format!("drafted {} tile + {} (slot {})", terrain, wl, req.market_index)
            }
            _ => format!("drafted slot {}", req.market_index),
        };
        let wl_placed = req.wildlife_q.is_some();
        events.push(format!("P1 {} at ({},{}) rot {}{}",
            action, req.q, req.r, req.rotation,
            if wl_placed { " + placed wildlife" } else { " (wildlife discarded)" }));
    }

    // Solo 4p sim: auto-advance opponents with greedy play until back to player 0
    let solo_sim = *state.solo_sim.lock().unwrap();
    if solo_sim {
        while !game.is_game_over() && game.current_player != 0 {
            let p = game.current_player;
            // Greedy opponents ALWAYS take the free 3-of-a-kind replacement when available
            if let Some(overflow_wl) = game.can_replace_overflow() {
                game.replace_overflow();
                state.events.lock().unwrap().push(
                    format!("P{} used free replacement (3× {:?})", p + 1, overflow_wl)
                );
            }
            // Use NNUE-guided move selection for opponents (if weights loaded)
            let opp_mv = if let Some(ref net) = state.nnue {
                cascadia_ai::nnue_train::pick_best_move_nnue(&game, net)
            } else {
                cascadia_ai::mce::best_move_no_rollouts(&game)
            };
            match opp_mv {
                Some(mv) => {
                    let market_idx = mv.market_index;
                    // Capture market info BEFORE executing
                    let tile_info = game.market.pairs[market_idx].as_ref().map(|pair| {
                        let t1 = terrain_name(pair.tile.terrain1).to_string();
                        let t2 = pair.tile.terrain2.map(|t| terrain_name(t).to_string());
                        let wl = wildlife_name(pair.wildlife).to_string();
                        match t2 {
                            Some(t2) => format!("{}/{} + {}", t1, t2, wl),
                            None => format!("{} + {}", t1, wl),
                        }
                    });
                    let indep_wl = mv.wildlife_market_index.and_then(|wmi| {
                        if wmi != market_idx {
                            game.market.pairs[wmi].as_ref().map(|p| wildlife_name(p.wildlife).to_string())
                        } else { None }
                    });
                    if !cascadia_ai::search::execute_scored_move(&mut game, &mv) { break; }
                    let desc = match (tile_info, indep_wl) {
                        (Some(t), Some(wl)) => format!("🌲 pinecone: {} tile + {}", t, wl),
                        (Some(t), None) => format!("drafted {} (slot {})", t, market_idx),
                        _ => format!("drafted slot {}", market_idx),
                    };
                    state.events.lock().unwrap().push(
                        format!("P{} {} at ({},{})", p + 1, desc, mv.tile_q, mv.tile_r)
                    );
                }
                None => break,
            }
        }
    }

    let events = state.events.lock().unwrap().clone();
    Ok(Json(build_game_view_with_events(&mut game, events)))
}

async fn mulligan(
    State(state): State<std::sync::Arc<AppState>>,
) -> Result<Json<GameView>, (StatusCode, String)> {
    let mut game = state.game.lock().unwrap();
    // Snapshot for undo
    {
        let mut hist = state.history.lock().unwrap();
        hist.push(game.clone());
        if hist.len() > 50 { hist.remove(0); }
    }
    if !game.mulligan_wildlife() {
        state.history.lock().unwrap().pop();
        return Err((StatusCode::BAD_REQUEST, "No nature tokens to spend".to_string()));
    }
    state.events.lock().unwrap().push("P1 🌲 spent pinecone for mulligan".to_string());
    let events = state.events.lock().unwrap().clone();
    Ok(Json(build_game_view_with_events(&mut game, events)))
}

/// Returns a single recommended action for the current turn WITHOUT mutating state.
/// The action is one of:
///   { "action": "replace_overflow" } — take the free 3-of-a-kind replacement
///   { "action": "mulligan" } — spend a pinecone to mulligan
///   { "action": "move", "mv": {...} } — execute this move
/// After the user takes the action, they can call suggest again for the next step.
async fn suggest_move(
    State(state): State<std::sync::Arc<AppState>>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let game = state.game.lock().unwrap();
    if game.is_game_over() {
        return Err((StatusCode::BAD_REQUEST, "Game is over".to_string()));
    }

    // Step 1: free 3-of-a-kind replacement always recommended (no downside)
    if game.can_replace_overflow().is_some() {
        return Ok(Json(serde_json::json!({ "action": "replace_overflow" })));
    }

    let net_opt = state.nnue.clone();
    let mut rng = state.rng.lock().unwrap();
    let mut search_rng = StdRng::seed_from_u64(rng.gen());
    drop(rng);

    // Step 2: check if mulligan is worth it (only with NNUE)
    if let Some(ref net) = net_opt {
        let player = game.current_player;
        if game.boards[player].nature_tokens > 0 {
            let baseline = cascadia_ai::mce::best_move_mce(&game, net, 750, &mut search_rng)
                .map(|m| m.score as f32).unwrap_or(0.0);
            let mut total = 0.0f32;
            let mut samples = 0;
            for _ in 0..3 {
                let mut t = game.clone();
                t.shuffle_bags(&mut search_rng);
                if t.mulligan_wildlife() {
                    total += cascadia_ai::mce::best_move_mce(&t, net, 750, &mut search_rng)
                        .map(|m| m.score as f32).unwrap_or(0.0);
                    samples += 1;
                }
            }
            if samples > 0 {
                let expected = total / samples as f32;
                if expected > baseline + 1.5 {
                    return Ok(Json(serde_json::json!({
                        "action": "mulligan",
                        "expected_gain": expected - baseline,
                    })));
                }
            }
        }
    }

    // Step 3: suggest the best move (plus top-10 alternatives with MCE scores)
    let scored_candidates: Vec<(cascadia_ai::eval::ScoredMove, f64)> = if let Some(ref net) = net_opt {
        cascadia_ai::mce::top_moves_mce(&game, net, 750, &mut search_rng, 10)
    } else {
        let mp: Vec<_> = game.market.available()
            .map(|(i, p)| (i, p.tile, p.wildlife)).collect();
        let player = game.current_player;
        let mut board = game.boards[player].clone();
        cascadia_ai::eval::best_move_with_potential(&mut board, &mp, &game.scoring_cards, game.turns_remaining)
            .into_iter().map(|m| (m, m.score as f64)).collect()
    };

    fn mv_to_json(mv: &cascadia_ai::eval::ScoredMove, avg_score: f64) -> serde_json::Value {
        let mut obj = serde_json::json!({
            "market_index": mv.market_index,
            "q": mv.tile_q,
            "r": mv.tile_r,
            "rotation": mv.rotation,
            "score": avg_score.round() as i32,
            "avg_score": avg_score,
        });
        if let (Some(wq), Some(wr)) = (mv.wildlife_q, mv.wildlife_r) {
            obj["wildlife_q"] = serde_json::json!(wq);
            obj["wildlife_r"] = serde_json::json!(wr);
        }
        if let Some(wmi) = mv.wildlife_market_index {
            obj["wildlife_market_index"] = serde_json::json!(wmi);
        }
        obj
    }

    if let Some((best_mv, best_avg)) = scored_candidates.first() {
        let mv_obj = mv_to_json(best_mv, *best_avg);
        let cand_list: Vec<serde_json::Value> = scored_candidates.iter()
            .map(|(m, avg)| mv_to_json(m, *avg))
            .collect();
        Ok(Json(serde_json::json!({
            "action": "move",
            "mv": mv_obj,
            "candidates": cand_list,
        })))
    } else {
        Err((StatusCode::BAD_REQUEST, "No moves available".to_string()))
    }
}

async fn best_move_endpoint(
    State(state): State<std::sync::Arc<AppState>>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let mut game = state.game.lock().unwrap();

    if game.is_game_over() {
        return Err((StatusCode::BAD_REQUEST, "Game is over".to_string()));
    }

    let cards = game.scoring_cards;

    // If NNUE weights are loaded, use MCE with mulligan-aware pre-move optimization
    // (our best-in-class strategy at 92.9 mean, P90=101)
    let mv = if let Some(ref net) = state.nnue {
        let mut rng = state.rng.lock().unwrap();
        let mut search_rng = StdRng::seed_from_u64(rng.gen());
        drop(rng);

        // Pre-move: decide whether to replace 3-of-a-kind or mulligan.
        // Commits actions to the REAL game state (snapshotted for undo).
        let mut applied_replace_overflow = false;
        let mut applied_mulligans = 0u32;
        {
            // Snapshot for undo BEFORE any pre-move actions
            let mut hist = state.history.lock().unwrap();
            hist.push(game.clone());
            if hist.len() > 50 { hist.remove(0); }
            drop(hist);

            const MULLIGAN_SAMPLES: usize = 3;
            const MAX_MULLIGANS: usize = 5;
            let player = game.current_player;

            let mut mulligans_used = 0;
            loop {
                let baseline = cascadia_ai::mce::best_move_mce(&game, net, 750, &mut search_rng)
                    .map(|m| m.score as f32).unwrap_or(0.0);

                // Always take the free 3-of-a-kind replacement (no downside — it's free)
                if let Some(overflow_wl) = game.can_replace_overflow() {
                    game.replace_overflow();
                    applied_replace_overflow = true;
                    state.events.lock().unwrap().push(
                        format!("P1 🤖 used free 3-of-a-kind replacement (3× {:?})", overflow_wl)
                    );
                    continue;
                }
                if mulligans_used < MAX_MULLIGANS && game.boards[player].nature_tokens > 0 {
                    let mut total = 0.0f32;
                    let mut samples = 0;
                    for _ in 0..MULLIGAN_SAMPLES {
                        let mut t = game.clone();
                        t.shuffle_bags(&mut search_rng);
                        if t.mulligan_wildlife() {
                            total += cascadia_ai::mce::best_move_mce(&t, net, 750, &mut search_rng)
                                .map(|m| m.score as f32).unwrap_or(0.0);
                            samples += 1;
                        }
                    }
                    if samples > 0 {
                        let expected = total / samples as f32;
                        if expected > baseline + 1.5 {
                            if game.mulligan_wildlife() {
                                mulligans_used += 1;
                                applied_mulligans += 1;
                                state.events.lock().unwrap().push(
                                    format!("P1 🤖 🌲 spent pinecone for mulligan (EV +{:.1})", expected - baseline)
                                );
                                continue;
                            }
                        }
                    }
                }
                break;
            }
        }

        // Pick best move from the (now possibly mulliganed) game state
        let best = cascadia_ai::mce::best_move_mce(&game, net, 750, &mut search_rng);
        if let Some(mv) = best {
            let mut result = serde_json::json!({
                "market_index": mv.market_index,
                "q": mv.tile_q,
                "r": mv.tile_r,
                "rotation": mv.rotation,
                "score": mv.score,
                "strategy": "mce+mulligan",
                "applied_replace_overflow": applied_replace_overflow,
                "applied_mulligans": applied_mulligans,
            });
            if let (Some(wq), Some(wr)) = (mv.wildlife_q, mv.wildlife_r) {
                result["wildlife_q"] = serde_json::json!(wq);
                result["wildlife_r"] = serde_json::json!(wr);
            }
            if let Some(wmi) = mv.wildlife_market_index {
                result["wildlife_market_index"] = serde_json::json!(wmi);
            }
            return Ok(Json(result));
        }
        None
    } else {
        // Fallback: greedy
        let market_pairs: Vec<_> = game.market.available()
            .map(|(i, pair)| (i, pair.tile, pair.wildlife)).collect();
        let player = game.current_player;
        let board = &mut game.boards[player];
        cascadia_ai::eval::best_move(board, &market_pairs, &cards)
    };

    match mv {
        Some(mv) => {
            let mut result = serde_json::json!({
                "market_index": mv.market_index,
                "q": mv.tile_q,
                "r": mv.tile_r,
                "rotation": mv.rotation,
                "score": mv.score,
                "strategy": "greedy",
            });
            if let (Some(wq), Some(wr)) = (mv.wildlife_q, mv.wildlife_r) {
                result["wildlife_q"] = serde_json::json!(wq);
                result["wildlife_r"] = serde_json::json!(wr);
            }
            if let Some(wmi) = mv.wildlife_market_index {
                result["wildlife_market_index"] = serde_json::json!(wmi);
            }
            Ok(Json(result))
        }
        None => Err((StatusCode::BAD_REQUEST, "No moves available".to_string())),
    }
}

async fn replace_overflow(
    State(state): State<std::sync::Arc<AppState>>,
) -> Result<Json<GameView>, (StatusCode, String)> {
    let mut game = state.game.lock().unwrap();
    if game.is_game_over() {
        return Err((StatusCode::BAD_REQUEST, "Game is over".to_string()));
    }
    // Snapshot for undo
    {
        let mut hist = state.history.lock().unwrap();
        hist.push(game.clone());
        if hist.len() > 50 { hist.remove(0); }
    }
    if !game.replace_overflow() {
        state.history.lock().unwrap().pop();
        return Err((StatusCode::BAD_REQUEST, "No 3-of-a-kind to replace".to_string()));
    }
    state.events.lock().unwrap().push("P1 used free 3-of-a-kind replacement".to_string());
    let events = state.events.lock().unwrap().clone();
    Ok(Json(build_game_view_with_events(&mut game, events)))
}

async fn undo(
    State(state): State<std::sync::Arc<AppState>>,
) -> Result<Json<GameView>, (StatusCode, String)> {
    let mut hist = state.history.lock().unwrap();
    let prev = hist.pop().ok_or((
        StatusCode::BAD_REQUEST,
        "Nothing to undo".to_string(),
    ))?;
    let mut game = state.game.lock().unwrap();
    *game = prev;
    Ok(Json(build_game_view(&mut game)))
}

async fn new_game(
    State(state): State<std::sync::Arc<AppState>>,
    Json(req): Json<NewGameRequest>,
) -> Json<GameView> {
    let num_players = req.num_players.unwrap_or(1).clamp(1, 4);
    let mut rng = state.rng.lock().unwrap();
    let new_game = GameState::new(num_players, ScoringCards::all_a(), &mut *rng);
    let mut game = state.game.lock().unwrap();
    *game = new_game;
    *state.solo_sim.lock().unwrap() = req.solo_sim.unwrap_or(false);
    state.history.lock().unwrap().clear();
    state.events.lock().unwrap().clear();
    state.events.lock().unwrap().push(format!(
        "New game: {} player{}{}",
        num_players,
        if num_players == 1 { "" } else { "s" },
        if req.solo_sim.unwrap_or(false) { " (solo sim)" } else { "" }
    ));
    Json(build_game_view(&mut game))
}

async fn index() -> Html<&'static str> {
    Html(include_str!("index.html"))
}

#[tokio::main]
async fn main() {
    let mut rng = StdRng::from_entropy();
    // Default to Solo 4p sim
    let game = GameState::new(4, ScoringCards::all_a(), &mut rng);

    // Try to load the best NNUE weights for MCE-powered suggestions
    let nnue = {
        let candidates = ["nnue_weights_mce93.bin", "nnue_weights.bin"];
        candidates.iter()
            .find_map(|path| {
                cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(path)).ok()
            })
            .map(std::sync::Arc::new)
    };
    if nnue.is_some() {
        println!("✓ Loaded NNUE weights — /api/best-move will use MCE with mulligan-aware pre-move");
    } else {
        println!("⚠ No NNUE weights found — /api/best-move will use greedy baseline");
    }

    let state = std::sync::Arc::new(AppState {
        game: Mutex::new(game),
        rng: Mutex::new(rng),
        nnue,
        solo_sim: Mutex::new(true), // default to solo 4p sim
        history: Mutex::new(Vec::new()),
        events: Mutex::new(Vec::new()),
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/api/state", get(get_state))
        .route("/api/move", post(make_move))
        .route("/api/undo", post(undo))
        .route("/api/new-game", post(new_game))
        .route("/api/mulligan", post(mulligan))
        .route("/api/replace-overflow", post(replace_overflow))
        .route("/api/best-move", get(best_move_endpoint))
        .route("/api/suggest-move", get(suggest_move))
        .with_state(state);

    let addr = "0.0.0.0:3000";
    println!("Cascadia web UI running at http://localhost:3000");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
