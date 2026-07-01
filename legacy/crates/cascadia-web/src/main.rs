use std::sync::Mutex;

use axum::{
    extract::State,
    http::StatusCode,
    response::Html,
    routing::{get, post},
    Json, Router,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use cascadia_core::game::{GameState, PlayerMove};
use cascadia_core::hex::HexCoord;
use cascadia_core::scoring::ScoreBreakdown;
use cascadia_core::types::*;

/// AI strength tiers for both the human player's suggestion AI and the
/// opponents' move selection. Ordered weakest → strongest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Strength {
    /// Plain greedy with potential-aware top-8 joint tile+wildlife scoring.
    /// No neural net required. ~76 mean in benchmarks.
    Greedy,
    /// NNUE direct evaluation (~25 candidate afterstate forward passes).
    /// Fast, deterministic. ~90 mean with v4 iter10 weights.
    Nnue,
    /// MCE with 50 rollouts. Medium-cost search; ~91 mean.
    Mce50,
    /// MCE with 750 rollouts + mulligan-aware pre-move. Production-strength
    /// ~95-96 mean. VERY SLOW for opponents (3-5 min per move × 3 opponents).
    Mce750,
}

impl Default for Strength {
    fn default() -> Self {
        Strength::Mce750
    }
}

impl Strength {
    fn label(&self) -> &'static str {
        match self {
            Strength::Greedy => "greedy",
            Strength::Nnue => "nnue",
            Strength::Mce50 => "mce(50)",
            Strength::Mce750 => "mce(750)+mulligan",
        }
    }
}

struct AppState {
    game: Mutex<GameState>,
    rng: Mutex<StdRng>,
    nnue: Option<std::sync::Arc<cascadia_ai::nnue::NNUENetwork>>,
    solo_sim: Mutex<bool>,
    history: Mutex<Vec<GameState>>,
    events: Mutex<Vec<String>>,
    /// Strength used for the human player's suggest/best-move (autoplay) AI.
    /// Default Mce750. Set via /api/set-strengths.
    human_strength: Mutex<Strength>,
    /// Strength used for opponents (players 1-3) in solo-sim mode.
    /// Default Nnue (fast). Set via /api/set-strengths.
    opponent_strength: Mutex<Strength>,
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
    bag_remaining: [u8; 5], // [Bear, Elk, Salmon, Hawk, Fox] remaining in bag
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
    /// Number of placed wildlife of each type. Used by UI to compute pts/animal.
    wildlife_counts: [u16; 5],
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

/// Parse a display-only ScoringCards override of the form "B,A,C,A,A"
/// (5 letters, one per wildlife in [Bear, Elk, Salmon, Hawk, Fox] order).
/// Falls back to all-A on any parse error or wrong length.
fn parse_display_cards(s: &str) -> Option<cascadia_core::types::ScoringCards> {
    use cascadia_core::types::{ScoringCardVariant, ScoringCards};
    let parts: Vec<&str> = s.split(',').map(|x| x.trim()).collect();
    if parts.len() != 5 {
        return None;
    }
    let mut cards = [ScoringCardVariant::A; 5];
    for (i, p) in parts.iter().enumerate() {
        cards[i] = match p.to_ascii_uppercase().as_str() {
            "A" => ScoringCardVariant::A,
            "B" => ScoringCardVariant::B,
            "C" => ScoringCardVariant::C,
            "D" => ScoringCardVariant::D,
            _ => return None,
        };
    }
    Some(ScoringCards { cards })
}

fn build_game_view_for_with(
    game: &mut GameState,
    view_player: usize,
    events: Vec<String>,
) -> GameView {
    build_game_view_for_with_cards(game, view_player, events, None)
}

fn build_game_view_for_with_cards(
    game: &mut GameState,
    view_player: usize,
    events: Vec<String>,
    display_cards: Option<cascadia_core::types::ScoringCards>,
) -> GameView {
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
    // Use display_cards override (UI selector) when provided; otherwise use the
    // game's actual scoring_cards (always Card A — AI logic is unaffected).
    let cards_for_display = display_cards.unwrap_or(game.scoring_cards);
    let scores: Vec<ScoreView> = (0..game.num_players)
        .map(|p| {
            let breakdown = if is_game_over && is_multiplayer {
                ScoreBreakdown::compute_with_bonuses(&mut game.boards, &cards_for_display, p)
            } else {
                ScoreBreakdown::compute(&mut game.boards[p], &cards_for_display)
            };
            let board = &game.boards[p];
            let mut wildlife_counts = [0u16; 5];
            for w in 0..5 {
                wildlife_counts[w] = board.wildlife_positions[w].len() as u16;
            }
            ScoreView {
                player: p,
                habitat: breakdown.habitat,
                wildlife: breakdown.wildlife,
                wildlife_counts,
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
        bag_remaining: {
            let bag_info = cascadia_ai::nnue::BagInfo::from_game(game);
            bag_info.remaining
        },
    }
}

#[derive(Deserialize)]
struct StateQuery {
    view: Option<usize>,
    /// Display-only scoring-card override, e.g. "B,A,C,A,D" for
    /// Bear=B, Elk=A, Salmon=C, Hawk=A, Fox=D. Server recomputes the score
    /// breakdowns with these cards but the game state and AI still use Card A.
    display_cards: Option<String>,
}

/// Pick a move for the given game state using the given AI strength.
/// Used by both the human-player suggest/autoplay path (Mce750 default) and
/// the opponent loop in solo_sim (Nnue default).
///
/// - `Greedy`: plain potential-aware greedy, no NNUE required
/// - `Nnue`: NNUE direct candidate-afterstate evaluation (requires net)
/// - `Mce50`: Monte Carlo with 50 rollouts (requires net)
/// - `Mce750`: Monte Carlo with 750 rollouts (requires net)
///
/// If the caller requests an NNUE-backed strength but no net is loaded,
/// falls back to greedy (with strategic candidates) so the caller always
/// gets a move.
fn pick_move_by_strength(
    game: &cascadia_core::game::GameState,
    strength: Strength,
    net: Option<&cascadia_ai::nnue::NNUENetwork>,
    rng: &Mutex<StdRng>,
) -> Option<cascadia_ai::eval::ScoredMove> {
    use cascadia_ai::nnue_train::pick_best_move_nnue;
    match strength {
        Strength::Greedy => cascadia_ai::mce::best_move_no_rollouts(game),
        Strength::Nnue => match net {
            Some(n) => pick_best_move_nnue(game, n),
            None => cascadia_ai::mce::best_move_no_rollouts(game),
        },
        Strength::Mce50 => match net {
            Some(n) => {
                let seed = rng.lock().unwrap().gen::<u64>();
                let mut search_rng = StdRng::seed_from_u64(seed);
                cascadia_ai::mce::best_move_mce(game, n, 50, &mut search_rng)
            }
            None => cascadia_ai::mce::best_move_no_rollouts(game),
        },
        Strength::Mce750 => match net {
            Some(n) => {
                let seed = rng.lock().unwrap().gen::<u64>();
                let mut search_rng = StdRng::seed_from_u64(seed);
                cascadia_ai::mce::best_move_mce(game, n, 750, &mut search_rng)
            }
            None => cascadia_ai::mce::best_move_no_rollouts(game),
        },
    }
}

async fn get_state(
    State(state): State<std::sync::Arc<AppState>>,
    axum::extract::Query(q): axum::extract::Query<StateQuery>,
) -> Json<GameView> {
    let mut game = state.game.lock().unwrap();
    let view = q.view.unwrap_or(game.current_player);
    let events = state.events.lock().unwrap().clone();
    let display_cards = q.display_cards.as_deref().and_then(parse_display_cards);
    Json(build_game_view_for_with_cards(
        &mut game,
        view,
        events,
        display_cards,
    ))
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
        (
            terrain.to_string(),
            terrain2.map(|s| s.to_string()),
            wildlife.to_string(),
        )
    });
    let independent_wildlife = req.wildlife_market_index.and_then(|wmi| {
        if wmi != req.market_index {
            game.market.pairs[wmi]
                .as_ref()
                .map(|p| wildlife_name(p.wildlife).to_string())
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
        return Err((StatusCode::BAD_REQUEST, "Invalid move".to_string()));
    }

    // Log the player's move
    {
        let mut events = state.events.lock().unwrap();
        // Use [T:terrain1] or [T:terrain1/terrain2] tags for frontend to render as colored boxes
        let tile_desc = drafted_tile_info.as_ref().map(|(t1, t2, wl)| {
            let terrain_tag = match t2 {
                Some(t2) => format!("[T:{}/{}]", t1, t2),
                None => format!("[T:{}]", t1),
            };
            (terrain_tag, wl.clone())
        });
        let action = match (independent_wildlife, tile_desc) {
            (Some(wl), Some((terrain, _))) => {
                format!("🌲 {} + {}", terrain, wl)
            }
            (None, Some((terrain, wl))) => {
                format!("{} + {}", terrain, wl)
            }
            _ => format!("drafted slot {}", req.market_index),
        };
        let wl_placed = req.wildlife_q.is_some();
        events.push(format!(
            "P1 {} at ({},{}){}",
            action,
            req.q,
            req.r,
            if wl_placed { "" } else { " (skipped wildlife)" }
        ));
    }

    // Solo 4p sim: auto-advance opponents using the configured opponent strength.
    //
    // Opponents ALWAYS take the free 3-of-a-kind replacement when available — this
    // is strictly an improvement over the drafted market state and any rational
    // player would take it. The CLI's `simulate_game_inner` was missing this and
    // the fix has been mirrored there so web and CLI benchmarks match.
    let solo_sim = *state.solo_sim.lock().unwrap();
    let opponent_strength = *state.opponent_strength.lock().unwrap();
    if solo_sim {
        while !game.is_game_over() && game.current_player != 0 {
            let p = game.current_player;
            if let Some(overflow_wl) = game.can_replace_overflow() {
                game.replace_overflow();
                state.events.lock().unwrap().push(format!(
                    "P{} used free replacement (3× {:?})",
                    p + 1,
                    overflow_wl
                ));
            }
            // Dispatch opponent move selection on the configured strength.
            let opp_mv =
                pick_move_by_strength(&game, opponent_strength, state.nnue.as_deref(), &state.rng);
            match opp_mv {
                Some(mv) => {
                    let market_idx = mv.market_index;
                    // Capture market info BEFORE executing
                    let tile_info = game.market.pairs[market_idx].as_ref().map(|pair| {
                        let t1 = terrain_name(pair.tile.terrain1).to_string();
                        let t2 = pair.tile.terrain2.map(|t| terrain_name(t).to_string());
                        let wl = wildlife_name(pair.wildlife).to_string();
                        let terrain_tag = match t2 {
                            Some(t2) => format!("[T:{}/{}]", t1, t2),
                            None => format!("[T:{}]", t1),
                        };
                        (terrain_tag, wl)
                    });
                    let indep_wl = mv.wildlife_market_index.and_then(|wmi| {
                        if wmi != market_idx {
                            game.market.pairs[wmi]
                                .as_ref()
                                .map(|p| wildlife_name(p.wildlife).to_string())
                        } else {
                            None
                        }
                    });
                    if !cascadia_ai::search::execute_scored_move(&mut game, &mv) {
                        break;
                    }
                    let desc = match (tile_info, indep_wl) {
                        (Some((terrain, _)), Some(wl)) => format!("🌲 {} + {}", terrain, wl),
                        (Some((terrain, wl)), None) => format!("{} + {}", terrain, wl),
                        _ => format!("drafted slot {}", market_idx),
                    };
                    state.events.lock().unwrap().push(format!(
                        "P{} {} at ({},{})",
                        p + 1,
                        desc,
                        mv.tile_q,
                        mv.tile_r
                    ));
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
        if hist.len() > 50 {
            hist.remove(0);
        }
    }
    if !game.mulligan_wildlife() {
        state.history.lock().unwrap().pop();
        return Err((
            StatusCode::BAD_REQUEST,
            "No nature tokens to spend".to_string(),
        ));
    }
    state
        .events
        .lock()
        .unwrap()
        .push("P1 🌲 spent pinecone for mulligan".to_string());
    let events = state.events.lock().unwrap().clone();
    Ok(Json(build_game_view_with_events(&mut game, events)))
}

/// Evaluate pre-move actions using enumerated mulligan analysis (exact EV).
fn evaluate_pre_moves(
    game: &cascadia_core::game::GameState,
    net: &cascadia_ai::nnue::NNUENetwork,
    _search_rng: &mut StdRng,
) -> Option<serde_json::Value> {
    let analysis = cascadia_ai::mce::analyze_mulligan_fast(game, net);

    // Check free 3-of-a-kind replace
    if game.can_replace_overflow().is_some() {
        let mut test = game.clone();
        test.replace_overflow();
        let post = cascadia_ai::mce::analyze_mulligan_fast(&test, net);
        return Some(serde_json::json!({
            "type": "replace_overflow",
            "score_before": (analysis.current_best * 10.0).round() / 10.0,
            "score_after": (post.current_best * 10.0).round() / 10.0,
            "recommended": post.current_best > analysis.current_best,
        }));
    }

    // Check mulligan (exact enumeration)
    let player = game.current_player;
    if game.boards[player].nature_tokens > 0 {
        let gain = analysis.mulligan_ev - analysis.current_best;
        return Some(serde_json::json!({
            "type": "mulligan",
            "score_before": (analysis.current_best * 10.0).round() / 10.0,
            "score_after": (analysis.mulligan_ev * 10.0).round() / 10.0,
            "expected_gain": (gain * 10.0).round() / 10.0,
            "recommended": analysis.should_mulligan,
        }));
    }

    None
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

    let net_opt = state.nnue.clone();
    let human_strength = *state.human_strength.lock().unwrap();
    let mut rng = state.rng.lock().unwrap();
    let mut search_rng = StdRng::seed_from_u64(rng.gen());
    drop(rng);

    // Pre-move evaluation only makes sense when we have a net and a search-based strength.
    // For pure greedy, there's no mulligan reasoning — just flag overflow if available.
    let use_mulligan_reasoning = matches!(
        human_strength,
        Strength::Nnue | Strength::Mce50 | Strength::Mce750
    );
    let pre_action = if use_mulligan_reasoning {
        if let Some(ref net) = net_opt {
            evaluate_pre_moves(&game, net, &mut search_rng)
        } else if game.can_replace_overflow().is_some() {
            Some(serde_json::json!({ "type": "replace_overflow", "recommended": true }))
        } else {
            None
        }
    } else if game.can_replace_overflow().is_some() {
        Some(serde_json::json!({ "type": "replace_overflow", "recommended": true }))
    } else {
        None
    };

    // Compute top-10 candidates using the configured strength.
    let scored_candidates: Vec<(cascadia_ai::eval::ScoredMove, f64)> =
        match (human_strength, net_opt.as_ref()) {
            (Strength::Mce750, Some(net)) => {
                cascadia_ai::mce::top_moves_mce(&game, net, 750, &mut search_rng, 10)
            }
            (Strength::Mce50, Some(net)) => {
                cascadia_ai::mce::top_moves_mce(&game, net, 50, &mut search_rng, 10)
            }
            (Strength::Nnue, Some(net)) => {
                // NNUE-only: use pick_best_move_nnue to get the single best move,
                // then fall back to greedy candidates for the remaining 9 slots.
                // This is a "best + context" presentation for the UI.
                let mut cands: Vec<(cascadia_ai::eval::ScoredMove, f64)> = Vec::new();
                if let Some(best) = cascadia_ai::nnue_train::pick_best_move_nnue(&game, net) {
                    cands.push((best, best.score as f64));
                }
                let mp: Vec<_> = game
                    .market
                    .available()
                    .map(|(i, p)| (i, p.tile, p.wildlife))
                    .collect();
                let mut board = game.boards[game.current_player].clone();
                if let Some(g) = cascadia_ai::eval::best_move_with_potential(
                    &mut board,
                    &mp,
                    &game.scoring_cards,
                    game.turns_remaining,
                ) {
                    if cands.iter().all(|(m, _)| {
                        !(m.tile_q == g.tile_q
                            && m.tile_r == g.tile_r
                            && m.rotation == g.rotation
                            && m.wildlife_q == g.wildlife_q)
                    }) {
                        cands.push((g, g.score as f64));
                    }
                }
                cands
            }
            // Greedy (or no-net fallback): just greedy candidates
            _ => {
                let mp: Vec<_> = game
                    .market
                    .available()
                    .map(|(i, p)| (i, p.tile, p.wildlife))
                    .collect();
                let player = game.current_player;
                let mut board = game.boards[player].clone();
                cascadia_ai::eval::best_move_with_potential(
                    &mut board,
                    &mp,
                    &game.scoring_cards,
                    game.turns_remaining,
                )
                .into_iter()
                .map(|m| (m, m.score as f64))
                .collect()
            }
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
        let cand_list: Vec<serde_json::Value> = scored_candidates
            .iter()
            .map(|(m, avg)| mv_to_json(m, *avg))
            .collect();
        let mut result = serde_json::json!({
            "action": if pre_action.as_ref().map(|pa| pa["recommended"].as_bool().unwrap_or(true)).unwrap_or(false) {
                match pre_action.as_ref().unwrap()["type"].as_str().unwrap_or("") {
                    "replace_overflow" => "replace_overflow",
                    "mulligan" => "mulligan",
                    _ => "move",
                }
            } else { "move" },
            "strategy": human_strength.label(),
            "mv": mv_obj,
            "candidates": cand_list,
        });
        if let Some(ref pa) = pre_action {
            result["pre_action"] = pa.clone();
            if let Some(gain) = pa.get("expected_gain") {
                result["expected_gain"] = gain.clone();
            }
        }
        Ok(Json(result))
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

    let human_strength = *state.human_strength.lock().unwrap();
    let use_mulligan = matches!(
        human_strength,
        Strength::Nnue | Strength::Mce50 | Strength::Mce750
    );

    // If the strength is search-based and weights are loaded, run mulligan-aware
    // pre-move optimization. This is IDENTICAL to the CLI's `pre_move_optimize` so
    // the web player's scoring matches CLI benchmarks:
    //   - `analyze_mulligan_fast` for exact enumerated EV (over all 625 possible draws)
    //   - Same replace_overflow / mulligan / mulligan_pinecone priority chain
    //   - Same MAX_MULLIGANS = 5 cap
    //
    // Previously this used a 3-sample greedy heuristic which under-performed the
    // CLI by ~1.4 points in 4-game web-API self-tests. The enumerated version
    // matches the CLI bench (95.9 base mean).
    let mut applied_replace_overflow = false;
    let mut applied_mulligans = 0u32;
    if use_mulligan {
        if let Some(ref net) = state.nnue {
            // Snapshot for undo BEFORE any pre-move actions
            let mut hist = state.history.lock().unwrap();
            hist.push(game.clone());
            if hist.len() > 50 {
                hist.remove(0);
            }
            drop(hist);

            const MAX_MULLIGANS: usize = 5;
            let mut mulligans_used = 0usize;
            loop {
                let analysis = cascadia_ai::mce::analyze_mulligan_fast(&game, net);

                // Option 1: Free 3-of-a-kind replacement — only if it improves
                //           current_best (not just the header row).
                if let Some(overflow_wl) = game.can_replace_overflow() {
                    let mut test = game.clone();
                    test.replace_overflow();
                    let post_analysis = cascadia_ai::mce::analyze_mulligan_fast(&test, net);
                    if post_analysis.current_best > analysis.current_best {
                        let gain = post_analysis.current_best - analysis.current_best;
                        game.replace_overflow();
                        applied_replace_overflow = true;
                        state.events.lock().unwrap().push(format!(
                            "P1 🤖 used free 3-of-a-kind replacement (3× {:?}, +{:.1})",
                            overflow_wl, gain
                        ));
                        continue;
                    }
                }

                // Option 2: Enumerated single-token mulligan
                if mulligans_used < MAX_MULLIGANS && analysis.should_mulligan {
                    if game.mulligan_wildlife() {
                        mulligans_used += 1;
                        applied_mulligans += 1;
                        let ev_gain = analysis.mulligan_ev - 1.0 - analysis.current_best;
                        state.events.lock().unwrap().push(format!(
                            "P1 🤖 🌲 spent pinecone for mulligan (EV +{:.1})",
                            ev_gain
                        ));
                        continue;
                    }
                }

                // Option 3: Mulligan + pinecone (2-token, exact EV)
                if mulligans_used < MAX_MULLIGANS && analysis.should_mulligan_pinecone {
                    if game.mulligan_wildlife() {
                        mulligans_used += 1;
                        applied_mulligans += 1;
                        let ev_gain = analysis.mulligan_pinecone_ev - 2.0 - analysis.current_best;
                        state.events.lock().unwrap().push(format!(
                            "P1 🤖 🌲🌲 spent 2 pinecones for mulligan (EV +{:.1})",
                            ev_gain
                        ));
                        continue;
                    }
                }

                break;
            }
        }
    }

    // Pick best move using the configured strength.
    let mv = pick_move_by_strength(&game, human_strength, state.nnue.as_deref(), &state.rng);

    if let Some(mv) = mv {
        let mut result = serde_json::json!({
            "market_index": mv.market_index,
            "q": mv.tile_q,
            "r": mv.tile_r,
            "rotation": mv.rotation,
            "score": mv.score,
            "strategy": human_strength.label(),
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
    Err((StatusCode::BAD_REQUEST, "No moves available".to_string()))
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
        if hist.len() > 50 {
            hist.remove(0);
        }
    }
    if !game.replace_overflow() {
        state.history.lock().unwrap().pop();
        return Err((
            StatusCode::BAD_REQUEST,
            "No 3-of-a-kind to replace".to_string(),
        ));
    }
    state
        .events
        .lock()
        .unwrap()
        .push("P1 used free 3-of-a-kind replacement".to_string());
    let events = state.events.lock().unwrap().clone();
    Ok(Json(build_game_view_with_events(&mut game, events)))
}

async fn undo(
    State(state): State<std::sync::Arc<AppState>>,
) -> Result<Json<GameView>, (StatusCode, String)> {
    let mut hist = state.history.lock().unwrap();
    let prev = hist
        .pop()
        .ok_or((StatusCode::BAD_REQUEST, "Nothing to undo".to_string()))?;
    let mut game = state.game.lock().unwrap();
    *game = prev;
    Ok(Json(build_game_view(&mut game)))
}

#[derive(Serialize)]
struct StrengthsResponse {
    human: Strength,
    opponent: Strength,
    nnue_available: bool,
}

#[derive(Deserialize)]
struct SetStrengthsRequest {
    human: Option<Strength>,
    opponent: Option<Strength>,
}

async fn get_strengths(State(state): State<std::sync::Arc<AppState>>) -> Json<StrengthsResponse> {
    Json(StrengthsResponse {
        human: *state.human_strength.lock().unwrap(),
        opponent: *state.opponent_strength.lock().unwrap(),
        nnue_available: state.nnue.is_some(),
    })
}

async fn set_strengths(
    State(state): State<std::sync::Arc<AppState>>,
    Json(req): Json<SetStrengthsRequest>,
) -> Json<StrengthsResponse> {
    if let Some(h) = req.human {
        *state.human_strength.lock().unwrap() = h;
    }
    if let Some(o) = req.opponent {
        *state.opponent_strength.lock().unwrap() = o;
    }
    Json(StrengthsResponse {
        human: *state.human_strength.lock().unwrap(),
        opponent: *state.opponent_strength.lock().unwrap(),
        nnue_available: state.nnue.is_some(),
    })
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
        if req.solo_sim.unwrap_or(false) {
            " (solo sim)"
        } else {
            ""
        }
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

    // Try to load the best NNUE weights for MCE-powered suggestions.
    //
    // Order is FEATURE-GATED: each binary prioritises weights that natively
    // match its compiled feature set, because NNUENetwork::load is permissive
    // and will silently load mismatched weights with truncation/zero-padding,
    // producing a network that "loads" but plays poorly. Putting the matched
    // file first ensures the right pairing wins.
    #[cfg(feature = "v6-peak")]
    let nnue_candidates: &[&str] = &[
        "nnue_weights_v6peak_iter20.bin", // native — v6peak features (17,608)
        "nnue_weights_v4opp_modal_iter3.bin", // partial-match fallback
        "nnue_weights_mce93.bin",
        "nnue_weights.bin",
    ];
    #[cfg(all(feature = "v4-opp", not(feature = "v6-peak")))]
    let nnue_candidates: &[&str] = &[
        "nnue_weights_v4opp_modal_iter3.bin", // native — v4opp features (11,231)
        "nnue_weights_v3_iter20.bin",
        "nnue_weights_v4_iter10.bin",
        "nnue_weights_mce93.bin",
        "nnue_weights.bin",
    ];
    #[cfg(not(any(feature = "v4-opp", feature = "v6-peak")))]
    let nnue_candidates: &[&str] = &[
        "nnue_weights_v3_iter20.bin",
        "nnue_weights_v4_iter10.bin",
        "nnue_weights_mce93.bin",
        "nnue_weights.bin",
    ];
    let mut loaded_weight_path: Option<&str> = None;
    let nnue = nnue_candidates
        .iter()
        .find_map(
            |&path| match cascadia_ai::nnue::NNUENetwork::load(std::path::Path::new(path)) {
                Ok(net) => {
                    loaded_weight_path = Some(path);
                    Some(net)
                }
                Err(_) => None,
            },
        )
        .map(std::sync::Arc::new);
    if let Some(path) = loaded_weight_path {
        println!("✓ Loaded NNUE weights from {}", path);
        println!("  /api/best-move will use MCE with mulligan-aware pre-move");
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
        human_strength: Mutex::new(Strength::Mce750),
        opponent_strength: Mutex::new(Strength::Nnue),
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
        .route("/api/strengths", get(get_strengths).post(set_strengths))
        .with_state(state);

    let addr = "0.0.0.0:3000";
    println!("Cascadia web UI running at http://localhost:3000");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
