//! Stateless local application services over the canonical v2 rules engine.

mod champion;
mod cluster;
mod cluster_history;

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Json, Router,
    extract::{Query, State},
    http::{Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use cascadia_game::{
    Board, DraftChoice, GameConfig, GameSeed, GameState, HexCoord, Market, MarketPrelude,
    MarketSlot, PlacedTile, Replay, RuleError, ScoreBreakdown, Terrain, Tile, TurnAction, Wildlife,
    score_board, score_game,
};
use cascadia_search::{
    LateConservativeBasePolicyImprovementConfig, LateConservativeBasePolicyImprovementStrategy,
};
use cascadia_sim::{
    PATTERN_AWARE_STRATEGY_ID, PatternAwareConfig, rank_greedy_actions, rank_pattern_actions,
};
use serde::{Deserialize, Serialize};
use tower_http::{
    cors::{Any, CorsLayer},
    services::{ServeDir, ServeFile},
};

const API_SCHEMA_VERSION: u16 = 2;
const SAVED_GAME_SCHEMA_VERSION: u16 = 1;

pub fn router(static_dir: Option<PathBuf>) -> Router {
    router_with_cluster_history(
        static_dir,
        Path::new("artifacts/cluster/telemetry-v1.jsonl"),
    )
    .expect("cluster history store should initialize")
}

pub fn router_with_cluster_history(
    static_dir: Option<PathBuf>,
    history_path: impl Into<PathBuf>,
) -> std::io::Result<Router> {
    let history = Arc::new(cluster_history::ClusterHistoryStore::open(
        history_path,
        unix_time_millis(),
    )?);
    cluster_history::spawn_sampler(Arc::clone(&history));
    let state = AppState { history };

    let app = Router::new()
        .route("/api/v1/health", get(health))
        .route("/api/v1/cluster", get(cluster_health))
        .route("/api/v1/cluster/history", get(cluster_history))
        .route("/api/v1/capabilities", get(capabilities))
        .route("/api/v1/games/new", post(new_game_handler))
        .route("/api/v1/games/view", post(view_game_handler))
        .route("/api/v1/games/turn-options", post(turn_options_handler))
        .route(
            "/api/v1/games/placement-options",
            post(placement_options_handler),
        )
        .route("/api/v1/games/apply", post(apply_handler))
        .route("/api/v1/games/undo", post(undo_handler))
        .route("/api/v1/games/suggest", post(suggest_handler))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_headers(Any)
                .allow_methods([Method::GET, Method::POST]),
        )
        .with_state(state);

    Ok(if let Some(static_dir) = static_dir {
        let index = static_dir.join("index.html");
        app.fallback_service(ServeDir::new(static_dir).fallback(ServeFile::new(index)))
    } else {
        app
    })
}

#[derive(Clone)]
struct AppState {
    history: Arc<cluster_history::ClusterHistoryStore>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub api_schema_version: u16,
}

#[derive(Debug, Clone, Serialize)]
pub struct CapabilitiesResponse {
    pub api_schema_version: u16,
    pub strengths: Vec<StrengthCapability>,
    pub max_players: u8,
    pub scoring_variants: [&'static str; 4],
}

#[derive(Debug, Clone, Serialize)]
pub struct StrengthCapability {
    pub id: &'static str,
    pub label: &'static str,
    pub available: bool,
    pub latency: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedGame {
    pub schema_version: u16,
    pub seed: u64,
    pub replay: Replay,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewGameRequest {
    pub seed: u64,
    pub config: GameConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedGameRequest {
    pub game: SavedGame,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnOptionsRequest {
    pub game: SavedGame,
    #[serde(default)]
    pub prelude: MarketPrelude,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementOptionsRequest {
    pub game: SavedGame,
    #[serde(default)]
    pub prelude: MarketPrelude,
    pub draft: DraftChoice,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplyTurnRequest {
    pub game: SavedGame,
    pub action: TurnAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestRequest {
    pub game: SavedGame,
    #[serde(default = "default_candidate_count")]
    pub candidates: usize,
    #[serde(default)]
    pub strength: SuggestionStrength,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SuggestionStrength {
    #[default]
    Instant,
    Interactive,
    Research,
    Champion,
    ChampionDeep,
}

fn ordered(value: Option<&f64>) -> ordered_float_key::Key {
    ordered_float_key::Key(value.copied().unwrap_or(f64::NEG_INFINITY))
}

mod ordered_float_key {
    #[derive(PartialEq)]
    pub struct Key(pub f64);
    impl Eq for Key {}
    impl PartialOrd for Key {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Key {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0.total_cmp(&other.0)
        }
    }
}

fn default_candidate_count() -> usize {
    8
}

#[derive(Debug, Clone, Serialize)]
pub struct GameDocument {
    pub game: SavedGame,
    pub view: GameView,
}

#[derive(Debug, Clone, Serialize)]
pub struct GameView {
    pub config: GameConfig,
    pub seed: u64,
    pub current_player: usize,
    pub completed_turns: u16,
    pub total_turns: u16,
    pub game_over: bool,
    pub state_hash: String,
    pub free_overpopulation_replacement: bool,
    pub boards: Vec<BoardView>,
    pub market: Vec<MarketPairView>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BoardView {
    pub player: usize,
    pub nature_tokens: u8,
    pub tiles: Vec<BoardTileView>,
    pub frontier: Vec<HexCoord>,
    pub score: ScoreBreakdown,
}

#[derive(Debug, Clone, Serialize)]
pub struct BoardTileView {
    pub coord: HexCoord,
    pub tile: TileView,
    pub rotation: u8,
    pub wildlife: Option<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TileView {
    pub id: u8,
    pub terrain_a: &'static str,
    pub terrain_b: Option<&'static str>,
    pub wildlife: Vec<&'static str>,
    pub keystone: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct MarketPairView {
    pub slot: u8,
    pub tile: Option<TileView>,
    pub wildlife: Option<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TurnOptionsResponse {
    pub market: Vec<MarketPairView>,
    pub drafts: Vec<DraftChoice>,
    pub nature_tokens_remaining: u8,
    pub can_add_paid_wipe: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlacementOptionsResponse {
    pub placements: Vec<PlacementOption>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlacementOption {
    pub coord: HexCoord,
    pub rotations: Vec<RotationOption>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RotationOption {
    pub rotation: u8,
    pub wildlife: Vec<HexCoord>,
    pub may_skip_wildlife: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SuggestionResponse {
    pub strategy: String,
    pub candidates: Vec<SuggestionCandidate>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SuggestionCandidate {
    pub rank: usize,
    pub action: TurnAction,
    pub resulting_score: ScoreBreakdown,
    pub base_score_delta: i32,
    pub search_value: f64,
    pub evaluation_stddev: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiErrorBody {
    pub code: &'static str,
    pub message: String,
}

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn bad_request(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code,
            message: message.into(),
        }
    }

    fn from_rules(error: RuleError) -> Self {
        Self::bad_request("invalid-turn", error.to_string())
    }

    fn internal(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ApiErrorBody {
                code: self.code,
                message: self.message,
            }),
        )
            .into_response()
    }
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        api_schema_version: API_SCHEMA_VERSION,
    })
}

async fn cluster_health(
    State(state): State<AppState>,
) -> Result<Json<cluster::ClusterResponse>, ApiError> {
    let history = Arc::clone(&state.history);
    let response = tokio::task::spawn_blocking(move || {
        let response = cluster::collect_cluster();
        history.record(&response).map(|_| response)
    })
    .await
    .map_err(|error| ApiError::internal("cluster-probe-failed", error.to_string()))?
    .map_err(|error| ApiError::internal("cluster-history-write-failed", error.to_string()))?;
    Ok(Json(response))
}

async fn cluster_history(
    State(state): State<AppState>,
    Query(query): Query<cluster_history::ClusterHistoryQuery>,
) -> Result<Json<cluster_history::ClusterHistoryResponse>, ApiError> {
    state
        .history
        .query(query.range, unix_time_millis())
        .map(Json)
        .map_err(|error| ApiError::internal("cluster-history-read-failed", error.to_string()))
}

fn unix_time_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

async fn capabilities() -> Json<CapabilitiesResponse> {
    Json(CapabilitiesResponse {
        api_schema_version: API_SCHEMA_VERSION,
        strengths: vec![
            StrengthCapability {
                id: "instant",
                label: "Instant",
                available: true,
                latency: "near-instant",
            },
            StrengthCapability {
                id: "interactive",
                label: "Interactive",
                available: true,
                latency: "typically under 25 ms per move",
            },
            StrengthCapability {
                id: "research",
                label: "Research",
                available: true,
                latency: "typically under 500 ms per move",
            },
            StrengthCapability {
                id: "champion",
                label: "Champion (CascadiaFormer + Gumbel search)",
                available: champion::champion_available(),
                latency: "roughly 3-10 s per move",
            },
            StrengthCapability {
                id: "champion-deep",
                label: "Champion Deep (full n1024/d16 search)",
                available: champion::champion_available(),
                latency: "roughly 10-30 s per move",
            },
        ],
        max_players: 4,
        scoring_variants: ["A", "B", "C", "D"],
    })
}

async fn new_game_handler(
    Json(request): Json<NewGameRequest>,
) -> Result<Json<GameDocument>, ApiError> {
    Ok(Json(create_game(request)?))
}

async fn view_game_handler(
    Json(request): Json<SavedGameRequest>,
) -> Result<Json<GameDocument>, ApiError> {
    Ok(Json(view_game(request.game)?))
}

async fn turn_options_handler(
    Json(request): Json<TurnOptionsRequest>,
) -> Result<Json<TurnOptionsResponse>, ApiError> {
    Ok(Json(turn_options(request)?))
}

async fn placement_options_handler(
    Json(request): Json<PlacementOptionsRequest>,
) -> Result<Json<PlacementOptionsResponse>, ApiError> {
    Ok(Json(placement_options(request)?))
}

async fn apply_handler(
    Json(request): Json<ApplyTurnRequest>,
) -> Result<Json<GameDocument>, ApiError> {
    Ok(Json(apply_turn(request)?))
}

async fn undo_handler(
    Json(request): Json<SavedGameRequest>,
) -> Result<Json<GameDocument>, ApiError> {
    Ok(Json(undo_turn(request.game)?))
}

async fn suggest_handler(
    Json(request): Json<SuggestRequest>,
) -> Result<Json<SuggestionResponse>, ApiError> {
    let response = tokio::task::spawn_blocking(move || suggest(request))
        .await
        .map_err(|error| ApiError::bad_request("suggestion-failed", error.to_string()))??;
    Ok(Json(response))
}

pub fn create_game(request: NewGameRequest) -> Result<GameDocument, ApiError> {
    let seed = GameSeed::from_u64(request.seed);
    GameState::new(request.config, seed).map_err(ApiError::from_rules)?;
    view_game(SavedGame {
        schema_version: SAVED_GAME_SCHEMA_VERSION,
        seed: request.seed,
        replay: Replay::new(request.config, seed),
    })
}

pub fn view_game(mut saved: SavedGame) -> Result<GameDocument, ApiError> {
    let state = load_saved_game(&saved)?;
    if state.is_game_over() {
        saved.replay.final_state_hash = Some(*state.canonical_hash().as_bytes());
    }
    Ok(GameDocument {
        view: game_view(&state, saved.seed),
        game: saved,
    })
}

pub fn turn_options(request: TurnOptionsRequest) -> Result<TurnOptionsResponse, ApiError> {
    let state = load_saved_game(&request.game)?;
    let staged = state
        .preview_market_prelude(&request.prelude)
        .map_err(ApiError::from_rules)?;
    let tokens = staged.boards()[staged.current_player()].nature_tokens();
    let mut drafts: Vec<_> = MarketSlot::ALL
        .into_iter()
        .filter(|slot| staged.market().paired(*slot).is_some())
        .map(|slot| DraftChoice::Paired { slot })
        .collect();
    if tokens > 0 {
        for tile_slot in MarketSlot::ALL {
            for wildlife_slot in MarketSlot::ALL {
                if staged.market().tiles[tile_slot.index()].is_some()
                    && staged.market().wildlife[wildlife_slot.index()].is_some()
                {
                    drafts.push(DraftChoice::Independent {
                        tile_slot,
                        wildlife_slot,
                    });
                }
            }
        }
    }
    Ok(TurnOptionsResponse {
        market: market_view(staged.market()),
        drafts,
        nature_tokens_remaining: tokens,
        can_add_paid_wipe: tokens > 0,
    })
}

pub fn placement_options(
    request: PlacementOptionsRequest,
) -> Result<PlacementOptionsResponse, ApiError> {
    let state = load_saved_game(&request.game)?;
    let actions = state
        .legal_turn_actions_for_draft(&request.prelude, request.draft)
        .map_err(ApiError::from_rules)?;
    let mut grouped: BTreeMap<HexCoord, BTreeMap<u8, (bool, Vec<HexCoord>)>> = BTreeMap::new();
    for action in actions {
        let rotation = action.tile.rotation.get();
        let entry = grouped
            .entry(action.tile.coord)
            .or_default()
            .entry(rotation)
            .or_insert_with(|| (false, Vec::new()));
        if let Some(coord) = action.wildlife {
            entry.1.push(coord);
        } else {
            entry.0 = true;
        }
    }
    let placements = grouped
        .into_iter()
        .map(|(coord, rotations)| PlacementOption {
            coord,
            rotations: rotations
                .into_iter()
                .map(|(rotation, (may_skip_wildlife, mut wildlife))| {
                    wildlife.sort_unstable();
                    wildlife.dedup();
                    RotationOption {
                        rotation,
                        wildlife,
                        may_skip_wildlife,
                    }
                })
                .collect(),
        })
        .collect();
    Ok(PlacementOptionsResponse { placements })
}

pub fn apply_turn(request: ApplyTurnRequest) -> Result<GameDocument, ApiError> {
    let mut state = load_saved_game(&request.game)?;
    state.apply(&request.action).map_err(ApiError::from_rules)?;
    let mut replay = request.game.replay;
    replay.final_state_hash = None;
    replay.turns.push(request.action);
    view_game(SavedGame {
        schema_version: SAVED_GAME_SCHEMA_VERSION,
        seed: request.game.seed,
        replay,
    })
}

pub fn undo_turn(mut saved: SavedGame) -> Result<GameDocument, ApiError> {
    load_saved_game(&saved)?;
    if saved.replay.turns.pop().is_none() {
        return Err(ApiError::bad_request(
            "nothing-to-undo",
            "the game has no completed turns",
        ));
    }
    saved.replay.final_state_hash = None;
    view_game(saved)
}

pub fn suggest(request: SuggestRequest) -> Result<SuggestionResponse, ApiError> {
    let state = load_saved_game(&request.game)?;
    if state.is_game_over() {
        return Err(ApiError::bad_request(
            "game-over",
            "the completed game has no next move",
        ));
    }
    let current_score = score_board(
        &state.boards()[state.current_player()],
        state.config().scoring_cards,
    );
    let limit = request.candidates.clamp(1, 32);
    match request.strength {
        SuggestionStrength::Instant => {
            let prelude = MarketPrelude {
                replace_three_of_a_kind: state.market().three_of_a_kind().is_some(),
                wildlife_wipes: Vec::new(),
            };
            let ranked = rank_greedy_actions(&state, &prelude, Some(limit))
                .map_err(|error| ApiError::bad_request("suggestion-failed", error.to_string()))?;
            Ok(SuggestionResponse {
                strategy: "greedy-v1".to_owned(),
                candidates: ranked
                    .into_iter()
                    .enumerate()
                    .map(|(index, candidate)| {
                        suggestion_candidate(
                            &state,
                            current_score,
                            index,
                            candidate.action,
                            f64::from(candidate.resulting_base_score),
                            0.0,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            })
        }
        SuggestionStrength::Interactive => {
            let prelude = MarketPrelude {
                replace_three_of_a_kind: state.market().three_of_a_kind().is_some(),
                wildlife_wipes: Vec::new(),
            };
            let ranked = rank_pattern_actions(&state, &prelude, PatternAwareConfig::default())
                .map_err(|error| ApiError::bad_request("suggestion-failed", error.to_string()))?;
            Ok(SuggestionResponse {
                strategy: PATTERN_AWARE_STRATEGY_ID.to_owned(),
                candidates: ranked
                    .into_iter()
                    .take(limit)
                    .enumerate()
                    .map(|(index, candidate)| {
                        suggestion_candidate(
                            &state,
                            current_score,
                            index,
                            candidate.action,
                            candidate.heuristic_value,
                            0.0,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            })
        }
        SuggestionStrength::Research => research_suggestions(&state, current_score, limit),
        SuggestionStrength::Champion | SuggestionStrength::ChampionDeep => {
            let deep = matches!(request.strength, SuggestionStrength::ChampionDeep);
            let reply = champion::suggest(&state, deep)
                .map_err(|error| ApiError::bad_request("suggestion-failed", error))?;
            if reply.game_over || reply.actions.is_empty() {
                return Err(ApiError::bad_request(
                    "suggestion-failed",
                    "the champion found no legal actions",
                ));
            }
            // Chosen action first (the search's actual pick), remaining
            // menu ranked by completed Q.
            let mut order: Vec<usize> = (0..reply.actions.len()).collect();
            order.sort_by(|&left, &right| {
                let left_key = (left != reply.chosen_index, std::cmp::Reverse(ordered(reply.completed_q.get(left))));
                let right_key = (right != reply.chosen_index, std::cmp::Reverse(ordered(reply.completed_q.get(right))));
                left_key.cmp(&right_key)
            });
            Ok(SuggestionResponse {
                strategy: if deep {
                    "cascadiaformer-distq-gumbel-n1024d16".to_owned()
                } else {
                    "cascadiaformer-distq-gumbel-n256d4".to_owned()
                },
                candidates: order
                    .into_iter()
                    .take(limit)
                    .enumerate()
                    .map(|(rank, action_index)| {
                        suggestion_candidate(
                            &state,
                            current_score,
                            rank,
                            reply.actions[action_index].clone(),
                            reply.completed_q.get(action_index).copied().unwrap_or(0.0),
                            0.0,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            })
        }
    }
}

fn research_suggestions(
    state: &GameState,
    current_score: ScoreBreakdown,
    limit: usize,
) -> Result<SuggestionResponse, ApiError> {
    let strategy = LateConservativeBasePolicyImprovementStrategy::new(
        LateConservativeBasePolicyImprovementConfig::default(),
    )
    .map_err(|error| ApiError::bad_request("suggestion-failed", error.to_string()))?;
    let prelude = MarketPrelude {
        replace_three_of_a_kind: state.market().three_of_a_kind().is_some(),
        wildlife_wipes: Vec::new(),
    };
    let pattern_ranked = rank_pattern_actions(state, &prelude, PatternAwareConfig::default())
        .map_err(|error| ApiError::bad_request("suggestion-failed", error.to_string()))?;
    let anchor = pattern_ranked
        .first()
        .ok_or_else(|| ApiError::bad_request("suggestion-failed", "no legal actions"))?
        .action
        .clone();

    if !strategy.uses_terminal_search(state) {
        return Ok(SuggestionResponse {
            strategy: strategy.strategy_id().to_owned(),
            candidates: pattern_ranked
                .into_iter()
                .take(limit)
                .enumerate()
                .map(|(index, candidate)| {
                    suggestion_candidate(
                        state,
                        current_score,
                        index,
                        candidate.action,
                        candidate.heuristic_value,
                        0.0,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
        });
    }

    let (mut ranked, selected) = strategy
        .rank_and_select_terminal_deterministic(state, &anchor)
        .map_err(|error| ApiError::bad_request("suggestion-failed", error.to_string()))?;
    let selected_index = ranked
        .iter()
        .position(|candidate| candidate.action == selected)
        .ok_or_else(|| {
            ApiError::bad_request(
                "suggestion-failed",
                "selected action is absent from terminal ranking",
            )
        })?;
    let selected_candidate = ranked.remove(selected_index);
    ranked.insert(0, selected_candidate);

    Ok(SuggestionResponse {
        strategy: strategy.strategy_id().to_owned(),
        candidates: ranked
            .into_iter()
            .take(limit)
            .enumerate()
            .map(|(index, candidate)| {
                suggestion_candidate(
                    state,
                    current_score,
                    index,
                    candidate.action,
                    candidate.mean_leaf_score,
                    candidate.leaf_score_stddev,
                )
            })
            .collect::<Result<Vec<_>, _>>()?,
    })
}

fn suggestion_candidate(
    state: &GameState,
    current_score: ScoreBreakdown,
    index: usize,
    action: TurnAction,
    search_value: f64,
    evaluation_stddev: f64,
) -> Result<SuggestionCandidate, ApiError> {
    let board = state
        .preview_active_board(&action)
        .map_err(ApiError::from_rules)?;
    let resulting_score = score_board(&board, state.config().scoring_cards);
    Ok(SuggestionCandidate {
        rank: index + 1,
        action,
        base_score_delta: i32::from(resulting_score.base_total)
            - i32::from(current_score.base_total),
        resulting_score,
        search_value,
        evaluation_stddev,
    })
}

fn load_saved_game(saved: &SavedGame) -> Result<GameState, ApiError> {
    if saved.schema_version != SAVED_GAME_SCHEMA_VERSION {
        return Err(ApiError::bad_request(
            "unsupported-save",
            format!(
                "saved-game schema {} is not supported",
                saved.schema_version
            ),
        ));
    }
    let expected_seed = GameSeed::from_u64(saved.seed);
    if saved.replay.seed != expected_seed {
        return Err(ApiError::bad_request(
            "seed-mismatch",
            "the displayed seed does not match the replay seed",
        ));
    }
    saved
        .replay
        .play()
        .map_err(|error| ApiError::bad_request("invalid-replay", error.to_string()))
}

fn game_view(state: &GameState, seed: u64) -> GameView {
    let scores = score_game(state);
    GameView {
        config: state.config(),
        seed,
        current_player: state.current_player(),
        completed_turns: state.completed_turns(),
        total_turns: state.total_turns(),
        game_over: state.is_game_over(),
        state_hash: encode_hex(state.canonical_hash().as_bytes()),
        free_overpopulation_replacement: state.market().three_of_a_kind().is_some(),
        boards: state
            .boards()
            .iter()
            .enumerate()
            .map(|(player, board)| board_view(player, board, scores[player]))
            .collect(),
        market: market_view(state.market()),
    }
}

fn board_view(player: usize, board: &Board, score: ScoreBreakdown) -> BoardView {
    BoardView {
        player,
        nature_tokens: board.nature_tokens(),
        tiles: board
            .placed_tiles()
            .map(|(coord, placed)| board_tile_view(coord, placed))
            .collect(),
        frontier: board.frontier(),
        score,
    }
}

fn board_tile_view(coord: HexCoord, placed: &PlacedTile) -> BoardTileView {
    BoardTileView {
        coord,
        tile: tile_view(placed.tile),
        rotation: placed.rotation.get(),
        wildlife: placed.wildlife.map(wildlife_id),
    }
}

fn market_view(market: &Market) -> Vec<MarketPairView> {
    MarketSlot::ALL
        .into_iter()
        .map(|slot| MarketPairView {
            slot: slot.index() as u8,
            tile: market.tiles[slot.index()].map(tile_view),
            wildlife: market.wildlife[slot.index()].map(wildlife_id),
        })
        .collect()
}

fn tile_view(tile: Tile) -> TileView {
    TileView {
        id: tile.id.0,
        terrain_a: terrain_id(tile.terrain_a),
        terrain_b: tile.terrain_b.map(terrain_id),
        wildlife: tile.wildlife.iter().map(wildlife_id).collect(),
        keystone: tile.keystone,
    }
}

const fn terrain_id(terrain: Terrain) -> &'static str {
    match terrain {
        Terrain::Mountain => "mountain",
        Terrain::Forest => "forest",
        Terrain::Prairie => "prairie",
        Terrain::Wetland => "wetland",
        Terrain::River => "river",
    }
}

const fn wildlife_id(wildlife: Wildlife) -> &'static str {
    match wildlife {
        Wildlife::Bear => "bear",
        Wildlife::Elk => "elk",
        Wildlife::Salmon => "salmon",
        Wildlife::Hawk => "hawk",
        Wildlife::Fox => "fox",
    }
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameMode, Rotation, ScoringCards};
    use cascadia_search::LATE_CONSERVATIVE_BASE_POLICY_IMPROVEMENT_STRATEGY_ID;
    use cascadia_sim::{MatchConfig, StrategyKind, play_match};

    use super::*;

    fn new_two_player_game(seed: u64) -> GameDocument {
        create_game(NewGameRequest {
            seed,
            config: GameConfig {
                player_count: 2,
                mode: GameMode::Standard,
                scoring_cards: ScoringCards::AAAAA,
                habitat_bonuses: false,
            },
        })
        .unwrap()
    }

    fn late_two_player_game(seed: u64) -> SavedGame {
        let config = GameConfig::research_aaaaa(2).unwrap();
        let result = play_match(&MatchConfig::symmetric(
            config,
            GameSeed::from_u64(seed),
            StrategyKind::PatternAware,
        ))
        .unwrap();
        let mut replay = result.replay;
        replay.turns.truncate(30);
        replay.final_state_hash = None;
        SavedGame {
            schema_version: SAVED_GAME_SCHEMA_VERSION,
            seed,
            replay,
        }
    }

    #[test]
    fn create_apply_and_undo_round_trip_the_canonical_state() {
        let initial = new_two_player_game(42);
        let options = turn_options(TurnOptionsRequest {
            game: initial.game.clone(),
            prelude: MarketPrelude::default(),
        })
        .unwrap();
        let draft = options.drafts[0];
        let placements = placement_options(PlacementOptionsRequest {
            game: initial.game.clone(),
            prelude: MarketPrelude::default(),
            draft,
        })
        .unwrap();
        let placement = &placements.placements[0];
        let rotation = &placement.rotations[0];
        let action = TurnAction {
            replace_three_of_a_kind: false,
            wildlife_wipes: Vec::new(),
            draft,
            tile: cascadia_game::TilePlacement {
                coord: placement.coord,
                rotation: Rotation::new(rotation.rotation).unwrap(),
            },
            wildlife: rotation.wildlife.first().copied(),
        };

        let advanced = apply_turn(ApplyTurnRequest {
            game: initial.game.clone(),
            action,
        })
        .unwrap();
        assert_eq!(advanced.view.completed_turns, 1);

        let restored = undo_turn(advanced.game).unwrap();
        assert_eq!(restored.view.state_hash, initial.view.state_hash);
        assert_eq!(restored.game.replay, initial.game.replay);
    }

    #[test]
    fn save_seed_mismatch_is_rejected() {
        let mut document = new_two_player_game(7);
        document.game.seed = 8;

        assert_eq!(view_game(document.game).unwrap_err().code, "seed-mismatch");
    }

    #[test]
    fn suggestions_are_ranked_and_every_action_is_legal() {
        let document = new_two_player_game(11);
        let response = suggest(SuggestRequest {
            game: document.game.clone(),
            candidates: 4,
            strength: SuggestionStrength::Instant,
        })
        .unwrap();
        let state = load_saved_game(&document.game).unwrap();

        assert_eq!(response.candidates.len(), 4);
        for pair in response.candidates.windows(2) {
            assert!(pair[0].resulting_score.base_total >= pair[1].resulting_score.base_total);
        }
        for candidate in response.candidates {
            state.transition(&candidate.action).unwrap();
        }
    }

    #[test]
    fn interactive_suggestions_are_ranked_reproducible_and_legal() {
        let document = new_two_player_game(12);
        let request = SuggestRequest {
            game: document.game.clone(),
            candidates: 4,
            strength: SuggestionStrength::Interactive,
        };
        let left = suggest(request.clone()).unwrap();
        let right = suggest(request).unwrap();
        let state = load_saved_game(&document.game).unwrap();

        assert_eq!(left.strategy, PATTERN_AWARE_STRATEGY_ID);
        assert_eq!(
            serde_json::to_value(&left).unwrap(),
            serde_json::to_value(&right).unwrap()
        );
        assert!(
            left.candidates
                .windows(2)
                .all(|pair| pair[0].search_value >= pair[1].search_value)
        );
        for candidate in left.candidates {
            state.transition(&candidate.action).unwrap();
        }
    }

    #[tokio::test]
    async fn capabilities_advertise_explicit_research_search() {
        let response = capabilities().await.0;
        let research = response
            .strengths
            .iter()
            .find(|strength| strength.id == "research")
            .unwrap();

        assert!(research.available);
        assert_eq!(research.label, "Research");
    }

    #[test]
    fn research_terminal_suggestion_is_reproducible_and_legal() {
        let game = late_two_player_game(13);
        let request = SuggestRequest {
            game: game.clone(),
            candidates: 4,
            strength: SuggestionStrength::Research,
        };
        let left = suggest(request.clone()).unwrap();
        let right = suggest(request).unwrap();
        let state = load_saved_game(&game).unwrap();

        assert_eq!(
            left.strategy,
            LateConservativeBasePolicyImprovementConfig::default().strategy_id()
        );
        assert!(
            left.strategy
                .starts_with(LATE_CONSERVATIVE_BASE_POLICY_IMPROVEMENT_STRATEGY_ID)
        );
        assert_eq!(
            serde_json::to_value(&left).unwrap(),
            serde_json::to_value(&right).unwrap()
        );
        assert_eq!(left.candidates.len(), 4);
        assert!(left.candidates[0].evaluation_stddev >= 0.0);
        for candidate in left.candidates {
            state.transition(&candidate.action).unwrap();
        }
    }
}
