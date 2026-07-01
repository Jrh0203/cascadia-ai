import type {
  ApiErrorBody,
  Capabilities,
  ClusterExperimentsResponse,
  ClusterHistoryRange,
  ClusterHistoryResponse,
  ClusterQueueResponse,
  ClusterResponse,
  DraftChoice,
  GameConfig,
  GameDocument,
  MarketPrelude,
  PlacementOptions,
  R2MapStatusResponse,
  SavedGame,
  StrengthCapability,
  SuggestionResponse,
  TurnAction,
  TurnOptions,
} from "./types";

export class ApiError extends Error {
  readonly code: string;
  readonly status: number;

  constructor(status: number, body: ApiErrorBody) {
    super(body.message);
    this.name = "ApiError";
    this.code = body.code;
    this.status = status;
  }
}

async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    const error = (await response.json().catch(() => ({
      code: "request-failed",
      message: `Request failed with status ${response.status}`,
    }))) as ApiErrorBody;
    throw new ApiError(response.status, error);
  }
  return (await response.json()) as T;
}

export const api = {
  capabilities: () => request<Capabilities>("/api/v1/capabilities"),
  cluster: () => request<ClusterResponse>("/api/v1/cluster"),
  clusterQueue: () => request<ClusterQueueResponse>("/api/v1/cluster/queue"),
  clusterExperiments: () =>
    request<ClusterExperimentsResponse>("/api/v1/cluster/experiments"),
  r2MapStatus: () => request<R2MapStatusResponse>("/api/v1/cluster/r2-map"),
  clusterHistory: (range: ClusterHistoryRange) =>
    request<ClusterHistoryResponse>(
      `/api/v1/cluster/history?range=${encodeURIComponent(range)}`,
    ),
  newGame: (seed: number, config: GameConfig) =>
    request<GameDocument>("/api/v1/games/new", { seed, config }),
  viewGame: (game: SavedGame) =>
    request<GameDocument>("/api/v1/games/view", { game }),
  turnOptions: (game: SavedGame, prelude: MarketPrelude) =>
    request<TurnOptions>("/api/v1/games/turn-options", { game, prelude }),
  placementOptions: (
    game: SavedGame,
    prelude: MarketPrelude,
    draft: DraftChoice,
  ) =>
    request<PlacementOptions>("/api/v1/games/placement-options", {
      game,
      prelude,
      draft,
    }),
  apply: (game: SavedGame, action: TurnAction) =>
    request<GameDocument>("/api/v1/games/apply", { game, action }),
  undo: (game: SavedGame) =>
    request<GameDocument>("/api/v1/games/undo", { game }),
  suggest: (
    game: SavedGame,
    candidates = 8,
    strength: StrengthCapability["id"] = "instant",
  ) =>
    request<SuggestionResponse>("/api/v1/games/suggest", {
      game,
      candidates,
      strength,
    }),
};
