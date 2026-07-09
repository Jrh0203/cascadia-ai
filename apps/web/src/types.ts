export type Terrain = "mountain" | "forest" | "prairie" | "wetland" | "river";
export type Wildlife = "bear" | "elk" | "salmon" | "hawk" | "fox";
export type ScoringVariant = "A" | "B" | "C" | "D";

export interface HexCoord {
  q: number;
  r: number;
}

export interface ScoringCards {
  bear: ScoringVariant;
  elk: ScoringVariant;
  salmon: ScoringVariant;
  hawk: ScoringVariant;
  fox: ScoringVariant;
}

export interface GameConfig {
  player_count: number;
  mode: "Standard" | "Solo";
  scoring_cards: ScoringCards;
  habitat_bonuses: boolean;
}

export interface TileView {
  id: number;
  terrain_a: Terrain;
  terrain_b: Terrain | null;
  wildlife: Wildlife[];
  keystone: boolean;
}

export interface BoardTileView {
  coord: HexCoord;
  tile: TileView;
  rotation: number;
  wildlife: Wildlife | null;
}

export interface ScoreBreakdown {
  habitat: number[];
  wildlife: number[];
  nature_tokens: number;
  habitat_bonus: number[];
  base_total: number;
  total: number;
}

export interface BoardView {
  player: number;
  nature_tokens: number;
  tiles: BoardTileView[];
  frontier: HexCoord[];
  score: ScoreBreakdown;
}

export interface MarketPairView {
  slot: number;
  tile: TileView | null;
  wildlife: Wildlife | null;
}

export interface GameView {
  ruleset_id: string;
  config: GameConfig;
  seed: number;
  current_player: number;
  completed_turns: number;
  total_turns: number;
  game_over: boolean;
  state_hash: string;
  free_overpopulation_replacement: boolean;
  boards: BoardView[];
  market: MarketPairView[];
}

export interface Replay {
  schema_version: number;
  config: GameConfig;
  seed: number[];
  turns: TurnAction[];
  final_state_hash: number[] | null;
}

export interface SavedGame {
  schema_version: number;
  seed: number;
  replay: Replay;
}

export interface GameDocument {
  game: SavedGame;
  view: GameView;
}

export type DraftChoice =
  | { Paired: { slot: number } }
  | { Independent: { tile_slot: number; wildlife_slot: number } };

export interface WildlifeWipe {
  slots: number[];
}

export interface MarketPrelude {
  replace_three_of_a_kind: boolean;
  wildlife_wipes: WildlifeWipe[];
}

export interface TurnAction {
  replace_three_of_a_kind: boolean;
  wildlife_wipes: WildlifeWipe[];
  draft: DraftChoice;
  tile: {
    coord: HexCoord;
    rotation: number;
  };
  wildlife: HexCoord | null;
}

export interface TurnOptions {
  market: MarketPairView[];
  drafts: DraftChoice[];
  nature_tokens_remaining: number;
  can_add_paid_wipe: boolean;
}

export interface PlacementOption {
  coord: HexCoord;
  rotations: {
    rotation: number;
    wildlife: HexCoord[];
    may_skip_wildlife: boolean;
  }[];
}

export interface PlacementOptions {
  placements: PlacementOption[];
}

export interface SuggestionCandidate {
  rank: number;
  action: TurnAction;
  resulting_score: ScoreBreakdown;
  base_score_delta: number;
  search_value: number;
  evaluation_stddev: number;
}

export interface SuggestionResponse {
  strategy: string;
  candidates: SuggestionCandidate[];
}

export interface StrengthCapability {
  id: "instant" | "interactive" | "research" | "champion" | "champion-deep";
  label: string;
  available: boolean;
  latency: string;
}

export interface Capabilities {
  api_schema_version: number;
  strengths: StrengthCapability[];
  max_players: number;
  scoring_variants: ScoringVariant[];
}

export type NodeHealth = "healthy" | "busy" | "warning" | "offline";

export interface ClusterJob {
  pid: number;
  elapsed: string;
  cpu_percent: number;
  memory_percent: number;
  workload: string;
  command: string;
}

export interface ClusterNode {
  id: string;
  label: string;
  role: string;
  address: string;
  health: NodeHealth;
  reachable: boolean;
  sample_latency_ms: number;
  error: string | null;
  hostname: string | null;
  os_version: string | null;
  cores: number;
  cpu_percent: number;
  load_average: [number, number, number];
  memory_used_bytes: number;
  memory_total_bytes: number;
  memory_used_percent: number;
  disk_used_bytes: number;
  disk_total_bytes: number;
  disk_available_bytes: number;
  disk_used_percent: number;
  uptime_seconds: number;
  power: {
    source: string | null;
    system_sleep_minutes: number | null;
    auto_restart: boolean | null;
  };
  readiness: {
    repository_present: boolean;
    release_binary_present: boolean;
    mlx_runtime_present: boolean;
    branch: string | null;
    revision: string | null;
    uncommitted_changes: number | null;
  };
  jobs: ClusterJob[];
}

export interface ClusterResponse {
  schema_version: number;
  collected_at_unix_ms: number;
  collection_duration_ms: number;
  summary: {
    total_nodes: number;
    online_nodes: number;
    busy_nodes: number;
    degraded_nodes: number;
    total_cores: number;
    active_jobs: number;
    average_cpu_percent: number;
    memory_used_bytes: number;
    memory_total_bytes: number;
  };
  nodes: ClusterNode[];
}

export type ClusterHistoryRange = "1d" | "7d";

export interface ClusterHistoryPoint {
  timestamp_unix_ms: number;
  reachable_percent: number;
  cpu_percent: number | null;
  memory_percent: number | null;
}

export interface ClusterHistorySummary {
  observed_samples: number;
  reachable_percent: number;
  average_cpu_percent: number | null;
  peak_cpu_percent: number | null;
  average_memory_percent: number | null;
  peak_memory_percent: number | null;
}

export interface ClusterHistorySeries {
  node_id: string;
  node_label: string;
  summary: ClusterHistorySummary;
  points: ClusterHistoryPoint[];
}

export interface ClusterHistoryResponse {
  schema_version: number;
  range: ClusterHistoryRange;
  generated_at_unix_ms: number;
  start_unix_ms: number;
  end_unix_ms: number;
  oldest_sample_unix_ms: number | null;
  newest_sample_unix_ms: number | null;
  source_sample_interval_seconds: number;
  bucket_seconds: number;
  raw_sample_count: number;
  series: ClusterHistorySeries[];
}

export interface SeatConfig {
  kind: "human" | "ai";
  strength: StrengthCapability["id"];
}

export interface ApiErrorBody {
  code: string;
  message: string;
}
