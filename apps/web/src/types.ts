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
  id: "instant" | "interactive" | "research";
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
    cpu_used_cores: number;
    cpu_used_percent: number;
    memory_used_bytes: number;
    memory_total_bytes: number;
  };
  nodes: ClusterNode[];
  scheduler: SchedulerStatus;
}

export interface SchedulerStatus {
  configured: boolean;
  reachable: boolean;
  version: string;
  web_ui_url: string;
  error: string | null;
  summary: {
    queued: number;
    running: number;
    successful: number;
    retrying: number;
    failed: number;
    cancelled: number;
  };
  nodes: SchedulerNode[];
  jobs: SchedulerJob[];
  services: {
    registry_healthy: boolean;
    object_store_healthy: boolean;
  };
}

export interface SchedulerNode {
  node_id: string;
  label: string;
  connected: boolean;
  running_executions: number;
  enqueued_executions: number;
  cpu_allocated: number;
  cpu_capacity: number;
  memory_allocated_bytes: number;
  memory_capacity_bytes: number;
}

export interface SchedulerJob {
  id: string;
  name: string;
  request_id: string | null;
  item_id: string | null;
  experiment_id: string | null;
  state: string;
  attempts: number;
  failure_reason: string | null;
  created_unix_ns: number;
  modified_unix_ns: number;
  detail_url: string;
}

export type QueueTaskStatus =
  | "blocked"
  | "ready"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export interface QueueHost {
  root: string;
  intent: "available" | "working" | "reserved" | "intentionally-idle";
  reason: string | null;
  updated_unix_ms: number;
}

export interface QueueTask {
  id: string;
  title: string;
  experiment_id: string;
  decision: string;
  workload_class:
    | "independent-experiment"
    | "divisible-evidence"
    | "shared-prerequisite"
    | "replica";
  priority: number;
  decision_value: number;
  expected_runtime_seconds: number;
  critical_path: boolean;
  decision_terminal: boolean;
  compatible_hosts: string[];
  dependencies: string[];
  artifact_path: string;
  stop_rule: string;
  resources: {
    cpu_cores: number;
    memory_gib: number;
    uses_mlx: boolean;
  };
  status: QueueTaskStatus;
  claim: {
    host: string;
    claimed_unix_ms: number;
    heartbeat_unix_ms: number;
    lease_expires_unix_ms: number;
  } | null;
}

export interface ClusterQueueResponse {
  schema_version: number;
  configured: boolean;
  source_path: string;
  campaign_id: string | null;
  updated_unix_ms: number | null;
  summary: {
    total: number;
    blocked: number;
    ready: number;
    running: number;
    completed: number;
    failed: number;
    cancelled: number;
    critical_path_ready: number;
    duplicate_running: number;
    decisions_completed: number;
    ready_decision_value: number;
  };
  hosts: Record<string, QueueHost>;
  tasks: QueueTask[];
  error: string | null;
}

export type ResearchExperimentStatus =
  | "planned"
  | "running"
  | "completed"
  | "cancelled";

export type ResearchExperimentOutcome =
  | "pending"
  | "passed"
  | "failed"
  | "inconclusive"
  | "invalid";

export type ResearchMetricTone = "good" | "bad" | "warn" | "neutral";

export interface ResearchExperimentMetric {
  label: string;
  value: string;
  tone: ResearchMetricTone;
}

export interface ResearchExperimentCriterion {
  label: string;
  passed: boolean | null;
  observed: string | null;
}

export interface ResearchExperimentArtifact {
  label: string;
  path: string;
}

export interface ResearchExperiment {
  id: string;
  title: string;
  hypothesis: string;
  summary: string;
  status: ResearchExperimentStatus;
  outcome: ResearchExperimentOutcome;
  verdict: string | null;
  plan_section: string | null;
  started_unix_ms: number | null;
  completed_unix_ms: number | null;
  updated_unix_ms: number;
  hosts: string[];
  tags: string[];
  task_ids: string[];
  metrics: ResearchExperimentMetric[];
  criteria: ResearchExperimentCriterion[];
  notes: string[];
  artifacts: ResearchExperimentArtifact[];
}

export interface ClusterExperimentsResponse {
  schema_version: number;
  configured: boolean;
  source_path: string;
  updated_unix_ms: number | null;
  experiments: ResearchExperiment[];
  error: string | null;
}

export type R2MapStatusCondition = "unconfigured" | "fresh" | "stale" | "invalid";
export type R2MapHostIntent =
  | "control"
  | "generate"
  | "validate"
  | "train"
  | "benchmark"
  | "candidate-gate"
  | "idle";
export type R2MapClassification =
  | "pending"
  | "promote"
  | "reject"
  | "inconclusive"
  | "invalid";

export interface R2MapModelIdentity {
  id: string;
  blake3: string | null;
}

export interface R2MapDistributionSummary {
  mean: number;
  p10: number;
  p50: number;
  p90: number;
}

export interface R2MapHostStatus {
  intent: R2MapHostIntent;
  detail: string | null;
  generation_games_completed: number;
  generation_games_target: number | null;
  generation_seed_prefix: string | null;
  benchmark_pairs_completed: number;
  benchmark_pairs_total: number | null;
  eta_seconds: number | null;
  throughput_games_per_second: number | null;
  rss_bytes: number | null;
  swap_delta_bytes: number | null;
}

export interface R2MapLossSample {
  step: number;
  train_total: number;
  validation_total: number | null;
}

export interface R2MapFocalStatus {
  base_total: R2MapDistributionSummary;
  animals: {
    aggregate: R2MapDistributionSummary;
    bear: R2MapDistributionSummary;
    elk: R2MapDistributionSummary;
    salmon: R2MapDistributionSummary;
    hawk: R2MapDistributionSummary;
    fox: R2MapDistributionSummary;
  };
  habitat: {
    aggregate: R2MapDistributionSummary;
    mountain: R2MapDistributionSummary;
    forest: R2MapDistributionSummary;
    prairie: R2MapDistributionSummary;
    wetland: R2MapDistributionSummary;
    river: R2MapDistributionSummary;
  };
  pinecones: {
    earned: R2MapDistributionSummary;
    independent_draft_spend: R2MapDistributionSummary;
    paid_wipe_spend: R2MapDistributionSummary;
    total_spend: R2MapDistributionSummary;
    remaining: R2MapDistributionSummary;
    free_replacements: R2MapDistributionSummary;
  };
}

export interface R2MapCampaignStatus {
  schema_version: number;
  schema_id: string;
  campaign_id: string;
  updated_unix_ms: number;
  stale_after_seconds: number;
  phase: string;
  legal_next_transitions: string[];
  round_index: number | null;
  models: {
    incumbent: R2MapModelIdentity | null;
    candidate: R2MapModelIdentity | null;
    opponent_pool: R2MapModelIdentity[];
  };
  hosts: Record<string, R2MapHostStatus>;
  training: {
    active: boolean;
    latest_verified_checkpoint: R2MapModelIdentity | null;
    current_step: number | null;
    total_steps: number | null;
    eta_seconds: number | null;
    examples_per_second: number | null;
    loss_samples: R2MapLossSample[];
  };
  benchmark: {
    active: boolean;
    stage: string | null;
    pairs_completed: number;
    pairs_total: number | null;
    eta_seconds: number | null;
    throughput_games_per_second: number | null;
    peak_rss_bytes: number | null;
    swap_delta_bytes: number | null;
    focal: R2MapFocalStatus | null;
    paired_delta: { mean: number; confidence_95: [number, number] } | null;
    classification: R2MapClassification;
    scheduler_work_items?: {
      completed: number;
      total: number;
      states: Record<string, number>;
      retry_attempts: number;
      utilization: {
        sample_count: number;
        observed_seconds: number;
        cpu_capacity_min: number;
        cpu_capacity_max: number;
        cpu_allocated_mean: number;
        cpu_allocated_peak: number;
        cpu_utilization_mean: number;
        cpu_utilization_peak: number;
      } | null;
    } | null;
  };
}

export interface R2MapStatusResponse {
  schema_version: number;
  configured: boolean;
  condition: R2MapStatusCondition;
  source_path: string;
  observed_unix_ms: number;
  updated_unix_ms: number | null;
  age_seconds: number | null;
  stale_after_seconds: number | null;
  status: R2MapCampaignStatus | null;
  error: string | null;
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
