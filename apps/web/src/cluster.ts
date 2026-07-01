import type {
  ClusterHistoryPoint,
  ClusterHistoryRange,
  NodeHealth,
  ResearchExperiment,
  ResearchExperimentOutcome,
  ResearchExperimentStatus,
  R2MapLossSample,
  R2MapStatusCondition,
} from "./types";

export type ClusterHistoryMetric = "cpu_percent" | "memory_percent";

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 GB";
  }
  const gibibytes = bytes / 1024 ** 3;
  return `${gibibytes >= 100 ? gibibytes.toFixed(0) : gibibytes.toFixed(1)} GB`;
}

export function formatSignedBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes === 0) {
    return "0 GB";
  }
  return `${bytes > 0 ? "+" : "−"}${formatBytes(Math.abs(bytes))}`;
}

export function formatUptime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return "Unavailable";
  }
  const days = Math.floor(seconds / 86_400);
  const hours = Math.floor((seconds % 86_400) / 3_600);
  const minutes = Math.floor((seconds % 3_600) / 60);
  if (days > 0) {
    return `${days}d ${hours}h`;
  }
  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  return `${minutes}m`;
}

export function formatPercent(value: number): string {
  return `${Math.round(Number.isFinite(value) ? value : 0)}%`;
}

export function formatCoreUsage(used: number, total: number): string {
  const boundedUsed = Number.isFinite(used) ? Math.max(0, used) : 0;
  const boundedTotal = Number.isFinite(total) ? Math.max(0, total) : 0;
  return `${boundedUsed.toFixed(1)} / ${boundedTotal} cores`;
}

export function r2MapConditionLabel(condition: R2MapStatusCondition): string {
  switch (condition) {
    case "fresh":
      return "Live";
    case "stale":
      return "Stale";
    case "invalid":
      return "Invalid";
    case "unconfigured":
      return "Unconfigured";
  }
}

export function lossPolyline(
  samples: R2MapLossSample[],
  field: "train_total" | "validation_total",
  width: number,
  height: number,
): string {
  const points = samples
    .map((sample) => ({ step: sample.step, value: sample[field] }))
    .filter((point): point is { step: number; value: number } => point.value !== null);
  if (points.length === 0) {
    return "";
  }
  const minStep = samples[0]?.step ?? points[0].step;
  const maxStep = samples.at(-1)?.step ?? points.at(-1)?.step ?? minStep;
  const values = samples.flatMap(({ train_total, validation_total }) =>
    validation_total === null ? [train_total] : [train_total, validation_total],
  );
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  return points
    .map(({ step, value }, index) => {
      const x = ((step - minStep) / Math.max(1, maxStep - minStep)) * width;
      const y = height - ((value - minValue) / Math.max(1e-12, maxValue - minValue)) * height;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return "—";
  }
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  }
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m`;
  }
  const hours = Math.floor(minutes / 60);
  const remainder = minutes % 60;
  return remainder === 0 ? `${hours}h` : `${hours}h ${remainder}m`;
}

export function healthLabel(health: NodeHealth): string {
  switch (health) {
    case "healthy":
      return "Ready";
    case "busy":
      return "Working";
    case "warning":
      return "Attention";
    case "offline":
      return "Offline";
  }
}

export function resourceTone(value: number): "good" | "warm" | "hot" {
  if (value >= 90) {
    return "hot";
  }
  if (value >= 70) {
    return "warm";
  }
  return "good";
}

export function historyRangeLabel(range: ClusterHistoryRange): string {
  return range === "1d" ? "24 hours" : "7 days";
}

const EXPERIMENT_STATUS_ORDER: Record<ResearchExperimentStatus, number> = {
  running: 0,
  planned: 1,
  completed: 2,
  cancelled: 3,
};

export function experimentOutcomeLabel(
  status: ResearchExperimentStatus,
  outcome: ResearchExperimentOutcome,
): string {
  if (status === "running") {
    return "Running";
  }
  if (status === "planned") {
    return "Planned";
  }
  if (status === "cancelled") {
    return "Cancelled";
  }
  switch (outcome) {
    case "passed":
      return "Passed";
    case "failed":
      return "Failed";
    case "inconclusive":
      return "Inconclusive";
    case "invalid":
      return "Invalid";
    case "pending":
      return "Pending";
  }
}

export function sortResearchExperiments(
  experiments: ResearchExperiment[],
): ResearchExperiment[] {
  return [...experiments].sort(
    (left, right) =>
      EXPERIMENT_STATUS_ORDER[left.status] -
        EXPERIMENT_STATUS_ORDER[right.status] ||
      (right.completed_unix_ms ?? right.updated_unix_ms) -
        (left.completed_unix_ms ?? left.updated_unix_ms) ||
      left.id.localeCompare(right.id),
  );
}

export function formatHistoryTime(timestamp: number, range: ClusterHistoryRange): string {
  return new Date(timestamp).toLocaleString([], {
    weekday: "short",
    month: range === "7d" ? "short" : undefined,
    day: range === "7d" ? "numeric" : undefined,
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function historyLineSegments(
  points: ClusterHistoryPoint[],
  metric: ClusterHistoryMetric,
  startTimestamp: number,
  endTimestamp: number,
  width: number,
  height: number,
  expectedGapMs: number,
): string[] {
  const duration = Math.max(1, endTimestamp - startTimestamp);
  const segments: string[] = [];
  let commands: string[] = [];
  let previousTimestamp: number | null = null;

  const flush = () => {
    if (commands.length > 0) {
      segments.push(commands.join(" "));
      commands = [];
    }
  };

  for (const point of points) {
    const value = point[metric];
    if (
      value === null ||
      point.reachable_percent <= 0 ||
      (previousTimestamp !== null &&
        point.timestamp_unix_ms - previousTimestamp > expectedGapMs * 2.5)
    ) {
      flush();
      previousTimestamp = value === null ? null : point.timestamp_unix_ms;
      if (value === null || point.reachable_percent <= 0) {
        continue;
      }
    }
    const x = ((point.timestamp_unix_ms - startTimestamp) / duration) * width;
    const y = height - (Math.max(0, Math.min(100, value)) / 100) * height;
    commands.push(`${commands.length === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`);
    previousTimestamp = point.timestamp_unix_ms;
  }
  flush();
  return segments;
}
