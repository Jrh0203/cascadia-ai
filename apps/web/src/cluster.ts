import type {
  ClusterHistoryPoint,
  ClusterHistoryRange,
  NodeHealth,
} from "./types";

export type ClusterHistoryMetric = "cpu_percent" | "memory_percent";

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 GB";
  }
  const gibibytes = bytes / 1024 ** 3;
  return `${gibibytes >= 100 ? gibibytes.toFixed(0) : gibibytes.toFixed(1)} GB`;
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
