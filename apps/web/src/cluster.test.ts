import { describe, expect, it } from "vitest";

import {
  formatBytes,
  formatHistoryTime,
  formatPercent,
  formatUptime,
  healthLabel,
  historyLineSegments,
  historyRangeLabel,
  resourceTone,
} from "./cluster";

describe("cluster display helpers", () => {
  it("formats capacity and uptime for compact operational labels", () => {
    expect(formatBytes(16 * 1024 ** 3)).toBe("16.0 GB");
    expect(formatBytes(512 * 1024 ** 3)).toBe("512 GB");
    expect(formatUptime(90_061)).toBe("1d 1h");
    expect(formatUptime(3_720)).toBe("1h 2m");
  });

  it("maps health and pressure thresholds consistently", () => {
    expect(healthLabel("busy")).toBe("Working");
    expect(formatPercent(84.6)).toBe("85%");
    expect(resourceTone(69.9)).toBe("good");
    expect(resourceTone(70)).toBe("warm");
    expect(resourceTone(90)).toBe("hot");
  });

  it("formats history ranges and timestamps", () => {
    expect(historyRangeLabel("1d")).toBe("24 hours");
    expect(historyRangeLabel("7d")).toBe("7 days");
    expect(formatHistoryTime(0, "1d")).toContain(":");
  });

  it("builds bounded chart paths and breaks across outages", () => {
    const points = [
      {
        timestamp_unix_ms: 0,
        reachable_percent: 100,
        cpu_percent: 25,
        memory_percent: 50,
      },
      {
        timestamp_unix_ms: 1_000,
        reachable_percent: 100,
        cpu_percent: 75,
        memory_percent: 60,
      },
      {
        timestamp_unix_ms: 10_000,
        reachable_percent: 100,
        cpu_percent: 50,
        memory_percent: 70,
      },
    ];
    expect(historyLineSegments(points, "cpu_percent", 0, 10_000, 100, 100, 1_000)).toEqual([
      "M 0.00 75.00 L 10.00 25.00",
      "M 100.00 50.00",
    ]);
  });
});
