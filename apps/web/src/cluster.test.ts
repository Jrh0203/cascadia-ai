import { describe, expect, it } from "vitest";

import {
  formatBytes,
  formatCoreUsage,
  formatDuration,
  formatHistoryTime,
  formatPercent,
  formatSignedBytes,
  formatUptime,
  experimentOutcomeLabel,
  healthLabel,
  historyLineSegments,
  historyRangeLabel,
  lossPolyline,
  r2MapConditionLabel,
  resourceTone,
  sortResearchExperiments,
} from "./cluster";
import type { ResearchExperiment } from "./types";

describe("cluster display helpers", () => {
  it("formats capacity and uptime for compact operational labels", () => {
    expect(formatBytes(16 * 1024 ** 3)).toBe("16.0 GB");
    expect(formatBytes(512 * 1024 ** 3)).toBe("512 GB");
    expect(formatSignedBytes(2 * 1024 ** 3)).toBe("+2.0 GB");
    expect(formatSignedBytes(-2 * 1024 ** 3)).toBe("−2.0 GB");
    expect(formatCoreUsage(3.2, 10)).toBe("3.2 / 10 cores");
    expect(formatDuration(32)).toBe("32s");
    expect(formatDuration(3_900)).toBe("1h 5m");
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

  it("labels R2-MAP mirror health and renders compact loss paths", () => {
    expect(r2MapConditionLabel("fresh")).toBe("Live");
    expect(r2MapConditionLabel("stale")).toBe("Stale");
    const samples = [
      { step: 10, train_total: 3, validation_total: 3.2 },
      { step: 20, train_total: 2, validation_total: null },
      { step: 30, train_total: 1, validation_total: 1.2 },
    ];
    expect(lossPolyline(samples, "train_total", 100, 50)).toBe(
      "M 0.00 4.55 L 50.00 27.27 L 100.00 50.00",
    );
    expect(lossPolyline(samples, "validation_total", 100, 50)).toBe(
      "M 0.00 0.00 L 100.00 45.45",
    );
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

  it("orders live experiments ahead of completed results", () => {
    const experiment = (
      id: string,
      status: ResearchExperiment["status"],
      updated: number,
    ): ResearchExperiment => ({
      id,
      title: id,
      hypothesis: "Hypothesis",
      summary: "Summary",
      status,
      outcome: status === "completed" ? "passed" : "pending",
      verdict: null,
      plan_section: null,
      started_unix_ms: status === "running" ? 1 : null,
      completed_unix_ms: status === "completed" ? updated : null,
      updated_unix_ms: updated,
      hosts: [],
      tags: [],
      task_ids: [],
      metrics: [],
      criteria: [],
      notes: [],
      artifacts: [],
    });
    expect(
      sortResearchExperiments([
        experiment("old-result", "completed", 10),
        experiment("live", "running", 5),
        experiment("new-result", "completed", 20),
      ]).map(({ id }) => id),
    ).toEqual(["live", "new-result", "old-result"]);
    expect(experimentOutcomeLabel("completed", "failed")).toBe("Failed");
    expect(experimentOutcomeLabel("running", "pending")).toBe("Running");
  });
});
