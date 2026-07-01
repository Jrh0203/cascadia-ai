import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import ClusterFleetOverview from "./ClusterFleetOverview";
import R2MapCampaignPanel from "./R2MapCampaignPanel";
import type {
  ClusterNode,
  R2MapCampaignStatus,
  R2MapDistributionSummary,
  R2MapStatusResponse,
} from "./types";

function node(
  id: string,
  cpuPercent: number,
  memoryPercent: number,
  diskPercent: number,
): ClusterNode {
  return {
    id,
    label: id,
    role: "worker",
    address: "127.0.0.1",
    health: "healthy",
    reachable: true,
    sample_latency_ms: 1,
    error: null,
    hostname: id,
    os_version: "test",
    cores: 10,
    cpu_percent: cpuPercent,
    load_average: [1, 1, 1],
    memory_used_bytes: memoryPercent,
    memory_total_bytes: 100,
    memory_used_percent: memoryPercent,
    disk_used_bytes: diskPercent,
    disk_total_bytes: 100,
    disk_available_bytes: 100 - diskPercent,
    disk_used_percent: diskPercent,
    uptime_seconds: 1,
    power: { source: "AC", system_sleep_minutes: 0, auto_restart: true },
    readiness: {
      repository_present: true,
      release_binary_present: true,
      mlx_runtime_present: true,
      branch: "test",
      revision: "test",
      uncommitted_changes: 0,
    },
    jobs: [],
  };
}

const distribution = (mean: number): R2MapDistributionSummary => ({
  mean,
  p10: mean - 1,
  p50: mean,
  p90: mean + 1,
});

afterEach(cleanup);

function response(
  condition: R2MapStatusResponse["condition"],
): R2MapStatusResponse {
  return {
    schema_version: 1,
    configured: condition !== "unconfigured",
    condition,
    source_path: "/campaign/control/dashboard-status.json",
    observed_unix_ms: 101_000,
    updated_unix_ms: condition === "unconfigured" ? null : 100_000,
    age_seconds: condition === "unconfigured" ? null : 1,
    stale_after_seconds: condition === "unconfigured" ? null : 30,
    error: condition === "invalid" ? "schema mismatch" : null,
    status:
      condition === "unconfigured" || condition === "invalid"
        ? null
        : {
            schema_version: 1,
            schema_id: "cascadia.r2-map.dashboard-status.v1",
            campaign_id: "r2-map-expert-iteration-v1",
            updated_unix_ms: 100_000,
            stale_after_seconds: 30,
            phase: "training-on-john1-benchmarking-on-john2-john3",
            legal_next_transitions: ["candidate-verified-benchmark-complete"],
            round_index: 1,
            models: {
              incumbent: { id: "c0", blake3: null },
              candidate: { id: "t1", blake3: null },
              opponent_pool: [{ id: "greedy-v1", blake3: null }],
            },
            hosts: Object.fromEntries(
              ["john1", "john2", "john3"].map((host) => [
                host,
                {
                  intent: host === "john1" ? "train" : "benchmark",
                  detail:
                    host === "john1" ? "MLX training" : "Previous checkpoint",
                  generation_games_completed: 100,
                  generation_games_target: 100,
                  generation_seed_prefix: `${host}-seed`,
                  benchmark_pairs_completed: host === "john1" ? 0 : 20,
                  benchmark_pairs_total: host === "john1" ? null : 50,
                  eta_seconds: 30,
                  throughput_games_per_second: host === "john1" ? null : 5,
                  rss_bytes: 1_024,
                  swap_delta_bytes: 0,
                },
              ]),
            ) as R2MapCampaignStatus["hosts"],
            training: {
              active: true,
              latest_verified_checkpoint: {
                id: "checkpoint-100",
                blake3: null,
              },
              current_step: 120,
              total_steps: 500,
              eta_seconds: 60,
              examples_per_second: 400,
              loss_samples: [
                { step: 100, train_total: 2, validation_total: 2.1 },
                { step: 120, train_total: 1.8, validation_total: 1.9 },
              ],
            },
            benchmark: {
              active: true,
              stage: "fixed-250-development",
              pairs_completed: 40,
              pairs_total: 250,
              eta_seconds: 60,
              throughput_games_per_second: 8,
              peak_rss_bytes: 2_048,
              swap_delta_bytes: 0,
              classification: "pending",
              scheduler_work_items: {
                completed: 40,
                total: 250,
                states: { running: 14, queued: 196, succeeded: 40 },
                retry_attempts: 1,
                utilization: {
                  sample_count: 12,
                  observed_seconds: 180,
                  cpu_capacity_min: 29,
                  cpu_capacity_max: 29,
                  cpu_allocated_mean: 27,
                  cpu_allocated_peak: 28,
                  cpu_utilization_mean: 27 / 29,
                  cpu_utilization_peak: 28 / 29,
                },
              },
              paired_delta: { mean: 1.2, confidence_95: [0.2, 2.2] },
              focal: {
                base_total: distribution(96),
                animals: {
                  aggregate: distribution(58),
                  bear: distribution(10),
                  elk: distribution(11),
                  salmon: distribution(12),
                  hawk: distribution(12),
                  fox: distribution(13),
                },
                habitat: {
                  aggregate: distribution(30),
                  mountain: distribution(6),
                  forest: distribution(6),
                  prairie: distribution(6),
                  wetland: distribution(6),
                  river: distribution(6),
                },
                pinecones: {
                  earned: distribution(7),
                  independent_draft_spend: distribution(2),
                  paid_wipe_spend: distribution(1),
                  total_spend: distribution(3),
                  remaining: distribution(4),
                  free_replacements: distribution(1),
                },
              },
            },
          },
  };
}

describe("R2MapCampaignPanel", () => {
  it("makes the unconfigured and invalid states explicit", () => {
    const { rerender, unmount } = render(
      <R2MapCampaignPanel
        requestError={null}
        response={response("unconfigured")}
      />,
    );
    expect(screen.getByText("Unconfigured")).toBeTruthy();
    expect(screen.getByText(/not scanning campaign artifacts/i)).toBeTruthy();

    rerender(
      <R2MapCampaignPanel requestError={null} response={response("invalid")} />,
    );
    expect(screen.getByText("Invalid")).toBeTruthy();
    expect(screen.getByText(/schema mismatch/i)).toBeTruthy();
    unmount();
  });

  it("renders one compact live command deck with host intent and training progress", () => {
    render(
      <R2MapCampaignPanel requestError={null} response={response("fresh")} />,
    );
    expect(
      screen.getByRole("heading", { name: "MLX training on John1" }),
    ).toBeTruthy();
    expect(
      screen
        .getByRole("progressbar", { name: "MLX training progress" })
        .getAttribute("aria-valuenow"),
    ).toBe("24");
    expect(screen.getAllByText("MLX training").length).toBeGreaterThan(0);
    expect(screen.getAllByText("checkpoint-100").length).toBeGreaterThan(0);
    expect(
      screen.getByText(/Let MLX finish; do not benchmark or generate/i),
    ).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "Pinecones" })).toBeNull();
  });

  it("keeps focal tails and score anatomy in the benchmark tab", () => {
    render(
      <R2MapCampaignPanel
        activeView="benchmark"
        requestError={null}
        response={response("fresh")}
      />,
    );
    expect(screen.getByText("96.000")).toBeTruthy();
    expect(screen.getByText("+1.200")).toBeTruthy();
    expect(screen.getByText("93.1% / 96.6%")).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Pinecones" })).toBeTruthy();
  });

  it("shows scheduler progress during a nonbenchmark distributed phase", () => {
    const live = response("fresh");
    if (!live.status) throw new Error("fixture must include status");
    live.status.schema_id = "cascadia.v3.dashboard-status.v1";
    live.status.phase = "bootstrap_labeling";
    live.status.training.active = false;
    live.status.benchmark.active = false;
    live.status.benchmark.pairs_completed = 0;
    live.status.benchmark.pairs_total = null;
    live.status.benchmark.scheduler_work_items = {
      completed: 12,
      total: 120,
      states: { running: 29, queued: 79, succeeded: 12 },
      retry_attempts: 0,
      utilization: null,
    };

    render(<R2MapCampaignPanel requestError={null} response={live} />);
    expect(
      screen.getByRole("heading", { name: "Bootstrap labeling" }),
    ).toBeTruthy();
    expect(
      screen
        .getByRole("progressbar", { name: "Distributed work progress" })
        .getAttribute("aria-valuenow"),
    ).toBe("10");
    expect(screen.getByText("12 / 120")).toBeTruthy();
    expect(screen.getByText(/running 29/i)).toBeTruthy();
  });

  it("labels a historical greedy promotion as a completed baseline", () => {
    const current = response("fresh");
    if (!current.status) throw new Error("fresh fixture must carry status");
    current.status.training.active = false;
    current.status.benchmark.active = false;
    current.status.benchmark.stage = "lagged-greedy-baseline-5000";
    current.status.benchmark.classification = "promote";
    render(
      <R2MapCampaignPanel
        activeView="benchmark"
        requestError={null}
        response={current}
      />,
    );
    expect(screen.getByText("Baseline complete")).toBeTruthy();
    expect(screen.queryByText("promote")).toBeNull();
  });

  it("surfaces the John1 generation runtime verification and cleanup state", () => {
    const current = response("fresh");
    if (!current.status) throw new Error("fresh fixture must carry status");
    current.status.hosts.john1.detail =
      `runtime-stage:verified run=bootstrap-0001 sha256=${"a".repeat(64)} ` +
      "bytes=67108864 cleanup=pending";
    render(<R2MapCampaignPanel requestError={null} response={current} />);
    expect(screen.getByText("Campaign metadata")).toBeTruthy();
    expect(
      screen.getByText(/runtime-stage:verified run=bootstrap-0001/),
    ).toBeTruthy();
    expect(screen.getByText(/cleanup=pending/)).toBeTruthy();
  });
});

describe("ClusterFleetOverview", () => {
  it("renders all four nodes with compact resources and explicit MLX state", () => {
    const current = response("fresh");
    if (!current.status) throw new Error("fresh fixture must carry status");
    render(
      <ClusterFleetOverview
        nodes={[node("john1", 72, 91, 36)]}
        status={current.status}
      />,
    );
    expect(screen.getByText("Fleet status")).toBeTruthy();
    expect(
      document.querySelectorAll(".cluster-fleet-strip article"),
    ).toHaveLength(4);
    expect(screen.getByText("john4")).toBeTruthy();
    expect(screen.getByText("Available")).toBeTruthy();
    expect(screen.getAllByText("CPU")).toHaveLength(4);
    expect(screen.getAllByText("Memory")).toHaveLength(4);
    expect(screen.getAllByText("Disk")).toHaveLength(4);
    expect(screen.getAllByText("MLX")).toHaveLength(4);
    expect(screen.getAllByRole("progressbar")).toHaveLength(12);
    expect(screen.getAllByText("Yes")).toHaveLength(1);
    expect(screen.getAllByText("No")).toHaveLength(3);
    const cpu = screen.getByRole("progressbar", {
      name: "john1 CPU utilization",
    });
    const memory = screen.getByRole("progressbar", {
      name: "john1 Memory utilization",
    });
    expect(cpu.getAttribute("aria-valuenow")).toBe("72");
    expect(cpu.parentElement?.getAttribute("data-tone")).toBe("warm");
    expect(memory.parentElement?.getAttribute("data-tone")).toBe("hot");
    expect(
      cpu.querySelector(".cluster-resource-ring-value")?.getAttribute("style"),
    ).toContain("72 28");
  });
});
