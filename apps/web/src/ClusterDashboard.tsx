import {
  Activity,
  ArrowLeft,
  CalendarDays,
  Check,
  Clock3,
  Cpu,
  HardDrive,
  MemoryStick,
  RefreshCw,
  Server,
  Terminal,
  TriangleAlert,
  Wifi,
  WifiOff,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";

import { api } from "./api";
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
import type { ClusterHistoryMetric } from "./cluster";
import type {
  ClusterHistoryPoint,
  ClusterHistoryRange,
  ClusterHistoryResponse,
  ClusterHistorySeries,
  ClusterJob,
  ClusterNode,
  ClusterResponse,
  NodeHealth,
} from "./types";

const POLL_INTERVAL_MS = 5_000;
const HISTORY_POLL_INTERVAL_MS = 30_000;
const CHART_WIDTH = 840;
const CHART_HEIGHT = 248;
const PLOT_LEFT = 48;
const PLOT_TOP = 18;
const PLOT_WIDTH = 774;
const PLOT_HEIGHT = 188;
const NODE_COLORS: Record<string, string> = {
  john1: "#7bd995",
  john2: "#6dbec3",
  john3: "#e1b85d",
};

function messageFrom(reason: unknown): string {
  return reason instanceof Error ? reason.message : "Cluster telemetry failed";
}

function HealthIcon({ health }: { health: NodeHealth }) {
  if (health === "offline") {
    return <WifiOff aria-hidden="true" />;
  }
  if (health === "warning") {
    return <TriangleAlert aria-hidden="true" />;
  }
  if (health === "busy") {
    return <Activity aria-hidden="true" />;
  }
  return <Check aria-hidden="true" />;
}

function ResourceBar({
  label,
  value,
  detail,
}: {
  label: string;
  value: number;
  detail: string;
}) {
  const bounded = Math.max(0, Math.min(100, value));
  return (
    <div className="cluster-resource">
      <div className="cluster-resource-label">
        <span>{label}</span>
        <strong>{formatPercent(bounded)}</strong>
      </div>
      <div
        className="cluster-meter"
        data-tone={resourceTone(bounded)}
        role="progressbar"
        aria-label={`${label} utilization`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(bounded)}
      >
        <span style={{ width: `${bounded}%` }} />
      </div>
      <small>{detail}</small>
    </div>
  );
}

function NodeCard({ node }: { node: ClusterNode }) {
  return (
    <article className="cluster-node" data-health={node.health}>
      <header className="cluster-node-heading">
        <div className="node-identity">
          <span className="node-index">{node.id.slice(-1)}</span>
          <div>
            <h2>{node.label}</h2>
            <p>{node.role}</p>
          </div>
        </div>
        <span className="health-chip" data-health={node.health}>
          <HealthIcon health={node.health} />
          {healthLabel(node.health)}
        </span>
      </header>

      <div className="node-address">
        {node.reachable ? <Wifi aria-hidden="true" /> : <WifiOff aria-hidden="true" />}
        <span>{node.address}</span>
        <small>{node.reachable ? `${node.sample_latency_ms} ms probe` : node.error}</small>
      </div>

      {node.reachable ? (
        <>
          <div className="cluster-resource-grid">
            <ResourceBar
              label="CPU"
              value={node.cpu_percent}
              detail={`${node.cores} cores · load ${node.load_average[0].toFixed(1)}`}
            />
            <ResourceBar
              label="Memory"
              value={node.memory_used_percent}
              detail={`${formatBytes(node.memory_used_bytes)} / ${formatBytes(node.memory_total_bytes)}`}
            />
            <ResourceBar
              label="Disk"
              value={node.disk_used_percent}
              detail={`${formatBytes(node.disk_available_bytes)} available`}
            />
          </div>

          <dl className="node-facts">
            <div>
              <dt>
                <Clock3 aria-hidden="true" />
                Uptime
              </dt>
              <dd>{formatUptime(node.uptime_seconds)}</dd>
            </div>
          </dl>
        </>
      ) : (
        <div className="node-offline">
          <WifiOff aria-hidden="true" />
          <strong>No response over Tailscale</strong>
          <span>{node.error ?? "SSH connection unavailable"}</span>
        </div>
      )}
    </article>
  );
}

interface VisibleJob extends ClusterJob {
  nodeId: string;
  nodeLabel: string;
}

function WorkloadTable({ nodes }: { nodes: ClusterNode[] }) {
  const jobs = useMemo(
    () =>
      nodes.flatMap((node) =>
        node.jobs.map((job) => ({
          ...job,
          nodeId: node.id,
          nodeLabel: node.label,
        })),
      ),
    [nodes],
  );

  return (
    <section className="workload-section" aria-labelledby="workload-title">
      <header className="cluster-section-heading">
        <div>
          <span className="cluster-kicker">Scheduler view</span>
          <h2 id="workload-title">Active workloads</h2>
        </div>
        <span className="workload-count">{jobs.length} running</span>
      </header>

      {jobs.length > 0 ? (
        <div className="workload-table-wrap">
          <table className="workload-table">
            <thead>
              <tr>
                <th>Node</th>
                <th>Workload</th>
                <th>PID</th>
                <th>Elapsed</th>
                <th>CPU</th>
                <th>Memory</th>
                <th>Command</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((job: VisibleJob) => (
                <tr key={`${job.nodeId}-${job.pid}`}>
                  <td>
                    <span className="table-node">{job.nodeLabel}</span>
                  </td>
                  <td>
                    <strong>{job.workload}</strong>
                  </td>
                  <td>{job.pid}</td>
                  <td>{job.elapsed}</td>
                  <td>{(job.cpu_percent / 100).toFixed(1)} cores</td>
                  <td>{job.memory_percent.toFixed(1)}%</td>
                  <td>
                    <code title={job.command}>{job.command}</code>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="workload-empty">
          <Terminal aria-hidden="true" />
          <strong>No Cascadia jobs are running</strong>
          <span>All reachable workers are available.</span>
        </div>
      )}
    </section>
  );
}

function nearestHistoryPoint(
  series: ClusterHistorySeries,
  timestamp: number,
  metric: ClusterHistoryMetric,
): ClusterHistoryPoint | null {
  let nearest: ClusterHistoryPoint | null = null;
  let nearestDistance = Number.POSITIVE_INFINITY;
  for (const point of series.points) {
    if (point[metric] === null || point.reachable_percent <= 0) {
      continue;
    }
    const distance = Math.abs(point.timestamp_unix_ms - timestamp);
    if (distance < nearestDistance) {
      nearest = point;
      nearestDistance = distance;
    }
  }
  return nearest;
}

function HistoryChart({
  history,
  metric,
  title,
}: {
  history: ClusterHistoryResponse;
  metric: ClusterHistoryMetric;
  title: string;
}) {
  const [hoverTimestamp, setHoverTimestamp] = useState<number | null>(null);
  const metricIsCpu = metric === "cpu_percent";
  const displayTimestamp =
    hoverTimestamp ?? history.newest_sample_unix_ms ?? history.end_unix_ms;
  const expectedGapMs = Math.max(
    history.source_sample_interval_seconds,
    history.bucket_seconds,
  ) * 1_000;

  const handlePointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    const viewX = ((event.clientX - bounds.left) / bounds.width) * CHART_WIDTH;
    const plotRatio = Math.max(0, Math.min(1, (viewX - PLOT_LEFT) / PLOT_WIDTH));
    setHoverTimestamp(
      history.start_unix_ms +
        plotRatio * (history.end_unix_ms - history.start_unix_ms),
    );
  };

  return (
    <article className="history-chart" data-metric={metricIsCpu ? "cpu" : "memory"}>
      <header className="history-chart-heading">
        <div>
          {metricIsCpu ? <Cpu aria-hidden="true" /> : <MemoryStick aria-hidden="true" />}
          <h3>{title}</h3>
        </div>
        <time dateTime={new Date(displayTimestamp).toISOString()}>
          {formatHistoryTime(displayTimestamp, history.range)}
        </time>
      </header>

      <div className="history-legend" aria-label={`${title} node summary`}>
        {history.series.map((series) => {
          const point = nearestHistoryPoint(series, displayTimestamp, metric);
          const value = point?.[metric] ?? null;
          const average = metricIsCpu
            ? series.summary.average_cpu_percent
            : series.summary.average_memory_percent;
          const peak = metricIsCpu
            ? series.summary.peak_cpu_percent
            : series.summary.peak_memory_percent;
          return (
            <div key={series.node_id}>
              <span
                className="history-swatch"
                style={{ backgroundColor: NODE_COLORS[series.node_id] }}
              />
              <strong>{series.node_label}</strong>
              <b>{value === null ? "—" : formatPercent(value)}</b>
              <small>
                avg {average === null ? "—" : formatPercent(average)} · peak{" "}
                {peak === null ? "—" : formatPercent(peak)}
              </small>
            </div>
          );
        })}
      </div>

      <div className="history-plot">
        <svg
          viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
          role="img"
          aria-label={`${title} over ${historyRangeLabel(history.range)}`}
          onPointerMove={handlePointerMove}
          onPointerLeave={() => setHoverTimestamp(null)}
        >
          <title>
            {title} for John 1, John 2, and John 3 over{" "}
            {historyRangeLabel(history.range)}
          </title>
          {[0, 25, 50, 75, 100].map((value) => {
            const y = PLOT_TOP + PLOT_HEIGHT - (value / 100) * PLOT_HEIGHT;
            return (
              <g key={value}>
                <line
                  className="history-gridline"
                  x1={PLOT_LEFT}
                  x2={PLOT_LEFT + PLOT_WIDTH}
                  y1={y}
                  y2={y}
                />
                <text className="history-y-label" x={PLOT_LEFT - 10} y={y + 4}>
                  {value}
                </text>
              </g>
            );
          })}

          <g transform={`translate(${PLOT_LEFT} ${PLOT_TOP})`}>
            {history.series.flatMap((series) =>
              historyLineSegments(
                series.points,
                metric,
                history.start_unix_ms,
                history.end_unix_ms,
                PLOT_WIDTH,
                PLOT_HEIGHT,
                expectedGapMs,
              ).map((path, index) => (
                <path
                  className="history-line"
                  d={path}
                  key={`${series.node_id}-${index}`}
                  stroke={NODE_COLORS[series.node_id]}
                />
              )),
            )}
            {history.series.map((series) => {
              const point = nearestHistoryPoint(
                series,
                history.newest_sample_unix_ms ?? history.end_unix_ms,
                metric,
              );
              const value = point?.[metric] ?? null;
              if (point === null || value === null) {
                return null;
              }
              const x =
                ((point.timestamp_unix_ms - history.start_unix_ms) /
                  Math.max(1, history.end_unix_ms - history.start_unix_ms)) *
                PLOT_WIDTH;
              const y = PLOT_HEIGHT - (Math.max(0, Math.min(100, value)) / 100) * PLOT_HEIGHT;
              return (
                <circle
                  className="history-latest"
                  cx={x}
                  cy={y}
                  fill={NODE_COLORS[series.node_id]}
                  key={series.node_id}
                  r={3.5}
                />
              );
            })}
          </g>

          {hoverTimestamp !== null && (
            <line
              className="history-crosshair"
              x1={
                PLOT_LEFT +
                ((hoverTimestamp - history.start_unix_ms) /
                  Math.max(1, history.end_unix_ms - history.start_unix_ms)) *
                  PLOT_WIDTH
              }
              x2={
                PLOT_LEFT +
                ((hoverTimestamp - history.start_unix_ms) /
                  Math.max(1, history.end_unix_ms - history.start_unix_ms)) *
                  PLOT_WIDTH
              }
              y1={PLOT_TOP}
              y2={PLOT_TOP + PLOT_HEIGHT}
            />
          )}

          {[0, 0.5, 1].map((ratio) => {
            const timestamp =
              history.start_unix_ms +
              ratio * (history.end_unix_ms - history.start_unix_ms);
            return (
              <text
                className="history-x-label"
                key={ratio}
                textAnchor={ratio === 0 ? "start" : ratio === 1 ? "end" : "middle"}
                x={PLOT_LEFT + ratio * PLOT_WIDTH}
                y={CHART_HEIGHT - 10}
              >
                {formatHistoryTime(timestamp, history.range)}
              </text>
            );
          })}
        </svg>
        {history.raw_sample_count === 0 && (
          <div className="history-empty">
            <Activity aria-hidden="true" />
            <strong>Waiting for the first retained sample</strong>
          </div>
        )}
      </div>
    </article>
  );
}

function UtilizationHistory({
  history,
  range,
  loading,
  error,
  onRangeChange,
}: {
  history: ClusterHistoryResponse | null;
  range: ClusterHistoryRange;
  loading: boolean;
  error: string | null;
  onRangeChange: (range: ClusterHistoryRange) => void;
}) {
  return (
    <section className="history-section" aria-labelledby="history-title">
      <header className="cluster-section-heading history-section-heading">
        <div>
          <span className="cluster-kicker">Telemetry archive</span>
          <h2 id="history-title">Utilization history</h2>
        </div>
        <div className="history-heading-tools">
          {history && (
            <span className="history-sample-count">
              {history.raw_sample_count.toLocaleString()} samples ·{" "}
              {history.source_sample_interval_seconds}s capture
            </span>
          )}
          <div className="history-range" aria-label="Utilization history range">
            {(["1d", "7d"] as const).map((value) => (
              <button
                aria-pressed={range === value}
                className={range === value ? "is-active" : ""}
                key={value}
                onClick={() => onRangeChange(value)}
                type="button"
              >
                <CalendarDays aria-hidden="true" />
                {value === "1d" ? "1D" : "7D"}
              </button>
            ))}
          </div>
        </div>
      </header>

      {error && (
        <div className="cluster-alert history-alert" role="alert">
          <TriangleAlert aria-hidden="true" />
          <span>{error}. Showing the most recent retained history.</span>
        </div>
      )}

      {history ? (
        <div className="history-chart-grid" aria-busy={loading}>
          <HistoryChart history={history} metric="cpu_percent" title="CPU utilization" />
          <HistoryChart
            history={history}
            metric="memory_percent"
            title="Memory utilization"
          />
        </div>
      ) : (
        <div className="history-chart-grid" aria-busy="true">
          {[0, 1].map((index) => (
            <div className="history-chart history-chart-loading" key={index}>
              <span />
              <span />
              <span />
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

export default function ClusterDashboard() {
  const [cluster, setCluster] = useState<ClusterResponse | null>(null);
  const [history, setHistory] = useState<ClusterHistoryResponse | null>(null);
  const [historyRange, setHistoryRange] = useState<ClusterHistoryRange>("1d");
  const [error, setError] = useState<string | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [historyRefreshing, setHistoryRefreshing] = useState(false);
  const requestRunning = useRef(false);
  const historyRequestRunning = useRef(false);

  const refresh = useCallback(async () => {
    if (requestRunning.current) {
      return;
    }
    requestRunning.current = true;
    setRefreshing(true);
    try {
      const next = await api.cluster();
      setCluster(next);
      setError(null);
    } catch (reason) {
      setError(messageFrom(reason));
    } finally {
      requestRunning.current = false;
      setRefreshing(false);
    }
  }, []);

  const refreshHistory = useCallback(async (range: ClusterHistoryRange) => {
    if (historyRequestRunning.current) {
      return;
    }
    historyRequestRunning.current = true;
    setHistoryRefreshing(true);
    try {
      const next = await api.clusterHistory(range);
      setHistory(next);
      setHistoryError(null);
    } catch (reason) {
      setHistoryError(messageFrom(reason));
    } finally {
      historyRequestRunning.current = false;
      setHistoryRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    void refreshHistory(historyRange);
    const timer = window.setInterval(
      () => void refreshHistory(historyRange),
      HISTORY_POLL_INTERVAL_MS,
    );
    return () => window.clearInterval(timer);
  }, [historyRange, refreshHistory]);

  const displayedHistory = history?.range === historyRange ? history : null;
  const memoryPercent =
    cluster && cluster.summary.memory_total_bytes > 0
      ? (cluster.summary.memory_used_bytes / cluster.summary.memory_total_bytes) * 100
      : 0;
  const sampledAt = cluster
    ? new Date(cluster.collected_at_unix_ms).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      })
    : "Awaiting first sample";

  return (
    <div className="cluster-shell">
      <header className="cluster-topbar">
        <a className="cluster-back" href="/" title="Return to Cascadia Lab">
          <ArrowLeft aria-hidden="true" />
          <span>Lab</span>
        </a>
        <div className="cluster-wordmark">
          <span className="cluster-wordmark-icon">
            <Server aria-hidden="true" />
          </span>
          <div>
            <strong>Cascadia Compute</strong>
            <span>Local research cluster</span>
          </div>
        </div>
        <div className="cluster-live">
          <span className="live-dot" data-error={Boolean(error)} />
          <div>
            <strong>{error ? "Telemetry delayed" : "Live telemetry"}</strong>
            <small>{sampledAt}</small>
          </div>
          <button
            className="cluster-refresh"
            type="button"
            title="Refresh cluster telemetry"
            onClick={() => {
              void refresh();
              void refreshHistory(historyRange);
            }}
            disabled={refreshing || historyRefreshing}
          >
            <RefreshCw
              className={refreshing || historyRefreshing ? "is-spinning" : ""}
              aria-hidden="true"
            />
          </button>
        </div>
      </header>

      <main className="cluster-main">
        <section className="cluster-overview" aria-label="Cluster summary">
          <div className="cluster-summary" aria-label="Cluster summary">
            <div>
              <span>Online</span>
              <strong>
                {cluster?.summary.online_nodes ?? "—"}
                <small>/{cluster?.summary.total_nodes ?? 3}</small>
              </strong>
              <em data-tone={cluster?.summary.degraded_nodes === 0 ? "good" : "hot"}>
                {cluster?.summary.degraded_nodes === 0 ? "Full fleet" : "Degraded"}
              </em>
            </div>
            <div>
              <span>CPU load</span>
              <strong>{cluster ? formatPercent(cluster.summary.average_cpu_percent) : "—"}</strong>
              <em data-tone={resourceTone(cluster?.summary.average_cpu_percent ?? 0)}>
                Fleet average
              </em>
            </div>
            <div>
              <span>Memory</span>
              <strong>{cluster ? formatPercent(memoryPercent) : "—"}</strong>
              <em data-tone={resourceTone(memoryPercent)}>
                {cluster ? `${formatBytes(cluster.summary.memory_used_bytes)} in use` : "Sampling"}
              </em>
            </div>
            <div>
              <span>Workloads</span>
              <strong>{cluster?.summary.active_jobs ?? "—"}</strong>
              <em data-tone={cluster?.summary.active_jobs ? "warm" : "good"}>
                {cluster?.summary.busy_nodes ?? 0}{" "}
                {cluster?.summary.busy_nodes === 1 ? "node" : "nodes"} occupied
              </em>
            </div>
          </div>
        </section>

        {error && (
          <div className="cluster-alert" role="alert">
            <TriangleAlert aria-hidden="true" />
            <span>{error}. Showing the most recent successful sample.</span>
          </div>
        )}

        <UtilizationHistory
          error={historyError}
          history={displayedHistory}
          loading={historyRefreshing}
          onRangeChange={setHistoryRange}
          range={historyRange}
        />

        <section className="node-section" aria-labelledby="nodes-title">
          <header className="cluster-section-heading">
            <div>
              <span className="cluster-kicker">Fleet detail</span>
              <h2 id="nodes-title">Nodes</h2>
            </div>
            <span>
              {cluster ? `${cluster.collection_duration_ms} ms collection` : "Connecting"}
            </span>
          </header>
          <div className="cluster-node-grid">
            {cluster?.nodes.map((node) => <NodeCard key={node.id} node={node} />) ??
              [1, 2, 3].map((index) => (
                <div className="cluster-node cluster-node-loading" key={index}>
                  <span />
                  <span />
                  <span />
                </div>
              ))}
          </div>
        </section>

        <WorkloadTable nodes={cluster?.nodes ?? []} />

        <footer className="cluster-footer">
          <span>
            <Cpu aria-hidden="true" />
            CPU is normalized across logical cores
          </span>
          <span>
            <MemoryStick aria-hidden="true" />
            macOS pressure-aware memory
          </span>
          <span>
            <HardDrive aria-hidden="true" />
            Data volume capacity
          </span>
        </footer>
      </main>
    </div>
  );
}
