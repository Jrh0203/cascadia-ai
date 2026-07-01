import {
  Activity,
  ArrowLeft,
  Boxes,
  CalendarDays,
  Check,
  CheckCircle2,
  ChevronDown,
  Clock3,
  CircleDot,
  Cpu,
  Database,
  ExternalLink,
  FileText,
  FlaskConical,
  ListChecks,
  LockKeyhole,
  MemoryStick,
  Play,
  RefreshCw,
  Server,
  Terminal,
  TriangleAlert,
  Wifi,
  WifiOff,
  XCircle,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";

import { api } from "./api";
import ClusterFleetOverview from "./ClusterFleetOverview";
import R2MapCampaignPanel from "./R2MapCampaignPanel";
import type { ClusterDetailView } from "./R2MapCampaignPanel";
import {
  formatBytes,
  formatCoreUsage,
  formatDuration,
  formatHistoryTime,
  formatPercent,
  formatUptime,
  experimentOutcomeLabel,
  healthLabel,
  historyLineSegments,
  historyRangeLabel,
  resourceTone,
  sortResearchExperiments,
} from "./cluster";
import type { ClusterHistoryMetric } from "./cluster";
import type {
  ClusterHistoryPoint,
  ClusterHistoryRange,
  ClusterHistoryResponse,
  ClusterHistorySeries,
  ClusterExperimentsResponse,
  ClusterJob,
  ClusterNode,
  ClusterQueueResponse,
  ClusterResponse,
  NodeHealth,
  QueueTask,
  QueueTaskStatus,
  ResearchExperiment,
  ResearchExperimentCriterion,
  R2MapStatusResponse,
  SchedulerStatus,
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
  john4: "#d58ab4",
};
const FLEET_NODE_IDS = new Set(["john1", "john2", "john3", "john4"]);

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
              detail={`${formatCoreUsage(
                (node.cpu_percent / 100) * node.cores,
                node.cores,
              )} · load ${node.load_average[0].toFixed(1)}`}
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
          <span className="cluster-kicker">Process telemetry</span>
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

function SchedulerFabric({ scheduler }: { scheduler: SchedulerStatus }) {
  const liveJobs = scheduler.jobs.filter((job) => job.state !== "completed");
  const totalCpu = scheduler.nodes.reduce((sum, node) => sum + node.cpu_capacity, 0);
  const allocatedCpu = scheduler.nodes.reduce(
    (sum, node) => sum + node.cpu_allocated,
    0,
  );

  return (
    <section className="scheduler-fabric" aria-labelledby="scheduler-fabric-title">
      <header className="scheduler-fabric-heading">
        <div className="scheduler-fabric-identity">
          <span className="scheduler-fabric-mark">
            <Boxes aria-hidden="true" />
          </span>
          <div>
            <span className="cluster-kicker">Execution fabric</span>
            <h2 id="scheduler-fabric-title">Bacalhau</h2>
          </div>
          <span
            className="scheduler-live-chip"
            data-live={scheduler.reachable}
          >
            <span />
            {scheduler.reachable ? `${scheduler.version} live` : "Unavailable"}
          </span>
        </div>
        <a
          className="scheduler-open"
          href={scheduler.web_ui_url}
          target="_blank"
          rel="noreferrer"
        >
          Job detail
          <ExternalLink aria-hidden="true" />
        </a>
      </header>

      {scheduler.reachable ? (
        <>
          <div className="scheduler-fabric-grid">
            <div className="scheduler-state-rail" aria-label="Bacalhau job totals">
              {(
                [
                  ["Queued", scheduler.summary.queued],
                  ["Running", scheduler.summary.running],
                  ["Retrying", scheduler.summary.retrying],
                  ["Failed", scheduler.summary.failed],
                  ["Complete", scheduler.summary.successful],
                ] as const
              ).map(([label, value]) => (
                <div data-state={label.toLowerCase()} key={label}>
                  <strong>{value}</strong>
                  <span>{label}</span>
                </div>
              ))}
            </div>

            <div className="scheduler-capacity">
              <div className="scheduler-capacity-heading">
                <span>Placement capacity</span>
                <strong>
                  {allocatedCpu.toFixed(1)} / {totalCpu.toFixed(0)} CPU allocated
                </strong>
              </div>
              <div className="scheduler-node-strip">
                {scheduler.nodes.map((node) => {
                  const cpuPercent =
                    node.cpu_capacity > 0
                      ? (node.cpu_allocated / node.cpu_capacity) * 100
                      : 0;
                  const memoryPercent =
                    node.memory_capacity_bytes > 0
                      ? (node.memory_allocated_bytes / node.memory_capacity_bytes) * 100
                      : 0;
                  return (
                    <div className="scheduler-node" data-connected={node.connected} key={node.node_id}>
                      <div>
                        <span>{node.label}</span>
                        <strong>
                          {node.running_executions > 0
                            ? `${node.running_executions} running`
                            : "Available"}
                        </strong>
                      </div>
                      <div className="scheduler-node-meter" aria-label={`${node.label} CPU allocation`}>
                        <span style={{ width: `${Math.min(100, cpuPercent)}%` }} />
                      </div>
                      <small>
                        {node.cpu_allocated.toFixed(1)}/{node.cpu_capacity.toFixed(0)} CPU ·{" "}
                        {Math.round(memoryPercent)}% memory
                      </small>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="scheduler-services" aria-label="Fabric services">
              <div data-healthy={scheduler.services.registry_healthy}>
                <Boxes aria-hidden="true" />
                <span>Registry</span>
                <strong>{scheduler.services.registry_healthy ? "Ready" : "Starting"}</strong>
              </div>
              <div data-healthy={scheduler.services.object_store_healthy}>
                <Database aria-hidden="true" />
                <span>Artifacts</span>
                <strong>{scheduler.services.object_store_healthy ? "Ready" : "Starting"}</strong>
              </div>
            </div>
          </div>

          {liveJobs.length > 0 && (
            <div className="scheduler-live-jobs">
              {liveJobs.slice(0, 6).map((job) => (
                <a href={job.detail_url} target="_blank" rel="noreferrer" key={job.id}>
                  <span className="scheduler-job-state" data-state={job.state}>
                    {job.state}
                  </span>
                  <strong>{job.item_id ?? job.name}</strong>
                  <small>
                    {job.experiment_id ?? job.request_id ?? job.id} · {job.attempts || 1} attempt
                    {job.attempts === 1 ? "" : "s"}
                  </small>
                  {job.failure_reason && <em>{job.failure_reason}</em>}
                </a>
              ))}
            </div>
          )}
        </>
      ) : (
        <div className="scheduler-unavailable">
          <TriangleAlert aria-hidden="true" />
          <strong>Scheduler telemetry unavailable</strong>
          <span>{scheduler.error ?? "The Bacalhau API did not respond."}</span>
        </div>
      )}
    </section>
  );
}

const TASK_STATUS_ORDER: Record<QueueTaskStatus, number> = {
  running: 0,
  ready: 1,
  blocked: 2,
  failed: 3,
  completed: 4,
  cancelled: 5,
};

const TASK_STATUS_LABEL: Record<QueueTaskStatus, string> = {
  running: "Running",
  ready: "Ready",
  blocked: "Waiting",
  failed: "Failed",
  completed: "Complete",
  cancelled: "Cancelled",
};

const WORKLOAD_CLASS_LABEL: Record<QueueTask["workload_class"], string> = {
  "independent-experiment": "Experiment",
  "divisible-evidence": "Evidence",
  "shared-prerequisite": "Prerequisite",
  replica: "Replica",
};

function QueuePanel({
  queue,
  error,
}: {
  queue: ClusterQueueResponse | null;
  error: string | null;
}) {
  const tasks = useMemo(
    () =>
      [...(queue?.tasks ?? [])].sort(
        (left, right) =>
          TASK_STATUS_ORDER[left.status] - TASK_STATUS_ORDER[right.status] ||
          left.priority - right.priority ||
          Number(right.critical_path) - Number(left.critical_path) ||
          left.id.localeCompare(right.id),
      ),
    [queue],
  );
  const openTasks = tasks.filter(
    (task) => task.status !== "completed" && task.status !== "cancelled",
  );
  const visibleTasks = openTasks.length > 0 ? openTasks : tasks.slice(0, 8);
  const hosts = Object.entries(queue?.hosts ?? {}).sort(([left], [right]) =>
    left.localeCompare(right),
  );

  return (
    <section className="queue-section" aria-labelledby="queue-title">
      <header className="cluster-section-heading">
        <div>
          <span className="cluster-kicker">Scheduler view</span>
          <h2 id="queue-title">Research queue</h2>
        </div>
        <span className="queue-campaign">
          {queue?.campaign_id ?? "No active campaign"}
        </span>
      </header>

      {(error || queue?.error) && (
        <div className="cluster-alert queue-alert" role="alert">
          <TriangleAlert aria-hidden="true" />
          <span>{error ?? queue?.error}</span>
        </div>
      )}

      {queue?.configured ? (
        <>
          <div className="queue-summary">
            <div>
              <span>Running</span>
              <strong>{queue.summary.running}</strong>
              <small>{queue.summary.duplicate_running} replicas</small>
            </div>
            <div>
              <span>Ready</span>
              <strong>{queue.summary.ready}</strong>
              <small>{queue.summary.critical_path_ready} critical</small>
            </div>
            <div>
              <span>Blocked</span>
              <strong>{queue.summary.blocked}</strong>
              <small>dependency wait</small>
            </div>
            <div>
              <span>Complete</span>
              <strong>{queue.summary.completed}</strong>
              <small>{queue.summary.decisions_completed} decisions</small>
            </div>
          </div>

          <div className="queue-host-strip" aria-label="Host scheduling intent">
            {hosts.map(([host, state]) => (
              <div data-intent={state.intent} key={host}>
                <span>{host}</span>
                <strong>{state.intent.replaceAll("-", " ")}</strong>
                <small title={state.reason ?? undefined}>
                  {state.reason ?? "No reservation"}
                </small>
              </div>
            ))}
          </div>

          {visibleTasks.length > 0 ? (
            <div className="queue-table-wrap">
              <table className="queue-table">
                <thead>
                  <tr>
                    <th>Status</th>
                    <th>Task</th>
                    <th>Decision</th>
                    <th>Class</th>
                    <th>Host</th>
                    <th>Resources</th>
                    <th>Expected</th>
                  </tr>
                </thead>
                <tbody>
                  {visibleTasks.map((task) => (
                    <tr key={task.id}>
                      <td>
                        <span className="queue-status" data-status={task.status}>
                          {task.status === "running" ? (
                            <Play aria-hidden="true" />
                          ) : task.status === "blocked" ? (
                            <LockKeyhole aria-hidden="true" />
                          ) : (
                            <ListChecks aria-hidden="true" />
                          )}
                          {TASK_STATUS_LABEL[task.status]}
                        </span>
                      </td>
                      <td>
                        <strong>{task.title}</strong>
                        <small>
                          {task.critical_path ? "Critical path · " : ""}
                          {task.experiment_id}
                        </small>
                      </td>
                      <td>
                        <span className="queue-decision" title={task.decision}>
                          {task.decision}
                        </span>
                      </td>
                      <td>{WORKLOAD_CLASS_LABEL[task.workload_class]}</td>
                      <td>{task.claim?.host ?? task.compatible_hosts.join(", ")}</td>
                      <td>
                        {task.resources.uses_mlx ? "MLX" : `${task.resources.cpu_cores} CPU`}
                      </td>
                      <td>{formatDuration(task.expected_runtime_seconds)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="workload-empty">
              <ListChecks aria-hidden="true" />
              <strong>Campaign queue is empty</strong>
              <span>No work is waiting for assignment.</span>
            </div>
          )}
        </>
      ) : (
        <div className="workload-empty">
          <ListChecks aria-hidden="true" />
          <strong>No research campaign is configured</strong>
          <span>The fleet remains available for direct work.</span>
        </div>
      )}
    </section>
  );
}

function experimentTime(experiment: ResearchExperiment): string {
  const timestamp =
    experiment.completed_unix_ms ??
    experiment.started_unix_ms ??
    experiment.updated_unix_ms;
  return new Date(timestamp).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function ExperimentStateIcon({ experiment }: { experiment: ResearchExperiment }) {
  if (experiment.status === "running") {
    return <CircleDot aria-hidden="true" />;
  }
  if (experiment.status === "completed" && experiment.outcome === "passed") {
    return <CheckCircle2 aria-hidden="true" />;
  }
  if (experiment.status === "completed" && experiment.outcome === "failed") {
    return <XCircle aria-hidden="true" />;
  }
  if (
    experiment.status === "completed" &&
    (experiment.outcome === "invalid" || experiment.outcome === "inconclusive")
  ) {
    return <TriangleAlert aria-hidden="true" />;
  }
  return <FlaskConical aria-hidden="true" />;
}

function CriterionIcon({ criterion }: { criterion: ResearchExperimentCriterion }) {
  if (criterion.passed === true) {
    return <CheckCircle2 aria-hidden="true" />;
  }
  if (criterion.passed === false) {
    return <XCircle aria-hidden="true" />;
  }
  return <CircleDot aria-hidden="true" />;
}

function ExperimentRow({ experiment }: { experiment: ResearchExperiment }) {
  const [expanded, setExpanded] = useState(false);
  const detailId = `experiment-${experiment.id}-details`;
  const visibleMetrics = experiment.metrics.slice(0, 4);
  const stateLabel = experimentOutcomeLabel(
    experiment.status,
    experiment.outcome,
  );

  return (
    <article
      className="experiment-row"
      data-outcome={experiment.outcome}
      data-status={experiment.status}
    >
      <button
        aria-controls={detailId}
        aria-expanded={expanded}
        className="experiment-row-toggle"
        onClick={() => setExpanded((value) => !value)}
        type="button"
      >
        <span className="experiment-state" data-outcome={experiment.outcome}>
          <ExperimentStateIcon experiment={experiment} />
          {stateLabel}
        </span>
        <span className="experiment-primary">
          <strong>{experiment.title}</strong>
          <small>{experiment.summary}</small>
        </span>
        <span className="experiment-quick-metrics">
          {visibleMetrics.map((metric, index) => (
            <span data-tone={metric.tone} key={`${metric.label}-${index}`}>
              <small>{metric.label}</small>
              <strong>{metric.value}</strong>
            </span>
          ))}
        </span>
        <span className="experiment-when">
          <strong>{experiment.hosts.join(", ") || "Unassigned"}</strong>
          <small>
            {experiment.status === "completed" ? "Completed" : "Updated"}{" "}
            {experimentTime(experiment)}
          </small>
        </span>
        <ChevronDown
          aria-hidden="true"
          className={expanded ? "is-expanded" : ""}
        />
      </button>

      {expanded && (
        <div className="experiment-details" id={detailId}>
          <div className="experiment-detail-copy">
            <div>
              <span>Hypothesis</span>
              <p>{experiment.hypothesis}</p>
            </div>
            <div>
              <span>Verdict</span>
              <p>
                {experiment.verdict ??
                  (experiment.status === "running"
                    ? "Pending the preregistered gates."
                    : "No formal verdict recorded.")}
              </p>
            </div>
          </div>

          {experiment.criteria.length > 0 && (
            <div className="experiment-criteria">
              <h4>Success criteria</h4>
              <ul>
                {experiment.criteria.map((criterion) => (
                  <li data-passed={criterion.passed} key={criterion.label}>
                    <CriterionIcon criterion={criterion} />
                    <span>
                      <strong>{criterion.label}</strong>
                      {criterion.observed && <small>{criterion.observed}</small>}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="experiment-detail-grid">
            <div>
              <h4>Research notes</h4>
              {experiment.notes.length > 0 ? (
                <ul className="experiment-notes">
                  {experiment.notes.map((note) => (
                    <li key={note}>{note}</li>
                  ))}
                </ul>
              ) : (
                <p className="experiment-muted">No additional notes recorded.</p>
              )}
            </div>
            <div>
              <h4>Artifacts</h4>
              {experiment.artifacts.length > 0 ? (
                <ul className="experiment-artifacts">
                  {experiment.artifacts.map((artifact) => (
                    <li key={`${artifact.label}-${artifact.path}`}>
                      <FileText aria-hidden="true" />
                      <span>
                        <strong>{artifact.label}</strong>
                        <code>{artifact.path}</code>
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="experiment-muted">No artifacts recorded.</p>
              )}
            </div>
          </div>

          <dl className="experiment-metadata">
            {experiment.plan_section && (
              <div>
                <dt>Plan</dt>
                <dd>{experiment.plan_section}</dd>
              </div>
            )}
            <div>
              <dt>Experiment</dt>
              <dd>{experiment.id}</dd>
            </div>
            {experiment.task_ids.length > 0 && (
              <div>
                <dt>Queue tasks</dt>
                <dd>{experiment.task_ids.join(", ")}</dd>
              </div>
            )}
            {experiment.tags.length > 0 && (
              <div>
                <dt>Tags</dt>
                <dd>{experiment.tags.join(", ")}</dd>
              </div>
            )}
          </dl>
        </div>
      )}
    </article>
  );
}

function ExperimentPanel({
  experiments,
  error,
}: {
  experiments: ClusterExperimentsResponse | null;
  error: string | null;
}) {
  const records = useMemo(
    () => sortResearchExperiments(experiments?.experiments ?? []),
    [experiments],
  );
  const running = records.filter(({ status }) => status === "running").length;
  const completed = records.filter(({ status }) => status === "completed").length;

  return (
    <section
      className="experiment-section"
      aria-labelledby="experiments-title"
    >
      <header className="cluster-section-heading">
        <div>
          <span className="cluster-kicker">Decision ledger</span>
          <h2 id="experiments-title">Research experiments</h2>
        </div>
        <span>
          {records.length > 0
            ? `${running} running · ${completed} completed`
            : "Awaiting ledger"}
        </span>
      </header>

      {(error || experiments?.error) && (
        <div className="cluster-alert experiment-alert" role="alert">
          <TriangleAlert aria-hidden="true" />
          <span>{error ?? experiments?.error}</span>
        </div>
      )}

      {experiments?.configured ? (
        records.length > 0 ? (
          <div className="experiment-list">
            {records.map((experiment) => (
              <ExperimentRow experiment={experiment} key={experiment.id} />
            ))}
          </div>
        ) : (
          <div className="workload-empty">
            <FlaskConical aria-hidden="true" />
            <strong>The research ledger is empty</strong>
            <span>No experiments have been recorded.</span>
          </div>
        )
      ) : (
        <div className="workload-empty">
          <FlaskConical aria-hidden="true" />
          <strong>No research ledger is configured</strong>
          <span>Live scheduling remains available.</span>
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
            {title} for all configured nodes over {historyRangeLabel(history.range)}
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
  const [queue, setQueue] = useState<ClusterQueueResponse | null>(null);
  const [experiments, setExperiments] =
    useState<ClusterExperimentsResponse | null>(null);
  const [r2MapStatus, setR2MapStatus] = useState<R2MapStatusResponse | null>(null);
  const [history, setHistory] = useState<ClusterHistoryResponse | null>(null);
  const [historyRange, setHistoryRange] = useState<ClusterHistoryRange>("1d");
  const [detailView, setDetailView] = useState<ClusterDetailView>("training");
  const [error, setError] = useState<string | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [queueError, setQueueError] = useState<string | null>(null);
  const [experimentError, setExperimentError] = useState<string | null>(null);
  const [r2MapError, setR2MapError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [historyRefreshing, setHistoryRefreshing] = useState(false);
  const requestRunning = useRef(false);
  const historyRequestRunning = useRef(false);
  const queueRequestRunning = useRef(false);
  const experimentRequestRunning = useRef(false);
  const r2MapRequestRunning = useRef(false);

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

  const refreshQueue = useCallback(async () => {
    if (queueRequestRunning.current) {
      return;
    }
    queueRequestRunning.current = true;
    try {
      const next = await api.clusterQueue();
      setQueue(next);
      setQueueError(null);
    } catch (reason) {
      setQueueError(messageFrom(reason));
    } finally {
      queueRequestRunning.current = false;
    }
  }, []);

  const refreshExperiments = useCallback(async () => {
    if (experimentRequestRunning.current) {
      return;
    }
    experimentRequestRunning.current = true;
    try {
      const next = await api.clusterExperiments();
      setExperiments(next);
      setExperimentError(null);
    } catch (reason) {
      setExperimentError(messageFrom(reason));
    } finally {
      experimentRequestRunning.current = false;
    }
  }, []);

  const refreshR2Map = useCallback(async () => {
    if (r2MapRequestRunning.current) {
      return;
    }
    r2MapRequestRunning.current = true;
    try {
      const next = await api.r2MapStatus();
      setR2MapStatus(next);
      setR2MapError(null);
    } catch (reason) {
      setR2MapError(messageFrom(reason));
    } finally {
      r2MapRequestRunning.current = false;
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
    void refreshQueue();
    void refreshExperiments();
    void refreshR2Map();
    const timer = window.setInterval(() => {
      void refresh();
      void refreshQueue();
      void refreshExperiments();
      void refreshR2Map();
    }, POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [refresh, refreshExperiments, refreshQueue, refreshR2Map]);

  useEffect(() => {
    void refreshHistory(historyRange);
    const timer = window.setInterval(
      () => void refreshHistory(historyRange),
      HISTORY_POLL_INTERVAL_MS,
    );
    return () => window.clearInterval(timer);
  }, [historyRange, refreshHistory]);

  const displayedHistory = useMemo(() => {
    if (history?.range !== historyRange) return null;
    return {
      ...history,
      series: history.series.filter(({ node_id }) => FLEET_NODE_IDS.has(node_id)),
    };
  }, [history, historyRange]);
  const activeNodes = useMemo(
    () => cluster?.nodes.filter(({ id }) => FLEET_NODE_IDS.has(id)) ?? [],
    [cluster],
  );
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
              void refreshQueue();
              void refreshExperiments();
              void refreshR2Map();
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
        <ClusterFleetOverview
          nodes={activeNodes}
          status={r2MapStatus?.status ?? null}
        />

        {cluster && <SchedulerFabric scheduler={cluster.scheduler} />}

        {error && (
          <div className="cluster-alert" role="alert">
            <TriangleAlert aria-hidden="true" />
            <span>{error}. Showing the most recent successful sample.</span>
          </div>
        )}

        <R2MapCampaignPanel
          activeView={detailView}
          nodes={activeNodes}
          onViewChange={setDetailView}
          requestError={r2MapError}
          response={r2MapStatus}
        />

        <UtilizationHistory
          error={historyError}
          history={displayedHistory}
          loading={historyRefreshing}
          onRangeChange={setHistoryRange}
          range={historyRange}
        />

        {detailView === "fleet" && (
          <div
            aria-labelledby="cluster-tab-fleet"
            className="cluster-tab-panel"
            id="cluster-panel-fleet"
            role="tabpanel"
          >
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
                {cluster
                  ? activeNodes.map((node) => <NodeCard key={node.id} node={node} />)
                  : [1, 2, 3, 4].map((index) => (
                      <div className="cluster-node cluster-node-loading" key={index}>
                        <span />
                        <span />
                        <span />
                      </div>
                    ))}
              </div>
            </section>
            <WorkloadTable nodes={activeNodes} />
          </div>
        )}

        {detailView === "research" && (
          <div
            aria-labelledby="cluster-tab-research"
            className="cluster-tab-panel"
            id="cluster-panel-research"
            role="tabpanel"
          >
            <ExperimentPanel error={experimentError} experiments={experiments} />
            <QueuePanel error={queueError} queue={queue} />
          </div>
        )}

      </main>
    </div>
  );
}
