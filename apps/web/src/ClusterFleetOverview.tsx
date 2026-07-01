import { healthLabel, resourceTone } from "./cluster";
import type { ClusterNode, R2MapCampaignStatus } from "./types";

const FLEET_HOSTS = ["john1", "john2", "john3", "john4"] as const;

function ResourceRing({
  host,
  label,
  value,
}: {
  host: string;
  label: string;
  value: number | null;
}) {
  const bounded = value === null ? 0 : Math.max(0, Math.min(100, value));
  const tone = value === null ? "unknown" : resourceTone(bounded);
  return (
    <div className="cluster-resource-ring" data-tone={tone}>
      <div
        aria-label={`${host} ${label} utilization`}
        aria-valuemax={100}
        aria-valuemin={0}
        aria-valuenow={value === null ? undefined : Math.round(bounded)}
        role="progressbar"
      >
        <svg aria-hidden="true" viewBox="0 0 48 48">
          <circle className="cluster-resource-ring-track" cx="24" cy="24" r="18" />
          <circle
            className="cluster-resource-ring-value"
            cx="24"
            cy="24"
            pathLength="100"
            r="18"
            style={{ strokeDasharray: `${bounded} ${100 - bounded}` }}
          />
        </svg>
        <strong>{value === null ? "—" : `${Math.round(bounded)}%`}</strong>
      </div>
      <span>{label}</span>
    </div>
  );
}

function phaseLabel(value: string): string {
  return value.replaceAll("-", " ");
}

function hostIntentLabel(
  host: string,
  status: R2MapCampaignStatus | null,
  node: ClusterNode | undefined,
): string {
  const hostStatus = status?.hosts[host];
  if (host === "john1" && status?.training.active && hostStatus?.intent === "train") {
    return "MLX training";
  }
  if (hostStatus?.intent === "benchmark") return "Benchmark";
  if (hostStatus?.intent === "generate") return "Self-play";
  if (hostStatus && hostStatus.intent !== "idle") return phaseLabel(hostStatus.intent);
  return node?.jobs.length ? "External workload" : "Available";
}

function isNodeMlxTraining(
  host: string,
  status: R2MapCampaignStatus | null,
  node: ClusterNode | undefined,
): boolean {
  if (
    status?.training.active &&
    host === "john1" &&
    status.hosts[host]?.intent === "train"
  ) {
    return true;
  }
  return Boolean(
    node?.jobs.some(({ command, workload }) =>
      /(?:mlx.*train|train.*mlx|cascadia[-_](?:v3_)?mlx)/i.test(`${workload} ${command}`),
    ),
  );
}

export default function ClusterFleetOverview({
  nodes,
  status,
}: {
  nodes: ClusterNode[];
  status: R2MapCampaignStatus | null;
}) {
  return (
    <section className="cluster-fleet-overview" aria-labelledby="cluster-fleet-title">
      <header>
        <strong id="cluster-fleet-title">Fleet status</strong>
        <span>CPU · memory · disk · MLX</span>
      </header>
      <div className="cluster-fleet-strip" aria-label="Cluster fleet hosts">
        {FLEET_HOSTS.map((host) => {
          const hostStatus = status?.hosts[host];
          const node = nodes.find(({ id }) => id === host);
          const intent = hostStatus?.intent ?? (node?.jobs.length ? "external" : "available");
          const mlxTraining = isNodeMlxTraining(host, status, node);
          return (
            <article
              data-health={node?.health ?? "offline"}
              data-intent={intent}
              data-mlx-training={mlxTraining}
              key={host}
              title={hostStatus?.detail ?? node?.jobs[0]?.workload ?? undefined}
            >
              <header>
                <span
                  className="cluster-fleet-health"
                  aria-label={node ? healthLabel(node.health) : "No telemetry"}
                />
                <strong>{host}</strong>
                <b>{hostIntentLabel(host, status, node)}</b>
              </header>
              <div className="cluster-fleet-metrics">
                <ResourceRing
                  host={host}
                  label="CPU"
                  value={node?.reachable ? node.cpu_percent : null}
                />
                <ResourceRing
                  host={host}
                  label="Memory"
                  value={node?.reachable ? node.memory_used_percent : null}
                />
                <ResourceRing
                  host={host}
                  label="Disk"
                  value={node?.reachable ? node.disk_used_percent : null}
                />
                <div className="cluster-fleet-mlx" data-active={mlxTraining}>
                  <strong>{mlxTraining ? "Yes" : "No"}</strong>
                  <span>MLX</span>
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
