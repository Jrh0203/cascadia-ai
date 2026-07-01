import {
  Activity,
  CheckCircle2,
  CircleGauge,
  Clock3,
  Cpu,
  MemoryStick,
  TriangleAlert,
} from "lucide-react";

import {
  formatBytes,
  formatDuration,
  formatSignedBytes,
  lossPolyline,
  r2MapConditionLabel,
} from "./cluster";
import type {
  ClusterNode,
  R2MapCampaignStatus,
  R2MapDistributionSummary,
  R2MapFocalStatus,
  R2MapStatusResponse,
} from "./types";

export type ClusterDetailView = "training" | "benchmark" | "fleet" | "research";

const DETAIL_VIEWS: ReadonlyArray<readonly [ClusterDetailView, string]> = [
  ["training", "Training"],
  ["benchmark", "Benchmark"],
  ["fleet", "Fleet"],
  ["research", "Research"],
];

const CAMPAIGN_HOSTS = ["john1", "john2", "john3"] as const;

function compactIdentity(id: string | undefined): string {
  if (!id) return "Not assigned";
  return id.length > 28 ? `${id.slice(0, 25)}…` : id;
}

function phaseLabel(phase: string): string {
  const normalized = phase.replaceAll(/[-_]/g, " ");
  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

function progress(completed: number, total: number | null): string {
  return total === null
    ? completed.toLocaleString()
    : `${completed.toLocaleString()} / ${total.toLocaleString()}`;
}

function progressPercent(
  completed: number,
  total: number | null,
): number | null {
  return total && total > 0 ? Math.min(100, (completed / total) * 100) : null;
}

function benchmarkStateLabel(status: R2MapCampaignStatus): string {
  if (
    !status.benchmark.active &&
    status.benchmark.stage?.includes("lagged-greedy-baseline")
  ) {
    return "Baseline complete";
  }
  if (status.benchmark.active) return "Running";
  return phaseLabel(status.benchmark.classification);
}

function DistributionCell({ value }: { value: R2MapDistributionSummary }) {
  return (
    <span className="r2map-distribution">
      <strong>{value.mean.toFixed(2)}</strong>
      <small>
        {value.p10.toFixed(1)} / {value.p50.toFixed(1)} / {value.p90.toFixed(1)}
      </small>
    </span>
  );
}

function AnatomyTable({ focal }: { focal: R2MapFocalStatus }) {
  const animals = [
    ["All wildlife", focal.animals.aggregate],
    ["Bear", focal.animals.bear],
    ["Elk", focal.animals.elk],
    ["Salmon", focal.animals.salmon],
    ["Hawk", focal.animals.hawk],
    ["Fox", focal.animals.fox],
  ] as const;
  const habitat = [
    ["All habitat", focal.habitat.aggregate],
    ["Mountain", focal.habitat.mountain],
    ["Forest", focal.habitat.forest],
    ["Prairie", focal.habitat.prairie],
    ["Wetland", focal.habitat.wetland],
    ["River", focal.habitat.river],
  ] as const;
  const pinecones = [
    ["Earned", focal.pinecones.earned],
    ["Independent", focal.pinecones.independent_draft_spend],
    ["Paid wipe", focal.pinecones.paid_wipe_spend],
    ["Total spent", focal.pinecones.total_spend],
    ["Remaining", focal.pinecones.remaining],
    ["Free replaces", focal.pinecones.free_replacements],
  ] as const;
  const groups: ReadonlyArray<
    readonly [
      string,
      ReadonlyArray<readonly [string, R2MapDistributionSummary]>,
    ]
  > = [
    ["Animals", animals],
    ["Habitat", habitat],
    ["Pinecones", pinecones],
  ];

  return (
    <div className="r2map-anatomy-grid">
      {groups.map(([title, rows]) => (
        <article className="r2map-anatomy" key={title}>
          <h4>{title}</h4>
          <div className="r2map-anatomy-labels">
            <span>Component</span>
            <span>Mean · P10 / P50 / P90</span>
          </div>
          {rows.map(([label, value]) => (
            <div className="r2map-anatomy-row" key={label}>
              <span>{label}</span>
              <DistributionCell value={value} />
            </div>
          ))}
        </article>
      ))}
    </div>
  );
}

function CampaignMetadata({ status }: { status: R2MapCampaignStatus }) {
  return (
    <details className="r2map-machine-details">
      <summary>Campaign metadata</summary>
      <div>
        <dl>
          <div>
            <dt>Incumbent</dt>
            <dd title={status.models.incumbent?.id}>
              {compactIdentity(status.models.incumbent?.id)}
            </dd>
          </div>
          <div>
            <dt>Candidate</dt>
            <dd title={status.models.candidate?.id}>
              {compactIdentity(status.models.candidate?.id)}
            </dd>
          </div>
          <div>
            <dt>Opponent pool</dt>
            <dd>{status.models.opponent_pool.length} frozen policies</dd>
          </div>
          <div>
            <dt>Round</dt>
            <dd>{status.round_index ?? "—"}</dd>
          </div>
        </dl>
        <ul>
          {CAMPAIGN_HOSTS.map((host) => {
            const detail = status.hosts[host]?.detail;
            return detail ? (
              <li key={host}>
                <strong>{host}</strong>
                <code>{detail}</code>
              </li>
            ) : null;
          })}
        </ul>
      </div>
    </details>
  );
}

export default function R2MapCampaignPanel({
  response,
  requestError,
  nodes = [],
  activeView = "training",
  onViewChange = () => undefined,
}: {
  response: R2MapStatusResponse | null;
  requestError: string | null;
  nodes?: ClusterNode[];
  activeView?: ClusterDetailView;
  onViewChange?: (view: ClusterDetailView) => void;
}) {
  const condition = response?.condition ?? "unconfigured";
  const status = response?.status ?? null;
  const isV3 = status?.schema_id === "cascadia.v3.dashboard-status.v1";
  const campaignLabel = isV3
    ? "V3 NNUE · Radius-7 campaign"
    : "R2-MAP · Expert iteration";
  const lossSamples = status?.training.loss_samples ?? [];
  const latestLoss = lossSamples.at(-1);
  const trainPath = lossPolyline(lossSamples, "train_total", 640, 128);
  const validationPath = lossPolyline(
    lossSamples,
    "validation_total",
    640,
    128,
  );
  const john1 = status?.hosts.john1 ?? null;
  const isMlxTraining =
    status?.training.active === true && john1?.intent === "train";
  const generationHosts = status
    ? CAMPAIGN_HOSTS.filter((host) => status.hosts[host]?.intent === "generate")
    : [];
  const generationCompleted = generationHosts.reduce(
    (sum, host) => sum + (status?.hosts[host]?.generation_games_completed ?? 0),
    0,
  );
  const generationTarget = generationHosts.reduce<number | null>(
    (sum, host) => {
      const target = status?.hosts[host]?.generation_games_target;
      return sum === null || target === null || target === undefined
        ? null
        : sum + target;
    },
    0,
  );
  const trainingPercent = status
    ? progressPercent(
        status.training.current_step ?? 0,
        status.training.total_steps,
      )
    : null;
  const benchmarkPercent = status
    ? progressPercent(
        status.benchmark.pairs_completed,
        status.benchmark.pairs_total,
      )
    : null;
  const schedulerItems = status?.benchmark.scheduler_work_items ?? null;
  const isDistributedWork =
    schedulerItems !== null &&
    status?.training.active !== true &&
    status?.benchmark.active !== true;
  const schedulerPercent = schedulerItems
    ? progressPercent(schedulerItems.completed, schedulerItems.total)
    : null;
  const schedulerUtilization = schedulerItems?.utilization ?? null;
  const schedulerStateLabel = schedulerItems
    ? Object.entries(schedulerItems.states)
        .filter(([, count]) => count > 0)
        .map(([state, count]) => `${state} ${count}`)
        .join(" · ") || "Awaiting submission"
    : "—";
  const generationPercent = progressPercent(
    generationCompleted,
    generationTarget,
  );
  const focusPercent = isMlxTraining
    ? trainingPercent
    : status?.benchmark.active
      ? benchmarkPercent
      : isDistributedWork
        ? schedulerPercent
        : generationHosts.length > 0
          ? generationPercent
          : null;
  const focusProgressLabel = isMlxTraining
    ? "MLX training progress"
    : status?.benchmark.active
      ? "Benchmark progress"
      : isDistributedWork
        ? "Distributed work progress"
        : "Self-play generation progress";
  const focusTitle = isMlxTraining
    ? "MLX training on John1"
    : status?.benchmark.active
      ? "Benchmarking the previous checkpoint"
      : isDistributedWork
        ? status
          ? phaseLabel(status.phase)
          : "Distributed campaign work"
        : generationHosts.length > 0
          ? "Generating self-play games"
          : status
            ? phaseLabel(status.phase)
            : "Waiting for campaign status";
  const focusKicker = isMlxTraining
    ? "MLX training"
    : status?.benchmark.active
      ? "Paired benchmark"
      : isDistributedWork
        ? "Distributed compute"
        : generationHosts.length > 0
          ? "Self-play generation"
          : "Campaign state";
  const focusMetrics = status
    ? isMlxTraining
      ? [
          [
            "Step",
            progress(
              status.training.current_step ?? 0,
              status.training.total_steps,
            ),
          ],
          ["Loss", latestLoss?.train_total.toFixed(4) ?? "—"],
          [
            "Checkpoint",
            compactIdentity(status.training.latest_verified_checkpoint?.id),
          ],
          [
            "ETA",
            john1?.eta_seconds === null
              ? "—"
              : formatDuration(john1?.eta_seconds ?? 0),
          ],
          [
            "Rate",
            `${status.training.examples_per_second?.toFixed(0) ?? "—"} ex/s`,
          ],
        ]
      : status.benchmark.active
        ? [
            [
              "Pairs",
              progress(
                status.benchmark.pairs_completed,
                status.benchmark.pairs_total,
              ),
            ],
            ["Mean", status.benchmark.focal?.base_total.mean.toFixed(3) ?? "—"],
            [
              "Delta",
              status.benchmark.paired_delta
                ? `${status.benchmark.paired_delta.mean >= 0 ? "+" : ""}${status.benchmark.paired_delta.mean.toFixed(3)}`
                : "—",
            ],
            ["ETA", formatDuration(status.benchmark.eta_seconds ?? 0)],
            [
              "Rate",
              `${status.benchmark.throughput_games_per_second?.toFixed(2) ?? "—"} g/s`,
            ],
          ]
        : isDistributedWork && schedulerItems
          ? [
              [
                "Work items",
                progress(schedulerItems.completed, schedulerItems.total),
              ],
              ["Scheduler", schedulerStateLabel],
              ["Retries", String(schedulerItems.retry_attempts)],
              ["ETA", formatDuration(john1?.eta_seconds ?? 0)],
            ]
          : generationHosts.length > 0
            ? [
                ["Games", progress(generationCompleted, generationTarget)],
                ["Workers", generationHosts.join(" + ")],
                ["Round", String(status.round_index ?? "—")],
                ["Incumbent", compactIdentity(status.models.incumbent?.id)],
                ["Candidate", compactIdentity(status.models.candidate?.id)],
              ]
            : [
                ["Phase", phaseLabel(status.phase)],
                ["Round", String(status.round_index ?? "—")],
                ["Incumbent", compactIdentity(status.models.incumbent?.id)],
                ["Candidate", compactIdentity(status.models.candidate?.id)],
                ["Updated", `${Math.round(response?.age_seconds ?? 0)}s ago`],
              ]
    : [];
  const nextAction = isMlxTraining
    ? "Let MLX finish; do not benchmark or generate from the in-flight candidate."
    : status?.benchmark.active
      ? "Apply the promotion gate when paired evidence completes."
      : isDistributedWork
        ? "Let the scheduler finish this phase; the next barrier starts automatically."
        : generationHosts.length > 0
          ? "Train the next checkpoint on John1 when all shards return."
          : status?.legal_next_transitions.length
            ? status.legal_next_transitions.map(phaseLabel).join(" · ")
            : "Review the terminal campaign result.";
  const unhealthyNodes = nodes.filter(
    ({ health }) => health === "warning" || health === "offline",
  );
  const alertLabel =
    condition !== "fresh"
      ? r2MapConditionLabel(condition)
      : unhealthyNodes.length > 0
        ? `${unhealthyNodes.map(({ label }) => label).join(", ")} need attention`
        : nodes.length === 0
          ? "Fleet telemetry loading"
          : "No active alerts";
  const conditionMessage = requestError
    ? `Status request failed: ${requestError}`
    : condition === "unconfigured"
      ? "No campaign status mirror is configured yet. The API is not scanning campaign artifacts."
      : condition === "invalid"
        ? `The status mirror is invalid: ${response?.error ?? "unknown validation failure"}`
        : condition === "stale"
          ? `The last atomic update is ${Math.round(response?.age_seconds ?? 0)} seconds old.`
          : null;

  return (
    <section className="r2map-section" aria-labelledby="r2map-title">
      <header className="r2map-command-heading">
        <div>
          <span>{campaignLabel}</span>
          <h2 id="r2map-title">{focusTitle}</h2>
        </div>
        <span
          className="r2map-condition"
          data-condition={requestError ? "invalid" : condition}
        >
          {r2MapConditionLabel(requestError ? "invalid" : condition)}
        </span>
      </header>

      {conditionMessage && (
        <div
          className="r2map-state-message"
          data-condition={condition}
          role="status"
        >
          <TriangleAlert aria-hidden="true" />
          <span>{conditionMessage}</span>
        </div>
      )}

      {status && (
        <>
          <div className="r2map-command-deck" role="status" aria-live="polite">
            <div className="r2map-command-focus">
              <span className="r2map-live-pulse" aria-hidden="true">
                {isMlxTraining ? <Activity /> : <CircleGauge />}
              </span>
              <div>
                <span>{focusKicker}</span>
                <strong>{phaseLabel(status.phase)}</strong>
                <small>Round {status.round_index ?? "—"}</small>
              </div>
            </div>

            <div className="r2map-command-metrics">
              {focusMetrics.map(([label, value]) => (
                <div key={label}>
                  <span>{label}</span>
                  <strong title={value}>{value}</strong>
                </div>
              ))}
            </div>

            {focusPercent !== null && (
              <div className="r2map-command-progress">
                <div
                  aria-label={focusProgressLabel}
                  aria-valuemax={100}
                  aria-valuemin={0}
                  aria-valuenow={focusPercent}
                  role="progressbar"
                >
                  <i style={{ width: `${focusPercent}%` }} />
                </div>
                <strong>{focusPercent.toFixed(1)}%</strong>
              </div>
            )}
          </div>
        </>
      )}

      {status && (
        <>
          <div className="r2map-next-action">
            <span>Next</span>
            <strong>{nextAction}</strong>
            <small
              data-alert={unhealthyNodes.length > 0 || condition !== "fresh"}
            >
              {unhealthyNodes.length > 0 || condition !== "fresh" ? (
                <TriangleAlert aria-hidden="true" />
              ) : (
                <CheckCircle2 aria-hidden="true" />
              )}
              {alertLabel}
            </small>
          </div>

          <div
            className="cluster-detail-tabs"
            role="tablist"
            aria-label="Cluster detail"
          >
            {DETAIL_VIEWS.map(([view, label]) => (
              <button
                aria-controls={`cluster-panel-${view}`}
                aria-selected={activeView === view}
                className={activeView === view ? "is-active" : ""}
                id={`cluster-tab-${view}`}
                key={view}
                onClick={() => onViewChange(view)}
                role="tab"
                type="button"
              >
                {label}
              </button>
            ))}
          </div>

          {activeView === "training" && (
            <div
              aria-labelledby="cluster-tab-training"
              className="r2map-detail-panel"
              id="cluster-panel-training"
              role="tabpanel"
            >
              <article className="r2map-training-detail">
                <header>
                  <div>
                    <Activity aria-hidden="true" />
                    <span>Training loss</span>
                  </div>
                  <strong>{status.training.active ? "Live" : "Idle"}</strong>
                </header>
                <div
                  className="r2map-loss-chart"
                  aria-label="Training and validation loss"
                >
                  {lossSamples.length > 0 ? (
                    <svg role="img" viewBox="0 0 640 128">
                      <path className="train" d={trainPath} />
                      {validationPath && (
                        <path className="validation" d={validationPath} />
                      )}
                    </svg>
                  ) : (
                    <span>No loss samples published yet</span>
                  )}
                  <footer>
                    <span>
                      <i className="train" /> Train
                    </span>
                    <span>
                      <i className="validation" /> Validation
                    </span>
                    <small>{lossSamples.length} retained samples</small>
                  </footer>
                </div>
                <div className="r2map-training-support">
                  <div>
                    <span>Verified checkpoint</span>
                    <strong
                      title={status.training.latest_verified_checkpoint?.id}
                    >
                      {compactIdentity(
                        status.training.latest_verified_checkpoint?.id,
                      )}
                    </strong>
                  </div>
                  <div>
                    <span>John1 RSS</span>
                    <strong>
                      {john1?.rss_bytes === null
                        ? "—"
                        : formatBytes(john1?.rss_bytes ?? 0)}
                    </strong>
                  </div>
                  <div>
                    <span>Swap delta</span>
                    <strong>
                      {john1?.swap_delta_bytes === null
                        ? "—"
                        : formatSignedBytes(john1?.swap_delta_bytes ?? 0)}
                    </strong>
                  </div>
                </div>
              </article>
              <CampaignMetadata status={status} />
            </div>
          )}

          {activeView === "benchmark" && (
            <div
              aria-labelledby="cluster-tab-benchmark"
              className="r2map-detail-panel"
              id="cluster-panel-benchmark"
              role="tabpanel"
            >
              <article className="r2map-benchmark-detail">
                <header>
                  <div>
                    <CircleGauge aria-hidden="true" />
                    <span>
                      {status.benchmark.stage
                        ? phaseLabel(status.benchmark.stage)
                        : "Focal benchmark"}
                    </span>
                  </div>
                  <strong data-classification={status.benchmark.classification}>
                    {benchmarkStateLabel(status)}
                  </strong>
                </header>
                <div className="r2map-benchmark-summary">
                  <div>
                    <span>Pairs</span>
                    <strong>
                      {progress(
                        status.benchmark.pairs_completed,
                        status.benchmark.pairs_total,
                      )}
                    </strong>
                  </div>
                  <div>
                    <Clock3 aria-hidden="true" />
                    <span>ETA</span>
                    <strong>
                      {status.benchmark.active
                        ? formatDuration(status.benchmark.eta_seconds ?? 0)
                        : "Complete"}
                    </strong>
                  </div>
                  <div>
                    <Cpu aria-hidden="true" />
                    <span>Throughput</span>
                    <strong>
                      {status.benchmark.throughput_games_per_second?.toFixed(
                        2,
                      ) ?? "—"}{" "}
                      g/s
                    </strong>
                  </div>
                  <div>
                    <MemoryStick aria-hidden="true" />
                    <span>Peak RSS / swap</span>
                    <strong>
                      {status.benchmark.peak_rss_bytes === null
                        ? "—"
                        : formatBytes(status.benchmark.peak_rss_bytes)}{" "}
                      /{" "}
                      {status.benchmark.swap_delta_bytes === null
                        ? "—"
                        : formatSignedBytes(status.benchmark.swap_delta_bytes)}
                    </strong>
                  </div>
                  <div>
                    <Activity aria-hidden="true" />
                    <span>Scheduler jobs</span>
                    <strong>{schedulerStateLabel}</strong>
                  </div>
                  <div>
                    <span>Retries</span>
                    <strong>{schedulerItems?.retry_attempts ?? "—"}</strong>
                  </div>
                  <div>
                    <CircleGauge aria-hidden="true" />
                    <span>Campaign CPU mean / peak</span>
                    <strong>
                      {schedulerUtilization
                        ? `${(schedulerUtilization.cpu_utilization_mean * 100).toFixed(1)}% / ${(schedulerUtilization.cpu_utilization_peak * 100).toFixed(1)}%`
                        : "—"}
                    </strong>
                  </div>
                </div>
                {status.benchmark.focal ? (
                  <div className="r2map-focal-score">
                    <div>
                      <span>Focal mean</span>
                      <strong>
                        {status.benchmark.focal.base_total.mean.toFixed(3)}
                      </strong>
                      <small>
                        P10 / P50 / P90{" "}
                        {status.benchmark.focal.base_total.p10.toFixed(1)} /{" "}
                        {status.benchmark.focal.base_total.p50.toFixed(1)} /{" "}
                        {status.benchmark.focal.base_total.p90.toFixed(1)}
                      </small>
                    </div>
                    <div>
                      <span>Paired delta</span>
                      <strong>
                        {status.benchmark.paired_delta
                          ? `${status.benchmark.paired_delta.mean >= 0 ? "+" : ""}${status.benchmark.paired_delta.mean.toFixed(3)}`
                          : "—"}
                      </strong>
                      <small>
                        {status.benchmark.paired_delta
                          ? `95% CI [${status.benchmark.paired_delta.confidence_95[0].toFixed(3)}, ${status.benchmark.paired_delta.confidence_95[1].toFixed(3)}]`
                          : "Waiting for paired analysis"}
                      </small>
                    </div>
                  </div>
                ) : (
                  <div className="r2map-empty-benchmark">
                    No focal score has been opened.
                  </div>
                )}
              </article>
              {status.benchmark.focal && (
                <AnatomyTable focal={status.benchmark.focal} />
              )}
              <CampaignMetadata status={status} />
            </div>
          )}
        </>
      )}
    </section>
  );
}
