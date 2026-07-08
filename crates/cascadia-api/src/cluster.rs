use std::{
    collections::BTreeMap,
    io::Write,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde::Serialize;

const CLUSTER_SCHEMA_VERSION: u16 = 1;

const METRICS_SCRIPT: &str = r#"
emit() {
  printf 'metric\t%s\t%s\n' "$1" "$2"
}

hostname_value="$(scutil --get ComputerName 2>/dev/null || hostname)"
cores="$(sysctl -n hw.ncpu 2>/dev/null || printf '0')"
memory_total="$(sysctl -n hw.memsize 2>/dev/null || printf '0')"
cpu_sum="$(ps -A -o %cpu= 2>/dev/null | awk '{ total += $1 } END { printf "%.2f", total }')"
memory_free_pct="$(memory_pressure -Q 2>/dev/null | awk '/free percentage/ { gsub(/%/, "", $NF); print $NF; exit }')"
load_average="$(sysctl -n vm.loadavg 2>/dev/null | tr -d '{}')"
boot_epoch="$(sysctl -n kern.boottime 2>/dev/null | sed -E 's/^\{ sec = ([0-9]+),.*/\1/')"
now_epoch="$(date +%s)"
uptime_seconds="$((now_epoch - boot_epoch))"
os_version="$(sw_vers -productVersion 2>/dev/null || printf 'unknown')"
sleep_minutes="$(pmset -g custom 2>/dev/null | awk '$1 == "sleep" { print $2; exit }')"
auto_restart="$(pmset -g custom 2>/dev/null | awk '$1 == "autorestart" { print $2; exit }')"
power_source="$(pmset -g batt 2>/dev/null | sed -n "s/Now drawing from '\(.*\)'/\1/p")"
repo="$HOME/cascadia"

emit hostname "$hostname_value"
emit cores "$cores"
emit memory_total_bytes "$memory_total"
emit cpu_sum_percent "$cpu_sum"
emit memory_free_percent "${memory_free_pct:-0}"
emit load_average "$load_average"
emit uptime_seconds "$uptime_seconds"
emit os_version "$os_version"
emit sleep_minutes "${sleep_minutes:-unknown}"
emit auto_restart "${auto_restart:-unknown}"
emit power_source "${power_source:-unknown}"

df -k /System/Volumes/Data 2>/dev/null | awk '
  NR == 2 {
    printf "metric\tdisk_total_bytes\t%.0f\n", $2 * 1024
    printf "metric\tdisk_used_bytes\t%.0f\n", $3 * 1024
    printf "metric\tdisk_available_bytes\t%.0f\n", $4 * 1024
  }
'

if [ -d "$repo/.git" ]; then
  emit repo_present true
  emit repo_branch "$(git -C "$repo" branch --show-current 2>/dev/null || printf 'unknown')"
  emit repo_revision "$(git -C "$repo" rev-parse --short HEAD 2>/dev/null || printf 'unknown')"
  emit repo_changes "$(git -C "$repo" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
else
  emit repo_present false
fi

if [ -x "$repo/target/release/cascadia-v2" ] || [ -x "$repo/target/release/cascadia-cli" ]; then
  emit release_binary_present true
else
  emit release_binary_present false
fi

if command -v mlx_lm >/dev/null 2>&1 || [ -x "$repo/.venv/bin/python" ]; then
  emit mlx_runtime_present true
else
  emit mlx_runtime_present false
fi

ps -axo pid=,etime=,%cpu=,%mem=,comm=,args= 2>/dev/null | awk '
  $5 !~ /(^|\/)awk$/ &&
  (index($0, "target/release/cascadia-v2") ||
   index($0, "target/release/cascadia-cli") ||
   index($0, "cascadia-mlx") ||
   index($0, "cascadia_mlx")) {
    printf "job\t%s %s %s %s", $1, $2, $3, $4
    for (field = 6; field <= NF; field++) {
      printf " %s", $field
    }
    printf "\n"
  }
'
"#;

#[derive(Debug, Clone, Copy)]
struct NodeSpec {
    id: &'static str,
    label: &'static str,
    role: &'static str,
    address: &'static str,
    ssh_host: Option<&'static str>,
}

const NODES: [NodeSpec; 3] = [
    NodeSpec {
        id: "john1",
        label: "John 1",
        role: "Coordinator / research",
        address: "100.110.109.6",
        ssh_host: None,
    },
    NodeSpec {
        id: "john2",
        label: "John 2",
        role: "Simulation worker",
        address: "100.100.43.38",
        ssh_host: Some("john2"),
    },
    NodeSpec {
        id: "john3",
        label: "John 3",
        role: "Simulation worker",
        address: "100.71.97.55",
        ssh_host: Some("john3"),
    },
];

pub(crate) fn node_id_labels() -> [(&'static str, &'static str); 3] {
    NODES.map(|node| (node.id, node.label))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum NodeHealth {
    Healthy,
    Busy,
    Warning,
    Offline,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClusterResponse {
    pub schema_version: u16,
    pub collected_at_unix_ms: u128,
    pub collection_duration_ms: u128,
    pub summary: ClusterSummary,
    pub nodes: Vec<ClusterNode>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClusterSummary {
    pub total_nodes: usize,
    pub online_nodes: usize,
    pub busy_nodes: usize,
    pub degraded_nodes: usize,
    pub total_cores: u32,
    pub active_jobs: usize,
    pub average_cpu_percent: f64,
    pub memory_used_bytes: u64,
    pub memory_total_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClusterNode {
    pub id: &'static str,
    pub label: &'static str,
    pub role: &'static str,
    pub address: &'static str,
    pub health: NodeHealth,
    pub reachable: bool,
    pub sample_latency_ms: u128,
    pub error: Option<String>,
    pub hostname: Option<String>,
    pub os_version: Option<String>,
    pub cores: u32,
    pub cpu_percent: f64,
    pub load_average: [f64; 3],
    pub memory_used_bytes: u64,
    pub memory_total_bytes: u64,
    pub memory_used_percent: f64,
    pub disk_used_bytes: u64,
    pub disk_total_bytes: u64,
    pub disk_available_bytes: u64,
    pub disk_used_percent: f64,
    pub uptime_seconds: u64,
    pub power: NodePower,
    pub readiness: NodeReadiness,
    pub jobs: Vec<ClusterJob>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodePower {
    pub source: Option<String>,
    pub system_sleep_minutes: Option<u32>,
    pub auto_restart: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeReadiness {
    pub repository_present: bool,
    pub release_binary_present: bool,
    pub mlx_runtime_present: bool,
    pub branch: Option<String>,
    pub revision: Option<String>,
    pub uncommitted_changes: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClusterJob {
    pub pid: u32,
    pub elapsed: String,
    pub cpu_percent: f64,
    pub memory_percent: f64,
    pub workload: &'static str,
    pub command: String,
}

pub fn collect_cluster() -> ClusterResponse {
    let started = Instant::now();
    let collected_at_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();

    let nodes = thread::scope(|scope| {
        let handles: Vec<_> = NODES
            .iter()
            .map(|spec| scope.spawn(move || collect_node(*spec)))
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .expect("cluster metrics worker should not panic")
            })
            .collect::<Vec<_>>()
    });

    let online: Vec<_> = nodes.iter().filter(|node| node.reachable).collect();
    let total_cores = online.iter().map(|node| node.cores).sum();
    let active_jobs = online.iter().map(|node| node.jobs.len()).sum();
    let average_cpu_percent = if online.is_empty() {
        0.0
    } else {
        online.iter().map(|node| node.cpu_percent).sum::<f64>() / online.len() as f64
    };

    let summary = ClusterSummary {
        total_nodes: nodes.len(),
        online_nodes: online.len(),
        busy_nodes: nodes
            .iter()
            .filter(|node| node.health == NodeHealth::Busy)
            .count(),
        degraded_nodes: nodes
            .iter()
            .filter(|node| matches!(node.health, NodeHealth::Warning | NodeHealth::Offline))
            .count(),
        total_cores,
        active_jobs,
        average_cpu_percent: round_one(average_cpu_percent),
        memory_used_bytes: online.iter().map(|node| node.memory_used_bytes).sum(),
        memory_total_bytes: online.iter().map(|node| node.memory_total_bytes).sum(),
    };

    ClusterResponse {
        schema_version: CLUSTER_SCHEMA_VERSION,
        collected_at_unix_ms,
        collection_duration_ms: started.elapsed().as_millis(),
        summary,
        nodes,
    }
}

fn collect_node(spec: NodeSpec) -> ClusterNode {
    let started = Instant::now();
    match run_metrics_command(spec) {
        Ok(output) => parse_node(spec, started.elapsed(), &output),
        Err(error) => offline_node(spec, started.elapsed(), error),
    }
}

fn run_metrics_command(spec: NodeSpec) -> Result<String, String> {
    let mut command = if let Some(host) = spec.ssh_host {
        let mut command = Command::new("/usr/bin/ssh");
        command.args([
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=3",
            "-o",
            "ConnectionAttempts=1",
            "-o",
            "ServerAliveInterval=2",
            "-o",
            "ServerAliveCountMax=1",
            host,
            "/bin/zsh",
            "-s",
        ]);
        command
    } else {
        let mut command = Command::new("/bin/zsh");
        command.arg("-s");
        command
    };

    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("could not start metrics probe: {error}"))?;
    child
        .stdin
        .take()
        .ok_or_else(|| "metrics probe stdin was unavailable".to_string())?
        .write_all(METRICS_SCRIPT.as_bytes())
        .map_err(|error| format!("could not send metrics probe: {error}"))?;

    let output = child
        .wait_with_output()
        .map_err(|error| format!("metrics probe failed: {error}"))?;
    if output.status.success() {
        String::from_utf8(output.stdout)
            .map_err(|error| format!("metrics probe returned invalid UTF-8: {error}"))
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let detail = stderr
            .lines()
            .rev()
            .find(|line| !line.trim().is_empty())
            .unwrap_or("SSH probe exited unsuccessfully");
        Err(truncate(detail.trim(), 180))
    }
}

fn parse_node(spec: NodeSpec, latency: Duration, output: &str) -> ClusterNode {
    let mut metrics = BTreeMap::new();
    let mut jobs = Vec::new();

    for line in output.lines() {
        if let Some(rest) = line.strip_prefix("metric\t") {
            if let Some((key, value)) = rest.split_once('\t') {
                metrics.insert(key, value.trim().to_string());
            }
        } else if let Some(job) = line.strip_prefix("job\t").and_then(parse_job) {
            jobs.push(job);
        }
    }

    let cores = parse_metric(&metrics, "cores");
    let cpu_sum_percent: f64 = parse_metric(&metrics, "cpu_sum_percent");
    let cpu_percent = if cores == 0 {
        0.0
    } else {
        (cpu_sum_percent / f64::from(cores)).clamp(0.0, 100.0)
    };
    let memory_total_bytes = parse_metric(&metrics, "memory_total_bytes");
    let memory_free_percent: f64 =
        parse_metric::<f64>(&metrics, "memory_free_percent").clamp(0.0, 100.0);
    let memory_used_percent = 100.0 - memory_free_percent;
    let memory_used_bytes =
        ((memory_used_percent / 100.0) * memory_total_bytes as f64).round() as u64;
    let disk_total_bytes = parse_metric(&metrics, "disk_total_bytes");
    let disk_used_bytes = parse_metric(&metrics, "disk_used_bytes");
    let disk_used_percent = if disk_total_bytes == 0 {
        0.0
    } else {
        disk_used_bytes as f64 / disk_total_bytes as f64 * 100.0
    };
    let load_average = parse_load(metrics.get("load_average").map(String::as_str));

    let power = NodePower {
        source: optional_metric(&metrics, "power_source"),
        system_sleep_minutes: optional_parse_metric(&metrics, "sleep_minutes"),
        auto_restart: optional_parse_metric::<u8>(&metrics, "auto_restart").map(|value| value == 1),
    };
    let readiness = NodeReadiness {
        repository_present: bool_metric(&metrics, "repo_present"),
        release_binary_present: bool_metric(&metrics, "release_binary_present"),
        mlx_runtime_present: bool_metric(&metrics, "mlx_runtime_present"),
        branch: optional_metric(&metrics, "repo_branch"),
        revision: optional_metric(&metrics, "repo_revision"),
        uncommitted_changes: optional_parse_metric(&metrics, "repo_changes"),
    };
    let health = classify_health(
        cpu_percent,
        memory_used_percent,
        disk_used_percent,
        load_average[0],
        cores,
        &power,
        !jobs.is_empty(),
    );

    ClusterNode {
        id: spec.id,
        label: spec.label,
        role: spec.role,
        address: spec.address,
        health,
        reachable: true,
        sample_latency_ms: latency.as_millis(),
        error: None,
        hostname: optional_metric(&metrics, "hostname"),
        os_version: optional_metric(&metrics, "os_version"),
        cores,
        cpu_percent: round_one(cpu_percent),
        load_average,
        memory_used_bytes,
        memory_total_bytes,
        memory_used_percent: round_one(memory_used_percent),
        disk_used_bytes,
        disk_total_bytes,
        disk_available_bytes: parse_metric(&metrics, "disk_available_bytes"),
        disk_used_percent: round_one(disk_used_percent),
        uptime_seconds: parse_metric(&metrics, "uptime_seconds"),
        power,
        readiness,
        jobs,
    }
}

fn offline_node(spec: NodeSpec, latency: Duration, error: String) -> ClusterNode {
    ClusterNode {
        id: spec.id,
        label: spec.label,
        role: spec.role,
        address: spec.address,
        health: NodeHealth::Offline,
        reachable: false,
        sample_latency_ms: latency.as_millis(),
        error: Some(error),
        hostname: None,
        os_version: None,
        cores: 0,
        cpu_percent: 0.0,
        load_average: [0.0; 3],
        memory_used_bytes: 0,
        memory_total_bytes: 0,
        memory_used_percent: 0.0,
        disk_used_bytes: 0,
        disk_total_bytes: 0,
        disk_available_bytes: 0,
        disk_used_percent: 0.0,
        uptime_seconds: 0,
        power: NodePower {
            source: None,
            system_sleep_minutes: None,
            auto_restart: None,
        },
        readiness: NodeReadiness {
            repository_present: false,
            release_binary_present: false,
            mlx_runtime_present: false,
            branch: None,
            revision: None,
            uncommitted_changes: None,
        },
        jobs: Vec::new(),
    }
}

fn parse_job(line: &str) -> Option<ClusterJob> {
    let mut fields = line.split_whitespace();
    let pid = fields.next()?.parse().ok()?;
    let elapsed = fields.next()?.to_string();
    let cpu_percent = fields.next()?.parse().ok()?;
    let memory_percent = fields.next()?.parse().ok()?;
    let command = fields.collect::<Vec<_>>().join(" ");
    if command.is_empty() {
        return None;
    }
    let workload = if command.contains("collect-counterfactual") {
        "Counterfactual collection"
    } else if command.contains("cascadia-mlx") {
        "MLX training"
    } else if command.contains("cascadia-cli") {
        "Legacy simulation"
    } else {
        "Cascadia simulation"
    };
    Some(ClusterJob {
        pid,
        elapsed,
        cpu_percent,
        memory_percent,
        workload,
        command: truncate(&command, 240),
    })
}

fn classify_health(
    cpu_percent: f64,
    memory_percent: f64,
    disk_percent: f64,
    load_one: f64,
    cores: u32,
    power: &NodePower,
    has_jobs: bool,
) -> NodeHealth {
    let power_warning = power
        .system_sleep_minutes
        .is_some_and(|minutes| minutes != 0)
        || power.auto_restart == Some(false);
    let resource_warning = memory_percent >= 90.0
        || disk_percent >= 90.0
        || (!has_jobs && cores > 0 && load_one > f64::from(cores) * 1.5);
    if power_warning || resource_warning {
        NodeHealth::Warning
    } else if has_jobs || cpu_percent >= 70.0 || (cores > 0 && load_one > f64::from(cores)) {
        NodeHealth::Busy
    } else {
        NodeHealth::Healthy
    }
}

fn parse_load(value: Option<&str>) -> [f64; 3] {
    let mut values = value
        .unwrap_or_default()
        .split_whitespace()
        .filter_map(|part| part.parse::<f64>().ok());
    [
        round_one(values.next().unwrap_or_default()),
        round_one(values.next().unwrap_or_default()),
        round_one(values.next().unwrap_or_default()),
    ]
}

fn parse_metric<T>(metrics: &BTreeMap<&str, String>, key: &str) -> T
where
    T: Default + std::str::FromStr,
{
    optional_parse_metric(metrics, key).unwrap_or_default()
}

fn optional_parse_metric<T>(metrics: &BTreeMap<&str, String>, key: &str) -> Option<T>
where
    T: std::str::FromStr,
{
    metrics.get(key)?.parse().ok()
}

fn optional_metric(metrics: &BTreeMap<&str, String>, key: &str) -> Option<String> {
    metrics
        .get(key)
        .filter(|value| !value.is_empty() && value.as_str() != "unknown")
        .cloned()
}

fn bool_metric(metrics: &BTreeMap<&str, String>, key: &str) -> bool {
    metrics.get(key).is_some_and(|value| value == "true")
}

fn round_one(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

fn truncate(value: &str, max_chars: usize) -> String {
    let mut chars = value.chars();
    let head: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{head}...")
    } else {
        head
    }
}

#[cfg(test)]
pub(crate) fn test_cluster_response(
    timestamp_unix_ms: u128,
    cpu_percent: f64,
    memory_percent: f64,
) -> ClusterResponse {
    let nodes = NODES
        .into_iter()
        .map(|spec| {
            let mut node = offline_node(spec, Duration::ZERO, "test fixture".to_string());
            node.reachable = true;
            node.health = NodeHealth::Healthy;
            node.error = None;
            node.cores = 10;
            node.cpu_percent = cpu_percent;
            node.memory_total_bytes = 16 * 1024 * 1024 * 1024;
            node.memory_used_percent = memory_percent;
            node.memory_used_bytes =
                (node.memory_total_bytes as f64 * memory_percent / 100.0).round() as u64;
            node
        })
        .collect();
    ClusterResponse {
        schema_version: CLUSTER_SCHEMA_VERSION,
        collected_at_unix_ms: timestamp_unix_ms,
        collection_duration_ms: 0,
        summary: ClusterSummary {
            total_nodes: 3,
            online_nodes: 3,
            busy_nodes: 0,
            degraded_nodes: 0,
            total_cores: 30,
            active_jobs: 0,
            average_cpu_percent: cpu_percent,
            memory_used_bytes: 0,
            memory_total_bytes: 0,
        },
        nodes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
metric\thostname\tJohns-Mac-mini
metric\tcores\t10
metric\tmemory_total_bytes\t17179869184
metric\tcpu_sum_percent\t850.0
metric\tmemory_free_percent\t42
metric\tload_average\t8.50 7.25 6.10
metric\tuptime_seconds\t86400
metric\tos_version\t26.4
metric\tsleep_minutes\t0
metric\tauto_restart\t1
metric\tpower_source\tAC Power
metric\tdisk_total_bytes\t494384795648
metric\tdisk_used_bytes\t100000000000
metric\tdisk_available_bytes\t394384795648
metric\trepo_present\ttrue
metric\trepo_branch\tcodex/cascadia-v2
metric\trepo_revision\tabc1234
metric\trepo_changes\t2
metric\trelease_binary_present\ttrue
metric\tmlx_runtime_present\ttrue
job\t 73935 01:12:13 842.0 12.5 /tmp/cascadia-v2 collect-counterfactual --games 128
";

    #[test]
    fn parses_and_normalizes_node_metrics() {
        let node = parse_node(NODES[0], Duration::from_millis(17), SAMPLE);
        assert_eq!(node.hostname.as_deref(), Some("Johns-Mac-mini"));
        assert_eq!(node.cores, 10);
        assert_eq!(node.cpu_percent, 85.0);
        assert_eq!(node.memory_used_percent, 58.0);
        assert_eq!(node.load_average, [8.5, 7.3, 6.1]);
        assert_eq!(node.sample_latency_ms, 17);
        assert_eq!(node.health, NodeHealth::Busy);
        assert_eq!(node.jobs.len(), 1);
        assert_eq!(node.jobs[0].workload, "Counterfactual collection");
        assert!(node.readiness.repository_present);
        assert_eq!(node.readiness.branch.as_deref(), Some("codex/cascadia-v2"));
    }

    #[test]
    fn power_and_capacity_problems_are_warnings() {
        let power = NodePower {
            source: Some("AC Power".to_string()),
            system_sleep_minutes: Some(15),
            auto_restart: Some(true),
        };
        assert_eq!(
            classify_health(10.0, 20.0, 30.0, 1.0, 10, &power, false),
            NodeHealth::Warning
        );
        assert_eq!(
            classify_health(
                10.0,
                95.0,
                30.0,
                1.0,
                10,
                &NodePower {
                    system_sleep_minutes: Some(0),
                    ..power
                },
                false
            ),
            NodeHealth::Warning
        );
    }

    #[test]
    fn unreachable_node_retains_identity() {
        let node = offline_node(
            NODES[2],
            Duration::from_secs(3),
            "operation timed out".to_string(),
        );
        assert_eq!(node.id, "john3");
        assert_eq!(node.address, "100.71.97.55");
        assert_eq!(node.health, NodeHealth::Offline);
        assert!(!node.reachable);
        assert_eq!(node.error.as_deref(), Some("operation timed out"));
    }
}
