use std::{
    collections::BTreeMap,
    fs, io,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

const QUEUE_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Deserialize)]
struct QueueFile {
    schema_version: u16,
    campaign_id: String,
    updated_unix_ms: u128,
    hosts: BTreeMap<String, QueueHost>,
    tasks: Vec<QueueTask>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueHost {
    pub root: String,
    pub intent: String,
    pub reason: Option<String>,
    pub updated_unix_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueClaim {
    pub host: String,
    pub claimed_unix_ms: u128,
    pub heartbeat_unix_ms: u128,
    pub lease_expires_unix_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueResources {
    pub cpu_cores: u32,
    pub memory_gib: f64,
    pub uses_mlx: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueTask {
    pub id: String,
    pub title: String,
    pub experiment_id: String,
    pub decision: String,
    pub workload_class: String,
    pub priority: u32,
    pub decision_value: f64,
    pub expected_runtime_seconds: f64,
    pub critical_path: bool,
    #[serde(default)]
    pub decision_terminal: bool,
    pub compatible_hosts: Vec<String>,
    pub dependencies: Vec<String>,
    pub artifact_path: String,
    pub stop_rule: String,
    pub resources: QueueResources,
    pub status: String,
    pub claim: Option<QueueClaim>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct QueueSummary {
    pub total: usize,
    pub blocked: usize,
    pub ready: usize,
    pub running: usize,
    pub completed: usize,
    pub failed: usize,
    pub cancelled: usize,
    pub critical_path_ready: usize,
    pub duplicate_running: usize,
    pub decisions_completed: usize,
    pub ready_decision_value: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueueResponse {
    pub schema_version: u16,
    pub configured: bool,
    pub source_path: PathBuf,
    pub campaign_id: Option<String>,
    pub updated_unix_ms: Option<u128>,
    pub summary: QueueSummary,
    pub hosts: BTreeMap<String, QueueHost>,
    pub tasks: Vec<QueueTask>,
    pub error: Option<String>,
}

impl QueueResponse {
    fn unavailable(path: &Path, error: Option<String>) -> Self {
        Self {
            schema_version: QUEUE_SCHEMA_VERSION,
            configured: false,
            source_path: path.to_path_buf(),
            campaign_id: None,
            updated_unix_ms: None,
            summary: QueueSummary::default(),
            hosts: BTreeMap::new(),
            tasks: Vec::new(),
            error,
        }
    }
}

pub fn load(path: &Path) -> QueueResponse {
    match load_file(path) {
        Ok(queue) => {
            let summary = summarize(&queue.tasks);
            QueueResponse {
                schema_version: queue.schema_version,
                configured: true,
                source_path: path.to_path_buf(),
                campaign_id: Some(queue.campaign_id),
                updated_unix_ms: Some(queue.updated_unix_ms),
                summary,
                hosts: queue.hosts,
                tasks: queue.tasks,
                error: None,
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            QueueResponse::unavailable(path, None)
        }
        Err(error) => QueueResponse::unavailable(path, Some(error.to_string())),
    }
}

fn load_file(path: &Path) -> io::Result<QueueFile> {
    let bytes = fs::read(path)?;
    let queue: QueueFile = serde_json::from_slice(&bytes).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("cluster queue is invalid: {error}"),
        )
    })?;
    if queue.schema_version != QUEUE_SCHEMA_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported cluster queue schema {}; expected {}",
                queue.schema_version, QUEUE_SCHEMA_VERSION
            ),
        ));
    }
    Ok(queue)
}

fn summarize(tasks: &[QueueTask]) -> QueueSummary {
    let mut summary = QueueSummary {
        total: tasks.len(),
        ..QueueSummary::default()
    };
    for task in tasks {
        match task.status.as_str() {
            "blocked" => summary.blocked += 1,
            "ready" => {
                summary.ready += 1;
                summary.ready_decision_value += task.decision_value;
                if task.critical_path {
                    summary.critical_path_ready += 1;
                }
            }
            "running" => {
                summary.running += 1;
                if task.workload_class == "replica" {
                    summary.duplicate_running += 1;
                }
            }
            "completed" => summary.completed += 1,
            "failed" => summary.failed += 1,
            "cancelled" => summary.cancelled += 1,
            _ => {}
        }
        if task.status == "completed" && task.decision_terminal {
            summary.decisions_completed += 1;
        }
    }
    summary
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn temporary_queue_path(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "cascadia-cluster-queue-{name}-{}-{nonce}.json",
            std::process::id()
        ))
    }

    fn queue_json() -> serde_json::Value {
        serde_json::json!({
            "schema_version": 1,
            "campaign_id": "research-v1",
            "created_unix_ms": 1,
            "updated_unix_ms": 2,
            "hosts": {
                "john1": {
                    "root": "/tmp/cascadia",
                    "intent": "working",
                    "reason": "origin",
                    "updated_unix_ms": 2
                }
            },
            "tasks": [
                {
                    "id": "origin",
                    "title": "Origin",
                    "experiment_id": "experiment-v1",
                    "decision": "Test the treatment",
                    "workload_class": "independent-experiment",
                    "priority": 1,
                    "decision_value": 2.0,
                    "expected_runtime_seconds": 60.0,
                    "critical_path": true,
                    "decision_terminal": false,
                    "compatible_hosts": ["john1"],
                    "dependencies": [],
                    "command": ["true"],
                    "artifact_path": "artifact.json",
                    "stop_rule": "Complete once.",
                    "resources": {
                        "cpu_cores": 1,
                        "memory_gib": 1.0,
                        "uses_mlx": true
                    },
                    "status": "running",
                    "claim": {
                        "host": "john1",
                        "token": "secret-not-served",
                        "claimed_unix_ms": 1,
                        "heartbeat_unix_ms": 2,
                        "lease_expires_unix_ms": 100
                    },
                    "attempts": [],
                    "result": null,
                    "created_unix_ms": 1
                },
                {
                    "id": "analysis",
                    "title": "Analysis",
                    "experiment_id": "analysis-v1",
                    "decision": "Classify the result",
                    "workload_class": "shared-prerequisite",
                    "priority": 2,
                    "decision_value": 3.5,
                    "expected_runtime_seconds": 30.0,
                    "critical_path": true,
                    "decision_terminal": true,
                    "compatible_hosts": ["john1"],
                    "dependencies": [],
                    "command": ["true"],
                    "artifact_path": "analysis.json",
                    "stop_rule": "Write the report.",
                    "resources": {
                        "cpu_cores": 1,
                        "memory_gib": 1.0,
                        "uses_mlx": false
                    },
                    "status": "ready",
                    "claim": null,
                    "attempts": [],
                    "result": null,
                    "created_unix_ms": 1
                }
            ],
            "events": []
        })
    }

    #[test]
    fn loads_and_summarizes_a_configured_queue() {
        let path = temporary_queue_path("configured");
        fs::write(&path, serde_json::to_vec(&queue_json()).unwrap()).unwrap();
        let response = load(&path);

        assert!(response.configured);
        assert_eq!(response.campaign_id.as_deref(), Some("research-v1"));
        assert_eq!(response.summary.total, 2);
        assert_eq!(response.summary.running, 1);
        assert_eq!(response.summary.ready, 1);
        assert_eq!(response.summary.critical_path_ready, 1);
        assert_eq!(response.summary.ready_decision_value, 3.5);
        assert_eq!(response.summary.decisions_completed, 0);
        assert_eq!(response.tasks[0].claim.as_ref().unwrap().host, "john1");
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn missing_queue_is_an_empty_unconfigured_response() {
        let response = load(Path::new("/definitely/not/a/queue.json"));
        assert!(!response.configured);
        assert!(response.error.is_none());
        assert!(response.tasks.is_empty());
    }

    #[test]
    fn malformed_queue_is_visible_without_crashing_the_dashboard() {
        let path = temporary_queue_path("malformed");
        fs::write(&path, b"{not json").unwrap();
        let response = load(&path);
        assert!(!response.configured);
        assert!(response.error.as_deref().unwrap().contains("invalid"));
        fs::remove_file(path).unwrap();
    }
}
