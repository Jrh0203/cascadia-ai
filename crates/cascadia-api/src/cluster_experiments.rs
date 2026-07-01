use std::{
    collections::BTreeSet,
    fs, io,
    path::{Component, Path, PathBuf},
};

use serde::{Deserialize, Serialize};

const EXPERIMENT_SCHEMA_VERSION: u16 = 1;
const STATUSES: &[&str] = &["planned", "running", "completed", "cancelled"];
const OUTCOMES: &[&str] = &["pending", "passed", "failed", "inconclusive", "invalid"];
const TONES: &[&str] = &["good", "bad", "warn", "neutral"];

#[derive(Debug, Clone, Deserialize)]
struct ExperimentFile {
    schema_version: u16,
    updated_unix_ms: u128,
    experiments: Vec<ExperimentRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentMetric {
    pub label: String,
    pub value: String,
    pub tone: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentCriterion {
    pub label: String,
    pub passed: Option<bool>,
    pub observed: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentArtifact {
    pub label: String,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentRecord {
    pub id: String,
    pub title: String,
    pub hypothesis: String,
    pub summary: String,
    pub status: String,
    pub outcome: String,
    pub verdict: Option<String>,
    pub plan_section: Option<String>,
    pub started_unix_ms: Option<u128>,
    pub completed_unix_ms: Option<u128>,
    pub updated_unix_ms: u128,
    #[serde(default)]
    pub hosts: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub task_ids: Vec<String>,
    #[serde(default)]
    pub metrics: Vec<ExperimentMetric>,
    #[serde(default)]
    pub criteria: Vec<ExperimentCriterion>,
    #[serde(default)]
    pub notes: Vec<String>,
    #[serde(default)]
    pub artifacts: Vec<ExperimentArtifact>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExperimentResponse {
    pub schema_version: u16,
    pub configured: bool,
    pub source_path: PathBuf,
    pub updated_unix_ms: Option<u128>,
    pub experiments: Vec<ExperimentRecord>,
    pub error: Option<String>,
}

impl ExperimentResponse {
    fn unavailable(path: &Path, error: Option<String>) -> Self {
        Self {
            schema_version: EXPERIMENT_SCHEMA_VERSION,
            configured: false,
            source_path: path.to_path_buf(),
            updated_unix_ms: None,
            experiments: Vec::new(),
            error,
        }
    }
}

pub fn load(path: &Path) -> ExperimentResponse {
    match load_file(path) {
        Ok(file) => ExperimentResponse {
            schema_version: file.schema_version,
            configured: true,
            source_path: path.to_path_buf(),
            updated_unix_ms: Some(file.updated_unix_ms),
            experiments: file.experiments,
            error: None,
        },
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            ExperimentResponse::unavailable(path, None)
        }
        Err(error) => ExperimentResponse::unavailable(path, Some(error.to_string())),
    }
}

fn load_file(path: &Path) -> io::Result<ExperimentFile> {
    let bytes = fs::read(path)?;
    let file: ExperimentFile = serde_json::from_slice(&bytes).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("cluster experiment ledger is invalid: {error}"),
        )
    })?;
    validate(file).map_err(|message| io::Error::new(io::ErrorKind::InvalidData, message))
}

fn validate(file: ExperimentFile) -> Result<ExperimentFile, String> {
    if file.schema_version != EXPERIMENT_SCHEMA_VERSION {
        return Err(format!(
            "unsupported cluster experiment schema {}; expected {}",
            file.schema_version, EXPERIMENT_SCHEMA_VERSION
        ));
    }
    let mut ids = BTreeSet::new();
    for experiment in &file.experiments {
        require_nonempty(&experiment.id, "experiment id")?;
        if !ids.insert(experiment.id.as_str()) {
            return Err(format!("duplicate experiment id {}", experiment.id));
        }
        for (field, value) in [
            ("title", &experiment.title),
            ("hypothesis", &experiment.hypothesis),
            ("summary", &experiment.summary),
        ] {
            require_nonempty(value, &format!("experiment {} {field}", experiment.id))?;
        }
        if !STATUSES.contains(&experiment.status.as_str()) {
            return Err(format!(
                "experiment {} has an invalid status",
                experiment.id
            ));
        }
        if !OUTCOMES.contains(&experiment.outcome.as_str()) {
            return Err(format!(
                "experiment {} has an invalid outcome",
                experiment.id
            ));
        }
        if experiment.status == "completed" {
            if experiment.completed_unix_ms.is_none() || experiment.outcome == "pending" {
                return Err(format!(
                    "completed experiment {} requires a completion time and outcome",
                    experiment.id
                ));
            }
        } else if experiment.completed_unix_ms.is_some() {
            return Err(format!(
                "non-completed experiment {} cannot have a completion time",
                experiment.id
            ));
        }
        if experiment.status == "running" && experiment.started_unix_ms.is_none() {
            return Err(format!(
                "running experiment {} requires a start time",
                experiment.id
            ));
        }
        if let (Some(started), Some(completed)) =
            (experiment.started_unix_ms, experiment.completed_unix_ms)
            && completed < started
        {
            return Err(format!(
                "experiment {} completes before it starts",
                experiment.id
            ));
        }
        require_unique_nonempty(&experiment.hosts, &experiment.id, "hosts")?;
        require_unique_nonempty(&experiment.tags, &experiment.id, "tags")?;
        require_unique_nonempty(&experiment.task_ids, &experiment.id, "task_ids")?;
        for metric in &experiment.metrics {
            require_nonempty(
                &metric.label,
                &format!("experiment {} metric label", experiment.id),
            )?;
            require_nonempty(
                &metric.value,
                &format!("experiment {} metric value", experiment.id),
            )?;
            if !TONES.contains(&metric.tone.as_str()) {
                return Err(format!(
                    "experiment {} metric {} has an invalid tone",
                    experiment.id, metric.label
                ));
            }
        }
        for criterion in &experiment.criteria {
            require_nonempty(
                &criterion.label,
                &format!("experiment {} criterion", experiment.id),
            )?;
        }
        for note in &experiment.notes {
            require_nonempty(note, &format!("experiment {} note", experiment.id))?;
        }
        for artifact in &experiment.artifacts {
            require_nonempty(
                &artifact.label,
                &format!("experiment {} artifact label", experiment.id),
            )?;
            require_safe_relative_path(&artifact.path, &experiment.id)?;
        }
    }
    Ok(file)
}

fn require_nonempty(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{field} must be nonempty"))
    } else {
        Ok(())
    }
}

fn require_unique_nonempty(
    values: &[String],
    experiment_id: &str,
    field: &str,
) -> Result<(), String> {
    let mut unique = BTreeSet::new();
    for value in values {
        require_nonempty(value, &format!("experiment {experiment_id} {field}"))?;
        if !unique.insert(value) {
            return Err(format!(
                "experiment {experiment_id} {field} must not contain duplicates"
            ));
        }
    }
    Ok(())
}

fn require_safe_relative_path(value: &str, experiment_id: &str) -> Result<(), String> {
    require_nonempty(value, &format!("experiment {experiment_id} artifact path"))?;
    let path = Path::new(value);
    if path.is_absolute()
        || path
            .components()
            .any(|component| component == Component::ParentDir)
    {
        return Err(format!(
            "experiment {experiment_id} artifact path must stay inside the repository"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn temporary_path(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "cascadia-cluster-experiments-{name}-{}-{nonce}.json",
            std::process::id()
        ))
    }

    fn ledger_json() -> serde_json::Value {
        serde_json::json!({
            "schema_version": 1,
            "updated_unix_ms": 10,
            "experiments": [{
                "id": "experiment-v1",
                "title": "Experiment",
                "hypothesis": "The treatment improves recall.",
                "summary": "The treatment passed.",
                "status": "completed",
                "outcome": "passed",
                "verdict": "treatment_sufficient",
                "plan_section": "P2",
                "started_unix_ms": 1,
                "completed_unix_ms": 9,
                "updated_unix_ms": 10,
                "hosts": ["john1"],
                "tags": ["representation"],
                "task_ids": ["origin"],
                "metrics": [{
                    "label": "Recall",
                    "value": "98.0%",
                    "tone": "good"
                }],
                "criteria": [{
                    "label": "Recall at least 95%",
                    "passed": true,
                    "observed": "98.0%"
                }],
                "notes": ["Validation was opened once."],
                "artifacts": [{
                    "label": "Result",
                    "path": "docs/v2/reports/result.md"
                }]
            }]
        })
    }

    #[test]
    fn loads_valid_experiment_ledger() {
        let path = temporary_path("valid");
        fs::write(&path, serde_json::to_vec(&ledger_json()).unwrap()).unwrap();
        let response = load(&path);
        assert!(response.configured);
        assert_eq!(response.experiments.len(), 1);
        assert_eq!(response.experiments[0].outcome, "passed");
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_duplicate_experiment_ids() {
        let path = temporary_path("duplicate");
        let mut value = ledger_json();
        let duplicate = value["experiments"][0].clone();
        value["experiments"].as_array_mut().unwrap().push(duplicate);
        fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        let response = load(&path);
        assert!(!response.configured);
        assert!(response.error.unwrap().contains("duplicate"));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn missing_ledger_is_unconfigured_without_error() {
        let response = load(Path::new("/definitely/not/an/experiment-ledger.json"));
        assert!(!response.configured);
        assert!(response.error.is_none());
    }
}
