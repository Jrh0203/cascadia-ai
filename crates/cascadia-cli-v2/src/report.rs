use std::{
    env,
    fmt::Write as _,
    io,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use cascadia_provenance::{checksum_file, repository_root, source_provenance};
use serde::Serialize;
use serde_json::{Map, Value};

const REPORT_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Serialize)]
pub struct ReportProvenance {
    created_unix_seconds: u64,
    git_revision: String,
    git_dirty: bool,
    git_status_blake3: String,
    v2_source_blake3: String,
    executable: String,
    executable_blake3: String,
    command_argv: Vec<String>,
    hardware: HardwareProvenance,
    toolchain: ToolchainProvenance,
    input_artifacts: Vec<InputArtifactProvenance>,
}

#[derive(Debug, Clone, Serialize)]
struct HardwareProvenance {
    chip: String,
    logical_cpu_count: usize,
    memory_bytes: String,
    operating_system: String,
    architecture: String,
}

#[derive(Debug, Clone, Serialize)]
struct ToolchainProvenance {
    rustc: String,
    cargo: String,
    package_version: String,
}

#[derive(Debug, Clone, Serialize)]
struct InputArtifactProvenance {
    role: String,
    path: String,
    manifest: String,
    manifest_blake3: String,
}

pub struct ReportContext {
    configuration: Value,
    provenance: ReportProvenance,
}

impl ReportContext {
    pub fn capture(configuration: Value) -> io::Result<Self> {
        let repository = repository_root()?;
        let source = source_provenance()?;
        let executable = env::current_exe()?;
        let provenance = ReportProvenance {
            created_unix_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(io::Error::other)?
                .as_secs(),
            git_revision: source.git_revision,
            git_dirty: source.git_dirty,
            git_status_blake3: source.git_status_blake3,
            v2_source_blake3: source.v2_source_blake3,
            executable: executable.display().to_string(),
            executable_blake3: checksum_file(&executable)?,
            command_argv: env::args().collect(),
            hardware: HardwareProvenance {
                chip: command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
                    .unwrap_or_else(|| env::consts::ARCH.to_owned()),
                logical_cpu_count: std::thread::available_parallelism()
                    .map(usize::from)
                    .unwrap_or(1),
                memory_bytes: command_output("sysctl", &["-n", "hw.memsize"])
                    .unwrap_or_else(|| "unknown".to_owned()),
                operating_system: command_output("sw_vers", &["-productVersion"])
                    .map(|version| format!("macOS {version}"))
                    .unwrap_or_else(|| env::consts::OS.to_owned()),
                architecture: env::consts::ARCH.to_owned(),
            },
            toolchain: ToolchainProvenance {
                rustc: command_output("rustc", &["--version"])
                    .unwrap_or_else(|| "unavailable".to_owned()),
                cargo: command_output("cargo", &["--version"])
                    .unwrap_or_else(|| "unavailable".to_owned()),
                package_version: env!("CARGO_PKG_VERSION").to_owned(),
            },
            input_artifacts: input_artifacts(&configuration, &repository)?,
        };
        Ok(Self {
            configuration,
            provenance,
        })
    }

    pub fn to_json<T: Serialize>(&self, report: &T) -> Result<String, serde_json::Error> {
        let value = self.enrich_value(serde_json::to_value(report)?)?;
        serde_json::to_string_pretty(&value)
    }

    pub fn enrich_value(&self, mut value: Value) -> Result<Value, serde_json::Error> {
        let object = value
            .as_object_mut()
            .expect("benchmark reports serialize as JSON objects");
        object.insert(
            "report_schema_version".to_owned(),
            REPORT_SCHEMA_VERSION.into(),
        );
        object.insert("configuration".to_owned(), self.configuration.clone());
        object.insert(
            "provenance".to_owned(),
            serde_json::to_value(&self.provenance)?,
        );
        Ok(value)
    }
}

pub fn write_report(
    json_path: &Path,
    json: &str,
    markdown: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = json_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(json_path, format!("{json}\n"))?;
    let mut rendered_markdown = markdown.trim_end().to_owned();
    if let Some(reproduction) = reproduction_markdown(json)? {
        rendered_markdown.push_str(&reproduction);
    }
    rendered_markdown.push('\n');
    std::fs::write(json_path.with_extension("md"), rendered_markdown)?;
    Ok(())
}

pub fn model_source_args(
    run_dir: Option<PathBuf>,
    model_dir: Option<PathBuf>,
) -> Result<Vec<std::ffi::OsString>, Box<dyn std::error::Error>> {
    match (run_dir, model_dir) {
        (Some(path), None) => Ok(vec![
            std::ffi::OsString::from("--run-dir"),
            path.into_os_string(),
        ]),
        (None, Some(path)) => Ok(vec![
            std::ffi::OsString::from("--model-dir"),
            path.into_os_string(),
        ]),
        _ => Err("exactly one of --run-dir or --model-dir is required".into()),
    }
}

fn reproduction_markdown(json: &str) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let value: serde_json::Value = serde_json::from_str(json)?;
    let Some(configuration) = value.get("configuration") else {
        return Ok(None);
    };
    let Some(provenance) = value.get("provenance") else {
        return Ok(None);
    };
    let text = |key: &str| {
        provenance
            .get(key)
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unavailable")
    };
    let hardware = provenance
        .get("hardware")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let toolchain = provenance
        .get("toolchain")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let artifacts = provenance
        .get("input_artifacts")
        .cloned()
        .unwrap_or_else(|| serde_json::Value::Array(Vec::new()));
    let mut output = String::from("\n\n## Reproduction\n\n");
    writeln!(output, "- Git revision: `{}`", text("git_revision"))?;
    writeln!(
        output,
        "- Dirty tree / status digest: {} / `{}`",
        provenance
            .get("git_dirty")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true),
        text("git_status_blake3")
    )?;
    writeln!(output, "- V2 source digest: `{}`", text("v2_source_blake3"))?;
    writeln!(
        output,
        "- Executable digest: `{}`",
        text("executable_blake3")
    )?;
    writeln!(
        output,
        "- Hardware: `{}`",
        serde_json::to_string(&hardware)?
    )?;
    writeln!(
        output,
        "- Toolchain: `{}`",
        serde_json::to_string(&toolchain)?
    )?;
    writeln!(
        output,
        "- Input artifacts: `{}`",
        serde_json::to_string(&artifacts)?
    )?;
    write!(
        output,
        "\n### Typed Configuration\n\n```json\n{}\n```",
        serde_json::to_string_pretty(configuration)?
    )?;
    Ok(Some(output))
}

fn input_artifacts(
    configuration: &Value,
    repository: &Path,
) -> io::Result<Vec<InputArtifactProvenance>> {
    let mut paths = Vec::new();
    collect_artifact_paths(configuration, &mut paths);
    paths.sort();
    paths.dedup();
    paths
        .into_iter()
        .filter_map(|(role, configured_path)| {
            let root = if configured_path.is_absolute() {
                configured_path
            } else {
                repository.join(configured_path)
            };
            let manifest = ["model.json", "run.json", "dataset.json"]
                .into_iter()
                .map(|name| root.join(name))
                .find(|path| path.is_file())?;
            Some((role, root, manifest))
        })
        .map(|(role, root, manifest)| {
            Ok(InputArtifactProvenance {
                role,
                path: root.display().to_string(),
                manifest: manifest.display().to_string(),
                manifest_blake3: checksum_file(&manifest)?,
            })
        })
        .collect()
}

fn collect_artifact_paths(value: &Value, paths: &mut Vec<(String, PathBuf)>) {
    match value {
        Value::Object(object) => collect_object_artifact_paths(object, paths),
        Value::Array(values) => {
            for value in values {
                collect_artifact_paths(value, paths);
            }
        }
        _ => {}
    }
}

fn collect_object_artifact_paths(object: &Map<String, Value>, paths: &mut Vec<(String, PathBuf)>) {
    for (key, value) in object {
        if (key.ends_with("model_dir")
            || matches!(
                key.as_str(),
                "run_dir" | "dataset" | "train_dataset" | "validation_dataset"
            ))
            && let Some(path) = value.as_str()
        {
            paths.push((key.clone(), PathBuf::from(path)));
        }
        collect_artifact_paths(value, paths);
    }
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn report_enrichment_preserves_metrics_and_adds_reproduction_data() {
        let context = ReportContext::capture(json!({
            "benchmark": {
                "games": 4,
                "first_seed": 0
            }
        }))
        .unwrap();
        let value = context.enrich_value(json!({"mean_score": 86.25})).unwrap();

        assert_eq!(value["mean_score"], 86.25);
        assert_eq!(value["report_schema_version"], REPORT_SCHEMA_VERSION);
        assert_eq!(value["configuration"]["benchmark"]["games"], 4);
        assert!(value["provenance"]["v2_source_blake3"].is_string());
        assert!(value["provenance"]["executable_blake3"].is_string());
    }
}
