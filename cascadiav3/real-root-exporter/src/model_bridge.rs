//! JSONL stdio model-service bridge.
//!
//! One session owns one spawned Python bridge process. `eval` sends a single
//! root; `eval_batch` sends many roots in one request when the bridge
//! advertises the `eval_batch` protocol feature in its hello payload, and
//! degrades to sequential single evals otherwise.

use std::collections::HashSet;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use crate::canonical_json;

/// Batches are chunked service-side; this cap only bounds the timeout scaling.
const BATCH_TIMEOUT_SCALE_CAP: u32 = 1024;

#[derive(Debug, Clone)]
pub struct ModelEval {
    pub priors: Vec<Value>,
    pub q: Option<Vec<f64>>,
    pub score_to_go: Option<Vec<f64>>,
    pub value: Option<Vec<f64>>,
    pub model_fallback: bool,
    pub response: Value,
}

#[derive(Debug, Clone)]
pub struct BridgeConfig {
    pub model_manifest: Option<PathBuf>,
    pub model_timeout_ms: u64,
    pub allow_model_fallback: bool,
}

impl BridgeConfig {
    pub fn from_args(args: &crate::Args) -> Self {
        Self {
            model_manifest: args.model_manifest.clone(),
            model_timeout_ms: args.model_timeout_ms,
            allow_model_fallback: args.allow_model_fallback,
        }
    }
}

pub fn uniform_model_eval(action_count: usize) -> ModelEval {
    let prior = 1.0 / action_count as f64;
    ModelEval {
        priors: vec![json!(prior); action_count],
        q: Some(vec![0.0; action_count]),
        score_to_go: Some(vec![0.0; action_count]),
        value: None,
        model_fallback: true,
        response: json!({
            "type": "eval_response",
            "model_fallback": true,
            "source": "uniform_prior_fallback",
        }),
    }
}

/// Elementwise mean of several evals of the same action menu (symmetry TTA
/// variants). A mean of normalized prior vectors stays normalized, so no
/// re-normalization is needed. Optional arrays are averaged only when every
/// variant provides them (a partially available head cannot be averaged
/// consistently). The first variant's raw response is kept for provenance;
/// `model_fallback` is OR-ed so a fallback in any variant stays visible.
pub fn average_model_evals(evals: &[&ModelEval]) -> Result<ModelEval> {
    let first = *evals
        .first()
        .context("averaging requires at least one eval")?;
    if evals.len() == 1 {
        return Ok(first.clone());
    }
    let action_count = first.priors.len();
    let variant_count = evals.len() as f64;
    let mut priors = vec![0.0_f64; action_count];
    for eval in evals {
        if eval.priors.len() != action_count {
            bail!("TTA variant menu size mismatch");
        }
        for (slot, value) in eval.priors.iter().enumerate() {
            priors[slot] += value.as_f64().context("model prior must be numeric")? / variant_count;
        }
    }
    let average_optional = |field: &dyn Fn(&ModelEval) -> Option<&Vec<f64>>,
                            expected_len: usize|
     -> Result<Option<Vec<f64>>> {
        let mut sums = vec![0.0_f64; expected_len];
        for eval in evals {
            let Some(values) = field(eval) else {
                return Ok(None);
            };
            if values.len() != expected_len {
                bail!("TTA variant array length mismatch");
            }
            for (slot, value) in values.iter().enumerate() {
                sums[slot] += value / variant_count;
            }
        }
        Ok(Some(sums))
    };
    Ok(ModelEval {
        priors: priors.into_iter().map(|prior| json!(prior)).collect(),
        q: average_optional(&|eval| eval.q.as_ref(), action_count)?,
        score_to_go: average_optional(&|eval| eval.score_to_go.as_ref(), action_count)?,
        value: average_optional(&|eval| eval.value.as_ref(), 4)?,
        model_fallback: evals.iter().any(|eval| eval.model_fallback),
        response: first.response.clone(),
    })
}

pub struct ModelServiceSession {
    child: Child,
    stdin: ChildStdin,
    line_rx: mpsc::Receiver<Result<String, String>>,
    timeout: Duration,
    allow_model_fallback: bool,
    protocol_features: HashSet<String>,
}

impl ModelServiceSession {
    pub fn spawn(command: &str, config: &BridgeConfig) -> Result<Self> {
        if let Some(manifest) = &config.model_manifest {
            if !manifest.exists() {
                bail!("model manifest {} does not exist", manifest.display());
            }
        }
        let mut child = Command::new("sh")
            .arg("-c")
            .arg(command)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("spawning model service command {command:?}"))?;
        let stdin = child
            .stdin
            .take()
            .context("model service stdin unavailable")?;
        let stdout = child
            .stdout
            .take()
            .context("model service stdout unavailable")?;
        let mut reader = std::io::BufReader::new(stdout);
        let (line_tx, line_rx) = mpsc::channel::<Result<String, String>>();
        std::thread::spawn(move || {
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        if line_tx.send(Ok(line)).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = line_tx.send(Err(error.to_string()));
                        break;
                    }
                }
            }
        });
        let timeout = Duration::from_millis(config.model_timeout_ms);

        let hello_line = recv_model_line(&line_rx, &mut child, timeout, "hello")?;
        let hello: Value = serde_json::from_str(hello_line.trim())
            .with_context(|| format!("invalid model service hello: {hello_line:?}"))?;
        if hello.get("type").and_then(Value::as_str) == Some("error") {
            bail!("model service hello error: {}", canonical_json(&hello));
        }
        if hello.get("type").and_then(Value::as_str) != Some("hello") {
            bail!(
                "model service did not send hello: {}",
                canonical_json(&hello)
            );
        }
        let protocol_features = hello
            .get("protocol_features")
            .and_then(Value::as_array)
            .map(|features| {
                features
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect::<HashSet<_>>()
            })
            .unwrap_or_default();
        Ok(Self {
            child,
            stdin,
            line_rx,
            timeout,
            allow_model_fallback: config.allow_model_fallback,
            protocol_features,
        })
    }

    pub fn supports_eval_batch(&self) -> bool {
        self.protocol_features.contains("eval_batch")
    }

    /// Bridge accepts precomputed feature arrays in eval requests
    /// ("packed_features"), skipping Python-side per-row extraction.
    pub fn supports_packed_features(&self) -> bool {
        self.protocol_features.contains("packed_features")
    }

    /// Bridge can return per-row outputs as base64 little-endian f64 arrays
    /// ("packed_response") instead of JSON float lists. Values are bit
    /// identical to the JSON path; only the encoding changes.
    pub fn supports_packed_response(&self) -> bool {
        self.protocol_features.contains("packed_response")
    }

    pub fn eval(&mut self, root_request: &Value) -> Result<ModelEval> {
        let request = json!({
            "type": "eval_request",
            "root": root_request,
            "allow_model_fallback": self.allow_model_fallback,
            "timeout_ms": self.timeout.as_millis(),
            "packed_response": self.supports_packed_response(),
        });
        writeln!(self.stdin, "{}", canonical_json(&request))
            .context("writing model eval request")?;
        self.stdin.flush().context("flushing model eval request")?;

        let response_line = recv_model_line(
            &self.line_rx,
            &mut self.child,
            self.timeout,
            "eval_response",
        )?;
        let response: Value = serde_json::from_str(response_line.trim())
            .with_context(|| format!("invalid model eval response: {response_line:?}"))?;
        parse_model_response(root_request, response, self.allow_model_fallback)
    }

    /// One request, many roots. Falls back to sequential `eval` calls when the
    /// bridge did not advertise `eval_batch`.
    pub fn eval_batch(&mut self, root_requests: &[Value]) -> Result<Vec<ModelEval>> {
        if root_requests.is_empty() {
            return Ok(Vec::new());
        }
        if !self.supports_eval_batch() {
            return root_requests
                .iter()
                .map(|root_request| self.eval(root_request))
                .collect();
        }
        let request = json!({
            "type": "eval_batch_request",
            "roots": root_requests,
            "allow_model_fallback": self.allow_model_fallback,
            "timeout_ms": self.timeout.as_millis(),
            "packed_response": self.supports_packed_response(),
        });
        writeln!(self.stdin, "{}", canonical_json(&request))
            .context("writing model eval batch request")?;
        self.stdin
            .flush()
            .context("flushing model eval batch request")?;

        let scale = (root_requests.len() as u32).min(BATCH_TIMEOUT_SCALE_CAP).max(1);
        let response_line = recv_model_line(
            &self.line_rx,
            &mut self.child,
            self.timeout.saturating_mul(scale),
            "eval_batch_response",
        )?;
        let response: Value = serde_json::from_str(response_line.trim())
            .with_context(|| format!("invalid model eval batch response: {response_line:?}"))?;
        if response.get("type").and_then(Value::as_str) == Some("error") {
            bail!("model service error: {}", canonical_json(&response));
        }
        if response.get("type").and_then(Value::as_str) != Some("eval_batch_response") {
            bail!(
                "model service response is not eval_batch_response: {}",
                canonical_json(&response)
            );
        }
        let results = response
            .get("results")
            .and_then(Value::as_array)
            .context("eval_batch_response results missing")?;
        if results.len() != root_requests.len() {
            bail!(
                "eval_batch_response results length mismatch: got {}, expected {}",
                results.len(),
                root_requests.len()
            );
        }
        root_requests
            .iter()
            .zip(results.iter().cloned())
            .map(|(root_request, result)| {
                parse_model_response(root_request, result, self.allow_model_fallback)
            })
            .collect()
    }

    pub fn shutdown(&mut self) {
        let shutdown = json!({"type": "shutdown"});
        let _ = writeln!(self.stdin, "{}", canonical_json(&shutdown));
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}

impl Drop for ModelServiceSession {
    fn drop(&mut self) {
        let _ = writeln!(
            self.stdin,
            "{}",
            canonical_json(&json!({"type": "shutdown"}))
        );
        let _ = self.stdin.flush();
        if let Ok(None) = self.child.try_wait() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn recv_model_line(
    line_rx: &mpsc::Receiver<Result<String, String>>,
    child: &mut Child,
    timeout: Duration,
    label: &str,
) -> Result<String> {
    match line_rx.recv_timeout(timeout) {
        Ok(Ok(line)) => Ok(line),
        Ok(Err(error)) => bail!("reading model service {label} failed: {error}"),
        Err(mpsc::RecvTimeoutError::Timeout) => {
            let _ = child.kill();
            bail!(
                "model service timed out waiting for {label} after {:?}",
                timeout
            );
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            let status = child.try_wait().ok().flatten();
            bail!("model service closed stdout before {label}; status={status:?}");
        }
    }
}

pub fn parse_model_response(
    root_request: &Value,
    response: Value,
    allow_model_fallback: bool,
) -> Result<ModelEval> {
    if response.get("type").and_then(Value::as_str) == Some("error") {
        bail!("model service error: {}", canonical_json(&response));
    }
    if response.get("type").and_then(Value::as_str) != Some("eval_response") {
        bail!(
            "model service response is not eval_response: {}",
            canonical_json(&response)
        );
    }
    let service_model_fallback = response
        .get("model_fallback")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if service_model_fallback && !allow_model_fallback {
        bail!("model service declared model_fallback=true but --allow-model-fallback was not set");
    }
    let expected_action_ids = root_request
        .get("action_ids")
        .and_then(Value::as_array)
        .context("root request action_ids missing")?;
    let response_action_ids = response
        .get("action_ids")
        .and_then(Value::as_array)
        .context("model response action_ids missing")?;
    if response_action_ids != expected_action_ids {
        bail!("model response action_ids do not match root legal action order");
    }
    let action_count = expected_action_ids.len();
    // Packed responses ("packed_response" protocol feature) carry base64
    // little-endian f64 arrays; values decode bit-identically to what the
    // JSON float-list path yields (serde_json has float_roundtrip enabled).
    let (raw_priors, q, score_to_go, value) = if let Some(packed) = response.get("packed") {
        let priors = parse_packed_model_float_array(packed, "priors_f64_b64", action_count)?
            .context("model response priors missing")?;
        (
            priors,
            parse_packed_model_float_array(packed, "q_f64_b64", action_count)?,
            parse_packed_model_float_array(packed, "score_to_go_f64_b64", action_count)?,
            parse_packed_model_float_array(packed, "value_f64_b64", 4)?,
        )
    } else {
        let priors_json = response
            .get("priors")
            .and_then(Value::as_array)
            .context("model response priors missing")?;
        if priors_json.len() != action_count {
            bail!("model response priors length mismatch");
        }
        let mut priors = Vec::with_capacity(priors_json.len());
        for value in priors_json {
            priors.push(value.as_f64().context("model prior must be numeric")?);
        }
        (
            priors,
            parse_optional_model_float_array(&response, "q", action_count)?,
            parse_optional_model_float_array(&response, "score_to_go", action_count)?,
            parse_optional_model_float_array(&response, "value", 4)?,
        )
    };
    let mut sum = 0.0_f64;
    for &prior in &raw_priors {
        if !prior.is_finite() || prior < 0.0 {
            bail!("model prior must be finite and non-negative");
        }
        sum += prior;
    }
    if sum <= 0.0 || !sum.is_finite() {
        bail!("model priors must have positive finite sum");
    }
    let priors = raw_priors
        .into_iter()
        .map(|prior| json!(prior / sum))
        .collect::<Vec<_>>();
    Ok(ModelEval {
        priors,
        q,
        score_to_go,
        value,
        model_fallback: service_model_fallback,
        response,
    })
}

pub fn parse_optional_model_float_array(
    response: &Value,
    key: &str,
    expected_len: usize,
) -> Result<Option<Vec<f64>>> {
    let Some(raw) = response.get(key) else {
        return Ok(None);
    };
    let values = raw
        .as_array()
        .with_context(|| format!("model response {key} must be an array"))?;
    if values.len() != expected_len {
        bail!(
            "model response {key} length mismatch: got {}, expected {}",
            values.len(),
            expected_len
        );
    }
    let mut parsed = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let number = value
            .as_f64()
            .with_context(|| format!("model response {key}[{index}] must be numeric"))?;
        if !number.is_finite() {
            bail!("model response {key}[{index}] must be finite");
        }
        parsed.push(number);
    }
    Ok(Some(parsed))
}

/// Decodes one base64 little-endian f64 array from a packed response object.
/// Mirrors `parse_optional_model_float_array` semantics (optional key,
/// exact length, finite values).
pub fn parse_packed_model_float_array(
    packed: &Value,
    key: &str,
    expected_len: usize,
) -> Result<Option<Vec<f64>>> {
    use base64::Engine as _;

    let Some(raw) = packed.get(key) else {
        return Ok(None);
    };
    let encoded = raw
        .as_str()
        .with_context(|| format!("model response packed {key} must be a base64 string"))?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .with_context(|| format!("model response packed {key} is not valid base64"))?;
    if bytes.len() != expected_len * 8 {
        bail!(
            "model response packed {key} length mismatch: got {} bytes, expected {}",
            bytes.len(),
            expected_len * 8
        );
    }
    let mut parsed = Vec::with_capacity(expected_len);
    for (index, chunk) in bytes.chunks_exact(8).enumerate() {
        let number = f64::from_le_bytes(chunk.try_into().expect("8-byte chunk"));
        if !number.is_finite() {
            bail!("model response packed {key}[{index}] must be finite");
        }
        parsed.push(number);
    }
    Ok(Some(parsed))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_bridge_command(extra: &str) -> String {
        format!(
            "python3 {}/../tests/mock_model_bridge.py {extra}",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    fn test_config() -> BridgeConfig {
        BridgeConfig {
            model_manifest: None,
            model_timeout_ms: 20_000,
            allow_model_fallback: false,
        }
    }

    fn root_request(index: usize) -> Value {
        json!({
            "state_hash": format!("hash-{index}"),
            "action_ids": ["a0", "a1", "a2"],
            "legal_actions": [
                {"action_id": "a0"},
                {"action_id": "a1"},
                {"action_id": "a2"},
            ],
            "exact_afterstate_score_active": [5.0 + index as f64, 7.0, 3.5],
        })
    }

    #[test]
    fn eval_batch_matches_sequential_eval() {
        let roots: Vec<Value> = (0..3).map(root_request).collect();

        let mut batch_session = ModelServiceSession::spawn(&mock_bridge_command(""), &test_config())
            .expect("spawn batch mock");
        assert!(batch_session.supports_eval_batch());
        let batch_evals = batch_session.eval_batch(&roots).expect("batch evals");
        batch_session.shutdown();

        let mut single_session =
            ModelServiceSession::spawn(&mock_bridge_command("--no-batch"), &test_config())
                .expect("spawn no-batch mock");
        assert!(!single_session.supports_eval_batch());
        // eval_batch on a non-batch bridge falls back to sequential evals.
        let sequential_evals = single_session.eval_batch(&roots).expect("sequential evals");
        single_session.shutdown();

        assert_eq!(batch_evals.len(), roots.len());
        assert_eq!(sequential_evals.len(), roots.len());
        for (batch_eval, single_eval) in batch_evals.iter().zip(sequential_evals.iter()) {
            assert_eq!(batch_eval.priors, single_eval.priors);
            assert_eq!(batch_eval.q, single_eval.q);
            assert_eq!(batch_eval.score_to_go, single_eval.score_to_go);
            assert_eq!(batch_eval.value, single_eval.value);
        }
        assert_eq!(batch_evals[0].q, Some(vec![6.0, 8.0, 4.5]));
        assert_eq!(batch_evals[0].value, Some(vec![80.0, 80.0, 80.0, 80.0]));
    }

    #[test]
    fn empty_batch_is_a_no_op() {
        let mut session = ModelServiceSession::spawn(&mock_bridge_command(""), &test_config())
            .expect("spawn mock");
        assert!(session.eval_batch(&[]).expect("empty batch").is_empty());
        session.shutdown();
    }

    #[test]
    fn packed_response_matches_json_response_exactly() {
        let roots: Vec<Value> = (0..4).map(root_request).collect();

        let mut packed_session =
            ModelServiceSession::spawn(&mock_bridge_command(""), &test_config())
                .expect("spawn packed mock");
        assert!(packed_session.supports_packed_response());
        let packed_evals = packed_session.eval_batch(&roots).expect("packed evals");
        let packed_single = packed_session.eval(&roots[0]).expect("packed single eval");
        packed_session.shutdown();

        let mut json_session =
            ModelServiceSession::spawn(&mock_bridge_command("--no-packed-response"), &test_config())
                .expect("spawn json mock");
        assert!(!json_session.supports_packed_response());
        assert!(json_session.supports_eval_batch());
        let json_evals = json_session.eval_batch(&roots).expect("json evals");
        let json_single = json_session.eval(&roots[0]).expect("json single eval");
        json_session.shutdown();

        // The packed session really used the packed encoding and the json
        // session really did not.
        assert!(packed_evals[0].response.get("packed").is_some());
        assert!(json_evals[0].response.get("packed").is_none());
        assert!(packed_single.response.get("packed").is_some());
        assert!(json_single.response.get("packed").is_none());

        // Exact equality: packed base64 f64 must decode bit-identically to
        // the JSON float-list path.
        for (packed_eval, json_eval) in packed_evals
            .iter()
            .chain(std::iter::once(&packed_single))
            .zip(json_evals.iter().chain(std::iter::once(&json_single)))
        {
            assert_eq!(packed_eval.priors, json_eval.priors);
            assert_eq!(packed_eval.q, json_eval.q);
            assert_eq!(packed_eval.score_to_go, json_eval.score_to_go);
            assert_eq!(packed_eval.value, json_eval.value);
            assert_eq!(packed_eval.model_fallback, json_eval.model_fallback);
        }
        assert_eq!(packed_evals[0].q, Some(vec![6.0, 8.0, 4.5]));
        assert_eq!(packed_evals[0].value, Some(vec![80.0, 80.0, 80.0, 80.0]));
    }

    #[test]
    fn packed_array_decoder_validates_shape_and_encoding() {
        use base64::Engine as _;

        let engine = base64::engine::general_purpose::STANDARD;
        let values = [0.125_f64, -3.5, 1.0e-12];
        let mut bytes = Vec::new();
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let packed = json!({"q_f64_b64": engine.encode(&bytes)});
        let decoded = parse_packed_model_float_array(&packed, "q_f64_b64", 3)
            .expect("decode")
            .expect("present");
        assert_eq!(decoded, values.to_vec());
        // Missing key is None, wrong length and bad base64 are errors.
        assert!(
            parse_packed_model_float_array(&packed, "missing_f64_b64", 3)
                .expect("missing key ok")
                .is_none()
        );
        assert!(parse_packed_model_float_array(&packed, "q_f64_b64", 4).is_err());
        assert!(
            parse_packed_model_float_array(&json!({"q_f64_b64": "@@not-base64@@"}), "q_f64_b64", 3)
                .is_err()
        );
        let nan_bytes = f64::NAN.to_le_bytes();
        assert!(
            parse_packed_model_float_array(
                &json!({"q_f64_b64": engine.encode(nan_bytes)}),
                "q_f64_b64",
                1
            )
            .is_err()
        );
    }

    /// Micro-benchmark: response decode (serde_json parse + parse_model_response)
    /// for the JSON float-list encoding vs the packed base64 f64 encoding at a
    /// realistic 32 rows x 256 actions. Run with:
    /// `cargo test response_decode_microbench -- --nocapture`
    #[test]
    fn response_decode_microbench() {
        use base64::Engine as _;
        use std::fmt::Write as _;

        let rows = 32usize;
        let actions = 256usize;
        let engine = base64::engine::general_purpose::STANDARD;

        let mut roots = Vec::with_capacity(rows);
        let mut json_lines = Vec::with_capacity(rows);
        let mut packed_lines = Vec::with_capacity(rows);
        for row in 0..rows {
            let action_ids: Vec<String> = (0..actions).map(|i| format!("a{row}-{i}")).collect();
            let priors: Vec<f64> = (0..actions).map(|i| 1.0 / (i + 1) as f64).collect();
            let q: Vec<f64> = (0..actions).map(|i| (row * actions + i) as f64 * 0.37 + 0.001).collect();
            let score_to_go: Vec<f64> = q.iter().map(|value| value - 11.25).collect();
            let value = vec![80.5, 79.25, 63.0, 41.125];
            roots.push(json!({
                "state_hash": format!("hash-{row}"),
                "action_ids": action_ids,
            }));
            json_lines.push(
                serde_json::to_string(&json!({
                    "type": "eval_response",
                    "state_hash": format!("hash-{row}"),
                    "action_ids": roots[row]["action_ids"],
                    "priors": priors,
                    "q": q,
                    "score_to_go": score_to_go,
                    "uncertainty": vec![1.0; actions],
                    "value": value,
                    "model_fallback": false,
                }))
                .expect("json response"),
            );
            let pack = |values: &[f64]| {
                let mut bytes = Vec::with_capacity(values.len() * 8);
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                engine.encode(&bytes)
            };
            packed_lines.push(
                serde_json::to_string(&json!({
                    "type": "eval_response",
                    "state_hash": format!("hash-{row}"),
                    "action_ids": roots[row]["action_ids"],
                    "packed": {
                        "priors_f64_b64": pack(&priors),
                        "q_f64_b64": pack(&q),
                        "score_to_go_f64_b64": pack(&score_to_go),
                        "uncertainty_f64_b64": pack(&vec![1.0; actions]),
                        "value_f64_b64": pack(&value),
                    },
                    "model_fallback": false,
                }))
                .expect("packed response"),
            );
        }

        let decode_all = |lines: &[String]| -> Vec<ModelEval> {
            lines
                .iter()
                .zip(roots.iter())
                .map(|(line, root)| {
                    let response: Value = serde_json::from_str(line).expect("parse line");
                    parse_model_response(root, response, false).expect("parse response")
                })
                .collect()
        };

        // Parity first: identical parsed values on both encodings.
        let json_evals = decode_all(&json_lines);
        let packed_evals = decode_all(&packed_lines);
        for (json_eval, packed_eval) in json_evals.iter().zip(packed_evals.iter()) {
            assert_eq!(json_eval.priors, packed_eval.priors);
            assert_eq!(json_eval.q, packed_eval.q);
            assert_eq!(json_eval.score_to_go, packed_eval.score_to_go);
            assert_eq!(json_eval.value, packed_eval.value);
        }

        let iterations = 30usize;
        let time = |lines: &[String]| {
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                std::hint::black_box(decode_all(lines));
            }
            start.elapsed() / iterations as u32
        };
        let json_time = time(&json_lines);
        let packed_time = time(&packed_lines);
        let json_bytes: usize = json_lines.iter().map(String::len).sum();
        let packed_bytes: usize = packed_lines.iter().map(String::len).sum();
        let mut report = String::new();
        writeln!(
            report,
            "response decode microbench ({rows} rows x {actions} actions, mean of {iterations}):"
        )
        .unwrap();
        writeln!(report, "  json:   {json_time:>10.2?}  payload {json_bytes} bytes").unwrap();
        writeln!(report, "  packed: {packed_time:>10.2?}  payload {packed_bytes} bytes").unwrap();
        writeln!(
            report,
            "  speedup: {:.2}x, payload {:.2}x smaller",
            json_time.as_secs_f64() / packed_time.as_secs_f64(),
            json_bytes as f64 / packed_bytes as f64
        )
        .unwrap();
        eprintln!("{report}");
    }
}

/// Shared bridge: one spawned Python bridge (one CUDA context) serving many
/// worker threads. Workers submit eval jobs through a channel; an aggregator
/// thread merges concurrently pending jobs into a single cross-worker
/// eval_batch request (the Python side already chunks by memory budget) and
/// demuxes the responses. This removes the N-CUDA-context thrash and buys
/// large-batch GPU efficiency without a separate server process.
struct AggregateJob {
    requests: Vec<Value>,
    reply: std::sync::mpsc::SyncSender<Result<Vec<ModelEval>, String>>,
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

/// How long the aggregator waits for more jobs before dispatching a merged
/// batch. `CASCADIA_SHARED_GATHER_US` (microseconds), default 2000 (2ms).
pub fn shared_gather_window() -> Duration {
    Duration::from_micros(env_usize("CASCADIA_SHARED_GATHER_US", 2_000) as u64)
}

/// Row cap for one merged shared-bridge request.
/// `CASCADIA_SHARED_ROW_CAP`, default 192.
pub fn shared_row_cap() -> usize {
    env_usize("CASCADIA_SHARED_ROW_CAP", 192)
}

pub struct SharedBridge {
    tx: std::sync::mpsc::Sender<AggregateJob>,
    packed_features: bool,
}

#[derive(Clone)]
pub struct SharedBridgeClient {
    tx: std::sync::mpsc::Sender<AggregateJob>,
    packed_features: bool,
}

impl SharedBridge {
    /// `max_rows` bounds how many rows one merged request may carry; the
    /// gather window is short (default 2ms, `CASCADIA_SHARED_GATHER_US`) so
    /// lone jobs are not delayed.
    pub fn spawn(command: &str, config: &BridgeConfig, max_rows: usize) -> Result<Self> {
        let mut session = ModelServiceSession::spawn(command, config)?;
        let packed_features = session.supports_packed_features();
        let gather_window = shared_gather_window();
        let (tx, rx) = std::sync::mpsc::channel::<AggregateJob>();
        std::thread::spawn(move || {
            while let Ok(first) = rx.recv() {
                let mut jobs = vec![first];
                let mut rows = jobs[0].requests.len();
                while rows < max_rows {
                    match rx.recv_timeout(gather_window) {
                        Ok(job) => {
                            rows += job.requests.len();
                            jobs.push(job);
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => break,
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                }
                let merged: Vec<Value> = jobs
                    .iter()
                    .flat_map(|job| job.requests.iter().cloned())
                    .collect();
                match session.eval_batch(&merged) {
                    Ok(mut evals) => {
                        for job in jobs {
                            let rest = evals.split_off(job.requests.len());
                            let mine = std::mem::replace(&mut evals, rest);
                            let _ = job.reply.send(Ok(mine));
                        }
                    }
                    Err(error) => {
                        let message = format!("{error:#}");
                        for job in jobs {
                            let _ = job.reply.send(Err(message.clone()));
                        }
                    }
                }
            }
            session.shutdown();
        });
        Ok(Self {
            tx,
            packed_features,
        })
    }

    pub fn client(&self) -> SharedBridgeClient {
        SharedBridgeClient {
            tx: self.tx.clone(),
            packed_features: self.packed_features,
        }
    }
}

impl SharedBridgeClient {
    pub fn supports_packed_features(&self) -> bool {
        self.packed_features
    }

    pub fn eval_batch(&self, requests: &[Value]) -> Result<Vec<ModelEval>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
        self.tx
            .send(AggregateJob {
                requests: requests.to_vec(),
                reply: reply_tx,
            })
            .map_err(|_| anyhow::anyhow!("shared bridge aggregator has shut down"))?;
        reply_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("shared bridge dropped the reply channel"))?
            .map_err(|message| anyhow::anyhow!("shared bridge eval failed: {message}"))
    }
}

#[cfg(test)]
mod shared_tests {
    use super::*;

    fn mock_command() -> String {
        format!(
            "python3 {}/../tests/mock_model_bridge.py",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    fn config() -> BridgeConfig {
        BridgeConfig {
            model_manifest: None,
            model_timeout_ms: 20_000,
            allow_model_fallback: false,
        }
    }

    fn request(tag: usize, actions: usize) -> Value {
        let action_ids: Vec<String> = (0..actions).map(|i| format!("a{tag}-{i}")).collect();
        json!({
            "state_hash": format!("hash-{tag}"),
            "action_ids": action_ids,
            "legal_actions": action_ids.iter().map(|id| json!({"action_id": id})).collect::<Vec<_>>(),
            "exact_afterstate_score_active": (0..actions).map(|i| (tag * 10 + i) as f64).collect::<Vec<_>>(),
        })
    }

    #[test]
    fn shared_bridge_demuxes_concurrent_jobs_correctly() {
        let shared =
            SharedBridge::spawn(&mock_command(), &config(), 64).expect("spawn shared bridge");
        let mut handles = Vec::new();
        for worker in 0..8usize {
            let client = shared.client();
            handles.push(std::thread::spawn(move || {
                for round in 0..5usize {
                    let tag = worker * 100 + round;
                    let actions = 2 + (tag % 4);
                    let requests = vec![request(tag, actions), request(tag + 50, actions)];
                    let evals = client.eval_batch(&requests).expect("shared eval");
                    assert_eq!(evals.len(), 2);
                    // Mock returns q = exact + 1, so alignment errors are
                    // detectable per request.
                    let expected: Vec<f64> =
                        (0..actions).map(|i| (tag * 10 + i) as f64 + 1.0).collect();
                    assert_eq!(evals[0].q.as_ref().expect("q"), &expected);
                    assert_eq!(evals[0].priors.len(), actions);
                }
            }));
        }
        for handle in handles {
            handle.join().expect("worker thread");
        }
    }
}
