use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::{OsStr, OsString},
    fs,
    io::{Read, Write},
    ops::Range,
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
};

use cascadia_data::R2MapProtocolIdentity;
use cascadia_game::{
    Wildlife, public_market_replacement_is_universally_safe,
    public_market_universally_safe_wipe_masks,
};
use cascadia_r2::{
    BOARD_SLOTS, GLOBAL_FEATURES, MARKET_FEATURES, PLAYER_FEATURES, R2_MAP_ACTION_BYTES,
    R2_MAP_BOARD_TOKEN_CAPACITY, R2_MAP_MARKET_ACTION_BYTES, R2_MAP_TOKEN_FEATURES,
    R2MapMarketDecisionKind, R2MapPublicTensors,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;

const BOARD_TOKEN_CAPACITY: usize = R2_MAP_BOARD_TOKEN_CAPACITY;

const PROTOCOL_MAGIC: [u8; 4] = *b"R2MP";
const PROTOCOL_VERSION: u16 = 3;
const MESSAGE_SCORE_GROUPS: u16 = 0x20;
const MESSAGE_SCORE_MARKET_GROUPS: u16 = 0x21;
const MESSAGE_SHUTDOWN: u16 = 0x02;
const MESSAGE_SCORE_RESPONSE: u16 = 0x8020;
const MESSAGE_SCORE_MARKET_RESPONSE: u16 = 0x8021;
const MESSAGE_ERROR: u16 = 0xffff;
const FRAME_HEADER_SIZE: usize = 20;
const MAX_METADATA_BYTES: usize = 8 * 1024 * 1024;
const MAX_TENSOR_BYTES: usize = 1024 * 1024 * 1024;
const MAX_GROUPS: usize = 16;
pub const R2_MAP_PROTOCOL_MAX_CANDIDATES_PER_GROUP: usize = 8_192;
// A complete candidate contributes about 138 KiB of fixed-capacity public
// tensors to an R2MP request. Keep each physical frame below the 1 GiB tensor
// byte ceiling while the versioned protocol continues to accept up to 8,192
// candidates in a frame. Logical screens have no candidate ceiling and are
// transparently partitioned at this transport-only bound.
const R2_MAP_WIRE_FRAME_CANDIDATES: usize = 1_024;
const MAX_TOTAL_CANDIDATES: usize = MAX_GROUPS * R2_MAP_PROTOCOL_MAX_CANDIDATES_PER_GROUP;
const REQUEST_SCHEMA: &str = "r2-map-grouped-exhaustive-request-v3";
const RESPONSE_SCHEMA: &str = "r2-map-grouped-exhaustive-response-v3";
pub const R2_MAP_MARKET_REQUEST_SCHEMA: &str = "r2-map-public-market-decision-request-v3";
pub const R2_MAP_MARKET_RESPONSE_SCHEMA: &str = "r2-map-public-market-decision-response-v2";
// BLAKE3 of the canonical public market request contract manifest shared with
// python/cascadia_mlx/r2_map_serve.py. This is replaced only by a versioned
// contract change, never by host- or checkpoint-local state.
pub const R2_MAP_MARKET_REQUEST_SCHEMA_BLAKE3: &str =
    "68ca4d115e6a2ca5981a75d8c979752efbd8b1d466fb25afa278e9de103e9082";
pub const R2_MAP_MARKET_RESPONSE_SCHEMA_BLAKE3: &str =
    "d0d00527f75b7bbcb868433da7cb9f2cd415d0f9fb4e7591a966e51728c708c5";
const REQUEST_SCHEMA_BLAKE3: &str =
    "bce9b1e6701dd86debc7a0fae496e6e55d72acac554eb572dcdcbf5356b6b8fa";

const EXACT_SCORE_TENSOR: (&str, &str) = ("exact_afterstate_scores", "<f4");
// Live gameplay consumes only the four selection/value tensors. Opponent,
// survival, and wipe heads remain part of training and fixed-panel
// verification, but serializing them here would add tens of MiB to a maximum-
// width response without any inference consumer.
const RESPONSE_TENSORS: [(&str, &str); 4] = [
    ("action_scores", "<f4"),
    ("predicted_score_to_go", "<f4"),
    ("predicted_score_components_to_go", "<f4"),
    ("bootstrap_policy_logits", "<f4"),
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct R2MapModelIdentity {
    pub checkpoint_id: String,
    pub checkpoint_manifest_blake3: String,
    pub model_config_blake3: String,
    pub model_weights_blake3: String,
    pub verification_id: String,
}

impl R2MapModelIdentity {
    pub fn validate(&self) -> Result<(), R2MapModelError> {
        if self.checkpoint_id.is_empty()
            || [
                &self.checkpoint_manifest_blake3,
                &self.model_config_blake3,
                &self.model_weights_blake3,
                &self.verification_id,
            ]
            .into_iter()
            .any(|value| !is_blake3(value))
        {
            return Err(R2MapModelError::InvalidModelIdentity);
        }
        Ok(())
    }
}

pub const R2_MAP_SERVING_BUNDLE_SCHEMA: &str = "r2-map-local-serving-bundle-v2";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct R2MapServingBundleEntry {
    /// Compact policy identity stored in trajectories and opponent pools.
    pub manifest_identity_blake3: String,
    pub run_dir: PathBuf,
    pub checkpoint_path: PathBuf,
    pub model: R2MapModelIdentity,
    #[serde(default)]
    pub pinned: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct R2MapServingBundle {
    pub schema_version: u16,
    pub schema_id: String,
    pub protocols: R2MapProtocolIdentity,
    pub entries: Vec<R2MapServingBundleEntry>,
}

impl R2MapServingBundle {
    pub fn read(path: impl AsRef<Path>) -> Result<Self, R2MapModelError> {
        let value: Self = serde_json::from_slice(&fs::read(path)?)?;
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), R2MapModelError> {
        if self.schema_version != 2
            || self.schema_id != R2_MAP_SERVING_BUNDLE_SCHEMA
            || self.entries.is_empty()
        {
            return Err(R2MapModelError::InvalidServingBundle);
        }
        self.protocols
            .validate()
            .map_err(|_| R2MapModelError::InvalidServingBundle)?;
        let mut compact = std::collections::HashSet::new();
        let mut checkpoints = std::collections::HashSet::new();
        for entry in &self.entries {
            entry.model.validate()?;
            if !is_blake3(&entry.manifest_identity_blake3)
                || !entry.run_dir.is_absolute()
                || !entry.checkpoint_path.is_absolute()
                || !compact.insert(entry.manifest_identity_blake3.as_str())
                || !checkpoints.insert(entry.model.checkpoint_id.as_str())
            {
                return Err(R2MapModelError::InvalidServingBundle);
            }
        }
        Ok(())
    }

    /// Load and cryptographically bind every compact policy identity to a
    /// complete checkpoint, its exact model bytes, and a fixed-panel receipt.
    pub fn read_verified(path: impl AsRef<Path>) -> Result<Self, R2MapModelError> {
        let value = Self::read(path)?;
        for entry in &value.entries {
            verify_serving_entry(entry)?;
        }
        Ok(value)
    }

    pub fn model_for_manifest_identity(
        &self,
        identity: [u8; 32],
    ) -> Result<R2MapModelIdentity, R2MapModelError> {
        let identity = hex_hash(identity);
        self.entries
            .iter()
            .find(|entry| entry.manifest_identity_blake3 == identity)
            .map(|entry| entry.model.clone())
            .ok_or(R2MapModelError::MissingServingModel)
    }
}

fn verify_serving_entry(entry: &R2MapServingBundleEntry) -> Result<(), R2MapModelError> {
    let manifest_path = entry.checkpoint_path.join("checkpoint.json");
    let manifest_bytes = fs::read(&manifest_path)?;
    let mut manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    let checkpoint_name = entry
        .checkpoint_path
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or(R2MapModelError::InvalidServingBundle)?;
    let manifest_object = manifest
        .as_object_mut()
        .ok_or(R2MapModelError::InvalidServingBundle)?;
    let claimed_manifest_identity = manifest_object
        .remove("manifest_identity_blake3")
        .and_then(|value| value.as_str().map(ToOwned::to_owned))
        .ok_or(R2MapModelError::InvalidServingBundle)?;
    if manifest_object
        .get("schema_version")
        .and_then(Value::as_u64)
        != Some(2)
        || manifest_object.get("schema_id").and_then(Value::as_str) != Some("r2-map-checkpoint-v2")
        || manifest_object.get("checkpoint_id").and_then(Value::as_str) != Some(checkpoint_name)
        || entry.model.checkpoint_id != checkpoint_name
        || claimed_manifest_identity != entry.manifest_identity_blake3
        || blake3::hash(&canonical_json(&Value::Object(manifest_object.clone()))?)
            .to_hex()
            .to_string()
            != claimed_manifest_identity
        || blake3::hash(&manifest_bytes).to_hex().to_string()
            != entry.model.checkpoint_manifest_blake3
    {
        return Err(R2MapModelError::InvalidServingBundle);
    }
    let identity = manifest_object
        .get("identity")
        .and_then(Value::as_object)
        .ok_or(R2MapModelError::InvalidServingBundle)?;
    if identity.get("model_config_blake3").and_then(Value::as_str)
        != Some(&entry.model.model_config_blake3)
        || manifest_object
            .get("model_config")
            .map(canonical_json)
            .transpose()?
            .map(|bytes| blake3::hash(&bytes).to_hex().to_string())
            .as_deref()
            != Some(&entry.model.model_config_blake3)
    {
        return Err(R2MapModelError::InvalidServingBundle);
    }
    let files = manifest_object
        .get("files")
        .and_then(Value::as_object)
        .ok_or(R2MapModelError::InvalidServingBundle)?;
    let required = [
        "fixed-prediction-panel.safetensors",
        "model.safetensors",
        "optimizer.safetensors",
        "state.json",
    ];
    if files.len() != required.len() || required.iter().any(|name| !files.contains_key(*name)) {
        return Err(R2MapModelError::InvalidServingBundle);
    }
    for name in required {
        let descriptor = files[name]
            .as_object()
            .ok_or(R2MapModelError::InvalidServingBundle)?;
        let path = entry.checkpoint_path.join(name);
        let metadata = fs::metadata(&path)?;
        let expected_bytes = descriptor
            .get("bytes")
            .and_then(Value::as_u64)
            .ok_or(R2MapModelError::InvalidServingBundle)?;
        let expected_hash = descriptor
            .get("blake3")
            .and_then(Value::as_str)
            .ok_or(R2MapModelError::InvalidServingBundle)?;
        if metadata.len() != expected_bytes || file_blake3(&path)? != expected_hash {
            return Err(R2MapModelError::InvalidServingBundle);
        }
    }
    if files["model.safetensors"]["blake3"].as_str() != Some(&entry.model.model_weights_blake3) {
        return Err(R2MapModelError::InvalidServingBundle);
    }
    let state: Value =
        serde_json::from_slice(&fs::read(entry.checkpoint_path.join("state.json"))?)?;
    let state_object = state
        .as_object()
        .ok_or(R2MapModelError::InvalidServingBundle)?;
    let dataset_contract = state_object
        .get("dataset_contract")
        .ok_or(R2MapModelError::InvalidServingBundle)?;
    let dataset_contract_blake3 = blake3::hash(&canonical_json(dataset_contract)?)
        .to_hex()
        .to_string();
    let next_batch_identity = state_object
        .get("next_batch_identity")
        .and_then(Value::as_str)
        .filter(|identity| !identity.is_empty() && identity.len() <= 256)
        .ok_or(R2MapModelError::InvalidServingBundle)?;
    let receipt_path = entry
        .run_dir
        .join("verifications")
        .join(format!("{checkpoint_name}.json"));
    let mut receipt: Value = serde_json::from_slice(&fs::read(receipt_path)?)?;
    let receipt_object = receipt
        .as_object_mut()
        .ok_or(R2MapModelError::InvalidServingBundle)?;
    let claimed_verification = receipt_object
        .remove("verification_id")
        .and_then(|value| value.as_str().map(ToOwned::to_owned))
        .ok_or(R2MapModelError::InvalidServingBundle)?;
    let receipt_fields = [
        "schema_version",
        "schema_id",
        "checkpoint_id",
        "checkpoint_manifest_blake3",
        "prediction_panel_id",
        "dataset_contract_blake3",
        "prediction_tensor_blake3",
        "loss_stream_offset_bytes",
        "loss_stream_prefix_blake3",
        "exact_prediction_match",
        "next_batch_identity",
        "exact_next_batch_match",
    ];
    let prediction_tensor_digests_valid = receipt_object
        .get("prediction_tensor_blake3")
        .and_then(Value::as_object)
        .is_some_and(|digests| {
            !digests.is_empty()
                && digests
                    .values()
                    .all(|digest| digest.as_str().is_some_and(is_blake3))
        });
    if receipt_object.len() != receipt_fields.len()
        || receipt_fields
            .iter()
            .any(|field| !receipt_object.contains_key(*field))
        || receipt_object.get("schema_version").and_then(Value::as_u64) != Some(2)
        || receipt_object.get("schema_id").and_then(Value::as_str)
            != Some("r2-map-checkpoint-verification-v2")
        || receipt_object.get("checkpoint_id").and_then(Value::as_str) != Some(checkpoint_name)
        || receipt_object
            .get("checkpoint_manifest_blake3")
            .and_then(Value::as_str)
            != Some(&entry.model.checkpoint_manifest_blake3)
        || receipt_object
            .get("prediction_panel_id")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
        || receipt_object
            .get("dataset_contract_blake3")
            .and_then(Value::as_str)
            != Some(dataset_contract_blake3.as_str())
        || !prediction_tensor_digests_valid
        || receipt_object
            .get("loss_stream_offset_bytes")
            .and_then(Value::as_u64)
            .is_none()
        || !receipt_object
            .get("loss_stream_prefix_blake3")
            .and_then(Value::as_str)
            .is_some_and(is_blake3)
        || receipt_object
            .get("exact_prediction_match")
            .and_then(Value::as_bool)
            != Some(true)
        || receipt_object
            .get("next_batch_identity")
            .and_then(Value::as_str)
            != Some(next_batch_identity)
        || receipt_object
            .get("exact_next_batch_match")
            .and_then(Value::as_bool)
            != Some(true)
        || claimed_verification != entry.model.verification_id
        || !is_blake3(&claimed_verification)
        || blake3::hash(&canonical_json(&Value::Object(receipt_object.clone()))?)
            .to_hex()
            .to_string()
            != claimed_verification
    {
        return Err(R2MapModelError::InvalidServingBundle);
    }
    Ok(())
}

fn file_blake3(path: &Path) -> Result<String, R2MapModelError> {
    let mut reader = fs::File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

#[derive(Debug, Clone, PartialEq)]
pub struct R2MapInferenceCandidate {
    pub action_id: [u8; 32],
    pub afterstate: R2MapPublicTensors,
    pub action_bytes: [u8; R2_MAP_ACTION_BYTES],
    pub exact_afterstate_score: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct R2MapInferenceGroup {
    pub group_id: [u8; 32],
    pub decision_id: [u8; 32],
    pub model: R2MapModelIdentity,
    pub parent: R2MapPublicTensors,
    pub candidates: Vec<R2MapInferenceCandidate>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct R2MapPredictionGroup {
    pub group_id: [u8; 32],
    pub decision_id: [u8; 32],
    pub action_ids: Vec<[u8; 32]>,
    pub action_scores: Vec<f32>,
    pub predicted_score_to_go: Vec<f32>,
    pub predicted_score_components_to_go: Vec<[f32; 11]>,
    pub bootstrap_policy_logits: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct R2MapMarketInferenceCandidate {
    pub action_id: [u8; 32],
    pub action_bytes: [u8; R2_MAP_MARKET_ACTION_BYTES],
}

#[derive(Debug, Clone, PartialEq)]
pub struct R2MapMarketInferenceGroup {
    pub group_id: [u8; 32],
    pub decision_id: [u8; 32],
    pub model: R2MapModelIdentity,
    pub parent: R2MapPublicTensors,
    pub exact_current_score: f32,
    pub decision_kind: R2MapMarketDecisionKind,
    pub public_nature_tokens: u8,
    pub public_wildlife_bag_counts: [u8; 5],
    pub public_wildlife_bag_total: u8,
    pub public_market_wildlife: [u8; 4],
    pub candidates: Vec<R2MapMarketInferenceCandidate>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct R2MapMarketPredictionGroup {
    pub group_id: [u8; 32],
    pub decision_id: [u8; 32],
    pub action_ids: Vec<[u8; 32]>,
    pub action_scores: Vec<f32>,
    pub predicted_score_to_go: Vec<f32>,
}

pub struct R2MapModelProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    program: OsString,
    args: Vec<OsString>,
    next_request_id: u32,
    closed: bool,
}

impl R2MapModelProcess {
    pub fn spawn<I, S>(program: impl AsRef<OsStr>, args: I) -> Result<Self, R2MapModelError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let program = program.as_ref().to_os_string();
        let args = args
            .into_iter()
            .map(|value| value.as_ref().to_os_string())
            .collect::<Vec<_>>();
        let (child, stdin, stdout) = spawn_child(&program, &args)?;
        Ok(Self {
            child,
            stdin,
            stdout,
            program,
            args,
            next_request_id: 1,
            closed: false,
        })
    }

    pub fn score_groups(
        &mut self,
        groups: &[R2MapInferenceGroup],
    ) -> Result<Vec<R2MapPredictionGroup>, R2MapModelError> {
        if groups
            .iter()
            .try_fold(0usize, |total, group| {
                total.checked_add(group.candidates.len())
            })
            .is_some_and(|total| total <= R2_MAP_WIRE_FRAME_CANDIDATES)
        {
            return self.score_groups_frame(groups);
        }
        validate_logical_groups(groups)?;
        let mut merged_groups = Vec::with_capacity(groups.len());
        for group in groups {
            let mut merged = R2MapPredictionGroup {
                group_id: group.group_id,
                decision_id: group.decision_id,
                action_ids: Vec::with_capacity(group.candidates.len()),
                action_scores: Vec::with_capacity(group.candidates.len()),
                predicted_score_to_go: Vec::with_capacity(group.candidates.len()),
                predicted_score_components_to_go: Vec::with_capacity(group.candidates.len()),
                bootstrap_policy_logits: Vec::with_capacity(group.candidates.len()),
            };
            for range in logical_frame_ranges(group.candidates.len())? {
                let frame_group = R2MapInferenceGroup {
                    group_id: group.group_id,
                    decision_id: group.decision_id,
                    model: group.model.clone(),
                    parent: group.parent.clone(),
                    candidates: group.candidates[range.clone()].to_vec(),
                };
                let mut predictions = self.score_groups_frame(&[frame_group])?;
                let prediction = predictions.pop().ok_or(R2MapModelError::InvalidResponse(
                    "missing partition response",
                ))?;
                append_partition_prediction(&mut merged, group, range, prediction)?;
            }
            if merged.action_ids.len() != group.candidates.len() {
                return Err(R2MapModelError::InvalidResponse(
                    "partitioned candidate count",
                ));
            }
            merged_groups.push(merged);
        }
        Ok(merged_groups)
    }

    fn score_groups_frame(
        &mut self,
        groups: &[R2MapInferenceGroup],
    ) -> Result<Vec<R2MapPredictionGroup>, R2MapModelError> {
        validate_groups(groups)?;
        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.wrapping_add(1).max(1);
        let (metadata, payload, request_identity) = encode_request(groups)?;
        write_frame(
            &mut self.stdin,
            MESSAGE_SCORE_GROUPS,
            request_id,
            &metadata,
            &payload,
        )?;
        let (message_type, response_id, metadata, payload) = read_frame(&mut self.stdout)?;
        if response_id != request_id {
            return Err(R2MapModelError::RequestIdMismatch {
                expected: request_id,
                actual: response_id,
            });
        }
        if message_type == MESSAGE_ERROR {
            return Err(R2MapModelError::Service(
                String::from_utf8_lossy(&payload).into_owned(),
            ));
        }
        if message_type != MESSAGE_SCORE_RESPONSE {
            return Err(R2MapModelError::InvalidResponse("message type"));
        }
        decode_response(groups, &request_identity, &metadata, &payload)
    }

    pub fn score_market_groups(
        &mut self,
        groups: &[R2MapMarketInferenceGroup],
    ) -> Result<Vec<R2MapMarketPredictionGroup>, R2MapModelError> {
        validate_market_groups(groups)?;
        let request_id = self.next_request_id;
        self.next_request_id = self.next_request_id.wrapping_add(1).max(1);
        let (metadata, payload, request_identity) = encode_market_request(groups)?;
        write_frame(
            &mut self.stdin,
            MESSAGE_SCORE_MARKET_GROUPS,
            request_id,
            &metadata,
            &payload,
        )?;
        let (message_type, response_id, metadata, payload) = read_frame(&mut self.stdout)?;
        if response_id != request_id {
            return Err(R2MapModelError::RequestIdMismatch {
                expected: request_id,
                actual: response_id,
            });
        }
        if message_type == MESSAGE_ERROR {
            return Err(R2MapModelError::Service(
                String::from_utf8_lossy(&payload).into_owned(),
            ));
        }
        if message_type != MESSAGE_SCORE_MARKET_RESPONSE {
            return Err(R2MapModelError::InvalidResponse("market message type"));
        }
        decode_market_response(groups, &request_identity, &metadata, &payload)
    }

    pub fn restart(&mut self) -> Result<(), R2MapModelError> {
        self.terminate();
        let (child, stdin, stdout) = spawn_child(&self.program, &self.args)?;
        self.child = child;
        self.stdin = stdin;
        self.stdout = stdout;
        self.next_request_id = 1;
        self.closed = false;
        Ok(())
    }

    pub fn shutdown(mut self) -> Result<(), R2MapModelError> {
        let request_id = self.next_request_id;
        self.stdin
            .write_all(&frame_header(MESSAGE_SHUTDOWN, request_id, 0, 0))?;
        self.stdin.flush()?;
        let status = self.child.wait()?;
        self.closed = true;
        if status.success() {
            Ok(())
        } else {
            Err(R2MapModelError::ProcessExit(status.code()))
        }
    }

    fn terminate(&mut self) {
        if !self.closed {
            let _ = self.child.kill();
            let _ = self.child.wait();
            self.closed = true;
        }
    }
}

fn logical_frame_ranges(candidate_count: usize) -> Result<Vec<Range<usize>>, R2MapModelError> {
    if candidate_count == 0 {
        return Err(R2MapModelError::InvalidCandidateCount(0));
    }
    Ok((0..candidate_count)
        .step_by(R2_MAP_WIRE_FRAME_CANDIDATES)
        .map(|start| {
            start
                ..candidate_count.min(
                    start
                        .checked_add(R2_MAP_WIRE_FRAME_CANDIDATES)
                        .expect("wire frame width cannot overflow usize"),
                )
        })
        .collect())
}

fn append_partition_prediction(
    merged: &mut R2MapPredictionGroup,
    group: &R2MapInferenceGroup,
    range: Range<usize>,
    prediction: R2MapPredictionGroup,
) -> Result<(), R2MapModelError> {
    let count = range.len();
    let expected_ids = group.candidates[range]
        .iter()
        .map(|candidate| candidate.action_id)
        .collect::<Vec<_>>();
    if prediction.group_id != group.group_id
        || prediction.decision_id != group.decision_id
        || prediction.action_ids != expected_ids
        || prediction.action_scores.len() != count
        || prediction.predicted_score_to_go.len() != count
        || prediction.predicted_score_components_to_go.len() != count
        || prediction.bootstrap_policy_logits.len() != count
    {
        return Err(R2MapModelError::InvalidResponse(
            "partition response identity or width",
        ));
    }
    merged.action_ids.extend(prediction.action_ids);
    merged.action_scores.extend(prediction.action_scores);
    merged
        .predicted_score_to_go
        .extend(prediction.predicted_score_to_go);
    merged
        .predicted_score_components_to_go
        .extend(prediction.predicted_score_components_to_go);
    merged
        .bootstrap_policy_logits
        .extend(prediction.bootstrap_policy_logits);
    Ok(())
}

impl Drop for R2MapModelProcess {
    fn drop(&mut self) {
        self.terminate();
    }
}

fn spawn_child(
    program: &OsStr,
    args: &[OsString],
) -> Result<(Child, ChildStdin, ChildStdout), R2MapModelError> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let stdin = child
        .stdin
        .take()
        .ok_or(R2MapModelError::MissingPipe("stdin"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or(R2MapModelError::MissingPipe("stdout"))?;
    Ok((child, stdin, stdout))
}

fn validate_groups(groups: &[R2MapInferenceGroup]) -> Result<(), R2MapModelError> {
    if groups.is_empty() || groups.len() > MAX_GROUPS {
        return Err(R2MapModelError::InvalidGroupCount(groups.len()));
    }
    let mut total = 0usize;
    let mut group_ids = std::collections::HashSet::new();
    for group in groups {
        group.model.validate()?;
        if !group_ids.insert(group.group_id) {
            return Err(R2MapModelError::DuplicateGroupIdentity);
        }
        if group.candidates.is_empty()
            || group.candidates.len() > R2_MAP_PROTOCOL_MAX_CANDIDATES_PER_GROUP
        {
            return Err(R2MapModelError::InvalidCandidateCount(
                group.candidates.len(),
            ));
        }
        total = total
            .checked_add(group.candidates.len())
            .ok_or(R2MapModelError::InvalidCandidateCount(usize::MAX))?;
        let mut actions = std::collections::HashSet::new();
        for candidate in &group.candidates {
            if !actions.insert(candidate.action_id) || !candidate.exact_afterstate_score.is_finite()
            {
                return Err(R2MapModelError::InvalidCandidateIdentity);
            }
            validate_public_tensors(&candidate.afterstate)?;
        }
        validate_public_tensors(&group.parent)?;
    }
    if total > MAX_TOTAL_CANDIDATES {
        return Err(R2MapModelError::InvalidCandidateCount(total));
    }
    Ok(())
}

fn validate_logical_groups(groups: &[R2MapInferenceGroup]) -> Result<(), R2MapModelError> {
    if groups.is_empty() || groups.len() > MAX_GROUPS {
        return Err(R2MapModelError::InvalidGroupCount(groups.len()));
    }
    let mut group_ids = std::collections::HashSet::new();
    for group in groups {
        group.model.validate()?;
        if group.candidates.is_empty() || !group_ids.insert(group.group_id) {
            return Err(if group.candidates.is_empty() {
                R2MapModelError::InvalidCandidateCount(0)
            } else {
                R2MapModelError::DuplicateGroupIdentity
            });
        }
        validate_public_tensors(&group.parent)?;
        let mut actions = std::collections::HashSet::with_capacity(group.candidates.len());
        for candidate in &group.candidates {
            if !actions.insert(candidate.action_id) || !candidate.exact_afterstate_score.is_finite()
            {
                return Err(R2MapModelError::InvalidCandidateIdentity);
            }
            validate_public_tensors(&candidate.afterstate)?;
        }
    }
    Ok(())
}

fn validate_market_groups(groups: &[R2MapMarketInferenceGroup]) -> Result<(), R2MapModelError> {
    if groups.is_empty() || groups.len() > MAX_GROUPS {
        return Err(R2MapModelError::InvalidGroupCount(groups.len()));
    }
    let mut total = 0usize;
    let mut group_ids = std::collections::HashSet::new();
    for group in groups {
        group.model.validate()?;
        validate_public_tensors(&group.parent)?;
        if !group.exact_current_score.is_finite() || !group_ids.insert(group.group_id) {
            return Err(R2MapModelError::InvalidCandidateIdentity);
        }
        if group.candidates.is_empty()
            || group.candidates.len() > R2_MAP_PROTOCOL_MAX_CANDIDATES_PER_GROUP
        {
            return Err(R2MapModelError::InvalidCandidateCount(
                group.candidates.len(),
            ));
        }
        total = total
            .checked_add(group.candidates.len())
            .ok_or(R2MapModelError::InvalidCandidateCount(usize::MAX))?;
        let mut actions = std::collections::HashSet::new();
        for candidate in &group.candidates {
            if !actions.insert(candidate.action_id)
                || candidate.action_bytes[0] != 1
                || candidate.action_bytes[4..] != [0; 4]
            {
                return Err(R2MapModelError::InvalidCandidateIdentity);
            }
        }
        let bag_total = group
            .public_wildlife_bag_counts
            .into_iter()
            .try_fold(0u8, u8::checked_add)
            .ok_or(R2MapModelError::InvalidCandidateIdentity)?;
        if bag_total != group.public_wildlife_bag_total {
            return Err(R2MapModelError::InvalidCandidateIdentity);
        }
        let expected = expected_market_action_bytes(
            group.decision_kind,
            group.public_nature_tokens,
            group.public_wildlife_bag_counts,
            group.public_market_wildlife,
        )?;
        if group
            .candidates
            .iter()
            .map(|candidate| candidate.action_bytes)
            .ne(expected)
        {
            return Err(R2MapModelError::InvalidCandidateIdentity);
        }
    }
    if total > MAX_TOTAL_CANDIDATES {
        return Err(R2MapModelError::InvalidCandidateCount(total));
    }
    Ok(())
}

fn expected_market_action_bytes(
    kind: R2MapMarketDecisionKind,
    public_nature_tokens: u8,
    public_wildlife_bag_counts: [u8; 5],
    public_market_wildlife: [u8; 4],
) -> Result<Vec<[u8; R2_MAP_MARKET_ACTION_BYTES]>, R2MapModelError> {
    let mut market = [Wildlife::Bear; 4];
    for (index, encoded) in public_market_wildlife.into_iter().enumerate() {
        market[index] = *Wildlife::ALL
            .get(usize::from(encoded))
            .ok_or(R2MapModelError::InvalidCandidateIdentity)?;
    }
    if Wildlife::ALL
        .into_iter()
        .any(|wildlife| market.iter().all(|shown| *shown == wildlife))
    {
        return Err(R2MapModelError::InvalidCandidateIdentity);
    }
    let mut expected = Vec::with_capacity(16);
    match kind {
        R2MapMarketDecisionKind::FreeThreeOfAKind => {
            expected.push([1, 0, 0, 0, 0, 0, 0, 0]);
            let replacement_species = Wildlife::ALL
                .into_iter()
                .find(|wildlife| market.iter().filter(|shown| *shown == wildlife).count() == 3);
            let replacement_species =
                replacement_species.ok_or(R2MapModelError::InvalidCandidateIdentity)?;
            let slot_mask = market
                .iter()
                .enumerate()
                .filter(|(_, wildlife)| **wildlife == replacement_species)
                .fold(0u8, |mask, (slot, _)| mask | (1 << slot));
            if public_market_replacement_is_universally_safe(
                public_wildlife_bag_counts,
                market,
                slot_mask,
            ) {
                expected.push([1, 0, 1, 0, 0, 0, 0, 0]);
            }
        }
        R2MapMarketDecisionKind::PaidWipes => {
            expected.push([1, 1, 2, 0, 0, 0, 0, 0]);
            if public_nature_tokens > 0 {
                expected.extend(
                    public_market_universally_safe_wipe_masks(public_wildlife_bag_counts, market)
                        .into_iter()
                        .map(|mask| [1, 1, 3, mask, 0, 0, 0, 0]),
                );
            }
        }
    }
    Ok(expected)
}

fn validate_public_tensors(value: &R2MapPublicTensors) -> Result<(), R2MapModelError> {
    if value.token_features.len() != BOARD_SLOTS * BOARD_TOKEN_CAPACITY * R2_MAP_TOKEN_FEATURES
        || value.token_types.len() != BOARD_SLOTS * BOARD_TOKEN_CAPACITY
        || value.token_mask.len() != BOARD_SLOTS * BOARD_TOKEN_CAPACITY
        || value.market_features.len() != 4 * MARKET_FEATURES
        || value.player_features.len() != BOARD_SLOTS * PLAYER_FEATURES
        || value.token_mask.iter().any(|item| *item > 1)
        || value.market_mask.iter().any(|item| *item > 1)
        || value.player_mask.iter().any(|item| *item > 1)
    {
        return Err(R2MapModelError::InvalidTensorShape("public state"));
    }
    if value
        .token_features
        .iter()
        .chain(&value.market_features)
        .chain(&value.player_features)
        .chain(&value.global_features)
        .any(|item| !item.is_finite())
    {
        return Err(R2MapModelError::NonFiniteTensor("public state"));
    }
    Ok(())
}

fn encode_request(
    groups: &[R2MapInferenceGroup],
) -> Result<(Vec<u8>, Vec<u8>, String), R2MapModelError> {
    let counts = groups
        .iter()
        .map(|group| group.candidates.len())
        .collect::<Vec<_>>();
    let total = counts.iter().sum::<usize>();
    let mut payload = Vec::new();
    let mut descriptors = Vec::new();
    let mut offsets = vec![0i32];
    for count in &counts {
        offsets.push(
            offsets.last().copied().expect("offset zero")
                + i32::try_from(*count)
                    .map_err(|_| R2MapModelError::InvalidCandidateCount(*count))?,
        );
    }
    append_tensor_i32(
        &mut payload,
        &mut descriptors,
        "candidate_offsets",
        &[groups.len() + 1],
        &offsets,
    );
    append_parent_tensors(&mut payload, &mut descriptors, groups)?;
    append_candidate_tensors(&mut payload, &mut descriptors, groups, total)?;

    let metadata_groups = groups
        .iter()
        .map(|group| {
            let action_ids = group
                .candidates
                .iter()
                .map(|candidate| hex_hash(candidate.action_id))
                .collect::<Vec<_>>();
            json!({
                "group_id": hex_hash(group.group_id),
                "decision_id": hex_hash(group.decision_id),
                "model": group.model,
                "expected_legal_action_count": group.candidates.len(),
                "action_ids": action_ids,
                "enumeration_indices": (0..group.candidates.len()).collect::<Vec<_>>(),
                "ordered_action_ids_blake3": ordered_r2_map_action_ids_blake3(&group.candidates.iter().map(|candidate| candidate.action_id).collect::<Vec<_>>()),
            })
        })
        .collect::<Vec<_>>();
    let base = json!({
        "schema_version": 1,
        "schema_id": REQUEST_SCHEMA,
        "request_schema_blake3": REQUEST_SCHEMA_BLAKE3,
        "group_count": groups.len(),
        "candidate_count": total,
        "groups": metadata_groups,
    });
    let request_identity = blake3::hash(&canonical_json(&base)?).to_hex().to_string();
    let mut envelope = base
        .as_object()
        .expect("request metadata is an object")
        .clone();
    envelope.insert("tensors".to_owned(), serde_json::to_value(descriptors)?);
    envelope.insert(
        "tensor_payload_blake3".to_owned(),
        Value::String(blake3::hash(&payload).to_hex().to_string()),
    );
    Ok((
        canonical_json(&Value::Object(envelope))?,
        payload,
        request_identity,
    ))
}

fn encode_market_request(
    groups: &[R2MapMarketInferenceGroup],
) -> Result<(Vec<u8>, Vec<u8>, String), R2MapModelError> {
    let counts = groups
        .iter()
        .map(|group| group.candidates.len())
        .collect::<Vec<_>>();
    let total = counts.iter().sum::<usize>();
    let mut payload = Vec::new();
    let mut descriptors = Vec::new();
    let mut offsets = vec![0i32];
    for count in &counts {
        offsets.push(
            offsets.last().copied().expect("offset zero")
                + i32::try_from(*count)
                    .map_err(|_| R2MapModelError::InvalidCandidateCount(*count))?,
        );
    }
    append_tensor_i32(
        &mut payload,
        &mut descriptors,
        "action_offsets",
        &[groups.len() + 1],
        &offsets,
    );
    append_market_parent_tensors(&mut payload, &mut descriptors, groups)?;
    append_tensor_u8(
        &mut payload,
        &mut descriptors,
        "action_bytes",
        &[total, R2_MAP_MARKET_ACTION_BYTES],
        groups
            .iter()
            .flat_map(|group| &group.candidates)
            .flat_map(|candidate| candidate.action_bytes),
    );
    append_tensor_f32(
        &mut payload,
        &mut descriptors,
        "exact_current_scores",
        &[groups.len()],
        groups.iter().map(|group| group.exact_current_score),
    );

    let metadata_groups = groups
        .iter()
        .map(|group| {
            let action_ids = group
                .candidates
                .iter()
                .map(|candidate| hex_hash(candidate.action_id))
                .collect::<Vec<_>>();
            json!({
                "group_id": hex_hash(group.group_id),
                "decision_id": hex_hash(group.decision_id),
                "model": group.model,
                "expected_legal_action_count": group.candidates.len(),
                "action_ids": action_ids,
                "enumeration_indices": (0..group.candidates.len()).collect::<Vec<_>>(),
                "ordered_action_ids_blake3": ordered_market_action_ids_blake3(group),
                "public_nature_tokens": group.public_nature_tokens,
                "public_wildlife_bag_counts": group.public_wildlife_bag_counts,
                "public_wildlife_bag_total": group.public_wildlife_bag_total,
                "public_market_wildlife": group.public_market_wildlife,
                "decision_kind": group.decision_kind as u8,
            })
        })
        .collect::<Vec<_>>();
    let base = json!({
        "schema_version": 1,
        "schema_id": R2_MAP_MARKET_REQUEST_SCHEMA,
        "request_schema_blake3": R2_MAP_MARKET_REQUEST_SCHEMA_BLAKE3,
        "group_count": groups.len(),
        "action_count": total,
        "groups": metadata_groups,
    });
    let request_identity = blake3::hash(&canonical_json(&base)?).to_hex().to_string();
    let mut envelope = base
        .as_object()
        .expect("market request metadata is an object")
        .clone();
    envelope.insert("tensors".to_owned(), serde_json::to_value(descriptors)?);
    envelope.insert(
        "tensor_payload_blake3".to_owned(),
        Value::String(blake3::hash(&payload).to_hex().to_string()),
    );
    Ok((
        canonical_json(&Value::Object(envelope))?,
        payload,
        request_identity,
    ))
}

fn append_parent_tensors(
    payload: &mut Vec<u8>,
    descriptors: &mut Vec<TensorDescriptor>,
    groups: &[R2MapInferenceGroup],
) -> Result<(), R2MapModelError> {
    append_tensor_f32(
        payload,
        descriptors,
        "parent_token_features",
        &[
            groups.len(),
            BOARD_SLOTS,
            BOARD_TOKEN_CAPACITY,
            R2_MAP_TOKEN_FEATURES,
        ],
        groups
            .iter()
            .flat_map(|group| group.parent.token_features.iter().copied()),
    );
    append_tensor_i32(
        payload,
        descriptors,
        "parent_token_types",
        &[groups.len(), BOARD_SLOTS, BOARD_TOKEN_CAPACITY],
        &groups
            .iter()
            .flat_map(|group| group.parent.token_types.iter().copied())
            .collect::<Vec<_>>(),
    );
    append_tensor_u8(
        payload,
        descriptors,
        "parent_token_mask",
        &[groups.len(), BOARD_SLOTS, BOARD_TOKEN_CAPACITY],
        groups
            .iter()
            .flat_map(|group| group.parent.token_mask.iter().copied()),
    );
    append_tensor_f32(
        payload,
        descriptors,
        "parent_market_features",
        &[groups.len(), 4, MARKET_FEATURES],
        groups
            .iter()
            .flat_map(|group| group.parent.market_features.iter().copied()),
    );
    append_tensor_u8(
        payload,
        descriptors,
        "parent_market_mask",
        &[groups.len(), 4],
        groups.iter().flat_map(|group| group.parent.market_mask),
    );
    append_tensor_f32(
        payload,
        descriptors,
        "parent_player_features",
        &[groups.len(), BOARD_SLOTS, PLAYER_FEATURES],
        groups
            .iter()
            .flat_map(|group| group.parent.player_features.iter().copied()),
    );
    append_tensor_u8(
        payload,
        descriptors,
        "parent_player_mask",
        &[groups.len(), BOARD_SLOTS],
        groups.iter().flat_map(|group| group.parent.player_mask),
    );
    append_tensor_f32(
        payload,
        descriptors,
        "parent_global_features",
        &[groups.len(), GLOBAL_FEATURES],
        groups.iter().flat_map(|group| group.parent.global_features),
    );
    Ok(())
}

fn append_market_parent_tensors(
    payload: &mut Vec<u8>,
    descriptors: &mut Vec<TensorDescriptor>,
    groups: &[R2MapMarketInferenceGroup],
) -> Result<(), R2MapModelError> {
    append_tensor_f32(
        payload,
        descriptors,
        "parent_token_features",
        &[
            groups.len(),
            BOARD_SLOTS,
            BOARD_TOKEN_CAPACITY,
            R2_MAP_TOKEN_FEATURES,
        ],
        groups
            .iter()
            .flat_map(|group| group.parent.token_features.iter().copied()),
    );
    append_tensor_i32(
        payload,
        descriptors,
        "parent_token_types",
        &[groups.len(), BOARD_SLOTS, BOARD_TOKEN_CAPACITY],
        &groups
            .iter()
            .flat_map(|group| group.parent.token_types.iter().copied())
            .collect::<Vec<_>>(),
    );
    append_tensor_u8(
        payload,
        descriptors,
        "parent_token_mask",
        &[groups.len(), BOARD_SLOTS, BOARD_TOKEN_CAPACITY],
        groups
            .iter()
            .flat_map(|group| group.parent.token_mask.iter().copied()),
    );
    append_tensor_f32(
        payload,
        descriptors,
        "parent_market_features",
        &[groups.len(), 4, MARKET_FEATURES],
        groups
            .iter()
            .flat_map(|group| group.parent.market_features.iter().copied()),
    );
    append_tensor_u8(
        payload,
        descriptors,
        "parent_market_mask",
        &[groups.len(), 4],
        groups.iter().flat_map(|group| group.parent.market_mask),
    );
    append_tensor_f32(
        payload,
        descriptors,
        "parent_player_features",
        &[groups.len(), BOARD_SLOTS, PLAYER_FEATURES],
        groups
            .iter()
            .flat_map(|group| group.parent.player_features.iter().copied()),
    );
    append_tensor_u8(
        payload,
        descriptors,
        "parent_player_mask",
        &[groups.len(), BOARD_SLOTS],
        groups.iter().flat_map(|group| group.parent.player_mask),
    );
    append_tensor_f32(
        payload,
        descriptors,
        "parent_global_features",
        &[groups.len(), GLOBAL_FEATURES],
        groups.iter().flat_map(|group| group.parent.global_features),
    );
    Ok(())
}

fn append_candidate_tensors(
    payload: &mut Vec<u8>,
    descriptors: &mut Vec<TensorDescriptor>,
    groups: &[R2MapInferenceGroup],
    total: usize,
) -> Result<(), R2MapModelError> {
    let candidates = groups
        .iter()
        .flat_map(|group| &group.candidates)
        .collect::<Vec<_>>();
    append_tensor_f32(
        payload,
        descriptors,
        "candidate_token_features",
        &[
            total,
            BOARD_SLOTS,
            BOARD_TOKEN_CAPACITY,
            R2_MAP_TOKEN_FEATURES,
        ],
        candidates
            .iter()
            .flat_map(|candidate| candidate.afterstate.token_features.iter().copied()),
    );
    append_tensor_i32(
        payload,
        descriptors,
        "candidate_token_types",
        &[total, BOARD_SLOTS, BOARD_TOKEN_CAPACITY],
        &candidates
            .iter()
            .flat_map(|candidate| candidate.afterstate.token_types.iter().copied())
            .collect::<Vec<_>>(),
    );
    append_tensor_u8(
        payload,
        descriptors,
        "candidate_token_mask",
        &[total, BOARD_SLOTS, BOARD_TOKEN_CAPACITY],
        candidates
            .iter()
            .flat_map(|candidate| candidate.afterstate.token_mask.iter().copied()),
    );
    append_tensor_f32(
        payload,
        descriptors,
        "candidate_market_features",
        &[total, 4, MARKET_FEATURES],
        candidates
            .iter()
            .flat_map(|candidate| candidate.afterstate.market_features.iter().copied()),
    );
    append_tensor_u8(
        payload,
        descriptors,
        "candidate_market_mask",
        &[total, 4],
        candidates
            .iter()
            .flat_map(|candidate| candidate.afterstate.market_mask),
    );
    append_tensor_f32(
        payload,
        descriptors,
        "candidate_player_features",
        &[total, BOARD_SLOTS, PLAYER_FEATURES],
        candidates
            .iter()
            .flat_map(|candidate| candidate.afterstate.player_features.iter().copied()),
    );
    append_tensor_u8(
        payload,
        descriptors,
        "candidate_player_mask",
        &[total, BOARD_SLOTS],
        candidates
            .iter()
            .flat_map(|candidate| candidate.afterstate.player_mask),
    );
    append_tensor_f32(
        payload,
        descriptors,
        "candidate_global_features",
        &[total, GLOBAL_FEATURES],
        candidates
            .iter()
            .flat_map(|candidate| candidate.afterstate.global_features),
    );
    append_tensor_u8(
        payload,
        descriptors,
        "action_bytes",
        &[total, R2_MAP_ACTION_BYTES],
        candidates
            .iter()
            .flat_map(|candidate| candidate.action_bytes),
    );
    append_tensor_f32(
        payload,
        descriptors,
        EXACT_SCORE_TENSOR.0,
        &[total],
        candidates
            .iter()
            .map(|candidate| candidate.exact_afterstate_score),
    );
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TensorDescriptor {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    offset: usize,
    bytes: usize,
    blake3: String,
}

fn append_tensor_f32(
    payload: &mut Vec<u8>,
    descriptors: &mut Vec<TensorDescriptor>,
    name: &str,
    shape: &[usize],
    values: impl IntoIterator<Item = f32>,
) {
    let start = payload.len();
    for value in values {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    push_descriptor(payload, descriptors, name, "<f4", shape, start);
}

fn append_tensor_i32(
    payload: &mut Vec<u8>,
    descriptors: &mut Vec<TensorDescriptor>,
    name: &str,
    shape: &[usize],
    values: &[i32],
) {
    let start = payload.len();
    for value in values {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    push_descriptor(payload, descriptors, name, "<i4", shape, start);
}

fn append_tensor_u8(
    payload: &mut Vec<u8>,
    descriptors: &mut Vec<TensorDescriptor>,
    name: &str,
    shape: &[usize],
    values: impl IntoIterator<Item = u8>,
) {
    let start = payload.len();
    payload.extend(values);
    push_descriptor(payload, descriptors, name, "|u1", shape, start);
}

fn push_descriptor(
    payload: &[u8],
    descriptors: &mut Vec<TensorDescriptor>,
    name: &str,
    dtype: &str,
    shape: &[usize],
    start: usize,
) {
    let bytes = payload.len() - start;
    descriptors.push(TensorDescriptor {
        name: name.to_owned(),
        dtype: dtype.to_owned(),
        shape: shape.to_vec(),
        offset: start,
        bytes,
        blake3: blake3::hash(&payload[start..]).to_hex().to_string(),
    });
}

fn decode_response(
    groups: &[R2MapInferenceGroup],
    expected_request_identity: &str,
    metadata_bytes: &[u8],
    payload: &[u8],
) -> Result<Vec<R2MapPredictionGroup>, R2MapModelError> {
    let mut metadata: Value = serde_json::from_slice(metadata_bytes)?;
    let object = metadata
        .as_object_mut()
        .ok_or(R2MapModelError::InvalidResponse("metadata"))?;
    let descriptors: Vec<TensorDescriptor> = serde_json::from_value(
        object
            .remove("tensors")
            .ok_or(R2MapModelError::InvalidResponse("tensor descriptors"))?,
    )?;
    let payload_hash = object
        .remove("tensor_payload_blake3")
        .and_then(|value| value.as_str().map(ToOwned::to_owned))
        .ok_or(R2MapModelError::InvalidResponse("payload hash"))?;
    if payload_hash != blake3::hash(payload).to_hex().to_string() {
        return Err(R2MapModelError::InvalidResponse("payload checksum"));
    }
    let total = groups
        .iter()
        .map(|group| group.candidates.len())
        .sum::<usize>();
    validate_response_metadata(groups, expected_request_identity, &metadata, total)?;
    let tensors = validate_response_tensors(&descriptors, payload, total)?;
    let action_scores = tensor_f32(tensors["action_scores"])?;
    let to_go = tensor_f32(tensors["predicted_score_to_go"])?;
    let components = tensor_f32(tensors["predicted_score_components_to_go"])?;
    let policy = tensor_f32(tensors["bootstrap_policy_logits"])?;
    let mut output = Vec::with_capacity(groups.len());
    let mut start = 0;
    for group in groups {
        let stop = start + group.candidates.len();
        output.push(R2MapPredictionGroup {
            group_id: group.group_id,
            decision_id: group.decision_id,
            action_ids: group
                .candidates
                .iter()
                .map(|candidate| candidate.action_id)
                .collect(),
            action_scores: action_scores[start..stop].to_vec(),
            predicted_score_to_go: to_go[start..stop].to_vec(),
            predicted_score_components_to_go: components[start * 11..stop * 11]
                .chunks_exact(11)
                .map(|row| row.try_into().expect("component width"))
                .collect(),
            bootstrap_policy_logits: policy[start..stop].to_vec(),
        });
        start = stop;
    }
    Ok(output)
}

fn decode_market_response(
    groups: &[R2MapMarketInferenceGroup],
    expected_request_identity: &str,
    metadata_bytes: &[u8],
    payload: &[u8],
) -> Result<Vec<R2MapMarketPredictionGroup>, R2MapModelError> {
    let mut metadata: Value = serde_json::from_slice(metadata_bytes)?;
    let object = metadata
        .as_object_mut()
        .ok_or(R2MapModelError::InvalidResponse("market metadata"))?;
    let descriptors: Vec<TensorDescriptor> =
        serde_json::from_value(object.remove("tensors").ok_or(
            R2MapModelError::InvalidResponse("market tensor descriptors"),
        )?)?;
    let payload_hash = object
        .remove("tensor_payload_blake3")
        .and_then(|value| value.as_str().map(ToOwned::to_owned))
        .ok_or(R2MapModelError::InvalidResponse("market payload hash"))?;
    if payload_hash != blake3::hash(payload).to_hex().to_string() {
        return Err(R2MapModelError::InvalidResponse("market payload checksum"));
    }
    let total = groups
        .iter()
        .map(|group| group.candidates.len())
        .sum::<usize>();
    validate_market_response_metadata(groups, expected_request_identity, &metadata, total)?;
    let tensors = validate_market_response_tensors(&descriptors, payload, total)?;
    let action_scores = tensor_f32(tensors["market_action_scores"])?;
    let to_go = tensor_f32(tensors["market_predicted_score_to_go"])?;
    let mut output = Vec::with_capacity(groups.len());
    let mut start = 0usize;
    for group in groups {
        let stop = start + group.candidates.len();
        output.push(R2MapMarketPredictionGroup {
            group_id: group.group_id,
            decision_id: group.decision_id,
            action_ids: group
                .candidates
                .iter()
                .map(|candidate| candidate.action_id)
                .collect(),
            action_scores: action_scores[start..stop].to_vec(),
            predicted_score_to_go: to_go[start..stop].to_vec(),
        });
        start = stop;
    }
    Ok(output)
}

fn validate_market_response_metadata(
    groups: &[R2MapMarketInferenceGroup],
    expected_request_identity: &str,
    metadata: &Value,
    total: usize,
) -> Result<(), R2MapModelError> {
    let object = metadata
        .as_object()
        .ok_or(R2MapModelError::InvalidResponse("market metadata object"))?;
    let expected_top = BTreeSet::from([
        "action_count",
        "diagnostics",
        "group_count",
        "groups",
        "request_identity_blake3",
        "request_schema_blake3",
        "response_schema_blake3",
        "schema_id",
        "schema_version",
    ]);
    if object.keys().map(String::as_str).collect::<BTreeSet<_>>() != expected_top {
        return Err(R2MapModelError::InvalidResponse(
            "market response field set",
        ));
    }
    if object.get("schema_version").and_then(Value::as_u64) != Some(1)
        || object.get("schema_id").and_then(Value::as_str) != Some(R2_MAP_MARKET_RESPONSE_SCHEMA)
        || object.get("request_schema_blake3").and_then(Value::as_str)
            != Some(R2_MAP_MARKET_REQUEST_SCHEMA_BLAKE3)
        || object.get("response_schema_blake3").and_then(Value::as_str)
            != Some(R2_MAP_MARKET_RESPONSE_SCHEMA_BLAKE3)
        || object.get("group_count").and_then(Value::as_u64) != Some(groups.len() as u64)
        || object.get("action_count").and_then(Value::as_u64) != Some(total as u64)
        || object
            .get("request_identity_blake3")
            .and_then(Value::as_str)
            != Some(expected_request_identity)
    {
        return Err(R2MapModelError::InvalidResponse("market response identity"));
    }
    let response_groups = object
        .get("groups")
        .and_then(Value::as_array)
        .ok_or(R2MapModelError::InvalidResponse("market response groups"))?;
    if response_groups.len() != groups.len() {
        return Err(R2MapModelError::InvalidResponse(
            "market response group count",
        ));
    }
    let mut offset = 0usize;
    for (source, response) in groups.iter().zip(response_groups) {
        let response = response
            .as_object()
            .ok_or(R2MapModelError::InvalidResponse("market response group"))?;
        let expected_group = BTreeSet::from([
            "action_count",
            "action_ids",
            "action_offset",
            "decision_id",
            "decision_kind",
            "diagnostics",
            "group_id",
            "model",
            "ordered_action_ids_blake3",
            "public_market_wildlife",
            "public_nature_tokens",
            "public_wildlife_bag_counts",
            "public_wildlife_bag_total",
        ]);
        if response.keys().map(String::as_str).collect::<BTreeSet<_>>() != expected_group {
            return Err(R2MapModelError::InvalidResponse(
                "market response group field set",
            ));
        }
        let ids = source
            .candidates
            .iter()
            .map(|candidate| hex_hash(candidate.action_id))
            .collect::<Vec<_>>();
        let group_diagnostics = response
            .get("diagnostics")
            .and_then(Value::as_object)
            .ok_or(R2MapModelError::InvalidResponse(
                "market response group diagnostics",
            ))?;
        let expected_group_diagnostics = BTreeSet::from([
            "actions_enumerated",
            "actions_scored",
            "complete_cardinality",
            "hidden_refill_inputs",
        ]);
        if group_diagnostics
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            != expected_group_diagnostics
        {
            return Err(R2MapModelError::InvalidResponse(
                "market response group diagnostics field set",
            ));
        }
        if response.get("group_id").and_then(Value::as_str) != Some(&hex_hash(source.group_id))
            || response.get("decision_id").and_then(Value::as_str)
                != Some(&hex_hash(source.decision_id))
            || response.get("action_offset").and_then(Value::as_u64) != Some(offset as u64)
            || response.get("action_count").and_then(Value::as_u64)
                != Some(source.candidates.len() as u64)
            || response.get("action_ids") != Some(&serde_json::to_value(&ids)?)
            || response
                .get("ordered_action_ids_blake3")
                .and_then(Value::as_str)
                != Some(&ordered_market_action_ids_blake3(source))
            || response.get("model") != Some(&serde_json::to_value(&source.model)?)
            || response.get("decision_kind").and_then(Value::as_u64)
                != Some(source.decision_kind as u64)
            || response.get("public_nature_tokens").and_then(Value::as_u64)
                != Some(u64::from(source.public_nature_tokens))
            || response.get("public_wildlife_bag_counts")
                != Some(&serde_json::to_value(source.public_wildlife_bag_counts)?)
            || response
                .get("public_wildlife_bag_total")
                .and_then(Value::as_u64)
                != Some(u64::from(source.public_wildlife_bag_total))
            || response.get("public_market_wildlife")
                != Some(&serde_json::to_value(source.public_market_wildlife)?)
            || response
                .get("diagnostics")
                .and_then(Value::as_object)
                .and_then(|value| value.get("actions_enumerated"))
                .and_then(Value::as_u64)
                != Some(source.candidates.len() as u64)
            || response
                .get("diagnostics")
                .and_then(Value::as_object)
                .and_then(|value| value.get("actions_scored"))
                .and_then(Value::as_u64)
                != Some(source.candidates.len() as u64)
            || response
                .get("diagnostics")
                .and_then(Value::as_object)
                .and_then(|value| value.get("complete_cardinality"))
                .and_then(Value::as_bool)
                != Some(true)
            || response
                .get("diagnostics")
                .and_then(Value::as_object)
                .and_then(|value| value.get("hidden_refill_inputs"))
                .and_then(Value::as_u64)
                != Some(0)
        {
            return Err(R2MapModelError::InvalidResponse(
                "market response group identity",
            ));
        }
        offset += source.candidates.len();
    }
    let diagnostics = object.get("diagnostics").and_then(Value::as_object).ok_or(
        R2MapModelError::InvalidResponse("market response diagnostics"),
    )?;
    let expected_diagnostics = BTreeSet::from([
        "checkpoint_waves",
        "future_refill_tensors",
        "pruned_actions",
        "reference_exhaustive",
    ]);
    if diagnostics
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        != expected_diagnostics
    {
        return Err(R2MapModelError::InvalidResponse(
            "market response diagnostics field set",
        ));
    }
    if diagnostics
        .get("reference_exhaustive")
        .and_then(Value::as_bool)
        != Some(true)
        || diagnostics.get("pruned_actions").and_then(Value::as_u64) != Some(0)
        || diagnostics
            .get("future_refill_tensors")
            .and_then(Value::as_u64)
            != Some(0)
        || diagnostics
            .get("checkpoint_waves")
            .and_then(Value::as_u64)
            .is_none()
    {
        return Err(R2MapModelError::InvalidResponse(
            "market response completeness",
        ));
    }
    Ok(())
}

fn validate_market_response_tensors<'a>(
    descriptors: &[TensorDescriptor],
    payload: &'a [u8],
    total: usize,
) -> Result<BTreeMap<&'a str, &'a [u8]>, R2MapModelError> {
    let expected = [
        ("market_action_scores", vec![total]),
        ("market_predicted_score_to_go", vec![total]),
    ];
    if descriptors.len() != expected.len() {
        return Err(R2MapModelError::InvalidResponse(
            "market response tensor count",
        ));
    }
    let mut offset = 0usize;
    let mut result = BTreeMap::new();
    for ((name, shape), descriptor) in expected.iter().zip(descriptors) {
        if descriptor.name != *name
            || descriptor.dtype != "<f4"
            || descriptor.shape != *shape
            || descriptor.offset != offset
            || descriptor.bytes != shape.iter().product::<usize>() * 4
            || offset + descriptor.bytes > payload.len()
        {
            return Err(R2MapModelError::InvalidResponse(
                "market response tensor descriptor",
            ));
        }
        let bytes = &payload[offset..offset + descriptor.bytes];
        if descriptor.blake3 != blake3::hash(bytes).to_hex().to_string() {
            return Err(R2MapModelError::InvalidResponse(
                "market response tensor checksum",
            ));
        }
        tensor_f32(bytes)?;
        result.insert(*name, bytes);
        offset += descriptor.bytes;
    }
    if offset != payload.len() {
        return Err(R2MapModelError::InvalidResponse(
            "market response trailing bytes",
        ));
    }
    Ok(result)
}

fn validate_response_metadata(
    groups: &[R2MapInferenceGroup],
    expected_request_identity: &str,
    metadata: &Value,
    total: usize,
) -> Result<(), R2MapModelError> {
    let object = metadata
        .as_object()
        .ok_or(R2MapModelError::InvalidResponse("metadata object"))?;
    if object.get("schema_version").and_then(Value::as_u64) != Some(1)
        || object.get("schema_id").and_then(Value::as_str) != Some(RESPONSE_SCHEMA)
        || object.get("request_schema_blake3").and_then(Value::as_str)
            != Some(REQUEST_SCHEMA_BLAKE3)
        || object.get("group_count").and_then(Value::as_u64) != Some(groups.len() as u64)
        || object.get("candidate_count").and_then(Value::as_u64) != Some(total as u64)
        || object
            .get("request_identity_blake3")
            .and_then(Value::as_str)
            != Some(expected_request_identity)
    {
        return Err(R2MapModelError::InvalidResponse("response identity"));
    }
    let response_groups = object
        .get("groups")
        .and_then(Value::as_array)
        .ok_or(R2MapModelError::InvalidResponse("response groups"))?;
    if response_groups.len() != groups.len() {
        return Err(R2MapModelError::InvalidResponse("response group count"));
    }
    let mut offset = 0usize;
    for (source, response) in groups.iter().zip(response_groups) {
        let response = response
            .as_object()
            .ok_or(R2MapModelError::InvalidResponse("response group"))?;
        let ids = source
            .candidates
            .iter()
            .map(|candidate| hex_hash(candidate.action_id))
            .collect::<Vec<_>>();
        if response.get("group_id").and_then(Value::as_str) != Some(&hex_hash(source.group_id))
            || response.get("decision_id").and_then(Value::as_str)
                != Some(&hex_hash(source.decision_id))
            || response.get("candidate_offset").and_then(Value::as_u64) != Some(offset as u64)
            || response.get("candidate_count").and_then(Value::as_u64)
                != Some(source.candidates.len() as u64)
            || response.get("action_ids") != Some(&serde_json::to_value(&ids)?)
            || response
                .get("ordered_action_ids_blake3")
                .and_then(Value::as_str)
                != Some(&ordered_r2_map_action_ids_blake3(
                    &source
                        .candidates
                        .iter()
                        .map(|candidate| candidate.action_id)
                        .collect::<Vec<_>>(),
                ))
            || response.get("model") != Some(&serde_json::to_value(&source.model)?)
            || response
                .get("diagnostics")
                .and_then(Value::as_object)
                .and_then(|value| value.get("actions_enumerated"))
                .and_then(Value::as_u64)
                != Some(source.candidates.len() as u64)
            || response
                .get("diagnostics")
                .and_then(Value::as_object)
                .and_then(|value| value.get("actions_scored"))
                .and_then(Value::as_u64)
                != Some(source.candidates.len() as u64)
            || response
                .get("diagnostics")
                .and_then(Value::as_object)
                .and_then(|value| value.get("complete_cardinality"))
                .and_then(Value::as_bool)
                != Some(true)
        {
            return Err(R2MapModelError::InvalidResponse("response group identity"));
        }
        offset += source.candidates.len();
    }
    let diagnostics = object
        .get("diagnostics")
        .and_then(Value::as_object)
        .ok_or(R2MapModelError::InvalidResponse("response diagnostics"))?;
    if diagnostics
        .get("reference_exhaustive")
        .and_then(Value::as_bool)
        != Some(true)
        || diagnostics.get("pruned_actions").and_then(Value::as_u64) != Some(0)
        || diagnostics.get("remote_inference").and_then(Value::as_bool) != Some(false)
    {
        return Err(R2MapModelError::InvalidResponse("response completeness"));
    }
    Ok(())
}

fn validate_response_tensors<'a>(
    descriptors: &[TensorDescriptor],
    payload: &'a [u8],
    total: usize,
) -> Result<BTreeMap<&'a str, &'a [u8]>, R2MapModelError> {
    let shapes = [vec![total], vec![total], vec![total, 11], vec![total]];
    if descriptors.len() != RESPONSE_TENSORS.len() {
        return Err(R2MapModelError::InvalidResponse("response tensor count"));
    }
    let mut offset = 0;
    let mut result = BTreeMap::new();
    for ((name, dtype), (descriptor, shape)) in
        RESPONSE_TENSORS.iter().zip(descriptors.iter().zip(shapes))
    {
        if descriptor.name != *name
            || descriptor.dtype != *dtype
            || descriptor.shape != shape
            || descriptor.offset != offset
            || descriptor.bytes != shape.iter().product::<usize>() * 4
            || offset + descriptor.bytes > payload.len()
        {
            return Err(R2MapModelError::InvalidResponse(
                "response tensor descriptor",
            ));
        }
        let bytes = &payload[offset..offset + descriptor.bytes];
        if descriptor.blake3 != blake3::hash(bytes).to_hex().to_string() {
            return Err(R2MapModelError::InvalidResponse("response tensor checksum"));
        }
        tensor_f32(bytes)?;
        result.insert(*name, bytes);
        offset += descriptor.bytes;
    }
    if offset != payload.len() {
        return Err(R2MapModelError::InvalidResponse(
            "response tensor trailing bytes",
        ));
    }
    Ok(result)
}

fn tensor_f32(bytes: &[u8]) -> Result<Vec<f32>, R2MapModelError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(R2MapModelError::InvalidResponse("float tensor size"));
    }
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(R2MapModelError::NonFiniteTensor("response"));
    }
    Ok(values)
}

fn write_frame(
    writer: &mut impl Write,
    message_type: u16,
    request_id: u32,
    metadata: &[u8],
    payload: &[u8],
) -> Result<(), R2MapModelError> {
    if metadata.len() > MAX_METADATA_BYTES || payload.len() > MAX_TENSOR_BYTES {
        return Err(R2MapModelError::FrameTooLarge);
    }
    writer.write_all(&frame_header(
        message_type,
        request_id,
        metadata.len(),
        payload.len(),
    ))?;
    writer.write_all(metadata)?;
    writer.write_all(payload)?;
    writer.flush()?;
    Ok(())
}

fn read_frame(reader: &mut impl Read) -> Result<(u16, u32, Vec<u8>, Vec<u8>), R2MapModelError> {
    let mut header = [0; FRAME_HEADER_SIZE];
    reader.read_exact(&mut header)?;
    if header[..4] != PROTOCOL_MAGIC
        || u16::from_le_bytes(header[4..6].try_into().unwrap()) != PROTOCOL_VERSION
    {
        return Err(R2MapModelError::ProtocolMismatch);
    }
    let message = u16::from_le_bytes(header[6..8].try_into().unwrap());
    let request = u32::from_le_bytes(header[8..12].try_into().unwrap());
    let metadata_len = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
    let payload_len = u32::from_le_bytes(header[16..20].try_into().unwrap()) as usize;
    if metadata_len > MAX_METADATA_BYTES || payload_len > MAX_TENSOR_BYTES {
        return Err(R2MapModelError::FrameTooLarge);
    }
    let mut metadata = vec![0; metadata_len];
    let mut payload = vec![0; payload_len];
    reader.read_exact(&mut metadata)?;
    reader.read_exact(&mut payload)?;
    Ok((message, request, metadata, payload))
}

fn frame_header(
    message_type: u16,
    request_id: u32,
    metadata: usize,
    payload: usize,
) -> [u8; FRAME_HEADER_SIZE] {
    let mut bytes = [0; FRAME_HEADER_SIZE];
    bytes[..4].copy_from_slice(&PROTOCOL_MAGIC);
    bytes[4..6].copy_from_slice(&PROTOCOL_VERSION.to_le_bytes());
    bytes[6..8].copy_from_slice(&message_type.to_le_bytes());
    bytes[8..12].copy_from_slice(&request_id.to_le_bytes());
    bytes[12..16].copy_from_slice(&(metadata as u32).to_le_bytes());
    bytes[16..20].copy_from_slice(&(payload as u32).to_le_bytes());
    bytes
}

pub fn ordered_r2_map_action_ids_blake3(action_ids: &[[u8; 32]]) -> String {
    let values = action_ids
        .iter()
        .map(|id| hex_hash(*id))
        .collect::<Vec<_>>();
    blake3::hash(
        &canonical_json(&serde_json::to_value(values).expect("hash ids serialize"))
            .expect("hash ids encode"),
    )
    .to_hex()
    .to_string()
}

pub fn ordered_market_action_ids_blake3(group: &R2MapMarketInferenceGroup) -> String {
    ordered_r2_map_action_ids_blake3(
        &group
            .candidates
            .iter()
            .map(|candidate| candidate.action_id)
            .collect::<Vec<_>>(),
    )
}
fn canonical_json(value: &Value) -> Result<Vec<u8>, R2MapModelError> {
    Ok(serde_json::to_vec(value)?)
}
fn hex_hash(value: [u8; 32]) -> String {
    value.iter().map(|byte| format!("{byte:02x}")).collect()
}
fn is_blake3(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

#[derive(Debug, Error)]
pub enum R2MapModelError {
    #[error("R2-MAP group count {0} is invalid")]
    InvalidGroupCount(usize),
    #[error("R2-MAP candidate count {0} is invalid")]
    InvalidCandidateCount(usize),
    #[error("R2-MAP model identity is invalid")]
    InvalidModelIdentity,
    #[error("R2-MAP local serving bundle is invalid")]
    InvalidServingBundle,
    #[error("R2-MAP local serving bundle does not contain the requested policy")]
    MissingServingModel,
    #[error("R2-MAP group identity is duplicated")]
    DuplicateGroupIdentity,
    #[error("R2-MAP action identity is invalid or duplicated")]
    InvalidCandidateIdentity,
    #[error("R2-MAP tensor {0} has an invalid shape")]
    InvalidTensorShape(&'static str),
    #[error("R2-MAP tensor {0} contains a non-finite value")]
    NonFiniteTensor(&'static str),
    #[error("R2-MAP frame exceeds the protocol ceiling")]
    FrameTooLarge,
    #[error("R2-MAP protocol magic or version differs")]
    ProtocolMismatch,
    #[error("R2-MAP response request id {actual} differs from {expected}")]
    RequestIdMismatch { expected: u32, actual: u32 },
    #[error("R2-MAP response is invalid: {0}")]
    InvalidResponse(&'static str),
    #[error("R2-MAP service rejected the request: {0}")]
    Service(String),
    #[error("spawned R2-MAP process did not expose {0}")]
    MissingPipe(&'static str),
    #[error("R2-MAP process exited unsuccessfully with code {0:?}")]
    ProcessExit(Option<i32>),
    #[error("R2-MAP process lock was poisoned")]
    ProcessLockPoisoned,
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_verified_bundle_fixture() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let root = tempfile::tempdir().unwrap();
        let run_dir = root.path().join("run");
        let checkpoint = run_dir.join("checkpoints/checkpoint-verified");
        fs::create_dir_all(&checkpoint).unwrap();
        let dataset_contract = json!({
            "schema_version": 1,
            "dataset_blake3": "d".repeat(64),
            "d6_schema": "r2-map-d6-cyclic-offset-v1",
            "d6_cycle_epochs": 12,
            "imitation_subset_schema": "r2-map-draft-imitation-subset-v1",
            "imitation_subset_parts_per_million": 10_000,
            "collection_kind": "bootstrap",
            "example_count": 1,
            "imitation_example_count": 1,
            "market_decision_count": 1,
            "market_policy_target_count": 1,
        });
        let next_batch_identity = "synthetic-next-batch-v1";
        let state = json!({
            "dataset_contract": dataset_contract,
            "next_batch_identity": next_batch_identity,
        });
        let contents = [
            ("model.safetensors", b"model".to_vec()),
            ("optimizer.safetensors", b"optimizer".to_vec()),
            ("state.json", serde_json::to_vec_pretty(&state).unwrap()),
            ("fixed-prediction-panel.safetensors", b"panel".to_vec()),
        ];
        for (name, bytes) in &contents {
            fs::write(checkpoint.join(name), bytes).unwrap();
        }
        let model_config = json!({"hidden_dim": 192});
        let config_hash = blake3::hash(&canonical_json(&model_config).unwrap())
            .to_hex()
            .to_string();
        let files = contents
            .iter()
            .map(|(name, bytes)| {
                (
                    (*name).to_owned(),
                    json!({
                        "bytes": bytes.len(),
                        "blake3": blake3::hash(bytes).to_hex().to_string(),
                    }),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        let mut manifest = json!({
            "schema_version": 2,
            "schema_id": "r2-map-checkpoint-v2",
            "checkpoint_id": "checkpoint-verified",
            "identity": {
                "checkpoint_id": "checkpoint-verified",
                "run_id": "run-verified",
                "branch_id": "branch-verified",
                "source_blake3": "5".repeat(64),
                "dataset_blake3": "6".repeat(64),
                "model_config_blake3": config_hash,
                "training_config_blake3": "7".repeat(64),
                "loss_contract_blake3": "8".repeat(64),
            },
            "model_config": model_config,
            "resume_state_blake3": "9".repeat(64),
            "prediction_panel": {},
            "files": files,
        });
        let compact = blake3::hash(&canonical_json(&manifest).unwrap())
            .to_hex()
            .to_string();
        manifest["manifest_identity_blake3"] = Value::String(compact.clone());
        let manifest_bytes = serde_json::to_vec_pretty(&manifest).unwrap();
        fs::write(checkpoint.join("checkpoint.json"), &manifest_bytes).unwrap();
        let manifest_file_hash = blake3::hash(&manifest_bytes).to_hex().to_string();
        fs::create_dir_all(run_dir.join("verifications")).unwrap();
        let mut receipt = json!({
            "schema_version": 2,
            "schema_id": "r2-map-checkpoint-verification-v2",
            "checkpoint_id": "checkpoint-verified",
            "checkpoint_manifest_blake3": manifest_file_hash,
            "prediction_panel_id": "panel",
            "dataset_contract_blake3": blake3::hash(&canonical_json(&dataset_contract).unwrap()).to_hex().to_string(),
            "prediction_tensor_blake3": {"action_scores": "b".repeat(64)},
            "loss_stream_offset_bytes": 0,
            "loss_stream_prefix_blake3": "a".repeat(64),
            "exact_prediction_match": true,
            "next_batch_identity": next_batch_identity,
            "exact_next_batch_match": true,
        });
        let verification = blake3::hash(&canonical_json(&receipt).unwrap())
            .to_hex()
            .to_string();
        receipt["verification_id"] = Value::String(verification.clone());
        fs::write(
            run_dir.join("verifications/checkpoint-verified.json"),
            serde_json::to_vec_pretty(&receipt).unwrap(),
        )
        .unwrap();
        let bundle = R2MapServingBundle {
            schema_version: 2,
            schema_id: R2_MAP_SERVING_BUNDLE_SCHEMA.into(),
            protocols: R2MapProtocolIdentity {
                collector_hash: [1; 32],
                source_hash: [2; 32],
                serving_protocol_hash: [3; 32],
            },
            entries: vec![R2MapServingBundleEntry {
                manifest_identity_blake3: compact,
                run_dir,
                checkpoint_path: checkpoint.clone(),
                model: R2MapModelIdentity {
                    checkpoint_id: "checkpoint-verified".into(),
                    checkpoint_manifest_blake3: manifest_file_hash,
                    model_config_blake3: config_hash,
                    model_weights_blake3: blake3::hash(b"model").to_hex().to_string(),
                    verification_id: verification,
                },
                pinned: true,
            }],
        };
        let bundle_path = root.path().join("bundle.json");
        fs::write(&bundle_path, serde_json::to_vec_pretty(&bundle).unwrap()).unwrap();
        (root, bundle_path, checkpoint)
    }

    fn rewrite_verification_receipt(
        bundle_path: &Path,
        checkpoint: &Path,
        mutate: impl FnOnce(&mut serde_json::Map<String, Value>),
    ) {
        let run_dir = checkpoint
            .parent()
            .and_then(Path::parent)
            .expect("checkpoint is nested below the run directory");
        let receipt_path = run_dir.join("verifications/checkpoint-verified.json");
        let mut receipt: Value = serde_json::from_slice(&fs::read(&receipt_path).unwrap()).unwrap();
        {
            let object = receipt.as_object_mut().unwrap();
            object.remove("verification_id");
            mutate(object);
        }
        let verification_id = blake3::hash(&canonical_json(&receipt).unwrap())
            .to_hex()
            .to_string();
        receipt.as_object_mut().unwrap().insert(
            "verification_id".into(),
            Value::String(verification_id.clone()),
        );
        fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();

        let mut bundle: Value = serde_json::from_slice(&fs::read(bundle_path).unwrap()).unwrap();
        bundle["entries"][0]["model"]["verification_id"] = Value::String(verification_id);
        fs::write(bundle_path, serde_json::to_vec_pretty(&bundle).unwrap()).unwrap();
    }

    fn identity(digit: char) -> R2MapModelIdentity {
        R2MapModelIdentity {
            checkpoint_id: format!("checkpoint-{digit}"),
            checkpoint_manifest_blake3: digit.to_string().repeat(64),
            model_config_blake3: "2".repeat(64),
            model_weights_blake3: "3".repeat(64),
            verification_id: "4".repeat(64),
        }
    }

    fn public() -> R2MapPublicTensors {
        R2MapPublicTensors {
            token_features: vec![0.0; BOARD_SLOTS * BOARD_TOKEN_CAPACITY * R2_MAP_TOKEN_FEATURES],
            token_types: vec![0; BOARD_SLOTS * BOARD_TOKEN_CAPACITY],
            token_mask: vec![0; BOARD_SLOTS * BOARD_TOKEN_CAPACITY],
            market_features: vec![0.0; 4 * MARKET_FEATURES],
            market_mask: [1; 4],
            player_features: vec![0.0; BOARD_SLOTS * PLAYER_FEATURES],
            player_mask: [1; BOARD_SLOTS],
            global_features: [0.0; GLOBAL_FEATURES],
        }
    }

    fn group() -> R2MapInferenceGroup {
        R2MapInferenceGroup {
            group_id: [5; 32],
            decision_id: [6; 32],
            model: identity('1'),
            parent: public(),
            candidates: vec![R2MapInferenceCandidate {
                action_id: [7; 32],
                afterstate: public(),
                action_bytes: [9; R2_MAP_ACTION_BYTES],
                exact_afterstate_score: 42.0,
            }],
        }
    }

    fn group_with_candidates(count: usize) -> R2MapInferenceGroup {
        let mut value = group();
        let template = value.candidates[0].clone();
        value.candidates = (0..count)
            .map(|index| {
                let mut candidate = template.clone();
                candidate.action_id = blake3::hash(&index.to_le_bytes()).into();
                candidate.exact_afterstate_score = index as f32;
                candidate
            })
            .collect();
        value
    }

    fn prediction_for(group: &R2MapInferenceGroup, range: Range<usize>) -> R2MapPredictionGroup {
        let indices = range.clone().map(|index| index as f32).collect::<Vec<_>>();
        R2MapPredictionGroup {
            group_id: group.group_id,
            decision_id: group.decision_id,
            action_ids: group.candidates[range]
                .iter()
                .map(|candidate| candidate.action_id)
                .collect(),
            action_scores: indices.clone(),
            predicted_score_to_go: indices.clone(),
            predicted_score_components_to_go: indices.iter().map(|value| [*value; 11]).collect(),
            bootstrap_policy_logits: indices,
        }
    }

    fn empty_prediction(group: &R2MapInferenceGroup) -> R2MapPredictionGroup {
        R2MapPredictionGroup {
            group_id: group.group_id,
            decision_id: group.decision_id,
            action_ids: Vec::new(),
            action_scores: Vec::new(),
            predicted_score_to_go: Vec::new(),
            predicted_score_components_to_go: Vec::new(),
            bootstrap_policy_logits: Vec::new(),
        }
    }

    #[test]
    fn logical_candidate_screens_partition_at_wire_boundaries_without_pruning() {
        assert_eq!(R2_MAP_PROTOCOL_MAX_CANDIDATES_PER_GROUP, 8_192);
        assert_eq!(R2_MAP_WIRE_FRAME_CANDIDATES, 1_024);
        assert_eq!(
            REQUEST_SCHEMA_BLAKE3,
            "bce9b1e6701dd86debc7a0fae496e6e55d72acac554eb572dcdcbf5356b6b8fa"
        );
        assert_eq!(
            logical_frame_ranges(8_192).unwrap(),
            vec![
                0..1_024,
                1_024..2_048,
                2_048..3_072,
                3_072..4_096,
                4_096..5_120,
                5_120..6_144,
                6_144..7_168,
                7_168..8_192,
            ]
        );
        assert_eq!(
            logical_frame_ranges(8_193).unwrap(),
            vec![
                0..1_024,
                1_024..2_048,
                2_048..3_072,
                3_072..4_096,
                4_096..5_120,
                5_120..6_144,
                6_144..7_168,
                7_168..8_192,
                8_192..8_193,
            ]
        );
        assert_eq!(
            logical_frame_ranges(2_049).unwrap(),
            vec![0..1_024, 1_024..2_048, 2_048..2_049]
        );
        assert!(logical_frame_ranges(0).is_err());
        for count in [8_192, 8_193, 2_049] {
            let ranges = logical_frame_ranges(count).unwrap();
            assert!(
                ranges
                    .iter()
                    .all(|range| range.len() <= R2_MAP_WIRE_FRAME_CANDIDATES
                        && range.len() <= R2_MAP_PROTOCOL_MAX_CANDIDATES_PER_GROUP)
            );
            assert_eq!(ranges.iter().map(Range::len).sum::<usize>(), count);
            assert_eq!(ranges.first().unwrap().start, 0);
            assert_eq!(ranges.last().unwrap().end, count);
            assert!(ranges.windows(2).all(|pair| pair[0].end == pair[1].start));
        }
    }

    #[test]
    fn partition_responses_merge_in_exact_action_order_and_match_single_frame() {
        let group = group_with_candidates(5);
        let expected = prediction_for(&group, 0..5);
        let mut merged = empty_prediction(&group);
        for range in [0..2, 2..4, 4..5] {
            append_partition_prediction(
                &mut merged,
                &group,
                range.clone(),
                prediction_for(&group, range),
            )
            .unwrap();
        }
        assert_eq!(merged, expected);
        assert_eq!(merged.action_ids.len(), group.candidates.len());
    }

    #[test]
    fn partition_merge_rejects_partial_or_reordered_frames_before_mutation() {
        let group = group_with_candidates(3);
        let mut merged = empty_prediction(&group);
        let mut partial = prediction_for(&group, 0..2);
        partial.action_scores.pop();
        assert!(append_partition_prediction(&mut merged, &group, 0..2, partial).is_err());
        assert_eq!(merged, empty_prediction(&group));

        let mut reordered = prediction_for(&group, 0..2);
        reordered.action_ids.swap(0, 1);
        assert!(append_partition_prediction(&mut merged, &group, 0..2, reordered).is_err());
        assert_eq!(merged, empty_prediction(&group));
    }

    fn market_group(
        kind: R2MapMarketDecisionKind,
        public_nature_tokens: u8,
        public_wildlife_bag_total: u8,
    ) -> R2MapMarketInferenceGroup {
        let decision_id = [6; 32];
        assert_eq!(public_wildlife_bag_total % 5, 0);
        let public_wildlife_bag_counts = [public_wildlife_bag_total / 5; 5];
        let public_market_wildlife = [0, 1, 2, 3];
        let candidates = expected_market_action_bytes(
            kind,
            public_nature_tokens,
            public_wildlife_bag_counts,
            public_market_wildlife,
        )
        .unwrap()
        .into_iter()
        .map(|action_bytes| {
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"r2-map-market-action-identity-v1");
            hasher.update(&decision_id);
            hasher.update(&action_bytes);
            R2MapMarketInferenceCandidate {
                action_id: *hasher.finalize().as_bytes(),
                action_bytes,
            }
        })
        .collect();
        R2MapMarketInferenceGroup {
            group_id: [5; 32],
            decision_id,
            model: identity('1'),
            parent: public(),
            exact_current_score: 7.0,
            decision_kind: kind,
            public_nature_tokens,
            public_wildlife_bag_counts,
            public_wildlife_bag_total,
            public_market_wildlife,
            candidates,
        }
    }

    #[test]
    fn protocol_v3_emits_canonical_action_bytes_and_stable_request_identity() {
        let group = group();
        let (metadata, payload, identity) = encode_request(&[group]).unwrap();
        let value: Value = serde_json::from_slice(&metadata).unwrap();
        assert_eq!(value["schema_id"], REQUEST_SCHEMA);
        assert_eq!(value["request_schema_blake3"], REQUEST_SCHEMA_BLAKE3);
        assert!(is_blake3(&identity));
        let descriptors: Vec<TensorDescriptor> =
            serde_json::from_value(value["tensors"].clone()).unwrap();
        let action = descriptors
            .iter()
            .find(|descriptor| descriptor.name == "action_bytes")
            .unwrap();
        assert_eq!(action.dtype, "|u1");
        assert_eq!(action.shape, vec![1, 128]);
        assert_eq!(
            &payload[action.offset..action.offset + action.bytes],
            &[9; 128]
        );
    }

    #[test]
    fn market_request_is_exhaustive_ordered_and_uses_the_frozen_public_fields() {
        let group = market_group(R2MapMarketDecisionKind::PaidWipes, 2, 10);
        assert_eq!(group.candidates.len(), 16);
        validate_market_groups(std::slice::from_ref(&group)).unwrap();
        let (metadata, payload, request_identity) =
            encode_market_request(std::slice::from_ref(&group)).unwrap();
        let value: Value = serde_json::from_slice(&metadata).unwrap();
        assert_eq!(value["schema_id"], R2_MAP_MARKET_REQUEST_SCHEMA);
        assert_eq!(
            value["request_schema_blake3"],
            R2_MAP_MARKET_REQUEST_SCHEMA_BLAKE3
        );
        assert_eq!(value["action_count"], 16);
        assert_eq!(value["groups"][0]["decision_kind"], 1);
        assert_eq!(value["groups"][0]["public_nature_tokens"], 2);
        assert_eq!(
            value["groups"][0]["public_wildlife_bag_counts"],
            json!([2, 2, 2, 2, 2])
        );
        assert_eq!(value["groups"][0]["public_wildlife_bag_total"], 10);
        assert_eq!(
            value["groups"][0]["public_market_wildlife"],
            json!([0, 1, 2, 3])
        );
        assert!(is_blake3(&request_identity));
        let descriptors: Vec<TensorDescriptor> =
            serde_json::from_value(value["tensors"].clone()).unwrap();
        assert_eq!(descriptors[0].name, "action_offsets");
        let actions = descriptors
            .iter()
            .find(|descriptor| descriptor.name == "action_bytes")
            .unwrap();
        assert_eq!(actions.shape, vec![16, R2_MAP_MARKET_ACTION_BYTES]);
        assert_eq!(
            &payload[actions.offset..actions.offset + R2_MAP_MARKET_ACTION_BYTES],
            &[1, 1, 2, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            &payload[actions.offset + R2_MAP_MARKET_ACTION_BYTES
                ..actions.offset + 2 * R2_MAP_MARKET_ACTION_BYTES],
            &[1, 1, 3, 1, 0, 0, 0, 0]
        );
    }

    #[test]
    fn market_request_rejects_partial_reordered_and_resource_inconsistent_screens() {
        let group = market_group(R2MapMarketDecisionKind::PaidWipes, 1, 40);
        let mut partial = group.clone();
        partial.candidates.pop();
        assert!(matches!(
            validate_market_groups(&[partial]),
            Err(R2MapModelError::InvalidCandidateIdentity)
        ));
        let mut reordered = group.clone();
        reordered.candidates.swap(1, 2);
        assert!(matches!(
            validate_market_groups(&[reordered]),
            Err(R2MapModelError::InvalidCandidateIdentity)
        ));
        let mut no_tokens = group;
        no_tokens.public_nature_tokens = 0;
        assert!(matches!(
            validate_market_groups(&[no_tokens]),
            Err(R2MapModelError::InvalidCandidateIdentity)
        ));

        let mut drifted_total = market_group(R2MapMarketDecisionKind::PaidWipes, 1, 40);
        drifted_total.public_wildlife_bag_total = 39;
        assert!(matches!(
            validate_market_groups(&[drifted_total]),
            Err(R2MapModelError::InvalidCandidateIdentity)
        ));
    }

    #[test]
    fn market_request_uses_candidate_specific_public_universal_legality() {
        let mut group = market_group(R2MapMarketDecisionKind::PaidWipes, 1, 40);
        group.public_wildlife_bag_counts = [1, 3, 0, 0, 0];
        group.public_wildlife_bag_total = 4;
        group.candidates = expected_market_action_bytes(
            group.decision_kind,
            group.public_nature_tokens,
            group.public_wildlife_bag_counts,
            group.public_market_wildlife,
        )
        .unwrap()
        .into_iter()
        .map(|action_bytes| R2MapMarketInferenceCandidate {
            action_id: cascadia_game::public_market_action_identity(
                group.decision_id,
                action_bytes,
            ),
            action_bytes,
        })
        .collect();
        validate_market_groups(std::slice::from_ref(&group)).unwrap();
        let masks = group
            .candidates
            .iter()
            .skip(1)
            .map(|candidate| candidate.action_bytes[3])
            .collect::<Vec<_>>();
        assert!(!masks.contains(&13));
        assert!(masks.contains(&15));
    }

    #[test]
    fn rust_market_contract_matches_the_shared_compile_independent_fixture() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/r2_map/public-market-decision-protocol-v3.json");
        let mut fixture: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        let object = fixture.as_object_mut().unwrap();
        let fixture_hash = object
            .remove("fixture_blake3")
            .and_then(|value| value.as_str().map(ToOwned::to_owned))
            .unwrap();
        assert_eq!(
            blake3::hash(&canonical_json(&fixture).unwrap())
                .to_hex()
                .to_string(),
            fixture_hash
        );
        assert_eq!(
            fixture["request_schema_blake3"],
            R2_MAP_MARKET_REQUEST_SCHEMA_BLAKE3
        );
        assert_eq!(
            fixture["response_schema_blake3"],
            R2_MAP_MARKET_RESPONSE_SCHEMA_BLAKE3
        );
        assert_eq!(
            fixture["action_schema_blake3"],
            cascadia_r2::R2_MAP_MARKET_ACTION_SCHEMA_BLAKE3
        );
        for case in fixture["cases"].as_array().unwrap() {
            let decision_id = decode_fixture_hex::<32>(case["decision_id"].as_str().unwrap());
            let rows = case["action_bytes_hex"].as_array().unwrap();
            let expected_ids = case["action_ids"].as_array().unwrap();
            let action_ids = rows
                .iter()
                .zip(expected_ids)
                .map(|(row, expected)| {
                    let row =
                        decode_fixture_hex::<R2_MAP_MARKET_ACTION_BYTES>(row.as_str().unwrap());
                    let mut hasher = blake3::Hasher::new();
                    hasher.update(b"r2-map-market-action-identity-v1");
                    hasher.update(&decision_id);
                    hasher.update(&row);
                    let action_id = *hasher.finalize().as_bytes();
                    assert_eq!(hex_hash(action_id), expected.as_str().unwrap());
                    action_id
                })
                .collect::<Vec<_>>();
            assert_eq!(
                ordered_r2_map_action_ids_blake3(&action_ids),
                case["ordered_action_ids_blake3"].as_str().unwrap()
            );
        }
    }

    fn decode_fixture_hex<const N: usize>(value: &str) -> [u8; N] {
        assert_eq!(value.len(), N * 2);
        std::array::from_fn(|index| {
            u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).unwrap()
        })
    }

    #[test]
    fn serving_bundle_is_strict_and_resolves_compact_manifest_identity() {
        let root = std::env::temp_dir();
        let bundle = R2MapServingBundle {
            schema_version: 2,
            schema_id: R2_MAP_SERVING_BUNDLE_SCHEMA.into(),
            protocols: R2MapProtocolIdentity {
                collector_hash: [1; 32],
                source_hash: [2; 32],
                serving_protocol_hash: [3; 32],
            },
            entries: vec![R2MapServingBundleEntry {
                manifest_identity_blake3: "a".repeat(64),
                run_dir: root.clone(),
                checkpoint_path: root.join("checkpoint-1"),
                model: identity('1'),
                pinned: true,
            }],
        };
        bundle.validate().unwrap();
        assert_eq!(
            bundle.model_for_manifest_identity([0xaa; 32]).unwrap(),
            identity('1')
        );
        let mut duplicate = bundle.clone();
        duplicate.entries.push(bundle.entries[0].clone());
        assert!(matches!(
            duplicate.validate(),
            Err(R2MapModelError::InvalidServingBundle)
        ));
    }

    #[test]
    fn verified_bundle_rejects_checkpoint_and_mapping_tampering_before_use() {
        let (_root, path, checkpoint) = write_verified_bundle_fixture();
        R2MapServingBundle::read_verified(&path).unwrap();
        fs::write(checkpoint.join("model.safetensors"), b"tampered").unwrap();
        assert!(matches!(
            R2MapServingBundle::read_verified(&path),
            Err(R2MapModelError::InvalidServingBundle)
        ));

        let (_root, path, _checkpoint) = write_verified_bundle_fixture();
        let mut value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        value["entries"][0]["manifest_identity_blake3"] = Value::String("f".repeat(64));
        fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        assert!(matches!(
            R2MapServingBundle::read_verified(&path),
            Err(R2MapModelError::InvalidServingBundle)
        ));
    }

    #[test]
    fn verified_bundle_rejects_rehashed_resume_proof_drift() {
        let (_root, path, checkpoint) = write_verified_bundle_fixture();
        rewrite_verification_receipt(&path, &checkpoint, |receipt| {
            receipt.insert(
                "dataset_contract_blake3".into(),
                Value::String("e".repeat(64)),
            );
        });
        assert!(matches!(
            R2MapServingBundle::read_verified(&path),
            Err(R2MapModelError::InvalidServingBundle)
        ));

        let (_root, path, checkpoint) = write_verified_bundle_fixture();
        rewrite_verification_receipt(&path, &checkpoint, |receipt| {
            receipt.insert(
                "next_batch_identity".into(),
                Value::String("different-next-batch".into()),
            );
        });
        assert!(matches!(
            R2MapServingBundle::read_verified(&path),
            Err(R2MapModelError::InvalidServingBundle)
        ));

        let (_root, path, checkpoint) = write_verified_bundle_fixture();
        rewrite_verification_receipt(&path, &checkpoint, |receipt| {
            receipt.insert("exact_next_batch_match".into(), Value::Bool(false));
        });
        assert!(matches!(
            R2MapServingBundle::read_verified(&path),
            Err(R2MapModelError::InvalidServingBundle)
        ));

        let (_root, path, checkpoint) = write_verified_bundle_fixture();
        rewrite_verification_receipt(&path, &checkpoint, |receipt| {
            receipt.remove("dataset_contract_blake3");
        });
        assert!(matches!(
            R2MapServingBundle::read_verified(&path),
            Err(R2MapModelError::InvalidServingBundle)
        ));
    }

    #[test]
    fn process_restart_and_clean_shutdown_are_supported() {
        let mut process =
            R2MapModelProcess::spawn("sh", ["-c", "dd bs=20 count=1 of=/dev/null 2>/dev/null"])
                .unwrap();
        process.restart().unwrap();
        process.shutdown().unwrap();
    }

    #[test]
    fn rust_client_scores_through_the_python_protocol_fixture_and_restarts() {
        if !Path::new(".venv/bin/python").is_file() {
            return;
        }
        let args = [
            "PYTHONDONTWRITEBYTECODE=1",
            "PYTHONPATH=python",
            ".venv/bin/python",
            "-m",
            "cascadia_mlx.r2_map_protocol_fixture",
        ];
        let mut process = R2MapModelProcess::spawn("env", args).unwrap();
        let request = group();
        let first = process
            .score_groups(std::slice::from_ref(&request))
            .unwrap();
        assert_eq!(first[0].action_scores, vec![42.0]);
        assert_eq!(first[0].action_ids, vec![[7; 32]]);
        process.restart().unwrap();
        let second = process.score_groups(&[request]).unwrap();
        assert_eq!(second[0].action_scores, vec![42.0]);
        process.shutdown().unwrap();
    }
}
