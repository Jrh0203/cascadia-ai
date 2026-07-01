//! Paired focal-seat comparison between R2-MAP and the qualified exact NNUE.
//!
//! The comparison deliberately keeps the other three seats greedy in both
//! arms. This isolates the focal policy, preserves identical game seeds and
//! opponent RNG domains, and pays the K32/R600 cost only for the control seat.

use std::{ffi::OsString, fs::File, io::Read, path::Path, time::Instant};

use cascadia_eval::{
    focal::{
        FOCAL_MAX_RSS_BYTES, FocalBenchmarkError, FocalGameRecord, FocalRuntimeObservation, PairArm,
    },
    focal_campaign::{FocalGameExecutor, FocalGameRequest},
    focal_gameplay::{RuntimeResourceProbe, SystemRuntimeResourceProbe},
    r2_map_binding::R2MapImplementationBinding,
};
use cascadia_game::GameConfig;
use cascadia_model::{R2MapModelError, R2MapModelIdentity, R2MapModelProcess, R2MapServingBundle};
use cascadia_search::select_r2_map_turn;
use cascadia_sim::{
    SimulationError, play_match_with_seat_selector, select_greedy_action, strategy_rng,
};
use thiserror::Error;

use crate::legacy_teacher::{BridgeError, ExactRustLegacyTeacher, load_legacy_weights};

pub const QUALIFIED_EXACT_NNUE_CHECKPOINT_ID: &str =
    "canonical-action-legacy-exact-mlx-v1-k32-r600-lmr-no-paid-prelude";
pub const CROSS_ARCH_INFERENCE_SETTINGS_ID: &str =
    "r2-map-exhaustive-argmax-vs-qualified-exact-nnue-k32-r600-v1";
pub const QUALIFIED_NNUE_WEIGHTS_BLAKE3: &str =
    "9e1d568693274fc537ac4f6d6f729abb1ee8da8330a78d1f78a1f62b733de400";
pub const EXACT_ROLLOUT_PARITY_REPORT_BLAKE3: &str =
    "ff10f31941e3a49b6dc9acfc06c34a1d0e7fba5e680ad42494af283c3aafc4dc";

/// Concrete benchmark executor owning both long-lived inference services.
pub struct CrossArchitectureFocalExecutor {
    candidate_checkpoint_id: String,
    control_checkpoint_id: String,
    implementation_binding: R2MapImplementationBinding,
    r2_model: R2MapModelIdentity,
    r2_process: Option<R2MapModelProcess>,
    exact_teacher: Option<ExactRustLegacyTeacher>,
    resource_probe: SystemRuntimeResourceProbe,
    pending_r2_load_seconds: Option<f64>,
    pending_exact_load_seconds: Option<f64>,
}

impl CrossArchitectureFocalExecutor {
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        candidate_checkpoint_id: impl Into<String>,
        control_checkpoint_id: impl Into<String>,
        implementation_binding: R2MapImplementationBinding,
        r2_bundle_path: &Path,
        r2_backend_parity_receipt: &Path,
        r2_python: &Path,
        r2_python_path: &Path,
        exact_weights: &Path,
        exact_rollout_parity_report: &Path,
        exact_rollouts: usize,
    ) -> Result<Self, CrossArchitectureFocalError> {
        let candidate_checkpoint_id = candidate_checkpoint_id.into();
        let control_checkpoint_id = control_checkpoint_id.into();
        if candidate_checkpoint_id.trim().is_empty()
            || candidate_checkpoint_id == "greedy-v1"
            || control_checkpoint_id != QUALIFIED_EXACT_NNUE_CHECKPOINT_ID
            || exact_rollouts != 600
            || std::env::var("MCE_LMR").as_deref() != Ok("1")
            || std::env::var("MCE_DIVERSE_PREFILTER").as_deref() != Ok("1")
        {
            return Err(CrossArchitectureFocalError::InvalidConfiguration);
        }
        implementation_binding
            .validate()
            .map_err(|_| CrossArchitectureFocalError::InvalidConfiguration)?;
        // Freeze the system-swap baseline before either inference service or
        // its checkpoint is loaded. The smoke gate covers the complete work
        // item lifecycle, not only gameplay after model startup.
        let resource_probe = SystemRuntimeResourceProbe::new()?;

        let bundle = R2MapServingBundle::read_verified(r2_bundle_path)?;
        if bundle.protocols != implementation_binding.protocols {
            return Err(CrossArchitectureFocalError::ProtocolIdentityMismatch);
        }
        let r2_model = bundle
            .entries
            .iter()
            .find(|entry| entry.model.checkpoint_id == candidate_checkpoint_id)
            .map(|entry| entry.model.clone())
            .ok_or_else(|| {
                CrossArchitectureFocalError::MissingCandidate(candidate_checkpoint_id.clone())
            })?;
        verify_r2_backend_parity_receipt(r2_backend_parity_receipt, &r2_model)?;

        let r2_started = Instant::now();
        let r2_args = vec![
            OsString::from("PYTHONDONTWRITEBYTECODE=1"),
            OsString::from(format!("PYTHONPATH={}", r2_python_path.display())),
            r2_python.as_os_str().to_owned(),
            OsString::from("-m"),
            OsString::from("cascadia_mlx.r2_map_serve"),
            OsString::from("--bundle"),
            r2_bundle_path.as_os_str().to_owned(),
            OsString::from("--backend"),
            OsString::from("numpy"),
        ];
        let r2_process = R2MapModelProcess::spawn("/usr/bin/env", r2_args)?;
        let r2_load_seconds = r2_started.elapsed().as_secs_f64();

        verify_exact_control_artifacts(exact_weights, exact_rollout_parity_report)?;
        let exact_started = Instant::now();
        let exact_teacher =
            ExactRustLegacyTeacher::new(load_legacy_weights(exact_weights)?, exact_rollouts)?;
        let exact_load_seconds = exact_started.elapsed().as_secs_f64();

        Ok(Self {
            candidate_checkpoint_id,
            control_checkpoint_id,
            implementation_binding,
            r2_model,
            r2_process: Some(r2_process),
            exact_teacher: Some(exact_teacher),
            resource_probe,
            pending_r2_load_seconds: Some(r2_load_seconds),
            pending_exact_load_seconds: Some(exact_load_seconds),
        })
    }

    /// Shut down the framed R2 service after the in-process exact NNUE ends.
    fn finish_services(&mut self) -> Result<(), CrossArchitectureFocalError> {
        let r2_result = self
            .r2_process
            .take()
            .expect("R2 process is present until shutdown")
            .shutdown();
        self.exact_teacher
            .take()
            .expect("exact teacher is present until shutdown");
        r2_result?;
        Ok(())
    }

    pub fn shutdown(mut self) -> Result<(), CrossArchitectureFocalError> {
        self.finish_services()
    }

    fn validate_request(
        &self,
        request: &FocalGameRequest,
    ) -> Result<(), CrossArchitectureFocalError> {
        let expected_focal = match request.identity.arm {
            PairArm::Candidate => &self.candidate_checkpoint_id,
            PairArm::Control => &self.control_checkpoint_id,
        };
        if request.implementation_binding != self.implementation_binding
            || request.identity.focal_checkpoint_id != *expected_focal
            || request.identity.inference_settings_id != CROSS_ARCH_INFERENCE_SETTINGS_ID
            || request.identity.opponents.len() != 3
            || request
                .identity
                .opponents
                .iter()
                .any(|opponent| opponent.checkpoint_id != "greedy-v1")
        {
            return Err(CrossArchitectureFocalError::InvalidRequest);
        }
        let mut occupied = [false; 4];
        let focal = usize::from(request.focal_seat);
        if focal >= occupied.len() {
            return Err(CrossArchitectureFocalError::InvalidRequest);
        }
        occupied[focal] = true;
        for opponent in &request.identity.opponents {
            let seat = usize::from(opponent.seat);
            if seat >= occupied.len() || std::mem::replace(&mut occupied[seat], true) {
                return Err(CrossArchitectureFocalError::InvalidRequest);
            }
        }
        if !occupied.into_iter().all(|value| value) {
            return Err(CrossArchitectureFocalError::InvalidRequest);
        }
        Ok(())
    }

    fn execute_game(
        &mut self,
        request: &FocalGameRequest,
    ) -> Result<cascadia_sim::MatchResult, CrossArchitectureFocalError> {
        let mut identities = vec!["greedy-v1".to_owned(); 4];
        identities[usize::from(request.focal_seat)] = request.identity.focal_checkpoint_id.clone();
        let mut greedy_rngs = identities
            .iter()
            .enumerate()
            .map(|(seat, checkpoint_id)| {
                strategy_rng(
                    request.game_seed,
                    seat,
                    &format!(
                        "{}:opponent:{checkpoint_id}:seat-{seat}",
                        request.identity.field_manifest_id
                    ),
                )
            })
            .collect::<Vec<_>>();
        let focal = usize::from(request.focal_seat);
        let candidate_checkpoint_id = self.candidate_checkpoint_id.clone();
        let control_checkpoint_id = self.control_checkpoint_id.clone();
        let model = self.r2_model.clone();
        let game_index = u64::try_from(request.identity.pair_index)
            .map_err(|_| CrossArchitectureFocalError::InvalidRequest)?;
        let r2_process = self
            .r2_process
            .as_mut()
            .ok_or(CrossArchitectureFocalError::ServiceAlreadyShutdown)?;
        let exact_teacher = self
            .exact_teacher
            .as_mut()
            .ok_or(CrossArchitectureFocalError::ServiceAlreadyShutdown)?;

        play_match_with_seat_selector(
            GameConfig::research_aaaaa(4)?,
            request.game_seed,
            &identities,
            |seat, game| {
                if seat != focal {
                    let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
                    return select_greedy_action(game, &prelude, &mut greedy_rngs[seat]);
                }
                let checkpoint_id = &identities[seat];
                if checkpoint_id == &candidate_checkpoint_id {
                    return select_r2_map_turn(r2_process, game, game_index, model.clone(), |_| {
                        Ok(None)
                    })
                    .map(|selected| selected.action)
                    .map_err(|error| SimulationError::Strategy(error.to_string()));
                }
                if checkpoint_id == &control_checkpoint_id {
                    return exact_teacher
                        .select_action(game)
                        .map_err(|error| SimulationError::Strategy(error.to_string()));
                }
                Err(SimulationError::Strategy(
                    "unregistered cross-architecture focal checkpoint".to_owned(),
                ))
            },
        )
        .map_err(CrossArchitectureFocalError::Simulation)
    }
}

fn verify_exact_control_artifacts(
    weights: &Path,
    parity_report: &Path,
) -> Result<(), CrossArchitectureFocalError> {
    if file_blake3(weights)? != QUALIFIED_NNUE_WEIGHTS_BLAKE3
        || file_blake3(parity_report)? != EXACT_ROLLOUT_PARITY_REPORT_BLAKE3
    {
        return Err(CrossArchitectureFocalError::ExactControlArtifactMismatch);
    }
    let report: serde_json::Value = serde_json::from_reader(File::open(parity_report)?)?;
    let gates = report
        .get("gates")
        .and_then(serde_json::Value::as_object)
        .ok_or(CrossArchitectureFocalError::ExactControlArtifactMismatch)?;
    let required = [
        "new_native_exact",
        "mlx_r600_selected_actions",
        "mlx_r600_maximum_error",
        "clean_shutdown",
    ];
    if report.get("passed") != Some(&serde_json::Value::Bool(true))
        || required
            .iter()
            .any(|gate| gates.get(*gate) != Some(&serde_json::Value::Bool(true)))
    {
        return Err(CrossArchitectureFocalError::ExactControlArtifactMismatch);
    }
    Ok(())
}

fn verify_r2_backend_parity_receipt(
    path: &Path,
    model: &R2MapModelIdentity,
) -> Result<(), CrossArchitectureFocalError> {
    let mut receipt: serde_json::Value = serde_json::from_reader(File::open(path)?)?;
    let claimed = receipt
        .as_object_mut()
        .and_then(|object| object.remove("receipt_blake3"))
        .and_then(|value| value.as_str().map(str::to_owned))
        .ok_or(CrossArchitectureFocalError::R2BackendParityMismatch)?;
    let object = receipt
        .as_object()
        .ok_or(CrossArchitectureFocalError::R2BackendParityMismatch)?;
    let observed = r2_backend_parity_receipt_identity(object)?;
    let maximum = object
        .get("maximum_absolute_error")
        .and_then(serde_json::Value::as_f64)
        .ok_or(CrossArchitectureFocalError::R2BackendParityMismatch)?;
    let tolerance = object
        .get("tolerance")
        .and_then(serde_json::Value::as_f64)
        .ok_or(CrossArchitectureFocalError::R2BackendParityMismatch)?;
    if claimed != observed
        || object.get("schema_id").and_then(serde_json::Value::as_str)
            != Some("cascadia.r2-map.mlx-numpy-checkpoint-parity.v1")
        || object.get("passed") != Some(&serde_json::Value::Bool(true))
        || object.get("finite") != Some(&serde_json::Value::Bool(true))
        || object
            .get("checkpoint_id")
            .and_then(serde_json::Value::as_str)
            != Some(&model.checkpoint_id)
        || object
            .get("checkpoint_manifest_blake3")
            .and_then(serde_json::Value::as_str)
            != Some(&model.checkpoint_manifest_blake3)
        || object
            .get("model_weights_blake3")
            .and_then(serde_json::Value::as_str)
            != Some(&model.model_weights_blake3)
        || object
            .get("verification_id")
            .and_then(serde_json::Value::as_str)
            != Some(&model.verification_id)
        || !maximum.is_finite()
        || !tolerance.is_finite()
        || tolerance <= 0.0
        || maximum > tolerance
    {
        return Err(CrossArchitectureFocalError::R2BackendParityMismatch);
    }
    Ok(())
}

fn r2_backend_parity_receipt_identity(
    object: &serde_json::Map<String, serde_json::Value>,
) -> Result<String, CrossArchitectureFocalError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"r2-map-mlx-numpy-checkpoint-parity-receipt-v1");
    for name in [
        "schema_id",
        "checkpoint_id",
        "checkpoint_manifest_blake3",
        "model_weights_blake3",
        "verification_id",
    ] {
        let value = object
            .get(name)
            .and_then(serde_json::Value::as_str)
            .ok_or(CrossArchitectureFocalError::R2BackendParityMismatch)?;
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    let finite = object
        .get("finite")
        .and_then(serde_json::Value::as_bool)
        .ok_or(CrossArchitectureFocalError::R2BackendParityMismatch)?;
    let passed = object
        .get("passed")
        .and_then(serde_json::Value::as_bool)
        .ok_or(CrossArchitectureFocalError::R2BackendParityMismatch)?;
    hasher.update(&[u8::from(finite), u8::from(passed)]);
    Ok(hasher.finalize().to_hex().to_string())
}

fn file_blake3(path: &Path) -> Result<String, CrossArchitectureFocalError> {
    let mut stream = File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = stream.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

impl FocalGameExecutor for CrossArchitectureFocalExecutor {
    type Error = CrossArchitectureFocalError;

    fn execute(&mut self, request: &FocalGameRequest) -> Result<FocalGameRecord, Self::Error> {
        self.validate_request(request)?;
        let result = self.execute_game(request)?;
        let resources = self
            .resource_probe
            .observe()
            .map_err(CrossArchitectureFocalError::ResourceProbe)?;
        let checkpoint_load_seconds = match request.identity.arm {
            PairArm::Candidate => self.pending_r2_load_seconds.take().unwrap_or(0.0),
            PairArm::Control => self.pending_exact_load_seconds.take().unwrap_or(0.0),
        };
        Ok(FocalGameRecord::from_match(
            request.identity.clone(),
            request.focal_seat,
            FocalRuntimeObservation {
                checkpoint_load_seconds,
                peak_rss_bytes: resources.peak_rss_bytes,
                swap_delta_bytes: resources.swap_delta_bytes,
                clean_shutdown: resources.clean_shutdown,
            },
            &result,
        )?)
    }

    fn finish(&mut self) -> Result<(), Self::Error> {
        self.finish_services()?;
        let resources = self
            .resource_probe
            .observe()
            .map_err(CrossArchitectureFocalError::ResourceProbe)?;
        if resources.peak_rss_bytes > FOCAL_MAX_RSS_BYTES || resources.swap_delta_bytes > 0 {
            return Err(CrossArchitectureFocalError::FinalResourceGate {
                peak_rss_bytes: resources.peak_rss_bytes,
                swap_delta_bytes: resources.swap_delta_bytes,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum CrossArchitectureFocalError {
    #[error(
        "cross-architecture focal configuration must be R2 versus the qualified exact K32/R600 control"
    )]
    InvalidConfiguration,
    #[error("focal request does not match the frozen cross-architecture contract")]
    InvalidRequest,
    #[error("R2 bundle protocol identities differ from the campaign implementation binding")]
    ProtocolIdentityMismatch,
    #[error("candidate checkpoint is absent from the verified R2 serving bundle: {0}")]
    MissingCandidate(String),
    #[error("cross-architecture service was already shut down")]
    ServiceAlreadyShutdown,
    #[error("qualified exact-NNUE weights or native/MLX rollout-parity evidence differs")]
    ExactControlArtifactMismatch,
    #[error("R2 MLX/NumPy parity receipt differs from the frozen candidate")]
    R2BackendParityMismatch,
    #[error("resource probe failed: {0}")]
    ResourceProbe(std::io::Error),
    #[error(
        "full work-item resource gate failed: peak RSS {peak_rss_bytes} bytes, swap delta {swap_delta_bytes} bytes"
    )]
    FinalResourceGate {
        peak_rss_bytes: u64,
        swap_delta_bytes: i64,
    },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    R2Model(#[from] R2MapModelError),
    #[error(transparent)]
    Bridge(#[from] BridgeError),
    #[error(transparent)]
    Rule(#[from] cascadia_game::RuleError),
    #[error("game simulation failed: {0}")]
    Simulation(SimulationError),
    #[error(transparent)]
    Focal(#[from] FocalBenchmarkError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_eval::{
        focal::{BenchmarkStage, FocalRecordIdentity, OpponentIdentity},
        r2_map_binding::{R2_MAP_OPEN_REFERENCE_SEED_DOMAIN_V1_1, R2MapImplementationBinding},
    };
    use cascadia_game::GameSeed;

    fn binding() -> R2MapImplementationBinding {
        R2MapImplementationBinding::new(
            "11".repeat(32),
            "22".repeat(32),
            "33".repeat(32),
            "44".repeat(32),
            "55".repeat(32),
            "66".repeat(32),
            "77".repeat(32),
            "88".repeat(32),
            "99".repeat(32),
            "aa".repeat(32),
            "bb".repeat(32),
            "cc".repeat(32),
            R2_MAP_OPEN_REFERENCE_SEED_DOMAIN_V1_1.to_owned(),
        )
        .unwrap()
    }

    fn request(arm: PairArm) -> FocalGameRequest {
        FocalGameRequest {
            benchmark_id: "cross-arch-smoke".to_owned(),
            implementation_binding: binding(),
            identity: FocalRecordIdentity {
                stage: BenchmarkStage::StrengthBlindedSmoke,
                pair_index: 3,
                arm,
                focal_checkpoint_id: match arm {
                    PairArm::Candidate => "r2-candidate".to_owned(),
                    PairArm::Control => QUALIFIED_EXACT_NNUE_CHECKPOINT_ID.to_owned(),
                },
                opponents: [0, 1, 2]
                    .into_iter()
                    .map(|seat| OpponentIdentity {
                        seat,
                        checkpoint_id: "greedy-v1".to_owned(),
                    })
                    .collect(),
                field_manifest_id: "all-greedy".to_owned(),
                inference_settings_id: CROSS_ARCH_INFERENCE_SETTINGS_ID.to_owned(),
            },
            game_seed: GameSeed([7; 32]),
            focal_seat: 3,
        }
    }

    #[test]
    fn contract_constants_bind_the_qualified_control() {
        assert!(QUALIFIED_EXACT_NNUE_CHECKPOINT_ID.contains("k32-r600"));
        assert!(CROSS_ARCH_INFERENCE_SETTINGS_ID.contains("qualified-exact-nnue"));
    }

    #[test]
    fn request_shape_keeps_three_greedy_opponents_and_one_focal_policy() {
        for arm in [PairArm::Candidate, PairArm::Control] {
            let request = request(arm);
            assert_eq!(request.identity.opponents.len(), 3);
            assert!(
                request
                    .identity
                    .opponents
                    .iter()
                    .all(|opponent| opponent.checkpoint_id == "greedy-v1")
            );
            assert!(
                request
                    .identity
                    .opponents
                    .iter()
                    .all(|opponent| opponent.seat != request.focal_seat)
            );
        }
    }
}
