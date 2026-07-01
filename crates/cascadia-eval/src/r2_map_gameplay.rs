//! Verified R2-MAP serving adapter for focal and longitudinal benchmarks.
//!
//! The campaign artifact layers describe seats with durable checkpoint ids.
//! This adapter resolves those ids through one pre-verified local serving
//! bundle, constructs a benchmark-only [`R2MapGameRequest`], and delegates the
//! complete game to the production exhaustive R2-MAP runner.  It deliberately
//! performs no candidate filtering and records a bounded, deterministic
//! service restart before retrying the exact same request.

use std::{collections::BTreeMap, time::Instant};

use cascadia_data::{
    R2MapCollectionKind, R2MapExplorationIdentity, R2MapGameIdentity, R2MapGameRequest,
    R2MapGameRunner, R2MapPolicyIdentity, R2MapRecordContext, R2MapRngIdentity, R2MapSeedPurpose,
};
use cascadia_model::{R2MapModelError, R2MapModelProcess, R2MapServingBundle};
use cascadia_search::R2MapLocalGameRunner;
use thiserror::Error;

use crate::{
    focal::{FocalGameRecord, FocalRuntimeObservation, PairArm},
    focal_campaign::{FocalGameExecutor, FocalGameRequest},
    focal_gameplay::RuntimeResourceProbe,
    longitudinal::LongitudinalGameExecutor,
    r2_map_binding::R2MapImplementationBinding,
};

/// Frozen identities that are not inferable from a focal campaign contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct R2MapBenchmarkExecutionConfig {
    pub campaign_id: String,
    pub iteration: u32,
    pub seed_purpose: R2MapSeedPurpose,
    pub implementation_binding: R2MapImplementationBinding,
    /// At most this many process restarts may be consumed by one work item.
    pub maximum_service_restarts: usize,
}

impl<R, P> LongitudinalGameExecutor for R2MapBenchmarkGameplayExecutor<R, P>
where
    R: RestartableR2MapBenchmarkRunner,
    P: RuntimeResourceProbe,
{
    type Error = R2MapBenchmarkGameplayError;

    fn execute_longitudinal(
        &mut self,
        request: &FocalGameRequest,
    ) -> Result<FocalGameRecord, Self::Error> {
        self.execute_benchmark_request(request)
    }
}

impl R2MapBenchmarkExecutionConfig {
    pub fn candidate_gate(
        campaign_id: impl Into<String>,
        iteration: u32,
        implementation_binding: R2MapImplementationBinding,
    ) -> Self {
        Self {
            campaign_id: campaign_id.into(),
            iteration,
            seed_purpose: R2MapSeedPurpose::CandidateGate,
            implementation_binding,
            maximum_service_restarts: 1,
        }
    }

    pub fn longitudinal(
        campaign_id: impl Into<String>,
        iteration: u32,
        implementation_binding: R2MapImplementationBinding,
    ) -> Self {
        Self {
            campaign_id: campaign_id.into(),
            iteration,
            seed_purpose: R2MapSeedPurpose::LongitudinalBenchmark,
            implementation_binding,
            maximum_service_restarts: 1,
        }
    }
}

/// Narrow runner boundary used by fault/restart tests without a Python child.
pub trait RestartableR2MapBenchmarkRunner {
    fn play_benchmark_game(
        &self,
        request: &R2MapGameRequest,
    ) -> Result<cascadia_data::R2MapPlayedGame, cascadia_data::R2MapCollectorError>;

    fn restart_benchmark_service(&self) -> Result<(), R2MapModelError>;
}

impl RestartableR2MapBenchmarkRunner for R2MapLocalGameRunner<R2MapModelProcess> {
    fn play_benchmark_game(
        &self,
        request: &R2MapGameRequest,
    ) -> Result<cascadia_data::R2MapPlayedGame, cascadia_data::R2MapCollectorError> {
        self.play_game(request)
    }

    fn restart_benchmark_service(&self) -> Result<(), R2MapModelError> {
        self.restart_service()
    }
}

pub struct R2MapBenchmarkGameplayExecutor<R, P> {
    runner: R,
    resource_probe: P,
    checkpoint_hashes: BTreeMap<String, [u8; 32]>,
    config: R2MapBenchmarkExecutionConfig,
    pending_checkpoint_load_seconds: f64,
    service_restarts: usize,
}

impl<R, P> R2MapBenchmarkGameplayExecutor<R, P> {
    pub fn new(
        bundle: &R2MapServingBundle,
        runner: R,
        resource_probe: P,
        config: R2MapBenchmarkExecutionConfig,
        checkpoint_load_seconds: f64,
    ) -> Result<Self, R2MapBenchmarkGameplayError> {
        bundle.validate()?;
        config
            .implementation_binding
            .validate()
            .map_err(|_| R2MapBenchmarkGameplayError::InvalidConfig)?;
        if config.campaign_id.trim().is_empty()
            || !checkpoint_load_seconds.is_finite()
            || checkpoint_load_seconds < 0.0
        {
            return Err(R2MapBenchmarkGameplayError::InvalidConfig);
        }
        let checkpoint_hashes = bundle
            .entries
            .iter()
            .map(|entry| {
                Ok((
                    entry.model.checkpoint_id.clone(),
                    decode_hash(&entry.manifest_identity_blake3)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>, R2MapBenchmarkGameplayError>>()?;
        if checkpoint_hashes.len() != bundle.entries.len() {
            return Err(R2MapBenchmarkGameplayError::DuplicateCheckpointId);
        }
        Ok(Self {
            runner,
            resource_probe,
            checkpoint_hashes,
            config,
            pending_checkpoint_load_seconds: checkpoint_load_seconds,
            service_restarts: 0,
        })
    }

    pub fn service_restarts(&self) -> usize {
        self.service_restarts
    }

    pub fn into_runner(self) -> R {
        self.runner
    }

    /// Execute one benchmark game and derive replay-authoritative focal telemetry.
    ///
    /// Longitudinal callers use a control-arm identity because the focal model
    /// is the incumbent/historical checkpoint. Paired gates call this through
    /// [`FocalGameExecutor`], where the candidate arm becomes `Newest` and the
    /// control arm becomes `Historical` (or canonical greedy).
    pub fn execute_benchmark_request(
        &mut self,
        request: &FocalGameRequest,
    ) -> Result<FocalGameRecord, R2MapBenchmarkGameplayError>
    where
        R: RestartableR2MapBenchmarkRunner,
        P: RuntimeResourceProbe,
    {
        let game_request = self.r2_map_request(request)?;
        let started = Instant::now();
        let played = match self.runner.play_benchmark_game(&game_request) {
            Ok(played) => played,
            Err(first) if self.service_restarts < self.config.maximum_service_restarts => {
                self.runner.restart_benchmark_service().map_err(|restart| {
                    R2MapBenchmarkGameplayError::Restart {
                        first: first.to_string(),
                        restart: restart.to_string(),
                    }
                })?;
                self.service_restarts += 1;
                self.runner
                    .play_benchmark_game(&game_request)
                    .map_err(|second| R2MapBenchmarkGameplayError::Retry {
                        first: first.to_string(),
                        second: second.to_string(),
                    })?
            }
            Err(error) => return Err(R2MapBenchmarkGameplayError::Runner(error.to_string())),
        };
        if !played.exploration_draws.is_empty() {
            return Err(R2MapBenchmarkGameplayError::BenchmarkExploration);
        }
        let expected_public_turns =
            cascadia_data::reconstruct_r2_map_public_turns(&played.result.replay)
                .map_err(|error| R2MapBenchmarkGameplayError::Runner(error.to_string()))?;
        if played.public_turns != expected_public_turns {
            return Err(R2MapBenchmarkGameplayError::Runner(
                "runner public market trace differs from replay".to_owned(),
            ));
        }
        let resources = self
            .resource_probe
            .observe()
            .map_err(|error| R2MapBenchmarkGameplayError::ResourceProbe(error.to_string()))?;
        let checkpoint_load_seconds = std::mem::take(&mut self.pending_checkpoint_load_seconds);
        let record = FocalGameRecord::from_match(
            request.identity.clone(),
            request.focal_seat,
            FocalRuntimeObservation {
                checkpoint_load_seconds,
                peak_rss_bytes: resources.peak_rss_bytes,
                swap_delta_bytes: resources.swap_delta_bytes,
                clean_shutdown: resources.clean_shutdown,
            },
            &played.result,
        )?;
        // The MatchResult carries its own end-to-end timing. Keep this clock as
        // an independent guard against a malformed negative/non-finite result.
        if !started.elapsed().as_secs_f64().is_finite() {
            return Err(R2MapBenchmarkGameplayError::InvalidRuntime);
        }
        Ok(record)
    }

    fn r2_map_request(
        &self,
        request: &FocalGameRequest,
    ) -> Result<R2MapGameRequest, R2MapBenchmarkGameplayError> {
        if usize::from(request.focal_seat) >= 4
            || request.implementation_binding != self.config.implementation_binding
        {
            return Err(R2MapBenchmarkGameplayError::InvalidRequest);
        }
        let mut seats = vec![None; 4];
        let focal = if request.identity.focal_checkpoint_id == "greedy-v1" {
            R2MapPolicyIdentity::greedy()
        } else {
            let hash = self.checkpoint_hash(&request.identity.focal_checkpoint_id)?;
            match request.identity.arm {
                PairArm::Candidate => {
                    R2MapPolicyIdentity::newest(request.identity.focal_checkpoint_id.clone(), hash)
                }
                PairArm::Control => R2MapPolicyIdentity::historical(
                    request.identity.focal_checkpoint_id.clone(),
                    hash,
                ),
            }
        };
        seats[usize::from(request.focal_seat)] = Some(focal);
        for opponent in &request.identity.opponents {
            let seat = usize::from(opponent.seat);
            if seat >= seats.len() || seats[seat].is_some() {
                return Err(R2MapBenchmarkGameplayError::InvalidRequest);
            }
            let policy = if opponent.checkpoint_id == "greedy-v1" {
                R2MapPolicyIdentity::greedy()
            } else {
                R2MapPolicyIdentity::historical(
                    opponent.checkpoint_id.clone(),
                    self.checkpoint_hash(&opponent.checkpoint_id)?,
                )
            };
            seats[seat] = Some(policy);
        }
        let seats = seats
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(R2MapBenchmarkGameplayError::InvalidRequest)?;
        let identity = R2MapGameIdentity::new(
            self.config.campaign_id.clone(),
            self.config.iteration,
            "scheduler",
            u64::try_from(request.identity.pair_index)
                .map_err(|_| R2MapBenchmarkGameplayError::InvalidRequest)?,
            request.game_seed,
        );
        Ok(R2MapGameRequest {
            seed: request.game_seed,
            context: R2MapRecordContext {
                collection_kind: R2MapCollectionKind::Benchmark,
                identity,
                seed_purpose: self.config.seed_purpose,
                focal_seat: request.focal_seat,
                seats,
                rng: R2MapRngIdentity::default(),
                exploration: R2MapExplorationIdentity::disabled(),
                protocols: self.config.implementation_binding.protocols.clone(),
            },
        })
    }

    fn checkpoint_hash(
        &self,
        checkpoint_id: &str,
    ) -> Result<[u8; 32], R2MapBenchmarkGameplayError> {
        self.checkpoint_hashes
            .get(checkpoint_id)
            .copied()
            .ok_or_else(|| R2MapBenchmarkGameplayError::UnknownCheckpoint(checkpoint_id.to_owned()))
    }
}

impl<R, P> FocalGameExecutor for R2MapBenchmarkGameplayExecutor<R, P>
where
    R: RestartableR2MapBenchmarkRunner,
    P: RuntimeResourceProbe,
{
    type Error = R2MapBenchmarkGameplayError;

    fn execute(&mut self, request: &FocalGameRequest) -> Result<FocalGameRecord, Self::Error> {
        if request.identity.arm == PairArm::Candidate
            && request
                .identity
                .opponents
                .iter()
                .any(|opponent| opponent.checkpoint_id == request.identity.focal_checkpoint_id)
        {
            return Err(R2MapBenchmarkGameplayError::CandidateInOpponentField);
        }
        self.execute_benchmark_request(request)
    }
}

fn decode_hash(value: &str) -> Result<[u8; 32], R2MapBenchmarkGameplayError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(R2MapBenchmarkGameplayError::InvalidBundleHash);
    }
    let mut output = [0u8; 32];
    for (index, byte) in output.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| R2MapBenchmarkGameplayError::InvalidBundleHash)?;
    }
    Ok(output)
}

#[derive(Debug, Error)]
pub enum R2MapBenchmarkGameplayError {
    #[error("R2-MAP benchmark execution configuration is invalid")]
    InvalidConfig,
    #[error("R2-MAP benchmark request has an invalid host, focal seat, seat map, or W0 binding")]
    InvalidRequest,
    #[error("R2-MAP serving bundle contains a malformed compact hash")]
    InvalidBundleHash,
    #[error("R2-MAP serving bundle repeats a checkpoint id")]
    DuplicateCheckpointId,
    #[error("checkpoint is absent from the verified local serving bundle: {0}")]
    UnknownCheckpoint(String),
    #[error("candidate checkpoint appears in its paired opponent field")]
    CandidateInOpponentField,
    #[error("benchmark runner emitted exploration draws")]
    BenchmarkExploration,
    #[error("benchmark runner failed: {0}")]
    Runner(String),
    #[error("benchmark runner failed ({first}) and its service restart failed ({restart})")]
    Restart { first: String, restart: String },
    #[error("benchmark retry failed after service restart; first={first}; second={second}")]
    Retry { first: String, second: String },
    #[error("runtime resource probe failed: {0}")]
    ResourceProbe(String),
    #[error("benchmark runtime measurement is invalid")]
    InvalidRuntime,
    #[error(transparent)]
    Model(#[from] R2MapModelError),
    #[error(transparent)]
    Experience(#[from] cascadia_data::R2MapExperienceError),
    #[error(transparent)]
    Focal(#[from] crate::focal::FocalBenchmarkError),
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, convert::Infallible};

    use cascadia_game::GameSeed;
    use cascadia_model::{
        R2_MAP_SERVING_BUNDLE_SCHEMA, R2MapModelIdentity, R2MapServingBundleEntry,
    };
    use cascadia_sim::{MatchConfig, StrategyKind, play_match};

    use super::*;
    use crate::{
        focal::{BenchmarkStage, FocalRecordIdentity, OpponentIdentity},
        focal_gameplay::{NoopRuntimeResourceProbe, RuntimeResourceSnapshot},
    };

    struct FailOnceRunner {
        calls: Cell<usize>,
        restarts: Cell<usize>,
    }

    impl RestartableR2MapBenchmarkRunner for FailOnceRunner {
        fn play_benchmark_game(
            &self,
            request: &R2MapGameRequest,
        ) -> Result<cascadia_data::R2MapPlayedGame, cascadia_data::R2MapCollectorError> {
            self.calls.set(self.calls.get() + 1);
            if self.calls.get() == 1 {
                return Err(cascadia_data::R2MapCollectorError::RunnerContract(
                    "injected",
                ));
            }
            let result = play_match(&MatchConfig::symmetric(
                cascadia_game::GameConfig::research_aaaaa(4).unwrap(),
                request.seed,
                StrategyKind::Greedy,
            ))
            .unwrap();
            let public_turns =
                cascadia_data::reconstruct_r2_map_public_turns(&result.replay).unwrap();
            Ok(cascadia_data::R2MapPlayedGame {
                result,
                exploration_draws: Vec::new(),
                public_turns,
            })
        }

        fn restart_benchmark_service(&self) -> Result<(), R2MapModelError> {
            self.restarts.set(self.restarts.get() + 1);
            Ok(())
        }
    }

    fn bundle() -> R2MapServingBundle {
        let temporary = std::env::temp_dir();
        R2MapServingBundle {
            schema_version: 2,
            schema_id: R2_MAP_SERVING_BUNDLE_SCHEMA.to_owned(),
            protocols: cascadia_data::R2MapProtocolIdentity {
                collector_hash: [1; 32],
                source_hash: [2; 32],
                serving_protocol_hash: [3; 32],
            },
            entries: vec![R2MapServingBundleEntry {
                manifest_identity_blake3: "a".repeat(64),
                run_dir: temporary.clone(),
                checkpoint_path: temporary.join("candidate-v1"),
                model: R2MapModelIdentity {
                    checkpoint_id: "candidate-v1".to_owned(),
                    checkpoint_manifest_blake3: "b".repeat(64),
                    model_config_blake3: "c".repeat(64),
                    model_weights_blake3: "d".repeat(64),
                    verification_id: "e".repeat(64),
                },
                pinned: true,
            }],
        }
    }

    fn request() -> FocalGameRequest {
        let focal_seat = 0;
        FocalGameRequest {
            benchmark_id: "restart-smoke".to_owned(),
            implementation_binding: implementation_binding(),
            identity: FocalRecordIdentity {
                stage: BenchmarkStage::Development,
                pair_index: 0,
                arm: PairArm::Candidate,
                focal_checkpoint_id: "candidate-v1".to_owned(),
                opponents: [1, 2, 3]
                    .into_iter()
                    .map(|seat| OpponentIdentity {
                        seat,
                        checkpoint_id: "greedy-v1".to_owned(),
                    })
                    .collect(),
                field_manifest_id: "greedy-field-v1".to_owned(),
                inference_settings_id: "argmax-v1".to_owned(),
            },
            game_seed: GameSeed::from_u64(42),
            focal_seat,
        }
    }

    fn implementation_binding() -> R2MapImplementationBinding {
        R2MapImplementationBinding::new(
            "11".repeat(32),
            "22".repeat(32),
            "33".repeat(32),
            "44".repeat(32),
            "55".repeat(32),
            "66".repeat(32),
            "88".repeat(32),
            "99".repeat(32),
            "aa".repeat(32),
            "bb".repeat(32),
            "cc".repeat(32),
            "77".repeat(32),
            "r2-map-open-reference-performance-100-v1".to_owned(),
        )
        .unwrap()
    }

    #[test]
    fn retries_the_exact_benchmark_request_once_after_service_failure() {
        let runner = FailOnceRunner {
            calls: Cell::new(0),
            restarts: Cell::new(0),
        };
        let mut executor = R2MapBenchmarkGameplayExecutor::new(
            &bundle(),
            runner,
            NoopRuntimeResourceProbe,
            R2MapBenchmarkExecutionConfig::candidate_gate("campaign", 0, implementation_binding()),
            0.25,
        )
        .unwrap();
        let record = executor.execute(&request()).unwrap();
        assert_eq!(executor.service_restarts(), 1);
        assert_eq!(record.runtime.checkpoint_load_seconds, 0.25);
        assert_eq!(record.game_seed, request().game_seed);
        assert_eq!(record.focal_decision_seconds.len(), 20);
    }

    #[test]
    fn candidate_cannot_enter_its_own_opponent_field() {
        let mut invalid = request();
        invalid.identity.opponents[0].checkpoint_id = "candidate-v1".to_owned();
        let runner = FailOnceRunner {
            calls: Cell::new(1),
            restarts: Cell::new(0),
        };
        let mut executor = R2MapBenchmarkGameplayExecutor::new(
            &bundle(),
            runner,
            NoopRuntimeResourceProbe,
            R2MapBenchmarkExecutionConfig::candidate_gate("campaign", 0, implementation_binding()),
            0.0,
        )
        .unwrap();
        assert!(matches!(
            executor.execute(&invalid),
            Err(R2MapBenchmarkGameplayError::CandidateInOpponentField)
        ));
    }

    #[test]
    fn request_cannot_substitute_a_different_registered_panel() {
        let mut invalid = request();
        invalid.implementation_binding.maximum_width_panel_sha256 = "88".repeat(32);
        let runner = FailOnceRunner {
            calls: Cell::new(1),
            restarts: Cell::new(0),
        };
        let mut executor = R2MapBenchmarkGameplayExecutor::new(
            &bundle(),
            runner,
            NoopRuntimeResourceProbe,
            R2MapBenchmarkExecutionConfig::candidate_gate("campaign", 0, implementation_binding()),
            0.0,
        )
        .unwrap();
        assert!(matches!(
            executor.execute(&invalid),
            Err(R2MapBenchmarkGameplayError::InvalidRequest)
        ));
    }

    #[allow(dead_code)]
    fn _probe_error_is_send_sync(_: Infallible, _: RuntimeResourceSnapshot) {}
}
