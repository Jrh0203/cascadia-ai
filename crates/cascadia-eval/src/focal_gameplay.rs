//! Production gameplay bridge for focal benchmark campaign requests.
//!
//! The bridge builds one frozen strategy per seat, preserves registered seat
//! identities in the replay, and converts the completed match through the
//! replay-verifying [`FocalGameRecord::from_match`] path. MLX serving plugs in
//! by implementing [`FrozenSeatStrategyFactory`]; the benchmark artifact layer
//! does not need to know how a strategy is served.

use std::{collections::BTreeSet, error::Error, io, mem::MaybeUninit, time::Instant};

#[cfg(target_os = "macos")]
use std::process::Command;

use cascadia_game::{GameConfig, GameSeed, GameState, TurnAction};
use cascadia_sim::{
    SimulationError, play_match_with_seat_selector, select_greedy_action, strategy_rng,
};
use rand_chacha::ChaCha8Rng;
use thiserror::Error;

use crate::{
    focal::{FocalGameRecord, FocalRuntimeObservation, PairArm, validate_focal_record},
    focal_campaign::{FocalGameExecutor, FocalGameRequest},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeatStrategyBuildRequest {
    pub checkpoint_id: String,
    pub rng_domain_id: String,
    pub game_seed: GameSeed,
    pub seat: usize,
    pub focal: bool,
}

pub trait FrozenSeatStrategy {
    fn select_action(&mut self, game: &GameState) -> Result<TurnAction, SimulationError>;
}

pub trait FrozenSeatStrategyFactory {
    type Strategy: FrozenSeatStrategy;
    type Error: Error + Send + Sync + 'static;

    fn build(&mut self, request: &SeatStrategyBuildRequest) -> Result<Self::Strategy, Self::Error>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeResourceSnapshot {
    pub peak_rss_bytes: u64,
    pub swap_delta_bytes: i64,
    pub clean_shutdown: bool,
}

pub trait RuntimeResourceProbe {
    type Error: Error + Send + Sync + 'static;

    fn observe(&mut self) -> Result<RuntimeResourceSnapshot, Self::Error>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NoopRuntimeResourceProbe;

impl RuntimeResourceProbe for NoopRuntimeResourceProbe {
    type Error = std::convert::Infallible;

    fn observe(&mut self) -> Result<RuntimeResourceSnapshot, Self::Error> {
        Ok(RuntimeResourceSnapshot {
            peak_rss_bytes: 0,
            swap_delta_bytes: 0,
            clean_shutdown: true,
        })
    }
}

/// Cross-platform process RSS and system-swap probe used by benchmark workers.
#[derive(Debug, Clone, Copy)]
pub struct SystemRuntimeResourceProbe {
    initial_swap_bytes: u64,
}

impl SystemRuntimeResourceProbe {
    pub fn new() -> Result<Self, io::Error> {
        Ok(Self {
            initial_swap_bytes: system_swap_used_bytes()?,
        })
    }
}

impl RuntimeResourceProbe for SystemRuntimeResourceProbe {
    type Error = io::Error;

    fn observe(&mut self) -> Result<RuntimeResourceSnapshot, Self::Error> {
        let swap_bytes = system_swap_used_bytes()?;
        Ok(RuntimeResourceSnapshot {
            peak_rss_bytes: process_peak_rss_bytes()?,
            swap_delta_bytes: i64::try_from(swap_bytes)
                .unwrap_or(i64::MAX)
                .saturating_sub(i64::try_from(self.initial_swap_bytes).unwrap_or(i64::MAX)),
            clean_shutdown: true,
        })
    }
}

/// Backward-compatible name retained for existing host-local callers.
pub type MacOsRuntimeResourceProbe = SystemRuntimeResourceProbe;

pub struct FocalGameplayExecutor<F, P = NoopRuntimeResourceProbe> {
    factory: F,
    resource_probe: P,
}

impl<F> FocalGameplayExecutor<F, NoopRuntimeResourceProbe> {
    pub fn new(factory: F) -> Self {
        Self {
            factory,
            resource_probe: NoopRuntimeResourceProbe,
        }
    }
}

impl<F, P> FocalGameplayExecutor<F, P> {
    pub fn with_resource_probe(factory: F, resource_probe: P) -> Self {
        Self {
            factory,
            resource_probe,
        }
    }

    pub fn factory(&self) -> &F {
        &self.factory
    }
}

impl<F, P> FocalGameExecutor for FocalGameplayExecutor<F, P>
where
    F: FrozenSeatStrategyFactory,
    P: RuntimeResourceProbe,
{
    type Error = FocalGameplayError;

    fn execute(&mut self, request: &FocalGameRequest) -> Result<FocalGameRecord, Self::Error> {
        validate_request(request)?;
        let identities = seat_identities(request)?;
        let load_started = Instant::now();
        let mut strategies = identities
            .iter()
            .enumerate()
            .map(|(seat, checkpoint_id)| {
                let focal = seat == usize::from(request.focal_seat);
                let rng_domain_id = if focal {
                    format!(
                        "{}:{}:paired-focal",
                        request.benchmark_id, request.identity.field_manifest_id
                    )
                } else {
                    format!(
                        "{}:opponent:{checkpoint_id}:seat-{seat}",
                        request.identity.field_manifest_id
                    )
                };
                self.factory
                    .build(&SeatStrategyBuildRequest {
                        checkpoint_id: checkpoint_id.clone(),
                        rng_domain_id,
                        game_seed: request.game_seed,
                        seat,
                        focal,
                    })
                    .map_err(|error| FocalGameplayError::Factory {
                        seat,
                        checkpoint_id: checkpoint_id.clone(),
                        detail: error.to_string(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let checkpoint_load_seconds = load_started.elapsed().as_secs_f64();
        let game_config = GameConfig::research_aaaaa(4)?;
        let result = play_match_with_seat_selector(
            game_config,
            request.game_seed,
            &identities,
            |seat, game| strategies[seat].select_action(game),
        )?;
        let resources = self
            .resource_probe
            .observe()
            .map_err(|error| FocalGameplayError::ResourceProbe(error.to_string()))?;
        let record = FocalGameRecord::from_match(
            request.identity.clone(),
            request.focal_seat,
            FocalRuntimeObservation {
                checkpoint_load_seconds,
                peak_rss_bytes: resources.peak_rss_bytes,
                swap_delta_bytes: resources.swap_delta_bytes,
                clean_shutdown: resources.clean_shutdown,
            },
            &result,
        )?;
        validate_focal_record(&record)?;
        Ok(record)
    }
}

fn validate_request(request: &FocalGameRequest) -> Result<(), FocalGameplayError> {
    if request.identity.arm == PairArm::Candidate
        && request
            .identity
            .opponents
            .iter()
            .any(|opponent| opponent.checkpoint_id == request.identity.focal_checkpoint_id)
    {
        return Err(FocalGameplayError::CandidateInOpponentSeat(
            request.identity.pair_index,
        ));
    }
    if request.identity.opponents.len() != 3 {
        return Err(FocalGameplayError::OpponentCount(
            request.identity.opponents.len(),
        ));
    }
    Ok(())
}

fn seat_identities(request: &FocalGameRequest) -> Result<Vec<String>, FocalGameplayError> {
    let mut identities = vec![None; 4];
    identities[usize::from(request.focal_seat)] =
        Some(request.identity.focal_checkpoint_id.clone());
    for opponent in &request.identity.opponents {
        let seat = usize::from(opponent.seat);
        if seat >= identities.len() || identities[seat].is_some() {
            return Err(FocalGameplayError::OpponentSeat(opponent.seat));
        }
        identities[seat] = Some(opponent.checkpoint_id.clone());
    }
    identities
        .into_iter()
        .enumerate()
        .map(|(seat, identity)| identity.ok_or(FocalGameplayError::MissingSeat(seat)))
        .collect()
}

#[derive(Debug, Clone)]
pub struct LocalGreedyStrategyFactory {
    allowed_checkpoint_ids: BTreeSet<String>,
}

impl LocalGreedyStrategyFactory {
    pub fn new(ids: impl IntoIterator<Item = String>) -> Result<Self, LocalStrategyError> {
        let allowed_checkpoint_ids = ids.into_iter().collect::<BTreeSet<_>>();
        if allowed_checkpoint_ids.is_empty()
            || allowed_checkpoint_ids
                .iter()
                .any(|identity| identity.trim().is_empty())
        {
            return Err(LocalStrategyError::EmptyRegistry);
        }
        Ok(Self {
            allowed_checkpoint_ids,
        })
    }
}

pub struct LocalGreedyStrategy {
    rng: ChaCha8Rng,
}

impl FrozenSeatStrategy for LocalGreedyStrategy {
    fn select_action(&mut self, game: &GameState) -> Result<TurnAction, SimulationError> {
        let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
        select_greedy_action(game, &prelude, &mut self.rng)
    }
}

impl FrozenSeatStrategyFactory for LocalGreedyStrategyFactory {
    type Strategy = LocalGreedyStrategy;
    type Error = LocalStrategyError;

    fn build(&mut self, request: &SeatStrategyBuildRequest) -> Result<Self::Strategy, Self::Error> {
        if !self.allowed_checkpoint_ids.contains(&request.checkpoint_id) {
            return Err(LocalStrategyError::UnknownCheckpoint(
                request.checkpoint_id.clone(),
            ));
        }
        Ok(LocalGreedyStrategy {
            rng: strategy_rng(request.game_seed, request.seat, &request.rng_domain_id),
        })
    }
}

fn process_peak_rss_bytes() -> Result<u64, io::Error> {
    let mut usage = MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `usage` points to writable storage for `rusage`; getrusage
    // initializes it on success and does not retain the pointer.
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if status != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: status zero guarantees getrusage initialized the structure.
    let usage = unsafe { usage.assume_init() };
    #[cfg(target_os = "macos")]
    let bytes = u64::try_from(usage.ru_maxrss).unwrap_or(0);
    #[cfg(target_os = "linux")]
    let bytes = (u64::try_from(usage.ru_maxrss).unwrap_or(0) * 1_024)
        .max(linux_cgroup_memory_peak_bytes()?.unwrap_or(0));
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    let bytes = u64::try_from(usage.ru_maxrss).unwrap_or(0) * 1_024;
    Ok(bytes)
}

#[cfg(target_os = "linux")]
fn linux_cgroup_memory_peak_bytes() -> Result<Option<u64>, io::Error> {
    match std::fs::read_to_string("/sys/fs/cgroup/memory.peak") {
        Ok(value) => parse_linux_cgroup_memory_peak(&value).map(Some),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

#[cfg(target_os = "linux")]
fn parse_linux_cgroup_memory_peak(value: &str) -> Result<u64, io::Error> {
    value
        .trim()
        .parse::<u64>()
        .map_err(|error| io::Error::other(format!("invalid cgroup memory peak: {error}")))
}

fn system_swap_used_bytes() -> Result<u64, io::Error> {
    #[cfg(target_os = "macos")]
    {
        let output = Command::new("/usr/sbin/sysctl")
            .args(["-n", "vm.swapusage"])
            .output()?;
        if !output.status.success() {
            return Err(io::Error::other("sysctl vm.swapusage failed"));
        }
        parse_swap_used(&String::from_utf8_lossy(&output.stdout))
    }
    #[cfg(not(target_os = "macos"))]
    {
        linux_swap_used_bytes()
    }
}

#[cfg(target_os = "linux")]
fn linux_swap_used_bytes() -> Result<u64, io::Error> {
    let meminfo = std::fs::read_to_string("/proc/meminfo")?;
    linux_swap_used_bytes_from(&meminfo)
}

#[cfg(target_os = "linux")]
fn linux_swap_used_bytes_from(meminfo: &str) -> Result<u64, io::Error> {
    let kib = |field: &str| -> Result<u64, io::Error> {
        meminfo
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                (name == field).then(|| value.split_whitespace().next()?.parse::<u64>().ok())?
            })
            .ok_or_else(|| io::Error::other(format!("/proc/meminfo omits {field}")))
    };
    Ok(kib("SwapTotal")?.saturating_sub(kib("SwapFree")?) * 1_024)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn linux_swap_used_bytes() -> Result<u64, io::Error> {
    Ok(0)
}

#[cfg(target_os = "macos")]
fn parse_swap_used(value: &str) -> Result<u64, io::Error> {
    let used = value
        .split_whitespace()
        .collect::<Vec<_>>()
        .windows(3)
        .find_map(|window| (window[0] == "used" && window[1] == "=").then_some(window[2]))
        .ok_or_else(|| io::Error::other("vm.swapusage does not contain used value"))?;
    let (number, multiplier) = match used.chars().last() {
        Some('K') => (&used[..used.len() - 1], 1_024.0),
        Some('M') => (&used[..used.len() - 1], 1_048_576.0),
        Some('G') => (&used[..used.len() - 1], 1_073_741_824.0),
        _ => (used, 1.0),
    };
    let parsed = number
        .parse::<f64>()
        .map_err(|error| io::Error::other(error.to_string()))?;
    Ok((parsed * multiplier).round() as u64)
}

#[derive(Debug, Error)]
pub enum LocalStrategyError {
    #[error("local strategy registry must contain only non-empty checkpoint ids")]
    EmptyRegistry,
    #[error("checkpoint is not registered for local greedy serving: {0}")]
    UnknownCheckpoint(String),
}

#[derive(Debug, Error)]
pub enum FocalGameplayError {
    #[error("candidate checkpoint appears in an opponent seat for pair {0}")]
    CandidateInOpponentSeat(usize),
    #[error("expected three opponents, found {0}")]
    OpponentCount(usize),
    #[error("invalid or duplicate opponent seat {0}")]
    OpponentSeat(u8),
    #[error("seat {0} has no registered strategy")]
    MissingSeat(usize),
    #[error("failed to load seat {seat} checkpoint {checkpoint_id}: {detail}")]
    Factory {
        seat: usize,
        checkpoint_id: String,
        detail: String,
    },
    #[error("runtime resource probe failed: {0}")]
    ResourceProbe(String),
    #[error(transparent)]
    Rules(#[from] cascadia_game::RuleError),
    #[error(transparent)]
    Simulation(#[from] SimulationError),
    #[error(transparent)]
    Benchmark(#[from] crate::focal::FocalBenchmarkError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        focal::{BenchmarkStage, FocalRecordIdentity, OpponentIdentity},
        focal_campaign::FocalGameRequest,
        r2_map_binding::R2MapImplementationBinding,
    };

    fn request(arm: PairArm) -> FocalGameRequest {
        let focal_seat = 1;
        FocalGameRequest {
            benchmark_id: "greedy-integrity-smoke-v1".to_owned(),
            implementation_binding: R2MapImplementationBinding::new(
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
            .unwrap(),
            identity: FocalRecordIdentity {
                stage: BenchmarkStage::StrengthBlindedSmoke,
                pair_index: 1,
                arm,
                focal_checkpoint_id: match arm {
                    PairArm::Candidate => "greedy-candidate-v1",
                    PairArm::Control => "greedy-control-v1",
                }
                .to_owned(),
                opponents: [0, 2, 3]
                    .into_iter()
                    .map(|seat| OpponentIdentity {
                        seat,
                        checkpoint_id: format!("greedy-opponent-{seat}"),
                    })
                    .collect(),
                field_manifest_id: "greedy-field-v1".to_owned(),
                inference_settings_id: "deterministic-greedy-v1".to_owned(),
            },
            game_seed: GameSeed::from_u64(987_001),
            focal_seat,
        }
    }

    fn factory() -> LocalGreedyStrategyFactory {
        LocalGreedyStrategyFactory::new(
            [
                "greedy-candidate-v1",
                "greedy-control-v1",
                "greedy-opponent-0",
                "greedy-opponent-2",
                "greedy-opponent-3",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap()
    }

    #[test]
    fn paired_local_greedy_arms_have_identical_gameplay_and_exact_accounting() {
        let mut executor = FocalGameplayExecutor::new(factory());
        let candidate = executor.execute(&request(PairArm::Candidate)).unwrap();
        let control = executor.execute(&request(PairArm::Control)).unwrap();
        assert_eq!(candidate.score, control.score);
        assert_eq!(candidate.pinecones, control.pinecones);
        assert_eq!(candidate.final_state_hash, control.final_state_hash);
        assert_eq!(candidate.replay_blake3, control.replay_blake3);
        assert_eq!(candidate.focal_decision_seconds.len(), 20);
        assert!(candidate.pinecones.conservation_holds());
    }

    #[test]
    fn candidate_checkpoint_in_opponent_seat_is_rejected_before_gameplay() {
        let mut invalid = request(PairArm::Candidate);
        invalid.identity.opponents[0].checkpoint_id = invalid.identity.focal_checkpoint_id.clone();
        let mut executor = FocalGameplayExecutor::new(factory());
        assert!(matches!(
            executor.execute(&invalid),
            Err(FocalGameplayError::CandidateInOpponentSeat(1))
        ));
    }

    #[test]
    fn unknown_local_checkpoint_fails_closed() {
        let mut invalid = request(PairArm::Control);
        invalid.identity.focal_checkpoint_id = "unregistered".to_owned();
        let mut executor = FocalGameplayExecutor::new(factory());
        assert!(matches!(
            executor.execute(&invalid),
            Err(FocalGameplayError::Factory { seat: 1, .. })
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_swap_parser_reports_used_bytes() {
        assert_eq!(
            linux_swap_used_bytes_from(
                "MemTotal:       16384 kB\nSwapTotal:       8192 kB\nSwapFree:        6144 kB\n"
            )
            .unwrap(),
            2 * 1024 * 1024
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_cgroup_peak_parser_reports_container_bytes() {
        assert_eq!(
            parse_linux_cgroup_memory_peak("4294967296\n").unwrap(),
            4_294_967_296
        );
        assert!(parse_linux_cgroup_memory_peak("max\n").is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn swap_parser_accepts_native_sysctl_shape() {
        assert_eq!(
            parse_swap_used("total = 2048.00M  used = 12.50M  free = 2035.50M").unwrap(),
            13_107_200
        );
    }
}
