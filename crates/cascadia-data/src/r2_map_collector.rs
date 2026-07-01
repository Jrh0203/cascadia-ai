//! Resumable, immutable trajectory collection for R2-MAP.
//!
//! This module owns dataset/shard transactions and leaves action selection
//! behind [`R2MapGameRunner`].  The bootstrap runner is available now; W4 can
//! implement the same boundary with four frozen local model handles without
//! changing dataset identity or recovery semantics.

use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use blake3::Hasher;
use cascadia_game::{GameConfig, GameSeed};
use cascadia_sim::{MatchConfig, MatchResult, StrategyKind, play_match};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    R2_MAP_EXPERIENCE_SCHEMA_VERSION, R2MapCollectionKind, R2MapExperienceError,
    R2MapExplorationDraw, R2MapExplorationIdentity, R2MapGameRecord, R2MapPolicyIdentity,
    R2MapPolicyRole, R2MapProtocolIdentity, R2MapRecordContext, R2MapRngIdentity, R2MapSeedLease,
    R2MapSeedPurpose, assign_iterative_seats, focal_seat_for_game, read_r2_map_shard,
    reconstruct_r2_map_public_turns, validate_r2_map_record_batch, write_r2_map_shard,
};

pub const R2_MAP_COLLECTOR_SCHEMA_VERSION: u16 = 1;
pub const R2_MAP_COLLECTOR_MANIFEST_FILE: &str = "dataset.json";
const COLLECTOR_LOCK_FILE: &str = ".collector.lock";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum R2MapCollectorPolicy {
    BootstrapGreedy,
    Iterative {
        newest: R2MapPolicyIdentity,
        opponent_pool: Vec<R2MapPolicyIdentity>,
    },
}

impl R2MapCollectorPolicy {
    fn collection_kind(&self) -> R2MapCollectionKind {
        match self {
            Self::BootstrapGreedy => R2MapCollectionKind::Bootstrap,
            Self::Iterative { .. } => R2MapCollectionKind::IterativeTraining,
        }
    }

    fn seats(
        &self,
        identity: &crate::R2MapGameIdentity,
        seed: GameSeed,
    ) -> Result<Vec<R2MapPolicyIdentity>, R2MapCollectorError> {
        match self {
            Self::BootstrapGreedy => Ok(vec![R2MapPolicyIdentity::greedy(); 4]),
            Self::Iterative {
                newest,
                opponent_pool,
            } => Ok(assign_iterative_seats(
                identity,
                seed,
                newest.clone(),
                opponent_pool,
            )?),
        }
    }
}

#[derive(Debug, Clone)]
pub struct R2MapCollectorConfig {
    pub output: PathBuf,
    pub lease: R2MapSeedLease,
    pub policy: R2MapCollectorPolicy,
    pub shard_games: usize,
    pub resume: bool,
    pub protocols: R2MapProtocolIdentity,
    pub exploration: R2MapExplorationIdentity,
}

impl R2MapCollectorConfig {
    pub fn bootstrap(
        output: PathBuf,
        lease: R2MapSeedLease,
        shard_games: usize,
        resume: bool,
        protocols: R2MapProtocolIdentity,
    ) -> Self {
        Self {
            output,
            lease,
            policy: R2MapCollectorPolicy::BootstrapGreedy,
            shard_games,
            resume,
            protocols,
            exploration: R2MapExplorationIdentity::disabled(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn iterative(
        output: PathBuf,
        lease: R2MapSeedLease,
        shard_games: usize,
        resume: bool,
        protocols: R2MapProtocolIdentity,
        newest: R2MapPolicyIdentity,
        opponent_pool: Vec<R2MapPolicyIdentity>,
        temperature_parts_per_million: u32,
    ) -> Self {
        let iteration = lease.iteration;
        Self {
            output,
            lease,
            policy: R2MapCollectorPolicy::Iterative {
                newest,
                opponent_pool,
            },
            shard_games,
            resume,
            protocols,
            exploration: R2MapExplorationIdentity::training(
                iteration,
                temperature_parts_per_million,
            ),
        }
    }

    fn validate(&self) -> Result<(), R2MapCollectorError> {
        self.lease.validate()?;
        self.protocols.validate()?;
        if self.shard_games == 0 {
            return Err(R2MapCollectorError::InvalidConfig(
                "shard game count must be positive",
            ));
        }
        match (&self.policy, self.lease.purpose) {
            (R2MapCollectorPolicy::BootstrapGreedy, R2MapSeedPurpose::Bootstrap) => {
                if self.exploration != R2MapExplorationIdentity::disabled() {
                    return Err(R2MapCollectorError::InvalidConfig(
                        "bootstrap exploration must be disabled",
                    ));
                }
            }
            (
                R2MapCollectorPolicy::Iterative {
                    newest,
                    opponent_pool,
                },
                R2MapSeedPurpose::Generation,
            ) => {
                newest.validate()?;
                for opponent in opponent_pool {
                    opponent.validate()?;
                }
                if !self.exploration.enabled {
                    return Err(R2MapCollectorError::InvalidConfig(
                        "iterative collection requires exploration",
                    ));
                }
            }
            _ => {
                return Err(R2MapCollectorError::InvalidConfig(
                    "collector policy and seed purpose disagree",
                ));
            }
        }
        Ok(())
    }

    fn request(&self, global_game_index: u64) -> Result<R2MapGameRequest, R2MapCollectorError> {
        let seed = self.lease.seed(global_game_index)?;
        let identity = self.lease.game_identity(global_game_index)?;
        let focal_seat = focal_seat_for_game(global_game_index);
        let seats = self.policy.seats(&identity, seed)?;
        Ok(R2MapGameRequest {
            seed,
            context: R2MapRecordContext {
                collection_kind: self.policy.collection_kind(),
                identity,
                seed_purpose: self.lease.purpose,
                focal_seat,
                seats,
                rng: R2MapRngIdentity::default(),
                exploration: self.exploration.clone(),
                protocols: self.protocols.clone(),
            },
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct R2MapGameRequest {
    pub seed: GameSeed,
    pub context: R2MapRecordContext,
}

#[derive(Debug, Clone)]
pub struct R2MapPlayedGame {
    pub result: MatchResult,
    pub exploration_draws: Vec<R2MapExplorationDraw>,
    pub public_turns: Vec<crate::R2MapPublicTurnTrace>,
}

/// Typed collection boundary for the future four-local-model serving layer.
///
/// Implementations must use the exact identities in `request.context.seats`,
/// must return the same sealed seed, and must report deterministic exploration
/// draws for focal actions. The record constructor independently rejects any
/// mismatch.
pub trait R2MapGameRunner: Sync {
    fn play_game(&self, request: &R2MapGameRequest)
    -> Result<R2MapPlayedGame, R2MapCollectorError>;

    fn play_batch(
        &self,
        requests: &[R2MapGameRequest],
    ) -> Result<Vec<R2MapPlayedGame>, R2MapCollectorError> {
        requests
            .par_iter()
            .map(|request| self.play_game(request))
            .collect()
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct GreedyBootstrapRunner;

impl R2MapGameRunner for GreedyBootstrapRunner {
    fn play_game(
        &self,
        request: &R2MapGameRequest,
    ) -> Result<R2MapPlayedGame, R2MapCollectorError> {
        if request.context.collection_kind != R2MapCollectionKind::Bootstrap
            || request.context.exploration != R2MapExplorationIdentity::disabled()
            || request.context.seats != vec![R2MapPolicyIdentity::greedy(); 4]
        {
            return Err(R2MapCollectorError::RunnerContract(
                "greedy bootstrap runner received a non-bootstrap request",
            ));
        }
        let result = play_match(&MatchConfig::symmetric(
            GameConfig::research_aaaaa(4)?,
            request.seed,
            StrategyKind::Greedy,
        ))?;
        let public_turns = reconstruct_r2_map_public_turns(&result.replay)?;
        Ok(R2MapPlayedGame {
            result,
            exploration_draws: Vec::new(),
            public_turns,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct R2MapCollectorShardManifest {
    pub file: String,
    pub first_game_index: u64,
    pub game_count: usize,
    pub primary_example_count: usize,
    pub byte_count: u64,
    pub blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct R2MapCollectorManifest {
    pub schema_version: u16,
    pub trajectory_schema_version: u16,
    pub dataset_id: String,
    pub game: GameConfig,
    pub collection_kind: R2MapCollectionKind,
    pub lease: R2MapSeedLease,
    pub policy: R2MapCollectorPolicy,
    pub exploration: R2MapExplorationIdentity,
    pub rng: R2MapRngIdentity,
    pub protocols: R2MapProtocolIdentity,
    pub shard_games: usize,
    pub requested_games: usize,
    pub completed_games: usize,
    pub primary_example_count: usize,
    pub created_unix_seconds: u64,
    pub updated_unix_seconds: u64,
    pub shards: Vec<R2MapCollectorShardManifest>,
}

impl R2MapCollectorManifest {
    fn new(config: &R2MapCollectorConfig) -> Result<Self, R2MapCollectorError> {
        let now = unix_seconds()?;
        let mut manifest = Self {
            schema_version: R2_MAP_COLLECTOR_SCHEMA_VERSION,
            trajectory_schema_version: R2_MAP_EXPERIENCE_SCHEMA_VERSION,
            dataset_id: String::new(),
            game: GameConfig::research_aaaaa(4)?,
            collection_kind: config.policy.collection_kind(),
            lease: config.lease.clone(),
            policy: config.policy.clone(),
            exploration: config.exploration.clone(),
            rng: R2MapRngIdentity::default(),
            protocols: config.protocols.clone(),
            shard_games: config.shard_games,
            requested_games: usize::try_from(config.lease.game_count)
                .map_err(|_| R2MapCollectorError::InvalidConfig("game count exceeds usize"))?,
            completed_games: 0,
            primary_example_count: 0,
            created_unix_seconds: now,
            updated_unix_seconds: now,
            shards: Vec::new(),
        };
        manifest.dataset_id = dataset_id(&manifest)?;
        Ok(manifest)
    }

    fn contract_eq(&self, expected: &Self) -> bool {
        self.schema_version == expected.schema_version
            && self.trajectory_schema_version == expected.trajectory_schema_version
            && self.dataset_id == expected.dataset_id
            && self.game == expected.game
            && self.collection_kind == expected.collection_kind
            && self.lease == expected.lease
            && self.policy == expected.policy
            && self.exploration == expected.exploration
            && self.rng == expected.rng
            && self.protocols == expected.protocols
            && self.shard_games == expected.shard_games
            && self.requested_games == expected.requested_games
    }
}

struct CollectorLock {
    _path: PathBuf,
    file: File,
}

impl CollectorLock {
    fn acquire(root: &Path) -> Result<Self, R2MapCollectorError> {
        let path = root.join(COLLECTOR_LOCK_FILE);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        match file.try_lock() {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => {
                return Err(R2MapCollectorError::CollectionLocked(path));
            }
            Err(std::fs::TryLockError::Error(error)) => return Err(error.into()),
        }
        file.set_len(0)?;
        writeln!(file, "pid={}", std::process::id())?;
        file.sync_all()?;
        sync_directory(root)?;
        Ok(Self { _path: path, file })
    }
}

impl Drop for CollectorLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

pub struct R2MapCollectorWriter {
    root: PathBuf,
    manifest_path: PathBuf,
    manifest: R2MapCollectorManifest,
    _lock: CollectorLock,
}

impl R2MapCollectorWriter {
    pub fn open(config: &R2MapCollectorConfig) -> Result<Self, R2MapCollectorError> {
        config.validate()?;
        fs::create_dir_all(&config.output)?;
        let lock = CollectorLock::acquire(&config.output)?;
        let manifest_path = config.output.join(R2_MAP_COLLECTOR_MANIFEST_FILE);
        let expected = R2MapCollectorManifest::new(config)?;
        let manifest = if manifest_path.exists() {
            if !config.resume {
                return Err(R2MapCollectorError::DatasetExists(config.output.clone()));
            }
            let manifest: R2MapCollectorManifest =
                serde_json::from_reader(BufReader::new(File::open(&manifest_path)?))?;
            if !manifest.contract_eq(&expected) {
                return Err(R2MapCollectorError::ResumeDrift);
            }
            validate_r2_map_collector_dataset_inner(&config.output, &manifest, true)?;
            remove_unfinished_next_shard(&config.output, &manifest)?;
            manifest
        } else {
            ensure_new_dataset_directory_is_empty(&config.output)?;
            write_manifest_atomic(&manifest_path, &expected)?;
            expected
        };
        Ok(Self {
            root: config.output.clone(),
            manifest_path,
            manifest,
            _lock: lock,
        })
    }

    pub fn manifest(&self) -> &R2MapCollectorManifest {
        &self.manifest
    }

    fn append_shard(&mut self, records: &[R2MapGameRecord]) -> Result<(), R2MapCollectorError> {
        if records.is_empty() || records.len() > self.manifest.shard_games {
            return Err(R2MapCollectorError::InvalidConfig(
                "shard record count is empty or exceeds the configured bound",
            ));
        }
        validate_r2_map_record_batch(records)?;
        let expected_first = self.manifest.lease.first_game_index
            + u64::try_from(self.manifest.completed_games)
                .map_err(|_| R2MapCollectorError::InvalidConfig("completed games exceed u64"))?;
        if records[0].identity.global_game_index != expected_first
            || self.manifest.completed_games + records.len() > self.manifest.requested_games
        {
            return Err(R2MapCollectorError::InvalidConfig(
                "shard does not continue the registered lease prefix",
            ));
        }
        let primary_per_game = primary_examples_per_game(self.manifest.collection_kind);
        let primary_example_count = records.len() * primary_per_game;
        for record in records {
            validate_record_against_manifest(record, &self.manifest)?;
            if record.extract_primary_examples()?.len() != primary_per_game {
                return Err(R2MapCollectorError::InvalidConfig(
                    "record produced the wrong primary-example count",
                ));
            }
        }
        let file = shard_file_name(self.manifest.shards.len());
        let path = self.root.join(&file);
        if path.exists() {
            return Err(R2MapCollectorError::UnexpectedArtifact(path));
        }
        write_r2_map_shard(&path, records)?;
        let shard = R2MapCollectorShardManifest {
            file,
            first_game_index: expected_first,
            game_count: records.len(),
            primary_example_count,
            byte_count: fs::metadata(&path)?.len(),
            blake3: checksum_file(&path)?,
        };
        self.manifest.completed_games += records.len();
        self.manifest.primary_example_count += primary_example_count;
        self.manifest.updated_unix_seconds = unix_seconds()?;
        self.manifest.shards.push(shard);
        write_manifest_atomic(&self.manifest_path, &self.manifest)?;
        Ok(())
    }
}

pub fn collect_r2_map_with_runner<R: R2MapGameRunner>(
    config: &R2MapCollectorConfig,
    runner: &R,
) -> Result<R2MapCollectorManifest, R2MapCollectorError> {
    let mut writer = R2MapCollectorWriter::open(config)?;
    while writer.manifest.completed_games < writer.manifest.requested_games {
        let remaining = writer.manifest.requested_games - writer.manifest.completed_games;
        let game_count = writer.manifest.shard_games.min(remaining);
        let first = writer.manifest.lease.first_game_index
            + u64::try_from(writer.manifest.completed_games)
                .map_err(|_| R2MapCollectorError::InvalidConfig("completed games exceed u64"))?;
        let requests = (0..game_count)
            .map(|offset| {
                let offset = u64::try_from(offset)
                    .map_err(|_| R2MapCollectorError::InvalidConfig("offset exceeds u64"))?;
                config.request(first + offset)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let played = runner.play_batch(&requests)?;
        if played.len() != requests.len() {
            return Err(R2MapCollectorError::RunnerContract(
                "runner returned the wrong number of games",
            ));
        }
        let records = requests
            .into_iter()
            .zip(played)
            .map(|(request, played)| {
                if played.result.seed != request.seed {
                    return Err(R2MapCollectorError::RunnerContract(
                        "runner returned a different game seed",
                    ));
                }
                if played.public_turns != reconstruct_r2_map_public_turns(&played.result.replay)? {
                    return Err(R2MapCollectorError::RunnerContract(
                        "runner public market trace differs from replay",
                    ));
                }
                Ok(R2MapGameRecord::from_match(
                    request.context,
                    &played.result,
                    &played.exploration_draws,
                )?)
            })
            .collect::<Result<Vec<_>, R2MapCollectorError>>()?;
        writer.append_shard(&records)?;
    }
    validate_r2_map_collector_dataset(&writer.root, &writer.manifest)?;
    Ok(writer.manifest.clone())
}

pub fn collect_r2_map_bootstrap(
    config: &R2MapCollectorConfig,
) -> Result<R2MapCollectorManifest, R2MapCollectorError> {
    if config.policy != R2MapCollectorPolicy::BootstrapGreedy {
        return Err(R2MapCollectorError::InvalidConfig(
            "bootstrap collector requires the greedy bootstrap policy",
        ));
    }
    collect_r2_map_with_runner(config, &GreedyBootstrapRunner)
}

pub fn validate_r2_map_collector_dataset(
    root: &Path,
    manifest: &R2MapCollectorManifest,
) -> Result<(), R2MapCollectorError> {
    validate_r2_map_collector_dataset_inner(root, manifest, false)
}

fn validate_r2_map_collector_dataset_inner(
    root: &Path,
    manifest: &R2MapCollectorManifest,
    allow_unfinished_next: bool,
) -> Result<(), R2MapCollectorError> {
    if manifest.schema_version != R2_MAP_COLLECTOR_SCHEMA_VERSION
        || manifest.trajectory_schema_version != R2_MAP_EXPERIENCE_SCHEMA_VERSION
        || manifest.game != GameConfig::research_aaaaa(4)?
        || manifest.shard_games == 0
        || manifest.requested_games
            != usize::try_from(manifest.lease.game_count)
                .map_err(|_| R2MapCollectorError::InvalidManifest("requested games exceed usize"))?
        || manifest.dataset_id != dataset_id(manifest)?
    {
        return Err(R2MapCollectorError::InvalidManifest(
            "collector manifest schema or identity does not match",
        ));
    }
    manifest.lease.validate()?;
    manifest.protocols.validate()?;
    manifest
        .exploration
        .validate(manifest.collection_kind, manifest.lease.iteration)?;
    if manifest.rng != R2MapRngIdentity::default() {
        return Err(R2MapCollectorError::InvalidManifest(
            "collector RNG identity drifted",
        ));
    }
    let expected_seed_purpose = match manifest.collection_kind {
        R2MapCollectionKind::Bootstrap => R2MapSeedPurpose::Bootstrap,
        R2MapCollectionKind::IterativeTraining => R2MapSeedPurpose::Generation,
        R2MapCollectionKind::Benchmark => {
            return Err(R2MapCollectorError::InvalidManifest(
                "benchmark records do not belong in training collectors",
            ));
        }
    };
    if manifest.lease.purpose != expected_seed_purpose
        || manifest.policy.collection_kind() != manifest.collection_kind
    {
        return Err(R2MapCollectorError::InvalidManifest(
            "policy, collection kind, and seed purpose disagree",
        ));
    }
    match &manifest.policy {
        R2MapCollectorPolicy::BootstrapGreedy => {}
        R2MapCollectorPolicy::Iterative {
            newest,
            opponent_pool,
        } => {
            newest.validate()?;
            if newest.role != R2MapPolicyRole::Newest {
                return Err(R2MapCollectorError::InvalidManifest(
                    "iterative focal policy is not newest",
                ));
            }
            for opponent in opponent_pool {
                opponent.validate()?;
                if !matches!(
                    opponent.role,
                    R2MapPolicyRole::Historical | R2MapPolicyRole::Greedy
                ) {
                    return Err(R2MapCollectorError::InvalidManifest(
                        "iterative opponent pool contains a newest policy",
                    ));
                }
            }
        }
    }
    let primary_per_game = primary_examples_per_game(manifest.collection_kind);
    let mut completed_games = 0usize;
    let mut primary_example_count = 0usize;
    for (index, shard) in manifest.shards.iter().enumerate() {
        let expected_first = manifest.lease.first_game_index
            + u64::try_from(completed_games)
                .map_err(|_| R2MapCollectorError::InvalidManifest("completed games exceed u64"))?;
        if shard.file != shard_file_name(index)
            || shard.first_game_index != expected_first
            || shard.game_count == 0
            || shard.game_count > manifest.shard_games
            || shard.primary_example_count != shard.game_count * primary_per_game
        {
            return Err(R2MapCollectorError::InvalidManifest(
                "shard sequence or counts are invalid",
            ));
        }
        let path = root.join(&shard.file);
        if fs::metadata(&path)?.len() != shard.byte_count || checksum_file(&path)? != shard.blake3 {
            return Err(R2MapCollectorError::ShardChecksum(path));
        }
        let records = read_r2_map_shard(&path)?;
        if records.len() != shard.game_count {
            return Err(R2MapCollectorError::InvalidManifest(
                "shard manifest game count disagrees with payload",
            ));
        }
        for record in &records {
            validate_record_against_manifest(record, manifest)?;
            if record.extract_primary_examples()?.len() != primary_per_game {
                return Err(R2MapCollectorError::InvalidManifest(
                    "record primary-example count is invalid",
                ));
            }
        }
        completed_games += shard.game_count;
        primary_example_count += shard.primary_example_count;
    }
    if completed_games != manifest.completed_games
        || primary_example_count != manifest.primary_example_count
        || completed_games > manifest.requested_games
    {
        return Err(R2MapCollectorError::InvalidManifest(
            "manifest aggregate counts disagree with shards",
        ));
    }
    validate_registered_directory_entries(root, manifest, allow_unfinished_next)?;
    Ok(())
}

fn validate_record_against_manifest(
    record: &R2MapGameRecord,
    manifest: &R2MapCollectorManifest,
) -> Result<(), R2MapCollectorError> {
    record.validate()?;
    if record.collection_kind != manifest.collection_kind
        || record.identity.campaign_id != manifest.lease.campaign_id
        || record.identity.iteration != manifest.lease.iteration
        || record.identity.host_id != manifest.lease.host_id
        || record.seed_purpose != manifest.lease.purpose
        || record.protocols != manifest.protocols
        || record.rng != manifest.rng
        || record.exploration != manifest.exploration
    {
        return Err(R2MapCollectorError::InvalidManifest(
            "record identity drifts from its collector manifest",
        ));
    }
    let offset = record
        .identity
        .global_game_index
        .checked_sub(manifest.lease.first_game_index)
        .ok_or(R2MapCollectorError::InvalidManifest(
            "record precedes registered lease",
        ))?;
    if offset >= manifest.lease.game_count {
        return Err(R2MapCollectorError::InvalidManifest(
            "record exceeds registered lease",
        ));
    }
    Ok(())
}

fn dataset_id(manifest: &R2MapCollectorManifest) -> Result<String, R2MapCollectorError> {
    #[derive(Serialize)]
    struct Identity<'a> {
        schema_version: u16,
        trajectory_schema_version: u16,
        game: GameConfig,
        collection_kind: R2MapCollectionKind,
        lease: &'a R2MapSeedLease,
        policy: &'a R2MapCollectorPolicy,
        exploration: &'a R2MapExplorationIdentity,
        rng: &'a R2MapRngIdentity,
        protocols: &'a R2MapProtocolIdentity,
        shard_games: usize,
        requested_games: usize,
    }
    let bytes = postcard::to_allocvec(&Identity {
        schema_version: manifest.schema_version,
        trajectory_schema_version: manifest.trajectory_schema_version,
        game: manifest.game,
        collection_kind: manifest.collection_kind,
        lease: &manifest.lease,
        policy: &manifest.policy,
        exploration: &manifest.exploration,
        rng: &manifest.rng,
        protocols: &manifest.protocols,
        shard_games: manifest.shard_games,
        requested_games: manifest.requested_games,
    })?;
    let digest = blake3::hash(&bytes).to_hex();
    Ok(format!(
        "r2-map-{}-{}-{}-{}",
        match manifest.collection_kind {
            R2MapCollectionKind::Bootstrap => "bootstrap",
            R2MapCollectionKind::IterativeTraining => "iteration",
            R2MapCollectionKind::Benchmark => "benchmark",
        },
        manifest.lease.host_id,
        manifest.lease.first_game_index,
        &digest[..16]
    ))
}

fn ensure_new_dataset_directory_is_empty(root: &Path) -> Result<(), R2MapCollectorError> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_name() != COLLECTOR_LOCK_FILE {
            return Err(R2MapCollectorError::UnexpectedArtifact(entry.path()));
        }
    }
    Ok(())
}

fn validate_registered_directory_entries(
    root: &Path,
    manifest: &R2MapCollectorManifest,
    allow_unfinished_next: bool,
) -> Result<(), R2MapCollectorError> {
    let next_shard = shard_file_name(manifest.shards.len());
    let next_temp = Path::new(&next_shard).with_extension("r2sh.tmp");
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let name = entry.file_name();
        let registered = name == R2_MAP_COLLECTOR_MANIFEST_FILE
            || name == COLLECTOR_LOCK_FILE
            || manifest
                .shards
                .iter()
                .any(|shard| name == shard.file.as_str());
        let unfinished_next = name == next_shard.as_str() || name == next_temp.as_os_str();
        if !(registered || allow_unfinished_next && unfinished_next) {
            return Err(R2MapCollectorError::UnexpectedArtifact(entry.path()));
        }
    }
    Ok(())
}

fn remove_unfinished_next_shard(
    root: &Path,
    manifest: &R2MapCollectorManifest,
) -> Result<(), R2MapCollectorError> {
    let next = root.join(shard_file_name(manifest.shards.len()));
    let temp = next.with_extension("r2sh.tmp");
    for path in [next, temp] {
        if path.exists() {
            fs::remove_file(path)?;
        }
    }
    sync_directory(root)?;
    Ok(())
}

fn primary_examples_per_game(kind: R2MapCollectionKind) -> usize {
    match kind {
        R2MapCollectionKind::Bootstrap => 80,
        R2MapCollectionKind::IterativeTraining => 20,
        R2MapCollectionKind::Benchmark => 0,
    }
}

fn shard_file_name(index: usize) -> String {
    format!("shard-{index:05}.r2sh")
}

fn write_manifest_atomic(
    path: &Path,
    manifest: &R2MapCollectorManifest,
) -> Result<(), R2MapCollectorError> {
    let temp = path.with_extension("json.tmp");
    let mut writer = BufWriter::new(File::create(&temp)?);
    serde_json::to_writer_pretty(&mut writer, manifest)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(&temp, path)?;
    if let Some(parent) = path.parent() {
        sync_directory(parent)?;
    }
    Ok(())
}

fn checksum_file(path: &Path) -> Result<String, R2MapCollectorError> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Hasher::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn sync_directory(path: &Path) -> Result<(), R2MapCollectorError> {
    File::open(path)?.sync_all()?;
    Ok(())
}

fn unix_seconds() -> Result<u64, R2MapCollectorError> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| R2MapCollectorError::ClockBeforeEpoch)?
        .as_secs())
}

#[derive(Debug, Error)]
pub enum R2MapCollectorError {
    #[error("invalid R2-MAP collector configuration: {0}")]
    InvalidConfig(&'static str),
    #[error("invalid R2-MAP collector manifest: {0}")]
    InvalidManifest(&'static str),
    #[error("R2-MAP collector dataset already exists at {0}; pass --resume")]
    DatasetExists(PathBuf),
    #[error("R2-MAP collector resume contract drifted")]
    ResumeDrift,
    #[error("R2-MAP collector is already locked at {0}")]
    CollectionLocked(PathBuf),
    #[error("unexpected artifact in R2-MAP collector directory: {0}")]
    UnexpectedArtifact(PathBuf),
    #[error("R2-MAP shard checksum failed: {0}")]
    ShardChecksum(PathBuf),
    #[error("R2-MAP runner contract failed: {0}")]
    RunnerContract(&'static str),
    #[error("system clock is before the Unix epoch")]
    ClockBeforeEpoch,
    #[error(transparent)]
    Experience(#[from] R2MapExperienceError),
    #[error(transparent)]
    Simulation(#[from] cascadia_sim::SimulationError),
    #[error(transparent)]
    Rules(#[from] cascadia_game::RuleError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Postcard(#[from] postcard::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    use cascadia_sim::{play_match_with_seat_selector, select_greedy_action, strategy_rng};

    use crate::expected_r2_map_exploration_draws;

    use super::*;

    fn temp_root(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "cascadia-r2-map-collector-{name}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn protocols() -> R2MapProtocolIdentity {
        R2MapProtocolIdentity {
            collector_hash: [1; 32],
            source_hash: [2; 32],
            serving_protocol_hash: [3; 32],
        }
    }

    fn config(root: PathBuf, games: u64, shard_games: usize, resume: bool) -> R2MapCollectorConfig {
        R2MapCollectorConfig::bootstrap(
            root,
            R2MapSeedLease {
                campaign_id: "r2-map-expert-iteration-v1".to_owned(),
                iteration: 0,
                purpose: R2MapSeedPurpose::Bootstrap,
                host_id: "john2".to_owned(),
                first_game_index: 40,
                game_count: games,
            },
            shard_games,
            resume,
            protocols(),
        )
    }

    #[test]
    fn bootstrap_collection_writes_exact_shards_and_eighty_examples_per_game() {
        let root = temp_root("bootstrap");
        let manifest = collect_r2_map_bootstrap(&config(root.clone(), 5, 2, false)).unwrap();
        assert_eq!(manifest.completed_games, 5);
        assert_eq!(manifest.primary_example_count, 400);
        assert_eq!(manifest.shards.len(), 3);
        assert_eq!(manifest.shards[0].game_count, 2);
        assert_eq!(manifest.shards[2].game_count, 1);
        validate_r2_map_collector_dataset(&root, &manifest).unwrap();
        let first = read_r2_map_shard(&root.join(&manifest.shards[0].file)).unwrap();
        assert_eq!(first[0].identity.global_game_index, 40);
        assert_eq!(first[1].identity.global_game_index, 41);
        assert!(first.iter().all(|record| record.decisions.len() == 80));
        fs::remove_dir_all(root).unwrap();
    }

    #[derive(Default)]
    struct CountingRunner {
        calls: AtomicUsize,
    }

    impl R2MapGameRunner for CountingRunner {
        fn play_game(
            &self,
            request: &R2MapGameRequest,
        ) -> Result<R2MapPlayedGame, R2MapCollectorError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            GreedyBootstrapRunner.play_game(request)
        }
    }

    #[test]
    fn resume_skips_only_registered_valid_shards() {
        let root = temp_root("resume");
        let initial = config(root.clone(), 4, 2, false);
        let mut writer = R2MapCollectorWriter::open(&initial).unwrap();
        let requests = (40..42)
            .map(|index| initial.request(index).unwrap())
            .collect::<Vec<_>>();
        let played = GreedyBootstrapRunner.play_batch(&requests).unwrap();
        let records = requests
            .into_iter()
            .zip(played)
            .map(|(request, played)| {
                R2MapGameRecord::from_match(
                    request.context,
                    &played.result,
                    &played.exploration_draws,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        writer.append_shard(&records).unwrap();
        let partial_manifest = writer.manifest().clone();
        drop(writer);

        let orphan = root.join("shard-00001.r2sh");
        fs::write(&orphan, b"unfinished unregistered bytes").unwrap();
        assert!(matches!(
            validate_r2_map_collector_dataset(&root, &partial_manifest),
            Err(R2MapCollectorError::UnexpectedArtifact(_))
        ));
        let runner = CountingRunner::default();
        let manifest =
            collect_r2_map_with_runner(&config(root.clone(), 4, 2, true), &runner).unwrap();
        assert_eq!(runner.calls.load(Ordering::SeqCst), 2);
        assert_eq!(manifest.completed_games, 4);
        assert_eq!(read_r2_map_shard(&orphan).unwrap().len(), 2);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resume_rejects_protocol_range_and_shard_drift() {
        let root = temp_root("drift");
        let original = config(root.clone(), 2, 1, false);
        let manifest = collect_r2_map_bootstrap(&original).unwrap();

        let mut drifted = config(root.clone(), 2, 1, true);
        drifted.protocols.source_hash = [9; 32];
        assert!(matches!(
            R2MapCollectorWriter::open(&drifted),
            Err(R2MapCollectorError::ResumeDrift)
        ));

        let shard = root.join(&manifest.shards[0].file);
        let mut bytes = fs::read(&shard).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        fs::write(&shard, bytes).unwrap();
        assert!(matches!(
            R2MapCollectorWriter::open(&config(root.clone(), 2, 1, true)),
            Err(R2MapCollectorError::ShardChecksum(_))
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn lock_and_unexpected_artifacts_fail_closed() {
        let root = temp_root("lock");
        let collector = R2MapCollectorWriter::open(&config(root.clone(), 1, 1, false)).unwrap();
        assert!(matches!(
            R2MapCollectorWriter::open(&config(root.clone(), 1, 1, true)),
            Err(R2MapCollectorError::CollectionLocked(_))
        ));
        drop(collector);
        fs::write(root.join("mystery.bin"), b"drift").unwrap();
        assert!(matches!(
            R2MapCollectorWriter::open(&config(root.clone(), 1, 1, true)),
            Err(R2MapCollectorError::UnexpectedArtifact(_))
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[derive(Default)]
    struct IdentityAwareGreedyRunner;

    impl R2MapGameRunner for IdentityAwareGreedyRunner {
        fn play_game(
            &self,
            request: &R2MapGameRequest,
        ) -> Result<R2MapPlayedGame, R2MapCollectorError> {
            let strategy_ids = request
                .context
                .seats
                .iter()
                .map(|policy| policy.policy_id.clone())
                .collect::<Vec<_>>();
            let mut rngs = strategy_ids
                .iter()
                .enumerate()
                .map(|(seat, strategy_id)| strategy_rng(request.seed, seat, strategy_id))
                .collect::<Vec<_>>();
            let result = play_match_with_seat_selector(
                GameConfig::research_aaaaa(4)?,
                request.seed,
                &strategy_ids,
                |seat, game| {
                    let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
                    select_greedy_action(game, &prelude, &mut rngs[seat])
                },
            )?;
            let exploration_draws =
                expected_r2_map_exploration_draws(&request.context, &result.replay)?;
            let public_turns = reconstruct_r2_map_public_turns(&result.replay)?;
            Ok(R2MapPlayedGame {
                result,
                exploration_draws,
                public_turns,
            })
        }
    }

    #[test]
    fn iterative_runner_boundary_retains_one_newest_seat_and_twenty_examples() {
        let root = temp_root("iterative");
        let config = R2MapCollectorConfig::iterative(
            root.clone(),
            R2MapSeedLease {
                campaign_id: "r2-map-expert-iteration-v1".to_owned(),
                iteration: 0,
                purpose: R2MapSeedPurpose::Generation,
                host_id: "john2".to_owned(),
                first_game_index: 0,
                game_count: 2,
            },
            2,
            false,
            protocols(),
            R2MapPolicyIdentity::newest("greedy-v1", [10; 32]),
            vec![
                R2MapPolicyIdentity::greedy(),
                R2MapPolicyIdentity::historical("greedy-v1", [11; 32]),
            ],
            1_000_000,
        );
        assert_eq!(config.exploration.epsilon_parts_per_million, 100_000);
        let manifest = collect_r2_map_with_runner(&config, &IdentityAwareGreedyRunner).unwrap();
        assert_eq!(manifest.completed_games, 2);
        assert_eq!(manifest.primary_example_count, 40);
        let records = read_r2_map_shard(&root.join(&manifest.shards[0].file)).unwrap();
        for record in records {
            assert_eq!(
                record
                    .seats
                    .iter()
                    .filter(|seat| seat.role == R2MapPolicyRole::Newest)
                    .count(),
                1
            );
            assert_eq!(record.extract_primary_examples().unwrap().len(), 20);
        }
        fs::remove_dir_all(root).unwrap();
    }
}
