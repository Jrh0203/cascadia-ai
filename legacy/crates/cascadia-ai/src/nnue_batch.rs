//! Evaluator-independent batching for the qualified historical NNUE search.

use std::{
    collections::{HashMap, HashSet},
    convert::Infallible,
    error::Error,
    fmt::{self, Display},
    hash::{BuildHasherDefault, Hasher},
    ops::Range,
    sync::mpsc::{self, Receiver, SyncSender},
    thread,
    time::Instant,
};

use arrayvec::ArrayVec;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;

use crate::{
    eval::ScoredMove,
    mce::{
        deterministic_market_representatives, mce_score_total, scored_move_identity,
        MceMoveEstimate,
    },
    nnue::{extract_features_with_bag, BagInfo, NNUENetwork},
    nnue_train::{
        prepare_nnue_move_direct_mut, prepare_nnue_move_from_template_mut,
        prepare_nnue_move_template, select_prepared_nnue_candidate_index, PreparedNnueMove,
    },
    search::{candidate_board_cache_key, candidate_cache_key, execute_scored_move, greedy_move},
};
use cascadia_core::{game::GameState, hex::HexCoord, scoring::ScoreBreakdown};

pub trait SparseNnueEvaluator: Send {
    type Error: Send;

    fn evaluate_sparse(&mut self, feature_sets: &[Vec<u16>]) -> Result<Vec<f32>, Self::Error>;

    fn evaluate_sparse_owned(
        &mut self,
        feature_sets: Vec<Vec<u16>>,
    ) -> Result<Vec<f32>, Self::Error> {
        self.evaluate_sparse(&feature_sets)
    }

    /// Opt into exact rollout pipelining and choose the number of independent
    /// rollout states prepared per inference cohort. The logical rollout wave,
    /// row order, global deduplication, and diagnostics remain unchanged.
    fn rollout_pipeline_chunk_states(&self) -> Option<usize> {
        None
    }
}

impl SparseNnueEvaluator for NNUENetwork {
    type Error = Infallible;

    fn evaluate_sparse(&mut self, feature_sets: &[Vec<u16>]) -> Result<Vec<f32>, Self::Error> {
        Ok(feature_sets
            .iter()
            .map(|features| self.forward(features))
            .collect())
    }
}

#[derive(Debug)]
pub enum BatchedNnueError<E> {
    Evaluator(E),
    InvalidPredictionWidth { expected: usize, actual: usize },
    NonFinitePrediction { index: usize },
    InvalidPipelineChunkSize(usize),
    EvaluatorWorkerDisconnected,
    EvaluatorWorkerPanicked,
    UnsupportedConfiguration(&'static str),
}

impl<E: Display> Display for BatchedNnueError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Evaluator(error) => write!(formatter, "sparse NNUE evaluator failed: {error}"),
            Self::InvalidPredictionWidth { expected, actual } => write!(
                formatter,
                "sparse NNUE returned {actual} values for {expected} rows"
            ),
            Self::NonFinitePrediction { index } => {
                write!(formatter, "sparse NNUE prediction {index} was not finite")
            }
            Self::InvalidPipelineChunkSize(size) => {
                write!(
                    formatter,
                    "sparse NNUE pipeline chunk size {size} is invalid"
                )
            }
            Self::EvaluatorWorkerDisconnected => {
                write!(formatter, "sparse NNUE evaluator worker disconnected")
            }
            Self::EvaluatorWorkerPanicked => {
                write!(formatter, "sparse NNUE evaluator worker panicked")
            }
            Self::UnsupportedConfiguration(name) => {
                write!(
                    formatter,
                    "batched NNUE search does not support active option {name}"
                )
            }
        }
    }
}

impl<E: Error + 'static> Error for BatchedNnueError<E> {}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BatchedNnueDiagnostics {
    pub neural_batches: u64,
    pub neural_rows: u64,
    pub physical_neural_rows: u64,
    pub reuse_observed_physical_rows: u64,
    pub reuse_repeated_physical_rows: u64,
    pub minimum_batch_rows: usize,
    pub maximum_batch_rows: usize,
    pub rollout_waves: u64,
    pub rollout_samples: u64,
    pub bootstrapped_samples: u64,
    pub policy_fallbacks: u64,
    pub template_state_requests: u64,
    pub unique_public_template_states: u64,
    pub unique_board_template_states: u64,
    pub stage_timings: BatchedNnueStageTimings,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BatchedNnueStageTimings {
    pub rollout_state_initialization_ns: u64,
    pub opponent_advance_ns: u64,
    pub candidate_keying_ns: u64,
    pub template_preparation_ns: u64,
    pub candidate_preparation_ns: u64,
    pub row_assembly_ns: u64,
    pub row_deduplication_ns: u64,
    pub row_materialization_ns: u64,
    pub neural_evaluation_ns: u64,
    pub prediction_postprocess_ns: u64,
    pub action_selection_ns: u64,
    pub terminal_collection_ns: u64,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BatchedRolloutConfig {
    /// Maximum number of focal-player moves represented in each rollout,
    /// including the root candidate. `None` preserves full terminal rollouts.
    pub max_focal_turns: Option<u8>,
    pub leaf_timing: BatchedRolloutLeafTiming,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum BatchedRolloutLeafTiming {
    /// Let the other players complete the round before evaluating the leaf.
    /// This preserves the opponent-aware state transition used by the scalar
    /// truncated-rollout implementation.
    #[default]
    AfterOpponentRound,
    /// Evaluate the focal player's afterstate immediately. This is faster, but
    /// changes the information available to opponent-detail features.
    AfterFocalMove,
}

impl BatchedRolloutConfig {
    pub const fn full() -> Self {
        Self {
            max_focal_turns: None,
            leaf_timing: BatchedRolloutLeafTiming::AfterOpponentRound,
        }
    }

    pub fn truncated(max_focal_turns: usize) -> Result<Self, &'static str> {
        Self::truncated_with_timing(
            max_focal_turns,
            BatchedRolloutLeafTiming::AfterOpponentRound,
        )
    }

    pub fn truncated_afterstate(max_focal_turns: usize) -> Result<Self, &'static str> {
        Self::truncated_with_timing(max_focal_turns, BatchedRolloutLeafTiming::AfterFocalMove)
    }

    fn truncated_with_timing(
        max_focal_turns: usize,
        leaf_timing: BatchedRolloutLeafTiming,
    ) -> Result<Self, &'static str> {
        let max_focal_turns =
            u8::try_from(max_focal_turns).map_err(|_| "rollout turn limit exceeds u8")?;
        if max_focal_turns == 0 {
            return Err("rollout turn limit must be positive");
        }
        Ok(Self {
            max_focal_turns: Some(max_focal_turns),
            leaf_timing,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RolloutSeedCoupling {
    Independent,
    CommonWithinRound,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RolloutValueSample {
    pub rollout_seed: u64,
    pub personal_turn: u8,
    pub immediate_score: f32,
    pub target_remaining: f32,
    pub features: Vec<u16>,
}

#[derive(Debug, Clone)]
pub struct BatchedMceResult {
    pub estimates: Vec<MceMoveEstimate>,
    pub rollout_value_samples: Vec<RolloutValueSample>,
}

impl BatchedNnueDiagnostics {
    pub fn record_batch(&mut self, rows: usize) {
        self.record_batch_work(rows, rows);
    }

    pub fn record_batch_work(&mut self, logical_rows: usize, physical_rows: usize) {
        self.neural_batches += 1;
        self.neural_rows += logical_rows as u64;
        self.physical_neural_rows += physical_rows as u64;
        if self.minimum_batch_rows == 0 {
            self.minimum_batch_rows = logical_rows;
        } else {
            self.minimum_batch_rows = self.minimum_batch_rows.min(logical_rows);
        }
        self.maximum_batch_rows = self.maximum_batch_rows.max(logical_rows);
    }

    pub fn merge_from(&mut self, source: Self) {
        self.neural_batches = self.neural_batches.saturating_add(source.neural_batches);
        self.neural_rows = self.neural_rows.saturating_add(source.neural_rows);
        self.physical_neural_rows = self
            .physical_neural_rows
            .saturating_add(source.physical_neural_rows);
        self.reuse_observed_physical_rows = self
            .reuse_observed_physical_rows
            .saturating_add(source.reuse_observed_physical_rows);
        self.reuse_repeated_physical_rows = self
            .reuse_repeated_physical_rows
            .saturating_add(source.reuse_repeated_physical_rows);
        if source.minimum_batch_rows != 0 {
            self.minimum_batch_rows = if self.minimum_batch_rows == 0 {
                source.minimum_batch_rows
            } else {
                self.minimum_batch_rows.min(source.minimum_batch_rows)
            };
        }
        self.maximum_batch_rows = self.maximum_batch_rows.max(source.maximum_batch_rows);
        self.rollout_waves = self.rollout_waves.saturating_add(source.rollout_waves);
        self.rollout_samples = self.rollout_samples.saturating_add(source.rollout_samples);
        self.bootstrapped_samples = self
            .bootstrapped_samples
            .saturating_add(source.bootstrapped_samples);
        self.policy_fallbacks = self
            .policy_fallbacks
            .saturating_add(source.policy_fallbacks);
        self.template_state_requests = self
            .template_state_requests
            .saturating_add(source.template_state_requests);
        self.unique_public_template_states = self
            .unique_public_template_states
            .saturating_add(source.unique_public_template_states);
        self.unique_board_template_states = self
            .unique_board_template_states
            .saturating_add(source.unique_board_template_states);
        self.stage_timings.merge_from(source.stage_timings);
    }
}

impl BatchedNnueStageTimings {
    pub fn total_ns(self) -> u64 {
        self.rollout_state_initialization_ns
            .saturating_add(self.opponent_advance_ns)
            .saturating_add(self.candidate_keying_ns)
            .saturating_add(self.template_preparation_ns)
            .saturating_add(self.candidate_preparation_ns)
            .saturating_add(self.row_assembly_ns)
            .saturating_add(self.row_deduplication_ns)
            .saturating_add(self.row_materialization_ns)
            .saturating_add(self.neural_evaluation_ns)
            .saturating_add(self.prediction_postprocess_ns)
            .saturating_add(self.action_selection_ns)
            .saturating_add(self.terminal_collection_ns)
    }

    fn merge_from(&mut self, source: Self) {
        self.rollout_state_initialization_ns = self
            .rollout_state_initialization_ns
            .saturating_add(source.rollout_state_initialization_ns);
        self.opponent_advance_ns = self
            .opponent_advance_ns
            .saturating_add(source.opponent_advance_ns);
        self.candidate_keying_ns = self
            .candidate_keying_ns
            .saturating_add(source.candidate_keying_ns);
        self.template_preparation_ns = self
            .template_preparation_ns
            .saturating_add(source.template_preparation_ns);
        self.candidate_preparation_ns = self
            .candidate_preparation_ns
            .saturating_add(source.candidate_preparation_ns);
        self.row_assembly_ns = self.row_assembly_ns.saturating_add(source.row_assembly_ns);
        self.row_deduplication_ns = self
            .row_deduplication_ns
            .saturating_add(source.row_deduplication_ns);
        self.row_materialization_ns = self
            .row_materialization_ns
            .saturating_add(source.row_materialization_ns);
        self.neural_evaluation_ns = self
            .neural_evaluation_ns
            .saturating_add(source.neural_evaluation_ns);
        self.prediction_postprocess_ns = self
            .prediction_postprocess_ns
            .saturating_add(source.prediction_postprocess_ns);
        self.action_selection_ns = self
            .action_selection_ns
            .saturating_add(source.action_selection_ns);
        self.terminal_collection_ns = self
            .terminal_collection_ns
            .saturating_add(source.terminal_collection_ns);
    }
}

#[inline]
fn stage_timings_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CASCADIA_NNUE_STAGE_TIMINGS")
            .ok()
            .map(|value| !value.is_empty() && value != "0")
            .unwrap_or(false)
    })
}

#[inline]
fn stage_timer(enabled: bool) -> Option<Instant> {
    enabled.then(Instant::now)
}

#[inline]
fn elapsed_ns(started: Option<Instant>) -> u64 {
    started
        .map(|started| started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

fn evaluate_checked<E: SparseNnueEvaluator>(
    evaluator: &mut E,
    feature_sets: &[Vec<u16>],
    diagnostics: &mut BatchedNnueDiagnostics,
) -> Result<Vec<f32>, BatchedNnueError<E::Error>> {
    evaluate_checked_with_dedup(
        evaluator,
        feature_sets,
        diagnostics,
        sparse_row_dedup_enabled(),
    )
}

pub fn evaluate_sparse_rows_deduplicated<E: SparseNnueEvaluator>(
    evaluator: &mut E,
    feature_sets: &[Vec<u16>],
    diagnostics: &mut BatchedNnueDiagnostics,
) -> Result<Vec<f32>, BatchedNnueError<E::Error>> {
    evaluate_checked_with_dedup(evaluator, feature_sets, diagnostics, true)
}

#[derive(Default)]
struct IdentityU64Hasher(u64);

impl Hasher for IdentityU64Hasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for &byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        self.0 = hash;
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }
}

type IdentityU64Map<V> = HashMap<u64, V, BuildHasherDefault<IdentityU64Hasher>>;

struct SparseRowReuseTracker {
    first_by_fingerprint: IdentityU64Map<Vec<u16>>,
    collisions: IdentityU64Map<Vec<Vec<u16>>>,
}

impl SparseRowReuseTracker {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            first_by_fingerprint: IdentityU64Map::with_capacity_and_hasher(
                capacity,
                Default::default(),
            ),
            collisions: IdentityU64Map::default(),
        }
    }

    fn observe(&mut self, features: &[u16], fingerprint: u64) -> bool {
        match self.first_by_fingerprint.get(&fingerprint) {
            None => {
                self.first_by_fingerprint
                    .insert(fingerprint, features.to_vec());
                false
            }
            Some(first) if first.as_slice() == features => true,
            Some(_) => {
                if self
                    .collisions
                    .get(&fingerprint)
                    .is_some_and(|rows| rows.iter().any(|row| row.as_slice() == features))
                {
                    true
                } else {
                    self.collisions
                        .entry(fingerprint)
                        .or_default()
                        .push(features.to_vec());
                    false
                }
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct SparseRowDedup {
    unique_indices: Vec<usize>,
    row_to_unique: Vec<usize>,
}

struct SparseRowDedupBuilder {
    first_by_fingerprint: IdentityU64Map<usize>,
    collisions: IdentityU64Map<Vec<usize>>,
    unique_indices: Vec<usize>,
    row_to_unique: Vec<usize>,
}

impl SparseRowDedupBuilder {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            first_by_fingerprint: IdentityU64Map::with_capacity_and_hasher(
                capacity,
                Default::default(),
            ),
            collisions: IdentityU64Map::default(),
            unique_indices: Vec::with_capacity(capacity),
            row_to_unique: Vec::with_capacity(capacity),
        }
    }

    fn push_with_fingerprint(
        &mut self,
        feature_sets: &[Vec<u16>],
        row_index: usize,
        fingerprint: u64,
    ) -> (usize, bool) {
        let features = &feature_sets[row_index];
        let (unique_index, is_new) = match self.first_by_fingerprint.get(&fingerprint).copied() {
            None => {
                let unique_index = self.unique_indices.len();
                self.unique_indices.push(row_index);
                self.first_by_fingerprint.insert(fingerprint, unique_index);
                (unique_index, true)
            }
            Some(first_unique_index)
                if feature_sets[self.unique_indices[first_unique_index]].as_slice()
                    == features.as_slice() =>
            {
                (first_unique_index, false)
            }
            Some(first_unique_index) => {
                let matching_index = self.collisions.get(&fingerprint).and_then(|indices| {
                    indices.iter().copied().find(|&unique_index| {
                        feature_sets[self.unique_indices[unique_index]].as_slice()
                            == features.as_slice()
                    })
                });
                if let Some(matching_index) = matching_index {
                    (matching_index, false)
                } else {
                    let unique_index = self.unique_indices.len();
                    self.unique_indices.push(row_index);
                    self.collisions
                        .entry(fingerprint)
                        .or_insert_with(|| vec![first_unique_index])
                        .push(unique_index);
                    (unique_index, true)
                }
            }
        };
        self.row_to_unique.push(unique_index);
        (unique_index, is_new)
    }

    fn finish(self) -> SparseRowDedup {
        SparseRowDedup {
            unique_indices: self.unique_indices,
            row_to_unique: self.row_to_unique,
        }
    }
}

#[inline]
fn sparse_row_fingerprint(features: &[u16]) -> u64 {
    const SEED: u64 = 0x9e37_79b1_85eb_ca87;
    const PRIME: u64 = 0xc2b2_ae3d_27d4_eb4f;

    let mut hash = SEED;
    let mut chunks = features.chunks_exact(4);
    for chunk in &mut chunks {
        let packed = u64::from(chunk[0])
            | (u64::from(chunk[1]) << 16)
            | (u64::from(chunk[2]) << 32)
            | (u64::from(chunk[3]) << 48);
        hash ^= packed.wrapping_mul(PRIME);
        hash = hash.rotate_left(29).wrapping_mul(SEED);
    }

    let mut tail = 0_u64;
    for (index, &feature) in chunks.remainder().iter().enumerate() {
        tail |= u64::from(feature) << (index * 16);
    }
    hash ^= tail.wrapping_mul(PRIME);
    hash ^= (features.len() as u64).wrapping_mul(PRIME);
    finalize_sparse_row_fingerprint(hash)
}

#[inline]
fn finalize_sparse_row_fingerprint(mut hash: u64) -> u64 {
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    hash ^ (hash >> 33)
}

#[cfg(test)]
fn deduplicate_sparse_rows_with<F>(feature_sets: &[Vec<u16>], fingerprint: F) -> SparseRowDedup
where
    F: Fn(&[u16]) -> u64,
{
    let mut builder = SparseRowDedupBuilder::with_capacity(feature_sets.len());

    for (row_index, features) in feature_sets.iter().enumerate() {
        builder.push_with_fingerprint(feature_sets, row_index, fingerprint(features));
    }
    builder.finish()
}

fn deduplicate_sparse_rows(feature_sets: &[Vec<u16>]) -> SparseRowDedup {
    let mut builder = SparseRowDedupBuilder::with_capacity(feature_sets.len());
    for (row_index, features) in feature_sets.iter().enumerate() {
        builder.push_with_fingerprint(feature_sets, row_index, sparse_row_fingerprint(features));
    }
    builder.finish()
}

fn evaluate_checked_with_dedup<E: SparseNnueEvaluator>(
    evaluator: &mut E,
    feature_sets: &[Vec<u16>],
    diagnostics: &mut BatchedNnueDiagnostics,
    deduplicate: bool,
) -> Result<Vec<f32>, BatchedNnueError<E::Error>> {
    if feature_sets.is_empty() {
        return Ok(Vec::new());
    }

    let logical_rows = feature_sets.len();
    let timings_enabled = stage_timings_enabled();
    let dedup_started = stage_timer(timings_enabled);
    let dedup = if deduplicate && logical_rows > 1 {
        deduplicate_sparse_rows(feature_sets)
    } else {
        SparseRowDedup {
            unique_indices: Vec::new(),
            row_to_unique: Vec::new(),
        }
    };
    diagnostics.stage_timings.row_deduplication_ns = diagnostics
        .stage_timings
        .row_deduplication_ns
        .saturating_add(elapsed_ns(dedup_started));

    let physical_rows = if dedup.row_to_unique.is_empty() {
        logical_rows
    } else {
        dedup.unique_indices.len()
    };
    diagnostics.record_batch_work(logical_rows, physical_rows);

    let materialize_unique_rows = physical_rows != logical_rows;
    let (values, expected_rows) = if !materialize_unique_rows {
        let evaluation_started = stage_timer(timings_enabled);
        let values = evaluator.evaluate_sparse(feature_sets);
        diagnostics.stage_timings.neural_evaluation_ns = diagnostics
            .stage_timings
            .neural_evaluation_ns
            .saturating_add(elapsed_ns(evaluation_started));
        (values.map_err(BatchedNnueError::Evaluator)?, logical_rows)
    } else {
        let materialization_started = stage_timer(timings_enabled);
        let unique_feature_sets = dedup
            .unique_indices
            .into_iter()
            .map(|index| feature_sets[index].clone())
            .collect::<Vec<_>>();
        diagnostics.stage_timings.row_materialization_ns = diagnostics
            .stage_timings
            .row_materialization_ns
            .saturating_add(elapsed_ns(materialization_started));
        let evaluation_started = stage_timer(timings_enabled);
        let values = evaluator.evaluate_sparse(&unique_feature_sets);
        diagnostics.stage_timings.neural_evaluation_ns = diagnostics
            .stage_timings
            .neural_evaluation_ns
            .saturating_add(elapsed_ns(evaluation_started));
        (values.map_err(BatchedNnueError::Evaluator)?, physical_rows)
    };
    let postprocess_started = stage_timer(timings_enabled);
    if values.len() != expected_rows {
        return Err(BatchedNnueError::InvalidPredictionWidth {
            expected: expected_rows,
            actual: values.len(),
        });
    }
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(BatchedNnueError::NonFinitePrediction { index });
    }
    let values = if !materialize_unique_rows {
        values
    } else {
        dedup
            .row_to_unique
            .into_iter()
            .map(|unique_index| values[unique_index])
            .collect()
    };
    diagnostics.stage_timings.prediction_postprocess_ns = diagnostics
        .stage_timings
        .prediction_postprocess_ns
        .saturating_add(elapsed_ns(postprocess_started));
    Ok(values)
}

fn sparse_row_dedup_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CASCADIA_NNUE_ROW_DEDUP")
            .ok()
            .map(|value| !value.is_empty() && value != "0")
            .unwrap_or(true)
    })
}

fn sparse_row_reuse_diagnostics_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| option_enabled("CASCADIA_NNUE_ROW_REUSE_DIAGNOSTICS"))
}

fn template_reuse_diagnostics_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| option_enabled("CASCADIA_NNUE_TEMPLATE_REUSE_DIAGNOSTICS"))
}

fn option_enabled(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|value| !value.is_empty() && value != "0")
}

fn validate_qualified_environment<E>() -> Result<(), BatchedNnueError<E>> {
    if std::env::var("MCE_OPP_TEMPERATURE")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .is_some_and(|value| value > 0.0)
    {
        return Err(BatchedNnueError::UnsupportedConfiguration(
            "MCE_OPP_TEMPERATURE",
        ));
    }
    for name in [
        "MCE_CONTROL_VARIATES",
        "CASCADIA_MCE_DECOUPLE_OPP",
        "MCE_GUMBEL_HALVING",
        "MCE_STRATEGY_BIAS",
    ] {
        if option_enabled(name) {
            return Err(BatchedNnueError::UnsupportedConfiguration(name));
        }
    }
    if std::env::var("CASCADIA_MCE_TRUNC")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .is_some_and(|value| value > 0)
    {
        return Err(BatchedNnueError::UnsupportedConfiguration(
            "CASCADIA_MCE_TRUNC",
        ));
    }
    if std::env::var("MCE_ROLLOUT_OPP")
        .ok()
        .is_some_and(|value| value.eq_ignore_ascii_case("nnue") || value == "1")
    {
        return Err(BatchedNnueError::UnsupportedConfiguration(
            "MCE_ROLLOUT_OPP",
        ));
    }
    if std::env::var("MCE_ROLLOUT_POLICY")
        .ok()
        .is_some_and(|value| !value.is_empty() && !value.eq_ignore_ascii_case("nnue"))
    {
        return Err(BatchedNnueError::UnsupportedConfiguration(
            "MCE_ROLLOUT_POLICY",
        ));
    }
    if std::env::var("MCE_PREFILTER_ENSEMBLE")
        .ok()
        .is_some_and(|value| !value.trim().is_empty())
    {
        return Err(BatchedNnueError::UnsupportedConfiguration(
            "MCE_PREFILTER_ENSEMBLE",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct SparseNnueAfterstate {
    pub movement: ScoredMove,
    pub immediate_score: f32,
    pub features: Vec<u16>,
}

pub fn prepare_sparse_nnue_afterstates(
    game: &GameState,
    candidates: &[ScoredMove],
) -> Vec<SparseNnueAfterstate> {
    let player = game.current_player;
    let cards = game.scoring_cards;
    candidates
        .par_iter()
        .filter_map(|movement| {
            let mut after = game.clone();
            if !execute_scored_move(&mut after, movement) {
                return None;
            }
            let bag = BagInfo::from_game_for_player(&after, player);
            let board = &after.boards[player];
            Some(SparseNnueAfterstate {
                movement: *movement,
                immediate_score: ScoreBreakdown::compute(&mut board.clone(), &cards).total as f32,
                features: extract_features_with_bag(board, Some(&bag)),
            })
        })
        .collect()
}

fn score_afterstates<E: SparseNnueEvaluator>(
    evaluator: &mut E,
    afterstates: Vec<SparseNnueAfterstate>,
    diagnostics: &mut BatchedNnueDiagnostics,
) -> Result<Vec<(f32, ScoredMove)>, BatchedNnueError<E::Error>> {
    let features = afterstates
        .iter()
        .map(|afterstate| afterstate.features.clone())
        .collect::<Vec<_>>();
    let remaining = evaluate_checked(evaluator, &features, diagnostics)?;
    Ok(afterstates
        .into_iter()
        .zip(remaining)
        .map(|(afterstate, value)| (afterstate.immediate_score + value, afterstate.movement))
        .collect())
}

pub fn nnue_prefilter_candidates_batched<E: SparseNnueEvaluator>(
    game: &GameState,
    evaluator: &mut E,
    candidates: Vec<ScoredMove>,
    k: usize,
    diagnostics: &mut BatchedNnueDiagnostics,
) -> Result<Vec<ScoredMove>, BatchedNnueError<E::Error>> {
    validate_qualified_environment()?;
    let diverse = option_enabled("MCE_DIVERSE_PREFILTER");
    let mut scored = score_afterstates(
        evaluator,
        prepare_sparse_nnue_afterstates(game, &candidates),
        diagnostics,
    )?;
    scored.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| scored_move_identity(&left.1).cmp(&scored_move_identity(&right.1)))
    });
    let all_scored = scored
        .into_iter()
        .map(|(_, movement)| movement)
        .collect::<Vec<_>>();
    if !diverse || all_scored.is_empty() {
        return Ok(all_scored.into_iter().take(k).collect());
    }

    let player = game.current_player;
    let cards = game.scoring_cards;
    let mut selected = Vec::with_capacity(k + 4);
    let mut used_keys: HashSet<(usize, i8, i8, u8, Option<i8>, Option<i8>)> = HashSet::new();
    let add =
        |movement: ScoredMove, selected: &mut Vec<ScoredMove>, keys: &mut HashSet<_>| -> bool {
            let key = (
                movement.market_index,
                movement.tile_q,
                movement.tile_r,
                movement.rotation,
                movement.wildlife_q,
                movement.wildlife_r,
            );
            if !keys.insert(key) {
                return false;
            }
            selected.push(movement);
            true
        };
    for movement in deterministic_market_representatives(&all_scored) {
        add(movement, &mut selected, &mut used_keys);
    }

    struct VariantAfterstate {
        base_index: usize,
        q: i8,
        r: i8,
        immediate: f32,
        features: Vec<u16>,
    }
    let variants = all_scored
        .iter()
        .take(3)
        .enumerate()
        .flat_map(|(base_index, base)| {
            let Some(market_pair) = game.market.pairs[base.market_index] else {
                return Vec::new();
            };
            let wildlife = if let Some(index) = base.wildlife_market_index {
                let Some(pair) = game.market.pairs[index] else {
                    return Vec::new();
                };
                pair.wildlife
            } else {
                market_pair.wildlife
            };
            let mut after = game.clone();
            if after.boards[player]
                .place_tile(
                    HexCoord::new(base.tile_q, base.tile_r),
                    market_pair.tile,
                    base.rotation,
                )
                .is_none()
            {
                return Vec::new();
            }
            let placed = after.boards[player]
                .placed_tiles
                .iter()
                .copied()
                .collect::<Vec<_>>();
            let mut rows = Vec::new();
            for tile_index in placed {
                if !after.boards[player]
                    .grid
                    .get(tile_index as usize)
                    .can_place_wildlife(wildlife)
                {
                    continue;
                }
                let Some(undo) = after.boards[player].place_wildlife(tile_index as usize, wildlife)
                else {
                    continue;
                };
                let bag = BagInfo::from_game_for_player(&after, player);
                let immediate =
                    ScoreBreakdown::compute(&mut after.boards[player].clone(), &cards).total as f32;
                let features = extract_features_with_bag(&after.boards[player], Some(&bag));
                let coordinate = HexCoord::from_index(tile_index as usize);
                rows.push(VariantAfterstate {
                    base_index,
                    q: coordinate.q,
                    r: coordinate.r,
                    immediate,
                    features,
                });
                after.boards[player].undo(undo);
            }
            rows
        })
        .collect::<Vec<_>>();
    let variant_features = variants
        .iter()
        .map(|variant| variant.features.clone())
        .collect::<Vec<_>>();
    let variant_values = evaluate_checked(evaluator, &variant_features, diagnostics)?;
    let mut grouped = vec![Vec::<(i8, i8, f32)>::new(); all_scored.len().min(3)];
    for (variant, remaining) in variants.into_iter().zip(variant_values) {
        grouped[variant.base_index].push((variant.q, variant.r, variant.immediate + remaining));
    }
    for (base_index, options) in grouped.iter_mut().enumerate() {
        options.sort_by(|left, right| {
            right
                .2
                .partial_cmp(&left.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| (left.0, left.1).cmp(&(right.0, right.1)))
        });
        let base = all_scored[base_index];
        let mut added = 0;
        for &(q, r, score) in options.iter() {
            if added >= 2 {
                break;
            }
            if base.wildlife_q == Some(q) && base.wildlife_r == Some(r) {
                continue;
            }
            let movement = ScoredMove {
                wildlife_q: Some(q),
                wildlife_r: Some(r),
                score: score as u16,
                eval: (score * 1000.0) as i32,
                ..base
            };
            if add(movement, &mut selected, &mut used_keys) {
                added += 1;
            }
        }
    }
    for movement in all_scored {
        if selected.len() >= k {
            break;
        }
        add(movement, &mut selected, &mut used_keys);
    }
    selected.truncate(k);
    Ok(selected)
}

fn candidate_priors<E: SparseNnueEvaluator>(
    game: &GameState,
    candidates: &[ScoredMove],
    evaluator: &mut E,
    diagnostics: &mut BatchedNnueDiagnostics,
) -> Result<Vec<f32>, BatchedNnueError<E::Error>> {
    let player = game.current_player;
    let afterstates = candidates
        .par_iter()
        .map(|movement| {
            let mut after = game.clone();
            if !execute_scored_move(&mut after, movement) {
                return None;
            }
            let bag = BagInfo::from_game_for_player(&after, player);
            let immediate = mce_score_total(&mut after.clone(), player) as f32;
            let features = extract_features_with_bag(&after.boards[player], Some(&bag));
            Some((immediate, features))
        })
        .collect::<Vec<_>>();
    let rows = afterstates
        .iter()
        .filter_map(|afterstate| afterstate.as_ref().map(|(_, features)| features.clone()))
        .collect::<Vec<_>>();
    let values = evaluate_checked(evaluator, &rows, diagnostics)?;
    let mut value_index = 0;
    Ok(afterstates
        .into_iter()
        .map(|afterstate| {
            let Some((immediate, _)) = afterstate else {
                return 0.0;
            };
            let value = immediate + values[value_index];
            value_index += 1;
            value
        })
        .collect())
}

struct RolloutState {
    candidate_index: usize,
    game: GameState,
    player: usize,
    focal_turns: u8,
    score: Option<u64>,
    rollout_seed: u64,
    trace: Option<Vec<RolloutTracePoint>>,
}

#[derive(Debug, Clone)]
struct RolloutTracePoint {
    personal_turn: u8,
    immediate_score: f32,
    features: Vec<u16>,
}

impl RolloutState {
    fn new(
        game: &GameState,
        player: usize,
        seed: u64,
        candidate_index: usize,
        movement: &ScoredMove,
        trace_modulus: Option<u64>,
    ) -> Self {
        let mut game = game.clone();
        let mut rng = StdRng::seed_from_u64(seed);
        game.shuffle_bags(&mut rng);
        let score = (!execute_scored_move(&mut game, movement)).then_some(0);
        let trace = if score.is_none()
            && trace_modulus.is_some_and(|modulus| modulus != 0 && seed % modulus == 0)
        {
            let bag = BagInfo::from_game_for_player(&game, player);
            let immediate_score = mce_score_total(&mut game.clone(), player) as f32;
            Some(vec![RolloutTracePoint {
                personal_turn: personal_turn(&game, player),
                immediate_score,
                features: extract_features_with_bag(&game.boards[player], Some(&bag)),
            }])
        } else {
            None
        };
        Self {
            candidate_index,
            game,
            player,
            focal_turns: 1,
            score,
            rollout_seed: seed,
            trace,
        }
    }

    fn advance_to_player(&mut self) {
        if self.score.is_some() {
            return;
        }
        while !self.game.is_game_over() && self.game.current_player != self.player {
            if self.game.can_replace_overflow().is_some() {
                self.game.replace_overflow();
            }
            let Some(movement) = greedy_move(&self.game) else {
                self.finish();
                return;
            };
            if !execute_scored_move(&mut self.game, &movement) {
                self.finish();
                return;
            }
        }
        if self.game.is_game_over() {
            self.finish();
        } else if self.game.can_replace_overflow().is_some() {
            self.game.replace_overflow();
        }
    }

    fn finish(&mut self) {
        self.score = Some(mce_score_total(&mut self.game, self.player) as u64);
    }
}

fn personal_turn(game: &GameState, player: usize) -> u8 {
    game.boards[player].tile_count.saturating_sub(3).min(20) as u8
}

struct RolloutBatchResult {
    scores: Vec<(usize, u64)>,
    samples: Vec<RolloutValueSample>,
}

struct PreparedPipelinedRolloutChunk {
    state_indices: Vec<usize>,
    state_range: Range<usize>,
    prepared: Vec<PreparedNnueMove>,
    group_rows: Vec<Range<usize>>,
    unique_start: usize,
    unique_end: usize,
    request_rows: Vec<Vec<u16>>,
}

struct SparseEvaluationRequest {
    rows: Vec<Vec<u16>>,
}

struct SparseEvaluationResponse<E> {
    values: Result<Vec<f32>, E>,
    elapsed_ns: u64,
}

fn prepare_pipelined_rollout_chunk(
    states: &mut [RolloutState],
    active: &[usize],
    all_rows: &mut Vec<Vec<u16>>,
    dedup: &mut SparseRowDedupBuilder,
    mut reuse_tracker: Option<&mut SparseRowReuseTracker>,
    diagnostics: &mut BatchedNnueDiagnostics,
    timings_enabled: bool,
) -> PreparedPipelinedRolloutChunk {
    debug_assert!(active.windows(2).all(|pair| pair[0] < pair[1]));
    debug_assert!(active
        .iter()
        .all(|&state_index| states[state_index].score.is_none()));
    let state_start = *active
        .first()
        .expect("a pipelined rollout chunk must contain an active state");
    let state_end = active
        .last()
        .expect("a pipelined rollout chunk must contain an active state")
        .saturating_add(1);
    let state_range = state_start..state_end;
    debug_assert_eq!(
        states[state_range.clone()]
            .iter()
            .filter(|state| state.score.is_none())
            .count(),
        active.len()
    );

    let direct_preparation_started = stage_timer(timings_enabled);
    let prepared = states[state_range.clone()]
        .par_iter_mut()
        .filter_map(|state| {
            state
                .score
                .is_none()
                .then(|| prepare_nnue_move_direct_mut(&mut state.game))
        })
        .collect::<Vec<_>>();
    diagnostics.stage_timings.template_preparation_ns = diagnostics
        .stage_timings
        .template_preparation_ns
        .saturating_add(elapsed_ns(direct_preparation_started));
    debug_assert_eq!(prepared.len(), active.len());

    let row_assembly_started = stage_timer(timings_enabled);
    let mut prepared = prepared;
    let group_rows = prepared
        .iter_mut()
        .map(|group| {
            let start = all_rows.len();
            all_rows.extend(
                group
                    .candidates
                    .iter_mut()
                    .map(|candidate| std::mem::take(&mut candidate.features)),
            );
            start..all_rows.len()
        })
        .collect::<Vec<_>>();
    let chunk_row_start = group_rows
        .first()
        .map_or(all_rows.len(), |range| range.start);
    diagnostics.stage_timings.row_assembly_ns = diagnostics
        .stage_timings
        .row_assembly_ns
        .saturating_add(elapsed_ns(row_assembly_started));

    let dedup_started = stage_timer(timings_enabled);
    let unique_start = dedup.unique_indices.len();
    for row_index in chunk_row_start..all_rows.len() {
        let fingerprint = sparse_row_fingerprint(&all_rows[row_index]);
        let (_, is_new) = dedup.push_with_fingerprint(all_rows, row_index, fingerprint);
        if is_new {
            if let Some(tracker) = reuse_tracker.as_deref_mut() {
                diagnostics.reuse_observed_physical_rows =
                    diagnostics.reuse_observed_physical_rows.saturating_add(1);
                if tracker.observe(&all_rows[row_index], fingerprint) {
                    diagnostics.reuse_repeated_physical_rows =
                        diagnostics.reuse_repeated_physical_rows.saturating_add(1);
                }
            }
        }
    }
    let unique_end = dedup.unique_indices.len();
    diagnostics.stage_timings.row_deduplication_ns = diagnostics
        .stage_timings
        .row_deduplication_ns
        .saturating_add(elapsed_ns(dedup_started));

    let materialization_started = stage_timer(timings_enabled);
    let request_rows = dedup.unique_indices[unique_start..unique_end]
        .iter()
        .map(|&row_index| all_rows[row_index].clone())
        .collect();
    diagnostics.stage_timings.row_materialization_ns = diagnostics
        .stage_timings
        .row_materialization_ns
        .saturating_add(elapsed_ns(materialization_started));

    PreparedPipelinedRolloutChunk {
        state_indices: active.to_vec(),
        state_range,
        prepared,
        group_rows,
        unique_start,
        unique_end,
        request_rows,
    }
}

fn prepare_next_pipelined_rollout_chunk(
    states: &mut [RolloutState],
    active: &[usize],
    all_rows: &mut Vec<Vec<u16>>,
    dedup: &mut SparseRowDedupBuilder,
    reuse_tracker: Option<&mut SparseRowReuseTracker>,
    diagnostics: &mut BatchedNnueDiagnostics,
    timings_enabled: bool,
    chunk_states: usize,
    next_active_offset: &mut usize,
) -> Option<PreparedPipelinedRolloutChunk> {
    if *next_active_offset >= active.len() {
        return None;
    }
    let start = *next_active_offset;
    let end = (start + chunk_states).min(active.len());
    *next_active_offset = end;
    Some(prepare_pipelined_rollout_chunk(
        states,
        &active[start..end],
        all_rows,
        dedup,
        reuse_tracker,
        diagnostics,
        timings_enabled,
    ))
}

fn submit_pipelined_rollout_chunk<E>(
    requests: &SyncSender<SparseEvaluationRequest>,
    chunk: &mut PreparedPipelinedRolloutChunk,
) -> Result<bool, BatchedNnueError<E>> {
    if chunk.request_rows.is_empty() {
        return Ok(false);
    }
    requests
        .send(SparseEvaluationRequest {
            rows: std::mem::take(&mut chunk.request_rows),
        })
        .map_err(|_| BatchedNnueError::EvaluatorWorkerDisconnected)?;
    Ok(true)
}

fn receive_pipelined_rollout_chunk<E>(
    responses: &Receiver<SparseEvaluationResponse<E>>,
    chunk: &PreparedPipelinedRolloutChunk,
    unique_values: &mut Vec<f32>,
    diagnostics: &mut BatchedNnueDiagnostics,
    timings_enabled: bool,
) -> Result<(), BatchedNnueError<E>> {
    let response = responses
        .recv()
        .map_err(|_| BatchedNnueError::EvaluatorWorkerDisconnected)?;
    if timings_enabled {
        diagnostics.stage_timings.neural_evaluation_ns = diagnostics
            .stage_timings
            .neural_evaluation_ns
            .saturating_add(response.elapsed_ns);
    }
    let postprocess_started = stage_timer(timings_enabled);
    let values = response.values.map_err(BatchedNnueError::Evaluator)?;
    let expected = chunk.unique_end - chunk.unique_start;
    if values.len() != expected {
        return Err(BatchedNnueError::InvalidPredictionWidth {
            expected,
            actual: values.len(),
        });
    }
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(BatchedNnueError::NonFinitePrediction {
            index: chunk.unique_start + index,
        });
    }
    debug_assert_eq!(unique_values.len(), chunk.unique_start);
    unique_values.extend(values);
    diagnostics.stage_timings.prediction_postprocess_ns = diagnostics
        .stage_timings
        .prediction_postprocess_ns
        .saturating_add(elapsed_ns(postprocess_started));
    Ok(())
}

fn apply_pipelined_rollout_chunk(
    states: &mut [RolloutState],
    player: usize,
    chunk: &PreparedPipelinedRolloutChunk,
    all_rows: &[Vec<u16>],
    row_to_unique: &[usize],
    unique_values: &[f32],
    diagnostics: &mut BatchedNnueDiagnostics,
    timings_enabled: bool,
) {
    let action_selection_started = stage_timer(timings_enabled);
    let mut policy_fallbacks = 0_u64;
    for ((&state_index, group), rows) in chunk
        .state_indices
        .iter()
        .zip(&chunk.prepared)
        .zip(&chunk.group_rows)
    {
        let state = &mut states[state_index];
        let mut values = ArrayVec::<f32, 16>::new();
        values.extend(
            rows.clone()
                .map(|row_index| unique_values[row_to_unique[row_index]]),
        );
        let selected_index = select_prepared_nnue_candidate_index(group, values.as_slice());
        let movement = selected_index
            .map(|index| group.candidates[index].movement)
            .or(group.fallback)
            .or_else(|| greedy_move(&state.game));
        let Some(movement) = movement else {
            policy_fallbacks += 1;
            state.finish();
            continue;
        };
        if !execute_scored_move(&mut state.game, &movement) {
            policy_fallbacks += 1;
            state.finish();
            continue;
        }

        state.focal_turns = state.focal_turns.saturating_add(1);
        if state.trace.is_some() {
            let trace_point = if let Some(candidate_index) = selected_index {
                let candidate = &group.candidates[candidate_index];
                RolloutTracePoint {
                    personal_turn: personal_turn(&state.game, player),
                    immediate_score: candidate.actual_score,
                    features: all_rows[rows.start + candidate_index].clone(),
                }
            } else {
                let bag = BagInfo::from_game_for_player(&state.game, player);
                let immediate_score = mce_score_total(&mut state.game.clone(), player) as f32;
                RolloutTracePoint {
                    personal_turn: personal_turn(&state.game, player),
                    immediate_score,
                    features: extract_features_with_bag(&state.game.boards[player], Some(&bag)),
                }
            };
            state
                .trace
                .as_mut()
                .expect("trace presence was checked")
                .push(trace_point);
        }
    }
    diagnostics.policy_fallbacks = diagnostics
        .policy_fallbacks
        .saturating_add(policy_fallbacks);
    diagnostics.stage_timings.action_selection_ns = diagnostics
        .stage_timings
        .action_selection_ns
        .saturating_add(elapsed_ns(action_selection_started));

    let opponent_advance_started = stage_timer(timings_enabled);
    states[chunk.state_range.clone()]
        .par_iter_mut()
        .for_each(RolloutState::advance_to_player);
    diagnostics.stage_timings.opponent_advance_ns = diagnostics
        .stage_timings
        .opponent_advance_ns
        .saturating_add(elapsed_ns(opponent_advance_started));
}

fn run_pipelined_rollout_wave<E>(
    states: &mut [RolloutState],
    active: &[usize],
    player: usize,
    chunk_states: usize,
    requests: &SyncSender<SparseEvaluationRequest>,
    responses: &Receiver<SparseEvaluationResponse<E>>,
    mut reuse_tracker: Option<&mut SparseRowReuseTracker>,
    diagnostics: &mut BatchedNnueDiagnostics,
    timings_enabled: bool,
) -> Result<(), BatchedNnueError<E>> {
    diagnostics.rollout_waves += 1;
    let candidate_keying_started = stage_timer(timings_enabled);
    if template_reuse_diagnostics_enabled() {
        let unique_public_states = active
            .par_iter()
            .map(|&index| candidate_cache_key(&states[index].game))
            .collect::<HashSet<_>>()
            .len();
        diagnostics.template_state_requests = diagnostics
            .template_state_requests
            .saturating_add(active.len() as u64);
        diagnostics.unique_public_template_states = diagnostics
            .unique_public_template_states
            .saturating_add(unique_public_states as u64);
        let unique_board_states = active
            .par_iter()
            .map(|&index| candidate_board_cache_key(&states[index].game))
            .collect::<HashSet<_>>()
            .len();
        diagnostics.unique_board_template_states = diagnostics
            .unique_board_template_states
            .saturating_add(unique_board_states as u64);
    }
    diagnostics.stage_timings.candidate_keying_ns = diagnostics
        .stage_timings
        .candidate_keying_ns
        .saturating_add(elapsed_ns(candidate_keying_started));

    let mut all_rows = Vec::new();
    let mut dedup = SparseRowDedupBuilder::with_capacity(active.len().saturating_mul(15));
    let mut unique_values = Vec::new();
    let mut next_active_offset = 0;
    let mut current = prepare_next_pipelined_rollout_chunk(
        states,
        active,
        &mut all_rows,
        &mut dedup,
        reuse_tracker.as_deref_mut(),
        diagnostics,
        timings_enabled,
        chunk_states,
        &mut next_active_offset,
    )
    .expect("a pipelined rollout wave must contain an active state");
    let mut current_submitted = submit_pipelined_rollout_chunk(requests, &mut current)?;

    loop {
        let mut next = prepare_next_pipelined_rollout_chunk(
            states,
            active,
            &mut all_rows,
            &mut dedup,
            reuse_tracker.as_deref_mut(),
            diagnostics,
            timings_enabled,
            chunk_states,
            &mut next_active_offset,
        );
        if current_submitted {
            receive_pipelined_rollout_chunk(
                responses,
                &current,
                &mut unique_values,
                diagnostics,
                timings_enabled,
            )?;
        } else {
            debug_assert_eq!(current.unique_start, current.unique_end);
        }
        let next_submitted = if let Some(next) = next.as_mut() {
            submit_pipelined_rollout_chunk(requests, next)?
        } else {
            false
        };
        apply_pipelined_rollout_chunk(
            states,
            player,
            &current,
            &all_rows,
            &dedup.row_to_unique,
            &unique_values,
            diagnostics,
            timings_enabled,
        );
        let Some(next) = next else {
            break;
        };
        current = next;
        current_submitted = next_submitted;
    }

    if !all_rows.is_empty() {
        diagnostics.record_batch_work(all_rows.len(), dedup.unique_indices.len());
    }
    Ok(())
}

fn collect_rollout_batch(
    states: Vec<RolloutState>,
    diagnostics: &mut BatchedNnueDiagnostics,
    timings_enabled: bool,
) -> RolloutBatchResult {
    let terminal_collection_started = stage_timer(timings_enabled);
    let mut scores = Vec::with_capacity(states.len());
    let mut samples = Vec::new();
    for state in states {
        let terminal_score = state.score.unwrap_or(0);
        scores.push((state.candidate_index, terminal_score));
        if let Some(trace) = state.trace {
            samples.extend(trace.into_iter().map(|point| RolloutValueSample {
                rollout_seed: state.rollout_seed,
                personal_turn: point.personal_turn,
                immediate_score: point.immediate_score,
                target_remaining: terminal_score as f32 - point.immediate_score,
                features: point.features,
            }));
        }
    }
    diagnostics.stage_timings.terminal_collection_ns = diagnostics
        .stage_timings
        .terminal_collection_ns
        .saturating_add(elapsed_ns(terminal_collection_started));
    RolloutBatchResult { scores, samples }
}

fn run_rollout_batch_full_pipelined<E>(
    game: &GameState,
    player: usize,
    candidates: &[ScoredMove],
    work_items: &[(usize, u64)],
    evaluator: &mut E,
    mut reuse_tracker: Option<&mut SparseRowReuseTracker>,
    diagnostics: &mut BatchedNnueDiagnostics,
    trace_modulus: Option<u64>,
    chunk_states: usize,
) -> Result<RolloutBatchResult, BatchedNnueError<E::Error>>
where
    E: SparseNnueEvaluator,
{
    if chunk_states == 0 {
        return Err(BatchedNnueError::InvalidPipelineChunkSize(chunk_states));
    }
    let timings_enabled = stage_timings_enabled();
    diagnostics.rollout_samples += work_items.len() as u64;
    let initialization_started = stage_timer(timings_enabled);
    let mut states = work_items
        .par_iter()
        .map(|&(candidate_index, seed)| {
            RolloutState::new(
                game,
                player,
                seed,
                candidate_index,
                &candidates[candidate_index],
                trace_modulus,
            )
        })
        .collect::<Vec<_>>();
    diagnostics.stage_timings.rollout_state_initialization_ns = diagnostics
        .stage_timings
        .rollout_state_initialization_ns
        .saturating_add(elapsed_ns(initialization_started));

    let opponent_advance_started = stage_timer(timings_enabled);
    states
        .par_iter_mut()
        .for_each(RolloutState::advance_to_player);
    diagnostics.stage_timings.opponent_advance_ns = diagnostics
        .stage_timings
        .opponent_advance_ns
        .saturating_add(elapsed_ns(opponent_advance_started));

    thread::scope(|scope| {
        let (request_sender, request_receiver) = mpsc::sync_channel::<SparseEvaluationRequest>(1);
        let (response_sender, response_receiver) =
            mpsc::sync_channel::<SparseEvaluationResponse<E::Error>>(1);
        let worker = scope.spawn(move || {
            while let Ok(request) = request_receiver.recv() {
                let evaluation_started = stage_timer(timings_enabled);
                let values = evaluator.evaluate_sparse_owned(request.rows);
                let elapsed_ns = elapsed_ns(evaluation_started);
                let stop = values.is_err();
                if response_sender
                    .send(SparseEvaluationResponse { values, elapsed_ns })
                    .is_err()
                {
                    break;
                }
                if stop {
                    break;
                }
            }
        });

        let result = (|| {
            loop {
                let active = states
                    .iter()
                    .enumerate()
                    .filter_map(|(index, state)| state.score.is_none().then_some(index))
                    .collect::<Vec<_>>();
                if active.is_empty() {
                    break;
                }
                run_pipelined_rollout_wave(
                    &mut states,
                    &active,
                    player,
                    chunk_states,
                    &request_sender,
                    &response_receiver,
                    reuse_tracker.as_deref_mut(),
                    diagnostics,
                    timings_enabled,
                )?;
            }
            Ok(())
        })();
        drop(request_sender);
        let worker_result = worker.join();
        match (result, worker_result) {
            (Err(error), _) => Err(error),
            (Ok(()), Err(_)) => Err(BatchedNnueError::EvaluatorWorkerPanicked),
            (Ok(()), Ok(())) => Ok(()),
        }
    })?;

    Ok(collect_rollout_batch(states, diagnostics, timings_enabled))
}

fn bootstrap_rollout_states<E, L>(
    states: &mut [RolloutState],
    indices: &[usize],
    evaluator: &mut E,
    leaf_evaluator: Option<&mut L>,
    diagnostics: &mut BatchedNnueDiagnostics,
) -> Result<(), BatchedNnueError<E::Error>>
where
    E: SparseNnueEvaluator,
    L: SparseNnueEvaluator<Error = E::Error>,
{
    if indices.is_empty() {
        return Ok(());
    }
    let prepared = indices
        .par_iter()
        .map(|&index| {
            let state = &states[index];
            let bag = BagInfo::from_game_for_player(&state.game, state.player);
            let immediate = mce_score_total(&mut state.game.clone(), state.player) as f32;
            let features = extract_features_with_bag(&state.game.boards[state.player], Some(&bag));
            (immediate, features)
        })
        .collect::<Vec<_>>();
    let rows = prepared
        .iter()
        .map(|(_, features)| features.clone())
        .collect::<Vec<_>>();
    let values = if let Some(leaf_evaluator) = leaf_evaluator {
        evaluate_checked(leaf_evaluator, &rows, diagnostics)?
    } else {
        evaluate_checked(evaluator, &rows, diagnostics)?
    };
    diagnostics.bootstrapped_samples += indices.len() as u64;
    for ((&state_index, (immediate, _)), remaining) in indices.iter().zip(prepared).zip(values) {
        states[state_index].score = Some((immediate + remaining).max(0.0) as u64);
    }
    Ok(())
}

fn run_rollout_batch_with_leaf<E, L>(
    game: &GameState,
    player: usize,
    candidates: &[ScoredMove],
    work_items: &[(usize, u64)],
    evaluator: &mut E,
    leaf_evaluator: Option<&mut L>,
    reuse_tracker: Option<&mut SparseRowReuseTracker>,
    diagnostics: &mut BatchedNnueDiagnostics,
    trace_modulus: Option<u64>,
    config: BatchedRolloutConfig,
) -> Result<RolloutBatchResult, BatchedNnueError<E::Error>>
where
    E: SparseNnueEvaluator,
    L: SparseNnueEvaluator<Error = E::Error>,
{
    let pipeline_chunk_states = (config == BatchedRolloutConfig::full()
        && leaf_evaluator.is_none()
        && sparse_row_dedup_enabled())
    .then(|| evaluator.rollout_pipeline_chunk_states())
    .flatten();
    if let Some(chunk_states) = pipeline_chunk_states {
        return run_rollout_batch_full_pipelined(
            game,
            player,
            candidates,
            work_items,
            evaluator,
            reuse_tracker,
            diagnostics,
            trace_modulus,
            chunk_states,
        );
    }
    run_rollout_batch_synchronous_with_leaf(
        game,
        player,
        candidates,
        work_items,
        evaluator,
        leaf_evaluator,
        diagnostics,
        trace_modulus,
        config,
    )
}

fn run_rollout_batch_synchronous_with_leaf<E, L>(
    game: &GameState,
    player: usize,
    candidates: &[ScoredMove],
    work_items: &[(usize, u64)],
    evaluator: &mut E,
    mut leaf_evaluator: Option<&mut L>,
    diagnostics: &mut BatchedNnueDiagnostics,
    trace_modulus: Option<u64>,
    config: BatchedRolloutConfig,
) -> Result<RolloutBatchResult, BatchedNnueError<E::Error>>
where
    E: SparseNnueEvaluator,
    L: SparseNnueEvaluator<Error = E::Error>,
{
    let timings_enabled = stage_timings_enabled();
    diagnostics.rollout_samples += work_items.len() as u64;
    let initialization_started = stage_timer(timings_enabled);
    let mut states = work_items
        .par_iter()
        .map(|&(candidate_index, seed)| {
            RolloutState::new(
                game,
                player,
                seed,
                candidate_index,
                &candidates[candidate_index],
                trace_modulus,
            )
        })
        .collect::<Vec<_>>();
    diagnostics.stage_timings.rollout_state_initialization_ns = diagnostics
        .stage_timings
        .rollout_state_initialization_ns
        .saturating_add(elapsed_ns(initialization_started));
    while states.iter().any(|state| state.score.is_none()) {
        if config.leaf_timing == BatchedRolloutLeafTiming::AfterFocalMove {
            let bootstrap = states
                .iter()
                .enumerate()
                .filter_map(|(index, state)| {
                    (state.score.is_none()
                        && config
                            .max_focal_turns
                            .is_some_and(|limit| state.focal_turns >= limit))
                    .then_some(index)
                })
                .collect::<Vec<_>>();
            bootstrap_rollout_states(
                &mut states,
                &bootstrap,
                evaluator,
                leaf_evaluator.as_deref_mut(),
                diagnostics,
            )?;
        }

        let opponent_advance_started = stage_timer(timings_enabled);
        states
            .par_iter_mut()
            .for_each(RolloutState::advance_to_player);
        let mut active = states
            .iter()
            .enumerate()
            .filter_map(|(index, state)| state.score.is_none().then_some(index))
            .collect::<Vec<_>>();
        diagnostics.stage_timings.opponent_advance_ns = diagnostics
            .stage_timings
            .opponent_advance_ns
            .saturating_add(elapsed_ns(opponent_advance_started));
        if active.is_empty() {
            break;
        }
        if config.leaf_timing == BatchedRolloutLeafTiming::AfterOpponentRound {
            let (bootstrap, remaining): (Vec<_>, Vec<_>) = active.into_iter().partition(|&index| {
                config
                    .max_focal_turns
                    .is_some_and(|limit| states[index].focal_turns >= limit)
            });
            bootstrap_rollout_states(
                &mut states,
                &bootstrap,
                evaluator,
                leaf_evaluator.as_deref_mut(),
                diagnostics,
            )?;
            active = remaining;
            if active.is_empty() {
                continue;
            }
        }

        diagnostics.rollout_waves += 1;
        let candidate_keying_started = stage_timer(timings_enabled);
        let public_keys = active
            .par_iter()
            .map(|&index| candidate_cache_key(&states[index].game))
            .collect::<Vec<_>>();
        let mut unique_lookup = HashMap::with_capacity(public_keys.len());
        let mut unique_state_indices = Vec::new();
        let mut template_indices = Vec::with_capacity(public_keys.len());
        for (&state_index, key) in active.iter().zip(public_keys) {
            let template_index = match unique_lookup.get(&key) {
                Some(&index) => index,
                None => {
                    let index = unique_state_indices.len();
                    unique_lookup.insert(key, index);
                    unique_state_indices.push(state_index);
                    index
                }
            };
            template_indices.push(template_index);
        }
        if template_reuse_diagnostics_enabled() {
            diagnostics.template_state_requests = diagnostics
                .template_state_requests
                .saturating_add(active.len() as u64);
            diagnostics.unique_public_template_states = diagnostics
                .unique_public_template_states
                .saturating_add(unique_state_indices.len() as u64);
            let unique_board_states = active
                .par_iter()
                .map(|&index| candidate_board_cache_key(&states[index].game))
                .collect::<HashSet<_>>()
                .len();
            diagnostics.unique_board_template_states = diagnostics
                .unique_board_template_states
                .saturating_add(unique_board_states as u64);
        }
        diagnostics.stage_timings.candidate_keying_ns = diagnostics
            .stage_timings
            .candidate_keying_ns
            .saturating_add(elapsed_ns(candidate_keying_started));
        let template_preparation_started = stage_timer(timings_enabled);
        let templates = unique_state_indices
            .par_iter()
            .map(|&index| prepare_nnue_move_template(&states[index].game))
            .collect::<Vec<_>>();
        diagnostics.stage_timings.template_preparation_ns = diagnostics
            .stage_timings
            .template_preparation_ns
            .saturating_add(elapsed_ns(template_preparation_started));
        let candidate_preparation_started = stage_timer(timings_enabled);
        let mut template_by_state = vec![None; states.len()];
        for (&state_index, &template_index) in active.iter().zip(&template_indices) {
            template_by_state[state_index] = Some(template_index);
        }
        let mut prepared_by_state = states
            .par_iter_mut()
            .enumerate()
            .map(|(index, state)| {
                template_by_state[index].map(|template_index| {
                    prepare_nnue_move_from_template_mut(&mut state.game, &templates[template_index])
                })
            })
            .collect::<Vec<_>>();
        let mut prepared = active
            .iter()
            .map(|&index| {
                prepared_by_state[index]
                    .take()
                    .expect("active rollout state must have a prepared policy")
            })
            .collect::<Vec<_>>();
        diagnostics.stage_timings.candidate_preparation_ns = diagnostics
            .stage_timings
            .candidate_preparation_ns
            .saturating_add(elapsed_ns(candidate_preparation_started));
        let row_assembly_started = stage_timer(timings_enabled);
        let offsets = prepared
            .iter()
            .scan(0usize, |offset, group| {
                let start = *offset;
                *offset += group.candidates.len();
                Some((start, *offset))
            })
            .collect::<Vec<_>>();
        let rows = prepared
            .iter_mut()
            .flat_map(|group| {
                group
                    .candidates
                    .iter_mut()
                    .map(|candidate| std::mem::take(&mut candidate.features))
            })
            .collect::<Vec<_>>();
        diagnostics.stage_timings.row_assembly_ns = diagnostics
            .stage_timings
            .row_assembly_ns
            .saturating_add(elapsed_ns(row_assembly_started));
        let values = evaluate_checked(evaluator, &rows, diagnostics)?;
        let action_selection_started = stage_timer(timings_enabled);
        for ((&state_index, group), &(start, end)) in active.iter().zip(&prepared).zip(&offsets) {
            let selected_index = select_prepared_nnue_candidate_index(group, &values[start..end]);
            let movement = selected_index
                .map(|index| group.candidates[index].movement)
                .or(group.fallback)
                .or_else(|| greedy_move(&states[state_index].game));
            let Some(movement) = movement else {
                diagnostics.policy_fallbacks += 1;
                states[state_index].finish();
                continue;
            };
            if !execute_scored_move(&mut states[state_index].game, &movement) {
                diagnostics.policy_fallbacks += 1;
                states[state_index].finish();
            } else {
                states[state_index].focal_turns = states[state_index].focal_turns.saturating_add(1);
                if states[state_index].trace.is_some() {
                    let trace_point = if let Some(candidate_index) = selected_index {
                        let candidate = &group.candidates[candidate_index];
                        RolloutTracePoint {
                            personal_turn: personal_turn(&states[state_index].game, player),
                            immediate_score: candidate.actual_score,
                            features: rows[start + candidate_index].clone(),
                        }
                    } else {
                        let bag = BagInfo::from_game_for_player(&states[state_index].game, player);
                        let immediate_score =
                            mce_score_total(&mut states[state_index].game.clone(), player) as f32;
                        RolloutTracePoint {
                            personal_turn: personal_turn(&states[state_index].game, player),
                            immediate_score,
                            features: extract_features_with_bag(
                                &states[state_index].game.boards[player],
                                Some(&bag),
                            ),
                        }
                    };
                    states[state_index]
                        .trace
                        .as_mut()
                        .expect("trace presence was checked")
                        .push(trace_point);
                }
            }
        }
        diagnostics.stage_timings.action_selection_ns = diagnostics
            .stage_timings
            .action_selection_ns
            .saturating_add(elapsed_ns(action_selection_started));
    }
    let terminal_collection_started = stage_timer(timings_enabled);
    let mut scores = Vec::with_capacity(states.len());
    let mut samples = Vec::new();
    for state in states {
        let terminal_score = state.score.unwrap_or(0);
        scores.push((state.candidate_index, terminal_score));
        if let Some(trace) = state.trace {
            samples.extend(trace.into_iter().map(|point| RolloutValueSample {
                rollout_seed: state.rollout_seed,
                personal_turn: point.personal_turn,
                immediate_score: point.immediate_score,
                target_remaining: terminal_score as f32 - point.immediate_score,
                features: point.features,
            }));
        }
    }
    diagnostics.stage_timings.terminal_collection_ns = diagnostics
        .stage_timings
        .terminal_collection_ns
        .saturating_add(elapsed_ns(terminal_collection_started));
    Ok(RolloutBatchResult { scores, samples })
}

#[cfg(test)]
fn run_rollout_batch<E: SparseNnueEvaluator>(
    game: &GameState,
    player: usize,
    candidates: &[ScoredMove],
    work_items: &[(usize, u64)],
    evaluator: &mut E,
    diagnostics: &mut BatchedNnueDiagnostics,
    trace_modulus: Option<u64>,
    config: BatchedRolloutConfig,
) -> Result<RolloutBatchResult, BatchedNnueError<E::Error>> {
    run_rollout_batch_with_leaf(
        game,
        player,
        candidates,
        work_items,
        evaluator,
        None::<&mut E>,
        None,
        diagnostics,
        trace_modulus,
        config,
    )
}

pub fn score_nnue_rollout_mce_seq_halving_batched<E: SparseNnueEvaluator>(
    game: &GameState,
    evaluator: &mut E,
    num_rollouts: usize,
    candidates: Vec<ScoredMove>,
    rng: &mut StdRng,
    diagnostics: &mut BatchedNnueDiagnostics,
) -> Result<Vec<MceMoveEstimate>, BatchedNnueError<E::Error>> {
    score_nnue_rollout_mce_seq_halving_batched_with_coupling(
        game,
        evaluator,
        num_rollouts,
        candidates,
        rng,
        diagnostics,
        RolloutSeedCoupling::Independent,
    )
}

pub fn score_nnue_rollout_mce_seq_halving_batched_with_coupling<E: SparseNnueEvaluator>(
    game: &GameState,
    evaluator: &mut E,
    num_rollouts: usize,
    candidates: Vec<ScoredMove>,
    rng: &mut StdRng,
    diagnostics: &mut BatchedNnueDiagnostics,
    seed_coupling: RolloutSeedCoupling,
) -> Result<Vec<MceMoveEstimate>, BatchedNnueError<E::Error>> {
    score_nnue_rollout_mce_seq_halving_batched_with_config_and_coupling(
        game,
        evaluator,
        num_rollouts,
        candidates,
        rng,
        diagnostics,
        BatchedRolloutConfig::full(),
        seed_coupling,
    )
}

pub fn score_nnue_rollout_mce_seq_halving_batched_with_config_and_coupling<
    E: SparseNnueEvaluator,
>(
    game: &GameState,
    evaluator: &mut E,
    num_rollouts: usize,
    candidates: Vec<ScoredMove>,
    rng: &mut StdRng,
    diagnostics: &mut BatchedNnueDiagnostics,
    config: BatchedRolloutConfig,
    seed_coupling: RolloutSeedCoupling,
) -> Result<Vec<MceMoveEstimate>, BatchedNnueError<E::Error>> {
    Ok(score_nnue_rollout_mce_seq_halving_batched_inner(
        game,
        evaluator,
        None::<&mut E>,
        num_rollouts,
        candidates,
        rng,
        diagnostics,
        None,
        seed_coupling,
        config,
    )?
    .estimates)
}

pub fn score_nnue_rollout_mce_seq_halving_batched_with_leaf_config_and_coupling<E, L>(
    game: &GameState,
    evaluator: &mut E,
    leaf_evaluator: &mut L,
    num_rollouts: usize,
    candidates: Vec<ScoredMove>,
    rng: &mut StdRng,
    diagnostics: &mut BatchedNnueDiagnostics,
    config: BatchedRolloutConfig,
    seed_coupling: RolloutSeedCoupling,
) -> Result<Vec<MceMoveEstimate>, BatchedNnueError<E::Error>>
where
    E: SparseNnueEvaluator,
    L: SparseNnueEvaluator<Error = E::Error>,
{
    Ok(score_nnue_rollout_mce_seq_halving_batched_inner(
        game,
        evaluator,
        Some(leaf_evaluator),
        num_rollouts,
        candidates,
        rng,
        diagnostics,
        None,
        seed_coupling,
        config,
    )?
    .estimates)
}

pub fn score_nnue_rollout_mce_seq_halving_batched_with_samples<E: SparseNnueEvaluator>(
    game: &GameState,
    evaluator: &mut E,
    num_rollouts: usize,
    candidates: Vec<ScoredMove>,
    rng: &mut StdRng,
    diagnostics: &mut BatchedNnueDiagnostics,
    trace_modulus: u64,
) -> Result<BatchedMceResult, BatchedNnueError<E::Error>> {
    if trace_modulus == 0 {
        return Err(BatchedNnueError::UnsupportedConfiguration(
            "trace_modulus=0",
        ));
    }
    score_nnue_rollout_mce_seq_halving_batched_inner(
        game,
        evaluator,
        None::<&mut E>,
        num_rollouts,
        candidates,
        rng,
        diagnostics,
        Some(trace_modulus),
        RolloutSeedCoupling::Independent,
        BatchedRolloutConfig::full(),
    )
}

pub fn score_nnue_rollout_mce_seq_halving_batched_with_samples_and_coupling<
    E: SparseNnueEvaluator,
>(
    game: &GameState,
    evaluator: &mut E,
    num_rollouts: usize,
    candidates: Vec<ScoredMove>,
    rng: &mut StdRng,
    diagnostics: &mut BatchedNnueDiagnostics,
    trace_modulus: u64,
    seed_coupling: RolloutSeedCoupling,
) -> Result<BatchedMceResult, BatchedNnueError<E::Error>> {
    score_nnue_rollout_mce_seq_halving_batched_with_samples_config_and_coupling(
        game,
        evaluator,
        num_rollouts,
        candidates,
        rng,
        diagnostics,
        trace_modulus,
        BatchedRolloutConfig::full(),
        seed_coupling,
    )
}

pub fn score_nnue_rollout_mce_seq_halving_batched_with_samples_config_and_coupling<
    E: SparseNnueEvaluator,
>(
    game: &GameState,
    evaluator: &mut E,
    num_rollouts: usize,
    candidates: Vec<ScoredMove>,
    rng: &mut StdRng,
    diagnostics: &mut BatchedNnueDiagnostics,
    trace_modulus: u64,
    config: BatchedRolloutConfig,
    seed_coupling: RolloutSeedCoupling,
) -> Result<BatchedMceResult, BatchedNnueError<E::Error>> {
    if trace_modulus == 0 {
        return Err(BatchedNnueError::UnsupportedConfiguration(
            "trace_modulus=0",
        ));
    }
    score_nnue_rollout_mce_seq_halving_batched_inner(
        game,
        evaluator,
        None::<&mut E>,
        num_rollouts,
        candidates,
        rng,
        diagnostics,
        Some(trace_modulus),
        seed_coupling,
        config,
    )
}

fn score_nnue_rollout_mce_seq_halving_batched_inner<E, L>(
    game: &GameState,
    evaluator: &mut E,
    mut leaf_evaluator: Option<&mut L>,
    num_rollouts: usize,
    candidates: Vec<ScoredMove>,
    rng: &mut StdRng,
    diagnostics: &mut BatchedNnueDiagnostics,
    trace_modulus: Option<u64>,
    seed_coupling: RolloutSeedCoupling,
    config: BatchedRolloutConfig,
) -> Result<BatchedMceResult, BatchedNnueError<E::Error>>
where
    E: SparseNnueEvaluator,
    L: SparseNnueEvaluator<Error = E::Error>,
{
    validate_qualified_environment()?;
    let player = game.current_player;
    if candidates.is_empty() {
        return Ok(BatchedMceResult {
            estimates: Vec::new(),
            rollout_value_samples: Vec::new(),
        });
    }
    let candidate_count = candidates.len();
    let mut totals = vec![0u64; candidate_count];
    let mut sumsq = vec![0u64; candidate_count];
    let mut counts = vec![0u32; candidate_count];
    let use_lmr = option_enabled("MCE_LMR");
    let priors = if use_lmr {
        candidate_priors(game, &candidates, evaluator, diagnostics)?
    } else {
        Vec::new()
    };
    let ranks = if use_lmr && !priors.is_empty() {
        let mut indexed = priors.iter().copied().enumerate().collect::<Vec<_>>();
        indexed.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut ranks = vec![0usize; priors.len()];
        for (rank, (index, _)) in indexed.into_iter().enumerate() {
            ranks[index] = rank;
        }
        ranks
    } else {
        Vec::new()
    };
    let lmr_multiplier = |index: usize| -> f64 {
        if !use_lmr || ranks.is_empty() {
            return 1.0;
        }
        match ranks[index] {
            0 => 2.0,
            1 => 1.5,
            _ => 1.0,
        }
    };
    let rounds = (candidate_count as f64).log2().ceil().max(1.0) as usize;
    let budget_per_round = (num_rollouts / rounds).max(candidate_count);
    let mut alive = (0..candidate_count).collect::<Vec<_>>();
    let mut rollout_value_samples = Vec::new();
    let mut reuse_tracker = sparse_row_reuse_diagnostics_enabled()
        .then(|| SparseRowReuseTracker::with_capacity(num_rollouts.saturating_mul(256)));
    for round in 0..rounds {
        if alive.is_empty() {
            break;
        }
        let base_per = (budget_per_round / alive.len()).max(1);
        let raw = alive
            .iter()
            .map(|&index| base_per as f64 * lmr_multiplier(index))
            .collect::<Vec<_>>();
        let raw_sum = raw.iter().sum::<f64>();
        let target_sum = base_per as f64 * alive.len() as f64;
        let scale = if raw_sum > 0.0 {
            target_sum / raw_sum
        } else {
            1.0
        };
        let allocations = raw
            .iter()
            .map(|value| ((value * scale).round() as usize).max(1))
            .collect::<Vec<_>>();
        let work_items = round_work_items(&alive, &allocations, rng, seed_coupling);
        let batch = run_rollout_batch_with_leaf(
            game,
            player,
            &candidates,
            &work_items,
            evaluator,
            leaf_evaluator.as_deref_mut(),
            reuse_tracker.as_mut(),
            diagnostics,
            trace_modulus,
            config,
        )?;
        rollout_value_samples.extend(batch.samples);
        for (candidate_index, score) in batch.scores {
            totals[candidate_index] += score;
            sumsq[candidate_index] += score * score;
            counts[candidate_index] += 1;
        }
        if round < rounds - 1 {
            let mut scored = alive
                .iter()
                .filter_map(|&index| {
                    (counts[index] > 0)
                        .then_some((index, totals[index] as f64 / counts[index] as f64))
                })
                .collect::<Vec<_>>();
            scored.sort_by(|left, right| {
                right
                    .1
                    .partial_cmp(&left.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            alive = scored
                .into_iter()
                .take((alive.len() + 1) / 2)
                .map(|(index, _)| index)
                .collect();
        }
    }
    let mut estimates = candidates
        .iter()
        .enumerate()
        .filter_map(|(index, movement)| {
            let samples = counts[index];
            if samples == 0 {
                return None;
            }
            let count = f64::from(samples);
            let rollout_mean = totals[index] as f64 / count;
            let variance = (sumsq[index] as f64 / count - rollout_mean * rollout_mean).max(0.0);
            Some(MceMoveEstimate {
                movement: *movement,
                rollout_mean,
                rollout_stddev: variance.sqrt(),
                samples,
            })
        })
        .collect::<Vec<_>>();
    estimates.sort_by(|left, right| {
        right
            .rollout_mean
            .partial_cmp(&left.rollout_mean)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(BatchedMceResult {
        estimates,
        rollout_value_samples,
    })
}

fn round_work_items(
    alive: &[usize],
    allocations: &[usize],
    rng: &mut StdRng,
    seed_coupling: RolloutSeedCoupling,
) -> Vec<(usize, u64)> {
    assert_eq!(alive.len(), allocations.len());
    let mut work_items = Vec::with_capacity(allocations.iter().sum());
    match seed_coupling {
        RolloutSeedCoupling::Independent => {
            for (&candidate_index, &allocation) in alive.iter().zip(allocations) {
                for _ in 0..allocation {
                    work_items.push((candidate_index, rng.gen()));
                }
            }
        }
        RolloutSeedCoupling::CommonWithinRound => {
            let maximum = allocations.iter().copied().max().unwrap_or(0);
            let shared = (0..maximum).map(|_| rng.gen()).collect::<Vec<_>>();
            for (&candidate_index, &allocation) in alive.iter().zip(allocations) {
                work_items.extend(
                    shared
                        .iter()
                        .take(allocation)
                        .map(|&seed| (candidate_index, seed)),
                );
            }
        }
    }
    work_items
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_core::types::ScoringCards;

    #[test]
    fn diagnostic_merge_preserves_totals_extrema_and_stage_timings() {
        let mut combined = BatchedNnueDiagnostics {
            neural_batches: 2,
            neural_rows: 20,
            physical_neural_rows: 16,
            minimum_batch_rows: 4,
            maximum_batch_rows: 12,
            rollout_waves: 3,
            stage_timings: BatchedNnueStageTimings {
                row_assembly_ns: 11,
                neural_evaluation_ns: 13,
                ..BatchedNnueStageTimings::default()
            },
            ..BatchedNnueDiagnostics::default()
        };
        combined.merge_from(BatchedNnueDiagnostics {
            neural_batches: 5,
            neural_rows: 50,
            physical_neural_rows: 40,
            reuse_observed_physical_rows: 7,
            reuse_repeated_physical_rows: 2,
            minimum_batch_rows: 2,
            maximum_batch_rows: 30,
            rollout_waves: 8,
            rollout_samples: 9,
            bootstrapped_samples: 10,
            policy_fallbacks: 11,
            template_state_requests: 12,
            unique_public_template_states: 13,
            unique_board_template_states: 14,
            stage_timings: BatchedNnueStageTimings {
                row_assembly_ns: 17,
                neural_evaluation_ns: 19,
                action_selection_ns: 23,
                ..BatchedNnueStageTimings::default()
            },
        });

        assert_eq!(combined.neural_batches, 7);
        assert_eq!(combined.neural_rows, 70);
        assert_eq!(combined.physical_neural_rows, 56);
        assert_eq!(combined.reuse_observed_physical_rows, 7);
        assert_eq!(combined.reuse_repeated_physical_rows, 2);
        assert_eq!(combined.minimum_batch_rows, 2);
        assert_eq!(combined.maximum_batch_rows, 30);
        assert_eq!(combined.rollout_waves, 11);
        assert_eq!(combined.rollout_samples, 9);
        assert_eq!(combined.bootstrapped_samples, 10);
        assert_eq!(combined.policy_fallbacks, 11);
        assert_eq!(combined.template_state_requests, 12);
        assert_eq!(combined.unique_public_template_states, 13);
        assert_eq!(combined.unique_board_template_states, 14);
        assert_eq!(combined.stage_timings.row_assembly_ns, 28);
        assert_eq!(combined.stage_timings.neural_evaluation_ns, 32);
        assert_eq!(combined.stage_timings.action_selection_ns, 23);
    }

    #[test]
    fn sparse_row_reuse_tracker_checks_exact_rows_after_fingerprint_collision() {
        let mut tracker = SparseRowReuseTracker::with_capacity(4);
        let fingerprint = 17;

        assert!(!tracker.observe(&[1, 2, 3], fingerprint));
        assert!(tracker.observe(&[1, 2, 3], fingerprint));
        assert!(!tracker.observe(&[4, 5, 6], fingerprint));
        assert!(tracker.observe(&[4, 5, 6], fingerprint));
        assert!(!tracker.observe(&[7, 8], fingerprint));
        assert!(tracker.observe(&[7, 8], fingerprint));
    }

    #[derive(Default)]
    struct ZeroEvaluator;

    impl SparseNnueEvaluator for ZeroEvaluator {
        type Error = Infallible;

        fn evaluate_sparse(&mut self, feature_sets: &[Vec<u16>]) -> Result<Vec<f32>, Self::Error> {
            Ok(vec![0.0; feature_sets.len()])
        }
    }

    struct ConstantEvaluator(f32);

    impl SparseNnueEvaluator for ConstantEvaluator {
        type Error = Infallible;

        fn evaluate_sparse(&mut self, feature_sets: &[Vec<u16>]) -> Result<Vec<f32>, Self::Error> {
            Ok(vec![self.0; feature_sets.len()])
        }
    }

    #[derive(Default)]
    struct RecordingEvaluator {
        rows: Vec<Vec<u16>>,
    }

    impl SparseNnueEvaluator for RecordingEvaluator {
        type Error = Infallible;

        fn evaluate_sparse(&mut self, feature_sets: &[Vec<u16>]) -> Result<Vec<f32>, Self::Error> {
            self.rows = feature_sets.to_vec();
            Ok(feature_sets
                .iter()
                .map(|features| features.iter().copied().map(f32::from).sum())
                .collect())
        }
    }

    struct PipelinedRecordingEvaluator {
        chunk_states: Option<usize>,
        requests: Vec<Vec<Vec<u16>>>,
    }

    impl PipelinedRecordingEvaluator {
        fn synchronous() -> Self {
            Self {
                chunk_states: None,
                requests: Vec::new(),
            }
        }

        fn pipelined(chunk_states: usize) -> Self {
            Self {
                chunk_states: Some(chunk_states),
                requests: Vec::new(),
            }
        }

        fn physical_rows(&self) -> usize {
            self.requests.iter().map(Vec::len).sum()
        }
    }

    impl SparseNnueEvaluator for PipelinedRecordingEvaluator {
        type Error = Infallible;

        fn evaluate_sparse(&mut self, feature_sets: &[Vec<u16>]) -> Result<Vec<f32>, Self::Error> {
            self.requests.push(feature_sets.to_vec());
            Ok(feature_sets
                .iter()
                .map(|features| features.iter().copied().map(f32::from).sum())
                .collect())
        }

        fn rollout_pipeline_chunk_states(&self) -> Option<usize> {
            self.chunk_states
        }
    }

    fn fresh_game(seed: u64) -> GameState {
        let mut rng = StdRng::seed_from_u64(seed);
        GameState::new(4, ScoringCards::all_a(), &mut rng)
    }

    #[test]
    fn exact_sparse_rows_are_deduplicated_and_scattered_in_original_order() {
        let rows = vec![vec![1, 2], vec![3, 4], vec![1, 2], vec![8], vec![3, 4]];
        let mut evaluator = RecordingEvaluator::default();
        let mut diagnostics = BatchedNnueDiagnostics::default();

        let values =
            evaluate_sparse_rows_deduplicated(&mut evaluator, &rows, &mut diagnostics).unwrap();

        assert_eq!(evaluator.rows, vec![vec![1, 2], vec![3, 4], vec![8]]);
        assert_eq!(values, vec![3.0, 7.0, 3.0, 8.0, 7.0]);
        assert_eq!(diagnostics.neural_batches, 1);
        assert_eq!(diagnostics.neural_rows, 5);
        assert_eq!(diagnostics.physical_neural_rows, 3);
    }

    #[test]
    fn sparse_row_dedup_remains_exact_when_fingerprints_collide() {
        let rows = vec![vec![1, 2], vec![3, 4], vec![1, 2], vec![8], vec![3, 4]];

        let dedup = deduplicate_sparse_rows_with(&rows, |_| 0);

        assert_eq!(dedup.unique_indices, vec![0, 1, 3]);
        assert_eq!(dedup.row_to_unique, vec![0, 1, 0, 2, 1]);
    }

    #[test]
    fn sparse_row_dedup_remains_exact_across_incremental_chunk_boundaries() {
        let chunks = [
            vec![vec![1, 2], vec![3, 4]],
            vec![vec![1, 2], vec![8]],
            vec![vec![3, 4]],
        ];
        let mut rows = Vec::new();
        let mut builder = SparseRowDedupBuilder::with_capacity(5);
        for chunk in chunks {
            let start = rows.len();
            rows.extend(chunk);
            for row_index in start..rows.len() {
                builder.push_with_fingerprint(&rows, row_index, 0);
            }
        }
        let dedup = builder.finish();

        assert_eq!(dedup.unique_indices, vec![0, 1, 3]);
        assert_eq!(dedup.row_to_unique, vec![0, 1, 0, 2, 1]);
    }

    #[test]
    fn independent_round_work_items_preserve_candidate_major_seed_order() {
        let mut actual_rng = StdRng::seed_from_u64(91);
        let actual = round_work_items(
            &[4, 9],
            &[2, 3],
            &mut actual_rng,
            RolloutSeedCoupling::Independent,
        );
        let mut expected_rng = StdRng::seed_from_u64(91);
        let expected = vec![
            (4, expected_rng.gen()),
            (4, expected_rng.gen()),
            (9, expected_rng.gen()),
            (9, expected_rng.gen()),
            (9, expected_rng.gen()),
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn common_round_work_items_share_ordered_seed_prefixes() {
        let mut rng = StdRng::seed_from_u64(92);
        let work = round_work_items(
            &[4, 9, 12],
            &[2, 3, 2],
            &mut rng,
            RolloutSeedCoupling::CommonWithinRound,
        );
        let candidate_four = work
            .iter()
            .filter_map(|&(candidate, seed)| (candidate == 4).then_some(seed))
            .collect::<Vec<_>>();
        let candidate_nine = work
            .iter()
            .filter_map(|&(candidate, seed)| (candidate == 9).then_some(seed))
            .collect::<Vec<_>>();
        let candidate_twelve = work
            .iter()
            .filter_map(|&(candidate, seed)| (candidate == 12).then_some(seed))
            .collect::<Vec<_>>();
        assert_eq!(candidate_four, candidate_twelve);
        assert_eq!(candidate_four, candidate_nine[..2]);
        assert_eq!(work.len(), 7);
    }

    #[test]
    fn common_random_number_search_replays_deterministically() {
        let game = fresh_game(93);
        let candidates = crate::mce::expanded_candidates(&game)
            .into_iter()
            .take(4)
            .collect::<Vec<_>>();
        assert_eq!(candidates.len(), 4);

        let run = || {
            let mut evaluator = ZeroEvaluator;
            let mut diagnostics = BatchedNnueDiagnostics::default();
            let mut rng = StdRng::seed_from_u64(94);
            let estimates = score_nnue_rollout_mce_seq_halving_batched_with_coupling(
                &game,
                &mut evaluator,
                8,
                candidates.clone(),
                &mut rng,
                &mut diagnostics,
                RolloutSeedCoupling::CommonWithinRound,
            )
            .unwrap();
            (estimates, diagnostics)
        };

        let (first, first_diagnostics) = run();
        let (second, second_diagnostics) = run();
        assert_eq!(first_diagnostics, second_diagnostics);
        assert!(first_diagnostics.rollout_samples > 0);
        assert_eq!(first.len(), second.len());
        for (left, right) in first.iter().zip(&second) {
            assert_eq!(
                scored_move_identity(&left.movement),
                scored_move_identity(&right.movement)
            );
            assert_eq!(left.rollout_mean.to_bits(), right.rollout_mean.to_bits());
            assert_eq!(
                left.rollout_stddev.to_bits(),
                right.rollout_stddev.to_bits()
            );
            assert_eq!(left.samples, right.samples);
        }
    }

    #[test]
    fn pipelined_rollout_matches_synchronous_trace_and_logical_diagnostics() {
        let game = fresh_game(95);
        let player = game.current_player;
        let candidates = crate::mce::expanded_candidates(&game)
            .into_iter()
            .take(3)
            .collect::<Vec<_>>();
        assert_eq!(candidates.len(), 3);
        let work = [(0, 101), (0, 102), (1, 103), (2, 104)];

        let mut synchronous_evaluator = PipelinedRecordingEvaluator::synchronous();
        let mut synchronous_diagnostics = BatchedNnueDiagnostics::default();
        let synchronous = run_rollout_batch(
            &game,
            player,
            &candidates,
            &work,
            &mut synchronous_evaluator,
            &mut synchronous_diagnostics,
            Some(1),
            BatchedRolloutConfig::full(),
        )
        .unwrap();

        let mut pipelined_evaluator = PipelinedRecordingEvaluator::pipelined(1);
        let mut pipelined_diagnostics = BatchedNnueDiagnostics::default();
        let pipelined = run_rollout_batch(
            &game,
            player,
            &candidates,
            &work,
            &mut pipelined_evaluator,
            &mut pipelined_diagnostics,
            Some(1),
            BatchedRolloutConfig::full(),
        )
        .unwrap();

        assert_eq!(pipelined.scores, synchronous.scores);
        assert_eq!(pipelined.samples, synchronous.samples);
        assert_eq!(pipelined_diagnostics, synchronous_diagnostics);
        assert_eq!(
            synchronous_evaluator.physical_rows(),
            synchronous_diagnostics.physical_neural_rows as usize
        );
        assert_eq!(
            pipelined_evaluator.physical_rows(),
            pipelined_diagnostics.physical_neural_rows as usize
        );
        assert!(pipelined_evaluator.requests.len() > synchronous_evaluator.requests.len());
    }

    #[test]
    fn rollout_trace_sampling_preserves_policy_afterstates_and_targets() {
        let game = fresh_game(71);
        let player = game.current_player;
        let movement = greedy_move(&game).expect("fresh game has a legal move");
        let rollout_seed = 8;

        let mut expected_game = game.clone();
        let mut shuffle_rng = StdRng::seed_from_u64(rollout_seed);
        expected_game.shuffle_bags(&mut shuffle_rng);
        assert!(execute_scored_move(&mut expected_game, &movement));
        let expected_bag = BagInfo::from_game_for_player(&expected_game, player);
        let expected_immediate = mce_score_total(&mut expected_game.clone(), player) as f32;
        let expected_features =
            extract_features_with_bag(&expected_game.boards[player], Some(&expected_bag));

        let mut traced_evaluator = ZeroEvaluator;
        let mut traced_diagnostics = BatchedNnueDiagnostics::default();
        let traced = run_rollout_batch(
            &game,
            player,
            &[movement],
            &[(0, rollout_seed)],
            &mut traced_evaluator,
            &mut traced_diagnostics,
            Some(8),
            BatchedRolloutConfig::full(),
        )
        .unwrap();

        assert_eq!(traced.scores.len(), 1);
        assert_eq!(traced.samples.len(), 20);
        assert_eq!(traced.samples[0].personal_turn, 1);
        assert_eq!(traced.samples[0].immediate_score, expected_immediate);
        assert_eq!(traced.samples[0].features, expected_features);
        let terminal_score = traced.scores[0].1 as f32;
        for sample in &traced.samples {
            assert_eq!(sample.rollout_seed, rollout_seed);
            assert!((1..=20).contains(&sample.personal_turn));
            assert!(!sample.features.is_empty());
            assert!(sample
                .features
                .iter()
                .all(|&feature| (feature as usize) < crate::nnue::NUM_FEATURES));
            assert_eq!(
                sample.immediate_score + sample.target_remaining,
                terminal_score
            );
        }
        assert_eq!(traced_diagnostics.policy_fallbacks, 0);

        let mut untraced_evaluator = ZeroEvaluator;
        let mut untraced_diagnostics = BatchedNnueDiagnostics::default();
        let untraced = run_rollout_batch(
            &game,
            player,
            &[movement],
            &[(0, rollout_seed)],
            &mut untraced_evaluator,
            &mut untraced_diagnostics,
            None,
            BatchedRolloutConfig::full(),
        )
        .unwrap();
        assert_eq!(untraced.scores, traced.scores);
        assert!(untraced.samples.is_empty());
        assert_eq!(untraced_diagnostics, traced_diagnostics);

        let mut skipped_evaluator = ZeroEvaluator;
        let mut skipped_diagnostics = BatchedNnueDiagnostics::default();
        let skipped = run_rollout_batch(
            &game,
            player,
            &[movement],
            &[(0, rollout_seed + 1)],
            &mut skipped_evaluator,
            &mut skipped_diagnostics,
            Some(8),
            BatchedRolloutConfig::full(),
        )
        .unwrap();
        assert!(skipped.samples.is_empty());
    }

    #[test]
    fn truncated_rollout_bootstraps_at_the_requested_focal_turn() {
        let game = fresh_game(72);
        let player = game.current_player;
        let movement = greedy_move(&game).expect("fresh game has a legal move");
        let work = [(0, 17)];

        let mut full_evaluator = ZeroEvaluator;
        let mut full_diagnostics = BatchedNnueDiagnostics::default();
        let full = run_rollout_batch(
            &game,
            player,
            &[movement],
            &work,
            &mut full_evaluator,
            &mut full_diagnostics,
            None,
            BatchedRolloutConfig::full(),
        )
        .unwrap();

        let mut truncated_evaluator = ZeroEvaluator;
        let mut truncated_diagnostics = BatchedNnueDiagnostics::default();
        let truncated = run_rollout_batch(
            &game,
            player,
            &[movement],
            &work,
            &mut truncated_evaluator,
            &mut truncated_diagnostics,
            None,
            BatchedRolloutConfig::truncated(2).unwrap(),
        )
        .unwrap();

        assert_eq!(truncated.scores.len(), 1);
        assert_eq!(truncated_diagnostics.bootstrapped_samples, 1);
        assert_eq!(truncated_diagnostics.rollout_waves, 1);
        assert!(full_diagnostics.rollout_waves > truncated_diagnostics.rollout_waves);
        assert!(truncated.scores[0].1 > 0);
        assert_eq!(full.scores.len(), 1);

        let mut afterstate_evaluator = ZeroEvaluator;
        let mut afterstate_diagnostics = BatchedNnueDiagnostics::default();
        let afterstate = run_rollout_batch(
            &game,
            player,
            &[movement],
            &work,
            &mut afterstate_evaluator,
            &mut afterstate_diagnostics,
            None,
            BatchedRolloutConfig::truncated_afterstate(2).unwrap(),
        )
        .unwrap();
        assert_eq!(afterstate.scores.len(), 1);
        assert_eq!(afterstate_diagnostics.bootstrapped_samples, 1);
        assert_eq!(afterstate_diagnostics.rollout_waves, 1);

        let mut policy_evaluator = ZeroEvaluator;
        let mut leaf_evaluator = ConstantEvaluator(10.0);
        let mut separate_leaf_diagnostics = BatchedNnueDiagnostics::default();
        let separate_leaf = run_rollout_batch_with_leaf(
            &game,
            player,
            &[movement],
            &work,
            &mut policy_evaluator,
            Some(&mut leaf_evaluator),
            None,
            &mut separate_leaf_diagnostics,
            None,
            BatchedRolloutConfig::truncated_afterstate(2).unwrap(),
        )
        .unwrap();
        assert_eq!(separate_leaf.scores[0].1, afterstate.scores[0].1 + 10);
        assert_eq!(separate_leaf_diagnostics.bootstrapped_samples, 1);
    }
}
