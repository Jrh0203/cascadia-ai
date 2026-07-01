//! Evaluator-independent batching for the qualified historical NNUE search.

use std::{
    collections::HashSet,
    convert::Infallible,
    error::Error,
    fmt::{self, Display},
};

use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;

use crate::{
    eval::ScoredMove,
    mce::{
        deterministic_market_representatives, mce_score_total, scored_move_identity,
        MceMoveEstimate,
    },
    nnue::{extract_features_with_bag, BagInfo, NNUENetwork},
    nnue_train::{prepare_nnue_move, select_prepared_nnue_candidate_index},
    search::{execute_scored_move, greedy_move},
};
use cascadia_core::{game::GameState, hex::HexCoord, scoring::ScoreBreakdown};

pub trait SparseNnueEvaluator {
    type Error;

    fn evaluate_sparse(&mut self, feature_sets: &[Vec<u16>]) -> Result<Vec<f32>, Self::Error>;
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
    pub minimum_batch_rows: usize,
    pub maximum_batch_rows: usize,
    pub rollout_waves: u64,
    pub rollout_samples: u64,
    pub policy_fallbacks: u64,
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
        self.neural_batches += 1;
        self.neural_rows += rows as u64;
        if self.minimum_batch_rows == 0 {
            self.minimum_batch_rows = rows;
        } else {
            self.minimum_batch_rows = self.minimum_batch_rows.min(rows);
        }
        self.maximum_batch_rows = self.maximum_batch_rows.max(rows);
    }
}

fn evaluate_checked<E: SparseNnueEvaluator>(
    evaluator: &mut E,
    feature_sets: &[Vec<u16>],
    diagnostics: &mut BatchedNnueDiagnostics,
) -> Result<Vec<f32>, BatchedNnueError<E::Error>> {
    if feature_sets.is_empty() {
        return Ok(Vec::new());
    }
    diagnostics.record_batch(feature_sets.len());
    let values = evaluator
        .evaluate_sparse(feature_sets)
        .map_err(BatchedNnueError::Evaluator)?;
    if values.len() != feature_sets.len() {
        return Err(BatchedNnueError::InvalidPredictionWidth {
            expected: feature_sets.len(),
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
    Ok(values)
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

fn run_rollout_batch<E: SparseNnueEvaluator>(
    game: &GameState,
    player: usize,
    candidates: &[ScoredMove],
    work_items: &[(usize, u64)],
    evaluator: &mut E,
    diagnostics: &mut BatchedNnueDiagnostics,
    trace_modulus: Option<u64>,
) -> Result<RolloutBatchResult, BatchedNnueError<E::Error>> {
    diagnostics.rollout_samples += work_items.len() as u64;
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
    while states.iter().any(|state| state.score.is_none()) {
        states
            .par_iter_mut()
            .for_each(RolloutState::advance_to_player);
        let active = states
            .iter()
            .enumerate()
            .filter_map(|(index, state)| state.score.is_none().then_some(index))
            .collect::<Vec<_>>();
        if active.is_empty() {
            break;
        }
        diagnostics.rollout_waves += 1;
        let prepared = active
            .par_iter()
            .map(|&index| prepare_nnue_move(&states[index].game))
            .collect::<Vec<_>>();
        let offsets = prepared
            .iter()
            .scan(0usize, |offset, group| {
                let start = *offset;
                *offset += group.candidates.len();
                Some((start, *offset))
            })
            .collect::<Vec<_>>();
        let rows = prepared
            .iter()
            .flat_map(|group| {
                group
                    .candidates
                    .iter()
                    .map(|candidate| candidate.features.clone())
            })
            .collect::<Vec<_>>();
        let values = evaluate_checked(evaluator, &rows, diagnostics)?;
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
            } else if states[state_index].trace.is_some() {
                let trace_point = if let Some(candidate_index) = selected_index {
                    let candidate = &group.candidates[candidate_index];
                    RolloutTracePoint {
                        personal_turn: personal_turn(&states[state_index].game, player),
                        immediate_score: candidate.actual_score,
                        features: candidate.features.clone(),
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
    Ok(RolloutBatchResult { scores, samples })
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
    Ok(score_nnue_rollout_mce_seq_halving_batched_inner(
        game,
        evaluator,
        num_rollouts,
        candidates,
        rng,
        diagnostics,
        None,
        seed_coupling,
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
        num_rollouts,
        candidates,
        rng,
        diagnostics,
        Some(trace_modulus),
        RolloutSeedCoupling::Independent,
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
    if trace_modulus == 0 {
        return Err(BatchedNnueError::UnsupportedConfiguration(
            "trace_modulus=0",
        ));
    }
    score_nnue_rollout_mce_seq_halving_batched_inner(
        game,
        evaluator,
        num_rollouts,
        candidates,
        rng,
        diagnostics,
        Some(trace_modulus),
        seed_coupling,
    )
}

fn score_nnue_rollout_mce_seq_halving_batched_inner<E: SparseNnueEvaluator>(
    game: &GameState,
    evaluator: &mut E,
    num_rollouts: usize,
    candidates: Vec<ScoredMove>,
    rng: &mut StdRng,
    diagnostics: &mut BatchedNnueDiagnostics,
    trace_modulus: Option<u64>,
    seed_coupling: RolloutSeedCoupling,
) -> Result<BatchedMceResult, BatchedNnueError<E::Error>> {
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
        let batch = run_rollout_batch(
            game,
            player,
            &candidates,
            &work_items,
            evaluator,
            diagnostics,
            trace_modulus,
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

    #[derive(Default)]
    struct ZeroEvaluator;

    impl SparseNnueEvaluator for ZeroEvaluator {
        type Error = Infallible;

        fn evaluate_sparse(&mut self, feature_sets: &[Vec<u16>]) -> Result<Vec<f32>, Self::Error> {
            Ok(vec![0.0; feature_sets.len()])
        }
    }

    fn fresh_game(seed: u64) -> GameState {
        let mut rng = StdRng::seed_from_u64(seed);
        GameState::new(4, ScoringCards::all_a(), &mut rng)
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
        )
        .unwrap();
        assert!(skipped.samples.is_empty());
    }
}
