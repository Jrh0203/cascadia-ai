use std::collections::HashMap;

use blake3::Hasher;
use cascadia_data::r2_map_draft_action_id;
use cascadia_game::{
    D6Transform, GameState, MarketDecision, MarketDecisionSession, MarketDecisionStage,
    MarketPrelude, TurnAction, public_market_action_identity, public_market_decision_identity,
    rescore_after_tile_with_habitat_analysis, rescore_after_wildlife_placement,
    rescore_with_wildlife_scores, score_board,
};
use cascadia_model::{
    R2MapInferenceCandidate, R2MapInferenceGroup, R2MapMarketInferenceCandidate,
    R2MapMarketInferenceGroup, R2MapMarketPredictionGroup, R2MapModelIdentity, R2MapModelProcess,
    R2MapPredictionGroup,
};
#[cfg(test)]
use cascadia_r2::encode_r2_map_action_bytes;
use cascadia_r2::{
    R2MapActionEncoder, R2MapIncrementalMaterializer, R2MapMarketDecisionKind,
    encode_r2_map_market_action_bytes, encode_r2_map_public_tensors,
};
use rayon::prelude::*;
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(test)]
use std::time::Instant;

use crate::SearchError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct R2MapExplorationChoice {
    pub gate_draw_u64: u64,
    pub epsilon_parts_per_million: u32,
    pub temperature_parts_per_million: u32,
    pub action_draws_u64: Vec<u64>,
}

impl R2MapExplorationChoice {
    fn validate(self, action_count: usize) -> Result<Self, SearchError> {
        if self.epsilon_parts_per_million > 1_000_000
            || self.temperature_parts_per_million == 0
            || self.action_draws_u64.len() != action_count
        {
            return Err(SearchError::InvalidConfig(
                "R2-MAP exploration epsilon or temperature is invalid",
            ));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct R2MapPreparedDecision {
    pub actions: Vec<TurnAction>,
    pub request: R2MapInferenceGroup,
}

#[derive(Debug, Clone, PartialEq)]
pub struct R2MapScoredDecision {
    pub action: TurnAction,
    pub selected_index: usize,
    pub explored: bool,
    pub action_ids: Vec<[u8; 32]>,
    pub scores: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct R2MapPreparedMarketDecision {
    pub decisions: Vec<MarketDecision>,
    pub request: R2MapMarketInferenceGroup,
}

#[derive(Debug, Clone, PartialEq)]
pub struct R2MapScoredMarketDecision {
    pub decision: MarketDecision,
    pub selected_index: usize,
    pub explored: bool,
    pub action_ids: Vec<[u8; 32]>,
    pub scores: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum R2MapTurnDecisionKind {
    Market(MarketDecisionStage),
    Draft,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct R2MapTurnDecisionContext {
    pub kind: R2MapTurnDecisionKind,
    pub ordinal: u8,
    pub decision_id: [u8; 32],
    pub parent_public_hash: [u8; 32],
    pub action_ids: Vec<[u8; 32]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct R2MapMarketDecisionTrace {
    pub context: R2MapTurnDecisionContext,
    pub selected_index: usize,
    pub selected: MarketDecision,
    pub resulting_public_hash: [u8; 32],
    pub explored: bool,
    pub scores: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct R2MapScoredTurn {
    pub action: TurnAction,
    pub bundled_action_id: [u8; 32],
    pub market_decisions: Vec<R2MapMarketDecisionTrace>,
    pub draft_context: R2MapTurnDecisionContext,
    pub draft: R2MapScoredDecision,
}

#[cfg(test)]
static REFERENCE_PREPARATION_NANOS: AtomicU64 = AtomicU64::new(0);
#[cfg(test)]
static RAYON_CACHE_PREPARATION_NANOS: AtomicU64 = AtomicU64::new(0);
#[cfg(test)]
static OPTIMIZED_PREPARATION_NANOS: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
pub(crate) fn reset_reference_parity_timing() {
    REFERENCE_PREPARATION_NANOS.store(0, Ordering::Relaxed);
    RAYON_CACHE_PREPARATION_NANOS.store(0, Ordering::Relaxed);
    OPTIMIZED_PREPARATION_NANOS.store(0, Ordering::Relaxed);
}

#[cfg(test)]
pub(crate) fn reference_parity_timing_seconds() -> (f64, f64, f64) {
    (
        REFERENCE_PREPARATION_NANOS.load(Ordering::Relaxed) as f64 / 1_000_000_000.0,
        RAYON_CACHE_PREPARATION_NANOS.load(Ordering::Relaxed) as f64 / 1_000_000_000.0,
        OPTIMIZED_PREPARATION_NANOS.load(Ordering::Relaxed) as f64 / 1_000_000_000.0,
    )
}

pub trait R2MapPredictor {
    fn score_r2_map_groups(
        &mut self,
        groups: &[R2MapInferenceGroup],
    ) -> Result<Vec<R2MapPredictionGroup>, cascadia_model::R2MapModelError>;

    fn score_r2_map_market_groups(
        &mut self,
        groups: &[R2MapMarketInferenceGroup],
    ) -> Result<Vec<R2MapMarketPredictionGroup>, cascadia_model::R2MapModelError>;
}

/// Deterministic pipeline-smoke predictor that ranks drafts by their exact
/// immediate afterstate score and leaves learned score-to-go at zero.
///
/// This is deliberately not a strength model. It exists so Linux worker images
/// can exercise the complete iterative collector, exploration, replay, and
/// checksum path before a portable learned-model backend is attached.
#[derive(Debug, Default)]
pub struct R2MapExactScoreReferencePredictor;

impl R2MapPredictor for R2MapExactScoreReferencePredictor {
    fn score_r2_map_groups(
        &mut self,
        groups: &[R2MapInferenceGroup],
    ) -> Result<Vec<R2MapPredictionGroup>, cascadia_model::R2MapModelError> {
        Ok(groups
            .iter()
            .map(|group| {
                let action_scores = group
                    .candidates
                    .iter()
                    .map(|candidate| candidate.exact_afterstate_score)
                    .collect::<Vec<_>>();
                R2MapPredictionGroup {
                    group_id: group.group_id,
                    decision_id: group.decision_id,
                    action_ids: group
                        .candidates
                        .iter()
                        .map(|candidate| candidate.action_id)
                        .collect(),
                    predicted_score_to_go: vec![0.0; action_scores.len()],
                    predicted_score_components_to_go: vec![[0.0; 11]; action_scores.len()],
                    bootstrap_policy_logits: vec![0.0; action_scores.len()],
                    action_scores,
                }
            })
            .collect())
    }

    fn score_r2_map_market_groups(
        &mut self,
        groups: &[R2MapMarketInferenceGroup],
    ) -> Result<Vec<R2MapMarketPredictionGroup>, cascadia_model::R2MapModelError> {
        Ok(groups
            .iter()
            .map(|group| R2MapMarketPredictionGroup {
                group_id: group.group_id,
                decision_id: group.decision_id,
                action_ids: group
                    .candidates
                    .iter()
                    .map(|candidate| candidate.action_id)
                    .collect(),
                action_scores: vec![0.0; group.candidates.len()],
                predicted_score_to_go: vec![0.0; group.candidates.len()],
            })
            .collect())
    }
}

impl R2MapPredictor for R2MapModelProcess {
    fn score_r2_map_groups(
        &mut self,
        groups: &[R2MapInferenceGroup],
    ) -> Result<Vec<R2MapPredictionGroup>, cascadia_model::R2MapModelError> {
        self.score_groups(groups)
    }

    fn score_r2_map_market_groups(
        &mut self,
        groups: &[R2MapMarketInferenceGroup],
    ) -> Result<Vec<R2MapMarketPredictionGroup>, cascadia_model::R2MapModelError> {
        self.score_market_groups(groups)
    }
}

pub fn prepare_r2_map_draft_decision(
    session: &MarketDecisionSession,
    game_index: u64,
    model: R2MapModelIdentity,
    transform: D6Transform,
) -> Result<R2MapPreparedDecision, SearchError> {
    let game = session.staged_game();
    if session.stage() != MarketDecisionStage::Draft {
        return Err(SearchError::InvalidConfig(
            "R2-MAP draft preparation requires a completed market session",
        ));
    }
    let materializer = R2MapIncrementalMaterializer::new(game, game_index, transform)?;
    let cards = game.config().scoring_cards;
    let active_board = &game.boards()[game.current_player()];
    let baseline = score_board(active_board, cards);
    let habitat = active_board.habitat_analysis();
    let mut wildlife_score_cache = HashMap::new();
    let evaluated = game.evaluate_legal_turn_actions_with_tile_context(
        &MarketPrelude::default(),
        |board, placement, tile| {
            let after_tile = rescore_after_tile_with_habitat_analysis(
                board, cards, baseline, &habitat, placement, tile,
            );
            materializer
                .capture_tile_board(board)
                .map(|context| (context, after_tile))
        },
        |board, tile_context, placed_wildlife| {
            let (tile_context, after_tile) = tile_context.as_ref().map_err(|error| {
                cascadia_r2::R2Error::DatasetContract(format!(
                    "incremental tile context failed: {error}"
                ))
            })?;
            let delta =
                materializer.capture_wildlife_sibling(board, tile_context, placed_wildlife)?;
            let exact_afterstate_score = placed_wildlife.map_or(*after_tile, |value| {
                let wildlife_scores = *wildlife_score_cache.entry(value).or_insert_with(|| {
                    rescore_after_wildlife_placement(board, cards, *after_tile, value.0).wildlife
                });
                rescore_with_wildlife_scores(board, *after_tile, wildlife_scores)
            });
            Ok::<_, cascadia_r2::R2Error>((delta, exact_afterstate_score))
        },
    )?;
    if evaluated.is_empty() {
        return Err(SearchError::NoLegalActions);
    }
    let action_encoder = R2MapActionEncoder::new(game, transform)?;
    // Indexed parallel iteration preserves canonical action order. The
    // expensive spatial work was captured during one apply/undo enumeration;
    // these workers only expand cached board-local rows and metadata.
    let finalized = evaluated
        .into_par_iter()
        .map(|(action, captured)| {
            let (delta, exact_afterstate_score) = captured?;
            let action_id = r2_map_draft_action_id(&action)?;
            let candidate = R2MapInferenceCandidate {
                action_id,
                afterstate: materializer.materialize_afterstate(&delta, &action)?,
                action_bytes: action_encoder
                    .encode_staged_after_score(&action, exact_afterstate_score)?,
                exact_afterstate_score: f32::from(exact_afterstate_score.base_total),
            };
            Ok::<_, SearchError>((action, candidate))
        })
        .collect::<Result<Vec<_>, SearchError>>()?;
    let (actions, candidates) = finalized.into_iter().unzip();
    finish_prepared_draft_decision(
        game,
        actions,
        materializer.parent_tensors().clone(),
        candidates,
        model,
    )
}

/// Previous no-pruning Rayon/cache implementation retained as an independent
/// production-shaped oracle for the P1 three-way parity panel.
#[cfg(test)]
fn prepare_r2_map_draft_decision_rayon_cache(
    session: &MarketDecisionSession,
    game_index: u64,
    model: R2MapModelIdentity,
    transform: D6Transform,
) -> Result<R2MapPreparedDecision, SearchError> {
    let game = session.staged_game();
    let actions = session.legal_draft_actions()?;
    if actions.is_empty() {
        return Err(SearchError::NoLegalActions);
    }
    let perspective_seat = game.current_player();
    let parent = encode_r2_map_public_tensors(
        &game.public_state(),
        game_index,
        perspective_seat,
        transform,
        false,
    )?;
    let action_encoder = R2MapActionEncoder::new(game, transform)?;
    let candidates = actions
        .par_iter()
        .map(|action| {
            materialize_r2_map_candidate(
                game,
                action,
                game_index,
                perspective_seat,
                transform,
                &action_encoder,
            )
        })
        .collect::<Result<Vec<_>, SearchError>>()?;
    finish_prepared_draft_decision(game, actions, parent, candidates, model)
}

#[cfg(test)]
fn materialize_r2_map_candidate(
    game: &GameState,
    action: &TurnAction,
    game_index: u64,
    perspective_seat: usize,
    transform: D6Transform,
    action_encoder: &R2MapActionEncoder<'_>,
) -> Result<R2MapInferenceCandidate, SearchError> {
    materialize_r2_map_candidate_with_action_bytes(
        game,
        action,
        game_index,
        perspective_seat,
        transform,
        action_encoder.encode(action)?,
    )
}

#[cfg(test)]
fn materialize_r2_map_candidate_with_action_bytes(
    game: &GameState,
    action: &TurnAction,
    game_index: u64,
    perspective_seat: usize,
    transform: D6Transform,
    action_bytes: [u8; cascadia_r2::R2_MAP_ACTION_BYTES],
) -> Result<R2MapInferenceCandidate, SearchError> {
    let action_id = r2_map_draft_action_id(action)?;
    let afterstate = game.preview_public_afterstate(action)?;
    let exact_afterstate_score = f32::from(
        score_board(
            &afterstate.boards()[perspective_seat],
            game.config().scoring_cards,
        )
        .base_total,
    );
    Ok(R2MapInferenceCandidate {
        action_id,
        afterstate: encode_r2_map_public_tensors(
            &afterstate,
            game_index,
            perspective_seat,
            transform,
            true,
        )?,
        action_bytes,
        exact_afterstate_score,
    })
}

fn finish_prepared_draft_decision(
    game: &GameState,
    actions: Vec<TurnAction>,
    parent: cascadia_r2::R2MapPublicTensors,
    candidates: Vec<R2MapInferenceCandidate>,
    model: R2MapModelIdentity,
) -> Result<R2MapPreparedDecision, SearchError> {
    let decision_id = *game.public_state().canonical_hash().as_bytes();
    let mut hasher = Hasher::new();
    hasher.update(b"r2-map-inference-group-v1");
    hasher.update(&decision_id);
    hasher.update(model.model_weights_blake3.as_bytes());
    for candidate in &candidates {
        hasher.update(&candidate.action_id);
    }
    Ok(R2MapPreparedDecision {
        actions,
        request: R2MapInferenceGroup {
            group_id: *hasher.finalize().as_bytes(),
            decision_id,
            model,
            parent,
            candidates,
        },
    })
}

#[cfg(test)]
fn prepare_r2_map_draft_decision_reference(
    session: &MarketDecisionSession,
    game_index: u64,
    model: R2MapModelIdentity,
    transform: D6Transform,
) -> Result<R2MapPreparedDecision, SearchError> {
    let game = session.staged_game();
    let actions = session.legal_draft_actions()?;
    if actions.is_empty() {
        return Err(SearchError::NoLegalActions);
    }
    let perspective_seat = game.current_player();
    let parent = encode_r2_map_public_tensors(
        &game.public_state(),
        game_index,
        perspective_seat,
        transform,
        false,
    )?;
    let candidates = actions
        .iter()
        .map(|action| {
            materialize_r2_map_candidate_with_action_bytes(
                game,
                action,
                game_index,
                perspective_seat,
                transform,
                encode_r2_map_action_bytes(game, action, transform)?,
            )
        })
        .collect::<Result<Vec<_>, SearchError>>()?;
    finish_prepared_draft_decision(game, actions, parent, candidates, model)
}

pub fn prepare_r2_map_market_decision(
    session: &MarketDecisionSession,
    game_index: u64,
    turn_index: u16,
    ordinal: u8,
    model: R2MapModelIdentity,
    transform: D6Transform,
) -> Result<R2MapPreparedMarketDecision, SearchError> {
    let stage = session.stage();
    let decision_kind = match stage {
        MarketDecisionStage::FreeThreeOfAKind => R2MapMarketDecisionKind::FreeThreeOfAKind,
        MarketDecisionStage::PaidWipes => R2MapMarketDecisionKind::PaidWipes,
        MarketDecisionStage::Draft => {
            return Err(SearchError::InvalidConfig(
                "market decision requested after Stop",
            ));
        }
    };
    let decisions = session.legal_decisions();
    if decisions.is_empty() {
        return Err(SearchError::NoLegalActions);
    }
    let game = session.staged_game();
    let perspective_seat = game.current_player();
    let public = session.public_state();
    let parent_public_hash = *public.canonical_hash().as_bytes();
    let decision_id =
        public_market_decision_identity(parent_public_hash, turn_index, ordinal, stage);
    let candidates = decisions
        .iter()
        .map(|decision| {
            let action_bytes = encode_r2_map_market_action_bytes(stage, decision)?;
            Ok(R2MapMarketInferenceCandidate {
                action_id: public_market_action_identity(decision_id, action_bytes),
                action_bytes,
            })
        })
        .collect::<Result<Vec<_>, cascadia_r2::R2Error>>()?;
    let mut group_hasher = Hasher::new();
    group_hasher.update(b"r2-map-market-inference-group-v1");
    group_hasher.update(&decision_id);
    group_hasher.update(model.model_weights_blake3.as_bytes());
    for candidate in &candidates {
        group_hasher.update(&candidate.action_id);
    }
    let public_wildlife_bag_counts = game.public_supply().wildlife_bag;
    let public_wildlife_bag_total = public_wildlife_bag_counts
        .into_iter()
        .try_fold(0u8, u8::checked_add)
        .ok_or(SearchError::InvalidConfig(
            "public wildlife bag total exceeds u8",
        ))?;
    let public_market_wildlife = game
        .market()
        .wildlife
        .map(|wildlife| wildlife.expect("active market is complete") as u8);
    Ok(R2MapPreparedMarketDecision {
        decisions,
        request: R2MapMarketInferenceGroup {
            group_id: *group_hasher.finalize().as_bytes(),
            decision_id,
            model,
            parent: encode_r2_map_public_tensors(
                &public,
                game_index,
                perspective_seat,
                transform,
                false,
            )?,
            exact_current_score: f32::from(
                score_board(
                    &game.boards()[perspective_seat],
                    game.config().scoring_cards,
                )
                .base_total,
            ),
            decision_kind,
            public_nature_tokens: game.boards()[perspective_seat].nature_tokens(),
            public_wildlife_bag_counts,
            public_wildlife_bag_total,
            public_market_wildlife,
            candidates,
        },
    })
}

pub fn score_r2_map_decision(
    predictor: &mut impl R2MapPredictor,
    prepared: R2MapPreparedDecision,
    exploration: Option<R2MapExplorationChoice>,
) -> Result<R2MapScoredDecision, SearchError> {
    let mut predictions = predictor.score_r2_map_groups(std::slice::from_ref(&prepared.request))?;
    if predictions.len() != 1 {
        return Err(SearchError::PredictionCount {
            expected: 1,
            actual: predictions.len(),
        });
    }
    let prediction = predictions.pop().expect("one prediction was checked");
    let expected_ids = prepared
        .request
        .candidates
        .iter()
        .map(|candidate| candidate.action_id)
        .collect::<Vec<_>>();
    if prediction.group_id != prepared.request.group_id
        || prediction.decision_id != prepared.request.decision_id
        || prediction.action_ids != expected_ids
        || prediction.action_scores.len() != prepared.actions.len()
    {
        return Err(SearchError::PredictionCount {
            expected: prepared.actions.len(),
            actual: prediction.action_scores.len(),
        });
    }
    let (selected_index, explored) = if let Some(choice) = exploration {
        select_with_exploration(
            &prediction.action_scores,
            choice.validate(prediction.action_scores.len())?,
        )?
    } else {
        (argmax_first(&prediction.action_scores)?, false)
    };
    Ok(R2MapScoredDecision {
        action: prepared.actions[selected_index].clone(),
        selected_index,
        explored,
        action_ids: expected_ids,
        scores: prediction.action_scores,
    })
}

pub fn score_r2_map_market_decision(
    predictor: &mut impl R2MapPredictor,
    prepared: R2MapPreparedMarketDecision,
    exploration: Option<R2MapExplorationChoice>,
) -> Result<R2MapScoredMarketDecision, SearchError> {
    let mut predictions =
        predictor.score_r2_map_market_groups(std::slice::from_ref(&prepared.request))?;
    if predictions.len() != 1 {
        return Err(SearchError::PredictionCount {
            expected: 1,
            actual: predictions.len(),
        });
    }
    let prediction = predictions
        .pop()
        .expect("one market prediction was checked");
    let expected_ids = prepared
        .request
        .candidates
        .iter()
        .map(|candidate| candidate.action_id)
        .collect::<Vec<_>>();
    if prediction.group_id != prepared.request.group_id
        || prediction.decision_id != prepared.request.decision_id
        || prediction.action_ids != expected_ids
        || prediction.action_scores.len() != prepared.decisions.len()
    {
        return Err(SearchError::PredictionCount {
            expected: prepared.decisions.len(),
            actual: prediction.action_scores.len(),
        });
    }
    let (selected_index, explored) = if let Some(choice) = exploration {
        select_with_exploration(
            &prediction.action_scores,
            choice.validate(prediction.action_scores.len())?,
        )?
    } else {
        (argmax_first(&prediction.action_scores)?, false)
    };
    Ok(R2MapScoredMarketDecision {
        decision: prepared.decisions[selected_index].clone(),
        selected_index,
        explored,
        action_ids: expected_ids,
        scores: prediction.action_scores,
    })
}

pub fn select_r2_map_turn(
    predictor: &mut impl R2MapPredictor,
    game: &GameState,
    game_index: u64,
    model: R2MapModelIdentity,
    mut exploration_for: impl FnMut(
        &R2MapTurnDecisionContext,
    ) -> Result<Option<R2MapExplorationChoice>, SearchError>,
) -> Result<R2MapScoredTurn, SearchError> {
    select_r2_map_turn_with_preparer(
        predictor,
        game,
        game_index,
        model,
        &mut exploration_for,
        prepare_r2_map_draft_decision,
    )
}

fn select_r2_map_turn_with_preparer(
    predictor: &mut impl R2MapPredictor,
    game: &GameState,
    game_index: u64,
    model: R2MapModelIdentity,
    exploration_for: &mut impl FnMut(
        &R2MapTurnDecisionContext,
    ) -> Result<Option<R2MapExplorationChoice>, SearchError>,
    mut prepare_draft: impl FnMut(
        &MarketDecisionSession,
        u64,
        R2MapModelIdentity,
        D6Transform,
    ) -> Result<R2MapPreparedDecision, SearchError>,
) -> Result<R2MapScoredTurn, SearchError> {
    let turn_index = game.completed_turns();
    let mut session = MarketDecisionSession::begin(game)?;
    let mut market_decisions = Vec::new();
    let mut ordinal = 0u8;
    while session.stage() != MarketDecisionStage::Draft {
        let stage = session.stage();
        let parent_public_hash = *session.public_state().canonical_hash().as_bytes();
        let prepared = prepare_r2_map_market_decision(
            &session,
            game_index,
            turn_index,
            ordinal,
            model.clone(),
            D6Transform::IDENTITY,
        )?;
        let context = R2MapTurnDecisionContext {
            kind: R2MapTurnDecisionKind::Market(stage),
            ordinal,
            decision_id: prepared.request.decision_id,
            parent_public_hash,
            action_ids: prepared
                .request
                .candidates
                .iter()
                .map(|candidate| candidate.action_id)
                .collect(),
        };
        let selected =
            score_r2_map_market_decision(predictor, prepared, exploration_for(&context)?)?;
        session.commit(&selected.decision)?;
        market_decisions.push(R2MapMarketDecisionTrace {
            context,
            selected_index: selected.selected_index,
            selected: selected.decision,
            resulting_public_hash: *session.public_state().canonical_hash().as_bytes(),
            explored: selected.explored,
            scores: selected.scores,
        });
        ordinal = ordinal
            .checked_add(1)
            .ok_or(SearchError::InvalidConfig("too many market decisions"))?;
    }
    let parent_public_hash = *session.public_state().canonical_hash().as_bytes();
    let prepared = prepare_draft(&session, game_index, model, D6Transform::IDENTITY)?;
    let draft_context = R2MapTurnDecisionContext {
        kind: R2MapTurnDecisionKind::Draft,
        ordinal,
        decision_id: prepared.request.decision_id,
        parent_public_hash,
        action_ids: prepared
            .request
            .candidates
            .iter()
            .map(|candidate| candidate.action_id)
            .collect(),
    };
    let draft = score_r2_map_decision(predictor, prepared, exploration_for(&draft_context)?)?;
    let action = session.bundle_action(&draft.action)?;
    if game.transition(&action)? != session.staged_game().transition(&draft.action)? {
        return Err(SearchError::InvalidConfig(
            "bundled market and staged draft transitions differ",
        ));
    }
    Ok(R2MapScoredTurn {
        bundled_action_id: r2_map_draft_action_id(&action)?,
        action,
        market_decisions,
        draft_context,
        draft,
    })
}

#[cfg(test)]
pub(crate) fn select_r2_map_turn_with_reference_parity(
    predictor: &mut impl R2MapPredictor,
    game: &GameState,
    game_index: u64,
    model: R2MapModelIdentity,
    mut exploration_for: impl FnMut(
        &R2MapTurnDecisionContext,
    ) -> Result<Option<R2MapExplorationChoice>, SearchError>,
) -> Result<R2MapScoredTurn, SearchError> {
    select_r2_map_turn_with_preparer(
        predictor,
        game,
        game_index,
        model,
        &mut exploration_for,
        |session, game_index, model, transform| {
            let reference_started = Instant::now();
            let reference = prepare_r2_map_draft_decision_reference(
                session,
                game_index,
                model.clone(),
                transform,
            )?;
            REFERENCE_PREPARATION_NANOS.fetch_add(
                u64::try_from(reference_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );
            let rayon_cache_started = Instant::now();
            let rayon_cache = prepare_r2_map_draft_decision_rayon_cache(
                session,
                game_index,
                model.clone(),
                transform,
            )?;
            RAYON_CACHE_PREPARATION_NANOS.fetch_add(
                u64::try_from(rayon_cache_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );
            let optimized_started = Instant::now();
            let optimized = prepare_r2_map_draft_decision(session, game_index, model, transform)?;
            OPTIMIZED_PREPARATION_NANOS.fetch_add(
                u64::try_from(optimized_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );
            if reference != rayon_cache || reference != optimized {
                return Err(SearchError::InvalidConfig(
                    "Rayon/cache or incremental draft preparation differs from the sequential oracle",
                ));
            }
            Ok(optimized)
        },
    )
}

pub fn select_r2_map_argmax(
    predictor: &mut impl R2MapPredictor,
    game: &GameState,
    game_index: u64,
    model: R2MapModelIdentity,
) -> Result<R2MapScoredTurn, SearchError> {
    select_r2_map_turn(predictor, game, game_index, model, |_| Ok(None))
}

fn argmax_first(scores: &[f32]) -> Result<usize, SearchError> {
    let mut best = None;
    for (index, score) in scores.iter().copied().enumerate() {
        if !score.is_finite() {
            return Err(SearchError::NonFinitePrediction { index });
        }
        if best.is_none_or(|(_, best_score)| score > best_score) {
            best = Some((index, score));
        }
    }
    best.map(|(index, _)| index)
        .ok_or(SearchError::NoLegalActions)
}

fn select_with_exploration(
    scores: &[f32],
    choice: R2MapExplorationChoice,
) -> Result<(usize, bool), SearchError> {
    let greedy = argmax_first(scores)?;
    if u128::from(choice.gate_draw_u64) * 1_000_000
        >= u128::from(u64::MAX) * u128::from(choice.epsilon_parts_per_million)
    {
        return Ok((greedy, false));
    }
    let temperature = f64::from(choice.temperature_parts_per_million) / 1_000_000.0;
    let mut best = None;
    for (index, (&score, &draw)) in scores.iter().zip(&choice.action_draws_u64).enumerate() {
        let uniform = (draw as f64 + 0.5) / (u64::MAX as f64 + 1.0);
        let perturbed = f64::from(score) / temperature - (-uniform.ln()).ln();
        if best.is_none_or(|(_, best_score)| perturbed > best_score) {
            best = Some((index, perturbed));
        }
    }
    Ok((best.expect("nonempty scores were checked").0, true))
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameConfig, GameSeed, GameState, Replay, ScoreBreakdown, score_game};

    use super::*;

    fn model() -> R2MapModelIdentity {
        R2MapModelIdentity {
            checkpoint_id: "checkpoint-test".into(),
            checkpoint_manifest_blake3: "1".repeat(64),
            model_config_blake3: "2".repeat(64),
            model_weights_blake3: "3".repeat(64),
            verification_id: "4".repeat(64),
        }
    }

    struct ExactFakePredictor;

    impl R2MapPredictor for ExactFakePredictor {
        fn score_r2_map_groups(
            &mut self,
            groups: &[R2MapInferenceGroup],
        ) -> Result<Vec<R2MapPredictionGroup>, cascadia_model::R2MapModelError> {
            Ok(groups
                .iter()
                .map(|group| {
                    let scores = group
                        .candidates
                        .iter()
                        .map(|candidate| candidate.exact_afterstate_score)
                        .collect::<Vec<_>>();
                    R2MapPredictionGroup {
                        group_id: group.group_id,
                        decision_id: group.decision_id,
                        action_ids: group
                            .candidates
                            .iter()
                            .map(|candidate| candidate.action_id)
                            .collect(),
                        predicted_score_to_go: vec![0.0; scores.len()],
                        predicted_score_components_to_go: vec![[0.0; 11]; scores.len()],
                        bootstrap_policy_logits: vec![0.0; scores.len()],
                        action_scores: scores,
                    }
                })
                .collect())
        }

        fn score_r2_map_market_groups(
            &mut self,
            groups: &[R2MapMarketInferenceGroup],
        ) -> Result<Vec<R2MapMarketPredictionGroup>, cascadia_model::R2MapModelError> {
            Ok(groups
                .iter()
                .map(|group| R2MapMarketPredictionGroup {
                    group_id: group.group_id,
                    decision_id: group.decision_id,
                    action_ids: group
                        .candidates
                        .iter()
                        .map(|candidate| candidate.action_id)
                        .collect(),
                    action_scores: vec![0.0; group.candidates.len()],
                    predicted_score_to_go: vec![0.0; group.candidates.len()],
                })
                .collect())
        }
    }

    struct TensorFingerprintPredictor;

    impl R2MapPredictor for TensorFingerprintPredictor {
        fn score_r2_map_groups(
            &mut self,
            groups: &[R2MapInferenceGroup],
        ) -> Result<Vec<R2MapPredictionGroup>, cascadia_model::R2MapModelError> {
            Ok(groups
                .iter()
                .map(|group| {
                    let action_scores = group
                        .candidates
                        .par_iter()
                        .map(|candidate| {
                            let tensors = &candidate.afterstate;
                            let token_term = tensors
                                .token_features
                                .iter()
                                .enumerate()
                                .filter(|(index, _)| index % 97 == 0)
                                .map(|(index, value)| f64::from(*value) * ((index % 17 + 1) as f64))
                                .sum::<f64>();
                            let context_term = tensors
                                .market_features
                                .iter()
                                .chain(&tensors.player_features)
                                .chain(&tensors.global_features)
                                .enumerate()
                                .map(|(index, value)| f64::from(*value) * ((index % 13 + 1) as f64))
                                .sum::<f64>();
                            let action_term = candidate
                                .action_bytes
                                .iter()
                                .enumerate()
                                .map(|(index, value)| f64::from(*value) * ((index % 7 + 1) as f64))
                                .sum::<f64>();
                            f64::from(candidate.exact_afterstate_score)
                                + token_term * 0.000_001
                                + context_term * 0.000_01
                                + action_term * 0.000_001
                        })
                        .map(|score| score as f32)
                        .collect::<Vec<_>>();
                    R2MapPredictionGroup {
                        group_id: group.group_id,
                        decision_id: group.decision_id,
                        action_ids: group
                            .candidates
                            .iter()
                            .map(|candidate| candidate.action_id)
                            .collect(),
                        predicted_score_to_go: vec![0.0; action_scores.len()],
                        predicted_score_components_to_go: vec![[0.0; 11]; action_scores.len()],
                        bootstrap_policy_logits: vec![0.0; action_scores.len()],
                        action_scores,
                    }
                })
                .collect())
        }

        fn score_r2_map_market_groups(
            &mut self,
            groups: &[R2MapMarketInferenceGroup],
        ) -> Result<Vec<R2MapMarketPredictionGroup>, cascadia_model::R2MapModelError> {
            ExactFakePredictor.score_r2_map_market_groups(groups)
        }
    }

    fn stopped_session(game: &GameState) -> MarketDecisionSession {
        let mut session = MarketDecisionSession::begin(game).unwrap();
        if session.stage() == MarketDecisionStage::FreeThreeOfAKind {
            session.commit(&MarketDecision::KeepThreeOfAKind).unwrap();
        }
        session.commit(&MarketDecision::StopWiping).unwrap();
        session
    }

    fn hash_public_tensors(hasher: &mut Hasher, tensors: &cascadia_r2::R2MapPublicTensors) {
        for value in &tensors.token_features {
            hasher.update(&value.to_bits().to_le_bytes());
        }
        for value in &tensors.token_types {
            hasher.update(&value.to_le_bytes());
        }
        hasher.update(&tensors.token_mask);
        for value in &tensors.market_features {
            hasher.update(&value.to_bits().to_le_bytes());
        }
        hasher.update(&tensors.market_mask);
        for value in &tensors.player_features {
            hasher.update(&value.to_bits().to_le_bytes());
        }
        hasher.update(&tensors.player_mask);
        for value in tensors.global_features {
            hasher.update(&value.to_bits().to_le_bytes());
        }
    }

    fn prepared_tensor_fingerprint(prepared: &R2MapPreparedDecision) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(b"r2-map-p1-prepared-tensor-fingerprint-v1");
        hasher.update(&prepared.request.group_id);
        hasher.update(&prepared.request.decision_id);
        hash_public_tensors(&mut hasher, &prepared.request.parent);
        let candidate_hashes = prepared
            .request
            .candidates
            .par_iter()
            .map(|candidate| {
                let mut candidate_hasher = Hasher::new();
                candidate_hasher.update(b"r2-map-p1-candidate-tensor-fingerprint-v1");
                candidate_hasher.update(&candidate.action_id);
                candidate_hasher.update(&candidate.action_bytes);
                candidate_hasher.update(&candidate.exact_afterstate_score.to_bits().to_le_bytes());
                hash_public_tensors(&mut candidate_hasher, &candidate.afterstate);
                *candidate_hasher.finalize().as_bytes()
            })
            .collect::<Vec<_>>();
        for candidate_hash in candidate_hashes {
            hasher.update(&candidate_hash);
        }
        *hasher.finalize().as_bytes()
    }

    #[derive(Clone, Copy)]
    #[repr(u8)]
    enum P1SampleBin {
        Early = 0,
        Middle = 1,
        Late = 2,
        MaximumWidth = 3,
    }

    #[derive(Clone, Copy)]
    struct P1SampleIdentityInput {
        game_offset: u64,
        seed: GameSeed,
        game_index: u64,
        turn: u64,
        parent_hash: [u8; 32],
        transform: D6Transform,
        width: usize,
        bin: P1SampleBin,
    }

    impl P1SampleIdentityInput {
        fn identity_hash(self) -> [u8; 32] {
            let mut hasher = Hasher::new();
            hasher.update(b"r2-map-p1-open-sample-identity-v3");
            hasher.update(&self.game_offset.to_le_bytes());
            hasher.update(&self.seed.0);
            hasher.update(&self.game_index.to_le_bytes());
            hasher.update(&self.turn.to_le_bytes());
            hasher.update(&self.parent_hash);
            hasher.update(&[self.transform.id()]);
            hasher.update(&(self.width as u64).to_le_bytes());
            hasher.update(&[self.bin as u8]);
            *hasher.finalize().as_bytes()
        }
    }

    fn update_score_digest(hasher: &mut Hasher, score: &ScoreBreakdown) {
        for value in score.habitat {
            hasher.update(&value.to_le_bytes());
        }
        for value in score.wildlife {
            hasher.update(&value.to_le_bytes());
        }
        hasher.update(&score.nature_tokens.to_le_bytes());
        hasher.update(&score.habitat_bonus);
        hasher.update(&score.base_total.to_le_bytes());
        hasher.update(&score.total.to_le_bytes());
    }

    fn update_p1_corpus_game_header(
        hasher: &mut Hasher,
        game_offset: u64,
        seed: GameSeed,
        action_count: u64,
    ) {
        hasher.update(&game_offset.to_le_bytes());
        hasher.update(&seed.0);
        hasher.update(&action_count.to_le_bytes());
    }

    fn update_p1_corpus_totals(hasher: &mut Hasher, games: u64, actions: u64) {
        hasher.update(b"r2-map-p1-open-corpus-total-v1");
        hasher.update(&games.to_le_bytes());
        hasher.update(&actions.to_le_bytes());
    }

    fn p1_count_binding_test_digest(per_game_actions: u64, total_actions: u64) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(b"r2-map-p1-open-corpus-v2");
        update_p1_corpus_game_header(&mut hasher, 0, GameSeed::from_u64(7), per_game_actions);
        update_p1_corpus_totals(&mut hasher, 1, total_actions);
        *hasher.finalize().as_bytes()
    }

    #[test]
    fn p1_corpus_digest_binds_per_game_and_total_action_counts() {
        let baseline = p1_count_binding_test_digest(100, 100);
        assert_ne!(baseline, p1_count_binding_test_digest(101, 100));
        assert_ne!(baseline, p1_count_binding_test_digest(100, 101));
    }

    #[test]
    fn p1_sample_identity_binds_seed_game_index_transform_width_and_bin() {
        let seed = GameSeed::from_u64(7);
        let parent = [9; 32];
        let baseline_input = P1SampleIdentityInput {
            game_offset: 1,
            seed,
            game_index: 11,
            turn: 40,
            parent_hash: parent,
            transform: D6Transform::IDENTITY,
            width: 123,
            bin: P1SampleBin::Middle,
        };
        let baseline = baseline_input.identity_hash();
        assert_eq!(
            baseline,
            [
                0x8e, 0x56, 0xad, 0xb0, 0xd2, 0xa7, 0x30, 0x28, 0xd9, 0x50, 0xb4, 0xc5, 0x86, 0xbe,
                0x41, 0x27, 0x70, 0xd4, 0x77, 0x2c, 0xea, 0xd7, 0xb8, 0xf4, 0x6c, 0xb9, 0x88, 0xa5,
                0x95, 0x17, 0xdf, 0x9e,
            ]
        );
        for changed in [
            P1SampleIdentityInput {
                game_offset: 2,
                ..baseline_input
            }
            .identity_hash(),
            P1SampleIdentityInput {
                seed: GameSeed::from_u64(8),
                ..baseline_input
            }
            .identity_hash(),
            P1SampleIdentityInput {
                game_index: 12,
                ..baseline_input
            }
            .identity_hash(),
            P1SampleIdentityInput {
                turn: 41,
                ..baseline_input
            }
            .identity_hash(),
            P1SampleIdentityInput {
                parent_hash: [10; 32],
                ..baseline_input
            }
            .identity_hash(),
            P1SampleIdentityInput {
                transform: D6Transform::ALL[1],
                ..baseline_input
            }
            .identity_hash(),
            P1SampleIdentityInput {
                width: 124,
                ..baseline_input
            }
            .identity_hash(),
            P1SampleIdentityInput {
                bin: P1SampleBin::MaximumWidth,
                ..baseline_input
            }
            .identity_hash(),
        ] {
            assert_ne!(baseline, changed);
        }
    }

    #[test]
    fn argmax_ties_choose_first_enumeration_index() {
        assert_eq!(argmax_first(&[2.0, 3.0, 3.0]).unwrap(), 1);
    }

    #[test]
    fn exploration_is_counter_deterministic_and_only_after_scores_exist() {
        let choice = R2MapExplorationChoice {
            gate_draw_u64: 11,
            epsilon_parts_per_million: 1_000_000,
            temperature_parts_per_million: 1_000_000,
            action_draws_u64: vec![1, 2, 3],
        };
        let left = select_with_exploration(&[1.0, 2.0, 3.0], choice.clone()).unwrap();
        let right = select_with_exploration(&[1.0, 2.0, 3.0], choice).unwrap();
        assert_eq!(left, right);
        assert!(left.1);
    }

    #[test]
    fn prepared_decision_contains_every_legal_action_once_and_selects_after_scoring() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(9001),
        )
        .unwrap();
        let session = stopped_session(&game);
        let legal = session.legal_draft_actions().unwrap();
        let prepared =
            prepare_r2_map_draft_decision(&session, 7, model(), D6Transform::IDENTITY).unwrap();
        assert_eq!(prepared.actions, legal);
        assert_eq!(prepared.request.candidates.len(), legal.len());
        let unique = prepared
            .request
            .candidates
            .iter()
            .map(|candidate| candidate.action_id)
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(unique.len(), legal.len());
        let selected = score_r2_map_decision(&mut ExactFakePredictor, prepared, None).unwrap();
        assert_eq!(selected.scores.len(), legal.len());
        assert_eq!(selected.action_ids.len(), legal.len());
    }

    fn assert_staged_action_encoding_parity_for_all_d6(game: &GameState) -> usize {
        let actions = game.legal_turn_actions(&MarketPrelude::default()).unwrap();
        let cards = game.config().scoring_cards;
        let after_scores = actions
            .par_iter()
            .map(|action| score_board(&game.preview_active_board(action).unwrap(), cards))
            .collect::<Vec<_>>();
        for transform in D6Transform::ALL {
            let encoder = R2MapActionEncoder::new(game, transform).unwrap();
            actions
                .par_iter()
                .zip(&after_scores)
                .for_each(|(action, after)| {
                    assert_eq!(
                        encoder.encode_staged_after_score(action, *after).unwrap(),
                        encoder.encode(action).unwrap()
                    );
                });
        }
        actions.len() * D6Transform::ALL.len()
    }

    fn free_replacement_staged_games() -> (GameState, GameState) {
        let game = public_market_variant_fixture();
        let session = MarketDecisionSession::begin(&game).unwrap();
        assert_eq!(session.stage(), MarketDecisionStage::FreeThreeOfAKind);
        assert!(
            session
                .legal_decisions()
                .contains(&MarketDecision::ReplaceThreeOfAKind)
        );
        let mut kept = session.clone();
        kept.commit(&MarketDecision::KeepThreeOfAKind).unwrap();
        kept.commit(&MarketDecision::StopWiping).unwrap();
        let mut replaced = session;
        replaced
            .commit(&MarketDecision::ReplaceThreeOfAKind)
            .unwrap();
        replaced.commit(&MarketDecision::StopWiping).unwrap();
        (kept.staged_game().clone(), replaced.staged_game().clone())
    }

    fn public_market_variant_fixture() -> GameState {
        // Build the same conservation-respecting public fixture as the game
        // crate's exhaustive market tests through the canonical JSON schema,
        // then require the full simulator validator before using it.
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(10_106),
        )
        .unwrap();
        let mut value = serde_json::to_value(game).unwrap();
        let root = value.as_object_mut().unwrap();
        let mut bag = std::mem::take(root["wildlife_bag"].as_array_mut().unwrap());
        let market = root["market"]["wildlife"].as_array_mut().unwrap();
        for slot in market.iter_mut() {
            bag.push(slot.take());
        }
        let desired = [
            cascadia_game::Wildlife::Bear,
            cascadia_game::Wildlife::Bear,
            cascadia_game::Wildlife::Bear,
            cascadia_game::Wildlife::Elk,
        ]
        .map(|wildlife| serde_json::to_value(wildlife).unwrap());
        for (slot, wildlife) in market.iter_mut().zip(desired) {
            let index = bag
                .iter()
                .position(|candidate| *candidate == wildlife)
                .unwrap();
            *slot = bag.swap_remove(index);
        }
        let discarded = root["discarded_wildlife"].as_array_mut().unwrap();
        discarded.append(&mut bag);
        for wildlife in cascadia_game::Wildlife::ALL {
            let encoded = serde_json::to_value(wildlife).unwrap();
            for _ in 0..2 {
                let index = discarded
                    .iter()
                    .position(|candidate| *candidate == encoded)
                    .unwrap();
                bag.push(discarded.swap_remove(index));
            }
        }
        root["wildlife_bag"] = serde_json::Value::Array(bag);
        let game: GameState = serde_json::from_value(value).unwrap();
        game.validate().unwrap();
        game
    }

    fn paid_wipe_staged_game() -> GameState {
        let config = GameConfig::research_aaaaa(4).unwrap();
        for raw_seed in 0..256 {
            let game = GameState::new(config, GameSeed::from_u64(raw_seed)).unwrap();
            let mut value = serde_json::to_value(game).unwrap();
            value["boards"][0]["nature_tokens"] = serde_json::json!(1);
            let game: GameState = serde_json::from_value(value).unwrap();
            game.validate().unwrap();
            let mut session = MarketDecisionSession::begin(&game).unwrap();
            if session.stage() == MarketDecisionStage::FreeThreeOfAKind {
                session.commit(&MarketDecision::KeepThreeOfAKind).unwrap();
            }
            if let Some(paid) = session
                .legal_decisions()
                .into_iter()
                .find(|decision| matches!(decision, MarketDecision::PaidWipe(_)))
            {
                session.commit(&paid).unwrap();
                session.commit(&MarketDecision::StopWiping).unwrap();
                return session.staged_game().clone();
            }
        }
        panic!("test seed search did not reach a legal paid-wipe branch");
    }

    #[test]
    fn staged_action_encoder_matches_canonical_for_market_variants_and_all_d6() {
        let stop = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(0x5354_4f50),
        )
        .unwrap();
        let (kept, replaced) = free_replacement_staged_games();
        let paid = paid_wipe_staged_game();
        let checks = [&stop, &kept, &replaced, &paid]
            .into_iter()
            .map(assert_staged_action_encoding_parity_for_all_d6)
            .sum::<usize>();
        assert!(checks > 0);
    }

    #[test]
    fn incremental_and_rayon_cache_match_sequential_oracle_byte_for_byte() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(0x0052_324d_4150),
        )
        .unwrap();
        let session = stopped_session(&game);
        for transform in [D6Transform::IDENTITY, D6Transform::ALL[7]] {
            let reference =
                prepare_r2_map_draft_decision_reference(&session, 17, model(), transform).unwrap();
            let rayon_cache =
                prepare_r2_map_draft_decision_rayon_cache(&session, 17, model(), transform)
                    .unwrap();
            let incremental =
                prepare_r2_map_draft_decision(&session, 17, model(), transform).unwrap();
            assert_eq!(
                incremental.actions.len(),
                session.legal_draft_actions().unwrap().len()
            );
            assert_eq!(rayon_cache, reference);
            assert_eq!(incremental, reference);
        }
    }

    #[test]
    fn first_legal_93_token_replay_matches_authoritative_encoder_for_all_d6() {
        let replay: Replay = serde_json::from_str(include_str!(
            "../../../tests/fixtures/r2_map/p1-v34-first-93-token-replay.json"
        ))
        .unwrap();
        assert_eq!(replay.turns.len(), 78);
        assert_eq!(
            replay.seed.0,
            [
                166, 98, 109, 28, 252, 10, 196, 39, 3, 24, 208, 213, 77, 179, 118, 9, 99, 239, 41,
                118, 105, 190, 150, 216, 218, 161, 205, 235, 121, 222, 142, 230,
            ]
        );
        let game = replay.play().unwrap();
        assert_eq!(game.completed_turns(), 78);
        let session = stopped_session(&game);
        assert_eq!(session.legal_draft_actions().unwrap().len(), 1_176);
        let game_index = 0x5031_4f50_454e_001bu64;

        for transform in D6Transform::ALL {
            let authoritative =
                prepare_r2_map_draft_decision_reference(&session, game_index, model(), transform)
                    .unwrap();
            let incremental =
                prepare_r2_map_draft_decision(&session, game_index, model(), transform).unwrap();
            assert_eq!(incremental, authoritative);
            assert_eq!(incremental.actions.len(), 1_176);
            assert_eq!(incremental.request.candidates.len(), 1_176);
            let first_93_token_candidate = incremental
                .request
                .candidates
                .iter()
                .find(|candidate| {
                    let active_types = &candidate.afterstate.token_types
                        [..cascadia_r2::R2_MAP_BOARD_TOKEN_CAPACITY];
                    [1, 2, 3, 4].map(|token_type| {
                        active_types
                            .iter()
                            .filter(|value| **value == token_type)
                            .count()
                    }) == [23, 30, 20, 20]
                })
                .expect("the captured failure decision must retain its first 93-token candidate");
            assert_eq!(
                first_93_token_candidate.afterstate.token_mask
                    [..cascadia_r2::R2_MAP_BOARD_TOKEN_CAPACITY]
                    .iter()
                    .map(|value| usize::from(*value))
                    .sum::<usize>(),
                93
            );
        }
    }

    #[test]
    #[ignore = "canonical John2 P1 open-corpus parity and throughput gate"]
    fn incremental_open_corpus_exhaustive_reference_legacy() {
        let games = std::env::var("R2_MAP_P1_GAMES")
            .ok()
            .map(|value| value.parse::<u64>().expect("R2_MAP_P1_GAMES is u64"))
            .unwrap_or(100);
        let calibration = std::env::var("R2_MAP_P1_ALLOW_CALIBRATION").as_deref() == Ok("1");
        assert!(
            games >= 100 || calibration,
            "P1 canonical gate requires at least 100 games"
        );
        let first_seed = 0x5031_4f50_454e_0000u64;
        let config = GameConfig::research_aaaaa(4).unwrap();
        let mut turns = 0u64;
        let mut actions = 0u64;
        let mut parent_restore_checks = 0u64;
        let mut prediction_checks = 0u64;
        let mut pinecone_checks = 0u64;
        let mut reference_seconds = 0.0f64;
        let mut rayon_cache_seconds = 0.0f64;
        let mut incremental_seconds = 0.0f64;
        let mut max_prediction_delta = 0.0f32;

        for game_offset in 0..games {
            let seed = GameSeed::from_u64(first_seed + game_offset);
            let mut reference_game = GameState::new(config, seed).unwrap();
            let mut incremental_game = reference_game.clone();
            let mut reference_replay = Replay::new(config, seed);
            let mut incremental_replay = Replay::new(config, seed);
            while !reference_game.is_game_over() {
                assert_eq!(reference_game, incremental_game);
                let mut session = MarketDecisionSession::begin(&reference_game).unwrap();
                if session.stage() == MarketDecisionStage::FreeThreeOfAKind {
                    session.commit(&MarketDecision::KeepThreeOfAKind).unwrap();
                }
                session.commit(&MarketDecision::StopWiping).unwrap();
                let parent_hash = *session.public_state().canonical_hash().as_bytes();
                let transform =
                    D6Transform::ALL[usize::try_from((turns + game_offset) % 12).unwrap()];
                let game_index = first_seed + game_offset;

                let started = Instant::now();
                let reference = prepare_r2_map_draft_decision_reference(
                    &session,
                    game_index,
                    model(),
                    transform,
                )
                .unwrap();
                reference_seconds += started.elapsed().as_secs_f64();
                let started = Instant::now();
                let rayon_cache = prepare_r2_map_draft_decision_rayon_cache(
                    &session,
                    game_index,
                    model(),
                    transform,
                )
                .unwrap();
                rayon_cache_seconds += started.elapsed().as_secs_f64();
                let started = Instant::now();
                let incremental =
                    prepare_r2_map_draft_decision(&session, game_index, model(), transform)
                        .unwrap();
                incremental_seconds += started.elapsed().as_secs_f64();

                let action_count = u64::try_from(reference.actions.len()).unwrap();
                assert_eq!(reference, rayon_cache);
                assert_eq!(reference, incremental);
                actions += action_count;
                parent_restore_checks += action_count;
                assert_eq!(
                    *session.public_state().canonical_hash().as_bytes(),
                    parent_hash,
                    "preparation must restore the exact public parent"
                );

                let reference_scored =
                    score_r2_map_decision(&mut TensorFingerprintPredictor, reference, None)
                        .unwrap();
                let rayon_scored =
                    score_r2_map_decision(&mut TensorFingerprintPredictor, rayon_cache, None)
                        .unwrap();
                let incremental_scored =
                    score_r2_map_decision(&mut TensorFingerprintPredictor, incremental, None)
                        .unwrap();
                for ((reference_score, rayon_score), incremental_score) in reference_scored
                    .scores
                    .iter()
                    .zip(&rayon_scored.scores)
                    .zip(&incremental_scored.scores)
                {
                    let delta = (reference_score - incremental_score)
                        .abs()
                        .max((reference_score - rayon_score).abs());
                    max_prediction_delta = max_prediction_delta.max(delta);
                    assert!(delta <= 1.0e-6);
                    prediction_checks += 1;
                }
                assert_eq!(reference_scored.selected_index, rayon_scored.selected_index);
                assert_eq!(
                    reference_scored.selected_index,
                    incremental_scored.selected_index
                );
                assert_eq!(reference_scored.action, rayon_scored.action);
                assert_eq!(reference_scored.action, incremental_scored.action);

                let reference_action = session.bundle_action(&reference_scored.action).unwrap();
                let incremental_action = session.bundle_action(&incremental_scored.action).unwrap();
                let seat = reference_game.current_player();
                let reference_before = reference_game.boards()[seat].nature_tokens();
                let incremental_before = incremental_game.boards()[seat].nature_tokens();
                let next_reference = reference_game.transition(&reference_action).unwrap();
                let next_incremental = incremental_game.transition(&incremental_action).unwrap();
                let reference_after = next_reference.boards()[seat].nature_tokens();
                let incremental_after = next_incremental.boards()[seat].nature_tokens();
                let spent = u8::from(matches!(
                    reference_action.draft,
                    cascadia_game::DraftChoice::Independent { .. }
                ));
                let reference_earned =
                    i16::from(reference_after) + i16::from(spent) - i16::from(reference_before);
                let incremental_earned =
                    i16::from(incremental_after) + i16::from(spent) - i16::from(incremental_before);
                assert_eq!(reference_before, incremental_before);
                assert_eq!(reference_after, incremental_after);
                assert_eq!(reference_earned, incremental_earned);
                assert!((0..=1).contains(&reference_earned));
                assert_eq!(
                    i16::from(reference_before) + reference_earned - i16::from(spent),
                    i16::from(reference_after)
                );
                pinecone_checks += 1;

                reference_replay.turns.push(reference_action);
                incremental_replay.turns.push(incremental_action);
                reference_game = next_reference;
                incremental_game = next_incremental;
                assert_eq!(
                    reference_game.canonical_hash(),
                    incremental_game.canonical_hash()
                );
                turns += 1;
            }
            assert_eq!(turns % 80, 0);
            assert_eq!(score_game(&reference_game), score_game(&incremental_game));
            assert_eq!(
                reference_replay.seal().unwrap(),
                incremental_replay.seal().unwrap()
            );
            assert_eq!(reference_replay.play().unwrap(), reference_game);
            assert_eq!(incremental_replay.play().unwrap(), incremental_game);
        }
        let reference_actions_per_second = actions as f64 / reference_seconds;
        let rayon_actions_per_second = actions as f64 / rayon_cache_seconds;
        let incremental_actions_per_second = actions as f64 / incremental_seconds;
        let speedup = reference_seconds / incremental_seconds;
        let projected_three_host_games_45m = 3.0 * 2_700.0 * games as f64 / incremental_seconds;
        eprintln!(
            "{{\"schema\":\"r2-map-p1-open-corpus-gate-v1\",\"games\":{games},\"turns\":{turns},\"actions\":{actions},\"parent_restore_checks\":{parent_restore_checks},\"prediction_checks\":{prediction_checks},\"pinecone_checks\":{pinecone_checks},\"max_prediction_delta\":{max_prediction_delta},\"reference_seconds\":{reference_seconds},\"rayon_cache_seconds\":{rayon_cache_seconds},\"incremental_seconds\":{incremental_seconds},\"reference_actions_per_second\":{reference_actions_per_second},\"rayon_actions_per_second\":{rayon_actions_per_second},\"incremental_actions_per_second\":{incremental_actions_per_second},\"incremental_speedup\":{speedup},\"projected_three_host_games_45m\":{projected_three_host_games_45m}}}"
        );
        assert_eq!(turns, games * 80);
        assert_eq!(parent_restore_checks, actions);
        assert_eq!(prediction_checks, actions);
        assert_eq!(pinecone_checks, turns);
        if !calibration {
            assert!(incremental_actions_per_second >= 100_000.0);
            assert!(speedup >= 2.0);
        }
    }

    #[test]
    #[ignore = "canonical John2 P1 open-corpus parity and throughput gate"]
    fn incremental_open_corpus_complete_game_gate() {
        let games = std::env::var("R2_MAP_P1_GAMES")
            .ok()
            .map(|value| value.parse::<u64>().expect("R2_MAP_P1_GAMES is u64"))
            .unwrap_or(100);
        let calibration = std::env::var("R2_MAP_P1_ALLOW_CALIBRATION").as_deref() == Ok("1");
        assert!(
            games >= 100 || calibration,
            "P1 canonical gate requires at least 100 games"
        );
        let first_seed = 0x5031_4f50_454e_0000u64;
        let config = GameConfig::research_aaaaa(4).unwrap();
        let mut turns = 0u64;
        let mut actions = 0u64;
        let mut parent_restore_checks = 0u64;
        let mut wildlife_sibling_restore_checks = 0u64;
        let mut tile_parent_restore_checks = 0u64;
        let mut draft_root_restore_checks = 0u64;
        let mut candidate_integrity_checks = 0u64;
        let mut token_capacity_checks = 0u64;
        let mut maximum_active_tokens_per_board = 0u64;
        let mut legacy_capacity_exceedances = 0u64;
        let mut action_encoding_parity_checks = 0u64;
        let mut action_encoding_parity_seconds = 0.0f64;
        let mut prediction_checks = 0u64;
        let mut pinecone_checks = 0u64;
        let mut independent_draft_spends = 0u64;
        let mut pinecones_earned = 0u64;
        let mut incremental_seconds = 0.0f64;
        let mut determinism_seconds = 0.0f64;
        let mut prediction_seconds = 0.0f64;
        let mut sampled_reference_seconds = 0.0f64;
        let mut sampled_incremental_seconds = 0.0f64;
        let mut sampled_authoritative_decisions = 0u64;
        let mut sampled_authoritative_actions = 0u64;
        let mut early_samples = 0u64;
        let mut middle_samples = 0u64;
        let mut late_samples = 0u64;
        let mut maximum_width_samples = 0u64;
        let mut global_max_width_sum = 0u64;
        let mut supplemental_width_sum = 0u64;
        let mut global_max_fixed_only_games = 0u64;
        let mut sampled_identities = std::collections::HashSet::new();
        let mut sampled_identity_hasher = Hasher::new();
        sampled_identity_hasher.update(b"r2-map-p1-open-sample-set-v3");
        let mut corpus_hasher = Hasher::new();
        corpus_hasher.update(b"r2-map-p1-open-corpus-v2");
        let mut max_prediction_delta = 0.0f32;
        let mut per_game_actions_per_second = Vec::with_capacity(usize::try_from(games).unwrap());

        for game_offset in 0..games {
            let actions_before_game = actions;
            let incremental_seconds_before_game = incremental_seconds;
            let seed = GameSeed::from_u64(first_seed + game_offset);
            let mut reference_game = GameState::new(config, seed).unwrap();
            let mut incremental_game = reference_game.clone();
            let mut reference_replay = Replay::new(config, seed);
            let mut incremental_replay = Replay::new(config, seed);
            let mut widest_non_fixed_sample = None;
            let mut global_max_width = 0usize;
            let mut fixed_max_width = 0usize;
            while !reference_game.is_game_over() {
                assert_eq!(reference_game, incremental_game);
                let mut session = MarketDecisionSession::begin(&reference_game).unwrap();
                if session.stage() == MarketDecisionStage::FreeThreeOfAKind {
                    session.commit(&MarketDecision::KeepThreeOfAKind).unwrap();
                }
                session.commit(&MarketDecision::StopWiping).unwrap();
                let parent_hash = *session.public_state().canonical_hash().as_bytes();
                let transform =
                    D6Transform::ALL[usize::try_from((turns + game_offset) % 12).unwrap()];
                let game_index = first_seed + game_offset;
                let turn_in_game = u64::from(reference_game.completed_turns());
                let legal = session.legal_draft_actions().unwrap();
                let action_count = u64::try_from(legal.len()).unwrap();
                let (audited_legal, restore_audit) = session
                    .staged_game()
                    .audit_legal_turn_action_enumerator_restores(&MarketPrelude::default())
                    .unwrap();
                assert_eq!(audited_legal, legal);
                assert_eq!(restore_audit.emitted_actions, action_count);
                assert_eq!(
                    restore_audit.emitted_actions,
                    restore_audit.wildlife_sibling_restores + restore_audit.tile_parent_restores
                );
                assert_eq!(
                    restore_audit.root_blake3,
                    *session.staged_game().boards()[session.staged_game().current_player()]
                        .canonical_hash()
                        .as_bytes()
                );
                let fixed_sample = matches!(turn_in_game, 0 | 40 | 79);
                global_max_width = global_max_width.max(legal.len());
                if fixed_sample {
                    fixed_max_width = fixed_max_width.max(legal.len());
                }
                if !fixed_sample
                    && widest_non_fixed_sample
                        .as_ref()
                        .is_none_or(|(width, _, _, _, _, _)| legal.len() > *width)
                {
                    widest_non_fixed_sample = Some((
                        legal.len(),
                        turn_in_game,
                        parent_hash,
                        session.clone(),
                        game_index,
                        transform,
                    ));
                }

                let started = Instant::now();
                let incremental =
                    prepare_r2_map_draft_decision(&session, game_index, model(), transform)
                        .unwrap_or_else(|error| {
                            panic!(
                                "P1 preparation failed at game_offset={game_offset}, seed={:?}, game_index={game_index}, turn={turn_in_game}, transform={}, width={}, replay={}: {error:?}",
                                seed.0,
                                transform.id(),
                                legal.len(),
                                serde_json::to_string(&reference_replay).unwrap(),
                            )
                        });
                let incremental_elapsed = started.elapsed().as_secs_f64();
                incremental_seconds += incremental_elapsed;

                assert_eq!(incremental.actions, legal);
                assert_eq!(incremental.request.candidates.len(), legal.len());
                let incremental_fingerprint = prepared_tensor_fingerprint(&incremental);
                let independently_derived_action_ids = legal
                    .iter()
                    .map(|action| r2_map_draft_action_id(action).unwrap())
                    .collect::<Vec<_>>();
                let candidate_action_ids = incremental
                    .request
                    .candidates
                    .iter()
                    .map(|candidate| candidate.action_id)
                    .collect::<Vec<_>>();
                assert_eq!(candidate_action_ids, independently_derived_action_ids);
                assert_eq!(
                    candidate_action_ids
                        .iter()
                        .copied()
                        .collect::<std::collections::HashSet<_>>()
                        .len(),
                    legal.len()
                );
                candidate_integrity_checks += 1;
                for tensors in std::iter::once(&incremental.request.parent).chain(
                    incremental
                        .request
                        .candidates
                        .iter()
                        .map(|candidate| &candidate.afterstate),
                ) {
                    for board in 0..cascadia_r2::BOARD_SLOTS {
                        let start = board * cascadia_r2::R2_MAP_BOARD_TOKEN_CAPACITY;
                        let end = start + cascadia_r2::R2_MAP_BOARD_TOKEN_CAPACITY;
                        let active = tensors.token_mask[start..end]
                            .iter()
                            .map(|value| u64::from(*value))
                            .sum::<u64>();
                        assert!(
                            active
                                <= u64::try_from(cascadia_r2::R2_MAP_BOARD_TOKEN_CAPACITY).unwrap()
                        );
                        maximum_active_tokens_per_board =
                            maximum_active_tokens_per_board.max(active);
                        legacy_capacity_exceedances += u64::from(
                            active > u64::try_from(cascadia_r2::BOARD_TOKEN_CAPACITY).unwrap(),
                        );
                        token_capacity_checks += 1;
                    }
                }
                let encoding_parity_started = Instant::now();
                let cards = session.staged_game().config().scoring_cards;
                let exact_after_scores = legal
                    .par_iter()
                    .map(|action| {
                        score_board(
                            &session.staged_game().preview_active_board(action).unwrap(),
                            cards,
                        )
                    })
                    .collect::<Vec<_>>();
                let parity_transforms = if calibration {
                    D6Transform::ALL.as_slice()
                } else {
                    std::slice::from_ref(&transform)
                };
                for &parity_transform in parity_transforms {
                    let encoder =
                        R2MapActionEncoder::new(session.staged_game(), parity_transform).unwrap();
                    legal
                        .par_iter()
                        .zip(&exact_after_scores)
                        .for_each(|(action, after)| {
                            assert_eq!(
                                encoder.encode_staged_after_score(action, *after).unwrap(),
                                encoder.encode(action).unwrap()
                            );
                        });
                }
                action_encoding_parity_checks +=
                    u64::try_from(legal.len().checked_mul(parity_transforms.len()).unwrap())
                        .unwrap();
                action_encoding_parity_seconds += encoding_parity_started.elapsed().as_secs_f64();
                actions += action_count;
                parent_restore_checks += restore_audit.emitted_actions;
                wildlife_sibling_restore_checks += restore_audit.wildlife_sibling_restores;
                tile_parent_restore_checks += restore_audit.tile_parent_restores;
                draft_root_restore_checks += restore_audit.draft_root_restores;
                assert_eq!(
                    *session.public_state().canonical_hash().as_bytes(),
                    parent_hash,
                    "preparation must restore the exact public parent"
                );

                if fixed_sample {
                    let sample_bin = match turn_in_game {
                        0 => P1SampleBin::Early,
                        40 => P1SampleBin::Middle,
                        79 => P1SampleBin::Late,
                        _ => unreachable!(),
                    };
                    let sample_identity = P1SampleIdentityInput {
                        game_offset,
                        seed,
                        game_index,
                        turn: turn_in_game,
                        parent_hash,
                        transform,
                        width: legal.len(),
                        bin: sample_bin,
                    }
                    .identity_hash();
                    assert!(sampled_identities.insert(sample_identity));
                    sampled_identity_hasher.update(&sample_identity);
                    let started = Instant::now();
                    let authoritative = prepare_r2_map_draft_decision_reference(
                        &session,
                        game_index,
                        model(),
                        transform,
                    )
                    .unwrap();
                    sampled_reference_seconds += started.elapsed().as_secs_f64();
                    sampled_incremental_seconds += incremental_elapsed;
                    sampled_authoritative_decisions += 1;
                    sampled_authoritative_actions += action_count;
                    match turn_in_game {
                        0 => early_samples += 1,
                        40 => middle_samples += 1,
                        79 => late_samples += 1,
                        _ => unreachable!(),
                    }
                    assert_eq!(authoritative, incremental);
                }

                let started = Instant::now();
                let incremental_scored =
                    score_r2_map_decision(&mut TensorFingerprintPredictor, incremental, None)
                        .unwrap();
                prediction_seconds += started.elapsed().as_secs_f64();

                // The first complete request has been consumed before the
                // deterministic replay is allocated. This keeps the gate's
                // peak representative of one production request instead of
                // manufacturing a second maximum-width tensor resident at
                // the same time.
                let started = Instant::now();
                let deterministic_repeat =
                    prepare_r2_map_draft_decision(&session, game_index, model(), transform)
                        .unwrap();
                determinism_seconds += started.elapsed().as_secs_f64();
                assert_eq!(deterministic_repeat.actions, legal);
                assert_eq!(
                    prepared_tensor_fingerprint(&deterministic_repeat),
                    incremental_fingerprint
                );
                let deterministic_scored = score_r2_map_decision(
                    &mut TensorFingerprintPredictor,
                    deterministic_repeat,
                    None,
                )
                .unwrap();
                for (incremental_score, deterministic_score) in incremental_scored
                    .scores
                    .iter()
                    .zip(&deterministic_scored.scores)
                {
                    let delta = (incremental_score - deterministic_score).abs();
                    max_prediction_delta = max_prediction_delta.max(delta);
                    assert!(delta <= 1.0e-6);
                    prediction_checks += 1;
                }
                assert_eq!(
                    incremental_scored.selected_index,
                    deterministic_scored.selected_index
                );
                assert_eq!(incremental_scored.action, deterministic_scored.action);

                let reference_action = session.bundle_action(&incremental_scored.action).unwrap();
                let incremental_action = session.bundle_action(&incremental_scored.action).unwrap();
                let seat = reference_game.current_player();
                let reference_before = reference_game.boards()[seat].nature_tokens();
                let incremental_before = incremental_game.boards()[seat].nature_tokens();
                let next_reference = reference_game.transition(&reference_action).unwrap();
                let next_incremental = incremental_game.transition(&incremental_action).unwrap();
                let reference_after = next_reference.boards()[seat].nature_tokens();
                let incremental_after = next_incremental.boards()[seat].nature_tokens();
                let spent = u8::from(matches!(
                    reference_action.draft,
                    cascadia_game::DraftChoice::Independent { .. }
                ));
                independent_draft_spends += u64::from(spent);
                let reference_earned =
                    i16::from(reference_after) + i16::from(spent) - i16::from(reference_before);
                let incremental_earned =
                    i16::from(incremental_after) + i16::from(spent) - i16::from(incremental_before);
                assert_eq!(reference_before, incremental_before);
                assert_eq!(reference_after, incremental_after);
                assert_eq!(reference_earned, incremental_earned);
                assert!((0..=1).contains(&reference_earned));
                assert_eq!(
                    i16::from(reference_before) + reference_earned - i16::from(spent),
                    i16::from(reference_after)
                );
                pinecones_earned += u64::try_from(reference_earned).unwrap();
                pinecone_checks += 1;

                reference_replay.turns.push(reference_action);
                incremental_replay.turns.push(incremental_action);
                reference_game = next_reference;
                incremental_game = next_incremental;
                assert_eq!(
                    reference_game.canonical_hash(),
                    incremental_game.canonical_hash()
                );
                turns += 1;
            }

            let (
                supplemental_width,
                supplemental_turn,
                supplemental_parent_hash,
                maximum_session,
                maximum_game_index,
                maximum_transform,
            ) = widest_non_fixed_sample.expect("complete game has a non-fixed widest decision");
            let sample_identity = P1SampleIdentityInput {
                game_offset,
                seed,
                game_index: maximum_game_index,
                turn: supplemental_turn,
                parent_hash: supplemental_parent_hash,
                transform: maximum_transform,
                width: supplemental_width,
                bin: P1SampleBin::MaximumWidth,
            }
            .identity_hash();
            assert!(sampled_identities.insert(sample_identity));
            sampled_identity_hasher.update(&sample_identity);
            let started = Instant::now();
            let maximum_authoritative = prepare_r2_map_draft_decision_reference(
                &maximum_session,
                maximum_game_index,
                model(),
                maximum_transform,
            )
            .unwrap();
            sampled_reference_seconds += started.elapsed().as_secs_f64();
            let started = Instant::now();
            let maximum_incremental = prepare_r2_map_draft_decision(
                &maximum_session,
                maximum_game_index,
                model(),
                maximum_transform,
            )
            .unwrap();
            sampled_incremental_seconds += started.elapsed().as_secs_f64();
            assert_eq!(maximum_incremental.actions.len(), supplemental_width);
            assert_eq!(maximum_authoritative, maximum_incremental);
            sampled_authoritative_decisions += 1;
            sampled_authoritative_actions += u64::try_from(supplemental_width).unwrap();
            maximum_width_samples += 1;
            global_max_width_sum += u64::try_from(global_max_width).unwrap();
            supplemental_width_sum += u64::try_from(supplemental_width).unwrap();
            assert_eq!(global_max_width, fixed_max_width.max(supplemental_width));
            if supplemental_width < global_max_width {
                assert_eq!(fixed_max_width, global_max_width);
                global_max_fixed_only_games += 1;
            }

            assert_eq!(turns % 80, 0);
            let reference_scores = score_game(&reference_game);
            let incremental_scores = score_game(&incremental_game);
            assert_eq!(reference_scores, incremental_scores);
            let reference_seal = reference_replay.seal().unwrap();
            let incremental_seal = incremental_replay.seal().unwrap();
            assert_eq!(reference_seal, incremental_seal);
            assert_eq!(reference_replay, incremental_replay);
            assert_eq!(reference_replay.play().unwrap(), reference_game);
            assert_eq!(incremental_replay.play().unwrap(), incremental_game);
            let final_state_bytes = reference_game.canonical_bytes();
            let replay_bytes = serde_json::to_vec(&reference_replay).unwrap();
            let game_action_count = actions - actions_before_game;
            update_p1_corpus_game_header(&mut corpus_hasher, game_offset, seed, game_action_count);
            corpus_hasher.update(&reference_seal);
            corpus_hasher.update(&(final_state_bytes.len() as u64).to_le_bytes());
            corpus_hasher.update(&final_state_bytes);
            corpus_hasher.update(&(replay_bytes.len() as u64).to_le_bytes());
            corpus_hasher.update(&replay_bytes);
            corpus_hasher.update(&(reference_scores.len() as u64).to_le_bytes());
            for score in &reference_scores {
                update_score_digest(&mut corpus_hasher, score);
            }
            per_game_actions_per_second.push(
                game_action_count as f64 / (incremental_seconds - incremental_seconds_before_game),
            );
        }
        per_game_actions_per_second.sort_by(f64::total_cmp);
        let incremental_actions_per_second = actions as f64 / incremental_seconds;
        let minimum_game_actions_per_second = per_game_actions_per_second[0];
        let median_game_actions_per_second =
            per_game_actions_per_second[per_game_actions_per_second.len() / 2];
        let sampled_speedup = sampled_reference_seconds / sampled_incremental_seconds;
        let projected_three_host_preparation_only_games_45m =
            3.0 * 2_700.0 * games as f64 / incremental_seconds;
        let last_seed_exclusive = first_seed + games;
        let sampled_identity_blake3 = sampled_identity_hasher.finalize().to_hex().to_string();
        update_p1_corpus_totals(&mut corpus_hasher, games, actions);
        let corpus_blake3 = corpus_hasher.finalize().to_hex().to_string();
        let mut metrics = serde_json::Map::new();
        macro_rules! metric {
            ($name:literal, $value:expr) => {
                metrics.insert($name.to_owned(), serde_json::json!($value));
            };
        }
        metric!("schema", "r2-map-p1-open-corpus-gate-v5");
        metric!(
            "corpus_digest_contract",
            "final-state-replay-score-per-game-actions-total-actions-v2"
        );
        metric!("seed_domain", "p1-open-test-v1");
        metric!("first_seed", first_seed.to_string());
        metric!("last_seed_exclusive", last_seed_exclusive.to_string());
        metric!("games", games);
        metric!("turns", turns);
        metric!("actions", actions);
        metric!("parent_restore_checks", parent_restore_checks);
        metric!(
            "wildlife_sibling_restore_checks",
            wildlife_sibling_restore_checks
        );
        metric!("tile_parent_restore_checks", tile_parent_restore_checks);
        metric!("draft_root_restore_checks", draft_root_restore_checks);
        metric!("candidate_integrity_checks", candidate_integrity_checks);
        metric!("token_capacity_checks", token_capacity_checks);
        metric!(
            "rules_complete_board_token_capacity",
            cascadia_r2::R2_MAP_BOARD_TOKEN_CAPACITY
        );
        metric!(
            "frozen_foundation_board_token_capacity",
            cascadia_r2::BOARD_TOKEN_CAPACITY
        );
        metric!(
            "maximum_active_tokens_per_board",
            maximum_active_tokens_per_board
        );
        metric!("legacy_capacity_exceedances", legacy_capacity_exceedances);
        metric!(
            "action_encoding_parity_checks",
            action_encoding_parity_checks
        );
        metric!(
            "action_encoding_parity_seconds",
            action_encoding_parity_seconds
        );
        metric!("prediction_checks", prediction_checks);
        metric!("pinecone_checks", pinecone_checks);
        metric!("independent_draft_spends", independent_draft_spends);
        metric!("pinecones_earned", pinecones_earned);
        metric!("corpus_blake3", corpus_blake3);
        metric!("max_prediction_delta", max_prediction_delta);
        metric!(
            "sampled_authoritative_decisions",
            sampled_authoritative_decisions
        );
        metric!(
            "sampled_authoritative_actions",
            sampled_authoritative_actions
        );
        metric!("sampled_identity_blake3", sampled_identity_blake3);
        metric!("early_samples", early_samples);
        metric!("middle_samples", middle_samples);
        metric!("late_samples", late_samples);
        metric!("maximum_width_samples", maximum_width_samples);
        metric!("global_max_width_sum", global_max_width_sum);
        metric!("supplemental_width_sum", supplemental_width_sum);
        metric!("global_max_fixed_only_games", global_max_fixed_only_games);
        metric!("sampled_reference_seconds", sampled_reference_seconds);
        metric!("sampled_incremental_seconds", sampled_incremental_seconds);
        metric!("incremental_seconds", incremental_seconds);
        metric!("determinism_seconds", determinism_seconds);
        metric!("prediction_seconds", prediction_seconds);
        metric!(
            "incremental_actions_per_second",
            incremental_actions_per_second
        );
        metric!(
            "minimum_game_actions_per_second",
            minimum_game_actions_per_second
        );
        metric!(
            "median_game_actions_per_second",
            median_game_actions_per_second
        );
        metric!("sampled_speedup", sampled_speedup);
        metric!(
            "projection_scope",
            "preparation-only; excludes real MLX service, batching, IPC, model inference, and game advancement"
        );
        metric!(
            "projected_three_host_preparation_only_games_45m",
            projected_three_host_preparation_only_games_45m
        );
        eprintln!("{}", serde_json::Value::Object(metrics));
        assert_eq!(turns, games * 80);
        assert_eq!(parent_restore_checks, actions);
        assert_eq!(
            parent_restore_checks,
            wildlife_sibling_restore_checks + tile_parent_restore_checks
        );
        assert_eq!(candidate_integrity_checks, turns);
        assert_eq!(
            action_encoding_parity_checks,
            actions * if calibration { 12 } else { 1 }
        );
        assert_eq!(prediction_checks, actions);
        assert_eq!(pinecone_checks, turns);
        assert_eq!(early_samples, games);
        assert_eq!(middle_samples, games);
        assert_eq!(late_samples, games);
        assert_eq!(maximum_width_samples, games);
        assert_eq!(sampled_authoritative_decisions, games * 4);
        assert_eq!(
            sampled_identities.len(),
            usize::try_from(games * 4).unwrap()
        );
        if !calibration {
            assert!(incremental_actions_per_second >= 100_000.0);
            assert!(minimum_game_actions_per_second >= 100_000.0);
            assert!(sampled_speedup >= 2.0);
            assert!(independent_draft_spends > 0);
            assert!(pinecones_earned > 0);
        }
    }

    #[test]
    fn d6_changes_wire_geometry_without_changing_canonical_action_identity_order() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(9002),
        )
        .unwrap();
        let session = stopped_session(&game);
        let identity =
            prepare_r2_map_draft_decision(&session, 8, model(), D6Transform::IDENTITY).unwrap();
        let transformed =
            prepare_r2_map_draft_decision(&session, 8, model(), D6Transform::ALL[7]).unwrap();
        assert_eq!(identity.actions, transformed.actions);
        assert_eq!(
            identity
                .request
                .candidates
                .iter()
                .map(|candidate| candidate.action_id)
                .collect::<Vec<_>>(),
            transformed
                .request
                .candidates
                .iter()
                .map(|candidate| candidate.action_id)
                .collect::<Vec<_>>()
        );
        assert!(
            identity
                .request
                .candidates
                .iter()
                .zip(&transformed.request.candidates)
                .any(|(left, right)| left.action_bytes != right.action_bytes)
        );
    }

    #[test]
    fn staged_argmax_scores_market_choices_before_the_complete_post_stop_draft() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(9003),
        )
        .unwrap();
        let selected = select_r2_map_argmax(&mut ExactFakePredictor, &game, 9, model()).unwrap();
        assert!(!selected.market_decisions.is_empty());
        assert!(matches!(
            selected.market_decisions.last().unwrap().selected,
            MarketDecision::StopWiping
        ));
        assert_eq!(
            selected.draft_context.ordinal as usize,
            selected.market_decisions.len()
        );
        assert_eq!(selected.draft_context.kind, R2MapTurnDecisionKind::Draft);
        assert_eq!(selected.draft.action_ids.len(), selected.draft.scores.len());
        assert_eq!(
            selected.bundled_action_id,
            r2_map_draft_action_id(&selected.action).unwrap()
        );
        game.transition(&selected.action).unwrap();
    }
}
