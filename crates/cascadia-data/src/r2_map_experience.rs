//! Compact, replay-authoritative experience for the R2-MAP expert-iteration loop.
//!
//! The file format deliberately stores actions and identities, not expanded R2
//! tensors.  Parent/afterstate tensors are reconstructed from the sealed replay
//! on the training node and are therefore always checked against the canonical
//! game engine.

use std::{
    collections::HashSet,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

use blake3::Hasher;
use cascadia_game::{
    DraftChoice, GameConfig, GameSeed, GameState, MarketDecision, MarketDecisionSession,
    MarketDecisionStage, PublicGameState, Replay, ScoreBreakdown, TurnAction,
    public_market_action_identity, public_market_decision_identity, score_board, score_game,
};
use cascadia_sim::MatchResult;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const R2_MAP_EXPERIENCE_SCHEMA_VERSION: u16 = 2;
pub const R2_MAP_EXPERIENCE_MAGIC: &[u8; 8] = b"CSDR2XP\0";
pub const R2_MAP_EXPERIENCE_HEADER_SIZE: usize = 50;
pub const R2_MAP_SHARD_MAGIC: &[u8; 8] = b"CSDR2SH\0";
pub const R2_MAP_SHARD_HEADER_SIZE: usize = 64;
pub const R2_MAP_COMPONENT_COUNT: usize = 11;
pub const R2_MAP_ASSIGNMENT_RNG_DOMAIN: &str = "r2-map-seat-assignment-v1";
pub const R2_MAP_EXPLORATION_RNG_DOMAIN: &str = "r2-map-exploration-v2";
pub const R2_MAP_STRATEGY_RNG_DOMAIN: &str = "cascadia-v2-strategy-rng";
pub const R2_MAP_EXPLORATION_SCHEDULE_ID: &str = "epsilon-0.10-decay-0.85-floor-0.02-v1";

pub type IdentityHash = [u8; 32];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum R2MapCollectionKind {
    Bootstrap,
    IterativeTraining,
    Benchmark,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum R2MapSeedPurpose {
    Bootstrap,
    Generation,
    LongitudinalBenchmark,
    CandidateGate,
    ProtectedFinal,
}

impl R2MapSeedPurpose {
    const fn id(self) -> &'static str {
        match self {
            Self::Bootstrap => "bootstrap",
            Self::Generation => "generation",
            Self::LongitudinalBenchmark => "longitudinal-benchmark",
            Self::CandidateGate => "candidate-gate",
            Self::ProtectedFinal => "protected-final",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum R2MapPolicyRole {
    Newest,
    Historical,
    Greedy,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct R2MapPolicyIdentity {
    pub policy_id: String,
    pub role: R2MapPolicyRole,
    pub checkpoint_hash: Option<IdentityHash>,
}

impl R2MapPolicyIdentity {
    pub fn newest(policy_id: impl Into<String>, checkpoint_hash: IdentityHash) -> Self {
        Self {
            policy_id: policy_id.into(),
            role: R2MapPolicyRole::Newest,
            checkpoint_hash: Some(checkpoint_hash),
        }
    }

    pub fn historical(policy_id: impl Into<String>, checkpoint_hash: IdentityHash) -> Self {
        Self {
            policy_id: policy_id.into(),
            role: R2MapPolicyRole::Historical,
            checkpoint_hash: Some(checkpoint_hash),
        }
    }

    pub fn greedy() -> Self {
        Self {
            policy_id: "greedy-v1".to_owned(),
            role: R2MapPolicyRole::Greedy,
            checkpoint_hash: None,
        }
    }

    pub fn validate(&self) -> Result<(), R2MapExperienceError> {
        if self.policy_id.trim().is_empty() {
            return Err(R2MapExperienceError::InvalidIdentity(
                "policy identity is empty",
            ));
        }
        match (self.role, self.checkpoint_hash) {
            (R2MapPolicyRole::Greedy, None)
            | (R2MapPolicyRole::Newest | R2MapPolicyRole::Historical, Some(_)) => Ok(()),
            _ => Err(R2MapExperienceError::InvalidIdentity(
                "checkpoint policy roles require a hash and greedy forbids one",
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct R2MapGameIdentity {
    pub campaign_id: String,
    pub iteration: u32,
    pub host_id: String,
    pub global_game_index: u64,
    pub game_id: IdentityHash,
}

impl R2MapGameIdentity {
    pub fn new(
        campaign_id: impl Into<String>,
        iteration: u32,
        host_id: impl Into<String>,
        global_game_index: u64,
        seed: GameSeed,
    ) -> Self {
        let campaign_id = campaign_id.into();
        let host_id = host_id.into();
        let game_id = hash_parts(
            b"r2-map-game-identity-v1",
            &[
                campaign_id.as_bytes(),
                &iteration.to_le_bytes(),
                host_id.as_bytes(),
                &global_game_index.to_le_bytes(),
                &seed.0,
            ],
        );
        Self {
            campaign_id,
            iteration,
            host_id,
            global_game_index,
            game_id,
        }
    }

    fn validate(&self, seed: GameSeed) -> Result<(), R2MapExperienceError> {
        if self.campaign_id.trim().is_empty()
            || !matches!(
                self.host_id.as_str(),
                "john1" | "john2" | "john3" | "scheduler"
            )
        {
            return Err(R2MapExperienceError::InvalidIdentity(
                "campaign must be non-empty and execution identity must be a worker or scheduler",
            ));
        }
        let expected = Self::new(
            self.campaign_id.clone(),
            self.iteration,
            self.host_id.clone(),
            self.global_game_index,
            seed,
        );
        if self.game_id != expected.game_id {
            return Err(R2MapExperienceError::IdentityMismatch("game"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct R2MapSeedLease {
    pub campaign_id: String,
    pub iteration: u32,
    pub purpose: R2MapSeedPurpose,
    pub host_id: String,
    pub first_game_index: u64,
    pub game_count: u64,
}

impl R2MapSeedLease {
    pub fn validate(&self) -> Result<(), R2MapExperienceError> {
        if self.campaign_id.trim().is_empty()
            || !matches!(self.host_id.as_str(), "john1" | "john2" | "john3")
            || self.game_count == 0
            || self.first_game_index.checked_add(self.game_count).is_none()
        {
            return Err(R2MapExperienceError::InvalidIdentity(
                "seed lease has an invalid campaign, host, or range",
            ));
        }
        Ok(())
    }

    pub fn end_game_index(&self) -> Result<u64, R2MapExperienceError> {
        self.first_game_index.checked_add(self.game_count).ok_or(
            R2MapExperienceError::InvalidIdentity("seed lease range overflows"),
        )
    }

    pub fn seed(&self, global_game_index: u64) -> Result<GameSeed, R2MapExperienceError> {
        self.validate()?;
        if !(self.first_game_index..self.end_game_index()?).contains(&global_game_index) {
            return Err(R2MapExperienceError::InvalidIdentity(
                "game index is outside its seed lease",
            ));
        }
        Ok(r2_map_game_seed(
            &self.campaign_id,
            self.purpose,
            self.iteration,
            global_game_index,
        ))
    }

    pub fn game_identity(
        &self,
        global_game_index: u64,
    ) -> Result<R2MapGameIdentity, R2MapExperienceError> {
        let seed = self.seed(global_game_index)?;
        Ok(R2MapGameIdentity::new(
            self.campaign_id.clone(),
            self.iteration,
            self.host_id.clone(),
            global_game_index,
            seed,
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct R2MapProtocolIdentity {
    pub collector_hash: IdentityHash,
    pub source_hash: IdentityHash,
    pub serving_protocol_hash: IdentityHash,
}

impl R2MapProtocolIdentity {
    pub fn validate(&self) -> Result<(), R2MapExperienceError> {
        if [
            self.collector_hash,
            self.source_hash,
            self.serving_protocol_hash,
        ]
        .contains(&[0; 32])
        {
            return Err(R2MapExperienceError::InvalidIdentity(
                "protocol hashes must be populated",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct R2MapRngIdentity {
    pub assignment_domain: String,
    pub exploration_domain: String,
    pub strategy_domain: String,
}

impl Default for R2MapRngIdentity {
    fn default() -> Self {
        Self {
            assignment_domain: R2_MAP_ASSIGNMENT_RNG_DOMAIN.to_owned(),
            exploration_domain: R2_MAP_EXPLORATION_RNG_DOMAIN.to_owned(),
            strategy_domain: R2_MAP_STRATEGY_RNG_DOMAIN.to_owned(),
        }
    }
}

impl R2MapRngIdentity {
    fn validate(&self) -> Result<(), R2MapExperienceError> {
        if self != &Self::default() {
            return Err(R2MapExperienceError::InvalidIdentity(
                "RNG domains do not match the frozen v1 contract",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct R2MapExplorationIdentity {
    pub schedule_id: String,
    pub epsilon_parts_per_million: u32,
    pub temperature_parts_per_million: u32,
    pub enabled: bool,
}

impl R2MapExplorationIdentity {
    pub fn disabled() -> Self {
        Self {
            schedule_id: "disabled".to_owned(),
            epsilon_parts_per_million: 0,
            temperature_parts_per_million: 0,
            enabled: false,
        }
    }

    pub fn training(iteration: u32, temperature_parts_per_million: u32) -> Self {
        Self {
            schedule_id: R2_MAP_EXPLORATION_SCHEDULE_ID.to_owned(),
            epsilon_parts_per_million: exploration_epsilon_ppm(iteration),
            temperature_parts_per_million,
            enabled: true,
        }
    }

    pub fn validate(
        &self,
        collection_kind: R2MapCollectionKind,
        iteration: u32,
    ) -> Result<(), R2MapExperienceError> {
        match collection_kind {
            R2MapCollectionKind::IterativeTraining => {
                if !self.enabled
                    || self.schedule_id != R2_MAP_EXPLORATION_SCHEDULE_ID
                    || self.epsilon_parts_per_million != exploration_epsilon_ppm(iteration)
                    || self.temperature_parts_per_million == 0
                {
                    return Err(R2MapExperienceError::Exploration(
                        "iterative training must use the frozen nonzero schedule",
                    ));
                }
            }
            R2MapCollectionKind::Bootstrap | R2MapCollectionKind::Benchmark => {
                if self != &Self::disabled() {
                    return Err(R2MapExperienceError::Exploration(
                        "bootstrap and benchmark trajectories disable exploration",
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Integer-only implementation of `max(0.02, 0.10 * 0.85^iteration)`.
pub fn exploration_epsilon_ppm(iteration: u32) -> u32 {
    let mut epsilon = 100_000u64;
    for _ in 0..iteration {
        epsilon = (epsilon * 85 + 50) / 100;
    }
    u32::try_from(epsilon.max(20_000)).expect("bounded epsilon")
}

/// Stable game-level inputs for the counter-based exploration RNG.
///
/// Keeping these values together prevents callers from accidentally mixing a
/// seed, focal seat, or campaign identity from different games when deriving
/// decision and action draws.
#[derive(Debug, Clone, Copy)]
pub struct R2MapExplorationRngContext<'a> {
    identity: &'a R2MapGameIdentity,
    seed: GameSeed,
    focal_seat: u8,
}

impl<'a> R2MapExplorationRngContext<'a> {
    pub fn new(identity: &'a R2MapGameIdentity, seed: GameSeed, focal_seat: u8) -> Self {
        Self {
            identity,
            seed,
            focal_seat,
        }
    }
}

/// Exact public-decision identity used by the counter-based exploration RNG.
///
/// This is deliberately distinct from an exploration draw: it names the
/// public decision before either the epsilon gate or action draw is sampled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct R2MapExplorationDecisionIdentity {
    turn_index: u16,
    ordinal: u8,
    stage: MarketDecisionStage,
    decision_id: IdentityHash,
    parent_public_hash: IdentityHash,
}

impl R2MapExplorationDecisionIdentity {
    pub fn new(
        turn_index: u16,
        ordinal: u8,
        stage: MarketDecisionStage,
        decision_id: IdentityHash,
        parent_public_hash: IdentityHash,
    ) -> Self {
        Self {
            turn_index,
            ordinal,
            stage,
            decision_id,
            parent_public_hash,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct R2MapExplorationDraw {
    pub turn_index: u16,
    pub ordinal: u8,
    pub stage: MarketDecisionStage,
    pub decision_id: IdentityHash,
    pub parent_public_hash: IdentityHash,
    pub selected_action_id: IdentityHash,
    pub explore_draw_u64: u64,
    pub action_draw_u64: Option<u64>,
    pub explored: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct R2MapExplorationTrace {
    pub ordinal: u8,
    pub stage: MarketDecisionStage,
    pub decision_id: IdentityHash,
    pub parent_public_hash: IdentityHash,
    pub selected_action_id: IdentityHash,
    pub explore_draw_u64: u64,
    pub action_draw_u64: Option<u64>,
    pub explored: bool,
    pub decision_rng_key: IdentityHash,
    pub selected_action_rng_key: IdentityHash,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct R2MapMarketDecisionRecord {
    pub ordinal: u8,
    pub stage: MarketDecisionStage,
    pub decision_id: IdentityHash,
    pub parent_public_hash: IdentityHash,
    pub selected_index: u8,
    pub selected_action_id: IdentityHash,
    pub selected: MarketDecision,
    pub resulting_public_hash: IdentityHash,
    pub legal_action_count: u8,
    pub ordered_legal_action_ids_blake3: IdentityHash,
    pub exploration: Option<R2MapExplorationTrace>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct R2MapPineconeEvent {
    pub before: u8,
    pub earned: u8,
    pub spent_on_wipes: u8,
    pub spent_on_independent_draft: u8,
    pub after: u8,
}

impl R2MapPineconeEvent {
    pub fn total_spent(self) -> u8 {
        self.spent_on_wipes + self.spent_on_independent_draft
    }

    fn validate(self) -> Result<(), R2MapExperienceError> {
        if self.earned > 1
            || self.spent_on_independent_draft > 1
            || i16::from(self.before) + i16::from(self.earned) - i16::from(self.total_spent())
                != i16::from(self.after)
        {
            return Err(R2MapExperienceError::PineconeConservation);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct R2MapDecisionRecord {
    pub turn_index: u16,
    pub seat: u8,
    pub position_id: IdentityHash,
    pub action_id: IdentityHash,
    pub bundled_action_contract_id: IdentityHash,
    pub parent_public_hash: IdentityHash,
    pub afterstate_public_hash: IdentityHash,
    pub market_decisions: Vec<R2MapMarketDecisionRecord>,
    pub draft_decision_id: IdentityHash,
    pub draft_parent_public_hash: IdentityHash,
    pub draft_action_id: IdentityHash,
    pub draft_legal_action_count: u32,
    pub draft_ordered_action_ids_blake3: IdentityHash,
    pub pinecones: R2MapPineconeEvent,
    pub exploration: Option<R2MapExplorationTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct R2MapRecordContext {
    pub collection_kind: R2MapCollectionKind,
    pub identity: R2MapGameIdentity,
    pub seed_purpose: R2MapSeedPurpose,
    pub focal_seat: u8,
    pub seats: Vec<R2MapPolicyIdentity>,
    pub rng: R2MapRngIdentity,
    pub exploration: R2MapExplorationIdentity,
    pub protocols: R2MapProtocolIdentity,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct R2MapGameRecord {
    pub schema_version: u16,
    pub collection_kind: R2MapCollectionKind,
    pub identity: R2MapGameIdentity,
    pub config: GameConfig,
    pub seed: GameSeed,
    pub seed_purpose: R2MapSeedPurpose,
    pub focal_seat: u8,
    pub focal_checkpoint_hash: Option<IdentityHash>,
    pub seats: Vec<R2MapPolicyIdentity>,
    pub rng: R2MapRngIdentity,
    pub exploration: R2MapExplorationIdentity,
    pub decisions: Vec<R2MapDecisionRecord>,
    pub scores: Vec<ScoreBreakdown>,
    pub terminal_public_state_hash: IdentityHash,
    pub terminal_replay_hash: IdentityHash,
    pub protocols: R2MapProtocolIdentity,
    pub replay: Replay,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct R2MapPublicTurnTrace {
    pub turn_index: u16,
    pub bundled_action_id: IdentityHash,
    pub market_decisions: Vec<R2MapMarketDecisionRecord>,
    pub draft_decision_id: IdentityHash,
    pub draft_parent_public_hash: IdentityHash,
    pub draft_action_id: IdentityHash,
    pub draft_legal_action_count: u32,
    pub draft_ordered_action_ids_blake3: IdentityHash,
}

pub fn reconstruct_r2_map_public_turn(
    game: &GameState,
    turn_index: u16,
    bundled: &TurnAction,
) -> Result<R2MapPublicTurnTrace, R2MapExperienceError> {
    let mut session = MarketDecisionSession::begin(game)?;
    let mut selections = Vec::new();
    if session.stage() == MarketDecisionStage::FreeThreeOfAKind {
        selections.push(if bundled.replace_three_of_a_kind {
            MarketDecision::ReplaceThreeOfAKind
        } else {
            MarketDecision::KeepThreeOfAKind
        });
    } else if bundled.replace_three_of_a_kind {
        return Err(R2MapExperienceError::InvalidRecord(
            "bundled free replacement is unavailable",
        ));
    }
    selections.extend(
        bundled
            .wildlife_wipes
            .iter()
            .cloned()
            .map(MarketDecision::PaidWipe),
    );
    selections.push(MarketDecision::StopWiping);

    let mut market_decisions = Vec::with_capacity(selections.len());
    for (index, selected) in selections.into_iter().enumerate() {
        let ordinal = u8::try_from(index).map_err(|_| {
            R2MapExperienceError::InvalidRecord("market decision ordinal exceeds u8")
        })?;
        let stage = session.stage();
        let parent_public_hash = *session.public_state().canonical_hash().as_bytes();
        let decision_id =
            public_market_decision_identity(parent_public_hash, turn_index, ordinal, stage);
        let legal = session.legal_decisions();
        let action_ids = legal
            .iter()
            .map(|decision| {
                let bytes = decision.public_wire_bytes(stage)?;
                Ok(public_market_action_identity(decision_id, bytes))
            })
            .collect::<Result<Vec<_>, cascadia_game::RuleError>>()?;
        let selected_index = legal
            .iter()
            .position(|decision| decision == &selected)
            .ok_or(R2MapExperienceError::InvalidRecord(
                "replay market choice is absent from its public legal screen",
            ))?;
        let selected_index = u8::try_from(selected_index)
            .map_err(|_| R2MapExperienceError::InvalidRecord("market selected index exceeds u8"))?;
        let selected_action_id = action_ids[usize::from(selected_index)];
        session.commit(&selected)?;
        market_decisions.push(R2MapMarketDecisionRecord {
            ordinal,
            stage,
            decision_id,
            parent_public_hash,
            selected_index,
            selected_action_id,
            selected,
            resulting_public_hash: *session.public_state().canonical_hash().as_bytes(),
            legal_action_count: u8::try_from(action_ids.len()).map_err(|_| {
                R2MapExperienceError::InvalidRecord("market legal screen exceeds u8")
            })?,
            ordered_legal_action_ids_blake3: r2_map_ordered_action_ids_blake3(&action_ids)?,
            exploration: None,
        });
    }

    let mut draft = bundled.clone();
    draft.replace_three_of_a_kind = false;
    draft.wildlife_wipes.clear();
    if session.bundle_action(&draft)? != *bundled {
        return Err(R2MapExperienceError::InvalidRecord(
            "staged public decisions do not reconstruct the bundled action",
        ));
    }
    let draft_parent_public_hash = *session.public_state().canonical_hash().as_bytes();
    let draft_decision_id = draft_parent_public_hash;
    let legal_drafts = session.legal_draft_actions()?;
    let draft_action_ids = legal_drafts
        .iter()
        .map(r2_map_draft_action_id)
        .collect::<Result<Vec<_>, _>>()?;
    let draft_index = legal_drafts
        .iter()
        .position(|candidate| candidate == &draft)
        .ok_or(R2MapExperienceError::InvalidRecord(
            "bundled draft is absent from the post-stop legal screen",
        ))?;
    Ok(R2MapPublicTurnTrace {
        turn_index,
        bundled_action_id: r2_map_draft_action_id(bundled)?,
        market_decisions,
        draft_decision_id,
        draft_parent_public_hash,
        draft_action_id: draft_action_ids[draft_index],
        draft_legal_action_count: u32::try_from(draft_action_ids.len())
            .map_err(|_| R2MapExperienceError::InvalidRecord("draft legal screen exceeds u32"))?,
        draft_ordered_action_ids_blake3: r2_map_ordered_action_ids_blake3(&draft_action_ids)?,
    })
}

pub fn reconstruct_r2_map_public_turns(
    replay: &Replay,
) -> Result<Vec<R2MapPublicTurnTrace>, R2MapExperienceError> {
    let mut game = GameState::new(replay.config, replay.seed)?;
    let mut turns = Vec::with_capacity(replay.turns.len());
    for (index, action) in replay.turns.iter().enumerate() {
        let turn_index = u16::try_from(index)
            .map_err(|_| R2MapExperienceError::InvalidRecord("turn index exceeds u16"))?;
        turns.push(reconstruct_r2_map_public_turn(&game, turn_index, action)?);
        game.apply(action)?;
    }
    Ok(turns)
}

/// Reconstruct the exact counter-based exploration evidence expected from a
/// sealed replay. This is an independent verifier/helper; production runners
/// still return their observed draws and `from_match` compares them one by one.
pub fn expected_r2_map_exploration_draws(
    context: &R2MapRecordContext,
    replay: &Replay,
) -> Result<Vec<R2MapExplorationDraw>, R2MapExperienceError> {
    if !context.exploration.enabled {
        return Ok(Vec::new());
    }
    let rng_context =
        R2MapExplorationRngContext::new(&context.identity, replay.seed, context.focal_seat);
    let mut game = GameState::new(replay.config, replay.seed)?;
    let mut draws = Vec::new();
    for (turn, action) in replay.turns.iter().enumerate() {
        let turn_index = u16::try_from(turn)
            .map_err(|_| R2MapExperienceError::Exploration("turn exceeds u16"))?;
        if game.current_player() == usize::from(context.focal_seat) {
            let public_turn = reconstruct_r2_map_public_turn(&game, turn_index, action)?;
            for market in &public_turn.market_decisions {
                draws.push(expected_exploration_draw(
                    rng_context,
                    context.exploration.epsilon_parts_per_million,
                    R2MapExplorationDecisionIdentity::new(
                        turn_index,
                        market.ordinal,
                        market.stage,
                        market.decision_id,
                        market.parent_public_hash,
                    ),
                    market.selected_action_id,
                ));
            }
            draws.push(expected_exploration_draw(
                rng_context,
                context.exploration.epsilon_parts_per_million,
                R2MapExplorationDecisionIdentity::new(
                    turn_index,
                    u8::try_from(public_turn.market_decisions.len()).map_err(|_| {
                        R2MapExperienceError::Exploration("draft decision ordinal exceeds u8")
                    })?,
                    MarketDecisionStage::Draft,
                    public_turn.draft_decision_id,
                    public_turn.draft_parent_public_hash,
                ),
                public_turn.draft_action_id,
            ));
        }
        game.apply(action)?;
    }
    Ok(draws)
}

fn expected_exploration_draw(
    rng_context: R2MapExplorationRngContext<'_>,
    epsilon_parts_per_million: u32,
    decision: R2MapExplorationDecisionIdentity,
    selected_action_id: IdentityHash,
) -> R2MapExplorationDraw {
    let explore_draw_u64 = r2_map_explore_draw_u64(rng_context, decision);
    let explored = u128::from(explore_draw_u64) * 1_000_000
        < u128::from(u64::MAX) * u128::from(epsilon_parts_per_million);
    R2MapExplorationDraw {
        turn_index: decision.turn_index,
        ordinal: decision.ordinal,
        stage: decision.stage,
        decision_id: decision.decision_id,
        parent_public_hash: decision.parent_public_hash,
        selected_action_id,
        explore_draw_u64,
        action_draw_u64: explored
            .then(|| r2_map_action_draw_u64(rng_context, decision, selected_action_id)),
        explored,
    }
}

impl R2MapGameRecord {
    pub fn from_match(
        context: R2MapRecordContext,
        result: &MatchResult,
        exploration_draws: &[R2MapExplorationDraw],
    ) -> Result<Self, R2MapExperienceError> {
        if result.seed != result.replay.seed {
            return Err(R2MapExperienceError::IdentityMismatch("match seed"));
        }
        context.identity.validate(result.seed)?;
        validate_context(&context, result)?;
        if exploration_draws != expected_r2_map_exploration_draws(&context, &result.replay)? {
            return Err(R2MapExperienceError::Exploration(
                "runner exploration evidence differs from replay-derived counter draws",
            ));
        }

        let rng_context =
            R2MapExplorationRngContext::new(&context.identity, result.seed, context.focal_seat);
        let mut game = GameState::new(result.replay.config, result.replay.seed)?;
        let mut decisions = Vec::with_capacity(result.replay.turns.len());
        let mut draw_index = 0usize;
        for (turn, action) in result.replay.turns.iter().enumerate() {
            let turn_index = u16::try_from(turn)
                .map_err(|_| R2MapExperienceError::InvalidRecord("turn index exceeds u16"))?;
            let seat = u8::try_from(game.current_player())
                .map_err(|_| R2MapExperienceError::InvalidRecord("seat exceeds u8"))?;
            let parent_public_hash = *game.public_state().canonical_hash().as_bytes();
            let position_id =
                position_identity(context.identity.game_id, turn_index, parent_public_hash);
            let action_id = action_identity(position_id, action)?;
            let mut public_turn = reconstruct_r2_map_public_turn(&game, turn_index, action)?;
            let before = game.boards()[usize::from(seat)].nature_tokens();
            let selected_afterstate = game.preview_public_afterstate(action)?;
            let next = game.transition(action)?;
            let after = next.boards()[usize::from(seat)].nature_tokens();
            let spent_on_wipes = u8::try_from(action.wildlife_wipes.len())
                .map_err(|_| R2MapExperienceError::PineconeConservation)?;
            let spent_on_independent_draft =
                u8::from(matches!(action.draft, DraftChoice::Independent { .. }));
            let total_spent = spent_on_wipes + spent_on_independent_draft;
            let earned = i16::from(after) + i16::from(total_spent) - i16::from(before);
            let earned =
                u8::try_from(earned).map_err(|_| R2MapExperienceError::PineconeConservation)?;
            let pinecones = R2MapPineconeEvent {
                before,
                earned,
                spent_on_wipes,
                spent_on_independent_draft,
                after,
            };
            pinecones.validate()?;

            let exploration = if context.exploration.enabled && seat == context.focal_seat {
                for market in &mut public_turn.market_decisions {
                    let draw = exploration_draws.get(draw_index).ok_or(
                        R2MapExperienceError::Exploration(
                            "missing focal market-decision exploration draw",
                        ),
                    )?;
                    draw_index += 1;
                    market.exploration = Some(exploration_trace(
                        rng_context,
                        context.exploration.epsilon_parts_per_million,
                        *draw,
                        R2MapExplorationDecisionIdentity::new(
                            turn_index,
                            market.ordinal,
                            market.stage,
                            market.decision_id,
                            market.parent_public_hash,
                        ),
                        market.selected_action_id,
                    )?);
                }
                let draft_ordinal =
                    u8::try_from(public_turn.market_decisions.len()).map_err(|_| {
                        R2MapExperienceError::Exploration("draft decision ordinal exceeds u8")
                    })?;
                let draw =
                    exploration_draws
                        .get(draw_index)
                        .ok_or(R2MapExperienceError::Exploration(
                            "missing focal draft-decision exploration draw",
                        ))?;
                draw_index += 1;
                Some(exploration_trace(
                    rng_context,
                    context.exploration.epsilon_parts_per_million,
                    *draw,
                    R2MapExplorationDecisionIdentity::new(
                        turn_index,
                        draft_ordinal,
                        MarketDecisionStage::Draft,
                        public_turn.draft_decision_id,
                        public_turn.draft_parent_public_hash,
                    ),
                    public_turn.draft_action_id,
                )?)
            } else {
                None
            };

            decisions.push(R2MapDecisionRecord {
                turn_index,
                seat,
                position_id,
                action_id,
                bundled_action_contract_id: public_turn.bundled_action_id,
                parent_public_hash,
                afterstate_public_hash: *selected_afterstate.canonical_hash().as_bytes(),
                market_decisions: public_turn.market_decisions,
                draft_decision_id: public_turn.draft_decision_id,
                draft_parent_public_hash: public_turn.draft_parent_public_hash,
                draft_action_id: public_turn.draft_action_id,
                draft_legal_action_count: public_turn.draft_legal_action_count,
                draft_ordered_action_ids_blake3: public_turn.draft_ordered_action_ids_blake3,
                pinecones,
                exploration,
            });
            game = next;
        }
        if draw_index != exploration_draws.len() {
            return Err(R2MapExperienceError::Exploration(
                "unused exploration draws remain",
            ));
        }
        let terminal_replay_hash = *game.canonical_hash().as_bytes();
        if result.replay.final_state_hash != Some(terminal_replay_hash) {
            return Err(R2MapExperienceError::HashMismatch("sealed replay"));
        }
        let record = Self {
            schema_version: R2_MAP_EXPERIENCE_SCHEMA_VERSION,
            collection_kind: context.collection_kind,
            identity: context.identity,
            config: result.replay.config,
            seed: result.seed,
            seed_purpose: context.seed_purpose,
            focal_seat: context.focal_seat,
            focal_checkpoint_hash: context
                .seats
                .get(usize::from(context.focal_seat))
                .and_then(|policy| policy.checkpoint_hash),
            seats: context.seats,
            rng: context.rng,
            exploration: context.exploration,
            decisions,
            scores: result.scores.clone(),
            terminal_public_state_hash: *game.public_state().canonical_hash().as_bytes(),
            terminal_replay_hash,
            protocols: context.protocols,
            replay: result.replay.clone(),
        };
        record.validate()?;
        Ok(record)
    }

    pub fn validate(&self) -> Result<(), R2MapExperienceError> {
        if self.schema_version != R2_MAP_EXPERIENCE_SCHEMA_VERSION {
            return Err(R2MapExperienceError::UnsupportedSchema(self.schema_version));
        }
        if self.config != GameConfig::research_aaaaa(4)?
            || self.replay.config != self.config
            || self.replay.seed != self.seed
        {
            return Err(R2MapExperienceError::InvalidRecord(
                "trajectory is not a four-player Card A no-bonus game",
            ));
        }
        self.identity.validate(self.seed)?;
        validate_seed_contract(
            self.collection_kind,
            self.seed_purpose,
            &self.identity,
            self.seed,
        )?;
        self.protocols.validate()?;
        self.rng.validate()?;
        self.exploration
            .validate(self.collection_kind, self.identity.iteration)?;
        validate_seat_contract(
            self.collection_kind,
            self.identity.global_game_index,
            self.focal_seat,
            &self.seats,
        )?;
        let expected_focal_hash = self
            .seats
            .get(usize::from(self.focal_seat))
            .and_then(|policy| policy.checkpoint_hash);
        if self.focal_checkpoint_hash != expected_focal_hash {
            return Err(R2MapExperienceError::IdentityMismatch("focal checkpoint"));
        }

        let rng_context =
            R2MapExplorationRngContext::new(&self.identity, self.seed, self.focal_seat);
        let mut game = GameState::new(self.config, self.seed)?;
        if self.decisions.len() != self.replay.turns.len() {
            return Err(R2MapExperienceError::InvalidRecord(
                "decision and replay action counts differ",
            ));
        }
        let mut focal_decisions = 0usize;
        for (turn, (decision, action)) in self.decisions.iter().zip(&self.replay.turns).enumerate()
        {
            let turn_index = u16::try_from(turn)
                .map_err(|_| R2MapExperienceError::InvalidRecord("turn exceeds u16"))?;
            let seat = u8::try_from(game.current_player())
                .map_err(|_| R2MapExperienceError::InvalidRecord("seat exceeds u8"))?;
            let parent_public_hash = *game.public_state().canonical_hash().as_bytes();
            let expected_position_id =
                position_identity(self.identity.game_id, turn_index, parent_public_hash);
            let expected_action_id = action_identity(expected_position_id, action)?;
            let expected_public_turn = reconstruct_r2_map_public_turn(&game, turn_index, action)?;
            let before = game.boards()[usize::from(seat)].nature_tokens();
            let selected_afterstate = game.preview_public_afterstate(action)?;
            let next = game.transition(action)?;
            let after = next.boards()[usize::from(seat)].nature_tokens();
            let spent_on_wipes = u8::try_from(action.wildlife_wipes.len())
                .map_err(|_| R2MapExperienceError::PineconeConservation)?;
            let spent_on_independent_draft =
                u8::from(matches!(action.draft, DraftChoice::Independent { .. }));
            let total_spent = spent_on_wipes + spent_on_independent_draft;
            let earned =
                u8::try_from(i16::from(after) + i16::from(total_spent) - i16::from(before))
                    .map_err(|_| R2MapExperienceError::PineconeConservation)?;
            let expected_pinecones = R2MapPineconeEvent {
                before,
                earned,
                spent_on_wipes,
                spent_on_independent_draft,
                after,
            };
            if decision.turn_index != turn_index
                || decision.seat != seat
                || decision.parent_public_hash != parent_public_hash
                || decision.position_id != expected_position_id
                || decision.action_id != expected_action_id
                || decision.bundled_action_contract_id != expected_public_turn.bundled_action_id
                || decision.afterstate_public_hash
                    != *selected_afterstate.canonical_hash().as_bytes()
                || decision.draft_decision_id != expected_public_turn.draft_decision_id
                || decision.draft_parent_public_hash
                    != expected_public_turn.draft_parent_public_hash
                || decision.draft_action_id != expected_public_turn.draft_action_id
                || decision.draft_legal_action_count
                    != expected_public_turn.draft_legal_action_count
                || decision.draft_ordered_action_ids_blake3
                    != expected_public_turn.draft_ordered_action_ids_blake3
                || decision.market_decisions.len() != expected_public_turn.market_decisions.len()
                || decision.pinecones != expected_pinecones
            {
                return Err(R2MapExperienceError::ReplayMismatch { turn });
            }
            for (actual, mut expected) in decision
                .market_decisions
                .iter()
                .zip(expected_public_turn.market_decisions)
            {
                expected.exploration = None;
                let mut observed = actual.clone();
                observed.exploration = None;
                if observed != expected {
                    return Err(R2MapExperienceError::ReplayMismatch { turn });
                }
            }
            decision.pinecones.validate()?;
            if seat == self.focal_seat {
                focal_decisions += 1;
                match (&decision.exploration, self.exploration.enabled) {
                    (Some(trace), true) => {
                        let draft_ordinal =
                            u8::try_from(decision.market_decisions.len()).map_err(|_| {
                                R2MapExperienceError::Exploration(
                                    "draft decision ordinal exceeds u8",
                                )
                            })?;
                        validate_exploration_trace(
                            rng_context,
                            self.exploration.epsilon_parts_per_million,
                            R2MapExplorationDecisionIdentity::new(
                                turn_index,
                                draft_ordinal,
                                MarketDecisionStage::Draft,
                                decision.draft_decision_id,
                                decision.draft_parent_public_hash,
                            ),
                            decision.draft_action_id,
                            trace,
                        )?;
                    }
                    (None, false) => {}
                    _ => {
                        return Err(R2MapExperienceError::Exploration(
                            "focal exploration trace presence is wrong",
                        ));
                    }
                }
                if self.exploration.enabled {
                    for market in &decision.market_decisions {
                        let trace = market.exploration.as_ref().ok_or(
                            R2MapExperienceError::Exploration(
                                "focal market decision omitted its RNG trace",
                            ),
                        )?;
                        validate_exploration_trace(
                            rng_context,
                            self.exploration.epsilon_parts_per_million,
                            R2MapExplorationDecisionIdentity::new(
                                turn_index,
                                market.ordinal,
                                market.stage,
                                market.decision_id,
                                market.parent_public_hash,
                            ),
                            market.selected_action_id,
                            trace,
                        )?;
                    }
                } else if decision
                    .market_decisions
                    .iter()
                    .any(|market| market.exploration.is_some())
                {
                    return Err(R2MapExperienceError::Exploration(
                        "disabled focal market decisions carry RNG traces",
                    ));
                }
            } else if decision.exploration.is_some() {
                return Err(R2MapExperienceError::Exploration(
                    "opponent decisions cannot carry focal exploration draws",
                ));
            } else if decision
                .market_decisions
                .iter()
                .any(|market| market.exploration.is_some())
            {
                return Err(R2MapExperienceError::Exploration(
                    "opponent market decisions cannot carry focal exploration draws",
                ));
            }
            game = next;
        }
        if self.decisions.len() != 80 || focal_decisions != 20 || !game.is_game_over() {
            return Err(R2MapExperienceError::InvalidRecord(
                "complete four-player trajectory must contain 80 decisions and 20 focal turns",
            ));
        }
        let terminal_replay_hash = *game.canonical_hash().as_bytes();
        if self.replay.final_state_hash != Some(terminal_replay_hash)
            || self.terminal_replay_hash != terminal_replay_hash
            || self.terminal_public_state_hash != *game.public_state().canonical_hash().as_bytes()
        {
            return Err(R2MapExperienceError::HashMismatch("terminal state"));
        }
        let scores = score_game(&game);
        if self.scores != scores || self.scores.len() != 4 {
            return Err(R2MapExperienceError::ScoreMismatch);
        }
        for score in &self.scores {
            validate_score_identity(*score)?;
        }
        Ok(())
    }

    pub fn extract_primary_examples(
        &self,
    ) -> Result<Vec<R2MapPrimaryExample>, R2MapExperienceError> {
        self.extract_primary_examples_for_seat(None)
    }

    /// Extract exactly one deterministic seat from every four-player game.
    ///
    /// Bootstrap collection durably stores all four seats, but expert iteration
    /// controls exactly one newest-model seat.  Training on the same one-seat
    /// projection avoids four correlated copies of an all-greedy trajectory and
    /// makes bootstrap and iterative rounds share one sampling contract.
    pub fn extract_focal_seat_examples(
        &self,
        focal_seat: u8,
    ) -> Result<Vec<R2MapPrimaryExample>, R2MapExperienceError> {
        if focal_seat >= 4 {
            return Err(R2MapExperienceError::InvalidRecord(
                "focal example seat exceeds the four-player game",
            ));
        }
        self.extract_primary_examples_for_seat(Some(focal_seat))
    }

    fn extract_primary_examples_for_seat(
        &self,
        forced_focal_seat: Option<u8>,
    ) -> Result<Vec<R2MapPrimaryExample>, R2MapExperienceError> {
        self.validate()?;
        let retain_all = self.collection_kind == R2MapCollectionKind::Bootstrap;
        let mut game = GameState::new(self.config, self.seed)?;
        let mut examples = Vec::with_capacity(if retain_all && forced_focal_seat.is_none() {
            80
        } else {
            20
        });
        for (decision, action) in self.decisions.iter().zip(&self.replay.turns) {
            let seat = game.current_player();
            let parent = game.public_state();
            let selected_afterstate = game.preview_public_afterstate(action)?;
            let next = game.transition(action)?;
            let retain = forced_focal_seat.map_or(
                retain_all || seat == usize::from(self.focal_seat),
                |focal| seat == usize::from(focal),
            );
            if retain {
                let current = score_board(
                    &selected_afterstate.boards()[seat],
                    self.config.scoring_cards,
                );
                let terminal = self.scores[seat];
                let current_components = score_components(current);
                let terminal_components = score_components(terminal);
                let mut residual = [0i16; R2_MAP_COMPONENT_COUNT];
                for index in 0..R2_MAP_COMPONENT_COUNT {
                    residual[index] = i16::try_from(
                        i32::from(terminal_components[index])
                            - i32::from(current_components[index]),
                    )
                    .map_err(|_| {
                        R2MapExperienceError::InvalidRecord("score residual exceeds i16")
                    })?;
                }
                examples.push(R2MapPrimaryExample {
                    game_id: self.identity.game_id,
                    turn_index: decision.turn_index,
                    seat: decision.seat,
                    position_id: decision.position_id,
                    action_id: decision.action_id,
                    parent,
                    action: action.clone(),
                    afterstate: selected_afterstate,
                    current: current_components,
                    residual,
                    terminal: terminal_components,
                });
            }
            game = next;
        }
        let expected = if retain_all && forced_focal_seat.is_none() {
            80
        } else {
            20
        };
        if examples.len() != expected {
            return Err(R2MapExperienceError::InvalidRecord(
                "primary example retention count is wrong",
            ));
        }
        Ok(examples)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct R2MapPrimaryExample {
    pub game_id: IdentityHash,
    pub turn_index: u16,
    pub seat: u8,
    pub position_id: IdentityHash,
    pub action_id: IdentityHash,
    pub parent: PublicGameState,
    pub action: TurnAction,
    pub afterstate: PublicGameState,
    pub current: [u16; R2_MAP_COMPONENT_COUNT],
    pub residual: [i16; R2_MAP_COMPONENT_COUNT],
    pub terminal: [u16; R2_MAP_COMPONENT_COUNT],
}

impl R2MapPrimaryExample {
    pub fn validate(&self) -> Result<(), R2MapExperienceError> {
        for index in 0..R2_MAP_COMPONENT_COUNT {
            if i32::from(self.current[index]) + i32::from(self.residual[index])
                != i32::from(self.terminal[index])
            {
                return Err(R2MapExperienceError::ScoreMismatch);
            }
        }
        Ok(())
    }
}

pub fn focal_seat_for_game(global_game_index: u64) -> u8 {
    (global_game_index % 4) as u8
}

/// Produces a host-independent seed from the global lease index. Host leases
/// must be disjoint, so a reassigned or duplicated index cannot quietly produce
/// a different game merely because a different machine claimed it.
pub fn r2_map_game_seed(
    campaign_id: &str,
    purpose: R2MapSeedPurpose,
    iteration: u32,
    global_game_index: u64,
) -> GameSeed {
    GameSeed(hash_parts(
        b"r2-map-game-seed-v1",
        &[
            campaign_id.as_bytes(),
            purpose.id().as_bytes(),
            &iteration.to_le_bytes(),
            &global_game_index.to_le_bytes(),
        ],
    ))
}

pub fn validate_r2_map_seed_leases(leases: &[R2MapSeedLease]) -> Result<(), R2MapExperienceError> {
    for lease in leases {
        lease.validate()?;
    }
    for (index, left) in leases.iter().enumerate() {
        for right in &leases[index + 1..] {
            let same_domain = left.campaign_id == right.campaign_id
                && left.iteration == right.iteration
                && left.purpose == right.purpose;
            let overlaps = left.first_game_index < right.end_game_index()?
                && right.first_game_index < left.end_game_index()?;
            if same_domain && overlaps {
                return Err(R2MapExperienceError::DuplicateIdentity("seed lease"));
            }
        }
    }
    Ok(())
}

/// Deterministically assigns exactly one newest checkpoint and up to three
/// distinct historical opponents. Greedy fills any unoccupied opponent seats.
pub fn assign_iterative_seats(
    identity: &R2MapGameIdentity,
    seed: GameSeed,
    newest: R2MapPolicyIdentity,
    opponent_pool: &[R2MapPolicyIdentity],
) -> Result<Vec<R2MapPolicyIdentity>, R2MapExperienceError> {
    if newest.role != R2MapPolicyRole::Newest {
        return Err(R2MapExperienceError::SeatContract(
            "focal policy is not marked newest",
        ));
    }
    newest.validate()?;
    identity.validate(seed)?;
    let mut unique = HashSet::new();
    let mut ranked = Vec::new();
    for opponent in opponent_pool {
        opponent.validate()?;
        if !matches!(
            opponent.role,
            R2MapPolicyRole::Historical | R2MapPolicyRole::Greedy
        ) {
            return Err(R2MapExperienceError::SeatContract(
                "opponent pool contains a newest policy",
            ));
        }
        if opponent.checkpoint_hash == newest.checkpoint_hash {
            return Err(R2MapExperienceError::SeatContract(
                "newest checkpoint appears in historical pool",
            ));
        }
        let key = postcard::to_allocvec(opponent)?;
        if unique.insert(key.clone()) {
            let rank = hash_parts(
                R2_MAP_ASSIGNMENT_RNG_DOMAIN.as_bytes(),
                &[&identity.game_id, &seed.0, &key],
            );
            ranked.push((rank, opponent.clone()));
        }
    }
    ranked.sort_by_key(|(rank, _)| *rank);
    let mut opponents: Vec<_> = ranked
        .into_iter()
        .take(3)
        .map(|(_, policy)| policy)
        .collect();
    while opponents.len() < 3 {
        opponents.push(R2MapPolicyIdentity::greedy());
    }
    let focal_seat = focal_seat_for_game(identity.global_game_index);
    let mut seats = Vec::with_capacity(4);
    let mut opponent = opponents.into_iter();
    for seat in 0..4 {
        seats.push(if seat == usize::from(focal_seat) {
            newest.clone()
        } else {
            opponent.next().expect("exactly three opponent seats")
        });
    }
    validate_seat_contract(
        R2MapCollectionKind::IterativeTraining,
        identity.global_game_index,
        focal_seat,
        &seats,
    )?;
    Ok(seats)
}

pub fn validate_r2_map_record_batch(
    records: &[R2MapGameRecord],
) -> Result<(), R2MapExperienceError> {
    let mut games = HashSet::new();
    let mut seeds = HashSet::new();
    let mut positions = HashSet::new();
    let mut actions = HashSet::new();
    let mut ordered: Vec<_> = records.iter().collect();
    ordered.sort_by_key(|record| record.identity.global_game_index);
    if let Some(first) = ordered.first()
        && ordered.iter().any(|record| {
            record.identity.campaign_id != first.identity.campaign_id
                || record.identity.iteration != first.identity.iteration
                || record.collection_kind != first.collection_kind
                || record.seed_purpose != first.seed_purpose
        })
    {
        return Err(R2MapExperienceError::InvalidRecord(
            "batch mixes campaign, iteration, collection, or seed domains",
        ));
    }
    for (index, record) in ordered.iter().enumerate() {
        record.validate()?;
        if !games.insert(record.identity.game_id) {
            return Err(R2MapExperienceError::DuplicateIdentity("game"));
        }
        if !seeds.insert(record.seed) {
            return Err(R2MapExperienceError::DuplicateIdentity("seed"));
        }
        for decision in &record.decisions {
            if !positions.insert(decision.position_id) {
                return Err(R2MapExperienceError::DuplicateIdentity("position"));
            }
            if !actions.insert(decision.action_id) {
                return Err(R2MapExperienceError::DuplicateIdentity("action"));
            }
        }
        if index > 0 {
            let previous = ordered[index - 1].identity.global_game_index;
            if record.identity.global_game_index != previous + 1 {
                return Err(R2MapExperienceError::InvalidRecord(
                    "batch game indices must form a completed contiguous prefix",
                ));
            }
        }
        let mut counts = [0usize; 4];
        for prefix_record in &ordered[..=index] {
            counts[usize::from(prefix_record.focal_seat)] += 1;
        }
        let minimum = *counts.iter().min().expect("four seats");
        let maximum = *counts.iter().max().expect("four seats");
        if maximum - minimum > 1 {
            return Err(R2MapExperienceError::SeatContract(
                "focal seats are not balanced over the completed prefix",
            ));
        }
    }
    Ok(())
}

pub fn write_r2_map_record(
    path: &Path,
    record: &R2MapGameRecord,
) -> Result<(), R2MapExperienceError> {
    record.validate()?;
    let payload = postcard::to_allocvec(record)?;
    let payload_hash = blake3::hash(&payload);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temp = path.with_extension("r2xp.tmp");
    let mut writer = BufWriter::new(File::create(&temp)?);
    writer.write_all(R2_MAP_EXPERIENCE_MAGIC)?;
    writer.write_all(&R2_MAP_EXPERIENCE_SCHEMA_VERSION.to_le_bytes())?;
    writer.write_all(&(payload.len() as u64).to_le_bytes())?;
    writer.write_all(payload_hash.as_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(temp, path)?;
    sync_parent_directory(path)?;
    Ok(())
}

pub fn read_r2_map_record(path: &Path) -> Result<R2MapGameRecord, R2MapExperienceError> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut header = [0u8; R2_MAP_EXPERIENCE_HEADER_SIZE];
    reader.read_exact(&mut header)?;
    if &header[..8] != R2_MAP_EXPERIENCE_MAGIC {
        return Err(R2MapExperienceError::InvalidFile("magic"));
    }
    let version = u16::from_le_bytes([header[8], header[9]]);
    if version != R2_MAP_EXPERIENCE_SCHEMA_VERSION {
        return Err(R2MapExperienceError::UnsupportedSchema(version));
    }
    let payload_len = u64::from_le_bytes(header[10..18].try_into().expect("fixed header"));
    let expected_file_len = R2_MAP_EXPERIENCE_HEADER_SIZE as u64 + payload_len;
    if fs::metadata(path)?.len() != expected_file_len {
        return Err(R2MapExperienceError::InvalidFile("length"));
    }
    let payload_len = usize::try_from(payload_len)
        .map_err(|_| R2MapExperienceError::InvalidFile("payload length"))?;
    let mut payload = vec![0u8; payload_len];
    reader.read_exact(&mut payload)?;
    if header[18..50] != *blake3::hash(&payload).as_bytes() {
        return Err(R2MapExperienceError::InvalidFile("checksum"));
    }
    let record: R2MapGameRecord = postcard::from_bytes(&payload)?;
    record.validate()?;
    Ok(record)
}

pub fn write_r2_map_shard(
    path: &Path,
    records: &[R2MapGameRecord],
) -> Result<(), R2MapExperienceError> {
    if records.is_empty() {
        return Err(R2MapExperienceError::InvalidFile("empty shard"));
    }
    validate_r2_map_record_batch(records)?;
    let mut ordered: Vec<_> = records.iter().collect();
    ordered.sort_by_key(|record| record.identity.global_game_index);
    let first_game_index = ordered[0].identity.global_game_index;
    let next_game_index = ordered
        .last()
        .expect("non-empty records")
        .identity
        .global_game_index
        .checked_add(1)
        .ok_or(R2MapExperienceError::InvalidFile("shard range overflow"))?;
    let mut payload = Vec::new();
    for record in ordered {
        let bytes = postcard::to_allocvec(record)?;
        let length = u32::try_from(bytes.len())
            .map_err(|_| R2MapExperienceError::InvalidFile("record exceeds u32"))?;
        payload.extend_from_slice(&length.to_le_bytes());
        payload.extend_from_slice(&bytes);
    }
    let payload_hash = blake3::hash(&payload);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temp = path.with_extension("r2sh.tmp");
    let mut writer = BufWriter::new(File::create(&temp)?);
    writer.write_all(R2_MAP_SHARD_MAGIC)?;
    writer.write_all(&R2_MAP_EXPERIENCE_SCHEMA_VERSION.to_le_bytes())?;
    writer.write_all(&(R2_MAP_SHARD_HEADER_SIZE as u16).to_le_bytes())?;
    writer.write_all(
        &u32::try_from(records.len())
            .map_err(|_| R2MapExperienceError::InvalidFile("record count exceeds u32"))?
            .to_le_bytes(),
    )?;
    writer.write_all(&first_game_index.to_le_bytes())?;
    writer.write_all(&next_game_index.to_le_bytes())?;
    writer.write_all(payload_hash.as_bytes())?;
    debug_assert_eq!(8 + 2 + 2 + 4 + 8 + 8 + 32, R2_MAP_SHARD_HEADER_SIZE);
    writer.write_all(&payload)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(&temp, path)?;
    sync_parent_directory(path)?;
    Ok(())
}

pub fn read_r2_map_shard(path: &Path) -> Result<Vec<R2MapGameRecord>, R2MapExperienceError> {
    read_r2_map_shard_inner(path, true)
}

/// Read a framed shard whose exact bytes and full semantic validation are
/// already bound by a collector completion/aggregate receipt.
///
/// Header, payload checksum, decoding, identity uniqueness, and contiguous
/// range checks still run. Expensive replay/scoring validation is deliberately
/// omitted so downstream indexing does not repeat the collector gate.
pub fn read_r2_map_shard_after_semantic_validation(
    path: &Path,
) -> Result<Vec<R2MapGameRecord>, R2MapExperienceError> {
    read_r2_map_shard_inner(path, false)
}

fn read_r2_map_shard_inner(
    path: &Path,
    validate_semantics: bool,
) -> Result<Vec<R2MapGameRecord>, R2MapExperienceError> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut header = [0u8; R2_MAP_SHARD_HEADER_SIZE];
    reader.read_exact(&mut header)?;
    if &header[..8] != R2_MAP_SHARD_MAGIC {
        return Err(R2MapExperienceError::InvalidFile("shard magic"));
    }
    let version = u16::from_le_bytes(header[8..10].try_into().expect("fixed header"));
    let header_size = u16::from_le_bytes(header[10..12].try_into().expect("fixed header"));
    if version != R2_MAP_EXPERIENCE_SCHEMA_VERSION
        || usize::from(header_size) != R2_MAP_SHARD_HEADER_SIZE
    {
        return Err(R2MapExperienceError::UnsupportedSchema(version));
    }
    let record_count =
        u32::from_le_bytes(header[12..16].try_into().expect("fixed header")) as usize;
    let first_game_index = u64::from_le_bytes(header[16..24].try_into().expect("fixed header"));
    let next_game_index = u64::from_le_bytes(header[24..32].try_into().expect("fixed header"));
    if record_count == 0 || next_game_index < first_game_index {
        return Err(R2MapExperienceError::InvalidFile("shard range"));
    }
    let payload_len = fs::metadata(path)?
        .len()
        .checked_sub(R2_MAP_SHARD_HEADER_SIZE as u64)
        .ok_or(R2MapExperienceError::InvalidFile("shard length"))?;
    let payload_len = usize::try_from(payload_len)
        .map_err(|_| R2MapExperienceError::InvalidFile("shard payload length"))?;
    let mut payload = vec![0u8; payload_len];
    reader.read_exact(&mut payload)?;
    if header[32..64] != *blake3::hash(&payload).as_bytes() {
        return Err(R2MapExperienceError::InvalidFile("shard checksum"));
    }
    let mut cursor = 0usize;
    let mut records = Vec::with_capacity(record_count);
    for _ in 0..record_count {
        let end_length = cursor
            .checked_add(4)
            .filter(|end| *end <= payload.len())
            .ok_or(R2MapExperienceError::InvalidFile("truncated record length"))?;
        let record_len = u32::from_le_bytes(
            payload[cursor..end_length]
                .try_into()
                .expect("four-byte slice"),
        ) as usize;
        cursor = end_length;
        let end_record = cursor
            .checked_add(record_len)
            .filter(|end| *end <= payload.len())
            .ok_or(R2MapExperienceError::InvalidFile("truncated record"))?;
        let record: R2MapGameRecord = postcard::from_bytes(&payload[cursor..end_record])?;
        records.push(record);
        cursor = end_record;
    }
    if cursor != payload.len() {
        return Err(R2MapExperienceError::InvalidFile("trailing shard payload"));
    }
    if validate_semantics {
        validate_r2_map_record_batch(&records)?;
    } else {
        validate_r2_map_record_batch_identities(&records)?;
    }
    if records[0].identity.global_game_index != first_game_index
        || records
            .last()
            .expect("validated non-empty shard")
            .identity
            .global_game_index
            .checked_add(1)
            != Some(next_game_index)
    {
        return Err(R2MapExperienceError::InvalidFile(
            "shard header range does not match records",
        ));
    }
    Ok(records)
}

fn validate_r2_map_record_batch_identities(
    records: &[R2MapGameRecord],
) -> Result<(), R2MapExperienceError> {
    let Some(first) = records.first() else {
        return Err(R2MapExperienceError::InvalidRecord("batch is empty"));
    };
    let mut games = HashSet::new();
    let mut seeds = HashSet::new();
    let mut positions = HashSet::new();
    let mut actions = HashSet::new();
    for (offset, record) in records.iter().enumerate() {
        if record.identity.campaign_id != first.identity.campaign_id
            || record.identity.iteration != first.identity.iteration
            || record.collection_kind != first.collection_kind
            || record.seed_purpose != first.seed_purpose
            || record.identity.global_game_index
                != first.identity.global_game_index
                    + u64::try_from(offset).map_err(|_| {
                        R2MapExperienceError::InvalidRecord("batch offset exceeds u64")
                    })?
            || !games.insert(record.identity.game_id)
            || !seeds.insert(record.seed)
        {
            return Err(R2MapExperienceError::InvalidRecord(
                "batch identity or contiguous range differs",
            ));
        }
        for decision in &record.decisions {
            if !positions.insert(decision.position_id) || !actions.insert(decision.action_id) {
                return Err(R2MapExperienceError::DuplicateIdentity(
                    "position or action",
                ));
            }
        }
    }
    Ok(())
}

fn sync_parent_directory(path: &Path) -> Result<(), R2MapExperienceError> {
    if let Some(parent) = path.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

fn validate_context(
    context: &R2MapRecordContext,
    result: &MatchResult,
) -> Result<(), R2MapExperienceError> {
    if result.replay.config != GameConfig::research_aaaaa(4)?
        || result.strategies.len() != 4
        || context.seats.len() != 4
        || result.strategies
            != context
                .seats
                .iter()
                .map(|policy| policy.policy_id.clone())
                .collect::<Vec<_>>()
    {
        return Err(R2MapExperienceError::SeatContract(
            "match strategy identities do not match record context",
        ));
    }
    context.protocols.validate()?;
    context.rng.validate()?;
    validate_seed_contract(
        context.collection_kind,
        context.seed_purpose,
        &context.identity,
        result.seed,
    )?;
    context
        .exploration
        .validate(context.collection_kind, context.identity.iteration)?;
    validate_seat_contract(
        context.collection_kind,
        context.identity.global_game_index,
        context.focal_seat,
        &context.seats,
    )
}

fn validate_seed_contract(
    collection_kind: R2MapCollectionKind,
    purpose: R2MapSeedPurpose,
    identity: &R2MapGameIdentity,
    seed: GameSeed,
) -> Result<(), R2MapExperienceError> {
    let purpose_allowed = match collection_kind {
        R2MapCollectionKind::Bootstrap => purpose == R2MapSeedPurpose::Bootstrap,
        R2MapCollectionKind::IterativeTraining => purpose == R2MapSeedPurpose::Generation,
        R2MapCollectionKind::Benchmark => matches!(
            purpose,
            R2MapSeedPurpose::LongitudinalBenchmark
                | R2MapSeedPurpose::CandidateGate
                | R2MapSeedPurpose::ProtectedFinal
        ),
    };
    if !purpose_allowed {
        return Err(R2MapExperienceError::InvalidIdentity(
            "seed purpose does not match collection kind",
        ));
    }
    let expected = r2_map_game_seed(
        &identity.campaign_id,
        purpose,
        identity.iteration,
        identity.global_game_index,
    );
    if seed != expected {
        return Err(R2MapExperienceError::IdentityMismatch("game seed"));
    }
    Ok(())
}

fn validate_seat_contract(
    collection_kind: R2MapCollectionKind,
    global_game_index: u64,
    focal_seat: u8,
    seats: &[R2MapPolicyIdentity],
) -> Result<(), R2MapExperienceError> {
    if seats.len() != 4 || usize::from(focal_seat) >= seats.len() {
        return Err(R2MapExperienceError::SeatContract(
            "four seats and a valid focal seat are required",
        ));
    }
    for policy in seats {
        policy.validate()?;
    }
    if focal_seat != focal_seat_for_game(global_game_index) {
        return Err(R2MapExperienceError::SeatContract(
            "focal seat does not match deterministic rotation",
        ));
    }
    let newest: Vec<_> = seats
        .iter()
        .enumerate()
        .filter(|(_, policy)| policy.role == R2MapPolicyRole::Newest)
        .collect();
    match collection_kind {
        R2MapCollectionKind::Bootstrap => {
            if !newest.is_empty()
                || seats
                    .iter()
                    .any(|policy| policy.role != R2MapPolicyRole::Greedy)
            {
                return Err(R2MapExperienceError::SeatContract(
                    "bootstrap must contain four greedy seats and no newest checkpoint",
                ));
            }
        }
        R2MapCollectionKind::IterativeTraining => {
            if newest.len() != 1 || newest[0].0 != usize::from(focal_seat) {
                return Err(R2MapExperienceError::SeatContract(
                    "exactly one newest checkpoint must occupy the focal seat",
                ));
            }
            let newest_hash = newest[0]
                .1
                .checkpoint_hash
                .expect("validated newest policy has hash");
            if seats.iter().enumerate().any(|(seat, policy)| {
                seat != usize::from(focal_seat) && policy.checkpoint_hash == Some(newest_hash)
            }) {
                return Err(R2MapExperienceError::SeatContract(
                    "newest checkpoint is duplicated in an opponent seat",
                ));
            }
            if seats.iter().enumerate().any(|(seat, policy)| {
                seat != usize::from(focal_seat)
                    && !matches!(
                        policy.role,
                        R2MapPolicyRole::Historical | R2MapPolicyRole::Greedy
                    )
            }) {
                return Err(R2MapExperienceError::SeatContract(
                    "opponent seat is not historical or greedy",
                ));
            }
        }
        R2MapCollectionKind::Benchmark => {
            if newest
                .iter()
                .any(|(seat, _)| *seat != usize::from(focal_seat))
            {
                return Err(R2MapExperienceError::SeatContract(
                    "benchmark opponents cannot use the newest checkpoint role",
                ));
            }
            let focal_hash = seats[usize::from(focal_seat)].checkpoint_hash;
            if focal_hash.is_some()
                && seats.iter().enumerate().any(|(seat, policy)| {
                    seat != usize::from(focal_seat) && policy.checkpoint_hash == focal_hash
                })
            {
                return Err(R2MapExperienceError::SeatContract(
                    "benchmark focal checkpoint is duplicated in an opponent seat",
                ));
            }
            if seats.iter().enumerate().any(|(seat, policy)| {
                seat != usize::from(focal_seat)
                    && !matches!(
                        policy.role,
                        R2MapPolicyRole::Historical | R2MapPolicyRole::Greedy
                    )
            }) {
                return Err(R2MapExperienceError::SeatContract(
                    "benchmark opponent seat is not historical or greedy",
                ));
            }
        }
    }
    Ok(())
}

fn exploration_trace(
    rng_context: R2MapExplorationRngContext<'_>,
    epsilon_parts_per_million: u32,
    draw: R2MapExplorationDraw,
    decision: R2MapExplorationDecisionIdentity,
    selected_action_id: IdentityHash,
) -> Result<R2MapExplorationTrace, R2MapExperienceError> {
    if draw.turn_index >= 80
        || draw.turn_index != decision.turn_index
        || draw.ordinal != decision.ordinal
        || draw.stage != decision.stage
        || draw.decision_id != decision.decision_id
        || draw.parent_public_hash != decision.parent_public_hash
        || draw.selected_action_id != selected_action_id
        || draw.explored != draw.action_draw_u64.is_some()
    {
        return Err(R2MapExperienceError::Exploration(
            "exploration draw does not identify its exact public decision",
        ));
    }
    let decision_rng_key = exploration_decision_key(rng_context, decision);
    let selected_action_rng_key = hash_parts(
        b"r2-map-exploration-action-v2",
        &[&decision_rng_key, &selected_action_id],
    );
    let trace = R2MapExplorationTrace {
        ordinal: decision.ordinal,
        stage: decision.stage,
        decision_id: decision.decision_id,
        parent_public_hash: decision.parent_public_hash,
        selected_action_id,
        explore_draw_u64: draw.explore_draw_u64,
        action_draw_u64: draw.action_draw_u64,
        explored: draw.explored,
        decision_rng_key,
        selected_action_rng_key,
    };
    let gate_open = u128::from(trace.explore_draw_u64) * 1_000_000
        < u128::from(u64::MAX) * u128::from(epsilon_parts_per_million);
    if epsilon_parts_per_million > 1_000_000
        || trace.explored != gate_open
        || trace.explore_draw_u64
            != counter_draw_u64(b"r2-map-explore-counter-v2", decision_rng_key)
        || trace.action_draw_u64
            != trace
                .explored
                .then(|| counter_draw_u64(b"r2-map-action-counter-v2", selected_action_rng_key))
    {
        return Err(R2MapExperienceError::Exploration(
            "exploration draw differs from the counter-based RNG contract",
        ));
    }
    Ok(trace)
}

pub fn r2_map_explore_draw_u64(
    rng_context: R2MapExplorationRngContext<'_>,
    decision: R2MapExplorationDecisionIdentity,
) -> u64 {
    let decision_rng_key = exploration_decision_key(rng_context, decision);
    counter_draw_u64(b"r2-map-explore-counter-v2", decision_rng_key)
}

pub fn r2_map_action_draw_u64(
    rng_context: R2MapExplorationRngContext<'_>,
    decision: R2MapExplorationDecisionIdentity,
    action_id: IdentityHash,
) -> u64 {
    let decision_rng_key = exploration_decision_key(rng_context, decision);
    let selected_action_rng_key = hash_parts(
        b"r2-map-exploration-action-v2",
        &[&decision_rng_key, &action_id],
    );
    counter_draw_u64(b"r2-map-action-counter-v2", selected_action_rng_key)
}

fn exploration_decision_key(
    rng_context: R2MapExplorationRngContext<'_>,
    decision: R2MapExplorationDecisionIdentity,
) -> IdentityHash {
    hash_parts(
        R2_MAP_EXPLORATION_RNG_DOMAIN.as_bytes(),
        &[
            rng_context.identity.campaign_id.as_bytes(),
            &rng_context.identity.iteration.to_le_bytes(),
            &rng_context.seed.0,
            &decision.turn_index.to_le_bytes(),
            &[rng_context.focal_seat],
            &[decision.ordinal, decision.stage as u8],
            &decision.decision_id,
            &decision.parent_public_hash,
        ],
    )
}

fn validate_exploration_trace(
    rng_context: R2MapExplorationRngContext<'_>,
    epsilon_parts_per_million: u32,
    decision: R2MapExplorationDecisionIdentity,
    selected_action_id: IdentityHash,
    trace: &R2MapExplorationTrace,
) -> Result<(), R2MapExperienceError> {
    let reconstructed = exploration_trace(
        rng_context,
        epsilon_parts_per_million,
        R2MapExplorationDraw {
            turn_index: decision.turn_index,
            ordinal: trace.ordinal,
            stage: trace.stage,
            decision_id: trace.decision_id,
            parent_public_hash: trace.parent_public_hash,
            selected_action_id: trace.selected_action_id,
            explore_draw_u64: trace.explore_draw_u64,
            action_draw_u64: trace.action_draw_u64,
            explored: trace.explored,
        },
        decision,
        selected_action_id,
    )?;
    if &reconstructed != trace {
        return Err(R2MapExperienceError::Exploration(
            "focal RNG trace is not reproducibly identified",
        ));
    }
    Ok(())
}

fn counter_draw_u64(domain: &[u8], key: IdentityHash) -> u64 {
    u64::from_le_bytes(
        hash_parts(domain, &[&key])[..8]
            .try_into()
            .expect("eight bytes"),
    )
}

fn position_identity(
    game_id: IdentityHash,
    turn_index: u16,
    parent_public_hash: IdentityHash,
) -> IdentityHash {
    hash_parts(
        b"r2-map-position-identity-v1",
        &[&game_id, &turn_index.to_le_bytes(), &parent_public_hash],
    )
}

fn action_identity(
    position_id: IdentityHash,
    action: &TurnAction,
) -> Result<IdentityHash, R2MapExperienceError> {
    let action_bytes = postcard::to_allocvec(action)?;
    Ok(hash_parts(
        b"r2-map-action-identity-v1",
        &[&position_id, &action_bytes],
    ))
}

/// Canonical identity used by the R2-MAP exhaustive draft request. It is
/// intentionally independent of D6 augmentation and the enclosing position;
/// the request decision identity supplies the latter binding.
pub fn r2_map_draft_action_id(action: &TurnAction) -> Result<IdentityHash, R2MapExperienceError> {
    let mut hasher = Hasher::new();
    hasher.update(b"cascadia-v2-full-legal-action-v1");
    hasher.update(&serde_json::to_vec(action)?);
    Ok(*hasher.finalize().as_bytes())
}

/// Hash an ordered complete legal screen exactly as the Rust/Python transport
/// does: canonical JSON of lowercase hexadecimal action identities.
pub fn r2_map_ordered_action_ids_blake3(
    action_ids: &[IdentityHash],
) -> Result<IdentityHash, R2MapExperienceError> {
    let values = action_ids
        .iter()
        .map(|id| {
            id.iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        })
        .collect::<Vec<_>>();
    Ok(*blake3::hash(&serde_json::to_vec(&values)?).as_bytes())
}

fn score_components(score: ScoreBreakdown) -> [u16; R2_MAP_COMPONENT_COUNT] {
    let mut components = [0; R2_MAP_COMPONENT_COUNT];
    components[..5].copy_from_slice(&score.habitat);
    components[5..10].copy_from_slice(&score.wildlife);
    components[10] = score.nature_tokens;
    components
}

fn validate_score_identity(score: ScoreBreakdown) -> Result<(), R2MapExperienceError> {
    let base = score.habitat.iter().sum::<u16>()
        + score.wildlife.iter().sum::<u16>()
        + score.nature_tokens;
    let bonuses = score
        .habitat_bonus
        .iter()
        .map(|value| u16::from(*value))
        .sum::<u16>();
    if score.base_total != base || score.total != base + bonuses || bonuses != 0 {
        return Err(R2MapExperienceError::ScoreMismatch);
    }
    Ok(())
}

fn hash_parts(domain: &[u8], parts: &[&[u8]]) -> IdentityHash {
    let mut hasher = Hasher::new();
    hasher.update(domain);
    for part in parts {
        hasher.update(&(part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    *hasher.finalize().as_bytes()
}

#[derive(Debug, Error)]
pub enum R2MapExperienceError {
    #[error("R2-MAP experience schema version {0} is unsupported")]
    UnsupportedSchema(u16),
    #[error("invalid R2-MAP identity: {0}")]
    InvalidIdentity(&'static str),
    #[error("R2-MAP {0} identity does not match")]
    IdentityMismatch(&'static str),
    #[error("invalid seat contract: {0}")]
    SeatContract(&'static str),
    #[error("invalid exploration contract: {0}")]
    Exploration(&'static str),
    #[error("Pinecone earned/spent/remaining conservation failed")]
    PineconeConservation,
    #[error("trajectory replay diverged at turn {turn}")]
    ReplayMismatch { turn: usize },
    #[error("terminal {0} hash does not match replay")]
    HashMismatch(&'static str),
    #[error("score components do not reconcile")]
    ScoreMismatch,
    #[error("duplicate {0} identity")]
    DuplicateIdentity(&'static str),
    #[error("invalid R2-MAP record: {0}")]
    InvalidRecord(&'static str),
    #[error("invalid R2-MAP file {0}")]
    InvalidFile(&'static str),
    #[error(transparent)]
    Rules(#[from] cascadia_game::RuleError),
    #[error(transparent)]
    Replay(#[from] cascadia_game::ReplayError),
    #[error(transparent)]
    Encode(#[from] postcard::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use cascadia_game::{ScoringCards, score_game};
    use cascadia_sim::{
        MatchConfig, StrategyKind, play_match, play_match_with_seat_selector, select_greedy_action,
        strategy_rng,
    };
    use proptest::prelude::*;

    use super::*;

    fn nonzero_hash(value: u8) -> IdentityHash {
        [value.max(1); 32]
    }

    fn protocols() -> R2MapProtocolIdentity {
        R2MapProtocolIdentity {
            collector_hash: nonzero_hash(1),
            source_hash: nonzero_hash(2),
            serving_protocol_hash: nonzero_hash(3),
        }
    }

    #[test]
    fn exploration_counter_vector_preserves_v42_context_and_trace_identity() {
        let seed = GameSeed::from_u64(42);
        let identity = R2MapGameIdentity::new("test-campaign", 7, "john2", 0, seed);
        let rng_context = R2MapExplorationRngContext::new(&identity, seed, 3);
        let decision = R2MapExplorationDecisionIdentity::new(
            19,
            2,
            MarketDecisionStage::PaidWipes,
            [0x11; 32],
            [0x22; 32],
        );
        let action_id = [0x33; 32];
        let expected_decision_key = [
            0x33, 0xd3, 0x42, 0xeb, 0x96, 0xad, 0x1b, 0x69, 0xca, 0x48, 0xc7, 0x42, 0x0b, 0x23,
            0xf2, 0xf4, 0xbb, 0x73, 0xfb, 0x71, 0xe2, 0xab, 0x65, 0xef, 0x13, 0xbf, 0x8b, 0x93,
            0x43, 0x7a, 0xf3, 0x77,
        ];
        let expected_action_key = [
            0x3d, 0x3b, 0xaf, 0x8e, 0x95, 0x1a, 0xd3, 0x01, 0x5e, 0xad, 0x51, 0xfc, 0xaa, 0xf4,
            0xf3, 0xec, 0x11, 0x99, 0xef, 0x09, 0x96, 0x97, 0xec, 0x49, 0xe8, 0x9a, 0x01, 0x66,
            0x70, 0x8f, 0x69, 0x56,
        ];
        let expected_explore_draw = 1_030_276_283_339_359_281;
        let expected_action_draw = 6_223_913_895_286_580_137;

        assert_eq!(
            exploration_decision_key(rng_context, decision),
            expected_decision_key
        );
        assert_eq!(
            r2_map_explore_draw_u64(rng_context, decision),
            expected_explore_draw
        );
        assert_eq!(
            r2_map_action_draw_u64(rng_context, decision, action_id),
            expected_action_draw
        );

        let trace = exploration_trace(
            rng_context,
            100_000,
            R2MapExplorationDraw {
                turn_index: 19,
                ordinal: 2,
                stage: MarketDecisionStage::PaidWipes,
                decision_id: [0x11; 32],
                parent_public_hash: [0x22; 32],
                selected_action_id: action_id,
                explore_draw_u64: expected_explore_draw,
                action_draw_u64: Some(expected_action_draw),
                explored: true,
            },
            decision,
            action_id,
        )
        .unwrap();
        assert_eq!(trace.decision_rng_key, expected_decision_key);
        assert_eq!(trace.selected_action_rng_key, expected_action_key);
        validate_exploration_trace(rng_context, 100_000, decision, action_id, &trace).unwrap();
    }

    fn iterative_record(campaign_tag: u64, game_index: u64) -> R2MapGameRecord {
        let campaign_id = format!("test-campaign-{campaign_tag}");
        let seed = r2_map_game_seed(&campaign_id, R2MapSeedPurpose::Generation, 0, game_index);
        let identity = R2MapGameIdentity::new(campaign_id, 0, "john2", game_index, seed);
        let newest = R2MapPolicyIdentity::newest("greedy-v1", nonzero_hash(9));
        let pool = vec![
            R2MapPolicyIdentity::historical("greedy-v1", nonzero_hash(10)),
            R2MapPolicyIdentity::historical("greedy-v1", nonzero_hash(11)),
        ];
        let seats = assign_iterative_seats(&identity, seed, newest, &pool).unwrap();
        let ids: Vec<_> = seats
            .iter()
            .map(|policy| policy.policy_id.clone())
            .collect();
        let mut rngs: Vec<_> = ids
            .iter()
            .enumerate()
            .map(|(seat, id)| strategy_rng(seed, seat, id))
            .collect();
        let result = play_match_with_seat_selector(
            GameConfig::research_aaaaa(4).unwrap(),
            seed,
            &ids,
            |seat, game| {
                let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
                select_greedy_action(game, &prelude, &mut rngs[seat])
            },
        )
        .unwrap();
        let focal = focal_seat_for_game(game_index);
        let context = R2MapRecordContext {
            collection_kind: R2MapCollectionKind::IterativeTraining,
            identity,
            seed_purpose: R2MapSeedPurpose::Generation,
            focal_seat: focal,
            seats,
            rng: R2MapRngIdentity::default(),
            exploration: R2MapExplorationIdentity::training(0, 1_000_000),
            protocols: protocols(),
        };
        let draws = expected_r2_map_exploration_draws(&context, &result.replay).unwrap();
        R2MapGameRecord::from_match(context, &result, &draws).unwrap()
    }

    fn bootstrap_record(campaign_tag: u64, game_index: u64) -> R2MapGameRecord {
        let campaign_id = format!("test-bootstrap-{campaign_tag}");
        let seed = r2_map_game_seed(&campaign_id, R2MapSeedPurpose::Bootstrap, 0, game_index);
        let game = GameConfig::research_aaaaa(4).unwrap();
        let result = play_match(&MatchConfig::symmetric(game, seed, StrategyKind::Greedy)).unwrap();
        R2MapGameRecord::from_match(
            R2MapRecordContext {
                collection_kind: R2MapCollectionKind::Bootstrap,
                identity: R2MapGameIdentity::new(campaign_id, 0, "john2", game_index, seed),
                seed_purpose: R2MapSeedPurpose::Bootstrap,
                focal_seat: focal_seat_for_game(game_index),
                seats: vec![R2MapPolicyIdentity::greedy(); 4],
                rng: R2MapRngIdentity::default(),
                exploration: R2MapExplorationIdentity::disabled(),
                protocols: protocols(),
            },
            &result,
            &[],
        )
        .unwrap()
    }

    #[test]
    fn mixed_seat_record_replays_and_retains_exactly_twenty_focal_examples() {
        let record = iterative_record(91, 2);
        record.validate().unwrap();
        let examples = record.extract_primary_examples().unwrap();
        assert_eq!(examples.len(), 20);
        assert!(examples.iter().all(|example| example.seat == 2));
        for example in examples {
            example.validate().unwrap();
            assert_eq!(
                example.afterstate.completed_turns(),
                example.parent.completed_turns() + 1
            );
            assert!(
                example
                    .afterstate
                    .market()
                    .tiles
                    .iter()
                    .any(Option::is_none)
            );
            assert!(
                example
                    .afterstate
                    .market()
                    .wildlife
                    .iter()
                    .any(Option::is_none)
            );
        }
    }

    #[test]
    fn bootstrap_retains_all_eighty_decisions() {
        let record = bootstrap_record(92, 0);
        let examples = record.extract_primary_examples().unwrap();
        assert_eq!(examples.len(), 80);
        assert_eq!(
            examples.iter().filter(|example| example.seat == 0).count(),
            20
        );
        assert_eq!(
            examples.iter().filter(|example| example.seat == 3).count(),
            20
        );
        assert!(record.decisions.iter().all(|decision| {
            matches!(
                decision
                    .market_decisions
                    .last()
                    .map(|market| &market.selected),
                Some(MarketDecision::StopWiping)
            )
        }));
        assert!(record.decisions.iter().all(|decision| {
            decision.market_decisions.iter().all(|market| {
                market.legal_action_count > 0
                    && usize::from(market.selected_index) < usize::from(market.legal_action_count)
                    && market.exploration.is_none()
            })
        }));
    }

    #[test]
    fn bootstrap_focal_projection_retains_game_index_mod_four_seat() {
        let record = bootstrap_record(93, 7);
        let focal = focal_seat_for_game(record.identity.global_game_index);
        assert_eq!(focal, 3);
        let examples = record.extract_focal_seat_examples(focal).unwrap();
        assert_eq!(examples.len(), 20);
        assert!(examples.iter().all(|example| example.seat == focal));
        assert_eq!(
            examples
                .iter()
                .map(|example| example.turn_index)
                .collect::<Vec<_>>(),
            (3u16..80).step_by(4).collect::<Vec<_>>()
        );
    }

    #[test]
    fn replay_validation_rejects_intermediate_market_identity_and_hash_tampering() {
        let record = bootstrap_record(921, 0);
        let mut selected = record.clone();
        selected.decisions[0].market_decisions[0].selected_action_id[0] ^= 1;
        assert!(matches!(
            selected.validate(),
            Err(R2MapExperienceError::ReplayMismatch { turn: 0 })
        ));

        let mut resulting = record.clone();
        resulting.decisions[0].market_decisions[0].resulting_public_hash[0] ^= 1;
        assert!(matches!(
            resulting.validate(),
            Err(R2MapExperienceError::ReplayMismatch { turn: 0 })
        ));

        let mut omitted_stop = record;
        omitted_stop.decisions[0].market_decisions.pop();
        assert!(matches!(
            omitted_stop.validate(),
            Err(R2MapExperienceError::ReplayMismatch { turn: 0 })
        ));
    }

    #[test]
    fn seat_assignment_is_deterministic_unique_and_balanced() {
        let newest = R2MapPolicyIdentity::newest("r2-map-c0", nonzero_hash(20));
        let pool = vec![
            R2MapPolicyIdentity::greedy(),
            R2MapPolicyIdentity::historical("r2-map-c-old1", nonzero_hash(21)),
            R2MapPolicyIdentity::historical("r2-map-c-old2", nonzero_hash(22)),
            R2MapPolicyIdentity::historical("r2-map-c-old3", nonzero_hash(23)),
        ];
        let mut focal_counts = [0usize; 4];
        for index in 0..101 {
            let seed = r2_map_game_seed("campaign", R2MapSeedPurpose::Generation, 7, index);
            let identity = R2MapGameIdentity::new("campaign", 7, "john2", index, seed);
            let left = assign_iterative_seats(&identity, seed, newest.clone(), &pool).unwrap();
            let right = assign_iterative_seats(&identity, seed, newest.clone(), &pool).unwrap();
            assert_eq!(left, right);
            let focal = usize::from(focal_seat_for_game(index));
            focal_counts[focal] += 1;
            assert_eq!(
                left.iter()
                    .filter(|seat| seat.role == R2MapPolicyRole::Newest)
                    .count(),
                1
            );
            assert_eq!(left[focal], newest);
            let historical: HashSet<_> = left
                .iter()
                .filter_map(|seat| {
                    (seat.role == R2MapPolicyRole::Historical).then_some(seat.checkpoint_hash)
                })
                .collect();
            assert_eq!(
                historical.len(),
                left.iter()
                    .filter(|seat| seat.role == R2MapPolicyRole::Historical)
                    .count()
            );
            assert!(focal_counts.iter().max().unwrap() - focal_counts.iter().min().unwrap() <= 1);
        }
    }

    #[test]
    fn benchmark_accepts_each_focal_role_but_forbids_newest_or_duplicate_opponents() {
        let focal_seat = focal_seat_for_game(0);
        let focal = usize::from(focal_seat);
        let opponents = [
            R2MapPolicyIdentity::historical("historical-a", nonzero_hash(31)),
            R2MapPolicyIdentity::historical("historical-b", nonzero_hash(32)),
            R2MapPolicyIdentity::greedy(),
        ];

        for focal_policy in [
            R2MapPolicyIdentity::newest("candidate", nonzero_hash(30)),
            R2MapPolicyIdentity::historical("incumbent", nonzero_hash(33)),
            R2MapPolicyIdentity::greedy(),
        ] {
            let mut seats = opponents.to_vec();
            seats.insert(focal, focal_policy);
            validate_seat_contract(R2MapCollectionKind::Benchmark, 0, focal_seat, &seats).unwrap();
        }

        let mut newest_opponent = vec![R2MapPolicyIdentity::greedy(); 4];
        newest_opponent[(focal + 1) % 4] =
            R2MapPolicyIdentity::newest("candidate", nonzero_hash(34));
        assert!(matches!(
            validate_seat_contract(
                R2MapCollectionKind::Benchmark,
                0,
                focal_seat,
                &newest_opponent
            ),
            Err(R2MapExperienceError::SeatContract(
                "benchmark opponents cannot use the newest checkpoint role"
            ))
        ));

        let duplicate_hash = nonzero_hash(35);
        let mut duplicate_focal = vec![R2MapPolicyIdentity::greedy(); 4];
        duplicate_focal[focal] = R2MapPolicyIdentity::historical("incumbent", duplicate_hash);
        duplicate_focal[(focal + 1) % 4] =
            R2MapPolicyIdentity::historical("duplicate", duplicate_hash);
        assert!(matches!(
            validate_seat_contract(
                R2MapCollectionKind::Benchmark,
                0,
                focal_seat,
                &duplicate_focal
            ),
            Err(R2MapExperienceError::SeatContract(
                "benchmark focal checkpoint is duplicated in an opponent seat"
            ))
        ));
    }

    #[test]
    fn batch_validator_rejects_duplicate_game_and_gap() {
        let first = iterative_record(100, 0);
        assert!(matches!(
            validate_r2_map_record_batch(&[first.clone(), first.clone()]),
            Err(R2MapExperienceError::DuplicateIdentity("game"))
        ));
        let gap = iterative_record(100, 2);
        assert!(matches!(
            validate_r2_map_record_batch(&[first, gap]),
            Err(R2MapExperienceError::InvalidRecord(_))
        ));
    }

    #[test]
    fn receipt_gated_identity_validator_rejects_range_and_identity_tamper() {
        let first = iterative_record(100, 0);
        let mut gap = iterative_record(100, 2);
        assert!(matches!(
            validate_r2_map_record_batch_identities(&[first.clone(), gap.clone()]),
            Err(R2MapExperienceError::InvalidRecord(
                "batch identity or contiguous range differs"
            ))
        ));
        gap.identity.global_game_index = 1;
        gap.identity.game_id = first.identity.game_id;
        assert!(matches!(
            validate_r2_map_record_batch_identities(&[first.clone(), gap.clone()]),
            Err(R2MapExperienceError::InvalidRecord(
                "batch identity or contiguous range differs"
            ))
        ));
        gap.identity.game_id = iterative_record(100, 1).identity.game_id;
        gap.decisions[0].position_id = first.decisions[0].position_id;
        assert!(matches!(
            validate_r2_map_record_batch_identities(&[first, gap]),
            Err(R2MapExperienceError::DuplicateIdentity(
                "position or action"
            ))
        ));
    }

    #[test]
    fn framed_record_round_trip_and_checksum_rejection() {
        let record = iterative_record(93, 3);
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cascadia-r2-map-experience-{}-{nonce}",
            std::process::id()
        ));
        let path = root.join("game-00003.r2xp");
        write_r2_map_record(&path, &record).unwrap();
        assert_eq!(read_r2_map_record(&path).unwrap(), record);
        let mut bytes = fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        fs::write(&path, bytes).unwrap();
        assert!(matches!(
            read_r2_map_record(&path),
            Err(R2MapExperienceError::InvalidFile("checksum"))
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn shard_round_trip_is_ordered_compact_and_checksum_protected() {
        let records = vec![iterative_record(97, 1), iterative_record(97, 0)];
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cascadia-r2-map-shard-{}-{nonce}",
            std::process::id()
        ));
        let path = root.join("shard-00000.r2sh");
        write_r2_map_shard(&path, &records).unwrap();
        let decoded = read_r2_map_shard(&path).unwrap();
        assert_eq!(decoded[0].identity.global_game_index, 0);
        assert_eq!(decoded[1].identity.global_game_index, 1);
        assert!(fs::metadata(&path).unwrap().len() < 256 * 1024);

        let mut bytes = fs::read(&path).unwrap();
        bytes[R2_MAP_SHARD_HEADER_SIZE + 8] ^= 1;
        fs::write(&path, bytes).unwrap();
        assert!(matches!(
            read_r2_map_shard(&path),
            Err(R2MapExperienceError::InvalidFile("shard checksum"))
        ));
        assert!(matches!(
            read_r2_map_shard_after_semantic_validation(&path),
            Err(R2MapExperienceError::InvalidFile("shard checksum"))
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn seed_leases_are_host_independent_disjoint_and_exclude_john4() {
        let john2 = R2MapSeedLease {
            campaign_id: "campaign".to_owned(),
            iteration: 2,
            purpose: R2MapSeedPurpose::Generation,
            host_id: "john2".to_owned(),
            first_game_index: 100,
            game_count: 50,
        };
        let john3 = R2MapSeedLease {
            host_id: "john3".to_owned(),
            first_game_index: 150,
            ..john2.clone()
        };
        validate_r2_map_seed_leases(&[john2.clone(), john3.clone()]).unwrap();
        assert_eq!(
            john2.seed(100).unwrap(),
            r2_map_game_seed("campaign", R2MapSeedPurpose::Generation, 2, 100)
        );
        let overlap = R2MapSeedLease {
            host_id: "john1".to_owned(),
            first_game_index: 149,
            game_count: 10,
            ..john2.clone()
        };
        assert!(matches!(
            validate_r2_map_seed_leases(&[john2.clone(), overlap]),
            Err(R2MapExperienceError::DuplicateIdentity("seed lease"))
        ));
        let john4 = R2MapSeedLease {
            host_id: "john4".to_owned(),
            ..john2
        };
        assert!(john4.validate().is_err());
    }

    #[test]
    fn tampered_pinecone_and_score_metadata_are_rejected() {
        let mut record = iterative_record(94, 0);
        record.decisions[0].pinecones.earned += 1;
        assert!(record.validate().is_err());

        let mut record = iterative_record(95, 0);
        record.scores[0].wildlife[0] += 1;
        assert!(matches!(
            record.validate(),
            Err(R2MapExperienceError::ScoreMismatch)
        ));

        let mut record = iterative_record(98, 0);
        record.decisions[0]
            .exploration
            .as_mut()
            .unwrap()
            .explore_draw_u64 ^= 1;
        assert!(matches!(
            record.validate(),
            Err(R2MapExperienceError::Exploration(_))
        ));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(6))]

        #[test]
        fn replay_hashes_and_score_targets_hold_across_seeds(seed in any::<u64>()) {
            let record = bootstrap_record(seed, 0);
            prop_assert_eq!(&record.scores, &score_game(&record.replay.play().unwrap()));
            for example in record.extract_primary_examples().unwrap() {
                example.validate().unwrap();
                for component in 0..R2_MAP_COMPONENT_COUNT {
                    prop_assert_eq!(
                        i32::from(example.current[component]) + i32::from(example.residual[component]),
                        i32::from(example.terminal[component])
                    );
                }
            }
        }
    }

    #[test]
    fn frozen_research_rules_are_card_a_without_habitat_bonus() {
        let record = iterative_record(96, 0);
        assert_eq!(record.config.scoring_cards, ScoringCards::AAAAA);
        assert!(!record.config.habitat_bonuses);
    }
}
