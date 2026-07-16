//! Forced-action terminal continuation protocol for the CPU reference harness.
//!
//! P1 runs visibly typed proxy continuations only.  The harness preserves
//! public prelude -> reveal -> draft chronology, isolates memory by seat,
//! derives a fresh policy/redetermination stream for every public root, and
//! checks that sequential staging is byte-identical to one canonical compound
//! `TurnAction` before advancing the physical world.

use std::path::Path;

use cascadia_game::{
    GameSeed, GameState, MarketPrelude, PublicSupply, RuleError, ScoreBreakdown, TurnAction,
    score_game,
};
use serde::de::{self, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use thiserror::Error;

use crate::scenario::ReferenceWorld;
use crate::{
    ActionContentId, BoxedPolicyError, CandidateActionOccurrenceId, EvaluationBranch, Fidelity,
    FrozenPolicy, FrozenPolicyIdentity, HonestWorldSampler, INDEPENDENT_SCENARIO_SAMPLER_ID,
    IncumbentCandidateMenu, IncumbentMenuHash, IndependentScenarioSampler, InnerRngCoordinate,
    MenuComposer, ObservationError, PolicyMemoryBank, PrivateSimState, PublicRootId,
    RNG_CONTRACT_ID, ResearchRulesetIdentity, RngFactory, RootActionOccurrenceId,
    RootDecisionRecord, RulesDecision, RulesMenuHash, ScenarioCoordinate, ScenarioMode, SeatIndex,
    SeatLocalMemory, SelectedDecisionKind, Sha256Digest, replay_policy_decision_trace,
};

pub const PROXY_TERMINAL_TRAJECTORY_SCHEMA_ID: &str =
    "cascadiav3.rival_proxy_terminal_trajectory.v1";
pub const PROXY_TERMINAL_PAIR_SCHEMA_ID: &str = "cascadiav3.rival_terminal_pair_ledger.v1";
pub const VERIFIED_TERMINAL_PAIR_RECEIPT_SCHEMA_ID: &str =
    "cascadiav3.rival_verified_terminal_pair_receipt.v1";
pub const TERMINAL_PAIR_VERIFIER_CONTRACT_ID: &str = "cascadia-rival.verify-terminal-pair.v1";
/// Inclusive byte ceiling for one serialized proxy terminal-pair ledger.
///
/// Both the Rust verifier CLI and the in-process parsing boundary enforce this
/// before JSON deserialization. The CLI additionally bounds its file read, so
/// a concurrently growing artifact cannot cause unbounded allocation.
pub const MAX_TERMINAL_PAIR_LEDGER_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_TERMINAL_TRAJECTORY_TURNS: usize = 80;
pub const MAX_POLICY_DECISIONS_PER_TURN: usize = 32;
const MAX_TERMINAL_CANDIDATE_MENU_ENTRIES: usize = u16::MAX as usize + 1;
const MAX_TERMINAL_STATE_HASHES: usize = MAX_TERMINAL_TRAJECTORY_TURNS + 1;
const RESEARCH_PLAYER_COUNT: usize = 4;

fn ensure_terminal_ledger_byte_limit(byte_len: usize) -> Result<(), TerminalError> {
    let actual = u64::try_from(byte_len).unwrap_or(u64::MAX);
    if actual > MAX_TERMINAL_PAIR_LEDGER_BYTES {
        return Err(TerminalError::LedgerByteLimitExceeded {
            actual,
            maximum: MAX_TERMINAL_PAIR_LEDGER_BYTES,
        });
    }
    Ok(())
}

fn challenger_branch_for_candidate_index(
    candidate_index: usize,
) -> Result<EvaluationBranch, TerminalError> {
    let ordinal = u16::try_from(candidate_index)
        .map_err(|_| TerminalError::ChallengerBranchOrdinalOutOfRange(candidate_index))?;
    Ok(EvaluationBranch::Challenger(ordinal))
}

fn deserialize_bounded_vec<'de, D, T, const MAX: usize>(
    deserializer: D,
    field: &'static str,
) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct BoundedVecVisitor<T, const MAX: usize> {
        field: &'static str,
        marker: std::marker::PhantomData<T>,
    }

    impl<'de, T, const MAX: usize> Visitor<'de> for BoundedVecVisitor<T, MAX>
    where
        T: Deserialize<'de>,
    {
        type Value = Vec<T>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "at most {MAX} entries in {}", self.field)
        }

        fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let capacity = sequence.size_hint().unwrap_or(0).min(MAX);
            let mut values = Vec::with_capacity(capacity);
            while values.len() < MAX {
                match sequence.next_element()? {
                    Some(value) => values.push(value),
                    None => return Ok(values),
                }
            }
            if sequence.next_element::<de::IgnoredAny>()?.is_some() {
                return Err(de::Error::custom(format_args!(
                    "{} exceeds the hard limit of {MAX} entries",
                    self.field
                )));
            }
            Ok(values)
        }
    }

    deserializer.deserialize_seq(BoundedVecVisitor::<T, MAX> {
        field,
        marker: std::marker::PhantomData,
    })
}

fn deserialize_candidate_rules_indices<'de, D>(deserializer: D) -> Result<Vec<u32>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, u32, MAX_TERMINAL_CANDIDATE_MENU_ENTRIES>(
        deserializer,
        "source_candidate_rules_indices",
    )
}

fn deserialize_actions<'de, D>(deserializer: D) -> Result<Vec<TurnAction>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, TurnAction, MAX_TERMINAL_TRAJECTORY_TURNS>(deserializer, "actions")
}

#[derive(Deserialize)]
#[serde(transparent)]
struct BoundedDecisionRecords(
    #[serde(deserialize_with = "deserialize_decision_records")] Vec<RootDecisionRecord>,
);

fn deserialize_decision_records<'de, D>(
    deserializer: D,
) -> Result<Vec<RootDecisionRecord>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, RootDecisionRecord, MAX_POLICY_DECISIONS_PER_TURN>(
        deserializer,
        "turn_root_decisions[]",
    )
}

fn deserialize_turn_root_decisions<'de, D>(
    deserializer: D,
) -> Result<Vec<Vec<RootDecisionRecord>>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, BoundedDecisionRecords, MAX_TERMINAL_TRAJECTORY_TURNS>(
        deserializer,
        "turn_root_decisions",
    )
    .map(|turns| turns.into_iter().map(|turn| turn.0).collect())
}

fn deserialize_state_hashes<'de, D>(deserializer: D) -> Result<Vec<[u8; 32]>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, [u8; 32], MAX_TERMINAL_STATE_HASHES>(deserializer, "state_hashes")
}

fn deserialize_final_scores<'de, D>(deserializer: D) -> Result<Vec<ScoreBreakdown>, D::Error>
where
    D: Deserializer<'de>,
{
    deserialize_bounded_vec::<D, ScoreBreakdown, RESEARCH_PLAYER_COUNT>(
        deserializer,
        "final_scores",
    )
}

/// Replay-complete evidence for one forced first action and its continuation.
/// The private starting state is evidence data, never policy input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "ProxyTerminalTrajectoryWire")]
pub struct ProxyTerminalTrajectory {
    schema_id: String,
    ruleset: ResearchRulesetIdentity,
    source_game_identity_sha256: Sha256Digest,
    scenario_sampler_identity_sha256: Sha256Digest,
    continuation_policy_identity_sha256: Sha256Digest,
    policy_rng_factory_identity_sha256: Sha256Digest,
    sampler_contract_id: String,
    rng_contract_id: String,
    scenario_mode: ScenarioMode,
    scenario_coordinate: ScenarioCoordinate,
    world_redetermination_seed: GameSeed,
    target_seat: SeatIndex,
    source_world: GameState,
    source_public_root_id: PublicRootId,
    source_rules_menu_hash: RulesMenuHash,
    source_candidate_menu_hash: IncumbentMenuHash,
    source_candidate_rules_indices: Vec<u32>,
    forced_candidate_index: u32,
    forced_rules_index: u32,
    forced_root_action_occurrence_id: RootActionOccurrenceId,
    forced_candidate_occurrence_id: CandidateActionOccurrenceId,
    source_memories: [SeatLocalMemory; 4],
    continuation_memories: [SeatLocalMemory; 4],
    initial_world: GameState,
    initial_public_state_hash: [u8; 32],
    initial_public_supply: PublicSupply,
    forced_action_content_id: ActionContentId,
    actions: Vec<TurnAction>,
    turn_root_decisions: Vec<Vec<RootDecisionRecord>>,
    state_hashes: Vec<[u8; 32]>,
    final_scores: Vec<ScoreBreakdown>,
    proxy_policy: bool,
    trajectory_sha256: Sha256Digest,
}

#[derive(Serialize)]
struct TrajectoryContent<'a> {
    schema_id: &'a str,
    ruleset: &'a ResearchRulesetIdentity,
    source_game_identity_sha256: &'a Sha256Digest,
    scenario_sampler_identity_sha256: &'a Sha256Digest,
    continuation_policy_identity_sha256: &'a Sha256Digest,
    policy_rng_factory_identity_sha256: &'a Sha256Digest,
    sampler_contract_id: &'a str,
    rng_contract_id: &'a str,
    scenario_mode: ScenarioMode,
    scenario_coordinate: &'a ScenarioCoordinate,
    world_redetermination_seed: GameSeed,
    target_seat: SeatIndex,
    source_world: &'a GameState,
    source_public_root_id: &'a PublicRootId,
    source_rules_menu_hash: &'a RulesMenuHash,
    source_candidate_menu_hash: &'a IncumbentMenuHash,
    source_candidate_rules_indices: &'a [u32],
    forced_candidate_index: u32,
    forced_rules_index: u32,
    forced_root_action_occurrence_id: &'a RootActionOccurrenceId,
    forced_candidate_occurrence_id: &'a CandidateActionOccurrenceId,
    source_memories: &'a [SeatLocalMemory; 4],
    continuation_memories: &'a [SeatLocalMemory; 4],
    initial_world: &'a GameState,
    initial_public_state_hash: [u8; 32],
    initial_public_supply: PublicSupply,
    forced_action_content_id: &'a ActionContentId,
    actions: &'a [TurnAction],
    turn_root_decisions: &'a [Vec<RootDecisionRecord>],
    state_hashes: &'a [[u8; 32]],
    final_scores: &'a [ScoreBreakdown],
    proxy_policy: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProxyTerminalTrajectoryWire {
    schema_id: String,
    ruleset: ResearchRulesetIdentity,
    source_game_identity_sha256: Sha256Digest,
    scenario_sampler_identity_sha256: Sha256Digest,
    continuation_policy_identity_sha256: Sha256Digest,
    policy_rng_factory_identity_sha256: Sha256Digest,
    sampler_contract_id: String,
    rng_contract_id: String,
    scenario_mode: ScenarioMode,
    scenario_coordinate: ScenarioCoordinate,
    world_redetermination_seed: GameSeed,
    target_seat: SeatIndex,
    source_world: GameState,
    source_public_root_id: PublicRootId,
    source_rules_menu_hash: RulesMenuHash,
    source_candidate_menu_hash: IncumbentMenuHash,
    #[serde(deserialize_with = "deserialize_candidate_rules_indices")]
    source_candidate_rules_indices: Vec<u32>,
    forced_candidate_index: u32,
    forced_rules_index: u32,
    forced_root_action_occurrence_id: RootActionOccurrenceId,
    forced_candidate_occurrence_id: CandidateActionOccurrenceId,
    source_memories: [SeatLocalMemory; 4],
    continuation_memories: [SeatLocalMemory; 4],
    initial_world: GameState,
    initial_public_state_hash: [u8; 32],
    initial_public_supply: PublicSupply,
    forced_action_content_id: ActionContentId,
    #[serde(deserialize_with = "deserialize_actions")]
    actions: Vec<TurnAction>,
    #[serde(deserialize_with = "deserialize_turn_root_decisions")]
    turn_root_decisions: Vec<Vec<RootDecisionRecord>>,
    #[serde(deserialize_with = "deserialize_state_hashes")]
    state_hashes: Vec<[u8; 32]>,
    #[serde(deserialize_with = "deserialize_final_scores")]
    final_scores: Vec<ScoreBreakdown>,
    proxy_policy: bool,
    trajectory_sha256: Sha256Digest,
}

impl ProxyTerminalTrajectory {
    pub fn replay(&self) -> Result<GameState, TerminalError> {
        if self.schema_id != PROXY_TERMINAL_TRAJECTORY_SCHEMA_ID || !self.proxy_policy {
            return Err(TerminalError::WrongTrajectorySchema);
        }
        self.validate_collection_bounds()?;
        if let EvaluationBranch::Challenger(actual) = self.scenario_coordinate.branch {
            let expected = match challenger_branch_for_candidate_index(
                usize::try_from(self.forced_candidate_index)
                    .map_err(|_| TerminalError::MenuTooLarge)?,
            )? {
                EvaluationBranch::Challenger(ordinal) => ordinal,
                EvaluationBranch::Incumbent => unreachable!("helper always returns challenger"),
            };
            if actual != expected {
                return Err(TerminalError::ChallengerBranchOrdinalMismatch { expected, actual });
            }
        }
        self.ruleset
            .validate()
            .map_err(|_| TerminalError::WrongTrajectoryIdentity)?;
        if self.initial_world.config()
            != cascadia_game::GameConfig::research_aaaaa(4).expect("valid research config")
            || self.sampler_contract_id != INDEPENDENT_SCENARIO_SAMPLER_ID
            || self.rng_contract_id != RNG_CONTRACT_ID
            || self.scenario_mode != ScenarioMode::IndependentHiddenOrderReference
        {
            return Err(TerminalError::WrongTrajectoryIdentity);
        }
        self.initial_world
            .validate()
            .map_err(TerminalError::InvalidInitialWorld)?;
        if self.actions.len() != self.turn_root_decisions.len()
            || self.state_hashes.len() != self.actions.len() + 1
            || self.actions.is_empty()
            || self.actions.len()
                != usize::from(
                    self.initial_world
                        .total_turns()
                        .saturating_sub(self.initial_world.completed_turns()),
                )
        {
            return Err(TerminalError::IncompleteTrajectory);
        }
        self.source_world
            .validate()
            .map_err(TerminalError::InvalidInitialWorld)?;
        let source_identity = Sha256Digest::of_bytes(&serde_json::to_vec(&self.source_world)?);
        if source_identity != self.source_game_identity_sha256
            || self.source_world.public_state() != self.initial_world.public_state()
            || self.source_world.public_supply() != self.initial_world.public_supply()
        {
            return Err(TerminalError::SourceWorldIdentityMismatch);
        }
        let mut game = self.initial_world.clone();
        let actor = SeatIndex::new(self.source_world.current_player() as u8)?;
        if actor != self.target_seat {
            return Err(TerminalError::InvalidForcedRoot);
        }
        for seat in 0..4usize {
            if seat != usize::from(self.target_seat.get())
                && self.source_memories[seat] != self.continuation_memories[seat]
            {
                return Err(TerminalError::CrossSeatForcedMemoryMutation(seat as u8));
            }
        }
        let observation = PrivateSimState::new(self.source_world.clone())?.public_observation(
            actor,
            self.source_memories[usize::from(actor.get())].clone(),
        )?;
        let source_root = PublicRootId::new(&observation, crate::RootKind::DraftPolicyRoot);
        let rules_menu = MenuComposer::draft_root(&self.source_world, &MarketPrelude::default())?;
        let candidate_indices = self
            .source_candidate_rules_indices
            .iter()
            .map(|index| usize::try_from(*index).map_err(|_| TerminalError::MenuTooLarge))
            .collect::<Result<Vec<_>, _>>()?;
        let candidate_menu =
            IncumbentCandidateMenu::from_rules_indices(&rules_menu, candidate_indices)?;
        let candidate_index = usize::try_from(self.forced_candidate_index)
            .map_err(|_| TerminalError::MenuTooLarge)?;
        let rules_index = *candidate_menu
            .rules_indices()
            .get(candidate_index)
            .ok_or(TerminalError::ForcedCandidateOutsideMenu)?;
        let forced_action = candidate_menu
            .draft_action(candidate_index)
            .ok_or(TerminalError::ForcedCandidateOutsideMenu)?;
        if source_root != self.source_public_root_id
            || rules_menu.hash() != self.source_rules_menu_hash
            || candidate_menu.hash() != self.source_candidate_menu_hash
            || u32::try_from(rules_index).map_err(|_| TerminalError::MenuTooLarge)?
                != self.forced_rules_index
            || RootActionOccurrenceId::new(&source_root, &rules_menu, rules_index)?
                != self.forced_root_action_occurrence_id
            || CandidateActionOccurrenceId::new(&source_root, &candidate_menu, candidate_index)?
                != self.forced_candidate_occurrence_id
            || forced_action != &self.actions[0]
        {
            return Err(TerminalError::ForcedCandidateProvenanceMismatch);
        }
        if game.is_game_over()
            || *game.public_state().canonical_hash().as_bytes() != self.initial_public_state_hash
            || game.public_supply() != self.initial_public_supply
            || ActionContentId::canonical(&self.actions[0]) != self.forced_action_content_id
            || !self.turn_root_decisions[0].is_empty()
        {
            return Err(TerminalError::InvalidForcedRoot);
        }
        let mut redetermined = self.source_world.clone();
        redetermined.redeterminize_hidden(self.world_redetermination_seed);
        if redetermined.canonical_hash() != self.initial_world.canonical_hash() {
            return Err(TerminalError::WorldSeedMismatch);
        }
        if *game.canonical_hash().as_bytes() != self.state_hashes[0] {
            return Err(TerminalError::TrajectoryStateHashMismatch(0));
        }
        let mut memories = PolicyMemoryBank::from_four(self.continuation_memories.clone());
        let mut expected_decision_ordinal = 0u32;
        for (index, ((action, decisions), expected_hash)) in self
            .actions
            .iter()
            .zip(&self.turn_root_decisions)
            .zip(self.state_hashes.iter().skip(1))
            .enumerate()
        {
            if index == 0 {
                if !decisions.is_empty() {
                    return Err(TerminalError::InvalidForcedRoot);
                }
            } else {
                replay_policy_decision_trace(
                    &game,
                    action,
                    decisions,
                    &mut memories,
                    &mut expected_decision_ordinal,
                )?;
            }
            game.apply(action)?;
            if game.canonical_hash().as_bytes() != expected_hash {
                return Err(TerminalError::TrajectoryStateHashMismatch(index + 1));
            }
        }
        if !game.is_game_over() {
            return Err(TerminalError::NonterminalTrajectory);
        }
        if score_game(&game) != self.final_scores {
            return Err(TerminalError::FinalScoreMismatch);
        }
        if self.recompute_hash()? != self.trajectory_sha256 {
            return Err(TerminalError::TrajectoryContentHashMismatch);
        }
        Ok(game)
    }

    pub fn target_score(&self) -> Result<u16, TerminalError> {
        self.replay()?;
        self.target_score_unchecked()
    }

    fn target_score_unchecked(&self) -> Result<u16, TerminalError> {
        self.final_scores
            .get(usize::from(self.target_seat.get()))
            .map(|score| score.total)
            .ok_or(TerminalError::MissingTargetScore)
    }

    pub fn actions(&self) -> &[TurnAction] {
        &self.actions
    }

    pub fn source_memories(&self) -> &[SeatLocalMemory; 4] {
        &self.source_memories
    }

    pub fn continuation_memories(&self) -> &[SeatLocalMemory; 4] {
        &self.continuation_memories
    }

    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, TerminalError> {
        ensure_terminal_ledger_byte_limit(bytes.len())?;
        Ok(serde_json::from_slice(bytes)?)
    }

    pub fn write_json_immutable(&self, destination: &Path) -> Result<(), TerminalError> {
        self.replay()?;
        let bytes = serde_json::to_vec_pretty(self)?;
        ensure_terminal_ledger_byte_limit(bytes.len())?;
        crate::ledger::write_immutable_bytes(destination, &bytes)?;
        Ok(())
    }

    fn validate_collection_bounds(&self) -> Result<(), TerminalError> {
        let bounded = [
            (
                "source_candidate_rules_indices",
                self.source_candidate_rules_indices.len(),
                MAX_TERMINAL_CANDIDATE_MENU_ENTRIES,
            ),
            ("actions", self.actions.len(), MAX_TERMINAL_TRAJECTORY_TURNS),
            (
                "turn_root_decisions",
                self.turn_root_decisions.len(),
                MAX_TERMINAL_TRAJECTORY_TURNS,
            ),
            (
                "state_hashes",
                self.state_hashes.len(),
                MAX_TERMINAL_STATE_HASHES,
            ),
            (
                "final_scores",
                self.final_scores.len(),
                RESEARCH_PLAYER_COUNT,
            ),
        ];
        for (field, actual, maximum) in bounded {
            if actual > maximum {
                return Err(TerminalError::TrajectoryCollectionLimitExceeded {
                    field,
                    actual,
                    maximum,
                });
            }
        }
        if self.final_scores.len() != RESEARCH_PLAYER_COUNT {
            return Err(TerminalError::IncompleteTrajectory);
        }
        for decisions in &self.turn_root_decisions {
            if decisions.len() > MAX_POLICY_DECISIONS_PER_TURN {
                return Err(TerminalError::TrajectoryCollectionLimitExceeded {
                    field: "turn_root_decisions[]",
                    actual: decisions.len(),
                    maximum: MAX_POLICY_DECISIONS_PER_TURN,
                });
            }
        }
        Ok(())
    }

    fn recompute_hash(&self) -> Result<Sha256Digest, TerminalError> {
        let content = TrajectoryContent {
            schema_id: &self.schema_id,
            ruleset: &self.ruleset,
            source_game_identity_sha256: &self.source_game_identity_sha256,
            scenario_sampler_identity_sha256: &self.scenario_sampler_identity_sha256,
            continuation_policy_identity_sha256: &self.continuation_policy_identity_sha256,
            policy_rng_factory_identity_sha256: &self.policy_rng_factory_identity_sha256,
            sampler_contract_id: &self.sampler_contract_id,
            rng_contract_id: &self.rng_contract_id,
            scenario_mode: self.scenario_mode,
            scenario_coordinate: &self.scenario_coordinate,
            world_redetermination_seed: self.world_redetermination_seed,
            target_seat: self.target_seat,
            source_world: &self.source_world,
            source_public_root_id: &self.source_public_root_id,
            source_rules_menu_hash: &self.source_rules_menu_hash,
            source_candidate_menu_hash: &self.source_candidate_menu_hash,
            source_candidate_rules_indices: &self.source_candidate_rules_indices,
            forced_candidate_index: self.forced_candidate_index,
            forced_rules_index: self.forced_rules_index,
            forced_root_action_occurrence_id: &self.forced_root_action_occurrence_id,
            forced_candidate_occurrence_id: &self.forced_candidate_occurrence_id,
            source_memories: &self.source_memories,
            continuation_memories: &self.continuation_memories,
            initial_world: &self.initial_world,
            initial_public_state_hash: self.initial_public_state_hash,
            initial_public_supply: self.initial_public_supply,
            forced_action_content_id: &self.forced_action_content_id,
            actions: &self.actions,
            turn_root_decisions: &self.turn_root_decisions,
            state_hashes: &self.state_hashes,
            final_scores: &self.final_scores,
            proxy_policy: self.proxy_policy,
        };
        let value = serde_json::to_value(content)?;
        Ok(Sha256Digest::of_bytes(&serde_json::to_vec(&value)?))
    }
}

impl TryFrom<ProxyTerminalTrajectoryWire> for ProxyTerminalTrajectory {
    type Error = TerminalError;

    fn try_from(value: ProxyTerminalTrajectoryWire) -> Result<Self, Self::Error> {
        let trajectory = Self {
            schema_id: value.schema_id,
            ruleset: value.ruleset,
            source_game_identity_sha256: value.source_game_identity_sha256,
            scenario_sampler_identity_sha256: value.scenario_sampler_identity_sha256,
            continuation_policy_identity_sha256: value.continuation_policy_identity_sha256,
            policy_rng_factory_identity_sha256: value.policy_rng_factory_identity_sha256,
            sampler_contract_id: value.sampler_contract_id,
            rng_contract_id: value.rng_contract_id,
            scenario_mode: value.scenario_mode,
            scenario_coordinate: value.scenario_coordinate,
            world_redetermination_seed: value.world_redetermination_seed,
            target_seat: value.target_seat,
            source_world: value.source_world,
            source_public_root_id: value.source_public_root_id,
            source_rules_menu_hash: value.source_rules_menu_hash,
            source_candidate_menu_hash: value.source_candidate_menu_hash,
            source_candidate_rules_indices: value.source_candidate_rules_indices,
            forced_candidate_index: value.forced_candidate_index,
            forced_rules_index: value.forced_rules_index,
            forced_root_action_occurrence_id: value.forced_root_action_occurrence_id,
            forced_candidate_occurrence_id: value.forced_candidate_occurrence_id,
            source_memories: value.source_memories,
            continuation_memories: value.continuation_memories,
            initial_world: value.initial_world,
            initial_public_state_hash: value.initial_public_state_hash,
            initial_public_supply: value.initial_public_supply,
            forced_action_content_id: value.forced_action_content_id,
            actions: value.actions,
            turn_root_decisions: value.turn_root_decisions,
            state_hashes: value.state_hashes,
            final_scores: value.final_scores,
            proxy_policy: value.proxy_policy,
            trajectory_sha256: value.trajectory_sha256,
        };
        trajectory.replay()?;
        Ok(trajectory)
    }
}

/// One incumbent/challenger reference difference.  It can validate plumbing
/// but is structurally barred from high-fidelity or Rival-MF claims.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "ProxyTerminalPairWire")]
pub struct ProxyTerminalPair {
    schema_id: String,
    parent_manifest_sha256: Sha256Digest,
    source_game_identity_sha256: Sha256Digest,
    scenario_sampler_identity_sha256: Sha256Digest,
    continuation_policy_identity_sha256: Sha256Digest,
    policy_rng_factory_identity_sha256: Sha256Digest,
    incumbent: ProxyTerminalTrajectory,
    challenger: ProxyTerminalTrajectory,
    target_score_difference: i32,
    fidelity: Fidelity,
    beta_cv_required: i32,
    proxy_policy: bool,
    pair_sha256: Sha256Digest,
}

#[derive(Serialize)]
struct PairContent<'a> {
    schema_id: &'a str,
    parent_manifest_sha256: &'a Sha256Digest,
    source_game_identity_sha256: &'a Sha256Digest,
    scenario_sampler_identity_sha256: &'a Sha256Digest,
    continuation_policy_identity_sha256: &'a Sha256Digest,
    policy_rng_factory_identity_sha256: &'a Sha256Digest,
    incumbent: &'a ProxyTerminalTrajectory,
    challenger: &'a ProxyTerminalTrajectory,
    target_score_difference: i32,
    fidelity: Fidelity,
    beta_cv_required: i32,
    proxy_policy: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProxyTerminalPairWire {
    schema_id: String,
    parent_manifest_sha256: Sha256Digest,
    source_game_identity_sha256: Sha256Digest,
    scenario_sampler_identity_sha256: Sha256Digest,
    continuation_policy_identity_sha256: Sha256Digest,
    policy_rng_factory_identity_sha256: Sha256Digest,
    incumbent: ProxyTerminalTrajectory,
    challenger: ProxyTerminalTrajectory,
    target_score_difference: i32,
    fidelity: Fidelity,
    beta_cv_required: i32,
    proxy_policy: bool,
    pair_sha256: Sha256Digest,
}

impl ProxyTerminalPair {
    pub fn validate(&self) -> Result<(), TerminalError> {
        self.incumbent.replay()?;
        self.challenger.replay()?;
        self.validate_cross_branch()
    }

    /// Validate pair-level joins after both private trajectory values have
    /// already crossed their validating deserialization boundary.
    fn validate_cross_branch(&self) -> Result<(), TerminalError> {
        let expected_challenger_branch = challenger_branch_for_candidate_index(
            usize::try_from(self.challenger.forced_candidate_index)
                .map_err(|_| TerminalError::MenuTooLarge)?,
        )?;
        if self.schema_id != PROXY_TERMINAL_PAIR_SCHEMA_ID
            || !self.proxy_policy
            || self.beta_cv_required != 0
        {
            return Err(TerminalError::WrongPairSchema);
        }
        if self.incumbent.source_game_identity_sha256 != self.source_game_identity_sha256
            || self.challenger.source_game_identity_sha256 != self.source_game_identity_sha256
            || self.incumbent.scenario_sampler_identity_sha256
                != self.scenario_sampler_identity_sha256
            || self.challenger.scenario_sampler_identity_sha256
                != self.scenario_sampler_identity_sha256
            || self.incumbent.continuation_policy_identity_sha256
                != self.continuation_policy_identity_sha256
            || self.challenger.continuation_policy_identity_sha256
                != self.continuation_policy_identity_sha256
            || self.incumbent.policy_rng_factory_identity_sha256
                != self.policy_rng_factory_identity_sha256
            || self.challenger.policy_rng_factory_identity_sha256
                != self.policy_rng_factory_identity_sha256
            || self.incumbent.target_seat != self.challenger.target_seat
            || self.incumbent.ruleset != self.challenger.ruleset
            || self.incumbent.source_public_root_id != self.challenger.source_public_root_id
            || self.incumbent.source_rules_menu_hash != self.challenger.source_rules_menu_hash
            || self.incumbent.source_candidate_menu_hash
                != self.challenger.source_candidate_menu_hash
            || self.incumbent.source_candidate_rules_indices
                != self.challenger.source_candidate_rules_indices
            || self.incumbent.source_memories != self.challenger.source_memories
            || self.incumbent.source_world != self.challenger.source_world
            || self.incumbent.initial_world.public_state()
                != self.challenger.initial_world.public_state()
            || self.incumbent.initial_world.public_supply()
                != self.challenger.initial_world.public_supply()
            || self.incumbent.world_redetermination_seed
                == self.challenger.world_redetermination_seed
            || self.incumbent.forced_candidate_occurrence_id
                == self.challenger.forced_candidate_occurrence_id
            || self.incumbent.scenario_coordinate.panel_id
                != self.challenger.scenario_coordinate.panel_id
            || self.incumbent.scenario_coordinate.unit_index
                != self.challenger.scenario_coordinate.unit_index
            || self.incumbent.scenario_coordinate.fidelity
                != self.challenger.scenario_coordinate.fidelity
            || self.incumbent.scenario_coordinate.fidelity != self.fidelity
            || self.incumbent.scenario_coordinate.branch != EvaluationBranch::Incumbent
            || self.challenger.scenario_coordinate.branch != expected_challenger_branch
        {
            return Err(TerminalError::ScenarioCoordinateMismatch);
        }
        let recomputed = i32::from(self.challenger.target_score_unchecked()?)
            - i32::from(self.incumbent.target_score_unchecked()?);
        if recomputed != self.target_score_difference {
            return Err(TerminalError::PairDifferenceMismatch);
        }
        if self.recompute_hash()? != self.pair_sha256 {
            return Err(TerminalError::PairContentHashMismatch);
        }
        Ok(())
    }

    pub fn validate_pinned(
        &self,
        expected_pair: &Sha256Digest,
        expected_parent_manifest: &Sha256Digest,
    ) -> Result<(), TerminalError> {
        // Every public constructor and Deserialize implementation validates
        // both nested trajectories before a pair value can exist. Recheck the
        // pair-level join and external pins without replaying both branches a
        // second time.
        self.validate_cross_branch()?;
        if &self.pair_sha256 != expected_pair {
            return Err(TerminalError::PairPinMismatch);
        }
        if &self.parent_manifest_sha256 != expected_parent_manifest {
            return Err(TerminalError::ParentManifestPinMismatch);
        }
        Ok(())
    }

    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, TerminalError> {
        ensure_terminal_ledger_byte_limit(bytes.len())?;
        Ok(serde_json::from_slice(bytes)?)
    }

    pub fn from_json_slice_pinned(
        bytes: &[u8],
        expected_pair: &Sha256Digest,
        expected_parent_manifest: &Sha256Digest,
    ) -> Result<Self, TerminalError> {
        let pair = Self::from_json_slice(bytes)?;
        pair.validate_pinned(expected_pair, expected_parent_manifest)?;
        Ok(pair)
    }

    pub fn write_json_immutable(&self, destination: &Path) -> Result<(), TerminalError> {
        self.validate()?;
        let bytes = serde_json::to_vec_pretty(self)?;
        ensure_terminal_ledger_byte_limit(bytes.len())?;
        crate::ledger::write_immutable_bytes(destination, &bytes)?;
        Ok(())
    }

    pub fn incumbent(&self) -> &ProxyTerminalTrajectory {
        &self.incumbent
    }

    pub fn challenger(&self) -> &ProxyTerminalTrajectory {
        &self.challenger
    }

    pub fn beta_cv_required(&self) -> i32 {
        self.beta_cv_required
    }

    pub fn proxy_policy(&self) -> bool {
        self.proxy_policy
    }

    pub fn pair_sha256(&self) -> &Sha256Digest {
        &self.pair_sha256
    }

    fn recompute_hash(&self) -> Result<Sha256Digest, TerminalError> {
        let content = PairContent {
            schema_id: &self.schema_id,
            parent_manifest_sha256: &self.parent_manifest_sha256,
            source_game_identity_sha256: &self.source_game_identity_sha256,
            scenario_sampler_identity_sha256: &self.scenario_sampler_identity_sha256,
            continuation_policy_identity_sha256: &self.continuation_policy_identity_sha256,
            policy_rng_factory_identity_sha256: &self.policy_rng_factory_identity_sha256,
            incumbent: &self.incumbent,
            challenger: &self.challenger,
            target_score_difference: self.target_score_difference,
            fidelity: self.fidelity,
            beta_cv_required: self.beta_cv_required,
            proxy_policy: self.proxy_policy,
        };
        let value = serde_json::to_value(content)?;
        Ok(Sha256Digest::of_bytes(&serde_json::to_vec(&value)?))
    }
}

impl TryFrom<ProxyTerminalPairWire> for ProxyTerminalPair {
    type Error = TerminalError;

    fn try_from(value: ProxyTerminalPairWire) -> Result<Self, Self::Error> {
        let pair = Self {
            schema_id: value.schema_id,
            parent_manifest_sha256: value.parent_manifest_sha256,
            source_game_identity_sha256: value.source_game_identity_sha256,
            scenario_sampler_identity_sha256: value.scenario_sampler_identity_sha256,
            continuation_policy_identity_sha256: value.continuation_policy_identity_sha256,
            policy_rng_factory_identity_sha256: value.policy_rng_factory_identity_sha256,
            incumbent: value.incumbent,
            challenger: value.challenger,
            target_score_difference: value.target_score_difference,
            fidelity: value.fidelity,
            beta_cv_required: value.beta_cv_required,
            proxy_policy: value.proxy_policy,
            pair_sha256: value.pair_sha256,
        };
        pair.validate_cross_branch()?;
        Ok(pair)
    }
}

/// Rust-authenticated receipt emitted only after full pair deserialization,
/// canonical replay, content-hash verification, and cross-branch validation.
/// `ledger_file_sha256` binds the receipt to the exact input bytes as well as
/// to the pair's canonical scientific identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VerifiedTerminalPairReceipt {
    schema_id: String,
    verifier_contract_id: String,
    verifier_executable_sha256: Sha256Digest,
    ledger_file_sha256: Sha256Digest,
    pair_sha256: Sha256Digest,
    parent_manifest_sha256: Sha256Digest,
    ruleset_identity_sha256: Sha256Digest,
    source_game_identity_sha256: Sha256Digest,
    scenario_sampler_identity_sha256: Sha256Digest,
    continuation_policy_identity_sha256: Sha256Digest,
    policy_rng_factory_identity_sha256: Sha256Digest,
    source_public_root_id: PublicRootId,
    source_rules_menu_hash: RulesMenuHash,
    source_candidate_menu_hash: IncumbentMenuHash,
    panel_id: Sha256Digest,
    unit_index: u32,
    fidelity: Fidelity,
    target_seat: SeatIndex,
    challenger_branch_ordinal: u16,
    incumbent_candidate_occurrence_id: CandidateActionOccurrenceId,
    challenger_candidate_occurrence_id: CandidateActionOccurrenceId,
    incumbent_action_content_id: ActionContentId,
    challenger_action_content_id: ActionContentId,
    incumbent_post_action_memory_sha256: Sha256Digest,
    challenger_post_action_memory_sha256: Sha256Digest,
    incumbent_world_redetermination_seed_sha256: Sha256Digest,
    challenger_world_redetermination_seed_sha256: Sha256Digest,
    target_score_difference: i32,
    proxy_policy: bool,
    beta_cv_required: i32,
    receipt_sha256: Sha256Digest,
}

#[derive(Serialize)]
struct VerifiedReceiptContent<'a> {
    schema_id: &'a str,
    verifier_contract_id: &'a str,
    verifier_executable_sha256: &'a Sha256Digest,
    ledger_file_sha256: &'a Sha256Digest,
    pair_sha256: &'a Sha256Digest,
    parent_manifest_sha256: &'a Sha256Digest,
    ruleset_identity_sha256: &'a Sha256Digest,
    source_game_identity_sha256: &'a Sha256Digest,
    scenario_sampler_identity_sha256: &'a Sha256Digest,
    continuation_policy_identity_sha256: &'a Sha256Digest,
    policy_rng_factory_identity_sha256: &'a Sha256Digest,
    source_public_root_id: &'a PublicRootId,
    source_rules_menu_hash: &'a RulesMenuHash,
    source_candidate_menu_hash: &'a IncumbentMenuHash,
    panel_id: &'a Sha256Digest,
    unit_index: u32,
    fidelity: Fidelity,
    target_seat: SeatIndex,
    challenger_branch_ordinal: u16,
    incumbent_candidate_occurrence_id: &'a CandidateActionOccurrenceId,
    challenger_candidate_occurrence_id: &'a CandidateActionOccurrenceId,
    incumbent_action_content_id: &'a ActionContentId,
    challenger_action_content_id: &'a ActionContentId,
    incumbent_post_action_memory_sha256: &'a Sha256Digest,
    challenger_post_action_memory_sha256: &'a Sha256Digest,
    incumbent_world_redetermination_seed_sha256: &'a Sha256Digest,
    challenger_world_redetermination_seed_sha256: &'a Sha256Digest,
    target_score_difference: i32,
    proxy_policy: bool,
    beta_cv_required: i32,
}

impl ProxyTerminalPair {
    /// Cross the complete Rust verifier boundary from exact artifact bytes to
    /// one validated pair and its self-identifying receipt. Callers cannot
    /// inject precomputed file or executable digests.
    pub fn verify_bytes_and_create_receipt(
        ledger_bytes: &[u8],
        verifier_executable_bytes: &[u8],
        expected_pair_sha256: &Sha256Digest,
        expected_parent_manifest_sha256: &Sha256Digest,
    ) -> Result<(Self, VerifiedTerminalPairReceipt), TerminalError> {
        let pair = Self::from_json_slice(ledger_bytes)?;
        pair.validate_pinned(expected_pair_sha256, expected_parent_manifest_sha256)?;
        let receipt = pair.seal_verified_receipt(
            Sha256Digest::of_bytes(ledger_bytes),
            Sha256Digest::of_bytes(verifier_executable_bytes),
        )?;
        Ok((pair, receipt))
    }

    fn seal_verified_receipt(
        &self,
        ledger_file_sha256: Sha256Digest,
        verifier_executable_sha256: Sha256Digest,
    ) -> Result<VerifiedTerminalPairReceipt, TerminalError> {
        let mut receipt = VerifiedTerminalPairReceipt {
            schema_id: VERIFIED_TERMINAL_PAIR_RECEIPT_SCHEMA_ID.to_owned(),
            verifier_contract_id: TERMINAL_PAIR_VERIFIER_CONTRACT_ID.to_owned(),
            verifier_executable_sha256,
            ledger_file_sha256,
            pair_sha256: self.pair_sha256.clone(),
            parent_manifest_sha256: self.parent_manifest_sha256.clone(),
            ruleset_identity_sha256: canonical_ruleset_identity_sha256()?,
            source_game_identity_sha256: self.source_game_identity_sha256.clone(),
            scenario_sampler_identity_sha256: self.scenario_sampler_identity_sha256.clone(),
            continuation_policy_identity_sha256: self.continuation_policy_identity_sha256.clone(),
            policy_rng_factory_identity_sha256: self.policy_rng_factory_identity_sha256.clone(),
            source_public_root_id: self.incumbent.source_public_root_id.clone(),
            source_rules_menu_hash: self.incumbent.source_rules_menu_hash.clone(),
            source_candidate_menu_hash: self.incumbent.source_candidate_menu_hash.clone(),
            panel_id: self.incumbent.scenario_coordinate.panel_id.clone(),
            unit_index: self.incumbent.scenario_coordinate.unit_index,
            fidelity: self.fidelity,
            target_seat: self.incumbent.target_seat,
            challenger_branch_ordinal: match challenger_branch_for_candidate_index(
                usize::try_from(self.challenger.forced_candidate_index)
                    .map_err(|_| TerminalError::MenuTooLarge)?,
            )? {
                EvaluationBranch::Challenger(ordinal) => ordinal,
                EvaluationBranch::Incumbent => unreachable!("helper always returns challenger"),
            },
            incumbent_candidate_occurrence_id: self
                .incumbent
                .forced_candidate_occurrence_id
                .clone(),
            challenger_candidate_occurrence_id: self
                .challenger
                .forced_candidate_occurrence_id
                .clone(),
            incumbent_action_content_id: self.incumbent.forced_action_content_id.clone(),
            challenger_action_content_id: self.challenger.forced_action_content_id.clone(),
            incumbent_post_action_memory_sha256: seat_memory_sha256(
                &self.incumbent.continuation_memories
                    [usize::from(self.incumbent.target_seat.get())],
            )?,
            challenger_post_action_memory_sha256: seat_memory_sha256(
                &self.challenger.continuation_memories
                    [usize::from(self.challenger.target_seat.get())],
            )?,
            incumbent_world_redetermination_seed_sha256: Sha256Digest::of_bytes(
                &self.incumbent.world_redetermination_seed.0,
            ),
            challenger_world_redetermination_seed_sha256: Sha256Digest::of_bytes(
                &self.challenger.world_redetermination_seed.0,
            ),
            target_score_difference: self.target_score_difference,
            proxy_policy: self.proxy_policy,
            beta_cv_required: self.beta_cv_required,
            receipt_sha256: Sha256Digest::of_bytes(b"unsealed"),
        };
        receipt.receipt_sha256 = receipt.recompute_hash()?;
        receipt.validate()?;
        Ok(receipt)
    }
}

impl VerifiedTerminalPairReceipt {
    pub fn validate(&self) -> Result<(), TerminalError> {
        if self.schema_id != VERIFIED_TERMINAL_PAIR_RECEIPT_SCHEMA_ID
            || self.verifier_contract_id != TERMINAL_PAIR_VERIFIER_CONTRACT_ID
            || !self.proxy_policy
            || self.beta_cv_required != 0
            || self.ruleset_identity_sha256 != canonical_ruleset_identity_sha256()?
            || self.incumbent_candidate_occurrence_id == self.challenger_candidate_occurrence_id
        {
            return Err(TerminalError::WrongVerifiedReceiptSchema);
        }
        if self.recompute_hash()? != self.receipt_sha256 {
            return Err(TerminalError::VerifiedReceiptContentHashMismatch);
        }
        Ok(())
    }

    pub fn receipt_sha256(&self) -> &Sha256Digest {
        &self.receipt_sha256
    }

    pub fn ruleset_identity_sha256(&self) -> &Sha256Digest {
        &self.ruleset_identity_sha256
    }

    pub fn incumbent_world_redetermination_seed_sha256(&self) -> &Sha256Digest {
        &self.incumbent_world_redetermination_seed_sha256
    }

    pub fn challenger_world_redetermination_seed_sha256(&self) -> &Sha256Digest {
        &self.challenger_world_redetermination_seed_sha256
    }

    pub const fn challenger_branch_ordinal(&self) -> u16 {
        self.challenger_branch_ordinal
    }

    pub fn write_json_immutable(&self, destination: &Path) -> Result<(), TerminalError> {
        self.validate()?;
        let bytes = serde_json::to_vec_pretty(self)?;
        crate::ledger::write_immutable_bytes(destination, &bytes)?;
        Ok(())
    }

    fn recompute_hash(&self) -> Result<Sha256Digest, TerminalError> {
        let content = VerifiedReceiptContent {
            schema_id: &self.schema_id,
            verifier_contract_id: &self.verifier_contract_id,
            verifier_executable_sha256: &self.verifier_executable_sha256,
            ledger_file_sha256: &self.ledger_file_sha256,
            pair_sha256: &self.pair_sha256,
            parent_manifest_sha256: &self.parent_manifest_sha256,
            ruleset_identity_sha256: &self.ruleset_identity_sha256,
            source_game_identity_sha256: &self.source_game_identity_sha256,
            scenario_sampler_identity_sha256: &self.scenario_sampler_identity_sha256,
            continuation_policy_identity_sha256: &self.continuation_policy_identity_sha256,
            policy_rng_factory_identity_sha256: &self.policy_rng_factory_identity_sha256,
            source_public_root_id: &self.source_public_root_id,
            source_rules_menu_hash: &self.source_rules_menu_hash,
            source_candidate_menu_hash: &self.source_candidate_menu_hash,
            panel_id: &self.panel_id,
            unit_index: self.unit_index,
            fidelity: self.fidelity,
            target_seat: self.target_seat,
            challenger_branch_ordinal: self.challenger_branch_ordinal,
            incumbent_candidate_occurrence_id: &self.incumbent_candidate_occurrence_id,
            challenger_candidate_occurrence_id: &self.challenger_candidate_occurrence_id,
            incumbent_action_content_id: &self.incumbent_action_content_id,
            challenger_action_content_id: &self.challenger_action_content_id,
            incumbent_post_action_memory_sha256: &self.incumbent_post_action_memory_sha256,
            challenger_post_action_memory_sha256: &self.challenger_post_action_memory_sha256,
            incumbent_world_redetermination_seed_sha256: &self
                .incumbent_world_redetermination_seed_sha256,
            challenger_world_redetermination_seed_sha256: &self
                .challenger_world_redetermination_seed_sha256,
            target_score_difference: self.target_score_difference,
            proxy_policy: self.proxy_policy,
            beta_cv_required: self.beta_cv_required,
        };
        let value = serde_json::to_value(content)?;
        Ok(Sha256Digest::of_bytes(&serde_json::to_vec(&value)?))
    }
}

fn canonical_ruleset_identity_sha256() -> Result<Sha256Digest, TerminalError> {
    let canonical = serde_json::to_value(ResearchRulesetIdentity::canonical())?;
    Ok(Sha256Digest::of_bytes(&serde_json::to_vec(&canonical)?))
}

fn seat_memory_sha256(memory: &SeatLocalMemory) -> Result<Sha256Digest, TerminalError> {
    let canonical = serde_json::to_value(memory)?;
    Ok(Sha256Digest::of_bytes(&serde_json::to_vec(&canonical)?))
}

/// Complete, typed input for one P1 proxy pair. The harness asks one frozen
/// prototype for eight behaviorally clean instances: four seats in each
/// branch. The only branch-local recurrent policy state supplied by the appeal
/// root is the target seat's explicit post-forced-action memory.
pub struct ProxyTerminalPairRequest<'a, P> {
    pub sampler: &'a IndependentScenarioSampler,
    pub parent_manifest_sha256: Sha256Digest,
    pub incumbent_coordinate: &'a ScenarioCoordinate,
    pub challenger_coordinate: &'a ScenarioCoordinate,
    pub candidate_menu: &'a IncumbentCandidateMenu,
    pub incumbent_candidate_index: usize,
    pub challenger_candidate_index: usize,
    pub incumbent_post_action_memory: SeatLocalMemory,
    pub challenger_post_action_memory: SeatLocalMemory,
    pub policy_prototype: P,
    pub rng_factory: &'a RngFactory,
    pub target_seat: SeatIndex,
}

#[derive(Clone)]
struct FrozenCandidate {
    candidate_index: usize,
    rules_index: usize,
    action: TurnAction,
    root_occurrence_id: RootActionOccurrenceId,
    candidate_occurrence_id: CandidateActionOccurrenceId,
}

#[derive(Clone)]
struct TerminalRunIdentity {
    source_game: Sha256Digest,
    scenario_sampler: Sha256Digest,
    continuation_policy: Sha256Digest,
    policy_rng_factory: Sha256Digest,
}

struct BranchRunContext<'a> {
    forced: &'a FrozenCandidate,
    rules_menu_hash: RulesMenuHash,
    candidate_menu_hash: IncumbentMenuHash,
    candidate_rules_indices: Vec<u32>,
    source_root: PublicRootId,
    source_world: GameState,
    source_memories: [SeatLocalMemory; 4],
    post_action_memory: SeatLocalMemory,
    identity: &'a TerminalRunIdentity,
    rng_factory: &'a RngFactory,
    target_seat: SeatIndex,
}

fn validate_policy_instances<P: FrozenPolicy>(
    incumbent: &[P; 4],
    challenger: &[P; 4],
) -> Result<Sha256Digest, TerminalError> {
    let expected = incumbent[0].identity().identity_sha256();
    for policy in incumbent.iter().chain(challenger) {
        policy
            .identity()
            .fields()
            .validate_for_cpu_reference_harness()?;
        if policy.identity().identity_sha256() != expected {
            return Err(TerminalError::ContinuationIdentityMismatch);
        }
    }
    Ok(expected)
}

fn freeze_candidate(
    root: &PublicRootId,
    rules_menu: &crate::RulesLegalMenu,
    candidate_menu: &IncumbentCandidateMenu,
    candidate_index: usize,
) -> Result<FrozenCandidate, TerminalError> {
    let action = candidate_menu
        .draft_action(candidate_index)
        .ok_or(TerminalError::ForcedCandidateOutsideMenu)?
        .clone();
    if action.prelude() != MarketPrelude::default() {
        return Err(TerminalError::ForcedActionContainsPrelude);
    }
    let rules_index = *candidate_menu
        .rules_indices()
        .get(candidate_index)
        .ok_or(TerminalError::ForcedCandidateOutsideMenu)?;
    if rules_menu.draft_action(rules_index) != Some(&action) {
        return Err(TerminalError::CandidateMenuRootMismatch);
    }
    Ok(FrozenCandidate {
        candidate_index,
        rules_index,
        root_occurrence_id: RootActionOccurrenceId::new(root, rules_menu, rules_index)?,
        candidate_occurrence_id: CandidateActionOccurrenceId::new(
            root,
            candidate_menu,
            candidate_index,
        )?,
        action,
    })
}

/// Run one P1 proxy terminal pair on independently keyed hidden-order worlds.
pub fn run_proxy_terminal_pair<P: FrozenPolicy>(
    request: ProxyTerminalPairRequest<'_, P>,
) -> Result<ProxyTerminalPair, TerminalError> {
    let mut incumbent_policies = std::array::from_fn(|_| request.policy_prototype.fresh_instance());
    let mut challenger_policies =
        std::array::from_fn(|_| request.policy_prototype.fresh_instance());
    let continuation_policy_identity_sha256 =
        validate_policy_instances(&incumbent_policies, &challenger_policies)?;
    let source_rules_menu = request.sampler.rules_menu()?;
    let reconstructed_candidate_menu = IncumbentCandidateMenu::from_rules_indices(
        &source_rules_menu,
        request.candidate_menu.rules_indices().iter().copied(),
    )?;
    if &reconstructed_candidate_menu != request.candidate_menu {
        return Err(TerminalError::CandidateMenuRootMismatch);
    }
    if request.incumbent_candidate_index == request.challenger_candidate_index {
        return Err(TerminalError::DuplicateForcedCandidate);
    }
    let incumbent_candidate = freeze_candidate(
        request.sampler.public_root_id(),
        &source_rules_menu,
        request.candidate_menu,
        request.incumbent_candidate_index,
    )?;
    let challenger_candidate = freeze_candidate(
        request.sampler.public_root_id(),
        &source_rules_menu,
        request.candidate_menu,
        request.challenger_candidate_index,
    )?;
    let expected_challenger_branch =
        challenger_branch_for_candidate_index(challenger_candidate.candidate_index)?;
    if request.sampler.source().current_player() != usize::from(request.target_seat.get())
        || request.incumbent_coordinate.branch != EvaluationBranch::Incumbent
        || request.challenger_coordinate.branch != expected_challenger_branch
        || request.incumbent_coordinate.panel_id != request.challenger_coordinate.panel_id
        || request.incumbent_coordinate.unit_index != request.challenger_coordinate.unit_index
        || request.incumbent_coordinate.fidelity != request.challenger_coordinate.fidelity
    {
        return Err(TerminalError::ScenarioCoordinateMismatch);
    }

    let identity = TerminalRunIdentity {
        source_game: request.sampler.source_game_identity_sha256().clone(),
        scenario_sampler: request.sampler.sampler_identity_sha256().clone(),
        continuation_policy: continuation_policy_identity_sha256,
        policy_rng_factory: request.rng_factory.identity_sha256(),
    };
    let candidate_rules_indices = request
        .candidate_menu
        .rules_indices()
        .iter()
        .map(|index| u32::try_from(*index).map_err(|_| TerminalError::MenuTooLarge))
        .collect::<Result<Vec<_>, _>>()?;
    let common_context = |forced, post_action_memory| BranchRunContext {
        forced,
        rules_menu_hash: source_rules_menu.hash(),
        candidate_menu_hash: request.candidate_menu.hash(),
        candidate_rules_indices: candidate_rules_indices.clone(),
        source_root: request.sampler.public_root_id().clone(),
        source_world: request.sampler.source().clone(),
        source_memories: request.sampler.initial_memories().clone(),
        post_action_memory,
        identity: &identity,
        rng_factory: request.rng_factory,
        target_seat: request.target_seat,
    };
    let incumbent_world = request.sampler.sample(request.incumbent_coordinate);
    let challenger_world = request.sampler.sample(request.challenger_coordinate);
    if incumbent_world.public_hash() != challenger_world.public_hash() {
        return Err(TerminalError::ScenarioCoordinateMismatch);
    }
    let incumbent_context =
        common_context(&incumbent_candidate, request.incumbent_post_action_memory);
    let incumbent = run_proxy_branch(incumbent_world, &mut incumbent_policies, incumbent_context)?;
    let challenger_context =
        common_context(&challenger_candidate, request.challenger_post_action_memory);
    let challenger = run_proxy_branch(
        challenger_world,
        &mut challenger_policies,
        challenger_context,
    )?;
    let fidelity = request.incumbent_coordinate.fidelity;
    let mut pair = ProxyTerminalPair {
        schema_id: PROXY_TERMINAL_PAIR_SCHEMA_ID.to_owned(),
        parent_manifest_sha256: request.parent_manifest_sha256,
        source_game_identity_sha256: identity.source_game,
        scenario_sampler_identity_sha256: identity.scenario_sampler,
        continuation_policy_identity_sha256: identity.continuation_policy,
        policy_rng_factory_identity_sha256: identity.policy_rng_factory,
        target_score_difference: i32::from(challenger.target_score_unchecked()?)
            - i32::from(incumbent.target_score_unchecked()?),
        incumbent,
        challenger,
        fidelity,
        beta_cv_required: 0,
        proxy_policy: true,
        pair_sha256: Sha256Digest::of_bytes(b"unsealed"),
    };
    pair.pair_sha256 = pair.recompute_hash()?;
    pair.validate_cross_branch()?;
    Ok(pair)
}

fn run_proxy_branch<P: FrozenPolicy>(
    mut world: ReferenceWorld,
    policies: &mut [P; 4],
    context: BranchRunContext<'_>,
) -> Result<ProxyTerminalTrajectory, TerminalError> {
    let initial_world = world.game().clone();
    let mut actions = Vec::new();
    let mut turn_root_decisions = Vec::new();
    let mut state_hashes = vec![*world.canonical_hash().as_bytes()];
    world.game_mut().apply(&context.forced.action)?;
    actions.push(context.forced.action.clone());
    turn_root_decisions.push(Vec::new());
    state_hashes.push(*world.canonical_hash().as_bytes());

    let mut continuation_memories = context.source_memories.clone();
    continuation_memories[usize::from(context.target_seat.get())] = context.post_action_memory;
    let mut memories = PolicyMemoryBank::from_four(continuation_memories.clone());
    let mut decision_ordinal = 0u32;
    while !world.game().is_game_over() {
        let actor = world.game().current_player();
        let (action, decisions) = select_proxy_turn(
            world.game(),
            &mut policies[actor],
            context.rng_factory,
            world.coordinate(),
            &mut memories,
            &mut decision_ordinal,
        )?;
        world.game_mut().apply(&action)?;
        actions.push(action);
        turn_root_decisions.push(decisions);
        state_hashes.push(*world.canonical_hash().as_bytes());
    }
    let final_scores = score_game(world.game());
    let mut trajectory = ProxyTerminalTrajectory {
        schema_id: PROXY_TERMINAL_TRAJECTORY_SCHEMA_ID.to_owned(),
        ruleset: ResearchRulesetIdentity::canonical(),
        source_game_identity_sha256: context.identity.source_game.clone(),
        scenario_sampler_identity_sha256: context.identity.scenario_sampler.clone(),
        continuation_policy_identity_sha256: context.identity.continuation_policy.clone(),
        policy_rng_factory_identity_sha256: context.identity.policy_rng_factory.clone(),
        sampler_contract_id: INDEPENDENT_SCENARIO_SAMPLER_ID.to_owned(),
        rng_contract_id: RNG_CONTRACT_ID.to_owned(),
        scenario_mode: ScenarioMode::IndependentHiddenOrderReference,
        scenario_coordinate: world.coordinate().clone(),
        world_redetermination_seed: world.redetermination_seed(),
        target_seat: context.target_seat,
        source_world: context.source_world,
        source_public_root_id: context.source_root,
        source_rules_menu_hash: context.rules_menu_hash,
        source_candidate_menu_hash: context.candidate_menu_hash,
        source_candidate_rules_indices: context.candidate_rules_indices,
        forced_candidate_index: u32::try_from(context.forced.candidate_index)
            .map_err(|_| TerminalError::MenuTooLarge)?,
        forced_rules_index: u32::try_from(context.forced.rules_index)
            .map_err(|_| TerminalError::MenuTooLarge)?,
        forced_root_action_occurrence_id: context.forced.root_occurrence_id.clone(),
        forced_candidate_occurrence_id: context.forced.candidate_occurrence_id.clone(),
        source_memories: context.source_memories,
        continuation_memories,
        initial_public_state_hash: *initial_world.public_state().canonical_hash().as_bytes(),
        initial_public_supply: initial_world.public_supply(),
        forced_action_content_id: ActionContentId::canonical(&context.forced.action),
        initial_world,
        actions,
        turn_root_decisions,
        state_hashes,
        final_scores,
        proxy_policy: true,
        trajectory_sha256: Sha256Digest::of_bytes(b"unsealed"),
    };
    trajectory.trajectory_sha256 = trajectory.recompute_hash()?;
    trajectory.replay()?;
    Ok(trajectory)
}

fn select_proxy_turn<P: FrozenPolicy>(
    source: &GameState,
    policy: &mut P,
    rng_factory: &RngFactory,
    coordinate: &ScenarioCoordinate,
    memories: &mut PolicyMemoryBank,
    decision_ordinal: &mut u32,
) -> Result<(TurnAction, Vec<RootDecisionRecord>), TerminalError> {
    let actor = SeatIndex::new(source.current_player() as u8)?;
    let turn_source = source.clone();
    let mut staged = source.clone();
    let mut accumulated = MarketPrelude::default();
    let mut records = Vec::new();
    let mut invocation = PolicyInvocationContext {
        rng_factory,
        coordinate,
        actor,
        memories,
        decision_ordinal,
    };

    let prelude_menu = MenuComposer::prelude_root(&staged)?;
    let prelude = if prelude_menu.len() == 1 {
        match prelude_menu.decision(0) {
            Some(RulesDecision::Prelude(prelude)) if prelude == &MarketPrelude::default() => {
                prelude.clone()
            }
            _ => return Err(TerminalError::WrongDecisionKind),
        }
    } else {
        let (prelude_index, prelude_record) =
            invoke_policy(&staged, policy, &prelude_menu, &mut invocation)?;
        records.push(prelude_record);
        match prelude_menu.decision(prelude_index) {
            Some(RulesDecision::Prelude(prelude)) => prelude.clone(),
            _ => return Err(TerminalError::WrongDecisionKind),
        }
    };
    staged = staged.preview_market_prelude(&prelude)?;
    accumulated.replace_three_of_a_kind = prelude.replace_three_of_a_kind;

    loop {
        if records.len() >= MAX_POLICY_DECISIONS_PER_TURN {
            return Err(TerminalError::DecisionLimitExceeded);
        }
        let menu = MenuComposer::draft_root(&staged, &MarketPrelude::default())?;
        let (index, record) = invoke_policy(&staged, policy, &menu, &mut invocation)?;
        match menu.decision(index) {
            Some(RulesDecision::PaidWipe(wipe)) => {
                let one_step = MarketPrelude {
                    replace_three_of_a_kind: false,
                    wildlife_wipes: vec![wipe.clone()],
                };
                staged = staged.preview_market_prelude(&one_step)?;
                accumulated.wildlife_wipes.push(wipe.clone());
                records.push(record);
            }
            Some(RulesDecision::Draft(root_local_action)) => {
                let mut compound = root_local_action.clone();
                compound.replace_three_of_a_kind = accumulated.replace_three_of_a_kind;
                compound.wildlife_wipes = accumulated.wildlife_wipes;
                let staged_next = staged.transition(root_local_action)?;
                let compound_next = turn_source.transition(&compound)?;
                if staged_next.canonical_hash() != compound_next.canonical_hash() {
                    return Err(TerminalError::CompoundChronologyMismatch);
                }
                records.push(record);
                return Ok((compound, records));
            }
            Some(RulesDecision::Prelude(_)) | None => return Err(TerminalError::WrongDecisionKind),
        }
    }
}

struct PolicyInvocationContext<'a> {
    rng_factory: &'a RngFactory,
    coordinate: &'a ScenarioCoordinate,
    actor: SeatIndex,
    memories: &'a mut PolicyMemoryBank,
    decision_ordinal: &'a mut u32,
}

fn invoke_policy<P: FrozenPolicy>(
    state: &GameState,
    policy: &mut P,
    menu: &crate::RulesLegalMenu,
    context: &mut PolicyInvocationContext<'_>,
) -> Result<(usize, RootDecisionRecord), TerminalError> {
    let private = PrivateSimState::new(state.clone())?;
    let memory = context.memories.get(context.actor)?.clone();
    let observation = private.public_observation(context.actor, memory)?;
    let root = PublicRootId::new(&observation, menu.root_kind());
    let rng_coordinate = InnerRngCoordinate::new(
        &root,
        &context.coordinate.panel_id,
        context.coordinate.branch,
        context.coordinate.fidelity,
        context.actor,
        context.coordinate.unit_index,
        *context.decision_ordinal,
    );
    let redetermination = context.rng_factory.redetermination(rng_coordinate);
    let worlds = HonestWorldSampler::new(&private, redetermination);
    let mut policy_rng = context.rng_factory.policy(rng_coordinate);
    let record_ordinal = *context.decision_ordinal;
    let decision = policy
        .act(&observation, menu, &worlds, &mut policy_rng)
        .map_err(|error| TerminalError::Policy(BoxedPolicyError::new(error)))?;
    let next_memory = decision.next_memory().clone();
    let index = decision.choice().get();
    let selected = menu
        .decision(index)
        .ok_or(TerminalError::PolicySelectedOutsideMenu)?;
    let (selected_kind, draft_occurrence_id) = match selected {
        RulesDecision::Prelude(_) => (SelectedDecisionKind::Prelude, None),
        RulesDecision::PaidWipe(_) => (SelectedDecisionKind::PaidWipe, None),
        RulesDecision::Draft(_) => (
            SelectedDecisionKind::Draft,
            Some(RootActionOccurrenceId::new(&root, menu, index)?),
        ),
    };
    context
        .memories
        .replace(context.actor, decision.into_next_memory())?;
    *context.decision_ordinal = context
        .decision_ordinal
        .checked_add(1)
        .ok_or(TerminalError::DecisionOrdinalOverflow)?;
    let record = RootDecisionRecord {
        decision_ordinal: record_ordinal,
        root_kind: menu.root_kind(),
        public_observation: observation,
        public_root_id: root,
        ordered_menu_hash: menu.hash(),
        menu_len: menu
            .len()
            .try_into()
            .map_err(|_| TerminalError::MenuTooLarge)?,
        selected_index: index.try_into().map_err(|_| TerminalError::MenuTooLarge)?,
        selected_kind,
        draft_occurrence_id,
        next_memory,
    };
    record.validate()?;
    Ok((index, record))
}

#[derive(Debug, Error)]
pub enum TerminalError {
    #[error("unsupported or non-proxy terminal trajectory schema")]
    WrongTrajectorySchema,
    #[error("terminal trajectory ruleset, sampler, RNG, or mode identity is invalid")]
    WrongTrajectoryIdentity,
    #[error("terminal trajectory initial canonical state is invalid: {0}")]
    InvalidInitialWorld(&'static str),
    #[error("terminal trajectory forced root, seat, public state, or first action is invalid")]
    InvalidForcedRoot,
    #[error("forced action mutated nonacting seat {0} memory")]
    CrossSeatForcedMemoryMutation(u8),
    #[error("forced candidate is outside the frozen incumbent candidate menu")]
    ForcedCandidateOutsideMenu,
    #[error("forced candidate provenance does not reconstruct at the frozen source root")]
    ForcedCandidateProvenanceMismatch,
    #[error("candidate menu was not derived from the frozen source rules menu")]
    CandidateMenuRootMismatch,
    #[error("incumbent and challenger must be distinct frozen candidate occurrences")]
    DuplicateForcedCandidate,
    #[error("challenger candidate-menu index {0} exceeds the u16 branch-ordinal contract")]
    ChallengerBranchOrdinalOutOfRange(usize),
    #[error(
        "challenger branch ordinal {actual} does not equal frozen candidate-menu index {expected}"
    )]
    ChallengerBranchOrdinalMismatch { expected: u16, actual: u16 },
    #[error("post-prelude forced action unexpectedly contains a market prelude")]
    ForcedActionContainsPrelude,
    #[error("recorded redetermination seed does not reproduce the initial private world")]
    WorldSeedMismatch,
    #[error("source world does not match its pinned identity or sampled public root")]
    SourceWorldIdentityMismatch,
    #[error("terminal trajectory ends before canonical game-over")]
    NonterminalTrajectory,
    #[error("unsupported or non-proxy terminal pair schema")]
    WrongPairSchema,
    #[error("trajectory row counts are incomplete")]
    IncompleteTrajectory,
    #[error("trajectory field {field} has {actual} entries; hard maximum is {maximum}")]
    TrajectoryCollectionLimitExceeded {
        field: &'static str,
        actual: usize,
        maximum: usize,
    },
    #[error("terminal-pair ledger has {actual} bytes; hard maximum is {maximum}")]
    LedgerByteLimitExceeded { actual: u64, maximum: u64 },
    #[error("trajectory state hash mismatch at step {0}")]
    TrajectoryStateHashMismatch(usize),
    #[error("trajectory final score vector mismatch")]
    FinalScoreMismatch,
    #[error("trajectory canonical content hash mismatch")]
    TrajectoryContentHashMismatch,
    #[error("terminal target-seat score is absent")]
    MissingTargetScore,
    #[error("terminal-pair scenario coordinates are inconsistent")]
    ScenarioCoordinateMismatch,
    #[error("terminal-pair target score difference is inconsistent")]
    PairDifferenceMismatch,
    #[error("terminal-pair canonical content hash mismatch")]
    PairContentHashMismatch,
    #[error("terminal-pair hash does not match the caller-pinned identity")]
    PairPinMismatch,
    #[error("terminal-pair parent manifest does not match the caller-pinned identity")]
    ParentManifestPinMismatch,
    #[error("unsupported or internally inconsistent verified terminal-pair receipt")]
    WrongVerifiedReceiptSchema,
    #[error("verified terminal-pair receipt canonical content hash mismatch")]
    VerifiedReceiptContentHashMismatch,
    #[error("branch continuation policy identities differ")]
    ContinuationIdentityMismatch,
    #[error("policy returned a decision kind invalid at the current root")]
    WrongDecisionKind,
    #[error("policy selected outside its supplied menu")]
    PolicySelectedOutsideMenu,
    #[error("sequential reveal execution differs from the canonical compound turn")]
    CompoundChronologyMismatch,
    #[error("policy exceeded the hard per-turn decision bound")]
    DecisionLimitExceeded,
    #[error("policy decision ordinal overflowed")]
    DecisionOrdinalOverflow,
    #[error("policy menu exceeds the ledger's u32 index space")]
    MenuTooLarge,
    #[error(transparent)]
    Rules(#[from] RuleError),
    #[error(transparent)]
    Observation(#[from] ObservationError),
    #[error(transparent)]
    Menu(#[from] crate::MenuError),
    #[error(transparent)]
    Ledger(#[from] crate::LedgerError),
    #[error(transparent)]
    ActionId(#[from] crate::ActionIdError),
    #[error(transparent)]
    PolicyIdentity(#[from] crate::PolicyIdentityError),
    #[error(transparent)]
    Scenario(#[from] crate::ScenarioError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("policy failed: {0}")]
    Policy(#[source] BoxedPolicyError),
}

#[cfg(test)]
mod tests {
    use crate::{
        FailureBehavior, FailureDisposition, ForbiddenCapabilities, NumericalMode, PiLIdentity,
        PolicyDecision, PolicyIdentityFields, Precision, RNG_CONTRACT_ID, RivalSeed,
        RngContractIdentity, RootKind, RulesLegalMenu,
    };
    use cascadia_game::GameConfig;
    use std::convert::Infallible;
    use std::sync::OnceLock;

    use super::*;

    #[derive(Clone)]
    struct FirstLegalProxy {
        identity: PiLIdentity,
        calls: u8,
    }

    impl FrozenPolicy for FirstLegalProxy {
        type Identity = PiLIdentity;
        type Error = Infallible;

        fn identity(&self) -> &Self::Identity {
            &self.identity
        }

        fn fresh_instance(&self) -> Self {
            Self {
                identity: self.identity.clone(),
                calls: 0,
            }
        }

        fn act(
            &mut self,
            _observation: &crate::PublicPolicyObs,
            menu: &RulesLegalMenu,
            _worlds: &HonestWorldSampler,
            _rng: &mut crate::PolicyRng,
        ) -> Result<PolicyDecision, Self::Error> {
            self.calls = self.calls.checked_add(1).expect("bounded fixture calls");
            let index = match menu.root_kind() {
                RootKind::PreludePolicyRoot => 0,
                RootKind::DraftPolicyRoot => menu.first_draft_index().unwrap(),
            };
            Ok(PolicyDecision::new(index, menu, SeatLocalMemory::new(vec![self.calls])).unwrap())
        }
    }

    #[derive(Clone)]
    struct ScriptedChronologyProxy {
        identity: PiLIdentity,
        accept_free_replacement: bool,
        paid_wipes_to_select: u8,
        paid_wipes_selected: u8,
        calls: u8,
    }

    impl FrozenPolicy for ScriptedChronologyProxy {
        type Identity = PiLIdentity;
        type Error = Infallible;

        fn identity(&self) -> &Self::Identity {
            &self.identity
        }

        fn fresh_instance(&self) -> Self {
            Self {
                identity: self.identity.clone(),
                accept_free_replacement: self.accept_free_replacement,
                paid_wipes_to_select: self.paid_wipes_to_select,
                paid_wipes_selected: 0,
                calls: 0,
            }
        }

        fn act(
            &mut self,
            _observation: &crate::PublicPolicyObs,
            menu: &RulesLegalMenu,
            _worlds: &HonestWorldSampler,
            _rng: &mut crate::PolicyRng,
        ) -> Result<PolicyDecision, Self::Error> {
            self.calls = self.calls.checked_add(1).expect("bounded fixture calls");
            let index = match menu.root_kind() {
                RootKind::PreludePolicyRoot if self.accept_free_replacement => menu
                    .decisions()
                    .iter()
                    .position(|decision| {
                        matches!(
                            decision,
                            RulesDecision::Prelude(prelude) if prelude.replace_three_of_a_kind
                        )
                    })
                    .expect("scripted fixture has an accept choice"),
                RootKind::PreludePolicyRoot => 0,
                RootKind::DraftPolicyRoot
                    if self.paid_wipes_selected < self.paid_wipes_to_select =>
                {
                    self.paid_wipes_selected += 1;
                    menu.decisions()
                        .iter()
                        .position(|decision| matches!(decision, RulesDecision::PaidWipe(_)))
                        .expect("scripted fixture has another paid wipe")
                }
                RootKind::DraftPolicyRoot => menu.first_draft_index().unwrap(),
            };
            Ok(PolicyDecision::new(index, menu, SeatLocalMemory::new(vec![self.calls])).unwrap())
        }
    }

    fn digest(label: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(label.as_bytes())
    }

    fn proxy_identity() -> PiLIdentity {
        PiLIdentity::new(PolicyIdentityFields {
            ruleset: crate::ResearchRulesetIdentity::canonical(),
            source_revision: "cpu-fixture".to_owned(),
            source_digest: digest("source"),
            executable_sha256: digest("executable"),
            model_manifest_sha256: digest("model-manifest"),
            checkpoint_sha256: digest("checkpoint"),
            weights_sha256: digest("weights"),
            bridge_protocol: "none-cpu-proxy".to_owned(),
            tensor_schema: "none-cpu-proxy".to_owned(),
            numerical_mode: NumericalMode::Deterministic,
            precision: Precision::Fp32,
            gumbel_config_sha256: digest("gumbel"),
            search_config_sha256: digest("search"),
            refresh_config_sha256: digest("refresh"),
            exact_endgame_config_sha256: digest("endgame"),
            action_content_id_version: crate::ACTION_CONTENT_ID_VERSION.to_owned(),
            rules_action_occurrence_id_version: crate::ROOT_ACTION_OCCURRENCE_ID_VERSION.to_owned(),
            candidate_action_occurrence_id_version: crate::CANDIDATE_ACTION_OCCURRENCE_ID_VERSION
                .to_owned(),
            rules_menu_hash_version: crate::RULES_MENU_HASH_VERSION.to_owned(),
            incumbent_menu_hash_version: crate::INCUMBENT_MENU_HASH_VERSION.to_owned(),
            rng_contracts: RngContractIdentity {
                physical: RNG_CONTRACT_ID.to_owned(),
                policy: RNG_CONTRACT_ID.to_owned(),
                redetermination: RNG_CONTRACT_ID.to_owned(),
                search: RNG_CONTRACT_ID.to_owned(),
                tie_break: RNG_CONTRACT_ID.to_owned(),
            },
            public_observation_schema: crate::PUBLIC_POLICY_OBSERVATION_SCHEMA_ID.to_owned(),
            policy_memory_schema: crate::SEAT_LOCAL_MEMORY_SCHEMA_ID.to_owned(),
            failure_behavior: FailureBehavior {
                timeout: FailureDisposition::RecordIncompleteNoLabel,
                incomplete_unit: FailureDisposition::RecordIncompleteNoLabel,
                oom: FailureDisposition::RecordIncompleteNoLabel,
                fallback: FailureDisposition::Forbidden,
            },
            compiler_identity: crate::DENSE_COMPILER_ID.to_owned(),
            simulator_identity: "cascadia-game-canonical".to_owned(),
            sampler_identity: crate::INDEPENDENT_SCENARIO_SAMPLER_ID.to_owned(),
            candidate_generator_identity: "first-legal-proxy".to_owned(),
            forbidden_capabilities: ForbiddenCapabilities {
                table_total_utility: false,
                table_native_q: false,
                true_hidden_peeking: false,
                model_fallback: false,
            },
        })
        .unwrap()
    }

    fn late_source() -> GameState {
        let mut game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(123),
        )
        .unwrap();
        while game.completed_turns() < 75 {
            let action = game
                .legal_turn_actions(&MarketPrelude::default())
                .unwrap()
                .into_iter()
                .next()
                .unwrap();
            game.apply(&action).unwrap();
        }
        game
    }

    fn build_pair() -> ProxyTerminalPair {
        let source = late_source();
        let rules_menu = MenuComposer::draft_root(&source, &MarketPrelude::default()).unwrap();
        let draft_indices: Vec<_> = rules_menu
            .decisions()
            .iter()
            .enumerate()
            .filter_map(|(index, decision)| {
                matches!(decision, RulesDecision::Draft(_)).then_some(index)
            })
            .take(2)
            .collect();
        assert_eq!(draft_indices.len(), 2);
        let candidate_menu =
            IncumbentCandidateMenu::from_rules_indices(&rules_menu, draft_indices).unwrap();
        let outer_factory = RngFactory::new(RivalSeed::from_u64(501));
        let source_memories =
            std::array::from_fn(|seat| SeatLocalMemory::new(vec![10 + seat as u8]));
        let sampler =
            IndependentScenarioSampler::new(source.clone(), source_memories, outer_factory)
                .unwrap();
        let inner_factory = RngFactory::new(RivalSeed::from_u64(502));
        let panel = digest("panel-h");
        let incumbent_coordinate = ScenarioCoordinate {
            panel_id: panel.clone(),
            unit_index: 0,
            branch: EvaluationBranch::Incumbent,
            fidelity: Fidelity::High,
        };
        let challenger_coordinate = ScenarioCoordinate {
            panel_id: panel,
            unit_index: 0,
            branch: EvaluationBranch::Challenger(1),
            fidelity: Fidelity::High,
        };
        let identity = proxy_identity();
        run_proxy_terminal_pair(ProxyTerminalPairRequest {
            sampler: &sampler,
            parent_manifest_sha256: digest("parent-manifest"),
            incumbent_coordinate: &incumbent_coordinate,
            challenger_coordinate: &challenger_coordinate,
            candidate_menu: &candidate_menu,
            incumbent_candidate_index: 0,
            challenger_candidate_index: 1,
            incumbent_post_action_memory: SeatLocalMemory::new(vec![0xa1]),
            challenger_post_action_memory: SeatLocalMemory::new(vec![0xb1]),
            policy_prototype: FirstLegalProxy {
                identity: identity.clone(),
                calls: 99,
            },
            rng_factory: &inner_factory,
            target_seat: SeatIndex::new(source.current_player() as u8).unwrap(),
        })
        .unwrap()
    }

    fn pair_fixture() -> ProxyTerminalPair {
        static PAIR: OnceLock<ProxyTerminalPair> = OnceLock::new();
        PAIR.get_or_init(build_pair).clone()
    }

    #[test]
    fn proxy_pair_is_replay_complete_and_cannot_claim_multifidelity() {
        let pair = pair_fixture();
        pair.validate().unwrap();
        assert!(pair.proxy_policy());
        assert_eq!(pair.beta_cv_required(), 0);
        assert_eq!(pair.incumbent().actions().len(), 5);
        assert_eq!(pair.challenger().actions().len(), 5);
        assert!(matches!(
            pair.validate_pinned(&digest("wrong-pair"), &pair.parent_manifest_sha256),
            Err(TerminalError::PairPinMismatch)
        ));
        assert!(matches!(
            pair.validate_pinned(&pair.pair_sha256, &digest("wrong-parent")),
            Err(TerminalError::ParentManifestPinMismatch)
        ));
    }

    #[test]
    fn challenger_branch_is_candidate_index_derived_and_not_substitutable() {
        let source = late_source();
        let rules_menu = MenuComposer::draft_root(&source, &MarketPrelude::default()).unwrap();
        let draft_indices = rules_menu
            .decisions()
            .iter()
            .enumerate()
            .filter_map(|(index, decision)| {
                matches!(decision, RulesDecision::Draft(_)).then_some(index)
            })
            .take(2)
            .collect::<Vec<_>>();
        let candidate_menu =
            IncumbentCandidateMenu::from_rules_indices(&rules_menu, draft_indices).unwrap();
        let sampler = IndependentScenarioSampler::new(
            source.clone(),
            std::array::from_fn(|seat| SeatLocalMemory::new(vec![10 + seat as u8])),
            RngFactory::new(RivalSeed::from_u64(501)),
        )
        .unwrap();
        let panel_id = digest("branch-binding-panel");
        let incumbent_coordinate = ScenarioCoordinate {
            panel_id: panel_id.clone(),
            unit_index: 91,
            branch: EvaluationBranch::Incumbent,
            fidelity: Fidelity::High,
        };
        let valid_challenger_coordinate = ScenarioCoordinate {
            panel_id: panel_id.clone(),
            unit_index: 91,
            branch: EvaluationBranch::Challenger(1),
            fidelity: Fidelity::High,
        };
        let substituted_challenger_coordinate = ScenarioCoordinate {
            panel_id,
            unit_index: 91,
            branch: EvaluationBranch::Challenger(2),
            fidelity: Fidelity::High,
        };
        assert_ne!(
            sampler
                .sample(&valid_challenger_coordinate)
                .redetermination_seed(),
            sampler
                .sample(&substituted_challenger_coordinate)
                .redetermination_seed(),
            "changing only the branch ordinal must change the physical world seed"
        );

        let inner_factory = RngFactory::new(RivalSeed::from_u64(502));
        let error = run_proxy_terminal_pair(ProxyTerminalPairRequest {
            sampler: &sampler,
            parent_manifest_sha256: digest("parent-manifest"),
            incumbent_coordinate: &incumbent_coordinate,
            challenger_coordinate: &substituted_challenger_coordinate,
            candidate_menu: &candidate_menu,
            incumbent_candidate_index: 0,
            challenger_candidate_index: 1,
            incumbent_post_action_memory: SeatLocalMemory::new(vec![0xa1]),
            challenger_post_action_memory: SeatLocalMemory::new(vec![0xb1]),
            policy_prototype: FirstLegalProxy {
                identity: proxy_identity(),
                calls: 99,
            },
            rng_factory: &inner_factory,
            target_seat: SeatIndex::new(source.current_player() as u8).unwrap(),
        })
        .unwrap_err();
        assert!(matches!(error, TerminalError::ScenarioCoordinateMismatch));

        let mut substituted_pair = pair_fixture();
        substituted_pair.challenger.scenario_coordinate.branch = EvaluationBranch::Challenger(2);
        substituted_pair.challenger.trajectory_sha256 =
            substituted_pair.challenger.recompute_hash().unwrap();
        substituted_pair.pair_sha256 = substituted_pair.recompute_hash().unwrap();
        let bytes = serde_json::to_vec(&substituted_pair).unwrap();
        assert!(ProxyTerminalPair::from_json_slice(&bytes).is_err());
    }

    #[test]
    fn continuation_uses_four_seat_isolated_policy_instances_and_branch_memory() {
        let pair = pair_fixture();
        for (trajectory, target_memory) in [
            (&pair.incumbent, &[0xa1][..]),
            (&pair.challenger, &[0xb1][..]),
        ] {
            let decisions = &trajectory.turn_root_decisions[1..];
            assert_eq!(decisions.len(), 4);
            let mut seen = [false; 4];
            for turn in decisions {
                let first = turn.first().expect("every continuation turn drafts");
                let seat = usize::from(first.public_observation.seat().get());
                assert!(
                    !seen[seat],
                    "late fixture gives each seat one continuation turn"
                );
                seen[seat] = true;
                assert_eq!(
                    first.next_memory.payload(),
                    &[1],
                    "each seat-local policy instance must begin at call one"
                );
                let expected_memory = if seat == usize::from(trajectory.target_seat.get()) {
                    target_memory
                } else {
                    trajectory.source_memories[seat].payload()
                };
                assert_eq!(first.public_observation.memory().payload(), expected_memory);
            }
            assert!(seen.into_iter().all(|value| value));
        }
    }

    fn prelude_fixture(choice_count: usize) -> GameState {
        for seed in 0..2_000 {
            let game = GameState::new(
                GameConfig::research_aaaaa(4).unwrap(),
                GameSeed::from_u64(seed),
            )
            .unwrap();
            if MenuComposer::prelude_root(&game).unwrap().len() == choice_count {
                return game;
            }
        }
        panic!("fixture search found no prelude menu of length {choice_count}");
    }

    fn proxy_coordinate() -> ScenarioCoordinate {
        ScenarioCoordinate {
            panel_id: digest("singleton-semantics-panel"),
            unit_index: 17,
            branch: EvaluationBranch::Incumbent,
            fidelity: Fidelity::High,
        }
    }

    #[test]
    fn singleton_decline_consumes_no_policy_call_memory_transition_or_rng_ordinal() {
        let source = prelude_fixture(1);
        let identity = proxy_identity();
        let mut policy = FirstLegalProxy { identity, calls: 0 };
        let rng = RngFactory::new(RivalSeed::from_u64(8_001));
        let mut memories = PolicyMemoryBank::from_four(std::array::from_fn(|seat| {
            SeatLocalMemory::new(vec![20 + seat as u8])
        }));
        let before = memories.snapshot_four().unwrap();
        let mut ordinal = 0;
        let (_, records) = select_proxy_turn(
            &source,
            &mut policy,
            &rng,
            &proxy_coordinate(),
            &mut memories,
            &mut ordinal,
        )
        .unwrap();
        assert_eq!(policy.calls, 1, "only the draft root invokes the policy");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].selected_kind, SelectedDecisionKind::Draft);
        assert_eq!(records[0].decision_ordinal, 0);
        assert_eq!(ordinal, 1);
        let after = memories.snapshot_four().unwrap();
        for (observed, expected) in after.iter().zip(&before).skip(1) {
            assert_eq!(observed, expected);
        }
    }

    #[test]
    fn genuine_prelude_choice_is_recorded_before_the_draft() {
        let source = prelude_fixture(2);
        let mut policy = FirstLegalProxy {
            identity: proxy_identity(),
            calls: 0,
        };
        let rng = RngFactory::new(RivalSeed::from_u64(8_002));
        let mut memories = PolicyMemoryBank::new(4).unwrap();
        let mut ordinal = 0;
        let (action, records) = select_proxy_turn(
            &source,
            &mut policy,
            &rng,
            &proxy_coordinate(),
            &mut memories,
            &mut ordinal,
        )
        .unwrap();
        assert_eq!(policy.calls, 2);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].selected_kind, SelectedDecisionKind::Prelude);
        assert_eq!(records[1].selected_kind, SelectedDecisionKind::Draft);
        assert_eq!(
            (records[0].decision_ordinal, records[1].decision_ordinal),
            (0, 1)
        );
        assert!(!action.replace_three_of_a_kind);
    }

    #[test]
    fn accepted_free_replacement_is_compound_and_replay_complete() {
        let source = prelude_fixture(2);
        let mut policy = ScriptedChronologyProxy {
            identity: proxy_identity(),
            accept_free_replacement: true,
            paid_wipes_to_select: 0,
            paid_wipes_selected: 0,
            calls: 0,
        };
        let rng = RngFactory::new(RivalSeed::from_u64(8_003));
        let mut memories = PolicyMemoryBank::new(4).unwrap();
        let mut ordinal = 0;
        let (action, records) = select_proxy_turn(
            &source,
            &mut policy,
            &rng,
            &proxy_coordinate(),
            &mut memories,
            &mut ordinal,
        )
        .unwrap();
        assert!(action.replace_three_of_a_kind);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].selected_kind, SelectedDecisionKind::Prelude);
        assert_eq!(records[1].selected_kind, SelectedDecisionKind::Draft);
        assert_eq!(
            (records[0].decision_ordinal, records[1].decision_ordinal),
            (0, 1)
        );
        source.transition(&action).unwrap();

        let mut replay_memories = PolicyMemoryBank::new(4).unwrap();
        let mut replay_ordinal = 0;
        replay_policy_decision_trace(
            &source,
            &action,
            &records,
            &mut replay_memories,
            &mut replay_ordinal,
        )
        .unwrap();
        assert_eq!(replay_ordinal, 2);
        assert_eq!(
            replay_memories.snapshot_four().unwrap(),
            memories.snapshot_four().unwrap()
        );
    }

    fn with_current_nature_tokens(game: GameState, count: u8) -> GameState {
        let actor = game.current_player();
        let mut value = serde_json::to_value(game).unwrap();
        value["boards"][actor]["nature_tokens"] = serde_json::json!(count);
        let game: GameState = serde_json::from_value(value).unwrap();
        game.validate().unwrap();
        game
    }

    #[test]
    fn sequential_paid_wipes_reveal_recompose_and_replay_in_order() {
        let source = with_current_nature_tokens(prelude_fixture(1), 2);
        let mut policy = ScriptedChronologyProxy {
            identity: proxy_identity(),
            accept_free_replacement: false,
            paid_wipes_to_select: 2,
            paid_wipes_selected: 0,
            calls: 0,
        };
        let rng = RngFactory::new(RivalSeed::from_u64(8_004));
        let mut memories = PolicyMemoryBank::new(4).unwrap();
        let mut ordinal = 0;
        let (action, records) = select_proxy_turn(
            &source,
            &mut policy,
            &rng,
            &proxy_coordinate(),
            &mut memories,
            &mut ordinal,
        )
        .unwrap();
        assert_eq!(action.wildlife_wipes.len(), 2);
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].selected_kind, SelectedDecisionKind::PaidWipe);
        assert_eq!(records[1].selected_kind, SelectedDecisionKind::PaidWipe);
        assert_eq!(records[2].selected_kind, SelectedDecisionKind::Draft);
        assert_eq!(
            records
                .iter()
                .map(|record| record.decision_ordinal)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        source.transition(&action).unwrap();

        let mut replay_memories = PolicyMemoryBank::new(4).unwrap();
        let mut replay_ordinal = 0;
        replay_policy_decision_trace(
            &source,
            &action,
            &records,
            &mut replay_memories,
            &mut replay_ordinal,
        )
        .unwrap();
        assert_eq!(replay_ordinal, 3);
        assert_eq!(
            replay_memories.snapshot_four().unwrap(),
            memories.snapshot_four().unwrap()
        );
    }

    #[test]
    fn semantic_tampering_fails_even_when_content_hashes_are_resealed() {
        let mut cross_seat = pair_fixture();
        cross_seat.incumbent.continuation_memories[0] = SeatLocalMemory::new(vec![0xff]);
        cross_seat.incumbent.trajectory_sha256 = cross_seat.incumbent.recompute_hash().unwrap();
        cross_seat.pair_sha256 = cross_seat.recompute_hash().unwrap();
        assert!(matches!(
            cross_seat.validate(),
            Err(TerminalError::CrossSeatForcedMemoryMutation(0))
        ));

        let mut wrong_root_memory = pair_fixture();
        let target = usize::from(wrong_root_memory.incumbent.target_seat.get());
        wrong_root_memory.incumbent.source_memories[target] = SeatLocalMemory::new(vec![0xee]);
        wrong_root_memory.incumbent.trajectory_sha256 =
            wrong_root_memory.incumbent.recompute_hash().unwrap();
        wrong_root_memory.pair_sha256 = wrong_root_memory.recompute_hash().unwrap();
        assert!(matches!(
            wrong_root_memory.validate(),
            Err(TerminalError::ForcedCandidateProvenanceMismatch)
        ));

        let mut wrong_panel = pair_fixture();
        wrong_panel.challenger.scenario_coordinate.panel_id = digest("different-panel");
        wrong_panel.challenger.trajectory_sha256 = wrong_panel.challenger.recompute_hash().unwrap();
        wrong_panel.pair_sha256 = wrong_panel.recompute_hash().unwrap();
        assert!(matches!(
            wrong_panel.validate(),
            Err(TerminalError::ScenarioCoordinateMismatch)
        ));
    }

    #[test]
    fn pair_deserialization_and_pinned_receipt_generation_are_fail_closed() {
        let pair = pair_fixture();
        let bytes = serde_json::to_vec_pretty(&pair).unwrap();
        assert_eq!(ProxyTerminalPair::from_json_slice(&bytes).unwrap(), pair);

        let (decoded_pair, receipt) = ProxyTerminalPair::verify_bytes_and_create_receipt(
            &bytes,
            b"verifier-executable",
            &pair.pair_sha256,
            &pair.parent_manifest_sha256,
        )
        .unwrap();
        assert_eq!(decoded_pair, pair);
        receipt.validate().unwrap();
        let canonical_ruleset = serde_json::to_value(ResearchRulesetIdentity::canonical()).unwrap();
        assert_eq!(
            receipt.ruleset_identity_sha256(),
            &Sha256Digest::of_bytes(&serde_json::to_vec(&canonical_ruleset).unwrap())
        );
        assert_eq!(
            serde_json::to_value(&receipt).unwrap()["schema_id"],
            VERIFIED_TERMINAL_PAIR_RECEIPT_SCHEMA_ID
        );
        assert_eq!(receipt.challenger_branch_ordinal(), 1);
        assert_eq!(
            serde_json::to_value(&receipt).unwrap()["challenger_branch_ordinal"],
            serde_json::json!(1)
        );
        assert_eq!(
            receipt.incumbent_post_action_memory_sha256,
            seat_memory_sha256(&SeatLocalMemory::new(vec![0xa1])).unwrap()
        );
        assert_eq!(
            receipt.challenger_post_action_memory_sha256,
            seat_memory_sha256(&SeatLocalMemory::new(vec![0xb1])).unwrap()
        );
        assert_ne!(
            receipt.incumbent_post_action_memory_sha256,
            receipt.challenger_post_action_memory_sha256
        );
        assert_eq!(
            receipt.incumbent_world_redetermination_seed_sha256(),
            &Sha256Digest::of_bytes(&pair.incumbent.world_redetermination_seed.0)
        );
        assert_eq!(
            receipt.challenger_world_redetermination_seed_sha256(),
            &Sha256Digest::of_bytes(&pair.challenger.world_redetermination_seed.0)
        );
        assert_ne!(
            receipt.incumbent_world_redetermination_seed_sha256(),
            receipt.challenger_world_redetermination_seed_sha256()
        );

        let mut resealed_wrong_ruleset = receipt.clone();
        resealed_wrong_ruleset.ruleset_identity_sha256 = digest("wrong-ruleset");
        resealed_wrong_ruleset.receipt_sha256 = resealed_wrong_ruleset.recompute_hash().unwrap();
        assert!(matches!(
            resealed_wrong_ruleset.validate(),
            Err(TerminalError::WrongVerifiedReceiptSchema)
        ));

        let mut pair_value = serde_json::to_value(&pair).unwrap();
        pair_value["unexpected"] = serde_json::json!(true);
        assert!(serde_json::from_value::<ProxyTerminalPair>(pair_value).is_err());
    }

    #[test]
    fn terminal_ledger_and_nested_trajectory_limits_fail_closed() {
        assert!(matches!(
            ensure_terminal_ledger_byte_limit(
                usize::try_from(MAX_TERMINAL_PAIR_LEDGER_BYTES).unwrap() + 1
            ),
            Err(TerminalError::LedgerByteLimitExceeded { .. })
        ));

        let pair = pair_fixture();
        let mut too_many_turns = serde_json::to_value(&pair).unwrap();
        let action = too_many_turns["challenger"]["actions"][0].clone();
        too_many_turns["challenger"]["actions"] =
            serde_json::Value::Array(vec![action; MAX_TERMINAL_TRAJECTORY_TURNS + 1]);
        assert!(serde_json::from_value::<ProxyTerminalPair>(too_many_turns).is_err());

        let mut too_many_decisions = pair.clone();
        let record = too_many_decisions.challenger.turn_root_decisions[1][0].clone();
        too_many_decisions.challenger.turn_root_decisions[1] =
            vec![record; MAX_POLICY_DECISIONS_PER_TURN + 1];
        assert!(matches!(
            too_many_decisions.challenger.replay(),
            Err(TerminalError::TrajectoryCollectionLimitExceeded {
                field: "turn_root_decisions[]",
                ..
            })
        ));
        let too_many_decisions = serde_json::to_value(&too_many_decisions).unwrap();
        assert!(serde_json::from_value::<ProxyTerminalPair>(too_many_decisions).is_err());
    }
}
