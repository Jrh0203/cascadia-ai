//! Immutable canonical trajectory ledgers and deterministic replay.
//!
//! A summary is never the primary evidence.  This ledger stores every applied
//! complete turn, its canonical before/after hashes, additive action IDs, and
//! the public policy-root decisions that produced it.  Replay starts from the
//! canonical game seed and refuses the first mismatch.

use std::{
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use cascadia_game::{
    GameConfig, GameSeed, GameState, RuleError, ScoreBreakdown, TurnAction, score_game,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    ActionContentId, ActionIdError, LegacyActionIdV0, MenuComposer, PolicyMemoryBank,
    PrivateSimState, PublicPolicyObs, PublicRootId, ResearchRulesetIdentity,
    RootActionOccurrenceId, RootKind, RulesDecision, RulesMenuHash, SeatIndex, SeatLocalMemory,
    Sha256Digest,
};

pub const TRAJECTORY_LEDGER_SCHEMA_ID: &str = "cascadiav3.rival_trajectory_ledger.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectedDecisionKind {
    Prelude,
    PaidWipe,
    Draft,
}

/// One policy invocation.  It intentionally contains no outer physical key,
/// hidden inventory, event priority, or true-state bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RootDecisionRecord {
    pub decision_ordinal: u32,
    pub root_kind: RootKind,
    pub public_observation: PublicPolicyObs,
    pub public_root_id: PublicRootId,
    pub ordered_menu_hash: RulesMenuHash,
    pub menu_len: u32,
    pub selected_index: u32,
    pub selected_kind: SelectedDecisionKind,
    pub draft_occurrence_id: Option<RootActionOccurrenceId>,
    pub next_memory: SeatLocalMemory,
}

impl RootDecisionRecord {
    pub fn validate(&self) -> Result<(), LedgerError> {
        if self.public_observation.seat().get() >= 4 {
            return Err(LedgerError::DecisionObservationMismatch);
        }
        if self.menu_len == 0 || self.selected_index >= self.menu_len {
            return Err(LedgerError::InvalidDecisionIndex {
                selected: self.selected_index,
                menu_len: self.menu_len,
            });
        }
        match (self.root_kind, self.selected_kind) {
            (RootKind::PreludePolicyRoot, SelectedDecisionKind::Prelude)
            | (RootKind::DraftPolicyRoot, SelectedDecisionKind::PaidWipe)
            | (RootKind::DraftPolicyRoot, SelectedDecisionKind::Draft) => {}
            _ => return Err(LedgerError::DecisionKindRootMismatch),
        }
        if (self.selected_kind == SelectedDecisionKind::Draft) != self.draft_occurrence_id.is_some()
        {
            return Err(LedgerError::DraftOccurrenceMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnEvidenceKind {
    /// A deterministic parity/replay fixture, not a policy claim.
    CanonicalFixture,
    /// An externally frozen first action; no policy decision trace is implied.
    ForcedAction,
    /// A complete public-root policy trace from prelude through final draft.
    PolicyDecisionTrace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LedgerCompletion {
    Partial,
    Terminal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TurnLedgerRecord {
    pub turn_index: u16,
    pub actor: u8,
    pub evidence_kind: TurnEvidenceKind,
    pub pre_state_hash: [u8; 32],
    pub root_decisions: Vec<RootDecisionRecord>,
    pub legacy_action_id_v0: LegacyActionIdV0,
    pub action_content_id: ActionContentId,
    pub action: TurnAction,
    pub post_state_hash: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "TrajectoryLedgerWire", into = "TrajectoryLedgerWire")]
pub struct TrajectoryLedger {
    schema_id: String,
    ruleset: ResearchRulesetIdentity,
    source_game_id: String,
    config: GameConfig,
    seed: GameSeed,
    initial_state_hash: [u8; 32],
    turns: Vec<TurnLedgerRecord>,
    completion: LedgerCompletion,
    final_state_hash: [u8; 32],
    terminal_scores: Option<Vec<ScoreBreakdown>>,
    ledger_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrajectoryLedgerWire {
    schema_id: String,
    ruleset: ResearchRulesetIdentity,
    source_game_id: String,
    config: GameConfig,
    seed: GameSeed,
    initial_state_hash: [u8; 32],
    turns: Vec<TurnLedgerRecord>,
    completion: LedgerCompletion,
    final_state_hash: [u8; 32],
    terminal_scores: Option<Vec<ScoreBreakdown>>,
    ledger_sha256: Sha256Digest,
}

#[derive(Serialize)]
struct LedgerContent<'a> {
    schema_id: &'a str,
    ruleset: &'a ResearchRulesetIdentity,
    source_game_id: &'a str,
    config: GameConfig,
    seed: GameSeed,
    initial_state_hash: [u8; 32],
    turns: &'a [TurnLedgerRecord],
    completion: LedgerCompletion,
    final_state_hash: [u8; 32],
    terminal_scores: &'a Option<Vec<ScoreBreakdown>>,
}

impl TrajectoryLedger {
    pub fn replay(&self) -> Result<GameState, LedgerError> {
        self.validate_static_fields()?;
        let mut game = GameState::new(self.config, self.seed)?;
        if *game.canonical_hash().as_bytes() != self.initial_state_hash {
            return Err(LedgerError::InitialStateHashMismatch);
        }
        let mut memories = PolicyMemoryBank::new(4)?;
        let mut expected_decision_ordinal = 0u32;
        for (expected_index, record) in self.turns.iter().enumerate() {
            if record.turn_index as usize != expected_index {
                return Err(LedgerError::TurnIndexMismatch {
                    expected: expected_index as u16,
                    observed: record.turn_index,
                });
            }
            if game.current_player() != usize::from(record.actor) {
                return Err(LedgerError::ActorMismatch {
                    turn: record.turn_index,
                    expected: game.current_player() as u8,
                    observed: record.actor,
                });
            }
            if *game.canonical_hash().as_bytes() != record.pre_state_hash {
                return Err(LedgerError::PreStateHashMismatch(record.turn_index));
            }
            match record.evidence_kind {
                TurnEvidenceKind::CanonicalFixture | TurnEvidenceKind::ForcedAction => {
                    if !record.root_decisions.is_empty() {
                        return Err(LedgerError::UnexpectedDecisionTrace(record.turn_index));
                    }
                }
                TurnEvidenceKind::PolicyDecisionTrace => {
                    replay_policy_decision_trace(
                        &game,
                        &record.action,
                        &record.root_decisions,
                        &mut memories,
                        &mut expected_decision_ordinal,
                    )?;
                }
            }
            if LegacyActionIdV0::new(&record.action)? != record.legacy_action_id_v0 {
                return Err(LedgerError::LegacyActionIdMismatch(record.turn_index));
            }
            if ActionContentId::canonical(&record.action) != record.action_content_id {
                return Err(LedgerError::ActionContentIdMismatch(record.turn_index));
            }
            game.apply(&record.action)?;
            if *game.canonical_hash().as_bytes() != record.post_state_hash {
                return Err(LedgerError::PostStateHashMismatch(record.turn_index));
            }
        }
        if *game.canonical_hash().as_bytes() != self.final_state_hash {
            return Err(LedgerError::FinalStateHashMismatch);
        }
        match (&self.completion, &self.terminal_scores) {
            (LedgerCompletion::Terminal, Some(expected)) if game.is_game_over() => {
                if &score_game(&game) != expected {
                    return Err(LedgerError::FinalScoreMismatch);
                }
            }
            (LedgerCompletion::Partial, None) if !game.is_game_over() => {}
            _ => return Err(LedgerError::CompletionMismatch),
        }
        if self.recompute_hash()? != self.ledger_sha256 {
            return Err(LedgerError::LedgerHashMismatch);
        }
        Ok(game)
    }

    pub fn verify(&self) -> Result<(), LedgerError> {
        self.replay().map(|_| ())
    }

    pub fn turns(&self) -> &[TurnLedgerRecord] {
        &self.turns
    }

    pub fn config(&self) -> GameConfig {
        self.config
    }

    pub fn seed(&self) -> GameSeed {
        self.seed
    }

    pub fn source_game_id(&self) -> &str {
        &self.source_game_id
    }

    /// The canonical pre-game state this ledger's trajectory started from,
    /// hash-verified against the sealed initial-state commitment.
    pub fn initial_state(&self) -> Result<GameState, LedgerError> {
        self.validate_static_fields()?;
        let game = GameState::new(self.config, self.seed)?;
        if *game.canonical_hash().as_bytes() != self.initial_state_hash {
            return Err(LedgerError::InitialStateHashMismatch);
        }
        Ok(game)
    }

    /// Hash-anchored physical replay: applies the sealed actions in order
    /// and returns the state after every turn, checking each transition
    /// against its sealed pre/post canonical hashes.
    ///
    /// Unlike [`Self::replay`], this does not re-verify policy decision
    /// traces (which recompose every root menu, an expensive canonical
    /// proof).  Every constructed [`TrajectoryLedger`] value has already
    /// passed the full verification at its sealing or deserialization
    /// boundary; consumers that only need the physical trajectory can rely
    /// on the sealed per-turn hash anchors.
    pub fn raw_state_trajectory(&self) -> Result<Vec<GameState>, LedgerError> {
        let mut state = self.initial_state()?;
        let mut states = Vec::with_capacity(self.turns.len());
        for record in &self.turns {
            if *state.canonical_hash().as_bytes() != record.pre_state_hash {
                return Err(LedgerError::PreStateHashMismatch(record.turn_index));
            }
            state = state.transition(&record.action)?;
            if *state.canonical_hash().as_bytes() != record.post_state_hash {
                return Err(LedgerError::PostStateHashMismatch(record.turn_index));
            }
            states.push(state.clone());
        }
        if *state.canonical_hash().as_bytes() != self.final_state_hash {
            return Err(LedgerError::FinalStateHashMismatch);
        }
        Ok(states)
    }

    /// The hash-anchored final state of [`Self::raw_state_trajectory`].
    pub fn raw_final_state(&self) -> Result<GameState, LedgerError> {
        let mut state = self.initial_state()?;
        for record in &self.turns {
            state = state.transition(&record.action)?;
            if *state.canonical_hash().as_bytes() != record.post_state_hash {
                return Err(LedgerError::PostStateHashMismatch(record.turn_index));
            }
        }
        if *state.canonical_hash().as_bytes() != self.final_state_hash {
            return Err(LedgerError::FinalStateHashMismatch);
        }
        Ok(state)
    }

    pub fn completion(&self) -> LedgerCompletion {
        self.completion
    }

    pub fn final_state_hash(&self) -> &[u8; 32] {
        &self.final_state_hash
    }

    pub fn terminal_scores(&self) -> Option<&[ScoreBreakdown]> {
        self.terminal_scores.as_deref()
    }

    pub fn ledger_sha256(&self) -> &Sha256Digest {
        &self.ledger_sha256
    }

    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, LedgerError> {
        let ledger: Self = serde_json::from_slice(bytes)?;
        ledger.verify()?;
        Ok(ledger)
    }

    pub fn canonical_json_bytes(&self) -> Result<Vec<u8>, LedgerError> {
        self.verify()?;
        Ok(serde_json::to_vec_pretty(self)?)
    }

    /// Durably publish through a same-directory temporary without ever
    /// replacing an existing scientific artifact.
    pub fn write_json_immutable(&self, destination: &Path) -> Result<(), LedgerError> {
        let bytes = self.canonical_json_bytes()?;
        write_immutable_bytes(destination, &bytes)
    }

    fn validate_static_fields(&self) -> Result<(), LedgerError> {
        if self.schema_id != TRAJECTORY_LEDGER_SCHEMA_ID {
            return Err(LedgerError::WrongSchema);
        }
        self.ruleset
            .validate()
            .map_err(|_| LedgerError::WrongRuleset)?;
        if self.config != GameConfig::research_aaaaa(4).expect("valid research config") {
            return Err(LedgerError::WrongRuleset);
        }
        if self.source_game_id.trim().is_empty() {
            return Err(LedgerError::EmptySourceGameId);
        }
        Ok(())
    }

    fn recompute_hash(&self) -> Result<Sha256Digest, LedgerError> {
        let content = LedgerContent {
            schema_id: &self.schema_id,
            ruleset: &self.ruleset,
            source_game_id: &self.source_game_id,
            config: self.config,
            seed: self.seed,
            initial_state_hash: self.initial_state_hash,
            turns: &self.turns,
            completion: self.completion,
            final_state_hash: self.final_state_hash,
            terminal_scores: &self.terminal_scores,
        };
        let value = serde_json::to_value(&content)?;
        Ok(Sha256Digest::of_bytes(&serde_json::to_vec(&value)?))
    }
}

impl From<TrajectoryLedger> for TrajectoryLedgerWire {
    fn from(value: TrajectoryLedger) -> Self {
        Self {
            schema_id: value.schema_id,
            ruleset: value.ruleset,
            source_game_id: value.source_game_id,
            config: value.config,
            seed: value.seed,
            initial_state_hash: value.initial_state_hash,
            turns: value.turns,
            completion: value.completion,
            final_state_hash: value.final_state_hash,
            terminal_scores: value.terminal_scores,
            ledger_sha256: value.ledger_sha256,
        }
    }
}

impl TryFrom<TrajectoryLedgerWire> for TrajectoryLedger {
    type Error = LedgerError;

    fn try_from(value: TrajectoryLedgerWire) -> Result<Self, Self::Error> {
        let ledger = Self {
            schema_id: value.schema_id,
            ruleset: value.ruleset,
            source_game_id: value.source_game_id,
            config: value.config,
            seed: value.seed,
            initial_state_hash: value.initial_state_hash,
            turns: value.turns,
            completion: value.completion,
            final_state_hash: value.final_state_hash,
            terminal_scores: value.terminal_scores,
            ledger_sha256: value.ledger_sha256,
        };
        ledger.verify()?;
        Ok(ledger)
    }
}

/// Replay one complete public policy trace against canonical root menus.
///
/// A prelude record exists only when accept/decline is a real policy choice.
/// A singleton decline is deterministic orchestration and consumes no policy
/// call, memory transition, or RNG ordinal. Zero or more one-step paid wipes
/// must then terminate in exactly one draft matching the compound action.
pub fn replay_policy_decision_trace(
    source: &GameState,
    compound_action: &TurnAction,
    records: &[RootDecisionRecord],
    memories: &mut PolicyMemoryBank,
    expected_decision_ordinal: &mut u32,
) -> Result<(), LedgerError> {
    if records.is_empty() {
        return Err(LedgerError::IncompleteDecisionTrace);
    }
    let actor = SeatIndex::new(source.current_player() as u8)?;
    let prelude_menu = MenuComposer::prelude_root(source)?;
    let turn_source = source.clone();
    let (mut staged, mut accumulated, draft_records) = if prelude_menu.len() == 1 {
        let Some(RulesDecision::Prelude(prelude)) = prelude_menu.decision(0) else {
            return Err(LedgerError::DecisionKindRootMismatch);
        };
        if prelude != &cascadia_game::MarketPrelude::default() {
            return Err(LedgerError::DecisionKindRootMismatch);
        }
        (
            source.preview_market_prelude(prelude)?,
            prelude.clone(),
            records,
        )
    } else {
        let prelude_record = &records[0];
        let expected_observation = PrivateSimState::new(source.clone())?
            .public_observation(actor, memories.get(actor)?.clone())?;
        let prelude_decision = verify_root_record(
            prelude_record,
            &expected_observation,
            &prelude_menu,
            expected_decision_ordinal,
        )?;
        let RulesDecision::Prelude(prelude) = prelude_decision else {
            return Err(LedgerError::DecisionKindRootMismatch);
        };
        memories.replace(actor, prelude_record.next_memory.clone())?;
        (
            source.preview_market_prelude(prelude)?,
            prelude.clone(),
            &records[1..],
        )
    };
    if draft_records.is_empty() {
        return Err(LedgerError::IncompleteDecisionTrace);
    }
    for (offset, record) in draft_records.iter().enumerate() {
        let menu = MenuComposer::draft_root(&staged, &cascadia_game::MarketPrelude::default())?;
        let expected_observation = PrivateSimState::new(staged.clone())?
            .public_observation(actor, memories.get(actor)?.clone())?;
        let decision = verify_root_record(
            record,
            &expected_observation,
            &menu,
            expected_decision_ordinal,
        )?;
        memories.replace(actor, record.next_memory.clone())?;
        match decision {
            RulesDecision::PaidWipe(wipe) => {
                if offset + 1 == draft_records.len() {
                    return Err(LedgerError::IncompleteDecisionTrace);
                }
                let one_step = cascadia_game::MarketPrelude {
                    replace_three_of_a_kind: false,
                    wildlife_wipes: vec![wipe.clone()],
                };
                staged = staged.preview_market_prelude(&one_step)?;
                accumulated.wildlife_wipes.push(wipe.clone());
            }
            RulesDecision::Draft(root_local_action) => {
                if offset + 1 != draft_records.len() {
                    return Err(LedgerError::TrailingDecisionAfterDraft);
                }
                let mut reconstructed = root_local_action.clone();
                reconstructed.replace_three_of_a_kind = accumulated.replace_three_of_a_kind;
                reconstructed.wildlife_wipes = accumulated.wildlife_wipes;
                if &reconstructed != compound_action {
                    return Err(LedgerError::CompoundActionMismatch);
                }
                let staged_next = staged.transition(root_local_action)?;
                let compound_next = turn_source.transition(compound_action)?;
                if staged_next.canonical_hash() != compound_next.canonical_hash() {
                    return Err(LedgerError::CompoundActionMismatch);
                }
                return Ok(());
            }
            RulesDecision::Prelude(_) => return Err(LedgerError::DecisionKindRootMismatch),
        }
    }
    Err(LedgerError::IncompleteDecisionTrace)
}

fn verify_root_record<'a>(
    record: &RootDecisionRecord,
    expected_observation: &PublicPolicyObs,
    menu: &'a crate::RulesLegalMenu,
    expected_decision_ordinal: &mut u32,
) -> Result<&'a RulesDecision, LedgerError> {
    record.validate()?;
    if record.decision_ordinal != *expected_decision_ordinal {
        return Err(LedgerError::DecisionOrdinalMismatch {
            expected: *expected_decision_ordinal,
            observed: record.decision_ordinal,
        });
    }
    *expected_decision_ordinal = expected_decision_ordinal
        .checked_add(1)
        .ok_or(LedgerError::DecisionOrdinalOverflow)?;
    if &record.public_observation != expected_observation {
        return Err(LedgerError::DecisionObservationMismatch);
    }
    let root = PublicRootId::new(expected_observation, menu.root_kind());
    if record.root_kind != menu.root_kind()
        || record.public_root_id != root
        || record.ordered_menu_hash != menu.hash()
        || record.menu_len as usize != menu.len()
    {
        return Err(LedgerError::RootReconstructionMismatch);
    }
    let selected =
        menu.decision(record.selected_index as usize)
            .ok_or(LedgerError::InvalidDecisionIndex {
                selected: record.selected_index,
                menu_len: record.menu_len,
            })?;
    let (kind, occurrence) = match selected {
        RulesDecision::Prelude(_) => (SelectedDecisionKind::Prelude, None),
        RulesDecision::PaidWipe(_) => (SelectedDecisionKind::PaidWipe, None),
        RulesDecision::Draft(_) => (
            SelectedDecisionKind::Draft,
            Some(RootActionOccurrenceId::new(
                &root,
                menu,
                record.selected_index as usize,
            )?),
        ),
    };
    if record.selected_kind != kind || record.draft_occurrence_id != occurrence {
        return Err(LedgerError::SelectedDecisionMismatch);
    }
    Ok(selected)
}

/// Validating builder; no record can be appended without a canonical apply.
pub struct TrajectoryLedgerBuilder {
    source_game_id: String,
    initial_state_hash: [u8; 32],
    game: GameState,
    turns: Vec<TurnLedgerRecord>,
}

impl TrajectoryLedgerBuilder {
    pub fn new(source_game_id: impl Into<String>, game: GameState) -> Result<Self, LedgerError> {
        let source_game_id = source_game_id.into();
        if source_game_id.trim().is_empty() {
            return Err(LedgerError::EmptySourceGameId);
        }
        game.validate().map_err(LedgerError::InvalidSource)?;
        if game.config() != GameConfig::research_aaaaa(4).expect("valid research config") {
            return Err(LedgerError::WrongRuleset);
        }
        Ok(Self {
            source_game_id,
            initial_state_hash: *game.canonical_hash().as_bytes(),
            game,
            turns: Vec::new(),
        })
    }

    pub fn game(&self) -> &GameState {
        &self.game
    }

    fn push_turn(
        &mut self,
        action: TurnAction,
        root_decisions: Vec<RootDecisionRecord>,
        evidence_kind: TurnEvidenceKind,
    ) -> Result<(), LedgerError> {
        match evidence_kind {
            TurnEvidenceKind::CanonicalFixture | TurnEvidenceKind::ForcedAction => {
                if !root_decisions.is_empty() {
                    return Err(LedgerError::UnexpectedDecisionTrace(
                        self.game.completed_turns(),
                    ));
                }
            }
            TurnEvidenceKind::PolicyDecisionTrace => {
                if root_decisions.is_empty() {
                    return Err(LedgerError::IncompleteDecisionTrace);
                }
            }
        }
        let turn_index = self.game.completed_turns();
        let actor = self.game.current_player() as u8;
        let pre_state_hash = *self.game.canonical_hash().as_bytes();
        let legacy_action_id_v0 = LegacyActionIdV0::new(&action)?;
        let action_content_id = ActionContentId::canonical(&action);
        self.game.apply(&action)?;
        self.turns.push(TurnLedgerRecord {
            turn_index,
            actor,
            evidence_kind,
            pre_state_hash,
            root_decisions,
            legacy_action_id_v0,
            action_content_id,
            action,
            post_state_hash: *self.game.canonical_hash().as_bytes(),
        });
        Ok(())
    }

    pub fn push_fixture_turn(&mut self, action: TurnAction) -> Result<(), LedgerError> {
        self.push_turn(action, Vec::new(), TurnEvidenceKind::CanonicalFixture)
    }

    pub fn push_forced_turn(&mut self, action: TurnAction) -> Result<(), LedgerError> {
        self.push_turn(action, Vec::new(), TurnEvidenceKind::ForcedAction)
    }

    pub fn push_policy_turn(
        &mut self,
        action: TurnAction,
        root_decisions: Vec<RootDecisionRecord>,
    ) -> Result<(), LedgerError> {
        self.push_turn(
            action,
            root_decisions,
            TurnEvidenceKind::PolicyDecisionTrace,
        )
    }

    pub fn seal_partial(self) -> Result<TrajectoryLedger, LedgerError> {
        self.seal_with_completion(LedgerCompletion::Partial)
    }

    pub fn seal_terminal(self) -> Result<TrajectoryLedger, LedgerError> {
        self.seal_with_completion(LedgerCompletion::Terminal)
    }

    fn seal_with_completion(
        self,
        completion: LedgerCompletion,
    ) -> Result<TrajectoryLedger, LedgerError> {
        if (completion == LedgerCompletion::Terminal) != self.game.is_game_over() {
            return Err(LedgerError::CompletionMismatch);
        }
        let final_state_hash = *self.game.canonical_hash().as_bytes();
        let terminal_scores = self.game.is_game_over().then(|| score_game(&self.game));
        let mut ledger = TrajectoryLedger {
            schema_id: TRAJECTORY_LEDGER_SCHEMA_ID.to_owned(),
            ruleset: ResearchRulesetIdentity::canonical(),
            source_game_id: self.source_game_id,
            config: self.game.config(),
            seed: self.game.seed(),
            initial_state_hash: self.initial_state_hash,
            turns: self.turns,
            completion,
            final_state_hash,
            terminal_scores,
            ledger_sha256: Sha256Digest::of_bytes(b"unsealed"),
        };
        ledger.ledger_sha256 = ledger.recompute_hash()?;
        ledger.verify()?;
        Ok(ledger)
    }
}

static TEMPORARY_NONCE: AtomicU64 = AtomicU64::new(0);

struct TemporaryArtifact(PathBuf);

impl Drop for TemporaryArtifact {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

pub(crate) fn write_immutable_bytes(destination: &Path, bytes: &[u8]) -> Result<(), LedgerError> {
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    if destination.exists() {
        return Err(LedgerError::ArtifactAlreadyExists(
            destination.to_path_buf(),
        ));
    }

    let (temporary, mut file) = loop {
        let temporary = temporary_path(destination);
        match OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
        {
            Ok(file) => break (TemporaryArtifact(temporary), file),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    };
    file.write_all(bytes)?;
    file.flush()?;
    file.sync_all()?;
    drop(file);

    // A same-directory hard link is the portable std-only no-replace publish:
    // it atomically creates `destination` only when that name is absent.
    match fs::hard_link(&temporary.0, destination) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            return Err(LedgerError::ArtifactAlreadyExists(
                destination.to_path_buf(),
            ));
        }
        Err(error) => return Err(error.into()),
    }
    fs::remove_file(&temporary.0)?;
    OpenOptions::new().read(true).open(parent)?.sync_all()?;
    Ok(())
}

fn temporary_path(destination: &Path) -> PathBuf {
    let name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("ledger.json");
    let nonce = TEMPORARY_NONCE.fetch_add(1, Ordering::Relaxed);
    destination.with_file_name(format!(".{name}.partial-{}-{nonce}", std::process::id()))
}

#[derive(Debug, Error)]
pub enum LedgerError {
    #[error("unsupported trajectory ledger schema")]
    WrongSchema,
    #[error("trajectory ledger is not corrected-rules four-player research AAAAA")]
    WrongRuleset,
    #[error("source_game_id must be non-empty")]
    EmptySourceGameId,
    #[error("trajectory source is not a valid canonical game state: {0}")]
    InvalidSource(&'static str),
    #[error("initial canonical state hash does not match")]
    InitialStateHashMismatch,
    #[error("turn index mismatch: expected {expected}, observed {observed}")]
    TurnIndexMismatch { expected: u16, observed: u16 },
    #[error("turn {turn} actor mismatch: expected {expected}, observed {observed}")]
    ActorMismatch {
        turn: u16,
        expected: u8,
        observed: u8,
    },
    #[error("turn {0} pre-state hash mismatch")]
    PreStateHashMismatch(u16),
    #[error("turn {0} legacy action ID mismatch")]
    LegacyActionIdMismatch(u16),
    #[error("turn {0} action-content ID mismatch")]
    ActionContentIdMismatch(u16),
    #[error("turn {0} post-state hash mismatch")]
    PostStateHashMismatch(u16),
    #[error("final canonical state hash mismatch")]
    FinalStateHashMismatch,
    #[error("final canonical scores mismatch")]
    FinalScoreMismatch,
    #[error("trajectory ledger content hash mismatch")]
    LedgerHashMismatch,
    #[error("selected menu index {selected} is outside menu length {menu_len}")]
    InvalidDecisionIndex { selected: u32, menu_len: u32 },
    #[error("selected decision kind is impossible at this root kind")]
    DecisionKindRootMismatch,
    #[error("exactly draft decisions must carry a root-action occurrence ID")]
    DraftOccurrenceMismatch,
    #[error("turn {0} is not a policy trace but contains root decisions")]
    UnexpectedDecisionTrace(u16),
    #[error("policy decision trace must contain one prelude and a terminating draft")]
    IncompleteDecisionTrace,
    #[error("policy decision trace contains a decision after its terminating draft")]
    TrailingDecisionAfterDraft,
    #[error("policy decision ordinal mismatch: expected {expected}, observed {observed}")]
    DecisionOrdinalMismatch { expected: u32, observed: u32 },
    #[error("policy decision ordinal overflowed")]
    DecisionOrdinalOverflow,
    #[error("recorded public observation does not match canonical public state and seat memory")]
    DecisionObservationMismatch,
    #[error("recorded public root or ordered menu does not reconstruct canonically")]
    RootReconstructionMismatch,
    #[error("recorded selected decision does not match its canonical menu occurrence")]
    SelectedDecisionMismatch,
    #[error("staged prelude decisions do not reconstruct the applied compound action")]
    CompoundActionMismatch,
    #[error("ledger completion marker, terminal state, and terminal scores disagree")]
    CompletionMismatch,
    #[error(transparent)]
    Rules(#[from] RuleError),
    #[error(transparent)]
    ActionId(#[from] ActionIdError),
    #[error(transparent)]
    Observation(#[from] crate::ObservationError),
    #[error(transparent)]
    Menu(#[from] crate::MenuError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("immutable scientific artifact already exists: {0}")]
    ArtifactAlreadyExists(PathBuf),
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use cascadia_game::MarketPrelude;

    use super::*;

    fn build(turns: usize) -> TrajectoryLedger {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(44),
        )
        .unwrap();
        let mut builder = TrajectoryLedgerBuilder::new("cpu-fixture-44", game).unwrap();
        for _ in 0..turns {
            let action = builder
                .game()
                .legal_turn_actions(&MarketPrelude::default())
                .unwrap()
                .into_iter()
                .next()
                .unwrap();
            builder.push_fixture_turn(action).unwrap();
        }
        builder.seal_partial().unwrap()
    }

    #[test]
    fn ledger_replay_reconstructs_policy_game_trajectory() {
        let ledger = build(12);
        let replayed = ledger.replay().unwrap();
        assert_eq!(
            replayed.canonical_hash().as_bytes(),
            &ledger.final_state_hash
        );
        assert!(ledger.terminal_scores.is_none());
        assert!(!replayed.is_game_over());
    }

    #[test]
    fn action_or_state_tampering_fails_closed() {
        let ledger = build(3);
        let bytes = serde_json::to_vec(&ledger).unwrap();
        assert_eq!(TrajectoryLedger::from_json_slice(&bytes).unwrap(), ledger);

        let mut value = serde_json::to_value(&ledger).unwrap();
        let original = value["turns"][1]["post_state_hash"][0].as_u64().unwrap();
        value["turns"][1]["post_state_hash"][0] = serde_json::json!((original + 1) % 256);
        let tampered = serde_json::to_vec(&value).unwrap();
        assert!(TrajectoryLedger::from_json_slice(&tampered).is_err());
    }

    #[test]
    fn durable_write_is_hash_verified_and_atomic() {
        let ledger = build(2);
        let directory =
            std::env::temp_dir().join(format!("cascadia-rival-ledger-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        let destination = directory.join("trajectory.json");
        ledger.write_json_immutable(&destination).unwrap();
        let loaded = TrajectoryLedger::from_json_slice(&fs::read(&destination).unwrap()).unwrap();
        assert_eq!(loaded, ledger);
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn immutable_write_never_replaces_an_existing_artifact() {
        let ledger = build(2);
        let directory = std::env::temp_dir().join(format!(
            "cascadia-rival-ledger-no-replace-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();
        let destination = directory.join("trajectory.json");
        let original = b"already-published-scientific-evidence";
        fs::write(&destination, original).unwrap();
        assert!(matches!(
            ledger.write_json_immutable(&destination),
            Err(LedgerError::ArtifactAlreadyExists(path)) if path == destination
        ));
        assert_eq!(fs::read(&destination).unwrap(), original);
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn concurrent_immutable_publish_has_exactly_one_winner() {
        let directory = std::env::temp_dir().join(format!(
            "cascadia-rival-ledger-race-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).unwrap();
        let destination = Arc::new(directory.join("artifact.bin"));
        let barrier = Arc::new(Barrier::new(2));
        let handles: Vec<_> = [b"left".as_slice(), b"right".as_slice()]
            .into_iter()
            .map(|payload| {
                let destination = Arc::clone(&destination);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    write_immutable_bytes(&destination, payload)
                })
            })
            .collect();
        let results: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(
            results
                .iter()
                .filter(|result| matches!(result, Err(LedgerError::ArtifactAlreadyExists(_))))
                .count(),
            1
        );
        let published = fs::read(destination.as_ref()).unwrap();
        assert!(published == b"left" || published == b"right");
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn terminal_and_partial_completion_cannot_be_substituted() {
        let partial = build(3);
        let mut value = serde_json::to_value(&partial).unwrap();
        value["completion"] = serde_json::json!("terminal");
        value["terminal_scores"] =
            serde_json::to_value(score_game(&partial.replay().unwrap())).unwrap();
        assert!(serde_json::from_value::<TrajectoryLedger>(value).is_err());

        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(45),
        )
        .unwrap();
        let mut builder = TrajectoryLedgerBuilder::new("terminal-fixture-45", game).unwrap();
        while !builder.game().is_game_over() {
            let action = builder
                .game()
                .legal_turn_actions(&MarketPrelude::default())
                .unwrap()
                .into_iter()
                .next()
                .unwrap();
            builder.push_fixture_turn(action).unwrap();
        }
        let terminal = builder.seal_terminal().unwrap();
        assert!(terminal.replay().unwrap().is_game_over());
        assert!(terminal.terminal_scores.is_some());
    }

    #[test]
    fn builder_rejects_invalid_deserialized_source_state() {
        let mut value = serde_json::to_value(
            GameState::new(
                GameConfig::research_aaaaa(4).unwrap(),
                GameSeed::from_u64(46),
            )
            .unwrap(),
        )
        .unwrap();
        value["current_player"] = serde_json::json!(9);
        let invalid: GameState = serde_json::from_value(value).unwrap();
        assert!(matches!(
            TrajectoryLedgerBuilder::new("invalid-source", invalid),
            Err(LedgerError::InvalidSource(_))
        ));
    }
}
