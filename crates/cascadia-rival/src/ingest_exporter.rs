//! Ingest bridge: raw incumbent exporter games become sealed trajectory
//! ledgers.
//!
//! The Gate 0 champion battery is produced by
//! `cascadiav3/real-root-exporter` as one `gumbel_game_seed_<seed>.jsonl`
//! file per game: 80 `gumbel_decision` rows plus one terminal
//! `gumbel_game_done` row.  A decision row does not carry the action itself,
//! only the exporter's content address of the chosen root-local action:
//! `action_id = "sha256:" + hex(SHA-256(serde_json::to_vec(&TurnAction)))`
//! (`real-root-exporter/src/main.rs`, `fn action_id`).  The exporter builds
//! its root menu on the post-prelude staged state with
//! `MarketPrelude::default()`, so the hashed action is always root-local:
//! `replace_three_of_a_kind == false` and no wildlife wipes, with the free
//! three-of-a-kind decision recorded separately in
//! `free_three_of_a_kind_choice`.
//!
//! This module replays each game deterministically from its canonical seed:
//! per ply it resolves the recorded free three-of-a-kind choice against the
//! canonical prelude root, recomposes the canonical draft root on the staged
//! state, recomputes the exporter's action hash for every legal draft, and
//! requires the recorded `chosen_action_id` to resolve to EXACTLY one legal
//! draft.  Every turn is appended as a complete
//! [`crate::TurnEvidenceKind::PolicyDecisionTrace`], so the sealed ledger
//! qualifies for [`crate::TomographyEvidenceDomain::IncumbentMeasured`] when
//! its policy identity lives in the [`INCUMBENT_POLICY_NAMESPACE`].
//!
//! ## Fail-closed rules
//!
//! * `ruleset_id` must equal [`LEGACY_RESEARCH_RULESET_ID`] on every row.
//! * The chosen action hash must match exactly one legal draft; zero or
//!   multiple matches abort the whole file, never skip a ply.
//! * Plies must be contiguous from zero in file order; duplicates, gaps, and
//!   reordered rows are refused.
//! * The replayed terminal [`ScoreBreakdown`]s must equal the recorded
//!   `gumbel_game_done` scores field-for-field on every seat.
//! * The `gumbel_game_done` row must be the final row, appear exactly once,
//!   and its `decision_count` must equal the number of decision rows.
//!
//! Nothing here consults the exporter's search outputs (values, visit
//! counts, timings): those fields are schema-validated and ignored, because
//! the ledger's evidence is the canonical replay, not the exporter's claims.

use std::{fs, path::Path};

use cascadia_game::{
    GameConfig, GameSeed, GameState, MarketPrelude, RuleError, ScoreBreakdown, TurnAction,
    score_game,
};
use serde::Deserialize;
use thiserror::Error;

use crate::{
    ActionIdError, INCUMBENT_POLICY_NAMESPACE, LEGACY_RESEARCH_RULESET_ID, LedgerError,
    MenuComposer, MenuError, ObservationError, PrivateSimState, PublicRootId,
    RootActionOccurrenceId, RootDecisionRecord, RulesDecision, RulesLegalMenu, SeatIndex,
    SeatLocalMemory, SelectedDecisionKind, Sha256Digest, TrajectoryLedger,
    TrajectoryLedgerBuilder,
};

/// Raw exporter game file name shape: `gumbel_game_seed_<seed>.jsonl`.
pub const EXPORTER_GAME_FILE_PREFIX: &str = "gumbel_game_seed_";
pub const EXPORTER_GAME_FILE_SUFFIX: &str = ".jsonl";
pub const EXPORTER_DECISION_ROW_TYPE: &str = "gumbel_decision";
pub const EXPORTER_GAME_DONE_ROW_TYPE: &str = "gumbel_game_done";

/// One raw `gumbel_decision` row.  Every exporter field is bound so a schema
/// drift in the producer fails deserialization instead of being silently
/// ignored; fields prefixed with an underscore are schema-validated but not
/// consulted by the replay (they are search claims, not physical evidence).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDecisionRow {
    #[serde(rename = "type")]
    row_type: String,
    ruleset_id: String,
    seed: u64,
    ply: u64,
    active_seat: u8,
    action_count: u64,
    chosen_action_id: String,
    free_three_of_a_kind_choice: FreeThreeOfAKindChoice,
    #[serde(rename = "root_value")]
    _root_value: f64,
    #[serde(rename = "simulations_run")]
    _simulations_run: u64,
    #[serde(rename = "market_branches_searched")]
    _market_branches_searched: u64,
    #[serde(rename = "market_chance_samples")]
    _market_chance_samples: u64,
    #[serde(rename = "total_simulations_run")]
    _total_simulations_run: u64,
    #[serde(rename = "exact_endgame")]
    _exact_endgame: bool,
    #[serde(rename = "decision_seconds")]
    _decision_seconds: f64,
}

/// The terminal `gumbel_game_done` row.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGameDoneRow {
    #[serde(rename = "type")]
    row_type: String,
    ruleset_id: String,
    seed: u64,
    decision_count: u64,
    scores: Vec<RawScoreBreakdown>,
    #[serde(rename = "elapsed_seconds")]
    _elapsed_seconds: f64,
    #[serde(rename = "search")]
    _search: RawSearchConfig,
}

/// Recorded per-seat scores (the exporter's `score_breakdown_json`).  The
/// corrected research rules have no habitat bonuses, so the exporter records
/// no bonus vector; the canonical replay independently produces one and it
/// must be all zeros for the totals to reconcile.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawScoreBreakdown {
    total: u16,
    base_total: u16,
    nature_tokens: u16,
    habitat: [u16; 5],
    wildlife: [u16; 5],
}

/// The exporter's search configuration block: schema-bound, replay-ignored.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSearchConfig {
    #[serde(rename = "n_simulations")]
    _n_simulations: u64,
    #[serde(rename = "top_m")]
    _top_m: u64,
    #[serde(rename = "depth_rounds")]
    _depth_rounds: u64,
    #[serde(rename = "determinization_samples")]
    _determinization_samples: u64,
    #[serde(rename = "market_decision_samples")]
    _market_decision_samples: u64,
    #[serde(rename = "exact_endgame_turns")]
    _exact_endgame_turns: u64,
    #[serde(rename = "rollout_blend_weight")]
    _rollout_blend_weight: f64,
    #[serde(rename = "parallel_leaf_rollouts")]
    _parallel_leaf_rollouts: bool,
    #[serde(rename = "exploration")]
    _exploration: bool,
    #[serde(rename = "k_interior")]
    _k_interior: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FreeThreeOfAKindChoice {
    Accept,
    Decline,
    NotAvailable,
}

#[derive(Debug)]
struct RawExporterGame {
    decisions: Vec<RawDecisionRow>,
    done: RawGameDoneRow,
}

/// One ingested incumbent game: the sealed terminal ledger plus the raw
/// bookkeeping a caller needs for its manifest line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestedExporterGame {
    pub seed: u64,
    pub decision_count: usize,
    pub ledger: TrajectoryLedger,
}

impl IngestedExporterGame {
    /// Per-seat terminal totals in seat order, for one-line summaries.
    pub fn terminal_totals(&self) -> Vec<u16> {
        self.ledger
            .terminal_scores()
            .unwrap_or_default()
            .iter()
            .map(|score| score.total)
            .collect()
    }
}

/// Strict raw-game filename parse: `gumbel_game_seed_<canonical u64>.jsonl`.
/// Anything else (stderr logs, manifests, zero-padded seeds) is not a game
/// file.
pub fn parse_exporter_game_file_name(file_name: &str) -> Option<u64> {
    let digits = file_name
        .strip_prefix(EXPORTER_GAME_FILE_PREFIX)?
        .strip_suffix(EXPORTER_GAME_FILE_SUFFIX)?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let seed: u64 = digits.parse().ok()?;
    if seed.to_string() != digits {
        return None;
    }
    Some(seed)
}

/// Exactly the incumbent exporter's action content address
/// (`real-root-exporter/src/main.rs`, `fn action_id`):
/// `"sha256:" + lowercase hex(SHA-256(serde_json::to_vec(&action)))`.
///
/// The exporter hashes the same `cascadia_game::TurnAction` serde encoding
/// this workspace compiles, so recomputing the digest over the canonical
/// menu's actions reproduces the exporter's ids bit-for-bit; the real Gate 0
/// fixture proves the equivalence end-to-end in tests.
pub fn exporter_action_id(action: &TurnAction) -> Result<Sha256Digest, IngestError> {
    let bytes = serde_json::to_vec(action)?;
    Ok(Sha256Digest::of_bytes(&bytes))
}

/// An ingested policy identity must live in the incumbent namespace with a
/// non-empty, whitespace-free remainder; anything else could silently
/// downgrade the tomography evidence domain.
pub fn validate_incumbent_policy_id(policy_id: &str) -> Result<(), IngestError> {
    let remainder = policy_id
        .strip_prefix(INCUMBENT_POLICY_NAMESPACE)
        .ok_or_else(|| IngestError::PolicyIdOutsideIncumbentNamespace(policy_id.to_owned()))?;
    if remainder.is_empty() || policy_id.chars().any(char::is_whitespace) {
        return Err(IngestError::PolicyIdOutsideIncumbentNamespace(
            policy_id.to_owned(),
        ));
    }
    Ok(())
}

/// Ingests one raw exporter game file (see [`ingest_exporter_game`]); the
/// filename seed must match the recorded rows.
pub fn ingest_exporter_game_file(
    path: &Path,
    policy_id: &str,
) -> Result<IngestedExporterGame, IngestError> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| IngestError::NotAnExporterGameFile(path.display().to_string()))?;
    let seed = parse_exporter_game_file_name(file_name)
        .ok_or_else(|| IngestError::NotAnExporterGameFile(file_name.to_owned()))?;
    let contents = fs::read_to_string(path)?;
    ingest_exporter_game(&contents, Some(seed), policy_id)
}

/// Parses, validates, deterministically replays, and seals one raw exporter
/// game.  The returned ledger is terminal, carries a complete policy
/// decision trace on every turn, and declares
/// `source_game_id = "<policy_id>-<seed>"` so the tomography harness derives
/// the population identity `policy_id` and the `IncumbentMeasured` domain.
pub fn ingest_exporter_game(
    contents: &str,
    expected_seed: Option<u64>,
    policy_id: &str,
) -> Result<IngestedExporterGame, IngestError> {
    validate_incumbent_policy_id(policy_id)?;
    let raw = parse_exporter_game(contents)?;
    validate_raw_game(&raw, expected_seed)?;
    let ledger = replay_and_seal(&raw, policy_id)?;
    Ok(IngestedExporterGame {
        seed: raw.done.seed,
        decision_count: raw.decisions.len(),
        ledger,
    })
}

fn parse_exporter_game(contents: &str) -> Result<RawExporterGame, IngestError> {
    let mut decisions = Vec::new();
    let mut done: Option<RawGameDoneRow> = None;
    for (index, line) in contents.lines().enumerate() {
        let line_number = index + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if done.is_some() {
            return Err(IngestError::RowAfterGameDone { line: line_number });
        }
        let value: serde_json::Value =
            serde_json::from_str(trimmed).map_err(|source| IngestError::MalformedRow {
                line: line_number,
                source,
            })?;
        match value.get("type").and_then(serde_json::Value::as_str) {
            Some(EXPORTER_DECISION_ROW_TYPE) => {
                let row: RawDecisionRow = serde_json::from_value(value).map_err(|source| {
                    IngestError::MalformedRow {
                        line: line_number,
                        source,
                    }
                })?;
                debug_assert_eq!(row.row_type, EXPORTER_DECISION_ROW_TYPE);
                decisions.push(row);
            }
            Some(EXPORTER_GAME_DONE_ROW_TYPE) => {
                let row: RawGameDoneRow = serde_json::from_value(value).map_err(|source| {
                    IngestError::MalformedRow {
                        line: line_number,
                        source,
                    }
                })?;
                debug_assert_eq!(row.row_type, EXPORTER_GAME_DONE_ROW_TYPE);
                done = Some(row);
            }
            observed => {
                return Err(IngestError::UnknownRowType {
                    line: line_number,
                    observed: observed.unwrap_or("<missing>").to_owned(),
                });
            }
        }
    }
    let done = done.ok_or(IngestError::MissingGameDone)?;
    Ok(RawExporterGame { decisions, done })
}

fn validate_raw_game(raw: &RawExporterGame, expected_seed: Option<u64>) -> Result<(), IngestError> {
    if raw.done.ruleset_id != LEGACY_RESEARCH_RULESET_ID {
        return Err(IngestError::WrongRulesetId {
            observed: raw.done.ruleset_id.clone(),
        });
    }
    if let Some(expected) = expected_seed {
        if raw.done.seed != expected {
            return Err(IngestError::SeedMismatch {
                expected,
                observed: raw.done.seed,
            });
        }
    }
    if raw.decisions.is_empty() {
        return Err(IngestError::NoDecisionRows);
    }
    for (index, row) in raw.decisions.iter().enumerate() {
        if row.ruleset_id != LEGACY_RESEARCH_RULESET_ID {
            return Err(IngestError::WrongRulesetId {
                observed: row.ruleset_id.clone(),
            });
        }
        if row.seed != raw.done.seed {
            return Err(IngestError::SeedMismatch {
                expected: raw.done.seed,
                observed: row.seed,
            });
        }
        if row.ply != index as u64 {
            return Err(IngestError::PlyMismatch {
                expected: index as u64,
                observed: row.ply,
            });
        }
        if row.action_count == 0 {
            return Err(IngestError::EmptyRecordedMenu { ply: row.ply });
        }
    }
    if raw.done.decision_count != raw.decisions.len() as u64 {
        return Err(IngestError::DecisionCountMismatch {
            declared: raw.done.decision_count,
            observed: raw.decisions.len() as u64,
        });
    }
    Ok(())
}

fn replay_and_seal(raw: &RawExporterGame, policy_id: &str) -> Result<TrajectoryLedger, IngestError> {
    let config = GameConfig::research_aaaaa(4)?;
    let game = GameState::new(config, GameSeed::from_u64(raw.done.seed))?;
    let source_game_id = format!("{policy_id}-{}", raw.done.seed);
    let mut builder = TrajectoryLedgerBuilder::new(source_game_id, game)?;
    let mut decision_ordinal = 0u32;

    for row in &raw.decisions {
        let state = builder.game().clone();
        if state.is_game_over() {
            return Err(IngestError::DecisionAfterTerminal { ply: row.ply });
        }
        if state.current_player() != usize::from(row.active_seat) {
            return Err(IngestError::SeatMismatch {
                ply: row.ply,
                expected: state.current_player() as u8,
                observed: row.active_seat,
            });
        }
        let actor = SeatIndex::new(row.active_seat)?;

        let prelude_menu = MenuComposer::prelude_root(&state)?;
        let market_has_three = state.market().three_of_a_kind().is_some();
        let mut records = Vec::with_capacity(2);
        let prelude = match row.free_three_of_a_kind_choice {
            FreeThreeOfAKindChoice::Accept => {
                let index = prelude_menu
                    .decisions()
                    .iter()
                    .position(|decision| {
                        matches!(
                            decision,
                            RulesDecision::Prelude(prelude) if prelude.replace_three_of_a_kind
                        )
                    })
                    .ok_or(IngestError::FreeChoiceInconsistent {
                        ply: row.ply,
                        reason: "recorded accept, but the canonical prelude root has no accept",
                    })?;
                records.push(decision_record(
                    &state,
                    actor,
                    &prelude_menu,
                    index,
                    &mut decision_ordinal,
                )?);
                let Some(RulesDecision::Prelude(prelude)) = prelude_menu.decision(index) else {
                    return Err(IngestError::FreeChoiceInconsistent {
                        ply: row.ply,
                        reason: "canonical prelude root decision is not a prelude",
                    });
                };
                prelude.clone()
            }
            FreeThreeOfAKindChoice::Decline | FreeThreeOfAKindChoice::NotAvailable => {
                if (row.free_three_of_a_kind_choice == FreeThreeOfAKindChoice::Decline)
                    != market_has_three
                {
                    return Err(IngestError::FreeChoiceInconsistent {
                        ply: row.ply,
                        reason: "recorded decline/not_available disagrees with the market",
                    });
                }
                match prelude_menu.decision(0) {
                    Some(RulesDecision::Prelude(prelude))
                        if *prelude == MarketPrelude::default() => {}
                    _ => {
                        return Err(IngestError::FreeChoiceInconsistent {
                            ply: row.ply,
                            reason: "canonical prelude root does not lead with the decline",
                        });
                    }
                }
                // A singleton decline is deterministic orchestration and
                // consumes no policy decision; a two-way root records the
                // decline as a real policy choice (index 0).
                if prelude_menu.len() > 1 {
                    records.push(decision_record(
                        &state,
                        actor,
                        &prelude_menu,
                        0,
                        &mut decision_ordinal,
                    )?);
                }
                MarketPrelude::default()
            }
        };

        let staged = state.preview_market_prelude(&prelude)?;
        let draft_menu = MenuComposer::draft_root(&staged, &MarketPrelude::default())?;
        let recorded_id: Sha256Digest =
            row.chosen_action_id
                .parse()
                .map_err(|_| IngestError::InvalidChosenActionId {
                    ply: row.ply,
                    observed: row.chosen_action_id.clone(),
                })?;
        let mut matched_indices = Vec::new();
        for (index, decision) in draft_menu.decisions().iter().enumerate() {
            if let RulesDecision::Draft(action) = decision {
                if exporter_action_id(action)? == recorded_id {
                    matched_indices.push(index);
                }
            }
        }
        let selected_index = match matched_indices.as_slice() {
            [index] => *index,
            [] => {
                return Err(IngestError::ChosenActionNotFound {
                    ply: row.ply,
                    chosen_action_id: row.chosen_action_id.clone(),
                    legal_drafts: draft_menu
                        .decisions()
                        .iter()
                        .filter(|decision| matches!(decision, RulesDecision::Draft(_)))
                        .count(),
                });
            }
            _ => {
                return Err(IngestError::AmbiguousChosenAction {
                    ply: row.ply,
                    chosen_action_id: row.chosen_action_id.clone(),
                    matches: matched_indices.len(),
                });
            }
        };
        records.push(decision_record(
            &staged,
            actor,
            &draft_menu,
            selected_index,
            &mut decision_ordinal,
        )?);

        let Some(RulesDecision::Draft(root_local)) = draft_menu.decision(selected_index) else {
            return Err(IngestError::ChosenActionNotFound {
                ply: row.ply,
                chosen_action_id: row.chosen_action_id.clone(),
                legal_drafts: 0,
            });
        };
        let mut compound = root_local.clone();
        compound.replace_three_of_a_kind = prelude.replace_three_of_a_kind;
        compound.wildlife_wipes = prelude.wildlife_wipes.clone();
        builder.push_policy_turn(compound, records)?;
    }

    if !builder.game().is_game_over() {
        return Err(IngestError::NotTerminalAfterAllDecisions {
            decisions: raw.decisions.len(),
        });
    }

    // Validate the recorded terminal scores against the canonical replay
    // BEFORE paying for the seal's own full re-verification, so a recording
    // mismatch fails closed on the cheapest complete evidence.
    let canonical_scores = score_game(builder.game());
    if canonical_scores.len() != raw.done.scores.len() {
        return Err(IngestError::ScoreSeatCountMismatch {
            recorded: raw.done.scores.len(),
            canonical: canonical_scores.len(),
        });
    }
    for (seat, (canonical, recorded)) in canonical_scores
        .iter()
        .zip(raw.done.scores.iter())
        .enumerate()
    {
        if !recorded_score_matches(recorded, canonical) {
            return Err(IngestError::ScoreMismatch {
                seat: seat as u8,
                recorded: format!("{recorded:?}"),
                canonical: format!("{canonical:?}"),
            });
        }
    }
    Ok(builder.seal_terminal()?)
}

fn recorded_score_matches(recorded: &RawScoreBreakdown, canonical: &ScoreBreakdown) -> bool {
    recorded.total == canonical.total
        && recorded.base_total == canonical.base_total
        && recorded.nature_tokens == canonical.nature_tokens
        && recorded.habitat == canonical.habitat
        && recorded.wildlife == canonical.wildlife
        && canonical.habitat_bonus == [0; 5]
}

/// One verified public-root decision record.  The incumbent policy is
/// stateless across turns, so the seat-local memory is the empty canonical
/// memory before and after every decision, exactly as ledger replay's fresh
/// [`crate::PolicyMemoryBank`] expects.
fn decision_record(
    state: &GameState,
    actor: SeatIndex,
    menu: &RulesLegalMenu,
    index: usize,
    decision_ordinal: &mut u32,
) -> Result<RootDecisionRecord, IngestError> {
    let memory = SeatLocalMemory::empty();
    let observation =
        PrivateSimState::new(state.clone())?.public_observation(actor, memory.clone())?;
    let root = PublicRootId::new(&observation, menu.root_kind());
    let (selected_kind, draft_occurrence_id) = match menu
        .decision(index)
        .ok_or(IngestError::MenuIndexOutOfRange(index))?
    {
        RulesDecision::Prelude(_) => (SelectedDecisionKind::Prelude, None),
        RulesDecision::PaidWipe(_) => (SelectedDecisionKind::PaidWipe, None),
        RulesDecision::Draft(_) => (
            SelectedDecisionKind::Draft,
            Some(RootActionOccurrenceId::new(&root, menu, index)?),
        ),
    };
    let record = RootDecisionRecord {
        decision_ordinal: *decision_ordinal,
        root_kind: menu.root_kind(),
        public_observation: observation,
        public_root_id: root,
        ordered_menu_hash: menu.hash(),
        menu_len: menu.len() as u32,
        selected_index: index as u32,
        selected_kind,
        draft_occurrence_id,
        next_memory: memory,
    };
    *decision_ordinal = decision_ordinal
        .checked_add(1)
        .ok_or(IngestError::DecisionOrdinalOverflow)?;
    Ok(record)
}

#[derive(Debug, Error)]
pub enum IngestError {
    #[error("not a raw exporter game file: {0}")]
    NotAnExporterGameFile(String),
    #[error("policy id {0:?} is not inside the 'incumbent:' namespace")]
    PolicyIdOutsideIncumbentNamespace(String),
    #[error("line {line} is not valid exporter row JSON: {source}")]
    MalformedRow {
        line: usize,
        source: serde_json::Error,
    },
    #[error("line {line} has unknown exporter row type {observed:?}")]
    UnknownRowType { line: usize, observed: String },
    #[error("line {line} follows the terminal gumbel_game_done row")]
    RowAfterGameDone { line: usize },
    #[error("raw game has no terminal gumbel_game_done row")]
    MissingGameDone,
    #[error("raw game has no gumbel_decision rows")]
    NoDecisionRows,
    #[error("ruleset_id {observed:?} is not the corrected research ruleset")]
    WrongRulesetId { observed: String },
    #[error("seed mismatch: expected {expected}, observed {observed}")]
    SeedMismatch { expected: u64, observed: u64 },
    #[error("ply mismatch: expected {expected}, observed {observed} (duplicate, missing, or reordered ply)")]
    PlyMismatch { expected: u64, observed: u64 },
    #[error("decision row ply {ply} records an empty search menu")]
    EmptyRecordedMenu { ply: u64 },
    #[error("gumbel_game_done declares {declared} decisions but the file has {observed}")]
    DecisionCountMismatch { declared: u64, observed: u64 },
    #[error("ply {ply} follows a terminal state")]
    DecisionAfterTerminal { ply: u64 },
    #[error("ply {ply} seat mismatch: canonical replay has seat {expected}, row says {observed}")]
    SeatMismatch { ply: u64, expected: u8, observed: u8 },
    #[error("ply {ply} free three-of-a-kind choice is inconsistent: {reason}")]
    FreeChoiceInconsistent { ply: u64, reason: &'static str },
    #[error("ply {ply} chosen_action_id {observed:?} is not a canonical sha256 digest")]
    InvalidChosenActionId { ply: u64, observed: String },
    #[error(
        "ply {ply}: chosen action {chosen_action_id} matches none of the {legal_drafts} legal drafts"
    )]
    ChosenActionNotFound {
        ply: u64,
        chosen_action_id: String,
        legal_drafts: usize,
    },
    #[error("ply {ply}: chosen action {chosen_action_id} matches {matches} legal drafts")]
    AmbiguousChosenAction {
        ply: u64,
        chosen_action_id: String,
        matches: usize,
    },
    #[error("game is not terminal after all {decisions} recorded decisions")]
    NotTerminalAfterAllDecisions { decisions: usize },
    #[error("recorded scores cover {recorded} seats but the canonical replay has {canonical}")]
    ScoreSeatCountMismatch { recorded: usize, canonical: usize },
    #[error(
        "seat {seat} terminal score mismatch: recorded {recorded}, canonical replay {canonical}"
    )]
    ScoreMismatch {
        seat: u8,
        recorded: String,
        canonical: String,
    },
    #[error("selected menu index {0} is outside the canonical menu")]
    MenuIndexOutOfRange(usize),
    #[error("policy decision ordinal overflowed")]
    DecisionOrdinalOverflow,
    #[error(transparent)]
    Rules(#[from] RuleError),
    #[error(transparent)]
    Menu(#[from] MenuError),
    #[error(transparent)]
    Observation(#[from] ObservationError),
    #[error(transparent)]
    ActionId(#[from] ActionIdError),
    #[error(transparent)]
    Ledger(#[from] LedgerError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn game_file_names_parse_strictly() {
        assert_eq!(
            parse_exporter_game_file_name("gumbel_game_seed_2027160000.jsonl"),
            Some(2027160000)
        );
        assert_eq!(parse_exporter_game_file_name("gumbel_game_seed_0.jsonl"), Some(0));
        for rejected in [
            "gumbel_game_seed_2027160000.jsonl.stderr",
            "gumbel_game_seed_.jsonl",
            "gumbel_game_seed_007.jsonl",
            "gumbel_game_seed_1x.jsonl",
            "gumbel_game_seed_99999999999999999999999999.jsonl",
            "manifest.json",
            "gumbel_game_seed_1.json",
        ] {
            assert_eq!(parse_exporter_game_file_name(rejected), None, "{rejected}");
        }
    }

    #[test]
    fn policy_ids_must_live_in_the_incumbent_namespace() {
        assert!(validate_incumbent_policy_id("incumbent:gate0-b0").is_ok());
        for rejected in ["incumbent:", "cpu-proxy:gate0", "incumbent: padded", ""] {
            assert!(validate_incumbent_policy_id(rejected).is_err(), "{rejected}");
        }
    }

    #[test]
    fn exporter_action_id_is_the_sha256_of_the_serde_json_action_bytes() {
        let action = TurnAction::paired(
            cascadia_game::MarketSlot::ZERO,
            cascadia_game::HexCoord { q: 0, r: 0 },
            cascadia_game::Rotation::new(0).unwrap(),
        );
        let expected = Sha256Digest::of_bytes(&serde_json::to_vec(&action).unwrap());
        assert_eq!(exporter_action_id(&action).unwrap(), expected);
        assert!(exporter_action_id(&action).unwrap().as_str().starts_with("sha256:"));
    }

    #[test]
    fn a_game_done_row_must_terminate_the_file_exactly_once() {
        assert!(matches!(
            parse_exporter_game(""),
            Err(IngestError::MissingGameDone)
        ));
        let done = r#"{"type":"gumbel_game_done","ruleset_id":"x","seed":1,"decision_count":0,"scores":[],"elapsed_seconds":1.0,"search":{"n_simulations":1,"top_m":1,"depth_rounds":1,"determinization_samples":1,"market_decision_samples":1,"exact_endgame_turns":0,"rollout_blend_weight":0.5,"parallel_leaf_rollouts":false,"exploration":false,"k_interior":1}}"#;
        let doubled = format!("{done}\n{done}\n");
        assert!(matches!(
            parse_exporter_game(&doubled),
            Err(IngestError::RowAfterGameDone { line: 2 })
        ));
        let unknown = r#"{"type":"gumbel_mystery"}"#;
        assert!(matches!(
            parse_exporter_game(unknown),
            Err(IngestError::UnknownRowType { line: 1, .. })
        ));
    }

    #[test]
    fn schema_drift_in_a_row_fails_closed() {
        let extra_field = r#"{"type":"gumbel_game_done","ruleset_id":"x","seed":1,"decision_count":0,"scores":[],"elapsed_seconds":1.0,"surprise":true,"search":{"n_simulations":1,"top_m":1,"depth_rounds":1,"determinization_samples":1,"market_decision_samples":1,"exact_endgame_turns":0,"rollout_blend_weight":0.5,"parallel_leaf_rollouts":false,"exploration":false,"k_interior":1}}"#;
        assert!(matches!(
            parse_exporter_game(extra_field),
            Err(IngestError::MalformedRow { line: 1, .. })
        ));
    }
}
