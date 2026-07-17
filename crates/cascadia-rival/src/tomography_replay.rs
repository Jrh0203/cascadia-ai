//! Chronology-preserving hindsight replay tomography (build-scope work item
//! 2b, the optimizer the scope numbers "T1").
//!
//! Given one seat's complete realized chronology from a sealed
//! [`TrajectoryLedger`] — the exact draft pairs in order, refresh
//! decisions, paid wipes, and nature transactions — this optimizer holds the
//! chronology fixed and re-optimizes only the seat's legal *placement*
//! decisions (tile coordinate/rotation, wildlife placement or decline)
//! through real [`GameState`] transitions in original order.
//!
//! ## The interleaving/infeasibility rule
//!
//! A single seat's chronology in a four-player game interleaves with the
//! other seats' turns.  The other seats' recorded actions are replayed
//! VERBATIM from the ledger; only the target seat's placements are
//! re-chosen.  A re-chosen placement can still perturb the shared physical
//! stream: declining a wildlife token that was realized as placed (or vice
//! versa) changes the bag, and moving wildlife off keystones changes the
//! seat's own later token legality.  Chronology preservation therefore means
//! the recorded public resource stream must remain *exactly realizable*: a
//! branch is INFEASIBLE — pruned, never patched — the moment
//!
//! 1. any later recorded action (any seat) fails to apply legally, or
//! 2. after any later turn, the public market or any non-target seat's board
//!    differs from the realized reference at that turn.
//!
//! The realized trajectory itself always survives every check, so the
//! search's witness is never below the realized score.
//!
//! ## Labeling
//!
//! The build scope numbers this optimizer "T1" positionally, but the
//! serialized taxonomy assigns evidence kinds by information boundary.  A
//! hindsight replay re-chooses placements with the entire realized future —
//! the known exogenous chance tape and the opponents' realized actions — so
//! its results are emitted as [`TomographyKind::T3KnownWorldOneSeatOracle`]
//! with [`crate::InformationBoundary::KnownExogenousChanceTape`].  Labeling
//! it `T1PublicOneSeatWitness` (public-information-only) would overclaim:
//! this witness is future-knowledge and is not a policy.  Evidence stays
//! [`TomographyEvidence::BestFound`]: a beam search never certifies
//! optimality, and the witness is a lower bound only.

use cascadia_game::{
    Board, DraftChoice, GameState, MarketPrelude, RuleError, ScoreBreakdown, ScoringCards,
    TurnAction, rescore_after_tile_with_habitat_analysis, rescore_after_wildlife_placement,
    score_board, score_game,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    LedgerError, SeatIndex, Sha256Digest, TomographyError, TomographyEvidence, TomographyKind,
    TomographyPopulation, TomographyResult, TomographyResultInput, TrajectoryLedger,
};

pub const REPLAY_SOLVER_ID: &str = "cascadiav3.rival_tomography_replay_solver.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayConfig {
    /// Recorded for provenance; the beam itself is fully deterministic.
    pub seed: u64,
    /// Surviving branches per target-seat decision point.
    pub beam_width: usize,
    /// Candidates fully transitioned per branch per decision point (chosen
    /// by exact own-board score, deterministic tie-break by enumeration
    /// order).  The recorded action is always additionally expanded on the
    /// realized branch.
    pub candidate_cap: usize,
}

/// A certified hindsight-placement witness: the target seat's re-chosen
/// compound turn actions in chronological order.  [`ReplayWitness::certify`]
/// replays them against the sealed ledger through the canonical engine and
/// fails closed on any legality, chronology, or claimed-score mismatch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayWitness {
    pub solver_id: String,
    pub source_game_id: String,
    pub ledger_sha256: Sha256Digest,
    pub seat: u8,
    pub seed: u64,
    pub beam_width: usize,
    pub candidate_cap: usize,
    pub explored_nodes: u64,
    pub realized_score: ScoreBreakdown,
    pub witness_score: ScoreBreakdown,
    pub score_delta: i32,
    pub chosen_actions: Vec<TurnAction>,
}

impl ReplayWitness {
    /// Replays the witness from the sealed initial state: other seats'
    /// recorded actions verbatim, the target seat's chosen actions in order
    /// (chronology — prelude and draft — must match the record), with the
    /// full public-stream feasibility check at every turn.
    pub fn certify(&self, ledger: &TrajectoryLedger) -> Result<(), ReplayError> {
        if self.ledger_sha256 != *ledger.ledger_sha256() {
            return Err(ReplayError::WitnessLedgerMismatch);
        }
        let seat = usize::from(self.seat);
        let reference = reference_states(ledger)?;
        let mut state = ledger.initial_state()?;
        let mut chosen = self.chosen_actions.iter();
        for (index, record) in ledger.turns().iter().enumerate() {
            let action = if usize::from(record.actor) == seat {
                let action = chosen.next().ok_or(ReplayError::ChosenActionCount)?;
                if action.prelude() != record.action.prelude()
                    || action.draft != record.action.draft
                {
                    return Err(ReplayError::DeviationChangesChronology);
                }
                action
            } else {
                &record.action
            };
            state = state
                .transition(action)
                .map_err(|_| ReplayError::ChronologyViolation(index as u16))?;
            if !public_stream_matches(&state, &reference[index], seat) {
                return Err(ReplayError::ChronologyViolation(index as u16));
            }
        }
        if chosen.next().is_some() {
            return Err(ReplayError::ChosenActionCount);
        }
        if !state.is_game_over() {
            return Err(ReplayError::NotTerminal);
        }
        let scores = score_game(&state);
        if scores.get(seat).copied() != Some(self.witness_score) {
            return Err(ReplayError::WitnessScoreMismatch);
        }
        if self.score_delta
            != i32::from(self.witness_score.total) - i32::from(self.realized_score.total)
        {
            return Err(ReplayError::DeltaArithmetic);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayOutcome {
    pub witness: ReplayWitness,
    pub result: TomographyResult,
}

#[derive(Clone)]
struct BeamEntry {
    state: GameState,
    chosen: Vec<TurnAction>,
    is_realized: bool,
    heuristic: u16,
}

pub fn replay_seat(
    ledger: &TrajectoryLedger,
    seat: SeatIndex,
    config: &ReplayConfig,
    population: &TomographyPopulation,
) -> Result<ReplayOutcome, ReplayError> {
    population.validate()?;
    if config.beam_width == 0 {
        return Err(ReplayError::ZeroBeamWidth);
    }
    if config.candidate_cap == 0 {
        return Err(ReplayError::ZeroCandidateCap);
    }
    if ledger.completion() != crate::LedgerCompletion::Terminal {
        return Err(ReplayError::NotTerminal);
    }
    let seat_index = usize::from(seat.get());
    let cards = ledger.config().scoring_cards;
    let reference = reference_states(ledger)?;
    let realized_score = *ledger
        .terminal_scores()
        .ok_or(ReplayError::NotTerminal)?
        .get(seat_index)
        .ok_or(ReplayError::SeatOutOfRange(seat.get()))?;

    let initial = ledger.initial_state()?;
    if initial.boards().len() <= seat_index {
        return Err(ReplayError::SeatOutOfRange(seat.get()));
    }
    let mut explored_nodes = 0u64;
    let mut beam = vec![BeamEntry {
        heuristic: score_board(&initial.boards()[seat_index], cards).total,
        state: initial,
        chosen: Vec::new(),
        is_realized: true,
    }];

    for (turn_index, record) in ledger.turns().iter().enumerate() {
        let reference_after = &reference[turn_index];
        if usize::from(record.actor) != seat_index {
            let mut survivors = Vec::with_capacity(beam.len());
            for mut entry in beam {
                match entry.state.transition(&record.action) {
                    Ok(next) if public_stream_matches(&next, reference_after, seat_index) => {
                        entry.state = next;
                        survivors.push(entry);
                    }
                    _ if entry.is_realized => {
                        return Err(ReplayError::RealizedPathDiverged(record.turn_index));
                    }
                    _ => {}
                }
            }
            beam = survivors;
        } else {
            let prelude = record.action.prelude();
            let draft = record.action.draft;
            let mut children: Vec<BeamEntry> = Vec::new();
            for entry in &beam {
                let evaluated =
                    match enumerate_own_turn_candidates(&entry.state, &prelude, draft, cards) {
                        Ok(evaluated) if !evaluated.is_empty() => evaluated,
                        Ok(_) | Err(_) if !entry.is_realized => continue,
                        Ok(_) | Err(_) => {
                            return Err(ReplayError::RealizedPathDiverged(record.turn_index));
                        }
                    };
                // Deterministic exact-heuristic candidate selection.
                let mut ranked: Vec<usize> = (0..evaluated.len()).collect();
                ranked.sort_by_key(|&index| std::cmp::Reverse(evaluated[index].1));
                ranked.truncate(config.candidate_cap);
                if entry.is_realized {
                    let recorded = evaluated
                        .iter()
                        .position(|(action, _)| action == &record.action)
                        .ok_or(ReplayError::RealizedPathDiverged(record.turn_index))?;
                    if !ranked.contains(&recorded) {
                        ranked.push(recorded);
                    }
                }
                for candidate_index in ranked {
                    let (action, heuristic) = &evaluated[candidate_index];
                    explored_nodes += 1;
                    let Ok(next) = entry.state.transition(action) else {
                        if entry.is_realized && action == &record.action {
                            return Err(ReplayError::RealizedPathDiverged(record.turn_index));
                        }
                        continue;
                    };
                    let is_realized_child = entry.is_realized && action == &record.action;
                    if !public_stream_matches(&next, reference_after, seat_index) {
                        if is_realized_child {
                            return Err(ReplayError::RealizedPathDiverged(record.turn_index));
                        }
                        continue;
                    }
                    let mut chosen = entry.chosen.clone();
                    chosen.push(action.clone());
                    children.push(BeamEntry {
                        state: next,
                        chosen,
                        is_realized: is_realized_child,
                        heuristic: *heuristic,
                    });
                }
            }
            // Stable rank keeps ties in deterministic generation order.
            children.sort_by_key(|child| std::cmp::Reverse(child.heuristic));
            let realized_position = children.iter().position(|child| child.is_realized);
            let Some(realized_position) = realized_position else {
                return Err(ReplayError::RealizedPathDiverged(record.turn_index));
            };
            if realized_position >= config.beam_width {
                let realized_child = children.remove(realized_position);
                children.truncate(config.beam_width.saturating_sub(1));
                children.push(realized_child);
            } else {
                children.truncate(config.beam_width);
            }
            beam = children;
        }
    }

    let mut best: Option<(&BeamEntry, ScoreBreakdown)> = None;
    for entry in &beam {
        if !entry.state.is_game_over() {
            return Err(ReplayError::NotTerminal);
        }
        let score = score_game(&entry.state)[seat_index];
        if best
            .as_ref()
            .is_none_or(|(_, best_score)| score.total > best_score.total)
        {
            best = Some((entry, score));
        }
    }
    let (best_entry, witness_score) = best.ok_or(ReplayError::EmptyBeam)?;
    if witness_score.total < realized_score.total {
        return Err(ReplayError::WitnessBelowRealized);
    }

    let witness = ReplayWitness {
        solver_id: REPLAY_SOLVER_ID.to_owned(),
        source_game_id: ledger.source_game_id().to_owned(),
        ledger_sha256: ledger.ledger_sha256().clone(),
        seat: seat.get(),
        seed: config.seed,
        beam_width: config.beam_width,
        candidate_cap: config.candidate_cap,
        explored_nodes,
        realized_score,
        witness_score,
        score_delta: i32::from(witness_score.total) - i32::from(realized_score.total),
        chosen_actions: best_entry.chosen.clone(),
    };
    // Fail-closed self-certification through a fresh canonical replay.
    witness.certify(ledger)?;

    let solver_config_sha256 = Sha256Digest::of_bytes(&serde_json::to_vec(&serde_json::json!({
        "solver_id": REPLAY_SOLVER_ID,
        "seed": config.seed,
        "beam_width": config.beam_width,
        "candidate_cap": config.candidate_cap,
    }))?);
    let witness_ledger_sha256 = Sha256Digest::of_bytes(&serde_json::to_vec(&witness)?);
    let result = TomographyResult::try_new_in_domain(
        TomographyResultInput {
            kind: TomographyKind::T3KnownWorldOneSeatOracle,
            root_id: Sha256Digest::of_bytes(
                format!(
                    "{}:{}:t3-chronology-preserving-replay",
                    ledger.ledger_sha256(),
                    seat.get()
                )
                .as_bytes(),
            ),
            source_game_id: ledger.source_game_id().to_owned(),
            acting_seat: seat.get(),
            incumbent_policy_id: population.incumbent_policy_id.clone(),
            opponent_population_id: population.opponent_population_id.clone(),
            evidence: TomographyEvidence::BestFound {
                score_delta: witness.score_delta,
                solver_config_sha256,
                witness_ledger_sha256,
                explored_nodes,
            },
            natural_frequency_weight_numerator: 1,
            natural_frequency_weight_denominator: 1,
        },
        population.evidence_domain,
    )?;
    Ok(ReplayOutcome { witness, result })
}

/// Feasibility of a single held-chronology placement deviation.
///
/// Replays the ledger verbatim up to `turn_index` (which must be a turn of
/// `seat`), applies `alternative` there — its prelude and draft must equal
/// the recorded chronology, only the placement may differ — and then
/// continues every remaining recorded action verbatim.  Returns `Ok(false)`
/// the moment the branch becomes infeasible under the public-stream rule
/// documented at module level; `Ok(true)` if the full recorded chronology
/// remains exactly realizable.
pub fn chronology_deviation_is_feasible(
    ledger: &TrajectoryLedger,
    seat: SeatIndex,
    turn_index: usize,
    alternative: &TurnAction,
) -> Result<bool, ReplayError> {
    if ledger.completion() != crate::LedgerCompletion::Terminal {
        return Err(ReplayError::NotTerminal);
    }
    let seat_index = usize::from(seat.get());
    let turns = ledger.turns();
    let record = turns
        .get(turn_index)
        .ok_or(ReplayError::TurnIndexOutOfRange(turn_index))?;
    if usize::from(record.actor) != seat_index {
        return Err(ReplayError::DeviationNotOwnTurn(turn_index));
    }
    if alternative.prelude() != record.action.prelude() || alternative.draft != record.action.draft
    {
        return Err(ReplayError::DeviationChangesChronology);
    }
    let reference = reference_states(ledger)?;
    let mut state = ledger.initial_state()?;
    for (index, turn) in turns.iter().enumerate() {
        let action = if index == turn_index {
            alternative
        } else {
            &turn.action
        };
        state = match state.transition(action) {
            Ok(next) => next,
            Err(_) if index >= turn_index => return Ok(false),
            Err(_) => return Err(ReplayError::RealizedPathDiverged(turn.turn_index)),
        };
        if !public_stream_matches(&state, &reference[index], seat_index) {
            if index >= turn_index {
                return Ok(false);
            }
            return Err(ReplayError::RealizedPathDiverged(turn.turn_index));
        }
    }
    Ok(true)
}

/// Enumerates every legal placement completion of the recorded prelude and
/// draft, paired with the exact after-placement own-board score total.
///
/// The score is computed through the canonical incremental rescoring path
/// ([`rescore_after_tile_with_habitat_analysis`] plus
/// [`rescore_after_wildlife_placement`]), which is bit-identical to a full
/// [`score_board`] of the candidate board (asserted by unit test) but avoids
/// recomputing the five habitat components per candidate.  Enumeration order
/// is the engine's deterministic visitor order.
fn enumerate_own_turn_candidates(
    state: &GameState,
    prelude: &MarketPrelude,
    draft: DraftChoice,
    cards: ScoringCards,
) -> Result<Vec<(TurnAction, u16)>, RuleError> {
    let staged = state.preview_market_prelude(prelude)?;
    let own_board = &staged.boards()[staged.current_player()];
    let analysis = own_board.habitat_analysis();
    let baseline = cascadia_game::score_board_with_habitat_analysis(own_board, cards, &analysis);
    let mut evaluated: Vec<(TurnAction, u16)> = Vec::new();
    staged.visit_legal_turn_actions_with_tile_context(
        &MarketPrelude::default(),
        &mut evaluated,
        |candidate_draft| candidate_draft == draft,
        |evaluated, capacity| evaluated.reserve_exact(capacity),
        &mut |board: &Board, placement, tile| {
            rescore_after_tile_with_habitat_analysis(
                board, cards, baseline, &analysis, placement, tile,
            )
        },
        &mut |evaluated: &mut Vec<(TurnAction, u16)>,
              board: &Board,
              candidate_draft,
              placement,
              after_tile: &ScoreBreakdown,
              placed_wildlife| {
            let score = match placed_wildlife {
                None => *after_tile,
                Some((wildlife, _)) => {
                    rescore_after_wildlife_placement(board, cards, *after_tile, wildlife)
                }
            };
            evaluated.push((
                TurnAction {
                    replace_three_of_a_kind: prelude.replace_three_of_a_kind,
                    wildlife_wipes: prelude.wildlife_wipes.clone(),
                    draft: candidate_draft,
                    tile: placement,
                    wildlife: placed_wildlife.map(|(_, coord)| coord),
                },
                score.total,
            ));
        },
    )?;
    Ok(evaluated)
}

/// Verifies and materializes the realized state after every recorded turn
/// (hash-anchored against the sealed per-turn commitments).
fn reference_states(ledger: &TrajectoryLedger) -> Result<Vec<GameState>, ReplayError> {
    Ok(ledger.raw_state_trajectory()?)
}

/// The public-stream feasibility check: the market and every non-target
/// seat's board must equal the realized reference after the same turn.
fn public_stream_matches(state: &GameState, reference: &GameState, seat: usize) -> bool {
    if state.market() != reference.market() {
        return false;
    }
    state
        .boards()
        .iter()
        .zip(reference.boards())
        .enumerate()
        .all(|(index, (left, right))| index == seat || left == right)
}

#[derive(Debug, Error)]
pub enum ReplayError {
    #[error("hindsight replay requires a sealed terminal trajectory ledger")]
    NotTerminal,
    #[error("seat {0} is outside the four-player research table")]
    SeatOutOfRange(u8),
    #[error("beam width must be at least one")]
    ZeroBeamWidth,
    #[error("candidate cap must be at least one")]
    ZeroCandidateCap,
    #[error("witness does not belong to the supplied ledger")]
    WitnessLedgerMismatch,
    #[error("the sealed realized trajectory failed its own feasibility checks at turn {0}")]
    RealizedPathDiverged(u16),
    #[error("branch breaks the recorded public resource stream at turn {0}; pruned as infeasible")]
    ChronologyViolation(u16),
    #[error("deviation must preserve the recorded prelude and draft chronology")]
    DeviationChangesChronology,
    #[error("turn {0} does not belong to the target seat")]
    DeviationNotOwnTurn(usize),
    #[error("turn index {0} is outside the recorded trajectory")]
    TurnIndexOutOfRange(usize),
    #[error("witness action count does not match the seat's recorded turns")]
    ChosenActionCount,
    #[error("witness does not reproduce its exact claimed score")]
    WitnessScoreMismatch,
    #[error("witness delta arithmetic is inconsistent")]
    DeltaArithmetic,
    #[error("search returned a witness below the realized score")]
    WitnessBelowRealized,
    #[error("beam search lost every branch including the realized trajectory")]
    EmptyBeam,
    #[error(transparent)]
    Ledger(#[from] LedgerError),
    #[error(transparent)]
    Rules(#[from] RuleError),
    #[error(transparent)]
    Tomography(#[from] TomographyError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameConfig, GameSeed};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use crate::{TomographyEvidenceDomain, TrajectoryLedgerBuilder};

    use super::*;

    fn terminal_fixture(seed: u64) -> TrajectoryLedger {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(seed),
        )
        .unwrap();
        let mut builder =
            TrajectoryLedgerBuilder::new(format!("replay-unit-fixture-{seed}"), game).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x7265_706c);
        while !builder.game().is_game_over() {
            let preludes = builder.game().free_three_of_a_kind_choices().unwrap();
            let prelude = &preludes[rng.gen_range(0..preludes.len())];
            let actions = builder.game().legal_turn_actions(prelude).unwrap();
            let action = actions[rng.gen_range(0..actions.len())].clone();
            builder.push_fixture_turn(action).unwrap();
        }
        builder.seal_terminal().unwrap()
    }

    fn proxy_population() -> TomographyPopulation {
        TomographyPopulation {
            incumbent_policy_id: "replay-unit-fixture".to_owned(),
            opponent_population_id: "replay-unit-fixture:table".to_owned(),
            evidence_domain: TomographyEvidenceDomain::CpuProxy,
        }
    }

    #[test]
    fn incremental_candidate_heuristic_is_bit_identical_to_full_rescoring() {
        let ledger = terminal_fixture(6304);
        let cards = ledger.config().scoring_cards;
        let mut state = ledger.initial_state().unwrap();
        for record in ledger.turns().iter().take(24) {
            let prelude = record.action.prelude();
            let draft = record.action.draft;
            let fast = enumerate_own_turn_candidates(&state, &prelude, draft, cards).unwrap();
            let full = state
                .evaluate_legal_draft_actions(&prelude, draft, |board| {
                    score_board(board, cards).total
                })
                .unwrap();
            assert!(!fast.is_empty());
            assert_eq!(fast, full, "turn {}", record.turn_index);
            state = state.transition(&record.action).unwrap();
        }
    }

    #[test]
    fn recorded_trajectory_is_always_a_feasible_deviation_of_itself() {
        let ledger = terminal_fixture(6301);
        let seat = SeatIndex::new(0).unwrap();
        for (index, record) in ledger.turns().iter().enumerate() {
            if usize::from(record.actor) != 0 {
                continue;
            }
            assert!(
                chronology_deviation_is_feasible(&ledger, seat, index, &record.action).unwrap(),
                "the realized action at turn {index} must be feasible"
            );
        }
    }

    #[test]
    fn wildlife_decline_flip_that_perturbs_the_bag_is_pruned_as_infeasible() {
        // Flipping place<->decline changes the wildlife bag.  Under the
        // public-stream rule at least one such flip in a full fixture game
        // must surface as a later market divergence and be rejected; none
        // may be patched.  The fixture is fully deterministic.
        let ledger = terminal_fixture(6302);
        let seat = SeatIndex::new(0).unwrap();
        let mut flips = 0usize;
        let mut pruned = 0usize;
        for (index, record) in ledger.turns().iter().enumerate() {
            if usize::from(record.actor) != 0 || record.action.wildlife.is_none() {
                continue;
            }
            let mut flipped = record.action.clone();
            flipped.wildlife = None;
            flips += 1;
            if !chronology_deviation_is_feasible(&ledger, seat, index, &flipped).unwrap() {
                pruned += 1;
            }
        }
        assert!(flips > 0, "fixture must contain wildlife placements");
        assert!(
            pruned > 0,
            "at least one decline flip must break the recorded stream and be pruned"
        );
    }

    #[test]
    fn deviation_that_changes_the_chronology_is_an_error_not_a_branch() {
        let ledger = terminal_fixture(6301);
        let seat_zero_turn = ledger
            .turns()
            .iter()
            .position(|record| record.actor == 0)
            .unwrap();
        let mut alternative = ledger.turns()[seat_zero_turn].action.clone();
        alternative.replace_three_of_a_kind = !alternative.replace_three_of_a_kind;
        assert!(matches!(
            chronology_deviation_is_feasible(
                &ledger,
                SeatIndex::new(0).unwrap(),
                seat_zero_turn,
                &alternative,
            ),
            Err(ReplayError::DeviationChangesChronology)
        ));
        assert!(matches!(
            chronology_deviation_is_feasible(
                &ledger,
                SeatIndex::new(1).unwrap(),
                seat_zero_turn,
                &ledger.turns()[seat_zero_turn].action,
            ),
            Err(ReplayError::DeviationNotOwnTurn(_))
        ));
    }

    #[test]
    fn replay_is_deterministic_improving_and_certified() {
        let ledger = terminal_fixture(6303);
        let config = ReplayConfig {
            seed: 11,
            beam_width: 2,
            candidate_cap: 4,
        };
        let seat = SeatIndex::new(3).unwrap();
        let left = replay_seat(&ledger, seat, &config, &proxy_population()).unwrap();
        let right = replay_seat(&ledger, seat, &config, &proxy_population()).unwrap();
        assert_eq!(left, right);
        assert!(left.witness.witness_score.total >= left.witness.realized_score.total);
        assert_eq!(left.witness.chosen_actions.len(), 20);
        assert_eq!(
            left.result.kind(),
            TomographyKind::T3KnownWorldOneSeatOracle
        );
        assert_eq!(left.result.evidence().upper_bound(), None);
        left.witness.certify(&ledger).unwrap();
    }

    #[test]
    fn tampered_witness_fails_certification() {
        let ledger = terminal_fixture(6303);
        let config = ReplayConfig {
            seed: 11,
            beam_width: 1,
            candidate_cap: 2,
        };
        let outcome = replay_seat(
            &ledger,
            SeatIndex::new(2).unwrap(),
            &config,
            &proxy_population(),
        )
        .unwrap();
        let mut tampered = outcome.witness.clone();
        tampered.witness_score.total += 1;
        assert!(matches!(
            tampered.certify(&ledger),
            Err(ReplayError::WitnessScoreMismatch)
        ));
        let mut truncated = outcome.witness.clone();
        truncated.chosen_actions.pop();
        assert!(matches!(
            truncated.certify(&ledger),
            Err(ReplayError::ChosenActionCount)
        ));
    }
}
