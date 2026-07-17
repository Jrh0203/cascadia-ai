//! T0 selfish tomography: static own-seat terminal repacking.
//!
//! Given one seat's terminal board from a sealed [`TrajectoryLedger`], this
//! module freezes that seat's realized multiset — drafted tile identities,
//! placed wildlife multiset, the starter cluster (position and rotation),
//! and the terminal nature-token count — and searches for a better *legal
//! terminal arrangement* of exactly that multiset.  Every candidate is
//! materialized through real [`cascadia_game::Board`] placement calls and
//! scored by the canonical scorer; nothing is approximated.
//!
//! ## Frozen-economy rules (documented constraints)
//!
//! - The starter cluster is fixed in place: it is dealt, not played, so a
//!   repack that moved it would not be an arrangement of the seat's own
//!   decisions.
//! - Wildlife may be reassigned to any compatible tile, but the number of
//!   keystone tiles hosting wildlife is held equal to the realized count.
//!   Keystone placements mint nature tokens; freezing the hosted-keystone
//!   count (and then re-spending the realized number of spent tokens through
//!   [`cascadia_game::Board::spend_nature_token`]) keeps the terminal
//!   nature-token score component *identical* to the realized game, so the
//!   whole witness delta is arrangement headroom, never token-economy
//!   invention.  This slightly narrows the search space; the witness remains
//!   a valid feasible lower bound.
//!
//! ## Evidence discipline
//!
//! The output is a [`TomographyResult`] of kind
//! [`TomographyKind::T0OwnBoardRepack`] with
//! [`TomographyEvidence::BestFound`] — a feasible-witness LOWER bound on the
//! repacked optimum, explicitly optimistic (future-agnostic, non-policy) via
//! [`crate::InformationBoundary::AcquiredResourcesOnly`].  A heuristic best
//! is never emitted as an upper bound.
//!
//! Determinism: the annealer runs on a ChaCha8 stream derived from the
//! configured seed, the sealed ledger hash, and the seat; a fixed seed
//! yields a byte-identical witness.

use std::collections::{BTreeSet, HashMap, VecDeque};

use cascadia_game::{
    Board, BoardError, HexCoord, Rotation, RuleError, ScoreBreakdown, ScoringCards, Tile, Wildlife,
    score_board,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    LedgerError, SeatIndex, Sha256Digest, TomographyError, TomographyEvidence, TomographyKind,
    TomographyPopulation, TomographyResult, TomographyResultInput, TrajectoryLedger,
};

pub const REPACK_SOLVER_ID: &str = "cascadiav3.rival_tomography_repack_solver.v1";

/// Largest uphill score move the threshold annealer will accept at step 0;
/// the allowance decays linearly to zero (pure hill climbing) by the final
/// step.  Integer-only so acceptance is platform-exact.
const MAX_UPHILL_ALLOWANCE: i32 = 4;
/// Bounded in-proposal retries before a move kind is abandoned this step.
const PROPOSAL_RETRIES: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepackConfig {
    /// Deterministic search seed; recorded in the witness and result.
    pub seed: u64,
    /// Annealing steps (candidate proposals).
    pub iterations: u32,
}

/// One drafted-tile placement of the witness program, in the exact order a
/// verifier must replay it (each placement attaches to the board built so
/// far).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WitnessTilePlacement {
    pub tile: Tile,
    pub coord: HexCoord,
    pub rotation: Rotation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WitnessWildlifePlacement {
    pub coord: HexCoord,
    pub wildlife: Wildlife,
}

/// A materializable repacking witness.  [`RepackWitness::rebuild_board`]
/// replays the program through the canonical engine and fails closed on any
/// legality or claimed-score mismatch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepackWitness {
    pub solver_id: String,
    pub source_game_id: String,
    pub ledger_sha256: Sha256Digest,
    pub seat: u8,
    pub seed: u64,
    pub iterations: u32,
    pub explored_nodes: u64,
    pub realized_score: ScoreBreakdown,
    pub witness_score: ScoreBreakdown,
    pub score_delta: i32,
    pub tile_placements: Vec<WitnessTilePlacement>,
    pub wildlife_placements: Vec<WitnessWildlifePlacement>,
    /// Nature tokens spent after the rebuild to restore the frozen realized
    /// terminal count (realized spends are not re-optimized).
    pub tokens_spent: u8,
}

impl RepackWitness {
    /// Round-trips the witness through the canonical engine: rebuilds the
    /// board with real placement calls from the ledger's sealed starter
    /// cluster and asserts the exact claimed score.
    pub fn rebuild_board(&self, ledger: &TrajectoryLedger) -> Result<Board, RepackError> {
        if self.ledger_sha256 != *ledger.ledger_sha256() {
            return Err(RepackError::WitnessLedgerMismatch);
        }
        let initial = ledger.initial_state()?;
        let seat = usize::from(self.seat);
        let mut board = initial
            .boards()
            .get(seat)
            .ok_or(RepackError::SeatOutOfRange(self.seat))?
            .clone();
        for placement in &self.tile_placements {
            board.place_tile(placement.coord, placement.tile, placement.rotation)?;
        }
        for placement in &self.wildlife_placements {
            board.place_wildlife(placement.coord, placement.wildlife)?;
        }
        for _ in 0..self.tokens_spent {
            if !board.spend_nature_token() {
                return Err(RepackError::TokenAccounting);
            }
        }
        let score = score_board(&board, ledger.config().scoring_cards);
        if score != self.witness_score {
            return Err(RepackError::WitnessScoreMismatch);
        }
        if self.score_delta
            != i32::from(self.witness_score.total) - i32::from(self.realized_score.total)
        {
            return Err(RepackError::DeltaArithmetic);
        }
        Ok(board)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepackOutcome {
    pub witness: RepackWitness,
    pub result: TomographyResult,
}

/// One board slot of the search state: a tile, whether it is part of the
/// frozen starter cluster, and its (re)assigned wildlife.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Slot {
    tile: Tile,
    fixed: bool,
    coord: HexCoord,
    rotation: Rotation,
    wildlife: Option<Wildlife>,
}

struct Frozen {
    initial_board: Board,
    cards: ScoringCards,
    /// Realized count of keystone tiles hosting wildlife (token mints).
    awarded_tokens: u8,
    /// Realized token spends (mints minus terminal count).
    tokens_spent: u8,
    realized_score: ScoreBreakdown,
}

pub fn repack_seat(
    ledger: &TrajectoryLedger,
    seat: SeatIndex,
    config: &RepackConfig,
    population: &TomographyPopulation,
) -> Result<RepackOutcome, RepackError> {
    population.validate()?;
    if config.iterations == 0 {
        return Err(RepackError::ZeroIterations);
    }
    let (frozen, realized_slots) = extract_frozen_problem(ledger, seat)?;

    let mut rng = ChaCha8Rng::from_seed(derive_search_seed(
        config.seed,
        ledger.ledger_sha256(),
        seat.get(),
    ));
    let mut current = realized_slots;
    let mut current_score = materialize(&current, &frozen)?.1;
    if current_score != frozen.realized_score {
        return Err(RepackError::RealizedScoreMismatch);
    }
    let mut best = current.clone();
    let mut best_score = current_score;
    let mut explored_nodes = 1u64;

    for step in 0..config.iterations {
        let Some(candidate) = propose(&current, &mut rng) else {
            continue;
        };
        let Ok((_, candidate_score)) = materialize(&candidate, &frozen) else {
            explored_nodes += 1;
            continue;
        };
        explored_nodes += 1;
        let allowance = (i64::from(MAX_UPHILL_ALLOWANCE) * i64::from(config.iterations - 1 - step)
            / i64::from(config.iterations)) as i32;
        if i32::from(candidate_score.total) + allowance >= i32::from(current_score.total) {
            current = candidate;
            current_score = candidate_score;
            if current_score.total > best_score.total {
                best = current.clone();
                best_score = current_score;
            }
        }
    }

    let (witness_board, witness_score) = materialize(&best, &frozen)?;
    assert_frozen_multiset(&witness_board, &best, &frozen)?;
    if witness_score.total < frozen.realized_score.total {
        // Impossible by construction (search starts from the realized
        // arrangement), but the lower-bound claim must never rest on an
        // unchecked invariant.
        return Err(RepackError::WitnessBelowRealized);
    }

    let (tile_placements, wildlife_placements) = witness_program(&best)?;
    let witness = RepackWitness {
        solver_id: REPACK_SOLVER_ID.to_owned(),
        source_game_id: ledger.source_game_id().to_owned(),
        ledger_sha256: ledger.ledger_sha256().clone(),
        seat: seat.get(),
        seed: config.seed,
        iterations: config.iterations,
        explored_nodes,
        realized_score: frozen.realized_score,
        witness_score,
        score_delta: i32::from(witness_score.total) - i32::from(frozen.realized_score.total),
        tile_placements,
        wildlife_placements,
        tokens_spent: frozen.tokens_spent,
    };
    // Fail-closed self-verification: the emitted witness must round-trip
    // legality and its exact claimed score through the canonical engine.
    let rebuilt = witness.rebuild_board(ledger)?;
    if rebuilt != witness_board {
        return Err(RepackError::WitnessScoreMismatch);
    }

    let solver_config_sha256 = Sha256Digest::of_bytes(&serde_json::to_vec(&serde_json::json!({
        "solver_id": REPACK_SOLVER_ID,
        "seed": config.seed,
        "iterations": config.iterations,
    }))?);
    let witness_ledger_sha256 = Sha256Digest::of_bytes(&serde_json::to_vec(&witness)?);
    let result = TomographyResult::try_new_in_domain(
        TomographyResultInput {
            kind: TomographyKind::T0OwnBoardRepack,
            root_id: Sha256Digest::of_bytes(
                format!(
                    "{}:{}:t0-own-board-repack",
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
    Ok(RepackOutcome { witness, result })
}

fn derive_search_seed(seed: u64, ledger_sha256: &Sha256Digest, seat: u8) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(b"cascadiav3.rival_tomography_repack_seed.v1");
    hasher.update(seed.to_le_bytes());
    hasher.update(ledger_sha256.as_str().as_bytes());
    hasher.update([seat]);
    hasher.finalize().into()
}

fn extract_frozen_problem(
    ledger: &TrajectoryLedger,
    seat: SeatIndex,
) -> Result<(Frozen, Vec<Slot>), RepackError> {
    if ledger.completion() != crate::LedgerCompletion::Terminal {
        return Err(RepackError::NotTerminal);
    }
    let initial = ledger.initial_state()?;
    let terminal = ledger.raw_final_state()?;
    let seat_index = usize::from(seat.get());
    let initial_board = initial
        .boards()
        .get(seat_index)
        .ok_or(RepackError::SeatOutOfRange(seat.get()))?
        .clone();
    let terminal_board = terminal
        .boards()
        .get(seat_index)
        .ok_or(RepackError::SeatOutOfRange(seat.get()))?;

    let starter_coords: BTreeSet<HexCoord> = initial_board
        .placed_tiles()
        .map(|(coord, _)| coord)
        .collect();
    let mut slots = Vec::with_capacity(terminal_board.tile_count());
    // Fixed starter slots first, in the initial board's deterministic order.
    for (coord, placed) in initial_board.placed_tiles() {
        let terminal_placed = terminal_board
            .tile_at(coord)
            .ok_or(RepackError::StarterTileMissing(coord))?;
        if terminal_placed.tile != placed.tile || terminal_placed.rotation != placed.rotation {
            return Err(RepackError::StarterTileMissing(coord));
        }
        slots.push(Slot {
            tile: placed.tile,
            fixed: true,
            coord,
            rotation: placed.rotation,
            wildlife: terminal_placed.wildlife,
        });
    }
    for (coord, placed) in terminal_board.placed_tiles() {
        if starter_coords.contains(&coord) {
            continue;
        }
        slots.push(Slot {
            tile: placed.tile,
            fixed: false,
            coord,
            rotation: placed.rotation,
            wildlife: placed.wildlife,
        });
    }

    let awarded_tokens = slots
        .iter()
        .filter(|slot| slot.tile.keystone && slot.wildlife.is_some())
        .count() as u8;
    let terminal_tokens = terminal_board.nature_tokens();
    let tokens_spent = awarded_tokens
        .checked_sub(terminal_tokens)
        .ok_or(RepackError::TokenAccounting)?;

    let realized_score = *ledger
        .terminal_scores()
        .ok_or(RepackError::NotTerminal)?
        .get(seat_index)
        .ok_or(RepackError::SeatOutOfRange(seat.get()))?;
    let cards = ledger.config().scoring_cards;
    // The research ruleset scores without habitat bonuses, so the per-board
    // canonical scorer must reproduce the sealed terminal score exactly.
    if score_board(terminal_board, cards) != realized_score {
        return Err(RepackError::RealizedScoreMismatch);
    }

    Ok((
        Frozen {
            initial_board,
            cards,
            awarded_tokens,
            tokens_spent,
            realized_score,
        },
        slots,
    ))
}

/// Rebuilds a candidate arrangement through canonical placement calls in a
/// deterministic attach order (BFS from the starter cluster), then scores it
/// with the canonical scorer.  Any illegality rejects the candidate.
fn materialize(slots: &[Slot], frozen: &Frozen) -> Result<(Board, ScoreBreakdown), RepackError> {
    let order = attach_order(slots)?;
    let mut board = frozen.initial_board.clone();
    for &index in &order {
        let slot = &slots[index];
        if !slot.fixed {
            board.place_tile(slot.coord, slot.tile, slot.rotation)?;
        }
    }
    let mut wildlife: Vec<(HexCoord, Wildlife)> = slots
        .iter()
        .filter_map(|slot| slot.wildlife.map(|wildlife| (slot.coord, wildlife)))
        .collect();
    wildlife.sort_unstable_by_key(|(coord, wildlife)| (*coord, *wildlife as u8));
    for (coord, animal) in wildlife {
        board.place_wildlife(coord, animal)?;
    }
    if board.nature_tokens() != frozen.awarded_tokens {
        return Err(RepackError::TokenAccounting);
    }
    for _ in 0..frozen.tokens_spent {
        if !board.spend_nature_token() {
            return Err(RepackError::TokenAccounting);
        }
    }
    let score = score_board(&board, frozen.cards);
    Ok((board, score))
}

/// Deterministic buildable order: BFS over the occupied cells from the fixed
/// starter cluster.  Errors if two slots collide or the arrangement is not
/// one connected environment (such an arrangement is not reachable by legal
/// play and must be rejected, not repaired).
fn attach_order(slots: &[Slot]) -> Result<Vec<usize>, RepackError> {
    let mut by_coord: HashMap<HexCoord, usize> = HashMap::with_capacity(slots.len());
    for (index, slot) in slots.iter().enumerate() {
        if slot.coord.to_index().is_none() {
            return Err(RepackError::OutOfBounds(slot.coord));
        }
        if by_coord.insert(slot.coord, index).is_some() {
            return Err(RepackError::OverlappingSlots(slot.coord));
        }
    }
    let mut order = Vec::with_capacity(slots.len());
    let mut visited = vec![false; slots.len()];
    let mut queue = VecDeque::new();
    for (index, slot) in slots.iter().enumerate() {
        if slot.fixed {
            visited[index] = true;
            queue.push_back(index);
        }
    }
    if queue.is_empty() {
        return Err(RepackError::EmptyStarterCluster);
    }
    while let Some(index) = queue.pop_front() {
        order.push(index);
        for edge in 0..6 {
            let neighbor = slots[index].coord.neighbor(edge);
            if let Some(&neighbor_index) = by_coord.get(&neighbor)
                && !visited[neighbor_index]
            {
                visited[neighbor_index] = true;
                queue.push_back(neighbor_index);
            }
        }
    }
    if order.len() != slots.len() {
        return Err(RepackError::DisconnectedArrangement);
    }
    Ok(order)
}

fn propose(slots: &[Slot], rng: &mut ChaCha8Rng) -> Option<Vec<Slot>> {
    let movable: Vec<usize> = slots
        .iter()
        .enumerate()
        .filter_map(|(index, slot)| (!slot.fixed).then_some(index))
        .collect();
    for _ in 0..PROPOSAL_RETRIES {
        let kind = rng.gen_range(0..5u8);
        let mut next = slots.to_vec();
        let applied = match kind {
            0 => propose_swap_tiles(&mut next, &movable, rng),
            1 => propose_rotate(&mut next, &movable, rng),
            2 => propose_relocate(&mut next, &movable, rng),
            3 => propose_swap_wildlife(&mut next, rng),
            _ => propose_move_wildlife(&mut next, rng),
        };
        if applied {
            return Some(next);
        }
    }
    None
}

fn propose_swap_tiles(slots: &mut [Slot], movable: &[usize], rng: &mut ChaCha8Rng) -> bool {
    if movable.len() < 2 {
        return false;
    }
    let a = movable[rng.gen_range(0..movable.len())];
    let b = movable[rng.gen_range(0..movable.len())];
    if a == b {
        return false;
    }
    let (left, right) = (slots[a].coord, slots[b].coord);
    slots[a].coord = right;
    slots[b].coord = left;
    true
}

fn propose_rotate(slots: &mut [Slot], movable: &[usize], rng: &mut ChaCha8Rng) -> bool {
    let duals: Vec<usize> = movable
        .iter()
        .copied()
        .filter(|&index| slots[index].tile.terrain_b.is_some())
        .collect();
    if duals.is_empty() {
        return false;
    }
    let index = duals[rng.gen_range(0..duals.len())];
    let rotation = Rotation::ALL[rng.gen_range(0..Rotation::ALL.len())];
    if rotation == slots[index].rotation {
        return false;
    }
    slots[index].rotation = rotation;
    true
}

fn propose_relocate(slots: &mut [Slot], movable: &[usize], rng: &mut ChaCha8Rng) -> bool {
    if movable.is_empty() {
        return false;
    }
    let moving = movable[rng.gen_range(0..movable.len())];
    let occupied: BTreeSet<HexCoord> = slots
        .iter()
        .enumerate()
        .filter_map(|(index, slot)| (index != moving).then_some(slot.coord))
        .collect();
    let mut frontier: BTreeSet<HexCoord> = BTreeSet::new();
    for coord in &occupied {
        for edge in 0..6 {
            let neighbor = coord.neighbor(edge);
            if neighbor.to_index().is_some()
                && !occupied.contains(&neighbor)
                && neighbor != slots[moving].coord
            {
                frontier.insert(neighbor);
            }
        }
    }
    if frontier.is_empty() {
        return false;
    }
    let target_index = rng.gen_range(0..frontier.len());
    let target = *frontier.iter().nth(target_index).expect("index in range");
    slots[moving].coord = target;
    true
}

fn propose_swap_wildlife(slots: &mut [Slot], rng: &mut ChaCha8Rng) -> bool {
    let hosting: Vec<usize> = slots
        .iter()
        .enumerate()
        .filter_map(|(index, slot)| slot.wildlife.is_some().then_some(index))
        .collect();
    if hosting.len() < 2 {
        return false;
    }
    let a = hosting[rng.gen_range(0..hosting.len())];
    let b = hosting[rng.gen_range(0..hosting.len())];
    if a == b {
        return false;
    }
    let (wildlife_a, wildlife_b) = (slots[a].wildlife, slots[b].wildlife);
    let (Some(animal_a), Some(animal_b)) = (wildlife_a, wildlife_b) else {
        return false;
    };
    if animal_a == animal_b
        || !slots[a].tile.wildlife.contains(animal_b)
        || !slots[b].tile.wildlife.contains(animal_a)
    {
        return false;
    }
    slots[a].wildlife = Some(animal_b);
    slots[b].wildlife = Some(animal_a);
    true
}

/// Moves one hosted token to an empty compatible tile with the same keystone
/// status, preserving the frozen minted-token count.
fn propose_move_wildlife(slots: &mut [Slot], rng: &mut ChaCha8Rng) -> bool {
    let hosting: Vec<usize> = slots
        .iter()
        .enumerate()
        .filter_map(|(index, slot)| slot.wildlife.is_some().then_some(index))
        .collect();
    if hosting.is_empty() {
        return false;
    }
    let from = hosting[rng.gen_range(0..hosting.len())];
    let animal = slots[from].wildlife.expect("hosting slot has wildlife");
    let keystone = slots[from].tile.keystone;
    let targets: Vec<usize> = slots
        .iter()
        .enumerate()
        .filter_map(|(index, slot)| {
            (index != from
                && slot.wildlife.is_none()
                && slot.tile.keystone == keystone
                && slot.tile.wildlife.contains(animal))
            .then_some(index)
        })
        .collect();
    if targets.is_empty() {
        return false;
    }
    let to = targets[rng.gen_range(0..targets.len())];
    slots[from].wildlife = None;
    slots[to].wildlife = Some(animal);
    true
}

fn witness_program(
    slots: &[Slot],
) -> Result<(Vec<WitnessTilePlacement>, Vec<WitnessWildlifePlacement>), RepackError> {
    let order = attach_order(slots)?;
    let tile_placements = order
        .iter()
        .filter(|&&index| !slots[index].fixed)
        .map(|&index| WitnessTilePlacement {
            tile: slots[index].tile,
            coord: slots[index].coord,
            rotation: slots[index].rotation,
        })
        .collect();
    let mut wildlife_placements: Vec<WitnessWildlifePlacement> = slots
        .iter()
        .filter_map(|slot| {
            slot.wildlife.map(|wildlife| WitnessWildlifePlacement {
                coord: slot.coord,
                wildlife,
            })
        })
        .collect();
    wildlife_placements
        .sort_unstable_by_key(|placement| (placement.coord, placement.wildlife as u8));
    Ok((tile_placements, wildlife_placements))
}

/// Frozen-multiset invariants a witness must satisfy verbatim.
fn assert_frozen_multiset(
    board: &Board,
    slots: &[Slot],
    frozen: &Frozen,
) -> Result<(), RepackError> {
    let mut witness_tiles: Vec<u8> = board
        .placed_tiles()
        .map(|(_, placed)| placed.tile.id.0)
        .collect();
    witness_tiles.sort_unstable();
    let mut frozen_tiles: Vec<u8> = slots.iter().map(|slot| slot.tile.id.0).collect();
    frozen_tiles.sort_unstable();
    if witness_tiles != frozen_tiles {
        return Err(RepackError::FrozenMultisetViolation);
    }
    let mut witness_wildlife: Vec<u8> = board
        .placed_tiles()
        .filter_map(|(_, placed)| placed.wildlife.map(|wildlife| wildlife as u8))
        .collect();
    witness_wildlife.sort_unstable();
    let mut frozen_wildlife: Vec<u8> = slots
        .iter()
        .filter_map(|slot| slot.wildlife.map(|wildlife| wildlife as u8))
        .collect();
    frozen_wildlife.sort_unstable();
    if witness_wildlife != frozen_wildlife {
        return Err(RepackError::FrozenMultisetViolation);
    }
    for (coord, placed) in frozen.initial_board.placed_tiles() {
        let witness_placed = board
            .tile_at(coord)
            .ok_or(RepackError::StarterTileMissing(coord))?;
        if witness_placed.tile != placed.tile || witness_placed.rotation != placed.rotation {
            return Err(RepackError::StarterTileMissing(coord));
        }
    }
    if board.nature_tokens() + frozen.tokens_spent != frozen.awarded_tokens {
        return Err(RepackError::TokenAccounting);
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum RepackError {
    #[error("repacking requires a sealed terminal trajectory ledger")]
    NotTerminal,
    #[error("repack configuration must run at least one iteration")]
    ZeroIterations,
    #[error("seat {0} is outside the four-player research table")]
    SeatOutOfRange(u8),
    #[error("witness does not belong to the supplied ledger")]
    WitnessLedgerMismatch,
    #[error("starter cluster tile at {0:?} is missing or altered")]
    StarterTileMissing(HexCoord),
    #[error("candidate arrangement places two tiles at {0:?}")]
    OverlappingSlots(HexCoord),
    #[error("candidate coordinate {0:?} is outside the supported board")]
    OutOfBounds(HexCoord),
    #[error("seat has no fixed starter cluster to attach to")]
    EmptyStarterCluster,
    #[error("candidate arrangement is not one connected environment")]
    DisconnectedArrangement,
    #[error("nature-token accounting does not reproduce the frozen terminal count")]
    TokenAccounting,
    #[error("canonical per-board score does not reproduce the sealed terminal score")]
    RealizedScoreMismatch,
    #[error("witness board does not reproduce its exact claimed score")]
    WitnessScoreMismatch,
    #[error("witness delta arithmetic is inconsistent")]
    DeltaArithmetic,
    #[error("search returned a witness below the realized score")]
    WitnessBelowRealized,
    #[error("witness violates the frozen tile/wildlife multiset")]
    FrozenMultisetViolation,
    #[error(transparent)]
    Ledger(#[from] LedgerError),
    #[error(transparent)]
    Rules(#[from] RuleError),
    #[error(transparent)]
    Board(#[from] BoardError),
    #[error(transparent)]
    Tomography(#[from] TomographyError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameConfig, GameSeed, GameState};

    use crate::TrajectoryLedgerBuilder;

    use super::*;

    fn terminal_fixture(seed: u64) -> TrajectoryLedger {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(seed),
        )
        .unwrap();
        let mut builder =
            TrajectoryLedgerBuilder::new(format!("repack-unit-fixture-{seed}"), game).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x7265_7061);
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
            incumbent_policy_id: "repack-unit-fixture".to_owned(),
            opponent_population_id: "repack-unit-fixture:table".to_owned(),
            evidence_domain: crate::TomographyEvidenceDomain::CpuProxy,
        }
    }

    #[test]
    fn realized_arrangement_materializes_to_the_sealed_score() {
        let ledger = terminal_fixture(9101);
        for seat in 0..4u8 {
            let (frozen, slots) =
                extract_frozen_problem(&ledger, SeatIndex::new(seat).unwrap()).unwrap();
            let (_, score) = materialize(&slots, &frozen).unwrap();
            assert_eq!(score, frozen.realized_score);
            assert_eq!(
                score,
                ledger.terminal_scores().unwrap()[usize::from(seat)],
                "canonical per-board scoring must reproduce the sealed terminal score"
            );
        }
    }

    #[test]
    fn disconnected_and_overlapping_candidates_are_rejected_not_repaired() {
        let ledger = terminal_fixture(9101);
        let (frozen, mut slots) =
            extract_frozen_problem(&ledger, SeatIndex::new(0).unwrap()).unwrap();
        let movable = slots.iter().position(|slot| !slot.fixed).unwrap();
        let original = slots[movable].coord;
        slots[movable].coord = HexCoord::new(20, 20);
        assert!(matches!(
            materialize(&slots, &frozen),
            Err(RepackError::DisconnectedArrangement)
        ));
        slots[movable].coord = slots[0].coord;
        assert!(matches!(
            materialize(&slots, &frozen),
            Err(RepackError::OverlappingSlots(_))
        ));
        slots[movable].coord = original;
        materialize(&slots, &frozen).unwrap();
    }

    #[test]
    fn repack_is_deterministic_and_improving() {
        let ledger = terminal_fixture(9102);
        let config = RepackConfig {
            seed: 7,
            iterations: 120,
        };
        let seat = SeatIndex::new(1).unwrap();
        let left = repack_seat(&ledger, seat, &config, &proxy_population()).unwrap();
        let right = repack_seat(&ledger, seat, &config, &proxy_population()).unwrap();
        assert_eq!(left, right);
        assert!(left.witness.witness_score.total >= left.witness.realized_score.total);
        assert_eq!(
            left.witness.score_delta,
            i32::from(left.witness.witness_score.total)
                - i32::from(left.witness.realized_score.total)
        );
        assert_eq!(
            left.result.evidence().lower_bound(),
            left.witness.score_delta
        );
        assert_eq!(left.result.evidence().upper_bound(), None);
        left.witness.rebuild_board(&ledger).unwrap();
    }

    #[test]
    fn tampered_witness_fails_rebuild() {
        let ledger = terminal_fixture(9103);
        let config = RepackConfig {
            seed: 3,
            iterations: 60,
        };
        let outcome = repack_seat(
            &ledger,
            SeatIndex::new(2).unwrap(),
            &config,
            &proxy_population(),
        )
        .unwrap();
        let mut tampered = outcome.witness.clone();
        tampered.witness_score.total += 1;
        assert!(tampered.rebuild_board(&ledger).is_err());
    }
}
