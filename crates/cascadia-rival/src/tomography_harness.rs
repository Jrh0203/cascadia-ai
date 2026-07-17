//! CPU driver for the selfish tomography optimizers (build-scope WI-2).
//!
//! The harness consumes a directory of sealed terminal
//! [`TrajectoryLedger`] JSON files, runs the T0 repacking optimizer and the
//! chronology-preserving replay optimizer for every seat of every game, and
//! emits ONE deterministic, fail-closed JSON summary
//! (`cascadiav3.rival_tomography_summary.v1`).
//!
//! ## Population discipline
//!
//! Each ledger's producing policy is read from ledger metadata: the
//! population identity is the ledger's `source_game_id` with one trailing
//! `-<decimal digits>` game ordinal removed (`rival-pr-cpu-battery-042` →
//! `rival-pr-cpu-battery`).  Every input in one run must derive the same
//! population identity; mixed populations are REFUSED, never averaged.
//!
//! The evidence domain is decided by the same declaration:
//! a population inside the [`crate::INCUMBENT_POLICY_NAMESPACE`]
//! (`incumbent:`) is labeled
//! [`TomographyEvidenceDomain::IncumbentMeasured`] and additionally requires
//! every recorded turn of every input to carry a complete
//! [`TurnEvidenceKind::PolicyDecisionTrace`]; anything else is labeled
//! [`TomographyEvidenceDomain::CpuProxy`].  The trajectory-ledger schema
//! (v1) carries no stronger policy identity, so the namespace declaration
//! plus the decision-trace requirement is the strongest gate available
//! without a ledger schema change; the scientific M1 run owns the
//! correctness of that declaration.
//!
//! ## Kill-bar discipline
//!
//! Every summary carries `witness_semantics = "lower_bound_only"` and fails
//! validation without it: every witness in this file is a feasible LOWER
//! bound.  A heuristic best is never an upper bound, can fund, and can never
//! kill on its own.

use std::{fs, path::Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    INCUMBENT_POLICY_NAMESPACE, LedgerCompletion, LedgerError, RepackConfig, RepackError,
    ReplayConfig, ReplayError, SeatIndex, Sha256Digest, TomographyError, TomographyEvidence,
    TomographyEvidenceDomain, TomographyKind, TomographyPopulation, TomographyResult,
    TrajectoryLedger, TurnEvidenceKind, repack_seat, replay_seat,
};

pub const TOMOGRAPHY_SUMMARY_SCHEMA_ID: &str = "cascadiav3.rival_tomography_summary.v1";
/// The only admissible witness semantics: every witness is a feasible lower
/// bound; none is an optimum or ceiling claim.
pub const WITNESS_SEMANTICS_LOWER_BOUND_ONLY: &str = "lower_bound_only";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HarnessConfig {
    pub seed: u64,
    pub repack_iterations: u32,
    pub beam_width: u32,
    pub candidate_cap: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SummaryInput {
    pub file_name: String,
    /// SHA-256 of the exact input file bytes.
    pub file_sha256: Sha256Digest,
    /// The sealed ledger's own content hash.
    pub ledger_sha256: Sha256Digest,
    pub source_game_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SummarySeatRow {
    pub seat: u8,
    pub realized_total: u16,
    pub repack_witness_total: u16,
    pub repack_delta: i32,
    pub repack_explored_nodes: u64,
    pub replay_witness_total: u16,
    pub replay_delta: i32,
    pub replay_explored_nodes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SummaryGameRow {
    pub source_game_id: String,
    pub seats: Vec<SummarySeatRow>,
}

/// Exact integer distribution statistics for one tomography kind.  The mean
/// is carried as an exact sum/count pair (plus a truncated millipoint
/// convenience value) so the summary is byte-deterministic across platforms;
/// the median is the lower median and the p90 is the nearest-rank 90th
/// percentile of the per-seat deltas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KindAggregate {
    pub kind: TomographyKind,
    pub count: u64,
    pub delta_sum: i64,
    pub mean_delta_millipoints: i64,
    pub median_delta: i32,
    pub p90_delta: i32,
    pub min_delta: i32,
    pub max_delta: i32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "SummaryWire", into = "SummaryWire")]
pub struct TomographySummary {
    schema_id: String,
    witness_semantics: String,
    population: TomographyPopulation,
    seed: u64,
    repack_iterations: u32,
    beam_width: u32,
    candidate_cap: u32,
    inputs: Vec<SummaryInput>,
    games: Vec<SummaryGameRow>,
    aggregates: Vec<KindAggregate>,
    results: Vec<TomographyResult>,
    summary_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SummaryWire {
    schema_id: String,
    witness_semantics: String,
    population: TomographyPopulation,
    seed: u64,
    repack_iterations: u32,
    beam_width: u32,
    candidate_cap: u32,
    inputs: Vec<SummaryInput>,
    games: Vec<SummaryGameRow>,
    aggregates: Vec<KindAggregate>,
    results: Vec<TomographyResult>,
    summary_sha256: Sha256Digest,
}

#[derive(Serialize)]
struct SummaryContent<'a> {
    schema_id: &'a str,
    witness_semantics: &'a str,
    population: &'a TomographyPopulation,
    seed: u64,
    repack_iterations: u32,
    beam_width: u32,
    candidate_cap: u32,
    inputs: &'a [SummaryInput],
    games: &'a [SummaryGameRow],
    aggregates: &'a [KindAggregate],
    results: &'a [TomographyResult],
}

impl TomographySummary {
    pub fn population(&self) -> &TomographyPopulation {
        &self.population
    }

    pub fn witness_semantics(&self) -> &str {
        &self.witness_semantics
    }

    pub fn inputs(&self) -> &[SummaryInput] {
        &self.inputs
    }

    pub fn games(&self) -> &[SummaryGameRow] {
        &self.games
    }

    pub fn aggregates(&self) -> &[KindAggregate] {
        &self.aggregates
    }

    pub fn results(&self) -> &[TomographyResult] {
        &self.results
    }

    pub fn summary_sha256(&self) -> &Sha256Digest {
        &self.summary_sha256
    }

    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, HarnessError> {
        let summary: Self = serde_json::from_slice(bytes)?;
        summary.validate()?;
        Ok(summary)
    }

    pub fn canonical_json_bytes(&self) -> Result<Vec<u8>, HarnessError> {
        self.validate()?;
        Ok(serde_json::to_vec_pretty(self)?)
    }

    /// Durably publish through a same-directory temporary without ever
    /// replacing an existing artifact.
    pub fn write_json_immutable(&self, destination: &Path) -> Result<(), HarnessError> {
        let bytes = self.canonical_json_bytes()?;
        crate::ledger::write_immutable_bytes(destination, &bytes)?;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), HarnessError> {
        if self.schema_id != TOMOGRAPHY_SUMMARY_SCHEMA_ID {
            return Err(HarnessError::WrongSchema);
        }
        if self.witness_semantics != WITNESS_SEMANTICS_LOWER_BOUND_ONLY {
            return Err(HarnessError::WrongWitnessSemantics);
        }
        self.population.validate()?;
        if self.repack_iterations == 0 || self.beam_width == 0 || self.candidate_cap == 0 {
            return Err(HarnessError::InvalidSolverParameters);
        }
        if self.inputs.is_empty() {
            return Err(HarnessError::EmptyInputSet);
        }
        if !self
            .inputs
            .windows(2)
            .all(|pair| pair[0].file_name < pair[1].file_name)
        {
            return Err(HarnessError::UnsortedInputs);
        }
        if self.games.len() != self.inputs.len() {
            return Err(HarnessError::RowAccounting);
        }
        if !self
            .games
            .windows(2)
            .all(|pair| pair[0].source_game_id < pair[1].source_game_id)
        {
            return Err(HarnessError::UnsortedInputs);
        }
        let mut repack_deltas = Vec::new();
        let mut replay_deltas = Vec::new();
        for game in &self.games {
            if game.seats.len() != 4 {
                return Err(HarnessError::RowAccounting);
            }
            for (expected_seat, row) in game.seats.iter().enumerate() {
                if usize::from(row.seat) != expected_seat {
                    return Err(HarnessError::RowAccounting);
                }
                if row.repack_delta < 0
                    || row.replay_delta < 0
                    || i32::from(row.repack_witness_total) - i32::from(row.realized_total)
                        != row.repack_delta
                    || i32::from(row.replay_witness_total) - i32::from(row.realized_total)
                        != row.replay_delta
                {
                    return Err(HarnessError::DeltaArithmetic);
                }
                repack_deltas.push(row.repack_delta);
                replay_deltas.push(row.replay_delta);
            }
        }
        let expected_aggregates = vec![
            aggregate(TomographyKind::T0OwnBoardRepack, &repack_deltas)?,
            aggregate(TomographyKind::T3KnownWorldOneSeatOracle, &replay_deltas)?,
        ];
        if self.aggregates != expected_aggregates {
            return Err(HarnessError::AggregateMismatch);
        }
        if self.results.len() != 2 * repack_deltas.len() {
            return Err(HarnessError::RowAccounting);
        }
        let mut results = self.results.iter();
        for game in &self.games {
            for row in &game.seats {
                for (kind, delta) in [
                    (TomographyKind::T0OwnBoardRepack, row.repack_delta),
                    (TomographyKind::T3KnownWorldOneSeatOracle, row.replay_delta),
                ] {
                    let result = results.next().ok_or(HarnessError::RowAccounting)?;
                    result.validate()?;
                    if result.kind() != kind
                        || result.acting_seat() != row.seat
                        || result.source_game_id() != game.source_game_id
                        || result.evidence_domain() != self.population.evidence_domain
                        || result.incumbent_policy_id() != self.population.incumbent_policy_id
                        || !matches!(result.evidence(), TomographyEvidence::BestFound { .. })
                        || result.evidence().lower_bound() != delta
                        || result.evidence().upper_bound().is_some()
                    {
                        return Err(HarnessError::ResultRowMismatch);
                    }
                }
            }
        }
        if self.recompute_hash()? != self.summary_sha256 {
            return Err(HarnessError::SummaryHashMismatch);
        }
        Ok(())
    }

    fn recompute_hash(&self) -> Result<Sha256Digest, HarnessError> {
        let content = SummaryContent {
            schema_id: &self.schema_id,
            witness_semantics: &self.witness_semantics,
            population: &self.population,
            seed: self.seed,
            repack_iterations: self.repack_iterations,
            beam_width: self.beam_width,
            candidate_cap: self.candidate_cap,
            inputs: &self.inputs,
            games: &self.games,
            aggregates: &self.aggregates,
            results: &self.results,
        };
        let value = serde_json::to_value(&content)?;
        Ok(Sha256Digest::of_bytes(&serde_json::to_vec(&value)?))
    }
}

impl From<TomographySummary> for SummaryWire {
    fn from(value: TomographySummary) -> Self {
        Self {
            schema_id: value.schema_id,
            witness_semantics: value.witness_semantics,
            population: value.population,
            seed: value.seed,
            repack_iterations: value.repack_iterations,
            beam_width: value.beam_width,
            candidate_cap: value.candidate_cap,
            inputs: value.inputs,
            games: value.games,
            aggregates: value.aggregates,
            results: value.results,
            summary_sha256: value.summary_sha256,
        }
    }
}

impl TryFrom<SummaryWire> for TomographySummary {
    type Error = HarnessError;

    fn try_from(value: SummaryWire) -> Result<Self, Self::Error> {
        let summary = Self {
            schema_id: value.schema_id,
            witness_semantics: value.witness_semantics,
            population: value.population,
            seed: value.seed,
            repack_iterations: value.repack_iterations,
            beam_width: value.beam_width,
            candidate_cap: value.candidate_cap,
            inputs: value.inputs,
            games: value.games,
            aggregates: value.aggregates,
            results: value.results,
            summary_sha256: value.summary_sha256,
        };
        summary.validate()?;
        Ok(summary)
    }
}

/// The population identity a sealed ledger declares through its
/// `source_game_id`: the id with one trailing `-<decimal digits>` game
/// ordinal removed, or the whole id when no such ordinal exists.
pub fn derive_population_id(source_game_id: &str) -> &str {
    match source_game_id.rfind('-') {
        Some(position)
            if position > 0
                && position + 1 < source_game_id.len()
                && source_game_id[position + 1..]
                    .bytes()
                    .all(|byte| byte.is_ascii_digit()) =>
        {
            &source_game_id[..position]
        }
        _ => source_game_id,
    }
}

pub fn run_directory(
    directory: &Path,
    config: &HarnessConfig,
) -> Result<TomographySummary, HarnessError> {
    if config.repack_iterations == 0 || config.beam_width == 0 || config.candidate_cap == 0 {
        return Err(HarnessError::InvalidSolverParameters);
    }

    let mut file_names = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            return Err(HarnessError::ForeignInput(
                entry.path().display().to_string(),
            ));
        }
        let file_name = entry
            .file_name()
            .into_string()
            .map_err(|name| HarnessError::ForeignInput(name.to_string_lossy().into_owned()))?;
        if !file_name.ends_with(".json") {
            return Err(HarnessError::ForeignInput(file_name));
        }
        file_names.push(file_name);
    }
    file_names.sort_unstable();
    if file_names.is_empty() {
        return Err(HarnessError::EmptyInputSet);
    }

    let mut inputs = Vec::with_capacity(file_names.len());
    let mut ledgers = Vec::with_capacity(file_names.len());
    let mut population_id: Option<String> = None;
    for file_name in &file_names {
        let bytes = fs::read(directory.join(file_name))?;
        let file_sha256 = Sha256Digest::of_bytes(&bytes);
        let ledger = TrajectoryLedger::from_json_slice(&bytes)
            .map_err(|error| HarnessError::InvalidInput(file_name.clone(), error))?;
        if ledger.completion() != LedgerCompletion::Terminal {
            return Err(HarnessError::NotTerminal(file_name.clone()));
        }
        let derived = derive_population_id(ledger.source_game_id()).to_owned();
        match &population_id {
            None => population_id = Some(derived),
            Some(existing) if *existing != derived => {
                return Err(HarnessError::MixedPopulations {
                    left: existing.clone(),
                    right: derived,
                });
            }
            Some(_) => {}
        }
        if ledgers
            .iter()
            .any(|(_, existing): &(String, TrajectoryLedger)| {
                existing.source_game_id() == ledger.source_game_id()
            })
        {
            return Err(HarnessError::DuplicateSourceGameId(
                ledger.source_game_id().to_owned(),
            ));
        }
        inputs.push(SummaryInput {
            file_name: file_name.clone(),
            file_sha256,
            ledger_sha256: ledger.ledger_sha256().clone(),
            source_game_id: ledger.source_game_id().to_owned(),
        });
        ledgers.push((file_name.clone(), ledger));
    }
    let population_id = population_id.expect("non-empty input set has a population");

    let evidence_domain = if population_id.starts_with(INCUMBENT_POLICY_NAMESPACE) {
        for (file_name, ledger) in &ledgers {
            if ledger
                .turns()
                .iter()
                .any(|turn| turn.evidence_kind != TurnEvidenceKind::PolicyDecisionTrace)
            {
                return Err(HarnessError::IncumbentWithoutDecisionTraces(
                    file_name.clone(),
                ));
            }
        }
        TomographyEvidenceDomain::IncumbentMeasured
    } else {
        TomographyEvidenceDomain::CpuProxy
    };
    let population = TomographyPopulation {
        opponent_population_id: format!("{population_id}:table"),
        incumbent_policy_id: population_id,
        evidence_domain,
    };
    population.validate()?;

    let repack_config = RepackConfig {
        seed: config.seed,
        iterations: config.repack_iterations,
    };
    let replay_config = ReplayConfig {
        seed: config.seed,
        beam_width: config.beam_width as usize,
        candidate_cap: config.candidate_cap as usize,
    };

    let mut games: Vec<(SummaryGameRow, Vec<TomographyResult>)> = Vec::with_capacity(ledgers.len());
    for (_, ledger) in &ledgers {
        let realized = ledger
            .terminal_scores()
            .ok_or_else(|| HarnessError::NotTerminal(ledger.source_game_id().to_owned()))?;
        let mut seats = Vec::with_capacity(4);
        let mut game_results = Vec::with_capacity(8);
        for seat in 0..4u8 {
            let seat_index = SeatIndex::new(seat).map_err(|_| HarnessError::RowAccounting)?;
            let repack = repack_seat(ledger, seat_index, &repack_config, &population)?;
            let replay = replay_seat(ledger, seat_index, &replay_config, &population)?;
            seats.push(SummarySeatRow {
                seat,
                realized_total: realized[usize::from(seat)].total,
                repack_witness_total: repack.witness.witness_score.total,
                repack_delta: repack.witness.score_delta,
                repack_explored_nodes: repack.witness.explored_nodes,
                replay_witness_total: replay.witness.witness_score.total,
                replay_delta: replay.witness.score_delta,
                replay_explored_nodes: replay.witness.explored_nodes,
            });
            game_results.push(repack.result);
            game_results.push(replay.result);
        }
        games.push((
            SummaryGameRow {
                source_game_id: ledger.source_game_id().to_owned(),
                seats,
            },
            game_results,
        ));
    }
    games.sort_by(|left, right| left.0.source_game_id.cmp(&right.0.source_game_id));

    let repack_deltas: Vec<i32> = games
        .iter()
        .flat_map(|(game, _)| game.seats.iter().map(|row| row.repack_delta))
        .collect();
    let replay_deltas: Vec<i32> = games
        .iter()
        .flat_map(|(game, _)| game.seats.iter().map(|row| row.replay_delta))
        .collect();
    let aggregates = vec![
        aggregate(TomographyKind::T0OwnBoardRepack, &repack_deltas)?,
        aggregate(TomographyKind::T3KnownWorldOneSeatOracle, &replay_deltas)?,
    ];

    let mut game_rows = Vec::with_capacity(games.len());
    let mut results = Vec::with_capacity(games.len() * 8);
    for (game, game_results) in games {
        game_rows.push(game);
        results.extend(game_results);
    }

    let mut summary = TomographySummary {
        schema_id: TOMOGRAPHY_SUMMARY_SCHEMA_ID.to_owned(),
        witness_semantics: WITNESS_SEMANTICS_LOWER_BOUND_ONLY.to_owned(),
        population,
        seed: config.seed,
        repack_iterations: config.repack_iterations,
        beam_width: config.beam_width,
        candidate_cap: config.candidate_cap,
        inputs,
        games: game_rows,
        aggregates,
        results,
        summary_sha256: Sha256Digest::of_bytes(b"unsealed"),
    };
    summary.summary_sha256 = summary.recompute_hash()?;
    summary.validate()?;
    Ok(summary)
}

fn aggregate(kind: TomographyKind, deltas: &[i32]) -> Result<KindAggregate, HarnessError> {
    if deltas.is_empty() {
        return Err(HarnessError::EmptyInputSet);
    }
    let mut sorted = deltas.to_vec();
    sorted.sort_unstable();
    let count = sorted.len();
    let delta_sum: i64 = sorted.iter().map(|delta| i64::from(*delta)).sum();
    Ok(KindAggregate {
        kind,
        count: count as u64,
        delta_sum,
        mean_delta_millipoints: delta_sum * 1000 / count as i64,
        median_delta: sorted[(count - 1) / 2],
        p90_delta: sorted[(9 * count).div_ceil(10) - 1],
        min_delta: sorted[0],
        max_delta: sorted[count - 1],
    })
}

#[derive(Debug, Error)]
pub enum HarnessError {
    #[error("unsupported tomography summary schema")]
    WrongSchema,
    #[error("tomography summaries must declare lower_bound_only witness semantics")]
    WrongWitnessSemantics,
    #[error("solver parameters must all be nonzero")]
    InvalidSolverParameters,
    #[error("tomography harness requires at least one sealed input ledger")]
    EmptyInputSet,
    #[error("summary inputs and games must be strictly sorted and unique")]
    UnsortedInputs,
    #[error("input directory entry is not a sealed ledger JSON file: {0}")]
    ForeignInput(String),
    #[error("input ledger {0} failed sealed verification: {1}")]
    InvalidInput(String, LedgerError),
    #[error("input {0} is not a sealed terminal trajectory")]
    NotTerminal(String),
    #[error("mixed input populations refused: {left:?} vs {right:?}")]
    MixedPopulations { left: String, right: String },
    #[error("duplicate source game id across inputs: {0}")]
    DuplicateSourceGameId(String),
    #[error("incumbent-population input {0} lacks complete policy decision traces")]
    IncumbentWithoutDecisionTraces(String),
    #[error("summary rows do not account for every input seat exactly once")]
    RowAccounting,
    #[error("summary delta arithmetic is inconsistent")]
    DeltaArithmetic,
    #[error("summary aggregates do not recompute from the per-seat rows")]
    AggregateMismatch,
    #[error("summary results do not match their per-seat rows")]
    ResultRowMismatch,
    #[error("summary content hash mismatch")]
    SummaryHashMismatch,
    #[error(transparent)]
    Repack(#[from] RepackError),
    #[error(transparent)]
    Replay(#[from] ReplayError),
    #[error(transparent)]
    Tomography(#[from] TomographyError),
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
    fn population_identity_strips_exactly_one_trailing_game_ordinal() {
        assert_eq!(
            derive_population_id("rival-pr-cpu-battery-042"),
            "rival-pr-cpu-battery"
        );
        assert_eq!(
            derive_population_id("incumbent:b0-serving-17"),
            "incumbent:b0-serving"
        );
        assert_eq!(derive_population_id("no-ordinal-x"), "no-ordinal-x");
        assert_eq!(derive_population_id("plain"), "plain");
        assert_eq!(derive_population_id("-7"), "-7");
        assert_eq!(derive_population_id("trailing-"), "trailing-");
    }

    #[test]
    fn nearest_rank_statistics_are_exact_integers() {
        let stats = aggregate(TomographyKind::T0OwnBoardRepack, &[5, 0, 10, 1]).unwrap();
        assert_eq!(stats.count, 4);
        assert_eq!(stats.delta_sum, 16);
        assert_eq!(stats.mean_delta_millipoints, 4000);
        assert_eq!(stats.median_delta, 1);
        assert_eq!(stats.p90_delta, 10);
        assert_eq!(stats.min_delta, 0);
        assert_eq!(stats.max_delta, 10);
        let single = aggregate(TomographyKind::T0OwnBoardRepack, &[3]).unwrap();
        assert_eq!(single.median_delta, 3);
        assert_eq!(single.p90_delta, 3);
    }
}
