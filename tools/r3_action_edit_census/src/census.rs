use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use cascadia_data::{ExactSemanticSupply, PositionRecord};
use cascadia_game::{
    D6Transform, GameConfig, GameSeed, GameState, MarketPrelude, RuleError, TurnAction,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    ActionEdit, AppliedPublicState, PreparedPublicStateTrunk, PublicStateTrunk, R3Error, Result,
    SupplySnapshot, canonical_blake3,
    source::{RuntimeIdentity, capture_runtime_identity, validate_runtime_identity},
};

pub const R3_EXPERIMENT_ID: &str = "r3-action-edit-foundation-v1";
pub const R3_CENSUS_PROTOCOL_ID: &str = "r3-action-edit-open-corpus-v1";
pub const R3_CENSUS_SCHEMA_VERSION: u16 = 2;
pub const R3_SHARD_ARTIFACT_KIND: &str = "r3_action_edit_census_shard";
pub const PRODUCTION_SHARD_COUNT: usize = 4;
pub const POSITIONS_PER_GAME: u64 = 80;
pub const SHARD_PARTITION_RULE: &str = "(seed - cohort_first_seed) % shard_count == shard_index";
pub const DEFAULT_TRAIN_FIRST_SEED: u64 = 3_300_000;
pub const DEFAULT_VALIDATION_FIRST_SEED: u64 = 3_400_000;
pub const DEFAULT_TRAIN_GAMES: u32 = 16;
pub const DEFAULT_VALIDATION_GAMES: u32 = 4;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusContract {
    pub train_first_seed: u64,
    pub train_games: u32,
    pub validation_first_seed: u64,
    pub validation_games: u32,
    pub include_paid_wipe_sentinels: bool,
    pub d6_sentinel_per_position: bool,
}

impl Default for CorpusContract {
    fn default() -> Self {
        Self {
            train_first_seed: DEFAULT_TRAIN_FIRST_SEED,
            train_games: DEFAULT_TRAIN_GAMES,
            validation_first_seed: DEFAULT_VALIDATION_FIRST_SEED,
            validation_games: DEFAULT_VALIDATION_GAMES,
            include_paid_wipe_sentinels: true,
            d6_sentinel_per_position: true,
        }
    }
}

impl CorpusContract {
    pub fn validate(&self) -> Result<()> {
        if self.train_games == 0 || self.validation_games == 0 {
            return Err(invalid(
                "R3 census requires nonempty train and validation cohorts",
            ));
        }
        if !self.include_paid_wipe_sentinels || !self.d6_sentinel_per_position {
            return Err(invalid(
                "the frozen R3 protocol requires paid-wipe and D6 sentinels",
            ));
        }
        let train_end = self
            .train_first_seed
            .checked_add(u64::from(self.train_games))
            .ok_or_else(|| invalid("R3 train seed range overflowed"))?;
        let validation_end = self
            .validation_first_seed
            .checked_add(u64::from(self.validation_games))
            .ok_or_else(|| invalid("R3 validation seed range overflowed"))?;
        if self.train_first_seed < validation_end && self.validation_first_seed < train_end {
            return Err(invalid("R3 train and validation seed ranges overlap"));
        }
        Ok(())
    }

    pub fn is_frozen_production(&self) -> bool {
        self == &Self::default()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CensusConfig {
    pub experiment_id: String,
    pub protocol_id: String,
    pub corpus: CorpusContract,
    pub shard_index: usize,
    pub shard_count: usize,
}

impl Default for CensusConfig {
    fn default() -> Self {
        Self {
            experiment_id: R3_EXPERIMENT_ID.to_owned(),
            protocol_id: R3_CENSUS_PROTOCOL_ID.to_owned(),
            corpus: CorpusContract::default(),
            shard_index: 0,
            shard_count: PRODUCTION_SHARD_COUNT,
        }
    }
}

impl CensusConfig {
    pub fn production_shard(shard_index: usize) -> Self {
        Self {
            shard_index,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.experiment_id != R3_EXPERIMENT_ID {
            return Err(invalid(format!(
                "unexpected R3 experiment ID {}",
                self.experiment_id
            )));
        }
        if self.protocol_id != R3_CENSUS_PROTOCOL_ID {
            return Err(invalid(format!(
                "unexpected R3 census protocol {}",
                self.protocol_id
            )));
        }
        self.corpus.validate()?;
        if self.shard_count == 0 {
            return Err(invalid("R3 shard count must be positive"));
        }
        if self.shard_index >= self.shard_count {
            return Err(invalid(format!(
                "R3 shard index {} must be less than shard count {}",
                self.shard_index, self.shard_count
            )));
        }
        if modulo_owned_seeds(
            self.corpus.train_first_seed,
            self.corpus.train_games,
            self.shard_index,
            self.shard_count,
        )?
        .is_empty()
            || modulo_owned_seeds(
                self.corpus.validation_first_seed,
                self.corpus.validation_games,
                self.shard_index,
                self.shard_count,
            )?
            .is_empty()
        {
            return Err(invalid(
                "R3 shard must own at least one train and one validation seed",
            ));
        }
        Ok(())
    }

    pub fn is_production_coverage(&self) -> bool {
        self.experiment_id == R3_EXPERIMENT_ID
            && self.protocol_id == R3_CENSUS_PROTOCOL_ID
            && self.corpus.is_frozen_production()
            && self.shard_count == PRODUCTION_SHARD_COUNT
            && self.shard_index < PRODUCTION_SHARD_COUNT
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CohortOwnership {
    pub first_seed: u64,
    pub games: u32,
    pub owned_seeds: Vec<u64>,
    pub owned_seeds_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardOwnership {
    pub shard_index: usize,
    pub shard_count: usize,
    pub partition_rule: String,
    pub train: CohortOwnership,
    pub validation: CohortOwnership,
}

impl ShardOwnership {
    pub fn from_config(config: &CensusConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            shard_index: config.shard_index,
            shard_count: config.shard_count,
            partition_rule: SHARD_PARTITION_RULE.to_owned(),
            train: cohort_ownership(
                config.corpus.train_first_seed,
                config.corpus.train_games,
                config.shard_index,
                config.shard_count,
            )?,
            validation: cohort_ownership(
                config.corpus.validation_first_seed,
                config.corpus.validation_games,
                config.shard_index,
                config.shard_count,
            )?,
        })
    }

    pub fn validate_against(&self, config: &CensusConfig) -> Result<()> {
        if self != &Self::from_config(config)? {
            return Err(invalid(
                "R3 shard ownership differs from deterministic modulo ownership",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributionSummary {
    pub count: u64,
    pub minimum: u64,
    pub mean_milli: u64,
    pub median: u64,
    pub p90: u64,
    pub p99: u64,
    pub maximum: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ExactHistogram {
    pub bins: BTreeMap<u64, u64>,
}

impl ExactHistogram {
    fn observe(&mut self, value: u64) -> Result<()> {
        let count = self.bins.entry(value).or_default();
        *count = count
            .checked_add(1)
            .ok_or_else(|| invalid("R3 histogram bin overflowed"))?;
        Ok(())
    }

    pub(crate) fn merge(&mut self, other: &Self) -> Result<()> {
        other.validate_nonempty()?;
        for (value, count) in &other.bins {
            let target = self.bins.entry(*value).or_default();
            *target = target
                .checked_add(*count)
                .ok_or_else(|| invalid("R3 merged histogram bin overflowed"))?;
        }
        Ok(())
    }

    pub fn count(&self) -> Result<u64> {
        self.bins.values().try_fold(0u64, |total, count| {
            total
                .checked_add(*count)
                .ok_or_else(|| invalid("R3 histogram count overflowed"))
        })
    }

    pub fn sum(&self) -> Result<u128> {
        self.bins.iter().try_fold(0u128, |total, (value, count)| {
            total
                .checked_add(u128::from(*value) * u128::from(*count))
                .ok_or_else(|| invalid("R3 histogram sum overflowed"))
        })
    }

    pub(crate) fn validate_nonempty(&self) -> Result<()> {
        if self.bins.is_empty() || self.bins.values().any(|count| *count == 0) {
            return Err(invalid(
                "R3 exact histogram is empty or contains a zero-count bin",
            ));
        }
        let _ = self.count()?;
        let _ = self.sum()?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactDistribution {
    pub histogram: ExactHistogram,
    pub summary: DistributionSummary,
}

impl ExactDistribution {
    pub(crate) fn from_histogram(histogram: ExactHistogram) -> Result<Self> {
        let summary = summarize_histogram(&histogram)?;
        Ok(Self { histogram, summary })
    }

    pub(crate) fn merge<'a>(distributions: impl IntoIterator<Item = &'a Self>) -> Result<Self> {
        let mut histogram = ExactHistogram::default();
        let mut observed = false;
        for distribution in distributions {
            distribution.validate()?;
            histogram.merge(&distribution.histogram)?;
            observed = true;
        }
        if !observed {
            return Err(invalid("cannot merge zero R3 distributions"));
        }
        Self::from_histogram(histogram)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        self.histogram.validate_nonempty()?;
        if self.summary != summarize_histogram(&self.histogram)? {
            return Err(invalid(
                "R3 distribution summary differs from its exact histogram",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RadiusCoverageSummary {
    pub radius: u8,
    pub changed_coordinates: u64,
    pub covered_coordinates: u64,
    pub complete_actions: u64,
    pub total_actions: u64,
    pub covered_parts_per_million: u64,
    pub complete_parts_per_million: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionAssessment {
    pub exact_apply_pass: bool,
    pub authoritative_public_successor_pass: bool,
    pub supply_delta_parity_pass: bool,
    pub regenerated_global_edit_parity_pass: bool,
    pub codec_round_trip_pass: bool,
    pub d6_equivariance_pass: bool,
    pub no_truncation_pass: bool,
    pub trunk_reuse_pass: bool,
    pub median_edit_tokens_le_128: bool,
    pub p99_edit_tokens_le_256: bool,
    pub maximum_edit_tokens_le_384: bool,
    pub p99_edit_bytes_le_8192: bool,
    pub promote_to_matched_mlx_prototype: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct CensusCounters {
    pub train_positions: u64,
    pub validation_positions: u64,
    pub canonical_decisions: u64,
    pub state_trunk_encodings: u64,
    pub canonical_actions: u64,
    pub paid_wipe_sentinel_actions: u64,
    pub exact_apply_checks: u64,
    pub authoritative_public_successor_checks: u64,
    pub supply_delta_parity_checks: u64,
    pub regenerated_global_edit_checks: u64,
    pub codec_round_trip_checks: u64,
    pub d6_checks: u64,
    pub maximum_wildlife_wipe_sequence: u64,
}

impl CensusCounters {
    pub(crate) fn merge(&mut self, other: &Self) -> Result<()> {
        macro_rules! add {
            ($field:ident) => {
                self.$field = self.$field.checked_add(other.$field).ok_or_else(|| {
                    invalid(concat!("R3 counter overflowed: ", stringify!($field)))
                })?;
            };
        }
        add!(train_positions);
        add!(validation_positions);
        add!(canonical_decisions);
        add!(state_trunk_encodings);
        add!(canonical_actions);
        add!(paid_wipe_sentinel_actions);
        add!(exact_apply_checks);
        add!(authoritative_public_successor_checks);
        add!(supply_delta_parity_checks);
        add!(regenerated_global_edit_checks);
        add!(codec_round_trip_checks);
        add!(d6_checks);
        self.maximum_wildlife_wipe_sequence = self
            .maximum_wildlife_wipe_sequence
            .max(other.maximum_wildlife_wipe_sequence);
        Ok(())
    }

    pub(crate) fn total_verified_actions(&self) -> Result<u64> {
        self.canonical_actions
            .checked_add(self.paid_wipe_sentinel_actions)
            .ok_or_else(|| invalid("R3 verified-action count overflowed"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScientificCensus {
    pub schema_version: u16,
    pub artifact_kind: String,
    pub runtime: RuntimeIdentity,
    pub config: CensusConfig,
    pub ownership: ShardOwnership,
    pub production_coverage: bool,
    pub counters: CensusCounters,
    pub action_count: ExactDistribution,
    pub trunk_tokens: ExactDistribution,
    pub trunk_packed_bytes: ExactDistribution,
    pub edit_tokens: ExactDistribution,
    pub edit_packed_bytes: ExactDistribution,
    pub radius_coverage: [RadiusCoverageSummary; 3],
    pub promotion: PromotionAssessment,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CensusReport {
    pub scientific: ScientificCensus,
    pub scientific_blake3: String,
}

#[derive(Default, Clone, Copy)]
struct RadiusAccumulator {
    changed: u64,
    covered: u64,
    complete: u64,
    actions: u64,
}

impl RadiusAccumulator {
    fn merge(&mut self, other: Self) -> Result<()> {
        self.changed = self
            .changed
            .checked_add(other.changed)
            .ok_or_else(|| invalid("R3 merged radius changed-coordinate count overflowed"))?;
        self.covered = self
            .covered
            .checked_add(other.covered)
            .ok_or_else(|| invalid("R3 merged radius covered-coordinate count overflowed"))?;
        self.complete = self
            .complete
            .checked_add(other.complete)
            .ok_or_else(|| invalid("R3 merged radius complete-action count overflowed"))?;
        self.actions = self
            .actions
            .checked_add(other.actions)
            .ok_or_else(|| invalid("R3 merged radius action count overflowed"))?;
        Ok(())
    }
}

#[derive(Default)]
struct CensusAccumulator {
    counters: CensusCounters,
    action_count: ExactHistogram,
    trunk_tokens: ExactHistogram,
    trunk_bytes: ExactHistogram,
    edit_tokens: ExactHistogram,
    edit_bytes: ExactHistogram,
    radius: [RadiusAccumulator; 3],
}

impl CensusAccumulator {
    fn merge(&mut self, other: Self) -> Result<()> {
        self.counters.merge(&other.counters)?;
        self.action_count.merge(&other.action_count)?;
        self.trunk_tokens.merge(&other.trunk_tokens)?;
        self.trunk_bytes.merge(&other.trunk_bytes)?;
        self.edit_tokens.merge(&other.edit_tokens)?;
        self.edit_bytes.merge(&other.edit_bytes)?;
        for (target, source) in self.radius.iter_mut().zip(other.radius) {
            target.merge(source)?;
        }
        Ok(())
    }
}

pub fn run_census(config: &CensusConfig) -> Result<CensusReport> {
    config.validate()?;
    let runtime_before = capture_runtime_identity()?;
    let ownership = ShardOwnership::from_config(config)?;
    let (train_evidence, validation_evidence) = rayon::join(
        || run_cohort(&ownership.train.owned_seeds, true, config),
        || run_cohort(&ownership.validation.owned_seeds, false, config),
    );
    let mut evidence = train_evidence?;
    evidence.merge(validation_evidence?)?;

    let scientific = build_scientific(
        runtime_before.clone(),
        config.clone(),
        ownership,
        evidence.counters,
        ExactDistribution::from_histogram(evidence.action_count)?,
        ExactDistribution::from_histogram(evidence.trunk_tokens)?,
        ExactDistribution::from_histogram(evidence.trunk_bytes)?,
        ExactDistribution::from_histogram(evidence.edit_tokens)?,
        ExactDistribution::from_histogram(evidence.edit_bytes)?,
        radius_summaries(evidence.radius),
    )?;
    let runtime_after = capture_runtime_identity()?;
    if runtime_after != runtime_before {
        return Err(invalid(
            "R3 source bundle or executable changed while the shard was running",
        ));
    }
    let report = CensusReport {
        scientific_blake3: canonical_blake3(&scientific)?,
        scientific,
    };
    validate_census_report(&report, &runtime_after)?;
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_scientific(
    runtime: RuntimeIdentity,
    config: CensusConfig,
    ownership: ShardOwnership,
    counters: CensusCounters,
    action_count: ExactDistribution,
    trunk_tokens: ExactDistribution,
    trunk_packed_bytes: ExactDistribution,
    edit_tokens: ExactDistribution,
    edit_packed_bytes: ExactDistribution,
    radius_coverage: [RadiusCoverageSummary; 3],
) -> Result<ScientificCensus> {
    let promotion = assess_promotion(&counters, &edit_tokens, &edit_packed_bytes)?;
    let scientific = ScientificCensus {
        schema_version: R3_CENSUS_SCHEMA_VERSION,
        artifact_kind: R3_SHARD_ARTIFACT_KIND.to_owned(),
        runtime,
        production_coverage: config.is_production_coverage(),
        config,
        ownership,
        counters,
        action_count,
        trunk_tokens,
        trunk_packed_bytes,
        edit_tokens,
        edit_packed_bytes,
        radius_coverage,
        promotion,
    };
    validate_scientific_census(&scientific)?;
    Ok(scientific)
}

pub(crate) fn validate_census_report(
    report: &CensusReport,
    current_runtime: &RuntimeIdentity,
) -> Result<()> {
    crate::source::validate_blake3("shard scientific", &report.scientific_blake3)?;
    if report.scientific_blake3 != canonical_blake3(&report.scientific)? {
        return Err(invalid("R3 shard scientific BLAKE3 drifted"));
    }
    validate_scientific_census(&report.scientific)?;
    if &report.scientific.runtime != current_runtime {
        return Err(invalid(
            "R3 shard source bundle or executable differs from the current runtime",
        ));
    }
    Ok(())
}

pub(crate) fn validate_scientific_census(scientific: &ScientificCensus) -> Result<()> {
    if scientific.schema_version != R3_CENSUS_SCHEMA_VERSION
        || scientific.artifact_kind != R3_SHARD_ARTIFACT_KIND
    {
        return Err(invalid("R3 shard report schema or artifact kind drifted"));
    }
    validate_runtime_identity(&scientific.runtime)?;
    scientific.config.validate()?;
    scientific.ownership.validate_against(&scientific.config)?;
    if scientific.production_coverage != scientific.config.is_production_coverage() {
        return Err(invalid("R3 shard production-coverage flag drifted"));
    }
    for distribution in [
        &scientific.action_count,
        &scientific.trunk_tokens,
        &scientific.trunk_packed_bytes,
        &scientific.edit_tokens,
        &scientific.edit_packed_bytes,
    ] {
        distribution.validate()?;
    }
    validate_radius_coverage(&scientific.radius_coverage)?;
    validate_counter_evidence(scientific)?;
    if scientific.promotion
        != assess_promotion(
            &scientific.counters,
            &scientific.edit_tokens,
            &scientific.edit_packed_bytes,
        )?
    {
        return Err(invalid(
            "R3 shard promotion assessment differs from exact evidence",
        ));
    }
    Ok(())
}

fn validate_counter_evidence(scientific: &ScientificCensus) -> Result<()> {
    let counters = &scientific.counters;
    let expected_train_positions = u64::try_from(scientific.ownership.train.owned_seeds.len())?
        .checked_mul(POSITIONS_PER_GAME)
        .ok_or_else(|| invalid("R3 expected train position count overflowed"))?;
    let expected_validation_positions =
        u64::try_from(scientific.ownership.validation.owned_seeds.len())?
            .checked_mul(POSITIONS_PER_GAME)
            .ok_or_else(|| invalid("R3 expected validation position count overflowed"))?;
    let expected_decisions = expected_train_positions
        .checked_add(expected_validation_positions)
        .ok_or_else(|| invalid("R3 expected decision count overflowed"))?;
    let total_verified_actions = counters.total_verified_actions()?;
    let expected_d6_checks = expected_decisions
        .checked_mul(12)
        .ok_or_else(|| invalid("R3 expected D6 check count overflowed"))?;
    if counters.train_positions != expected_train_positions
        || counters.validation_positions != expected_validation_positions
        || counters.canonical_decisions != expected_decisions
        || counters.state_trunk_encodings != expected_decisions
        || counters.d6_checks != expected_d6_checks
        || counters.exact_apply_checks != total_verified_actions
        || counters.authoritative_public_successor_checks != total_verified_actions
        || counters.supply_delta_parity_checks != total_verified_actions
        || counters.regenerated_global_edit_checks != total_verified_actions
        || counters.codec_round_trip_checks != total_verified_actions
    {
        return Err(invalid(
            "R3 shard counters do not match owned seeds or exact verification totals",
        ));
    }
    if scientific.action_count.summary.count != expected_decisions
        || scientific.action_count.histogram.sum()? != u128::from(counters.canonical_actions)
        || scientific.trunk_tokens.summary.count != expected_decisions
        || scientific.trunk_packed_bytes.summary.count != expected_decisions
        || scientific.edit_tokens.summary.count != total_verified_actions
        || scientific.edit_packed_bytes.summary.count != total_verified_actions
    {
        return Err(invalid(
            "R3 exact histogram counts do not match shard counters",
        ));
    }
    if scientific
        .radius_coverage
        .iter()
        .any(|coverage| coverage.total_actions != total_verified_actions)
    {
        return Err(invalid(
            "R3 radius coverage action totals do not match verified actions",
        ));
    }
    Ok(())
}

pub(crate) fn assess_promotion(
    counters: &CensusCounters,
    edit_tokens: &ExactDistribution,
    edit_packed_bytes: &ExactDistribution,
) -> Result<PromotionAssessment> {
    let total = counters.total_verified_actions()?;
    let mut promotion = PromotionAssessment {
        exact_apply_pass: counters.exact_apply_checks == total,
        authoritative_public_successor_pass: counters.authoritative_public_successor_checks
            == total,
        supply_delta_parity_pass: counters.supply_delta_parity_checks == total,
        regenerated_global_edit_parity_pass: counters.regenerated_global_edit_checks == total,
        codec_round_trip_pass: counters.codec_round_trip_checks == total,
        d6_equivariance_pass: counters.d6_checks == counters.canonical_decisions.saturating_mul(12),
        no_truncation_pass: true,
        trunk_reuse_pass: counters.state_trunk_encodings == counters.canonical_decisions,
        median_edit_tokens_le_128: edit_tokens.summary.median <= 128,
        p99_edit_tokens_le_256: edit_tokens.summary.p99 <= 256,
        maximum_edit_tokens_le_384: edit_tokens.summary.maximum <= 384,
        p99_edit_bytes_le_8192: edit_packed_bytes.summary.p99 <= 8_192,
        promote_to_matched_mlx_prototype: false,
    };
    promotion.promote_to_matched_mlx_prototype = promotion.exact_apply_pass
        && promotion.authoritative_public_successor_pass
        && promotion.supply_delta_parity_pass
        && promotion.regenerated_global_edit_parity_pass
        && promotion.codec_round_trip_pass
        && promotion.d6_equivariance_pass
        && promotion.no_truncation_pass
        && promotion.trunk_reuse_pass
        && promotion.median_edit_tokens_le_128
        && promotion.p99_edit_tokens_le_256
        && promotion.maximum_edit_tokens_le_384
        && promotion.p99_edit_bytes_le_8192;
    Ok(promotion)
}

pub fn modulo_owned_seeds(
    first_seed: u64,
    games: u32,
    shard_index: usize,
    shard_count: usize,
) -> Result<Vec<u64>> {
    if shard_count == 0 || shard_index >= shard_count {
        return Err(invalid("invalid modulo shard index or count"));
    }
    let mut seeds = Vec::new();
    for offset in 0..games {
        if usize::try_from(offset)? % shard_count == shard_index {
            seeds.push(
                first_seed
                    .checked_add(u64::from(offset))
                    .ok_or_else(|| invalid("R3 cohort seed range overflowed"))?,
            );
        }
    }
    Ok(seeds)
}

fn cohort_ownership(
    first_seed: u64,
    games: u32,
    shard_index: usize,
    shard_count: usize,
) -> Result<CohortOwnership> {
    let owned_seeds = modulo_owned_seeds(first_seed, games, shard_index, shard_count)?;
    Ok(CohortOwnership {
        first_seed,
        games,
        owned_seeds_blake3: canonical_blake3(&owned_seeds)?,
        owned_seeds,
    })
}

fn run_cohort(seeds: &[u64], train: bool, config: &CensusConfig) -> Result<CensusAccumulator> {
    let per_seed = seeds
        .par_iter()
        .map(|raw_seed| run_seed(*raw_seed, train, config))
        .collect::<Vec<_>>();
    let mut combined = CensusAccumulator::default();
    for evidence in per_seed {
        combined.merge(evidence?)?;
    }
    Ok(combined)
}

fn run_seed(raw_seed: u64, train: bool, config: &CensusConfig) -> Result<CensusAccumulator> {
    let mut evidence = CensusAccumulator::default();
    let mut game = GameState::new(GameConfig::research_aaaaa(4)?, GameSeed::from_u64(raw_seed))?;
    while !game.is_game_over() {
        if train {
            evidence.counters.train_positions = evidence
                .counters
                .train_positions
                .checked_add(1)
                .ok_or_else(|| invalid("R3 train position count overflowed"))?;
        } else {
            evidence.counters.validation_positions = evidence
                .counters
                .validation_positions
                .checked_add(1)
                .ok_or_else(|| invalid("R3 validation position count overflowed"))?;
        }
        evidence.counters.canonical_decisions = evidence
            .counters
            .canonical_decisions
            .checked_add(1)
            .ok_or_else(|| invalid("R3 decision count overflowed"))?;
        let game_index = raw_seed
            .checked_mul(100)
            .and_then(|value| value.checked_add(u64::from(game.completed_turns())))
            .ok_or_else(|| invalid("R3 game index overflowed"))?;
        let trunk = PublicStateTrunk::observe(&game, game_index)?;
        let prepared = trunk.prepare_action_edits()?;
        evidence.counters.state_trunk_encodings = evidence
            .counters
            .state_trunk_encodings
            .checked_add(1)
            .ok_or_else(|| invalid("R3 trunk-encoding count overflowed"))?;
        evidence
            .trunk_tokens
            .observe(u64::try_from(trunk.token_count())?)?;
        evidence
            .trunk_bytes
            .observe(u64::try_from(prepared.packed_bytes().len())?)?;
        let (free_prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
        let observed = prepared.observe_legal_actions(&game, &free_prelude)?;
        if observed.is_empty() {
            return Err(invalid("nonterminal census position has no legal actions"));
        }
        evidence
            .action_count
            .observe(u64::try_from(observed.len())?)?;
        for (action, edit) in &observed {
            observe_edit(
                &game,
                action,
                &prepared,
                edit,
                false,
                &mut evidence.counters,
                &mut evidence.edit_tokens,
                &mut evidence.edit_bytes,
                &mut evidence.radius,
            )?;
        }
        if config.corpus.include_paid_wipe_sentinels {
            observe_paid_wipe_sentinels(
                &game,
                &prepared,
                &free_prelude,
                &mut evidence.counters,
                &mut evidence.edit_tokens,
                &mut evidence.edit_bytes,
                &mut evidence.radius,
            )?;
        }
        if config.corpus.d6_sentinel_per_position {
            let (sentinel, source_edit) =
                &observed[deterministic_index(raw_seed, game.completed_turns(), observed.len())];
            verify_d6_sentinel(
                &game,
                sentinel,
                source_edit,
                game_index,
                &mut evidence.counters,
            )?;
        }
        let selected = observed
            [deterministic_index(raw_seed, game.completed_turns(), observed.len())]
        .0
        .clone();
        game.apply(&selected)?;
    }
    if evidence.counters.canonical_decisions != POSITIONS_PER_GAME {
        return Err(invalid(format!(
            "R3 seed {raw_seed} did not produce exactly {POSITIONS_PER_GAME} decisions"
        )));
    }
    Ok(evidence)
}

#[allow(clippy::too_many_arguments)]
fn observe_edit(
    game: &GameState,
    action: &TurnAction,
    prepared: &PreparedPublicStateTrunk<'_>,
    edit: &ActionEdit,
    paid_wipe_sentinel: bool,
    counters: &mut CensusCounters,
    edit_tokens: &mut ExactHistogram,
    edit_bytes: &mut ExactHistogram,
    radius: &mut [RadiusAccumulator; 3],
) -> Result<()> {
    let packed = edit.to_packed_bytes()?;
    let decoded = ActionEdit::from_packed_bytes(&packed)?;
    if &decoded != edit {
        return Err(invalid("action edit codec round trip changed the edit"));
    }
    let applied = prepared.apply(&decoded)?;
    counters.regenerated_global_edit_checks = counters
        .regenerated_global_edit_checks
        .checked_add(1)
        .ok_or_else(|| invalid("R3 regenerated-global-edit check count overflowed"))?;
    verify_authoritative_successor(
        game,
        action,
        &applied,
        prepared.trunk().sparse.global.game_index,
    )?;
    counters.authoritative_public_successor_checks = counters
        .authoritative_public_successor_checks
        .checked_add(1)
        .ok_or_else(|| invalid("R3 authoritative-successor check count overflowed"))?;
    counters.supply_delta_parity_checks = counters
        .supply_delta_parity_checks
        .checked_add(1)
        .ok_or_else(|| invalid("R3 supply-delta check count overflowed"))?;
    counters.exact_apply_checks = counters
        .exact_apply_checks
        .checked_add(1)
        .ok_or_else(|| invalid("R3 exact-apply check count overflowed"))?;
    counters.codec_round_trip_checks = counters
        .codec_round_trip_checks
        .checked_add(1)
        .ok_or_else(|| invalid("R3 codec-round-trip check count overflowed"))?;
    counters.maximum_wildlife_wipe_sequence = counters
        .maximum_wildlife_wipe_sequence
        .max(u64::try_from(edit.factors.wildlife_wipe_masks.len())?);
    if paid_wipe_sentinel {
        counters.paid_wipe_sentinel_actions = counters
            .paid_wipe_sentinel_actions
            .checked_add(1)
            .ok_or_else(|| invalid("R3 paid-wipe sentinel count overflowed"))?;
    } else {
        counters.canonical_actions = counters
            .canonical_actions
            .checked_add(1)
            .ok_or_else(|| invalid("R3 canonical action count overflowed"))?;
    }
    edit_tokens.observe(u64::try_from(edit.token_count())?)?;
    edit_bytes.observe(u64::try_from(packed.len())?)?;
    for (index, coverage) in edit.radius_coverage.iter().enumerate() {
        radius[index].changed = radius[index]
            .changed
            .checked_add(u64::from(coverage.changed_coordinate_count))
            .ok_or_else(|| invalid("R3 radius changed-coordinate count overflowed"))?;
        radius[index].covered = radius[index]
            .covered
            .checked_add(u64::from(coverage.covered_coordinate_count))
            .ok_or_else(|| invalid("R3 radius covered-coordinate count overflowed"))?;
        radius[index].actions = radius[index]
            .actions
            .checked_add(1)
            .ok_or_else(|| invalid("R3 radius action count overflowed"))?;
        radius[index].complete = radius[index]
            .complete
            .checked_add(u64::from(coverage.complete))
            .ok_or_else(|| invalid("R3 radius complete-action count overflowed"))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn observe_paid_wipe_sentinels(
    game: &GameState,
    prepared: &PreparedPublicStateTrunk<'_>,
    free_prelude: &MarketPrelude,
    counters: &mut CensusCounters,
    edit_tokens: &mut ExactHistogram,
    edit_bytes: &mut ExactHistogram,
    radius: &mut [RadiusAccumulator; 3],
) -> Result<()> {
    let after_free = game.preview_market_prelude(free_prelude)?;
    let wipes = after_free.legal_wildlife_wipes();
    if wipes.is_empty() {
        return Ok(());
    }
    for wipe in &wipes {
        let prelude = MarketPrelude {
            replace_three_of_a_kind: free_prelude.replace_three_of_a_kind,
            wildlife_wipes: vec![wipe.clone()],
        };
        let observed = match prepared.observe_legal_actions(game, &prelude) {
            Ok(observed) => observed,
            Err(R3Error::Rule(RuleError::WildlifeBagEmpty)) => continue,
            Err(error) => return Err(error),
        };
        for index in sentinel_indices(observed.len()) {
            let (action, edit) = &observed[index];
            observe_edit(
                game,
                action,
                prepared,
                edit,
                true,
                counters,
                edit_tokens,
                edit_bytes,
                radius,
            )?;
        }
    }
    if after_free.boards()[after_free.current_player()].nature_tokens() >= 2 {
        let mut observed_two_wipe = None;
        'sequences: for first in &wipes {
            for second in &wipes {
                let prelude = MarketPrelude {
                    replace_three_of_a_kind: free_prelude.replace_three_of_a_kind,
                    wildlife_wipes: vec![first.clone(), second.clone()],
                };
                match prepared.observe_legal_actions(game, &prelude) {
                    Ok(observed) => {
                        observed_two_wipe = Some(observed);
                        break 'sequences;
                    }
                    Err(R3Error::Rule(RuleError::WildlifeBagEmpty)) => {}
                    Err(error) => return Err(error),
                }
            }
        }
        let observed = observed_two_wipe.ok_or_else(|| {
            invalid("two Nature Tokens were available but no two-wipe sentinel was feasible")
        })?;
        if let Some((action, edit)) = observed.get(observed.len() / 2) {
            observe_edit(
                game,
                action,
                prepared,
                edit,
                true,
                counters,
                edit_tokens,
                edit_bytes,
                radius,
            )?;
        }
    }
    Ok(())
}

fn verify_d6_sentinel(
    game: &GameState,
    action: &TurnAction,
    source: &ActionEdit,
    game_index: u64,
    counters: &mut CensusCounters,
) -> Result<()> {
    for transform in D6Transform::ALL {
        let transformed_game = game.transformed(transform)?;
        let transformed_action = game.transform_turn_action(action, transform)?;
        let transformed_trunk = PublicStateTrunk::observe(&transformed_game, game_index)?;
        let transformed =
            ActionEdit::observe(&transformed_game, &transformed_trunk, &transformed_action)?;
        if transformed.canonical != source.canonical
            || transformed.selected != source.selected
            || transformed.score_delta != source.score_delta
            || transformed.radius_coverage != source.radius_coverage
        {
            let mut differences = Vec::new();
            if transformed.canonical != source.canonical {
                differences.push(format!(
                    "canonical:{}!={}",
                    canonical_blake3(&source.canonical)?,
                    canonical_blake3(&transformed.canonical)?
                ));
            }
            if transformed.selected != source.selected {
                differences.push(format!(
                    "selected:{}!={}",
                    canonical_blake3(&source.selected)?,
                    canonical_blake3(&transformed.selected)?
                ));
            }
            if transformed.score_delta != source.score_delta {
                differences.push(format!(
                    "score_delta:{}!={}",
                    canonical_blake3(&source.score_delta)?,
                    canonical_blake3(&transformed.score_delta)?
                ));
            }
            if transformed.radius_coverage != source.radius_coverage {
                differences.push(format!(
                    "radius_coverage:{}!={}",
                    canonical_blake3(&source.radius_coverage)?,
                    canonical_blake3(&transformed.radius_coverage)?
                ));
            }
            return Err(invalid(format!(
                "D6 action/edit equivariance failed at game_index {game_index}, completed_turns {}, \
                 transform {}, action {action:?}, differences [{}]",
                game.completed_turns(),
                transform.id(),
                differences.join(", ")
            )));
        }
        let applied = transformed.apply(&transformed_trunk)?;
        verify_authoritative_successor(
            &transformed_game,
            &transformed_action,
            &applied,
            game_index,
        )?;
        counters.d6_checks = counters
            .d6_checks
            .checked_add(1)
            .ok_or_else(|| invalid("R3 D6 check count overflowed"))?;
    }
    Ok(())
}

fn verify_authoritative_successor(
    game: &GameState,
    action: &TurnAction,
    applied: &AppliedPublicState,
    game_index: u64,
) -> Result<()> {
    let public_afterstate = game.preview_public_afterstate(action)?;
    let authoritative_record = PositionRecord::observe_public_for_seat(
        &public_afterstate,
        game_index,
        game.current_player(),
    );
    if applied.record.to_bytes() != authoritative_record.to_bytes() {
        return Err(invalid(
            "applied edit differs from the authoritative normalized public successor",
        ));
    }
    let authoritative_supply =
        SupplySnapshot::from_exact(&ExactSemanticSupply::from_public_state(&public_afterstate)?);
    if applied.supply != authoritative_supply {
        return Err(invalid(
            "applied supply delta differs from the authoritative public semantic supply",
        ));
    }
    Ok(())
}

fn deterministic_index(seed: u64, turn: u16, len: usize) -> usize {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"cascadia-r3-open-corpus-action-v1");
    hasher.update(&seed.to_le_bytes());
    hasher.update(&turn.to_le_bytes());
    let mut bytes = [0; 8];
    bytes.copy_from_slice(&hasher.finalize().as_bytes()[..8]);
    (u64::from_le_bytes(bytes) % len as u64) as usize
}

fn sentinel_indices(len: usize) -> Vec<usize> {
    if len == 0 {
        return Vec::new();
    }
    [0, len / 2, len - 1]
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn summarize_histogram(histogram: &ExactHistogram) -> Result<DistributionSummary> {
    histogram.validate_nonempty()?;
    let count = histogram.count()?;
    let sum = histogram.sum()?;
    let minimum = *histogram
        .bins
        .first_key_value()
        .expect("nonempty histogram was validated")
        .0;
    let maximum = *histogram
        .bins
        .last_key_value()
        .expect("nonempty histogram was validated")
        .0;
    Ok(DistributionSummary {
        count,
        minimum,
        mean_milli: u64::try_from((sum * 1_000 + u128::from(count / 2)) / u128::from(count))
            .map_err(|_| invalid("R3 distribution mean overflowed"))?,
        median: nearest_rank(&histogram.bins, count, 50),
        p90: nearest_rank(&histogram.bins, count, 90),
        p99: nearest_rank(&histogram.bins, count, 99),
        maximum,
    })
}

fn nearest_rank(counts: &BTreeMap<u64, u64>, total: u64, percentile: u64) -> u64 {
    let target = (total * percentile).div_ceil(100).max(1);
    let mut seen = 0;
    for (value, count) in counts {
        seen += count;
        if seen >= target {
            return *value;
        }
    }
    *counts
        .last_key_value()
        .expect("nonempty histogram was validated")
        .0
}

pub(crate) fn merge_radius_coverage(
    inputs: impl IntoIterator<Item = [RadiusCoverageSummary; 3]>,
) -> Result<[RadiusCoverageSummary; 3]> {
    let mut totals = [RadiusAccumulator::default(); 3];
    let mut observed = false;
    for rows in inputs {
        validate_radius_coverage(&rows)?;
        observed = true;
        for (index, row) in rows.iter().enumerate() {
            totals[index].changed = totals[index]
                .changed
                .checked_add(row.changed_coordinates)
                .ok_or_else(|| invalid("R3 merged radius changed count overflowed"))?;
            totals[index].covered = totals[index]
                .covered
                .checked_add(row.covered_coordinates)
                .ok_or_else(|| invalid("R3 merged radius covered count overflowed"))?;
            totals[index].complete = totals[index]
                .complete
                .checked_add(row.complete_actions)
                .ok_or_else(|| invalid("R3 merged radius complete count overflowed"))?;
            totals[index].actions = totals[index]
                .actions
                .checked_add(row.total_actions)
                .ok_or_else(|| invalid("R3 merged radius action count overflowed"))?;
        }
    }
    if !observed {
        return Err(invalid("cannot merge zero R3 radius summaries"));
    }
    Ok(radius_summaries(totals))
}

fn radius_summaries(radius: [RadiusAccumulator; 3]) -> [RadiusCoverageSummary; 3] {
    std::array::from_fn(|index| RadiusCoverageSummary {
        radius: index as u8 + 1,
        changed_coordinates: radius[index].changed,
        covered_coordinates: radius[index].covered,
        complete_actions: radius[index].complete,
        total_actions: radius[index].actions,
        covered_parts_per_million: ratio_ppm(radius[index].covered, radius[index].changed),
        complete_parts_per_million: ratio_ppm(radius[index].complete, radius[index].actions),
    })
}

pub(crate) fn validate_radius_coverage(rows: &[RadiusCoverageSummary; 3]) -> Result<()> {
    for (index, row) in rows.iter().enumerate() {
        if row.radius != index as u8 + 1
            || row.covered_coordinates > row.changed_coordinates
            || row.complete_actions > row.total_actions
            || row.covered_parts_per_million
                != ratio_ppm(row.covered_coordinates, row.changed_coordinates)
            || row.complete_parts_per_million != ratio_ppm(row.complete_actions, row.total_actions)
        {
            return Err(invalid("R3 radius coverage summary drifted"));
        }
    }
    Ok(())
}

fn ratio_ppm(numerator: u64, denominator: u64) -> u64 {
    if denominator == 0 {
        1_000_000
    } else {
        ((u128::from(numerator) * 1_000_000 + u128::from(denominator / 2))
            / u128::from(denominator)) as u64
    }
}

pub fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let tmp = temporary_path(path);
    {
        let mut writer = BufWriter::new(File::create(&tmp)?);
        serde_json::to_writer_pretty(&mut writer, value)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
    }
    fs::rename(tmp, path)?;
    Ok(())
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut value = path.as_os_str().to_owned();
    value.push(".tmp");
    PathBuf::from(value)
}

fn invalid(message: impl Into<String>) -> R3Error {
    R3Error::Invariant(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modulo_partitions_cover_both_frozen_cohorts_exactly() {
        let mut train = Vec::new();
        let mut validation = Vec::new();
        for shard_index in 0..PRODUCTION_SHARD_COUNT {
            let ownership =
                ShardOwnership::from_config(&CensusConfig::production_shard(shard_index)).unwrap();
            assert_eq!(ownership.train.owned_seeds.len(), 4);
            assert_eq!(ownership.validation.owned_seeds.len(), 1);
            train.extend(ownership.train.owned_seeds);
            validation.extend(ownership.validation.owned_seeds);
        }
        train.sort_unstable();
        validation.sort_unstable();
        assert_eq!(
            train,
            (DEFAULT_TRAIN_FIRST_SEED..DEFAULT_TRAIN_FIRST_SEED + u64::from(DEFAULT_TRAIN_GAMES))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            validation,
            (DEFAULT_VALIDATION_FIRST_SEED
                ..DEFAULT_VALIDATION_FIRST_SEED + u64::from(DEFAULT_VALIDATION_GAMES))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn exact_histogram_merge_recomputes_quantiles_from_bins() {
        let left = ExactDistribution::from_histogram(ExactHistogram {
            bins: BTreeMap::from([(1, 3), (100, 1)]),
        })
        .unwrap();
        let right = ExactDistribution::from_histogram(ExactHistogram {
            bins: BTreeMap::from([(2, 1), (3, 3)]),
        })
        .unwrap();
        let merged = ExactDistribution::merge([&left, &right]).unwrap();
        assert_eq!(
            merged.histogram.bins,
            BTreeMap::from([(1, 3), (2, 1), (3, 3), (100, 1)])
        );
        assert_eq!(merged.summary.median, 2);
        assert_eq!(merged.summary.p90, 100);
        assert_eq!(merged.summary.maximum, 100);
    }
}
