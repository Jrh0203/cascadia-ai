use std::collections::BTreeMap;

use cascadia_data::{CanonicalTileArchetype, standard_semantic_archetype_catalog};
use cascadia_game::{GameState, Rotation, Wildlife};
use r2_sparse_entity_census::{AxialCoord, FrontierToken, SparsePublicState, SuppliedTile};
use r3_action_edit_census::{
    ActionEdit, AppliedPublicState, PublicStateTrunk, SupplySnapshot, TileSemantic,
};
use serde::{Deserialize, Serialize};

use crate::{
    BoardGraph, CommonConfig, ExperimentLane, RelationalStateGraph, ReportEnvelope, Result,
    WildlifeOpportunitySummary,
    common::{deterministic_index, envelope, run_games, unix_ms},
    invalid,
};

const ACTION_SAMPLE_CAP: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HabitatDerivative {
    pub component_count_delta: [i16; 5],
    pub internal_edge_delta: [i16; 5],
    pub open_boundary_delta: [i16; 5],
    pub cycle_rank_delta: [i16; 5],
    pub bridge_delta: [i16; 5],
    pub articulation_delta: [i16; 5],
    pub merge_frontier_delta: [i16; 5],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MotifDerivative {
    pub eligible_empty_cell_delta: [i16; 5],
    pub bear_singleton_delta: i16,
    pub bear_pair_delta: i16,
    pub bear_oversize_delta: i16,
    pub bear_pair_completion_delta: i16,
    pub bear_oversize_risk_delta: i16,
    pub elk_line_delta: [i16; 5],
    pub elk_extension_delta: i16,
    pub elk_overlap_delta: i16,
    pub salmon_valid_run_delta: i16,
    pub salmon_invalid_component_delta: i16,
    pub salmon_endpoint_delta: i16,
    pub salmon_branch_conflict_delta: i16,
    pub salmon_continuation_delta: i16,
    pub hawk_conflict_edge_delta: i16,
    pub hawk_isolated_delta: i16,
    pub hawk_opportunity_delta: i16,
    pub fox_center_delta: i16,
    pub fox_diversity_delta: i16,
    pub fox_missing_type_delta: i16,
    pub fox_compatible_cell_delta: i16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontierDerivative {
    pub frontier_count_delta: i16,
    pub degree_histogram_delta: [i16; 7],
    pub bridge_frontier_delta: [i16; 5],
    pub repeated_contact_delta: [i16; 5],
    pub maximum_resulting_size_delta: [i16; 5],
    pub sum_resulting_size_delta: [i32; 5],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SupplyDerivative {
    pub wildlife_bag_delta: [i16; 5],
    pub selected_tile_archetype_before: u16,
    pub selected_tile_archetype_after: u16,
    pub destination_match_copy_delta: [i64; 7],
    pub frontier_match_copy_delta: [i64; 7],
    pub frontier_matching_edge_mass_delta: i64,
    pub remaining_tile_frontier_copy_mass: u64,
    pub remaining_wildlife_slot_mass: [u64; 5],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarketDerivative {
    pub selected_tile_opponent_access: [u32; 3],
    pub selected_wildlife_opponent_access: [u16; 3],
    pub total_tile_opponent_access_delta: [i64; 3],
    pub total_wildlife_opponent_access_delta: [i64; 3],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpportunityDerivative {
    pub immediate_score_delta: [i16; 12],
    pub habitat: HabitatDerivative,
    pub motif: MotifDerivative,
    pub frontier: FrontierDerivative,
    pub lost_future_placements: [u16; 5],
    pub new_future_placements: [u16; 5],
    pub supply: SupplyDerivative,
    pub market: MarketDerivative,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormalizationTransform {
    Identity,
    RobustDivide,
    SignedLog1pRobustDivide,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureScale {
    pub count: u64,
    pub nonzero_count: u64,
    pub minimum: i64,
    pub maximum: i64,
    pub p99_absolute: u64,
    pub maximum_absolute: u64,
    pub transform: NormalizationTransform,
    pub divisor: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct S5Metrics {
    pub positions: u64,
    pub complete_actions_seen: u64,
    pub sampled_actions: u64,
    pub exact_replay_checks: u64,
    pub exact_replay_failures: u64,
    pub score_delta_checks: u64,
    pub score_delta_failures: u64,
    pub feature_field_count: u64,
    pub feature_scales: BTreeMap<String, FeatureScale>,
    pub raw_p99_scale_ratio_ppm: u64,
    pub normalization_contract_pass: bool,
    pub exact_replay_gate_pass: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct SupplyProfile {
    match_copies: [u64; 7],
    matching_edge_mass: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct OpponentAccess {
    tile: [u64; 3],
    wildlife: [u64; 3],
}

#[derive(Debug, Clone)]
pub struct OpportunityDerivativeContext {
    before_graph: RelationalStateGraph,
    before_supply_profile: SupplyProfile,
    before_market_access: OpponentAccess,
}

impl OpportunityDerivativeContext {
    pub fn from_trunk(trunk: &PublicStateTrunk) -> Result<Self> {
        let before_graph = RelationalStateGraph::from_sparse(&trunk.sparse)?;
        Ok(Self {
            before_supply_profile: frontier_supply_profile(&trunk.sparse, 0, &trunk.supply)?,
            before_market_access: market_access(&trunk.sparse, &before_graph),
            before_graph,
        })
    }

    pub fn derive(
        &self,
        trunk: &PublicStateTrunk,
        edit: &ActionEdit,
        applied: &AppliedPublicState,
    ) -> Result<OpportunityDerivative> {
        let after_sparse = after_sparse_state(trunk, edit, &applied.record)?;
        let after_graph = RelationalStateGraph::from_sparse(&after_sparse)?;
        build_derivative(
            trunk,
            &self.before_graph,
            self.before_supply_profile,
            self.before_market_access,
            edit,
            &after_sparse,
            &after_graph,
            &applied.supply,
        )
    }
}

#[derive(Default)]
struct FeatureAccumulator {
    values: Vec<i64>,
}

impl FeatureAccumulator {
    fn observe(&mut self, value: i64) {
        self.values.push(value);
    }

    fn finish(mut self) -> Result<FeatureScale> {
        if self.values.is_empty() {
            return Err(invalid("S5 feature scale has no observations"));
        }
        self.values.sort_unstable();
        let absolute = self
            .values
            .iter()
            .map(|value| value.unsigned_abs())
            .collect::<Vec<_>>();
        let mut sorted_absolute = absolute.clone();
        sorted_absolute.sort_unstable();
        let p99_absolute = sorted_absolute[(sorted_absolute.len() - 1) * 99 / 100];
        let maximum_absolute = *sorted_absolute.last().expect("nonempty feature");
        let transform = if maximum_absolute == 0 {
            NormalizationTransform::Identity
        } else if p99_absolute > 0 && maximum_absolute > p99_absolute.saturating_mul(16) {
            NormalizationTransform::SignedLog1pRobustDivide
        } else {
            NormalizationTransform::RobustDivide
        };
        Ok(FeatureScale {
            count: self.values.len() as u64,
            nonzero_count: self.values.iter().filter(|value| **value != 0).count() as u64,
            minimum: self.values[0],
            maximum: *self.values.last().expect("nonempty feature"),
            p99_absolute,
            maximum_absolute,
            transform,
            divisor: p99_absolute.max(1),
        })
    }
}

#[derive(Default)]
struct SeedMetrics {
    positions: u64,
    complete_actions_seen: u64,
    sampled_actions: u64,
    exact_replay_checks: u64,
    exact_replay_failures: u64,
    score_delta_checks: u64,
    score_delta_failures: u64,
    features: BTreeMap<String, FeatureAccumulator>,
}

impl SeedMetrics {
    fn merge(&mut self, other: Self) {
        self.positions += other.positions;
        self.complete_actions_seen += other.complete_actions_seen;
        self.sampled_actions += other.sampled_actions;
        self.exact_replay_checks += other.exact_replay_checks;
        self.exact_replay_failures += other.exact_replay_failures;
        self.score_delta_checks += other.score_delta_checks;
        self.score_delta_failures += other.score_delta_failures;
        for (name, mut values) in other.features {
            self.features
                .entry(name)
                .or_default()
                .values
                .append(&mut values.values);
        }
    }

    fn observe_derivative(&mut self, derivative: &OpportunityDerivative) {
        for (name, value) in opportunity_derivative_features(derivative) {
            self.features.entry(name).or_default().observe(value);
        }
    }
}

pub fn run_s5(config: CommonConfig) -> Result<ReportEnvelope<S5Metrics>> {
    if config.lane != ExperimentLane::S5Derivatives {
        return Err(invalid("S5 runner received a non-S5 lane"));
    }
    let started = unix_ms()?;
    let per_seed = run_games(&config, run_seed)?;
    let mut combined = SeedMetrics::default();
    for (_, metrics) in per_seed {
        combined.merge(metrics);
    }
    let mut feature_scales = BTreeMap::new();
    for (name, accumulator) in combined.features {
        feature_scales.insert(name, accumulator.finish()?);
    }
    let mut nonzero_p99 = feature_scales
        .values()
        .filter_map(|scale| (scale.p99_absolute > 0).then_some(scale.p99_absolute))
        .collect::<Vec<_>>();
    nonzero_p99.sort_unstable();
    let raw_p99_scale_ratio_ppm = if nonzero_p99.is_empty() {
        0
    } else {
        nonzero_p99[nonzero_p99.len() - 1]
            .checked_mul(1_000_000)
            .ok_or_else(|| invalid("S5 scale ratio overflowed"))?
            / nonzero_p99[nonzero_p99.len() / 2].max(1)
    };
    let feature_field_count = feature_scales.len() as u64;
    let normalization_contract_pass = feature_field_count == 154
        && feature_scales
            .values()
            .all(|scale| scale.count == combined.sampled_actions && scale.divisor >= 1);
    let exact_replay_gate_pass = combined.exact_replay_failures == 0
        && combined.score_delta_failures == 0
        && combined.exact_replay_checks == combined.sampled_actions
        && combined.score_delta_checks == combined.sampled_actions;
    let passed = normalization_contract_pass && exact_replay_gate_pass;
    let classification = if passed {
        "s5_exact_opportunity_derivatives_promoted"
    } else if !exact_replay_gate_pass {
        "s5_exact_derivative_replay_failed"
    } else {
        "s5_derivative_normalization_contract_failed"
    };
    envelope(
        config,
        S5Metrics {
            positions: combined.positions,
            complete_actions_seen: combined.complete_actions_seen,
            sampled_actions: combined.sampled_actions,
            exact_replay_checks: combined.exact_replay_checks,
            exact_replay_failures: combined.exact_replay_failures,
            score_delta_checks: combined.score_delta_checks,
            score_delta_failures: combined.score_delta_failures,
            feature_field_count,
            feature_scales,
            raw_p99_scale_ratio_ppm,
            normalization_contract_pass,
            exact_replay_gate_pass,
        },
        passed,
        classification,
        started,
    )
}

fn run_seed(seed: u64, mut game: GameState) -> Result<SeedMetrics> {
    let mut metrics = SeedMetrics::default();
    while !game.is_game_over() {
        metrics.positions += 1;
        let game_index = seed
            .checked_mul(100)
            .and_then(|value| value.checked_add(u64::from(game.completed_turns())))
            .ok_or_else(|| invalid("S5 game index overflowed"))?;
        let trunk = PublicStateTrunk::observe(&game, game_index)?;
        let prepared = trunk.prepare_action_edits()?;
        let before_graph = RelationalStateGraph::from_sparse(&trunk.sparse)?;
        let before_supply_profile = frontier_supply_profile(&trunk.sparse, 0, &trunk.supply)?;
        let before_market_access = market_access(&trunk.sparse, &before_graph);
        let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
        let observed = prepared.observe_legal_actions(&game, &prelude)?;
        if observed.is_empty() {
            return Err(invalid("S5 nonterminal position has no legal actions"));
        }
        metrics.complete_actions_seen += observed.len() as u64;
        for index in
            sampled_action_indices(&observed, seed, game.completed_turns(), ACTION_SAMPLE_CAP)?
        {
            let (_, edit) = &observed[index];
            let applied = prepared.apply(edit)?;
            metrics.exact_replay_checks += 1;
            if applied.canonical_record_hash() != edit.expected_public_afterstate_blake3 {
                metrics.exact_replay_failures += 1;
            }
            let after_sparse = after_sparse_state(&trunk, edit, &applied.record)?;
            let after_graph = RelationalStateGraph::from_sparse(&after_sparse)?;
            let derivative = build_derivative(
                &trunk,
                &before_graph,
                before_supply_profile,
                before_market_access,
                edit,
                &after_sparse,
                &after_graph,
                &applied.supply,
            )?;
            metrics.score_delta_checks += 1;
            let expected = [
                edit.score_delta.habitat[0],
                edit.score_delta.habitat[1],
                edit.score_delta.habitat[2],
                edit.score_delta.habitat[3],
                edit.score_delta.habitat[4],
                edit.score_delta.wildlife[0],
                edit.score_delta.wildlife[1],
                edit.score_delta.wildlife[2],
                edit.score_delta.wildlife[3],
                edit.score_delta.wildlife[4],
                edit.score_delta.nature_tokens,
                edit.score_delta.base_total,
            ];
            if derivative.immediate_score_delta != expected {
                metrics.score_delta_failures += 1;
            }
            metrics.sampled_actions += 1;
            metrics.observe_derivative(&derivative);
        }
        let selected = observed
            [deterministic_index(seed, game.completed_turns(), observed.len(), b"s5-advance")]
        .0
        .clone();
        game.apply(&selected)?;
    }
    if metrics.positions != 80 {
        return Err(invalid(format!(
            "S5 seed {seed} produced {} positions instead of 80",
            metrics.positions
        )));
    }
    Ok(metrics)
}

#[allow(clippy::too_many_arguments)]
fn build_derivative(
    trunk: &PublicStateTrunk,
    before_graph: &RelationalStateGraph,
    before_supply_profile: SupplyProfile,
    before_market_access: OpponentAccess,
    edit: &ActionEdit,
    after_sparse: &SparsePublicState,
    after_graph: &RelationalStateGraph,
    after_supply: &SupplySnapshot,
) -> Result<OpportunityDerivative> {
    let before_board = &before_graph.boards[0];
    let after_board = &after_graph.boards[0];
    let before_score = before_board.score_anatomy();
    let immediate_score_delta = after_board.score_anatomy().delta(before_score);
    let habitat = habitat_derivative(before_board, after_board);
    let motif = motif_derivative(&before_board.opportunity, &after_board.opportunity);
    let frontier = frontier_derivative(before_board, after_board);
    let lost_future_placements = std::array::from_fn(|index| {
        before_board.opportunity.eligible_empty_cells[index]
            .saturating_sub(after_board.opportunity.eligible_empty_cells[index])
    });
    let new_future_placements = std::array::from_fn(|index| {
        after_board.opportunity.eligible_empty_cells[index]
            .saturating_sub(before_board.opportunity.eligible_empty_cells[index])
    });
    let after_supply_profile = frontier_supply_profile(after_sparse, 0, after_supply)?;
    let before_destination_profile =
        destination_supply_profile(&trunk.sparse, edit.factors.tile_destination, &trunk.supply)?;
    let after_destination_profile =
        destination_supply_profile(after_sparse, edit.factors.tile_destination, after_supply)?;
    let archetype = edit
        .selected
        .tile
        .semantic_archetype_id
        .ok_or_else(|| invalid("S5 selected tile lacks a semantic archetype ID"))?;
    let selected_tile_archetype_before = trunk
        .supply
        .archetype_counts
        .get(usize::from(archetype))
        .copied()
        .ok_or_else(|| invalid("S5 selected archetype is outside supply"))?;
    let selected_tile_archetype_after = after_supply
        .archetype_counts
        .get(usize::from(archetype))
        .copied()
        .ok_or_else(|| invalid("S5 selected archetype is outside after supply"))?;
    let supply = SupplyDerivative {
        wildlife_bag_delta: std::array::from_fn(|index| {
            after_supply.wildlife_bag[index] as i16 - trunk.supply.wildlife_bag[index] as i16
        }),
        selected_tile_archetype_before,
        selected_tile_archetype_after,
        destination_match_copy_delta: signed_array_delta(
            before_destination_profile.match_copies,
            after_destination_profile.match_copies,
        ),
        frontier_match_copy_delta: signed_array_delta(
            before_supply_profile.match_copies,
            after_supply_profile.match_copies,
        ),
        frontier_matching_edge_mass_delta: after_supply_profile.matching_edge_mass as i64
            - before_supply_profile.matching_edge_mass as i64,
        remaining_tile_frontier_copy_mass: after_supply_profile.match_copies.iter().sum(),
        remaining_wildlife_slot_mass: std::array::from_fn(|index| {
            u64::from(after_supply.wildlife_bag[index])
                * u64::from(after_board.opportunity.eligible_empty_cells[index])
        }),
    };
    let after_market_access = market_access(after_sparse, after_graph);
    let (selected_tile_opponent_access, selected_wildlife_opponent_access) =
        selected_opponent_access(
            &trunk.sparse,
            before_graph,
            &edit.selected.tile,
            edit.selected.wildlife,
        );
    let market = MarketDerivative {
        selected_tile_opponent_access,
        selected_wildlife_opponent_access,
        total_tile_opponent_access_delta: std::array::from_fn(|index| {
            after_market_access.tile[index] as i64 - before_market_access.tile[index] as i64
        }),
        total_wildlife_opponent_access_delta: std::array::from_fn(|index| {
            after_market_access.wildlife[index] as i64 - before_market_access.wildlife[index] as i64
        }),
    };
    Ok(OpportunityDerivative {
        immediate_score_delta,
        habitat,
        motif,
        frontier,
        lost_future_placements,
        new_future_placements,
        supply,
        market,
    })
}

fn after_sparse_state(
    parent: &PublicStateTrunk,
    edit: &ActionEdit,
    after_record: &cascadia_data::PositionRecord,
) -> Result<SparsePublicState> {
    let mut geometry_record = after_record.clone();
    geometry_record.market_entities = parent.public_record()?.market_entities;
    let mut sparse = SparsePublicState::from_position_record(&geometry_record, None)?;
    sparse.market = edit
        .placement
        .market_after
        .slots
        .iter()
        .map(|slot| r2_sparse_entity_census::MarketToken {
            slot: slot.slot,
            tile: slot.tile.as_ref().map(|tile| SuppliedTile {
                terrain_a: tile.terrain_a,
                terrain_b: tile.terrain_b,
                wildlife_eligibility: tile.wildlife_eligibility,
                keystone: tile.keystone,
            }),
            wildlife: slot.wildlife,
        })
        .collect();
    Ok(sparse)
}

fn habitat_derivative(before: &BoardGraph, after: &BoardGraph) -> HabitatDerivative {
    let before_summary = habitat_summary(before);
    let after_summary = habitat_summary(after);
    HabitatDerivative {
        component_count_delta: signed_array_delta(before_summary.0, after_summary.0),
        internal_edge_delta: signed_array_delta(before_summary.1, after_summary.1),
        open_boundary_delta: signed_array_delta(before_summary.2, after_summary.2),
        cycle_rank_delta: signed_array_delta(before_summary.3, after_summary.3),
        bridge_delta: signed_array_delta(before_summary.4, after_summary.4),
        articulation_delta: signed_array_delta(before_summary.5, after_summary.5),
        merge_frontier_delta: signed_array_delta(before_summary.6, after_summary.6),
    }
}

type HabitatSummary = (
    [u16; 5],
    [u16; 5],
    [u16; 5],
    [u16; 5],
    [u16; 5],
    [u16; 5],
    [u16; 5],
);

fn habitat_summary(board: &BoardGraph) -> HabitatSummary {
    let mut result = ([0; 5], [0; 5], [0; 5], [0; 5], [0; 5], [0; 5], [0; 5]);
    for component in &board.habitat_components {
        let index = component.terrain as usize;
        result.0[index] += 1;
        result.1[index] += component.matching_internal_edge_count;
        result.2[index] += component.open_boundary_edge_count;
        result.3[index] += component.cycle_rank;
        result.4[index] += component.bridge_count;
        result.5[index] += component.articulation_count;
        result.6[index] += component.merge_frontier_count;
    }
    result
}

fn motif_derivative(
    before: &WildlifeOpportunitySummary,
    after: &WildlifeOpportunitySummary,
) -> MotifDerivative {
    MotifDerivative {
        eligible_empty_cell_delta: signed_array_delta(
            before.eligible_empty_cells,
            after.eligible_empty_cells,
        ),
        bear_singleton_delta: signed(after.bear_singletons, before.bear_singletons),
        bear_pair_delta: signed(after.bear_pairs, before.bear_pairs),
        bear_oversize_delta: signed(
            after.bear_oversize_components,
            before.bear_oversize_components,
        ),
        bear_pair_completion_delta: signed(
            after.bear_pair_completion_cells,
            before.bear_pair_completion_cells,
        ),
        bear_oversize_risk_delta: signed(
            after.bear_oversize_risk_cells,
            before.bear_oversize_risk_cells,
        ),
        elk_line_delta: signed_array_delta(before.elk_lines_by_length, after.elk_lines_by_length),
        elk_extension_delta: signed(
            after.elk_eligible_extensions,
            before.elk_eligible_extensions,
        ),
        elk_overlap_delta: signed(
            after.elk_overlapping_members,
            before.elk_overlapping_members,
        ),
        salmon_valid_run_delta: signed(after.salmon_valid_runs, before.salmon_valid_runs),
        salmon_invalid_component_delta: signed(
            after.salmon_invalid_components,
            before.salmon_invalid_components,
        ),
        salmon_endpoint_delta: signed(after.salmon_endpoints, before.salmon_endpoints),
        salmon_branch_conflict_delta: signed(
            after.salmon_branch_conflicts,
            before.salmon_branch_conflicts,
        ),
        salmon_continuation_delta: signed(
            after.salmon_legal_continuations,
            before.salmon_legal_continuations,
        ),
        hawk_conflict_edge_delta: signed(after.hawk_conflict_edges, before.hawk_conflict_edges),
        hawk_isolated_delta: signed(after.hawk_isolated, before.hawk_isolated),
        hawk_opportunity_delta: signed(
            after.hawk_isolated_opportunities,
            before.hawk_isolated_opportunities,
        ),
        fox_center_delta: signed(after.fox_centers, before.fox_centers),
        fox_diversity_delta: signed(after.fox_diversity_sum, before.fox_diversity_sum),
        fox_missing_type_delta: signed(after.fox_missing_types, before.fox_missing_types),
        fox_compatible_cell_delta: signed(after.fox_compatible_cells, before.fox_compatible_cells),
    }
}

fn frontier_derivative(before: &BoardGraph, after: &BoardGraph) -> FrontierDerivative {
    FrontierDerivative {
        frontier_count_delta: signed(
            after.frontier.frontier_count,
            before.frontier.frontier_count,
        ),
        degree_histogram_delta: signed_array_delta(
            before.frontier.degree_histogram,
            after.frontier.degree_histogram,
        ),
        bridge_frontier_delta: signed_array_delta(
            before.frontier.bridge_frontiers_by_terrain,
            after.frontier.bridge_frontiers_by_terrain,
        ),
        repeated_contact_delta: signed_array_delta(
            before.frontier.repeated_contact_frontiers_by_terrain,
            after.frontier.repeated_contact_frontiers_by_terrain,
        ),
        maximum_resulting_size_delta: signed_array_delta(
            before.frontier.maximum_resulting_size_by_terrain,
            after.frontier.maximum_resulting_size_by_terrain,
        ),
        sum_resulting_size_delta: std::array::from_fn(|index| {
            after.frontier.sum_resulting_size_by_terrain[index] as i32
                - before.frontier.sum_resulting_size_by_terrain[index] as i32
        }),
    }
}

fn frontier_supply_profile(
    sparse: &SparsePublicState,
    relative_seat: u8,
    supply: &SupplySnapshot,
) -> Result<SupplyProfile> {
    let frontiers = sparse
        .legal_frontier
        .iter()
        .filter(|frontier| frontier.relative_seat == relative_seat)
        .collect::<Vec<_>>();
    supply_profile_for_frontiers(&frontiers, supply)
}

fn destination_supply_profile(
    sparse: &SparsePublicState,
    destination: r3_action_edit_census::AxialCoord,
    supply: &SupplySnapshot,
) -> Result<SupplyProfile> {
    let coord = AxialCoord::new(destination.q, destination.r);
    let frontiers = sparse
        .legal_frontier
        .iter()
        .filter(|frontier| frontier.relative_seat == 0 && frontier.coord == coord)
        .collect::<Vec<_>>();
    supply_profile_for_frontiers(&frontiers, supply)
}

fn supply_profile_for_frontiers(
    frontiers: &[&FrontierToken],
    supply: &SupplySnapshot,
) -> Result<SupplyProfile> {
    let catalog = standard_semantic_archetype_catalog();
    if supply.archetype_counts.len() != catalog.len() {
        return Err(invalid("S5 supply/catalog length mismatch"));
    }
    let mut result = SupplyProfile::default();
    for frontier in frontiers {
        for (definition, count) in catalog.definitions().iter().zip(&supply.archetype_counts) {
            if *count == 0 {
                continue;
            }
            let best = best_matching_edges(definition.archetype, frontier);
            result.match_copies[usize::from(best)] += u64::from(*count);
            result.matching_edge_mass += u64::from(*count) * u64::from(best);
        }
    }
    Ok(result)
}

fn best_matching_edges(archetype: CanonicalTileArchetype, frontier: &FrontierToken) -> u8 {
    let rotations = if archetype.secondary_terrain.is_some() {
        &Rotation::ALL[..]
    } else {
        &Rotation::ALL[..1]
    };
    rotations
        .iter()
        .map(|rotation| {
            (0..6)
                .filter(|edge| {
                    frontier.neighbor_facing_terrains[*edge].is_some_and(|terrain| {
                        archetype.terrain_on_edge(*rotation, *edge) == terrain
                    })
                })
                .count() as u8
        })
        .max()
        .unwrap_or(0)
}

fn market_access(sparse: &SparsePublicState, graph: &RelationalStateGraph) -> OpponentAccess {
    let mut result = OpponentAccess::default();
    for opponent in 0..3 {
        let relative_seat = opponent + 1;
        let frontiers = sparse
            .legal_frontier
            .iter()
            .filter(|frontier| usize::from(frontier.relative_seat) == relative_seat)
            .collect::<Vec<_>>();
        let board = &graph.boards[relative_seat];
        for market in &sparse.market {
            if let Some(tile) = market.tile {
                result.tile[opponent] += u64::from(tile_positive_placements(tile, &frontiers));
            }
            if let Some(wildlife) = market.wildlife {
                result.wildlife[opponent] +=
                    u64::from(board.opportunity.eligible_empty_cells[wildlife as usize]);
            }
        }
    }
    result
}

fn selected_opponent_access(
    sparse: &SparsePublicState,
    graph: &RelationalStateGraph,
    tile: &TileSemantic,
    wildlife: Wildlife,
) -> ([u32; 3], [u16; 3]) {
    let supplied = SuppliedTile {
        terrain_a: tile.terrain_a,
        terrain_b: tile.terrain_b,
        wildlife_eligibility: tile.wildlife_eligibility,
        keystone: tile.keystone,
    };
    (
        std::array::from_fn(|opponent| {
            let relative_seat = opponent + 1;
            let frontiers = sparse
                .legal_frontier
                .iter()
                .filter(|frontier| usize::from(frontier.relative_seat) == relative_seat)
                .collect::<Vec<_>>();
            tile_positive_placements(supplied, &frontiers)
        }),
        std::array::from_fn(|opponent| {
            graph.boards[opponent + 1].opportunity.eligible_empty_cells[wildlife as usize]
        }),
    )
}

fn tile_positive_placements(tile: SuppliedTile, frontiers: &[&FrontierToken]) -> u32 {
    let archetype = CanonicalTileArchetype {
        primary_terrain: tile.terrain_a,
        secondary_terrain: tile.terrain_b,
        directed_edges: tile.directed_edges(Rotation::ZERO),
        wildlife: tile.wildlife_eligibility,
        keystone: tile.keystone,
    };
    let rotations = if tile.terrain_b.is_some() {
        &Rotation::ALL[..]
    } else {
        &Rotation::ALL[..1]
    };
    frontiers
        .iter()
        .map(|frontier| {
            rotations
                .iter()
                .filter(|rotation| {
                    (0..6).any(|edge| {
                        frontier.neighbor_facing_terrains[edge].is_some_and(|terrain| {
                            archetype.terrain_on_edge(**rotation, edge) == terrain
                        })
                    })
                })
                .count() as u32
        })
        .sum()
}

fn sampled_action_indices(
    observed: &[(cascadia_game::TurnAction, ActionEdit)],
    seed: u64,
    turn: u16,
    cap: usize,
) -> Result<Vec<usize>> {
    if observed.len() <= cap {
        return Ok((0..observed.len()).collect());
    }
    let mut ranked = observed
        .iter()
        .enumerate()
        .map(|(index, (action, _))| {
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"cascadia-s5-action-sample-v1");
            hasher.update(&seed.to_le_bytes());
            hasher.update(&turn.to_le_bytes());
            hasher.update(&postcard::to_allocvec(action)?);
            Ok((*hasher.finalize().as_bytes(), index))
        })
        .collect::<Result<Vec<_>>>()?;
    ranked.sort_unstable();
    let mut selected = ranked
        .into_iter()
        .take(cap)
        .map(|(_, index)| index)
        .collect::<Vec<_>>();
    selected.sort_unstable();
    Ok(selected)
}

fn signed(after: u16, before: u16) -> i16 {
    after as i16 - before as i16
}

fn signed_array_delta<const N: usize, T, U>(before: [T; N], after: [T; N]) -> [U; N]
where
    T: Copy + TryInto<i64>,
    <T as TryInto<i64>>::Error: std::fmt::Debug,
    U: TryFrom<i64> + Copy,
    <U as TryFrom<i64>>::Error: std::fmt::Debug,
{
    std::array::from_fn(|index| {
        U::try_from(
            after[index]
                .try_into()
                .expect("bounded Cascadia value fits i64")
                - before[index]
                    .try_into()
                    .expect("bounded Cascadia value fits i64"),
        )
        .expect("bounded Cascadia derivative fits target integer")
    })
}

pub fn opportunity_derivative_features(derivative: &OpportunityDerivative) -> Vec<(String, i64)> {
    let output = collect_opportunity_derivative(derivative, FeatureOutput::Named(Vec::new()));
    match output {
        FeatureOutput::Named(values) => values,
        FeatureOutput::Values(_) => unreachable!(),
    }
}

pub fn opportunity_derivative_values(derivative: &OpportunityDerivative) -> Vec<i64> {
    let output = collect_opportunity_derivative(derivative, FeatureOutput::Values(Vec::new()));
    match output {
        FeatureOutput::Values(values) => values,
        FeatureOutput::Named(_) => unreachable!(),
    }
}

fn collect_opportunity_derivative(
    derivative: &OpportunityDerivative,
    mut values: FeatureOutput,
) -> FeatureOutput {
    push_array(&mut values, "score", &derivative.immediate_score_delta);
    push_array(
        &mut values,
        "habitat.component_count",
        &derivative.habitat.component_count_delta,
    );
    push_array(
        &mut values,
        "habitat.internal_edge",
        &derivative.habitat.internal_edge_delta,
    );
    push_array(
        &mut values,
        "habitat.open_boundary",
        &derivative.habitat.open_boundary_delta,
    );
    push_array(
        &mut values,
        "habitat.cycle_rank",
        &derivative.habitat.cycle_rank_delta,
    );
    push_array(
        &mut values,
        "habitat.bridge",
        &derivative.habitat.bridge_delta,
    );
    push_array(
        &mut values,
        "habitat.articulation",
        &derivative.habitat.articulation_delta,
    );
    push_array(
        &mut values,
        "habitat.merge_frontier",
        &derivative.habitat.merge_frontier_delta,
    );
    push_array(
        &mut values,
        "motif.eligible_empty",
        &derivative.motif.eligible_empty_cell_delta,
    );
    push_scalar(
        &mut values,
        "motif.bear_singleton",
        derivative.motif.bear_singleton_delta,
    );
    push_scalar(
        &mut values,
        "motif.bear_pair",
        derivative.motif.bear_pair_delta,
    );
    push_scalar(
        &mut values,
        "motif.bear_oversize",
        derivative.motif.bear_oversize_delta,
    );
    push_scalar(
        &mut values,
        "motif.bear_pair_completion",
        derivative.motif.bear_pair_completion_delta,
    );
    push_scalar(
        &mut values,
        "motif.bear_oversize_risk",
        derivative.motif.bear_oversize_risk_delta,
    );
    push_array(
        &mut values,
        "motif.elk_line",
        &derivative.motif.elk_line_delta,
    );
    push_scalar(
        &mut values,
        "motif.elk_extension",
        derivative.motif.elk_extension_delta,
    );
    push_scalar(
        &mut values,
        "motif.elk_overlap",
        derivative.motif.elk_overlap_delta,
    );
    push_scalar(
        &mut values,
        "motif.salmon_valid",
        derivative.motif.salmon_valid_run_delta,
    );
    push_scalar(
        &mut values,
        "motif.salmon_invalid",
        derivative.motif.salmon_invalid_component_delta,
    );
    push_scalar(
        &mut values,
        "motif.salmon_endpoint",
        derivative.motif.salmon_endpoint_delta,
    );
    push_scalar(
        &mut values,
        "motif.salmon_branch",
        derivative.motif.salmon_branch_conflict_delta,
    );
    push_scalar(
        &mut values,
        "motif.salmon_continuation",
        derivative.motif.salmon_continuation_delta,
    );
    push_scalar(
        &mut values,
        "motif.hawk_conflict",
        derivative.motif.hawk_conflict_edge_delta,
    );
    push_scalar(
        &mut values,
        "motif.hawk_isolated",
        derivative.motif.hawk_isolated_delta,
    );
    push_scalar(
        &mut values,
        "motif.hawk_opportunity",
        derivative.motif.hawk_opportunity_delta,
    );
    push_scalar(
        &mut values,
        "motif.fox_center",
        derivative.motif.fox_center_delta,
    );
    push_scalar(
        &mut values,
        "motif.fox_diversity",
        derivative.motif.fox_diversity_delta,
    );
    push_scalar(
        &mut values,
        "motif.fox_missing",
        derivative.motif.fox_missing_type_delta,
    );
    push_scalar(
        &mut values,
        "motif.fox_compatible",
        derivative.motif.fox_compatible_cell_delta,
    );
    push_scalar(
        &mut values,
        "frontier.count",
        derivative.frontier.frontier_count_delta,
    );
    push_array(
        &mut values,
        "frontier.degree",
        &derivative.frontier.degree_histogram_delta,
    );
    push_array(
        &mut values,
        "frontier.bridge",
        &derivative.frontier.bridge_frontier_delta,
    );
    push_array(
        &mut values,
        "frontier.repeated",
        &derivative.frontier.repeated_contact_delta,
    );
    push_array(
        &mut values,
        "frontier.max_result",
        &derivative.frontier.maximum_resulting_size_delta,
    );
    push_array(
        &mut values,
        "frontier.sum_result",
        &derivative.frontier.sum_resulting_size_delta,
    );
    push_array(
        &mut values,
        "future.lost",
        &derivative.lost_future_placements,
    );
    push_array(&mut values, "future.new", &derivative.new_future_placements);
    push_array(
        &mut values,
        "supply.wildlife_bag",
        &derivative.supply.wildlife_bag_delta,
    );
    push_scalar(
        &mut values,
        "supply.selected_tile_before",
        derivative.supply.selected_tile_archetype_before,
    );
    push_scalar(
        &mut values,
        "supply.selected_tile_after",
        derivative.supply.selected_tile_archetype_after,
    );
    push_array(
        &mut values,
        "supply.destination_match",
        &derivative.supply.destination_match_copy_delta,
    );
    push_array(
        &mut values,
        "supply.frontier_match",
        &derivative.supply.frontier_match_copy_delta,
    );
    push_scalar(
        &mut values,
        "supply.frontier_edge_mass",
        derivative.supply.frontier_matching_edge_mass_delta,
    );
    push_scalar(
        &mut values,
        "supply.remaining_tile_frontier_mass",
        derivative.supply.remaining_tile_frontier_copy_mass,
    );
    push_array(
        &mut values,
        "supply.remaining_wildlife_slot_mass",
        &derivative.supply.remaining_wildlife_slot_mass,
    );
    push_array(
        &mut values,
        "market.selected_tile_access",
        &derivative.market.selected_tile_opponent_access,
    );
    push_array(
        &mut values,
        "market.selected_wildlife_access",
        &derivative.market.selected_wildlife_opponent_access,
    );
    push_array(
        &mut values,
        "market.total_tile_access_delta",
        &derivative.market.total_tile_opponent_access_delta,
    );
    push_array(
        &mut values,
        "market.total_wildlife_access_delta",
        &derivative.market.total_wildlife_opponent_access_delta,
    );
    values
}

enum FeatureOutput {
    Named(Vec<(String, i64)>),
    Values(Vec<i64>),
}

fn push_scalar<T>(values: &mut FeatureOutput, name: &str, value: T)
where
    T: Copy + TryInto<i64>,
    <T as TryInto<i64>>::Error: std::fmt::Debug,
{
    let value = value.try_into().expect("bounded Cascadia feature fits i64");
    match values {
        FeatureOutput::Named(values) => values.push((name.to_owned(), value)),
        FeatureOutput::Values(values) => values.push(value),
    }
}

fn push_array<T, const N: usize>(values: &mut FeatureOutput, prefix: &str, array: &[T; N])
where
    T: Copy + TryInto<i64>,
    <T as TryInto<i64>>::Error: std::fmt::Debug,
{
    for (index, value) in array.iter().enumerate() {
        let value = (*value)
            .try_into()
            .expect("bounded Cascadia feature fits i64");
        match values {
            FeatureOutput::Named(values) => values.push((format!("{prefix}.{index}"), value)),
            FeatureOutput::Values(values) => values.push(value),
        }
    }
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameConfig, GameSeed, GameState};

    use super::*;

    #[test]
    fn derivative_matches_authoritative_score_delta() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(5),
        )
        .unwrap();
        let trunk = PublicStateTrunk::observe(&game, 5).unwrap();
        let prepared = trunk.prepare_action_edits().unwrap();
        let before_graph = RelationalStateGraph::from_sparse(&trunk.sparse).unwrap();
        let before_supply_profile =
            frontier_supply_profile(&trunk.sparse, 0, &trunk.supply).unwrap();
        let before_market_access = market_access(&trunk.sparse, &before_graph);
        let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible().unwrap();
        let observed = prepared.observe_legal_actions(&game, &prelude).unwrap();
        let edit = &observed[observed.len() / 2].1;
        let applied = prepared.apply(edit).unwrap();
        let after_sparse = after_sparse_state(&trunk, edit, &applied.record).unwrap();
        let after_graph = RelationalStateGraph::from_sparse(&after_sparse).unwrap();
        let derivative = build_derivative(
            &trunk,
            &before_graph,
            before_supply_profile,
            before_market_access,
            edit,
            &after_sparse,
            &after_graph,
            &applied.supply,
        )
        .unwrap();
        assert_eq!(
            derivative.immediate_score_delta[11],
            edit.score_delta.base_total
        );
        assert_eq!(opportunity_derivative_features(&derivative).len(), 154);
    }

    #[test]
    fn public_context_matches_direct_derivative_construction() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(17),
        )
        .unwrap();
        let trunk = PublicStateTrunk::observe(&game, 17).unwrap();
        let prepared = trunk.prepare_action_edits().unwrap();
        let context = OpportunityDerivativeContext::from_trunk(&trunk).unwrap();
        let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible().unwrap();
        let (_, edit) = prepared
            .observe_legal_actions(&game, &prelude)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let applied = prepared.apply(&edit).unwrap();
        let derivative = context.derive(&trunk, &edit, &applied).unwrap();
        assert_eq!(opportunity_derivative_features(&derivative).len(), 154);
        assert_eq!(
            derivative.immediate_score_delta[11],
            edit.score_delta.base_total,
        );
    }

    #[test]
    fn stable_sampler_is_bounded_and_sorted() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(6),
        )
        .unwrap();
        let trunk = PublicStateTrunk::observe(&game, 6).unwrap();
        let prepared = trunk.prepare_action_edits().unwrap();
        let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible().unwrap();
        let observed = prepared.observe_legal_actions(&game, &prelude).unwrap();
        let indices = sampled_action_indices(&observed, 6, 0, 16).unwrap();
        assert!(indices.len() <= 16);
        assert!(indices.windows(2).all(|pair| pair[0] < pair[1]));
    }
}
