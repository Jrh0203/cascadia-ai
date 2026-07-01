use cascadia_game::{D6Transform, GameState, score_board};
use r2_sparse_entity_census::SparsePublicState;
use r3_action_edit_census::PublicStateTrunk;
use serde::{Deserialize, Serialize};

use crate::{
    BoardGraph, CommonConfig, DistributionSummary, ExperimentLane, RelationalStateGraph,
    ReportEnvelope, Result,
    common::{deterministic_index, envelope, run_games, unix_ms},
    invalid,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct S3Metrics {
    pub positions: u64,
    pub board_score_decoder_checks: u64,
    pub board_score_decoder_failures: u64,
    pub action_delta_decoder_checks: u64,
    pub action_delta_decoder_failures: u64,
    pub d6_invariance_checks: u64,
    pub d6_invariance_failures: u64,
    pub raw_control_bytes: DistributionSummary,
    pub component_only_bytes: DistributionSummary,
    pub motif_only_bytes: DistributionSummary,
    pub combined_bytes: DistributionSummary,
    pub combined_frontier_bytes: DistributionSummary,
    pub component_tokens: DistributionSummary,
    pub motif_tokens: DistributionSummary,
    pub frontier_tokens: DistributionSummary,
    pub boards_with_elk_extensions: u64,
    pub boards_with_salmon_continuations: u64,
    pub boards_with_hawk_opportunities: u64,
    pub boards_with_bear_pair_opportunities: u64,
    pub semantic_decoder_accuracy_ppm: u64,
    pub semantic_decoder_gate_pass: bool,
    pub opportunity_coverage_gate_pass: bool,
    pub d6_gate_pass: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct ComponentOnlyView<'a> {
    boards: Vec<&'a [crate::HabitatComponentGraph]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct MotifOnlyBoardView<'a> {
    bear: &'a [crate::graph::WildlifeComponentGraph],
    elk: &'a [crate::graph::ElkLineGraph],
    salmon: &'a [crate::graph::SalmonComponentGraph],
    hawk_positions: &'a [r2_sparse_entity_census::AxialCoord],
    hawk_edges: &'a [[r2_sparse_entity_census::AxialCoord; 2]],
    fox: &'a [crate::graph::FoxCenterGraph],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct CombinedBoardView<'a> {
    components: &'a [crate::HabitatComponentGraph],
    motifs: MotifOnlyBoardView<'a>,
    nature_tokens: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct CombinedView<'a> {
    boards: Vec<CombinedBoardView<'a>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct MotifOnlyView<'a> {
    boards: Vec<MotifOnlyBoardView<'a>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct InvariantBoardSignature {
    habitat: Vec<HabitatInvariant>,
    bear: Vec<(usize, u16, u16, u8)>,
    elk: Vec<(usize, u8)>,
    salmon: Vec<(usize, u16, u16, u16, bool, usize)>,
    hawk_positions: usize,
    hawk_edges: usize,
    fox: Vec<(u32, u32, usize)>,
    opportunity: crate::WildlifeOpportunitySummary,
    frontier: crate::graph::FrontierGraphSummary,
    score: crate::CardAScoreAnatomy,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
struct HabitatInvariant {
    terrain: u8,
    members: u16,
    internal_edges: u16,
    open_boundary: u16,
    frontier_contacts: u16,
    cycle_rank: u16,
    bridges: u16,
    articulations: u16,
    merge_frontiers: u16,
}

#[derive(Default)]
struct SeedMetrics {
    positions: u64,
    board_score_decoder_checks: u64,
    board_score_decoder_failures: u64,
    action_delta_decoder_checks: u64,
    action_delta_decoder_failures: u64,
    d6_invariance_checks: u64,
    d6_invariance_failures: u64,
    raw_control_bytes: Vec<u64>,
    component_only_bytes: Vec<u64>,
    motif_only_bytes: Vec<u64>,
    combined_bytes: Vec<u64>,
    combined_frontier_bytes: Vec<u64>,
    component_tokens: Vec<u64>,
    motif_tokens: Vec<u64>,
    frontier_tokens: Vec<u64>,
    boards_with_elk_extensions: u64,
    boards_with_salmon_continuations: u64,
    boards_with_hawk_opportunities: u64,
    boards_with_bear_pair_opportunities: u64,
}

impl SeedMetrics {
    fn merge(&mut self, other: Self) {
        self.positions += other.positions;
        self.board_score_decoder_checks += other.board_score_decoder_checks;
        self.board_score_decoder_failures += other.board_score_decoder_failures;
        self.action_delta_decoder_checks += other.action_delta_decoder_checks;
        self.action_delta_decoder_failures += other.action_delta_decoder_failures;
        self.d6_invariance_checks += other.d6_invariance_checks;
        self.d6_invariance_failures += other.d6_invariance_failures;
        self.raw_control_bytes.extend(other.raw_control_bytes);
        self.component_only_bytes.extend(other.component_only_bytes);
        self.motif_only_bytes.extend(other.motif_only_bytes);
        self.combined_bytes.extend(other.combined_bytes);
        self.combined_frontier_bytes
            .extend(other.combined_frontier_bytes);
        self.component_tokens.extend(other.component_tokens);
        self.motif_tokens.extend(other.motif_tokens);
        self.frontier_tokens.extend(other.frontier_tokens);
        self.boards_with_elk_extensions += other.boards_with_elk_extensions;
        self.boards_with_salmon_continuations += other.boards_with_salmon_continuations;
        self.boards_with_hawk_opportunities += other.boards_with_hawk_opportunities;
        self.boards_with_bear_pair_opportunities += other.boards_with_bear_pair_opportunities;
    }
}

pub fn run_s3(config: CommonConfig) -> Result<ReportEnvelope<S3Metrics>> {
    if config.lane != ExperimentLane::S3ComponentMotif {
        return Err(invalid("S3 runner received a non-S3 lane"));
    }
    let started = unix_ms()?;
    let per_seed = run_games(&config, run_seed)?;
    let mut combined = SeedMetrics::default();
    for (_, metrics) in per_seed {
        combined.merge(metrics);
    }
    let semantic_checks = combined
        .board_score_decoder_checks
        .checked_add(combined.action_delta_decoder_checks)
        .ok_or_else(|| invalid("S3 semantic check count overflowed"))?;
    let semantic_failures = combined
        .board_score_decoder_failures
        .checked_add(combined.action_delta_decoder_failures)
        .ok_or_else(|| invalid("S3 semantic failure count overflowed"))?;
    let semantic_decoder_accuracy_ppm = semantic_checks
        .saturating_sub(semantic_failures)
        .checked_mul(1_000_000)
        .ok_or_else(|| invalid("S3 semantic accuracy overflowed"))?
        / semantic_checks.max(1);
    let semantic_decoder_gate_pass = semantic_decoder_accuracy_ppm >= 990_000;
    let opportunity_coverage_gate_pass = combined.boards_with_elk_extensions > 0
        && combined.boards_with_salmon_continuations > 0
        && combined.boards_with_hawk_opportunities > 0
        && combined.boards_with_bear_pair_opportunities > 0;
    let d6_gate_pass = combined.d6_invariance_failures == 0
        && combined.d6_invariance_checks == combined.positions * 12;
    let passed = semantic_decoder_gate_pass && opportunity_coverage_gate_pass && d6_gate_pass;
    let classification = if passed {
        "s3_exact_component_motif_graph_promoted"
    } else if !semantic_decoder_gate_pass {
        "s3_semantic_decoder_failed"
    } else if !opportunity_coverage_gate_pass {
        "s3_opportunity_coverage_failed"
    } else {
        "s3_d6_invariance_failed"
    };
    envelope(
        config,
        S3Metrics {
            positions: combined.positions,
            board_score_decoder_checks: combined.board_score_decoder_checks,
            board_score_decoder_failures: combined.board_score_decoder_failures,
            action_delta_decoder_checks: combined.action_delta_decoder_checks,
            action_delta_decoder_failures: combined.action_delta_decoder_failures,
            d6_invariance_checks: combined.d6_invariance_checks,
            d6_invariance_failures: combined.d6_invariance_failures,
            raw_control_bytes: DistributionSummary::from_values(combined.raw_control_bytes)?,
            component_only_bytes: DistributionSummary::from_values(combined.component_only_bytes)?,
            motif_only_bytes: DistributionSummary::from_values(combined.motif_only_bytes)?,
            combined_bytes: DistributionSummary::from_values(combined.combined_bytes)?,
            combined_frontier_bytes: DistributionSummary::from_values(
                combined.combined_frontier_bytes,
            )?,
            component_tokens: DistributionSummary::from_values(combined.component_tokens)?,
            motif_tokens: DistributionSummary::from_values(combined.motif_tokens)?,
            frontier_tokens: DistributionSummary::from_values(combined.frontier_tokens)?,
            boards_with_elk_extensions: combined.boards_with_elk_extensions,
            boards_with_salmon_continuations: combined.boards_with_salmon_continuations,
            boards_with_hawk_opportunities: combined.boards_with_hawk_opportunities,
            boards_with_bear_pair_opportunities: combined.boards_with_bear_pair_opportunities,
            semantic_decoder_accuracy_ppm,
            semantic_decoder_gate_pass,
            opportunity_coverage_gate_pass,
            d6_gate_pass,
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
            .ok_or_else(|| invalid("S3 game index overflowed"))?;
        let trunk = PublicStateTrunk::observe(&game, game_index)?;
        let prepared = trunk.prepare_action_edits()?;
        let graph = RelationalStateGraph::from_sparse(&trunk.sparse)?;
        validate_scores(&game, &graph, &mut metrics);
        observe_coverage(&graph, &mut metrics);
        observe_sizes(&trunk, &graph, &mut metrics)?;
        validate_d6(&trunk.sparse, &graph, &mut metrics)?;

        let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
        let observed = prepared.observe_legal_actions(&game, &prelude)?;
        if observed.is_empty() {
            return Err(invalid("S3 nonterminal position has no legal actions"));
        }
        let index = deterministic_index(
            seed,
            game.completed_turns(),
            observed.len(),
            b"s3-action-delta",
        );
        let (selected, edit) = &observed[index];
        let applied = prepared.apply(edit)?;
        let mut geometry_record = applied.record.clone();
        geometry_record.market_entities = trunk.public_record()?.market_entities;
        let after_sparse = SparsePublicState::from_position_record(&geometry_record, None)?;
        let after_graph = BoardGraph::from_sparse(&after_sparse, 0)?;
        let before_score = graph.boards[0].score_anatomy();
        let decoded = after_graph.score_anatomy().delta(before_score);
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
        metrics.action_delta_decoder_checks += 1;
        if decoded != expected {
            metrics.action_delta_decoder_failures += 1;
        }
        game.apply(selected)?;
    }
    if metrics.positions != 80 {
        return Err(invalid(format!(
            "S3 seed {seed} produced {} positions instead of 80",
            metrics.positions
        )));
    }
    Ok(metrics)
}

fn validate_scores(game: &GameState, graph: &RelationalStateGraph, metrics: &mut SeedMetrics) {
    for (relative_seat, board) in graph.boards.iter().enumerate() {
        let absolute = (game.current_player() + relative_seat) % game.boards().len();
        let expected = score_board(&game.boards()[absolute], game.config().scoring_cards);
        let actual = board.score_anatomy();
        metrics.board_score_decoder_checks += 1;
        if actual.habitat != expected.habitat
            || actual.wildlife != expected.wildlife
            || actual.nature_tokens != expected.nature_tokens
            || actual.base_total != expected.base_total
        {
            metrics.board_score_decoder_failures += 1;
        }
    }
}

fn observe_coverage(graph: &RelationalStateGraph, metrics: &mut SeedMetrics) {
    for board in &graph.boards {
        metrics.boards_with_elk_extensions +=
            u64::from(board.opportunity.elk_eligible_extensions > 0);
        metrics.boards_with_salmon_continuations +=
            u64::from(board.opportunity.salmon_legal_continuations > 0);
        metrics.boards_with_hawk_opportunities +=
            u64::from(board.opportunity.hawk_isolated_opportunities > 0);
        metrics.boards_with_bear_pair_opportunities +=
            u64::from(board.opportunity.bear_pair_completion_cells > 0);
    }
}

fn observe_sizes(
    trunk: &PublicStateTrunk,
    graph: &RelationalStateGraph,
    metrics: &mut SeedMetrics,
) -> Result<()> {
    metrics
        .raw_control_bytes
        .push(trunk.to_packed_bytes()?.len() as u64);
    let component_only = ComponentOnlyView {
        boards: graph
            .boards
            .iter()
            .map(|board| board.habitat_components.as_slice())
            .collect(),
    };
    let motif_only = MotifOnlyView {
        boards: graph.boards.iter().map(motif_view).collect(),
    };
    let combined = CombinedView {
        boards: graph
            .boards
            .iter()
            .map(|board| CombinedBoardView {
                components: &board.habitat_components,
                motifs: motif_view(board),
                nature_tokens: board.nature_tokens,
            })
            .collect(),
    };
    metrics
        .component_only_bytes
        .push(postcard::to_allocvec(&component_only)?.len() as u64);
    metrics
        .motif_only_bytes
        .push(postcard::to_allocvec(&motif_only)?.len() as u64);
    metrics
        .combined_bytes
        .push(postcard::to_allocvec(&combined)?.len() as u64);
    metrics
        .combined_frontier_bytes
        .push(postcard::to_allocvec(graph)?.len() as u64);
    metrics.component_tokens.push(
        graph
            .boards
            .iter()
            .map(|board| board.component_token_count() as u64)
            .sum(),
    );
    metrics.motif_tokens.push(
        graph
            .boards
            .iter()
            .map(|board| board.motif_token_count() as u64)
            .sum(),
    );
    metrics.frontier_tokens.push(
        trunk
            .sparse
            .legal_frontier
            .len()
            .try_into()
            .map_err(|_| invalid("frontier count does not fit u64"))?,
    );
    Ok(())
}

fn motif_view(board: &BoardGraph) -> MotifOnlyBoardView<'_> {
    MotifOnlyBoardView {
        bear: &board.bear_components,
        elk: &board.elk_lines,
        salmon: &board.salmon_components,
        hawk_positions: &board.hawk_positions,
        hawk_edges: &board.hawk_conflict_edges,
        fox: &board.fox_centers,
    }
}

fn validate_d6(
    sparse: &SparsePublicState,
    graph: &RelationalStateGraph,
    metrics: &mut SeedMetrics,
) -> Result<()> {
    let expected = graph
        .boards
        .iter()
        .map(invariant_signature)
        .collect::<Vec<_>>();
    for transform in D6Transform::ALL {
        let transformed = sparse.transformed(transform)?;
        let transformed_graph = RelationalStateGraph::from_sparse(&transformed)?;
        let actual = transformed_graph
            .boards
            .iter()
            .map(invariant_signature)
            .collect::<Vec<_>>();
        metrics.d6_invariance_checks += 1;
        if actual != expected {
            metrics.d6_invariance_failures += 1;
        }
    }
    Ok(())
}

fn invariant_signature(board: &BoardGraph) -> InvariantBoardSignature {
    let mut habitat = board
        .habitat_components
        .iter()
        .map(|component| HabitatInvariant {
            terrain: component.terrain as u8,
            members: component.member_count,
            internal_edges: component.matching_internal_edge_count,
            open_boundary: component.open_boundary_edge_count,
            frontier_contacts: component.frontier_contact_count,
            cycle_rank: component.cycle_rank,
            bridges: component.bridge_count,
            articulations: component.articulation_count,
            merge_frontiers: component.merge_frontier_count,
        })
        .collect::<Vec<_>>();
    habitat.sort_unstable();
    let mut bear = board
        .bear_components
        .iter()
        .map(|component| {
            (
                component.members.len(),
                component.edge_count,
                component.endpoint_count,
                component.maximum_degree,
            )
        })
        .collect::<Vec<_>>();
    bear.sort_unstable();
    let mut elk = board
        .elk_lines
        .iter()
        .map(|line| (line.members.len(), line.eligible_extension_count))
        .collect::<Vec<_>>();
    // Axis labels rotate under D6, so compare only the line multiset.
    elk.sort_unstable();
    let mut salmon = board
        .salmon_components
        .iter()
        .map(|component| {
            (
                component.members.len(),
                component.edge_count,
                component.endpoint_count,
                component.branch_conflict_count,
                component.valid_run,
                component.legal_continuations.len(),
            )
        })
        .collect::<Vec<_>>();
    salmon.sort_unstable();
    let mut fox = board
        .fox_centers
        .iter()
        .map(|center| {
            (
                center.neighbor_diversity_mask.count_ones(),
                center.missing_wildlife_mask.count_ones(),
                center.compatible_cells.len(),
            )
        })
        .collect::<Vec<_>>();
    fox.sort_unstable();
    InvariantBoardSignature {
        habitat,
        bear,
        elk,
        salmon,
        hawk_positions: board.hawk_positions.len(),
        hawk_edges: board.hawk_conflict_edges.len(),
        fox,
        opportunity: board.opportunity.clone(),
        frontier: board.frontier.clone(),
        score: board.score_anatomy(),
    }
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameConfig, GameSeed, GameState};

    use super::*;

    #[test]
    fn invariant_signature_survives_all_d6_transforms() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(3),
        )
        .unwrap();
        let trunk = PublicStateTrunk::observe(&game, 3).unwrap();
        let graph = RelationalStateGraph::from_sparse(&trunk.sparse).unwrap();
        let expected = graph
            .boards
            .iter()
            .map(invariant_signature)
            .collect::<Vec<_>>();
        for transform in D6Transform::ALL {
            let transformed = trunk.sparse.transformed(transform).unwrap();
            let actual = RelationalStateGraph::from_sparse(&transformed)
                .unwrap()
                .boards
                .iter()
                .map(invariant_signature)
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
    }
}
