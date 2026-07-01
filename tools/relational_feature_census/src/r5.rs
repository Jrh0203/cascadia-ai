use std::collections::BTreeSet;

use cascadia_game::{GameState, Rotation, Terrain};
use r2_sparse_entity_census::{AxialCoord as R2Coord, OccupiedTileToken};
use r3_action_edit_census::{
    ActionEdit, AxialCoord, BoardTileToken, DraftFactor, ImmediateScoreDelta, PublicStateTrunk,
    TileSemantic,
};
use serde::{Deserialize, Serialize};

use crate::{
    BoardGraph, CommonConfig, DistributionSummary, ExperimentLane, ReportEnvelope, Result,
    common::{deterministic_index, envelope, run_games, unix_ms},
    invalid,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct R5Metrics {
    pub positions: u64,
    pub complete_actions: u64,
    pub current_score_decoder_checks: u64,
    pub current_score_decoder_failures: u64,
    pub control_affordance_checks: u64,
    pub control_affordance_failures: u64,
    pub quotient_affordance_underdetermined: u64,
    pub local_affordance_checks: u64,
    pub local_affordance_failures: u64,
    pub local_score_delta_checks: u64,
    pub local_score_delta_failures: u64,
    pub control_parent_bytes: DistributionSummary,
    pub quotient_parent_bytes: DistributionSummary,
    pub local_action_bytes: DistributionSummary,
    pub hybrid_parent_bytes: DistributionSummary,
    pub control_parent_tokens: DistributionSummary,
    pub quotient_parent_tokens: DistributionSummary,
    pub hybrid_parent_tokens: DistributionSummary,
    pub quotient_to_control_parent_bytes_ppm: u64,
    pub quotient_to_control_parent_tokens_ppm: u64,
    pub local_exact_decoding_pass: bool,
    pub quotient_score_preservation_pass: bool,
    pub material_parent_compaction_pass: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct ComponentMotifStateView<'a> {
    global: &'a r2_sparse_entity_census::GlobalMetadata,
    players: &'a [r2_sparse_entity_census::PlayerMetadata],
    market: &'a [r2_sparse_entity_census::MarketToken],
    supply: &'a r3_action_edit_census::SupplySnapshot,
    boards: Vec<ComponentMotifBoardView<'a>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct ComponentMotifBoardView<'a> {
    relative_seat: u8,
    habitat_components: &'a [crate::HabitatComponentGraph],
    bear_components: &'a [crate::graph::WildlifeComponentGraph],
    elk_lines: &'a [crate::graph::ElkLineGraph],
    salmon_components: &'a [crate::graph::SalmonComponentGraph],
    hawk_positions: &'a [R2Coord],
    hawk_conflict_edges: &'a [[R2Coord; 2]],
    fox_centers: &'a [crate::graph::FoxCenterGraph],
    nature_tokens: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct LocalNeighbor {
    edge_from_destination: u8,
    token: BoardTileToken,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct WildlifeSite {
    coord: AxialCoord,
    token: Option<BoardTileToken>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ActionLocalGeometry {
    tile_destination: AxialCoord,
    destination_is_frontier: bool,
    neighbors: Vec<LocalNeighbor>,
    wildlife_site: Option<WildlifeSite>,
    active_nature_tokens: u8,
}

#[derive(Default)]
struct SeedMetrics {
    positions: u64,
    complete_actions: u64,
    current_score_decoder_checks: u64,
    current_score_decoder_failures: u64,
    control_affordance_checks: u64,
    control_affordance_failures: u64,
    quotient_affordance_underdetermined: u64,
    local_affordance_checks: u64,
    local_affordance_failures: u64,
    local_score_delta_checks: u64,
    local_score_delta_failures: u64,
    control_parent_bytes: Vec<u64>,
    quotient_parent_bytes: Vec<u64>,
    local_action_bytes: Vec<u64>,
    hybrid_parent_bytes: Vec<u64>,
    control_parent_tokens: Vec<u64>,
    quotient_parent_tokens: Vec<u64>,
    hybrid_parent_tokens: Vec<u64>,
}

impl SeedMetrics {
    fn merge(&mut self, other: Self) {
        self.positions += other.positions;
        self.complete_actions += other.complete_actions;
        self.current_score_decoder_checks += other.current_score_decoder_checks;
        self.current_score_decoder_failures += other.current_score_decoder_failures;
        self.control_affordance_checks += other.control_affordance_checks;
        self.control_affordance_failures += other.control_affordance_failures;
        self.quotient_affordance_underdetermined += other.quotient_affordance_underdetermined;
        self.local_affordance_checks += other.local_affordance_checks;
        self.local_affordance_failures += other.local_affordance_failures;
        self.local_score_delta_checks += other.local_score_delta_checks;
        self.local_score_delta_failures += other.local_score_delta_failures;
        self.control_parent_bytes.extend(other.control_parent_bytes);
        self.quotient_parent_bytes
            .extend(other.quotient_parent_bytes);
        self.local_action_bytes.extend(other.local_action_bytes);
        self.hybrid_parent_bytes.extend(other.hybrid_parent_bytes);
        self.control_parent_tokens
            .extend(other.control_parent_tokens);
        self.quotient_parent_tokens
            .extend(other.quotient_parent_tokens);
        self.hybrid_parent_tokens.extend(other.hybrid_parent_tokens);
    }
}

pub fn run_r5(config: CommonConfig) -> Result<ReportEnvelope<R5Metrics>> {
    if config.lane != ExperimentLane::R5Quotient {
        return Err(invalid("R5 runner received a non-R5 lane"));
    }
    let started = unix_ms()?;
    let per_seed = run_games(&config, run_seed)?;
    let mut combined = SeedMetrics::default();
    for (_, metrics) in per_seed {
        combined.merge(metrics);
    }
    let control_parent_bytes = DistributionSummary::from_values(combined.control_parent_bytes)?;
    let quotient_parent_bytes = DistributionSummary::from_values(combined.quotient_parent_bytes)?;
    let local_action_bytes = DistributionSummary::from_values(combined.local_action_bytes)?;
    let hybrid_parent_bytes = DistributionSummary::from_values(combined.hybrid_parent_bytes)?;
    let control_parent_tokens = DistributionSummary::from_values(combined.control_parent_tokens)?;
    let quotient_parent_tokens = DistributionSummary::from_values(combined.quotient_parent_tokens)?;
    let hybrid_parent_tokens = DistributionSummary::from_values(combined.hybrid_parent_tokens)?;
    let quotient_to_control_parent_bytes_ppm = quotient_parent_bytes
        .median
        .checked_mul(1_000_000)
        .ok_or_else(|| invalid("R5 compactness ratio overflowed"))?
        / control_parent_bytes.median.max(1);
    let quotient_to_control_parent_tokens_ppm = quotient_parent_tokens
        .median
        .checked_mul(1_000_000)
        .ok_or_else(|| invalid("R5 token compactness ratio overflowed"))?
        / control_parent_tokens.median.max(1);
    let local_exact_decoding_pass = combined.control_affordance_failures == 0
        && combined.local_affordance_failures == 0
        && combined.local_score_delta_failures == 0
        && combined.complete_actions > 0;
    let quotient_score_preservation_pass = combined.current_score_decoder_failures == 0;
    let material_parent_compaction_pass = quotient_to_control_parent_bytes_ppm <= 800_000
        || quotient_to_control_parent_tokens_ppm <= 800_000;
    let passed = local_exact_decoding_pass
        && quotient_score_preservation_pass
        && material_parent_compaction_pass
        && combined.quotient_affordance_underdetermined == combined.complete_actions;
    let classification = if passed {
        "r5_local_geometry_exact_and_quotient_compact"
    } else if !local_exact_decoding_pass {
        "r5_action_local_exactness_failed"
    } else if !quotient_score_preservation_pass {
        "r5_component_motif_score_decoder_failed"
    } else {
        "r5_component_motif_compactness_failed"
    };
    envelope(
        config,
        R5Metrics {
            positions: combined.positions,
            complete_actions: combined.complete_actions,
            current_score_decoder_checks: combined.current_score_decoder_checks,
            current_score_decoder_failures: combined.current_score_decoder_failures,
            control_affordance_checks: combined.control_affordance_checks,
            control_affordance_failures: combined.control_affordance_failures,
            quotient_affordance_underdetermined: combined.quotient_affordance_underdetermined,
            local_affordance_checks: combined.local_affordance_checks,
            local_affordance_failures: combined.local_affordance_failures,
            local_score_delta_checks: combined.local_score_delta_checks,
            local_score_delta_failures: combined.local_score_delta_failures,
            control_parent_bytes,
            quotient_parent_bytes,
            local_action_bytes,
            hybrid_parent_bytes,
            control_parent_tokens,
            quotient_parent_tokens,
            hybrid_parent_tokens,
            quotient_to_control_parent_bytes_ppm,
            quotient_to_control_parent_tokens_ppm,
            local_exact_decoding_pass,
            quotient_score_preservation_pass,
            material_parent_compaction_pass,
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
            .ok_or_else(|| invalid("R5 game index overflowed"))?;
        let trunk = PublicStateTrunk::observe(&game, game_index)?;
        let prepared = trunk.prepare_action_edits()?;
        let state_graph = crate::RelationalStateGraph::from_sparse(&trunk.sparse)?;
        for (relative_seat, graph) in state_graph.boards.iter().enumerate() {
            metrics.current_score_decoder_checks += 1;
            let absolute = (game.current_player() + relative_seat) % game.boards().len();
            let expected =
                cascadia_game::score_board(&game.boards()[absolute], game.config().scoring_cards);
            let actual = graph.score_anatomy();
            if actual.habitat != expected.habitat
                || actual.wildlife != expected.wildlife
                || actual.nature_tokens != expected.nature_tokens
                || actual.base_total != expected.base_total
            {
                metrics.current_score_decoder_failures += 1;
            }
        }

        let quotient = component_motif_view(&trunk, &state_graph);
        let quotient_bytes = postcard::to_allocvec(&quotient)?.len() as u64;
        let graph_tokens = state_graph
            .boards
            .iter()
            .map(|board| board.component_token_count() + board.motif_token_count())
            .sum::<usize>();
        metrics
            .control_parent_bytes
            .push(prepared.packed_bytes().len() as u64);
        metrics.quotient_parent_bytes.push(quotient_bytes);
        metrics.hybrid_parent_bytes.push(
            postcard::to_allocvec(&(&trunk.sparse, &trunk.supply, &state_graph))?.len() as u64,
        );
        metrics
            .control_parent_tokens
            .push(trunk.token_count() as u64);
        metrics.quotient_parent_tokens.push(
            (1 + trunk.sparse.players.len()
                + trunk.sparse.market.len()
                + graph_tokens
                + 1
                + trunk.supply.archetype_counts.len()) as u64,
        );
        metrics
            .hybrid_parent_tokens
            .push((trunk.token_count() + graph_tokens) as u64);

        let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
        let observed = prepared.observe_legal_actions(&game, &prelude)?;
        if observed.is_empty() {
            return Err(invalid("R5 nonterminal position has no legal actions"));
        }
        metrics.complete_actions += observed.len() as u64;
        let active_graph = &state_graph.boards[0];
        for (_, edit) in &observed {
            metrics.control_affordance_checks += 1;
            if !raw_board_affordance(&trunk, edit)? {
                metrics.control_affordance_failures += 1;
            }
            metrics.quotient_affordance_underdetermined += 1;
            let local = action_local_geometry(&trunk, edit)?;
            metrics
                .local_action_bytes
                .push(postcard::to_allocvec(&local)?.len() as u64);
            metrics.local_affordance_checks += 1;
            if !local_board_affordance(&local, edit)? {
                metrics.local_affordance_failures += 1;
            }
            metrics.local_score_delta_checks += 1;
            let decoded = decode_local_score_delta(active_graph, &local, edit)?;
            if decoded != edit.score_delta {
                metrics.local_score_delta_failures += 1;
            }
        }
        let selected = observed
            [deterministic_index(seed, game.completed_turns(), observed.len(), b"r5-advance")]
        .0
        .clone();
        game.apply(&selected)?;
    }
    if metrics.positions != 80 {
        return Err(invalid(format!(
            "R5 seed {seed} produced {} positions instead of 80",
            metrics.positions
        )));
    }
    Ok(metrics)
}

fn component_motif_view<'a>(
    trunk: &'a PublicStateTrunk,
    graph: &'a crate::RelationalStateGraph,
) -> ComponentMotifStateView<'a> {
    ComponentMotifStateView {
        global: &trunk.sparse.global,
        players: &trunk.sparse.players,
        market: &trunk.sparse.market,
        supply: &trunk.supply,
        boards: graph
            .boards
            .iter()
            .map(|board| ComponentMotifBoardView {
                relative_seat: board.relative_seat,
                habitat_components: &board.habitat_components,
                bear_components: &board.bear_components,
                elk_lines: &board.elk_lines,
                salmon_components: &board.salmon_components,
                hawk_positions: &board.hawk_positions,
                hawk_conflict_edges: &board.hawk_conflict_edges,
                fox_centers: &board.fox_centers,
                nature_tokens: board.nature_tokens,
            })
            .collect(),
    }
}

fn action_local_geometry(
    trunk: &PublicStateTrunk,
    edit: &ActionEdit,
) -> Result<ActionLocalGeometry> {
    let destination = edit.factors.tile_destination;
    let destination_r2 = R2Coord::new(destination.q, destination.r);
    let frontier = trunk
        .sparse
        .legal_frontier
        .iter()
        .find(|frontier| frontier.relative_seat == 0 && frontier.coord == destination_r2);
    let board = trunk
        .sparse
        .occupied_tiles
        .iter()
        .filter(|tile| tile.relative_seat == 0)
        .map(|tile| (tile.coord, tile))
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut neighbors = Vec::new();
    for edge in 0..6 {
        let coord = destination_r2.neighbor(edge);
        if let Some(tile) = board.get(&coord) {
            neighbors.push(LocalNeighbor {
                edge_from_destination: edge as u8,
                token: board_tile_from_r2(tile),
            });
        }
    }
    let wildlife_site = edit.factors.wildlife_destination.map(|coord| {
        let r2 = R2Coord::new(coord.q, coord.r);
        WildlifeSite {
            coord,
            token: board.get(&r2).map(|tile| board_tile_from_r2(tile)),
        }
    });
    Ok(ActionLocalGeometry {
        tile_destination: destination,
        destination_is_frontier: frontier.is_some(),
        neighbors,
        wildlife_site,
        active_nature_tokens: trunk.sparse.players[0].nature_tokens,
    })
}

fn board_tile_from_r2(tile: &OccupiedTileToken) -> BoardTileToken {
    BoardTileToken {
        coord: AxialCoord::new(tile.coord.q, tile.coord.r),
        tile: TileSemantic::new(
            tile.terrain_a,
            tile.terrain_b,
            tile.wildlife_eligibility,
            tile.keystone,
        ),
        rotation: tile.rotation.get(),
        directed_edge_terrains: tile.directed_edge_terrains,
        placed_wildlife: tile.placed_wildlife,
    }
}

fn raw_board_affordance(trunk: &PublicStateTrunk, edit: &ActionEdit) -> Result<bool> {
    local_board_affordance(&action_local_geometry(trunk, edit)?, edit)
}

fn local_board_affordance(local: &ActionLocalGeometry, edit: &ActionEdit) -> Result<bool> {
    if !local.destination_is_frontier || local.tile_destination != edit.factors.tile_destination {
        return Ok(false);
    }
    let rotation = Rotation::new(edit.factors.tile_rotation)
        .ok_or_else(|| invalid("R5 action rotation is invalid"))?;
    let selected = edit.selected.tile.as_tile();
    if selected.canonical_rotation(rotation) != rotation
        || std::array::from_fn(|edge| selected.terrain_on_edge(rotation, edge))
            != edit.factors.tile_directed_edges
    {
        return Ok(false);
    }
    let token_cost = edit.factors.wildlife_wipe_masks.len()
        + usize::from(matches!(
            edit.factors.draft,
            DraftFactor::Independent { .. }
        ));
    if token_cost > usize::from(local.active_nature_tokens) {
        return Ok(false);
    }
    let Some(destination) = edit.factors.wildlife_destination else {
        return Ok(true);
    };
    if destination == edit.factors.tile_destination {
        return Ok(edit
            .selected
            .tile
            .wildlife_eligibility
            .contains(edit.selected.wildlife));
    }
    let Some(site) = &local.wildlife_site else {
        return Ok(false);
    };
    let Some(tile) = &site.token else {
        return Ok(false);
    };
    Ok(site.coord == destination
        && tile.placed_wildlife.is_none()
        && tile
            .tile
            .wildlife_eligibility
            .contains(edit.selected.wildlife))
}

fn decode_local_score_delta(
    before: &BoardGraph,
    local: &ActionLocalGeometry,
    edit: &ActionEdit,
) -> Result<ImmediateScoreDelta> {
    let before_score = before.score_anatomy();
    let mut after_habitat = before_score.habitat;
    for terrain in Terrain::ALL {
        if !edit.selected.tile.as_tile().contains_terrain(terrain) {
            continue;
        }
        let mut touched = BTreeSet::new();
        for neighbor in &local.neighbors {
            let edge = usize::from(neighbor.edge_from_destination);
            if edit.factors.tile_directed_edges[edge] != terrain
                || neighbor.token.directed_edge_terrains[(edge + 3) % 6] != terrain
            {
                continue;
            }
            if let Some(component) = before.habitat_components.iter().find(|component| {
                component.terrain == terrain
                    && component.members.iter().any(|member| {
                        member.q == neighbor.token.coord.q && member.r == neighbor.token.coord.r
                    })
            }) {
                touched.insert(component.component_id);
            }
        }
        let resulting = 1u16
            + touched
                .iter()
                .filter_map(|component_id| {
                    before
                        .habitat_components
                        .iter()
                        .find(|component| {
                            component.terrain == terrain && component.component_id == *component_id
                        })
                        .map(|component| component.member_count)
                })
                .sum::<u16>();
        after_habitat[terrain as usize] = after_habitat[terrain as usize].max(resulting);
    }
    let added_wildlife = edit
        .factors
        .wildlife_destination
        .map(|coord| (R2Coord::new(coord.q, coord.r), edit.selected.wildlife));
    let after_wildlife = before.wildlife_scores_with_added(added_wildlife);
    let mut nature_tokens = i16::from(before.nature_tokens)
        - i16::try_from(edit.factors.wildlife_wipe_masks.len())?
        - i16::from(matches!(
            edit.factors.draft,
            DraftFactor::Independent { .. }
        ));
    if let Some(destination) = edit.factors.wildlife_destination {
        let keystone = if destination == edit.factors.tile_destination {
            edit.selected.tile.keystone
        } else {
            local
                .wildlife_site
                .as_ref()
                .and_then(|site| site.token.as_ref())
                .is_some_and(|tile| tile.tile.keystone)
        };
        nature_tokens += i16::from(keystone);
    }
    if nature_tokens < 0 {
        return Err(invalid("R5 decoded negative Nature Token count"));
    }
    let habitat_delta = std::array::from_fn(|index| {
        after_habitat[index] as i16 - before_score.habitat[index] as i16
    });
    let wildlife_delta = std::array::from_fn(|index| {
        after_wildlife[index] as i16 - before_score.wildlife[index] as i16
    });
    let nature_delta = nature_tokens - i16::from(before.nature_tokens);
    Ok(ImmediateScoreDelta {
        habitat: habitat_delta,
        wildlife: wildlife_delta,
        nature_tokens: nature_delta,
        base_total: habitat_delta.iter().sum::<i16>()
            + wildlife_delta.iter().sum::<i16>()
            + nature_delta,
    })
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameConfig, GameSeed};

    use super::*;

    #[test]
    fn local_decoder_matches_every_action_in_one_opening_position() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(7),
        )
        .unwrap();
        let trunk = PublicStateTrunk::observe(&game, 7).unwrap();
        let prepared = trunk.prepare_action_edits().unwrap();
        let graph = BoardGraph::from_sparse(&trunk.sparse, 0).unwrap();
        let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible().unwrap();
        for (_, edit) in prepared.observe_legal_actions(&game, &prelude).unwrap() {
            let local = action_local_geometry(&trunk, &edit).unwrap();
            assert!(local_board_affordance(&local, &edit).unwrap());
            assert_eq!(
                decode_local_score_delta(&graph, &local, &edit).unwrap(),
                edit.score_delta
            );
        }
    }
}
