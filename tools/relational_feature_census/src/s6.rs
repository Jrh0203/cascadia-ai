use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    time::Instant,
};

use cascadia_game::{D6Transform, GameState, Terrain, Wildlife};
use r2_sparse_entity_census::{AxialCoord, OccupiedTileToken, SparsePublicState};
use r3_action_edit_census::PublicStateTrunk;
use serde::{Deserialize, Serialize};

use crate::{
    BoardGraph, CommonConfig, DistributionSummary, ExperimentLane, RelationalStateGraph,
    ReportEnvelope, Result,
    common::{canonical_blake3, deterministic_index, envelope, run_games, unix_ms},
    invalid,
};

const S6_ENCODING_SCHEMA_VERSION: u16 = 1;
const WALK_SCALE: u64 = 1_000_000;
const WALK_STEPS: [usize; 4] = [2, 3, 4, 6];
const LAPLACIAN_MOMENTS: usize = 6;
const EXTRACTION_P99_LIMIT_NS: u64 = 2_000_000;
const MEDIAN_ENCODING_LIMIT_BYTES: u64 = 4_096;
const MIN_UNIQUE_TOPOLOGY_ENCODINGS: u64 = 128;
const MIN_UNIQUE_PATH_ENCODINGS: u64 = 16;
const MIN_UNIQUE_RANDOM_WALK_ENCODINGS: u64 = 128;
const MIN_UNIQUE_SPECTRAL_ENCODINGS: u64 = 128;
const MIN_UNIQUE_FULL_ENCODINGS: u64 = 256;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphChannelEncoding {
    pub node_count: u16,
    pub edge_count: u16,
    pub component_count: u16,
    pub boundary_half_edges: u16,
    pub cycle_rank: u16,
    pub bridge_count: u16,
    pub articulation_count: u16,
    pub hole_count: u16,
    pub diameter: u16,
    pub maximum_component_radius: u16,
    pub reachable_pair_count: u16,
    pub distance_sum: u32,
    pub degree_histogram: [u16; 7],
    pub random_walk_return_ppm: [u32; 4],
    pub laplacian_trace_moments: [u64; LAPLACIAN_MOMENTS],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarkedPathSummary {
    pub marked_node_count: u16,
    pub reachable_pair_count: u16,
    pub unreachable_pair_count: u16,
    pub minimum_distance: u16,
    pub median_distance: u16,
    pub maximum_distance: u16,
    pub mean_distance_milli: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TopologicalBoardEncoding {
    pub relative_seat: u8,
    pub occupancy: GraphChannelEncoding,
    pub habitat: [GraphChannelEncoding; 5],
    pub wildlife: [GraphChannelEncoding; 5],
    pub elk_endpoint_paths: MarkedPathSummary,
    pub salmon_continuation_paths: MarkedPathSummary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct S6StateEncoding {
    pub schema_version: u16,
    pub boards: Vec<TopologicalBoardEncoding>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct S6Metrics {
    pub positions: u64,
    pub board_encodings: u64,
    pub topology_decoder_checks: u64,
    pub topology_decoder_failures: u64,
    pub d6_invariance_checks: u64,
    pub d6_invariance_failures: u64,
    pub adversarial_checks: u64,
    pub adversarial_failures: u64,
    pub baseline_collision_pairs: u64,
    pub topology_separated_pairs: u64,
    pub path_separated_pairs: u64,
    pub random_walk_separated_pairs: u64,
    pub spectral_separated_pairs: u64,
    pub full_encoding_separated_pairs: u64,
    pub long_range_collision_pairs: u64,
    pub long_range_separated_pairs: u64,
    pub unique_topology_encodings: u64,
    pub unique_path_encodings: u64,
    pub unique_random_walk_encodings: u64,
    pub unique_spectral_encodings: u64,
    pub unique_full_encodings: u64,
    pub boards_with_long_range_paths: u64,
    pub boards_with_geometric_holes: u64,
    pub full_separation_rate_ppm: u64,
    pub encoding_bytes: DistributionSummary,
    pub extraction_ns: DistributionSummary,
    pub isolated_extraction_ns: DistributionSummary,
    pub exactness_gate_pass: bool,
    pub d6_gate_pass: bool,
    pub adversarial_gate_pass: bool,
    pub feature_variation_gate_pass: bool,
    pub long_range_gate_pass: bool,
    pub isolated_latency_gate_pass: bool,
    pub compactness_gate_pass: bool,
}

#[derive(Debug, Clone)]
struct UndirectedGraph {
    coords: Vec<AxialCoord>,
    adjacency: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, Serialize)]
struct BaselineBoardSignature {
    habitat: Vec<BaselineHabitatComponent>,
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
struct BaselineHabitatComponent {
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct ScalarProjection {
    node_count: u16,
    edge_count: u16,
    component_count: u16,
    boundary_half_edges: u16,
    cycle_rank: u16,
    bridge_count: u16,
    articulation_count: u16,
    hole_count: u16,
    diameter: u16,
    maximum_component_radius: u16,
    reachable_pair_count: u16,
    distance_sum: u32,
    degree_histogram: [u16; 7],
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LegacyScalarProjection {
    node_count: u16,
    edge_count: u16,
    component_count: u16,
    boundary_half_edges: u16,
    cycle_rank: u16,
    bridge_count: u16,
    articulation_count: u16,
    degree_histogram: [u16; 7],
}

#[derive(Debug, Clone)]
struct CollisionObservation {
    baseline_hash: String,
    topology_hash: String,
    path_hash: String,
    random_walk_hash: String,
    spectral_hash: String,
    full_hash: String,
    long_range: bool,
}

#[derive(Debug, Default)]
struct CollisionMetrics {
    baseline_pairs: u64,
    topology_separated: u64,
    path_separated: u64,
    random_walk_separated: u64,
    spectral_separated: u64,
    full_separated: u64,
    long_range_pairs: u64,
    long_range_separated: u64,
}

#[derive(Debug)]
struct FeatureVariationMetrics {
    topology: u64,
    path: u64,
    random_walk: u64,
    spectral: u64,
    full: u64,
}

#[derive(Debug, Default)]
struct SeedMetrics {
    positions: u64,
    board_encodings: u64,
    topology_decoder_checks: u64,
    topology_decoder_failures: u64,
    d6_invariance_checks: u64,
    d6_invariance_failures: u64,
    boards_with_long_range_paths: u64,
    boards_with_geometric_holes: u64,
    encoding_bytes: Vec<u64>,
    extraction_ns: Vec<u64>,
    observations: Vec<CollisionObservation>,
}

impl SeedMetrics {
    fn merge(&mut self, other: Self) {
        self.positions += other.positions;
        self.board_encodings += other.board_encodings;
        self.topology_decoder_checks += other.topology_decoder_checks;
        self.topology_decoder_failures += other.topology_decoder_failures;
        self.d6_invariance_checks += other.d6_invariance_checks;
        self.d6_invariance_failures += other.d6_invariance_failures;
        self.boards_with_long_range_paths += other.boards_with_long_range_paths;
        self.boards_with_geometric_holes += other.boards_with_geometric_holes;
        self.encoding_bytes.extend(other.encoding_bytes);
        self.extraction_ns.extend(other.extraction_ns);
        self.observations.extend(other.observations);
    }
}

#[derive(Debug)]
struct AdversarialSummary {
    checks: u64,
    failures: u64,
}

pub fn run_s6(config: CommonConfig) -> Result<ReportEnvelope<S6Metrics>> {
    if config.lane != ExperimentLane::S6Topology {
        return Err(invalid("S6 runner received a non-S6 lane"));
    }
    config.validate()?;
    let started = unix_ms()?;
    let adversarial = adversarial_suite()?;
    let timing_seed = config
        .first_seed
        .checked_sub(1)
        .ok_or_else(|| invalid("S6 timing seed underflowed"))?;
    let isolated_extraction_ns =
        DistributionSummary::from_values(run_isolated_timing_probe(timing_seed)?)?;
    let per_seed = run_games(&config, run_seed)?;
    let mut combined = SeedMetrics::default();
    for (_, metrics) in per_seed {
        combined.merge(metrics);
    }
    let collision = collision_metrics(&combined.observations)?;
    let variation = feature_variation_metrics(&combined.observations)?;
    let encoding_bytes = DistributionSummary::from_values(combined.encoding_bytes)?;
    let extraction_ns = DistributionSummary::from_values(combined.extraction_ns)?;
    let full_separation_rate_ppm = collision
        .full_separated
        .checked_mul(1_000_000)
        .ok_or_else(|| invalid("S6 separation-rate numerator overflowed"))?
        / collision.baseline_pairs.max(1);

    let exactness_gate_pass = combined.topology_decoder_failures == 0
        && combined.topology_decoder_checks == combined.board_encodings * 11;
    let d6_gate_pass = combined.d6_invariance_failures == 0
        && combined.d6_invariance_checks == combined.positions * 12;
    let adversarial_gate_pass = adversarial.failures == 0 && adversarial.checks == 4;
    let feature_variation_gate_pass = variation.topology >= MIN_UNIQUE_TOPOLOGY_ENCODINGS
        && variation.path >= MIN_UNIQUE_PATH_ENCODINGS
        && variation.random_walk >= MIN_UNIQUE_RANDOM_WALK_ENCODINGS
        && variation.spectral >= MIN_UNIQUE_SPECTRAL_ENCODINGS
        && variation.full >= MIN_UNIQUE_FULL_ENCODINGS;
    let long_range_gate_pass = combined.boards_with_long_range_paths > 0;
    let isolated_latency_gate_pass = isolated_extraction_ns.p99 <= EXTRACTION_P99_LIMIT_NS;
    let compactness_gate_pass = encoding_bytes.median <= MEDIAN_ENCODING_LIMIT_BYTES;
    let passed = exactness_gate_pass
        && d6_gate_pass
        && adversarial_gate_pass
        && feature_variation_gate_pass
        && long_range_gate_pass
        && isolated_latency_gate_pass
        && compactness_gate_pass;
    let classification = if passed {
        "s6_topological_spectral_foundation_v2_authorized"
    } else if !exactness_gate_pass {
        "s6_topology_decoder_failed"
    } else if !d6_gate_pass {
        "s6_d6_invariance_failed"
    } else if !adversarial_gate_pass {
        "s6_adversarial_separation_failed"
    } else if !feature_variation_gate_pass {
        "s6_feature_variation_futile"
    } else if !long_range_gate_pass {
        "s6_long_range_coverage_futile"
    } else if !isolated_latency_gate_pass {
        "s6_isolated_latency_failed"
    } else {
        "s6_encoding_compactness_failed"
    };

    envelope(
        config,
        S6Metrics {
            positions: combined.positions,
            board_encodings: combined.board_encodings,
            topology_decoder_checks: combined.topology_decoder_checks,
            topology_decoder_failures: combined.topology_decoder_failures,
            d6_invariance_checks: combined.d6_invariance_checks,
            d6_invariance_failures: combined.d6_invariance_failures,
            adversarial_checks: adversarial.checks,
            adversarial_failures: adversarial.failures,
            baseline_collision_pairs: collision.baseline_pairs,
            topology_separated_pairs: collision.topology_separated,
            path_separated_pairs: collision.path_separated,
            random_walk_separated_pairs: collision.random_walk_separated,
            spectral_separated_pairs: collision.spectral_separated,
            full_encoding_separated_pairs: collision.full_separated,
            long_range_collision_pairs: collision.long_range_pairs,
            long_range_separated_pairs: collision.long_range_separated,
            unique_topology_encodings: variation.topology,
            unique_path_encodings: variation.path,
            unique_random_walk_encodings: variation.random_walk,
            unique_spectral_encodings: variation.spectral,
            unique_full_encodings: variation.full,
            boards_with_long_range_paths: combined.boards_with_long_range_paths,
            boards_with_geometric_holes: combined.boards_with_geometric_holes,
            full_separation_rate_ppm,
            encoding_bytes,
            extraction_ns,
            isolated_extraction_ns,
            exactness_gate_pass,
            d6_gate_pass,
            adversarial_gate_pass,
            feature_variation_gate_pass,
            long_range_gate_pass,
            isolated_latency_gate_pass,
            compactness_gate_pass,
        },
        passed,
        classification,
        started,
    )
}

pub fn topological_state_encoding(state: &SparsePublicState) -> Result<S6StateEncoding> {
    let relational = RelationalStateGraph::from_sparse(state)?;
    topological_state_encoding_with_graph(state, &relational)
}

fn topological_state_encoding_with_graph(
    state: &SparsePublicState,
    relational: &RelationalStateGraph,
) -> Result<S6StateEncoding> {
    if relational.boards.len() != usize::from(state.global.player_count) {
        return Err(invalid(
            "S6 relational board count does not match player count",
        ));
    }
    let mut boards = Vec::with_capacity(relational.boards.len());
    for board in &relational.boards {
        boards.push(topological_board_encoding(state, board)?);
    }
    Ok(S6StateEncoding {
        schema_version: S6_ENCODING_SCHEMA_VERSION,
        boards,
    })
}

fn topological_board_encoding(
    state: &SparsePublicState,
    board: &BoardGraph,
) -> Result<TopologicalBoardEncoding> {
    let tiles = state
        .occupied_tiles
        .iter()
        .filter(|tile| tile.relative_seat == board.relative_seat)
        .collect::<Vec<_>>();
    let occupancy_graph = graph_from_coords(tiles.iter().map(|tile| tile.coord))?;
    let occupancy_boundary = boundary_half_edges(&occupancy_graph)?;
    let occupancy_holes = geometric_hole_count(&occupancy_graph.coords)?;
    let occupancy = analyze_graph(&occupancy_graph, occupancy_boundary, occupancy_holes)?;

    let mut habitat = Vec::with_capacity(5);
    for terrain in Terrain::ALL {
        let graph = habitat_graph(&tiles, terrain)?;
        let boundary = board
            .habitat_components
            .iter()
            .filter(|component| component.terrain == terrain)
            .map(|component| u32::from(component.open_boundary_edge_count))
            .sum::<u32>();
        habitat.push(analyze_graph(&graph, u16::try_from(boundary)?, 0)?);
    }
    let habitat: [GraphChannelEncoding; 5] = habitat
        .try_into()
        .map_err(|_| invalid("S6 habitat channel count is not five"))?;

    let mut wildlife = Vec::with_capacity(5);
    for species in Wildlife::ALL {
        let graph = graph_from_coords(
            tiles
                .iter()
                .filter_map(|tile| (tile.placed_wildlife == Some(species)).then_some(tile.coord)),
        )?;
        let boundary = boundary_half_edges(&graph)?;
        wildlife.push(analyze_graph(&graph, boundary, 0)?);
    }
    let wildlife: [GraphChannelEncoding; 5] = wildlife
        .try_into()
        .map_err(|_| invalid("S6 wildlife channel count is not five"))?;

    let eligible_elk = eligible_empty_coords(&tiles, Wildlife::Elk);
    let elk_endpoints = board
        .elk_lines
        .iter()
        .flat_map(|line| [line.negative_extension, line.positive_extension])
        .filter(|coord| eligible_elk.contains(coord))
        .collect::<BTreeSet<_>>();
    let salmon_continuations = board
        .salmon_components
        .iter()
        .flat_map(|component| component.legal_continuations.iter().copied())
        .collect::<BTreeSet<_>>();

    Ok(TopologicalBoardEncoding {
        relative_seat: board.relative_seat,
        occupancy,
        habitat,
        wildlife,
        elk_endpoint_paths: marked_path_summary(&occupancy_graph, &elk_endpoints)?,
        salmon_continuation_paths: marked_path_summary(&occupancy_graph, &salmon_continuations)?,
    })
}

fn run_seed(seed: u64, mut game: GameState) -> Result<SeedMetrics> {
    let mut metrics = SeedMetrics::default();
    while !game.is_game_over() {
        metrics.positions += 1;
        let game_index = seed
            .checked_mul(100)
            .and_then(|value| value.checked_add(u64::from(game.completed_turns())))
            .ok_or_else(|| invalid("S6 game index overflowed"))?;
        let trunk = PublicStateTrunk::observe(&game, game_index)?;
        let relational = RelationalStateGraph::from_sparse(&trunk.sparse)?;

        let started = Instant::now();
        let encoding = topological_state_encoding_with_graph(&trunk.sparse, &relational)?;
        metrics.extraction_ns.push(
            started
                .elapsed()
                .as_nanos()
                .try_into()
                .map_err(|_| invalid("S6 extraction duration does not fit u64"))?,
        );
        metrics
            .encoding_bytes
            .push(postcard::to_allocvec(&encoding)?.len() as u64);
        metrics.board_encodings += encoding.boards.len() as u64;

        for (board, encoded) in relational.boards.iter().zip(&encoding.boards) {
            validate_board(&trunk.sparse, board, encoded, &mut metrics)?;
            metrics.boards_with_long_range_paths += u64::from(is_long_range(encoded));
            metrics.boards_with_geometric_holes += u64::from(encoded.occupancy.hole_count > 0);
            metrics
                .observations
                .push(collision_observation(board, encoded)?);
        }
        validate_d6(&trunk.sparse, &encoding, &mut metrics)?;

        apply_deterministic_action(seed, &mut game, &trunk, b"s6-corpus-action")?;
    }
    if metrics.positions != 80 {
        return Err(invalid(format!(
            "S6 seed {seed} produced {} positions instead of 80",
            metrics.positions
        )));
    }
    Ok(metrics)
}

fn run_isolated_timing_probe(seed: u64) -> Result<Vec<u64>> {
    let mut game = GameState::new(
        cascadia_game::GameConfig::research_aaaaa(4)?,
        cascadia_game::GameSeed::from_u64(seed),
    )?;
    let mut timings = Vec::with_capacity(80);
    while !game.is_game_over() {
        let game_index = seed
            .checked_mul(100)
            .and_then(|value| value.checked_add(u64::from(game.completed_turns())))
            .ok_or_else(|| invalid("S6 timing game index overflowed"))?;
        let trunk = PublicStateTrunk::observe(&game, game_index)?;
        let started = Instant::now();
        let encoding = topological_state_encoding(&trunk.sparse)?;
        std::hint::black_box(encoding);
        timings.push(
            started
                .elapsed()
                .as_nanos()
                .try_into()
                .map_err(|_| invalid("S6 isolated duration does not fit u64"))?,
        );
        apply_deterministic_action(seed, &mut game, &trunk, b"s6-isolated-timing-action")?;
    }
    if timings.len() != 80 {
        return Err(invalid(format!(
            "S6 isolated timing seed {seed} produced {} positions instead of 80",
            timings.len()
        )));
    }
    Ok(timings)
}

fn apply_deterministic_action(
    seed: u64,
    game: &mut GameState,
    trunk: &PublicStateTrunk,
    domain: &[u8],
) -> Result<()> {
    let prepared = trunk.prepare_action_edits()?;
    let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible()?;
    let observed = prepared.observe_legal_actions(game, &prelude)?;
    if observed.is_empty() {
        return Err(invalid("S6 nonterminal position has no legal actions"));
    }
    let index = deterministic_index(seed, game.completed_turns(), observed.len(), domain);
    game.apply(&observed[index].0)?;
    Ok(())
}

fn validate_board(
    state: &SparsePublicState,
    board: &BoardGraph,
    encoding: &TopologicalBoardEncoding,
    metrics: &mut SeedMetrics,
) -> Result<()> {
    let tiles = state
        .occupied_tiles
        .iter()
        .filter(|tile| tile.relative_seat == board.relative_seat)
        .collect::<Vec<_>>();
    metrics.topology_decoder_checks += 1;
    if encoding.occupancy.node_count != u16::try_from(tiles.len())?
        || encoding.occupancy.component_count != u16::from(!tiles.is_empty())
    {
        metrics.topology_decoder_failures += 1;
    }

    for terrain in Terrain::ALL {
        let components = board
            .habitat_components
            .iter()
            .filter(|component| component.terrain == terrain)
            .collect::<Vec<_>>();
        let expected_nodes = components
            .iter()
            .flat_map(|component| component.members.iter().copied())
            .collect::<BTreeSet<_>>()
            .len();
        let expected_edges = components
            .iter()
            .map(|component| u32::from(component.matching_internal_edge_count))
            .sum::<u32>();
        let expected_boundary = components
            .iter()
            .map(|component| u32::from(component.open_boundary_edge_count))
            .sum::<u32>();
        let expected_cycle = components
            .iter()
            .map(|component| u32::from(component.cycle_rank))
            .sum::<u32>();
        let expected_bridges = components
            .iter()
            .map(|component| u32::from(component.bridge_count))
            .sum::<u32>();
        let expected_articulations = components
            .iter()
            .map(|component| u32::from(component.articulation_count))
            .sum::<u32>();
        let actual = &encoding.habitat[terrain as usize];
        metrics.topology_decoder_checks += 1;
        if actual.node_count != u16::try_from(expected_nodes)?
            || actual.edge_count != u16::try_from(expected_edges)?
            || actual.component_count != u16::try_from(components.len())?
            || actual.boundary_half_edges != u16::try_from(expected_boundary)?
            || actual.cycle_rank != u16::try_from(expected_cycle)?
            || actual.bridge_count != u16::try_from(expected_bridges)?
            || actual.articulation_count != u16::try_from(expected_articulations)?
        {
            metrics.topology_decoder_failures += 1;
        }
    }

    for species in Wildlife::ALL {
        let motifs = state
            .wildlife_motifs
            .iter()
            .filter(|motif| motif.relative_seat == board.relative_seat && motif.wildlife == species)
            .collect::<Vec<_>>();
        let expected_edges = motifs
            .iter()
            .map(|motif| motif.same_species_neighbor_bits.count_ones())
            .sum::<u32>()
            / 2;
        let actual = &encoding.wildlife[species as usize];
        metrics.topology_decoder_checks += 1;
        if actual.node_count != u16::try_from(motifs.len())?
            || actual.edge_count != u16::try_from(expected_edges)?
        {
            metrics.topology_decoder_failures += 1;
        }
    }
    Ok(())
}

fn validate_d6(
    state: &SparsePublicState,
    expected: &S6StateEncoding,
    metrics: &mut SeedMetrics,
) -> Result<()> {
    for transform in D6Transform::ALL {
        let transformed = state.transformed(transform)?;
        let actual = topological_state_encoding(&transformed)?;
        metrics.d6_invariance_checks += 1;
        if actual != *expected {
            metrics.d6_invariance_failures += 1;
        }
    }
    Ok(())
}

fn graph_from_coords(coords: impl IntoIterator<Item = AxialCoord>) -> Result<UndirectedGraph> {
    let coords = coords.into_iter().collect::<BTreeSet<_>>();
    let ordered = coords.iter().copied().collect::<Vec<_>>();
    let index = ordered
        .iter()
        .enumerate()
        .map(|(index, coord)| (*coord, index))
        .collect::<HashMap<_, _>>();
    let mut adjacency = vec![Vec::new(); ordered.len()];
    for (left, coord) in ordered.iter().enumerate() {
        for neighbor in coord.neighbors() {
            if let Some(&right) = index.get(&neighbor) {
                adjacency[left].push(right);
            }
        }
        adjacency[left].sort_unstable();
        adjacency[left].dedup();
    }
    UndirectedGraph::new(ordered, adjacency)
}

fn habitat_graph(tiles: &[&OccupiedTileToken], terrain: Terrain) -> Result<UndirectedGraph> {
    let by_coord = tiles
        .iter()
        .map(|tile| (tile.coord, *tile))
        .collect::<BTreeMap<_, _>>();
    let coords = tiles
        .iter()
        .filter(|tile| tile.terrain_a == terrain || tile.terrain_b == Some(terrain))
        .map(|tile| tile.coord)
        .collect::<BTreeSet<_>>();
    let ordered = coords.iter().copied().collect::<Vec<_>>();
    let index = ordered
        .iter()
        .enumerate()
        .map(|(index, coord)| (*coord, index))
        .collect::<HashMap<_, _>>();
    let mut adjacency = vec![Vec::new(); ordered.len()];
    for (left, coord) in ordered.iter().enumerate() {
        let tile = by_coord
            .get(coord)
            .ok_or_else(|| invalid("S6 habitat graph coordinate lacks a tile"))?;
        for edge in 0..6 {
            if tile.directed_edge_terrains[edge] != terrain {
                continue;
            }
            let neighbor = coord.neighbor(edge);
            let Some(&right) = index.get(&neighbor) else {
                continue;
            };
            let neighbor_tile = by_coord
                .get(&neighbor)
                .ok_or_else(|| invalid("S6 habitat neighbor lacks a tile"))?;
            if neighbor_tile.directed_edge_terrains[(edge + 3) % 6] == terrain {
                adjacency[left].push(right);
            }
        }
        adjacency[left].sort_unstable();
        adjacency[left].dedup();
    }
    UndirectedGraph::new(ordered, adjacency)
}

impl UndirectedGraph {
    fn new(coords: Vec<AxialCoord>, mut adjacency: Vec<Vec<usize>>) -> Result<Self> {
        if coords.len() != adjacency.len() {
            return Err(invalid("S6 graph coordinate and adjacency counts differ"));
        }
        let mut unique = coords.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != coords.len() {
            return Err(invalid("S6 graph coordinates are not unique"));
        }
        for (node, neighbors) in adjacency.iter_mut().enumerate() {
            neighbors.sort_unstable();
            neighbors.dedup();
            if neighbors
                .iter()
                .any(|neighbor| *neighbor >= coords.len() || *neighbor == node)
            {
                return Err(invalid("S6 graph contains an invalid edge"));
            }
        }
        for (node, neighbors) in adjacency.iter().enumerate() {
            for &neighbor in neighbors {
                if adjacency[neighbor].binary_search(&node).is_err() {
                    return Err(invalid("S6 graph adjacency is not symmetric"));
                }
            }
        }
        Ok(Self { coords, adjacency })
    }

    fn from_adjacency(adjacency: &[&[usize]]) -> Result<Self> {
        let coords = (0..adjacency.len())
            .map(|index| AxialCoord::new(i16::try_from(index).expect("small test graph"), 0))
            .collect::<Vec<_>>();
        Self::new(coords, adjacency.iter().map(|row| row.to_vec()).collect())
    }
}

fn analyze_graph(
    graph: &UndirectedGraph,
    boundary_half_edges: u16,
    hole_count: u16,
) -> Result<GraphChannelEncoding> {
    let node_count = u16::try_from(graph.adjacency.len())?;
    let degree_sum = graph
        .adjacency
        .iter()
        .map(|neighbors| neighbors.len() as u32)
        .sum::<u32>();
    if degree_sum % 2 != 0 {
        return Err(invalid("S6 undirected degree sum is odd"));
    }
    let edge_count = u16::try_from(degree_sum / 2)?;
    let component_count = component_count(&graph.adjacency)?;
    let cycle_rank = edge_count
        .checked_add(component_count)
        .and_then(|value| value.checked_sub(node_count))
        .ok_or_else(|| invalid("S6 cycle rank underflowed"))?;
    let (bridge_count, articulation_count) = bridge_articulation_counts(&graph.adjacency)?;
    let distance = distance_summary(&graph.adjacency)?;
    let mut degree_histogram = [0u16; 7];
    for neighbors in &graph.adjacency {
        let degree = neighbors.len();
        if degree >= degree_histogram.len() {
            return Err(invalid("S6 hex graph degree exceeds six"));
        }
        degree_histogram[degree] += 1;
    }
    Ok(GraphChannelEncoding {
        node_count,
        edge_count,
        component_count,
        boundary_half_edges,
        cycle_rank,
        bridge_count,
        articulation_count,
        hole_count,
        diameter: distance.diameter,
        maximum_component_radius: distance.maximum_component_radius,
        reachable_pair_count: distance.reachable_pair_count,
        distance_sum: distance.distance_sum,
        degree_histogram,
        random_walk_return_ppm: random_walk_return_ppm(&graph.adjacency)?,
        laplacian_trace_moments: laplacian_trace_moments(&graph.adjacency)?,
    })
}

#[derive(Debug)]
struct DistanceSummary {
    diameter: u16,
    maximum_component_radius: u16,
    reachable_pair_count: u16,
    distance_sum: u32,
}

fn component_count(adjacency: &[Vec<usize>]) -> Result<u16> {
    let mut seen = vec![false; adjacency.len()];
    let mut count = 0u16;
    for start in 0..adjacency.len() {
        if seen[start] {
            continue;
        }
        count = count
            .checked_add(1)
            .ok_or_else(|| invalid("S6 component count overflowed"))?;
        seen[start] = true;
        let mut queue = VecDeque::from([start]);
        while let Some(node) = queue.pop_front() {
            for &neighbor in &adjacency[node] {
                if !seen[neighbor] {
                    seen[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
    }
    Ok(count)
}

fn bridge_articulation_counts(adjacency: &[Vec<usize>]) -> Result<(u16, u16)> {
    struct Tarjan<'a> {
        adjacency: &'a [Vec<usize>],
        time: usize,
        discovery: Vec<usize>,
        low: Vec<usize>,
        parent: Vec<Option<usize>>,
        articulation: Vec<bool>,
        bridges: usize,
    }
    impl Tarjan<'_> {
        fn visit(&mut self, node: usize) {
            self.time += 1;
            self.discovery[node] = self.time;
            self.low[node] = self.time;
            let mut children = 0;
            for &neighbor in &self.adjacency[node] {
                if self.discovery[neighbor] == 0 {
                    children += 1;
                    self.parent[neighbor] = Some(node);
                    self.visit(neighbor);
                    self.low[node] = self.low[node].min(self.low[neighbor]);
                    if self.parent[node].is_none() && children > 1 {
                        self.articulation[node] = true;
                    }
                    if self.parent[node].is_some() && self.low[neighbor] >= self.discovery[node] {
                        self.articulation[node] = true;
                    }
                    if self.low[neighbor] > self.discovery[node] {
                        self.bridges += 1;
                    }
                } else if self.parent[node] != Some(neighbor) {
                    self.low[node] = self.low[node].min(self.discovery[neighbor]);
                }
            }
        }
    }
    let mut state = Tarjan {
        adjacency,
        time: 0,
        discovery: vec![0; adjacency.len()],
        low: vec![0; adjacency.len()],
        parent: vec![None; adjacency.len()],
        articulation: vec![false; adjacency.len()],
        bridges: 0,
    };
    for node in 0..adjacency.len() {
        if state.discovery[node] == 0 {
            state.visit(node);
        }
    }
    Ok((
        u16::try_from(state.bridges)?,
        u16::try_from(
            state
                .articulation
                .into_iter()
                .filter(|value| *value)
                .count(),
        )?,
    ))
}

fn distance_summary(adjacency: &[Vec<usize>]) -> Result<DistanceSummary> {
    let mut all_distances = Vec::with_capacity(adjacency.len());
    for start in 0..adjacency.len() {
        all_distances.push(bfs_distances(adjacency, start));
    }
    let mut diameter = 0u16;
    let mut reachable_pair_count = 0u16;
    let mut distance_sum = 0u32;
    for (left, distances) in all_distances.iter().enumerate() {
        for distance in distances.iter().skip(left + 1) {
            let Some(distance) = *distance else {
                continue;
            };
            diameter = diameter.max(distance);
            reachable_pair_count = reachable_pair_count
                .checked_add(1)
                .ok_or_else(|| invalid("S6 reachable-pair count overflowed"))?;
            distance_sum = distance_sum
                .checked_add(u32::from(distance))
                .ok_or_else(|| invalid("S6 distance sum overflowed"))?;
        }
    }
    let mut seen = vec![false; adjacency.len()];
    let mut maximum_component_radius = 0u16;
    for start in 0..adjacency.len() {
        if seen[start] {
            continue;
        }
        let members = all_distances[start]
            .iter()
            .enumerate()
            .filter_map(|(index, distance)| distance.map(|_| index))
            .collect::<Vec<_>>();
        for &member in &members {
            seen[member] = true;
        }
        let radius = members
            .iter()
            .map(|&center| {
                members
                    .iter()
                    .filter_map(|&member| all_distances[center][member])
                    .max()
                    .unwrap_or(0)
            })
            .min()
            .unwrap_or(0);
        maximum_component_radius = maximum_component_radius.max(radius);
    }
    Ok(DistanceSummary {
        diameter,
        maximum_component_radius,
        reachable_pair_count,
        distance_sum,
    })
}

fn bfs_distances(adjacency: &[Vec<usize>], start: usize) -> Vec<Option<u16>> {
    let mut distances = vec![None; adjacency.len()];
    distances[start] = Some(0);
    let mut queue = VecDeque::from([start]);
    while let Some(node) = queue.pop_front() {
        let next = distances[node].expect("queued node has a distance") + 1;
        for &neighbor in &adjacency[node] {
            if distances[neighbor].is_none() {
                distances[neighbor] = Some(next);
                queue.push_back(neighbor);
            }
        }
    }
    distances
}

fn random_walk_return_ppm(adjacency: &[Vec<usize>]) -> Result<[u32; 4]> {
    if adjacency.is_empty() {
        return Ok([0; 4]);
    }
    let size = adjacency.len();
    let mut transition = vec![vec![0u64; size]; size];
    for (node, neighbors) in adjacency.iter().enumerate() {
        if neighbors.is_empty() {
            transition[node][node] = WALK_SCALE;
            continue;
        }
        let weight = WALK_SCALE / u64::try_from(neighbors.len())?;
        for &neighbor in neighbors {
            transition[node][neighbor] = weight;
        }
    }
    let mut power = transition.clone();
    let mut result = [0u32; 4];
    let mut result_index = 0;
    for step in 1..=*WALK_STEPS.last().expect("walk steps are nonempty") {
        if step > 1 {
            power = scaled_matrix_multiply(&power, &transition)?;
        }
        if WALK_STEPS.contains(&step) {
            let trace = (0..size)
                .map(|index| u128::from(power[index][index]))
                .sum::<u128>();
            result[result_index] =
                u32::try_from(trace / u128::try_from(size).expect("usize fits u128"))?;
            result_index += 1;
        }
    }
    Ok(result)
}

fn scaled_matrix_multiply(left: &[Vec<u64>], right: &[Vec<u64>]) -> Result<Vec<Vec<u64>>> {
    let size = left.len();
    if right.len() != size
        || left.iter().any(|row| row.len() != size)
        || right.iter().any(|row| row.len() != size)
    {
        return Err(invalid("S6 scaled matrix dimensions are not square"));
    }
    let mut output = vec![vec![0u64; size]; size];
    for row in 0..size {
        for column in 0..size {
            let mut sum = 0u128;
            for inner in 0..size {
                sum = sum
                    .checked_add(
                        u128::from(left[row][inner])
                            .checked_mul(u128::from(right[inner][column]))
                            .ok_or_else(|| invalid("S6 random-walk product overflowed"))?,
                    )
                    .ok_or_else(|| invalid("S6 random-walk sum overflowed"))?;
            }
            output[row][column] =
                u64::try_from((sum + u128::from(WALK_SCALE / 2)) / u128::from(WALK_SCALE))?;
        }
    }
    Ok(output)
}

fn laplacian_trace_moments(adjacency: &[Vec<usize>]) -> Result<[u64; LAPLACIAN_MOMENTS]> {
    if adjacency.is_empty() {
        return Ok([0; LAPLACIAN_MOMENTS]);
    }
    let size = adjacency.len();
    let mut laplacian = vec![vec![0i128; size]; size];
    for (node, neighbors) in adjacency.iter().enumerate() {
        laplacian[node][node] = i128::try_from(neighbors.len())?;
        for &neighbor in neighbors {
            laplacian[node][neighbor] = -1;
        }
    }
    let mut power = laplacian.clone();
    let mut moments = [0u64; LAPLACIAN_MOMENTS];
    for moment in &mut moments {
        let trace = (0..size)
            .map(|index| power[index][index])
            .try_fold(0i128, |sum, value| {
                sum.checked_add(value)
                    .ok_or_else(|| invalid("S6 Laplacian trace overflowed"))
            })?;
        *moment = u64::try_from(trace)
            .map_err(|_| invalid("S6 Laplacian trace is negative or too large"))?;
        power = integer_matrix_multiply(&power, &laplacian)?;
    }
    Ok(moments)
}

fn integer_matrix_multiply(left: &[Vec<i128>], right: &[Vec<i128>]) -> Result<Vec<Vec<i128>>> {
    let size = left.len();
    if right.len() != size
        || left.iter().any(|row| row.len() != size)
        || right.iter().any(|row| row.len() != size)
    {
        return Err(invalid("S6 integer matrix dimensions are not square"));
    }
    let mut output = vec![vec![0i128; size]; size];
    for row in 0..size {
        for column in 0..size {
            let mut sum = 0i128;
            for inner in 0..size {
                sum = sum
                    .checked_add(
                        left[row][inner]
                            .checked_mul(right[inner][column])
                            .ok_or_else(|| invalid("S6 Laplacian product overflowed"))?,
                    )
                    .ok_or_else(|| invalid("S6 Laplacian sum overflowed"))?;
            }
            output[row][column] = sum;
        }
    }
    Ok(output)
}

fn boundary_half_edges(graph: &UndirectedGraph) -> Result<u16> {
    let nodes = u32::try_from(graph.adjacency.len())?;
    let degree_sum = graph
        .adjacency
        .iter()
        .map(|neighbors| u32::try_from(neighbors.len()))
        .collect::<std::result::Result<Vec<_>, _>>()?
        .into_iter()
        .sum::<u32>();
    u16::try_from(
        nodes
            .checked_mul(6)
            .and_then(|value| value.checked_sub(degree_sum))
            .ok_or_else(|| invalid("S6 boundary half-edge count underflowed"))?,
    )
    .map_err(Into::into)
}

fn geometric_hole_count(coords: &[AxialCoord]) -> Result<u16> {
    if coords.is_empty() {
        return Ok(0);
    }
    let occupied = coords.iter().copied().collect::<BTreeSet<_>>();
    let minimum_q = coords
        .iter()
        .map(|coord| coord.q)
        .min()
        .expect("nonempty coordinates")
        .checked_sub(1)
        .ok_or_else(|| invalid("S6 hole bounding box q underflowed"))?;
    let maximum_q = coords
        .iter()
        .map(|coord| coord.q)
        .max()
        .expect("nonempty coordinates")
        .checked_add(1)
        .ok_or_else(|| invalid("S6 hole bounding box q overflowed"))?;
    let minimum_r = coords
        .iter()
        .map(|coord| coord.r)
        .min()
        .expect("nonempty coordinates")
        .checked_sub(1)
        .ok_or_else(|| invalid("S6 hole bounding box r underflowed"))?;
    let maximum_r = coords
        .iter()
        .map(|coord| coord.r)
        .max()
        .expect("nonempty coordinates")
        .checked_add(1)
        .ok_or_else(|| invalid("S6 hole bounding box r overflowed"))?;

    let mut remaining = BTreeSet::new();
    for q in minimum_q..=maximum_q {
        for r in minimum_r..=maximum_r {
            let coord = AxialCoord::new(q, r);
            if !occupied.contains(&coord) {
                remaining.insert(coord);
            }
        }
    }
    let mut holes = 0u16;
    while let Some(start) = remaining.iter().next().copied() {
        remaining.remove(&start);
        let mut queue = VecDeque::from([start]);
        let mut exterior = false;
        while let Some(coord) = queue.pop_front() {
            if coord.q == minimum_q
                || coord.q == maximum_q
                || coord.r == minimum_r
                || coord.r == maximum_r
            {
                exterior = true;
            }
            for neighbor in coord.neighbors() {
                if neighbor.q < minimum_q
                    || neighbor.q > maximum_q
                    || neighbor.r < minimum_r
                    || neighbor.r > maximum_r
                {
                    exterior = true;
                } else if remaining.remove(&neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }
        if !exterior {
            holes = holes
                .checked_add(1)
                .ok_or_else(|| invalid("S6 hole count overflowed"))?;
        }
    }
    Ok(holes)
}

fn eligible_empty_coords(tiles: &[&OccupiedTileToken], wildlife: Wildlife) -> BTreeSet<AxialCoord> {
    tiles
        .iter()
        .filter(|tile| {
            tile.placed_wildlife.is_none() && tile.wildlife_eligibility.contains(wildlife)
        })
        .map(|tile| tile.coord)
        .collect()
}

fn marked_path_summary(
    graph: &UndirectedGraph,
    marked: &BTreeSet<AxialCoord>,
) -> Result<MarkedPathSummary> {
    let index = graph
        .coords
        .iter()
        .enumerate()
        .map(|(index, coord)| (*coord, index))
        .collect::<HashMap<_, _>>();
    let marked = marked
        .iter()
        .filter_map(|coord| index.get(coord).copied())
        .collect::<Vec<_>>();
    marked_path_summary_indices(&graph.adjacency, &marked)
}

fn marked_path_summary_indices(
    adjacency: &[Vec<usize>],
    marked: &[usize],
) -> Result<MarkedPathSummary> {
    let mut marked = marked.to_vec();
    marked.sort_unstable();
    marked.dedup();
    if marked.iter().any(|node| *node >= adjacency.len()) {
        return Err(invalid("S6 marked path references an invalid node"));
    }
    let mut distances = Vec::new();
    let mut unreachable = 0u16;
    for (position, &left) in marked.iter().enumerate() {
        let from_left = bfs_distances(adjacency, left);
        for &right in &marked[(position + 1)..] {
            if let Some(distance) = from_left[right] {
                distances.push(distance);
            } else {
                unreachable = unreachable
                    .checked_add(1)
                    .ok_or_else(|| invalid("S6 marked unreachable count overflowed"))?;
            }
        }
    }
    distances.sort_unstable();
    let reachable = u16::try_from(distances.len())?;
    let sum = distances.iter().map(|value| u64::from(*value)).sum::<u64>();
    let mean_distance_milli = if distances.is_empty() {
        0
    } else {
        u32::try_from(
            sum.checked_mul(1_000)
                .ok_or_else(|| invalid("S6 marked path mean overflowed"))?
                / u64::try_from(distances.len())?,
        )?
    };
    Ok(MarkedPathSummary {
        marked_node_count: u16::try_from(marked.len())?,
        reachable_pair_count: reachable,
        unreachable_pair_count: unreachable,
        minimum_distance: distances.first().copied().unwrap_or(0),
        median_distance: distances
            .get(distances.len().saturating_sub(1) / 2)
            .copied()
            .unwrap_or(0),
        maximum_distance: distances.last().copied().unwrap_or(0),
        mean_distance_milli,
    })
}

fn collision_observation(
    board: &BoardGraph,
    encoding: &TopologicalBoardEncoding,
) -> Result<CollisionObservation> {
    let channels = all_channels(encoding);
    let topology = channels
        .iter()
        .map(|channel| scalar_projection(channel))
        .collect::<Vec<_>>();
    let paths = (
        &encoding.elk_endpoint_paths,
        &encoding.salmon_continuation_paths,
    );
    let random_walk = channels
        .iter()
        .map(|channel| channel.random_walk_return_ppm)
        .collect::<Vec<_>>();
    let spectral = channels
        .iter()
        .map(|channel| channel.laplacian_trace_moments)
        .collect::<Vec<_>>();
    Ok(CollisionObservation {
        baseline_hash: canonical_blake3(&baseline_signature(board))?,
        topology_hash: canonical_blake3(&topology)?,
        path_hash: canonical_blake3(&paths)?,
        random_walk_hash: canonical_blake3(&random_walk)?,
        spectral_hash: canonical_blake3(&spectral)?,
        full_hash: full_feature_hash(encoding)?,
        long_range: is_long_range(encoding),
    })
}

fn full_feature_hash(encoding: &TopologicalBoardEncoding) -> Result<String> {
    canonical_blake3(&(
        &encoding.occupancy,
        &encoding.habitat,
        &encoding.wildlife,
        &encoding.elk_endpoint_paths,
        &encoding.salmon_continuation_paths,
    ))
}

fn all_channels(encoding: &TopologicalBoardEncoding) -> Vec<&GraphChannelEncoding> {
    std::iter::once(&encoding.occupancy)
        .chain(encoding.habitat.iter())
        .chain(encoding.wildlife.iter())
        .collect()
}

fn scalar_projection(channel: &GraphChannelEncoding) -> ScalarProjection {
    ScalarProjection {
        node_count: channel.node_count,
        edge_count: channel.edge_count,
        component_count: channel.component_count,
        boundary_half_edges: channel.boundary_half_edges,
        cycle_rank: channel.cycle_rank,
        bridge_count: channel.bridge_count,
        articulation_count: channel.articulation_count,
        hole_count: channel.hole_count,
        diameter: channel.diameter,
        maximum_component_radius: channel.maximum_component_radius,
        reachable_pair_count: channel.reachable_pair_count,
        distance_sum: channel.distance_sum,
        degree_histogram: channel.degree_histogram,
    }
}

fn legacy_scalar_projection(channel: &GraphChannelEncoding) -> LegacyScalarProjection {
    LegacyScalarProjection {
        node_count: channel.node_count,
        edge_count: channel.edge_count,
        component_count: channel.component_count,
        boundary_half_edges: channel.boundary_half_edges,
        cycle_rank: channel.cycle_rank,
        bridge_count: channel.bridge_count,
        articulation_count: channel.articulation_count,
        degree_histogram: channel.degree_histogram,
    }
}

fn baseline_signature(board: &BoardGraph) -> BaselineBoardSignature {
    let mut habitat = board
        .habitat_components
        .iter()
        .map(|component| BaselineHabitatComponent {
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
    BaselineBoardSignature {
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

fn is_long_range(encoding: &TopologicalBoardEncoding) -> bool {
    encoding.elk_endpoint_paths.maximum_distance >= 4
        || encoding.salmon_continuation_paths.maximum_distance >= 4
        || encoding.occupancy.diameter >= 8
}

fn collision_metrics(observations: &[CollisionObservation]) -> Result<CollisionMetrics> {
    let mut groups: BTreeMap<&str, Vec<&CollisionObservation>> = BTreeMap::new();
    for observation in observations {
        groups
            .entry(observation.baseline_hash.as_str())
            .or_default()
            .push(observation);
    }
    let mut metrics = CollisionMetrics::default();
    for group in groups.values() {
        for left in 0..group.len() {
            for right in (left + 1)..group.len() {
                let left = group[left];
                let right = group[right];
                metrics.baseline_pairs = checked_increment(metrics.baseline_pairs)?;
                metrics.topology_separated += u64::from(left.topology_hash != right.topology_hash);
                metrics.path_separated += u64::from(left.path_hash != right.path_hash);
                metrics.random_walk_separated +=
                    u64::from(left.random_walk_hash != right.random_walk_hash);
                metrics.spectral_separated += u64::from(left.spectral_hash != right.spectral_hash);
                let separated = left.full_hash != right.full_hash;
                metrics.full_separated += u64::from(separated);
                if left.long_range && right.long_range {
                    metrics.long_range_pairs = checked_increment(metrics.long_range_pairs)?;
                    metrics.long_range_separated += u64::from(separated);
                }
            }
        }
    }
    Ok(metrics)
}

fn feature_variation_metrics(
    observations: &[CollisionObservation],
) -> Result<FeatureVariationMetrics> {
    let topology = observations
        .iter()
        .map(|observation| observation.topology_hash.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let path = observations
        .iter()
        .map(|observation| observation.path_hash.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let random_walk = observations
        .iter()
        .map(|observation| observation.random_walk_hash.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let spectral = observations
        .iter()
        .map(|observation| observation.spectral_hash.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let full = observations
        .iter()
        .map(|observation| observation.full_hash.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    Ok(FeatureVariationMetrics {
        topology: u64::try_from(topology)?,
        path: u64::try_from(path)?,
        random_walk: u64::try_from(random_walk)?,
        spectral: u64::try_from(spectral)?,
        full: u64::try_from(full)?,
    })
}

fn checked_increment(value: u64) -> Result<u64> {
    value
        .checked_add(1)
        .ok_or_else(|| invalid("S6 collision pair count overflowed"))
}

fn adversarial_suite() -> Result<AdversarialSummary> {
    let mut result = AdversarialSummary {
        checks: 4,
        failures: 0,
    };

    let ring = AxialCoord::ORIGIN.neighbors().to_vec();
    let line = (0..6).map(|q| AxialCoord::new(q, 0)).collect::<Vec<_>>();
    result.failures +=
        u64::from(geometric_hole_count(&ring)? != 1 || geometric_hole_count(&line)? != 0);

    let path_rows: &[&[usize]] = &[&[1], &[0, 2], &[1, 3], &[2, 4], &[3, 5], &[4]];
    let path = UndirectedGraph::from_adjacency(path_rows)?;
    let near = marked_path_summary_indices(&path.adjacency, &[0, 1])?;
    let far = marked_path_summary_indices(&path.adjacency, &[0, 5])?;
    result.failures +=
        u64::from(near.maximum_distance != 1 || far.maximum_distance != 5 || near == far);

    let first_rows: &[&[usize]] = &[&[2, 4, 5], &[2, 3], &[0, 1], &[1], &[0], &[0]];
    let second_rows: &[&[usize]] = &[&[1, 2, 5], &[0, 4], &[0, 3], &[2], &[1], &[0]];
    let first = UndirectedGraph::from_adjacency(first_rows)?;
    let second = UndirectedGraph::from_adjacency(second_rows)?;
    let first_encoding = analyze_graph(&first, boundary_half_edges(&first)?, 0)?;
    let second_encoding = analyze_graph(&second, boundary_half_edges(&second)?, 0)?;
    result.failures += u64::from(
        legacy_scalar_projection(&first_encoding) != legacy_scalar_projection(&second_encoding)
            || first_encoding.random_walk_return_ppm == second_encoding.random_walk_return_ppm,
    );
    result.failures += u64::from(
        legacy_scalar_projection(&first_encoding) != legacy_scalar_projection(&second_encoding)
            || first_encoding.laplacian_trace_moments == second_encoding.laplacian_trace_moments,
    );
    Ok(result)
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameConfig, GameSeed};

    use super::*;

    #[test]
    fn ring_has_one_geometric_hole() {
        let ring = AxialCoord::ORIGIN.neighbors().to_vec();
        assert_eq!(geometric_hole_count(&ring).unwrap(), 1);
        let mut filled = ring;
        filled.push(AxialCoord::ORIGIN);
        assert_eq!(geometric_hole_count(&filled).unwrap(), 0);
    }

    #[test]
    fn marked_paths_distinguish_near_and_far_endpoints() {
        let rows: &[&[usize]] = &[&[1], &[0, 2], &[1, 3], &[2, 4], &[3, 5], &[4]];
        let graph = UndirectedGraph::from_adjacency(rows).unwrap();
        let near = marked_path_summary_indices(&graph.adjacency, &[0, 1]).unwrap();
        let far = marked_path_summary_indices(&graph.adjacency, &[0, 5]).unwrap();
        assert_eq!(near.maximum_distance, 1);
        assert_eq!(far.maximum_distance, 5);
    }

    #[test]
    fn walk_and_laplacian_separate_a_scalar_topology_collision() {
        let first_rows: &[&[usize]] = &[&[2, 4, 5], &[2, 3], &[0, 1], &[1], &[0], &[0]];
        let second_rows: &[&[usize]] = &[&[1, 2, 5], &[0, 4], &[0, 3], &[2], &[1], &[0]];
        let first = UndirectedGraph::from_adjacency(first_rows).unwrap();
        let second = UndirectedGraph::from_adjacency(second_rows).unwrap();
        let first = analyze_graph(&first, boundary_half_edges(&first).unwrap(), 0).unwrap();
        let second = analyze_graph(&second, boundary_half_edges(&second).unwrap(), 0).unwrap();
        assert_eq!(
            legacy_scalar_projection(&first),
            legacy_scalar_projection(&second)
        );
        assert_ne!(first.random_walk_return_ppm, second.random_walk_return_ppm);
        assert_ne!(
            first.laplacian_trace_moments,
            second.laplacian_trace_moments
        );
    }

    #[test]
    fn state_encoding_is_invariant_under_all_d6_transforms() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(73),
        )
        .unwrap();
        let trunk = PublicStateTrunk::observe(&game, 73).unwrap();
        let expected = topological_state_encoding(&trunk.sparse).unwrap();
        for transform in D6Transform::ALL {
            let transformed = trunk.sparse.transformed(transform).unwrap();
            assert_eq!(topological_state_encoding(&transformed).unwrap(), expected);
        }
    }

    #[test]
    fn novelty_hash_excludes_relative_seat_identity() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(74),
        )
        .unwrap();
        let trunk = PublicStateTrunk::observe(&game, 74).unwrap();
        let encoding = topological_state_encoding(&trunk.sparse).unwrap();
        let first = &encoding.boards[0];
        let mut relabeled = first.clone();
        relabeled.relative_seat = 3;
        assert_eq!(
            full_feature_hash(first).unwrap(),
            full_feature_hash(&relabeled).unwrap()
        );
    }

    #[test]
    fn registered_adversarial_suite_passes() {
        let summary = adversarial_suite().unwrap();
        assert_eq!(summary.checks, 4);
        assert_eq!(summary.failures, 0);
    }
}
