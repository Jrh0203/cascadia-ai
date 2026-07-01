use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use cascadia_game::{Terrain, Wildlife};
use r2_sparse_entity_census::{
    AxialCoord, FrontierToken, HabitatComponentToken, OccupiedTileToken, SparsePublicState,
};
use serde::{Deserialize, Serialize};

use crate::{Result, invalid};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CardAScoreAnatomy {
    pub habitat: [u16; 5],
    pub wildlife: [u16; 5],
    pub nature_tokens: u16,
    pub base_total: u16,
}

impl CardAScoreAnatomy {
    pub fn delta(self, before: Self) -> [i16; 12] {
        let mut result = [0i16; 12];
        for index in 0..5 {
            result[index] = self.habitat[index] as i16 - before.habitat[index] as i16;
            result[5 + index] = self.wildlife[index] as i16 - before.wildlife[index] as i16;
        }
        result[10] = self.nature_tokens as i16 - before.nature_tokens as i16;
        result[11] = self.base_total as i16 - before.base_total as i16;
        result
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HabitatComponentGraph {
    pub component_id: u16,
    pub terrain: Terrain,
    pub members: Vec<AxialCoord>,
    pub member_count: u16,
    pub matching_internal_edge_count: u16,
    pub open_boundary_edge_count: u16,
    pub frontier_contact_count: u16,
    pub cycle_rank: u16,
    pub bridge_count: u16,
    pub articulation_count: u16,
    pub size_rank: u16,
    pub merge_frontier_count: u16,
    pub largest_merge_result: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WildlifeComponentGraph {
    pub members: Vec<AxialCoord>,
    pub edge_count: u16,
    pub endpoint_count: u16,
    pub maximum_degree: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElkLineGraph {
    pub axis: u8,
    pub members: Vec<AxialCoord>,
    pub negative_extension: AxialCoord,
    pub positive_extension: AxialCoord,
    pub eligible_extension_count: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SalmonComponentGraph {
    pub members: Vec<AxialCoord>,
    pub edge_count: u16,
    pub endpoint_count: u16,
    pub branch_conflict_count: u16,
    pub valid_run: bool,
    pub legal_continuations: Vec<AxialCoord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoxCenterGraph {
    pub coord: AxialCoord,
    pub neighbor_diversity_mask: u8,
    pub missing_wildlife_mask: u8,
    pub compatible_cells: Vec<AxialCoord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct WildlifeOpportunitySummary {
    pub eligible_empty_cells: [u16; 5],
    pub bear_singletons: u16,
    pub bear_pairs: u16,
    pub bear_oversize_components: u16,
    pub bear_pair_completion_cells: u16,
    pub bear_oversize_risk_cells: u16,
    pub elk_lines_by_length: [u16; 5],
    pub elk_eligible_extensions: u16,
    pub elk_overlapping_members: u16,
    pub salmon_valid_runs: u16,
    pub salmon_invalid_components: u16,
    pub salmon_endpoints: u16,
    pub salmon_branch_conflicts: u16,
    pub salmon_legal_continuations: u16,
    pub hawk_conflict_edges: u16,
    pub hawk_isolated: u16,
    pub hawk_isolated_opportunities: u16,
    pub fox_centers: u16,
    pub fox_diversity_sum: u16,
    pub fox_missing_types: u16,
    pub fox_compatible_cells: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrontierGraphSummary {
    pub frontier_count: u16,
    pub degree_histogram: [u16; 7],
    pub bridge_frontiers_by_terrain: [u16; 5],
    pub repeated_contact_frontiers_by_terrain: [u16; 5],
    pub maximum_resulting_size_by_terrain: [u16; 5],
    pub sum_resulting_size_by_terrain: [u32; 5],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardGraph {
    pub relative_seat: u8,
    pub habitat_components: Vec<HabitatComponentGraph>,
    pub bear_components: Vec<WildlifeComponentGraph>,
    pub elk_lines: Vec<ElkLineGraph>,
    pub salmon_components: Vec<SalmonComponentGraph>,
    pub hawk_positions: Vec<AxialCoord>,
    pub hawk_conflict_edges: Vec<[AxialCoord; 2]>,
    pub fox_centers: Vec<FoxCenterGraph>,
    pub opportunity: WildlifeOpportunitySummary,
    pub frontier: FrontierGraphSummary,
    pub nature_tokens: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationalStateGraph {
    pub boards: Vec<BoardGraph>,
}

impl RelationalStateGraph {
    pub fn from_sparse(state: &SparsePublicState) -> Result<Self> {
        let mut boards = Vec::with_capacity(usize::from(state.global.player_count));
        for relative_seat in 0..state.global.player_count {
            boards.push(BoardGraph::from_sparse(state, relative_seat)?);
        }
        Ok(Self { boards })
    }
}

impl BoardGraph {
    pub fn from_sparse(state: &SparsePublicState, relative_seat: u8) -> Result<Self> {
        let tiles = state
            .occupied_tiles
            .iter()
            .filter(|tile| tile.relative_seat == relative_seat)
            .cloned()
            .collect::<Vec<_>>();
        let frontiers = state
            .legal_frontier
            .iter()
            .filter(|frontier| frontier.relative_seat == relative_seat)
            .cloned()
            .collect::<Vec<_>>();
        let components = state
            .habitat_components
            .iter()
            .filter(|component| component.relative_seat == relative_seat)
            .cloned()
            .collect::<Vec<_>>();
        let player = state
            .players
            .get(usize::from(relative_seat))
            .ok_or_else(|| invalid(format!("missing relative seat {relative_seat}")))?;
        Self::from_parts(
            relative_seat,
            &tiles,
            &frontiers,
            &components,
            player.nature_tokens,
        )
    }

    pub fn from_parts(
        relative_seat: u8,
        tiles: &[OccupiedTileToken],
        frontiers: &[FrontierToken],
        components: &[HabitatComponentToken],
        nature_tokens: u8,
    ) -> Result<Self> {
        if tiles.iter().any(|tile| tile.relative_seat != relative_seat)
            || frontiers
                .iter()
                .any(|frontier| frontier.relative_seat != relative_seat)
            || components
                .iter()
                .any(|component| component.relative_seat != relative_seat)
        {
            return Err(invalid("board graph inputs span multiple relative seats"));
        }
        let by_coord = tiles
            .iter()
            .map(|tile| (tile.coord, tile))
            .collect::<BTreeMap<_, _>>();
        let wildlife_at = tiles
            .iter()
            .filter_map(|tile| tile.placed_wildlife.map(|wildlife| (tile.coord, wildlife)))
            .collect::<BTreeMap<_, _>>();
        let eligible_empty = tiles
            .iter()
            .filter(|tile| tile.placed_wildlife.is_none())
            .collect::<Vec<_>>();

        let habitat_components = rich_habitat_components(components, frontiers, &by_coord)?;
        let bear_positions = wildlife_positions(&wildlife_at, Wildlife::Bear);
        let elk_positions = wildlife_positions(&wildlife_at, Wildlife::Elk);
        let salmon_positions = wildlife_positions(&wildlife_at, Wildlife::Salmon);
        let hawk_positions = wildlife_positions(&wildlife_at, Wildlife::Hawk);
        let fox_positions = wildlife_positions(&wildlife_at, Wildlife::Fox);

        let bear_components = wildlife_components(&bear_positions);
        let elk_lines = elk_lines(&elk_positions, &eligible_empty);
        let salmon_components = salmon_components(&salmon_positions, &eligible_empty);
        let hawk_conflict_edges = adjacency_edges(&hawk_positions);
        let fox_centers = fox_centers(&fox_positions, &wildlife_at, &eligible_empty);
        let opportunity = opportunity_summary(
            &eligible_empty,
            &bear_components,
            &bear_positions,
            &elk_lines,
            &salmon_components,
            &hawk_positions,
            &hawk_conflict_edges,
            &fox_centers,
        );
        let frontier = frontier_summary(frontiers);

        Ok(Self {
            relative_seat,
            habitat_components,
            bear_components,
            elk_lines,
            salmon_components,
            hawk_positions,
            hawk_conflict_edges,
            fox_centers,
            opportunity,
            frontier,
            nature_tokens,
        })
    }

    pub fn score_anatomy(&self) -> CardAScoreAnatomy {
        let mut habitat = [0u16; 5];
        for component in &self.habitat_components {
            habitat[component.terrain as usize] =
                habitat[component.terrain as usize].max(component.member_count);
        }
        let bear_pairs = self
            .bear_components
            .iter()
            .filter(|component| component.members.len() == 2)
            .count();
        let bears = match bear_pairs {
            0 => 0,
            1 => 4,
            2 => 11,
            3 => 19,
            _ => 27,
        };
        let elk_positions = self
            .elk_lines
            .iter()
            .flat_map(|line| line.members.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let elk = score_elk_a(&elk_positions);
        let salmon = self
            .salmon_components
            .iter()
            .filter(|component| component.valid_run)
            .map(|component| salmon_a_score(component.members.len()))
            .sum();
        let isolated_hawks = self
            .hawk_positions
            .iter()
            .filter(|hawk| {
                !self
                    .hawk_conflict_edges
                    .iter()
                    .any(|edge| edge[0] == **hawk || edge[1] == **hawk)
            })
            .count();
        let hawks = hawk_a_score(isolated_hawks);
        let foxes = self
            .fox_centers
            .iter()
            .map(|center| center.neighbor_diversity_mask.count_ones() as u16)
            .sum();
        let wildlife = [bears, elk, salmon, hawks, foxes];
        let nature_tokens = u16::from(self.nature_tokens);
        let base_total = habitat.iter().sum::<u16>() + wildlife.iter().sum::<u16>() + nature_tokens;
        CardAScoreAnatomy {
            habitat,
            wildlife,
            nature_tokens,
            base_total,
        }
    }

    pub fn component_token_count(&self) -> usize {
        self.habitat_components.len()
    }

    pub fn motif_token_count(&self) -> usize {
        self.bear_components.len()
            + self.elk_lines.len()
            + self.salmon_components.len()
            + self.hawk_positions.len()
            + self.fox_centers.len()
    }

    pub fn wildlife_positions(&self, wildlife: Wildlife) -> Vec<AxialCoord> {
        match wildlife {
            Wildlife::Bear => self
                .bear_components
                .iter()
                .flat_map(|component| component.members.iter().copied())
                .collect(),
            Wildlife::Elk => self
                .elk_lines
                .iter()
                .flat_map(|line| line.members.iter().copied())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
            Wildlife::Salmon => self
                .salmon_components
                .iter()
                .flat_map(|component| component.members.iter().copied())
                .collect(),
            Wildlife::Hawk => self.hawk_positions.clone(),
            Wildlife::Fox => self.fox_centers.iter().map(|center| center.coord).collect(),
        }
    }

    pub fn wildlife_scores_with_added(&self, added: Option<(AxialCoord, Wildlife)>) -> [u16; 5] {
        let mut positions: [Vec<AxialCoord>; 5] =
            std::array::from_fn(|index| self.wildlife_positions(Wildlife::ALL[index]));
        if let Some((coord, wildlife)) = added {
            positions[wildlife as usize].push(coord);
            positions[wildlife as usize].sort_unstable();
            positions[wildlife as usize].dedup();
        }
        score_wildlife_positions(&positions)
    }
}

fn rich_habitat_components(
    components: &[HabitatComponentToken],
    frontiers: &[FrontierToken],
    tiles: &BTreeMap<AxialCoord, &OccupiedTileToken>,
) -> Result<Vec<HabitatComponentGraph>> {
    let mut ranks = HashMap::new();
    for terrain in Terrain::ALL {
        let mut ordered = components
            .iter()
            .filter(|component| component.terrain == terrain)
            .collect::<Vec<_>>();
        ordered.sort_by(|left, right| {
            right
                .member_count
                .cmp(&left.member_count)
                .then_with(|| left.members.cmp(&right.members))
        });
        for (index, component) in ordered.into_iter().enumerate() {
            ranks.insert(
                (terrain as u8, component.component_id),
                u16::try_from(index + 1).expect("at most 23 components"),
            );
        }
    }

    let mut result = Vec::with_capacity(components.len());
    for component in components {
        let adjacency = habitat_adjacency(component, tiles)?;
        let (bridge_count, articulation_count) = bridge_articulation_counts(&adjacency);
        let cycle_rank = component
            .matching_internal_edge_count
            .saturating_add(1)
            .saturating_sub(component.member_count);
        let mut merge_frontier_count = 0u16;
        let mut largest_merge_result = component.member_count;
        for frontier in frontiers {
            let touches = frontier
                .touched_habitat_components
                .iter()
                .filter(|touch| touch.terrain == component.terrain)
                .collect::<Vec<_>>();
            if touches.len() < 2
                || !touches
                    .iter()
                    .any(|touch| touch.component_id == component.component_id)
            {
                continue;
            }
            merge_frontier_count += 1;
            largest_merge_result = largest_merge_result
                .max(frontier.resulting_size_by_terrain[component.terrain as usize]);
        }
        result.push(HabitatComponentGraph {
            component_id: component.component_id,
            terrain: component.terrain,
            members: component.members.clone(),
            member_count: component.member_count,
            matching_internal_edge_count: component.matching_internal_edge_count,
            open_boundary_edge_count: component.open_boundary_edge_count,
            frontier_contact_count: component.frontier_contact_count,
            cycle_rank,
            bridge_count,
            articulation_count,
            size_rank: ranks[&(component.terrain as u8, component.component_id)],
            merge_frontier_count,
            largest_merge_result,
        });
    }
    result.sort_by_key(|component| (component.terrain as u8, component.component_id));
    Ok(result)
}

fn habitat_adjacency(
    component: &HabitatComponentToken,
    tiles: &BTreeMap<AxialCoord, &OccupiedTileToken>,
) -> Result<Vec<Vec<usize>>> {
    let index = component
        .members
        .iter()
        .enumerate()
        .map(|(index, coord)| (*coord, index))
        .collect::<HashMap<_, _>>();
    let mut adjacency = vec![Vec::new(); component.members.len()];
    for (left_index, coord) in component.members.iter().enumerate() {
        let tile = tiles
            .get(coord)
            .ok_or_else(|| invalid("habitat component references a missing tile"))?;
        for edge in 0..6 {
            if tile.directed_edge_terrains[edge] != component.terrain {
                continue;
            }
            let neighbor = coord.neighbor(edge);
            let Some(&right_index) = index.get(&neighbor) else {
                continue;
            };
            let right = tiles[&neighbor];
            if right.directed_edge_terrains[(edge + 3) % 6] == component.terrain
                && !adjacency[left_index].contains(&right_index)
            {
                adjacency[left_index].push(right_index);
            }
        }
        adjacency[left_index].sort_unstable();
    }
    Ok(adjacency)
}

fn bridge_articulation_counts(adjacency: &[Vec<usize>]) -> (u16, u16) {
    if adjacency.is_empty() {
        return (0, 0);
    }
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
    let mut tarjan = Tarjan {
        adjacency,
        time: 0,
        discovery: vec![0; adjacency.len()],
        low: vec![0; adjacency.len()],
        parent: vec![None; adjacency.len()],
        articulation: vec![false; adjacency.len()],
        bridges: 0,
    };
    for node in 0..adjacency.len() {
        if tarjan.discovery[node] == 0 {
            tarjan.visit(node);
        }
    }
    (
        tarjan.bridges as u16,
        tarjan
            .articulation
            .into_iter()
            .filter(|value| *value)
            .count() as u16,
    )
}

fn wildlife_positions(
    wildlife_at: &BTreeMap<AxialCoord, Wildlife>,
    target: Wildlife,
) -> Vec<AxialCoord> {
    wildlife_at
        .iter()
        .filter_map(|(coord, wildlife)| (*wildlife == target).then_some(*coord))
        .collect()
}

fn wildlife_components(positions: &[AxialCoord]) -> Vec<WildlifeComponentGraph> {
    let positions = positions.iter().copied().collect::<BTreeSet<_>>();
    let mut remaining = positions.clone();
    let mut result = Vec::new();
    while let Some(start) = remaining.iter().next().copied() {
        remaining.remove(&start);
        let mut queue = VecDeque::from([start]);
        let mut members = Vec::new();
        while let Some(coord) = queue.pop_front() {
            members.push(coord);
            for neighbor in coord.neighbors() {
                if remaining.remove(&neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }
        members.sort_unstable();
        let degrees = members
            .iter()
            .map(|coord| {
                coord
                    .neighbors()
                    .into_iter()
                    .filter(|neighbor| positions.contains(neighbor))
                    .count() as u8
            })
            .collect::<Vec<_>>();
        result.push(WildlifeComponentGraph {
            edge_count: degrees.iter().map(|degree| u16::from(*degree)).sum::<u16>() / 2,
            endpoint_count: degrees.iter().filter(|degree| **degree <= 1).count() as u16,
            maximum_degree: degrees.into_iter().max().unwrap_or(0),
            members,
        });
    }
    result.sort_by(|left, right| left.members.cmp(&right.members));
    result
}

fn elk_lines(positions: &[AxialCoord], eligible_empty: &[&OccupiedTileToken]) -> Vec<ElkLineGraph> {
    let occupied = positions.iter().copied().collect::<BTreeSet<_>>();
    let eligible = eligible_empty
        .iter()
        .filter(|tile| tile.wildlife_eligibility.contains(Wildlife::Elk))
        .map(|tile| tile.coord)
        .collect::<BTreeSet<_>>();
    let directions = [(1i16, 0i16), (1, -1), (0, -1)];
    let mut lines = Vec::new();
    for (axis, (dq, dr)) in directions.into_iter().enumerate() {
        for start in &occupied {
            let previous = AxialCoord::new(start.q - dq, start.r - dr);
            if occupied.contains(&previous) {
                continue;
            }
            let mut members = Vec::new();
            let mut current = *start;
            while occupied.contains(&current) {
                members.push(current);
                current = AxialCoord::new(current.q + dq, current.r + dr);
            }
            let negative_extension = previous;
            let positive_extension = current;
            let eligible_extension_count = u8::from(eligible.contains(&negative_extension))
                + u8::from(eligible.contains(&positive_extension));
            lines.push(ElkLineGraph {
                axis: axis as u8,
                members,
                negative_extension,
                positive_extension,
                eligible_extension_count,
            });
        }
    }
    lines.sort_by_key(|line| (line.axis, line.members[0]));
    lines
}

fn salmon_components(
    positions: &[AxialCoord],
    eligible_empty: &[&OccupiedTileToken],
) -> Vec<SalmonComponentGraph> {
    let base_components = wildlife_components(positions);
    let eligible = eligible_empty
        .iter()
        .filter(|tile| tile.wildlife_eligibility.contains(Wildlife::Salmon))
        .map(|tile| tile.coord)
        .collect::<Vec<_>>();
    let existing = positions.iter().copied().collect::<BTreeSet<_>>();
    let mut result = Vec::with_capacity(base_components.len());
    for component in base_components {
        let member_set = component.members.iter().copied().collect::<BTreeSet<_>>();
        let legal_continuations = eligible
            .iter()
            .copied()
            .filter(|candidate| {
                candidate
                    .neighbors()
                    .into_iter()
                    .any(|neighbor| member_set.contains(&neighbor))
                    && salmon_candidate_is_valid(&existing, *candidate)
            })
            .collect::<Vec<_>>();
        let branch_conflict_count = component
            .members
            .iter()
            .filter(|coord| {
                coord
                    .neighbors()
                    .into_iter()
                    .filter(|neighbor| member_set.contains(neighbor))
                    .count()
                    > 2
            })
            .count() as u16;
        result.push(SalmonComponentGraph {
            members: component.members,
            edge_count: component.edge_count,
            endpoint_count: component.endpoint_count,
            branch_conflict_count,
            valid_run: component.maximum_degree <= 2,
            legal_continuations,
        });
    }
    result
}

fn salmon_candidate_is_valid(existing: &BTreeSet<AxialCoord>, candidate: AxialCoord) -> bool {
    let mut with_candidate = existing.clone();
    with_candidate.insert(candidate);
    let mut component = BTreeSet::new();
    let mut queue = VecDeque::from([candidate]);
    component.insert(candidate);
    while let Some(coord) = queue.pop_front() {
        for neighbor in coord.neighbors() {
            if with_candidate.contains(&neighbor) && component.insert(neighbor) {
                queue.push_back(neighbor);
            }
        }
    }
    component.iter().all(|coord| {
        coord
            .neighbors()
            .into_iter()
            .filter(|neighbor| component.contains(neighbor))
            .count()
            <= 2
    })
}

fn adjacency_edges(positions: &[AxialCoord]) -> Vec<[AxialCoord; 2]> {
    let set = positions.iter().copied().collect::<BTreeSet<_>>();
    let mut edges = Vec::new();
    for left in positions {
        for right in left.neighbors() {
            if *left < right && set.contains(&right) {
                edges.push([*left, right]);
            }
        }
    }
    edges.sort_unstable();
    edges
}

fn fox_centers(
    fox_positions: &[AxialCoord],
    wildlife_at: &BTreeMap<AxialCoord, Wildlife>,
    eligible_empty: &[&OccupiedTileToken],
) -> Vec<FoxCenterGraph> {
    let eligible = eligible_empty
        .iter()
        .map(|tile| (tile.coord, tile.wildlife_eligibility))
        .collect::<BTreeMap<_, _>>();
    fox_positions
        .iter()
        .map(|fox| {
            let mut diversity = 0u8;
            for neighbor in fox.neighbors() {
                if let Some(wildlife) = wildlife_at.get(&neighbor) {
                    diversity |= 1 << *wildlife as u8;
                }
            }
            let missing = 0b1_1111 & !diversity;
            let compatible_cells = fox
                .neighbors()
                .into_iter()
                .filter(|coord| {
                    eligible
                        .get(coord)
                        .is_some_and(|mask| mask.bits() & missing != 0)
                })
                .collect();
            FoxCenterGraph {
                coord: *fox,
                neighbor_diversity_mask: diversity,
                missing_wildlife_mask: missing,
                compatible_cells,
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn opportunity_summary(
    eligible_empty: &[&OccupiedTileToken],
    bear_components: &[WildlifeComponentGraph],
    bear_positions: &[AxialCoord],
    elk_lines: &[ElkLineGraph],
    salmon_components: &[SalmonComponentGraph],
    hawk_positions: &[AxialCoord],
    hawk_conflict_edges: &[[AxialCoord; 2]],
    fox_centers: &[FoxCenterGraph],
) -> WildlifeOpportunitySummary {
    let mut result = WildlifeOpportunitySummary::default();
    for tile in eligible_empty {
        for wildlife in Wildlife::ALL {
            if tile.wildlife_eligibility.contains(wildlife) {
                result.eligible_empty_cells[wildlife as usize] += 1;
            }
        }
    }
    result.bear_singletons = bear_components
        .iter()
        .filter(|component| component.members.len() == 1)
        .count() as u16;
    result.bear_pairs = bear_components
        .iter()
        .filter(|component| component.members.len() == 2)
        .count() as u16;
    result.bear_oversize_components = bear_components
        .iter()
        .filter(|component| component.members.len() > 2)
        .count() as u16;
    let bear_set = bear_positions.iter().copied().collect::<BTreeSet<_>>();
    for tile in eligible_empty
        .iter()
        .filter(|tile| tile.wildlife_eligibility.contains(Wildlife::Bear))
    {
        let adjacent = tile
            .coord
            .neighbors()
            .into_iter()
            .filter(|neighbor| bear_set.contains(neighbor))
            .collect::<Vec<_>>();
        if adjacent.len() == 1
            && bear_components.iter().any(|component| {
                component.members.len() == 1 && component.members[0] == adjacent[0]
            })
        {
            result.bear_pair_completion_cells += 1;
        }
        if adjacent.iter().any(|neighbor| {
            bear_components.iter().any(|component| {
                component.members.len() >= 2 && component.members.contains(neighbor)
            })
        }) || adjacent.len() >= 2
        {
            result.bear_oversize_risk_cells += 1;
        }
    }
    for line in elk_lines {
        let length = line.members.len().min(4);
        result.elk_lines_by_length[length] += 1;
        result.elk_eligible_extensions += u16::from(line.eligible_extension_count);
    }
    let mut elk_membership = BTreeMap::<AxialCoord, u8>::new();
    for line in elk_lines.iter().filter(|line| line.members.len() >= 2) {
        for member in &line.members {
            *elk_membership.entry(*member).or_default() += 1;
        }
    }
    result.elk_overlapping_members = elk_membership
        .values()
        .filter(|memberships| **memberships > 1)
        .count() as u16;
    result.salmon_valid_runs = salmon_components
        .iter()
        .filter(|component| component.valid_run)
        .count() as u16;
    result.salmon_invalid_components = salmon_components
        .iter()
        .filter(|component| !component.valid_run)
        .count() as u16;
    result.salmon_endpoints = salmon_components
        .iter()
        .map(|component| component.endpoint_count)
        .sum();
    result.salmon_branch_conflicts = salmon_components
        .iter()
        .map(|component| component.branch_conflict_count)
        .sum();
    result.salmon_legal_continuations = salmon_components
        .iter()
        .flat_map(|component| component.legal_continuations.iter().copied())
        .collect::<BTreeSet<_>>()
        .len() as u16;
    result.hawk_conflict_edges = hawk_conflict_edges.len() as u16;
    result.hawk_isolated = hawk_positions
        .iter()
        .filter(|hawk| {
            !hawk_conflict_edges
                .iter()
                .any(|edge| edge[0] == **hawk || edge[1] == **hawk)
        })
        .count() as u16;
    let hawk_set = hawk_positions.iter().copied().collect::<BTreeSet<_>>();
    result.hawk_isolated_opportunities = eligible_empty
        .iter()
        .filter(|tile| {
            tile.wildlife_eligibility.contains(Wildlife::Hawk)
                && tile
                    .coord
                    .neighbors()
                    .into_iter()
                    .all(|neighbor| !hawk_set.contains(&neighbor))
        })
        .count() as u16;
    result.fox_centers = fox_centers.len() as u16;
    result.fox_diversity_sum = fox_centers
        .iter()
        .map(|center| center.neighbor_diversity_mask.count_ones() as u16)
        .sum();
    result.fox_missing_types = fox_centers
        .iter()
        .map(|center| center.missing_wildlife_mask.count_ones() as u16)
        .sum();
    result.fox_compatible_cells = fox_centers
        .iter()
        .flat_map(|center| center.compatible_cells.iter().copied())
        .collect::<BTreeSet<_>>()
        .len() as u16;
    result
}

fn frontier_summary(frontiers: &[FrontierToken]) -> FrontierGraphSummary {
    let mut result = FrontierGraphSummary {
        frontier_count: frontiers.len() as u16,
        degree_histogram: [0; 7],
        bridge_frontiers_by_terrain: [0; 5],
        repeated_contact_frontiers_by_terrain: [0; 5],
        maximum_resulting_size_by_terrain: [0; 5],
        sum_resulting_size_by_terrain: [0; 5],
    };
    for frontier in frontiers {
        result.degree_histogram[frontier.neighbor_presence_bits.count_ones() as usize] += 1;
        for terrain in Terrain::ALL {
            let index = terrain as usize;
            if frontier.habitat_bridge_terrain_bits & (1 << index) != 0 {
                result.bridge_frontiers_by_terrain[index] += 1;
            }
            if frontier.repeated_component_contact_terrain_bits & (1 << index) != 0 {
                result.repeated_contact_frontiers_by_terrain[index] += 1;
            }
            let size = frontier.resulting_size_by_terrain[index];
            result.maximum_resulting_size_by_terrain[index] =
                result.maximum_resulting_size_by_terrain[index].max(size);
            result.sum_resulting_size_by_terrain[index] += u32::from(size);
        }
    }
    result
}

fn score_elk_a(positions: &[AxialCoord]) -> u16 {
    if positions.is_empty() {
        return 0;
    }
    let mut groups = Vec::new();
    for index in 0..positions.len() {
        groups.push((1u32 << index, 2u16));
    }
    let directions = [(1i16, 0i16), (1, -1), (0, -1)];
    for (start_index, start) in positions.iter().enumerate() {
        for (dq, dr) in directions {
            let mut mask = 1u32 << start_index;
            let mut current = *start;
            for length in 2..=4 {
                current = AxialCoord::new(current.q + dq, current.r + dr);
                let Some(index) = positions.iter().position(|coord| *coord == current) else {
                    break;
                };
                mask |= 1u32 << index;
                groups.push((
                    mask,
                    match length {
                        2 => 5,
                        3 => 9,
                        _ => 13,
                    },
                ));
            }
        }
    }
    maximize_disjoint_groups(positions.len(), &groups)
}

pub(crate) fn score_wildlife_positions(positions: &[Vec<AxialCoord>; 5]) -> [u16; 5] {
    let bear_components = wildlife_components(&positions[Wildlife::Bear as usize]);
    let bear_pairs = bear_components
        .iter()
        .filter(|component| component.members.len() == 2)
        .count();
    let bears = match bear_pairs {
        0 => 0,
        1 => 4,
        2 => 11,
        3 => 19,
        _ => 27,
    };
    let elk = score_elk_a(&positions[Wildlife::Elk as usize]);
    let salmon = wildlife_components(&positions[Wildlife::Salmon as usize])
        .iter()
        .filter(|component| component.maximum_degree <= 2)
        .map(|component| salmon_a_score(component.members.len()))
        .sum();
    let hawk_edges = adjacency_edges(&positions[Wildlife::Hawk as usize]);
    let hawk_isolated = positions[Wildlife::Hawk as usize]
        .iter()
        .filter(|hawk| {
            !hawk_edges
                .iter()
                .any(|edge| edge[0] == **hawk || edge[1] == **hawk)
        })
        .count();
    let hawks = hawk_a_score(hawk_isolated);
    let wildlife_at = positions
        .iter()
        .enumerate()
        .flat_map(|(index, coords)| {
            coords
                .iter()
                .map(move |coord| (*coord, Wildlife::ALL[index]))
        })
        .collect::<BTreeMap<_, _>>();
    let foxes = positions[Wildlife::Fox as usize]
        .iter()
        .map(|fox| {
            fox.neighbors()
                .into_iter()
                .filter_map(|neighbor| wildlife_at.get(&neighbor).copied())
                .fold(0u8, |mask, wildlife| mask | (1 << wildlife as u8))
                .count_ones() as u16
        })
        .sum();
    [bears, elk, salmon, hawks, foxes]
}

fn maximize_disjoint_groups(n: usize, groups: &[(u32, u16)]) -> u16 {
    let state_count = 1usize << n;
    let mut dp = vec![0u16; state_count];
    for state in 1..state_count {
        let first = state.trailing_zeros();
        for (group, score) in groups {
            if group & (1 << first) != 0 && group & state as u32 == *group {
                dp[state] = dp[state].max(*score + dp[state & !(*group as usize)]);
            }
        }
    }
    dp[state_count - 1]
}

fn salmon_a_score(length: usize) -> u16 {
    match length {
        0 => 0,
        1 => 2,
        2 => 5,
        3 => 8,
        4 => 12,
        5 => 16,
        6 => 20,
        _ => 25,
    }
}

fn hawk_a_score(count: usize) -> u16 {
    match count {
        0 => 0,
        1 => 2,
        2 => 5,
        3 => 8,
        4 => 11,
        5 => 14,
        6 => 18,
        7 => 22,
        _ => 26,
    }
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameConfig, GameSeed, GameState, score_board};
    use r3_action_edit_census::PublicStateTrunk;

    use super::*;

    #[test]
    fn graph_decodes_card_a_scores_across_one_game() {
        let mut game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(91),
        )
        .unwrap();
        while !game.is_game_over() {
            let trunk =
                PublicStateTrunk::observe(&game, u64::from(game.completed_turns())).unwrap();
            let graph = BoardGraph::from_sparse(&trunk.sparse, 0).unwrap();
            let expected = score_board(
                &game.boards()[game.current_player()],
                game.config().scoring_cards,
            );
            let actual = graph.score_anatomy();
            assert_eq!(actual.habitat, expected.habitat);
            assert_eq!(actual.wildlife, expected.wildlife);
            assert_eq!(actual.nature_tokens, expected.nature_tokens);
            assert_eq!(actual.base_total, expected.base_total);
            let (prelude, _) = game.preview_free_three_of_a_kind_if_feasible().unwrap();
            let actions = game.legal_turn_actions(&prelude).unwrap();
            let index = usize::from(game.completed_turns()) % actions.len();
            game.apply(&actions[index]).unwrap();
        }
    }

    #[test]
    fn bridge_and_articulation_handles_path_and_cycle() {
        let path = vec![vec![1], vec![0, 2], vec![1]];
        assert_eq!(bridge_articulation_counts(&path), (2, 1));
        let cycle = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        assert_eq!(bridge_articulation_counts(&cycle), (0, 0));
    }
}
