//! Exact public-information demand/supply objects for O2 research.
//!
//! The graph deliberately contains no learned targets or hidden game state.
//! It is constructed from `PublicGameState`, exact semantic supply, and exact
//! counterfactual rules operations. The matching teacher is exact for this
//! capacitated component graph; it is a factual research label, not a player
//! evaluation function.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use cascadia_game::{
    Board, D6Transform, HexCoord, MarketSlot, PublicGameState, Rotation, STANDARD_TILES, Terrain,
    Tile, Wildlife, score_wildlife_type,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    ExactSemanticSupply, SemanticArchetypeId, SemanticSupplyError, SemanticTileReference,
    standard_semantic_archetype_catalog,
};

pub const OPPORTUNITY_GRAPH_SCHEMA_VERSION: u16 = 1;
pub const OPPORTUNITY_GRAPH_SCHEMA: &str = "opportunity-graph-v1";
const GRAPH_MAGIC: &[u8; 8] = b"CSOPPG1\0";
const SUMMARY_MAGIC: &[u8; 8] = b"CSOPPM1\0";
const TEACHER_SCALE: i64 = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum OpportunityDemandKind {
    WildlifePlacement = 0,
    HabitatFrontier = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum OpportunitySupplyKind {
    MarketWildlife = 0,
    MarketTile = 1,
    UnseenWildlife = 2,
    UnseenTileArchetype = 3,
}

/// A semantic demand identity. Coordinates are part of the public board and
/// transform covariantly under D6; the remaining fields are invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DemandId {
    pub kind: OpportunityDemandKind,
    pub subject: u8,
    pub coord: HexCoord,
}

impl DemandId {
    pub fn transformed(self, transform: D6Transform) -> Result<Self, OpportunityGraphError> {
        Ok(Self {
            coord: transform.transform_coord(self.coord)?,
            ..self
        })
    }
}

/// Supply IDs do not depend on board orientation. Market slots, wildlife
/// species, and semantic archetype IDs are public canonical identities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SupplyId {
    pub kind: OpportunitySupplyKind,
    pub subject: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct OpportunityEdgeId {
    pub demand: DemandId,
    pub supply: SupplyId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpportunityDemand {
    pub id: DemandId,
    pub deadline_turns: u8,
    pub exact_completion_delta: u16,
    pub local_same_species_neighbors: u8,
    pub frontier_neighbor_count: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpportunitySupply {
    pub id: SupplyId,
    pub capacity: u16,
    pub availability_numerator: u16,
    pub availability_denominator: u16,
    pub access_delay_turns: u8,
    pub opponents_before_access: u8,
    pub wildlife: Option<Wildlife>,
    pub archetype_id: Option<SemanticArchetypeId>,
    pub market_slot: Option<MarketSlot>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpportunityEdge {
    pub id: OpportunityEdgeId,
    /// Canonical-archetype rotation mask for tile edges; zero for wildlife.
    pub compatible_rotation_mask: u8,
    pub best_matching_edges: u8,
    pub exact_completion_delta: u16,
    /// Exact integer definition used by the deterministic matching teacher.
    pub teacher_value_micros: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpportunityGraphV1 {
    pub schema_version: u16,
    pub source_public_state_hash: [u8; 32],
    pub focal_seat: u8,
    pub completed_turns: u16,
    pub personal_turns_remaining: u8,
    pub exact_supply_hash: [u8; 32],
    pub demands: Vec<OpportunityDemand>,
    pub supplies: Vec<OpportunitySupply>,
    pub edges: Vec<OpportunityEdge>,
}

/// Draft-level immutable inputs for repeated opportunity extraction over many
/// legal afterstate boards. The market, public supply, deadlines, and opponent
/// access are identical across placement siblings and are compiled once.
#[derive(Debug, Clone)]
pub struct OpportunityGraphBuildContext {
    state: PublicGameState,
    focal_seat: usize,
    personal_turns_remaining: u8,
    exact_supply_hash: [u8; 32],
    supplies: Vec<OpportunitySupply>,
    source_public_state_hash: [u8; 32],
}

/// Tile-sibling wildlife demands grouped by species. A placed wildlife token
/// can only change its own scoring card and fox adjacency; all other species'
/// exact completion deltas remain unchanged apart from the newly occupied
/// coordinate no longer being a legal placement.
#[derive(Debug, Clone)]
pub struct PreparedWildlifeOpportunityDemands {
    by_species: [Vec<OpportunityDemand>; 5],
}

#[derive(Debug, Clone)]
struct MatchingFrontierState {
    teacher_value_micros: i64,
    assignments: Vec<OpportunityAssignment>,
}

/// Exact cardinality frontiers for the five wildlife components and the
/// habitat/tile component. Fields remain private so callers cannot combine
/// incompatible graph generations.
#[derive(Debug, Clone)]
pub struct PreparedOpportunityMatching {
    habitat: Vec<MatchingFrontierState>,
    habitat_demand_ids: Vec<DemandId>,
    personal_turns_remaining: u8,
}

#[derive(Debug, Clone)]
pub struct PreparedWildlifeMatching {
    components: [Vec<MatchingFrontierState>; 5],
    demand_ids: [Vec<DemandId>; 5],
    personal_turns_remaining: u8,
}

impl OpportunityGraphBuildContext {
    pub fn new(state: &PublicGameState, focal_seat: usize) -> Result<Self, OpportunityGraphError> {
        state
            .boards()
            .get(focal_seat)
            .ok_or(OpportunityGraphError::InvalidFocalSeat {
                focal: focal_seat,
                players: state.boards().len(),
            })?;
        let exact = ExactSemanticSupply::from_public_state(state)?;
        let personal_turns_remaining =
            u8::try_from(state.turns_remaining_for_player(focal_seat))
                .map_err(|_| OpportunityGraphError::CountOverflow("personal turns remaining"))?;
        let supplies = build_supplies(state, focal_seat, personal_turns_remaining, &exact)?;
        Ok(Self {
            state: state.clone(),
            focal_seat,
            personal_turns_remaining,
            exact_supply_hash: *exact.canonical_hash().as_bytes(),
            supplies,
            source_public_state_hash: *state.canonical_hash().as_bytes(),
        })
    }

    /// Build the placement-dependent portion. All rows are emitted in the
    /// same canonical order as `OpportunityGraphV1::from_public_state`.
    pub fn build_for_board(
        &self,
        board: &Board,
    ) -> Result<OpportunityGraphV1, OpportunityGraphError> {
        self.build_for_board_filtered(board, None)
    }

    /// Build one independent side of the opportunity graph. Wildlife demands
    /// only connect to wildlife supplies and habitat demands only connect to
    /// tile supplies, so the exact maximum-weight matching is the union of the
    /// two independently solved graphs.
    pub fn build_for_board_kind(
        &self,
        board: &Board,
        kind: OpportunityDemandKind,
    ) -> Result<OpportunityGraphV1, OpportunityGraphError> {
        self.build_for_board_filtered(board, Some(kind))
    }

    fn build_for_board_filtered(
        &self,
        board: &Board,
        kind: Option<OpportunityDemandKind>,
    ) -> Result<OpportunityGraphV1, OpportunityGraphError> {
        static PROFILE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let profile =
            *PROFILE.get_or_init(|| std::env::var_os("CASCADIA_V3_PROFILE_OPPORTUNITY").is_some());
        let started = std::time::Instant::now();
        let mut demands = build_demands(
            &self.state,
            board,
            self.focal_seat,
            self.personal_turns_remaining,
            kind,
        )?;
        let demand_seconds = started.elapsed().as_secs_f64();
        let started = std::time::Instant::now();
        let edges = build_edges(board, &mut demands, &self.supplies)?;
        if profile {
            static PRINTED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            if PRINTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                eprintln!(
                    "V3_OPPORTUNITY_BUILD_PROFILE {}",
                    serde_json::json!({
                        "demand_seconds": demand_seconds,
                        "edge_seconds": started.elapsed().as_secs_f64(),
                        "demands": demands.len(),
                        "edges": edges.len(),
                    })
                );
            }
        }
        Ok(OpportunityGraphV1 {
            schema_version: OPPORTUNITY_GRAPH_SCHEMA_VERSION,
            source_public_state_hash: self.source_public_state_hash,
            focal_seat: u8::try_from(self.focal_seat)
                .map_err(|_| OpportunityGraphError::CountOverflow("focal seat"))?,
            completed_turns: self.state.completed_turns(),
            personal_turns_remaining: self.personal_turns_remaining,
            exact_supply_hash: self.exact_supply_hash,
            demands,
            supplies: self.supplies.clone(),
            edges,
        })
    }

    /// Build and solve a graph created by this validated context without
    /// redundantly reconstructing validation maps on the serving hot path.
    pub fn build_and_solve_for_board(
        &self,
        board: &Board,
    ) -> Result<(OpportunityGraphV1, Vec<OpportunityAssignment>), OpportunityGraphError> {
        let graph = self.build_for_board(board)?;
        let assignments = solve_capacitated_matching(&graph)?;
        Ok((graph, assignments))
    }

    /// Exact kind-filtered counterpart to `build_and_solve_for_board`.
    pub fn build_and_solve_for_board_kind(
        &self,
        board: &Board,
        kind: OpportunityDemandKind,
    ) -> Result<(OpportunityGraphV1, Vec<OpportunityAssignment>), OpportunityGraphError> {
        let graph = self.build_for_board_kind(board, kind)?;
        let assignments = solve_capacitated_matching(&graph)?;
        Ok((graph, assignments))
    }

    /// Reuse the placement-only habitat subgraph for a wildlife sibling, then
    /// run the original single constrained matcher over the exact merged
    /// graph. This preserves the shared total-flow budget; only graph
    /// construction is decomposed.
    pub fn build_and_solve_with_cached_habitat(
        &self,
        board: &Board,
        habitat: &OpportunityGraphV1,
    ) -> Result<(OpportunityGraphV1, Vec<OpportunityAssignment>), OpportunityGraphError> {
        let wildlife =
            self.build_for_board_kind(board, OpportunityDemandKind::WildlifePlacement)?;
        merge_and_solve_opportunity_kinds(wildlife, habitat)
    }

    pub fn prepare_wildlife_demands(
        &self,
        board: &Board,
    ) -> Result<PreparedWildlifeOpportunityDemands, OpportunityGraphError> {
        let mut by_species: [Vec<OpportunityDemand>; 5] = std::array::from_fn(|_| Vec::new());
        for wildlife in Wildlife::ALL {
            by_species[wildlife as usize] = build_wildlife_demands_for_species(
                &self.state,
                board,
                self.personal_turns_remaining,
                wildlife,
            )?;
        }
        Ok(PreparedWildlifeOpportunityDemands { by_species })
    }

    /// Exact sibling evaluator that reuses unchanged species demands and the
    /// cached habitat graph, then executes the original shared matcher.
    pub fn build_and_solve_with_cached_tile_opportunities(
        &self,
        board: &Board,
        habitat: &OpportunityGraphV1,
        prepared_wildlife: &PreparedWildlifeOpportunityDemands,
        placed_wildlife: Option<(Wildlife, HexCoord)>,
    ) -> Result<(OpportunityGraphV1, Vec<OpportunityAssignment>), OpportunityGraphError> {
        let wildlife = self.build_wildlife_graph_with_cached_demands(
            board,
            prepared_wildlife,
            placed_wildlife,
        )?;
        merge_and_solve_opportunity_kinds(wildlife, habitat)
    }

    pub fn prepare_habitat_matching_frontier(
        &self,
        habitat: &OpportunityGraphV1,
    ) -> Result<PreparedOpportunityMatching, OpportunityGraphError> {
        Ok(PreparedOpportunityMatching {
            habitat: matching_component_frontier(habitat, 5)?,
            habitat_demand_ids: habitat.demands.iter().map(|demand| demand.id).collect(),
            personal_turns_remaining: habitat.personal_turns_remaining,
        })
    }

    pub fn prepare_wildlife_matching_frontiers(
        &self,
        board: &Board,
        prepared_wildlife: &PreparedWildlifeOpportunityDemands,
    ) -> Result<PreparedWildlifeMatching, OpportunityGraphError> {
        let graph =
            self.build_wildlife_graph_with_cached_demands(board, prepared_wildlife, None)?;
        let mut components: [Vec<MatchingFrontierState>; 5] = std::array::from_fn(|_| Vec::new());
        for (component, frontier) in components.iter_mut().enumerate() {
            *frontier = matching_component_frontier(&graph, component)?;
        }
        let all_demand_ids = matching_component_demand_ids(&graph);
        Ok(PreparedWildlifeMatching {
            components,
            demand_ids: std::array::from_fn(|component| all_demand_ids[component].clone()),
            personal_turns_remaining: graph.personal_turns_remaining,
        })
    }

    pub fn build_and_solve_with_matching_frontiers(
        &self,
        board: &Board,
        habitat: &OpportunityGraphV1,
        prepared_wildlife: &PreparedWildlifeOpportunityDemands,
        prepared_matching: &PreparedOpportunityMatching,
        prepared_wildlife_matching: &PreparedWildlifeMatching,
        placed_wildlife: Option<(Wildlife, HexCoord)>,
    ) -> Result<(OpportunityGraphV1, Vec<OpportunityAssignment>), OpportunityGraphError> {
        let wildlife = self.build_wildlife_graph_with_cached_demands(
            board,
            prepared_wildlife,
            placed_wildlife,
        )?;
        let graph = merge_opportunity_kinds(wildlife, habitat)?;
        if prepared_matching.personal_turns_remaining != graph.personal_turns_remaining {
            return Err(OpportunityGraphError::FlowInvariant(
                "prepared matching frontier has the wrong turn horizon",
            ));
        }
        if prepared_wildlife_matching.personal_turns_remaining != graph.personal_turns_remaining {
            return Err(OpportunityGraphError::FlowInvariant(
                "prepared wildlife frontier has the wrong turn horizon",
            ));
        }
        let current_demand_ids = matching_component_demand_ids(&graph);
        if current_demand_ids[5] != prepared_matching.habitat_demand_ids {
            return Err(OpportunityGraphError::FlowInvariant(
                "prepared habitat frontier has different demands",
            ));
        }
        let mut wildlife_overrides: [Option<Vec<MatchingFrontierState>>; 5] =
            std::array::from_fn(|_| None);
        for component in 0..5 {
            let score_affected = placed_wildlife.is_some_and(|(wildlife, _)| {
                component == wildlife as usize || component == Wildlife::Fox as usize
            });
            if score_affected
                || current_demand_ids[component] != prepared_wildlife_matching.demand_ids[component]
            {
                wildlife_overrides[component] =
                    Some(matching_component_frontier(&graph, component)?);
            }
        }
        let components: [&[MatchingFrontierState]; 6] = std::array::from_fn(|component| {
            if component == 5 {
                prepared_matching.habitat.as_slice()
            } else {
                wildlife_overrides[component]
                    .as_deref()
                    .unwrap_or(&prepared_wildlife_matching.components[component])
            }
        });
        let assignments = combine_matching_frontiers(
            &components,
            usize::from(graph.personal_turns_remaining).saturating_mul(2),
        )?;
        Ok((graph, assignments))
    }

    fn build_wildlife_graph_with_cached_demands(
        &self,
        board: &Board,
        prepared_wildlife: &PreparedWildlifeOpportunityDemands,
        placed_wildlife: Option<(Wildlife, HexCoord)>,
    ) -> Result<OpportunityGraphV1, OpportunityGraphError> {
        let mut demands = Vec::new();
        for wildlife in Wildlife::ALL {
            let affected = placed_wildlife
                .is_some_and(|(placed, _)| wildlife == placed || wildlife == Wildlife::Fox);
            if affected {
                demands.extend(build_wildlife_demands_for_species(
                    &self.state,
                    board,
                    self.personal_turns_remaining,
                    wildlife,
                )?);
            } else {
                let occupied = placed_wildlife.map(|(_, coord)| coord);
                demands.extend(
                    prepared_wildlife.by_species[wildlife as usize]
                        .iter()
                        .filter(|demand| Some(demand.id.coord) != occupied)
                        .cloned(),
                );
            }
        }
        demands.sort_by_key(|demand| demand.id);
        let edges = build_edges(board, &mut demands, &self.supplies)?;
        let wildlife = OpportunityGraphV1 {
            schema_version: OPPORTUNITY_GRAPH_SCHEMA_VERSION,
            source_public_state_hash: self.source_public_state_hash,
            focal_seat: u8::try_from(self.focal_seat)
                .map_err(|_| OpportunityGraphError::CountOverflow("focal seat"))?,
            completed_turns: self.state.completed_turns(),
            personal_turns_remaining: self.personal_turns_remaining,
            exact_supply_hash: self.exact_supply_hash,
            demands,
            supplies: self.supplies.clone(),
            edges,
        };
        Ok(wildlife)
    }

    pub fn state(&self) -> &PublicGameState {
        &self.state
    }
}

fn merge_and_solve_opportunity_kinds(
    wildlife: OpportunityGraphV1,
    habitat: &OpportunityGraphV1,
) -> Result<(OpportunityGraphV1, Vec<OpportunityAssignment>), OpportunityGraphError> {
    let graph = merge_opportunity_kinds(wildlife, habitat)?;
    let assignments = solve_capacitated_matching(&graph)?;
    Ok((graph, assignments))
}

fn merge_opportunity_kinds(
    wildlife: OpportunityGraphV1,
    habitat: &OpportunityGraphV1,
) -> Result<OpportunityGraphV1, OpportunityGraphError> {
    if wildlife
        .demands
        .iter()
        .any(|demand| demand.id.kind != OpportunityDemandKind::WildlifePlacement)
        || wildlife
            .edges
            .iter()
            .any(|edge| edge.id.demand.kind != OpportunityDemandKind::WildlifePlacement)
        || habitat
            .demands
            .iter()
            .any(|demand| demand.id.kind != OpportunityDemandKind::HabitatFrontier)
        || habitat
            .edges
            .iter()
            .any(|edge| edge.id.demand.kind != OpportunityDemandKind::HabitatFrontier)
        || wildlife.completed_turns != habitat.completed_turns
        || wildlife.personal_turns_remaining != habitat.personal_turns_remaining
        || wildlife.focal_seat != habitat.focal_seat
    {
        return Err(OpportunityGraphError::FlowInvariant(
            "cached habitat graph is incompatible with wildlife sibling",
        ));
    }
    for edge in &habitat.edges {
        let cached = habitat
            .supplies
            .binary_search_by_key(&edge.id.supply, |supply| supply.id)
            .ok()
            .and_then(|index| habitat.supplies.get(index));
        let current = wildlife
            .supplies
            .binary_search_by_key(&edge.id.supply, |supply| supply.id)
            .ok()
            .and_then(|index| wildlife.supplies.get(index));
        if cached != current {
            return Err(OpportunityGraphError::FlowInvariant(
                "cached habitat supply changed across wildlife siblings",
            ));
        }
    }
    let mut demands = wildlife.demands;
    demands.extend_from_slice(&habitat.demands);
    demands.sort_by_key(|demand| demand.id);
    let mut edges = wildlife.edges;
    edges.extend_from_slice(&habitat.edges);
    edges.sort_by_key(|edge| edge.id);
    Ok(OpportunityGraphV1 {
        schema_version: OPPORTUNITY_GRAPH_SCHEMA_VERSION,
        source_public_state_hash: wildlife.source_public_state_hash,
        focal_seat: wildlife.focal_seat,
        completed_turns: wildlife.completed_turns,
        personal_turns_remaining: wildlife.personal_turns_remaining,
        exact_supply_hash: wildlife.exact_supply_hash,
        demands,
        supplies: wildlife.supplies,
        edges,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpportunityAssignment {
    pub demand: DemandId,
    pub supply: SupplyId,
    pub exact_completion_delta: u16,
    pub teacher_value_micros: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpportunityMatchingSummary {
    pub schema_version: u16,
    pub graph_hash: [u8; 32],
    pub demand_count: u32,
    pub supply_count: u32,
    pub edge_count: u32,
    pub matched_demands: u32,
    pub unmatched_demands: u32,
    pub wildlife_matches: u32,
    pub habitat_matches: u32,
    pub market_matches: u32,
    pub unseen_matches: u32,
    pub exact_completion_value: u64,
    pub teacher_value_micros: i64,
    pub assignments: Vec<OpportunityAssignment>,
}

impl OpportunityGraphV1 {
    pub fn from_public_state(
        state: &PublicGameState,
        focal_seat: usize,
    ) -> Result<Self, OpportunityGraphError> {
        let board =
            state
                .boards()
                .get(focal_seat)
                .ok_or(OpportunityGraphError::InvalidFocalSeat {
                    focal: focal_seat,
                    players: state.boards().len(),
                })?;
        let graph = OpportunityGraphBuildContext::new(state, focal_seat)?.build_for_board(board)?;
        graph.validate()?;
        Ok(graph)
    }

    /// Matching assignments without summary hashing. This hot-path API is
    /// semantically identical to `solve_matching().assignments`; callers that
    /// need the evidentiary summary should continue using `solve_matching`.
    pub fn solve_matching_assignments(
        &self,
    ) -> Result<Vec<OpportunityAssignment>, OpportunityGraphError> {
        self.validate()?;
        solve_capacitated_matching(self)
    }

    pub fn validate(&self) -> Result<(), OpportunityGraphError> {
        if self.schema_version != OPPORTUNITY_GRAPH_SCHEMA_VERSION {
            return Err(OpportunityGraphError::InvalidSchema(self.schema_version));
        }
        if !strictly_sorted_unique(self.demands.iter().map(|demand| demand.id)) {
            return Err(OpportunityGraphError::NonCanonical("demand IDs"));
        }
        if !strictly_sorted_unique(self.supplies.iter().map(|supply| supply.id)) {
            return Err(OpportunityGraphError::NonCanonical("supply IDs"));
        }
        if !strictly_sorted_unique(self.edges.iter().map(|edge| edge.id)) {
            return Err(OpportunityGraphError::NonCanonical("edge IDs"));
        }
        let demand_by_id = self
            .demands
            .iter()
            .map(|demand| (demand.id, demand))
            .collect::<BTreeMap<_, _>>();
        let supply_by_id = self
            .supplies
            .iter()
            .map(|supply| (supply.id, supply))
            .collect::<BTreeMap<_, _>>();
        for supply in &self.supplies {
            if supply.capacity == 0
                || supply.availability_denominator == 0
                || supply.availability_numerator > supply.availability_denominator
                || supply.wildlife.is_some() == supply.archetype_id.is_some()
            {
                return Err(OpportunityGraphError::InvalidSupply(supply.id));
            }
        }
        for edge in &self.edges {
            let demand = demand_by_id
                .get(&edge.id.demand)
                .ok_or(OpportunityGraphError::DanglingEdge(edge.id))?;
            let supply = supply_by_id
                .get(&edge.id.supply)
                .ok_or(OpportunityGraphError::DanglingEdge(edge.id))?;
            if edge.exact_completion_delta == 0
                || edge.teacher_value_micros <= 0
                || edge.exact_completion_delta > demand.exact_completion_delta
                || matches!(demand.id.kind, OpportunityDemandKind::WildlifePlacement)
                    != supply.wildlife.is_some()
                || (supply.archetype_id.is_some() && edge.compatible_rotation_mask == 0)
                || (supply.wildlife.is_some() && edge.compatible_rotation_mask != 0)
            {
                return Err(OpportunityGraphError::InvalidEdge(edge.id));
            }
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, OpportunityGraphError> {
        self.validate()?;
        let payload = postcard::to_allocvec(self)?;
        let mut bytes = Vec::with_capacity(GRAPH_MAGIC.len() + 4 + payload.len());
        bytes.extend_from_slice(GRAPH_MAGIC);
        bytes.extend_from_slice(&OPPORTUNITY_GRAPH_SCHEMA_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, OpportunityGraphError> {
        let payload = canonical_payload(bytes, GRAPH_MAGIC)?;
        let graph: Self = postcard::from_bytes(payload)?;
        graph.validate()?;
        if graph.canonical_bytes()? != bytes {
            return Err(OpportunityGraphError::NonCanonical("graph encoding"));
        }
        Ok(graph)
    }

    pub fn canonical_hash(&self) -> Result<blake3::Hash, OpportunityGraphError> {
        Ok(blake3::hash(&self.canonical_bytes()?))
    }

    /// Solve exact maximum-weight capacitated bipartite matching for the
    /// integer edge objective frozen by schema v1.
    pub fn solve_matching(&self) -> Result<OpportunityMatchingSummary, OpportunityGraphError> {
        self.validate()?;
        let assignments = solve_capacitated_matching(self)?;
        let demand_by_id = self
            .demands
            .iter()
            .map(|demand| (demand.id, demand))
            .collect::<BTreeMap<_, _>>();
        let mut wildlife_matches = 0u32;
        let mut habitat_matches = 0u32;
        let mut market_matches = 0u32;
        let mut unseen_matches = 0u32;
        let mut exact_completion_value = 0u64;
        let mut teacher_value_micros = 0i64;
        for assignment in &assignments {
            match demand_by_id[&assignment.demand].id.kind {
                OpportunityDemandKind::WildlifePlacement => wildlife_matches += 1,
                OpportunityDemandKind::HabitatFrontier => habitat_matches += 1,
            }
            match assignment.supply.kind {
                OpportunitySupplyKind::MarketWildlife | OpportunitySupplyKind::MarketTile => {
                    market_matches += 1
                }
                OpportunitySupplyKind::UnseenWildlife
                | OpportunitySupplyKind::UnseenTileArchetype => unseen_matches += 1,
            }
            exact_completion_value += u64::from(assignment.exact_completion_delta);
            teacher_value_micros = teacher_value_micros
                .checked_add(assignment.teacher_value_micros)
                .ok_or(OpportunityGraphError::CountOverflow("matching objective"))?;
        }
        let matched_demands = u32::try_from(assignments.len())
            .map_err(|_| OpportunityGraphError::CountOverflow("matched demands"))?;
        Ok(OpportunityMatchingSummary {
            schema_version: OPPORTUNITY_GRAPH_SCHEMA_VERSION,
            graph_hash: *self.canonical_hash()?.as_bytes(),
            demand_count: as_u32(self.demands.len(), "demand count")?,
            supply_count: as_u32(self.supplies.len(), "supply count")?,
            edge_count: as_u32(self.edges.len(), "edge count")?,
            matched_demands,
            unmatched_demands: as_u32(self.demands.len(), "demand count")? - matched_demands,
            wildlife_matches,
            habitat_matches,
            market_matches,
            unseen_matches,
            exact_completion_value,
            teacher_value_micros,
            assignments,
        })
    }

    /// Verify D6 covariance against a graph rebuilt from the transformed
    /// public state. Source hashes are intentionally excluded because the
    /// transformed public serialization is a different exact state identity.
    pub fn verify_d6_covariance(
        &self,
        transformed: &Self,
        transform: D6Transform,
    ) -> Result<(), OpportunityGraphError> {
        self.validate()?;
        transformed.validate()?;
        if self.focal_seat != transformed.focal_seat
            || self.completed_turns != transformed.completed_turns
            || self.personal_turns_remaining != transformed.personal_turns_remaining
            || self.exact_supply_hash != transformed.exact_supply_hash
            || self.supplies != transformed.supplies
        {
            return Err(OpportunityGraphError::D6Mismatch("global or supply fields"));
        }
        let expected_demands = self
            .demands
            .iter()
            .map(|demand| {
                let transformed = OpportunityDemand {
                    id: demand.id.transformed(transform)?,
                    ..demand.clone()
                };
                Ok((transformed.id, transformed))
            })
            .collect::<Result<BTreeMap<_, _>, OpportunityGraphError>>()?;
        let observed_demands = transformed
            .demands
            .iter()
            .cloned()
            .map(|demand| (demand.id, demand))
            .collect::<BTreeMap<_, _>>();
        if expected_demands.values().cloned().collect::<Vec<_>>()
            != observed_demands.values().cloned().collect::<Vec<_>>()
        {
            return Err(OpportunityGraphError::D6Mismatch("demands"));
        }
        let expected_edges = self
            .edges
            .iter()
            .map(|edge| {
                let demand = edge.id.demand.transformed(transform)?;
                let supply = self
                    .supplies
                    .binary_search_by_key(&edge.id.supply, |candidate| candidate.id)
                    .ok()
                    .map(|index| &self.supplies[index])
                    .ok_or(OpportunityGraphError::DanglingEdge(edge.id))?;
                Ok(OpportunityEdge {
                    id: OpportunityEdgeId {
                        demand,
                        supply: edge.id.supply,
                    },
                    compatible_rotation_mask: transform_rotation_mask(
                        edge.compatible_rotation_mask,
                        supply,
                        transform,
                    )?,
                    ..edge.clone()
                })
            })
            .collect::<Result<Vec<_>, OpportunityGraphError>>()?;
        let mut expected_edges = expected_edges;
        expected_edges.sort_by_key(|edge| edge.id);
        if expected_edges != transformed.edges {
            return Err(OpportunityGraphError::D6Mismatch("edges"));
        }
        Ok(())
    }
}

impl OpportunityMatchingSummary {
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, OpportunityGraphError> {
        if self.schema_version != OPPORTUNITY_GRAPH_SCHEMA_VERSION
            || self.matched_demands + self.unmatched_demands != self.demand_count
            || self.assignments.len() != self.matched_demands as usize
            || !strictly_sorted_unique(
                self.assignments
                    .iter()
                    .map(|assignment| (assignment.demand, assignment.supply)),
            )
        {
            return Err(OpportunityGraphError::NonCanonical("matching summary"));
        }
        let payload = postcard::to_allocvec(self)?;
        let mut bytes = Vec::with_capacity(SUMMARY_MAGIC.len() + 4 + payload.len());
        bytes.extend_from_slice(SUMMARY_MAGIC);
        bytes.extend_from_slice(&OPPORTUNITY_GRAPH_SCHEMA_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, OpportunityGraphError> {
        let payload = canonical_payload(bytes, SUMMARY_MAGIC)?;
        let summary: Self = postcard::from_bytes(payload)?;
        if summary.canonical_bytes()? != bytes {
            return Err(OpportunityGraphError::NonCanonical(
                "matching summary encoding",
            ));
        }
        Ok(summary)
    }

    pub fn canonical_hash(&self) -> Result<blake3::Hash, OpportunityGraphError> {
        Ok(blake3::hash(&self.canonical_bytes()?))
    }
}

fn build_demands(
    state: &PublicGameState,
    board: &Board,
    focal_seat: usize,
    deadline: u8,
    kind: Option<OpportunityDemandKind>,
) -> Result<Vec<OpportunityDemand>, OpportunityGraphError> {
    let mut demands = Vec::new();
    if kind.is_none_or(|value| value == OpportunityDemandKind::WildlifePlacement) {
        for wildlife in Wildlife::ALL {
            demands.extend(build_wildlife_demands_for_species(
                state, board, deadline, wildlife,
            )?);
        }
    }
    if kind.is_none_or(|value| value == OpportunityDemandKind::HabitatFrontier) {
        let habitat = board.habitat_analysis();
        for coord in board.frontier() {
            let neighbor_count = coord
                .neighbors()
                .into_iter()
                .filter(|neighbor| board.tile_at(*neighbor).is_some())
                .count() as u8;
            for terrain in Terrain::ALL {
                // The exact edge builder determines whether any public supply can
                // actually grow this component. A unit lower bound keeps demand
                // semantics target-free while excluding non-growth edges later.
                demands.push(OpportunityDemand {
                    id: DemandId {
                        kind: OpportunityDemandKind::HabitatFrontier,
                        subject: terrain as u8,
                        coord,
                    },
                    deadline_turns: deadline,
                    exact_completion_delta: u16::from(habitat.largest(terrain).max(1)),
                    local_same_species_neighbors: 0,
                    frontier_neighbor_count: neighbor_count,
                });
            }
        }
    }
    demands.sort_by_key(|demand| demand.id);
    // Demands without an exact positive completion edge are not opportunities.
    let _ = focal_seat;
    Ok(demands)
}

fn build_wildlife_demands_for_species(
    state: &PublicGameState,
    board: &Board,
    deadline: u8,
    wildlife: Wildlife,
) -> Result<Vec<OpportunityDemand>, OpportunityGraphError> {
    let cards = state.config().scoring_cards;
    let baseline = score_wildlife_type(board, cards, wildlife);
    let mut after = board.clone();
    let mut demands = Vec::new();
    for coord in board.wildlife_placements(wildlife) {
        let delta_handle = after.place_wildlife(coord, wildlife)?;
        let scored = score_wildlife_type(&after, cards, wildlife);
        after.undo(delta_handle)?;
        let delta = scored.saturating_sub(baseline);
        if delta == 0 {
            continue;
        }
        let local_same_species_neighbors = coord
            .neighbors()
            .into_iter()
            .filter(|neighbor| board.wildlife_at(*neighbor) == Some(wildlife))
            .count() as u8;
        demands.push(OpportunityDemand {
            id: DemandId {
                kind: OpportunityDemandKind::WildlifePlacement,
                subject: wildlife as u8,
                coord,
            },
            deadline_turns: deadline,
            exact_completion_delta: delta,
            local_same_species_neighbors,
            frontier_neighbor_count: 0,
        });
    }
    Ok(demands)
}

fn build_supplies(
    state: &PublicGameState,
    focal_seat: usize,
    personal_turns_remaining: u8,
    exact: &ExactSemanticSupply,
) -> Result<Vec<OpportunitySupply>, OpportunityGraphError> {
    let mut supplies = Vec::new();
    for slot in MarketSlot::ALL {
        if let Some(wildlife) = state.market().wildlife[slot.index()] {
            supplies.push(OpportunitySupply {
                id: SupplyId {
                    kind: OpportunitySupplyKind::MarketWildlife,
                    subject: slot.index() as u16,
                },
                capacity: 1,
                availability_numerator: 1,
                availability_denominator: 1,
                access_delay_turns: u8::from(state.current_player() != focal_seat),
                opponents_before_access: opponents_before_access(state, focal_seat),
                wildlife: Some(wildlife),
                archetype_id: None,
                market_slot: Some(slot),
            });
        }
        if let Some(tile) = state.market().tiles[slot.index()] {
            let reference = standard_semantic_archetype_catalog().reference_for_tile(tile)?;
            supplies.push(OpportunitySupply {
                id: SupplyId {
                    kind: OpportunitySupplyKind::MarketTile,
                    subject: slot.index() as u16,
                },
                capacity: 1,
                availability_numerator: 1,
                availability_denominator: 1,
                access_delay_turns: u8::from(state.current_player() != focal_seat),
                opponents_before_access: opponents_before_access(state, focal_seat),
                wildlife: None,
                archetype_id: Some(reference.archetype_id),
                market_slot: Some(slot),
            });
        }
    }
    let wildlife_total = exact.wildlife_bag_counts().into_iter().sum::<u16>();
    if personal_turns_remaining > 1 && wildlife_total > 0 {
        for wildlife in Wildlife::ALL {
            let count = exact.wildlife_bag_counts()[wildlife as usize];
            if count == 0 {
                continue;
            }
            supplies.push(OpportunitySupply {
                id: SupplyId {
                    kind: OpportunitySupplyKind::UnseenWildlife,
                    subject: wildlife as u16,
                },
                capacity: count.min(u16::from(personal_turns_remaining - 1)),
                availability_numerator: count,
                availability_denominator: wildlife_total,
                access_delay_turns: 1,
                opponents_before_access: state.boards().len().saturating_sub(1) as u8,
                wildlife: Some(wildlife),
                archetype_id: None,
                market_slot: None,
            });
        }
    }
    if personal_turns_remaining > 1 && exact.unseen_tile_count() > 0 {
        for definition in standard_semantic_archetype_catalog().definitions() {
            let count = exact.count(definition.id).unwrap_or(0);
            if count == 0 {
                continue;
            }
            supplies.push(OpportunitySupply {
                id: SupplyId {
                    kind: OpportunitySupplyKind::UnseenTileArchetype,
                    subject: definition.id.code(),
                },
                capacity: count.min(u16::from(personal_turns_remaining - 1)),
                availability_numerator: count,
                availability_denominator: exact.unseen_tile_count(),
                access_delay_turns: 1,
                opponents_before_access: state.boards().len().saturating_sub(1) as u8,
                wildlife: None,
                archetype_id: Some(definition.id),
                market_slot: None,
            });
        }
    }
    supplies.sort_by_key(|supply| supply.id);
    Ok(supplies)
}

fn build_edges(
    board: &Board,
    demands: &mut Vec<OpportunityDemand>,
    supplies: &[OpportunitySupply],
) -> Result<Vec<OpportunityEdge>, OpportunityGraphError> {
    let habitat = demands
        .iter()
        .any(|demand| demand.id.kind == OpportunityDemandKind::HabitatFrontier)
        .then(|| board.habitat_analysis());
    let mut wildlife_supply_indices: [Vec<usize>; 5] = std::array::from_fn(|_| Vec::new());
    let mut tile_supply_indices = Vec::new();
    for (index, supply) in supplies.iter().enumerate() {
        if let Some(wildlife) = supply.wildlife {
            wildlife_supply_indices[wildlife as usize].push(index);
        } else if supply.archetype_id.is_some() {
            tile_supply_indices.push(index);
        }
    }
    let mut edges = Vec::new();
    for demand in demands.iter() {
        match demand.id.kind {
            OpportunityDemandKind::WildlifePlacement => {
                for &supply_index in &wildlife_supply_indices[usize::from(demand.id.subject)] {
                    let supply = &supplies[supply_index];
                    if supply.access_delay_turns > demand.deadline_turns {
                        continue;
                    }
                    let teacher_value_micros =
                        teacher_value(demand.exact_completion_delta, supply)?;
                    if teacher_value_micros > 0 {
                        edges.push(OpportunityEdge {
                            id: OpportunityEdgeId {
                                demand: demand.id,
                                supply: supply.id,
                            },
                            compatible_rotation_mask: 0,
                            best_matching_edges: 0,
                            exact_completion_delta: demand.exact_completion_delta,
                            teacher_value_micros,
                        });
                    }
                }
            }
            OpportunityDemandKind::HabitatFrontier => {}
        }
    }
    if let Some(habitat) = habitat.as_ref() {
        let deadline = demands
            .iter()
            .find(|demand| demand.id.kind == OpportunityDemandKind::HabitatFrontier)
            .map_or(0, |demand| demand.deadline_turns);
        let frontier = demands
            .iter()
            .filter(|demand| demand.id.kind == OpportunityDemandKind::HabitatFrontier)
            .map(|demand| demand.id.coord)
            .collect::<BTreeSet<_>>();
        for coord in frontier {
            for &supply_index in &tile_supply_indices {
                let supply = &supplies[supply_index];
                if supply.access_delay_turns > deadline {
                    continue;
                }
                let archetype_id = supply
                    .archetype_id
                    .expect("tile supply index has an archetype");
                let (tile, reference) = archetype_representative(archetype_id)?;
                let mut best_delta = [0u16; 5];
                let mut best_matching_edges = [0u8; 5];
                let mut rotation_mask = [0u8; 5];
                for &game_rotation in tile_rotations(*tile) {
                    let (largest, matching_edges) = habitat
                        .largest_all_and_matching_edges_after_tile(
                            board,
                            coord,
                            *tile,
                            game_rotation,
                        );
                    let canonical_rotation = reference.canonical_rotation_for_game(game_rotation);
                    for terrain in Terrain::ALL {
                        if !tile.contains_terrain(terrain) {
                            continue;
                        }
                        let index = terrain as usize;
                        let delta =
                            u16::from(largest[index].saturating_sub(habitat.largest(terrain)));
                        if delta > best_delta[index] {
                            best_delta[index] = delta;
                            best_matching_edges[index] = matching_edges;
                            rotation_mask[index] = 1 << canonical_rotation.get();
                        } else if delta != 0 && delta == best_delta[index] {
                            best_matching_edges[index] =
                                best_matching_edges[index].max(matching_edges);
                            rotation_mask[index] |= 1 << canonical_rotation.get();
                        }
                    }
                }
                for terrain in Terrain::ALL {
                    let index = terrain as usize;
                    if best_delta[index] == 0 {
                        continue;
                    }
                    let teacher_value_micros = teacher_value(best_delta[index], supply)?;
                    if teacher_value_micros <= 0 {
                        continue;
                    }
                    edges.push(OpportunityEdge {
                        id: OpportunityEdgeId {
                            demand: DemandId {
                                kind: OpportunityDemandKind::HabitatFrontier,
                                subject: terrain as u8,
                                coord,
                            },
                            supply: supply.id,
                        },
                        compatible_rotation_mask: rotation_mask[index],
                        best_matching_edges: best_matching_edges[index],
                        exact_completion_delta: best_delta[index],
                        teacher_value_micros,
                    });
                }
            }
        }
    }
    // Replace habitat demand placeholders with their maximum exact edge delta
    // and remove facts that have no compatible public supply.
    let maximum_delta = edges.iter().fold(BTreeMap::new(), |mut maximum, edge| {
        maximum
            .entry(edge.id.demand)
            .and_modify(|value: &mut u16| *value = (*value).max(edge.exact_completion_delta))
            .or_insert(edge.exact_completion_delta);
        maximum
    });
    demands.retain_mut(|demand| {
        let Some(&delta) = maximum_delta.get(&demand.id) else {
            return false;
        };
        demand.exact_completion_delta = delta;
        true
    });
    edges.sort_by_key(|edge| edge.id);
    Ok(edges)
}

fn teacher_value(
    completion_delta: u16,
    supply: &OpportunitySupply,
) -> Result<i64, OpportunityGraphError> {
    let gross = i64::from(completion_delta)
        .checked_mul(TEACHER_SCALE)
        .and_then(|value| value.checked_mul(i64::from(supply.availability_numerator)))
        .ok_or(OpportunityGraphError::CountOverflow("teacher gross value"))?
        / i64::from(supply.availability_denominator);
    let exposure =
        1 + i64::from(supply.access_delay_turns) + i64::from(supply.opponents_before_access);
    Ok(gross / exposure)
}

fn solve_capacitated_matching(
    graph: &OpportunityGraphV1,
) -> Result<Vec<OpportunityAssignment>, OpportunityGraphError> {
    let (mut flow, source, sink) = build_matching_flow(graph)?;
    flow.augment_while_negative(
        source,
        sink,
        usize::from(graph.personal_turns_remaining).saturating_mul(2),
    )?;
    assignments_for_labels(graph, flow.used_labels())
}

fn build_matching_flow(
    graph: &OpportunityGraphV1,
) -> Result<(MinCostFlow, usize, usize), OpportunityGraphError> {
    let demand_count = graph.demands.len();
    let supply_count = graph.supplies.len();
    let source = 0usize;
    let demand_start = 1usize;
    let supply_start = demand_start + demand_count;
    let sink = supply_start + supply_count;
    let indexed_edges = graph
        .edges
        .iter()
        .map(|edge| {
            let demand_index = graph
                .demands
                .binary_search_by_key(&edge.id.demand, |candidate| candidate.id)
                .map_err(|_| OpportunityGraphError::DanglingEdge(edge.id))?;
            let supply_index = graph
                .supplies
                .binary_search_by_key(&edge.id.supply, |candidate| candidate.id)
                .map_err(|_| OpportunityGraphError::DanglingEdge(edge.id))?;
            Ok((edge, demand_index, supply_index))
        })
        .collect::<Result<Vec<_>, OpportunityGraphError>>()?;
    let mut capacities = vec![0usize; sink + 1];
    capacities[source] = demand_count;
    capacities[sink] = supply_count;
    for demand_index in 0..demand_count {
        capacities[demand_start + demand_index] = 1;
    }
    for supply_index in 0..supply_count {
        capacities[supply_start + supply_index] = 1;
    }
    for (_, demand_index, supply_index) in &indexed_edges {
        capacities[demand_start + demand_index] += 1;
        capacities[supply_start + supply_index] += 1;
    }
    let mut flow = MinCostFlow::with_capacities(&capacities);
    for demand_index in 0..demand_count {
        flow.add_edge(source, demand_start + demand_index, 1, 0, None);
    }
    for (supply_index, supply) in graph.supplies.iter().enumerate() {
        flow.add_edge(
            supply_start + supply_index,
            sink,
            i32::from(supply.capacity),
            0,
            None,
        );
    }
    for (edge, demand_index, supply_index) in indexed_edges {
        flow.add_edge(
            demand_start + demand_index,
            supply_start + supply_index,
            1,
            -edge.teacher_value_micros,
            Some(edge.id),
        );
    }
    Ok((flow, source, sink))
}

fn assignments_for_labels(
    graph: &OpportunityGraphV1,
    labels: Vec<OpportunityEdgeId>,
) -> Result<Vec<OpportunityAssignment>, OpportunityGraphError> {
    let mut assignments = labels
        .into_iter()
        .map(|id| {
            let edge = graph
                .edges
                .binary_search_by_key(&id, |edge| edge.id)
                .ok()
                .and_then(|index| graph.edges.get(index))
                .expect("used flow labels originate from graph edges");
            OpportunityAssignment {
                demand: id.demand,
                supply: id.supply,
                exact_completion_delta: edge.exact_completion_delta,
                teacher_value_micros: edge.teacher_value_micros,
            }
        })
        .collect::<Vec<_>>();
    assignments.sort_by_key(|assignment| (assignment.demand, assignment.supply));
    Ok(assignments)
}

fn matching_component_demand_ids(graph: &OpportunityGraphV1) -> [Vec<DemandId>; 6] {
    let mut components: [Vec<DemandId>; 6] = std::array::from_fn(|_| Vec::new());
    for demand in &graph.demands {
        let component = if demand.id.kind == OpportunityDemandKind::HabitatFrontier {
            5
        } else {
            usize::from(demand.id.subject)
        };
        components[component].push(demand.id);
    }
    components
}

fn matching_component_frontier(
    graph: &OpportunityGraphV1,
    component: usize,
) -> Result<Vec<MatchingFrontierState>, OpportunityGraphError> {
    let belongs = |demand: DemandId| {
        if component == 5 {
            demand.kind == OpportunityDemandKind::HabitatFrontier
        } else {
            demand.kind == OpportunityDemandKind::WildlifePlacement
                && usize::from(demand.subject) == component
        }
    };
    let component_graph = OpportunityGraphV1 {
        schema_version: graph.schema_version,
        source_public_state_hash: graph.source_public_state_hash,
        focal_seat: graph.focal_seat,
        completed_turns: graph.completed_turns,
        personal_turns_remaining: graph.personal_turns_remaining,
        exact_supply_hash: graph.exact_supply_hash,
        demands: graph
            .demands
            .iter()
            .filter(|demand| belongs(demand.id))
            .cloned()
            .collect(),
        supplies: graph.supplies.clone(),
        edges: graph
            .edges
            .iter()
            .filter(|edge| belongs(edge.id.demand))
            .cloned()
            .collect(),
    };
    let (mut flow, source, sink) = build_matching_flow(&component_graph)?;
    flow.augment_frontier(
        source,
        sink,
        usize::from(component_graph.personal_turns_remaining).saturating_mul(2),
    )?
    .into_iter()
    .map(|labels| {
        let assignments = assignments_for_labels(&component_graph, labels)?;
        let teacher_value_micros = assignments.iter().try_fold(0i64, |total, assignment| {
            total.checked_add(assignment.teacher_value_micros).ok_or(
                OpportunityGraphError::CountOverflow("matching frontier objective"),
            )
        })?;
        Ok(MatchingFrontierState {
            teacher_value_micros,
            assignments,
        })
    })
    .collect()
}

fn combine_matching_frontiers(
    components: &[&[MatchingFrontierState]; 6],
    maximum_flow: usize,
) -> Result<Vec<OpportunityAssignment>, OpportunityGraphError> {
    #[derive(Clone, Copy)]
    struct Choice {
        value: i64,
        counts: [u8; 6],
    }

    let mut dynamic = vec![None; maximum_flow + 1];
    dynamic[0] = Some(Choice {
        value: 0,
        counts: [0; 6],
    });
    for (component, frontier) in components.iter().enumerate() {
        let mut next = vec![None; maximum_flow + 1];
        for (used, choice) in dynamic.iter().enumerate() {
            let Some(choice) = choice else { continue };
            for (count, state) in frontier.iter().enumerate() {
                let total = used + count;
                if total > maximum_flow || count > usize::from(u8::MAX) {
                    break;
                }
                let mut candidate = *choice;
                candidate.value = candidate
                    .value
                    .checked_add(state.teacher_value_micros)
                    .ok_or(OpportunityGraphError::CountOverflow(
                        "combined matching objective",
                    ))?;
                candidate.counts[component] = count as u8;
                let replace = next[total].is_none_or(|current: Choice| {
                    candidate.value > current.value
                        || (candidate.value == current.value && candidate.counts > current.counts)
                });
                if replace {
                    next[total] = Some(candidate);
                }
            }
        }
        dynamic = next;
    }
    let best = dynamic.into_iter().flatten().max_by(|left, right| {
        left.value
            .cmp(&right.value)
            .then_with(|| left.counts.cmp(&right.counts))
    });
    let Some(best) = best else {
        return Ok(Vec::new());
    };
    let mut assignments = Vec::new();
    for component in 0..6 {
        assignments.extend(
            components[component][usize::from(best.counts[component])]
                .assignments
                .iter()
                .cloned(),
        );
    }
    assignments.sort_by_key(|assignment| (assignment.demand, assignment.supply));
    Ok(assignments)
}

#[derive(Debug, Clone)]
struct FlowEdge {
    to: usize,
    reverse: usize,
    capacity: i32,
    cost: i64,
    label: Option<OpportunityEdgeId>,
    original_capacity: i32,
}

struct MinCostFlow {
    adjacency: Vec<Vec<FlowEdge>>,
}

impl MinCostFlow {
    fn with_capacities(capacities: &[usize]) -> Self {
        Self {
            adjacency: capacities
                .iter()
                .map(|capacity| Vec::with_capacity(*capacity))
                .collect(),
        }
    }

    fn add_edge(
        &mut self,
        from: usize,
        to: usize,
        capacity: i32,
        cost: i64,
        label: Option<OpportunityEdgeId>,
    ) {
        let forward_reverse = self.adjacency[to].len();
        let reverse_forward = self.adjacency[from].len();
        self.adjacency[from].push(FlowEdge {
            to,
            reverse: forward_reverse,
            capacity,
            cost,
            label,
            original_capacity: capacity,
        });
        self.adjacency[to].push(FlowEdge {
            to: from,
            reverse: reverse_forward,
            capacity: 0,
            cost: -cost,
            label: None,
            original_capacity: 0,
        });
    }

    fn augment_while_negative(
        &mut self,
        source: usize,
        sink: usize,
        maximum_flow: usize,
    ) -> Result<(), OpportunityGraphError> {
        self.augment_internal(source, sink, maximum_flow, false)?;
        Ok(())
    }

    fn augment_frontier(
        &mut self,
        source: usize,
        sink: usize,
        maximum_flow: usize,
    ) -> Result<Vec<Vec<OpportunityEdgeId>>, OpportunityGraphError> {
        self.augment_internal(source, sink, maximum_flow, true)
    }

    fn augment_internal(
        &mut self,
        source: usize,
        sink: usize,
        maximum_flow: usize,
        capture_frontier: bool,
    ) -> Result<Vec<Vec<OpportunityEdgeId>>, OpportunityGraphError> {
        let n = self.adjacency.len();
        let mut distance = vec![i64::MAX; n];
        let mut previous = vec![None; n];
        let mut queued = vec![false; n];
        let mut queue = VecDeque::with_capacity(n);
        let mut frontier = Vec::with_capacity(maximum_flow + 1);
        if capture_frontier {
            frontier.push(Vec::new());
        }
        for _ in 0..maximum_flow {
            distance.fill(i64::MAX);
            previous.fill(None);
            queued.fill(false);
            queue.clear();
            distance[source] = 0;
            queue.push_back(source);
            queued[source] = true;
            while let Some(node) = queue.pop_front() {
                queued[node] = false;
                for (edge_index, edge) in self.adjacency[node].iter().enumerate() {
                    if edge.capacity <= 0 || distance[node] == i64::MAX {
                        continue;
                    }
                    let candidate = distance[node]
                        .checked_add(edge.cost)
                        .ok_or(OpportunityGraphError::CountOverflow("flow distance"))?;
                    if candidate < distance[edge.to] {
                        distance[edge.to] = candidate;
                        previous[edge.to] = Some((node, edge_index));
                        if !queued[edge.to] {
                            queued[edge.to] = true;
                            queue.push_back(edge.to);
                        }
                    }
                }
            }
            if distance[sink] >= 0 || distance[sink] == i64::MAX {
                break;
            }
            let mut node = sink;
            while node != source {
                let (prior, edge_index) = previous[node].ok_or(
                    OpportunityGraphError::FlowInvariant("broken predecessor chain"),
                )?;
                let reverse = self.adjacency[prior][edge_index].reverse;
                self.adjacency[prior][edge_index].capacity -= 1;
                self.adjacency[node][reverse].capacity += 1;
                node = prior;
            }
            if capture_frontier {
                frontier.push(self.used_labels());
            }
        }
        Ok(frontier)
    }

    fn used_labels(&self) -> Vec<OpportunityEdgeId> {
        self.adjacency
            .iter()
            .flatten()
            .filter_map(|edge| {
                (edge.original_capacity == 1 && edge.capacity == 0)
                    .then_some(edge.label)
                    .flatten()
            })
            .collect()
    }
}

fn archetype_representatives() -> &'static Vec<Option<(Tile, SemanticTileReference)>> {
    static REPRESENTATIVES: std::sync::OnceLock<Vec<Option<(Tile, SemanticTileReference)>>> =
        std::sync::OnceLock::new();
    REPRESENTATIVES.get_or_init(|| {
        let catalog = standard_semantic_archetype_catalog();
        let mut values = vec![None; catalog.definitions().len()];
        for tile in STANDARD_TILES.iter().copied() {
            let reference = catalog
                .reference_for_tile(tile)
                .expect("standard tile has a semantic archetype");
            values[reference.archetype_id.index()].get_or_insert((tile, reference));
        }
        values
    })
}

fn archetype_representative(
    id: SemanticArchetypeId,
) -> Result<&'static (Tile, SemanticTileReference), OpportunityGraphError> {
    archetype_representatives()
        .get(id.index())
        .and_then(Option::as_ref)
        .ok_or(OpportunityGraphError::UnknownArchetype(id.code()))
}

fn tile_rotations(tile: Tile) -> &'static [Rotation] {
    if tile.terrain_b.is_some() {
        &Rotation::ALL
    } else {
        &Rotation::ALL[..1]
    }
}

fn transform_rotation_mask(
    mask: u8,
    supply: &OpportunitySupply,
    transform: D6Transform,
) -> Result<u8, OpportunityGraphError> {
    if mask == 0 {
        return Ok(0);
    }
    let archetype_id = supply
        .archetype_id
        .ok_or(OpportunityGraphError::InvalidSupply(supply.id))?;
    let (tile, reference) = archetype_representative(archetype_id)?;
    let mut transformed = 0u8;
    for canonical in Rotation::ALL {
        if mask & (1 << canonical.get()) == 0 {
            continue;
        }
        let game = reference.game_rotation_for_canonical(canonical);
        let transformed_game = transform.transform_tile_rotation(*tile, game);
        let transformed_canonical = reference.canonical_rotation_for_game(transformed_game);
        transformed |= 1 << transformed_canonical.get();
    }
    Ok(transformed)
}

fn opponents_before_access(state: &PublicGameState, focal_seat: usize) -> u8 {
    if state.current_player() == focal_seat {
        return 0;
    }
    let players = state.boards().len();
    ((focal_seat + players - state.current_player()) % players) as u8
}

fn strictly_sorted_unique<T: Ord>(values: impl IntoIterator<Item = T>) -> bool {
    let mut previous = None;
    for value in values {
        if previous.as_ref().is_some_and(|prior| prior >= &value) {
            return false;
        }
        previous = Some(value);
    }
    true
}

fn canonical_payload<'a>(
    bytes: &'a [u8],
    expected_magic: &[u8; 8],
) -> Result<&'a [u8], OpportunityGraphError> {
    if bytes.len() < 14 || &bytes[..8] != expected_magic {
        return Err(OpportunityGraphError::InvalidEncoding("magic or length"));
    }
    let version = u16::from_le_bytes([bytes[8], bytes[9]]);
    if version != OPPORTUNITY_GRAPH_SCHEMA_VERSION {
        return Err(OpportunityGraphError::InvalidSchema(version));
    }
    let length = u32::from_le_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]) as usize;
    if bytes.len() != 14 + length {
        return Err(OpportunityGraphError::InvalidEncoding("payload length"));
    }
    Ok(&bytes[14..])
}

fn as_u32(value: usize, label: &'static str) -> Result<u32, OpportunityGraphError> {
    u32::try_from(value).map_err(|_| OpportunityGraphError::CountOverflow(label))
}

#[derive(Debug, Error)]
pub enum OpportunityGraphError {
    #[error("focal seat {focal} is outside {players} public boards")]
    InvalidFocalSeat { focal: usize, players: usize },
    #[error("unsupported opportunity graph schema version {0}")]
    InvalidSchema(u16),
    #[error("opportunity graph {0} are not sorted and unique")]
    NonCanonical(&'static str),
    #[error("invalid opportunity supply {0:?}")]
    InvalidSupply(SupplyId),
    #[error("invalid opportunity edge {0:?}")]
    InvalidEdge(OpportunityEdgeId),
    #[error("dangling opportunity edge {0:?}")]
    DanglingEdge(OpportunityEdgeId),
    #[error("unknown semantic tile archetype {0}")]
    UnknownArchetype(u16),
    #[error("opportunity graph count overflowed: {0}")]
    CountOverflow(&'static str),
    #[error("invalid opportunity graph encoding: {0}")]
    InvalidEncoding(&'static str),
    #[error("D6 covariance mismatch in {0}")]
    D6Mismatch(&'static str),
    #[error("matching flow invariant failed: {0}")]
    FlowInvariant(&'static str),
    #[error(transparent)]
    Board(#[from] cascadia_game::BoardError),
    #[error(transparent)]
    D6(#[from] cascadia_game::D6Error),
    #[error(transparent)]
    SemanticSupply(#[from] SemanticSupplyError),
    #[error(transparent)]
    Postcard(#[from] postcard::Error),
}

#[cfg(test)]
mod tests {
    use cascadia_game::{GameConfig, GameSeed, GameState, MarketPrelude};

    use super::*;

    fn nontrivial_game() -> GameState {
        let mut game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(2026061702),
        )
        .unwrap();
        for _ in 0..8 {
            let action = game
                .legal_turn_actions(&MarketPrelude::default())
                .unwrap()
                .into_iter()
                .find(|action| action.wildlife.is_some())
                .unwrap_or_else(|| {
                    game.legal_turn_actions(&MarketPrelude::default())
                        .unwrap()
                        .into_iter()
                        .next()
                        .unwrap()
                });
            game.apply(&action).unwrap();
        }
        game
    }

    #[test]
    fn graph_and_matching_round_trip_canonically() {
        let game = nontrivial_game();
        let graph = OpportunityGraphV1::from_public_state(&game.public_state(), 0).unwrap();
        assert!(!graph.demands.is_empty());
        assert!(!graph.supplies.is_empty());
        assert!(!graph.edges.is_empty());
        let bytes = graph.canonical_bytes().unwrap();
        assert_eq!(
            OpportunityGraphV1::from_canonical_bytes(&bytes).unwrap(),
            graph
        );
        let summary = graph.solve_matching().unwrap();
        assert!(summary.matched_demands > 0);
        let summary_bytes = summary.canonical_bytes().unwrap();
        assert_eq!(
            OpportunityMatchingSummary::from_canonical_bytes(&summary_bytes).unwrap(),
            summary
        );
    }

    #[test]
    fn replay_and_public_only_construction_are_deterministic() {
        let game = nontrivial_game();
        let public = game.public_state();
        let first = OpportunityGraphV1::from_public_state(&public, 0).unwrap();
        let second = OpportunityGraphV1::from_public_state(&public, 0).unwrap();
        assert_eq!(
            first.canonical_bytes().unwrap(),
            second.canonical_bytes().unwrap()
        );
        assert_eq!(
            first.solve_matching().unwrap(),
            second.solve_matching().unwrap()
        );
    }

    #[test]
    fn all_d6_transforms_are_covariant() {
        let game = nontrivial_game();
        let public = game.public_state();
        let graph = OpportunityGraphV1::from_public_state(&public, 0).unwrap();
        for transform in D6Transform::ALL {
            let transformed_public = public.transformed(transform).unwrap();
            let transformed =
                OpportunityGraphV1::from_public_state(&transformed_public, 0).unwrap();
            graph
                .verify_d6_covariance(&transformed, transform)
                .unwrap_or_else(|error| panic!("transform {}: {error}", transform.id()));
        }
    }

    #[test]
    fn codecs_reject_trailing_bytes_and_unknown_versions() {
        let game = nontrivial_game();
        let graph = OpportunityGraphV1::from_public_state(&game.public_state(), 0).unwrap();
        let mut bytes = graph.canonical_bytes().unwrap();
        bytes.push(0);
        assert!(matches!(
            OpportunityGraphV1::from_canonical_bytes(&bytes),
            Err(OpportunityGraphError::InvalidEncoding("payload length"))
        ));
        bytes.pop();
        bytes[8..10].copy_from_slice(&2u16.to_le_bytes());
        assert!(matches!(
            OpportunityGraphV1::from_canonical_bytes(&bytes),
            Err(OpportunityGraphError::InvalidSchema(2))
        ));
    }

    #[test]
    fn matching_respects_supply_capacity_and_uses_global_optimum() {
        let demand_a = DemandId {
            kind: OpportunityDemandKind::WildlifePlacement,
            subject: Wildlife::Bear as u8,
            coord: HexCoord::new(0, 0),
        };
        let demand_b = DemandId {
            coord: HexCoord::new(1, 0),
            ..demand_a
        };
        let supply_a = SupplyId {
            kind: OpportunitySupplyKind::MarketWildlife,
            subject: 0,
        };
        let supply_b = SupplyId {
            kind: OpportunitySupplyKind::MarketWildlife,
            subject: 1,
        };
        let demands = [demand_a, demand_b]
            .into_iter()
            .map(|id| OpportunityDemand {
                id,
                deadline_turns: 1,
                exact_completion_delta: 1,
                local_same_species_neighbors: 0,
                frontier_neighbor_count: 0,
            })
            .collect::<Vec<_>>();
        let supplies = [supply_a, supply_b]
            .into_iter()
            .map(|id| OpportunitySupply {
                id,
                capacity: 1,
                availability_numerator: 1,
                availability_denominator: 1,
                access_delay_turns: 0,
                opponents_before_access: 0,
                wildlife: Some(Wildlife::Bear),
                archetype_id: None,
                market_slot: MarketSlot::new(id.subject as u8),
            })
            .collect::<Vec<_>>();
        let values = [
            (demand_a, supply_a, 9),
            (demand_a, supply_b, 8),
            (demand_b, supply_a, 8),
            (demand_b, supply_b, 1),
        ];
        let edges = values
            .into_iter()
            .map(|(demand, supply, value)| OpportunityEdge {
                id: OpportunityEdgeId { demand, supply },
                compatible_rotation_mask: 0,
                best_matching_edges: 0,
                exact_completion_delta: 1,
                teacher_value_micros: value,
            })
            .collect::<Vec<_>>();
        let graph = OpportunityGraphV1 {
            schema_version: OPPORTUNITY_GRAPH_SCHEMA_VERSION,
            source_public_state_hash: [0; 32],
            focal_seat: 0,
            completed_turns: 0,
            personal_turns_remaining: 1,
            exact_supply_hash: [0; 32],
            demands,
            supplies,
            edges,
        };
        let summary = graph.solve_matching().unwrap();
        assert_eq!(summary.teacher_value_micros, 16);
        assert_eq!(summary.matched_demands, 2);
    }
}
