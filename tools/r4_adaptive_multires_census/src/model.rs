use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use cascadia_data::{
    BOARD_ENTITY_SIZE, BOARD_SLOTS, MARKET_ENTITY_SIZE, MAX_BOARD_TILES, PositionRecord, TARGET_DIM,
};
use cascadia_game::{D6Transform, HexCoord, Wildlife};
use r2_sparse_entity_census::{
    AxialCoord, FrontierToken, GlobalMetadata, HabitatComponentToken, MarketToken,
    OccupiedTileToken, PlayerMetadata, SparsePublicState, SuppliedTile, WildlifeMotifToken,
};
use serde::{Deserialize, Serialize};

use crate::{R4Error, Result};

const NONE: u8 = u8::MAX;
const MODEL_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[repr(u8)]
pub enum NearFieldRadius {
    Radius4 = 4,
    Radius5 = 5,
}

pub type RadiusId = NearFieldRadius;

impl NearFieldRadius {
    pub const ALL: [Self; 2] = [Self::Radius4, Self::Radius5];

    pub const fn id(self) -> &'static str {
        match self {
            Self::Radius4 => "radius4-61",
            Self::Radius5 => "radius5-91",
        }
    }

    pub const fn radius(self) -> u8 {
        self as u8
    }

    pub const fn capacity(self) -> usize {
        centered_hex_capacity(self.radius())
    }

    pub const fn code(self) -> u8 {
        self as u8
    }

    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            4 => Some(Self::Radius4),
            5 => Some(Self::Radius5),
            _ => None,
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|radius| radius.id() == id)
    }
}

pub const fn centered_hex_capacity(radius: u8) -> usize {
    let radius = radius as usize;
    1 + 3 * radius * (radius + 1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureBlockSet {
    pub habitat: bool,
    pub wildlife: bool,
    pub frontier: bool,
    pub exact_far: bool,
}

impl FeatureBlockSet {
    pub const fn new(habitat: bool, wildlife: bool, frontier: bool, exact_far: bool) -> Self {
        Self {
            habitat,
            wildlife,
            frontier,
            exact_far,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[repr(u8)]
pub enum FeatureAblation {
    NearOnly = 0,
    Habitat = 1,
    Wildlife = 2,
    Frontier = 3,
    HabitatWildlife = 4,
    HabitatFrontier = 5,
    WildlifeFrontier = 6,
    AllTopology = 7,
    ExactFarControl = 8,
}

pub const ABLATIONS: [FeatureAblation; 9] = [
    FeatureAblation::NearOnly,
    FeatureAblation::Habitat,
    FeatureAblation::Wildlife,
    FeatureAblation::Frontier,
    FeatureAblation::HabitatWildlife,
    FeatureAblation::HabitatFrontier,
    FeatureAblation::WildlifeFrontier,
    FeatureAblation::AllTopology,
    FeatureAblation::ExactFarControl,
];

impl FeatureAblation {
    pub const fn id(self) -> &'static str {
        match self {
            Self::NearOnly => "n0-near-only",
            Self::Habitat => "h-habitat",
            Self::Wildlife => "w-wildlife",
            Self::Frontier => "f-frontier",
            Self::HabitatWildlife => "hw-habitat-wildlife",
            Self::HabitatFrontier => "hf-habitat-frontier",
            Self::WildlifeFrontier => "wf-wildlife-frontier",
            Self::AllTopology => "hwf-all-topology",
            Self::ExactFarControl => "e-exact-far-control",
        }
    }

    pub const fn blocks(self) -> FeatureBlockSet {
        match self {
            Self::NearOnly => FeatureBlockSet::new(false, false, false, false),
            Self::Habitat => FeatureBlockSet::new(true, false, false, false),
            Self::Wildlife => FeatureBlockSet::new(false, true, false, false),
            Self::Frontier => FeatureBlockSet::new(false, false, true, false),
            Self::HabitatWildlife => FeatureBlockSet::new(true, true, false, false),
            Self::HabitatFrontier => FeatureBlockSet::new(true, false, true, false),
            Self::WildlifeFrontier => FeatureBlockSet::new(false, true, true, false),
            Self::AllTopology => FeatureBlockSet::new(true, true, true, false),
            Self::ExactFarControl => FeatureBlockSet::new(false, false, false, true),
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        ABLATIONS.into_iter().find(|arm| arm.id() == id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RadialCount {
    pub distance: u16,
    pub count: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SectorCount {
    pub sector_bits: u8,
    pub count: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct HabitatPortal {
    pub local_index: u8,
    pub edge: u8,
    pub terrain: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FarHabitatComponent {
    pub relative_seat: u8,
    pub terrain: u8,
    pub member_count: u16,
    pub near_member_count: u16,
    pub far_member_count: u16,
    pub matching_internal_edge_count: u16,
    pub far_internal_edge_count: u16,
    pub near_far_crossing_edge_count: u16,
    pub open_boundary_edge_count: u16,
    pub frontier_contact_count: u16,
    pub degree_histogram: [u16; 7],
    pub radial_counts: Vec<RadialCount>,
    pub sector_counts: Vec<SectorCount>,
    pub local_member_indices: Vec<u8>,
    pub portals: Vec<HabitatPortal>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WildlifePortal {
    pub local_index: u8,
    pub edge: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FarWildlifeComponent {
    pub relative_seat: u8,
    pub wildlife: u8,
    pub member_count: u16,
    pub near_member_count: u16,
    pub far_member_count: u16,
    pub internal_edge_count: u16,
    pub near_far_crossing_edge_count: u16,
    pub degree_histogram: [u16; 7],
    pub endpoint_count: u16,
    pub branch_count: u16,
    pub graph_diameter: u16,
    pub edge_direction_counts: [u16; 3],
    pub max_collinear_run_by_axis: [u16; 3],
    pub radial_counts: Vec<RadialCount>,
    pub sector_counts: Vec<SectorCount>,
    pub local_member_indices: Vec<u8>,
    pub portals: Vec<WildlifePortal>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FarWildlifeMotifSignature {
    pub relative_seat: u8,
    pub wildlife: u8,
    pub distance: u16,
    pub sector_bits: u8,
    pub adjacent_wildlife_counts: [u8; 5],
    pub same_species_neighbor_count: u8,
    pub occupied_neighbor_count: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FarWildlifeMotifBucket {
    pub signature: FarWildlifeMotifSignature,
    pub count: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FarFrontierTouch {
    pub terrain: u8,
    pub component_size: u16,
    pub near_member_count: u16,
    pub far_member_count: u16,
    pub contact_edge_count: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FrontierBoundaryContact {
    pub local_index: u8,
    pub edge: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FarFrontierSignature {
    pub relative_seat: u8,
    pub distance: u16,
    pub sector_bits: u8,
    pub occupied_neighbor_count: u8,
    pub occupied_neighbor_runs: u8,
    pub opposite_neighbor_pair_count: u8,
    pub facing_terrain_counts: [u8; 5],
    pub adjacent_wildlife_counts: [u8; 5],
    pub touched_components: Vec<FarFrontierTouch>,
    pub resulting_size_by_terrain: [u16; 5],
    pub habitat_bridge_terrain_bits: u8,
    pub repeated_component_contact_terrain_bits: u8,
    pub boundary_contacts: Vec<FrontierBoundaryContact>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FarFrontierBucket {
    pub signature: FarFrontierSignature,
    pub count: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NearHabitatTouch {
    pub terrain: u8,
    pub component_size: u16,
    pub near_member_count: u16,
    pub far_member_count: u16,
    pub contact_edge_bits: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NearRotationCompatibility {
    pub rotation: u8,
    pub matching_edge_bits: u8,
    pub matching_edge_count: u8,
    pub all_present_edges_match: bool,
    pub resulting_size_by_terrain: [u16; 5],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NearSuppliedTileCompatibility {
    pub terrain_compatible_rotations: Vec<u8>,
    pub best_matching_edge_count: u8,
    pub rotations: Vec<NearRotationCompatibility>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NearOccupiedCell {
    pub semantic: [u8; 6],
    pub directed_edge_terrains: [u8; 6],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NearFrontierCell {
    pub neighbor_presence_bits: u8,
    pub neighbor_facing_terrains: [u8; 6],
    pub adjacent_wildlife_counts: [u8; 5],
    pub occupied_neighbor_runs: u8,
    pub opposite_neighbor_pair_bits: u8,
    pub touched_habitat_components: Vec<NearHabitatTouch>,
    pub resulting_size_by_terrain: [u16; 5],
    pub habitat_bridge_terrain_bits: u8,
    pub repeated_component_contact_terrain_bits: u8,
    pub supplied_tile_compatibility: Option<NearSuppliedTileCompatibility>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum NearCellState {
    OutsideRules,
    Empty,
    Frontier(NearFrontierCell),
    Occupied(NearOccupiedCell),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NearCell {
    pub index: u8,
    pub relative_q: i8,
    pub relative_r: i8,
    pub state: NearCellState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexedOccupiedTile {
    pub index: u8,
    pub tile: OccupiedTileToken,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExactFarBoard {
    pub relative_seat: u8,
    pub occupied_tiles: Vec<OccupiedTileToken>,
    pub legal_frontier: Vec<FrontierToken>,
    pub habitat_components: Vec<HabitatComponentToken>,
    pub wildlife_motifs: Vec<WildlifeMotifToken>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdaptiveBoard {
    pub relative_seat: u8,
    pub is_focal: bool,
    pub center: AxialCoord,
    pub authority_local_occupied: Vec<IndexedOccupiedTile>,
    pub authority_overflow_occupied: Vec<OccupiedTileToken>,
    pub near_cells: Vec<NearCell>,
    pub far_habitat_components: Vec<FarHabitatComponent>,
    pub far_wildlife_components: Vec<FarWildlifeComponent>,
    pub far_wildlife_motif_buckets: Vec<FarWildlifeMotifBucket>,
    pub far_frontier_buckets: Vec<FarFrontierBucket>,
    pub exact_far: ExactFarBoard,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdaptiveMultiResolutionState {
    pub schema_version: u16,
    pub radius: NearFieldRadius,
    pub focal_relative_seat: u8,
    pub global: GlobalMetadata,
    pub players: Vec<PlayerMetadata>,
    pub market: Vec<MarketToken>,
    pub supplied_tile: Option<SuppliedTile>,
    pub boards: Vec<AdaptiveBoard>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelGlobalView {
    pub turn: u8,
    pub current_relative_seat: u8,
    pub player_count: u8,
    pub total_turns: u8,
    pub scoring_cards: [u8; 5],
    pub habitat_bonuses: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelPlayerView {
    pub relative_seat: u8,
    pub turns_taken: u8,
    pub turns_until_next_action: u8,
    pub occupied_count: u8,
    pub nature_tokens: u8,
    pub wildlife_counts: [u8; 5],
    pub largest_habitats: [u8; 5],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdaptiveFeatureView {
    pub schema_version: u16,
    pub radius: NearFieldRadius,
    pub ablation: FeatureAblation,
    pub global: ModelGlobalView,
    pub players: Vec<ModelPlayerView>,
    pub market: Vec<MarketToken>,
    pub supplied_tile: Option<SuppliedTile>,
    pub near_cells: Vec<NearCell>,
    pub far_habitat_components: Vec<FarHabitatComponent>,
    pub far_wildlife_components: Vec<FarWildlifeComponent>,
    pub far_wildlife_motif_buckets: Vec<FarWildlifeMotifBucket>,
    pub far_frontier_buckets: Vec<FarFrontierBucket>,
    pub exact_far: Vec<ExactFarBoard>,
}

impl AdaptiveFeatureView {
    pub fn canonical_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(R4Error::from)
    }

    pub fn canonical_blake3(&self) -> Result<String> {
        Ok(blake3::hash(&self.canonical_bytes()?).to_hex().to_string())
    }

    pub fn spatial_token_count(&self) -> usize {
        self.near_cells.len()
            + self.far_habitat_components.len()
            + self.far_wildlife_components.len()
            + self.far_wildlife_motif_buckets.len()
            + self.far_frontier_buckets.len()
            + self
                .exact_far
                .iter()
                .map(|board| {
                    board.occupied_tiles.len()
                        + board.legal_frontier.len()
                        + board.habitat_components.len()
                        + board.wildlife_motifs.len()
                })
                .sum::<usize>()
    }
}

#[derive(Debug, Clone, Copy)]
struct ComponentScope {
    near_members: u16,
    far_members: u16,
}

impl AdaptiveMultiResolutionState {
    pub fn from_position_record(
        record: &PositionRecord,
        supplied_tile: Option<SuppliedTile>,
        radius: NearFieldRadius,
    ) -> Result<Self> {
        let sparse = SparsePublicState::from_position_record(record, supplied_tile)?;
        Self::from_sparse_state(&sparse, radius)
    }

    pub fn from_sparse_state(sparse: &SparsePublicState, radius: NearFieldRadius) -> Result<Self> {
        let centers = (0..sparse.global.player_count)
            .map(|seat| {
                let coordinates = sparse
                    .occupied_tiles
                    .iter()
                    .filter(|tile| tile.relative_seat == seat)
                    .map(|tile| tile.coord)
                    .collect::<Vec<_>>();
                deterministic_integer_center(&coordinates)
            })
            .collect::<Vec<_>>();
        Self::assemble(sparse, radius, &centers)
    }

    pub(crate) fn assemble(
        sparse: &SparsePublicState,
        radius: NearFieldRadius,
        centers: &[AxialCoord],
    ) -> Result<Self> {
        if centers.len() != usize::from(sparse.global.player_count) {
            return Err(R4Error::InvalidState(
                "center count does not match active player count".to_owned(),
            ));
        }
        let focal_relative_seat = sparse.global.current_relative_seat;
        if focal_relative_seat >= sparse.global.player_count {
            return Err(R4Error::InvalidState(
                "focal relative seat is not active".to_owned(),
            ));
        }

        let mut boards = Vec::with_capacity(centers.len());
        for relative_seat in 0..sparse.global.player_count {
            boards.push(build_board(
                sparse,
                relative_seat,
                focal_relative_seat,
                centers[usize::from(relative_seat)],
                radius,
            )?);
        }
        let state = Self {
            schema_version: MODEL_SCHEMA_VERSION,
            radius,
            focal_relative_seat,
            global: sparse.global.clone(),
            players: sparse.players.clone(),
            market: sparse.market.clone(),
            supplied_tile: sparse.supplied_tile,
            boards,
        };
        state.validate()?;
        if state.to_sparse_state()? != *sparse {
            return Err(R4Error::InvalidState(
                "adaptive representation changed the authoritative R2 state".to_owned(),
            ));
        }
        Ok(state)
    }

    pub fn to_position_record(&self) -> Result<PositionRecord> {
        self.validate()?;
        let mut record = PositionRecord {
            game_index: self.global.game_index,
            turn: self.global.turn,
            active_seat: self.global.perspective_absolute_seat,
            player_count: self.global.player_count,
            total_turns: self.global.total_turns,
            board_counts: [0; BOARD_SLOTS],
            nature_tokens: [0; BOARD_SLOTS],
            scoring_cards: self.global.scoring_cards,
            habitat_bonuses: self.global.habitat_bonuses,
            wildlife_counts: [[0; 5]; BOARD_SLOTS],
            habitat_sizes: [[0; 5]; BOARD_SLOTS],
            board_entities: [[[NONE; BOARD_ENTITY_SIZE]; MAX_BOARD_TILES]; BOARD_SLOTS],
            market_entities: [[NONE; MARKET_ENTITY_SIZE]; 4],
            targets: [0; TARGET_DIM],
        };
        for player in &self.players {
            let seat = usize::from(player.relative_seat);
            record.board_counts[seat] = player.occupied_count;
            record.nature_tokens[seat] = player.nature_tokens;
            record.wildlife_counts[seat] = player.wildlife_counts;
            record.habitat_sizes[seat] = player.largest_habitats;
            let board = &self.boards[seat];
            let mut tiles = board
                .authority_local_occupied
                .iter()
                .map(|indexed| indexed.tile.clone())
                .chain(board.authority_overflow_occupied.iter().cloned())
                .collect::<Vec<_>>();
            tiles.sort_unstable_by_key(|tile| (tile.coord.q, tile.coord.r));
            if tiles.len() != usize::from(player.occupied_count) {
                return Err(R4Error::InvalidState(format!(
                    "relative seat {} reconstructed {} tiles; expected {}",
                    player.relative_seat,
                    tiles.len(),
                    player.occupied_count
                )));
            }
            for (row, tile) in tiles.iter().enumerate() {
                record.board_entities[seat][row] = occupied_to_record_entity(tile)?;
            }
        }
        for token in &self.market {
            record.market_entities[usize::from(token.slot)] = market_to_record_entity(token);
        }
        Ok(record)
    }

    pub fn to_sparse_state(&self) -> Result<SparsePublicState> {
        Ok(SparsePublicState::from_position_record(
            &self.to_position_record()?,
            self.supplied_tile,
        )?)
    }

    pub fn transformed(&self, transform: D6Transform) -> Result<Self> {
        let sparse = self.to_sparse_state()?.transformed(transform)?;
        let centers = self
            .boards
            .iter()
            .map(|board| board.center.transformed(transform))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Self::assemble(&sparse, self.radius, &centers)
    }

    pub fn feature_view(&self, ablation: FeatureAblation) -> Result<AdaptiveFeatureView> {
        self.validate()?;
        let blocks = ablation.blocks();
        let focal = self
            .boards
            .get(usize::from(self.focal_relative_seat))
            .ok_or_else(|| R4Error::InvalidFeatureView("missing focal board".to_owned()))?;
        let view = AdaptiveFeatureView {
            schema_version: MODEL_SCHEMA_VERSION,
            radius: self.radius,
            ablation,
            global: ModelGlobalView {
                turn: self.global.turn,
                current_relative_seat: self.global.current_relative_seat,
                player_count: self.global.player_count,
                total_turns: self.global.total_turns,
                scoring_cards: self.global.scoring_cards,
                habitat_bonuses: self.global.habitat_bonuses,
            },
            players: self
                .players
                .iter()
                .map(|player| ModelPlayerView {
                    relative_seat: player.relative_seat,
                    turns_taken: player.turns_taken,
                    turns_until_next_action: player.turns_until_next_action,
                    occupied_count: player.occupied_count,
                    nature_tokens: player.nature_tokens,
                    wildlife_counts: player.wildlife_counts,
                    largest_habitats: player.largest_habitats,
                })
                .collect(),
            market: self.market.clone(),
            supplied_tile: self.supplied_tile,
            near_cells: focal.near_cells.clone(),
            far_habitat_components: if blocks.habitat {
                self.boards
                    .iter()
                    .flat_map(|board| board.far_habitat_components.clone())
                    .collect()
            } else {
                Vec::new()
            },
            far_wildlife_components: if blocks.wildlife {
                self.boards
                    .iter()
                    .flat_map(|board| board.far_wildlife_components.clone())
                    .collect()
            } else {
                Vec::new()
            },
            far_wildlife_motif_buckets: if blocks.wildlife {
                self.boards
                    .iter()
                    .flat_map(|board| board.far_wildlife_motif_buckets.clone())
                    .collect()
            } else {
                Vec::new()
            },
            far_frontier_buckets: if blocks.frontier {
                self.boards
                    .iter()
                    .flat_map(|board| board.far_frontier_buckets.clone())
                    .collect()
            } else {
                Vec::new()
            },
            exact_far: if blocks.exact_far {
                self.boards
                    .iter()
                    .map(|board| board.exact_far.clone())
                    .collect()
            } else {
                Vec::new()
            },
        };
        Ok(view)
    }

    pub fn canonical_blake3(&self) -> Result<String> {
        Ok(blake3::hash(&self.to_packed_bytes()?).to_hex().to_string())
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.schema_version != MODEL_SCHEMA_VERSION {
            return Err(R4Error::InvalidState(format!(
                "schema version {} is unsupported",
                self.schema_version
            )));
        }
        if self.global.player_count == 0
            || usize::from(self.global.player_count) > BOARD_SLOTS
            || self.boards.len() != usize::from(self.global.player_count)
            || self.players.len() != usize::from(self.global.player_count)
        {
            return Err(R4Error::InvalidState(
                "active player, board, and metadata counts disagree".to_owned(),
            ));
        }
        if self.focal_relative_seat != self.global.current_relative_seat
            || self.focal_relative_seat >= self.global.player_count
        {
            return Err(R4Error::InvalidState(
                "focal seat differs from the current relative seat".to_owned(),
            ));
        }
        for (seat, board) in self.boards.iter().enumerate() {
            if usize::from(board.relative_seat) != seat
                || board.is_focal != (board.relative_seat == self.focal_relative_seat)
            {
                return Err(R4Error::InvalidState(
                    "board order or focal marker is noncanonical".to_owned(),
                ));
            }
            validate_board(board, self.radius)?;
        }
        if self.boards.iter().filter(|board| board.is_focal).count() != 1 {
            return Err(R4Error::InvalidState(
                "exactly one board must be focal".to_owned(),
            ));
        }
        Ok(())
    }
}

fn build_board(
    sparse: &SparsePublicState,
    relative_seat: u8,
    focal_relative_seat: u8,
    center: AxialCoord,
    radius: NearFieldRadius,
) -> Result<AdaptiveBoard> {
    let is_focal = relative_seat == focal_relative_seat;
    let occupied = sparse
        .occupied_tiles
        .iter()
        .filter(|tile| tile.relative_seat == relative_seat)
        .cloned()
        .collect::<Vec<_>>();
    let frontier = sparse
        .legal_frontier
        .iter()
        .filter(|token| token.relative_seat == relative_seat)
        .cloned()
        .collect::<Vec<_>>();
    let components = sparse
        .habitat_components
        .iter()
        .filter(|token| token.relative_seat == relative_seat)
        .cloned()
        .collect::<Vec<_>>();
    let motifs = sparse
        .wildlife_motifs
        .iter()
        .filter(|token| token.relative_seat == relative_seat)
        .cloned()
        .collect::<Vec<_>>();

    let mut authority_local_occupied = Vec::new();
    let mut authority_overflow_occupied = Vec::new();
    for tile in &occupied {
        let relative = subtract_coords(tile.coord, center)?;
        if let Some(index) = hex_disk_index(radius.radius(), relative) {
            authority_local_occupied.push(IndexedOccupiedTile {
                index: u8::try_from(index).expect("radius-five index fits in u8"),
                tile: tile.clone(),
            });
        } else {
            authority_overflow_occupied.push(tile.clone());
        }
    }
    authority_local_occupied.sort_unstable_by_key(|tile| tile.index);
    authority_overflow_occupied.sort_unstable_by_key(|tile| (tile.coord.q, tile.coord.r));

    let component_scopes = component_scopes(&components, center, radius, is_focal);
    let near_cells = if is_focal {
        build_near_cells(
            &occupied,
            &frontier,
            &components,
            &component_scopes,
            center,
            radius,
        )?
    } else {
        Vec::new()
    };
    let far_habitat_components =
        build_far_habitat_components(&occupied, &components, center, radius, is_focal)?;
    let (far_wildlife_components, far_wildlife_motif_buckets) =
        build_far_wildlife(&occupied, &motifs, center, radius, is_focal)?;
    let far_frontier_buckets = build_far_frontier(
        &frontier,
        &components,
        &component_scopes,
        center,
        radius,
        is_focal,
    )?;
    let exact_far = ExactFarBoard {
        relative_seat,
        occupied_tiles: occupied
            .iter()
            .filter(|tile| !is_local(tile.coord, center, radius, is_focal))
            .cloned()
            .collect(),
        legal_frontier: frontier
            .iter()
            .filter(|token| !is_local(token.coord, center, radius, is_focal))
            .cloned()
            .collect(),
        habitat_components: components
            .iter()
            .filter(|component| {
                component
                    .members
                    .iter()
                    .any(|coord| !is_local(*coord, center, radius, is_focal))
            })
            .cloned()
            .collect(),
        wildlife_motifs: motifs
            .iter()
            .filter(|motif| !is_local(motif.coord, center, radius, is_focal))
            .cloned()
            .collect(),
    };

    Ok(AdaptiveBoard {
        relative_seat,
        is_focal,
        center,
        authority_local_occupied,
        authority_overflow_occupied,
        near_cells,
        far_habitat_components,
        far_wildlife_components,
        far_wildlife_motif_buckets,
        far_frontier_buckets,
        exact_far,
    })
}

fn validate_board(board: &AdaptiveBoard, radius: NearFieldRadius) -> Result<()> {
    if board
        .authority_local_occupied
        .windows(2)
        .any(|pair| pair[0].index >= pair[1].index)
    {
        return Err(R4Error::InvalidState(format!(
            "relative seat {} local occupied indices are not strictly ordered",
            board.relative_seat
        )));
    }
    if board
        .authority_overflow_occupied
        .windows(2)
        .any(|pair| (pair[0].coord.q, pair[0].coord.r) >= (pair[1].coord.q, pair[1].coord.r))
    {
        return Err(R4Error::InvalidState(format!(
            "relative seat {} overflow coordinates are not strictly ordered",
            board.relative_seat
        )));
    }
    let mut coordinates = BTreeSet::new();
    for indexed in &board.authority_local_occupied {
        let relative = hex_disk_coord(radius.radius(), u16::from(indexed.index))
            .ok_or_else(|| R4Error::InvalidState("local occupied index is invalid".to_owned()))?;
        if add_coords(relative, board.center)? != indexed.tile.coord
            || !coordinates.insert(indexed.tile.coord)
        {
            return Err(R4Error::InvalidState(
                "local occupied coordinate/index mismatch".to_owned(),
            ));
        }
    }
    for tile in &board.authority_overflow_occupied {
        if hex_disk_index(radius.radius(), subtract_coords(tile.coord, board.center)?).is_some()
            || !coordinates.insert(tile.coord)
        {
            return Err(R4Error::InvalidState(
                "overflow occupied tile is local or duplicated".to_owned(),
            ));
        }
    }
    if board.is_focal {
        if board.near_cells.len() != radius.capacity()
            || board
                .near_cells
                .iter()
                .enumerate()
                .any(|(index, cell)| usize::from(cell.index) != index)
        {
            return Err(R4Error::InvalidState(
                "focal near field is not the complete canonical disk".to_owned(),
            ));
        }
    } else if !board.near_cells.is_empty() {
        return Err(R4Error::InvalidState(
            "nonfocal board contains near-field cells".to_owned(),
        ));
    }
    Ok(())
}

fn component_scopes(
    components: &[HabitatComponentToken],
    center: AxialCoord,
    radius: NearFieldRadius,
    is_focal: bool,
) -> HashMap<u16, ComponentScope> {
    components
        .iter()
        .map(|component| {
            let near_members = component
                .members
                .iter()
                .filter(|coord| is_local(**coord, center, radius, is_focal))
                .count() as u16;
            (
                component.component_id,
                ComponentScope {
                    near_members,
                    far_members: component.member_count - near_members,
                },
            )
        })
        .collect()
}

fn build_near_cells(
    occupied: &[OccupiedTileToken],
    frontier: &[FrontierToken],
    components: &[HabitatComponentToken],
    scopes: &HashMap<u16, ComponentScope>,
    center: AxialCoord,
    radius: NearFieldRadius,
) -> Result<Vec<NearCell>> {
    let occupied_by_coord = occupied
        .iter()
        .map(|tile| (tile.coord, tile))
        .collect::<HashMap<_, _>>();
    let frontier_by_coord = frontier
        .iter()
        .map(|token| (token.coord, token))
        .collect::<HashMap<_, _>>();
    let components_by_id = components
        .iter()
        .map(|component| (component.component_id, component))
        .collect::<HashMap<_, _>>();
    let mut cells = Vec::with_capacity(radius.capacity());
    for raw_index in 0..radius.capacity() {
        let index = raw_index as u16;
        let relative = hex_disk_coord(radius.radius(), index)
            .expect("complete disk index is valid by construction");
        let absolute = add_coords(relative, center)?;
        let state = if !is_in_rules_grid(absolute) {
            NearCellState::OutsideRules
        } else if let Some(tile) = occupied_by_coord.get(&absolute) {
            NearCellState::Occupied(NearOccupiedCell {
                semantic: occupied_semantic(tile),
                directed_edge_terrains: tile.directed_edge_terrains.map(|terrain| terrain as u8),
            })
        } else if let Some(token) = frontier_by_coord.get(&absolute) {
            NearCellState::Frontier(near_frontier(token, &components_by_id, scopes)?)
        } else {
            NearCellState::Empty
        };
        cells.push(NearCell {
            index: raw_index as u8,
            relative_q: i8::try_from(relative.q).expect("radius-five coordinate fits i8"),
            relative_r: i8::try_from(relative.r).expect("radius-five coordinate fits i8"),
            state,
        });
    }
    Ok(cells)
}

fn near_frontier(
    token: &FrontierToken,
    components: &HashMap<u16, &HabitatComponentToken>,
    scopes: &HashMap<u16, ComponentScope>,
) -> Result<NearFrontierCell> {
    let touched_habitat_components = token
        .touched_habitat_components
        .iter()
        .map(|touch| {
            let component = components.get(&touch.component_id).ok_or_else(|| {
                R4Error::InvalidState("frontier references an absent component".to_owned())
            })?;
            let scope = scopes[&touch.component_id];
            if component.member_count != touch.component_size {
                return Err(R4Error::InvalidState(
                    "frontier component size disagrees with component token".to_owned(),
                ));
            }
            Ok(NearHabitatTouch {
                terrain: touch.terrain as u8,
                component_size: touch.component_size,
                near_member_count: scope.near_members,
                far_member_count: scope.far_members,
                contact_edge_bits: touch.contact_edge_bits,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let supplied_tile_compatibility =
        token
            .supplied_tile_compatibility
            .as_ref()
            .map(|compat| NearSuppliedTileCompatibility {
                terrain_compatible_rotations: compat.terrain_compatible_rotations.clone(),
                best_matching_edge_count: compat.best_matching_edge_count,
                rotations: compat
                    .rotations
                    .iter()
                    .map(|rotation| {
                        let mut resulting_size_by_terrain = [0u16; 5];
                        for merge in &rotation.habitat_merges {
                            resulting_size_by_terrain[merge.terrain as usize] =
                                merge.resulting_size;
                        }
                        NearRotationCompatibility {
                            rotation: rotation.rotation,
                            matching_edge_bits: rotation.matching_edge_bits,
                            matching_edge_count: rotation.matching_edge_count,
                            all_present_edges_match: rotation.all_present_edges_match,
                            resulting_size_by_terrain,
                        }
                    })
                    .collect(),
            });
    Ok(NearFrontierCell {
        neighbor_presence_bits: token.neighbor_presence_bits,
        neighbor_facing_terrains: token
            .neighbor_facing_terrains
            .map(|terrain| terrain.map_or(NONE, |terrain| terrain as u8)),
        adjacent_wildlife_counts: token.adjacent_wildlife_counts,
        occupied_neighbor_runs: token.occupied_neighbor_runs,
        opposite_neighbor_pair_bits: token.opposite_neighbor_pair_bits,
        touched_habitat_components,
        resulting_size_by_terrain: token.resulting_size_by_terrain,
        habitat_bridge_terrain_bits: token.habitat_bridge_terrain_bits,
        repeated_component_contact_terrain_bits: token.repeated_component_contact_terrain_bits,
        supplied_tile_compatibility,
    })
}

fn build_far_habitat_components(
    occupied: &[OccupiedTileToken],
    components: &[HabitatComponentToken],
    center: AxialCoord,
    radius: NearFieldRadius,
    is_focal: bool,
) -> Result<Vec<FarHabitatComponent>> {
    let occupied_by_coord = occupied
        .iter()
        .map(|tile| (tile.coord, tile))
        .collect::<HashMap<_, _>>();
    let mut output = Vec::new();
    for component in components {
        let local_indices = component
            .members
            .iter()
            .filter_map(|coord| {
                is_local(*coord, center, radius, is_focal)
                    .then(|| local_index_u8(*coord, center, radius))
                    .flatten()
            })
            .collect::<Vec<_>>();
        let near_member_count = local_indices.len() as u16;
        let far_member_count = component.member_count - near_member_count;
        if far_member_count == 0 {
            continue;
        }
        let members = component.members.iter().copied().collect::<HashSet<_>>();
        let mut degree_histogram = [0u16; 7];
        let mut far_internal_edge_count = 0u16;
        let mut near_far_crossing_edge_count = 0u16;
        let mut portals = Vec::new();
        for coord in &component.members {
            let tile = occupied_by_coord[coord];
            let local = is_local(*coord, center, radius, is_focal);
            let mut degree = 0usize;
            for edge in 0..6 {
                if tile.directed_edge_terrains[edge] != component.terrain {
                    continue;
                }
                let neighbor = coord.neighbor(edge);
                if !members.contains(&neighbor) {
                    continue;
                }
                degree += 1;
                if *coord < neighbor {
                    let neighbor_local = is_local(neighbor, center, radius, is_focal);
                    if !local && !neighbor_local {
                        far_internal_edge_count += 1;
                    } else if local != neighbor_local {
                        near_far_crossing_edge_count += 1;
                        let (local_coord, local_edge) = if local {
                            (*coord, edge)
                        } else {
                            (neighbor, (edge + 3) % 6)
                        };
                        portals.push(HabitatPortal {
                            local_index: local_index_u8(local_coord, center, radius)
                                .expect("crossing edge has one local endpoint"),
                            edge: local_edge as u8,
                            terrain: component.terrain as u8,
                        });
                    }
                }
            }
            degree_histogram[degree] += 1;
        }
        portals.sort_unstable();
        let far_members = component
            .members
            .iter()
            .copied()
            .filter(|coord| !is_local(*coord, center, radius, is_focal))
            .collect::<Vec<_>>();
        output.push(FarHabitatComponent {
            relative_seat: component.relative_seat,
            terrain: component.terrain as u8,
            member_count: component.member_count,
            near_member_count,
            far_member_count,
            matching_internal_edge_count: component.matching_internal_edge_count,
            far_internal_edge_count,
            near_far_crossing_edge_count,
            open_boundary_edge_count: component.open_boundary_edge_count,
            frontier_contact_count: component.frontier_contact_count,
            degree_histogram,
            radial_counts: radial_counts(&far_members, center),
            sector_counts: sector_counts(&far_members, center),
            local_member_indices: local_indices,
            portals,
        });
    }
    output.sort_unstable_by(|left, right| {
        serde_json::to_vec(left)
            .expect("far habitat token serializes")
            .cmp(&serde_json::to_vec(right).expect("far habitat token serializes"))
    });
    Ok(output)
}

fn build_far_wildlife(
    occupied: &[OccupiedTileToken],
    motifs: &[WildlifeMotifToken],
    center: AxialCoord,
    radius: NearFieldRadius,
    is_focal: bool,
) -> Result<(Vec<FarWildlifeComponent>, Vec<FarWildlifeMotifBucket>)> {
    let wildlife_by_coord = occupied
        .iter()
        .filter_map(|tile| tile.placed_wildlife.map(|wildlife| (tile.coord, wildlife)))
        .collect::<HashMap<_, _>>();
    let mut components = Vec::new();
    for wildlife in [
        Wildlife::Bear,
        Wildlife::Elk,
        Wildlife::Salmon,
        Wildlife::Hawk,
        Wildlife::Fox,
    ] {
        let mut remaining = wildlife_by_coord
            .iter()
            .filter_map(|(coord, species)| (*species == wildlife).then_some(*coord))
            .collect::<BTreeSet<_>>();
        while let Some(start) = remaining.pop_first() {
            let mut members = Vec::new();
            let mut queue = VecDeque::from([start]);
            while let Some(coord) = queue.pop_front() {
                members.push(coord);
                for neighbor in coord.neighbors() {
                    if wildlife_by_coord.get(&neighbor) == Some(&wildlife)
                        && remaining.remove(&neighbor)
                    {
                        queue.push_back(neighbor);
                    }
                }
            }
            members.sort_unstable();
            let near_member_count = members
                .iter()
                .filter(|coord| is_local(**coord, center, radius, is_focal))
                .count() as u16;
            let far_member_count = members.len() as u16 - near_member_count;
            if far_member_count == 0 {
                continue;
            }
            let member_set = members.iter().copied().collect::<HashSet<_>>();
            let mut degree_histogram = [0u16; 7];
            let mut internal_edge_count = 0u16;
            let mut crossing = 0u16;
            let mut edge_direction_counts = [0u16; 3];
            let mut portals = Vec::new();
            for coord in &members {
                let local = is_local(*coord, center, radius, is_focal);
                let mut degree = 0usize;
                for edge in 0..6 {
                    let neighbor = coord.neighbor(edge);
                    if !member_set.contains(&neighbor) {
                        continue;
                    }
                    degree += 1;
                    if *coord < neighbor {
                        internal_edge_count += 1;
                        edge_direction_counts[edge % 3] += 1;
                        let neighbor_local = is_local(neighbor, center, radius, is_focal);
                        if local != neighbor_local {
                            crossing += 1;
                            let (local_coord, local_edge) = if local {
                                (*coord, edge)
                            } else {
                                (neighbor, (edge + 3) % 6)
                            };
                            portals.push(WildlifePortal {
                                local_index: local_index_u8(local_coord, center, radius)
                                    .expect("wildlife crossing has one local endpoint"),
                                edge: local_edge as u8,
                            });
                        }
                    }
                }
                degree_histogram[degree] += 1;
            }
            portals.sort_unstable();
            let far_members = members
                .iter()
                .copied()
                .filter(|coord| !is_local(*coord, center, radius, is_focal))
                .collect::<Vec<_>>();
            let local_member_indices = members
                .iter()
                .filter_map(|coord| local_index_u8(*coord, center, radius))
                .collect();
            components.push(FarWildlifeComponent {
                relative_seat: occupied.first().map_or(0, |tile| tile.relative_seat),
                wildlife: wildlife as u8,
                member_count: members.len() as u16,
                near_member_count,
                far_member_count,
                internal_edge_count,
                near_far_crossing_edge_count: crossing,
                degree_histogram,
                endpoint_count: degree_histogram[1],
                branch_count: degree_histogram[3..].iter().sum(),
                graph_diameter: graph_diameter(&members),
                edge_direction_counts,
                max_collinear_run_by_axis: max_collinear_runs(&member_set),
                radial_counts: radial_counts(&far_members, center),
                sector_counts: sector_counts(&far_members, center),
                local_member_indices,
                portals,
            });
        }
    }
    components.sort_unstable_by(|left, right| {
        serde_json::to_vec(left)
            .expect("far wildlife component serializes")
            .cmp(&serde_json::to_vec(right).expect("far wildlife component serializes"))
    });

    let mut buckets = BTreeMap::<FarWildlifeMotifSignature, u16>::new();
    for motif in motifs
        .iter()
        .filter(|motif| !is_local(motif.coord, center, radius, is_focal))
    {
        let signature = FarWildlifeMotifSignature {
            relative_seat: motif.relative_seat,
            wildlife: motif.wildlife as u8,
            distance: hex_distance(motif.coord, center),
            sector_bits: direction_sector_bits(subtract_coords(motif.coord, center)?),
            adjacent_wildlife_counts: motif.adjacent_wildlife_counts,
            same_species_neighbor_count: motif.same_species_neighbor_bits.count_ones() as u8,
            occupied_neighbor_count: motif
                .neighbor_wildlife
                .iter()
                .filter(|value| value.is_some())
                .count() as u8,
        };
        *buckets.entry(signature).or_default() += 1;
    }
    Ok((
        components,
        buckets
            .into_iter()
            .map(|(signature, count)| FarWildlifeMotifBucket { signature, count })
            .collect(),
    ))
}

fn build_far_frontier(
    frontier: &[FrontierToken],
    components: &[HabitatComponentToken],
    scopes: &HashMap<u16, ComponentScope>,
    center: AxialCoord,
    radius: NearFieldRadius,
    is_focal: bool,
) -> Result<Vec<FarFrontierBucket>> {
    let component_by_id = components
        .iter()
        .map(|component| (component.component_id, component))
        .collect::<HashMap<_, _>>();
    let mut buckets = BTreeMap::<FarFrontierSignature, u16>::new();
    for token in frontier
        .iter()
        .filter(|token| !is_local(token.coord, center, radius, is_focal))
    {
        let relative = subtract_coords(token.coord, center)?;
        let mut facing_terrain_counts = [0u8; 5];
        for terrain in token.neighbor_facing_terrains.iter().flatten() {
            facing_terrain_counts[*terrain as usize] += 1;
        }
        let mut touched_components = token
            .touched_habitat_components
            .iter()
            .map(|touch| {
                let component = component_by_id.get(&touch.component_id).ok_or_else(|| {
                    R4Error::InvalidState("far frontier references absent component".to_owned())
                })?;
                let scope = scopes[&touch.component_id];
                if component.member_count != touch.component_size {
                    return Err(R4Error::InvalidState(
                        "far frontier component size mismatch".to_owned(),
                    ));
                }
                Ok(FarFrontierTouch {
                    terrain: touch.terrain as u8,
                    component_size: touch.component_size,
                    near_member_count: scope.near_members,
                    far_member_count: scope.far_members,
                    contact_edge_count: touch.contact_edge_bits.count_ones() as u8,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        touched_components.sort_unstable();
        let mut boundary_contacts = Vec::new();
        if is_focal {
            for edge in 0..6 {
                let neighbor = token.coord.neighbor(edge);
                if let Some(local_index) = local_index_u8(neighbor, center, radius) {
                    boundary_contacts.push(FrontierBoundaryContact {
                        local_index,
                        edge: edge as u8,
                    });
                }
            }
            boundary_contacts.sort_unstable();
        }
        let signature = FarFrontierSignature {
            relative_seat: token.relative_seat,
            distance: hex_distance(token.coord, center),
            sector_bits: direction_sector_bits(relative),
            occupied_neighbor_count: token.neighbor_presence_bits.count_ones() as u8,
            occupied_neighbor_runs: token.occupied_neighbor_runs,
            opposite_neighbor_pair_count: token.opposite_neighbor_pair_bits.count_ones() as u8,
            facing_terrain_counts,
            adjacent_wildlife_counts: token.adjacent_wildlife_counts,
            touched_components,
            resulting_size_by_terrain: token.resulting_size_by_terrain,
            habitat_bridge_terrain_bits: token.habitat_bridge_terrain_bits,
            repeated_component_contact_terrain_bits: token.repeated_component_contact_terrain_bits,
            boundary_contacts,
        };
        *buckets.entry(signature).or_default() += 1;
    }
    Ok(buckets
        .into_iter()
        .map(|(signature, count)| FarFrontierBucket { signature, count })
        .collect())
}

fn radial_counts(coordinates: &[AxialCoord], center: AxialCoord) -> Vec<RadialCount> {
    let mut counts = BTreeMap::<u16, u16>::new();
    for coord in coordinates {
        *counts.entry(hex_distance(*coord, center)).or_default() += 1;
    }
    counts
        .into_iter()
        .map(|(distance, count)| RadialCount { distance, count })
        .collect()
}

fn sector_counts(coordinates: &[AxialCoord], center: AxialCoord) -> Vec<SectorCount> {
    let mut counts = BTreeMap::<u8, u16>::new();
    for coord in coordinates {
        let relative = AxialCoord::new(coord.q - center.q, coord.r - center.r);
        *counts.entry(direction_sector_bits(relative)).or_default() += 1;
    }
    counts
        .into_iter()
        .map(|(sector_bits, count)| SectorCount { sector_bits, count })
        .collect()
}

fn direction_sector_bits(relative: AxialCoord) -> u8 {
    if relative == AxialCoord::ORIGIN {
        return 0;
    }
    let q = i32::from(relative.q);
    let r = i32::from(relative.r);
    let s = -q - r;
    let cube_directions = [
        (1, 0, -1),
        (1, -1, 0),
        (0, -1, 1),
        (-1, 0, 1),
        (-1, 1, 0),
        (0, 1, -1),
    ];
    let scores = cube_directions.map(|(dq, dr, ds)| q * dq + r * dr + s * ds);
    let maximum = *scores.iter().max().expect("six directions exist");
    scores.iter().enumerate().fold(0u8, |bits, (index, score)| {
        bits | u8::from(*score == maximum) << index
    })
}

fn graph_diameter(members: &[AxialCoord]) -> u16 {
    if members.len() <= 1 {
        return 0;
    }
    let member_set = members.iter().copied().collect::<HashSet<_>>();
    let mut diameter = 0usize;
    for start in members {
        let mut distances = HashMap::from([(*start, 0usize)]);
        let mut queue = VecDeque::from([*start]);
        while let Some(coord) = queue.pop_front() {
            let distance = distances[&coord];
            diameter = diameter.max(distance);
            for neighbor in coord.neighbors() {
                if member_set.contains(&neighbor) && !distances.contains_key(&neighbor) {
                    distances.insert(neighbor, distance + 1);
                    queue.push_back(neighbor);
                }
            }
        }
    }
    diameter as u16
}

fn max_collinear_runs(members: &HashSet<AxialCoord>) -> [u16; 3] {
    let axes = [(1i16, 0i16), (1, -1), (0, -1)];
    std::array::from_fn(|axis| {
        let (dq, dr) = axes[axis];
        members
            .iter()
            .filter(|coord| !members.contains(&AxialCoord::new(coord.q - dq, coord.r - dr)))
            .map(|start| {
                let mut count = 0u16;
                let mut cursor = *start;
                while members.contains(&cursor) {
                    count += 1;
                    cursor = AxialCoord::new(cursor.q + dq, cursor.r + dr);
                }
                count
            })
            .max()
            .unwrap_or(0)
    })
}

fn occupied_semantic(tile: &OccupiedTileToken) -> [u8; 6] {
    [
        tile.terrain_a as u8,
        tile.terrain_b.map_or(NONE, |terrain| terrain as u8),
        tile.rotation.get(),
        tile.wildlife_eligibility.bits(),
        tile.placed_wildlife.map_or(NONE, |wildlife| wildlife as u8),
        u8::from(tile.keystone),
    ]
}

pub(crate) fn occupied_to_record_entity(
    tile: &OccupiedTileToken,
) -> Result<[u8; BOARD_ENTITY_SIZE]> {
    let q = i8::try_from(tile.coord.q).map_err(|_| {
        R4Error::InvalidState(format!(
            "coordinate q={} does not fit the source record",
            tile.coord.q
        ))
    })?;
    let r = i8::try_from(tile.coord.r).map_err(|_| {
        R4Error::InvalidState(format!(
            "coordinate r={} does not fit the source record",
            tile.coord.r
        ))
    })?;
    let semantic = occupied_semantic(tile);
    Ok([
        q as u8,
        r as u8,
        semantic[0],
        semantic[1],
        semantic[2],
        semantic[3],
        semantic[4],
        semantic[5],
    ])
}

pub(crate) fn market_to_record_entity(token: &MarketToken) -> [u8; MARKET_ENTITY_SIZE] {
    if token.tile.is_none() && token.wildlife.is_none() {
        return [NONE; MARKET_ENTITY_SIZE];
    }
    [
        token.tile.map_or(NONE, |tile| tile.terrain_a as u8),
        token
            .tile
            .and_then(|tile| tile.terrain_b)
            .map_or(NONE, |terrain| terrain as u8),
        token
            .tile
            .map_or(0, |tile| tile.wildlife_eligibility.bits()),
        token.wildlife.map_or(NONE, |wildlife| wildlife as u8),
        token.tile.map_or(0, |tile| u8::from(tile.keystone)),
        0,
        0,
        0,
    ]
}

fn is_local(
    coord: AxialCoord,
    center: AxialCoord,
    radius: NearFieldRadius,
    is_focal: bool,
) -> bool {
    is_focal && hex_distance(coord, center) <= u16::from(radius.radius())
}

fn local_index_u8(coord: AxialCoord, center: AxialCoord, radius: NearFieldRadius) -> Option<u8> {
    let relative = AxialCoord::new(coord.q - center.q, coord.r - center.r);
    hex_disk_index(radius.radius(), relative).map(|index| index as u8)
}

pub fn deterministic_integer_center(coordinates: &[AxialCoord]) -> AxialCoord {
    if coordinates.is_empty() {
        return AxialCoord::ORIGIN;
    }
    let mut min_q = i16::MAX;
    let mut max_q = i16::MIN;
    let mut min_r = i16::MAX;
    let mut max_r = i16::MIN;
    let mut min_s = i16::MAX;
    let mut max_s = i16::MIN;
    for coord in coordinates {
        let s = -coord.q - coord.r;
        min_q = min_q.min(coord.q);
        max_q = max_q.max(coord.q);
        min_r = min_r.min(coord.r);
        max_r = max_r.max(coord.r);
        min_s = min_s.min(s);
        max_s = max_s.max(s);
    }
    let lower_bound = [max_q - min_q, max_r - min_r, max_s - min_s]
        .into_iter()
        .max()
        .map_or(0, |span| (span + 1) / 2);
    for radius in lower_bound..=48 {
        let q_low = max_q - radius;
        let q_high = min_q + radius;
        let r_low = max_r - radius;
        let r_high = min_r + radius;
        let s_low = max_s - radius;
        let s_high = min_s + radius;
        if q_low > q_high
            || r_low > r_high
            || s_low > s_high
            || q_low + r_low + s_low > 0
            || q_high + r_high + s_high < 0
        {
            continue;
        }
        for q in q_low..=q_high {
            let minimum_r = r_low.max(-s_high - q);
            let maximum_r = r_high.min(-s_low - q);
            if minimum_r <= maximum_r {
                return AxialCoord::new(q, minimum_r);
            }
        }
    }
    unreachable!("supported V2 coordinates always have an integer minimax center")
}

pub fn hex_disk_index(radius: u8, coord: AxialCoord) -> Option<u16> {
    if hex_distance(coord, AxialCoord::ORIGIN) > u16::from(radius) {
        return None;
    }
    let radius = i16::from(radius);
    let mut index = 0usize;
    for q in -radius..coord.q {
        let (low, high) = axial_r_bounds(radius, q);
        index += usize::try_from(high - low + 1).ok()?;
    }
    let (low, high) = axial_r_bounds(radius, coord.q);
    if !(low..=high).contains(&coord.r) {
        return None;
    }
    index += usize::try_from(coord.r - low).ok()?;
    u16::try_from(index).ok()
}

pub fn hex_disk_coord(radius: u8, index: u16) -> Option<AxialCoord> {
    if usize::from(index) >= centered_hex_capacity(radius) {
        return None;
    }
    let radius = i16::from(radius);
    let mut remaining = usize::from(index);
    for q in -radius..=radius {
        let (low, high) = axial_r_bounds(radius, q);
        let row_len = usize::try_from(high - low + 1).ok()?;
        if remaining < row_len {
            return Some(AxialCoord::new(q, low + remaining as i16));
        }
        remaining -= row_len;
    }
    None
}

const fn axial_r_bounds(radius: i16, q: i16) -> (i16, i16) {
    let low_from_s = -q - radius;
    let high_from_s = -q + radius;
    let low = if -radius > low_from_s {
        -radius
    } else {
        low_from_s
    };
    let high = if radius < high_from_s {
        radius
    } else {
        high_from_s
    };
    (low, high)
}

fn hex_distance(left: AxialCoord, right: AxialCoord) -> u16 {
    let q = i32::from(left.q) - i32::from(right.q);
    let r = i32::from(left.r) - i32::from(right.r);
    let s = -q - r;
    q.unsigned_abs().max(r.unsigned_abs()).max(s.unsigned_abs()) as u16
}

fn subtract_coords(coord: AxialCoord, center: AxialCoord) -> Result<AxialCoord> {
    Ok(AxialCoord::new(
        coord
            .q
            .checked_sub(center.q)
            .ok_or_else(|| R4Error::InvalidState("coordinate subtraction overflowed".to_owned()))?,
        coord
            .r
            .checked_sub(center.r)
            .ok_or_else(|| R4Error::InvalidState("coordinate subtraction overflowed".to_owned()))?,
    ))
}

fn add_coords(relative: AxialCoord, center: AxialCoord) -> Result<AxialCoord> {
    Ok(AxialCoord::new(
        relative
            .q
            .checked_add(center.q)
            .ok_or_else(|| R4Error::InvalidState("coordinate addition overflowed".to_owned()))?,
        relative
            .r
            .checked_add(center.r)
            .ok_or_else(|| R4Error::InvalidState("coordinate addition overflowed".to_owned()))?,
    ))
}

fn is_in_rules_grid(coord: AxialCoord) -> bool {
    let Ok(q) = i8::try_from(coord.q) else {
        return false;
    };
    let Ok(r) = i8::try_from(coord.r) else {
        return false;
    };
    HexCoord::new(q, r).to_index().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacities_indices_and_no_121_disk_are_exact() {
        assert_eq!(centered_hex_capacity(4), 61);
        assert_eq!(centered_hex_capacity(5), 91);
        assert_eq!(centered_hex_capacity(6), 127);
        assert!((0..=64).all(|radius| centered_hex_capacity(radius) != 121));
        for radius in NearFieldRadius::ALL {
            for index in 0..radius.capacity() {
                let coord = hex_disk_coord(radius.radius(), index as u16).unwrap();
                assert_eq!(hex_disk_index(radius.radius(), coord), Some(index as u16));
            }
        }
    }

    #[test]
    fn sector_bits_retain_symmetric_ties() {
        assert_eq!(direction_sector_bits(AxialCoord::ORIGIN), 0);
        assert_eq!(direction_sector_bits(AxialCoord::new(4, 0)), 0b000001);
        assert_eq!(direction_sector_bits(AxialCoord::new(2, -1)), 0b000011);
    }

    #[test]
    fn deterministic_center_uses_stable_f2_tie_break() {
        assert_eq!(deterministic_integer_center(&[]), AxialCoord::ORIGIN);
        assert_eq!(
            deterministic_integer_center(&[AxialCoord::new(10, 0), AxialCoord::new(11, 0)]),
            AxialCoord::new(10, 0)
        );
    }
}
