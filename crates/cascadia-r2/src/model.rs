use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    str::FromStr,
};

use cascadia_data::{
    BOARD_ENTITY_SIZE, BOARD_SLOTS, MARKET_ENTITY_SIZE, MAX_BOARD_TILES, PositionRecord, TARGET_DIM,
};
use cascadia_game::{
    Board, D6Transform, HexCoord, Rotation, Terrain, Tile, TileId, Wildlife, WildlifeMask,
};
use serde::{Deserialize, Serialize};

use crate::{R2Error, Result};

const NONE: u8 = u8::MAX;
const SPARSE_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AxialCoord {
    pub q: i16,
    pub r: i16,
}

impl AxialCoord {
    pub const ORIGIN: Self = Self { q: 0, r: 0 };
    pub const DIRECTIONS: [(i16, i16); 6] = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)];

    pub const fn new(q: i16, r: i16) -> Self {
        Self { q, r }
    }

    pub fn neighbor(self, edge: usize) -> Self {
        let (dq, dr) = Self::DIRECTIONS[edge % 6];
        Self::new(self.q + dq, self.r + dr)
    }

    pub fn neighbors(self) -> [Self; 6] {
        std::array::from_fn(|edge| self.neighbor(edge))
    }

    pub(crate) fn from_record_bytes(q: u8, r: u8) -> Self {
        Self::new(i16::from(q as i8), i16::from(r as i8))
    }

    pub(crate) fn to_record_bytes(self) -> Result<[u8; 2]> {
        let q = i8::try_from(self.q).map_err(|_| {
            R2Error::InvalidRecord(format!(
                "coordinate q={} cannot be represented by compact-entity-v2",
                self.q
            ))
        })?;
        let r = i8::try_from(self.r).map_err(|_| {
            R2Error::InvalidRecord(format!(
                "coordinate r={} cannot be represented by compact-entity-v2",
                self.r
            ))
        })?;
        Ok([q as u8, r as u8])
    }

    pub fn transformed(self, transform: D6Transform) -> Result<Self> {
        let [q, r] = self.to_record_bytes()?;
        let source = HexCoord::new(q as i8, r as i8);
        let transformed =
            transform
                .transform_coord(source)
                .map_err(|error| R2Error::D6Coordinate {
                    transform_id: transform.id(),
                    coord: self,
                    reason: error.to_string(),
                })?;
        Ok(Self::new(
            i16::from(transformed.q),
            i16::from(transformed.r),
        ))
    }

    fn is_in_rules_grid(self) -> bool {
        let Ok(q) = i8::try_from(self.q) else {
            return false;
        };
        let Ok(r) = i8::try_from(self.r) else {
            return false;
        };
        HexCoord::new(q, r).to_index().is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GlobalMetadata {
    pub game_index: u64,
    pub turn: u8,
    pub perspective_absolute_seat: u8,
    pub current_absolute_seat: u8,
    pub current_relative_seat: u8,
    pub player_count: u8,
    pub total_turns: u8,
    pub scoring_cards: [u8; 5],
    pub habitat_bonuses: bool,
    pub targets_omitted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlayerMetadata {
    pub relative_seat: u8,
    pub absolute_seat: u8,
    pub turns_taken: u8,
    pub turns_until_next_action: u8,
    pub occupied_count: u8,
    pub nature_tokens: u8,
    pub wildlife_counts: [u8; 5],
    pub largest_habitats: [u8; 5],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuppliedTile {
    pub terrain_a: Terrain,
    pub terrain_b: Option<Terrain>,
    pub wildlife_eligibility: WildlifeMask,
    pub keystone: bool,
}

impl SuppliedTile {
    pub fn validate(self) -> Result<()> {
        validate_tile_semantics(
            self.terrain_a,
            self.terrain_b,
            self.wildlife_eligibility,
            self.keystone,
        )
        .map_err(R2Error::InvalidRecord)
    }

    pub fn directed_edges(self, rotation: Rotation) -> [Terrain; 6] {
        let tile = self.as_game_tile();
        std::array::from_fn(|edge| tile.terrain_on_edge(rotation, edge))
    }

    pub(crate) fn canonical_rotations(self) -> &'static [Rotation] {
        if self.terrain_b.is_some() {
            &Rotation::ALL
        } else {
            &Rotation::ALL[..1]
        }
    }

    pub(crate) fn as_game_tile(self) -> Tile {
        Tile {
            id: TileId(0),
            terrain_a: self.terrain_a,
            terrain_b: self.terrain_b,
            wildlife: self.wildlife_eligibility,
            keystone: self.keystone,
        }
    }
}

impl FromStr for SuppliedTile {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        let fields = value.split(',').map(str::trim).collect::<Vec<_>>();
        if fields.len() != 4 {
            return Err(
                "supplied tile must be TERRAIN_A,TERRAIN_B_OR_NONE,WILDLIFE_MASK,KEYSTONE"
                    .to_owned(),
            );
        }
        let terrain_a = parse_terrain(fields[0])
            .ok_or_else(|| format!("invalid primary terrain {}", fields[0]))?;
        let terrain_b =
            if fields[1].eq_ignore_ascii_case("none") || fields[1] == "-" || fields[1] == "255" {
                None
            } else {
                Some(
                    parse_terrain(fields[1])
                        .ok_or_else(|| format!("invalid secondary terrain {}", fields[1]))?,
                )
            };
        let mask = if let Some(hex) = fields[2].strip_prefix("0x") {
            u8::from_str_radix(hex, 16)
        } else {
            fields[2].parse::<u8>()
        }
        .map_err(|_| format!("invalid wildlife mask {}", fields[2]))?;
        let keystone = match fields[3].to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" => true,
            "false" | "0" | "no" => false,
            _ => return Err(format!("invalid keystone flag {}", fields[3])),
        };
        let tile = Self {
            terrain_a,
            terrain_b,
            wildlife_eligibility: WildlifeMask::from_bits(mask),
            keystone,
        };
        tile.validate().map_err(|error| error.to_string())?;
        Ok(tile)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarketToken {
    pub slot: u8,
    pub tile: Option<SuppliedTile>,
    pub wildlife: Option<Wildlife>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OccupiedTileToken {
    pub relative_seat: u8,
    pub coord: AxialCoord,
    pub terrain_a: Terrain,
    pub terrain_b: Option<Terrain>,
    pub rotation: Rotation,
    pub directed_edge_terrains: [Terrain; 6],
    pub wildlife_eligibility: WildlifeMask,
    pub placed_wildlife: Option<Wildlife>,
    pub keystone: bool,
}

impl OccupiedTileToken {
    fn from_entity(relative_seat: u8, row: usize, entity: [u8; BOARD_ENTITY_SIZE]) -> Result<Self> {
        if entity.iter().all(|value| *value == NONE) {
            return Err(R2Error::InvalidOccupiedTile {
                seat: relative_seat,
                row,
                reason: "active row is the all-NONE padding sentinel".to_owned(),
            });
        }
        let terrain_a =
            terrain_from_code(entity[2]).ok_or_else(|| R2Error::InvalidOccupiedTile {
                seat: relative_seat,
                row,
                reason: "primary terrain code is outside [0, 4]".to_owned(),
            })?;
        let terrain_b =
            optional_terrain_from_code(entity[3]).ok_or_else(|| R2Error::InvalidOccupiedTile {
                seat: relative_seat,
                row,
                reason: "secondary terrain code is neither NONE nor [0, 4]".to_owned(),
            })?;
        let rotation = Rotation::new(entity[4]).ok_or_else(|| R2Error::InvalidOccupiedTile {
            seat: relative_seat,
            row,
            reason: "rotation code is outside [0, 5]".to_owned(),
        })?;
        let wildlife_eligibility = WildlifeMask::from_bits(entity[5]);
        if wildlife_eligibility.bits() != entity[5] {
            return Err(R2Error::InvalidOccupiedTile {
                seat: relative_seat,
                row,
                reason: "wildlife mask uses bits outside the five species".to_owned(),
            });
        }
        let placed_wildlife =
            optional_wildlife_from_code(entity[6]).ok_or_else(|| R2Error::InvalidOccupiedTile {
                seat: relative_seat,
                row,
                reason: "placed wildlife code is neither NONE nor [0, 4]".to_owned(),
            })?;
        let keystone = match entity[7] {
            0 => false,
            1 => true,
            _ => {
                return Err(R2Error::InvalidOccupiedTile {
                    seat: relative_seat,
                    row,
                    reason: "keystone code must be zero or one".to_owned(),
                });
            }
        };
        validate_tile_semantics(terrain_a, terrain_b, wildlife_eligibility, keystone).map_err(
            |reason| R2Error::InvalidOccupiedTile {
                seat: relative_seat,
                row,
                reason,
            },
        )?;
        if terrain_b.is_none() && rotation != Rotation::ZERO {
            return Err(R2Error::InvalidOccupiedTile {
                seat: relative_seat,
                row,
                reason: "single-terrain tile rotation is not canonical zero".to_owned(),
            });
        }
        if let Some(wildlife) = placed_wildlife
            && !wildlife_eligibility.contains(wildlife)
        {
            return Err(R2Error::InvalidOccupiedTile {
                seat: relative_seat,
                row,
                reason: "placed wildlife is absent from the eligibility mask".to_owned(),
            });
        }
        let coord = AxialCoord::from_record_bytes(entity[0], entity[1]);
        if !coord.is_in_rules_grid() {
            return Err(R2Error::InvalidOccupiedTile {
                seat: relative_seat,
                row,
                reason: format!(
                    "coordinate ({}, {}) is outside the rules backing grid",
                    coord.q, coord.r
                ),
            });
        }
        Ok(Self::new(
            relative_seat,
            coord,
            terrain_a,
            terrain_b,
            rotation,
            wildlife_eligibility,
            placed_wildlife,
            keystone,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        relative_seat: u8,
        coord: AxialCoord,
        terrain_a: Terrain,
        terrain_b: Option<Terrain>,
        rotation: Rotation,
        wildlife_eligibility: WildlifeMask,
        placed_wildlife: Option<Wildlife>,
        keystone: bool,
    ) -> Self {
        let tile = Tile {
            id: TileId(0),
            terrain_a,
            terrain_b,
            wildlife: wildlife_eligibility,
            keystone,
        };
        Self {
            relative_seat,
            coord,
            terrain_a,
            terrain_b,
            rotation,
            directed_edge_terrains: std::array::from_fn(|edge| {
                tile.terrain_on_edge(rotation, edge)
            }),
            wildlife_eligibility,
            placed_wildlife,
            keystone,
        }
    }

    fn transformed(&self, transform: D6Transform) -> Result<Self> {
        let tile = Tile {
            id: TileId(0),
            terrain_a: self.terrain_a,
            terrain_b: self.terrain_b,
            wildlife: self.wildlife_eligibility,
            keystone: self.keystone,
        };
        Ok(Self::new(
            self.relative_seat,
            self.coord.transformed(transform)?,
            self.terrain_a,
            self.terrain_b,
            transform.transform_tile_rotation(tile, self.rotation),
            self.wildlife_eligibility,
            self.placed_wildlife,
            self.keystone,
        ))
    }

    fn contains_terrain(&self, terrain: Terrain) -> bool {
        self.terrain_a == terrain || self.terrain_b == Some(terrain)
    }

    pub(crate) fn semantic_bytes(&self) -> [u8; 6] {
        [
            self.terrain_a as u8,
            self.terrain_b.map_or(NONE, |terrain| terrain as u8),
            self.rotation.get(),
            self.wildlife_eligibility.bits(),
            self.placed_wildlife.map_or(NONE, |wildlife| wildlife as u8),
            u8::from(self.keystone),
        ]
    }

    pub(crate) fn from_semantic_bytes(
        relative_seat: u8,
        coord: AxialCoord,
        semantic: [u8; 6],
    ) -> Result<Self> {
        let [q, r] = coord.to_record_bytes()?;
        Self::from_entity(
            relative_seat,
            0,
            [
                q,
                r,
                semantic[0],
                semantic[1],
                semantic[2],
                semantic[3],
                semantic[4],
                semantic[5],
            ],
        )
    }

    pub(crate) fn to_entity(&self) -> Result<[u8; BOARD_ENTITY_SIZE]> {
        let [q, r] = self.coord.to_record_bytes()?;
        let semantic = self.semantic_bytes();
        Ok([
            q,
            r,
            semantic[0],
            semantic[1],
            semantic[2],
            semantic[3],
            semantic[4],
            semantic[5],
        ])
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HabitatComponentToken {
    pub relative_seat: u8,
    pub component_id: u16,
    pub terrain: Terrain,
    pub members: Vec<AxialCoord>,
    pub member_count: u16,
    pub matching_internal_edge_count: u16,
    pub open_boundary_edge_count: u16,
    pub frontier_contact_count: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrontierHabitatTouch {
    pub terrain: Terrain,
    pub component_id: u16,
    pub component_size: u16,
    pub contact_edge_bits: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HabitatMerge {
    pub terrain: Terrain,
    pub touched_component_ids: Vec<u16>,
    pub resulting_size: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RotationCompatibility {
    pub rotation: u8,
    pub matching_edge_bits: u8,
    pub matching_edge_count: u8,
    pub all_present_edges_match: bool,
    pub habitat_merges: Vec<HabitatMerge>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuppliedTileCompatibility {
    /// Terrain-compatible means at least one directed habitat edge matches.
    /// Cascadia rules still permit every canonical rotation.
    pub terrain_compatible_rotations: Vec<u8>,
    pub best_matching_edge_count: u8,
    pub rotations: Vec<RotationCompatibility>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrontierToken {
    pub relative_seat: u8,
    pub coord: AxialCoord,
    pub neighbor_presence_bits: u8,
    pub neighbor_facing_terrains: [Option<Terrain>; 6],
    pub adjacent_wildlife_counts: [u8; 5],
    pub occupied_neighbor_runs: u8,
    pub opposite_neighbor_pair_bits: u8,
    pub touched_habitat_components: Vec<FrontierHabitatTouch>,
    pub resulting_size_by_terrain: [u16; 5],
    pub habitat_bridge_terrain_bits: u8,
    pub repeated_component_contact_terrain_bits: u8,
    pub supplied_tile_compatibility: Option<SuppliedTileCompatibility>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WildlifeMotifToken {
    pub relative_seat: u8,
    pub coord: AxialCoord,
    pub wildlife: Wildlife,
    pub neighbor_wildlife: [Option<Wildlife>; 6],
    pub adjacent_wildlife_counts: [u8; 5],
    pub same_species_neighbor_bits: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparsePublicState {
    pub schema_version: u16,
    pub global: GlobalMetadata,
    pub players: Vec<PlayerMetadata>,
    pub market: Vec<MarketToken>,
    pub supplied_tile: Option<SuppliedTile>,
    pub occupied_tiles: Vec<OccupiedTileToken>,
    pub legal_frontier: Vec<FrontierToken>,
    pub habitat_components: Vec<HabitatComponentToken>,
    pub wildlife_motifs: Vec<WildlifeMotifToken>,
}

/// Exact spatial state for one relative board.
///
/// This is the reusable production form of the R6 sibling-action
/// accumulator substrate.  A caller may apply and undo actions on one
/// canonical [`Board`] and materialize only the active board, while the
/// other three relative boards remain cached.  It deliberately contains no
/// market, supply, hidden-order, or terminal-target information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SparseBoardState {
    pub(crate) relative_seat: u8,
    pub(crate) occupied_tiles: Vec<OccupiedTileToken>,
    pub(crate) legal_frontier: Vec<FrontierToken>,
    pub(crate) habitat_components: Vec<HabitatComponentToken>,
    pub(crate) wildlife_motifs: Vec<WildlifeMotifToken>,
}

impl SparseBoardState {
    #[cfg(test)]
    pub(crate) fn from_sparse_public_state(state: &SparsePublicState, relative_seat: u8) -> Self {
        Self {
            relative_seat,
            occupied_tiles: state
                .occupied_tiles
                .iter()
                .filter(|token| token.relative_seat == relative_seat)
                .cloned()
                .collect(),
            legal_frontier: state
                .legal_frontier
                .iter()
                .filter(|token| token.relative_seat == relative_seat)
                .cloned()
                .collect(),
            habitat_components: state
                .habitat_components
                .iter()
                .filter(|token| token.relative_seat == relative_seat)
                .cloned()
                .collect(),
            wildlife_motifs: state
                .wildlife_motifs
                .iter()
                .filter(|token| token.relative_seat == relative_seat)
                .cloned()
                .collect(),
        }
    }

    pub(crate) fn from_board(relative_seat: u8, board: &Board) -> Result<Self> {
        if usize::from(relative_seat) >= cascadia_data::BOARD_SLOTS {
            return Err(R2Error::InvalidRecord(format!(
                "relative seat {relative_seat} exceeds the four-board schema"
            )));
        }
        // The canonical GameState legal enumerator owns place/undo and
        // preserves Board invariants. The exhaustive P1 oracle independently
        // compares every emitted row, so repeating the same graph-oracle
        // traversals for every sibling is correctness-neutral hot-path work.
        let occupied_tiles = occupied_tiles_from_board(relative_seat, board);
        let habitat_components = build_habitat_components(relative_seat, &occupied_tiles)?;
        let legal_frontier =
            build_frontier(relative_seat, &occupied_tiles, &habitat_components, None)?;
        let wildlife_motifs = build_wildlife_motifs(relative_seat, &occupied_tiles);
        Ok(Self {
            relative_seat,
            occupied_tiles,
            legal_frontier,
            habitat_components,
            wildlife_motifs,
        })
    }

    /// Reuse habitat components for siblings under one already-applied tile
    /// placement that differ only by their optional wildlife placement.
    pub(crate) fn from_wildlife_placement(
        tile_parent: &Self,
        wildlife_coord: HexCoord,
        wildlife: Wildlife,
    ) -> Result<Self> {
        let wildlife_coord =
            AxialCoord::new(i16::from(wildlife_coord.q), i16::from(wildlife_coord.r));
        let mut occupied_tiles = tile_parent.occupied_tiles.clone();
        let placed = occupied_tiles
            .iter_mut()
            .find(|tile| tile.coord == wildlife_coord)
            .ok_or_else(|| {
                R2Error::DatasetContract(
                    "wildlife sibling placement has no occupied tile".to_owned(),
                )
            })?;
        if placed.placed_wildlife.is_some() || !placed.wildlife_eligibility.contains(wildlife) {
            return Err(R2Error::DatasetContract(
                "wildlife sibling placement is not legal on its tile".to_owned(),
            ));
        }
        placed.placed_wildlife = Some(wildlife);
        let habitat_components = tile_parent.habitat_components.clone();
        let mut legal_frontier = tile_parent.legal_frontier.clone();
        for frontier in &mut legal_frontier {
            if frontier.coord.neighbors().contains(&wildlife_coord) {
                frontier.adjacent_wildlife_counts[wildlife as usize] = frontier
                    .adjacent_wildlife_counts[wildlife as usize]
                    .checked_add(1)
                    .ok_or_else(|| {
                        R2Error::DatasetContract(
                            "frontier adjacent wildlife count overflow".to_owned(),
                        )
                    })?;
            }
        }
        let wildlife_motifs = build_wildlife_motifs(tile_parent.relative_seat, &occupied_tiles);
        Ok(Self {
            relative_seat: tile_parent.relative_seat,
            occupied_tiles,
            legal_frontier,
            habitat_components,
            wildlife_motifs,
        })
    }

    pub(crate) fn wildlife_counts(&self) -> [u8; 5] {
        wildlife_counts(&self.occupied_tiles)
    }

    pub(crate) fn largest_habitats(&self) -> [u8; 5] {
        largest_habitats(&self.habitat_components)
    }
}

fn occupied_tiles_from_board(relative_seat: u8, board: &Board) -> Vec<OccupiedTileToken> {
    let mut occupied_tiles = board
        .placed_tiles()
        .map(|(coord, placed)| OccupiedTileToken {
            relative_seat,
            coord: AxialCoord::new(i16::from(coord.q), i16::from(coord.r)),
            terrain_a: placed.tile.terrain_a,
            terrain_b: placed.tile.terrain_b,
            rotation: placed.rotation,
            directed_edge_terrains: std::array::from_fn(|edge| {
                placed.tile.terrain_on_edge(placed.rotation, edge)
            }),
            wildlife_eligibility: placed.tile.wildlife,
            placed_wildlife: placed.wildlife,
            keystone: placed.tile.keystone,
        })
        .collect::<Vec<_>>();
    occupied_tiles.sort_unstable_by_key(|tile| (tile.coord.q, tile.coord.r));
    occupied_tiles
}

impl SparsePublicState {
    pub fn from_position_record(
        record: &PositionRecord,
        supplied_tile: Option<SuppliedTile>,
    ) -> Result<Self> {
        Self::from_position_record_with_market_mode(record, supplied_tile, false)
    }

    /// Decode a selected post-action public state before hidden refill.
    ///
    /// Board and turn-order validation remains exact. The only relaxation is
    /// that the consumed tile and wildlife market slots may be absent.
    pub fn from_selected_afterstate_record(
        record: &PositionRecord,
        supplied_tile: Option<SuppliedTile>,
    ) -> Result<Self> {
        Self::from_position_record_with_market_mode(record, supplied_tile, true)
    }

    fn from_position_record_with_market_mode(
        record: &PositionRecord,
        supplied_tile: Option<SuppliedTile>,
        allow_partial_market: bool,
    ) -> Result<Self> {
        validate_record_header(record)?;
        if let Some(tile) = supplied_tile {
            tile.validate()?;
        }
        let global = global_metadata(record);
        let mut players = Vec::with_capacity(usize::from(record.player_count));
        let mut occupied_tiles = Vec::new();

        for relative_seat in 0..record.player_count {
            let seat = usize::from(relative_seat);
            let count = usize::from(record.board_counts[seat]);
            if count > MAX_BOARD_TILES {
                return Err(R2Error::InvalidRecord(format!(
                    "relative seat {relative_seat} board count {count} exceeds {MAX_BOARD_TILES}"
                )));
            }
            let mut board_tokens = Vec::with_capacity(count);
            let mut previous = None;
            for row in 0..count {
                let token = OccupiedTileToken::from_entity(
                    relative_seat,
                    row,
                    record.board_entities[seat][row],
                )?;
                if previous.is_some_and(|coord| coord >= token.coord) {
                    return Err(R2Error::NonCanonicalOccupiedOrder {
                        seat: relative_seat,
                    });
                }
                previous = Some(token.coord);
                board_tokens.push(token);
            }
            for row in count..MAX_BOARD_TILES {
                if record.board_entities[seat][row] != [NONE; BOARD_ENTITY_SIZE] {
                    return Err(R2Error::InvalidOccupiedTile {
                        seat: relative_seat,
                        row,
                        reason: "padding row contains non-NONE data".to_owned(),
                    });
                }
            }
            ensure_unique_coordinates(relative_seat, &board_tokens)?;
            ensure_connected(relative_seat, &board_tokens)?;

            let absolute_seat = (record.active_seat + relative_seat) % record.player_count;
            let turns_taken = turns_taken(record.turn, record.player_count, absolute_seat);
            let expected_count = 3u8
                .checked_add(turns_taken)
                .ok_or_else(|| R2Error::InvalidRecord("board count overflow".to_owned()))?;
            if record.board_counts[seat] != expected_count {
                return Err(R2Error::InvalidRecord(format!(
                    "relative seat {relative_seat} has {} occupied tiles; turn order requires {expected_count}",
                    record.board_counts[seat]
                )));
            }

            let wildlife_counts = wildlife_counts(&board_tokens);
            if record.wildlife_counts[seat] != wildlife_counts {
                return Err(R2Error::InvalidRecord(format!(
                    "relative seat {relative_seat} wildlife counts disagree with occupied entities"
                )));
            }
            if wildlife_counts
                .iter()
                .map(|count| usize::from(*count))
                .sum::<usize>()
                > usize::from(turns_taken)
            {
                return Err(R2Error::InvalidRecord(format!(
                    "relative seat {relative_seat} has more placed wildlife than completed turns"
                )));
            }
            let components = build_habitat_components(relative_seat, &board_tokens)?;
            validate_habitat_oracle(relative_seat, &board_tokens, &components)?;
            let largest_habitats = largest_habitats(&components);
            if record.habitat_sizes[seat] != largest_habitats {
                return Err(R2Error::InvalidRecord(format!(
                    "relative seat {relative_seat} largest-habitat metadata disagrees with directed-edge components"
                )));
            }
            let earned_token_upper_bound = board_tokens
                .iter()
                .filter(|tile| tile.keystone && tile.placed_wildlife.is_some())
                .count();
            if usize::from(record.nature_tokens[seat]) > earned_token_upper_bound {
                return Err(R2Error::InvalidRecord(format!(
                    "relative seat {relative_seat} has {} Nature Tokens but only {earned_token_upper_bound} wildlife-bearing keystones",
                    record.nature_tokens[seat]
                )));
            }
            players.push(PlayerMetadata {
                relative_seat,
                absolute_seat,
                turns_taken,
                turns_until_next_action: turns_until_next_action(
                    record.turn,
                    record.player_count,
                    absolute_seat,
                ),
                occupied_count: record.board_counts[seat],
                nature_tokens: record.nature_tokens[seat],
                wildlife_counts,
                largest_habitats,
            });
            occupied_tiles.extend(board_tokens);
        }

        validate_inactive_slots(record)?;
        let market = parse_market(record, allow_partial_market)?;
        Self::assemble(global, players, market, supplied_tile, occupied_tiles)
    }

    pub(crate) fn assemble(
        global: GlobalMetadata,
        players: Vec<PlayerMetadata>,
        market: Vec<MarketToken>,
        supplied_tile: Option<SuppliedTile>,
        mut occupied_tiles: Vec<OccupiedTileToken>,
    ) -> Result<Self> {
        occupied_tiles
            .sort_unstable_by_key(|tile| (tile.relative_seat, tile.coord.q, tile.coord.r));
        validate_core_metadata(&global, &players, &market, supplied_tile, &occupied_tiles)?;

        let mut habitat_components = Vec::new();
        let mut legal_frontier = Vec::new();
        let mut wildlife_motifs = Vec::new();
        for relative_seat in 0..global.player_count {
            let board = occupied_tiles
                .iter()
                .filter(|tile| tile.relative_seat == relative_seat)
                .cloned()
                .collect::<Vec<_>>();
            ensure_unique_coordinates(relative_seat, &board)?;
            ensure_connected(relative_seat, &board)?;
            let components = build_habitat_components(relative_seat, &board)?;
            validate_habitat_oracle(relative_seat, &board, &components)?;
            let frontier = build_frontier(relative_seat, &board, &components, supplied_tile)?;
            validate_frontier_oracle(relative_seat, &board, &frontier)?;
            let motifs = build_wildlife_motifs(relative_seat, &board);
            validate_wildlife_motifs(relative_seat, &board, &motifs)?;
            habitat_components.extend(components);
            legal_frontier.extend(frontier);
            wildlife_motifs.extend(motifs);
        }

        Ok(Self {
            schema_version: SPARSE_SCHEMA_VERSION,
            global,
            players,
            market,
            supplied_tile,
            occupied_tiles,
            legal_frontier,
            habitat_components,
            wildlife_motifs,
        })
    }

    pub fn transformed(&self, transform: D6Transform) -> Result<Self> {
        let occupied_tiles = self
            .occupied_tiles
            .iter()
            .map(|tile| tile.transformed(transform))
            .collect::<Result<Vec<_>>>()?;
        let transformed = Self::assemble(
            self.global.clone(),
            self.players.clone(),
            self.market.clone(),
            self.supplied_tile,
            occupied_tiles,
        )?;

        for source in &self.occupied_tiles {
            let target_coord = source.coord.transformed(transform)?;
            let target = transformed
                .occupied_tiles
                .iter()
                .find(|tile| {
                    tile.relative_seat == source.relative_seat && tile.coord == target_coord
                })
                .ok_or_else(|| {
                    R2Error::InvalidRecord("D6 transform lost an occupied tile token".to_owned())
                })?;
            for edge in 0..6 {
                let target_edge = transform.transform_edge(edge).map_err(|error| {
                    R2Error::InvalidRecord(format!("D6 edge transform failed: {error}"))
                })?;
                if target.directed_edge_terrains[target_edge] != source.directed_edge_terrains[edge]
                {
                    return Err(R2Error::InvalidRecord(
                        "D6 transform changed a directed edge terrain".to_owned(),
                    ));
                }
            }
        }
        Ok(transformed)
    }

    pub fn reconstruct_position_record(
        &self,
        targets: [u16; TARGET_DIM],
    ) -> Result<PositionRecord> {
        validate_core_metadata(
            &self.global,
            &self.players,
            &self.market,
            self.supplied_tile,
            &self.occupied_tiles,
        )?;
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
            targets,
        };
        for player in &self.players {
            let seat = usize::from(player.relative_seat);
            record.board_counts[seat] = player.occupied_count;
            record.nature_tokens[seat] = player.nature_tokens;
            record.wildlife_counts[seat] = player.wildlife_counts;
            record.habitat_sizes[seat] = player.largest_habitats;
            for (row, tile) in self
                .occupied_tiles
                .iter()
                .filter(|tile| tile.relative_seat == player.relative_seat)
                .enumerate()
            {
                record.board_entities[seat][row] = tile.to_entity()?;
            }
        }
        for token in &self.market {
            record.market_entities[usize::from(token.slot)] = market_to_entity(token);
        }
        Ok(record)
    }

    pub fn total_spatial_tokens(&self) -> usize {
        self.occupied_tiles.len()
            + self.legal_frontier.len()
            + self.habitat_components.len()
            + self.wildlife_motifs.len()
    }

    pub fn board_token_counts(&self, relative_seat: u8) -> [usize; 5] {
        let occupied = self
            .occupied_tiles
            .iter()
            .filter(|token| token.relative_seat == relative_seat)
            .count();
        let frontier = self
            .legal_frontier
            .iter()
            .filter(|token| token.relative_seat == relative_seat)
            .count();
        let components = self
            .habitat_components
            .iter()
            .filter(|token| token.relative_seat == relative_seat)
            .count();
        let motifs = self
            .wildlife_motifs
            .iter()
            .filter(|token| token.relative_seat == relative_seat)
            .count();
        [
            occupied,
            frontier,
            components,
            motifs,
            occupied + frontier + components + motifs,
        ]
    }
}

fn validate_record_header(record: &PositionRecord) -> Result<()> {
    if !(1..=BOARD_SLOTS as u8).contains(&record.player_count) {
        return Err(R2Error::InvalidRecord(format!(
            "player count {} is outside [1, {BOARD_SLOTS}]",
            record.player_count
        )));
    }
    if record.active_seat >= record.player_count {
        return Err(R2Error::InvalidRecord(format!(
            "perspective seat {} does not exist in a {}-player game",
            record.active_seat, record.player_count
        )));
    }
    let expected_total = 20u16 * u16::from(record.player_count);
    if u16::from(record.total_turns) != expected_total {
        return Err(R2Error::InvalidRecord(format!(
            "total turns {} does not equal 20 times player count",
            record.total_turns
        )));
    }
    if record.turn > record.total_turns {
        return Err(R2Error::InvalidRecord(format!(
            "turn {} exceeds total turns {}",
            record.turn, record.total_turns
        )));
    }
    if record.scoring_cards.iter().any(|card| *card > 3) {
        return Err(R2Error::InvalidRecord(
            "scoring-card code is outside [0, 3]".to_owned(),
        ));
    }
    Ok(())
}

fn global_metadata(record: &PositionRecord) -> GlobalMetadata {
    let current_absolute_seat = record.turn % record.player_count;
    let current_relative_seat =
        (current_absolute_seat + record.player_count - record.active_seat) % record.player_count;
    GlobalMetadata {
        game_index: record.game_index,
        turn: record.turn,
        perspective_absolute_seat: record.active_seat,
        current_absolute_seat,
        current_relative_seat,
        player_count: record.player_count,
        total_turns: record.total_turns,
        scoring_cards: record.scoring_cards,
        habitat_bonuses: record.habitat_bonuses,
        targets_omitted: true,
    }
}

fn validate_inactive_slots(record: &PositionRecord) -> Result<()> {
    for seat in usize::from(record.player_count)..BOARD_SLOTS {
        if record.board_counts[seat] != 0
            || record.nature_tokens[seat] != 0
            || record.wildlife_counts[seat] != [0; 5]
            || record.habitat_sizes[seat] != [0; 5]
            || record.board_entities[seat] != [[NONE; BOARD_ENTITY_SIZE]; MAX_BOARD_TILES]
        {
            return Err(R2Error::InvalidRecord(format!(
                "inactive relative seat {seat} contains public board data"
            )));
        }
    }
    Ok(())
}

fn parse_market(record: &PositionRecord, allow_partial: bool) -> Result<Vec<MarketToken>> {
    let mut market = Vec::with_capacity(4);
    for (slot, entity) in record.market_entities.iter().copied().enumerate() {
        if entity == [NONE; MARKET_ENTITY_SIZE] {
            if record.turn < record.total_turns && !allow_partial {
                return Err(R2Error::InvalidRecord(format!(
                    "market slot {slot} is empty before game end"
                )));
            }
            market.push(MarketToken {
                slot: slot as u8,
                tile: None,
                wildlife: None,
            });
            continue;
        }
        if entity[5..] != [0, 0, 0] {
            return Err(R2Error::InvalidRecord(format!(
                "market slot {slot} reserved bytes are nonzero"
            )));
        }
        let tile = if entity[0] == NONE {
            if entity[1] != NONE || entity[2] != 0 || entity[4] != 0 {
                return Err(R2Error::InvalidRecord(format!(
                    "market slot {slot} has partial tile semantics"
                )));
            }
            None
        } else {
            let terrain_a = terrain_from_code(entity[0]).ok_or_else(|| {
                R2Error::InvalidRecord(format!("market slot {slot} primary terrain is invalid"))
            })?;
            let terrain_b = optional_terrain_from_code(entity[1]).ok_or_else(|| {
                R2Error::InvalidRecord(format!("market slot {slot} secondary terrain is invalid"))
            })?;
            let wildlife_eligibility = WildlifeMask::from_bits(entity[2]);
            if wildlife_eligibility.bits() != entity[2] {
                return Err(R2Error::InvalidRecord(format!(
                    "market slot {slot} wildlife mask is invalid"
                )));
            }
            let keystone = match entity[4] {
                0 => false,
                1 => true,
                _ => {
                    return Err(R2Error::InvalidRecord(format!(
                        "market slot {slot} keystone code is invalid"
                    )));
                }
            };
            let tile = SuppliedTile {
                terrain_a,
                terrain_b,
                wildlife_eligibility,
                keystone,
            };
            tile.validate()?;
            Some(tile)
        };
        let wildlife = optional_wildlife_from_code(entity[3]).ok_or_else(|| {
            R2Error::InvalidRecord(format!("market slot {slot} wildlife code is invalid"))
        })?;
        if record.turn < record.total_turns
            && !allow_partial
            && (tile.is_none() || wildlife.is_none())
        {
            return Err(R2Error::InvalidRecord(format!(
                "market slot {slot} is incomplete before game end"
            )));
        }
        market.push(MarketToken {
            slot: slot as u8,
            tile,
            wildlife,
        });
    }
    Ok(market)
}

fn market_to_entity(token: &MarketToken) -> [u8; MARKET_ENTITY_SIZE] {
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

fn validate_core_metadata(
    global: &GlobalMetadata,
    players: &[PlayerMetadata],
    market: &[MarketToken],
    supplied_tile: Option<SuppliedTile>,
    occupied_tiles: &[OccupiedTileToken],
) -> Result<()> {
    if global.schema_invariants_invalid() {
        return Err(R2Error::InvalidRecord(
            "global metadata invariants are invalid".to_owned(),
        ));
    }
    if players.len() != usize::from(global.player_count) {
        return Err(R2Error::InvalidRecord(
            "player metadata count does not equal player count".to_owned(),
        ));
    }
    if market.len() != 4
        || market
            .iter()
            .enumerate()
            .any(|(slot, token)| usize::from(token.slot) != slot)
    {
        return Err(R2Error::InvalidRecord(
            "market tokens are not the canonical four slots".to_owned(),
        ));
    }
    if let Some(tile) = supplied_tile {
        tile.validate()?;
    }
    for relative_seat in 0..global.player_count {
        let player = players
            .get(usize::from(relative_seat))
            .ok_or_else(|| R2Error::InvalidRecord("missing player metadata".to_owned()))?;
        if player.relative_seat != relative_seat {
            return Err(R2Error::InvalidRecord(
                "player metadata is not in relative-seat order".to_owned(),
            ));
        }
        let expected_absolute =
            (global.perspective_absolute_seat + relative_seat) % global.player_count;
        if player.absolute_seat != expected_absolute
            || player.turns_taken
                != turns_taken(global.turn, global.player_count, expected_absolute)
            || player.turns_until_next_action
                != turns_until_next_action(global.turn, global.player_count, expected_absolute)
        {
            return Err(R2Error::InvalidRecord(format!(
                "relative seat {relative_seat} turn-order metadata is inconsistent"
            )));
        }
        let board = occupied_tiles
            .iter()
            .filter(|tile| tile.relative_seat == relative_seat)
            .cloned()
            .collect::<Vec<_>>();
        if board.len() != usize::from(player.occupied_count)
            || wildlife_counts(&board) != player.wildlife_counts
        {
            return Err(R2Error::InvalidRecord(format!(
                "relative seat {relative_seat} player metadata disagrees with occupied tokens"
            )));
        }
        if player
            .wildlife_counts
            .iter()
            .map(|count| usize::from(*count))
            .sum::<usize>()
            > usize::from(player.turns_taken)
        {
            return Err(R2Error::InvalidRecord(format!(
                "relative seat {relative_seat} has more placed wildlife than completed turns"
            )));
        }
        let components = build_habitat_components(relative_seat, &board)?;
        if largest_habitats(&components) != player.largest_habitats {
            return Err(R2Error::InvalidRecord(format!(
                "relative seat {relative_seat} habitat metadata disagrees with occupied tokens"
            )));
        }
    }
    if occupied_tiles
        .iter()
        .any(|tile| tile.relative_seat >= global.player_count)
    {
        return Err(R2Error::InvalidRecord(
            "occupied token has an inactive relative seat".to_owned(),
        ));
    }
    if occupied_tiles.windows(2).any(|pair| {
        (pair[0].relative_seat, pair[0].coord.q, pair[0].coord.r)
            >= (pair[1].relative_seat, pair[1].coord.q, pair[1].coord.r)
    }) {
        return Err(R2Error::NonCanonicalPacked(
            "occupied tokens are not strictly ordered by seat and coordinate".to_owned(),
        ));
    }
    for token in market {
        if let Some(tile) = token.tile {
            tile.validate()?;
        }
    }
    Ok(())
}

impl GlobalMetadata {
    fn schema_invariants_invalid(&self) -> bool {
        self.player_count == 0
            || self.player_count > BOARD_SLOTS as u8
            || self.perspective_absolute_seat >= self.player_count
            || self.current_absolute_seat != self.turn % self.player_count
            || self.current_relative_seat
                != (self.current_absolute_seat + self.player_count - self.perspective_absolute_seat)
                    % self.player_count
            || u16::from(self.total_turns) != 20 * u16::from(self.player_count)
            || self.turn > self.total_turns
            || self.scoring_cards.iter().any(|card| *card > 3)
            || !self.targets_omitted
    }
}

fn turns_taken(turn: u8, player_count: u8, absolute_seat: u8) -> u8 {
    turn / player_count + u8::from(absolute_seat < turn % player_count)
}

fn turns_until_next_action(turn: u8, player_count: u8, absolute_seat: u8) -> u8 {
    let current = turn % player_count;
    (absolute_seat + player_count - current) % player_count
}

fn ensure_unique_coordinates(relative_seat: u8, board: &[OccupiedTileToken]) -> Result<()> {
    let mut coordinates = BTreeSet::new();
    for token in board {
        if !coordinates.insert(token.coord) {
            return Err(R2Error::DuplicateCoordinate {
                seat: relative_seat,
                coord: token.coord,
            });
        }
    }
    Ok(())
}

fn ensure_connected(relative_seat: u8, board: &[OccupiedTileToken]) -> Result<()> {
    if board.is_empty() {
        return Err(R2Error::DisconnectedBoard {
            seat: relative_seat,
        });
    }
    let occupied = board.iter().map(|tile| tile.coord).collect::<HashSet<_>>();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::from([board[0].coord]);
    while let Some(coord) = queue.pop_front() {
        if !visited.insert(coord) {
            continue;
        }
        for neighbor in coord.neighbors() {
            if occupied.contains(&neighbor) && !visited.contains(&neighbor) {
                queue.push_back(neighbor);
            }
        }
    }
    if visited.len() != board.len() {
        return Err(R2Error::DisconnectedBoard {
            seat: relative_seat,
        });
    }
    Ok(())
}

fn wildlife_counts(board: &[OccupiedTileToken]) -> [u8; 5] {
    let mut counts = [0u8; 5];
    for wildlife in board.iter().filter_map(|tile| tile.placed_wildlife) {
        counts[wildlife as usize] += 1;
    }
    counts
}

fn largest_habitats(components: &[HabitatComponentToken]) -> [u8; 5] {
    let mut largest = [0u8; 5];
    for component in components {
        largest[component.terrain as usize] =
            largest[component.terrain as usize].max(component.member_count as u8);
    }
    largest
}

#[derive(Debug)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            rank: vec![0; len],
        }
    }

    fn find(&mut self, value: usize) -> usize {
        if self.parent[value] != value {
            let root = self.find(self.parent[value]);
            self.parent[value] = root;
        }
        self.parent[value]
    }

    fn union(&mut self, left: usize, right: usize) {
        let mut left_root = self.find(left);
        let mut right_root = self.find(right);
        if left_root == right_root {
            return;
        }
        if self.rank[left_root] < self.rank[right_root] {
            std::mem::swap(&mut left_root, &mut right_root);
        }
        self.parent[right_root] = left_root;
        if self.rank[left_root] == self.rank[right_root] {
            self.rank[left_root] += 1;
        }
    }
}

fn build_habitat_components(
    relative_seat: u8,
    board: &[OccupiedTileToken],
) -> Result<Vec<HabitatComponentToken>> {
    let by_coord = board
        .iter()
        .enumerate()
        .map(|(index, tile)| (tile.coord, index))
        .collect::<BTreeMap<_, _>>();
    let mut drafts = Vec::<(Terrain, Vec<AxialCoord>)>::new();

    for terrain in Terrain::ALL {
        let mut union_find = UnionFind::new(board.len());
        for (index, tile) in board.iter().enumerate() {
            if !tile.contains_terrain(terrain) {
                continue;
            }
            for edge in 0..3 {
                if tile.directed_edge_terrains[edge] != terrain {
                    continue;
                }
                let neighbor_coord = tile.coord.neighbor(edge);
                let Some(&neighbor_index) = by_coord.get(&neighbor_coord) else {
                    continue;
                };
                let neighbor = &board[neighbor_index];
                if neighbor.directed_edge_terrains[(edge + 3) % 6] == terrain {
                    union_find.union(index, neighbor_index);
                }
            }
        }
        let mut groups = BTreeMap::<usize, Vec<AxialCoord>>::new();
        for (index, tile) in board.iter().enumerate() {
            if tile.contains_terrain(terrain) {
                let root = union_find.find(index);
                groups.entry(root).or_default().push(tile.coord);
            }
        }
        for mut members in groups.into_values() {
            members.sort_unstable();
            drafts.push((terrain, members));
        }
    }
    drafts.sort_unstable_by(|left, right| (left.0 as u8, &left.1).cmp(&(right.0 as u8, &right.1)));

    let mut components = Vec::with_capacity(drafts.len());
    for (component_id, (terrain, members)) in drafts.into_iter().enumerate() {
        let mut internal_edges = 0u16;
        let mut boundary_edges = 0u16;
        let mut frontier_contacts = BTreeSet::new();
        for coord in &members {
            let tile = &board[*by_coord.get(coord).expect("component member exists")];
            for edge in 0..6 {
                if tile.directed_edge_terrains[edge] != terrain {
                    continue;
                }
                let neighbor_coord = coord.neighbor(edge);
                match by_coord.get(&neighbor_coord).map(|index| &board[*index]) {
                    Some(neighbor)
                        if neighbor.directed_edge_terrains[(edge + 3) % 6] == terrain =>
                    {
                        if *coord < neighbor_coord {
                            internal_edges += 1;
                        }
                    }
                    Some(_) => boundary_edges += 1,
                    None => {
                        boundary_edges += 1;
                        frontier_contacts.insert(neighbor_coord);
                    }
                }
            }
        }
        components.push(HabitatComponentToken {
            relative_seat,
            component_id: component_id as u16,
            terrain,
            member_count: members.len() as u16,
            members,
            matching_internal_edge_count: internal_edges,
            open_boundary_edge_count: boundary_edges,
            frontier_contact_count: frontier_contacts.len() as u16,
        });
    }
    Ok(components)
}

fn validate_habitat_oracle(
    relative_seat: u8,
    board: &[OccupiedTileToken],
    components: &[HabitatComponentToken],
) -> Result<()> {
    let by_coord = board
        .iter()
        .map(|tile| (tile.coord, tile))
        .collect::<HashMap<_, _>>();
    let mut oracle = Vec::<(u8, Vec<AxialCoord>)>::new();
    for terrain in Terrain::ALL {
        let mut remaining = board
            .iter()
            .filter(|tile| tile.contains_terrain(terrain))
            .map(|tile| tile.coord)
            .collect::<BTreeSet<_>>();
        while let Some(start) = remaining.pop_first() {
            let mut members = Vec::new();
            let mut queue = VecDeque::from([start]);
            while let Some(coord) = queue.pop_front() {
                members.push(coord);
                let tile = by_coord[&coord];
                for edge in 0..6 {
                    if tile.directed_edge_terrains[edge] != terrain {
                        continue;
                    }
                    let neighbor_coord = coord.neighbor(edge);
                    let Some(neighbor) = by_coord.get(&neighbor_coord) else {
                        continue;
                    };
                    if neighbor.directed_edge_terrains[(edge + 3) % 6] == terrain
                        && remaining.remove(&neighbor_coord)
                    {
                        queue.push_back(neighbor_coord);
                    }
                }
            }
            members.sort_unstable();
            oracle.push((terrain as u8, members));
        }
    }
    oracle.sort_unstable();
    let production = components
        .iter()
        .map(|component| (component.terrain as u8, component.members.clone()))
        .collect::<Vec<_>>();
    if oracle != production {
        return Err(R2Error::HabitatOracleMismatch {
            seat: relative_seat,
        });
    }
    Ok(())
}

fn build_frontier(
    relative_seat: u8,
    board: &[OccupiedTileToken],
    components: &[HabitatComponentToken],
    supplied_tile: Option<SuppliedTile>,
) -> Result<Vec<FrontierToken>> {
    #[derive(Debug)]
    struct FrontierAccumulator {
        presence: u8,
        facing: [Option<Terrain>; 6],
        adjacent_wildlife_counts: [u8; 5],
    }

    let by_coord = board
        .iter()
        .map(|tile| (tile.coord, tile))
        .collect::<BTreeMap<_, _>>();
    let mut frontier = BTreeMap::<AxialCoord, FrontierAccumulator>::new();
    for tile in board {
        for edge in 0..6 {
            let frontier_coord = tile.coord.neighbor(edge);
            if !frontier_coord.is_in_rules_grid() || by_coord.contains_key(&frontier_coord) {
                continue;
            }
            let frontier_edge = (edge + 3) % 6;
            let entry = frontier
                .entry(frontier_coord)
                .or_insert(FrontierAccumulator {
                    presence: 0,
                    facing: [None; 6],
                    adjacent_wildlife_counts: [0; 5],
                });
            if entry.presence & (1 << frontier_edge) != 0 {
                return Err(R2Error::InvalidRecord(
                    "multiple occupied tiles map to one frontier direction".to_owned(),
                ));
            }
            entry.presence |= 1 << frontier_edge;
            entry.facing[frontier_edge] = Some(tile.directed_edge_terrains[edge]);
            if let Some(wildlife) = tile.placed_wildlife {
                entry.adjacent_wildlife_counts[wildlife as usize] += 1;
            }
        }
    }

    let component_lookup = components
        .iter()
        .flat_map(|component| {
            component.members.iter().map(move |coord| {
                (
                    (component.terrain as u8, *coord),
                    (component.component_id, component.member_count),
                )
            })
        })
        .collect::<HashMap<_, _>>();

    let mut tokens = Vec::with_capacity(frontier.len());
    for (coord, accumulator) in frontier {
        let mut touches = BTreeMap::<(u8, u16), FrontierHabitatTouch>::new();
        for edge in 0..6 {
            let Some(terrain) = accumulator.facing[edge] else {
                continue;
            };
            let neighbor_coord = coord.neighbor(edge);
            let (component_id, component_size) = component_lookup[&(terrain as u8, neighbor_coord)];
            let touch =
                touches
                    .entry((terrain as u8, component_id))
                    .or_insert(FrontierHabitatTouch {
                        terrain,
                        component_id,
                        component_size,
                        contact_edge_bits: 0,
                    });
            touch.contact_edge_bits |= 1 << edge;
        }
        let touched_habitat_components = touches.into_values().collect::<Vec<_>>();
        let mut resulting_size_by_terrain = [1u16; 5];
        let mut habitat_bridge_terrain_bits = 0u8;
        let mut repeated_component_contact_terrain_bits = 0u8;
        for terrain in Terrain::ALL {
            let terrain_touches = touched_habitat_components
                .iter()
                .filter(|touch| touch.terrain == terrain)
                .collect::<Vec<_>>();
            resulting_size_by_terrain[terrain as usize] += terrain_touches
                .iter()
                .map(|touch| touch.component_size)
                .sum::<u16>();
            if terrain_touches.len() >= 2 {
                habitat_bridge_terrain_bits |= 1 << terrain as u8;
            }
            if terrain_touches
                .iter()
                .any(|touch| touch.contact_edge_bits.count_ones() >= 2)
            {
                repeated_component_contact_terrain_bits |= 1 << terrain as u8;
            }
        }
        tokens.push(FrontierToken {
            relative_seat,
            coord,
            neighbor_presence_bits: accumulator.presence,
            neighbor_facing_terrains: accumulator.facing,
            adjacent_wildlife_counts: accumulator.adjacent_wildlife_counts,
            occupied_neighbor_runs: circular_runs(accumulator.presence),
            opposite_neighbor_pair_bits: opposite_pair_bits(accumulator.presence),
            supplied_tile_compatibility: supplied_tile.map(|tile| {
                supplied_tile_compatibility(
                    tile,
                    accumulator.presence,
                    accumulator.facing,
                    &touched_habitat_components,
                )
            }),
            touched_habitat_components,
            resulting_size_by_terrain,
            habitat_bridge_terrain_bits,
            repeated_component_contact_terrain_bits,
        });
    }
    Ok(tokens)
}

fn validate_frontier_oracle(
    relative_seat: u8,
    board: &[OccupiedTileToken],
    frontier: &[FrontierToken],
) -> Result<()> {
    let occupied = board.iter().map(|tile| tile.coord).collect::<HashSet<_>>();
    let oracle = board
        .iter()
        .flat_map(|tile| tile.coord.neighbors())
        .filter(|coord| coord.is_in_rules_grid() && !occupied.contains(coord))
        .collect::<HashSet<_>>();
    let production = frontier
        .iter()
        .map(|token| token.coord)
        .collect::<HashSet<_>>();
    if oracle != production || production.len() != frontier.len() {
        return Err(R2Error::FrontierOracleMismatch {
            seat: relative_seat,
        });
    }
    Ok(())
}

fn supplied_tile_compatibility(
    tile: SuppliedTile,
    neighbor_presence_bits: u8,
    neighbor_facing_terrains: [Option<Terrain>; 6],
    touches: &[FrontierHabitatTouch],
) -> SuppliedTileCompatibility {
    let mut rotations = Vec::new();
    let mut terrain_compatible_rotations = Vec::new();
    let mut best_matching_edge_count = 0u8;
    for rotation in tile.canonical_rotations() {
        let edges = tile.directed_edges(*rotation);
        let mut matching_edge_bits = 0u8;
        let mut component_ids_by_terrain = [const { Vec::<u16>::new() }; 5];
        for edge in 0..6 {
            if neighbor_facing_terrains[edge] == Some(edges[edge]) {
                matching_edge_bits |= 1 << edge;
                let terrain = edges[edge];
                for touch in touches.iter().filter(|touch| {
                    touch.terrain == terrain && touch.contact_edge_bits & (1 << edge) != 0
                }) {
                    if !component_ids_by_terrain[terrain as usize].contains(&touch.component_id) {
                        component_ids_by_terrain[terrain as usize].push(touch.component_id);
                    }
                }
            }
        }
        let matching_edge_count = matching_edge_bits.count_ones() as u8;
        best_matching_edge_count = best_matching_edge_count.max(matching_edge_count);
        if matching_edge_count > 0 {
            terrain_compatible_rotations.push(rotation.get());
        }
        let mut habitat_merges = Vec::new();
        for terrain in Terrain::ALL {
            if !tile.as_game_tile().contains_terrain(terrain) {
                continue;
            }
            let mut touched_component_ids = component_ids_by_terrain[terrain as usize].clone();
            touched_component_ids.sort_unstable();
            let resulting_size = 1 + touched_component_ids
                .iter()
                .map(|component_id| {
                    touches
                        .iter()
                        .find(|touch| touch.component_id == *component_id)
                        .expect("compatibility component comes from frontier touches")
                        .component_size
                })
                .sum::<u16>();
            habitat_merges.push(HabitatMerge {
                terrain,
                touched_component_ids,
                resulting_size,
            });
        }
        rotations.push(RotationCompatibility {
            rotation: rotation.get(),
            matching_edge_bits,
            matching_edge_count,
            all_present_edges_match: matching_edge_bits == neighbor_presence_bits,
            habitat_merges,
        });
    }
    SuppliedTileCompatibility {
        terrain_compatible_rotations,
        best_matching_edge_count,
        rotations,
    }
}

fn circular_runs(bits: u8) -> u8 {
    let bits = bits & 0b11_1111;
    if bits == 0 {
        return 0;
    }
    if bits == 0b11_1111 {
        return 1;
    }
    (0..6)
        .filter(|edge| bits & (1 << edge) != 0 && bits & (1 << ((edge + 5) % 6)) == 0)
        .count() as u8
}

fn opposite_pair_bits(bits: u8) -> u8 {
    let mut pairs = 0;
    for pair in 0..3 {
        if bits & (1 << pair) != 0 && bits & (1 << (pair + 3)) != 0 {
            pairs |= 1 << pair;
        }
    }
    pairs
}

fn build_wildlife_motifs(
    relative_seat: u8,
    board: &[OccupiedTileToken],
) -> Vec<WildlifeMotifToken> {
    let by_coord = board
        .iter()
        .map(|tile| (tile.coord, tile))
        .collect::<HashMap<_, _>>();
    board
        .iter()
        .filter_map(|tile| {
            let wildlife = tile.placed_wildlife?;
            let neighbor_wildlife = std::array::from_fn(|edge| {
                by_coord
                    .get(&tile.coord.neighbor(edge))
                    .and_then(|neighbor| neighbor.placed_wildlife)
            });
            let mut adjacent_wildlife_counts = [0u8; 5];
            let mut same_species_neighbor_bits = 0u8;
            for (edge, neighbor) in neighbor_wildlife.iter().copied().enumerate() {
                if let Some(neighbor) = neighbor {
                    adjacent_wildlife_counts[neighbor as usize] += 1;
                    if neighbor == wildlife {
                        same_species_neighbor_bits |= 1 << edge;
                    }
                }
            }
            Some(WildlifeMotifToken {
                relative_seat,
                coord: tile.coord,
                wildlife,
                neighbor_wildlife,
                adjacent_wildlife_counts,
                same_species_neighbor_bits,
            })
        })
        .collect()
}

fn validate_wildlife_motifs(
    relative_seat: u8,
    board: &[OccupiedTileToken],
    motifs: &[WildlifeMotifToken],
) -> Result<()> {
    let source = board
        .iter()
        .filter_map(|tile| {
            tile.placed_wildlife
                .map(|wildlife| (tile.coord, wildlife as u8))
        })
        .collect::<Vec<_>>();
    let projected = motifs
        .iter()
        .map(|motif| (motif.coord, motif.wildlife as u8))
        .collect::<Vec<_>>();
    if source != projected {
        return Err(R2Error::WildlifeMotifMismatch {
            seat: relative_seat,
        });
    }
    Ok(())
}

fn validate_tile_semantics(
    terrain_a: Terrain,
    terrain_b: Option<Terrain>,
    wildlife_eligibility: WildlifeMask,
    keystone: bool,
) -> std::result::Result<(), String> {
    let mask = wildlife_eligibility.bits();
    if mask == 0 || mask & !0b1_1111 != 0 {
        return Err("wildlife eligibility must use at least one of the five species".to_owned());
    }
    match terrain_b {
        Some(terrain_b) => {
            if terrain_a == terrain_b {
                return Err("dual-terrain tile repeats the same terrain".to_owned());
            }
            if keystone {
                return Err("dual-terrain tile cannot be a keystone".to_owned());
            }
        }
        None => {
            if !keystone {
                return Err("single-terrain tile must be a keystone".to_owned());
            }
            if mask.count_ones() != 1 {
                return Err("keystone tile must allow exactly one wildlife species".to_owned());
            }
        }
    }
    Ok(())
}

fn parse_terrain(value: &str) -> Option<Terrain> {
    match value.to_ascii_lowercase().as_str() {
        "0" | "mountain" => Some(Terrain::Mountain),
        "1" | "forest" => Some(Terrain::Forest),
        "2" | "prairie" => Some(Terrain::Prairie),
        "3" | "wetland" => Some(Terrain::Wetland),
        "4" | "river" => Some(Terrain::River),
        _ => None,
    }
}

pub(crate) const fn terrain_from_code(code: u8) -> Option<Terrain> {
    match code {
        0 => Some(Terrain::Mountain),
        1 => Some(Terrain::Forest),
        2 => Some(Terrain::Prairie),
        3 => Some(Terrain::Wetland),
        4 => Some(Terrain::River),
        _ => None,
    }
}

pub(crate) const fn optional_terrain_from_code(code: u8) -> Option<Option<Terrain>> {
    if code == NONE {
        Some(None)
    } else {
        match terrain_from_code(code) {
            Some(terrain) => Some(Some(terrain)),
            None => None,
        }
    }
}

pub(crate) const fn optional_wildlife_from_code(code: u8) -> Option<Option<Wildlife>> {
    match code {
        NONE => Some(None),
        0 => Some(Some(Wildlife::Bear)),
        1 => Some(Some(Wildlife::Elk)),
        2 => Some(Some(Wildlife::Salmon)),
        3 => Some(Some(Wildlife::Hawk)),
        4 => Some(Some(Wildlife::Fox)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cascadia_game::{GameConfig, GameSeed, GameState};

    #[test]
    fn circular_neighbor_runs_are_exact() {
        assert_eq!(circular_runs(0), 0);
        assert_eq!(circular_runs(0b00_0001), 1);
        assert_eq!(circular_runs(0b00_0011), 1);
        assert_eq!(circular_runs(0b10_0001), 1);
        assert_eq!(circular_runs(0b00_0101), 2);
        assert_eq!(circular_runs(0b11_1111), 1);
    }

    #[test]
    fn supplied_tile_parser_is_strict() {
        let tile: SuppliedTile = "forest,river,0x1f,false".parse().unwrap();
        assert_eq!(tile.terrain_a, Terrain::Forest);
        assert_eq!(tile.terrain_b, Some(Terrain::River));
        assert!("forest,none,3,true".parse::<SuppliedTile>().is_err());
        assert!("forest,forest,1,false".parse::<SuppliedTile>().is_err());
    }

    #[test]
    fn public_state_rejects_wildlife_on_a_zero_turn_starter_tile() {
        let game = GameState::new(
            GameConfig::research_aaaaa(4).unwrap(),
            GameSeed::from_u64(0x5232_4d41_505f_4d4f),
        )
        .unwrap();
        let mut record = PositionRecord::observe(&game, 0);
        let eligibility = record.board_entities[0][0][5];
        let wildlife = eligibility.trailing_zeros() as usize;
        assert!(wildlife < 5);
        record.board_entities[0][0][6] = wildlife as u8;
        record.wildlife_counts[0][wildlife] = 1;
        let error = SparsePublicState::from_position_record(&record, None).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("more placed wildlife than completed turns")
        );
    }
}
