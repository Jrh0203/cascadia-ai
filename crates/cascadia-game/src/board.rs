use arrayvec::ArrayVec;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    D6Error, D6Transform, GRID_SIZE, HexCoord, Rotation, StarterPlacement, Terrain, Tile, Wildlife,
};

pub const MAX_BOARD_TILES: usize = 23;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlacedTile {
    pub tile: Tile,
    pub rotation: Rotation,
    pub wildlife: Option<Wildlife>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoardDelta {
    TilePlaced {
        index: usize,
    },
    WildlifePlaced {
        index: usize,
        awarded_nature_token: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BoardError {
    #[error("hex coordinate {0:?} is outside the supported board")]
    OutOfBounds(HexCoord),
    #[error("hex coordinate {0:?} is already occupied")]
    Occupied(HexCoord),
    #[error("tile at {0:?} would not touch the existing environment")]
    Detached(HexCoord),
    #[error("hex coordinate {0:?} does not contain a tile")]
    MissingTile(HexCoord),
    #[error("a Cascadia environment cannot contain more than {MAX_BOARD_TILES} tiles")]
    TileLimitReached,
    #[error("tile at {0:?} already contains wildlife")]
    WildlifeOccupied(HexCoord),
    #[error("tile at {coord:?} cannot support {wildlife:?}")]
    UnsupportedWildlife { coord: HexCoord, wildlife: Wildlife },
    #[error("undo operations must be applied in reverse order")]
    InvalidUndoOrder,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Board {
    cells: Vec<Option<PlacedTile>>,
    placed_indices: Vec<u16>,
    nature_tokens: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HabitatAnalysis {
    component_ids: [[u8; GRID_SIZE]; 5],
    component_sizes: [[u8; 24]; 5],
    largest: [u8; 5],
    matching_edges: u16,
}

impl HabitatAnalysis {
    pub fn largest(&self, terrain: Terrain) -> u8 {
        self.largest[terrain as usize]
    }

    pub fn matching_edges(&self) -> u16 {
        self.matching_edges
    }

    pub fn largest_after_tile(
        &self,
        board: &Board,
        coord: HexCoord,
        tile: Tile,
        rotation: Rotation,
        terrain: Terrain,
    ) -> u8 {
        if !tile.contains_terrain(terrain) {
            return self.largest(terrain);
        }

        let terrain_index = terrain as usize;
        let mut connected_components = [0u8; 6];
        let mut connected_count = 0usize;
        let mut component_size = 1u8;
        for edge in 0..6 {
            if tile.terrain_on_edge(rotation, edge) != terrain {
                continue;
            }
            let neighbor = coord.neighbor(edge);
            let Some(neighbor_index) = neighbor.to_index() else {
                continue;
            };
            let Some(neighbor_tile) = board.cells[neighbor_index] else {
                continue;
            };
            if neighbor_tile
                .tile
                .terrain_on_edge(neighbor_tile.rotation, (edge + 3) % 6)
                != terrain
            {
                continue;
            }
            let component = self.component_ids[terrain_index][neighbor_index];
            if component == 0 || connected_components[..connected_count].contains(&component) {
                continue;
            }
            connected_components[connected_count] = component;
            connected_count += 1;
            component_size += self.component_sizes[terrain_index][usize::from(component)];
        }
        self.largest(terrain).max(component_size)
    }

    pub fn matching_edges_after_tile(
        &self,
        board: &Board,
        coord: HexCoord,
        tile: Tile,
        rotation: Rotation,
    ) -> u16 {
        self.matching_edges
            + (0..6)
                .filter(|edge| {
                    board
                        .tile_at(coord.neighbor(*edge))
                        .is_some_and(|neighbor| {
                            tile.terrain_on_edge(rotation, *edge)
                                == neighbor
                                    .tile
                                    .terrain_on_edge(neighbor.rotation, (*edge + 3) % 6)
                        })
                })
                .count() as u16
    }

    /// Evaluate one terrain's exact component growth and all matching edges
    /// in a single neighbor pass. Opportunity extraction needs both values for
    /// every candidate archetype rotation.
    pub fn largest_and_matching_edges_after_tile(
        &self,
        board: &Board,
        coord: HexCoord,
        tile: Tile,
        rotation: Rotation,
        terrain: Terrain,
    ) -> (u8, u8) {
        let terrain_index = terrain as usize;
        let mut connected_components = [0u8; 6];
        let mut connected_count = 0usize;
        let mut component_size = 1u8;
        let mut matching_edges = 0u8;
        for edge in 0..6 {
            let neighbor = coord.neighbor(edge);
            let Some(neighbor_index) = neighbor.to_index() else {
                continue;
            };
            let Some(neighbor_tile) = board.cells[neighbor_index] else {
                continue;
            };
            let tile_terrain = tile.terrain_on_edge(rotation, edge);
            let neighbor_terrain = neighbor_tile
                .tile
                .terrain_on_edge(neighbor_tile.rotation, (edge + 3) % 6);
            if tile_terrain == neighbor_terrain {
                matching_edges += 1;
            }
            if tile_terrain != terrain || neighbor_terrain != terrain {
                continue;
            }
            let component = self.component_ids[terrain_index][neighbor_index];
            if component == 0 || connected_components[..connected_count].contains(&component) {
                continue;
            }
            connected_components[connected_count] = component;
            connected_count += 1;
            component_size += self.component_sizes[terrain_index][usize::from(component)];
        }
        (self.largest(terrain).max(component_size), matching_edges)
    }

    /// Evaluate all terrain component sizes plus matching edges in one pass.
    /// Only the one or two terrains present on the tile can grow.
    pub fn largest_all_and_matching_edges_after_tile(
        &self,
        board: &Board,
        coord: HexCoord,
        tile: Tile,
        rotation: Rotation,
    ) -> ([u8; 5], u8) {
        let mut connected_components = [[0u8; 6]; 5];
        let mut connected_counts = [0usize; 5];
        let mut component_sizes = [1u8; 5];
        let mut matching_edges = 0u8;
        for edge in 0..6 {
            let neighbor = coord.neighbor(edge);
            let Some(neighbor_index) = neighbor.to_index() else {
                continue;
            };
            let Some(neighbor_tile) = board.cells[neighbor_index] else {
                continue;
            };
            let terrain = tile.terrain_on_edge(rotation, edge);
            if neighbor_tile
                .tile
                .terrain_on_edge(neighbor_tile.rotation, (edge + 3) % 6)
                != terrain
            {
                continue;
            }
            matching_edges += 1;
            let terrain_index = terrain as usize;
            let component = self.component_ids[terrain_index][neighbor_index];
            let count = connected_counts[terrain_index];
            if component == 0 || connected_components[terrain_index][..count].contains(&component) {
                continue;
            }
            connected_components[terrain_index][count] = component;
            connected_counts[terrain_index] += 1;
            component_sizes[terrain_index] +=
                self.component_sizes[terrain_index][usize::from(component)];
        }
        let largest = std::array::from_fn(|terrain_index| {
            let terrain = Terrain::ALL[terrain_index];
            if tile.contains_terrain(terrain) {
                self.largest[terrain_index].max(component_sizes[terrain_index])
            } else {
                self.largest[terrain_index]
            }
        });
        (largest, matching_edges)
    }
}

impl Board {
    pub fn empty() -> Self {
        Self {
            cells: vec![None; GRID_SIZE],
            placed_indices: Vec::with_capacity(23),
            nature_tokens: 0,
        }
    }

    pub fn from_starter(starter: &[StarterPlacement; 3]) -> Self {
        let mut board = Self::empty();
        for placement in starter {
            board
                .insert_starter(placement.coord, placement.tile, placement.rotation)
                .expect("starter clusters are valid");
        }
        board
    }

    pub fn nature_tokens(&self) -> u8 {
        self.nature_tokens
    }

    /// Stable digest of the complete mutable board state used by incremental
    /// apply/undo audits.
    pub fn canonical_hash(&self) -> blake3::Hash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"cascadia-board-canonical-v2");
        hasher.update(&[self.nature_tokens]);
        hasher.update(&(self.placed_indices.len() as u16).to_le_bytes());
        for &index in &self.placed_indices {
            hasher.update(&index.to_le_bytes());
        }

        // Hash every storage slot, including an explicit marker for empty
        // cells. The placed-index sequence is part of the mutable undo state,
        // but it is not a substitute for proving that no unindexed cell was
        // changed.
        let mut cells = [0u8; GRID_SIZE * 8];
        for (index, cell) in self.cells.iter().enumerate() {
            let offset = index * 8;
            let Some(placed) = cell else {
                continue;
            };
            cells[offset] = 1;
            cells[offset + 1] = placed.tile.id.0;
            cells[offset + 2] = placed.tile.terrain_a as u8;
            cells[offset + 3] = placed
                .tile
                .terrain_b
                .map_or(u8::MAX, |terrain| terrain as u8);
            cells[offset + 4] = placed.tile.wildlife.bits();
            cells[offset + 5] = u8::from(placed.tile.keystone);
            cells[offset + 6] = placed.rotation.get();
            cells[offset + 7] = placed.wildlife.map_or(u8::MAX, |wildlife| wildlife as u8);
        }
        hasher.update(&cells);
        hasher.finalize()
    }

    pub fn tile_count(&self) -> usize {
        self.placed_indices.len()
    }

    pub fn placed_tiles(&self) -> impl Iterator<Item = (HexCoord, &PlacedTile)> {
        self.placed_indices.iter().map(|index| {
            let index = usize::from(*index);
            let coord = HexCoord::from_index(index).expect("stored board index is valid");
            let tile = self.cells[index]
                .as_ref()
                .expect("stored board index is occupied");
            (coord, tile)
        })
    }

    pub fn tile_at(&self, coord: HexCoord) -> Option<&PlacedTile> {
        coord
            .to_index()
            .and_then(|index| self.cells[index].as_ref())
    }

    pub fn frontier(&self) -> Vec<HexCoord> {
        let mut frontier = Vec::with_capacity(self.placed_indices.len() * 3);
        for &index in &self.placed_indices {
            let coord = HexCoord::from_index(usize::from(index)).expect("valid board index");
            for neighbor in coord.neighbors() {
                let Some(neighbor_index) = neighbor.to_index() else {
                    continue;
                };
                if self.cells[neighbor_index].is_none() {
                    frontier.push(neighbor);
                }
            }
        }
        frontier.sort_unstable();
        frontier.dedup();
        frontier
    }

    pub fn wildlife_placements(&self, wildlife: Wildlife) -> ArrayVec<HexCoord, MAX_BOARD_TILES> {
        self.placed_tiles()
            .filter_map(|(coord, placed)| {
                (placed.wildlife.is_none() && placed.tile.wildlife.contains(wildlife))
                    .then_some(coord)
            })
            .collect()
    }

    pub fn wildlife_at(&self, coord: HexCoord) -> Option<Wildlife> {
        self.tile_at(coord).and_then(|placed| placed.wildlife)
    }

    pub fn wildlife_positions(&self, wildlife: Wildlife) -> ArrayVec<HexCoord, MAX_BOARD_TILES> {
        self.placed_tiles()
            .filter_map(|(coord, placed)| (placed.wildlife == Some(wildlife)).then_some(coord))
            .collect()
    }

    pub fn place_tile(
        &mut self,
        coord: HexCoord,
        tile: Tile,
        rotation: Rotation,
    ) -> Result<BoardDelta, BoardError> {
        let index = coord.to_index().ok_or(BoardError::OutOfBounds(coord))?;
        if self.cells[index].is_some() {
            return Err(BoardError::Occupied(coord));
        }
        if self.placed_indices.len() == MAX_BOARD_TILES {
            return Err(BoardError::TileLimitReached);
        }
        if !self.placed_indices.is_empty()
            && !coord.neighbors().into_iter().any(|neighbor| {
                neighbor
                    .to_index()
                    .is_some_and(|neighbor_index| self.cells[neighbor_index].is_some())
            })
        {
            return Err(BoardError::Detached(coord));
        }

        self.cells[index] = Some(PlacedTile {
            tile,
            rotation: tile.canonical_rotation(rotation),
            wildlife: None,
        });
        self.placed_indices.push(index as u16);
        Ok(BoardDelta::TilePlaced { index })
    }

    pub fn place_wildlife(
        &mut self,
        coord: HexCoord,
        wildlife: Wildlife,
    ) -> Result<BoardDelta, BoardError> {
        let index = coord.to_index().ok_or(BoardError::OutOfBounds(coord))?;
        let placed = self.cells[index]
            .as_mut()
            .ok_or(BoardError::MissingTile(coord))?;
        if placed.wildlife.is_some() {
            return Err(BoardError::WildlifeOccupied(coord));
        }
        if !placed.tile.wildlife.contains(wildlife) {
            return Err(BoardError::UnsupportedWildlife { coord, wildlife });
        }

        placed.wildlife = Some(wildlife);
        let awarded_nature_token = placed.tile.keystone;
        if awarded_nature_token {
            self.nature_tokens = self
                .nature_tokens
                .checked_add(1)
                .expect("a board cannot exhaust u8 nature tokens");
        }
        Ok(BoardDelta::WildlifePlaced {
            index,
            awarded_nature_token,
        })
    }

    pub fn undo(&mut self, delta: BoardDelta) -> Result<(), BoardError> {
        match delta {
            BoardDelta::TilePlaced { index } => {
                if self.placed_indices.last().copied() != Some(index as u16)
                    || self.cells[index].is_none()
                {
                    return Err(BoardError::InvalidUndoOrder);
                }
                self.cells[index] = None;
                self.placed_indices.pop();
            }
            BoardDelta::WildlifePlaced {
                index,
                awarded_nature_token,
            } => {
                let Some(placed) = self.cells[index].as_mut() else {
                    return Err(BoardError::InvalidUndoOrder);
                };
                if placed.wildlife.take().is_none() {
                    return Err(BoardError::InvalidUndoOrder);
                }
                if awarded_nature_token {
                    self.nature_tokens = self
                        .nature_tokens
                        .checked_sub(1)
                        .ok_or(BoardError::InvalidUndoOrder)?;
                }
            }
        }
        Ok(())
    }

    pub fn spend_nature_token(&mut self) -> bool {
        if self.nature_tokens == 0 {
            false
        } else {
            self.nature_tokens -= 1;
            true
        }
    }

    pub(crate) fn refund_nature_token(&mut self) {
        self.nature_tokens = self
            .nature_tokens
            .checked_add(1)
            .expect("a board cannot exhaust u8 nature tokens");
    }

    #[cfg(test)]
    pub(crate) fn grant_nature_tokens(&mut self, count: u8) {
        self.nature_tokens = self
            .nature_tokens
            .checked_add(count)
            .expect("a board cannot exhaust u8 nature tokens");
    }

    pub fn largest_habitat(&self, terrain: Terrain) -> u8 {
        let mut seen = [false; GRID_SIZE];
        let mut largest = 0usize;

        for &start in &self.placed_indices {
            let start = usize::from(start);
            let Some(start_tile) = self.cells[start] else {
                continue;
            };
            if seen[start] || !start_tile.tile.contains_terrain(terrain) {
                continue;
            }

            let mut stack = [0usize; 23];
            stack[0] = start;
            let mut stack_len = 1;
            seen[start] = true;
            let mut size = 0usize;
            while stack_len > 0 {
                stack_len -= 1;
                let index = stack[stack_len];
                size += 1;
                let coord = HexCoord::from_index(index).expect("valid board index");
                let placed = self.cells[index].expect("component cells are occupied");
                for edge in 0..6 {
                    if placed.tile.terrain_on_edge(placed.rotation, edge) != terrain {
                        continue;
                    }
                    let neighbor = coord.neighbor(edge);
                    let Some(neighbor_index) = neighbor.to_index() else {
                        continue;
                    };
                    let Some(neighbor_tile) = self.cells[neighbor_index] else {
                        continue;
                    };
                    if !seen[neighbor_index]
                        && neighbor_tile
                            .tile
                            .terrain_on_edge(neighbor_tile.rotation, (edge + 3) % 6)
                            == terrain
                    {
                        seen[neighbor_index] = true;
                        stack[stack_len] = neighbor_index;
                        stack_len += 1;
                    }
                }
            }
            largest = largest.max(size);
        }

        largest as u8
    }

    pub fn habitat_analysis(&self) -> HabitatAnalysis {
        let mut analysis = HabitatAnalysis {
            component_ids: [[0; GRID_SIZE]; 5],
            component_sizes: [[0; 24]; 5],
            largest: [0; 5],
            matching_edges: 0,
        };

        for terrain in Terrain::ALL {
            let terrain_index = terrain as usize;
            let mut next_component = 1u8;
            for &start in &self.placed_indices {
                let start = usize::from(start);
                let Some(start_tile) = self.cells[start] else {
                    continue;
                };
                if analysis.component_ids[terrain_index][start] != 0
                    || !start_tile.tile.contains_terrain(terrain)
                {
                    continue;
                }

                let component = next_component;
                next_component += 1;
                let mut stack = [0usize; 23];
                stack[0] = start;
                let mut stack_len = 1;
                analysis.component_ids[terrain_index][start] = component;
                let mut size = 0u8;
                while stack_len > 0 {
                    stack_len -= 1;
                    let index = stack[stack_len];
                    size += 1;
                    let coord = HexCoord::from_index(index).expect("valid board index");
                    let placed = self.cells[index].expect("component cells are occupied");
                    for edge in 0..6 {
                        if placed.tile.terrain_on_edge(placed.rotation, edge) != terrain {
                            continue;
                        }
                        let neighbor = coord.neighbor(edge);
                        let Some(neighbor_index) = neighbor.to_index() else {
                            continue;
                        };
                        let Some(neighbor_tile) = self.cells[neighbor_index] else {
                            continue;
                        };
                        if analysis.component_ids[terrain_index][neighbor_index] == 0
                            && neighbor_tile
                                .tile
                                .terrain_on_edge(neighbor_tile.rotation, (edge + 3) % 6)
                                == terrain
                        {
                            analysis.component_ids[terrain_index][neighbor_index] = component;
                            stack[stack_len] = neighbor_index;
                            stack_len += 1;
                        }
                    }
                }
                analysis.component_sizes[terrain_index][usize::from(component)] = size;
                analysis.largest[terrain_index] = analysis.largest[terrain_index].max(size);
            }
        }

        let doubled_matches = self
            .placed_tiles()
            .map(|(coord, placed)| {
                (0..6)
                    .filter(|edge| {
                        self.tile_at(coord.neighbor(*edge)).is_some_and(|neighbor| {
                            placed.tile.terrain_on_edge(placed.rotation, *edge)
                                == neighbor
                                    .tile
                                    .terrain_on_edge(neighbor.rotation, (*edge + 3) % 6)
                        })
                    })
                    .count() as u16
            })
            .sum::<u16>();
        analysis.matching_edges = doubled_matches / 2;
        analysis
    }

    /// Returns an exact transformed copy or an error if the finite backing grid
    /// cannot represent every transformed occupied or frontier coordinate.
    pub fn transformed(&self, transform: D6Transform) -> Result<Self, D6Error> {
        let mut transformed = Self {
            cells: vec![None; GRID_SIZE],
            placed_indices: Vec::with_capacity(self.placed_indices.len()),
            nature_tokens: self.nature_tokens,
        };

        for &index in &self.placed_indices {
            let source_coord =
                HexCoord::from_index(usize::from(index)).expect("stored board index is valid");
            let target_coord = transform.transform_coord(source_coord)?;
            let target_index =
                target_coord
                    .to_index()
                    .ok_or(D6Error::BoardCoordinateOutOfBounds {
                        transform,
                        source_coord,
                        transformed: target_coord,
                    })?;
            if transformed.cells[target_index].is_some() {
                return Err(D6Error::BoardCoordinateCollision(target_coord));
            }
            let placed = self.cells[usize::from(index)].expect("stored board index is occupied");
            transformed.cells[target_index] = Some(PlacedTile {
                tile: placed.tile,
                rotation: transform.transform_tile_rotation(placed.tile, placed.rotation),
                wildlife: placed.wildlife,
            });
            transformed.placed_indices.push(target_index as u16);
        }

        let mut expected_frontier = self
            .frontier()
            .into_iter()
            .map(|coord| {
                let transformed_coord = transform.transform_coord(coord)?;
                transformed_coord.to_index().map_or_else(
                    || {
                        Err(D6Error::BoardCoordinateOutOfBounds {
                            transform,
                            source_coord: coord,
                            transformed: transformed_coord,
                        })
                    },
                    |_| Ok(transformed_coord),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        expected_frontier.sort_unstable();
        expected_frontier.dedup();
        if transformed.frontier() != expected_frontier {
            return Err(D6Error::FrontierMismatch(transform));
        }
        transformed.validate().map_err(D6Error::Invariant)?;
        Ok(transformed)
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.cells.len() != GRID_SIZE {
            return Err("board cell array has the wrong length");
        }
        if self.placed_indices.len() > MAX_BOARD_TILES {
            return Err("board contains more than 23 tiles");
        }
        let occupied = self.cells.iter().filter(|cell| cell.is_some()).count();
        if occupied != self.placed_indices.len() {
            return Err("occupied cells and placed index list disagree");
        }
        for &index in &self.placed_indices {
            let Some(placed) = self.cells[usize::from(index)] else {
                return Err("placed index points to an empty cell");
            };
            if placed.tile.terrain_b.is_none() && placed.rotation != Rotation::ZERO {
                return Err("single-terrain tile has a noncanonical rotation");
            }
        }
        Ok(())
    }

    fn insert_starter(
        &mut self,
        coord: HexCoord,
        tile: Tile,
        rotation: Rotation,
    ) -> Result<(), BoardError> {
        let index = coord.to_index().ok_or(BoardError::OutOfBounds(coord))?;
        if self.cells[index].is_some() {
            return Err(BoardError::Occupied(coord));
        }
        if self.placed_indices.len() == MAX_BOARD_TILES {
            return Err(BoardError::TileLimitReached);
        }
        self.cells[index] = Some(PlacedTile {
            tile,
            rotation: tile.canonical_rotation(rotation),
            wildlife: None,
        });
        self.placed_indices.push(index as u16);
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn insert_scoring_fixture(
        &mut self,
        coord: HexCoord,
        tile: Tile,
        rotation: Rotation,
        wildlife: Option<Wildlife>,
    ) -> Result<(), BoardError> {
        let index = coord.to_index().ok_or(BoardError::OutOfBounds(coord))?;
        if self.cells[index].is_some() {
            return Err(BoardError::Occupied(coord));
        }
        if self.placed_indices.len() == MAX_BOARD_TILES {
            return Err(BoardError::TileLimitReached);
        }
        self.cells[index] = Some(PlacedTile {
            tile,
            rotation: tile.canonical_rotation(rotation),
            wildlife,
        });
        self.placed_indices.push(index as u16);
        Ok(())
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use crate::{STANDARD_TILES, STARTER_CLUSTERS};

    use super::*;

    #[test]
    fn tile_placement_requires_adjacency_and_is_undoable() {
        let mut board = Board::from_starter(&STARTER_CLUSTERS[0]);
        let before = board.clone();

        assert_eq!(
            board.place_tile(HexCoord::new(10, 10), STANDARD_TILES[0], Rotation::ZERO),
            Err(BoardError::Detached(HexCoord::new(10, 10)))
        );

        let delta = board
            .place_tile(HexCoord::new(-1, 0), STANDARD_TILES[0], Rotation::ZERO)
            .unwrap();
        board.undo(delta).unwrap();
        assert_eq!(board, before);
    }

    #[test]
    fn canonical_hash_covers_empty_markers_and_unindexed_storage_cells() {
        let empty = Board::empty();
        let mut unindexed = empty.clone();
        unindexed.cells[0] = Some(PlacedTile {
            tile: STANDARD_TILES[0],
            rotation: Rotation::ZERO,
            wildlife: None,
        });

        assert_eq!(unindexed.placed_indices, empty.placed_indices);
        assert_ne!(unindexed.canonical_hash(), empty.canonical_hash());
    }

    #[test]
    fn keystone_wildlife_awards_and_undoes_nature_token() {
        let mut board = Board::empty();
        let tile_delta = board
            .place_tile(HexCoord::ORIGIN, STANDARD_TILES[2], Rotation::ZERO)
            .unwrap();
        let wildlife_delta = board
            .place_wildlife(HexCoord::ORIGIN, Wildlife::Bear)
            .unwrap();
        assert_eq!(board.nature_tokens(), 1);

        board.undo(wildlife_delta).unwrap();
        assert_eq!(board.nature_tokens(), 0);
        board.undo(tile_delta).unwrap();
        assert_eq!(board.tile_count(), 0);
    }

    #[test]
    fn habitat_connectivity_uses_matching_oriented_edges() {
        let mut board = Board::empty();
        let bear = 1 << Wildlife::Bear as u8;
        let first = Tile::dual(200, Terrain::Forest, Terrain::River, bear);
        let second = Tile::dual(201, Terrain::Forest, Terrain::River, bear);
        board
            .place_tile(HexCoord::ORIGIN, first, Rotation::ZERO)
            .unwrap();
        board
            .place_tile(HexCoord::new(1, 0), second, Rotation::THREE)
            .unwrap();

        assert_eq!(board.largest_habitat(Terrain::Forest), 2);
        assert_eq!(board.largest_habitat(Terrain::River), 1);
    }

    #[test]
    fn frontier_has_no_duplicates_or_occupied_cells() {
        let board = Board::from_starter(&STARTER_CLUSTERS[0]);
        let frontier = board.frontier();
        let mut deduplicated = frontier.clone();
        deduplicated.sort_unstable();
        deduplicated.dedup();

        assert_eq!(frontier, deduplicated);
        assert!(frontier.iter().all(|coord| board.tile_at(*coord).is_none()));
    }

    #[test]
    fn official_environment_size_is_enforced() {
        let mut board = Board::empty();
        for q in 0..MAX_BOARD_TILES {
            board
                .place_tile(
                    HexCoord::new(q as i8, 0),
                    STANDARD_TILES[q % STANDARD_TILES.len()],
                    Rotation::ZERO,
                )
                .unwrap();
        }

        assert_eq!(
            board.place_tile(
                HexCoord::new(MAX_BOARD_TILES as i8, 0),
                STANDARD_TILES[0],
                Rotation::ZERO,
            ),
            Err(BoardError::TileLimitReached)
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn arbitrary_tile_and_wildlife_changes_undo_exactly(
            rotation in 0u8..6,
            wildlife_index in 0usize..Wildlife::ALL.len(),
        ) {
            let original = Board::empty();
            let mut board = original.clone();
            let tile = Tile::dual(250, Terrain::Forest, Terrain::River, 0b1_1111);
            let tile_delta = board
                .place_tile(
                    HexCoord::ORIGIN,
                    tile,
                    Rotation::new(rotation).expect("generated rotation is valid"),
                )
                .unwrap();
            let wildlife_delta = board
                .place_wildlife(HexCoord::ORIGIN, Wildlife::ALL[wildlife_index])
                .unwrap();

            board.undo(wildlife_delta).unwrap();
            board.undo(tile_delta).unwrap();

            prop_assert_eq!(board, original);
        }
    }
}
